# OpenICL/openicl/icl_retriever/icl_dpp_sgt_retriever.py
"""
DPP + Real SGT Prior Retriever

This retriever implements a subset-level SGT (Smoothed Good-Toulmin) prior
that is combined with DPP during greedy MAP selection.

Objective:
    argmax_{|S|=ice_num}  log det(K_S)  +  lambda_sgt * Khat_SGT(S)

where:
    - log det(K_S) is the DPP diversity term (from fast_map_dpp greedy)
    - Khat_SGT(S) = seen(S) + unseen_hat(S) computed from frequency spectrum
    - unseen_hat is estimated via smoothed_good_toulmin_sgt() from the subset's
      cluster frequency spectrum (how many clusters appear exactly s times)

Key differences from static rarity weighting:
    - SGT prior is subset-dependent (cannot be precomputed per point)
    - Khat_SGT depends on cluster multiplicities within the selected subset
    - Matches selection.py semantics: ignore_label excluded, unseen_hat clamped to 0

The greedy selection at each step chooses item i that maximizes:
    score_i = log(max(di2s[i], eps)) + sgt_lambda * (khat_after_adding_i - khat_current)

This combines the DPP marginal gain (log det improvement) with the SGT marginal gain
(cluster diversity improvement estimated via frequency spectrum).

Alignment with selection.py:
    This implementation matches selection.py's _khat_sgt_for_indices() semantics exactly:
    1. ignore_label is excluded from the type universe (cluster IDs matching ignore_label
       are skipped in add() and simulate_add())
    2. unseen_hat from smoothed_good_toulmin_sgt() is clamped to 0 if negative/NaN/non-finite
    3. Khat_SGT = seen + unseen_hat, where seen is the number of distinct clusters
       (excluding ignore_label) with multiplicity >= 1
    4. If seen == 0, khat() returns 0.0
    5. Frequency spectrum freq_by_s maps multiplicity s -> number of clusters with that multiplicity
    6. Uses same SGT call arguments: t_list=[sgt_t], bin_size, smooth_count, mute=True, offset
"""

from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.logging import get_logger
from typing import Optional, List
import tqdm
import numpy as np
import math
from accelerate import Accelerator
from collections import Counter, defaultdict

# Import SGT function - try multiple paths for robustness
try:
    from utils_sgt import smoothed_good_toulmin_sgt
except ImportError:
    try:
        from icl_select.utils_sgt import smoothed_good_toulmin_sgt
    except ImportError:
        # Fallback: try relative import if we're in OpenICL package
        import sys
        from pathlib import Path
        # Try to find utils_sgt in parent directories
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from icl_select.utils_sgt import smoothed_good_toulmin_sgt

logger = get_logger(__name__)


class SGTSpectrumState:
    """
    Incremental state for tracking cluster frequency spectrum and computing Khat_SGT.
    
    Matches selection.py's SGTSpectrumState semantics exactly:
    - type_counts: Counter mapping cluster_id -> multiplicity in current subset
    - freq_by_s: defaultdict mapping multiplicity s -> number of clusters with that multiplicity
    - seen: number of distinct clusters (with multiplicity >= 1, excluding ignore_label)
    - khat(): returns seen + unseen_hat, where unseen_hat comes from SGT estimation
    """
    def __init__(self, t: int, bin_size: int, smooth_count: bool, offset: float, ignore_label: int = -1):
        self.t = t
        self.bin_size = bin_size
        self.smooth_count = smooth_count
        self.offset = offset
        self.ignore_label = ignore_label
        
        self.type_counts = Counter()          # cluster -> multiplicity in subset
        self.freq_by_s = defaultdict(int)     # s -> #types with multiplicity s
        self.seen = 0                         # number of distinct clusters (excluding ignore_label)
    
    def add(self, c: int) -> None:
        """Add one occurrence of cluster c to the subset."""
        if c == self.ignore_label:
            return
        
        m = self.type_counts[c]
        if m > 0:
            # Cluster already seen: decrement old multiplicity count
            self.freq_by_s[m] -= 1
            if self.freq_by_s[m] == 0:
                del self.freq_by_s[m]
        else:
            # New cluster: increment seen count
            self.seen += 1
        
        # Update counts
        self.type_counts[c] += 1
        self.freq_by_s[m + 1] += 1
    
    def khat(self) -> float:
        """
        Compute Khat_SGT = seen + unseen_hat.
        
        Matches selection.py semantics:
        - If seen == 0, return 0.0
        - Call smoothed_good_toulmin_sgt on frequency spectrum
        - Clamp unseen_hat to 0 if negative/NaN/non-finite
        """
        if self.seen == 0:
            return 0.0
        
        unseen_list, _ = smoothed_good_toulmin_sgt(
            dict(self.freq_by_s),
            t_list=[self.t],
            bin_size=self.bin_size,
            smooth_count=self.smooth_count,
            mute=True,
            offset=self.offset,
        )
        unseen_hat = float(unseen_list[0])
        if not np.isfinite(unseen_hat) or unseen_hat < 0:
            unseen_hat = 0.0
        
        return float(self.seen + unseen_hat)
    
    def simulate_add(self, c: int) -> float:
        """
        Simulate adding cluster c and return the new khat value.
        Does not modify state (temporary simulation).
        """
        if c == self.ignore_label:
            return self.khat()
        
        # Temporarily update state
        m = self.type_counts.get(c, 0)
        old_seen = self.seen
        
        if m == 0:
            # Would add a new cluster
            new_seen = old_seen + 1
        else:
            new_seen = old_seen
        
        # Build temporary frequency spectrum
        temp_freq_by_s = defaultdict(int, self.freq_by_s)
        if m > 0:
            temp_freq_by_s[m] -= 1
            if temp_freq_by_s[m] == 0:
                del temp_freq_by_s[m]
        temp_freq_by_s[m + 1] += 1
        
        # Compute khat with temporary spectrum
        if new_seen == 0:
            return 0.0
        
        unseen_list, _ = smoothed_good_toulmin_sgt(
            dict(temp_freq_by_s),
            t_list=[self.t],
            bin_size=self.bin_size,
            smooth_count=self.smooth_count,
            mute=True,
            offset=self.offset,
        )
        unseen_hat = float(unseen_list[0])
        if not np.isfinite(unseen_hat) or unseen_hat < 0:
            unseen_hat = 0.0
        
        return float(new_seen + unseen_hat)


class DPPSGTRetriever(TopkRetriever):
    """
    DPP Retriever with Real SGT Prior (subset-dependent cluster diversity).
    
    This retriever combines DPP diversity (log det) with SGT-based cluster diversity
    during greedy MAP selection. The SGT prior is computed from the frequency spectrum
    of cluster IDs within the currently selected subset, making it subset-dependent.
    
    Args:
        cluster_ids: np.ndarray shape (N_train,), aligned with index dataset order.
                        For any retrieved candidate id `i` (FAISS id), we use cluster_ids[i].
        sgt_lambda: Weight for SGT term in combined objective (default 0.5).
        sgt_t: SGT parameter t (default 5).
        sgt_bin_size: SGT bin_size parameter (default 20).
        sgt_smooth_count: Whether to smooth frequency spectrum before SGT (default False).
        sgt_offset: SGT offset parameter (default 1.0).
        ignore_label: Cluster label to ignore in SGT computation (default -1).
        sgt_eval_top_m_candidates: If not None, only evaluate SGT for top-m candidates
                                   by di2s to reduce computation (default None).
    """
    model = None

    def __init__(
        self,
        dataset_reader: DatasetReader,
        cluster_ids: np.ndarray,
        sgt_lambda: float = 0.5,
        sgt_t: int = 5,
        sgt_bin_size: int = 20,
        sgt_smooth_count: bool = False,
        sgt_offset: float = 1.0,
        ignore_label: int = -1,
        sgt_eval_top_m_candidates: Optional[int] = None,
        ice_separator: Optional[str] = "\n",
        ice_eos_token: Optional[str] = "\n",
        prompt_eos_token: Optional[str] = "",
        sentence_transformers_model_name: Optional[str] = "all-mpnet-base-v2",
        ice_num: Optional[int] = 1,
        candidate_num: Optional[int] = 1,
        index_split: Optional[str] = "train",
        test_split: Optional[str] = "test",
        tokenizer_name: Optional[str] = "gpt2-xl",
        batch_size: Optional[int] = 1,
        accelerator: Optional[Accelerator] = None,
        seed: Optional[int] = 1,
        scale_factor: Optional[float] = 0.1,
    ) -> None:
        super().__init__(
            dataset_reader,
            ice_separator,
            ice_eos_token,
            prompt_eos_token,
            sentence_transformers_model_name,
            ice_num,
            index_split,
            test_split,
            tokenizer_name,
            batch_size,
            accelerator,
        )
        self.candidate_num = candidate_num
        self.seed = seed
        self.scale_factor = scale_factor

        if cluster_ids is None:
            raise ValueError("DPPSGTRetriever requires cluster_ids.")
        self.cluster_ids = np.asarray(cluster_ids).astype(int)
        
        # Validate cluster_ids length matches index dataset size
        index_size = len(self.index_ds)
        if len(self.cluster_ids) != index_size:
            raise ValueError(
                f"cluster_ids length ({len(self.cluster_ids)}) must match "
                f"index dataset size ({index_size})"
            )

        # SGT parameters
        self.sgt_lambda = float(sgt_lambda)
        self.sgt_t = int(sgt_t)
        self.sgt_bin_size = int(sgt_bin_size)
        self.sgt_smooth_count = bool(sgt_smooth_count)
        self.sgt_offset = float(sgt_offset)
        self.ignore_label = int(ignore_label)
        self.sgt_eval_top_m_candidates = sgt_eval_top_m_candidates

    def retrieve(self):
        return self.dpp_search()

    def dpp_search(self):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry["metadata"]["id"]

            # Stage 1: TopK candidates
            embed = np.expand_dims(entry["embed"], axis=0)
            near_ids = np.array(self.index.search(embed, self.candidate_num)[1][0].tolist())

            # Stage 2: DPP kernel (NO static prior - kernel is quality * similarity * quality)
            near_reps, rel_scores, kernel_matrix = self.get_kernel(embed, near_ids.tolist())

            # Stage 3: Greedy MAP with SGT prior
            samples_ids = self.fast_map_dpp_with_sgt(
                kernel_matrix=kernel_matrix,
                max_length=self.ice_num,
                candidate_faiss_ids=near_ids.tolist(),
            )

            # Order chosen items by original relevance (not SGT-modified)
            samples_ids = np.array(samples_ids)
            samples_scores = np.array([rel_scores[i] for i in samples_ids])
            samples_ids = samples_ids[(-samples_scores).argsort()].tolist()

            rtr_sub_list = [int(near_ids[i]) for i in samples_ids]
            rtr_idx_list[idx] = rtr_sub_list

        return rtr_idx_list

    def get_kernel(self, embed: np.ndarray, candidates: List[int]):
        """
        Compute DPP kernel matrix WITHOUT static prior reweighting.
        
        Kernel = quality * similarity * quality, where:
        - quality = exp(relevance / scale_factor)
        - similarity = cosine similarity matrix
        """
        # Reconstruct candidate vectors from FAISS
        near_reps = np.stack([self.index.index.reconstruct(i) for i in candidates], axis=0)

        # Normalize first (cosine sim)
        embed = embed / (np.linalg.norm(embed) + 1e-12)
        near_reps = near_reps / (np.linalg.norm(near_reps, keepdims=True, axis=1) + 1e-12)

        # Relevance scores in [0,1]
        rel_scores = np.matmul(embed, near_reps.T)[0]
        rel_scores = (rel_scores + 1.0) / 2.0

        # Stabilize exp
        rel_scores -= rel_scores.max()
        rel_scores = np.exp(rel_scores / (2.0 * self.scale_factor + 1e-12))

        # NO static prior multiplication here - kernel is pure DPP

        # Similarity matrix in [0,1]
        sim_matrix = np.matmul(near_reps, near_reps.T)
        sim_matrix = (sim_matrix + 1.0) / 2.0

        # DPP kernel (quality * similarity * quality)
        kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
        return near_reps, rel_scores, kernel_matrix

    def fast_map_dpp_with_sgt(
        self,
        kernel_matrix: np.ndarray,
        max_length: int,
        candidate_faiss_ids: List[int],
    ) -> List[int]:
        """
        Greedy MAP inference combining DPP log det gain with SGT cluster diversity gain.
        
        At each step, selects item i that maximizes:
            score_i = log(max(di2s[i], eps)) + sgt_lambda * (khat_after_adding_i - khat_current)
        
        Args:
            kernel_matrix: DPP kernel matrix (candidate_num x candidate_num)
            max_length: Number of items to select (ice_num)
            candidate_faiss_ids: List of FAISS IDs for candidates (aligned with kernel_matrix rows)
        
        Returns:
            List of selected candidate indices (local indices into kernel_matrix)
        """
        item_size = kernel_matrix.shape[0]
        if max_length > item_size:
            logger.warning(
                f"Requested {max_length} items but only {item_size} candidates available. "
                f"Returning {item_size} items."
            )
            max_length = item_size
        
        # Initialize DPP state (same as fast_map_dpp)
        cis = np.zeros((max_length, item_size))
        di2s = np.copy(np.diag(kernel_matrix))
        
        # Initialize SGT state
        sgt_state = SGTSpectrumState(
            t=self.sgt_t,
            bin_size=self.sgt_bin_size,
            smooth_count=self.sgt_smooth_count,
            offset=self.sgt_offset,
            ignore_label=self.ignore_label,
        )
        
        selected_items = []
        eps = 1e-12
        
        # First item: argmax diag(K) (same as standard DPP)
        selected_item = int(np.argmax(di2s))
        selected_items.append(selected_item)
        
        # Add first item's cluster to SGT state
        first_faiss_id = candidate_faiss_ids[selected_item]
        first_cluster_id = int(self.cluster_ids[first_faiss_id])
        sgt_state.add(first_cluster_id)
        
        current_khat = sgt_state.khat()
        logger.debug(f"Initial khat: {current_khat:.4f}")
        
        # Greedy selection loop
        while len(selected_items) < max_length:
            k = len(selected_items) - 1
            selected_item_prev = selected_items[-1]
            
            # Update DPP state (orthogonalization step)
            ci_optimal = cis[:k, selected_item_prev]
            di_optimal = math.sqrt(max(di2s[selected_item_prev], eps))
            elements = kernel_matrix[selected_item_prev, :]
            eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
            cis[k, :] = eis
            di2s -= np.square(eis)
            
            # Find remaining candidates (not yet selected)
            remaining = [i for i in range(item_size) if i not in selected_items]
            if not remaining:
                break
            
            # Optionally limit SGT evaluation to top candidates by di2s
            if self.sgt_eval_top_m_candidates is not None and len(remaining) > self.sgt_eval_top_m_candidates:
                # Sort remaining by di2s (descending)
                remaining_sorted = sorted(remaining, key=lambda i: di2s[i], reverse=True)
                candidates_to_eval = remaining_sorted[:self.sgt_eval_top_m_candidates]
            else:
                candidates_to_eval = remaining
            
            # Evaluate each candidate: DPP gain + SGT gain
            best_item = None
            best_score = -np.inf
            
            for candidate_idx in candidates_to_eval:
                # DPP marginal gain: log(di2s[i])
                dpp_gain = math.log(max(di2s[candidate_idx], eps))
                
                # SGT marginal gain: khat_after_adding - khat_current
                candidate_faiss_id = candidate_faiss_ids[candidate_idx]
                candidate_cluster_id = int(self.cluster_ids[candidate_faiss_id])
                khat_after = sgt_state.simulate_add(candidate_cluster_id)
                sgt_gain = khat_after - current_khat
                
                # Combined score
                score = dpp_gain + self.sgt_lambda * sgt_gain
                
                if score > best_score:
                    best_score = score
                    best_item = candidate_idx
            
            # If we limited candidates, check if best is in top-m, else fall back to argmax di2s
            if best_item is None:
                # Fallback: select by DPP only (argmax di2s from remaining items)
                # Only consider remaining items to avoid selecting already-selected items
                remaining_di2s = np.array([di2s[i] for i in remaining])
                remaining_indices = np.array(remaining)
                best_remaining_idx = int(np.argmax(remaining_di2s))
                best_item = int(remaining_indices[best_remaining_idx])
                logger.debug("Fallback to DPP-only selection (best_item was None)")
            
            selected_items.append(best_item)
            
            # Update SGT state permanently
            best_faiss_id = candidate_faiss_ids[best_item]
            best_cluster_id = int(self.cluster_ids[best_faiss_id])
            sgt_state.add(best_cluster_id)
            current_khat = sgt_state.khat()
            
            logger.debug(
                f"Step {len(selected_items)}: selected item {best_item} "
                f"(cluster {best_cluster_id}), khat={current_khat:.4f}"
            )
        
        # Validation: ensure unique indices
        assert len(selected_items) == len(set(selected_items)), "Selected items must be unique"
        assert len(selected_items) <= max_length, f"Selected {len(selected_items)} items, expected <= {max_length}"
        
        # Log final statistics
        if len(selected_items) > 0:
            selected_clusters = [int(self.cluster_ids[candidate_faiss_ids[i]]) for i in selected_items]
            cluster_counts = Counter(selected_clusters)
            mean_multiplicity = np.mean(list(cluster_counts.values()))
            var_multiplicity = np.var(list(cluster_counts.values()))
            logger.debug(
                f"Final selection: {len(selected_items)} items, "
                f"khat={current_khat:.4f}, "
                f"mean cluster multiplicity={mean_multiplicity:.2f}, "
                f"var={var_multiplicity:.2f}"
            )
        
        return selected_items

