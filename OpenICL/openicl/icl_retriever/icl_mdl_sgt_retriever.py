# OpenICL/openicl/icl_retriever/icl_mdl_sgt_retriever.py

from typing import List, Optional, Tuple, Dict
import numpy as np
import tqdm
from collections import Counter, defaultdict

from openicl import DatasetReader, PromptTemplate
from openicl.utils.logging import get_logger
from openicl.icl_retriever.icl_mdl_retriever import MDLRetriever

# Import SGT function - try multiple paths for robustness (matches other SGT retrievers)
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
        try:
            from icl_select.utils_sgt import smoothed_good_toulmin_sgt
        except ImportError:
            # Last resort: define a minimal fallback (should not happen in normal usage)
            logger = get_logger(__name__)
            logger.warning(
                "Could not import smoothed_good_toulmin_sgt from utils_sgt. "
                "Using fallback implementation. This may not match canonical behavior."
            )
            # Minimal fallback - should not be used in production
            def smoothed_good_toulmin_sgt(counts, t_list, bin_size=20, smooth_count=False, 
                                         adaptive=False, mute=True, offset=1.0):
                """Fallback implementation - should not be used."""
                import scipy.stats
                from collections import defaultdict
                if isinstance(t_list, (int, float)):
                    t_list = [t_list]
                freq_counts = defaultdict(int, counts)
                unseen_estimates = []
                unseen_stds = []
                for t in t_list:
                    unseen_estimate = 0.0
                    unseen_variance = 0.0
                    for s in range(1, bin_size + 1):
                        freq_s = freq_counts.get(s, 0)  # Use .get() to avoid KeyError
                        if freq_s == 0:
                            continue
                        prob = 1.0 - scipy.stats.binom.cdf(s - 1, bin_size, offset / (t + offset))
                        delta = -((-t) ** s) * prob
                        unseen_estimate += delta * float(freq_s)
                        unseen_variance += (delta ** 2) * float(freq_s)
                    unseen_estimate = max(0.0, unseen_estimate)
                    unseen_std = np.sqrt(max(0.0, unseen_variance))
                    unseen_estimates.append(int(unseen_estimate))
                    unseen_stds.append(int(unseen_std))
                return unseen_estimates, unseen_stds

logger = get_logger(__name__)


def _khat_sgt_for_subset(
    cluster_ids: np.ndarray,
    subset_indices: List[int],
    t: float = 5.0,
    bin_size: int = 20,
    smooth_count: bool = False,
    offset: float = 1.0,
    ignore_label: int = -1,
) -> float:
    """
    Compute K̂_S = seen + unseen_hat using Smoothed Good-Toulmin.
    
    This matches the canonical implementation in selection.py::_khat_sgt_for_indices.
    Uses the canonical smoothed_good_toulmin_sgt function from utils_sgt.py.
    
    Args:
        cluster_ids: Full array of cluster IDs for all training examples
        subset_indices: Indices of examples in subset S
        t: SGT extrapolation parameter (can be int or float)
        bin_size: Binning parameter for stability
        smooth_count: Whether to apply power-law smoothing
        offset: Offset parameter (1 <= offset <= 2)
        ignore_label: Cluster ID to ignore (e.g., -1 for DBSCAN noise)
    
    Returns:
        K̂_S = number of unique clusters seen + estimated unseen clusters
    """
    if len(subset_indices) == 0:
        return 0.0
    
    subset_cluster_ids = cluster_ids[subset_indices]
    # Filter out ignored labels
    subset_cluster_ids = subset_cluster_ids[subset_cluster_ids != ignore_label]
    
    if len(subset_cluster_ids) == 0:
        return 0.0
    
    # Count frequency of each cluster in subset
    cluster_freq = Counter(int(c) for c in subset_cluster_ids)
    seen = len(cluster_freq)
    
    # Build frequency-of-frequencies: f_s = #clusters appearing exactly s times
    freq_by_s = defaultdict(int)
    for count in cluster_freq.values():
        if count > 0:
            freq_by_s[count] += 1
    
    if not freq_by_s:
        return float(seen)
    
    # Call canonical SGT estimator (returns lists: unseen_estimates, unseen_stds)
    # Convert t to int if it's a float close to an integer (for consistency with canonical)
    t_for_sgt = int(round(t)) if abs(t - round(t)) < 1e-6 else t
    unseen_list, _ = smoothed_good_toulmin_sgt(
        freq_by_s,
        t_list=[t_for_sgt],  # Canonical expects t_list (list of t values)
        bin_size=bin_size,
        smooth_count=smooth_count,
        mute=True,  # Suppress print statements
        offset=offset,
    )
    # Extract first (and only) estimate from list
    unseen_hat = float(unseen_list[0]) if isinstance(unseen_list, (list, tuple)) and len(unseen_list) > 0 else 0.0
    
    # Clamp negative/NaN to 0 (stability)
    if not np.isfinite(unseen_hat) or unseen_hat < 0:
        unseen_hat = 0.0
    
    return float(seen + unseen_hat)


class MDLSGTRetriever(MDLRetriever):
    """
    MDL Retriever + Smoothed Good-Toulmin (SGT) subset prior.

    Candidate subset S is scored by:
        score(S) = MDL(S) + sgt_lambda * prior(S)

    where prior(S) = K̂_S (expected total clusters) computed via SGT on cluster IDs in S.
    
    The SGT prior encourages diversity by estimating how many unique clusters
    (knowledge types) are represented in the subset, including an estimate of
    unseen clusters based on the frequency spectrum.
    
    Notes:
    - MDL is negative entropy (higher is better), so we maximize score(S).
    - Prior is K̂_S = seen + unseen_hat, which increases with diversity.
    - sgt_lambda controls the trade-off: higher values favor more diverse subsets.
    - Default sgt_lambda=0.1 means prior contributes ~10% of score scale (adjust based on MDL range).
    """

    def __init__(
        self,
        dataset_reader: DatasetReader,
        cluster_ids: np.ndarray,
        # ---- original MDL args ----
        ice_separator: Optional[str] = "\n",
        ice_eos_token: Optional[str] = "\n",
        prompt_eos_token: Optional[str] = "",
        sentence_transformers_model_name: Optional[str] = "all-mpnet-base-v2",
        ice_num: Optional[int] = 1,
        candidate_num: Optional[int] = 1,
        index_split: Optional[str] = "train",
        test_split: Optional[str] = "test",
        tokenizer_name: Optional[str] = "gpt2-xl",
        ce_model_name: Optional[str] = "gpt2-xl",
        batch_size: Optional[int] = 1,
        select_time: Optional[int] = 5,
        accelerator=None,
        ice_template: Optional[PromptTemplate] = None,
        prompt_template: Optional[PromptTemplate] = None,
        labels: Optional[List] = None,
        seed: Optional[int] = 1,
        # ---- SGT knobs (upgraded to real SGT) ----
        sgt_lambda: float = 0.1,
        sgt_t: float = 5.0,
        sgt_bin_size: int = 20,
        sgt_smooth_count: bool = False,
        sgt_offset: float = 1.0,
        sgt_ignore_label: int = -1,
        prior_eps: float = 0.0,
        prior_clip: Optional[Tuple[float, float]] = None,
        prior_normalize: bool = False,  # Changed default: K̂_S is already interpretable
        prior_log_stats: bool = False,  # Debug logging
    ) -> None:
        super().__init__(
            dataset_reader=dataset_reader,
            ice_separator=ice_separator,
            ice_eos_token=ice_eos_token,
            prompt_eos_token=prompt_eos_token,
            sentence_transformers_model_name=sentence_transformers_model_name,
            ice_num=ice_num,
            candidate_num=candidate_num,
            index_split=index_split,
            test_split=test_split,
            tokenizer_name=tokenizer_name,
            ce_model_name=ce_model_name,
            batch_size=batch_size,
            select_time=select_time,
            accelerator=accelerator,
            ice_template=ice_template,
            prompt_template=prompt_template,
            labels=labels,
            seed=seed,
        )

        self.cluster_ids = np.asarray(cluster_ids).astype(np.int64)
        if len(self.cluster_ids) != len(self.index_ds):
            raise ValueError(
                f"cluster_ids length ({len(self.cluster_ids)}) must match index_ds length ({len(self.index_ds)})."
            )

        # SGT parameters (matching canonical implementation)
        self.sgt_lambda = float(sgt_lambda)
        self.sgt_t = float(sgt_t)
        self.sgt_bin_size = int(sgt_bin_size)
        self.sgt_smooth_count = bool(sgt_smooth_count)
        self.sgt_offset = float(sgt_offset)
        self.sgt_ignore_label = int(sgt_ignore_label)
        
        # Prior post-processing
        self.prior_eps = float(prior_eps)
        self.prior_clip = prior_clip
        self.prior_normalize = bool(prior_normalize)
        self.prior_log_stats = bool(prior_log_stats)
        
        # Validate offset
        if not (1.0 <= self.sgt_offset <= 2.0):
            raise ValueError(f"sgt_offset must be in [1, 2], got {self.sgt_offset}")

    def _subset_prior(self, subset_indices: List[int]) -> float:
        """
        Compute SGT prior scalar for a subset S (list of train indices).
        
        Returns K̂_S = seen + unseen_hat, where unseen_hat is estimated via
        Smoothed Good-Toulmin from the frequency spectrum of cluster IDs in S.
        
        This matches the canonical implementation in selection.py::_khat_sgt_for_indices.
        
        Args:
            subset_indices: List of training example indices in subset S
        
        Returns:
            Prior scalar (K̂_S, optionally normalized/clipped)
        """
        if len(subset_indices) == 0:
            return 0.0
        
        # Compute K̂_S using canonical SGT
        khat = _khat_sgt_for_subset(
            cluster_ids=self.cluster_ids,
            subset_indices=subset_indices,
            t=self.sgt_t,
            bin_size=self.sgt_bin_size,
            smooth_count=self.sgt_smooth_count,
            offset=self.sgt_offset,
            ignore_label=self.sgt_ignore_label,
        )
        
        # Optional debug logging
        if self.prior_log_stats:
            subset_cluster_ids = self.cluster_ids[subset_indices]
            valid_clusters = subset_cluster_ids[subset_cluster_ids != self.sgt_ignore_label]
            if len(valid_clusters) > 0:
                cluster_freq = Counter(int(c) for c in valid_clusters)
                seen = len(cluster_freq)
                freq_by_s = defaultdict(int)
                for cnt in cluster_freq.values():
                    if cnt > 0:
                        freq_by_s[cnt] += 1
                singleton_count = freq_by_s.get(1, 0)
                logger.debug(
                    f"SGT prior stats: |S|={len(subset_indices)}, seen={seen}, "
                    f"singletons={singleton_count}, K̂_S={khat:.2f}"
                )
        
        prior = float(khat)
        
        # Optional normalization (by default, K̂_S is already interpretable)
        if self.prior_normalize and len(subset_indices) > 0:
            prior = prior / float(len(subset_indices))
        
        # Add epsilon offset
        prior = prior + self.prior_eps
        
        # Optional clipping (useful if prior can spike with many singletons)
        if self.prior_clip is not None:
            lo, hi = self.prior_clip
            prior = float(np.clip(prior, lo, hi))
        
        return float(prior)

    # Only override the selection loop; everything else (CE model, etc.) stays MDL.
    def topk_search(self):
        np.random.seed(self.seed)
        res_list = self.forward(self.dataloader)
        rtr_idx_list = [[] for _ in range(len(res_list))]

        logger.info("Retrieving data for test set (MDL + SGT prior)...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry["metadata"]["id"]

            embed = np.expand_dims(entry["embed"], axis=0)
            near_ids = self.index.search(embed, min(self.candidate_num, len(self.index_ds)))[1][0].tolist()

            best_subset = None
            best_score = -1e18

            for j in range(self.select_time):
                if j == 0:
                    subset = near_ids[: self.ice_num]
                else:
                    subset = np.random.choice(near_ids, self.ice_num, replace=False)
                    subset = [int(i) for i in subset]

                # ----- MDL score (same as MDLRetriever) -----
                ice = self.generate_ice(subset, ice_template=self.ice_template)
                mask_length = len(self.tokenizer(ice + self.ice_eos_token, verbose=False)["input_ids"])

                if self.labels is None:
                    labels = self.get_labels(self.ice_template, self.prompt_template)
                else:
                    labels = self.labels

                prompt_list = []
                for label in labels:
                    prompt = self.generate_label_prompt(idx, ice, label, self.ice_template, self.prompt_template)
                    prompt_list.append(prompt)

                loss_list = self.cal_ce(prompt_list, mask_length=mask_length)
                probs = np.exp(-np.array(loss_list))
                normalized_probs = probs / probs.sum(0, keepdims=True)
                mdl = float(-self._entropy(normalized_probs))

                # ----- SGT subset prior -----
                prior = self._subset_prior(subset)

                # ----- combine -----
                score = mdl + self.sgt_lambda * prior

                if score > best_score:
                    best_score = score
                    best_subset = subset

            rtr_idx_list[idx] = [int(i) for i in best_subset]

        return rtr_idx_list

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        """
        Small local entropy helper to avoid importing entropy() just for one call.
        p: shape [n_labels] or [n_labels, ...]
        """
        p = np.clip(p, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum(axis=0))
