# openicl/icl_retriever/icl_votek_sgt_retriever.py
"""
VoteK Retriever with Real SGT Prior

This retriever combines VoteK base votes with an SGT (Smoothed Good-Toulmin) derived
cluster-level prior. The prior is computed from the cluster-size frequency spectrum
over the entire pool using the same SGT machinery used in selection.py.

Objective:
    score(i) = v(i) + sgt_lambda * log(w_{cluster(i)})

where:
    - v(i) is the VoteK base vote count (how many points select i as neighbor)
    - w_{cluster(i)} is an SGT-derived weight for cluster(i) (normalized to mean 1.0)
    - sgt_lambda is the weight for the SGT prior term (default 0.1, consistent with dpp_sgt/mdl_sgt)

SGT Prior Computation (Option A):
    1. Compute cluster sizes n_c for each cluster c (ignore ignore_label)
    2. Build frequency-of-frequencies spectrum: f_s = #clusters with size exactly s
    3. Fit a parametric spectrum model (powerlaw/poisson/negbin) to smooth f_s → fhat_s
    4. Convert to probability over sizes: P(size=s) ∝ fhat_s
    5. Set w_c = 1 / (fhat_{n_c} + eps)  (rare size groups get higher weight)
    6. Normalize weights to have mean 1.0 for stability

This is a "real SGT prior" because fhat_s is derived from the Good-Toulmin/SGT
smoothing machinery, not an ad-hoc count^{-alpha} power law.

The retriever selects a corpus-level demo set (same demos for every test point),
like standard VoteK, and is deterministic given a seed.
"""

import os
import json
import random
import math
from typing import Optional, List, Dict, Any
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from accelerate import Accelerator
from scipy.optimize import curve_fit, minimize
from scipy.stats import nbinom, poisson

from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.logging import get_logger

logger = get_logger(__name__)


# Minimal smoothing utilities (copied from utils_sgt.py to avoid path issues)
def smooth_counts_powerlaw(counts, k=None):
    """Fit power-law model to frequency spectrum."""
    if k is None:
        k = len(counts)
    if len(counts) == 0:
        return np.array([])
    r_nonzero = np.arange(1, len(counts) + 1)
    
    # Power-law function: C * (beta+r)^(-alpha)
    def power_law(r, C, alpha, beta):
        return C * (beta + r) ** (-alpha)
    
    try:
        params, _ = curve_fit(power_law, r_nonzero[:k], counts[:k], maxfev=1000)
        C_fit, alpha_fit, beta_fit = params
        fitted_data = power_law(r_nonzero, C_fit, alpha_fit, beta_fit)
        # Ensure non-negative
        fitted_data = np.maximum(fitted_data, 0.0)
    except Exception as e:
        logger.warning(f"Power-law fitting failed: {e}. Using original counts.")
        fitted_data = np.array(counts)
    
    return fitted_data


def smooth_counts_poisson(counts, k=None):
    """Fit Poisson model to frequency spectrum."""
    counts = np.asarray(counts)
    if len(counts) == 0:
        return np.array([])
    if k is None:
        k = len(counts)
    s_vals = np.arange(1, len(counts) + 1)
    N = np.sum(counts[:k])  # total number of distinct types
    T = np.sum(s_vals[:k] * counts[:k])  # total number of tokens
    
    if N == 0:
        return np.zeros_like(counts)
    
    # MLE for λ is average frequency per type
    lambda_mle = T / N
    
    # Generate fitted spectrum using Poisson PMF
    fitted_counts = T * poisson.pmf(s_vals, mu=lambda_mle)
    fitted_counts = np.maximum(fitted_counts, 0.0)
    return fitted_counts


def smooth_counts_binomial(observed_counts):
    """Fit negative binomial model to frequency spectrum."""
    if len(observed_counts) == 0:
        return np.array([])
    
    # Rank positions
    ranks = np.arange(1, len(observed_counts) + 1)
    
    # Negative Binomial PMF model
    def negative_binomial_pmf(r, r_param, p_param, scale):
        return scale * nbinom.pmf(r - 1, r_param, p_param)
    
    # Loss function to minimize
    def loss(params):
        r_param, p_param, scale = params
        fitted = negative_binomial_pmf(ranks, r_param, p_param, scale)
        return np.sum((observed_counts - fitted) ** 2)
    
    # Initial guesses and bounds
    initial_params = [10, 0.5, 500]
    bounds = [(1e-5, None), (1e-5, 1 - 1e-5), (1e-5, None)]
    
    try:
        result = minimize(loss, initial_params, bounds=bounds, method='L-BFGS-B')
        r_fit, p_fit, scale_fit = result.x
        fitted_counts = negative_binomial_pmf(ranks, r_fit, p_fit, scale_fit)
        fitted_counts = np.maximum(fitted_counts, 0.0)
    except Exception as e:
        logger.warning(f"Negative binomial fitting failed: {e}. Using original counts.")
        fitted_counts = np.array(observed_counts)
    
    return fitted_counts


class VotekSGTRetriever(TopkRetriever):
    """
    VoteK Retriever with Real SGT Prior.
    
    Combines VoteK base votes with an SGT-derived cluster-level prior computed from
    the cluster-size frequency spectrum. The prior is factorized and corpus-level
    (same demos for all test points).
    
    Args:
        cluster_ids: np.ndarray shape (N_train,), aligned with index dataset order.
                    For any retrieved candidate id `i` (embedding index), we use cluster_ids[i].
        ignore_label: Cluster label to ignore in SGT computation (default -1).
        t_prior: SGT parameter t for prior estimation (default 5, not used in Option A but kept for API).
        bin_size: SGT bin_size parameter (default 20, not used in Option A but kept for API).
        smooth_count: Whether to smooth frequency spectrum before fitting (default False).
        spectrum_model: Model for spectrum smoothing: "powerlaw", "poisson", "negbin", or "none" (default "powerlaw").
        sgt_lambda: Weight for SGT prior term: score = v(i) + sgt_lambda * log(w_c) (default 0.1, consistent with dpp_sgt/mdl_sgt).
        prior_eps: Epsilon for numerical stability in weight computation (default 1e-12).
        include_zero_vote_candidates: If True, candidates with 0 votes are still ranked with v=0 (default True).
        seed: Random seed for deterministic behavior (default 42).
        Other args: Same as TopkRetriever and VotekRetriever.
    """
    
    def __init__(
        self,
        dataset_reader: DatasetReader,
        cluster_ids: np.ndarray,
        ignore_label: int = -1,
        t_prior: int = 5,
        bin_size: int = 20,
        smooth_count: bool = False,
        spectrum_model: str = "powerlaw",
        sgt_lambda: float = 0.1,
        prior_eps: float = 1e-12,
        include_zero_vote_candidates: bool = True,
        ice_separator: Optional[str] = "\n",
        ice_eos_token: Optional[str] = "\n",
        prompt_eos_token: Optional[str] = "",
        sentence_transformers_model_name: Optional[str] = "all-mpnet-base-v2",
        ice_num: Optional[int] = 1,
        index_split: Optional[str] = "train",
        test_split: Optional[str] = "test",
        tokenizer_name: Optional[str] = "gpt2-xl",
        batch_size: Optional[int] = 1,
        votek_k: Optional[int] = 3,
        accelerator: Optional[Accelerator] = None,
        seed: Optional[int] = 42,
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
        self.votek_k = int(votek_k)
        
        # cluster_ids must align with index embeddings (self.embed_list order)
        self.cluster_ids = np.asarray(cluster_ids).astype(int)
        self.ignore_label = int(ignore_label)
        self.t_prior = int(t_prior)
        self.bin_size = int(bin_size)
        self.smooth_count = bool(smooth_count)
        self.spectrum_model = str(spectrum_model)
        self.sgt_lambda = float(sgt_lambda)
        self.prior_eps = float(prior_eps)
        self.include_zero_vote_candidates = bool(include_zero_vote_candidates)
        self.seed = seed if seed is not None else 42
        
        if self.cluster_ids.ndim != 1:
            raise ValueError("cluster_ids must be a 1D array aligned with embeddings.")
        
        if self.spectrum_model not in {"powerlaw", "poisson", "negbin", "none"}:
            raise ValueError(f"spectrum_model must be one of: powerlaw, poisson, negbin, none. Got: {self.spectrum_model}")
        
        # Set random seed for deterministic behavior
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Cache for computed cluster weights
        self._cluster_weight: Optional[Dict[int, float]] = None
    
    def _compute_cluster_weights(self, n: int) -> Dict[int, float]:
        """
        Compute a factorized Good-Turing / SGT-inspired prior weight per cluster.

        This method computes dataset-level cluster sizes n_c and uses the
        frequency-of-frequencies spectrum over sizes:
            f_r = #{clusters c with size n_c = r}.

        We optionally smooth the observed spectrum f_r using a simple parametric
        smoother (powerlaw / poisson / negbin) over the non-zero size bins to
        obtain a nonnegative smoothed spectrum \\hat f_r.

        We then apply the Good-Turing adjusted-count rule to estimate the
        probability mass of a cluster of size r:
            r*(r) = (r+1) * \\hat f_{r+1} / \\hat f_r,
            p(r)  = r*(r) / N,
        where N is the total number of (non-ignored) examples.

        The cluster weight is set inversely proportional to this estimated mass:
            w_c = 1 / (p(n_c) + eps).

        For numerical robustness:
        - if \\hat f_r is zero or r*(r) is non-positive, we fall back to r*(r)=r
            (i.e., no GT correction for that bin);
        - ignore_label clusters are assigned weight 0;
        - all non-ignore_label weights are normalized to have mean 1.0.

        Returns:
            Dict[int, float]: mapping cluster_id -> normalized weight.
        """
        if len(self.cluster_ids) != n:
            raise ValueError(
                f"cluster_ids length mismatch: len(cluster_ids)={len(self.cluster_ids)} "
                f"but embeddings n={n}. They must be aligned."
            )
        
        # Step 1: Compute cluster sizes (ignore ignore_label)
        valid_cluster_ids = self.cluster_ids[self.cluster_ids != self.ignore_label]
        if len(valid_cluster_ids) == 0:
            logger.warning("No valid cluster IDs found (all are ignore_label). Using uniform weights.")
            # Return uniform weights for all clusters
            unique_clusters = np.unique(self.cluster_ids)
            return {int(c): 1.0 for c in unique_clusters}
        
        cluster_counts = Counter(int(c) for c in valid_cluster_ids)
        cluster_sizes = np.array(list(cluster_counts.values()))
        
        # Step 2: Build frequency-of-frequencies spectrum
        # f_s = number of clusters with size exactly s
        size_counts = Counter(cluster_sizes)
        max_size = int(max(cluster_sizes)) if len(cluster_sizes) > 0 else 1
        freq_by_size = np.zeros(max_size, dtype=float)
        for size, count in size_counts.items():
            if 1 <= size <= max_size:
                freq_by_size[size - 1] = float(count)
        
        # Step 3: Fit parametric spectrum model
        if self.spectrum_model == "none":
            logger.info("spectrum_model='none': using raw frequency spectrum (no smoothing) + Good-Turing weights.")
            # No smoothing: use the raw frequency spectrum
            fhat_full = freq_by_size.copy()
        else:
            # Fit spectrum model
            non_zero_indices = np.where(freq_by_size > 0)[0]
            if len(non_zero_indices) == 0:
                logger.warning("Empty frequency spectrum. Using uniform weights.")
                unique_clusters = np.unique(self.cluster_ids)
                return {int(c): 1.0 for c in unique_clusters}
            
            # Extract non-zero portion for fitting
            non_zero_sizes = non_zero_indices + 1  # sizes are 1-indexed
            non_zero_counts = freq_by_size[non_zero_indices]
            
            # Handle edge case: if only one unique size, fitting may fail or be degenerate
            if len(non_zero_counts) == 1:
                # All clusters have the same size - use uniform weights (or inverse count)
                logger.info(f"All clusters have the same size ({non_zero_sizes[0]}). Using uniform weights.")
                unique_clusters = np.unique(self.cluster_ids)
                valid_clusters = [c for c in unique_clusters if int(c) != self.ignore_label]
                return {int(c): 1.0 for c in valid_clusters}
            
            # Fit spectrum model
            if self.spectrum_model == "powerlaw":
                fitted = smooth_counts_powerlaw(non_zero_counts, k=len(non_zero_counts))
            elif self.spectrum_model == "poisson":
                fitted = smooth_counts_poisson(non_zero_counts, k=len(non_zero_counts))
            elif self.spectrum_model == "negbin":
                fitted = smooth_counts_binomial(non_zero_counts)
            else:
                raise ValueError(f"Unknown spectrum_model: {self.spectrum_model}")
            
            # Ensure fitted has correct length
            if len(fitted) != len(non_zero_counts):
                logger.warning(f"Fitted spectrum length ({len(fitted)}) != input length ({len(non_zero_counts)}). Truncating/padding.")
                if len(fitted) < len(non_zero_counts):
                    fitted = np.pad(fitted, (0, len(non_zero_counts) - len(fitted)), mode='constant', constant_values=0.0)
                else:
                    fitted = fitted[:len(non_zero_counts)]
            
            # Reconstruct full spectrum (pad with zeros for missing sizes)
            fhat_full = np.zeros(max_size, dtype=float)
            for i, size in enumerate(non_zero_sizes):
                if 1 <= size <= max_size:
                    fhat_full[size - 1] = max(0.0, fitted[i] if i < len(fitted) else 0.0)
        
        fhat_full = np.maximum(fhat_full, 0.0)    # Ensure non-negative
        # Step 4: Good–Turing adjusted mass p(r) from smoothed fhat
        # fhat_full[r-1] ~= \hat f_r
        N = int(len(valid_cluster_ids))  # total examples excluding ignore_label

        # Build shifted fhat_{r+1} with safe padding
        fhat_r = fhat_full  # length max_size, index r-1
        fhat_rplus = np.zeros_like(fhat_r)
        if max_size >= 2:
            fhat_rplus[:-1] = fhat_r[1:]  # fhat_{r+1} at index r-1
        fhat_rplus[-1] = 0.0  # no estimate for fhat_{max_size+1}

        # Compute r* and p(r); clip for numerical safety
        r_vals = np.arange(1, max_size + 1, dtype=float)

        # Avoid division by 0: if fhat_r is 0, fall back to empirical r (no GT correction)
        den = np.maximum(fhat_r, self.prior_eps)
        ratio = fhat_rplus / den
        r_star = (r_vals + 1.0) * ratio

        # Fallback for degenerate cases: if r_star <= 0, use empirical r
        r_star = np.where(r_star > 0.0, r_star, r_vals)

        # Convert to probability mass
        p_r = r_star / max(float(N), 1.0)   # p(r) for each r=1..max_size

        # Turn into weights (larger weight for smaller mass)
        weights: Dict[int, float] = {}
        for c, size in cluster_counts.items():
            if 1 <= size <= max_size:
                p = float(p_r[size - 1])
                weights[c] = 1.0 / (p + self.prior_eps)
            else:
                # size out of range; use empirical mass
                p = float(size) / max(float(N), 1.0)
                weights[c] = 1.0 / (p + self.prior_eps)
        
        # Step 4.5: Ensure ignore_label has weight 0 and every cluster id has an entry
        all_clusters = np.unique(self.cluster_ids)
        for c in all_clusters:
            c = int(c)
            if c == self.ignore_label:
                weights[c] = 0.0
            elif c not in weights:
                weights[c] = 0.0
                
        
        # Step 5: Normalize weights to have mean 1.0 (only over non-ignore_label clusters)
        valid_weights = {c: w for c, w in weights.items() if int(c) != self.ignore_label and w > 0}
        if len(valid_weights) > 0:
            mean_weight = np.mean(list(valid_weights.values()))
            if mean_weight > 0:
                # Normalize only valid weights, keep ignore_label at 0
                weights = {c: (w / mean_weight if int(c) != self.ignore_label and w > 0 else 0.0) 
                          for c, w in weights.items()}
        
        logger.info(
            f"Computed SGT cluster weights: {len(weights)} clusters, "
            f"mean={np.mean(list(weights.values())):.4f}, "
            f"min={np.min(list(weights.values())):.4f}, "
            f"max={np.max(list(weights.values())):.4f}"
        )
        
        return weights
    
    def votek_select(
        self,
        embeddings: np.ndarray,
        select_num: int,
        k: int,
        overlap_threshold: float = 1.0,
        vote_file: Optional[str] = None,
    ) -> List[int]:
        """
        VoteK selection with SGT-weighted ranking.
        
        Same structure as original VoteK, but ranks by:
            score(i) = v(i) + sgt_lambda * log(w_{cluster(i)})
        
        where v(i) is the vote count and w_{cluster(i)} is the SGT-derived weight.
        This additive formulation is consistent with dpp_sgt and mdl_sgt retrievers.
        """
        n = len(embeddings)
        if n == 0:
            return []
        
        # Validate cluster_ids length matches embeddings
        if len(self.cluster_ids) != n:
            raise ValueError(
                f"cluster_ids length ({len(self.cluster_ids)}) must match "
                f"embeddings length ({n})"
            )
        
        # Precompute SGT cluster weights once per run
        if self._cluster_weight is None:
            self._cluster_weight = self._compute_cluster_weights(n)
        
        # Load or compute vote_stat: candidate -> list of voters
        if vote_file is not None and os.path.isfile(vote_file):
            with open(vote_file, "r") as f:
                vote_stat = json.load(f)
            # json keys are strings; convert to ints / list of ints
            vote_stat = {int(k): [int(x) for x in v] for k, v in vote_stat.items()}
        else:
            vote_stat = defaultdict(list)
            # NOTE: This is O(n^2). Use FAISS if n is large.
            logger.info(f"Computing vote statistics for {n} points (O(n^2), may take a while)...")
            for i in range(n):
                cur_emb = embeddings[i].reshape(1, -1)
                cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
                sorted_indices = np.argsort(cur_scores).tolist()[-k - 1 : -1]
                for idx in sorted_indices:
                    if idx != i:
                        vote_stat[int(idx)].append(int(i))
            
            if vote_file is not None:
                os.makedirs(os.path.dirname(vote_file) if os.path.dirname(vote_file) else ".", exist_ok=True)
                with open(vote_file, "w") as f:
                    json.dump({str(k): v for k, v in vote_stat.items()}, f)
        
        # Build ranked list by weighted votes
        ranked = []
        all_candidates = set(range(n)) if self.include_zero_vote_candidates else set(vote_stat.keys())
        
        for cand in all_candidates:
            voters = vote_stat.get(cand, [])
            c = int(self.cluster_ids[int(cand)])
            
            # Skip ignore_label clusters
            if c == self.ignore_label:
                continue
            
            # Get cluster weight
            w = self._cluster_weight.get(c, 0.0)
            
            # Compute weighted score: v(i) + sgt_lambda * log(w_c)
            # Use log for consistency with dpp_sgt/mdl_sgt additive formulation
            # Handle edge case where w might be very small (use log(max(w, eps)))
            vote_count = len(voters)
            log_weight = math.log(max(float(w), self.prior_eps))
            weighted_score = float(vote_count) + self.sgt_lambda * log_weight
            
            ranked.append((int(cand), weighted_score, vote_count, c, w))
        
        # Sort by weighted score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Overlap filtering: compare against previously SELECTED items' voter sets
        # (better diversity among selected than among ranked)
        selected_indices: List[int] = []
        selected_voter_sets: List[set] = []
        
        for cand, score, vote_count, cluster_id, weight in ranked:
            if len(selected_indices) >= select_num:
                break
            
            candidate_voter_set = set(vote_stat.get(cand, []))
            
            # Check overlap with previously selected items
            flag = True
            for prev_voter_set in selected_voter_sets:
                overlap = len(candidate_voter_set.intersection(prev_voter_set))
                if overlap >= overlap_threshold * max(1, len(candidate_voter_set)):
                    flag = False
                    break
            
            if flag:
                selected_indices.append(cand)
                selected_voter_sets.append(candidate_voter_set)
        
        # Fallback: random fill if we couldn't pick enough
        # Use deterministic random sampling with seed
        if len(selected_indices) < select_num:
            unselected_indices = [i for i in range(n) if i not in set(selected_indices)]
            need = select_num - len(selected_indices)
            if unselected_indices:
                # Use module-level random with seed set in __init__ for determinism
                # Note: random.sample is deterministic if seed is set
                selected_indices += random.sample(unselected_indices, k=min(need, len(unselected_indices)))
        
        if selected_indices:
            sample_vote_counts = [len(vote_stat.get(i, [])) for i in selected_indices[:5]]
            sample_clusters = [int(self.cluster_ids[i]) for i in selected_indices[:10]]
            logger.debug(
                f"Selected {len(selected_indices)} items: "
                f"sample_vote_counts={sample_vote_counts}, "
                f"sample_clusters={Counter(sample_clusters)}"
            )
        
        return selected_indices[:select_num]
    
    def vote_k_search(self) -> List[List[int]]:
        """
        VoteK search: corpus-level selection (same demos for all test points).
        """
        vote_k_idxs = self.votek_select(
            embeddings=self.embed_list,
            select_num=self.ice_num,
            k=self.votek_k,
            overlap_threshold=1.0,
        )
        # corpus-level retriever: same demos for every test point
        return [vote_k_idxs[:] for _ in range(len(self.test_ds))]
    
    def retrieve(self) -> List[List[int]]:
        """Retrieve in-context examples using VoteK with SGT prior."""
        return self.vote_k_search()

