# OpenICL/openicl/icl_retriever/icl_dpp_sgt_retriever.py
"""DPP + SGT(Rarity) Retriever

This retriever is a plug-in extension of OpenICL's DPPRetriever:
  - Stage 1: TopK candidates via FAISS
  - Stage 2: DPP MAP (greedy) for diversity
Add-on:
  - Multiply DPP quality term by a cluster-based rarity prior:
      prior_i = (count(cluster_i) + eps)^(-alpha/2)
    so that kernel becomes:
      K_ij = (rel_i * prior_i) * sim_ij * (rel_j * prior_j)

The prior is static (computed once from cluster_ids).
"""

from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.logging import get_logger
from typing import Optional, List, Tuple
import tqdm
import numpy as np
import math
from accelerate import Accelerator

logger = get_logger(__name__)


class DPPSGTRetriever(TopkRetriever):
    """DPP Retriever with SGT-style rarity prior (cluster-frequency reweighting).

    Args:
        cluster_ids:
            np.ndarray shape (N_train,), aligned with index dataset order (train split).
            For any retrieved candidate id `i` (FAISS id), we use cluster_ids[i].
        rarity_alpha:
            exponent alpha in w_c = 1/(count(c)^alpha). Internally we use sqrt effect
            for kernel-factorization: prior_i = (count(c)+eps)^(-alpha/2).
        ignore_label:
            cluster label to ignore (prior=1). Default -1.
        prior_eps:
            numerical stability epsilon added to cluster count.
        prior_clip:
            optional clip (min,max) on prior_i to avoid extreme weights.
        prior_normalize:
            if True, normalize priors so that mean(prior_i) ~= 1 over train pool.
            This keeps scale more stable across different clustering granularities.
    """
    model = None

    def __init__(
        self,
        dataset_reader: DatasetReader,
        cluster_ids: np.ndarray,
        rarity_alpha: float = 0.5,
        ignore_label: int = -1,
        prior_eps: float = 1.0,
        prior_clip: Optional[Tuple[float, float]] = (0.2, 5.0),
        prior_normalize: bool = True,
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

        self.rarity_alpha = float(rarity_alpha)
        self.ignore_label = int(ignore_label)
        self.prior_eps = float(prior_eps)
        self.prior_clip = prior_clip
        self.prior_normalize = bool(prior_normalize)

        # Precompute point-level priors aligned with train/index ids
        self.point_prior = self._build_point_priors(self.cluster_ids)

    def _build_point_priors(self, cluster_ids: np.ndarray) -> np.ndarray:
        # cluster counts
        uniq, cnt = np.unique(cluster_ids, return_counts=True)
        count_map = {int(u): int(c) for u, c in zip(uniq, cnt)}

        pri = np.ones_like(cluster_ids, dtype=np.float32)
        if self.rarity_alpha <= 0:
            return pri

        # prior_i = (count(c)+eps)^(-alpha/2)  (sqrt because kernel uses q_i * q_j)
        for i, c in enumerate(cluster_ids):
            if c == self.ignore_label:
                pri[i] = 1.0
            else:
                pri[i] = float((count_map.get(int(c), 1) + self.prior_eps) ** (-0.5 * self.rarity_alpha))

        if self.prior_clip is not None:
            lo, hi = self.prior_clip
            pri = np.clip(pri, lo, hi)

        if self.prior_normalize:
            m = float(np.mean(pri))
            if m > 0:
                pri = pri / m

        return pri.astype(np.float32)

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

            # Stage 2: DPP kernel (with rarity prior baked into rel_scores)
            near_reps, rel_scores, kernel_matrix = self.get_kernel(embed, near_ids.tolist())

            # MAP inference (greedy)
            samples_ids = fast_map_dpp(kernel_matrix, self.ice_num)

            # order chosen items by relevance (after applying prior)
            samples_ids = np.array(samples_ids)
            samples_scores = np.array([rel_scores[i] for i in samples_ids])
            samples_ids = samples_ids[(-samples_scores).argsort()].tolist()

            rtr_sub_list = [int(near_ids[i]) for i in samples_ids]
            rtr_idx_list[idx] = rtr_sub_list

        return rtr_idx_list

    def get_kernel(self, embed: np.ndarray, candidates: List[int]):
        # reconstruct candidate vectors from FAISS
        near_reps = np.stack([self.index.index.reconstruct(i) for i in candidates], axis=0)

        # normalize first (cosine sim)
        embed = embed / (np.linalg.norm(embed) + 1e-12)
        near_reps = near_reps / (np.linalg.norm(near_reps, keepdims=True, axis=1) + 1e-12)

        # relevance scores in [0,1]
        rel_scores = np.matmul(embed, near_reps.T)[0]
        rel_scores = (rel_scores + 1.0) / 2.0

        # stabilize exp
        rel_scores -= rel_scores.max()
        rel_scores = np.exp(rel_scores / (2.0 * self.scale_factor + 1e-12))

        # ---- SGT rarity prior (add-on) ----
        # point_prior aligned with train ids, map each candidate id -> prior
        priors = np.array([self.point_prior[int(i)] for i in candidates], dtype=np.float32)
        rel_scores = rel_scores * priors
        # ----------------------------------

        # similarity matrix in [0,1]
        sim_matrix = np.matmul(near_reps, near_reps.T)
        sim_matrix = (sim_matrix + 1.0) / 2.0

        # DPP kernel (quality * similarity * quality)
        kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
        return near_reps, rel_scores, kernel_matrix


def fast_map_dpp(kernel_matrix, max_length):
    """
    Fast implementation of the greedy algorithm
    reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
    paper: Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(int(selected_item))
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(max(di2s[selected_item], 1e-12))
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        selected_items.append(int(selected_item))
    return selected_items