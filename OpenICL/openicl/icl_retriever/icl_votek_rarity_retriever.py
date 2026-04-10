# openicl/icl_retriever/icl_votek_rarity_retriever.py

import os
import json
import random
from typing import Optional, List, Dict, Any
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from accelerate import Accelerator

from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever


class VotekRarityRetriever(TopkRetriever):
    """
    VoteK with cluster-rarity weighting.

    Original VoteK ranks candidates by number of votes (how many points select it as a neighbor).
    This variant ranks by:
        weighted_votes(i) = votes(i) * rarity_weight(cluster(i))
    where:
        rarity_weight(c) = 1 / (count(c) ** rarity_alpha)

    This is a cheap way to inject "tail / unseen-ish" preference (rare clusters) into VoteK,
    without any LLM calls.
    """

    def __init__(
        self,
        dataset_reader: DatasetReader,
        cluster_ids: np.ndarray,
        rarity_alpha: float = 0.5,
        ignore_label: int = -1,
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
        self.cluster_ids = np.asarray(cluster_ids)
        self.rarity_alpha = float(rarity_alpha)
        self.ignore_label = int(ignore_label)

        if self.cluster_ids.ndim != 1:
            raise ValueError("cluster_ids must be a 1D array aligned with embeddings.")
        # We can't assert length here if embeddings not built yet, but we will check in votek_select.

        self._cluster_weight: Optional[Dict[int, float]] = None

    def _compute_cluster_weights(self, n: int) -> Dict[int, float]:
        """
        Compute rarity weights w_c = 1 / (count(c)^alpha).
        """
        if len(self.cluster_ids) != n:
            raise ValueError(
                f"cluster_ids length mismatch: len(cluster_ids)={len(self.cluster_ids)} "
                f"but embeddings n={n}. They must be aligned."
            )

        cnt = Counter(int(c) for c in self.cluster_ids.tolist() if int(c) != self.ignore_label)
        weights: Dict[int, float] = {}
        for c, k in cnt.items():
            # avoid division by 0
            weights[c] = 1.0 / (float(k) ** self.rarity_alpha + 1e-12)
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
        Same structure as original VoteK, but ranks by weighted votes.
        """
        n = len(embeddings)
        if n == 0:
            return []

        # Precompute rarity weights once per run
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
            for i in range(n):
                cur_emb = embeddings[i].reshape(1, -1)
                cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
                sorted_indices = np.argsort(cur_scores).tolist()[-k - 1 : -1]
                for idx in sorted_indices:
                    if idx != i:
                        vote_stat[int(idx)].append(int(i))

            if vote_file is not None:
                with open(vote_file, "w") as f:
                    json.dump({str(k): v for k, v in vote_stat.items()}, f)

        # Build ranked list by weighted votes
        ranked = []
        for cand, voters in vote_stat.items():
            c = int(self.cluster_ids[int(cand)])
            if c == self.ignore_label:
                continue
            w = self._cluster_weight.get(c, 0.0)
            score = float(len(voters)) * float(w)
            ranked.append((int(cand), score, len(voters)))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # Original overlap filtering logic (kept)
        selected_indices: List[int] = []
        j = 0

        # For overlap checking we need sets of voters for already-considered items
        # We'll keep a parallel list for convenience
        ranked_voter_sets = [set(vote_stat[cand]) for cand, _, _ in ranked]

        while len(selected_indices) < select_num and j < len(ranked):
            candidate_set = ranked_voter_sets[j]
            flag = True
            # Compare to previously ranked items (same as original code's spirit)
            for pre in range(j):
                cur_set = ranked_voter_sets[pre]
                if len(candidate_set.intersection(cur_set)) >= overlap_threshold * max(1, len(candidate_set)):
                    flag = False
                    break
            if not flag:
                j += 1
                continue
            selected_indices.append(int(ranked[j][0]))
            j += 1

        # Fallback: random fill if we couldn't pick enough
        if len(selected_indices) < select_num:
            unselected_indices = [i for i in range(n) if i not in set(selected_indices)]
            need = select_num - len(selected_indices)
            if unselected_indices:
                selected_indices += random.sample(unselected_indices, k=min(need, len(unselected_indices)))

        return selected_indices[:select_num]

    def vote_k_search(self) -> List[List[int]]:
        vote_k_idxs = self.votek_select(
            embeddings=self.embed_list,
            select_num=self.ice_num,
            k=self.votek_k,
            overlap_threshold=1.0,
        )
        # corpus-level retriever: same demos for every test point
        return [vote_k_idxs[:] for _ in range(len(self.test_ds))]

    def retrieve(self) -> List[List[int]]:
        return self.vote_k_search()
