# selection.py

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import random
from collections import Counter, defaultdict

from utils_sgt import smoothed_good_toulmin_sgt


def init_seed_indices(
    n_points: int,
    seed_size: int,
    rng: random.Random,
) -> List[int]:
    """Randomly choose an initial seed set of indices."""
    assert seed_size <= n_points
    return rng.sample(range(n_points), seed_size)

def _khat_sgt_for_indices(
    cluster_ids: np.ndarray,
    indices: List[int],
    t: int = 5,
    bin_size: int = 20,
    smooth_count: bool = False,
    offset: float = 1.0,
    ignore_label: int = -1,
) -> float:
    """
    Return K̂_S = seen + unseen_hat(t), where unseen_hat is estimated by SGT
    from the subset frequency spectrum.

    Notes:
    - Excludes ignore_label (e.g., DBSCAN noise = -1) from the type universe.
    - If SGT returns negative/NaN, we clamp unseen_hat at 0.
    """
    if len(indices) == 0:
        return 0.0

    subset_cluster_ids = cluster_ids[indices]
    subset_cluster_ids = subset_cluster_ids[subset_cluster_ids != ignore_label]
    if len(subset_cluster_ids) == 0:
        return 0.0

    cluster_freq = Counter(int(c) for c in subset_cluster_ids)
    seen = len(cluster_freq)

    # Build spectrum: f_s = #types that appear exactly s times
    freq_by_s = defaultdict(int)
    for cnt in cluster_freq.values():
        if cnt > 0:
            freq_by_s[cnt] += 1

    if not freq_by_s:
        return float(seen)

    unseen_list, _ = smoothed_good_toulmin_sgt(
        freq_by_s,
        t_list=[t],
        bin_size=bin_size,
        smooth_count=smooth_count,
        mute=True,
        offset=offset,
    )
    unseen_hat = float(unseen_list[0])
    if not np.isfinite(unseen_hat) or unseen_hat < 0:
        unseen_hat = 0.0

    return float(seen + unseen_hat)


def _sgt_score_for_indices(
    cluster_ids: np.ndarray,
    indices: List[int],
    t: int = 5,
    bin_size: int = 20,
    smooth_count: bool = False,
    offset: float = 1.0,
    total_clusters: int = None,
) -> float:
    """
    Compute SGT-based coverage score for a given subset of indices.

    Steps:
    - Count how many times each cluster appears in the subset.
    - Build a frequency spectrum: counts[s] = #clusters that appear exactly s times.
    - Run SGT to estimate unseen clusters, OR use total_clusters if provided.
    - Return score = seen / (seen + unseen) or seen / total_clusters if total_clusters is provided.

    Args:
        total_clusters: If provided, use this instead of SGT estimate for unseen clusters.
                        This is more accurate when we know the total number of clusters in the pool.
    """
    if len(indices) == 0:
        return 0.0

    subset_cluster_ids = cluster_ids[indices]
    # cluster -> how many times it appears in the subset
    cluster_freq = Counter(int(c) for c in subset_cluster_ids)
    seen = len(cluster_freq)

    # If we know the total number of clusters, use that instead of SGT estimate
    if total_clusters is not None:
        unseen = max(0, total_clusters - seen)
        score = seen / (seen + unseen + 1e-8)
        return float(score)

    # Otherwise, use SGT to estimate unseen clusters
    # Build spectrum: s -> number of clusters that appear exactly s times
    freq_by_s = defaultdict(int)
    for count in cluster_freq.values():
        if count > 0:
            freq_by_s[count] += 1

    if not freq_by_s:
        return 0.0

    # SGT expects: counts[s] = number of types with frequency s
    unseen_list, _ = smoothed_good_toulmin_sgt(
        freq_by_s,
        t_list=[t],
        bin_size=bin_size,
        smooth_count=smooth_count,
        mute=True,
        offset=offset,
    )
    unseen = max(unseen_list[0], 0)

    score = seen / (seen + unseen + 1e-8)
    
    # Debug: log the frequency spectrum and SGT estimate
    import logging
    logging.debug(f"SGT score computation: seen={seen}, unseen_estimate={unseen}, "
                 f"frequency_spectrum={dict(freq_by_s)}, score={score:.4f}")
    
    return float(score)


def greedy_sgt_selection(
    cluster_ids: np.ndarray,
    budget: int,
    seed_size: int,
    rng: random.Random,
    t: int = 5,
    bin_size: int = 20,
    smooth_count: bool = False,
    offset: float = 1.0,
    normalize_by_kall: bool = False,
    K_all: int = None,
    ignore_label: int = -1,
) -> Tuple[List[int], List[float]]:
    """
    Greedy selection to maximize K̂_S = seen + unseen_hat(t) (SGT).

    If normalize_by_kall=True, returns score history for (K̂_S / K_all).
    """
    n_points = len(cluster_ids)
    assert budget >= seed_size
    assert budget <= n_points

    selected: List[int] = init_seed_indices(n_points, seed_size, rng)
    remaining = set(range(n_points)) - set(selected)

    def score(indices: List[int]) -> float:
        khat = _khat_sgt_for_indices(
            cluster_ids, indices,
            t=t, bin_size=bin_size, smooth_count=smooth_count, offset=offset,
            ignore_label=ignore_label
        )
        if normalize_by_kall:
            if K_all is None:
                raise ValueError("If normalize_by_kall=True, you must pass K_all.")
            return float(khat) / float(K_all + 1e-12)
        return float(khat)

    current_score = score(selected)
    score_history: List[float] = [current_score]

    while len(selected) < budget and remaining:
        best_idx = None
        best_new_score = current_score

        for idx in remaining:
            new_score = score(selected + [idx])
            if new_score > best_new_score:
                best_new_score = new_score
                best_idx = idx

        if best_idx is None:
            best_idx = remaining.pop()
            best_new_score = score(selected + [best_idx])
        else:
            remaining.remove(best_idx)

        selected.append(best_idx)
        current_score = best_new_score
        score_history.append(current_score)

    return selected, score_history

class SGTSpectrumState:
    def __init__(self, t: int, bin_size: int, smooth_count: bool, offset: float, ignore_label: int = -1):
        self.t = t
        self.bin_size = bin_size
        self.smooth_count = smooth_count
        self.offset = offset
        self.ignore_label = ignore_label

        self.type_counts = Counter()          # cluster -> multiplicity in subset
        self.freq_by_s = defaultdict(int)     # s -> #types with multiplicity s
        self.seen = 0

    def add(self, c: int) -> None:
        if c == self.ignore_label:
            return
        m = self.type_counts[c]
        if m > 0:
            self.freq_by_s[m] -= 1
            if self.freq_by_s[m] == 0:
                del self.freq_by_s[m]
        else:
            self.seen += 1

        self.type_counts[c] += 1
        self.freq_by_s[m + 1] += 1

    def khat(self) -> float:
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

def greedy_sgt_selection_fast(
    cluster_ids: np.ndarray,
    budget: int,
    seed_size: int,
    rng: random.Random,
    t: int = 5,
    bin_size: int = 20,
    smooth_count: bool = False,
    offset: float = 1.0,
    ignore_label: int = -1,
    candidate_pool_per_step: int = 256,
) -> Tuple[List[int], List[float]]:
    """
    Fast greedy: each step evaluates only a random subset of remaining candidates,
    biased toward clusters that are currently under-covered.
    """
    n = len(cluster_ids)
    assert budget >= seed_size

    selected = init_seed_indices(n, seed_size, rng)
    remaining = set(range(n)) - set(selected)

    state = SGTSpectrumState(t=t, bin_size=bin_size, smooth_count=smooth_count, offset=offset, ignore_label=ignore_label)
    for idx in selected:
        state.add(int(cluster_ids[idx]))

    score_history = [state.khat()]

    # Build mapping cluster -> indices for sampling
    cluster_to_indices = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        if int(c) != ignore_label:
            cluster_to_indices[int(c)].append(i)

    while len(selected) < budget and remaining:
        # candidate pool: mix of (a) unseen clusters, (b) random
        candidates = []

        # (a) sample from clusters with low current multiplicity
        # pick clusters with smallest counts first
        low_clusters = sorted(cluster_to_indices.keys(), key=lambda c: state.type_counts[c])[:128]
        for c in low_clusters:
            # try to pick an unused index from this cluster
            inds = cluster_to_indices[c]
            rng.shuffle(inds)
            for i in inds[:5]:
                if i in remaining:
                    candidates.append(i)
                    break
            if len(candidates) >= candidate_pool_per_step // 2:
                break

        # (b) fill with random remaining
        if len(candidates) < candidate_pool_per_step:
            extra = rng.sample(list(remaining), k=min(candidate_pool_per_step - len(candidates), len(remaining)))
            candidates.extend(extra)

        # evaluate candidates by simulating one add
        best_idx = None
        best_score = score_history[-1]

        for idx in candidates:
            c = int(cluster_ids[idx])
            if c == ignore_label:
                continue

            # simulate add with local delta update
            m = state.type_counts[c]
            # apply
            if m > 0:
                state.freq_by_s[m] -= 1
                if state.freq_by_s[m] == 0:
                    del state.freq_by_s[m]
            else:
                state.seen += 1
            state.type_counts[c] += 1
            state.freq_by_s[m + 1] += 1

            s = state.khat()

            # revert
            state.freq_by_s[m + 1] -= 1
            if state.freq_by_s[m + 1] == 0:
                del state.freq_by_s[m + 1]
            state.type_counts[c] -= 1
            if state.type_counts[c] == 0:
                del state.type_counts[c]
                state.seen -= 1
            else:
                state.freq_by_s[m] += 1

            if s > best_score:
                best_score = s
                best_idx = idx

        if best_idx is None:
            best_idx = remaining.pop()
        else:
            remaining.remove(best_idx)

        selected.append(best_idx)
        state.add(int(cluster_ids[best_idx]))
        score_history.append(state.khat())

    return selected, score_history

def _top_m_cluster_ids(
    cluster_ids: np.ndarray,
    top_m: int,
    ignore_label: int = -1,
) -> List[int]:
    """
    Return the cluster IDs of the top_m largest clusters (by count),
    excluding ignore_label.
    """
    cluster_ids = np.asarray(cluster_ids)
    valid = cluster_ids[cluster_ids != ignore_label]
    if valid.size == 0:
        return []

    counts = Counter(int(c) for c in valid)
    # Sort by (size desc, cluster_id asc) for determinism
    sorted_clusters = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_clusters = [cid for cid, _ in sorted_clusters[:max(0, int(top_m))]]
    return top_clusters


def robust_sgt_selection_fast_top_clusters(
    cluster_ids: np.ndarray,
    budget: int,
    seed_size: int,
    rng: random.Random,
    top_m: int = 50,
    t: int = 5,
    bin_size: int = 20,
    smooth_count: bool = False,
    offset: float = 1.0,
    ignore_label: int = -1,
    candidate_pool_per_step: int = 256,
    fill_strategy: str = "random_tail",  # {"random_tail", "random_head", "none"}
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Robust SGT selection that restricts the SGT objective to the top_m largest clusters.

    Steps:
      1) Identify top_m clusters by size (excluding ignore_label).
      2) Run greedy_sgt_selection_fast on the induced subproblem (eligible indices).
      3) If selected < budget, optionally fill remaining slots using fill_strategy.

    Returns:
      selected_global_indices, debug_info

    Notes:
      - This is meant to avoid chasing tiny/noisy clusters (e.g., dict_argmax tail, DBSCAN noise).
      - If top_m is too small, you may under-cover label diversity; treat top_m as a knob.
    """
    cluster_ids = np.asarray(cluster_ids)
    n = int(cluster_ids.shape[0])
    assert 0 < budget <= n
    assert 0 < seed_size <= budget

    top_clusters = _top_m_cluster_ids(cluster_ids, top_m=top_m, ignore_label=ignore_label)

    # Eligible indices are those in the head clusters
    if len(top_clusters) > 0:
        head_mask = np.isin(cluster_ids, np.array(top_clusters, dtype=cluster_ids.dtype))
        eligible = np.where(head_mask)[0]
    else:
        eligible = np.array([], dtype=np.int64)

    debug: Dict[str, Any] = {
        "top_m": int(top_m),
        "n_points": n,
        "budget": int(budget),
        "seed_size": int(seed_size),
        "n_top_clusters": int(len(top_clusters)),
        "n_eligible_points": int(len(eligible)),
        "fill_strategy": str(fill_strategy),
    }

    selected_global: List[int] = []

    # If we have enough eligible points, do SGT on head.
    # Otherwise, do as much as we can on head and fill later.
    if len(eligible) >= seed_size:
        # Build reduced cluster_ids on eligible points
        reduced_cluster_ids = cluster_ids[eligible]

        # Run existing fast greedy on the reduced problem
        # NOTE: ignore_label is still respected in case eligible contains it (shouldn't).
        local_budget = min(int(budget), int(len(eligible)))
        local_seed_size = min(int(seed_size), int(local_budget))

        selected_local, score_hist = greedy_sgt_selection_fast(
            cluster_ids=reduced_cluster_ids,
            budget=local_budget,
            seed_size=local_seed_size,
            rng=rng,
            t=t,
            bin_size=bin_size,
            smooth_count=smooth_count,
            offset=offset,
            ignore_label=ignore_label,
            candidate_pool_per_step=candidate_pool_per_step,
        )

        selected_global = [int(eligible[i]) for i in selected_local]
        debug["sgt_score_history"] = score_hist
        debug["n_selected_head"] = int(len(selected_global))
    else:
        # Not enough head points even for seed; fall back to random head (or empty) then fill
        debug["sgt_score_history"] = []
        debug["n_selected_head"] = 0
        if len(eligible) > 0:
            # pick as many as possible from head
            k = min(int(budget), int(len(eligible)))
            selected_global = rng.sample([int(i) for i in eligible.tolist()], k=k)

    # Fill if needed
    if len(selected_global) < budget:
        remaining = [i for i in range(n) if i not in set(selected_global)]

        if fill_strategy == "none":
            pass
        elif fill_strategy == "random_head":
            # Fill from remaining head points
            remaining_head = [i for i in remaining if (cluster_ids[i] in set(top_clusters))]
            need = budget - len(selected_global)
            if remaining_head:
                take = rng.sample(remaining_head, k=min(need, len(remaining_head)))
                selected_global.extend(int(i) for i in take)
        elif fill_strategy == "random_tail":
            # Fill from remaining non-head points (tail), purely random
            remaining_tail = [i for i in remaining if (cluster_ids[i] not in set(top_clusters))]
            need = budget - len(selected_global)
            if remaining_tail:
                take = rng.sample(remaining_tail, k=min(need, len(remaining_tail)))
                selected_global.extend(int(i) for i in take)
            # If still short, fill from whatever remains
            if len(selected_global) < budget:
                remaining2 = [i for i in range(n) if i not in set(selected_global)]
                need2 = budget - len(selected_global)
                if remaining2:
                    take2 = rng.sample(remaining2, k=min(need2, len(remaining2)))
                    selected_global.extend(int(i) for i in take2)
        else:
            raise ValueError(f"Unknown fill_strategy={fill_strategy}. Use 'random_tail', 'random_head', or 'none'.")

    # Final sanity: unique + correct length
    seen = set()
    dedup = []
    for i in selected_global:
        if i not in seen:
            seen.add(i)
            dedup.append(int(i))
    selected_global = dedup[:budget]

    debug["n_selected_total"] = int(len(selected_global))
    debug["n_selected_tail"] = int(len(selected_global) - debug.get("n_selected_head", 0))

    return selected_global, debug