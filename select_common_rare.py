# src/select_common_rare.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
import numpy as np


@dataclass
class CommonRareSelectionConfig:
    budget: int = 100
    frac_rare: float = 0.4            # 40% rare, 60% common/rep
    rare_quantile: float = 0.3        # atoms in bottom 30% (non-zero) are "rare"
    common_quantile: float = 0.7      # atoms in top 30% (non-zero) are "common"

    # Greedy objective: score = new_covered + lambda_rep * rep_score
    lambda_rep: float = 0.15

    # If True, require that a candidate contains at least one target atom during that phase
    enforce_phase_membership: bool = True

    random_state: int = 0


def _cosine_to_centroid(X: np.ndarray) -> np.ndarray:
    """
    rep_score[i] = cosine(x_i, centroid)
    """
    X = np.asarray(X, dtype=np.float32)
    centroid = X.mean(axis=0, keepdims=True)
    # normalize
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cn = centroid / (np.linalg.norm(centroid) + 1e-12)
    return (Xn @ cn.T).reshape(-1)


def _build_atom_groups(freq: np.ndarray, rare_q: float, common_q: float) -> Tuple[Set[int], Set[int]]:
    """
    rare_atoms = bottom rare_q quantile of non-zero freqs
    common_atoms = top (1-common_q) quantile of non-zero freqs
    """
    freq = np.asarray(freq)
    nz = freq[freq > 0]
    if len(nz) == 0:
        return set(), set()

    rare_thr = np.quantile(nz, rare_q)
    common_thr = np.quantile(nz, common_q)

    rare_atoms = set(int(j) for j in np.where((freq > 0) & (freq <= rare_thr))[0])
    common_atoms = set(int(j) for j in np.where(freq >= common_thr)[0])
    return rare_atoms, common_atoms


def _greedy_cover_phase(
    active_sets: List[Set[int]],
    candidate_indices: np.ndarray,
    target_atoms: Set[int],
    budget: int,
    rep_score: np.ndarray,
    lambda_rep: float,
    already_selected: Set[int],
    enforce_membership: bool,
    rng: np.random.Generator,
) -> List[int]:
    """
    Greedy set cover with representativeness bonus.
    """
    if budget <= 0 or len(candidate_indices) == 0 or len(target_atoms) == 0:
        return []

    selected: List[int] = []
    covered: Set[int] = set()

    # For speed: pre-intersect each candidate's active atoms with target_atoms once
    cand_atoms: Dict[int, Set[int]] = {}
    for idx in candidate_indices:
        i = int(idx)
        if i in already_selected:
            continue
        inter = active_sets[i].intersection(target_atoms)
        if enforce_membership and len(inter) == 0:
            continue
        cand_atoms[i] = inter

    remaining = set(cand_atoms.keys())
    if len(remaining) == 0:
        return []

    while len(selected) < budget and len(remaining) > 0:
        best_i = None
        best_score = -1e18

        # Greedy: maximize (newly covered atoms) + lambda * representativeness
        for i in list(remaining):
            new_cov = len(cand_atoms[i] - covered)
            score = float(new_cov) + float(lambda_rep) * float(rep_score[i])
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        selected.append(best_i)
        already_selected.add(best_i)
        covered |= cand_atoms[best_i]
        remaining.remove(best_i)

        # If everything is covered, we can optionally stop early
        if covered >= target_atoms:
            break

    return selected


def select_common_plus_rare(
    X: np.ndarray,
    active_sets: List[Set[int]],
    atom_freq: np.ndarray,
    config: CommonRareSelectionConfig = CommonRareSelectionConfig(),
) -> Tuple[List[int], Dict[str, object]]:
    """
    Returns:
      selected_indices, debug_info
    """
    rng = np.random.default_rng(config.random_state)
    N = X.shape[0]
    B = int(config.budget)
    assert B <= N

    rep_score = _cosine_to_centroid(X)
    rare_atoms, common_atoms = _build_atom_groups(atom_freq, config.rare_quantile, config.common_quantile)

    # budgets
    B_rare = int(round(B * config.frac_rare))
    B_common = B - B_rare

    all_indices = np.arange(N, dtype=np.int64)
    already_selected: Set[int] = set()

    # Phase 1: rare coverage (tail)
    sel_rare = _greedy_cover_phase(
        active_sets=active_sets,
        candidate_indices=all_indices,
        target_atoms=rare_atoms,
        budget=B_rare,
        rep_score=rep_score,
        lambda_rep=config.lambda_rep,
        already_selected=already_selected,
        enforce_membership=config.enforce_phase_membership,
        rng=rng,
    )

    # Phase 2: common + representativeness (head)
    # Here, enforcing membership can be too strict (you still want typical points),
    # so we often relax it. But keep it configurable.
    sel_common = _greedy_cover_phase(
        active_sets=active_sets,
        candidate_indices=all_indices,
        target_atoms=common_atoms if len(common_atoms) > 0 else set(range(len(atom_freq))),
        budget=B_common,
        rep_score=rep_score,
        lambda_rep=max(config.lambda_rep, 0.5),  # emphasize representativeness in common phase
        already_selected=already_selected,
        enforce_membership=False,  # <--- important: avoid "all-tail" behavior
        rng=rng,
    )

    selected = sel_rare + sel_common

    # If still short (e.g., strict filters), fill randomly among remaining
    if len(selected) < B:
        remaining = np.array([i for i in range(N) if i not in already_selected], dtype=np.int64)
        if len(remaining) > 0:
            fill = rng.choice(remaining, size=min(B - len(selected), len(remaining)), replace=False).tolist()
            selected.extend(int(i) for i in fill)

    debug = {
        "N": N,
        "budget": B,
        "B_rare": B_rare,
        "B_common": B_common,
        "n_rare_atoms": len(rare_atoms),
        "n_common_atoms": len(common_atoms),
        "avg_rep_selected": float(np.mean(rep_score[selected])) if len(selected) > 0 else None,
    }
    return selected, debug
