# src/dict_knowledge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Optional, Tuple

import numpy as np
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning, PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class DictKnowledgeConfig:
    # Dictionary size
    n_components: int = 64

    # Sparsity/regularization (used mainly for Lasso-based transforms)
    alpha: float = 1.0

    # Regularization type: "l1" (Lasso, sparse) or "l2" (Ridge, less sparse)
    regularization_type: str = "l2"

    # Iterations
    max_iter: int = 100  # Reduced from 300 for speed; usually converges much earlier

    # Repro
    random_state: int = 0

    # Activation extraction from codes
    tau: float = 1e-3
    top_k: int = 8  # recommended; also used to configure OMP

    # Preprocessing
    standardize: bool = True

    # === Speed knobs (NEW) ===
    # Use minibatch by default (MUCH faster on large N,D)
    use_minibatch: bool = True
    batch_size: int = 512

    # Optional dimensionality reduction BEFORE dict learning (huge win for 5120-dim embeddings)
    # Set to e.g. 256/512/1024. None disables PCA.
    pca_dim: Optional[int] = 512
    pca_randomized: bool = True

    # Transform algorithm: for speed, OMP is usually best when you want fixed sparsity (top_k)
    # Options: "omp", "lasso_cd", "ridge" (L2 regularization, less sparse)
    transform_algorithm: str = "omp"

    # Fit algorithm for full-batch DictionaryLearning (if use_minibatch=False)
    # Options: "cd", "lars"
    fit_algorithm: str = "cd"

    # Parallelism where sklearn supports it
    n_jobs: int = -1


@dataclass
class DictKnowledgeModel:
    config: DictKnowledgeConfig
    scaler: Optional[StandardScaler]
    pca: Optional[PCA]
    dict_learner: object  # DictionaryLearning or MiniBatchDictionaryLearning
    # sklearn stores dictionary as (K, D_reduced)
    components_: np.ndarray  # (K, D_reduced)
    ridge_transform: Optional[object] = None  # Ridge regression for L2 transform (if used)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X

    def transform_codes(self, X: np.ndarray) -> np.ndarray:
        """
        Compute sparse codes R for inputs X. Uses the same preprocessing
        (standardization + optional PCA) as training.
        """
        Xp = self._preprocess(X)
        
        # Use Ridge transform if configured (L2 regularization)
        if self.ridge_transform is not None:
            R = self.ridge_transform.predict(Xp)  # (N, K)
        else:
            # Use standard dictionary learning transform (L1 or OMP)
            R = self.dict_learner.transform(Xp)  # (N, K)
        return R

    def active_atoms(
        self,
        R: np.ndarray,
        top_k: Optional[int] = None,
        tau: Optional[float] = None,
    ) -> List[Set[int]]:
        """
        Convert sparse codes R into per-example activated atom sets.
        Uses top-k by |coeff| by default (stable).
        """
        top_k = self.config.top_k if top_k is None else top_k
        tau = self.config.tau if tau is None else tau

        R = np.asarray(R)
        N, K = R.shape

        if top_k is not None and top_k > 0:
            kk = min(int(top_k), K)
            absR = np.abs(R)

            # argpartition: O(K) per row rather than full sort
            idx = np.argpartition(-absR, kth=kk - 1, axis=1)[:, :kk]

            active: List[Set[int]] = []
            for i in range(N):
                js = idx[i]
                # filter weak coeffs (tau) as a safety check
                mask = absR[i, js] > tau
                active.append(set(int(j) for j in js[mask]))
            return active

        # fallback: pure thresholding (can be slower if codes are dense)
        active = []
        for i in range(N):
            js = np.where(np.abs(R[i]) > tau)[0]
            active.append(set(int(j) for j in js))
        return active


def fit_dictionary_knowledge(
    X: np.ndarray,
    config: DictKnowledgeConfig = DictKnowledgeConfig(),
) -> DictKnowledgeModel:
    """
    Fit dictionary learning on embedding matrix X (N, D).

    Key speed improvements vs your old version:
      - MiniBatchDictionaryLearning by default
      - OMP transform with fixed top_k sparsity by default
      - Optional PCA to reduce 5120-d embeddings before fitting
      - Avoid lars/lasso_lars (extremely slow at this scale)
    """
    X = np.asarray(X, dtype=np.float32)

    # 1) Standardize (often helps dictionary learning a lot)
    scaler = None
    if config.standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X

    # 2) Optional PCA (biggest speed lever for 5120-d embeddings)
    pca = None
    if config.pca_dim is not None and config.pca_dim > 0 and config.pca_dim < X_fit.shape[1]:
        svd_solver = "randomized" if config.pca_randomized else "full"
        pca = PCA(
            n_components=int(config.pca_dim),
            svd_solver=svd_solver,
            random_state=config.random_state,
        )
        X_fit = pca.fit_transform(X_fit)

    # 3) Choose transform algorithm + sparsity control
    # If you want fixed top_k nonzeros, OMP is usually MUCH faster than lasso-based transforms.
    transform_algorithm = config.transform_algorithm
    ridge_transform = None
    
    dl_kwargs = dict(
        n_components=config.n_components,
        max_iter=config.max_iter,
        random_state=config.random_state,
        tol=1e-4,  # Early stopping tolerance (converges faster)
    )

    if transform_algorithm == "omp":
        # Fixed sparsity
        dl_kwargs.update(
            transform_algorithm="omp",
            transform_n_nonzero_coefs=int(max(1, config.top_k)),
        )
        # alpha is ignored for OMP transform; keep it in config for compatibility
    elif transform_algorithm == "lasso_cd":
        # L1 sparsity via coordinate descent (faster than lars)
        dl_kwargs.update(
            transform_algorithm="lasso_cd",
            transform_alpha=float(config.alpha),
        )
    elif transform_algorithm == "ridge":
        # L2 regularization (less sparse, smoother codes)
        # Use OMP for dictionary fitting, but we'll use Ridge for transform
        dl_kwargs.update(
            transform_algorithm="omp",  # Use OMP for fitting, but we'll override transform
            transform_n_nonzero_coefs=int(max(1, config.top_k)),  # Not used if we use Ridge
        )
        # Ridge transform will be fitted after dictionary learning
    else:
        raise ValueError(f"Unknown transform_algorithm={transform_algorithm}. Use 'omp', 'lasso_cd', or 'ridge'.")

    # 4) Fit dictionary
    if config.use_minibatch:
        dict_learner = MiniBatchDictionaryLearning(
            **dl_kwargs,
            batch_size=int(config.batch_size),
            n_jobs=int(config.n_jobs),
        )
    else:
        dict_learner = DictionaryLearning(
            **dl_kwargs,
            alpha=float(config.alpha),          # used for fitting
            fit_algorithm=str(config.fit_algorithm),  # "cd" is usually faster than "lars"
        )

    dict_learner.fit(X_fit)

    # If using Ridge transform (L2), fit Ridge regression for each atom
    if transform_algorithm == "ridge":
        # Fit Ridge regression: for each sample, predict coefficients using dictionary
        # D is the dictionary (n_components, n_features)
        D = dict_learner.components_  # (K, D_reduced)
        
        # For each sample x, solve: x ≈ D @ r, where r is the coefficient vector
        # Using Ridge: r = (D^T @ D + alpha * I)^(-1) @ D^T @ x
        # For dictionary learning with Ridge, we want:
        # For each x, minimize ||x - D @ r||^2 + alpha * ||r||^2
        # This is solved by: r = (D^T @ D + alpha * I)^(-1) @ D^T @ x
        
        class RidgeDictTransform:
            def __init__(self, dictionary, alpha):
                self.D = dictionary  # (K, D_reduced)
                self.alpha = alpha
                # Precompute (D^T @ D + alpha * I)^(-1) @ D^T for efficiency
                DTD = self.D @ self.D.T  # (K, K)
                I = np.eye(DTD.shape[0])
                self.coef = np.linalg.solve(DTD + alpha * I, self.D)  # (K, D_reduced)
            
            def predict(self, X):
                # For each x in X, compute r = coef @ x
                # X: (N, D_reduced), coef: (K, D_reduced)
                # Returns: (N, K)
                return X @ self.coef.T
        
        ridge_transform = RidgeDictTransform(D, config.alpha)
    else:
        ridge_transform = None

    return DictKnowledgeModel(
        config=config,
        scaler=scaler,
        pca=pca,
        dict_learner=dict_learner,
        components_=dict_learner.components_.copy(),
        ridge_transform=ridge_transform,
    )


def atom_frequencies(active_sets: List[Set[int]], n_atoms: int) -> np.ndarray:
    """
    freq[j] = number of examples that activate atom j
    """
    freq = np.zeros(int(n_atoms), dtype=np.int64)
    for s in active_sets:
        for j in s:
            if 0 <= j < n_atoms:
                freq[j] += 1
    return freq
