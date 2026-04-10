# src/main_validate_coverage.py
"""
Validate hypothesis: subsets with higher coverage ratio (unseen knowledge coverage)
should have better ICL performance.

This script:
1. Randomly samples 10 different subsets for ICL demonstrations
2. Computes coverage ratio for each subset (from low to high)
3. Evaluates ICL performance for each subset
4. Plots coverage ratio vs performance to show positive correlation
"""

import os
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr, spearmanr

from data_utils import (
    load_banking77, load_banking77_label_names,
    load_bbh_task, list_available_bbh_tasks
)
from embed_and_cluster import get_or_compute_embeddings
from embed_and_cluster import _make_cache_path
from selection import _sgt_score_for_indices
from icl_eval import run_icl_eval
from dictionary_embed import DictionaryEmbedder


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "coverage_validation_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )
    logging.info(f"Logging to {log_file}")


def compute_coverage_ratio(
    cluster_ids: np.ndarray,
    subset_indices: List[int],
    total_clusters: int = None,
    t: int = 20,
    bin_size: int = 20,
    smooth_count: bool = False,
    offset: float = 1.0,
) -> float:
    """
    Compute coverage ratio for a subset using SGT-based score.
    
    Returns a value between 0 and 1, where:
    - Higher values indicate better coverage of clusters
    - Uses the same scoring function as greedy_sgt_selection
    """
    return _sgt_score_for_indices(
        cluster_ids=cluster_ids,
        indices=subset_indices,
        t=t,
        bin_size=bin_size,
        smooth_count=smooth_count,
        offset=offset,
        total_clusters=total_clusters,
    )


def plot_coverage_vs_performance(
    coverage_ratios: List[float],
    accuracies: List[float],
    output_path: Path,
    correlation_pearson: float,
    correlation_spearman: float,
):
    """
    Plot coverage ratio vs ICL accuracy with correlation statistics.
    """
    # Sort by coverage ratio for better visualization
    sorted_pairs = sorted(zip(coverage_ratios, accuracies))
    sorted_coverage, sorted_accuracy = zip(*sorted_pairs)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(sorted_coverage, sorted_accuracy, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(sorted_coverage, sorted_accuracy, 1)
    p = np.poly1d(z)
    plt.plot(sorted_coverage, p(sorted_coverage), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
    
    plt.xlabel("Coverage Ratio (SGT-based)", fontsize=12)
    plt.ylabel("ICL Accuracy", fontsize=12)
    plt.title(f"Coverage Ratio vs ICL Performance\n"
              f"Pearson r={correlation_pearson:.3f}, Spearman ρ={correlation_spearman:.3f}", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Plot saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate coverage-performance hypothesis")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="Model name for embeddings and ICL evaluation",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=256,
        help="Number of dictionary components",
    )
    parser.add_argument(
        "--dict_alpha",
        type=float,
        default=0.001,  # Reduced from 0.01 to avoid all-zero codes
        help="Dictionary learning alpha parameter (lower = less sparse, higher = more sparse)",
    )
    parser.add_argument(
        "--dict_max_iter",
        type=int,
        default=100,
        help="Maximum iterations for dictionary learning",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Budget (size) for each subset",
    )
    parser.add_argument(
        "--n_subsets",
        type=int,
        default=10,
        help="Number of random subsets to sample",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of test examples for ICL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="kmeans",  # Changed default to kmeans (more robust when codes are sparse)
        choices=["argmax", "kmeans"],
        help="Clustering method: 'kmeans' is more robust when codes are sparse",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="banking77",
        choices=["banking77", "bbh"],
        help="Dataset type",
    )
    parser.add_argument(
        "--bbh_task_name",
        type=str,
        default=None,
        help="BBH task name (required if dataset_type='bbh')",
    )
    parser.add_argument(
        "--icl_batch_size",
        type=int,
        default=8,
        help="Batch size for ICL evaluation",
    )
    parser.add_argument(
        "--skip_dict_learning",
        action="store_true",
        help="Skip dictionary learning and use embeddings directly for clustering (ablation)",
    )
    parser.add_argument(
        "--transform_algorithm",
        type=str,
        default=None,
        choices=[None, "lasso_lars", "lasso_cd", "omp"],
        help="Transform algorithm for dictionary learning. None = auto-select",
    )
    parser.add_argument(
        "--n_nonzero_coefs",
        type=int,
        default=None,
        help="Number of non-zero coefficients for OMP algorithm",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "test_all_algorithms", "test_alpha_range"],
        help="Ablation mode: 'test_all_algorithms' tests all transform algorithms, 'test_alpha_range' tests multiple alpha values",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    seed = args.seed
    model_name = args.model_name
    n_components = args.n_components
    dict_alpha = args.dict_alpha
    dict_max_iter = args.dict_max_iter
    cluster_method = args.cluster_method
    dataset_type = args.dataset_type
    bbh_task_name = args.bbh_task_name
    budget = args.budget
    n_subsets = args.n_subsets
    test_size = args.test_size
    max_length = 256
    batch_size = 16
    icl_batch_size = args.icl_batch_size
    max_new_tokens = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    
    if dataset_type == "bbh":
        if bbh_task_name is None:
            raise ValueError("--bbh_task_name is required when --dataset_type=bbh")
        output_dir_name = f"validate_coverage_{dataset_type}_{bbh_task_name}_{model_short}_{timestamp}"
    else:
        output_dir_name = f"validate_coverage_{dataset_type}_{model_short}_{timestamp}"
    
    output_dir = Path("outputs") / output_dir_name
    setup_logging(output_dir)

    logging.info("="*60)
    logging.info("Coverage-Performance Validation Study")
    logging.info("="*60)
    logging.info(f"Model: {model_name}")
    logging.info(f"Dataset type: {dataset_type}")
    logging.info(f"Budget: {budget}, Number of subsets: {n_subsets}")
    logging.info(f"Test size: {test_size}, Device: {device}")

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    if dataset_type == "banking77":
        dataset_name = "banking77_train"
        texts, labels = load_banking77("train")
        label_names = load_banking77_label_names()
        logging.info(f"Loaded {len(texts)} Banking77 train examples with {len(label_names)} intents.")
        labels = np.array(labels)
        bbh_targets = None
    elif dataset_type == "bbh":
        if bbh_task_name is None:
            raise ValueError("--bbh_task_name is required when --dataset_type=bbh")
        dataset_name = f"bbh_{bbh_task_name}"
        inputs, targets = load_bbh_task(bbh_task_name)
        logging.info(f"Loaded {len(inputs)} examples from BBH task: {bbh_task_name}")
        texts = inputs
        labels = None
        label_names = None
        bbh_targets = np.array(targets)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Check cache first to avoid loading model twice
    cache_root = Path("outputs") / "LLM_embeddings"
    cache_path = _make_cache_path(
        cache_root=cache_root,
        dataset_name=dataset_name,
        model_name=model_name,
        layer=-1,
        pooling="mean",
        l2_normalize=True,
    )
    
    # Load embeddings from cache if available
    if cache_path.exists():
        logging.info(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
        logging.info(f"Loaded embeddings shape: {embeddings.shape}")
        need_compute_embeddings = False
    else:
        logging.info("Cache not found, will compute embeddings after loading model")
        need_compute_embeddings = True
        embeddings = None
    
    # Load model and tokenizer ONCE (will be reused for embeddings if needed and ICL evaluation)
    logging.info("\nLoading model (will be reused for embeddings and ICL evaluation)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        output_hidden_states=True,  # Needed for embeddings
    )
    model.eval()
    
    # Compute embeddings if cache was missing
    if need_compute_embeddings:
        logging.info("Computing embeddings using loaded model...")
        from tqdm import tqdm
        
        device = next(model.parameters()).device
        layer_idx = -1  # Last layer
        all_embs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch = texts[i:i+batch_size]
            tokenized = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**tokenized)
                # Use last layer (-1) for embeddings
                hidden = outputs.hidden_states[layer_idx]
                emb = hidden.mean(dim=1)  # Mean pooling
                all_embs.append(emb.cpu().numpy())
        
        if len(all_embs) == 0:
            raise ValueError("No embeddings computed!")
        embeddings = np.vstack(all_embs)
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        
        logging.info(f"Computed embeddings shape: {embeddings.shape}")
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logging.info(f"Saved embeddings to cache: {cache_path}")
    
    logging.info(f"Embeddings shape: {embeddings.shape}")
    
    # Validate embeddings are not all zeros
    if np.all(embeddings == 0) or np.allclose(embeddings, 0, atol=1e-10):
        logging.error("="*60)
        logging.error("CRITICAL: Embeddings are all zeros!")
        logging.error("="*60)
        logging.error("This indicates a problem with embedding computation or corrupted cache.")
        logging.error(f"Embeddings stats: min={embeddings.min():.6f}, max={embeddings.max():.6f}, "
                    f"mean={embeddings.mean():.6f}, std={embeddings.std():.6f}")
        
        # Try to find and delete the corrupted cache file
        cache_path = _make_cache_path(
            cache_root=cache_root,
            dataset_name=dataset_name,
            model_name=model_name,
            layer=-1,
            pooling="mean",
            l2_normalize=True,
        )
        if cache_path.exists():
            logging.error(f"Corrupted cache file found: {cache_path}")
            logging.error("Deleting corrupted cache file. Please rerun to recompute embeddings.")
            cache_path.unlink()
        
        raise RuntimeError("Embeddings are all zeros. This is likely due to corrupted cache or embedding computation bug. "
                         f"Deleted cache file at {cache_path}. Please rerun the script to recompute embeddings.")
    
    # Additional validation: check if embeddings have reasonable variance
    emb_variance = np.var(embeddings, axis=0)
    n_zero_variance_dims = np.sum(emb_variance < 1e-10)
    if n_zero_variance_dims == embeddings.shape[1]:
        logging.error("="*60)
        logging.error("CRITICAL: All embedding dimensions have zero variance!")
        logging.error("="*60)
        logging.error("This means all samples have identical embeddings.")
        raise RuntimeError("All embedding dimensions have zero variance - all samples are identical!")
    elif n_zero_variance_dims > 0:
        logging.warning(f"Warning: {n_zero_variance_dims}/{embeddings.shape[1]} embedding dimensions have zero variance")

    # Dictionary learning (or skip for ablation)
    if args.skip_dict_learning:
        logging.info("="*60)
        logging.info("ABLATION: Skipping dictionary learning, using embeddings directly")
        logging.info("="*60)
        codes = embeddings
        dict_embedder = None
        logging.info(f"Using embeddings directly for clustering (shape: {codes.shape})")
        # Force kmeans when using embeddings directly
        if cluster_method == "argmax":
            logging.warning(f"argmax not applicable to raw embeddings, switching to kmeans")
            cluster_method = "kmeans"
    elif args.ablation_mode == "test_all_algorithms":
        logging.info("="*60)
        logging.info("ABLATION: Testing all transform algorithms")
        logging.info("="*60)
        algorithms = ["lasso_lars", "lasso_cd", "omp"]
        best_codes = None
        best_algorithm = None
        best_stats = None
        
        for alg in algorithms:
            logging.info(f"\nTesting algorithm: {alg}")
            try:
                dict_embedder_test = DictionaryEmbedder(
                    n_components=n_components,
                    alpha=dict_alpha if alg != "omp" else 1.0,  # OMP doesn't use alpha
                    max_iter=dict_max_iter,
                    transform_algorithm=alg,
                    random_state=seed,
                    n_nonzero_coefs=args.n_nonzero_coefs,
                )
                codes_test = dict_embedder_test.fit_transform(embeddings)
                
                # Compute diagnostics
                n_nonzero_per_sample = np.count_nonzero(codes_test, axis=1)
                n_nonzero_per_atom = np.count_nonzero(codes_test, axis=0)
                max_per_sample = np.argmax(codes_test, axis=1)
                atoms_used_as_max = len(np.unique(max_per_sample))
                code_variance = np.var(codes_test, axis=0)
                non_zero_variance = np.sum(code_variance > 1e-10)
                
                stats = {
                    "mean_nonzero_per_sample": n_nonzero_per_sample.mean(),
                    "atoms_with_nonzero": np.count_nonzero(n_nonzero_per_atom),
                    "atoms_used_as_max": atoms_used_as_max,
                    "non_zero_variance_dims": non_zero_variance,
                    "mean_code_variance": code_variance.mean(),
                }
                
                logging.info(f"  Mean non-zero coefficients per sample: {stats['mean_nonzero_per_sample']:.2f}")
                logging.info(f"  Atoms with non-zero coefficients: {stats['atoms_with_nonzero']}/{n_components}")
                logging.info(f"  Atoms used as argmax (clusters): {stats['atoms_used_as_max']}/{n_components}")
                logging.info(f"  Dimensions with variance > 1e-10: {stats['non_zero_variance_dims']}/{n_components}")
                logging.info(f"  Mean code variance: {stats['mean_code_variance']:.6f}")
                
                # Check if this is better (more atoms used, more variance)
                if best_codes is None or (stats['atoms_used_as_max'] > best_stats['atoms_used_as_max'] and 
                                         stats['non_zero_variance_dims'] > best_stats['non_zero_variance_dims']):
                    best_codes = codes_test
                    best_algorithm = alg
                    best_stats = stats
                    dict_embedder = dict_embedder_test
                    
            except Exception as e:
                logging.warning(f"  Algorithm {alg} failed: {e}")
                continue
        
        if best_codes is None:
            logging.error("All algorithms failed! Falling back to embeddings directly.")
            codes = embeddings
            dict_embedder = None
            cluster_method = "kmeans"
        else:
            codes = best_codes
            logging.info(f"\nBest algorithm: {best_algorithm}")
            logging.info(f"Using codes from {best_algorithm} for clustering")
            
    elif args.ablation_mode == "test_alpha_range":
        logging.info("="*60)
        logging.info("ABLATION: Testing multiple alpha values")
        logging.info("="*60)
        alpha_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
        best_codes = None
        best_alpha = None
        best_stats = None
        
        for alpha_test in alpha_values:
            logging.info(f"\nTesting alpha: {alpha_test}")
            try:
                dict_embedder_test = DictionaryEmbedder(
                    n_components=n_components,
                    alpha=alpha_test,
                    max_iter=dict_max_iter,
                    transform_algorithm=args.transform_algorithm,
                    random_state=seed,
                )
                codes_test = dict_embedder_test.fit_transform(embeddings)
                
                # Compute diagnostics
                n_nonzero_per_sample = np.count_nonzero(codes_test, axis=1)
                n_nonzero_per_atom = np.count_nonzero(codes_test, axis=0)
                max_per_sample = np.argmax(codes_test, axis=1)
                atoms_used_as_max = len(np.unique(max_per_sample))
                code_variance = np.var(codes_test, axis=0)
                non_zero_variance = np.sum(code_variance > 1e-10)
                
                stats = {
                    "mean_nonzero_per_sample": n_nonzero_per_sample.mean(),
                    "atoms_with_nonzero": np.count_nonzero(n_nonzero_per_atom),
                    "atoms_used_as_max": atoms_used_as_max,
                    "non_zero_variance_dims": non_zero_variance,
                    "mean_code_variance": code_variance.mean(),
                }
                
                logging.info(f"  Mean non-zero coefficients per sample: {stats['mean_nonzero_per_sample']:.2f}")
                logging.info(f"  Atoms with non-zero coefficients: {stats['atoms_with_nonzero']}/{n_components}")
                logging.info(f"  Atoms used as argmax (clusters): {stats['atoms_used_as_max']}/{n_components}")
                logging.info(f"  Dimensions with variance > 1e-10: {stats['non_zero_variance_dims']}/{n_components}")
                
                # Check if this is better
                if best_codes is None or (stats['atoms_used_as_max'] > best_stats['atoms_used_as_max'] and 
                                         stats['non_zero_variance_dims'] > best_stats['non_zero_variance_dims']):
                    best_codes = codes_test
                    best_alpha = alpha_test
                    best_stats = stats
                    dict_embedder = dict_embedder_test
                    
            except Exception as e:
                logging.warning(f"  Alpha {alpha_test} failed: {e}")
                continue
        
        if best_codes is None:
            logging.error("All alpha values failed! Falling back to embeddings directly.")
            codes = embeddings
            dict_embedder = None
            cluster_method = "kmeans"
        else:
            codes = best_codes
            logging.info(f"\nBest alpha: {best_alpha}")
            logging.info(f"Using codes from alpha={best_alpha} for clustering")
    else:
        # Normal dictionary learning
        logging.info("Fitting dictionary on embeddings...")
        dict_embedder = DictionaryEmbedder(
            n_components=n_components,
            alpha=dict_alpha,
            max_iter=dict_max_iter,
            transform_algorithm=args.transform_algorithm,  # Can be None for auto-select
            random_state=seed,
            n_nonzero_coefs=args.n_nonzero_coefs,
        )
        logging.info(f"Using transform algorithm: {dict_embedder.transform_algorithm} (alpha={dict_alpha})")
        codes = dict_embedder.fit_transform(embeddings)
        logging.info(f"Codes shape: {codes.shape}")
    
    # Diagnostic: Check code sparsity (or embedding statistics if skipping dict learning)
    atoms_used_as_max = None  # Initialize for later checks
    if args.skip_dict_learning:
        logging.info(f"Embedding statistics (using embeddings directly):")
        logging.info(f"  Embedding shape: {codes.shape}")
        logging.info(f"  Embedding mean: {codes.mean():.6f}, std: {codes.std():.6f}")
        logging.info(f"  Embedding variance per dimension: min={np.var(codes, axis=0).min():.6f}, "
                    f"max={np.var(codes, axis=0).max():.6f}, mean={np.var(codes, axis=0).mean():.6f}")
        # Check for constant dimensions
        code_variance = np.var(codes, axis=0)
        constant_dims = np.sum(code_variance < 1e-10)
        logging.info(f"  Dimensions with variance < 1e-10 (constant): {constant_dims}/{codes.shape[1]}")
    else:
        n_nonzero_per_sample = np.count_nonzero(codes, axis=1)
        n_nonzero_per_atom = np.count_nonzero(codes, axis=0)
        max_per_sample = np.argmax(codes, axis=1)
        atoms_used_as_max = np.unique(max_per_sample)
        code_variance = np.var(codes, axis=0)
        constant_dims = np.sum(code_variance < 1e-10)
        
        logging.info(f"Code sparsity diagnostics:")
        logging.info(f"  Mean non-zero coefficients per sample: {n_nonzero_per_sample.mean():.2f} "
                    f"(min={n_nonzero_per_sample.min()}, max={n_nonzero_per_sample.max()})")
        logging.info(f"  Atoms with non-zero coefficients: {np.count_nonzero(n_nonzero_per_atom)}/{n_components}")
        logging.info(f"  Atoms used as argmax (clusters): {len(atoms_used_as_max)}/{n_components}")
        logging.info(f"  Code variance per dimension: min={code_variance.min():.6f}, "
                    f"max={code_variance.max():.6f}, mean={code_variance.mean():.6f}")
        logging.info(f"  Dimensions with variance < 1e-10 (constant): {constant_dims}/{n_components}")
        
        # Detailed analysis of why we might get one cluster
        if len(atoms_used_as_max) == 1:
            logging.warning("="*60)
            logging.warning("DIAGNOSIS: Only 1 cluster from argmax!")
            logging.warning("="*60)
            logging.warning(f"All samples have argmax = {atoms_used_as_max[0]}")
            logging.warning(f"This means all samples have their maximum coefficient at atom {atoms_used_as_max[0]}")
            logging.warning(f"Atom {atoms_used_as_max[0]} statistics:")
            atom_codes = codes[:, atoms_used_as_max[0]]
            logging.warning(f"  Mean coefficient: {atom_codes.mean():.6f}")
            logging.warning(f"  Max coefficient: {atom_codes.max():.6f}")
            logging.warning(f"  Min coefficient: {atom_codes.min():.6f}")
            logging.warning(f"  Std coefficient: {atom_codes.std():.6f}")
            
            # Check other atoms
            other_atoms = [i for i in range(n_components) if i != atoms_used_as_max[0]]
            if len(other_atoms) > 0:
                other_max = codes[:, other_atoms].max(axis=1)
                logging.warning(f"  Max coefficient in other atoms: mean={other_max.mean():.6f}, "
                              f"max={other_max.max():.6f}")
                logging.warning(f"  Samples where max atom > other max: "
                              f"{np.sum(atom_codes > other_max)}/{len(codes)}")
    
    # Check if all codes are zero (critical error) - only if we did dictionary learning
    if not args.skip_dict_learning and np.all(codes == 0):
        logging.error(f"CRITICAL: All codes are zero! Dictionary learning failed.")
        logging.error(f"This usually means alpha ({dict_alpha}) is too high relative to embedding scale.")
        logging.error(f"Consider using --skip_dict_learning or --ablation_mode test_all_algorithms")
        raise RuntimeError("Dictionary learning produced all-zero codes. Use ablation modes to diagnose.")
    
    # Check for extreme sparsity BEFORE clustering
    if not args.skip_dict_learning and atoms_used_as_max is not None:
        if len(atoms_used_as_max) < 5:
            logging.warning(f"Only {len(atoms_used_as_max)} unique clusters from argmax! "
                          f"This suggests codes are too sparse. Consider:")
            logging.warning(f"  1. Using kmeans clustering instead of argmax (--cluster_method kmeans)")
            logging.warning(f"  2. Reducing dictionary alpha (current: {dict_alpha})")
            logging.warning(f"  3. Increasing max_iter (current: {dict_max_iter})")
            logging.warning(f"  4. Using --skip_dict_learning to use embeddings directly")
            logging.warning(f"  5. Using --ablation_mode test_all_algorithms to find best algorithm")
            
            # If argmax would produce too few clusters, force kmeans
            if cluster_method == "argmax" and len(atoms_used_as_max) == 1:
                logging.warning(f"CRITICAL: argmax would produce only 1 cluster! Forcing kmeans clustering.")
                cluster_method = "kmeans"

    # Clustering
    logging.info(f"Assigning points to clusters using method: {cluster_method}")
    cluster_ids = None
    
    if cluster_method == "argmax":
        cluster_ids = np.argmax(codes, axis=1)
        total_clusters = len(np.unique(cluster_ids))
        logging.info(f"Dictionary-based clustering (argmax) done. #clusters={total_clusters}")
        
        # Validate that we got more than 1 cluster
        if total_clusters == 1:
            logging.error(f"CRITICAL: argmax clustering produced only 1 cluster!")
            logging.error(f"This means all samples have the same maximum coefficient atom.")
            logging.error(f"Switching to kmeans as fallback...")
            cluster_method = "kmeans"
            cluster_ids = None  # Reset to force kmeans
    
    if cluster_method == "kmeans" or cluster_ids is None:
        from sklearn.cluster import KMeans
        
        # Determine number of clusters and input data
        if args.skip_dict_learning:
            n_clusters_kmeans = min(50, len(embeddings) // 10, len(embeddings) - 1)
            input_data = codes  # codes is actually embeddings in this case
            logging.info(f"Running k-means on embeddings with {n_clusters_kmeans} clusters...")
        else:
            n_clusters_kmeans = min(n_components, 50)
            input_data = codes
            logging.info(f"Running k-means on codes with {n_clusters_kmeans} clusters...")
        
        # Check if input data has sufficient variance for kmeans
        data_variance = np.var(input_data, axis=0)
        non_zero_variance = np.sum(data_variance > 1e-10)
        data_dim = input_data.shape[1] if len(input_data.shape) > 1 else 1
        logging.info(f"Data variance check: {non_zero_variance}/{data_dim} dimensions have variance > 1e-10")
        
        if non_zero_variance == 0:
            logging.error(f"CRITICAL: All data dimensions have zero variance! Data is identical.")
            logging.error(f"This means all samples are identical, which is highly unusual.")
            logging.error(f"Checking sample-to-sample variance...")
            sample_variance = np.var(input_data, axis=1)
            logging.error(f"  Variance across features per sample: min={sample_variance.min():.6f}, "
                         f"max={sample_variance.max():.6f}, mean={sample_variance.mean():.6f}")
            if np.all(sample_variance < 1e-10):
                logging.error(f"  All samples are identical! This suggests a data loading or preprocessing issue.")
            raise RuntimeError("Cannot cluster: all dimensions have zero variance")
        
        # Remove constant dimensions before clustering
        if non_zero_variance < data_dim:
            logging.warning(f"Removing {data_dim - non_zero_variance} constant dimensions before clustering")
            valid_dims = data_variance > 1e-10
            input_data = input_data[:, valid_dims]
            logging.info(f"Using {input_data.shape[1]} dimensions with variance for clustering")
        
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=seed, n_init=10)
        cluster_ids = kmeans.fit_predict(input_data)
        total_clusters = len(np.unique(cluster_ids))
        logging.info(f"K-means clustering done. #clusters={total_clusters}")
        
        # Final validation
        if total_clusters == 1:
            logging.error(f"CRITICAL: kmeans produced only 1 cluster!")
            logging.error(f"This suggests the input data has insufficient variance for clustering.")
            logging.error(f"Diagnostics:")
            logging.error(f"  Input data shape: {input_data.shape}")
            logging.error(f"  Input data mean: {input_data.mean():.6f}, std: {input_data.std():.6f}")
            logging.error(f"  Input data variance: min={np.var(input_data, axis=0).min():.6f}, "
                         f"max={np.var(input_data, axis=0).max():.6f}")
            logging.error(f"  Number of clusters attempted: {n_clusters_kmeans}")
            logging.error(f"Consider:")
            logging.error(f"  1. Checking if embeddings are properly normalized")
            logging.error(f"  2. Using --skip_dict_learning to use raw embeddings")
            logging.error(f"  3. Checking dataset diversity")
            logging.error(f"  4. Reducing n_clusters_kmeans (current: {n_clusters_kmeans})")
    
    # Final cluster count
    total_clusters = len(np.unique(cluster_ids))
    logging.info(f"Dictionary learning + clustering: Total clusters created = {total_clusters}")
    
    # Log cluster size statistics
    unique_clusters, cluster_counts = np.unique(cluster_ids, return_counts=True)
    logging.info(f"Cluster size stats: min={cluster_counts.min()}, "
                f"max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")

    # Train / Test split
    all_indices = np.arange(len(texts))
    np.random.shuffle(all_indices)
    test_indices = all_indices[:test_size]
    pool_indices = all_indices[test_size:]

    logging.info(f"Test set size: {len(test_indices)}")
    logging.info(f"Pool size (for selection): {len(pool_indices)}")

    # Prepare test data
    if dataset_type == "banking77":
        test_texts = [texts[i] for i in test_indices]
        test_labels = [int(labels[i]) for i in test_indices]
        test_targets = None
    else:  # bbh
        test_inputs = [texts[i] for i in test_indices]
        test_targets = [bbh_targets[i] for i in test_indices]
        test_texts = test_inputs
        test_labels = None

    # Get pool cluster IDs
    pool_cluster_ids = cluster_ids[pool_indices]
    pool_total_clusters = len(np.unique(pool_cluster_ids))
    logging.info(f"Dictionary learning + SGT: Total clusters in train pool = {pool_total_clusters}")

    # Create mapping from absolute index to relative index in pool (for efficient lookup)
    pool_indices_list = list(pool_indices)  # Convert to list for .index() method
    pool_index_map = {idx: i for i, idx in enumerate(pool_indices_list)}

    # Sample n_subsets random subsets
    logging.info(f"\nSampling {n_subsets} random subsets of size {budget}...")
    if len(pool_indices) < budget:
        raise ValueError(f"Pool size ({len(pool_indices)}) is smaller than budget ({budget})")
    
    # Use different seeds for each subset to ensure diversity
    subset_seeds = [seed + i for i in range(n_subsets)]
    random_subsets = []
    for i, subset_seed in enumerate(subset_seeds):
        rng = random.Random(subset_seed)
        subset = rng.sample(pool_indices_list, budget)
        random_subsets.append(subset)
        logging.info(f"Subset {i+1}: sampled {len(subset)} indices")

    # Compute coverage ratio for each subset
    logging.info(f"\nComputing coverage ratios for each subset...")
    coverage_ratios = []
    for i, subset in enumerate(random_subsets):
        # Convert to relative indices within pool using the mapping
        relative_indices = [pool_index_map[idx] for idx in subset]
        coverage = compute_coverage_ratio(
            cluster_ids=pool_cluster_ids,
            subset_indices=relative_indices,
            total_clusters=pool_total_clusters,
            t=20,
            bin_size=20,
            smooth_count=False,
            offset=1.0,
        )
        coverage_ratios.append(coverage)
        logging.info(f"Subset {i+1}: coverage ratio = {coverage:.4f}")

    # Sort subsets by coverage ratio (low to high)
    sorted_indices = sorted(range(n_subsets), key=lambda i: coverage_ratios[i])
    logging.info(f"\nSubsets sorted by coverage ratio (low to high):")
    for idx in sorted_indices:
        logging.info(f"  Subset {idx+1}: coverage = {coverage_ratios[idx]:.4f}")

    # Model and tokenizer are already loaded above (reused for ICL evaluation)
    logging.info("\nUsing already-loaded model for ICL evaluation...")

    # Evaluate ICL performance for each subset
    logging.info(f"\nEvaluating ICL performance for each subset...")
    accuracies = []
    
    for i, subset in enumerate(random_subsets):
        logging.info(f"\nEvaluating subset {i+1}/{n_subsets} (coverage={coverage_ratios[i]:.4f})...")
        
        # Prepare training data
        if dataset_type == "banking77":
            train_texts = [texts[idx] for idx in subset]
            train_labels = [int(labels[idx]) for idx in subset]
            acc = run_icl_eval(
                model_name=model_name,
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                label_names=label_names,
                dataset_type="banking77",
                max_new_tokens=max_new_tokens,
                device=device,
                model=model,
                tokenizer=tokenizer,
                batch_size=icl_batch_size,
            )
        else:  # bbh
            train_inputs = [texts[idx] for idx in subset]
            train_targets = [bbh_targets[idx] for idx in subset]
            acc = run_icl_eval(
                model_name=model_name,
                train_inputs=train_inputs,
                train_targets=train_targets,
                test_inputs=test_inputs,
                test_targets=test_targets,
                dataset_type="bbh",
                max_new_tokens=max_new_tokens,
                device=device,
                model=model,
                tokenizer=tokenizer,
                batch_size=icl_batch_size,
            )
        
        accuracies.append(acc)
        logging.info(f"Subset {i+1}: accuracy = {acc:.4f}, coverage = {coverage_ratios[i]:.4f}")
        
        # Clear cache to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute correlations
    pearson_r, pearson_p = pearsonr(coverage_ratios, accuracies)
    spearman_rho, spearman_p = spearmanr(coverage_ratios, accuracies)
    
    logging.info("\n" + "="*60)
    logging.info("RESULTS")
    logging.info("="*60)
    logging.info(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    logging.info(f"Spearman correlation: ρ = {spearman_rho:.4f}, p = {spearman_p:.4f}")
    
    # Print sorted results
    logging.info("\nSubsets sorted by coverage ratio:")
    for idx in sorted_indices:
        logging.info(f"  Subset {idx+1}: coverage={coverage_ratios[idx]:.4f}, accuracy={accuracies[idx]:.4f}")

    # Create plot
    plot_coverage_vs_performance(
        coverage_ratios=coverage_ratios,
        accuracies=accuracies,
        output_path=output_dir / "coverage_vs_performance.png",
        correlation_pearson=pearson_r,
        correlation_spearman=spearman_rho,
    )

    # Save results
    results = {
        "model_name": model_name,
        "dataset_type": dataset_type,
        "budget": budget,
        "n_subsets": n_subsets,
        "test_size": test_size,
        "seed": seed,
        "total_clusters": total_clusters,
        "pool_total_clusters": pool_total_clusters,
        "ablation_config": {
            "skip_dict_learning": args.skip_dict_learning,
            "ablation_mode": args.ablation_mode,
            "transform_algorithm": args.transform_algorithm,
            "dict_alpha": dict_alpha if not args.skip_dict_learning else None,
            "cluster_method": cluster_method,
        },
        "correlations": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
        },
        "subsets": [
            {
                "subset_id": i+1,
                "indices": [int(idx) for idx in subset],
                "coverage_ratio": float(coverage_ratios[i]),
                "accuracy": float(accuracies[i]),
            }
            for i, subset in enumerate(random_subsets)
        ],
    }
    
    if dataset_type == "bbh":
        results["bbh_task_name"] = bbh_task_name
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to {output_dir / 'results.json'}")
    logging.info("Validation study finished.")


if __name__ == "__main__":
    main()

