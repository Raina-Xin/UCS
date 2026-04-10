"""
Run clustering only (dict_dbscan) for hyperparameter tuning.
This script has the same setup as main.py but skips selection and evaluation steps.

Example:
  python run_clustering.py \
    --dataset_type banking77 \
    --embedding_model_name Qwen/Qwen2.5-7B-Instruct \
    --dict_n_components 64 \
    --dict_alpha 10.0 \
    --dict_top_k 4 \
    --dict_tau 1e-3 \
    --dict_transform_algorithm omp \
    --dict_regularization_type l2 \
    --dict_max_iter 50 \
    --dict_pca_dim 128 \
    --dict_batch_size 512 \
    --clustering dict_dbscan \
    --dbscan_k 20 \
    --dbscan_q 0.25 \
    --dbscan_min_samples 5 \
    --seed 42 \
    --n_runs 1
"""

import os
# Disable tokenizer parallelism to avoid warnings when sklearn forks processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import random
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

from data_utils import (
    load_banking77, load_banking77_label_names,
    load_clinc150, load_clinc150_label_names,
    load_hwu64, load_hwu64_label_names
)
from embed_and_cluster import run_dbscan_thresholded, get_or_compute_embeddings
from dict_knowledge import fit_dictionary_knowledge, DictKnowledgeConfig, atom_frequencies
from plot import plot_aggregate_cluster_statistics, plot_cluster_distribution


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "clustering_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run clustering only (dict_dbscan) for hyperparameter tuning. "
                    "Same setup as main.py but skips selection and evaluation."
    )

    # Core
    parser.add_argument("--dataset_type", type=str, default="banking77", choices=["banking77", "clinc150", "hwu64"])
    parser.add_argument("--embedding_model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        help="Causal LM for computing embeddings")
    parser.add_argument("--l2_normalize", action="store_true", default=False,
                        help="L2 normalize LLM embeddings (default: off)")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    # Clustering method
    parser.add_argument(
        "--clustering",
        type=str,
        default="dict_dbscan",
        choices=["dbscan", "dict_argmax", "dict_dbscan"],
        help="Clustering method (default: dict_dbscan)",
    )

    # dbscan knobs
    parser.add_argument("--dbscan_k", type=int, default=10)
    parser.add_argument("--robust_sgt_top_m", type=int, default=20)
    parser.add_argument("--dbscan_q", type=float, default=0.05)
    parser.add_argument("--dbscan_min_samples", type=int, default=1)

    # dict learning knobs
    parser.add_argument("--dict_n_components", type=int, default=64)
    parser.add_argument("--dict_alpha", type=float, default=1.0)
    parser.add_argument("--dict_top_k", type=int, default=4)
    parser.add_argument("--dict_tau", type=float, default=1e-3)

    parser.add_argument("--dict_standardize", action="store_true", default=False,
                        help="Standardize embeddings before dictionary learning (default: off)")
    parser.add_argument("--no_dict_standardize", dest="dict_standardize", action="store_false",
                        help="Disable standardization before dictionary learning")

    parser.add_argument("--dict_use_minibatch", action="store_true", default=True)
    parser.add_argument("--no_dict_use_minibatch", dest="dict_use_minibatch", action="store_false")
    parser.add_argument("--dict_batch_size", type=int, default=512)
    parser.add_argument("--dict_pca_dim", type=int, default=512,
                        help="PCA dim before dict learning (<=0 disables PCA)")
    parser.add_argument("--dict_transform_algorithm", type=str, default="ridge", choices=["omp", "lasso_cd", "ridge"],
                        help="Transform algorithm: 'omp' (fixed sparsity), 'lasso_cd' (L1, sparse), 'ridge' (L2, less sparse)")
    parser.add_argument("--dict_regularization_type", type=str, default="l2", choices=["l1", "l2"],
                        help="Regularization type for dictionary learning: 'l1' (sparse) or 'l2' (less sparse).")
    parser.add_argument("--dict_max_iter", type=int, default=100,
                        help="Max iterations for dictionary learning")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_type = args.dataset_type
    seed = args.seed
    embedding_model_name = args.embedding_model_name
    n_runs = args.n_runs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_short = embedding_model_name.replace("/", "_").replace("-", "_")
    output_dir = Path("outputs") / f"clustering_{dataset_type}_{embedding_short}_{args.clustering}_{timestamp}"
    setup_logging(output_dir)
    
    logging.info(
        f"Dataset: {dataset_type}, "
        f"Embedding Model: {embedding_model_name}, "
        f"L2 Normalize: {args.l2_normalize}, "
        f"N runs: {n_runs}, "
        f"Clustering: {args.clustering}"
    )

    # Load data - use actual train/test splits
    if dataset_type == "banking77":
        train_texts, train_labels = load_banking77("train")
        label_names = load_banking77_label_names()
        dataset_cache_name = "banking77_train_full"
    elif dataset_type == "clinc150":
        train_texts, train_labels = load_clinc150("train")
        label_names = load_clinc150_label_names()
        dataset_cache_name = "clinc150_train_full"
    elif dataset_type == "hwu64":
        train_texts, train_labels = load_hwu64("train")
        label_names = load_hwu64_label_names()
        dataset_cache_name = "hwu64_train_full"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    train_inputs = train_texts
    train_pool_size = len(train_texts)
    texts_for_embeddings = train_texts
    
    logging.info(f"Train pool: {train_pool_size} examples")

    # Compute or load cached embeddings
    cache_root = Path("outputs") / "LLM_embeddings"
    train_embeddings = get_or_compute_embeddings(
        texts=texts_for_embeddings,
        model_name=embedding_model_name,
        cache_root=cache_root,
        dataset_name=dataset_cache_name,
        layer=-1,
        batch_size=16,
        max_length=256,
        pooling="mean",
        l2_normalize=args.l2_normalize,
    )
    logging.info(f"Train embeddings shape: {train_embeddings.shape}")
    
    # Store results across runs
    cluster_counts_per_run = []
    all_cluster_ids = []  # Store cluster_ids for each run

    # Run multiple times with different random seeds
    for run_idx in range(n_runs):
        logging.info(f"\n{'='*60}")
        logging.info(f"Run {run_idx + 1}/{n_runs}")
        logging.info(f"{'='*60}")
        
        # Use different seed for each run
        run_seed = seed + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        
        logging.info(f"Run {run_idx + 1}: Train pool: {len(train_inputs)}")
        
        # -------------------------
        # Clustering → produce cluster_ids
        # -------------------------
        cluster_ids = None
        n_clusters = None
        
        # dict artifacts (only present for dict clustering)
        dict_active_sets = None
        dict_atom_freq = None
        dict_R = None
        
        if args.clustering == "dbscan":
            logging.info(f"Run {run_idx + 1}: Clustering = DBSCAN")
            cluster_ids, eps = run_dbscan_thresholded(
                train_embeddings,
                k=args.dbscan_k,
                q=args.dbscan_q,
                min_samples=args.dbscan_min_samples,
                verbose=True,
            )
        
            # map noise -1 to singleton clusters
            n_noise = int(np.sum(cluster_ids == -1))
            if n_noise > 0:
                max_cluster_id = int(np.max(cluster_ids[cluster_ids >= 0])) if np.any(cluster_ids >= 0) else -1
                cluster_ids = cluster_ids.copy()
                noise_indices = np.where(cluster_ids == -1)[0]
                for i, noise_idx in enumerate(noise_indices):
                    cluster_ids[noise_idx] = max_cluster_id + 1 + i
                logging.info(f"Assigned {n_noise} noise points to {n_noise} singleton clusters (IDs: {max_cluster_id + 1} to {max_cluster_id + n_noise})")
        
            if len(cluster_ids) != len(train_inputs):
                raise ValueError(f"Mismatch: cluster_ids={len(cluster_ids)} vs train_inputs={len(train_inputs)}")
        
            n_clusters = int(len(np.unique(cluster_ids)))
            logging.info(f"Run {run_idx + 1}: DBSCAN produced {n_clusters} clusters")
            plot_cluster_distribution(cluster_ids, output_dir, run_idx)
        
            if len(cluster_counts_per_run) <= run_idx:
                cluster_counts_per_run.append(n_clusters)
            else:
                cluster_counts_per_run[run_idx] = n_clusters
        
        elif args.clustering == "dict_argmax":
            logging.info(f"Run {run_idx + 1}: Clustering = Dictionary(argmax |R|)")
        
            dk_cfg = DictKnowledgeConfig(
                n_components=args.dict_n_components,
                alpha=args.dict_alpha,
                regularization_type=args.dict_regularization_type,
                top_k=args.dict_top_k,
                tau=args.dict_tau,
                standardize=args.dict_standardize,
                random_state=run_seed,
                use_minibatch=args.dict_use_minibatch,
                batch_size=args.dict_batch_size,
                pca_dim=(None if args.dict_pca_dim <= 0 else args.dict_pca_dim),
                transform_algorithm=args.dict_transform_algorithm,
                max_iter=args.dict_max_iter,
            )
            km = fit_dictionary_knowledge(train_embeddings, dk_cfg)
        
            dict_R = km.transform_codes(train_embeddings)
            dict_active_sets = km.active_atoms(dict_R)
            dict_atom_freq = atom_frequencies(dict_active_sets, n_atoms=dk_cfg.n_components)
        
            # Investigate dict code norm spread
            nr = np.linalg.norm(dict_R, axis=1)
            print("||R|| min/med/max:", nr.min(), np.median(nr), nr.max())
            if np.median(nr) > 0:
                max_med_ratio = nr.max() / np.median(nr)
                if max_med_ratio > 10:
                    logging.warning(
                        f"Large norm spread detected: max/median = {max_med_ratio:.2f}. "
                        f"Euclidean clustering may be unreliable without normalization."
                    )
        
            absR = np.abs(dict_R)
            cluster_ids = absR.argmax(axis=1).astype(np.int32)
        
            n_clusters = int(len(np.unique(cluster_ids)))
            logging.info(f"Run {run_idx + 1}: Dict clustering produced {n_clusters} argmax-atoms")
            plot_cluster_distribution(cluster_ids, output_dir, run_idx)
            if len(cluster_counts_per_run) <= run_idx:
                cluster_counts_per_run.append(n_clusters)
            else:
                cluster_counts_per_run[run_idx] = n_clusters
        
        elif args.clustering == "dict_dbscan":
            logging.info(f"Run {run_idx + 1}: Clustering = Dictionary Learning + DBSCAN")
            logging.info(f"Run {run_idx + 1}: Robust SGT top_m = {args.robust_sgt_top_m}")
        
            dk_cfg = DictKnowledgeConfig(
                n_components=args.dict_n_components,
                alpha=args.dict_alpha,
                regularization_type=args.dict_regularization_type,
                top_k=args.dict_top_k,
                tau=args.dict_tau,
                standardize=args.dict_standardize,
                random_state=run_seed,
                use_minibatch=args.dict_use_minibatch,
                batch_size=args.dict_batch_size,
                pca_dim=(None if args.dict_pca_dim <= 0 else args.dict_pca_dim),
                transform_algorithm=args.dict_transform_algorithm,
                max_iter=args.dict_max_iter,
            )
            km = fit_dictionary_knowledge(train_embeddings, dk_cfg)
        
            dict_R = km.transform_codes(train_embeddings)
            dict_active_sets = km.active_atoms(dict_R)
            dict_atom_freq = atom_frequencies(dict_active_sets, n_atoms=dk_cfg.n_components)
            
            # Investigate dict code norm spread (before normalization)
            nr = np.linalg.norm(dict_R, axis=1)
            print("||R|| min/med/max:", nr.min(), np.median(nr), nr.max())
            if np.median(nr) > 0:
                max_med_ratio = nr.max() / np.median(nr)
                if max_med_ratio > 10:
                    logging.warning(
                        f"Large norm spread detected: max/median = {max_med_ratio:.2f}. "
                        f"Euclidean clustering will be garbage unless normalized. Normalizing now..."
                    )
                else:
                    logging.info(f"Norm spread is reasonable: max/median = {max_med_ratio:.2f}")

            # sparsity
            nnz = np.sum(np.abs(dict_R) > 1e-6, axis=1)
            print("R nnz:", np.min(nnz), np.median(nnz), np.max(nnz))

            # IMPORTANT: normalize to prevent norm artifacts dominating distances
            dict_R = dict_R / (np.linalg.norm(dict_R, axis=1, keepdims=True) + 1e-12)
            
            # Verify normalization worked
            nr_normalized = np.linalg.norm(dict_R, axis=1)
            print("||R|| after normalization min/med/max:", nr_normalized.min(), np.median(nr_normalized), nr_normalized.max())

            cluster_ids, eps = run_dbscan_thresholded(
                dict_R,
                k=args.dbscan_k,
                q=args.dbscan_q,      # try 0.05–0.15 for sparse codes
                min_samples=args.dbscan_min_samples,
                verbose=True,
            )

            # map noise -1 to singleton clusters
            n_noise = int(np.sum(cluster_ids == -1))
            if n_noise > 0:
                max_cluster_id = int(np.max(cluster_ids[cluster_ids >= 0])) if np.any(cluster_ids >= 0) else -1
                cluster_ids = cluster_ids.copy()
                noise_indices = np.where(cluster_ids == -1)[0]
                for i, noise_idx in enumerate(noise_indices):
                    cluster_ids[noise_idx] = max_cluster_id + 1 + i
                logging.info(f"Assigned {n_noise} noise points to {n_noise} singleton clusters (IDs: {max_cluster_id + 1} to {max_cluster_id + n_noise})")
        
            if len(cluster_ids) != len(train_inputs):
                raise ValueError(f"Mismatch: cluster_ids={len(cluster_ids)} vs train_inputs={len(train_inputs)}")
        
            n_clusters = int(len(np.unique(cluster_ids)))
            logging.info(f"Run {run_idx + 1}: Dict+DBSCAN clustering produced {n_clusters} clusters (eps={eps:.4f})")
            plot_cluster_distribution(cluster_ids, output_dir, run_idx)
        
            if len(cluster_counts_per_run) <= run_idx:
                cluster_counts_per_run.append(n_clusters)
            else:
                cluster_counts_per_run[run_idx] = n_clusters
        
        else:
            raise ValueError(f"Unknown clustering method: {args.clustering}")
        
        # Save cluster_ids for this run
        all_cluster_ids.append(cluster_ids.copy())
        cluster_ids_file = output_dir / f"cluster_ids_run_{run_idx + 1}.npy"
        np.save(cluster_ids_file, cluster_ids)
        logging.info(f"Saved cluster_ids to {cluster_ids_file}")
        
        # Save dict artifacts if available
        if dict_R is not None:
            dict_R_file = output_dir / f"dict_R_run_{run_idx + 1}.npy"
            np.save(dict_R_file, dict_R)
            logging.info(f"Saved dict_R to {dict_R_file}")
        
        if dict_atom_freq is not None:
            dict_atom_freq_file = output_dir / f"dict_atom_freq_run_{run_idx + 1}.npy"
            np.save(dict_atom_freq_file, dict_atom_freq)
            logging.info(f"Saved dict_atom_freq to {dict_atom_freq_file}")

    # Compute cluster count statistics
    valid_cluster_counts = [x for x in cluster_counts_per_run if x is not None]
    if valid_cluster_counts:
        cluster_stats = {
            "mean": float(np.mean(valid_cluster_counts)),
            "std": float(np.std(valid_cluster_counts)),
            "min": int(np.min(valid_cluster_counts)),
            "max": int(np.max(valid_cluster_counts)),
            "values": [int(x) for x in valid_cluster_counts],
        }
        logging.info(
            f"Cluster counts: mean={cluster_stats['mean']:.1f} ± {cluster_stats['std']:.1f}, "
            f"range=[{cluster_stats['min']}, {cluster_stats['max']}]"
        )
        # Create aggregate plot across all runs
        plot_aggregate_cluster_statistics(cluster_counts_per_run, output_dir)
    else:
        cluster_stats = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "values": [],
        }

    # Save summary metrics
    metrics = {
        "embedding_model_name": embedding_model_name,
        "dataset_type": dataset_type,
        "train_pool_size": train_pool_size,
        "n_runs": n_runs,
        "seed": seed,
        "clustering": args.clustering,
        "cluster_counts": cluster_stats,
        "config": {
            "dict_n_components": args.dict_n_components,
            "dict_alpha": args.dict_alpha,
            "dict_top_k": args.dict_top_k,
            "dict_tau": args.dict_tau,
            "dict_standardize": args.dict_standardize,
            "dict_use_minibatch": args.dict_use_minibatch,
            "dict_batch_size": args.dict_batch_size,
            "dict_pca_dim": args.dict_pca_dim,
            "dict_transform_algorithm": args.dict_transform_algorithm,
            "dict_regularization_type": args.dict_regularization_type,
            "dict_max_iter": args.dict_max_iter,
            "dbscan_k": args.dbscan_k,
            "dbscan_q": args.dbscan_q,
            "dbscan_min_samples": args.dbscan_min_samples,
        },
    }
    
    with open(output_dir / "clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {output_dir / 'clustering_metrics.json'}")
    
    logging.info("Clustering finished.")


if __name__ == "__main__":
    main()

