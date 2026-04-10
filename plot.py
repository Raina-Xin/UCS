#!/usr/bin/env python3
"""
Plot comparison metrics from metrics.json files.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List


def plot_metrics(metrics_path: Path, output_path: Path = None):
    """Create an aesthetic bar plot comparing different methods."""
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract data
    summary = metrics.get('summary', {})
    task_name = metrics.get('bbh_task_name', 'Unknown Task')
    model_name = metrics.get('model_name', 'Unknown Model')
    budget = metrics.get('budget', 'N/A')
    n_runs = metrics.get('n_runs', 1)
    
    # Prepare data for plotting
    methods = []
    means = []
    stds = []
    colors = []
    
    # Method names and colors
    method_configs = {
        'dict_selected': ('Dictionary Learning + SGT', '#2E86AB'),  # Blue
        'random': ('Random', '#A23B72'),  # Purple
        'openicl_dpp': ('OpenICL-DPP', '#F18F01'),  # Orange
        'openicl_mdl': ('OpenICL-MDL', '#C73E1D'),  # Red
        'openicl_votek': ('OpenICL-VoteK', '#6A994E'),  # Green
    }
    
    for key, (display_name, color) in method_configs.items():
        if key in summary:
            methods.append(display_name)
            means.append(summary[key]['mean'])
            stds.append(summary[key]['std'])
            colors.append(color)
    
    # Create figure with better styling
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        label = f'{mean_val:.3f}'
        if std_val > 0:
            label += f' ± {std_val:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize axes
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.set_ylim([0, max(means) * 1.2])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Title
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    title = f'ICL Performance Comparison: {task_name.replace("_", " ").title()}\n'
    title += f'Model: {model_short} | Budget: {budget} | Runs: {n_runs}'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Add cluster count info if available
    if 'cluster_counts' in metrics:
        cluster_info = metrics['cluster_counts']
        cluster_text = f"Clusters: {cluster_info.get('mean', 'N/A'):.1f}"
        if cluster_info.get('std', 0) > 0:
            cluster_text += f" ± {cluster_info['std']:.1f}"
        ax.text(0.02, 0.98, cluster_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        # Save next to metrics.json
        output_path = metrics_path.parent / f"{metrics_path.stem}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.close()

def plot_cluster_distribution(cluster_ids: np.ndarray, output_path: Path, run_idx: int):
    """
    Plot and save the distribution of cluster sizes.
    
    Args:
        cluster_ids: Array of cluster assignments for each example
        output_path: Directory to save the plot
        run_idx: Run index for labeling
    """
    # Count examples per cluster
    unique_clusters, cluster_counts = np.unique(cluster_ids, return_counts=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Histogram of cluster sizes 1-10 showing number of points
    ax1 = axes[0]
    # Count number of examples (points) in clusters of each size 1-10
    size_range = np.arange(1, 11)  # Sizes 1 to 10
    points_per_size = []
    for size in size_range:
        # Find clusters of this size and sum up the total number of examples
        clusters_of_size = cluster_counts[cluster_counts == size]
        total_points = len(clusters_of_size) * size  # number of clusters * cluster size
        points_per_size.append(total_points)
    
    # Create bar plot (histogram-like)
    bars = ax1.bar(size_range, points_per_size, edgecolor='black', alpha=0.7, width=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, points_per_size):
        if value > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Cluster Size (number of examples per cluster)', fontsize=12)
    ax1.set_ylabel('Number of Points (examples)', fontsize=12)
    ax1.set_title(f'Run {run_idx + 1}: Number of Points in Clusters of Size 1-10\n'
                  f'Total clusters: {len(unique_clusters)}, '
                  f'Mean size: {np.mean(cluster_counts):.2f}, '
                  f'Median size: {np.median(cluster_counts):.2f}, '
                  f'Min size: {np.min(cluster_counts)}, '
                  f'Max size: {np.max(cluster_counts)}', fontsize=11)
    ax1.set_xticks(size_range)
    ax1.set_xticklabels([str(s) for s in size_range])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Bar plot of cluster sizes (sorted, top clusters)
    ax2 = axes[1]
    sorted_indices = np.argsort(cluster_counts)[::-1]  # Sort descending
    top_n = min(50, len(unique_clusters))  # Show top 50 clusters
    top_clusters = unique_clusters[sorted_indices[:top_n]]
    top_counts = cluster_counts[sorted_indices[:top_n]]
    
    ax2.bar(range(len(top_counts)), top_counts, alpha=0.7, edgecolor='black')
    ax2.set_xlabel(f'Cluster ID (top {top_n} clusters by size)', fontsize=12)
    ax2.set_ylabel('Number of Examples', fontsize=12)
    ax2.set_title(f'Run {run_idx + 1}: Top {top_n} Clusters by Size', fontsize=11)
    ax2.set_xticks(range(len(top_clusters)))
    ax2.set_xticklabels([f'C{int(cid)}' for cid in top_clusters], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / f"cluster_distribution_run_{run_idx + 1}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved cluster distribution plot to {plot_file}")
    
    # Also save summary statistics
    stats = {
        "n_clusters": int(len(unique_clusters)),
        "total_examples": int(len(cluster_ids)),
        "mean_cluster_size": float(np.mean(cluster_counts)),
        "median_cluster_size": float(np.median(cluster_counts)),
        "std_cluster_size": float(np.std(cluster_counts)),
        "min_cluster_size": int(np.min(cluster_counts)),
        "max_cluster_size": int(np.max(cluster_counts)),
        "cluster_sizes": [int(c) for c in cluster_counts],
    }
    
    stats_file = output_path / f"cluster_stats_run_{run_idx + 1}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info(f"Saved cluster statistics to {stats_file}")


def plot_aggregate_cluster_statistics(cluster_counts_per_run: List[int], output_path: Path):
    """
    Plot aggregate statistics across all runs.
    
    Args:
        cluster_counts_per_run: List of cluster counts for each run
        output_path: Directory to save the plot
    """
    if not cluster_counts_per_run or all(x is None for x in cluster_counts_per_run):
        logging.warning("No cluster counts available for aggregate plot")
        return
    
    valid_counts = [x for x in cluster_counts_per_run if x is not None]
    if not valid_counts:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cluster count per run
    ax1 = axes[0]
    run_numbers = [i + 1 for i, x in enumerate(cluster_counts_per_run) if x is not None]
    ax1.plot(run_numbers, valid_counts, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Run Number', fontsize=12)
    ax1.set_ylabel('Number of Clusters', fontsize=12)
    ax1.set_title(f'Number of Clusters Across Runs\n'
                  f'Mean: {np.mean(valid_counts):.1f}, '
                  f'Std: {np.std(valid_counts):.1f}, '
                  f'Range: [{np.min(valid_counts)}, {np.max(valid_counts)}]', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(run_numbers)
    
    # Plot 2: Distribution of cluster counts
    ax2 = axes[1]
    ax2.hist(valid_counts, bins=min(20, len(valid_counts)), edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(np.mean(valid_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_counts):.1f}')
    ax2.axvline(np.median(valid_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_counts):.1f}')
    ax2.set_xlabel('Number of Clusters', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Cluster Counts Across All Runs', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / "cluster_statistics_aggregate.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved aggregate cluster statistics plot to {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from JSON file")
    parser.add_argument("metrics_json", type=str, help="Path to metrics.json file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for plot")
    args = parser.parse_args()
    
    metrics_path = Path(args.metrics_json)
    output_path = Path(args.output) if args.output else None
    
    plot_metrics(metrics_path, output_path)

