#!/usr/bin/env python3
"""
分析embedding分布和计算选中样本的det(XX^T)（体积）

用法:
    python analyze_embeddings.py --output_dir <output_dir> [--embedding_cache <path>]
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def to_python_type(x):
    """将numpy类型转换为Python原生类型（JSON可序列化）"""
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    elif isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    elif isinstance(x, np.ndarray):
        return [to_python_type(item) for item in x]
    elif isinstance(x, list):
        return [to_python_type(item) for item in x]
    else:
        return x


def load_embeddings(cache_path: Path) -> np.ndarray:
    """加载embedding缓存"""
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")
    embs = np.load(cache_path)
    logging.info(f"Loaded embeddings: shape={embs.shape}")
    return embs


def analyze_embedding_distribution(embeddings: np.ndarray) -> Dict:
    """
    分析embedding的分布
    
    由于embeddings是L2归一化的，它们应该在单位球面上。
    我们检查：
    1. 是否真的在单位球面上（L2 norm ≈ 1）
    2. 在球面上的分布是否均匀
    3. 各维度是否独立正态分布（投影到主成分后）
    """
    n_samples, dim = embeddings.shape
    
    # 1. 检查L2范数
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    # 2. 检查是否在单位球面上
    on_sphere = np.allclose(norms, 1.0, atol=1e-5)
    
    # 3. 计算成对距离的分布（用于检查球面上的均匀性）
    # 对于高维单位球面上的均匀分布，成对距离应该集中在某个值附近
    # 采样一部分来计算（避免计算所有对）
    sample_size = min(1000, n_samples)
    sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
    sample_embs = embeddings[sample_indices]
    
    # 计算成对余弦距离（对于单位向量，1 - cosine = 1 - dot product）
    # 对于均匀分布在球面上的点，余弦距离的分布应该接近某个分布
    pairwise_dots = np.dot(sample_embs, sample_embs.T)
    # 只取上三角（不包括对角线）
    upper_tri = pairwise_dots[np.triu_indices(len(sample_embs), k=1)]
    pairwise_cosine_dist = 1 - upper_tri
    
    # 4. PCA分析 - 检查主成分的方差分布
    # 如果是均匀分布在球面上，主成分的方差应该相对均匀
    pca = PCA(n_components=min(50, dim))
    pca.fit(embeddings)
    explained_var = pca.explained_variance_ratio_
    
    # 5. 检查各维度的分布（投影到原始维度）
    # 对于单位球面上的均匀分布，各维度的值应该接近正态分布（中心极限定理）
    dim_means = np.mean(embeddings, axis=0)
    dim_stds = np.std(embeddings, axis=0)
    
    # 6. 检查是否接近正态分布（使用Shapiro-Wilk测试的简化版本）
    # 随机选择几个维度进行测试
    test_dims = np.random.choice(dim, size=min(10, dim), replace=False)
    normality_pvals = []
    for d in test_dims:
        _, pval = stats.normaltest(embeddings[:, d])
        normality_pvals.append(pval)
    
    results = {
        "n_samples": int(n_samples),
        "dimension": int(dim),
        "mean_norm": float(mean_norm),
        "std_norm": float(std_norm),
        "on_unit_sphere": bool(on_sphere),
        "pairwise_cosine_dist_mean": float(np.mean(pairwise_cosine_dist)),
        "pairwise_cosine_dist_std": float(np.std(pairwise_cosine_dist)),
        "pairwise_cosine_dist_min": float(np.min(pairwise_cosine_dist)),
        "pairwise_cosine_dist_max": float(np.max(pairwise_cosine_dist)),
        "pca_explained_var_top10": [float(x) for x in explained_var[:10]],
        "pca_explained_var_mean": float(np.mean(explained_var)),
        "dim_means_mean": float(np.mean(dim_means)),
        "dim_means_std": float(np.std(dim_means)),
        "dim_stds_mean": float(np.mean(dim_stds)),
        "dim_stds_std": float(np.std(dim_stds)),
        "normality_test_pvals_mean": float(np.mean(normality_pvals)),
    }
    
    return results


def compute_determinant_volume(X: np.ndarray) -> Tuple[float, Dict]:
    """
    计算 det(XX^T)，即embedding向量张成的多面体的体积
    
    Args:
        X: (n_samples, dim) 的embedding矩阵
    
    Returns:
        det_value: det(XX^T)的值
        info: 包含相关信息的字典
    """
    n_samples, dim = X.shape
    
    # 确保数据类型是float32或float64（numpy.linalg不支持float16）
    if X.dtype == np.float16:
        X = X.astype(np.float32)
    elif X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    
    # 如果样本数大于维度，我们需要使用不同的方法
    # det(XX^T) = det(X^T X) 当 n_samples >= dim
    # 但更稳定的方法是使用SVD
    
    if n_samples >= dim:
        # 计算 X^T X (dim x dim)
        XTX = np.dot(X.T, X)
        try:
            det_value = np.linalg.det(XTX)
            # 如果行列式太小，可能数值不稳定，使用log det
            log_det = np.linalg.slogdet(XTX)[1]
        except np.linalg.LinAlgError:
            det_value = 0.0
            log_det = -np.inf
    else:
        # 如果 n_samples < dim，XX^T 是 (n_samples x n_samples)
        XXT = np.dot(X, X.T)
        try:
            det_value = np.linalg.det(XXT)
            log_det = np.linalg.slogdet(XXT)[1]
        except np.linalg.LinAlgError:
            det_value = 0.0
            log_det = -np.inf
    
    # 计算条件数（衡量矩阵的数值稳定性）
    if n_samples >= dim:
        try:
            cond_num = np.linalg.cond(XTX.astype(np.float64))
        except:
            cond_num = np.inf
    else:
        try:
            cond_num = np.linalg.cond(XXT.astype(np.float64))
        except:
            cond_num = np.inf
    
    # 计算有效秩（通过SVD）
    # 确保使用支持的数据类型
    X_svd = X.astype(np.float64) if X.dtype == np.float16 else X
    U, s, Vt = np.linalg.svd(X_svd, full_matrices=False)
    # 有效秩：奇异值大于最大奇异值的1e-10倍的数量
    effective_rank = np.sum(s > s[0] * 1e-10)
    
    # 计算体积的几何解释
    # 对于单位球面上的点，体积与点之间的角度分布有关
    # 更分散的点 -> 更大的体积
    
    info = {
        "det_XXT": float(det_value),
        "log_det_XXT": float(log_det),
        "condition_number": float(cond_num),
        "effective_rank": int(effective_rank),
        "n_samples": int(n_samples),
        "dimension": int(dim),
        "singular_values_top10": [float(x) if isinstance(x, (np.floating, np.integer)) else float(x) for x in s[:10]],
        "singular_values_mean": float(np.mean(s)),
    }
    
    return det_value, info


def analyze_selected_samples(
    embeddings: np.ndarray,
    selected_indices: List[int],
    method_name: str,
    pool_indices: Optional[List[int]] = None,
) -> Dict:
    """
    分析选中样本的embedding
    
    Args:
        embeddings: 所有样本的embeddings (n_total, dim)
        selected_indices: 选中样本的索引（相对于pool或全局）
        method_name: 方法名称
        pool_indices: 如果selected_indices是相对于pool的，提供pool的全局索引
    """
    # 如果提供了pool_indices，需要映射到全局索引
    if pool_indices is not None:
        global_indices = [pool_indices[i] for i in selected_indices]
    else:
        global_indices = selected_indices
    
    # 提取选中样本的embeddings
    selected_embs = embeddings[global_indices]
    
    # 计算det(XX^T)
    det_value, det_info = compute_determinant_volume(selected_embs)
    
    # 计算选中样本的统计信息
    selected_norms = np.linalg.norm(selected_embs, axis=1)
    
    # 计算选中样本之间的平均距离
    if len(selected_embs) > 1:
        pairwise_dots = np.dot(selected_embs, selected_embs.T)
        upper_tri = pairwise_dots[np.triu_indices(len(selected_embs), k=1)]
        avg_cosine_sim = np.mean(upper_tri)
        std_cosine_sim = np.std(upper_tri)
        min_cosine_sim = np.min(upper_tri)
        max_cosine_sim = np.max(upper_tri)
    else:
        avg_cosine_sim = 1.0
        std_cosine_sim = 0.0
        min_cosine_sim = 1.0
        max_cosine_sim = 1.0
    
    # 确保所有值都是Python原生类型（JSON可序列化）
    results = {
        "method": method_name,
        "n_selected": len(selected_indices),
        "det_XXT": to_python_type(det_value),
        "log_det_XXT": to_python_type(det_info["log_det_XXT"]),
        "condition_number": to_python_type(det_info["condition_number"]),
        "effective_rank": to_python_type(det_info["effective_rank"]),
        "mean_norm": to_python_type(np.mean(selected_norms)),
        "std_norm": to_python_type(np.std(selected_norms)),
        "avg_pairwise_cosine_sim": to_python_type(avg_cosine_sim),
        "std_pairwise_cosine_sim": to_python_type(std_cosine_sim),
        "min_pairwise_cosine_sim": to_python_type(min_cosine_sim),
        "max_pairwise_cosine_sim": to_python_type(max_cosine_sim),
        "singular_values": to_python_type(det_info["singular_values_top10"]),
    }
    
    return results


def plot_determinant_comparison(selected_analysis_results: Dict, output_dir: Path):
    """绘制不同方法的det(XX^T)对比图"""
    if not selected_analysis_results:
        return
    
    methods = list(selected_analysis_results.keys())
    log_det_means = [selected_analysis_results[m]["mean_log_det_XXT"] for m in methods]
    log_det_stds = [selected_analysis_results[m]["std_log_det_XXT"] for m in methods]
    effective_ranks = [selected_analysis_results[m]["mean_effective_rank"] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Log det(XX^T)对比
    ax1 = axes[0]
    x_pos = np.arange(len(methods))
    bars = ax1.bar(x_pos, log_det_means, yerr=log_det_stds, capsize=5, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Log det(XX^T)')
    ax1.set_title('Volume Comparison: Log det(XX^T) by Method')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(log_det_means, log_det_stds)):
        ax1.text(i, mean + std + 0.5, f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Effective rank对比
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, effective_ranks, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Effective Rank')
    ax2.set_title('Effective Rank by Method')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, rank in enumerate(effective_ranks):
        ax2.text(i, rank + 0.5, f'{rank:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / "determinant_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved determinant comparison plot to {output_path}")


def plot_embedding_analysis(embeddings: np.ndarray, output_dir: Path):
    """绘制embedding分布的可视化"""
    n_samples, dim = embeddings.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. L2范数分布
    ax1 = axes[0, 0]
    norms = np.linalg.norm(embeddings, axis=1)
    ax1.hist(norms, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Unit sphere')
    ax1.set_xlabel('L2 Norm')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of L2 Norms\nMean: {np.mean(norms):.6f}, Std: {np.std(norms):.6f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 成对余弦距离分布（采样）
    ax2 = axes[0, 1]
    sample_size = min(500, n_samples)
    sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
    sample_embs = embeddings[sample_indices]
    pairwise_dots = np.dot(sample_embs, sample_embs.T)
    upper_tri = pairwise_dots[np.triu_indices(len(sample_embs), k=1)]
    pairwise_cosine_dist = 1 - upper_tri
    ax2.hist(pairwise_cosine_dist, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Cosine Distance (1 - cosine similarity)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Pairwise Cosine Distance Distribution\n(sample of {sample_size} points)')
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA解释方差
    ax3 = axes[1, 0]
    pca = PCA(n_components=min(50, dim))
    pca.fit(embeddings)
    explained_var = pca.explained_variance_ratio_
    ax3.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linewidth=2, markersize=4)
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Explained Variance Ratio')
    ax3.set_title('PCA Explained Variance')
    ax3.grid(True, alpha=0.3)
    
    # 4. 随机选择几个维度的分布
    ax4 = axes[1, 1]
    n_dims_to_plot = min(5, dim)
    dims_to_plot = np.random.choice(dim, size=n_dims_to_plot, replace=False)
    for i, d in enumerate(dims_to_plot):
        ax4.hist(embeddings[:, d], bins=30, alpha=0.5, label=f'Dim {d}')
    ax4.set_xlabel('Embedding Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Distribution of Random Dimensions (sample of {n_dims_to_plot})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "embedding_distribution_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved embedding distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="分析embedding分布和计算选中样本的det(XX^T)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录（包含metrics.json或selected_indices）")
    parser.add_argument("--embedding_cache", type=str, default=None,
                        help="Embedding缓存路径（如果不提供，会从output_dir推断）")
    parser.add_argument("--embedding_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        help="Embedding模型名称（用于推断缓存路径）")
    parser.add_argument("--dataset_name", type=str, default="banking77_train_full",
                        help="数据集名称（用于推断缓存路径）")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # 确定embedding缓存路径
    if args.embedding_cache:
        emb_cache_path = Path(args.embedding_cache)
    else:
        # 从embed_and_cluster推断路径
        cache_root = Path("outputs") / "LLM_embeddings"
        model_name_clean = args.embedding_model.replace("/", "__")
        emb_cache_path = cache_root / args.dataset_name / f"{model_name_clean}_layerlast_pool-mean_l2-1.npy"
    
    logging.info(f"Loading embeddings from: {emb_cache_path}")
    all_embeddings = load_embeddings(emb_cache_path)
    
    # 分析所有embedding的分布
    logging.info("Analyzing embedding distribution...")
    dist_results = analyze_embedding_distribution(all_embeddings)
    
    # 绘制分布图
    plot_embedding_analysis(all_embeddings, output_dir)
    
    # 保存分布分析结果
    with open(output_dir / "embedding_distribution_analysis.json", "w") as f:
        json.dump(dist_results, f, indent=2)
    logging.info(f"Saved embedding distribution analysis to {output_dir / 'embedding_distribution_analysis.json'}")
    
    # 打印关键发现
    print("\n" + "="*60)
    print("Embedding Distribution Analysis")
    print("="*60)
    print(f"Total samples: {dist_results['n_samples']}")
    print(f"Dimension: {dist_results['dimension']}")
    print(f"Mean L2 norm: {dist_results['mean_norm']:.6f} ± {dist_results['std_norm']:.6f}")
    print(f"On unit sphere: {dist_results['on_unit_sphere']}")
    print(f"Pairwise cosine distance: {dist_results['pairwise_cosine_dist_mean']:.4f} ± {dist_results['pairwise_cosine_dist_std']:.4f}")
    print(f"  Range: [{dist_results['pairwise_cosine_dist_min']:.4f}, {dist_results['pairwise_cosine_dist_max']:.4f}]")
    print(f"PCA top component explains: {dist_results['pca_explained_var_top10'][0]:.4f} of variance")
    print(f"PCA mean explained variance: {dist_results['pca_explained_var_mean']:.4f}")
    
    # 尝试加载metrics.json来获取选中索引并分析
    metrics_path = output_dir / "metrics.json"
    selected_analysis_results = {}
    
    if metrics_path.exists():
        logging.info(f"Found metrics.json, analyzing selected samples...")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # 获取选中索引和pool索引
        selected_indices_dict = metrics.get("selected_indices", {})
        pool_indices_list = metrics.get("pool_indices", [])
        
        if selected_indices_dict and pool_indices_list:
            logging.info(f"Found selected indices for {len(selected_indices_dict)} methods across {len(pool_indices_list)} runs")
            
            # 对每个方法进行分析
            for method_name, indices_per_run in selected_indices_dict.items():
                method_results = []
                
                for run_idx, indices in enumerate(indices_per_run):
                    if run_idx < len(pool_indices_list):
                        pool_indices = pool_indices_list[run_idx]
                        result = analyze_selected_samples(
                            all_embeddings,
                            indices,
                            f"{method_name}_run_{run_idx + 1}",
                            pool_indices=pool_indices,
                        )
                        method_results.append(result)
                
                # 计算跨run的统计
                if method_results:
                    det_values = [r["det_XXT"] for r in method_results]
                    log_det_values = [r["log_det_XXT"] for r in method_results]
                    effective_ranks = [r["effective_rank"] for r in method_results]
                    
                    selected_analysis_results[method_name] = {
                        "mean_det_XXT": to_python_type(np.mean(det_values)),
                        "std_det_XXT": to_python_type(np.std(det_values)),
                        "mean_log_det_XXT": to_python_type(np.mean(log_det_values)),
                        "std_log_det_XXT": to_python_type(np.std(log_det_values)),
                        "mean_effective_rank": to_python_type(np.mean(effective_ranks)),
                        "std_effective_rank": to_python_type(np.std(effective_ranks)),
                        "n_runs": len(method_results),
                        "per_run": method_results,  # method_results已经包含Python原生类型
                    }
            
            # 保存选中样本分析结果
            with open(output_dir / "selected_samples_analysis.json", "w") as f:
                json.dump(selected_analysis_results, f, indent=2)
            logging.info(f"Saved selected samples analysis to {output_dir / 'selected_samples_analysis.json'}")
            
            # 打印结果
            print("\n" + "="*60)
            print("Selected Samples Analysis (det(XX^T) - Volume)")
            print("="*60)
            for method_name, results in sorted(selected_analysis_results.items()):
                print(f"\n{method_name}:")
                print(f"  Mean det(XX^T): {results['mean_det_XXT']:.6e} ± {results['std_det_XXT']:.6e}")
                print(f"  Mean log det(XX^T): {results['mean_log_det_XXT']:.4f} ± {results['std_log_det_XXT']:.4f}")
                print(f"  Mean effective rank: {results['mean_effective_rank']:.2f} ± {results['std_effective_rank']:.2f}")
                print(f"  Number of runs: {results['n_runs']}")
            
            # 创建对比图
            plot_determinant_comparison(selected_analysis_results, output_dir)
        else:
            logging.warning("Selected indices or pool indices not found in metrics.json")
            print("\nNote: Selected indices not found in metrics.json. Run the main script with updated code to save indices.")
    else:
        logging.warning(f"metrics.json not found in {output_dir}")
        print("\nNote: metrics.json not found. Cannot analyze selected samples without indices.")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()

