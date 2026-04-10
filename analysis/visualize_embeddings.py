#!/usr/bin/env python3
"""
可视化高维embedding的分布，检查是否在球面上或接近正态分布

支持多种降维和可视化方法：
1. t-SNE / UMAP 降维到2D/3D
2. PCA 前几个主成分
3. 随机投影
4. 范数和角度分布
5. 成对距离分布
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 需要导入3D支持
from pathlib import Path
from typing import Optional, Tuple
import logging
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    logging.warning("sklearn not available, t-SNE will be skipped")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logging.warning("umap-learn not available, UMAP will be skipped")

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def load_embeddings(cache_path: Path) -> np.ndarray:
    """加载embedding缓存"""
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")
    embs = np.load(cache_path)
    logging.info(f"Loaded embeddings: shape={embs.shape}")
    return embs


def sample_embeddings(embeddings: np.ndarray, n_samples: int = 2000, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    采样embeddings用于可视化（高维数据可视化需要采样）
    
    Returns:
        sampled_embs: 采样后的embeddings
        sample_indices: 采样索引
    """
    n_total = len(embeddings)
    if n_samples >= n_total:
        return embeddings, np.arange(n_total)
    
    np.random.seed(random_seed)
    sample_indices = np.random.choice(n_total, size=n_samples, replace=False)
    sampled_embs = embeddings[sample_indices]
    
    logging.info(f"Sampled {n_samples} embeddings from {n_total} total")
    return sampled_embs, sample_indices


def plot_norm_distribution(embeddings: np.ndarray, output_path: Path):
    """绘制L2范数分布"""
    norms = np.linalg.norm(embeddings, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    ax1 = axes[0]
    ax1.hist(norms, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Unit sphere (norm=1)')
    ax1.axvline(np.mean(norms), color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(norms):.6f}')
    ax1.set_xlabel('L2 Norm')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of L2 Norms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot against normal distribution
    ax2 = axes[1]
    stats.probplot(norms, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Norms vs Normal Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved norm distribution plot to {output_path}")


def plot_angle_distribution(embeddings: np.ndarray, output_path: Path, n_pairs: int = 10000):
    """
    绘制角度分布（用于检查是否在球面上均匀分布）
    
    对于单位球面上的点，如果均匀分布，角度应该遵循特定分布
    """
    n_samples = len(embeddings)
    n_pairs = min(n_pairs, n_samples * (n_samples - 1) // 2)
    
    # 采样成对点
    np.random.seed(42)
    if n_pairs < n_samples * (n_samples - 1) // 2:
        # 随机采样成对点
        pairs = []
        max_attempts = n_pairs * 10
        attempts = 0
        while len(pairs) < n_pairs and attempts < max_attempts:
            i, j = np.random.choice(n_samples, size=2, replace=False)
            if i < j:
                pairs.append((i, j))
            attempts += 1
        pairs = pairs[:n_pairs]
    else:
        # 使用所有对
        pairs = [(i, j) for i in range(n_samples) for j in range(i+1, n_samples)]
    
    # 计算角度（通过点积，因为都是单位向量）
    angles = []
    for i, j in pairs:
        dot_product = np.dot(embeddings[i], embeddings[j])
        # 限制在[-1, 1]范围内（数值误差）
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)  # 角度在[0, π]
        angles.append(angle)
    
    angles = np.array(angles)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 角度直方图
    ax1 = axes[0, 0]
    ax1.hist(angles, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax1.axvline(np.mean(angles), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(angles):.4f} rad ({np.degrees(np.mean(angles)):.2f}°)')
    ax1.set_xlabel('Angle (radians)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution of Pairwise Angles\n(n={len(angles)} pairs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 角度度数分布
    ax2 = axes[0, 1]
    angles_deg = np.degrees(angles)
    ax2.hist(angles_deg, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax2.axvline(np.mean(angles_deg), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(angles_deg):.2f}°')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Pairwise Angles (degrees)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 余弦相似度分布
    ax3 = axes[1, 0]
    cosine_sims = np.cos(angles)
    ax3.hist(cosine_sims, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax3.axvline(np.mean(cosine_sims), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(cosine_sims):.4f}')
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Cosine Similarities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 与理论均匀分布对比（高维球面上均匀分布的角度分布）
    # 对于d维单位球面上的均匀分布，角度分布的概率密度函数为：
    # f(θ) = (sin θ)^(d-2) / B((d-1)/2, 1/2)
    # 这里我们绘制理论曲线
    ax4 = axes[1, 1]
    dim = embeddings.shape[1]
    theta = np.linspace(0, np.pi, 1000)
    # 简化的理论分布（忽略归一化常数）
    theoretical = np.power(np.sin(theta), dim - 2)
    theoretical = theoretical / np.trapz(theoretical, theta)  # 归一化
    
    ax4.hist(angles, bins=50, edgecolor='black', alpha=0.5, density=True, label='Observed')
    ax4.plot(theta, theoretical, 'r-', linewidth=2, label=f'Theoretical (uniform on {dim}D sphere)')
    ax4.set_xlabel('Angle (radians)')
    ax4.set_ylabel('Density')
    ax4.set_title('Observed vs Theoretical (Uniform on Sphere)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved angle distribution plot to {output_path}")


def plot_pca_visualization(embeddings: np.ndarray, output_path: Path, n_components: int = 3):
    """PCA降维可视化"""
    logging.info(f"Computing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)
    
    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)
    
    fig = plt.figure(figsize=(16, 5))
    
    if n_components >= 2:
        # 2D scatter plot (PC1 vs PC2)
        ax1 = plt.subplot(1, 3, 1)
        scatter = ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                             alpha=0.5, s=1, c=embeddings_pca[:, 2] if n_components >= 3 else None,
                             cmap='viridis' if n_components >= 3 else None)
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.2%} variance)')
        ax1.set_title('PCA: PC1 vs PC2')
        if n_components >= 3:
            plt.colorbar(scatter, ax=ax1, label='PC3')
        ax1.grid(True, alpha=0.3)
    
    if n_components >= 3:
        # 3D scatter plot
        ax2 = plt.subplot(1, 3, 2, projection='3d')
        ax2.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2],
                   alpha=0.5, s=1)
        ax2.set_xlabel(f'PC1 ({explained_var[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({explained_var[1]:.2%})')
        ax2.set_zlabel(f'PC3 ({explained_var[2]:.2%})')
        ax2.set_title('PCA: 3D View')
    
    # Explained variance
    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, edgecolor='black')
    ax3.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-', linewidth=2, markersize=4)
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Explained Variance Ratio')
    ax3.set_title('PCA Explained Variance')
    ax3.legend(['Cumulative', 'Individual'])
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved PCA visualization to {output_path}")
    
    return embeddings_pca, explained_var


def plot_tsne_visualization(embeddings: np.ndarray, output_path: Path, n_components: int = 2):
    """t-SNE降维可视化"""
    if not HAS_TSNE:
        logging.warning("sklearn not available, skipping t-SNE")
        return None
    
    logging.info(f"Computing t-SNE with {n_components} components (this may take a while)...")
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if n_components == 2:
        ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.5, s=1)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2],
                  alpha=0.5, s=1)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
    
    ax.set_title('t-SNE Visualization')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved t-SNE visualization to {output_path}")
    
    return embeddings_tsne


def plot_umap_visualization(embeddings: np.ndarray, output_path: Path, n_components: int = 2):
    """UMAP降维可视化"""
    if not HAS_UMAP:
        logging.warning("umap-learn not available, skipping UMAP")
        return None
    
    logging.info(f"Computing UMAP with {n_components} components...")
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_umap = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if n_components == 2:
        ax.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], alpha=0.5, s=1)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], embeddings_umap[:, 2],
                  alpha=0.5, s=1)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
    
    ax.set_title('UMAP Visualization')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved UMAP visualization to {output_path}")
    
    return embeddings_umap


def plot_dimension_distribution(embeddings: np.ndarray, output_path: Path, n_dims_to_plot: int = 10):
    """检查各维度的分布（是否接近正态分布）"""
    n_samples, dim = embeddings.shape
    n_dims_to_plot = min(n_dims_to_plot, dim)
    
    # 随机选择一些维度
    np.random.seed(42)
    dims_to_plot = np.random.choice(dim, size=n_dims_to_plot, replace=False)
    
    n_cols = 3
    n_rows = (n_dims_to_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, d in enumerate(dims_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        values = embeddings[:, d]
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, density=True)
        
        # 拟合正态分布
        mu, sigma = np.mean(values), np.std(values)
        x = np.linspace(values.min(), values.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        
        # 正态性检验
        _, pval = stats.normaltest(values)
        
        ax.set_xlabel(f'Value (Dimension {d})')
        ax.set_ylabel('Density')
        ax.set_title(f'Dim {d}: p={pval:.4f} (normality test)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_dims_to_plot, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved dimension distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化高维embedding的分布")
    parser.add_argument("--embedding_cache", type=str, required=True,
                        help="Embedding缓存路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="用于可视化的采样数量（默认2000）")
    parser.add_argument("--skip_tsne", action="store_true",
                        help="跳过t-SNE（计算较慢）")
    parser.add_argument("--skip_umap", action="store_true",
                        help="跳过UMAP")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载embeddings
    emb_cache_path = Path(args.embedding_cache)
    logging.info(f"Loading embeddings from: {emb_cache_path}")
    all_embeddings = load_embeddings(emb_cache_path)
    
    # 采样用于可视化
    embeddings, sample_indices = sample_embeddings(all_embeddings, n_samples=args.n_samples)
    
    logging.info("Generating visualizations...")
    
    # 1. 范数分布
    plot_norm_distribution(embeddings, output_dir / "norm_distribution.png")
    
    # 2. 角度分布（检查球面分布）
    plot_angle_distribution(embeddings, output_dir / "angle_distribution.png")
    
    # 3. PCA可视化
    embeddings_pca, explained_var = plot_pca_visualization(
        embeddings, output_dir / "pca_visualization.png", n_components=3
    )
    
    # 4. t-SNE（可选，较慢）
    if not args.skip_tsne and HAS_TSNE:
        plot_tsne_visualization(embeddings, output_dir / "tsne_visualization.png", n_components=2)
    
    # 5. UMAP（可选）
    if not args.skip_umap and HAS_UMAP:
        plot_umap_visualization(embeddings, output_dir / "umap_visualization.png", n_components=2)
    
    # 6. 维度分布（检查正态分布）
    plot_dimension_distribution(embeddings, output_dir / "dimension_distribution.png")
    
    # 保存统计信息
    stats_info = {
        "n_samples": int(len(embeddings)),
        "dimension": int(embeddings.shape[1]),
        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "pca_explained_var_top10": [float(x) for x in explained_var[:10]],
        "pca_cumulative_var_top10": [float(x) for x in np.cumsum(explained_var)[:10]],
    }
    
    with open(output_dir / "visualization_stats.json", "w") as f:
        json.dump(stats_info, f, indent=2)
    
    logging.info(f"All visualizations saved to {output_dir}")
    print("\n" + "="*60)
    print("Visualization Summary")
    print("="*60)
    print(f"PCA top 3 components explain: {sum(explained_var[:3]):.2%} of variance")
    print(f"PCA top 10 components explain: {sum(explained_var[:10]):.2%} of variance")
    print(f"Mean L2 norm: {stats_info['mean_norm']:.6f} ± {stats_info['std_norm']:.6f}")
    print("\nGenerated plots:")
    print("  - norm_distribution.png: L2 norm distribution")
    print("  - angle_distribution.png: Pairwise angle distribution (sphere check)")
    print("  - pca_visualization.png: PCA 2D/3D projection")
    if not args.skip_tsne and HAS_TSNE:
        print("  - tsne_visualization.png: t-SNE 2D projection")
    if not args.skip_umap and HAS_UMAP:
        print("  - umap_visualization.png: UMAP 2D projection")
    print("  - dimension_distribution.png: Distribution of random dimensions")
    print("="*60)


if __name__ == "__main__":
    main()

