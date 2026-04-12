"""
Visualization module for summarization evaluation results.

Provides publication-ready plots and tables:
- ROUGE comparison charts with error bars
- Score distributions (histogram, violin plots)
- Correlation heatmaps
- Pairwise comparisons
- LaTeX table export
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_rouge_comparison(
    results_df: pd.DataFrame,
    metric: str = "rouge1",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart comparing methods with error bars.

    Args:
        results_df: DataFrame with columns: Method, {metric}_mean, {metric}_std
        metric: ROUGE metric to plot
        title: Plot title (default: auto-generated)
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib Figure object
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in results_df.columns or std_col not in results_df.columns:
        raise ValueError(f"DataFrame must have columns: {mean_col}, {std_col}")

    fig, ax = plt.subplots(figsize=figsize)

    methods = results_df["Method"]
    means = results_df[mean_col]
    stds = results_df[std_col]

    x_pos = np.arange(len(methods))

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')

    # Color bars by performance
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(methods)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    ax.set_title(title or f'{metric.upper()} Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_score_distribution(
    scores: List[float],
    method_name: str,
    metric: str = "rouge1",
    plot_type: str = "both",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Histogram and/or violin plot of score distribution.

    Args:
        scores: List of scores
        method_name: Name of the method
        metric: ROUGE metric name
        plot_type: 'histogram', 'violin', or 'both'
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    if plot_type == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        ax2 = None

    # Histogram
    if plot_type in ["histogram", "both"]:
        ax = ax1 if plot_type == "both" else ax1
        ax.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
        ax.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.3f}')
        ax.set_xlabel(f'{metric.upper()} Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'Score Distribution: {method_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # Violin plot
    if plot_type in ["violin", "both"] and ax2 is not None:
        parts = ax2.violinplot([scores], positions=[0], widths=0.7, showmeans=True, showmedians=True)

        # Customize violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)

        ax2.set_ylabel(f'{metric.upper()} Score', fontsize=11, fontweight='bold')
        ax2.set_title(f'Score Distribution (Violin): {method_name}', fontsize=12, fontweight='bold')
        ax2.set_xticks([])
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Heatmap showing correlations between document features and scores.

    Args:
        corr_df: DataFrame with 'characteristic' and 'correlation' columns
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    if "characteristic" not in corr_df.columns or "correlation" not in corr_df.columns:
        raise ValueError("DataFrame must have 'characteristic' and 'correlation' columns")

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for heatmap
    characteristics = corr_df["characteristic"].tolist()
    correlations = corr_df["correlation"].values.reshape(-1, 1)

    # Create heatmap
    im = ax.imshow(correlations, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    # Set ticks and labels
    ax.set_yticks(np.arange(len(characteristics)))
    ax.set_yticklabels(characteristics)
    ax.set_xticks([0])
    ax.set_xticklabels(['Correlation'])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')

    # Add correlation values as text
    for i, corr in enumerate(correlations.flatten()):
        color = 'white' if abs(corr) > 0.5 else 'black'
        ax.text(0, i, f'{corr:.3f}', ha='center', va='center', color=color, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_pairwise_comparison(
    scores1: List[float],
    scores2: List[float],
    method1_name: str,
    method2_name: str,
    metric: str = "rouge1",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot comparing two methods document-by-document.

    Args:
        scores1: Scores from method 1
        scores2: Scores from method 2
        method1_name: Name of method 1
        method2_name: Name of method 2
        metric: ROUGE metric name
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score lists must have same length")

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(scores1, scores2, alpha=0.6, s=50, edgecolors='black')

    # Diagonal line (equal performance)
    min_val = min(min(scores1), min(scores2))
    max_val = max(max(scores1), max(scores2))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal performance')

    # Compute correlation
    correlation = np.corrcoef(scores1, scores2)[0, 1]

    ax.set_xlabel(f'{method1_name} ({metric.upper()} Score)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{method2_name} ({metric.upper()} Score)', fontsize=11, fontweight='bold')
    ax.set_title(f'Pairwise Comparison\n(Correlation: {correlation:.3f})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_method_comparison_matrix(
    results_list: List[Dict[str, any]],
    metric: str = "rouge1",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Matrix plot comparing all methods against each other.

    Args:
        results_list: List of dicts with 'method', 'scores' keys
        metric: ROUGE metric
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    n_methods = len(results_list)
    method_names = [r['method'] for r in results_list]

    # Compute pairwise correlations
    corr_matrix = np.zeros((n_methods, n_methods))

    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                scores_i = results_list[i]['scores'][metric]
                scores_j = results_list[j]['scores'][metric]
                corr_matrix[i, j] = np.corrcoef(scores_i, scores_j)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(n_methods))
    ax.set_yticks(np.arange(n_methods))
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.set_yticklabels(method_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11, fontweight='bold')

    # Add correlation values
    for i in range(n_methods):
        for j in range(n_methods):
            color = 'white' if abs(corr_matrix[i, j]) > 0.7 else 'black'
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                   ha='center', va='center', color=color, fontsize=10)

    ax.set_title(f'Method Correlation Matrix ({metric.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def export_latex_table(
    df: pd.DataFrame,
    filepath: str,
    caption: str = "Results Table",
    label: str = "tab:results",
    float_format: str = "%.4f"
) -> None:
    """
    Export DataFrame to LaTeX table format.

    Args:
        df: DataFrame to export
        filepath: Output file path
        caption: Table caption
        label: LaTeX label
        float_format: Format for floating point numbers
    """
    latex_str = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False
    )

    # Write to file
    with open(filepath, 'w') as f:
        f.write(latex_str)

    print(f"LaTeX table exported to: {filepath}")


def create_results_dashboard(
    results_df: pd.DataFrame,
    scores_dict: Dict[str, Dict[str, List[float]]],
    corr_df: pd.DataFrame,
    metric: str = "rouge1",
    save_dir: Optional[str] = None
) -> List[plt.Figure]:
    """
    Create a comprehensive dashboard of visualizations.

    Args:
        results_df: Summary results DataFrame
        scores_dict: Dictionary mapping method names to score dictionaries
        corr_df: Correlation DataFrame
        metric: ROUGE metric
        save_dir: Directory to save figures (optional)

    Returns:
        List of Figure objects
    """
    figures = []

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROUGE comparison bar chart
    fig1 = plot_rouge_comparison(
        results_df,
        metric=metric,
        save_path=str(save_dir / f"{metric}_comparison.png") if save_dir else None
    )
    figures.append(fig1)

    # 2. Score distributions for each method
    for method_name, scores in scores_dict.items():
        if metric in scores:
            fig = plot_score_distribution(
                scores[metric],
                method_name,
                metric=metric,
                save_path=str(save_dir / f"{method_name}_{metric}_dist.png") if save_dir else None
            )
            figures.append(fig)

    # 3. Correlation heatmap
    fig3 = plot_correlation_heatmap(
        corr_df,
        title=f"Document Characteristics vs {metric.upper()} Score",
        save_path=str(save_dir / f"{metric}_correlation_heatmap.png") if save_dir else None
    )
    figures.append(fig3)

    # 4. Pairwise comparisons (if multiple methods)
    method_names = list(scores_dict.keys())
    if len(method_names) >= 2:
        method1, method2 = method_names[0], method_names[1]
        if metric in scores_dict[method1] and metric in scores_dict[method2]:
            fig4 = plot_pairwise_comparison(
                scores_dict[method1][metric],
                scores_dict[method2][metric],
                method1,
                method2,
                metric=metric,
                save_path=str(save_dir / f"{metric}_pairwise_{method1}_vs_{method2}.png") if save_dir else None
            )
            figures.append(fig4)

    return figures


if __name__ == '__main__':
    """
    Demo script showing visualization capabilities.
    """
    print("=" * 60)
    print("Visualization Module Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)

    methods = ["Lead-3", "Random", "TextRank", "LexRank", "K-Means"]
    n_docs = 50

    # Simulate results
    results_data = []
    for i, method in enumerate(methods):
        mean_score = 0.30 + i * 0.02 + np.random.normal(0, 0.01)
        std_score = 0.05 + np.random.normal(0, 0.01)
        results_data.append({
            "Method": method,
            "N": n_docs,
            "rouge1_mean": mean_score,
            "rouge1_std": abs(std_score)
        })

    results_df = pd.DataFrame(results_data)

    print("\n1. Creating ROUGE comparison chart...")
    fig1 = plot_rouge_comparison(results_df, metric="rouge1")
    print("   Chart created successfully")
    plt.close(fig1)

    print("\n2. Creating score distribution plot...")
    scores = np.random.beta(5, 2, n_docs) * 0.5 + 0.2  # Simulate scores
    fig2 = plot_score_distribution(scores, "TextRank", plot_type="both")
    print("   Distribution plot created successfully")
    plt.close(fig2)

    print("\n3. Creating correlation heatmap...")
    corr_data = pd.DataFrame({
        "characteristic": ["num_words", "num_sentences", "lexical_diversity", "entity_density"],
        "correlation": [0.45, 0.32, -0.28, 0.15]
    })
    fig3 = plot_correlation_heatmap(corr_data)
    print("   Heatmap created successfully")
    plt.close(fig3)

    print("\n4. Creating pairwise comparison...")
    scores1 = np.random.beta(5, 2, n_docs) * 0.5 + 0.2
    scores2 = scores1 + np.random.normal(0, 0.05, n_docs)
    fig4 = plot_pairwise_comparison(scores1, scores2, "Method1", "Method2")
    print("   Pairwise comparison created successfully")
    plt.close(fig4)

    print("\n5. Exporting LaTeX table...")
    export_latex_table(
        results_df,
        "results_table.tex",
        caption="ROUGE-1 Scores for Different Methods"
    )

    print("\n" + "=" * 60)
    print("Demo completed! All visualizations generated successfully.")
    print("=" * 60)
