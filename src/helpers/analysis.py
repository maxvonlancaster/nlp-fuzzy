"""
Analysis module for understanding summarization performance.

Provides error analysis, variance explanation, and pattern detection:
- Document characteristics analysis
- Performance correlation analysis
- Failure pattern detection
- Per-document breakdown
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy import stats
import re


def document_characteristics(text: str) -> Dict[str, float]:
    """
    Compute characteristics of a document.

    Args:
        text: Input document

    Returns:
        Dictionary with document characteristics:
        - num_sentences: Number of sentences
        - num_words: Number of words
        - avg_sentence_length: Average sentence length in words
        - lexical_diversity: Type-token ratio
        - num_entities: Approximate named entity count (capitalized words)
        - entity_density: Entities per 100 words
        - compression_potential: Estimate of how compressible the text is
    """
    # Sentence count (simple split on period)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    num_sentences = len(sentences)

    # Word count
    words = text.split()
    num_words = len(words)

    # Average sentence length
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    # Lexical diversity (type-token ratio)
    unique_words = set(w.lower() for w in words)
    lexical_diversity = len(unique_words) / num_words if num_words > 0 else 0

    # Named entities (approximate: capitalized words that aren't at start of sentence)
    # This is a rough heuristic
    capitalized_words = [w for w in words if w and w[0].isupper()]
    # Exclude first word of each sentence
    sentence_starts = {s.split()[0] for s in sentences if s.split()}
    estimated_entities = [w for w in capitalized_words if w not in sentence_starts]
    num_entities = len(estimated_entities)
    entity_density = (num_entities / num_words * 100) if num_words > 0 else 0

    # Compression potential (heuristic based on redundancy)
    # Lower lexical diversity suggests higher compression potential
    compression_potential = 1 - lexical_diversity

    return {
        "num_sentences": float(num_sentences),
        "num_words": float(num_words),
        "avg_sentence_length": float(avg_sentence_length),
        "lexical_diversity": float(lexical_diversity),
        "num_entities": float(num_entities),
        "entity_density": float(entity_density),
        "compression_potential": float(compression_potential)
    }


def per_document_analysis(
    documents: List[str],
    references: List[str],
    scores: Dict[str, List[float]],
    method_name: str,
    metric: str = "rouge1"
) -> pd.DataFrame:
    """
    Analyze performance on individual documents.

    Args:
        documents: List of documents
        references: List of reference summaries
        scores: Dictionary of ROUGE scores (metric -> list)
        method_name: Name of the method
        metric: ROUGE metric to analyze

    Returns:
        DataFrame with per-document analysis
    """
    if metric not in scores:
        raise ValueError(f"Metric '{metric}' not found in scores")

    doc_scores = scores[metric]

    rows = []
    for i, (doc, ref, score) in enumerate(zip(documents, references, doc_scores)):
        char = document_characteristics(doc)
        row = {
            "doc_id": i,
            "method": method_name,
            f"{metric}_score": score,
            **char
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add performance category
    q25 = df[f"{metric}_score"].quantile(0.25)
    q75 = df[f"{metric}_score"].quantile(0.75)

    def categorize(score):
        if score >= q75:
            return "high"
        elif score <= q25:
            return "low"
        else:
            return "medium"

    df["performance"] = df[f"{metric}_score"].apply(categorize)

    return df


def analyze_variance(
    df: pd.DataFrame,
    metric: str = "rouge1"
) -> Dict[str, Any]:
    """
    Analyze what factors contribute to score variance.

    Args:
        df: DataFrame from per_document_analysis
        metric: ROUGE metric column name

    Returns:
        Dictionary with variance analysis results
    """
    score_col = f"{metric}_score"

    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in DataFrame")

    # Overall variance
    variance = df[score_col].var()
    std = df[score_col].std()
    range_scores = df[score_col].max() - df[score_col].min()

    # Identify numeric characteristic columns
    char_cols = [col for col in df.columns
                 if col not in ["doc_id", "method", score_col, "performance"]
                 and pd.api.types.is_numeric_dtype(df[col])]

    # Compute correlations
    correlations = {}
    for col in char_cols:
        if df[col].std() > 0:  # Only if there's variance
            corr = df[col].corr(df[score_col])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return {
        "variance": float(variance),
        "std": float(std),
        "range": float(range_scores),
        "min_score": float(df[score_col].min()),
        "max_score": float(df[score_col].max()),
        "correlations": dict(sorted_corrs),
        "strongest_predictor": sorted_corrs[0] if sorted_corrs else None
    }


def failure_pattern_detection(
    df: pd.DataFrame,
    metric: str = "rouge1",
    threshold_percentile: float = 25
) -> Dict[str, Any]:
    """
    Detect patterns in low-performing documents.

    Args:
        df: DataFrame from per_document_analysis
        metric: ROUGE metric column name
        threshold_percentile: Percentile below which documents are "failures"

    Returns:
        Dictionary with failure pattern analysis
    """
    score_col = f"{metric}_score"
    threshold = df[score_col].quantile(threshold_percentile / 100)

    low_performers = df[df[score_col] <= threshold]
    high_performers = df[df[score_col] >= df[score_col].quantile(0.75)]

    if len(low_performers) == 0:
        return {"message": "No low performers found"}

    # Compare characteristics of low vs high performers
    char_cols = [col for col in df.columns
                 if col not in ["doc_id", "method", score_col, "performance"]
                 and pd.api.types.is_numeric_dtype(df[col])]

    patterns = {}
    for col in char_cols:
        low_mean = low_performers[col].mean()
        high_mean = high_performers[col].mean()

        if high_mean > 0:
            difference_pct = ((low_mean - high_mean) / high_mean) * 100
        else:
            difference_pct = 0

        # Statistical test
        if len(low_performers) > 1 and len(high_performers) > 1:
            t_stat, p_value = stats.ttest_ind(
                low_performers[col].dropna(),
                high_performers[col].dropna()
            )
        else:
            t_stat, p_value = None, None

        patterns[col] = {
            "low_mean": float(low_mean),
            "high_mean": float(high_mean),
            "difference_pct": float(difference_pct),
            "p_value": float(p_value) if p_value is not None else None
        }

    return {
        "threshold_score": float(threshold),
        "num_low_performers": len(low_performers),
        "num_high_performers": len(high_performers),
        "patterns": patterns
    }


def correlation_analysis(
    df: pd.DataFrame,
    metric: str = "rouge1"
) -> pd.DataFrame:
    """
    Compute correlation matrix between document characteristics and scores.

    Args:
        df: DataFrame from per_document_analysis
        metric: ROUGE metric column name

    Returns:
        DataFrame with correlation coefficients and p-values
    """
    score_col = f"{metric}_score"

    # Identify numeric columns
    char_cols = [col for col in df.columns
                 if col not in ["doc_id", "method", score_col, "performance"]
                 and pd.api.types.is_numeric_dtype(df[col])]

    rows = []
    for col in char_cols:
        if df[col].std() > 0:  # Only if there's variance
            corr_coef = df[col].corr(df[score_col])

            # Pearson correlation with p-value
            corr, p_value = stats.pearsonr(df[col].dropna(), df[score_col].dropna())

            rows.append({
                "characteristic": col,
                "correlation": corr_coef,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "interpretation": "positive" if corr_coef > 0 else "negative"
            })

    result_df = pd.DataFrame(rows)

    # Sort by absolute correlation
    result_df["abs_correlation"] = result_df["correlation"].abs()
    result_df = result_df.sort_values("abs_correlation", ascending=False)
    result_df = result_df.drop("abs_correlation", axis=1)

    return result_df


def compare_methods_by_difficulty(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str = "rouge1"
) -> pd.DataFrame:
    """
    Compare two methods across documents of different difficulty.

    Args:
        df1: DataFrame for method 1 (from per_document_analysis)
        df2: DataFrame for method 2
        metric: ROUGE metric

    Returns:
        DataFrame comparing methods by difficulty level
    """
    score_col = f"{metric}_score"

    # Assume both dataframes have same documents in same order
    if len(df1) != len(df2):
        raise ValueError("DataFrames must have same number of documents")

    # Categorize by difficulty based on average performance
    avg_scores = (df1[score_col] + df2[score_col]) / 2
    q25 = avg_scores.quantile(0.25)
    q75 = avg_scores.quantile(0.75)

    def categorize_difficulty(score):
        if score >= q75:
            return "easy"
        elif score <= q25:
            return "hard"
        else:
            return "medium"

    difficulty = avg_scores.apply(categorize_difficulty)

    # Compare methods by difficulty
    comparison = []
    for diff_level in ["easy", "medium", "hard"]:
        mask = difficulty == diff_level
        if mask.sum() == 0:
            continue

        method1_mean = df1[mask][score_col].mean()
        method2_mean = df2[mask][score_col].mean()

        comparison.append({
            "difficulty": diff_level,
            "count": mask.sum(),
            f"{df1['method'].iloc[0]}_mean": method1_mean,
            f"{df2['method'].iloc[0]}_mean": method2_mean,
            "difference": method2_mean - method1_mean
        })

    return pd.DataFrame(comparison)


def generate_analysis_report(
    df: pd.DataFrame,
    metric: str = "rouge1",
    method_name: str = "Method"
) -> str:
    """
    Generate a text report summarizing the analysis.

    Args:
        df: DataFrame from per_document_analysis
        metric: ROUGE metric
        method_name: Name of the method

    Returns:
        Formatted text report
    """
    score_col = f"{metric}_score"

    # Variance analysis
    var_analysis = analyze_variance(df, metric)

    # Failure pattern detection
    failure_patterns = failure_pattern_detection(df, metric)

    # Correlation analysis
    corr_df = correlation_analysis(df, metric)

    lines = [
        f"=" * 60,
        f"Analysis Report: {method_name}",
        f"Metric: {metric.upper()}",
        f"=" * 60,
        "",
        "1. SCORE DISTRIBUTION",
        f"   Mean: {df[score_col].mean():.4f}",
        f"   Std:  {var_analysis['std']:.4f}",
        f"   Range: [{var_analysis['min_score']:.4f}, {var_analysis['max_score']:.4f}]",
        f"   Variance explained range: {var_analysis['range']:.4f}",
        "",
        "2. TOP PREDICTORS OF PERFORMANCE",
    ]

    for i, row in corr_df.head(5).iterrows():
        lines.append(
            f"   {row['characteristic']:25s}: r={row['correlation']:+.3f} "
            f"(p={'<0.001' if row['p_value'] < 0.001 else f'{row["p_value"]:.3f}'})"
        )

    lines.extend([
        "",
        "3. FAILURE PATTERNS (Bottom 25%)",
        f"   Threshold: {failure_patterns.get('threshold_score', 'N/A')}",
        f"   Number of low performers: {failure_patterns.get('num_low_performers', 0)}",
        ""
    ])

    if "patterns" in failure_patterns:
        lines.append("   Characteristics of low vs high performers:")
        for char, stats in list(failure_patterns["patterns"].items())[:5]:
            diff = stats["difference_pct"]
            lines.append(f"   {char:25s}: {diff:+.1f}% difference")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == '__main__':
    """
    Demo script showing analysis capabilities.
    """
    print("=" * 60)
    print("Analysis Module Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)

    documents = [
        "Short text. Very brief." * i for i in range(1, 11)
    ]
    references = ["Reference summary"] * 10

    # Simulate scores with some correlation to document length
    scores = {
        "rouge1": [0.2 + 0.04 * i + np.random.normal(0, 0.05) for i in range(10)]
    }

    print("\n1. Computing document characteristics...")
    char = document_characteristics(documents[5])
    print(f"   Sample document characteristics:")
    for key, value in char.items():
        print(f"     {key}: {value:.2f}")

    print("\n2. Per-document analysis...")
    df = per_document_analysis(documents, references, scores, "Test Method")
    print(f"   Created DataFrame with {len(df)} documents")
    print(f"   Performance categories: {df['performance'].value_counts().to_dict()}")

    print("\n3. Variance analysis...")
    var_analysis = analyze_variance(df)
    print(f"   Score variance: {var_analysis['variance']:.4f}")
    print(f"   Score range: {var_analysis['range']:.4f}")
    if var_analysis['strongest_predictor']:
        char_name, corr = var_analysis['strongest_predictor']
        print(f"   Strongest predictor: {char_name} (r={corr:.3f})")

    print("\n4. Generating analysis report...")
    report = generate_analysis_report(df, method_name="Test Method")
    print(report)

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
