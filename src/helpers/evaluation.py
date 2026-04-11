"""
Evaluation module for text summarization.

Provides ROUGE evaluation with statistical analysis including:
- Confidence intervals via bootstrapping
- Significance testing (paired t-test, Wilcoxon)
- Results aggregation and formatting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Optional, Tuple, Any
from rouge_score import rouge_scorer
from scipy import stats
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """
    Container for evaluation results with statistics.
    """
    method_name: str
    rouge_scores: Dict[str, List[float]]  # metric -> list of scores
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    num_documents: int


class RougeEvaluator:
    """
    ROUGE evaluation with statistical analysis.
    """

    def __init__(
        self,
        metrics: List[str] = None,
        use_stemmer: bool = True,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ):
        """
        Initialize ROUGE evaluator.

        Args:
            metrics: ROUGE metrics to compute (default: rouge1, rouge2, rougeL)
            use_stemmer: Use Porter stemmer for ROUGE
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level (0.95 = 95%)
            random_seed: Random seed for reproducibility
        """
        if metrics is None:
            metrics = ["rouge1", "rouge2", "rougeL"]

        self.metrics = metrics
        self.use_stemmer = use_stemmer
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.random_seed = random_seed

        # Initialize ROUGE scorer
        self.scorer = rouge_scorer.RougeScorer(
            self.metrics,
            use_stemmer=self.use_stemmer
        )

    def score_single(self, reference: str, summary: str) -> Dict[str, float]:
        """
        Compute ROUGE scores for a single document.

        Args:
            reference: Reference summary
            summary: Generated summary

        Returns:
            Dictionary mapping metric names to F-measure scores
        """
        scores = self.scorer.score(reference, summary)
        return {
            metric: scores[metric].fmeasure
            for metric in self.metrics
        }

    def score_batch(
        self,
        references: List[str],
        summaries: List[str]
    ) -> Dict[str, List[float]]:
        """
        Compute ROUGE scores for a batch of documents.

        Args:
            references: List of reference summaries
            summaries: List of generated summaries

        Returns:
            Dictionary mapping metric names to lists of scores
        """
        if len(references) != len(summaries):
            raise ValueError(
                f"Mismatch: {len(references)} references vs {len(summaries)} summaries"
            )

        # Initialize results
        results = {metric: [] for metric in self.metrics}

        # Compute scores for each document
        for ref, summ in zip(references, summaries):
            scores = self.score_single(ref, summ)
            for metric, score in scores.items():
                results[metric].append(score)

        return results


def compute_statistics(
    scores: List[float],
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Compute statistics for a list of scores.

    Args:
        scores: List of scores
        bootstrap_samples: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_seed: Random seed

    Returns:
        Dictionary with mean, std, min, max, median, and confidence interval
    """
    scores = np.array(scores)

    # Basic statistics
    stats_dict = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
    }

    # Bootstrap confidence interval
    np.random.seed(random_seed)
    bootstrap_means = []

    for _ in range(bootstrap_samples):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    stats_dict["ci_lower"] = float(ci_lower)
    stats_dict["ci_upper"] = float(ci_upper)

    return stats_dict


def evaluate_method(
    method_func: Callable,
    documents: List[str],
    references: List[str],
    method_name: str,
    evaluator: RougeEvaluator,
    method_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> EvaluationResult:
    """
    Evaluate a summarization method on a set of documents.

    Args:
        method_func: Summarization function (text, **kwargs) -> List[str]
        documents: List of documents to summarize
        references: List of reference summaries
        method_name: Name of the method for reporting
        evaluator: RougeEvaluator instance
        method_kwargs: Keyword arguments to pass to method_func
        verbose: Print progress

    Returns:
        EvaluationResult object
    """
    if method_kwargs is None:
        method_kwargs = {}

    if len(documents) != len(references):
        raise ValueError(
            f"Mismatch: {len(documents)} documents vs {len(references)} references"
        )

    if verbose:
        print(f"Evaluating {method_name} on {len(documents)} documents...")

    # Generate summaries
    summaries = []
    for i, doc in enumerate(documents):
        try:
            summary_sentences = method_func(doc, **method_kwargs)
            # Join sentences if method returns list
            if isinstance(summary_sentences, list):
                summary = " ".join(summary_sentences)
            else:
                summary = summary_sentences
            summaries.append(summary)
        except Exception as e:
            if verbose:
                print(f"  Warning: Error on document {i}: {e}")
            summaries.append("")  # Empty summary on error

    # Compute ROUGE scores
    rouge_scores = evaluator.score_batch(references, summaries)

    # Compute statistics for each metric
    mean_scores = {}
    std_scores = {}
    confidence_intervals = {}

    for metric, scores in rouge_scores.items():
        stats_dict = compute_statistics(
            scores,
            bootstrap_samples=evaluator.bootstrap_samples,
            confidence_level=evaluator.confidence_level,
            random_seed=evaluator.random_seed
        )
        mean_scores[metric] = stats_dict["mean"]
        std_scores[metric] = stats_dict["std"]
        confidence_intervals[metric] = (stats_dict["ci_lower"], stats_dict["ci_upper"])

    if verbose:
        print(f"  {method_name} - ROUGE-1: {mean_scores['rouge1']:.4f} ± {std_scores['rouge1']:.4f}")

    return EvaluationResult(
        method_name=method_name,
        rouge_scores=rouge_scores,
        mean_scores=mean_scores,
        std_scores=std_scores,
        confidence_intervals=confidence_intervals,
        num_documents=len(documents)
    )


def significance_test(
    scores1: List[float],
    scores2: List[float],
    test: str = "paired_t",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical significance test between two sets of scores.

    Args:
        scores1: Scores from method 1
        scores2: Scores from method 2
        test: Test type ('paired_t' or 'wilcoxon')
        alpha: Significance level

    Returns:
        Dictionary with test statistic, p-value, and significance
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have same length for paired test")

    if test == "paired_t":
        # Paired t-test
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        test_name = "Paired t-test"

    elif test == "wilcoxon":
        # Wilcoxon signed-rank test (non-parametric)
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        test_name = "Wilcoxon signed-rank test"

    else:
        raise ValueError(f"Unknown test: {test}")

    is_significant = p_value < alpha

    return {
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": is_significant,
        "interpretation": (
            f"Method difference IS statistically significant (p={p_value:.4f} < {alpha})"
            if is_significant
            else f"Method difference is NOT statistically significant (p={p_value:.4f} >= {alpha})"
        )
    }


def compare_methods(
    result1: EvaluationResult,
    result2: EvaluationResult,
    metric: str = "rouge1",
    test: str = "paired_t",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compare two methods using significance testing.

    Args:
        result1: EvaluationResult for method 1
        result2: EvaluationResult for method 2
        metric: ROUGE metric to compare
        test: Test type ('paired_t' or 'wilcoxon')
        alpha: Significance level

    Returns:
        Comparison results dictionary
    """
    scores1 = result1.rouge_scores[metric]
    scores2 = result2.rouge_scores[metric]

    sig_test = significance_test(scores1, scores2, test=test, alpha=alpha)

    return {
        "method1": result1.method_name,
        "method2": result2.method_name,
        "metric": metric,
        "mean1": result1.mean_scores[metric],
        "mean2": result2.mean_scores[metric],
        "difference": result2.mean_scores[metric] - result1.mean_scores[metric],
        **sig_test
    }


def generate_results_table(
    results: List[EvaluationResult],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate a formatted results table from evaluation results.

    Args:
        results: List of EvaluationResult objects
        metrics: Metrics to include (None for all)

    Returns:
        DataFrame with results
    """
    if not results:
        return pd.DataFrame()

    if metrics is None:
        metrics = list(results[0].mean_scores.keys())

    rows = []
    for result in results:
        row = {"Method": result.method_name, "N": result.num_documents}

        for metric in metrics:
            mean = result.mean_scores[metric]
            std = result.std_scores[metric]
            ci_lower, ci_upper = result.confidence_intervals[metric]

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci"] = f"[{ci_lower:.4f}, {ci_upper:.4f}]"

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_comparison_table(
    results: List[EvaluationResult],
    baseline_name: str,
    metric: str = "rouge1",
    test: str = "paired_t",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Generate pairwise comparison table against a baseline.

    Args:
        results: List of EvaluationResult objects
        baseline_name: Name of baseline method
        metric: ROUGE metric to compare
        test: Test type
        alpha: Significance level

    Returns:
        DataFrame with pairwise comparisons
    """
    # Find baseline result
    baseline = None
    for result in results:
        if result.method_name == baseline_name:
            baseline = result
            break

    if baseline is None:
        raise ValueError(f"Baseline method '{baseline_name}' not found")

    rows = []
    for result in results:
        if result.method_name == baseline_name:
            continue

        comparison = compare_methods(
            baseline, result, metric=metric, test=test, alpha=alpha
        )

        rows.append({
            "Method": result.method_name,
            f"{metric} (mean)": f"{result.mean_scores[metric]:.4f}",
            "Difference": f"{comparison['difference']:+.4f}",
            "p-value": f"{comparison['p_value']:.4f}",
            "Significant": "Yes" if comparison['significant'] else "No"
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    """
    Demo script showing evaluation capabilities.
    """
    print("=" * 60)
    print("Evaluation Framework Demo")
    print("=" * 60)

    # Create sample data
    documents = [
        "The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree. The fox ran away quickly.",
        "Machine learning is a subset of artificial intelligence. It enables computers to learn from data. Deep learning is a type of machine learning.",
        "Climate change is affecting the planet. Rising temperatures cause ice to melt. Sea levels are increasing globally."
    ]

    references = [
        "A fox jumps over a sleeping dog.",
        "Machine learning enables computers to learn from data.",
        "Climate change causes rising sea levels."
    ]

    # Simple dummy summarizer
    def dummy_summarizer(text, n=1, **kwargs):
        sentences = text.split('. ')
        return sentences[:n]

    # Initialize evaluator
    evaluator = RougeEvaluator(
        metrics=["rouge1", "rouge2", "rougeL"],
        bootstrap_samples=100  # Reduced for demo
    )

    print("\n1. Evaluating dummy summarizer...")
    result = evaluate_method(
        method_func=dummy_summarizer,
        documents=documents,
        references=references,
        method_name="Dummy (first sentence)",
        evaluator=evaluator,
        method_kwargs={"n": 1}
    )

    print(f"\nResults:")
    print(f"  ROUGE-1: {result.mean_scores['rouge1']:.4f} ± {result.std_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {result.mean_scores['rouge2']:.4f} ± {result.std_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {result.mean_scores['rougeL']:.4f} ± {result.std_scores['rougeL']:.4f}")

    print(f"\n2. Generating results table...")
    df = generate_results_table([result])
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
