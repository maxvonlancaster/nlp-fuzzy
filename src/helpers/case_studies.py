"""
Case study module for qualitative analysis of summarization performance.

Provides detailed analysis of individual documents:
- Representative case selection (high/medium/low performers)
- Side-by-side method comparisons
- Error categorization and pattern detection
- Markdown export for thesis inclusion
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaseStudy:
    """Container for a case study."""
    doc_id: int
    performance_category: str  # 'high', 'medium', 'low'
    original_text: str
    reference_summary: str
    method_summaries: Dict[str, str]  # method_name -> summary
    method_scores: Dict[str, float]  # method_name -> ROUGE-1 score
    analysis: str
    word_count: int
    sentence_count: int


def select_representative_cases(
    documents: List[str],
    references: List[str],
    scores_dict: Dict[str, Dict[str, List[float]]],
    metric: str = 'rouge1',
    n_cases: int = 3
) -> List[int]:
    """
    Select representative cases covering performance spectrum.

    Args:
        documents: List of documents
        references: List of reference summaries
        scores_dict: Dictionary mapping method names to score dictionaries
        metric: ROUGE metric to use for selection
        n_cases: Number of cases to select (default 3: high/medium/low)

    Returns:
        List of document indices
    """
    if not scores_dict:
        raise ValueError("scores_dict cannot be empty")

    # Compute average score across all methods for each document
    method_names = list(scores_dict.keys())
    avg_scores = []

    for doc_idx in range(len(documents)):
        doc_scores = [
            scores_dict[method][metric][doc_idx]
            for method in method_names
            if metric in scores_dict[method]
        ]
        avg_scores.append(np.mean(doc_scores))

    avg_scores = np.array(avg_scores)

    # Select cases
    selected_indices = []

    if n_cases == 1:
        # Select median
        selected_indices.append(int(np.argsort(avg_scores)[len(avg_scores) // 2]))

    elif n_cases == 2:
        # Select high and low
        selected_indices.append(int(np.argmax(avg_scores)))
        selected_indices.append(int(np.argmin(avg_scores)))

    elif n_cases >= 3:
        # Select high, medium, low
        sorted_indices = np.argsort(avg_scores)

        # High performer
        selected_indices.append(int(sorted_indices[-1]))

        # Medium performer (median)
        selected_indices.append(int(sorted_indices[len(sorted_indices) // 2]))

        # Low performer
        selected_indices.append(int(sorted_indices[0]))

        # If more cases requested, add intermediate ones
        for i in range(3, n_cases):
            # Add cases at quartiles
            quartile_idx = int(len(sorted_indices) * (i - 2) / (n_cases - 2))
            selected_indices.append(int(sorted_indices[quartile_idx]))

    return selected_indices


def generate_case_study(
    doc_idx: int,
    document: str,
    reference: str,
    method_summaries: Dict[str, str],
    method_scores: Dict[str, float],
    performance_category: str,
    max_text_length: int = 500
) -> CaseStudy:
    """
    Generate a detailed case study for a document.

    Args:
        doc_idx: Document index
        document: Original document text
        reference: Reference summary
        method_summaries: Dictionary mapping method names to summaries
        method_scores: Dictionary mapping method names to ROUGE-1 scores
        performance_category: 'high', 'medium', or 'low'
        max_text_length: Maximum length of original text to include

    Returns:
        CaseStudy object
    """
    # Count words and sentences
    words = document.split()
    sentences = document.split('.')
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])

    # Truncate text if too long
    original_text = document
    if len(document.split()) > max_text_length:
        original_text = ' '.join(words[:max_text_length]) + '...'

    # Generate analysis
    analysis = _analyze_case(
        document=document,
        reference=reference,
        method_summaries=method_summaries,
        method_scores=method_scores,
        performance_category=performance_category
    )

    return CaseStudy(
        doc_id=doc_idx,
        performance_category=performance_category,
        original_text=original_text,
        reference_summary=reference,
        method_summaries=method_summaries,
        method_scores=method_scores,
        analysis=analysis,
        word_count=word_count,
        sentence_count=sentence_count
    )


def _analyze_case(
    document: str,
    reference: str,
    method_summaries: Dict[str, str],
    method_scores: Dict[str, float],
    performance_category: str
) -> str:
    """
    Generate analysis text for a case study.

    Args:
        document: Original document
        reference: Reference summary
        method_summaries: Method summaries
        method_scores: Method ROUGE scores
        performance_category: Performance category

    Returns:
        Analysis text
    """
    lines = []

    # Overall assessment
    avg_score = np.mean(list(method_scores.values()))
    lines.append(f"**Overall Performance:** {performance_category.capitalize()} (avg ROUGE-1: {avg_score:.3f})")
    lines.append("")

    # Best and worst methods
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
    best_method, best_score = sorted_methods[0]
    worst_method, worst_score = sorted_methods[-1]

    lines.append(f"**Best Method:** {best_method} (ROUGE-1: {best_score:.3f})")
    lines.append(f"**Worst Method:** {worst_method} (ROUGE-1: {worst_score:.3f})")
    lines.append(f"**Performance Spread:** {best_score - worst_score:.3f}")
    lines.append("")

    # Key observations
    lines.append("**Key Observations:**")

    if performance_category == 'high':
        lines.append(f"- All methods performed well, indicating the document is well-suited for extractive summarization")
        lines.append(f"- The reference summary likely contains sentences very similar to the source")

    elif performance_category == 'medium':
        lines.append(f"- Methods showed moderate performance with some variation")
        lines.append(f"- Different approaches captured different aspects of the content")

    elif performance_category == 'low':
        lines.append(f"- All methods struggled with this document")
        lines.append(f"- Possible causes: abstractive reference summary, complex content structure, or domain-specific language")

    # Check for agreement between methods
    summaries_list = list(method_summaries.values())
    if len(summaries_list) >= 2:
        # Simple agreement check: compare first sentences
        first_sentences = [s.split('.')[0] for s in summaries_list if s]
        unique_first = len(set(first_sentences))
        if unique_first == 1:
            lines.append(f"- Strong agreement: all methods selected the same first sentence")
        elif unique_first == len(summaries_list):
            lines.append(f"- No agreement: each method selected different content")
        else:
            lines.append(f"- Partial agreement: some methods converged on similar content")

    return '\n'.join(lines)


def categorize_errors(
    summaries: List[str],
    references: List[str],
    documents: List[str],
    method_name: str
) -> Dict[str, List[int]]:
    """
    Categorize errors in summaries.

    Args:
        summaries: Generated summaries
        references: Reference summaries
        documents: Original documents
        method_name: Name of the method

    Returns:
        Dictionary mapping error types to lists of document indices

    Error categories:
    - redundancy: Summary contains repetitive information
    - missing_key_info: Summary misses critical information from reference
    - length_mismatch: Summary too short or too long
    - poor_coverage: Low lexical overlap with reference
    """
    error_categories = {
        'redundancy': [],
        'missing_key_info': [],
        'length_mismatch': [],
        'poor_coverage': []
    }

    for idx, (summary, reference, document) in enumerate(zip(summaries, references, documents)):
        # Check for redundancy (repeated words beyond normal)
        summary_words = summary.lower().split()
        unique_ratio = len(set(summary_words)) / len(summary_words) if summary_words else 1.0
        if unique_ratio < 0.7:  # Less than 70% unique words
            error_categories['redundancy'].append(idx)

        # Check for missing key info (low lexical overlap with reference)
        ref_words = set(reference.lower().split())
        summary_words_set = set(summary_words)
        overlap = len(ref_words & summary_words_set) / len(ref_words) if ref_words else 0
        if overlap < 0.2:  # Less than 20% overlap
            error_categories['missing_key_info'].append(idx)

        # Check length mismatch
        ref_length = len(reference.split())
        summary_length = len(summary.split())
        if summary_length < ref_length * 0.5 or summary_length > ref_length * 2.0:
            error_categories['length_mismatch'].append(idx)

        # Check poor coverage (very short summary)
        if summary_length < 10:
            error_categories['poor_coverage'].append(idx)

    return error_categories


def export_case_study_markdown(
    case_studies: List[CaseStudy],
    filepath: str,
    title: str = "Summarization Case Studies"
) -> None:
    """
    Export case studies to markdown format.

    Args:
        case_studies: List of CaseStudy objects
        filepath: Output file path
        title: Document title
    """
    lines = [
        f"# {title}",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Number of cases: {len(case_studies)}",
        "",
        "---",
        ""
    ]

    for i, case in enumerate(case_studies, 1):
        lines.extend([
            f"## Case Study {i}: {case.performance_category.upper()} Performance",
            "",
            f"**Document ID:** {case.doc_id}",
            f"**Word Count:** {case.word_count}",
            f"**Sentence Count:** {case.sentence_count}",
            f"**Average ROUGE-1:** {np.mean(list(case.method_scores.values())):.4f}",
            "",
            "### Original Text (excerpt)",
            "",
            case.original_text,
            "",
            "### Reference Summary",
            "",
            case.reference_summary,
            "",
            "### Method Summaries and Scores",
            ""
        ])

        # Sort methods by score
        sorted_methods = sorted(case.method_scores.items(), key=lambda x: x[1], reverse=True)

        for method_name, score in sorted_methods:
            summary = case.method_summaries.get(method_name, "N/A")
            lines.extend([
                f"#### {method_name} (ROUGE-1: {score:.4f})",
                "",
                summary,
                ""
            ])

        lines.extend([
            "### Analysis",
            "",
            case.analysis,
            "",
            "---",
            ""
        ])

    # Write to file
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Case studies exported to: {filepath}")


def generate_error_report(
    error_categories: Dict[str, List[int]],
    method_name: str
) -> str:
    """
    Generate a text report of error patterns.

    Args:
        error_categories: Dictionary from categorize_errors()
        method_name: Name of the method

    Returns:
        Formatted error report
    """
    lines = [
        f"# Error Analysis: {method_name}",
        "",
        "## Error Categories",
        ""
    ]

    total_docs = max(max(indices) for indices in error_categories.values() if indices) + 1 if any(error_categories.values()) else 0

    for error_type, doc_indices in error_categories.items():
        count = len(doc_indices)
        percentage = (count / total_docs * 100) if total_docs > 0 else 0

        lines.append(f"### {error_type.replace('_', ' ').title()}")
        lines.append(f"- **Count:** {count} documents ({percentage:.1f}%)")
        if doc_indices:
            lines.append(f"- **Examples:** {', '.join(map(str, doc_indices[:5]))}")
        lines.append("")

    return '\n'.join(lines)


def compare_methods_on_case(
    case_study: CaseStudy,
    focus_methods: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate a comparison table for methods on a specific case.

    Args:
        case_study: CaseStudy object
        focus_methods: Methods to include (None for all)

    Returns:
        DataFrame with comparison
    """
    if focus_methods is None:
        focus_methods = list(case_study.method_scores.keys())

    rows = []
    for method in focus_methods:
        if method in case_study.method_scores:
            summary = case_study.method_summaries.get(method, "")
            score = case_study.method_scores[method]

            rows.append({
                'Method': method,
                'ROUGE-1': score,
                'Summary Length (words)': len(summary.split()),
                'First Sentence': summary.split('.')[0][:80] + '...' if summary else 'N/A'
            })

    df = pd.DataFrame(rows)
    df = df.sort_values('ROUGE-1', ascending=False).reset_index(drop=True)

    return df


if __name__ == '__main__':
    """
    Demo script showing case study capabilities.
    """
    print("=" * 60)
    print("Case Studies Demo")
    print("=" * 60)

    # Load sample data
    print("\n1. Loading sample data...")
    from .dataset_loader import load_mtsamples
    from .baselines import summarize
    from .evaluation import RougeEvaluator

    info, df_data = load_mtsamples(limit=30, seed=42)
    documents = df_data['transcription'].tolist()
    references = df_data['description'].tolist()

    print(f"   Loaded {len(documents)} documents")

    # Evaluate multiple methods
    print("\n2. Evaluating methods...")
    evaluator = RougeEvaluator(bootstrap_samples=50)

    methods = ['lead', 'textrank']
    scores_dict = {}
    summaries_dict = {}

    for method in methods:
        print(f"   Evaluating {method}...")
        summaries = []
        for doc in documents:
            summary_sentences = summarize(doc, method=method, n=3, seed=42)
            summaries.append(' '.join(summary_sentences))
        summaries_dict[method] = summaries

        # Compute scores
        rouge_scores = evaluator.score_batch(references, summaries)
        scores_dict[method] = rouge_scores

    # Select representative cases
    print("\n3. Selecting representative cases...")
    case_indices = select_representative_cases(
        documents=documents,
        references=references,
        scores_dict=scores_dict,
        n_cases=3
    )
    print(f"   Selected cases: {case_indices}")

    # Generate case studies
    print("\n4. Generating case studies...")
    case_studies = []

    categories = ['high', 'medium', 'low']
    for idx, category in zip(case_indices, categories):
        method_summaries = {method: summaries_dict[method][idx] for method in methods}
        method_scores = {method: scores_dict[method]['rouge1'][idx] for method in methods}

        case = generate_case_study(
            doc_idx=idx,
            document=documents[idx],
            reference=references[idx],
            method_summaries=method_summaries,
            method_scores=method_scores,
            performance_category=category
        )
        case_studies.append(case)

    print(f"   Generated {len(case_studies)} case studies")

    # Export to markdown
    print("\n5. Exporting to markdown...")
    export_case_study_markdown(
        case_studies=case_studies,
        filepath="case_studies_demo.md",
        title="Summarization Method Comparison"
    )

    # Error categorization
    print("\n6. Categorizing errors...")
    for method in methods:
        errors = categorize_errors(
            summaries=summaries_dict[method],
            references=references,
            documents=documents,
            method_name=method
        )
        print(f"\n   {method.upper()} errors:")
        for error_type, doc_list in errors.items():
            print(f"     {error_type}: {len(doc_list)} cases")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
