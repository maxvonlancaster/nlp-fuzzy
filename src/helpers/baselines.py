"""
Baseline summarization methods for evaluation and comparison.

This module provides standard baseline methods including:
- Lead-N: Select first N sentences
- Random: Random sentence selection
- TextRank: Graph-based ranking with TF-IDF similarity
- LexRank: Graph-based ranking with sentence embeddings
"""

import numpy as np
import nltk
import networkx as nx
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Download NLTK data if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Global random seed for reproducibility
RANDOM_SEED = 42


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.

    Args:
        text: Input text

    Returns:
        List of sentences (strings)
    """
    sentences = nltk.sent_tokenize(text)
    # Filter out very short or empty sentences
    sentences = [s.strip() for s in sentences if s and len(s.strip()) > 1]
    return sentences


def lead_n_baseline(text: str, n: int = 3, **kwargs) -> List[str]:
    """
    Lead-N baseline: Select the first N sentences.

    This is a surprisingly strong baseline for news articles where
    the most important information appears at the beginning.

    Args:
        text: Input text
        n: Number of sentences to select
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        List of first N sentences
    """
    sentences = split_sentences(text)
    n = min(n, len(sentences))
    return sentences[:n]


def random_baseline(text: str, n: int = 3, seed: int = RANDOM_SEED, **kwargs) -> List[str]:
    """
    Random baseline: Randomly select N sentences.

    Provides a lower bound for performance evaluation.

    Args:
        text: Input text
        n: Number of sentences to select
        seed: Random seed for reproducibility
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        List of N randomly selected sentences in original order
    """
    sentences = split_sentences(text)
    n = min(n, len(sentences))

    # Random selection with seed
    np.random.seed(seed)
    indices = np.random.choice(len(sentences), size=n, replace=False)

    # Sort indices to maintain original order
    indices = sorted(indices)

    return [sentences[i] for i in indices]


def sentence_similarity_matrix(
    sentences: List[str],
    method: str = "tfidf",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Compute sentence similarity matrix.

    Args:
        sentences: List of sentences
        method: Similarity method ('tfidf' or 'embeddings')
        embedding_model: SBERT model name (if method='embeddings')

    Returns:
        Similarity matrix (n_sentences × n_sentences)
    """
    if method == "tfidf":
        # TF-IDF based similarity
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 5),
            smooth_idf=True,
            use_idf=True,
            stop_words='english'
        )
        X = vectorizer.fit_transform(sentences)
        similarity_matrix = (X * X.T).toarray()

    elif method == "embeddings":
        # SBERT embedding-based similarity
        model = SentenceTransformer(embedding_model)
        embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings)

    else:
        raise ValueError(f"Unknown similarity method: {method}")

    return similarity_matrix


def textrank_summarizer(
    text: str,
    n: int = 3,
    similarity_metric: str = "tfidf",
    damping_factor: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-4,
    **kwargs
) -> List[str]:
    """
    TextRank algorithm for extractive summarization.

    Uses graph-based ranking to identify important sentences based on
    their similarity to other sentences. Sentences are nodes, edges
    are weighted by similarity.

    Args:
        text: Input text
        n: Number of sentences to select
        similarity_metric: 'tfidf' or 'embeddings'
        damping_factor: PageRank damping factor (default 0.85)
        max_iter: Maximum PageRank iterations
        tol: Convergence tolerance
        **kwargs: Additional arguments (for compatibility)

    Returns:
        List of top N sentences in original order

    Reference:
        Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into text.
        EMNLP 2004.
    """
    sentences = split_sentences(text)

    if len(sentences) <= n:
        return sentences

    # Compute similarity matrix
    similarity_matrix = sentence_similarity_matrix(
        sentences,
        method=similarity_metric,
        embedding_model=kwargs.get('embedding_model', 'all-MiniLM-L6-v2')
    )

    # Build graph from similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # Run PageRank
    scores = nx.pagerank(
        nx_graph,
        alpha=damping_factor,
        max_iter=max_iter,
        tol=tol
    )

    # Sort sentences by score
    ranked_sentences = sorted(
        ((scores[i], i, s) for i, s in enumerate(sentences)),
        reverse=True
    )

    # Select top N and return in original order
    selected_indices = sorted([idx for _, idx, _ in ranked_sentences[:n]])
    return [sentences[i] for i in selected_indices]


def lexrank_summarizer(
    text: str,
    n: int = 3,
    similarity_threshold: float = 0.1,
    embedding_model: str = "all-MiniLM-L6-v2",
    use_continuous: bool = True,
    damping_factor: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-4,
    **kwargs
) -> List[str]:
    """
    LexRank algorithm for extractive summarization.

    Similar to TextRank but uses continuous similarity weights and
    typically uses sentence embeddings rather than TF-IDF.

    Args:
        text: Input text
        n: Number of sentences to select
        similarity_threshold: Minimum similarity to create edge (if not continuous)
        embedding_model: SBERT model name
        use_continuous: Use continuous weights (True) or binary edges (False)
        damping_factor: PageRank damping factor
        max_iter: Maximum PageRank iterations
        tol: Convergence tolerance
        **kwargs: Additional arguments (for compatibility)

    Returns:
        List of top N sentences in original order

    Reference:
        Erkan, G., & Radev, D. R. (2004). LexRank: Graph-based lexical
        centrality as salience in text summarization. JAIR, 22, 457-479.
    """
    sentences = split_sentences(text)

    if len(sentences) <= n:
        return sentences

    # Compute similarity matrix using embeddings
    similarity_matrix = sentence_similarity_matrix(
        sentences,
        method="embeddings",
        embedding_model=embedding_model
    )

    if not use_continuous:
        # Binary edges: threshold similarity
        similarity_matrix = (similarity_matrix > similarity_threshold).astype(float)

    # Build graph
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # Run PageRank
    try:
        scores = nx.pagerank(
            nx_graph,
            alpha=damping_factor,
            max_iter=max_iter,
            tol=tol,
            weight='weight' if use_continuous else None
        )
    except nx.PowerIterationFailedConvergence:
        # Fallback: use simplified scoring
        scores = {i: similarity_matrix[i].sum() for i in range(len(sentences))}

    # Sort sentences by score
    ranked_sentences = sorted(
        ((scores[i], i, s) for i, s in enumerate(sentences)),
        reverse=True
    )

    # Select top N and return in original order
    selected_indices = sorted([idx for _, idx, _ in ranked_sentences[:n]])
    return [sentences[i] for i in selected_indices]


def get_summary_text(summary_sentences: List[str]) -> str:
    """
    Join summary sentences into a single text.

    Args:
        summary_sentences: List of sentences

    Returns:
        Concatenated summary text
    """
    return " ".join(summary_sentences)


# Mapping of method names to functions
BASELINE_METHODS = {
    "lead": lead_n_baseline,
    "random": random_baseline,
    "textrank": textrank_summarizer,
    "lexrank": lexrank_summarizer,
}


def summarize(
    text: str,
    method: str = "lead",
    n: int = 3,
    **kwargs
) -> List[str]:
    """
    Unified interface for all baseline methods.

    Args:
        text: Input text
        method: Method name ('lead', 'random', 'textrank', 'lexrank')
        n: Number of sentences to select
        **kwargs: Method-specific arguments

    Returns:
        List of selected sentences

    Raises:
        ValueError: If method is unknown

    Example:
        >>> summary = summarize(text, method='textrank', n=3)
    """
    if method not in BASELINE_METHODS:
        available = ', '.join(BASELINE_METHODS.keys())
        raise ValueError(
            f"Unknown method: {method}. Available methods: {available}"
        )

    method_func = BASELINE_METHODS[method]
    return method_func(text, n=n, **kwargs)


if __name__ == '__main__':
    """
    Demo script showing all baseline methods.
    """
    print("=" * 60)
    print("Baseline Methods Demo")
    print("=" * 60)

    # Test text
    test_text = """
    The Chernivtsi National University is a public university in the city of Chernivtsi in Western Ukraine.
    One of the leading Ukrainian institutions for higher education, it was founded in 1875 as the Franz-Josephs-Universität Czernowitz
    when Chernivtsi (Czernowitz) was the capital of the Duchy of Bukovina, a Cisleithanian crown land of Austria-Hungary.
    Today the university is based at the Residence of Bukovinian and Dalmatian Metropolitans building complex, a
    UNESCO World Heritage Site since 2011.
    Yuriy Fedkovych Chernivtsi National University consists of 17 buildings having a total of 105 units.
    The total area is 110,800 square meters, including training buildings of 66 square meters.
    The architectural ensemble of the main campus of the university, the Residence of Bukovinian and Dalmatian Metropolitans,
    is included on the list of UNESCO World Heritage Sites.
    The university operates Chernivtsi Botanical Garden, which features over a thousand different spices and an arboretum located
    on the territory of the main campus.
    """

    methods = ['lead', 'random', 'textrank', 'lexrank']

    for method_name in methods:
        print(f"\n{'=' * 60}")
        print(f"Method: {method_name.upper()}")
        print(f"{'=' * 60}")

        try:
            summary_sentences = summarize(test_text, method=method_name, n=2, seed=42)
            summary = get_summary_text(summary_sentences)
            print(f"\nSummary ({len(summary_sentences)} sentences):")
            print(summary)
        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'=' * 60}")
    print("Demo completed!")
    print(f"{'=' * 60}")
