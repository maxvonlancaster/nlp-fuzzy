"""
Ablation study module for identifying key components and optimal parameters.

Provides systematic testing of:
- Embedding methods (Word2Vec, TF-IDF, SBERT variants)
- Dimensionality reduction (PCA, UMAP, None)
- Parameter sensitivity (K clusters, SOM grid size)
- Component contributions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import time

# Import evaluation framework
from .evaluation import RougeEvaluator, evaluate_method
from .baselines import split_sentences, get_summary_text

# Import ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
# sentence_transformers import moved to functions to avoid PyTorch DLL issues
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


@dataclass
class AblationResult:
    """Container for ablation study results."""
    config_name: str
    config_params: Dict[str, Any]
    rouge1_mean: float
    rouge1_std: float
    rouge2_mean: float
    rouge2_std: float
    rougeL_mean: float
    rougeL_std: float
    runtime: float
    num_documents: int


def embedding_ablation(
    documents: List[str],
    references: List[str],
    n_sentences: int = 3,
    embedding_methods: Optional[List[str]] = None,
    evaluator: Optional[RougeEvaluator] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare different embedding methods for K-Means clustering.

    Args:
        documents: List of documents to summarize
        references: List of reference summaries
        n_sentences: Number of sentences to extract
        embedding_methods: List of methods to test
        evaluator: RougeEvaluator instance
        verbose: Print progress

    Returns:
        DataFrame with ablation results

    Embedding methods tested:
    - 'tfidf': TF-IDF vectors
    - 'word2vec': Word2Vec mean pooling
    - 'sbert-mini': all-MiniLM-L6-v2
    - 'sbert-base': all-mpnet-base-v2
    """
    if embedding_methods is None:
        embedding_methods = ['tfidf', 'word2vec', 'sbert-mini']

    if evaluator is None:
        evaluator = RougeEvaluator(bootstrap_samples=100)

    if verbose:
        print("=" * 60)
        print("EMBEDDING ABLATION STUDY")
        print("=" * 60)

    results = []

    for embedding_method in embedding_methods:
        if verbose:
            print(f"\nTesting: {embedding_method}")

        start_time = time.time()

        # Create summarization function for this embedding
        def summarizer(text):
            sentences = split_sentences(text)
            if len(sentences) <= n_sentences:
                return sentences

            # Get embeddings based on method
            if embedding_method == 'tfidf':
                vectorizer = TfidfVectorizer(stop_words='english')
                embeddings = vectorizer.fit_transform(sentences).toarray()

            elif embedding_method == 'word2vec':
                # Train Word2Vec on sentences
                tokenized = [simple_preprocess(s) for s in sentences]
                model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1, workers=2, epochs=50)

                # Mean pooling
                embeddings = []
                for tokens in tokenized:
                    vectors = [model.wv[w] for w in tokens if w in model.wv]
                    if vectors:
                        embeddings.append(np.mean(vectors, axis=0))
                    else:
                        embeddings.append(np.zeros(100))
                embeddings = np.array(embeddings)

            elif embedding_method == 'sbert-mini':
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode(sentences, show_progress_bar=False)

            elif embedding_method == 'sbert-base':
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-mpnet-base-v2')
                embeddings = model.encode(sentences, show_progress_bar=False)

            else:
                raise ValueError(f"Unknown embedding method: {embedding_method}")

            # K-Means clustering
            k = min(n_sentences, len(sentences))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_ids = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_

            # Select representative sentences
            chosen_indices = []
            for cluster_id in range(k):
                cluster_indices = np.where(cluster_ids == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_embs = embeddings[cluster_indices]
                centroid = centroids[cluster_id].reshape(1, -1)
                sims = cosine_similarity(cluster_embs, centroid).flatten()
                best_idx = cluster_indices[np.argmax(sims)]
                chosen_indices.append(best_idx)

            # Return in original order
            chosen_indices = sorted(chosen_indices)
            return [sentences[i] for i in chosen_indices]

        # Evaluate
        result = evaluate_method(
            method_func=summarizer,
            documents=documents,
            references=references,
            method_name=embedding_method,
            evaluator=evaluator,
            verbose=False
        )

        runtime = time.time() - start_time

        ablation_result = AblationResult(
            config_name=embedding_method,
            config_params={'embedding': embedding_method},
            rouge1_mean=result.mean_scores['rouge1'],
            rouge1_std=result.std_scores['rouge1'],
            rouge2_mean=result.mean_scores['rouge2'],
            rouge2_std=result.std_scores['rouge2'],
            rougeL_mean=result.mean_scores['rougeL'],
            rougeL_std=result.std_scores['rougeL'],
            runtime=runtime,
            num_documents=len(documents)
        )

        results.append(ablation_result)

        if verbose:
            print(f"  ROUGE-1: {ablation_result.rouge1_mean:.4f} ± {ablation_result.rouge1_std:.4f}")
            print(f"  Runtime: {runtime:.2f}s")

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'config': r.config_name,
            'rouge1_mean': r.rouge1_mean,
            'rouge1_std': r.rouge1_std,
            'rouge2_mean': r.rouge2_mean,
            'rougeL_mean': r.rougeL_mean,
            'runtime_sec': r.runtime
        }
        for r in results
    ])

    # Sort by ROUGE-1
    df = df.sort_values('rouge1_mean', ascending=False).reset_index(drop=True)

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(df.to_string(index=False))

    return df


def dimensionality_reduction_ablation(
    documents: List[str],
    references: List[str],
    n_sentences: int = 3,
    embedding_method: str = 'sbert-mini',
    reduction_methods: Optional[List[str]] = None,
    evaluator: Optional[RougeEvaluator] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Test impact of dimensionality reduction on clustering performance.

    Args:
        documents: List of documents
        references: List of reference summaries
        n_sentences: Number of sentences to extract
        embedding_method: Embedding method to use
        reduction_methods: List of reduction methods ('none', 'pca', 'umap')
        evaluator: RougeEvaluator instance
        verbose: Print progress

    Returns:
        DataFrame with ablation results
    """
    if reduction_methods is None:
        reduction_methods = ['none', 'pca']

    if evaluator is None:
        evaluator = RougeEvaluator(bootstrap_samples=100)

    if verbose:
        print("=" * 60)
        print("DIMENSIONALITY REDUCTION ABLATION")
        print("=" * 60)

    results = []

    for reduction_method in reduction_methods:
        if verbose:
            print(f"\nTesting: {reduction_method}")

        start_time = time.time()

        def summarizer(text):
            sentences = split_sentences(text)
            if len(sentences) <= n_sentences:
                return sentences

            # Get embeddings
            if embedding_method == 'sbert-mini':
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode(sentences, show_progress_bar=False)
            else:
                raise ValueError(f"Embedding method {embedding_method} not supported in this ablation")

            # Apply dimensionality reduction
            if reduction_method == 'pca':
                n_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
                pca = PCA(n_components=n_components)
                embeddings = pca.fit_transform(embeddings)
                explained_var = pca.explained_variance_ratio_.sum()

            elif reduction_method == 'umap':
                try:
                    import umap
                    n_components = min(50, embeddings.shape[0] - 1)
                    reducer = umap.UMAP(n_components=n_components, random_state=42)
                    embeddings = reducer.fit_transform(embeddings)
                    explained_var = None
                except ImportError:
                    if verbose:
                        print("  Warning: UMAP not available, skipping")
                    return None

            else:  # 'none'
                explained_var = None

            # K-Means clustering
            k = min(n_sentences, len(sentences))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_ids = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_

            # Select representative sentences
            chosen_indices = []
            for cluster_id in range(k):
                cluster_indices = np.where(cluster_ids == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_embs = embeddings[cluster_indices]
                centroid = centroids[cluster_id].reshape(1, -1)
                sims = cosine_similarity(cluster_embs, centroid).flatten()
                best_idx = cluster_indices[np.argmax(sims)]
                chosen_indices.append(best_idx)

            chosen_indices = sorted(chosen_indices)
            return [sentences[i] for i in chosen_indices]

        # Evaluate
        result = evaluate_method(
            method_func=summarizer,
            documents=documents,
            references=references,
            method_name=reduction_method,
            evaluator=evaluator,
            verbose=False
        )

        if result is None:
            continue

        runtime = time.time() - start_time

        ablation_result = AblationResult(
            config_name=reduction_method,
            config_params={'reduction': reduction_method},
            rouge1_mean=result.mean_scores['rouge1'],
            rouge1_std=result.std_scores['rouge1'],
            rouge2_mean=result.mean_scores['rouge2'],
            rouge2_std=result.std_scores['rouge2'],
            rougeL_mean=result.mean_scores['rougeL'],
            rougeL_std=result.std_scores['rougeL'],
            runtime=runtime,
            num_documents=len(documents)
        )

        results.append(ablation_result)

        if verbose:
            print(f"  ROUGE-1: {ablation_result.rouge1_mean:.4f} ± {ablation_result.rouge1_std:.4f}")
            print(f"  Runtime: {runtime:.2f}s")

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'reduction': r.config_name,
            'rouge1_mean': r.rouge1_mean,
            'rouge1_std': r.rouge1_std,
            'rouge2_mean': r.rouge2_mean,
            'rougeL_mean': r.rougeL_mean,
            'runtime_sec': r.runtime
        }
        for r in results
    ])

    df = df.sort_values('rouge1_mean', ascending=False).reset_index(drop=True)

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(df.to_string(index=False))

    return df


def parameter_sensitivity(
    documents: List[str],
    references: List[str],
    param_name: str,
    param_range: List[Any],
    base_config: Optional[Dict[str, Any]] = None,
    evaluator: Optional[RougeEvaluator] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Test sensitivity to a specific parameter.

    Args:
        documents: List of documents
        references: List of reference summaries
        param_name: Parameter to vary ('n_sentences', 'k_clusters', 'som_size')
        param_range: List of parameter values to test
        base_config: Base configuration dictionary
        evaluator: RougeEvaluator instance
        verbose: Print progress

    Returns:
        DataFrame with results for each parameter value
    """
    if base_config is None:
        base_config = {'embedding': 'sbert-mini', 'method': 'kmeans'}

    if evaluator is None:
        evaluator = RougeEvaluator(bootstrap_samples=100)

    if verbose:
        print("=" * 60)
        print(f"PARAMETER SENSITIVITY: {param_name}")
        print("=" * 60)

    results = []

    for param_value in param_range:
        if verbose:
            print(f"\nTesting {param_name} = {param_value}")

        start_time = time.time()

        def summarizer(text):
            sentences = split_sentences(text)
            n = param_value if param_name == 'n_sentences' else base_config.get('n_sentences', 3)

            if len(sentences) <= n:
                return sentences

            # Get embeddings
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(sentences, show_progress_bar=False)

            # K-Means clustering
            k = param_value if param_name == 'k_clusters' else min(n, len(sentences))
            k = min(k, len(sentences))

            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_ids = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_

            # Select representatives
            chosen_indices = []
            for cluster_id in range(k):
                cluster_indices = np.where(cluster_ids == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_embs = embeddings[cluster_indices]
                centroid = centroids[cluster_id].reshape(1, -1)
                sims = cosine_similarity(cluster_embs, centroid).flatten()
                best_idx = cluster_indices[np.argmax(sims)]
                chosen_indices.append(best_idx)

            chosen_indices = sorted(chosen_indices)[:n]  # Limit to n sentences
            return [sentences[i] for i in chosen_indices]

        # Evaluate
        result = evaluate_method(
            method_func=summarizer,
            documents=documents,
            references=references,
            method_name=f"{param_name}={param_value}",
            evaluator=evaluator,
            verbose=False
        )

        runtime = time.time() - start_time

        results.append({
            param_name: param_value,
            'rouge1_mean': result.mean_scores['rouge1'],
            'rouge1_std': result.std_scores['rouge1'],
            'rouge2_mean': result.mean_scores['rouge2'],
            'rougeL_mean': result.mean_scores['rougeL'],
            'runtime_sec': runtime
        })

        if verbose:
            print(f"  ROUGE-1: {result.mean_scores['rouge1']:.4f}")

    df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(df.to_string(index=False))

    return df


def component_contribution_table(ablation_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate table showing contribution of each component.

    Args:
        ablation_results: Dictionary mapping ablation names to result DataFrames

    Returns:
        DataFrame summarizing component contributions
    """
    contributions = []

    for ablation_name, df in ablation_results.items():
        if len(df) < 2:
            continue

        # Compute range of ROUGE-1 scores
        rouge1_min = df['rouge1_mean'].min()
        rouge1_max = df['rouge1_mean'].max()
        rouge1_range = rouge1_max - rouge1_min

        # Find best and worst configurations
        best_config = df.iloc[0]['config'] if 'config' in df.columns else df.iloc[0][df.columns[0]]
        worst_config = df.iloc[-1]['config'] if 'config' in df.columns else df.iloc[-1][df.columns[0]]

        contributions.append({
            'component': ablation_name,
            'best_config': str(best_config),
            'best_rouge1': rouge1_max,
            'worst_config': str(worst_config),
            'worst_rouge1': rouge1_min,
            'contribution': rouge1_range,
            'relative_impact': rouge1_range / rouge1_max * 100 if rouge1_max > 0 else 0
        })

    df = pd.DataFrame(contributions)
    df = df.sort_values('contribution', ascending=False).reset_index(drop=True)

    return df


if __name__ == '__main__':
    """
    Demo script showing ablation study capabilities.
    """
    print("=" * 60)
    print("Ablation Study Demo")
    print("=" * 60)

    # Load small sample
    print("\nLoading sample data...")
    from .dataset_loader import load_mtsamples

    info, df_data = load_mtsamples(limit=20, seed=42)
    documents = df_data['transcription'].tolist()
    references = df_data['description'].tolist()

    print(f"Loaded {len(documents)} documents")

    # Test embedding ablation
    print("\n" + "=" * 60)
    print("1. EMBEDDING ABLATION")
    print("=" * 60)
    embedding_df = embedding_ablation(
        documents=documents,
        references=references,
        embedding_methods=['tfidf', 'sbert-mini'],
        verbose=True
    )

    # Test dimensionality reduction
    print("\n" + "=" * 60)
    print("2. DIMENSIONALITY REDUCTION ABLATION")
    print("=" * 60)
    reduction_df = dimensionality_reduction_ablation(
        documents=documents,
        references=references,
        reduction_methods=['none', 'pca'],
        verbose=True
    )

    # Test parameter sensitivity
    print("\n" + "=" * 60)
    print("3. PARAMETER SENSITIVITY")
    print("=" * 60)
    param_df = parameter_sensitivity(
        documents=documents,
        references=references,
        param_name='n_sentences',
        param_range=[2, 3, 4, 5],
        verbose=True
    )

    # Component contribution
    print("\n" + "=" * 60)
    print("4. COMPONENT CONTRIBUTION")
    print("=" * 60)
    ablation_results = {
        'embedding': embedding_df,
        'reduction': reduction_df,
        'n_sentences': param_df
    }
    contribution_df = component_contribution_table(ablation_results)
    print(contribution_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
