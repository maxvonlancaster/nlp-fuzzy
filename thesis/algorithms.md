# Algorithm Descriptions and Complexity Analysis

This document provides detailed pseudocode and complexity analysis for all summarization methods evaluated in this research.

---

## 1. Lead-N Baseline

### Description
The Lead-N baseline selects the first N sentences from the document. Despite its simplicity, this is a surprisingly strong baseline for news articles where important information typically appears at the beginning (inverted pyramid style).

### Pseudocode

```
ALGORITHM: Lead-N Baseline
INPUT: text, n (number of sentences)
OUTPUT: summary (list of n sentences)

1. sentences ← TOKENIZE_SENTENCES(text)
2. IF length(sentences) ≤ n THEN
3.     RETURN sentences
4. ELSE
5.     RETURN sentences[0:n]
6. END IF
```

### Complexity Analysis

- **Time Complexity:** O(m)
  - m = number of words in text
  - Sentence tokenization: O(m)
  - Selection: O(1)

- **Space Complexity:** O(m)
  - Storage for sentence list: O(m)

**Advantages:**
- Extremely fast - O(m) time
- No computation required
- Preserves document structure
- Strong baseline for news and formal documents

**Limitations:**
- Assumes important information is at the beginning
- Fails for documents with delayed key information
- No semantic understanding

---

## 2. Random Baseline

### Description
Random baseline selects N sentences uniformly at random from the document. This serves as a lower bound for performance - any reasonable method should outperform random selection.

### Pseudocode

```
ALGORITHM: Random Baseline
INPUT: text, n (number of sentences), seed (random seed)
OUTPUT: summary (list of n sentences in original order)

1. sentences ← TOKENIZE_SENTENCES(text)
2. IF length(sentences) ≤ n THEN
3.     RETURN sentences
4. END IF
5. SET_RANDOM_SEED(seed)
6. indices ← RANDOM_SAMPLE(range(0, length(sentences)), n, replace=False)
7. indices ← SORT(indices)  // Maintain original order
8. RETURN [sentences[i] for i in indices]
```

### Complexity Analysis

- **Time Complexity:** O(m + n log n)
  - Sentence tokenization: O(m)
  - Random sampling: O(n)
  - Sorting indices: O(n log n)

- **Space Complexity:** O(m)
  - Storage for sentences: O(m)

**Purpose:**
- Establishes performance lower bound
- Any method with ROUGE scores below random is failing
- Provides statistical baseline for significance testing

---

## 3. TextRank

### Description
TextRank applies PageRank algorithm to sentences, treating them as nodes in a graph where edges represent similarity. Sentences with high centrality (connected to many other sentences) are considered important.

### Pseudocode

```
ALGORITHM: TextRank
INPUT: text, n (sentences), damping_factor=0.85, max_iter=100, tol=1e-4
OUTPUT: summary (list of n sentences in original order)

1. sentences ← TOKENIZE_SENTENCES(text)
2. IF length(sentences) ≤ n THEN
3.     RETURN sentences
4. END IF

5. // Build similarity matrix using TF-IDF
6. tfidf_matrix ← COMPUTE_TFIDF(sentences)
7. similarity_matrix ← tfidf_matrix · tfidf_matrix^T

8. // Build graph
9. graph ← EMPTY_GRAPH(num_nodes=length(sentences))
10. FOR i = 0 TO length(sentences)-1 DO
11.     FOR j = i+1 TO length(sentences)-1 DO
12.         weight ← similarity_matrix[i][j]
13.         IF weight > threshold THEN
14.             ADD_EDGE(graph, i, j, weight)
15.         END IF
16.     END FOR
17. END FOR

18. // Run PageRank
19. scores ← PAGERANK(graph, damping_factor, max_iter, tol)

20. // Select top N sentences
21. ranked_indices ← ARGSORT(scores, descending=True)
22. selected_indices ← SORT(ranked_indices[0:n])
23. RETURN [sentences[i] for i in selected_indices]
```

### Complexity Analysis

- **Time Complexity:** O(s² · d + s³)
  - s = number of sentences
  - d = average TF-IDF vector dimension (vocabulary size)
  - TF-IDF computation: O(s · w · v) where w = avg words/sentence, v = vocab size
  - Similarity matrix computation: O(s² · d)
  - PageRank iterations: O(k · s²) where k = number of iterations (typically ~100)
  - Dominant term: O(s² · d) for TF-IDF similarity, O(s³) for dense graphs

- **Space Complexity:** O(s² + s · d)
  - TF-IDF matrix: O(s · d)
  - Similarity matrix: O(s²)
  - Graph adjacency: O(s²) worst case

**Reference:**
Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into text. EMNLP, 2004.

---

## 4. LexRank

### Description
LexRank is similar to TextRank but uses continuous similarity weights and was originally designed with cosine similarity on TF-IDF vectors. In this implementation, it uses sentence embeddings from pre-trained transformers (SBERT) for better semantic similarity.

### Pseudocode

```
ALGORITHM: LexRank
INPUT: text, n, embedding_model, damping_factor=0.85, use_continuous=True
OUTPUT: summary (list of n sentences in original order)

1. sentences ← TOKENIZE_SENTENCES(text)
2. IF length(sentences) ≤ n THEN
3.     RETURN sentences
4. END IF

5. // Compute sentence embeddings
6. model ← LOAD_SENTENCE_TRANSFORMER(embedding_model)
7. embeddings ← model.ENCODE(sentences)

8. // Compute cosine similarity matrix
9. similarity_matrix ← COSINE_SIMILARITY(embeddings)

10. IF NOT use_continuous THEN
11.     // Binary edges (original LexRank)
12.     similarity_matrix ← (similarity_matrix > threshold)
13. END IF

14. // Build weighted graph
15. graph ← GRAPH_FROM_MATRIX(similarity_matrix)

16. // Run PageRank with continuous weights
17. scores ← PAGERANK(graph, damping_factor, max_iter, tol)

18. // Select top N sentences
19. ranked_indices ← ARGSORT(scores, descending=True)
20. selected_indices ← SORT(ranked_indices[0:n])
21. RETURN [sentences[i] for i in selected_indices]
```

### Complexity Analysis

- **Time Complexity:** O(s · d_emb + s² · d_emb + s³)
  - d_emb = embedding dimension (typically 384 for SBERT-Mini, 768 for SBERT-Base)
  - Embedding computation: O(s · w · T) where w = avg words/sentence, T = transformer forward pass ≈ O(w²)
    - In practice: O(s · w²) for SBERT inference
  - Cosine similarity: O(s² · d_emb)
  - PageRank: O(k · s²) where k ≈ 100 iterations
  - Dominant term: **O(s · w² · T) for embedding + O(s³) for PageRank**

- **Space Complexity:** O(s · d_emb + s²)
  - Embeddings matrix: O(s · d_emb)
  - Similarity matrix: O(s²)

**Note:** LexRank with SBERT embeddings is more expensive than TextRank due to transformer inference, but captures semantic similarity better than TF-IDF.

**Reference:**
Erkan, G., & Radev, D. R. (2004). LexRank: Graph-based lexical centrality as salience in text summarization. Journal of Artificial Intelligence Research, 22, 457-479.

---

## 5. LSA (Latent Semantic Analysis)

### Description
LSA uses Singular Value Decomposition (SVD) to identify latent topics in the TF-IDF matrix. Sentences with high representation in the most important topics are selected. This method works without PyTorch and is computationally efficient.

### Pseudocode

```
ALGORITHM: LSA Summarization
INPUT: text, n (sentences), n_components=5
OUTPUT: summary (list of n sentences in original order)

1. sentences ← TOKENIZE_SENTENCES(text)
2. IF length(sentences) ≤ n THEN
3.     RETURN sentences
4. END IF

5. // Create TF-IDF matrix
6. tfidf_matrix ← COMPUTE_TFIDF(sentences)  // Shape: (s × d)

7. // Apply Truncated SVD
8. n_components_actual ← MIN(n_components, MIN(s, d) - 1)
9. U, Σ, V^T ← TRUNCATED_SVD(tfidf_matrix, n_components_actual)
10. lsa_matrix ← U · Σ  // Sentence representations in topic space

11. // Compute sentence importance scores
12. FOR i = 0 TO length(sentences)-1 DO
13.     scores[i] ← ||lsa_matrix[i]||_2  // L2 norm across topics
14. END FOR

15. // Select top N sentences
16. ranked_indices ← ARGSORT(scores, descending=True)
17. selected_indices ← SORT(ranked_indices[0:n])
18. RETURN [sentences[i] for i in selected_indices]
```

### Complexity Analysis

- **Time Complexity:** O(s · d · k)
  - s = number of sentences
  - d = TF-IDF vocabulary dimension
  - k = number of SVD components (typically 5-10)
  - TF-IDF computation: O(s · w · v)
  - Truncated SVD: O(s · d · k) using iterative algorithms (Lanczos/Arnoldi)
  - Sentence scoring: O(s · k)
  - Dominant term: **O(s · d · k)**

- **Space Complexity:** O(s · d + s · k)
  - TF-IDF matrix: O(s · d), typically sparse
  - LSA matrix: O(s · k)

**Advantages over SBERT-based methods:**
- No PyTorch dependency
- Faster inference: O(s · d · k) vs O(s · w² · T)
- Works well for longer documents
- Captures topic-level importance

**Reference:**
Steinberger, J., & Ježek, K. (2004). Using latent semantic analysis in text summarization and summary evaluation. ISIM, 2004.

---

## Complexity Comparison Summary

| Method    | Time Complexity              | Space Complexity | Dependencies        | Semantic Understanding |
|-----------|------------------------------|------------------|---------------------|------------------------|
| Lead-N    | O(m)                         | O(m)             | NLTK                | None                   |
| Random    | O(m + n log n)               | O(m)             | NLTK                | None                   |
| TextRank  | O(s² · d + s³)               | O(s² + s · d)    | NLTK, sklearn, networkx | TF-IDF based       |
| LexRank   | O(s · w² + s² · d_emb + s³)  | O(s · d_emb + s²)| NLTK, sklearn, networkx, PyTorch, SBERT | Transformer-based |
| LSA       | O(s · d · k)                 | O(s · d + s · k) | NLTK, sklearn       | Topic-based            |

**Variables:**
- m = total words in document
- s = number of sentences
- w = average words per sentence
- d = TF-IDF vocabulary dimension
- d_emb = embedding dimension (384-768)
- k = number of SVD components (~5-10)

**Scalability Notes:**
1. **Lead-N and Random**: Scale linearly with document size - suitable for any length
2. **TextRank and LSA**: Scale quadratically with number of sentences - good for documents with <100 sentences
3. **LexRank (SBERT)**: Transformer inference dominates - expensive for long documents with many sentences

**Practical Performance (estimated for 500-word document ~30 sentences):**
- Lead-N: <1ms
- Random: <1ms
- TextRank: 10-50ms
- LSA: 20-80ms
- LexRank (SBERT): 100-500ms (depends on GPU availability)

---

## Implementation Notes

All methods are implemented in `src/helpers/baselines.py` with:
- Consistent API: `method(text, n, **kwargs) -> List[str]`
- Sentence order preservation in output
- Reproducible random seeding
- Fallback handling for edge cases

For full implementation details, see the source code and evaluation framework documentation.
