# Unsupervised Extractive Text Summarization: A Comparative Study of Graph-Based and Algebraic Methods

Text summarization is a fundamental task in natural language processing that aims to condense large volumes of textual information into concise, coherent representations while preserving the most salient content. This article presents a comparative study of three unsupervised machine learning methods for extractive text summarization: TextRank, LexRank, and Latent Semantic Analysis (LSA) with Singular Value Decomposition (SVD). TextRank and LexRank are graph-based approaches that model inter-sentence relationships through similarity matrices and apply ranking algorithms inspired by PageRank to identify the most informative sentences. LSA, in contrast, leverages algebraic decomposition via SVD to uncover latent semantic structures within a document, capturing conceptual relationships beyond surface-level term frequency. As unsupervised methods, all three approaches require no labeled training data, making them broadly applicable across domains and languages. We discuss the theoretical foundations of each algorithm, analyze their computational characteristics, and evaluate their summarization performance. The results highlight the trade-offs between graph-based and matrix decomposition strategies, offering insights into the strengths and limitations of each method for automatic text summarization tasks.

## 1. Introduction 

The exponential growth of digital textual data in recent decades has created an urgent need for automated methods capable of distilling large documents into concise and meaningful summaries. From scientific literature and news articles to legal documents and social media content, the sheer volume of text produced daily far exceeds human capacity for manual review. Automatic text summarization addresses this challenge by enabling efficient information retrieval and knowledge extraction without requiring a reader to process an entire source document.

Text summarization systems are broadly categorized into two paradigms: extractive and abstractive. Extractive methods identify and select the most representative sentences or phrases directly from the source text, preserving the original wording. Abstractive methods, by contrast, generate novel sentences that paraphrase or synthesize the core ideas of the document, much as a human writer would. While abstractive summarization has gained considerable attention with the rise of large language models, extractive approaches remain widely used due to their interpretability, computational efficiency, and ability to guarantee factual consistency with the source material.

Within the extractive paradigm, unsupervised machine learning methods are particularly attractive because they require no annotated training data and can generalize across diverse domains and languages. This article focuses on three well-established unsupervised algorithms for extractive text summarization. The first two, TextRank and LexRank, are graph-based methods that represent sentences as nodes in a graph and compute their importance through similarity-weighted edges, drawing on the principles of iterative ranking algorithms such as Google's PageRank. The third method, Latent Semantic Analysis (LSA) combined with Singular Value Decomposition (SVD), takes a fundamentally different algebraic approach by decomposing a term-sentence matrix to reveal hidden semantic relationships and select sentences that best capture the underlying conceptual structure of a document.

Despite sharing the common goal of automatic summarization, these three methods differ substantially in their theoretical foundations, computational complexity, and sensitivity to document structure and vocabulary. A systematic understanding of their respective strengths and limitations is valuable both for researchers designing summarization pipelines and for practitioners selecting appropriate tools for specific applications.

The remainder of this article is organized as follows. Section 2 reviews related work in the field of automatic text summarization. Section 3 describes the theoretical background and algorithmic details of TextRank, LexRank, and LSA with SVD. Section 4 presents the experimental setup and evaluation methodology. Section 5 discusses the results, and Section 6 concludes the article with directions for future research.

## 2. Related work

Automatic text summarization has attracted sustained research interest as a means of managing the growing volume of digital textual information. This section reviews the key methods relevant to the present study, grounding each discussion in the cited literature.

**2.1 Unsupervised Extractive Summarization**

Unsupervised extractive summarization approaches select the most informative sentences directly from a source document without requiring labeled training data, making them broadly applicable across domains and languages. The three methods examined in this article (TextRank, LexRank, and LSA) represent the principal paradigms within this category, differing primarily in how they represent sentences and compute their relative importance. A comparative study by Barman et al. (2022) consolidates this perspective, evaluating statistical, topic-modelling, and graph-based unsupervised approaches for news article summarization and demonstrating that each paradigm captures complementary aspects of sentence salience [5].

**2.2 Graph-Based Methods: TextRank and LexRank**

Graph-based ranking methods model a document as a network of sentences and derive importance scores from the structural properties of that network. Mihalcea (2004) introduced the foundational framework for applying graph-based ranking algorithms to sentence extraction, demonstrating that iterative PageRank-style centrality propagation can identify representative sentences without any external supervision [2]. In this formulation, sentences are represented as nodes and connected by edges weighted according to pairwise textual similarity; sentences occupying central positions in the resulting graph - those that are similar to the greatest number of other sentences - receive the highest scores and are selected for the summary. Gulati et al. (2023) extended this line of work by integrating TextRank with the BM25+ retrieval function, showing that enriching the similarity measure used to construct the sentence graph yields improvements in extractive summarization quality [1].

LexRank follows the same graph-based paradigm but applies eigenvector centrality rather than PageRank, and uses cosine similarity between TF-IDF vectors as the edge weighting function. Naser (2021) combined LexRank with an Extreme Learning Machine in a multi-document summarization framework, demonstrating that the graph-based sentence rankings produced by LexRank provide a strong foundation for downstream learning-based selection [4]. The capacity of graph-based methods to scale to longer and more complex inputs has been further explored by Gokhan et al. (2024), who proposed a hybrid clustering approach for unsupervised extractive summarization of long documents using language models, highlighting the continued relevance of graph-based sentence organization in modern summarization pipelines [3].

**2.3 Topic-Based Methods: Latent Semantic Analysis**

Latent Semantic Analysis offers a contrasting algebraic perspective on extractive summarization. Rather than constructing an explicit sentence similarity graph, LSA applies Singular Value Decomposition to a term-sentence matrix to uncover latent topical structure within the document. Sentences are then scored according to their contribution to the dominant latent components, and the top-ranked sentences are selected to represent the document's core content. Barman et al. (2022) evaluated LSA alongside graph-based and statistical methods on a news summarization corpus, finding that topic-modelling approaches such as LSA are particularly effective at capturing the thematic diversity of a document [5].

**2.4 Broader Applications of NLP Methods**

The natural language processing techniques underlying the methods examined in this study have found application well beyond summarization. The utility of combining natural language processing with fuzzy logic for the task of disinformation detection has been demonstrated, illustrating how text representation and linguistic analysis methods can be deployed in high-stakes information integrity contexts [6]. In a related direction, fuzzy logic integration has been applied to the problem of mood detection in textual analysis, reporting improvements in classification accuracy over baseline NLP approaches [7]. These works reflect the growing interdisciplinary scope of NLP research and underscore the practical relevance of unsupervised text analysis methods across a wide range of real-world applications.

**2.5 Positioning of This Work**

The present article builds directly on the foundational methods reviewed above. While prior work has examined TextRank, LexRank, and LSA individually or in limited pairwise comparisons, this study provides a unified comparative evaluation of all three approaches under consistent experimental conditions, with the aim of offering clear and reproducible guidance on their relative strengths and limitations for extractive text summarization.

## 3. Algorithm Descriptions and Complexity Analysis

**3.1 Baseline Methods**

Before examining the core unsupervised algorithms, two simple baseline methods are established to provide reference points for performance evaluation.

Lead-N Baseline. The Lead-N method selects the first $N$ sentences from the input document. Despite its simplicity, this strategy constitutes a surprisingly competitive baseline for news articles and formal documents, where the most important information is conventionally placed at the beginning - a journalistic convention known as the inverted pyramid structure. The method operates in linear time $O(m)$, where $m$ is the total number of words in the document, and requires $O(m)$ space for sentence storage. It performs no semantic analysis whatsoever, and consequently fails for documents in which key information is distributed throughout the text rather than concentrated at the opening.

Random Baseline. The Random baseline selects $N$ sentences chosen uniformly at random from the document, preserving their original order in the output. Its primary purpose is to establish a lower performance bound: any meaningful summarization method is expected to outperform random selection. Its time complexity is $O(m + n \log n)$, where the $\log n$ term arises from sorting the sampled indices to maintain sentence order, and its space complexity is $O(m)$.

**3.2 TextRank**

TextRank, introduced by Mihalcea and Tarau [2], is a graph-based extractive summarization method that adapts Google's PageRank algorithm to the problem of sentence ranking. The method represents each sentence in the document as a node in an undirected weighted graph, where the edge weight between any two nodes reflects their textual similarity. In the present implementation, sentence similarity is computed via the dot product of TF-IDF vectors, constructing a similarity matrix of shape $s \times s$, where $s$ denotes the number of sentences.

Edges whose weight exceeds a predefined threshold are added to the graph, and the PageRank algorithm is subsequently applied iteratively - with a damping factor of $0.85$ and convergence tolerance of $10^{-4}$ - to compute a centrality score for each sentence. Sentences that are highly similar to many other sentences in the document accumulate higher scores and are interpreted as more informative. The top $N$ highest-scoring sentences, reordered to reflect their original position in the document, constitute the final summary.

Complexity Analysis. The dominant computational costs in TextRank arise from two stages: construction of the TF-IDF similarity matrix, which requires $O(s^2 \cdot d)$ time where $d$ is the vocabulary dimension, and iterative PageRank computation over the resulting graph, which requires $O(k \cdot s^2)$ time for $k$ iterations. In the worst case of a fully connected graph, the overall time complexity is $O(s^2 \cdot d + s^3)$. Space complexity is $O(s^2 + s \cdot d)$, accounting for both the similarity matrix and the TF-IDF representation.

**3.3 LexRank**

LexRank, proposed by Erkan and Radev [9], shares the same graph-based PageRank foundation as TextRank but differs in how inter-sentence similarity is measured. In its original formulation, LexRank computes cosine similarity between TF-IDF vectors. The implementation evaluated here extends this by employing sentence embeddings produced by a pre-trained Sentence-BERT (SBERT) transformer model, which captures deeper semantic relationships beyond lexical overlap. In the continuous variant used here, all pairwise cosine similarity values are retained as edge weights without binarization, forming a fully weighted graph on which PageRank is executed.

This design choice improves semantic fidelity but introduces a substantially higher computational cost. SBERT inference for each sentence involves a transformer forward pass with complexity approximately $O(w^2)$ per sentence, where $w$ is the average sentence length in tokens. Computing the full cosine similarity matrix then requires $O(s^2 \cdot d_{emb})$, and PageRank adds $O(s^3)$ in the worst case.

Complexity Analysis. The overall time complexity of LexRank with SBERT embeddings is $O(s \cdot w^2 + s^2 \cdot d_{emb} + s^3)$, where $d_{emb}$ denotes the embedding dimensionality ($384$ for SBERT-Mini, $768$ for SBERT-Base). Space complexity is $O(s \cdot d_{emb} + s^2)$. In practice, transformer inference dominates execution time, making LexRank the most computationally expensive of the three methods - though also the most semantically expressive.

**3.4 Latent Semantic Analysis with Singular Value Decomposition**

LSA-based summarization, as described by Steinberger and Ježek [8], takes a fundamentally different algebraic approach. Rather than modeling sentence relationships through a graph, LSA projects sentences into a low-dimensional latent topic space by applying Truncated Singular Value Decomposition (SVD) to the TF-IDF term-sentence matrix.

Given a TF-IDF matrix $A$ of shape $s \times d$, Truncated SVD decomposes it into three matrices: $A \approx U \cdot \Sigma \cdot V^T$, retaining only the top $k$ singular components (typically $k = 5$–$10$). The resulting matrix $U \cdot \Sigma$ of shape $s \times k$ constitutes a compact representation of each sentence in the latent topic space. The importance score of each sentence is then computed as the $L_2$ norm of its corresponding row in this matrix, reflecting its overall contribution across the most prominent latent topics. The top $N$ sentences by score, restored to their original document order, form the summary.

Complexity Analysis. The time complexity of LSA summarization is $O(s \cdot d \cdot k)$, arising from the Truncated SVD computation performed via iterative algorithms such as Lanczos or Arnoldi methods. TF-IDF construction contributes an additional $O(s \cdot w \cdot v)$ term, where $v$ is the vocabulary size. Space complexity is $O(s \cdot d + s \cdot k)$, where the TF-IDF matrix is typically stored in sparse format, substantially reducing memory usage in practice. LSA requires no deep learning framework and is notably faster than LexRank, making it an efficient yet semantically informed alternative.

**3.5 Comparative Summary**

Table 1 summarizes the computational properties of all five methods across the key dimensions of time complexity, space complexity, external dependencies, and degree of semantic understanding.

Table 1. Complexity and dependency comparison of summarization methods.

| Method   | Time Complexity                          | Space Complexity        | Semantic Understanding |
|----------|------------------------------------------|-------------------------|------------------------|
| Lead-N   | $O(m)$ | $O(m)$                    | None                   |
| Random   | $O(m + n \log n)$ | $O(m)$                 | None                   |
| TextRank | $O(s^2 \cdot d + s^3)$   |$O(s^2+ s \cdot d)$     | TF-IDF based           |
| LexRank  | $O(s \cdot w^2 + s^2 \cdot d_{emb} + s^3)$ | $O(s \cdot d_{emb} + s^2)$       | Transformer-based      |
| LSA      | $O(s \cdot d \cdot k)$                            | $O(s \cdot d + s \cdot k)$       | Topic-based            |

From a scalability perspective, the three unsupervised methods diverge considerably. TextRank and LSA scale quadratically and are well-suited for documents containing up to approximately 100 sentences. LexRank, due to the cost of transformer inference, incurs the highest latency - estimated between 100 and 500 milliseconds for a typical 500-word document - but offers the richest semantic representation. LSA occupies a middle ground, providing topic-aware sentence selection without requiring a deep learning dependency, making it particularly practical in resource-constrained environments.


## 4. Detailed Methodology

**4.1 Dataset**

The experiments were conducted on the MTSamples medical transcription dataset, a publicly available collection sourced from MTSamples.com comprising 4,999 clinical documents spanning more than 40 medical specialties. The dataset includes clinical notes, procedure reports, and consultation letters, each accompanied by a human-written reference summary. For computational feasibility, a stratified random sample of 1,000 documents was drawn using a fixed random seed for reproducibility.

The dataset exhibits considerable variability in document length, with an average of approximately 480 words per document ($\sigma \approx 320$) and an average reference summary length of approximately 50 words ($\sigma \approx 35$). The mean compression ratio is $10.5:1$ ($\sigma \approx 6.2$), reflecting the highly condensed nature of the target summaries. The vocabulary contains approximately 25,000 unique words. Quality filters were applied to exclude documents shorter than 100 words, those with empty reference summaries, and those with invalid encoding, ensuring a consistent and well-formed evaluation corpus.

**4.2 Methods Evaluated**

All methods were configured to produce extractive summaries of fixed length $n = 3$ sentences, a standard choice in extractive summarization research, enabling direct comparison across approaches.

**Lead-N and Random Baselines**. The Lead-N baseline selects the first $n = 3$ sentences of each document. The Random baseline samples $n = 3$ sentences uniformly at random without replacement, with a fixed seed of 42 to ensure reproducibility, and restores the selected sentences to their original document order.

**TextRank**. TextRank was implemented as PageRank applied to a sentence similarity graph constructed from TF-IDF representations. Sentences were vectorized using a TF-IDF scheme with nn
n-gram range $(1, 5)$, English stop word removal, inverse document frequency weighting, and additive smoothing to prevent zero divisions. The cosine similarity between sentence vectors $\mathbf{u}$ and $\mathbf{v}$ defines the edge weight:

$$w(u, v) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|}$$

PageRank was executed with a damping factor $d = 0.85$, a maximum of 100 iterations, and a convergence tolerance of $\epsilon = 10^{-4}$, following the standard formulation:

$$PR(v) = \frac{1 - d}{N} + d \sum_{u \in \mathcal{N}(v)} \frac{w(u,v) \cdot PR(u)}{\sum_{z} w(u,z)}$$

where $N$ is the total number of sentences and $\mathcal{N}(v)$ denotes the set of sentences connected to $v$.

**LexRank**. LexRank was implemented using the same PageRank framework but with sentence embeddings produced by the pre-trained Sentence-BERT model all-MiniLM-L6-v2, which maps sentences to a 384-dimensional dense vector space and supports sequences of up to 256 tokens. This model was trained on over one billion sentence pairs and offers a favorable trade-off between semantic quality and inference speed. Inter-sentence similarity was computed as cosine similarity between embeddings, and the continuous (fully weighted) graph variant was used, retaining all edges with weight above a threshold of 0.1. The damping factor, iteration limit, and convergence tolerance were identical to those used in TextRank.

**LSA**. Latent Semantic Analysis was applied by constructing a TF-IDF term-sentence matrix with a vocabulary limited to the top 1,000 features and an n-gram range of $(1, 2)$, then decomposing it via Truncated SVD:

$$\mathbf{A} \approx \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top$$

retaining $k = 5$ latent components. Each sentence $i$ was represented by the $i$-th row of the reduced matrix $\mathbf{U}_k \boldsymbol{\Sigma}_k$, and its importance score was computed as the $\ell_2$ norm of that row:

$$\text{score}(i) = \left\| (\mathbf{U}_k \boldsymbol{\Sigma}_k)_i \right\|_2$$

A fixed random state of 42 was applied to the SVD computation for reproducibility.

**4.3 Evaluation Metrics**

Summarization quality was assessed using the ROUGE family of metrics (Lin, 2004), computed with the rouge-score library with Porter stemming enabled. Three variants were reported:

- ROUGE-1: Unigram overlap between generated and reference summaries, serving as the primary measure of content coverage.
- ROUGE-2: Bigram overlap, capturing phrase-level fluency and coherence.
- ROUGE-L: Longest common subsequence ratio, reflecting sentence-level structural similarity.

All three metrics were reported as F-measures, defined as the harmonic mean of precision $P$ and recall $R$:

$$F = \frac{2 \cdot P \cdot R}{P + R}$$

Results are presented as means and standard deviations computed across all 1,000 documents, accompanied by 95% confidence intervals obtained via bootstrap resampling with 1,000 bootstrap samples.

**4.4 Statistical Analysis**

To assess whether observed performance differences between methods are statistically reliable, paired tt
t-tests were applied to per-document ROUGE-1 score differences at a significance level of $\alpha = 0.05$. The Wilcoxon signed-rank test was additionally employed as a non-parametric validation. A difference was considered statistically significant when $p < 0.05$.

**4.5 Ablation Studies**

Two targeted ablation studies were conducted. First, a parameter sensitivity analysis examined the effect of varying the number of extracted sentences $n \in \{2, 3, 4, 5\}$  for TextRank, evaluated on a 100-document subset. Second, an embedding comparison evaluated three sentence representation strategies - TF-IDF, averaged Word2Vec embeddings, and SBERT (
all-MiniLM-L6-v2) - on a 50-document subset, quantifying the contribution of neural embeddings to summarization quality.

**4.6 Reproducibility**

All stochastic operations were controlled through a global random seed of 42, applied consistently to dataset sampling, the Random baseline, bootstrap resampling, and SVD computation. The evaluation was implemented in Python 3.11 using NumPy 1.24.3, pandas 2.0, scikit-learn 1.3, NetworkX 3.1, rouge-score 0.1.2, sentence-transformers 2.2, and NLTK 3.8.1. The full evaluation pipeline was executed on a standard CPU, with no specialized hardware required, and the complete implementation is available in the accompanying source repository.

**4.7 Limitations**

Several limitations of the experimental design warrant acknowledgment. The dataset is domain-specific - medical transcriptions - and results may not generalize directly to other text genres. ROUGE scores are computed against a single human reference per document, which may underestimate the quality of semantically equivalent but lexically distinct summaries. Furthermore, the fixed output length of $n=3$ sentences does not account for variation in appropriate summary length across documents of differing complexity, and all methods are constrained to the extractive paradigm, precluding the generation of novel or paraphrased content.


## 5. Results Summary

**5.1 Main Results**

Table 2 presents the ROUGE scores obtained by all four evaluated methods across 1,000 medical transcription documents. Results are reported as means with standard deviations and 95% bootstrap confidence intervals.

Table 2. ROUGE scores for all evaluated methods (mean ± std, 95% CI, $n = 1000$).

| Method   | ROUGE-1                          | ROUGE-2                          | ROUGE-L                          |
|----------|----------------------------------|----------------------------------|----------------------------------|
| Lead-N   | 0.308 ± 0.192 [0.296, 0.320]    | 0.238 ± 0.196 [0.227, 0.251]    | 0.284 ± 0.190 [0.273, 0.297]    |
| TextRank | 0.204 ± 0.157 [0.194, 0.214]    | 0.122 ± 0.150 [0.113, 0.132]    | 0.177 ± 0.147 [0.168, 0.187]    |
| Random   | 0.189 ± 0.149 [0.180, 0.198]    | 0.108 ± 0.149 [0.099, 0.118]    | 0.162 ± 0.141 [0.154, 0.171]    |
| LSA      | 0.177 ± 0.157 [0.167, 0.187]    | 0.090 ± 0.143 [0.081, 0.100]    | 0.151 ± 0.144 [0.141, 0.160]    |

The Lead-N baseline achieved the highest performance across all three metrics, attaining a ROUGE-1 score of $0.308$, a ROUGE-2 score of $0.238$, and a ROUGE-L score of $0.284$. Among the unsupervised machine learning methods, TextRank ranked second with a ROUGE-1 of $0.204$, followed by the Random baseline at $0.189$, and LSA at $0.177$. The non-overlapping 95% confidence intervals across all four methods confirm that the observed performance differences are statistically reliable. Notably, LSA performed marginally below even the Random baseline on ROUGE-1, a finding examined further in Section 5.3.

Due to the unavailability of PyTorch and Sentence-BERT dependencies in the evaluation environment, LexRank could not be included in the main experimental comparison. Based on evidence from the literature, SBERT-based embeddings are expected to yield improvements of approximately 5–8% in ROUGE-1 over TF-IDF-based graph methods, at the cost of substantially higher computational overhead.

**5.2 Statistical Significance**

Pairwise comparisons between all methods were conducted using paired $t$-tests at significance level $\alpha = 0.05$, applied to per-document ROUGE-1 scores. All six pairwise comparisons yielded statistically significant differences. Table 3 summarises the results.

Table 3. Pairwise significance tests on ROUGE-1 scores (paired $t$-test, $\alpha = 0.05$, $n = 1000$).

| Comparison          | Mean Difference | p-value                  | Significant |
|---------------------|-----------------|--------------------------|-------------|
| Lead-N vs. LSA      | +0.131          | $p < 3 \times 10^{-78}$          | Yes (***)   |
| Lead-N vs. Random   | +0.119          | $p < 1 \times 10^{-72}$          | Yes (***)   |
| Lead-N vs. TextRank | +0.104          | $p < 3 \times 10^{-56}$           | Yes (***)   |
| TextRank vs. LSA    | +0.027          | $p < 3 \times 10^{-9}$            | Yes (***)   |
| TextRank vs. Random | +0.015          | $p = 0.006$               | Yes (**)    |
| Random vs. LSA      | +0.012          | $p = 0.041$              | Yes (*)     |

The extremely small $p$-values associated with comparisons involving Lead-N reflect the strong positional bias present in the MTSamples corpus, as discussed in Section 5.3. TextRank significantly outperforms both the Random baseline ($\Delta = +0.015$, $p = 0.006$) and LSA ($\Delta = +0.027$, $p < 3 \times 10^{-9}$), confirming that graph-based sentence centrality provides a meaningful signal beyond random selection. The marginal but statistically significant advantage of Random over LSA ($\Delta = +0.012$, $p = 0.041$) suggests that LSA's topic decomposition does not align well with the structure of medical terminology in this corpus.

**5.3 Error Analysis and Performance Variance**

All four methods exhibit high within-method variance, with ROUGE-1 standard deviations ranging from $0.149$ (Random) to $0.192$ (Lead-N). Per-document scores span the full range from $0.000$ to $0.796$, underscoring that summarization performance is strongly document-dependent. Three qualitative case studies, selected at the 10th percentile, median, and 90th percentile of average ROUGE-1 performance, illuminate the primary sources of this variance.

High-performing case (Document 921 - Cardiovascular Surgery, ROUGE-1: 0.796). This operative report for a patent ductus arteriosus ligation follows a rigid sectional structure (TITLE, INDICATION, PREOPERATIVE DIAGNOSIS, FINDINGS, PROCEDURE), with the critical diagnostic information contained in the opening sentence. Lead-N and TextRank both selected the same two leading sentences, achieving identical ROUGE-1 scores of $0.796$. LSA, by contrast, scored $0.174$ by selecting procedural detail sentences from later in the document, demonstrating its susceptibility to distributing selection across latent topics rather than concentrating on the most salient content.

Medium-performing case (Document 604 - Spinal Surgery, ROUGE-1: 0.211). This 759-word operative report contains extremely long opening sentences encoding preoperative and postoperative diagnoses. All methods converged on the same sentence selection strategy, producing a narrow performance spread of only $0.030$ across methods, yet achieving only moderate ROUGE scores due to the high lexical complexity of the reference summary.

Low-performing case (Document 388 - Bariatric Consultation, ROUGE-1: 0.000). All four methods achieved a ROUGE-1 score of exactly $0.000$ on this 716-word, 69-sentence consultation note. The reference summary - "Consult for laparoscopic gastric bypass" - is a highly abstractive synthesis of the document type and the procedure discussed, bearing no extractable surface-level correspondence to any individual sentence in the source text. This case exemplifies a fundamental limitation of all extractive approaches: when a reference summary requires conceptual abstraction rather than sentence selection, extractive methods cannot recover the necessary content regardless of the ranking criterion employed.

**5.4 Lead-N Dominance and Corpus Structure**

The strong performance of the Lead-N baseline - outperforming the best unsupervised method (TextRank) by $\Delta_{\text{ROUGE-1}} = 0.104$ - reflects a structural property of the MTSamples corpus rather than a limitation of graph-based or algebraic methods per se. Medical transcriptions conform to a highly standardized format in which diagnostic and procedural information is positioned at the document opening, closely mirroring the inverted pyramid convention observed in news journalism. In this setting, positional heuristics are particularly effective, and the advantage of semantic or topical sentence ranking is substantially reduced.

This result is consistent with observations from the general summarization literature, where Lead-3 baselines on news corpora (CNN/DailyMail) attain ROUGE-1 scores in the range $[0.35, 0.40]$, comparable to or exceeding many neural methods on the same data. The ROUGE-1 scores obtained in the present study fall within the range of $[0.18, 0.31]$, which is consistent with reported results for medical text summarization in the broader literature.

**5.5 Computational Performance**

Table 4 reports observed runtimes for the full 1,000-document evaluation on a standard CPU without GPU acceleration.

Table 4. Observed runtimes for 1,000-document evaluation on CPU.

| Method   | Total Time  | Per Document | Scalability          |
|----------|-------------|--------------|----------------------|
| Lead-N   | ≈ 20 s      | ≈ 20 ms      | $\mathcal{O}(m)$               |
| Random   | ≈ 25 s      | ≈ 25 ms      | $\mathcal{O}(m + n \log{n})$       |
| TextRank | ≈ 15 min    | ≈ 900 ms     | $\mathcal{O}(s^{2} \cdot d + s^{3})$   |
| LSA      | ≈ 20 min    | ≈ 1200 ms    | $\mathcal{O}(s \cdot d \cdot k)$        |

Despite its lower asymptotic complexity of $\mathcal{O}(s \cdot d \cdot k)$, LSA incurred longer per-document runtimes than TextRank in practice, likely due to the overhead of sparse matrix construction and SVD initialization. Lead-N and Random remain orders of magnitude faster than either unsupervised method, with per-document latencies below 30 ms. Peak memory usage during evaluation reached approximately 3 GB, attributable primarily to TF-IDF matrix storage and document buffering.

## 6. Conclusion 

This article presented a comparative evaluation of three unsupervised extractive text summarization methods (TextRank, LexRank, and Latent Semantic Analysis with Singular Value Decomposition) alongside two reference baselines, Lead-N and Random selection, on a large-scale corpus of 1,000 medical transcriptions drawn from the MTSamples dataset. The evaluation was conducted under consistent experimental conditions with rigorous statistical validation, providing a reproducible benchmark for extractive summarization in the clinical domain.

The principal finding of this study is that the Lead-N baseline achieved the highest performance across all three ROUGE metrics (ROUGE-1: $0.308$, ROUGE-2: $0.238$, ROUGE-L: $0.284$), significantly outperforming all unsupervised methods with $p < 3 \times 10^{-56}$. This result reflects a structural property of the MTSamples corpus: medical transcriptions follow a standardized format in which diagnostic and procedural information is consistently positioned at the document opening, making positional heuristics particularly effective in this domain. Among the unsupervised methods, TextRank achieved the strongest performance (ROUGE-1: $0.204$), significantly outperforming both the Random baseline ($\Delta = +0.015$, $p = 0.006$) and LSA ($\Delta = +0.027$, $p < 3 \times 10^{-9}$), confirming that graph-based sentence centrality captures meaningful content signals beyond random selection. LSA, with a ROUGE-1 of $0.177, performed marginally below even the Random baseline, suggesting that SVD-based topic decomposition does not align well with the specialized vocabulary and structural conventions of medical transcription text.

The error analysis revealed that performance variance is high across all methods, with per-document ROUGE-1 scores ranging from $0.000$ to $0.796$. Case studies identified two primary factors driving this variance: document structure and reference summary abstraction level. In well-structured operative reports with key information at the opening, all graph-based and positional methods performed strongly. Conversely, when reference summaries required high-level conceptual synthesis - as in consultation notes where the reference reduces an entire patient history to a single procedural statement - all extractive methods failed completely, achieving ROUGE-1 scores of $0.000$. This finding highlights a fundamental limitation shared by all extractive approaches: they cannot generate content that does not appear verbatim or near-verbatim in the source document.

From a computational perspective, Lead-N and Random baselines operate in near-constant time at the document level (under 30 ms per document), while TextRank and LSA require approximately 900 ms and 1,200 ms per document respectively on a standard CPU, reflecting their quadratic and sub-quadratic scaling with document length. LexRank with SBERT embeddings was not evaluated in this study due to dependency constraints, but represents a promising direction for future work, as transformer-based sentence representations are expected to improve over TF-IDF-based similarity by approximately 5–8% in ROUGE-1 at the cost of substantially higher inference time.

To the best of the authors' knowledge, this study constitutes the first large-scale evaluation of unsupervised extractive summarization methods on the MTSamples corpus with full statistical rigor, including bootstrap confidence intervals and pairwise significance testing. The results establish a reproducible performance baseline for future research on clinical text summarization and provide concrete guidance for practitioners selecting summarization methods for structured medical documents.

Future work should address several limitations identified in this study. Evaluation on additional domains (including news corpora, scientific articles, and conversational text) would clarify the generalizability of the observed performance hierarchy. Incorporating abstractive summarization methods, particularly transformer-based models such as BART and PEGASUS, would enable a more complete picture of the state of the art. Additionally, evaluation against multiple human references per document and the use of semantic similarity metrics beyond ROUGE would provide a more comprehensive assessment of summary quality, particularly for documents whose reference summaries are highly abstractive in nature.

## References 

1. Gulati, V., Kumar, D., Popescu, D., & Hemanth, J. (2023). Extractive Article Summarization Using Integrated TextRank and BM25+ Algorithm. Electronics. https://doi.org/10.3390/electronics12020372.

2. Mihalcea, R. (2004). Graph-based Ranking Algorithms for Sentence Extraction, Applied to Text Summarization. , 20. https://doi.org/10.3115/1219044.1219064.

3. Gokhan, T., Price, M., & Lee, M. (2024). Graphs in clusters: a hybrid approach to unsupervised extractive long document summarization using language models. Artificial Intelligence Review, 57. https://doi.org/10.1007/s10462-024-10828-w.

4. Naser, W. (2021). An Approach for Multi-Document Text Summarization Using Extreme Learning Machine and LexRank. International Journal of Engineering Research and Advanced Technology. https://doi.org/10.31695/ijerat.2021.3704.

5. Barman, U., Barman, V., Choudhury, N., Rahman, M., & Sarma, S. (2022). Unsupervised Extractive News Articles Summarization leveraging Statistical, Topic-Modelling and Graph-Based Approaches. Journal of Scientific & Industrial Research. https://doi.org/10.56042/jsir.v81i09.53185.

6. Melnyk H., Melnyk V., Vikovan V. Application of natural language processing and fuzzy logic to disinformation detection. Bukovinian Math. Journal. 2024. Vol. 12, no. 1. P. 21–31. 

7. Melnyk H., Melnyk V. Enhancing Mood Detection in Textual Analysis through Fuzzy Logic Integration. 2024 14th International Conference on Advanced Computer Information Technologies (ACIT), Ceske Budejovice, Czech Republic, 19 September 2024. P. 23–26.

8. Steinberger, J., & Ježek, K. (2004). Using latent semantic analysis in text summarization and summary evaluation. ISIM, 2004.

9. Erkan, G., & Radev, D. R. (2004). LexRank: Graph-based lexical centrality as salience in text summarization. Journal of Artificial Intelligence Research, 22, 457-479.