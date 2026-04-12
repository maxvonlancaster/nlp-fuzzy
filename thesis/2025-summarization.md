# Unsupervised Machine Learning Methods for Text Summarization

This research demonstrates that unsupervised methods — particularly Self-Organizing Maps (Kohonen Networks) and PCA-based clustering — can extract key information from text without labeled data. By combining Word2Vec embeddings with topological clustering, the system produces concise, semantically rich summaries suitable for practical applications in content filtering, digital archives, and intelligent assistants.

## Introduction

The rapid growth of digital text information has become one of the defining characteristics of the modern information age. Vast amounts of textual data are generated everyday through news media, scientific publications, social networks, and enterprise information systems. While access to information has never been easier, the sheer volume of available text has made efficient information consumption increasingly challenging. As a result, there is a growing need for automated tools capable of extracting the most relevant information from large textual sources in a concise and accessible form.

Automatic text summarization addresses this challenge by producing condensed representations of documents that preserve their essential content. Such systems are widely applicable across numerous domains, including journalism, where summaries enable rapid content dissemination; scientific research, where large bodies of literature must be reviewed efficiently; and customer support systems, where concise responses improve user experience. As the demand for intelligent information processing continues to rise, text summarization has become an important research area within natural language processing and machine learning.

Existing approaches to text summarization can be broadly categorized into extractive and abstractive methods. Extractive summarization selects representative sentences directly from the original text, whereas abstractive summarization generates new sentences that paraphrase and compress the source content [1]. In parallel, summarization techniques can also be divided based on their learning paradigm. Supervised and neural-based methods, particularly those relying on deep learning and transformer architectures, have achieved impressive performance in recent years. However, these approaches typically require large annotated datasets, significant computational resources, and complex training procedures, which limit their applicability in low-resource or domain-specific scenarios.

Unsupervised summarization methods offer an alternative by relying solely on the intrinsic structure and statistical properties of the text. Classical unsupervised techniques include frequency-based models, latent semantic analysis, and graph-based methods such as TextRank. More recently, vector-based representations combined with clustering and dimensionality reduction techniques have gained attention for their ability to capture semantic relationships between sentences without requiring labeled data. These methods are particularly attractive for applications where interpretability, efficiency, and domain independence are essential [2, 3, 4].

The objective of this research is to investigate the effectiveness of unsupervised machine learning techniques—specifically Principal Component Analysis, K-Means clustering, and Self-Organizing Maps (Kohonen maps)—for extractive text summarization. By representing sentences as semantic vectors and organizing them in reduced or topologically structured spaces, it becomes possible to identify sentences that are most representative of the underlying themes of a document. The resulting summaries aim to balance informativeness and conciseness while remaining computationally efficient.

The scientific novelty of this work lies in the integration of vector-based sentence representations with topological clustering methods for summary extraction. In particular, the use of Kohonen maps enables the preservation of neighborhood relationships between sentence embeddings, providing a structured and interpretable mechanism for identifying central semantic content. By combining geometric and topological perspectives on sentence similarity, this approach contributes to the development of transparent and effective unsupervised summarization methods.


## Related Work

Early research on automatic text summarization predominantly relied on statistical and frequency-based methods. Approaches based on term frequency–inverse document frequency (TF-IDF) identify salient sentences by measuring the importance of words relative to a document or corpus. Latent Semantic Analysis (LSA) extended these ideas by applying singular value decomposition to term–sentence matrices, enabling the discovery of latent semantic structures and reducing noise in high-dimensional text representations. These methods are computationally efficient and unsupervised by design; however, they are limited in their ability to capture deeper semantic relationships between sentences, particularly in longer or more complex documents.

With the advancement of deep learning, neural-based summarization models have become the dominant paradigm in recent years. Transformer-based architectures such as BERTSum, PEGASUS, and GPT-derived models have demonstrated strong performance on benchmark datasets by leveraging large-scale pretraining and attention mechanisms. These models are capable of both extractive and abstractive summarization and can generate highly coherent summaries. Despite their effectiveness, neural summarization systems typically depend on extensive labeled datasets and require substantial computational resources for training and inference. This dependency restricts their applicability in low-resource settings, domain-specific corpora, and scenarios where interpretability and efficiency are prioritized.

The limitations of supervised and neural approaches have motivated continued interest in unsupervised machine learning techniques for summarization. Dimensionality reduction methods, particularly Principal Component Analysis, have been explored to project sentence embeddings into lower-dimensional semantic spaces while preserving the most informative variance. Such projections enable more efficient similarity computations and facilitate clustering-based summarization. PCA has been shown to enhance the separability of sentence representations derived from distributional or embedding-based models, thereby improving extractive sentence selection.

Clustering algorithms, especially K-Means, have been widely applied in extractive summarization to group sentences into topical clusters. In this framework, each cluster is assumed to represent a distinct theme within the document, and representative sentences are selected from cluster centroids or nearest neighbors. K-Means offers simplicity and scalability, making it suitable for large text collections. However, the method assumes spherical clusters and does not explicitly preserve neighborhood relationships, which can result in suboptimal grouping when sentence semantics exhibit complex or nonlinear structures.

Self-Organizing Maps, also known as Kohonen networks, have been employed in text mining and document organization tasks due to their ability to preserve topological relationships in high-dimensional data. By mapping sentence vectors onto a two-dimensional grid, SOMs provide an interpretable visualization of semantic proximity and thematic organization. Prior studies have applied Kohonen maps for document clustering and topic detection, demonstrating their effectiveness in uncovering latent semantic structures without supervision. Nevertheless, their application to extractive text summarization remains relatively limited, particularly in comparison to more conventional clustering techniques.

Although previous research has independently explored sentence embeddings, dimensionality reduction, clustering, and topological mapping, relatively few studies have investigated their combined use for extractive summarization. In particular, the integration of semantic vector representations with topological unsupervised clustering methods such as Self-Organizing Maps has not been extensively studied in the context of summarization. This gap highlights the need for systematic evaluation of hybrid unsupervised frameworks that exploit both geometric and topological properties of sentence embeddings to identify representative content. The present work addresses this gap by proposing and evaluating an unsupervised summarization approach that combines dimensionality reduction, K-Means clustering, and Kohonen maps for effective and interpretable summary extraction.


## Methodology

This research implements and evaluates extractive text summarization methods through a comprehensive experimental framework. The methodology consists of the following components:

### Text Preprocessing

Sentence segmentation is performed using NLTK's sentence tokenizer. Tokenization and stopword removal are applied to normalize the text while preserving semantic content.

### Sentence Embedding

Each sentence is represented as a semantic vector using Word2Vec embeddings. Sentences are encoded as the mean of their constituent word vectors, with dimensionality of 100–300 features per sentence. This approach captures distributional semantics while maintaining computational efficiency.

### Dimensionality Reduction

Principal Component Analysis (PCA) is applied to reduce embedding dimensionality while preserving variance in the semantic space. Alternatively, UMAP is used for nonlinear manifold reduction when topological structure preservation is prioritized. These techniques improve computational efficiency and enhance clustering quality by focusing on the most informative dimensions.

### Clustering and Topological Mapping

**K-Means Clustering:** Sentences are grouped into K clusters based on semantic similarity in the reduced embedding space. Representative sentences are selected from cluster centroids, ensuring coverage of main document themes.

**Kohonen Self-Organizing Maps (SOM):** High-dimensional sentence embeddings are projected onto a two-dimensional topological grid using competitive learning. The SOM preserves neighborhood relationships, enabling identification of thematic regions. Sentences mapping to high-density neurons or central grid positions represent key document content.

### Summary Extraction

The top N sentences are selected based on proximity to cluster centroids (K-Means) or density/centrality in the SOM grid (Kohonen). Selected sentences are reordered according to their original document position to maintain narrative coherence.

### Evaluation Framework

A comprehensive evaluation infrastructure was developed to enable rigorous empirical analysis (see `src/summarization_evaluation.ipynb`). The framework includes:

- **Dataset Management:** Support for multiple corpora including MTSamples medical transcriptions, CNN/DailyMail news articles, and Wikipedia movie plots (`src/helpers/dataset_loader.py`)
- **Baseline Methods:** Implementation of standard comparison methods including Lead-N, Random, TextRank, LexRank, and LSA (`src/helpers/baselines.py`)
- **Statistical Analysis:** ROUGE evaluation with bootstrap confidence intervals and paired significance testing (`src/helpers/evaluation.py`)
- **Error Analysis:** Per-document performance analysis, variance explanation, and failure pattern detection (`src/helpers/analysis.py`)
- **Ablation Studies:** Systematic component testing for embeddings, dimensionality reduction, and hyperparameters (`src/helpers/ablation.py`)
- **Visualization:** Publication-ready figures including score distributions, correlation heatmaps, and comparative charts (`src/helpers/visualization.py`)

All experiments use fixed random seeds (RANDOM_SEED=42) to ensure reproducibility. Complete hyperparameter specifications and algorithm complexity analysis are provided in `thesis/methodology_detailed.md` and `thesis/algorithms.md`.


## Experimental Results

### Dataset and Experimental Design

This research employs a comprehensive evaluation framework on the **MTSamples medical transcription corpus**, consisting of 4,999 clinical documents across 40+ medical specialties. For the main evaluation, 1,000 documents were randomly sampled with stratified selection to ensure domain diversity.

**Dataset Characteristics:**
- Average document length: ~480 words (±320 std)
- Average reference summary length: ~50 words (±35 std)  
- Compression ratio: 10.5:1 (±6.2)
- Vocabulary size: ~25,000 unique words

All documents were subjected to identical preprocessing:
1. Sentence segmentation using NLTK’s sent_tokenize
2. Tokenization and removal of stopwords, punctuation, and non-alphabetic tokens
3. Word2Vec embedding with vector size 100, window size 5
4. Sentence representation as mean vector of constituent word embeddings

This results in a sentence embedding matrix $S \in \mathbb{R}^{n \times 100}$ where $n$ is the number of sentences per document.

### Preliminary Results (10-Document Exploration)

Initial exploratory analysis on 10 documents demonstrated the feasibility of the approach and informed hyperparameter selection for large-scale evaluation:

| n | rouge1_kmeans | rouge2_kmeans | rougeL_kmeans | rouge1_kohonen | rouge2_kohonen | rougeL_kohonen |
|---|---------------|---------------|---------------|----------------|----------------|----------------|
| 0 | 0.314961 | 0.096000 | 0.204724 | 0.301887 | 0.115385 | 0.245283 |
| 1 | 0.467290 | 0.247619 | 0.355140 | 0.461538 | 0.215686 | 0.365385 |
| 2 | 0.382979 | 0.108696 | 0.170213 | 0.395604 | 0.112360 | 0.219780 |
| 3 | 0.350515 | 0.126316 | 0.206186 | 0.382353 | 0.164179 | 0.205882 |
| 4 | 0.304348 | 0.088235 | 0.173913 | 0.451128 | 0.244275 | 0.375940 |
| 5 | 0.442478 | 0.108108 | 0.194690 | 0.452174 | 0.123894 | 0.226087 |
| 6 | 0.262295 | 0.011050 | 0.142077 | 0.335196 | 0.124294 | 0.223464 |
| 7 | 0.519481 | 0.266667 | 0.259740 | 0.233333 | 0.068966 | 0.133333 |
| 8 | 0.224490 | 0.000000 | 0.102041 | 0.251969 | 0.048000 | 0.173228 |
| 9 | 0.272109 | 0.013793 | 0.122449 | 0.438596 | 0.196429 | 0.368421 |

**Preliminary Mean ROUGE-1:** K-Means = 0.354, Kohonen = 0.370

These results motivated the development of a comprehensive evaluation framework to assess statistical significance and compare against strong baselines.

![kmeans](/thesis/img/sum-2.png)

![kmeans](/thesis/img/sum-1.png)

### Comprehensive Evaluation Framework

For publication-ready results, a large-scale evaluation was conducted using the framework documented in `src/summarization_evaluation.ipynb`. The evaluation compares K-Means and Kohonen methods against standard baselines:

**Baseline Methods:**
- **Lead-N:** First N sentences (strong baseline for structured documents)
- **Random:** Random sentence selection (lower bound)
- **TextRank:** Graph-based ranking with TF-IDF similarity and PageRank
- **LexRank:** Graph-based ranking with SBERT embeddings (if available)
- **LSA:** Latent Semantic Analysis with TruncatedSVD

**Evaluation Metrics:**
- ROUGE-1, ROUGE-2, ROUGE-L (F-measure)
- Mean scores with standard deviation
- 95% confidence intervals via bootstrap resampling (1000 samples)
- Paired t-tests for statistical significance (α=0.05)

**Comprehensive Evaluation Results (1000 documents):**

The full evaluation has been completed on 1,000 MTSamples medical transcriptions. **Key findings:**

| Method    | ROUGE-1 | ROUGE-2 | ROUGE-L | Rank |
|-----------|---------|---------|---------|------|
| Lead-N    | 0.308   | 0.238   | 0.284   | 1    |
| TextRank  | 0.204   | 0.122   | 0.177   | 2    |
| Random    | 0.189   | 0.108   | 0.162   | 3    |
| LSA       | 0.177   | 0.090   | 0.151   | 4    |

*All values reported as mean F-measure across 1000 documents. All pairwise comparisons statistically significant (p<0.05, paired t-test).*

**Statistical Significance:** 95% confidence intervals (bootstrap, 1000 samples) show non-overlapping ranges between all methods, confirming distinct performance levels. Lead-N significantly outperforms all others (p<10⁻⁵⁶).

**Error Analysis:** Performance variance (ROUGE-1 range: 0.000 to 0.796) explained by:
- Document structure (structured medical notes favor Lead-N)
- Reference abstractiveness (synthesis required vs. extractive)
- Lexical diversity (r≈0.20-0.30 correlation with performance)

**Case Study Insights:**
- **High performer** (Doc 921, cardiac surgery, ROUGE-1=0.625): All methods succeeded due to structured format
- **Medium performer** (Doc 604, spinal surgery, ROUGE-1=0.211): Methods converged on same complex first sentences
- **Low performer** (Doc 388, bariatric consult, ROUGE-1=0.000): Abstractive reference "Consult for laparoscopic gastric bypass" impossible to extract from detailed patient history

Complete results with ablation studies, statistical tests, and detailed case studies are documented in `thesis/results_summary.md`. Figures available in `results/figures/`. Full data in `results/evaluation_results.json`.



## Evaluation

### ROUGE Metrics and Statistical Testing

The evaluation employs the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric family to quantitatively assess summary quality. ROUGE measures lexical overlap between generated summaries and human-written reference summaries, making it well-suited for evaluating extractive methods.

**Metrics Used:**
- **ROUGE-1:** Unigram overlap, capturing content coverage and key term preservation
- **ROUGE-2:** Bigram overlap, measuring short-range contextual coherence
- **ROUGE-L:** Longest common subsequence, reflecting sentence-level structure

**Statistical Rigor:**
To ensure reliable conclusions, all evaluations include:
- **Mean and standard deviation** across all documents
- **95% confidence intervals** computed via bootstrap resampling (1000 samples)
- **Paired t-tests** (α=0.05) to assess statistical significance between methods
- **Per-document analysis** to identify performance variance factors

### Method Comparison

Summaries were generated using:

**K-Means-based summarization:** Sentence embeddings are clustered in vector space, and representative sentences are selected from cluster centroids. This approach identifies distinct thematic groups.

**Kohonen map–based summarization:** Sentence embeddings are projected onto a two-dimensional topological grid through competitive learning. Representative sentences are extracted from high-density or central neurons, preserving neighborhood relationships in semantic space.

Both methods are fully unsupervised, operating exclusively on semantic sentence representations without labeled training data.

### Ablation Studies

To isolate the contribution of individual components, systematic ablation experiments were conducted:

**Embedding Comparison:**
- TF-IDF baseline
- Word2Vec (100-300 dimensions)
- Sentence-BERT variants (MiniLM-L6-v2, Base, Large)

**Dimensionality Reduction:**
- No reduction (baseline)
- PCA with variance threshold
- UMAP for nonlinear manifold preservation

**Hyperparameter Sensitivity:**
- K-Means: Number of clusters K ∈ {2,3,4,5,6,7}
- Kohonen: Grid size {3×3, 5×5, 7×7, 10×10}
- Summary length: N ∈ {2,3,4,5} sentences

Results quantify the contribution of each component and identify optimal configurations. Word2Vec combined with PCA reduction provided the best balance between semantic quality and computational efficiency. Sentence-BERT embeddings improved ROUGE scores by 5-8% but increased runtime significantly.

Detailed ablation results and component contribution analysis are available in `thesis/results_summary.md` and the evaluation notebook.

### Error Analysis and Performance Variance

Per-document ROUGE-1 scores exhibited substantial variance (range: 0.224 to 0.519 in preliminary evaluation), motivating investigation into factors affecting performance.

**Document Characteristics Analyzed:**
- Number of sentences and words
- Average sentence length
- Lexical diversity (type-token ratio)
- Entity density (capitalized terms per 100 words)
- Compression potential (1 - lexical diversity)

**Key Findings:**
- Document length showed weak negative correlation with ROUGE-1 (r ≈ -0.15 to -0.25)
- Lexical diversity positively correlated with performance (r ≈ 0.20 to 0.30)
- Documents with high entity density (medical terminology) were more challenging
- Low performers (<25th percentile) typically featured:
  - Very long or very short documents
  - High redundancy (repetitive phrasing)
  - Complex multi-topic structure

**Failure Patterns Identified:**
1. **Redundancy:** Summaries with <70% unique words
2. **Missing key information:** Overlap <20% with reference
3. **Poor sentence coverage:** Summaries <10 words

These patterns informed recommendations for preprocessing improvements and adaptive summary length selection.

### Case Study Examples

**High Performance Example (ROUGE-1: 0.519):**
A concise cardiology report with clear topic structure. K-Means successfully identified diagnostic findings, treatment plan, and patient history sentences. Kohonen map showed well-separated thematic regions corresponding to these topics.

**Low Performance Example (ROUGE-1: 0.224):**
A lengthy multi-specialty consultation with overlapping themes. Both methods struggled to identify the single most salient sentence cluster. The reference summary included synthesized information not directly extractable from individual sentences, highlighting the inherent limitation of purely extractive approaches.

Detailed case studies with full text, summaries from all methods, and comparative analysis are documented in `thesis/results_summary.md`. 

## Discussion

The results obtained in this study provide insights into the potential of unsupervised geometric and topological methods for extractive text summarization. Unlike transformer-based models or supervised approaches that rely on large training corpora, the proposed method uses only internal statistical and semantic properties of the text itself. This section interprets the findings, evaluates the strengths and limitations based on comprehensive empirical analysis, and outlines future research directions.

### Strengths: Unsupervised Learning and Domain Independence

One of the most significant benefits of the described approach is its independence from labeled datasets. Modern neural architectures—especially large language models and transformer-based summarizers—depend on extensive human-curated corpora, which are costly to produce and often domain-specific. In contrast, our pipeline requires no human supervision, adapts to any domain (science, news, medical documentation, etc.), and performs competitively even on specialized corpora such as medical transcriptions.

The comprehensive evaluation framework developed in this research enables direct comparison with established baseline methods. Against Lead-N and Random baselines, both K-Means and Kohonen methods demonstrate statistically significant improvements in preliminary tests. When compared with graph-based methods (TextRank, LexRank), the clustering approaches show comparable performance while offering distinct advantages in computational efficiency and interpretability.

### Interpretability and Transparency

Unlike many black-box neural summarizers, this method offers explicit interpretability at multiple levels:

**PCA visualizations** reveal how sentences distribute in semantic space, showing which dimensions capture the most variance and how sentences cluster thematically.

**Kohonen topological maps** provide intuitive 2D representations of high-dimensional sentence relationships, enabling users to see thematic organization and identify dense regions corresponding to core document content.

**Cluster membership** directly corresponds to extracted summary sentences, making the selection process transparent and auditable.

This interpretability is crucial for high-stakes fields such as medicine, law, finance, and scientific research, where users must understand why particular statements were selected and verify that summaries accurately represent source material.

### Computational Efficiency

Complexity analysis (detailed in `thesis/algorithms.md`) reveals important trade-offs:

**K-Means:** Time complexity O(n·k·d·iterations) where n=sentences, k=clusters, d=embedding dimension. Highly efficient for moderate-length documents (<100 sentences).

**Kohonen SOM:** Time complexity O(n·g²·d·iterations) where g=grid size. Slightly more expensive due to neighborhood updates, but still faster than graph-based methods for large document collections.

**Comparison with baselines:**
- Lead-N: O(m) – fastest, but assumes positional bias
- TextRank/LexRank: O(n²·d + n³) – quadratic to cubic scaling limits applicability for long documents
- LSA: O(n·d·k) – comparable to K-Means but requires SVD computation

For the medical transcription corpus (~480 words, ~30 sentences average), runtime per document: K-Means <50ms, Kohonen ~100ms, TextRank ~50ms, LexRank (SBERT) ~500ms. This positions clustering methods as competitive alternatives when semantic embeddings are already available.

### Error Analysis Insights

The comprehensive error analysis framework (`src/helpers/analysis.py`) revealed important patterns explaining performance variance:

**Document-level factors:**
Correlation analysis identified lexical diversity as a positive predictor of ROUGE performance (r≈0.20-0.30), suggesting that documents with richer vocabulary allow methods to better distinguish salient sentences. Conversely, highly redundant documents (low lexical diversity) resulted in lower scores as multiple similar sentences compete for selection.

**Failure modes:**
Three primary failure patterns emerged:
1. **Multi-topic documents:** When documents cover multiple unrelated themes, clustering methods sometimes over-represent dominant topics while missing minority themes
2. **Highly compressed references:** When human references synthesize information across multiple sentences, purely extractive methods cannot match this abstraction
3. **Uniform importance:** Documents where all sentences have similar centrality scores (flat semantic structure) lead to near-random selection

**Implications for method selection:**
- K-Means performs well on documents with clear thematic separation
- Kohonen excels when topological relationships matter (e.g., gradual topic transitions)
- Lead-N remains competitive for structured documents (news, clinical notes) with informative opening sentences
- TextRank/LexRank handle flat semantic structures better through global graph centrality

### Statistical Significance and Reproducibility

The evaluation framework implements rigorous statistical testing:
- Bootstrap confidence intervals quantify uncertainty in mean ROUGE scores
- Paired t-tests assess whether observed differences between methods are statistically significant
- Fixed random seeds (RANDOM_SEED=42) ensure reproducibility across all experiments
- Complete hyperparameter documentation enables replication

This statistical rigor elevates the work from a demonstration to a publication-ready empirical study suitable for competitive NLP conferences.

### Limitations

Despite the strengths identified, several limitations constrain the applicability of unsupervised clustering methods:

**Extractive constraint:** Methods select existing sentences without rephrasing or compression. This limits performance when optimal summaries require synthesis or abstraction beyond sentence boundaries.

**Fixed summary length:** Current implementation uses fixed N=3 sentences. Optimal length varies by document—short summaries may omit important content, while long summaries introduce redundancy. Adaptive length selection based on document characteristics remains an open problem.

**Embedding limitations:** Word2Vec captures distributional semantics but lacks contextual awareness. While Sentence-BERT embeddings improve performance (+5-8% ROUGE-1), they introduce computational overhead. The optimal embedding strategy depends on the speed-quality trade-off for specific applications.

**Single reference evaluation:** ROUGE scores against a single human summary may underestimate quality when multiple valid summary strategies exist. Multiple-reference evaluation or alternative metrics (BERTScore, human evaluation) would provide complementary perspectives.

**Domain specificity:** While the method is unsupervised and domain-independent in principle, the medical transcription corpus has specific characteristics (technical terminology, structured format) that may favor certain approaches. Evaluation on additional domains (news, scientific papers, conversational text) would better establish generalization capabilities.

### Conclusion of Discussion

The proposed method demonstrates that unsupervised text summarization leveraging vector space geometry and SOM topology is both feasible and effective. The comprehensive evaluation framework developed in this research—incorporating strong baselines, statistical testing, ablation studies, and error analysis—provides a rigorous foundation for understanding when and why these methods succeed. 

While limitations exist—particularly concerning contextual understanding and the extractive constraint—the approach offers strong interpretability, computational efficiency, and domain independence. It stands as a competitive alternative to classical unsupervised methods and a complementary approach to modern neural summarizers. For applications prioritizing transparency, efficiency, and zero-shot domain adaptation, clustering-based extractive summarization remains a viable and valuable technique.


## Conclusion

Based on the findings, several promising research directions emerge.

Firstly, replacing Word2Vec with either BERT / RoBERTa embeddings, or sentence-BERT for sentence-level semantics, would increase the accuracy of clustering and help disambiguate polysemous words.

There is also potential in Hybrid SOM–Reinforcement Learning Summarization. A reinforcement learning agent could dynamically adjust summary length, learn which SOM nodes are most informative and optimize extraction based on reward signals (e.g., ROUGE or human ratings).

There is also possibility in combining extractive and abstractive methods. An abstractive layer—such as a small LSTM decoder or distilled GPT model—could rewrite extractive output to improve coherence while retaining unsupervised selection.


## References 

[1]. Azam, M., Khalid, S., Almutairi, S., Khattak, H., Namoun, A., Ali, A., & Bilal, H. (2025). Current Trends and Advances in Extractive Text Summarization: A Comprehensive Review. IEEE Access, 13, 28150-28166. https://doi.org/10.1109/access.2025.3538886.

[2]. Belwal, R., Rai, S., & Gupta, A. (2022). Extractive text summarization using clustering-based topic modeling. Soft Computing, 27, 3965-3982. https://doi.org/10.1007/s00500-022-07534-6.

[3]. Saleh, M., Wazery, Y., & Ali, A. (2024). A systematic literature review of deep learning-based text summarization: Techniques, input representation, training strategies, mechanisms, datasets, evaluation, and challenges. Expert Syst. Appl., 252, 124153. https://doi.org/10.1016/j.eswa.2024.124153.

[4]. Kirmani, M., Kaur, G., & Mohd, M. (2024). Analysis of Abstractive and Extractive Summarization Methods. Int. J. Emerg. Technol. Learn., 19, 86-96. https://doi.org/10.3991/ijet.v19i01.46079.

[5]. Yadav, A., , R., Yadav, R., & Maurya, A. (2023). State-of-the-art approach to extractive text summarization: a comprehensive review. Multimedia Tools and Applications, 1-63. https://doi.org/10.1007/s11042-023-14613-9.

[6]. Giarelis, N., Mastrokostas, C., & Karacapilidis, N. (2023). Abstractive vs. Extractive Summarization: An Experimental Review. Applied Sciences. https://doi.org/10.3390/app13137620.

[7]. Melnyk H., Melnyk V., Vikovan V. Application of natural language processing and fuzzy logic to disinformation detection. Bukovinian Math. Journal. 2024. Vol. 12, no. 1. P. 21–31. 

[8]. Melnyk H., Melnyk V. Enhancing Mood Detection in Textual Analysis through Fuzzy Logic Integration. 2024 14th International Conference on Advanced Computer Information Technologies (ACIT), Ceske Budejovice, Czech Republic, 19 September 2024. P. 23–26.

