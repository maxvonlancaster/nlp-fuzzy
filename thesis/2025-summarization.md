# Unsupervised Machine Learning Methods for Text Summarization

This research demonstrates that unsupervised methods — particularly Self-Organizing Maps (Kohonen Networks) and PCA-based clustering — can extract key information from text without labeled data. By combining Word2Vec embeddings with topological clustering, the system produces concise, semantically rich summaries suitable for practical applications in content filtering, digital archives, and intelligent assistants.

## Introduction

Problem statement: rapid growth of digital text information.

Importance of automatic summarization (journalism, research, customer support).

Overview of summarization techniques: Extractive vs. Abstractive. Supervised (neural-based) vs. Unsupervised (clustering, projection).

The research objective is to investigate the use of unsupervised machine learning techniques (PCA, K-Means, Kohonen Maps) for extractive summarization.

Scientific novelty lies in the integration of vector-based sentence representations with topological clustering for summary extraction.


## Related Work

Traditional statistical summarization methods (TF-IDF, LSA).

Neural summarization models (BERTSum, Pegasus, GPT-based models).

Limitations of supervised approaches (need for labeled data, high computational cost).

Prior research using: Principal Component Analysis (PCA) for dimensionality reduction of sentence embeddings. K-Means clustering for identifying topic clusters. Self-Organizing Maps (Kohonen Networks) for topological text grouping.

Gap identified: Lack of studies combining semantic embeddings with topological unsupervised clustering for extractive summarization.


## Methodology

Step 1: Text Preprocessing. In the context of this step we first perform sentence segmentation via the NLTK. After that, we perform tokenization and stopword removal.

Step 2: Sentence Embedding. We use Word2Vec to represent each sentence as the mean of its word vectors. We use dimensionality of 100–300 features per sentence.

Step 3: Dimensionality Reduction. The algorithm PCA (Principal component analysis) is used to reduce dimensionality while preserving variance. Alternatively, we also used UMAP for nonlinear manifold reduction.

Step 4: Clustering and Topological Mapping. K-Means is used for grouping of semantically similar sentences. Kohonen SOM (MiniSom) projects high-dimensional embeddings onto a 2D map. Sentences in dense regions or cluster centers represent main ideas.

Step 5: Summary Extraction. The algorithm selects the top N sentences closest to cluster centroids (or from largest SOM nodes). Reordering by their original position is performed to ensure coherence.

Step 6: Visualization. We display Kohonen map with sentence indices to illustrate topic clusters.


## Experimental Results

A small but diverse dataset was assembled consisting of 10–50 documents, including: scientific research abstracts, news articles from technology and science domains, general long-form explanatory texts. The intention behind choosing a compact dataset was to test the robustness of unsupervised methods in low-resource scenarios, where labeled training sets are unavailable.

Texts ranged from 300 to 1,200 words, enabling evaluation on both short and medium-length content.

Each document was subjected to identical preprocessing steps:

1. Sentence segmentation using NLTK’s sent_tokenize.

2. Tokenization and removal of: stopwords, punctuation and non-alphabetic tokens.

3. Training or loading a Word2Vec model with vector size of 100 and window size of 5.

4. Representing each sentence as the mean vector of its constituent word embeddings.

This resulted in a matrix of dimensionality $S\in \mathbb{R}^{n\times 100}$ where $n$ is the number of sentences in the document.


## Evaluation

The goal of evaluation is to define how to measure the quality of the summaries objectively and subjectively.

The evaluation mainly consists of quantitative metrics and qualitative (human-based) evaluation.


## Discussion

The results obtained in this study provide insights into the potential of unsupervised geometric and topological methods for extractive text summarization. Unlike transformer-based models or supervised approaches that rely on large training corpora, the proposed method uses only internal statistical and semantic properties of the text itself. This section interprets the findings, evaluates the strengths and limitations, and outlines future research directions.

One of the most significant benefits of the described approach is its independence from labeled datasets. Classic summarization methods—especially modern neural architectures—depend on large human-curated corpora, which are costly and often domain-specific. In contrast, our pipeline requires no human supervision, adapts to any domain (science, news, technical documentation, etc.) and performs well even on relatively small text collections.

Unlike many black-box neural summarizers, this method offers explicit interpretability. PCA visualizations show how sentences distribute in semantic space. Kohonen maps reveal topological relationships and thematic clusters. Clusters correspond directly to extracted summary sentences. The interpretability is crucial for high-stakes fields such as medicine, law, finance, and scientific research, where users must understand why particular statements were selected.

The proposed method demonstrates that unsupervised text summarization leveraging vector space geometry and SOM topology is both feasible and effective. While limitations exist—particularly concerning contextual understanding and small datasets—the approach offers strong interpretability, efficiency, and domain independence. It stands as a competitive alternative to classical unsupervised methods and a complementary approach to modern neural summarizers.


## Conclusion

Based on the findings, several promising research directions emerge.

Firstly, replacing Word2Vec with either BERT / RoBERTa embeddings, or sentence-BERT for sentence-level semantics, would increase the accuracy of clustering and help disambiguate polysemous words.

There is also potential in Hybrid SOM–Reinforcement Learning Summarization. A reinforcement learning agent could dynamically adjust summary length, learn which SOM nodes are most informative and optimize extraction based on reward signals (e.g., ROUGE or human ratings).

There is also possibility in combining extractive and abstractive methods. An abstractive layer—such as a small LSTM decoder or distilled GPT model—could rewrite extractive output to improve coherence while retaining unsupervised selection.


