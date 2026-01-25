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

![kmeans](/thesis/img/sum-2.png)

![kmeans](/thesis/img/sum-1.png)



## Evaluation

The evaluation was conducted using a dataset containing pairs of documents and corresponding human-written reference summaries. For each document, summaries were generated using two unsupervised approaches.

K-Means-based summarization, where sentence embeddings were clustered in vector space and representative sentences were selected from cluster centroids.

Kohonen map–based summarization, where sentence embeddings were projected onto a two-dimensional topological grid, and representative sentences were extracted from high-density or central neurons.

Both methods operate exclusively on semantic sentence representations derived from vector embeddings and do not rely on labeled data during training.

To quantitatively evaluate the quality of the generated summaries, the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric family was employed

The following ROUGE variants were used in the evaluation: ROUGE-1, measuring unigram (word-level) overlap between the generated summary and the reference summary; ROUGE-2, measuring bigram overlap and capturing limited contextual coherence, and finally, ROUGE-L, measuring the longest common subsequence (LCS), which reflects sentence-level structure and ordering.

Both K-Means and Kohonen-based methods produce extractive summaries, selecting sentences directly from the original text. ROUGE is specifically designed to evaluate extractive summarization by measuring lexical overlap between candidate and reference summaries. As a result, it provides a reliable approximation of how much salient content is preserved.

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

Higher ROUGE-1 scores indicate that the summarization method successfully captures key terms and concepts present in the reference summaries. ROUGE-2 provides insight into short-range contextual coherence, while ROUGE-L reflects the structural similarity between summaries.

Overall, mean for ROUGE-1 for kmeans based text summarization is 0.35, while for Kohonen-map based text summarization the value is 0.37. 

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


## References 

[1]. Azam, M., Khalid, S., Almutairi, S., Khattak, H., Namoun, A., Ali, A., & Bilal, H. (2025). Current Trends and Advances in Extractive Text Summarization: A Comprehensive Review. IEEE Access, 13, 28150-28166. https://doi.org/10.1109/access.2025.3538886.

[2]. Belwal, R., Rai, S., & Gupta, A. (2022). Extractive text summarization using clustering-based topic modeling. Soft Computing, 27, 3965-3982. https://doi.org/10.1007/s00500-022-07534-6.

[3]. Saleh, M., Wazery, Y., & Ali, A. (2024). A systematic literature review of deep learning-based text summarization: Techniques, input representation, training strategies, mechanisms, datasets, evaluation, and challenges. Expert Syst. Appl., 252, 124153. https://doi.org/10.1016/j.eswa.2024.124153.

[4]. Kirmani, M., Kaur, G., & Mohd, M. (2024). Analysis of Abstractive and Extractive Summarization Methods. Int. J. Emerg. Technol. Learn., 19, 86-96. https://doi.org/10.3991/ijet.v19i01.46079.

[5]. Yadav, A., , R., Yadav, R., & Maurya, A. (2023). State-of-the-art approach to extractive text summarization: a comprehensive review. Multimedia Tools and Applications, 1-63. https://doi.org/10.1007/s11042-023-14613-9.

[6]. Giarelis, N., Mastrokostas, C., & Karacapilidis, N. (2023). Abstractive vs. Extractive Summarization: An Experimental Review. Applied Sciences. https://doi.org/10.3390/app13137620.

[7]. Melnyk H., Melnyk V., Vikovan V. Application of natural language processing and fuzzy logic to disinformation detection. Bukovinian Math. Journal. 2024. Vol. 12, no. 1. P. 21–31. 

[8]. Melnyk H., Melnyk V. Enhancing Mood Detection in Textual Analysis through Fuzzy Logic Integration. 2024 14th International Conference on Advanced Computer Information Technologies (ACIT), Ceske Budejovice, Czech Republic, 19 September 2024. P. 23–26.

