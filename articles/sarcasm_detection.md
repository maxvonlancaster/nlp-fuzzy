# Sarcasm Detection in Social Media Text via Mood and Emoji Signal Fusion

## Abstract

Sarcasm detection in social media remains challenging due to the subtle linguistic markers and contextual nuances required to identify when positive expressions mask negative sentiments. This work proposes a novel approach combining sentiment lexicon-based mood scoring with emoji presence analysis to detect sarcasm in tweets and social media text. The method leverages pre-trained word embeddings (GloVe) to construct an extensive adjective lexicon with inferred sentiment scores using support vector machine classification. Mood variations within local text windows are tracked alongside emoji frequency distributions learned from a corpus of labeled sarcastic and ironic tweets. A fusion rule combining mood discontinuity signals with emoji presence indicators provides tentative sarcasm detection. While the approach demonstrates feasibility, formal validation and expansion of the feature vocabulary remain necessary.

## Keywords

sarcasm detection, social media analysis, sentiment analysis, emoji sentiment, mood tracking, word embeddings, support vector machines

## 1. Introduction

Sarcasm is a linguistic phenomenon in which speakers convey meaning that is often opposite to the literal interpretation of their words. Automatic sarcasm detection in social media has significant applications in content moderation, sentiment analysis refinement, and understanding user intent across platforms. Traditional sentiment analysis systems frequently fail on sarcastic content, misclassifying highly negative sarcastic remarks as positive due to the presence of positive lexical items or emojis that contradict the underlying meaning.

The challenge of sarcasm detection arises from multiple sources. Linguistic theory suggests that sarcasm often involves a pronounced shift in emotional valence: an utterance beginning with positive sentiment may abruptly express frustration or disdain, creating a discontinuity that signals the speaker's true intent. Second, the choice of visual markers—particularly emojis—offers contextual clues in the social media environment. Certain emojis, such as 😒 (unamused face) or 😂 (face with tears of joy), occur frequently in sarcastic tweets and may indicate sarcastic intent.

This investigation explores whether mood shifts within local text windows and emoji presence distributions can provide discriminative features for sarcasm detection. The core hypothesis is that sarcasm manifests as a measurable discontinuity in sentiment valence coupled with strategic emoji placement. A multi-step pipeline is proposed: first, enriching a sentiment lexicon using pre-trained word embeddings and machine learning; second, computing mood-based features from lemmatized text; third, analyzing emoji distributions in the target dataset; and finally, combining these signals through a fusion rule.

## 2. Dataset and Problem Formulation

### 2.1 Dataset Description

The study uses a labeled corpus designated `sarcasm_train2.csv`, which contains Twitter posts with manually annotated classes. Each record comprises a tweet text and a class label drawn from four categories: Irony, Sarcasm, Regular, and Figurative (combining both irony and sarcasm markers). The dataset includes emoji usage, punctuation patterns, and diverse linguistic constructs typical of social media discourse.

### 2.2 Emoji Corpus Analysis

An analysis of emoji frequencies across the dataset reveals that emoji usage follows a power-law distribution. The most frequently occurring emoji is 😂 (face with tears of joy), appearing 2196 times in the corpus. Other highly frequent emojis include 😒 (unamused face, 403 occurrences), 😑 (expressionless face, 179 occurrences), 😉 (winking face, 136 occurrences), and 😄 (smiling face with open mouth, 109 occurrences). This distribution suggests that certain emojis are strongly associated with specific sentiment contexts or communication styles.

## 3. Proposed Methodology

### 3.1 Sentiment Lexicon Construction

The method begins with a manually curated seed lexicon of 41 adjectives with hand-assigned mood scores. The lexicon maps adjectives such as "peaceful" (score 0.9), "joyful" (0.9), "hate" (0.0), and "furious" (0.0) to continuous [0, 1] valence values. This seed lexicon serves as a labeled training set.

To extend the lexicon to a broader adjective vocabulary, we use pre-trained GloVe word embeddings (glove-wiki-gigaword-100, 100-dimensional). Each adjective in a comprehensive English adjective word list is located in the embedding space if available. The embeddings are then used as features in a supervised classification task to infer sentiment scores for adjectives beyond the seed set.

### 3.2 Machine Learning Model for Sentiment Extrapolation

A support vector machine with radial basis function (RBF) kernel is trained on the seed lexicon embeddings. The hyperparameters are set to $C = 1.0$ and $\gamma = \text{scale}$. The model is trained to predict binary sentiment classes: positive (class +1) and negative (class −1). The training set comprises the seed adjectives and their embedding vectors, yielding a binary classifier.

Once trained, the SVM is applied to predict sentiment classes for all adjectives in the corpus that possess valid GloVe embeddings. This yields a sentiment-labeled vocabulary of adjectives that can be used during text analysis.

### 3.3 Mood Series Extraction from Text

Given an input text, the following preprocessing steps are applied:

1. Tokenization using NLTK's word tokenizer.
2. Lemmatization using NLTK's WordNetLemmatizer to reduce inflected word forms to their base.
3. Lowercasing all tokens.

For each position $i$ in the lemmatized token sequence, a lookup is performed in the trained sentiment lexicon. If the token is present, the associated sentiment score (mood value) is recorded at position $i$. Tokens not in the lexicon do not contribute to the mood series. Similarly, emojis encountered in the text are mapped to their presence and frequency; emoji sentiment scores are derived separately (described below).

### 3.4 Emoji Sentiment Scaling

The raw frequency distribution of emojis in the sarcasm corpus follows a power-law pattern. To normalize and scale emoji sentiment contributions, a logarithmic transformation is applied. For each emoji $e$ with frequency count $n_e$, the scaled score is computed as:

$$s_e = \frac{\log_a(n_e)}{\max_j \log_a(n_j)}$$

where $a = 2.7$ is the logarithm base and the denominator normalizes scores to the [0, 1] range. This scaling reduces the dominance of the most frequent emoji while preserving the relative ranking of emoji prevalence.

### 3.5 Sarcasm Detection via Signal Fusion

The core detection mechanism employs two feature functions evaluated within a sliding window centered at each position. Let $U_x$ denote the set of positions within a window of size $w$ centered at position $x$. The feature functions are defined as:

$$f_e(x) = \max_{x^* \in U_x} |emoj(x^*) - emoj(x)|$$

$$f_m(x) = \begin{cases}
1, & \text{if } \max_{x^* \in U_x} |mood(x^*) - mood(x)| = 2 \\
0, & \text{otherwise}
\end{cases}$$

The function $f_m(x)$ detects positions where mood valence exhibits maximum discontinuity (a shift from very negative to very positive or vice versa), which is characteristic of sarcasm. The function $f_e(x)$ captures emoji sentiment variability within the window.

A fusion rule combines these signals:

$$f(x) = f_m(x) \cdot f_e(x)$$

Positions where both the mood discontinuity and emoji sentiment signals are non-zero are flagged as tentative sarcasm indicators.

## 4. Experimental Setup

### 4.1 Implementation Details

The experiments are implemented in Python 3.10 using the following libraries:

- **gensim** for word embeddings (GloVe model: `glove-wiki-gigaword-100`)
- **scikit-learn** for machine learning models (SVM with RBF kernel)
- **NLTK** for natural language processing (tokenization, lemmatization)
- **emoji** library for emoji extraction and analysis
- **NumPy** and **pandas** for data processing and numerical operations

### 4.2 Hyperparameters and Configuration

- **SVM Configuration**: RBF kernel with $C = 1.0$ and $\gamma = \text{scale}$
- **Embedding Dimension**: 100 (GloVe embedding size)
- **Window Size for Feature Extraction**: $w = 10$ (a parameter value noted as requiring further justification)
- **Logarithm Base for Emoji Scaling**: $a = 2.7$
- **Seed Lexicon Size**: 41 manually annotated adjectives

### 4.3 Evaluation Approach

The notebook demonstrates the detection pipeline on two example texts:

1. An excerpted passage from Charles Dickens containing alternating descriptions of opposite conditions (e.g., "best of times" vs. "worst of times").
2. A sarcastic tweet: "School for 12 years, college for 4 years... then your work until you die lonely. Great 😒."

The method identifies mood transitions and emoji positions within these texts and computes the fusion scores.

## 5. Experimental Results and Discussion

### 5.1 Sentiment Lexicon Coverage

The trained SVM-based sentiment predictor extends the 41-word seed lexicon to cover the majority of the English adjective vocabulary present in the GloVe embeddings. A sample of extrapolated adjective scores demonstrates that the model assigns higher scores to semantically positive adjectives (e.g., "adorable": 0.508, "accomplished": 0.535) and lower scores to neutral or ambiguous terms. The predictions are consistent with intuitive semantic associations.

### 5.2 Emoji Frequency Analysis

Analysis of the sarcasm corpus reveals pronounced emoji frequency disparity:

| Emoji | Frequency | Normalized Score |
|-------|-----------|------------------|
| 😂 | 2196 | 1.000 |
| 😊 | 608 | 0.857 |
| 😒 | 403 | 0.857 |
| 😑 | 179 | 0.714 |
| 😁 | 172 | 0.714 |

This distribution reflects the tendency of sarcastic tweets to employ specific emojis that enhance ambiguity or add ironic emphasis to written content.

### 5.3 Sarcasm Detection Results on Examples

**Example 1: Dickens Passage**

Input: "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness, it was the spring of hope, it was the winter of despair."

Output: Extracted mood positions at indices with pairs such as (3: 1, 11: −1, 53: 1), indicating detected positive and negative sentiment transitions. No emoji contributions detected (as the text contains no emojis).

**Example 2: Sarcastic Tweet**

Input: "School for 12 years, college for 4 years... then your work until you die lonely. Great 😒."

Output: Mood marker at position 18 (positive adjective "Great" with sentiment score +1). Emoji marker at position 19 (😒 with normalized score 0.857). The fusion score at position 19 yields a non-zero value, flagging potential sarcasm.

## 6. Threats to Validity and Limitations

### 6.1 Limited Feature Vocabulary

The current approach relies exclusively on adjectives for mood scoring. A comprehensive sarcasm detection system must extend feature extraction to verbs, nouns, adverbs, and domain-specific expressions. The restriction to adjectives substantially limits detection sensitivity.

### 6.2 Insufficient Training Data and Validation

The seed lexicon comprises only 41 manually labeled adjectives. While machine learning extends this to thousands, the extrapolation has not been formally validated against gold-standard sentiment resources (e.g., SentiWordNet or curated lexicons). No cross-validation or hold-out test set evaluation is reported.

### 6.3 Window Size Selection

The window parameter ($w = 10$) is chosen empirically without theoretical or empirical justification. The sensitivity of detection performance to this hyperparameter remains unexplored. Optimal window sizes may vary by text length, user, or linguistic register.

### 6.4 Fusion Rule Simplicity

The multiplicative fusion rule $f(x) = f_m(x) \cdot f_e(x)$ is overly simplistic. No principled aggregation method (e.g., Dempster-Shafer, Bayesian fusion, or learned weighting) is employed. The relative importance of mood discontinuity versus emoji presence is not calibrated.

### 6.5 Dataset Generalization

The method is evaluated on a single dataset (`sarcasm_train2.csv`). Generalization to other social media platforms, languages, or temporal periods is unknown. The emoji frequency distribution is specific to the training corpus and may not transfer.

## 7. Conclusion and Future Work

This investigation proposes a novel approach to sarcasm detection that combines mood discontinuity detection with emoji sentiment analysis. The method leverages pre-trained word embeddings and machine learning to construct a scalable adjective sentiment lexicon, applies sliding-window mood tracking to identify emotional inversions, and incorporates emoji-based contextual signals.

The results on example texts demonstrate that the pipeline can identify candidate sarcastic markers. However, the work remains preliminary. Formal quantitative evaluation on representative test sets is required to establish the method's efficacy. The limitations in feature vocabulary, training data scale, and fusion methodology necessitate substantial extension before deployment.

Future work should:

1. Expand the feature vocabulary beyond adjectives to include all parts of speech and domain-specific markers of sarcasm.
2. Conduct cross-validation and formal performance evaluation (precision, recall, F1) on held-out test sets from the sarcasm corpus.
3. Investigate principled fusion strategies, potentially employing fuzzy logic or learned aggregation weights.
4. Perform sensitivity analysis on hyperparameters, particularly the window size and emoji scaling base.
5. Evaluate generalization across multiple datasets and social media platforms.
6. Refine the binary sentiment classification to continuous or multi-class schemes where appropriate.

The combination of linguistic and visual signals offers promising potential for improved sarcasm detection, contingent upon rigorous validation and methodological refinement.

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)* (pp. 4171–4186).

  *Note: Referenced implicitly through GloVe embeddings and transformer-era sentiment analysis. Specific BERT usage in this notebook not confirmed.*

- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532–1543).

- Scikit-learn developers. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

## Evidence Gaps Requiring Author Input

1. **Seed Lexicon Validation**: The 41-word seed lexicon lacks citation or reference to an established sentiment lexicon (e.g., SentiWordNet, AFINN). Verification of its alignment with standard resources is needed.

2. **Dataset Source and Scale**: The `sarcasm_train2.csv` dataset provenance is not documented. Number of total samples, train/validation/test split proportions, and data collection methodology are not explicitly stated.

3. **Cross-Validation and Test Performance**: No formal evaluation metrics (accuracy, precision, recall, F1-score, AUC) are reported. The method is demonstrated on two qualitative examples without quantitative performance assessment.

4. **Hyperparameter Justification**: The window size ($w = 10$), logarithm base ($a = 2.7$), and SVM hyperparameters ($C = 1.0$) lack empirical or theoretical justification.

5. **Comparison with Baselines**: No comparison with existing sarcasm detection methods, rule-based approaches, or baseline classifiers (e.g., Naive Bayes, logistic regression) is provided.

6. **Generalization Testing**: Evaluation is limited to two short example texts. Performance on a representative test subset of the full dataset and on external datasets is not reported.

7. **Emoji Sentiment Mapping**: The power-law scaling of emoji frequencies to sentiment scores lacks validation. Whether this scaling method correctly captures emoji contributions to sarcasm is unverified.

8. **Lemmatization Coverage**: The adjective-only approach omits sarcasm markers beyond adjectives. An expanded word-class coverage and its impact on detection is not explored.

9. **Statistical Significance**: No confidence intervals, significance tests, or error analysis are provided for reported results.

