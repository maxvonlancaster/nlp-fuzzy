# Detailed Methodology

This document provides a comprehensive description of the experimental methodology used in this research, including hyperparameters, evaluation protocols, and reproducibility procedures.

---

## 1. Dataset

### MTSamples Medical Transcriptions

**Source:** MTSamples.com medical transcription dataset  
**Location:** `resources/mtsamples.csv`  
**Total Documents:** 4,999 medical transcriptions  
**Evaluation Set:** 1,000 documents (randomly sampled)

**Dataset Characteristics:**
- **Domain:** Medical transcriptions across 40+ medical specialties
- **Average Document Length:** ~480 words (±320 std)
- **Average Reference Length:** ~50 words (±35 std)
- **Average Compression Ratio:** 10.5:1 (±6.2)
- **Vocabulary Size:** ~25,000 unique words
- **Document Structure:** Clinical notes, procedure reports, consultation letters

**Sampling Procedure:**
```python
# Reproducible stratified sampling
info, df_data = load_mtsamples(
    limit=1000,
    seed=42,
    min_text_length=100,
    base_path='../resources'
)
```

**Quality Filters:**
- Minimum text length: 100 words
- Non-empty reference summaries
- Valid UTF-8 encoding

---

## 2. Methods Evaluated

### 2.1 Baseline Methods

#### Lead-N Baseline
**Purpose:** Strong baseline for comparison  
**Implementation:** Select first N sentences

**Hyperparameters:**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| n | 3 | Standard for extractive summarization |

#### Random Baseline
**Purpose:** Lower bound for performance

**Hyperparameters:**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| n | 3 | Match other methods |
| seed | 42 | Reproducibility |

### 2.2 Graph-Based Methods

#### TextRank
**Implementation:** PageRank on sentence similarity graph with TF-IDF

**Hyperparameters:**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| n | 3 | Standard summary length |
| similarity_metric | 'tfidf' | Classic TextRank approach |
| damping_factor | 0.85 | Standard PageRank value (Brin & Page, 1998) |
| max_iter | 100 | Sufficient for convergence |
| tol | 1e-4 | Standard convergence tolerance |
| ngram_range | (1, 5) | Capture phrases up to 5-grams |
| stop_words | 'english' | Remove common words |
| use_idf | True | Weight by inverse document frequency |
| smooth_idf | True | Prevent zero divisions |

**TF-IDF Configuration:**
```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 5),
    smooth_idf=True,
    use_idf=True,
    stop_words='english'
)
```

#### LexRank (with SBERT embeddings)
**Implementation:** PageRank on sentence similarity graph with transformer embeddings

**Hyperparameters:**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| n | 3 | Standard summary length |
| embedding_model | 'all-MiniLM-L6-v2' | Fast SBERT model, good quality/speed tradeoff |
| similarity_threshold | 0.1 | Minimum edge weight (continuous mode) |
| damping_factor | 0.85 | Standard PageRank value |
| max_iter | 100 | Sufficient for convergence |
| tol | 1e-4 | Standard convergence tolerance |
| use_continuous | True | Use weighted edges (better than binary) |

**SBERT Model Specifications:**
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Embedding Dimension:** 384
- **Max Sequence Length:** 256 tokens
- **Training:** Trained on 1B+ sentence pairs
- **Performance:** Competitive with larger models, 5x faster

#### LSA (Latent Semantic Analysis)
**Implementation:** SVD-based topic extraction from TF-IDF matrix

**Hyperparameters:**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| n | 3 | Standard summary length |
| n_components | 5 | Capture 5 latent topics |
| max_features | 1000 | Limit vocabulary for efficiency |
| ngram_range | (1, 2) | Unigrams and bigrams |
| random_state | 42 | Reproducibility |

---

## 3. Evaluation Metrics

### 3.1 ROUGE Metrics

**Implementation:** `rouge-score` library with Porter stemming

**Metrics Computed:**
- **ROUGE-1:** Unigram overlap (captures content coverage)
- **ROUGE-2:** Bigram overlap (captures fluency)
- **ROUGE-L:** Longest common subsequence (captures sentence-level structure)

**Configuration:**
```python
evaluator = RougeEvaluator(
    metrics=['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True,  # Porter stemmer
    bootstrap_samples=1000,
    confidence_level=0.95,
    random_seed=42
)
```

**Metric Interpretation:**
- ROUGE-1: Primary metric for content coverage
- ROUGE-2: Indicates phrase-level quality
- ROUGE-L: Measures structural similarity

**Reported Values:**
- F-measure (harmonic mean of precision and recall)
- Mean across all documents
- Standard deviation
- 95% confidence intervals via bootstrapping

### 3.2 Statistical Significance Testing

**Test Used:** Paired t-test (parametric)  
**Alternative:** Wilcoxon signed-rank test (non-parametric, for validation)  
**Significance Level:** α = 0.05

**Procedure:**
1. Compute per-document ROUGE-1 scores for both methods
2. Apply paired t-test on score differences
3. Report p-value and effect size (mean difference)
4. Interpret: p < 0.05 indicates statistically significant difference

**Confidence Intervals:**
- Method: Bootstrap resampling (1000 samples)
- Level: 95% confidence
- Procedure: Sample with replacement, compute mean, take percentiles

---

## 4. Experimental Protocol

### 4.1 Evaluation Pipeline

```
1. Data Loading
   └─> Load 1000 documents with seed=42
   └─> Extract transcriptions and reference summaries

2. For each method:
   └─> Generate summaries for all documents
   └─> Compute ROUGE scores against references
   └─> Compute statistics (mean, std, CI)
   └─> Store summaries for analysis

3. Statistical Analysis
   └─> Pairwise significance tests
   └─> Generate comparison tables

4. Error Analysis
   └─> Per-document characteristics
   └─> Correlation with performance
   └─> Failure pattern detection

5. Case Studies
   └─> Select high/medium/low performers
   └─> Detailed qualitative analysis
```

### 4.2 Reproducibility Measures

**Random Seed Control:**
```python
RANDOM_SEED = 42

# Set all random seeds
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Pass to all stochastic operations
- Dataset sampling: seed=RANDOM_SEED
- Random baseline: seed=RANDOM_SEED
- Bootstrap resampling: random_seed=RANDOM_SEED
- LSA/SVD: random_state=RANDOM_SEED
```

**Software Versions:**
- Python: 3.11.x
- NumPy: 1.24.3
- pandas: 2.0.x
- scikit-learn: 1.3.x
- networkx: 3.1.x
- rouge-score: 0.1.2
- sentence-transformers: 2.2.x (for LexRank)
- NLTK: 3.8.1

**Hardware:**
- CPU-based evaluation (no GPU required for most methods)
- LexRank with SBERT benefits from GPU but not required

---

## 5. Ablation Studies

### 5.1 Parameter Sensitivity Analysis

**Method:** TextRank (representative graph-based method)  
**Variable Parameter:** Number of sentences (n)  
**Range:** {2, 3, 4, 5}  
**Fixed:** All other hyperparameters  
**Evaluation Set:** 100 documents (subset for efficiency)

**Procedure:**
```python
for n_sentences in [2, 3, 4, 5]:
    result = evaluate_method(
        method=textrank_summarizer,
        documents=documents[:100],
        kwargs={'n': n_sentences, ...}
    )
    record(n_sentences, result.mean_scores)
```

### 5.2 Embedding Comparison

**Methods Tested:**
1. **TF-IDF:** Classic bag-of-words (baseline)
2. **Word2Vec:** Word embeddings averaged per sentence
3. **SBERT-Mini:** Sentence-transformers/all-MiniLM-L6-v2 (384-dim)

**Evaluation:** ROUGE-1 on 50-document subset  
**Purpose:** Quantify benefit of neural embeddings vs traditional methods

---

## 6. Error Analysis Framework

### 6.1 Document Characteristics

**Features Computed:**
```python
{
    'num_sentences': int,
    'num_words': int,
    'avg_sentence_length': float,
    'lexical_diversity': float,  # Type-token ratio
    'num_entities': int,  # Approximate (capitalized words)
    'entity_density': float,  # Entities per 100 words
    'compression_potential': float  # 1 - lexical_diversity
}
```

### 6.2 Performance Analysis

**Correlation Analysis:**
- Compute Pearson correlation between each characteristic and ROUGE-1
- Identify strongest predictors of performance
- Statistical significance testing (p-value < 0.05)

**Failure Pattern Detection:**
- Categorize documents by performance (high/medium/low)
- Compare characteristics of low vs high performers
- Identify systematic failure modes

**Error Categories:**
```python
{
    'redundancy': [],  # <70% unique words in summary
    'missing_key_info': [],  # <20% overlap with reference
    'length_mismatch': [],  # Length ratio outside [0.5, 2.0]
    'poor_coverage': []  # Summary <10 words
}
```

---

## 7. Qualitative Analysis

### 7.1 Case Study Selection

**Selection Criteria:**
1. Compute average ROUGE-1 across all methods for each document
2. Sort documents by average score
3. Select:
   - **High performer:** Top 10th percentile
   - **Medium performer:** 50th percentile (median)
   - **Low performer:** Bottom 10th percentile

**Purpose:**
- Understand when methods succeed/fail
- Identify qualitative differences between methods
- Validate quantitative findings

### 7.2 Case Study Content

For each selected document:
- Original text (first 500 words)
- Reference summary (human-written)
- All method summaries with ROUGE-1 scores
- Comparative analysis:
  - Best/worst performing methods
  - Agreement between methods
  - Error patterns
  - Domain-specific challenges

---

## 8. Results Export

### 8.1 Formats

**CSV Tables:**
- Main results: ROUGE scores with confidence intervals
- Pairwise comparisons: Significance test results
- Ablation results: Component contributions
- Per-document analysis: Characteristics and scores

**LaTeX Tables:**
- Publication-ready formatting
- Proper escaping and alignment
- Captions and labels for referencing

**JSON:**
- Complete results in machine-readable format
- Metadata: dataset, configuration, timestamps
- All metrics and statistics

**Markdown:**
- Case studies for thesis inclusion
- Formatted for readability
- Side-by-side method comparisons

### 8.2 Visualizations

**Figure Specifications:**
- Format: PNG
- Resolution: 300 DPI (publication quality)
- Style: seaborn whitegrid
- Colors: Colorblind-friendly palette

**Generated Figures:**
1. ROUGE-1 comparison (bar chart with error bars)
2. Score distributions (histograms + violin plots)
3. Correlation heatmap (characteristics vs performance)

---

## 9. Computational Requirements

### 9.1 Runtime Estimates

**Full Evaluation (1000 documents, 3-4 methods):**
- Lead-N: <1 minute
- Random: <1 minute
- TextRank: 10-20 minutes
- LSA: 15-25 minutes
- LexRank (SBERT): 30-60 minutes (CPU) or 5-10 minutes (GPU)

**Total:** ~60-90 minutes on modern CPU

### 9.2 Memory Requirements

- Peak memory: ~2-4 GB
- Primarily from:
  - Document storage: ~500 MB
  - TF-IDF matrices: ~200-500 MB
  - SBERT embeddings: ~400 MB (if used)

---

## 10. Limitations and Considerations

### 10.1 Dataset Limitations

- **Domain-specific:** Medical transcriptions may not generalize to other domains
- **Reference quality:** Human-written descriptions vary in quality and style
- **Compression ratio:** High compression (10:1) favors extreme selectivity

### 10.2 Methodological Limitations

- **Extractive only:** Cannot generate novel phrasings
- **Sentence granularity:** Cannot split or merge sentences
- **Single reference:** ROUGE scores against single reference may underestimate quality
- **Length constraint:** Fixed N=3 sentences may not suit all documents

### 10.3 Evaluation Limitations

- **ROUGE limitations:** 
  - Lexical overlap doesn't capture semantic equivalence
  - Cannot measure coherence or fluency
  - Favors extractive methods over abstractive
- **Single dataset:** Results may not generalize
- **Hyperparameter selection:** Not exhaustively tuned for each method

---

## 11. Reproducibility Checklist

✅ **Dataset:** Publicly available MTSamples dataset  
✅ **Code:** Complete implementation in `src/` directory  
✅ **Random seeds:** Fixed at 42 throughout  
✅ **Software versions:** Documented in requirements.txt  
✅ **Hyperparameters:** Fully specified in this document and config files  
✅ **Evaluation protocol:** Step-by-step procedure documented  
✅ **Statistical tests:** Method and parameters specified  
✅ **Hardware:** CPU-based, no specialized hardware required  
✅ **Results:** Complete outputs exported in multiple formats  

**To Reproduce:**
```bash
cd src
jupyter notebook summarization_evaluation.ipynb
# Run all cells (Set N_DOCUMENTS=1000 for full evaluation)
```

---

## References

**Datasets:**
- MTSamples: https://www.mtsamples.com/

**Methods:**
- Mihalcea & Tarau (2004). TextRank. EMNLP.
- Erkan & Radev (2004). LexRank. JAIR, 22.
- Steinberger & Ježek (2004). LSA in Summarization. ISIM.

**Evaluation:**
- Lin (2004). ROUGE: A Package for Automatic Evaluation. ACL.

**Embeddings:**
- Reimers & Gurevych (2019). Sentence-BERT. EMNLP.
