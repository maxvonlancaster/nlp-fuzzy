# Methods and Experimental Setup (Draft)

## 1. Dataset and preprocessing

We used the MTSamples medical transcriptions file in `resources/mtsamples.csv`, which provides a `transcription` text and a `description` field for each record. The loader (`load_mtsamples`) applies the following steps in order. First, records whose transcription has fewer than 100 whitespace-separated words are removed (`min_text_length = 100`). Second, when cleaning is enabled, records with missing or whitespace-only descriptions are removed. Third, exact duplicates are removed by matching the transcription text, keeping the first occurrence. Only then are documents sampled: 1,000 records are drawn with pandas `sample(n=1000, random_state=42)`. Without a limit, the cleaned pool contains 2,248 unique transcriptions.

The `description` field is used as a proxy reference, not a verified gold-standard summary; ROUGE scores measure agreement with this proxy.

## 2. Extractive summarisation methods

All methods select three sentences (`N_SENTENCES = 3`) and return them in original order. Sentences are obtained with NLTK `sent_tokenize`, and fragments of one character or less are discarded.
- **Lead-3** returns the first three sentences.
- **Random** draws three sentence indices without replacement from a local NumPy `default_rng`. Its seed is derived per document from the base seed (42) and the document text via an 8-byte BLAKE2b digest, so the same document and seed give the same selection and the global NumPy random state is not used.
- **TextRank** builds a sentence graph from the TF-IDF similarity matrix. The vectoriser uses n-grams of length 1 to 5, English stop-word removal, smoothed IDF and L2-normalised rows, so the entries are cosine similarities. The diagonal is set to zero, the graph is weighted by these similarities, and PageRank is run with damping factor 0.85, at most 100 iterations and tolerance 1e-4. Sentences are ranked by score, with ties broken by original position.
- **LexRank (TF-IDF)** uses the same TF-IDF similarity matrix as TextRank. Edges are binary: two sentences are connected when their similarity is strictly greater than 0.1. The diagonal is zero, and PageRank is unweighted, with damping 0.85, at most 100 iterations and tolerance 1e-4. If PageRank fails to converge, a degree-based fallback is used. Ties are broken by original position. This method does not use sentence embeddings or PyTorch.
- **LSA** vectorises sentences with TF-IDF using n-grams of length 1 to 2, English stop words and at most 1,000 features. TruncatedSVD with five components and `random_state = 42` is applied, and each sentence is scored by the L2 norm of its latent vector. The top three sentences are selected. Tie handling follows NumPy `argsort` ordering, which we did not verify to be index-stable.

If the TF-IDF vocabulary is empty, TextRank and LexRank return the first three sentences.

## 3. Evaluation protocol

ROUGE is computed with the `rouge_score` package (ROUGE-1, ROUGE-2, ROUGE-L; F-measure; Porter stemming). The reference is the `description`, and the candidate is the three selected sentences joined by spaces. Per-document scores are aligned across methods by the SHA-256 hash of the transcription. Summarisation exceptions are recorded with their document index, and the notebook stops if any method yields a missing summary or an error; the final run recorded none.

Confidence intervals for each method's mean ROUGE score use 1,000 bootstrap resamples at the 95% level, with random seed 42.

## 4. Statistical analysis

Pairwise comparisons cover all ten pairs of the five methods on ROUGE-1. Differences are defined as the second method minus the first. For each pair we computed a paired t-test (`scipy.stats.ttest_rel`) and a Wilcoxon signed-rank test (`scipy.stats.wilcoxon` with zero differences dropped). Holm step-down correction is applied separately across the ten t-test p-values and across the ten Wilcoxon p-values, at alpha = 0.05. We report the 95% CI of the mean difference from 2,000 paired bootstrap resamples of documents (seed 42), percentile-based, and Cohen's dz, computed as the mean difference divided by the sample standard deviation (ddof = 1) of the within-document differences.

Distributional descriptors (sign counts, median, trimmed mean at 10%, sample skewness) were computed for the LSA-minus-Random differences to support interpretation of the test disagreement.

## 5. Positional-bias and robustness methodology

For positional analysis we restricted the 1,000 sampled documents to those with at least 200 whitespace-separated transcription words, so that the first and last 100-word segments do not overlap. Text is lowercased and tokenised with the pattern `[a-z0-9]+`. For each document we form the set of unique description tokens and compute recall in the first 100 words and in the last 100 words: the proportion of description tokens present in each segment. Positional bias is first-segment recall minus last-segment recall. Documents whose description contains no tokens are excluded; none were excluded in the final run.

The outcome is the per-document ROUGE-1 difference between Lead-3 and LexRank. Association is assessed with Spearman correlation between positional bias and this advantage. Document and reference length are the whitespace word counts of the transcription and of the description. For partial correlation, bias and advantage are rank-transformed, each is residualised on the ranks of the two length variables with an intercept, and the Pearson correlation of the residuals is reported. Quartiles of positional bias are assigned from average ranks, so tied values share a quartile. The positive-bias versus non-positive-bias difference in mean advantage is bootstrapped with 2,000 document-level resamples (seed 42), both on the raw outcome and on an outcome residualised on the two length variables.

These analyses are associational: the token overlap defining positional bias is itself lexical overlap with the reference.
