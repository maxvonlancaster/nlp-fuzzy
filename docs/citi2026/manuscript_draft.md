# Positional Reference Bias in Evaluating Unsupervised Extractive Summarization of Medical Transcriptions

## Abstract

Automatic evaluation of extractive summarisation relies on ROUGE scores against reference texts, yet the rankings it produces may depend on how references relate to source structure. We compare five unsupervised extractive methods (Lead-3, Random, TextRank, LexRank and LSA) on 1,000 cleaned MTSamples transcriptions, selecting three sentences per summary and scoring against the short `description` field, a proxy reference rather than a gold-standard summary. Lead-3 achieves the highest mean ROUGE-1 (0.309), followed by LexRank (0.224), TextRank (0.207), LSA (0.170) and Random (0.156). Among 864 documents with at least 200 words, description tokens overlap transcription openings more than their endings (mean recall 0.861 versus 0.357), and this positional overlap correlates with Lead's advantage over LexRank (rho = +0.275). However, when descriptions are reassigned across documents in 100 stratified permutations, the shuffled median correlation is +0.267, and the observed value lies within the shuffled distribution. On the same subset, the Lead-minus-LexRank mean difference is +0.092 with the true references and +0.001 under shuffling. The correlation therefore does not provide independent evidence of a pairing-specific positional mechanism. It reveals a diagnostic limitation of reference-based evaluation on this corpus, and benchmark rankings on proxy references should be interpreted with that limitation in mind.

**Keywords:** positional reference bias; negative controls; reference-based evaluation; ROUGE; unsupervised extractive summarisation; medical transcriptions

## 1. Introduction

Unsupervised extractive summarisation selects sentences from a source document without training data, which makes it attractive when labelled summaries are scarce or costly to produce. Graph-based methods such as TextRank and LexRank rank sentences by their centrality in a similarity graph, while algebraic methods such as latent semantic analysis (LSA) project sentences into a low-dimensional topic space and score them by their latent weight. These approaches remain common baselines.

The most common metric for this comparison is ROUGE, which measures n-gram overlap between a system summary and one or more reference summaries. ROUGE is convenient and reproducible, but its rankings depend on the references that are used. When references are short, templated or written from a particular part of the source, a metric can reward methods whose selections happen to match that structure rather than methods that select the most informative content. Document structure compounds the problem: if a reference draws mainly on the opening of a document, a simple lead baseline may score highly regardless of the quality of more sophisticated selection strategies.

This concern is particularly relevant for clinical text. Medical transcriptions are long and heterogeneous, and the short descriptions attached to them are commonly used as proxy references. Such descriptions are not verified gold-standard summaries, so a comparison of methods on them measures agreement with a proxy rather than summary quality. Positional bias has been examined in news summarisation (Grenander et al., 2019), and to our knowledge comparable diagnostics for classical unsupervised methods on clinical transcriptions have received less attention.

We therefore address a narrower gap: positional reference bias has been studied in news, and its relation to algorithm rankings on proxy clinical references appears to have received limited examination with classical methods and length-controlled checks. Our objective is to compare five unsupervised extractive methods (Lead-3, Random, TextRank, LexRank and LSA) on a cleaned medical transcription corpus, and to assess whether the observed ranking is accompanied by a positional bias in the reference. We ask three research questions. First, how do the methods compare on ROUGE-1, ROUGE-2 and ROUGE-L, and which pairwise differences survive multiple-comparison correction? Second, is the description's token overlap with the opening of a transcription associated with the advantage of Lead over LexRank? Third, does that association persist after controlling for document and reference length, and how does it compare with alternative transcript windows and reassigned descriptions?

The contributions are fourfold. We report a five-method comparison of graph-based, algebraic and baseline extractive methods under a single, reproducible protocol. We describe a reproducibly cleaned evaluation sample of MTSamples, obtained by removing records with missing descriptions and exact duplicate transcriptions, together with an audit of the redundancy in the raw file; the sample is not released as a public dataset. We apply paired statistical analysis with Holm correction, complemented by Wilcoxon tests, bootstrap confidence intervals and effect sizes, so that significance and practical magnitude are reported separately. Finally, we add positional-bias diagnostics with quartile analysis, length-controlled robustness checks, an alternative-window control and a stratified shuffled-reference control with 100 permutations.

The positional analysis is framed as an evaluation concern rather than a demonstrated mechanism. Its results describe associations in one dataset, and they do not establish that the reference was written from the opening text or that any method is superior in general.

## 2. Related Work

### 2.1 Graph-based and algebraic extractive summarisation

Graph-based extractive methods treat sentences as nodes and their pairwise similarities as edges. TextRank applies the PageRank ranking model to such a graph to select sentences (Mihalcea and Tarau, 2004). LexRank similarly ranks sentences by their centrality in a lexical similarity graph, and is commonly applied with TF-IDF-based similarity (Erkan and Radev, 2004). Algebraic methods instead project sentences into a latent space. Latent semantic analysis (LSA) decomposes a term-sentence matrix with a singular value decomposition and selects sentences according to their weight in the leading latent topics (Steinberger and Ježek, 2004), a foundational formulation for LSA-based summarisation. Our implementation uses scikit-learn's TruncatedSVD on TF-IDF features, which is not identical to the original method. These methods are unsupervised and interpretable, and they remain common baselines because they require no training data.

### 2.2 Automatic evaluation and its limitations

ROUGE measures n-gram overlap between a candidate summary and reference summaries, and is the standard automatic metric for extractive and abstractive summarisation (Lin, 2004). Its rankings depend on the references used, and several studies have questioned whether ROUGE-based comparisons remain stable across evaluation settings. Bhandari et al. (2020) re-examine evaluation practice in text summarisation, and Mao et al. (2020) propose reference-based facet-aware evaluation for extractive summarisation, which matches source support sentences to reference facets rather than relying on whole-summary overlap alone. Fabbri et al. (2021) re-evaluate automatic summarisation metrics, including ROUGE, against human judgements. Gigant et al. (2024) address reference quality by using reference-free metrics, and Casola et al. (2025) investigate how variation in the reference set affects summarisation evaluation. These studies examine metric and reference reliability in general; they do not analyse positional structure in clinical transcriptions, and our experiments do not test reference-free metrics or reference-set variation. These works motivate examining how reference characteristics shape the outcome of a comparison, rather than relying on aggregate scores alone.

### 2.3 Positional bias and the motivation for this study

Positional bias has been studied in news summarisation. Grenander et al. (2019) study lead bias in neural news summarisation, in which the opening sentences of an article carry much of the reference content, and propose multi-stage training with auxiliary losses to counter its effects. Their setting involves trained neural models on news corpora. Our study differs in three respects. It uses classical, training-free extractive methods; it operates on clinical transcriptions rather than news; and its references are the short `description` fields of MTSamples, which are proxy references and not verified gold-standard summaries. We do not introduce a new summarisation algorithm. Instead, we provide a dataset-specific empirical analysis that compares five unsupervised methods, measures the association between positional reference overlap and the advantage of Lead over LexRank, and checks whether that association persists after controlling for document and reference length.

## 3. Methods and Experimental Setup

### 3.1 Dataset and preprocessing

We used the MTSamples file `resources/mtsamples.csv` (`transcription` and `description` fields). The loader (`load_mtsamples`) applies these steps in order. First, records whose transcription has fewer than 100 whitespace-separated words are removed (`min_text_length = 100`). Second, when cleaning is enabled, records with missing or whitespace-only descriptions are removed. Third, exact duplicates are removed by matching the transcription text, keeping the first occurrence. Only then are documents sampled: 1,000 records are drawn with pandas `sample(n=1000, random_state=42)`. Without a limit, the cleaned pool contains 2,248 unique transcriptions. Descriptions are repetitive: in the raw file (4,999 records), 2,139 descriptions occur more than once and cover 4,790 records, and template-like texts such as "An example/template for a routine normal male physical exam" recur up to 12 times. The cleaning steps do not remove repeated descriptions, and their share within the cleaned sample was not separately quantified.

The `description` field is used as a proxy reference, not a verified gold-standard summary; ROUGE scores measure agreement with this proxy.

The transcriptions come from the publicly available MTSamples collection (Boyle, n.d.), which is originally sourced from mtsamples.com. The exact acquisition route and licence of the local CSV have not been independently established; the Kaggle listing reports CC0, which does not verify the local file's licence.

*Data availability.* The public GitHub repository for this study includes the source CSV (`resources/mtsamples.csv`), the scripts, and the per-document evaluation scores keyed by SHA-256 hash. The 1,000-document evaluation sample is reproducibly derived with the documented cleaning procedure and seed 42. The cleaned sample and generated summaries are not separately published.

### 3.2 Extractive summarisation methods

All methods select three sentences (`N_SENTENCES = 3`) and return them in original order. Sentences are obtained with NLTK `sent_tokenize`, and fragments of one character or less are discarded.
- **Lead-3** returns the first three sentences.
- **Random** draws three sentence indices without replacement from a local NumPy `default_rng`. Its seed is derived per document from the base seed (42) and the document text via an 8-byte BLAKE2b digest, so the same document and seed give the same selection and the global NumPy random state is not used.
- **TextRank** builds a sentence graph from the TF-IDF similarity matrix. The vectoriser uses n-grams of length 1 to 5, English stop-word removal, smoothed IDF and L2-normalised rows, so the entries are cosine similarities. The diagonal is set to zero, the graph is weighted by these similarities, and PageRank is run with damping factor 0.85, at most 100 iterations and tolerance 1e-4. Sentences are ranked by score, with ties broken by original position.
- **LexRank (TF-IDF)** uses the same TF-IDF similarity matrix as TextRank. Edges are binary: two sentences are connected when their similarity is strictly greater than 0.1. The diagonal is zero, and PageRank is unweighted, with damping 0.85, at most 100 iterations and tolerance 1e-4. If PageRank fails to converge, a degree-based fallback is used. Ties are broken by original position. This is the TF-IDF-based, thresholded, unweighted graph variant implemented for this study; it was not verified against the full specification of the original LexRank system. Because TextRank uses a weighted graph and LexRank a binary one, the LexRank-TextRank comparison is configuration-dependent rather than a clean comparison of the two algorithms.
- **LSA** vectorises sentences with TF-IDF using n-grams of length 1 to 2, English stop words and at most 1,000 features. TruncatedSVD with at most five components, computed as `min(5, min(tfidf_matrix.shape) - 1)`, and `random_state = 42` is applied; each sentence is scored by the L2 norm of its latent vector. The top three sentences are selected via NumPy `argsort` without explicit positional tie-breaking, so no index-stable ordering is guaranteed for tied scores.

If the TF-IDF vocabulary is empty, TextRank and LexRank return the first three sentences. All parameters were fixed in the configuration and notebook cells; no hyperparameter tuning procedure was run, and the parameter-sensitivity ablation was deferred. The origin of the n-gram ranges and the LSA component limit is not recorded in the repository.

### 3.3 Evaluation protocol

ROUGE is computed with the `rouge_score` package (ROUGE-1, ROUGE-2, ROUGE-L; F-measure; Porter stemming). The reference is the `description`, and the candidate is the three selected sentences joined by spaces. Per-document scores are aligned across methods by the SHA-256 hash of the transcription. Exceptions are recorded per document, and the notebook stops on any missing summary or error; the final run recorded none.

Confidence intervals for each method's mean ROUGE score use 1,000 bootstrap resamples at the 95% level, with random seed 42. These bootstrap statistics reset the global NumPy RNG to seed 42; the Random baseline uses an independent document-seeded local generator and is unaffected.

### 3.4 Statistical analysis

Pairwise comparisons cover all ten pairs of the five methods on ROUGE-1. Differences are defined as the second method minus the first. For each pair we computed a paired t-test (`scipy.stats.ttest_rel`) and a Wilcoxon signed-rank test (`scipy.stats.wilcoxon` called with `zero_method='wilcox'`; the script does not set the `method` argument, so the library default applies and is not specified here). Holm step-down correction (Holm, 1979) is applied separately across the ten t-test p-values and across the ten Wilcoxon p-values, at alpha = 0.05. We report the 95% CI of the mean difference from 2,000 paired bootstrap resamples of documents (seed 42), percentile-based, and Cohen's dz, computed as the mean difference divided by the sample standard deviation (ddof = 1) of the within-document differences.

Distributional descriptors (sign counts, median, 10% trimmed mean, skewness) were computed for LSA-minus-Random differences to interpret the test disagreement.

### 3.5 Positional-bias and robustness methodology

For positional analysis we restricted the 1,000 sampled documents to those with at least 200 whitespace-separated transcription words, so that the first and last 100-word segments do not overlap. Text is lowercased and tokenised with the pattern `[a-z0-9]+`. For each document we form the set of unique description tokens and compute recall in the first 100 words and in the last 100 words: the proportion of description tokens present in each segment. Positional bias is first-segment recall minus last-segment recall. Documents whose description contains no tokens are excluded; none were excluded in the final run.

The outcome is the per-document ROUGE-1 difference between Lead-3 and LexRank. Association is assessed with Spearman correlation between positional bias and this advantage. Document and reference length are the whitespace word counts of the transcription and of the description. For partial correlation, bias and advantage are rank-transformed, each is residualised on the ranks of the two length variables with an intercept, and the Pearson correlation of the residuals is reported. Quartiles of positional bias are assigned from average ranks, so tied values share a quartile. The positive-bias versus non-positive-bias difference in mean advantage is bootstrapped with 2,000 document-level resamples (seed 42), both on the raw outcome and on an outcome residualised on the two length variables.

These analyses are associational: the token overlap defining positional bias is itself lexical overlap with the reference.

### 3.6 Negative-control methodology

Two negative controls were applied. The alternative-position control defines non-overlapping first, centred middle and last 100-word windows for the 672 documents with at least 300 words, computes description-token recall in each window, and correlates it with the Lead-minus-LexRank ROUGE-1 advantage. Differences between correlations are estimated by a paired document-level bootstrap (2,000 resamples, seed 42), with and without rank residualisation on document and reference length. The shuffled-reference control reassigns descriptions within strata of medical specialty and reference-length tercile over 100 permutations (n = 864, seed 42, local NumPy generator). Lead-3 and LexRank summaries were reconstructed once, matched to saved per-document scores within 1e-9, and reused across permutations. The first control tests whether the association is specific to the opening but cannot remove shared-reference coupling, because one description enters both measures. The second tests dependence on the true description-transcript pairing while preserving that coupling within each permutation. Neither control establishes causality.

## 4. Results and Discussion

### 4.1 Comparative ROUGE performance

We evaluated five unsupervised extractive methods on 1,000 cleaned MTSamples transcriptions (seed 42), selecting three sentences per summary. Table 1 reports mean ROUGE F-scores over documents.

**Table 1.** Mean ROUGE scores by method (1,000 documents).

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-3 | 0.309 | 0.243 | 0.288 |
| LexRank (TF-IDF) | 0.224 | 0.143 | 0.197 |
| TextRank (TF-IDF) | 0.207 | 0.127 | 0.180 |
| LSA | 0.170 | 0.083 | 0.145 |
| Random | 0.156 | 0.065 | 0.126 |

Lead-3 scores highest on all three metrics. Among the graph and latent methods, LexRank leads, followed by TextRank and LSA. Random selection is the lowest.

### 4.2 Statistical significance and effect sizes

We ran paired t-tests and Wilcoxon signed-rank tests on ROUGE-1 for all ten pairwise comparisons, aligning documents by SHA-256 hash. Holm correction was applied separately to each test family (alpha = 0.05). Differences are reported as the second method minus the first.

Under the t-test, all ten comparisons remain significant after Holm correction. Under the Wilcoxon test, nine remain significant; the exception is LSA versus Random. The three highlighted comparisons are:

- **Lead versus LexRank:** -0.084 (95% bootstrap CI [-0.097, -0.073]); Holm-adjusted p = 2.8e-40 (t) and 1.0e-42 (Wilcoxon); Cohen's dz = -0.45, a medium effect.
- **LexRank versus TextRank:** +0.017 (CI [+0.009, +0.026]); Holm-adjusted p = 1.9e-4 (t) and 1.1e-4 (Wilcoxon); dz = +0.12, a small effect.
- **LSA versus Random:** +0.014 (CI [+0.005, +0.023]); Holm-adjusted p = 3.6e-3 (t) and 0.135 (Wilcoxon); dz = +0.09, negligible.

The two tests address different questions, which bears on their disagreement for LSA versus Random. The paired t-test targets the mean difference. The Wilcoxon signed-rank test ranks the absolute differences and compares the rank sums of positive and negative differences; it is a test of the median only under a symmetry assumption, which we do not establish here. For LSA minus Random, the 1,000 paired differences show 469 positive, 467 negative and 64 zero values, with mean +0.0140, median 0, 10% trimmed mean +0.0070 and skewness +0.462. The mean exceeds the trimmed mean and median while the sign counts are nearly balanced. These observations are consistent with a small number of large positive differences in the right tail driving the mean, while the typical document shows little shift; they are consistent with, but do not prove, this explanation for the differing p-values. The LSA advantage over Random is therefore practically negligible and is not consistently supported by both tests.

Statistical significance at this sample size does not indicate practical importance. Effect sizes, particularly dz, should guide interpretation: the Lead-versus-LexRank gap is a medium effect (dz = -0.45), whereas the LexRank-versus-TextRank and LSA-versus-Random gaps are small.

### 4.3 Positional reference bias and robustness

Because MTSamples `description` fields are short clinical summaries, we measured how much of each description's unique token set appears in the first and last 100 words of the transcription, for 864 documents with at least 200 words. First-segment recall averaged 0.861 (median 1.000), compared with 0.357 for the last segment (median 0.312). Positional bias (first minus last recall) was positive for 87.5% of documents.

Positional bias was associated with Lead's advantage over LexRank (Spearman rho = 0.28, p = 1.7e-16). Quartiles of positional bias showed a mean Lead-minus-LexRank advantage of +0.009 in the lowest quartile, rising to +0.104, +0.130 and +0.131 in the upper three. Controlling for document and reference length by rank residualisation left the association largely unchanged (partial rho = 0.26, p = 7.7e-15). The positive-bias group exceeded the non-positive group by +0.144 (95% CI [+0.117, +0.172]), or +0.128 (CI [+0.101, +0.157]) after length adjustment. The direction therefore remains consistent under length control. Section 4.4 reports negative controls that qualify this association.

These findings describe associations, not causes. Positional bias is computed from lexical overlap between the reference and the transcript, so part of the association may reflect shared tokens rather than a genuine preference for leading content in the reference.

### 4.4 Negative-control results

Window correlations with the Lead-minus-LexRank advantage were +0.245 for the first window, -0.091 for the middle and -0.166 for the last (n = 672). The first-minus-middle difference was +0.335 (95% CI [+0.237, +0.428]) and the first-minus-last difference +0.412 (CI [+0.305, +0.523]), so the positive association is specific to the opening rather than shared across positions.

Under the shuffled-reference control (n = 864, 100 permutations), the observed positional-bias correlation was +0.275. The shuffled median was +0.267, with a 2.5th-97.5th percentile interval of [+0.226, +0.317] and a one-sided exceedance of 0.33. The observed correlation is therefore compatible with the shuffled-reference distribution. These percentiles describe the control distribution, not a confidence interval for the observed correlation.

The mean advantage behaves differently. On the same 864 documents, the original Lead-minus-LexRank mean ROUGE-1 difference was +0.092, against a shuffled median of +0.001. On this dataset and control, Lead's mean advantage depends on the true reference pairing; this is a separate finding from the positional correlation and is specific to this corpus and control. Reassigned descriptions changed text in 99.94% of cases on average.

The opening-window association is stronger than for other positions, but the evidence for a pairing-specific positional mechanism is weak. The controls neither isolate mathematical coupling nor establish causality.

### 4.5 Discussion and limitations

The results suggest that Lead-3 outperforms the unsupervised graph and latent methods on this dataset, and that description tokens overlap opening content more than closing content. This is a property of the evaluation setting, not evidence of a universally superior algorithm. Rankings could change on datasets with different structure, such as corpora whose references are not drawn from the opening text; we did not test such datasets.

The dataset audit showed substantial redundancy: of 4,750 eligible records with descriptions, only 2,248 transcriptions are unique, and many descriptions are shared across records. We used exact transcription-text deduplication, which removes identical transcriptions but not near-duplicates. Results should be read as a single sample (seed 42) rather than a population estimate. Repeated, template-like descriptions also remain in the evaluation sample; their effect on the ROUGE rankings was not separately measured.

Further limitations include: only ROUGE-1 was tested for significance in the paired analysis; the random baseline depends on a single seeding scheme; summary length was fixed at three sentences; and parameters were fixed without tuning, with the parameter-sensitivity analysis deferred. `description` is a proxy reference rather than a verified gold-standard summary, so ROUGE scores measure agreement with that proxy, not summary quality in an absolute sense.

## 5. Conclusions

This study compared five unsupervised extractive methods on 1,000 cleaned MTSamples transcriptions, using short descriptions as proxy references. For the first question, Lead-3 obtained the highest ROUGE scores, and all ten pairwise t-tests remained significant after Holm correction. LexRank outperformed TextRank by a small margin (dz = 0.12), and the Wilcoxon test did not support the LSA-over-Random difference, which has negligible effect size.

For the second question, description tokens overlapped the opening 100 words of transcriptions far more than their endings, and this overlap correlated positively with Lead's advantage over LexRank (rho = +0.275). For the third, the correlation persisted after rank residualisation on document and reference length, and the alternative-window control indicated that it is specific to the opening window. However, under 100 stratified reassignments of descriptions, the shuffled median correlation was +0.267, and the observed value lay within the shuffled distribution. The true-reference Lead advantage (+0.092) also exceeded its shuffled median (+0.001), a separate finding from the correlation.

These results separate three things: Lead-3's observed performance on this corpus, the positional overlap of descriptions with transcription openings, and a correlation that is robust to length but does not distinguish genuine document-reference pairing from the shuffled control. The negative controls reveal an evaluation-diagnostic limitation: shared lexical structure between references and transcripts can produce an association without establishing a positional mechanism. They do not demonstrate causality. Future work should test these diagnostics on corpora with independently written references.

## References

Bhandari, M., Gour, P. N., Ashfaq, A., Liu, P., and Neubig, G. (2020). Re-evaluating evaluation in text summarization. In *Proceedings of EMNLP 2020*. https://aclanthology.org/2020.emnlp-main.751/

Boyle, T. (n.d.). Medical Transcriptions [Dataset]. Kaggle. https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

Casola, S., Liu, Y. J., Peng, S., Kraus, O., Gatt, A., and Plank, B. (2025). References matter: Investigating the impact of reference set variation on summarization evaluation. In *Proceedings of INLG 2025*, pp. 274–291. https://aclanthology.org/2025.inlg-main.18/

Erkan, G., and Radev, D. R. (2004). LexRank: Graph-based lexical centrality as salience in text summarization. *Journal of Artificial Intelligence Research*, 22, 457–479. https://doi.org/10.1613/JAIR.1523

Fabbri, A. R., Kryściński, W., McCann, B., Xiong, C., Socher, R., and Radev, D. (2021). SummEval: Re-evaluating summarization evaluation. *Transactions of the Association for Computational Linguistics*, 9, 391–409. https://aclanthology.org/2021.tacl-1.24/

Gigant, T., Guinaudeau, C., Decombas, M., and Dufaux, F. (2024). Mitigating the impact of reference quality on evaluation of summarization systems with reference-free metrics. In *Proceedings of EMNLP 2024*, pp. 19355–19368. https://aclanthology.org/2024.emnlp-main.1078/

Grenander, M., Dong, Y., Cheung, J. C. K., and Louis, A. (2019). Countering the effects of lead bias in news summarization via multi-stage training and auxiliary losses. In *Proceedings of EMNLP-IJCNLP 2019*. https://aclanthology.org/D19-1620/

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. https://www.jstor.org/stable/4615733

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *Text Summarization Branches Out*. https://aclanthology.org/W04-1013/

Mao, Y., Liu, L., Zhu, Q., Ren, X., and Han, J. (2020). Facet-aware evaluation for extractive summarization. In *Proceedings of ACL 2020*. https://aclanthology.org/2020.acl-main.445/

Mihalcea, R., and Tarau, P. (2004). TextRank: Bringing order into text. In *Proceedings of EMNLP 2004*. https://aclanthology.org/W04-3252/

Steinberger, J., and Ježek, K. (2004). Using latent semantic analysis in text summarization and summary evaluation. In *Proceedings of ISIM 2004*. https://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf
