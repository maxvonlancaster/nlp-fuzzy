# Results and Discussion (Draft)

## 1. Comparative ROUGE performance

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

## 2. Statistical significance and effect sizes

We ran paired t-tests and Wilcoxon signed-rank tests on ROUGE-1 for all ten pairwise comparisons, aligning documents by SHA-256 hash. Holm correction was applied separately to each test family (alpha = 0.05). Differences are reported as the second method minus the first.

Under the t-test, all ten comparisons remain significant after Holm correction. Under the Wilcoxon test, nine remain significant; the exception is LSA versus Random. The three highlighted comparisons are:

- **Lead versus LexRank:** -0.084 (95% bootstrap CI [-0.097, -0.073]); Holm-adjusted p = 2.8e-40 (t) and 1.0e-42 (Wilcoxon); Cohen's dz = -0.45, a medium effect.
- **LexRank versus TextRank:** +0.017 (CI [+0.009, +0.026]); Holm-adjusted p = 1.9e-4 (t) and 1.1e-4 (Wilcoxon); dz = +0.12, a small effect.
- **LSA versus Random:** +0.014 (CI [+0.005, +0.023]); Holm-adjusted p = 3.6e-3 (t) and 0.135 (Wilcoxon); dz = +0.09, negligible.

The two tests address different questions, which bears on their disagreement for LSA versus Random. The paired t-test targets the mean difference. The Wilcoxon signed-rank test ranks the absolute differences and compares the rank sums of positive and negative differences; it is a test of the median only under a symmetry assumption, which we do not establish here. For LSA minus Random, the 1,000 paired differences show 469 positive, 467 negative and 64 zero values, with mean +0.0140, median 0, 10% trimmed mean +0.0070 and skewness +0.462. The mean exceeds the trimmed mean and median while the sign counts are nearly balanced. These observations are consistent with a small number of large positive differences in the right tail driving the mean, while the typical document shows little shift; they are consistent with, but do not prove, this explanation for the differing p-values. The LSA advantage over Random is therefore practically negligible and is not consistently supported by both tests.

Statistical significance at this sample size does not indicate practical importance. Effect sizes, particularly dz, should guide interpretation: the Lead-versus-LexRank gap is substantial, whereas the LexRank-versus-TextRank and LSA-versus-Random gaps are small.

## 3. Positional reference bias and robustness

Because MTSamples `description` fields are short clinical summaries, we measured how much of each description's unique token set appears in the first and last 100 words of the transcription, for 864 documents with at least 200 words. First-segment recall averaged 0.861 (median 1.000), compared with 0.357 for the last segment (median 0.312). Positional bias (first minus last recall) was positive for 87.5% of documents.

Positional bias was associated with Lead's advantage over LexRank (Spearman rho = 0.28, p = 1.7e-16). Quartiles of positional bias showed a mean Lead-minus-LexRank advantage of +0.009 in the lowest quartile, rising to +0.104, +0.130 and +0.131 in the upper three. Controlling for document and reference length by rank residualisation left the association largely unchanged (partial rho = 0.26, p = 7.7e-15). The positive-bias group exceeded the non-positive group by +0.144 (95% CI [+0.117, +0.172]), or +0.128 (CI [+0.101, +0.157]) after length adjustment. The direction therefore remains consistent under length control.

These findings describe associations, not causes. Positional bias is computed from lexical overlap between the reference and the transcript, so part of the association may reflect shared tokens rather than a genuine preference for leading content in the reference.

## 4. Discussion and limitations

The results suggest that Lead-3 outperforms the unsupervised graph and latent methods on this dataset, and that the reference set favours opening content. This is a property of the evaluation setting, not evidence of a universally superior algorithm. Rankings could change on datasets with different structure, such as corpora whose references are not drawn from the opening text; we did not test such datasets.

The dataset audit showed substantial redundancy: of 4,750 eligible records with descriptions, only 2,248 transcriptions are unique, and many descriptions are shared across records. We used exact transcription-text deduplication, which removes identical transcriptions but not near-duplicates. Results should be read as a single sample (seed 42) rather than a population estimate.

Further limitations include: only ROUGE-1 was tested for significance in the paired analysis; the random baseline depends on a single seeding scheme; and summary length was fixed at three sentences. `description` is a proxy reference rather than a verified gold-standard summary, so ROUGE scores measure agreement with that proxy, not summary quality in an absolute sense.
