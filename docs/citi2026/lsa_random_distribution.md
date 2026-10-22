# LSA minus Random: Paired ROUGE-1 Differences (Final Run)

Written for: CITI-2026 reviewers checking the LSA-versus-Random result.

Source: `results/citi2026_final/tables/per_document_scores.csv`, aligned by SHA-256 (1000 documents). Differences are LSA minus Random ROUGE-1; positive favours LSA.

## Counts

- Positive: 469 (46.9%)
- Negative: 467 (46.7%)
- Zero: 64 (6.4%)
- Among non-zero differences, positive share: 50.1%

## Distribution

- Mean: +0.0140
- Median: +0.0000
- Standard deviation: 0.1513
- 10th percentile: -0.1428; 90th percentile: +0.1988
- Skewness (sample, bias-corrected): +0.462
- 10% trimmed mean: +0.0070

## Test results (for reference)

- Paired t-test: t = 2.921, p = 3.569e-03
- Wilcoxon signed-rank (zero differences dropped): W = 206905.0, p = 1.354e-01

## Interpretation

The paired t-test compares the mean difference against zero, so it is driven by the average shift and by the spread of the differences. The Wilcoxon signed-rank test compares the ranks of the absolute differences by sign, so it measures whether positive differences tend to be larger than negative ones, and it is less sensitive to the size of individual differences. These are different statistical properties, so the two tests can disagree when the mean and the typical (median or rank-based) difference point in different directions or when a few large differences dominate the mean.

The measurements above show how the mean, median, trimmed mean and skewness relate for this distribution; they do not identify a cause for the discrepancy, and the report does not attribute one.
