# Positional-Bias Robustness: Final Run

Written for: CITI-2026 reviewers checking whether the positional-bias finding survives controls for document and reference length.

## Methodology

- Inputs: `position_bias_analysis.csv` and `per_document_scores.csv` (final run). Records were joined by document SHA-256; all 864 eligible documents matched.
- Lengths: document words and reference words, taken from the Lead rows of the per-document scores.
- Quartiles: bias values are ranked with average ranks, so tied values receive the same rank and fall in the same quartile.
- Partial Spearman: bias and Lead-minus-LexRank ROUGE-1 advantage are rank-transformed, each is residualised on the rank of document and reference length (with intercept), and the residuals are correlated.
- Group difference: positive-bias minus non-positive-bias mean advantage, with a percentile bootstrap (2,000 resamples, seed 42). A length-adjusted version uses advantage residuals after regressing on raw document and reference length.

Bias is computed from description tokens that occur in transcript segments. Shared reference-token overlap may therefore contribute to the association. The `description` field is a proxy reference, not a verified gold summary, and the results describe associations, not causation.

## Findings

| Quartile | Bias range | n | Mean Lead-minus-LexRank |
|---|---|---|---|
| Q1 | -0.875 to 0.333 | 224 | +0.009 |
| Q2 | 0.348 to 0.611 | 208 | +0.104 |
| Q3 | 0.615 to 0.762 | 216 | +0.130 |
| Q4 | 0.765 to 1.000 | 216 | +0.131 |

- Partial Spearman, controlling for document and reference length: rho = +0.260 (p = 7.7e-15). Uncontrolled rho = +0.276.
- Positive minus non-positive group difference: +0.144 (95% CI [+0.117, +0.172]).
- Length-adjusted group difference: +0.128 (95% CI [+0.101, +0.157]).
- The direction is the same before and after length control, and both CIs exclude zero.

## Interpretation

The association between positional bias and Lead's advantage is largely unchanged after controlling for length. The advantage is near zero in the lowest-bias quartile and roughly flat at about +0.10 to +0.13 in the upper three quartiles. Length does not explain the pattern.

This does not show that the reference was written from the opening text, nor that Lead is a better summariser. Because the bias itself is lexical overlap with the reference, part of the association may reflect shared tokens rather than positional content. Treat the result as evidence that the reference set favours leading content, not as a measure of summary quality.

## Limitations

- Only Lead and LexRank are compared; the lowest-bias quartile may behave differently for other methods.
- Length controls are linear rank residualisation; other confounders (e.g. specialty) are not controlled.
- The bootstrap resamples documents, not the summarisation process; ROUGE-1 only.
- One dataset (cleaned MTSamples), one summary length (3 sentences), one sample of 864 eligible documents.

Output: `results/citi2026_final/tables/bias_robustness.csv`.
