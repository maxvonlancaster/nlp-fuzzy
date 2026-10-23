# Shuffled-Reference Negative Control

Written for: reviewers assessing whether the positional-bias association depends on the true description–transcript pairing (reviewer_audit.md, Issue 1; negative_controls_plan.md, Control 2).

## Method

- Data: cleaned MTSamples sample (seed 42). All 1,000 documents matched to saved hashes. Lead-3 and LexRank summaries were reconstructed once; their original-reference ROUGE-1 matched the saved per-document scores exactly (tolerance 1e-9).
- Analysis subset: 864 documents with at least 200 words and non-empty description tokens.
- Permutation: descriptions were reassigned among documents 100 times with a local NumPy RNG (seed 42). Reassignment was stratified by medical specialty and a reference-length tercile; strata with fewer than two documents were pooled by length tercile. Redraws (up to 20) avoided fixed points; no stratum needed the fallback, and 100% of description assignments changed.
- Per permutation: positional bias (first-minus-last description-token recall, computed against the shuffled description), Lead-minus-LexRank ROUGE-1 against the shuffled description, and Spearman correlation between them. Summaries were not regenerated.

## Results

| Quantity | Value |
|---|---|
| Observed Spearman rho | +0.275 |
| Shuffled rho, median | +0.267 |
| Shuffled rho, 2.5th–97.5th percentiles | +0.226 to +0.317 |
| One-sided exceedance (shuffled ≥ observed) | 0.33 |
| Shuffled mean ROUGE-1, Lead-3 / LexRank | 0.082 / 0.080 |
| Zero-score prevalence, Lead-3 / LexRank | 18.5% / 17.8% |
| Invalid permutations (undefined rho) | 0 |

Observed ROUGE-1 with the true references is 0.309 for Lead-3 and 0.224 for LexRank, so the true pairing contributes most of the ROUGE signal.

## Interpretation

The observed correlation lies inside the shuffled null distribution. Randomly reassigned descriptions reproduce a positive association of similar size. This means the association does not depend on the true description–transcript pairing.

The control cannot separate shared-reference coupling from other effects. Within each permutation, the positional measure and ROUGE still use the same reassigned description, so the coupling mechanism is intact by construction. What the result shows is that the coupling alone, operating on transcript-level lexical features, is sufficient to produce the association. A plausible source is generic, structurally repeated vocabulary in transcript openings, which both measures can pick up. This is a hypothesis; the control does not test it.

The result therefore weakens the claim that the association reflects positional content in the reference, and it does not establish causality in either direction. The shuffled references are not valid clinical summaries, and shuffling also removes the meaningful document–reference alignment.

## Limitations

- The null preserves transcript features, so it cannot isolate their contribution.
- Stratification approximates the marginal distribution of references but does not preserve their token content.
- The 100 permutations are one seed; the exceedance rate has limited resolution.
- Results are for one corpus, one summary length and two classical methods.
