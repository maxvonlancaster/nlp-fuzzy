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
| Description text changed, mean / minimum across permutations | 99.94% / 99.65% |
| Description index changed, mean across permutations | 100% |

Descriptions are repeated across records, so a few reassignments leave the text unchanged; the text-change rate is therefore reported separately from index changes.

**Full-sample versus subset means.** Full-sample ROUGE-1 (n = 1,000) is 0.309 for Lead-3 and 0.224 for LexRank. On the 864-document analysis subset they are 0.315 and 0.222, giving an observed mean difference of +0.092.

**Shuffled mean difference.** Under reassignment, the Lead-minus-LexRank mean difference has median +0.001 (2.5th–97.5th percentiles −0.001 to +0.004). The observed difference lies far outside this distribution. Lead's mean advantage therefore depends on the true reference pairing, whereas the rank correlation with positional bias does not.

## Interpretation

The observed correlation (+0.275) lies inside the shuffled null distribution (exceedance 0.33). Randomly reassigned descriptions reproduce a positive association of similar size, so the correlation does not depend on the true description–transcript pairing.

The control does not isolate shared-reference coupling. Within each permutation, the positional measure and ROUGE use the same reassigned description, so the coupling is present by construction. The result is consistent with the association being produced by transcript-level lexical features acting through both measures. This is a hypothesis; the control does not test it.

The mean-difference result is a separate finding. It is not a causal estimate of Lead's advantage, but it shows that the level of Lead's advantage requires matched references, while the rank association does not.

The result therefore weakens the claim that the association reflects positional content in the reference, and it does not establish causality in either direction. The shuffled references are not valid clinical summaries, and shuffling also removes the meaningful document–reference alignment.

## Limitations

- The null preserves transcript features, so it cannot isolate their contribution.
- Stratification approximates the marginal distribution of references but does not preserve their token content.
- The 100 permutations are one seed; the exceedance rate has limited resolution.
- Results are for one corpus, one summary length and two classical methods.
