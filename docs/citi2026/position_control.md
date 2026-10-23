# Alternative-Position Control

Written for: reviewers assessing whether the positional association is specific to the opening of transcriptions (reviewer_audit.md, Issue 1; negative_controls_plan.md, Control 1).

## Method

- Data: cleaned MTSamples sample (seed 42), matched to `per_document_scores.csv` by SHA-256. Documents with at least 300 whitespace words: 672 (no document lacked description tokens).
- Windows: first 100 words, centred middle 100 words (start at `(n - 100) // 2`), and last 100 words. For documents with at least 300 words the three windows do not overlap.
- Recall: unique description tokens (`[a-z0-9]+`, lowercased) present in each window, divided by the number of unique description tokens.
- Outcome: per-document ROUGE-1 advantage of Lead-3 over LexRank (from the saved final run; no re-scoring).
- Statistics: Spearman correlation of each window's recall with the advantage. Differences (first minus middle; first minus last) are estimated with a paired document-level bootstrap (2,000 resamples, seed 42). The same comparisons are repeated after rank residualisation on document and reference length.

## Results

| Window | Spearman rho (raw) | Partial rho (length-controlled) |
|---|---|---|
| First | +0.245 | +0.246 |
| Middle | -0.091 | -0.087 |
| Last | -0.166 | -0.160 |

| Comparison | Raw difference (95% CI) | Length-controlled difference (95% CI) |
|---|---|---|
| First minus middle | +0.335 [+0.237, +0.428] | +0.332 [+0.231, +0.425] |
| First minus last | +0.412 [+0.305, +0.523] | +0.407 [+0.300, +0.520] |

## Interpretation

The positive association is specific to the opening window. Recall in the middle and final windows shows weak negative correlations with the Lead advantage, and both first-window differences exclude zero under raw and length-controlled analyses.

This is not evidence that the association is free of shared-reference coupling. The outcome (ROUGE-1 against the same description) and the exposure (description recall) still depend on one reference. The windows also share a denominator: a description's tokens are partitioned across windows, so a high first-window recall mechanically limits the recall available elsewhere. The negative middle and last correlations may partly reflect this partitioning rather than a substantive property of the transcripts. The control does not establish causality or positional content in the reference.

## Limitations

- Window contents differ in structure (headers, templates, boilerplate); recall in each window is not a matched null.
- The middle window is one choice of interior segment; other interior positions were not examined.
- The shuffled-reference control (negative_controls_plan.md, Control 2) is still required to address shared-reference coupling.
- Results are for one corpus and one summary length.
