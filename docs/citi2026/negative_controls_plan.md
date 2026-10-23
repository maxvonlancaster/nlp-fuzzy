# Negative Controls Plan: Reference Coupling

Addresses reviewer_audit.md, Issue 1 (shared-reference coupling in the positional-bias analysis).

## Available data

- Saved: `per_document_scores.csv` (per-document ROUGE-1/2/L for all five methods, SHA-256, document and reference word counts); `position_bias_analysis.csv` (first/last recall and bias for 864 documents, Lead-minus-LexRank ROUGE-1 advantage).
- Not saved: generated summaries (held only in notebook memory), middle-window recall, shuffled-reference scores.
- Reconstructible: the 1,000 documents from `load_mtsamples(..., clean_for_evaluation=True)`, since the sample is seeded. Lead-3 and LexRank are deterministic functions of the transcription.

## Control 1: Alternative-position control (no new summarisation)

- Restrict to documents with at least 300 words so that opening, middle and ending 100-word windows do not overlap. Compute description-token recall in each window using the existing tokenisation.
- Test whether Lead-minus-LexRank ROUGE-1 correlates more strongly with opening recall than with middle or ending recall, using dependent-correlation bootstrap comparisons with length residualisation as in Section 3.5.
- Interpretation: a specific opening-window association supports positional content; a similar association across all windows suggests a generic lexical-overlap effect.
- Limitations: window content varies (headers, templates); the middle window is not a null distribution; the result is still correlational.

## Control 2: Shuffled-reference control (requires summary reconstruction)

- Randomly permute descriptions across documents, stratified by specialty and reference length to preserve their marginal distribution. Use a fixed seed.
- Recompute description-token recall for the first and last windows, and ROUGE-1 for the fixed Lead-3 and LexRank summaries against the shuffled references.
- Interpretation: if the positional association persists under shuffling, it reflects document structure rather than the original reference pairing; if it disappears, the association depends on matched reference content.
- Required: Lead-3 and LexRank summaries must be regenerated from the reconstructed sample. Both are deterministic, so regeneration should reproduce `per_document_scores.csv` exactly before any shuffled scores are used. Summaries for other methods are not needed.
- Limitations: stratified shuffling may still produce implausible reference-document pairs; shuffling removes the true reference, so the outcome measures a different target.

## Distinguishing the claims

- Statistical association: the correlation between positional recall and ROUGE advantage, as reported.
- Shared-reference coupling: the part of that association produced by both measures depending on the same description; Control 2 targets it.
- Genuine positional specificity: an opening-specific association that persists under Control 1 and Control 2. Neither control establishes causality.
