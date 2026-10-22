# Positional Reference Bias: Final Run

Written for: CITI-2026 reviewers assessing whether the MTSamples reference set favours positional methods.

## Methodology

- Documents: the 1,000 cleaned MTSamples documents from `load_mtsamples(limit=1000, seed=42, clean_for_evaluation=True)`. All 1,000 matched the saved per-document results by SHA-256 (no mismatches).
- Restriction: documents with at least 200 whitespace-separated words, so the first and last 100-word segments do not overlap. 864 documents qualified; none were dropped for empty description tokens.
- Tokenisation: lowercase, `[a-z0-9]+`, as in the earlier dataset audit.
- Recall: |unique description tokens found in segment| / |unique description tokens|, computed for the first 100 and last 100 words.
- Positional bias: first recall minus last recall. Positive means the description overlaps the opening more than the ending.
- Outcome: per-document ROUGE-1 advantage of Lead over LexRank (both from the saved final run; no re-scoring).

The `description` field is a proxy reference, not a verified human-written gold summary. The findings below describe associations and do not establish causation.

## Findings

- First-segment recall: mean 0.861, median 1.000.
- Last-segment recall: mean 0.357, median 0.312.
- Positive bias in 87.5% of documents (mean bias +0.505, median +0.613).
- Spearman correlation between positional bias and Lead-minus-LexRank ROUGE-1 advantage: rho = +0.275 (p = 1.7e-16).
- Mean Lead-minus-LexRank advantage: +0.110 for positive-bias documents (n = 756) versus -0.034 for non-positive-bias documents (n = 108).

## Interpretation

Documents whose descriptions draw on the opening of the transcript show a larger Lead advantage over LexRank. A rho of 0.28 is a weak-to-moderate monotonic association. It is consistent with the reference set rewarding leading content, which would partly explain Lead's strong ROUGE scores. It does not show that the reference was written from the opening, nor that Lead is a better summariser.

## Limitations

- Lexical overlap with a proxy reference is not a measure of summary quality.
- Only Lead and LexRank are compared; the same analysis for other methods is not included.
- The positive-bias and non-positive groups differ in size (756 vs 108), so the group comparison is less precise for the smaller group.
- Only ROUGE-1 is used for the outcome.
- Results cover one dataset (cleaned MTSamples), one summary length (3 sentences), and one sample (1,000 documents, seed 42).

Per-document values are in `results/citi2026_final/tables/position_bias_analysis.csv` (document hash plus numeric values only).
