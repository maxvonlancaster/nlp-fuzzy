# Baseline Snapshot: Unsupervised Extractive Summarization

Sources: `results/evaluation_results.json` (saved outputs), `src/summarization_evaluation.ipynb` cells 3, 5, 8, 24 (source only).

## Configuration

| Item | Saved metadata | Notebook configuration |
|---|---|---|
| Dataset | MTSamples Medical Transcriptions | `load_mtsamples(limit=N_DOCUMENTS, seed=RANDOM_SEED)` |
| Sample size | 1000 documents | `N_DOCUMENTS = 1000` |
| Summary length | 3 sentences (`n_sentences`) | `N_SENTENCES = 3` |
| Random seed | 42 | `RANDOM_SEED = 42` |
| Evaluation date | 2026-04-12 08:57:00 | Set at export time (cell 24 `datetime.now()`) |
| Bootstrap samples | Not saved | `BOOTSTRAP_SAMPLES = 1000` |
| Confidence level | Not saved | `CONFIDENCE_LEVEL = 0.95` |

## Evaluated Methods

LexRank is **not** included. Cell 8 adds it only when `LEXRANK_AVAILABLE` is true (requires PyTorch); the saved JSON has no `lexrank` entry.

ROUGE F-scores are mean values over 1000 documents. Each CI is the saved 95% interval.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-3 | 0.3077 | 0.2385 | 0.2845 |
| Random | 0.1887 | 0.1083 | 0.1621 |
| TextRank (TF-IDF) | 0.2039 | 0.1221 | 0.1769 |
| LSA | 0.1770 | 0.0903 | 0.1510 |

## ROUGE-1 95% Confidence Intervals

| Method | Lower | Upper |
|---|---|---|
| Lead-3 | 0.2960 | 0.3201 |
| Random | 0.1799 | 0.1980 |
| TextRank (TF-IDF) | 0.1944 | 0.2142 |
| LSA | 0.1666 | 0.1873 |

## Discrepancies and Gaps

1. **LexRank absent.** The notebook defines it conditionally on `LEXRANK_AVAILABLE`, and the saved JSON contains no LexRank results. The saved output does not record whether PyTorch was available, so the cause is unverified.
2. **Missing parameters in saved metadata.** The JSON does not record bootstrap samples (1000), confidence level (0.95), LSA `n_components` (5), or TextRank settings (similarity metric, damping, max_iter, tol). These exist only in the notebook configuration.
3. **Date semantics.** The `date` field is the export timestamp from cell 24, not the start of evaluation.
4. **Summary length unit.** Length is defined only in sentences (3). No word or token budget is recorded.

No conflicts were found between saved sample size, seed, sentence count, or dataset name and the notebook configuration.
