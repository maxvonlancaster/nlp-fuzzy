# Reviewer Audit: CITI-2026 Manuscript Draft

Reviewed: `docs/citi2026/manuscript_draft.md` (3,341 words), checked against the final statistics (`statistical_summary_final.md`, `lsa_random_distribution.md`), position-bias and robustness reports, and the implementations in `src/helpers/baselines.py`, `dataset_loader.py` and `evaluation.py`. No manuscript or code was changed.

## Issues

### 1. Shared-reference coupling in the positional-bias analysis (Critical)
- **Sections:** 3.5, 4.3, Abstract, Conclusions.
- **Problem:** Positional bias is first-segment minus last-segment recall of description tokens, and the outcome is ROUGE-1 against the same description. Both sides of the Spearman and quartile analyses therefore depend on one reference's unigrams, so a positive association is expected even without any positional content in the reference. The manuscript acknowledges this in one sentence (4.3, 3.5) but does not test it. Length residualisation does not address it.
- **Correction:** Add negative controls. Compare Lead's advantage using description recall from a random or middle 100-word window, and use a description-shuffling control (pair each transcription with another document's description). Report the association under these controls, or state that the claim is limited to lexical coupling. Keep the wording as association-only.

### 2. Outdated and incomplete bibliography (Critical)
- **Sections:** 2, References.
- **Problem:** All seven references date from 2004–2020. None is from the last five to eight years, which CITI-2026 asks for. Several named methods and tests have no citation: Holm correction, Wilcoxon signed-rank test, Cohen's dz, PageRank, Spearman, scikit-learn TruncatedSVD, and NLTK punkt. The MTSamples source is not cited at all.
- **Correction:** Add recent work on ROUGE validity and reference-based evaluation, and recent unsupervised extractive or lead-bias studies, verified against the publisher pages. Add standard references for the named statistical and graph methods, and cite MTSamples with its source and licence. Do not add entries that have not been checked.

### 3. LexRank versus TextRank confounds algorithm and configuration (Major)
- **Sections:** 2.1, 3.2, 4.2, 5.
- **Problem:** LexRank uses a binary, thresholded graph (similarity > 0.1, unweighted PageRank), whereas TextRank uses a weighted complete graph. The Conclusions attribute the LexRank-over-TextRank gap (dz = 0.12) to the method, but it could equally reflect graph weighting or the threshold. The manuscript also calls the implementation "LexRank" without stating that it is a simplified variant, and it does not say whether thresholds or n-gram ranges were tuned.
- **Correction:** Rename the method to a LexRank-style variant (or state the simplification explicitly), state that no hyperparameters were tuned if that is the case, and either run a weighted/binary ablation or qualify the conclusion as configuration-dependent.

### 4. Dataset provenance and template-heavy references not addressed (Major)
- **Sections:** 3.1, 4.4.
- **Problem:** The audit shows 2,139 descriptions repeated across 4,790 records, including template text such as "An example/template for a routine normal male physical exam" (up to 12 records). This templated structure directly affects ROUGE and the positional result, but Methods does not mention it. Source, version, licence and collection date are also absent.
- **Correction:** Describe the repetition and its effect in Methods, report the share of template descriptions, and add a sensitivity result excluding repeated descriptions (or state it as future work). Add MTSamples provenance and a data availability statement.

### 5. Inconsistent effect-size language and under-specified tests (Major)
- **Sections:** 4.2 (paragraph after the highlighted list), Abstract, 3.4.
- **Problem:** The Lead-versus-LexRank gap is "medium" in 4.2 (bullet list), "moderate" in the Abstract, and "substantial" in the paragraph after the list. Section 3.4 does not state the Wilcoxon method or tie handling, although 26–143 zero differences per comparison and ROUGE ties are present. Section 2.2 says "several studies have questioned" ROUGE stability, but only two works are cited.
- **Correction:** Use one effect-size label (for example, "medium" with the dz value), specify the Wilcoxon `method` and tie treatment in 3.4, and either cite the studies in 2.2 or remove the claim.

### 6. Length and format (Minor)
- **Sections:** whole manuscript.
- **Problem:** 3,341 words is likely beyond a short paper (5–9 pages) in CEUR-ART single-column layout; the paper does not state whether it is a short or regular submission. Results and Conclusions repeat numbers already in the Abstract.
- **Correction:** Decide the category, then trim repeated numbers from Conclusions and 4.4, and keep the Table 1 numbers only once.

### 7. Table and reporting completeness (Minor)
- **Sections:** 4.1.
- **Problem:** Table 1 reports means only, while 3.3 describes per-method bootstrap CIs and the statistical CIs are given only for highlighted pairs.
- **Correction:** Add the 95% CI columns to Table 1, or state that per-method CIs are in the supplementary tables.

### 8. Naming and wording (Minor)
- **Sections:** 3.4, 3.5, 4.3.
- **Problem:** The "partial Spearman" value is a Pearson correlation of rank residuals; the term is acceptable but should be defined as such. Section 3.3 places a note about global RNG state inside the CI description.
- **Correction:** Define the statistic explicitly and move the RNG note to a footnote or the reproducibility section.

## Priorities
1. Shared-reference coupling and negative controls (Issue 1).
2. Recent and complete bibliography, including MTSamples and named methods (Issue 2).
3. LexRank versus TextRank confound and tuning statement (Issue 3).

Methodological problems: Issues 1, 3, 4 and 5. Editorial: Issues 6–8 and the reference-list formatting.
