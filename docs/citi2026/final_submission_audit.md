# Final Pre-Submission Audit: CITI-2026 Manuscript

Reviewed: `docs/citi2026/manuscript_draft.md` (4,169 words total; 3,886 before References), `reviewer_audit.md`, the final statistics, positional and negative-control reports, and the implementation files for the claims in §3.2–3.6.

## 1. Numerical consistency

All reported values were checked against the saved reports: Table 1 (rounded means), the ten paired-test results and Holm-adjusted p-values, the LSA-minus-Random distribution (469/467/64; mean +0.0140; median 0; trimmed mean +0.0070; skewness +0.462), the positional correlations (rho = +0.28 raw, 0.26 partial; quartile means), the window correlations (+0.245, −0.091, −0.166; differences +0.335 and +0.412), the shuffled-reference results (observed +0.275; median +0.267; exceedance 0.33; subset mean difference +0.092 versus +0.001), and the 99.94% text-change figure. No discrepancies found.

## 2. Consistency of claims and framing

- **Shuffled-reference control (Critical, unresolved).** The control shows the correlation is reproduced under reassignment, so it does not depend on the true description–transcript pairing. The manuscript states this in the Abstract, §4.4 and Conclusions. However, the positional measure and ROUGE share the same reassigned description in every permutation, so the design cannot isolate coupling. The title ("Positional Reference Bias") still presents positional bias as the subject. Smallest correction: keep the title as required, but state in the Abstract that the positional correlation is a null-compatible diagnostic, not evidence of bias.
- **Opening-specificity overstated (Major).** §4.4 ("the positive association is specific to the opening") and Conclusions ("specific to the opening window") overstate the evidence. The three windows partition the same description tokens, so the middle and last correlations are not independent of the first (position_control.md). Correction: "larger for the opening window than for the middle and last windows; because the windows partition the same description tokens, these correlations are not independent."
- **LexRank–TextRank caveat missing in Conclusions (Major).** §3.2 states the comparison is configuration-dependent; the Conclusions state "LexRank outperformed TextRank by a small margin" without it. Correction: append "(a configuration-dependent comparison; see §3.2)".
- **Title and Abstract alignment.** The Abstract's correlation and shuffled results are consistent with §4.4; no other mismatch found.

## 3. Bibliography and citations

12 entries; all cited in text. No placeholders remain. Holm (1979) is cited in §3.4. Remaining: §2.2 says "several studies have questioned" ROUGE stability; the cited works support this only in general terms (Minor, rephrase to "a few recent studies examine").

## 4. Dataset and data availability

- Raw-corpus counts (4,999 records; 2,139 repeated descriptions) match the dataset audit; cleaned-sample counts (2,248 unique transcriptions) match the loader check.
- Data availability (§3.1) follows the requested wording. Unverified: that the per-document score files are tracked in the public repository (not checked here; the `results/` folder should be confirmed before submission) (Minor).
- §1 says the cleaned sample "is not released as a public dataset", while §3.1 says the repository includes the source CSV. Both are accurate for different objects but read as contradictory (Minor; add "the cleaned sample" to the Introduction sentence).

## 5. Methods accuracy

§3.2–3.5 match the implementations (LexRank threshold > 0.1, unweighted; TextRank weighted with damping 0.85; LSA component limit; Random BLAKE2b seeding; Wilcoxon `zero_method='wilcox'`). Remaining gaps:
- §3.6 does not state that sparse specialty strata were pooled by length tercile (Minor; add one clause).
- The Wilcoxon `method` default is unspecified (Minor; already disclosed).

## 6. Reviewer-audit status

- Issue 1 (shared-reference coupling): **partially resolved** (controls added and reported; coupling not isolated).
- Issue 2 (bibliography): **partially resolved** (2024–2025 sources and dataset citation added; most methodological citations remain 2004–2020).
- Issue 3 (LexRank–TextRank confound): **partially resolved** (caveat in §3.2; no ablation; Conclusions caveat missing).
- Issue 4 (templates): **partially resolved** (described; effect not measured).
- Issue 5 (effect-size wording): **resolved**.
- Issue 6 (length/format): **unresolved** (4,169 words; see §7).
- Issue 7 (Table 1 CIs): **unresolved** (Minor).
- Issue 8 (naming/wording): **resolved**.

## 7. Format and length

Title: 11 words (within 13). Section structure is consistent (1–5, plus references). The manuscript is about 3,900 words of body text, which is at the upper end for a CEUR-ART short paper (5–9 pages single column); a regular paper (10–20 pages) would fit. Duplication: the positional/news framing appears in §1, §2.3 and the Abstract; the dataset-redundancy point appears in §3.1 and §4.5.

## Summary

- Critical: 1 (shuffled-reference interpretation/title framing).
- Major: 3 (opening-specificity overstatement; Conclusions LexRank caveat; length/format).
- Minor: 6 (§2.2 "several studies"; §1 versus §3.1 release wording; repository tracking unverified; §3.6 pooling; Table 1 CIs; duplicated framing).

Top three corrections:
1. §4.4 and Conclusions: replace "specific to the opening" with the non-independence qualification.
2. Conclusions: add the configuration-dependence caveat to the LexRank–TextRank sentence.
3. Decide short versus regular paper; if short, trim duplicated framing in §1, §2.3 and §4.5 (about 500 words).
