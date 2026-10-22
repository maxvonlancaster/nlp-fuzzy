"""Distribution of paired ROUGE-1 differences, LSA minus Random (final run).

Uses only results/citi2026_final/tables/per_document_scores.csv.
Writes the report to docs/citi2026/lsa_random_distribution.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SCORES_CSV = ROOT / "results" / "citi2026_final" / "tables" / "per_document_scores.csv"
REPORT = ROOT / "docs" / "citi2026" / "lsa_random_distribution.md"
N_DOCS = 1000
TRIM = 0.10


def main():
    errors = []
    df = pd.read_csv(SCORES_CSV, dtype={"document_sha256": str})
    sub = df[df["method"].isin(["lsa", "random"])]
    wide = sub.pivot(index="document_sha256", columns="method", values="rouge1")
    if len(wide) != N_DOCS or wide.isna().any().any():
        errors.append(f"aligned documents={len(wide)}, missing={int(wide.isna().sum().sum())}")

    d = (wide["lsa"] - wide["random"]).to_numpy()
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_zero = int(np.sum(d == 0))
    mean = d.mean()
    median = np.median(d)
    sd = d.std(ddof=1)
    p10, p90 = np.percentile(d, [10, 90])
    skew = stats.skew(d, bias=False)
    trimmed = stats.trim_mean(d, TRIM)
    t_stat, t_p = stats.ttest_1samp(d, 0.0)
    w_stat, w_p = stats.wilcoxon(d, zero_method="wilcox")
    pos_frac = n_pos / (n_pos + n_neg)

    lines = [
        "# LSA minus Random: Paired ROUGE-1 Differences (Final Run)",
        "",
        "Written for: CITI-2026 reviewers checking the LSA-versus-Random result.",
        "",
        "Source: `results/citi2026_final/tables/per_document_scores.csv`, aligned by SHA-256 "
        f"({len(d)} documents). Differences are LSA minus Random ROUGE-1; positive favours LSA.",
        "",
        "## Counts",
        "",
        f"- Positive: {n_pos} ({100 * n_pos / len(d):.1f}%)",
        f"- Negative: {n_neg} ({100 * n_neg / len(d):.1f}%)",
        f"- Zero: {n_zero} ({100 * n_zero / len(d):.1f}%)",
        f"- Among non-zero differences, positive share: {100 * pos_frac:.1f}%",
        "",
        "## Distribution",
        "",
        f"- Mean: {mean:+.4f}",
        f"- Median: {median:+.4f}",
        f"- Standard deviation: {sd:.4f}",
        f"- 10th percentile: {p10:+.4f}; 90th percentile: {p90:+.4f}",
        f"- Skewness (sample, bias-corrected): {skew:+.3f}",
        f"- 10% trimmed mean: {trimmed:+.4f}",
        "",
        "## Test results (for reference)",
        "",
        f"- Paired t-test: t = {t_stat:.3f}, p = {t_p:.3e}",
        f"- Wilcoxon signed-rank (zero differences dropped): W = {w_stat:.1f}, p = {w_p:.3e}",
        "",
        "## Interpretation",
        "",
        "The paired t-test compares the mean difference against zero, so it is driven by the "
        "average shift and by the spread of the differences. The Wilcoxon signed-rank test "
        "compares the ranks of the absolute differences by sign, so it measures whether "
        "positive differences tend to be larger than negative ones, and it is less sensitive "
        "to the size of individual differences. These are different statistical properties, "
        "so the two tests can disagree when the mean and the typical (median or rank-based) "
        "difference point in different directions or when a few large differences dominate "
        "the mean.",
        "",
        "The measurements above show how the mean, median, trimmed mean and skewness relate "
        "for this distribution; they do not identify a cause for the discrepancy, and the "
        "report does not attribute one.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"{'PASS' if not errors else 'FAIL'}: {len(d)} aligned documents")
    print(f"pos/neg/zero = {n_pos}/{n_neg}/{n_zero}")
    print(f"mean={mean:+.4f} median={median:+.4f} sd={sd:.4f} p10={p10:+.4f} p90={p90:+.4f}")
    print(f"skew={skew:+.3f} trimmed(10%)={trimmed:+.4f}")
    print(f"t p={t_p:.3e}; Wilcoxon p={w_p:.3e}")
    print(f"report: {REPORT}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
