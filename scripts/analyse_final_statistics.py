"""Final paired statistics for ROUGE-1 (CITI-2026 final run).

Inputs (read only, results/citi2026_final/):
  tables/per_document_scores.csv
  evaluation_results.json
Output:
  tables/paired_statistics.csv

Tests: paired t-test and Wilcoxon signed-rank, each with Holm correction over
the 10 pairwise comparisons (alpha 0.05). Also paired bootstrap 95% CIs
(2000 resamples, seed 42) and Cohen's dz on within-document differences.
"""
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "citi2026_final"
SCORES_CSV = FINAL / "tables" / "per_document_scores.csv"
RESULTS_JSON = FINAL / "evaluation_results.json"
OUT_CSV = FINAL / "tables" / "paired_statistics.csv"

ALPHA = 0.05
METRIC = "rouge1"
N_DOCS = 1000
N_METHODS = 5
N_BOOT = 2000
BOOT_SEED = 42
TOL = 1e-9
METHOD_ORDER = ["lead", "random", "textrank", "lsa", "lexrank"]


def holm(p_values):
    """Holm step-down adjusted p-values (NaN treated as 1.0), input order."""
    p = [1.0 if (v is None or math.isnan(v)) else v for v in p_values]
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def main():
    errors = []
    df = pd.read_csv(SCORES_CSV, dtype={"document_sha256": str})
    with open(RESULTS_JSON, encoding="utf-8") as f:
        results = json.load(f)
    json_means = {k: v["mean_scores"][METRIC] for k, v in results["methods"].items()}

    if len(df) != N_DOCS * N_METHODS:
        errors.append(f"rows={len(df)}, expected {N_DOCS * N_METHODS}")
    if df["document_sha256"].nunique() != N_DOCS:
        errors.append(f"unique documents={df['document_sha256'].nunique()}")
    if not (df.groupby("document_sha256")["method"].nunique() == N_METHODS).all():
        errors.append("not every document has five method records")
    if set(df["method"]) != set(METHOD_ORDER):
        errors.append(f"methods in CSV: {sorted(set(df['method']))}")

    # Align by document hash: one row per (document, method)
    wide = df.pivot(index="document_sha256", columns="method", values=METRIC).sort_index()
    if wide.isna().any().any():
        errors.append("missing ROUGE-1 values after alignment")

    for m in METHOD_ORDER:
        mean_csv = wide[m].mean()
        if abs(mean_csv - json_means[m]) > TOL:
            errors.append(f"{m} mean {mean_csv:.12f} != JSON {json_means[m]:.12f}")

    pairs = list(combinations(METHOD_ORDER, 2))
    rows = []
    t_p, w_p = [], []
    for m1, m2 in pairs:
        x1 = wide[m1].to_numpy()
        x2 = wide[m2].to_numpy()
        d = x2 - x1  # M2 - M1; negative means M1 scores higher
        t_stat, t_pval = stats.ttest_rel(x2, x1)
        n_zero = int(np.sum(d == 0))
        if n_zero == len(d):
            w_stat, w_pval = float("nan"), 1.0  # degenerate: no non-zero differences
        else:
            w_res = stats.wilcoxon(x2, x1, zero_method="wilcox")
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)

        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        boot_means = d[idx].mean(axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        sd = d.std(ddof=1)
        dz = float(d.mean() / sd) if sd > 0 else float("nan")

        t_p.append(float(t_pval))
        w_p.append(w_pval)
        rows.append({
            "Method 1": m1,
            "Method 2": m2,
            "Mean difference (M2 - M1)": float(d.mean()),
            "95% CI low": float(ci_low),
            "95% CI high": float(ci_high),
            "Higher ROUGE-1": m2 if d.mean() > 0 else m1,
            "t statistic": float(t_stat),
            "p (t, raw)": float(t_pval),
            "Wilcoxon statistic": w_stat,
            "p (Wilcoxon, raw)": w_pval,
            "Zero differences": n_zero,
            "Cohen dz": dz,
        })

    t_holm = holm(t_p)
    w_holm = holm(w_p)
    for i, r in enumerate(rows):
        r["p (t, Holm)"] = t_holm[i]
        r["p (Wilcoxon, Holm)"] = w_holm[i]
        r["Significant t (Holm)"] = "Yes" if t_holm[i] < ALPHA else "No"
        r["Significant Wilcoxon (Holm)"] = "Yes" if w_holm[i] < ALPHA else "No"

    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and k.startswith("p ") and v == 0.0:
                errors.append(f"p-value underflowed to 0 for {r['Method 1']} vs {r['Method 2']}")

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})

    status = "PASS" if not errors else "FAIL"
    print(f"{status}: {len(rows)} pairwise comparisons, ROUGE-1")
    for r in rows:
        if {r["Method 1"], r["Method 2"]} in ({"lead", "lexrank"}, {"textrank", "lexrank"}, {"lsa", "random"}):
            print(f"  {r['Method 1']} vs {r['Method 2']}: diff={r['Mean difference (M2 - M1)']:+.4f} "
                  f"CI=[{r['95% CI low']:+.4f},{r['95% CI high']:+.4f}] "
                  f"p_t_holm={r['p (t, Holm)']:.3e} p_w_holm={r['p (Wilcoxon, Holm)']:.3e} "
                  f"dz={r['Cohen dz']:.3f}")
    print(f"output: {OUT_CSV}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
