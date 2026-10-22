"""Holm correction for the 10 pairwise paired t-test comparisons (CITI-2026 run).

Inputs (read only):
  results/citi2026_clean/tables/pairwise_comparisons.csv
  results/citi2026_clean/evaluation_results.json
Output:
  results/citi2026_clean/tables/pairwise_holm.csv
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "citi2026_clean" / "tables"
PAIRS_CSV = TABLES / "pairwise_comparisons.csv"
RESULTS_JSON = ROOT / "results" / "citi2026_clean" / "evaluation_results.json"
OUT_CSV = TABLES / "pairwise_holm.csv"
ALPHA = 0.05
EXPECTED_METHODS = {"lead", "random", "textrank", "lsa", "lexrank"}


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        value = min(1.0, (m - rank) * p_values[i])
        running_max = max(running_max, value)
        adjusted[i] = running_max
    return adjusted


def main():
    with open(RESULTS_JSON, encoding="utf-8") as f:
        results = json.load(f)
    means = {name: m["mean_scores"]["rouge1"] for name, m in results["methods"].items()}

    with open(PAIRS_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    errors = []
    if len(rows) != 10:
        errors.append(f"expected 10 comparisons, found {len(rows)}")

    methods_in_pairs = {r["Method 1"] for r in rows} | {r["Method 2"] for r in rows}
    if methods_in_pairs != EXPECTED_METHODS or set(means) != EXPECTED_METHODS:
        errors.append(f"method sets differ: pairs={sorted(methods_in_pairs)}")

    p_raw = []
    for r in rows:
        p = float(r["p-value"])
        diff = float(r["Difference"])
        if not (0.0 <= p <= 1.0):
            errors.append(f"invalid p-value {p} for {r['Method 1']} vs {r['Method 2']}")
        # Difference is defined as Method 2 mean minus Method 1 mean
        expected = means[r["Method 2"]] - means[r["Method 1"]]
        if abs(diff - expected) > 1e-9:
            errors.append(f"difference mismatch for {r['Method 1']} vs {r['Method 2']}")
        p_raw.append(p)

    p_holm = holm_adjust(p_raw)
    order = sorted(range(len(rows)), key=lambda i: p_raw[i])
    rank_of = {i: rank + 1 for rank, i in enumerate(order)}

    out_rows = []
    for i, r in enumerate(rows):
        diff = float(r["Difference"])
        out_rows.append({
            "Method 1": r["Method 1"],
            "Method 2": r["Method 2"],
            "Difference (M2 - M1)": f"{diff:.6f}",
            "Higher ROUGE-1": r["Method 2"] if diff > 0 else r["Method 1"],
            "p-value (raw)": f"{p_raw[i]:.3e}",
            "Holm rank": rank_of[i],
            "p-value (Holm)": f"{p_holm[i]:.3e}",
            "Significant (Holm, alpha=0.05)": "Yes" if p_holm[i] < ALPHA else "No",
        })

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    status = "PASS" if not errors else "FAIL"
    n_sig = sum(1 for r in out_rows if r["Significant (Holm, alpha=0.05)"] == "Yes")
    print(f"{status}: {n_sig}/{len(rows)} significant after Holm")
    for r in out_rows:
        print(f"  {r['Method 1']} vs {r['Method 2']}: diff={r['Difference (M2 - M1)']} "
              f"p_holm={r['p-value (Holm)']}")
    print(f"output: {OUT_CSV}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
