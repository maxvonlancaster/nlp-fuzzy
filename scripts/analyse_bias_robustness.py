"""Robustness of the positional-bias / Lead-advantage association (final run).

Uses only results/citi2026_final/tables/ CSVs (no re-scoring, no MTSamples reload).
Associations only. Shared reference-token overlap may contribute to the association,
because the bias is computed from description tokens found in transcript segments.

Outputs:
  results/citi2026_final/tables/bias_robustness.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "citi2026_final" / "tables"
BIAS_CSV = TABLES / "position_bias_analysis.csv"
SCORES_CSV = TABLES / "per_document_scores.csv"
OUT_CSV = TABLES / "bias_robustness.csv"
EXPECTED_ELIGIBLE = 864
N_BOOT = 2000
BOOT_SEED = 42


def residualise(y, covariates):
    """Residuals of y after OLS on an intercept and the covariates."""
    X = np.column_stack([np.ones(len(y))] + covariates)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def main():
    errors = []
    bias = pd.read_csv(BIAS_CSV, dtype={"document_sha256": str})
    scores = pd.read_csv(SCORES_CSV, dtype={"document_sha256": str})

    if len(bias) != EXPECTED_ELIGIBLE:
        errors.append(f"eligible documents={len(bias)}, expected {EXPECTED_ELIGIBLE}")
    if bias["document_sha256"].duplicated().any():
        errors.append("duplicate hashes in position_bias_analysis.csv")

    lengths = scores[scores["method"] == "lead"][["document_sha256", "document_words", "reference_words"]]
    df = bias.merge(lengths, on="document_sha256", how="left", validate="one_to_one")
    if df[["document_words", "reference_words"]].isna().any().any():
        errors.append("eligible documents missing from per_document_scores.csv")
    if len(df) != EXPECTED_ELIGIBLE:
        errors.append(f"joined records={len(df)}")

    b = df["position_bias"].to_numpy()
    adv = df["lead_minus_lexrank_rouge1"].to_numpy()
    dlen = df["document_words"].to_numpy(dtype=float)
    rlen = df["reference_words"].to_numpy(dtype=float)
    n = len(df)

    # Quartiles: tied bias values get the same average rank, so they share a quartile
    avg_rank = stats.rankdata(b, method="average")
    quartile = np.minimum(3, np.floor(4 * (avg_rank - 1) / n).astype(int)) + 1
    df["quartile"] = quartile

    out_rows = []
    for q in range(1, 5):
        sub = df[df["quartile"] == q]
        out_rows.append({
            "section": "quartile", "label": f"Q{q}", "n": len(sub),
            "bias_min": sub["position_bias"].min(), "bias_max": sub["position_bias"].max(),
            "mean_lead_minus_lexrank": sub["lead_minus_lexrank_rouge1"].mean(),
            "ci_low": np.nan, "ci_high": np.nan, "statistic": np.nan, "p_value": np.nan,
        })

    # Partial Spearman: rank-transform, residualise on rank(length) covariates, then correlate
    r_b = stats.rankdata(b)
    r_a = stats.rankdata(adv)
    r_d = stats.rankdata(dlen)
    r_r = stats.rankdata(rlen)
    res_b = residualise(r_b, [r_d, r_r])
    res_a = residualise(r_a, [r_d, r_r])
    partial_rho, partial_p = stats.pearsonr(res_b, res_a)
    raw_rho, raw_p = stats.spearmanr(b, adv)
    out_rows.append({"section": "partial_spearman", "label": "controls: doc_words, reference_words",
                     "n": n, "bias_min": np.nan, "bias_max": np.nan, "mean_lead_minus_lexrank": np.nan,
                     "ci_low": np.nan, "ci_high": np.nan, "statistic": partial_rho, "p_value": partial_p})
    out_rows.append({"section": "spearman_raw", "label": "no controls", "n": n,
                     "bias_min": np.nan, "bias_max": np.nan, "mean_lead_minus_lexrank": np.nan,
                     "ci_low": np.nan, "ci_high": np.nan, "statistic": raw_rho, "p_value": raw_p})

    # Group difference: positive-bias minus non-positive-bias mean advantage, bootstrap CI
    pos_mask = b > 0
    diffs, adj_diffs = [], []
    res_adv_len = residualise(adv, [dlen, rlen])  # length-adjusted outcome
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    for row in idx:
        pm = pos_mask[row]
        if pm.any() and (~pm).any():
            diffs.append(adv[row][pm].mean() - adv[row][~pm].mean())
            adj_diffs.append(res_adv_len[row][pm].mean() - res_adv_len[row][~pm].mean())
    diffs = np.array(diffs)
    adj_diffs = np.array(adj_diffs)
    point = adv[pos_mask].mean() - adv[~pos_mask].mean()
    point_adj = res_adv_len[pos_mask].mean() - res_adv_len[~pos_mask].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    lo_a, hi_a = np.percentile(adj_diffs, [2.5, 97.5])

    out_rows.append({"section": "group_difference", "label": "positive minus non-positive bias (raw)",
                     "n": f"{int(pos_mask.sum())} vs {int((~pos_mask).sum())}",
                     "bias_min": np.nan, "bias_max": np.nan, "mean_lead_minus_lexrank": point,
                     "ci_low": lo, "ci_high": hi, "statistic": np.nan, "p_value": np.nan})
    out_rows.append({"section": "group_difference", "label": "positive minus non-positive bias (length-adjusted)",
                     "n": f"{int(pos_mask.sum())} vs {int((~pos_mask).sum())}",
                     "bias_min": np.nan, "bias_max": np.nan, "mean_lead_minus_lexrank": point_adj,
                     "ci_low": lo_a, "ci_high": hi_a, "statistic": np.nan, "p_value": np.nan})

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.10g")

    consistent = (point > 0) == (point_adj > 0) and (lo > 0) == (lo_a > 0)
    print(f"{'PASS' if not errors else 'FAIL'}: joined {n} eligible documents")
    for _, r in out[out.section == "quartile"].iterrows():
        print(f"  {r.label}: n={int(r.n)} bias[{r.bias_min:.3f},{r.bias_max:.3f}] "
              f"mean Lead-LexRank={r.mean_lead_minus_lexrank:+.4f}")
    print(f"partial Spearman (length-controlled) rho={partial_rho:+.4f} p={partial_p:.3e}; "
          f"raw rho={raw_rho:+.4f}")
    print(f"group diff raw={point:+.4f} CI[{lo:+.4f},{hi:+.4f}]; "
          f"length-adjusted={point_adj:+.4f} CI[{lo_a:+.4f},{hi_a:+.4f}]")
    print(f"directionally consistent after length control: {consistent}")
    print(f"output: {OUT_CSV}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
