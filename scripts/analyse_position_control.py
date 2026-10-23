"""Alternative-position control: first vs centred-middle vs last 100-word windows.

Reuses the final positional methodology (tokenisation [a-z0-9]+, description-token
recall, Lead-minus-LexRank ROUGE-1 advantage, rank residualisation on length).
Prints aggregate numbers only; no clinical text is written or printed.
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helpers.dataset_loader import load_mtsamples  # noqa: E402

SCORES_CSV = ROOT / "results" / "citi2026_final" / "tables" / "per_document_scores.csv"
N_DOCS = 1000
SEED = 42
N_BOOT = 2000
WIN = 100
MIN_WORDS = 300
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text):
    return set(TOKEN_RE.findall(text.lower()))


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def residualise(y, covariates):
    X = np.column_stack([np.ones(len(y))] + covariates)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_rho(x, y, z1, z2):
    """Spearman-style partial correlation: rank-transform, residualise on length ranks, Pearson."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    r1 = stats.rankdata(z1)
    r2 = stats.rankdata(z2)
    return stats.pearsonr(residualise(rx, [r1, r2]), residualise(ry, [r1, r2]))[0]


def main():
    errors = []
    _, df = load_mtsamples(limit=N_DOCS, seed=SEED, base_path=str(ROOT / "resources"),
                           clean_for_evaluation=True)
    loaded = df.reset_index(drop=True)
    hashes = [sha256(t) for t in loaded["transcription"]]
    loaded["sha"] = hashes

    scores = pd.read_csv(SCORES_CSV, dtype={"document_sha256": str})
    wide = scores.pivot(index="document_sha256", columns="method", values="rouge1")
    if len(set(hashes)) != N_DOCS or set(hashes) != set(scores["document_sha256"]):
        errors.append("loaded documents do not match per_document_scores.csv by SHA-256")

    rows = []
    no_tokens = 0
    for text, desc, h in zip(loaded["transcription"], loaded["description"], hashes):
        words = text.split()
        n = len(words)
        if n < MIN_WORDS:
            continue
        d = tokens(str(desc))
        if not d:
            no_tokens += 1
            continue
        mid_start = (n - WIN) // 2
        windows = {
            "first": words[:WIN],
            "middle": words[mid_start:mid_start + WIN],
            "last": words[-WIN:],
        }
        if not (windows["first"] and mid_start >= WIN and mid_start + WIN <= n - WIN):
            errors.append("windows overlap")
            break
        rec = {k: len(d & tokens(" ".join(w))) / len(d) for k, w in windows.items()}
        rows.append({
            "sha": h,
            **{f"recall_{k}": v for k, v in rec.items()},
            "adv": float(wide.loc[h, "lead"] - wide.loc[h, "lexrank"]),
            "doc_words": n,
            "ref_words": len(str(desc).split()),
        })

    out = pd.DataFrame(rows)
    n = len(out)
    adv = out["adv"].to_numpy()
    dw = out["doc_words"].to_numpy(dtype=float)
    rw = out["ref_words"].to_numpy(dtype=float)
    rec = {k: out[f"recall_{k}"].to_numpy() for k in ["first", "middle", "last"]}

    rho_raw = {k: stats.spearmanr(rec[k], adv)[0] for k in rec}
    rho_part = {k: partial_rho(rec[k], adv, dw, rw) for k in rec}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = {"raw": {"fm": [], "fl": []}, "partial": {"fm": [], "fl": []}}
    for b in idx:
        bo = {k: rec[k][b] for k in rec}
        ba, bd, bw, br = adv[b], dw[b], rw[b], rw
        r_raw = {k: stats.spearmanr(bo[k], ba)[0] for k in bo}
        r_part = {k: partial_rho(bo[k], ba, bd, bw) for k in bo}
        boot["raw"]["fm"].append(r_raw["first"] - r_raw["middle"])
        boot["raw"]["fl"].append(r_raw["first"] - r_raw["last"])
        boot["partial"]["fm"].append(r_part["first"] - r_part["middle"])
        boot["partial"]["fl"].append(r_part["first"] - r_part["last"])

    def ci(a):
        return np.percentile(a, [2.5, 97.5])

    status = "PASS" if not errors else "FAIL"
    print(f"{status}: {n} documents with >= {MIN_WORDS} words (skipped {no_tokens} with no description tokens)")
    for label, rho in [("raw", rho_raw), ("partial", rho_part)]:
        print(f"{label} rho: first={rho['first']:+.4f} middle={rho['middle']:+.4f} last={rho['last']:+.4f}")
    for label in ["raw", "partial"]:
        fm = np.array(boot[label]["fm"]); fl = np.array(boot[label]["fl"])
        lo1, hi1 = ci(fm); lo2, hi2 = ci(fl)
        p_fm = rho_raw["first"] if label == "raw" else rho_part["first"]
        print(f"{label}: first-middle={np.mean(fm):+.4f} CI[{lo1:+.4f},{hi1:+.4f}]; "
              f"first-last={np.mean(fl):+.4f} CI[{lo2:+.4f},{hi2:+.4f}]")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
