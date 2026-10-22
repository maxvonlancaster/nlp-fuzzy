"""Positional reference bias analysis on the final CITI-2026 run.

Lexical diagnostic: how much of each description's unique token set appears in
the first vs last 100 transcription words, and whether that positional bias
is associated with Lead's ROUGE-1 advantage over LexRank. Associations only.
The description is a proxy reference, not a verified gold summary.

Inputs:
  results/citi2026_final/tables/per_document_scores.csv
  MTSamples via helpers.dataset_loader.load_mtsamples(limit=1000, seed=42, clean_for_evaluation=True)
Output:
  results/citi2026_final/tables/position_bias_analysis.csv (numeric values and hashes only)
"""
import csv
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
OUT_CSV = ROOT / "results" / "citi2026_final" / "tables" / "position_bias_analysis.csv"
TOKEN_RE = re.compile(r"[a-z0-9]+")
SEGMENT_WORDS = 100
MIN_WORDS = 2 * SEGMENT_WORDS
N_DOCS = 1000
SEED = 42


def tokens(text):
    return set(TOKEN_RE.findall(text.lower()))


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    errors = []
    _, df = load_mtsamples(limit=N_DOCS, seed=SEED, base_path=str(ROOT / "resources"),
                           clean_for_evaluation=True)
    loaded = df.reset_index(drop=True)
    loaded_hashes = [sha256(t) for t in loaded["transcription"]]

    scores = pd.read_csv(SCORES_CSV, dtype={"document_sha256": str})
    csv_hashes = set(scores["document_sha256"])
    if len(loaded_hashes) != N_DOCS or len(set(loaded_hashes)) != N_DOCS:
        errors.append(f"loaded {len(loaded_hashes)} docs, {len(set(loaded_hashes))} unique")
    if set(loaded_hashes) != csv_hashes:
        errors.append(f"hash mismatch: {len(set(loaded_hashes) - csv_hashes)} loaded docs not in CSV, "
                      f"{len(csv_hashes - set(loaded_hashes))} CSV docs not loaded")

    wide = scores.pivot(index="document_sha256", columns="method", values="rouge1")
    advantage = wide["lead"] - wide["lexrank"]  # Lead minus LexRank, ROUGE-1

    rows = []
    skipped_no_tokens = 0
    for text, desc, h in zip(loaded["transcription"], loaded["description"], loaded_hashes):
        words = text.split()
        if len(words) < MIN_WORDS:
            continue
        desc_tokens = tokens(str(desc))
        if not desc_tokens:
            skipped_no_tokens += 1
            continue
        first = len(desc_tokens & tokens(" ".join(words[:SEGMENT_WORDS]))) / len(desc_tokens)
        last = len(desc_tokens & tokens(" ".join(words[-SEGMENT_WORDS:]))) / len(desc_tokens)
        rows.append({
            "document_sha256": h,
            "first_recall": first,
            "last_recall": last,
            "position_bias": first - last,
            "lead_minus_lexrank_rouge1": float(advantage[h]),
        })

    out = pd.DataFrame(rows)
    if out["lead_minus_lexrank_rouge1"].isna().any():
        errors.append("missing Lead/LexRank scores for analysed documents")

    bias = out["position_bias"].to_numpy()
    adv = out["lead_minus_lexrank_rouge1"].to_numpy()
    rho, rho_p = stats.spearmanr(bias, adv)
    pos = adv[bias > 0]
    nonpos = adv[bias <= 0]

    out.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL, float_format="%.10g")

    n = len(out)
    print(f"{'PASS' if not errors else 'FAIL'}: analysed {n} documents "
          f"(skipped {skipped_no_tokens} with no description tokens)")
    print(f"first recall mean {out.first_recall.mean():.4f} median {out.first_recall.median():.4f}; "
          f"last recall mean {out.last_recall.mean():.4f} median {out.last_recall.median():.4f}")
    print(f"positive bias: {100 * (bias > 0).mean():.1f}% | bias mean {bias.mean():+.4f} "
          f"median {np.median(bias):+.4f}")
    print(f"Spearman rho(bias, Lead-LexRank) = {rho:+.4f} (p={rho_p:.3e})")
    print(f"Lead-LexRank mean: positive-bias {pos.mean():+.4f} (n={len(pos)}); "
          f"non-positive {nonpos.mean():+.4f} (n={len(nonpos)})")
    print(f"output: {OUT_CSV}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
