"""Validate reconstruction of Lead-3 and LexRank summaries against saved per-document scores.

Required before any shuffled-reference control. Only Lead-3 and TF-IDF LexRank are
regenerated; no summaries or clinical text are saved or printed.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helpers.dataset_loader import load_mtsamples  # noqa: E402
from helpers.baselines import lead_n_baseline, lexrank_tfidf_summarizer  # noqa: E402
from helpers.evaluation import RougeEvaluator  # noqa: E402

SCORES_CSV = ROOT / "results" / "citi2026_final" / "tables" / "per_document_scores.csv"
N_DOCS = 1000
SEED = 42
N_SENTENCES = 3
METRICS = ["rouge1", "rouge2", "rougeL"]
TOL = 1e-9
METHODS = {
    "lead": (lead_n_baseline, {"n": N_SENTENCES}),
    "lexrank": (lexrank_tfidf_summarizer, {"n": N_SENTENCES, "similarity_threshold": 0.1,
                                           "damping_factor": 0.85}),
}


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    errors = []
    _, df = load_mtsamples(limit=N_DOCS, seed=SEED, base_path=str(ROOT / "resources"),
                           clean_for_evaluation=True)
    loaded = df.reset_index(drop=True)
    hashes = [sha256(t) for t in loaded["transcription"]]
    if len(set(hashes)) != N_DOCS:
        errors.append(f"reloaded {len(set(hashes))} unique documents, expected {N_DOCS}")

    saved = pd.read_csv(SCORES_CSV, dtype={"document_sha256": str})
    saved = saved[saved["method"].isin(METHODS)]
    saved_idx = saved.set_index(["method", "document_sha256"])
    unmatched = set(hashes) - set(saved["document_sha256"])
    if unmatched:
        errors.append(f"{len(unmatched)} reloaded documents not found in saved scores")

    evaluator = RougeEvaluator(metrics=METRICS, use_stemmer=True, bootstrap_samples=1000,
                               confidence_level=0.95, random_seed=SEED)
    references = loaded["description"].astype(str).tolist()

    report = {}
    matched_docs = 0
    for name, (func, kwargs) in METHODS.items():
        summaries = []
        failures = 0
        for doc in loaded["transcription"]:
            try:
                out = func(doc, **kwargs)
                summaries.append(" ".join(out) if isinstance(out, list) else out)
            except Exception:
                failures += 1
                summaries.append("")
        if failures:
            errors.append(f"{name}: {failures} summarisation exceptions")

        scores = evaluator.score_batch(references, summaries)
        max_diff = {}
        mismatches = 0
        for i, h in enumerate(hashes):
            if (name, h) not in saved_idx.index:
                mismatches += 1
                continue
            row = saved_idx.loc[(name, h)]
            for m in METRICS:
                d = abs(float(scores[m][i]) - float(row[m]))
                max_diff[m] = max(max_diff.get(m, 0.0), d)
                if d > TOL:
                    mismatches += 1
        if mismatches:
            errors.append(f"{name}: {mismatches} score mismatches above tolerance")
        matched_docs = max(matched_docs, sum((name, h) in saved_idx.index for h in hashes))
        report[name] = max_diff

    status = "PASS" if not errors else "FAIL"
    print(f"{status}: {matched_docs} documents matched by SHA-256")
    for name, diffs in report.items():
        parts = " ".join(f"{m}={diffs.get(m, float('nan')):.3e}" for m in METRICS)
        print(f"  {name} max |diff|: {parts}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
