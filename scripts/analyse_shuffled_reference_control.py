"""Shuffled-reference negative control for the positional-bias association.

Summaries (Lead-3, LexRank) are reconstructed once and reused. Descriptions are
permuted within strata (specialty x reference-length tercile) 100 times with a
local NumPy RNG (seed 42). Aggregate statistics only; no clinical text or
generated summaries are saved or printed.
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
from helpers.baselines import lead_n_baseline, lexrank_tfidf_summarizer  # noqa: E402
from helpers.evaluation import RougeEvaluator  # noqa: E402

SCORES_CSV = ROOT / "results" / "citi2026_final" / "tables" / "per_document_scores.csv"
N_DOCS = 1000
SEED = 42
N_PERM = 100
MAX_REDRAW = 20
WIN = 100
MIN_WORDS = 200
TOL = 1e-9
TOKEN_RE = re.compile(r"[a-z0-9]+")
METHODS = {
    "lead": (lead_n_baseline, {"n": 3}),
    "lexrank": (lexrank_tfidf_summarizer, {"n": 3, "similarity_threshold": 0.1, "damping_factor": 0.85}),
}


def tokens(text):
    return set(TOKEN_RE.findall(str(text).lower()))


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stratified_permutation(rng, keys, n_fixed_tries=MAX_REDRAW):
    """Permute indices within strata; redraw to avoid fixed points where feasible."""
    m = len(keys)
    perm = np.arange(m)
    fallbacks = 0
    for key in np.unique(keys):
        members = np.where(keys == key)[0]
        if len(members) < 2:
            continue
        for attempt in range(n_fixed_tries):
            shuffled = rng.permutation(members)
            if not np.any(shuffled == members):
                break
        else:
            fallbacks += 1
        perm[members] = shuffled
    return perm, fallbacks


def main():
    errors = []
    _, df = load_mtsamples(limit=N_DOCS, seed=SEED, base_path=str(ROOT / "resources"),
                           clean_for_evaluation=True)
    df = df.reset_index(drop=True)
    hashes = [sha256(t) for t in df["transcription"]]
    saved = pd.read_csv(SCORES_CSV, dtype={"document_sha256": str})
    saved = saved[saved["method"].isin(METHODS)].set_index(["method", "document_sha256"])
    if len(set(hashes)) != N_DOCS or any(("lead", h) not in saved.index for h in hashes):
        errors.append("reloaded documents do not match saved hashes")

    evaluator = RougeEvaluator(metrics=["rouge1", "rouge2", "rougeL"], use_stemmer=True,
                               bootstrap_samples=1000, confidence_level=0.95, random_seed=SEED)
    refs_all = df["description"].astype(str).tolist()
    texts = df["transcription"].tolist()

    summaries = {}
    for name, (func, kwargs) in METHODS.items():
        out = []
        for doc in texts:
            s = func(doc, **kwargs)
            out.append(" ".join(s) if isinstance(s, list) else s)
        summaries[name] = out
        scores = evaluator.score_batch(refs_all, out)["rouge1"]
        for i, h in enumerate(hashes):
            if abs(float(scores[i]) - float(saved.loc[(name, h), "rouge1"])) > TOL:
                errors.append(f"{name} ROUGE-1 mismatch at document {i}")
                break

    # Analysis subset: >= 200 words and non-empty description tokens
    desc_tok_all = [tokens(d) for d in refs_all]
    sub = [i for i, t in enumerate(texts) if len(t.split()) >= MIN_WORDS and desc_tok_all[i]]
    m = len(sub)
    sub_texts = [texts[i] for i in sub]
    first_tok = [tokens(" ".join(w.split()[:WIN])) for w in sub_texts]
    last_tok = [tokens(" ".join(w.split()[-WIN:])) for w in sub_texts]
    D = [desc_tok_all[i] for i in sub]
    sub_refs = [refs_all[i] for i in sub]
    lead_sub = [summaries["lead"][i] for i in sub]
    lex_sub = [summaries["lexrank"][i] for i in sub]

    def bias_for(Dsets):
        return np.array([len(d & f) / len(d) - len(d & l) / len(d)
                         for d, f, l in zip(Dsets, first_tok, last_tok)])

    lead_r1 = np.array([float(saved.loc[("lead", hashes[i]), "rouge1"]) for i in sub])
    lex_r1 = np.array([float(saved.loc[("lexrank", hashes[i]), "rouge1"]) for i in sub])
    obs_adv = lead_r1 - lex_r1
    obs_bias = bias_for(D)
    obs_rho = stats.spearmanr(obs_bias, obs_adv)[0]

    # Strata: specialty x reference-length tercile (pool sparse specialties by length only)
    spec = df["medical_specialty"].astype(str).to_numpy()[sub]
    ref_len = np.array([len(r.split()) for r in sub_refs])
    cuts = np.quantile(ref_len, [1 / 3, 2 / 3])
    band = np.digitize(ref_len, cuts)
    keys_raw = np.array([f"{s}|{b}" for s, b in zip(spec, band)])
    uniq, counts = np.unique(keys_raw, return_counts=True)
    small = set(uniq[counts < 2])
    keys = np.array([f"pool|{b}" if k in small else k for k, b in zip(keys_raw, band)])

    rng = np.random.default_rng(SEED)
    null_rho, changed_frac, zero_lead, zero_lex = [], [], [], []
    mean_lead, mean_lex, fallback_total, invalid = [], [], 0, 0
    for _ in range(N_PERM):
        perm, fb = stratified_permutation(rng, keys)
        fallback_total += fb
        changed_frac.append(float(np.mean(perm != np.arange(m))))
        Dk = [D[p] for p in perm]
        refs_k = [sub_refs[p] for p in perm]
        bias_k = bias_for(Dk)
        lead_k = evaluator.score_batch(refs_k, lead_sub)["rouge1"]
        lex_k = evaluator.score_batch(refs_k, lex_sub)["rouge1"]
        lead_k = np.asarray(lead_k, dtype=float)
        lex_k = np.asarray(lex_k, dtype=float)
        adv_k = lead_k - lex_k
        rho_k = stats.spearmanr(bias_k, adv_k)[0]
        if np.isnan(rho_k):
            invalid += 1
        else:
            null_rho.append(rho_k)
        mean_lead.append(lead_k.mean())
        mean_lex.append(lex_k.mean())
        zero_lead.append(float(np.mean(lead_k == 0)))
        zero_lex.append(float(np.mean(lex_k == 0)))

    null_rho = np.array(null_rho)
    exceed = float(np.mean(null_rho >= obs_rho))
    p2, p50, p97 = np.percentile(null_rho, [2.5, 50, 97.5])

    status = "PASS" if not errors else "FAIL"
    print(f"{status}: {m} analysed documents; {N_PERM} permutations; "
          f"reconstruction ROUGE-1 verified for {N_DOCS} documents")
    print(f"observed rho={obs_rho:+.4f}; shuffled median={p50:+.4f} "
          f"[2.5%={p2:+.4f}, 97.5%={p97:+.4f}]; one-sided exceedance={exceed:.3f}")
    print(f"changed description fraction (mean)={np.mean(changed_frac):.3f}; "
          f"strata fallbacks={fallback_total}; invalid iterations={invalid}")
    print(f"shuffled mean ROUGE-1: lead={np.mean(mean_lead):.4f} lexrank={np.mean(mean_lex):.4f}; "
          f"zero-score prevalence: lead={np.mean(zero_lead):.4f} lexrank={np.mean(zero_lex):.4f}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
