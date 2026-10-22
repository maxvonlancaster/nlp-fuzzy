"""Integration smoke test: five summarizers on 5 clean MTSamples documents.

Parameters mirror notebook cell 8. Each summarizer is called exactly once per
document (inside evaluate_method); returned sentence lists are captured by a
wrapper so selections can be compared without extra calls.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helpers.dataset_loader import load_mtsamples  # noqa: E402
from helpers.config import TextRankConfig  # noqa: E402
from helpers.baselines import (  # noqa: E402
    lead_n_baseline, random_baseline, textrank_summarizer,
    lsa_summarizer, lexrank_tfidf_summarizer, split_sentences,
)
from helpers.evaluation import RougeEvaluator, evaluate_method  # noqa: E402

SEED = 42
N_DOCS = 10
N_SENTENCES = 3
LSA_N_COMPONENTS = 5
METRICS = ['rouge1', 'rouge2', 'rougeL']

textrank_config = TextRankConfig(random_seed=SEED)


def make_methods():
    """Same functions and kwargs as notebook cell 8, with call recording."""
    specs = {
        'lead': (lead_n_baseline, {'n': N_SENTENCES}),
        'random': (random_baseline, {'n': N_SENTENCES, 'seed': SEED}),
        'textrank': (textrank_summarizer, {
            'n': N_SENTENCES,
            'similarity_metric': textrank_config.similarity_metric,
            'damping_factor': textrank_config.damping_factor,
            'max_iter': textrank_config.max_iter,
            'tol': textrank_config.tol,
        }),
        'lsa': (lsa_summarizer, {'n': N_SENTENCES, 'n_components': LSA_N_COMPONENTS}),
        'lexrank': (lexrank_tfidf_summarizer, {
            'n': N_SENTENCES, 'similarity_threshold': 0.1, 'damping_factor': 0.85,
        }),
    }
    return specs


def main():
    failures = []

    _, df = load_mtsamples(limit=N_DOCS, seed=SEED, base_path=str(ROOT / "resources"),
                           clean_for_evaluation=True)
    documents = df['transcription'].tolist()
    references = df['description'].tolist()
    if len(documents) != N_DOCS or len(set(documents)) != N_DOCS:
        failures.append(f"documents: {len(documents)} loaded, {len(set(documents))} unique")

    evaluator = RougeEvaluator(metrics=METRICS, use_stemmer=True,
                               bootstrap_samples=100, confidence_level=0.95, random_seed=SEED)

    selections = {}  # method -> list of returned sentence lists (one per doc)
    lines = []
    for name, (func, kwargs) in make_methods().items():
        calls = {'n': 0}
        returned = []

        def recording(text, _func=func, **kw):
            calls['n'] += 1
            out = _func(text, **kw)
            returned.append(out)
            return out

        summaries_out, errors_out = [], []
        result = evaluate_method(
            method_func=recording, documents=documents, references=references,
            method_name=name, evaluator=evaluator, method_kwargs=kwargs,
            verbose=False, summaries_out=summaries_out, errors_out=errors_out,
        )
        selections[name] = returned

        if len(summaries_out) != N_DOCS:
            failures.append(f"{name}: {len(summaries_out)} summaries")
        if calls['n'] != N_DOCS:
            failures.append(f"{name}: {calls['n']} calls")
        if errors_out:
            failures.append(f"{name}: {len(errors_out)} errors, first {errors_out[0]}")
        for metric in METRICS:
            vals = result.rouge_scores[metric]
            if len(vals) != N_DOCS or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in vals):
                failures.append(f"{name}: {metric} scores not finite/in [0,1]")
        too_long = [len(s) for s in returned if len(s) > N_SENTENCES]
        if too_long:
            failures.append(f"{name}: summaries longer than {N_SENTENCES} sentences: {too_long}")

        lines.append(f"{name}: R1={result.mean_scores['rouge1']:.4f} "
                     f"R2={result.mean_scores['rouge2']:.4f} "
                     f"RL={result.mean_scores['rougeL']:.4f}")

    agree = 0
    for doc, tr, lx in zip(documents, selections['textrank'], selections['lexrank']):
        sents = split_sentences(doc)
        tr_idx = [sents.index(s) for s in tr]
        lx_idx = [sents.index(s) for s in lx]
        agree += tr_idx == lx_idx

    status = "PASS" if not failures else "FAIL"
    print(f"{status}")
    for line in lines:
        print(line)
    print(f"TextRank vs LexRank identical selections: {agree}/{N_DOCS} documents")
    for f in failures:
        print(f"  - {f}")

    # Edge-case probe (findings only, does not affect PASS/FAIL)
    edge_doc = "The and of is. A the to in. It was a of. This is the of."
    print(f"Edge probe: {len(split_sentences(edge_doc))} stopword-only sentences")
    for name, func in [('textrank', textrank_summarizer), ('lexrank', lexrank_tfidf_summarizer)]:
        try:
            out = func(edge_doc, **make_methods()[name][1])
            print(f"  {name}: no exception, returned {len(out)} sentences")
        except Exception as e:
            print(f"  {name}: exception {type(e).__name__}: {str(e)[:80]}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
