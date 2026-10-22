"""Smoke test: Lead-3 on 5 clean MTSamples documents through evaluate_method()."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helpers.dataset_loader import load_mtsamples  # noqa: E402
from helpers.baselines import lead_n_baseline  # noqa: E402
from helpers.evaluation import RougeEvaluator, evaluate_method  # noqa: E402

SEED = 42
N_DOCS = 5
N_SENTENCES = 3
METRICS = ['rouge1', 'rouge2', 'rougeL']

calls = {'n': 0}


def counting_lead(text, **kwargs):
    calls['n'] += 1
    return lead_n_baseline(text, **kwargs)


def main():
    failures = []

    _, df = load_mtsamples(limit=N_DOCS, seed=SEED, base_path=str(ROOT / "resources"),
                           clean_for_evaluation=True)
    documents = df['transcription'].tolist()
    references = df['description'].tolist()
    unique_docs = len(set(documents))
    if len(documents) != N_DOCS or unique_docs != N_DOCS:
        failures.append(f"documents: {len(documents)} loaded, {unique_docs} unique (expected {N_DOCS})")

    evaluator = RougeEvaluator(metrics=METRICS, use_stemmer=True,
                               bootstrap_samples=100, confidence_level=0.95, random_seed=SEED)
    summaries_out, errors_out = [], []
    result = evaluate_method(
        method_func=counting_lead, documents=documents, references=references,
        method_name='lead', evaluator=evaluator,
        method_kwargs={'n': N_SENTENCES}, verbose=False,
        summaries_out=summaries_out, errors_out=errors_out,
    )

    if len(summaries_out) != N_DOCS:
        failures.append(f"summaries: {len(summaries_out)} (expected {N_DOCS})")
    if errors_out:
        failures.append(f"errors: {len(errors_out)} (first: {errors_out[0]})")
    if calls['n'] != N_DOCS:
        failures.append(f"summarizer calls: {calls['n']} (expected {N_DOCS})")

    for metric in METRICS:
        scores = result.rouge_scores[metric]
        bad = [s for s in scores if not (0.0 <= float(s) <= 1.0)]
        if len(scores) != N_DOCS or bad:
            failures.append(f"{metric}: {len(scores)} scores, {len(bad)} out of [0,1]")
        else:
            print(f"{metric} mean={result.mean_scores[metric]:.4f}")

    status = "PASS" if not failures else "FAIL"
    print(f"{status}: docs={len(documents)} unique={unique_docs} summaries={len(summaries_out)} "
          f"errors={len(errors_out)} calls={calls['n']}")
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
