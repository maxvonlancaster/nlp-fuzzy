"""Run the full CITI-2026 evaluation notebook (1000 cleaned documents, all five methods).

Notebook parameters are NOT overridden. Cell output goes to
results/citi2026_clean/execution.log; only a short summary is printed.
Refuses to start if results/citi2026_clean/evaluation_results.json already exists.
"""
import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results"
OUTPUT_NAMES = ["citi2026_clean", "citi2026_final"]
OUT = RESULTS / "citi2026_clean"  # set from --output-name in main()
OLD_OUTPUT_LINE = 'Path("../results/citi2026_clean")'  # cell 3 output_dir, patched in memory only
NOTEBOOK = SRC / "summarization_evaluation.ipynb"
EXPECTED_METHODS = {"lead", "random", "textrank", "lsa", "lexrank"}
METRICS = ["rouge1", "rouge2", "rougeL"]


def legacy_snapshot():
    """Size and mtime of every results file outside citi2026_clean."""
    snap = {}
    for p in RESULTS.rglob("*"):
        if p.is_file() and OUT not in p.parents:
            snap[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
    return snap


def main():
    global OUT
    parser = argparse.ArgumentParser(description="Run the CITI-2026 evaluation notebook.")
    parser.add_argument("--output-name", choices=OUTPUT_NAMES, default="citi2026_clean")
    parser.add_argument("--check-only", action="store_true",
                        help="validate configuration and output routing; run nothing")
    args = parser.parse_args()
    OUT = RESULTS / args.output_name
    json_path = OUT / "evaluation_results.json"

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell3 = "".join(nb["cells"][3]["source"])
    occurrences = cell3.count(OLD_OUTPUT_LINE)
    if occurrences != 1:
        print(f"ERROR: output-path line found {occurrences} times in cell 3 (expected 1)")
        return 1
    patched_cell3 = cell3.replace(OLD_OUTPUT_LINE, f'Path(r"{OUT}")')

    if args.check_only:
        guard = "present (a real run would be refused)" if json_path.exists() else "absent"
        print(f"CHECK OK: output {OUT}")
        print("output-path replacement: 1 occurrence in cell 3 (in memory only)")
        print(f"overwrite guard: evaluation_results.json {guard}")
        print("legacy protection: snapshot check active; no cells run, no files created")
        return 0
    if json_path.exists():
        print(f"REFUSED: {json_path} already exists; move it aside to re-run.")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "execution.log"
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    before = legacy_snapshot()

    original_cwd = os.getcwd()
    original_path = list(sys.path)
    os.chdir(SRC)
    sys.path.insert(0, str(SRC))

    ns = {"__name__": "__full_evaluation__"}
    failing = None
    error_text = ""
    executed = 0
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            for idx, cell in enumerate(nb["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                src = patched_cell3 if idx == 3 else "".join(cell["source"])
                log.write(f"\n===== cell {idx} =====\n")
                log.flush()
                try:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        exec(compile(src, f"<cell {idx}>", "exec"), ns)
                    executed += 1
                except Exception:
                    failing = idx
                    error_text = traceback.format_exc()
                    log.write(error_text)
                    break
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path

    errors = []
    if failing is not None:
        errors.append(f"cell {failing} failed: {error_text.strip().splitlines()[-1]}")

    if not json_path.exists():
        errors.append("evaluation_results.json not written")
    else:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        methods = data.get("methods", {})
        if set(methods) != EXPECTED_METHODS:
            errors.append(f"methods: {sorted(methods)}")
        if meta.get("n_documents") != 1000:
            errors.append(f"n_documents={meta.get('n_documents')}")
        if meta.get("random_seed") != 42:
            errors.append(f"random_seed={meta.get('random_seed')}")
        if meta.get("clean_for_evaluation") is not True:
            errors.append("clean_for_evaluation is not True")
        for name, m in methods.items():
            for metric in METRICS:
                v = m["mean_scores"].get(metric)
                if v is None or not (0.0 <= v <= 1.0):
                    errors.append(f"{name} {metric}={v}")

    csv_path = OUT / "tables" / "main_results.csv"
    if not csv_path.exists():
        errors.append("tables/main_results.csv missing")

    after = legacy_snapshot()
    changed = [p for p in before if before[p] != after.get(p)] + [p for p in after if p not in before]
    if changed:
        errors.append(f"legacy results changed: {len(changed)} file(s)")

    status = "PASS" if not errors else "FAIL"
    print(f"{status}: executed {executed} code cells")
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for name in sorted(data["methods"]):
            m = data["methods"][name]["mean_scores"]
            print(f"  {name}: R1={m['rouge1']:.4f} R2={m['rouge2']:.4f} RL={m['rougeL']:.4f}")
        meta = data["metadata"]
        print(f"  docs={meta['n_documents']} seed={meta['random_seed']} "
              f"clean={meta['clean_for_evaluation']}")
    print(f"output: {OUT}")
    print(f"log: {log_path}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
