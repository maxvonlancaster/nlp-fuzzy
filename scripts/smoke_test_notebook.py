"""End-to-end smoke test: run the evaluation notebook's code cells on 5 documents.

Cell source is modified in memory only (N_DOCUMENTS=5, output_dir -> temp dir).
The notebook file and results/ are never written to.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NOTEBOOK = SRC / "summarization_evaluation.ipynb"
EXPECTED_METHODS = {"lead", "random", "textrank", "lsa", "lexrank"}


def cell_source(cell):
    return "".join(cell["source"])


def patch(src, tmp_out):
    src = src.replace("N_DOCUMENTS = 1000", "N_DOCUMENTS = 5", 1)
    src = src.replace('Path("../results/citi2026_clean")', f'Path(r"{tmp_out}")', 1)
    return src


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tmp_out = Path(tempfile.mkdtemp(prefix="nb_smoke_"))

    original_cwd = os.getcwd()
    original_path = list(sys.path)
    os.chdir(SRC)  # notebook uses relative paths ('../resources')
    sys.path.insert(0, str(SRC))

    ns = {"__name__": "__notebook_smoke__"}
    failing = None
    error_text = ""
    executed = 0
    try:
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = patch(cell_source(cell), tmp_out)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(src, f"<cell {idx}>", "exec"), ns)
                executed += 1
            except Exception:
                failing = idx
                error_text = traceback.format_exc(limit=3)
                break
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path

    failures = []
    if failing is not None:
        failures.append(f"cell {failing} failed")

    json_path = tmp_out / "evaluation_results.json"
    csv_path = tmp_out / "tables" / "main_results.csv"
    methods_found = set()
    if json_path.exists():
        methods_found = set(json.loads(json_path.read_text(encoding="utf-8"))["methods"])
        if methods_found != EXPECTED_METHODS:
            failures.append(f"methods in JSON: {sorted(methods_found)}")
    else:
        failures.append("evaluation_results.json missing")
    if not csv_path.exists():
        failures.append("main_results.csv missing")

    status = "PASS" if not failures else "FAIL"
    print(f"{status}: executed {executed} code cells")
    print(f"output dir (temp): {tmp_out}")
    print(f"JSON methods: {sorted(methods_found)}; CSV exists: {csv_path.exists()}")
    for f in failures:
        print(f"  - {f}")
    if error_text:
        print(error_text.strip().splitlines()[-1])
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
