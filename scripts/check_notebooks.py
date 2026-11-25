"""Static + executed-output check for every notebook in the deliverable set.

Run before executing or snapshotting:
    .venv/Scripts/python.exe DL_PBL/scripts/check_notebooks.py
Exits non-zero if a cell cannot compile, if an executed notebook holds an error output or printed
traceback, or if a code cell was never executed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
problems = []
checked = 0
for nb_path in sorted((ROOT / "DL_PBL/notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    checked += 1
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    for i, c in enumerate(nb.get("cells", [])):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            compile(src, f"{nb_path.name}:{i}", "exec")
        except SyntaxError as exc:
            problems.append(f"SYNTAX  {nb_path.name} cell {i}: {exc.msg} (line {exc.lineno})")
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                problems.append(f"ERROR   {nb_path.name} cell {i}: {o['ename']}: {o['evalue'][:120]}")
            if o.get("output_type") == "stream" and "Traceback" in "".join(o.get("text", [])):
                problems.append(f"TRACEBK {nb_path.name} cell {i} printed a traceback")
    executed = [c for c in code_cells if c.get("execution_count")]
    if code_cells and len(executed) != len(code_cells):
        problems.append(f"STALE   {nb_path.name}: {len(executed)}/{len(code_cells)} code cells executed")

print(f"notebooks checked: {checked}")
for line in problems:
    print(line)
if not problems:
    print("all notebooks compile and ran clean")
sys.exit(1 if problems else 0)
