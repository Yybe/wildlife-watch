"""Print the failing notebook cell and its exception from an nbconvert log.

Repeated `tr | grep` spelunking over JSON-escaped logs wasted turns, so this is the one command:
    .venv/Scripts/python.exe DL_PBL/scripts/nb_error.py DL_PBL/outputs/logs/notebook08_run9.log
"""
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
i = raw.find("An error occurred while executing the following cell")
if i < 0:
    print("no CellExecutionError in log; tail:\n", raw[-1500:])
    raise SystemExit(1)
seg = raw[i:i + 6000]
end = seg.find("-----", 60)
cell = seg[:seg.find("-----", 200)]
print("=== FAILING CELL ===")
print("\n".join(l for l in cell.splitlines() if l.strip() and "error occurred" not in l)[:1200])
print("=== TRACEBACK / ERROR ===")
print(seg[end:end + 1600])
