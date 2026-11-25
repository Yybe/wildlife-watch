#!/usr/bin/env bash
# Re-run notebook 09 after the queue's equal-length control releases the GPU, then re-check the deliverable.
#
# Why a separate waiter: the queue ran notebook 09 at 15:53 and it failed on a real bug (a filtered array
# re-combined with the full-length mask, `pred[m] == 0) & m`), which is now fixed in both
# scripts/make_notebook09.py and the notebook. The queue had already moved past that step, so nothing else
# would re-execute it. Running it alongside the training control was also rejected: the GPU sat at 5957/6144
# MiB with both, which is an OOM race that would have cost the control's 100 minutes.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
export PATH="$ROOT/.venv/Scripts:$PATH"
export PYTHONIOENCODING=utf-8
LOG="$ROOT/DL_PBL/outputs/logs/finish_nb09.log"
QLOG="$ROOT/DL_PBL/outputs/logs/orchestrate.log"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

: > "$LOG"
say "waiting for the queue to reach its final step"
for _ in $(seq 1 240); do
  grep -q "claim verifier exit=" "$QLOG" 2>/dev/null && break
  sleep 60
done
grep -q "claim verifier exit=" "$QLOG" || say "WARN: queue did not finish within the wait budget"

say "executing 09_error_analysis.ipynb"
cd "$ROOT/DL_PBL/notebooks" || exit 1
"$ROOT/.venv/Scripts/python.exe" -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 09_error_analysis.ipynb >> "$LOG" 2>&1
say "notebook 09 exit=$?"

"$ROOT/.venv/Scripts/python.exe" "$ROOT/DL_PBL/scripts/check_notebooks.py" >> "$LOG" 2>&1
say "notebook checker exit=$?"
"$ROOT/.venv/Scripts/python.exe" "$ROOT/DL_PBL/scripts/verify_claims.py" >> "$LOG" 2>&1
say "claim verifier exit=$?"
say "done; snapshot when both are 0: bash DL_PBL/scripts/phase_i_snapshot.sh"
