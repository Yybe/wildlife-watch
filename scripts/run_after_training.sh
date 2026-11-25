#!/usr/bin/env bash
# Run the post-training notebook queue once the 30-epoch training job finishes.
#
# Two failures this version is written against:
#  1. The previous copy watched notebook07_full30.log for a sentinel string its launcher never wrote, so it
#     would have waited out its whole budget and then run the queue against stale artifacts. It now waits on
#     the exit line the wake-lock wrapper actually prints, and refuses to continue on a non-zero exit.
#  2. On 2026-10-24 the laptop power-cycled, and afterwards `python -m jupyter nbconvert` resolved the
#     nbconvert subcommand through PATH to Anaconda, whose kernel has no torch - the notebook died in 8.8 s
#     with ModuleNotFoundError. So: call the nbconvert module directly and pin the venv's Scripts to the
#     front of PATH, because both kernelspecs spell their interpreter as bare "python".
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
VENV_BIN="$ROOT/.venv/Scripts"
export PATH="$VENV_BIN:$PATH"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="$VENV_BIN/python.exe"
LOG="$ROOT/DL_PBL/outputs/logs/orchestrate.log"
TRAIN_LOG="${P11_TRAIN_LOG:-$ROOT/DL_PBL/outputs/logs/notebook07_30ep_v3.log}"
TRIES="${P11_QUEUE_TRIES:-400}"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

: > "$LOG"
say "queue armed; watching $TRAIN_LOG"
exit_line=""
for _ in $(seq 1 "$TRIES"); do
  exit_line="$(grep -E '^[0-9:]+ exit=[0-9]+ after' "$TRAIN_LOG" 2>/dev/null | tail -1)"
  [ -n "$exit_line" ] && break
  sleep 60
done
if [ -z "$exit_line" ]; then
  say "ABORT: training never reported an exit code within the wait budget"
  exit 1
fi
say "training finished: $exit_line"
case "$exit_line" in
  *" exit=0 "*) ;;
  *) say "ABORT: training exited non-zero, artifacts would be stale"; exit 1 ;;
esac

# Prove the artifacts the queue depends on are from a run that actually satisfied the epoch requirement.
# A timestamp comparison cannot do this: summary.csv is written by the notebook's last cell, so it is always
# older than the launcher's completion line.
CKPT="$ROOT/DL_PBL/outputs/checkpoints/finetune_reg.pt"
[ -f "$CKPT" ] || { say "ABORT: no arm A checkpoint at $CKPT"; exit 1; }
"$PY" "$ROOT/DL_PBL/scripts/check_training_artifacts.py" --min-epochs "${P11_MIN_EPOCHS:-25}" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  say "ABORT: training artifacts do not meet the epoch gate (see $LOG)"
  tail -4 "$LOG"
  exit 1
fi
say "artifact gate passed: both arms ran >= ${P11_MIN_EPOCHS:-25} epochs"

cd "$ROOT/DL_PBL/notebooks" || exit 1
run_nb() {  # $1 notebook, $2 timeout seconds, $3 label
  say "running $1 ($3)"
  "$PY" -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout="$2" "$1" >> "$LOG" 2>&1
  say "$1 exit=$?"
}
run_nb 08_predictive_alert.ipynb 3600 "predictive alert, CPU"
run_nb 06_architecture_trace.ipynb 1200 "architecture trace, GPU now free"
run_nb 03_from_scratch_fundamentals.ipynb 1800 "numbers were computed on the superseded recarved splits"
run_nb 09_error_analysis.ipynb 1800 "error analysis, uses the arm A checkpoint"

# Arm A's advantage over notebook 02's naive fine-tune is currently confounded with training length (19 epochs
# vs 8). Re-run the naive recipe - no best-val restore, final-epoch selection - at 30 epochs so the two arms
# differ only in recipe. Deliberately not --inplace and into its own artifact dir: notebooks/02 remains the
# executed baseline-vs-fine-tune deliverable the docs cite, this is supplementary evidence.
say "running 02's recipe at 30 epochs as an equal-length naive control (notebook02_30ep)"
P11_LIMIT=0 P11_EPOCHS=30 P11_PATIENCE=0 P11_BATCH=64 \
  P11_OUT="$ROOT/DL_PBL/outputs/logs/notebook02_30ep" \
  "$PY" -m nbconvert --to notebook --execute \
        --output nb02_equal_length_control.ipynb \
        --output-dir "$ROOT/DL_PBL/outputs/logs/notebook02_30ep" \
        --ExecutePreprocessor.timeout=7200 "$ROOT/DL_PBL/notebooks/02_baseline_vs_finetune.ipynb" >> "$LOG" 2>&1
say "02 naive control exit=$?"

"$PY" "$ROOT/DL_PBL/scripts/check_notebooks.py" >> "$LOG" 2>&1
say "notebook checker exit=$?"
"$PY" "$ROOT/DL_PBL/scripts/verify_claims.py" >> "$LOG" 2>&1
say "claim verifier exit=$?  (non-zero is expected until docs carry the new 30-epoch numbers)"

# Independent recomputation of arm A's metrics and gate numbers from the saved checkpoint, so no headline
# number in the docs survives only because a notebook printed it once.
"$PY" "$ROOT/DL_PBL/scripts/check_armA_gate.py" >> "$LOG" 2>&1
say "arm A gate re-verification exit=$?  (its output is the recomputed table, compare with MODEL_CARD)"

# Replication on cloud GPUs was withdrawn: every number in this repo comes from the local run.
