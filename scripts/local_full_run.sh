#!/usr/bin/env bash
# Wait for the iWildCam download, verify it, extract it, then run the full classifier comparison
# locally on the GPU. Each step checks the previous one's artifact and appends to a log, so a
# truncated run still says where it stopped.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
ZIP="$ROOT/DL_PBL/data/raw/_dl/iwildcam-v2-0-2020-wilds-dataset.zip"
IMG_DIR="$ROOT/DL_PBL/data/raw/iwildcam_v2.0/train"
LOG="$ROOT/DL_PBL/outputs/logs/local_run.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "waiting for download to finish"
for _ in $(seq 1 240); do
  [ ! -f "$ZIP.kaggle-partial" ] && break
  sleep 30
done
if [ -f "$ZIP.kaggle-partial" ]; then say "FAIL download still not finished after 2h"; exit 1; fi
say "zip present: $(du -h "$ZIP" | cut -f1)"

if ! "$PY" -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); bad=z.testzip(); sys.exit(1 if bad else 0)" "$ZIP" 2>>"$LOG"; then
  say "FAIL zip integrity (testzip)"; exit 1
fi
say "zip integrity ok"

if [ ! -d "$IMG_DIR" ] || [ "$(ls "$IMG_DIR" | wc -l)" -lt 1000 ]; then
  say "extracting to DL_PBL/data/raw"
  "$PY" -c "import zipfile; zipfile.ZipFile(r'$ZIP').extractall(r'$ROOT/DL_PBL/data/raw')" >>"$LOG" 2>&1 || { say "FAIL extract"; exit 1; }
fi
N=$(ls "$IMG_DIR" 2>/dev/null | wc -l)
say "images on disk in train dir: $N"
if [ "$N" -lt 100000 ]; then say "FAIL expected ~129,809 images, got $N"; exit 1; fi

export PYTHONIOENCODING=utf-8 P11_LIMIT=0 P11_EPOCHS=30 P11_PATIENCE=0 P11_BATCH=64
export P11_OUT="C:/Users/xxshi/Desktop/PYTHOM/DL_PBL/outputs/logs/notebook02_full"
mkdir -p "$P11_OUT"
say "training: notebook02 full corpus, 30 epochs without early stopping, batch 64"
cd DL_PBL/notebooks || exit 1
# -m nbconvert, not -m jupyter nbconvert: the jupyter front-end resolves subcommands through PATH and the
# kernelspecs name their interpreter as bare "python", so an Anaconda install can hijack the run.
"$PY" -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 02_baseline_vs_finetune.ipynb >>"$LOG" 2>&1
RC=$?
say "nbconvert exit=$RC"
if [ $RC -ne 0 ]; then tail -40 "$LOG" >&2; exit $RC; fi
say "DONE artifacts in $P11_OUT"
