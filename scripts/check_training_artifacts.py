"""Gate the post-training queue on the training actually having run the requested number of epochs.

Why a content check and not a file timestamp: `summary.csv` is written by the notebook's own final cell, so
it is always older than the launcher's completion line. An mtime comparison therefore aborts a successful
run. Reading `epochs_run` asks the question we actually care about - did the arms run at least 25 epochs?

Run:  .venv/Scripts/python.exe DL_PBL/scripts/check_training_artifacts.py [--min-epochs 25]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUM = ROOT / "DL_PBL/outputs/logs/notebook07/summary.csv"
ARMS = ("finetune_reg", "head_only_aug")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-epochs", type=int, default=25)
    ap.add_argument("--suggest-at", type=int, default=25,
                    help="best_epoch at or above this means the epoch budget is binding, so extending it is "
                         "worth the GPU time; -1 disables the recommendation")
    ap.add_argument("--summary", default=str(SUM), help="summary.csv to read (default: the notebook 07 one)")
    a = ap.parse_args()
    sum_path = Path(a.summary)
    if not sum_path.is_file():
        print(f"FAIL no summary at {sum_path}")
        return 1
    df = pd.read_csv(sum_path)
    if "epochs_run" not in df.columns:
        print(f"FAIL {sum_path.name} has no epochs_run column; columns are {list(df.columns)}")
        return 1
    bad = []
    picks = []
    for arm in ARMS:
        row = df[df.model == arm]
        if row.empty:
            bad.append(f"{arm}: absent from summary")
            continue
        r = row.iloc[0]
        got = int(r["epochs_run"])
        best = int(r["best_epoch"]) if "best_epoch" in df.columns else -1
        picks.append((arm, best, got))
        print(f"{arm}: epochs_run={got} best_epoch={best} test_macro_f1={r.get('test_macro_f1', float('nan'))}")
        if got < a.min_epochs:
            bad.append(f"{arm}: ran {got} epochs, below the required {a.min_epochs}")
    if bad:
        print("FAIL " + "; ".join(bad))
        return 1
    print(f"PASS both arms ran >= {a.min_epochs} epochs")
    # Pre-registered 50-epoch rule, applied mechanically so the decision is not made after seeing the score:
    # a longer budget is only worth spending if the *selected* epoch sits at the end of the current budget,
    # i.e. the budget is binding. If selection lands early, epochs 31-50 cannot change the reported model.
    if a.suggest_at >= 0:
        tail = [(arm, best) for arm, best, got in picks if best >= a.suggest_at]
        if tail:
            print(f"RECOMMEND +20 epochs: selection at the end of the budget for "
                  + ", ".join(f"{arm} (best_epoch={b})" for arm, b in tail))
        else:
            print(f"DO NOT extend to 50: no arm's best_epoch is in the last "
                  f"{a.suggest_at}-plus window, so more epochs cannot change the selected checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
