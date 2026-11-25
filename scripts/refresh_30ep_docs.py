"""Refresh the 30-epoch numbers in the human-facing docs from the artifact that produced them.

Why this exists: the queue finishes around 16:10, long after an interactive session is likely to be watching,
and hand-copying numbers from `summary.csv` into three documents is exactly how drift happens. This script
copies values *from the artifact*, so it cannot invent one, and it refuses to touch anything it cannot locate
by an exact anchor.

It deliberately does NOT edit `scripts/verify_claims.py`. Rewriting an expected value to match the artifact
would make that claim unfalsifiable, so it prints the constants a human must review and change instead.

Run:  .venv/Scripts/python.exe DL_PBL/scripts/refresh_30ep_docs.py [--dry-run]
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUM = ROOT / "DL_PBL/outputs/logs/notebook07/summary.csv"
REPORT = ROOT / "DL_PBL/docs/REPORT.md"
CARD = ROOT / "DL_PBL/docs/MODEL_CARD.md"
CONFIG = ROOT / "DL_PBL/configs/experiments.yaml"


def sub_once(path: Path, pattern: str, repl: str, label: str, dry: bool, log: list) -> None:
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        log.append(f"SKIP   {label}: anchor not found in {path.name} ({pattern[:60]})")
        return
    if new == text:
        log.append(f"NOCHANGE {label} in {path.name}")
        return
    if not dry:
        path.write_text(new, encoding="utf-8")
    log.append(f"UPDATE {label} in {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", default=str(SUM), help="notebook 07 summary.csv to read")
    a = ap.parse_args()
    sum_path = Path(a.summary)
    if not sum_path.is_file():
        print(f"FAIL no {sum_path}")
        return 1
    df = pd.read_csv(sum_path)
    need = {"finetune_reg", "head_only_aug"}
    if not need.issubset(set(df.model)):
        print(f"FAIL summary.csv lacks arms: {sorted(need - set(df.model))}")
        return 1
    arm = {m: df[df.model == m].iloc[0] for m in need}
    for m, r in arm.items():
        if int(r["epochs_run"]) < 25:
            print(f"FAIL {m} ran only {int(r['epochs_run'])} epochs; refusing to write it as the 30-epoch result")
            return 1

    A, B = arm["finetune_reg"], arm["head_only_aug"]
    log: list[str] = []

    sub_once(REPORT, r"\| 30 full epochs, no early stopping \(nb 07 rerun\) \|[^\n]*\|",
             f"| 30 full epochs, no early stopping (nb 07 rerun) | {A.val_best:.4f} | {A.test_macro_f1:.4f} "
             f"| {A.test_macro_f1_rich:.4f} | {A.test_accuracy:.4f} |",
             "REPORT results row (arm A)", a.dry_run, log)
    sub_once(REPORT, r"> \*\*\(pending\)\*\* are waiting on the 30-epoch pass and notebook 08[^\n]*",
             "> All numbers below are from executed artifacts: the 30-epoch pass (notebook 07, both arms, "
             "`patience=0`), the predictive alert rule (notebook 08) and the error analysis (notebook 09). "
             "`scripts/verify_claims.py` checks each one against the file that produced it.",
             "REPORT pending banner", a.dry_run, log)

    labels = {"finetune_reg": ("| notebook 07 arm A: fine-tuned `fc,layer4`, strong aug, label smoothing 0.1, ",
                               r"\| notebook 07 arm A:[^\n]*\|"),
              "head_only_aug": ("| notebook 07 arm B: frozen backbone, head only, same aug, ",
                                r"\| notebook 07 arm B:[^\n]*\|")}
    for m, r in (("finetune_reg", A), ("head_only_aug", B)):
        prefix, pattern = labels[m]
        score = f"**{r.test_macro_f1:.4f}**" if m == "finetune_reg" else f"{r.test_macro_f1:.4f}"
        row = (f"{prefix}30 epochs, best-val epoch {int(r.best_epoch)} | pixels | {r.val_best:.4f} "
               f"| {score} | {r.test_macro_f1_rich:.4f} | {r.test_accuracy:.4f} |")
        sub_once(CARD, pattern, row, f"MODEL_CARD {m} row", a.dry_run, log)

    sub_once(CONFIG, r"full_30_epoch_pass: \{[^}]*\}",
             f"full_30_epoch_pass: {{patience: 0, epochs: 30, status: done, armA_epochs_run: {int(A.epochs_run)}, "
             f"armA_best_epoch: {int(A.best_epoch)}, armA_val_best: {A.val_best:.4f}, "
             f"armA_test_macro_f1: {A.test_macro_f1:.4f}, armB_epochs_run: {int(B.epochs_run)}, "
             f"armB_best_epoch: {int(B.best_epoch)}, armB_test_macro_f1: {B.test_macro_f1:.4f}}}\n",
             "configs full_30_epoch_pass", a.dry_run, log)

    print("\n".join(log))
    print("\nManual step (deliberately not automated): update these verify_claims constants after reading "
          "the artifact values, because rewriting an expectation to match its own artifact would make the "
          "claim unfalsifiable:")
    for label, path, key, old in (
            ("nb07 armA test macro F1", "summary.csv", "finetune_reg:test_macro_f1", 0.6461),
            ("nb07 armB test macro F1", "summary.csv", "head_only_aug:test_macro_f1", 0.5702),
            ("nb07 armA epochs run", "summary.csv", "finetune_reg:epochs_run", 19),
            ("nb07 armB epochs run", "summary.csv", "head_only_aug:epochs_run", 15)):
        new = {"finetune_reg:test_macro_f1": A.test_macro_f1, "head_only_aug:test_macro_f1": B.test_macro_f1,
               "finetune_reg:epochs_run": int(A.epochs_run), "head_only_aug:epochs_run": int(B.epochs_run)}[key]
        print(f"  {label:30s} {old} -> {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
