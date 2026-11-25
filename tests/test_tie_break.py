"""Tie-break control for `forecast.day_level_budget`.

A rule whose scores are all equal has no ranking at all, so whatever number it prints is a property of how
ties are broken. This test pins that down: it builds an all-tied case whose positives live only in the last
11 of 40 days, shows that the previous calendar-order tie-break reports a clean-looking 0.000, and shows that
the seeded tie-break reports a value near chance instead. `tied_at_cut` must state the tie openly.

Run:  .venv/Scripts/python.exe DL_PBL/tests/test_tie_break.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DL_PBL/src/sequence"))
import forecast as F  # noqa: E402

OUT = ROOT / "DL_PBL/outputs/logs/test_tie_break_summary.txt"
K_DAYS, N_DAYS, N_POS = 10, 40, 11


def calendar_tie_break(points: pd.DataFrame, col: str, k: int) -> int:
    """The behaviour this test replaces: sort by score then by date."""
    days = points.groupby(["site", "day"], as_index=False)[["label", col]].max()
    top = days.sort_values([col, "day"], ascending=[False, True]).head(k)
    return int(top.label.sum())


def main() -> int:
    flat = pd.DataFrame({"site": 1, "day": [f"d{i:02d}" for i in range(N_DAYS)],
                         "label": [1 if i >= N_DAYS - N_POS else 0 for i in range(N_DAYS)],
                         "const": 0.0})
    chance = K_DAYS * N_POS / N_DAYS
    seeded = F.day_level_budget(flat, ["const"], K_DAYS).iloc[0]
    by_date = calendar_tie_break(flat, "const", K_DAYS)

    checks = []
    checks.append(("tie is reported, not hidden", int(seeded.tied_at_cut) == N_DAYS))
    checks.append(("budget spent exactly", int(seeded.true_days) + int(seeded.false_days) == K_DAYS))
    checks.append(("calendar tie-break yields the fabricated zero", by_date == 0))
    checks.append(("seeded tie-break is not the fabricated zero", int(seeded.true_days) > 0))
    checks.append(("seeded result sits in the chance band", abs(int(seeded.true_days) - chance) <= 2))
    same = F.day_level_budget(flat, ["const"], K_DAYS).iloc[0]
    checks.append(("same seed reproduces", (int(same.true_days), same.recall)
                   == (int(seeded.true_days), seeded.recall)))

    lines = [f"n_days={N_DAYS} positives={N_POS} budget={K_DAYS}",
             f"chance_true_days={chance:.2f}",
             f"calendar_tie_break_true_days={by_date}",
             f"seeded_tie_break_true_days={int(seeded.true_days)}",
             f"seeded_tie_break_recall={seeded.recall}",
             f"tied_at_cut={int(seeded.tied_at_cut)}"]
    failed = [name for name, ok in checks if not ok]
    lines.insert(0, f"{len(checks) - len(failed)}/{len(checks)} checks passed"
                    + (f" | FAILED: {failed}" if failed else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
