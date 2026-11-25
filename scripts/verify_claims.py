"""Check that every number quoted in the human-facing docs still matches the artifacts that produced it.

Why this exists: in one session the same metric appeared in a notebook, a model card and a config file,
and one of them was a stale hour-point result reported as a pass. Documentation drift is the failure mode
that survives a green test suite, so the claim table below is checked against the JSON/CSV artifacts
rather than against memory.

Run:  .venv/Scripts/python.exe DL_PBL/scripts/verify_claims.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "DL_PBL/outputs/logs"


def jsonl_last(name: str) -> dict:
    """Last row of an experiment log; `name` is relative to the repo root."""
    path = ROOT / name
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{name} has no rows")
    return rows[-1]


def dig(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


# (claim label, artifact, dotted path, expected). Expected values come from executed runs; if a rerun
# changes a number, this script fails until the docs are updated to match, which is the point.
CLAIMS = [
    ("splits mode", "DL_PBL/data/splits/manifest.json", "mode", "official"),
    ("split train size", "DL_PBL/data/splits/manifest.json", "split_sizes.train", 36060),
    ("split test size", "DL_PBL/data/splits/manifest.json", "split_sizes.test", 19928),
    ("test camera sites", "DL_PBL/data/splits/manifest.json", "split_locations.test", 44),
    ("no sequence overlap", "DL_PBL/data/splits/manifest.json", "leakage_checks.train_vs_test.seq", 0),
    ("no camera-day overlap", "DL_PBL/data/splits/manifest.json", "leakage_checks.train_vs_test.camera_day", 0),
    ("nb02 baseline test macro F1", "DL_PBL/outputs/logs/notebook02_full/experiment_log.jsonl",
     "baseline_test_macro_f1", 0.6245),
    ("nb02 fine-tune lost to baseline", "DL_PBL/outputs/logs/notebook02_full/experiment_log.jsonl",
     "delta_vs_baseline", -0.0202),
    ("nb05 detector mAP50 dedup", "DL_PBL/outputs/logs/notebook05/experiment_log.jsonl",
     "test_deduplicated.mAP50", 0.9469),
    ("nb05 detector mAP50-95 dedup", "DL_PBL/outputs/logs/notebook05/experiment_log.jsonl",
     "test_deduplicated.mAP50_95", 0.794),
    ("nb05 latency ms/frame", "DL_PBL/outputs/logs/notebook05/experiment_log.jsonl", "ms_per_frame", 26.8),
    ("nb04 camera-days in test", "DL_PBL/outputs/logs/notebook04/experiment_log.jsonl",
     "gate.camera_days", 633),
    ("nb04 per-frame false alerts", "DL_PBL/outputs/logs/notebook04/experiment_log.jsonl",
     "gate.per_frame.false_alerts", 58),
    ("nb04 gate false alerts", "DL_PBL/outputs/logs/notebook04/experiment_log.jsonl",
     "gate.gated.false_alerts", 11),
    ("nb04 gate recall", "DL_PBL/outputs/logs/notebook04/experiment_log.jsonl",
     "gate.gated.recall_of_qualifying_days", 0.4532),
    ("nb07 armA test macro F1", "DL_PBL/outputs/logs/notebook07/summary.csv", "row:finetune_reg:test_macro_f1",
     0.6403),
    ("nb07 armB test macro F1", "DL_PBL/outputs/logs/notebook07/summary.csv", "row:head_only_aug:test_macro_f1",
     0.548),
    # How long each arm actually trained is a claim too: the run requirement was >=25 epochs, and before
    # 2026-10-24 the arms had stopped at 19 and 15 under patience=8.
    ("nb07 armA epochs run", "DL_PBL/outputs/logs/notebook07/summary.csv", "row:finetune_reg:epochs_run", 30),
    ("nb07 armB epochs run", "DL_PBL/outputs/logs/notebook07/summary.csv", "row:head_only_aug:epochs_run", 30),
    # Notebook 08 forecast arms: the ceiling, the domain-shift ablation and both bar verdicts.
    ("nb08 evidence ceiling", "DL_PBL/outputs/logs/notebook08/experiment_log.jsonl",
     "evidence_cap_recall_ceiling", 0.8723),
    ("nb08 logistic AUC with site fingerprint", "DL_PBL/outputs/logs/notebook08/experiment_log.jsonl",
     "ablation_no_site_prior.auc_test_with_site", 0.3122),
    ("nb08 logistic AUC without site fingerprint", "DL_PBL/outputs/logs/notebook08/experiment_log.jsonl",
     "ablation_no_site_prior.auc_test_without_site", 0.8541),
    ("nb08 val AUC without site fingerprint", "DL_PBL/outputs/logs/notebook08/experiment_log.jsonl",
     "ablation_no_site_prior.auc_val_without_site", 0.5624),
    ("nb08 forecast recall without site fingerprint", "DL_PBL/outputs/logs/notebook08/experiment_log.jsonl",
     "ablation_no_site_prior.recall_at_budget_without", 0.4894),
    ("nb08 bar vs max(volume,persistence) not cleared", "DL_PBL/outputs/logs/notebook08/experiment_log.jsonl",
     "ablation_no_site_prior.bar_vs_max_counter", False),
    ("nb08 bar vs volume cleared", "DL_PBL/outputs/logs/notebook08/experiment_log.jsonl",
     "ablation_no_site_prior.bar_vs_volume", True),
    # Detection leakage, re-measured independently of notebook 05 by scripts/audit_yolo_split_overlap.py.
    ("detection extra duplicate copies", "DL_PBL/outputs/logs/yolo_split_overlap.txt", "extra_copies", 41),
    ("detection cross-split groups", "DL_PBL/outputs/logs/yolo_split_overlap.txt", "groups_spanning_splits", 17),
    ("test images leaked into other splits", "DL_PBL/outputs/logs/yolo_split_overlap.txt",
     "test_images_duplicated_elsewhere", 11),
    ("leaked test share", "DL_PBL/outputs/logs/yolo_split_overlap.txt", "test_share_duplicated", 0.0728),
    ("fundamentals gradient check", "DL_PBL/outputs/logs/test_from_scratch_summary.txt",
     "gradient_check_max_abs_error", 3.513e-10),
    ("fundamentals svd rank", "DL_PBL/outputs/logs/test_from_scratch_summary.txt",
     "svd_rank_for_95pct_variance", 6),
    ("fundamentals nesterov beats momentum", "DL_PBL/outputs/logs/test_from_scratch_summary.txt",
     "loss_nesterov", 0.3855),
    ("fundamentals checks all pass", "DL_PBL/outputs/logs/test_from_scratch_summary.txt",
     "21/21", "21/21"),
]

# Licence prose must keep quoting the source text, not our paraphrase of it. Each phrase below is a
# verbatim quote from the pages recorded in docs/LICENCE_EVIDENCE.md; if the evidence file is edited
# until the quote disappears, the card's claim loses its support and this check fails.
for _label, _file, _phrase in [
    ("licence: CDLA clause 3.4 quoted", "DL_PBL/docs/LICENCE_EVIDENCE.md",
     "This Agreement imposes no obligations or restrictions on Your Use or Publication of Results"),
    ("licence: CDLA sharing condition quoted", "DL_PBL/docs/LICENCE_EVIDENCE.md",
     "preserve all credit or attribution to the Data Provider"),
    ("licence: upstream imagery quote", "DL_PBL/docs/LICENCE_EVIDENCE.md",
     "Images captured by Snapshot Safari are licensed under a Creative Commons Attribution 4.0"),
    ("dataset card states CDLA", "DL_PBL/docs/DATASET_CARD.md", "CDLA-Permissive-1.0"),
    ("dataset card states CC BY 4.0", "DL_PBL/docs/DATASET_CARD.md", "CC BY 4.0"),
]:
    CLAIMS.append((_label, _file, "has:" + _phrase, _phrase))

DOC_FILES = ["DL_PBL/docs/MODEL_CARD.md", "DL_PBL/docs/REPORT.md", "DL_PBL/configs/experiments.yaml",
             "DL_PBL/docs/DATASET_CARD.md"]


def summary_value(text: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)}=(-?[0-9.eE+-]+)", text)
    if not m:
        raise KeyError(key)
    return float(m.group(1))


def lookup(artifact: str, key: str):
    path = ROOT / artifact
    if path.suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if key.startswith("has:"):          # phrase must appear verbatim in the file
            phrase = key[4:]
            return phrase if phrase in text else None
        if "/" in key:                      # a literal like "21/21" is a substring claim
            return key if key in text else None
        return summary_value(text, key)
    if path.suffix == ".jsonl":
        return dig(jsonl_last(artifact), key)
    data = json.loads(path.read_text(encoding="utf-8"))
    return dig(data, key) if isinstance(data, dict) else data


def csv_lookup(artifact: str, key: str):
    # key format: row:<model value>:<column>
    _, model, column = key.split(":")
    for row in csv.DictReader((ROOT / artifact).open(encoding="utf-8")):
        if row.get("model") == model:
            return float(row[column])
    raise KeyError(f"{model}:{column} not in {artifact}")


def main() -> int:
    failures = []
    for label, artifact, key, expected in CLAIMS:
        try:
            actual = csv_lookup(artifact, key) if artifact.endswith(".csv") else lookup(artifact, key)
        except Exception as exc:  # noqa: BLE001 - report which artifact is missing, then keep checking
            failures.append(f"MISSING  {label}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(expected, str) or actual is None:
            if str(actual) != str(expected):
                failures.append(f"STALE    {label}: docs say {expected}, artifact says {actual}")
        elif abs(float(actual) - float(expected)) > 5e-4:
            failures.append(f"STALE    {label}: docs say {expected}, artifact says {actual}")
    print(f"claims checked: {len(CLAIMS)}")

    # A number quoted in prose must exist somewhere in the artifacts, or it is a typo we invented.
    blob = "\n".join((ROOT / f).read_text(encoding="utf-8") for f in DOC_FILES if (ROOT / f).is_file())
    # Evidence corpus: experiment logs, result CSVs, the split manifest, and the executed notebooks
    # themselves - a number printed by a cell is an artifact, a number typed into prose is not.
    sources = (list(LOGS.rglob("*.jsonl")) + list(LOGS.rglob("*.csv")) + list(LOGS.rglob("*.md"))
               + list((ROOT / "DL_PBL/notebooks").rglob("*.ipynb"))
               + list((ROOT / "DL_PBL/tests").glob("*.py"))
               + list(LOGS.glob("test_*_summary.txt"))
               + [ROOT / "DL_PBL/data/splits/manifest.json"])
    artifact_text = ""
    for p in sources:
        if p.is_file():
            artifact_text += p.read_text(encoding="utf-8", errors="replace")
    quoted = {n for n in json_number_like(blob) if not acceptable(n, artifact_text)}
    for n in sorted(quoted)[:15]:
        failures.append(f"UNSUPPORTED number {n} appears in docs but not in any artifact or rounded value")
    for line in failures:
        print(line)
    if failures:
        return 1
    print("every documented number is backed by an artifact")
    return 0


def json_number_like(text: str):
    """Numbers in prose that look like measured results."""
    import re
    return re.findall(r"\b0\.\d{4}\b|\b\d+\.\d{3}\b", text)


def acceptable(num: str, artifact_text: str) -> bool:
    if num in artifact_text:
        return True
    val = float(num)
    # accept a documented value that is the artifact value rounded to the quoted precision
    for found in set(json_number_like(artifact_text)):
        try:
            if abs(val - float(found)) < 10 ** -len(num.split(".")[1]):
                return True
        except ValueError:
            continue
    return False


if __name__ == "__main__":
    raise SystemExit(main())
