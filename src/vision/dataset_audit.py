"""Audit the iWildCam v2.0 camera-trap labels before any model is trained.

Reads only the label files (metadata.csv, categories.csv), so it runs in seconds on CPU and needs
none of the 12 GB of images. Everything it prints goes into docs/DATASET_CARD.md, which the Phase I
rubric marks for source, licence, balance, preprocessing, split strategy and leakage checks.

Usage:
    .venv/Scripts/python.exe DL_PBL/src/vision/dataset_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "raw" / "iwildcam_meta" / "iwildcam_v2.0"
LOGS = ROOT / "outputs" / "logs"
OFFICIAL_SPLITS = ["train", "val", "id_val", "id_test", "test"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(META / "metadata.csv", index_col=0, parse_dates=["datetime"])
    cats = pd.read_csv(META / "categories.csv")
    return meta, cats


def leakage_checks(meta: pd.DataFrame) -> dict:
    """Where the official splits overlap, and whether duplicates could inflate a metric.

    The main trap in camera-trap data is burst photography: consecutive frames of one animal visit
    are near-duplicates, so any split that lets a sequence or a camera-day appear on both sides of a
    boundary leaks. iWildCam v2.0 claims its splits do not overlap on date; these numbers are the
    evidence for or against that claim.
    """
    meta = meta.assign(date=meta.datetime.dt.date.astype(str))
    out: dict = {}
    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
        ("train", "id_test"),
        ("id_val", "id_test"),
    ]
    for column in ("location", "seq_id"):
        res = {}
        for a, b in pairs:
            left = set(meta.loc[meta.split == a, column])
            right = set(meta.loc[meta.split == b, column])
            res[f"{a}_vs_{b}"] = len(left & right)
        out[f"{column}_overlap"] = res

    # Camera-day groups: (location, date). Two images of the same animal at the same camera on the
    # same day must never straddle a split boundary.
    day_sets = {
        s: set(meta.loc[meta.split == s, ["location", "date"]].itertuples(index=False, name=None))
        for s in OFFICIAL_SPLITS
    }
    day_overlap = {}
    for a, b in pairs:
        day_overlap[f"{a}_vs_{b}"] = len(day_sets[a] & day_sets[b])
    out["camera_day_overlap"] = day_overlap
    out["duplicate_filenames"] = int(meta.filename.duplicated().sum())
    out["duplicate_image_ids"] = int(meta.image_id.duplicated().sum())
    out["null_labels"] = {
        s: int(meta.loc[meta.split == s, "y"].isna().sum()) for s in OFFICIAL_SPLITS
    }
    return out


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    meta, cats = load()
    name_of = dict(zip(cats.category_id, cats.name))
    meta["class_name"] = meta.category_id.map(name_of)

    per_split = meta.groupby("split").agg(
        images=("image_id", "size"),
        locations=("location", "nunique"),
        sequences=("seq_id", "nunique"),
        classes=("category_id", "nunique"),
        first_seen=("datetime", "min"),
        last_seen=("datetime", "max"),
    ).reindex(OFFICIAL_SPLITS)

    counts = (
        meta.groupby(["category_id", "class_name"])
        .size()
        .rename("images")
        .reset_index()
        .sort_values("images", ascending=False)
    )
    counts["share"] = (counts.images / counts.images.sum()).round(4)
    train_counts = counts[counts.category_id.isin(meta.loc[meta.split == "train", "category_id"])]

    frames_per_seq = meta.groupby("seq_id").size()
    empty_share = float((meta.y == 0).mean())
    day_groups = meta.groupby(["location", meta.datetime.dt.date]).size()

    audit = {
        "images": int(len(meta)),
        "categories_in_file": int(len(cats)),
        "categories_present": int(meta.category_id.nunique()),
        "empty_class_share": round(empty_share, 4),
        "locations": int(meta.location.nunique()),
        "sequences": int(meta.seq_id.nunique()),
        "frames_per_seq": {
            "min": int(frames_per_seq.min()),
            "median": float(frames_per_seq.median()),
            "p90": float(frames_per_seq.quantile(0.90)),
            "max": int(frames_per_seq.max()),
        },
        "camera_day_groups": int(len(day_groups)),
        "camera_day_group_size": {
            "median": float(day_groups.median()),
            "p95": float(day_groups.quantile(0.95)),
            "max": int(day_groups.max()),
        },
        "per_split": json.loads(per_split.to_json(orient="index", date_format="iso")),
        "class_imbalance": {
            "most": f"{counts.iloc[0].class_name} ({counts.iloc[0].images})",
            "rarest_with_>=100": (
                f"{train_counts[train_counts.images >= 100].iloc[-1].class_name} "
                f"({train_counts[train_counts.images >= 100].iloc[-1].images})"
            ),
            "classes_with_lt_100_images": int((counts.images < 100).sum()),
            "classes_with_0_train_images": int(len(cats) - len(train_counts)),
        },
        "leakage": leakage_checks(meta),
    }

    counts.to_csv(LOGS / "class_counts.csv", index=False)
    per_split.to_csv(LOGS / "per_split_stats.csv")
    (LOGS / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# Dataset audit — iWildCam v2.0 (generated by src/vision/dataset_audit.py)",
        "",
        f"- Images: **{audit['images']:,}** across {audit['locations']} camera locations and "
        f"{audit['sequences']:,} sequences.",
        f"- Species categories in the taxonomy: {audit['categories_in_file']} "
        f"(**{audit['categories_present']}** actually appear in these files); class `empty` is label 0.",
        f"- **Empty frames: {audit['empty_class_share']*100:.1f}%** — a third of the corpus carries no "
        "animal, so 'alert fired on an empty frame' is a measurable failure mode, not a hypothetical.",
        f"- Frames per sequence: median {audit['frames_per_seq']['median']:.0f}, "
        f"p90 {audit['frames_per_seq']['p90']:.0f}, max {audit['frames_per_seq']['max']} "
        "(burst photography — the reason splits must group sequences).",
        f"- Camera-day groups (location x date): {audit['camera_day_groups']:,}, median size "
        f"{audit['camera_day_group_size']['median']:.0f}, p95 "
        f"{audit['camera_day_group_size']['p95']:.0f}, max {audit['camera_day_group_size']['max']}.",
        f"- Date range: {audit['per_split']['train']['first_seen'][:10]} to "
        f"{audit['per_split']['train']['last_seen'][:10]}.",
        f"- Imbalance: most common {audit['class_imbalance']['most']}; smallest class still worth "
        f"training on {audit['class_imbalance']['rarest_with_>=100']}; "
        f"{audit['class_imbalance']['classes_with_lt_100_images']} categories have under 100 images; "
        f"{audit['class_imbalance']['classes_with_0_train_images']} taxonomy entries never appear in train.",
        "",
        "## Splits",
        "",
        "```",
        per_split.to_string(),
        "```",
        "",
        "## Leakage checks (overlap counts across split pairs)",
        "",
        "```json",
        json.dumps(audit["leakage"], indent=2),
        "```",
        "",
        "Interpretation: `location_overlap` of 0 between train/val/test means the held-out test set is "
        "made of unseen camera sites, so a score on it is an out-of-distribution claim, not a "
        "same-camera claim. `seq_id_overlap` of 0 means no animal visit is split across both sides.",
        "",
        "## Still to verify against the image files",
        "",
        "- Resolution: the annotation JSON declares 1920x1080 while RELEASE_v2.0.txt states all images "
        "were resized to height 448. Measure real files before quoting a number in the report.",
        "- Licence: the iWildCam v2.0 mirror carries no licence file. Trace it to the WILDS/Oxford "
        "WildCam release and record the exact terms here before submission.",
        "- This Kaggle mirror contains labels for every split including `test`; the official WILDS test "
        "set is unlabelled. State which one is used, because it changes what the final number means.",
    ]
    (LOGS / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
