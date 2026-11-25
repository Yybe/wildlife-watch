"""Fetch a stratified sample of camera-trap images for local development.

Only the images referenced by the manifests are pulled, one file at a time, and the download is
resume-safe: a file already on disk is skipped, so an interrupted run continues where it stopped.
This exists because the full corpus is 12 GB and the local link moves ~0.5 MB/s — a sample is enough
to execute every notebook end to end, while real training runs on Kaggle with the dataset attached.

Usage:
    .venv/Scripts/python.exe DL_PBL/src/vision/fetch_images.py --train 240 --val 90 --test 90
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "iwildcam_v2.0" / "train"
LOGS = ROOT / "outputs" / "logs"
REF = "thanhquang71/iwildcam-v2-0-2020-wilds-dataset"
IN_DATASET_DIR = "iwildcam_v2.0/train"


def pick(manifest_dir: str, split: str, n: int, seed: int) -> pd.DataFrame:
    """Take n images spread over the classes so a smoke test touches every head."""
    df = pd.read_csv(Path(manifest_dir) / f"{split}.csv")
    per_class = max(1, n // df.label.nunique())
    df = df.sample(frac=1.0, random_state=seed)
    df["_rank"] = df.groupby("label").cumcount()
    return df[df["_rank"] < per_class].drop(columns="_rank").head(n).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--val", type=int, default=90)
    ap.add_argument("--test", type=int, default=90)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--manifests", default=str(ROOT / "data" / "splits"))
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    rows, skipped, failed = [], 0, 0
    staging = RAW / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", a.train), ("val", a.val), ("test", a.test)):
        for _, r in pick(a.manifests, split, n, a.seed).iterrows():
            dest = RAW / r.filename
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
            else:
                try:
                    # Kaggle recreates the in-dataset folder tree under `path`, so land it in staging
                    # and flatten to RAW/<filename>; without this the resume check never matches.
                    api.dataset_download_file(REF, f"{IN_DATASET_DIR}/{r.filename}", path=str(staging), quiet=True)
                    landed = staging / IN_DATASET_DIR / r.filename
                    if landed.exists():
                        landed.replace(dest)
                    else:
                        found = list(staging.rglob(r.filename))
                        if not found:
                            raise FileNotFoundError(f"{r.filename} not found after download")
                        found[0].replace(dest)
                except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                    failed += 1
                    print(f"FAILED {r.filename}: {type(exc).__name__}: {str(exc)[:120]}")
                    continue
            rows.append({"split": split, "label": int(r.label), "class_name": r.class_name,
                         "seq_id": r.seq_id, "location": int(r.location), "path": str(dest)})
        print(f"{split}: {sum(1 for x in rows if x['split'] == split)} ready, "
              f"{skipped} pre-existing, {failed} failed")

    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "sample_manifest.csv", index=False)
    print(f"sample written: {len(df)} rows -> {LOGS / 'sample_manifest.csv'}; images in {RAW}")


if __name__ == "__main__":
    main()
