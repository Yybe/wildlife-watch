"""Build leakage-safe train/val/test manifests for the wildlife conflict project.

Two constraints shape this file.

1. Burst photography. A camera trap fires a run of near-identical frames when an animal visits
   (median 5 frames, up to 10 per sequence in this corpus), so a sequence must land wholly inside one
   split, and subsampling has to pick whole sequences. Random per-image splits leak and inflate every
   number that follows.
2. Image layout. Every split's images sit in ONE folder, `iwildcam_v2.0/train/`, so probing
   `iwildcam_v2.0/val/<uuid>.jpg` returns 404 even though the file exists under `train/`. Verified by
   name after the full corpus was on disk: all 129,809 train, 14,961 val, 42,791 test, 7,314 id_val and
   8,154 id_test filenames are present, plus 14,611 unlabelled files that no manifest row references.
   Default mode therefore keeps the benchmark's own `train`/`val`/`test`, which are disjoint on camera
   site and on date. `--splits recarved` retains the alternative holdouts (whole sites for test,
   camera-days for val) for a robustness check.

Usage:
    .venv/Scripts/python.exe DL_PBL/src/vision/make_splits.py --top-k 12 --max-per-class 3000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "raw" / "iwildcam_meta" / "iwildcam_v2.0"
SPLITS = ROOT / "data" / "splits"
IMAGE_SUBDIR = "iwildcam_v2.0/train"  # the only image directory the mirror serves


def classify_groups(pool: pd.DataFrame, test_site_frac: float, val_day_frac: float, seed: int):
    """Assign every image to train/val/test at sequence and camera-site granularity."""
    rng = np.random.default_rng(seed)
    # Walk camera sites in seeded random order until the target image share is covered. Taking the
    # biggest sites first produced a test set of only a handful of sites, which is both thin as an
    # out-of-distribution claim and badly skewed in species coverage.
    sites = pool.groupby("location").size()
    order = list(sites.index)
    rng.shuffle(order)
    target = test_site_frac * len(pool)
    test_sites, total = [], 0
    for location in order:
        if total >= target:
            break
        test_sites.append(int(location))
        total += int(sites.loc[location])

    in_test = pool.location.isin(test_sites)
    test = pool[in_test]

    rest = pool[~in_test].copy()
    rest["camera_day"] = rest.location.astype(str) + "@" + rest.date
    days = rest.groupby("camera_day").seq_id.agg(lambda s: sorted(set(s)))
    day_ids = list(days.index)
    rng.shuffle(day_ids)
    n_val_days = int(round(val_day_frac * len(day_ids)))
    val_days = set(day_ids[:n_val_days])

    val_mask = rest.camera_day.isin(val_days)
    val = rest[val_mask]
    train = rest[~val_mask]
    return train, val, test, test_sites, len(val_days), len(day_ids)


def cap_by_sequence(part: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Trim each class to max_per_class images by keeping whole sequences."""
    sizes = part.groupby(["label", "seq_id"]).size().rename("n").reset_index()
    chosen = []
    for label, grp in sizes.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=int(seed) + int(label)).reset_index(drop=True)
        take, total = [], 0
        for seq_id, n in zip(grp.seq_id, grp.n):
            if total + n > max_per_class and take:
                break
            take.append(seq_id)
            total += n
        chosen.append(grp[grp.seq_id.isin(take)])
    keep = pd.concat(chosen, ignore_index=True)
    # Join on (label, seq_id): a visit can hold two species, and joining on seq_id alone duplicates rows.
    return part.merge(keep[["label", "seq_id"]], on=["label", "seq_id"], how="inner")


def build(top_k: int, max_per_class: int, seed: int, test_site_frac: float, val_day_frac: float, mode: str = "official") -> dict:
    meta = pd.read_csv(META / "metadata.csv", index_col=0, parse_dates=["datetime"])
    cats = pd.read_csv(META / "categories.csv")
    meta["class_name"] = meta.category_id.map(dict(zip(cats.category_id, cats.name)))
    meta["date"] = meta.datetime.dt.strftime("%Y-%m-%d")

    base = meta if mode == "official" else meta[meta.split == "train"].copy()
    keep_species = (
        base[base.category_id != 0].groupby("category_id").size().nlargest(top_k).index.tolist()
    )
    keep = sorted([0] + keep_species)  # label 0 = `empty`, kept so the alert can abstain
    pool = base[base.category_id.isin(keep)].copy()
    label_map = {cat: idx for idx, cat in enumerate(keep)}
    pool["label"] = pool.category_id.map(label_map)
    names = {idx: str(pool.loc[pool.category_id == cat, "class_name"].iloc[0]) for cat, idx in label_map.items()}

    if mode == "official":
        # The mirror ships images for every official split (all under iwildcam_v2.0/train/), so the
        # benchmark's own train/val/test is used: 243/32/48 sites, disjoint on site and on date.
        parts = {sp: cap_by_sequence(pool[pool.split == sp], max_per_class, seed)
                 for sp in ("train", "val", "test")}
        test_sites, n_val_days, n_days = [], 0, 0
    else:
        train, val, test, test_sites, n_val_days, n_days = classify_groups(
            pool, test_site_frac, val_day_frac, seed)
        parts = {sp: cap_by_sequence(p, max_per_class, seed)
                 for sp, p in (("train", train), ("val", val), ("test", test))}
    train, val, test = (parts[sp] for sp in ("train", "val", "test"))
    out = pd.concat([p.assign(split=sp) for sp, p in parts.items()], ignore_index=True)

    overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ga, gb = parts[a], parts[b]
        overlap[f"{a}_vs_{b}"] = {
            "seq": len(set(ga.seq_id) & set(gb.seq_id)),
            "camera_day": len(
                set(zip(ga.location, ga.date)) & set(zip(gb.location, gb.date))
            ),
            "location": len(set(ga.location) & set(gb.location)),
        }
    checks = {**overlap, "duplicate_filenames": int(out.filename.duplicated().sum())}
    for key in ("train_vs_val", "train_vs_test", "val_vs_test"):
        assert checks[key]["seq"] == 0, f"{key}: a sequence straddles the boundary"
        assert checks[key]["camera_day"] == 0, f"{key}: a camera-day straddles the boundary"
    assert checks["train_vs_test"]["location"] == 0, "test camera sites are not fully held out"
    assert checks["duplicate_filenames"] == 0, "duplicate image filenames in the manifest"

    SPLITS.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "filename", "label", "category_id", "class_name",
            "seq_id", "location", "date", "datetime", "split"]
    for split, part in parts.items():
        part[cols].sort_values(["label", "seq_id", "datetime"]).to_csv(SPLITS / f"{split}.csv", index=False)
    pd.DataFrame([{"label": i, "category_id": c, "class_name": names[i]} for c, i in label_map.items()]
                 ).to_csv(SPLITS / "class_map.csv", index=False)

    manifest = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "species_coverage": {
            "train_classes": int(parts["train"].label.nunique()),
            "test_classes": int(parts["test"].label.nunique()),
            "classes_absent_from_test": [names[i] for i in sorted(set(parts["train"].label) - set(parts["test"].label))],
            "test_classes_under_100_images": [names[i] for i in sorted(parts["test"].label.unique())
                                              if int((parts["test"].label == i).sum()) < 100],
        },
        "source": "thanhquang71/iwildcam-v2-0-2020-wilds-dataset (iWildCam v2.0 metadata.csv + train images)",
        "image_dir_in_dataset": IMAGE_SUBDIR,
        "seed": seed,
        "top_k_species": top_k,
        "max_images_per_class": max_per_class,
        "empty_class_retained": True,
        "mode": mode,
        "image_paths_note": "the mirror stores every split's image under iwildcam_v2.0/train/; "
        "probing iwildcam_v2.0/val/<uuid>.jpg returns 404 and is not evidence of missing images",
        "test_is_out_of_distribution": True,
        "held_out_test_sites": len(test_sites) if mode == "recarved" else "n/a (official split sites)",
        "val_camera_days": f"{n_val_days} of {n_days}" if mode == "recarved" else "n/a (official split)",
        "capping_unit": "sequence (burst-safe)",
        "class_counts": {s: {names[i]: int((p.label == i).sum()) for i in sorted(p.label.unique())}
                         for s, p in parts.items()},
        "split_sizes": {s: int(len(p)) for s, p in parts.items()},
        "split_locations": {s: int(p.location.nunique()) for s, p in parts.items()},
        "split_sequences": {s: int(p.seq_id.nunique()) for s, p in parts.items()},
        "leakage_checks": checks,
        "manifest_sha256_16": {f"{s}.csv": hashlib.sha256((SPLITS / f"{s}.csv").read_bytes()).hexdigest()[:16]
                               for s in parts},
    }
    (SPLITS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=12, help="species kept; the empty class is always included")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--test-site-frac", type=float, default=0.15, help="share of images in held-out camera sites")
    ap.add_argument("--val-day-frac", type=float, default=0.12, help="share of camera-days used for validation")
    ap.add_argument("--splits", choices=["official", "recarved"], default="official",
                    help="official = the benchmark's own train/val/test; recarved = held-out sites and camera-days")
    a = ap.parse_args()
    build(a.top_k, a.max_per_class, a.seed, a.test_site_frac, a.val_day_frac, a.splits)
