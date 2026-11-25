"""Generate notebooks without hand-writing JSON (shell quoting eats notebook sources otherwise).

Run:  .venv/Scripts/python.exe DL_PBL/scripts/make_notebook05.py
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.splitlines(keepends=True)})


md("""# 05 — Detection: localise the animal, measure mAP and IoU

Project 11 requires mAP and IoU, and the species corpus (iWildCam v2.0) carries **no bounding boxes at all**, so localisation trains on a second camera-trap dataset: `ankanghosh651/object-detection-wildlife-dataset-yolo-format` (Kaggle, license reported by the API as MIT, 471 MB, 4 classes: buffalo, elephant, rhino, zebra). The corpus switch is declared in `docs/SOURCE_LOG.md` row 2.

Elephant and buffalo are the species that actually damage crops and endanger people, so this is the conflict half of the product, not an unrelated dataset. The second architecture the compendium asks for (Faster R-CNN on the same labels) follows in notebook 06; Phase I reports one detector with honest numbers rather than two half-working ones.""")

code('''import json, os, time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

ROOT = next(a for a in [Path.cwd(), *Path.cwd().parents] if (a / "data/splits").exists())
DETECT = ROOT / "data/raw/_yolo/final_data"
OUT = Path(os.environ.get("P11_OUT", ROOT / "outputs/logs/notebook05"))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 13
EPOCHS = int(os.environ.get("P11_EPOCHS", 50))
IMGSZ = int(os.environ.get("P11_IMGSZ", 640))
BATCH = int(os.environ.get("P11_BATCH", 8))
assert DETECT.is_dir(), f"missing detection dataset at {DETECT}"
print(json.dumps({"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                  "cuda": torch.cuda.is_available(),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "dataset": str(DETECT), "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                  "out": str(OUT)}, indent=2))''')

md("""## 1. Audit before training

Images and label files must pair up, and the class distribution decides whether mAP is meaningful. This set has no `empty` class, which is exactly why the false-alert metric cannot be measured here - that metric stays on the species corpus.""")

code('''CLASSES = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}
splits = {}
for s in ("train", "valid", "test"):
    imgs = sorted((DETECT / s / "images").glob("*.*"))
    lbls = sorted((DETECT / s / "labels").glob("*.txt"))
    counts = Counter()
    boxes = []
    bad_lines = 0
    for p in lbls:
        for line in p.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                bad_lines += 1
                continue
            counts[CLASSES[int(parts[0])]] += 1
            boxes.append((float(parts[3]), float(parts[4])))
    splits[s] = {"images": len(imgs), "labels": len(lbls),
                 "unpaired_stems": len({p.stem for p in imgs} ^ {p.stem for p in lbls}),
                 "malformed_lines": bad_lines, "boxes": int(sum(counts.values())),
                 "per_class": dict(counts)}
    if s == "train" and boxes:
        bw, bh = zip(*boxes)
        smallest = min(counts, key=counts.get)
        print(f"train boxes: median width {np.median(bw):.3f}, median height {np.median(bh):.3f} "
              f"(normalised fractions); smallest class {smallest} with {counts[smallest]} boxes")
pd.DataFrame(splits).T.drop(columns=["per_class"])''')

code('''train_counts = pd.Series(splits["train"]["per_class"])
ax = train_counts.plot.bar(figsize=(6, 3), rot=0, title="Instances per class (train split)")
ax.set_ylabel("boxes")
plt.tight_layout()
print(train_counts.to_string())''')

code('''yaml_path = DETECT / "wildlife_local.yaml"
names_block = "".join(f"  {i}: {n}" + chr(10) for i, n in CLASSES.items())
yaml_path.write_text("path: " + DETECT.as_posix() + chr(10)
                     + "train: train/images" + chr(10) + "val: valid/images" + chr(10)
                     + "test: test/images" + chr(10) + "names:" + chr(10) + names_block,
                     encoding="utf-8")
print(yaml_path.read_text(encoding="utf-8"))''')

md("""## 1b. Duplicate audit and de-leaked split lists

Byte-level hashing found this dataset leaking: some `valid`/`test` images are identical to `train`
images. Measuring mAP on those copies measures memorisation, not detection, so the audit is reported and
the evaluation is repeated on deduplicated lists (a copy is kept only in the earliest split, and a
16x16 average hash flags near-duplicates that a byte hash misses).""")

code('''import hashlib

SPLIT_ORDER = ("train", "valid", "test")


def ahash(path, size=16):
    from PIL import Image
    gray = Image.open(path).convert("L").resize((size, size))
    px = np.asarray(gray, dtype=np.float32)
    bits = (px > px.mean()).flatten()
    value = 0
    for b in bits.astype(np.uint8):
        value = (value << 1) | int(b)
    return value


def hamming(a, b):
    return bin(a ^ b).count("1")


exact, near = {}, {}
for s in SPLIT_ORDER:
    for p in sorted((DETECT / s / "images").glob("*.*")):
        exact.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append((s, p.name))
        near.setdefault(ahash(p), []).append((s, p.name))

dup_groups = [v for v in exact.values() if len(v) > 1]
cross = [(k, v) for k, v in exact.items() if len({s for s, _ in v}) > 1]
print(f"byte-identical groups: {len(dup_groups)} (extra copies {sum(len(v) - 1 for v in dup_groups)})")
print(f"groups spanning more than one split: {len(cross)}")
for k, v in cross[:6]:
    print("   ", v)

near_cross = 0
keys = sorted(near)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        if hamming(a, b) <= 4 and {s for s, _ in near[a]} != {s for s, _ in near[b]}:
            near_cross += 1
print(f"near-duplicate (hamming<=4) pairs across splits: {near_cross}")

keep = {}
for h, items in exact.items():
    for j, key in enumerate(items):
        keep[key] = (j == 0)
lists = {s: [] for s in SPLIT_ORDER}
dropped = Counter()
for (s, name), is_first in keep.items():
    (lists[s].append(name) if is_first else dropped.update([s]))
for s in SPLIT_ORDER:
    print(f"{s}: {len(lists[s])} kept, {dropped[s]} duplicate copies dropped")

# Ultralytics resolves list files against stale label caches, so the de-leaked split is materialised as
# real folders with their label files, and stale caches are removed before anything trains on it.
import shutil
DEDUP_ROOT = DETECT.parent / "final_data_dedup"
if DEDUP_ROOT.exists():
    shutil.rmtree(DEDUP_ROOT)
for s in SPLIT_ORDER:
    for sub in ("images", "labels"):
        (DEDUP_ROOT / s / sub).mkdir(parents=True, exist_ok=True)
    for name in lists[s]:
        shutil.copy2(DETECT / s / "images" / name, DEDUP_ROOT / s / "images" / name)
        stem = Path(name).stem
        lbl = DETECT / s / "labels" / f"{stem}.txt"
        if lbl.is_file():
            shutil.copy2(lbl, DEDUP_ROOT / s / "labels" / f"{stem}.txt")
for cache in DETECT.rglob("*.cache"):
    cache.unlink()

after = {}
for s in SPLIT_ORDER:
    for p in sorted((DEDUP_ROOT / s / "images").glob("*.*")):
        after.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), set()).add(s)
still_cross = sum(1 for v in after.values() if len(v) > 1)
total_kept = sum(len(lists[s]) for s in SPLIT_ORDER)
print("images in deduplicated tree:", sum(len(list((DEDUP_ROOT / s / 'images').glob('*.*'))) for s in SPLIT_ORDER))
assert total_kept == len(exact), (total_kept, len(exact))
assert still_cross == 0, f"{still_cross} duplicate groups still span splits"
assert all((DEDUP_ROOT / s / 'labels' / f'{Path(n).stem}.txt').exists() for s in SPLIT_ORDER for n in lists[s]), \
    "an image was copied without its label file"
print("verification: every unique image kept exactly once, no split overlap, every label present")

dedup_yaml = DEDUP_ROOT / "wildlife_local_dedup.yaml"
dedup_yaml.write_text("path: " + DEDUP_ROOT.as_posix() + chr(10)
                      + "train: train/images" + chr(10) + "val: valid/images" + chr(10)
                      + "test: test/images" + chr(10) + "names:" + chr(10) + names_block,
                      encoding="utf-8")
print(dedup_yaml.read_text(encoding="utf-8"))
print("note: only byte-identical copies are removed; the near-duplicate count above is reported, not filtered")''')

md("""## 2. Train YOLOv8n

Nano backbone at 640 px, batch 8 - the configuration a 6 GB laptop GPU can actually run, which is the deployment constraint the product claims to meet. COCO-pretrained weights are fine-tuned here and declared in the source log.""")

code('''t0 = time.time()
CKPT = os.environ.get("P11_CKPT", "")
if CKPT and Path(CKPT).is_file():
    model = YOLO(CKPT)
    train_sec = None
    print("loaded trained checkpoint " + CKPT + " - training skipped for this pass")
else:
    model = YOLO("yolov8n.pt")
    res = model.train(data=str(dedup_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
                      project=str(OUT.parent), name="yolo_runs_dedup", exist_ok=True, plots=True,
                      verbose=False)
    train_sec = round(time.time() - t0, 1)
print("train seconds", train_sec, "| trained on deduplicated train list")''')

code('''metrics = model.val(data=str(yaml_path), split="test", imgsz=IMGSZ, batch=BATCH, verbose=False)
summary = {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
           "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)}
metrics_dd = model.val(data=str(dedup_yaml), split="test", imgsz=IMGSZ, batch=BATCH, verbose=False)
dedup_summary = {"mAP50": float(metrics_dd.box.map50), "mAP50_95": float(metrics_dd.box.map),
                 "precision": float(metrics_dd.box.mp), "recall": float(metrics_dd.box.mr)}
print(json.dumps({"test_as_shipped": {k: round(v, 4) for k, v in summary.items()},
                  "test_deduplicated": {k: round(v, 4) for k, v in dedup_summary.items()},
                  "delta_from_removing_duplicates": {k: round(dedup_summary[k] - summary[k], 4)
                                                     for k in summary}}, indent=2))
ap_index = getattr(metrics_dd.box, "ap_class_index", None)
idx = list(range(len(CLASSES))) if ap_index is None or len(ap_index) == 0 else [int(i) for i in ap_index]
per_class = pd.DataFrame({"class": [CLASSES[i] for i in idx],
                          "AP50_dedup": [round(float(metrics_dd.box.ap50[i]), 4) for i in idx],
                          "AP50_95_dedup": [round(float(metrics_dd.box.ap[i]), 4) for i in idx],
                          "AP50_as_shipped": [round(float(metrics.box.ap50[i]), 4) for i in idx]})
per_class''')

md("""## 3. Latency and memory — the edge constraint

Measured after warm-up on the GPU that trained the model, with peak memory from the CUDA allocator. This is a proxy for field power use and is labelled as a proxy: no camera-trap board was available.""")

code('''imgs = sorted((DETECT / "test" / "images").glob("*.*"))[:100]
_ = model.predict(str(imgs[0]), imgsz=IMGSZ, verbose=False)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
dets = 0
for p in imgs:
    r = model.predict(str(p), imgsz=IMGSZ, conf=0.25, verbose=False)[0]
    dets += len(r.boxes)
lat_ms = round((time.time() - t0) / len(imgs) * 1000, 1)
peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None
print(json.dumps({"images_timed": len(imgs), "ms_per_frame": lat_ms, "fps": round(1000 / lat_ms, 2),
                  "peak_gpu_mb": peak_mb, "avg_detections_per_frame": round(dets / len(imgs), 2)}, indent=2))''')

code('''fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, p in zip(axes.ravel(), imgs[:4]):
    r = model.predict(str(p), imgsz=IMGSZ, conf=0.35, verbose=False)[0]
    ax.imshow(np.asarray(r.orig_img[..., ::-1]))
    labels = ", ".join(f"{CLASSES[int(b.cls)]} {float(b.conf):.2f}" for b in r.boxes) or "no detection"
    ax.set_title(labels)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "detections_examples.png", dpi=110)
print("saved", OUT / "detections_examples.png")''')

code('''run = {"run_utc": pd.Timestamp.now(tz="UTC").isoformat(), "experiment": "notebook05_yolov8n_detection",
       "seed": SEED, "dataset": "ankanghosh651/object-detection-wildlife-dataset-yolo-format",
       "classes": list(CLASSES.values()),
       "split_audit": {s: {k: v for k, v in d.items() if k != "per_class"} for s, d in splits.items()},
       "model": "yolov8n.pt (COCO pretrained, fine-tuned)", "epochs": EPOCHS, "imgsz": IMGSZ,
       "batch": BATCH, "train_sec": train_sec,
       "test": {k: round(v, 4) for k, v in summary.items()},
       "test_deduplicated": {k: round(v, 4) for k, v in dedup_summary.items()},
       "per_class": per_class.to_dict("records"),
       "ms_per_frame": lat_ms, "fps": round(1000 / lat_ms, 2), "peak_gpu_mb": peak_mb,
       "ultralytics": ultralytics.__version__,
       "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
with open(OUT / "experiment_log.jsonl", "a", encoding="utf-8") as fh:
    fh.write(json.dumps(run) + chr(10))
print(json.dumps({k: run[k] for k in ("test", "ms_per_frame", "peak_gpu_mb")}, indent=2))''')

md("""## 4. What these numbers do and do not support

* **The shipped split leaks.** Byte hashing found exact copies of train images inside `valid` and `test` (see 1b), so the as-shipped mAP is measured partly on memorised images. Both numbers are printed; the deduplicated one is the one to quote.
* mAP and IoU are measured on a **different corpus** from the species classifier (4 African species, no empty frames). Any sentence that merges the two must say so: alert metrics come from iWildCam, detection metrics from here.
* ~1.2k train images and 4 classes is small, so per-class AP matters more than the mean and the smallest class (rhino, 445 boxes) drives the error bars.
* Latency and peak memory are the edge-deployment proxy. Real power draw on a camera-trap board was not measurable here and is not claimed.
* Next for the compendium's two-architecture benchmark: torchvision Faster R-CNN (ResNet-FPN) on the same labels after converting YOLO txt to COCO, at equal epochs, the same seed, and the same deduplicated lists.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
dest = Path("DL_PBL/notebooks/05_detection_yolo.ipynb")
dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
json.loads(dest.read_text(encoding="utf-8"))
print(f"wrote {dest} with {len(cells)} cells; JSON valid")
