"""Live demo server: camera-trap frame in, species + confidence + conflict-alert decision out.

Wraps the project's own artifacts without re-implementing them:
  * classifier  = outputs/checkpoints/finetune_reg.pt  (fine-tuned ResNet-50, arm A)
  * features    = src/vision/wildlife.py               (same eval transform as training)
  * alert rule  = src/sequence/alert.py                (same score_site_day as notebook 08)

Run:
    .venv/Scripts/python.exe DL_PBL/demo/server.py
then open http://127.0.0.1:8765
"""
from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = next(a for a in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (a / "data/splits").exists())
sys.path.insert(0, str(ROOT / "src/vision"))
sys.path.insert(0, str(ROOT / "src/sequence"))

import alert as A
import wildlife as W

IMG_ROOT = ROOT / "data/raw/iwildcam_v2.0/train"
CKPT = ROOT / "outputs/checkpoints/finetune_reg.pt"
DEMO_DIR = Path(__file__).resolve().parent
MAX_BYTES = 25 * 2**20
MAX_REPLAY_FRAMES = 160

COMMON = {
    "empty": "No animal", "tayassu pecari": "White-lipped peccary",
    "dasyprocta punctata": "Central American agouti", "puma concolor": "Puma / cougar",
    "leopardus pardalis": "Ocelot", "panthera onca": "Jaguar", "bos taurus": "Cattle",
    "aepyceros melampus": "Impala", "cephalophus nigrifrons": "Black-fronted duiker",
    "urocyon cinereoargenteus": "Gray fox", "meleagris ocellata": "Ocellated turkey",
    "crax rubra": "Great curassow", "mazama temama": "Red brocket deer",
}

app = Flask(__name__)
LOCK = threading.Lock()

import torch
from PIL import Image
from torchvision import transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_TF = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

CLASS_MAP = pd.read_csv(ROOT / "data/splits/class_map.csv")
NAMES = CLASS_MAP.sort_values("label").class_name.tolist()
TEST = pd.read_csv(ROOT / "data/splits/test.csv", parse_dates=["datetime"])
TEST["day"] = TEST.datetime.dt.strftime("%Y-%m-%d")
ALLOWED = set(TEST.filename)

MODEL = W.load_resnet50(n_classes=len(NAMES), pretrained=False)
MODEL.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
MODEL.to(DEVICE).eval()


def tier_of(species: str) -> int:
    return A.DANGER_TIER.get(species, A.DEFAULT_TIER)


def pick_samples() -> list[dict]:
    """One representative frame per class: the middle frame of its longest test sequence."""
    out = []
    for label, grp in TEST.groupby("label"):
        sizes = grp.groupby("seq_id").size()
        seq = sizes.idxmax()
        rows = grp[grp.seq_id == seq].sort_values("datetime")
        r = rows.iloc[len(rows) // 2]
        out.append({"name": r.filename, "label": int(label), "species": r.class_name,
                    "common": COMMON.get(r.class_name, r.class_name),
                    "site": int(r.location), "day": r.day, "seq": r.seq_id,
                    "burst_n": int(sizes[seq]),
                    "datetime": r.datetime.isoformat(sep=" ")})
    return sorted(out, key=lambda s: s["label"])


SAMPLES = pick_samples()


def day_candidates() -> list[dict]:
    """Test site-days with enough frames to replay and real big-cat evidence on the ground."""
    rows = []
    for (site, day), grp in TEST.groupby(["location", "day"]):
        tiers = grp.class_name.map(tier_of)
        q = int((tiers >= 3).sum())
        if 20 <= len(grp) <= MAX_REPLAY_FRAMES and q >= 4:
            rows.append({"site": int(site), "day": day, "frames": len(grp),
                         "qualifying_frames": q,
                         "species": sorted(set(grp[tiers > 0].class_name))})
    return sorted(rows, key=lambda r: -r["qualifying_frames"])[:6]


DAYS = day_candidates()


@torch.no_grad()
def infer(images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.time()
    x = torch.stack([EVAL_TF(im.convert("RGB")) for im in images]).to(DEVICE)
    with LOCK:
        proba = torch.softmax(MODEL(x), dim=1).cpu().numpy()
    return proba.argmax(1), proba, (time.time() - t0) * 1000


def prediction_json(proba: np.ndarray, ms: float) -> dict:
    order = np.argsort(-proba)
    top = [{"species": NAMES[i], "common": COMMON.get(NAMES[i], NAMES[i]),
            "p": round(float(proba[i]), 4)} for i in order[:5]]
    best = int(order[0])
    return {"species": NAMES[best], "common": COMMON.get(NAMES[best], NAMES[best]),
            "confidence": round(float(proba[best]), 4), "danger_tier": tier_of(NAMES[best]),
            "top5": top, "ms": round(ms, 1), "device": str(DEVICE)}


@app.get("/")
def index():
    return send_from_directory(DEMO_DIR, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify({
        "classes": [{"species": s, "common": COMMON.get(s, s), "tier": tier_of(s)} for s in NAMES],
        "default_cfg": vars(A.AlertConfig()),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "checkpoint": CKPT.name,
        "test_stats": {"images": len(TEST), "sites": int(TEST.location.nunique()),
                       "macro_f1": 0.6403, "accuracy": 0.7751},
        "samples": SAMPLES, "days": DAYS,
    })


@app.get("/api/image/<name>")
def image(name):
    if name not in ALLOWED:
        return jsonify({"error": "image not in the held-out test split"}), 404
    return send_file(IMG_ROOT / name, mimetype="image/jpeg")


@app.post("/api/predict")
def predict():
    if "file" in request.files:
        raw = request.files["file"].read()
        if len(raw) > MAX_BYTES:
            return jsonify({"error": f"file too large (max {MAX_BYTES // 2**20} MB)"}), 400
        try:
            im = Image.open(io.BytesIO(raw))
            im.verify()
            im = Image.open(io.BytesIO(raw))
        except Exception:
            return jsonify({"error": "not a decodable image file"}), 400
        meta_in = {"site": "LIVE", "day": pd.Timestamp.now().strftime("%Y-%m-%d"),
                   "datetime": pd.Timestamp.now().isoformat(sep=" ", timespec="seconds"),
                   "seq": "upload-" + str(int(time.time() * 1000)), "burst_n": 1, "truth": None}
    else:
        name = (request.get_json(silent=True) or {}).get("name")
        if name not in ALLOWED:
            return jsonify({"error": "unknown sample"}), 400
        im = Image.open(IMG_ROOT / name)
        row = TEST[TEST.filename == name].iloc[0]
        meta_in = {"site": int(row.location), "day": row.day, "seq": row.seq_id,
                   "burst_n": int((TEST.seq_id == row.seq_id).sum()),
                   "datetime": row.datetime.isoformat(sep=" "), "truth": row.class_name}
    label, proba, ms = infer([im])
    out = prediction_json(proba[0], ms)
    out["meta"] = meta_in
    return jsonify(out)


@app.post("/api/burst")
def burst():
    """Classify every frame of one camera burst and collapse it to a single visit,
    exactly as alert.visits_from_manifest does: modal species, max confidence."""
    name = (request.get_json(silent=True) or {}).get("name")
    if name not in ALLOWED:
        return jsonify({"error": "unknown sample"}), 400
    row = TEST[TEST.filename == name].iloc[0]
    grp = TEST[TEST.seq_id == row.seq_id].sort_values("datetime").head(8)
    images = [Image.open(IMG_ROOT / n) for n in grp.filename]
    preds, proba, ms = infer(images)
    modal = int(pd.Series(preds).mode().iloc[0])
    conf = float(proba[np.arange(len(preds)), preds].max())
    out = prediction_json(proba.mean(0), ms)
    out.update({"species": NAMES[modal], "common": COMMON.get(NAMES[modal], NAMES[modal]),
                "confidence": round(conf, 4), "danger_tier": tier_of(NAMES[modal]),
                "n_frames": len(grp)})
    out["meta"] = {"site": int(row.location), "day": row.day, "seq": row.seq_id,
                   "burst_n": len(grp), "datetime": grp.datetime.iloc[0].isoformat(sep=" "),
                   "truth": grp.class_name.mode().iloc[0]}
    return jsonify(out)


@app.post("/api/score")
def score():
    body = request.get_json(force=True)
    cfg = A.AlertConfig(**{**vars(A.AlertConfig()), **body.get("cfg", {})})
    visits = [(pd.Timestamp(v["t"]), int(v["tier"]), float(v["conf"])) for v in body["visits"]]
    if not visits:
        return jsonify({"score": 0.0, "alert": False, "qualifying": 0, "at": None})
    at = max(t for t, _, _ in visits)
    s = A.score_site_day(visits, at, cfg)
    qualifying = sum(1 for _, tier, _ in visits if tier >= cfg.tier_min)
    return jsonify({"score": round(s, 4),
                    "alert": bool(s >= cfg.score_threshold and qualifying >= cfg.min_visits),
                    "qualifying": qualifying, "at": at.isoformat(sep=" ")})


@app.post("/api/replay")
def replay():
    body = request.get_json(force=True)
    cfg = A.AlertConfig(**{**vars(A.AlertConfig()), **body.get("cfg", {})})
    grp = TEST[(TEST.location == body["site"]) & (TEST.day == body["day"])].sort_values("datetime")
    if grp.empty:
        return jsonify({"error": "no such site-day in the test split"}), 400
    if len(grp) > MAX_REPLAY_FRAMES:
        grp = grp.iloc[np.linspace(0, len(grp) - 1, MAX_REPLAY_FRAMES).astype(int)]
    images = [Image.open(IMG_ROOT / n) for n in grp.filename]
    t0 = time.time()
    preds, proba, _ = infer(images)
    conf = proba[np.arange(len(preds)), preds]
    best = grp.assign(pred=preds, conf=conf).groupby("seq_id")["conf"].idxmax()
    rep_name = dict(zip(grp.loc[best, "seq_id"], grp.loc[best, "filename"]))
    rep_truth = dict(zip(grp.loc[best, "seq_id"], grp.loc[best, "class_name"]))
    visits_df = A.visits_from_manifest(grp, confidence=conf, predicted=preds)
    species_of = dict(enumerate(NAMES))
    rows = [(pd.Timestamp(r.start), tier_of(species_of[int(r.label_used)]), float(r.conf))
            for r in visits_df.itertuples()]
    day0 = pd.Timestamp(body["day"])
    grid = [{"h": h, "score": round(A.score_site_day([r for r in rows if r[0] <= day0 + pd.Timedelta(hours=h)],
                                                      day0 + pd.Timedelta(hours=h), cfg), 4)}
            for h in range(25)]
    peak = max(g["score"] for g in grid)
    qualifying = sum(1 for _, tier, _ in rows if tier >= cfg.tier_min)
    truth_q = bool((grp.class_name.map(tier_of) >= cfg.tier_min).any())
    return jsonify({
        "site": body["site"], "day": body["day"], "n_frames": len(grp),
        "seconds": round(time.time() - t0, 2),
        "visits": [{"start": pd.Timestamp(r.start).isoformat(sep=" "),
                    "name": rep_name.get(r.seq_id), "truth": rep_truth.get(r.seq_id),
                    "species": species_of[int(r.label_used)],
                    "common": COMMON.get(species_of[int(r.label_used)], "?"),
                    "tier": tier_of(species_of[int(r.label_used)]),
                    "conf": round(float(r.conf), 3), "frames": int(r.frames)}
                   for r in visits_df.itertuples()],
        "grid": grid, "peak": peak,
        "alert": bool(peak >= cfg.score_threshold and qualifying >= cfg.min_visits),
        "qualifying": qualifying, "truth_qualifying": truth_q,
    })


if __name__ == "__main__":
    with torch.no_grad():
        MODEL(torch.zeros(1, 3, 224, 224, device=DEVICE))
    print(f"model ready on {DEVICE} | {len(SAMPLES)} samples | {len(DAYS)} replay days", flush=True)
    app.run(host="127.0.0.1", port=8765, debug=False)