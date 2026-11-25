"""Command-line conflict-alert prototype: image in, species + confidence + risk decision out.

Phase I's product evidence is a working prototype with input validation, a prediction and a
confidence/interpretation output. The classifier here is the frozen ResNet-50 + head from
`notebooks/04`; the risk half is the designed rule in `src/sequence/alert.py`, so the same scoring code
serves both the product and the ablation.

Usage:
    .venv/Scripts/python.exe DL_PBL/src/app/predict.py --image some.jpg
    .venv/Scripts/python.exe DL_PBL/src/app/predict.py --image some.jpg --site 12 --history visits.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(a for a in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (a / "data/splits").exists())
sys.path.insert(0, str(ROOT / "src/vision"))
sys.path.insert(0, str(ROOT / "src/sequence"))

MAX_BYTES = 25 * 2**20
HEAD_PATH = ROOT / "outputs/checkpoints/species_head.joblib"


def validate_image(path: str) -> Path:
    """Trust boundary: reject anything that is not a readable, plausible image file."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"not a file: {path}")
    size = p.stat().st_size
    if size < 100:
        raise ValueError(f"suspiciously small file ({size} bytes): {path}")
    if size > MAX_BYTES:
        raise ValueError(f"file too large ({size} bytes, max {MAX_BYTES}): {path}")
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        raise ValueError(f"unsupported image extension: {p.suffix}")
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(p) as im:
            im.verify()
    except (UnidentifiedImageError, SyntaxError, OSError) as exc:
        raise ValueError(f"not a decodable image ({type(exc).__name__}): {path}") from None
    return p


def load_head(device):
    import joblib
    import wildlife as W
    from torchvision import transforms as T

    if not HEAD_PATH.is_file():
        raise FileNotFoundError(
            f"{HEAD_PATH} missing - fit and save the head from notebooks/04 first")
    bundle = joblib.load(HEAD_PATH)
    encoder = W.load_resnet50(bundle["weights_path"], n_classes=0, pretrained=True).to(device).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return encoder, tf, bundle


def predict_species(encoder, tf, bundle, path: Path, device) -> dict:
    import torch
    from PIL import Image

    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = encoder(x).cpu().numpy()
    proba = bundle["pipeline"].predict_proba(bundle["scaler"].transform(feat))[0]
    label = int(proba.argmax())
    names = bundle["class_names"]
    species = names[label]
    import alert as A
    return {"species": species, "label": label, "confidence": round(float(proba[label]), 4),
            "runner_up": f"{names[int(proba.argsort()[-2])]} {float(sorted(proba)[-2]):.3f}",
            "danger_tier": A.DANGER_TIER.get(species, A.DEFAULT_TIER)}


def risk_from_history(result: dict, site: str, history: Path) -> dict:
    import pandas as pd
    import alert as A

    cfg = A.AlertConfig()
    now = pd.Timestamp.now()
    rows = []
    if history.is_file():
        h = pd.read_csv(history, parse_dates=["start"])
        rows = [(pd.Timestamp(r.start), A.DANGER_TIER.get(str(r.species), A.DEFAULT_TIER), float(r.conf))
                for r in h.itertuples() if str(r.site) == str(site)]
    today = [(t, tier, conf) for t, tier, conf in rows if (now - t).days == 0]
    today.append((now, result["danger_tier"], result["confidence"]))
    qualifying = sum(1 for _, tier, _ in today if tier >= cfg.tier_min)
    score = A.score_site_day(today, now, cfg)
    return {"site": site, "score": round(score, 4), "alert": bool(
        score >= cfg.score_threshold and qualifying >= cfg.min_visits),
        "qualifying_visits_today": qualifying, "rule": vars(cfg),
        "note": "min_visits suppresses a lone sighting; see PLAN.md §3 - alerting policy is under revision"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Wildlife conflict alert prototype")
    ap.add_argument("--image", required=True)
    ap.add_argument("--site", default="1")
    ap.add_argument("--history", default=None, help="CSV with columns site,start,species,conf")
    args = ap.parse_args(argv)

    try:
        path = validate_image(args.image)
    except ValueError as exc:
        print(f"input rejected: {exc}", file=sys.stderr)
        return 2

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        encoder, tf, bundle = load_head(device)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    out = predict_species(encoder, tf, bundle, path, device)
    if args.history:
        out["risk"] = risk_from_history(out, args.site, Path(args.history))
    out["device"] = str(device)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
