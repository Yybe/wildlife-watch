"""Independent check of notebook 07's arm A: recompute the test metrics and the alert gate from the
saved checkpoint, so a stale number inside a notebook cannot survive.

Run:  .venv/Scripts/python.exe DL_PBL/scripts/check_armA_gate.py
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DL_PBL/src/vision"))
sys.path.insert(0, str(ROOT / "DL_PBL/src/sequence"))
import alert as A
import wildlife as W

IMG = ROOT / "DL_PBL/data/raw/iwildcam_v2.0/train"
SP = ROOT / "DL_PBL/data/splits"
CKPT = ROOT / "DL_PBL/outputs/checkpoints/finetune_reg.pt"
BANK = ROOT / "DL_PBL/data/processed/feature_bank.npz"
HEAD = ROOT / "DL_PBL/outputs/checkpoints/species_head.joblib"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128, help="eval batch size; the notebook used 32")
    ap.add_argument("--json-out", default="", help="write the record here as well as stdout")
    a = ap.parse_args()
    class_map = pd.read_csv(SP / "class_map.csv").set_index("label")["class_name"]
    N = len(class_map)
    species_of = {i: str(class_map[i]) for i in range(N)}
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    te = W.on_disk(pd.read_csv(SP / "test.csv"), IMG, verbose=False)

    model = W.load_resnet50(None, n_classes=N, pretrained=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    model = model.to(device).eval()
    probs, ys = [], []
    with torch.no_grad():
        # num_workers=0: this file has no child-process safety at import time, and Windows spawn
        # re-executes the module for every worker.
        for x, y in DataLoader(W.WildlifeImages(te, IMG, tf), batch_size=a.batch, num_workers=0):
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
            ys.append(y.numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    pred, conf = probs.argmax(1), probs.max(1)

    gate = A.evaluate_per_frame_vs_gated(A.visits_from_manifest(te.iloc[:len(pred)], confidence=conf,
                                                                predicted=pred),
                                         A.visits_from_manifest(te.iloc[:len(pred)]),
                                         species_of, A.AlertConfig())
    macro = round(W.metrics(ys, pred, np.arange(N))["macro_f1"], 4)
    print("arm A recomputed: test macro F1", macro)
    print("  gated   ", json.dumps(gate["gated"]))
    print("  perframe", json.dumps(gate["per_frame"]))

    bundle = joblib.load(HEAD)
    lp = bundle["pipeline"].predict_proba(bundle["scaler"].transform(
        np.load(BANK)["test_x"].astype(np.float32))).argmax(1)
    agree = round(float((lp == pred).mean()), 4)
    print("agreement armA vs saved logreg head:", agree, "over", len(pred), "images")
    record = {"armA_test_macro_f1": macro, "gated": gate["gated"], "per_frame": gate["per_frame"],
              "agreement_vs_logreg": agree, "n_test": int(len(pred)), "device": str(device),
              "eval_batch": a.batch, "checkpoint": str(CKPT),
              "checkpoint_mtime_utc": pd.Timestamp(CKPT.stat().st_mtime, unit="s", tz="UTC").isoformat()}
    print(json.dumps(record))
    if a.json_out:
        out = Path(a.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + chr(10))
    return 0


if __name__ == "__main__":
    sys.exit(main())
