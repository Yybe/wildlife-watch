"""Diagnose why notebook 08's decision points came back empty. Run:
    .venv/Scripts/python.exe DL_PBL/scripts/diag_forecast.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DL_PBL/src/vision"))
sys.path.insert(0, str(ROOT / "DL_PBL/src/sequence"))
import alert as A
import forecast as F
import wildlife as W

SP = ROOT / "DL_PBL/data/splits"
class_map = pd.read_csv(SP / "class_map.csv").set_index("label")["class_name"]
species_of = {int(l): str(class_map[l]) for l in range(len(class_map))}
tbl = pd.read_csv(SP / "train.csv").head(4000)
print("tbl columns:", list(tbl.columns))
ev = F.visits(tbl, species_of, A.DANGER_TIER, A.DEFAULT_TIER)
print("events:", len(ev), list(ev.columns))
print(ev.head(3).to_string())
cfg = F.ForecastConfig()
prior, g = F.priors(ev, cfg)
print("prior sites:", len(prior), "global:", round(g, 4))
pts = F.site_day_windows(ev, cfg, prior, g)
print("decision points:", len(pts), "columns:", list(pts.columns)[:6], "...")
if len(pts):
    print(pts.groupby("label").size().to_dict())
    print("positive rate:", round(float(pts.label.mean()), 4))
