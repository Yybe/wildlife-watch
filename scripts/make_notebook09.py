"""Generate notebook 09: error analysis of the species classifier and the alert stream.

Runs on whichever predictor is available: the fine-tuned checkpoint from notebook 07 if present,
otherwise the cached feature bank plus saved head. The banner it prints states which, because the two
pipelines are not interchangeable.

Run:  .venv/Scripts/python.exe DL_PBL/scripts/make_notebook09.py
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.splitlines(keepends=True)})


md("""# 09 — Error analysis: what the model gets wrong, where, and how much it costs the alert

A macro F1 number is not an analysis. This notebook answers the questions the viva asks: which species are
confused with which, does the model fail more at night or at unseen camera sites, how confident is it when
it is wrong, and which specific errors produce missed alerts versus false alerts.

Predictor used: the notebook 07 checkpoint when it exists, otherwise the cached frozen-feature bank plus
the saved head — the code prints which, since the two are different models and the numbers are not
transferable between them.""")

code('''import json, os, sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = next(a for a in [Path.cwd(), *Path.cwd().parents] if (a / "data/splits").exists())
sys.path.insert(0, str(ROOT / "src/vision"))
sys.path.insert(0, str(ROOT / "src/sequence"))
import alert as A
import wildlife as W

IMG = ROOT / "data/raw/iwildcam_v2.0/train"
SP = ROOT / "data/splits"
BANK = ROOT / "data/processed/feature_bank.npz"
HEAD = ROOT / "outputs/checkpoints/species_head.joblib"
CKPT = ROOT / "outputs/checkpoints/finetune_reg.pt"
OUT = Path(os.environ.get("P11_OUT", ROOT / "outputs/logs/notebook09"))
OUT.mkdir(parents=True, exist_ok=True)
W.set_seed(W.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")

class_map = pd.read_csv(SP / "class_map.csv").set_index("label")["class_name"]
N = len(class_map)
species_of = {i: str(class_map[i]) for i in range(N)}
test = W.on_disk(pd.read_csv(SP / "test.csv"), IMG)
print("test images:", len(test), "| sites:", test.location.nunique(),
      "| camera-days:", len(test.groupby(["location", test.datetime.astype(str).str[:10]])),
      "| device:", DEVICE, "| predictor:", "finetuned checkpoint" if CKPT.is_file() else "bank + head")''')

code('''if CKPT.is_file():
    model = W.load_resnet50(None, n_classes=N, pretrained=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = model.to(DEVICE).eval()
    probs, ys = [], []
    with torch.no_grad():
        # shuffle=False keeps batch order, so the concatenated arrays still line up with the manifest
        for x, yy in DataLoader(W.WildlifeImages(test, IMG, tf), batch_size=256, num_workers=4):
            probs.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
            ys.append(yy.numpy())
    P, y = np.concatenate(probs), np.concatenate(ys)
    source = "finetuned_checkpoint"
else:
    z = np.load(BANK)
    bundle = joblib.load(HEAD)
    P = bundle["pipeline"].predict_proba(bundle["scaler"].transform(z["test_x"].astype(np.float32)))
    y = z["test_y"]
    source = "bank_frozen_features_logreg"
print("predictions:", P.shape, "| source:", source, "| accuracy:", round(float((P.argmax(1) == y).mean()), 4))
pred = P.argmax(1)
conf = P.max(1)
runner_up = np.sort(P, 1)[:, -2]
margin = conf - runner_up''')

md("""## 1. Per-class accuracy: who is being missed

Support and recall side by side, because a rare class can look fine on average and still be invisible.""")

code('''rows = []
for l in range(N):
    m = y == l
    rows.append({"class": species_of[l], "tier": A.DANGER_TIER.get(species_of[l], A.DEFAULT_TIER),
                 "test_images": int(m.sum()), "recall": round(float((pred[m] == l).mean()) if m.sum() else 0.0, 4),
                 "missed_as_empty": int((pred[m] == 0).sum()) if m.sum() else 0,
                 "mean_conf_when_wrong": round(float(conf[m & (pred != l)].mean()) if (m & (pred != l)).any() else float("nan"), 3)})
per_class = pd.DataFrame(rows).sort_values("recall")
per_class.to_csv(OUT / "per_class_errors.csv", index=False)
per_class.to_string(index=False)''')

md("""## 2. Confusion pairs

Aggregating the matrix into ordered pairs makes the biology visible: the errors that matter are big-cat
against big-cat and animal-against-empty, because those are exactly the alert decisions.""")

code('''err = pred != y
pairs = (pd.DataFrame({"true": pd.Series(y[err]).map(species_of), "pred": pd.Series(pred[err]).map(species_of)})
         .groupby(["true", "pred"], observed=True).size().reset_index(name="count")
         .sort_values("count", ascending=False))
pairs["alert_relevant"] = np.where(
    pairs["true"].map(lambda s: A.DANGER_TIER.get(s, 0) >= 3) |
    pairs["pred"].map(lambda s: A.DANGER_TIER.get(s, 0) >= 3), "yes", "no")
print(f"{err.sum()} errors of {len(y)} images ({err.mean():.1%})")
pairs.head(15).to_string(index=False)
pairs.to_csv(OUT / "confusion_pairs.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4))
top = pairs.head(10)
ax.barh([f"{r['true']} -> {r['pred']}" for _, r in top.iterrows()][::-1], top["count"][::-1])
ax.set_xlabel("errors"); ax.set_title("Top 10 confusion pairs on held-out camera sites")
plt.tight_layout(); plt.savefig(OUT / "confusion_pairs.png", dpi=120)''')

md("""## 3. Where errors cluster: site, night, and burst position

Errors are not independent. A camera site with a bad angle or a species it rarely sees can dominate the
whole test score, which is why per-site spread is reported next to the average.""")

code('''t = test.iloc[:len(pred)].copy()
t["pred"], t["conf"], t["margin"], t["error"] = pred, conf, margin, err
t["hour"] = pd.to_datetime(t.datetime).dt.hour
t["night"] = t.hour >= 18
t["correct"] = ~err
by_hour = t.groupby("hour").agg(n=("correct", "size"), acc=("correct", "mean"))
by_hour["night"] = by_hour.index >= 18
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(by_hour.index, by_hour.acc, color=np.where(by_hour.night, "steelblue", "lightsteelblue"))
ax[0].set_xlabel("hour of day"); ax[0].set_ylabel("accuracy"); ax[0].set_title("Accuracy by hour (blue = 18:00-23:00)")
site = t.groupby("location").agg(n=("correct", "size"), acc=("correct", "mean")).query("n >= 50")
ax[1].hist(site.acc, bins=20, color="dimgray")
ax[1].axvline(t.correct.mean(), color="crimson", ls="--", label="overall")
ax[1].set_xlabel("per-site accuracy"); ax[1].set_ylabel("camera sites")
ax[1].set_title(f"Site spread (n={len(site)} sites with >=50 images)"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "error_clusters.png", dpi=120)
print("day (00-17) accuracy", round(float(t[~t.night].correct.mean()), 4),
      "| night (18-23) accuracy", round(float(t[t.night].correct.mean()), 4))
print("worst 5 sites:", site.nsmallest(5, "acc").round(3).to_dict()["acc"])
print("site accuracy range:", round(float(site.acc.min()), 3), "to", round(float(site.acc.max()), 3),
      "| std", round(float(site.acc.std()), 3))''')

md("""## 4. Confidence when wrong, and calibration

A confident wrong answer is what makes a false alert. If confidence is well calibrated, a threshold can
filter it; if the model is confidently wrong in bursts, no threshold saves it — and that is the measured
finding behind the gate's residual false alerts.""")

code('''def ece(conf, correct, bins=10):
    idx = np.clip((conf * bins).astype(int), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)

bins = pd.cut(t.conf, np.linspace(0, 1, 6), include_lowest=True)
rel = t.groupby(bins, observed=True).agg(n=("correct", "size"), predicted=("conf", "mean"),
                                         actual=("correct", "mean"))
rel["gap"] = (rel.actual - rel.predicted).round(4)
print(rel.to_string())
print("expected calibration error:", round(ece(t.conf.to_numpy(), t.correct.to_numpy()), 4))
print("mean margin on errors:", round(float(t.margin[t.error].mean()), 4),
      "| on correct:", round(float(t.margin[~t.error].mean()), 4))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(rel.predicted, rel.actual, "o-", label="model")
ax[0].plot([0, 1], [0, 1], ":", label="perfect")
ax[0].set_xlabel("predicted confidence"); ax[0].set_ylabel("observed accuracy"); ax[0].legend()
ax[0].set_title("Reliability diagram")
ax[1].hist(t.margin[~t.error], bins=30, alpha=0.6, label="correct", density=True)
ax[1].hist(t.margin[t.error], bins=30, alpha=0.6, label="wrong", density=True)
ax[1].set_xlabel("top1 - top2 probability"); ax[1].legend(); ax[1].set_title("Decision margin")
plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=120)
rel.to_csv(OUT / "reliability.csv")''')

md("""## 5. Cost to the alert stream

The same predictions feed the alert rule, so each model error becomes either a missed alert or a false
alert. This is the bridge between the classifier result and the product claim, and it is where the
residual false alerts in notebook 04 come from.""")

code('''tbl = test.iloc[:len(pred)].copy()
evidence = A.visits_from_manifest(tbl, confidence=conf, predicted=pred)
truth = A.visits_from_manifest(tbl)
# visits_from_manifest returns location/seq_id/frames/label_used/conf only; the tier and species the
# merge below reads belong to the visit stream, so derive them here rather than assuming the module
# exports the forecast module's columns.
for _v in (evidence, truth):
    _v["species"] = _v["label_used"].map(species_of)
    _v["tier"] = _v["species"].map(lambda s: A.DANGER_TIER.get(s, A.DEFAULT_TIER))
cfg = A.AlertConfig()
res = A.evaluate_per_frame_vs_gated(evidence, truth, species_of, cfg)
print(json.dumps(res, indent=2))

# Which species the model's mistakes turn into alerts, and which real incidents it drops.
fp = evidence.merge(truth[["location", "seq_id", "tier", "species"]], on=["location", "seq_id"],
                   suffixes=("_pred", "_true"))
bad = fp[(fp.tier_pred >= cfg.tier_min) & (fp.tier_true < cfg.tier_min)]
missed = fp[(fp.tier_true >= cfg.tier_min) & (fp.tier_pred < cfg.tier_min)]
print("visits alerted that really were not dangerous:", len(bad))
print(bad.species_pred.value_counts().to_frame("predicted").join(
    bad.species_true.value_counts().rename("truth").to_frame(), how="left").fillna(0).astype(int).to_string())
print("dangerous visits the model did not flag:", len(missed))
print(missed.groupby(["species_true", "species_pred"]).size().sort_values(ascending=False).head(8).to_string())
bad.to_csv(OUT / "alert_false_visits.csv", index=False)
missed.to_csv(OUT / "alert_missed_visits.csv", index=False)''')

code('''fig, axes = plt.subplots(2, 3, figsize=(13, 8))
show = pd.concat([t[t.error].groupby(t[t.error].pred).head(1),
                  t[(t.pred == 0) & (t.label != 0)].head(2)])
for ax, (_, r) in zip(axes.ravel(), show.iterrows()):
    with Image.open(IMG / r.filename) as im:
        ax.imshow(np.asarray(im.convert("RGB")))
    ax.set_title(f"true {species_of[int(r.label)]}\\npredicted {species_of[int(r.pred)]} "
                 f"conf {r.conf:.2f}", fontsize=8)
    ax.axis("off")
for ax in axes.ravel()[len(show):]:
    ax.axis("off")
plt.suptitle("Representative errors (one per failure mode shown)")
fig.text(0.5, 0.01, "Imagery: Snapshot Safari / Oxford WildCam camera traps, CC BY 4.0. Release used: "
                    "iWildCam v2.0 (WILDS), CDLA-Permissive-1.0. Credit: Beery, Cole & Gjoka (2020).",
         ha="center", fontsize=7, color="#444444")
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(OUT / "error_examples.png", dpi=120)
print("saved", OUT / "error_examples.png")''')

md("""## 6. What this changes for the next step

Read the outputs before writing conclusions. The decisions this analysis is designed to inform: whether a
confidence threshold on the margin distribution is worth adding to the alert rule, whether the residual
false alerts are a classifier problem (confident multi-visit confusion) or a rule problem, and which
species need more training data before the conflict claim can be made for them at all.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
dest = Path("DL_PBL/notebooks/09_error_analysis.ipynb")
dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
json.loads(dest.read_text(encoding="utf-8"))
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        try:
            compile("".join(c["source"]), f"cell{i}", "exec")
        except SyntaxError as exc:
            raise SystemExit(f"cell {i} does not compile: {exc}")
print(f"wrote {dest}: {len(cells)} cells, all code cells compile")
