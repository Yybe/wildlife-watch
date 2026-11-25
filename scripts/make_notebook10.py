"""Generate notebooks/10_results_showcase.ipynb.

Why a separate notebook instead of adding cells to 07: notebook 07's plots come from a 30-epoch training run,
and appending cells there means re-running both arms (~3 h of GPU) to regenerate the file. Everything here is
derived from artifacts that already exist - the per-epoch history CSVs and the saved arm A checkpoint - so it
executes in minutes and can be re-run any time.

It answers three things the deliverable was missing: a validation-accuracy curve, ROC/AUC for the classifier,
and a figure showing real camera-trap frames with the species the model reported and the conflict tier each one
carries. Validation *loss* per epoch is not recoverable retroactively - the training loop never logged it - so
this notebook reports val loss at the selected checkpoint and says so out loud rather than inventing a curve.
"""
import json
from pathlib import Path

MD = {}
CODE = {}

MD["title"] = """# 10 — Results showcase: curves, ROC, and what the model actually sees

**The one to open first.** Every figure here is rebuilt from artifacts that already exist in this repo — the
per-epoch history CSVs written during training and the saved arm A checkpoint — so nothing is retrained and
nothing is asserted that is not plotted from a file.

Contents: (1) training and validation curves for both arms, (2) ROC curves and AUC for the species classifier on
held-out camera sites, (3) real camera-trap frames with the model's reported species, confidence and danger tier,
(4) a conflict-alert decision for one camera site on one day.

One limitation stated up front: the training loop logged `train_loss`, `val_macro_f1` and `val_accuracy` per
epoch but **not** `val_loss`, so a validation-loss curve cannot be drawn for the finished run. The value at the
selected checkpoint is computed below, and `src/vision/wildlife.py` now logs `val_loss` per epoch for any future
run."""

CODE["setup"] = '''import json, os, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.metrics import roc_curve, auc, roc_auc_score

ROOT = next(a for a in [Path.cwd(), *Path.cwd().parents] if (a / "data/splits").exists())
sys.path.insert(0, str(ROOT / "src/vision"))
sys.path.insert(0, str(ROOT / "src/sequence"))
import alert as A
import wildlife as W

IMG = ROOT / "data/raw/iwildcam_v2.0/train"
SP = ROOT / "data/splits"
CKPT = ROOT / "outputs/checkpoints/finetune_reg.pt"
OUT = Path(os.environ.get("P11_OUT", ROOT / "outputs/logs/notebook10"))
OUT.mkdir(parents=True, exist_ok=True)
W.set_seed(W.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")

class_map = pd.read_csv(SP / "class_map.csv").set_index("label")["class_name"]
N = len(class_map)
species_of = {i: str(class_map[i]) for i in range(N)}
tier_of = {i: A.DANGER_TIER.get(species_of[i], A.DEFAULT_TIER) for i in range(N)}
test = W.on_disk(pd.read_csv(SP / "test.csv"), IMG)
print(json.dumps({"checkpoint": str(CKPT), "exists": CKPT.is_file(), "test_images": len(test),
                  "test_sites": int(test.location.nunique()), "device": str(DEVICE)}, indent=2))'''

MD["curves"] = """## 1. Training and validation behaviour, from the CSVs the training loop wrote

Both arms, 30 epochs each, no early stopping. The dashed line marks the epoch each arm's checkpoint was
selected on validation — the model that is scored everywhere else in this project is that epoch, not the last
one."""

CODE["curves"] = '''hist = {}
for arm, fname in (("finetune_reg (arm A)", "finetune_reg_history.csv"),
                   ("head_only_aug (arm B)", "head_only_aug_history.csv")):
    hist[arm] = pd.read_csv(ROOT / "outputs/logs/notebook07" / fname)
best = {arm: int(h.val_macro_f1.idxmax()) + 1 for arm, h in hist.items()}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, col, ttl in ((axes[0], "train_loss", "Training loss per epoch"),
                     (axes[1], "val_macro_f1", "Validation macro F1 per epoch"),
                     (axes[2], "val_accuracy", "Validation accuracy per epoch")):
    for (arm, h), mk in zip(hist.items(), ["o-", "s--"]):
        ax.plot(h.epoch, h[col], mk, label=arm, ms=3)
    ax.axvline(best["finetune_reg (arm A)"], color="crimson", ls=":", lw=1, label="arm A selected epoch")
    ax.set_xlabel("epoch"); ax.set_title(ttl); ax.grid(alpha=.3)
axes[0].set_ylabel("cross-entropy"); axes[1].set_ylabel("macro F1"); axes[2].set_ylabel("accuracy")
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "training_and_validation_curves.png", dpi=130)
print("selected epochs:", best, "| arm A final val acc", round(float(hist["finetune_reg (arm A)"].val_accuracy.iloc[-1]), 4))'''

CODE["predict"] = '''model = W.load_resnet50(None, n_classes=N, pretrained=False)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.to(DEVICE).eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
probs, ys, files = [], [], []
with torch.no_grad():
    for x, y in DataLoader(W.WildlifeImages(test, IMG, tf), batch_size=256, num_workers=4):
        probs.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy()); ys.append(y.numpy())
P, y = np.concatenate(probs), np.concatenate(ys)
pred = P.argmax(1)
acc = float((pred == y).mean())
m = W.metrics(y, pred, np.arange(N))
val_tbl = W.on_disk(pd.read_csv(SP / "val.csv"), IMG)
vp, vy = [], []
with torch.no_grad():
    for x, yy in DataLoader(W.WildlifeImages(val_tbl, IMG, tf), batch_size=256, num_workers=4):
        vp.append(model(x.to(DEVICE)).cpu().numpy()); vy.append(yy.numpy())
val_loss = float(torch.nn.functional.cross_entropy(torch.tensor(np.concatenate(vp)),
                                                   torch.tensor(np.concatenate(vy))))
print("validation loss at the selected checkpoint:", round(val_loss, 4),
      "over", len(np.concatenate(vy)), "val images")
print(json.dumps({"accuracy": round(acc, 4), "macro_f1_all13": round(m["macro_f1"], 4),
                  "n_evaluated": int(len(y))}, indent=2))'''

MD["roc"] = """## 2. ROC and AUC on held-out camera sites

One-vs-rest ROC per class, plus the micro-average over all classes. AUC is the rank-quality view of the same
predictions macro F1 scores threshold-wise: it answers "can the model rank the right species above the others"
independent of the argmax cut-off. Classes with fewer than 20 test images are still drawn, but their AUC is
read with that sample size in mind."""

CODE["roc"] = '''y_bin = np.zeros((len(y), N), dtype=int)
y_bin[np.arange(len(y)), y] = 1
present = [l for l in range(N) if 0 < y_bin[:, l].sum() < len(y)]
fpr, tpr, roc_auc = {}, {}, {}
for l in present:
    fpr[l], tpr[l], _ = roc_curve(y_bin[:, l], P[:, l])
    roc_auc[l] = float(roc_auc_score(y_bin[:, l], P[:, l]))
micro_fpr = np.unique(np.concatenate([fpr[l] for l in present]))
mean_tpr = np.zeros_like(micro_fpr)
for l in present:
    mean_tpr += np.interp(micro_fpr, fpr[l], tpr[l])
mean_tpr /= len(present)
micro_auc = float(auc(micro_fpr, mean_tpr))
macro_auc = float(np.mean([roc_auc[l] for l in present]))

fig, ax = plt.subplots(figsize=(7.2, 6.4))
for l in sorted(present, key=lambda k: -roc_auc[k]):
    ax.plot(fpr[l], tpr[l], lw=1.2, label=f"{species_of[l][:26]} ({roc_auc[l]:.3f})")
ax.plot(micro_fpr, mean_tpr, "k-", lw=2.2, label=f"micro-average ({micro_auc:.3f})")
ax.plot([0, 1], [0, 1], ":", c="grey", label="chance")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title(f"ROC per species, held-out camera sites\\nmicro AUC {micro_auc:.4f} / macro AUC {macro_auc:.4f}")
ax.legend(fontsize=6.5, loc="lower right"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "roc_curves.png", dpi=130)

per_class = pd.DataFrame({"class": [species_of[l] for l in present],
                          "tier": [tier_of[l] for l in present],
                          "test_images": [int(y_bin[:, l].sum()) for l in present],
                          "auc": [round(roc_auc[l], 4) for l in present],
                          "f1": [round(m["per_class_f1"].get(l, 0.0), 4) for l in present]}
                         ).sort_values("auc", ascending=False)
per_class.to_csv(OUT / "roc_per_class.csv", index=False)
print(per_class.to_string(index=False))
print(json.dumps({"micro_auc": round(micro_auc, 4), "macro_auc": round(macro_auc, 4),
                  "accuracy": round(acc, 4), "classes_in_roc": len(present)}, indent=2))'''

MD["showcase"] = """## 3. Real frames, with what the model reported

This is the figure the problem statement asks for in picture form: camera-trap images, the species named on
each, the confidence behind that name, and the danger tier that drives the alert. Correct and incorrect
predictions are both shown — a showcase of only successes would not survive a viva.

Imagery: Snapshot Safari / Oxford WildCam camera traps, CC BY 4.0; release used iWildCam v2.0 (WILDS),
CDLA-Permissive-1.0; credit Beery, Cole & Gjoka (2020)."""

CODE["showcase"] = '''tbl = test.iloc[:len(pred)].copy()
tbl["pred_lab"], tbl["conf"], tbl["correct"] = pred, P.max(1), pred == y
rng = np.random.default_rng(W.SEED)
picks = []
for l in [i for i in range(N) if i != 0]:                       # one confident correct frame per animal class
    cand = tbl[(tbl.label == l) & tbl.correct]
    if len(cand):
        picks.append(cand.sample(min(1, len(cand)), random_state=int(rng.integers(1e6))).iloc[0])
for l in [i for i in range(N) if i != 0]:                       # the two most confident mistakes overall
    cand = tbl[(tbl.label == l) & ~tbl.correct]
    if len(cand):
        picks.append(cand.nlargest(1, "conf").iloc[0])
picks.append(tbl[tbl.label == 0].nlargest(1, "conf").iloc[0])   # an empty frame called as an animal
show = pd.DataFrame(picks).drop_duplicates(subset=["filename"])[:16]

fig, axes = plt.subplots(3, 6, figsize=(16, 8.6))
for ax in axes.ravel():
    ax.axis("off")
for ax, (_, r) in zip(axes.ravel(), show.iterrows()):
    try:
        with Image.open(IMG / r.filename) as im:
            ax.imshow(np.asarray(im.convert("RGB")))
    except Exception:
        continue
    mark = "OK" if r.correct else "WRONG"
    col = "#1a7f37" if r.correct else "#b3261e"
    ax.set_title(f"{species_of[int(r.label)]}\\n-> {species_of[int(r.pred_lab)]}  {r.conf:.2f}  [{mark}]\\n"
                 f"danger tier {tier_of[int(r.pred_lab)]}", fontsize=7, color=col)
    ax.axis("off")
fig.suptitle("Camera-trap frames with the model's reported species, confidence and conflict tier", fontsize=12)
fig.text(0.5, 0.005, "Imagery: Snapshot Safari / Oxford WildCam, CC BY 4.0. Release: iWildCam v2.0 (WILDS), "
                     "CDLA-Permissive-1.0. Credit: Beery, Cole & Gjoka (2020).",
         ha="center", fontsize=7, color="#444444")
plt.tight_layout(rect=(0, 0.03, 1, 0.97)); plt.savefig(OUT / "prediction_showcase.png", dpi=120)
print("frames shown:", len(show), "| correct:", int(show.correct.sum()), "| wrong:", int((~show.correct).sum()))'''

MD["alert"] = """## 4. From a species name to a conflict decision

The last step the problem statement asks for. One camera site's test-day timeline: each animal visit plotted at
its time with its danger tier, the risk score the rule assigns across the day, and whether that crosses the
alert threshold. The tier-weighted, recency-decayed score is defined in `src/sequence/alert.py`."""

CODE["alert"] = '''ev = A.visits_from_manifest(tbl, confidence=tbl.conf.to_numpy(), predicted=tbl.pred_lab.to_numpy())
truth = A.visits_from_manifest(tbl)
cfg = A.AlertConfig()
site_days = A.build_site_days(ev, species_of, cfg, step_hours=1.0)
if site_days:
    top = max(site_days, key=lambda sd: sd.score)
    day = pd.Timestamp(top.day)
    here = ev[(ev.location == top.site) & (pd.to_datetime(ev.start).dt.normalize() == day)]
    hours = pd.date_range(day, day + pd.Timedelta(days=1), freq="1h")[:-1]
    scores = [A.score_site_day([(pd.Timestamp(r.start), A.DANGER_TIER.get(species_of.get(int(r.label_used), "?"),
                                                     A.DEFAULT_TIER), float(r.conf))
                                for _, r in here.iterrows()], h, cfg) for h in hours]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(hours, scores, "o-", ms=3, label="risk score across the day")
    ax.axhline(cfg.score_threshold, color="crimson", ls="--", label=f"alert threshold {cfg.score_threshold}")
    for _, r in here.iterrows():
        t = A.DANGER_TIER.get(species_of.get(int(r.label_used), "?"), A.DEFAULT_TIER)
        ax.axvline(pd.Timestamp(r.start), color={4: "black", 3: "darkred", 2: "darkorange"}.get(t, "grey"),
                   alpha=.45, lw=1.4)
    ax.set_ylim(0, 1.05); ax.set_ylabel("risk score")
    ax.set_title(f"Site {top.site} on {day.date()}: {len(here)} visits, peak score "
                 f"{max(scores):.3f} -> {'ALERT' if top.alert else 'no alert'}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "alert_decision_example.png", dpi=130)
    print(f"site {top.site} {day.date()} visits {len(here)} peak {max(scores):.3f} "
          f"rule score {top.score:.3f} alerted {top.alert} qualifying {top.qualifying}")
else:
    print("no site-day reached the scoring stage on this split")'''

CODE["summary"] = '''summary = {"accuracy": round(acc, 4), "macro_f1_all13": round(m["macro_f1"], 4),
           "micro_auc": round(micro_auc, 4), "macro_auc": round(macro_auc, 4),
           "val_loss_at_selected_checkpoint": round(val_loss, 4) if val_loss is not None else None,
           "selected_epoch_armA": best["finetune_reg (arm A)"], "epochs_run_armA": int(hist["finetune_reg (arm A)"].epoch.max()),
           "figures": ["training_and_validation_curves.png", "roc_curves.png", "prediction_showcase.png",
                       "alert_decision_example.png"]}
(OUT / "experiment_log.jsonl").write_text(json.dumps({"run_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                                      "experiment": "notebook10_results_showcase", **summary}) + chr(10),
                                          encoding="utf-8")
print(json.dumps(summary, indent=2))'''


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in t.splitlines()]}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in t.splitlines()]}


def main() -> None:
    cells = [md(MD["title"]), code(CODE["setup"]), md(MD["curves"]), code(CODE["curves"]),
             code(CODE["predict"]), md(MD["roc"]), code(CODE["roc"]),
             md(MD["showcase"]), code(CODE["showcase"]), md(MD["alert"]), code(CODE["alert"]),
             code(CODE["summary"])]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11.9"}},
          "nbformat": 4, "nbformat_minor": 5}
    dest = Path(__file__).resolve().parents[1] / "notebooks/10_results_showcase.ipynb"
    dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.loads(dest.read_text(encoding="utf-8"))
    print(f"wrote {dest}: {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code), JSON valid")


if __name__ == "__main__":
    main()
