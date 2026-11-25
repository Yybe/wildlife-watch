"""Generate notebooks from plain Python source strings.

Hand-writing .ipynb JSON keeps breaking on escaping and nested loops, so every notebook that needs new
code is emitted by a generator like this one and validated before execution.

Run:  .venv/Scripts/python.exe DL_PBL/scripts/make_notebook07.py
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.splitlines(keepends=True)})


md("""# 07 — Longer training with checkpoint selection (30 epochs)

Notebook 02's fine-tune lost to the frozen-feature baseline out of distribution (0.6043 vs 0.6245) while
its training loss kept falling: it overfitted the camera sites it had seen. Two 30-epoch arms test whether
that verdict survives proper training discipline — augmentation, label smoothing, patience-based best
checkpoint restore, and **validation-only** model selection so the test split is scored once at the end.

* **Arm A `finetune_reg`** — ResNet-50 head + `layer4` trainable, strong augmentation, label smoothing 0.1.
* **Arm B `head_only_aug`** — backbone fully frozen, only the head trains, same augmentation. This is the
  cheaper arm and the one that would win if the backbone really is the problem.

Reference points, measured earlier: majority class 0.0201, random 0.0645, frozen features + logistic
regression **0.6245**, notebook 02 fine-tune **0.6043**. A claim of improvement needs to clear 0.6245 by a
margin, on the same seed and split files.""")

code('''import json, os, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = next(a for a in [Path.cwd(), *Path.cwd().parents] if (a / "data/splits").exists())
sys.path.insert(0, str(ROOT / "src/vision"))
sys.path.insert(0, str(ROOT / "src/sequence"))
import wildlife as W
import alert as A

IMG_ROOT = ROOT / "data/raw/iwildcam_v2.0/train"
SPLITS_DIR = ROOT / "data/splits"
WEIGHTS = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"))
OUT = Path(os.environ.get("P11_OUT", ROOT / "outputs/logs/notebook07"))
CKPT = ROOT / "outputs/checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("P11_SEED", W.SEED))
EPOCHS = int(os.environ.get("P11_EPOCHS", 30))
PATIENCE = int(os.environ.get("P11_PATIENCE", 8))
BATCH = int(os.environ.get("P11_BATCH", 32))
W.set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REF = {"majority": 0.0201, "random": 0.0645, "frozen_logreg": 0.6245, "notebook02_finetune": 0.6043}
print(json.dumps({"seed": SEED, "epochs_max": EPOCHS, "patience": PATIENCE, "batch": BATCH,
                  "references": REF, **W.hardware_note()}, indent=2))
assert WEIGHTS.is_file() and IMG_ROOT.is_dir()''')

code('''class_map = pd.read_csv(SPLITS_DIR / "class_map.csv").set_index("label")["class_name"]
N_CLASSES = len(class_map)
LABELS = np.arange(N_CLASSES)
tables = {s: W.on_disk(pd.read_csv(SPLITS_DIR / f"{s}.csv"), IMG_ROOT) for s in ("train", "val", "test")}

# Train-time augmentation is stronger than notebook 02's: wider crop range, blur, and a rotation, which
# are the perturbations a fixed camera at a forest edge actually produces (fog, rain, animals at angles).
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                      T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                      T.RandomGrayscale(p=0.1), T.GaussianBlur(3), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
eval_tf = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dfs = {s: W.WildlifeImages(tables[s], IMG_ROOT, train_tf if s == "train" else eval_tf)
       for s in tables}
loaders = {s: DataLoader(d, batch_size=BATCH, shuffle=(s == "train"), num_workers=4,
                         pin_memory=True, persistent_workers=True) for s, d in dfs.items()}
class_weight = W.head_weights(tables["train"], N_CLASSES)
species_of = {int(l): str(class_map[l]) for l in range(N_CLASSES)}
print({s: len(d) for s, d in dfs.items()}, "| classes:", N_CLASSES)''')

code('''def score_test(model, name, history):
    """One test-split pass per arm: metrics, per-class detail, and the alert stream it would produce."""
    probs, truths = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loaders["test"]:
            probs.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
            truths.append(y.numpy())
    probs, ytest = np.concatenate(probs), np.concatenate(truths)
    pred = probs.argmax(1)
    m = W.metrics(ytest, pred, LABELS)
    counts = pd.Series(ytest).value_counts().reindex(LABELS, fill_value=0)
    rich = np.array([l for l in LABELS if counts[l] >= 50])
    m["macro_f1_rich_classes"] = W.metrics(ytest, pred, rich)["macro_f1"]
    m["classes_present_in_test"] = int(len(rich))

    tbl = tables["test"].iloc[:len(pred)].copy()
    evidence = A.visits_from_manifest(tbl, confidence=probs.max(1), predicted=pred)
    truth = A.visits_from_manifest(tbl)
    gate = A.evaluate_per_frame_vs_gated(evidence, truth, species_of, A.AlertConfig())
    curve = A.budget_curve(evidence, truth, species_of, A.AlertConfig(), budgets=(50, 75, 100, 125))
    torch.save(model.state_dict(), CKPT / f"{name}.pt")
    return {"model": name, "test_macro_f1": round(m["macro_f1"], 4),
            "test_macro_f1_rich": round(m["macro_f1_rich_classes"], 4),
            "test_accuracy": round(m["accuracy"], 4),
            "best_epoch": int(history.loc[history.val_macro_f1.idxmax(), "epoch"]),
            "val_best": float(history.val_macro_f1.max()), "epochs_run": len(history),
            "gate_alerts": gate["gated"]["alerts"], "gate_false": gate["gated"]["false_alerts"],
            "gate_missed": gate["gated"]["missed_alerts"], "gate_precision": gate["gated"]["precision"],
            "gate_recall": gate["gated"]["recall_of_qualifying_days"],
            "perframe_alerts": gate["per_frame"]["alerts"],
            "perframe_recall": gate["per_frame"]["recall_of_qualifying_days"]}, m, gate, curve


def run_arm(name, unfreeze, lr, smoothing):
    """One arm from the same pretrained weights and seed; selection on val only."""
    W.set_seed(SEED)
    model = W.load_resnet50(WEIGHTS, n_classes=N_CLASSES, pretrained=True)
    n_train = W.freeze_except(model, keep=unfreeze)
    t0 = time.time()
    hist = W.train_classifier(model, loaders, DEVICE, epochs=EPOCHS, lr=lr, weight=class_weight,
                              log=OUT / f"{name}_history.csv", eval_splits=("val",),
                              patience=PATIENCE, label_smoothing=smoothing)
    wall = round(time.time() - t0, 1)
    row, m, gate, curve = score_test(model, name, hist)
    row.update({"unfrozen": unfreeze, "lr": lr, "label_smoothing": smoothing, "trainable": n_train,
                "train_wall_sec": wall})
    return model, hist, row, m, gate, curve''')

md("""## Arm A — fine-tune `layer4` + head, regularised, 30 epochs""")

code('''model_a, hist_a, row_a, m_a, gate_a, curve_a = run_arm("finetune_reg", "fc,layer4", 1e-4, 0.1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist_a.epoch, hist_a.train_loss, "o-")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss")
ax[1].plot(hist_a.epoch, hist_a.val_macro_f1, "o-", label="val macro F1 (selection)")
ax[1].axhline(REF["frozen_logreg"], ls=":", color="crimson", label="frozen-features baseline test F1")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro F1"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "armA_curves.png", dpi=120)
print(json.dumps(row_a, indent=2))''')

md("""## Arm B — frozen backbone, head only, same augmentation and budget

If the backbone really is the problem on unseen camera sites, this arm should beat arm A while costing a
fraction of the compute. It also isolates one variable at a time against arm A: which parameters receive
gradients.""")

code('''model_b, hist_b, row_b, m_b, gate_b, curve_b = run_arm("head_only_aug", "fc", 3e-4, 0.1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist_b.epoch, hist_b.train_loss, "s-", label="arm B head only")
ax[0].plot(hist_a.epoch, hist_a.train_loss, "o-", label="arm A fc+layer4")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].legend()
ax[1].plot(hist_b.epoch, hist_b.val_macro_f1, "s-", label="arm B val")
ax[1].plot(hist_a.epoch, hist_a.val_macro_f1, "o-", label="arm A val")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val macro F1"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "armA_vs_armB.png", dpi=120)
print(json.dumps(row_b, indent=2))''')

md("""## Verdict

Compare against the frozen-feature logistic baseline (0.6245) on the same seed and split files. Whatever
wins here becomes the classifier that feeds the alert rule, and the gate columns show whether a better
classifier actually reduces missed alerts — the two are not the same claim.""")

code('''summary = pd.DataFrame([{"model": "majority_class", "test_macro_f1": REF["majority"]},
                       {"model": "random", "test_macro_f1": REF["random"]},
                       {"model": "frozen_logreg_baseline", "test_macro_f1": REF["frozen_logreg"]},
                       {"model": "notebook02_finetune_8ep", "test_macro_f1": REF["notebook02_finetune"]},
                       row_a, row_b])
cols = [c for c in ("model", "val_best", "best_epoch", "epochs_run", "test_macro_f1",
                    "test_macro_f1_rich", "test_accuracy", "gate_false", "gate_missed",
                    "gate_precision", "gate_recall", "train_wall_sec") if c in summary]
summary = summary[cols]
summary["delta_vs_frozen_logreg"] = (summary.test_macro_f1 - REF["frozen_logreg"]).round(4)
best_arm = max([row_a, row_b], key=lambda r: r["val_best"])
print("selected on val:", best_arm["model"], "| its test macro F1", best_arm["test_macro_f1"],
      "| baseline 0.6245")
print("clears baseline:", best_arm["test_macro_f1"] > REF["frozen_logreg"])
summary.to_csv(OUT / "summary.csv", index=False)
summary.to_string(index=False)''')

code('''fig, ax = plt.subplots(figsize=(9, 4.5))
sub = summary.dropna(subset=["gate_recall"])
x = np.arange(len(sub))
ax.bar(x - 0.17, sub.gate_precision, 0.34, label="gate precision")
ax.bar(x + 0.17, sub.gate_recall, 0.34, label="gate recall of qualifying days")
ax.set_xticks(x); ax.set_xticklabels(sub.model, rotation=15, ha="right")
ax.set_ylim(0, 1); ax.legend(); ax.set_title("Alert quality per classifier arm (equal rule, equal budget)")
plt.tight_layout(); plt.savefig(OUT / "gate_by_arm.png", dpi=120)
pd.concat([curve_a.assign(arm="A"), curve_b.assign(arm="B")], ignore_index=True)''')

code('''rows = []
for row, gate, curve, hist in ((row_a, gate_a, curve_a, hist_a), (row_b, gate_b, curve_b, hist_b)):
    r = {"run_utc": pd.Timestamp.now(tz="UTC").isoformat(),
         "experiment": "notebook07_longer_training", "seed": SEED, "splits": "data/splits/*.csv",
         "preprocessing": "RandomResizedCrop(0.5-1.0)+flip+jitter+grayscale+blur / resize256+center224",
         "loss": f"CrossEntropyLoss(label_smoothing={row['label_smoothing']}, class weights)",
         "optimizer": f"AdamW head lr={row['lr']} backbone lr/10 wd=1e-4",
         "epochs_max": EPOCHS, "patience": PATIENCE, "batch": BATCH,
         "metrics": row, "gate": gate, "budget_curve": curve.to_dict("records"),
         "history_last": hist.iloc[-1].to_dict(), "hardware": W.hardware_note()}
    with open(OUT / "experiment_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + chr(10))
    rows.append(r["experiment"] + " " + row["model"])
print("logged:", rows)''')

md("""## Honest reading rules for this notebook

* Selection uses **val** only. If arm A or B beats 0.6245 on test by less than about one standard error of
  this split size, the correct statement is "no better than the baseline", not "an improvement".
* `test_macro_f1_rich` drops classes with fewer than 50 test images; both numbers are reported because the
  all-class average is dominated by one class holding 2 test images.
* If both arms land below the logistic baseline, the Phase I model stays the baseline and the product's
  contribution is the alert rule plus calibration — recorded as a design decision with evidence, not a
  failure to hide.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
dest = Path("DL_PBL/notebooks/07_longer_training.ipynb")
dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
json.loads(dest.read_text(encoding="utf-8"))
print(f"wrote {dest}: {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code), JSON valid")
