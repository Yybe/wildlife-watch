"""Shared plumbing for the wildlife classifier: dataset, frozen features, metrics, train loop.

Kept deliberately small and dependency-free (torch + torchvision + scikit-learn, all already in
`Desktop/PYTHOM/.venv`) because the notebooks are the deliverable and a live viva test may need to
modify this while they run.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

SEED = 13


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WildlifeImages(Dataset):
    """Images named by uuid under a flat root, labelled by a manifest produced by make_splits.py."""

    def __init__(self, manifest: "str | Path | pd.DataFrame", img_root: "str | Path",
                 transform=None, limit: int = 0, seed: int = SEED, cache: dict | None = None):
        df = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest
        if limit and limit < len(df):
            per = max(1, limit // max(1, int(df.label.nunique())))
            df = (df.sample(frac=1.0, random_state=seed).groupby("label", group_keys=False)
                  .head(per).head(limit))
        self.df = df.reset_index(drop=True)
        self.root = Path(img_root)
        self.transform = transform
        self.cache = cache  # optional dict for decoded tensors, shared across epochs

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        key = str(row.filename)
        if self.cache is not None and key in self.cache:
            img = self.cache[key]
        else:
            img = Image.open(self.root / key).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            if self.cache is not None:
                self.cache[key] = img
        return img, int(row.label)


def class_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("label").size().sort_index()


def on_disk(df: pd.DataFrame, img_root: "str | Path", verbose: bool = True) -> pd.DataFrame:
    """Keep only manifest rows whose file actually exists. A partial local sample or a mirror that
    dropped a few images must shrink the split, not crash halfway through an epoch."""
    have = {p.name for p in Path(img_root).glob("*.jpg")}
    keep = df.filename.isin(have)
    if verbose:
        print(f"{int(keep.sum())}/{len(df)} manifest images present in {img_root}")
    return df[keep].reset_index(drop=True)


def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass with the classifier head removed, i.e. a frozen-backbone feature extractor."""
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> dict:
    """Macro F1 is the headline number: with 110 rare classes, accuracy alone can rise while a
    species is never recognised."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    out["per_class_f1"] = {int(l): round(float(f), 4) for l, f in zip(labels if labels is not None
                                                                       else np.unique(y_true), per_class)}
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def train_classifier(model: torch.nn.Module, loaders: dict, device: torch.device, epochs: int = 5,
                     lr: float = 1e-4, weight: torch.Tensor | None = None,
                     log: "str | Path | None" = None, eval_splits: tuple = ("val", "test"),
                     patience: int = 0, label_smoothing: float = 0.0) -> pd.DataFrame:
    """Cross-entropy + AdamW, one metrics row per epoch; head at lr, unfrozen backbone at lr/10.

    `eval_splits` decides what is scored each epoch: scoring `test` every round only burns minutes and
    invites peeking, so long runs pass ("val",) and evaluate test once at the end. With `patience > 0`
    the best-by-val weights are kept and restored before returning, so the model handed back is the one
    checkpoint selection would have chosen rather than whatever epoch finished last.
    """
    model.to(device)
    crit = torch.nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None,
                                     label_smoothing=label_smoothing)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head = [p for n, p in trainable if n.startswith("fc.")]
    body = [p for n, p in trainable if not n.startswith("fc.")]
    groups = [{"params": head, "lr": lr}]
    if body:
        groups.append({"params": body, "lr": lr / 10})  # backbone moves slowly, head fast
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    rows = []
    best = {"val_macro_f1": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, epochs + 1):
        model.train()
        t0, run, seen, nb = time.time(), 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += float(loss) * len(y)
            seen += len(y)
            nb += 1
        evals = {}
        for split in eval_splits:
            if split in loaders:
                preds, truths, vloss = predict(model, loaders[split], device, loss=True)
                evals[split] = metrics(truths, preds)
                evals[split]["loss"] = vloss
        row = {"epoch": epoch, "train_loss": round(run / max(seen, 1), 5),
               "sec": round(time.time() - t0, 1), "steps": nb,
               "val_macro_f1": round(evals.get("val", {}).get("macro_f1", float("nan")), 4),
               "val_accuracy": round(evals.get("val", {}).get("accuracy", float("nan")), 4),
               "val_loss": round(evals.get("val", {}).get("loss", float("nan")), 5)}
        if "test" in evals:
            row["test_macro_f1"] = round(evals["test"]["macro_f1"], 4)
        rows.append(row)
        print(row, flush=True)
        if log:
            pd.DataFrame(rows).to_csv(log, index=False)
        if patience and evals.get("val"):
            if evals["val"]["macro_f1"] > best["val_macro_f1"]:
                best = {"val_macro_f1": evals["val"]["macro_f1"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            elif epoch - best["epoch"] >= patience:
                print(f"early stop: no val gain for {patience} epochs (best epoch {best['epoch']})", flush=True)
                break
    if best["state"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
        print(f"restored best checkpoint: epoch {best['epoch']}, val macro F1 {best['val_macro_f1']:.4f}",
              flush=True)
    return pd.DataFrame(rows)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device,
            loss: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, float]:
    """Forward pass over a loader. With loss=True also returns mean cross-entropy, for val_loss logging."""
    model.eval()
    preds, truths, run, seen = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if loss:
            run += float(torch.nn.functional.cross_entropy(logits, y)) * len(y)
            seen += len(y)
        preds.append(logits.argmax(1).cpu().numpy())
        truths.append(y.cpu().numpy())
    if loss:
        return np.concatenate(preds), np.concatenate(truths), run / max(seen, 1)
    return np.concatenate(preds), np.concatenate(truths)


def load_resnet50(weights_path: "str | Path | None" = None, n_classes: int = 13,
                  pretrained: bool = True) -> torch.nn.Module:
    """ResNet-50 with the ImageNet head replaced. Kaggle notebooks run offline, so `weights_path`
    points at the .pth attached as a dataset instead of letting torchvision download it."""
    from torchvision import models

    model = models.resnet50(weights=None)
    if pretrained:
        if weights_path and Path(weights_path).exists():
            sd = torch.load(str(weights_path), map_location="cpu")
            for k in [k for k in sd if k.startswith("fc.")]:
                sd.pop(k)
            msg = model.load_state_dict(sd, strict=False)
            stray = [k for k in msg.missing_keys if not k.startswith("fc.")]
            assert not stray, f"weights file did not supply {stray[:5]}"
            print(f"loaded {Path(weights_path).name} ({len(sd)} tensors), head replaced")
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity() if n_classes == 0 else torch.nn.Linear(model.fc.in_features, n_classes)
    return model


def freeze_except(model: torch.nn.Module, keep: str = "fc") -> int:
    n = 0
    for name, p in model.named_parameters():
        train = any(name.startswith(k) for k in keep.split(","))
        p.requires_grad = train
        n += int(train)
    return n


def head_weights(df: pd.DataFrame, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights; the long tail is the reason macro F1 is the target."""
    c = class_counts(df).reindex(range(n_classes), fill_value=0).astype(float)
    w = c.sum() / (n_classes * c.clip(lower=1))
    return torch.tensor(w.values, dtype=torch.float32)


def hardware_note() -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    note = {"device": str(dev), "torch": torch.__version__}
    if dev.type == "cuda" and torch.cuda.device_count() > 0:
        note["gpu"] = torch.cuda.get_device_name(0)
        note["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    return note


if __name__ == "__main__":
    # Self-check: fails loudly if the metric or weighting logic drifts.
    set_seed()
    y = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 1, 1, 1, 2])
    m = metrics(y, p, labels=np.array([0, 1, 2]))
    assert abs(m["accuracy"] - 5 / 6) < 1e-9, m
    assert abs(m["macro_f1"] - (0.8 + 0.8 + 1.0) / 3) < 1e-6, m
    assert metrics(y, y, labels=np.array([0, 1, 2]))["macro_f1"] == 1.0
    # A class missing from the truth must score 0, not raise - the test split lacks one species.
    assert metrics(np.array([0, 0]), np.array([1, 1]), labels=np.array([0, 1]))["macro_f1"] == 0.0
    w = head_weights(pd.DataFrame({"label": [0, 0, 0, 0, 1]}), n_classes=2)
    assert w[0] < w[1], w  # the rare class must be weighted up
    assert confusion(y, p, np.array([0, 1, 2])).shape == (3, 3)
    # Training-loop control, proven non-inert: same initial weights, patience on vs off.
    # Val target is the inverse of what training chases, so val must fall every epoch and the restored
    # (best) model must beat the last-epoch model. Different inits would make the two arms incomparable,
    # so both deepcopy one seeded model.
    import copy
    torch.manual_seed(0)
    xt = torch.randn(64, 4)
    yt = (xt[:, 0] > 0).long()
    yv = 1 - yt
    ds = torch.utils.data.TensorDataset
    loaders = {"train": torch.utils.data.DataLoader(ds(xt, yt), batch_size=16),
               "val": torch.utils.data.DataLoader(ds(xt, yv), batch_size=16)}
    base = torch.nn.Linear(4, 2)

    m1 = copy.deepcopy(base)
    hist = train_classifier(m1, loaders, torch.device("cpu"), epochs=6, lr=5e-2,
                            eval_splits=("val",), patience=1)
    assert len(hist) < 6, f"patience did not stop early: {len(hist)} rows"
    assert (hist.val_macro_f1.diff().dropna() < 0).all(), hist
    restored = metrics(yv.numpy(), predict(m1, loaders["val"], torch.device("cpu"))[0])["macro_f1"]
    assert abs(restored - hist.val_macro_f1.iloc[0]) < 1e-4, (restored, hist)

    m2 = copy.deepcopy(base)
    train_classifier(m2, loaders, torch.device("cpu"), epochs=len(hist), lr=5e-2,
                     eval_splits=("val",), patience=0)
    last = metrics(yv.numpy(), predict(m2, loaders["val"], torch.device("cpu"))[0])["macro_f1"]
    assert restored > last, (restored, last)

    m3 = copy.deepcopy(base)
    hist3 = train_classifier(m3, loaders, torch.device("cpu"), epochs=2, lr=5e-2,
                             eval_splits=("val",), patience=0)
    assert list(hist3.columns[:6]) == ["epoch", "train_loss", "sec", "steps", "val_macro_f1", "val_accuracy"], hist3
    assert "test_macro_f1" not in hist3.columns, "test leaked into a val-only run"

    print("wildlife.py self-check passed:", m)
