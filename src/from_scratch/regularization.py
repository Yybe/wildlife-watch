"""Unit III - initialisation, normalisation, weight decay, dropout and augmentation, measured.

All four studies run on the same tabular wildlife problem produced by `tensors.wildlife_problem`
(real camera-trap rows from `metadata.csv`, no pixels), with the train/val split held apart by camera
day so nothing can leak from the validation side into the training side. The point of the file is a
single curve: with regularisation switched off the training loss falls while the validation loss
climbs, and every regulariser moves that divergence.

Tabular augmentation for these features is additive Gaussian jitter with sigma given as a fraction of
the standardised column scale, applied on the fly to each minibatch. That is the honest analogue of
an image flip here - no claim is made about geometric or photometric augmentation, because no pixel
data is involved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .neurons import hidden_unit_collapse
    from .optimizers import train_mlp
    from .tensors import SEED, wildlife_problem, zscore
except ImportError:  # self-check run as a plain script rather than as part of the package
    from neurons import hidden_unit_collapse  # type: ignore[no-redef]
    from optimizers import train_mlp  # type: ignore[no-redef]
    from tensors import SEED, wildlife_problem, zscore  # type: ignore[no-redef]

INIT_ARMS = ("zeros", "small", "xavier", "he")


def jitter_augment(X: np.ndarray, rng: np.random.Generator, sigma: float = 0.15) -> np.ndarray:
    """Gaussian jitter on every feature, resampled per minibatch so the net never sees it twice."""
    return X + rng.normal(0.0, sigma, size=X.shape)


def _problem(per_class_train: int = 40, per_class_val: int = 40, seed: int = SEED,
             standardize: bool = True) -> dict:
    prob = wildlife_problem(per_class_train=per_class_train, per_class_val=per_class_val, seed=seed)
    if standardize:
        prob["X"], prob["Xv"] = zscore(prob["X"], prob["Xv"])
    prob["n_features"] = prob["X"].shape[1]
    return prob


def _tail_gap(curve: pd.DataFrame, fraction: float = 0.2) -> float:
    """Mean (val loss - train loss) over the last `fraction` of evaluation points."""
    ev = curve.dropna(subset=["train_loss", "val_loss"])
    take = max(1, int(round(fraction * len(ev))))
    return float((ev["val_loss"].to_numpy()[-take:] - ev["train_loss"].to_numpy()[-take:]).mean())


def initialization_comparison(hidden: int = 64, updates: int = 400, lr: float = 0.05,
                              seed: int = SEED, per_class_train: int = 40) -> dict:
    """What each initialisation does to the first hidden layer, on the identical problem and seed.

    `zeros` is the control: with zero weights every hidden unit computes the same number, receives the
    same gradient, and therefore stays identical forever, so the layer behaves like one unit and the
    cross-entropy cannot fall below the uniform-logit value.
    """
    prob = _problem(per_class_train=per_class_train, seed=seed)
    curves, rows = {}, []
    for arm in INIT_ARMS:
        run = train_mlp(prob["X"], prob["y"], prob["Xv"], prob["yv"], optimizer="sgd", updates=updates,
                        lr=lr, hidden=hidden, seed=seed, init=arm, eval_every=25)
        collapse = hidden_unit_collapse(run["net"])
        ev = run["curve"].dropna(subset=["train_loss"])
        curves[arm] = (ev["update"].to_numpy(), ev["train_loss"].to_numpy())
        rows.append({"init": arm, "train_loss_final": float(ev["train_loss"].iloc[-1]),
                     "val_loss_final": float(run["val_loss"]),
                     "train_accuracy_final": float(ev["train_accuracy"].iloc[-1]),
                     "hidden_pairwise_distance": collapse["mean_pairwise_weight_distance"],
                     "distinct_hidden_column_norms": collapse["distinct_column_norms"]})
    return {"table": pd.DataFrame(rows).set_index("init"), "curves": curves,
            "n_train": len(prob["y"]), "n_features": prob["n_features"], "uniform_loss": float(np.log(13))}


def normalization_comparison(hidden: int = 64, updates: int = 400, lr: float = 0.05,
                             seed: int = SEED) -> dict:
    """Raw feature scales versus train-fitted z-scoring, same seed, same rows, same architecture."""
    arms, curves = [], {}
    for standardize in (False, True):
        prob = _problem(seed=seed, standardize=standardize)
        run = train_mlp(prob["X"], prob["y"], prob["Xv"], prob["yv"], optimizer="sgd", updates=updates,
                        lr=lr, hidden=hidden, seed=seed, eval_every=25)
        label = "zscored" if standardize else "raw"
        ev = run["curve"].dropna(subset=["train_loss"])
        curves[label] = (ev["update"].to_numpy(), ev["train_loss"].to_numpy())
        scale = np.abs(prob["X"]).mean(0)
        arms.append({"preprocessing": label, "mean_abs_feature": float(scale.mean()),
                     "max_feature_scale": float(np.abs(prob["X"]).max()),
                     "train_loss_final": float(ev["train_loss"].iloc[-1]),
                     "val_loss_final": float(run["val_loss"]),
                     "train_accuracy_final": float(ev["train_accuracy"].iloc[-1])})
    return {"table": pd.DataFrame(arms).set_index("preprocessing"), "curves": curves}


def regularization_ablation(hidden: int = 64, updates: int = 500, lr: float = 0.05, batch_size: int = 64,
                            seed: int = SEED, per_class_train: int = 40, l2: float = 1e-2,
                            dropout: float = 0.35, sigma: float = 0.5) -> dict:
    """Train-versus-validation under each regulariser alone and combined, on one fixed problem.

    `sigma` is 0.5 rather than a gentler default because `augmentation_strength` shows the jitter only
    bites at that scale on these features; a weaker setting would look like a null result.
    """
    prob = _problem(per_class_train=per_class_train, seed=seed)
    arms = {
        "none": {},
        "l2": {"l2": l2},
        "dropout": {"dropout": dropout},
        "augmentation": {"augment": lambda x, rng: jitter_augment(x, rng, sigma)},
        "all three": {"l2": l2, "dropout": dropout,
                      "augment": lambda x, rng: jitter_augment(x, rng, sigma)},
    }
    curves, rows = {}, []
    for name, kwargs in arms.items():
        run = train_mlp(prob["X"], prob["y"], prob["Xv"], prob["yv"], optimizer="sgd", updates=updates,
                        lr=lr, batch_size=batch_size, hidden=hidden, seed=seed, eval_every=25, **kwargs)
        ev = run["curve"].dropna(subset=["train_loss", "val_loss"])
        curves[name] = {"train": (ev["update"].to_numpy(), ev["train_loss"].to_numpy()),
                        "val": (ev["update"].to_numpy(), ev["val_loss"].to_numpy())}
        val = ev["val_loss"].to_numpy()
        rows.append({"arm": name, "l2": kwargs.get("l2", 0.0), "dropout": kwargs.get("dropout", 0.0),
                     "augmented": kwargs.get("augment") is not None,
                     "train_loss_final": float(ev["train_loss"].iloc[-1]),
                     "val_loss_final": float(val[-1]), "val_loss_best": float(val.min()),
                     "val_best_update": int(ev["update"].to_numpy()[int(val.argmin())]),
                     "overfit_depth": float(val[-1] - val.min()), "final_gap": _tail_gap(ev)})
    table = pd.DataFrame(rows).set_index("arm")
    return {"table": table, "curves": curves, "n_train": len(prob["y"]), "n_val": len(prob["yv"]),
            "n_features": prob["n_features"]}


def augmentation_strength(strengths: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5), hidden: int = 64,
                          updates: int = 400, seed: int = SEED) -> pd.DataFrame:
    """Dose-response for the jitter sigma, so the augmentation arm is not a single lucky setting."""
    prob = _problem(seed=seed)
    rows = []
    for sigma in strengths:
        augment = None if sigma == 0.0 else (lambda x, rng, s=sigma: jitter_augment(x, rng, s))
        run = train_mlp(prob["X"], prob["y"], prob["Xv"], prob["yv"], optimizer="sgd", updates=updates,
                        hidden=hidden, seed=seed, eval_every=25, augment=augment)
        ev = run["curve"].dropna(subset=["train_loss", "val_loss"])
        rows.append({"sigma": sigma, "train_loss_final": float(ev["train_loss"].iloc[-1]),
                     "val_loss_final": float(run["val_loss"]),
                     "val_accuracy_final": float(run["val_accuracy"]),
                     "final_gap": _tail_gap(ev)})
    return pd.DataFrame(rows).set_index("sigma")


if __name__ == "__main__":
    init = initialization_comparison(updates=200)
    print(init["table"].round(4).to_string(), "\nuniform loss", init["uniform_loss"])
    print(normalization_comparison(updates=200)["table"].round(4).to_string(), "")
    reg = regularization_ablation(updates=300)
    print(reg["table"].round(4).to_string())
    print("rows train/val:", reg["n_train"], reg["n_val"], "features:", reg["n_features"])
