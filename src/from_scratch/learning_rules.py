"""Unit III - the unsupervised learning rules, plus error correction as the contrast.

Four families, one shared question: what signal actually moves the weights?

* Hebbian - correlation between a unit's input and its own output. No target anywhere, and no bound:
  the weights keep growing, which is why the second arm adds a leak term.
* Modified Hebbian - the same correlation minus a leak proportional to the current weight, so growth
  saturates. The direction still tracks the leading principal component, checked against
  `numpy.linalg.svd` rather than asserted.
* Competitive - activity only, and only the winning unit (largest projection on the input) moves, a
  fraction of the way towards the input. The "error" is distance to the winner's own reference vector,
  not distance to a label.
* Error correction - `MLP.backward`: the derivative of a loss against the true label, propagated
  backwards through the weights.
* Boltzmann/RBM with contrastive divergence - the difference between correlations observed in the data
  and correlations observed in the model's own reconstruction, i.e. a likelihood gradient with no label.

The inputs are the real camera-trap tabular problem from `tensors.wildlife_problem`; the energy-based
arm uses binary rows cut from the same metadata rows, so no synthetic dataset is invented anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .neurons import MLP
    from .optimizers import train_mlp
    from .tensors import SEED, wildlife_problem, zscore
except ImportError:  # self-check run as a plain script rather than as part of the package
    from neurons import MLP  # type: ignore[no-redef]
    from optimizers import train_mlp  # type: ignore[no-redef]
    from tensors import SEED, wildlife_problem, zscore  # type: ignore[no-redef]

RULE_SIGNALS = {
    "hebbian": "Hebbian: the update is the correlation of pre- and post-synaptic activity "
               "(dw = lr * y * x). No label, no error term, and nothing to stop the weights growing.",
    "modified_hebbian": "Modified Hebbian: the same correlation minus a leak proportional to the "
                        "current weight and the square of the unit's own activity "
                        "(dw = lr * y * x - decay * y^2 * w), which bounds the growth.",
    "competitive": "Competitive: winner-take-all. Only the unit with the largest projection moves, "
                   "towards the input (dw = lr * (x - w) for the winner). Driven by activity alone.",
    "error_correction": "Error correction: the gradient of a labelled loss, propagated backwards "
                        "(dw = -lr * dL/dw, with dL from the true class). Needs the ground-truth label.",
    "contrastive_divergence": "Contrastive divergence (Boltzmann/RBM): positive phase from the data, "
                              "negative phase from the model's own reconstruction; a likelihood "
                              "gradient with no labels.",
}
HOUR_BINS, SEASON_BINS, BURST_BINS = 8, 4, 4


def print_rule(name: str) -> None:
    print(RULE_SIGNALS[name])


def wildlife_features(per_class_train: int = 120, per_class_val: int = 40, seed: int = SEED) -> dict:
    """The tabular wildlife problem with train-fitted standardisation on both blocks."""
    prob = wildlife_problem(per_class_train=per_class_train, per_class_val=per_class_val, seed=seed)
    prob["X"], prob["Xv"] = zscore(prob["X"], prob["Xv"])
    return prob


def principal_directions(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of the centred feature block."""
    _, _, vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    return vt[:k]


def project_2d(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Rank-2 SVD projection of a feature block, plus the explained-variance share of each axis."""
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    energy = (s ** 2) / (s ** 2).sum()
    return Xc @ vt[:2].T, float(energy[0]), float(energy[1])


def hebbian_weights(X: np.ndarray, units: int = 2, updates: int = 20000, lr: float = 2e-4,
                    decay: float = 0.0, seed: int = SEED, record_every: int = 2000) -> dict:
    """Linear Hebbian rule (`decay=0.0`) and the modified rule with a leak, in one function.

    The leak is activity-dependent: `dw = lr * y x - decay * y^2 w`, i.e. a weight is decayed in
    proportion to how strongly its unit is firing, which is Oja's modified Hebbian rule. Rows are drawn
    with replacement and the recorded curve is the Frobenius norm, so the arms are separated by
    magnitude - the property under test - and not by a lucky direction.
    """
    print_rule("hebbian" if decay == 0.0 else "modified_hebbian")
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, (units, X.shape[1]))
    norms, at = [], []
    for step in range(1, updates + 1):
        x = X[rng.integers(len(X))]
        y = w @ x
        w = w + lr * np.outer(y, x) - decay * (y * y)[:, None] * w
        if step % record_every == 0 or step == updates:
            norms.append(float(np.linalg.norm(w)))
            at.append(step)
    unit_norms = np.linalg.norm(w, axis=1, keepdims=True)
    alignment = np.abs((w / unit_norms) @ principal_directions(X, units).T)
    return {"weights": w, "norm_curve": np.asarray(norms), "at_updates": np.asarray(at),
            "decay": decay, "lr": lr, "growth_ratio": float(norms[-1] / norms[0]),
            "monotone_growth": bool(np.all(np.diff(norms) > 0.0)),
            "unit_alignment_with_pcs": alignment,
            "alignment_with_leading_pc": float(alignment[:, 0].max()),
            "final_norm": float(norms[-1])}


def competitive_clusters(X: np.ndarray, units: int = 4, updates: int = 6000, lr: float = 0.05,
                         seed: int = SEED) -> dict:
    """Winner-take-all vector quantisation on unit-length centroids and unit-length inputs."""
    print_rule("competitive")
    rng = np.random.default_rng(seed)
    unit_rows = X / np.linalg.norm(X, axis=1, keepdims=True)
    w = unit_rows[rng.choice(len(unit_rows), size=units, replace=False)].copy()
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    for x in unit_rows[rng.integers(len(unit_rows), size=updates)]:
        winner = int(np.argmax(w @ x))
        w[winner] += lr * (x - w[winner])
        w[winner] /= np.linalg.norm(w[winner])
    assignments = np.argmax(unit_rows @ w.T, axis=1)
    counts = np.bincount(assignments, minlength=units)
    return {"centroids": w, "assignments": assignments, "win_counts": counts,
            "distinct_winners": int((counts > 0).sum()),
            "largest_cluster_share": float(counts.max() / counts.sum()),
            "centroid_separation": float(np.mean([np.linalg.norm(w[i] - w[j])
                                                  for i in range(units) for j in range(i + 1, units)])),
            "units": units, "n_rows": len(X)}


def cluster_summary(rows: pd.DataFrame, assignments: np.ndarray, species: pd.Series) -> pd.DataFrame:
    """What each winning unit captured, read off the real metadata columns of the rows it won."""
    table = pd.DataFrame({"cluster": assignments, "hour": rows["hour"].to_numpy(),
                          "label": rows["label"].to_numpy()})
    table["species"] = table["label"].map(species)
    night = ((table["hour"] < 6) | (table["hour"] >= 18)).astype(float)
    grouped = table.groupby("cluster")
    return pd.DataFrame({
        "members": grouped.size(),
        "mean_hour": grouped["hour"].mean().round(2),
        "night_share": night.groupby(table["cluster"]).mean().round(3),
        "species_count": grouped["label"].nunique(),
        "dominant_species": grouped["species"].agg(lambda s: s.mode().iloc[0]),
    })


def binary_wildlife_rows(prob: dict, n_rows: int = 800, seed: int = SEED) -> np.ndarray:
    """Binary visible vectors for the energy-based arm, cut from the same real metadata rows.

    Four one-hot groups - hour bucket, season bucket, position inside the capture burst, species - so
    the correlations the machine has to model are real activity structure.
    """
    rng = np.random.default_rng(seed)
    take = rng.choice(len(prob["y"]), size=min(n_rows, len(prob["y"])), replace=False)
    rows = prob["rows"].iloc[take]
    hour = np.eye(HOUR_BINS)[np.minimum((rows["hour"].to_numpy() // 3).astype(int), HOUR_BINS - 1)]
    season = np.eye(SEASON_BINS)[np.minimum((rows["dayofyear"].to_numpy() // 92).astype(int), SEASON_BINS - 1)]
    burst = np.eye(BURST_BINS)[np.minimum((rows["frame_pos"].to_numpy() // 3).astype(int), BURST_BINS - 1)]
    label = np.eye(prob["n_classes"])[rows["label"].to_numpy()]
    return np.hstack([hour, season, burst, label])


def rbm_contrastive_divergence(V: np.ndarray, hidden: int = 24, epochs: int = 2000, lr: float = 0.1,
                               seed: int = SEED) -> dict:
    """RBM trained with CD-1: one Gibbs step, weight update = positive phase minus negative phase.

    Free energy of a visible vector with binary hidden units is
    F(v) = -b.v - sum_j softplus((W^T v + c)_j); the mean over the data is reported before and after,
    because pushing that down is exactly what this rule is for.
    """
    print_rule("contrastive_divergence")
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, (V.shape[1], hidden))
    b = np.zeros(V.shape[1])
    c = np.zeros(hidden)

    def sample_binary(z: np.ndarray) -> np.ndarray:
        return (rng.random(z.shape) < 1.0 / (1.0 + np.exp(-z))).astype(float)

    def free_energy(visible: np.ndarray) -> np.ndarray:
        return -(visible @ b) - np.logaddexp(0.0, visible @ w + c).sum(axis=1)

    energy_before = free_energy(V)
    histogram_before = w.ravel().copy()
    for _ in range(epochs):
        h_pos = sample_binary(V @ w + c)
        v_neg = sample_binary(h_pos @ w.T + b)
        h_neg = sample_binary(v_neg @ w + c)
        w += lr * (V.T @ h_pos / len(V) - v_neg.T @ h_neg / len(V))
        b += lr * (V.mean(0) - v_neg.mean(0))
        c += lr * (h_pos.mean(0) - h_neg.mean(0))
    energy_after = free_energy(V)
    h = (V @ w + c > 0.0).astype(float)
    reconstructed = (h @ w.T + b > 0.0).astype(float)
    return {"weights": w, "visible_bias": b, "hidden_bias": c,
            "free_energy_before": float(energy_before.mean()),
            "free_energy_after": float(energy_after.mean()),
            "free_energy_drop": float(energy_before.mean() - energy_after.mean()),
            "histogram_before": histogram_before, "histogram_after": w.ravel(),
            "reconstruction_bit_error": float(np.mean(reconstructed != V)),
            "always_zero_bit_error": float(V.mean()),
            "max_abs_weight": float(np.abs(w).max()),
            "hidden_unit_top_bit": [int(np.argmax(w[:, j])) for j in range(hidden)],
            "n_visible": int(V.shape[1]), "n_rows": int(len(V))}


def error_correction_demo(prob: dict, updates: int = 300, seed: int = SEED, hidden: int = 32) -> dict:
    """Backpropagation on the same features: the arm the other rules are contrasted against."""
    print_rule("error_correction")
    run = train_mlp(prob["X"], prob["y"], prob["Xv"], prob["yv"], optimizer="sgd", updates=updates,
                    hidden=hidden, seed=seed, eval_every=25)
    ev = run["curve"].dropna(subset=["train_loss"])
    return {"curve": run["curve"], "train_loss_curve": ev["train_loss"].to_numpy(),
            "update_axis": ev["update"].to_numpy(),
            "final_train_loss": float(ev["train_loss"].iloc[-1]),
            "final_val_loss": float(run["val_loss"]), "final_val_accuracy": float(run["val_accuracy"]),
            "net": run["net"]}


def hebbian_versus_backprop_on_trained_net(prob: dict, updates: int = 300, lr: float = 5e-3,
                                          seed: int = SEED) -> dict:
    """Apply each rule to the *same* converged network and measure what it does to its accuracy.

    Both arms see the same rows in the same order at the same step size; the only difference is the
    driving signal - activity correlation versus label error. The reported `weight_drift` keeps the
    comparison honest, because equal step size on these two gradients is not equal movement: the
    correlation term has no zero at the solution, so it keeps pushing the weights and the trained
    network's accuracy falls, while the label-error term is already near zero.
    """
    base = error_correction_demo(prob, updates=200, seed=seed)["net"]
    y = prob["y"]
    rng = np.random.default_rng(seed)
    order = rng.integers(len(prob["X"]), size=updates)
    out = {}
    for name in ("hebbian", "backprop"):
        net = MLP(list(base.sizes), init="zeros", seed=seed)
        net.params = [p.copy() for p in base.params]
        net.grads = [np.zeros_like(p) for p in net.params]
        before = float(net.accuracy(prob["Xv"], prob["yv"]))
        for i in order:
            x = prob["X"][[i]]
            net.training = False
            if name == "hebbian":
                net.forward(x)
                net.params[0] = net.params[0] + lr * (x.T @ net.cache["activations"][1])
            else:
                net.backward(x, y[[i]])
                net.params = [p - lr * g for p, g in zip(net.params, net.grads)]
        out[name] = {"val_accuracy_before": before,
                     "val_accuracy_after": float(net.accuracy(prob["Xv"], prob["yv"])),
                     "val_loss_after": float(net.loss(prob["Xv"], prob["yv"])),
                     "weight_drift": float(np.linalg.norm(net.params[0] - base.params[0]))}
        out[name]["accuracy_change"] = out[name]["val_accuracy_after"] - out[name]["val_accuracy_before"]
    return out


if __name__ == "__main__":
    prob = wildlife_features()
    X = prob["X"]
    plain = hebbian_weights(X, decay=0.0)
    leaky = hebbian_weights(X, decay=2e-4)
    print("hebbian   norm", round(plain["final_norm"], 3), "growth x", round(plain["growth_ratio"], 2))
    print("modified  norm", round(leaky["final_norm"], 3), "growth x", round(leaky["growth_ratio"], 2))
    print("alignment to leading PC:", round(plain["alignment_with_leading_pc"], 4),
          round(leaky["alignment_with_leading_pc"], 4))
    comp = competitive_clusters(X, units=4)
    print("win counts", comp["win_counts"], "| distinct winners", comp["distinct_winners"],
          "| largest share", round(comp["largest_cluster_share"], 3))
    species = prob["frames"].drop_duplicates("label").set_index("label")["class_name"]
    print(cluster_summary(prob["rows"], comp["assignments"], species).to_string())
    V = binary_wildlife_rows(prob)
    print("rbm visible block", V.shape, "bit density", round(float(V.mean()), 4))
    rbm = rbm_contrastive_divergence(V)
    print({k: round(v, 4) for k, v in rbm.items() if isinstance(v, float)})
