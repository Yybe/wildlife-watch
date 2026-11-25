"""Unit II - the numpy tensor sandbox that the rest of `src/from_scratch` stands on.

Only numpy and pandas are used here, so the whole module runs on a CPU-only kernel and every line can
be defended in the viva (torch never appears in `src/from_scratch/`).

The matrices are real camera-trap data: the 203k-row iWildCam v2.0 label table
(`data/raw/iwildcam_meta/iwildcam_v2.0/metadata.csv`) restricted to the 13 species of
`data/splits/class_map.csv`, engineered into a tabular problem (time of day, day of year, position
inside the capture burst, per-site species occupancy). No pixels are needed for any of it - the ~400
local JPGs carry the same rows, but SVD, broadcasting and a gradient check on invented numbers would
prove nothing, so every shape below is traced back to a printed count from this table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META_CSV = ROOT / "data" / "raw" / "iwildcam_meta" / "iwildcam_v2.0" / "metadata.csv"
CLASS_MAP_CSV = ROOT / "data" / "splits" / "class_map.csv"
SEED = 13  # same seed the split generator used, so numbers stay comparable with notebooks/01

_FRAME_CACHE: pd.DataFrame | None = None


def class_map() -> pd.DataFrame:
    return pd.read_csv(CLASS_MAP_CSV)


def n_classes() -> int:
    return len(class_map())


def load_labelled_frames() -> pd.DataFrame:
    """One row per labelled image of the 13 project species, plus engineered time/burst columns.

    `camera_day` (site | date) is the unit that may never straddle a train/val split, and `frame_pos`
    is the index of a frame inside its capture burst - consecutive frames of a burst are
    near-duplicates, which is why grouping is done at that level rather than per image.
    """
    global _FRAME_CACHE
    if _FRAME_CACHE is not None:
        return _FRAME_CACHE
    meta = pd.read_csv(META_CSV, index_col=0, parse_dates=["datetime"])
    cm = class_map()
    id_to_label = dict(zip(cm.category_id, cm.label))
    df = meta[meta.category_id.isin(id_to_label)].copy()
    df["label"] = df.category_id.map(id_to_label).astype(int)
    df["class_name"] = df.label.map(dict(zip(cm.label, cm.class_name)))
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["dayofyear"] = dt.dt.dayofyear
    df["date"] = dt.dt.strftime("%Y-%m-%d")
    df["camera_day"] = df["location"].astype(str) + "|" + df["date"]
    df = df.sort_values(["seq_id", "datetime"], kind="mergesort").reset_index(drop=True)
    df["frame_pos"] = df.groupby("seq_id").cumcount()
    df["seq_len"] = df.groupby("seq_id")["filename"].transform("size")
    start = df.groupby("seq_id")["datetime"].transform("min")
    df["secs_into_seq"] = (df["datetime"] - start).dt.total_seconds()
    _FRAME_CACHE = df
    return df


def split_camera_days(frames: pd.DataFrame, train_fraction: float = 0.55, seed: int = SEED,
                      ) -> tuple[list[str], list[str]]:
    """Hand whole camera-days to the train pool and the rest to the val pool, so the two problem
    halves can never share a site-day session (and therefore never share a capture burst)."""
    rng = np.random.default_rng(seed)
    days = np.asarray(sorted(frames["camera_day"].unique()), dtype=object)
    order = rng.permutation(days)
    cut = int(round(train_fraction * len(order)))
    return list(order[:cut]), list(order[cut:])


def sample_pool(frames: pd.DataFrame, days: list[str], per_class: int, seed: int = SEED) -> np.ndarray:
    """Row positions of whole capture bursts taken from `days` until every species has `per_class`.

    Bursts are the sampling unit because consecutive frames of one visit are near-duplicates; a pool
    built per image would silently oversample the same animal.
    """
    rng = np.random.default_rng(seed)
    pool = frames[frames["camera_day"].isin(days)]
    seq_ids = np.asarray(sorted(pool["seq_id"].unique()), dtype=object)
    order = rng.permutation(seq_ids)
    rows_per_seq = pool.groupby("seq_id").size()
    labels_per_seq = pool.groupby("seq_id")["label"].apply(lambda s: sorted(set(s.tolist())))
    have = np.zeros(n_classes(), dtype=np.int64)
    chosen: list[str] = []
    for seq in order:
        if have.min() >= per_class:
            break
        labs = labels_per_seq[seq]
        if all(have[int(l)] >= per_class for l in labs):
            continue
        chosen.append(seq)
        have[[int(l) for l in labs]] += rows_per_seq[seq]
    return pool.index[np.isin(pool["seq_id"].to_numpy(), np.asarray(chosen, dtype=object))]


def design_matrix(frames: pd.DataFrame, fit_idx: np.ndarray, take_idx: np.ndarray,
                  ) -> tuple[np.ndarray, list[str]]:
    """Numeric feature block for the rows in `take_idx`, with every statistic fit on `fit_idx` only.

    The per-site occupancy vector is a target-encoded feature, so it is estimated on the training
    rows with add-one smoothing and looked up for the rest; a validation row can never contribute to
    the numbers used to encode it.
    """
    k = n_classes()
    fit, take = frames.iloc[fit_idx], frames.iloc[take_idx]
    prior = (fit["label"].value_counts().reindex(range(k), fill_value=0).to_numpy(float) + 1.0)
    prior /= prior.sum()

    site_counts = fit.groupby(["location", "label"]).size().unstack(fill_value=0).reindex(
        columns=range(k), fill_value=0).to_numpy(float)
    site_total = site_counts.sum(1, keepdims=True)
    site_prior = (site_counts + 1.0) / (site_total + k)  # add-one smoothing towards the global prior
    loc_index = {loc: i for i, loc in enumerate(fit["location"].unique())}
    site_rows = np.array([loc_index.get(loc, -1) for loc in take["location"]])
    occ = np.where((site_rows >= 0)[:, None], site_prior[np.clip(site_rows, 0, None)], prior[None, :])
    occ = np.nan_to_num(occ, nan=0.0, posinf=0.0, neginf=0.0)
    site_size = np.log1p(fit.groupby("location").size().reindex(take["location"]).fillna(0).to_numpy(float))

    hour = take["hour"].to_numpy(float)
    doy = take["dayofyear"].to_numpy(float)
    base = np.column_stack([
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / 366.0), np.cos(2 * np.pi * doy / 366.0),
        take["minute"].to_numpy(float) / 59.0, ((hour < 6) | (hour >= 18)).astype(float),
        take["frame_pos"].to_numpy(float) / 9.0,
        np.log1p(take["secs_into_seq"].to_numpy(float)),
        np.log1p(take["seq_len"].to_numpy(float)),
    ])
    base_names = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "minute", "is_night",
                  "frame_pos", "log_secs_into_burst", "log_seq_len"]
    hour_onehot = np.eye(8, dtype=np.float64)[(hour // 3).astype(int).clip(0, 7)]
    names = base_names + [f"hour_bin_{i}" for i in range(8)] + \
        [f"occ_species_{c}" for c in range(k)] + ["log_site_size"]
    X = np.hstack([base, hour_onehot, occ, site_size[:, None]])
    return X.astype(np.float64), names


def wildlife_problem(per_class_train: int = 120, per_class_val: int = 40, seed: int = SEED,
                     ) -> dict:
    """The tabular wildlife problem used by every from-scratch experiment: real rows, no pixels."""
    frames = load_labelled_frames()
    train_days, val_days = split_camera_days(frames, seed=seed)
    train_idx = sample_pool(frames, train_days, per_class_train, seed=seed)
    val_idx = sample_pool(frames, val_days, per_class_val, seed=seed + 1)
    X, names = design_matrix(frames, train_idx, train_idx)
    Xv, _ = design_matrix(frames, train_idx, val_idx)
    y = frames["label"].to_numpy()[train_idx]
    yv = frames["label"].to_numpy()[val_idx]
    taken = frames.iloc[np.concatenate([train_idx, val_idx])]
    return {
        "X": X, "y": y, "Xv": Xv, "yv": yv, "feature_names": names, "frames": frames,
        "rows": frames.iloc[train_idx].reset_index(drop=True),
        "val_rows": frames.iloc[val_idx].reset_index(drop=True),
        "train_days": sorted({d for d in taken["camera_day"].to_numpy()[: len(train_idx)]}),
        "val_days": sorted({d for d in taken["camera_day"].to_numpy()[len(train_idx):]}),
        "train_sequences": int(frames.iloc[train_idx]["seq_id"].nunique()),
        "val_sequences": int(frames.iloc[val_idx]["seq_id"].nunique()),
        "overlap_days": sorted(set(frames.iloc[train_idx]["camera_day"]) &
                               set(frames.iloc[val_idx]["camera_day"])),
        "overlap_sequences": sorted(set(frames.iloc[train_idx]["seq_id"]) &
                                    set(frames.iloc[val_idx]["seq_id"])),
        "n_classes": int(n_classes()),
        "class_counts_train": np.bincount(y, minlength=n_classes()).tolist(),
        "class_counts_val": np.bincount(yv, minlength=n_classes()).tolist(),
    }


def zscore(X: np.ndarray, *refs: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardise with the statistics of `X` (call it on the train block first) and return the
    transformed copies of every reference block."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return tuple((r - mu) / sd for r in ((X,) + refs))


def svd_reduce(X: np.ndarray, k: int) -> np.ndarray:
    """Rank-k approximation U_k S_k V_k^T from numpy.linalg.svd."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def svd_rank_curve(X: np.ndarray, max_rank: int | None = None) -> pd.DataFrame:
    """Explained-variance ratio and relative Frobenius error of the rank-k reconstruction."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    max_rank = int(max_rank if max_rank is not None else min(X.shape))
    energy = S[:max_rank] ** 2
    cum = np.cumsum(energy) / (S ** 2).sum()
    total = np.linalg.norm(X, "fro")
    rel = [np.linalg.norm(X - (U[:, :k] * S[:k]) @ Vt[:k], "fro") / total for k in range(1, max_rank + 1)]
    return pd.DataFrame({"rank": np.arange(1, max_rank + 1), "explained_variance_ratio": cum,
                         "relative_reconstruction_error": np.asarray(rel)})


def finite_difference_grad(loss_fn, params: list[np.ndarray], eps: float = 1e-6) -> list[np.ndarray]:
    """Central-difference gradient of `loss_fn()` with respect to every entry of `params`."""
    grads = []
    for p in params:
        g = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            here = p[idx]
            p[idx] = here + eps
            high = loss_fn()
            p[idx] = here - eps
            low = loss_fn()
            p[idx] = here
            g[idx] = (high - low) / (2.0 * eps)
        grads.append(g)
    return grads


def gradient_check_mlp(X: np.ndarray, y: np.ndarray, hidden: int = 5, n_samples: int = 9,
                       seed: int = SEED, eps: float = 1e-6) -> dict:
    """Compare the hand-written MLP backward pass with central differences, entry by entry.

    Called with the network in eval mode: dropout is a stochastic part of the forward pass and has no
    deterministic finite-difference counterpart, so a gradient check is only meaningful with it off.
    """
    try:  # local import: neurons has no dependency on this file, and the fallback lets the
        from .neurons import MLP  # module self-check run as a plain script
    except ImportError:
        from neurons import MLP  # type: ignore[no-redef]
    X, y = X[:n_samples], np.asarray(y[:n_samples], dtype=int)
    net = MLP([X.shape[1], hidden, int(y.max()) + 1], init="xavier", seed=seed, dropout=0.0)
    net.training = False
    net.forward(X)
    net.backward(X, y)
    analytic = [g.copy() for g in net.grads]
    numeric = finite_difference_grad(lambda: net.loss(X, y), net.params, eps=eps)
    rows = []
    for i, (a, n) in enumerate(zip(analytic, numeric)):
        rows.append({"parameter": f"block{i}", "shape": str(a.shape),
                    "max_abs_error": float(np.abs(a - n).max()),
                    "max_rel_error": float(np.abs(a - n).max() / max(np.abs(n).max(), 1e-12)),
                    "analytic_l2": float(np.linalg.norm(a)), "numeric_l2": float(np.linalg.norm(n))})
    return {"per_parameter": pd.DataFrame(rows),
            "max_abs_error": float(max(r["max_abs_error"] for r in rows)),
            "hidden": hidden, "samples": int(n_samples), "eps": eps}


def shape_report(X: np.ndarray) -> list[dict]:
    """Tensor creation, shapes and dtypes on the real feature block."""
    v = np.array([1.0, 2.0, 3.0])
    rows = [
        {"object": "feature block X", "shape": str(X.shape), "dtype": str(X.dtype),
         "note": f"{X.size:,} float64 entries"},
        {"object": "column vector v", "shape": str(v.shape), "dtype": str(v.dtype), "note": "rank 1"},
        {"object": "v as row (1,3)", "shape": str(v.reshape(1, 3).shape), "dtype": str(v.dtype),
         "note": "reshape changes strides, not data"},
        {"object": "v as column (3,1)", "shape": str(v.reshape(3, 1).shape), "dtype": str(v.dtype),
         "note": "same buffer, new axes"},
        {"object": "ones((2,3))", "shape": str(np.ones((2, 3)).shape), "dtype": str(np.ones((2, 3)).dtype),
         "note": "constant tensor"},
        {"object": "eye(3)", "shape": str(np.eye(3).shape), "dtype": str(np.eye(3).dtype),
         "note": "identity, the neutral element of matmul"},
    ]
    return rows


def broadcast_report() -> list[dict]:
    """Broadcasting rules, including the mismatch that numpy refuses.

    The rule is right-aligned dimension matching: 1 (or absent) stretches, anything else must be
    equal. `(n,3) + (4,)` is the failure a hand-written bias vector produces when its length does not
    match the layer width, which is exactly why `MLP` builds biases from `sizes`.
    """
    out = []
    a = np.ones((3, 1))
    b = np.ones((1, 4))
    out.append({"case": "(3,1) + (1,4)", "result": str((a + b).shape), "why": "both stretch to (3,4)"})
    x = np.ones((2, 3))
    bias3 = np.arange(3.0)
    out.append({"case": "(2,3) + (3,)", "result": str((x + bias3).shape),
                "why": "(3,) is right-aligned onto the last axis"})
    scal = np.ones((2, 3)) * 2.0
    out.append({"case": "(2,3) * scalar", "result": str(scal.shape), "why": "rank-0 broadcasts freely"})
    batch = np.ones((4, 2, 3))
    w = np.ones((3, 5))
    out.append({"case": "matmul (4,2,3) @ (3,5)", "result": str((batch @ w).shape),
                "why": "stacked matmul: batch axis carried through"})
    try:
        _ = np.ones((2, 3)) + np.arange(4.0)
        raise AssertionError("the (2,3) + (4,) mismatch should not broadcast")
    except ValueError as err:
        out.append({"case": "(2,3) + (4,)", "result": "ValueError", "why": str(err)})
    return out


def inner_product_report(X: np.ndarray) -> list[dict]:
    """Inner products, projections and norms - the arithmetic a layer is made of."""
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(6.0).reshape(3, 2)
    u = X[0]
    v = X[1]
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    gram = X.T @ X
    return [
        {"operation": "dot((2,3),(3,2))", "value": str((a @ b).shape), "detail": "matrix product"},
        {"operation": "row . row (feature inner product)", "value": f"{float(u @ v):.4f}",
         "detail": f"cosine between two real frames = {cos:.4f}"},
        {"operation": "Gram X^T X", "value": str(gram.shape),
         "detail": f"symmetric: max|G-G^T| = {np.abs(gram - gram.T).max():.2e}"},
        {"operation": "||X||_F", "value": f"{np.linalg.norm(X, 'fro'):.4f}", "detail": "Frobenius norm"},
        {"operation": "mean |cosine| over 200 row pairs",
         "value": f"{_mean_abs_cosine(X):.4f}", "detail": "how alike two camera-trap rows are"},
    ]


def _mean_abs_cosine(X: np.ndarray, pairs: int = 200, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, X.shape[0], pairs)
    j = rng.integers(0, X.shape[0], pairs)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    z = X / np.where(n < 1e-12, 1.0, n)
    return float(np.abs(np.sum(z[i] * z[j], axis=1)).mean())


def reshape_transpose_report(X: np.ndarray) -> list[dict]:
    """Reshape / transpose identities that a hand-written backprop depends on."""
    small = np.arange(24.0).reshape(4, 6)
    flat = small.reshape(-1)
    back = flat.reshape(2, 4, 3)
    batch, feats = X.shape
    out = [
        {"view": "arange(24).reshape(4,6)", "shape": str(small.shape),
         "detail": f"C-order, strides {small.strides}"},
        {"view": ".reshape(-1)", "shape": str(flat.shape),
         "detail": f"first 4 entries {flat[:4].astype(int).tolist()}"},
        {"view": ".reshape(2,4,3)", "shape": str(back.shape),
         "detail": f"data preserved: {np.array_equal(back.reshape(4, 6), small)}"},
        {"view": "T (transpose)", "shape": str(small.T.shape),
         "detail": f"same buffer, swapped strides {small.T.strides}"},
        {"view": "X.reshape(-1) size", "shape": str((batch * feats,)),
         "detail": f"{batch:,} x {feats} = flattened parameter count if X were a weight matrix"},
        {"view": "X.T @ X vs (X @ X.T).trace()", "shape": "scalar",
         "detail": f"trace identity holds to {abs(np.trace(X.T @ X) - np.trace(X @ X.T)):.2e} "
                    f"on {np.trace(X.T @ X):.2f} (both equal the sum of squares)"},
    ]
    return out


if __name__ == "__main__":
    prob = wildlife_problem()
    print(f"rows: train {len(prob['y'])}, val {len(prob['yv'])}, features {prob['X'].shape[1]}")
    print(f"camera-days: {len(prob['train_days'])} train / {len(prob['val_days'])} val, "
          f"overlap {len(prob['overlap_days'])}")
    print("class counts train:", prob["class_counts_train"])
    curve = svd_rank_curve(zscore(prob["X"])[0], max_rank=20)
    print(curve.head(4).to_string(index=False))
    chk = gradient_check_mlp(prob["X"], prob["y"])
    print("gradient check max abs error:", chk["max_abs_error"])
