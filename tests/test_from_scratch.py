"""Executable checks for `src/from_scratch`, run as a script rather than through pytest.

    .venv/Scripts/python.exe DL_PBL/tests/test_from_scratch.py

Every check asserts a *measured* property, and each is written so that a wrong implementation fails:
gradient checks against finite differences, the McCulloch-Pitts truth tables, the XOR asymmetry between
a single unit and a hidden layer, distinct optimiser trajectories on an identical gradient sequence,
bounded versus unbounded Hebbian growth, more than one competitive winner, and a validation-training
gap that regularisation must actually move. A metric that comes out identical between two arms is a
dead code path, so those cases raise rather than pass.

Exit code is 0 only when every check passes; the last line is the summary with the key numbers.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.from_scratch import learning_rules as LR  # noqa: E402
from src.from_scratch import neurons as N  # noqa: E402
from src.from_scratch import optimizers as O  # noqa: E402
from src.from_scratch import regularization as RG  # noqa: E402
from src.from_scratch import tensors as T  # noqa: E402

MEASURED: dict[str, float] = {}
MODULE_FILES = ["tensors.py", "neurons.py", "optimizers.py", "learning_rules.py", "regularization.py"]


def test_shapes_and_dtypes_are_as_reported() -> None:
    prob = T.wildlife_problem()
    rows = {r["object"]: r for r in T.shape_report(prob["X"])}
    assert rows["feature block X"]["shape"] == str(prob["X"].shape), rows
    assert rows["feature block X"]["dtype"] == "float64", rows
    assert rows["v as column (3,1)"]["shape"] == "(3, 1)", rows
    assert np.eye(3).shape == (3, 3)


def test_broadcasting_rules_including_the_refused_case() -> None:
    cases = {c["case"]: c["result"] for c in T.broadcast_report()}
    assert cases["(3,1) + (1,4)"] == "(3, 4)", cases
    assert cases["(2,3) + (3,)"] == "(2, 3)", cases
    assert cases["matmul (4,2,3) @ (3,5)"] == "(4, 2, 5)", cases
    assert cases["(2,3) + (4,)"] == "ValueError", cases
    caught = [c["why"] for c in T.broadcast_report() if c["result"] == "ValueError"]
    assert "broadcast" in caught[0].lower(), caught


def test_inner_products_and_reshape_identities() -> None:
    prob = T.wildlife_problem()
    X = prob["X"]
    values = {r["operation"]: r["value"] for r in T.inner_product_report(X)}
    assert values["dot((2,3),(3,2))"] == "(2, 2)", values
    gram = X.T @ X
    assert np.allclose(gram, gram.T, atol=1e-12), "Gram matrix must be symmetric"
    views = {r["view"]: r for r in T.reshape_transpose_report(X)}
    assert views["arange(24).reshape(4,6)"]["shape"] == "(4, 6)", views
    assert views["T (transpose)"]["shape"] == "(6, 4)", views
    assert "True" in views[".reshape(2,4,3)"]["detail"], views


def test_svd_reconstruction_error_falls_with_rank() -> None:
    prob = T.wildlife_problem()
    raw, scaled = prob["X"], T.zscore(prob["X"])[0]
    curve = T.svd_rank_curve(raw)
    err = curve["relative_reconstruction_error"].to_numpy()
    var = curve["explained_variance_ratio"].to_numpy()
    effective = int(np.linalg.matrix_rank(raw))
    assert effective < raw.shape[1], "the one-hot groups make the block rank deficient"
    assert np.all(np.diff(err[:effective]) < 0.0), err[:effective]
    assert np.all(np.diff(var) >= 0.0), var
    assert err[effective - 1] < 1e-12, err[effective - 1]
    assert err[0] > 0.1 and var[effective - 1] > 0.999, (err[0], var[effective - 1])
    k = 5
    approx = T.svd_reduce(raw, k)
    assert int(np.linalg.matrix_rank(approx)) <= k, int(np.linalg.matrix_rank(approx))
    singular = np.linalg.svd(approx, compute_uv=False)
    assert singular[k] < 1e-12 * singular[0], "a rank-k reconstruction must kill the tail"
    rel = np.linalg.norm(raw - approx, "fro") / np.linalg.norm(raw, "fro")
    assert abs(rel - err[k - 1]) < 1e-9, (rel, err[k - 1])
    hits = curve.loc[curve["explained_variance_ratio"] >= 0.95, "rank"]
    assert len(hits) > 0, curve.tail().to_string()
    rank95 = int(hits.iloc[0])
    assert 1 <= rank95 < effective, (rank95, effective)
    scaled_curve = T.svd_rank_curve(scaled)
    scaled_hits = scaled_curve.loc[scaled_curve["explained_variance_ratio"] >= 0.95, "rank"]
    assert int(scaled_hits.iloc[0]) > rank95, "standardising must spread the variance, not hide it"
    MEASURED["svd_error_at_rank5"] = float(err[k - 1])
    MEASURED["svd_rank_for_95pct_variance"] = float(rank95)
    MEASURED["svd_rank_for_95pct_variance_zscored"] = float(scaled_hits.iloc[0])
    MEASURED["svd_effective_rank"] = float(effective)


def test_gradient_check_of_mlp_against_finite_differences() -> None:
    prob = T.wildlife_problem()
    X, _ = T.zscore(prob["X"], prob["Xv"])
    report = T.gradient_check_mlp(X, prob["y"], hidden=4, n_samples=7, eps=1e-6)
    table = report["per_parameter"]
    assert len(table) == 4, table  # W1, b1, W2, b2 for a two-layer network
    assert report["max_abs_error"] < 1e-5, table.to_string()
    assert np.allclose(table["analytic_l2"], table["numeric_l2"], rtol=1e-4, atol=1e-6), table
    MEASURED["gradient_check_max_abs_error"] = report["max_abs_error"]


def test_sigmoid_neuron_hand_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 5)
    w = rng.normal(0.0, 1.0, 5)
    b = float(rng.normal())
    target = float(rng.integers(0, 2))

    def loss(w_, b_) -> float:
        a = float(N.sigmoid(x @ w_ + b_))
        return -(target * np.log(a) + (1 - target) * np.log(1 - a))

    grad_w, grad_b = N.sigmoid_neuron_grad(x, w, b, target)
    eps = 1e-7
    num_w = np.array([(loss(w + np.eye(5)[i] * eps, b) - loss(w - np.eye(5)[i] * eps, b)) / (2 * eps)
                      for i in range(5)])
    num_b = (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps)
    assert np.abs(grad_w - num_w).max() < 1e-5, (grad_w, num_w)
    assert abs(grad_b[0] - num_b) < 1e-5, (grad_b[0], num_b)
    assert abs(grad_b[0] - (float(N.sigmoid(x @ w + b)) - target)) < 1e-9, "delta must collapse to a - t"


def test_mcculloch_pitts_units_reproduce_the_gates() -> None:
    table = N.mp_report()
    for gate, truth in N.MP_TRUTH.items():
        part = table[table["gate"] == gate]
        assert len(part) == 4, part
        assert part["correct"].all(), part.to_string()
        assert part["output"].tolist() == truth, (gate, part["output"].tolist())
    assert table["correct"].all(), table.to_string()


def test_xor_is_impossible_for_one_unit_and_learnable_by_the_mlp() -> None:
    y = np.asarray(N.XOR_TRUTH, dtype=int)
    grid = N.single_unit_grid_search(N.BOOLEAN_PAIRS, y, resolution=25)
    assert grid["best_accuracy"] <= 0.75, grid
    proof = N.xor_contradiction()
    assert "theta < 0" in proof["conflict"], proof
    perc_xor = N.perceptron_train(N.BOOLEAN_PAIRS, y, max_sweeps=200)
    assert not perc_xor["converged"], perc_xor
    assert perc_xor["accuracy"] < 1.0, perc_xor
    perc_and = N.perceptron_train(N.BOOLEAN_PAIRS, np.asarray(N.MP_TRUTH["AND"], dtype=int))
    assert perc_and["converged"] and perc_and["accuracy"] == 1.0, perc_and
    neuron = N.sigmoid_neuron_fit(N.BOOLEAN_PAIRS, y.astype(float), epochs=300, lr=0.5)
    assert neuron["accuracy"] < 1.0, neuron["accuracy"]

    X, yxor = N.xor_dataset(repeats=16, seed=13)
    run = O.train_mlp(X, yxor, optimizer="sgd", updates=400, batch_size=8, lr=0.3, hidden=8,
                      n_classes=2, eval_every=50)
    assert run["net"].accuracy(N.BOOLEAN_PAIRS, y) == 1.0, run["net"].predict(N.BOOLEAN_PAIRS)
    MEASURED["xor_single_unit_best_accuracy"] = grid["best_accuracy"]
    MEASURED["xor_mlp_accuracy"] = float(run["net"].accuracy(N.BOOLEAN_PAIRS, y))


def test_optimizers_are_distinct_on_one_fixed_problem() -> None:
    prob = T.wildlife_problem()
    X, Xv = T.zscore(prob["X"], prob["Xv"])
    result = O.compare_optimizers(X, prob["y"], Xv, prob["yv"], updates=150, lr=0.05)
    losses = result["table"]["loss_final"].to_numpy()
    assert len(set(np.round(losses, 9))) == len(losses), result["table"].to_string()
    assert losses.std() > 1e-3, result["table"].to_string()
    assert result["table"].loc["momentum", "loss_final"] < result["table"].loc["sgd", "loss_final"], \
        result["table"].to_string()
    MEASURED["loss_batch_gd"] = float(result["table"].loc["batch_gd", "loss_final"])
    MEASURED["loss_sgd"] = float(result["table"].loc["sgd", "loss_final"])
    MEASURED["loss_momentum"] = float(result["table"].loc["momentum", "loss_final"])
    MEASURED["loss_nesterov"] = float(result["table"].loc["nesterov", "loss_final"])


def test_nesterov_separates_from_momentum_on_the_same_gradients() -> None:
    params = [np.full((3, 2), 0.5), np.arange(6.0).reshape(3, 2) * 0.1]
    grads = [np.tile(np.arange(2.0) / 2.0, (3, 1)) * 0.5, np.ones((2,)) * 0.3]
    tracks = {"sgd": [], "momentum": [], "nesterov": []}
    states = {"sgd": {}, "momentum": {}, "nesterov": {}}
    current = {k: [p.copy() for p in params] for k in tracks}
    for _ in range(8):
        for name in tracks:
            fn = getattr(O, {"sgd": "stochastic_gradient_descent",
                            "momentum": "momentum_gradient_descent",
                            "nesterov": "nesterov_gradient_descent"}[name])
            current[name], states[name] = fn(current[name], grads, states[name], lr=0.05, mu=0.9)
            tracks[name].append([p.copy() for p in current[name]])
    dist = lambda a, b: max(np.abs(x - y).max() for x, y in zip(a[-1], b[-1]))  # noqa: E731
    assert dist(tracks["nesterov"], tracks["momentum"]) > 1e-6, "Nesterov equals momentum"
    assert dist(tracks["nesterov"], tracks["sgd"]) > 1e-6, "Nesterov equals plain SGD"
    assert dist(tracks["momentum"], tracks["sgd"]) > 1e-6, "momentum equals plain SGD"
    MEASURED["nesterov_vs_momentum_gap"] = dist(tracks["nesterov"], tracks["momentum"])


def test_hebbian_grows_without_bound_and_the_leaky_rule_does_not() -> None:
    prob = T.wildlife_problem()
    X = T.zscore(prob["X"])[0]
    plain = LR.hebbian_weights(X, units=2, updates=20000, decay=0.0, record_every=2000)
    leaky = LR.hebbian_weights(X, units=2, updates=20000, decay=2e-4, record_every=2000)
    assert plain["monotone_growth"], plain["norm_curve"]
    assert plain["norm_curve"][-1] > 100.0 * plain["norm_curve"][0], plain["norm_curve"]
    assert leaky["norm_curve"][-1] < 2.0, leaky["norm_curve"]
    assert leaky["norm_curve"][-1] / leaky["norm_curve"][-3] < 1.01, leaky["norm_curve"]
    assert plain["growth_ratio"] > 20.0 * leaky["growth_ratio"], (plain["growth_ratio"], leaky["growth_ratio"])
    assert leaky["alignment_with_leading_pc"] > 0.9, leaky["unit_alignment_with_pcs"]
    MEASURED["hebbian_final_norm"] = float(plain["norm_curve"][-1])
    MEASURED["modified_hebbian_final_norm"] = float(leaky["norm_curve"][-1])


def test_competitive_learning_does_not_collapse_to_one_winner() -> None:
    prob = T.wildlife_problem()
    X = T.zscore(prob["X"])[0]
    comp = LR.competitive_clusters(X, units=4, updates=4000)
    assert comp["distinct_winners"] > 1, comp["win_counts"]
    assert comp["distinct_winners"] == 4, comp["win_counts"]
    assert comp["largest_cluster_share"] < 0.9, comp["win_counts"]
    assert comp["centroid_separation"] > 0.1, comp["centroid_separation"]
    summary = LR.cluster_summary(prob["rows"], comp["assignments"],
                                prob["frames"].drop_duplicates("label").set_index("label")["class_name"])
    assert len(summary) == 4 and int(summary["members"].min()) > 0, summary
    assert summary["mean_hour"].nunique() > 1, summary
    MEASURED["competitive_largest_cluster_share"] = comp["largest_cluster_share"]


def test_rbm_contrastive_divergence_lowers_free_energy() -> None:
    prob = T.wildlife_problem()
    V = LR.binary_wildlife_rows(prob, n_rows=400)
    assert V.shape[1] == LR.HOUR_BINS + LR.SEASON_BINS + LR.BURST_BINS + prob["n_classes"], V.shape
    assert set(np.unique(V)) == {0.0, 1.0}, "visible units must be binary"
    rbm = LR.rbm_contrastive_divergence(V, hidden=16, epochs=400, lr=0.1)
    assert rbm["free_energy_drop"] > 0.0, rbm["free_energy_before"]
    assert rbm["reconstruction_bit_error"] < rbm["always_zero_bit_error"], rbm
    moved = float(np.std(rbm["histogram_after"])) > float(np.std(rbm["histogram_before"]))
    assert moved, "the weight histogram never changed - the CD updates are dead code"
    MEASURED["rbm_free_energy_drop"] = rbm["free_energy_drop"]
    MEASURED["rbm_reconstruction_bit_error"] = rbm["reconstruction_bit_error"]


def test_hebbian_update_does_not_optimise_the_labelled_objective() -> None:
    """Hebbian learning has no label term, so it must not reduce cross-entropy the way error-correction does.

    The earlier version of this check asserted accuracy would fall under Hebbian. That passed on the
    recarved split and failed on the official one, which shows it was measuring an accident of one data
    draw, not the property being taught. Loss, drift and the ordering against backprop are the claims that
    survive a change of dataset.
    """
    prob = LR.wildlife_features(per_class_train=40)
    out = LR.hebbian_versus_backprop_on_trained_net(prob, updates=300, lr=5e-3)
    assert out["hebbian"]["val_loss_after"] > 2.0 * out["backprop"]["val_loss_after"], out
    assert out["backprop"]["accuracy_change"] > out["hebbian"]["accuracy_change"], out
    assert out["hebbian"]["weight_drift"] > 10.0 * out["backprop"]["weight_drift"], out
    MEASURED["hebbian_drift_accuracy_change"] = out["hebbian"]["accuracy_change"]
    MEASURED["backprop_accuracy_change"] = out["backprop"]["accuracy_change"]


def test_zero_initialisation_collapses_and_he_does_not() -> None:
    init = RG.initialization_comparison(hidden=32, updates=200)
    table = init["table"]
    assert abs(table.loc["zeros", "train_loss_final"] - init["uniform_loss"]) < 0.02, table
    assert table.loc["zeros", "hidden_pairwise_distance"] < 1e-12, table
    assert table.loc["zeros", "distinct_hidden_column_norms"] == 1, table
    for arm in ("small", "xavier", "he"):
        assert table.loc[arm, "train_loss_final"] < table.loc["zeros", "train_loss_final"] - 0.5, table
        assert table.loc[arm, "hidden_pairwise_distance"] > 1e-3, table
    assert len(set(np.round(table["train_loss_final"].to_numpy(), 6))) == 4, table
    MEASURED["zeros_init_train_loss"] = float(table.loc["zeros", "train_loss_final"])
    MEASURED["he_init_train_loss"] = float(table.loc["he", "train_loss_final"])


def test_standardisation_changes_the_optimisation() -> None:
    norm = RG.normalization_comparison(hidden=32, updates=200)["table"]
    delta = float(abs(norm.loc["raw", "train_loss_final"] - norm.loc["zscored", "train_loss_final"]))
    assert delta > 0.05, norm.to_string()
    MEASURED["normalization_loss_delta"] = delta


def test_regularization_shrinks_the_val_minus_train_gap() -> None:
    study = RG.regularization_ablation(hidden=64, updates=400)
    table = study["table"]
    assert table.loc["none", "train_loss_final"] < 0.5, table
    assert table.loc["none", "overfit_depth"] > 0.3, table
    assert table.loc["none", "val_best_update"] < 400, table
    assert table.loc["none", "final_gap"] > 2.0, table
    for arm in ("l2", "dropout", "augmentation", "all three"):
        gap_delta = table.loc["none", "final_gap"] - table.loc[arm, "final_gap"]
        assert gap_delta > 0.05, f"{arm} did not move the gap: {table.to_string()}"
        assert table.loc[arm, "train_loss_final"] > table.loc["none", "train_loss_final"], table
    assert table.loc["all three", "val_loss_final"] < table.loc["none", "val_loss_final"], table
    MEASURED["gap_none"] = float(table.loc["none", "final_gap"])
    MEASURED["gap_l2"] = float(table.loc["l2", "final_gap"])
    MEASURED["gap_dropout"] = float(table.loc["dropout", "final_gap"])
    MEASURED["gap_augmentation"] = float(table.loc["augmentation", "final_gap"])
    MEASURED["gap_all"] = float(table.loc["all three", "final_gap"])


def test_augmentation_dose_response_is_monotone_in_the_gap() -> None:
    doses = RG.augmentation_strength(strengths=(0.0, 0.4, 0.8), updates=300)
    assert doses["final_gap"].is_monotonic_decreasing, doses.to_string()
    assert doses.loc[0.8, "final_gap"] < doses.loc[0.0, "final_gap"] - 0.1, doses.to_string()


def test_same_seed_reproduces_and_another_seed_does_not() -> None:
    prob = T.wildlife_problem()
    X, Xv = T.zscore(prob["X"], prob["Xv"])
    a = O.train_mlp(X, prob["y"], Xv, prob["yv"], optimizer="momentum", updates=60, eval_every=20)
    b = O.train_mlp(X, prob["y"], Xv, prob["yv"], optimizer="momentum", updates=60, eval_every=20)
    c = O.train_mlp(X, prob["y"], Xv, prob["yv"], optimizer="momentum", updates=60, eval_every=20,
                    seed=99)
    assert np.array_equal(a["net"].params[0], b["net"].params[0]), "same seed must replay exactly"
    assert np.array_equal(a["curve"]["batch_loss"].to_numpy(), b["curve"]["batch_loss"].to_numpy())
    assert not np.array_equal(a["net"].params[0], c["net"].params[0]), "seed must change the trajectory"


def test_split_is_camera_day_disjoint_and_carries_every_species() -> None:
    prob = T.wildlife_problem(per_class_train=60, per_class_val=30)
    assert prob["overlap_days"] == [], prob["overlap_days"]
    assert prob["overlap_sequences"] == [], prob["overlap_sequences"]
    assert len(prob["class_counts_train"]) == prob["n_classes"]
    assert min(prob["class_counts_train"]) >= 30, prob["class_counts_train"]
    assert min(prob["class_counts_val"]) >= 15, prob["class_counts_val"]
    assert np.all(np.isfinite(prob["X"])) and np.all(np.isfinite(prob["Xv"]))
    MEASURED["n_train_rows"] = float(len(prob["y"]))
    MEASURED["n_val_rows"] = float(len(prob["yv"]))
    MEASURED["n_features"] = float(prob["X"].shape[1])


def test_no_torch_is_imported_by_the_from_scratch_modules() -> None:
    for name in MODULE_FILES:
        source = (ROOT / "src" / "from_scratch" / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import torch|from torch)\b", source, flags=re.MULTILINE), name
        assert "torch." not in source, name


def run(pattern: str = "") -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and (not pattern or pattern in n)]
    failures = []
    started = time.time()
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as err:  # noqa: BLE001 - a failing check must be reported, not raised
            print(f"FAIL  {name}: {type(err).__name__}: {err}")
            failures.append(name)
    elapsed = time.time() - started
    key = (" ".join(f"{k}={v:.4g}" for k, v in sorted(MEASURED.items()))
           if len(failures) == 0 else f"failing: {failures}")
    print(f"test_from_scratch: {len(tests) - len(failures)}/{len(tests)} checks passed in "
          f"{elapsed:.1f}s | seed {T.SEED} | {key}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else ""))
