"""Unit I - the four first-order update rules the syllabus lists, as plain numpy functions.

Every rule has the same shape: it receives the current parameters, the gradient that the hand-written
`MLP.backward` produced, and a small state dict (velocity for the momentum family), and it returns new
parameters plus the new state. Keeping them pure functions is what makes them comparable line by line,
and what lets a test drive them with an identical gradient sequence.

    batch GD   theta <- theta - lr * grad(theta)
    SGD        same step, but the gradient comes from one minibatch
    Momentum   v <- mu * v + grad ; theta <- theta - lr * v          (Polyak)
    Nesterov   v <- mu * v + grad ; theta <- theta + mu*(v - v_prev) - lr * v
               i.e. the same velocity recursion, but the iterate carried forward is the extrapolated
               point y = theta - mu*v, so the next gradient is evaluated ahead of the step instead of
               at the current position.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

try:
    from .neurons import MLP
    from .tensors import SEED
except ImportError:  # self-check run as a plain script rather than as part of the package
    from neurons import MLP  # type: ignore[no-redef]
    from tensors import SEED  # type: ignore[no-redef]

Params = list[np.ndarray]
State = dict


def batch_gradient_descent(params: Params, grads: Params, state: State | None = None,
                           lr: float = 0.05, mu: float = 0.9) -> tuple[Params, State]:
    """Plain full-batch descent: the whole dataset's gradient, one step.

    `mu` is accepted and ignored so all four rules share one signature and can be swapped in a loop.
    """
    return [p - lr * g for p, g in zip(params, grads)], dict(state or {})


def stochastic_gradient_descent(params: Params, grads: Params, state: State | None = None,
                                lr: float = 0.05, mu: float = 0.9) -> tuple[Params, State]:
    """The same step as batch GD, applied to a minibatch gradient; the sampling lives in `train_mlp`."""
    return [p - lr * g for p, g in zip(params, grads)], dict(state or {})


def momentum_gradient_descent(params: Params, grads: Params, state: State | None = None,
                              lr: float = 0.05, mu: float = 0.9) -> tuple[Params, State]:
    """Classic Polyak momentum: an exponentially weighted average of past gradients drives the step."""
    velocity = [v * mu + g for v, g in zip(_velocity(state, params), grads)]
    new = [p - lr * v for p, v in zip(params, velocity)]
    return new, {"velocity": velocity}


def nesterov_gradient_descent(params: Params, grads: Params, state: State | None = None,
                              lr: float = 0.05, mu: float = 0.9) -> tuple[Params, State]:
    """Nesterov accelerated gradient in its extrapolated-point form.

    With `params` holding y_t = theta_t - mu*v_t, the recursion below is the standard
    v_{t+1} = mu*v_t + grad(y_t), theta_{t+1} = theta_t - lr*v_{t+1} pair rewritten in terms of y,
    which is where the extra `mu*(v_prev - v)` term comes from and why the trajectory separates from
    plain momentum even on an identical gradient sequence.
    """
    previous = _velocity(state, params)
    velocity = [mu * v + g for v, g in zip(previous, grads)]
    new = [p + mu * (v_old - v) - lr * v for p, v, v_old in zip(params, velocity, previous)]
    return new, {"velocity": velocity, "lookahead": True}


def _velocity(state: State | None, params: Params) -> Params:
    if state and "velocity" in state:
        return state["velocity"]
    return [np.zeros_like(p) for p in params]


OPTIMIZERS = {
    "batch_gd": batch_gradient_descent,
    "sgd": stochastic_gradient_descent,
    "momentum": momentum_gradient_descent,
    "nesterov": nesterov_gradient_descent,
}


def weight_l2_gradient(params: Params, grads: Params, l2: float) -> Params:
    """Add the L2 term l2 * w to the gradient of every weight matrix; biases are never decayed.

    The learning rate is not applied here - that is the update rule's job - so the arms stay
    comparable.
    """
    if l2 <= 0.0:
        return grads
    return [g + l2 * p if i % 2 == 0 else g for i, (p, g) in enumerate(zip(params, grads))]


def _minibatches(n: int, batch_size: int, rng: np.random.Generator):
    while True:
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            yield order[start:start + batch_size]


def train_mlp(X: np.ndarray, y: np.ndarray, Xv: np.ndarray | None = None, yv: np.ndarray | None = None,
              optimizer: str = "sgd", updates: int = 300, batch_size: int = 64, lr: float = 0.05,
              hidden: int = 32, seed: int = SEED, init: str = "he", l2: float = 0.0, mu: float = 0.9,
              dropout: float = 0.0, hidden_act: str = "relu", n_classes: int | None = None,
              augment=None, eval_every: int = 10) -> dict:
    """Train one MLP for a fixed number of weight updates, logging loss and accuracy on a schedule.

    `updates` rather than epochs is the clock, so four optimizers with very different cost per step
    are compared on the same horizontal axis; `samples_seen` records how many rows each arm actually
    consumed, because equal updates is not equal compute.
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; have {sorted(OPTIMIZERS)}")
    k = int(n_classes if n_classes is not None else int(y.max()) + 1)
    net = MLP([X.shape[1], hidden, k], hidden_act=hidden_act, init=init, seed=seed, dropout=dropout)
    net.training = True
    step = OPTIMIZERS[optimizer]
    state: State = {}
    rng = np.random.default_rng(seed)
    full_batch = optimizer == "batch_gd"
    batches = _minibatches(len(X), batch_size, rng) if not full_batch else None
    y = np.asarray(y, dtype=int)
    yv = None if yv is None else np.asarray(yv, dtype=int)

    curve, seen, tick = [], 0, 0
    for _ in range(updates):
        if full_batch:
            xb, yb = X, y
        else:
            idx = next(batches)
            xb, yb = X[idx], y[idx]
            if augment is not None:
                xb = augment(xb, rng)
        loss = net.backward(xb, yb)
        net.grads = weight_l2_gradient(net.params, net.grads, l2)
        net.params, state = step(net.params, net.grads, state, lr=lr, mu=mu)
        seen += len(xb)
        tick += 1
        curve.append({"update": tick, "samples_seen": seen, "batch_loss": loss})
        if tick % eval_every == 0 or tick == updates:
            net.training = False
            row = {"update": tick, "train_loss": net.loss(X, y),
                   "train_accuracy": net.accuracy(X, y)}
            if Xv is not None:
                row["val_loss"] = net.loss(Xv, yv)
                row["val_accuracy"] = net.accuracy(Xv, yv)
            curve[-1].update(row)
            net.training = True
    net.training = False
    out = {"curve": pd.DataFrame(curve), "net": net, "final_loss": float(curve[-1]["batch_loss"]),
           "state": state, "optimizer": optimizer, "lr": lr, "seed": seed}
    if Xv is not None:
        out["val_loss"] = float(net.loss(Xv, yv))
        out["val_accuracy"] = float(net.accuracy(Xv, yv))
    return out


def compare_optimizers(X: np.ndarray, y: np.ndarray, Xv: np.ndarray | None = None,
                       yv: np.ndarray | None = None, updates: int = 250, lr: float = 0.05,
                       batch_size: int = 64, hidden: int = 32, seed: int = SEED,
                       mu: float = 0.9) -> dict:
    """One fixed problem (same seed, same rows, same architecture), four update rules.

    The curves are the loss of the *whole* training block, not the minibatch loss, so the four arms
    are measured by one identical objective; minibatch loss is a noisy, batch-dependent quantity and
    comparing it across arms would compare sampling as much as optimisation. Momentum-family arms
    share `lr` with the plain arms: retuning each arm separately would compare hyperparameters, not
    update rules. `samples_seen` is reported because equal updates is not equal compute.
    """
    curves, summary = {}, []
    for name in OPTIMIZERS:
        t0 = time.perf_counter()
        run = train_mlp(X, y, Xv, yv, optimizer=name, updates=updates, lr=lr, batch_size=batch_size,
                        hidden=hidden, seed=seed, mu=mu, eval_every=1)
        seconds = round(time.perf_counter() - t0, 2)
        loss = run["curve"]["train_loss"].to_numpy()
        curves[name] = (run["curve"]["update"].to_numpy(), loss)
        row = {"optimizer": name, "updates": updates, "seconds": seconds,
               "samples_seen": int(run["curve"]["samples_seen"].iloc[-1]),
               "loss_iter_1": float(loss[0]), "loss_mid": float(loss[updates // 2]),
               "loss_final": float(loss[-1]), "best_loss": float(loss.min())}
        if Xv is not None:
            row["val_loss"] = run["val_loss"]
            row["val_accuracy"] = run["val_accuracy"]
        summary.append(row)
    return {"curves": curves, "table": pd.DataFrame(summary).set_index("optimizer"),
            "momentum_mu": mu}


if __name__ == "__main__":
    try:
        from .tensors import wildlife_problem, zscore
    except ImportError:
        from tensors import wildlife_problem, zscore  # type: ignore[no-redef]
    prob = wildlife_problem()
    X, Xv = zscore(prob["X"], prob["Xv"])
    result = compare_optimizers(X, prob["y"], Xv, prob["yv"], updates=120)
    print(result["table"].to_string())
    print("spread of final loss across arms:", round(float(result["table"]["loss_final"].max()
                                                       - result["table"]["loss_final"].min()), 6))
