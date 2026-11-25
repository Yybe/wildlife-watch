"""Unit I - neurons and the multilayer perceptron, written out by hand in numpy.

No autograd and no torch anywhere in this file: `MLP.backward` is the derivative chain, and
`tensors.gradient_check_mlp` verifies it against central differences so the derivation is measured
rather than asserted. Everything is float64 because a gradient check at eps = 1e-6 is meaningless in
float32.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 13

# McCulloch-Pitts units: fire when the weighted sum reaches the threshold. Weights and thresholds are
# the ones the syllabus table uses (excitatory 1, inhibitory -1).
MP_UNITS = {
    "AND": {"weights": np.array([1.0, 1.0]), "threshold": 2.0},
    "OR": {"weights": np.array([1.0, 1.0]), "threshold": 1.0},
    "NAND": {"weights": np.array([-1.0, -1.0]), "threshold": -1.0},
}
MP_TRUTH = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0]}
XOR_TRUTH = [0, 1, 1, 0]
BOOLEAN_PAIRS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])


def step(z: np.ndarray | float, threshold: float = 0.0) -> float:
    """The McCulloch-Pitts activation: 1 at or above the threshold, 0 below."""
    return float(np.sum(z) >= threshold)


def mp_neuron(x, weights, threshold: float) -> int:
    """A McCulloch-Pitts unit over one input vector: the step of its weighted sum."""
    return int(step(np.dot(x, weights), threshold))


def mp_report() -> pd.DataFrame:
    """AND / OR / NAND truth tables produced by the units, never copied from a textbook."""
    rows = []
    for name, unit in MP_UNITS.items():
        truth = MP_TRUTH[name]
        for i, x in enumerate(BOOLEAN_PAIRS):
            rows.append({"gate": name, "x1": int(x[0]), "x2": int(x[1]),
                         "weighted_sum": float(x @ unit["weights"]),
                         "threshold": unit["threshold"], "output": mp_neuron(x, unit["weights"], unit["threshold"]),
                         "truth": truth[i]})
    table = pd.DataFrame(rows)
    table["correct"] = table.output == table.truth
    return table


def xor_contradiction() -> dict:
    """Why no single threshold unit computes XOR: the four rows give inconsistent inequalities.

    Writing the requirement "fire on (1,0) and (0,1), stay silent on (0,0) and (1,1)" as
    w1 >= t, w2 >= t, 0 < t, w1 + w2 < t forces 2t <= w1 + w2 < t, i.e. t < 0 against 0 < t.
    """
    return {
        "constraints": ["w1 >= theta  (row 1,0 must fire)", "w2 >= theta  (row 0,1 must fire)",
                        "0 < theta    (row 0,0 must stay silent)", "w1 + w2 < theta  (row 1,1 must stay silent)"],
        "derived": "w1 >= theta and w2 >= theta  =>  w1 + w2 >= 2*theta",
        "conflict": "w1 + w2 >= 2*theta together with w1 + w2 < theta requires theta < 0, "
                    "which contradicts the silence condition 0 < theta",
        "conclusion": "XOR is not linearly separable, so no McCulloch-Pitts unit with any weights can "
                      "implement it; a hidden layer is required.",
    }


def single_unit_grid_search(X: np.ndarray, y: np.ndarray, resolution: int = 25) -> dict:
    """Exhaustive search over bounded threshold units: the best a single neuron can do on `X, y`."""
    grid = np.linspace(-3.0, 3.0, resolution)
    best_acc, best = 0.0, None
    for w1 in grid:
        for w2 in grid:
            sums = X @ np.array([w1, w2])
            for theta in grid:
                acc = float(np.mean((sums >= theta).astype(int) == y))
                if acc > best_acc:
                    best_acc, best = acc, (w1, w2, theta)
    return {"best_accuracy": best_acc, "best_weights": best,
            "units_examined": int(resolution ** 3)}


def perceptron_train(X: np.ndarray, y: np.ndarray, max_sweeps: int = 200, lr: float = 1.0,
                    seed: int = SEED) -> dict:
    """Rosenblatt perceptron learning: correct every misclassification and stop when a sweep is clean.

    Runs on boolean inputs only, which is where the convergence theorem applies: it terminates for
    separable targets and keeps cycling on XOR.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    b = 0.0
    for sweep in range(1, max_sweeps + 1):
        errors = 0
        for i in rng.permutation(len(X)):
            prediction = int(step(X[i] @ w + b, 0.0))
            error = int(y[i]) - prediction
            if error:
                errors += 1
                w = w + lr * error * X[i]
                b = b + lr * error
        if errors == 0:
            correct = ((X @ w + b >= 0.0).astype(int) == y).mean()
            return {"converged": True, "sweeps": sweep, "weights": w, "bias": b,
                    "accuracy": float(correct)}
    return {"converged": False, "sweeps": max_sweeps, "weights": w, "bias": b,
            "accuracy": float(np.mean((X @ w + b >= 0.0).astype(int) == y))}


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid_neuron_grad(x: np.ndarray, w: np.ndarray, b: float, target: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of the binary cross-entropy for one sigmoid neuron, left in its hand-derived factors.

    dL/da = -t/a + (1-t)/(1-a) and da/dz = a(1-a); their product is exactly (a - t), and the layer
    derivative is that error times the input. Keeping the factors explicit is what makes the
    cancellation checkable by hand in the viva.
    """
    z = float(x @ w + b)
    a = float(sigmoid(z))
    dloss_da = -target / a + (1.0 - target) / (1.0 - a)
    da_dz = a * (1.0 - a)
    delta = dloss_da * da_dz
    return delta * x, np.array([delta])


def sigmoid_neuron_fit(X: np.ndarray, y: np.ndarray, epochs: int = 400, lr: float = 0.5,
                       seed: int = SEED) -> dict:
    """Full-batch gradient descent on a single sigmoid neuron (no hidden layer)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.5, X.shape[1])
    b = np.array([0.0])
    losses, accuracies = [], []
    for _ in range(epochs):
        grad_w = np.zeros_like(w)
        grad_b = np.zeros(1)
        loss = 0.0
        for i in range(len(X)):
            a = float(sigmoid(X[i] @ w + b[0]))
            t = float(y[i])
            loss += -(t * np.log(a) + (1 - t) * np.log(1 - a))
            gw, gb = sigmoid_neuron_grad(X[i], w, b[0], t)
            grad_w += gw
            grad_b += gb
        w -= lr * grad_w / len(X)
        b -= lr * grad_b / len(X)
        losses.append(loss / len(X))
        accuracies.append(float(np.mean((sigmoid(X @ w + b[0]) >= 0.5).astype(int) == y)))
    return {"weights": w, "bias": float(b[0]), "loss_curve": np.array(losses),
            "accuracy_curve": np.array(accuracies), "accuracy": accuracies[-1]}


ACTIVATIONS = {"relu": (relu, relu_grad), "tanh": (tanh, lambda z: 1.0 - np.tanh(z) ** 2),
               "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1.0 - sigmoid(z)))}


def init_parameters(sizes: list[int], kind: str, rng: np.random.Generator) -> list[np.ndarray]:
    """Weight/bias draws for the four initialisation arms.

    `zeros` is not a bad implementation, it is the symmetry argument made visible: every hidden unit
    then has the same receptive field and receives the same gradient, so the layer can never split
    into different detectors.
    """
    params: list[np.ndarray] = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        if kind == "zeros":
            w = np.zeros((fan_in, fan_out))
            b = np.zeros(fan_out)
        elif kind == "small":
            w = rng.uniform(-0.05, 0.05, (fan_in, fan_out))
            b = np.zeros(fan_out)
        elif kind == "he":
            w = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            b = np.zeros(fan_out)
        elif kind == "xavier":
            w = rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out))
            b = np.zeros(fan_out)
        else:
            raise ValueError(f"unknown initialisation {kind!r}")
        params += [w, b]
    return params


class MLP:
    """Dense net: [input, hidden..., n_classes] with ReLU-family hidden layers and softmax output.

    `forward` caches the pre-activations and the dropout masks, `backward` walks that cache backwards
    and fills `self.grads`. Nothing here is differentiated automatically.
    """

    def __init__(self, sizes: list[int], hidden_act: str = "relu", init: str = "he",
                 seed: int = SEED, dropout: float = 0.0):
        self.sizes = [int(s) for s in sizes]
        self.hidden_act = hidden_act
        self.dropout = float(dropout)
        self.rng = np.random.default_rng(seed)
        self.training = False
        self.params = init_parameters(self.sizes, init, self.rng)
        self.grads = [np.zeros_like(p) for p in self.params]
        self.cache: dict | None = None

    @property
    def n_classes(self) -> int:
        return self.sizes[-1]

    def _blocks(self):
        return zip(self.params[0::2], self.params[1::2])

    def forward(self, x: np.ndarray) -> np.ndarray:
        act_fn, _ = ACTIVATIONS[self.hidden_act]
        activations = [x]
        pre_activations: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        a = np.asarray(x, dtype=np.float64)
        for depth, (w, b) in enumerate(self._blocks()):
            z = a @ w + b
            last = depth == len(self.sizes) - 2
            pre_activations.append(z)
            if last:
                a = z
                break
            a = act_fn(z)
            if self.training and self.dropout > 0.0:
                keep = 1.0 - self.dropout
                mask = (self.rng.random(a.shape[1]) >= keep) / (1.0 - keep)
                masks.append(mask)
                a = a * mask
            else:
                masks.append(None)
            activations.append(a)
        activations.append(a)
        self.cache = {"activations": activations, "pre_activations": pre_activations, "masks": masks}
        return a

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean softmax cross-entropy; `y` holds class indices."""
        logits = self.forward(x)
        probs = softmax(logits)
        n = len(x)
        return float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

    def backward(self, x: np.ndarray, y: np.ndarray) -> float:
        """Backpropagation for one batch of class indices; fills `self.grads` and returns the loss.

        Output block: dL/dz = (softmax(z) - onehot(y)) / n. Hidden block l: the error term is
        backpropagated through the layer weights and multiplied by the activation derivative
        (and by the stored dropout mask, since that scaling happened in the forward pass).
        """
        logits = self.forward(x)
        probs = softmax(logits)
        n = len(x)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
        targets = np.zeros_like(logits)
        targets[np.arange(n), y] = 1.0
        delta = (probs - targets) / n
        acts = self.cache["activations"]
        pres = self.cache["pre_activations"]
        masks = self.cache["masks"]
        _, d_act = ACTIVATIONS[self.hidden_act]
        self.grads = [None] * len(self.params)  # type: ignore[list-item]
        for depth in reversed(range(len(pres))):
            a_in = acts[depth]
            self.grads[2 * depth] = a_in.T @ delta
            self.grads[2 * depth + 1] = delta.sum(axis=0)
            if depth == 0:
                break
            delta = (delta @ self.params[2 * depth].T) * d_act(pres[depth - 1])
            if masks[depth - 1] is not None:
                delta = delta * masks[depth - 1]
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.forward(x)).argmax(axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y, dtype=int)))


def hidden_unit_collapse(net: MLP) -> dict:
    """Symmetry measure over the first hidden layer: mean pairwise distance between unit weights.

    Zero initialisation gives 0.0 exactly, because the units are literal copies of one another and
    gradient descent cannot break a symmetry it never sees any difference in.
    """
    w = net.params[0]
    spread = float(np.mean([np.linalg.norm(w[:, i] - w[:, j])
                            for i in range(w.shape[1]) for j in range(i + 1, w.shape[1])]))
    column_norms = np.linalg.norm(w, axis=0)
    return {"mean_pairwise_weight_distance": spread,
            "min_column_norm": float(column_norms.min()),
            "max_column_norm": float(column_norms.max()),
            "distinct_column_norms": int(np.unique(np.round(column_norms, 12)).size)}


def xor_dataset(repeats: int = 32, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """XOR on the boolean square, replicated so a minibatch sampler has something to shuffle."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(BOOLEAN_PAIRS) * repeats)
    idx = np.mod(order, 4)
    return BOOLEAN_PAIRS[idx], np.asarray(XOR_TRUTH, dtype=int)[idx]


if __name__ == "__main__":
    print(mp_report().to_string(index=False))
    print()
    print("single unit on XOR:", single_unit_grid_search(BOOLEAN_PAIRS, np.asarray(XOR_TRUTH)))
    print("perceptron on AND:", {k: v for k, v in perceptron_train(
        BOOLEAN_PAIRS, np.asarray(MP_TRUTH["AND"])).items() if k != "weights"})
    print("perceptron on XOR:", {k: v for k, v in perceptron_train(
        BOOLEAN_PAIRS, np.asarray(XOR_TRUTH)).items() if k != "weights"})
    net = MLP([2, 8, 1], init="he")
    print("collapse(he):", hidden_unit_collapse(net))
