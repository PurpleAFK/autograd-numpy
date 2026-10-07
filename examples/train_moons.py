"""Train an MLP built on the scalar Value engine to classify sklearn's make_moons.

Run:  python examples/train_moons.py
Saves assets/loss_curve.png and assets/decision_boundary.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from autograd_numpy import MLP, Value

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def loss_and_accuracy(model: MLP, X: np.ndarray, y: np.ndarray, alpha: float = 1e-4):
    """SVM 'max-margin' loss: penalize any point whose score isn't on the right side by >= 1."""
    scores = [model(list(map(float, xi)))[0] for xi in X]
    losses = [(1 + -float(yi) * s).relu() for yi, s in zip(y, scores, strict=True)]
    data_loss = sum(losses, Value(0.0)) * (1.0 / len(losses))
    # L2 regularization keeps weights small, which gives a smoother decision boundary.
    reg_loss = alpha * sum((p * p for p in model.parameters()), Value(0.0))
    accuracy = np.mean([(s.data > 0) == (yi > 0) for yi, s in zip(y, scores, strict=True)])
    return data_loss + reg_loss, float(accuracy)


def main(steps: int = 100, seed: int = 1337) -> None:
    X, y = make_moons(n_samples=100, noise=0.1, random_state=seed)
    y = y * 2 - 1  # labels {0, 1} -> {-1, +1} for the margin loss

    model = MLP(2, [16, 16, 1], seed=seed)
    print(model, f"\n{len(model.parameters())} parameters")

    history = []
    for k in range(steps):
        loss, acc = loss_and_accuracy(model, X, y)  # 1. forward
        model.zero_grad()  # 2. clear grads left over from the previous step
        loss.backward()  # 3. backward
        lr = 1.0 - 0.9 * k / steps  # 4. SGD update, learning rate decays 1.0 -> 0.1
        for p in model.parameters():
            p.data -= lr * p.grad
        history.append((loss.data, acc))
        if k % 10 == 0 or k == steps - 1:
            print(f"step {k:3d}  loss {loss.data:.4f}  accuracy {acc:.0%}")

    ASSETS.mkdir(exist_ok=True)
    plot_history(history)
    plot_boundary(model, X, y)


def plot_history(history: list[tuple[float, float]]) -> None:
    losses, accs = zip(*history, strict=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(losses, color="tab:blue", label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(accs, color="tab:orange", label="accuracy")
    ax2.set_ylabel("accuracy", color="tab:orange")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("MLP(2, [16, 16, 1]) on make_moons")
    fig.tight_layout()
    fig.savefig(ASSETS / "loss_curve.png", dpi=120)
    plt.close(fig)


def plot_boundary(model: MLP, X: np.ndarray, y: np.ndarray, h: float = 0.1) -> None:
    xx, yy = np.meshgrid(
        np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, h),
        np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model(list(map(float, p)))[0].data > 0 for p in grid]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Spectral, edgecolors="k")
    ax.set_title("Decision boundary")
    fig.tight_layout()
    fig.savefig(ASSETS / "decision_boundary.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
