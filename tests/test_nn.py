"""Tests for Neuron, Layer and MLP."""

import pytest

from autograd_numpy import MLP, Neuron, Value
from tests.gradcheck import numerical_grad


def test_parameter_count() -> None:
    # 2 -> 4 -> 1: (2 weights + 1 bias) * 4 + (4 weights + 1 bias) * 1 = 12 + 5 = 17
    assert len(MLP(2, [4, 1], seed=0).parameters()) == 17


def test_neuron_forward_matches_formula() -> None:
    n = Neuron(2)
    n.w[0].data, n.w[1].data, n.b.data = 0.5, -1.0, 0.25
    out = n([2.0, 3.0])
    assert out.data == pytest.approx(Value(0.5 * 2 - 1.0 * 3 + 0.25).tanh().data)


def test_mlp_gradient_matches_finite_difference() -> None:
    # Backprop through the whole network for one weight, checked numerically.
    model = MLP(2, [3, 1], seed=1)
    x = [0.3, -0.7]
    w = model.layers[0].neurons[1].w[0]

    def loss_at(v: float) -> float:
        old = w.data
        w.data = v
        out = model(x)[0].data ** 2
        w.data = old
        return out

    model.zero_grad()
    (model(x)[0] ** 2).backward()
    assert w.grad == pytest.approx(numerical_grad(loss_at, w.data), abs=1e-5)


def test_zero_grad_resets_everything() -> None:
    model = MLP(2, [3, 1], seed=0)
    model([1.0, 2.0])[0].backward()
    assert any(p.grad != 0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())


def test_training_reduces_loss() -> None:
    # XOR, the classic problem a single neuron can't solve but a hidden layer can.
    xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ys = [-1.0, 1.0, 1.0, -1.0]
    model = MLP(2, [8, 1], seed=0)

    def loss() -> Value:
        return sum(((model(x)[0] - y) ** 2 for x, y in zip(xs, ys, strict=True)), Value(0.0))

    first = loss().data
    for _ in range(100):
        model.zero_grad()
        current = loss()
        current.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad
    assert loss().data < 0.1 * first
