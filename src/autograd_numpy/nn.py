"""A tiny neural-net library on top of Value: Neuron -> Layer -> MLP."""

from __future__ import annotations

import random
from collections.abc import Sequence

from autograd_numpy.engine import Value


class Module:
    """Shared behavior: list your parameters, and reset their grads before each backward()."""

    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """out = act(w . x + b), with act = tanh by default (or none, for the output layer)."""

    def __init__(self, nin: int, act: str = "tanh", rng: random.Random | None = None) -> None:
        rng = rng or random.Random()
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x: Sequence[Value | float]) -> Value:
        act: Value = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        if self.act == "tanh":
            return act.tanh()
        if self.act == "relu":
            return act.relu()
        return act  # "linear"

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]

    def __repr__(self) -> str:
        return f"Neuron({len(self.w)}, act={self.act})"


class Layer(Module):
    """nout independent neurons that all read the same inputs."""

    def __init__(self, nin: int, nout: int, act: str = "tanh", rng: random.Random | None = None):
        self.neurons = [Neuron(nin, act, rng) for _ in range(nout)]

    def __call__(self, x: Sequence[Value | float]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        n = self.neurons[0]
        return f"Layer({len(n.w)} -> {len(self.neurons)}, {n.act})"


class MLP(Module):
    """Stack of layers. Hidden layers use `act`; the last layer is linear (a raw score)."""

    def __init__(self, nin: int, nouts: list[int], act: str = "tanh", seed: int | None = None):
        rng = random.Random(seed)
        sizes = [nin, *nouts]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], act if i < len(nouts) - 1 else "linear", rng)
            for i in range(len(nouts))
        ]

    def __call__(self, x: Sequence[Value | float]) -> list[Value]:
        out: Sequence[Value | float] = x
        for layer in self.layers:
            out = layer(out)
        return list(out)  # type: ignore[arg-type]  # after >=1 layer these are all Values

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP([{', '.join(map(repr, self.layers))}])"
