"""Scalar reverse-mode autograd: a `Value` wraps one float and remembers how it was made."""

from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    """A node in the computation graph.

    data:      the number computed in the forward pass
    grad:      d(final output)/d(this node), filled in by backward()
    _prev:     the nodes this one was computed from (its children in the graph)
    _op:       the operation that produced it, for debugging/printing
    _backward: closure that pushes self.grad down to _prev via the chain rule
    """

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1 and d(a+b)/db = 1, so each input gets out.grad unchanged.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b and d(a*b)/db = a: each input's grad is scaled by the *other* input.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d tanh(x)/dx = 1 - tanh(x)^2, and tanh(x) is already stored as t.
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            # Slope 1 where the input was positive, 0 otherwise (we pick 0 at exactly x = 0).
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Fill in .grad for every node that self depends on."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)  # appended only after all its inputs: topological order

        build(self)
        self.grad = 1.0  # d(self)/d(self)
        for v in reversed(topo):
            v._backward()
