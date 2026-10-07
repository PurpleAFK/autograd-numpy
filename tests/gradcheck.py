"""Finite-difference gradient checking, used to verify autograd results."""

from collections.abc import Callable


def numerical_grad(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Estimate df/dx at x with the central difference (f(x+h) - f(x-h)) / 2h."""
    return (f(x + h) - f(x - h)) / (2 * h)
