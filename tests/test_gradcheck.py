"""Sanity-check numerical_grad on functions whose derivatives we know."""

import math

import pytest

from tests.gradcheck import numerical_grad


@pytest.mark.parametrize("x", [-3.0, -0.5, 0.0, 1.0, 2.5])
def test_square(x: float) -> None:
    assert numerical_grad(lambda t: t**2, x) == pytest.approx(2 * x, abs=1e-6)


@pytest.mark.parametrize("x", [-2.0, 0.0, 0.7, math.pi])
def test_sin(x: float) -> None:
    assert numerical_grad(math.sin, x) == pytest.approx(math.cos(x), abs=1e-6)
