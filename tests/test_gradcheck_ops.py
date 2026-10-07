"""Property tests: autograd gradients must match finite-difference estimates."""

from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from autograd_numpy import Value
from tests.gradcheck import numerical_grad

floats = st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)


def check_binary(f: Callable[[Value, Value], Value], x: float, y: float) -> None:
    """Compare autograd's d f/dx and d f/dy with central differences."""
    a, b = Value(x), Value(y)
    f(a, b).backward()
    assert a.grad == pytest.approx(
        numerical_grad(lambda t: f(Value(t), Value(y)).data, x), abs=1e-4
    )
    assert b.grad == pytest.approx(
        numerical_grad(lambda t: f(Value(x), Value(t)).data, y), abs=1e-4
    )


def check_unary(f: Callable[[Value], Value], x: float) -> None:
    a = Value(x)
    f(a).backward()
    assert a.grad == pytest.approx(numerical_grad(lambda t: f(Value(t)).data, x), abs=1e-4)


@given(floats, floats)
def test_add(x: float, y: float) -> None:
    check_binary(lambda a, b: a + b, x, y)


def test_same_node_used_twice_accumulates() -> None:
    # b = a + a, so db/da = 2. With `grad = ...` instead of `grad += ...`, the second
    # write would overwrite the first and give 1.
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0
