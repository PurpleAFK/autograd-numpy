"""Hand-checked gradients for the Value engine."""

import pytest

from autograd_numpy import Value


def test_add_forward_and_backward() -> None:
    a, b = Value(2.0), Value(-3.0)
    c = a + b
    c.backward()
    assert c.data == -1.0
    assert (a.grad, b.grad) == (1.0, 1.0)


def test_add_float() -> None:
    a = Value(2.0)
    c = a + 5
    c.backward()
    assert c.data == pytest.approx(7.0)
    assert a.grad == 1.0


def test_chain_of_adds() -> None:
    # d = (a + b) + c: every leaf gets grad 1, and backward must reach through the middle node.
    a, b, c = Value(1.0), Value(2.0), Value(3.0)
    d = (a + b) + c
    d.backward()
    assert d.data == 6.0
    assert (a.grad, b.grad, c.grad) == (1.0, 1.0, 1.0)


def test_mul_by_hand() -> None:
    # d = a*b + c with a=2, b=-3, c=10: dd/da = b = -3, dd/db = a = 2, dd/dc = 1.
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c
    d.backward()
    assert d.data == 4.0
    assert (a.grad, b.grad, c.grad) == (-3.0, 2.0, 1.0)
