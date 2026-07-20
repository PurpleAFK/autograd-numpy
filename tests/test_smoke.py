"""Smoke test: package imports and reports version."""


def test_import() -> None:
    import autograd_numpy

    assert autograd_numpy.__version__ == "0.0.1"
