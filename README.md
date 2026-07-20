# autograd-numpy

TODO: one-paragraph description.

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Check

```bash
ruff check .
ruff format --check .
mypy
pytest -v
```

## Layout

- `src/autograd_numpy/` — library code
- `tests/` — pytest + hypothesis
- `.github/workflows/test.yml` — CI (Python 3.11, ruff + mypy + pytest)

## TODO

Done:
- [x] Scaffold: uv, `pyproject.toml`, ruff, `mypy --strict`, pytest + hypothesis, GitHub Actions CI
- [x] Smoke test (import + version)

Engine, one commit per item:
- [ ] Finite-difference gradient helper `(f(x+h) - f(x-h)) / 2h`, tested on plain Python functions
- [ ] `Value` class (`data`, `grad`, `_prev`, `_op`, `__repr__`) + `+` + topological-sort `backward()`
- [ ] Hypothesis gradcheck property test for `+`, plus an `a + a` test (grads accumulate with `+=`)
- [ ] `*`
- [ ] `tanh`
- [ ] `relu` (hypothesis inputs kept away from the kink at 0)
- [ ] Optional: `-`, `__radd__`/`__rmul__`, `/`, `**`

Neural net:
- [ ] `Neuron`, `Layer`, `MLP` with `parameters()` and `zero_grad()`
- [ ] Train on `make_moons` (SGD loop: forward, zero grads, backward, update)
- [ ] Loss curve and decision boundary plots, saved in the repo
- [ ] Pushed, CI green

Wrap-up:
- [ ] Real description here and in `pyproject.toml` (replacing the TODO)
- [ ] README: usage example, results plot, how this differs from PyTorch tensor autograd
- [ ] Keep-or-cut decision for the resume (cut unless it trains on make_moons)

## License

MIT
