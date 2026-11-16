# kit-torch

The PyTorch integration for the mlopt game, split out of
`handl-lang/graphopt` (`packages/mlopt-torch`) in P3 Step D-1e and
renamed `mlopt-torch` → `kit-torch` / `mlopt_torch` → `kit_torch`.

`kit_torch` carries the torch-facing half of the optimizer: the
`torch.export` bridge (`torch_export`/`torch_bindings`/`torch_eval`/
`torch_bridge`), the `TorchSink`/`TorchSource` adapters, the
`TorchBackend`, executors, autotune, the calibrate/meter/report
measurement surface, the contraction/graph/learned policies (with
their bundled `.pt` artifacts), the model zoo, the RL trainer,
region fusion, and the device-placement/dtype-context machinery.

It sits in the **middle** of the torch stack: it depends on
`mlopt-core` + `mlopt-orchestrator` + `mlopt-dtypes`
(handl-lang/eqsat), `torch`, and `numpy` — and on nothing else
*declared*.  `kit-carriers` resolves beside it through a lazy,
undeclared edge (the carrier executor table degrades gracefully
without it); `kit-cuda`, `kit-discovery`, and the game layer sit
above it.

## Install

```sh
uv sync --group dev
```

or as a git dependency:

```toml
kit-torch = { git = "https://github.com/handl-lang/kit-torch.git", tag = "v0.1.0" }
```

Substrate pins (`mlopt-core`, `mlopt-orchestrator`, `mlopt-dtypes`)
resolve from `handl-lang/eqsat` through the consumer's own
`[tool.uv.sources]` — uv honours sources only at the workspace
root, so this repo carries them for its own dev env only.

## Verify

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src
.venv/bin/ty check
.venv/bin/lint-imports
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```
