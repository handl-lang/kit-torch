# Agent guide — kit-torch

The PyTorch integration for the mlopt game — the `torch.export`
bridge (`torch_export`/`torch_bindings`/`torch_eval`/`torch_bridge`),
the sink/source adapters (`adapters` — `TorchSink`/`TorchSource` plus
the lazy `kit_carriers` executor table), the torch backend
(`backend`), executors, autotune, calibrate/meter/report, the
contraction/graph/learned policies, the model zoo (`models/`), the
RL trainer, region fusion, and the device-placement/dtype-context
machinery.  Split from `handl-lang/graphopt`
`packages/mlopt-torch` in P3 Step D-1e (the fifth package of the
graphopt domain-package dissolution; the `mlopt-*` → `kit-*` rename).
Python 3.13, torch + numpy on the eqsat substrate.

## Layout

- `src/kit_torch/` — the one shipped package (33 modules):
  - `torch_bridge.py` — the facade over export/bindings/eval;
    importing it registers the concrete-eval backend;
  - `adapters.py` — `TorchSink`/`TorchSource` + `_executor_specs`,
    the carrier-executor table resolved lazily;
  - `backend.py` — `TorchBackend`; `executors/` — the executor
    protocol + base;
  - `autotune.py`, `calibrate.py`, `meter.py`, `report.py` — the
    measurement surface;
  - `contraction_policy.py`, `graph_policy.py`,
    `learned_policy.py`, `model_trajectories.py`, `rl.py`,
    `composer.py`, `meta_eval.py`, `regime.py` — the learned/search
    machinery (`artifacts/contraction_policy_{distilled,
    curriculum}.pt` ship as package-data);
  - `models/` — `hybrid`/`ssm` zoo modules; `runners.py` —
    `TorchCompileRunner` + `RegionFusionRunner`;
  - `aten_classify.py` — generated ATen tables (drift-gated by
    graphopt's `tests/test_torch_port_gen.py`, which stayed behind
    with `scripts/torch_port_gen.py`).
- `tests/` — the package-bound suite (65 files carved out of
  graphopt's `tests/` — every file whose imports stay inside this
  package's dep closure: `kit_torch` + `mlopt_core`/
  `mlopt_orchestrator`/`mlopt_dtypes` + torch/numpy/pytest).
  Cross-stack tests (anything touching `kit_discovery`,
  `kit_carriers`, `kit_cuda`, `kit_onnx`, `mlopt_host_tensor`,
  `mlopt_native`, gamekit, benchkit, bench, the `mlopt` product, or
  the `egglog`/`torchvision` optional deps) stayed in graphopt and
  import `kit_torch` as a git dep.

## Contracts

- **Mid-stack** — not a leaf, not a top: kit_torch binds the eqsat
  substrate (`mlopt-core`, `mlopt-orchestrator`, `mlopt-dtypes`,
  tag-pinned git subdir sources) + torch + numpy, and nothing else
  *declared*.  `kit_carriers` sits beside it in a genuinely
  bidirectional backend layer (the carrier law modules carry
  `TORCH_BINDINGS` and import `kit_torch`; `kit_torch.adapters`
  resolves the carrier lowerings lazily under
  `ModuleNotFoundError`), `kit_cuda` sits above it (its layering
  contract lives there), and `kit_discovery`/the game layer sit
  above all of them.
- **Lazy edges stay undeclared** — `kit_carriers` (adapters.py's
  executor table), `kit_cuda` (docstring-only; `CudaGraphRunner`
  must be imported from `kit_cuda`, never re-exported here), and
  `mlopt_native` (contraction_policy.py's guarded import) are all
  `ModuleNotFoundError`-guarded and deliberately absent from
  `[project.dependencies]` *and* from the import-linter forbidden
  list — a declared cycle is not installable, and banning the lazy
  binds would flag the sanctioned degradation path.  Do not add
  them as deps.
- The import-linter `forbidden` contract enforces the rest:
  `kit_discovery`, `kit_onnx`, `mlopt_host_tensor`, `gamekit`,
  `benchkit`, `discoverykit`, `mlopt` are banned outright.
- `requires_cuda` tests skip cleanly at setup when CUDA is absent or
  OOMs on a probe allocation (shared `tests/conftest.py`, which also
  registers `kit_torch.torch_bridge` eagerly so subset runs see the
  concrete-eval backend).
- History was preserved via `git subtree split` of
  `graphopt:packages/mlopt-torch` — `git log` here reaches back
  through the graphopt history.

## Verification

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src
.venv/bin/ruff format --check src
.venv/bin/ty check
.venv/bin/lint-imports
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```

- Coverage floor is `fail_under` in `[tool.coverage.report]`, set at
  measurement on the carved suite — the e2e/law/cross-stack tests
  stayed in graphopt and reach this package through the git dep, so
  the in-repo number is well below the old monorepo wall.  Tighten,
  never lower.

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
