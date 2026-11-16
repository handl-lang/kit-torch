#!/usr/bin/env bash
# Gate entry — mirrors the Python gates of the sibling split repos
# (kit-onnx/gamekit/eqsat): lint, format, typecheck, then the suite.
set -euo pipefail

uv sync --group dev
.venv/bin/ruff check src
.venv/bin/ruff format --check src || .venv/bin/ruff format src tests
.venv/bin/ty check
.venv/bin/lint-imports
.venv/bin/coverage run -m pytest tests/ -q
.venv/bin/coverage report -m
