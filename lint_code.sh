#!/usr/bin/env bash
set -euo pipefail
# SCOPE NOTE: lints src/python + tests ONLY. scripts/ (the fixture extractors) is NOT
# covered -- a known blind spot (Phase 4d T12 review finding); when touching an
# extractor, lint it manually: uv run ruff check --config=pyproject.toml scripts
echo "ruff: sorting imports..."
uv run ruff check --select I --fix --config=pyproject.toml src/python tests
echo "ruff: formatting..."
uv run ruff format --config=pyproject.toml src/python tests
echo "ruff: lint..."
uv run ruff check --config=pyproject.toml src/python tests
echo "mypy: type checking..."
uv run mypy --config-file pyproject.toml src/python tests
echo "All linters passed."
