#!/usr/bin/env bash
set -euo pipefail
echo "ruff: sorting imports..."
uv run ruff check --select I --fix --config=pyproject.toml src/python tests
echo "ruff: formatting..."
uv run ruff format --config=pyproject.toml src/python tests
echo "ruff: lint..."
uv run ruff check --config=pyproject.toml src/python tests
echo "mypy: type checking..."
uv run mypy --config-file pyproject.toml src/python tests
echo "All linters passed."
