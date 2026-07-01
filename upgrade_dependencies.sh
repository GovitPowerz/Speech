#!/usr/bin/env bash
set -euo pipefail
echo "Updating dependencies..."
uv sync --upgrade --group dev
echo "Rebuilding PyO3 bindings..."
./build.sh
