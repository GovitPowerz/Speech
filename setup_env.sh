#!/usr/bin/env bash
set -euo pipefail
rm -rf .venv
uv sync --group dev
uv tree
cat <<'EOF' >>.venv/bin/activate

export PYTHONPATH="$PWD:$PWD/src/python"
EOF
