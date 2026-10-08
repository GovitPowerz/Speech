#!/usr/bin/env bash
# 00: what every runner assumes -- the corpus root, a release build of the engine and the PyO3
# module (the ledger refuses a record from a non-release build), and runs/ ignored by git.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CORPUS_ROOT="${SPEECH_CORPUS_ROOT:-data/LRE03-LRE07}"
[[ -d $CORPUS_ROOT/train ]] || { echo "!! corpus root $CORPUS_ROOT has no train/ (set SPEECH_CORPUS_ROOT)"; exit 1; }
./build.sh
uv run python -c 'import speech_rs, sys; p = speech_rs.build_info()["profile"]; sys.exit(0 if p == "release" else f"!! speech_rs built as {p}, not release")'
git check-ignore -q runs/x || { echo "!! runs/ is not gitignored"; exit 1; }
echo "== ready: corpus $CORPUS_ROOT, release build, runs/ ignored"
