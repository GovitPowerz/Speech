#!/usr/bin/env bash
set -euo pipefail

RUST_DIR="src/rust"
PYO3_DIR="src/rust/speech-py"

echo "=== Building Rust engine ==="
cargo build --release --quiet --manifest-path "$RUST_DIR/Cargo.toml"

echo "=== Building PyO3 bindings (speech_rs) ==="
uv run maturin develop --release --quiet --manifest-path "$PYO3_DIR/Cargo.toml"

echo "=== Done ==="
echo "  Binary: $RUST_DIR/target/release/speech"
echo "  PyO3:   speech_rs installed in .venv"
