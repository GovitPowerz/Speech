#!/usr/bin/env bash
set -uo pipefail

cd src/rust
echo "Running all Rust checks..."

cargo test --quiet
test_status=$?

cargo fmt --all -- --check
fmt_status=$?

cargo clippy --all-targets --all-features --quiet -- -D warnings
clippy_status=$?

cd ../..

./build.sh
build_status=$?

echo ""
echo "=== Results ==="
[ $test_status -eq 0 ]   && echo "  Tests:      ok" || echo "  Tests:      FAILED"
[ $fmt_status -eq 0 ]    && echo "  Formatting: ok" || echo "  Formatting: FAILED"
[ $clippy_status -eq 0 ] && echo "  Clippy:     ok" || echo "  Clippy:     FAILED"
[ $build_status -eq 0 ]  && echo "  Build:      ok" || echo "  Build:      FAILED"
echo ""

if [ $test_status -ne 0 ] || [ $fmt_status -ne 0 ] || [ $clippy_status -ne 0 ] || [ $build_status -ne 0 ]; then
    echo "Some checks failed!"
    exit 1
fi
echo "All checks passed!"
