"""Emit deterministic cross-language fixtures for the regression-deltas kernel.

Builds bases from a fixed closed-form (`((i*7 + j*13) % 100) / 100.0`) - NOT
`random`/`np.random` - at a spread of shapes and delta orders, runs each through
`regression_deltas_oracle` (the numpy port of MelFilterBank.cpp:153-170), and
writes `tests/reference_data/phase1/deltas_cases.json`:

    [{"base": [[..]], "n": int, "expected": [[..]]}, ...]

The Rust cross-check (`src/rust/tests/phase1_mel_golden.rs`) loads this and
asserts the sequential `regression_deltas` matches bit-for-bit. Shapes include
T <= n edge cases (2x2 n=3) to pin the clamp-saturation + adim-includes-clamped-j
behavior.

Usage: uv run python scripts/extract_phase1_oracle_cases.py
"""

import json
from pathlib import Path

import numpy as np

from speech.features_oracle import regression_deltas_oracle

OUT = Path("tests/reference_data/phase1/deltas_cases.json")

# (T, W) shapes: a spread of tall/wide/degenerate plus the T<=n edge shape.
SHAPES = [(1, 2), (2, 3), (3, 1), (10, 4), (25, 3), (2, 2)]
N_VALUES = [1, 2, 3, 5]


def make_base(t: int, w: int) -> np.ndarray:
    return np.array([[((i * 7 + j * 13) % 100) / 100.0 for j in range(w)] for i in range(t)])


def main() -> None:
    cases = []
    for t, w in SHAPES:
        base = make_base(t, w)
        for n in N_VALUES:
            expected = regression_deltas_oracle(base, n)
            cases.append(
                {
                    "base": base.tolist(),
                    "n": n,
                    "expected": expected.tolist(),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1) + "\n")
    print(f"OK: wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
