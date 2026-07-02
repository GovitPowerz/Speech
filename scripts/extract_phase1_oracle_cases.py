"""Emit deterministic cross-language fixtures for the feature oracles.

Builds bases from a fixed closed-form (`((i*7 + j*13) % 100) / 100.0`) - NOT
`random`/`np.random` - at a spread of shapes and delta orders, runs each through
the numpy oracles, and writes three JSONs under `tests/reference_data/phase1/`:

  - `deltas_cases.json`: `[{"base": [[..]], "n": int, "expected": [[..]]}, ...]`
    for `regression_deltas_oracle` (MelFilterBank.cpp:153-170).
  - `sdc_cases.json`: `[{"mfcc": [[..]], "nb_dct": int, "expected": [[..]]}, ...]`
    for `sdc_oracle` (the SDC block stacking, MelFilterBank.cpp:224-262).
  - `ltsv_cases.json`: `[{"p": [[..]], "col": int, "freq_beg": int, "freq_end":
    int, "half_window": int, "expected": float}, ...]` for `ltsv_oracle`
    (`LongTermSpectralVariation::classifySequence`, :82-128), over the SAME
    closed-form 20x50 matrix as the harness's `synth_20x50.bin`, at the
    parameter grid R in {1,3,15} x band {(0,49),(5,20)} x cols {0,3,19}.

The Rust cross-checks (`src/rust/tests/phase1_mel_golden.rs`,
`phase1_ltsv_tdc_golden.rs`) load these and assert the sequential ports match
bit-for-bit. Deltas shapes include T <= n edge cases (2x2 n=3) to pin
clamp-saturation + adim-includes-clamped-j. SDC shapes straddle T = 21 (the
scratch is T + k*P = T + 21 tall, extracted at row 10) so both the T < 21
(extraction window overhangs the stacked blocks) and T > 21 regimes are
covered. LTSV cols 0/19 exercise the edge shrink at both ends of the 20-row
matrix; R=15 exercises full-window clamping (window size exceeds the matrix
height).

Usage: uv run python scripts/extract_phase1_oracle_cases.py
"""

import json
from pathlib import Path

import numpy as np

from speech.features_oracle import ltsv_oracle, regression_deltas_oracle, sdc_oracle

DELTAS_OUT = Path("tests/reference_data/phase1/deltas_cases.json")
SDC_OUT = Path("tests/reference_data/phase1/sdc_cases.json")
LTSV_OUT = Path("tests/reference_data/phase1/ltsv_cases.json")

# (T, W) shapes: a spread of tall/wide/degenerate plus the T<=n edge shape.
SHAPES = [(1, 2), (2, 3), (3, 1), (10, 4), (25, 3), (2, 2)]
N_VALUES = [1, 2, 3, 5]

# SDC (T, nb_dct) shapes straddling T=21: small T<21, T==21, T>21, plus a wider one.
SDC_SHAPES = [(4, 2), (10, 3), (21, 2), (25, 4), (40, 13)]

# LTSV parameter grid over the closed-form 20x50 matrix: R in {1,3,15} x
# band {(0,49),(5,20)} x cols {0,3,19}. R=15 exercises full-window clamping
# (window >= matrix height); cols 0/19 exercise the edge shrink at both ends.
LTSV_R_VALUES = [1, 3, 15]
LTSV_BANDS = [(0, 49), (5, 20)]
LTSV_COLS = [0, 3, 19]


def make_base(t: int, w: int) -> np.ndarray:
    return np.array([[((i * 7 + j * 13) % 100) / 100.0 for j in range(w)] for i in range(t)])


def main() -> None:
    deltas_cases = []
    for t, w in SHAPES:
        base = make_base(t, w)
        for n in N_VALUES:
            expected = regression_deltas_oracle(base, n)
            deltas_cases.append({"base": base.tolist(), "n": n, "expected": expected.tolist()})

    sdc_cases = []
    for t, nb_dct in SDC_SHAPES:
        mfcc = make_base(t, nb_dct)
        expected = sdc_oracle(mfcc, nb_dct)
        sdc_cases.append(
            {"mfcc": mfcc.tolist(), "nb_dct": nb_dct, "expected": expected.tolist()}
        )

    # Closed-form 20x50 matrix, matching the harness's synth_20x50.bin exactly.
    synth = make_base(20, 50)
    ltsv_cases = []
    for freq_beg, freq_end in LTSV_BANDS:
        for half_window in LTSV_R_VALUES:
            for col in LTSV_COLS:
                expected = ltsv_oracle(synth, col, freq_beg, freq_end, half_window)
                ltsv_cases.append(
                    {
                        "p": synth.tolist(),
                        "col": col,
                        "freq_beg": freq_beg,
                        "freq_end": freq_end,
                        "half_window": half_window,
                        "expected": expected,
                    }
                )

    DELTAS_OUT.parent.mkdir(parents=True, exist_ok=True)
    DELTAS_OUT.write_text(json.dumps(deltas_cases, indent=1) + "\n")
    SDC_OUT.write_text(json.dumps(sdc_cases, indent=1) + "\n")
    LTSV_OUT.write_text(json.dumps(ltsv_cases, indent=1) + "\n")
    print(f"OK: wrote {len(deltas_cases)} deltas cases to {DELTAS_OUT}")
    print(f"OK: wrote {len(sdc_cases)} sdc cases to {SDC_OUT}")
    print(f"OK: wrote {len(ltsv_cases)} ltsv cases to {LTSV_OUT}")


if __name__ == "__main__":
    main()
