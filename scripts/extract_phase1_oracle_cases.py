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
  - `stats_cases.json`: `[{"chunks": [{"rows": [[..]]}, ...], "expected_mean":
    [..], "expected_std": [..], "expected_n": int}, ...]` for
    `stats_merge_oracle` (`InputStatistics.cpp:6-51`), chains of 1-4 deterministic
    chunks built from the closed-form `((i*7+j*13) % 100) / 100.0`, including
    chains with an empty chunk (`{"rows": []}`) mid-chain and a leading empty
    chunk.

The Rust cross-checks (`src/rust/tests/phase1_mel_golden.rs`,
`phase1_ltsv_tdc_golden.rs`, `phase1_stats_golden.rs`) load these and assert the
sequential ports match bit-for-bit. Deltas shapes include T <= n edge cases (2x2
n=3) to pin clamp-saturation + adim-includes-clamped-j. SDC shapes straddle T =
21 (the scratch is T + k*P = T + 21 tall, extracted at row 10) so both the T < 21
(extraction window overhangs the stacked blocks) and T > 21 regimes are
covered. LTSV cols 0/19 exercise the edge shrink at both ends of the 20-row
matrix; R=15 exercises full-window clamping (window size exceeds the matrix
height). Stats chains exercise every `update()` branch: empty-into-empty
(leading empty chunk), nonempty-into-empty no-op (mid-chain empty chunk), and
the n1,n2>0 pooled merge (chained multi-row chunks).

Usage: uv run python scripts/extract_phase1_oracle_cases.py
"""

import json
import math
from pathlib import Path

import numpy as np

from speech.features_oracle import (
    ltsv_oracle,
    regression_deltas_oracle,
    sdc_oracle,
    stats_merge_oracle,
    tdc_oracle,
)

DELTAS_OUT = Path("tests/reference_data/phase1/deltas_cases.json")
SDC_OUT = Path("tests/reference_data/phase1/sdc_cases.json")
LTSV_OUT = Path("tests/reference_data/phase1/ltsv_cases.json")
TDC_OUT = Path("tests/reference_data/phase1/tdc_cases.json")
STATS_OUT = Path("tests/reference_data/phase1/stats_cases.json")

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


# TDC windows: deterministic (no random). Chosen to exercise the branch matrix -
# constant (no crossing, CrossCorr=0), alternating sign (R[0]>0 oscillating ->
# multiple crossings, third-crossing cross-corr path), a sinusoid (>=3 crossings),
# a closed-form window, a min_lag=0 window (R[0]=1 -> fmath_log(0) path), and a
# window whose R[0] is negative (hits the R[0]<0 polarity branch).
def _sinusoid(n: int, periods: float) -> list[float]:
    return [math.sin(2.0 * math.pi * periods * i / n) for i in range(n)]


def _closed_form(n: int) -> list[float]:
    return [(((i * 7 + 3) % 11) - 5) / 5.0 for i in range(n)]


# Stats chains: lists of row-count-per-chunk, using the SAME closed-form base as
# synth_20x50.bin (make_base slices), with `None` marking an empty chunk. Row
# offsets accumulate across chunks so each chunk is a distinct closed-form slice
# (not overlapping data reused verbatim). Column width fixed at 4.
# Chains cover: single chunk (no merge), two nonempty chunks (n1,n2>0 pooled
# merge), a mid-chain empty chunk (nonempty.update(empty) no-op), a LEADING
# empty chunk (empty.update(empty) then empty.update(nonempty) copy), and a
# longer 4-chunk chain mixing empties.
STATS_CHAIN_SPECS: list[list[int | None]] = [
    [3],  # single batch, no merge
    [2, 1],  # two-chunk merge (mirrors the anchor test)
    [4, 5],  # two-chunk merge, larger
    [3, None],  # mid-chain empty: nonempty.update(empty) no-op
    [None, 3],  # leading empty: empty.update(empty) then empty.update(nonempty)
    [None, 2, 3],  # leading empty + two real merges
    [2, None, 3, 1],  # 4-chunk chain mixing an empty mid-chain
    [1, 1, 1, 1],  # 4 singleton-row chunks, all n1,n2>0 merges
]
STATS_WIDTH = 4


# (window, min_lag, max_lag, balance)
TDC_CASES = [
    ([1.0] * 12, 1, 5, 0.7),  # constant: no crossing
    ([1.0 if i % 2 == 0 else -1.0 for i in range(20)], 1, 10, 0.7),  # alternating
    (_sinusoid(48, 3.0), 1, 24, 0.7),  # sinusoid, several periods -> >=3 crossings
    (_sinusoid(64, 5.0), 2, 30, 0.5),  # more periods -> more crossings
    (_closed_form(30), 1, 12, 0.7),  # closed-form deterministic
    (_closed_form(25), 0, 10, 0.7),  # min_lag=0 -> R[0]=1
    ([math.cos(math.pi * i) * (i + 1) for i in range(16)], 1, 8, 0.3),  # crafted
    ([(-1.0) ** i * 0.5 for i in range(18)], 0, 9, 0.9),  # min_lag=0 alternating
]


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

    tdc_cases = []
    for window, min_lag, max_lag, balance in TDC_CASES:
        expected = tdc_oracle(window, min_lag, max_lag, balance)
        tdc_cases.append(
            {
                "window": list(window),
                "min_lag": min_lag,
                "max_lag": max_lag,
                "balance": balance,
                "expected": expected,
            }
        )

    stats_cases = []
    row_offset = 0
    for spec in STATS_CHAIN_SPECS:
        chunks_json = []
        oracle_chunks: list[list[list[float]]] = []
        for nrows in spec:
            if nrows is None:
                chunks_json.append({"rows": []})
                oracle_chunks.append([])
                continue
            # Shift row indices by row_offset so each chunk draws a distinct
            # closed-form slice (same formula as synth_20x50.bin / make_base)
            # across the whole chain, rather than repeating identical rows.
            rows = [
                [((i + row_offset) * 7 + j * 13) % 100 / 100.0 for j in range(STATS_WIDTH)] for i in range(nrows)
            ]
            row_offset += nrows
            chunks_json.append({"rows": rows})
            oracle_chunks.append(rows)
        mean, std, n = stats_merge_oracle(oracle_chunks)
        stats_cases.append(
            {
                "chunks": chunks_json,
                "expected_mean": mean,
                "expected_std": std,
                "expected_n": n,
            }
        )

    DELTAS_OUT.parent.mkdir(parents=True, exist_ok=True)
    DELTAS_OUT.write_text(json.dumps(deltas_cases, indent=1) + "\n")
    SDC_OUT.write_text(json.dumps(sdc_cases, indent=1) + "\n")
    LTSV_OUT.write_text(json.dumps(ltsv_cases, indent=1) + "\n")
    TDC_OUT.write_text(json.dumps(tdc_cases, indent=1) + "\n")
    STATS_OUT.write_text(json.dumps(stats_cases, indent=1) + "\n")
    print(f"OK: wrote {len(deltas_cases)} deltas cases to {DELTAS_OUT}")
    print(f"OK: wrote {len(sdc_cases)} sdc cases to {SDC_OUT}")
    print(f"OK: wrote {len(ltsv_cases)} ltsv cases to {LTSV_OUT}")
    print(f"OK: wrote {len(tdc_cases)} tdc cases to {TDC_OUT}")
    print(f"OK: wrote {len(stats_cases)} stats cases to {STATS_OUT}")


if __name__ == "__main__":
    main()
