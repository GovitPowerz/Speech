"""Emit tests/reference_data/phase3/backward_cases.json: the deterministic backward
oracle-cases (weights + inputs + deltas + expected Nx2 derivs) the Phase 3 Task 9
nn_reference.py backward cross-check consumes.

Mirrors extract_phase2_oracle_cases.py in spirit, but the EXPECTED derivs are read
from the harness-produced reimpl goldens (tests/reference_data/phase3/bwd_*.bin,
written by extract_phase3_fixtures.py) rather than computed by a Python oracle here --
the Python backward oracles (lstm_backward_oracle/dense_backward_oracle) land in Task 9
and will be cross-checked bit-for-bit against these committed expected derivs (the same
strong triangulation the Phase 2 forward oracle used: harness reimpl == Python oracle ==
Rust layer). The standalone LSTM/dense cases use the SAME deterministic weight/input/
delta formulas the harness main.cpp uses, so a Task 9 oracle replaying them must
reproduce the committed derivs.

All f64 values are stored as raw u64 bit patterns (hex) for exact JSON round-trip.
Regenerate (AFTER extract_phase3_fixtures.py, which writes the bwd_*.bin goldens):

    uv run python scripts/extract_phase3_oracle_cases.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE3_DIR = REPO_ROOT / "tests" / "reference_data" / "phase3"

# Standalone LSTM backward cases: (I, O, T, deriv-golden basename). Only the I=3 case
# is dumped by the harness (bwd_lstm_derivs.bin); the I=5 case is measured in the same
# NN_TOL but not dumped, so only the dumped case gets an expected-derivs golden here.
LSTM_CASES = [(3, 2, 7, "bwd_lstm_derivs.bin")]

# Standalone dense backward cases: (I, O, T, last_layer, deriv-golden basename).
DENSE_CASES = [
    (4, 3, 6, False, "bwd_dense_hidden_derivs.bin"),
    (4, 3, 6, True, "bwd_dense_last_derivs.bin"),
    (4, 1, 6, True, "bwd_dense_logistic_derivs.bin"),
]


def _synth_flat(n: int) -> np.ndarray:
    # main.cpp synthFlat: w[k] = ((k*11+3) % 97)/97 - 0.5.
    return np.array([((k * 11 + 3) % 97) / 97.0 - 0.5 for k in range(n)], dtype=np.float64)


def _synth_input(t_len: int, cols: int) -> np.ndarray:
    # main.cpp makeInput: x[t,j] = ((t*37 + j*53 + 7) % 101)/101 - 0.5.
    return np.array(
        [[((t * 37 + j * 53 + 7) % 101) / 101.0 - 0.5 for j in range(cols)] for t in range(t_len)],
        dtype=np.float64,
    )


def _synth_deltas(t_len: int, o: int) -> np.ndarray:
    # main.cpp makeDeltas: d[t,j] = ((t*29 + j*41 + 5) % 83)/83 - 0.5.
    return np.array(
        [[((t * 29 + j * 41 + 5) % 83) / 83.0 - 0.5 for j in range(o)] for t in range(t_len)],
        dtype=np.float64,
    )


def _lstm_nb(i: int, o: int) -> int:
    return 4 * i * o + 4 * o * o + 12 * o + 4 * o


def _dense_nb(i: int, o: int) -> int:
    return o * (i + 1)


def _bits(v: float) -> str:
    return "0x" + format(struct.unpack("<Q", struct.pack("<d", float(v)))[0], "016x")


def _bits_matrix(m: np.ndarray) -> list[list[str]]:
    return [[_bits(m[r, c]) for c in range(m.shape[1])] for r in range(m.shape[0])]


def _read_bin(path: Path) -> np.ndarray:
    """Read a legacy .bin (i64 rows, i64 cols, column-major f64) into an (rows,cols) array."""
    with path.open("rb") as f:
        rows = int.from_bytes(f.read(8), "little", signed=True)
        cols = int.from_bytes(f.read(8), "little", signed=True)
        flat = np.frombuffer(f.read(rows * cols * 8), dtype="<f8")
    return flat.reshape((cols, rows)).T.copy()


def main() -> None:
    lstm_cases = []
    for i, o, t_len, golden in LSTM_CASES:
        flat = _synth_flat(_lstm_nb(i, o))
        x = _synth_input(t_len, i)
        d = _synth_deltas(t_len, o)
        derivs = _read_bin(PHASE3_DIR / golden)  # Nx2: col0 deriv, col1 count
        lstm_cases.append(
            {
                "name": f"{i}x{o}_T{t_len}",
                "I": i,
                "O": o,
                "T": t_len,
                "peepholes": [True, True, True],
                "weights_bits": [_bits(v) for v in flat],
                "input_bits": _bits_matrix(x),
                "deltas_bits": _bits_matrix(d),
                "expected_derivs_bits": _bits_matrix(derivs),
                "golden": golden,
            }
        )

    dense_cases = []
    for i, o, t_len, last, golden in DENSE_CASES:
        flat = _synth_flat(_dense_nb(i, o))
        x = _synth_input(t_len, i)
        d = _synth_deltas(t_len, o)
        derivs = _read_bin(PHASE3_DIR / golden)
        dense_cases.append(
            {
                "name": f"{i}x{o}_T{t_len}_{'last' if last else 'hidden'}",
                "I": i,
                "O": o,
                "T": t_len,
                "last_layer": last,
                "weights_bits": [_bits(v) for v in flat],
                "input_bits": _bits_matrix(x),
                "deltas_bits": _bits_matrix(d),
                "expected_derivs_bits": _bits_matrix(derivs),
                "golden": golden,
            }
        )

    payload = {
        "text": (
            "Phase 3 Task 1 backward oracle-cases: deterministic (weights, input, deltas) "
            "triples + the harness-reimpl-produced expected Nx2 derivs (col0 = summed "
            "derivative, col1 = frame count) for the standalone LSTMLayer::feedBackward and "
            "NeuronLayer::feedBackward paths. f64 stored as raw u64 bits (hex) for exact JSON "
            "round-trip. Task 9's Python backward oracles (nn_reference.py) replay the same "
            "weights/input/deltas and assert to_bits equality against expected_derivs_bits; "
            "the expected derivs are ALSO the harness goldens the Rust layers assert against, "
            "so harness-reimpl == Python-oracle == Rust-layer triangulate the backward."
        ),
        "weight_formula": "((k*11+3) % 97)/97 - 0.5",
        "input_formula": "((t*37 + j*53 + 7) % 101)/101 - 0.5",
        "deltas_formula": "((t*29 + j*41 + 5) % 83)/83 - 0.5",
        "lstm_cases": lstm_cases,
        "dense_cases": dense_cases,
    }
    out = PHASE3_DIR / "backward_cases.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"OK: {len(lstm_cases)} lstm + {len(dense_cases)} dense backward cases -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
