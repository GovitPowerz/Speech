"""Emit tests/reference_data/phase2/{lstm,dense}_cases.json: the Python NN oracles'
forward outputs over deterministic case grids, for the Rust cross-check.

Each LSTM case fixes (I, O, T, flags) and a deterministic flat weight vector + input;
the Python ``lstm_forward_oracle`` (nn_reference.py) computes (y, gates, cells). Each
dense case fixes (I, O, T, last_layer, in_cols) similarly, with
``dense_forward_oracle`` computing y. The Rust cross-check tests
(phase2_layers_golden.rs) replay the SAME weights/input through ``LstmLayer`` /
``NeuronLayer`` and assert bit-equality (``to_bits``) against these outputs -- both
sides are same-libm local computations, so this is a bit-exact same-machine check
(distinct from the harness goldens, which pin the C++ op order).

All f64 values are stored as their raw u64 bit patterns (hex), so the round-trip
through JSON is exact (no decimal-string precision loss). Regenerate:

    uv run python scripts/extract_phase2_oracle_cases.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from speech.nn_reference import dense_forward_oracle, lstm_forward_oracle

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE2_DIR = REPO_ROOT / "tests" / "reference_data" / "phase2"

# (I, O) shapes; T grid; flag grid (cells, gates, gates_rec).
SHAPES = [(1, 1), (3, 2), (4, 3)]
T_GRID = [1, 2, 7]
FLAGS = {
    "allon": (True, True, True),
    "cells": (True, False, False),
    "gates": (False, True, False),
    "gatesrec": (False, False, True),
    "alloff": (False, False, False),
}


def _nb_of_weights(i: int, o: int) -> int:
    return 4 * i * o + 4 * o * o + 12 * o + 4 * o


# Dense (NeuronLayer) case grid: (I, O) shapes x T grid x last_layer x width variant.
# in_cols_delta: 0 (exact), +2 (wide -> leftCols), -1 (narrow -> topRows).
DENSE_SHAPES = [(4, 3), (4, 1), (2, 2), (3, 1)]
DENSE_T_GRID = [1, 6]
DENSE_WIDTH_DELTAS = [0, 2, -1]


def _dense_nb_of_weights(i: int, o: int) -> int:
    return o * (i + 1)


def _dense_unpack(flat: np.ndarray, i: int, o: int) -> tuple[np.ndarray, np.ndarray]:
    """Column-major unpack matching NeuronLayer::setWeights: weights (I x O), then
    the bias row (1 x O)."""
    pos = 0
    weights = flat[pos : pos + i * o].reshape(o, i).T.copy()
    pos += i * o
    bias = flat[pos : pos + o].reshape(1, o).copy()
    return weights, bias


def _synthetic_flat(n: int) -> np.ndarray:
    # Same formula as the harness Task 3/4 dumps: w[k] = ((k*11+3) % 97)/97 - 0.5.
    return np.array([((k * 11 + 3) % 97) / 97.0 - 0.5 for k in range(n)], dtype=np.float64)


def _synthetic_input(t_len: int, cols: int) -> np.ndarray:
    # Same formula as the harness Task 4 input: x[t,j] = ((t*37+j*53+7) % 101)/101 - 0.5.
    return np.array(
        [[((t * 37 + j * 53 + 7) % 101) / 101.0 - 0.5 for j in range(cols)] for t in range(t_len)],
        dtype=np.float64,
    )


def _unpack(flat: np.ndarray, i: int, o: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Column-major unpack matching LSTMLayer::setWeights: InputW (I x 4O),
    FeedbackW (O x 4O), Peep (12 x O), Bias (1 x 4O)."""
    pos = 0
    input_w = flat[pos : pos + i * 4 * o].reshape(4 * o, i).T.copy()
    pos += i * 4 * o
    feedback_w = flat[pos : pos + o * 4 * o].reshape(4 * o, o).T.copy()
    pos += o * 4 * o
    peep = flat[pos : pos + 12 * o].reshape(o, 12).T.copy()
    pos += 12 * o
    bias = flat[pos : pos + 4 * o].reshape(1, 4 * o).copy()
    return input_w, feedback_w, peep, bias


def _bits(v: float) -> str:
    return "0x" + format(struct.unpack("<Q", struct.pack("<d", float(v)))[0], "016x")


def _bits_matrix(m: np.ndarray) -> list[list[str]]:
    return [[_bits(m[r, c]) for c in range(m.shape[1])] for r in range(m.shape[0])]


def main() -> None:
    cases = []
    for i, o in SHAPES:
        nb = _nb_of_weights(i, o)
        flat = _synthetic_flat(nb)
        input_w, feedback_w, peep, bias = _unpack(flat, i, o)
        for t_len in T_GRID:
            x = _synthetic_input(t_len, i)
            for label, fl in FLAGS.items():
                y, gates, cells = lstm_forward_oracle(input_w, feedback_w, peep, bias, x, fl)
                cases.append(
                    {
                        "name": f"{i}x{o}_T{t_len}_{label}",
                        "I": i,
                        "O": o,
                        "T": t_len,
                        "flags": list(fl),
                        "weights_bits": [_bits(v) for v in flat],
                        "input_bits": _bits_matrix(x),
                        "y_bits": _bits_matrix(y),
                        "gates_bits": _bits_matrix(gates),
                        "cells_bits": _bits_matrix(cells),
                    }
                )

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    out = PHASE2_DIR / "lstm_cases.json"
    payload = {
        "text": (
            "Python lstm_forward_oracle (nn_reference.py) forward outputs over a "
            "deterministic (I,O,T,flags) grid. f64 stored as raw u64 bits (hex) for "
            "exact JSON round-trip. Rust replays the same weights/input through "
            "LstmLayer and asserts to_bits equality (same-libm same-machine check)."
        ),
        "weight_formula": "((k*11+3) % 97)/97 - 0.5",
        "input_formula": "((t*37 + j*53 + 7) % 101)/101 - 0.5",
        "cases": cases,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"OK: {len(cases)} cases -> {out.relative_to(REPO_ROOT)}")

    # --- dense_cases.json (NeuronLayer) --------------------------------------
    # Cover all three activation paths (hidden/asinh, softmax O>1, logistic O==1)
    # and both width mismatches (wide: cols=I+2 -> leftCols; narrow: cols=I-1 ->
    # topRows), deterministic via the same weight/input formulas as the harness.
    dense_cases = []
    for i, o in DENSE_SHAPES:
        nb = _dense_nb_of_weights(i, o)
        flat = _synthetic_flat(nb)
        weights, bias = _dense_unpack(flat, i, o)
        for t_len in DENSE_T_GRID:
            for delta in DENSE_WIDTH_DELTAS:
                in_cols = i + delta
                if in_cols < 1:
                    continue
                x = _synthetic_input(t_len, in_cols)
                for last_layer in (False, True):
                    if delta != 0 and last_layer:
                        continue  # width-mismatch cases are hidden-layer only (brief: lastLayer=false)
                    y = dense_forward_oracle(weights, bias, x, last_layer)
                    width_tag = {0: "exact", 2: "wide", -1: "narrow"}[delta]
                    dense_cases.append(
                        {
                            "name": f"{i}x{o}_T{t_len}_{width_tag}_{'last' if last_layer else 'hidden'}",
                            "I": i,
                            "O": o,
                            "T": t_len,
                            "in_cols": in_cols,
                            "last_layer": last_layer,
                            "weights_bits": [_bits(v) for v in flat],
                            "input_bits": _bits_matrix(x),
                            "y_bits": _bits_matrix(y),
                        }
                    )

    dense_out = PHASE2_DIR / "dense_cases.json"
    dense_payload = {
        "text": (
            "Python dense_forward_oracle (nn_reference.py) forward outputs over a "
            "deterministic (I,O,T,last_layer,in_cols) grid, covering the hidden "
            "(asinh), softmax (O>1), logistic (O==1), and width-mismatch (wide/"
            "narrow) paths. f64 stored as raw u64 bits (hex) for exact JSON "
            "round-trip. Rust replays the same weights/input through NeuronLayer "
            "and asserts to_bits equality (same-libm same-machine check)."
        ),
        "weight_formula": "((k*11+3) % 97)/97 - 0.5",
        "input_formula": "((t*37 + j*53 + 7) % 101)/101 - 0.5",
        "cases": dense_cases,
    }
    dense_out.write_text(json.dumps(dense_payload, indent=2) + "\n")
    print(f"OK: {len(dense_cases)} cases -> {dense_out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
