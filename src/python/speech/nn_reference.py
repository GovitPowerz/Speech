"""Pure-Python BLSTM forward/backward oracle for gradient-check fixtures only.

Ported from legacy MATLAB: BLSTM_Forward.m, BLSTM_Backward.m, Hz2Mel.m, Mel2Hz.m,
and (Phase 2) the C++ engine LSTMLayer.cpp:312-421 and NeuronLayer.cpp:127-149. NOT a
production path -- used to validate the Rust engine (Phase 2-3).

The LSTM/dense oracles here are INDEPENDENT reimplementations: per-timestep SCALAR
Python loops with plain float accumulation, coded from the spec, NOT a
transliteration of the Rust block structure. They exist to cross-check the Rust
``LstmLayer``/``NeuronLayer`` from a second, structurally-different codebase.

IMPORTANT (bit-exactness caveat): the input projection and recurrence products are
accumulated with an explicit ascending Python ``for``-loop and a running float
``acc``, NOT ``numpy.dot`` / ``ndarray.sum`` (those use pairwise/blocked summation
that drifts by a few ULP from the engine's strictly-ascending order). Every
elementwise op is a Python float op. On the same platform/libm this reproduces the
engine's forward output; the JSON-fixture cross-check asserts bit-equality against
the Rust ``LstmLayer``/``NeuronLayer`` (both are same-libm local computations).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

# GatesFunction / Logistic saturate at +-expLimit = log(f64::MAX) (Log.hpp:193-195,
# ActivationFunctions.h:41-48,230-237). Computed, not hardcoded, so the boundary is
# the exact double the engine sees.
_EXP_LIMIT = math.log(np.finfo(np.float64).max)


def _gates_fn(x: float) -> float:
    """GatesFunction::fn (ActivationFunctions.h:230-237): sigmoid on 0.1*x, EXCLUSIVE
    saturation on the SCALED value (0.1*x < expLimit outer, 0.1*x > -expLimit inner;
    equality saturates)."""
    s = 0.1 * x
    if s < _EXP_LIMIT:
        if s > -_EXP_LIMIT:
            return 1.0 / (1.0 + math.exp(-s))
        return 0.0
    return 1.0


def _asinh(x: float) -> float:
    """Maxmin2::fn / Identity::fn (ActivationFunctions.h:158-160,206-208): std::asinh."""
    return math.asinh(x)


def _logistic_fn(x: float) -> float:
    """Logistic::fn (ActivationFunctions.h:41-48): plain sigmoid, INCLUSIVE
    saturation at x == +-expLimit (no 0.1 pre-scale, unlike GatesFunction)."""
    if x < _EXP_LIMIT:
        if x > -_EXP_LIMIT:
            return 1.0 / (1.0 + math.exp(-x))
        return 0.0
    return 1.0


def _matmul_seq_row(vec: list[float], mat: NDArray[np.float64]) -> list[float]:
    """Row-vector times matrix, k accumulated in strictly ascending order (matches the
    engine's ascending product contract; np.dot would drift a few ULP)."""
    k = len(vec)
    n = mat.shape[1]
    out = [0.0] * n
    for j in range(n):
        acc = 0.0
        for kk in range(k):
            acc += vec[kk] * float(mat[kk, j])
        out[j] = acc
    return out


def lstm_forward_oracle(
    input_w: NDArray[np.float64],
    feedback_w: NDArray[np.float64],
    peep: NDArray[np.float64],
    biases: NDArray[np.float64],
    x: NDArray[np.float64],
    flags: tuple[bool, bool, bool],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Independent scalar-loop oracle for LSTMLayer::feedForward (LSTMLayer.cpp:312-413).

    Args:
        input_w:    (I x 4O) input weights, gate column blocks [i|f|o|g].
        feedback_w: (O x 4O) recurrent weights, same blocks.
        peep:       (12 x O) peephole weights (rows: cells 0,1,2; gates 4,5,6,8,9,10;
                    gates-rec 3,7,11).
        biases:     (4O,) or (1 x 4O) bias row.
        x:          (T x cols) input sequence. Width tolerance: cols>I uses the left I
                    columns; cols<I uses the top cols weight rows (LSTMLayer.cpp:313-319).
        flags:      (cells_peep, gates_peep, gates_rec_peep).

    Returns:
        (y, gates, cells): y is (T x O) outputs; gates is (T x 4O) POST-activation gate
        cache; cells is (T x O) cell states.
    """
    cells_peep, gates_peep, gates_rec_peep = flags
    input_w = np.ascontiguousarray(input_w, dtype=np.float64)
    feedback_w = np.ascontiguousarray(feedback_w, dtype=np.float64)
    peep = np.ascontiguousarray(peep, dtype=np.float64)
    bias = np.ravel(np.asarray(biases, dtype=np.float64)).tolist()

    big_i = input_w.shape[0]
    o = input_w.shape[1] // 4
    t_len = x.shape[0]
    cols = x.shape[1]

    # Input projection with width tolerance (:313-319), ascending accumulation.
    if cols > big_i:
        proj_w = input_w
        used = big_i
    elif cols < big_i:
        proj_w = input_w[:cols, :]
        used = cols
    else:
        proj_w = input_w
        used = big_i

    gates = [[0.0] * (4 * o) for _ in range(t_len)]
    for t in range(t_len):
        row = [float(x[t, j]) for j in range(used)]
        proj = _matmul_seq_row(row, proj_w)
        for j in range(4 * o):
            gates[t][j] = proj[j] + bias[j]  # .rowwise() + biases

    cells = [[0.0] * o for _ in range(t_len)]
    cells_in = [[0.0] * o for _ in range(t_len)]
    y = [[0.0] * o for _ in range(t_len)]

    def pk(r: int, j: int) -> float:
        return float(peep[r, j])

    for t in range(t_len):
        if t == 0:
            # t=0 (:325-348): NO forget term, NO row-11 term.
            for j in range(2 * o):
                gates[t][j] = _gates_fn(gates[t][j])
            for j in range(o):
                gates[t][3 * o + j] = _asinh(gates[t][3 * o + j])
            for j in range(o):
                cells[t][j] = gates[t][j] * gates[t][3 * o + j]  # i_0 .* g_0
            # o extras IN ORDER: c_0*P2; i_0*P9; f_0*P10.
            if cells_peep:
                for j in range(o):
                    gates[t][2 * o + j] += cells[t][j] * pk(2, j)
            if gates_peep:
                for j in range(o):
                    gates[t][2 * o + j] += gates[t][j] * pk(9, j)
                for j in range(o):
                    gates[t][2 * o + j] += gates[t][o + j] * pk(10, j)
            for j in range(o):
                gates[t][2 * o + j] = _gates_fn(gates[t][2 * o + j])
            for j in range(o):
                cells_in[t][j] = _asinh(cells[t][j])
            for j in range(o):
                y[t][j] = gates[t][2 * o + j] * cells_in[t][j]
            continue

        # t>=1 (:350-412).
        rec = _matmul_seq_row(y[t - 1], feedback_w)  # y_{t-1} * FeedbackWeights
        for j in range(4 * o):
            gates[t][j] += rec[j]
        # cells peep into i,f (:353-356).
        if cells_peep:
            for j in range(o):
                gates[t][j] += cells[t - 1][j] * pk(0, j)
            for j in range(o):
                gates[t][o + j] += cells[t - 1][j] * pk(1, j)
        # gates-rec into i,f (:357-360).
        if gates_rec_peep:
            for j in range(o):
                gates[t][j] += gates[t - 1][j] * pk(3, j)
            for j in range(o):
                gates[t][o + j] += gates[t - 1][o + j] * pk(7, j)
        # gates peep, each ONE combined expression (:362-363).
        if gates_peep:
            for j in range(o):
                gates[t][j] += gates[t - 1][o + j] * pk(4, j) + gates[t - 1][2 * o + j] * pk(5, j)
            for j in range(o):
                gates[t][o + j] += gates[t - 1][j] * pk(6, j) + gates[t - 1][2 * o + j] * pk(8, j)
        # activate i,f (:370); activate g (:385).
        for j in range(2 * o):
            gates[t][j] = _gates_fn(gates[t][j])
        for j in range(o):
            gates[t][3 * o + j] = _asinh(gates[t][3 * o + j])
        # c_t = i_t*g_t + c_{t-1}*f_t (single combined expression, :387).
        for j in range(o):
            cells[t][j] = gates[t][j] * gates[t][3 * o + j] + cells[t - 1][j] * gates[t][o + j]
        # o extras IN ORDER: c_t*P2; i_t*P9; f_t*P10; o_{t-1}*P11 (:389-394).
        if cells_peep:
            for j in range(o):
                gates[t][2 * o + j] += cells[t][j] * pk(2, j)
        if gates_peep:
            for j in range(o):
                gates[t][2 * o + j] += gates[t][j] * pk(9, j)
            for j in range(o):
                gates[t][2 * o + j] += gates[t][o + j] * pk(10, j)
        if gates_rec_peep:
            for j in range(o):
                gates[t][2 * o + j] += gates[t - 1][2 * o + j] * pk(11, j)
        for j in range(o):
            gates[t][2 * o + j] = _gates_fn(gates[t][2 * o + j])
        for j in range(o):
            cells_in[t][j] = _asinh(cells[t][j])
        for j in range(o):
            y[t][j] = gates[t][2 * o + j] * cells_in[t][j]

    return (
        np.array(y, dtype=np.float64).reshape(t_len, o),
        np.array(gates, dtype=np.float64).reshape(t_len, 4 * o),
        np.array(cells, dtype=np.float64).reshape(t_len, o),
    )


def dense_forward_oracle(
    weights: NDArray[np.float64],
    biases: NDArray[np.float64],
    x: NDArray[np.float64],
    last_layer: bool,
) -> NDArray[np.float64]:
    """Independent scalar-loop oracle for NeuronLayer::feedForward (NeuronLayer.cpp:127-149).

    Args:
        weights: (I x O) projection matrix, no peepholes (unlike the LSTM gate blocks).
        biases:  (O,) or (1 x O) bias row.
        x:       (T x cols) input sequence. Width tolerance: cols>I uses the left I
                 columns; cols<I uses the top cols weight rows (:129-135).
        last_layer: True + O>1 -> UNSTABILIZED softmax (:138-142, no max-subtraction
                    overflow guard, sequential per-row sum); True + O==1 -> Logistic
                    (:144); False -> Maxmin2/asinh (:147).

    Returns:
        y: (T x O) outputs.
    """
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    bias = np.ravel(np.asarray(biases, dtype=np.float64)).tolist()

    big_i = weights.shape[0]
    o = weights.shape[1]
    t_len = x.shape[0]
    cols = x.shape[1]

    # Width-tolerant projection (:129-135), ascending accumulation (matches the
    # engine's matSeq contract, NOT numpy.dot).
    if cols > big_i:
        proj_w = weights
        used = big_i
    elif cols < big_i:
        proj_w = weights[:cols, :]
        used = cols
    else:
        proj_w = weights
        used = big_i

    pre_act = [[0.0] * o for _ in range(t_len)]
    for t in range(t_len):
        row = [float(x[t, j]) for j in range(used)]
        proj = _matmul_seq_row(row, proj_w)
        for j in range(o):
            pre_act[t][j] = proj[j] + bias[j]  # .rowwise() + biases, BEFORE activation

    y = [[0.0] * o for _ in range(t_len)]

    if last_layer and o > 1:
        # :138 exp(a+b), NO max-subtraction (unstabilized, load-bearing quirk).
        exp_out = [[math.exp(pre_act[t][j]) for j in range(o)] for t in range(t_len)]
        # :139 SEQUENTIAL per-row sum (ascending columns, not a reduction).
        row_sum = [0.0] * t_len
        for t in range(t_len):
            acc = 0.0
            for j in range(o):
                acc += exp_out[t][j]
            row_sum[t] = acc
        # :140-142 per-COLUMN cwiseQuotient by the row sum.
        for j in range(o):
            for t in range(t_len):
                y[t][j] = exp_out[t][j] / row_sum[t]
    elif last_layer:
        # O == 1: Logistic.
        for t in range(t_len):
            for j in range(o):
                y[t][j] = _logistic_fn(pre_act[t][j])
    else:
        # Maxmin2 (asinh).
        for t in range(t_len):
            for j in range(o):
                y[t][j] = _asinh(pre_act[t][j])

    return np.array(y, dtype=np.float64).reshape(t_len, o)


def blstm_forward(inputs: NDArray[np.float64], weights: dict[str, object]) -> NDArray[np.float64]:
    """Reference bidirectional-LSTM forward pass (Phase 2)."""
    ...
