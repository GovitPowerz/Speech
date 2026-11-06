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


def _gates_deriv(y: float) -> float:
    """GatesFunction derivative (ActivationFunctions.h:240-242): `0.1*y*(1-y)`, taking
    the POST-activation cached value `y` (the engine backward reads the cached gate,
    never recomputes sigmoid from the pre-activation z; LSTMLayer.cpp:566 etc read
    `_Gates` which already holds the activated value)."""
    return 0.1 * y * (1.0 - y)


def _asinh_deriv(y: float) -> float:
    """Maxmin2/Identity derivative (ActivationFunctions.h:162-165): `1/sqrt(1+sinh(y)^2)`,
    taking the POST-activation cached value `y` (`asinh(z)` s.t. `sinh(y) == z`) --
    the engine backward reads `_CellsIn`/`_Gates` (already `asinh`-activated) and
    applies this form directly, never re-deriving from the pre-activation argument."""
    s = math.sinh(y)
    return 1.0 / math.sqrt(1.0 + s * s)


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


def lstm_backward_oracle(
    input_w: NDArray[np.float64],
    feedback_w: NDArray[np.float64],
    peep: NDArray[np.float64],
    x: NDArray[np.float64],
    output: NDArray[np.float64],
    gates: NDArray[np.float64],
    cells: NDArray[np.float64],
    deltas: NDArray[np.float64],
    flags: tuple[bool, bool, bool],
    inv_sub_sampling_ratio: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Independent scalar-loop oracle for `LSTMLayer::feedBackward` (LSTMLayer.cpp:518-723,
    same source lines as the Rust `nn/layers.rs::LstmLayer::feed_backward`). ENGINE
    semantics ONLY -- NOT `BLSTM_Backward.m` (structure-reference only per the P2b T11
    verdict: that file pairs with the divergent MATLAB forward, sigmoid(x) gates, no
    softmax, internal /ncols normalization, and must not be ported).

    `cells_in` (`_CellsIn`, the asinh-activated cell cache) is recomputed here as
    `asinh(cells)` -- the identical deterministic op the forward oracle applied
    (`lstm_forward_oracle:` `cells_in[t][j] = _asinh(cells[t][j])`), not threaded as a
    fourth forward return, since it is a pure function of the already-threaded `cells`.

    Reverse-time loop (`row = T-1 .. 0`); per row the gate-derivative order is
    O -> state -> C -> F -> I (LSTMLayer.cpp:582/613/616/624/659), recomputed FROM the
    POST-activation caches (`gates`/`cells`/`cells_in`, i.e. the forward oracle's
    return values) via `_gates_deriv`/`_asinh_deriv` taking the cached activated value,
    per ActivationFunctions.h:162-165,240-242. Peephole cross-terms gated by `flags`
    (cells_peep, gates_peep, gates_rec_peep) at the SAME rows as the forward
    (0/1/2 cells; 4/5/6/8/9/10 gates; 3/7/11 gates-rec). All products (the two
    NN-wide-k `(FeedbackWeights*test)` / `(InputWeights*testM)` GEMMs, LSTMLayer.cpp:
    703/707) via `_matmul_seq_row` (plain ascending loops -- numpy pairwise summation
    drifts a few ULP from the engine's strictly-ascending accumulation).

    Args:
        x:      (T x cols) the SAME (possibly width-mismatched) input the forward
                pass consumed. Width tolerance mirrors the forward AND the Rust
                backward (`:534-543`): cols>I -> left I columns; cols<I -> zero-PAD
                columns up to I (the padded columns' input-weight-deriv rows then
                accumulate exactly 0, spec S11.5).
        output: (T x O) the layer's own forward output `y` (read at row-1 for the
                recurrent deriv blocks).
        deltas: (T x O) the seed deltas fed BACKWARD into this layer (`deltas.rows()`
                sets `linesNb`/the frame count harvested at col1).
        inv_sub_sampling_ratio: LSTMLayer.cpp:710-715 -- scales all four deriv blocks
                when > 1.

    Returns:
        (derivs, deltas_previous_layer): `derivs` is Nx2 (col0 summed derivative,
        col1 `linesNb` REPLICATED per element) in the flat weight-packer block order
        (InputWeights -> FeedbackWeights -> PeepWeight -> Biaises, each COLUMN-major,
        matching `LstmLayer::get_weights_derivatives`); `deltas_previous_layer` is
        (T x I).
    """
    cells_peep, gates_peep, gates_rec_peep = flags
    big_i = input_w.shape[0]
    o = input_w.shape[1] // 4
    cols = x.shape[1]
    lines_nb = deltas.shape[0]
    cells_in = np.asinh(cells)

    if cols > big_i:
        recon_input = x[:, :big_i]
    elif cols < big_i:
        padded = np.zeros((lines_nb, big_i), dtype=np.float64)
        padded[:, :cols] = x  # shape-checked assignment, mirrors the Rust .assign() panic
        recon_input = padded
    else:
        recon_input = x

    def p(r: int, j: int) -> float:
        return float(peep[r, j])

    iw_d = [[0.0] * (4 * o) for _ in range(big_i)]
    fw_d = [[0.0] * (4 * o) for _ in range(o)]
    pp_d = [[0.0] * o for _ in range(12)]
    bs_d = [0.0] * (4 * o)

    deltas_feedback = [0.0] * o
    deltas_input_gate = [0.0] * o
    deltas_forget_gate = [0.0] * o
    deltas_output_gate = [0.0] * o
    deltas_cells = [0.0] * o
    epsilon_state = [0.0] * o

    test_m = [[0.0] * lines_nb for _ in range(4 * o)]  # 4O x T

    for row in range(lines_nb - 1, -1, -1):
        last_row = row == lines_nb - 1

        # :580 epsilonCells = deltas.row(row) + deltasFeedback.
        epsilon_cells = [float(deltas[row, j]) + deltas_feedback[j] for j in range(o)]

        # :582-585 deltasOutputGate.
        tmp = [float(cells_in[row, j]) * epsilon_cells[j] for j in range(o)]
        if gates_peep:
            for j in range(o):
                tmp[j] += deltas_input_gate[j] * p(5, j) + deltas_forget_gate[j] * p(8, j)
        if gates_rec_peep:
            for j in range(o):
                tmp[j] += deltas_output_gate[j] * p(11, j)
        for j in range(o):
            deltas_output_gate[j] = _gates_deriv(float(gates[row, 2 * o + j])) * tmp[j]

        # :595-605 output-gate deriv accumulation.
        for ii in range(big_i):
            for j in range(o):
                iw_d[ii][2 * o + j] += float(recon_input[row, ii]) * deltas_output_gate[j]
        if row > 0:
            for ii in range(o):
                for j in range(o):
                    fw_d[ii][2 * o + j] += float(output[row - 1, ii]) * deltas_output_gate[j]
            if gates_rec_peep:
                for j in range(o):
                    pp_d[11][j] += deltas_output_gate[j] * float(gates[row - 1, 2 * o + j])
        if cells_peep:
            for j in range(o):
                pp_d[2][j] += deltas_output_gate[j] * float(cells[row, j])
        if gates_peep:
            for j in range(o):
                pp_d[9][j] += deltas_output_gate[j] * float(gates[row, j])
            for j in range(o):
                pp_d[10][j] += deltas_output_gate[j] * float(gates[row, o + j])
        for j in range(o):
            bs_d[2 * o + j] += deltas_output_gate[j]

        # :607-611 epsilonForget from the NEXT step's forget gate .* NEXT epsilonState.
        epsilon_forget = [0.0] * o
        if not last_row:
            for j in range(o):
                epsilon_forget[j] = float(gates[row + 1, o + j]) * epsilon_state[j]

        # :613-614 epsilonState.
        tmp_epsilon_state = [0.0] * o
        if cells_peep:
            for j in range(o):
                tmp_epsilon_state[j] = deltas_input_gate[j] * p(0, j) + deltas_forget_gate[j] * p(1, j) + deltas_output_gate[j] * p(2, j)
        for j in range(o):
            epsilon_state[j] = (
                float(gates[row, 2 * o + j]) * _asinh_deriv(float(cells_in[row, j])) * epsilon_cells[j] + epsilon_forget[j] + tmp_epsilon_state[j]
            )

        # :616 deltasCells.
        for j in range(o):
            deltas_cells[j] = float(gates[row, j]) * _asinh_deriv(float(gates[row, 3 * o + j])) * epsilon_state[j]
        for ii in range(big_i):
            for j in range(o):
                iw_d[ii][3 * o + j] += float(recon_input[row, ii]) * deltas_cells[j]
        if row > 0:
            for ii in range(o):
                for j in range(o):
                    fw_d[ii][3 * o + j] += float(output[row - 1, ii]) * deltas_cells[j]
        for j in range(o):
            bs_d[3 * o + j] += deltas_cells[j]

        # :624 save the PRIOR forget-gate delta (read at :660 by the input gate).
        deltas_forget_gate_tmp = list(deltas_forget_gate)

        # :624-657 deltasForgetGate.
        if row > 0:
            tmp_f = [float(cells[row - 1, j]) * epsilon_state[j] for j in range(o)]
            if gates_peep:
                for j in range(o):
                    tmp_f[j] += deltas_input_gate[j] * p(4, j) + deltas_output_gate[j] * p(10, j)
            if gates_rec_peep:
                for j in range(o):
                    tmp_f[j] += deltas_forget_gate[j] * p(7, j)
            for j in range(o):
                deltas_forget_gate[j] = _gates_deriv(float(gates[row, o + j])) * tmp_f[j]
            for ii in range(big_i):
                for j in range(o):
                    iw_d[ii][o + j] += float(recon_input[row, ii]) * deltas_forget_gate[j]
            for ii in range(o):
                for j in range(o):
                    fw_d[ii][o + j] += float(output[row - 1, ii]) * deltas_forget_gate[j]
            if cells_peep:
                for j in range(o):
                    pp_d[1][j] += deltas_forget_gate[j] * float(cells[row - 1, j])
            if gates_peep:
                for j in range(o):
                    pp_d[6][j] += deltas_forget_gate[j] * float(gates[row - 1, j])
                for j in range(o):
                    pp_d[8][j] += deltas_forget_gate[j] * float(gates[row - 1, 2 * o + j])
            if gates_rec_peep:
                for j in range(o):
                    pp_d[7][j] += deltas_forget_gate[j] * float(gates[row - 1, o + j])
            for j in range(o):
                bs_d[o + j] += deltas_forget_gate[j]
        else:
            # :648-657 row==0: no cellstate term.
            tmp_f = [0.0] * o
            if gates_peep:
                for j in range(o):
                    tmp_f[j] += deltas_input_gate[j] * p(4, j) + deltas_output_gate[j] * p(10, j)
            if gates_rec_peep:
                for j in range(o):
                    tmp_f[j] += deltas_forget_gate[j] * p(7, j)
            for j in range(o):
                deltas_forget_gate[j] = _gates_deriv(float(gates[row, o + j])) * tmp_f[j]
            for ii in range(big_i):
                for j in range(o):
                    iw_d[ii][o + j] += float(recon_input[row, ii]) * deltas_forget_gate[j]
            for j in range(o):
                bs_d[o + j] += deltas_forget_gate[j]

        # :659-682 deltasInputGate.
        tmp_i = [float(gates[row, 3 * o + j]) * epsilon_state[j] for j in range(o)]
        if gates_peep:
            for j in range(o):
                tmp_i[j] += deltas_forget_gate_tmp[j] * p(6, j) + deltas_output_gate[j] * p(9, j)
        if gates_rec_peep:
            for j in range(o):
                tmp_i[j] += deltas_input_gate[j] * p(3, j)
        for j in range(o):
            deltas_input_gate[j] = _gates_deriv(float(gates[row, j])) * tmp_i[j]
        for ii in range(big_i):
            for j in range(o):
                iw_d[ii][j] += float(recon_input[row, ii]) * deltas_input_gate[j]
        if row > 0:
            for ii in range(o):
                for j in range(o):
                    fw_d[ii][j] += float(output[row - 1, ii]) * deltas_input_gate[j]
            if cells_peep:
                for j in range(o):
                    pp_d[0][j] += deltas_input_gate[j] * float(cells[row - 1, j])
            if gates_peep:
                for j in range(o):
                    pp_d[4][j] += deltas_input_gate[j] * float(gates[row - 1, o + j])
                for j in range(o):
                    pp_d[5][j] += deltas_input_gate[j] * float(gates[row - 1, 2 * o + j])
            if gates_rec_peep:
                for j in range(o):
                    pp_d[3][j] += deltas_input_gate[j] * float(gates[row - 1, j])
        for j in range(o):
            bs_d[j] += deltas_input_gate[j]

        # :688-690 stack [i|f|o|c] into testM.col(row).
        for j in range(o):
            test_m[j][row] = deltas_input_gate[j]
            test_m[o + j][row] = deltas_forget_gate[j]
            test_m[2 * o + j][row] = deltas_output_gate[j]
            test_m[3 * o + j][row] = deltas_cells[j]

        # :703 deltasFeedback = (_FeedbackWeights * test)^T, test = testM.col(row).
        test_col = [test_m[r][row] for r in range(4 * o)]
        for j in range(o):
            acc = 0.0
            for kk in range(4 * o):
                acc += float(feedback_w[j, kk]) * test_col[kk]
            deltas_feedback[j] = acc

    # :707 deltasPreviousLayer = (_InputWeights * testM)^T -> T x I.
    deltas_previous_layer = np.zeros((lines_nb, big_i), dtype=np.float64)
    for ii in range(big_i):
        for row in range(lines_nb):
            acc = 0.0
            for kk in range(4 * o):
                acc += float(input_w[ii, kk]) * test_m[kk][row]
            deltas_previous_layer[row, ii] = acc

    # :710-715 invSubSamplingRatio > 1 scales all four deriv blocks.
    if inv_sub_sampling_ratio > 1:
        r = float(inv_sub_sampling_ratio)
        for ii in range(big_i):
            for j in range(4 * o):
                iw_d[ii][j] *= r
        for ii in range(o):
            for j in range(4 * o):
                fw_d[ii][j] *= r
        for rr in range(12):
            for j in range(o):
                pp_d[rr][j] *= r
        for j in range(4 * o):
            bs_d[j] *= r

    # Nx2 harvest, block order InputWeights -> FeedbackWeights -> PeepWeight -> Biaises,
    # each COLUMN-major (jj outer over cols, ii inner over rows) -- matches
    # `LstmLayer::get_weights_derivatives` (layers.rs:754-773).
    count = float(lines_nb)
    rows_out: list[list[float]] = []
    for jj in range(4 * o):
        for ii in range(big_i):
            rows_out.append([iw_d[ii][jj], count])
    for jj in range(4 * o):
        for ii in range(o):
            rows_out.append([fw_d[ii][jj], count])
    for jj in range(o):
        for ii in range(12):
            rows_out.append([pp_d[ii][jj], count])
    for jj in range(4 * o):
        rows_out.append([bs_d[jj], count])

    derivs = np.array(rows_out, dtype=np.float64).reshape(len(rows_out), 2)
    return derivs, deltas_previous_layer


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


def dense_backward_oracle(
    weights: NDArray[np.float64],
    x: NDArray[np.float64],
    deltas: NDArray[np.float64],
    last_layer: bool,
    inv_sub_sampling_ratio: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Independent scalar-loop oracle for `NeuronLayer::feedBackward`
    (NeuronLayer.cpp:151-207, same source lines as the Rust `nn/layers.rs::NeuronLayer::
    feed_backward`). ENGINE semantics -- NOT `BLSTM_Backward.m` (P2b T11 no-port verdict).

    - `weightsDerivatives += input.col(jj)*deltas.row(jj)` per row (`:178-182`, a plain
      ascending-loop outer product per row, NOT a single GEMM); `biaisesDerivatives +=
      deltas.colwise().sum()` (`:183`).
    - `_NbOfSeqFedBackward += InputSeq.rows()` (`:199` -- the ORIGINAL `x`'s row count,
      NOT `deltas.rows()`).
    - `lastLayer`: `deltas_out = deltas * weights^T`, NO activation derivative (the
      double-count trap, spec S5/R3 -- fusion lives in CostLaw, applying a Jacobian
      here would double-count it).
    - hidden: `deltas_out = (deltas * weights^T) .* Maxmin2'(x)`, the asinh-deriv taken
      on the RAW `x` parameter (`:204`), i.e. the UN-width-reconstructed input.

    Args:
        x: (T x cols) the layer's raw input (width tolerance mirrors the forward:
           cols>I -> left I columns feed the deriv accumulation; cols<I -> zero-padded).
        deltas: (T x O) seed deltas fed backward into this layer.

    Returns:
        (derivs, deltas_out): `derivs` is Nx2 (col0 summed derivative, col1
        `x.shape[0]` REPLICATED per element), block order weights (COLUMN-major) then
        bias -- matches `NeuronLayer::get_weights_derivatives` (layers.rs:1063-1079).
        `deltas_out` is (T x I).
    """
    big_i = weights.shape[0]
    o = weights.shape[1]
    t = x.shape[0]
    cols = x.shape[1]
    n_deltas = deltas.shape[0]

    if cols > big_i:
        recon_input = x[:, :big_i]
    elif cols < big_i:
        padded = np.zeros((t, big_i), dtype=np.float64)
        padded[:, :cols] = x
        recon_input = padded
    else:
        recon_input = x

    # :178-182 weightsDerivatives += input.col(jj)*deltas.row(jj) per row (ascending
    # outer-product accumulation, not a single GEMM).
    w_d = [[0.0] * o for _ in range(big_i)]
    for jj in range(n_deltas):
        for ii in range(big_i):
            in_val = float(recon_input[jj, ii])
            for j in range(o):
                w_d[ii][j] += in_val * float(deltas[jj, j])

    # :183 biaisesDerivatives += deltas.colwise().sum() (ascending row sum per column).
    b_d = [0.0] * o
    for j in range(o):
        acc = 0.0
        for jj in range(n_deltas):
            acc += float(deltas[jj, j])
        b_d[j] = acc

    if inv_sub_sampling_ratio > 1:
        r = float(inv_sub_sampling_ratio)
        for ii in range(big_i):
            for j in range(o):
                w_d[ii][j] *= r
        for j in range(o):
            b_d[j] *= r

    # :204 deltas_out = (deltas * weights^T) .* Maxmin2'(x) [hidden] or plain (last).
    # Ascending-k product (weights^T is O rows x I cols read column-major from `weights`).
    deltas_out = np.zeros((n_deltas, big_i), dtype=np.float64)
    for jj in range(n_deltas):
        for ii in range(big_i):
            acc = 0.0
            for kk in range(o):
                acc += float(deltas[jj, kk]) * float(weights[ii, kk])
            deltas_out[jj, ii] = acc
    if not last_layer:
        # :204 literally indexes the RAW `x` parameter at [jj, ii] with NO width
        # reconstruction (unlike the :178-182 accumulation half above, which uses
        # `recon_input`) -- only well-defined when `x.shape[1] >= big_i`, true on
        # every live legacy call site (BLSTMNeuralNetwork.h:29/.cpp:90 always feeds
        # cols == I); an out-of-range `cols < big_i` call mirrors the Rust panic.
        for jj in range(n_deltas):
            for ii in range(big_i):
                deltas_out[jj, ii] *= _asinh_deriv(float(x[jj, ii]))

    # Nx2 harvest: weights COLUMN-major (jj outer over cols, ii inner over rows), then
    # bias -- matches `NeuronLayer::get_weights_derivatives` (layers.rs:1069-1079).
    count = float(t)
    rows_out: list[list[float]] = []
    for jj in range(o):
        for ii in range(big_i):
            rows_out.append([w_d[ii][jj], count])
    for jj in range(o):
        rows_out.append([b_d[jj], count])

    derivs = np.array(rows_out, dtype=np.float64).reshape(len(rows_out), 2)
    return derivs, deltas_out


def _sub_sample(ratio: int, m: NDArray[np.float64]) -> NDArray[np.float64]:
    """NeuralNetwork<L>::SubSample (NeuralNetwork.hpp:125-134): T x C -> floor(T/R) x
    C*R, frame-contiguous temporal stacking, trailing T mod R rows dropped."""
    t_len, c = m.shape
    sub_len = t_len // ratio
    out = np.zeros((sub_len, c * ratio), dtype=np.float64)
    for jj in range(sub_len):
        for kk in range(ratio):
            out[jj, kk * c : (kk + 1) * c] = m[jj * ratio + kk, :]
    return out


def _lstm_net_forward(
    layers: list[dict[str, object]], sub_sampling: list[int], x: NDArray[np.float64], reverse: bool
) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    """NeuralNetwork<LSTMLayer>::feedForward / feedForwardReverse over a stack of LSTM
    layers with per-layer sub-sampling (NeuralNetwork.hpp:158-240). Each layer's input
    is sub-sampled first when its ratio > 1; the reversal lives inside the layer.

    Per `LstmLayer::feed_forward_reverse` (layers.rs:354-364): the layer reverses its
    OWN input, runs the forward kernel, then UN-REVERSES the returned `output` before
    handing it back to the container -- so `output`/`layers_output` at the container
    level are ALWAYS in forward-time order, whether or not `reverse` is set (risk R9
    only applies to the layer's INTERNAL `gates`/`cells` caches, which stay
    reversed-time and are never un-reversed).

    Returns `(output, caches)` in forward-time order throughout. `caches[jj]` retains
    this layer's `input` (forward-time, i.e. the POST-sub-sample `cur` BEFORE any
    reverse adjustment -- what the container threads between layers and what
    `feed_backward_reverse` receives as its `input` argument before reversing it
    itself), `output` (forward-time `y`), and the RAW `gates`/`cells` produced by
    `lstm_forward_oracle` on the time-reversed input when `reverse` (i.e. still in
    reversed-time order, consumed as-is by the backward per R9).
    """
    cur = x
    caches: list[dict[str, object]] = []
    for jj, layer in enumerate(layers):
        if sub_sampling[jj] > 1:
            cur = _sub_sample(sub_sampling[jj], cur)
        iw = layer["input_w"]
        fw = layer["feedback_w"]
        pp = layer["peep"]
        bs = layer["bias"]
        flags = layer["flags"]
        layer_in_fwd = cur  # forward-time orientation threaded to caches/next layer
        kernel_in = layer_in_fwd[::-1, :].copy() if reverse else layer_in_fwd
        y, gates, cells = lstm_forward_oracle(iw, fw, pp, bs, kernel_in, flags)  # type: ignore[arg-type]
        y_fwd = y[::-1, :].copy() if reverse else y
        caches.append({"input": layer_in_fwd, "output": y_fwd, "gates": gates, "cells": cells})
        cur = y_fwd
    return cur, caches


def _dense_net_forward(layers: list[dict[str, object]], sub_sampling: list[int], x: NDArray[np.float64]) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    """NeuralNetwork<NeuronLayer>::feedForward over a stack of dense layers with
    per-layer sub-sampling; only the FINAL layer runs with last_layer=True.

    Returns `(output, caches)`; `caches[jj]` retains this layer's (post-sub-sample)
    `input` and `output`, needed by the backward's per-layer `deltas`/`input`
    threading (`NeuralNetwork.hpp:251-300`, mirrored by `_dense_net_backward` below).
    """
    cur = x
    n = len(layers)
    caches: list[dict[str, object]] = []
    for jj, layer in enumerate(layers):
        if sub_sampling[jj] > 1:
            cur = _sub_sample(sub_sampling[jj], cur)
        layer_in = cur
        cur = dense_forward_oracle(
            layer["weights"],  # type: ignore[arg-type]
            layer["bias"],  # type: ignore[arg-type]
            layer_in,
            last_layer=(jj == n - 1),
        )
        caches.append({"input": layer_in, "output": cur})
    return cur, caches


def blstm_forward_oracle(
    x: NDArray[np.float64],
    norm_type: int,
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    fwd_layers: list[dict[str, object]],
    bwd_layers: list[dict[str, object]],
    lstm_sub_sampling: list[int],
    out_layers: list[dict[str, object]],
    out_sub_sampling: list[int],
    fwd_input_size: int,
) -> NDArray[np.float64]:
    """Independent scalar-loop oracle for the BLSTM wrapper forward path
    (BLSTMNeuralNetwork.cpp:419-437 core + :715-751/:777-786 normalization), plain
    (non-windowed) no-targets. Composes the LSTM/dense net oracles above.

    norm_type: 1 (external: (x-mean)/max(1e-12,std) per col < min(cols,mean.size)),
    -1 / -2 (whole-sequence self-norm: center, sqrt((sum sq + 1e-32)/rows), asinh(x/std)),
    else no normalization. Returns the output net's output (forward LSTM half LEFT).
    """
    xin = np.array(x, dtype=np.float64, copy=True)
    if norm_type == 1:
        max_col = min(xin.shape[1], len(mean))
        for jj in range(max_col):
            denom = max(1e-12, float(std[jj]))
            for r in range(xin.shape[0]):
                xin[r, jj] = (xin[r, jj] - float(mean[jj])) / denom
    elif norm_type in (-1, -2):
        # Whole-sequence self-norm with ASCENDING per-column sums (matches the engine's
        # accumulation order; numpy.sum would drift a few ULP). center; std =
        # sqrt((sum sq + 1e-32)/rows) with 1e-32 into the SUM; then asinh(x/std).
        rows, cols = xin.shape
        for c in range(cols):
            acc = 0.0
            for r in range(rows):
                acc += xin[r, c]
            m = acc / rows
            for r in range(rows):
                xin[r, c] -= m
        for c in range(cols):
            acc = 0.0
            for r in range(rows):
                acc += xin[r, c] * xin[r, c]
            sd = math.sqrt((acc + 1e-32) / rows)
            for r in range(rows):
                xin[r, c] = _asinh(xin[r, c] / sd)

    # Core forward (:419-437): leftCols gate then forward + reverse + double.
    fwd_in = xin
    bwd_in = xin
    if lstm_sub_sampling[0] > 1 and fwd_input_size < xin.shape[1]:
        fwd_in = xin[:, :fwd_input_size]
        bwd_in = xin[:, :fwd_input_size]
    out_forward, _ = _lstm_net_forward(fwd_layers, lstm_sub_sampling, fwd_in, reverse=False)
    out_backward, _ = _lstm_net_forward(bwd_layers, lstm_sub_sampling, bwd_in, reverse=True)
    hcat = np.concatenate([out_forward, out_backward], axis=1)  # forward LEFT
    out, _ = _dense_net_forward(out_layers, out_sub_sampling, hcat)
    return out


def _cost_deltas_multiclass(outputs: NDArray[np.float64], onehot_target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Softmax+CE fusion seed deltas, the multiclass branch of `CostLaw::computeDeltas`
    (CostLaw.cpp:403-418, ported to Rust at `cost.rs::CostLaw::compute_deltas`). No
    BackPropWER, no class ponderations (the SYNB/SYNW harness fixtures set neither):
    `target < 0` -> 0 (ignore mask); `target > 0.5` -> `output - 1.0` (on-class);
    else -> `output` (off-class). NO Jacobian here -- this IS the fused softmax-CE
    gradient; NeuronLayer's backward applies no further activation derivative on the
    last layer (the double-count trap, spec S5/R3)."""
    out = np.zeros_like(outputs)
    rows, cols = outputs.shape
    for r in range(rows):
        for c in range(cols):
            t = float(onehot_target[r, c])
            if t < 0.0:
                out[r, c] = 0.0
            elif t > 0.5:
                out[r, c] = float(outputs[r, c]) - 1.0
            else:
                out[r, c] = float(outputs[r, c])
    return out


def _lstm_net_backward(
    layers: list[dict[str, object]],
    sub_sampling: list[int],
    caches: list[dict[str, object]],
    deltas: NDArray[np.float64],
    reverse: bool,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """`NeuralNetwork<LSTMLayer>::feedBackward`/`feedBackwardReverse`
    (NeuralNetwork.hpp:251-300/302-351) over a stack of LSTM layers, PLAIN case only
    (every `sub_sampling[jj] == 1` -- the sub-sampled container branches, :255-263/
    :268-296 decimation arms, are NOT transcribed here; this oracle's synthetic
    cross-check target, the harness `blstm_bwd_plain` fixture, uses `sub_sampling ==
    [1]`). Reverse-order layer loop: the LAST layer reads the seed `deltas`
    (`last_layer=True` in the per-layer call, since it feeds the hcat/output net
    boundary -- wait: per NeuralNetwork.hpp the `lastLayer` flag passed to
    `LayerType::feedBackward` marks whether THIS layer is the FIRST one reached by the
    reverse walk, i.e. the network's own last layer; LSTM's `feed_backward` ignores it
    (signature parity only, layers.rs:394), so its value is immaterial here) then each
    earlier layer reads the running `deltas_out`. Returns `(deltas_out, per_layer_derivs)`
    where `per_layer_derivs[jj]` is layer `jj`'s Nx2 harvest.

    `reverse` selects `feed_backward_reverse` semantics for EVERY layer in the stack
    (mirrors the Rust `Network::feed_backward_reverse`, which dispatches the reverse
    per-layer kernel uniformly, not just the outermost one) -- each layer's `feed_
    backward_reverse` reverses its OWN input/output/deltas, runs the forward-order
    kernel, and reverses the result (LSTMLayer.cpp:725-732); since the caches were
    themselves recorded in reversed-time order by `_lstm_net_forward` (mirroring the
    Rust `feed_forward_reverse` doc, layers.rs:349-353), the deltas fed to a REVERSED
    layer here must ALSO already be in that same reversed-time order -- the caller
    passes each layer its immediately-downstream deltas as produced by this same
    function, so the orientation is self-consistent end to end.
    """
    n = len(layers)
    for r in sub_sampling:
        if r != 1:
            raise NotImplementedError("sub-sampled LSTM container backward not ported to the Python oracle")

    per_layer_derivs: list[NDArray[np.float64]] = []
    cur_deltas = deltas
    for kk in range(n):
        jj = n - 1 - kk
        cache = caches[jj]
        layer = layers[jj]
        layer_input = cache["input"]
        layer_output = cache["output"]
        gates = cache["gates"]
        cells = cache["cells"]
        if reverse:
            # feed_backward_reverse (LSTMLayer.cpp:725-732): reverse input/output/
            # deltas, run the forward-order kernel, reverse the returned deltas.
            # `gates`/`cells` are NOT re-reversed here: they are the layer's OWN
            # `_Gates`/`_CellStates` fields, already left in reversed-time order by
            # `feed_forward_reverse` and read as-is by `feed_backward` (layers.rs:
            # 349-353/725-732 -- the Rust `feed_backward_reverse` only reverses its
            # three ARGUMENTS, never touches `self.gates`/`self.cell_states`).
            input_rev = layer_input[::-1, :].copy()  # type: ignore[index]
            output_rev = layer_output[::-1, :].copy()  # type: ignore[index]
            deltas_rev = cur_deltas[::-1, :].copy()
            derivs, dpl = lstm_backward_oracle(
                layer["input_w"],  # type: ignore[arg-type]
                layer["feedback_w"],  # type: ignore[arg-type]
                layer["peep"],  # type: ignore[arg-type]
                input_rev,
                output_rev,
                gates,  # type: ignore[arg-type]
                cells,  # type: ignore[arg-type]
                deltas_rev,
                layer["flags"],  # type: ignore[arg-type]
                1,
            )
            cur_deltas = dpl[::-1, :].copy()
        else:
            derivs, dpl = lstm_backward_oracle(
                layer["input_w"],  # type: ignore[arg-type]
                layer["feedback_w"],  # type: ignore[arg-type]
                layer["peep"],  # type: ignore[arg-type]
                layer_input,  # type: ignore[arg-type]
                layer_output,  # type: ignore[arg-type]
                gates,  # type: ignore[arg-type]
                cells,  # type: ignore[arg-type]
                cur_deltas,
                layer["flags"],  # type: ignore[arg-type]
                1,
            )
            cur_deltas = dpl
        per_layer_derivs.append(derivs)

    per_layer_derivs.reverse()  # layer-0-first, matching get_weights_derivatives order
    return cur_deltas, per_layer_derivs


def _dense_net_backward(
    layers: list[dict[str, object]],
    sub_sampling: list[int],
    caches: list[dict[str, object]],
    deltas: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """`NeuralNetwork<NeuronLayer>::feedBackward` (NeuralNetwork.hpp:251-300), PLAIN case
    only (every `sub_sampling[jj] == 1`). Reverse-order layer loop: the last layer
    (index `n-1`) reads the seed `deltas` with `last_layer=True` (no output-activation
    derivative -- the fusion lives in the cost seam); every earlier layer reads the
    running `deltas_out` with `last_layer=False` (asinh-deriv fold on ITS OWN raw
    input, NeuronLayer.cpp:204). Returns `(deltas_out, per_layer_derivs)` layer-0-first.
    """
    n = len(layers)
    for r in sub_sampling:
        if r != 1:
            raise NotImplementedError("sub-sampled dense container backward not ported to the Python oracle")

    per_layer_derivs: list[NDArray[np.float64]] = []
    cur_deltas = deltas
    for kk in range(n):
        jj = n - 1 - kk
        cache = caches[jj]
        layer = layers[jj]
        derivs, dpl = dense_backward_oracle(
            layer["weights"],  # type: ignore[arg-type]
            cache["input"],  # type: ignore[arg-type]
            cur_deltas,
            jj == n - 1,
            1,
        )
        per_layer_derivs.append(derivs)
        cur_deltas = dpl

    per_layer_derivs.reverse()
    return cur_deltas, per_layer_derivs


def blstm_backward_oracle(
    x: NDArray[np.float64],
    target: NDArray[np.float64],
    fwd_layers: list[dict[str, object]],
    bwd_layers: list[dict[str, object]],
    lstm_sub_sampling: list[int],
    out_layers: list[dict[str, object]],
    out_sub_sampling: list[int],
) -> NDArray[np.float64]:
    """Independent scalar-loop oracle for the BLSTM wrapper backward fan-out
    (`BLSTMNeuralNetwork::feedBackward`, BLSTMNeuralNetwork.cpp:439-460, same source
    lines as the Rust `nn/blstm.rs::BlstmNetwork::feed_backward`), PLAIN
    (non-windowed, `InputNormalizationType == 0`, `TargetEnforcementStep == 0`,
    `BackPropOutputNetworkOnly == false`) no-sub-sampling case -- the harness
    `blstm_bwd_plain` fixture's shape (LSTM `[2,2]` sub `[1]`, output `[4,2]` sub
    `[1]`). `BLSTM_Backward.m` is STRUCTURE-REFERENCE ONLY and must not be ported
    (P2b T11 verdict; it pairs with the divergent MATLAB forward).

    Pipeline: forward (fwd LSTM net, bwd LSTM net reverse, hcat forward-left, output
    dense net -- mirrors `blstm_forward_oracle` but retaining every layer's caches);
    seed deltas via `_cost_deltas_multiclass` (multiclass softmax+CE fusion, since the
    harness fixture's targets are `n_classes > 1`) on the FINAL output; the output
    net's backward (`_dense_net_backward`) consumes the hcat input and returns
    `deltas_out` split left|right (forward-LSTM-sized | backward-LSTM-sized, forward
    LEFT per the hcat convention); the forward LSTM net backward (`_lstm_net_backward`,
    `reverse=False`) consumes the left half, the backward LSTM net backward
    (`reverse=True`) consumes the right half.

    Returns the FULL Nx2 flat derivs in the `BlstmNetwork::get_weights_derivatives`
    block order (fwd LSTM layers -> bwd LSTM layers -> output dense layers -> mean/std
    tail, each `[0.0, 1.0]`) (BLSTMNeuralNetwork.cpp:255-276). This oracle does not
    scale the fwd/bwd LSTM col0 by the output sub-sampling ratio (`:266-269`) since the
    target fixture's output ratio is 1; a caller driving an output-ratio > 1 net must
    apply that scaling itself.
    """
    out_forward, fwd_caches = _lstm_net_forward(fwd_layers, lstm_sub_sampling, x, reverse=False)
    out_backward, bwd_caches = _lstm_net_forward(bwd_layers, lstm_sub_sampling, x, reverse=True)
    hcat = np.concatenate([out_forward, out_backward], axis=1)  # forward LEFT
    final_out, out_caches = _dense_net_forward(out_layers, out_sub_sampling, hcat)

    n_classes = target.shape[1]
    if n_classes == 1:
        raise NotImplementedError("scalar-VAD seed path not ported to the Python BLSTM backward oracle")
    seed_deltas = _cost_deltas_multiclass(final_out, target)

    split, output_derivs = _dense_net_backward(out_layers, out_sub_sampling, out_caches, seed_deltas)
    half = split.shape[1] // 2
    left = split[:, :half]
    right = split[:, half:]

    _, fwd_derivs = _lstm_net_backward(fwd_layers, lstm_sub_sampling, fwd_caches, left, reverse=False)
    _, bwd_derivs = _lstm_net_backward(bwd_layers, lstm_sub_sampling, bwd_caches, right, reverse=True)

    fwd_input_w: NDArray[np.float64] = fwd_layers[0]["input_w"]  # type: ignore[assignment]
    input_size = fwd_input_w.shape[0]
    tail = [[0.0, 1.0] for _ in range(2 * input_size)]

    all_derivs = fwd_derivs + bwd_derivs + output_derivs
    rows = [row for block in all_derivs for row in block.tolist()] + tail
    return np.array(rows, dtype=np.float64).reshape(len(rows), 2)
