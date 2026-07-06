"""Phase 3 Task 9: the Python backward oracle (nn_reference.py), independent arbiter.

`lstm_backward_oracle`/`dense_backward_oracle`/`blstm_backward_oracle` are scalar-loop
transcriptions of the ENGINE backward (`LSTMLayer.cpp:518-723`, `NeuronLayer.cpp:151-207`,
`BLSTMNeuralNetwork.cpp:439-460`), the SAME source lines the Rust `nn/layers.rs`/
`nn/blstm.rs` port. `BLSTM_Backward.m` is STRUCTURE-REFERENCE ONLY (P2b T11 verdict: it
pairs with the divergent MATLAB forward -- sigmoid(x) gates, no softmax, internal /ncols
normalization -- and must NOT be ported).

Triangulation (this phase's cross-check contract) is TRANSITIVE, not a live cross-call:
the Task 1 JSON cases (`backward_cases.json`) carry expected Nx2 derivs SOURCED FROM THE
HARNESS REIMPL GOLDENS; the SAME goldens are independently reproduced by (a) the Rust
suite (`cargo test`, `phase3_*_golden.rs`, consuming the harness `.bin` dumps directly)
and (b) this Python oracle replaying `backward_cases.json`. Matching bit-for-bit in both
places closes the seam (harness-reimpl == Rust-layer, harness-reimpl == Python-oracle),
which is equivalent in force to a three-way agreement but there is NO live PyO3
cross-call here: `speech-py` (`src/rust/speech-py/src/lib.rs`) is an 18-line stub
exposing only `version()` -- the hot-loop binding is a Phase 4 deliverable. The
`test_json_replay_closes_t1_seam` test below replays `backward_cases.json` a second
time through the oracle; it does not build or call the Rust layers directly.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from speech.nn_reference import (
    _cost_deltas_multiclass,
    _gates_fn,
    blstm_backward_oracle,
    dense_backward_oracle,
    lstm_backward_oracle,
    lstm_forward_oracle,
)
from speech.weight_bridge import read_bin

from tests._libm_gate import assert_f64_matrix_close as _assert_f64_matrix_close

PHASE3_DIR = Path(__file__).resolve().parent / "reference_data" / "phase3"


def _bits_to_f(s: str) -> float:
    return float(struct.unpack("<d", struct.pack("<Q", int(s, 16)))[0])


def _mat_from_bits(rows: list[list[str]]) -> np.ndarray:
    return np.array([[_bits_to_f(v) for v in row] for row in rows], dtype=np.float64)


def _load_backward_cases() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((PHASE3_DIR / "backward_cases.json").read_text())
    return result


def _unpack_lstm_flat(flat: np.ndarray, i: int, o: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Column-major unpack per `LSTMLayer::setWeights` (InputWeights, FeedbackWeights,
    PeepWeight, Biaises)."""
    pos = 0
    input_w = flat[pos : pos + i * 4 * o].reshape(4 * o, i).T.copy()
    pos += i * 4 * o
    feedback_w = flat[pos : pos + o * 4 * o].reshape(4 * o, o).T.copy()
    pos += o * 4 * o
    peep = flat[pos : pos + 12 * o].reshape(o, 12).T.copy()
    pos += 12 * o
    bias = flat[pos : pos + 4 * o].reshape(1, 4 * o).copy()
    return input_w, feedback_w, peep, bias


def _unpack_dense_flat(flat: np.ndarray, i: int, o: int) -> tuple[np.ndarray, np.ndarray]:
    weights = flat[: i * o].reshape(o, i).T.copy()
    bias = flat[i * o : i * o + o].reshape(1, o).copy()
    return weights, bias


def _synth_flat(n: int) -> np.ndarray:
    # main.cpp synthFlat: w[k] = ((k*11+3) % 97)/97 - 0.5.
    return np.array([((k * 11 + 3) % 97) / 97.0 - 0.5 for k in range(n)], dtype=np.float64)


def _make_input(t_len: int, cols: int) -> np.ndarray:
    # main.cpp makeInput: x[t,j] = ((t*37 + j*53 + 7) % 101)/101 - 0.5.
    return np.array(
        [[((t * 37 + j * 53 + 7) % 101) / 101.0 - 0.5 for j in range(cols)] for t in range(t_len)],
        dtype=np.float64,
    )


def _make_deltas(t_len: int, o: int) -> np.ndarray:
    # main.cpp makeDeltas: d[t,j] = ((t*29 + j*41 + 5) % 83)/83 - 0.5.
    return np.array(
        [[((t * 29 + j * 41 + 5) % 83) / 83.0 - 0.5 for j in range(o)] for t in range(t_len)],
        dtype=np.float64,
    )


def _lstm_nb(i: int, o: int) -> int:
    return 4 * i * o + 4 * o * o + 12 * o + 4 * o


def _dense_nb(i: int, o: int) -> int:
    return o * (i + 1)


# === lstm_backward_matches_harness ==========================================


def test_lstm_backward_matches_harness() -> None:
    d = _load_backward_cases()
    for c in d["lstm_cases"]:
        i, o = c["I"], c["O"]
        flat = np.array([_bits_to_f(v) for v in c["weights_bits"]], dtype=np.float64)
        x = _mat_from_bits(c["input_bits"])
        deltas = _mat_from_bits(c["deltas_bits"])
        expected = _mat_from_bits(c["expected_derivs_bits"])
        flags = tuple(c["peepholes"])

        input_w, feedback_w, peep, bias = _unpack_lstm_flat(flat, i, o)
        y, gates, cells = lstm_forward_oracle(input_w, feedback_w, peep, bias, x, flags)  # type: ignore[arg-type]
        derivs, deltas_prev = lstm_backward_oracle(input_w, feedback_w, peep, x, y, gates, cells, deltas, flags, 1)

        assert derivs.shape == expected.shape, f"{c['name']}: shape {derivs.shape} != {expected.shape}"
        _assert_f64_matrix_close(derivs, expected, f"lstm backward {c['name']}")
        assert deltas_prev.shape == (c["T"], i), f"{c['name']}: deltas_prev shape"


def test_lstm_backward_variants_matches_harness() -> None:
    """Task 4 per-variant goldens (tools/oracle_harness/main.cpp:7021-7112): synthetic
    I=2,O=2,T=5, dumping BOTH the Nx2 derivs AND `deltasPreviousLayer` (T x I) per
    variant, vs the REAL compiled `LSTMLayer::feedBackward`/`feedBackwardReverse`.
    `peep_all`/`peep_none` isolate the peephole cross-terms (a row swap or a flag drop
    changes col0 without changing shape); `reverse` exercises `feed_backward_reverse`
    (LSTMLayer.cpp:725-732, risk R9 -- caches consumed as-is, never un-reversed);
    `subsample` (invSubSamplingRatio=2) pins the *2 block scaling (:710-715)."""
    i, o, t_len = 2, 2, 5
    flat = _synth_flat(_lstm_nb(i, o))

    variants = [
        ("peep_all", (True, True, True), False, 1),
        ("peep_none", (False, False, False), False, 1),
        ("reverse", (True, True, True), True, 1),
        ("subsample", (True, True, True), False, 2),
    ]
    for tag, flags, reverse, ratio in variants:
        input_w, feedback_w, peep, bias = _unpack_lstm_flat(flat, i, o)
        x = _make_input(t_len, i)
        deltas = _make_deltas(t_len, o)

        if reverse:
            x_rev = x[::-1, :].copy()
            y_rev, gates, cells = lstm_forward_oracle(input_w, feedback_w, peep, bias, x_rev, flags)  # type: ignore[arg-type]
            deltas_rev = deltas[::-1, :].copy()
            derivs, dpl_rev = lstm_backward_oracle(input_w, feedback_w, peep, x_rev, y_rev, gates, cells, deltas_rev, flags, ratio)
            deltas_prev = dpl_rev[::-1, :].copy()
        else:
            y, gates, cells = lstm_forward_oracle(input_w, feedback_w, peep, bias, x, flags)  # type: ignore[arg-type]
            derivs, deltas_prev = lstm_backward_oracle(input_w, feedback_w, peep, x, y, gates, cells, deltas, flags, ratio)

        _rows, _cols, flatbin = read_bin(PHASE3_DIR / f"lstm_bwd_derivs_{tag}.bin")
        expected_derivs = flatbin.reshape(_cols, _rows).T
        _assert_f64_matrix_close(derivs, np.ascontiguousarray(expected_derivs, dtype=np.float64), f"lstm backward variant {tag} derivs")

        _rows2, _cols2, flatbin2 = read_bin(PHASE3_DIR / f"lstm_bwd_deltasprev_{tag}.bin")
        expected_dpl = flatbin2.reshape(_cols2, _rows2).T
        _assert_f64_matrix_close(deltas_prev, np.ascontiguousarray(expected_dpl, dtype=np.float64), f"lstm backward variant {tag} deltas_prev")


def test_lstm_backward_signal_width_tolerance() -> None:
    """The `signal` variant (I=2,O=2 layer fed a 1-col input): width-tolerance
    zero-pad (LSTMLayer.cpp:534-543) means the dead (zero-padded) input-weight-deriv
    ROW accumulates EXACTLY 0.0 (spec S11.5), bit, not just approximately."""
    i, o, t_len = 2, 2, 5
    flat = _synth_flat(_lstm_nb(i, o))
    input_w, feedback_w, peep, bias = _unpack_lstm_flat(flat, i, o)
    flags = (True, True, True)

    x_narrow = _make_input(t_len, 1)
    y, gates, cells = lstm_forward_oracle(input_w, feedback_w, peep, bias, x_narrow, flags)  # type: ignore[arg-type]
    deltas = _make_deltas(t_len, o)
    derivs, _ = lstm_backward_oracle(input_w, feedback_w, peep, x_narrow, y, gates, cells, deltas, flags, 1)

    _rows, _cols, flatbin = read_bin(PHASE3_DIR / "lstm_bwd_signal_derivs.bin")
    expected = flatbin.reshape(_cols, _rows).T
    _assert_f64_matrix_close(derivs, np.ascontiguousarray(expected, dtype=np.float64), "lstm backward signal width tolerance")

    # Dead row: input_w row 1 (the zero-padded column) contributes to NO gate block --
    # its 4*O deriv entries (rows i=1, jj=0..4*O-1 in the column-major Nx2 harvest) are
    # exactly 0.0.
    dead_row_entries = [derivs[jj * i + 1, 0] for jj in range(4 * o)]
    assert all(v == 0.0 for v in dead_row_entries), "zero-padded input row must accumulate exactly 0.0"


# === dense_backward_matches_harness ==========================================


def test_dense_backward_matches_harness() -> None:
    d = _load_backward_cases()
    # golden basename (bwd_dense_*_derivs.bin) -> the matching deltas_out golden
    # (Task 3 also dumps deltas_out per variant -- the double-count trap fixture,
    # spec S11.3: last-layer deltas_out is EXACTLY deltas*W^T with no Jacobian).
    deltasout_golden = {
        "bwd_dense_hidden_derivs.bin": "bwd_dense_hidden_deltasout.bin",
        "bwd_dense_last_derivs.bin": "bwd_dense_last_deltasout.bin",
        "bwd_dense_logistic_derivs.bin": "bwd_dense_logistic_deltasout.bin",
    }

    for c in d["dense_cases"]:
        i, o = c["I"], c["O"]
        flat = np.array([_bits_to_f(v) for v in c["weights_bits"]], dtype=np.float64)
        x = _mat_from_bits(c["input_bits"])
        deltas = _mat_from_bits(c["deltas_bits"])
        expected = _mat_from_bits(c["expected_derivs_bits"])
        last = c["last_layer"]

        weights, _bias = _unpack_dense_flat(flat, i, o)
        derivs, deltas_out = dense_backward_oracle(weights, x, deltas, last, 1)

        assert derivs.shape == expected.shape, f"{c['name']}: shape {derivs.shape} != {expected.shape}"
        if last:
            # Last-layer deltas_out = deltas * weights^T, NO activation derivative --
            # pure arithmetic, strict comparator (no libm in this path at all).
            assert derivs.tobytes() == np.ascontiguousarray(expected, dtype=np.float64).tobytes(), f"{c['name']}: strict bit-exact"
        else:
            # Hidden path folds an asinh-deriv (`_asinh_deriv` -> sinh/sqrt) -- canary-gated.
            _assert_f64_matrix_close(derivs, expected, f"dense backward {c['name']}")
        assert deltas_out.shape == (c["T"], i), f"{c['name']}: deltas_out shape"

        # deltas_out vs the harness golden (Nx2 derivs alone can't catch a
        # deltas_out-only regression -- e.g. a spurious activation-deriv fold on the
        # last layer -- since it never touches the weight/bias gradient accumulators).
        _rows, _cols, flatbin = read_bin(PHASE3_DIR / deltasout_golden[c["golden"]])
        expected_deltasout = flatbin.reshape(_cols, _rows).T
        if last:
            assert deltas_out.tobytes() == np.ascontiguousarray(expected_deltasout, dtype=np.float64).tobytes(), f"{c['name']}: deltas_out strict bit-exact"
        else:
            _assert_f64_matrix_close(deltas_out, np.ascontiguousarray(expected_deltasout, dtype=np.float64), f"dense deltas_out {c['name']}")


def test_dense_backward_no_output_activation_derivative() -> None:
    """Double-count trap (spec S5/R3): the LAST layer's deltas_out is the pure
    `deltas @ weights^T` product with NO Jacobian folded in -- the fusion lives in
    CostLaw upstream. Distinguish it from the hidden path (which DOES fold an
    asinh-deriv on ITS OWN raw input) on the SAME weights/deltas: last-layer deltas_out
    must equal the plain product exactly; hidden deltas_out must NOT (generically)."""
    rng = np.random.default_rng(7)
    i, o, t = 3, 2, 5
    weights = rng.standard_normal((i, o))
    x = rng.standard_normal((t, i))
    deltas = rng.standard_normal((t, o))

    _derivs_last, deltas_out_last = dense_backward_oracle(weights, x, deltas, last_layer=True, inv_sub_sampling_ratio=1)
    plain_product = deltas @ weights.T
    assert np.allclose(deltas_out_last, plain_product, rtol=0, atol=1e-12), "last-layer deltas_out must be the unadorned product"

    _derivs_hidden, deltas_out_hidden = dense_backward_oracle(weights, x, deltas, last_layer=False, inv_sub_sampling_ratio=1)
    assert not np.allclose(deltas_out_hidden, plain_product), "hidden deltas_out must differ (asinh-deriv fold)"


# === blstm_backward_matches_harness ==========================================


def test_blstm_backward_matches_harness() -> None:
    """`blstm_backward_oracle` end-to-end on the harness `blstm_bwd_plain` synthetic
    net (LSTM [2,2] sub [1] + output [4,2] sub [1], T=8, plain non-windowed,
    TargetEnforcementStep=0, multiclass CE targets, InputNormalizationType=0) vs the
    REAL compiled `BLSTMNeuralNetwork::getWeightsDerivatives` golden
    (`blstm_bwd_plain_derivs.bin`, tools/oracle_harness/main.cpp:7573-7763)."""
    i, o = 2, 2
    out_i, out_o = 4, 2
    t_len = 8

    flat = _synth_flat(_lstm_nb(i, o) + _lstm_nb(i, o) + _dense_nb(out_i, out_o) + 2 * i)
    fwd_iw, fwd_fw, fwd_pp, fwd_bs = _unpack_lstm_flat(flat[: _lstm_nb(i, o)], i, o)
    bwd_iw, bwd_fw, bwd_pp, bwd_bs = _unpack_lstm_flat(flat[_lstm_nb(i, o) : 2 * _lstm_nb(i, o)], i, o)
    out_w, out_b = _unpack_dense_flat(flat[2 * _lstm_nb(i, o) :], out_i, out_o)

    fwd_layer: dict[str, object] = {"input_w": fwd_iw, "feedback_w": fwd_fw, "peep": fwd_pp, "bias": fwd_bs, "flags": (True, True, True)}
    bwd_layer: dict[str, object] = {"input_w": bwd_iw, "feedback_w": bwd_fw, "peep": bwd_pp, "bias": bwd_bs, "flags": (True, True, True)}
    out_layer: dict[str, object] = {"weights": out_w, "bias": out_b}

    x = _make_input(t_len, i)
    targets = np.zeros((t_len, 2), dtype=np.float64)
    for r in range(t_len):
        targets[r, r % 2] = 1.0

    derivs = blstm_backward_oracle(x, targets, [fwd_layer], [bwd_layer], [1], [out_layer], [1])

    rows, cols, flatbin = read_bin(PHASE3_DIR / "blstm_bwd_plain_derivs.bin")
    expected = flatbin.reshape(cols, rows).T  # column-major -> (rows, cols)

    assert derivs.shape == (rows, cols) == (142, 2)
    _assert_f64_matrix_close(derivs, np.ascontiguousarray(expected, dtype=np.float64), "blstm backward plain")

    # col1 (frame count) is a pure-arithmetic replication of T=8 (trained blocks) / 1
    # (mean/std tail) -- strict-bits regardless of libm (spec S11.4/S11.9).
    assert np.all(derivs[: 2 * _lstm_nb(i, o), 1] == 8.0), "LSTM blocks: count == T == 8"
    assert np.all(derivs[2 * _lstm_nb(i, o) : 2 * _lstm_nb(i, o) + _dense_nb(out_i, out_o), 1] == 8.0), "output block: count == T == 8"
    assert np.all(derivs[2 * _lstm_nb(i, o) + _dense_nb(out_i, out_o) :, 1] == 1.0), "mean/std tail: count == 1"
    assert np.all(derivs[2 * _lstm_nb(i, o) + _dense_nb(out_i, out_o) :, 0] == 0.0), "mean/std tail: deriv == 0"


# === test_json_replay_closes_t1_seam (transitive triangulation) =============


def test_json_replay_closes_t1_seam() -> None:
    """Transitive-triangulation cross-check (this phase's contract, NOT a live
    Rust cross-call): the Python oracle's Nx2 derivs on the SAME synthetic Task 1
    case must equal the committed `expected_derivs_bits`, which are themselves the
    harness-reimpl values the Rust `LstmLayer`/`NeuronLayer` unit/golden tests
    assert against independently (src/rust/tests -- `bwd_lstm_derivs.bin`/
    `bwd_dense_*_derivs.bin` consumed there via `cargo test`). Both legs replay the
    SAME `backward_cases.json` fixture against the SAME harness goldens, so a match
    here plus a green `cargo test` closes the seam transitively (harness-reimpl ==
    Rust-layer, harness-reimpl == Python-oracle) -- equivalent in force to a
    three-way agreement, but there is no in-process call from Python into the Rust
    layers. A mismatch here, per the binding arbiter rule, is a phase-critical
    finding, NOT a Python bug to silently patch around."""
    d = _load_backward_cases()

    for c in d["lstm_cases"]:
        i, o = c["I"], c["O"]
        flat = np.array([_bits_to_f(v) for v in c["weights_bits"]], dtype=np.float64)
        x = _mat_from_bits(c["input_bits"])
        deltas = _mat_from_bits(c["deltas_bits"])
        expected = _mat_from_bits(c["expected_derivs_bits"])
        flags = tuple(c["peepholes"])

        input_w, feedback_w, peep, bias = _unpack_lstm_flat(flat, i, o)
        y, gates, cells = lstm_forward_oracle(input_w, feedback_w, peep, bias, x, flags)  # type: ignore[arg-type]
        derivs, _ = lstm_backward_oracle(input_w, feedback_w, peep, x, y, gates, cells, deltas, flags, 1)
        _assert_f64_matrix_close(derivs, expected, f"triangulation lstm {c['name']}")

    for c in d["dense_cases"]:
        i, o = c["I"], c["O"]
        flat = np.array([_bits_to_f(v) for v in c["weights_bits"]], dtype=np.float64)
        x = _mat_from_bits(c["input_bits"])
        deltas = _mat_from_bits(c["deltas_bits"])
        expected = _mat_from_bits(c["expected_derivs_bits"])
        weights, _bias = _unpack_dense_flat(flat, i, o)
        derivs, _ = dense_backward_oracle(weights, x, deltas, c["last_layer"], 1)
        _assert_f64_matrix_close(derivs, expected, f"triangulation dense {c['name']}")


# === not_blstm_backward_m (doc/structural) ===================================


def test_not_blstm_backward_m_gate_activation() -> None:
    """Structural pin: the gate activation is ENGINE `sigmoid(0.1*z)` (GatesFunction,
    ActivationFunctions.h:230-237), NOT MATLAB `BLSTM_Backward.m`'s plain
    `sigmoid(z)`. Probing `_gates_fn` at a known z distinguishes the two: at z=10,
    `sigmoid(0.1*10) = sigmoid(1) != sigmoid(10)` by a wide margin (0.731 vs 0.99995) --
    if a future edit accidentally dropped the 0.1 pre-scale (reverting toward the
    MATLAB semantics), this assertion catches it immediately."""
    z = 10.0
    got = _gates_fn(z)
    engine_expected = 1.0 / (1.0 + math.exp(-0.1 * z))
    matlab_wrong = 1.0 / (1.0 + math.exp(-z))

    assert got == engine_expected, "GatesFunction must be sigmoid(0.1*z), the ENGINE semantics"
    assert abs(got - matlab_wrong) > 0.01, "must clearly diverge from the un-prescaled MATLAB sigmoid(z)"


def test_cost_deltas_multiclass_fusion_ignore_mask() -> None:
    """`_cost_deltas_multiclass` sanity: target<0 -> 0 (ignore), target>0.5 -> output-1,
    else -> output. No Jacobian (this IS the fused softmax+CE gradient, CostLaw.cpp:
    403-418 / cost.rs::compute_deltas, no-pond/no-WER branch)."""
    outputs = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]], dtype=np.float64)
    targets = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    got = _cost_deltas_multiclass(outputs, targets)
    want = np.array([[0.7 - 1.0, 0.2, 0.0], [0.3, 0.3 - 1.0, 0.4]], dtype=np.float64)
    assert np.array_equal(got, want)
