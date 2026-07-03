"""Anchor tests for the Phase 2 LSTM forward oracle (nn_reference.lstm_forward_oracle).

The oracle is an independent scalar-loop coding of LSTMLayer::feedForward
(LSTMLayer.cpp:312-413). The hand anchor here computes one small case longhand in the
spec S4.3 op order and requires the oracle to reproduce it bit-for-bit; the
cross-language `lstm_cases.json` fixture (consumed by the Rust golden test) is the
arbiter that the oracle and the Rust `LstmLayer` agree exactly. Two extra tests pin
the two load-bearing quirks: the t=0 no-forget-term and the input width tolerance.
"""

import math

import numpy as np
from speech.nn_reference import lstm_forward_oracle


def _gate(z: float) -> float:
    # GatesFunction: sigmoid on 0.1*z.
    return 1.0 / (1.0 + math.exp(-0.1 * z))


def test_lstm_hand_case_o1_t2() -> None:
    # I=1, O=1, T=2, all peephole flags on. Same distinct small rational weights as the
    # Rust hand anchor (phase2_layers_golden.rs::lstm_hand_case_o1_t2). Layout col-major
    # (I=1,O=1): input_w [Wi,Wf,Wo,Wg]; feedback [Ui,Uf,Uo,Ug]; peep P[0..12]; bias
    # [bi,bf,bo,bg].
    wi, wf, wo, wg = 1.0 / 8.0, -1.0 / 4.0, 3.0 / 8.0, -1.0 / 2.0
    ui, uf, uo, ug = 1.0 / 16.0, -3.0 / 16.0, 5.0 / 16.0, -7.0 / 16.0
    p = [
        1.0 / 32.0,
        -2.0 / 32.0,
        3.0 / 32.0,
        -4.0 / 32.0,
        5.0 / 32.0,
        -6.0 / 32.0,
        7.0 / 32.0,
        -8.0 / 32.0,
        9.0 / 32.0,
        -10.0 / 32.0,
        11.0 / 32.0,
        -12.0 / 32.0,
    ]
    bi, bf, bo, bg = 1.0 / 3.0, -1.0 / 6.0, 1.0 / 5.0, -1.0 / 7.0

    input_w = np.array([[wi, wf, wo, wg]], dtype=np.float64)  # 1 x 4
    feedback_w = np.array([[ui, uf, uo, ug]], dtype=np.float64)  # 1 x 4
    peep = np.array(p, dtype=np.float64).reshape(12, 1)  # 12 x 1
    bias = np.array([[bi, bf, bo, bg]], dtype=np.float64)  # 1 x 4

    x0, x1 = 0.3, -0.2
    x = np.array([[x0], [x1]], dtype=np.float64)

    def asinh(z: float) -> float:
        return math.asinh(z)

    # t=0 (:325-348), all flags on. No forget term, no row-11 term.
    i0 = _gate(wi * x0 + bi)
    f0 = _gate(wf * x0 + bf)
    g0 = asinh(wg * x0 + bg)
    c0 = i0 * g0
    o0_pre = wo * x0 + bo
    o0_pre += c0 * p[2]
    o0_pre += i0 * p[9]
    o0_pre += f0 * p[10]
    o0 = _gate(o0_pre)
    y0 = o0 * asinh(c0)

    # t=1 (:350-412), post-activation reads of t=0.
    i1 = wi * x1 + bi + y0 * ui
    f1 = wf * x1 + bf + y0 * uf
    o1 = wo * x1 + bo + y0 * uo
    g1_pre = wg * x1 + bg + y0 * ug
    i1 += c0 * p[0]
    f1 += c0 * p[1]
    i1 += i0 * p[3]
    f1 += f0 * p[7]
    i1 += f0 * p[4] + o0 * p[5]
    f1 += i0 * p[6] + o0 * p[8]
    i1 = _gate(i1)
    f1 = _gate(f1)
    g1 = asinh(g1_pre)
    c1 = i1 * g1 + c0 * f1
    o1 += c1 * p[2]
    o1 += i1 * p[9]
    o1 += f1 * p[10]
    o1 += o0 * p[11]
    o1 = _gate(o1)
    y1 = o1 * asinh(c1)

    y, gates, cells = lstm_forward_oracle(input_w, feedback_w, peep, bias, x, (True, True, True))

    # Same libm, same op order -> bit-identical.
    assert y[0, 0].tobytes() == np.float64(y0).tobytes(), "t=0 output"
    assert y[1, 0].tobytes() == np.float64(y1).tobytes(), "t=1 output"
    assert cells[0, 0].tobytes() == np.float64(c0).tobytes(), "c_0"
    assert cells[1, 0].tobytes() == np.float64(c1).tobytes(), "c_1"
    for col, val in enumerate((i1, f1, o1, g1)):
        assert gates[1, col].tobytes() == np.float64(val).tobytes(), f"gates col {col}"


def test_lstm_t0_no_forget_term() -> None:
    # c_0 = i_0 .* g_0 with NO forget contribution (:336). Flags = cells-only so f_0
    # does not leak into o_0 (gates-peep off). Perturbing the forget input weight +
    # bias must leave y[0] bit-identical, but shift y[1] (c_0 flows through f_1 at :387).
    i, o = 1, 1
    input_w = np.array([[0.2, -0.3, 0.4, -0.1]], dtype=np.float64)
    feedback_w = np.array([[0.05, -0.06, 0.07, -0.08]], dtype=np.float64)
    peep = (np.arange(12, dtype=np.float64) / 40.0 - 0.15).reshape(12, 1)
    bias = np.array([[0.1, -0.2, 0.15, -0.05]], dtype=np.float64)
    x = np.array([[0.25], [-0.35], [0.15]], dtype=np.float64)
    flags = (True, False, False)

    y_base, _, _ = lstm_forward_oracle(input_w, feedback_w, peep, bias, x, flags)

    input_w2 = input_w.copy()
    bias2 = bias.copy()
    input_w2[0, 1] += 3.0  # Wf
    bias2[0, 1] += 5.0  # bf
    y_pert, _, _ = lstm_forward_oracle(input_w2, feedback_w, peep, bias2, x, flags)

    assert y_base[0, 0].tobytes() == y_pert[0, 0].tobytes(), "t=0 independent of forget weights"
    assert y_base[1, 0].tobytes() != y_pert[1, 0].tobytes(), "t=1 depends on forget gate"
    assert i == 1 and o == 1


def test_lstm_input_width_tolerance() -> None:
    # cols > I uses the left I input columns (:313-314); cols < I uses the top cols
    # weight rows (:315-316). Verify both against an equivalent trimmed/exact run.
    big_i, o = 3, 2
    rng = np.random.default_rng(0)
    input_w = rng.standard_normal((big_i, 4 * o))
    feedback_w = rng.standard_normal((o, 4 * o))
    peep = rng.standard_normal((12, o))
    bias = rng.standard_normal((1, 4 * o))
    flags = (True, True, True)

    # Wide input (cols = I+2): only the left I cols are used.
    x_wide = rng.standard_normal((5, big_i + 2))
    y_wide, _, _ = lstm_forward_oracle(input_w, feedback_w, peep, bias, x_wide, flags)
    y_trim, _, _ = lstm_forward_oracle(input_w, feedback_w, peep, bias, x_wide[:, :big_i], flags)
    assert np.array_equal(y_wide, y_trim), "wide input uses left I columns only"

    # Narrow input (cols = I-1): only the top cols weight rows are used.
    x_narrow = rng.standard_normal((5, big_i - 1))
    y_narrow, _, _ = lstm_forward_oracle(input_w, feedback_w, peep, bias, x_narrow, flags)
    y_toprows, _, _ = lstm_forward_oracle(input_w[: big_i - 1, :], feedback_w, peep, bias, x_narrow, flags)
    assert np.array_equal(y_narrow, y_toprows), "narrow input uses top cols weight rows only"
