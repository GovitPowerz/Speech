"""Anchor tests for the regression-deltas numpy oracle (Task 6).

The oracle is an independent numpy coding of the same clamped-formula kernel the
Rust `regression_deltas` implements; the hand anchor here (and the cross-language
JSON in `deltas_cases.json`) is the arbiter that both codings agree.
"""

import numpy as np
from speech.features_oracle import regression_deltas_oracle


def test_regression_deltas_anchor() -> None:
    # base [[0],[1],[4]] (T=3, W=1), n=2. denom = 2*(1^2+2^2) = 10.
    # D[0] = (1*(x1-x0) + 2*(x2-x0)) / 10 = (1 + 8)/10 = 0.9
    # D[1] = (1*(x2-x0) + 2*(x2-x0))/10 = (4 + 8)/10 = 1.2   (t-1 clamped to 0, t+j clamped to T-1)
    # D[2] = (1*(x2-x1) + 2*(x2-x0))/10 = (3 + 8)/10 = 1.1
    base = np.array([[0.0], [1.0], [4.0]])
    got = regression_deltas_oracle(base, 2)
    expected = np.array([[0.9], [1.2], [1.1]])
    assert np.array_equal(got, expected)


def test_regression_deltas_edge_t_le_n() -> None:
    # T <= n: clamps saturate; denom still 2*sum_{j=1..n} j^2 including the
    # clamped j (the legacy adim accumulates j*j even when the shifted copy is
    # skipped). base [[0],[2]], n=3 -> denom = 2*(1+4+9) = 28.
    #   D[0] = (1*(x1-x0) + 2*(x1-x0) + 3*(x1-x0))/28 = 6*2/28 = 12/28
    #   D[1] = (1*(x1-x0) + 2*(x1-x0) + 3*(x1-x0))/28 = 6*2/28 = 12/28
    base = np.array([[0.0], [2.0]])
    got = regression_deltas_oracle(base, 3)
    expected = np.array([[12.0 / 28.0], [12.0 / 28.0]])
    assert np.array_equal(got, expected)
