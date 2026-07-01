"""Anchor tests for the regression-deltas numpy oracle (Task 6).

The oracle is an independent numpy coding of the same clamped-formula kernel the
Rust `regression_deltas` implements; the hand anchor here (and the cross-language
JSON in `deltas_cases.json`) is the arbiter that both codings agree.
"""

import numpy as np
from speech.features_oracle import regression_deltas_oracle, sdc_oracle


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


def test_sdc_oracle_block_placement_4x2() -> None:
    # T=4, nb_dct=2. Per-block time offset src = t + 10 - 3*kk (P=3, k=7); a cell is
    # D[src] when 0 <= src < 4 else 0. Hand-enumerating src ranges over t=0..3:
    #   kk=0: 10..13  -> all oob -> zero
    #   kk=1:  7..10  -> all oob -> zero
    #   kk=2:  4..7   -> all oob -> zero
    #   kk=3:  1..4   -> D[1],D[2],D[3],0
    #   kk=4: -2..1   -> 0,0,D[0],D[1]
    #   kk=5: -5..-2  -> zero
    #   kk=6: -8..-5  -> zero
    # Only blocks 3 and 4 are nonzero.
    mfcc = np.array([[0.0, 10.0], [1.0, 8.0], [4.0, 3.0], [9.0, 1.0]])
    d = regression_deltas_oracle(mfcc, 3)  # 4x2 n=3 delta
    got = sdc_oracle(mfcc, 2)  # 4 x (7*2)=14
    assert got.shape == (4, 14)

    expected = np.zeros((4, 14))
    # block 3 (cols 6..8): rows t=0,1,2 <- D[1],D[2],D[3]; t=3 zero.
    expected[0, 6:8] = d[1]
    expected[1, 6:8] = d[2]
    expected[2, 6:8] = d[3]
    # block 4 (cols 8..10): t=2 <- D[0], t=3 <- D[1]; t=0,1 zero.
    expected[2, 8:10] = d[0]
    expected[3, 8:10] = d[1]
    assert np.array_equal(got, expected)
