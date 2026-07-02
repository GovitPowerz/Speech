"""Anchor tests for the Phase 1 feature oracles (Tasks 6-9).

Each oracle is an independent coding of the kernel its Rust counterpart implements
sequentially; the hand anchors here (and the cross-language `*_cases.json` fixtures)
are the arbiters that both codings agree. The Task 9 fmath_log test additionally
pins the numpy oracle against the harness dump for EVERY sweep entry (the
double-pinning contract: harness dump == Rust port == numpy oracle).
"""

import math
from pathlib import Path

import numpy as np
from speech.features_oracle import (
    fmath_log_oracle,
    ltsv_oracle,
    regression_deltas_oracle,
    sdc_oracle,
    stats_merge_oracle,
    tdc_oracle,
)
from speech.weight_bridge import read_bin

FIXTURE_DIR = Path(__file__).resolve().parent / "reference_data" / "phase1"


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


def test_ltsv_oracle_anchor_matches_rust_hand_case() -> None:
    # Mirrors the Rust hand test ltsv_is_biased_variance_no_sqrt:
    # P = [[1,2],[3,4],[5,6]] (T=3, bins=2), col=1, R=1 -> window rows 0..=2, L=3
    # bin0: mean=3; r={1/3,1,5/3}; sum r(r-1) = -2/9+0+10/9 = 8/9; dzeta0=-(8/9)/3
    # bin1: mean=4; r={1/2,1,3/2}; sum = -1/4+0+3/4 = 1/2;        dzeta1=-(1/2)/3
    # return biased variance of [dzeta0, dzeta1] -- no sqrt.
    p = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    got = ltsv_oracle(p, col=1, freq_beg=0, freq_end=1, half_window=1)

    def dz(vals: np.ndarray) -> float:
        mean = max(vals.sum() / 3.0, 1e-12)
        s = 0.0
        for v in vals:
            r = v / mean
            s -= r * (r - 1.0)
        return s / 3.0

    d0 = dz(np.array([1.0, 3.0, 5.0]))
    d1 = dz(np.array([2.0, 4.0, 6.0]))
    m = (d0 + d1) / 2.0
    expected = ((d0 - m) ** 2 + (d1 - m) ** 2) / 2.0
    assert got == expected


def test_ltsv_oracle_edge_shrink_at_col_zero() -> None:
    # col=0, R=1 -> window clamps to [0,1] (length 2, no padding).
    p = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    got = ltsv_oracle(p, col=0, freq_beg=0, freq_end=1, half_window=1)

    def dz(vals: np.ndarray) -> float:
        mean = max(vals.sum() / 2.0, 1e-12)
        s = 0.0
        for v in vals:
            r = v / mean
            s -= r * (r - 1.0)
        return s / 2.0

    d0 = dz(np.array([1.0, 3.0]))
    d1 = dz(np.array([2.0, 4.0]))
    m = (d0 + d1) / 2.0
    expected = ((d0 - m) ** 2 + (d1 - m) ** 2) / 2.0
    assert got == expected


# === Task 9: fmath::log + TDC oracles ======================================


def test_fmath_log_oracle_matches_harness_dump_every_entry() -> None:
    # fmath_log_sweep.bin is 2 x N (column-major f64): row0 = input (f32 widened),
    # row1 = fmath::log(input) (f32 result widened). The numpy oracle must match the
    # harness dump bit-for-bit for EVERY sweep entry (double-pinning the port).
    rows, cols, flat = read_bin(FIXTURE_DIR / "fmath_log_sweep.bin")
    assert rows == 2
    # column-major: element (i, j) at index j*rows + i.
    for j in range(cols):
        x = np.float32(flat[j * rows + 0])  # input round-trips exactly through f64
        want = flat[j * rows + 1]  # f32 result stored as f64
        got = float(fmath_log_oracle(x))
        assert got == want, f"col {j}: x={x} oracle={got!r} dump={want!r}"


def test_fmath_log_oracle_at_zero_finite() -> None:
    v = float(fmath_log_oracle(np.float32(0.0)))
    assert math.isfinite(v)
    assert v == float(np.float32(-88.029694))


def test_tdc_oracle_anchor_matches_rust_hand_case() -> None:
    # Mirrors the Rust hand test tdc_fewer_than_three_crossings_zero_crosscorr:
    # all-ones, min_lag=1, max_lag=3, balance=0.5 -> R=[4/5,3/5,2/5], no crossing,
    # score = 0.5 * (-fmath_log(1 - 4/5)).
    got = tdc_oracle([1.0, 1.0, 1.0, 1.0, 1.0], 1, 3, 0.5)
    r_max = 4.0 / 5.0
    log_term = float(fmath_log_oracle(np.float32(1.0 - r_max)))
    expected = 0.5 * (-log_term)
    assert got == expected


def test_tdc_oracle_min_lag_zero_r0_is_one() -> None:
    # min_lag=0 -> R[0] is the lag-0 autocorrelation == 1 -> MaxPeak >= 1 ->
    # fmath_log evaluated at <= 0 (finite, no guard). Same window as the Rust test.
    w = [((i * 7) % 5) - 2.0 for i in range(9)]
    s = tdc_oracle(w, 0, 4, 0.7)
    assert math.isfinite(s)


# === Task 10: InputStatistics merge oracle =================================


def test_stats_merge_oracle_single_batch_anchor() -> None:
    # [[1],[2],[3]] as one chunk (batch, no merge): mean=2.0,
    # std = sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/3).
    mean, std, n = stats_merge_oracle([[[1.0], [2.0], [3.0]]])
    assert n == 3
    assert mean == [2.0]
    assert std == [math.sqrt(((1.0 - 2.0) ** 2 + (2.0 - 2.0) ** 2 + (3.0 - 2.0) ** 2) / 3.0)]


def test_stats_merge_oracle_two_chunks_equals_pooled_formula() -> None:
    # Merge of ([1],[2]) with ([3]): batch1 = {n=2, mean=1.5,
    # std=sqrt(((1-1.5)^2+(2-1.5)^2)/2) = sqrt(0.25)=0.5}; batch2 = {n=1, mean=3,
    # std=0}. Pooled: n=3, mean=(1.5*2+3*1)/3=2.0,
    # std = sqrt(((0.5^2+(2-1.5)^2)*2 + (0^2+(2-3)^2)*1)/3).
    mean, std, n = stats_merge_oracle([[[1.0], [2.0]], [[3.0]]])
    assert n == 3
    expected_mean = (1.5 * 2 + 3.0 * 1) / 3.0
    expected_std = math.sqrt(((0.5**2 + (expected_mean - 1.5) ** 2) * 2 + (0.0**2 + (expected_mean - 3.0) ** 2) * 1) / 3.0)
    assert mean == [expected_mean]
    assert std == [expected_std]

    # NOTE: this merge-of-chunks result is NOT asserted to bit-match a single
    # from_matrix([[1],[2],[3]]) batch (floating-point pooling is not exactly
    # associative) -- only the pooled-merge FORMULA above is the arbiter here.
    # In THIS particular case the values happen to coincide (mean=2.0 either
    # way; sqrt(0.5*2)... ), but that coincidence is not a general guarantee and
    # is not asserted as a cross-equality contract.


def test_stats_merge_oracle_empty_chunk_mid_chain_is_noop() -> None:
    # A middle empty chunk (self.n != 0, other.n == 0) is a silent no-op.
    mean_a, std_a, n_a = stats_merge_oracle([[[1.0], [2.0], [3.0]]])
    mean_b, std_b, n_b = stats_merge_oracle([[[1.0], [2.0], [3.0]], []])
    assert (mean_a, std_a, n_a) == (mean_b, std_b, n_b)


def test_stats_merge_oracle_leading_empty_chunk_is_copy() -> None:
    # A leading empty chunk (self.n == 0, other.n == 0) keeps n=0; the next
    # nonempty chunk then plain-copies in (self.n == 0 branch).
    mean, std, n = stats_merge_oracle([[], [[1.0], [2.0], [3.0]]])
    mean_direct, std_direct, n_direct = stats_merge_oracle([[[1.0], [2.0], [3.0]]])
    assert (mean, std, n) == (mean_direct, std_direct, n_direct)
