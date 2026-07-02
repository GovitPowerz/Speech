"""Numpy oracle for the shared regression-deltas kernel (Task 6).

Independent (vectorized) coding of the same clamped-formula kernel that the Rust
`speech::features::mel::regression_deltas` implements sequentially. The two are
cross-checked bit-for-bit via `tests/reference_data/phase1/deltas_cases.json`
(emitted by `scripts/extract_phase1_oracle_cases.py`); the hand anchor in
`tests/test_features_oracle.py` is the arbiter.

Legacy source: MelFilterBank.cpp:153-170 (the deltas block-op sequence). The
kernel is

    D[t] = sum_{j=1..n} j * (B[min(t+j, T-1)] - B[max(t-j, 0)]) / (2 * sum_{j} j^2)

where the denominator sums j^2 over ALL j in 1..n (the legacy `adim` accumulates
j*j even when T <= j and the shifted copy is skipped -- the clamp saturates but
the term still contributes).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ltsv_oracle(p: NDArray[np.float64], col: int, freq_beg: int, freq_end: int, half_window: int) -> float:
    """Independent numpy coding of `LongTermSpectralVariation::classifySequence`
    (LongTermSpectralVariation.cpp:82-128).

    `p` is (time x freq), matching the Rust `Array2<f64>` convention (rows=time,
    cols=freq bins). Context window is `[max(0,col-R), min(T-1,col+R)]` INCLUSIVE
    both ends (length shrinks at the edges, no padding). Per freq bin in
    `[freq_beg, freq_end]` inclusive: mean over the window, floored at 1e-12 (the
    mean only, never the numerator); dzeta[bin] = -(1/length) * sum_t r*(r-1) with
    r = p[t,bin]/mean. Returns the BIASED VARIANCE of dzeta over the bins (divide
    by nbBins, NO sqrt) -- the legacy comment says "standard deviation" and lies.

    The per-bin/per-sample reductions use plain Python accumulation (not
    `ndarray.sum`, whose pairwise/blocked reduction can reorder additions and
    drift by an ULP or two from a strictly sequential accumulation) so the
    cross-language JSON check can demand bit-exactness against the sequential
    Rust port.
    """
    t = p.shape[0]
    beg = max(0, col - half_window)
    end = min(t - 1, col + half_window)
    length = float(end - beg + 1)
    nb_bins = freq_end - freq_beg + 1

    mean_value = [0.0] * (freq_end + 1)
    for row in range(freq_beg, freq_end + 1):
        m = 0.0
        for ii in range(beg, end + 1):
            m += float(p[ii, row])
        m /= length
        if m < 1e-12:
            m = 1e-12
        mean_value[row] = m

    dzeta = [0.0] * (freq_end + 1)
    mean_dzeta = 0.0
    for row in range(freq_beg, freq_end + 1):
        d = 0.0
        for ii in range(beg, end + 1):
            tmp_log = float(p[ii, row]) / mean_value[row]
            d -= tmp_log * (tmp_log - 1.0)
        d /= length
        dzeta[row] = d
        mean_dzeta += d
    mean_dzeta /= nb_bins

    std_dzeta = 0.0
    for row in range(freq_beg, freq_end + 1):
        tmp = dzeta[row] - mean_dzeta
        std_dzeta += tmp * tmp
    std_dzeta /= nb_bins

    return std_dzeta


def regression_deltas_oracle(base: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    t = base.shape[0]
    acc = np.zeros_like(base)
    denom = 0.0
    for j in range(1, n + 1):
        plus = base[np.minimum(np.arange(t) + j, t - 1)]
        minus = base[np.maximum(np.arange(t) - j, 0)]
        acc += j * (plus - minus)
        denom += j * j
    return acc / (2.0 * denom)


def sdc_oracle(mfcc: NDArray[np.float64], nb_dct: int) -> NDArray[np.float64]:
    """Shifted-Delta-Cepstra block portion (legacy applyDCT Branch A stacking).

    Independent coding of the SDC scratch-and-extract in MelFilterBank.cpp:224-262:
    the delta of the statics (n=3 regression kernel, adim 28) is stacked k=7 times
    into a (T + k*P) x (k*nb_dct) scratch at vertical offset kk*P (P=3), then the
    block starting at row (k*P-1)/2 = 10 is extracted. This yields the SDC band ONLY
    (T x 7*nb_dct); the leading statics and the ignoreFirst clobber are the caller's
    concern. Out of the extraction window the scratch is zero, so those cells are 0.

    The Rust cross-check (phase1_mel_golden.rs) reproduces this via the closed-form
    per-block time offset out[t, block kk] = D[t + 10 - 3*kk] (zero-padded); this
    scratch-based coding is the independent arbiter.
    """
    t = mfcc.shape[0]
    d, p, k = 3, 3, 7
    delta = regression_deltas_oracle(mfcc[:, :nb_dct], d)  # T x nb_dct
    scratch = np.zeros((t + k * p, k * nb_dct))
    for kk in range(k):
        scratch[kk * p : kk * p + t, kk * nb_dct : (kk + 1) * nb_dct] = delta
    start = (k * p - 1) // 2  # 10
    return scratch[start : start + t, :].copy()
