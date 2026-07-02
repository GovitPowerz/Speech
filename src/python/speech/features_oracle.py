"""Numpy oracles for the Phase 1 feature kernels.

Independent codings of the kernels the Rust `speech::features` modules implement
sequentially, cross-checked bit-for-bit via the `*_cases.json` fixtures under
`tests/reference_data/phase1/` (emitted by `scripts/extract_phase1_oracle_cases.py`);
the hand anchors in `tests/test_features_oracle.py` are the arbiters.

- `regression_deltas_oracle` / `sdc_oracle` (Task 6/7): MelFilterBank.cpp:153-170,
  224-262. The deltas kernel is
      D[t] = sum_{j=1..n} j * (B[min(t+j, T-1)] - B[max(t-j, 0)]) / (2 * sum_{j} j^2)
  where the denominator sums j^2 over ALL j in 1..n (the legacy `adim` accumulates
  j*j even when T <= j and the shifted copy is skipped -- the clamp saturates but
  the term still contributes).
- `ltsv_oracle` (Task 8): LongTermSpectralVariation.cpp:82-128.
- `fmath_log_oracle` / `tdc_oracle` (Task 9): fmath.hpp:186-226,713-727 and
  TimeDomainCorrel.cpp:36-91.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import NDArray

_LOG_LEN = 11
_LOG_N = 1 << _LOG_LEN  # 2048


@functools.lru_cache(maxsize=1)
def _log_table() -> tuple[np.float32, NDArray[np.float32], NDArray[np.float32]]:
    """Build the 2048-entry fmath log table (fmath.hpp:201-217) with numpy float32
    SCALAR ops for the narrowings and f64 for the table build.

    Per the Task 8 oracle lesson, np.float32 per-op arithmetic is exact IEEE f32 (the
    hazard is only vectorized *reductions*, which this does not use). `c_log2 =
    logf(2f)/2^23` in f32; per bin `x = 1 + i/n` in f64, `app = (f32) ln x`, `rev` is
    the f64 slope `(ln(x+h-e)-ln(x)) / ((h-e)*2^23)` narrowed to f32 (last bin uses
    the exact derivative `1/(x*2^23)`). `h = 2^-11`, `e = 2^-24`.
    """
    c_log2 = np.float32(np.float32(np.log(np.float32(2.0))) / np.float32(1 << 23))
    e = 1.0 / float(1 << 24)
    h = 1.0 / float(1 << _LOG_LEN)
    n = _LOG_N
    scale = float(1 << 23)
    app = np.zeros(n, dtype=np.float32)
    rev = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = 1.0 + float(i) / n
        a = np.log(x)
        app[i] = np.float32(a)
        if i < n - 1:
            b = np.log(x + h - e)
            rev[i] = np.float32((b - a) / ((h - e) * scale))
        else:
            rev[i] = np.float32(1.0 / (x * scale))
    return c_log2, app, rev


def fmath_log_oracle(x: np.float32) -> np.float32:
    """Independent numpy coding of `fmath::log(float)` (fmath.hpp:713-727), the
    scalar f32 table log.

    Replicates the bit-manipulation eval: mask the exponent bits `a`, the top-11
    mantissa bits (table index `idx`), and the low-12 mantissa bits (`b2`); then
    `(a - (127<<23)) * c_log2 + app[idx] + b2 * rev[idx]`, with the exponent
    subtraction done as SIGNED int32 arithmetic before the f32 cast (matching the
    legacy `int a` masked-bits value). No `x <= 0` guard, so `fmath_log_oracle(0.0)`
    is finite. All arithmetic is np.float32 SCALAR (exact per-op IEEE f32).
    """
    c_log2, app, rev = _log_table()
    bits = int(np.float32(x).view(np.uint32))
    a = bits & (0xFF << 23)  # masked exponent bits (unsigned)
    b1 = bits & (0x7FF << 12)
    b2 = bits & 0xFFF
    idx = b1 >> 12
    a_signed = int(np.int32(np.uint32(a)) - np.int32(127 << 23))  # signed i32 subtraction
    # f = float(a - 127<<23)*c_log2 + app[idx] + float(b2)*rev[idx], all f32, left-to-right.
    t1 = np.float32(np.float32(a_signed) * c_log2)
    t2 = np.float32(app[idx])
    t3 = np.float32(np.float32(b2) * np.float32(rev[idx]))
    return np.float32(np.float32(t1 + t2) + t3)


def tdc_oracle(window: list[float], min_lag: int, max_lag: int, balance: float) -> float:
    """Independent Python coding of `TimeDomainCorrel::classifySequence`
    (TimeDomainCorrel.cpp:36-91).

    Uses PLAIN Python loops for every accumulation (squared norm, autocorrelation,
    cross-correlation) -- NOT numpy vectorized reductions -- so the cross-language
    JSON check can demand bit-exactness against the sequential Rust port (numpy's
    pairwise `.sum()` drifts by an ULP or two from a strictly sequential loop; see
    the Task 8 oracle lesson). `R[0]` is the autocorrelation at MIN_LAG (index 0),
    the zero-crossing polarity is keyed on its sign with the mixed strict/non-strict
    comparisons copied verbatim, the `mm=0` cross term double-counts (hence `/2`),
    `count` is a float, and `(1 - MaxPeak)` is narrowed to f32 before `fmath_log`
    then widened back.
    """
    n = len(window)
    adim = 0.0
    for v in window:
        adim += v * v
    if adim < 1e-12:
        adim = 1e-12

    r = [0.0] * (max_lag - min_lag + 1)
    max_peak = -1e20
    for k in range(min_lag, max_lag + 1):
        length = n - k
        s = 0.0
        for i in range(length):
            s += window[i] * window[i + k]
        r[k - min_lag] = s / adim
        if r[k - min_lag] > max_peak:
            max_peak = r[k - min_lag]

    cross_corr = 0.0
    period1_start = 0
    period2_start = 0
    count = 0.0
    for ll in range(0, max_lag - min_lag):
        crossing = (r[0] >= 0 and r[ll] > 0 and r[ll + 1] <= 0) or (r[0] < 0 and r[ll] < 0 and r[ll + 1] >= 0)
        if crossing:
            if period1_start == 0:
                period1_start = ll + 1
            elif period2_start == 0:
                period2_start = ll + 1
            else:
                period3_start = ll + 1
                min_size = period2_start - period1_start + 1
                size2 = period3_start - period2_start + 1
                if min_size > size2:
                    min_size = size2
                tmp_xcorr = -1e12
                for mm in range(min_size):
                    xcor = 0.0
                    for nn in range(min_size - mm):
                        xcor += r[period1_start + nn] * r[period2_start + nn + mm] + r[period1_start + mm + nn] * r[period2_start + nn]
                    if tmp_xcorr < xcor:
                        tmp_xcorr = xcor
                cross_corr += tmp_xcorr / 2
                period1_start = period2_start
                period2_start = period3_start
                count += 1
    if count != 0:
        cross_corr /= count
    log_term = float(fmath_log_oracle(np.float32(1.0 - max_peak)))
    return balance * (-log_term) + (1.0 - balance) * cross_corr


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
