//! Long-Term Spectral Variation + time-domain autocorrelation pitch features.
//!
//! Ported from legacy C++: LongTermSpectralVariation.*, TimeDomainCorrel.*. Phase 1.
//!
//! Task 8 (`ltsv_classify_sequence`, `get_ltsv`): ported from
//! `LongTermSpectralVariation.cpp:82-128` (`classifySequence`) and
//! `BLSTMSpectralSegmenter.cpp:316-339` (`getLTSV`). `p` is (time x freq),
//! matching the harness/oracle convention (rows = time frames, cols = freq bins).
//!
//! Parity hazards (all load-bearing, golden-tested):
//! - Context window `[max(0,col-R), min(T-1,col+R)]` is INCLUSIVE both ends and
//!   SHRINKS at the edges (no padding) -- do not pad to a fixed length.
//! - Per-bin mean is floored at 1e-12; the floor applies to the MEAN only, never
//!   to the numerator or to `r = p[t,bin]/mean`.
//! - `dzeta[bin]` accumulates `-r*(r-1)` over the window, THEN divides by the
//!   window length, THEN is added into `mean_dzeta` -- that exact order.
//! - The return is the BIASED VARIANCE of `dzeta` over the bins (divide by
//!   nbBins, NO sqrt): the legacy names it `std_dzeta` and comments "standard
//!   deviation", but the code never takes a square root. Reproduce the lie.
//! - `get_ltsv` computes at `jj = 0, shift, 2*shift, ...` and for `jj != 0`
//!   backfills the skipped rows `mm = 1..shift-1` via linear interpolation
//!   (`coeffInterp = mm/shift`); trailing rows after the last computed `jj`
//!   stay 0.0 (no backfill past the final computed sample).
//!
//! Task 9 (`fmath_log`, `tdc_classify_sequence`, `compute_pitch`, `get_pitch`,
//! `apply_homothety`): ported from `fmath.hpp:186-226,713-727` (the 2048-entry
//! f32 table log), `TimeDomainCorrel.cpp:36-91` (`classifySequence`),
//! `BLSTMSpectralSegmenter.cpp:172-192` (`computePitch`), `:399-437` (`getPitch`),
//! and `:762-775` (the homothety warp).
//!
//! Parity hazards (all load-bearing, golden-tested):
//! - `fmath_log` is a bit-trick f32 approximation, NOT `libm` log. The eval reads
//!   the f32 bit pattern, masks the exponent, top-11 mantissa (the table index)
//!   and low-12 mantissa bits, and reconstructs `log(x)` from a piecewise-linear
//!   table. The exponent term is `(a - (127<<23))` done as SIGNED i32 arithmetic
//!   before the `as f32` cast (matching the legacy `int a = ...` + `float(a - ...)`).
//!   There is NO `x <= 0` guard: `fmath_log(0.0)` is finite (~-88.0297).
//! - `c_log2 = ln(2f32)/2^23` is computed in f32 (the whole eval is f32); the
//!   table `app`/`rev` are built in f64 then narrowed to f32 at the exact points
//!   the legacy narrows (see `build_log_table`).
//! - TDC `R[0]` is the autocorrelation at MIN_LAG (index 0 of `R`), not lag 0. The
//!   zero-crossing polarity is keyed on `R[0]`'s sign with mixed strict/non-strict
//!   comparisons copied verbatim. The cross-correlation `mm=0` term double-counts
//!   (both product orders coincide), hence the `/2`.
//! - `count` is a DOUBLE in the legacy (it divides `CrossCorr` by it).
//! - The final `(1.0 - MaxPeak)` is NARROWED to f32 before `fmath_log`, then the
//!   f32 result is widened back to f64 -- reproduce the narrow/widen exactly.
//! - `compute_pitch` returns the legacy `frameRate/indiceMaxPeak` -- `long / long`
//!   INTEGER division (truncated estimate); the accept bounds are double divisions.
//! - `get_pitch`: SPEECH segments only; `numberOfFrames == 0 -> divide by 1` (so
//!   an empty accumulation returns the raw sum 0.0, not NaN). Frame range is
//!   `[begin*rate + half_window, end*rate - half_window]` inclusive, stepping shift.
//! - `apply_homothety` warps COLUMNS (frequency bins): `out(t,ii) = (1-alpha)*
//!   p(t,pos) + alpha*p(t,pos+1)` with `pos = (int)(coeff*ii)`, guarded by
//!   `pos < cols` / `pos+1 < cols` (signed comparisons, `pos >= 0` always true here).

use std::sync::LazyLock;

use ndarray::Array2;

use crate::audio::{Audio, get_sequence};
use crate::features::pipeline::TdcParams;
use crate::tasks::segmentation::{SegClass, Segmentation};

/// `LongTermSpectralVariation::classifySequence` (LongTermSpectralVariation.cpp:
/// 82-128), transcribed as a free function over `p` (time x freq). `col` is the
/// legacy `column`, `half_window` is the legacy `window_size`. The `row` loops
/// index by absolute freq-bin position (the parity contract, matching the
/// legacy's `freq_beg..=freq_end` walk), not a range-loop smell.
#[allow(clippy::needless_range_loop)]
pub fn ltsv_classify_sequence(
    p: &Array2<f64>,
    col: usize,
    freq_beg: usize,
    freq_end: usize,
    half_window: usize,
) -> f64 {
    let periodogram_cols = p.nrows(); // legacy periodogram_cols == TIME axis length
    let beg_conv = col.saturating_sub(half_window);
    let mut end_conv = col + half_window;
    if end_conv >= periodogram_cols {
        end_conv = periodogram_cols - 1;
    }
    let length = (end_conv - beg_conv + 1) as f64;
    let nb_rows = (freq_end - freq_beg + 1) as f64;

    let mut mean_value = vec![0.0_f64; p.ncols()];
    for row in freq_beg..=freq_end {
        let mut m = 0.0;
        for ii in beg_conv..=end_conv {
            m += p[[ii, row]];
        }
        m /= length;
        if m < 1e-12 {
            m = 1e-12;
        }
        mean_value[row] = m;
    }

    let mut dzeta = vec![0.0_f64; p.ncols()];
    let mut mean_dzeta = 0.0_f64;
    for row in freq_beg..=freq_end {
        let mut d = 0.0;
        for ii in beg_conv..=end_conv {
            let tmp_log = p[[ii, row]] / mean_value[row];
            d -= tmp_log * (tmp_log - 1.0);
        }
        d /= length;
        dzeta[row] = d;
        mean_dzeta += d;
    }
    mean_dzeta /= nb_rows;

    // Compute standard deviation (legacy comment -- it is actually variance, no sqrt).
    let mut std_dzeta = 0.0_f64;
    for row in freq_beg..=freq_end {
        let tmp = dzeta[row] - mean_dzeta;
        std_dzeta += tmp * tmp;
    }
    std_dzeta /= nb_rows;

    std_dzeta
}

/// `BLSTMSpectralSegmenter::getLTSV` loop (BLSTMSpectralSegmenter.cpp:316-339),
/// transcribed as a free function. Returns a `p.nrows()`-length column with
/// linear-interpolation backfill between computed samples and trailing zeros
/// after the last computed sample.
pub fn get_ltsv(
    p: &Array2<f64>,
    freq_beg: usize,
    freq_end: usize,
    half_window: usize,
    shift: usize,
) -> Vec<f64> {
    let vec_size = p.nrows();
    let mut ltsv = vec![0.0_f64; vec_size];
    let mut jj = 0_usize;
    while jj < vec_size {
        ltsv[jj] = ltsv_classify_sequence(p, jj, freq_beg, freq_end, half_window);
        if jj != 0 {
            for mm in 1..shift {
                let coeff_interp = (mm as f64) / (shift as f64);
                ltsv[jj - shift + mm] =
                    (1.0 - coeff_interp) * ltsv[jj - shift] + coeff_interp * ltsv[jj];
            }
        }
        jj += shift;
    }
    ltsv
}

// --- fmath::log (f32 table log) --------------------------------------------

/// `LOG_TABLE_SIZE - 1` (`fmath.hpp:72,189`): the table has `1 << 11 == 2048`
/// entries, indexed by the top 11 mantissa bits.
const LOG_LEN: u32 = 11;
const LOG_N: usize = 1 << LOG_LEN; // 2048

/// One table row: `app` = ln at the bin's left edge, `rev` = the reciprocal-scaled
/// slope used to interpolate within the bin.
struct LogTable {
    c_log2: f32,
    app: [f32; LOG_N],
    rev: [f32; LOG_N],
}

/// Build the 2048-entry table exactly as `fmath::LogVar` (`fmath.hpp:201-217`):
/// `c_log2 = logf(2f)/2^23` in f32; per bin `x = 1 + i/n` in f64, `app = (f32) ln x`,
/// and `rev` is the f64 slope `(ln(x+h-e)-ln(x)) / ((h-e)*2^23)` narrowed to f32
/// (the last bin uses the exact derivative `1/(x*2^23)`). `h = 2^-11`, `e = 2^-24`.
fn build_log_table() -> LogTable {
    let c_log2 = (2.0f32).ln() / ((1u32 << 23) as f32);
    let e = 1.0 / ((1u64 << 24) as f64);
    let h = 1.0 / ((1u64 << LOG_LEN) as f64);
    let n = LOG_N;
    let scale = (1u32 << 23) as f64;
    let mut app = [0.0f32; LOG_N];
    let mut rev = [0.0f32; LOG_N];
    for i in 0..n {
        let x = 1.0 + (i as f64) / (n as f64);
        let a = x.ln();
        app[i] = a as f32;
        if i < n - 1 {
            let b = (x + h - e).ln();
            rev[i] = ((b - a) / ((h - e) * scale)) as f32;
        } else {
            rev[i] = (1.0 / (x * scale)) as f32;
        }
    }
    LogTable { c_log2, app, rev }
}

static LOG_TABLE: LazyLock<LogTable> = LazyLock::new(build_log_table);

/// `fmath::log(float)` (`fmath.hpp:713-727`): the scalar f32 table-log.
///
/// Reads the f32 bit pattern, extracts the exponent bits `a`, the table index `idx`
/// (top `LOG_LEN` mantissa bits), and the sub-bin residual `b2` (low `23 - LOG_LEN`
/// mantissa bits), then reconstructs `(a - (127<<23)) * c_log2 + app[idx] + b2 *
/// rev[idx]`. The exponent subtraction is SIGNED i32 (the legacy `int a` masked-bits
/// value); no `x <= 0` guard, so `fmath_log(0.0)` is finite.
pub fn fmath_log(x: f32) -> f32 {
    let t = &*LOG_TABLE;
    let bits = x.to_bits();
    let a = (bits & (0xFFu32 << 23)) as i32; // masked exponent bits, as int
    let b1 = bits & (0x7FFu32 << 12); // mask(LOG_LEN) << (23 - LOG_LEN)
    let b2 = bits & 0xFFF; // mask(23 - LOG_LEN)
    let idx = (b1 >> 12) as usize;
    ((a - (127i32 << 23)) as f32) * t.c_log2 + t.app[idx] + (b2 as f32) * t.rev[idx]
}

// --- Time-domain correlation ------------------------------------------------

/// Autocorrelation `R[k-min_lag]` over `window` for `k` in `min_lag..=max_lag`,
/// normalized by the floored squared norm. Sequential sums (the parity contract).
/// Shared by `tdc_classify_sequence` and `compute_pitch`.
fn autocorrelation(window: &[f64], min_lag: usize, max_lag: usize) -> Vec<f64> {
    let len = window.len();
    let mut adim = 0.0;
    for &v in window {
        adim += v * v;
    }
    if adim < 1e-12 {
        adim = 1e-12;
    }
    let mut r = vec![0.0f64; max_lag - min_lag + 1];
    for k in min_lag..=max_lag {
        let length = len - k;
        let mut s = 0.0;
        for i in 0..length {
            s += window[i] * window[i + k];
        }
        r[k - min_lag] = s / adim;
    }
    r
}

/// `TimeDomainCorrel::classifySequence` (`TimeDomainCorrel.cpp:36-91`), transcribed
/// as a free function (`_Balance` -> `balance`). `window` is the windowed signal
/// (length `full_window`). Returns `balance*(-fmath_log(1-MaxPeak)) + (1-balance)*
/// CrossCorr` -- note the f64->f32 narrowing of `(1-MaxPeak)` before the log.
pub fn tdc_classify_sequence(window: &[f64], min_lag: usize, max_lag: usize, balance: f64) -> f64 {
    let r = autocorrelation(window, min_lag, max_lag);

    // Max correlation peak (plain sequential max, init -1e20).
    let mut max_peak = -1e20f64;
    for &v in &r {
        if v > max_peak {
            max_peak = v;
        }
    }

    // Cross-correlation over zero crossings. `count` is a DOUBLE (legacy).
    let mut cross_corr = 0.0f64;
    let mut period1_start = 0usize;
    let mut period2_start = 0usize;
    let mut count = 0.0f64;
    // ll in 0..(max_lag-min_lag) EXCLUSIVE; crossings keyed on R[0]'s sign.
    for ll in 0..(max_lag - min_lag) {
        let crossing = (r[0] >= 0.0 && r[ll] > 0.0 && r[ll + 1] <= 0.0)
            || (r[0] < 0.0 && r[ll] < 0.0 && r[ll + 1] >= 0.0);
        if crossing {
            if period1_start == 0 {
                period1_start = ll + 1;
            } else if period2_start == 0 {
                period2_start = ll + 1;
            } else {
                let period3_start = ll + 1;
                let mut min_size = period2_start - period1_start + 1;
                let size2 = period3_start - period2_start + 1;
                if min_size > size2 {
                    min_size = size2;
                }
                let mut tmp_xcorr = -1e12f64;
                for mm in 0..min_size {
                    let mut xcor = 0.0f64;
                    for nn in 0..(min_size - mm) {
                        xcor += r[period1_start + nn] * r[period2_start + nn + mm]
                            + r[period1_start + mm + nn] * r[period2_start + nn];
                    }
                    if tmp_xcorr < xcor {
                        tmp_xcorr = xcor;
                    }
                }
                cross_corr += tmp_xcorr / 2.0;
                period1_start = period2_start;
                period2_start = period3_start;
                count += 1.0;
            }
        }
    }
    if count != 0.0 {
        cross_corr /= count;
    }
    balance * (-(fmath_log((1.0 - max_peak) as f32) as f64)) + (1.0 - balance) * cross_corr
}

/// `BLSTMSpectralSegmenter::computePitch` (`BLSTMSpectralSegmenter.cpp:172-192`),
/// transcribed as a free function with the accept/reject bounds folded in from
/// `getPitch` (`:425`).
///
/// QUIRK (load-bearing, golden-pinned): the legacy signature is `computePitch(long
/// min_lag, long max_lag, long frameRate, ...)` and returns `frameRate/indiceMaxPeak`
/// -- `long / long`, i.e. INTEGER division, widened to double only on return. The
/// estimate is therefore the TRUNCATED `rate/argmax_lag` (8000/35 = 228, not
/// 228.57...). The accept bounds, by contrast, ARE double divisions (`getPitch`
/// casts both operands at :425): accepted iff strictly inside `(rate/max_lag,
/// rate/min_lag)`, else `None`. Tracked in IMPROVEMENTS.md.
pub fn compute_pitch(window: &[f64], min_lag: usize, max_lag: usize, rate: f64) -> Option<f64> {
    let r = autocorrelation(window, min_lag, max_lag);

    let mut max_peak = -1e20f64;
    let mut indice_max_peak = min_lag;
    for k in min_lag..=max_lag {
        let v = r[k - min_lag];
        if v > max_peak {
            max_peak = v;
            indice_max_peak = k;
        }
    }
    // long/long INTEGER division (the legacy `frameRate/indiceMaxPeak`).
    let est = ((rate as i64) / (indice_max_peak as i64)) as f64;
    if est > rate / (max_lag as f64) && est < rate / (min_lag as f64) {
        Some(est)
    } else {
        None
    }
}

/// `BLSTMSpectralSegmenter::getPitch` (`BLSTMSpectralSegmenter.cpp:399-437`).
///
/// Iterate the SPEECH segments (`it+1 != end` walk; a segment spans `[it.begin,
/// next.begin)`), stepping `tdc.shift` over `[begin*rate + half_window,
/// end*rate - half_window]` (inclusive, both ends converted with the legacy
/// truncating `size_type` casts), accumulating accepted pitch estimates. Divides
/// by the accepted count, or by 1 when the count is 0 (returning the raw sum 0.0).
pub fn get_pitch(
    audio: &Audio,
    seg: &Segmentation,
    chan: usize,
    tdc: &TdcParams,
    rate: f64,
) -> f64 {
    let mut number_of_frames = 0usize;
    let mut pitch = 0.0f64;
    if tdc.half_window == 0 {
        return 0.0;
    }
    let coeffs = tdc.coeffs.as_deref();
    let mut buf = Array2::<f64>::zeros((1, tdc.full_window));
    let segs = seg.segments();
    // while (it+1 != end): the End sentinel is the last element.
    for i in 0..segs.len().saturating_sub(1) {
        if segs[i].ty != SegClass::Speech {
            continue;
        }
        let row_begin = (segs[i].begin * rate) as usize + tdc.half_window;
        let mut row_end = (segs[i + 1].begin * rate) as usize;
        if row_end > tdc.half_window {
            row_end -= tdc.half_window;
        } else {
            row_end = 0;
        }
        if row_end >= row_begin {
            let mut jj = row_begin;
            while jj <= row_end {
                get_sequence(
                    &audio.data,
                    chan,
                    jj,
                    tdc.half_window,
                    false,
                    coeffs,
                    &mut buf,
                );
                let seq = buf.row(0).to_vec();
                if let Some(est) = compute_pitch(&seq, tdc.min_lag, tdc.max_lag, rate) {
                    pitch += est;
                    number_of_frames += 1;
                }
                jj += tdc.shift;
            }
        }
    }
    if number_of_frames == 0 {
        number_of_frames = 1;
    }
    pitch / (number_of_frames as f64)
}

/// Homothety (frequency-axis) warp of a periodogram (`BLSTMSpectralSegmenter.cpp:
/// 762-775`). `out(t,ii) = (1-alpha)*p(t,pos) + alpha*p(t,pos+1)` with `pos =
/// (int)(coeff*ii)`, `alpha = coeff*ii - pos`, each term gated on its column index
/// lying in `[0, cols)`. Warps COLUMNS (freq bins); rows (time frames) pass through.
pub fn apply_homothety(p: &Array2<f64>, coeff: f64) -> Array2<f64> {
    let (rows, cols) = p.dim();
    let cols_i = cols as isize;
    let mut out = Array2::<f64>::zeros((rows, cols));
    for ii in 0..cols {
        let pos = (coeff * ii as f64) as isize; // C++ (int) truncation toward zero
        let alpha = coeff * ii as f64 - pos as f64;
        for jj in 0..rows {
            let mut tmp = 0.0;
            if pos >= 0 && pos < cols_i {
                tmp += (1.0 - alpha) * p[[jj, pos as usize]];
            }
            if pos + 1 >= 0 && pos + 1 < cols_i {
                tmp += alpha * p[[jj, (pos + 1) as usize]];
            }
            out[[jj, ii]] = tmp;
        }
    }
    out
}
