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

use ndarray::Array2;

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
