//! Triangular Mel filterbank (1125/700, unit-peak), regression deltas.
//!
//! Ported from legacy C++: `MelFilterBank.*` (ctor `MelFilterBank.cpp:17-128`,
//! `applyFilterBank` `:130-216`, `Hz2Mel`/`Mel2Hz` `:362-368`). Phase 1 Task 6.
//!
//! Parity hazards (all load-bearing, golden-tested):
//! - Grids are built by SEQUENTIAL accumulation (`freq += step`), never `i*step`;
//!   the accumulated FP error is part of the contract.
//! - `_BegFreq`/`_EndFreq` are `floor`/`ceil` of caller-snapped freqs (a double
//!   round-trip through `freqStep`); reproduce literally.
//! - Triangle membership is INCLUSIVE on both edges; the center bin takes the
//!   FALLING branch (coeff exactly 1.0); unit peak, no normalization.
//! - WHOLE-BANK FALLBACK: if ANY filter collects zero bins, the entire bank
//!   collapses to raw-band pass-through (`nb_filters = end - beg + 1`, not mel).
//! - log+mel uses `ln(dot + 1e-24)` (additive floor INSIDE the log).
//! - The deltas-no-DCT output layout is `[delta | delta | delta-delta]`: the
//!   static log-mel block is OVERWRITTEN by the delta block (`:192-193`).
//!
//! Task 7 (`apply_dct`, `MelFilterBank.cpp:218-360`): DCT-II MFCCs + deltas + SDC.
//! Parity hazards (all load-bearing, golden-tested):
//! - The DCT product `melPeriodogram * _CoeffsDCT` is the feature path's ONE real
//!   Eigen GEMM. The mandatory harness pre-check found Eigen's blocked kernel does
//!   NOT accumulate in ascending order; per the Task 7 brief the product is done as
//!   an explicit ascending triple loop (`dct_product`), and the goldens are dumped
//!   with that order. Do NOT swap in a BLAS/`ndarray` `.dot()` -- the order matters.
//! - SDC (Branch A): the delta uses the n=3 regression kernel REGARDLESS of the
//!   `deltas_nb < 0` trigger; blocks stack at vertical offset `3*kk` in a
//!   `(T+21) x 7*nb_dct` scratch, extracted at row `(k*P-1)/2 = 10` -> per-block
//!   time offsets `{+10,+7,+4,+1,-2,-5,-8}`, zero-padded out of range.
//! - `ignoreFirst` in SDC clobbers the LAST static column: statics are written
//!   first, then `rightCols(7*nb_dct)` starting at col `nb_dct-1` overwrites c_{N-1}
//!   (c0 kept, the last static lost). Port the write order exactly.
//! - Branch B (`ignoreFirst && deltas_nb >= 0`) computes into a `T x (width+1)`
//!   temp then drops the FIRST column (static c0 only; delta-of-c0 kept).

use ndarray::Array2;

/// `Hz2Mel(f) = 1125 * ln(1 + f/700)` (plain double std math).
pub fn hz_to_mel(f: f64) -> f64 {
    1125.0 * (1.0 + f / 700.0).ln()
}

/// `Mel2Hz(m) = 700 * (exp(m/1125) - 1)`.
pub fn mel_to_hz(m: f64) -> f64 {
    700.0 * ((m / 1125.0).exp() - 1.0)
}

pub struct MelFilterBank {
    is_log: bool,
    is_mel: bool,
    is_dct_activated: bool,
    ignore_first_dct: bool,
    nb_filters: usize,
    nb_dct: i32,
    compute_deltas_nb: i32,
    compute_delta_deltas_nb: i32,
    beg_freq: usize,
    index_begin: Vec<usize>,
    coeffs: Vec<Vec<f64>>,
    coeffs_dct: Array2<f64>,
}

impl MelFilterBank {
    // Sequential index walks (`jj` over `freqs`) are the parity contract, not a
    // range-loop smell; the coeff/index run is inherently index-driven.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn new(
        min_mel: f64,
        max_mel: f64,
        nb_bins: i32,
        min_freq: f64,
        max_freq: f64,
        rate: f64,
        spectrum_size: usize,
        is_log: bool,
        nb_dct: i32,
        ignore_first: bool,
        deltas_nb: i32,
        delta_deltas_nb: i32,
    ) -> MelFilterBank {
        // freqs grid: SEQUENTIAL accumulation `freq += freqStep` while <= rate/2.
        let freq_step = rate / 2.0 / spectrum_size as f64;
        let mut freqs = Vec::new();
        let mut freq = 0.0;
        let freq_limit = rate / 2.0;
        while freq <= freq_limit {
            freqs.push(freq);
            freq += freq_step;
        }

        let mut is_valid_bank = true;
        // Double round-trip: floor/ceil of the caller-snapped band over freqStep.
        let beg_freq = (min_freq.max(0.0) / freq_step).floor() as usize;
        let end_freq = (max_freq.min(rate / 2.0) / freq_step).ceil() as usize;

        let nb_filters: usize;
        let is_mel: bool;
        let mut index_begin: Vec<usize> = Vec::new();
        let mut coeffs: Vec<Vec<f64>> = Vec::new();

        if nb_bins > 0 {
            // mels grid: down-then-up walk. melStep floored at Hz2Mel(1.0).
            let mut mels: Vec<f64> = Vec::new();
            let min_mel_v = hz_to_mel(min_mel);
            let max_mel_v = hz_to_mel(max_mel);
            let mut mel_step = (max_mel_v - min_mel_v) / (nb_bins as f64 + 1.0);
            if mel_step < hz_to_mel(1.0) {
                mel_step = hz_to_mel(1.0);
            }
            let mut mel = min_mel_v;
            while mel > hz_to_mel(min_freq.max(0.0)) - mel_step {
                mel -= mel_step;
            }
            let limit = hz_to_mel(max_freq.min(rate / 2.0));
            while mel < limit + 2.0 * mel_step {
                mels.push(mel_to_hz(mel));
                mel += mel_step;
            }

            // Triangles: filter ii spans [mels[ii-1], mels[ii+1]]; INCLUSIVE edges;
            // center-bin -> falling branch (coeff 1.0); unit peak, no normalization.
            let mut ii: usize = 1;
            while is_valid_bank && ii <= mels.len() - 2 {
                let mut tmp: Vec<f64> = Vec::new();
                for jj in beg_freq..=end_freq {
                    if freqs[jj] >= mels[ii - 1] && freqs[jj] <= mels[ii + 1] {
                        if tmp.is_empty() {
                            index_begin.push(jj);
                        }
                        let coeff = if freqs[jj] < mels[ii] {
                            (freqs[jj] - mels[ii - 1]) / (mels[ii] - mels[ii - 1])
                        } else {
                            (mels[ii + 1] - freqs[jj]) / (mels[ii + 1] - mels[ii])
                        };
                        tmp.push(coeff);
                    }
                }
                if !tmp.is_empty() {
                    coeffs.push(tmp);
                } else {
                    is_valid_bank = false;
                }
                ii += 1;
            }
            if is_valid_bank {
                nb_filters = index_begin.len();
                is_mel = true;
            } else {
                nb_filters = end_freq - beg_freq + 1;
                is_mel = false;
            }
        } else {
            nb_filters = end_freq - beg_freq + 1;
            is_mel = false;
        }

        // DCT coefficient table (stored now; consumed by Task 7 apply_dct).
        let mut nb_dct = nb_dct;
        let is_dct_activated;
        let coeffs_dct;
        if nb_dct <= 0 {
            is_dct_activated = false;
            coeffs_dct = Array2::<f64>::zeros((0, 0));
        } else {
            is_dct_activated = true;
            if nb_dct > nb_filters as i32 {
                nb_dct = nb_filters as i32;
            }
            let mut mat = Array2::<f64>::zeros((nb_filters, nb_dct as usize));
            for col in 0..nb_filters {
                for row in 0..nb_dct as usize {
                    mat[[col, row]] = (std::f64::consts::PI / nb_filters as f64
                        * (col as f64 + 0.5)
                        * row as f64)
                        .cos();
                }
            }
            coeffs_dct = mat;
        }

        MelFilterBank {
            is_log,
            is_mel,
            is_dct_activated,
            ignore_first_dct: ignore_first,
            nb_filters,
            nb_dct,
            compute_deltas_nb: deltas_nb,
            compute_delta_deltas_nb: delta_deltas_nb,
            beg_freq,
            index_begin,
            coeffs,
            coeffs_dct,
        }
    }

    /// Total output columns (getNbFilters, `MelFilterBank.h:32-44`): the raw
    /// `_NbFilters`, doubled/tripled when deltas-without-DCT are active.
    pub fn nb_filters(&self) -> usize {
        if !self.is_dct_activated && self.compute_deltas_nb > 0 {
            if self.compute_delta_deltas_nb > 0 {
                3 * self.nb_filters
            } else {
                2 * self.nb_filters
            }
        } else {
            self.nb_filters
        }
    }

    pub fn is_mel(&self) -> bool {
        self.is_mel
    }

    pub fn is_dct_activated(&self) -> bool {
        self.is_dct_activated
    }

    /// Output column count of `apply_dct` (getNbDCT, `MelFilterBank.h:45-89`).
    /// deltas>0: `(dd>0 ? 3 : 2)*nb_dct`, minus 1 if ignoreFirst. deltas<0 (SDC):
    /// `(ignoreFirst ? nb_dct-1 : nb_dct) + 7*nb_dct`. else: `nb_dct` minus 1 if
    /// ignoreFirst.
    pub fn nb_dct(&self) -> usize {
        let nb = self.nb_dct as usize;
        if self.compute_deltas_nb > 0 {
            let mult = if self.compute_delta_deltas_nb > 0 {
                3
            } else {
                2
            };
            mult * nb - usize::from(self.ignore_first_dct)
        } else if self.compute_deltas_nb < 0 {
            (nb - usize::from(self.ignore_first_dct)) + 7 * nb
        } else {
            nb - usize::from(self.ignore_first_dct)
        }
    }

    /// Explicit ascending triple-loop `melPeriodogram * _CoeffsDCT`
    /// (`MelFilterBank.cpp:223` under `EIGEN_DONT_VECTORIZE`, order verified). The
    /// legacy uses an Eigen GEMM; the harness pre-check proved that GEMM diverges
    /// from ascending accumulation, so the goldens (and this port) use this order.
    fn dct_product(&self, mel: &Array2<f64>) -> Array2<f64> {
        let t = mel.nrows();
        let n = self.coeffs_dct.nrows();
        let k = self.coeffs_dct.ncols();
        let mut out = Array2::<f64>::zeros((t, k));
        for tt in 0..t {
            for kk in 0..k {
                let mut acc = 0.0;
                for nn in 0..n {
                    acc += mel[[tt, nn]] * self.coeffs_dct[[nn, kk]];
                }
                out[[tt, kk]] = acc;
            }
        }
        out
    }

    /// Apply the DCT (`applyDCT`, `MelFilterBank.cpp:218-360`). `mel` is the
    /// `T x nb_filters` log-mel; returns the `T x nb_dct()` MFCC. Three branches:
    /// Branch A (deltas < 0) = SDC; Branch B (ignoreFirst, deltas >= 0) = drop c0;
    /// Branch C (!ignoreFirst, deltas >= 0) = statics + `[c|dc|(ddc)]`.
    pub fn apply_dct(&self, mel: &Array2<f64>) -> Array2<f64> {
        let t = mel.nrows();
        let nb = self.nb_dct as usize;
        let width = self.nb_dct();
        let mut mfcc = Array2::<f64>::zeros((t, width));
        let product = self.dct_product(mel); // T x nb

        if self.compute_deltas_nb < 0 {
            // Branch A - SDC. d=3, P=3, k=7; delta = n=3 kernel REGARDLESS of sign.
            let (d, p, k) = (3i32, 3usize, 7usize);
            // Statics written first (leftCols(nb) = product).
            mfcc.slice_mut(ndarray::s![.., 0..nb]).assign(&product);
            let mfcc_tmp = regression_deltas(&product, d);
            // SDC scratch (T + k*P) x (k*nb); block kk at vertical offset kk*P.
            let mut sdc = Array2::<f64>::zeros((t + k * p, k * nb));
            for kk in 0..k {
                for r in 0..t {
                    for c in 0..nb {
                        sdc[[kk * p + r, kk * nb + c]] = mfcc_tmp[[r, c]];
                    }
                }
            }
            // Extract at row (k*P-1)/2 = 10, width k*nb, into rightCols(k*nb). With
            // ignoreFirst this starts at col nb-1 and clobbers the last static.
            let start = (k * p - 1) / 2;
            let col0 = width - k * nb;
            for r in 0..t {
                for c in 0..k * nb {
                    mfcc[[r, col0 + c]] = sdc[[start + r, c]];
                }
            }
        } else if self.ignore_first_dct {
            // Branch B - ignoreFirst && deltas >= 0. Build a T x (width+1) temp
            // then drop the FIRST column.
            let mut tmp = Array2::<f64>::zeros((t, width + 1));
            if self.compute_deltas_nb > 0 {
                tmp.slice_mut(ndarray::s![.., 0..nb]).assign(&product);
                let delta = regression_deltas(&product, self.compute_deltas_nb);
                tmp.slice_mut(ndarray::s![.., nb..2 * nb]).assign(&delta);
                if self.compute_delta_deltas_nb > 0 {
                    let dd = regression_deltas(&delta, self.compute_delta_deltas_nb);
                    tmp.slice_mut(ndarray::s![.., 2 * nb..3 * nb]).assign(&dd);
                }
            } else {
                // deltas == 0: temp is just the product (T x nb == T x (width+1)).
                tmp.assign(&product);
            }
            // rightCols(width): drop column 0.
            mfcc.assign(&tmp.slice(ndarray::s![.., 1..width + 1]));
        } else {
            // Branch C - !ignoreFirst, deltas >= 0. Statics kept, [c|dc|(ddc)].
            if self.compute_deltas_nb > 0 {
                mfcc.slice_mut(ndarray::s![.., 0..nb]).assign(&product);
                let delta = regression_deltas(&product, self.compute_deltas_nb);
                mfcc.slice_mut(ndarray::s![.., nb..2 * nb]).assign(&delta);
                if self.compute_delta_deltas_nb > 0 {
                    let dd = regression_deltas(&delta, self.compute_delta_deltas_nb);
                    mfcc.slice_mut(ndarray::s![.., 2 * nb..3 * nb]).assign(&dd);
                }
            } else {
                mfcc.assign(&product);
            }
        }
        mfcc
    }

    /// Apply the filterbank. Allocates a zero-initialized `T x nb_filters()`
    /// output (mirroring the caller's `setZero` + the apply semantics). Four
    /// branches on (`is_log`, `is_mel`); the log+mel branch also folds in the
    /// deltas-no-DCT `[delta|delta|dd]` overwrite when active.
    // `ind_coeff` walks absolute periodogram columns alongside the coeff run; the
    // explicit counter is the legacy iteration, not a zip candidate.
    #[allow(clippy::explicit_counter_loop)]
    pub fn apply_filter_bank(&self, p: &Array2<f64>) -> Array2<f64> {
        let t = p.nrows();
        let mut out = Array2::<f64>::zeros((t, self.nb_filters()));

        if self.is_log {
            if self.is_mel {
                // Per-(frame, filter) sequential dot then ln(dot + 1e-24).
                for jj in 0..t {
                    for ii in 0..self.nb_filters {
                        let mut ind_coeff = self.index_begin[ii];
                        let mut acc = 0.0;
                        for &c in &self.coeffs[ii] {
                            acc += c * p[[jj, ind_coeff]];
                            ind_coeff += 1;
                        }
                        out[[jj, ii]] = (acc + 1e-24).ln();
                    }
                }
                if !self.is_dct_activated && self.compute_deltas_nb > 0 {
                    self.apply_deltas_overwrite(&mut out);
                }
            } else {
                // Raw-band pass-through: ln(block + 1e-24).
                for jj in 0..t {
                    for ii in 0..self.nb_filters {
                        out[[jj, ii]] = (p[[jj, self.beg_freq + ii]] + 1e-24).ln();
                    }
                }
            }
        } else if self.is_mel {
            // Plain triangle dots, no log.
            for jj in 0..t {
                for ii in 0..self.nb_filters {
                    let mut ind_coeff = self.index_begin[ii];
                    let mut acc = 0.0;
                    for &c in &self.coeffs[ii] {
                        acc += c * p[[jj, ind_coeff]];
                        ind_coeff += 1;
                    }
                    out[[jj, ii]] = acc;
                }
            }
        } else {
            // Raw-band copy.
            for jj in 0..t {
                for ii in 0..self.nb_filters {
                    out[[jj, ii]] = p[[jj, self.beg_freq + ii]];
                }
            }
        }

        out
    }

    /// The log+mel deltas-no-DCT tail (`MelFilterBank.cpp:150-194`). `out` enters
    /// as `T x (2 or 3)*F` with the static log-mel already in the FIRST `F`
    /// columns and the rest zero. Fills the delta / delta-delta blocks, then
    /// applies the `[delta|delta|dd]` overwrite: the static block is clobbered by
    /// the delta block.
    fn apply_deltas_overwrite(&self, out: &mut Array2<f64>) {
        let f = self.nb_filters;
        let t = out.nrows();

        // Static log-mel block (leftCols(F)) is the delta source.
        let static_block = out.slice(ndarray::s![.., 0..f]).to_owned();
        let delta = regression_deltas(&static_block, self.compute_deltas_nb);
        // Write delta into cols [F, 2F).
        out.slice_mut(ndarray::s![.., f..2 * f]).assign(&delta);

        if self.compute_delta_deltas_nb > 0 {
            // delta-delta over the just-written delta block, into cols [2F, 3F).
            let dd = regression_deltas(&delta, self.compute_delta_deltas_nb);
            out.slice_mut(ndarray::s![.., 2 * f..3 * f]).assign(&dd);
        }

        // Overwrite quirk: static (leftCols F) := delta block (:192-193).
        for jj in 0..t {
            for ii in 0..f {
                out[[jj, ii]] = out[[jj, f + ii]];
            }
        }
    }
}

/// Shared regression-deltas kernel (`MelFilterBank.cpp:153-170`), the block-op
/// sequence: for `j = 1..=n`, form a shifted-difference matrix (edge rows
/// corrected against row 0 / row T-1), accumulate `j * deltas`, and finally
/// divide by `2 * sum j^2`. When `j > T` the shifted copy is SKIPPED but `j*j`
/// still contributes to the denominator (the clamp saturates, the term stays).
/// Equivalent to the clamped closed form
///   `D[t] = sum_j j*(B[min(t+j,T-1)] - B[max(t-j,0)]) / (2 sum j^2)`.
///
/// `pub` (crate-internal reuse: Task 7's `apply_dct` will call it on the DCT
/// coefficient block) so the external JSON cross-check test can also reach it.
pub fn regression_deltas(base: &Array2<f64>, n: i32) -> Array2<f64> {
    let t = base.nrows();
    let f = base.ncols();
    let mut acc = Array2::<f64>::zeros((t, f));
    let mut adim = 0.0;
    for j in 1..=n {
        let mut deltas = Array2::<f64>::zeros((t, f));
        let mut length = j as usize;
        if length > t {
            length = t;
        } else {
            // topRows(T-j) = base.bottomRows(T-j)  == base[j..T]
            // bottomRows(T-j) -= base.topRows(T-j) == base[0..T-j]
            for r in 0..t - length {
                for c in 0..f {
                    deltas[[r, c]] = base[[r + length, c]];
                }
            }
            for r in length..t {
                for c in 0..f {
                    deltas[[r, c]] -= base[[r - length, c]];
                }
            }
        }
        for kk in 0..length {
            for c in 0..f {
                deltas[[kk, c]] -= base[[0, c]];
                deltas[[t - 1 - kk, c]] += base[[t - 1, c]];
            }
        }
        adim += (j * j) as f64;
        for r in 0..t {
            for c in 0..f {
                acc[[r, c]] += j as f64 * deltas[[r, c]];
            }
        }
    }
    adim *= 2.0;
    for r in 0..t {
        for c in 0..f {
            acc[[r, c]] /= adim;
        }
    }
    acc
}
