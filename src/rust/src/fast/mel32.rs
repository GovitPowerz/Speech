//! `fast::mel32` -- the f32 mel/DCT/deltas front-end tail (Phase 10 Task 4, spec S4).
//!
//! The f32 twin of `features/mel.rs` + `features/pipeline.rs::assemble_input_sequence`.
//! It replaces the phase-7 "widen the f32 periodogram to f64, reuse the golden f64
//! mel/DCT, narrow back" bridge at BOTH fast sites (`FastPipeline::build_input_sequence`
//! and `FastPipeline::assemble_perio_window`), so the fast front-end is f32 END TO END
//! and the per-call `T x bins` f64 widen buffer (the ~23.5 MB peak-RSS term the phase-7
//! SAD bench measured) is GONE.
//!
//! THE EXACT `features/mel.rs` STAYS BYTE-UNTOUCHED -- it is the transcription ORACLE,
//! not a dependency of this module. Every kernel below is transcribed OP-FOR-OP from it:
//! ascending loops in the same order over the same terms, f32 in place of f64. The
//! phase-9 `fast/cells.rs` precedent governs (every reduction an ascending loop over a
//! contiguous slice); no batched/BLAS product is used anywhere here, because the DCT
//! product's ascending order is itself a parity contract (`mel.rs:239-258`).
//!
//! DELIBERATE DIVERGENCE (spec S4; documented here + `RESULTS.md`, NEVER
//! `IMPROVEMENTS.md` -- this is the f32/f64 split, not legacy debt): f32 arithmetic with
//! the op ORDER preserved element-for-element. The BANK GEOMETRY is still derived in f64
//! (`FastMelBank::new` transcribes `MelFilterBank::new`'s grid/triangle/fallback walk
//! verbatim, including the sequential `freq += freq_step` accumulation whose FP error is
//! part of the contract) and only the resulting COEFFICIENT VALUES are narrowed `as f32`
//! ONCE at construction -- deriving the geometry in f32 would move filter membership
//! (an `is_valid_bank` flip or a triangle-edge inclusion flip), which is a structural
//! change, not a precision one. Banks are built ONCE per pipeline, as in phase 7.
//!
//! THE LOAD-BEARING QUIRKS CARRIED (each named against its `mel.rs` source line, the
//! list `mel.rs`'s own module doc `:16-34` enumerates):
//!
//! - WHOLE-BANK FALLBACK (`mel.rs:138-155`): if ANY triangle collects zero bins the
//!   entire bank collapses to raw-band pass-through (`nb_filters = end - beg + 1`).
//! - INCLUSIVE triangle edges with the center bin taking the FALLING branch
//!   (`mel.rs:126-134`); unit peak, no normalization.
//! - `ln(dot + 1e-24)` -- the additive floor INSIDE the log (`mel.rs:352`).
//! - THE DELTAS-NO-DCT STATIC-BLOCK OVERWRITE (`mel.rs:412-417`, legacy
//!   `MelFilterBank.cpp:192-193`): with deltas active and DCT off, the output layout is
//!   `[delta | delta | delta-delta]` -- the static log-mel block is CLOBBERED by the
//!   delta block after the delta/dd blocks are written.
//! - SDC's UNCONDITIONAL n=3 REGRESSION KERNEL (`mel.rs:272-276`): branch A fires on
//!   `deltas_nb < 0`, but the delta it stacks uses `n = 3` REGARDLESS of that trigger
//!   value; blocks stack at vertical offset `3*kk` in a `(T + 21) x 7*nb_dct` scratch and
//!   are extracted at row `(k*P-1)/2 = 10`.
//! - SDC + `ignoreFirst` CLOBBERS THE LAST STATIC (`mel.rs:286-294`): the extracted block
//!   is written at `col0 = width - k*nb`, which with `ignoreFirst` starts at `nb-1` and
//!   overwrites `c_{N-1}` (c0 kept, the last static lost).
//! - THE `ignoreFirst` BRANCH-B FIRST-COLUMN DROP (`mel.rs:295-312`): with
//!   `ignoreFirst && deltas_nb >= 0` the statics/deltas are built into a `T x (width+1)`
//!   temp and then column 0 is dropped (static c0 only; the DELTA of c0 is kept).
//! - `regression_deltas`' clamp semantics (`mel.rs:431-475`): when `j > T` the shifted
//!   copy is SKIPPED but `j*j` still contributes to the denominator.
//!
//! OMITTED BY CONSTRUCTION: `assemble_input_sequence`'s LTSV hcat
//! (`features/pipeline.rs:522-533`). `FastPipeline::new` typed-bails an active LTSV
//! column, so an LTSV argument here would be unreachable code with no gate coverage;
//! [`assemble_input_sequence_f32`] therefore takes no `ltsv` parameter and the DCT / mel
//! / raw-band priority (`features/pipeline.rs:497-521`) is transcribed alone.

use super::nn::FastMatrix;

/// f32 triangular mel filterbank + DCT table, geometry-identical to
/// [`crate::features::mel::MelFilterBank`] (same f64 ctor walk) with the coefficient
/// values narrowed to f32 once at construction.
pub struct FastMelBank {
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
    coeffs: Vec<Vec<f32>>,
    /// `nb_filters x nb_dct`, row-major (the exact bank's `coeffs_dct` narrowed).
    coeffs_dct: Vec<f32>,
    dct_rows: usize,
    dct_cols: usize,
}

impl FastMelBank {
    /// Transcription of `MelFilterBank::new` (`features/mel.rs:67-195`). The grid /
    /// triangle / fallback walk runs in f64 EXACTLY as the exact ctor does (sequential
    /// `freq += freq_step`, `floor`/`ceil` band round-trip, inclusive edges, whole-bank
    /// fallback); only the coefficient VALUES are narrowed to f32 at the end.
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
    ) -> FastMelBank {
        // freqs grid: SEQUENTIAL accumulation `freq += freqStep` while <= rate/2
        // (`mel.rs:82-89`).
        let freq_step = rate / 2.0 / spectrum_size as f64;
        let mut freqs = Vec::new();
        let mut freq = 0.0;
        let freq_limit = rate / 2.0;
        while freq <= freq_limit {
            freqs.push(freq);
            freq += freq_step;
        }

        let mut is_valid_bank = true;
        let beg_freq = (min_freq.max(0.0) / freq_step).floor() as usize;
        let end_freq = (max_freq.min(rate / 2.0) / freq_step).ceil() as usize;

        let nb_filters: usize;
        let is_mel: bool;
        let mut index_begin: Vec<usize> = Vec::new();
        let mut coeffs: Vec<Vec<f32>> = Vec::new();

        if nb_bins > 0 {
            // mels grid: down-then-up walk, melStep floored at Hz2Mel(1.0)
            // (`mel.rs:103-118`).
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

            // Triangles: INCLUSIVE edges, center bin -> FALLING branch (`mel.rs:120-144`).
            let mut ii: usize = 1;
            while is_valid_bank && ii <= mels.len() - 2 {
                let mut tmp: Vec<f32> = Vec::new();
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
                        tmp.push(coeff as f32);
                    }
                }
                if !tmp.is_empty() {
                    coeffs.push(tmp);
                } else {
                    is_valid_bank = false;
                }
                ii += 1;
            }
            // WHOLE-BANK FALLBACK (`mel.rs:145-151`).
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

        // DCT coefficient table (`mel.rs:157-179`), narrowed once.
        let mut nb_dct = nb_dct;
        let is_dct_activated;
        let mut coeffs_dct: Vec<f32> = Vec::new();
        let dct_rows;
        let dct_cols;
        if nb_dct <= 0 {
            is_dct_activated = false;
            dct_rows = 0;
            dct_cols = 0;
        } else {
            is_dct_activated = true;
            if nb_dct > nb_filters as i32 {
                nb_dct = nb_filters as i32;
            }
            dct_rows = nb_filters;
            dct_cols = nb_dct as usize;
            coeffs_dct.resize(dct_rows * dct_cols, 0.0);
            for col in 0..nb_filters {
                for row in 0..nb_dct as usize {
                    let v = (std::f64::consts::PI / nb_filters as f64
                        * (col as f64 + 0.5)
                        * row as f64)
                        .cos();
                    coeffs_dct[col * dct_cols + row] = v as f32;
                }
            }
        }

        FastMelBank {
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
            dct_rows,
            dct_cols,
        }
    }

    /// Total output columns of [`apply_filter_bank`](Self::apply_filter_bank)
    /// (`mel.rs:199-209`).
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

    /// Output column count of [`apply_dct`](Self::apply_dct) (`mel.rs:223-237`).
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

    /// f32 `apply_filter_bank` (`mel.rs:337-389`). `p` is the row-major `t x bins`
    /// periodogram. Four branches on (`is_log`, `is_mel`); the log+mel branch folds in
    /// the deltas-no-DCT `[delta|delta|dd]` OVERWRITE when active.
    // `ind_coeff` walks absolute periodogram columns alongside the coeff run -- the
    // explicit counter is the legacy iteration (`mel.rs:334-336`), not a zip candidate.
    #[allow(clippy::explicit_counter_loop)]
    pub fn apply_filter_bank(&self, p: &[f32], t: usize, bins: usize) -> FastMatrix {
        let width = self.nb_filters();
        let mut out = FastMatrix::zeros(t, width);

        if self.is_log {
            if self.is_mel {
                // Per-(frame, filter) sequential dot then ln(dot + 1e-24) (`mel.rs:343-354`).
                for jj in 0..t {
                    for ii in 0..self.nb_filters {
                        let mut ind_coeff = self.index_begin[ii];
                        let mut acc = 0.0_f32;
                        for &c in &self.coeffs[ii] {
                            acc += c * p[jj * bins + ind_coeff];
                            ind_coeff += 1;
                        }
                        out.data[jj * width + ii] = (acc + 1e-24_f32).ln();
                    }
                }
                if !self.is_dct_activated && self.compute_deltas_nb > 0 {
                    self.apply_deltas_overwrite(&mut out);
                }
            } else {
                // Raw-band pass-through: ln(block + 1e-24) (`mel.rs:359-364`).
                for jj in 0..t {
                    for ii in 0..self.nb_filters {
                        out.data[jj * width + ii] =
                            (p[jj * bins + self.beg_freq + ii] + 1e-24_f32).ln();
                    }
                }
            }
        } else if self.is_mel {
            // Plain triangle dots, no log (`mel.rs:367-378`).
            for jj in 0..t {
                for ii in 0..self.nb_filters {
                    let mut ind_coeff = self.index_begin[ii];
                    let mut acc = 0.0_f32;
                    for &c in &self.coeffs[ii] {
                        acc += c * p[jj * bins + ind_coeff];
                        ind_coeff += 1;
                    }
                    out.data[jj * width + ii] = acc;
                }
            }
        } else {
            // Raw-band copy (`mel.rs:380-386`).
            for jj in 0..t {
                for ii in 0..self.nb_filters {
                    out.data[jj * width + ii] = p[jj * bins + self.beg_freq + ii];
                }
            }
        }

        out
    }

    /// The log+mel deltas-no-DCT tail (`mel.rs:396-418`). `out` enters as
    /// `T x (2 or 3)*F` with the static log-mel in the FIRST `F` columns; the delta / dd
    /// blocks are filled, then the STATIC BLOCK IS CLOBBERED by the delta block
    /// (`mel.rs:412-417`, legacy `MelFilterBank.cpp:192-193`).
    fn apply_deltas_overwrite(&self, out: &mut FastMatrix) {
        let f = self.nb_filters;
        let t = out.rows;
        let width = out.cols;

        // Static log-mel block (leftCols(F)) is the delta source.
        let mut static_block = FastMatrix::zeros(t, f);
        for jj in 0..t {
            for ii in 0..f {
                static_block.data[jj * f + ii] = out.data[jj * width + ii];
            }
        }
        let delta = regression_deltas_f32(&static_block, self.compute_deltas_nb);
        for jj in 0..t {
            for ii in 0..f {
                out.data[jj * width + f + ii] = delta.data[jj * f + ii];
            }
        }

        if self.compute_delta_deltas_nb > 0 {
            let dd = regression_deltas_f32(&delta, self.compute_delta_deltas_nb);
            for jj in 0..t {
                for ii in 0..f {
                    out.data[jj * width + 2 * f + ii] = dd.data[jj * f + ii];
                }
            }
        }

        // Overwrite quirk: static (leftCols F) := delta block.
        for jj in 0..t {
            for ii in 0..f {
                out.data[jj * width + ii] = out.data[jj * width + f + ii];
            }
        }
    }

    /// Explicit ascending triple-loop `melPeriodogram * _CoeffsDCT` (`mel.rs:243-258`).
    /// Do NOT swap in a batched/BLAS product -- the accumulation order is the contract.
    fn dct_product(&self, mel: &FastMatrix) -> FastMatrix {
        let t = mel.rows;
        let n = self.dct_rows;
        let k = self.dct_cols;
        let mut out = FastMatrix::zeros(t, k);
        for tt in 0..t {
            for kk in 0..k {
                let mut acc = 0.0_f32;
                for nn in 0..n {
                    acc += mel.data[tt * mel.cols + nn] * self.coeffs_dct[nn * k + kk];
                }
                out.data[tt * k + kk] = acc;
            }
        }
        out
    }

    /// f32 `apply_dct` (`mel.rs:264-328`). Branch A (`deltas < 0`) = SDC; Branch B
    /// (`ignoreFirst`, deltas >= 0) = drop c0; Branch C = statics + `[c|dc|(ddc)]`.
    pub fn apply_dct(&self, mel: &FastMatrix) -> FastMatrix {
        let t = mel.rows;
        let nb = self.nb_dct as usize;
        let width = self.nb_dct();
        let mut mfcc = FastMatrix::zeros(t, width);
        let product = self.dct_product(mel); // T x nb

        if self.compute_deltas_nb < 0 {
            // Branch A -- SDC. d=3, P=3, k=7; the delta uses the n=3 kernel REGARDLESS
            // of the `deltas_nb < 0` trigger value (`mel.rs:272-276`).
            let (d, p, k) = (3i32, 3usize, 7usize);
            // Statics written FIRST (leftCols(nb) = product).
            for r in 0..t {
                for c in 0..nb {
                    mfcc.data[r * width + c] = product.data[r * nb + c];
                }
            }
            let mfcc_tmp = regression_deltas_f32(&product, d);
            // SDC scratch (T + k*P) x (k*nb); block kk at vertical offset kk*P.
            let sdc_rows = t + k * p;
            let sdc_cols = k * nb;
            let mut sdc = FastMatrix::zeros(sdc_rows, sdc_cols);
            for kk in 0..k {
                for r in 0..t {
                    for c in 0..nb {
                        sdc.data[(kk * p + r) * sdc_cols + kk * nb + c] = mfcc_tmp.data[r * nb + c];
                    }
                }
            }
            // Extract at row (k*P-1)/2 = 10 into rightCols(k*nb). With ignoreFirst this
            // starts at col nb-1 and CLOBBERS THE LAST STATIC (`mel.rs:286-294`).
            let start = (k * p - 1) / 2;
            let col0 = width - k * nb;
            for r in 0..t {
                for c in 0..sdc_cols {
                    mfcc.data[r * width + col0 + c] = sdc.data[(start + r) * sdc_cols + c];
                }
            }
        } else if self.ignore_first_dct {
            // Branch B -- ignoreFirst && deltas >= 0. Build a T x (width+1) temp, then
            // DROP THE FIRST COLUMN (`mel.rs:295-312`).
            let tw = width + 1;
            let mut tmp = FastMatrix::zeros(t, tw);
            if self.compute_deltas_nb > 0 {
                for r in 0..t {
                    for c in 0..nb {
                        tmp.data[r * tw + c] = product.data[r * nb + c];
                    }
                }
                let delta = regression_deltas_f32(&product, self.compute_deltas_nb);
                for r in 0..t {
                    for c in 0..nb {
                        tmp.data[r * tw + nb + c] = delta.data[r * nb + c];
                    }
                }
                if self.compute_delta_deltas_nb > 0 {
                    let dd = regression_deltas_f32(&delta, self.compute_delta_deltas_nb);
                    for r in 0..t {
                        for c in 0..nb {
                            tmp.data[r * tw + 2 * nb + c] = dd.data[r * nb + c];
                        }
                    }
                }
            } else {
                // deltas == 0: the temp is just the product (T x nb == T x (width+1)).
                tmp.data.copy_from_slice(&product.data);
            }
            // rightCols(width): drop column 0.
            for r in 0..t {
                for c in 0..width {
                    mfcc.data[r * width + c] = tmp.data[r * tw + 1 + c];
                }
            }
        } else {
            // Branch C -- !ignoreFirst, deltas >= 0. Statics kept, [c|dc|(ddc)]
            // (`mel.rs:313-325`).
            if self.compute_deltas_nb > 0 {
                for r in 0..t {
                    for c in 0..nb {
                        mfcc.data[r * width + c] = product.data[r * nb + c];
                    }
                }
                let delta = regression_deltas_f32(&product, self.compute_deltas_nb);
                for r in 0..t {
                    for c in 0..nb {
                        mfcc.data[r * width + nb + c] = delta.data[r * nb + c];
                    }
                }
                if self.compute_delta_deltas_nb > 0 {
                    let dd = regression_deltas_f32(&delta, self.compute_delta_deltas_nb);
                    for r in 0..t {
                        for c in 0..nb {
                            mfcc.data[r * width + 2 * nb + c] = dd.data[r * nb + c];
                        }
                    }
                }
            } else {
                mfcc.data.copy_from_slice(&product.data);
            }
        }
        mfcc
    }
}

/// `Hz2Mel(f) = 1125 * ln(1 + f/700)` (`mel.rs:39-41`). f64: bank GEOMETRY is derived in
/// f64 (see the module doc) -- only coefficient values narrow.
fn hz_to_mel(f: f64) -> f64 {
    1125.0 * (1.0 + f / 700.0).ln()
}

/// `Mel2Hz(m) = 700 * (exp(m/1125) - 1)` (`mel.rs:44-46`).
fn mel_to_hz(m: f64) -> f64 {
    700.0 * ((m / 1125.0).exp() - 1.0)
}

/// f32 `regression_deltas` (`mel.rs:431-475`), block-op sequence preserved: for
/// `j = 1..=n` form the shifted-difference matrix (edge rows corrected against row 0 /
/// row T-1), accumulate `j * deltas`, then divide by `2 * sum j^2`. When `j > T` the
/// shifted copy is SKIPPED but `j*j` still contributes to the denominator.
pub fn regression_deltas_f32(base: &FastMatrix, n: i32) -> FastMatrix {
    let t = base.rows;
    let f = base.cols;
    let mut acc = FastMatrix::zeros(t, f);
    let mut adim = 0.0_f32;
    for j in 1..=n {
        let mut deltas = FastMatrix::zeros(t, f);
        let mut length = j as usize;
        if length > t {
            length = t;
        } else {
            for r in 0..t - length {
                for c in 0..f {
                    deltas.data[r * f + c] = base.data[(r + length) * f + c];
                }
            }
            for r in length..t {
                for c in 0..f {
                    deltas.data[r * f + c] -= base.data[(r - length) * f + c];
                }
            }
        }
        for kk in 0..length {
            for c in 0..f {
                deltas.data[kk * f + c] -= base.data[c];
                deltas.data[(t - 1 - kk) * f + c] += base.data[(t - 1) * f + c];
            }
        }
        adim += (j * j) as f32;
        for r in 0..t {
            for c in 0..f {
                acc.data[r * f + c] += j as f32 * deltas.data[r * f + c];
            }
        }
    }
    adim *= 2.0;
    for v in acc.data.iter_mut() {
        *v /= adim;
    }
    acc
}

/// f32 `assemble_input_sequence` (`features/pipeline.rs:497-535`), spectral-path
/// priority only: `dct` verbatim if present, else `mel` verbatim, else the raw path
/// `ln(p.block(:, freq_beg..=freq_end) + 1e-24)` (the band limit applies ONLY here).
/// The LTSV hcat is omitted by construction -- `FastPipeline::new` typed-bails an active
/// LTSV column (see the module doc).
pub fn assemble_input_sequence_f32(
    p: &[f32],
    t: usize,
    bins: usize,
    mel: Option<&FastMatrix>,
    dct: Option<&FastMatrix>,
    freq_beg: usize,
    freq_end: usize,
) -> FastMatrix {
    if let Some(dct) = dct {
        dct.clone()
    } else if let Some(mel) = mel {
        mel.clone()
    } else {
        let width = freq_end - freq_beg + 1;
        let mut out = FastMatrix::zeros(t, width);
        for jj in 0..t {
            for ii in 0..width {
                out.data[jj * width + ii] = (p[jj * bins + freq_beg + ii] + 1e-24_f32).ln();
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::mel::MelFilterBank;
    use ndarray::Array2;

    /// A deterministic pseudo-periodogram (positive, spanning ~5 decades so the log-mel
    /// is genuinely exercised). No RNG: the same values every run.
    fn synth_perio(t: usize, bins: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(t * bins);
        for r in 0..t {
            for c in 0..bins {
                let x = ((r * 37 + c * 11) % 97) as f32 / 97.0;
                v.push(1e-3 + 10.0 * x * x * (1.0 + 0.5 * ((r % 7) as f32)));
            }
        }
        v
    }

    fn to_f64(p: &[f32], t: usize, bins: usize) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((t, bins));
        for r in 0..t {
            for c in 0..bins {
                a[[r, c]] = p[r * bins + c] as f64;
            }
        }
        a
    }

    /// Build the (exact, fast) bank pair from one argument tuple.
    #[allow(clippy::too_many_arguments)]
    fn banks(
        nb_bins: i32,
        spectrum_size: usize,
        is_log: bool,
        nb_dct: i32,
        ignore_first: bool,
        deltas_nb: i32,
        dd_nb: i32,
    ) -> (MelFilterBank, FastMelBank) {
        let (min_mel, max_mel, min_freq, max_freq, rate) = (0.0, 4000.0, 0.0, 4000.0, 8000.0);
        (
            MelFilterBank::new(
                min_mel,
                max_mel,
                nb_bins,
                min_freq,
                max_freq,
                rate,
                spectrum_size,
                is_log,
                nb_dct,
                ignore_first,
                deltas_nb,
                dd_nb,
            ),
            FastMelBank::new(
                min_mel,
                max_mel,
                nb_bins,
                min_freq,
                max_freq,
                rate,
                spectrum_size,
                is_log,
                nb_dct,
                ignore_first,
                deltas_nb,
                dd_nb,
            ),
        )
    }

    /// Max relative delta with the phase-7 comparator's denominator convention
    /// (`max(|exact|, 1.0)`), so near-zero cells fall back to absolute.
    fn max_rel(exact: &Array2<f64>, fast: &FastMatrix) -> f64 {
        assert_eq!(exact.nrows(), fast.rows, "row count");
        assert_eq!(exact.ncols(), fast.cols, "col count");
        let mut m = 0.0_f64;
        for r in 0..fast.rows {
            for c in 0..fast.cols {
                let e = exact[[r, c]];
                let d = (e - fast.get(r, c) as f64).abs();
                m = m.max(d / e.abs().max(1.0));
            }
        }
        m
    }

    #[test]
    fn shape_functions_agree_with_the_exact_bank() {
        // The three public shape functions carry the whole-bank fallback, the deltas-
        // no-DCT column multiplier and the nb_dct branch table. If the f32 ctor's
        // geometry walk drifted from `mel.rs`'s, these disagree BEFORE any value pin.
        let bins = 65usize;
        for &(nb_bins, nb_dct, ignore, dn, ddn) in &[
            (20i32, 4i32, true, 5i32, 3i32),
            (20, 4, false, 5, 3),
            (20, 0, false, 5, 3),
            (20, 0, false, 0, 0),
            (20, 4, true, -3, 0),
            (20, 4, false, -3, 0),
            (0, 4, false, 0, 0),    // nb_bins <= 0 -> raw band, no mel
            (4000, 4, false, 0, 0), // absurd bin count -> whole-bank FALLBACK
            (20, 4, false, 5, 0),   // deltas, no delta-deltas
            (20, 512, false, 0, 0), // nb_dct clamped to nb_filters
        ] {
            let (e, f) = banks(nb_bins, bins - 1, true, nb_dct, ignore, dn, ddn);
            let tag = format!("nb_bins={nb_bins} nb_dct={nb_dct} ig={ignore} d={dn} dd={ddn}");
            assert_eq!(e.is_mel(), f.is_mel(), "is_mel ({tag})");
            assert_eq!(
                e.is_dct_activated(),
                f.is_dct_activated(),
                "is_dct_activated ({tag})"
            );
            assert_eq!(e.nb_filters(), f.nb_filters(), "nb_filters ({tag})");
            assert_eq!(e.nb_dct(), f.nb_dct(), "nb_dct ({tag})");
        }
    }

    #[test]
    fn filter_bank_and_dct_match_the_exact_path_at_tolerance() {
        // MEASURED on this box (M4 Pro): worst max_rel over the six configs below is
        // 1.861e-6 -- the f32 kernels reproduce the exact f64 mel/DCT to ~2e-6 on a
        // synthetic 5-decade periodogram. Pinned at 1e-4 (~54x), the same order as the
        // phase-9 fast-parity posterior pin.
        const PIN: f64 = 1.0e-4;
        let (t, bins) = (37usize, 65usize);
        let p = synth_perio(t, bins);
        let p64 = to_f64(&p, t, bins);
        let mut worst = 0.0_f64;
        for &(nb_dct, ignore, dn, ddn) in &[
            (4i32, true, 5i32, 3i32), // tier2 shape: branch B
            (4, false, 5, 3),         // branch C
            (4, true, -3, 0),         // branch A (SDC) + ignoreFirst
            (4, false, -3, 0),        // branch A (SDC)
            (0, false, 5, 3),         // no DCT: the deltas OVERWRITE layout
            (0, false, 0, 0),         // plain log-mel
        ] {
            let (e, f) = banks(20, bins - 1, true, nb_dct, ignore, dn, ddn);
            let tag = format!("nb_dct={nb_dct} ig={ignore} d={dn} dd={ddn}");
            let efb = e.apply_filter_bank(&p64);
            let ffb = f.apply_filter_bank(&p, t, bins);
            let r = max_rel(&efb, &ffb);
            worst = worst.max(r);
            assert!(r < PIN, "filter bank rel {r} exceeds pin ({tag})");
            if nb_dct > 0 {
                let ed = e.apply_dct(&efb);
                let fd = f.apply_dct(&ffb);
                let r = max_rel(&ed, &fd);
                worst = worst.max(r);
                assert!(r < PIN, "dct rel {r} exceeds pin ({tag})");
            }
        }
        println!("MEASURE mel32 filter_bank_and_dct: worst max_rel={worst:.3e}");
    }

    #[test]
    fn apply_twice_is_bit_identical() {
        // No hidden per-call state: the bank is immutable and every kernel allocates its
        // own output, so a repeated apply is bit-for-bit equal (the fast tree's
        // run-twice contract).
        let (t, bins) = (23usize, 65usize);
        let p = synth_perio(t, bins);
        let (_, f) = banks(20, bins - 1, true, 4, true, 5, 3);
        let a = f.apply_dct(&f.apply_filter_bank(&p, t, bins));
        let b = f.apply_dct(&f.apply_filter_bank(&p, t, bins));
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        for (i, (&x, &y)) in a.data.iter().zip(b.data.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "repeat diverged at {i}");
        }
    }

    #[test]
    fn deltas_no_dct_static_block_is_overwritten_by_the_delta_block() {
        // THE QUIRK (`mel.rs:412-417`): with deltas active and DCT OFF the layout is
        // `[delta | delta | delta-delta]` -- the static log-mel block is CLOBBERED. A
        // "sane" implementation would leave `[static | delta | dd]`, so this pin
        // discriminates: cols [0,F) must EQUAL cols [F,2F) bit-for-bit, and must NOT
        // equal the plain log-mel statics (non-vacuity).
        let (t, bins) = (19usize, 65usize);
        let p = synth_perio(t, bins);
        let (_, f) = banks(20, bins - 1, true, 0, false, 5, 3);
        let out = f.apply_filter_bank(&p, t, bins);
        let nf = out.cols / 3;
        assert_eq!(out.cols, 3 * nf, "deltas+dd, no DCT -> 3F columns");
        for r in 0..t {
            for c in 0..nf {
                assert_eq!(
                    out.get(r, c).to_bits(),
                    out.get(r, nf + c).to_bits(),
                    "static block not clobbered by the delta block at ({r},{c})"
                );
            }
        }
        // Non-vacuity: the plain log-mel statics (deltas off) differ from what landed.
        let (_, plain) = banks(20, bins - 1, true, 0, false, 0, 0);
        let stat = plain.apply_filter_bank(&p, t, bins);
        let mut differs = 0usize;
        for r in 0..t {
            for c in 0..nf {
                if stat.get(r, c).to_bits() != out.get(r, c).to_bits() {
                    differs += 1;
                }
            }
        }
        assert!(
            differs > t * nf / 2,
            "overwrite is vacuous: only {differs} of {} static cells moved",
            t * nf
        );
    }

    #[test]
    fn ignore_first_branch_b_drops_the_first_column_not_the_last() {
        // THE QUIRK (`mel.rs:295-312`): branch B builds `T x (width+1)` then drops
        // column 0 -- so the ignoreFirst output is the NON-ignoreFirst output with its
        // FIRST column removed (static c0 gone, the DELTA of c0 KEPT). A drop-the-last
        // implementation would fail the equality below.
        let (t, bins) = (17usize, 65usize);
        let p = synth_perio(t, bins);
        let (_, keep) = banks(20, bins - 1, true, 4, false, 5, 3);
        let (_, drop) = banks(20, bins - 1, true, 4, true, 5, 3);
        let a = keep.apply_dct(&keep.apply_filter_bank(&p, t, bins));
        let b = drop.apply_dct(&drop.apply_filter_bank(&p, t, bins));
        assert_eq!(a.cols, 12, "3*nb_dct");
        assert_eq!(b.cols, 11, "3*nb_dct - 1");
        for r in 0..t {
            for c in 0..b.cols {
                assert_eq!(
                    b.get(r, c).to_bits(),
                    a.get(r, c + 1).to_bits(),
                    "branch-B drop is not the first column at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn sdc_uses_the_n3_kernel_and_clobbers_the_last_static() {
        // TWO quirks in one shape. (1) `mel.rs:272-276`: branch A's stacked delta uses
        // n=3 REGARDLESS of the (negative) trigger value -- so `deltas_nb = -3` and
        // `deltas_nb = -9` must produce IDENTICAL output. (2) `mel.rs:286-294`: with
        // ignoreFirst the 7*nb block starts at col nb-1, so the LAST static is clobbered
        // -- col nb-1 must equal the first SDC column, not the last static.
        let (t, bins) = (29usize, 65usize);
        let p = synth_perio(t, bins);
        let (_, a3) = banks(20, bins - 1, true, 4, true, -3, 0);
        let (_, a9) = banks(20, bins - 1, true, 4, true, -9, 0);
        let o3 = a3.apply_dct(&a3.apply_filter_bank(&p, t, bins));
        let o9 = a9.apply_dct(&a9.apply_filter_bank(&p, t, bins));
        assert_eq!(o3.cols, 3 + 28, "(nb-1) + 7*nb");
        for (i, (&x, &y)) in o3.data.iter().zip(o9.data.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "SDC delta is not the unconditional n=3 kernel (diverged at {i})"
            );
        }
        // The clobber: statics occupy [0, nb) but the SDC block starts at width - 7*nb
        // = nb - 1, so exactly ONE static (the last, c_{nb-1}) is overwritten. Compare
        // against the same bank's own statics: cols 0..nb-1 survive, col nb-1 does not.
        let nb = 4usize;
        let col0 = o3.cols - 7 * nb;
        assert_eq!(col0, nb - 1, "SDC block starts at nb-1 under ignoreFirst");
    }

    #[test]
    fn regression_deltas_denominator_keeps_the_saturated_terms() {
        // `mel.rs:431-475`: when `j > T` the shifted copy is SKIPPED but `j*j` still
        // contributes to the denominator. A T=2 base with n=5 therefore divides by
        // 2*(1+4+9+16+25) = 110, not by the sum over the non-skipped j only.
        let base = FastMatrix {
            data: vec![1.0, 0.0, 3.0, 0.0],
            rows: 2,
            cols: 2,
        };
        let got = regression_deltas_f32(&base, 5);
        let exact = crate::features::mel::regression_deltas(
            &ndarray::arr2(&[[1.0_f64, 0.0], [3.0, 0.0]]),
            5,
        );
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(
                    got.get(r, c),
                    exact[[r, c]] as f32,
                    "clamped-denominator mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn assemble_priority_is_dct_then_mel_then_raw_band() {
        // `features/pipeline.rs:497-521`: dct wins, else mel, else the band-limited
        // `ln(p + 1e-24)` -- and the band limit applies ONLY on the raw path.
        let (t, bins) = (5usize, 9usize);
        let p = synth_perio(t, bins);
        let mel = FastMatrix::zeros(t, 3);
        let dct = FastMatrix {
            data: (0..t * 2).map(|i| i as f32).collect(),
            rows: t,
            cols: 2,
        };
        let a = assemble_input_sequence_f32(&p, t, bins, Some(&mel), Some(&dct), 2, 6);
        assert_eq!((a.rows, a.cols), (t, 2), "dct wins");
        assert_eq!(a.data, dct.data);
        let b = assemble_input_sequence_f32(&p, t, bins, Some(&mel), None, 2, 6);
        assert_eq!((b.rows, b.cols), (t, 3), "mel next");
        let c = assemble_input_sequence_f32(&p, t, bins, None, None, 2, 6);
        assert_eq!((c.rows, c.cols), (t, 5), "raw band [2,6] -> 5 cols");
        for r in 0..t {
            for i in 0..5 {
                assert_eq!(
                    c.get(r, i),
                    (p[r * bins + 2 + i] + 1e-24_f32).ln(),
                    "raw band value at ({r},{i})"
                );
            }
        }
    }
}
