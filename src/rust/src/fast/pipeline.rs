//! `fast::pipeline` -- f32 feature extraction on realfft (Phase 7 Task 3).
//!
//! An `f32`, `realfft`-backed transcription of the exact feature front-end's
//! framing + periodogram, feeding the (reused) golden f64 mel/DCT apply. The exact
//! sources this mirrors line-for-line (cite them when auditing drift, per the plan's
//! R2):
//!
//! - `audio.rs::get_sequence` (`:269-314`, `AudioStruct::getSequence`
//!   `AudioStruct.cpp:456-485`): the per-frame windowed extraction -- left-edge /
//!   right-edge / interior branch order, the DC-offset mean over the full `2w+1`
//!   buffer, the window multiply LAST. See the STALE-BUFFER note below.
//! - `audio.rs::compute_segment_periodogram_estimates` (`:340-449`,
//!   `AudioStruct.cpp:512-584`): the framing walk -- `window_size = 1<<order`,
//!   `signal_window_size = window_size/2` (the half-window), `buffer_size =
//!   window_size+1`, `periodogram_length = window_size/2+1`, and `frame_nb` as the
//!   inclusive-count ceil-divide of `span = end-begin+1` over `shift`.
//! - `features/fft.rs::compute_two_real_periodogram` (`:76-111`,
//!   `AudioStruct.cpp:487-510`): the periodogram VALUE -- see the NO-TWO-REAL-PACKING
//!   divergence + equivalence proof below.
//! - `features/mel.rs::MelFilterBank` (`apply_filter_bank` `:337`, `apply_dct`
//!   `:264`) + `features/pipeline.rs::assemble_input_sequence` (`:497`): the mel/DCT
//!   apply + the DCT/mel/raw-band + LTSV priority. REUSED verbatim in f64 (see the
//!   MEL-REUSE note).
//! - `features/pipeline.rs::SpectralParams::derive` / `FeatureConfig::from_legacy`:
//!   CONSUMED, not re-derived (the brief's mandate -- derivation is shared config
//!   arithmetic).
//!
//! DELIBERATE DIVERGENCES FROM THE EXACT PATH (spec S4/R3; documented here +
//! RESULTS.md, NEVER IMPROVEMENTS.md -- this is the f32/f64 split, not legacy debt):
//!
//! 1. NO TWO-REAL PACKING. The exact GFFT packs TWO windowed frames into one complex
//!    FFT (`sig1` -> real, `sig2` -> imag) and unpacks two periodogram rows -- a
//!    GFFT-era trick to halve the transform count. `realfft` transforms ONE real frame
//!    natively (`window_size` reals -> `window_size/2+1` complex), so the fast path does
//!    ONE real FFT per frame and there is no packing/unpacking. EQUIVALENCE: for a real
//!    signal `s`, `Y[m] = RealFFT(s)[m]` is exactly the two-real unpacking's
//!    `FFT(sig1)[m] = (X[m] + conj(X[N-m]))/2`; the exact per-bin expression
//!    `((Re(X[m])+Re(X[N-m]))^2 + (Im(X[m])-Im(X[N-m]))^2) / (4N)` equals
//!    `|FFT(sig1)[m]|^2 / N`, and the DC/Nyquist special cases (`data[0]^2/N`, the
//!    `ii=N` inclusive Nyquist term) are the same `|Y[m]|^2 / N`. So the fast
//!    periodogram is the SINGLE uniform formula `(Y[m].re^2 + Y[m].im^2) / N` for every
//!    `m in 0..=N/2` -- algebraically the exact value, differing only by f32 precision +
//!    realfft's (rustfft) blocked reduction order vs the exact GFFT's running-twiddle
//!    recurrence.
//!
//! 2. f32 THROUGHOUT THE FFT/PERIODOGRAM. Samples are narrowed f64 -> f32 once at the
//!    caller seam (see the SEAM note); windowing, the FFT, and `|Y|^2/N` all run in
//!    f32.
//!
//! 3. STALE-BUFFER QUIRK IS A PROVEN NO-OP. The exact framing reuses ONE zero-init
//!    buffer pair across frames; a LEFT-edge `get_sequence` does NOT zero the buffer,
//!    so the left pad keeps stale contents. That stale pad is PROVABLY ZERO for the
//!    periodogram: within a buffer's frame subsequence the center index is strictly
//!    increasing, so `beg_win = half_window - index` is strictly DECREASING, so each
//!    left-edge frame's pad `[0, beg_win)` is a subset of the still-never-written
//!    zero-init region (the first frame's pad was zero and later frames only extend the
//!    written region leftward). The RIGHT edge and interior overwrite fully. So a
//!    fresh-zeroed per-frame buffer reproduces the exact periodogram bit-for-bit (mod
//!    f32). This path zeroes per frame -- simpler and equivalent. (`begin=0`,
//!    `end=frames-1` also makes `index >= frames` unreachable, so the `get_sequence`
//!    no-op branch never fires.) CAUTION (Phase 7 Task 10 battery, item 5): the
//!    per-frame reset in `fill_frame` is LOAD-BEARING beyond run-twice bit-identity --
//!    the no-op proof above holds ONLY for a per-frame-FRESH buffer. A right-edge frame
//!    writes only `[0, nb_elem)`, leaving stale INTERIOR data in the FFT-read window
//!    `[nb_elem, window_size)` on a REUSED buffer, so dropping the reset diverges WITHIN
//!    a single call, breaking the TOLERANCE parity pins, not merely the run-twice
//!    bit-identity pin. Do not refactor the reset away.
//!
//! 4. MEL/DCT REUSE (f64), scoped divergence. The mel filterbank + DCT table are built
//!    ONCE at construction (`MelFilterBank::new`, the load-bearing "banks built once" of
//!    the brief -- exactly what Task 8 protects), and the apply reuses the golden f64
//!    `apply_filter_bank`/`apply_dct`/`assemble_input_sequence` on the f32-derived
//!    periodogram WIDENED to f64, then narrows the assembled sequence to f32. Rationale:
//!    the mel/DCT is a light post-process (T x 20 dots, T x 4 DCT) dominated by the
//!    per-frame 1024-pt FFT, so the fast win lives in the FFT; and the exact `mel.rs` is
//!    UNTOUCHABLE (its ctor fields are private -- an f32 re-transcription of the gnarly
//!    triangle/grid/fallback ctor would be pure duplication risk with no measurable
//!    speed benefit). A fully-f32 mel apply is a possible future tightening (like the
//!    full-f32 decode the brief itself defers), not this task. The consequence: the
//!    measured delta is TIGHTER than a fully-f32 path (only the periodogram carries f32
//!    error into the mel).
//!
//! SEAM: the audio is held ONCE in f32 on this path. `build_input_sequence` consumes
//! `samples: &[f32]` -- one channel of the EXISTING `read_audio` output narrowed once
//! at the caller boundary (no duplicated symphonia plumbing, no data/data_raw pair).
//! The `(2*RMS+max)/2` normalization happens inside `read_audio` (f64) BEFORE this seam,
//! and preemph/noise are the DRIVER's mutation (Task 4, applied to the f64 audio before
//! narrowing) exactly as the exact scoring path does -- so this f32 path inherits them.
//!
//! UNEXERCISED PATHS TYPED-BAIL at construction (`new` returns `Result`, a deviation
//! from the brief's bare `-> FastPipeline`, matching the house typed-bail pattern +
//! `FastBlstm::from_flat`): an active LTSV column (`LTSVwindow > 0`) and an active
//! periodogram temporal convolution (`spectrum_temporal_convolution_size > 0`). Both
//! are OFF in every committed gate config; both would need NEW fast-path wiring with no
//! golden coverage. The pitch second pass (`TDCwindow > 0`) is the Task-4 DRIVER's bail,
//! not the pipeline's (the pipeline only exposes the periodogram for a would-be warp).

use std::sync::Arc;

use anyhow::{Result, bail};
use ndarray::Array2;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use super::nn::FastMatrix;
use crate::features::mel::MelFilterBank;
use crate::features::pipeline::{FeatureConfig, SpectralParams, assemble_input_sequence};

/// f32 feature pipeline: windowing -> one real FFT per frame -> periodogram -> reused
/// f64 mel/DCT -> f32 assembled input sequence. Banks + plan built ONCE.
pub struct FastPipeline {
    // Framing constants (from SpectralParams / FeatureConfig).
    window_size: usize, // 1<<order (the real FFT length)
    half_window: usize, // window_size/2 (the get_sequence half-window)
    buffer_size: usize, // window_size+1 (the extracted-frame width)
    bins: usize,        // window_size/2+1 (periodogram length == realfft complex_len)
    shift_frames: usize,
    flag_dc_offset: bool,
    win_coeffs: Option<Vec<f32>>, // length buffer_size, or None (rectangular)

    // Mel/DCT (built ONCE, f64, reused golden code).
    bank: Option<MelFilterBank>,
    use_dct: bool,
    freq_beg: usize,
    freq_end: usize,

    // realfft plan + preallocated scratch (never reallocated per frame).
    r2c: Arc<dyn RealToComplex<f32>>,
    fft_in: Vec<f32>,           // window_size
    fft_out: Vec<Complex<f32>>, // bins
    fft_scratch: Vec<Complex<f32>>,

    // Reusable per-frame windowed buffer (buffer_size).
    frame_buf: Vec<f32>,

    // Reused output buffers (never reallocated when the shape is stable), so repeated
    // calls are bit-identical and per-call allocation stays out of the hot loop.
    perio: FastMatrix, // T x bins
    input: FastMatrix, // T x cols (the assembled sequence)
}

impl FastPipeline {
    /// Build the pipeline from the pre-derived `SpectralParams` + `FeatureConfig` and
    /// the sample `rate` (needed for the mel-bank ctor + the freq grid, as the exact
    /// path threads `audio.sample_rate`). Precomputes ONCE: the realfft plan, the f32
    /// window coefficients, and the mel filterbank + DCT table. Typed-bails an active
    /// LTSV column or periodogram temporal convolution (see module docs).
    pub fn new(params: &SpectralParams, cfg: &FeatureConfig, rate: f64) -> Result<FastPipeline> {
        if params.ltsv_half_window >= 1 {
            bail!(
                "fast::pipeline: LTSV column not supported on the fast path (LTSVwindow active, ltsv_half_window={}); the committed gate configs use LTSVwindow 0",
                params.ltsv_half_window
            );
        }
        // Temporal convolution over the periodogram (convolution_vert): normalized
        // window over 2*conv_size+1, matching sad.rs's getTemporalConvolution. Active
        // iff it materializes a >1-length kernel.
        let temporal_conv = crate::audio::windowing_coefficients(
            &cfg.conv_type,
            true,
            2 * cfg.conv_size as usize + 1,
            0.83333,
        );
        if temporal_conv.as_deref().is_some_and(|c| c.len() > 1) {
            bail!(
                "fast::pipeline: periodogram temporal convolution not supported on the fast path (spectrum_temporal_convolution_size={} active); the committed gate configs use size 0",
                cfg.conv_size
            );
        }

        let window_size = params.window_size;
        let half_window = window_size / 2;
        let buffer_size = params.buffer_size;
        let bins = params.bins;

        // RATE-CONSISTENCY guard (Phase 7 Task 4, T3-review rider): `params` was
        // derived at some rate; this `rate` argument must be the SAME one, or the
        // mel bank (built below from `rate`) is placed on a frequency grid
        // inconsistent with the `params.freq_beg/freq_end` band. `derive` sets
        // `max_freq = freq_end * freqStep` with `freqStep = rate/2/(bins-1)`
        // (`features/pipeline.rs:351,367`); recomputing `freqStep` from THIS `rate`
        // and checking the identity is exact when the rates match (same inputs, same
        // f64 op) and fails by the rate ratio when they differ. `bins >= 2` for every
        // order >= 1, so `bins-1 >= 1`. Debug-only: the driver binds ONE
        // `audio.sample_rate` to both `derive` and `new`, so this never fires in
        // release; it is a belt-and-suspenders against a future caller that forgets.
        debug_assert!(
            {
                let freq_step = rate / 2.0 / (bins as f64 - 1.0);
                (params.freq_end as f64 * freq_step - params.max_freq).abs()
                    <= 1e-9 * params.max_freq.abs().max(1.0)
            },
            "FastPipeline::new rate {rate} is inconsistent with the SpectralParams it \
             was given (freq_end {} * freqStep(rate) != max_freq {}); derive and new \
             must share one sample_rate binding",
            params.freq_end,
            params.max_freq
        );

        // f32 window coefficients (narrowed once from the exact f64 coeffs).
        let win_coeffs =
            crate::audio::windowing_coefficients(&cfg.win_type, false, buffer_size, cfg.win_param)
                .map(|v| v.iter().map(|&x| x as f32).collect());

        // Mel filterbank + DCT table, built ONCE (f64, reused golden ctor). Args match
        // build_mel_bank (`features/pipeline.rs:544-563`, the exact path's canonical
        // MelFilterBank::new site since the Task 8 hoist) exactly.
        let bank = if cfg.nb_bins > 0 {
            Some(MelFilterBank::new(
                cfg.min_mel,
                cfg.max_mel,
                cfg.nb_bins,
                params.min_freq,
                params.max_freq,
                rate,
                params.bins - 1,
                cfg.is_log,
                cfg.nb_dct,
                cfg.ignore_first,
                cfg.deltas_nb,
                cfg.dd_nb,
            ))
        } else {
            None
        };

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(window_size);
        let fft_in = r2c.make_input_vec();
        let fft_out = r2c.make_output_vec();
        let fft_scratch = r2c.make_scratch_vec();
        debug_assert_eq!(
            fft_out.len(),
            bins,
            "realfft complex_len == periodogram bins"
        );

        Ok(FastPipeline {
            window_size,
            half_window,
            buffer_size,
            bins,
            shift_frames: params.shift_frames,
            flag_dc_offset: cfg.flag_dc_offset,
            win_coeffs,
            bank,
            use_dct: cfg.nb_dct > 0,
            freq_beg: params.freq_beg,
            freq_end: params.freq_end,
            r2c,
            fft_in,
            fft_out,
            fft_scratch,
            frame_buf: vec![0.0_f32; buffer_size],
            perio: FastMatrix::zeros(0, 0),
            input: FastMatrix::zeros(0, 0),
        })
    }

    /// Build the full BLSTM input sequence for one channel of f32 samples
    /// (`build_input_sequence`, `features/pipeline.rs:637-645`, f32). Returns the
    /// `T x cols` assembled MFCC/mel/raw sequence.
    pub fn build_input_sequence(&mut self, samples: &[f32]) -> &FastMatrix {
        self.build_input_sequence_parts(samples).0
    }

    /// [`build_input_sequence`] that ALSO returns the raw f32 periodogram and the
    /// pass-1 LTSV column (always `None` here -- LTSV bailed at construction), mirroring
    /// the exact `build_input_sequence_parts`' `(input_seq, perio, ltsv)` shape in f32
    /// for the Task-4 driver. The returned references borrow the pipeline's reused
    /// buffers (valid until the next call).
    pub fn build_input_sequence_parts(
        &mut self,
        samples: &[f32],
    ) -> (&FastMatrix, &FastMatrix, Option<Vec<f32>>) {
        self.compute_periodogram(samples);

        // Widen the f32 periodogram to f64 for the reused golden mel/DCT apply (the
        // f32 VALUE is preserved exactly by the widening).
        let t = self.perio.rows;
        let mut perio64 = Array2::<f64>::zeros((t, self.bins));
        for r in 0..t {
            let base = r * self.bins;
            for c in 0..self.bins {
                perio64[[r, c]] = self.perio.data[base + c] as f64;
            }
        }

        let (mel, dct) = match &self.bank {
            Some(bank) => {
                let fb = bank.apply_filter_bank(&perio64);
                if self.use_dct {
                    let d = bank.apply_dct(&fb);
                    (Some(fb), Some(d))
                } else {
                    (Some(fb), None)
                }
            }
            None => (None, None),
        };

        let input64 = assemble_input_sequence(
            &perio64,
            mel.as_ref(),
            dct.as_ref(),
            None, // LTSV bailed at construction
            self.freq_beg,
            self.freq_end,
        );

        // Narrow the assembled sequence to f32 into the reused output buffer.
        let (it, ic) = input64.dim();
        self.input.rows = it;
        self.input.cols = ic;
        self.input.data.clear();
        self.input.data.reserve(it * ic);
        for r in 0..it {
            for c in 0..ic {
                self.input.data.push(input64[[r, c]] as f32);
            }
        }

        (&self.input, &self.perio, None)
    }

    /// The f32 framing + realfft + periodogram, into the reused `self.perio`
    /// (`T x bins`). `T = frame_nb` is the inclusive-count ceil-divide of the sample
    /// span over the shift; each row is `|RealFFT(windowed_frame)[m]|^2 / window_size`.
    fn compute_periodogram(&mut self, samples: &[f32]) {
        let frames = samples.len();
        let shift = self.shift_frames;
        // frame_nb (`AudioStruct.cpp:531-534`): span = end-begin+1 = frames (begin 0,
        // end frames-1); ceil-divide.
        let frame_nb = if shift == 0 {
            0
        } else if (frames / shift) * shift == frames {
            frames / shift
        } else {
            frames / shift + 1
        };

        self.perio.rows = frame_nb;
        self.perio.cols = self.bins;
        self.perio.data.clear();
        self.perio.data.resize(frame_nb * self.bins, 0.0);

        let n = self.window_size;
        let n_f = n as f32;
        // Clone the plan Arc so the per-frame process call doesn't clash with the
        // simultaneous &mut borrows of fft_in/out/scratch (Arc clone is a refcount bump).
        let r2c = self.r2c.clone();
        for cf in 0..frame_nb {
            let center = cf * shift; // + begin(0)
            self.fill_frame(samples, center);
            self.fft_in.copy_from_slice(&self.frame_buf[..n]);
            r2c.process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch)
                .expect("realfft process (fixed power-of-two length)");
            let base = cf * self.bins;
            for m in 0..self.bins {
                let z = self.fft_out[m];
                self.perio.data[base + m] = (z.re * z.re + z.im * z.im) / n_f;
            }
        }
    }

    /// Fill `self.frame_buf` (buffer_size) with the windowed frame centred on sample
    /// `index` (`get_sequence`, `audio.rs:269-314`, f32). Fresh-zeroed per frame (the
    /// stale-buffer quirk is a proven no-op, see module docs); branch order left-edge ->
    /// right-edge -> interior; DC offset over the full buffer; window multiply last.
    fn fill_frame(&mut self, samples: &[f32], index: usize) {
        let frames = samples.len();
        let hw = self.half_window;
        let cols = self.buffer_size;
        let dc = self.flag_dc_offset;

        for v in self.frame_buf.iter_mut() {
            *v = 0.0;
        }
        // index < frames always here (begin=0/end=frames-1), so no get_sequence no-op.
        let (beg_win, nb_elem, beg_data) = if index < hw {
            (hw - index, hw + index + 1, 0usize)
        } else if index >= frames - hw {
            (0usize, frames + hw - index, index - hw)
        } else {
            (0usize, cols, index - hw)
        };
        self.frame_buf[beg_win..beg_win + nb_elem]
            .copy_from_slice(&samples[beg_data..beg_data + nb_elem]);
        if dc {
            let mut sum = 0.0_f32;
            for &v in self.frame_buf.iter() {
                sum += v;
            }
            let mean = sum / cols as f32;
            for v in self.frame_buf.iter_mut() {
                *v -= mean;
            }
        }
        if let Some(coeffs) = &self.win_coeffs {
            for (v, &c) in self.frame_buf.iter_mut().zip(coeffs.iter()) {
                *v *= c;
            }
        }
    }

    /// Test hook: the stored f32 window coefficients (narrowed once from the exact f64
    /// `windowing_coefficients`), for the f32-vs-f64 windowing-coefficient pin.
    #[cfg(feature = "test-support")]
    pub fn debug_win_coeffs(&self) -> Option<&[f32]> {
        self.win_coeffs.as_deref()
    }
}
