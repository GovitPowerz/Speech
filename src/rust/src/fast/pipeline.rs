//! `fast::pipeline` -- f32 feature extraction on realfft (Phase 7 Task 3).
//!
//! An `f32`, `realfft`-backed transcription of the exact feature front-end's
//! framing + periodogram, feeding the f32 mel/DCT apply in `fast::mel32` (Phase 10
//! Task 5 -- it fed the reused golden f64 apply through a widen bridge until then).
//! The exact sources this mirrors line-for-line (cite them when auditing drift, per the plan's
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
//!   apply + the DCT/mel/raw-band priority. TRANSCRIBED to f32 in `fast::mel32` since
//!   Phase 10 Task 5 (see the FULL-f32 MEL note); no longer called from here.
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
//! 4. FULL-f32 MEL/DCT (Phase 10 Task 5, spec S4). The mel filterbank + DCT table are
//!    built ONCE at construction and the apply runs in f32 through `fast::mel32`
//!    (`FastMelBank::apply_filter_bank`/`apply_dct` + `assemble_input_sequence_f32`), an
//!    op-for-op transcription of the exact `features/mel.rs` + `assemble_input_sequence`
//!    carrying their load-bearing quirks (see that module's doc for the per-quirk source
//!    lines). SUPERSEDES the phase-7 arrangement, which reused the golden f64 mel on the
//!    f32 periodogram WIDENED to f64 and narrowed the assembled sequence back: that
//!    bridge allocated a `T x bins` f64 buffer PER CALL (~23.5 MB on the 60 s SAD bench
//!    -- the measured mechanism behind phase 7's fast-path RSS being HIGHER than exact),
//!    and it is DELETED here, not kept as a mode. Both sites (`build_input_sequence_parts`
//!    and the streaming `assemble_perio_window`) drive ONE shared kernel
//!    ([`FastPipeline::assemble_rows`]), so offline-vs-streamed stays bit-identical BY
//!    CONSTRUCTION. The exact `features/mel.rs` remains byte-untouched -- it is the
//!    transcription ORACLE, no longer a dependency of this module. CONSEQUENCE: the
//!    measured fast-vs-exact deltas are WIDER than phase 7's (f32 error now accumulates
//!    through the mel dots, the DCT product and the delta regressions, not only the
//!    periodogram); every affected pin was re-measured in the S4 sweep (RESULTS.md), with
//!    boundary/argmax identity unchanged.
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
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use super::mel32::{FastMelBank, assemble_input_sequence_f32};
use super::nn::FastMatrix;
use crate::features::pipeline::{FeatureConfig, SpectralParams};

/// f32 feature pipeline: windowing -> one real FFT per frame -> periodogram -> f32
/// mel/DCT (`fast::mel32`) -> f32 assembled input sequence, f32 END TO END. Banks +
/// plan built ONCE.
pub struct FastPipeline {
    // Framing constants (from SpectralParams / FeatureConfig).
    window_size: usize, // 1<<order (the real FFT length)
    half_window: usize, // window_size/2 (the get_sequence half-window)
    buffer_size: usize, // window_size+1 (the extracted-frame width)
    bins: usize,        // window_size/2+1 (periodogram length == realfft complex_len)
    shift_frames: usize,
    flag_dc_offset: bool,
    win_coeffs: Option<Vec<f32>>, // length buffer_size, or None (rectangular)

    // Mel/DCT (built ONCE, f32 -- `fast::mel32`).
    bank: Option<FastMelBank>,
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

        // Mel filterbank + DCT table, built ONCE (f32 -- `FastMelBank::new` runs the
        // exact ctor's f64 geometry walk and narrows only the coefficient values). Args
        // match build_mel_bank (`features/pipeline.rs:544-563`, the exact path's
        // canonical MelFilterBank::new site since the Task 8 hoist) exactly.
        let bank = if cfg.nb_bins > 0 {
            Some(FastMelBank::new(
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

        // f32 mel/DCT/assembly, the SHARED kernel the streaming `assemble_perio_window`
        // also drives (so offline and streamed are bit-identical by construction).
        let t = self.perio.rows;
        let assembled = self.assemble_rows(&self.perio.data, t);

        // Copy into the reused output buffer (its allocation survives across calls).
        self.input.rows = assembled.rows;
        self.input.cols = assembled.cols;
        self.input.data.clear();
        self.input.data.extend_from_slice(&assembled.data);

        (&self.input, &self.perio, None)
    }

    /// THE ONE f32 mel/DCT/assembly kernel, driven by BOTH the whole-sequence path and
    /// the streaming window path: `perio` is a row-major `rows x bins` f32 periodogram
    /// (whole sequence or ring window), and the result is the assembled `rows x cols`
    /// feature sequence. Per-row except `regression_deltas`' cross-row window inside
    /// `apply_dct` -- which is exactly why the streaming caller sizes its window with
    /// the delta REACH (see `fast::stream`). `&self`: the bank is immutable and the
    /// kernel allocates its own output, so neither site can perturb the other.
    fn assemble_rows(&self, perio: &[f32], rows: usize) -> FastMatrix {
        let (mel, dct) = match &self.bank {
            Some(bank) => {
                let fb = bank.apply_filter_bank(perio, rows, self.bins);
                if self.use_dct {
                    let d = bank.apply_dct(&fb);
                    (Some(fb), Some(d))
                } else {
                    (Some(fb), None)
                }
            }
            None => (None, None),
        };

        assemble_input_sequence_f32(
            perio,
            rows,
            self.bins,
            mel.as_ref(),
            dct.as_ref(),
            self.freq_beg,
            self.freq_end,
        )
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

    // ==========================================================================
    // Phase 8 streaming: per-range feature extraction (fast::stream::StreamFrontEnd).
    //
    // The whole-sequence path above (`build_input_sequence*` / `compute_periodogram` /
    // `fill_frame`) is UNTOUCHED -- the phase-7 pipeline pins stay byte-stable. These
    // additions let a streaming session compute a CONTIGUOUS RANGE of periodogram
    // frames from a rolling sample ring, then assemble a RANGE of feature rows from a
    // rolling periodogram ring, both producing values BIT-IDENTICAL to the
    // whole-sequence path for the rows they cover (the per-frame FFT is independent, and
    // `assemble_input_sequence`/`apply_filter_bank`/`apply_dct` are per-row EXCEPT the
    // `regression_deltas` cross-row window inside `apply_dct`, whose interior rows depend
    // only on periodogram rows `[r-reach, r+reach]` and whose start/end rows clamp at the
    // TRUE sequence start/end -- see `fast::stream` for the finalization arithmetic).
    // ==========================================================================

    /// The number of periodogram frames a whole-sequence pass over `n` samples produces
    /// (`compute_periodogram`'s `frame_nb`, `AudioStruct.cpp:531-534`): the inclusive-count
    /// ceil-divide of the sample span over the shift. Public so the streaming front-end
    /// derives its EOS frame count from the same arithmetic (no divergence).
    pub fn frame_count(&self, n: usize) -> usize {
        let shift = self.shift_frames;
        if shift == 0 {
            0
        } else if (n / shift) * shift == n {
            n / shift
        } else {
            n / shift + 1
        }
    }

    /// Compute periodogram frames `[from_frame, to_frame)` from a caller-provided sample
    /// ring, appending one `bins`-wide row per frame to `out` (row-major). `samples` is a
    /// contiguous slice of the ring whose element 0 is ABSOLUTE sample index `ring_base`;
    /// `total_samples` is the total sample count seen so far (the `frames` the framing
    /// geometry clamps against -- mid-stream every requested frame is interior/left-edge,
    /// at EOS the tail frames use the right-edge zero-pad). Reuses the pipeline's own
    /// realfft plan + per-frame scratch (`fft_in`/`fft_out`/`fft_scratch`/`frame_buf`);
    /// each frame is independent, so a frame computed here is bit-identical to the same
    /// frame in a whole-sequence `compute_periodogram`.
    pub fn append_periodogram_frames(
        &mut self,
        samples: &[f32],
        ring_base: usize,
        from_frame: usize,
        to_frame: usize,
        total_samples: usize,
        out: &mut Vec<f32>,
    ) {
        let n = self.window_size;
        let n_f = n as f32;
        let shift = self.shift_frames;
        let r2c = self.r2c.clone();
        for cf in from_frame..to_frame {
            let center = cf * shift;
            self.fill_frame_ring(samples, ring_base, center, total_samples);
            self.fft_in.copy_from_slice(&self.frame_buf[..n]);
            r2c.process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch)
                .expect("realfft process (fixed power-of-two length)");
            for m in 0..self.bins {
                let z = self.fft_out[m];
                out.push((z.re * z.re + z.im * z.im) / n_f);
            }
        }
    }

    /// Assemble feature rows from a periodogram-ring WINDOW, returning only local rows
    /// `[from_local, to_local)`. `perio_window` is `window_rows x bins` (row-major f32),
    /// run through the SAME [`assemble_rows`](Self::assemble_rows) kernel the
    /// whole-sequence path drives -- f32 end to end since Phase 10 Task 5, so the two
    /// sites cannot diverge even in the last ULP. The caller sizes the window so every
    /// EXTRACTED row's `regression_deltas` reach lands on real neighbours (interior) or
    /// on the true sequence start/end (clamped edge), making the extracted rows
    /// bit-identical to the whole-sequence assembly. `&self` (read-only: reuses
    /// `bank`/`use_dct`/`freq_beg`/`freq_end`, never the whole-sequence `perio`/`input`
    /// buffers).
    pub fn assemble_perio_window(
        &self,
        perio_window: &[f32],
        window_rows: usize,
        from_local: usize,
        to_local: usize,
    ) -> FastMatrix {
        let full = self.assemble_rows(perio_window, window_rows);

        let ic = full.cols;
        let rows = to_local - from_local;
        let mut out = FastMatrix {
            data: Vec::with_capacity(rows * ic),
            rows,
            cols: ic,
        };
        out.data
            .extend_from_slice(&full.data[from_local * ic..to_local * ic]);
        out
    }

    /// [`fill_frame`] for the streaming sample ring: identical windowing arithmetic, but
    /// reading absolute sample index `i` from `samples[i - ring_base]` and taking the
    /// total frame count `total` explicitly (so the right-edge branch fires at EOS on the
    /// same geometry the whole-sequence `fill_frame` derives from `samples.len()`). Kept
    /// SEPARATE from `fill_frame` so the whole-sequence path is provably untouched.
    fn fill_frame_ring(&mut self, samples: &[f32], ring_base: usize, index: usize, total: usize) {
        let hw = self.half_window;
        let cols = self.buffer_size;
        let dc = self.flag_dc_offset;

        for v in self.frame_buf.iter_mut() {
            *v = 0.0;
        }
        let (beg_win, nb_elem, beg_data) = if index < hw {
            (hw - index, hw + index + 1, 0usize)
        } else if index >= total - hw {
            (0usize, total + hw - index, index - hw)
        } else {
            (0usize, cols, index - hw)
        };
        let src_start = beg_data - ring_base;
        self.frame_buf[beg_win..beg_win + nb_elem]
            .copy_from_slice(&samples[src_start..src_start + nb_elem]);
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
