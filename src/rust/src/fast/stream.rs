//! `fast::stream` -- the SAD streaming front-end (Phase 8 Task 2).
//!
//! [`StreamFrontEnd`] turns a chunked f32 sample stream into the SAME assembled BLSTM
//! feature rows the offline fast path ([`super::pipeline::FastPipeline::build_input_sequence`])
//! produces on the whole (frozen-gain) audio -- BIT-IDENTICAL at any chunking. It owns the
//! carried per-sample state and the two rolling rings; the stateless compute (framing,
//! FFT, mel/DCT) lives on the borrowed [`FastPipeline`] (`take_ready_rows`/`flush` take
//! `&mut FastPipeline`), reusing its plan/banks/scratch.
//!
//! OFFLINE ORDER REPRODUCED (the bit-equal contract, spec S1.2). The offline fast SAD
//! path is: `read_audio` applies the gain (f64) -> the driver applies pre-emphasis then
//! noise on the whole f64 channel (`fast/driver.rs:296-301`, gated `preemph_ratio > 0` /
//! `noise_seed > 0`) -> the channel is narrowed to f32 -> `build_input_sequence`. This
//! front-end applies, PER SAMPLE and in f64 (narrowing to f32 only when buffering, exactly
//! like the driver's `.map(|x| x as f32)` seam):
//!   1. fixed gain: `x / gain.max(1e-3)` (`audio.rs::apply_fixed_gain`).
//!   2. pre-emphasis with a 1-sample CARRY (`audio.rs::apply_preemph`, `:99-108`). The
//!      offline reverse loop `data[k] -= r*data[k-1]` leaves `out[0]=in[0]` and gives
//!      `out[k]=in[k]-r*in[k-1]` for `k>=1` (the subtracted `in[k-1]` is the pre-emphasis
//!      INPUT = the gained value, since the reverse order reads it before it is rewritten);
//!      the carry is that gained value, persisted ACROSS chunk boundaries.
//!   3. dither noise with a table-INDEX carry (`audio.rs::apply_noise`, `:112-121`):
//!      `+ 2*mag*random_uniform(k) - mag`, where `k` is the ABSOLUTE sample index (the
//!      running push count) into the fixed [`crate::constants::random_uniform`] table.
//!
//! Gain/preemph/noise all run in f64 then narrow once, so the buffered f32 samples equal
//! the offline driver's f32 channel exactly (the raw `i16/32768` values are f32-exact, so
//! widening the f32 input back to f64 for the gain is lossless).
//!
//! FRAME-COMPLETION ARITHMETIC. Framing mirrors `compute_periodogram`/`fill_frame`
//! (`fast/pipeline.rs`): periodogram frame `cf` is centred on sample `cf*shift` and reads
//! window `[cf*shift - hw, cf*shift + hw]` (`hw = window_size/2`), left-clamped to 0 and
//! right-clamped to the last sample. A frame is FINAL once its right edge sample
//! `cf*shift + hw` has arrived AND that arrival proves the frame is not a right-edge frame
//! (its window fits within the samples so far, so its interior/left-edge computation is
//! final and independent of the still-unknown total). Hence
//!   `pgram_ready(n) = (n >= hw+1) ? (n-1-hw)/shift + 1 : 0`
//! non-right-edge periodogram frames are finalizable from `n` samples. The right-edge tail
//! frames (window past EOS, zero-padded) wait for [`StreamFrontEnd::flush`], which knows
//! the total and uses the same `frame_count` the whole-sequence path derives.
//!
//! FEATURE-ROW REACH (the deltas). tier2's `apply_dct` runs `regression_deltas`
//! (`features/mel.rs`): a feature row is a per-frame mel/DCT PLUS delta/delta-delta over a
//! `+-reach` periodogram-row window (`reach = ComputeDeltasNb + ComputeDeltaDeltasNb` when
//! both active), so assembled feature row `r` depends on periodogram rows
//! `[r-reach, r+reach]`, edge-CLAMPED at the true sequence start/end. Thus feature row `r`
//! is final once periodogram frame `r+reach` is final:
//!   `feature_ready(n) = max(0, pgram_ready(n) - reach)`
//! and each ready batch is assembled from a periodogram-ring WINDOW padded by `reach` on
//! both sides ([`FastPipeline::assemble_perio_window`]), so every extracted row's delta
//! reach lands on real neighbours (interior) -- bit-identical to the whole-sequence
//! assembly. The start rows clamp at periodogram frame 0 (always retained until emitted);
//! the end rows clamp at the true last frame, resolved at `flush`. SDC (`ComputeDeltasNb <
//! 0`) has a different, un-gate-exercised reach and typed-bails at construction.

use anyhow::{Result, bail};

use crate::constants::random_uniform;
use crate::features::pipeline::{FeatureConfig, SpectralParams};

use super::nn::FastMatrix;
use super::pipeline::FastPipeline;

/// The chunked-input SAD feature front-end. See the module docs for the bit-equal
/// contract, the offline order, and the frame/feature finalization arithmetic.
pub struct StreamFrontEnd {
    // Framing constants (from SpectralParams -- the front-end derives the framing math
    // itself; the pipeline holds the identical values privately for compute).
    half_window: usize,  // window_size / 2 (the fill_frame half-window)
    shift_frames: usize, // periodogram frame stride, in samples
    bins: usize,         // periodogram width (== assemble window column count)
    reach: usize,        // delta context: feature row r needs periodogram rows [r-reach, r+reach]

    // Per-sample transforms (offline order: gain -> preemph -> noise, all f64).
    fixed_gain: f64,      // floored at 1e-3 once at construction (apply_fixed_gain)
    preemph_ratio: f64,   // > 0.0 activates (mirrors apply_preemph's internal `ratio > 0` gate)
    noise_magnitude: f64, // > 0.0 activates the dither

    // Carried per-sample state (persists across chunk boundaries -- the R3 catcher).
    prev_gained: f64, // the previous sample's GAINED value (the pre-emphasis carry)
    seen_first: bool, // false until the first sample (so out[0] = in[0], no carry subtraction)
    total_pushed: usize, // absolute sample count == the next sample's dither table index

    // Sample ring: absolute sample index `i` lives at `samples[i - sample_base]`.
    samples: Vec<f32>,
    sample_base: usize,

    // Periodogram ring: absolute frame `i` lives at `pgram[(i - pgram_base)*bins ..]`.
    pgram: Vec<f32>,
    pgram_base: usize,
    pgram_computed: usize, // number of periodogram frames computed (absolute)

    emitted: usize, // number of feature rows emitted (absolute)
    finished: bool, // flush has run (idempotent guard)
}

impl StreamFrontEnd {
    /// Build the front-end from the derived `params` + parsed `cfg` (the SAME pair the
    /// caller feeds [`FastPipeline::new`]), the sample `rate`, and the frozen per-sample
    /// transform scalars. `fixed_gain` is floored at `1e-3` here (once), mirroring
    /// `audio.rs::apply_fixed_gain`. `preemph`/`noise_magnitude` follow the driver gates:
    /// `preemph > 0.0` activates pre-emphasis (the session passes `feature_cfg.preemph_ratio`),
    /// `noise_magnitude > 0.0` activates the dither (the session passes `noise_ratio` iff
    /// `noise_seed > 0`, else `0.0`). Typed-bails SDC deltas (`ComputeDeltasNb < 0`), whose
    /// reach the streaming window is not sized for (no gate config uses it).
    pub fn new(
        params: &SpectralParams,
        cfg: &FeatureConfig,
        rate: f64,
        fixed_gain: f64,
        noise_magnitude: f64,
        preemph: f64,
    ) -> Result<StreamFrontEnd> {
        if cfg.deltas_nb < 0 {
            bail!(
                "fast::stream: SDC deltas (ComputeDeltasNb {} < 0) are not supported on the \
                 streaming front-end (the delta-context window is not sized for the SDC reach); \
                 the gate configs use ComputeDeltasNb >= 0",
                cfg.deltas_nb
            );
        }
        let reach = if cfg.deltas_nb > 0 {
            cfg.deltas_nb as usize + if cfg.dd_nb > 0 { cfg.dd_nb as usize } else { 0 }
        } else {
            0
        };

        // Rate-consistency guard (mirrors FastPipeline::new's belt-and-suspenders): the
        // params must have been derived at THIS rate, or the frame stride the front-end
        // reads from `params.shift_frames` is inconsistent with `params.shift_sec`.
        debug_assert!(
            (params.shift_frames as f64 - (params.shift_sec * rate).round()).abs() < 0.5,
            "StreamFrontEnd::new rate {rate} is inconsistent with the SpectralParams it was \
             given (shift_frames {} != round(shift_sec {} * rate))",
            params.shift_frames,
            params.shift_sec
        );

        Ok(StreamFrontEnd {
            half_window: params.window_size / 2,
            shift_frames: params.shift_frames,
            bins: params.bins,
            reach,
            fixed_gain: fixed_gain.max(1e-3),
            preemph_ratio: preemph,
            noise_magnitude,
            prev_gained: 0.0,
            seen_first: false,
            total_pushed: 0,
            samples: Vec::new(),
            sample_base: 0,
            pgram: Vec::new(),
            pgram_base: 0,
            pgram_computed: 0,
            emitted: 0,
            finished: false,
        })
    }

    /// Buffer a chunk of raw f32 samples, applying the frozen per-sample transforms in the
    /// offline order (gain -> preemph-with-carry -> noise-with-index-carry, all f64, then
    /// narrow to f32). Buffers only -- framing/FFT/assembly happen in
    /// [`take_ready_rows`](Self::take_ready_rows) (which has the pipeline).
    pub fn push(&mut self, samples: &[f32]) {
        self.samples.reserve(samples.len());
        for &x in samples {
            // 1. fixed gain in f64 (the raw i16/32768 values widen losslessly).
            let gained = x as f64 / self.fixed_gain;

            // 2. pre-emphasis with the 1-sample carry (out[0] = in[0]).
            let pre = if self.preemph_ratio > 0.0 {
                if self.seen_first {
                    gained - self.preemph_ratio * self.prev_gained
                } else {
                    gained
                }
            } else {
                gained
            };
            self.prev_gained = gained;
            self.seen_first = true;

            // 3. dither noise with the table-index carry (absolute sample index).
            let noised = if self.noise_magnitude > 0.0 {
                let m = self.noise_magnitude;
                pre + (2.0 * m * random_uniform(self.total_pushed) - m)
            } else {
                pre
            };

            self.samples.push(noised as f32);
            self.total_pushed += 1;
        }
    }

    /// Periodogram frames finalizable from the samples pushed so far (non-right-edge only;
    /// see the module docs' `pgram_ready`).
    fn pgram_ready(&self) -> usize {
        let hw = self.half_window;
        let n = self.total_pushed;
        if n < hw + 1 {
            0
        } else {
            (n - 1 - hw) / self.shift_frames + 1
        }
    }

    /// Compute any newly-finalizable periodogram frames + emit any newly-finalizable
    /// feature rows, reusing `pipeline`'s plan/banks/scratch. Returns the newly-completed
    /// feature rows (a `k x cols` [`FastMatrix`], `k == 0` when nothing completed this
    /// call). The concatenation of every `take_ready_rows` return plus the [`flush`](Self::flush)
    /// tail equals the offline `build_input_sequence` on the same audio, BIT-IDENTICAL.
    pub fn take_ready_rows(&mut self, pipeline: &mut FastPipeline) -> FastMatrix {
        // Advance the periodogram ring to the newly-final (non-right-edge) frames.
        let pgram_target = self.pgram_ready();
        if pgram_target > self.pgram_computed {
            pipeline.append_periodogram_frames(
                &self.samples,
                self.sample_base,
                self.pgram_computed,
                pgram_target,
                self.total_pushed,
                &mut self.pgram,
            );
            self.pgram_computed = pgram_target;
            self.trim_sample_ring();
        }

        // Feature rows finalizable now: r final once periodogram frame r+reach is final.
        let feature_target = self.pgram_computed.saturating_sub(self.reach);
        if feature_target <= self.emitted {
            return FastMatrix::zeros(0, 0);
        }
        let out = self.assemble_range(pipeline, self.emitted, feature_target, self.pgram_computed);
        self.emitted = feature_target;
        self.trim_pgram_ring();
        out
    }

    /// EOS: compute the right-edge tail periodogram frames (zero-padded windows past the
    /// last sample) and emit every remaining feature row (whose delta reach now clamps at
    /// the TRUE last frame). Returns the tail rows. Idempotent (a second call returns an
    /// empty matrix). After `flush`, the concatenation of all emitted rows is the complete
    /// offline `build_input_sequence`.
    pub fn flush(&mut self, pipeline: &mut FastPipeline) -> FastMatrix {
        if self.finished {
            return FastMatrix::zeros(0, 0);
        }
        self.finished = true;

        let frame_nb = pipeline.frame_count(self.total_pushed);
        if frame_nb > self.pgram_computed {
            pipeline.append_periodogram_frames(
                &self.samples,
                self.sample_base,
                self.pgram_computed,
                frame_nb,
                self.total_pushed,
                &mut self.pgram,
            );
            self.pgram_computed = frame_nb;
        }

        if frame_nb <= self.emitted {
            return FastMatrix::zeros(0, 0);
        }
        let out = self.assemble_range(pipeline, self.emitted, frame_nb, frame_nb);
        self.emitted = frame_nb;
        out
    }

    /// Assemble feature rows `[from, to)` from the periodogram-ring window `[from-reach,
    /// window_hi)` (padded by `reach` so every extracted row's delta reach is interior /
    /// true-edge-clamped). `window_hi` is the exclusive right bound of finalized
    /// periodogram frames available (`pgram_computed` mid-stream, `frame_nb` at flush).
    fn assemble_range(
        &self,
        pipeline: &FastPipeline,
        from: usize,
        to: usize,
        window_hi: usize,
    ) -> FastMatrix {
        let window_lo = from.saturating_sub(self.reach);
        debug_assert!(
            window_lo >= self.pgram_base,
            "assemble window underruns the ring"
        );
        debug_assert!(
            window_hi <= self.pgram_computed,
            "assemble window overruns computed frames"
        );
        let window_rows = window_hi - window_lo;
        let lo_off = (window_lo - self.pgram_base) * self.bins;
        let hi_off = (window_hi - self.pgram_base) * self.bins;
        pipeline.assemble_perio_window(
            &self.pgram[lo_off..hi_off],
            window_rows,
            from - window_lo,
            to - window_lo,
        )
    }

    /// Drop sample-ring elements no future frame needs. The next frame to compute is
    /// `pgram_computed`; its leftmost sample is `pgram_computed*shift - hw`.
    fn trim_sample_ring(&mut self) {
        let keep_from = (self.pgram_computed * self.shift_frames).saturating_sub(self.half_window);
        if keep_from > self.sample_base {
            self.samples.drain(0..keep_from - self.sample_base);
            self.sample_base = keep_from;
        }
    }

    /// Drop periodogram-ring rows no future feature row needs. The next feature row is
    /// `emitted`; its leftmost periodogram frame is `emitted - reach`.
    fn trim_pgram_ring(&mut self) {
        let keep_from = self.emitted.saturating_sub(self.reach);
        if keep_from > self.pgram_base {
            self.pgram
                .drain(0..(keep_from - self.pgram_base) * self.bins);
            self.pgram_base = keep_from;
        }
    }
}
