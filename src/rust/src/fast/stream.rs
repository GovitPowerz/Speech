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

use anyhow::{Result, anyhow, bail};
use indexmap::IndexMap;

use crate::constants::random_uniform;
use crate::features::pipeline::{FeatureConfig, SpectralParams};
use crate::nn::blstm::BlstmConfig;
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmenter::{DriverConfig, SegmenterConfig, smooth_segmentation};

use super::driver::build_aligned_spec;
use super::nn::{FastBlstm, FastMatrix, external_normalize_f32, window_begin, window_end};
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

/// The streaming OVERLAP engine (Phase 8 Task 3): the online twin of
/// [`FastBlstm::feed_forward_overlap`]. It consumes the (ALREADY-NORMALIZED) feature-row
/// stream [`StreamFrontEnd`] emits, fires each overlap window as soon as its input span
/// exists, accumulates per-output-row sums + window counts in a rolling ring, and emits
/// each output row (posterior = sum/count) the moment its LAST covering window has fired.
///
/// BIT-EQUAL CONTRACT (spec S2, the task gate). The concatenation of every
/// [`push_rows`](Self::push_rows) return plus the [`flush`](Self::flush) tail equals the
/// offline `feed_forward_overlap` on the WHOLE sequence, BIT-IDENTICAL -- including the
/// `0/0 = NaN` uncovered rows -- at any row-push granularity. Two properties carry it:
///  1. IDENTICAL WINDOW SET. The offline loop steps `jj` by `window_shift` while `jj <
///     input_rows`, each window spanning `[window_begin(jj), window_end(begin, ..,
///     input_rows)]`. `begin` is `input_rows`-independent; the end-clamp is the only
///     `input_rows`-dependent part. A window fires HERE mid-stream once `begin +
///     2*window_size < rows_so_far` -- exactly the offline UNCLAMPED windows (the clamp
///     branch of [`window_end`] is then dead, so the span matches); the CLAMPED tail
///     windows (largest `jj`, `begin + 2*window_size >= input_rows`) fire at
///     [`flush`](Self::flush) with the true total. Every offline window fires exactly once
///     (the fire gate assumes `window_size >= ssr`, true for every real overlap config --
///     the SAD gate config has window 163, ssr 4).
///  2. IDENTICAL ACCUMULATION ORDER. The `+=` is order-sensitive in f32; a row covered by
///     several windows must sum them in `jj`-ascending order. `window_begin` is monotonic
///     in `jj`, so windows become fireable in `jj` order and flush windows (the `jj` tail)
///     come strictly after every mid-stream window -- so each row accumulates in the
///     offline order, through the SHARED [`FastBlstm::overlap_window_step`].
///
/// FINALIZATION. Output row `orow` is covered by windows with `obeg <= orow` (`obeg =
/// begin/ssr`); its LAST covering window is the largest `jj` with `obeg(jj) <= orow`.
/// After firing a window, the next window is `next_jj`; no window at or beyond it covers a
/// row `< obeg(next_jj)` (monotone `obeg`), so rows `[emitted, obeg(next_jj))` are final
/// and emitted. The structural emission lag is a FULL window: `orow` finalizes no later
/// than input frame `orow*ssr + 2*window_size` (+ up to `ssr` grid slack).
///
/// SEAM (mono, spec S1.2). `push_rows` receives rows ALREADY type-1 normalized: type-1 is
/// a per-row affine (`(x - mean_col)/std_col`, [`super::nn::external_normalize_f32`]), so
/// it commutes with row-streaming -- the front-end/caller applies it per row before
/// pushing, keeping this layer purely about windowing. The whole-sequence type -1
/// self-normalization is NOT streamable and the session bails on it upstream (Task 1/T5).
/// Output rows start from ZERO (channel 0; the cross-channel `result_vec` seeding is a
/// multi-channel concern deferred by the mono-first design).
pub struct StreamOverlap {
    // Window geometry (from the SAD driver's getBLSTMParam, in periodogram-frame units).
    window_size: usize,
    window_shift: usize,
    ssr: usize,
    output_size: usize, // posterior width == emitted/accumulator column count

    // Input feature-row ring: absolute row `i` lives at `input_ring[(i - input_base)*cols]`.
    input_cols: usize, // 0 until the first push establishes it
    input_ring: Vec<f32>,
    input_base: usize,
    rows_so_far: usize, // total input feature rows pushed

    // Window cursor + output accumulator ring. `acc_base == emitted` invariant (emitted
    // rows are drained immediately), so absolute output row `i` lives at
    // `acc[(i - acc_base)*output_size]`, count at `counts[i - acc_base]`.
    next_jj: usize,
    acc: Vec<f32>,
    counts: Vec<f32>,
    acc_base: usize,
    emitted: usize,
    finished: bool,
}

impl StreamOverlap {
    /// Build the overlap engine from the SAD net's window geometry: `net_window`
    /// (`window_size`) + `net_shift` (`window_shift`) in periodogram-frame units, the
    /// whole-BLSTM `ssr` ([`FastBlstm::sub_sampling_ratio`]), and the posterior
    /// `output_size` ([`FastBlstm::output_size`]). `window_shift >= 1` and `ssr >= 1` are
    /// required (the driver's `get_blstm_param` floors the shift at 1 and bails the
    /// non-overlap path); `window_size >= ssr` is required too (every real overlap config;
    /// the partition proof above depends on it) -- all three are debug-asserted below.
    pub fn new(
        net_window: usize,
        net_shift: usize,
        ssr: usize,
        output_size: usize,
    ) -> StreamOverlap {
        debug_assert!(
            net_shift >= 1,
            "window_shift must be >= 1 (else the window loop stalls)"
        );
        debug_assert!(ssr >= 1, "ssr must be >= 1");
        // The type docs' partition proof (property 1, IDENTICAL WINDOW SET) assumes this.
        debug_assert!(
            net_window >= ssr,
            "window_size must be >= ssr (a ws < ssr config silently diverges from offline windowing)"
        );
        StreamOverlap {
            window_size: net_window,
            window_shift: net_shift,
            ssr,
            output_size,
            input_cols: 0,
            input_ring: Vec::new(),
            input_base: 0,
            rows_so_far: 0,
            next_jj: 0,
            acc: Vec::new(),
            counts: Vec::new(),
            acc_base: 0,
            emitted: 0,
            finished: false,
        }
    }

    /// Buffer a chunk of ALREADY-NORMALIZED feature rows, fire every window whose input
    /// span now exists (`begin + 2*window_size < rows_so_far`), and return the output rows
    /// that finalized this call (`k x output_size`, `k == 0` when nothing completed).
    pub fn push_rows(&mut self, rows: &FastMatrix, net: &mut FastBlstm) -> FastMatrix {
        if rows.rows > 0 {
            if self.input_cols == 0 {
                self.input_cols = rows.cols;
            }
            debug_assert_eq!(
                rows.cols, self.input_cols,
                "push_rows col count must be stable"
            );
            self.input_ring.extend_from_slice(&rows.data);
            self.rows_so_far += rows.rows;
        }

        // Fire every window whose UNCLAMPED span now fits (the clamp branch of window_end
        // is dead here, so the span == the offline unclamped span for these windows).
        loop {
            let begin = window_begin(self.next_jj, self.window_size, self.ssr);
            if begin + 2 * self.window_size >= self.rows_so_far {
                break;
            }
            let end = window_end(begin, self.window_size, self.ssr, self.rows_so_far);
            self.fire(net, begin, end + 1 - begin);
            self.next_jj += self.window_shift;
        }
        self.trim_input_ring();

        // Rows below the next (unfired) window's obeg can never be covered again -> final.
        let frontier = window_begin(self.next_jj, self.window_size, self.ssr) / self.ssr;
        self.emit_upto(frontier)
    }

    /// EOS: fire the remaining CLAMPED tail windows (partial windows snapped to the true
    /// total) and emit every remaining output row -- rows uncovered by any window divide
    /// `0/0 -> NaN`, exactly as the offline post-loop division does. Idempotent (a second
    /// call returns empty). After `flush`, the full concatenation is the offline overlap.
    pub fn flush(&mut self, net: &mut FastBlstm) -> FastMatrix {
        if self.finished {
            return FastMatrix::zeros(0, self.output_size);
        }
        self.finished = true;

        let total = self.rows_so_far;
        while self.next_jj < total {
            let begin = window_begin(self.next_jj, self.window_size, self.ssr);
            let end = window_end(begin, self.window_size, self.ssr, total);
            let length_seq = end + 1 - begin;
            if length_seq / self.ssr > 0 {
                self.fire(net, begin, length_seq);
            }
            self.next_jj += self.window_shift;
        }

        // The offline output has `input_rows / ssr` rows (== the driver's real_vec_size,
        // and == the max covered row for every overlap config); emit all remaining.
        let output_rows = total / self.ssr;
        debug_assert!(
            self.acc_base + self.counts.len() <= output_rows,
            "window coverage {} exceeds derived output rows {output_rows}",
            self.acc_base + self.counts.len()
        );
        self.emit_upto(output_rows)
    }

    /// Forward one window (`input[begin, begin+length_seq)`) and accumulate it into the
    /// ring at `obeg = begin/ssr`, via the SHARED [`FastBlstm::overlap_window_step`] so
    /// the forward + `+=` are bit-identical to the offline path. A `length_short == 0`
    /// window is skipped (matching the offline `if length_short > 0` gate).
    fn fire(&mut self, net: &mut FastBlstm, begin: usize, length_seq: usize) {
        let length_short = length_seq / self.ssr;
        if length_short == 0 {
            return;
        }
        let cols = self.input_cols;
        let lo = (begin - self.input_base) * cols;
        let hi = (begin + length_seq - self.input_base) * cols;
        let block = FastMatrix {
            data: self.input_ring[lo..hi].to_vec(),
            rows: length_seq,
            cols,
        };
        let obeg = begin / self.ssr;
        self.grow_acc(obeg + length_short);
        debug_assert!(
            obeg >= self.acc_base,
            "window obeg underruns the accumulator ring"
        );
        let obeg_local = obeg - self.acc_base;
        let got = net.overlap_window_step(
            &block,
            obeg_local,
            &mut self.acc,
            &mut self.counts,
            self.output_size,
        );
        debug_assert_eq!(got, length_short, "stream overlap window output-row count");
    }

    /// Grow the accumulator ring so it holds absolute output rows `[acc_base, top)` (new
    /// rows sum 0 / count 0 -> `NaN` on division for the uncovered ones).
    fn grow_acc(&mut self, top: usize) {
        let need = top - self.acc_base;
        if self.counts.len() < need {
            self.counts.resize(need, 0.0);
            self.acc.resize(need * self.output_size, 0.0);
        }
    }

    /// Emit finalized output rows `[emitted, frontier)` as `sum/count` (NaN for count 0),
    /// draining them from the ring. Returns a `(frontier-emitted) x output_size` matrix.
    fn emit_upto(&mut self, frontier: usize) -> FastMatrix {
        if frontier <= self.emitted {
            return FastMatrix::zeros(0, self.output_size);
        }
        self.grow_acc(frontier); // pad any uncovered gap/tail rows (count 0)
        let cols = self.output_size;
        let n = frontier - self.emitted;
        let mut data = Vec::with_capacity(n * cols);
        for row in self.emitted..frontier {
            let li = row - self.acc_base;
            let cnt = self.counts[li];
            for c in 0..cols {
                data.push(self.acc[li * cols + c] / cnt);
            }
        }
        let drop = frontier - self.acc_base;
        self.acc.drain(0..drop * cols);
        self.counts.drain(0..drop);
        self.acc_base = frontier;
        self.emitted = frontier;
        FastMatrix {
            data,
            rows: n,
            cols,
        }
    }

    /// Drop input-ring rows no future window reads. The next window to fire is `next_jj`;
    /// its (and every later window's) leftmost input row is `window_begin(next_jj)`. Capped
    /// at `rows_so_far`: with a large shift the next window can begin BEYOND the buffered
    /// rows (a gap not yet arrived), so we drop all buffered rows but never past what
    /// exists (the ring stays contiguous `[input_base, rows_so_far)`; `input_base` never
    /// passes the pending window's begin, so its rows are retained once they arrive).
    fn trim_input_ring(&mut self) {
        if self.input_cols == 0 {
            return;
        }
        let keep_from =
            window_begin(self.next_jj, self.window_size, self.ssr).min(self.rows_so_far);
        if keep_from > self.input_base {
            self.input_ring
                .drain(0..(keep_from - self.input_base) * self.input_cols);
            self.input_base = keep_from;
        }
    }
}

// ===========================================================================
// StreamDecision (Phase 8 Task 4): the incremental decision layer.
// ===========================================================================

/// One finalized streaming segment. `begin_s`/`end_s` are the SMOOTHED partition
/// boundaries (absolute audio time, `time_offset` folded in, post-smoothing);
/// `class` is the segment's type ([`SegClass::Speech`] or [`SegClass::Other`] -- the
/// smoothed partition covers the whole timeline); `emitted_at_audio_s` is the
/// posterior-row frontier time when the segment finalized (`received_rows * time_step
/// + time_offset`), for the latency accounting T5 pins (lag = emitted_at - end_s).
#[derive(Debug, Clone, PartialEq)]
pub struct EmittedSegment {
    pub begin_s: f64,
    pub end_s: f64,
    pub class: SegClass,
    pub emitted_at_audio_s: f64,
}

/// The persistent hysteresis-with-area state machine -- the streaming twin of
/// [`crate::tasks::segmenter::update_segmentation_raw`] (`segmenter.rs:290-415`),
/// transcribed LOOP-STATE-for-LOOP-STATE and advanced one posterior value at a time.
///
/// The batch loop keeps `begin/end/begin_area/end_area/has_begun/has_ended` plus the
/// running index `ii` and `results[ii-1]`; the only work is to persist all of them
/// across calls. `advance` runs one iteration of the batch body (the INIT check for the
/// first value, `:312-315`; the loop body for the rest, `:317-407`), returning a raw
/// segment `(begin+off, end+off)` on the iteration that labels one. `flush_tail` is the
/// tail force-emit (`:409-412`), where the legacy `dt * results.len()` end time uses
/// `results.len()` == the total convolved-value count == `idx` at EOS.
struct HystState {
    t_r: f64,
    a_r: f64,
    t_f: f64,
    a_f: f64,
    off: f64,
    dt: f64,
    begin: f64,
    end: f64,
    begin_area: f64,
    end_area: f64,
    has_begun: bool,
    has_ended: bool,
    idx: usize, // number of values processed so far (== the batch loop's `ii`)
    prev: f64,  // the previous value (the batch `results[ii-1]`)
}

impl HystState {
    fn new(cfg: &SegmenterConfig, off: f64, dt: f64) -> HystState {
        HystState {
            t_r: cfg.rising,
            a_r: cfg.area_rising,
            t_f: cfg.falling,
            a_f: cfg.area_falling,
            off,
            dt,
            begin: -1.0,
            end: -1.0,
            begin_area: -1.0,
            end_area: -1.0,
            has_begun: false,
            has_ended: false,
            idx: 0,
            prev: 0.0,
        }
    }

    /// Advance one convolved posterior value. Returns the raw segment labeled on this
    /// step, if any (at most one per value, exactly as the batch loop labels at most once
    /// per `ii`). Direct transcription of `update_segmentation_raw`.
    fn advance(&mut self, r: f64) -> Option<(f64, f64)> {
        if self.idx == 0 {
            // INIT (:312-315): begin/begin_area seed, WITHOUT has_begun; the ii>=1 body's
            // rising branch then keys off `begin < 0` being false.
            if r >= self.t_r {
                self.begin = 0.0;
                self.begin_area = 0.0;
            }
            self.prev = r;
            self.idx = 1;
            return None;
        }

        let ii = self.idx as f64;
        let r_prev = self.prev;
        let (t_r, a_r, t_f, a_f, dt) = (self.t_r, self.a_r, self.t_f, self.a_f, self.dt);
        let mut out = None;

        if !self.has_begun {
            if self.begin < 0.0 {
                if r >= t_r && r_prev < t_r {
                    self.begin = dt * (ii - (r - t_r) / (r - r_prev));
                    self.begin_area = (ii - self.begin / dt) * (r - t_r) / 2.0;
                    if self.begin_area >= a_r {
                        self.has_begun = true;
                    }
                }
            } else if r >= t_r {
                self.begin_area += (r + r_prev - t_r * 2.0) / 2.0;
                if self.begin_area >= a_r {
                    self.has_begun = true;
                }
            } else if r < t_r && r_prev >= t_r {
                self.begin_area += (r_prev - t_r) * (t_r - r_prev) / (r - r_prev) / 2.0;
                if self.begin_area >= a_r {
                    self.has_begun = true;
                } else {
                    self.begin = -1.0;
                    self.begin_area = -1.0;
                    self.has_begun = false;
                }
            } else {
                self.begin = -1.0;
                self.begin_area = -1.0;
                self.has_begun = false;
            }
        }

        if self.has_begun {
            if self.end < 0.0 {
                if r <= t_f && r_prev > t_f {
                    self.end = dt * (ii - (r - t_f) / (r - r_prev));
                    self.end_area = (ii - self.end / dt) * (t_f - r) / 2.0;
                    if self.end_area >= a_f {
                        self.has_ended = true;
                    }
                }
            } else if r <= t_f {
                self.end_area += (2.0 * t_f - r - r_prev) / 2.0;
                if self.end_area >= a_f {
                    self.has_ended = true;
                }
            } else if r > t_f && r_prev <= t_f {
                self.end_area += (t_f - r_prev) * (t_f - r_prev) / (r - r_prev) / 2.0;
                if self.end_area >= a_f {
                    self.has_ended = true;
                } else {
                    self.end = -1.0;
                    self.end_area = -1.0;
                    self.has_ended = false;
                }
            } else {
                self.end = -1.0;
                self.end_area = -1.0;
                self.has_ended = false;
            }

            if self.has_ended {
                if self.begin < self.end {
                    out = Some((self.begin + self.off, self.end + self.off));
                    self.begin = -1.0;
                    self.begin_area = -1.0;
                    self.has_begun = false;
                    self.end = -1.0;
                    self.end_area = -1.0;
                    self.has_ended = false;

                    // Re-run rising after label (:390-396) at the SAME ii.
                    if r >= t_r && r_prev < t_r {
                        self.begin = dt * (ii - (r - t_r) / (r - r_prev));
                        self.begin_area = (ii - self.begin / dt) * (r - t_r) / 2.0;
                        if self.begin_area >= a_r {
                            self.has_begun = true;
                        }
                    }
                } else {
                    self.begin = -1.0;
                    self.begin_area = -1.0;
                    self.has_begun = false;
                    self.end = -1.0;
                    self.end_area = -1.0;
                    self.has_ended = false;
                }
            }
        }

        self.prev = r;
        self.idx += 1;
        out
    }

    /// The tail force-emit (`:409-412`): if a segment is still open at EOS, close it at
    /// `dt * results.len()` (== `dt * idx`, since every convolved value has been fed).
    fn flush_tail(&mut self) -> Option<(f64, f64)> {
        if self.has_begun {
            let end = self.dt * self.idx as f64;
            self.has_begun = false;
            return Some((self.begin + self.off, end + self.off));
        }
        None
    }
}

/// The streaming edge-truncating convolution -- the online twin of the in-place
/// [`crate::audio::convolution_horiz_slice`] the offline
/// [`crate::tasks::segmenter::results_to_segmentation`] runs before the hysteresis
/// (`segmenter.rs:268-272`, gated `conv.len() > 1`).
///
/// The offline convolution snapshots the WHOLE result vector first, so each output
/// depends ONLY on inputs (no output feedback) -- streamable. Output `ii` reads
/// `[ii-half, ii+half]` edge-truncated at the TRUE stream ends (`audio.rs::conv_taps`):
/// a left-edge output (`ii < half`) drops the missing left taps (known from the first
/// value -- the true start), a right-edge output (`ii + half >= N`) drops the missing
/// right taps (known only at [`flush`](ConvStream::flush), with the true `N`). An
/// interior/left-edge output `ii` is FINAL once `ii + half < received` (then `ii+half <
/// N` regardless of the still-unknown `N`, so the full `2*half`-tap window applies and is
/// bit-identical to offline). Hence `received - half` values finalize mid-stream; the
/// last `half` (the right-edge tail) finalize at `flush` with `N == received`.
struct ConvStream {
    coeffs: Vec<f64>, // empty iff no convolution (offline `conv.len() <= 1`)
    half: usize,      // (coeffs.len()-1)/2, or 0 when no convolution
    ring: Vec<f64>,   // absolute index `i` at ring[i - base]
    base: usize,
    received: usize,  // total posterior scalars received
    finalized: usize, // convolved values finalized (fed to the hysteresis)
}

impl ConvStream {
    fn new(conv_coeff: Option<&[f64]>) -> ConvStream {
        // Offline gate: convolution applies iff coeffs present AND len > 1.
        let coeffs = match conv_coeff {
            Some(c) if c.len() > 1 => c.to_vec(),
            _ => Vec::new(),
        };
        let half = if coeffs.is_empty() {
            0
        } else {
            (coeffs.len() - 1) / 2
        };
        ConvStream {
            coeffs,
            half,
            ring: Vec::new(),
            base: 0,
            received: 0,
            finalized: 0,
        }
    }

    fn push(&mut self, v: f64) {
        self.ring.push(v);
        self.received += 1;
    }

    /// The convolved value at absolute index `ii`, computed against total length `n`
    /// exactly as `audio.rs::conv_taps` + `convolution_horiz_slice` do (same tap range,
    /// same ascending accumulation order -> bit-identical to offline).
    fn conv_at(&self, ii: usize, n: usize) -> f64 {
        if self.coeffs.is_empty() {
            return self.ring[ii - self.base];
        }
        let hw = self.half as isize;
        let iis = ii as isize;
        let len = n as isize;
        let (begin1, begin2) = if iis < hw {
            (-(hw - iis), (hw - iis) as usize)
        } else {
            (iis - hw, 0usize)
        };
        let end = if iis + hw < len {
            (2 * hw) as usize
        } else {
            (len - 1 - begin1) as usize
        };
        let mut acc = 0.0;
        for jj in begin2..=end {
            let src = (begin1 + jj as isize) as usize;
            acc += self.ring[src - self.base] * self.coeffs[jj];
        }
        acc
    }

    /// Finalize the newly-final (non-right-edge) convolved values `[finalized, received -
    /// half)` and return them in order. Each is computed with the FULL window (its
    /// `ii + half < received <= N`, so it is not a right-edge frame).
    fn take_ready(&mut self) -> Vec<f64> {
        let target = self.received.saturating_sub(self.half);
        if target <= self.finalized {
            return Vec::new();
        }
        let out: Vec<f64> = (self.finalized..target)
            .map(|ii| self.conv_at(ii, self.received))
            .collect();
        self.finalized = target;
        self.trim();
        out
    }

    /// EOS: finalize the right-edge tail `[finalized, received)` with the TRUE total
    /// `N == received` (the missing right taps truncate at the true stream end).
    fn flush(&mut self) -> Vec<f64> {
        let n = self.received;
        if n <= self.finalized {
            return Vec::new();
        }
        let out: Vec<f64> = (self.finalized..n).map(|ii| self.conv_at(ii, n)).collect();
        self.finalized = n;
        out
    }

    /// Drop ring elements no future finalize reads. Future output `ii >= finalized` reads
    /// sources back to `ii - half >= finalized - half` (and the flush tail reads back to
    /// `(received-half) - half`, the same floor once `finalized == received - half`).
    fn trim(&mut self) {
        let keep_from = self.finalized.saturating_sub(self.half);
        if keep_from > self.base {
            self.ring.drain(0..keep_from - self.base);
            self.base = keep_from;
        }
    }
}

/// The incremental SAD decision layer (Phase 8 Task 4): the streaming twin of the offline
/// [`crate::tasks::segmenter::results_to_segmentation`] + [`smooth_segmentation`] the fast
/// driver hands off to (`fast/driver.rs:405-419`). Consumes the FINALIZED posterior rows
/// [`StreamOverlap`] emits (f32, `k x 1` for the binary SAD net) and produces stable
/// [`EmittedSegment`]s as they settle, plus the complete [`Segmentation`] at EOS.
///
/// PIPELINE. Each posterior scalar is widened f32 -> f64 (the exact `fast/driver.rs:405`
/// seam), fed to the streaming [`ConvStream`] (the 19-tap edge-truncating convolution,
/// finalizing with a `half`-value lookahead), then the finalized convolved values drive
/// the persistent [`HystState`] (the latched hysteresis, boundaries never revised). Each
/// raw segment the hysteresis labels is FINAL (causal + latched), appended to
/// `raw_segments`.
///
/// RE-SMOOTH-AND-EMIT-STABLE-PREFIX. The 8-step [`smooth_segmentation`] is NOT streamable
/// incrementally, but it is cheap and its output is a pure function of the raw-segment
/// list + the End sentinel. On EVERY push (the T5-review consumed-frontier time-advance,
/// [`resmooth_and_emit`](Self::resmooth_and_emit)) we re-run the SHARED `smooth_segmentation`
/// on a fresh clone and emit the settled PREFIX -- segments whose end `< frontier - HOLDBACK`,
/// where `frontier = max(last_raw_boundary, consumed_frontier - dt)` advances with the
/// hysteresis's CONSUMED frontier even while no new raw segment lands. The final
/// [`flush`](Self::flush) re-runs the SAME shared code on the complete raw list seeded at the
/// true `audio_duration`, so it is BIT-IDENTICAL to the offline decision pipeline (same raw
/// segments, same `label_segment` replay, same smoothing).
///
/// THE HOLDBACK (the design's one new invariant, spec S1.3 / R2). A smoothed segment can
/// change only while later raw structure -- which lands at time `>= frontier` (the emission
/// frontier [`resmooth_and_emit`](Self::resmooth_and_emit) derives: `>= last_raw_boundary`
/// AND `>= consumed_frontier - dt`) -- can still reach it through the smoothing. Each
/// smoothing step moves/merges boundaries by
/// at most its own threshold: [`Segmentation::add_padding`] extends a Speech segment left by
/// `before` and right by `after`; [`Segmentation::suppress_short`] removes/merges a segment
/// only across its own `<= threshold` span. A future Speech segment can therefore reach an
/// earlier segment's END at most `pad_before + pad_after + suppress` to its left, and the
/// two paddings compose on one segment (the brief's warned worst case) -- chains through
/// EARLIER segments happen identically with or without the future segment (their gaps are
/// past-known), so they do not extend the future segment's reach. HOLDBACK is the
/// CONSERVATIVE sum of EVERY (clamped) smoothing threshold -- `sum(min_speech) +
/// sum(min_silence) + sum(padding)` -- which dominates any single reach and any composition
/// of them; it also keeps the mid-stream clone's End sentinel (placed at `now >= frontier`,
/// since the received frontier is `>= the consumed frontier`) at least HOLDBACK to the right
/// of the emitted region, so the tail special-casing never touches it. Param-driven at
/// construction: a config with different padding moves the
/// holdback. If it were ever too small, the prefix-consistency gate catches it as a
/// retraction (R2); too large only adds latency.
pub struct StreamDecision {
    seg_cfg: SegmenterConfig,
    class: SegClass, // SAD -> Speech (the offline driver's `SegClass::Speech`)
    dt: f64,         // time_step
    off: f64,        // time_offset
    holdback: f64,   // derived from seg_cfg at construction

    conv: ConvStream,
    hyst: HystState,

    /// Closed raw segments `(begin+off, end+off)`, append-only + final (the latched
    /// hysteresis never revises them). The offline `update_segmentation`'s `label_segment`
    /// replay list.
    raw_segments: Vec<(f64, f64)>,
    /// Count of smoothed-partition segments already emitted (a stable prefix index: the
    /// settled prefix is byte-stable across re-smooths, so this index is consistent).
    emitted_count: usize,
    finished: bool,
}

impl StreamDecision {
    /// Build from the driver config (the convolution kernel), the segmenter config (the
    /// decision thresholds + the smoothing params the holdback derives from), and the
    /// per-file `time_step`/`time_offset` the offline overlap branch computes
    /// (`fast/driver.rs:346-347`).
    ///
    /// NOTE ON THE SIGNATURE: the Task-4 brief sketches `new(driver_cfg, seg_cfg,
    /// time_step)`, but the hysteresis is defined against `time_offset` too (the offline
    /// `update_segmentation`'s `off`), so it is a required argument here. The T5 session
    /// computes both scalars exactly as the driver does and passes them.
    pub fn new(
        driver_cfg: DriverConfig,
        seg_cfg: SegmenterConfig,
        time_step: f64,
        time_offset: f64,
    ) -> StreamDecision {
        let holdback = seg_cfg.min_speech.iter().sum::<f64>()
            + seg_cfg.min_silence.iter().sum::<f64>()
            + seg_cfg.padding.iter().sum::<f64>();
        let conv = ConvStream::new(driver_cfg.conv_coeff.as_deref());
        let hyst = HystState::new(&seg_cfg, time_offset, time_step);
        StreamDecision {
            seg_cfg,
            class: SegClass::Speech,
            dt: time_step,
            off: time_offset,
            holdback,
            conv,
            hyst,
            raw_segments: Vec::new(),
            emitted_count: 0,
            finished: false,
        }
    }

    /// The derived smoothing holdback in seconds (see the type docs). Exposed for the
    /// latency accounting + the param-driven-derivation gate.
    pub fn holdback(&self) -> f64 {
        self.holdback
    }

    /// The closed raw-segment list `(begin+off, end+off)` accumulated so far (the offline
    /// `update_segmentation_raw` output; the flush tail is included after `flush`).
    pub fn raw_segments(&self) -> &[(f64, f64)] {
        &self.raw_segments
    }

    /// The posterior-row frontier time: `received_rows * dt + off`. Used as the mid-stream
    /// clone's End sentinel and as `emitted_at_audio_s`.
    fn current_audio_time(&self) -> f64 {
        self.conv.received as f64 * self.dt + self.off
    }

    /// Push a chunk of finalized posterior rows (`k x 1`, the binary SAD posterior column,
    /// f32) and return the segments that FINALIZED this call. Buffers the (widened)
    /// scalars through the streaming convolution, advances the latched hysteresis over the
    /// newly-final convolved values, then re-smooths and emits the newly-settled prefix on
    /// EVERY push (the consumed-frontier time-advance -- a settled segment can finalize as
    /// the frontier advances through silence, without waiting for a new raw segment).
    pub fn push_rows(&mut self, rows: &FastMatrix) -> Vec<EmittedSegment> {
        // The decision layer reads ONLY column 0 (the binary SAD posterior); a
        // multi-column posterior would silently drop columns and the offline
        // `results.len()` == rows*cols tail quirk would differ (T4 report concern 3,
        // T4-review rider). Pin the mono/binary-SAD assumption in the assert style of
        // the sibling StreamOverlap/StreamFrontEnd debug_asserts.
        debug_assert_eq!(
            rows.cols, 1,
            "StreamDecision::push_rows expects the binary SAD posterior column (output_size 1)"
        );
        for r in 0..rows.rows {
            self.conv.push(rows.get(r, 0) as f64);
        }
        let ready = self.conv.take_ready();
        for v in ready {
            if let Some(seg) = self.hyst.advance(v) {
                self.raw_segments.push(seg);
            }
        }
        // TIME-ADVANCE EMISSION: attempt an emit on EVERY push, not only when a new raw
        // segment lands. During a long inter-speech silence no raw segment closes, yet the
        // hysteresis keeps consuming convolved values -- so the CONSUMED frontier advances and
        // a settled earlier segment becomes emittable once that frontier (minus holdback)
        // passes it. This delivers bounded-latency output instead of waiting for the next
        // triggering raw segment. `resmooth_and_emit` is a no-op until at least one raw
        // segment exists and never re-emits an already-emitted segment (the `emitted_count`
        // prefix cursor), so calling it unconditionally is safe + idempotent.
        self.resmooth_and_emit()
    }

    /// EOS: finalize the right-edge convolution tail + the open hysteresis segment, then
    /// build the COMPLETE segmentation (seeded at the true `audio_duration`) via the SHARED
    /// offline smoothing and emit every remaining settled segment. Returns the final
    /// emissions + the complete [`Segmentation`] (BIT-IDENTICAL to the offline decision
    /// pipeline on the same rows). Idempotent (a second call returns no emissions + the
    /// same segmentation).
    pub fn flush(&mut self, audio_duration: f64) -> (Vec<EmittedSegment>, Segmentation) {
        if !self.finished {
            self.finished = true;
            for v in self.conv.flush() {
                if let Some(seg) = self.hyst.advance(v) {
                    self.raw_segments.push(seg);
                }
            }
            if let Some(seg) = self.hyst.flush_tail() {
                self.raw_segments.push(seg);
            }
        }

        // The complete segmentation: seed at the TRUE audio_duration, replay every raw
        // segment, smooth once -- identical to `results_to_segmentation` + smoothing.
        let final_seg = self.build_smoothed(audio_duration);

        // Emit every remaining segment (no holdback at EOS -- everything is final).
        let now = self.current_audio_time();
        let emitted = self.collect_prefix(&final_seg, f64::INFINITY, now);
        (emitted, final_seg)
    }

    /// Build a smoothed [`Segmentation`] seeded at `audio_dur` from the current raw
    /// segments -- the SHARED offline path (`Segmentation::new` -> `label_segment` replay
    /// -> `smooth_segmentation`), run read-only on a clone.
    fn build_smoothed(&self, audio_dur: f64) -> Segmentation {
        let mut seg = Segmentation::new(audio_dur);
        for &(b, e) in &self.raw_segments {
            seg.label_segment(b, e, self.class);
        }
        smooth_segmentation(&mut seg, &self.seg_cfg);
        seg
    }

    /// Re-smooth the current raw segments (clone seeded at the posterior frontier) and emit
    /// the newly-settled prefix -- segments whose end `< frontier - holdback`, where
    /// `frontier` is the CONSUMED-frontier trigger derived below.
    fn resmooth_and_emit(&mut self) -> Vec<EmittedSegment> {
        let last_rb = match self.raw_segments.last() {
            Some(&(_, e)) => e,
            None => return Vec::new(),
        };
        let now = self.current_audio_time();
        // The clone's End sentinel: past all processed structure (>= last_rb AND >= the
        // emission frontier -- `now >= frontier` since received >= finalized), so the emitted
        // region stays >= holdback to its left and the tail special-casing cannot reach it.
        let mid_dur = now.max(last_rb);
        let seg = self.build_smoothed(mid_dur);

        // THE EMISSION FRONTIER (the time-advance trigger). A smoothed segment can still be
        // revised only by FUTURE raw structure, which lands no earlier than
        //   frontier = max(last_rb, hyst_frontier - dt)
        // -- `last_rb` because raw segments close in ascending order (the next one begins
        // after this one ends), and `hyst_frontier - dt` because the CONSUMED frontier is
        // `hyst_frontier = conv.finalized*dt + off` (the hysteresis has advanced through
        // exactly `conv.finalized` convolved values; `StreamDecision` owns both `conv` and
        // `hyst`, so no session threading is needed) and the earliest a not-yet-consumed
        // value can interpolate a boundary back to is one grid step (`dt`) before it.
        //
        // THE SAFETY ARGUMENT: the T4 holdback-dominance proof (see the type docs) licenses
        // emitting any segment whose smoothed form cannot be reached by raw structure at or
        // beyond the frontier -- and the consumed frontier minus one grid step is the earliest
        // a future raw boundary can interpolate back to, so every segment ending strictly
        // before `frontier - holdback` is final. (Using the RECEIVED frontier `now` here
        // would be UNSAFE: `now` runs ahead of the consumed frontier by the convolution
        // lookahead + a grid step, so it could license emitting a segment a still-pending
        // convolved value can still move.) The prefix-consistency gate legs are the safety
        // net -- any too-large frontier surfaces there as a retraction.
        let hyst_frontier = self.conv.finalized as f64 * self.dt + self.off;
        let frontier = last_rb.max(hyst_frontier - self.dt);
        let threshold = frontier - self.holdback;
        self.collect_prefix(&seg, threshold, now)
    }

    /// Emit the contiguous prefix of `seg`'s partition (from `emitted_count`) whose
    /// segments END strictly before `threshold`, stamping `emitted_at`. Advances
    /// `emitted_count`. Segment `i` is `[segs[i].begin, segs[i+1].begin)` of type
    /// `segs[i].ty`; its end is `segs[i+1].begin`.
    fn collect_prefix(
        &mut self,
        seg: &Segmentation,
        threshold: f64,
        emitted_at: f64,
    ) -> Vec<EmittedSegment> {
        let segs = seg.segments();
        let mut out = Vec::new();
        let mut i = self.emitted_count;
        while i + 1 < segs.len() && segs[i + 1].begin < threshold {
            out.push(EmittedSegment {
                begin_s: segs[i].begin,
                end_s: segs[i + 1].begin,
                class: segs[i].ty,
                emitted_at_audio_s: emitted_at,
            });
            i += 1;
        }
        self.emitted_count = i;
        out
    }
}

// ===========================================================================
// StreamingSession (Phase 8 Task 5): the composed SAD streaming session + gate.
// ===========================================================================

/// The chunked-input SAD streaming session (Phase 8 Task 5): the online twin of the
/// offline fast algo-3 driver ([`crate::fast::driver::FastSpectralSegmenter`]), composing
/// the three landed layers -- [`StreamFrontEnd`] (T2: gain/preemph-carry/dither-index plus
/// per-range feature extraction), [`StreamOverlap`] (T3: the windowed BLSTM forward with
/// bounded lookahead), and [`StreamDecision`] (T4: the incremental hysteresis, convolution,
/// re-smooth-and-emit) -- over the phase-7 [`FastPipeline`] + [`FastBlstm`] kernels,
/// UNCHANGED. The session only WIRES the layers (feature rows -> type-1 norm -> overlap ->
/// decision) and owns the sample counter + latency accounting; every bit-equal contract is
/// already carried by T2/T3/T4.
///
/// MONO-FIRST, FROZEN-STATS (spec S1.2). The session processes ONE channel via
/// [`push`](Self::push), under FROZEN normalization: a config-fixed `Audio_fixed_gain`
/// (front-end-side, per sample, `audio.rs::apply_fixed_gain`) + the pack-carried type-1
/// input statistics (a per-column affine, applied per feature-row front-end-side --
/// [`super::nn::external_normalize_f32`]). Both whole-file statistics that block causality
/// (the `(2*RMS+max)/2` audio normalization + the type -1 per-sequence self-normalization,
/// spec S0) are thereby replaced by frozen constants, so nothing needs whole-file
/// lookahead. [`new`](Self::new) typed-bails everything outside this contract.
///
/// BIT-EQUAL TO OFFLINE-FROZEN (THE PHASE GATE, spec S1.6/S2). At any chunking,
/// [`finish`](Self::finish)'s [`Segmentation`] AND the (test-observable) posterior history
/// equal the offline fast bag run on the SAME audio under the SAME frozen config,
/// BIT-IDENTICAL. The three layers each carry that contract (T2 feature rows, T3 overlap
/// output incl. the `0/0=NaN` uncovered rows, T4 decision), and streaming changes only
/// TIMING, never arithmetic.
///
/// SIGNATURE NOTE (the T4-precedent correction). The Task-5 brief sketches `new(map)`, but
/// the offline path threads `audio.sample_rate` into `SpectralParams::derive` +
/// `getBLSTMParam` and the source channel count into the per-channel loop; neither lives in
/// the config. `new` therefore takes `rate` and `channels` too (the `speech stream` CLI /
/// PyO3 binding read both from the wav header / caller before pushing samples). `channels`
/// is validated `== 1` and then unused (the mono-first bail); it exists so a multi-channel
/// source fails LOUDLY at construction rather than silently dropping the cross-channel
/// `result_vec` seeding (deferred, spec S0/S1.2).
///
/// LATENCY (spec S1.8). Every emission is stamped with the session's audio-time-pushed
/// clock (`(total_pushed-1)/rate`) and its per-emission lag (`emitted_at - end_s`) is
/// tracked (running max/mean, O(1) memory). With the T5-review consumed-frontier time-advance
/// ([`StreamDecision::push_rows`]), a SPEECH segment is delivered within the DERIVED structural
/// budget [`derived_latency_bound_s`](Self::derived_latency_bound_s) = front-end reach + NN
/// window finalization + convolution half-width + the smoothing holdback -- it emits during
/// the FOLLOWING silence, no longer waiting for the next raw segment. An OTHER (silence)
/// segment still waits for the following speech to COMMIT (its right boundary is that speech's
/// onset, latched only at the falling edge), so its lag is that speech's DURATION + the forward
/// pipeline delay -- the data-dependent AREA term pinned per-class on the fixtures
/// (`tests/phase8_gate.rs`).
pub struct StreamingSession {
    front: StreamFrontEnd,
    pipeline: FastPipeline,
    overlap: StreamOverlap,
    net: FastBlstm,
    decision: StreamDecision,

    /// The pack-carried type-1 normalize tail (narrowed f32), applied per feature-row
    /// front-end-side (the documented T3/T4 seam: type 1 is a per-column affine, so it
    /// commutes with row-streaming and is bit-identical to the offline whole-sequence
    /// application).
    norm_mean: Vec<f32>,
    norm_std: Vec<f32>,

    rate: f64,
    total_pushed: usize, // absolute sample count == the audio-time clock source
    finished: bool,

    // Latency budget (config-derived once at construction; the pins read these back).
    feature_reach_s: f64, // (reach*shift_frames + half_window)/rate
    nn_window_s: f64,     // 2*window_size_feature * spectrum_shift_sec
    conv_delay_s: f64,    // conv_half * time_step

    // Running per-emission lag stats (O(1) memory -- the low-memory streaming goal).
    lag_max: f64,
    lag_sum: f64,
    lag_count: usize,

    /// Posterior history (the StreamOverlap output, `output_size`-wide f32) -- the gate's
    /// bit-equivalence observable, mirroring the offline `last_result_rows`. Accumulated
    /// ONLY under `test-support` (it grows unbounded; production streaming never keeps it,
    /// per the phase's low-memory goal), the SAME pattern as `BagOfProcessors`'
    /// `last_audio_abs_sum`.
    #[cfg(feature = "test-support")]
    posterior_history: Vec<f32>,
}

impl StreamingSession {
    /// Build the session from the SAME legacy config `map` the bag consumes, the stream
    /// `rate`, and the source `channels` count. Validates the streaming contract (each bail
    /// pinned by `tests/phase8_gate.rs::validation_bails`): algo 3, mono, `Audio_fixed_gain`
    /// present, `InputNormalizationType 1`, and the overlap windowing (plain/truncate bail).
    /// Loads the frozen net once and builds the phase-7 pipeline/BLSTM + the three streaming
    /// layers.
    pub fn new(
        map: &IndexMap<String, String>,
        rate: f64,
        channels: usize,
    ) -> Result<StreamingSession> {
        // --- Validation bails (each pinned) ---
        let algo = map
            .get("Algo_choice")
            .ok_or_else(|| anyhow!("streaming: missing Algo_choice"))?
            .trim()
            .parse::<i32>()
            .map_err(|e| anyhow!("streaming: Algo_choice parse: {e}"))?;
        if algo != 3 {
            bail!("streaming: only algo 3 (spectral SAD) is supported (got {algo})");
        }
        if channels != 1 {
            bail!(
                "streaming: mono only (got {channels} channels); multi-channel streaming (the \
                 cross-channel result_vec seeding) is deferred (spec S0/S1.2)"
            );
        }
        let fixed_gain = match map.get("Audio_fixed_gain") {
            Some(s) => s
                .trim()
                .parse::<f64>()
                .map_err(|e| anyhow!("streaming: Audio_fixed_gain parse: {e}"))?,
            None => bail!(
                "streaming requires Audio_fixed_gain (frozen-norm mode; the whole-file \
                 (2*RMS+max)/2 audio normalization is not streamable)"
            ),
        };
        let bc = BlstmConfig::from_legacy(map, "BLSTM")?;
        if bc.input_normalization_type != 1 {
            bail!(
                "streaming requires BLSTM_InputNormalizationType 1 (pack-carried frozen stats); \
                 got {} (type -1 self-normalization needs whole-sequence lookahead)",
                bc.input_normalization_type
            );
        }

        // --- Config surfaces (shared with the offline fast driver) ---
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let spec = build_aligned_spec(map, "BLSTM")?;

        // --- Frozen net (loaded once) ---
        let weights_file = map
            .get("BLSTM_weightsFile")
            .map(String::as_str)
            .unwrap_or("");
        if weights_file.is_empty() {
            bail!("streaming: BLSTM_weightsFile is empty (the frozen net must be loaded once)");
        }
        let flat = crate::io::binary::read_weight_vector(std::path::Path::new(weights_file))?;
        let net = FastBlstm::from_flat(&spec, &flat)?;

        // --- Framing + pipeline (built once, at `rate`) ---
        let params = SpectralParams::derive(&feature_cfg, rate);
        let pipeline = FastPipeline::new(&params, &feature_cfg, rate)?;

        let ssr = spec.lstm_subsampling.iter().product::<usize>()
            * spec.output_subsampling.iter().product::<usize>();
        // `params.shift_frames` == round(shift_sec*rate) == the quantized spectrum shift in
        // frames (`ssif`); `params.shift_sec` == ssif/rate (the offline driver's
        // `spectrum_shift_sec` after its own re-quantization, tasks/sad.rs:290-291).
        let ssif = params.shift_frames;
        let spectrum_shift_sec = params.shift_sec;

        // getBLSTMParam window/shift. `frame_count` is passed 0: only `real_vec_size`
        // depends on it, and StreamOverlap derives its OWN output-row count (`total/ssr`)
        // at flush -- the session never needs `real_vec_size`. window_size/window_shift/
        // no_overlap are frame_count-independent.
        let mut window_shift_sec = driver_cfg.window_shift_sec;
        let (window_size, window_shift, no_overlap, _real_vec_size) = get_blstm_param(
            driver_cfg.window_size_sec,
            &mut window_shift_sec,
            rate,
            ssif,
            ssr,
            &spec.lstm_subsampling,
            &spec.output_subsampling,
            0,
        );
        if window_size == 0 {
            bail!(
                "streaming: the plain (non-windowed) forward is unsupported (BLSTM_window \
                 resolves window_size 0); the gate config uses windowed overlap"
            );
        }
        if no_overlap {
            bail!(
                "streaming: the truncate (non-overlap) windowing is unsupported (window_shift \
                 resolves < 1); the gate config uses overlap"
            );
        }

        // timeStep/timeOffset, OVERLAP branch (fast/driver.rs:346-347) -- passed to the
        // decision layer exactly as the offline driver computes them.
        let time_step = spectrum_shift_sec * ssr as f64;
        let time_offset = time_step / 2.0 - spectrum_shift_sec / 2.0;

        // preemph/noise gates (the offline driver's, fast/driver.rs:296-301). Both no-ops on
        // tier2 (preemph -0.97 <= 0, noise_seed -3 <= 0). The front-end gates preemph on
        // `> 0.0` internally (passed the raw ratio); noise maps `noise_seed > 0 ?
        // noise_ratio : 0.0`. THE TYPED BAIL (T2-review rider; T5-review promoted it from an
        // assert! to a bail! for consistency with the four sibling construction bails): a
        // seeding config (`noise_seed > 0`) with a NEGATIVE `noise_ratio` would make the
        // offline `apply_noise(ratio)` (unconditional on the seed) and the front-end's
        // `noise_magnitude > 0.0` gate DISAGREE in shape -- refused loudly here.
        let preemph = feature_cfg.preemph_ratio;
        let noise_magnitude = if feature_cfg.noise_seed > 0 {
            if feature_cfg.noise_ratio < 0.0 {
                bail!(
                    "streaming: BLSTM_noise_ratio must be >= 0 when seeding (noise_seed > 0); \
                     the offline apply_noise / front-end noise-gate SHAPES only agree for a \
                     non-negative magnitude"
                );
            }
            feature_cfg.noise_ratio
        } else {
            0.0
        };

        let front = StreamFrontEnd::new(
            &params,
            &feature_cfg,
            rate,
            fixed_gain,
            noise_magnitude,
            preemph,
        )?;
        let output_size = net.output_size();
        // Binary-SAD only: the decision layer reads posterior column 0 and the offline
        // `results.len()` == rows*cols tail quirk would differ for a wider posterior. Refuse
        // loudly at construction (the sibling of `StreamDecision::push_rows`' cols==1
        // debug_assert, promoted to a construction bail per the T5 review).
        if output_size != 1 {
            bail!(
                "streaming: only the binary SAD net (output_size 1) is supported; got \
                 {output_size} (the decision layer reads column 0 only)"
            );
        }
        let overlap = StreamOverlap::new(window_size, window_shift, ssr, output_size);

        // Derived latency components (config-derived; `derived_latency_bound_s` sums these +
        // holdback). See that method + the module-level LATENCY note.
        let reach = if feature_cfg.deltas_nb > 0 {
            feature_cfg.deltas_nb as usize
                + if feature_cfg.dd_nb > 0 {
                    feature_cfg.dd_nb as usize
                } else {
                    0
                }
        } else {
            0
        };
        let half_window = params.window_size / 2;
        let feature_reach_s = (reach * ssif + half_window) as f64 / rate;
        let nn_window_s = 2.0 * window_size as f64 * spectrum_shift_sec;
        let conv_half = driver_cfg
            .conv_coeff
            .as_deref()
            .map_or(0, |c| if c.len() > 1 { (c.len() - 1) / 2 } else { 0 });
        let conv_delay_s = conv_half as f64 * time_step;

        let norm_mean = net.normalize_mean().to_vec();
        let norm_std = net.normalize_std().to_vec();
        let decision = StreamDecision::new(driver_cfg, seg_cfg, time_step, time_offset);

        Ok(StreamingSession {
            front,
            pipeline,
            overlap,
            net,
            decision,
            norm_mean,
            norm_std,
            rate,
            total_pushed: 0,
            finished: false,
            feature_reach_s,
            nn_window_s,
            conv_delay_s,
            lag_max: f64::NEG_INFINITY,
            lag_sum: 0.0,
            lag_count: 0,
            #[cfg(feature = "test-support")]
            posterior_history: Vec::new(),
        })
    }

    /// Push a chunk of RAW mono f32 samples (`i16/32768`-scaled, PRE-gain -- the front-end
    /// applies the frozen gain/preemph/noise per sample) and return the segments FINALIZED
    /// by this call, each stamped with the session's audio-time-pushed clock. Buffers the
    /// samples, extracts + type-1-normalizes any newly-final feature rows, fires the
    /// newly-fireable overlap windows, and drives the incremental decision layer.
    pub fn push(&mut self, samples: &[f32]) -> Vec<EmittedSegment> {
        if self.finished {
            return Vec::new();
        }
        self.front.push(samples);
        self.total_pushed += samples.len();
        let mut rows = self.front.take_ready_rows(&mut self.pipeline);
        let now = self.current_audio_time();
        let emitted = self.step_rows(&mut rows);
        self.stamp_and_track(emitted, now)
    }

    /// EOS: flush the front-end tail (right-edge feature rows), fire the clamped tail
    /// overlap windows, and flush the decision layer -- returning the final emissions + the
    /// COMPLETE [`Segmentation`] (BIT-IDENTICAL to the offline fast bag run on the same
    /// audio, spec S1.6). Idempotent (a second call returns no emissions + the same
    /// segmentation).
    pub fn finish(&mut self) -> (Vec<EmittedSegment>, Segmentation) {
        let audio_duration = self.current_audio_time();
        if self.finished {
            // Idempotent: the layers' own flush guards make this a re-derivation only.
            let (_e, seg) = self.decision.flush(audio_duration);
            return (Vec::new(), seg);
        }
        self.finished = true;

        // Front-end EOS -> the right-edge tail feature rows (delta reach clamps at the TRUE
        // last frame); type-1-normalize them front-end-side, exactly like mid-stream.
        let mut tail = self.front.flush(&mut self.pipeline);
        if tail.rows > 0 {
            external_normalize_f32(&mut tail, &self.norm_mean, &self.norm_std);
        }
        // These tail rows make the last UNCLAMPED windows fireable (push_rows), then the
        // overlap flush fires the CLAMPED tail windows -- contiguous posterior-row ranges,
        // fed to the decision layer in order.
        let p1 = self.overlap.push_rows(&tail, &mut self.net);
        let p2 = self.overlap.flush(&mut self.net);
        let mut emitted = self.consume_posteriors(&p1);
        emitted.extend(self.consume_posteriors(&p2));

        let (tail_emitted, seg) = self.decision.flush(audio_duration);
        emitted.extend(tail_emitted);

        let stamped = self.stamp_and_track(emitted, audio_duration);
        (stamped, seg)
    }

    /// Type-1-normalize + overlap + decision for a batch of feature rows (the shared push /
    /// finish body, minus the front-end fetch). Returns UNSTAMPED emissions.
    fn step_rows(&mut self, rows: &mut FastMatrix) -> Vec<EmittedSegment> {
        if rows.rows > 0 {
            // Type-1 is a per-column affine, so per-row == whole-sequence (the documented
            // T3/T4 seam) -- bit-identical to the offline `external_normalize_f32` over the
            // whole input sequence.
            external_normalize_f32(rows, &self.norm_mean, &self.norm_std);
        }
        let posts = self.overlap.push_rows(rows, &mut self.net);
        self.consume_posteriors(&posts)
    }

    /// Record the posterior rows (test-support observable) and drive the decision layer.
    fn consume_posteriors(&mut self, posts: &FastMatrix) -> Vec<EmittedSegment> {
        #[cfg(feature = "test-support")]
        self.posterior_history.extend_from_slice(&posts.data);
        self.decision.push_rows(posts)
    }

    /// Stamp each emission with `now` (the session audio-time-pushed clock) and fold its lag
    /// (`now - end_s`) into the running max/mean.
    fn stamp_and_track(
        &mut self,
        mut emitted: Vec<EmittedSegment>,
        now: f64,
    ) -> Vec<EmittedSegment> {
        for seg in emitted.iter_mut() {
            seg.emitted_at_audio_s = now;
            let lag = now - seg.end_s;
            if lag > self.lag_max {
                self.lag_max = lag;
            }
            self.lag_sum += lag;
            self.lag_count += 1;
        }
        emitted
    }

    /// The audio-time of the last-pushed sample (`(total_pushed-1)/rate`), the session's
    /// latency clock -- consistent with the offline `Segmentation` seed
    /// `dur = (frames-1)/rate`.
    fn current_audio_time(&self) -> f64 {
        if self.total_pushed == 0 {
            0.0
        } else {
            (self.total_pushed - 1) as f64 / self.rate
        }
    }

    /// The derived STRUCTURAL latency bound in seconds (spec S1.8, the T4 spec correction):
    /// front-end reach + NN window finalization + convolution half-width + the smoothing
    /// holdback -- ALL config-derived (never a literal). The measured per-emission lag is
    /// bounded by this plus a data-dependent AREA term (`tests/phase8_gate.rs`).
    pub fn derived_latency_bound_s(&self) -> f64 {
        self.feature_reach_s + self.nn_window_s + self.conv_delay_s + self.decision.holdback()
    }

    /// The feature-extraction reach component `(reach*shift_frames + half_window)/rate`.
    pub fn feature_reach_s(&self) -> f64 {
        self.feature_reach_s
    }
    /// The NN-window-finalization component `2*window_size_feature * spectrum_shift_sec`
    /// (the full-window lookahead the overlap incurs before an output row finalizes).
    pub fn nn_window_s(&self) -> f64 {
        self.nn_window_s
    }
    /// The convolution half-width component `conv_half * time_step`.
    pub fn conv_delay_s(&self) -> f64 {
        self.conv_delay_s
    }
    /// The smoothing holdback component (`StreamDecision::holdback`).
    pub fn holdback_s(&self) -> f64 {
        self.decision.holdback()
    }

    /// The maximum per-emission lag observed so far (`0.0` if nothing has been emitted).
    pub fn max_lag_s(&self) -> f64 {
        if self.lag_count == 0 {
            0.0
        } else {
            self.lag_max
        }
    }
    /// The mean per-emission lag observed so far (`0.0` if nothing has been emitted).
    pub fn mean_lag_s(&self) -> f64 {
        if self.lag_count == 0 {
            0.0
        } else {
            self.lag_sum / self.lag_count as f64
        }
    }
    /// The number of emissions tracked so far.
    pub fn emission_count(&self) -> usize {
        self.lag_count
    }

    /// Test hook: the accumulated posterior history (the StreamOverlap output, one
    /// `output_size`-wide row per posterior row, incl. the `NaN` uncovered rows) -- the
    /// gate's bit-equivalence observable vs the offline `last_result_rows`.
    #[cfg(feature = "test-support")]
    pub fn posterior_history(&self) -> &[f32] {
        &self.posterior_history
    }
}
