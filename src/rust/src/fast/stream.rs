//! `fast::stream` -- the SAD streaming path (Phase 8 Tasks 2-5, Phase 9 Task 7).
//!
//! PHASE 9 TASK 7 (spec S5.1-S5.6): the module now carries TWO NN stages behind one
//! [`StreamingSession`]. [`StreamOverlap`] (phase 8) streams the windowed bidirectional
//! peephole-LSTM twin; [`StreamCausal`] (this task) streams a CAUSAL stack
//! (`slstm`/`mamba` + `Direction forward`) ROW BY ROW, with no window and no lookahead.
//! [`StreamFrontEnd`], [`StreamDecision`] and the frozen-gain plumbing are SHARED and
//! UNCHANGED -- the causal arm swaps exactly one layer of the four-layer composition:
//!
//! ```text
//!   samples -> StreamFrontEnd -> type-1 norm (or nothing) -> | StreamOverlap |  -> StreamDecision
//!              (shared, T2)      (shared, widened to type 0) | StreamCausal  |     (shared, T4)
//! ```
//!
//! `StreamingSession::new` picks the arm off the config's `Cell_Type` x `Direction` pair
//! through `fast::driver`'s own `classify_fast_shape` choke point (spec S5.2), so
//! `push`/`finish`, the `speech stream` CLI and the PyO3 `StreamingSession` gain the causal
//! mode with ZERO new surface.
//!
//! THE SUB-SAMPLING DEVIATION -- SUPPORTED, NOT BAILED. The PLAN's Task-7 brief listed
//! `sub_sampling 1` among the S5.3 causal-streaming requirements; the spec's own S5.3 is
//! SILENT on sub-sampling (the plan text overshot it -- the Task-6 adjudication, recorded
//! in the ledger). The requirement is deliberately NOT
//! enforced, mirroring Task 6's adjudicated deviation on the offline side: the phase's own
//! causal SAD arm (`configs/training/lre_sad.toml` under the `Direction forward` overlay)
//! runs `lstm_sub_sampling = "4,1"`, so a sub-sampling bail would make the arm's own
//! checkpoint unstreamable and take the Task-9 corpus leg with it. [`StreamCausal`]
//! instead reproduces `Network::drive`'s decimation by BUFFERING (see its docs), which is
//! the offline semantics rather than new logic, and the bit-equal gate
//! (`tests/phase9_stream_causal.rs`) enforces that -- including the trailing `T mod R`
//! row drop. The three remaining S5.3 requirements (window 0, MONO, `Audio_fixed_gain`)
//! are enforced, and the normalization set is `{0, 1}` rather than phase 8's `{1}`: type 0
//! is the exact path's no-op arm, causality-trivial because there is no whole-file
//! statistic to freeze, and it is what the committed phase-9 causal fixtures carry.
//!
//! THE DECISION LAYER'S REPLAY IS CUT (the interstitial incremental-resmooth fix).
//! [`StreamDecision`] re-smooths on every push, and its input -- the raw-segment list --
//! is append-only, so the phase-8/9 form was `O(#segments)` per push (quadratic
//! cumulative, and `O(S^2)` within one replay through `label_segment`'s walk). It now
//! replays only a BOUNDED RETAINED WINDOW of raw segments and splices the result onto a
//! FROZEN prefix, with the output BIT-IDENTICAL to the full replay at every push. The
//! derivation lives on [`StreamDecision`] itself ("THE CUT"), where it EXTENDS the
//! holdback/frontier argument it depends on rather than restating it: the emission
//! frontier already proves nothing FUTURE can change the frozen prefix (rightward), and
//! the new half proves that truncating the replay's left context cannot change a single
//! bit at or right of the cut (leftward), with the margin expressed in holdback units.
//! The equivalence oracle (`tests/stream_incremental_resmooth.rs`) runs the two paths side
//! by side on randomized long streams -- the committed gates were BLIND to the phase-9 T9
//! bug, so the cut does not lean on them.
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
//! window `[cf*shift - hw, cf*shift + hw]` (`hw = window_size/2`, in SAMPLES -- the framing
//! half-window, 512 on tier2; DISTINCT from the NN half-window, which is a count of FEATURE
//! frames -- 163 on tier2 -- entering the latency budget as `nn_window`, not this `hw`),
//! left-clamped to 0 and right-clamped to the last sample. A frame is FINAL once its right edge sample
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
use crate::legacy_config::get_f64_opt;
use crate::nn::blstm::{BlstmConfig, file_pack_head};
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmentation::{SegClass, Segment, Segmentation};
use crate::tasks::segmenter::{DriverConfig, SegmenterConfig, smooth_segmentation};

use super::cells::{CellGeometry, FastCausalNet, FastCellState};
use super::driver::{FastNetShape, build_spec_aligned_to, classify_fast_shape, exact_pack_len};
use super::nn::{
    DenseRowChain, FastBlstm, FastMatrix, external_normalize_f32, window_begin, window_end,
};
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
// StreamCausal (Phase 9 Task 7): the per-row causal NN stage.
// ===========================================================================

/// The streaming CAUSAL engine (Phase 9 Task 7, spec S5.1): the online twin of
/// [`FastCausalNet::feed_forward`], and the layer that REPLACES [`StreamOverlap`] in the
/// four-layer composition when the config selects a causal net. It consumes the
/// (already-normalized) feature-row stream [`StreamFrontEnd`] emits and produces posterior
/// rows with NO lookahead at all: the `nn_window` term that dominates the phase-8 latency
/// budget simply does not exist here, because a causal cell needs no future frames.
///
/// BIT-EQUAL CONTRACT (the task gate). The concatenation of every
/// [`push_rows`](Self::push_rows) return equals the offline `FastCausalNet::feed_forward`
/// on the WHOLE sequence, BIT-IDENTICAL, at any row-push granularity. THREE facts carry
/// it, and each is a shared kernel rather than a re-derivation:
///
///  1. THE CELL STEP. `fast::cells`' `step(x, &mut state, out)` IS the offline kernel
///     (`run_sequence` is literally a loop over it), and a carried state reproduces an
///     unsplit run bit-for-bit -- pinned by `cells.rs`'s `*_split_state_reproduces_the_
///     unsplit_run` legs, which exist precisely as this task's precondition. Streaming
///     keeps ONE state per cell layer alive for the whole stream, so the state sequence
///     is the offline one.
///  2. THE SUB-SAMPLE BUFFERING. `Network::drive`'s decimation (`sub_sample_into`:
///     `R x C -> 1 x R*C`, row `jj*R+kk` into output columns `[kk*C, (kk+1)*C)`,
///     trailing `T mod R` rows DROPPED) is reproduced by accumulating `R` arriving rows
///     into one flat buffer IN ARRIVAL ORDER -- which is exactly that column layout --
///     and never completing a trailing partial group. Offline processes the sequence
///     layer-by-layer and streaming interleaves the layers, but each layer still sees
///     the identical input rows in the identical order, and a cell is a pure sequential
///     recurrence, so the interleaving is value-neutral.
///  3. THE DENSE MLP. The output stack runs [`DenseRowChain`] -- the SAME per-row chain
///     `FastCausalNet::feed_forward` was moved onto in this task. See that type's docs
///     for the measured reason (faer's batched projection is NOT row-count-independent
///     at `output_size > 1`, so the offline batched form was structurally unstreamable
///     for the real `24,12,1` causal SAD arm).
///
/// THE CROP GATE. `feed_forward` narrows its net input to `LSTMNeuronNb[0]` columns iff
/// `lstm_subsampling[0] > 1 && LSTMNeuronNb[0] < input.cols`. That predicate depends only
/// on the feature WIDTH, which is a stream-wide constant, so it is resolved once on the
/// first pushed batch and applied per row thereafter.
///
/// EOS. There is nothing to flush: a partially-filled sub-sample group is DROPPED, which
/// is the offline `T mod R` truncation. [`flush`](Self::flush) exists to say so
/// explicitly and returns an empty matrix.
///
/// ROW-COUNT IDENTITY (why no zero-padded tail is needed). The offline fast driver copies
/// the net's posteriors into a `real_vec_size`-row buffer that it zero-fills first, so a
/// SHORT net output would leave zero rows in the driver's result vector that streaming
/// never produces. Those two counts coincide STRUCTURALLY, not by luck:
/// `get_blstm_param`'s `vec_size` is `ceil(frame_count / spectrum_shift_in_frames)`, which
/// is exactly [`FastPipeline::frame_count`] -- the feature-row count the front-end emits --
/// and its `real_vec_size` then divides by each `lstm`/`output` sub-sampling ratio in the
/// same order and with the same integer truncation this chain does. The gate's
/// posterior-history length equality is the standing proof.
pub struct StreamCausal {
    net: FastCausalNet,
    /// One live carried state per cell layer (never reset mid-stream -- the whole point).
    states: Vec<FastCellState>,
    crop: usize,         // LSTMNeuronNb[0], the crop-gate threshold
    in_cols: usize,      // the CROPPED net-input width, resolved on the first pushed batch
    in_cols_seen: usize, // the RAW arriving width (0 = unresolved); the stability check

    // Per-cell-layer sub-sampling buffers (the `sub_sample_into` row concatenation).
    pend: Vec<Vec<f32>>,
    fill: Vec<usize>,
    outs: Vec<Vec<f32>>,

    dense: DenseRowChain,
    output_size: usize,
    emitted: usize,
}

impl StreamCausal {
    /// Take ownership of the frozen causal net (loaded once, like the phase-8 session's
    /// [`FastBlstm`]) and seed one seq-start state per cell layer.
    pub fn new(net: FastCausalNet) -> StreamCausal {
        let states: Vec<FastCellState> = net.cells().iter().map(|c| c.state()).collect();
        let n_cells = net.cells().len();
        let dense = DenseRowChain::new(net.output_layers().len());
        let crop = net.input_width();
        let output_size = net.output_size();
        StreamCausal {
            net,
            states,
            crop,
            in_cols: 0,
            in_cols_seen: 0,
            pend: vec![Vec::new(); n_cells],
            fill: vec![0; n_cells],
            outs: vec![Vec::new(); n_cells],
            dense,
            output_size,
            emitted: 0,
        }
    }

    /// The posterior width (`output_size`), mirroring [`FastBlstm::output_size`].
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Posterior rows emitted so far (the streaming twin of the offline output row count).
    pub fn emitted(&self) -> usize {
        self.emitted
    }

    /// Buffer + forward a chunk of ALREADY-NORMALIZED feature rows, returning the
    /// posterior rows that completed this call (`k x output_size`, `k == 0` while the
    /// sub-sampling stages are still filling).
    pub fn push_rows(&mut self, rows: &FastMatrix) -> FastMatrix {
        if rows.rows == 0 {
            return FastMatrix::zeros(0, self.output_size);
        }
        if self.in_cols_seen == 0 {
            // The `feed_forward` crop gate, resolved once (the feature width is a
            // stream-wide constant). `in_cols_seen` is the RAW arriving width, kept
            // separately from the possibly-narrower cropped `in_cols` so the stability
            // check below can compare like with like (the phase-8 sibling's `input_cols`).
            let subs = self.net.lstm_subsampling();
            self.in_cols_seen = rows.cols;
            self.in_cols = if !subs.is_empty() && subs[0] > 1 && self.crop < rows.cols {
                self.crop
            } else {
                rows.cols
            };
        }
        debug_assert_eq!(
            rows.cols, self.in_cols_seen,
            "StreamCausal::push_rows col count must be stable across pushes"
        );

        let mut data: Vec<f32> = Vec::new();
        let mut produced = 0usize;
        for r in 0..rows.rows {
            let src_lo = r * rows.cols;
            if self.step_stack(&rows.data[src_lo..src_lo + self.in_cols]) && self.dense_step() {
                data.extend_from_slice(self.dense.output(self.net.output_layers()));
                produced += 1;
            }
        }
        self.emitted += produced;
        FastMatrix {
            data,
            rows: produced,
            cols: self.output_size,
        }
    }

    /// EOS: nothing to flush (a partial sub-sample group is DROPPED, the offline
    /// `T mod R` truncation). Present for symmetry with [`StreamOverlap::flush`] and to
    /// make the absence of a tail an explicit, documented fact rather than an omission.
    pub fn flush(&mut self) -> FastMatrix {
        FastMatrix::zeros(0, self.output_size)
    }

    /// Drive ONE cropped feature row through the cell stack. Returns `true` when the
    /// stack produced a hidden row (readable at `self.outs[last]`), `false` while a
    /// sub-sampling stage is still buffering.
    fn step_stack(&mut self, row: &[f32]) -> bool {
        let cells = self.net.cells();
        let subs = self.net.lstm_subsampling();
        let n = cells.len();
        for jj in 0..n {
            let ratio = subs[jj].max(1);
            if ratio == 1 {
                self.pend[jj].clear();
            }
            if jj == 0 {
                self.pend[0].extend_from_slice(row);
            } else {
                let w = cells[jj - 1].output_size();
                self.pend[jj].extend_from_slice(&self.outs[jj - 1][..w]);
            }
            if ratio > 1 {
                self.fill[jj] += 1;
                if self.fill[jj] < ratio {
                    return false;
                }
                self.fill[jj] = 0;
            }
            let o = cells[jj].output_size();
            if self.outs[jj].len() < o {
                self.outs[jj].resize(o, 0.0);
            }
            cells[jj].step(
                &self.pend[jj],
                &mut self.states[jj],
                &mut self.outs[jj][..o],
            );
            self.pend[jj].clear();
        }
        true
    }

    /// Feed the stack's hidden row into the dense output MLP chain.
    fn dense_step(&mut self) -> bool {
        let cells = self.net.cells();
        let last = cells.len() - 1;
        let w = cells[last].output_size();
        self.dense.push_row(
            self.net.output_layers(),
            self.net.output_subsampling(),
            &self.outs[last][..w],
        )
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

    /// The audio-time BEGIN of a raw segment that is currently OPEN or TENTATIVELY open --
    /// i.e. a rising crossing has been recorded (`begin >= 0`) and not yet either closed
    /// into `raw_segments` or retracted by a failed area test. `None` when the hysteresis
    /// carries no candidate begin at all.
    ///
    /// THIS IS THE THIRD SOURCE OF FUTURE RAW STRUCTURE (Phase 9 Task 9 fix, found by the
    /// corpus streaming tier `tests/pyo3/test_phase9_parity.py`). A pending begin lies
    /// BEHIND the consumed frontier -- the hysteresis passed it and is still inside the
    /// segment -- so [`StreamDecision::resmooth_and_emit`]'s emission frontier must clamp to
    /// it; `max(last_raw_boundary, consumed_frontier - dt)` alone does NOT bound where the
    /// next raw segment begins. Both audio-time offsets match the closed-segment tuple
    /// (`begin + off`, `advance`'s label site).
    fn pending_begin(&self) -> Option<f64> {
        if self.begin >= 0.0 {
            Some(self.begin + self.off)
        } else {
            None
        }
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
/// frontier [`resmooth_and_emit`](Self::resmooth_and_emit) derives: `>= last_raw_boundary`,
/// `>= consumed_frontier - dt`, AND `>= the hysteresis's pending open begin` -- that third
/// term is the Phase-9 Task-9 fix, see the derivation there) -- can still reach it
/// through the smoothing. Each
/// smoothing step moves/merges boundaries by
/// at most its own threshold: [`Segmentation::add_padding`] extends a Speech segment left by
/// `before` and right by `after`; [`Segmentation::suppress_short`] removes/merges a segment
/// only across its own `<= threshold` span. A future Speech segment can therefore reach an
/// earlier segment's END at most `pad_before + suppress` to its LEFT -- the `after` paddings
/// extend the future segment RIGHTWARD (away from earlier segments), so they add NO leftward
/// reach; the true leftward drivers are the two `before` paddings plus the Speech
/// `suppress_short` spans (the worst-case chain the T4 review derived independently -- ~1.685 s
/// on the gate config). Chains through EARLIER segments happen identically with or without the
/// future segment (their gaps are past-known), so they do not extend the future segment's
/// reach. HOLDBACK is the CONSERVATIVE sum of EVERY (clamped) smoothing threshold --
/// `sum(min_speech) + sum(min_silence) + sum(padding)` -- which STRICTLY dominates that true
/// leftward reach precisely BECAUSE it ALSO sums in the rightward-only `after` paddings on top
/// (the measured ~0.604 s over-coverage on the gate config); it also keeps the mid-stream
/// clone's End sentinel (placed at `now >= frontier`,
/// since the received frontier is `>= the consumed frontier`) at least HOLDBACK to the right
/// of the emitted region, so the tail special-casing never touches it. Param-driven at
/// construction: a config with different padding moves the
/// holdback. If it were ever too small, the prefix-consistency gate catches it as a
/// retraction (R2); too large only adds latency.
///
/// # THE CUT (the incremental resmooth): bounded per-push replay
///
/// The re-smooth above is a PURE function of the raw-segment list, which is append-only,
/// so replaying ALL of it on EVERY push is `O(#segments)` per push -- quadratic
/// cumulative on an unbounded stream, and `label_segment`'s walk-to-insertion-point makes
/// one replay `O(S^2)` on its own. The fix keeps the arithmetic and shrinks the input:
///
/// * `keep_from` -- raw segments `[..keep_from]` are FROZEN OUT of the replay;
///   [`build_from`](Self::build_from) replays only `raw_segments[keep_from..]` into a
///   fresh `Segmentation::new(dur)` (so the retained replay's own left context is a
///   leading `Other@0`, NOT the true prefix -- that difference is what the argument below
///   bounds).
/// * `frozen` -- the smoothed entries with `begin < cut_time`, carried verbatim from the
///   build that first produced them. They are the head of the output list; the retained
///   build supplies everything from `cut_time` rightward and the two are SPLICED
///   ([`splice_index`](Self::splice_index)).
/// * `cut_time` -- monotone non-decreasing, advanced to the emission `threshold`
///   (`frontier - holdback`) on every push. Freezing what we EMIT is the same act, so
///   there is no second frontier to keep honest.
///
/// THE INVARIANT (maintained by [`advance_cut`](Self::advance_cut), and the whole
/// argument): `raw_segments[keep_from].0 + CUT_MARGIN_HOLDBACKS * holdback <= cut_time`.
/// Two independent claims meet at `cut_time` and they must not be conflated:
///
/// 1. RIGHTWARD (nothing future can change the frozen prefix) -- ALREADY PROVEN, this is
///    verbatim the emission argument above: every future raw segment begins at
///    `>= frontier` (the three-term bound, `pending_begin` clamp included) and the
///    smoothing's leftward reach is `<= holdback`, so every entry at
///    `begin < frontier - holdback == threshold` is final. Finality is absolute -- the
///    set of future segments only shrinks -- so a `cut_time` frozen at an earlier,
///    LARGER threshold stays valid when the frontier later retreats (it can: the
///    `pending_begin` clamp is a `min`). This is why `cut_time` is a running max.
/// 2. LEFTWARD (truncating the replay cannot change the output right of the cut) -- NEW,
///    derived below. This is the direction the phase-8/9 argument never needed.
///
/// ## The leftward (cut) derivation
///
/// Write `X` for `raw_segments[keep_from].0`, the first RETAINED raw begin. The raw
/// segments are disjoint and ascending (a labeled `end` resets the hysteresis, and the
/// same-step re-run-rising cannot fire on a falling crossing since it needs
/// `t_r <= r <= t_f` and `t_f < r_prev < t_r` at once), so the pre-smoothing list is a
/// plain concatenation and the FULL and RETAINED lists are IDENTICAL at `>= X`:
///
/// ```text
///   full:      Other@0, S@b_0, O@e_0, ... , O@e_{k-1}, | S@X, O@e_k, ... , End@dur
///   retained:                            Other@0,      | S@X, O@e_k, ... , End@dur
/// ```
///
/// -- same entries, same types, and the carried type immediately LEFT of `X` is `Other`
/// in both. Only the leading `Other`'s DURATION differs (`X` vs `X - e_{k-1}`). Define
/// `D_j` = the rightmost time such that the two lists agree (entries + types + the
/// carried type) on `[D_j, inf)` after pipeline step `j`; `D_0 = X`.
///
/// Every step is LOCAL IN BOTH SENSES, which is what makes the induction go through:
/// * `sanitize` -- rounding is per-boundary; the merge test reads `(ty[i], ty[i+1])`.
///   Reach 0.
/// * `suppress_short(th, cls)` -- the decision at `i` reads only
///   `(i == 0, i+2 == len, segs[i-1].ty, segs[i].ty, segs[i].begin, segs[i+1].begin,
///   segs[i+1].ty)`. `previous_type == segs[i-1].ty` is an INVARIANT, not a walk
///   accumulator: it is written only on the keep branch (`previous_type = segs[i].ty;
///   i += 1`) and the removal branches neither advance `i` nor rewrite it, so it always
///   names the last KEPT entry, i.e. `i-1`. Effects touch `i-1..i+1` only, and move a
///   boundary by `<= th` (head branch: `segs[i+1].begin = segs[i].begin`, a `<= th`
///   move; split branch: `-= dur/2 <= th/2`; merge branch: deletes the two entries of a
///   `<= th` span).
/// * `add_padding(before, after, cls)` -- the `before` sub-pass writes
///   `[b - before, b + 1e-6]`, the `after` sub-pass `[e - 1e-6, e + after]`; the
///   backward walk touches only indices `> ii`, so it never disturbs an unvisited one.
///
/// SO A DIFFERENCE CANNOT OUTRUN THE STEP'S OWN THRESHOLD. The only way a
/// prefix-only difference reaches right of `D_j` is by changing the DURATION of the
/// entry that STRADDLES `D_j` (its left end is in the differing region, its right end is
/// not); a straddler's right end then moves only if the step FLIPS it, which every
/// branch conditions on `dur <= th` -- so the flipped entry's right end sits at
/// `< D_j + th`. Hence `D_{j+1} <= D_j + th_j` and, composing the eight steps,
///
/// ```text
///   D_8 <= X^+ + sum(min_speech) + sum(min_silence) + padding[1] + padding[3] + 2e-6
///       <= X^+ + holdback + 2e-6                                 (padding[0,2] >= 0)
///       <  X  + CUT_MARGIN_HOLDBACKS * holdback  <  cut_time     (the invariant, STRICT)
/// ```
///
/// (`X^+` rather than `X` because of case 1(b) below -- the touching seam contributes an
/// infinitesimal that the slack absorbs.)
///
/// Only the two `after` paddings enter the reach (`before` extends a Speech segment
/// LEFTWARD, away from the region at risk; its sole rightward footprint is the `+1e-6`
/// re-close), so `holdback` -- which sums `before` too -- already over-covers; the margin
/// is the SECOND holdback. The margin is expressed in HOLDBACK UNITS, never as a second
/// per-term constant (the Task-9 adjudication).
///
/// THE PRECONDITION OF THAT LAST `<`, stated rather than assumed: `holdback > 2e-6`. Then
/// `holdback + 2e-6 < 2*holdback` and the step is strict. Two ends of the range:
/// * `holdback == 0` -- EXACT, not approximate: every threshold is zero, so `suppress_short`
///   returns early and `add_padding` skips both sides; the pipeline reduces to `sanitize`
///   alone, whose reach is 0 (case 1a) or `X^+` (case 1b), and `advance_cut`'s STRICT `<`
///   gives `X < cut_time`, which covers `X^+`. ONE caveat, sized honestly (review delta):
///   the ROUNDING variant of case 1(b) disagrees at `round(X)`, up to `5e-5` right of `X`
///   (half the `1e-4` grid), which `X < cut_time` does not formally cover -- there the
///   strict `<` is backed by DATA rather than the invariant alone: `cut_time - X` exceeds
///   the anchor segment's own duration, whose measured floor is `0.03552 s` unconvolved /
///   `0.32089 s` under the 19-tap kernel, 700x-6400x the displacement. No `1e-6` term
///   exists because the only code that writes one is inactive.
/// * `0 < holdback <= 2e-6` -- the one EXCLUDED sliver, and it is not a real config: an
///   active `before` padding would have to be positive yet below `2e-6 s`, i.e. under a
///   fiftieth of the `1e-4 s` grid `sanitize` snaps every boundary to, so it could not move
///   a boundary at all.
///
/// THE SEAM CASES, step by step (`D_j`'s advance, and why it cannot exceed it):
/// 1. `sanitize` -- TWO sub-cases, and the second one is the reason this list is not a
///    formality (it is the T9 shape again: a seam case whose stated reason covered only
///    the situation that first came to mind).
///    SUB-CASE (a), DISJOINT (`e_{k-1}` and `b_k` land on different `1e-4` grid points):
///    the entry at `X` is `Speech` preceded by `Other` in BOTH lists, so no merge crosses
///    the seam. `D_1 = X`.
///    SUB-CASE (b), TOUCHING (`e_{k-1} == b_k` in f64, or both rounding to the same grid
///    point): the FULL list's `label_segment(b_k, e_k)` stops its insertion walk BEFORE
///    `O@e_{k-1}` (`segs[i].begin < begin` is false at equality), inserts `S@b_k` in front
///    of it, and then ERASES it in the overwrite loop -- leaving `S@b_{k-1}, S@b_k`
///    ADJACENT, which `sanitize` merges, DELETING the entry at `X` that the retained list
///    keeps. (The rounding variant leaves a zero-duration `O@X` in the full list only, same
///    shape.) So the entry AT `X` differs and agreement resumes at `X^+`: `D_1 = X^+`, an
///    infinitesimal the slack absorbs. Reachable, not hypothetical -- the exact-arithmetic
///    separation `b_k > e_{k-1}` can vanish in f64 when the rising interpolation's fraction
///    rounds to 1 at the next step index. THIS IS ALSO WHY
///    [`advance_cut`](Self::advance_cut) slides `keep_from` on a STRICT `<`: the invariant
///    is then `X + margin < cut_time`, so `X^+ <= cut_time` holds even at `holdback == 0`,
///    where there is no slack at all to absorb it.
/// 2. `suppress_short(min_speech[0], Speech)` -- the first retained Speech has the SAME
///    duration in both and `previous_type == Other` in both (alternating list), so the
///    branch is identical. A prefix-only flip moves entries by `<= min_speech[0]`.
/// 3. `add_padding(padding[0], padding[1], Speech)` -- a Speech at `b >= D_2` pads left
///    ACROSS the seam and may merge with a differing left neighbour: the merged BEGIN
///    differs, but only left of `D_2`; the rightward footprint is `padding[1]` (+`1e-6`).
/// 4. `suppress_short(min_silence[0], Other)` -- THE LOAD-BEARING CASE, and the exact
///    shape of the phase-9 T9 lesson (a structure the naive argument does not see): the
///    straddling `Other`'s duration is `X` in the retained list and `X - e_{k-1}` in the
///    full one, so the full list can merge its two flanking Speech segments while the
///    retained one keeps them apart -- DELETING the entry at `X` in one and not the
///    other. It fires only at `dur <= min_silence[0]`, so the deleted entry sits at
///    `< D_3 + min_silence[0]`.
/// 5. `suppress_short(min_speech[1], Speech)` -- the straddling Speech's duration now
///    differs (from 3/4); the flip needs `dur <= min_speech[1]`, bounding its right end.
/// 6. `add_padding(padding[2], padding[3], Speech)` -- as 3.
/// 7. `suppress_short(min_silence[1], Other)` -- as 4.
/// 8. `suppress_short(min_speech[2], Speech)` -- as 5.
///
/// THE TAIL CASES, which no step index covers, and they need BOTH halves:
/// * NECESSARY -- the `End` sentinel and the two positional predicates
///   (`suppress_short`'s `i+2 == len`, `add_padding`'s backward `len-3` start) address the
///   LAST entries, shared verbatim by both lists; `i == 0` addresses the leading `Other`,
///   which is index 0 in both. None of the three can name a DIFFERENT segment in the two
///   lists, which is the only way a positional predicate could leak the prefix's length
///   into the shared region.
/// * SUFFICIENT -- and this is the half that keeps them away from the FROZEN region rather
///   than merely consistent between the lists: the sentinel sits at least one `holdback`
///   to the RIGHT of everything frozen. Mid-stream `mid_dur = max(now, last_rb) >= now`,
///   and `cut_time <= threshold = frontier - holdback <= now - holdback`, so the tail
///   special-casing's own `holdback`-bounded reach cannot touch an entry at
///   `begin < cut_time`. At EOS the sentinel comes from the CALLER's clock, so
///   [`flush`](Self::flush) re-checks this same inequality instead of assuming it (see its
///   sentinel guard) -- the one place the argument is enforced at runtime rather than by
///   construction.
///
/// WHAT THE CUT COSTS: nothing in output and nothing in latency (it never gates an
/// emission -- it only stops re-deriving what is already frozen), and `O(K^2)` per push
/// instead of `O(S^2)`, with `K` the retained count bounded by the window
/// `[cut_time - CUT_MARGIN_HOLDBACKS*holdback, now]`. Its width is
/// `(now - cut_time) + 2*holdback <= 3*holdback + (now - frontier)` (since `cut_time >=
/// frontier - holdback`) -- config-derived plus the open-segment age, and no raw segment
/// CLOSES while one is open, so a long open span adds at most that ONE segment.
/// `tests/stream_incremental_resmooth.rs::derived_retained_bound` turns this into a number
/// (155-254 on the four profiles it runs) against a measured worst of 6; the gap is the
/// last step's assumption that the hysteresis could close a segment at every grid step,
/// which the area gating forbids but this layer cannot see.
/// The full replay stays reachable two ways: [`set_full_replay`](Self::set_full_replay)
/// (the `test-support` equivalence oracle -- it simply never advances the cut, so the
/// phase-8 code path is recovered EXACTLY, not re-implemented) and the flush-time
/// sentinel guard in [`flush`](Self::flush).
///
/// WHAT THE CUT COSTS IN MEMORY, stated because it trades one growing list for two: the
/// cut does NOT drain `raw_segments` (it is public API the phase-8 suite compares against
/// the offline batch), and it ADDS `frozen`, a second `O(S)` list at 16 B/entry
/// (`Segment` = an `f64` plus a `repr(i32)` tag, padded). At a dense conversational rate
/// of one speech segment per 2 s that is ~2 entries/2 s, so a 24 h stream holds ~86400
/// entries ~ 1.4 MB -- immaterial next to the compute it buys, and the reason this design
/// spends memory rather than trying to bound it. Bounding `frozen` too would mean
/// emitting-and-forgetting the settled prefix, which would change `flush`'s contract (it
/// returns the COMPLETE `Segmentation`) and is a different design, not a tightening.
///
/// TWO NAMED FOLLOW-ONS, recorded rather than done (neither is a defect):
/// * A REALISED-REACH DIAGNOSTIC, prerequisite for any future margin tightening. The
///   oracle's mutation ladder localises the safe margin only to a quarter-holdback
///   (`0.0` passes, `-0.25` fails), because a pass/fail probe measures where the streams
///   BREAK, not how much reach they actually exercise. Instrumenting `D_8 - X` directly --
///   compare the retained build against a full build per push and report the rightmost
///   disagreement -- would turn "somewhere under one holdback" into a number, and nobody
///   should shrink `CUT_MARGIN_HOLDBACKS` on ladder evidence alone.
/// * THE STRADDLE COUNT IS PER-PUSH, so the oracle's headline figure is inflated by the
///   fine chunkings (chunk 1 re-counts the same straddling configuration on every push).
///   It is honest as a coverage floor and misleading as a headline; the quantity worth
///   reporting is the number of DISTINCT straddling raw-segment indices.
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
    /// Indexes the SPLICED list (`frozen` then the retained build's tail), which is the
    /// same global partition the phase-8 full replay produced.
    emitted_count: usize,

    /// THE CUT (see the type docs). Raw segments `[..keep_from]` are frozen out of the
    /// replay; the retained window is `raw_segments[keep_from..]`. Monotone.
    keep_from: usize,
    /// The smoothed entries with `begin < cut_time`, PROVEN final (rightward claim) and
    /// PROVEN correct when they were frozen (leftward claim). The head of the output list.
    frozen: Vec<Segment>,
    /// The freeze/cut time: `frozen` holds everything left of it, the retained build
    /// reproduces everything at or right of it. Monotone non-decreasing; `NEG_INFINITY`
    /// until the first freeze (so the splice index is 0 and the path is byte-identical to
    /// the phase-8 full replay).
    cut_time: f64,
    /// TEST-ONLY (see [`set_full_replay`](Self::set_full_replay)): suppress the cut, so
    /// every push replays the WHOLE raw list -- the phase-8 path, the equivalence oracle's
    /// reference. Always `false` in production (nothing outside `test-support` can set it).
    full_replay: bool,

    finished: bool,
}

/// The cut margin, in HOLDBACK UNITS -- the ONE constant family this design is allowed
/// (the Task-9 adjudication: never a second per-term constant). The retained replay
/// window must begin at least `CUT_MARGIN_HOLDBACKS * holdback` to the LEFT of the frozen
/// prefix's right edge; the derivation (which needs `1 * holdback + 2e-6`) is in
/// [`StreamDecision`]'s docs, and the second holdback is the slack that dominates the
/// `add_padding` `1e-6` re-close epsilons.
const CUT_MARGIN_HOLDBACKS: f64 = 2.0;

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
            keep_from: 0,
            frozen: Vec::new(),
            cut_time: f64::NEG_INFINITY,
            full_replay: false,
            finished: false,
        }
    }

    /// The derived smoothing holdback in seconds (see the type docs). Exposed for the
    /// latency accounting + the param-driven-derivation gate.
    pub fn holdback(&self) -> f64 {
        self.holdback
    }

    /// The number of raw segments the incremental replay currently re-runs -- the
    /// RETAINED window `raw_segments[keep_from..]`. Bounded by the cut arithmetic (see
    /// the type docs), NOT by the stream length; that is the whole point, and
    /// `tests/stream_incremental_resmooth.rs` pins it measured-vs-derived.
    pub fn retained_len(&self) -> usize {
        self.raw_segments.len() - self.keep_from
    }

    /// The current freeze/cut time (`NEG_INFINITY` before the first freeze). Everything
    /// left of it is in the frozen prefix; the retained replay reproduces the rest.
    pub fn cut_time(&self) -> f64 {
        self.cut_time
    }

    /// TEST-ONLY: suppress the cut, restoring the phase-8 FULL replay (every push replays
    /// the WHOLE raw list) as the equivalence oracle's reference path.
    ///
    /// This is a SUPPRESSION, not a second implementation: with the cut never advancing,
    /// `keep_from` stays 0, `frozen` stays empty and `cut_time` stays `NEG_INFINITY`, so
    /// [`build_from`](Self::build_from) replays everything, the splice index is 0 and
    /// [`collect_prefix`](Self::collect_prefix) walks exactly the entries the phase-8
    /// code walked. The oracle therefore compares the incremental path against the
    /// ORIGINAL code path, not against a paraphrase of it.
    #[cfg(feature = "test-support")]
    pub fn set_full_replay(&mut self, on: bool) {
        self.full_replay = on;
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

        // The complete segmentation: seed at the TRUE audio_duration, replay the RETAINED
        // raw segments, smooth once, and splice the frozen prefix back on -- identical to
        // `results_to_segmentation` + smoothing over the whole list (the cut derivation).
        //
        // SENTINEL SAFETY (the one place the frozen prefix's provenance is re-checked):
        // every frozen entry was proven final against a mid-stream sentinel at `now`, and
        // the argument needs the sentinel to stay at least one holdback RIGHT of the
        // frozen region (else the tail special-casing -- suppress_short's `i+2 == len`
        // branch and add_padding's backward `len-3` start -- could reach it). The caller's
        // `audio_duration` is a DIFFERENT clock from the decision layer's row frontier, so
        // this is checked rather than assumed; the streaming session always satisfies it
        // (`audio_duration >= now >= cut_time + holdback`), and a caller that does not
        // gets the full replay instead of a wrong answer.
        let now = self.current_audio_time();
        let (emitted, final_seg) = if audio_duration <= self.cut_time + self.holdback {
            let seg = self.build_from(audio_duration, 0);
            let emitted = Self::collect_prefix(
                &[],
                seg.segments(),
                &mut self.emitted_count,
                f64::INFINITY,
                now,
            );
            (emitted, seg)
        } else {
            let built = self.build_from(audio_duration, self.keep_from);
            let q = self.splice_index(built.segments());
            let tail = &built.segments()[q..];
            let emitted = Self::collect_prefix(
                &self.frozen,
                tail,
                &mut self.emitted_count,
                f64::INFINITY,
                now,
            );
            let mut all = Vec::with_capacity(self.frozen.len() + tail.len());
            all.extend_from_slice(&self.frozen);
            all.extend_from_slice(tail);
            (emitted, Segmentation::from_parts(all, audio_duration))
        };
        (emitted, final_seg)
    }

    /// Build a smoothed [`Segmentation`] seeded at `audio_dur` from the raw segments
    /// `[from..]` -- the SHARED offline path (`Segmentation::new` -> `label_segment`
    /// replay -> `smooth_segmentation`), run read-only on a clone. `from == 0` is the
    /// phase-8 full replay verbatim; `from == keep_from` is the retained window whose
    /// output is spliced onto [`frozen`](Self::frozen) (see the cut derivation).
    fn build_from(&self, audio_dur: f64, from: usize) -> Segmentation {
        let mut seg = Segmentation::new(audio_dur);
        for &(b, e) in &self.raw_segments[from..] {
            seg.label_segment(b, e, self.class);
        }
        smooth_segmentation(&mut seg, &self.seg_cfg);
        seg
    }

    /// The index in a retained build's entry list where the FROZEN prefix ends and the
    /// spliced tail begins: the first entry at `begin >= cut_time`.
    ///
    /// The half-open rule (`frozen` = every entry with `begin < cut_time`) is what makes
    /// this unambiguous even when the smoothed list carries a zero-duration segment (two
    /// entries sharing a `begin`): both freezing and splicing partition on the SAME
    /// predicate, so a tie lands wholly on one side. `cut_time == NEG_INFINITY` (nothing
    /// frozen yet) yields 0, the phase-8 whole-list walk.
    fn splice_index(&self, entries: &[Segment]) -> usize {
        entries
            .iter()
            .position(|s| s.begin >= self.cut_time)
            .unwrap_or(entries.len())
    }

    /// Advance the cut: freeze the entries this push's `threshold` proves final, then
    /// slide the retained window up to the margin the leftward derivation allows.
    ///
    /// `entries` is the CURRENT build's tail (already spliced at the OLD `cut_time`), so
    /// every entry it holds is at `>= cut_time_old >= X_old + CUT_MARGIN*holdback` and is
    /// therefore inside the region that build reproduces correctly -- which is the
    /// induction step that keeps `frozen` honest.
    fn advance_cut(&mut self, threshold: f64, entries: &[Segment]) {
        // `<=` rather than `!(>)`: a NaN threshold (unreachable -- every term is a finite
        // time -- but the comparison must still be total) leaves the cut where it is.
        if self.full_replay || threshold <= self.cut_time || threshold.is_nan() {
            return;
        }
        for e in entries {
            // The End sentinel travels with the stream (mid-stream it sits at `now`), so
            // it is never frozen. Unreachable in practice -- the sentinel is at
            // `>= now >= threshold` -- and cheap insurance if a caller's clocks disagree.
            if e.begin >= threshold || e.ty == SegClass::End {
                break;
            }
            self.frozen.push(*e);
        }
        self.cut_time = threshold;

        // Slide the retained window: keep the LAST raw segment beginning STRICTLY before
        // `cut_time - CUT_MARGIN_HOLDBACKS*holdback`, so the invariant
        // `raw_segments[keep_from].0 + margin < cut_time` holds by construction. At least
        // one raw segment is always retained -- it anchors the seam structure the
        // derivation reasons about (`S@X` preceded by `Other` in both lists).
        //
        // THE `<` IS STRICT ON PURPOSE (review I-1). Seam case 1(b) advances agreement to
        // `X^+`, not `X`, so a non-strict slide would leave `X == cut_time` admissible and
        // the splice would then READ the one entry the touching seam can differ on. At any
        // `holdback > 0` the margin swamps that, but at `holdback == 0` there is no slack
        // at all and the strict comparison is the whole difference between exact and wrong.
        let limit = self.cut_time - CUT_MARGIN_HOLDBACKS * self.holdback;
        while self.keep_from + 1 < self.raw_segments.len()
            && self.raw_segments[self.keep_from + 1].0 < limit
        {
            self.keep_from += 1;
        }
        debug_assert!(
            self.keep_from == 0
                || self.raw_segments[self.keep_from].0 + CUT_MARGIN_HOLDBACKS * self.holdback
                    < self.cut_time,
            "cut invariant broken: retained window starts at {} but cut_time is {}",
            self.raw_segments[self.keep_from].0,
            self.cut_time
        );
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
        // THE CUT: only the RETAINED raw segments are replayed here (`keep_from`), which is
        // what makes this push `O(K^2)` instead of `O(S^2)`. The frozen prefix supplies
        // everything left of `cut_time`; see the type docs' cut derivation for why the
        // truncated left context cannot change a single bit at or right of the cut.
        let built = self.build_from(mid_dur, self.keep_from);

        // THE EMISSION FRONTIER (the time-advance trigger). A smoothed segment can still be
        // revised only by FUTURE raw structure -- raw segments not yet in `raw_segments`.
        // There are THREE places such a segment's BEGIN can come from, and the frontier is
        // the EARLIEST of them:
        //   1. `last_rb` -- raw segments close in ascending order, so the next one begins
        //      after this one ends;
        //   2. `hyst_frontier - dt` -- the CONSUMED frontier is `hyst_frontier =
        //      conv.finalized*dt + off` (the hysteresis has advanced through exactly
        //      `conv.finalized` convolved values; `StreamDecision` owns both `conv` and
        //      `hyst`, so no session threading is needed) and the earliest a not-yet-consumed
        //      value can interpolate a boundary back to is one grid step (`dt`) before it;
        //   3. `hyst.pending_begin()` -- a segment the hysteresis has ALREADY OPENED (or
        //      tentatively opened) and not yet closed. Its begin lies BEHIND the consumed
        //      frontier, so 1+2 do NOT bound it.
        //
        // (3) IS A PHASE-9 TASK-9 FIX, not part of the original T5-review derivation. The
        // phase-8 argument enumerated only 1+2 and concluded "future raw structure lands no
        // earlier than the frontier" -- FALSE while a raw segment is open: speech that
        // resumes within the holdback of the previous segment's end is already open (and so
        // invisible to `last_rb`) at the moment the time-advance trigger would license
        // emitting that previous segment, and its `add_padding` `before` reach then merges
        // the two, MOVING an already-emitted boundary. Found on real data by the corpus
        // streaming tier (`tests/pyo3/test_phase9_parity.py`, mamba/forward: `[0, 7.3051]
        // Speech` emitted at 10.1999 s while `finish` reported one `[0, 74.9999]` segment --
        // a genuine R2 retraction), pinned synthetically by
        // `phase8_stream_decision.rs::pending_open_segment_blocks_emission`. It costs
        // LATENCY, never correctness: clamping the frontier only emits LESS.
        //
        // THE SAFETY ARGUMENT: the T4 holdback-dominance proof (see the type docs) licenses
        // emitting any segment whose smoothed form cannot be reached by raw structure at or
        // beyond the frontier -- and the three terms above bound where that structure can
        // begin, so every segment ending strictly before `frontier - holdback` is final.
        // (Using the RECEIVED frontier `now` here would be UNSAFE: `now` runs ahead of the
        // consumed frontier by the convolution lookahead + a grid step, so it could license
        // emitting a segment a still-pending convolved value can still move.) The
        // prefix-consistency gate legs are the safety net -- any too-large frontier surfaces
        // there as a retraction.
        let hyst_frontier = self.conv.finalized as f64 * self.dt + self.off;
        let mut frontier = last_rb.max(hyst_frontier - self.dt);
        if let Some(pending) = self.hyst.pending_begin() {
            frontier = frontier.min(pending);
        }
        let threshold = frontier - self.holdback;

        // FREEZE WHAT WE EMIT: the cut advances to the SAME threshold the emission uses
        // (one frontier, one holdback, one act), then the splice is recomputed against the
        // new cut and the settled prefix is read off the spliced view.
        let q_old = self.splice_index(built.segments());
        self.advance_cut(threshold, &built.segments()[q_old..]);
        let q = self.splice_index(built.segments());
        Self::collect_prefix(
            &self.frozen,
            &built.segments()[q..],
            &mut self.emitted_count,
            threshold,
            now,
        )
    }

    /// Emit the contiguous prefix of the SPLICED partition (`frozen` then `tail`, from
    /// `emitted_count`) whose segments END strictly before `threshold`, stamping
    /// `emitted_at`. Advances `emitted_count`. Segment `i` is
    /// `[entry(i).begin, entry(i+1).begin)` of type `entry(i).ty`.
    ///
    /// An associated function over the two slices rather than a `&mut self` method: it
    /// borrows `frozen` immutably and the cursor mutably, which are disjoint fields. With
    /// `frozen` empty and `tail` the whole build (the `full_replay` path, and every push
    /// before the first freeze) this walks exactly the entries the phase-8 version walked.
    fn collect_prefix(
        frozen: &[Segment],
        tail: &[Segment],
        emitted_count: &mut usize,
        threshold: f64,
        emitted_at: f64,
    ) -> Vec<EmittedSegment> {
        let n = frozen.len() + tail.len();
        let at = |i: usize| -> Segment {
            if i < frozen.len() {
                frozen[i]
            } else {
                tail[i - frozen.len()]
            }
        };
        let mut out = Vec::new();
        let mut i = *emitted_count;
        while i + 1 < n && at(i + 1).begin < threshold {
            let cur = at(i);
            out.push(EmittedSegment {
                begin_s: cur.begin,
                end_s: at(i + 1).begin,
                class: cur.ty,
                emitted_at_audio_s: emitted_at,
            });
            i += 1;
        }
        *emitted_count = i;
        out
    }
}

// ===========================================================================
// StreamingSession (Phase 8 Task 5): the composed SAD streaming session + gate.
// ===========================================================================

/// The session's NN stage, per the config's `Cell_Type` x `Direction` pair (spec S5.2).
/// A CLOSED set with static dispatch, exactly like `fast::driver`'s `FastSadNet`: the
/// windowed-BLSTM arm is the phase-8 path, BEHAVIOUR-UNTOUCHED; the causal arm is Task 7.
/// Both feed the SAME [`StreamDecision`], so `push`/`finish` keep one shape and the
/// `speech stream` CLI + the PyO3 `StreamingSession` gain the causal mode with ZERO new
/// surface.
enum StreamNn {
    /// Phase 8: windowed overlap over the bidirectional peephole-LSTM twin.
    Windowed {
        overlap: StreamOverlap,
        net: FastBlstm,
    },
    /// Phase 9 Task 7: the per-row causal stack (`slstm`/`mamba` + `Direction forward`).
    Causal(StreamCausal),
}

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
/// PHASE 9 TASK 7: the same session also drives the CAUSAL arm. `new` dispatches on
/// `Cell_Type` x `Direction` (spec S5.2) and builds either the phase-8
/// [`StreamOverlap`] + [`FastBlstm`] pair or [`StreamCausal`] over a
/// [`FastCausalNet`]; everything else here -- the front-end, the decision layer, the
/// frozen-gain contract, `push`/`finish`, the latency accounting -- is shared verbatim.
/// The bit-equal oracle on the causal arm is the offline fast CAUSAL bag run
/// (`tests/phase9_stream_causal.rs`). The input-normalization contract widens to `{0, 1}`
/// ON THE CAUSAL ARM ONLY (type 0 is causality-trivial -- nothing whole-file to freeze --
/// and it is what the committed phase-9 causal fixtures carry); the WINDOWED arm keeps
/// phase 8's `{1}` byte-identically, so no config phase 8 accepted or refused changed
/// status.
///
/// LATENCY (spec S1.8, S5.5). Every emission is stamped with the session's audio-time-pushed
/// clock (`(total_pushed-1)/rate`) and its per-emission lag (`emitted_at - end_s`) is
/// tracked (running max/mean, O(1) memory). With the T5-review consumed-frontier time-advance
/// ([`StreamDecision::push_rows`]), a SPEECH segment is delivered within the DERIVED structural
/// budget [`derived_latency_bound_s`](Self::derived_latency_bound_s) = front-end reach + NN
/// window finalization + sub-sample buffering + convolution half-width + the smoothing
/// holdback -- it emits during
/// the FOLLOWING silence, no longer waiting for the next raw segment. THE CAUSAL ARM DROPS
/// THE `nn_window` TERM ENTIRELY (spec S5.5: no window, no lookahead) and pays the far
/// smaller sub-sample buffering term instead, which is the phase-9 latency win. An OTHER (silence)
/// segment still waits for the following speech to COMMIT (its right boundary is that speech's
/// onset, latched only at the falling edge), so its lag is that speech's DURATION + the forward
/// pipeline delay -- the data-dependent AREA term pinned per-class on BOTH arms'
/// fixtures (`tests/phase8_gate.rs` windowed, `tests/phase9_stream_causal.rs` causal).
pub struct StreamingSession {
    front: StreamFrontEnd,
    pipeline: FastPipeline,
    nn: StreamNn,
    decision: StreamDecision,

    /// The pack-carried type-1 normalize tail (narrowed f32), applied per feature-row
    /// front-end-side (the documented T3/T4 seam: type 1 is a per-column affine, so it
    /// commutes with row-streaming and is bit-identical to the offline whole-sequence
    /// application).
    norm_mean: Vec<f32>,
    norm_std: Vec<f32>,
    /// `InputNormalizationType 1` -> apply the frozen tail per feature row; type 0 ->
    /// nothing (the exact path's `_ => {}` arm). No other value reaches construction.
    apply_norm: bool,

    rate: f64,
    total_pushed: usize, // absolute sample count == the audio-time clock source
    finished: bool,

    // Latency budget (config-derived once at construction; the pins read these back).
    feature_reach_s: f64, // (reach*shift_frames + half_window)/rate
    nn_window_s: f64,     // 2*window_size_feature * spectrum_shift_sec (0.0 when causal)
    sub_sample_s: f64,    // (ssr-1) * spectrum_shift_sec (0.0 when windowed)
    conv_delay_s: f64,    // conv_half * time_step

    // Running per-emission lag stats (O(1) memory -- the low-memory streaming goal).
    lag_max: f64,
    lag_sum: f64,
    lag_count: usize,

    /// Posterior history (the NN stage's output -- [`StreamOverlap`] on the windowed arm,
    /// [`StreamCausal`] on the causal one -- `output_size`-wide f32) -- the gate's
    /// bit-equivalence observable, mirroring the offline `last_result_rows`. Accumulated
    /// ONLY under `test-support` (it grows unbounded; production streaming never keeps it,
    /// per the phase's low-memory goal), the SAME pattern as `BagOfProcessors`'
    /// `last_audio_abs_sum`.
    #[cfg(feature = "test-support")]
    posterior_history: Vec<f32>,
}

impl StreamingSession {
    /// Build the session from the SAME legacy config `map` the bag consumes, the stream
    /// `rate`, and the source `channels` count. Loads the frozen net ONCE, builds the
    /// phase-7 [`FastPipeline`], and wires the front-end + NN stage + decision layer.
    ///
    /// THE NN STAGE IS DISPATCHED (spec S5.2) on `Cell_Type` x `Direction`, through
    /// `fast::driver`'s own `classify_fast_shape` choke point -- ONE classifier, so the
    /// session and the offline driver cannot drift apart on which shape a config selects:
    ///  - `lstm` + bidirectional -> [`StreamNn::Windowed`] ([`StreamOverlap`] + [`FastBlstm`]),
    ///    the phase-8 path;
    ///  - ANY cell + forward -> [`StreamNn::Causal`] ([`StreamCausal`] over a
    ///    [`FastCausalNet`]), Phase 9 Task 7 -- with `lstm` joining in phase-10 Task 8
    ///    (`FastLstm`), which is why the causal-LSTM refusal this doc used to describe is
    ///    gone rather than moved: the shape is implemented.
    ///
    /// ONE SHAPE IS REFUSED HERE THAT THE OFFLINE TREE ACCEPTS (phase-10 Task 7): a
    /// BIDIRECTIONAL new cell. `classify_fast_shape` now names it (`FastNetShape::BiCell`,
    /// built offline by `fast::bicell::FastBiCell`), but it is unstreamable BY
    /// CONSTRUCTION -- the reverse stack reads the whole sequence -- so the session bails
    /// on it, keeping the LEADING CLAUSE of the refusal that used to live in the
    /// classifier (what the phase-8/9 streaming legs pin) and rewriting its tail.
    ///
    /// Validated contract, each bail pinned (`tests/phase8_gate.rs::validation_bails` for the
    /// windowed arm, `tests/phase9_stream_causal.rs::validation_bails` +
    /// `causal_output_size_not_one_bails` for the causal one): algo 3, MONO,
    /// `Audio_fixed_gain` present, `output_size 1`, plus the two ARM-DEPENDENT rules --
    ///  - INPUT NORMALIZATION: `1` (the pack-carried frozen tail) on either arm, and `0`
    ///    (no normalization) on the CAUSAL arm only; `-1` (per-sequence self-normalization)
    ///    always bails, since it needs the whole sequence before the first frame;
    ///  - WINDOWING: the windowed arm requires OVERLAP (plain and truncate bail); the causal
    ///    arm requires the PLAIN regime (`window_size` 0), because a causal cell's state
    ///    would be reset at every window boundary.
    pub fn new(
        map: &IndexMap<String, String>,
        rate: f64,
        channels: usize,
    ) -> Result<StreamingSession> {
        // --- Validation bails (each pinned) ---
        let algo = map
            .get("Algo_choice")
            .ok_or_else(|| anyhow!("param 'Algo_choice' not found in config"))?
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
        let fixed_gain = match get_f64_opt(map, "Audio_fixed_gain")? {
            Some(g) => g,
            None => bail!(
                "streaming requires Audio_fixed_gain (frozen-norm mode; the whole-file \
                 (2*RMS+max)/2 audio normalization is not streamable)"
            ),
        };
        let bc = BlstmConfig::from_legacy(map, "BLSTM")?;

        // Cell x direction dispatch (spec S5.2), through the SAME `classify_fast_shape`
        // choke point the offline fast SAD driver uses, so both sides agree on the shape
        // by construction. The classifier is TOTAL since phase-10 Task 8; the ONE shape
        // refused here but NOT offline (a bidirectional cell) is handled immediately
        // below.
        let shape = classify_fast_shape(&bc);
        // BIDIRECTIONAL IS UNSTREAMABLE BY CONSTRUCTION (phase-10 Task 7, spec S5): the
        // reverse stack's state at time `t` is a function of the samples AFTER `t`, so
        // its output at the first frame depends on the last one. There is no bounded
        // lookahead that makes it causal -- unlike the phase-8 windowed BLSTM (bounded by
        // the window) or the phase-9 causal cells (no lookahead at all). The refusal used
        // to come from `classify_fast_shape` itself, which now NAMES the shape because the
        // OFFLINE fast tree implements it (`fast::bicell::FastBiCell`). The refusal moved
        // here with its LEADING CLAUSE preserved -- which is what `phase8_gate.rs` and
        // `phase9_stream_causal.rs` assert (a PREFIX SUBSTRING, not the body), hence both
        // stay green UNMODIFIED -- and its TAIL REWRITTEN, because the classifier's old
        // advice ("run this config on the exact path") is now wrong here: the offline fast
        // path DOES implement this shape, it is streaming that cannot.
        if let FastNetShape::BiCell(cell) = shape {
            bail!(
                "cell type '{}' is not supported on the fast inference path (net 'BLSTM') in the \
                 BIDIRECTIONAL direction when STREAMING; a bidirectional net is unstreamable by \
                 construction (the reverse pass reads the whole sequence) -- the OFFLINE fast \
                 path implements it (Inference_Path fast), streaming does not",
                cell.as_str()
            );
        }
        let causal = matches!(shape, FastNetShape::Causal(_));

        // Input normalization, ARM-DEPENDENT and deliberately CONSERVATIVE:
        //  - type 1 (the pack-carried frozen tail -- the phase-8 causality cut) on EITHER
        //    arm;
        //  - type 0 (NOTHING, the exact path's `_ => {}` arm) on the CAUSAL arm only. It is
        //    causality-trivial (no whole-file statistic exists, so there is nothing to
        //    freeze) and it is what the committed phase-9 causal fixtures carry -- but the
        //    WINDOWED arm keeps phase 8's `{1}` set BYTE-IDENTICALLY, message included, so
        //    no config phase 8 accepted or refused changes status here. Widening it there
        //    too would be defensible and untested; untested is the part that matters.
        //  - type -1 (per-sequence self-normalization) ALWAYS bails: it needs the whole
        //    sequence before the first frame can be normalized.
        let norm_ok = if causal {
            matches!(bc.input_normalization_type, 0 | 1)
        } else {
            bc.input_normalization_type == 1
        };
        if !norm_ok {
            if causal {
                bail!(
                    "streaming requires BLSTM_InputNormalizationType 1 (pack-carried frozen \
                     stats) or 0 (no input normalization); got {} (type -1 self-normalization \
                     needs whole-sequence lookahead)",
                    bc.input_normalization_type
                );
            }
            bail!(
                "streaming requires BLSTM_InputNormalizationType 1 (pack-carried frozen stats); \
                 got {} (type -1 self-normalization needs whole-sequence lookahead)",
                bc.input_normalization_type
            );
        }
        let apply_norm = bc.input_normalization_type == 1;

        // --- Config surfaces (shared with the offline fast driver) ---
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let spec = build_spec_aligned_to(map, "BLSTM", &bc)?;

        // --- Frozen net (loaded once) ---
        let weights_file = map
            .get("BLSTM_weightsFile")
            .map(String::as_str)
            .unwrap_or("");
        if weights_file.is_empty() {
            bail!("streaming: BLSTM_weightsFile is empty (the frozen net must be loaded once)");
        }
        // The FILE seam: the legacy head-first tolerance with its warning, a short file
        // refused, both in the exact `load_weights_file`'s words (issue #62).
        let pack = crate::io::binary::read_weight_vector(std::path::Path::new(weights_file))?;
        let flat = file_pack_head(&pack, exact_pack_len(&bc)?, weights_file)?;
        let causal_net = match shape {
            FastNetShape::Causal(cell) => Some(FastCausalNet::from_flat(
                &spec,
                cell,
                &CellGeometry::from(&bc),
                flat,
            )?),
            FastNetShape::Blstm => None,
            // Unreachable: the bidirectional-cell shape bailed above.
            FastNetShape::BiCell(_) => unreachable!("bidirectional is unstreamable"),
        };
        let blstm_net = match shape {
            FastNetShape::Blstm => Some(FastBlstm::from_flat(&spec, flat)?),
            FastNetShape::Causal(_) => None,
            FastNetShape::BiCell(_) => unreachable!("bidirectional is unstreamable"),
        };

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
        // The windowing regime is TIED TO THE ARM, exactly as it is in the offline fast
        // driver (`fast/driver.rs`'s `causal` dispatch):
        //  - WINDOWED (phase 8): overlap only -- plain (window_size 0) and truncate
        //    (window_shift < 1) typed-bail.
        //  - CAUSAL (Task 7, spec S5.3): the PLAIN regime only. A causal cell inside a
        //    window has its state RESET at every window boundary, and the streaming
        //    session's whole premise is one unbroken carried state -- so a causal config
        //    that resolves a nonzero window is refused STREAMING-side even though the
        //    offline fast driver merely bails it too.
        if causal {
            if window_size != 0 {
                bail!(
                    "streaming: causal streaming requires the plain regime (BLSTM_window \
                     resolves window_size {window_size}); a causal cell's state is reset at \
                     every window boundary -- the phase-9 causal configs use BLSTM_window 0"
                );
            }
        } else {
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
        }

        // timeStep/timeOffset (`tasks/sad.rs:1451-1461`, mirrored at
        // `fast/driver.rs:530-534`): the BASE pair is `_WindowShift`-derived and the
        // OVERLAP branch OVERRIDES it with `_SpectrumShift`. `window_size == 0` -- the
        // causal regime -- keeps the base pair, where `window_shift_sec` is the value
        // `get_blstm_param` just MUTATED (`window_shift * ssif / rate`, and `window_shift`
        // is floored to 1 whenever `window_size == 0`).
        let (time_step, time_offset) = if window_size > 0 {
            let ts = spectrum_shift_sec * ssr as f64;
            (ts, ts / 2.0 - spectrum_shift_sec / 2.0)
        } else {
            let ts = window_shift_sec * ssr as f64;
            (ts, ts / 2.0 - window_shift_sec / 2.0)
        };

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
        let (output_size, norm_mean, norm_std) = match (&causal_net, &blstm_net) {
            (Some(n), _) => (
                n.output_size(),
                n.normalize_mean().to_vec(),
                n.normalize_std().to_vec(),
            ),
            (_, Some(n)) => (
                n.output_size(),
                n.normalize_mean().to_vec(),
                n.normalize_std().to_vec(),
            ),
            // Unreachable: `shape` is a closed two-arm enum and each arm builds its net.
            _ => unreachable!("one net arm is always built"),
        };
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
        let nn = match (causal_net, blstm_net) {
            (Some(n), _) => StreamNn::Causal(StreamCausal::new(n)),
            (_, Some(net)) => StreamNn::Windowed {
                overlap: StreamOverlap::new(window_size, window_shift, ssr, output_size),
                net,
            },
            _ => unreachable!("one net arm is always built"),
        };

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
        // WINDOWED: the full-window lookahead an output row waits for. CAUSAL: 0.0 --
        // there is no window and no lookahead (spec S5.5: "the nn_window term DIES").
        let nn_window_s = 2.0 * window_size as f64 * spectrum_shift_sec;
        // CAUSAL: the sub-sampling BUFFERING delay. Posterior row `p` needs feature rows
        // through `(p+1)*ssr - 1`, i.e. up to `ssr - 1` feature rows past the first of its
        // group, so the wait is at most `(ssr-1) * spectrum_shift_sec`. (The exact wait,
        // measured against the row's own decision TIME `p*dt + time_offset`, is half that
        // -- `time_offset` is precisely `(ssr-1)*spectrum_shift_sec/2` -- so this term is
        // deliberately conservative by one `time_offset`.) WINDOWED: 0.0, the phase-8
        // budget is unchanged, and `x + 0.0 == x` exactly, so the phase-8 bound stays
        // BIT-IDENTICAL (`tests/phase8_gate.rs::latency_bounds` cross-checks it by bits).
        let sub_sample_s = if causal {
            (ssr - 1) as f64 * spectrum_shift_sec
        } else {
            0.0
        };
        let conv_half = driver_cfg
            .conv_coeff
            .as_deref()
            .map_or(0, |c| if c.len() > 1 { (c.len() - 1) / 2 } else { 0 });
        let conv_delay_s = conv_half as f64 * time_step;

        let decision = StreamDecision::new(driver_cfg, seg_cfg, time_step, time_offset);

        Ok(StreamingSession {
            front,
            pipeline,
            nn,
            decision,
            norm_mean,
            norm_std,
            apply_norm,
            rate,
            total_pushed: 0,
            finished: false,
            feature_reach_s,
            nn_window_s,
            sub_sample_s,
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
        self.normalize_rows(&mut tail);
        // WINDOWED: these tail rows make the last UNCLAMPED windows fireable (push_rows),
        // then the overlap flush fires the CLAMPED tail windows -- contiguous posterior-row
        // ranges, fed to the decision layer in order. CAUSAL: `push_rows` is the whole EOS
        // (there is no window flush; a partial sub-sample group is DROPPED, the offline
        // `T mod R` truncation), and `StreamCausal::flush` returns empty by construction.
        let (p1, p2) = match &mut self.nn {
            StreamNn::Windowed { overlap, net } => {
                (overlap.push_rows(&tail, net), overlap.flush(net))
            }
            StreamNn::Causal(c) => (c.push_rows(&tail), c.flush()),
        };
        let mut emitted = self.consume_posteriors(&p1);
        emitted.extend(self.consume_posteriors(&p2));

        let (tail_emitted, seg) = self.decision.flush(audio_duration);
        emitted.extend(tail_emitted);

        let stamped = self.stamp_and_track(emitted, audio_duration);
        (stamped, seg)
    }

    /// Apply the frozen type-1 input normalization to a batch of feature rows, if the
    /// config asked for it. Type 1 is a per-COLUMN affine, so applying it per row is
    /// bit-identical to the offline whole-sequence application (the documented T3/T4 seam);
    /// type 0 does nothing, mirroring the exact path's `_ => {}` arm.
    fn normalize_rows(&self, rows: &mut FastMatrix) {
        if rows.rows > 0 && self.apply_norm {
            external_normalize_f32(rows, &self.norm_mean, &self.norm_std);
        }
    }

    /// Normalize + NN stage + decision for a batch of feature rows (the shared push /
    /// finish body, minus the front-end fetch). Returns UNSTAMPED emissions.
    fn step_rows(&mut self, rows: &mut FastMatrix) -> Vec<EmittedSegment> {
        self.normalize_rows(rows);
        let posts = match &mut self.nn {
            StreamNn::Windowed { overlap, net } => overlap.push_rows(rows, net),
            StreamNn::Causal(c) => c.push_rows(rows),
        };
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

    /// The derived STRUCTURAL latency bound in seconds (spec S1.8 / S5.5): the sum of
    /// front-end reach, NN window finalization, sub-sample buffering, convolution
    /// half-width and the smoothing holdback -- ALL config-derived (never a literal). The
    /// measured per-emission lag is bounded by this plus a data-dependent AREA term
    /// (`tests/phase8_gate.rs`, `tests/phase9_stream_causal.rs`).
    ///
    /// The two arms populate DISJOINT middle terms: the windowed arm carries `nn_window`
    /// with `sub_sample == 0.0`, the causal arm carries `sub_sample` with
    /// `nn_window == 0.0` (spec S5.5: no lookahead exists). Adding an exact `0.0` is the
    /// f64 identity, so the phase-8 bound is bit-unchanged by the new term.
    pub fn derived_latency_bound_s(&self) -> f64 {
        self.feature_reach_s
            + self.nn_window_s
            + self.sub_sample_s
            + self.conv_delay_s
            + self.decision.holdback()
    }

    /// The feature-extraction reach component `(reach*shift_frames + half_window)/rate`.
    pub fn feature_reach_s(&self) -> f64 {
        self.feature_reach_s
    }
    /// The NN-window-finalization component `2*window_size_feature * spectrum_shift_sec`
    /// (the full-window lookahead the overlap incurs before an output row finalizes).
    /// EXACTLY `0.0` on the causal arm -- a causal cell has no lookahead (spec S5.5).
    pub fn nn_window_s(&self) -> f64 {
        self.nn_window_s
    }
    /// The sub-sampling buffering component `(ssr-1) * spectrum_shift_sec` (a posterior
    /// row waits for the last feature row of its decimation group). EXACTLY `0.0` on the
    /// windowed arm, whose far larger `nn_window` already dominates it.
    pub fn sub_sample_s(&self) -> f64 {
        self.sub_sample_s
    }
    /// `true` when this session runs the Phase-9 causal per-row stack, `false` for the
    /// phase-8 windowed BLSTM path (a test/report observable for the S5.2 dispatch).
    pub fn is_causal(&self) -> bool {
        matches!(self.nn, StreamNn::Causal(_))
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
