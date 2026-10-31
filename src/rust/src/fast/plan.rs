//! `fast::plan` -- the ONE algo-3 SAD setup for the fast tree (issue #24).
//!
//! Before this module the fast tree set up an algo-3 run twice: the offline driver
//! (`fast/driver.rs::FastSpectralSegmenter`) and the streaming session
//! (`fast/stream.rs::StreamingSession`) each repeated the config structs, the shape
//! classification, the peephole-aligned spec, the pack length, the normalization gate,
//! the `SpectralParams` + `get_blstm_param` derivation, the windowing-regime rule and the
//! `time_step`/`time_offset` formula -- and the two copies had already disagreed (the
//! normalization sets, and the pitch gate the stream lacked). Now both consume one
//! [`FastSadPlan`] and one [`SadTimeline`]; the plan is the shared INPUT, the step kernel
//! the shared ARITHMETIC (ADR-0004: the bit-equal gates `phase8_gate.rs` /
//! `phase9_stream_causal.rs` are the proof that nothing arithmetic moved).
//!
//! TWO VALUES, BECAUSE SETUP HAPPENS AT TWO TIMES. The offline driver learns the sample
//! rate with the audio, per file; the session learns it at construction. So:
//!
//! - [`FastSadPlan::from_map`] is PURE, RATE-FREE and does NO I/O: the three config
//!   structs, the [`FastNetShape`], the spec, the [`CellGeometry`], the normalization
//!   type, the exact pack length, the fixed gain and the weight-file key -- plus every
//!   check that needs no rate (algo 3, the pitch gate, the fast-tree-wide normalization
//!   set, a rate-free windowing pre-check). It fails at `BagOfProcessors::from_configs`
//!   instead of the first `get_segmentation`.
//! - [`FastSadPlan::timeline`] derives, at one rate, what the exact driver derives per
//!   file: `SpectralParams`, the quantized spectrum shift, `get_blstm_param`'s window /
//!   shift / overlap flag, the AUTHORITATIVE post-quantization regime check, and the
//!   `(time_step, time_offset)` pair. [`FastSadPlan::rows_for`] is the only
//!   frame-count-dependent output (`real_vec_size`).
//!
//! THE STATEFUL SHIFT QUANTIZATION IS GONE FROM THE FAST TREE -- a by-design fast
//! divergence, recorded here and in RESULTS.md (never IMPROVEMENTS.md: nothing legacy was
//! reproduced wrongly, a legacy carry was dropped on purpose). The exact driver
//! (`tasks/sad.rs`, frozen) re-quantizes `_SpectrumShift`, `_WindowShift` and
//! `_LTSVWindowShift` IN PLACE on every call, from the previous call's quantized value. At
//! one rate that is a fixed point -- `round(round(x*r)/r*r) == round(x*r)`, pinned by
//! `tests/fast_sad_plan.rs` -- so a run whose files share a rate sees no difference; the
//! carry is observable only across channels of DIFFERENT rates in one run, where the
//! second file's shift would be quantized off the first's. The timeline derives from the
//! config value every time, which is what the streaming session always did
//! (`params.shift_sec` off `SpectralParams::derive`) and what makes offline and streamed
//! setups one function.
//!
//! THE NORMALIZATION SETS ARE ADJUDICATED PER ARM, in one table
//! ([`accepted_norm_types`]), each with a test and a RESULTS.md note:
//! - the offline fast driver keeps `{-1, 0, 1}` (type -1 self-normalization needs the
//!   whole sequence, which the offline driver has);
//! - the stream's causal arm keeps `{0, 1}` (type -1 is acausal);
//! - the stream's windowed arm WIDENS `{1}` -> `{0, 1}`: type 0 is `apply_norm = false`
//!   and nothing else differs between the arms, so phase 8's `{1}` was an unextended
//!   gate, not a design; `phase8_gate.rs::stream_finish_equals_offline_type0` is the leg
//!   that earns it (windowed BLSTM, type 0, bit-equal to the offline fast run).
//!
//! Type -2 (the plain-FFB self-normalized COPY) is refused fast-tree-wide at `from_map`.
//!
//! THE PITCH GATE IS INHERITED BY THE STREAM. The offline driver refused
//! `BLSTM_TDCwindow > 0` (spec R4); the stream did not, so a pitch config streamed
//! silently without its second pass and was NOT bit-equal to the offline run it claims to
//! mirror. One gate, in `from_map`, both consumers.

use anyhow::{Result, anyhow, bail};
use indexmap::IndexMap;

use crate::config::NnetSpec;
use crate::features::pipeline::{FeatureConfig, SpectralParams};
use crate::legacy_config::{get_f64, get_f64_default, get_f64_opt};
use crate::nn::blstm::{BlstmConfig, check_pack_len};
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmenter::{DriverConfig, SegmenterConfig};

use super::cells::CellGeometry;
use super::driver::{
    FastNet, FastNetShape, LidScoreParams, build_net, build_spec_aligned_to, classify_fast_shape,
    exact_pack_len, read_file_pack,
};

/// The consumer a [`FastSadPlan`] is checked for, where the arms differ: the input
/// normalization set ([`accepted_norm_types`]). The regime rule does NOT differ per arm
/// (it follows the shape); the stream's extra rules (mono, `Audio_fixed_gain`, binary
/// output, no bidirectional new cell) are the session's own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FastSadArm {
    /// `FastSpectralSegmenter`: the whole sequence is in hand.
    Offline,
    /// `StreamingSession` over the windowed bidirectional LSTM (`StreamOverlap`).
    StreamWindowed,
    /// `StreamingSession` over a causal cell stack (`StreamCausal`).
    StreamCausal,
}

/// The `BLSTM_InputNormalizationType` values each arm runs -- THE table (issue #24's
/// adjudication; the module docs carry the reasons). `-1` is per-sequence
/// self-normalization (whole-sequence lookahead: offline only), `1` the pack-carried
/// frozen tail (a per-column affine, streams), `0` nothing (the exact path's `_ => {}`
/// arm, streams trivially). `-2` is in no set.
pub fn accepted_norm_types(arm: FastSadArm) -> &'static [i16] {
    match arm {
        FastSadArm::Offline => &[-1, 0, 1],
        FastSadArm::StreamWindowed => &[0, 1],
        FastSadArm::StreamCausal => &[0, 1],
    }
}

/// The fast-tree-wide normalization set: the union of the arms'. Checked once at
/// [`FastSadPlan::from_map`], so a `-2` config fails at bag construction on every arm.
const FAST_TREE_NORM_TYPES: [i16; 3] = [-1, 0, 1];

/// The per-prefix NET reading both plans share (issue #84): one `BLSTM`/`BLSTM_LID`
/// prefix read into the peephole-aligned spec, the fast shape, the three cell
/// geometries, the normalization type, the exact pack length, the whole-net
/// sub-sampling ratio and the weight-file key. [`FastSadPlan`] reads one (`BLSTM`) and
/// flattens it into its own fields; the LID plan holds two. The `BlstmConfig` is the
/// caller's because each plan reads its own gates off it.
#[derive(Clone, Debug)]
pub struct FastNetPlan {
    /// The peephole-aligned `NnetSpec` the fast net is built from
    /// (`build_spec_aligned_to`).
    pub spec: NnetSpec,
    /// `classify_fast_shape`: which fast kernel runs it.
    pub shape: FastNetShape,
    /// The three port-only cell geometries, inert unless `shape` names their cell.
    pub geometry: CellGeometry,
    /// `<prefix>_InputNormalizationType`, unchecked here (each plan owns its set).
    pub input_normalization_type: i16,
    /// `exact_pack_len`: what every pack, in-memory or file, is checked against.
    pub pack_len: usize,
    /// The whole-net sub-sampling ratio (`getSubSamplingRatio`): the product of every
    /// recurrent and output-layer ratio.
    pub ssr: usize,
    /// `<prefix>_weightsFile`, `""` when absent or empty (the legacy `size() != 0` guard).
    pub weights_file: String,
}

impl FastNetPlan {
    /// Read the `prefix` net off `map` and its already-parsed `bc`
    /// (`BlstmConfig::from_legacy(map, prefix)`). Pure, I/O-free, no gate.
    pub fn from_map(
        map: &IndexMap<String, String>,
        prefix: &str,
        bc: &BlstmConfig,
    ) -> Result<FastNetPlan> {
        let shape = classify_fast_shape(bc);
        let spec = build_spec_aligned_to(map, prefix, bc)?;
        let ssr = spec.lstm_subsampling.iter().product::<usize>()
            * spec.output_subsampling.iter().product::<usize>();
        let pack_len = exact_pack_len(bc)?;
        let weights_file = map
            .get(&format!("{prefix}_weightsFile"))
            .map(String::as_str)
            .unwrap_or("")
            .to_string();
        Ok(FastNetPlan {
            spec,
            shape,
            geometry: CellGeometry::from(bc),
            input_normalization_type: bc.input_normalization_type,
            pack_len,
            ssr,
            weights_file,
        })
    }

    /// Build the fast net from an in-memory pack: EXACT-LENGTH against [`Self::pack_len`]
    /// in the exact `set_weights`' words (`check_pack_len`), then `build_net` for
    /// [`Self::shape`].
    pub(crate) fn build_net(&self, flat: &[f64]) -> Result<FastNet> {
        check_pack_len(flat.len(), self.pack_len)?;
        build_net(&self.spec, self.shape, &self.geometry, flat)
    }

    /// `<prefix>_weightsFile` read under the legacy FILE-LOAD tolerance ([`read_file_pack`]:
    /// a short file is refused, a long one warns on stderr and keeps its head, cut to
    /// [`Self::pack_len`]); `None` for an empty key.
    pub fn read_weights_file(&self) -> Result<Option<Vec<f64>>> {
        read_file_pack(&self.weights_file, self.pack_len)
    }
}

/// `Algo_choice`, the one key both plans dispatch on; `who` prefixes a parse error.
fn read_algo(map: &IndexMap<String, String>, who: &str) -> Result<i32> {
    map.get("Algo_choice")
        .ok_or_else(|| anyhow!("param 'Algo_choice' not found in config"))?
        .trim()
        .parse::<i32>()
        .map_err(|e| anyhow!("{who}: Algo_choice parse: {e}"))
}

/// The three `BLSTM` config structs every fast plan reads (the SAD net's segmenter,
/// driver and feature config; on the Twin the SAD side is never run but its decision
/// layer, time axis and pre-emphasis gates read the same three).
fn blstm_configs(
    map: &IndexMap<String, String>,
) -> Result<(SegmenterConfig, DriverConfig, FeatureConfig)> {
    Ok((
        SegmenterConfig::from_config(map, "BLSTM")?,
        DriverConfig::from_config(map, "BLSTM")?,
        FeatureConfig::from_legacy(map, "BLSTM")?,
    ))
}

/// Everything an algo-3 fast SAD run needs that does not depend on the sample rate or on
/// the audio: the rate-free half of the setup, parsed ONCE from the config map by
/// [`Self::from_map`] and consumed by the offline driver and the streaming session alike.
/// The rate-dependent half is [`Self::timeline`].
#[derive(Clone, Debug)]
pub struct FastSadPlan {
    pub feature_cfg: FeatureConfig,
    pub seg_cfg: SegmenterConfig,
    pub driver_cfg: DriverConfig,
    /// The peephole-aligned `NnetSpec` the fast net is built from
    /// (`build_spec_aligned_to`).
    pub spec: NnetSpec,
    /// `classify_fast_shape` on the `BLSTM` net: which fast kernel runs it.
    pub shape: FastNetShape,
    /// The three port-only cell geometries, inert unless `shape` names their cell.
    pub geometry: CellGeometry,
    /// `BLSTM_InputNormalizationType`, inside [`FAST_TREE_NORM_TYPES`]; the arm's own
    /// subset is checked by [`Self::check_norm`].
    pub input_normalization_type: i16,
    /// `exact_pack_len`: what every pack, in-memory or file, is checked against.
    pub pack_len: usize,
    /// `Audio_fixed_gain`, the frozen audio gain (phase 8 S1.1). The bag applies it at
    /// `read_audio` for the offline driver; the stream requires it (the causality cut).
    pub fixed_gain: Option<f64>,
    /// `BLSTM_weightsFile`, `""` when absent or empty (the legacy `size() != 0` guard).
    pub weights_file: String,
    /// The whole-net sub-sampling ratio (`getSubSamplingRatio`): the product of every
    /// recurrent and output-layer ratio.
    pub ssr: usize,
}

impl FastSadPlan {
    /// Parse the rate-free setup from the SAME legacy config `map` the bag consumes. Pure
    /// and I/O-free: the weight file is only NAMED here (read by [`Self::read_weights_file`]).
    ///
    /// Refuses, in the exact words the gates pin: `Algo_choice != 3`; the pitch second pass
    /// (`BLSTM_TDCwindow > 0`, spec R4); a normalization type outside the fast tree's
    /// `{-1, 0, 1}`; and the rate-free windowing pre-check ([`Self::precheck_regime`]).
    pub fn from_map(map: &IndexMap<String, String>) -> Result<FastSadPlan> {
        let algo = read_algo(map, "fast SAD plan")?;
        if algo != 3 {
            bail!("fast SAD plan: only algo 3 (spectral SAD) is supported (got {algo})");
        }

        let (seg_cfg, driver_cfg, feature_cfg) = blstm_configs(map)?;

        // Pitch second pass unsupported on the fast path (spec R4). tier2 + the phase-6
        // SAD config use TDCwindow 0. One gate for both consumers: the stream used to
        // lack it and streamed a pitch config without its second pass.
        if feature_cfg.tdc_window > 0.0 {
            bail!(
                "fast SAD: the pitch second pass (BLSTM_TDCwindow > 0) is not supported on the \
                 fast inference path (spec R4); the gate configs use TDCwindow 0"
            );
        }

        // Cell x direction dispatch (Phase 9 Task 6, spec S4.2) + peephole-default
        // alignment (rider 1). Every pair maps to a shape since phase-10 Task 8.
        let bc = BlstmConfig::from_legacy(map, "BLSTM")?;
        let net = FastNetPlan::from_map(map, "BLSTM", &bc)?;

        // The fast-tree-wide set. Type 0 is the exact path's `_ => {}` arm (the
        // `nn::blstm::BlstmNetwork::feed_forward_backward` normalization match): a
        // genuine no-op on BOTH sides. -2 (the plain-FFB self-normalized COPY) is out.
        if !FAST_TREE_NORM_TYPES.contains(&bc.input_normalization_type) {
            bail!(
                "fast SAD: only InputNormalizationType -1 (self-normalization), 1 (external \
                 pack-carried mean/std, the phase-8 frozen mode) or 0 (none) are supported on \
                 the fast path (got {})",
                bc.input_normalization_type
            );
        }

        let fixed_gain = get_f64_opt(map, "Audio_fixed_gain")?;

        let FastNetPlan {
            spec,
            shape,
            geometry,
            input_normalization_type,
            pack_len,
            ssr,
            weights_file,
        } = net;
        let plan = FastSadPlan {
            feature_cfg,
            seg_cfg,
            driver_cfg,
            spec,
            shape,
            geometry,
            input_normalization_type,
            pack_len,
            fixed_gain,
            weights_file,
            ssr,
        };
        plan.precheck_regime()?;
        Ok(plan)
    }

    /// The input-normalization check for ONE arm ([`accepted_norm_types`]). Only a stream
    /// arm can refuse here: the offline set IS the fast-tree-wide set `from_map` enforced,
    /// so the offline driver does not call this. The words are the ones
    /// `phase8_gate.rs` / `phase9_stream_causal.rs` pin.
    pub fn check_norm(&self, arm: FastSadArm) -> Result<()> {
        let t = self.input_normalization_type;
        if accepted_norm_types(arm).contains(&t) {
            return Ok(());
        }
        bail!(
            "streaming requires BLSTM_InputNormalizationType 1 (pack-carried frozen stats) or 0 \
             (no input normalization); got {t} (type -1 self-normalization needs whole-sequence \
             lookahead)"
        )
    }

    /// The windowing-regime rule, ONE body for the rate-free pre-check and the timeline's
    /// authoritative check: BLSTM runs OVERLAP only, a causal cell runs PLAIN only (its
    /// state would reset at every window boundary), a bidirectional cell runs either;
    /// TRUNCATE is refused on every shape. `windowed` / `no_overlap` are the caller's
    /// resolution, `at` says how it was resolved (the pinned clauses are the fixed text).
    fn check_regime(&self, windowed: bool, no_overlap: bool, at: &str) -> Result<()> {
        let w = self.driver_cfg.window_size_sec;
        let s = self.driver_cfg.window_shift_sec;
        match self.shape {
            FastNetShape::Causal(_) if windowed => bail!(
                "fast causal SAD: windowed inference is unsupported (BLSTM_window {w} resolves a \
                 nonzero window{at}); a causal cell's state is reset at every window boundary \
                 (spec S1.2), so causal streaming requires the plain regime and so does the \
                 offline fast driver -- the phase-9 causal configs use BLSTM_window 0"
            ),
            FastNetShape::Blstm if !windowed => bail!(
                "fast SAD: the plain (non-windowed) forward is unsupported for the bidirectional \
                 LSTM (BLSTM_window {w} resolves window_size 0{at}); the gate configs use windowed \
                 overlap"
            ),
            FastNetShape::Blstm | FastNetShape::BiCell(_) if no_overlap => bail!(
                "fast SAD: the truncate (non-overlap) windowing is unsupported (BLSTM_shift {s}: \
                 window_shift resolves < 1{at}); the fast tree implements the plain and OVERLAP \
                 regimes (spec S5)"
            ),
            _ => Ok(()),
        }
    }

    /// The RATE-FREE pre-check: what the config's seconds settle before any rate is known.
    /// `get_blstm_param` resolves `window_size = round(window_size_sec * rate / 2 / ssif)`
    /// and `window_shift = round(window_shift_sec * rate / ssif)`, so a ZERO
    /// `window_size_sec` is the plain regime at every rate and a ZERO `window_shift_sec`
    /// under a nonzero window is the truncate regime at every rate. A causal config with a
    /// NONZERO window is refused here too, on the config's intent, even when a tiny window
    /// would quantize to 0 at some rate (the exact tree would run that plain).
    /// [`Self::timeline`] re-checks on the resolved values (a sub-frame shift that rounds
    /// to 0 at the actual rate is caught there).
    fn precheck_regime(&self) -> Result<()> {
        let windowed = self.driver_cfg.window_size_sec > 0.0;
        let no_overlap = windowed && self.driver_cfg.window_shift_sec == 0.0;
        self.check_regime(windowed, no_overlap, " at every rate")
    }

    /// The rate-dependent half of the setup, at `rate`: what the exact driver derives per
    /// file (`tasks::sad::BlstmSpectralSegmenter::get_segmentation`), PURE and from the
    /// config values (see the module docs on the dropped stateful carry). Runs the
    /// authoritative regime check on the RESOLVED window/shift.
    pub fn timeline(&self, rate: f64) -> Result<SadTimeline> {
        // `SpectralParams::derive` quantizes the spectrum shift the way the exact driver's
        // stateful member does on its first call: `shift_frames = round(shift_sec * rate)`,
        // `shift_sec = shift_frames / rate`.
        let params = SpectralParams::derive(&self.feature_cfg, rate);
        let ssif = params.shift_frames;
        let spectrum_shift_sec = params.shift_sec;

        // The one SAD-axis body ([`sad_axis`]): `get_blstm_param` off the config values,
        // then the authoritative regime check on the RESOLVED values (the pre-check's,
        // re-run), then the time axis. `frame_count` 0: only `rows` depends on it
        // ([`Self::rows_for`]).
        let axis = sad_axis(
            &self.driver_cfg,
            &self.spec,
            self.ssr,
            rate,
            ssif,
            spectrum_shift_sec,
            0,
        );
        self.check_regime(
            axis.window_size != 0,
            axis.no_overlap,
            &format!(" at {rate} Hz"),
        )?;
        let SadAxis {
            window_shift_sec,
            window_size,
            window_shift,
            time_step,
            time_offset,
            ..
        } = axis;

        Ok(SadTimeline {
            rate,
            params,
            ssif,
            spectrum_shift_sec,
            window_shift_sec,
            window_size,
            window_shift,
            time_step,
            time_offset,
        })
    }

    /// `real_vec_size` for a file of `frame_count` samples on timeline `tl`:
    /// `get_blstm_param`'s `ceil(frame_count / ssif)` divided sequentially by every
    /// sub-sampling ratio -- the ONE frame-count-dependent output, re-run through
    /// [`sad_axis`] rather than re-derived here (the window/shift it also returns are `tl`'s).
    pub fn rows_for(&self, tl: &SadTimeline, frame_count: usize) -> usize {
        sad_axis(
            &self.driver_cfg,
            &self.spec,
            self.ssr,
            tl.rate,
            tl.ssif,
            tl.spectrum_shift_sec,
            frame_count,
        )
        .rows
    }

    /// `BLSTM_weightsFile` read under the legacy FILE-LOAD tolerance ([`read_file_pack`]:
    /// a short file is refused, a long one warns on stderr and keeps its head, cut to
    /// [`Self::pack_len`]); `None` for an empty key. The ONE file seam of the fast tree
    /// (issue #62 / #65): the offline driver's deferred load and the session's construction
    /// both read through here.
    pub fn read_weights_file(&self) -> Result<Option<Vec<f64>>> {
        read_file_pack(&self.weights_file, self.pack_len)
    }

    /// Build the fast net from an in-memory pack: EXACT-LENGTH against [`Self::pack_len`]
    /// in the exact `set_weights`' words (`check_pack_len`), then `build_net` for
    /// [`Self::shape`].
    pub(crate) fn build_net(&self, flat: &[f64]) -> Result<FastNet> {
        check_pack_len(flat.len(), self.pack_len)?;
        build_net(&self.spec, self.shape, &self.geometry, flat)
    }
}

/// The SAD window resolution and the decision layer's time axis at one rate, ONE body for
/// both plans (issue #84): the exact spectral driver's per-file derivation
/// (`tasks::sad::BlstmSpectralSegmenter::get_segmentation`, the block it cites as legacy
/// `BLSTMSpectralSegmenter.cpp:724-734`), PURE and from the config values (the window
/// shift a local, never the carried `_WindowShift`: the dropped carry of the module docs).
/// `ssif` / `spectrum_shift_sec` are the caller's: `SpectralParams::derive`'s for the SAD
/// plan, the Mode-7 constant 80 for the LID plan. The BASE `(time_step, time_offset)` pair
/// is `window_shift_sec`-derived; the OVERLAP branch OVERRIDES it with the spectrum shift
/// (the asymmetry vs the signal driver); truncate keeps the base pair, where
/// `window_shift_sec` is the value `get_blstm_param` just quantized (floored to 1 at
/// window 0). `rows` is the only frame-count-dependent output.
pub(crate) fn sad_axis(
    driver_cfg: &DriverConfig,
    spec: &NnetSpec,
    ssr: usize,
    rate: f64,
    ssif: usize,
    spectrum_shift_sec: f64,
    frame_count: usize,
) -> SadAxis {
    let mut window_shift_sec = driver_cfg.window_shift_sec;
    let (window_size, window_shift, no_overlap, rows) = get_blstm_param(
        driver_cfg.window_size_sec,
        &mut window_shift_sec,
        rate,
        ssif,
        ssr,
        &spec.lstm_subsampling,
        &spec.output_subsampling,
        frame_count,
    );
    let ssr = ssr as f64;
    let (time_step, time_offset) = if window_size > 0 && !no_overlap {
        let ts = spectrum_shift_sec * ssr;
        (ts, ts / 2.0 - spectrum_shift_sec / 2.0)
    } else {
        let ts = window_shift_sec * ssr;
        (ts, ts / 2.0 - window_shift_sec / 2.0)
    };
    SadAxis {
        window_shift_sec,
        window_size,
        window_shift,
        no_overlap,
        rows,
        time_step,
        time_offset,
    }
}

/// What [`sad_axis`] resolves at one rate: `get_blstm_param`'s outputs and the decision
/// layer's time axis. [`SadTimeline`] flattens one into its fields (with the periodogram
/// params beside it); the LID plan's SAD half IS one.
#[derive(Clone, Debug)]
pub struct SadAxis {
    /// The window shift in seconds AFTER `get_blstm_param`'s quantization
    /// (`window_shift * ssif / rate`): the exact driver's post-call `_WindowShift`.
    pub window_shift_sec: f64,
    /// `get_blstm_param`'s window half-size in periodogram frames (0 = plain regime).
    pub window_size: usize,
    /// `get_blstm_param`'s window shift in periodogram frames (floored to 1).
    pub window_shift: usize,
    /// `get_blstm_param`'s truncate flag (a resolved shift below one frame under a window).
    pub no_overlap: bool,
    /// `real_vec_size` for `frame_count` samples: `ceil(frame_count / ssif)` divided
    /// sequentially by every sub-sampling ratio; 0 for `frame_count` 0.
    pub rows: usize,
    /// The decision layer's time axis (`results_to_segmentation`'s pair).
    pub time_step: f64,
    pub time_offset: f64,
}

/// The rate-dependent setup of an algo-3 fast SAD run ([`FastSadPlan::timeline`]): the
/// periodogram framing, the quantized shifts, the resolved window and the decision-layer
/// time axis. The offline driver derives one per file; the session one at construction.
/// [`FastSadPlan::rows_for`] sizes the result vector on it.
#[derive(Clone, Debug)]
pub struct SadTimeline {
    pub rate: f64,
    /// The periodogram/mel framing at `rate` (`SpectralParams::derive`), fed to
    /// `FastPipeline::new` / `StreamFrontEnd::new` with the SAME `rate` that derived it.
    pub params: SpectralParams,
    /// The quantized spectrum shift in samples (`params.shift_frames`).
    pub ssif: usize,
    /// The quantized spectrum shift in seconds (`params.shift_sec == ssif / rate`).
    pub spectrum_shift_sec: f64,
    /// The window shift in seconds AFTER `get_blstm_param`'s quantization
    /// (`window_shift * ssif / rate`): the exact driver's post-call `_WindowShift`.
    pub window_shift_sec: f64,
    /// `get_blstm_param`'s window half-size in periodogram frames (0 = plain regime).
    pub window_size: usize,
    /// `get_blstm_param`'s window shift in periodogram frames (floored to 1).
    pub window_shift: usize,
    /// The decision layer's time axis (`results_to_segmentation`'s pair).
    pub time_step: f64,
    pub time_offset: f64,
}

// ===========================================================================
// The LID plan (issue #84): the ONE algo-6 Mode-7 setup of the fast tree, consumed by
// the offline fast Twin (`fast::driver::FastTwinLid`) and the Utterance stream
// (`fast::stream_lid::StreamingLidSession`).
// ===========================================================================

/// Read one `i32` config scalar with a default (the Twin's reader, a copy of `tasks/lid.rs`'s; #87 retires both).
fn twin_i32_default(map: &IndexMap<String, String>, key: &str, default: i32) -> Result<i32> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<i32>()
            .map_err(|e| anyhow!("`{key}`: cannot parse: {e}")),
    }
}

/// Read one `bool` config scalar with a default (the Twin's reader, a copy of `tasks/lid.rs`'s; #87 retires both).
fn twin_bool_default(map: &IndexMap<String, String>, key: &str, default: bool) -> Result<bool> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<bool>()
            .map_err(|e| anyhow!("`{key}`: cannot parse as bool: {e}")),
    }
}

/// The Mode-7 `_SpectrumShiftInFrames` (`BLSTMSpectralSegmenter.cpp:209`): the
/// `!hasReadWavFile` override, always taken on the phSeq/cep arm the fast Twin runs, so the
/// LID plan's `ssif` is this constant at every rate.
pub const MODE7_SSIF: usize = 80;

/// Everything an algo-6 Mode-7 fast LID run needs that does not depend on the sample
/// rate, the target language or the audio: the rate-free half of the Twin's setup, parsed
/// ONCE from the config map by [`Self::from_map`] and consumed by the offline Twin and the
/// Utterance stream alike. The rate-dependent half is [`Self::timeline`].
///
/// The SAD side is the FROZEN-SAD one (CONTEXT.md): the SAD net is never run, so its plan
/// is read for its sub-sampling (the SAD result-vector sizing and time axis) and its pack
/// length (an in-memory SAD pack and the `BLSTM_weightsFile` are checked against it, as
/// the exact Twin checks them, issue #65) and nothing of it is gated -- no shape rule, no
/// normalization set, no windowing regime (every committed Twin config carries
/// `BLSTM_shift 0`, the truncate the SAD plan refuses).
#[derive(Clone, Debug)]
pub struct FastLidPlan {
    pub feature_cfg: FeatureConfig,
    pub seg_cfg: SegmenterConfig,
    pub driver_cfg: DriverConfig,
    /// The never-run SAD net (`BLSTM`): sub-sampling and pack length only.
    pub sad: FastNetPlan,
    /// The LID net (`BLSTM_LID`), the one that runs.
    pub lid: FastNetPlan,
    /// `max(2, the LID output width)`: the langID / confusion dimensionality.
    pub class_nb: usize,
    /// `BLSTM_LID_TwoSweeps`, read by the truncate sweep.
    pub two_sweeps: bool,
    /// `CostLaw::isCostModified` (`_BackPropWER >= 0`), gating the langID normalization.
    pub cost_modified: bool,
    /// `BLSTM_LID_window` (`_LIDWindowSize`, clamped `< 0 -> 0`).
    pub lid_window_size_sec: f64,
    /// `BLSTM_LID_shift` (`_LIDWindowShift`'s seed, clamped `< 0 -> 0`); the exact
    /// Twin's stateful re-quantization of it is NOT carried (the module docs).
    pub lid_window_shift_sec: f64,
    /// `BLSTM_LID_PostProcessMode`, `BLSTM_LID_MinNbOfFrames`, `BLSTM_LID_NoiseMagnitude`.
    pub post_process_mode: i32,
    pub min_nb_of_frames: i32,
    pub noise_magnitude: f64,
}

impl FastLidPlan {
    /// Parse the rate-free setup from the SAME legacy config `map` the bag consumes. Pure
    /// and I/O-free: the two weight files are only NAMED here
    /// ([`Self::read_weights_files`]).
    ///
    /// Typed-bails, in the words the legs pin: `Algo_choice != 6`; any `BLSTM_LID_Mode`
    /// but 7; the pitch second pass (`BLSTM_TDCwindow > 0`); `BLSTM_LID_DumpInternals`;
    /// a LID `InputNormalizationType != 0` (the fast scoring applies no normalization);
    /// a negative `BLSTM_LID_TargetEnforcementStep` (the exact cost block overwrites
    /// interior output rows with -0.5 for it, risk R7, which forward-only scoring never
    /// reproduces); MLP mode (`BLSTM_LID_LSTMNeuronNb[0] == 0`, the exact dispatch's
    /// `is_mlp` route to the MLP drivers, refused on the same derived flag so construction
    /// fails whether or not weights arrive); and the LID regime rule, rate-free
    /// ([`Self::precheck_regime`]).
    pub fn from_map(map: &IndexMap<String, String>) -> Result<FastLidPlan> {
        let algo = read_algo(map, "fast LID plan")?;
        if algo != 6 {
            bail!("fast LID plan: only algo 6 (Twin LID) is supported (got {algo})");
        }

        let (seg_cfg, driver_cfg, feature_cfg) = blstm_configs(map)?;

        let mode = twin_i32_default(map, "BLSTM_LID_Mode", 0)?;
        if mode != 7 {
            bail!(
                "fast TwinLid: only Mode 7 (phSeq/cep external-features LID) is supported on the \
                 fast path (got {mode}); the gate configs use Mode 7"
            );
        }

        // Pitch second pass unsupported (spec R4); the gate configs use TDCwindow 0.
        if feature_cfg.tdc_window > 0.0 {
            bail!(
                "fast TwinLid: the pitch second pass (BLSTM_TDCwindow > 0) is not supported on the \
                 fast inference path; the gate configs use TDCwindow 0"
            );
        }

        if twin_bool_default(map, "BLSTM_LID_DumpInternals", false)? {
            bail!(
                "fast TwinLid: BLSTM_LID_DumpInternals is unsupported on the fast path (the LID \
                 hidden states the dump emits are not tracked); the gate configs use false"
            );
        }

        // SAD net shape only (never run in Mode 7): its sub-sampling ratios size the SAD
        // result vector, and every cell carries those. Peephole-aligned spec, no gate.
        let sad_bc = BlstmConfig::from_legacy(map, "BLSTM")?;
        let sad = FastNetPlan::from_map(map, "BLSTM", &sad_bc)?;

        // LID net (the one that runs). Only InputNormalizationType 0 (the gate value) is
        // supported: the fast scoring applies no normalization.
        let lid_bc = BlstmConfig::from_legacy(map, "BLSTM_LID")?;
        if lid_bc.input_normalization_type != 0 {
            bail!(
                "fast TwinLid: only LID InputNormalizationType 0 is supported on the fast path \
                 (got {}); the gate configs use 0",
                lid_bc.input_normalization_type
            );
        }
        if lid_bc.target_enforcement_step < 0 {
            bail!(
                "fast TwinLid: a negative BLSTM_LID_TargetEnforcementStep is not supported on \
                 the fast path (got {}); the exact path's cost block overwrites interior \
                 output rows with -0.5 for a negative step (risk R7), \
                 which the fast path's forward-only scoring never reproduces; the gate configs \
                 use a non-negative step",
                lid_bc.target_enforcement_step
            );
        }
        if lid_bc.is_mlp {
            bail!(
                "fast TwinLid: MLP mode (BLSTM_LID_LSTMNeuronNb[0] == 0) is not supported on \
                 the fast path (the fast LID scoring runs the recurrent forwards only, never \
                 the MLP drivers); the gate configs use a non-zero LSTM width"
            );
        }
        let lid = FastNetPlan::from_map(map, "BLSTM_LID", &lid_bc)?;
        let class_nb = (*lid.spec.output_neuron_nb.last().unwrap()).max(2);

        let mut lid_window_size_sec = get_f64(map, "BLSTM_LID_window")?;
        if lid_window_size_sec < 0.0 {
            lid_window_size_sec = 0.0;
        }
        let mut lid_window_shift_sec = get_f64(map, "BLSTM_LID_shift")?;
        if lid_window_shift_sec < 0.0 {
            lid_window_shift_sec = 0.0;
        }
        let post_process_mode = twin_i32_default(map, "BLSTM_LID_PostProcessMode", 0)?;
        let min_nb_of_frames = twin_i32_default(map, "BLSTM_LID_MinNbOfFrames", 0)?;
        let noise_magnitude = get_f64_default(map, "BLSTM_LID_NoiseMagnitude", 0.0)?;

        let plan = FastLidPlan {
            feature_cfg,
            seg_cfg,
            driver_cfg,
            sad,
            lid,
            class_nb,
            two_sweeps: lid_bc.two_sweeps,
            cost_modified: lid_bc.cost_law.is_cost_modified(),
            lid_window_size_sec,
            lid_window_shift_sec,
            post_process_mode,
            min_nb_of_frames,
            noise_magnitude,
        };
        plan.precheck_regime()?;
        Ok(plan)
    }

    /// The LID regime rule (issue #57, D2), ONE body for the offline Twin and the Utterance
    /// stream, decided RATE-FREE on the config's intent (issue #84): a ZERO `BLSTM_LID_window`
    /// is the plain regime at every rate, a nonzero one under a ZERO `BLSTM_LID_shift` is
    /// truncate at every rate (what every committed LID config carries), and a nonzero shift
    /// under a nonzero window is OVERLAP on intent. A CAUSAL shape runs the PLAIN regime only
    /// (a causal cell inside a window has its state reset at every window boundary, the SAD
    /// driver's rule; the launcher writes `BLSTM_LID_window 0` on a forward LID net); every
    /// shape refuses OVERLAP (the fast LID scoring implements the truncate sweep and the
    /// plain forward, nothing accumulates across overlapping windows); a bidirectional shape
    /// runs either plain or truncate, exactly as the exact tree dispatches them. Raised at
    /// `BagOfProcessors::from_configs` / the session's construction. Stricter than the
    /// resolved regime where a sub-frame value quantizes to zero: a sub-frame positive shift
    /// the exact tree would quantize to truncate is refused as overlap here, and a sub-frame
    /// positive window it would quantize to plain is refused as windowed (causal) or as
    /// overlap (under a shift), so [`Self::timeline`] has nothing left to refuse and is
    /// infallible (the pinned clauses are the fixed text).
    fn precheck_regime(&self) -> Result<()> {
        let w = self.lid_window_size_sec;
        let s = self.lid_window_shift_sec;
        if let FastNetShape::Causal(cell) = self.lid.shape
            && w > 0.0
        {
            bail!(
                "fast LID plan: windowed LID inference is unsupported for a causal LID net (cell \
                 '{}', BLSTM_LID_window {w} > 0: windowed on the config's intent, refused at \
                 every rate whatever the rate quantizes it to); a causal cell's state is reset \
                 at every window boundary -- a forward LID net runs the plain regime \
                 (BLSTM_LID_window 0, what the launcher writes)",
                cell.as_str()
            );
        }
        if w > 0.0 && s > 0.0 {
            bail!(
                "fast LID plan: the overlap LID windowing is unsupported on the fast path \
                 (BLSTM_LID_shift {s} > 0 under BLSTM_LID_window {w}: overlap on the config's \
                 intent, whatever the rate quantizes it to); the fast LID scoring runs the \
                 truncate sweep or the plain forward -- the committed LID configs resolve to \
                 truncate (shift 0)"
            );
        }
        Ok(())
    }

    /// The rate-dependent half of the setup, at `rate` for the target `lang_index`: what
    /// the exact Twin derives per file (`tasks::lid::TwinBlstmSpectralLid::get_segmentation_mode7`),
    /// PURE and from the config values (the dropped carries: the SAD `_WindowShift` and the
    /// `_LIDWindowShift`, both re-quantized in place by the exact driver, are locals here;
    /// at one rate that is a fixed point, `tests/fast_lid_plan.rs`). `ssif` is the Mode-7
    /// constant [`MODE7_SSIF`]. Infallible: the regime rule is [`Self::precheck_regime`]'s,
    /// decided at construction.
    pub fn timeline(&self, rate: f64, lang_index: i32) -> LidTimeline {
        let ssif = MODE7_SSIF;
        let spectrum_shift_sec = ssif as f64 / rate;
        let sad = sad_axis(
            &self.driver_cfg,
            &self.sad.spec,
            self.sad.ssr,
            rate,
            ssif,
            spectrum_shift_sec,
            0,
        );
        let (lid_window_size, lid_window_shift, lid_no_overlap) = derive_lid_window(
            self.lid_window_size_sec,
            self.lid_window_shift_sec,
            self.lid.ssr,
            ssif,
            rate,
        );
        // targetIndex = clamp(lang_index, [0, classNb)) (`TwinBLSTMSpectralLID.cpp:626-628`).
        let mut ti = lang_index;
        if ti >= self.class_nb as i32 {
            ti = 0;
        }
        if ti < 0 {
            ti = 0;
        }
        let params = LidScoreParams {
            lid_ssr: self.lid.ssr,
            lid_lstm_sub: self.lid.spec.lstm_subsampling.clone(),
            lid_out_sub: self.lid.spec.output_subsampling.clone(),
            min_nb_of_frames: self.min_nb_of_frames,
            noise_magnitude: self.noise_magnitude,
            lid_window_size,
            two_sweeps: self.two_sweeps,
            post_process_mode: self.post_process_mode,
            class_nb: self.class_nb,
            ti: ti as usize,
            cost_modified: self.cost_modified,
        };
        LidTimeline {
            rate,
            ssif,
            spectrum_shift_sec,
            sad,
            lid_window_size,
            lid_window_shift,
            lid_no_overlap,
            ti: ti as usize,
            params,
        }
    }

    /// The SAD `real_vec_size` for a file of `frame_count` samples on timeline `tl`: the
    /// frozen `get_blstm_param` sizing on the never-run SAD net, through [`sad_axis`] (the
    /// SAD plan's shape: the rows are laid on the time axis `tl` carries).
    pub fn rows_for(&self, tl: &LidTimeline, frame_count: usize) -> usize {
        sad_axis(
            &self.driver_cfg,
            &self.sad.spec,
            self.sad.ssr,
            tl.rate,
            tl.ssif,
            tl.spectrum_shift_sec,
            frame_count,
        )
        .rows
    }

    /// The two `weightsFile` loads, in the exact Twin's order (`BLSTM` then `BLSTM_LID`,
    /// the legacy in-ctor loads), each under the legacy FILE-LOAD tolerance
    /// ([`read_file_pack`]). The SAD file is read and checked against the SAD net's exact
    /// length, then DISCARDED: Mode 7 never runs the SAD net, but the two trees must accept
    /// the same files (issue #65), so a short or missing SAD file is refused here as the
    /// exact Twin refuses it, and with both files bad the SAD error is the one raised.
    /// Returns the LID pack, `None` for an empty `BLSTM_LID_weightsFile`.
    pub fn read_weights_files(&self) -> Result<Option<Vec<f64>>> {
        self.sad.read_weights_file()?;
        self.lid.read_weights_file()
    }
}

/// The rate-dependent setup of an algo-6 Mode-7 fast LID run ([`FastLidPlan::timeline`]):
/// the SAD half (the time axis the synthesized constant-10 SAD result is laid on) and the
/// LID half (the resolved LID window and the per-channel scoring constants). The offline
/// Twin derives one per file; the Utterance stream one at construction.
#[derive(Clone, Debug)]
pub struct LidTimeline {
    pub rate: f64,
    /// [`MODE7_SSIF`].
    pub ssif: usize,
    /// `ssif / rate`.
    pub spectrum_shift_sec: f64,
    /// The SAD window resolution and time axis on the never-run SAD net.
    pub sad: SadAxis,
    /// `getLIDBLSTMParam`'s window half-size in periodogram frames (0 = plain regime).
    pub lid_window_size: usize,
    /// `getLIDBLSTMParam`'s window shift in periodogram frames, floored to 1 (the exact
    /// driver writes it back as `_LIDWindowShift = lid_window_shift * ssif / rate`).
    pub lid_window_shift: usize,
    /// The truncate flag (a resolved LID shift below one frame under a window).
    pub lid_no_overlap: bool,
    /// The target language, `clamp(lang_index, [0, class_nb))`.
    pub ti: usize,
    /// The per-channel scoring constants the shared kernel reads.
    pub(super) params: LidScoreParams,
}

/// LID window derivation (`getLIDBLSTMParam`, `TwinBLSTMSpectralLID.cpp:89-102`):
/// `(lid_window_size, lid_window_shift, lid_no_overlap)`, the shift floored to 1 as the
/// exact driver floors it before writing it back. `ssif` is the (mode-7 forced) spectrum
/// shift in frames.
pub fn derive_lid_window(
    size_sec: f64,
    shift_sec: f64,
    lid_ssr: usize,
    ssif: usize,
    rate: f64,
) -> (usize, usize, bool) {
    let ssifd = ssif as f64;
    let mut lid_window_size = f64::round(size_sec * rate / 2.0 / ssifd) as usize;
    if lid_window_size != 0 && lid_window_size < lid_ssr {
        lid_window_size = lid_ssr;
    }
    let mut lid_window_shift = f64::round(shift_sec * rate / ssifd) as i64;
    let mut lid_no_overlap = false;
    if lid_window_size != 0 && lid_window_shift < 1 {
        lid_no_overlap = true;
        let raw = f64::round(size_sec * rate / ssifd) as usize;
        lid_window_size = (raw / lid_ssr) * lid_ssr;
        if lid_window_size < 10 * lid_ssr {
            lid_window_size = 10 * lid_ssr;
        }
    }
    if lid_window_size == 0 || lid_window_shift < 1 {
        lid_window_shift = 1;
    }
    (lid_window_size, lid_window_shift as usize, lid_no_overlap)
}
