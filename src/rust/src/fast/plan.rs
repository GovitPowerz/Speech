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
//!   `(time_step, time_offset)` pair. [`SadTimeline::rows_for`] is the only
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

use anyhow::{Result, bail};
use indexmap::IndexMap;

use crate::config::NnetSpec;
use crate::features::pipeline::{FeatureConfig, SpectralParams};
use crate::legacy_config::get_f64_opt;
use crate::nn::blstm::{BlstmConfig, check_pack_len};
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmenter::{DriverConfig, SegmenterConfig};

use super::cells::CellGeometry;
use super::driver::{
    FastNet, FastNetShape, build_net, build_spec_aligned_to, classify_fast_shape, exact_pack_len,
    read_file_pack,
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
        let algo = map
            .get("Algo_choice")
            .ok_or_else(|| anyhow::anyhow!("param 'Algo_choice' not found in config"))?
            .trim()
            .parse::<i32>()
            .map_err(|e| anyhow::anyhow!("fast SAD plan: Algo_choice parse: {e}"))?;
        if algo != 3 {
            bail!("fast SAD plan: only algo 3 (spectral SAD) is supported (got {algo})");
        }

        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;

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
        let shape = classify_fast_shape(&bc);
        let spec = build_spec_aligned_to(map, "BLSTM", &bc)?;

        // The fast-tree-wide set. Type 0 is the exact path's `_ => {}` arm
        // (`nn/blstm.rs`, the `feed_forward_backward` normalization match): a genuine
        // no-op on BOTH sides. -2 (the plain-FFB self-normalized COPY) is out.
        if !FAST_TREE_NORM_TYPES.contains(&bc.input_normalization_type) {
            bail!(
                "fast SAD: only InputNormalizationType -1 (self-normalization), 1 (external \
                 pack-carried mean/std, the phase-8 frozen mode) or 0 (none) are supported on \
                 the fast path (got {})",
                bc.input_normalization_type
            );
        }

        let ssr = spec.lstm_subsampling.iter().product::<usize>()
            * spec.output_subsampling.iter().product::<usize>();
        let pack_len = exact_pack_len(&bc)?;
        let fixed_gain = get_f64_opt(map, "Audio_fixed_gain")?;
        let weights_file = map
            .get("BLSTM_weightsFile")
            .map(String::as_str)
            .unwrap_or("")
            .to_string();

        let plan = FastSadPlan {
            feature_cfg,
            seg_cfg,
            driver_cfg,
            spec,
            shape,
            geometry: CellGeometry::from(&bc),
            input_normalization_type: bc.input_normalization_type,
            pack_len,
            fixed_gain,
            weights_file,
            ssr,
        };
        plan.precheck_regime()?;
        Ok(plan)
    }

    /// The input-normalization check for ONE arm ([`accepted_norm_types`]). The offline
    /// set equals the fast-tree-wide set `from_map` already enforced, so this is a no-op
    /// there; the stream arms refuse `-1` here, in the words `phase8_gate.rs` /
    /// `phase9_stream_causal.rs` pin.
    pub fn check_norm(&self, arm: FastSadArm) -> Result<()> {
        let t = self.input_normalization_type;
        if accepted_norm_types(arm).contains(&t) {
            return Ok(());
        }
        match arm {
            FastSadArm::Offline => bail!(
                "fast SAD: InputNormalizationType {t} is not in the offline fast driver's set \
                 {:?}",
                accepted_norm_types(arm)
            ),
            FastSadArm::StreamWindowed | FastSadArm::StreamCausal => bail!(
                "streaming requires BLSTM_InputNormalizationType 1 (pack-carried frozen stats) \
                 or 0 (no input normalization); got {t} (type -1 self-normalization needs \
                 whole-sequence lookahead)"
            ),
        }
    }

    /// The RATE-FREE windowing pre-check: what the config's seconds settle before any rate
    /// is known. `get_blstm_param` resolves `window_size = round(window_size_sec * rate /
    /// 2 / ssif)` and `window_shift = round(window_shift_sec * rate / ssif)`, so a ZERO
    /// `window_size_sec` is the plain regime at every rate and a ZERO `window_shift_sec`
    /// under a nonzero window is the truncate regime at every rate; both are decided here.
    /// The shape rule: BLSTM runs OVERLAP only, a causal cell runs PLAIN only (its state
    /// would reset at every window boundary), a bidirectional cell runs either; TRUNCATE
    /// is refused on every shape. A causal config with a NONZERO window is refused here
    /// too, on the config's intent, even when a tiny window would quantize to 0 at some
    /// rate. [`Self::timeline`] re-checks AUTHORITATIVELY on the resolved values (a
    /// sub-frame shift that rounds to 0 at the actual rate is caught there).
    fn precheck_regime(&self) -> Result<()> {
        let w = self.driver_cfg.window_size_sec;
        let s = self.driver_cfg.window_shift_sec;
        match self.shape {
            FastNetShape::Causal(_) if w > 0.0 => bail!(
                "fast causal SAD: windowed inference is unsupported (BLSTM_window {w} > 0); a \
                 causal cell's state is reset at every window boundary (spec S1.2), so causal \
                 streaming requires the plain regime and so does the offline fast driver -- \
                 the phase-9 causal configs use BLSTM_window 0"
            ),
            FastNetShape::Blstm if w == 0.0 => bail!(
                "fast SAD: the plain (non-windowed) forward is unsupported for the \
                 bidirectional LSTM (BLSTM_window 0 resolves window_size 0); the gate configs \
                 use windowed overlap"
            ),
            FastNetShape::Blstm | FastNetShape::BiCell(_) if w > 0.0 && s == 0.0 => bail!(
                "fast SAD: the truncate (non-overlap) windowing is unsupported (BLSTM_shift 0: \
                 window_shift resolves < 1); the fast tree implements the plain and OVERLAP \
                 regimes (spec S5)"
            ),
            _ => Ok(()),
        }
    }

    /// The rate-dependent half of the setup, at `rate`: what the exact driver derives per
    /// file (`tasks/sad.rs::BlstmSpectralSegmenter::get_segmentation`), PURE and from the
    /// config values (see the module docs on the dropped stateful carry). Runs the
    /// authoritative regime check on the RESOLVED window/shift.
    pub fn timeline(&self, rate: f64) -> Result<SadTimeline> {
        // `SpectralParams::derive` quantizes the spectrum shift the way the exact driver's
        // stateful member does on its first call: `shift_frames = round(shift_sec * rate)`,
        // `shift_sec = shift_frames / rate`.
        let params = SpectralParams::derive(&self.feature_cfg, rate);
        let ssif = params.shift_frames;
        let spectrum_shift_sec = params.shift_sec;

        // getBLSTMParam (tasks/sad.rs): window/shift derivation, mutating the window shift
        // (the stateful `_WindowShift`, here a local). `frame_count` 0: only
        // `real_vec_size` depends on it ([`SadTimeline::rows_for`]).
        let mut window_shift_sec = self.driver_cfg.window_shift_sec;
        let (window_size, window_shift, no_overlap, _rows) = get_blstm_param(
            self.driver_cfg.window_size_sec,
            &mut window_shift_sec,
            rate,
            ssif,
            self.ssr,
            &self.spec.lstm_subsampling,
            &self.spec.output_subsampling,
            0,
        );

        // The windowing regime is TIED TO THE SHAPE (the pre-check's rule, on the resolved
        // values; the messages keep the pre-check's pinned clauses).
        match self.shape {
            FastNetShape::Causal(_) => {
                if window_size != 0 {
                    bail!(
                        "fast causal SAD: windowed inference is unsupported (BLSTM_window \
                         resolves window_size {window_size}); a causal cell's state is reset at \
                         every window boundary (spec S1.2), so causal streaming requires the \
                         plain regime and so does the offline fast driver -- the phase-9 causal \
                         configs use BLSTM_window 0"
                    );
                }
            }
            FastNetShape::Blstm => {
                if window_size == 0 {
                    bail!(
                        "fast SAD: the plain (non-windowed) forward is unsupported for the \
                         bidirectional LSTM (BLSTM_window resolves window_size 0); the gate \
                         configs use windowed overlap"
                    );
                }
                if no_overlap {
                    bail!(
                        "fast SAD: the truncate (non-overlap) windowing is unsupported \
                         (window_shift resolves < 1); the fast tree implements the plain and \
                         OVERLAP regimes (spec S5)"
                    );
                }
            }
            FastNetShape::BiCell(_) => {
                if no_overlap {
                    bail!(
                        "fast bidirectional SAD: the truncate (non-overlap) windowing is \
                         unsupported (window_shift resolves < 1); FastBiCell implements the \
                         plain and OVERLAP regimes (spec S5)"
                    );
                }
            }
        }

        // timeStep/timeOffset (tasks/sad.rs, the `:724-734` block): the BASE pair is
        // `_WindowShift`-derived; the OVERLAP branch OVERRIDES it with `_SpectrumShift`
        // (the asymmetry vs the signal driver). `window_size == 0` -- the causal regime --
        // keeps the base pair, where `window_shift_sec` is the value `get_blstm_param` just
        // mutated (`window_shift * ssif / rate`, `window_shift` floored to 1 at window 0).
        // `no_overlap` is refused above, so `window_size > 0` here means OVERLAP.
        let ssr = self.ssr as f64;
        let (time_step, time_offset) = if window_size > 0 {
            let ts = spectrum_shift_sec * ssr;
            (ts, ts / 2.0 - spectrum_shift_sec / 2.0)
        } else {
            let ts = window_shift_sec * ssr;
            (ts, ts / 2.0 - window_shift_sec / 2.0)
        };

        Ok(SadTimeline {
            rate,
            params,
            ssif,
            spectrum_shift_sec,
            window_shift_sec,
            window_size,
            window_shift,
            no_overlap,
            time_step,
            time_offset,
            window_size_sec: self.driver_cfg.window_size_sec,
            window_shift_sec_cfg: self.driver_cfg.window_shift_sec,
            ssr: self.ssr,
            lstm_subsampling: self.spec.lstm_subsampling.clone(),
            output_subsampling: self.spec.output_subsampling.clone(),
        })
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

/// The rate-dependent setup of an algo-3 fast SAD run ([`FastSadPlan::timeline`]): the
/// periodogram framing, the quantized shifts, the resolved window and the decision-layer
/// time axis. The offline driver derives one per file; the session one at construction.
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
    /// `get_blstm_param`'s truncate flag -- always `false` on a timeline that built (the
    /// regime check refuses it); kept so a reader sees the resolved regime whole.
    pub no_overlap: bool,
    /// The decision layer's time axis (`results_to_segmentation`'s pair).
    pub time_step: f64,
    pub time_offset: f64,

    // What `rows_for` needs to re-run `get_blstm_param` at a frame count.
    window_size_sec: f64,
    window_shift_sec_cfg: f64,
    ssr: usize,
    lstm_subsampling: Vec<usize>,
    output_subsampling: Vec<usize>,
}

impl SadTimeline {
    /// `real_vec_size` for a file of `frame_count` samples: `get_blstm_param`'s
    /// `ceil(frame_count / ssif)` divided sequentially by every sub-sampling ratio -- the
    /// ONE frame-count-dependent output, re-run on the frozen function rather than
    /// re-derived here. The window/shift it also returns are the timeline's own (the
    /// quantization is a fixed point, so the mutated shift equals [`Self::window_shift_sec`]).
    pub fn rows_for(&self, frame_count: usize) -> usize {
        let mut window_shift_sec = self.window_shift_sec_cfg;
        let (_w, _s, _n, rows) = get_blstm_param(
            self.window_size_sec,
            &mut window_shift_sec,
            self.rate,
            self.ssif,
            self.ssr,
            &self.lstm_subsampling,
            &self.output_subsampling,
            frame_count,
        );
        debug_assert_eq!(window_shift_sec, self.window_shift_sec);
        rows
    }
}
