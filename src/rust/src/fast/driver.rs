//! `fast::driver` -- the fast inference drivers behind `Inference_Path fast`
//! (Phase 7 Task 4: algo-3 spectral SAD).
//!
//! [`FastSpectralSegmenter`] is the f32 counterpart of the exact
//! [`crate::tasks::sad::BlstmSpectralSegmenter`] (algo 3). It implements the SAME
//! [`Segmenter`] trait and hands off to the SAME, UNCHANGED f64 decision layer
//! (`results_to_segmentation` + smoothing + `compute_errors`), so segments, cols,
//! and scoring flow exactly as the exact path's -- only the numeric heart (feature
//! FFT/mel + the BLSTM forward) runs in f32 (see [`super`] for the by-design
//! divergence). The exact source it mirrors line-for-line (cite when auditing drift,
//! per the plan's R2):
//!
//! - `tasks/sad.rs::BlstmSpectralSegmenter::get_segmentation` (`:1376-1629`): the
//!   per-file spectrum-shift quantization, preemph/noise gates, `getBLSTMParam`
//!   sizing, the OVERLAP time-step/offset (`_SpectrumShift`-based), the cross-channel
//!   `result_vec` reuse, and the `results_to_segmentation` + `compute_errors` handoff.
//! - `nn/blstm.rs::feed_forward_backward` (`:997-1059`): the input-normalization
//!   dispatch (type -1 self-normalization applied ONCE before the windowed dispatch)
//!   + the overlap driver selection (`window_size > 0 && overlaps`).
//! - `nn/blstm.rs::feed_forward_backward_overlap` (`:1594-1726`): reproduced in
//!   [`super::nn::FastBlstm::feed_forward_overlap`] (output-only).
//!
//! THREE NET SHAPES (Phase 9 Task 6 + phase-10 Tasks 7/8, spec S4.2/S5/S6).
//! `FastSpectralSegmenter` dispatches on `BLSTM_Cell_Type` x `BLSTM_Direction` at
//! construction ([`classify_fast_shape`]): `(lstm, bidirectional)` builds the phase-7
//! [`FastBlstm`], `(any cell, forward)` builds the causal [`FastCausalNet`]
//! ([`super::cells`]), and `(any cell, bidirectional)` builds [`FastBiCell`]
//! ([`super::bicell`]). NOTHING typed-bails on shape any more: phase-10 Task 8's
//! `FastLstm` filled the last hole (`(lstm, forward)`), so the classifier is TOTAL and
//! the cell set `{lstm, slstm, mamba, cfc}` was complete in both directions -- and stayed
//! total when phase-11 added a FIFTH cell, `transformer` (Task 5 causal, Task 6
//! bidirectional): the match here is generic over `CellType`, so neither arm needed a
//! transformer-specific line, only [`FastNetShape`]'s per-variant docs below did.
//!
//! THE WINDOWING REGIME FOLLOWS THE SHAPE. BLSTM runs the OVERLAP windowed driver only;
//! causal runs the PLAIN whole-sequence forward only (a causal cell inside a window has
//! its state reset at every window boundary); BiCell runs EITHER, per the config, exactly
//! as the exact tree does -- a bidirectional cell inside a window is the same well-defined
//! regime a bidirectional LSTM is. TRUNCATE (non-overlap) windowing is refused on every
//! shape.
//!
//! SCOPE (spec S1.2/S1.3, the house typed-bail pattern -- what the gate configs
//! exercise, everything else typed-bails loudly so scope creep is loud):
//! - The PITCH second pass (`BLSTM_TDCwindow > 0`, spec R4) typed-bails at
//!   construction.
//! - `InputNormalizationType` outside {-1, 0, 1} typed-bails at construction (the
//!   phase-7 gate configs -- `tier2_spectral.config` + the phase-6 `lre_sad.toml` --
//!   both use -1; type 1, the pack-carried external mean/std, joined in Phase 8
//!   Task 1 as the frozen-stats reference mode, spec S1.1; type 0, NO normalization
//!   at all -- the exact path's `_ => {}` arm and therefore a no-op on both sides --
//!   joined in Phase 9 Task 6, which is what the committed causal gate fixtures use).
//! - BLSTM shape: the PLAIN (window 0) and TRUNCATE (non-overlap) windowed variants
//!   typed-bail at `get_segmentation` (both phase-7/8 gate configs resolve to OVERLAP:
//!   `BLSTM_window 3.25 / BLSTM_shift 0.8` -> `window_size > 0`, `no_overlap false`).
//! - CAUSAL shape: any WINDOWED dispatch (`window_size != 0`) typed-bails there
//!   instead (the phase-9 causal configs use `BLSTM_window 0`).
//! - BiCELL shape: only TRUNCATE typed-bails; plain and overlap both run (spec S5).
//!
//! FORWARD-ONLY: the fast path is inference-only (spec: training stays exact f64), so
//! `cumulative_error`/`nb_of_classif` (the NN-cost result columns 4 / `len-1`) stay
//! ZERO -- a documented divergence from the exact driver when run scored (the exact
//! path harvests the cost from the targets; the fast path never builds targets). The
//! `compute_errors` columns (Pfa/Pmiss/error-rate) flow through the shared f64 layer
//! unchanged.

use anyhow::{Result, anyhow, bail};
use indexmap::IndexMap;
use ndarray::Array2;

use crate::audio::Audio;
use crate::config::NnetSpec;
use crate::constants::random_gauss;
use crate::legacy_config::{get_f64, get_f64_default};
use crate::nn::blstm::{
    BlstmConfig, CellType, CfcParams, Direction, MambaParams, TransformerParams,
};
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::compute_errors;
use crate::tasks::segmenter::{DriverConfig, Segmenter, SegmenterConfig, results_to_segmentation};

use super::bicell::FastBiCell;
use super::cells::FastCausalNet;
use super::nn::{
    FastBlstm, FastMatrix, external_normalize_f32, pad_replicate_ends_f32, self_normalize_f32,
    slice_rows_f32,
};
use super::pipeline::FastPipeline;
use crate::features::pipeline::{FeatureConfig, SpectralParams};

/// Which fast net shape a config's `Cell_Type` x `Direction` pair selects
/// (Phase 9 Task 6, spec S4.2). A CLOSED set: the phase-7 bidirectional peephole-LSTM
/// twin, or a causal (`Direction forward`) stack of one of the new cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FastNetShape {
    /// [`FastBlstm`] -- LSTM + bidirectional, the phase-7 path (BYTE-UNTOUCHED).
    Blstm,
    /// [`FastCausalNet`] -- ANY cell + `Direction forward`: `slstm`/`mamba` (phase 9),
    /// `cfc` (phase-10 Task 6), `lstm` (phase-10 Task 8, which completed the phase-10
    /// set), `transformer` (phase-11 Task 5).
    Causal(CellType),
    /// [`FastBiCell`] -- `slstm`/`mamba`/`cfc` (phase-10 Task 7, spec S5) /
    /// `transformer` (phase-11 Task 6, spec S5.3) + `Direction bidirectional`. NOT
    /// FRAME-STREAMABLE: `fast::stream` typed-bails this variant; the utterance-granular
    /// `fast::stream_lid` runs it whole per push (issue #57, ADR-0004's amendment).
    BiCell(CellType),
}

/// Classify a net's `(cell_type, direction)` pair into the fast net shape that runs it
/// (spec S4.2, completed by phase-10 spec S6).
///
/// TOTAL since phase-10 Task 8 -- every pair maps to a shape, so this returns a
/// [`FastNetShape`] rather than a `Result`:
/// - `(lstm, bidirectional)` -> [`FastNetShape::Blstm`], the phase-7 batched twin;
/// - `(any cell, forward)` -> [`FastNetShape::Causal`] (`cfc` joined in phase-10 Task 6,
///   `lstm` in Task 8 -- which is what emptied the last bail arm);
/// - `(any cell, bidirectional)` -> [`FastNetShape::BiCell`] (phase-10 Task 7, spec S5),
///   except the `lstm` case caught by the first arm.
///
/// THE TWO BAILS THAT USED TO LIVE HERE DID NOT DISAPPEAR AS HAZARDS, they moved or were
/// implemented. The hazard argument is unchanged and worth restating, because it is what
/// the SIZING checks downstream now carry: `NnetSpec` carries neither cell type nor
/// direction, so a pack at least as long as another architecture's would be consumed
/// head-first and RUN, producing one architecture's numbers under another's name, with no
/// tolerance to widen and no gate to catch it. Each shape's `element_count`
/// ([`FastBlstm::from_flat`], [`FastCausalNet::element_count`],
/// [`FastBiCell::element_count`]) closes it for a SHORTER pack only; an over-long one is
/// consumed head-first unless the caller demands the exact length (the Twin's in-memory
/// LID seam does, [`FastTwinLid::from_legacy`]).
///
/// A refusal survives exactly where a shape genuinely cannot run, NOT here:
/// `fast::stream::StreamingSession::new` refuses [`FastNetShape::BiCell`] (the FRAME
/// stream runs no bidirectional new cell), keeping the refusal's LEADING CLAUSE, which is
/// what `phase8_gate.rs` / `phase9_stream_causal.rs` assert (a PREFIX SUBSTRING, not the
/// body). The Twin's LID net classifies through this same function since issue #57
/// ([`FastLidNet`]); its two surviving refusals are REGIME rules (windowed causal,
/// overlap), raised where the window resolves against the rate.
///
/// TOTALITY IS NOT A LOOSENING. A future `CellType` variant added without a fast kernel
/// does not silently fall through here: it reaches `super::cells::cell_weight_count`,
/// whose `match` is exhaustive, so the omission is a COMPILE error -- a strictly earlier
/// failure than the runtime bail this arm used to raise.
pub(crate) fn classify_fast_shape(bc: &BlstmConfig) -> FastNetShape {
    match (bc.cell_type, bc.direction) {
        (CellType::Lstm, Direction::Bidirectional) => FastNetShape::Blstm,
        (cell, Direction::Forward) => FastNetShape::Causal(cell),
        (cell, Direction::Bidirectional) => FastNetShape::BiCell(cell),
    }
}

/// Build the `NnetSpec` for the fast net (under config `prefix`, e.g. `"BLSTM"` for
/// the SAD net or `"BLSTM_LID"` for the Twin's LID net) with peephole flags aligned to
/// the EXACT path's defaults (Phase 7 Task 4 rider 1 -- the peephole default asymmetry).
///
/// SHAPE-FREE since issue #57. Through phase 11 this was ALSO the Twin's BLSTM-only gate
/// (`bail_unsupported_shape`, refusing any `Cell_Type`/`Direction` pair but the phase-7
/// twin on BOTH of the Twin's prefixes). The Twin's LID net now classifies through
/// [`classify_fast_shape`] like the SAD driver ([`FastLidNet`]), and its SAD net is never
/// run in Mode 7 (only its sub-sampling ratios are read, which every cell carries), so no
/// shape gate is left to apply here. One parse + one alignment, nothing else.
///
/// The fast net reads a `NnetSpec`; the exact net reads a `BlstmConfig`. A peephole
/// mismatch between the two is a divergence with no tolerance floor, so the spec's
/// peepholes are overridden from `BlstmConfig` and both paths read them from the SAME
/// source. `NnetSpec::from_legacy` defaulted an ABSENT key to FALSE until issue #32 and
/// now applies `PeepholeFlags::from_legacy`'s rule itself, so the override is
/// value-preserving on every config; it stays so the agreement does not rest on two
/// readers staying in sync. Being value-preserving, the override itself is unobservable;
/// `tests/phase7_parity_sad.rs` pins the agreement instead, slot by slot, on an omitting,
/// an all-false and a mixed config.
pub fn build_aligned_spec(map: &IndexMap<String, String>, prefix: &str) -> Result<NnetSpec> {
    let bc = BlstmConfig::from_legacy(map, prefix)?;
    build_spec_aligned_to(map, prefix, &bc)
}

/// [`build_aligned_spec`] on an already-parsed [`BlstmConfig`]: the spec assembly alone.
/// Split out by Task 6 so a driver that classifies the shape itself builds its net off
/// the same peephole-aligned spec, with no second config parse.
pub(crate) fn build_spec_aligned_to(
    map: &IndexMap<String, String>,
    prefix: &str,
    bc: &BlstmConfig,
) -> Result<NnetSpec> {
    // `NnetSpec::from_legacy` REQUIRES `<prefix>_NNetInputSize`, but the Twin config omits
    // it (the exact `BlstmConfig` derives `input_size = LSTMNeuronNb[0]` instead). The fast
    // net also reads `LSTMNeuronNb[0]` for its input width and NEVER consults
    // `spec.input_size`, so when the key is absent we inject `LSTMNeuronNb[0]` (the same
    // value the exact path uses) purely to satisfy the parser -- value-preserving where
    // the key IS present (every SAD config sets it == LSTMNeuronNb[0]).
    let input_key = format!("{prefix}_NNetInputSize");
    let mut spec = if map.contains_key(&input_key) {
        NnetSpec::from_legacy(map, prefix)?
    } else {
        let mut m = map.clone();
        m.insert(input_key, bc.lstm_neuron_nb[0].to_string());
        NnetSpec::from_legacy(&m, prefix)?
    };
    spec.peepholes = [
        bc.forward_peep.cells,
        bc.backward_peep.cells,
        bc.forward_peep.gates,
        bc.backward_peep.gates,
        bc.forward_peep.gates_recurrent,
        bc.backward_peep.gates_recurrent,
    ];
    Ok(spec)
}

/// The algo-3 SAD net actually built, per [`FastNetShape`]. The windowing regime is
/// tied to the variant and NOT interchangeable, which is why the dispatch lives here
/// rather than behind a trait: [`FastSadNet::Blstm`] runs the OVERLAP windowed driver
/// (phase 7, `window_size > 0`), [`FastSadNet::Causal`] runs the PLAIN whole-sequence
/// forward (`window_size == 0`) -- a causal cell inside a window would have its state
/// reset at every window boundary, which is the "pointless-but-defined" regime spec
/// S1.2 names and S5.3 forbids for streaming -- and [`FastSadNet::BiCell`] runs EITHER
/// (phase-10 Task 7, spec S5: "BiCell runs OVERLAP or plain per the config, exactly as
/// the exact tree does"), because a bidirectional cell inside a window is the same
/// well-defined regime a bidirectional LSTM is.
///
/// `large_enum_variant` allowed on the house precedent (`Processor`, `CellLayer`,
/// `FastCell`): a `FastSadNet` is built ONCE per driver and then only borrowed -- one
/// per `FastSpectralSegmenter`, never a collection -- so the unused tag padding costs
/// nothing measurable, while boxing would put a pointer chase in front of the per-file
/// forward.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum FastSadNet {
    Blstm(FastBlstm),
    Causal(FastCausalNet),
    BiCell(FastBiCell),
}

impl FastSadNet {
    fn output_size(&self) -> usize {
        match self {
            FastSadNet::Blstm(n) => n.output_size(),
            FastSadNet::Causal(n) => n.output_size(),
            FastSadNet::BiCell(n) => n.output_size(),
        }
    }
    fn normalize_mean(&self) -> &[f32] {
        match self {
            FastSadNet::Blstm(n) => n.normalize_mean(),
            FastSadNet::Causal(n) => n.normalize_mean(),
            FastSadNet::BiCell(n) => n.normalize_mean(),
        }
    }
    fn normalize_std(&self) -> &[f32] {
        match self {
            FastSadNet::Blstm(n) => n.normalize_std(),
            FastSadNet::Causal(n) => n.normalize_std(),
            FastSadNet::BiCell(n) => n.normalize_std(),
        }
    }
}

/// Build the algo-3 SAD net for a classified [`FastNetShape`] from the flat f64 pack.
/// ONE place, so `from_legacy(map, Some(flat))` and the deferred
/// `load_weights_file` cannot pick different arms.
fn build_sad_net(
    spec: &NnetSpec,
    shape: FastNetShape,
    mamba: &MambaParams,
    cfc: &CfcParams,
    transformer: &TransformerParams,
    flat: &[f64],
) -> Result<FastSadNet> {
    Ok(match shape {
        FastNetShape::Blstm => FastSadNet::Blstm(FastBlstm::from_flat(spec, flat)?),
        FastNetShape::Causal(cell) => FastSadNet::Causal(FastCausalNet::from_flat(
            spec,
            cell,
            mamba,
            cfc,
            transformer,
            flat,
        )?),
        FastNetShape::BiCell(cell) => FastSadNet::BiCell(FastBiCell::from_flat(
            spec,
            cell,
            mamba,
            cfc,
            transformer,
            flat,
        )?),
    })
}

/// Copy a PLAIN (non-windowed) net forward's posteriors into the shared `result_buf`.
/// Used by the causal shape (always) and the bidirectional-cell shape (when the config
/// resolves `window_size 0`); `label` names the shape in the geometry-mismatch message.
///
/// A net output LONGER than the result vector is a geometry mismatch, not something to
/// truncate silently: the exact path indexes straight into its `real_vec_size x 1` buffer
/// and PANICS there (`NeuronLayer::feed_forward`'s `output[[t, j]]` write), so a quiet
/// clamp here would mask a real disagreement between `get_blstm_param`'s sizing and the
/// feature front-end's row count.
///
/// ONE DEGENERATE-CASE DIVERGENCE, documented not fixed: on a sequence so short that the
/// net emits ZERO rows, the exact plain path never reaches `NeuronLayer::feed_forward`
/// (`Network::drive` returns early on an empty input, `network.rs:329`) and leaves
/// `result_vec` UNTOUCHED -- i.e. holding the previous channel's contents -- while this
/// zeroes it. Unreachable on any real file (it needs fewer feature rows than the
/// sub-sampling ratio) and the fast behaviour is the saner of the two; recorded so a
/// future reader does not mistake it for an oversight.
fn copy_plain_output(label: &str, out: &FastMatrix, result_buf: &mut FastMatrix) -> Result<()> {
    if out.rows > result_buf.rows || out.cols != result_buf.cols {
        bail!(
            "{label}: net output {}x{} does not fit the result vector {}x{} (getBLSTMParam \
             sizing vs the feature row count)",
            out.rows,
            out.cols,
            result_buf.rows,
            result_buf.cols
        );
    }
    result_buf.data.fill(0.0);
    let cols = result_buf.cols;
    for r in 0..out.rows {
        for c in 0..cols {
            result_buf.data[r * cols + c] = out.data[r * out.cols + c];
        }
    }
    Ok(())
}

/// f32 spectral SAD segmenter (algo 3), the fast counterpart of
/// [`crate::tasks::sad::BlstmSpectralSegmenter`]. See the module docs for the mirrored
/// sources + scope.
///
/// `Clone` (deep-copying the fast net) so `Processor::FastSpectral` satisfies the bag's
/// `#[derive(Clone)]`; the fast path is inference-only, so a clone is off any hot path.
#[derive(Clone)]
pub struct FastSpectralSegmenter {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    feature_cfg: FeatureConfig,
    spec: NnetSpec,
    /// The net SHAPE the config selected (Phase 9 Task 6) -- kept so the deferred
    /// `load_weights_file` rebuild picks the same arm `from_legacy` did.
    shape: FastNetShape,
    /// The `Mamba_*` geometry, inert unless [`Self::shape`] is `Causal(Mamba)`.
    mamba: MambaParams,
    /// The `Cfc_*` geometry, inert unless [`Self::shape`] is `Causal(Cfc)`. Carried
    /// beside `mamba` for the same reason: the deferred `load_weights_file` rebuild
    /// must size the net exactly as `from_legacy` did.
    cfc: CfcParams,
    /// The `Transformer_*` geometry, inert unless [`Self::shape`] names the transformer
    /// cell. Carried for its two siblings' reason (the deferred rebuild must size the net
    /// exactly as `from_legacy` did) -- and here `window`/`heads` matter to the KERNEL
    /// even though only `d_ff` moves the pack LENGTH.
    transformer: TransformerParams,
    net: Option<FastSadNet>,

    /// Sub-sampling factors + whole-BLSTM ratio, cached from the spec for
    /// [`get_blstm_param`] (which needs them without a live net borrow).
    lstm_sub_sampling: Vec<usize>,
    output_sub_sampling: Vec<usize>,
    ssr: usize,

    /// `BLSTM_InputNormalizationType`: -1 (whole-sequence self-normalization, the
    /// phase-7 gate configs) or 1 (pack-carried external mean/std -- the Phase 8
    /// frozen-stats reference mode, S1.1); every other value typed-bails at
    /// construction. Dispatched per channel in `get_segmentation`, mirroring the
    /// exact `feed_forward_backward` top (`nn/blstm.rs:1008-1030`).
    input_normalization_type: i16,

    /// Stateful `_SpectrumShift`/`_SpectrumShiftInFrames`/`_WindowShift`/
    /// `_LTSVWindowShift` members, re-quantized per `get_segmentation` call exactly
    /// as the exact driver does (mirrors `tasks/sad.rs` for state faithfulness).
    spectrum_shift_sec: f64,
    spectrum_shift_in_frames: usize,
    window_shift_sec: f64,
    ltsv_shift_sec: f64,

    channels: usize,
    cumulative_error: Vec<f64>,
    nb_of_classif: Vec<i64>,
    /// Post-overlap-division, PRE-`results_to_segmentation` posterior rows, one per
    /// channel -- the f64-widened result_vec captured on the last `get_segmentation`
    /// call (mirrors `BlstmSpectralSegmenter::last_result_rows`, the port-side
    /// observation point). Consumed by the CI parity leg to measure the posterior
    /// delta vs the exact path.
    last_result_rows: Vec<Vec<f64>>,
}

impl FastSpectralSegmenter {
    /// Build from a legacy config map + optional f64 weight pack, mirroring
    /// [`crate::tasks::sad::BlstmSpectralSegmenter::from_legacy`]'s config surface.
    /// `weights: Some(flat)` builds the net SELECTED BY [`classify_fast_shape`]
    /// immediately -- [`FastBlstm`] or [`FastCausalNet`], both narrowing f64 -> f32
    /// once, after adim (new cells carry no adim by construction, spec S1.3); `None`
    /// defers to [`Self::load_weights_file`] (the bag's two-step
    /// `from_legacy(map, None)` + `load_weights_file` pattern).
    ///
    /// Typed-bails (loudly, at construction) the unsupported fast-mode surfaces: the
    /// pitch second pass (`TDCwindow > 0`) and any `InputNormalizationType` outside
    /// {-1, 0, 1} (1 joined in Phase 8 Task 1 -- the frozen-stats reference mode;
    /// 0 in Phase 9 Task 6 -- the exact path's no-op arm). The `Cell_Type` x `Direction`
    /// pair is NO LONGER among them: [`classify_fast_shape`] became total in phase-10
    /// Task 8.
    pub fn from_legacy(
        map: &IndexMap<String, String>,
        weights: Option<&[f64]>,
    ) -> Result<FastSpectralSegmenter> {
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;

        // Pitch second pass unsupported on the fast path (spec R4). tier2 + the
        // phase-6 SAD config use TDCwindow 0.
        if feature_cfg.tdc_window > 0.0 {
            bail!(
                "fast SAD: the pitch second pass (BLSTM_TDCwindow > 0) is not supported on the \
                 fast inference path (spec R4); the gate configs use TDCwindow 0"
            );
        }

        // Cell x direction dispatch (Phase 9 Task 6, spec S4.2) + peephole-default
        // alignment (rider 1). Every pair maps to a shape since phase-10 Task 8; the
        // remaining refusals are per-shape (pack length, windowing regime, streaming).
        let bc = BlstmConfig::from_legacy(map, "BLSTM")?;
        let shape = classify_fast_shape(&bc);
        let spec = build_spec_aligned_to(map, "BLSTM", &bc)?;

        // Normalization types on the fast SAD path: -1 (self-normalization, the
        // phase-7 gate configs), 1 (pack-carried external mean/std -- Phase 8 S1.1,
        // the frozen-stats reference mode) and 0 (NO normalization, joined in Phase 9
        // Task 6). Everything else typed-bails.
        //
        // Type 0 is the exact path's `_ => {}` arm (`nn/blstm.rs:1271`, the
        // `feed_forward_backward` normalization match): a genuine no-op on BOTH sides,
        // so admitting it cannot introduce a divergence -- it only stops a config that
        // asks for nothing from being rejected for asking for something unsupported.
        // The phase-9 committed gate fixtures use it, and the previously-bailing
        // configs it un-bails are exactly the ones on which the two paths agree by
        // construction. -2 (the plain-FFB self-normalized COPY) is still out.
        if ![-1, 0, 1].contains(&bc.input_normalization_type) {
            bail!(
                "fast SAD: only InputNormalizationType -1 (self-normalization), 1 (external \
                 pack-carried mean/std, the phase-8 frozen mode) or 0 (none) are supported on \
                 the fast path (got {})",
                bc.input_normalization_type
            );
        }

        let lstm_sub_sampling = spec.lstm_subsampling.clone();
        let output_sub_sampling = spec.output_subsampling.clone();
        let ssr = lstm_sub_sampling.iter().product::<usize>()
            * output_sub_sampling.iter().product::<usize>();

        let net = match weights {
            Some(flat) => Some(build_sad_net(
                &spec,
                shape,
                &bc.mamba,
                &bc.cfc,
                &bc.transformer,
                flat,
            )?),
            None => None,
        };

        let spectrum_shift_sec = feature_cfg.shift_sec;
        let window_shift_sec = driver_cfg.window_shift_sec;
        let ltsv_shift_sec = feature_cfg.ltsv_shift;

        Ok(FastSpectralSegmenter {
            driver_cfg,
            seg_cfg,
            feature_cfg,
            spec,
            shape,
            mamba: bc.mamba,
            cfc: bc.cfc,
            transformer: bc.transformer,
            net,
            lstm_sub_sampling,
            output_sub_sampling,
            ssr,
            input_normalization_type: bc.input_normalization_type,
            spectrum_shift_sec,
            spectrum_shift_in_frames: 0,
            window_shift_sec,
            ltsv_shift_sec,
            channels: 0,
            cumulative_error: Vec::new(),
            nb_of_classif: Vec::new(),
            last_result_rows: Vec::new(),
        })
    }

    /// `<prefix>_weightsFile` load (mirrors [`crate::nn::blstm::BlstmNetwork::
    /// load_weights_file`]): an EMPTY key leaves the net unloaded (a subsequent
    /// `get_segmentation` errors); otherwise read the `.bin` and (re)build the net
    /// from it -- through the SAME [`build_sad_net`] the ctor uses, so the deferred
    /// load cannot pick a different arm than [`Self::from_legacy`] classified. Both
    /// `FastBlstm::from_flat` and `FastCausalNet::from_flat` enforce the length check
    /// (`< element_count` -> `Err`).
    pub fn load_weights_file(&mut self, map: &IndexMap<String, String>) -> Result<()> {
        let weights_file = map
            .get("BLSTM_weightsFile")
            .map(String::as_str)
            .unwrap_or("");
        if weights_file.is_empty() {
            return Ok(());
        }
        let flat = crate::io::binary::read_weight_vector(std::path::Path::new(weights_file))?;
        self.net = Some(build_sad_net(
            &self.spec,
            self.shape,
            &self.mamba,
            &self.cfc,
            &self.transformer,
            &flat,
        )?);
        Ok(())
    }

    /// The configured dump directory (see [`crate::tasks::sad::BlstmSpectralSegmenter::
    /// dump_dir`]): gates the scored-branch VRCTS write.
    pub fn dump_dir(&self) -> &str {
        &self.driver_cfg.dump_dir
    }

    /// Per-channel NN cost (result col 4). ALWAYS zero on the fast path (forward-only,
    /// inference: no cost harvested), a documented divergence from the exact driver.
    pub fn cumulative_error(&self) -> &[f64] {
        &self.cumulative_error
    }

    /// Per-channel `_NbOfClassif` (result col `len-1`). ALWAYS zero on the fast path
    /// (forward-only), a documented divergence from the exact driver.
    pub fn nb_of_classif(&self) -> &[i64] {
        &self.nb_of_classif
    }

    /// Post-overlap-division, pre-`results_to_segmentation` posterior rows (one per
    /// channel) from the last `get_segmentation` -- the port-side observation point
    /// for the CI parity leg (mirrors `BlstmSpectralSegmenter::last_result_rows`).
    pub fn last_result_rows(&self) -> &[Vec<f64>] {
        &self.last_result_rows
    }
}

impl Segmenter for FastSpectralSegmenter {
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        _refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        let rate = audio.sample_rate as f64;

        // SpectralParams derived at `rate` -- the SAME binding fed to FastPipeline::new
        // below (rider 2: one sample_rate feeds both, structurally). Shared config
        // arithmetic with the exact path (not duplicated).
        let s = SpectralParams::derive(&self.feature_cfg, rate);

        // Stateful spectrum-shift quantization (tasks/sad.rs:1397-1399).
        self.spectrum_shift_in_frames = f64::round(self.spectrum_shift_sec * rate) as usize;
        self.spectrum_shift_sec = self.spectrum_shift_in_frames as f64 / rate;
        let ssif = self.spectrum_shift_in_frames;

        // preemph -> noise, same gates as the exact driver (both no-ops for tier2:
        // preemph_ratio -0.97 <= 0, noise_seed -3 <= 0).
        if self.feature_cfg.preemph_ratio > 0.0 {
            audio.apply_preemph(self.feature_cfg.preemph_ratio);
        }
        if self.feature_cfg.noise_seed > 0 {
            audio.apply_noise(self.feature_cfg.noise_ratio);
        }

        // getLTSVParam stateful re-quantization (tasks/sad.rs:1411-1414); dead for the
        // gate configs (LTSVwindow 0), kept for state faithfulness.
        let ltsv_ws = f64::round(self.feature_cfg.ltsv_shift * rate / ssif as f64) as i64;
        let ltsv_ws = if ltsv_ws < 1 { 1 } else { ltsv_ws };
        self.ltsv_shift_sec = (ltsv_ws * ssif as i64) as f64 / rate;

        // getBLSTMParam (tasks/sad.rs:1419-1428): window/shift + result-vec sizing;
        // mutates self.window_shift_sec (the stateful _WindowShift).
        let window_size_sec = self.driver_cfg.window_size_sec;
        let lstm_sub = self.lstm_sub_sampling.clone();
        let out_sub = self.output_sub_sampling.clone();
        let ssr = self.ssr;
        let (window_size, window_shift, no_overlap, real_vec_size) = get_blstm_param(
            window_size_sec,
            &mut self.window_shift_sec,
            rate,
            ssif,
            ssr,
            &lstm_sub,
            &out_sub,
            audio.data.ncols(),
        );

        // Dispatch (setProcessingType(window > 0, !noOverlap), tasks/sad.rs:1447). The
        // supported windowing regime is TIED TO THE NET SHAPE:
        //
        // - BLSTM (phase 7): the OVERLAP windowed driver ONLY -- what both phase-7/8
        //   gate configs (tier2 + phase-6 SAD, window 3.25 / shift 0.8) exercise.
        //   Plain and truncate typed-bail as unexercised in fast mode.
        // - CAUSAL (Task 6): the PLAIN whole-sequence forward ONLY (`window_size == 0`).
        //   Windowing a causal cell resets its state at every window boundary -- the
        //   "pointless-but-defined" regime of spec S1.2, which S5.3 forbids outright
        //   for the streaming session. The phase's causal configs use window 0.
        // - BiCELL (phase-10 Task 7, spec S5): BOTH the plain forward and the OVERLAP
        //   windowed driver, "exactly as the exact tree does" -- a bidirectional cell
        //   inside a window is the same well-defined regime a bidirectional LSTM is
        //   (each window is an independent whole-sequence run over its own rows), so
        //   there is nothing to refuse. TRUNCATE stays refused, as for `FastBlstm`.
        match self.shape {
            FastNetShape::Causal(_) => {
                if window_size != 0 {
                    bail!(
                        "fast causal SAD: windowed inference is unsupported (BLSTM_window \
                         resolves window_size {window_size}); a causal cell's state is reset at \
                         every window boundary (spec S1.2) -- the phase-9 causal configs use \
                         BLSTM_window 0"
                    );
                }
            }
            FastNetShape::Blstm => {
                if window_size == 0 {
                    bail!(
                        "fast SAD: the plain (non-windowed) forward is unsupported (BLSTM_window \
                         resolves window_size 0); the gate configs use windowed overlap"
                    );
                }
                if no_overlap {
                    bail!(
                        "fast SAD: the truncate (non-overlap) windowing is unsupported \
                         (window_shift resolves < 1); the gate configs use overlap"
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

        // timeStep/timeOffset (tasks/sad.rs:1451-1461). The BASE pair is
        // `_WindowShift`-derived; the OVERLAP branch OVERRIDES it with `_SpectrumShift`
        // (the asymmetry vs the signal driver). `window_size == 0` -- the causal
        // regime -- keeps the base pair, exactly as the exact driver does.
        // `self.window_shift_sec` here is the post-`get_blstm_param` value.
        let (time_step, time_offset) = if window_size > 0 {
            let ts = self.spectrum_shift_sec * ssr as f64;
            (ts, ts / 2.0 - self.spectrum_shift_sec / 2.0)
        } else {
            let ts = self.window_shift_sec * ssr as f64;
            (ts, ts / 2.0 - self.window_shift_sec / 2.0)
        };

        // Build the fast pipeline ONCE (banks + FFT plan), with the SAME `rate` that
        // derived `s` (rider 2). FastPipeline::new typed-bails an active LTSV column or
        // temporal convolution -- both off in the gate configs.
        let mut pipeline = FastPipeline::new(&s, &self.feature_cfg, rate)?;

        let channels = audio.data.nrows();
        self.channels = channels;
        // Forward-only: no NN cost harvested (documented divergence).
        self.cumulative_error = vec![0.0; channels];
        self.nb_of_classif = vec![0; channels];

        let net = self.net.as_mut().ok_or_else(|| {
            anyhow!("fast SAD net has no weights loaded (BLSTM_weightsFile empty)")
        })?;
        let output_size = net.output_size();
        // Type-1 external normalization needs the pack-carried mean/std tail; copied
        // out (tiny: input_size floats each) so the loop below can keep the single
        // `&mut net` borrow for feed_forward_overlap.
        let (norm_mean, norm_std) = if self.input_normalization_type == 1 {
            (net.normalize_mean().to_vec(), net.normalize_std().to_vec())
        } else {
            (Vec::new(), Vec::new())
        };

        // result_buf: ONE buffer reused across channels (the cross-channel reuse quirk,
        // tasks/sad.rs:1436). Zeros once; NOT re-zeroed between channels --
        // feed_forward_overlap accumulates in place, so channel N seeds from channel
        // N-1's post-division contents. Rider 3: the exact driver hardcodes `ncols = 1`
        // (`tasks/sad.rs:1436`, `resultVec2 = Zero(realVecSize, 1)`) because the SAD net is
        // binary; sizing by `output_size` here is IDENTICAL for a 1-output net (the only
        // shape the SAD gate configs use) and generalizes cleanly to a wider posterior.
        let mut result_buf = FastMatrix::zeros(real_vec_size, output_size);
        let mut last_rows: Vec<Vec<f64>> = Vec::with_capacity(channels);

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // Build the fast input sequence for this channel (f32), narrowing the
            // channel's samples once at the seam.
            let samples: Vec<f32> = audio.data.row(chan).iter().map(|&x| x as f32).collect();
            let mut input = pipeline.build_input_sequence(&samples).clone();

            // Input normalization over the whole sequence, ONCE (matching the exact
            // feed_forward_backward top, blstm.rs:1251-1272, before the windowed
            // dispatch): type -1 self-normalization, type 1 external pack-carried
            // mean/std (Phase 8 S1.1, the frozen-stats mode), or type 0 NOTHING (the
            // exact `_ => {}` arm). The construction bail guarantees no other value
            // reaches here; the `0` arm is spelled out rather than folded into the
            // catch-all, because "do nothing" and "self-normalize" are not
            // interchangeable defaults.
            match self.input_normalization_type {
                1 => external_normalize_f32(&mut input, &norm_mean, &norm_std),
                -1 => self_normalize_f32(&mut input),
                _ => {}
            }

            // The forward, per net shape. BLSTM: the overlap accumulation into the
            // SHARED result_buf (seeded from the prior channel; NOT zeroed here -- the
            // cross-channel reuse quirk). CAUSAL: the plain whole-sequence forward,
            // whose posteriors are COPIED into a result_buf that is zeroed first --
            // mirroring the exact plain path, where `NeuronLayer::feed_forward` does
            // its own `output.fill(0.0)` (`nn/layers.rs:948`) and then writes exactly
            // `output_length` rows, so any tail beyond the net's own output length
            // stays zero and NOTHING seeds from the previous channel.
            match net {
                FastSadNet::Blstm(n) => {
                    n.feed_forward_overlap(&input, window_size, window_shift, &mut result_buf);
                }
                FastSadNet::Causal(n) => {
                    let out = n.feed_forward(&input);
                    copy_plain_output("fast causal SAD", out, &mut result_buf)?;
                }
                // Phase-10 Task 7: the bidirectional cells run EITHER regime, selected
                // by the config exactly as the exact tree selects it
                // (`feed_forward_backward`'s `truncates_sequence && overlaps` dispatch,
                // `nn/blstm.rs:1373-1387`). `no_overlap` is refused above, so
                // `window_size > 0` here means OVERLAP and nothing else.
                FastSadNet::BiCell(n) => {
                    if window_size > 0 {
                        n.feed_forward_overlap(&input, window_size, window_shift, &mut result_buf);
                    } else {
                        let out = n.feed_forward(&input);
                        copy_plain_output("fast bidirectional SAD", out, &mut result_buf)?;
                    }
                }
            }

            // Widen the post-division f32 result to f64 (the posterior seam). For the
            // binary SAD net (output_size 1) this is the single result column.
            let result_vec2: Vec<f64> = result_buf.data.iter().map(|&x| x as f64).collect();
            last_rows.push(result_vec2.clone());

            // Shared, UNCHANGED f64 decision layer (results_to_segmentation + smoothing),
            // so boundaries + scoring flow exactly as the exact path's.
            let mut rv = result_vec2;
            results_to_segmentation(
                seg,
                time_step,
                time_offset,
                &mut rv,
                SegClass::Speech,
                self.driver_cfg.conv_coeff.as_deref(),
                &self.seg_cfg,
            );
        }
        self.last_result_rows = last_rows;

        // compute_errors after the full channel loop (tasks/sad.rs:1616): sanitize +
        // update_count each hypothesis (the no-reference side effect the exact driver
        // also runs).
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }

        Ok(())
    }
}

// ===========================================================================
// FastLidNet -- the Twin's LID net on the (cell x direction) matrix (issue #57).
// ===========================================================================

/// The Twin's Mode-7 LID net actually built, per [`FastNetShape`] -- the LID-side
/// sibling of [`FastSadNet`], one more consumer of the two cell twins. Merging the two
/// enums belongs with issue #24's collapse of the fast-tree net enums, not this one.
///
/// THE SCORING REGIME FOLLOWS THE SHAPE, as it does for the SAD net but with the LID
/// regimes: a BIDIRECTIONAL shape (`Blstm`, `BiCell`) runs the TwoSweeps/single-sweep
/// TRUNCATE windowing the committed LID configs resolve to (`BLSTM_LID_window 0.25`), or
/// the PLAIN whole-sequence forward (`window 0`); a CAUSAL shape runs PLAIN only (the
/// launcher writes `BLSTM_LID_window 0` on a forward LID net; a causal cell inside a
/// window has its state reset at every boundary, the SAD driver's rule). OVERLAP is
/// refused on every shape. Both refusals are raised where the window resolves against
/// the rate (`FastTwinLid::get_segmentation` / `lid_score_params`), not at construction.
///
/// `large_enum_variant` allowed on the `FastSadNet` precedent: one per Twin, borrowed
/// thereafter.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum FastLidNet {
    Blstm(FastBlstm),
    Causal(FastCausalNet),
    BiCell(FastBiCell),
}

/// The sub-sampling geometry the truncate sweep sizes its windows and output rows off.
struct TruncateGeometry<'a> {
    lstm_subsampling: &'a [usize],
    output_subsampling: &'a [usize],
    output_size: usize,
}

impl TruncateGeometry<'_> {
    /// `getSubSamplingRatio`: the recurrent ratios times the output-net ones.
    fn ssr(&self) -> usize {
        self.lstm_subsampling.iter().product::<usize>()
            * self.output_subsampling.iter().product::<usize>()
    }

    /// `len` divided SEQUENTIALLY by each recurrent ratio then each output ratio, the
    /// exact tree's integer-floor chain, applied only when the whole ratio is > 1.
    fn short_len(&self, len: usize) -> usize {
        let mut l = len;
        if self.ssr() > 1 {
            for &r in self.lstm_subsampling {
                l /= r;
            }
            for &r in self.output_subsampling {
                l /= r;
            }
        }
        l
    }
}

impl FastLidNet {
    fn output_size(&self) -> usize {
        match self {
            FastLidNet::Blstm(n) => n.output_size(),
            FastLidNet::Causal(n) => n.output_size(),
            FastLidNet::BiCell(n) => n.output_size(),
        }
    }

    /// ONE whole block through the shape's own forward; the posteriors stay in the net's
    /// reused output buffer (valid until the next call), as the phase-7 sweep read them.
    fn feed_forward(&mut self, block: &FastMatrix) -> &FastMatrix {
        match self {
            FastLidNet::Blstm(n) => n.feed_forward(block),
            FastLidNet::Causal(n) => n.feed_forward(block),
            FastLidNet::BiCell(n) => n.feed_forward(block),
        }
    }

    /// Scoring forward for the Mode-7 LID Twin, the f32 counterpart of the INFERENCE
    /// slice of `BlstmNetwork::feed_forward_scoring` (`nn/blstm.rs:1671-1766`; the `:NNN`
    /// cites below are that function's lines) on every shape. Mode 7 always passes
    /// `target_index >= 0`, so this mirrors: the sequential output-length division
    /// (`:1686-1694`), the windowed dispatch (`set_processing_type(window_size > 0,
    /// overlaps)`: `window_size == 0` is the PLAIN whole-sequence forward, `TwoSweeps`
    /// IGNORED exactly as the exact tree ignores it there; `window_size > 0` is the
    /// TRUNCATE sweep, the caller having refused overlap), and the BINARY expansion into
    /// `[1-p, p]` when `output_size == 1`. The `rows < ssr -> Zero(1, O)` guard
    /// (`:1681-1683`) is the caller's: `score_lid_entry` skips such an entry. The exact's
    /// target construction + cost/backward are SKIPPED: the fast path is forward-only, the
    /// forward is target-independent, and Mode 7's langID/confusion derive from the
    /// posteriors alone (the NN-cost columns are a documented divergence).
    ///
    /// Through phase 11 this body was `FastBlstm::feed_forward_scoring`; issue #57 moved
    /// it here unchanged for the BLSTM shape (the phase-7 LID legs at their pins are the
    /// proof) and generalized the block forward. The sub-sampling ratios come from the
    /// caller's [`LidScoreParams`] (the `lid_spec` every shape's `from_flat` copies).
    pub(crate) fn feed_forward_scoring(
        &mut self,
        input: &FastMatrix,
        lstm_subsampling: &[usize],
        output_subsampling: &[usize],
        window_size: usize,
        two_sweeps: bool,
    ) -> FastMatrix {
        let output_size = self.output_size();
        let geom = TruncateGeometry {
            lstm_subsampling,
            output_subsampling,
            output_size,
        };
        // :1686-1694 output length (only re-divided when the whole ratio > 1).
        let out_len = geom.short_len(input.rows);
        let output = if window_size == 0 {
            // The PLAIN regime (`feed_forward_backward_plain` -> `feed_forward` over the
            // whole entry, `nn/blstm.rs:1780`): the net's own output length is the same
            // sequential floor chain, so the rows agree by construction; a disagreement is
            // a geometry defect, not something to pad or truncate quietly.
            let out = self.feed_forward(input);
            assert_eq!(
                (out.rows, out.cols),
                (out_len, output_size),
                "fast LID plain forward: net output shape vs the scoring length"
            );
            FastMatrix {
                data: out.data[..out_len * output_size].to_vec(),
                rows: out_len,
                cols: output_size,
            }
        } else {
            self.truncate_forward(&geom, input, window_size, out_len, two_sweeps)
        };
        // :1755-1765 binary expansion into [1-p, p] (Mode 7 always has targets, so the
        // `target_index >= 0` half of the legacy gate is always true here).
        if output_size == 1 {
            let mut expanded = FastMatrix::zeros(out_len, 2);
            for ii in 0..out_len {
                let p = output.data[ii];
                expanded.data[ii * 2] = 1.0 - p;
                expanded.data[ii * 2 + 1] = p;
            }
            expanded
        } else {
            output
        }
    }

    /// f32 TwoSweeps/single-sweep truncate forward (`feed_forward_backward_truncate`,
    /// `nn/blstm.rs:2035`; the `:NNN` cites in the body are the legacy
    /// `BLSTMNeuralNetwork.cpp:548-590` lines that function transcribes), OUTPUT-ONLY. The
    /// exact stitches the fwd/bwd hidden states (`_OutputForward`/`_OutputBackward`) across
    /// windows + sweeps; Mode 7 reads them ONLY for `DumpLIDInternals` (off on the gate
    /// configs, the fast driver bails if it is on), so the fast path tracks only `output`.
    /// `out_len` is the caller's posterior row count. NET-AGNOSTIC: every shape runs it
    /// through [`Self::feed_forward`], one whole block per window. NOT one of the two
    /// byte-frozen streaming kernels, so lifting it out of `FastBlstm` (issue #57) was a
    /// pure code motion, the phase-7 LID parity legs at their pins the arbiter.
    fn truncate_forward(
        &mut self,
        geom: &TruncateGeometry<'_>,
        input: &FastMatrix,
        window_size: usize,
        out_len: usize,
        two_sweeps: bool,
    ) -> FastMatrix {
        let output_size = geom.output_size;
        if !two_sweeps {
            // :588 single TruncateSweep.
            let mut output = FastMatrix::zeros(out_len, output_size);
            self.truncate_sweep(geom, input, window_size, &mut output);
            return output;
        }

        // :550-552 shift/window sizing (INTEGER-division ORDER: /2 FIRST, then /ssr).
        let ssr = geom.ssr();
        let shift_short = (window_size / 2) / ssr;
        let shift = shift_short * ssr;
        let window_size_short = window_size / ssr;

        // :553-554 input padding (first/last row replicated window_size times).
        let input_padded = pad_replicate_ends_f32(input, window_size, window_size);

        // :565-568 sweep 1: input drops the FRONT window_size padding, keeps the back;
        // sweep1_out is the tail of the zero-padded output (out_len + window_size_short).
        let sweep1_in = slice_rows_f32(&input_padded, window_size, input_padded.rows);
        let mut sweep1_out = FastMatrix::zeros(out_len + window_size_short, output_size);
        self.truncate_sweep(geom, &sweep1_in, window_size, &mut sweep1_out);

        // :572-576 sweep 2: input from `shift`; output written into a fresh zero buffer
        // (output_padded2) offset by shift_short (the exact re-stitches the shifted block
        // back in place after the sweep, so we mirror that write).
        let sweep2_in = slice_rows_f32(&input_padded, shift, input_padded.rows);
        let mut sweep2_buf = FastMatrix::zeros(out_len + 2 * window_size_short, output_size);
        let mut sweep2_out = FastMatrix::zeros(sweep2_buf.rows - shift_short, output_size);
        self.truncate_sweep(geom, &sweep2_in, window_size, &mut sweep2_out);
        for r in 0..sweep2_out.rows {
            let dst = (shift_short + r) * output_size;
            let src = r * output_size;
            sweep2_buf.data[dst..dst + output_size]
                .copy_from_slice(&sweep2_out.data[src..src + output_size]);
        }

        // :578-579 output[r] = (sweep1[r] + sweep2[r]) / 2 over the first out_len rows.
        let mut output = FastMatrix::zeros(out_len, output_size);
        for r in 0..out_len {
            for c in 0..output_size {
                let s1 = sweep1_out.data[r * output_size + c];
                let s2 = sweep2_buf.data[(window_size_short + r) * output_size + c];
                output.data[r * output_size + c] = (s1 + s2) / 2.0;
            }
        }
        output
    }

    /// One truncate sweep (`feed_forward_backward_truncate_sweep`, `nn/blstm.rs:1912`; the
    /// `:NNN` cites are the legacy `BLSTMNeuralNetwork.cpp:488-546` lines), OUTPUT-ONLY:
    /// non-overlapping windows of `window_size`, each a whole-block forward over a
    /// contiguous row-slice, written at `begin/ssr`. A `length_short == 0` window is
    /// silently dropped (`:534`); a partial last window recomputes its length via the
    /// sequential sub-sampling floors (`:518-532`).
    fn truncate_sweep(
        &mut self,
        geom: &TruncateGeometry<'_>,
        input: &FastMatrix,
        window_size: usize,
        output: &mut FastMatrix,
    ) {
        let ssr = geom.ssr();
        let cols = output.cols;

        // :500-512 nominal length (window /= each LSTM then Output ratio when sub on).
        let nominal_len = geom.short_len(window_size);

        let input_rows = input.rows;
        let mut jj = 0;
        while jj < input_rows {
            let begin = jj;
            let mut end = jj + window_size - 1;
            if end >= input_rows {
                end = input_rows - 1;
            }
            let length_seq = end - begin + 1;
            let length_short = if length_seq != window_size {
                geom.short_len(length_seq)
            } else {
                nominal_len
            };
            if length_short > 0 {
                let block = slice_rows_f32(input, begin, begin + length_seq);
                let out_short = self.feed_forward(&block);
                debug_assert_eq!(out_short.rows, length_short, "truncate sweep row count");
                let obeg = begin / ssr;
                for r in 0..length_short {
                    let dst = (obeg + r) * cols;
                    let src = r * out_short.cols;
                    output.data[dst..dst + cols].copy_from_slice(&out_short.data[src..src + cols]);
                }
            }
            jj += window_size;
        }
    }
}

/// The flat element count the Twin's LID net needs for a classified [`FastNetShape`]:
/// the same count each shape's `from_flat` sizes against.
fn lid_element_count(
    spec: &NnetSpec,
    shape: FastNetShape,
    mamba: &MambaParams,
    cfc: &CfcParams,
    transformer: &TransformerParams,
) -> Result<usize> {
    Ok(match shape {
        FastNetShape::Blstm => crate::config::element_count(spec),
        FastNetShape::Causal(cell) => {
            FastCausalNet::element_count(spec, cell, mamba, cfc, transformer)?
        }
        FastNetShape::BiCell(cell) => {
            FastBiCell::element_count(spec, cell, mamba, cfc, transformer)?
        }
    })
}

/// Build the Twin's LID net for a classified [`FastNetShape`] from the flat f64 pack.
/// ONE place, so `from_legacy(map, _, Some(flat))` and the deferred `load_weights_file`
/// cannot pick different arms (the [`build_sad_net`] rule). Each shape's `from_flat`
/// refuses a SHORT pack only and consumes an over-long one head-first; the over-long
/// refusal is the caller's, per seam (the phase-11 split): `from_legacy`'s in-memory arm
/// demands the exact length, as the exact Twin's `set_weights` does, and
/// `load_weights_file` keeps the legacy file-load tolerance with its warning.
fn build_lid_net(
    spec: &NnetSpec,
    shape: FastNetShape,
    mamba: &MambaParams,
    cfc: &CfcParams,
    transformer: &TransformerParams,
    flat: &[f64],
) -> Result<FastLidNet> {
    Ok(match shape {
        FastNetShape::Blstm => FastLidNet::Blstm(FastBlstm::from_flat(spec, flat)?),
        FastNetShape::Causal(cell) => FastLidNet::Causal(FastCausalNet::from_flat(
            spec,
            cell,
            mamba,
            cfc,
            transformer,
            flat,
        )?),
        FastNetShape::BiCell(cell) => FastLidNet::BiCell(FastBiCell::from_flat(
            spec,
            cell,
            mamba,
            cfc,
            transformer,
            flat,
        )?),
    })
}

// ===========================================================================
// FastTwinLid -- the f32 Mode-7 LID Twin (algo 6).
// ===========================================================================

/// Read one `i32` config scalar with a default (mirrors `twin_i32_default`).
fn twin_i32_default(map: &IndexMap<String, String>, key: &str, default: i32) -> Result<i32> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<i32>()
            .map_err(|e| anyhow!("`{key}`: cannot parse: {e}")),
    }
}

/// Read one `bool` config scalar with a default (mirrors `twin_bool_default`).
fn twin_bool_default(map: &IndexMap<String, String>, key: &str, default: bool) -> Result<bool> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<bool>()
            .map_err(|e| anyhow!("`{key}`: cannot parse as bool: {e}")),
    }
}

/// f32 Mode-7 LID Twin (algo 6), the fast counterpart of
/// [`crate::tasks::lid::TwinBlstmSpectralLid`] restricted to its `abs(_Mode) == 7`
/// phSeq/cep arm (`tasks/lid.rs::get_segmentation_mode7`, `:1270-1655`). The SAD net is
/// NEVER RUN (its `result_vec` is the synthesized constant 10.0, `:1407`); only the LID
/// net scores each `audio.external_features[i]` block, in f32. The langID / confusion /
/// `is_lid_correct` decisions derive from the LID posteriors alone -- widened back to
/// f64 at the member seam so the shared `.scr` writer + confusion columns flow unchanged.
/// Since issue #57 the LID net is any shape of the (cell x direction) matrix
/// ([`FastLidNet`]), not the phase-7 BLSTM twin alone.
///
/// SCOPE (the house typed-bail pattern -- Mode 7 phSeq/cep is what the gate configs
/// exercise, everything else typed-bails loudly):
/// - Only `_Mode == 7` (`from_legacy`); the wav CNN arm (`periodogram.is_none()`), the
///   pitch second pass (`TDCwindow > 0`), and `DumpLIDInternals` typed-bail.
/// - The LID net's InputNormalizationType must be 0 (the gate configs' value; a non-0
///   type would need the normalization the fast scoring skips) -- bail at construction.
/// - A negative `BLSTM_LID_TargetEnforcementStep` typed-bails at construction (T5 finding
///   1, review): it makes the exact path's truncate-windowed cost block overwrite interior
///   OUTPUT rows with -0.5 (risk R7, `nn/blstm.rs:1244-1265`), which the fast path's
///   forward-only scoring never reproduces; the gate configs use a non-negative step.
/// - MLP mode (`BLSTM_LID_LSTMNeuronNb[0] == 0`) typed-bails at construction (T5 finding 3,
///   review, contract symmetry with the exact dispatch's is_mlp route to the MLP drivers).
/// - The LID windowed dispatch (issue #57, D2): TRUNCATE (`lid_window_size > 0`,
///   `lid_no_overlap true`, what the committed bidirectional LID configs give) or PLAIN
///   (`lid_window_size == 0`, what the launcher writes on a forward LID net) on a
///   bidirectional shape; PLAIN only on a causal one; OVERLAP bails on every shape --
///   both refusals at `get_segmentation` / `lid_score_params`, where the window resolves.
/// - The LID net's `Cell_Type` x `Direction` pair is NOT a bail any more: [`FastLidNet`]
///   runs the whole (cell x direction) matrix. The SAD net's pair is never read for a
///   kernel (frozen-SAD: only its sub-sampling ratios size the SAD result vector).
///
/// FORWARD-ONLY: `cumulative_error`/`nb_of_classif` (SAD, always 0 in Mode 7 anyway) and
/// `lid_cumulative_error`/`lid_nb_of_classif` (the LID NN-cost/count columns) stay ZERO
/// -- a documented divergence from the exact driver (the fast path never harvests cost),
/// matching the algo-3 fast SAD driver. The langID-derived members are the parity target.
#[derive(Clone)]
pub struct FastTwinLid {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    feature_cfg: FeatureConfig,

    /// SAD-net sub-sampling, cached from its spec for [`get_blstm_param`]'s SAD
    /// result-vec sizing / timeStep. The SAD net is never run, so only its shape is
    /// needed (no weights, no `FastBlstm`).
    sad_lstm_sub: Vec<usize>,
    sad_out_sub: Vec<usize>,
    sad_ssr: usize,

    /// The LID net (the only net actually run) + its spec (for the deferred
    /// `load_weights_file` rebuild). `lid_ssr`/`class_nb`/`lid_two_sweeps` are cached
    /// off the spec/config so the window derivation needs no live net borrow.
    lid_spec: NnetSpec,
    /// The LID net SHAPE the config selected ([`classify_fast_shape`] on the `BLSTM_LID`
    /// prefix) + the three cell geometries, carried so the deferred `load_weights_file`
    /// rebuild sizes and picks exactly as `from_legacy` did (the SAD driver's rule).
    lid_shape: FastNetShape,
    lid_mamba: MambaParams,
    lid_cfc: CfcParams,
    lid_transformer: TransformerParams,
    lid_net: Option<FastLidNet>,
    lid_ssr: usize,
    class_nb: usize,
    lid_two_sweeps: bool,

    /// Mode-7 LID scalars (`BLSTM_LID_*`).
    lid_window_size_sec: f64,
    lid_window_shift_sec: f64,
    mode: i32,
    post_process_mode: i32,
    min_nb_of_frames: i32,
    noise_magnitude: f64,
    /// `CostLaw::isCostModified` (`_BackPropWER >= 0`), gating the langID normalization.
    cost_modified: bool,

    /// Stateful spectrum-shift/window members, re-quantized per `get_segmentation`
    /// exactly as the exact driver does.
    spectrum_shift_sec: f64,
    spectrum_shift_in_frames: usize,
    window_shift_sec: f64,
    ltsv_shift_sec: f64,

    channels: usize,
    cumulative_error: Vec<f64>,
    nb_of_classif: Vec<i64>,
    last_result_rows: Vec<Vec<f64>>,
    lid_cumulative_error: Vec<f64>,
    lid_nb_of_classif: Vec<i64>,
    lid_classification_errors: Vec<Vec<f64>>,
    lid_segments_confusion: Vec<Array2<f64>>,
    is_lid_correct: Vec<i32>,
}

impl FastTwinLid {
    /// Build from a legacy config map + optional SAD/LID weight packs, mirroring
    /// [`crate::tasks::lid::TwinBlstmSpectralLid::from_legacy`]'s surface. `sad_weights`
    /// is accepted for signature symmetry but UNUSED (the SAD net is never run in Mode
    /// 7). `lid_weights: Some(flat)` builds the LID net ([`FastLidNet`], the shape
    /// [`classify_fast_shape`] selects on the `BLSTM_LID` prefix) immediately; `None`
    /// defers to [`Self::load_weights_file`].
    ///
    /// Typed-bails (loudly, at construction) the unsupported fast surfaces: any mode but
    /// 7, the pitch second pass (`TDCwindow > 0`), a LID `InputNormalizationType != 0`,
    /// and `DumpLIDInternals` (the fast path does not track the LID hidden states the
    /// dump emits). The LID net's `Cell_Type` x `Direction` pair is no longer among them
    /// (issue #57).
    pub fn from_legacy(
        map: &IndexMap<String, String>,
        _sad_weights: Option<&[f64]>,
        lid_weights: Option<&[f64]>,
    ) -> Result<FastTwinLid> {
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;

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
        // result vector, and every cell carries those. Peephole-aligned spec, but no net
        // and no shape gate (issue #57 removed the BLSTM-only one).
        let sad_spec = build_aligned_spec(map, "BLSTM")?;
        let sad_lstm_sub = sad_spec.lstm_subsampling.clone();
        let sad_out_sub = sad_spec.output_subsampling.clone();
        let sad_ssr =
            sad_lstm_sub.iter().product::<usize>() * sad_out_sub.iter().product::<usize>();

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

        // T5 finding 1 (review): Mode 7's target_index (the clamped audio.lang_index) is
        // ALWAYS >= 0, so the exact scoring path (`BlstmNetwork::feed_forward_scoring`)
        // always builds a non-empty target sequence -- gated on targets being PRESENT, not
        // on backprop being active. For the truncate windowing both gate configs resolve
        // to, that target sequence is fed through `feed_forward_backward_truncate` ->
        // `feed_forward_backward_truncate_sweep`, which runs `feed_forward_backward_plain`
        // per window block. `_TargetEnforcementStep < 0` there makes the COST block
        // (nn/blstm.rs:1244-1265) overwrite interior rows of the caller-visible OUTPUT
        // with -0.5 before `computeCost` -- not just the target. The fast path never
        // builds targets, so it always returns true posteriors: a silent structural
        // divergence for a negative step. Bail loudly instead.
        if lid_bc.target_enforcement_step < 0 {
            bail!(
                "fast TwinLid: a negative BLSTM_LID_TargetEnforcementStep is not supported on \
                 the fast path (got {}); the exact path's cost block overwrites interior \
                 output rows with -0.5 for a negative step (risk R7, nn/blstm.rs:1244-1265), \
                 which the fast path's forward-only scoring never reproduces; the gate configs \
                 use a non-negative step",
                lid_bc.target_enforcement_step
            );
        }

        // T5 finding 3 (review, minor): contract symmetry with the exact dispatch, which
        // routes is_mlp configs (LSTMNeuronNb[0]==0) to the MLP drivers
        // (nn/blstm.rs:1032-1043) instead of the recurrent forwards the fast LID scoring
        // runs. Every shape's `from_flat` independently bails on this shape, but only once
        // weights are actually loaded -- when `lid_weights` is `None` here (the deferred
        // `load_weights_file` path), `from_legacy` would otherwise return `Ok` for an
        // is_mlp config, so `from_flat`'s bail does NOT provably fire first on every such
        // config. Bail on the SAME derived flag the exact dispatch gates on so
        // construction fails loudly regardless of when weights arrive.
        if lid_bc.is_mlp {
            bail!(
                "fast TwinLid: MLP mode (BLSTM_LID_LSTMNeuronNb[0] == 0) is not supported on \
                 the fast path (the fast LID scoring runs the recurrent forwards only, never \
                 the MLP drivers); the gate configs use a non-zero LSTM width"
            );
        }

        let cost_modified = lid_bc.cost_law.is_cost_modified();
        let lid_two_sweeps = lid_bc.two_sweeps;

        // Cell x direction dispatch on the LID prefix (issue #57): the same TOTAL
        // classifier the SAD driver uses, off the same peephole-aligned spec.
        let lid_shape = classify_fast_shape(&lid_bc);
        let lid_spec = build_spec_aligned_to(map, "BLSTM_LID", &lid_bc)?;
        let lid_ssr = lid_spec.lstm_subsampling.iter().product::<usize>()
            * lid_spec.output_subsampling.iter().product::<usize>();
        // class_nb = LID output_size max 2 (the langID / confusion dimensionality).
        let class_nb = (*lid_spec.output_neuron_nb.last().unwrap()).max(2);

        let lid_net = match lid_weights {
            // EXACT-LENGTH, as the exact Twin's `set_weights` is since the phase-11
            // interstitial: an over-long in-memory pack is another architecture's, not a
            // file to tolerate. The short side stays `from_flat`'s "too short".
            Some(flat) => {
                let needed = lid_element_count(
                    &lid_spec,
                    lid_shape,
                    &lid_bc.mamba,
                    &lid_bc.cfc,
                    &lid_bc.transformer,
                )?;
                if flat.len() > needed {
                    bail!(
                        "The number of gains given is more than what's needed ({} > {needed}).",
                        flat.len()
                    );
                }
                Some(build_lid_net(
                    &lid_spec,
                    lid_shape,
                    &lid_bc.mamba,
                    &lid_bc.cfc,
                    &lid_bc.transformer,
                    flat,
                )?)
            }
            None => None,
        };

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

        let spectrum_shift_sec = feature_cfg.shift_sec;
        let window_shift_sec = driver_cfg.window_shift_sec;
        let ltsv_shift_sec = feature_cfg.ltsv_shift;

        Ok(FastTwinLid {
            driver_cfg,
            seg_cfg,
            feature_cfg,
            sad_lstm_sub,
            sad_out_sub,
            sad_ssr,
            lid_spec,
            lid_shape,
            lid_mamba: lid_bc.mamba,
            lid_cfc: lid_bc.cfc,
            lid_transformer: lid_bc.transformer,
            lid_net,
            lid_ssr,
            class_nb,
            lid_two_sweeps,
            lid_window_size_sec,
            lid_window_shift_sec,
            mode,
            post_process_mode,
            min_nb_of_frames,
            noise_magnitude,
            cost_modified,
            spectrum_shift_sec,
            spectrum_shift_in_frames: 0,
            window_shift_sec,
            ltsv_shift_sec,
            channels: 0,
            cumulative_error: Vec::new(),
            nb_of_classif: Vec::new(),
            last_result_rows: Vec::new(),
            lid_cumulative_error: Vec::new(),
            lid_nb_of_classif: Vec::new(),
            lid_classification_errors: Vec::new(),
            lid_segments_confusion: Vec::new(),
            is_lid_correct: Vec::new(),
        })
    }

    /// `BLSTM_LID_weightsFile` load (mirrors the exact Twin's LID-net load): an EMPTY key
    /// leaves the net unloaded (a subsequent `get_segmentation` errors); otherwise read
    /// the `.bin` and (re)build the LID net through the SAME [`build_lid_net`] the ctor
    /// uses, so the deferred load cannot pick a different arm than `from_legacy`
    /// classified.
    pub fn load_weights_file(&mut self, map: &IndexMap<String, String>) -> Result<()> {
        let weights_file = map
            .get("BLSTM_LID_weightsFile")
            .map(String::as_str)
            .unwrap_or("");
        if weights_file.is_empty() {
            return Ok(());
        }
        let flat = crate::io::binary::read_weight_vector(std::path::Path::new(weights_file))?;
        // The legacy file-load tolerance (`BLSTMNeuralNetwork.cpp:144-146`) with the warning
        // the exact `BlstmNetwork::load_weights_file` prints; `from_flat` consumes the head.
        let needed = lid_element_count(
            &self.lid_spec,
            self.lid_shape,
            &self.lid_mamba,
            &self.lid_cfc,
            &self.lid_transformer,
        )?;
        if flat.len() > needed {
            eprintln!(
                "Warning: The number of gains given in {weights_file} is more than what's needed \
                 ({} > {needed}); the extra {} are ignored.",
                flat.len(),
                flat.len() - needed
            );
        }
        self.lid_net = Some(build_lid_net(
            &self.lid_spec,
            self.lid_shape,
            &self.lid_mamba,
            &self.lid_cfc,
            &self.lid_transformer,
            &flat,
        )?);
        Ok(())
    }

    /// The LID net shape the config selected (the `BLSTM_LID` prefix through
    /// [`classify_fast_shape`]). The dispatch legs pin that each (cell, direction) pair
    /// builds its own arm.
    pub fn lid_shape(&self) -> FastNetShape {
        self.lid_shape
    }

    /// The configured dump directory (gates the scored-branch VRCTS write in the bag).
    pub fn dump_dir(&self) -> &str {
        &self.driver_cfg.dump_dir
    }

    pub fn mode(&self) -> i32 {
        self.mode
    }
    pub fn post_process_mode(&self) -> i32 {
        self.post_process_mode
    }

    /// Per-channel SAD NN cost (result col 4). ALWAYS zero in Mode 7 (the SAD net is
    /// never run), matching the exact driver's `:1421`.
    pub fn cumulative_error(&self) -> &[f64] {
        &self.cumulative_error
    }
    pub fn nb_of_classif(&self) -> &[i64] {
        &self.nb_of_classif
    }
    /// SAD `result_vec2` per channel (the synthesized constant-10.0 row, PRE-convolution)
    /// -- the port-side observation point, mirroring the exact driver's `last_result_rows`.
    pub fn last_result_rows(&self) -> &[Vec<f64>] {
        &self.last_result_rows
    }
    /// Per-channel LID NN cost. ALWAYS zero on the fast path (forward-only), a documented
    /// divergence from the exact driver.
    pub fn lid_cumulative_error(&self) -> &[f64] {
        &self.lid_cumulative_error
    }
    /// Per-channel LID `_NbOfClassif`. ALWAYS zero on the fast path (forward-only).
    pub fn lid_nb_of_classif(&self) -> &[i64] {
        &self.lid_nb_of_classif
    }
    /// Per-channel `100 * (langID - targetLID)` in-band LID scores (the `.scr`/confusion
    /// column source). The f32-divergent parity target.
    pub fn lid_classification_errors(&self) -> &[Vec<f64>] {
        &self.lid_classification_errors
    }
    pub fn lid_segments_confusion(&self) -> &[Array2<f64>] {
        &self.lid_segments_confusion
    }
    pub fn is_lid_correct(&self) -> &[i32] {
        &self.is_lid_correct
    }

    /// Test hook: the LID net's effective per-direction peephole flags (rider 2 -- wires
    /// the `FastBlstm::debug_peepholes` hook into a Task-5 pin, proving the LID net's
    /// spec was peephole-aligned under the `BLSTM_LID` prefix, not left at the
    /// `NnetSpec` default, FALSE until issue #32). `None` unless the LID net is the
    /// bidirectional peephole-LSTM twin (`FastLidNet::Blstm`). A causal `lstm` LID net
    /// carries the forward peephole triple too (`FastLstm::peep_flags` through
    /// `FastCausalNet::cells()`), but this hook does not report it, so its alignment is
    /// unpinned here; the other cells carry no peepholes.
    #[cfg(feature = "test-support")]
    pub fn debug_lid_peepholes(&self) -> Option<[bool; 6]> {
        match self.lid_net.as_ref() {
            Some(FastLidNet::Blstm(n)) => Some(n.debug_peepholes()),
            _ => None,
        }
    }
}

impl Segmenter for FastTwinLid {
    /// Port of the `abs(_Mode) == 7` external-features LID loop
    /// (`tasks/lid.rs::get_segmentation_mode7`), f32. The SAD net is synthesized (constant
    /// 10.0 -> the SHARED f64 decision layer, so the SAD side is bit-identical to the
    /// exact path); the LID net runs the fast scoring forward (plain or truncate, per the
    /// config and the shape) per external-features block.
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        _refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        // Wav CNN arm bail (the exact `:1276-1280` gate): only the phSeq/cep external-
        // features arm is ported; a real wav decode leaves `periodogram` unset.
        if audio.periodogram.is_none() {
            bail!(
                "fast TwinLid mode 7: the wav arm runs the CNN (not ported); only File_Type 1 \
                 (phSeq) / 2 (cep) external-features input is supported"
            );
        }
        let rate = audio.sample_rate as f64;
        let s = SpectralParams::derive(&self.feature_cfg, rate);
        if let Some(tdc) = s.tdc.as_ref()
            && tdc.half_window > 0
        {
            bail!("fast TwinLid mode 7: the pitch pass (TDCwindow > 0) is not supported");
        }

        // Stateful spectrum-shift quantization + the `!hasReadWavFile` -> 80 override
        // (the exact `:1296-1301`): `periodogram.is_some()` is the port's `!hasReadWavFile`.
        self.spectrum_shift_in_frames = f64::round(self.spectrum_shift_sec * rate) as usize;
        if audio.periodogram.is_some() {
            self.spectrum_shift_in_frames = 80;
        }
        self.spectrum_shift_sec = self.spectrum_shift_in_frames as f64 / rate;
        let ssif = self.spectrum_shift_in_frames;
        let spectrum_shift_sec = self.spectrum_shift_sec;

        // preemph/noise (harmless on the zeroed phSeq/cep waveform).
        if self.feature_cfg.preemph_ratio > 0.0 {
            audio.apply_preemph(self.feature_cfg.preemph_ratio);
        }
        if self.feature_cfg.noise_seed > 0 {
            audio.apply_noise(self.feature_cfg.noise_ratio);
        }

        // getLTSVParam re-quantize (dead for LTSVwindow 0).
        let ltsv_ws = f64::round(self.feature_cfg.ltsv_shift * rate / ssif as f64) as i64;
        let ltsv_ws = if ltsv_ws < 1 { 1 } else { ltsv_ws };
        self.ltsv_shift_sec = (ltsv_ws * ssif as i64) as f64 / rate;

        // getBLSTMParam: SAD result_vec sizing + timeStep (mutates self.window_shift_sec).
        let sad_ssr = self.sad_ssr;
        let sad_lstm_sub = self.sad_lstm_sub.clone();
        let sad_out_sub = self.sad_out_sub.clone();
        let (window_size, _sad_window_shift, no_overlap, real_vec_size) = get_blstm_param(
            self.driver_cfg.window_size_sec,
            &mut self.window_shift_sec,
            rate,
            ssif,
            sad_ssr,
            &sad_lstm_sub,
            &sad_out_sub,
            audio.data.ncols(),
        );

        // getLIDBLSTMParam: LID window/shift derivation (the exact `:1330-1353`), the regime
        // rule and the target clamp (the exact `:1396-1404`), all through the streaming
        // session's own `lid_score_params`, so the offline and streaming params cannot drift.
        // Called BEFORE the stateful shift re-quantization below: re-derived from a
        // re-quantized shift (>= 1 frame), truncate would read as overlap.
        // `lid_window_shift` is still derived inline for the stateful
        // `self.lid_window_shift_sec` bookkeeping (the epilogue reset at `:1160` reads it).
        let params = self.lid_score_params("fast TwinLid mode 7", rate, audio.lang_index)?;
        // Overlap is refused, so a resolved window means truncate (`lid_no_overlap`).
        let lid_no_overlap = params.lid_window_size > 0;
        let mut lid_window_shift =
            f64::round(self.lid_window_shift_sec * rate / ssif as f64) as i64;
        if params.lid_window_size == 0 || lid_window_shift < 1 {
            lid_window_shift = 1;
        }
        self.lid_window_shift_sec = (lid_window_shift * ssif as i64) as f64 / rate;

        // SAD timeStep/offset (the exact `:1356-1366`).
        let mut time_step = self.window_shift_sec * sad_ssr as f64;
        let mut time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
        if window_size > 0 {
            if no_overlap {
                time_step = self.window_shift_sec * sad_ssr as f64;
                time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
            } else {
                time_step = spectrum_shift_sec * sad_ssr as f64;
                time_offset = time_step / 2.0 - spectrum_shift_sec / 2.0;
            }
        }

        let conv_coeff = self.driver_cfg.conv_coeff.clone();
        let external_features = audio.external_features.clone();

        let channels = audio.data.nrows();
        self.channels = channels;
        self.cumulative_error = vec![0.0; channels];
        self.nb_of_classif = vec![0; channels];
        self.last_result_rows = Vec::with_capacity(channels);
        self.lid_cumulative_error = vec![0.0; channels];
        self.lid_nb_of_classif = vec![0; channels];
        self.lid_classification_errors = vec![Vec::new(); channels];
        self.lid_segments_confusion = vec![Array2::<f64>::zeros((0, 0)); channels];
        self.is_lid_correct = vec![0; channels];

        // Move the LID net out so self's result buffers are freely writable in the loop
        // (put back before returning; the loop body has no fallible op, so no early exit).
        let mut net = self
            .lid_net
            .take()
            .ok_or_else(|| anyhow!("fast TwinLid: LID net has no weights loaded"))?;

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // SAD else branch (`:1407`): result_vec = constant 10.0 -> the SHARED f64
            // decision layer (bit-identical to the exact path -- no NN, pure f64).
            let mut result_vec2 = vec![10.0f64; real_vec_size];
            self.last_result_rows.push(result_vec2.clone());
            results_to_segmentation(
                seg,
                time_step,
                time_offset,
                &mut result_vec2,
                SegClass::Speech,
                conv_coeff.as_deref(),
                &self.seg_cfg,
            );
            self.cumulative_error[chan] = 0.0;
            self.nb_of_classif[chan] = 0;

            // abs(_Mode)==7 external-features loop (`:1448-1591`) + epilogue (`:1599-1634`),
            // via the SHARED kernel (`score_lid_entry` / `finalize_lid_channel`) -- the SAME
            // code the streaming session folds through, so streaming is bit-identical to this
            // offline path by construction (see the kernel section below `get_segmentation`).
            let mut acc = LidChannelAcc::new(params.class_nb);
            for feat in &external_features {
                score_lid_entry(feat, &mut net, &params, &mut acc);
            }
            let agg = finalize_lid_channel(&acc, &params);

            // Member writes (`:1622-1634`). Forward-only: lid_cumulative_error /
            // lid_nb_of_classif stay 0 (documented divergence, the fast path never harvests
            // the LID NN cost/classif count).
            self.lid_cumulative_error[chan] = 0.0;
            self.lid_nb_of_classif[chan] = 0;
            self.lid_classification_errors[chan] = agg.classification_errors;
            self.lid_segments_confusion[chan] = agg.confusion;
            self.is_lid_correct[chan] = agg.is_lid_correct;
        }
        self.lid_net = Some(net);

        // Epilogue (`:1642-1653`). The deriv ponderation (`:1643-1644`) is training-only
        // (forward-only fast path -> skipped). The stateful shift resets are mirrored.
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }
        if no_overlap {
            self.window_shift_sec = 0.0;
        }
        if lid_no_overlap {
            self.lid_window_shift_sec = 0.0;
        }
        Ok(())
    }
}

// ===========================================================================
// The shared Mode-7 LID scoring kernel (Phase 8 Task 7).
//
// The offline `FastTwinLid::get_segmentation` per-channel loop AND the per-utterance
// `StreamingLidSession` (`fast::stream_lid`) fold `external_features` entries through the
// SAME per-entry body ([`score_lid_entry`]) + epilogue ([`finalize_lid_channel`]) below, so
// streaming is bit-identical to offline BY CONSTRUCTION (nothing is re-implemented) -- the
// phase-7 LID parity legs (`phase7_parity_lid.rs`) are the no-offline-regression proof.
// ===========================================================================

/// The finalized per-channel LID members -- the offline `get_segmentation` writes these into
/// its per-channel member vectors, and the streaming session returns this as its
/// utterance-granular result. `classification_errors`/`confusion`/`is_lid_correct` are
/// bit-identical to the offline `lid_classification_errors`/`lid_segments_confusion`/
/// `is_lid_correct`; `predicted_language`/`segments_count` are the streaming-useful extras
/// (the argmax of the normalized langID + the scored-utterance count so far).
#[derive(Debug, Clone, PartialEq)]
pub struct LidAggregate {
    /// In-band `100*(langID - targetLID)` per class (the `.scr`/confusion column source) --
    /// normalized langID, the offline `lid_classification_errors` row.
    pub classification_errors: Vec<f64>,
    /// The per-block argmax-count confusion matrix accumulated so far (the offline
    /// `lid_segments_confusion`; NOT normalized by the epilogue).
    pub confusion: Array2<f64>,
    /// `100` iff the normalized langID argmax is the target, else `0` (offline
    /// `is_lid_correct`).
    pub is_lid_correct: i32,
    /// The predicted language: argmax of the normalized langID (first-max).
    pub predicted_language: usize,
    /// Scored (not skipped) utterances folded so far -- the offline `segments_count`.
    pub segments_count: i32,
}

/// The per-channel LID scoring constants (derived once from config + rate + target
/// language), invariant across a channel's external-features entries.
pub(super) struct LidScoreParams {
    pub(super) lid_ssr: usize,
    pub(super) lid_lstm_sub: Vec<usize>,
    pub(super) lid_out_sub: Vec<usize>,
    pub(super) min_nb_of_frames: i32,
    pub(super) noise_magnitude: f64,
    pub(super) lid_window_size: usize,
    pub(super) two_sweeps: bool,
    pub(super) post_process_mode: i32,
    pub(super) class_nb: usize,
    pub(super) ti: usize,
    pub(super) cost_modified: bool,
}

/// The per-channel LID accumulator the Mode-7 external-features loop folds each entry into.
/// [`new`](Self::new) seeds the confusion index header exactly as the offline `:970-975`.
pub(super) struct LidChannelAcc {
    pub(super) langid: Vec<f64>,
    pub(super) confusion: Array2<f64>,
    pub(super) segments_count: i32,
    pub(super) number_of_frames: i32,
}

impl LidChannelAcc {
    pub(super) fn new(class_nb: usize) -> LidChannelAcc {
        let mut confusion = Array2::<f64>::zeros((class_nb + 2, class_nb + 2));
        for kk in 0..class_nb + 1 {
            confusion[[0, kk]] = kk as f64;
            confusion[[kk, 0]] = kk as f64;
        }
        LidChannelAcc {
            langid: vec![0.0f64; class_nb],
            confusion,
            segments_count: 0,
            number_of_frames: 0,
        }
    }
}

/// One scored utterance's per-entry observables: the offline `seg_lid` (post-
/// `_PostProcessMode` accumulator) + its argmax `j`.
pub(super) struct LidEntryOutcome {
    pub(super) seg_lid: Vec<f64>,
    pub(super) argmax: usize,
}

/// First-max argmax (strict `>`), matching the offline inline argmax (`:1075-1082`,
/// `:1147-1148`-adjacent langID pick).
fn argmax_f64(v: &[f64]) -> usize {
    let mut j = 0usize;
    let mut best = v[0];
    for (c, &x) in v.iter().enumerate().skip(1) {
        if x > best {
            best = x;
            j = c;
        }
    }
    j
}

/// LID window derivation (`get_segmentation_mode7` `:1330-1353`): returns
/// `(lid_window_size, lid_no_overlap)`. `ssif` is the (mode-7 forced) spectrum shift in
/// frames. Shared by `get_segmentation` and [`FastTwinLid::lid_score_params`] so the offline
/// and streaming window sizes cannot drift.
fn derive_lid_window(
    size_sec: f64,
    shift_sec: f64,
    lid_ssr: usize,
    ssif: usize,
    rate: f64,
) -> (usize, bool) {
    let ssifd = ssif as f64;
    let mut lid_window_size = f64::round(size_sec * rate / 2.0 / ssifd) as usize;
    if lid_window_size != 0 && lid_window_size < lid_ssr {
        lid_window_size = lid_ssr;
    }
    let lid_window_shift = f64::round(shift_sec * rate / ssifd) as i64;
    let mut lid_no_overlap = false;
    if lid_window_size != 0 && lid_window_shift < 1 {
        lid_no_overlap = true;
        let raw = f64::round(size_sec * rate / ssifd) as usize;
        lid_window_size = (raw / lid_ssr) * lid_ssr;
        if lid_window_size < 10 * lid_ssr {
            lid_window_size = 10 * lid_ssr;
        }
    }
    (lid_window_size, lid_no_overlap)
}

/// The LID regime rule (issue #57, D2), applied where the window has resolved against the
/// rate -- the offline `get_segmentation` and the streaming `lid_score_params` both call
/// it, so the two refusals cannot drift. A CAUSAL shape runs the PLAIN regime only (a
/// causal cell inside a window has its state reset at every window boundary, the SAD
/// driver's rule; the launcher writes `BLSTM_LID_window 0` on a forward LID net); every
/// shape refuses OVERLAP (the fast LID scoring implements the truncate sweep and the plain
/// forward, nothing accumulates across overlapping windows). A bidirectional shape runs
/// either plain or truncate, exactly as the exact tree dispatches them.
fn check_lid_regime(
    who: &str,
    shape: FastNetShape,
    lid_window_size: usize,
    lid_no_overlap: bool,
) -> Result<()> {
    if let FastNetShape::Causal(cell) = shape
        && lid_window_size != 0
    {
        bail!(
            "{who}: windowed LID inference is unsupported for a causal LID net (cell '{}', \
             BLSTM_LID_window resolves window_size {lid_window_size}); a causal cell's state is \
             reset at every window boundary -- a forward LID net runs the plain regime \
             (BLSTM_LID_window 0, what the launcher writes)",
            cell.as_str()
        );
    }
    if lid_window_size > 0 && !lid_no_overlap {
        bail!(
            "{who}: the overlap LID windowing is unsupported on the fast path (BLSTM_LID_shift \
             resolves >= 1 frame); the fast LID scoring runs the truncate sweep or the plain \
             forward -- the committed LID configs resolve to truncate (shift 0)"
        );
    }
    Ok(())
}

/// Fold ONE external-features entry into `acc` -- the exact body of the Mode-7 `:1448-1591`
/// inner loop. Returns `None` if the entry is skipped (`nrows < lid_ssr` or `< MinNbOfFrames`,
/// `:1450`), else the per-entry observables. The noise `randinit` is fixed 0 PER ENTRY
/// (`:989`), NOT entry-order-dependent, so a per-utterance call reproduces the offline
/// sequential call's noise indexing exactly.
pub(super) fn score_lid_entry(
    feat: &Array2<f64>,
    net: &mut FastLidNet,
    p: &LidScoreParams,
    acc: &mut LidChannelAcc,
) -> Option<LidEntryOutcome> {
    // :1450 _MinNbOfFrames + ssr guard.
    if feat.nrows() < p.lid_ssr || (feat.nrows() as i32) < p.min_nb_of_frames {
        return None;
    }
    // :1456-1466 Gaussian noise (randinit fixed 0), applied in f64 on the block.
    let mut feat_noised = feat.clone();
    if p.noise_magnitude > 0.0 {
        let cols = feat.ncols();
        let randinit = 0usize;
        for r in 0..feat.nrows() {
            for c in 0..cols {
                let m = random_gauss(r * cols + c + randinit) - 0.5;
                feat_noised[[r, c]] += p.noise_magnitude * m;
            }
        }
    }
    // Narrow the block to f32 at the driver seam, then score.
    let (fr, fc) = feat_noised.dim();
    let mut fdata = Vec::with_capacity(fr * fc);
    for r in 0..fr {
        for c in 0..fc {
            fdata.push(feat_noised[[r, c]] as f32);
        }
    }
    let fin = FastMatrix {
        data: fdata,
        rows: fr,
        cols: fc,
    };
    let out_f32 = net.feed_forward_scoring(
        &fin,
        &p.lid_lstm_sub,
        &p.lid_out_sub,
        p.lid_window_size,
        p.two_sweeps,
    );
    let out_cols = out_f32.cols;
    let output_seq: Vec<f64> = out_f32.data.iter().map(|&x| x as f64).collect();
    let get = |r: usize, c: usize| output_seq[r * out_cols + c];

    // :1486-1534 segLID accumulation per _PostProcessMode. Modes 1/2 stay transcribed-but-
    // unpinned (the gate configs use mode 0), mirroring the offline driver.
    let mut seg_lid = vec![0.0f64; out_cols];
    match p.post_process_mode {
        1 => {
            for kk in 0..out_f32.rows {
                if get(kk, 1) >= 0.0 {
                    let mut entropy = 0.0;
                    let mut log_out = vec![0.0f64; out_cols];
                    for (ll, lo) in log_out.iter_mut().enumerate() {
                        *lo = f64::max(1e-24, get(kk, ll)).ln();
                        entropy -= get(kk, ll) * *lo;
                    }
                    entropy /= 2.0f64.ln();
                    if entropy < 1e-24 {
                        entropy = 1e-24;
                    }
                    for (ll, sl) in seg_lid.iter_mut().enumerate() {
                        *sl += f64::max(1e-24, get(kk, ll) / entropy).ln();
                    }
                    acc.number_of_frames += 1;
                }
            }
        }
        2 => {
            for kk in 0..out_f32.rows {
                if get(kk, 1) >= 0.0 {
                    let mut j = 0usize;
                    let mut best = get(kk, 0);
                    for c in 1..out_cols {
                        if get(kk, c) > best {
                            best = get(kk, c);
                            j = c;
                        }
                    }
                    seg_lid[j] += 1.0;
                    acc.number_of_frames += 1;
                }
            }
        }
        _ => {
            for kk in 0..out_f32.rows {
                if get(kk, 1) >= 0.0 {
                    for (ll, sl) in seg_lid.iter_mut().enumerate() {
                        *sl += f64::max(1e-24, get(kk, ll)).ln();
                    }
                    acc.number_of_frames += 1;
                }
            }
        }
    }

    // :1537-1544 argmax j, :1545-1555 confusion.
    let j = argmax_f64(&seg_lid);
    let pos_target = p.ti + 1;
    if j == p.ti {
        acc.confusion[[pos_target, pos_target]] += 1.0;
        acc.confusion[[pos_target, p.class_nb + 1]] += 1.0;
        acc.confusion[[p.class_nb + 1, pos_target]] += 1.0;
    } else {
        let pos_best = j + 1;
        acc.confusion[[pos_target, pos_best]] += 1.0;
        acc.confusion[[pos_target, p.class_nb + 1]] += 1.0;
        acc.confusion[[p.class_nb + 1, pos_best]] += 1.0;
    }

    // :1558-1567 langID.
    let rows = out_f32.rows as f64;
    if p.cost_modified {
        for (c, &v) in seg_lid.iter().enumerate() {
            acc.langid[c] += v / rows;
        }
    } else {
        for (c, &v) in seg_lid.iter().enumerate() {
            acc.langid[c] += v;
        }
    }

    acc.segments_count += 1;
    Some(LidEntryOutcome { seg_lid, argmax: j })
}

/// The finalized per-channel members (`:1599-1634` epilogue applied to `acc`): normalize a
/// CLONE of langID (so a mid-stream snapshot never consumes the accumulator), then derive
/// the in-band classification errors + is_lid_correct + predicted language. Read-only in
/// `acc`, so the offline (once) and the streaming session (per push) both call it.
pub(super) fn finalize_lid_channel(acc: &LidChannelAcc, p: &LidScoreParams) -> LidAggregate {
    // :1599-1620 langID normalization (on a clone -- the offline mutated in place then read
    // it; the values are identical, and the offline never reads langID after the epilogue).
    let mut langid = acc.langid.clone();
    if acc.segments_count > 0 && langid.iter().sum::<f64>() != 0.0 {
        if p.cost_modified {
            for v in langid.iter_mut() {
                *v /= acc.segments_count as f64;
            }
        } else {
            for v in langid.iter_mut() {
                *v /= acc.number_of_frames as f64;
            }
        }
        if p.post_process_mode != 2 {
            for v in langid.iter_mut() {
                *v = v.exp();
            }
        }
        let adim: f64 = langid.iter().sum();
        for v in langid.iter_mut() {
            *v /= adim;
        }
    } else {
        langid[0] = 1.0;
    }

    // :1622-1634 member writes.
    let mut target_lid = vec![0.0f64; p.class_nb];
    target_lid[p.ti] = -2.0;
    let mut errors = vec![0.0f64; p.class_nb];
    for c in 0..p.class_nb {
        errors[c] = 100.0 * (langid[c] - target_lid[c]);
    }
    let max_langid = langid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let is_lid_correct = if langid[p.ti] == max_langid { 100 } else { 0 };
    let predicted_language = argmax_f64(&langid);

    LidAggregate {
        classification_errors: errors,
        confusion: acc.confusion.clone(),
        is_lid_correct,
        predicted_language,
        segments_count: acc.segments_count,
    }
}

impl FastTwinLid {
    /// Derive the per-channel LID scoring params (the window resolution + skip/scoring
    /// constants, minus the SAD side) for BOTH the offline `get_segmentation` and the
    /// streaming session; `who` prefixes a refusal. Mode 7 forces
    /// `_SpectrumShiftInFrames = 80` (`:1296-1301`: the phSeq/cep periodogram is always
    /// present), so `ssif` is a constant here. Applies the regime rule
    /// ([`check_lid_regime`]: plain or truncate per the shape, overlap refused).
    /// `ti = clamp(lang_index, [0, class_nb))`.
    pub(super) fn lid_score_params(
        &self,
        who: &str,
        rate: f64,
        lang_index: i32,
    ) -> Result<LidScoreParams> {
        let ssif = 80usize; // mode-7 forced (:1296-1301)
        let (lid_window_size, lid_no_overlap) = derive_lid_window(
            self.lid_window_size_sec,
            self.lid_window_shift_sec,
            self.lid_ssr,
            ssif,
            rate,
        );
        check_lid_regime(who, self.lid_shape, lid_window_size, lid_no_overlap)?;
        let mut ti = lang_index;
        if ti >= self.class_nb as i32 {
            ti = 0;
        }
        if ti < 0 {
            ti = 0;
        }
        Ok(LidScoreParams {
            lid_ssr: self.lid_ssr,
            lid_lstm_sub: self.lid_spec.lstm_subsampling.clone(),
            lid_out_sub: self.lid_spec.output_subsampling.clone(),
            min_nb_of_frames: self.min_nb_of_frames,
            noise_magnitude: self.noise_magnitude,
            lid_window_size,
            two_sweeps: self.lid_two_sweeps,
            post_process_mode: self.post_process_mode,
            class_nb: self.class_nb,
            ti: ti as usize,
            cost_modified: self.cost_modified,
        })
    }

    /// Whether the LID net is loaded -- so the streaming session bails at construction rather
    /// than deferring the failure to the first push.
    pub(super) fn lid_net_loaded(&self) -> bool {
        self.lid_net.is_some()
    }

    /// Fold one external-features entry through the LID net into `acc` (the streaming seam --
    /// delegates to the shared [`score_lid_entry`] with the internal net). Panics only if the
    /// net is unloaded, which the session forbids at construction ([`lid_net_loaded`](Self::
    /// lid_net_loaded)).
    pub(super) fn fold_entry(
        &mut self,
        feat: &Array2<f64>,
        p: &LidScoreParams,
        acc: &mut LidChannelAcc,
    ) -> Option<LidEntryOutcome> {
        let net = self
            .lid_net
            .as_mut()
            .expect("LID net loaded (checked at session construction)");
        score_lid_entry(feat, net, p, acc)
    }
}
