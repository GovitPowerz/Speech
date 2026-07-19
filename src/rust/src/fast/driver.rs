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
//! SCOPE (spec S1.2/S1.3, the house typed-bail pattern -- what the gate configs
//! exercise, everything else typed-bails loudly so scope creep is loud):
//! - The PITCH second pass (`BLSTM_TDCwindow > 0`, spec R4) typed-bails at
//!   construction.
//! - `InputNormalizationType != -1` typed-bails at construction (the gate configs --
//!   `tier2_spectral.config` + the phase-6 `lre_sad.toml` -- both use -1).
//! - The PLAIN (window 0) and TRUNCATE (non-overlap) windowed variants typed-bail at
//!   `get_segmentation` (both gate configs resolve to OVERLAP: `BLSTM_window 3.25 /
//!   BLSTM_shift 0.8` -> `window_size > 0`, `no_overlap false`).
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
use crate::nn::blstm::BlstmConfig;
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::compute_errors;
use crate::tasks::segmenter::{DriverConfig, Segmenter, SegmenterConfig, results_to_segmentation};

use super::nn::{FastBlstm, FastMatrix, self_normalize_f32};
use super::pipeline::FastPipeline;
use crate::features::pipeline::{FeatureConfig, SpectralParams};

/// Build the `NnetSpec` for the fast net (under config `prefix`, e.g. `"BLSTM"` for
/// the SAD net or `"BLSTM_LID"` for the Twin's LID net) with peephole flags aligned to
/// the EXACT path's defaults (Phase 7 Task 4 rider 1 -- the peephole default asymmetry).
///
/// `NnetSpec::from_legacy` (`config.rs:35`) defaults an ABSENT peephole key to FALSE,
/// but the exact `BlstmConfig`/`LSTMLayer` path (`blstm.rs:112`) defaults it TRUE. The
/// fast net reads a `NnetSpec`; the exact net reads a `BlstmConfig`; so a key-omitting
/// config would silently give the fast net a DIFFERENT peephole configuration than the
/// exact path -- a divergence with no tolerance floor. We resolve it by overriding the
/// spec's peepholes from `BlstmConfig` (default TRUE), so both paths read peepholes
/// from the SAME source and defaults. Every committed SAD/LID config sets all six keys
/// explicitly (so the override is value-preserving there), but the alignment is pinned
/// against an omitting config in `tests/phase7_parity_sad.rs`.
pub fn build_aligned_spec(map: &IndexMap<String, String>, prefix: &str) -> Result<NnetSpec> {
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
        let lstm0 = map
            .get(&format!("{prefix}_LSTMNeuronNb"))
            .ok_or_else(|| anyhow!("missing {prefix}_LSTMNeuronNb"))?
            .split(',')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let mut m = map.clone();
        m.insert(input_key, lstm0);
        NnetSpec::from_legacy(&m, prefix)?
    };
    let bc = BlstmConfig::from_legacy(map, prefix)?;
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
    net: Option<FastBlstm>,

    /// Sub-sampling factors + whole-BLSTM ratio, cached from the spec for
    /// [`get_blstm_param`] (which needs them without a live net borrow).
    lstm_sub_sampling: Vec<usize>,
    output_sub_sampling: Vec<usize>,
    ssr: usize,

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
    /// `weights: Some(flat)` builds the [`FastBlstm`] immediately (narrowing f64 ->
    /// f32 once, after adim); `None` defers to [`Self::load_weights_file`] (the bag's
    /// two-step `from_legacy(map, None)` + `load_weights_file` pattern).
    ///
    /// Typed-bails (loudly, at construction) the unsupported fast-mode surfaces: the
    /// pitch second pass (`TDCwindow > 0`) and any `InputNormalizationType != -1`.
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

        // Peephole-default alignment (rider 1).
        let spec = build_aligned_spec(map, "BLSTM")?;

        // Only self-normalization (type -1) is on the SAD gate path.
        let bc = BlstmConfig::from_legacy(map, "BLSTM")?;
        if bc.input_normalization_type != -1 {
            bail!(
                "fast SAD: only InputNormalizationType -1 (self-normalization) is supported on the \
                 fast path (got {}); the gate configs use -1",
                bc.input_normalization_type
            );
        }

        let lstm_sub_sampling = spec.lstm_subsampling.clone();
        let output_sub_sampling = spec.output_subsampling.clone();
        let ssr = lstm_sub_sampling.iter().product::<usize>()
            * output_sub_sampling.iter().product::<usize>();

        let net = match weights {
            Some(flat) => Some(FastBlstm::from_flat(&spec, flat)?),
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
            net,
            lstm_sub_sampling,
            output_sub_sampling,
            ssr,
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
    /// `get_segmentation` errors); otherwise read the `.bin` and (re)build the
    /// [`FastBlstm`] from it. `FastBlstm::from_flat` enforces the length check
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
        self.net = Some(FastBlstm::from_flat(&self.spec, &flat)?);
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

        // Dispatch (setProcessingType(window > 0, !noOverlap), tasks/sad.rs:1447): the
        // fast SAD path implements ONLY the overlap windowed driver -- what both gate
        // configs (tier2 + phase-6 SAD, window 3.25 / shift 0.8) exercise. Plain and
        // truncate typed-bail as unexercised in fast mode (spec S1.2 house pattern).
        if window_size == 0 {
            bail!(
                "fast SAD: the plain (non-windowed) forward is unsupported (BLSTM_window resolves \
                 window_size 0); the gate configs use windowed overlap"
            );
        }
        if no_overlap {
            bail!(
                "fast SAD: the truncate (non-overlap) windowing is unsupported (window_shift \
                 resolves < 1); the gate configs use overlap"
            );
        }

        // timeStep/timeOffset, OVERLAP branch (tasks/sad.rs:1459-1460): _SpectrumShift-
        // based (the asymmetry vs the signal driver's _WindowShift). self.window_shift_sec
        // here is the post-getBLSTMParam value.
        let time_step = self.spectrum_shift_sec * ssr as f64;
        let time_offset = time_step / 2.0 - self.spectrum_shift_sec / 2.0;

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

            // Type -1 self-normalization over the whole sequence, ONCE (matching the
            // exact feed_forward_backward top, blstm.rs:1027, before the overlap
            // windowing).
            self_normalize_f32(&mut input);

            // Overlap forward: accumulate into the shared result_buf (seeded from the
            // prior channel; NOT zeroed here).
            net.feed_forward_overlap(&input, window_size, window_shift, &mut result_buf);

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
// FastTwinLid -- the f32 Mode-7 LID Twin (algo 6).
// ===========================================================================

/// Read one required `f64` config scalar (mirrors `tasks/lid.rs::twin_scalar`).
fn twin_scalar(map: &IndexMap<String, String>, key: &str) -> Result<f64> {
    map.get(key)
        .ok_or_else(|| anyhow!("missing config key `{key}`"))?
        .trim()
        .parse::<f64>()
        .map_err(|e| anyhow!("`{key}`: cannot parse: {e}"))
}

/// Read one `f64` config scalar with a default (mirrors `twin_scalar_default`).
fn twin_scalar_default(map: &IndexMap<String, String>, key: &str, default: f64) -> Result<f64> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow!("`{key}`: cannot parse: {e}")),
    }
}

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
/// - The LID windowed dispatch must resolve to TRUNCATE (`lid_window_size > 0`,
///   `lid_no_overlap true`, what the gate configs give); plain/overlap bail at
///   `get_segmentation`.
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
    lid_net: Option<FastBlstm>,
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
    /// 7). `lid_weights: Some(flat)` builds the LID [`FastBlstm`] immediately; `None`
    /// defers to [`Self::load_weights_file`].
    ///
    /// Typed-bails (loudly, at construction) the unsupported fast surfaces: any mode but
    /// 7, the pitch second pass (`TDCwindow > 0`), a LID `InputNormalizationType != 0`,
    /// and `DumpLIDInternals` (the fast path does not track the LID hidden states the
    /// dump emits).
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

        // SAD net shape only (never run in Mode 7). Peephole-aligned spec, but no net.
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
        // (nn/blstm.rs:1032-1043) instead of the truncate BLSTM path the fast LID scoring
        // always runs. `FastBlstm::from_flat` (fast/nn.rs:340-342) independently bails on
        // this shape, but only once weights are actually loaded -- when `lid_weights` is
        // `None` here (the deferred `load_weights_file` path), `from_legacy` would
        // otherwise return `Ok` for an is_mlp config, so `from_flat`'s bail does NOT
        // provably fire first on every such config. Bail on the SAME derived flag the
        // exact dispatch gates on so construction fails loudly regardless of when weights
        // arrive.
        if lid_bc.is_mlp {
            bail!(
                "fast TwinLid: MLP mode (BLSTM_LID_LSTMNeuronNb[0] == 0) is not supported on \
                 the fast path (the fast LID scoring always runs the truncate BLSTM forward, \
                 never the MLP drivers); the gate configs use a non-zero LSTM width"
            );
        }

        let cost_modified = lid_bc.cost_law.is_cost_modified();
        let lid_two_sweeps = lid_bc.two_sweeps;

        let lid_spec = build_aligned_spec(map, "BLSTM_LID")?;
        let lid_ssr = lid_spec.lstm_subsampling.iter().product::<usize>()
            * lid_spec.output_subsampling.iter().product::<usize>();
        // class_nb = LID output_size max 2 (the langID / confusion dimensionality).
        let class_nb = (*lid_spec.output_neuron_nb.last().unwrap()).max(2);

        let lid_net = match lid_weights {
            Some(flat) => Some(FastBlstm::from_flat(&lid_spec, flat)?),
            None => None,
        };

        let mut lid_window_size_sec = twin_scalar(map, "BLSTM_LID_window")?;
        if lid_window_size_sec < 0.0 {
            lid_window_size_sec = 0.0;
        }
        let mut lid_window_shift_sec = twin_scalar(map, "BLSTM_LID_shift")?;
        if lid_window_shift_sec < 0.0 {
            lid_window_shift_sec = 0.0;
        }
        let post_process_mode = twin_i32_default(map, "BLSTM_LID_PostProcessMode", 0)?;
        let min_nb_of_frames = twin_i32_default(map, "BLSTM_LID_MinNbOfFrames", 0)?;
        let noise_magnitude = twin_scalar_default(map, "BLSTM_LID_NoiseMagnitude", 0.0)?;

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
    /// the `.bin` and (re)build the LID [`FastBlstm`].
    pub fn load_weights_file(&mut self, map: &IndexMap<String, String>) -> Result<()> {
        let weights_file = map
            .get("BLSTM_LID_weightsFile")
            .map(String::as_str)
            .unwrap_or("");
        if weights_file.is_empty() {
            return Ok(());
        }
        let flat = crate::io::binary::read_weight_vector(std::path::Path::new(weights_file))?;
        self.lid_net = Some(FastBlstm::from_flat(&self.lid_spec, &flat)?);
        Ok(())
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
    /// `NnetSpec`-default FALSE).
    #[cfg(feature = "test-support")]
    pub fn debug_lid_peepholes(&self) -> Option<[bool; 6]> {
        self.lid_net.as_ref().map(|n| n.debug_peepholes())
    }
}

impl Segmenter for FastTwinLid {
    /// Port of the `abs(_Mode) == 7` external-features LID loop
    /// (`tasks/lid.rs::get_segmentation_mode7`), f32. The SAD net is synthesized (constant
    /// 10.0 -> the SHARED f64 decision layer, so the SAD side is bit-identical to the
    /// exact path); the LID net runs the fast windowed forward per external-features block.
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

        // getLIDBLSTMParam: LID window/shift derivation (the exact `:1330-1353`).
        let lid_ssr = self.lid_ssr;
        let ssifd = ssif as f64;
        let mut lid_window_size =
            f64::round(self.lid_window_size_sec * rate / 2.0 / ssifd) as usize;
        if lid_window_size != 0 && lid_window_size < lid_ssr {
            lid_window_size = lid_ssr;
        }
        let mut lid_window_shift = f64::round(self.lid_window_shift_sec * rate / ssifd) as i64;
        let mut lid_no_overlap = false;
        if lid_window_size != 0 && lid_window_shift < 1 {
            lid_no_overlap = true;
            let raw = f64::round(self.lid_window_size_sec * rate / ssifd) as usize;
            lid_window_size = (raw / lid_ssr) * lid_ssr;
            if lid_window_size < 10 * lid_ssr {
                lid_window_size = 10 * lid_ssr;
            }
        }
        if lid_window_size == 0 || lid_window_shift < 1 {
            lid_window_shift = 1;
        }
        self.lid_window_shift_sec = (lid_window_shift * ssif as i64) as f64 / rate;

        // The fast LID scoring implements ONLY the TRUNCATE windowed forward (what the
        // gate configs resolve to: `set_processing_type(lid_window_size > 0,
        // !lid_no_overlap)` -> truncate iff `lid_window_size > 0 && lid_no_overlap`).
        // Plain / overlap bail (the house pattern, mirroring the SAD driver's).
        if lid_window_size == 0 {
            bail!(
                "fast TwinLid mode 7: the plain (non-windowed) LID forward is unsupported; the \
                 gate configs resolve to windowed truncate"
            );
        }
        if !lid_no_overlap {
            bail!(
                "fast TwinLid mode 7: the overlap LID windowing is unsupported; the gate configs \
                 resolve to truncate"
            );
        }

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

        let class_nb = self.class_nb;
        let cost_modified = self.cost_modified;
        let two_sweeps = self.lid_two_sweeps;
        let post_process_mode = self.post_process_mode;
        let min_nb_of_frames = self.min_nb_of_frames;
        let noise_magnitude = self.noise_magnitude;
        let conv_coeff = self.driver_cfg.conv_coeff.clone();
        let external_features = audio.external_features.clone();
        let lang_index = audio.lang_index;

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
            // targetIndex = clamp(lang_index, [0, classNb)) (the exact `:1396-1404`).
            let mut target_index = lang_index;
            if target_index >= class_nb as i32 {
                target_index = 0;
            }
            if target_index < 0 {
                target_index = 0;
            }
            let ti = target_index as usize;

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

            // --- abs(_Mode)==7 external-features loop (`:1448-1591`) ------------------
            let mut langid = vec![0.0f64; class_nb];
            let mut confusion = Array2::<f64>::zeros((class_nb + 2, class_nb + 2));
            for kk in 0..class_nb + 1 {
                confusion[[0, kk]] = kk as f64;
                confusion[[kk, 0]] = kk as f64;
            }
            let mut segments_count = 0i32;
            let mut number_of_frames = 0i32;

            for feat in &external_features {
                // _MinNbOfFrames + ssr guard (`:1450`).
                if feat.nrows() < lid_ssr || (feat.nrows() as i32) < min_nb_of_frames {
                    continue;
                }
                // Gaussian noise (`:1456-1466`), applied in f64 on the block (randinit
                // fixed 0, matching the exact port); the pure table lookup is portable.
                let mut feat_noised = feat.clone();
                if noise_magnitude > 0.0 {
                    let cols = feat.ncols();
                    let randinit = 0usize;
                    for r in 0..feat.nrows() {
                        for c in 0..cols {
                            let m = random_gauss(r * cols + c + randinit) - 0.5;
                            feat_noised[[r, c]] += noise_magnitude * m;
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
                let out_f32 = net.feed_forward_scoring(&fin, lid_window_size, two_sweeps);
                let out_cols = out_f32.cols;
                // Widen the posteriors back to f64 (the member seam).
                let output_seq: Vec<f64> = out_f32.data.iter().map(|&x| x as f64).collect();
                let get = |r: usize, c: usize| output_seq[r * out_cols + c];

                // segLID accumulation per _PostProcessMode (`:1486-1534`). The legacy
                // `short_result_vec > 0.5` gate is always true, so only `outputSeq(kk,1)
                // >= 0` gates (always true for the logistic posterior).
                let mut seg_lid = vec![0.0f64; out_cols];
                match post_process_mode {
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
                                number_of_frames += 1;
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
                                number_of_frames += 1;
                            }
                        }
                    }
                    _ => {
                        for kk in 0..out_f32.rows {
                            if get(kk, 1) >= 0.0 {
                                for (ll, sl) in seg_lid.iter_mut().enumerate() {
                                    *sl += f64::max(1e-24, get(kk, ll)).ln();
                                }
                                number_of_frames += 1;
                            }
                        }
                    }
                }

                // argmax j (`:1537-1544`), confusion (`:1545-1555`).
                let mut j = 0usize;
                let mut best = seg_lid[0];
                for (c, &v) in seg_lid.iter().enumerate().skip(1) {
                    if v > best {
                        best = v;
                        j = c;
                    }
                }
                let pos_target = ti + 1;
                if j == ti {
                    confusion[[pos_target, pos_target]] += 1.0;
                    confusion[[pos_target, class_nb + 1]] += 1.0;
                    confusion[[class_nb + 1, pos_target]] += 1.0;
                } else {
                    let pos_best = j + 1;
                    confusion[[pos_target, pos_best]] += 1.0;
                    confusion[[pos_target, class_nb + 1]] += 1.0;
                    confusion[[class_nb + 1, pos_best]] += 1.0;
                }

                // langID (`:1558-1567`).
                let rows = out_f32.rows as f64;
                if cost_modified {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        langid[c] += v / rows;
                    }
                } else {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        langid[c] += v;
                    }
                }

                segments_count += 1;
            }

            // langID normalization (`:1599-1620`).
            if segments_count > 0 && langid.iter().sum::<f64>() != 0.0 {
                if cost_modified {
                    for v in langid.iter_mut() {
                        *v /= segments_count as f64;
                    }
                } else {
                    for v in langid.iter_mut() {
                        *v /= number_of_frames as f64;
                    }
                }
                if post_process_mode != 2 {
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

            // Member writes (`:1622-1634`). Forward-only: lid_cumulative_error /
            // lid_nb_of_classif stay 0 (documented divergence, the fast path never
            // harvests the LID NN cost/classif count).
            let mut target_lid = vec![0.0f64; class_nb];
            target_lid[ti] = -2.0;
            self.lid_cumulative_error[chan] = 0.0;
            self.lid_nb_of_classif[chan] = 0;
            let mut lid_errors = vec![0.0f64; class_nb];
            for c in 0..class_nb {
                lid_errors[c] = 100.0 * (langid[c] - target_lid[c]);
            }
            self.lid_classification_errors[chan] = lid_errors;
            self.lid_segments_confusion[chan] = confusion;
            let max_langid = langid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.is_lid_correct[chan] = if langid[ti] == max_langid { 100 } else { 0 };
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
