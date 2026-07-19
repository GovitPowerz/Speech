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

use crate::audio::Audio;
use crate::config::NnetSpec;
use crate::nn::blstm::BlstmConfig;
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::compute_errors;
use crate::tasks::segmenter::{DriverConfig, Segmenter, SegmenterConfig, results_to_segmentation};

use super::nn::{FastBlstm, FastMatrix, self_normalize_f32};
use super::pipeline::FastPipeline;
use crate::features::pipeline::{FeatureConfig, SpectralParams};

/// Build the `NnetSpec` for the fast net with peephole flags aligned to the EXACT
/// path's defaults (Phase 7 Task 4 rider 1 -- the peephole default asymmetry).
///
/// `NnetSpec::from_legacy` (`config.rs:35`) defaults an ABSENT peephole key to FALSE,
/// but the exact `BlstmConfig`/`LSTMLayer` path (`blstm.rs:112`) defaults it TRUE. The
/// fast net reads a `NnetSpec`; the exact net reads a `BlstmConfig`; so a key-omitting
/// config would silently give the fast net a DIFFERENT peephole configuration than the
/// exact path -- a divergence with no tolerance floor. We resolve it by overriding the
/// spec's peepholes from `BlstmConfig` (default TRUE), so both paths read peepholes
/// from the SAME source and defaults. Every committed SAD config sets all six keys
/// explicitly (so the override is value-preserving there), but the alignment is pinned
/// against an omitting config in `tests/phase7_parity_sad.rs`.
pub fn build_aligned_spec(map: &IndexMap<String, String>) -> Result<NnetSpec> {
    let mut spec = NnetSpec::from_legacy(map, "BLSTM")?;
    let bc = BlstmConfig::from_legacy(map, "BLSTM")?;
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
        let spec = build_aligned_spec(map)?;

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
        // N-1's post-division contents.
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
