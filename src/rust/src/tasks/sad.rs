//! SAD segmenters: BLSTM over signal/spectral/LTSV features -> speech posterior.
//!
//! Ported from legacy C++: BLSTMSignalSegmenter.*, BLSTMSpectralSegmenter.*,
//! BLSTMSpectralLID.*. Phase 2-4.
//!
//! [`TdcSegmenter`] (Algo 1) is ported from `TimeDomainCorrel.{h,cpp}` (Phase 2b
//! Task 4): the ctor (`:16-27`) and `getSegmentation` (`:93-259`). No NN in the
//! chain -- `classifySequence` (ported as [`crate::features::ltsv_tdc::tdc_classify_sequence`])
//! is cwise autocorrelation + a table log, so the harness oracle calls the REAL
//! compiled `getSegmentation` directly (no transcription), confirmed by the Step 1
//! CHECK-FIRST: the golden `tdc_result_chan{1,2}.bin` dumps (an INDEPENDENT
//! replication of the framing loop using the same real `classifySequence`) match
//! the Phase 1 `tdc_classify_sequence` port bit-for-bit under the oracle-gated
//! comparator, so this port reuses that Phase 1 function verbatim.

use anyhow::{Result, anyhow};
use indexmap::IndexMap;

use crate::audio::{Audio, get_sequence, windowing_coefficients};
use crate::features::ltsv_tdc::tdc_classify_sequence;
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::{ScoreReport, compute_errors};
use crate::tasks::segmenter::{DriverConfig, Segmenter, SegmenterConfig, results_to_segmentation};

fn parse_scalar(m: &IndexMap<String, String>, key: &str) -> Result<f64> {
    m.get(key)
        .ok_or_else(|| anyhow!("missing config key `{key}`"))?
        .trim()
        .parse::<f64>()
        .map_err(|e| anyhow!("`{key}`: cannot parse: {e}"))
}

fn parse_list(m: &IndexMap<String, String>, key: &str) -> Result<Vec<f64>> {
    let raw = m
        .get(key)
        .ok_or_else(|| anyhow!("missing config key `{key}`"))?;
    raw.split(',')
        .map(|tok| {
            tok.trim()
                .parse::<f64>()
                .map_err(|e| anyhow!("`{key}`: cannot parse `{tok}`: {e}"))
        })
        .collect()
}

fn parse_bool(m: &IndexMap<String, String>, key: &str) -> Result<bool> {
    // legacy `conf.get<bool>` uses std::boolalpha: the strings "true"/"false".
    m.get(key)
        .ok_or_else(|| anyhow!("missing config key `{key}`"))?
        .trim()
        .parse::<bool>()
        .map_err(|e| anyhow!("`{key}`: cannot parse as bool: {e}"))
}

fn parse_i32(m: &IndexMap<String, String>, key: &str) -> Result<i32> {
    m.get(key)
        .ok_or_else(|| anyhow!("missing config key `{key}`"))?
        .trim()
        .parse::<i32>()
        .map_err(|e| anyhow!("`{key}`: cannot parse: {e}"))
}

fn parse_string(m: &IndexMap<String, String>, key: &str) -> Result<String> {
    m.get(key)
        .cloned()
        .ok_or_else(|| anyhow!("missing config key `{key}`"))
}

/// Time-domain correlation SAD segmenter (Algo 1; `TimeDomainCorrel.{h,cpp}`).
///
/// `window_shift_sec` is the stateful legacy `_WindowShift` member: re-quantized
/// on EVERY `get_segmentation` call using ITS CURRENT VALUE (floor at `1/rate`,
/// then `round(x*rate)/rate`, `TimeDomainCorrel.cpp:103-105`) -- a two-files-in-
/// sequence run therefore sees the SECOND call read back the FIRST call's
/// quantized value (idempotent once on a `1/rate` grid point, but not assumed:
/// the two-files golden pins this explicitly).
pub struct TdcSegmenter {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    lags: Vec<f64>,
    balance: f64,
    windowing_type: String,
    windowing_param: f64,
    flag_dc_offset: bool,
    preemph_ratio: f64,
    noise_seed: i32,
    noise_ratio: f64,
    window_shift_sec: f64,
    channels: usize,
}

impl TdcSegmenter {
    /// Port of the `TimeDomainCorrel(ConfigFile&, bool, bool)` ctor
    /// (`TimeDomainCorrel.cpp:16-27`): `buildFromConf(conf, "TDC", ...)` (the
    /// `Segmenter` residue, via [`SegmenterConfig::from_config`] +
    /// [`DriverConfig::from_config`]), then `TDC_lags` (>= 2 values required, ALL
    /// entries clamped `< 0.0 -> 0.0`) and `TDC_balance` (unclamped).
    pub fn from_legacy(map: &IndexMap<String, String>) -> Result<TdcSegmenter> {
        let seg_cfg = SegmenterConfig::from_config(map, "TDC")?;
        let driver_cfg = DriverConfig::from_config(map, "TDC")?;

        let mut lags = parse_list(map, "TDC_lags")?;
        if lags.len() < 2 {
            return Err(anyhow!("TDC_lags must contain 2 values"));
        }
        for v in lags.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let balance = parse_scalar(map, "TDC_balance")?;

        let flag_dc_offset = parse_bool(map, "TDC_flag_DCOffset")?;
        let preemph_ratio = parse_scalar(map, "TDC_preemph_ratio")?;
        let noise_seed = parse_i32(map, "TDC_noise_seed")?;
        let noise_ratio = parse_scalar(map, "TDC_noise_ratio")?;
        let windowing_type = parse_string(map, "TDC_windowing_type")?;
        let windowing_param = parse_scalar(map, "TDC_windowing_param")?;

        let window_shift_sec = driver_cfg.window_shift_sec;

        Ok(TdcSegmenter {
            driver_cfg,
            seg_cfg,
            lags,
            balance,
            windowing_type,
            windowing_param,
            flag_dc_offset,
            preemph_ratio,
            noise_seed,
            noise_ratio,
            window_shift_sec,
            channels: 0,
        })
    }

    /// Zero cost accumulators: TDC has no NN/cost path (`cumulative_error`/
    /// `nb_of_classif` stay at their legacy `Segmentation` ctor-seeded 0.0/0
    /// defaults for every channel -- `getSegmentation` never writes them). Sized
    /// by the channel count of the last `get_segmentation` call.
    pub fn cumulative_error(&self) -> Vec<f64> {
        vec![0.0; self.channels]
    }

    pub fn nb_of_classif(&self) -> Vec<i64> {
        vec![0; self.channels]
    }

    /// `Segmentation::compute_errors`, one call per channel (the Rust container
    /// is single-channel; the legacy loops channels internally -- see
    /// `tasks/segmentation.rs`'s module doc for the S6 restructure rationale).
    /// `hyp` is the caller's post-`get_segmentation` `seg_per_chan` slice.
    pub fn score(
        hyp: &mut [Segmentation],
        reference: Option<&[Segmentation]>,
        nb_words: i64,
    ) -> Vec<ScoreReport> {
        hyp.iter_mut()
            .enumerate()
            .map(|(chan, seg)| {
                let refc = reference.map(|r| &r[chan]);
                compute_errors(seg, refc, nb_words)
            })
            .collect()
    }
}

impl Segmenter for TdcSegmenter {
    /// Port of `TimeDomainCorrel::getSegmentation` (`TimeDomainCorrel.cpp:93-259`),
    /// the non-unit-test/non-plotting path (dump/`.mat`/PNG branches dropped, they
    /// are display/diagnostic-only and never touch `seg`).
    ///
    /// Per channel: `results_to_segmentation(seg, window_shift_sec /*post-
    /// quantization*/, 0.0, ...)` then `seg.compute_errors()` after the full
    /// channel loop (`:220,:238`).
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
    ) -> Result<()> {
        let rate = audio.sample_rate as f64;

        // window_size (`:100-101`): half = round(w*rate/2), full = 2*half+1 (ODD).
        let half_window = f64::round(self.driver_cfg.window_size_sec * rate / 2.0) as usize;
        let full_window = 2 * half_window + 1;

        // _WindowShift re-quantization using ITS CURRENT VALUE (`:103-105`): floor
        // at 1/rate, then round(x*rate)/rate. Mutates the stateful member.
        if self.window_shift_sec < 1.0 / rate {
            self.window_shift_sec = 1.0 / rate;
        }
        let window_shift = f64::round(self.window_shift_sec * rate) as usize;
        self.window_shift_sec = window_shift as f64 / rate;

        // min_lag/max_lag (`:107-109`): rounded to frames, max_lag clamped below
        // full_window. No min_lag/max_lag ordering guard in the legacy -- none
        // added here.
        let min_lag = f64::round(self.lags[0] * rate) as usize;
        let mut max_lag = f64::round(self.lags[1] * rate) as usize;
        if max_lag >= full_window {
            max_lag = full_window - 1;
        }

        // preemph -> noise (`:134-145`), UNNORMALIZED windowing coeffs (`:146`,
        // `getWindowingCoefficients(_WindowingType, false, full_window_size,
        // _WindowingParam)` -- note `normalized=false`, unlike DriverConfig's
        // convolution kernel which is always normalized).
        audio.apply_preemph(self.preemph_ratio);
        if self.noise_seed > 0 {
            audio.apply_noise(self.noise_ratio);
        }
        let windowing_coeff = windowing_coefficients(
            &self.windowing_type,
            false,
            full_window,
            self.windowing_param,
        );

        let frame_count = audio.data.ncols();
        let vec_size = if frame_count.is_multiple_of(window_shift) {
            frame_count / window_shift
        } else {
            frame_count / window_shift + 1
        };

        let channels = audio.data.nrows();
        self.channels = channels;
        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            let mut buf = ndarray::Array2::<f64>::zeros((1, full_window));
            let mut results = vec![0.0f64; vec_size];
            let mut jj = 0usize;
            while jj < frame_count {
                get_sequence(
                    &audio.data,
                    chan,
                    jj,
                    half_window,
                    self.flag_dc_offset,
                    windowing_coeff.as_deref(),
                    &mut buf,
                );
                let window = buf.row(0).to_vec();
                results[jj / window_shift] =
                    tdc_classify_sequence(&window, min_lag, max_lag, self.balance);
                jj += window_shift;
            }

            results_to_segmentation(
                seg,
                self.window_shift_sec,
                0.0,
                &mut results,
                SegClass::Speech,
                self.driver_cfg.conv_coeff.as_deref(),
                &self.seg_cfg,
            );
        }

        // `seg.compute_errors()` after the FULL channel loop (`:238`); the legacy
        // no-reference branch (Branch A) is a pure `sanitize` + `update_count`
        // side effect on each hypothesis -- discard the (unused, no-reference)
        // ScoreReport here. Callers wanting scored output call `TdcSegmenter::score`
        // explicitly afterward.
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }

        Ok(())
    }
}
