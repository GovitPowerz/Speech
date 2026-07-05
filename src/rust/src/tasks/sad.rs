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
use ndarray::Array2;

use crate::audio::{
    Audio, compute_segment_periodogram_estimates, get_sequence, windowing_coefficients,
};
use crate::features::ltsv_tdc::{ltsv_classify_sequence, tdc_classify_sequence};
use crate::features::mel::MelFilterBank;
use crate::features::pipeline::{FeatureConfig, derive_freq_band_ltsv_variant};
use crate::nn::blstm::{BlstmConfig, BlstmNetwork};
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

/// Long-Term Spectral Variation SAD segmenter (Algo 2;
/// `LongTermSpectralVariation.{h,cpp}`).
///
/// Two stateful legacy members are carried across calls, each re-quantized on
/// EVERY `get_segmentation` call using ITS CURRENT VALUE:
/// - `spectrum_shift_sec` (`_SpectrumShift`, `:154-155`): `round(x*rate)/rate`.
/// - `window_shift_sec` (`_WindowShift`, `:261-263`): `round(x*rate/spectrum_shift)
///   *spectrum_shift/rate` -- quantized in PERIODOGRAM-FRAME units (a spectrum_shift
///   multiple), unlike TDC's plain `1/rate` grid.
///
/// The LTSV-standalone window floor is `< 1 -> 1` (`:257-258`), which DIFFERS from
/// the BLSTM spectral segmenter's `< 1 -> 0` (disables LTSV) -- ported as written,
/// see IMPROVEMENTS.md.
pub struct LtsvSegmenter {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    feature_cfg: FeatureConfig,
    spectrum_shift_sec: f64,
    window_shift_sec: f64,
    channels: usize,
}

impl LtsvSegmenter {
    /// Port of the `LongTermSpectralVariation(ConfigFile&, bool, bool)` ctor
    /// (`LongTermSpectralVariation.cpp:18-20` -> `buildFromConf`, itself
    /// `Segmenter::buildFromConf` + the LTSV-specific reads): the `Segmenter`
    /// residue (via [`SegmenterConfig::from_config`] + [`DriverConfig::from_config`])
    /// plus the spectral/mel/freq-band config (via [`FeatureConfig::from_legacy`],
    /// whose reads + sanitization order match `LongTermSpectralVariation::
    /// buildFromConf` `:44-80` exactly under this generic `prefix`).
    pub fn from_legacy(map: &IndexMap<String, String>) -> Result<LtsvSegmenter> {
        let seg_cfg = SegmenterConfig::from_config(map, "LTSV")?;
        let driver_cfg = DriverConfig::from_config(map, "LTSV")?;
        let feature_cfg = FeatureConfig::from_legacy(map, "LTSV")?;

        let spectrum_shift_sec = feature_cfg.shift_sec;
        let window_shift_sec = driver_cfg.window_shift_sec;

        Ok(LtsvSegmenter {
            driver_cfg,
            seg_cfg,
            feature_cfg,
            spectrum_shift_sec,
            window_shift_sec,
            channels: 0,
        })
    }

    /// Zero cost accumulators: LTSV has no NN/cost path (same rationale as
    /// [`TdcSegmenter::cumulative_error`]).
    pub fn cumulative_error(&self) -> Vec<f64> {
        vec![0.0; self.channels]
    }

    pub fn nb_of_classif(&self) -> Vec<i64> {
        vec![0; self.channels]
    }

    /// `Segmentation::compute_errors`, one call per channel (see
    /// [`TdcSegmenter::score`]'s doc for the single-channel-container rationale).
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

impl Segmenter for LtsvSegmenter {
    /// Port of `LongTermSpectralVariation::getSegmentation`
    /// (`LongTermSpectralVariation.cpp:130-407`), the non-unit-test/non-plotting
    /// path (dump/`.mat`/PNG/log branches dropped, display/diagnostic-only).
    ///
    /// Per channel: `compute_segment_periodogram_estimates` over `[0, frame_count]`
    /// (the `+1` `vec_size` quirk -- `end_frame = audio.getFrameCount()`, ONE MORE
    /// than the raw last-valid-index, `:293-294`), then the DECIMATED result row
    /// `result_vec(0, jj/shift) = ltsv_classify_sequence(...)` (no interpolation
    /// backfill -- that is [`crate::features::ltsv_tdc::get_ltsv`], the spectral
    /// segmenter's, not this driver's); `results_to_segmentation(seg,
    /// window_shift_sec /*post-quantization*/, 0.0, ...)`; `compute_errors` after
    /// the full channel loop (`:386`).
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
    ) -> Result<()> {
        let rate = audio.sample_rate as f64;

        // Spectrum-order clamp to Max-1 == 19 (`:145-149`).
        let mut order = self.feature_cfg.order;
        if order > 19 {
            order = 19;
        }
        let window_size = 1usize << order;
        let full_signal_window_size = window_size + 1;
        let mut periodogram_length = (1usize << (order as u32 - 1)) + 1;

        // spectrum_shift re-quantization using ITS CURRENT VALUE (`:154-155`).
        let spectrum_shift = (self.spectrum_shift_sec * rate).round() as usize;
        self.spectrum_shift_sec = spectrum_shift as f64 / rate;

        // preemph -> noise (`:180-191`).
        audio.apply_preemph(self.feature_cfg.preemph_ratio);
        if self.feature_cfg.noise_seed > 0 {
            audio.apply_noise(self.feature_cfg.noise_ratio);
        }
        let windowing_coeff = windowing_coefficients(
            &self.feature_cfg.win_type,
            false,
            full_signal_window_size,
            self.feature_cfg.win_param,
        );

        // Freq band: the LTSV.cpp:200-209 clamp-order VARIANT (differs from the
        // BLSTM variant in `SpectralParams::derive`).
        let (mut freq_beg, mut freq_end, min_freq, max_freq) =
            derive_freq_band_ltsv_variant(&self.feature_cfg, rate, periodogram_length);

        // Mel/DCT branch (`:210-246`): the `_NbDCT` clamp-to-nb_filters + the band
        // reset to the mel/DCT output size.
        let mel_bank = if self.feature_cfg.nb_bins > 0 {
            let bank = MelFilterBank::new(
                self.feature_cfg.min_mel,
                self.feature_cfg.max_mel,
                self.feature_cfg.nb_bins,
                min_freq,
                max_freq,
                rate,
                periodogram_length - 1,
                self.feature_cfg.is_log,
                self.feature_cfg.nb_dct,
                self.feature_cfg.ignore_first,
                self.feature_cfg.deltas_nb,
                self.feature_cfg.dd_nb,
            );
            periodogram_length = bank.nb_filters();
            if self.feature_cfg.nb_dct > 0 {
                periodogram_length = bank.nb_dct();
            }
            freq_beg = 0;
            freq_end = periodogram_length - 1;
            Some(bank)
        } else {
            None
        };

        // Temporal convolution of the periodogram (`:248-249`; `verifyWindowingType`
        // is a logging-only side effect, dropped here per its established contract).
        let temporal_conv = windowing_coefficients(
            &self.feature_cfg.conv_type,
            true,
            2 * self.feature_cfg.conv_size as usize + 1,
            0.83333,
        );

        // LTSV window/shift in PERIODOGRAM-FRAME units (`:257-263`). The
        // LTSV-standalone floor is `< 1 -> 1` (NOT the spectral segmenter's
        // `< 1 -> 0` disable).
        let mut ltsv_half_window =
            (self.driver_cfg.window_size_sec * rate / 2.0 / spectrum_shift as f64).round() as i64;
        if ltsv_half_window < 1 {
            ltsv_half_window = 1;
        }
        let ltsv_half_window = ltsv_half_window as usize;
        let mut ltsv_window_shift =
            (self.window_shift_sec * rate / spectrum_shift as f64).round() as i64;
        if ltsv_window_shift < 1 {
            ltsv_window_shift = 1;
        }
        self.window_shift_sec = (ltsv_window_shift * spectrum_shift as i64) as f64 / rate;
        let ltsv_window_shift = ltsv_window_shift as usize;

        // vec_size: begin_frame=0, end_frame=audio.getFrameCount() (the +1 quirk
        // vs `build_input_sequence`'s end=ncols()-1), `:293-300`.
        let frame_count = audio.data.ncols();
        let vec_size_raw = frame_count + 1;
        let vec_size = if vec_size_raw.is_multiple_of(spectrum_shift) {
            vec_size_raw / spectrum_shift
        } else {
            vec_size_raw / spectrum_shift + 1
        };
        let real_vec_size = if vec_size.is_multiple_of(ltsv_window_shift) {
            vec_size / ltsv_window_shift
        } else {
            vec_size / ltsv_window_shift + 1
        };

        let channels = audio.data.nrows();
        self.channels = channels;
        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            let perio: Array2<f64> = compute_segment_periodogram_estimates(
                audio,
                order as u32,
                spectrum_shift,
                chan,
                self.feature_cfg.flag_dc_offset,
                windowing_coeff.as_deref(),
                temporal_conv.as_deref(),
                0,
                frame_count,
            );

            let source: Array2<f64> = match &mel_bank {
                Some(bank) => {
                    let fb = bank.apply_filter_bank(&perio);
                    if self.feature_cfg.nb_dct > 0 {
                        bank.apply_dct(&fb)
                    } else {
                        fb
                    }
                }
                None => perio,
            };

            let mut results = vec![0.0f64; real_vec_size];
            let mut jj = 0usize;
            while jj < vec_size {
                results[jj / ltsv_window_shift] =
                    ltsv_classify_sequence(&source, jj, freq_beg, freq_end, ltsv_half_window);
                jj += ltsv_window_shift;
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

        // `seg.compute_errors()` after the FULL channel loop (`:386`).
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }

        Ok(())
    }
}

/// BLSTM signal-domain SAD segmenter (Algo 4; `BLSTMSignalSegmenter.{h,cpp}`).
///
/// The FIRST driver with the NN in the chain: unlike TDC/LTSV, the per-channel
/// result vector is produced by the ported [`BlstmNetwork::feed_forward_backward`],
/// NOT a feature-extraction + classify loop. The "pipeline" is literally
/// `audio.data.row(chan)` transposed to an Nx1 column (`:188`) -- no periodogram, no
/// mel, no LTSV/TDC. Input normalization (type -1 self-normalization for the real
/// net) lives inside the NN.
///
/// `window_shift_sec` is the stateful legacy `_WindowShift`: re-quantized on EVERY
/// `get_segmentation` call (`round(x*rate)/rate`, `:99-108`), then RESET to `0.0`
/// when the noOverlap branch fired (`:376`, BEFORE `compute_errors`). Unlike TDC/
/// LTSV, this reset makes the noOverlap lifecycle genuinely re-entrant: file 2 sees
/// `_WindowShift == 0.0`, so `:99` rounds `0.0*rate -> 0`, re-triggers noOverlap,
/// and `:107-108` re-clamps to `1/rate`. The two-files golden pins the real round
/// trip (see `phase2b_signal_golden.rs`).
///
/// `two_sweeps` is the required-but-DEAD `BLSTM_TwoSweeps` flag (`:20,:28`): the only
/// consumers are commented out (`:282,:297`), so it is parsed (a missing key aborts
/// the legacy `conf.get<bool>`, so the parse itself is load-bearing) and stored, but
/// never read by the live path.
///
/// Divergences vs the spectral segmenter (`p2b_signal.md` 12-divergence list),
/// reproduced here: window/shift in SIGNAL samples (no `/spectrum_shift`); NO
/// `full_window = 0` when `window == 0` (stays 1, feeds a length-1 windowing coeff
/// that Rust's [`windowing_coefficients`] returns `None` for -- a no-op, matching the
/// legacy empty-coeff dump-only dead weight); `_FlagDCOffset` log-only (never applied
/// -- the input is raw `audio.data`); result-vec sizing GATES the ssr-division on
/// `(window == 0 || noOverlap)` (spectral's is unconditional); the overlap-branch
/// `timeStep = window_shift_sec, timeOffset = 0.0` asymmetry; fresh `result_vec` per
/// channel.
pub struct BlstmSignalSegmenter {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    net: BlstmNetwork,
    /// `_BLSTMTwoSweeps` (`:20,:28`): required-but-dead. Parsed so a missing key is an
    /// error (parity with `conf.get<bool>`); never consumed by the live path.
    two_sweeps: bool,
    windowing_type: String,
    windowing_param: f64,
    flag_dc_offset: bool,
    preemph_ratio: f64,
    noise_seed: i32,
    noise_ratio: f64,
    /// The stateful legacy `_WindowShift` member (`:108/:376`).
    window_shift_sec: f64,
    channels: usize,
    cumulative_error: Vec<f64>,
    nb_of_classif: Vec<i64>,
}

impl BlstmSignalSegmenter {
    /// Port of the `BLSTMSignalSegmenter(ConfigFile&, ...)` ctors
    /// (`BLSTMSignalSegmenter.cpp:16-29`): `buildFromConf(conf, "BLSTM", ...)` (the
    /// `Segmenter` residue, via [`SegmenterConfig::from_config`] +
    /// [`DriverConfig::from_config`]), `BLSTMNeuralNetwork(conf, "BLSTM", ...)`
    /// (via [`BlstmConfig::from_legacy`] + [`BlstmNetwork::from_config`]), then
    /// `BLSTM_TwoSweeps` (required bool). `weights` mirrors the two ctor overloads:
    /// `Some(flat)` -> `setWeights(weights)` (the weights-ctor, `:23-27`); `None` ->
    /// the no-weights ctor (`:16-21`, weights loaded later or left zero).
    pub fn from_legacy(
        map: &IndexMap<String, String>,
        weights: Option<&[f64]>,
    ) -> Result<BlstmSignalSegmenter> {
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;

        let blstm_cfg = BlstmConfig::from_legacy(map, "BLSTM")?;
        let mut net = BlstmNetwork::from_config(blstm_cfg)?;
        if let Some(flat) = weights {
            net.set_weights(flat)?;
        }

        let two_sweeps = parse_bool(map, "BLSTM_TwoSweeps")?;

        let flag_dc_offset = parse_bool(map, "BLSTM_flag_DCOffset")?;
        let preemph_ratio = parse_scalar(map, "BLSTM_preemph_ratio")?;
        let noise_seed = parse_i32(map, "BLSTM_noise_seed")?;
        let noise_ratio = parse_scalar(map, "BLSTM_noise_ratio")?;
        let windowing_type = parse_string(map, "BLSTM_windowing_type")?;
        let windowing_param = parse_scalar(map, "BLSTM_windowing_param")?;

        let window_shift_sec = driver_cfg.window_shift_sec;

        Ok(BlstmSignalSegmenter {
            driver_cfg,
            seg_cfg,
            net,
            two_sweeps,
            windowing_type,
            windowing_param,
            flag_dc_offset,
            preemph_ratio,
            noise_seed,
            noise_ratio,
            window_shift_sec,
            channels: 0,
            cumulative_error: Vec::new(),
            nb_of_classif: Vec::new(),
        })
    }

    /// `_BLSTMTwoSweeps` accessor (dead flag; exposed so a doc test can assert the
    /// field parsed without being consumed by the live path).
    pub fn two_sweeps(&self) -> bool {
        self.two_sweeps
    }

    /// Per-channel `seg._CumulativeError[chan] = NNCost` (`:346`): the NN cost from
    /// the last `get_segmentation`, one entry per channel.
    pub fn cumulative_error(&self) -> &[f64] {
        &self.cumulative_error
    }

    /// Per-channel `seg._NbOfClassif[chan] = nbOfClassif` (`:347`).
    pub fn nb_of_classif(&self) -> &[i64] {
        &self.nb_of_classif
    }

    /// `Segmentation::compute_errors`, one call per channel (see
    /// [`TdcSegmenter::score`]'s doc for the single-channel-container rationale).
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

impl Segmenter for BlstmSignalSegmenter {
    /// Port of `BLSTMSignalSegmenter::getSegmentation`
    /// (`BLSTMSignalSegmenter.cpp:93-398`), the non-unit-test/non-plotting path (the
    /// `.mat`/PNG/log branches dropped: display/diagnostic-only). The `_BLSTMTwoSweeps`
    /// block (`:282-305`) is DEAD (commented out in legacy) -- unported.
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
    ) -> Result<()> {
        let rate = audio.sample_rate as f64;
        let ssr = self.net.sub_sampling_ratio();

        // Window/shift in SIGNAL samples (`:96-108`). half = round(w*rate/2); the
        // `!= 0 && < ssr -> ssr` floor; full = 2*window+1 (ODD). NO `full = 0` when
        // window == 0 (stays 1, `p2b_signal.md` divergence 2).
        let mut window_size = f64::round(self.driver_cfg.window_size_sec * rate / 2.0) as usize;
        if window_size != 0 && window_size < ssr {
            window_size = ssr;
        }
        let mut full_window_size = 2 * window_size + 1;

        let mut window_shift = f64::round(self.window_shift_sec * rate) as i64;
        let mut no_overlap = false;
        if window_size != 0 && window_shift < 1 {
            no_overlap = true;
            // Integer floor-div then re-multiply (`:103`): (round(w*rate)/ssr)*ssr.
            window_size = (f64::round(self.driver_cfg.window_size_sec * rate) as usize / ssr) * ssr;
            if window_size < 10 * ssr {
                window_size = 10 * ssr;
            }
            full_window_size = window_size; // EVEN, not 2k+1 (`:105`).
        }
        if window_size == 0 || window_shift < 1 {
            window_shift = 1;
        }
        // MEMBER MUTATION (`:108`): _WindowShift = shift/rate.
        self.window_shift_sec = window_shift as f64 / rate;

        // `_FlagDCOffset` is LOG-ONLY in signal mode (`:116`, divergence 4): never
        // applied. Threaded nowhere below; kept as a field only for parity/logging.
        let _ = self.flag_dc_offset;

        // preemph -> noise (`:137-148`). windowing_coeff computed (`:149`) but never
        // applied (dump-only dead weight, divergence 3); Rust's `windowing_coefficients`
        // returns `None` for size <= 1 (the full == 1 window == 0 case) -- a no-op,
        // matching the legacy empty-coeff behavior. Computed here only for parity of
        // the (dropped) unit-test dump; discarded.
        if self.preemph_ratio > 0.0 {
            audio.apply_preemph(self.preemph_ratio);
        }
        if self.noise_seed > 0 {
            audio.apply_noise(self.noise_ratio);
        }
        let _windowing_coeff = windowing_coefficients(
            &self.windowing_type,
            false,
            full_window_size,
            self.windowing_param,
        );

        // resetWeightsDerivatives ONCE per file, before the channel loop (`:170`).
        self.net.reset_weights_derivatives();

        let window_shift_usize = window_shift as usize;
        let channels = audio.data.nrows();
        let frame_count = audio.data.ncols();
        self.channels = channels;
        self.cumulative_error = vec![0.0; channels];
        self.nb_of_classif = vec![0; channels];

        // setProcessingType((window > 0), !noOverlap) ONCE (`:259`): the flags do not
        // change across channels (window_size/no_overlap are per-file constants).
        self.net.set_processing_type(window_size > 0, !no_overlap);

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // inputSeq = audio.data.row(chan).transpose() -> Nx1 column (`:188`).
            let row = audio.data.row(chan);
            let mut input_seq = Array2::<f64>::zeros((frame_count, 1));
            for (i, &v) in row.iter().enumerate() {
                input_seq[[i, 0]] = v;
            }

            // Result-vec sizing (`:221-240`), FRESH per channel (divergence 6):
            // unconditional ceil-division by window_shift, then ssr-division GATED on
            // (window == 0 || noOverlap) (divergence 5).
            let mut real_vec_size = if frame_count.is_multiple_of(window_shift_usize) {
                frame_count / window_shift_usize
            } else {
                frame_count / window_shift_usize + 1
            };
            if (window_size == 0 || no_overlap) && ssr > 1 {
                for r in self.net.lstm_sub_sampling() {
                    real_vec_size /= r;
                }
                for r in self.net.output_sub_sampling() {
                    real_vec_size /= r;
                }
            }
            let mut result_vec = Array2::<f64>::zeros((real_vec_size, 1));

            // timeStep/timeOffset (`:244-254`): `window_shift_sec` here is the POST-:108
            // mutated value. Default/noOverlap: `shift*ssr` & `step/2 - shift/2`;
            // overlap: `shift` & 0.0 (divergence 7).
            let mut time_step = self.window_shift_sec * ssr as f64;
            let mut time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
            if window_size > 0 {
                if no_overlap {
                    time_step = self.window_shift_sec * ssr as f64;
                    time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
                } else {
                    time_step = self.window_shift_sec;
                    time_offset = 0.0;
                }
            }

            // getTargets when a reference exists (`:255-258`). No reference is set on a
            // fresh hypothesis Segmentation, so the target sequence stays empty; the
            // driver runs the no-target forward path. (Reference-driven scoring is
            // exercised via `score`, not this method, matching the goldens.)
            let target = Array2::<f64>::zeros((0, 0));

            // NN forward+backward (`:260`): input mutated in place by the internal
            // normalization (type -1 for the real net). window_size/window_shift are
            // the SIGNAL-sample values.
            self.net.feed_forward_backward(
                &mut input_seq,
                window_size,
                window_shift_usize,
                &mut result_vec,
                &target,
            );
            self.cumulative_error[chan] = self.net.cost;
            self.nb_of_classif[chan] = self.net.nb_of_classif;

            // result_vec2 = result_vec.transpose() -> ROW vector (`:315`).
            let mut result_vec2: Vec<f64> = result_vec.column(0).to_vec();

            // results2segmentation (`:344-345`): the `targetSeqTmp = result_vec` copy
            // quirk is a dead param in the live `results_to_segmentation` (see its doc).
            results_to_segmentation(
                seg,
                time_step,
                time_offset,
                &mut result_vec2,
                SegClass::Speech,
                self.driver_cfg.conv_coeff.as_deref(),
                &self.seg_cfg,
            );
        }

        // noOverlap `_WindowShift = 0.0` reset BEFORE compute_errors (`:376`,
        // divergence 11): re-arms the noOverlap trigger for the NEXT file. This is the
        // genuine cross-file state (unlike TDC/LTSV's idempotent re-quantization).
        if no_overlap {
            self.window_shift_sec = 0.0;
        }

        // `seg.compute_errors()` after the FULL channel loop (`:378`).
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }

        Ok(())
    }

    /// `setWeights` delegates to the NN (`BLSTMSignalSegmenter.cpp:38-40`).
    fn set_weights(&mut self, flat: &[f64]) -> Result<()> {
        self.net.set_weights(flat)
    }

    /// `getWeights` delegates to the NN (`:42-44`).
    fn get_weights(&self) -> Vec<f64> {
        self.net.get_weights()
    }
}
