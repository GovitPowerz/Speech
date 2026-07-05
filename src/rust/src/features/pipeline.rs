//! Feature-pipeline config + parameter derivation + input-sequence assembly.
//!
//! Ported from the legacy C++ feature front-end:
//! - [`FeatureConfig::from_legacy`]: the config key reads + sanitization from
//!   `LongTermSpectralVariation::buildFromConf` (`LongTermSpectralVariation.cpp:
//!   44-80`), `Segmenter::buildFromConf` (`Segmenter.cpp:94-102`), and
//!   `BLSTMSpectralSegmenter::buildFromConf` (`BLSTMSpectralSegmenter.cpp:44-83`).
//!   NOTE: `BLSTMSpectralSegmenter::buildFromConf` (`:45`) hardcodes the literal
//!   `"BLSTM"` prefix for its `LongTermSpectralVariation::buildFromConf` call
//!   regardless of the `specialization_preposition` parameter it was itself passed
//!   -- every legacy call site (`BLSTMSpectralSegmenter`, `BLSTMSpectralLID`,
//!   `TwinBLSTMSpectralLID`) happens to pass `"BLSTM"` too, so this never bites
//!   today, but a future LID port that calls `from_legacy` with a different
//!   `prefix` for the spectral subset would silently read the wrong keys (this
//!   port's `from_legacy(map, prefix)` takes `prefix` at face value and applies it
//!   uniformly -- it does NOT reproduce the hardcoding).
//! - [`SpectralParams::derive`]: the param derivation from
//!   `BLSTMSpectralSegmenter::initSpectralAnalysis` (`:194-314`) + `getLTSVParam`
//!   (`:300-314`) + `getTDCParam` (`:341-370`).
//! - [`assemble_input_sequence`]: `getBLSTMInputSequence` (`:561-591`).
//!
//! [`TdcParams`] is the Task 9 definition, kept VERBATIM (callers construct it
//! directly; [`SpectralParams::derive`] fills it from a [`TdcConfig`]).
//!
//! Parity hazards (all load-bearing):
//! - Sanitization ORDER: negative `minFreq`/`maxFreq` -> 0 BEFORE the swap; then
//!   swap-if-reversed; then span-widen (`|max-min| < 2` -> mean +- 1). Same for the
//!   mel pair (no negative-zeroing there, matching the legacy). Reproduce exactly.
//! - `_LTSVWindowShift != 0.0` UB guard (`BLSTMSpectralSegmenter.cpp:50`) is DROPPED:
//!   `_LTSVWindowShift` is an uninitialized double member at that read, so the legacy
//!   branch is UB. We read `LTSVshift` unconditionally when the key is present -- the
//!   oracle harness does the same, so the golden stays valid.
//! - Spectrum-order clamp to `Max-1 == 19` (`:199-203`, factory `Max=20`). VERIFIED.
//! - `freqStep = rate/2 / (bins-1)` (NOT `/bins`); the BLSTM freq-band clamp order at
//!   `:229-239` (`floor` for beg, `ceil` for end, each re-clamped against the other,
//!   then snap = `idx*freqStep`). The `LongTermSpectralVariation.cpp` LTSV variant
//!   differs -- this ports the BLSTM one.
//! - `freq_beg`/`freq_end` here are the PRE-mel spectral band (the raw-path band).
//!   For mel variants the legacy RESETS them to `(0, output_dim-1)` at `:270-271`,
//!   which is the LTSV band -- that reset lives in the caller (the E2E gate), NOT in
//!   `derive` (the struct carries the spectral band the raw path uses).
//! - `assemble_input_sequence` priority: DCT -> filterbank -> `ln(block + 1e-24)`
//!   (band-limit ONLY on the raw path); then hcat the LTSV column as the LAST column.

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;
use ndarray::Array2;

use crate::audio::{Audio, compute_segment_periodogram_estimates, windowing_coefficients};
use crate::features::ltsv_tdc::get_ltsv;
use crate::features::mel::MelFilterBank;

/// Time-domain-correlation window/lag/balance parameters.
///
/// `half_window` is the legacy `TDC_window_size` (round(window_sec*rate/2)), the
/// half-window handed to `get_sequence`; `full_window = 2*half_window+1` is the
/// windowed-frame width; `shift` is the frame stride in samples; `min_lag`/`max_lag`
/// are the autocorrelation lag bounds in samples; `coeffs` is the optional windowing
/// vector (length `full_window`); `balance` weighs MaxPeak vs CrossCorr in the score.
#[derive(Debug, Clone)]
pub struct TdcParams {
    pub half_window: usize,
    pub full_window: usize,
    pub shift: usize,
    pub min_lag: usize,
    pub max_lag: usize,
    pub coeffs: Option<Vec<f64>>,
    pub balance: f64,
}

/// The TDC config sub-block, read only when `TDCwindow != 0`
/// (`BLSTMSpectralSegmenter.cpp:59-74`). `lags` has its negatives clamped to 0 and
/// must carry >= 2 entries (else `from_legacy` errors).
#[derive(Debug, Clone)]
pub struct TdcConfig {
    pub shift: f64,
    pub lags: Vec<f64>,
    pub balance: f64,
    pub windowing_type: String,
    pub windowing_param: f64,
}

/// Parsed + sanitized feature-front-end config (the subset the spectral/LTSV/TDC
/// derivation consumes). Field names mirror the legacy members.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    pub order: i64,
    pub shift_sec: f64,
    pub conv_size: i64,
    pub conv_type: String,
    pub min_mel: f64,
    pub max_mel: f64,
    pub nb_bins: i32,
    pub is_log: bool,
    pub nb_dct: i32,
    pub ignore_first: bool,
    pub deltas_nb: i32,
    pub dd_nb: i32,
    pub min_freq: f64,
    pub max_freq: f64,
    pub win_type: String,
    pub win_param: f64,
    pub flag_dc_offset: bool,
    pub preemph_ratio: f64,
    pub noise_seed: i32,
    pub noise_ratio: f64,
    pub ltsv_window: f64,
    pub ltsv_shift: f64,
    pub tdc_window: f64,
    pub tdc: Option<TdcConfig>,
}

/// `conf.get<T>(name)` (no default): missing key -> the legacy `check()` aborts;
/// here that maps to an `Err` (README parity note: missing required key = process
/// exit in legacy = `Err` in Rust).
fn get_str<'a>(map: &'a IndexMap<String, String>, key: &str) -> Result<&'a str> {
    map.get(key)
        .map(String::as_str)
        .with_context(|| format!("param '{key}' not found in config"))
}

fn get_f64(map: &IndexMap<String, String>, key: &str) -> Result<f64> {
    let s = get_str(map, key)?;
    s.parse::<f64>()
        .with_context(|| format!("cannot read '{s}' as f64 for '{key}'"))
}

fn get_i64(map: &IndexMap<String, String>, key: &str) -> Result<i64> {
    let s = get_str(map, key)?;
    s.parse::<i64>()
        .with_context(|| format!("cannot read '{s}' as i64 for '{key}'"))
}

fn get_i32(map: &IndexMap<String, String>, key: &str) -> Result<i32> {
    let s = get_str(map, key)?;
    s.parse::<i32>()
        .with_context(|| format!("cannot read '{s}' as i32 for '{key}'"))
}

fn get_bool(map: &IndexMap<String, String>, key: &str) -> Result<bool> {
    // legacy `read<bool>` uses std::boolalpha: the strings "true"/"false".
    let s = get_str(map, key)?;
    s.parse::<bool>()
        .with_context(|| format!("cannot read '{s}' as bool for '{key}'"))
}

/// `conf.get<double>(name, default)`: missing key -> default.
fn get_f64_default(map: &IndexMap<String, String>, key: &str, default: f64) -> Result<f64> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .parse::<f64>()
            .with_context(|| format!("cannot read '{s}' as f64 for '{key}'")),
    }
}

impl FeatureConfig {
    /// Read + sanitize the feature config under `<prefix>_*` keys.
    pub fn from_legacy(map: &IndexMap<String, String>, prefix: &str) -> Result<FeatureConfig> {
        let k = |suffix: &str| format!("{prefix}{suffix}");

        // LongTermSpectralVariation::buildFromConf reads (no defaults).
        let order = get_i64(map, &k("_spectrum_order"))?;
        let shift_sec = get_f64(map, &k("_spectrum_shift"))?;
        let conv_size = get_i64(map, &k("_spectrum_temporal_convolution_size"))?;
        let conv_type = get_str(map, &k("_spectrum_temporal_convolution_type"))?.to_string();
        let mut min_mel = get_f64(map, &k("_minMelFreq"))?;
        let mut max_mel = get_f64(map, &k("_maxMelFreq"))?;
        let nb_bins = get_i32(map, &k("_nb_bins"))?;
        let is_log = get_bool(map, &k("_is_log_mel"))?;
        let nb_dct = get_i32(map, &k("_nb_DCT"))?;
        let ignore_first = get_bool(map, &k("_IgnoreFirstDCT"))?;
        let deltas_nb = get_i32(map, &k("_ComputeDeltasNb"))?;
        let dd_nb = get_i32(map, &k("_ComputeDeltaDeltasNb"))?;

        // minFreq/maxFreq: negatives -> 0 BEFORE the swap (:56-59).
        let mut min_freq = get_f64(map, &k("_minFreq"))?;
        if min_freq < 0.0 {
            min_freq = 0.0;
        }
        let mut max_freq = get_f64(map, &k("_maxFreq"))?;
        if max_freq < 0.0 {
            max_freq = 0.0;
        }
        // Mel pair swap + span-widen (:60-69).
        if max_mel < min_mel {
            std::mem::swap(&mut min_mel, &mut max_mel);
        }
        if (max_mel - min_mel).abs() < 2.0 {
            let mean = (min_mel + max_mel) / 2.0;
            max_mel = mean + 1.0;
            min_mel = mean - 1.0;
        }
        // Hz pair swap + span-widen (:70-79).
        if max_freq < min_freq {
            std::mem::swap(&mut min_freq, &mut max_freq);
        }
        if (max_freq - min_freq).abs() < 2.0 {
            let mean = (min_freq + max_freq) / 2.0;
            max_freq = mean + 1.0;
            min_freq = mean - 1.0;
        }

        // Segmenter::buildFromConf reads used downstream (Segmenter.cpp:94-102).
        let win_type = get_str(map, &k("_windowing_type"))?.to_string();
        let win_param = get_f64(map, &k("_windowing_param"))?;
        let flag_dc_offset = get_bool(map, &k("_flag_DCOffset"))?;
        let preemph_ratio = get_f64(map, &k("_preemph_ratio"))?;
        let noise_seed = get_i32(map, &k("_noise_seed"))?;
        let noise_ratio = get_f64(map, &k("_noise_ratio"))?;

        // BLSTMSpectralSegmenter LTSV reads (:48-55). LTSVwindow defaults to 0;
        // negatives -> 0. LTSVshift read unconditionally (UB guard dropped, see
        // module docs); negatives -> 0.
        let mut ltsv_window = get_f64_default(map, &k("_LTSVwindow"), 0.0)?;
        if ltsv_window < 0.0 {
            ltsv_window = 0.0;
        }
        let mut ltsv_shift = get_f64(map, &k("_LTSVshift"))?;
        if ltsv_shift < 0.0 {
            ltsv_shift = 0.0;
        }

        // BLSTMSpectralSegmenter TDC reads (:57-81). TDCwindow defaults to 0;
        // negatives -> 0. The TDC sub-block is read only when TDCwindow != 0.
        let mut tdc_window = get_f64_default(map, &k("_TDCwindow"), 0.0)?;
        if tdc_window < 0.0 {
            tdc_window = 0.0;
        }
        let tdc = if tdc_window != 0.0 {
            let mut shift = get_f64(map, &k("_TDCshift"))?;
            if shift < 0.0 {
                shift = 0.0;
            }
            let lags_str = get_str(map, &k("_TDC_lags"))?;
            let mut lags: Vec<f64> = Vec::new();
            for part in lags_str.split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                lags.push(
                    part.parse::<f64>()
                        .with_context(|| format!("cannot read '{part}' as f64 in TDC_lags"))?,
                );
            }
            if lags.len() < 2 {
                bail!("{}_TDC_lags must contain 2 values", prefix);
            }
            for v in lags.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let balance = get_f64(map, &k("_TDC_balance"))?;
            let windowing_type = get_str(map, &k("_TDC_windowing_type"))?.to_string();
            let windowing_param = get_f64(map, &k("_TDC_windowing_param"))?;
            Some(TdcConfig {
                shift,
                lags,
                balance,
                windowing_type,
                windowing_param,
            })
        } else {
            None
        };

        Ok(FeatureConfig {
            order,
            shift_sec,
            conv_size,
            conv_type,
            min_mel,
            max_mel,
            nb_bins,
            is_log,
            nb_dct,
            ignore_first,
            deltas_nb,
            dd_nb,
            min_freq,
            max_freq,
            win_type,
            win_param,
            flag_dc_offset,
            preemph_ratio,
            noise_seed,
            noise_ratio,
            ltsv_window,
            ltsv_shift,
            tdc_window,
            tdc,
        })
    }
}

/// The spectrum-order clamp (`BLSTMSpectralSegmenter.cpp:199-203`, factory `Max=20`):
/// order is clamped to `Max-1 == 19`. VERIFIED in the phase1 manifest.
const SPECTRUM_ORDER_MAX: i64 = 20;

/// Derived spectral / LTSV / TDC parameters.
///
/// `window_size = 1<<order`; `buffer_size = window_size+1`; `bins = 2^(order-1)+1`
/// (the `periodogram_length`); `freq_beg`/`freq_end` are the PRE-mel spectral band
/// (`min_freq`/`max_freq` are the snapped `idx*freqStep`); `ltsv_half_window`/
/// `ltsv_shift` are the LTSV `R`/`shift` (`R < 1` disables LTSV, `shift < 1 -> 1`);
/// `tdc` is `None` when `TDCwindow == 0`.
#[derive(Debug, Clone)]
pub struct SpectralParams {
    pub order: u32,
    pub window_size: usize,
    pub buffer_size: usize,
    pub bins: usize,
    pub shift_frames: usize,
    pub shift_sec: f64,
    pub freq_beg: usize,
    pub freq_end: usize,
    pub min_freq: f64,
    pub max_freq: f64,
    pub ltsv_half_window: usize,
    pub ltsv_shift: usize,
    pub tdc: Option<TdcParams>,
}

impl SpectralParams {
    /// Derive the spectral/LTSV/TDC params for the given sample `rate`
    /// (`BLSTMSpectralSegmenter::initSpectralAnalysis` :199-311 + `getLTSVParam`
    /// :302-305 + `getTDCParam` :344-369).
    pub fn derive(cfg: &FeatureConfig, rate: f64) -> SpectralParams {
        // Order clamp to Max-1 == 19 (:199-203).
        let mut order = cfg.order;
        if order > SPECTRUM_ORDER_MAX - 1 {
            order = SPECTRUM_ORDER_MAX - 1;
        }
        let window_size = 1usize << order;
        let buffer_size = window_size + 1;
        let periodogram_length = (1usize << (order - 1)) + 1;
        let bins = periodogram_length;

        // Shift quantization: round to frames, then re-quantize (:208-210). The
        // legacy `hasReadWavFile() -> false` override to 80 is skipped (the excerpt
        // is a real wav; `derive` is the wav-read path).
        let shift_frames = (cfg.shift_sec * rate).round() as usize;
        let shift_sec = shift_frames as f64 / rate;

        // BLSTM freq band (:229-239). freqStep divides by (bins-1) == periodogram
        // length - 1. `size_type` (usize) arithmetic mirrors the legacy casts.
        let mut freq_beg: usize = 0;
        let mut freq_end: usize = periodogram_length - 1;
        let freq_step = rate / 2.0 / freq_end as f64;
        let tmp = (cfg.min_freq / freq_step).floor() as usize;
        if freq_beg < tmp {
            freq_beg = tmp;
        }
        if freq_beg > freq_end {
            freq_beg = freq_end;
        }
        let tmp = (cfg.max_freq / freq_step).ceil() as usize;
        if freq_end > tmp {
            freq_end = tmp;
        }
        if freq_end < freq_beg {
            freq_end = freq_beg;
        }
        let min_freq = freq_beg as f64 * freq_step;
        let max_freq = freq_end as f64 * freq_step;

        // LTSV params (:302-305). R < 1 -> 0 (disables); shift < 1 -> 1.
        let r = (cfg.ltsv_window * rate / 2.0 / shift_frames as f64).round();
        let ltsv_half_window = if r < 1.0 { 0 } else { r as usize };
        let ls = (cfg.ltsv_shift * rate / shift_frames as f64).round();
        let ltsv_shift = if ls < 1.0 { 1 } else { ls as usize };

        // TDC params (getTDCParam :344-369 + TimeDomainCorrel.cpp:100-109).
        let tdc = cfg.tdc.as_ref().map(|tc| {
            let half_window = (cfg.tdc_window * rate / 2.0).round();
            let half_window = if half_window < 1.0 {
                0
            } else {
                half_window as usize
            };
            let full_window = 2 * half_window + 1;
            // TDCshift: floored at 1/rate, quantized to spectrum-shift multiples.
            let mut ws = tc.shift;
            if ws < 1.0 / rate {
                ws = 1.0 / rate;
            }
            let mut window_shift = (ws * rate / shift_frames as f64).round() as i64;
            if window_shift < 1 {
                window_shift = 1;
            }
            let shift = (window_shift * shift_frames as i64) as usize;
            // lags: rounded to samples, min_lag floored at 1, max_lag capped at
            // full_window-1 then floored at min_lag.
            let mut min_lag = (tc.lags[0] * rate).round() as i64;
            if min_lag < 1 {
                min_lag = 1;
            }
            let mut max_lag = (tc.lags[1] * rate).round() as i64;
            if max_lag >= full_window as i64 {
                max_lag = full_window as i64 - 1;
            }
            if max_lag < min_lag {
                max_lag = min_lag;
            }
            let coeffs = crate::audio::windowing_coefficients(
                &tc.windowing_type,
                false,
                full_window,
                tc.windowing_param,
            );
            TdcParams {
                half_window,
                full_window,
                shift,
                min_lag: min_lag as usize,
                max_lag: max_lag as usize,
                coeffs,
                balance: tc.balance,
            }
        });

        SpectralParams {
            order: order as u32,
            window_size,
            buffer_size,
            bins,
            shift_frames,
            shift_sec,
            freq_beg,
            freq_end,
            min_freq,
            max_freq,
            ltsv_half_window,
            ltsv_shift,
            tdc,
        }
    }
}

/// The STANDALONE `LongTermSpectralVariation::getSegmentation` freq-band clamp
/// order (`LongTermSpectralVariation.cpp:200-209`), a DIFFERENT variant from
/// [`SpectralParams::derive`]'s BLSTM freq band (`BLSTMSpectralSegmenter.cpp:
/// 229-239`).
///
/// Divergence (load-bearing, see the Task 5 crafted-case unit test): the BLSTM
/// variant re-clamps TWICE -- once immediately after `freq_beg` is raised
/// (`if freq_beg > freq_end: freq_beg = freq_end`, using freq_end's value
/// BEFORE the max-freq computation) and again after `freq_end` is computed
/// (`if freq_end < freq_beg: freq_end = freq_beg`). The LTSV variant here has
/// only ONE guard, evaluated AFTER both `freq_beg` and `freq_end` are computed:
/// `if freq_beg > freq_end: freq_beg = freq_end` -- there is no second reclamp
/// pulling `freq_end` back up. For a min/max pair where the intermediate
/// (pre-max-freq) `freq_beg` already exceeds the periodogram's original
/// `freq_end`, the two variants land on different final band values (the BLSTM
/// variant snaps to the ORIGINAL `freq_end` = `periodogram_length-1`; the LTSV
/// variant snaps to the POST-max-freq-clamped `freq_end`).
///
/// Returns `(freq_beg, freq_end, min_freq_snapped, max_freq_snapped)`, where the
/// snapped Hz values are `idx*freq_step` (`:208-209`), matching the legacy's
/// mutation of `_MinFreq`/`_MaxFreq`. `rate` is the sample rate; `bins` is the
/// `periodogram_length` (`freq_step = rate/2/(bins-1)`, matching `:202`'s
/// `freq_end` divisor at that point, i.e. `bins-1`).
pub fn derive_freq_band_ltsv_variant(
    cfg: &FeatureConfig,
    rate: f64,
    bins: usize,
) -> (usize, usize, f64, f64) {
    let mut freq_beg: usize = 0;
    let mut freq_end: usize = bins - 1;
    let freq_step = rate / 2.0 / freq_end as f64;

    let tmp = (cfg.min_freq / freq_step).floor() as usize;
    if freq_beg < tmp {
        freq_beg = tmp;
    }
    let tmp = (cfg.max_freq / freq_step).ceil() as usize;
    if freq_end > tmp {
        freq_end = tmp;
    }
    if freq_beg > freq_end {
        freq_beg = freq_end;
    }

    let min_freq = freq_beg as f64 * freq_step;
    let max_freq = freq_end as f64 * freq_step;
    (freq_beg, freq_end, min_freq, max_freq)
}

/// Assemble the BLSTM input sequence (`getBLSTMInputSequence`, `:561-591`).
///
/// Spectral-path priority (`:568-576`): if `dct` is `Some` use it verbatim (all
/// columns); else if `mel` is `Some` use it verbatim; else the raw path
/// `ln(p.block(:, freq_beg..=freq_end) + 1e-24)` (band-limit ONLY here). Then, if
/// `ltsv` is `Some`, hcat it as the LAST column (`:578-582`).
pub fn assemble_input_sequence(
    p: &Array2<f64>,
    mel: Option<&Array2<f64>>,
    dct: Option<&Array2<f64>>,
    ltsv: Option<&[f64]>,
    freq_beg: usize,
    freq_end: usize,
) -> Array2<f64> {
    let spectral: Array2<f64> = if let Some(dct) = dct {
        dct.to_owned()
    } else if let Some(mel) = mel {
        mel.to_owned()
    } else {
        // raw path: ln(block + 1e-24) over the spectral band [freq_beg, freq_end].
        let t = p.nrows();
        let width = freq_end - freq_beg + 1;
        let mut out = Array2::<f64>::zeros((t, width));
        for jj in 0..t {
            for ii in 0..width {
                out[[jj, ii]] = (p[[jj, freq_beg + ii]] + 1e-24).ln();
            }
        }
        out
    };

    match ltsv {
        None => spectral,
        Some(ltsv) => {
            let t = spectral.nrows();
            let cols = spectral.ncols();
            let mut merged = Array2::<f64>::zeros((t, cols + 1));
            for jj in 0..t {
                for ii in 0..cols {
                    merged[[jj, ii]] = spectral[[jj, ii]];
                }
                merged[[jj, cols]] = ltsv[jj];
            }
            merged
        }
    }
}

/// Build the full BLSTM input sequence for one channel: windowing -> periodogram
/// -> mel/DCT (per config) -> LTSV (band asymmetry, `R >= 1` gate) -> assemble.
///
/// Lifted verbatim from the Phase 1/2 gate test bodies (`run_pipeline` /
/// `assemble_real_input`) -- the two call sites differ only in config source; the
/// real config's LTSV is disabled via `R == 0`, already handled by the
/// `ltsv_half_window >= 1` gate below. Preemph/noise are NOT applied here -- callers
/// own audio mutation before passing `audio` in.
pub fn build_input_sequence(
    audio: &Audio,
    cfg: &FeatureConfig,
    s: &SpectralParams,
    chan: usize,
    temporal_conv: Option<&[f64]>,
) -> Array2<f64> {
    let rate = audio.sample_rate as f64;
    let win = windowing_coefficients(&cfg.win_type, false, s.buffer_size, cfg.win_param);
    let end = audio.data.ncols() - 1;
    let perio = compute_segment_periodogram_estimates(
        audio,
        s.order,
        s.shift_frames,
        chan,
        cfg.flag_dc_offset,
        win.as_deref(),
        temporal_conv,
        0,
        end,
    );

    // Mel bank per config, SNAPPED freq band, spectrum_size = bins - 1.
    let (mel_out, dct_out) = if cfg.nb_bins > 0 {
        let bank = MelFilterBank::new(
            cfg.min_mel,
            cfg.max_mel,
            cfg.nb_bins,
            s.min_freq,
            s.max_freq,
            rate,
            s.bins - 1,
            cfg.is_log,
            cfg.nb_dct,
            cfg.ignore_first,
            cfg.deltas_nb,
            cfg.dd_nb,
        );
        let fb = bank.apply_filter_bank(&perio);
        if cfg.nb_dct > 0 {
            let dct = bank.apply_dct(&fb);
            (Some(fb), Some(dct))
        } else {
            (Some(fb), None)
        }
    } else {
        (None, None)
    };

    // Spectral output columns (before LTSV) drive the mel-variant LTSV band.
    let spectral_cols = match (&dct_out, &mel_out) {
        (Some(d), _) => d.ncols(),
        (None, Some(m)) => m.ncols(),
        (None, None) => s.freq_end - s.freq_beg + 1,
    };

    // LTSV column (band asymmetry): mel active -> (0, spectral_cols-1); else the
    // spectral band. Appended when R >= 1.
    let ltsv = if s.ltsv_half_window >= 1 {
        let (lb, le) = if cfg.nb_bins > 0 {
            (0, spectral_cols - 1)
        } else {
            (s.freq_beg, s.freq_end)
        };
        Some(get_ltsv(&perio, lb, le, s.ltsv_half_window, s.ltsv_shift))
    } else {
        None
    };

    assemble_input_sequence(
        &perio,
        mel_out.as_ref(),
        dct_out.as_ref(),
        ltsv.as_deref(),
        s.freq_beg,
        s.freq_end,
    )
}
