//! Phase 1 Task 11: the END-TO-END FEATURE PARITY GATE (the phase acceptance test).
//!
//! `FeatureConfig::from_legacy` (key reads + sanitization,
//! `LongTermSpectralVariation.cpp:44-80` + `BLSTMSpectralSegmenter.cpp:44-83`),
//! `SpectralParams::derive` (param derivation, `BLSTMSpectralSegmenter.cpp:199-311`),
//! and `assemble_input_sequence` (`getBLSTMInputSequence`, `:561-591`), golden-tested
//! bit-for-bit against the oracle harness dumps for four variant configs.
//!
//! The full-pipeline gate for each variant reproduces the legacy exactly: read the
//! excerpt (0.35/2.0) -> preemph 0.97 -> derive -> windowing -> periodogram ->
//! mel/dct per config -> LTSV (band asymmetry: (0, dim-1) when mel active, else the
//! spectral band) -> assemble, then `assert_bits_eq` vs `inputseq_<v>_chan{1,2}.bin`.

mod common;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::audio::{Audio, read_audio, windowing_coefficients};
use speech::features::pipeline::{
    FeatureConfig, SpectralParams, assemble_input_sequence, build_input_sequence,
};
use speech::legacy_config::parse_legacy_config;

// --- Helpers ----------------------------------------------------------------

fn variant_cfg(name: &str) -> FeatureConfig {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase1")
        .join(format!("variant_{name}.config"));
    let text = std::fs::read_to_string(&path).unwrap();
    let map = parse_legacy_config(&text);
    FeatureConfig::from_legacy(&map, "BLSTM").unwrap()
}

fn excerpt_audio(preemph_ratio: f64) -> Audio {
    let mut audio = read_audio(&common::fixture("excerpt_2ch_8k.wav"), 0.35, 2.0).expect("decode");
    // legacy gate: `if (preemphRatio > 0)` (audio.rs:40) -- apply only when positive.
    if preemph_ratio > 0.0 {
        audio.apply_preemph(preemph_ratio);
    }
    // noise skipped: seed 0 in every variant config.
    audio
}

// --- from_legacy typed field asserts ----------------------------------------

#[test]
fn from_legacy_mfcc_deltas_fields() {
    let c = variant_cfg("mfcc_deltas");
    assert_eq!(c.order, 8);
    assert_eq!(c.shift_sec, 0.01);
    assert_eq!(c.conv_size, 0);
    assert_eq!(c.conv_type, "none");
    assert_eq!(c.min_mel, 64.0);
    assert_eq!(c.max_mel, 3800.0);
    assert_eq!(c.nb_bins, 26);
    assert!(c.is_log);
    assert_eq!(c.nb_dct, 13);
    assert!(!c.ignore_first);
    assert_eq!(c.deltas_nb, 3);
    assert_eq!(c.dd_nb, 3);
    assert_eq!(c.min_freq, 64.0);
    assert_eq!(c.max_freq, 3800.0);
    assert_eq!(c.win_type, "hamming");
    assert_eq!(c.win_param, 0.83333);
    assert!(c.flag_dc_offset);
    assert_eq!(c.preemph_ratio, 0.97);
    assert_eq!(c.noise_seed, 0);
    assert_eq!(c.noise_ratio, 0.001);
    assert_eq!(c.ltsv_window, 0.3);
    assert_eq!(c.ltsv_shift, 0.04);
    assert_eq!(c.tdc_window, 0.0);
    assert!(c.tdc.is_none());
}

#[test]
fn from_legacy_mfcc_sdc_fields() {
    let c = variant_cfg("mfcc_sdc");
    assert_eq!(c.deltas_nb, -1);
    assert_eq!(c.dd_nb, 0);
    assert!(c.ignore_first);
    assert_eq!(c.nb_dct, 13);
}

#[test]
fn from_legacy_logmel_fields() {
    let c = variant_cfg("logmel");
    assert_eq!(c.nb_dct, 0);
    assert_eq!(c.deltas_nb, 0);
    assert_eq!(c.dd_nb, 0);
    assert_eq!(c.nb_bins, 26);
    assert!(c.is_log);
}

#[test]
fn from_legacy_rawband_fields() {
    let c = variant_cfg("rawband_ltsv");
    assert_eq!(c.nb_bins, 0);
    assert_eq!(c.min_freq, 300.0);
    assert_eq!(c.max_freq, 3300.0);
}

// --- Sanitization unit cases ------------------------------------------------

fn base_map() -> IndexMap<String, String> {
    // A full valid map; individual cases mutate the keys under test.
    let text = "\
P_spectrum_order 8
P_spectrum_shift 0.01
P_spectrum_temporal_convolution_size 0
P_spectrum_temporal_convolution_type none
P_windowing_type hamming
P_windowing_param 0.5
P_flag_DCOffset true
P_preemph_ratio 0.97
P_noise_seed 0
P_noise_ratio 0.001
P_minFreq 64
P_maxFreq 3800
P_minMelFreq 64
P_maxMelFreq 3800
P_nb_bins 26
P_is_log_mel true
P_nb_DCT 13
P_IgnoreFirstDCT false
P_ComputeDeltasNb 3
P_ComputeDeltaDeltasNb 3
P_LTSVwindow 0.3
P_LTSVshift 0.04
P_TDCwindow 0
";
    parse_legacy_config(text)
}

#[test]
fn sanitize_negative_minfreq_zeroed_before_swap() {
    // minFreq=-5 -> 0 BEFORE the swap; maxFreq=3800 stays. No swap (0 < 3800).
    let mut m = base_map();
    m.insert("P_minFreq".into(), "-5".into());
    let c = FeatureConfig::from_legacy(&m, "P").unwrap();
    assert_eq!(c.min_freq, 0.0);
    assert_eq!(c.max_freq, 3800.0);
}

#[test]
fn sanitize_mel_pair_swapped_when_reversed() {
    // maxMelFreq < minMelFreq -> swapped.
    let mut m = base_map();
    m.insert("P_minMelFreq".into(), "3800".into());
    m.insert("P_maxMelFreq".into(), "64".into());
    let c = FeatureConfig::from_legacy(&m, "P").unwrap();
    assert_eq!(c.min_mel, 64.0);
    assert_eq!(c.max_mel, 3800.0);
}

#[test]
fn sanitize_span_widened_both_endpoints() {
    // |max - min| < 2 -> widened to mean +- 1 (both endpoints move).
    let mut m = base_map();
    m.insert("P_minFreq".into(), "100".into());
    m.insert("P_maxFreq".into(), "101".into());
    let c = FeatureConfig::from_legacy(&m, "P").unwrap();
    // mean = 100.5 -> [99.5, 101.5]
    assert_eq!(c.min_freq, 99.5);
    assert_eq!(c.max_freq, 101.5);
}

#[test]
fn sanitize_negative_then_swap_order() {
    // Ops commute here (both endpoints end up 0 either way), so this case can't by
    // itself distinguish "negatives zeroed before swap" from "swap then zero" --
    // the literal order (negative-zero, then swap-if-reversed, then span-widen) is
    // preserved by the implementation (see `from_legacy`) and exercised for the
    // no-swap case by `sanitize_negative_minfreq_zeroed_before_swap` above; this
    // test only pins the DOWNSTREAM span-widen composing correctly with negative
    // zeroing (both zeroed -> span 0 < 2 -> widened to mean(0) +- 1 = [-1, 1]).
    let mut m = base_map();
    m.insert("P_minFreq".into(), "-5".into());
    m.insert("P_maxFreq".into(), "-10".into());
    let c = FeatureConfig::from_legacy(&m, "P").unwrap();
    assert_eq!(c.min_freq, -1.0);
    assert_eq!(c.max_freq, 1.0);
}

#[test]
fn sanitize_tdc_lags_too_short_is_err() {
    let mut m = base_map();
    m.insert("P_TDCwindow".into(), "0.032".into());
    m.insert("P_TDCshift".into(), "0.01".into());
    m.insert("P_TDC_lags".into(), "0.002".into()); // only 1 entry
    m.insert("P_TDC_balance".into(), "0.7".into());
    m.insert("P_TDC_windowing_type".into(), "hamming".into());
    m.insert("P_TDC_windowing_param".into(), "0.8".into());
    assert!(FeatureConfig::from_legacy(&m, "P").is_err());
}

#[test]
fn sanitize_tdc_lags_negatives_clamped() {
    let mut m = base_map();
    m.insert("P_TDCwindow".into(), "0.032".into());
    m.insert("P_TDCshift".into(), "0.01".into());
    m.insert("P_TDC_lags".into(), "-0.002,0.016".into());
    m.insert("P_TDC_balance".into(), "0.7".into());
    m.insert("P_TDC_windowing_type".into(), "hamming".into());
    m.insert("P_TDC_windowing_param".into(), "0.8".into());
    let c = FeatureConfig::from_legacy(&m, "P").unwrap();
    let tdc = c.tdc.unwrap();
    assert_eq!(tdc.lags, vec![0.0, 0.016]);
    assert_eq!(tdc.balance, 0.7);
    assert_eq!(tdc.shift, 0.01);
}

#[test]
fn from_legacy_missing_required_key_is_err() {
    let mut m = base_map();
    m.shift_remove("P_spectrum_order");
    assert!(FeatureConfig::from_legacy(&m, "P").is_err());
}

// --- SpectralParams::derive vs params_<variant>.bin -------------------------

fn assert_params_match(name: &str) {
    let c = variant_cfg(name);
    let s = SpectralParams::derive(&c, 8000.0);
    let want = common::load_bin(&format!("params_{name}.bin")); // 1 x 8
    // dump order: [order, window_size, bins, shift_frames, freq_beg, freq_end,
    // ltsv_half_window, ltsv_shift]. `bins` == periodogram_length == 2^(order-1)+1.
    let got = [
        s.order as f64,
        s.window_size as f64,
        s.bins as f64,
        s.shift_frames as f64,
        s.freq_beg as f64,
        s.freq_end as f64,
        s.ltsv_half_window as f64,
        s.ltsv_shift as f64,
    ];
    for (i, gv) in got.iter().enumerate() {
        let wv = want[[0, i]];
        assert_eq!(
            gv.to_bits(),
            wv.to_bits(),
            "params_{name}[{i}]: rust={gv} want={wv}"
        );
    }
}

#[test]
fn derive_params_mfcc_deltas() {
    assert_params_match("mfcc_deltas");
}
#[test]
fn derive_params_mfcc_sdc() {
    assert_params_match("mfcc_sdc");
}
#[test]
fn derive_params_logmel() {
    assert_params_match("logmel");
}
#[test]
fn derive_params_rawband_ltsv() {
    assert_params_match("rawband_ltsv");
}

#[test]
fn derive_tdc_branch_params() {
    // Task 9 harness constants (BLSTMSpectralSegmenter.cpp:341-370 + TimeDomainCorrel.cpp:100-109).
    // None of the four variant configs exercise TDC (all set TDCwindow 0), so this
    // is the only test pinning `SpectralParams::derive`'s TDC branch.
    let mut m = base_map();
    m.insert("P_TDCwindow".into(), "0.032".into());
    m.insert("P_TDCshift".into(), "0.01".into());
    m.insert("P_TDC_lags".into(), "0.002,0.016".into());
    m.insert("P_TDC_balance".into(), "0.7".into());
    m.insert("P_TDC_windowing_type".into(), "hamming".into());
    m.insert("P_TDC_windowing_param".into(), "0.8".into());
    let c = FeatureConfig::from_legacy(&m, "P").unwrap();

    let s = SpectralParams::derive(&c, 8000.0);
    let tdc = s.tdc.expect("TDCwindow != 0 must produce Some(TdcParams)");

    // half_window = round(0.032*8000/2) = round(128.0) = 128.
    assert_eq!(tdc.half_window, 128);
    // full_window = 2*half_window+1 = 257.
    assert_eq!(tdc.full_window, 257);
    // shift: ws = max(0.01, 1/8000) = 0.01; window_shift = round(0.01*8000/80) = 1
    // (shift_frames = round(P_spectrum_shift(0.01)*8000) = 80); >= 1, so stays 1;
    // shift = 1*80 = 80.
    assert_eq!(tdc.shift, 80);
    // min_lag = round(0.002*8000) = round(16.0) = 16, >= 1 so unchanged.
    assert_eq!(tdc.min_lag, 16);
    // max_lag = round(0.016*8000) = round(128.0) = 128; < full_window-1 (256) and
    // >= min_lag, so unchanged.
    assert_eq!(tdc.max_lag, 128);
    assert_eq!(tdc.balance, 0.7);

    let want_coeffs = windowing_coefficients("hamming", false, 257, 0.8);
    assert_eq!(tdc.coeffs, want_coeffs);
}

// --- assemble_input_sequence unit tests -------------------------------------

#[test]
fn assemble_raw_path_bands_and_logs() {
    // No mel/dct: raw path = ln(p.block(:, freq_beg..=freq_end) + 1e-24).
    let p = ndarray::arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
    let out = assemble_input_sequence(&p, None, None, None, 1, 2);
    assert_eq!(out.dim(), (2, 2));
    assert_eq!(out[[0, 0]].to_bits(), (2.0f64 + 1e-24).ln().to_bits());
    assert_eq!(out[[0, 1]].to_bits(), (3.0f64 + 1e-24).ln().to_bits());
    assert_eq!(out[[1, 0]].to_bits(), (6.0f64 + 1e-24).ln().to_bits());
    assert_eq!(out[[1, 1]].to_bits(), (7.0f64 + 1e-24).ln().to_bits());
}

#[test]
fn assemble_dct_priority_takes_all_columns() {
    // dct present: taken verbatim, mel/raw ignored, no band-limit.
    let p = ndarray::arr2(&[[9.0, 9.0]]);
    let mel = ndarray::arr2(&[[1.0, 2.0, 3.0]]);
    let dct = ndarray::arr2(&[[10.0, 20.0]]);
    let out = assemble_input_sequence(&p, Some(&mel), Some(&dct), None, 0, 1);
    common::assert_bits_eq(&out, &dct, "dct priority");
}

#[test]
fn assemble_mel_when_no_dct_takes_all_columns() {
    let p = ndarray::arr2(&[[9.0, 9.0]]);
    let mel = ndarray::arr2(&[[1.0, 2.0, 3.0]]);
    let out = assemble_input_sequence(&p, Some(&mel), None, None, 0, 1);
    common::assert_bits_eq(&out, &mel, "mel path");
}

#[test]
fn assemble_hcat_ltsv_last_column() {
    // LTSV appended as the LAST column, order [spectral..., ltsv].
    let mel = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let p = ndarray::arr2(&[[0.0, 0.0]]);
    let ltsv = [7.0, 8.0];
    let out = assemble_input_sequence(&p, Some(&mel), None, Some(&ltsv), 0, 1);
    assert_eq!(out.dim(), (2, 3));
    assert_eq!(out[[0, 0]], 1.0);
    assert_eq!(out[[0, 1]], 2.0);
    assert_eq!(out[[0, 2]], 7.0);
    assert_eq!(out[[1, 2]], 8.0);
}

// --- THE END-TO-END GATE ----------------------------------------------------

fn run_pipeline(name: &str, chan: usize) -> Array2<f64> {
    let c = variant_cfg(name);
    let audio = excerpt_audio(c.preemph_ratio);
    let rate = audio.sample_rate as f64;
    let s = SpectralParams::derive(&c, rate);
    build_input_sequence(&audio, &c, &s, chan, None)
}

fn assert_e2e(name: &str) {
    // The E2E chain reaches cos (windowing + DCT), ln + exp (mel bank ctor / log
    // branch / raw-band ln) -> canary-gated (bit-exact on the oracle libm, hybrid <=4 ULP/abs
    // elsewhere). Only the derived-param goldens (round/floor, portable) stay bits.
    for chan in 0..2 {
        let got = run_pipeline(name, chan);
        let want = common::load_bin(&format!("inputseq_{name}_chan{}.bin", chan + 1));
        common::assert_oracle_eq(&got, &want, &format!("inputseq_{name}_chan{}", chan + 1));
    }
}

#[test]
fn e2e_mfcc_deltas() {
    assert_e2e("mfcc_deltas");
}
#[test]
fn e2e_mfcc_sdc() {
    assert_e2e("mfcc_sdc");
}
#[test]
fn e2e_logmel() {
    assert_e2e("logmel");
}
#[test]
fn e2e_rawband_ltsv() {
    assert_e2e("rawband_ltsv");
}
