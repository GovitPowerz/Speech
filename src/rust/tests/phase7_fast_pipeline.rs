//! Phase 7 Task 3: `fast::pipeline::FastPipeline` f32-on-realfft feature parity vs
//! the exact f64 `features::pipeline::build_input_sequence`.
//!
//! The fast path DIVERGES from exact BY DESIGN: f32 throughout the FFT/periodogram +
//! realfft's ONE-real-FFT-per-frame (vs the exact GFFT's two-real packing) => a
//! different reduction order in a different precision. Tolerances here are MEASURED
//! first (the `MEASURE` prints, `cargo test -- --nocapture`), then pinned with
//! headroom; the measured values are recorded in each test's docstring.
//!
//! Gate config is `phase4a/tier2_spectral.config` (Algo 3, spectrum_order 10 ->
//! 1024-pt FFT / 513 bins, nb_bins 20 + is_log_mel + nb_DCT 4 + deltas 5/3 +
//! IgnoreFirstDCT -> the T x 11 MFCC+delta assembled sequence; LTSVwindow 0 and
//! TDCwindow 0 and temporal_convolution_size 0 -> LTSV / pitch / temporal-conv all
//! OFF, so the assembled path is periodogram -> log-mel -> DCT).

use ndarray::Array2;
use speech::audio::{Audio, read_audio};
use speech::fast::pipeline::FastPipeline;
use speech::features::pipeline::{FeatureConfig, SpectralParams, build_input_sequence};

const TIER2: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/reference_data/phase4a/tier2_spectral.config"
);
const EXCERPT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/reference_data/phase1/excerpt_2ch_8k.wav"
);
const PRCTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/reference_data/phase4d/prcts_excerpt.wav"
);

/// Parse the tier2 config into the (map, FeatureConfig) the spectral path uses
/// (prefix "BLSTM", matching `BlstmSpectralSegmenter::from_legacy`).
fn tier2_cfg() -> (indexmap::IndexMap<String, String>, FeatureConfig) {
    let text = std::fs::read_to_string(TIER2).unwrap();
    let map = speech::legacy_config::parse_legacy_config(&text);
    let cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
    (map, cfg)
}

/// Read a wav fully (large max-duration) and narrow one channel to f32.
fn samples_f32(audio: &Audio, chan: usize) -> Vec<f32> {
    audio.data.row(chan).iter().map(|&x| x as f32).collect()
}

/// Run BOTH paths on `audio` channel `chan`: exact `build_input_sequence` (f64) and
/// `FastPipeline::build_input_sequence` (f32). Temporal conv is None (gate: size 0).
fn run_both(
    cfg: &FeatureConfig,
    s: &SpectralParams,
    audio: &Audio,
    chan: usize,
    rate: f64,
) -> (Array2<f64>, speech::fast::nn::FastMatrix) {
    let exact = build_input_sequence(audio, cfg, s, chan, None);
    let mut pipe = FastPipeline::new(s, cfg, rate).unwrap();
    let fast = pipe.build_input_sequence(&samples_f32(audio, chan)).clone();
    (exact, fast)
}

/// `(max_abs, max_rel)`: max absolute per-element delta, and max relative delta with
/// denom `max(|exact|, 1.0)` (near-zero delta cells fall back to absolute; large
/// static columns use true relative).
fn measure(exact: &Array2<f64>, fast: &speech::fast::nn::FastMatrix) -> (f64, f64) {
    assert_eq!(exact.nrows(), fast.rows, "row count mismatch");
    assert_eq!(exact.ncols(), fast.cols, "col count mismatch");
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for r in 0..fast.rows {
        for c in 0..fast.cols {
            let e = exact[[r, c]];
            let f = fast.get(r, c) as f64;
            let d = (e - f).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / e.abs().max(1.0));
        }
    }
    (max_abs, max_rel)
}

// ---------------------------------------------------------------------------
// Unit pins.
// ---------------------------------------------------------------------------

#[test]
fn windowing_coeffs_f32_match_f64_narrowed() {
    // The pipeline stores its window coefficients as f32 (narrowed once from the exact
    // f64 `windowing_coefficients`). Each stored f32 must bit-equal the f64 coeff
    // narrowed to f32 (buffer width = window_size + 1 = 1025 for the tier2 order-10).
    let (_, cfg) = tier2_cfg();
    let s = SpectralParams::derive(&cfg, 8000.0);
    let pipe = FastPipeline::new(&s, &cfg, 8000.0).unwrap();
    let f64_coeffs =
        speech::audio::windowing_coefficients(&cfg.win_type, false, s.buffer_size, cfg.win_param)
            .expect("tier2 hHCw window is non-None");
    let stored = pipe
        .debug_win_coeffs()
        .expect("pipeline stores window coeffs");
    assert_eq!(stored.len(), f64_coeffs.len(), "coeff length");
    assert_eq!(stored.len(), s.buffer_size, "coeff length == buffer_size");
    for (k, (&sf, &df)) in stored.iter().zip(f64_coeffs.iter()).enumerate() {
        assert_eq!(
            sf.to_bits(),
            (df as f32).to_bits(),
            "window coeff narrowing mismatch at {k}: stored={sf} expected={}",
            df as f32
        );
    }
}

#[test]
fn build_input_sequence_twice_bit_identical() {
    // Plan + scratch reuse must be clean: two build_input_sequence calls on the same
    // pipeline give BIT-IDENTICAL f32 output (the mel bank + realfft plan are built
    // once and reused; no per-call allocation contaminates the result).
    let (_, cfg) = tier2_cfg();
    let audio = read_audio(std::path::Path::new(EXCERPT), 0.0, 3.6e6, 0, None).unwrap();
    let s = SpectralParams::derive(&cfg, audio.sample_rate as f64);
    let mut pipe = FastPipeline::new(&s, &cfg, audio.sample_rate as f64).unwrap();
    let samples = samples_f32(&audio, 0);

    let first = pipe.build_input_sequence(&samples).clone();
    let second = pipe.build_input_sequence(&samples).clone();
    assert_eq!(first.rows, second.rows);
    assert_eq!(first.cols, second.cols);
    for (i, (&a, &b)) in first.data.iter().zip(second.data.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "repeated call diverged at {i}");
    }

    // A different-length input between the two identical calls (channel 1 has the same
    // length here, so use the 60 s file's channel to force a workspace resize) must not
    // perturb the repeat.
    let big = read_audio(std::path::Path::new(PRCTS), 0.0, 3.6e6, 0, None).unwrap();
    let _ = pipe.build_input_sequence(&samples_f32(&big, 0));
    let third = pipe.build_input_sequence(&samples).clone();
    for (i, (&a, &b)) in first.data.iter().zip(third.data.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "post-resize repeat diverged at {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// Tolerance parity.
// ---------------------------------------------------------------------------

#[test]
fn pipeline_parity_excerpt_3s() {
    // MEASURED (Apple Silicon dev box, Phase 10 Task 5 re-measurement): max_abs=3.589e-5,
    // max_rel=1.807e-5 over both channels of the committed 3 s excerpt (T x 11 MFCC+delta
    // sequence). Pinned at rel 2e-4 / abs 4e-4 (~11x each) over measured; the f32 +
    // realfft-vs-GFFT divergence is by design (module docs).
    //
    // RE-PINNED ONCE by the S4 full-f32-mel sweep (was: max_abs=8.263e-6 / max_rel=3.044e-6
    // at rel 5e-5 / abs 1e-4). The deltas grew ~4-6x because f32 error now accumulates
    // through the mel dots, the DCT product and the delta regressions, not only the
    // periodogram -- the phase-7 arrangement widened the f32 periodogram to f64 and ran
    // the golden f64 mel on it. This is the ONE sanctioned re-measurement (spec S9.3);
    // the decision-level gates it feeds (phase7_parity_sad's boundary count/types +
    // max_dt) did NOT move, which is what R1 actually protects.
    const REL_PIN: f64 = 2.0e-4;
    const ABS_PIN: f64 = 4.0e-4;
    let (_, cfg) = tier2_cfg();
    let audio = read_audio(std::path::Path::new(EXCERPT), 0.0, 3.6e6, 0, None).unwrap();
    let rate = audio.sample_rate as f64;
    let s = SpectralParams::derive(&cfg, rate);

    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    for chan in 0..audio.data.nrows() {
        let (exact, fast) = run_both(&cfg, &s, &audio, chan, rate);
        assert_eq!(fast.cols, 11, "tier2 assembled width (3*4 - ignoreFirst)");
        let (abs, rel) = measure(&exact, &fast);
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(rel);
        assert!(rel < REL_PIN, "rel {rel} exceeds pin (chan={chan})");
        assert!(abs < ABS_PIN, "abs {abs} exceeds pin (chan={chan})");
    }
    println!("MEASURE pipeline_parity_excerpt_3s: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}");
}

#[test]
fn pipeline_parity_prcts_60s() {
    // MEASURED (Apple Silicon dev box, Phase 10 Task 5 re-measurement): max_abs=5.914e-5,
    // max_rel=4.038e-5 over both channels of the committed 60 s prcts excerpt (6000
    // frames/channel). Pinned at rel 6e-4 (~15x) / abs 9e-4 (~15x) headroom over measured
    // -- the wider pin than the 3 s test absorbs both the larger frame count and
    // cross-platform realfft SIMD variance (CI is x86, this box is ARM), the same
    // rationale and the same headroom multiple as the phase-7 pin it replaces.
    //
    // RE-PINNED ONCE by the S4 full-f32-mel sweep (was: max_abs=1.805e-5 /
    // max_rel=9.805e-6 at rel 1.5e-4 / abs 3e-4). See the 3 s test for the mechanism.
    const REL_PIN: f64 = 6.0e-4;
    const ABS_PIN: f64 = 9.0e-4;
    let (_, cfg) = tier2_cfg();
    let audio = read_audio(std::path::Path::new(PRCTS), 0.0, 3.6e6, 0, None).unwrap();
    let rate = audio.sample_rate as f64;
    let s = SpectralParams::derive(&cfg, rate);

    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    for chan in 0..audio.data.nrows() {
        let (exact, fast) = run_both(&cfg, &s, &audio, chan, rate);
        let (abs, rel) = measure(&exact, &fast);
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(rel);
        assert!(rel < REL_PIN, "rel {rel} exceeds pin (chan={chan})");
        assert!(abs < ABS_PIN, "abs {abs} exceeds pin (chan={chan})");
    }
    println!("MEASURE pipeline_parity_prcts_60s: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}");
}

#[test]
fn pipeline_parity_dc_offset_branch() {
    // tier2 has flag_DCOffset false, so the fast pipeline's per-frame DC-subtraction
    // branch (`fill_frame`'s `if dc { ... }`, f32) is UNEXERCISED by the gate config
    // (Phase 7 Task 4 rider). Flip it true and re-check parity: both the exact
    // (`get_sequence`) and fast (`fill_frame`) framing subtract the full-buffer mean,
    // so the assembled sequences must still agree at tolerance -- but WIDER than the
    // no-DC path: subtracting an f32-computed full-buffer mean amplifies the f32 error
    // on low-energy periodogram bins (a near-zero bin's log-mel diverges), so the DC
    // branch diverges ~1.5 orders more than the flag-off 3 s test (rel 1.8e-5). MEASURED
    // (Apple Silicon dev box, Phase 10 Task 5 re-measurement): max_abs=9.784e-3,
    // max_rel=5.719e-4 over both channels of the 3 s excerpt. Pinned rel ~10x / abs ~5x
    // over measured (the DC branch is an UNEXERCISED gate path -- a documented, wider f32
    // tolerance, spec S4). The S4 full-f32 mel BARELY moved this leg (was abs 9.784e-3 /
    // rel 5.704e-4, pins unchanged): here the divergence is already dominated by the
    // f32 DC-mean subtraction upstream of the mel, so the mel's own f32 error is noise
    // against it -- the one leg where the phase-7 pins survive the sweep untouched.
    const REL_PIN: f64 = 6.0e-3;
    const ABS_PIN: f64 = 5.0e-2;
    let (mut map, _) = tier2_cfg();
    map.insert("BLSTM_flag_DCOffset".into(), "true".into());
    let cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
    let audio = read_audio(std::path::Path::new(EXCERPT), 0.0, 3.6e6, 0, None).unwrap();
    let rate = audio.sample_rate as f64;
    let s = SpectralParams::derive(&cfg, rate);

    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    for chan in 0..audio.data.nrows() {
        let (exact, fast) = run_both(&cfg, &s, &audio, chan, rate);
        let (abs, rel) = measure(&exact, &fast);
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(rel);
        assert!(
            rel < REL_PIN,
            "dc-branch rel {rel} exceeds pin (chan={chan})"
        );
        assert!(
            abs < ABS_PIN,
            "dc-branch abs {abs} exceeds pin (chan={chan})"
        );
    }
    println!(
        "MEASURE pipeline_parity_dc_offset_branch: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Typed-bail pins (unexercised gate paths).
// ---------------------------------------------------------------------------

#[test]
fn new_bails_on_ltsv_active() {
    // LTSVwindow > 0 -> ltsv_half_window >= 1 -> the LTSV column is NOT supported on the
    // fast path (unexercised by the gate). new() typed-bails at construction.
    let (mut map, _) = tier2_cfg();
    map.insert("BLSTM_LTSVwindow".into(), "0.5".into()); // -> ltsv_half_window 25
    let cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
    let s = SpectralParams::derive(&cfg, 8000.0);
    assert!(s.ltsv_half_window >= 1, "sanity: LTSV is active");
    match FastPipeline::new(&s, &cfg, 8000.0) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("ltsv"),
            "expected an LTSV bail, got: {e}"
        ),
        Ok(_) => panic!("expected FastPipeline::new to bail on active LTSV"),
    }
}

#[test]
fn new_bails_on_temporal_conv_active() {
    // spectrum_temporal_convolution_size > 0 (with a real window type) -> the
    // periodogram temporal convolution is active, unsupported on the fast path. Bail.
    let (mut map, _) = tier2_cfg();
    map.insert(
        "BLSTM_spectrum_temporal_convolution_size".into(),
        "4".into(),
    );
    let cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
    let s = SpectralParams::derive(&cfg, 8000.0);
    match FastPipeline::new(&s, &cfg, 8000.0) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("convolution"),
            "expected a temporal-convolution bail, got: {e}"
        ),
        Ok(_) => panic!("expected FastPipeline::new to bail on active temporal convolution"),
    }
}
