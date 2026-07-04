//! Phase 2b Task 2: `DriverConfig`, `results_to_segmentation`, the real `Segmenter`
//! trait, and `Segmentation::clear_hypothesis`.
//!
//! Goldens dumped by the harness `SegProbe` stage (`tools/oracle_harness/main.cpp`,
//! Phase 2b Task 2 block): the REAL `Segmenter::buildFromConf` + `results2segmentation`
//! run against the real `1_worker_1.config` (BLSTM prefix) and a deterministic
//! synthetic 1x60 result row. `r2s_conv_coeff.bin`/`r2s_convolved.bin` traverse cos
//! (hHCw windowing coefficients) -> oracle-gated compare; `r2s_boundaries*.bin`
//! begin times are arithmetic on the convolved values -> also oracle-gated; the type
//! column is exact ints, compared directly.

mod common;

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmenter::{DriverConfig, SegmenterConfig, results_to_segmentation};

fn cfg_text() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/1_worker_1.config");
    std::fs::read_to_string(p).unwrap()
}

fn real_map() -> IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(&cfg_text())
}

/// The exact synthetic row the harness SegProbe stage builds (`main.cpp`,
/// Phase 2b Task 2): `r[k] = 0.2 + 0.7*((k*13) mod 17)/16`, 60 columns.
fn synthetic_row() -> Vec<f64> {
    (0..60)
        .map(|k: i64| 0.2 + 0.7 * ((k * 13) % 17) as f64 / 16.0)
        .collect()
}

/// Read the audio_duration used by the harness Segmentation (the End-sentinel begin
/// time in the boundary dump, second row).
fn dump_audio_duration() -> f64 {
    let m = common::load_bin_phase2b("r2s_boundaries.bin");
    m[[m.nrows() - 1, 0]]
}

// === DriverConfig::from_config on the real config ============================

#[test]
fn driver_config_real_values() {
    let m = real_map();
    let cfg = DriverConfig::from_config(&m, "BLSTM").unwrap();
    assert_eq!(cfg.dump_dir, "");
    assert_eq!(cfg.window_size_sec, 3.25);
    assert_eq!(cfg.window_shift_sec, 0.8);
    assert_eq!(cfg.back_prop_wer, -1.0);
    assert!(cfg.conv_coeff.is_some());
}

#[test]
fn driver_config_conv_coeff_matches_dump() {
    let m = real_map();
    let cfg = DriverConfig::from_config(&m, "BLSTM").unwrap();
    let coeff = cfg.conv_coeff.unwrap();
    assert_eq!(coeff.len(), 19);

    let want = common::load_bin_phase2b("r2s_conv_coeff.bin");
    let got = ndarray::Array2::from_shape_vec((1, coeff.len()), coeff).unwrap();
    common::assert_oracle_eq(&got, &want, "driver_config_conv_coeff");
}

#[test]
fn verify_windowing_type_valid_is_noop() {
    assert_eq!(
        speech::tasks::segmenter::verify_windowing_type("hHCw"),
        None
    );
    assert_eq!(
        speech::tasks::segmenter::verify_windowing_type("none"),
        None
    );
}

#[test]
fn verify_windowing_type_invalid_warns_without_mutating() {
    // By-value no-fix quirk: the function returns a warning but the caller's stored
    // type is NOT reset to "none" anywhere in this port (there is nothing to mutate
    // -- verify_windowing_type takes `&str`, not a place to write back to).
    let warning = speech::tasks::segmenter::verify_windowing_type("bogus");
    assert!(warning.is_some());
    assert!(warning.unwrap().contains("not supported"));
}

// === results_to_segmentation replay vs the convolved-results golden ==========

#[test]
fn results_to_segmentation_convolved_matches_dump() {
    let m = real_map();
    let cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();
    let driver = DriverConfig::from_config(&m, "BLSTM").unwrap();
    let conv = driver.conv_coeff.clone().unwrap();

    let audio_duration = dump_audio_duration();
    let mut seg = Segmentation::new(audio_duration);
    let mut results = synthetic_row();

    results_to_segmentation(
        &mut seg,
        0.04,
        0.0,
        &mut results,
        SegClass::Speech,
        Some(&conv),
        &cfg,
    );

    let want = common::load_bin_phase2b("r2s_convolved.bin");
    let got = ndarray::Array2::from_shape_vec((1, results.len()), results.clone()).unwrap();
    common::assert_oracle_eq(&got, &want, "results_to_segmentation_convolved");
}

#[test]
fn results_to_segmentation_in_place_mutation_visible_to_caller() {
    let m = real_map();
    let cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();
    let driver = DriverConfig::from_config(&m, "BLSTM").unwrap();
    let conv = driver.conv_coeff.clone().unwrap();

    let audio_duration = dump_audio_duration();
    let mut seg = Segmentation::new(audio_duration);
    let original = synthetic_row();
    let mut results = original.clone();

    results_to_segmentation(
        &mut seg,
        0.04,
        0.0,
        &mut results,
        SegClass::Speech,
        Some(&conv),
        &cfg,
    );

    // The convolution kernel is not a trivial identity, so a bit-for-bit compare
    // against the pre-call copy proves the caller's Vec was actually mutated
    // in place (not just internally, e.g. via a hidden clone).
    assert_ne!(results, original);
}

#[test]
fn results_to_segmentation_boundaries_match_dump() {
    let m = real_map();
    let cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();
    let driver = DriverConfig::from_config(&m, "BLSTM").unwrap();
    let conv = driver.conv_coeff.clone().unwrap();

    let audio_duration = dump_audio_duration();
    let mut seg = Segmentation::new(audio_duration);
    let mut results = synthetic_row();

    results_to_segmentation(
        &mut seg,
        0.04,
        0.0,
        &mut results,
        SegClass::Speech,
        Some(&conv),
        &cfg,
    );

    let want = common::load_bin_phase2b("r2s_boundaries.bin");
    let segs = seg.segments();
    assert_eq!(segs.len(), want.nrows(), "boundary count mismatch");
    for (i, s) in segs.iter().enumerate() {
        let want_begin = ndarray::Array2::from_shape_vec((1, 1), vec![s.begin]).unwrap();
        let want_dump = ndarray::Array2::from_shape_vec((1, 1), vec![want[[i, 0]]]).unwrap();
        common::assert_oracle_eq(&want_begin, &want_dump, &format!("boundary[{i}].begin"));
        assert_eq!(
            s.ty as i32,
            want[[i, 1]] as i32,
            "boundary[{i}].type mismatch"
        );
    }
}

// === noconv variant: results untouched =======================================

#[test]
fn results_to_segmentation_noconv_leaves_results_untouched() {
    let m = real_map();
    let cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();

    let audio_duration = dump_audio_duration();
    let mut seg = Segmentation::new(audio_duration);
    let original = synthetic_row();
    let mut results = original.clone();

    results_to_segmentation(
        &mut seg,
        0.04,
        0.0,
        &mut results,
        SegClass::Speech,
        None,
        &cfg,
    );

    // Bit-exact: no conv -> no arithmetic touches `results` at all.
    assert_eq!(results, original);

    let want = common::load_bin_phase2b("r2s_boundaries_noconv.bin");
    let segs = seg.segments();
    assert_eq!(segs.len(), want.nrows());
    for (i, s) in segs.iter().enumerate() {
        let want_begin = ndarray::Array2::from_shape_vec((1, 1), vec![s.begin]).unwrap();
        let want_dump = ndarray::Array2::from_shape_vec((1, 1), vec![want[[i, 0]]]).unwrap();
        common::assert_oracle_eq(
            &want_begin,
            &want_dump,
            &format!("noconv boundary[{i}].begin"),
        );
        assert_eq!(
            s.ty as i32,
            want[[i, 1]] as i32,
            "noconv boundary[{i}].type mismatch"
        );
    }
}

// === clear_hypothesis unit test ==============================================

#[test]
fn clear_hypothesis_resets_to_seeded_state_preserving_audio_duration() {
    let mut seg = Segmentation::new(10.0);
    seg.label_segment(1.0, 3.0, SegClass::Speech);
    seg.label_segment(5.0, 6.0, SegClass::Excluded);
    assert!(seg.segments().len() > 2);

    seg.clear_hypothesis();

    let segs = seg.segments();
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].begin, 0.0);
    assert_eq!(segs[0].ty, SegClass::Other);
    assert_eq!(segs[1].begin, 10.0);
    assert_eq!(segs[1].ty, SegClass::End);
    assert_eq!(seg.audio_duration(), 10.0);
}
