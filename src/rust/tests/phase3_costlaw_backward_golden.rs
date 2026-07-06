//! Phase 3 Task 2: CostLaw backward seam VALIDATION golden tests.
//!
//! `cost.rs`'s backward chain (per-law `deriv()`, `compute_unitary_delta`,
//! `compute_deltas`) was already ported bit-exactly in Phase 0b-i. This suite
//! does NOT transcribe anything new -- it probes the REAL compiled
//! `CostLaw::computeUnitaryDeltas`/`computeDeltas` (CostLaw.cpp:278-445,
//! CostLaw.h:6-213) directly as the golden (design spec S2 tier 2) and asserts
//! `cost.rs` agrees. All dumps come from `tools/oracle_harness/main.cpp`'s
//! Phase 3 Task 2 stage via `scripts/extract_phase3_fixtures.py`.
//!
//! Comparator discipline: the square (polynomial) law + the multiclass/WER
//! fusion paths are pure arithmetic (no libm) -> `assert_bits_eq` (strict bits
//! everywhere). The log/sqrt laws call `ln`/`sqrt` -> `assert_oracle_eq_f64`
//! (canary-gated: bit-exact on the oracle env, hybrid ULP/absolute off it).

mod common;

use indexmap::IndexMap;
use speech::cost::CostLaw;

fn real_config_text() -> String {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/1_worker_1.config");
    std::fs::read_to_string(p).unwrap()
}

/// Base map from the real `1_worker_1.config`, with `BLSTM_weightsFile` erased
/// (mirrors the harness's `conf._Params.erase("BLSTM_weightsFile")` -- CostLaw
/// never reads it, but keeping the maps' provenance identical to the harness's
/// construction is the point).
fn base_map() -> IndexMap<String, String> {
    let mut m = speech::legacy_config::parse_legacy_config(&real_config_text());
    m.shift_remove("BLSTM_weightsFile");
    m
}

fn poly_law() -> CostLaw {
    let mut m = base_map();
    m.insert("BLSTM_CostLawSpeech".into(), "square".into());
    m.insert("BLSTM_CostLawNoSpeech".into(), "square".into());
    CostLaw::from_config(&m, "BLSTM")
}

fn log_law() -> CostLaw {
    // The real config's own default (log/log) -- no override needed.
    CostLaw::from_config(&base_map(), "BLSTM")
}

fn sqrt_law() -> CostLaw {
    let mut m = base_map();
    m.insert("BLSTM_CostLawSpeech".into(), "sqrt".into());
    m.insert("BLSTM_CostLawNoSpeech".into(), "sqrt".into());
    CostLaw::from_config(&m, "BLSTM")
}

/// Replay the harness's deterministic k/64 sweep (k in [0,64], 65 points)
/// through `compute_unitary_delta` for a fixed target, returning an Nx2
/// `[output | delta]` matrix laid out the same way `common::load_bin_phase3`
/// reads the fixture (column-major via `Array2`).
fn sweep(law: &CostLaw, target: f64) -> ndarray::Array2<f64> {
    let n = 65;
    let mut out = ndarray::Array2::<f64>::zeros((n, 2));
    for k in 0..=64usize {
        let output = k as f64 / 64.0;
        out[[k, 0]] = output;
        out[[k, 1]] = law.compute_unitary_delta(output, target);
    }
    out
}

#[test]
fn scalar_delta_polynomial_bit_exact() {
    let law = poly_law();
    let got_speech = sweep(&law, 1.0);
    let got_other = sweep(&law, 0.0);
    let want_speech = common::load_bin_phase3("cost_deriv_scalar_speech.bin");
    let want_other = common::load_bin_phase3("cost_deriv_scalar_other.bin");
    common::assert_bits_eq(&got_speech, &want_speech, "poly scalar delta (speech)");
    common::assert_bits_eq(&got_other, &want_other, "poly scalar delta (other)");

    // Also pin against the explicitly-named poly dumps (same values, alias names).
    let want_poly_speech = common::load_bin_phase3("cost_deriv_scalar_poly_speech.bin");
    let want_poly_other = common::load_bin_phase3("cost_deriv_scalar_poly_other.bin");
    common::assert_bits_eq(
        &got_speech,
        &want_poly_speech,
        "poly scalar delta (speech, alias)",
    );
    common::assert_bits_eq(
        &got_other,
        &want_poly_other,
        "poly scalar delta (other, alias)",
    );
}

#[test]
fn scalar_delta_log_sqrt_canary() {
    let log = log_law();
    let got_log_speech = sweep(&log, 1.0);
    let got_log_other = sweep(&log, 0.0);
    let want_log_speech = common::load_bin_phase3("cost_deriv_scalar_log_speech.bin");
    let want_log_other = common::load_bin_phase3("cost_deriv_scalar_log_other.bin");
    common::assert_oracle_eq(
        &got_log_speech,
        &want_log_speech,
        "log scalar delta (speech)",
    );
    common::assert_oracle_eq(&got_log_other, &want_log_other, "log scalar delta (other)");

    let sqrt = sqrt_law();
    let got_sqrt_speech = sweep(&sqrt, 1.0);
    let got_sqrt_other = sweep(&sqrt, 0.0);
    let want_sqrt_speech = common::load_bin_phase3("cost_deriv_scalar_sqrt_speech.bin");
    let want_sqrt_other = common::load_bin_phase3("cost_deriv_scalar_sqrt_other.bin");
    common::assert_oracle_eq(
        &got_sqrt_speech,
        &want_sqrt_speech,
        "sqrt scalar delta (speech)",
    );
    common::assert_oracle_eq(
        &got_sqrt_other,
        &want_sqrt_other,
        "sqrt scalar delta (other)",
    );
}

/// The multiclass fusion case fixture: 4 frames x 3 classes, row1 fully
/// ignore-masked, on-class placed at a different column per other row. Must
/// match the harness's `outputs`/`targets` construction exactly.
fn multiclass_outputs() -> Vec<f64> {
    #[rustfmt::skip]
    let ov: [[f64; 3]; 4] = [
        [0.7, 0.2, 0.1],
        [0.5, 0.3, 0.2],
        [0.6, 0.1, 0.3],
        [0.25, 0.35, 0.4],
    ];
    ov.iter().flatten().copied().collect()
}

fn multiclass_targets() -> Vec<f64> {
    #[rustfmt::skip]
    let tv: [[f64; 3]; 4] = [
        [0.0, 1.0, 0.0],
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    tv.iter().flatten().copied().collect()
}

fn deltas_to_array(flat: &[f64], n_frames: usize, n_classes: usize) -> ndarray::Array2<f64> {
    // Row-major flat -> Array2, then compare directly (load_bin_phase3 also
    // yields row x col shaped Array2, values addressed by (row, col) either way).
    ndarray::Array2::from_shape_vec((n_frames, n_classes), flat.to_vec()).unwrap()
}

#[test]
fn multiclass_deltas_fusion() {
    let outputs = multiclass_outputs();
    let targets = multiclass_targets();
    let law = CostLaw::from_config(&base_map(), "BLSTM");
    let mut out = vec![0.0; outputs.len()];
    law.compute_deltas(&outputs, &targets, 3, &mut out);
    let got = deltas_to_array(&out, 4, 3);
    let want = common::load_bin_phase3("cost_deltas_multiclass.bin");
    common::assert_bits_eq(&got, &want, "multiclass fusion deltas");
}

#[test]
fn multiclass_deltas_wer_and_pond() {
    // Ponderation variant of the multiclass (non-WER) fusion.
    let outputs = multiclass_outputs();
    let targets = multiclass_targets();
    let mut pond_map = base_map();
    pond_map.insert("BLSTM_classes_ponderations".into(), "2.0,3.0,4.0".into());
    let pond_law = CostLaw::from_config(&pond_map, "BLSTM");
    let mut pond_out = vec![0.0; outputs.len()];
    pond_law.compute_deltas(&outputs, &targets, 3, &mut pond_out);
    let got_pond = deltas_to_array(&pond_out, 4, 3);
    let want_pond = common::load_bin_phase3("cost_deltas_multiclass_pond.bin");
    common::assert_bits_eq(
        &got_pond,
        &want_pond,
        "multiclass fusion deltas (ponderation)",
    );

    // WER-scaled multiclass path: soft targets (see harness comment for why:
    // the WER scaling `10*(1-target)`/`10*target` vanishes at exact 1.0/0.0).
    #[rustfmt::skip]
    let wer_outputs: Vec<f64> = [
        [0.7, 0.2, 0.1],
        [0.5, 0.3, 0.2],
        [0.25, 0.35, 0.4],
    ].iter().flatten().copied().collect();
    #[rustfmt::skip]
    let wer_targets: Vec<f64> = [
        [0.05, 0.9, 0.05],
        [0.8, 0.1, 0.1],
        [-1.0, -1.0, -1.0],
    ].iter().flatten().copied().collect();

    let mut wer_map = base_map();
    wer_map.insert("BLSTM_BackPropWER".into(), "0.0".into());
    let wer_law = CostLaw::from_config(&wer_map, "BLSTM");
    let mut wer_out = vec![0.0; wer_outputs.len()];
    wer_law.compute_deltas(&wer_outputs, &wer_targets, 3, &mut wer_out);
    let got_wer = deltas_to_array(&wer_out, 3, 3);
    let want_wer = common::load_bin_phase3("cost_deltas_wer.bin");
    common::assert_bits_eq(&got_wer, &want_wer, "WER-scaled multiclass deltas");

    let mut wer_pond_map = base_map();
    wer_pond_map.insert("BLSTM_BackPropWER".into(), "0.0".into());
    wer_pond_map.insert("BLSTM_classes_ponderations".into(), "2.0,3.0,4.0".into());
    let wer_pond_law = CostLaw::from_config(&wer_pond_map, "BLSTM");
    let mut wer_pond_out = vec![0.0; wer_outputs.len()];
    wer_pond_law.compute_deltas(&wer_outputs, &wer_targets, 3, &mut wer_pond_out);
    let got_wer_pond = deltas_to_array(&wer_pond_out, 3, 3);
    let want_wer_pond = common::load_bin_phase3("cost_deltas_wer_pond.bin");
    common::assert_bits_eq(
        &got_wer_pond,
        &want_wer_pond,
        "WER-scaled multiclass deltas (ponderation)",
    );
}

/// Structural (risk R8): the fully ignore-masked row (target < 0 everywhere)
/// must produce EXACTLY 0.0 (bit) deltas in every fusion variant, proving the
/// ignore-mask branch fires independent of WER/ponderation state.
#[test]
fn ignore_mask_zeroes_row() {
    let outputs = multiclass_outputs();
    let targets = multiclass_targets();
    let law = CostLaw::from_config(&base_map(), "BLSTM");
    let mut out = vec![0.0; outputs.len()];
    law.compute_deltas(&outputs, &targets, 3, &mut out);
    // Row 1 (0-indexed) is the fully-masked row (target < 0 everywhere).
    let masked_row = 1;
    for k in 0..3 {
        let v = out[masked_row * 3 + k];
        assert_eq!(v.to_bits(), 0.0_f64.to_bits(), "masked row col {k}: {v}");
    }

    // Also true under the WER path (row 2 of the WER fixture is fully masked).
    #[rustfmt::skip]
    let wer_outputs: Vec<f64> = [
        [0.7, 0.2, 0.1],
        [0.5, 0.3, 0.2],
        [0.25, 0.35, 0.4],
    ].iter().flatten().copied().collect();
    #[rustfmt::skip]
    let wer_targets: Vec<f64> = [
        [0.05, 0.9, 0.05],
        [0.8, 0.1, 0.1],
        [-1.0, -1.0, -1.0],
    ].iter().flatten().copied().collect();
    let mut wer_map = base_map();
    wer_map.insert("BLSTM_BackPropWER".into(), "0.0".into());
    let wer_law = CostLaw::from_config(&wer_map, "BLSTM");
    let mut wer_out = vec![0.0; wer_outputs.len()];
    wer_law.compute_deltas(&wer_outputs, &wer_targets, 3, &mut wer_out);
    for k in 0..3 {
        let v = wer_out[2 * 3 + k];
        assert_eq!(
            v.to_bits(),
            0.0_f64.to_bits(),
            "WER masked row col {k}: {v}"
        );
    }
}

/// Structural (risk R4): a config with `classes_ponderations` set produces a
/// row scaled by the on-class ponderation vs the unscaled (no-pond) row, by
/// EXACTLY the ponderation factor for that on-class column.
#[test]
fn ponderation_plumbing() {
    let outputs = multiclass_outputs();
    let targets = multiclass_targets();

    let plain_law = CostLaw::from_config(&base_map(), "BLSTM");
    let mut plain_out = vec![0.0; outputs.len()];
    plain_law.compute_deltas(&outputs, &targets, 3, &mut plain_out);

    let mut pond_map = base_map();
    pond_map.insert("BLSTM_classes_ponderations".into(), "2.0,3.0,4.0".into());
    let pond_law = CostLaw::from_config(&pond_map, "BLSTM");
    let mut pond_out = vec![0.0; outputs.len()];
    pond_law.compute_deltas(&outputs, &targets, 3, &mut pond_out);

    // Row0 on-class is col1 -> ponderation 3.0; row2 on-class col0 -> 2.0;
    // row3 on-class col2 -> 4.0 (matches the harness's `classes_ponderations`
    // "2.0,3.0,4.0" and the fixture's on-class placement).
    let pond_by_row = [
        3.0, 1.0, /* masked row: 0*anything == 0, factor irrelevant */
        2.0, 4.0,
    ];
    for (row, &factor) in pond_by_row.iter().enumerate() {
        if row == 1 {
            continue; // masked row: both are exactly 0.0, covered by ignore_mask_zeroes_row.
        }
        for k in 0..3 {
            let idx = row * 3 + k;
            let expected = plain_out[idx] * factor;
            assert_eq!(
                pond_out[idx].to_bits(),
                expected.to_bits(),
                "row {row} col {k}: pond={} plain*factor={} (factor={factor})",
                pond_out[idx],
                expected
            );
        }
    }
    // Non-vacuity: the ponderated law must actually DIFFER from the plain law
    // somewhere (proves the ponderation enters the gradient at all, risk R4).
    assert_ne!(plain_out, pond_out, "ponderation must change the deltas");
}
