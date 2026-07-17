//! Bit-exact cost-law tests vs the legacy test_costlaws.mat sweeps.
//!
//! The sweep fixtures are stored 1x1000 (Task 1), so we read via `read_matrix`
//! and take the flat data vector (orientation-agnostic for a vector); the
//! bit-exact `assert_eq!` on f64 is the arbiter over all 1000 rows x 2 regimes.

use std::path::PathBuf;

use indexmap::IndexMap;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0b")
}

fn read_sweep(name: &str) -> Vec<f64> {
    let (_rows, _cols, data) =
        speech::io::binary::read_matrix(&ref_dir().join(format!("{name}.bin"))).unwrap();
    data
}

fn sweep_config() -> IndexMap<String, String> {
    // Mirror tests/reference_data/phase0b/costlaw_config.json (linear, cp=0.2, thresh 1/0).
    let mut m = IndexMap::new();
    for (k, v) in [
        ("BLSTM_CostLawSpeech", "linear"),
        ("BLSTM_CostLawNoSpeech", "linear"),
        ("BLSTM_CostPonderation", "0.2"),
        ("BLSTM_CostLawParamSpeech", "0"),
        ("BLSTM_CostLawParamNoSpeech", "0"),
        ("BLSTM_CostLawThreshSpeech", "1"),
        ("BLSTM_CostLawThreshNoSpeech", "0"),
    ] {
        m.insert(k.to_string(), v.to_string());
    }
    m
}

#[test]
fn cost_sweep_bit_exact() {
    let law = speech::cost::CostLaw::from_config(&sweep_config(), "BLSTM");
    let output = read_sweep("sweep_output");
    let cost_speech = read_sweep("sweep_cost_speech");
    let cost_nospeech = read_sweep("sweep_cost_nospeech");
    assert_eq!(output.len(), 1000);
    for i in 0..output.len() {
        assert_eq!(
            law.compute_unitary_cost(output[i], 1.0),
            cost_speech[i],
            "speech cost row {i}"
        );
        assert_eq!(
            law.compute_unitary_cost(output[i], 0.0),
            cost_nospeech[i],
            "nospeech cost row {i}"
        );
    }
}

#[test]
fn deriv_sweep_bit_exact() {
    let law = speech::cost::CostLaw::from_config(&sweep_config(), "BLSTM");
    let output = read_sweep("sweep_output");
    let d_speech = read_sweep("sweep_deriv_speech");
    let d_nospeech = read_sweep("sweep_deriv_nospeech");
    assert_eq!(output.len(), 1000);
    for i in 0..output.len() {
        assert_eq!(
            law.compute_unitary_delta(output[i], 1.0),
            d_speech[i],
            "speech deriv row {i}"
        );
        assert_eq!(
            law.compute_unitary_delta(output[i], 0.0),
            d_nospeech[i],
            "nospeech deriv row {i}"
        );
    }
}

// --- Per-law + double-read unit tests (expected values computed by hand) ---

fn cfg(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn target_negative_is_ignored() {
    let law = speech::cost::CostLaw::from_config(&sweep_config(), "BLSTM");
    assert_eq!(law.compute_unitary_cost(0.3, -1.0), 0.0);
    assert_eq!(law.compute_unitary_delta(0.3, -1.0), 0.0);
}

#[test]
fn linear_speech_cost_and_deriv() {
    // Speech linear, P=cp=0.2, q=0, t=clamp(1)=1-1e-6, branch selector raw=1.0.
    // A = -0.2*(1-0)/(1-1e-6), B = 0.2. output=0.001 < 1.0 => below-thresh linear.
    // cost = 0.2 + A*0.001 ; deriv = A ; delta = A*0.001*(1-0.001).
    let law = speech::cost::CostLaw::from_config(&sweep_config(), "BLSTM");
    let t = 1.0 - 1e-6;
    let a = -0.2 * (1.0 - 0.0) / t;
    let expected_cost = 0.2 + a * 0.001;
    let expected_delta = a * 0.001 * (1.0 - 0.001);
    assert_eq!(law.compute_unitary_cost(0.001, 1.0), expected_cost);
    assert_eq!(law.compute_unitary_delta(0.001, 1.0), expected_delta);
    // Anchor from the brief.
    assert_eq!(law.compute_unitary_cost(0.001, 1.0), 0.19979999979999982);
}

#[test]
fn linear_nospeech_cost_and_deriv() {
    // No-speech linear: P=1-cp=0.8, q=0, t=1-clamp(0)=1-1e-6, branch selector raw=0.0.
    // output=0.4 > 0.0 => below-thresh, fed flipped y=1-0.4=0.6.
    // A = -0.8*(1-0)/(1-1e-6), B = 0.8. cost = 0.8 + A*0.6.
    // delta = -A ; then *output*(1-output) with output=1-0.4=0.6 => -A*0.6*0.4.
    let law = speech::cost::CostLaw::from_config(&sweep_config(), "BLSTM");
    let t = 1.0 - 1e-6;
    let a = -0.8 * (1.0 - 0.0) / t;
    let expected_cost = 0.8 + a * 0.6;
    let expected_delta = -a * 0.6 * (1.0 - 0.6);
    assert_eq!(law.compute_unitary_cost(0.4, 0.0), expected_cost);
    assert_eq!(law.compute_unitary_delta(0.4, 0.0), expected_delta);
}

#[test]
fn square_speech_cost_and_deriv() {
    // Speech square, P=0.5, q=0, t=clamp(0.5)=0.5, branch raw=0.5.
    // A = 0.5*(1-0)/0.5/0.5 = 2.0, B = -0.5*2*(1-0)/0.5 = -2.0, C = 0.5.
    // output=0.25 < 0.5 => below (square): cost = 0.5 -2*0.25 +2*0.25^2 = 0.125.
    // deriv = B + 2A*y = -2 + 4*0.25 = -1.0 ; delta = -1.0*0.25*0.75 = -0.1875.
    let law = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_CostLawSpeech", "square"),
            ("BLSTM_CostLawNoSpeech", "square"),
            ("BLSTM_CostPonderation", "0.5"),
            ("BLSTM_CostLawParamSpeech", "0"),
            ("BLSTM_CostLawThreshSpeech", "0.5"),
        ]),
        "BLSTM",
    );
    assert_eq!(law.compute_unitary_cost(0.25, 1.0), 0.125);
    assert_eq!(law.compute_unitary_delta(0.25, 1.0), -0.1875);
}

#[test]
fn cubic_speech_cost_and_deriv() {
    // Speech cubic, P=0.5, q=0, t=0.5, branch raw=0.5.
    // A = 0.5*2*(1-0)/0.5^3 = 8.0, B = -0.5*3*(1-0)/0.5^2 = -6.0, D = 0.5.
    // output=0.25 < 0.5 => below (cubic): cost = 0.5 -6*0.0625 +8*0.015625 = 0.25.
    // deriv = 2B*y + 3A*y^2 = -12*0.25 + 24*0.0625 = -1.5 ; delta = -1.5*0.25*0.75.
    let law = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_CostLawSpeech", "cubic"),
            ("BLSTM_CostLawNoSpeech", "cubic"),
            ("BLSTM_CostPonderation", "0.5"),
            ("BLSTM_CostLawParamSpeech", "0"),
            ("BLSTM_CostLawThreshSpeech", "0.5"),
        ]),
        "BLSTM",
    );
    assert_eq!(law.compute_unitary_cost(0.25, 1.0), 0.25);
    assert_eq!(law.compute_unitary_delta(0.25, 1.0), -1.5 * 0.25 * 0.75);
}

#[test]
fn log_speech_cost_and_deriv_asymmetry() {
    // Speech log, P=0.5, q=0, t=0.5, branch raw=0.5.
    // A = -0.5*(1-0) = -0.5, B = 0.5*0 = 0, Adim = 0.5.
    // output=0.25 < 0.5 => below (log). z = 0.25/0.5 = 0.5, INTERIOR (not clamped).
    // cost: y = 0.25/0.5 = 0.5 (in [1e-24,1]); cost = 0 + (-0.5)*ln(0.5).
    // deriv (F7, phase5): in the interior the true derivative is A/output (the Adim
    // cancels via the chain rule) = -0.5/0.25 = -2.0 -- UNCHANGED by the F7 fix, which
    // only zeros the SATURATED region (see log_deriv_consistent_with_clamped_forward).
    // delta = -2.0 * 0.25 * 0.75.
    let law = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_CostLawSpeech", "log"),
            ("BLSTM_CostLawNoSpeech", "log"),
            ("BLSTM_CostPonderation", "0.5"),
            ("BLSTM_CostLawParamSpeech", "0"),
            ("BLSTM_CostLawThreshSpeech", "0.5"),
        ]),
        "BLSTM",
    );
    let expected_cost = -0.5 * (0.5_f64).ln();
    assert_eq!(law.compute_unitary_cost(0.25, 1.0), expected_cost);
    // Interior true derivative A/output = -2.0 (NOT A/(y/Adim)=-1.0, which was never the
    // derivative of anything; NOT 0, which is the fixed SATURATED value only).
    assert_eq!(law.compute_unitary_delta(0.25, 1.0), -2.0 * 0.25 * 0.75);
}

/// F7 (phase5): the LogLaw derivative is now CONSISTENT with the clamped forward.
/// `cost()` computes `b + A*ln(clamp(output/Adim, 1e-24, 1))`; where that argument is
/// clamped the cost is CONSTANT, so the true gradient is 0. The legacy deriv returned
/// `A/clamp(output,1e-24,1)` UNCONDITIONALLY -- a nonzero (wrong) gradient in the
/// saturated region (IMPROVEMENTS [0b-i]).
///
/// This region is only reachable at output -> {0,1} on the committed k/64 and
/// 1000-point VAD grids, where the logistic fold `output*(1-output)` in
/// `compute_unitary_delta` zeros the contribution -- so NO existing golden moves
/// (`deriv_sweep_bit_exact`, `scalar_delta_log_sqrt_canary` stay bit-exact; verified).
/// This directed test samples the saturated region OFF that grid.
///
/// Port-truth (no C++ harness): the fixed values are re-derived here (0 in the clamped
/// region); the legacy-would-be value is computed inline to prove non-vacuity. The C++
/// harness stays legacy (it dumps `A/y`); it is intentionally NOT rebuilt, since the
/// committed fixtures never sample the divergence. REVERT-MUTATION: reverting the fix
/// (deriv `A/y` unconditional) makes the two `== 0.0` asserts fail (legacy is nonzero).
#[test]
fn log_deriv_consistent_with_clamped_forward() {
    // thresh 0.5 -> branch selector raw=0.5 AND Adim=clamp(0.5)=0.5, so output<0.5 stays
    // in the below-log branch; A = -cp*(1-q) = -0.5, q=0.
    let law = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_CostLawSpeech", "log"),
            ("BLSTM_CostLawNoSpeech", "log"),
            ("BLSTM_CostPonderation", "0.5"),
            ("BLSTM_CostLawParamSpeech", "0"),
            ("BLSTM_CostLawThreshSpeech", "0.5"),
        ]),
        "BLSTM",
    );
    let a = -0.5_f64;

    // (1) INTERIOR (z = output/Adim in (1e-24,1)): the fix preserves the true derivative
    // A/output (Adim cancels) -- nonzero, unchanged from the legacy.
    let interior = 0.25_f64; // z = 0.5
    assert_eq!(
        law.compute_unitary_delta(interior, 1.0),
        (a / interior) * interior * (1.0 - interior)
    );
    assert_ne!(law.compute_unitary_delta(interior, 1.0), 0.0);

    // (2) LOWER-CLAMP saturated (z = output/Adim < 1e-24): forward constant -> gradient 0.
    let lo = 1e-25_f64; // z = 2e-25 < 1e-24
    assert_eq!(law.compute_unitary_delta(lo, 1.0), 0.0);
    // Non-vacuity: the LEGACY formula A/clamp(y,1e-24,1) * fold is decidedly nonzero here.
    let legacy_lo = (a / 1e-24_f64) * lo * (1.0 - lo);
    assert!(
        legacy_lo.abs() > 1e-3,
        "legacy_lo={legacy_lo} must be a real gradient, not a sliver"
    );

    // (3) UPPER-CLAMP saturated via the double-read: branch raw=10 keeps the below-log law
    // live up to output=1, while Adim=clamp(10)=1-1e-6 clamps output>1-1e-6.
    let law_hi = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_CostLawSpeech", "log"),
            ("BLSTM_CostLawNoSpeech", "log"),
            ("BLSTM_CostPonderation", "0.5"),
            ("BLSTM_CostLawParamSpeech", "0"),
            ("BLSTM_CostLawThreshSpeech", "10"),
        ]),
        "BLSTM",
    );
    let hi = 1.0 - 5e-7_f64; // > 1-1e-6 => clamped; < 10 => below-log branch
    assert_eq!(law_hi.compute_unitary_delta(hi, 1.0), 0.0);
    let legacy_hi = (a / hi) * hi * (1.0 - hi);
    assert_ne!(
        legacy_hi, 0.0,
        "legacy_hi must be nonzero (a genuine divergence point)"
    );
}

#[test]
fn sqrt_speech_cost_and_deriv() {
    // Speech sqrt, P=0.5, q=0, t=0.5, branch raw=0.5.
    // Below-sqrt: A = 0.5*(1-0) = 0.5, B = 0.5*0 = 0, Adim = 0.5.
    // output=0.25 < 0.5 => below-sqrt.
    // cost: y = 0.25/0.5 = 0.5; cost = 0 + 0.5*sqrt(1-0.5) = 0.5*sqrt(0.5).
    // deriv: y = 1 - 0.25/0.5 = 0.5 (>=1e-24); deriv = -0.5/2/sqrt(0.5).
    // delta = deriv * 0.25 * 0.75.
    let law = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_CostLawSpeech", "sqrt"),
            ("BLSTM_CostLawNoSpeech", "sqrt"),
            ("BLSTM_CostPonderation", "0.5"),
            ("BLSTM_CostLawParamSpeech", "0"),
            ("BLSTM_CostLawThreshSpeech", "0.5"),
        ]),
        "BLSTM",
    );
    let expected_cost = 0.5 * (0.5_f64).sqrt();
    let deriv = -0.5 / 2.0 / (0.5_f64).sqrt();
    assert_eq!(law.compute_unitary_cost(0.25, 1.0), expected_cost);
    assert_eq!(law.compute_unitary_delta(0.25, 1.0), deriv * 0.25 * 0.75);
}

#[test]
fn above_thresh_cubic_name_routing() {
    // AboveThreshCubic uses name-dependent coefficients (legacy quirk). With q=0,
    // the linear/log branch: A = P*((1-q)/t/(1-t)^2), B = P*(-(1-q)/t/(1-t)).
    // Speech linear, P=0.5, q=0, t=0.5, branch raw=0.5. output=0.75 >= 0.5 => above.
    // A = 0.5*(1/0.5/0.25) = 4.0 ; B = 0.5*(-1/0.5/0.5) = -2.0.
    // cost: y=1-0.75=0.25; cost = B*y^2 + A*y^3 = -2*0.0625 + 4*0.015625 = -0.0625.
    // For "square" name the coeffs would be A=-2*P*q/... = 0 (q=0), giving cost 0.
    let lin = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_CostLawSpeech", "linear"),
            ("BLSTM_CostLawNoSpeech", "linear"),
            ("BLSTM_CostPonderation", "0.5"),
            ("BLSTM_CostLawParamSpeech", "0"),
            ("BLSTM_CostLawThreshSpeech", "0.5"),
        ]),
        "BLSTM",
    );
    assert_eq!(lin.compute_unitary_cost(0.75, 1.0), -0.0625);
    let sq = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_CostLawSpeech", "square"),
            ("BLSTM_CostLawNoSpeech", "square"),
            ("BLSTM_CostPonderation", "0.5"),
            ("BLSTM_CostLawParamSpeech", "0"),
            ("BLSTM_CostLawThreshSpeech", "0.5"),
        ]),
        "BLSTM",
    );
    assert_eq!(sq.compute_unitary_cost(0.75, 1.0), 0.0);
}

#[test]
fn double_read_diverges_when_key_absent() {
    // With ThreshSpeech absent: law coeffs use clamped t=1-1e-6, but the branch
    // selector re-reads the RAW default 10.0. output 0.5 < 10.0 => below-thresh log
    // law is used (it would be above-thresh if the branch used the clamped 1-1e-6).
    let m = cfg(&[("BLSTM_CostLawSpeech", "log")]);
    let law = speech::cost::CostLaw::from_config(&m, "BLSTM");
    let c = law.compute_unitary_cost(0.5, 1.0);
    assert!(c.is_finite());
    // Below-thresh log at output=0.5 with default cp=0.5, q=0, Adim=1-1e-6:
    // A=-0.5, B=0, y=0.5/(1-1e-6) clamped; cost = -0.5*ln(y).
    let adim = 1.0 - 1e-6;
    let mut y: f64 = 0.5 / adim;
    if y > 1.0 {
        y = 1.0;
    }
    assert_eq!(c, -0.5 * y.ln());
}
