//! Phase 3 Task 8: the real-net E2E GRADIENT GATE (the phase acceptance crux).
//!
//! The real 33,671-weight net (`1_worker_1.config` + `NNweights_config1.bin`,
//! `BackPropagationActivated` OVERRIDDEN true -- the config ships it false,
//! `InputNormalizationType -1`) runs the real assembled `e2e_input.bin` (201 x 11,
//! width 11 < 23 -> the LSTM `topRows` tolerance) with deterministic scalar-VAD
//! targets (`(r%3==0)?1.0:0.0`) through ONE `feed_forward_backward`. Mirrors the
//! harness `blstm_real_backward` stage (`tools/oracle_harness/main.cpp`) exactly.
//!
//! Asserts (spec S8/S11.2, plan Task 8):
//!   - `real_net_gradient_bit_exact`: the Nx2 gradient col0 vs
//!     `bwd_blstm_real_derivs.bin` -- CANARY (the k=23/24/48 GEMM diverges from
//!     ascending accumulation, so the reimpl dump IS the golden); col1 (the
//!     replicated frame count) strict-int.
//!   - `real_net_one_irprop_step`: one `update_weights` (cost 1.0) -> weights vs
//!     `e2e_weights_after_step.bin`. The Rprop step is pure scalar (bit-portable),
//!     but its INPUT gradient is the canary-gated reimpl, so the whole assert is
//!     canary-gated (documented choice, plan Task 8).
//!   - `real_net_backtrack_fires`: a SECOND `update_weights` with the NEGATED
//!     gradient (col0 flipped) under a RISEN cost (2.0 > 1.0) -> weights vs
//!     `e2e_weights_after_backtrack.bin` (canary) + a STRUCTURAL assertion that the
//!     backtrack branch fired (weights moved back by exactly the step-1 dw on the
//!     fired elements, from `e2e_step1_dw.bin`).
//!   - `gradient_non_trivial_real` (S11.2): the Nx2 col0 is non-zero, non-saturated,
//!     no NaN/Inf.

mod common;

use std::path::PathBuf;

use ndarray::Array2;
use speech::legacy_config::parse_legacy_config;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

fn real_config_map() -> indexmap::IndexMap<String, String> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/1_worker_1.config");
    let text = std::fs::read_to_string(p).unwrap();
    let mut map = parse_legacy_config(&text);
    // Override BackPropagationActivated -> true (the config ships it false + no
    // RpropInit key -> default 1e-2), the established Phase-2/3 harness pattern
    // (main.cpp: conf.set_val<bool>("BLSTM_BackPropagationActivated", true)).
    map.insert("BLSTM_BackPropagationActivated".into(), "true".into());
    map
}

fn real_weights() -> Vec<f64> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/NNweights_config1.bin");
    speech::io::binary::read_weight_vector(&p).unwrap()
}

/// The real net with backprop active, weights loaded, plain processing type -- the
/// exact harness setup (`main.cpp` blstm_real_backward stage).
fn real_net() -> BlstmNetwork {
    let cfg = BlstmConfig::from_legacy(&real_config_map(), "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.set_weights(&real_weights()).unwrap();
    net.set_processing_type(false, false);
    net
}

/// The committed e2e input (201 x 11, pre-normalization -- the type -1 self-norm
/// mutates it in place inside FFB, so callers pass a mutable clone).
fn e2e_input() -> Array2<f64> {
    common::load_bin_phase2("e2e_input.bin")
}

/// Deterministic scalar-VAD targets over the decimated frames (harness: outLen =
/// rows/4, target = (r%3==0)?1.0:0.0).
fn e2e_targets(input_rows: usize) -> Array2<f64> {
    let out_len = input_rows / 4; // sub_sampling_ratio 4
    Array2::from_shape_fn((out_len, 1), |(r, _)| if r % 3 == 0 { 1.0 } else { 0.0 })
}

/// Run the real net's one plain feed_forward_backward and return the Nx2 gradient.
fn run_backward(net: &mut BlstmNetwork) -> Array2<f64> {
    let input = e2e_input();
    let targets = e2e_targets(input.nrows());
    let out_len = targets.nrows();
    let mut output = Array2::<f64>::zeros((out_len, 1));
    let mut mut_input = input.clone();
    net.reset_weights_derivatives();
    net.feed_forward_backward(&mut mut_input, 0, 0, &mut output, &targets);
    net.get_weights_derivatives()
}

#[test]
fn real_net_gradient_bit_exact() {
    let mut net = real_net();
    let derivs = run_backward(&mut net);
    let golden = common::load_bin_phase3("bwd_blstm_real_derivs.bin");
    assert_eq!(derivs.dim(), (33671, 2), "real-net Nx2 gradient shape");
    assert_eq!(golden.dim(), (33671, 2));

    // col0 (the summed derivative) -- CANARY: the reimpl dump IS the golden (k>=23
    // GEMM divergence). Compare the whole col0 as a 1-column matrix.
    let got_col0 = derivs.slice(ndarray::s![.., 0..1]).to_owned();
    let want_col0 = golden.slice(ndarray::s![.., 0..1]).to_owned();
    common::assert_oracle_eq(&got_col0, &want_col0, "e2e_grad_col0");

    // col1 (the replicated _NbOfSeqFedBackward frame count) -- structural integer,
    // strict on every platform (pure counting, no libm).
    for k in 0..33671 {
        assert_eq!(
            derivs[[k, 1]].to_bits(),
            golden[[k, 1]].to_bits(),
            "col1 count mismatch at {k}: {} vs {}",
            derivs[[k, 1]],
            golden[[k, 1]]
        );
    }
}

#[test]
fn gradient_non_trivial_real() {
    // S11.2: the real-net gradient col0 must be non-zero, non-saturated, no NaN/Inf.
    let mut net = real_net();
    let derivs = run_backward(&mut net);
    let col0: Vec<f64> = (0..derivs.nrows()).map(|k| derivs[[k, 0]]).collect();

    assert!(col0.iter().all(|v| v.is_finite()), "gradient has NaN/Inf");
    let nonzero = col0.iter().filter(|&&v| v != 0.0).count();
    assert!(nonzero > 0, "gradient is all zero");
    let min = col0.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = col0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(min < max, "gradient is constant (min == max): {min}");
    // Non-saturated: a real spread, not pinned at a single saturating value.
    assert!(
        max.abs() < 1e6 && min.abs() < 1e6,
        "gradient saturated: [{min}, {max}]"
    );
}

#[test]
fn real_net_one_irprop_step() {
    // One update_weights (cost 1.0, unused first call) on the real-net normalized
    // gradient -> weights vs e2e_weights_after_step.bin. Canary-gated: the Rprop
    // step is bit-portable, but the normalized gradient it consumes is the
    // canary-gated reimpl, so the resulting weights inherit that gate.
    let mut net = real_net();
    let derivs = run_backward(&mut net);
    net.update_weights(&derivs, 1.0);
    let got = Array2::from_shape_vec((33671, 1), net.get_weights()).unwrap();
    let want = common::load_bin_phase3("e2e_weights_after_step.bin");
    common::assert_oracle_eq(&got, &want, "e2e_weights_after_step");
}

#[test]
fn real_net_backtrack_fires() {
    // Step 1 then a SECOND step with the NEGATED gradient (col0 flipped, col1 count
    // preserved so col0/col1 == -norm) under a RISEN cost (2.0 > 1.0): every nonzero
    // element sign-flips (derivTimesPrev < 0) AND the cost gate opens, so the
    // cost-gated backtrack FIRES (weights[j] -= delta_weights[j], undoing step 1).
    let mut net = real_net();
    let derivs = run_backward(&mut net);

    net.update_weights(&derivs, 1.0);
    let w_step = net.get_weights();

    // Negate col0 for step 2 (Rust normalizes col0/col1 internally -> -norm).
    let mut neg = derivs.clone();
    for k in 0..neg.nrows() {
        neg[[k, 0]] = -neg[[k, 0]];
    }
    net.update_weights(&neg, 2.0);
    let w_back = net.get_weights();

    // Canary: the weights match the harness's REAL Rprop trajectory dump.
    let got = Array2::from_shape_vec((33671, 1), w_back.clone()).unwrap();
    let want = common::load_bin_phase3("e2e_weights_after_backtrack.bin");
    common::assert_oracle_eq(&got, &want, "e2e_weights_after_backtrack");

    // STRUCTURAL branch assertion (non-vacuity, S11.1): the backtrack branch fired.
    // The step-1 dw per element (e2e_step1_dw.bin: the <0 branch leaves _DeltasWeights
    // untouched, so it still holds step-1's dw after step 2). On every element with a
    // nonzero step-1 dw, the backtrack undid step 1: w_back == w_step - dw1 (bit-exact,
    // pure scalar Rprop -> strict). Assert at least one element fired AND every fired
    // element moved back by exactly -dw.
    let dw1 = common::load_bin_phase3("e2e_step1_dw.bin");
    assert_eq!(dw1.dim(), (33671, 1));
    let mut fired = 0usize;
    for k in 0..33671 {
        let d = dw1[[k, 0]];
        if d != 0.0 {
            // The backtrack branch fired for this element: undo the step-1 move.
            let expected = w_step[k] - d;
            assert_eq!(
                w_back[k].to_bits(),
                expected.to_bits(),
                "backtrack did not undo step 1 at {k}: w_back={} expected {}",
                w_back[k],
                expected
            );
            fired += 1;
        }
    }
    assert!(
        fired > 0,
        "the backtrack branch fired on 0 elements (vacuous)"
    );
    eprintln!("backtrack fired on {fired} / 33671 elements");
}
