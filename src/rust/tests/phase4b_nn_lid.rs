//! Phase 4b Task 3: `BlstmNetwork` LID additions -- `ponderate_weights_derivatives`,
//! `get_cost_ponderation`, and the scoring `feed_forward` overload verification.
//!
//! Ported from `BLSTMNeuralNetwork.cpp:289-297` (ponderateWeightsDerivatives),
//! `:333-350` (getCostPonderation), `:843-929` (the returning scoring feedForward
//! overload, `BLSTMNeuralNetwork.h:195`). The scoring overload's BINARY path is already
//! pinned by Phase 2 Task 10 (`phase2_blstm_golden.rs`); this file adds the MULTICLASS
//! (LID) path golden `scoring_multi_out.bin`, dumped from the harness reimpl and pinned
//! `max_ulp=0` vs the real compiled overload (`NN_TOL site=blstm_scoring_multi_*`).
//!
//! Comparator discipline: ponderation is a pure `*=` (no libm) -> STRICT bits; the
//! cost-ponderation clamps are pure comparisons/arithmetic -> STRICT; the scoring
//! output traverses the LSTM/output activation chain (asinh/softmax) -> canary-gated
//! (`assert_oracle_eq`).

mod common;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

// === Deterministic harness-matching generators ===============================

/// Harness synthetic flat-vector body: `w[k] = ((k*11+3) % 97)/97.0 - 0.5`, with the
/// last `2*input_size` entries a NONZERO mean/std tail (mean[j] = 0.1*(j+1), std[j] =
/// 1.0 + 0.05*j) -- EXACTLY as the harness Task 8/10 blocks build it.
fn synth_flat_tail(nb: usize, input_size: usize) -> Vec<f64> {
    let tail = 2 * input_size;
    let mut flat = vec![0.0_f64; nb];
    for (k, v) in flat.iter_mut().enumerate().take(nb - tail) {
        *v = ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5;
    }
    for j in 0..input_size {
        flat[nb - tail + j] = 0.1 * (j as f64 + 1.0);
        flat[nb - tail + input_size + j] = 1.0 + 0.05 * j as f64;
    }
    flat
}

/// The harness Task 8/10 synthetic input: `x[t,j] = ((t*29 + j*13 + 5) % 97)/97 - 0.5`,
/// T=12, 3 cols.
fn synth_input() -> Array2<f64> {
    Array2::from_shape_fn((12, 3), |(t, j)| {
        ((t * 29 + j * 13 + 5) % 97) as f64 / 97.0 - 0.5
    })
}

// ============================================================================
// ponderate_weights_derivatives (BLSTMNeuralNetwork.cpp:289-297)
// ============================================================================

/// A plain synthetic BLSTM (LSTM [3,4,2] sub [2,1], output [4,5,3] sub [1,1]) with
/// backprop active + no output sub-sampling (`output_network.sub_sampling_ratio() == 1`
/// so the fwd/bwd col0 is NOT `out_ratio`-scaled in `get_weights_derivatives` -- keeps
/// the ponderation comparison a clean `col0 * factor` with no associativity hazard).
fn ponderation_net() -> BlstmNetwork {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("SYNC_LSTMNeuronNb".into(), "3,4,2".into());
    m.insert("SYNC_LSTMSubSampling".into(), "2,1".into());
    m.insert("SYNC_OutputNeuronNb".into(), "4,5,3".into());
    m.insert("SYNC_OutputSubSampling".into(), "1,1".into());
    m.insert("SYNC_InputNormalizationType".into(), "0".into());
    m.insert("SYNC_TwoSweeps".into(), "false".into());
    m.insert("SYNC_BackPropagationActivated".into(), "true".into());
    m.insert("SYNC_BackPropOutputNetworkOnly".into(), "false".into());
    m.insert("SYNC_TargetEnforcementStep".into(), "0".into());
    let cfg = BlstmConfig::from_legacy(&m, "SYNC").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    let nb = net.nb_of_weights();
    net.set_weights(&synth_flat_tail(nb, 3)).unwrap();
    net
}

/// Seed the deriv accumulators with a real backward pass (multiclass one-hot targets).
fn seed_derivs(net: &mut BlstmNetwork) {
    net.reset_weights_derivatives();
    let mut input = synth_input();
    let out_len = 12 / 2; // fwd LSTM ratio 2*1 = 2 -> 6 rows
    let mut output = Array2::<f64>::zeros((out_len, 3));
    let targets = Array2::from_shape_fn((out_len, 3), |(r, c)| if c == r % 3 { 1.0 } else { 0.0 });
    net.set_processing_type(false, false);
    net.feed_forward_backward(&mut input, 0, 0, &mut output, &targets);
}

#[test]
fn ponderation_scales_col0_not_col1() {
    // Source (BLSTMNeuralNetwork.cpp:289-297 -> NeuralNetwork.hpp:119-123 -> the leaf
    // `_...Derivatives *= ponderation`, LSTMLayer.cpp:305-310 / NeuronLayer.cpp:122-125)
    // scales the derivative accumulators (col0) IN PLACE; `_NbOfSeqFedBackward` (col1) is
    // NEVER touched. Seed derivs, harvest D0, ponderate by a POSITIVE factor (so `0.0 *
    // factor` stays +0.0 in the [0,1] stats tail -- a negative factor would flip it to
    // -0.0 and break the strict-bit tail compare), harvest D1, assert col0*factor == col0
    // and col1 unchanged, BIT-EXACT.
    let factor = 3.0_f64;
    let mut net = ponderation_net();
    seed_derivs(&mut net);

    let d0 = net.get_weights_derivatives();
    net.ponderate_weights_derivatives(factor);
    let d1 = net.get_weights_derivatives();

    assert_eq!(
        d0.dim(),
        d1.dim(),
        "ponderation must not resize the Nx2 harvest"
    );
    let n = d0.nrows();
    assert!(n > 0, "harvest must be non-empty");

    // Non-vacuity: at least one trained-region col0 is nonzero (so `*factor` is a real
    // change, not `0*factor`).
    let any_nonzero = (0..n).any(|k| d0[[k, 0]] != 0.0);
    assert!(
        any_nonzero,
        "seeded derivatives must have a nonzero col0 somewhere"
    );

    for k in 0..n {
        let expected = d0[[k, 0]] * factor;
        assert_eq!(
            d1[[k, 0]].to_bits(),
            expected.to_bits(),
            "row {k}: col0 must be scaled by {factor} bit-exactly",
        );
        assert_eq!(
            d1[[k, 1]].to_bits(),
            d0[[k, 1]].to_bits(),
            "row {k}: col1 (frame count) must be UNCHANGED",
        );
    }
}

#[test]
fn ponderation_factor_one_is_identity() {
    // `factor == 1.0`: `deriv *= 1.0` is a bit-exact no-op on every finite value; the
    // whole Nx2 harvest is unchanged.
    let mut net = ponderation_net();
    seed_derivs(&mut net);
    let d0 = net.get_weights_derivatives();
    net.ponderate_weights_derivatives(1.0);
    let d1 = net.get_weights_derivatives();
    for k in 0..d0.nrows() {
        assert_eq!(d0[[k, 0]].to_bits(), d1[[k, 0]].to_bits(), "row {k} col0");
        assert_eq!(d0[[k, 1]].to_bits(), d1[[k, 1]].to_bits(), "row {k} col1");
    }
}

// ============================================================================
// get_cost_ponderation (BLSTMNeuralNetwork.cpp:333-350)
// ============================================================================

/// Build a net (no weights needed -- `get_cost_ponderation` reads only the CostLaw +
/// output size) from a config map.
fn net_from(map: &IndexMap<String, String>, prefix: &str) -> BlstmNetwork {
    let cfg = BlstmConfig::from_legacy(map, prefix).unwrap();
    BlstmNetwork::from_config(cfg).unwrap()
}

fn multiclass_map(prefix: &str, ponderations: Option<&str>) -> IndexMap<String, String> {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert(format!("{prefix}_LSTMNeuronNb"), "3,4,2".into());
    m.insert(format!("{prefix}_LSTMSubSampling"), "2,1".into());
    m.insert(format!("{prefix}_OutputNeuronNb"), "4,5,3".into()); // output_size = 3
    m.insert(format!("{prefix}_OutputSubSampling"), "1,1".into());
    m.insert(format!("{prefix}_InputNormalizationType"), "0".into());
    m.insert(format!("{prefix}_TwoSweeps"), "false".into());
    m.insert(format!("{prefix}_BackPropagationActivated"), "false".into());
    if let Some(p) = ponderations {
        m.insert(format!("{prefix}_classes_ponderations"), p.to_string());
    }
    m
}

fn binary_map(prefix: &str, cost_ponderation: &str) -> IndexMap<String, String> {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert(format!("{prefix}_LSTMNeuronNb"), "3,4,2".into());
    m.insert(format!("{prefix}_LSTMSubSampling"), "2,1".into());
    m.insert(format!("{prefix}_OutputNeuronNb"), "4,5,1".into()); // output_size = 1
    m.insert(format!("{prefix}_OutputSubSampling"), "1,1".into());
    m.insert(format!("{prefix}_InputNormalizationType"), "0".into());
    m.insert(format!("{prefix}_TwoSweeps"), "false".into());
    m.insert(format!("{prefix}_BackPropagationActivated"), "false".into());
    m.insert(
        format!("{prefix}_CostPonderation"),
        cost_ponderation.to_string(),
    );
    m
}

#[test]
fn cost_ponderation_clamps() {
    // Table-driven vs BLSTMNeuralNetwork.cpp:333-350.
    //
    // MULTICLASS (output_size 3 > 1, :334-341): in-range -> classes_ponderations[idx];
    // out-of-range/negative -> folds to 0 (:335) -> classes_ponderations[0]; an index in
    // [0, output_size) but >= ponderations.len() -> 1.0 (:340). Empty ponderations ->
    // always 1.0.
    let mc = net_from(&multiclass_map("MP", Some("0.2,0.3,0.5")), "MP");
    assert_eq!(mc.get_cost_ponderation(0), 0.2, "in-range class 0");
    assert_eq!(mc.get_cost_ponderation(1), 0.3, "in-range class 1");
    assert_eq!(mc.get_cost_ponderation(2), 0.5, "in-range class 2");
    assert_eq!(
        mc.get_cost_ponderation(3),
        0.2,
        "class == output_size -> fold to 0"
    );
    assert_eq!(
        mc.get_cost_ponderation(99),
        0.2,
        "class >> output_size -> fold to 0"
    );
    assert_eq!(mc.get_cost_ponderation(-1), 0.2, "negative -> fold to 0");

    // Fewer ponderations than classes: index 2 is < output_size (not clamped) but >=
    // ponderations.len() (2) -> 1.0 (:340).
    let mc_short = net_from(&multiclass_map("MS", Some("0.7,0.9")), "MS");
    assert_eq!(mc_short.get_cost_ponderation(0), 0.7);
    assert_eq!(mc_short.get_cost_ponderation(1), 0.9);
    assert_eq!(
        mc_short.get_cost_ponderation(2),
        1.0,
        "in-range but past ponderations -> 1.0"
    );

    // No ponderations at all -> classes_ponderations empty -> always 1.0.
    let mc_none = net_from(&multiclass_map("MN", None), "MN");
    assert_eq!(mc_none.get_cost_ponderation(0), 1.0);
    assert_eq!(mc_none.get_cost_ponderation(2), 1.0);
    assert_eq!(mc_none.get_cost_ponderation(-5), 1.0);

    // BINARY (output_size 1, :342-348): the guard uses `>` (NOT `>=`). CostPonderation
    // 0.3 -> cp = clamp(0.3, 0.01, 0.99) = 0.3. class 1 -> 2*cp = 0.6; else 2*(1-cp) =
    // 1.4. class == output_size (1) is NOT clamped (1 > 1 is false) -> stays 1 -> 0.6.
    let bin = net_from(&binary_map("BP", "0.3"), "BP");
    assert_eq!(
        bin.get_cost_ponderation(1),
        2.0 * 0.3,
        "binary class 1 -> 2*cp"
    );
    assert_eq!(
        bin.get_cost_ponderation(0),
        2.0 * (1.0 - 0.3),
        "binary class 0 -> 2*(1-cp)"
    );
    assert_eq!(
        bin.get_cost_ponderation(2),
        2.0 * (1.0 - 0.3),
        "binary class 2 (> 1) -> fold to 0 -> 2*(1-cp)"
    );
    assert_eq!(
        bin.get_cost_ponderation(-1),
        2.0 * (1.0 - 0.3),
        "binary negative -> fold to 0 -> 2*(1-cp)"
    );

    // CostPonderation clamp: value below 0.01 clamps to 0.01.
    let bin_lo = net_from(&binary_map("BL", "0.0"), "BL");
    assert_eq!(
        bin_lo.get_cost_ponderation(1),
        2.0 * 0.01,
        "cp clamped up to 0.01"
    );
}

// ============================================================================
// scoring feed_forward overload (BLSTMNeuralNetwork.cpp:843-929) -- MULTICLASS
// ============================================================================

/// Reconstruct the harness SYNC multiclass scoring net: LSTM [3,4,2] sub [2,1], output
/// [4,5,3] sub [1,1], norm 0, backprop off, enforcement `step`. Weights = the harness
/// synthetic flat (body + mean/std tail).
fn scoring_multi_net(step: i32) -> BlstmNetwork {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("SYNC_LSTMNeuronNb".into(), "3,4,2".into());
    m.insert("SYNC_LSTMSubSampling".into(), "2,1".into());
    m.insert("SYNC_OutputNeuronNb".into(), "4,5,3".into());
    m.insert("SYNC_OutputSubSampling".into(), "1,1".into());
    m.insert("SYNC_InputNormalizationType".into(), "0".into());
    m.insert("SYNC_TwoSweeps".into(), "false".into());
    m.insert("SYNC_BackPropagationActivated".into(), "false".into());
    m.insert("SYNC_TargetEnforcementStep".into(), step.to_string());
    let cfg = BlstmConfig::from_legacy(&m, "SYNC").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    let nb = net.nb_of_weights();
    net.set_weights(&synth_flat_tail(nb, 3)).unwrap();
    net.set_processing_type(false, false);
    net
}

// The harness multiclass scoring cases: (tag, target_index, step). The forward output is
// TARGET-INDEPENDENT (targets feed only cost/backward, both off here + step>=0 leaves the
// output un-rewritten), so ONE golden covers every case.
const SCORING_MULTI_CASES: [(&str, i64, i32); 4] = [
    ("ti0_step0", 0, 0),
    ("ti2_step0", 2, 0),
    ("ti2_step2", 2, 2),
    ("tiOOB_step0", 5, 0), // 5 >= 3 -> unknown-class fold -> 0
];

#[test]
fn scoring_overload_matches_reimpl() {
    // The scoring feed_forward on the SYNC multiclass net: builds a per-frame target
    // sequence from (target_index, modifier=1.0), runs the windowed FFB (plain path),
    // and -- because output_size > 1 -- returns the RAW length x output_size matrix (NO
    // binary [1-p,p] expansion; that gate is output_size == 1). The output is
    // target-independent here, so every case matches the single golden
    // `scoring_multi_out.bin` (harness reimpl, pinned max_ulp=0 vs the real overload).
    for (tag, target_index, step) in SCORING_MULTI_CASES {
        let mut net = scoring_multi_net(step);
        let mut input = synth_input();
        let out = net.feed_forward_scoring(&mut input, 4, 2, target_index, 1.0);
        assert_eq!(
            out.dim(),
            (6, 3),
            "{tag}: multiclass -> raw (length, output_size), NOT expanded"
        );
        common::assert_oracle_eq(
            &out,
            &common::load_bin_phase4b("scoring_multi_out.bin"),
            &format!("scoring_multi {tag} out"),
        );
    }
}

#[test]
fn scoring_multi_oob_index_folds_to_zero() {
    // target_index >= output_size -> the unknown-class fold sets it to 0 (:873). With
    // step 0 (every row enforced) the OOB case must accumulate the SAME cost as
    // target_index 0 -- validating the fold via the cost channel (the only observable
    // effect of the multiclass target construction), no fixture needed.
    let mut net0 = scoring_multi_net(0);
    let mut in0 = synth_input();
    let _ = net0.feed_forward_scoring(&mut in0, 4, 2, 0, 1.0);
    let cost0 = net0.cost;

    let mut net_oob = scoring_multi_net(0);
    let mut in_oob = synth_input();
    let _ = net_oob.feed_forward_scoring(&mut in_oob, 4, 2, 5, 1.0);
    assert_eq!(
        net_oob.cost.to_bits(),
        cost0.to_bits(),
        "OOB target_index 5 folds to 0 -> identical cost to target_index 0",
    );
    assert_eq!(
        net0.nb_of_classif, net_oob.nb_of_classif,
        "nb_of_classif identical"
    );
}

#[test]
fn scoring_multi_target_index_and_step_change_cost() {
    // The multiclass target construction (:872-886) is real: a different enforced column
    // (target_index) OR a different enforcement cadence (step) yields a different CE cost.
    // Fixture-free structural validation of the counter + column placement.
    let cost = |ti: i64, step: i32| {
        let mut net = scoring_multi_net(step);
        let mut input = synth_input();
        let _ = net.feed_forward_scoring(&mut input, 4, 2, ti, 1.0);
        net.cost
    };
    assert_ne!(
        cost(0, 0),
        cost(2, 0),
        "distinct target_index -> distinct cost"
    );
    assert_ne!(
        cost(2, 0),
        cost(2, 2),
        "distinct enforcement step -> distinct cost"
    );
}

#[test]
fn scoring_multi_rows_below_ratio_returns_zero_row() {
    // rows < sub_sampling_ratio() short-circuits to Zero(1, output_size()) (:845-846).
    // The SYNC net's ratio is lstm(2)*out(1) = 2; feed 1 row -> Zero(1, 3).
    let mut net = scoring_multi_net(0);
    let mut input = Array2::<f64>::from_shape_fn((1, 3), |(_, j)| j as f64);
    let out = net.feed_forward_scoring(&mut input, 4, 2, 0, 1.0);
    assert_eq!(
        out.dim(),
        (1, 3),
        "short-circuit shape Zero(1, output_size)"
    );
    assert!(out.iter().all(|&v| v == 0.0), "short-circuit is all zeros");
}

#[test]
fn scoring_multi_no_target_returns_raw_matrix() {
    // target_index < 0 -> no target sequence, and (output_size > 1) the raw multiclass
    // matrix returns unchanged (length x output_size).
    let mut net = scoring_multi_net(0);
    let mut input = synth_input();
    let out = net.feed_forward_scoring(&mut input, 4, 2, -1, 1.0);
    assert_eq!(
        out.dim(),
        (6, 3),
        "no-target multiclass -> raw (length, output_size)"
    );
    // Bit-identical to the with-target output (forward is target-independent).
    common::assert_oracle_eq(
        &out,
        &common::load_bin_phase4b("scoring_multi_out.bin"),
        "scoring_multi no-target out",
    );
}
