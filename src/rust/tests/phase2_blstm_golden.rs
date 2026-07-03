//! Phase 2 Task 7/8: `BlstmConfig` + `BlstmNetwork` construction + flat weight seam
//! (Task 7); input normalization + core bidirectional forward + plain
//! `feed_forward_backward` (Task 8).
//!
//! Task 7 goldens run against the real `1_worker_1.config` (BLSTM prefix) and the
//! real `NNweights_config1.bin` (33,671 x 1) -- pure construction/plumbing.
//!
//! Task 8 goldens run against the harness Task 8 dumps: a synthetic net (LSTM
//! [3,4,2] sub [2,1], output [4,5,3] sub [1,1], T=12) forwarded through each
//! normalization type {1, -1, -2, 0} (`blstm_norm<tag>_{out,input_after}.bin`) and
//! the REAL net on a synthetic 200x23 input (`blstm_real_fullseq_{out,fwd,bwd}.bin`).
//! The activation chains (asinh/exp/log) earn `assert_oracle_eq`'s hybrid bound off
//! the oracle env; the cost quirk + in-place semantics are expression-coded units.

mod common;

use std::path::PathBuf;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::config::NnetSpec;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

fn cfg_text() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/1_worker_1.config");
    std::fs::read_to_string(p).unwrap()
}

fn real_map() -> IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(&cfg_text())
}

fn weights_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/NNweights_config1.bin")
}

// === from_legacy on the real config ==========================================

#[test]
fn from_legacy_real_config_topology() {
    let m = real_map();
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    assert_eq!(cfg.lstm_neuron_nb, vec![23, 24, 24]);
    assert_eq!(cfg.lstm_sub_sampling, vec![4, 1]);
    assert_eq!(cfg.output_neuron_nb, vec![48, 12, 1]);
    assert_eq!(cfg.output_sub_sampling, vec![1, 1]);
    assert!(!cfg.is_mlp, "LSTMNeuronNb[0] = 23 != 0 -> non-MLP");
    assert_eq!(cfg.input_normalization_type, -1);
    assert!(!cfg.two_sweeps, "BLSTM_TwoSweeps false in the fixture");
    assert!(!cfg.back_propagation_activated);
    assert!(
        !cfg.back_prop_output_network_only,
        "BLSTM_BackPropOutputNetworkOnly absent -> default false"
    );
    assert_eq!(cfg.target_enforcement_step, 0, "key absent -> default 0");

    // Peephole flags: all six keys are `true` in the real fixture.
    assert!(cfg.forward_peep.cells);
    assert!(cfg.forward_peep.gates);
    assert!(cfg.forward_peep.gates_recurrent);
    assert!(cfg.backward_peep.cells);
    assert!(cfg.backward_peep.gates);
    assert!(cfg.backward_peep.gates_recurrent);
}

#[test]
fn from_legacy_missing_peephole_keys_default_true() {
    // A direction prefix with no peephole keys at all -> all three default true
    // (LSTMLayer.cpp:10-13 conf.get<bool>(..., true)).
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("X_LSTMNeuronNb".into(), "2,3".into());
    m.insert("X_LSTMSubSampling".into(), "1".into());
    m.insert("X_OutputNeuronNb".into(), "6,2".into());
    m.insert("X_OutputSubSampling".into(), "1".into());
    m.insert("X_InputNormalizationType".into(), "0".into());
    m.insert("X_TwoSweeps".into(), "false".into());
    let cfg = BlstmConfig::from_legacy(&m, "X").unwrap();
    assert!(cfg.forward_peep.cells);
    assert!(cfg.forward_peep.gates);
    assert!(cfg.forward_peep.gates_recurrent);
    assert!(cfg.backward_peep.cells);
    assert!(cfg.backward_peep.gates);
    assert!(cfg.backward_peep.gates_recurrent);
}

// === from_legacy validation errors ===========================================

fn base_map() -> IndexMap<String, String> {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("X_LSTMNeuronNb".into(), "2,3".into());
    m.insert("X_LSTMSubSampling".into(), "1".into());
    m.insert("X_OutputNeuronNb".into(), "6,2".into());
    m.insert("X_OutputSubSampling".into(), "1".into());
    m.insert("X_InputNormalizationType".into(), "0".into());
    m.insert("X_TwoSweeps".into(), "false".into());
    m
}

#[test]
fn from_legacy_err_lstm_neuron_nb_too_short() {
    let mut m = base_map();
    m.insert("X_LSTMNeuronNb".into(), "5".into());
    assert!(BlstmConfig::from_legacy(&m, "X").is_err());
}

#[test]
fn from_legacy_err_lstm_sub_sampling_wrong_len() {
    let mut m = base_map();
    m.insert("X_LSTMSubSampling".into(), "1,2".into()); // needs len 1
    assert!(BlstmConfig::from_legacy(&m, "X").is_err());
}

#[test]
fn from_legacy_err_output_neuron_nb_too_short() {
    let mut m = base_map();
    m.insert("X_OutputNeuronNb".into(), "6".into());
    assert!(BlstmConfig::from_legacy(&m, "X").is_err());
}

#[test]
fn from_legacy_err_output_sub_sampling_wrong_len() {
    let mut m = base_map();
    m.insert("X_OutputSubSampling".into(), "1,1".into()); // needs len 1
    assert!(BlstmConfig::from_legacy(&m, "X").is_err());
}

#[test]
fn from_legacy_err_output_first_layer_not_twice_last_lstm() {
    // LSTMNeuronNb.last() = 3 -> OutputNeuronNb[0] must be 6; give 5 instead.
    let mut m = base_map();
    m.insert("X_OutputNeuronNb".into(), "5,2".into());
    let err = BlstmConfig::from_legacy(&m, "X").unwrap_err();
    assert!(err.to_string().contains("twice"));
}

#[test]
fn from_legacy_mlp_mode_skips_twice_constraint() {
    // LSTMNeuronNb[0] == 0 -> the 2*last constraint is not checked at all
    // (BLSTMNeuralNetwork.cpp:43: `(LSTMNeuronNb[0] != 0) && ...`).
    let mut m = base_map();
    m.insert("X_LSTMNeuronNb".into(), "0,3".into());
    m.insert("X_OutputNeuronNb".into(), "5,2".into()); // would violate 2*3=6 if checked
    let cfg = BlstmConfig::from_legacy(&m, "X").unwrap();
    assert!(cfg.is_mlp);
}

// === nb_of_weights on the real net ===========================================

#[test]
fn real_net_nb_of_weights_is_33671() {
    let m = real_map();
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let net = BlstmNetwork::from_config(cfg).unwrap();
    assert_eq!(net.nb_of_weights(), 33671);
}

#[test]
fn nb_of_weights_cross_checks_config_element_count() {
    // Phase 0a seam agreement (spec decision 5): config::element_count on the
    // NnetSpec derived from the SAME real config must equal nb_of_weights().
    let m = real_map();
    let spec = NnetSpec::from_legacy(&m, "BLSTM").unwrap();
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let net = BlstmNetwork::from_config(cfg).unwrap();
    assert_eq!(speech::config::element_count(&spec), net.nb_of_weights());
}

// === set_weights / get_weights round trip ====================================

#[test]
fn set_weights_then_get_weights_round_trips_real_bin() {
    let m = real_map();
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();

    let flat = speech::io::binary::read_weight_vector(&weights_bin_path()).unwrap();
    assert_eq!(flat.len(), 33671);

    net.set_weights(&flat).unwrap();
    let round = net.get_weights();

    assert_eq!(round.len(), flat.len());
    for (i, (&a, &b)) in flat.iter().zip(round.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "mismatch at flat index {i}: a=0x{:016x} ({a}) b=0x{:016x} ({b})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn set_weights_err_when_too_short() {
    let m = real_map();
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();

    let mut flat = speech::io::binary::read_weight_vector(&weights_bin_path()).unwrap();
    flat.truncate(33670); // one short of the 33671 needed
    assert!(net.set_weights(&flat).is_err());
}

#[test]
fn set_weights_tolerates_longer_vector_and_returns_head_on_get() {
    // Legacy: NNWeights.size() > getNbOfWeights() -> warning, but setWeights
    // still proceeds (consuming only the head). Ported as silent Ok here.
    let m = real_map();
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();

    let mut flat = speech::io::binary::read_weight_vector(&weights_bin_path()).unwrap();
    flat.extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]); // 5 junk tail values
    assert_eq!(flat.len(), 33676);

    net.set_weights(&flat).unwrap();
    let round = net.get_weights();
    assert_eq!(
        round.len(),
        33671,
        "get_weights returns only the 33671 head"
    );
    for (i, (&a, &b)) in flat[..33671].iter().zip(round.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "mismatch at flat index {i}");
    }
}

// === MLP-mode structural test ================================================

#[test]
fn mlp_mode_structural_shape() {
    // LSTMNeuronNb[0] = 0 -> is_mlp, forward/backward nets absent; nb_of_weights
    // is just the output net's weights plus 2*OutputNeuronNb[0] (input_size in
    // MLP mode is the OUTPUT network's input size, per getInputSize :177-183).
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("X_LSTMNeuronNb".into(), "0,3".into());
    m.insert("X_LSTMSubSampling".into(), "1".into());
    m.insert("X_OutputNeuronNb".into(), "6,2".into());
    m.insert("X_OutputSubSampling".into(), "1".into());
    m.insert("X_InputNormalizationType".into(), "0".into());
    m.insert("X_TwoSweeps".into(), "false".into());

    let cfg = BlstmConfig::from_legacy(&m, "X").unwrap();
    assert!(cfg.is_mlp);
    let net = BlstmNetwork::from_config(cfg).unwrap();

    assert_eq!(
        net.input_size(),
        6,
        "MLP mode: input_size == OutputNeuronNb[0]"
    );
    // Output net: layer 0 is 6->2, weights = 2*(6+1) = 14. Tail = 2*6 = 12.
    let output_only_weights = 2 * (6 + 1);
    assert_eq!(net.nb_of_weights(), output_only_weights + 2 * 6);

    let flat: Vec<f64> = (0..net.nb_of_weights()).map(|i| i as f64 * 0.5).collect();
    let mut net = net;
    net.set_weights(&flat).unwrap();
    let round = net.get_weights();
    assert_eq!(round.len(), flat.len());
    for (i, (&a, &b)) in flat.iter().zip(round.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "mlp round-trip mismatch at {i}");
    }
}

#[test]
fn reset_weights_derivatives_resets_input_statistics() {
    let m = real_map();
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.input_statistics.n = 42;
    net.input_statistics.mean = vec![1.0, 2.0];
    net.reset_weights_derivatives();
    assert_eq!(net.input_statistics.n, 0);
    assert!(net.input_statistics.mean.is_empty());
}

// ============================================================================
// Task 8: input normalization + core forward + plain feed_forward_backward
// ============================================================================

/// Synthetic-net config map matching the harness Task 8 "SYNB" prefix: LSTM [3,4,2]
/// sub [2,1], output [4,5,3] sub [1,1]. `norm_type` is the only per-dump knob.
fn synth_map(norm_type: i16) -> IndexMap<String, String> {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("SYNB_LSTMNeuronNb".into(), "3,4,2".into());
    m.insert("SYNB_LSTMSubSampling".into(), "2,1".into());
    m.insert("SYNB_OutputNeuronNb".into(), "4,5,3".into());
    m.insert("SYNB_OutputSubSampling".into(), "1,1".into());
    m.insert("SYNB_InputNormalizationType".into(), norm_type.to_string());
    m.insert("SYNB_TwoSweeps".into(), "false".into());
    m.insert("SYNB_BackPropagationActivated".into(), "false".into());
    m
}

/// The synthetic net's flat weight vector, EXACTLY as the harness builds it: body =
/// `((k*11+3) % 97)/97 - 0.5`; the last `2*3` entries are a NONZERO mean/std tail
/// (mean[j] = 0.1*(j+1), std[j] = 1.0 + 0.05*j), so type 1 actually shifts/scales.
fn synth_flat() -> Vec<f64> {
    // nb: fwd LSTM net + bwd LSTM net + output dense net + 2*3 tail.
    let lstm_nb = |i: usize, o: usize| 4 * i * o + 4 * o * o + 12 * o + 4 * o;
    let fwd_nb = lstm_nb(6, 4) + lstm_nb(4, 2); // L0 in=3*2=6 out=4; L1 in=4*1=4 out=2
    let out_nb = 5 * (4 + 1) + 3 * (5 + 1); // L0 in=4 out=5; L1 in=5 out=3
    let tail = 2 * 3;
    let nb_total = fwd_nb + fwd_nb + out_nb + tail;
    let mut flat = vec![0.0_f64; nb_total];
    for (k, v) in flat.iter_mut().enumerate().take(nb_total - tail) {
        *v = ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5;
    }
    for j in 0..3 {
        flat[nb_total - tail + j] = 0.1 * (j as f64 + 1.0);
        flat[nb_total - tail + 3 + j] = 1.0 + 0.05 * j as f64;
    }
    flat
}

/// The harness Task 8 synthetic input: `x[t,j] = ((t*29 + j*13 + 5) % 97)/97 - 0.5`,
/// T=12, 3 cols.
fn synth_input() -> Array2<f64> {
    Array2::from_shape_fn((12, 3), |(t, j)| {
        ((t * 29 + j * 13 + 5) % 97) as f64 / 97.0 - 0.5
    })
}

/// Build + weight-load the synthetic net for a given normalization type, plain path.
fn make_synth_net(norm_type: i16) -> BlstmNetwork {
    let cfg = BlstmConfig::from_legacy(&synth_map(norm_type), "SYNB").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.set_weights(&synth_flat()).unwrap();
    net.set_processing_type(false, false);
    net
}

/// Tag used in the harness dump filenames: negative types get an `m` prefix.
fn norm_tag(norm_type: i16) -> String {
    if norm_type < 0 {
        format!("m{}", -norm_type)
    } else {
        norm_type.to_string()
    }
}

#[test]
fn synth_norm_out_matches_oracle_all_types() {
    for norm_type in [1_i16, -1, -2, 0] {
        let mut net = make_synth_net(norm_type);
        let mut input = synth_input();
        let mut output = Array2::<f64>::zeros((6, 3)); // 12/2/1 = 6 rows, O=3
        let empty = Array2::<f64>::zeros((0, 0));
        net.feed_forward_backward(&mut input, 4, 2, &mut output, &empty);

        let tag = norm_tag(norm_type);
        common::assert_oracle_eq(
            &output,
            &common::load_bin_phase2(&format!("blstm_norm{tag}_out.bin")),
            &format!("blstm_norm{tag}_out"),
        );
    }
}

#[test]
fn synth_norm_input_after_matches_oracle_all_types() {
    // The (possibly-mutated) input is dumped as blstm_norm<tag>_input_after.bin.
    // Types 1 and -1 mutate the caller's matrix IN PLACE; types -2 and 0 leave it
    // untouched. Comparing the post-call input pins the in-place semantics.
    for norm_type in [1_i16, -1, -2, 0] {
        let mut net = make_synth_net(norm_type);
        let mut input = synth_input();
        let mut output = Array2::<f64>::zeros((6, 3));
        let empty = Array2::<f64>::zeros((0, 0));
        net.feed_forward_backward(&mut input, 4, 2, &mut output, &empty);

        let tag = norm_tag(norm_type);
        common::assert_oracle_eq(
            &input,
            &common::load_bin_phase2(&format!("blstm_norm{tag}_input_after.bin")),
            &format!("blstm_norm{tag}_input_after"),
        );
    }
}

#[test]
fn type_minus2_and_zero_leave_input_untouched() {
    // Type -2 normalizes into a COPY (input untouched); type 0 does nothing. Assert
    // the post-call input is bit-identical to the pristine input (stronger than the
    // golden -- proves no mutation at all).
    for norm_type in [-2_i16, 0] {
        let mut net = make_synth_net(norm_type);
        let pristine = synth_input();
        let mut input = pristine.clone();
        let mut output = Array2::<f64>::zeros((6, 3));
        let empty = Array2::<f64>::zeros((0, 0));
        net.feed_forward_backward(&mut input, 4, 2, &mut output, &empty);
        common::assert_bits_eq(
            &input,
            &pristine,
            &format!("type {norm_type} input untouched"),
        );
    }
}

#[test]
fn type1_columns_beyond_mean_tail_untouched() {
    // External normalization touches only columns jj < min(cols, mean.size()). Build
    // a net whose input tail (mean/std) is SHORTER than the input width, and assert
    // the extra columns pass through unchanged. Net input size = LSTMNeuronNb[0] = 3,
    // so the mean/std tail has 3 entries; feed a 5-column input -> columns 3,4 must be
    // untouched, columns 0,1,2 normalized.
    let cfg = BlstmConfig::from_legacy(&synth_map(1), "SYNB").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.set_weights(&synth_flat()).unwrap();
    net.set_processing_type(false, false);

    // 12x5 input: first 3 cols per the synth formula, cols 3,4 are sentinels.
    let sentinel3 = f64::from_bits(0x4010_0000_0000_0000); // 4.0
    let sentinel4 = f64::from_bits(0x4014_0000_0000_0000); // 5.0
    let mut input = Array2::from_shape_fn((12, 5), |(t, j)| match j {
        0..=2 => ((t * 29 + j * 13 + 5) % 97) as f64 / 97.0 - 0.5,
        3 => sentinel3,
        _ => sentinel4,
    });
    let before_cols_34 = input.slice(ndarray::s![.., 3..5]).to_owned();

    // The forward net input size is 3, so leftCols(3) is used inside feed_forward;
    // the normalization must not have touched cols 3,4 in the caller's matrix.
    let mut output = Array2::<f64>::zeros((6, 3));
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 4, 2, &mut output, &empty);

    let after_cols_34 = input.slice(ndarray::s![.., 3..5]).to_owned();
    common::assert_bits_eq(
        &after_cols_34,
        &before_cols_34,
        "type-1 columns beyond mean tail untouched",
    );
    // And columns 0..3 DID change (sanity: normalization actually ran).
    let mut any_changed = false;
    for t in 0..12 {
        for j in 0..3 {
            let orig = ((t * 29 + j * 13 + 5) % 97) as f64 / 97.0 - 0.5;
            if input[[t, j]].to_bits() != orig.to_bits() {
                any_changed = true;
            }
        }
    }
    assert!(any_changed, "type-1 must have normalized columns 0..3");
}

#[test]
fn real_fullseq_matches_oracle() {
    // The REAL net (1_worker_1.config, InputNormalizationType == -1) on a synthetic
    // 200x23 input, plain FFB, no targets. Output (50x1) + forward/backward LSTM
    // hidden states (50x24). EXPECTED tiny divergence vs Eigen's blocked k=23 GEMM
    // (Task 1 lstm_input_gemm probe) -- assert_oracle_eq's hybrid bound covers it.
    let m = real_map();
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    let flat = speech::io::binary::read_weight_vector(&weights_bin_path()).unwrap();
    net.set_weights(&flat).unwrap();
    net.set_processing_type(false, false);

    let mut input = Array2::from_shape_fn((200, 23), |(t, j)| {
        ((t * 31 + j * 17) % 100) as f64 / 100.0 - 0.5
    });
    let mut output = Array2::<f64>::zeros((50, 1)); // 200/4/1 = 50 rows, O=1
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 4, 2, &mut output, &empty);

    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("blstm_real_fullseq_out.bin"),
        "blstm_real_fullseq_out",
    );
    common::assert_oracle_eq(
        &net.output_forward,
        &common::load_bin_phase2("blstm_real_fullseq_fwd.bin"),
        "blstm_real_fullseq_fwd",
    );
    common::assert_oracle_eq(
        &net.output_backward,
        &common::load_bin_phase2("blstm_real_fullseq_bwd.bin"),
        "blstm_real_fullseq_bwd",
    );
}

// === Cost quirk + nb_of_classif accounting ==================================

#[test]
fn cost_quirk_enforcement_neg_overwrites_output_interior_with_minus_half() {
    // BLSTMNeuralNetwork.cpp:815-824: with targets present and target_enforcement_step
    // < 0, the interior rows [1, rows-1) of the caller-visible output are overwritten
    // with -0.5 BEFORE the cost is computed, and the cost is accumulated against the
    // -0.5 targets (interior) rather than the caller's targets. Use a MULTI-CLASS
    // output (O=3) so the softmax cross-entropy path runs; verify (a) output interior
    // == -0.5 exactly after the call, and (b) cost equals the expression-coded cost
    // against the -0.5-interior targets, computed independently via speech::cost.
    let mut m = synth_map(0); // norm type 0: no input mutation, keep the math simple
    m.insert("SYNB_TargetEnforcementStep".into(), "-1".into());
    let cfg = BlstmConfig::from_legacy(&m, "SYNB").unwrap();
    let cost_law = cfg.cost_law.clone();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.set_weights(&synth_flat()).unwrap();
    net.set_processing_type(false, false);

    let mut input = synth_input();
    let mut output = Array2::<f64>::zeros((6, 3));
    // One-hot-ish targets (6 rows, 3 classes): on-class rotates by row.
    let target = Array2::from_shape_fn((6, 3), |(r, c)| if c == r % 3 { 1.0 } else { 0.0 });
    net.feed_forward_backward(&mut input, 4, 2, &mut output, &target);

    // (a) Output interior rows [1,5) overwritten with -0.5 exactly.
    for r in 1..5 {
        for c in 0..3 {
            assert_eq!(
                output[[r, c]].to_bits(),
                (-0.5_f64).to_bits(),
                "output interior ({r},{c}) must be exactly -0.5"
            );
        }
    }
    // Rows 0 and 5 (the boundary rows) are NOT overwritten -- they keep the softmax
    // output (each row sums to ~1, so not -0.5).
    assert_ne!(output[[0, 0]].to_bits(), (-0.5_f64).to_bits());
    assert_ne!(output[[5, 0]].to_bits(), (-0.5_f64).to_bits());

    // (b) Expression-coded cost: build the -0.5-interior targets, take the -0.5-interior
    // output (already mutated in `output`), and compute the softmax cost independently.
    let mut new_target = target.clone();
    for r in 1..5 {
        for c in 0..3 {
            new_target[[r, c]] = -0.5;
        }
    }
    let out_flat: Vec<f64> = output.iter().copied().collect();
    let tgt_flat: Vec<f64> = new_target.iter().copied().collect();
    let expected_cost = cost_law.compute_cost(&out_flat, &tgt_flat, 3);
    assert_eq!(
        net.cost.to_bits(),
        expected_cost.to_bits(),
        "cost must be accumulated against the -0.5-interior targets and output"
    );
    assert_eq!(net.nb_of_classif, 6, "nb_of_classif += output.rows() == 6");
}

#[test]
fn no_enforcement_uses_given_targets_and_leaves_output() {
    // target_enforcement_step >= 0 (default 0): no interior overwrite; cost is against
    // the given targets, output rows keep the softmax values. nb_of_classif == rows.
    let cfg = BlstmConfig::from_legacy(&synth_map(0), "SYNB").unwrap();
    let cost_law = cfg.cost_law.clone();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.set_weights(&synth_flat()).unwrap();
    net.set_processing_type(false, false);

    let mut input = synth_input();
    let mut output = Array2::<f64>::zeros((6, 3));
    let target = Array2::from_shape_fn((6, 3), |(r, c)| if c == r % 3 { 1.0 } else { 0.0 });
    net.feed_forward_backward(&mut input, 4, 2, &mut output, &target);

    // No -0.5 overwrite anywhere.
    for r in 0..6 {
        for c in 0..3 {
            assert_ne!(
                output[[r, c]].to_bits(),
                (-0.5_f64).to_bits(),
                "no enforcement: output ({r},{c}) must keep its softmax value"
            );
        }
    }
    let out_flat: Vec<f64> = output.iter().copied().collect();
    let tgt_flat: Vec<f64> = target.iter().copied().collect();
    let expected_cost = cost_law.compute_cost(&out_flat, &tgt_flat, 3);
    assert_eq!(net.cost.to_bits(), expected_cost.to_bits());
    assert_eq!(net.nb_of_classif, 6);
}

#[test]
fn no_targets_leaves_cost_and_classif_zero() {
    // Empty target (0 rows) -> no cost accumulation, nb_of_classif stays 0.
    let mut net = make_synth_net(-1);
    let mut input = synth_input();
    let mut output = Array2::<f64>::zeros((6, 3));
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 4, 2, &mut output, &empty);
    assert_eq!(net.cost.to_bits(), 0.0_f64.to_bits());
    assert_eq!(net.nb_of_classif, 0);
}

// ============================================================================
// Task 9: the four windowed forward drivers
// ============================================================================

/// The synthetic-net closed-form input at an arbitrary row count T (the harness Task
/// 9 `makeInput`: `x[t,j] = ((t*29 + j*13 + 5) % 97)/97 - 0.5`, 3 cols).
fn synth_input_t(t_rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((t_rows, 3), |(t, j)| {
        ((t * 29 + j * 13 + 5) % 97) as f64 / 97.0 - 0.5
    })
}

/// Build the synthetic BLSTM net (LSTM [3,4,2] sub [2,1], output [4,5,3] sub [1,1])
/// with `truncates`/`overlaps` processing flags and an optional `two_sweeps`. Norm
/// type 0 (no input mutation) matches the harness Task 9 dumps.
fn make_synth_net_windowed(truncates: bool, overlaps: bool, two_sweeps: bool) -> BlstmNetwork {
    let mut m = synth_map(0);
    m.insert("SYNB_TwoSweeps".into(), two_sweeps.to_string());
    let cfg = BlstmConfig::from_legacy(&m, "SYNB").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.set_weights(&synth_flat()).unwrap();
    net.set_processing_type(truncates, overlaps);
    net
}

/// The MLP-mode net from the harness Task 9 Mode D: LSTM [0,3] (is_mlp), output
/// [5,4,1] sub [2,1]. Flat body `((k*11+3) % 97)/97 - 0.5`; tail (input_size = 5)
/// mean[j]=0.1*(j+1), std[j]=1.0+0.05*j. Weights: L0 in = 5*2 = 10 out 4, L1 in = 4
/// out 1 -> dense 4*(10+1) + 1*(4+1) = 49; tail 2*5 = 10; total 59.
fn make_mlp_overlap_net() -> BlstmNetwork {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("SYNM_LSTMNeuronNb".into(), "0,3".into());
    m.insert("SYNM_LSTMSubSampling".into(), "1".into());
    m.insert("SYNM_OutputNeuronNb".into(), "5,4,1".into());
    m.insert("SYNM_OutputSubSampling".into(), "2,1".into());
    m.insert("SYNM_InputNormalizationType".into(), "0".into());
    m.insert("SYNM_TwoSweeps".into(), "false".into());
    let cfg = BlstmConfig::from_legacy(&m, "SYNM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();

    let out_nb = 4 * (10 + 1) + (4 + 1); // L0 4*(10+1)=44, L1 1*(4+1)=5 -> 49
    let tail = 2 * 5;
    let nb_total = out_nb + tail;
    let mut flat = vec![0.0_f64; nb_total];
    for (k, v) in flat.iter_mut().enumerate().take(nb_total - tail) {
        *v = ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5;
    }
    for j in 0..5 {
        flat[nb_total - tail + j] = 0.1 * (j as f64 + 1.0);
        flat[nb_total - tail + 5 + j] = 1.0 + 0.05 * j as f64;
    }
    net.set_weights(&flat).unwrap();
    net.set_processing_type(true, true); // truncate + overlap -> MLPOverLap
    net
}

/// The MLP-mode net's 5-col closed-form input, T rows (harness Mode D input formula).
fn mlp_input(t_rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((t_rows, 5), |(t, j)| {
        ((t * 29 + j * 13 + 5) % 97) as f64 / 97.0 - 0.5
    })
}

#[test]
fn truncate_out_matches_oracle() {
    // Truncate (no two-sweeps): T=19 window 6. outputSeq pre-seeded to 10 rows so the
    // trailing chunk (jj=18, 1 input row -> lengthShort==0) is SILENTLY DROPPED and
    // ROW 9 stays at its pre-seed (0). Matches blstm_truncate_{out,fwd,bwd}.bin.
    let mut net = make_synth_net_windowed(true, false, false);
    let mut input = synth_input_t(19);
    let mut output = Array2::<f64>::zeros((10, 3)); // pre-seeded IDENTICALLY to the harness
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 6, 3, &mut output, &empty);

    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("blstm_truncate_out.bin"),
        "blstm_truncate_out",
    );
    common::assert_oracle_eq(
        &net.output_forward,
        &common::load_bin_phase2("blstm_truncate_fwd.bin"),
        "blstm_truncate_fwd",
    );
    common::assert_oracle_eq(
        &net.output_backward,
        &common::load_bin_phase2("blstm_truncate_bwd.bin"),
        "blstm_truncate_bwd",
    );
}

#[test]
fn truncate_dropped_chunk_leaves_output_row_untouched() {
    // Pre-seed the output with a SENTINEL (not zero) and assert the dropped-chunk row
    // 9 retains it (proving the driver does not write there), while rows 0..9 ARE
    // overwritten. This is stronger than the zero-seeded golden.
    let mut net = make_synth_net_windowed(true, false, false);
    let mut input = synth_input_t(19);
    let sentinel = f64::from_bits(0x4059_0000_0000_0000); // 100.0
    let mut output = Array2::<f64>::from_elem((10, 3), sentinel);
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 6, 3, &mut output, &empty);

    // Row 9 (dropped chunk's begin/subRatio = 18/2 = 9 target) keeps the sentinel.
    for c in 0..3 {
        assert_eq!(
            output[[9, c]].to_bits(),
            sentinel.to_bits(),
            "row 9 col {c} must retain the sentinel (dropped chunk)"
        );
    }
    // Rows 0..9 were all written by the three full windows -> no sentinel survives.
    let mut any_sentinel = false;
    for r in 0..9 {
        for c in 0..3 {
            if output[[r, c]].to_bits() == sentinel.to_bits() {
                any_sentinel = true;
            }
        }
    }
    assert!(!any_sentinel, "rows 0..9 must all be overwritten");
}

#[test]
fn two_sweeps_out_matches_oracle_and_doubles_hidden_width() {
    // TwoSweeps: T=20 window 8. Output averaged over two sweeps; _OutputForward/
    // _OutputBackward are the HCAT of the two sweeps' hidden windows -> DOUBLE-WIDTH
    // (2*fwdOut = 2*2 = 4). Matches blstm_twosweeps_{out,fwd,bwd}.bin.
    let mut net = make_synth_net_windowed(true, false, true);
    let mut input = synth_input_t(20);
    let mut output = Array2::<f64>::zeros((10, 3));
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 8, 4, &mut output, &empty);

    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("blstm_twosweeps_out.bin"),
        "blstm_twosweeps_out",
    );
    // Double-width contract: the forward LSTM output size is 2, so the stitched
    // hidden states are 2*2 = 4 columns wide.
    assert_eq!(
        net.output_forward.ncols(),
        4,
        "TwoSweeps output_forward must be DOUBLE-WIDTH (2 * lstm out = 4)"
    );
    assert_eq!(net.output_backward.ncols(), 4);
    common::assert_oracle_eq(
        &net.output_forward,
        &common::load_bin_phase2("blstm_twosweeps_fwd.bin"),
        "blstm_twosweeps_fwd",
    );
    common::assert_oracle_eq(
        &net.output_backward,
        &common::load_bin_phase2("blstm_twosweeps_bwd.bin"),
        "blstm_twosweeps_bwd",
    );
}

#[test]
fn overlap_out_matches_oracle_full_coverage() {
    // OverLap: T=20 window_size 4 shift 3, grid snapping exercised, FULL coverage (no
    // NaN). Matches blstm_overlap_{out,fwd,bwd}.bin.
    let mut net = make_synth_net_windowed(true, true, false);
    let mut input = synth_input_t(20);
    let mut output = Array2::<f64>::zeros((10, 3));
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 4, 3, &mut output, &empty);

    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("blstm_overlap_out.bin"),
        "blstm_overlap_out",
    );
    common::assert_oracle_eq(
        &net.output_forward,
        &common::load_bin_phase2("blstm_overlap_fwd.bin"),
        "blstm_overlap_fwd",
    );
    common::assert_oracle_eq(
        &net.output_backward,
        &common::load_bin_phase2("blstm_overlap_bwd.bin"),
        "blstm_overlap_bwd",
    );
}

#[test]
fn overlap_uncovered_rows_are_nan() {
    // OverLap NaN case: window_size 3 shift 10 (> 2*3+1=7) leaves rows 6-9 uncovered
    // -> outputCount 0 -> 0/0 = NaN (REPRODUCED, no guard). Rows 0-5 are finite and
    // match the fixture; rows 6-9 are NaN in BOTH the Rust output and the golden.
    let mut net = make_synth_net_windowed(true, true, false);
    let mut input = synth_input_t(20);
    let mut output = Array2::<f64>::zeros((10, 3));
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 3, 10, &mut output, &empty);

    let golden = common::load_bin_phase2("blstm_overlap_nan_out.bin");
    let golden_fwd = common::load_bin_phase2("blstm_overlap_nan_fwd.bin");

    // Uncovered rows 6-9 are NaN in both output AND the fwd hidden states.
    for r in 6..10 {
        for c in 0..3 {
            assert!(output[[r, c]].is_nan(), "output ({r},{c}) must be NaN");
            assert!(golden[[r, c]].is_nan(), "golden ({r},{c}) must be NaN");
        }
        for c in 0..2 {
            assert!(
                net.output_forward[[r, c]].is_nan(),
                "output_forward ({r},{c}) must be NaN"
            );
            assert!(
                golden_fwd[[r, c]].is_nan(),
                "golden_fwd ({r},{c}) must be NaN"
            );
        }
    }
    // Covered rows 0-5 are finite and compare (assert_oracle_eq treats NaN via bits in
    // Strict mode; restrict to the finite rows so the comparison is meaningful).
    let out_finite = output.slice(ndarray::s![..6, ..]).to_owned();
    let golden_finite = golden.slice(ndarray::s![..6, ..]).to_owned();
    common::assert_oracle_eq(&out_finite, &golden_finite, "blstm_overlap_nan_out_finite");
    let fwd_finite = net.output_forward.slice(ndarray::s![..6, ..]).to_owned();
    let golden_fwd_finite = golden_fwd.slice(ndarray::s![..6, ..]).to_owned();
    common::assert_oracle_eq(
        &fwd_finite,
        &golden_fwd_finite,
        "blstm_overlap_nan_fwd_finite",
    );
}

#[test]
fn mlp_overlap_out_matches_oracle_and_ignores_window_size() {
    // MLPOverLap: T=12 shift 2. The passed window_size is IGNORED (overwritten with
    // getSubSamplingRatio()/2 = 1). Call with a DELIBERATELY WRONG window_size (99, as
    // the harness does) -> must still match blstm_mlpoverlap_out.bin. Row 0 is
    // edge-skipped (keeps caller zeros). _OutputForward/_OutputBackward EMPTIED.
    let mut net = make_mlp_overlap_net();
    let mut input = mlp_input(12);
    let mut output = Array2::<f64>::zeros((6, 1)); // row 0 stays 0
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input, 99, 2, &mut output, &empty);

    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("blstm_mlpoverlap_out.bin"),
        "blstm_mlpoverlap_out",
    );
    // _OutputForward/_OutputBackward emptied (0 elements).
    assert_eq!(
        net.output_forward.len(),
        0,
        "MLPOverLap empties output_forward"
    );
    assert_eq!(
        net.output_backward.len(),
        0,
        "MLPOverLap empties output_backward"
    );
    // Row 0 (edge-skipped) keeps the caller's zero.
    assert_eq!(output[[0, 0]].to_bits(), 0.0_f64.to_bits());
}

#[test]
fn mlp_overlap_window_size_is_ignored() {
    // Call twice with DIFFERENT (both wrong) passed window_sizes; the driver forces
    // window_size = getSubSamplingRatio()/2 regardless, so the outputs must be bitwise
    // identical.
    let run = |ws: usize| {
        let mut net = make_mlp_overlap_net();
        let mut input = mlp_input(12);
        let mut output = Array2::<f64>::zeros((6, 1));
        let empty = Array2::<f64>::zeros((0, 0));
        net.feed_forward_backward(&mut input, ws, 2, &mut output, &empty);
        output
    };
    let a = run(99);
    let b = run(3);
    common::assert_bits_eq(&a, &b, "MLPOverLap ignores the passed window_size");
}

#[test]
fn dispatch_routes_each_processing_type() {
    // set_processing_type routes the BLSTM dispatch: (false,false) -> plain (fwd/bwd
    // hidden states are the LSTM out width 2); (true,false) -> Truncate; (true,false)
    // + two_sweeps -> TwoSweeps (DOUBLE-WIDTH); (true,true) -> OverLap. Each path is
    // distinguishable by the resulting _OutputForward width / row coverage.
    let empty = Array2::<f64>::zeros((0, 0));

    // Plain: fwd width 2 (single sweep), full coverage.
    let mut plain = make_synth_net_windowed(false, false, false);
    let mut input = synth_input_t(20);
    let mut out = Array2::<f64>::zeros((10, 3));
    plain.feed_forward_backward(&mut input, 8, 4, &mut out, &empty);
    assert_eq!(
        plain.output_forward.ncols(),
        2,
        "plain path: single-width hidden"
    );

    // TwoSweeps: DOUBLE-WIDTH hidden (4).
    let mut two = make_synth_net_windowed(true, false, true);
    let mut input = synth_input_t(20);
    let mut out = Array2::<f64>::zeros((10, 3));
    two.feed_forward_backward(&mut input, 8, 4, &mut out, &empty);
    assert_eq!(
        two.output_forward.ncols(),
        4,
        "two-sweeps path: double-width hidden"
    );

    // OverLap with an uncovered-rows config -> NaN rows (only this path produces NaN).
    let mut ov = make_synth_net_windowed(true, true, false);
    let mut input = synth_input_t(20);
    let mut out = Array2::<f64>::zeros((10, 3));
    ov.feed_forward_backward(&mut input, 3, 10, &mut out, &empty);
    assert!(out[[9, 0]].is_nan(), "overlap path: uncovered row 9 is NaN");

    // MLPOverLap: empties the hidden states.
    let mut mlp = make_mlp_overlap_net();
    let mut input = mlp_input(12);
    let mut out = Array2::<f64>::zeros((6, 1));
    mlp.feed_forward_backward(&mut input, 4, 2, &mut out, &empty);
    assert_eq!(
        mlp.output_forward.len(),
        0,
        "mlp-overlap path: empties hidden states"
    );
}
