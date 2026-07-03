//! Phase 2 Task 7: `BlstmConfig` + `BlstmNetwork` construction + flat weight seam.
//!
//! Golden tests against the real `1_worker_1.config` (BLSTM prefix) and the real
//! `NNweights_config1.bin` (33,671 x 1). Pure construction/plumbing tests -- no
//! oracle harness dump needed (this is the config/weight seam, not numerics).

use std::path::PathBuf;

use indexmap::IndexMap;
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
