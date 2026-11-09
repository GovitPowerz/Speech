//! Task 3: NN lifecycle surface goldens.
//!
//! Covers the `<prefix>_weightsFile` load semantics (empty skip / too-few error /
//! too-many truncate) transcribed from `BLSTMNeuralNetwork.cpp:122-150`. The
//! `saveWeights` three-artifact write (`:312-331`) is gone (issue #136); its
//! successor, the bag's in-memory best pack, is pinned in `phase4a_save_update.rs`.
//! The `analyseInputSeq` accumulation across files (`:385-417`) is gone too (issue
//! #140, write-only once #136 retired its stats file).

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::io::binary::write_matrix;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0")
}

/// Build the real `BLSTM` net from `1_worker_1.config` (33,671 weights, `_IsMLP`
/// false), with `<prefix>_weightsFile` overridden to `value`.
fn real_net_with_weights_file(value: &str) -> (IndexMap<String, String>, BlstmNetwork) {
    let text = std::fs::read_to_string(ref_dir().join("1_worker_1.config")).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&text);
    m.insert("BLSTM_weightsFile".into(), value.to_string());
    let cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let net = BlstmNetwork::from_config(cfg).unwrap();
    (m, net)
}

#[test]
fn weights_file_empty_is_skip() {
    // Empty key value -> no load attempt (`:123` size() != 0 guard). The weights are
    // left exactly at their fresh-construct state (net body zeros, mean 0.0, std 1.0
    // tail); the load must be a no-op.
    let (m, mut net) = real_net_with_weights_file("");
    let before = net.get_weights();
    net.load_weights_file(&m, "BLSTM").unwrap();
    assert_eq!(net.get_weights(), before);
}

#[test]
fn weights_file_exact_loads() {
    // A file with exactly nb_of_weights() elements loads verbatim (`:147-148`).
    let path = ref_dir().join("NNweights_config1.bin");
    let (m, mut net) = real_net_with_weights_file(path.to_str().unwrap());
    net.load_weights_file(&m, "BLSTM").unwrap();
    let expected =
        speech::io::binary::read_weight_vector(&ref_dir().join("NNweights_config1.bin")).unwrap();
    assert_eq!(net.get_weights(), expected);
}

#[test]
fn weights_file_too_many_truncates() {
    // A .bin with MORE than nb_of_weights() elements: warning + setWeights consumes
    // only the head (`:144-146`). The loaded weights must equal the head of the file.
    let base =
        speech::io::binary::read_weight_vector(&ref_dir().join("NNweights_config1.bin")).unwrap();
    let mut too_many = base.clone();
    too_many.extend_from_slice(&[7.0, 8.0, 9.0]); // 3 extra
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("too_many.bin");
    write_matrix(&path, too_many.len(), 1, &too_many).unwrap();

    let (m, mut net) = real_net_with_weights_file(path.to_str().unwrap());
    net.load_weights_file(&m, "BLSTM").unwrap();
    assert_eq!(net.get_weights(), base);
}

#[test]
fn weights_file_too_few_errors() {
    // A .bin with FEWER than nb_of_weights() elements: the legacy exit(1); ported as
    // Err (`:141-143`).
    let base =
        speech::io::binary::read_weight_vector(&ref_dir().join("NNweights_config1.bin")).unwrap();
    let too_few = base[..base.len() - 1].to_vec();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("too_few.bin");
    write_matrix(&path, too_few.len(), 1, &too_few).unwrap();

    let (m, mut net) = real_net_with_weights_file(path.to_str().unwrap());
    assert!(net.load_weights_file(&m, "BLSTM").is_err());
}
