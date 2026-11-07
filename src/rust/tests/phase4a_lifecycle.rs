//! Task 3: NN lifecycle surface goldens.
//!
//! Covers `BlstmNetwork` InputStatistics accumulation across files (STRICT bits vs a
//! hand-chained `from_matrix`+`update`) and the `<prefix>_weightsFile` load semantics
//! (empty skip / too-few error / too-many truncate) transcribed from
//! `BLSTMNeuralNetwork.cpp:122-150,385-417`. The `saveWeights` three-artifact write
//! (`:312-331`) is gone (issue #136); its successor, the bag's in-memory best pack, is
//! pinned in `phase4a_save_update.rs`.

use std::path::PathBuf;

use indexmap::IndexMap;
use ndarray::Array2;

use speech::features::stats::InputStatistics;
use speech::io::binary::write_matrix;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0")
}

/// Minimal type-1-normalized MLP net (LSTM `[0,3]` -> `_IsMLP`; output `[5,4,1]`,
/// sub `[2,1]`), `input_size() == 5`. Type 1 is the branch whose scoring forward
/// folds `analyseInputSeq` into `_InputStatistics` (`:728`).
fn stats_net() -> BlstmNetwork {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("SYNT_LSTMNeuronNb".into(), "0,3".into());
    m.insert("SYNT_LSTMSubSampling".into(), "1".into());
    m.insert("SYNT_OutputNeuronNb".into(), "5,4,1".into());
    m.insert("SYNT_OutputSubSampling".into(), "2,1".into());
    m.insert("SYNT_InputNormalizationType".into(), "1".into());
    m.insert("SYNT_TwoSweeps".into(), "false".into());
    let cfg = BlstmConfig::from_legacy(&m, "SYNT").unwrap();
    BlstmNetwork::from_config(cfg).unwrap()
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
fn input_stats_accumulate_across_files() {
    // Two synthetic "files" scored through the net's type-1 forward: the resulting
    // `_InputStatistics` must equal a hand-chained `from_matrix(a)` then
    // `update(from_matrix(b))` on the SAME (cropped-to-input_size) matrices. STRICT
    // bits: pure arithmetic, no libm in the stats merge.
    let mut net = stats_net();
    assert_eq!(net.input_statistics().n, 0);

    // input_size() == 5; feed 5-column sequences so no crop occurs.
    let a = Array2::from_shape_fn((6, 5), |(r, c)| (r * 5 + c) as f64 * 0.5 - 3.0);
    let b = Array2::from_shape_fn((4, 5), |(r, c)| (2 * r + 3 * c) as f64 - 1.25);

    // The scoring forward mutates `input` in place (type-1 normalization). Score
    // each "file" once with no targets so only the stats accumulation runs.
    let mut a_in = a.clone();
    net.feed_forward_scoring(&mut a_in, 0, 0, -1, 0.0);
    let mut b_in = b.clone();
    net.feed_forward_scoring(&mut b_in, 0, 0, -1, 0.0);

    // The stats are computed over the POST-normalization input the forward sees:
    // reproduce the exact type-1 normalization on a copy, then chain.
    let a_normed = type1_normalize(&a, net.normalize_input_mean(), net.normalize_input_std());
    let b_normed = type1_normalize(&b, net.normalize_input_mean(), net.normalize_input_std());
    let mut expected = InputStatistics::from_matrix(&a_normed);
    expected.update(&InputStatistics::from_matrix(&b_normed));

    assert_eq!(net.input_statistics(), &expected);
    assert_eq!(net.input_statistics().n, 10);

    // reset_input_statistics wipes back to n == 0 with empty mean/std.
    net.reset_input_statistics();
    assert_eq!(net.input_statistics().n, 0);
    assert!(net.input_statistics().mean.is_empty());
}

/// Reproduce the type-1 in-place normalization (`BLSTMNeuralNetwork.cpp:720-723`):
/// per column jj < min(cols, mean.len()), (x - mean_jj)/max(1e-12, std_jj).
fn type1_normalize(m: &Array2<f64>, mean: &[f64], std: &[f64]) -> Array2<f64> {
    let mut out = m.clone();
    let (r, cols) = out.dim();
    let max_col = cols.min(mean.len());
    for jj in 0..max_col {
        let denom = 1e-12_f64.max(std[jj]);
        for row in 0..r {
            out[[row, jj]] = (out[[row, jj]] - mean[jj]) / denom;
        }
    }
    out
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
