//! Task 3: NN lifecycle surface goldens.
//!
//! Covers `BlstmNetwork` InputStatistics accumulation across files (STRICT bits vs a
//! hand-chained `from_matrix`+`update`), `save_weights` three-artifact write
//! (`weights_<f>`/`weightsDerivatives_<f>` .bin + the `<f>` .mat stats), and the
//! `<prefix>_weightsFile` load semantics (empty skip / too-few error / too-many
//! truncate) transcribed from `BLSTMNeuralNetwork.cpp:122-150,312-331,385-417`.

use std::path::PathBuf;
use std::sync::Mutex;

use indexmap::IndexMap;
use ndarray::Array2;

use speech::features::stats::InputStatistics;
use speech::io::binary::{read_matrix, write_matrix};
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

/// `save_weights`'s `weights_`/`weightsDerivatives_` prefix glues onto the WHOLE
/// filename string (`:319-321`), so its production shape is a BARE relative
/// filename with cwd == the run/output dir -- the `.bin` siblings land next to it
/// with no path-separator awareness. Exercising that verbatim needs a real chdir
/// (an absolute tempdir path would materialize `weights_/<abs...>` trees under
/// whatever the glued prefix resolves to, polluting the cargo CWD). `set_current_dir`
/// is process-global; serialize against `cargo test`'s parallel threads.
static CWD_LOCK: Mutex<()> = Mutex::new(());

struct CwdGuard {
    original: PathBuf,
}

impl CwdGuard {
    fn enter(dir: &std::path::Path) -> CwdGuard {
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir).unwrap();
        CwdGuard { original }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original);
    }
}

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
fn save_weights_writes_three_artifacts() {
    let (_m, mut net) = real_net_with_weights_file("");
    let flat =
        speech::io::binary::read_weight_vector(&ref_dir().join("NNweights_config1.bin")).unwrap();
    net.set_weights(&flat).unwrap();

    // Craft a non-trivial stats object + derivs so the artifacts are non-vacuous.
    let stats =
        InputStatistics::from_matrix(&Array2::from_shape_fn((4, net.input_size()), |(r, c)| {
            (r * net.input_size() + c) as f64 - 2.0
        }));
    let n = net.nb_of_weights();
    let derivs = Array2::from_shape_fn((n, 2), |(r, c)| if c == 1 { 1.0 } else { r as f64 * 0.01 });

    // The legacy glues the `weights_`/`weightsDerivatives_` PREFIX to the WHOLE
    // filename string (`:319-321`), NOT the basename. Its production shape is a bare
    // relative filename with cwd == the run/output dir, so the `.bin` siblings land
    // next to `<filename>` in that same dir. Reproduce that shape exactly: chdir into
    // a tempdir and pass a bare filename, instead of routing an absolute tempdir path
    // through the quirk (which glues the prefix ahead of the leading '/' and
    // materializes a `weights_<abs...>` tree under the cargo CWD).
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let _cwd = CwdGuard::enter(dir.path());

    let filename = "epoch1.mat";
    let weights_path = PathBuf::from(format!("weights_{filename}"));
    let derivs_path = PathBuf::from(format!("weightsDerivatives_{filename}"));
    let mat_path = PathBuf::from(filename);

    net.save_weights(filename, &derivs, &stats).unwrap();

    // 1) weights_<f>.mat is the flat weight vector as an N x 1 .bin (custom codec).
    let (wr, wc, wdata) = read_matrix(&weights_path).unwrap();
    assert_eq!((wr, wc), (n, 1));
    assert_eq!(wdata, net.get_weights());

    // 2) weightsDerivatives_<f>.mat is the N x 2 derivs .bin (column-major payload).
    let (dr, dc, ddata) = read_matrix(&derivs_path).unwrap();
    assert_eq!((dr, dc), (n, 2));
    // Column-major: all of col0 then col1. Reproduce that ordering from `derivs`.
    let mut expected_d = Vec::with_capacity(n * 2);
    for c in 0..2 {
        for r in 0..n {
            expected_d.push(derivs[[r, c]]);
        }
    }
    assert_eq!(ddata, expected_d);

    // 3) <f>.mat is a structurally valid MAT v5 file carrying nbOfInputs (1x1),
    // meanInputs (1 x D), stdInputs (1 x D).
    let bytes = std::fs::read(&mat_path).unwrap();
    assert!(bytes.len() >= 128);
    assert_eq!(&bytes[0..4], b"MATL");
    let version = u16::from_le_bytes(bytes[124..126].try_into().unwrap());
    assert_eq!(version, 0x0100);

    let d = net.input_size();
    // First variable: nbOfInputs, 1x1, value == stats.n.
    let mut off = 128;
    assert_eq!(u32_le(&bytes, off), 14, "miMATRIX");
    let first_size = u32_le(&bytes, off + 4) as usize;
    let mut cur = off + 8 + 16; // tag + array flags
    assert_eq!(i32_le(&bytes, cur + 8), 1, "nbOfInputs rows");
    assert_eq!(i32_le(&bytes, cur + 12), 1, "nbOfInputs cols");
    cur += 16 + 8; // dims + name tag/size
    assert_eq!(
        &bytes[cur..cur + 8],
        b"nbOfInpu"[..].get(..8).unwrap_or(&bytes[cur..cur + 8])
    );
    // name "nbOfInputs" is 10 bytes -> padded to 16.
    let name0 = &bytes[cur..cur + 10];
    assert_eq!(name0, b"nbOfInputs");
    cur += 16; // padded name
    assert_eq!(u32_le(&bytes, cur), 9, "miDOUBLE");
    cur += 8;
    assert_eq!(f64_le(&bytes, cur), stats.n as f64);

    // Second variable: meanInputs, 1 x D.
    off += 8 + first_size;
    assert_eq!(u32_le(&bytes, off), 14, "second miMATRIX");
    let second_size = u32_le(&bytes, off + 4) as usize;
    let mut cur2 = off + 8 + 16;
    assert_eq!(i32_le(&bytes, cur2 + 8), 1, "meanInputs rows");
    assert_eq!(i32_le(&bytes, cur2 + 12), d as i32, "meanInputs cols == D");
    cur2 += 16 + 8;
    assert_eq!(&bytes[cur2..cur2 + 10], b"meanInputs");

    // Third variable: stdInputs, 1 x D.
    off += 8 + second_size;
    assert_eq!(u32_le(&bytes, off), 14, "third miMATRIX");
    let mut cur3 = off + 8 + 16;
    assert_eq!(i32_le(&bytes, cur3 + 8), 1, "stdInputs rows");
    assert_eq!(i32_le(&bytes, cur3 + 12), d as i32, "stdInputs cols == D");
    cur3 += 16 + 8;
    assert_eq!(&bytes[cur3..cur3 + 9], b"stdInputs");
}

fn u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn i32_le(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn f64_le(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
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
