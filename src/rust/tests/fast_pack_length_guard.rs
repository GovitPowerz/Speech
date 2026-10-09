//! Issue #62: the fast SAD driver refuses exactly what the exact net refuses, in the same
//! words -- the fast half of `set_weights_length_guard.rs`'s adjudication.
//!
//! An IN-MEMORY pack (`FastSpectralSegmenter::set_weights`, which `BagOfProcessors::
//! set_weights` drives from Python, and the ctor's `Some(flat)` arm) is exact-length: one
//! element long or short is refused with `BlstmNetwork::set_weights`' message. A FILE pack
//! (`load_weights_file`) keeps the legacy head-first tolerance and refuses a short file
//! with `BlstmNetwork::load_weights_file`'s message. Every (cell, direction) pair is
//! covered: the nine phase-9 fixtures plus the phase-7 tier-2 config for the
//! bidirectional LSTM. The Twin's legs live in `fast_twin_lid_matrix.rs`, beside the
//! scoring they need.

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::fast::driver::FastSpectralSegmenter;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

fn map_at(rel: &str) -> IndexMap<String, String> {
    let path = ref_dir().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    speech::legacy_config::parse_legacy_config(&text)
}

const CONFIGS: [&str; 10] = [
    "phase4a/tier2_spectral.config",
    "phase9/lstm_forward.config",
    "phase9/slstm_forward.config",
    "phase9/slstm_bidirectional.config",
    "phase9/mamba_forward.config",
    "phase9/mamba_bidirectional.config",
    "phase9/cfc_forward.config",
    "phase9/cfc_bidirectional.config",
    "phase9/transformer_forward.config",
    "phase9/transformer_bidirectional.config",
];

fn exact_net(map: &IndexMap<String, String>) -> BlstmNetwork {
    BlstmNetwork::from_config(BlstmConfig::from_legacy(map, "BLSTM").unwrap()).unwrap()
}

/// Distinct, non-degenerate values (the guard is length arithmetic, indifferent to them).
fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + (i as f64) * 1e-3).collect()
}

#[test]
fn the_in_memory_seams_refuse_what_the_exact_net_refuses() {
    for rel in CONFIGS {
        let map = map_at(rel);
        let mut exact = exact_net(&map);
        let n = exact.nb_of_weights();
        let mut fast = FastSpectralSegmenter::from_legacy(&map, None)
            .unwrap_or_else(|e| panic!("{rel}: the fast driver must build: {e}"));
        fast.set_weights(&ramp(n))
            .unwrap_or_else(|e| panic!("{rel}: the exact length must load: {e}"));
        FastSpectralSegmenter::from_legacy(&map, Some(&ramp(n)))
            .unwrap_or_else(|e| panic!("{rel}: the exact length must build: {e}"));

        for len in [n + 1, n - 1] {
            let want = exact.set_weights(&ramp(len)).unwrap_err().to_string();
            let got = fast
                .set_weights(&ramp(len))
                .err()
                .unwrap_or_else(|| panic!("{rel}: set_weights must refuse {len} (needs {n})"))
                .to_string();
            assert_eq!(got, want, "{rel}: set_weights at {len}");
            let got = FastSpectralSegmenter::from_legacy(&map, Some(&ramp(len)))
                .err()
                .unwrap_or_else(|| panic!("{rel}: the ctor must refuse {len} (needs {n})"))
                .to_string();
            assert_eq!(got, want, "{rel}: from_legacy at {len}");
        }
    }
}

/// The FILE seam, on committed packs of the neighbouring architecture: the sLSTM net
/// (1603) loads the head of the LSTM's 1651-element file, and the LSTM net (1651) refuses
/// the sLSTM's 1603-element file in the exact `load_weights_file`'s words. The file load
/// hands its head to the strict `set_weights`, so a load that stopped slicing it would
/// fail the first leg.
#[test]
fn the_file_seam_keeps_the_legacy_head_first_tolerance() {
    let file = |name: &str| ref_dir().join(name).to_string_lossy().into_owned();

    let mut over = map_at("phase9/slstm_forward.config");
    over.insert(
        "BLSTM_weightsFile".into(),
        file("phase9/lstm_forward_seed.bin"),
    );
    FastSpectralSegmenter::from_legacy(&over, None)
        .unwrap()
        .load_weights_file(&over)
        .expect("the legacy tolerance: an over-long weightsFile loads its head");

    let mut short = map_at("phase9/lstm_forward.config");
    short.insert(
        "BLSTM_weightsFile".into(),
        file("phase9/slstm_forward_seed.bin"),
    );
    let got = FastSpectralSegmenter::from_legacy(&short, None)
        .unwrap()
        .load_weights_file(&short)
        .expect_err("a short weightsFile must be refused")
        .to_string();
    let want = exact_net(&short)
        .load_weights_file(&short, "BLSTM")
        .expect_err("the exact net refuses it too")
        .to_string();
    assert_eq!(got, want);
}
