//! adim + packer byte-parity tests (Rust), against the same fixtures as Python.
//!
//! Rust-pack == Python-pack == committed fixture is the cross-language agreement
//! that substitutes for an independent legacy golden (nnet_best is the CMA-ES
//! genome, not a weight-pack - see scripts/extract_phase0_fixtures.py).

use std::path::PathBuf;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0")
}

fn spec() -> speech::config::NnetSpec {
    let text = std::fs::read_to_string(ref_dir().join("1_worker_1.config")).unwrap();
    let m = speech::legacy_config::parse_legacy_config(&text);
    speech::config::NnetSpec::from_legacy(&m, "BLSTM").unwrap()
}

#[test]
fn element_count_is_33671() {
    assert_eq!(speech::config::element_count(&spec()), 33671);
}

#[test]
fn full_pipeline_matches_flat() {
    let (structured, s) = speech::config::load_structured(
        &ref_dir().join("best_config_manifest.json"),
        &ref_dir().join("best_config_domain.bin"),
    )
    .unwrap();
    let flat = speech::config::nnet_to_flat(&speech::config::config_to_nnet(&structured, &s), &s);
    let golden =
        speech::io::binary::read_weight_vector(&ref_dir().join("best_net_flat.bin")).unwrap();
    assert_eq!(flat, golden); // Rust pack == best_net_flat.bin
}

#[test]
fn packer_bijection_on_real_bin() {
    let s = spec();
    let flat =
        speech::io::binary::read_weight_vector(&ref_dir().join("NNweights_config1.bin")).unwrap();
    let rt = speech::config::nnet_to_flat(&speech::config::flat_to_nnet(&flat, &s), &s);
    assert_eq!(rt, flat);
}

#[test]
fn normalize_tail_anchor() {
    // Both nets share the corpus normalization: their last 46 values (mean[23]+std[23])
    // are identical, and the std tail (last 23) is strictly positive.
    let a = speech::io::binary::read_weight_vector(&ref_dir().join("best_net_flat.bin")).unwrap();
    let b =
        speech::io::binary::read_weight_vector(&ref_dir().join("NNweights_config1.bin")).unwrap();
    assert_eq!(a[a.len() - 46..], b[b.len() - 46..]);
    assert!(a[a.len() - 23..].iter().all(|&x| x > 0.0));
}
