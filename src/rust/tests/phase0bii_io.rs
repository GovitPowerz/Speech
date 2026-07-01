//! VRCTS XML byte-exact round-trip vs real fixtures.

use std::path::PathBuf;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0bii")
}

fn roundtrip(fixture: &str) {
    let raw = std::fs::read_to_string(ref_dir().join(fixture)).unwrap();
    // extract the AudioDoc name + path attributes from the real file to reproduce the header
    let (name, path_attr) = speech::tasks::segmentation_io::parse_audiodoc_attrs(&raw);
    // Load into a Segmentation. The real files use sigdur as the audio duration.
    let seg = speech::tasks::segmentation_io::load_vrcts(&raw, 0.0, 1.0e9);
    let out = speech::tasks::segmentation_io::to_vrcts_string(&seg, &name, &path_attr);
    assert_eq!(out, raw, "round-trip mismatch for {fixture}");
}

#[test]
fn vrcts_roundtrip_1seg() {
    roundtrip("vrcts_1seg.xml");
}
#[test]
fn vrcts_roundtrip_empty() {
    roundtrip("vrcts_empty.xml");
}
#[test]
fn vrcts_roundtrip_16seg() {
    roundtrip("vrcts_16seg.xml");
}
