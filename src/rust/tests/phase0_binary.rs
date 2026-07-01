//! Byte-parity tests for the .bin codec (Rust side).

use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/NNweights_config1.bin")
}

#[test]
fn header_and_size() {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture()).unwrap();
    assert_eq!((rows, cols), (33671, 1));
    assert_eq!(data.len(), 33671);
    let size = std::fs::metadata(fixture()).unwrap().len();
    assert_eq!(size, 16 + 8 * rows as u64 * cols as u64);
}

#[test]
fn roundtrip_bytes() {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture()).unwrap();
    let tmp = std::env::temp_dir().join("speech_rt_config1.bin");
    speech::io::binary::write_matrix(&tmp, rows, cols, &data).unwrap();
    assert_eq!(
        std::fs::read(&tmp).unwrap(),
        std::fs::read(fixture()).unwrap()
    );
}

#[test]
fn empty_guard() {
    let tmp = std::env::temp_dir().join("speech_empty.bin");
    std::fs::write(&tmp, [0u8; 16]).unwrap(); // rows=0, cols=0
    assert!(speech::io::binary::read_matrix(&tmp).is_err());
}
