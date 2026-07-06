//! Byte-structure tests for the uncompressed MAT v5 writer (Task 1).
//!
//! Layout asserted here mirrors the MathWorks MAT-File Format spec, cross-checked
//! against a real `scipy.io.savemat(..., format="5")` output (long name-field form).

use std::path::PathBuf;

use speech::io::matfile::MatWriter;

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("speech_matfile_{name}"))
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
fn header_is_v5() {
    let path = tmp_path("header.mat");
    let mut w = MatWriter::create(&path).unwrap();
    w.write_scalar("s", 1.0).unwrap();
    w.finish().unwrap();

    let data = std::fs::read(&path).unwrap();
    assert!(data.len() >= 128);
    assert_eq!(&data[0..4], b"MATL");
    let version = u16::from_le_bytes(data[124..126].try_into().unwrap());
    assert_eq!(version, 0x0100);
    assert_eq!(&data[126..128], b"IM");
}

#[test]
fn matrix_element_layout() {
    let path = tmp_path("matrix.mat");
    let mut w = MatWriter::create(&path).unwrap();
    // Column-major 2x3: col0=[1,4], col1=[2,5], col2=[3,6].
    let data_col_major = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    w.write_matrix("A", 2, 3, &data_col_major).unwrap();
    w.finish().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let mut off = 128;

    // miMATRIX tag.
    assert_eq!(u32_le(&bytes, off), 14, "miMATRIX type");
    let matrix_size = u32_le(&bytes, off + 4) as usize;
    off += 8;

    // Array flags sub-element: miUINT32(6), size 8, class byte = mxDOUBLE_CLASS(6).
    assert_eq!(u32_le(&bytes, off), 6, "array flags miUINT32 tag");
    assert_eq!(u32_le(&bytes, off + 4), 8, "array flags size");
    assert_eq!(bytes[off + 8], 6, "mxDOUBLE_CLASS");
    assert_eq!(bytes[off + 9], 0, "flags byte");
    off += 8 + 8;

    // Dimensions sub-element: miINT32(5), size 8, [2, 3].
    assert_eq!(u32_le(&bytes, off), 5, "dims miINT32 tag");
    assert_eq!(u32_le(&bytes, off + 4), 8, "dims size");
    assert_eq!(i32_le(&bytes, off + 8), 2, "rows");
    assert_eq!(i32_le(&bytes, off + 12), 3, "cols");
    off += 8 + 8;

    // Array name sub-element: miINT8(1), size 1 ("A"), body padded to 8 bytes.
    assert_eq!(u32_le(&bytes, off), 1, "name miINT8 tag");
    assert_eq!(u32_le(&bytes, off + 4), 1, "name size");
    assert_eq!(&bytes[off + 8..off + 9], b"A");
    off += 8 + 8; // name body padded 1 -> 8 bytes.

    // Real part sub-element: miDOUBLE(9), size 48 (6 * 8 bytes), column-major.
    assert_eq!(u32_le(&bytes, off), 9, "data miDOUBLE tag");
    assert_eq!(u32_le(&bytes, off + 4), 48, "data size");
    off += 8;
    let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    for (k, &want) in expected.iter().enumerate() {
        assert_eq!(f64_le(&bytes, off + k * 8), want, "data[{k}]");
    }
    off += 48;

    // miMATRIX size field covers everything after the miMATRIX tag itself.
    assert_eq!(off - 136, matrix_size, "miMATRIX size == payload bytes");
    // Every sub-element boundary (and the whole element) must be 8-byte aligned.
    assert_eq!(off % 8, 0);
    assert_eq!(bytes.len() % 8, 0);
}

#[test]
fn scalar_is_1x1() {
    let path = tmp_path("scalar.mat");
    let mut w = MatWriter::create(&path).unwrap();
    w.write_scalar("s", 1.5).unwrap();
    w.finish().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let mut off = 128;
    off += 8; // miMATRIX tag
    off += 16; // array flags sub-element
    assert_eq!(u32_le(&bytes, off), 5, "dims tag");
    assert_eq!(u32_le(&bytes, off + 4), 8, "dims size");
    assert_eq!(i32_le(&bytes, off + 8), 1, "rows == 1");
    assert_eq!(i32_le(&bytes, off + 12), 1, "cols == 1");
    off += 16;
    off += 8; // name tag+size
    assert_eq!(&bytes[off..off + 1], b"s");
    off += 8; // name body padded 1 -> 8
    assert_eq!(u32_le(&bytes, off), 9, "miDOUBLE tag");
    assert_eq!(u32_le(&bytes, off + 4), 8, "one f64");
    off += 8;
    assert_eq!(f64_le(&bytes, off), 1.5);
}

#[test]
fn two_variables_sequential() {
    let path = tmp_path("two_vars.mat");
    let mut w = MatWriter::create(&path).unwrap();
    w.write_matrix("A", 2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    w.write_scalar("s", 9.0).unwrap();
    w.finish().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let first_size = u32_le(&bytes, 128 + 4) as usize;
    let first_elem_total = 8 + first_size; // tag + payload, no padding needed since payload is 8-aligned
    assert_eq!(first_elem_total % 8, 0, "first element ends 8-aligned");

    let second_off = 128 + first_elem_total;
    assert_eq!(u32_le(&bytes, second_off), 14, "second miMATRIX tag");
    let second_size = u32_le(&bytes, second_off + 4) as usize;

    // Walk into the second element's dims to confirm it's the 1x1 scalar "s".
    let mut off = second_off + 8 + 16; // skip miMATRIX tag + array flags
    assert_eq!(i32_le(&bytes, off + 8), 1);
    assert_eq!(i32_le(&bytes, off + 12), 1);
    off += 16 + 8; // dims + name tag/size
    assert_eq!(&bytes[off..off + 1], b"s");
    off += 8;
    assert_eq!(u32_le(&bytes, off + 4), 8);
    off += 8;
    assert_eq!(f64_le(&bytes, off), 9.0);

    assert_eq!(bytes.len(), second_off + 8 + second_size);
}

/// Generates the committed sample fixture consumed by `tests/test_phase4a_matfile.py`.
/// Run explicitly: `cargo test --test phase4a_matfile -- --ignored write_sample`.
#[test]
#[ignore]
fn write_sample() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4a/matwriter_sample.mat");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut w = MatWriter::create(&path).unwrap();
    w.write_matrix("A", 2, 3, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        .unwrap();
    w.write_scalar("s", 1.5).unwrap();
    w.finish().unwrap();
}
