//! Shared test infrastructure for golden-fixture tests (Phase 1+).

use std::path::PathBuf;

use ndarray::{Array2, ShapeBuilder};

/// Absolute path to a file under `tests/reference_data/phase1/`.
pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase1")
        .join(name)
}

/// Load a phase1 `.bin` fixture (column-major f64) into an `Array2<f64>` (rows x cols).
pub fn load_bin(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture(name)).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

/// Elementwise bit-exact comparison; reports the first mismatch index + hex bits.
pub fn assert_bits_eq(a: &Array2<f64>, b: &Array2<f64>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape mismatch");
    for ((idx, av), bv) in a.indexed_iter().zip(b.iter()) {
        assert!(
            av.to_bits() == bv.to_bits(),
            "{label}: mismatch at {idx:?}: a=0x{:016x} ({av}) b=0x{:016x} ({bv})",
            av.to_bits(),
            bv.to_bits()
        );
    }
}
