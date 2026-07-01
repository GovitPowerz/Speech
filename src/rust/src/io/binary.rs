//! Weight `.bin` codec: i64 LE rows, i64 LE cols, then f64 LE column-major.
//!
//! Ported from legacy C++: Helpers.hpp. The MATLAB/Python seam - implemented and
//! round-trip golden-tested in Phase 0.

use std::path::Path;

/// Read a column-major f64 matrix from the legacy `.bin` layout.
pub fn read_matrix(path: &Path) -> anyhow::Result<(usize, usize, Vec<f64>)> {
    todo!("Phase 0: port BinaryFile2Vector")
}
