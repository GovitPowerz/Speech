//! Weight .bin codec: i64 LE rows, i64 LE cols, then column-major f64 LE payload.
//!
//! Ported from legacy C++: Helpers.hpp (BinaryFile2Vector / Matrix2BinaryFile).
//! The MATLAB/Python seam. Some .mat-named weight files are actually this format.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, bail};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Read the custom .bin matrix; returns (rows, cols, column-major f64 data).
pub fn read_matrix(path: &Path) -> anyhow::Result<(usize, usize, Vec<f64>)> {
    let mut fh = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let rows = fh.read_i64::<LittleEndian>()?;
    let cols = fh.read_i64::<LittleEndian>()?;
    if rows <= 0 || cols <= 0 {
        bail!("empty/invalid .bin: rows={rows} cols={cols}");
    }
    let n = (rows as usize)
        .checked_mul(cols as usize)
        .context("rows*cols overflow")?;
    let mut data = vec![0.0_f64; n];
    fh.read_f64_into::<LittleEndian>(&mut data)?;
    Ok((rows as usize, cols as usize, data))
}

/// Write rows/cols as i64 LE then column-major f64 LE (data is already column-major).
pub fn write_matrix(path: &Path, rows: usize, cols: usize, data: &[f64]) -> anyhow::Result<()> {
    if data.len() != rows * cols {
        bail!("data length {} != rows*cols {}", data.len(), rows * cols);
    }
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create `{}`", path.display()))?;
    let mut fh = std::io::BufWriter::new(file);
    fh.write_i64::<LittleEndian>(rows as i64)?;
    fh.write_i64::<LittleEndian>(cols as i64)?;
    for &x in data {
        fh.write_f64::<LittleEndian>(x)?;
    }
    fh.flush()?;
    Ok(())
}

/// Read an N x 1 weight .bin as a flat vector (asserts cols == 1).
pub fn read_weight_vector(path: &Path) -> anyhow::Result<Vec<f64>> {
    let (_rows, cols, data) = read_matrix(path)?;
    if cols != 1 {
        bail!("expected a column vector, got cols={cols}");
    }
    Ok(data)
}
