//! `.mat` v5 write for diagnostics and legacy parity.
//!
//! Ported from legacy C++: Helpers.hpp (Matrix2MatFile). Writes uncompressed MAT v5
//! (miMATRIX / mxDOUBLE_CLASS, real part only, column-major f64) via a fixed 128-byte
//! header, so output is byte-reproducible run to run (no timestamp).
//!
//! Parity contract is VALUE-level, not byte-level: the legacy writes zlib-compressed
//! MAT-files through matio (`Helpers.hpp:450,469`); matching that compression byte-for-byte
//! is out of scope by design (spec S6). No `.mat` reader is provided here either (S6).

use std::io::Write;
use std::path::Path;

use anyhow::bail;
use byteorder::{LittleEndian, WriteBytesExt};

const HEADER_TEXT: &[u8] = b"MATLAB 5.0 MAT-file, written by speech-rs";
const MI_MATRIX: u32 = 14;
const MI_UINT32: u32 = 6;
const MI_INT32: u32 = 5;
const MI_INT8: u32 = 1;
const MI_DOUBLE: u32 = 9;
const MX_DOUBLE_CLASS: u8 = 6;

/// Pad `len` up to the next multiple of 8.
fn padded_len(len: usize) -> usize {
    len.div_ceil(8) * 8
}

/// Sequential uncompressed MAT v5 writer: one `write_matrix`/`write_scalar` call per
/// variable, in the order variables should appear in the file.
pub struct MatWriter {
    file: std::io::BufWriter<std::fs::File>,
}

impl MatWriter {
    /// Opens `path` and writes the fixed 128-byte MAT v5 header.
    pub fn create(path: &Path) -> anyhow::Result<MatWriter> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);

        // 116-byte text header, space-padded.
        let mut header = [b' '; 116];
        header[..HEADER_TEXT.len()].copy_from_slice(HEADER_TEXT);
        file.write_all(&header)?;
        // 8-byte subsystem data offset: all-zero (unused).
        file.write_all(&[0u8; 8])?;
        // Version 0x0100, LE.
        file.write_u16::<LittleEndian>(0x0100)?;
        // Endian indicator.
        file.write_all(b"IM")?;

        Ok(MatWriter { file })
    }

    /// Writes one real-valued double-precision matrix as a miMATRIX element.
    /// `data_col_major` must be column-major, length `rows * cols`.
    pub fn write_matrix(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        data_col_major: &[f64],
    ) -> anyhow::Result<()> {
        if data_col_major.len() != rows * cols {
            bail!(
                "data length {} != rows*cols {}",
                data_col_major.len(),
                rows * cols
            );
        }

        // Sub-element sizes (unpadded payload bytes).
        let flags_size = 8u32;
        let dims_size = 8u32;
        let name_size = name.len() as u32;
        let data_size = (data_col_major.len() * 8) as u32;

        let payload_size = 8
            + padded_len(flags_size as usize)
            + 8
            + padded_len(dims_size as usize)
            + 8
            + padded_len(name_size as usize)
            + 8
            + padded_len(data_size as usize);

        self.file.write_u32::<LittleEndian>(MI_MATRIX)?;
        self.file.write_u32::<LittleEndian>(payload_size as u32)?;

        // Array flags: miUINT32, 8 bytes -> [class byte, flags byte, 2 unused bytes, 4 unused bytes].
        self.file.write_u32::<LittleEndian>(MI_UINT32)?;
        self.file.write_u32::<LittleEndian>(flags_size)?;
        self.file.write_all(&[MX_DOUBLE_CLASS, 0, 0, 0])?;
        self.file.write_all(&[0, 0, 0, 0])?;

        // Dimensions: miINT32, [rows, cols].
        self.file.write_u32::<LittleEndian>(MI_INT32)?;
        self.file.write_u32::<LittleEndian>(dims_size)?;
        self.file.write_i32::<LittleEndian>(rows as i32)?;
        self.file.write_i32::<LittleEndian>(cols as i32)?;

        // Array name: miINT8, long format unconditionally.
        self.file.write_u32::<LittleEndian>(MI_INT8)?;
        self.file.write_u32::<LittleEndian>(name_size)?;
        self.file.write_all(name.as_bytes())?;
        let name_pad = padded_len(name_size as usize) - name_size as usize;
        self.file.write_all(&vec![0u8; name_pad])?;

        // Real part: miDOUBLE, column-major.
        self.file.write_u32::<LittleEndian>(MI_DOUBLE)?;
        self.file.write_u32::<LittleEndian>(data_size)?;
        for &x in data_col_major {
            self.file.write_f64::<LittleEndian>(x)?;
        }
        let data_pad = padded_len(data_size as usize) - data_size as usize;
        self.file.write_all(&vec![0u8; data_pad])?;

        Ok(())
    }

    /// Writes a scalar as a 1x1 matrix.
    pub fn write_scalar(&mut self, name: &str, value: f64) -> anyhow::Result<()> {
        self.write_matrix(name, 1, 1, &[value])
    }

    /// Flushes and closes the file.
    pub fn finish(mut self) -> anyhow::Result<()> {
        self.file.flush()?;
        Ok(())
    }
}
