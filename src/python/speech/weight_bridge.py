"""Weight pack `.bin` I/O: the custom little-endian container, read and written.

Ported from legacy MATLAB `Write/ReadMatrixFromBinary.m`. The container is `rows, cols` as two
i64 LE then the payload as f64 LE in column-major order (matching `io::binary.rs`); a weight
pack is an `N x 1` vector, for which the row-/column-major flattenings coincide.

The flat-pack LAYOUT and the `adim_coeff` scaling live once, in Rust: the layout is read
through `speech_rs.weight_layout` (issue #27, `init_weights.py` draws per block from it) and
the adim seam is `config.rs`. The Python packer this module once carried (`nnet_to_flat` /
`flat_to_nnet` / `config_to_nnet`, the Phase 0a oracle) is gone; the Phase 0a / 4d goldens stay
proven by `src/rust/tests/phase0_packer.rs` and `phase4d_packer_golden.rs`.
"""

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_bin(path: Path) -> tuple[int, int, NDArray[np.float64]]:
    """Read the custom .bin matrix; returns (rows, cols, column-major f64 vector)."""
    raw = Path(path).read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    if rows * cols <= 0:
        raise ValueError(f"empty/invalid .bin: rows={rows} cols={cols}")
    data = np.frombuffer(raw[16 : 16 + 8 * rows * cols], dtype="<f8")
    if data.shape[0] != rows * cols:
        raise ValueError("truncated .bin payload")
    return rows, cols, np.array(data, dtype=np.float64)


def write_bin(rows: int, cols: int, data: NDArray[np.float64], path: Path) -> None:
    """Write rows/cols as i64 LE then column-major f64 LE (data is already column-major)."""
    flat = np.ascontiguousarray(data, dtype="<f8").reshape(-1)
    if flat.shape[0] != rows * cols:
        raise ValueError("data length does not match rows*cols")
    with Path(path).open("wb") as fh:
        fh.write(struct.pack("<qq", rows, cols))
        fh.write(flat.tobytes())


def read_weight_vector(path: Path) -> NDArray[np.float64]:
    """Read an N x 1 weight .bin as a flat vector (asserts cols == 1)."""
    rows, cols, data = read_bin(path)
    if cols != 1:
        raise ValueError(f"expected a column vector, got cols={cols}")
    return data
