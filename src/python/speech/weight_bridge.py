"""Neural-net weight (de)serialization: canonical flat vector <-> `.bin`.

Ported from legacy MATLAB: config2network*.m, network2config*.m,
config2weights*.m, nnet2MatFile.m, weights2nnet.m, Write/ReadMatrixFromBinary.m,
CombineNNets.m, ModifyOutputNetwork.m, ForceLSTMBiais.m. See design spec section 6.

High-risk: the column-major flat layout + adim_coeff scaling are the seam to the
Rust engine - round-trip property tests are mandatory when implemented (Phase 3).
"""

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def pack_weights(net: dict[str, object]) -> NDArray[np.float64]:
    """Flatten a network struct into the canonical column-major vector (Phase 3)."""
    ...


def unpack_weights(flat: NDArray[np.float64], arch: dict[str, object]) -> dict[str, object]:
    """Inverse of `pack_weights` (Phase 3)."""
    ...


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
