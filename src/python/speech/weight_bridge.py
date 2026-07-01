"""Neural-net weight (de)serialization: canonical flat vector <-> `.bin`.

Ported from legacy MATLAB: config2network*.m, network2config*.m,
config2weights*.m, nnet2MatFile.m, weights2nnet.m, Write/ReadMatrixFromBinary.m,
CombineNNets.m, ModifyOutputNetwork.m, ForceLSTMBiais.m. See design spec section 6.

High-risk: the column-major flat layout + adim_coeff scaling are the seam to the
Rust engine - round-trip property tests are mandatory when implemented (Phase 3).
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def pack_weights(net: dict[str, object]) -> NDArray[np.float64]:
    """Flatten a network struct into the canonical column-major vector (Phase 3)."""
    ...


def unpack_weights(flat: NDArray[np.float64], arch: dict[str, object]) -> dict[str, object]:
    """Inverse of `pack_weights` (Phase 3)."""
    ...


def write_bin(matrix: NDArray[np.float64], path: Path) -> None:
    """Write i64 LE rows, i64 LE cols, then f64 LE column-major (Phase 0)."""
    ...


def read_bin(path: Path) -> NDArray[np.float64]:
    """Inverse of `write_bin` (Phase 0)."""
    ...
