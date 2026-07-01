"""Byte-parity tests for the .bin codec (Python side)."""

from pathlib import Path

import numpy as np
from speech import weight_bridge

FIX = Path("tests/reference_data/phase0/NNweights_config1.bin")


def test_header_and_size() -> None:
    rows, cols, data = weight_bridge.read_bin(FIX)
    assert (rows, cols) == (33671, 1)
    assert data.shape == (33671,)
    assert FIX.stat().st_size == 16 + 8 * rows * cols


def test_roundtrip_bytes(tmp_path: Path) -> None:
    rows, cols, data = weight_bridge.read_bin(FIX)
    out = tmp_path / "rt.bin"
    weight_bridge.write_bin(rows, cols, data, out)
    assert out.read_bytes() == FIX.read_bytes()


def test_read_weight_vector() -> None:
    v = weight_bridge.read_weight_vector(FIX)
    assert v.shape == (33671,)
    assert v.dtype == np.float64
