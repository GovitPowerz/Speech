"""Byte-parity tests for the .bin codec (Python side)."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
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


@given(
    rows=st.integers(min_value=1, max_value=6),
    cols=st.integers(min_value=1, max_value=6),
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=64), min_size=36, max_size=36),
)
def test_write_read_bin_round_trips_bit_exact(rows: int, cols: int, values: list[float], tmp_path_factory: pytest.TempPathFactory) -> None:
    """`read_bin(write_bin(x)) == x` to the bit for any shape and any finite payload: the one
    weight-bridge property the module still owns (the pack LAYOUT is Rust's since issue #27)."""
    data = np.asarray(values[: rows * cols], dtype=np.float64)
    path = tmp_path_factory.mktemp("bin") / "rt.bin"
    weight_bridge.write_bin(rows, cols, data, path)
    got_rows, got_cols, got = weight_bridge.read_bin(path)
    assert (got_rows, got_cols) == (rows, cols)
    assert got.tobytes() == data.tobytes()
