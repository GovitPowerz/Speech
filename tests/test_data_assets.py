"""Assert the committed data assets are well-formed."""

import struct
from pathlib import Path

N = 100000
DATA = Path("data/random_tables.bin")


def test_random_tables_size() -> None:
    assert DATA.stat().st_size == 2 * N * 8


def test_random_tables_ranges() -> None:
    raw = DATA.read_bytes()
    values = struct.unpack(f"<{2 * N}d", raw)
    gauss = values[:N]
    uniform = values[N:]
    assert any(v < 0.0 for v in gauss[:1000])
    assert all(0.0 <= v < 1.0 for v in uniform[:1000])
