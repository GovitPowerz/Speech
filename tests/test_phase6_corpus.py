"""Phase 6: corpus-gated smoke for the cep layout + the shared corpus-gate helper.

Independent (pure-Python) confirmation of the `.plp8f0mvsdd` byte layout the Rust
`read_cep` reader (`src/rust/src/audio.rs`, `AudioStruct.cpp:183-256`) ports, and the
first consumer of the `tests/conftest.py` corpus-gate helper (`CORPUS_ROOT` +
`requires_corpus`) that every later Phase 6 task reuses. Skips cleanly when the licensed
LRE03/07 corpus is absent (e.g. in CI).
"""

import pathlib
import struct

import numpy as np
import pytest

from tests.conftest import CORPUS_ROOT, requires_corpus


def _first_cep_file() -> pathlib.Path | None:
    files = sorted((CORPUS_ROOT / "train" / "LID_Features").rglob("*.plp8f0mvsdd"))
    return files[0] if files else None


@pytest.mark.corpus
@requires_corpus
def test_cep_layout_byte_arithmetic() -> None:
    path = _first_cep_file()
    if path is None:
        pytest.skip("no .plp8f0mvsdd under train/LID_Features")
    b = path.read_bytes()

    # Header: int32 nbRecords | int16 vectorSize | int16 magic (little-endian).
    assert len(b) >= 8
    nb_records = struct.unpack_from("<i", b, 0)[0]
    vector_size = struct.unpack_from("<h", b, 4)[0]
    assert nb_records > 0
    assert vector_size == 23, "LRE cep vectorSize == NNetInputSize (23)"

    # File size == header arithmetic exactly (strict, independent of the Rust reader).
    expected_floats = 0
    for m in range(nb_records):
        vnb = struct.unpack_from("<i", b, 8 + 4 * m)[0]
        if vnb > 0:
            expected_floats += vnb * vector_size
    payload_off = 8 + 4 * nb_records
    assert len(b) == payload_off + 4 * expected_floats

    # float32 payload is finite, plausible-magnitude, not all-zero.
    payload = np.frombuffer(b, dtype="<f4", count=expected_floats, offset=payload_off)
    assert np.isfinite(payload).all()
    assert not (payload == 0).all()
    assert np.abs(payload).max() < 100.0
