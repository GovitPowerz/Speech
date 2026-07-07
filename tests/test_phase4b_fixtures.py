"""Phase 4b Task 1 LID confusion-core fixture guards.

The extractor (`scripts/extract_phase4b_fixtures.py`) runs the oracle harness's
`phase4b_confusion` stage: a crafted 4-row x 3-class `results_lid` matrix, the
harness's own transcription of `BagOfProcessors::PrintConfusionMatrix`'s
accumulation loop, and a 1x2 `[error1, error2]` pair cross-validating that
transcription against the REAL compiled `Confusion2String` (error1) and the
REAL compiled `PrintConfusionMatrix` end-to-end (error2). These tests guard the
committed fixtures WITHOUT the harness: the manifest is present and records the
measured shapes/values, every referenced `.bin` exists and parses to its
recorded shape, and error1/error2 are bit-identical (the non-vacuous
cross-check the harness itself already asserted before these fixtures were
committed).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

PHASE4B = Path("tests/reference_data/phase4b")
MANIFEST = PHASE4B / "manifest.json"

CONFUSION_BINS = ["confusion_input.bin", "confusion_matrix.bin", "confusion_error.bin"]


def _read_bin(path: Path) -> tuple[int, int, list[float]]:
    """Read an io::binary `.bin` (i64 LE rows, i64 LE cols, f64 LE column-major)."""
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    n = rows * cols
    data = list(struct.unpack(f"<{n}d", raw[16 : 16 + 8 * n]))
    return rows, cols, data


def _load_manifest() -> dict[str, object]:
    return cast(dict[str, object], json.loads(MANIFEST.read_text()))


def test_manifest_present() -> None:
    assert MANIFEST.is_file(), "phase4b manifest.json must be committed"
    manifest = _load_manifest()
    assert "confusion_input" in manifest
    assert "confusion_matrix" in manifest
    assert "confusion_error" in manifest
    assert "dead_code_not_ported" in manifest


def test_every_bin_exists_and_matches_manifest_shape() -> None:
    manifest = _load_manifest()
    shapes = {
        "confusion_input.bin": cast(list[int], cast(dict[str, object], manifest["confusion_input"])["shape"]),
        "confusion_matrix.bin": cast(list[int], cast(dict[str, object], manifest["confusion_matrix"])["shape"]),
        "confusion_error.bin": cast(list[int], cast(dict[str, object], manifest["confusion_error"])["shape"]),
    }
    for name in CONFUSION_BINS:
        path = PHASE4B / name
        assert path.is_file(), f"{path} must exist"
        rows, cols, _ = _read_bin(path)
        assert (rows, cols) == tuple(shapes[name]), f"{name}: shape ({rows},{cols}) != manifest {shapes[name]}"


def test_input_shape_is_4x3_three_class() -> None:
    rows, cols, _ = _read_bin(PHASE4B / "confusion_input.bin")
    assert (rows, cols) == (4, 3), "4 crafted rows x 3 classes"


def test_matrix_shape_is_classnb_plus_2_square() -> None:
    in_rows, in_cols, _ = _read_bin(PHASE4B / "confusion_input.bin")
    mat_rows, mat_cols, _ = _read_bin(PHASE4B / "confusion_matrix.bin")
    assert mat_rows == mat_cols == in_cols + 2, "(classNb+2) x (classNb+2)"


def test_error1_equals_error2_bit_exact() -> None:
    """The non-vacuity the harness itself gated on (SystemExit if false):
    the transcribed matrix's error via the REAL free Confusion2String must
    equal the REAL PrintConfusionMatrix's own return, called end-to-end on
    the identical crafted input."""
    rows, cols, data = _read_bin(PHASE4B / "confusion_error.bin")
    assert (rows, cols) == (1, 2)
    error1, error2 = data
    assert error1 == error2, f"error1={error1!r} != error2={error2!r}"
    manifest = _load_manifest()
    conf_err = cast(dict[str, object], manifest["confusion_error"])
    assert cast(float, conf_err["error1"]) == error1
    assert cast(float, conf_err["error2"]) == error2


def test_error_is_non_vacuous() -> None:
    """The crafted rows include real misclassifications (row B loses, row D
    ties, row C has no target sentinel), so the error aggregate must be
    strictly positive, not a vacuous 0.0 from an all-correct matrix."""
    _, _, data = _read_bin(PHASE4B / "confusion_error.bin")
    assert data[0] > 0.0
    assert data[0] == 200.0 / 3.0, "hand-derivable exact value (see manifest + Rust golden doc)"
