"""Phase 4a Task 8 tier-1 fixture guards.

The extractor (`scripts/extract_phase4a_fixtures.py`) runs the REAL compiled
CorpusProcessor stack and converts each `.mat` output to per-variable `.bin`
(io::binary column-major), masking the MultiConfigResults timing column. These tests
guard the committed fixtures WITHOUT the harness: the manifest is present and records
`masked_cols`, every referenced `.bin` exists and parses to its recorded shape, and
scipy re-reading the RAW `.mat` outputs reproduces the converted `.bin` values (with
the same col-6 mask) -- guarding the conversion itself.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

import numpy as np
import scipy.io

PHASE4A = Path("tests/reference_data/phase4a")
MANIFEST = PHASE4A / "manifest.json"

MAT_VARS = [
    "MultiConfigResults",
    "CostMem",
    "BadClassifMem",
    "CostLIDMem",
    "BadClassifLIDMem",
]
RUNS = ["solo_tdc", "train_ltsv", "multiconfig"]


def _read_bin(path: Path) -> np.ndarray:
    """Read an io::binary `.bin` (i64 LE rows, i64 LE cols, f64 LE column-major)."""
    with path.open("rb") as f:
        rows = struct.unpack("<q", f.read(8))[0]
        cols = struct.unpack("<q", f.read(8))[0]
        data = np.frombuffer(f.read(8 * rows * cols), dtype="<f8")
    # column-major -> reshape with Fortran order.
    return data.reshape((rows, cols), order="F")


def _load_manifest() -> dict[str, object]:
    return cast(dict[str, object], json.loads(MANIFEST.read_text()))


def test_manifest_present_and_records_masked_cols() -> None:
    assert MANIFEST.is_file(), "phase4a manifest.json must be committed"
    manifest = _load_manifest()
    mat_conv = cast(dict[str, object], manifest["mat_conversion"])
    masked = cast(list[int], mat_conv["masked_cols"])
    assert masked == [6], "masked_cols must record the MultiConfigResults timing column"
    assert cast(list[str], mat_conv["vars"]) == MAT_VARS
    # libmatio version recorded (local-only tool note).
    assert "libmatio_version" in manifest
    # corpus counts recorded (MEASURED by the real Corpus).
    corpus = cast(dict[str, object], manifest["corpus"])
    assert corpus["nb_files"] == 3


def test_every_referenced_bin_exists_and_parses() -> None:
    manifest = _load_manifest()
    runs = cast(dict[str, object], manifest["runs"])
    for run in RUNS:
        run_meta = cast(dict[str, object], runs[run])
        shapes = cast(dict[str, list[int]], run_meta["shapes"])
        for var in MAT_VARS:
            path = PHASE4A / f"{run}_{var}.bin"
            assert path.is_file(), f"{path} must exist"
            arr = _read_bin(path)
            expected = tuple(shapes[var])
            assert arr.shape == expected, f"{path.name}: shape {arr.shape} != manifest {expected}"


def test_raw_mat_matches_converted_bin() -> None:
    """scipy re-reads the RAW committed `.mat` and, after masking the timing column,
    reproduces the converted `.bin` values byte-for-byte -- guards the conversion."""
    masked_cols = [6]
    for run in RUNS:
        mat = scipy.io.loadmat(PHASE4A / f"{run}.mat")
        for var in MAT_VARS:
            raw = cast(np.ndarray, mat[var]).astype("<f8").copy()
            if var == "MultiConfigResults":
                for c in masked_cols:
                    if c < raw.shape[1]:
                        raw[:, c] = 0.0
            converted = _read_bin(PHASE4A / f"{run}_{var}.bin")
            assert raw.shape == converted.shape, f"{run}/{var}: shape mismatch"
            # Bit-exact: the conversion is a pure copy + timing mask, no arithmetic.
            assert np.array_equal(raw.view(np.uint64), converted.view(np.uint64)), f"{run}/{var}: raw .mat (masked) != converted .bin"


def test_vrcts_dumps_present() -> None:
    manifest = _load_manifest()
    vrcts = cast(dict[str, object], manifest["vrcts"])
    for name in cast(list[str], vrcts["committed_names"]):
        path = PHASE4A / name
        assert path.is_file(), f"VRCTS dump {name} must be committed"
        text = path.read_text()
        assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
        assert "<AudioDoc" in text


def test_train_costmem_rows_identical_across_epochs() -> None:
    """Non-vacuity mirror of the Rust golden (documented): LTSV (Algo 2) has NO
    weight updates, so the per-epoch mem rows are constant-by-construction. The
    committed train BadClassifMem (4 rows = epochs+2) must have identical rows;
    tier-1 pins the loop/aggregation plumbing, not weight evolution (tier-2/Task 9)."""
    bad = _read_bin(PHASE4A / "train_ltsv_BadClassifMem.bin")
    assert bad.shape == (4, 1), "train BadClassifMem is (epochs+2)x1 = 4x1"
    for e in range(1, 4):
        assert bad[e, 0] == bad[0, 0], f"BadClassifMem epoch {e} must equal epoch 0"
    # Genuinely nonzero -> the identity is a real pin, not a vacuous 0 == 0.
    assert bad[0, 0] != 0.0, "train BadClassifMem must be nonzero (scored LTSV)"
