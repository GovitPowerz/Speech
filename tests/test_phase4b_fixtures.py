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


# --- Task 7 FLAGSHIP: Mode-7 (phSeq) real-compiled member fixtures + DumpLIDInternals ---

MODE7_VARIANTS = ["twin_mode7", "twin_mode7_ppm1", "twin_mode7_ppm2"]
MODE7_FILES = ["s1", "s2", "s3"]


def test_mode7_member_fixtures_shapes_and_nonvacuity() -> None:
    """The REAL compiled TwinBLSTMSpectralLID::getSegmentation LID members per
    (variant, phSeq file): confusion is (classNb+2)x(classNb+2) = 4x4 (binary
    net -> classNb 2), liderr is 1x2, members is 1x3. Non-vacuity across files:
    at least one aggregate HIT (isCorrect 100) AND one MISS (0), and the >150
    targetLID sentinel present in every liderr row."""
    saw_hit = saw_miss = False
    for v in MODE7_VARIANTS:
        for f in MODE7_FILES:
            cr, cc, _ = _read_bin(PHASE4B / f"mode7_{v}_{f}_confusion.bin")
            assert (cr, cc) == (4, 4), f"mode7 {v} {f} confusion shape {cr}x{cc}"
            lr, lc, liderr = _read_bin(PHASE4B / f"mode7_{v}_{f}_liderr.bin")
            assert (lr, lc) == (1, 2), f"mode7 {v} {f} liderr shape {lr}x{lc}"
            assert any(x > 150.0 for x in liderr), f"mode7 {v} {f}: no >150 sentinel"
            mr, mc, members = _read_bin(PHASE4B / f"mode7_{v}_{f}_members.bin")
            assert (mr, mc) == (1, 3), f"mode7 {v} {f} members shape {mr}x{mc}"
            is_correct = members[2]
            assert is_correct in (0.0, 100.0), f"mode7 {v} {f} isCorrect {is_correct}"
            saw_hit = saw_hit or is_correct == 100.0
            saw_miss = saw_miss or is_correct == 0.0
            assert members[1] > 0.0, f"mode7 {v} {f} nb_of_classif not > 0"
    assert saw_hit and saw_miss, "need both an aggregate hit and a miss across the mode-7 corpus"


def test_manifest_mode7_section_present_and_matches_bins() -> None:
    """The extractor's Task 7 wiring (previously-dead MODE7_RE) parses + validates the
    harness's PHASE4B_MODE7 stdout lines and records them under manifest["mode7"];
    cross-check the recorded values against the committed .bin dumps independently of
    the extractor's own parsing."""
    manifest = _load_manifest()
    assert "mode7" in manifest
    mode7 = cast(dict[str, object], manifest["mode7"])
    assert mode7["lid_weights"] == 12409
    measured = cast(dict[str, object], mode7["measured"])
    assert set(measured) == set(MODE7_VARIANTS)
    for v in MODE7_VARIANTS:
        files_m = cast(dict[str, object], measured[v])
        assert set(files_m) == set(MODE7_FILES)
        for f in MODE7_FILES:
            entry = cast(dict[str, object], files_m[f])
            _mr, _mc, members = _read_bin(PHASE4B / f"mode7_{v}_{f}_members.bin")
            assert entry["lid_cumulative_error"] == members[0]
            assert entry["nb_of_classif"] == int(members[1])
            assert entry["is_correct"] == int(members[2])
            _cr, _cc, conf = _read_bin(PHASE4B / f"mode7_{v}_{f}_confusion.bin")
            assert entry["confusion_sum"] == sum(conf)


def test_manifest_lid_weight_provenance() -> None:
    """The committed LID_bestNNWeight_1.bin's provenance is now self-checked (Task 7
    IMPORTANT-2): the element count is asserted against the .bin's own header
    (independent of any legacy tree), and manifest["mode7"]["lid_weight_provenance"]
    records whether a fresh scipy re-conversion of the legacy source was reverified."""
    manifest = _load_manifest()
    provenance = cast(dict[str, object], cast(dict[str, object], manifest["mode7"])["lid_weight_provenance"])
    assert provenance["element_count"] == 12409
    assert provenance["source_variable"] == "weights"
    rows, cols, _ = _read_bin(PHASE4B / "LID_bestNNWeight_1.bin")
    assert rows * cols == 12409
    assert tuple(cast(list[int], provenance["shape"])) == (rows, cols)


# --- Task 4: lid5 (Algo 5) LID_STRUCT calibration -- cost_max_abs delta semantics ---

LID5_FILES = ["f1", "f2", "f3"]


def test_lid5_calibration_cost_max_abs_is_a_true_delta() -> None:
    """`cost_max_abs` (`tools/oracle_harness/main.cpp`'s lid5 LID_STRUCT probe) is the max
    per-channel |real - reimpl| `_LIDCumulativeError` delta, NOT the magnitude of the real
    value alone (a harness bug closed in Phase 4c Task 2: the reimpl's per-channel NNCost
    is now retained -- like `langidR` already was -- and diffed against
    `segReal._LIDCumulativeError[chan]`, `BLSTMSpectralLID.cpp:414`). Pin the fix: every
    calibration delta is TINY (same order as `langid_max_abs`'s ~1e-16 ascending-vs-Eigen
    forward noise), not O(1) -- the pre-fix magnitude-only readings were 19.47/0.1376/19.47."""
    manifest = _load_manifest()
    calibration = cast(dict[str, object], cast(dict[str, object], manifest["lid5"])["calibration"])
    assert set(calibration) == set(LID5_FILES)
    for f in LID5_FILES:
        entry = cast(dict[str, object], calibration[f])
        cost_max_abs = cast(float, entry["cost_max_abs"])
        assert cost_max_abs < 1e-6, f"lid5 {f}: cost_max_abs {cost_max_abs} not a small real-vs-reimpl delta"


def test_mode7_dump_lid_internals_scipy_valued() -> None:
    """DumpLIDInternals (:1136-1158): the harness real-Eigen `.mat` carries
    `features_<n>` = [_OutputForward | _OutputBackward] per kept phSeq block +
    a `matNb` scalar. Read via scipy (the 4a conversion pattern) and value-check:
    matNb == number of features_n vars (2 for s1's 7-/5-row blocks), each is a
    finite (rows x 96) matrix (TwoSweeps doubles the 24-wide fwd/bwd -> 48 each,
    hcat -> 96), and not all-zero."""
    import numpy as np
    import scipy.io

    mat = scipy.io.loadmat(PHASE4B / "mode7_dump_s1.mat")
    varnames = [k for k in mat if not k.startswith("__")]
    feats = sorted(v for v in varnames if v.startswith("features_"))
    assert feats == ["features_0", "features_1"], f"unexpected feature vars {feats}"
    assert "matNb" in varnames, "matNb missing"
    assert int(np.asarray(mat["matNb"]).ravel()[0]) == len(feats), "matNb != feature count"
    for name in feats:
        m = np.asarray(mat[name])
        assert m.shape[1] == 96, f"{name}: cols {m.shape[1]} != 96 (2*2*24 TwoSweeps [oF|oB])"
        assert m.shape[0] > 0 and np.all(np.isfinite(m)), f"{name}: empty or non-finite"
        assert np.any(m != 0.0), f"{name}: all-zero"
