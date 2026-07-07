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


# ==== Task 9 tier-2 fixture guards ==========================================


def test_tier2_manifest_trajectory_and_seg_struct() -> None:
    """The manifest records the MEASURED tier-2 trajectory (non-vacuity: gate fired
    AND skipped, every epoch pair differing) and the SEG_STRUCT secondary probe
    (count/type equality is the abort gate inside the harness; the recorded max_dt
    is asserted at its measured value, 0.0 here)."""
    manifest = _load_manifest()
    tier2 = cast(dict[str, object], manifest["tier2"])
    train = cast(dict[str, object], tier2["train"])
    nv = cast(dict[str, object], train["non_vacuity"])
    epochs = cast(int, train["epochs"])
    assert epochs == 5, "3 training epochs -> 5 saveAndUpdate rows (solo + 3 + final)"
    assert nv["epochs_differing"] == epochs - 1, "every consecutive epoch pair differs"
    assert cast(int, nv["gate_fired"]) >= 1, "best-cost gate fired at least once"
    assert cast(int, nv["gate_skipped"]) >= 1, "best-cost gate skipped at least once"
    trajectory = cast(list[float], nv["best_cost_trajectory"])
    assert len(trajectory) == epochs
    seg = cast(dict[str, object], train["seg_struct"])
    assert seg["ok"] == 1
    assert seg["max_dt"] == 0.0, "measured SEG_STRUCT boundary max_dt (reimpl vs real Eigen)"
    per_epoch = cast(list[float], seg["per_epoch_max_dt"])
    assert len(per_epoch) == epochs, "one SEG_STRUCT record per epoch"
    for e, dt in enumerate(per_epoch):
        assert dt == 0.0, f"epoch {e}: measured max_dt must be the recorded 0.0"
    calib = cast(dict[str, object], seg["calibration"])
    costs = cast(list[dict[str, float]], calib["per_epoch_costs"])
    assert len(costs) == epochs
    for e, c in enumerate(costs):
        # Calibration, not a gate: real-Eigen vs reimpl cost deltas are ULP-level.
        assert abs(c["cost_real"] - c["cost_reimpl"]) <= 1e-12 * abs(c["cost_reimpl"]), f"epoch {e}: real-vs-reimpl cost delta beyond ULP-level calibration"
    # The real-Eigen trajectory exercises the same gate pattern (fired AND skipped).
    assert cast(int, calib["real_gate_fired"]) >= 1
    assert cast(int, calib["real_gate_skipped"]) >= 1
    gc = cast(dict[str, object], tier2["gradcheck"])
    assert gc["gradcheck_max_weights"] == 10, "the capped sweep (deviation, recorded)"
    assert cast(float, gc["mean_relative_error"]) < 5e-4


def test_tier2_epoch_weight_fixtures_inventory() -> None:
    """Every per-epoch weight dump exists, parses, and is the full 33,671-weight
    vector; consecutive dumps DIFFER (the non-vacuity, re-checked from the committed
    bytes, not just the manifest)."""
    manifest = _load_manifest()
    train = cast(dict[str, object], cast(dict[str, object], manifest["tier2"])["train"])
    names = cast(list[str], train["epoch_weight_files"])
    assert names == [f"tier2_weights_epoch{e}.bin" for e in range(5)]
    prev: np.ndarray | None = None
    for name in names:
        path = PHASE4A / name
        assert path.is_file(), f"{name} must be committed"
        arr = _read_bin(path)
        assert arr.shape == (33671, 1), f"{name}: full real-net weight vector"
        if prev is not None:
            assert not np.array_equal(arr.view(np.uint64), prev.view(np.uint64)), f"{name} must differ from the previous epoch"
        prev = arr


def test_tier2_gradcheck_fixture_shape_and_seed() -> None:
    """tier2_gradcheck.bin: 10 per-weight triples + the means row; the means row
    reproduces the manifest values; the seed dump is the phase3 synth_flat pattern."""
    gc = _read_bin(PHASE4A / "tier2_gradcheck.bin")
    assert gc.shape == (11, 3)
    manifest = _load_manifest()
    meta = cast(dict[str, object], cast(dict[str, object], manifest["tier2"])["gradcheck"])
    assert gc[10, 0] == cast(float, meta["mean_error"])
    assert gc[10, 1] == cast(float, meta["mean_relative_error"])
    assert gc[10, 2] == 0.0
    seed = _read_bin(PHASE4A / "tier2_gradcheck_seed.bin")
    assert seed.shape == (137, 1), "the synthetic [2,2]+[4,1] net has 137 weights"
    expect = np.array([((k * 11 + 3) % 97) / 97.0 - 0.5 for k in range(137)]).reshape(-1, 1)
    assert np.array_equal(seed.view(np.uint64), expect.view(np.uint64))


def test_tier2_mat_conversion_and_artifacts() -> None:
    """The tier-2 result .mat reproduces its converted .bin fixtures (same col-6 mask
    guard as tier-1); the bestNNWeight artifact split is present and coherent (the
    weights/derivs are io::binary despite the legacy .mat suffix; the stats .mat
    carries an EMPTY InputStatistics -- type -1 never accumulates)."""
    mat = scipy.io.loadmat(PHASE4A / "tier2_spectral.mat")
    for var in MAT_VARS:
        raw = cast(np.ndarray, mat[var]).astype("<f8").copy()
        if var == "MultiConfigResults":
            raw[:, 6] = 0.0
        converted = _read_bin(PHASE4A / f"tier2_spectral_{var}.bin")
        assert raw.shape == converted.shape, f"tier2/{var}: shape mismatch"
        assert np.array_equal(raw.view(np.uint64), converted.view(np.uint64)), f"tier2/{var}: raw .mat (masked) != converted .bin"
    w = _read_bin(PHASE4A / "weights_bestNNWeight_1_tier2_spectral.mat")
    assert w.shape == (33671, 1)
    # saveWeights fires BEFORE updateWeights (BagOfProcessors.cpp:464 vs :466), so
    # the saved best == the PRE-update weights of the last firing epoch (the final
    # eval) == the post-epoch-3 vector.
    epoch3 = _read_bin(PHASE4A / "tier2_weights_epoch3.bin")
    assert np.array_equal(w.view(np.uint64), epoch3.view(np.uint64)), "best weights == post-epoch-3 (save-before-update order)"
    d = _read_bin(PHASE4A / "weightsDerivatives_bestNNWeight_1_tier2_spectral.mat")
    assert d.shape == (33671, 2)
    assert np.any(d[:, 0] != 0.0), "derivs col0 must be nonzero (live backward)"
    assert np.all(d[:, 1][d[:, 1] != 0.0] > 0.0), "derivs col1 counts are positive"
    stats = scipy.io.loadmat(PHASE4A / "bestNNWeight_1_tier2_spectral.mat")
    assert cast(np.ndarray, stats["nbOfInputs"])[0, 0] == 0.0
    assert cast(np.ndarray, stats["meanInputs"]).size == 0
    assert cast(np.ndarray, stats["stdInputs"]).size == 0


def test_tier2_vrcts_dumps_present() -> None:
    manifest = _load_manifest()
    train = cast(dict[str, object], cast(dict[str, object], manifest["tier2"])["train"])
    names = cast(list[str], train["vrcts"])
    assert names == [f"tier2_f{f}_chan_{c}.xml" for f in (1, 2) for c in (1, 2)]
    for name in names:
        path = PHASE4A / name
        assert path.is_file(), f"tier-2 VRCTS {name} must be committed"
        text = path.read_text()
        assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
        assert "<AudioDoc" in text
