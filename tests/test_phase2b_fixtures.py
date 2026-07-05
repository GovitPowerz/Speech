"""Assert the Phase 2b segmenter-wiring oracle manifest is present and consistent.

Produced by ``scripts/extract_phase2b_fixtures.py`` (which links the six vendored
legacy segmenter TUs into the oracle harness so the REAL ``Segmentation::
toFile_VRCTS`` becomes a byte golden, upgrades the ``iof::fmtr`` shim to faithful,
and records the deferred-port call-site verdicts). Task 1 is wiring + gating: this
guards that the fmtr self-test passed on every case and that both deferred ports
have zero live callers (spec decision 7).
"""

import json
from pathlib import Path
from typing import Any

REF = Path("tests/reference_data/phase2b")
EXPECTED_FMTR_CASES = {"fmtr_vrcts", "fmtr_f2", "fmtr_tail", "fmtr_f3"}
DEFERRED_FNS = {"computeCost", "computeSpectralPitch"}


def _manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((REF / "manifest.json").read_text())
    return manifest


def test_manifest_present() -> None:
    assert (REF / "manifest.json").is_file()


def test_records_strict_fp_flags() -> None:
    flags = _manifest()["flags"]
    for flag in ("-fno-fast-math", "-ffp-contract=off", "-DEIGEN_DONT_VECTORIZE"):
        assert flag in flags, flag


def test_segmenter_tus_linked() -> None:
    linkage = _manifest()["segmenter_linkage"]
    for tu in (
        "Segmenter.cpp",
        "Segmentation.cpp",
        "BLSTMSpectralSegmenter.cpp",
        "BLSTMSignalSegmenter.cpp",
        "LongTermSpectralVariation.cpp",
        "TimeDomainCorrel.cpp",
    ):
        assert tu in linkage["translation_units"], tu
    assert "png" in linkage["extra_link_libs"]
    # The include-graph analysis predicted no residual link errors beyond -lpng.
    assert linkage["additional_tus_required"] == []


def test_all_fmtr_checks_passed() -> None:
    checks = _manifest()["fmtr_shim"]["checks"]
    assert set(checks) == EXPECTED_FMTR_CASES
    for case, ok in checks.items():
        assert ok == 1, case


def test_deferred_ports_have_zero_live_callers() -> None:
    callsites = _manifest()["callsite_checks"]
    counts = callsites["counts"]
    assert set(counts) == DEFERRED_FNS
    for fn, n in counts.items():
        assert n == 0, (fn, callsites["matched_lines"][fn])


def test_results_to_segmentation_dumps_present_with_expected_shapes() -> None:
    expected_shapes = _manifest()["results_to_segmentation"]["expected_shapes"]
    assert expected_shapes == {
        "r2s_conv_coeff.bin": [1, 19],
        "r2s_convolved.bin": [1, 60],
        "r2s_boundaries.bin": [2, 2],
        "r2s_boundaries_noconv.bin": [2, 2],
    }
    for name, (rows, cols) in expected_shapes.items():
        path = REF / name
        assert path.is_file(), name
        with path.open("rb") as f:
            got_rows = int.from_bytes(f.read(8), "little", signed=True)
            got_cols = int.from_bytes(f.read(8), "little", signed=True)
        assert (got_rows, got_cols) == (rows, cols), name


def test_tdc_segmenter_dumps_present_with_expected_shapes() -> None:
    """Task 4: TdcSegmenter (Algo 1) -- result/convolved/boundaries/scores dumps."""
    tdc = _manifest()["tdc_segmenter"]
    expected_shapes = tdc["expected_shapes"]
    assert expected_shapes == {
        "tdc_result_chan1.bin": [1, 201],
        "tdc_result_chan2.bin": [1, 201],
        "tdc_convolved_chan1.bin": [1, 201],
        "tdc_convolved_chan2.bin": [1, 201],
        "tdc_boundaries_chan1.bin": [3, 2],
        "tdc_boundaries_chan2.bin": [2, 2],
        "tdc_boundaries_file2_chan1.bin": [3, 2],
        "tdc_scores.bin": [2, 3],
    }
    for name, (rows, cols) in expected_shapes.items():
        path = REF / name
        assert path.is_file(), name
        with path.open("rb") as f:
            got_rows = int.from_bytes(f.read(8), "little", signed=True)
            got_cols = int.from_bytes(f.read(8), "little", signed=True)
        assert (got_rows, got_cols) == (rows, cols), name

    for name in tdc["extra_files"]:
        assert (REF / name).is_file(), name

    constants = tdc["constants"]
    assert constants["rate"] == 8000
    assert constants["window_size_frames"] == 128
    assert constants["full_window_frames"] == 257
    assert constants["min_lag_frames"] == 16
    assert constants["max_lag_frames"] == 128
    assert constants["window_shift_frames"] == 80
    assert constants["vec_size"] == 201


def test_ltsv_segmenter_dumps_present_with_expected_shapes() -> None:
    """Task 5: LtsvSegmenter (Algo 2) -- primary + DCT-secondary + tiny-window +
    power-scale (Task 5 review Finding 1: non-vacuous decision-layer golden)."""
    ltsv = _manifest()["ltsv_segmenter"]
    expected_shapes = ltsv["expected_shapes"]
    assert expected_shapes == {
        "ltsv_result_chan1.bin": [1, 51],
        "ltsv_result_chan2.bin": [1, 51],
        "ltsv_convolved_chan1.bin": [1, 51],
        "ltsv_convolved_chan2.bin": [1, 51],
        "ltsv_boundaries_chan1.bin": [2, 2],
        "ltsv_boundaries_chan2.bin": [2, 2],
        "ltsv_boundaries_file2_chan1.bin": [2, 2],
        "ltsv_scores.bin": [2, 3],
        "ltsv_dct_result_chan1.bin": [1, 51],
        "ltsv_dct_result_chan2.bin": [1, 51],
        "ltsv_dct_convolved_chan1.bin": [1, 51],
        "ltsv_dct_convolved_chan2.bin": [1, 51],
        "ltsv_dct_boundaries_chan1.bin": [2, 2],
        "ltsv_tiny_boundaries_chan1.bin": [2, 2],
        "ltsv_powermel_result_chan1.bin": [1, 51],
        "ltsv_powermel_result_chan2.bin": [1, 51],
        "ltsv_powermel_convolved_chan1.bin": [1, 51],
        "ltsv_powermel_convolved_chan2.bin": [1, 51],
        "ltsv_powermel_boundaries_chan1.bin": [4, 2],
        "ltsv_powermel_boundaries_chan2.bin": [2, 2],
    }
    for name, (rows, cols) in expected_shapes.items():
        path = REF / name
        assert path.is_file(), name
        with path.open("rb") as f:
            got_rows = int.from_bytes(f.read(8), "little", signed=True)
            got_cols = int.from_bytes(f.read(8), "little", signed=True)
        assert (got_rows, got_cols) == (rows, cols), name

    for name in ltsv["extra_files"]:
        assert (REF / name).is_file(), name
    for name in ltsv["config_files"]:
        assert (REF / name).is_file(), name

    constants = ltsv["constants"]
    assert constants["rate"] == 8000
    assert constants["spectrum_order"] == 8
    assert constants["periodogram_length"] == 129
    assert constants["spectrum_shift_frames"] == 80
    assert constants["periodogram_vec_size"] == 201
    assert constants["ltsv_half_window_frames"] == 15
    assert constants["ltsv_shift_frames"] == 4
    assert constants["real_vec_size"] == 51
    assert constants["ltsv_tiny_window_half_frames"] == 1


def test_spectral_segmenter_dumps_present_with_expected_shapes() -> None:
    """Task 7: BlstmSpectralSegmenter (Algo 3, no pitch pass) -- the REAL
    1_worker_1.config's algorithm. Three variants (real/overlap/noOverlap) + the
    overlap cross-channel reuse dumps + the two-files noOverlap lifecycle + params."""
    spectral = _manifest()["spectral_segmenter"]
    expected_shapes = spectral["expected_shapes"]
    assert expected_shapes == {
        "spectral_real_inputseq_chan1.bin": [201, 11],
        "spectral_real_result_chan1.bin": [1, 50],
        "spectral_real_convolved_chan1.bin": [1, 50],
        "spectral_real_boundaries_chan1.bin": [3, 2],
        "spectral_real_scores.bin": [2, 3],
        "spectral_overlap_inputseq_chan1.bin": [201, 11],
        "spectral_overlap_result_chan1.bin": [1, 50],
        "spectral_overlap_result_chan2.bin": [1, 50],
        "spectral_overlap_convolved_chan1.bin": [1, 50],
        "spectral_overlap_convolved_chan2.bin": [1, 50],
        "spectral_overlap_boundaries_chan1.bin": [3, 2],
        "spectral_overlap_boundaries_chan2.bin": [2, 2],
        "spectral_overlap_scores.bin": [2, 3],
        "spectral_noOverlap_inputseq_chan1.bin": [201, 11],
        "spectral_noOverlap_result_chan1.bin": [1, 50],
        "spectral_noOverlap_convolved_chan1.bin": [1, 50],
        "spectral_noOverlap_boundaries_chan1.bin": [3, 2],
        "spectral_noOverlap_scores.bin": [2, 3],
        "spectral_noOverlap_boundaries_file1_chan1.bin": [3, 2],
        "spectral_noOverlap_boundaries_file2_chan1.bin": [3, 2],
        "spectral_noOverlap_params_file2.bin": [1, 4],
    }
    for name, (rows, cols) in expected_shapes.items():
        path = REF / name
        assert path.is_file(), name
        with path.open("rb") as f:
            got_rows = int.from_bytes(f.read(8), "little", signed=True)
            got_cols = int.from_bytes(f.read(8), "little", signed=True)
        assert (got_rows, got_cols) == (rows, cols), name

    for name in spectral["extra_files"]:
        assert (REF / name).is_file(), name

    # SECONDARY real-forward structural probe (spec decision 3): all three spectral
    # sites must be present with ok=1 (a structural mismatch would have aborted
    # generation) and the recorded max_dt.
    seg_struct = spectral["seg_struct"]
    assert set(seg_struct) == {"spectral_real", "spectral_overlap", "spectral_noOverlap"}
    for site, rec in seg_struct.items():
        assert rec["ok"] == 1, site

    constants = spectral["constants"]
    assert constants["rate"] == 8000
    assert constants["frame_count"] == 16001
    assert constants["sub_sampling_ratio"] == 4
    assert constants["spectrum_shift_in_frames"] == 80
    assert constants["periodogram_rows"] == 201
    assert constants["input_width_d"] == 11
    assert constants["real_vec_size"] == 50
    assert constants["nooverlap_window_size_periodogram_frames"] == 324
    assert constants["file2_params_row"] == {
        "window_shift": 1,
        "window_size": 324,
        "spectrum_shift_in_frames": 80,
        "real_vec_size": 50,
    }
