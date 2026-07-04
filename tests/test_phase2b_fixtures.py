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
