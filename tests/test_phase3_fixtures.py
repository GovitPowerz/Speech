"""Assert the Phase 3 backward-probe oracle manifest + goldens are present and consistent.

Produced by ``scripts/extract_phase3_fixtures.py`` (harness-local ascending-loop BACKWARD
reimpl family + NN_TOL/NN_PROBE backward probes) and ``scripts/extract_phase3_oracle_cases.py``
(the JSON backward oracle-cases Task 9 consumes). Task 1 is the foundation everything else
in Phase 3 validates against: this guards that every SYNTHETIC backward site is bit-exact
vs the real compiled class (max_ulp == 0), the real-net calibration line is recorded, and
the reimpl-produced Nx2 deriv goldens exist with the expected shapes.
"""

import json
import struct
from pathlib import Path
from typing import Any

REF = Path("tests/reference_data/phase3")

SYNTHETIC_TOL_SITES = {
    "lstm_backward",
    "dense_backward",
    "net_lstm_backward_fwd",
    "net_lstm_backward_rev",
    "net_dense_backward",
    "blstm_feedbackward",
}
EXPECTED_SHAPES = {
    "bwd_lstm_derivs.bin": (72, 2),
    "bwd_dense_hidden_derivs.bin": (15, 2),
    "bwd_dense_last_derivs.bin": (15, 2),
    "bwd_dense_logistic_derivs.bin": (5, 2),
    "bwd_net_lstm_fwd_derivs.bin": (304, 2),
    "bwd_net_lstm_rev_derivs.bin": (304, 2),
    "bwd_net_dense_derivs.bin": (29, 2),
    "bwd_blstm_derivs.bin": (651, 2),
    "bwd_blstm_real_derivs.bin": (33671, 2),
}
EXPECTED_NB_DERIVS = 33671


def _manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((REF / "manifest.json").read_text())
    return manifest


def _read_bin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        rows = int.from_bytes(f.read(8), "little", signed=True)
        cols = int.from_bytes(f.read(8), "little", signed=True)
    return rows, cols


def test_manifest_present() -> None:
    assert (REF / "manifest.json").is_file()


def test_records_strict_fp_flags() -> None:
    flags = _manifest()["flags"]
    for flag in ("-fno-fast-math", "-ffp-contract=off", "-DEIGEN_DONT_VECTORIZE"):
        assert flag in flags, flag


def test_synthetic_backward_sites_are_bit_exact() -> None:
    m = _manifest()
    got = {
        "lstm_backward": m["lstm_backward_tol"]["max_ulp"],
        "dense_backward": m["dense_backward_tol"]["max_ulp"],
        "net_lstm_backward_fwd": m["net_backward_tol"]["net_lstm_backward_fwd"]["max_ulp"],
        "net_lstm_backward_rev": m["net_backward_tol"]["net_lstm_backward_rev"]["max_ulp"],
        "net_dense_backward": m["net_backward_tol"]["net_dense_backward"]["max_ulp"],
        "blstm_feedbackward": m["blstm_backward_tol"]["synthetic"]["max_ulp"],
    }
    assert set(got) == SYNTHETIC_TOL_SITES
    for site, ulp in got.items():
        assert ulp == 0, (site, ulp)


def test_real_net_calibration_recorded() -> None:
    blstm = _manifest()["blstm_backward_tol"]
    # The real-net site is EXPECTED NONZERO (k>=23 GEMM divergence) -- recorded, not zero.
    assert blstm["real_nb_derivs"] == EXPECTED_NB_DERIVS
    cal = blstm["real_calibration"]
    assert "max_ulp" in cal and "max_abs" in cal
    assert int(cal["max_ulp"]) >= 0  # recorded (nonzero in practice on the oracle env)


def test_backward_product_probes_do_not_diverge() -> None:
    probes = _manifest()["nn_backward_product_probes"]["sites"]
    assert set(probes) == {"bwd_outer_product", "bwd_backproj"}
    for site, rec in probes.items():
        assert rec["diverged"] is False, site
        assert rec["mismatches"] == 0, site


def test_deriv_goldens_present_with_expected_shapes() -> None:
    for name, (rows, cols) in EXPECTED_SHAPES.items():
        path = REF / name
        assert path.is_file(), name
        assert _read_bin_shape(path) == (rows, cols), name


def test_deriv_goldens_are_non_vacuous() -> None:
    """A backward golden with an all-zero deriv column would be a vacuous grad fixture
    (S11.2). Assert the summed-derivative column (col0) has nonzero entries in a couple
    of representative goldens."""
    for name in ("bwd_lstm_derivs.bin", "bwd_blstm_derivs.bin", "bwd_blstm_real_derivs.bin"):
        path = REF / name
        with path.open("rb") as f:
            rows = int.from_bytes(f.read(8), "little", signed=True)
            int.from_bytes(f.read(8), "little", signed=True)  # cols == 2
            col0 = [struct.unpack("<d", f.read(8))[0] for _ in range(rows)]  # first col (col-major)
        assert any(v != 0.0 for v in col0), name


def test_backward_oracle_cases_present() -> None:
    path = REF / "backward_cases.json"
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert len(payload["lstm_cases"]) >= 1
    assert len(payload["dense_cases"]) >= 3
    # Each case carries weights, input, deltas, and the expected Nx2 derivs (bit hex).
    for case in payload["lstm_cases"] + payload["dense_cases"]:
        for key in ("weights_bits", "input_bits", "deltas_bits", "expected_derivs_bits"):
            assert case[key], (case["name"], key)
