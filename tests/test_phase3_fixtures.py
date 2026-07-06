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
    "dense_backward_deltasout",
    "net_lstm_backward_fwd",
    "net_lstm_backward_rev",
    "net_dense_backward",
    "net_single_layer_backward_subsample",
    "net_single_layer_backward_plain",
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
# Task 3: dense backward deltas_out (T x I) per NeuronLayer grid variant.
EXPECTED_DELTASOUT_SHAPES = {
    "bwd_dense_hidden_deltasout.bin": (6, 4),
    "bwd_dense_last_deltasout.bin": (6, 4),
    "bwd_dense_logistic_deltasout.bin": (6, 4),
}
# Task 4: LSTMLayer::feedBackward per-variant goldens (synthetic I=2,O=2,T=5).
# deltasPreviousLayer T x I = 5 x 2; derivs Nx2 with N = 4*I*O+4*O*O+12*O+4*O = 64.
LSTM_BWD_SHAPES = {
    "lstm_bwd_deltasprev_peep_all.bin": (5, 2),
    "lstm_bwd_deltasprev_peep_none.bin": (5, 2),
    "lstm_bwd_deltasprev_reverse.bin": (5, 2),
    "lstm_bwd_deltasprev_subsample.bin": (5, 2),
    "lstm_bwd_derivs_peep_all.bin": (64, 2),
    "lstm_bwd_derivs_peep_none.bin": (64, 2),
    "lstm_bwd_derivs_reverse.bin": (64, 2),
    "lstm_bwd_derivs_subsample.bin": (64, 2),
    "lstm_bwd_signal_derivs.bin": (64, 2),
}
# Fix-wave 1 (review finding 1): the single-layer container backward branch
# (neuronNb.size()==2, NeuralNetwork.hpp:255-263) -- LSTM [2,2] at T=7.
NET_SINGLE_LAYER_SHAPES = {
    "bwd_net_single_subsample_derivs.bin": (80, 2),
    "bwd_net_single_plain_derivs.bin": (64, 2),
    "bwd_net_single_subsample_deltasout.bin": (6, 2),
    "bwd_net_single_plain_deltasout.bin": (7, 2),
}

EXPECTED_NB_DERIVS = 33671

# Task 2: CostLaw backward REAL-probe goldens (dumped from the REAL compiled
# computeUnitaryDeltas/computeDeltas). Nx2 [output|delta] scalar sweeps (65-point
# k/64 grid, or the fix-wave-1 3-point mid-thresh above-branch coverage) and
# n_frames x n_classes multiclass fusion dumps.
COST_SHAPES = {
    "cost_deriv_scalar_poly_speech.bin": (65, 2),
    "cost_deriv_scalar_poly_other.bin": (65, 2),
    "cost_deriv_scalar_log_speech.bin": (65, 2),
    "cost_deriv_scalar_log_other.bin": (65, 2),
    "cost_deriv_scalar_sqrt_speech.bin": (65, 2),
    "cost_deriv_scalar_sqrt_other.bin": (65, 2),
    "cost_deltas_multiclass.bin": (4, 3),
    "cost_deltas_multiclass_pond.bin": (4, 3),
    "cost_deltas_wer.bin": (3, 3),
    "cost_deltas_wer_pond.bin": (3, 3),
    # Fix-wave 1 (review finding 1): mid-thresh (0.5/0.5) above-branch coverage
    # for the cubic and sqrt laws, at points away from 0/1.
    "cost_deriv_scalar_midthresh_cubic_speech.bin": (3, 2),
    "cost_deriv_scalar_midthresh_cubic_other.bin": (3, 2),
    "cost_deriv_scalar_midthresh_sqrt_speech.bin": (3, 2),
    "cost_deriv_scalar_midthresh_sqrt_other.bin": (3, 2),
}

# Fix-wave 1 (review finding 2): the brief's literal alias names are a byte copy
# (shutil.copy2, the extractor) of the poly-named dump -- both must exist and be
# byte-identical, since no second C++ dump block produces them anymore.
ALIAS_COPIES = {
    "cost_deriv_scalar_speech.bin": "cost_deriv_scalar_poly_speech.bin",
    "cost_deriv_scalar_other.bin": "cost_deriv_scalar_poly_other.bin",
}


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
        "dense_backward_deltasout": m["dense_backward_tol"]["deltasout"]["max_ulp"],
        "net_lstm_backward_fwd": m["net_backward_tol"]["net_lstm_backward_fwd"]["max_ulp"],
        "net_lstm_backward_rev": m["net_backward_tol"]["net_lstm_backward_rev"]["max_ulp"],
        "net_dense_backward": m["net_backward_tol"]["net_dense_backward"]["max_ulp"],
        "net_single_layer_backward_subsample": m["net_single_layer_backward_tol"]["subsample"]["max_ulp"],
        "net_single_layer_backward_plain": m["net_single_layer_backward_tol"]["plain"]["max_ulp"],
        "blstm_feedbackward": m["blstm_backward_tol"]["synthetic"]["max_ulp"],
    }
    assert set(got) == SYNTHETIC_TOL_SITES
    for site, ulp in got.items():
        assert ulp == 0, (site, ulp)


def test_single_layer_backward_goldens_present_with_expected_shapes() -> None:
    """Fix-wave 1 (review finding 1): presence + shape guard for the single-layer
    container backward branch goldens (previously zero coverage)."""
    for name, (rows, cols) in NET_SINGLE_LAYER_SHAPES.items():
        path = REF / name
        assert path.is_file(), name
        assert _read_bin_shape(path) == (rows, cols), name


def test_single_layer_backward_deltasout_calibration_recorded() -> None:
    cal = _manifest()["net_single_layer_backward_tol"]["deltasout_calibration"]
    assert set(cal) == {"subsample", "plain"}
    for site, rec in cal.items():
        assert "max_ulp" in rec and "max_abs" in rec, site
        assert int(rec["max_ulp"]) == 0, (site, rec)  # measured 0 at this synthetic shape


def test_single_layer_backward_derivs_are_non_vacuous() -> None:
    for name in ("bwd_net_single_subsample_derivs.bin", "bwd_net_single_plain_derivs.bin"):
        path = REF / name
        with path.open("rb") as f:
            rows = int.from_bytes(f.read(8), "little", signed=True)
            int.from_bytes(f.read(8), "little", signed=True)  # cols == 2
            col0 = [struct.unpack("<d", f.read(8))[0] for _ in range(rows)]
        assert any(v != 0.0 for v in col0), name


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


def test_deltasout_goldens_present_with_expected_shapes() -> None:
    for name, (rows, cols) in EXPECTED_DELTASOUT_SHAPES.items():
        path = REF / name
        assert path.is_file(), name
        assert _read_bin_shape(path) == (rows, cols), name


def test_lstm_backward_variant_goldens_present_with_expected_shapes() -> None:
    for name, (rows, cols) in LSTM_BWD_SHAPES.items():
        path = REF / name
        assert path.is_file(), name
        assert _read_bin_shape(path) == (rows, cols), name


def test_lstm_backward_variants_manifest_max_ulp_zero() -> None:
    # Task 4: the reimpl (dumped as the golden) reproduces the REAL compiled
    # LSTMLayer::feedBackward / feedBackwardReverse bit-for-bit at these shapes.
    tol = _manifest()["lstm_backward_variants_tol"]
    assert tol["max_ulp"] == 0


def test_lstm_signal_backward_dead_rows_zero() -> None:
    # Width-tolerance (S11.5): the 1-col signal into an I=2 layer zero-pads the
    # input's second row, so input_weights_derivatives row 1 (input col 1) is
    # exactly 0 in every gate column. input_weights_derivatives is the first
    # I*4O = 2*8 = 16 col0 entries (column-major, I x 4O); index (ii=1, jj) lands
    # at jj*I + 1.
    path = REF / "lstm_bwd_signal_derivs.bin"
    with path.open("rb") as f:
        rows = int.from_bytes(f.read(8), "little", signed=True)
        int.from_bytes(f.read(8), "little", signed=True)  # cols == 2
        col0 = [struct.unpack("<d", f.read(8))[0] for _ in range(rows)]  # first col (col-major)
    in_size, out_size = 2, 2
    for jj in range(4 * out_size):
        assert col0[jj * in_size + 1] == 0.0, f"dead input row deriv nonzero at gate col {jj}"
    # Non-vacuity: the live rows (input col 0) are NOT all zero.
    assert any(col0[jj * in_size + 0] != 0.0 for jj in range(4 * out_size)), "signal derivs all-zero"


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


# --- Task 2: CostLaw backward REAL-probe goldens (fix-wave 1, review finding 3) ----
# Mirrors the Task 1 pattern above (EXPECTED_SHAPES + presence test + manifest
# well-formedness): the cost_*.bin fixtures and the costlaw_backward manifest
# section had no presence/shape guard before this fix wave.


def test_cost_goldens_present_with_expected_shapes() -> None:
    for name, (rows, cols) in COST_SHAPES.items():
        path = REF / name
        assert path.is_file(), name
        assert _read_bin_shape(path) == (rows, cols), name
    # The alias filenames are on disk too (byte copies -- see ALIAS_COPIES),
    # same expected shape as their source.
    for alias_name, source_name in ALIAS_COPIES.items():
        path = REF / alias_name
        assert path.is_file(), alias_name
        assert _read_bin_shape(path) == COST_SHAPES[source_name], alias_name


def test_costlaw_backward_manifest_well_formed() -> None:
    section = _manifest()["costlaw_backward"]
    assert isinstance(section["text"], str) and section["text"]
    dumps = section["dumps"]
    assert set(dumps) == set(COST_SHAPES)
    for name, (rows, cols) in COST_SHAPES.items():
        assert dumps[name] == {"rows": rows, "cols": cols}, name
    alias_dumps = section["alias_dumps"]
    assert set(alias_dumps) == set(ALIAS_COPIES)
    for alias_name, source_name in ALIAS_COPIES.items():
        assert alias_dumps[alias_name]["copy_of"] == source_name, alias_name


def test_cost_alias_dumps_are_byte_identical_copies() -> None:
    # Review finding 2: the alias pair is a byte copy of the poly-named dump
    # (no second, independently-maintained C++ dump block).
    for alias_name, source_name in ALIAS_COPIES.items():
        alias_bytes = (REF / alias_name).read_bytes()
        source_bytes = (REF / source_name).read_bytes()
        assert alias_bytes == source_bytes, (alias_name, source_name)


def test_cost_scalar_goldens_are_non_vacuous() -> None:
    """A scalar delta golden that is all-zero (e.g. the above-thresh branch
    landing exactly at the logistic-fold zero, or a law param that zeroes the
    above-thresh coefficients) would silently pass any assertion -- assert the
    delta column (col1) has at least one nonzero entry (S11.9 non-vacuity),
    for every scalar sweep/mid-thresh dump."""
    scalar_names = [name for name in COST_SHAPES if name.startswith("cost_deriv_scalar")]
    for name in scalar_names:
        path = REF / name
        with path.open("rb") as f:
            rows = int.from_bytes(f.read(8), "little", signed=True)
            int.from_bytes(f.read(8), "little", signed=True)  # cols == 2
            f.read(8 * rows)  # skip col0 (output)
            col1 = [struct.unpack("<d", f.read(8))[0] for _ in range(rows)]
        assert any(v != 0.0 for v in col1), name


def test_costlaw_ignore_mask_row_is_exactly_zero() -> None:
    """Structural (risk R8): the fully ignore-masked row (target < 0 everywhere)
    in the multiclass fusion goldens must be EXACTLY 0.0 bit, independent of the
    WER/ponderation path."""
    for name, masked_row, n_classes in (
        ("cost_deltas_multiclass.bin", 1, 3),
        ("cost_deltas_wer.bin", 2, 3),
    ):
        path = REF / name
        with path.open("rb") as f:
            rows = int.from_bytes(f.read(8), "little", signed=True)
            cols = int.from_bytes(f.read(8), "little", signed=True)
            data = struct.unpack(f"<{rows * cols}d", f.read(8 * rows * cols))
        # column-major: element (row, col) is at data[col * rows + row].
        for kk in range(n_classes):
            v = data[kk * rows + masked_row]
            assert v == 0.0, (name, masked_row, kk, v)
