"""Assert the Phase 2 NN-probe oracle manifest is present and internally consistent.

Produced by ``scripts/extract_phase2_fixtures.py`` (which links the vendored legacy
NN stack into the oracle harness, constructs a real ``BLSTMNeuralNetwork<LSTMLayer>``
from the vendored config + weights, and measures Eigen-vs-ascending-loop product-order
divergence at four NN sites). Task 1 is measurement-only: this guards that the
harness really built the net (weights == 33671) and that every probe site recorded a
self-consistent divergence measurement. Later NN-forward tasks consume these numbers.
"""

import json
from pathlib import Path
from typing import Any

REF = Path("tests/reference_data/phase2")
PROBE_SITES = {"lstm_input_gemm", "lstm_recurrence_gemv", "dense_gemm", "softmax_rowsum"}


def _manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((REF / "manifest.json").read_text())
    return manifest


def test_manifest_present() -> None:
    assert (REF / "manifest.json").is_file()


def test_records_strict_fp_flags() -> None:
    flags = _manifest()["flags"]
    for flag in ("-fno-fast-math", "-ffp-contract=off", "-DEIGEN_DONT_VECTORIZE"):
        assert flag in flags, flag


def test_real_net_weight_count() -> None:
    manifest = _manifest()
    assert manifest["nb_of_weights"] == 33671
    src = manifest["nn_source"]
    assert src["layer_type"] == "LSTMLayer"
    assert src["weightsFile_cleared"] is True
    # The shapes that make the probe dims (23x96 input, 24x96 recurrence, 48x12 dense).
    assert src["lstm_neuron_nb"] == [23, 24, 24]
    assert src["output_neuron_nb"] == [48, 12, 1]


def test_all_probe_sites_recorded() -> None:
    sites = _manifest()["nn_product_probes"]["sites"]
    assert set(sites) == PROBE_SITES


def test_probe_fields_self_consistent() -> None:
    sites = _manifest()["nn_product_probes"]["sites"]
    for name, probe in sites.items():
        for field in ("diverged", "mismatches", "total", "first_index", "eigen_bits", "loop_bits"):
            assert field in probe, f"{name} missing {field}"
        assert probe["total"] > 0, name
        assert 0 <= probe["mismatches"] <= probe["total"], name
        if probe["diverged"]:
            # A recorded divergence must have >=1 mismatch, a real first index, and
            # the eigen/loop bit patterns at that index must actually differ.
            assert probe["mismatches"] >= 1, name
            assert probe["first_index"][0] >= 0, name
            assert probe["eigen_bits"] != probe["loop_bits"], name
        else:
            # Non-divergence must be exactly zero mismatches with the sentinel index.
            assert probe["mismatches"] == 0, name
            assert probe["first_index"][0] == -1, name


def test_recurrence_gemv_measured_over_many_timesteps() -> None:
    # The recurrence GEMV result is strategically load-bearing for the NN port: the
    # probe accumulates the (1x24)*(24x96) GEMV over many timesteps, so `total` must
    # be far larger than a single 96-wide row (>= 100 steps * 96).
    probe = _manifest()["nn_product_probes"]["sites"]["lstm_recurrence_gemv"]
    assert probe["total"] >= 100 * 96


def test_lstm_forward_tol_bit_exact_vs_real_layer() -> None:
    # Task 4: the ascending-loop reimpl (lstmForwardLoop) that produced the LSTM
    # forward/reverse dumps must match the REAL compiled LSTMLayer::feedForward
    # bit-for-bit over the whole grid -- max_ulp == 0 (Eigen does not diverge from the
    # ascending order at these small shapes), otherwise the dumps would encode Eigen's
    # order, not the portable one the Rust port reproduces.
    tol = _manifest()["lstm_forward_tol"]
    assert tol["site"] == "lstm_forward"
    assert tol["max_ulp"] == 0, tol
    assert float(tol["max_abs"]) == 0.0, tol


def test_dense_forward_tol_bit_exact_vs_real_layer() -> None:
    # Task 5: the ascending-loop reimpl (denseForwardLoop) that produced the dense
    # hidden/softmax/logistic/width-mismatch dumps must match the REAL compiled
    # NeuronLayer::feedForward bit-for-bit over the whole grid -- max_ulp == 0, same
    # rationale as the LSTM tolerance check above.
    tol = _manifest()["dense_forward_tol"]
    assert tol["site"] == "dense_forward"
    assert tol["max_ulp"] == 0, tol
    assert float(tol["max_abs"]) == 0.0, tol


def test_dense_dumps_recorded_in_inventory() -> None:
    dumps = _manifest()["dumps"]
    for name in (
        "dense_hidden.bin",
        "dense_softmax.bin",
        "dense_logistic.bin",
        "dense_wide.bin",
        "dense_narrow.bin",
    ):
        assert name in dumps, name


def _assert_tol_field(tol: dict[str, Any], site: str) -> None:
    assert tol["site"] == site, tol
    assert isinstance(tol["max_ulp"], int), tol
    assert tol["max_ulp"] >= 0, tol
    float(tol["max_abs"])  # must parse as a float


# Task 8: BLSTM input-normalization + core-forward hidden-state probes.
# _OutputForward/_OutputBackward are PUBLIC members (BLSTMNeuralNetwork.h:23-24, no
# accessor needed) -- probed against the real class alongside the output. The
# synthetic-net sites (small shapes, k<=6) are bit-exact; the real full-sequence site
# is EXPECTED NONZERO (k=23 input GEMM diverges from ascending accumulation, per the
# Task 1 lstm_input_gemm probe), same as its already-covered output site.
def test_blstm_norm_synthetic_hidden_states_bit_exact() -> None:
    blstm = _manifest()["blstm_forward_tol"]
    for tag in ("1", "m1", "m2", "0"):
        fwd = blstm["synthetic_fwd"][f"blstm_norm{tag}_fwd"]
        bwd = blstm["synthetic_bwd"][f"blstm_norm{tag}_bwd"]
        _assert_tol_field(fwd, f"blstm_norm{tag}_fwd")
        _assert_tol_field(bwd, f"blstm_norm{tag}_bwd")
        assert fwd["max_ulp"] == 0, fwd
        assert bwd["max_ulp"] == 0, bwd


def test_blstm_real_fullseq_hidden_states_recorded() -> None:
    blstm = _manifest()["blstm_forward_tol"]
    fwd = blstm["real_fullseq_fwd"]
    bwd = blstm["real_fullseq_bwd"]
    _assert_tol_field(fwd, "blstm_real_fullseq_fwd")
    _assert_tol_field(bwd, "blstm_real_fullseq_bwd")


# Task 9: the four windowed forward drivers' hidden-state probes. All five sites are
# bit-exact vs the REAL class (max_ulp=0) on OUTPUT AND HIDDEN STATES -- the reimpls
# transcribe the integer window arithmetic + write-back index math verbatim.
# blstm_mlpoverlap_fwd/_bwd are a degenerate 0x0-vs-0x0 comparison (the MLP path has
# no LSTM; feedForwardBackwardMLPOverLap empties both members, asserted in-harness).
def test_blstm_windowed_hidden_states_bit_exact() -> None:
    windowed = _manifest()["blstm_windowed_tol"]
    for site in (
        "blstm_truncate",
        "blstm_twosweeps",
        "blstm_overlap",
        "blstm_overlap_nan",
        "blstm_mlpoverlap",
    ):
        for suffix in ("_fwd", "_bwd"):
            tol = windowed[site + suffix]
            _assert_tol_field(tol, site + suffix)
            assert tol["max_ulp"] == 0, tol
