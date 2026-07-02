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
