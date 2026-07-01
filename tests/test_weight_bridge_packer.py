"""adim + packer byte-parity tests (Python)."""

from pathlib import Path

import numpy as np
from speech import config_bridge, weight_bridge

REF = Path("tests/reference_data/phase0")


def _spec() -> dict:
    cfg = config_bridge.parse_legacy_config((REF / "1_worker_1.config").read_text())
    return config_bridge.nnet_spec(cfg, prefix="BLSTM")


def test_element_count() -> None:
    assert weight_bridge.element_count(_spec()) == 33671
    rows, _, _ = weight_bridge.read_bin(REF / "NNweights_config1.bin")
    assert rows == 33671


def test_full_pipeline_matches_flat() -> None:
    # Determinism/integrity guard only: the golden here is self-derived via this same Python
    # path. The load-bearing independent check is the Rust full_pipeline_matches_flat in
    # src/rust/tests/phase0_packer.rs, which reproduces the same fixture via an independent
    # implementation (true cross-language agreement).
    structured, spec = weight_bridge.load_structured(REF / "best_config_manifest.json", REF / "best_config_domain.bin")
    flat = weight_bridge.nnet_to_flat(weight_bridge.config_to_nnet(structured, spec), spec)
    golden = weight_bridge.read_weight_vector(REF / "best_net_flat.bin")
    assert np.array_equal(flat, golden)


def test_packer_bijection_on_real_bin() -> None:
    spec = _spec()
    flat = weight_bridge.read_weight_vector(REF / "NNweights_config1.bin")
    rt = weight_bridge.nnet_to_flat(weight_bridge.flat_to_nnet(flat, spec), spec)
    assert np.array_equal(rt, flat)
    assert (flat[-23:] > 0).all()  # std tail strictly positive


def test_adim_leaves_bias_and_rescales_body() -> None:
    structured, spec = weight_bridge.load_structured(REF / "best_config_manifest.json", REF / "best_config_domain.bin")
    nnet = weight_bridge.config_to_nnet(structured, spec)
    for direction in ("forward", "backward"):
        for i, layer in enumerate(structured[direction]):
            adim = weight_bridge._lstm_adim(spec, i)
            for g in ("input", "forget", "output", "cell"):
                cfg_row, nn_row = layer[g], nnet[direction][i][g]
                assert np.array_equal(cfg_row[:, -1], nn_row[:, -1])  # bias column untouched
                assert np.allclose(nn_row[:, :-1] * adim, cfg_row[:, :-1])  # body divided by adim
    for i, cfg_mat in enumerate(structured["output"]):
        adim = weight_bridge._out_adim(spec, i)
        nn_mat = nnet["output"][i]
        assert np.array_equal(cfg_mat[:, -1], nn_mat[:, -1])
        assert np.allclose(nn_mat[:, :-1] * adim, cfg_mat[:, :-1])
