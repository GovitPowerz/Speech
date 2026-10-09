"""Phase 5 Task 3: seeded Xavier/He weight init over the `weight_bridge` packer spec.

Coverage for `init_weights.init_weights`: shape/element-count invariants against the
committed Phase 4d tuple-A/B fixture specs, forget-bias positions (located via
`weight_bridge.unpack_weights` -- the packer's own inverse -- never a hardcoded flat
offset), the identity normalize tail, determinism, and a xavier-vs-he distributional
sanity check.

The pyo3 `set_weights`/`weights()` round trip lives separately in
`tests/pyo3/test_phase5_init_weights_pyo3.py`: every existing `importorskip("speech_rs")`
in this repo lives under `tests/pyo3/` (see `test_engine_smoke.py`/`test_seam_replay.py`/
`test_exit_gate.py`) because the dedicated `python-pyo3` CI job runs only `pytest
tests/pyo3`, not the full `tests` tree -- a `speech_rs`-touching test placed directly under
`tests/` would import-skip in BOTH CI jobs and never actually execute there.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from speech import config_bridge
from speech import weight_bridge as wb
from speech.init_weights import init_weights

REF = Path("tests/reference_data/phase4d")


def _spec(config_name: str) -> dict[str, object]:
    cfg = config_bridge.parse_legacy_config((REF / config_name).read_text())
    return config_bridge.nnet_spec(cfg, prefix="BLSTM")


TUPLE_A_SPEC = _spec("tupleA_1_worker_1.config")
TUPLE_B_SPEC = _spec("tupleB_1_worker_1.config")
SPECS = {"A": (TUPLE_A_SPEC, 33671), "B": (TUPLE_B_SPEC, 42911)}


def _forward_layers(net: dict[str, object], direction: str) -> list[dict[str, np.ndarray]]:
    return cast(list[dict[str, np.ndarray]], net[direction])


def _output_layers(net: dict[str, object]) -> list[np.ndarray]:
    return cast(list[np.ndarray], net["output"])


@pytest.mark.parametrize("tag", ["A", "B"])
def test_pack_length_matches_spec_element_count(tag: str) -> None:
    spec, expected_count = SPECS[tag]
    assert wb.element_count(spec) == expected_count  # sanity: the fixture really is the shape we think it is.

    packs = init_weights(spec, np.random.default_rng(0))
    assert len(packs) == 1  # one net -> one flat pack.
    assert packs[0].dtype == np.float64
    assert packs[0].shape == (expected_count,)


# The two `twin_mode7.config` nets at `OutputSubSampling 2`, with the pack lengths
# `src/rust/tests/output_subsampling_pack.rs` pins against the exact net's `nb_of_weights` (#61).
OSUB2_SPECS: dict[str, tuple[dict[str, object], int]] = {
    "sad": ({"LSTMNeuronNb": [11, 12], "LSTMSubSampling": [4], "OutputNeuronNb": [24, 1], "OutputSubSampling": [2]}, 5831),
    "lid": ({"LSTMNeuronNb": [36, 24], "LSTMSubSampling": [1], "OutputNeuronNb": [48, 1], "OutputSubSampling": [2]}, 12457),
}


@pytest.mark.parametrize("tag", ["sad", "lid"])
def test_output_subsampling_widens_the_output_layer(tag: str) -> None:
    spec, expected_count = OSUB2_SPECS[tag]
    assert wb.element_count(spec) == expected_count

    pack = init_weights(spec, np.random.default_rng(0))[0]
    assert pack.shape == (expected_count,)
    outn = cast(list[int], spec["OutputNeuronNb"])
    assert _output_layers(wb.unpack_weights(pack, spec))[0].shape == (outn[1], outn[0] * 2 + 1)


@pytest.mark.parametrize("cell", ["lstm", "slstm", "mamba", "cfc", "transformer"])
def test_output_subsampling_widens_every_cell_pack_by_the_extra_fan_in(cell: str) -> None:
    """Both pack builders: `OutputSubSampling 2` on output layer 0 adds `OutputNeuronNb[1] *
    OutputNeuronNb[0]` weights, the second sub-sampled frame's block."""
    spec: dict[str, object] = {"LSTMNeuronNb": [5, 4], "LSTMSubSampling": [1], "OutputNeuronNb": [8, 3, 1], "OutputSubSampling": [1, 1], "CellType": cell}
    n1 = init_weights(spec, np.random.default_rng(0))[0].shape[0]
    n2 = init_weights({**spec, "OutputSubSampling": [2, 1]}, np.random.default_rng(0))[0].shape[0]
    assert n2 - n1 == 3 * 8


@pytest.mark.parametrize("tag", ["A", "B"])
def test_forget_bias_one_positions(tag: str) -> None:
    spec, _ = SPECS[tag]
    flat = init_weights(spec, np.random.default_rng(1), forget_bias_one=True)[0]
    net = wb.unpack_weights(flat, spec)

    for direction in ("forward", "backward"):
        for layer in _forward_layers(net, direction):
            assert np.all(layer["forget"][:, -1] == 1.0)
            assert np.all(layer["input"][:, -1] == 0.0)
            assert np.all(layer["output"][:, -1] == 0.0)
            assert np.all(layer["cell"][:, -1] == 0.0)
    for out_layer in _output_layers(net):
        assert np.all(out_layer[:, -1] == 0.0)


@pytest.mark.parametrize("tag", ["A", "B"])
def test_forget_bias_one_false_gives_every_bias_zero(tag: str) -> None:
    spec, _ = SPECS[tag]
    flat = init_weights(spec, np.random.default_rng(1), forget_bias_one=False)[0]
    net = wb.unpack_weights(flat, spec)

    for direction in ("forward", "backward"):
        for layer in _forward_layers(net, direction):
            for gate in ("input", "forget", "output", "cell"):
                assert np.all(layer[gate][:, -1] == 0.0)


@pytest.mark.parametrize("tag", ["A", "B"])
def test_normalize_tail_is_identity(tag: str) -> None:
    spec, _ = SPECS[tag]
    flat = init_weights(spec, np.random.default_rng(2))[0]
    net = wb.unpack_weights(flat, spec)

    mean, std = cast(np.ndarray, net["mean"]), cast(np.ndarray, net["std"])
    lstm0 = cast(list[int], spec["LSTMNeuronNb"])[0]
    assert mean.shape == (lstm0,)
    assert std.shape == (lstm0,)
    assert np.all(mean == 0.0)
    assert np.all(std == 1.0)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
def test_determinism_same_seed_is_bit_identical(scheme: str) -> None:
    packs_a = init_weights(TUPLE_A_SPEC, np.random.default_rng(42), scheme=scheme)  # type: ignore[arg-type]
    packs_b = init_weights(TUPLE_A_SPEC, np.random.default_rng(42), scheme=scheme)  # type: ignore[arg-type]
    assert np.array_equal(packs_a[0], packs_b[0])


def test_different_seeds_produce_different_packs() -> None:
    packs_a = init_weights(TUPLE_A_SPEC, np.random.default_rng(1))
    packs_b = init_weights(TUPLE_A_SPEC, np.random.default_rng(2))
    assert not np.array_equal(packs_a[0], packs_b[0])


def test_invalid_scheme_raises() -> None:
    with pytest.raises(ValueError):
        init_weights(TUPLE_A_SPEC, np.random.default_rng(0), scheme="bogus")  # type: ignore[arg-type]


def test_xavier_vs_he_variance_ratio_sanity_on_a_large_block() -> None:
    # A synthetic, deliberately large spec so the measured block has a big sample size
    # (low variance-estimate noise) -- loose bounds below, not a flaky tight assert.
    lstm_neuron_nb = [64, 128, 128]
    lstm_subsampling = [1, 1]
    spec: dict[str, object] = {
        "LSTMNeuronNb": lstm_neuron_nb,
        "LSTMSubSampling": lstm_subsampling,
        "OutputNeuronNb": [256, 10],
        "OutputSubSampling": [1, 1],
    }
    flat_x = init_weights(spec, np.random.default_rng(7), scheme="xavier")[0]
    flat_h = init_weights(spec, np.random.default_rng(7), scheme="he")[0]
    net_x = wb.unpack_weights(flat_x, spec)
    net_h = wb.unpack_weights(flat_h, spec)

    out, fin = lstm_neuron_nb[1], lstm_neuron_nb[0] * lstm_subsampling[0]
    combined_fan_in = out + fin
    block_x = _forward_layers(net_x, "forward")[0]["input"][:, :combined_fan_in]
    block_h = _forward_layers(net_h, "forward")[0]["input"][:, :combined_fan_in]
    assert block_x.size >= 10_000  # this really is "a large block".

    var_x, var_h = float(np.var(block_x)), float(np.var(block_h))
    expect_var_x = 2.0 / (combined_fan_in + out)  # Glorot uniform: (2*limit)**2/12 == 2/(fan_in+fan_out).
    expect_var_h = 2.0 / combined_fan_in  # He normal: std**2 == 2/fan_in.

    assert var_h > var_x  # guaranteed by construction whenever fan_out > 0 -- not just "close".
    assert 0.5 * expect_var_x < var_x < 2.0 * expect_var_x
    assert 0.5 * expect_var_h < var_h < 2.0 * expect_var_h


def test_peepholes_are_small_and_nonconstant() -> None:
    """Peephole columns get a small nonzero draw (documented as recurrent-diagonal init,
    not Xavier/He-scaled -- see `init_weights` module docstring), independent of `scheme`."""
    spec = TUPLE_A_SPEC
    flat = init_weights(spec, np.random.default_rng(3))[0]
    net = wb.unpack_weights(flat, spec)
    layer = _forward_layers(net, "forward")[0]
    out = cast(list[int], spec["LSTMNeuronNb"])[1]
    fin = cast(list[int], spec["LSTMNeuronNb"])[0] * cast(list[int], spec["LSTMSubSampling"])[0]
    ncols = fin + out + 5
    peep = layer["input"][:, ncols - 5 : ncols - 1]  # 3 gate-peephole cols + 1 recurrent-peephole col (bias excluded).
    assert np.all(np.abs(peep) <= 0.1)
    assert np.unique(peep).size > 1  # not a degenerate constant fill.
