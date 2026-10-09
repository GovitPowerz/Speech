"""Phase 4c Task 7: `pack_weights`/`unpack_weights` + mandated hypothesis round-trips.

Contract (see `weight_bridge.pack_weights`/`unpack_weights` docstrings for the full citation
of the Rust twins `config.rs::nnet_to_flat`/`flat_to_nnet`):
  - `net`/`arch` follow the module's `config_to_nnet`-shaped `Nnet` dict and the `NnetSpec`-shaped
    spec dict (`config_bridge.nnet_spec`'s output: `LSTMNeuronNb`/`LSTMSubSampling`/
    `OutputNeuronNb`/`OutputSubSampling`).
  - `pack_weights(net) -> flat` needs no `arch`: nnet-domain matrices are self-describing.
  - `unpack_weights(flat, arch) -> net` needs `arch`: a flat vector carries no shape.

Hypothesis profile: `tests/conftest.py` registers "ci" (derandomize=True) and "dev" (randomized),
selected via the `HYPOTHESIS_PROFILE` env var (CI sets it; local runs default to "dev"). To
replay a CI failure locally: `HYPOTHESIS_PROFILE=ci uv run pytest tests/test_phase4c_weight_bridge.py`.
"""

from pathlib import Path
from typing import cast

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray
from speech import config_bridge
from speech import weight_bridge as wb

REF = Path("tests/reference_data/phase0")


@st.composite
def _nnet_specs(draw: st.DrawFn) -> dict[str, object]:
    """NnetSpec-shaped dict: 1-3 LSTM layers, 1-3 output layers, neurons 1-8, subsampling 1-3.

    Subsampling is drawn > 1 sometimes on purpose: the real `1_worker_1.config` fixture has
    `BLSTM_LSTMSubSampling 4,1` - pinning subsampling to 1 would silently skip the
    `_spec_from_net` division branch that recovers it from `net`'s own shapes.
    """
    n_lstm_layers = draw(st.integers(min_value=1, max_value=3))
    lstm_neuron_nb = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=n_lstm_layers + 1, max_size=n_lstm_layers + 1))
    lstm_subsampling = draw(st.lists(st.integers(min_value=1, max_value=3), min_size=n_lstm_layers, max_size=n_lstm_layers))
    n_out_layers = draw(st.integers(min_value=1, max_value=3))
    output_neuron_nb = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=n_out_layers + 1, max_size=n_out_layers + 1))
    output_subsampling = draw(st.lists(st.integers(min_value=1, max_value=3), min_size=n_out_layers, max_size=n_out_layers))
    return {
        "LSTMNeuronNb": lstm_neuron_nb,
        "LSTMSubSampling": lstm_subsampling,
        "OutputNeuronNb": output_neuron_nb,
        "OutputSubSampling": output_subsampling,
    }


@st.composite
def _spec_and_flat(draw: st.DrawFn) -> tuple[dict[str, object], NDArray[np.float64]]:
    """A spec paired with a finite-float flat vector of exactly `element_count(spec)` length."""
    spec = draw(_nnet_specs())
    n = wb.element_count(spec)
    values = draw(st.lists(st.floats(allow_nan=False, allow_infinity=False, width=64), min_size=n, max_size=n))
    return spec, np.asarray(values, dtype=np.float64)


def _assert_nets_equal(a: dict[str, object], b: dict[str, object]) -> None:
    for direction in ("forward", "backward"):
        layers_a = cast(list[dict[str, np.ndarray]], a[direction])
        layers_b = cast(list[dict[str, np.ndarray]], b[direction])
        for layer_a, layer_b in zip(layers_a, layers_b, strict=True):
            for gate in ("input", "forget", "output", "cell"):
                assert np.array_equal(layer_a[gate], layer_b[gate])
    for mat_a, mat_b in zip(cast(list[np.ndarray], a["output"]), cast(list[np.ndarray], b["output"]), strict=True):
        assert np.array_equal(mat_a, mat_b)
    assert np.array_equal(cast(np.ndarray, a["mean"]), cast(np.ndarray, b["mean"]))
    assert np.array_equal(cast(np.ndarray, a["std"]), cast(np.ndarray, b["std"]))


@given(_spec_and_flat())
def test_pack_unpack_flat_roundtrip(spec_and_flat: tuple[dict[str, object], NDArray[np.float64]]) -> None:
    """pack(unpack(flat, spec)) == flat, bit-exact: no arithmetic, pure slice/reshape/concat."""
    spec, flat = spec_and_flat
    net = wb.unpack_weights(flat, spec)
    repacked = wb.pack_weights(net)
    assert repacked.dtype == np.float64
    assert np.array_equal(repacked, flat)


@given(_spec_and_flat())
def test_unpack_pack_net_roundtrip(spec_and_flat: tuple[dict[str, object], NDArray[np.float64]]) -> None:
    """unpack(pack(net), spec) == net for a self-consistent net (built by unpack in the first place)."""
    spec, flat = spec_and_flat
    net = wb.unpack_weights(flat, spec)
    net2 = wb.unpack_weights(wb.pack_weights(net), spec)
    _assert_nets_equal(net, net2)


@given(_spec_and_flat())
def test_element_count_matches(spec_and_flat: tuple[dict[str, object], NDArray[np.float64]]) -> None:
    spec, flat = spec_and_flat
    net = wb.unpack_weights(flat, spec)
    packed = wb.pack_weights(net)
    assert packed.shape[0] == wb.element_count(spec) == flat.shape[0]


@given(_spec_and_flat())
def test_output_layer_fan_in_is_neurons_times_subsampling(spec_and_flat: tuple[dict[str, object], NDArray[np.float64]]) -> None:
    """Output layer `i` reads `OutputNeuronNb[i] * OutputSubSampling[i]` inputs, as the exact net
    builds it (`Network::new`) and the legacy packs it (`vec2struct.m:908`). The round trips
    above only check the packer against itself, so they cannot see a fan-in that ignores the
    sub-sampling on both sides at once (#61)."""
    spec, flat = spec_and_flat
    outn = cast(list[int], spec["OutputNeuronNb"])
    osub = cast(list[int], spec["OutputSubSampling"])
    net = wb.unpack_weights(flat, spec)
    shapes = [mat.shape for mat in cast(list[np.ndarray], net["output"])]
    assert shapes == [(outn[i + 1], outn[i] * osub[i] + 1) for i in range(len(outn) - 1)]


def test_pack_weights_matches_nnet_to_flat_on_real_fixture() -> None:
    """Cross-check vs the committed real artifacts (mirrors test_full_pipeline_matches_flat)."""
    structured, spec = wb.load_structured(REF / "best_config_manifest.json", REF / "best_config_domain.bin")
    net = wb.config_to_nnet(structured, spec)
    golden = wb.read_weight_vector(REF / "best_net_flat.bin")
    assert np.array_equal(wb.pack_weights(net), golden)


def test_unpack_weights_matches_flat_to_nnet_on_real_fixture() -> None:
    cfg = config_bridge.parse_legacy_config((REF / "1_worker_1.config").read_text())
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    flat = wb.read_weight_vector(REF / "NNweights_config1.bin")
    net_direct = wb.flat_to_nnet(flat, spec)
    net_via_unpack = wb.unpack_weights(flat, spec)
    _assert_nets_equal(net_direct, net_via_unpack)
    # And the full pack(unpack(.)) == identity loop closes on real (non-hypothesis) data too.
    assert np.array_equal(wb.pack_weights(net_via_unpack), flat)
