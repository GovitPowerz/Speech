"""Phase 10 Task 2 (re-based on the seam, issue #27): the CfC draw rules, the sizing of
`Cfc_Backbone_Units`, and the baseline arm knob.

THE SIZING PROCEDURE (spec S1.4) -- how `CFC_DEFAULT_BACKBONE_UNITS = 45` was picked. One CfC
layer at `(fin, h, B, L=1)` holds `B*(fin+h+1) + 3*h*(B+1)` weights, so a full SAD net is
LINEAR in `B`: v1 (`LSTMNeuronNb 23,24,24`, sub 4,1; two stacks; MLP 48,12,1; tail 46) gives
`620*B + 935` against an LSTM pack of 33671; v2 (`11,24,24`) gives `524*B + 911` against 24431.
The S8.2 band is +-15% of the SAME lineage's LSTM pack. THE DEFAULT IS SIZED AGAINST v2 (the
lineage the phase targets): `B = 45` lands at 24491, +0.25%, and is the v2 argmin; it also
leaves v1 at -14.36%, inside its band, and is the SMALLEST integer that does -- so one default
serves both lineages. Every length below is READ off the engine (`speech_rs.weight_layout`),
the closed forms being the documented arithmetic they are pinned against, not a re-derivation
the test could drift with.

Byte identity with the pre-#27 builder is pinned by `tests/test_seed_pack_regeneration.py`
(the committed `cfc_*` and Twin CfC packs).
"""

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import speech_rs
from numpy.typing import NDArray
from speech import config_bridge
from speech.drivers import baseline
from speech.init_weights import init_weights

REF = Path("tests/reference_data/phase4d")
Block = dict[str, Any]

#: The finalized default (spec S1.4): the RUST constant `CFC_DEFAULT_BACKBONE_UNITS`, read back
#: through the seam by `test_the_default_is_the_rust_default` -- no Python mirror exists.
SIZED_BACKBONE_UNITS = 45

# (out, fin, B, L): the degenerate 1x1x1x1, a deep backbone, and three single-layer shapes.
SHAPES: tuple[tuple[int, int, int, int], ...] = ((1, 1, 1, 1), (4, 5, 8, 2), (2, 3, 4, 1), (3, 7, 8, 1), (3, 5, 4, 3))


def _layout(cfg: dict[str, str]) -> list[Block]:
    return cast(list[Block], speech_rs.weight_layout(cfg, "BLSTM"))


def _block(layout: list[Block], stack: str, layer: int, name: str) -> Block:
    return next(b for b in layout if (b["stack"], b["layer"], b["name"]) == (stack, layer, name))


def _slice(pack: NDArray[np.float64], b: Block) -> NDArray[np.float64]:
    return pack[b["offset"] : b["offset"] + b["len"]]


def _length(cfg: dict[str, str]) -> int:
    return sum(b["len"] for b in _layout(cfg))


def cell_cfg(direction: str = "bidirectional", b: int = 4, layers: int = 1, lstm: str = "5,3,4", lsub: str = "2,1", cell: str = "cfc") -> dict[str, str]:
    """A small synthetic net: 2 recurrent layers, a 2-layer output MLP, the `Cfc_*` geometry."""
    hidden = int(lstm.split(",")[-1])
    return {
        "BLSTM_LSTMNeuronNb": lstm,
        "BLSTM_LSTMSubSampling": lsub,
        "BLSTM_OutputNeuronNb": f"{hidden if direction == 'forward' else 2 * hidden},3,1",
        "BLSTM_OutputSubSampling": "1,1",
        "BLSTM_NNetInputSize": lstm.split(",")[0],
        "BLSTM_InputNormalizationType": "1",
        "BLSTM_TwoSweeps": "false",
        "BLSTM_Cell_Type": cell,
        "BLSTM_Direction": direction,
        "Cfc_Backbone_Units": str(b),
        "Cfc_Backbone_Layers": str(layers),
    }


def layer_cfg(out: int, fin: int, b: int, layers: int) -> dict[str, str]:
    """ONE forward CfC layer `(fin -> out)` plus a one-layer output MLP, so the layer-0 run is
    the whole recurrent part of the pack."""
    return {
        "BLSTM_LSTMNeuronNb": f"{fin},{out}",
        "BLSTM_LSTMSubSampling": "1",
        "BLSTM_OutputNeuronNb": f"{out},1",
        "BLSTM_OutputSubSampling": "1",
        "BLSTM_NNetInputSize": str(fin),
        "BLSTM_InputNormalizationType": "1",
        "BLSTM_TwoSweeps": "false",
        "BLSTM_Cell_Type": "cfc",
        "BLSTM_Direction": "forward",
        "Cfc_Backbone_Units": str(b),
        "Cfc_Backbone_Layers": str(layers),
    }


def _layer0(cfg: dict[str, str], pack: NDArray[np.float64]) -> NDArray[np.float64]:
    blocks = [x for x in _layout(cfg) if (x["stack"], x["layer"]) == ("forward", 0)]
    return pack[blocks[0]["offset"] : blocks[-1]["offset"] + blocks[-1]["len"]]


# ------------------------------------------------------------------------------------- #
# Layer-level: the S1.5 rules, block by block
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize(("out", "fin", "b", "layers"), SHAPES)
def test_layer_length_is_the_closed_form(out: int, fin: int, b: int, layers: int) -> None:
    cfg = layer_cfg(out, fin, b, layers)
    blocks = [x for x in _layout(cfg) if (x["stack"], x["layer"]) == ("forward", 0)]
    assert sum(x["len"] for x in blocks) == b * (fin + out + 1) + (layers - 1) * b * (b + 1) + 3 * out * (b + 1)
    assert [x["name"] for x in blocks] == [f"{k}_bb{li}" for li in range(layers) for k in ("W", "b")] + ["W_ff1", "b_ff1", "W_ff2", "b_ff2", "W_g", "b_g"]


@pytest.mark.parametrize(("out", "fin", "b", "layers"), SHAPES)
def test_every_bias_block_is_exactly_zero(out: int, fin: int, b: int, layers: int) -> None:
    """Spec S1.5: biases 0 EVERYWHERE. CfC has no forget-gate analogue to seed at 1 (the
    interpolation gate is a plain logistic between two live heads, not a memory valve), and
    no reference-lineage constants at all -- unlike Mamba's S3.4 block."""
    cfg = layer_cfg(out, fin, b, layers)
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(1))[0]
    biases = [x for x in _layout(cfg) if x["kind"] == "bias" and x["stack"] == "forward"]
    assert len(biases) == layers + 3
    for x in biases:
        assert np.all(_slice(flat, x) == 0.0), x["name"]


@pytest.mark.parametrize(("out", "fin", "b", "layers"), SHAPES)
def test_every_weight_block_is_drawn_and_finite(out: int, fin: int, b: int, layers: int) -> None:
    """Non-vacuity for the zero-bias pins above: a zero-filled projection would make them
    pass trivially."""
    cfg = layer_cfg(out, fin, b, layers)
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(2))[0]
    weights = [x for x in _layout(cfg) if x["kind"] == "weight" and x["stack"] == "forward"]
    assert len(weights) == layers + 3
    for x in weights:
        block = _slice(flat, x)
        assert np.all(np.isfinite(block)) and np.all(block != 0.0), x["name"]


def test_xavier_and_he_scale_the_same_block_differently() -> None:
    """The scheme reaches the blocks: He-normal (`sqrt(2/fan_in)`) and Xavier-uniform
    (`sqrt(6/(fan_in+fan_out))`, std = limit/sqrt(3)) have different standard deviations at
    the same fans -- the layer-0 backbone's being the CONCATENATED `in + h`."""
    cfg = layer_cfg(8, 8, 32, 1)
    xav = init_weights(cfg, "BLSTM", np.random.default_rng(3), "xavier")[0]
    he = init_weights(cfg, "BLSTM", np.random.default_rng(3), "he")[0]
    w = _block(_layout(cfg), "forward", 0, "W_bb0")
    fan_in, fan_out = 8 + 8, 32
    assert (w["fan_in"], w["fan_out"]) == (fan_in, fan_out)
    want = math.sqrt(2.0 / fan_in) / (math.sqrt(6.0 / (fan_in + fan_out)) / math.sqrt(3.0))
    assert np.std(_slice(he, w)) / np.std(_slice(xav, w)) == pytest.approx(want, rel=0.25)


# ------------------------------------------------------------------------------------- #
# THE RNG-ORDER RECONSTRUCTION PIN (the phase-9 shear lesson): the three heads are same-shape,
# same-fan draws, so a swap is invisible to every band and length pin and to an Engine round
# trip. Reconstructing the layer from a fresh generator pins position, scale, shape and order.
# ------------------------------------------------------------------------------------- #


def _xavier(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(rng.uniform(-limit, limit, size=shape), dtype=np.float64).reshape(-1)


def _he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    return np.asarray(rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape), dtype=np.float64).reshape(-1)


def _blk(rng: np.random.Generator, scheme: str, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    return _xavier(rng, shape, fan_in, fan_out) if scheme == "xavier" else _he(rng, shape, fan_in)


def _reconstruct(out: int, fin: int, b: int, layers: int, scheme: str, seed: int, head_order: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    want: list[np.ndarray] = []
    for li in range(layers):
        fan_in = fin + out if li == 0 else b
        want.append(_blk(rng, scheme, (b, fan_in), fan_in, b))
        want.append(np.zeros(b))
    heads = []
    for _ in range(3):  # [ff1 | ff2 | gate]
        heads.append(_blk(rng, scheme, (out, b), b, out))
    for k in head_order:
        want.append(heads[k])
        want.append(np.zeros(out))
    return np.concatenate(want)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
@pytest.mark.parametrize(("out", "fin", "b", "layers"), SHAPES)
def test_layer_is_reproducible_block_by_block(out: int, fin: int, b: int, layers: int, scheme: str) -> None:
    cfg = layer_cfg(out, fin, b, layers)
    got = init_weights(cfg, "BLSTM", np.random.default_rng(31), scheme)[0]  # type: ignore[arg-type]
    assert np.array_equal(_layer0(cfg, got), _reconstruct(out, fin, b, layers, scheme, 31))


def test_reconstruction_rejects_a_head_swap() -> None:
    """Non-vacuity for the pin above: `W_ff1` and `W_ff2` are same-shape, same-fan,
    same-scheme draws; only the stream position tells them apart."""
    cfg = layer_cfg(3, 5, 4, 1)
    got = _layer0(cfg, init_weights(cfg, "BLSTM", np.random.default_rng(31))[0])
    swapped = _reconstruct(3, 5, 4, 1, "xavier", 31, head_order=(1, 0, 2))
    assert got.shape == swapped.shape
    assert not np.array_equal(got, swapped)


# ------------------------------------------------------------------------------------- #
# Net-level assembly + determinism
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("direction", ["bidirectional", "forward"])
@pytest.mark.parametrize("layers", [1, 2])
def test_net_pack_length(direction: str, layers: int) -> None:
    cfg = cell_cfg(direction, b=6, layers=layers)
    packs = init_weights(cfg, "BLSTM", np.random.default_rng(0))
    assert len(packs) == 1
    assert packs[0].shape == (_length(cfg),)


def test_determinism_same_seed_is_bit_identical() -> None:
    cfg = cell_cfg()
    a = init_weights(cfg, "BLSTM", np.random.default_rng(11))[0]
    b = init_weights(cfg, "BLSTM", np.random.default_rng(11))[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, init_weights(cfg, "BLSTM", np.random.default_rng(12))[0])


# ------------------------------------------------------------------------------------- #
# Config bridge: the two `Cfc_*` keys reach the spec, through the engine
# ------------------------------------------------------------------------------------- #


def _tuple_a_cfg() -> dict[str, str]:
    return config_bridge.parse_legacy_config((REF / "tupleA_1_worker_1.config").read_text())


def test_nnet_spec_defaults_to_the_sized_geometry() -> None:
    spec = config_bridge.nnet_spec(_tuple_a_cfg(), prefix="BLSTM")
    assert spec["Cfc"] == {"backbone_units": SIZED_BACKBONE_UNITS, "backbone_layers": 1}
    assert spec["CellType"] == "lstm"  # the geometry is inert unless the cell is selected.


def test_nnet_spec_reads_the_port_only_keys() -> None:
    cfg = _tuple_a_cfg()
    cfg["BLSTM_Cell_Type"] = "cfc"
    cfg["Cfc_Backbone_Units"] = "12"
    cfg["Cfc_Backbone_Layers"] = "3"
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    assert spec["CellType"] == "cfc"
    # UNPREFIXED by design (one geometry per config, spec S1.4/S2) -- read the same way for
    # the LID net.
    assert spec["Cfc"] == {"backbone_units": 12, "backbone_layers": 3}


@pytest.mark.parametrize("key", ["Cfc_Backbone_Units", "Cfc_Backbone_Layers"])
def test_degenerate_cfc_geometry_is_rejected(key: str) -> None:
    """`CfcParams::from_legacy` (blstm.rs) hard-bails below 1; the seam carries the refusal
    to both readers as a `ValueError` naming the key, never a degenerate pack."""
    cfg = _tuple_a_cfg()
    cfg[key] = "0"
    with pytest.raises(ValueError, match=key):
        config_bridge.nnet_spec(cfg, prefix="BLSTM")
    with pytest.raises(ValueError, match=key):
        init_weights({**cfg, "BLSTM_Cell_Type": "cfc"}, "BLSTM", np.random.default_rng(0))


def test_the_default_is_the_rust_default() -> None:
    """The sizing lives in ONE language since issue #27: `blstm.rs::CFC_DEFAULT_BACKBONE_UNITS`
    sizes the net the engine builds AND the pack Python seeds, because both read it through
    the seam. This literal is the documented value, pinned against what the engine says."""
    cfg = {**_tuple_a_cfg(), "BLSTM_Cell_Type": "cfc"}
    w = _block(_layout(cfg), "forward", 0, "W_bb0")
    assert w["shape"] == [SIZED_BACKBONE_UNITS, 92 + 24]
    assert cast(dict[str, int], config_bridge.nnet_spec(cfg)["Cfc"])["backbone_units"] == SIZED_BACKBONE_UNITS


# ------------------------------------------------------------------------------------- #
# THE SIZING (spec S1.4) -- the arithmetic is in this module's docstring, measured off the seam
# ------------------------------------------------------------------------------------- #


def _lineage(lstm: str, cell: str, b: int = SIZED_BACKBONE_UNITS, direction: str = "bidirectional") -> dict[str, str]:
    outn = "24,12,1" if direction == "forward" else "48,12,1"
    return cell_cfg(direction, b=b, layers=1, lstm=lstm, lsub="4,1", cell=cell) | {"BLSTM_OutputNeuronNb": outn}


_V1, _V2 = "23,24,24", "11,24,24"


def test_lineage_pack_lengths_are_the_documented_arithmetic() -> None:
    """The two closed forms quoted in the module docstring, MEASURED off the engine. `620*B +
    935` (v1) and `524*B + 911` (v2) are what the sizing decision rests on."""
    assert _length(_lineage(_V1, "lstm")) == 33671  # the committed tuple-A pack length.
    assert _length(_lineage(_V2, "lstm")) == 24431
    for b in (1, 24, 45, 53):
        assert _length(_lineage(_V1, "cfc", b)) == 620 * b + 935
        assert _length(_lineage(_V2, "cfc", b)) == 524 * b + 911


def test_the_default_matches_the_v2_lstm_pack_within_15_percent() -> None:
    """THE SIZING RULE (spec S1.4/S8.2). The default is sized against the v2 lineage: at
    B = 45 the CfC pack is +0.25% of v2's LSTM pack, and the neighbouring integers are
    strictly worse, so this is the optimum and not merely an in-band value."""
    lstm = _length(_lineage(_V2, "lstm"))
    at = {b: _length(_lineage(_V2, "cfc", b)) for b in (44, 45, 46)}
    rel = {b: abs(v - lstm) / lstm for b, v in at.items()}
    assert at[SIZED_BACKBONE_UNITS] == 24491
    assert rel[SIZED_BACKBONE_UNITS] < 0.01
    assert rel[45] < rel[44] and rel[45] < rel[46]


def test_the_default_also_leaves_the_v1_lineage_in_band() -> None:
    """v1 CfC is a MECHANICAL leg with no param-match requirement, so this is a bonus, not a
    gate -- but it is the reason no per-config override of the default is needed anywhere:
    B = 45 is simultaneously v2's optimum and the SMALLEST integer inside v1's band."""
    lstm = _length(_lineage(_V1, "lstm"))
    assert abs(_length(_lineage(_V1, "cfc", SIZED_BACKBONE_UNITS)) - lstm) / lstm < 0.15
    assert abs(_length(_lineage(_V1, "cfc", SIZED_BACKBONE_UNITS - 1)) - lstm) / lstm > 0.15
    # v1's own optimum, quoted in the docstring, for the record.
    assert _length(_lineage(_V1, "cfc", 53)) == 33795


def test_forward_lineages_keep_the_same_ratios() -> None:
    """`cell_overlay` resizes the MLP input from 2h to h under `forward`, so both packs shed
    the same MLP rows and one stack -- the band verdict must not flip with the direction."""
    for lstm, want_ratio in ((_V2, 0.01), (_V1, 0.15)):
        ref = _length(_lineage(lstm, "lstm", direction="forward"))
        assert abs(_length(_lineage(lstm, "cfc", direction="forward")) - ref) / ref < want_ratio


# ------------------------------------------------------------------------------------- #
# Baseline arm knob (spec S7.2's tuple, fourth value)
# ------------------------------------------------------------------------------------- #

_SAD_FLAT: dict[str, str] = {
    "Algo_choice": "3",
    "BLSTM_LSTMNeuronNb": "23,24,24",
    "BLSTM_LSTMSubSampling": "4,1",
    "BLSTM_OutputNeuronNb": "48,12,1",
    "BLSTM_OutputSubSampling": "1,1",
    "fileslisting": "x.flst",
}


def test_cfc_knob_overlays_the_cell_key_only() -> None:
    """The `Cfc_*` geometry keys are deliberately NOT written (the `Mamba_*` precedent): the
    defaults live Rust-side and reach Python through the seam, so omitting them keeps the
    config text minimal and the two sides agreeing by construction."""
    assert baseline.cell_overlay(_SAD_FLAT, "cfc", "bidirectional") == {"BLSTM_Cell_Type": "cfc"}


def test_cfc_knob_composes_with_forward() -> None:
    assert baseline.cell_overlay(_SAD_FLAT, "cfc", "forward") == {
        "BLSTM_Cell_Type": "cfc",
        "BLSTM_Direction": "forward",
        "BLSTM_OutputNeuronNb": "24,12,1",
        "BLSTM_window": "0",
    }


def test_default_invocation_is_still_byte_identical() -> None:
    """The fourth cell is ADDITIVE: a default (`lstm`/`bidirectional`) run's overlay is still
    EMPTY, so its assembled config text -- and therefore its config hash -- is unchanged."""
    assert baseline.cell_overlay(_SAD_FLAT, "lstm", "bidirectional") == {}


def test_parser_exposes_cfc() -> None:
    args = baseline.build_parser().parse_args(["sad", "--corpus-root", "/tmp/c", "--out-dir", "/tmp/o", "--cell", "cfc"])
    assert args.cell == "cfc"


def test_overlaid_config_seeds_through_the_matching_builder() -> None:
    """End to end through the arm surface: the overlay lands on the flat config, the engine
    reads BOTH the cell type and the (defaulted) geometry back, and `init_weights` emits a
    CfC-sized pack at the SIZED default -- the v1-lineage number quoted in the docstring."""
    cfg = {**_SAD_FLAT, **baseline.cell_overlay(_SAD_FLAT, "cfc", "bidirectional")}
    cfg |= {
        "BLSTM_NNetInputSize": "23",
        "BLSTM_InputNormalizationType": "1",
        "BLSTM_TwoSweeps": "false",
        "BLSTM_CostLawSpeech": "log",
        "BLSTM_CostLawNoSpeech": "log",
        **{f"BLSTM_{d}_Is{p}Active": "true" for d in ("Forward", "Backward") for p in ("CellsPeepholes", "GatesPeepholes", "GatesRecurrentPeepholes")},
    }
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(0))[0]
    assert flat.shape == (620 * SIZED_BACKBONE_UNITS + 935,) == (_length(cfg),)


# ------------------------------------------------------------------------------------- #
# The three landed cells are BYTE-UNTOUCHED by the `Cfc_*` keys
# ------------------------------------------------------------------------------------- #


def test_other_cells_are_unchanged_by_the_cfc_geometry_keys() -> None:
    """A config carrying explicit default `Cfc_*` keys produces the SAME bytes as one without,
    for every other cell: the geometry is inert unless the cell is selected."""
    for cell, length in (("lstm", 33671), ("slstm", None), ("mamba", None), ("transformer", None)):
        bare = {**_tuple_a_cfg(), "BLSTM_Cell_Type": cell}
        explicit = {**bare, "Cfc_Backbone_Units": str(SIZED_BACKBONE_UNITS), "Cfc_Backbone_Layers": "1"}
        a = init_weights(bare, "BLSTM", np.random.default_rng(99))[0]
        b = init_weights(explicit, "BLSTM", np.random.default_rng(99))[0]
        assert np.array_equal(a, b), cell
        if length is not None:
            assert a.shape == (length,)
