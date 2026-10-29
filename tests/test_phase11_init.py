"""Phase 11 Task 3 (re-based on the seam, issue #27): the transformer draw rules, the sizing of
`Transformer_D_Ff`, and the baseline arm knob.

THE SIZING PROCEDURE (spec S3) -- how `TRANSFORMER_DEFAULT_D_FF = 64` was confirmed. One layer
at `(fin, H, d_ff)` holds `H(fin+1) + H + 3H(H+1) + H(H+1) + H + d_ff(H+1) + H(d_ff+1)` weights,
LINEAR in `d_ff` with slope `2H + 1 = 49` per layer -- `196` per full SAD net (two layers, two
stacks). `window` and `heads` carry NO weight. Against the lineage LSTM packs (v1 33671, v2
24431) the full nets are `16199 + 196 d` (v1) and `13871 + 196 d` (v2); the +-15% bands are
v2 [36, 72] and v1 [64, 114], intersection [64, 72]; v2's own argmin (54) sits OUTSIDE v1's
band, so the "inside both bands" rule decides and 64 is its smallest member AND v2's pick
inside the feasible set. Every length below is READ off the engine (`speech_rs.weight_layout`),
the closed forms being the documented arithmetic they are pinned against.

Byte identity with the pre-#27 builder is pinned by `tests/test_seed_pack_regeneration.py`
(the committed `transformer_*` and Twin transformer packs).
"""

import math
from pathlib import Path

import numpy as np
import pytest
import speech_rs
from speech import config_bridge
from speech.drivers import baseline
from speech.init_weights import init_weights

from tests.layout_helpers import block_of, block_slice, draw, layer0, layout_of, pack_length

REF = Path("tests/reference_data/phase4d")

#: The finalized geometry (spec S2/S3): the RUST constants `TRANSFORMER_DEFAULT_*`, read back
#: through the seam by `test_the_defaults_are_the_rust_defaults` -- no Python mirror exists.
SIZED_D_FF = 64
DEFAULT_WINDOW = 64
DEFAULT_HEADS = 4

# (out, fin, d_ff): the degenerate 1x1x1, a square-ish shape, a wide fan-in, and a d_ff SMALLER
# than H (so `W_1` and `W_2` are non-square in opposite directions). `heads = 1` throughout.
SHAPES: tuple[tuple[int, int, int], ...] = ((1, 1, 1), (4, 5, 4), (2, 3, 7), (3, 7, 2), (6, 11, 5))


def cell_cfg(
    direction: str = "bidirectional", d_ff: int = 5, lstm: str = "5,3,4", lsub: str = "2,1", cell: str = "transformer", heads: int = 1
) -> dict[str, str]:
    """A small synthetic net: 2 recurrent layers, a 2-layer output MLP, the `Transformer_*`
    geometry (one head, which divides every width)."""
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
        "Transformer_Heads": str(heads),
        "Transformer_D_Ff": str(d_ff),
    }


def layer_cfg(out: int, fin: int, d_ff: int) -> dict[str, str]:
    """ONE forward transformer layer `(fin -> out)` plus a one-layer output MLP."""
    return {
        "BLSTM_LSTMNeuronNb": f"{fin},{out}",
        "BLSTM_LSTMSubSampling": "1",
        "BLSTM_OutputNeuronNb": f"{out},1",
        "BLSTM_OutputSubSampling": "1",
        "BLSTM_NNetInputSize": str(fin),
        "BLSTM_InputNormalizationType": "1",
        "BLSTM_TwoSweeps": "false",
        "BLSTM_Cell_Type": "transformer",
        "BLSTM_Direction": "forward",
        "Transformer_Heads": "1",
        "Transformer_D_Ff": str(d_ff),
    }


NAMES = ["W_a", "b_a", "g_1", "W_qkv", "b_qkv", "W_o", "b_o", "g_2", "W_1", "b_1", "W_2", "b_2"]


# ------------------------------------------------------------------------------------- #
# Layer-level: the S1.2 layout and the init rules, block by block
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_layer_length_is_the_closed_form(out: int, fin: int, d_ff: int) -> None:
    cfg = layer_cfg(out, fin, d_ff)
    blocks = [x for x in layout_of(cfg) if (x["stack"], x["layer"]) == ("forward", 0)]
    assert [x["name"] for x in blocks] == NAMES
    assert sum(x["len"] for x in blocks) == out * (fin + 1) + out + 3 * out * (out + 1) + out * (out + 1) + out + d_ff * (out + 1) + out * (d_ff + 1)


@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_every_bias_block_is_exactly_zero(out: int, fin: int, d_ff: int) -> None:
    """Every bias seeds 0, `b_qkv` included: its `b_k` third is structurally dead (softmax
    shift-invariance), pinned Rust-side on the gradient, and seeded like the rest."""
    cfg = layer_cfg(out, fin, d_ff)
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(1))[0]
    layout = layout_of(cfg)
    for name in ("b_a", "b_qkv", "b_o", "b_1", "b_2"):
        assert np.all(block_slice(flat, block_of(layout, "forward", 0, name)) == 0.0), name


@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_the_two_rmsnorm_gains_are_exactly_one(out: int, fin: int, d_ff: int) -> None:
    cfg = layer_cfg(out, fin, d_ff)
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(1))[0]
    layout = layout_of(cfg)
    for name in ("g_1", "g_2"):
        g = block_of(layout, "forward", 0, name)
        assert g["kind"] == "gain"
        assert np.all(block_slice(flat, g) == 1.0), name


@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_every_weight_block_is_drawn_and_finite(out: int, fin: int, d_ff: int) -> None:
    cfg = layer_cfg(out, fin, d_ff)
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(2))[0]
    layout = layout_of(cfg)
    for name in ("W_a", "W_qkv", "W_o", "W_1", "W_2"):
        block = block_slice(flat, block_of(layout, "forward", 0, name))
        assert np.all(np.isfinite(block)) and np.all(block != 0.0), name


def test_xavier_and_he_scale_the_same_block_differently() -> None:
    """The scheme reaches the blocks: `W_qkv` is ONE combined matrix with ONE fan pair
    `(H, 3H)`, the layout says so, and the He / Xavier std ratio follows."""
    cfg = layer_cfg(16, 24, 32)
    xav = init_weights(cfg, "BLSTM", np.random.default_rng(4), "xavier")[0]
    he = init_weights(cfg, "BLSTM", np.random.default_rng(4), "he")[0]
    w = block_of(layout_of(cfg), "forward", 0, "W_qkv")
    fan_in, fan_out = 16, 48
    assert (w["fan_in"], w["fan_out"]) == (fan_in, fan_out)
    want = math.sqrt(2.0 / fan_in) / (math.sqrt(6.0 / (fan_in + fan_out)) / math.sqrt(3.0))
    assert np.std(block_slice(he, w)) / np.std(block_slice(xav, w)) == pytest.approx(want, rel=0.25)


# ------------------------------------------------------------------------------------- #
# THE RNG-ORDER RECONSTRUCTION PIN (the phase-9/10 shear lesson). This pack admits two
# permutations invisible to every band and length pin: the q/k/v row families inside `W_qkv`,
# and `W_1 (d_ff x H)` vs `W_2 (H x d_ff)` -- the SAME element count and, under Xavier, the
# SAME limit (symmetric in the fans). What it cannot pin, stated: the q/k/v BIAS thirds, all 0.
# ------------------------------------------------------------------------------------- #


def _reconstruct(out: int, fin: int, d_ff: int, scheme: str, seed: int, swap_ffn: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = [draw(rng, scheme, (out, fin), fin, out), np.zeros(out), np.ones(out)]
    parts += [draw(rng, scheme, (3 * out, out), out, 3 * out), np.zeros(3 * out)]
    parts += [draw(rng, scheme, (out, out), out, out), np.zeros(out), np.ones(out)]
    if swap_ffn:  # the two FFN draws REVERSED, each result left in the slot its own line occupies
        parts += [draw(rng, scheme, (out, d_ff), d_ff, out), np.zeros(d_ff), draw(rng, scheme, (d_ff, out), out, d_ff), np.zeros(out)]
    else:
        parts += [draw(rng, scheme, (d_ff, out), out, d_ff), np.zeros(d_ff), draw(rng, scheme, (out, d_ff), d_ff, out), np.zeros(out)]
    return np.concatenate(parts)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_layer_is_reproducible_block_by_block(out: int, fin: int, d_ff: int, scheme: str) -> None:
    cfg = layer_cfg(out, fin, d_ff)
    got = init_weights(cfg, "BLSTM", np.random.default_rng(41), scheme)[0]  # type: ignore[arg-type]
    assert np.array_equal(layer0(cfg, got), _reconstruct(out, fin, d_ff, scheme, 41))


@pytest.mark.parametrize(("scheme", "same_bytes"), [("xavier", True), ("he", False)])
def test_the_ffn_draw_order_is_only_visible_under_he(scheme: str, same_bytes: bool) -> None:
    """The measured LIMIT of the pin above, recorded rather than left as a surprise: reversing
    the two FFN draws is a BIT-LEVEL NO-OP under Xavier (same limit over the same count) and
    IS visible under He (`sqrt(2/fan_in)` distinguishes `H` from `d_ff`). The two schemes miss
    in OPPOSITE directions, which is what makes parametrizing over both load-bearing."""
    out, fin, d_ff = 3, 7, 5  # out != d_ff, so He's fan_in genuinely differs between the two.
    cfg = layer_cfg(out, fin, d_ff)
    got = init_weights(cfg, "BLSTM", np.random.default_rng(41), scheme)[0]  # type: ignore[arg-type]
    assert np.array_equal(layer0(cfg, got), _reconstruct(out, fin, d_ff, scheme, 41, swap_ffn=True)) is same_bytes


# ------------------------------------------------------------------------------------- #
# Net-level assembly + determinism
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("direction", ["bidirectional", "forward"])
@pytest.mark.parametrize("d_ff", [1, 5])
def test_net_pack_length(direction: str, d_ff: int) -> None:
    cfg = cell_cfg(direction, d_ff)
    packs = init_weights(cfg, "BLSTM", np.random.default_rng(0))
    assert len(packs) == 1
    assert packs[0].shape == (pack_length(cfg),)


def test_window_and_heads_never_move_the_pack() -> None:
    """Spec S3's premise, pinned rather than asserted in prose: ALiBi's slopes are constants
    and `heads` only reshapes the same `W_qkv`, so `d_ff` is the ONLY sized knob -- the layout
    does not list them and the bytes do not move (`heads` varied on widths it divides)."""
    base = init_weights(cell_cfg(), "BLSTM", np.random.default_rng(8))[0]
    for window in (1, 8, 4096):
        cfg = cell_cfg() | {"Transformer_Window": str(window)}
        assert np.array_equal(init_weights(cfg, "BLSTM", np.random.default_rng(8))[0], base), window
        assert layout_of(cfg) == layout_of(cell_cfg())
    wide = cell_cfg(lstm="5,4,4", heads=1)
    base = init_weights(wide, "BLSTM", np.random.default_rng(8))[0]
    for heads in (2, 4):
        cfg = cell_cfg(lstm="5,4,4", heads=heads)
        assert np.array_equal(init_weights(cfg, "BLSTM", np.random.default_rng(8))[0], base), heads
        assert layout_of(cfg) == layout_of(wide)


def test_determinism_same_seed_is_bit_identical() -> None:
    cfg = cell_cfg()
    a = init_weights(cfg, "BLSTM", np.random.default_rng(11))[0]
    b = init_weights(cfg, "BLSTM", np.random.default_rng(11))[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, init_weights(cfg, "BLSTM", np.random.default_rng(12))[0])


def test_a_non_dividing_head_count_is_refused() -> None:
    """`BlstmConfig::from_legacy` owns the `H % heads == 0` check; the seam carries it as a
    `ValueError` naming the key, before any draw."""
    with pytest.raises(ValueError, match="Transformer_Heads"):
        init_weights(cell_cfg(heads=5), "BLSTM", np.random.default_rng(0))


# ------------------------------------------------------------------------------------- #
# Config bridge: the three `Transformer_*` keys reach the spec, through the engine
# ------------------------------------------------------------------------------------- #


def _tuple_a_cfg() -> dict[str, str]:
    return config_bridge.parse_legacy_config((REF / "tupleA_1_worker_1.config").read_text())


def test_blstm_config_defaults_to_the_sized_geometry() -> None:
    spec = speech_rs.blstm_config(_tuple_a_cfg(), "BLSTM")
    assert spec["Transformer"] == {"window": DEFAULT_WINDOW, "heads": DEFAULT_HEADS, "d_ff": SIZED_D_FF}
    assert spec["CellType"] == "lstm"  # the geometry is inert unless the cell is selected.


def test_blstm_config_reads_the_port_only_keys() -> None:
    cfg = _tuple_a_cfg()
    cfg["BLSTM_Cell_Type"] = "transformer"
    cfg["Transformer_Window"] = "16"
    cfg["Transformer_Heads"] = "8"
    cfg["Transformer_D_Ff"] = "40"
    spec = speech_rs.blstm_config(cfg, "BLSTM")
    assert spec["CellType"] == "transformer"
    assert spec["Transformer"] == {"window": 16, "heads": 8, "d_ff": 40}


@pytest.mark.parametrize("key", ["Transformer_Window", "Transformer_Heads", "Transformer_D_Ff"])
def test_degenerate_transformer_geometry_is_rejected(key: str) -> None:
    """`TransformerParams::from_legacy` (blstm.rs) hard-bails below 1; the seam carries the
    refusal to both readers as a `ValueError` naming the key."""
    cfg = _tuple_a_cfg()
    cfg[key] = "0"
    with pytest.raises(ValueError, match=key):
        speech_rs.blstm_config(cfg, "BLSTM")
    with pytest.raises(ValueError, match=key):
        init_weights({**cfg, "BLSTM_Cell_Type": "transformer"}, "BLSTM", np.random.default_rng(0))


def test_the_defaults_are_the_rust_defaults() -> None:
    """The geometry lives in ONE language since issue #27: `blstm.rs::TRANSFORMER_DEFAULT_*`
    size the net the engine builds AND the pack Python seeds, both through the seam. These
    literals are the documented values, pinned against what the engine says."""
    cfg = {**_tuple_a_cfg(), "BLSTM_Cell_Type": "transformer"}
    w1 = block_of(layout_of(cfg), "forward", 0, "W_1")
    assert w1["shape"] == [SIZED_D_FF, 24]
    assert speech_rs.blstm_config(cfg, "BLSTM")["Transformer"] == {"window": DEFAULT_WINDOW, "heads": DEFAULT_HEADS, "d_ff": SIZED_D_FF}


# ------------------------------------------------------------------------------------- #
# THE SIZING (spec S3) -- the arithmetic is in this module's docstring, measured off the seam
# ------------------------------------------------------------------------------------- #

_V1, _V2 = "23,24,24", "11,24,24"
_BAND = 0.15


def _lineage(lstm: str, cell: str, d_ff: int = SIZED_D_FF, direction: str = "bidirectional") -> dict[str, str]:
    outn = "24,12,1" if direction == "forward" else "48,12,1"
    return cell_cfg(direction, d_ff, lstm=lstm, lsub="4,1", cell=cell, heads=DEFAULT_HEADS) | {"BLSTM_OutputNeuronNb": outn}


def _dev(lstm: str, d_ff: int, direction: str = "bidirectional") -> float:
    ref = pack_length(_lineage(lstm, "lstm", direction=direction))
    return (pack_length(_lineage(lstm, "transformer", d_ff, direction)) - ref) / ref


def test_lineage_pack_lengths_are_the_documented_arithmetic() -> None:
    """The two closed forms quoted in the module docstring, MEASURED off the engine. `16199 +
    196*d` (v1) and `13871 + 196*d` (v2) are what the sizing decision rests on."""
    assert pack_length(_lineage(_V1, "lstm")) == 33671  # the committed tuple-A pack length.
    assert pack_length(_lineage(_V2, "lstm")) == 24431
    for d in (1, 54, 64, 89):
        assert pack_length(_lineage(_V1, "transformer", d)) == 16199 + 196 * d
        assert pack_length(_lineage(_V2, "transformer", d)) == 13871 + 196 * d


def test_the_default_is_inside_both_lineage_bands() -> None:
    """THE SIZING RULE (spec S3): prefer an integer inside BOTH bands. 64 is, and it is the
    SMALLEST such integer -- 63 misses v1 by 0.22 percentage points."""
    assert abs(_dev(_V2, SIZED_D_FF)) < _BAND
    assert abs(_dev(_V1, SIZED_D_FF)) < _BAND
    assert pack_length(_lineage(_V2, "transformer")) == 26415
    assert pack_length(_lineage(_V1, "transformer")) == 28743
    assert abs(_dev(_V1, SIZED_D_FF - 1)) > _BAND  # 63 -> v1 at -15.22%, OUT.
    assert _dev(_V2, SIZED_D_FF) == pytest.approx(0.0812, abs=5e-5)
    assert _dev(_V1, SIZED_D_FF) == pytest.approx(-0.1464, abs=5e-5)


def test_the_two_bands_and_their_intersection() -> None:
    """The feasible set, from the two linear forms the test above measured: v2 tolerates
    [36, 72], v1 tolerates [64, 114], so the intersection is [64, 72], NON-EMPTY -- which is why
    the spec's v2-wins tiebreak never had to fire. v2's own argmin (54) sits OUTSIDE v1's band
    (-20.46%): the "inside both" rule decides, and v2's pick inside the feasible set is 64."""
    lstm_v1, lstm_v2 = pack_length(_lineage(_V1, "lstm")), pack_length(_lineage(_V2, "lstm"))
    dev_v1 = {d: (16199 + 196 * d - lstm_v1) / lstm_v1 for d in range(1, 200)}
    dev_v2 = {d: (13871 + 196 * d - lstm_v2) / lstm_v2 for d in range(1, 200)}
    v2_band = [d for d in range(1, 200) if abs(dev_v2[d]) <= _BAND]
    v1_band = [d for d in range(1, 200) if abs(dev_v1[d]) <= _BAND]
    assert (v2_band[0], v2_band[-1]) == (36, 72)
    assert (v1_band[0], v1_band[-1]) == (64, 114)
    both = sorted(set(v2_band) & set(v1_band))
    assert both == list(range(64, 73)) and both[0] == SIZED_D_FF
    argmin_v2 = min(dev_v2, key=lambda d: abs(dev_v2[d]))
    assert argmin_v2 == 54 and pack_length(_lineage(_V2, "transformer", 54)) == 24455
    assert dev_v1[argmin_v2] == pytest.approx(-0.2046, abs=5e-5)  # THE CONFLICT
    assert min(both, key=lambda d: abs(dev_v2[d])) == SIZED_D_FF
    assert min(dev_v1, key=lambda d: abs(dev_v1[d])) == 89  # v1's own argmin, for the record.


def test_forward_lineages_keep_the_same_verdict() -> None:
    """`cell_overlay` resizes the MLP input from 2H to H under `forward`, so both packs shed the
    same MLP rows and one stack -- the band verdict must not flip with the direction."""
    for lstm, lstm_want, trans_want in ((_V2, 12239, 13231), (_V1, 16871, 14407)):
        assert pack_length(_lineage(lstm, "lstm", direction="forward")) == lstm_want
        assert pack_length(_lineage(lstm, "transformer", direction="forward")) == trans_want
        assert abs(_dev(lstm, SIZED_D_FF, "forward")) < _BAND


# ------------------------------------------------------------------------------------- #
# The four landed cells are BYTE-UNTOUCHED by the `Transformer_*` keys
# ------------------------------------------------------------------------------------- #


def test_other_cells_are_unchanged_by_the_transformer_geometry_keys() -> None:
    for cell, length in (("lstm", 33671), ("slstm", None), ("mamba", None), ("cfc", None)):
        bare = {**_tuple_a_cfg(), "BLSTM_Cell_Type": cell}
        explicit = {**bare, "Transformer_Window": str(DEFAULT_WINDOW), "Transformer_Heads": str(DEFAULT_HEADS), "Transformer_D_Ff": str(SIZED_D_FF)}
        a = init_weights(bare, "BLSTM", np.random.default_rng(99))[0]
        b = init_weights(explicit, "BLSTM", np.random.default_rng(99))[0]
        assert np.array_equal(a, b), cell
        if length is not None:
            assert a.shape == (length,)


# ------------------------------------------------------------------------------------- #
# Baseline arm knob (phase-11 T9 fix round 1, review F2)
# ------------------------------------------------------------------------------------- #

_SAD_FLAT: dict[str, str] = {
    "Algo_choice": "3",
    "BLSTM_LSTMNeuronNb": "23,24,24",
    "BLSTM_LSTMSubSampling": "4,1",
    "BLSTM_OutputNeuronNb": "48,12,1",
    "BLSTM_OutputSubSampling": "1,1",
    "fileslisting": "x.flst",
}


def test_transformer_knob_overlays_the_cell_key_only() -> None:
    """Bidirectional writes ONLY `BLSTM_Cell_Type` -- no third derived key, per
    `cell_overlay`'s own "THE FIFTH CELL NEEDS NO THIRD DERIVED KEY" docstring paragraph."""
    assert baseline.cell_overlay(_SAD_FLAT, "transformer", "bidirectional") == {"BLSTM_Cell_Type": "transformer"}


def test_transformer_knob_composes_with_forward() -> None:
    assert baseline.cell_overlay(_SAD_FLAT, "transformer", "forward") == {
        "BLSTM_Cell_Type": "transformer",
        "BLSTM_Direction": "forward",
        "BLSTM_OutputNeuronNb": "24,12,1",
        "BLSTM_window": "0",
    }


def test_parser_exposes_transformer() -> None:
    args = baseline.build_parser().parse_args(["sad", "--corpus-root", "/tmp/c", "--out-dir", "/tmp/o", "--cell", "transformer"])
    assert args.cell == "transformer"


def test_overlaid_config_seeds_through_the_matching_builder() -> None:
    """End to end through the arm surface: the overlay lands on the flat config, the engine
    reads it back, and `init_weights` emits a transformer-sized pack at the v1-lineage
    bidirectional length (`16199 + 196 * 64`)."""
    cfg = {**_SAD_FLAT, **baseline.cell_overlay(_SAD_FLAT, "transformer", "bidirectional")}
    cfg |= {
        "BLSTM_NNetInputSize": "23",
        "BLSTM_InputNormalizationType": "1",
        "BLSTM_TwoSweeps": "false",
        "BLSTM_CostLawSpeech": "log",
        "BLSTM_CostLawNoSpeech": "log",
        **{f"BLSTM_{d}_Is{p}Active": "true" for d in ("Forward", "Backward") for p in ("CellsPeepholes", "GatesPeepholes", "GatesRecurrentPeepholes")},
    }
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(0))[0]
    assert flat.shape == (16199 + 196 * SIZED_D_FF,) == (pack_length(cfg),)
