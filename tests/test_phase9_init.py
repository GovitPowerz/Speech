"""Phase 9 Task 4 (re-based on the seam, issue #27): the sLSTM / Mamba draw rules, the
net-level assembly for every cell, and the baseline arm cell/direction knobs.

THE LAYOUT CONTRACT IS THE RUST LAYOUT, read through `speech_rs.weight_layout`: every block is
located by (stack, layer, name) off the engine's own list, and the per-cell pins here are about
the VALUES a block receives (the S2.4 / S3.4 rules), not about where it sits -- offsets are
pinned Rust-side by one-hot probes against `set_weights`. The ONE independent transcription
kept is the RNG-order reconstruction of a layer (the phase-9 "shear" lesson: a permutation of
same-shape, same-fan blocks is invisible to every band and length pin), which draws block by
block with helpers written out here rather than imported.

The bit-identical `speech_rs.Engine` round trip belongs to `tests/pyo3/test_phase9_seam.py`;
byte identity with the pre-#27 builders to `tests/test_seed_pack_regeneration.py`.
"""

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from speech import config_bridge
from speech.drivers import baseline
from speech.init_weights import init_weights

from tests.layout_helpers import block_of, block_slice, draw, layer0, layout_of, pack_length

REF = Path("tests/reference_data/phase4d")

TUPLE_A = config_bridge.parse_legacy_config((REF / "tupleA_1_worker_1.config").read_text())


def cell_cfg(cell: str, direction: str = "bidirectional", lstm: str = "5,3,4", **geometry: str) -> dict[str, str]:
    """A small synthetic net: 2 recurrent layers (fan-ins 10 and 3, widths 3 and 4), a 2-layer
    output MLP; `geometry` carries the unprefixed `Mamba_*` keys."""
    hidden = int(lstm.split(",")[-1])
    cfg = {
        "BLSTM_LSTMNeuronNb": lstm,
        "BLSTM_LSTMSubSampling": "2,1",
        "BLSTM_OutputNeuronNb": f"{hidden if direction == 'forward' else 2 * hidden},3,1",
        "BLSTM_OutputSubSampling": "1,1",
        "BLSTM_NNetInputSize": lstm.split(",")[0],
        "BLSTM_InputNormalizationType": "1",
        "BLSTM_TwoSweeps": "false",
        "BLSTM_Cell_Type": cell,
        "BLSTM_Direction": direction,
    }
    cfg.update(geometry)
    return cfg


# ------------------------------------------------------------------------------------- #
# sLSTM (spec S2.4 init)
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("forget_bias_one", [True, False])
def test_slstm_biases_follow_the_forget_rule(forget_bias_one: bool) -> None:
    cfg = cell_cfg("slstm")
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(1), forget_bias_one=forget_bias_one)[0]
    biases = [b for b in layout_of(cfg) if b["kind"] == "bias" and b["stack"] in ("forward", "backward")]
    assert {b["name"] for b in biases} == {"b_i", "b_f", "b_o", "b_z"}
    for b in biases:
        want = 1.0 if (b["name"] == "b_f" and forget_bias_one) else 0.0
        assert np.all(block_slice(flat, b) == want), (b["stack"], b["layer"], b["name"])


def test_slstm_weight_blocks_are_drawn_on_the_combined_fan() -> None:
    """`R` and `W` share one `fan_in = in + out` (one summed pre-activation), the layout says
    so, and the scheme reaches the values: He / Xavier std ratio at those fans."""
    cfg = cell_cfg("slstm", lstm="24,16,16")
    layout = layout_of(cfg)
    xav = init_weights(cfg, "BLSTM", np.random.default_rng(2), "xavier")[0]
    he = init_weights(cfg, "BLSTM", np.random.default_rng(2), "he")[0]
    r_i, w_i = block_of(layout, "forward", 0, "R_i"), block_of(layout, "forward", 0, "W_i")
    fan_in, fan_out = 24 * 2 + 16, 16
    assert (r_i["fan_in"], r_i["fan_out"]) == (fan_in, fan_out) == (w_i["fan_in"], w_i["fan_out"])
    for b in (r_i, w_i):
        assert np.all(block_slice(xav, b) != 0.0) and np.all(np.isfinite(block_slice(xav, b)))
    want = math.sqrt(2.0 / fan_in) / (math.sqrt(6.0 / (fan_in + fan_out)) / math.sqrt(3.0))
    block = slice(w_i["offset"], w_i["offset"] + w_i["len"])
    assert np.std(he[block]) / np.std(xav[block]) == pytest.approx(want, rel=0.25)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
def test_slstm_layer_is_reproducible_block_by_block(scheme: str) -> None:
    """Per gate `[i | f | o | z]`: `R (out, out)`, `W (out, in)`, `b` -- drawn in that order."""
    cfg = cell_cfg("slstm")
    out, fin = 3, 10
    rng = np.random.default_rng(21)
    want: list[np.ndarray] = []
    for gate in range(4):
        want.append(draw(rng, scheme, (out, out), fin + out, out))
        want.append(draw(rng, scheme, (out, fin), fin + out, out))
        want.append(np.full(out, 1.0 if gate == 1 else 0.0))
    got = init_weights(cfg, "BLSTM", np.random.default_rng(21), scheme)[0]  # type: ignore[arg-type]
    assert np.array_equal(layer0(cfg, got), np.concatenate(want))


# ------------------------------------------------------------------------------------- #
# Mamba (spec S3.4 init -- the lineage constants)
# ------------------------------------------------------------------------------------- #

MAMBA_GEOMS: tuple[dict[str, str], ...] = (
    {"Mamba_D_State": "2", "Mamba_D_Conv": "2", "Mamba_Expand": "1"},
    {"Mamba_D_State": "3", "Mamba_D_Conv": "4", "Mamba_Expand": "2", "Mamba_Dt_Rank": "2"},
    {},  # the S6 defaults (16/4/2/0), read from the Rust side
)


@pytest.mark.parametrize("geom", MAMBA_GEOMS)
def test_mamba_spec_pinned_constants(geom: dict[str, str]) -> None:
    cfg = cell_cfg("mamba", **geom)
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(3))[0]
    layout = layout_of(cfg)
    spec = cast(dict[str, int], config_bridge.nnet_spec(cfg)["Mamba"])
    for layer in (0, 1):
        d_inner, d_state = block_of(layout, "forward", layer, "A_log")["shape"]
        assert d_inner == spec["expand"] * int(cfg["BLSTM_LSTMNeuronNb"].split(",")[layer + 1])
        assert d_state == spec["d_state"]
        a_log = block_slice(flat, block_of(layout, "forward", layer, "A_log")).reshape(d_inner, d_state)
        assert np.array_equal(a_log, np.tile(np.log(np.arange(1, d_state + 1, dtype=np.float64)), (d_inner, 1)))
        assert np.all(block_slice(flat, block_of(layout, "forward", layer, "D")) == 1.0)
        assert np.all(block_slice(flat, block_of(layout, "forward", layer, "g")) == 1.0)
        assert np.all(block_slice(flat, block_of(layout, "forward", layer, "b_conv")) == 0.0)
        b_dt = block_slice(flat, block_of(layout, "forward", layer, "b_dt"))
        lo, hi = math.log(math.expm1(1e-3)), math.log(math.expm1(1e-1))
        assert np.all((b_dt >= lo - 1e-12) & (b_dt <= hi + 1e-12))
        assert np.unique(b_dt).size == d_inner  # per channel, not one shared draw
        w_dt = block_of(layout, "forward", layer, "W_dt")
        dt_rank = w_dt["shape"][1]
        assert dt_rank == (spec["dt_rank"] or max(1, -(-int(cfg["BLSTM_LSTMNeuronNb"].split(",")[layer + 1]) // 16)))
        assert np.all(np.abs(block_slice(flat, w_dt)) <= dt_rank**-0.5)
        conv = block_of(layout, "forward", layer, "conv")
        assert (conv["fan_in"], conv["fan_out"]) == (spec["d_conv"], spec["d_conv"])


def test_mamba_adapter_block_exists_only_when_widths_differ() -> None:
    cfg = cell_cfg("mamba", lstm="5,4,4", Mamba_D_State="2", Mamba_D_Conv="2", Mamba_Expand="1")
    layout = layout_of(cfg)
    names0 = [b["name"] for b in layout if (b["stack"], b["layer"]) == ("forward", 0)]  # in 10 != out 4
    names1 = [b["name"] for b in layout if (b["stack"], b["layer"]) == ("forward", 1)]  # in 4 == out 4
    assert names0[:3] == ["P", "p", "g"] and names1[0] == "g"
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(0))[0]
    assert np.all(block_slice(flat, block_of(layout, "forward", 0, "p")) == 0.0)
    assert np.all(block_slice(flat, block_of(layout, "forward", 0, "P")) != 0.0)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
def test_mamba_layer_is_reproducible_block_by_block(scheme: str) -> None:
    """S3.2 order at the S3.4 fans, drawn from a fresh generator -- position, scale, shape and
    draw order pinned at once (the shear lesson: `W_x` and `W_dt` are drawn from different
    distributions, so a swap moves draws across the stream)."""
    cfg = cell_cfg("mamba", Mamba_D_State="4", Mamba_D_Conv="3", Mamba_Expand="2")
    out, fin, d_state, d_conv = 3, 10, 4, 3
    d_inner, dt_rank = 2 * out, 1
    rng = np.random.default_rng(22)
    want: list[np.ndarray] = [
        draw(rng, scheme, (out, fin), fin, out),  # P
        np.zeros(out),  # p
        np.ones(out),  # g
        draw(rng, scheme, (2 * d_inner, out), out, 2 * d_inner),  # W_in
        draw(rng, scheme, (d_inner, d_conv), d_conv, d_conv),  # conv
        np.zeros(d_inner),  # b_conv
        draw(rng, scheme, (dt_rank + 2 * d_state, d_inner), d_inner, dt_rank + 2 * d_state),  # W_x
        np.asarray(rng.uniform(-(dt_rank**-0.5), dt_rank**-0.5, size=(d_inner, dt_rank))).reshape(-1),  # W_dt
    ]
    delta0 = np.exp(rng.uniform(math.log(1e-3), math.log(1e-1), size=d_inner))
    want.append(np.log(np.expm1(delta0)))  # b_dt = softplus^-1(Delta_0)
    want.append(np.tile(np.log(np.arange(1, d_state + 1, dtype=np.float64)), (d_inner, 1)).reshape(-1))  # A_log
    want.append(np.ones(d_inner))  # D
    want.append(draw(rng, scheme, (out, d_inner), d_inner, out))  # W_out
    got = layer0(cfg, init_weights(cfg, "BLSTM", np.random.default_rng(22), scheme)[0])  # type: ignore[arg-type]
    expected = np.concatenate(want)
    assert np.allclose(got, expected, rtol=0, atol=1e-13)
    # Bit-exact everywhere except `b_dt`, whose two softplus-inverse spellings agree to ~1e-13.
    b_dt = block_of(layout_of(cfg), "forward", 0, "b_dt")
    mask = np.ones(got.shape[0], dtype=bool)
    mask[b_dt["offset"] : b_dt["offset"] + b_dt["len"]] = False
    assert np.array_equal(got[mask], expected[mask])


# ------------------------------------------------------------------------------------- #
# Net-level assembly: cell stacks x direction + output MLP + normalize tail
# ------------------------------------------------------------------------------------- #

ALL_CELLS = ("lstm", "slstm", "mamba", "cfc", "transformer")
SMALL_GEOMETRY = {"Mamba_D_State": "2", "Mamba_D_Conv": "2", "Mamba_Expand": "1", "Cfc_Backbone_Units": "4", "Transformer_Heads": "1", "Transformer_D_Ff": "5"}


@pytest.mark.parametrize("cell", ALL_CELLS)
@pytest.mark.parametrize("direction", ["bidirectional", "forward"])
def test_net_pack_is_the_layout_pack_length(cell: str, direction: str) -> None:
    cfg = cell_cfg(cell, direction, **SMALL_GEOMETRY)
    packs = init_weights(cfg, "BLSTM", np.random.default_rng(0))
    assert len(packs) == 1
    assert packs[0].shape == (pack_length(cfg),)
    assert np.all(np.isfinite(packs[0]))


@pytest.mark.parametrize("cell", ALL_CELLS)
def test_forward_direction_drops_exactly_one_stack(cell: str) -> None:
    """Non-vacuity for the direction knob: dropping the backward stack removes EXACTLY one
    copy of the cell layers plus the halved MLP input (3 rows x hidden fewer weights)."""
    bi_cfg, fw_cfg = cell_cfg(cell, "bidirectional", **SMALL_GEOMETRY), cell_cfg(cell, "forward", **SMALL_GEOMETRY)
    bi = init_weights(bi_cfg, "BLSTM", np.random.default_rng(0))[0]
    fw = init_weights(fw_cfg, "BLSTM", np.random.default_rng(0))[0]
    stack = sum(b["len"] for b in layout_of(bi_cfg) if b["stack"] == "backward")
    assert stack > 0
    assert bi.shape[0] - fw.shape[0] == stack + 3 * 4


@pytest.mark.parametrize("cell", ALL_CELLS)
def test_normalize_tail_and_output_biases_for_every_cell(cell: str) -> None:
    cfg = cell_cfg(cell, **SMALL_GEOMETRY)
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(5))[0]
    layout = layout_of(cfg)
    assert np.all(block_slice(flat, block_of(layout, "tail", 0, "mean")) == 0.0)
    assert np.all(block_slice(flat, block_of(layout, "tail", 0, "std")) == 1.0)
    for layer in (0, 1):
        assert np.all(block_slice(flat, block_of(layout, "output", layer, "b")) == 0.0), layer
        assert np.all(block_slice(flat, block_of(layout, "output", layer, "W")) != 0.0), layer


@pytest.mark.parametrize("cell", ALL_CELLS)
def test_determinism_same_seed_is_bit_identical(cell: str) -> None:
    cfg = cell_cfg(cell, **SMALL_GEOMETRY)
    a = init_weights(cfg, "BLSTM", np.random.default_rng(11))[0]
    b = init_weights(cfg, "BLSTM", np.random.default_rng(11))[0]
    assert np.array_equal(a, b)
    c = init_weights(cfg, "BLSTM", np.random.default_rng(12))[0]
    assert not np.array_equal(a, c)


def test_unknown_scheme_raises_before_the_seam() -> None:
    with pytest.raises(ValueError, match="scheme"):
        init_weights(cell_cfg("slstm"), "BLSTM", np.random.default_rng(0), "bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(("key", "value"), [("Mamba_D_State", "0"), ("Mamba_D_Conv", "0"), ("Mamba_Expand", "0")])
def test_degenerate_mamba_geometry_is_rejected(key: str, value: str) -> None:
    """`MambaParams::from_legacy` (blstm.rs) hard-bails below 1; the seam carries that
    refusal to both `nnet_spec` and `init_weights` as a `ValueError` naming the key."""
    cfg = cell_cfg("mamba", **{key: value})
    with pytest.raises(ValueError, match=key):
        config_bridge.nnet_spec(cfg, prefix="BLSTM")
    with pytest.raises(ValueError, match=key):
        init_weights(cfg, "BLSTM", np.random.default_rng(0))
    # `dt_rank` may be 0 (= auto).
    assert cast(dict[str, int], config_bridge.nnet_spec(cell_cfg("mamba", Mamba_Dt_Rank="0"))["Mamba"])["dt_rank"] == 0


def test_unknown_cell_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown cell type"):
        init_weights(cell_cfg("gru"), "BLSTM", np.random.default_rng(0))


def test_unknown_direction_raises() -> None:
    with pytest.raises(ValueError, match="unknown direction"):
        init_weights(cell_cfg("lstm", "backward"), "BLSTM", np.random.default_rng(0))


# ------------------------------------------------------------------------------------- #
# The LSTM path is BYTE-UNTOUCHED by the port-only keys (spec S7.1)
# ------------------------------------------------------------------------------------- #


def test_lstm_path_is_byte_identical_with_and_without_the_new_keys() -> None:
    """A config carrying no cell/direction/geometry keys at all (every pre-phase-9 caller)
    and one carrying the explicit defaults produce the SAME bytes -- the dispatch is inert."""
    explicit = {
        **TUPLE_A,
        "BLSTM_Cell_Type": "lstm",
        "BLSTM_Direction": "bidirectional",
        "Mamba_D_State": "16",
        "Cfc_Backbone_Units": "45",
        "Transformer_D_Ff": "64",
    }
    a = init_weights(TUPLE_A, "BLSTM", np.random.default_rng(99))[0]
    b = init_weights(explicit, "BLSTM", np.random.default_rng(99))[0]
    assert np.array_equal(a, b)
    assert a.shape == (33671,)  # the tuple-A pin the phase-5 suite already carries.


# ------------------------------------------------------------------------------------- #
# Config bridge: the S6 keys reach the spec, through the engine
# ------------------------------------------------------------------------------------- #


def test_nnet_spec_defaults_to_the_legacy_shape() -> None:
    spec = config_bridge.nnet_spec(TUPLE_A, prefix="BLSTM")
    assert spec["CellType"] == "lstm"
    assert spec["Direction"] == "bidirectional"
    assert spec["Mamba"] == {"d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": 0}


def test_nnet_spec_reads_the_port_only_keys() -> None:
    cfg = dict(TUPLE_A)
    cfg["BLSTM_Cell_Type"] = "mamba"
    cfg["BLSTM_Direction"] = "forward"
    cfg["BLSTM_OutputNeuronNb"] = "24,12,1"
    cfg["Mamba_D_State"] = "8"
    cfg["Mamba_Expand"] = "1"
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    assert spec["CellType"] == "mamba"
    assert spec["Direction"] == "forward"
    # Mamba_* is UNPREFIXED by design (one geometry per config, spec S6/S3.3).
    assert spec["Mamba"] == {"d_state": 8, "d_conv": 4, "expand": 1, "dt_rank": 0}


# ------------------------------------------------------------------------------------- #
# Baseline arm knobs (spec S7.2)
# ------------------------------------------------------------------------------------- #

_SAD_FLAT: dict[str, str] = {
    "Algo_choice": "3",
    "BLSTM_LSTMNeuronNb": "23,24,24",
    "BLSTM_LSTMSubSampling": "4,1",
    "BLSTM_OutputNeuronNb": "48,12,1",
    "BLSTM_OutputSubSampling": "1,1",
    "fileslisting": "x.flst",
}


def test_default_knobs_add_nothing() -> None:
    """DEFAULT invocation is byte-identical to today: the overlay is EMPTY, so the assembled
    config text with and without the knobs is the SAME string."""
    assert baseline.cell_overlay(_SAD_FLAT, "lstm", "bidirectional") == {}
    plain = baseline.assemble_flat_config(_SAD_FLAT, fileslisting="t.flst", mapping="m.csv", sad_seed="s.bin", lanes=1)
    knobbed = baseline.assemble_flat_config(
        _SAD_FLAT,
        fileslisting="t.flst",
        mapping="m.csv",
        sad_seed="s.bin",
        lanes=1,
        extra=baseline.cell_overlay(_SAD_FLAT, "lstm", "bidirectional"),
    )
    assert baseline._config_text(plain) == baseline._config_text(knobbed)


def test_cell_type_knob_overlays_the_s6_key_only() -> None:
    for cell in ("slstm", "mamba"):
        assert baseline.cell_overlay(_SAD_FLAT, cell, "bidirectional") == {"BLSTM_Cell_Type": cell}


def test_direction_knob_also_resizes_the_output_mlp_input() -> None:
    """`BlstmConfig::from_legacy` REQUIRES `OutputNeuronNb[0] == hidden_multiplier * hidden`
    -- 24, not 48, under `forward`. A knob that only wrote the direction key would produce a
    config the engine refuses to build, so the overlay carries the derived resize."""
    overlay = baseline.cell_overlay(_SAD_FLAT, "lstm", "forward")
    assert overlay == {"BLSTM_Direction": "forward", "BLSTM_OutputNeuronNb": "24,12,1", "BLSTM_window": "0"}


def test_direction_knob_forces_the_plain_window_regime() -> None:
    """Phase 9 Task 6. `lre_sad.toml` carries `frame_window 3.25`, which resolves
    `window_size > 0` and dispatches the WINDOWED drivers -- and a window boundary RESETS a
    causal cell's recurrent state, so a windowed causal run is defined-but-pointless (spec
    S1.2), the streaming session refuses it (S5.3), and the f32 fast twin refuses it too. The
    overlay therefore forces `BLSTM_window 0` on `forward`, and MUST NOT touch the window
    keys on `bidirectional` (whose windowed overlap is the phase-6/7 regime)."""
    for cell in ("lstm", "slstm", "mamba"):
        assert baseline.cell_overlay(_SAD_FLAT, cell, "forward")["BLSTM_window"] == "0"
        bidir = baseline.cell_overlay(_SAD_FLAT, cell, "bidirectional")
        assert not any(k.startswith("BLSTM_window") or k == "BLSTM_shift" for k in bidir), bidir


def test_knobs_compose() -> None:
    overlay = baseline.cell_overlay(_SAD_FLAT, "mamba", "forward")
    assert overlay == {
        "BLSTM_Cell_Type": "mamba",
        "BLSTM_Direction": "forward",
        "BLSTM_OutputNeuronNb": "24,12,1",
        "BLSTM_window": "0",
    }


def test_overlaid_config_seeds_through_the_matching_builder() -> None:
    """End to end through the arm surface: the overlay lands on the flat config, the engine
    reads it back through the layout, and `init_weights` emits an sLSTM-sized forward pack."""
    cfg = {**_SAD_FLAT, **baseline.cell_overlay(_SAD_FLAT, "slstm", "forward")}
    cfg |= {
        "BLSTM_NNetInputSize": "23",
        "BLSTM_InputNormalizationType": "1",
        "BLSTM_TwoSweeps": "false",
        "BLSTM_CostLawSpeech": "log",
        "BLSTM_CostLawNoSpeech": "log",
        **{f"BLSTM_{d}_Is{p}Active": "true" for d in ("Forward", "Backward") for p in ("CellsPeepholes", "GatesPeepholes", "GatesRecurrentPeepholes")},
    }
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(0))[0]
    layout = layout_of(cfg)
    assert {b["stack"] for b in layout} == {"forward", "output", "tail"}
    assert flat.shape == (pack_length(cfg),) == (4 * 24 * (24 + 92 + 1) + 4 * 24 * (24 + 24 + 1) + (24 * 12 + 12) + (12 + 1) + 46,)


def test_parser_exposes_the_knobs_with_legacy_defaults() -> None:
    args = baseline.build_parser().parse_args(["sad", "--corpus-root", "/tmp/c", "--out-dir", "/tmp/o"])
    assert args.cell == "lstm"
    assert args.direction == "bidirectional"
    args = baseline.build_parser().parse_args(["sad", "--corpus-root", "/tmp/c", "--out-dir", "/tmp/o", "--cell", "mamba", "--direction", "forward"])
    assert args.cell == "mamba"
    assert args.direction == "forward"
