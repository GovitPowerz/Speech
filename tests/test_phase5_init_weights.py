"""Phase 5 Task 3 (re-based on the seam, issue #27): the legacy LSTM's seeded init.

Coverage for `init_weights.init_weights` on the legacy cell: pack lengths against the
committed Phase 4d tuple-A/B configs and the #61 output-sub-sampling shapes, the forget-bias
rule, the identity normalize tail, determinism, the xavier-vs-he scale and the peephole band.
Every block is LOCATED through `speech_rs.weight_layout` -- the engine's own block list --
never a hardcoded flat offset or a Python re-derivation.

Byte identity with the pre-#27 builder is pinned elsewhere, by
`tests/test_seed_pack_regeneration.py` (the committed `lstm_forward` and Twin LSTM packs).
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import speech_rs
from numpy.typing import NDArray
from speech.config_bridge import parse_legacy_config
from speech.init_weights import init_weights

REF = Path("tests/reference_data/phase4d")
Block = dict[str, Any]


def _cfg(name: str) -> dict[str, str]:
    return parse_legacy_config((REF / name).read_text())


TUPLE_A = _cfg("tupleA_1_worker_1.config")
TUPLE_B = _cfg("tupleB_1_worker_1.config")
CONFIGS = {"A": (TUPLE_A, 33671), "B": (TUPLE_B, 42911)}


def _layout(cfg: dict[str, str], prefix: str = "BLSTM") -> list[Block]:
    return cast(list[Block], speech_rs.weight_layout(cfg, prefix))


def _slice(pack: NDArray[np.float64], b: Block) -> NDArray[np.float64]:
    return pack[b["offset"] : b["offset"] + b["len"]]


def _synthetic(lstm: str, lsub: str, outn: str, osub: str, **extra: str) -> dict[str, str]:
    cfg = {
        "BLSTM_LSTMNeuronNb": lstm,
        "BLSTM_LSTMSubSampling": lsub,
        "BLSTM_OutputNeuronNb": outn,
        "BLSTM_OutputSubSampling": osub,
        "BLSTM_InputNormalizationType": "1",
        "BLSTM_TwoSweeps": "false",
    }
    cfg.update(extra)
    return cfg


@pytest.mark.parametrize("tag", ["A", "B"])
def test_pack_length_is_the_committed_shape(tag: str) -> None:
    cfg, expected_count = CONFIGS[tag]
    packs = init_weights(cfg, "BLSTM", np.random.default_rng(0))
    assert len(packs) == 1  # one net -> one flat pack.
    assert packs[0].dtype == np.float64
    assert packs[0].shape == (expected_count,)


# The two `twin_mode7.config` nets at `OutputSubSampling 2`, with the pack lengths
# `src/rust/tests/output_subsampling_pack.rs` pins against the exact net's `nb_of_weights` (#61).
OSUB2: dict[str, tuple[dict[str, str], int]] = {
    "sad": (_synthetic("11,12", "4", "24,1", "2"), 5831),
    "lid": (_synthetic("36,24", "1", "48,1", "2"), 12457),
}


@pytest.mark.parametrize("tag", ["sad", "lid"])
def test_output_subsampling_widens_the_output_layer(tag: str) -> None:
    cfg, expected_count = OSUB2[tag]
    pack = init_weights(cfg, "BLSTM", np.random.default_rng(0))[0]
    assert pack.shape == (expected_count,)
    out_w = next(b for b in _layout(cfg) if b["stack"] == "output" and b["name"] == "W")
    outn = [int(x) for x in cfg["BLSTM_OutputNeuronNb"].split(",")]
    assert out_w["shape"] == [outn[1], outn[0] * 2]
    assert (out_w["fan_in"], out_w["fan_out"]) == (outn[0] * 2, outn[1])


@pytest.mark.parametrize("cell", ["lstm", "slstm", "mamba", "cfc", "transformer"])
def test_output_subsampling_widens_every_cell_pack_by_the_extra_fan_in(cell: str) -> None:
    """`OutputSubSampling 2` on output layer 0 adds `OutputNeuronNb[1] * OutputNeuronNb[0]`
    weights, the second sub-sampled frame's block -- read off the engine's layout for every
    cell, so no Python builder can frame the output rows short (#61)."""
    cfg = _synthetic("5,4", "1", "8,3,1", "1,1", BLSTM_Cell_Type=cell)
    n1 = init_weights(cfg, "BLSTM", np.random.default_rng(0))[0].shape[0]
    n2 = init_weights({**cfg, "BLSTM_OutputSubSampling": "2,1"}, "BLSTM", np.random.default_rng(0))[0].shape[0]
    assert n2 - n1 == 3 * 8


@pytest.mark.parametrize("tag", ["A", "B"])
def test_forget_bias_one_positions(tag: str) -> None:
    cfg, _ = CONFIGS[tag]
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(1), forget_bias_one=True)[0]
    seen = {"b.f": 0, "other": 0, "out": 0}
    for b in _layout(cfg):
        if b["kind"] != "bias":
            continue
        if b["name"] == "b.f":
            assert np.all(_slice(flat, b) == 1.0), (b["stack"], b["layer"])
            seen["b.f"] += 1
        elif b["stack"] == "output":
            assert np.all(_slice(flat, b) == 0.0), b["layer"]
            seen["out"] += 1
        else:
            assert np.all(_slice(flat, b) == 0.0), (b["stack"], b["layer"], b["name"])
            seen["other"] += 1
    assert seen == {"b.f": 4, "other": 12, "out": 2}  # 2 stacks x 2 layers x (1 forget + 3 other), 2 output layers


@pytest.mark.parametrize("tag", ["A", "B"])
def test_forget_bias_one_false_gives_every_bias_zero(tag: str) -> None:
    cfg, _ = CONFIGS[tag]
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(1), forget_bias_one=False)[0]
    for b in _layout(cfg):
        if b["kind"] == "bias":
            assert np.all(_slice(flat, b) == 0.0), (b["stack"], b["layer"], b["name"])


@pytest.mark.parametrize("tag", ["A", "B"])
def test_normalize_tail_is_identity(tag: str) -> None:
    cfg, _ = CONFIGS[tag]
    flat = init_weights(cfg, "BLSTM", np.random.default_rng(2))[0]
    tail = [b for b in _layout(cfg) if b["stack"] == "tail"]
    lstm0 = int(cfg["BLSTM_LSTMNeuronNb"].split(",")[0])
    assert [(b["name"], b["len"]) for b in tail] == [("mean", lstm0), ("std", lstm0)]
    assert np.all(_slice(flat, tail[0]) == 0.0)
    assert np.all(_slice(flat, tail[1]) == 1.0)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
def test_determinism_same_seed_is_bit_identical(scheme: str) -> None:
    packs_a = init_weights(TUPLE_A, "BLSTM", np.random.default_rng(42), scheme=scheme)  # type: ignore[arg-type]
    packs_b = init_weights(TUPLE_A, "BLSTM", np.random.default_rng(42), scheme=scheme)  # type: ignore[arg-type]
    assert np.array_equal(packs_a[0], packs_b[0])


def test_different_seeds_produce_different_packs() -> None:
    packs_a = init_weights(TUPLE_A, "BLSTM", np.random.default_rng(1))
    packs_b = init_weights(TUPLE_A, "BLSTM", np.random.default_rng(2))
    assert not np.array_equal(packs_a[0], packs_b[0])


def test_invalid_scheme_raises() -> None:
    with pytest.raises(ValueError, match="scheme"):
        init_weights(TUPLE_A, "BLSTM", np.random.default_rng(0), scheme="bogus")  # type: ignore[arg-type]


def test_a_config_the_engine_refuses_raises_before_any_draw() -> None:
    """The layout comes from the engine, so its refusals are this function's refusals: a
    `ValueError` carrying the engine's message, never a default-shaped pack."""
    cfg = {**TUPLE_A, "BLSTM_OutputNeuronNb": "47,12,1"}
    with pytest.raises(ValueError, match="twice the number of LSTM cells"):
        init_weights(cfg, "BLSTM", np.random.default_rng(0))


def test_xavier_vs_he_variance_ratio_sanity_on_a_large_block() -> None:
    # A synthetic, deliberately large net so the measured block has a big sample size
    # (low variance-estimate noise) -- loose bounds below, not a flaky tight assert.
    cfg = _synthetic("64,128,128", "1,1", "256,10", "1")
    flat_x = init_weights(cfg, "BLSTM", np.random.default_rng(7), scheme="xavier")[0]
    flat_h = init_weights(cfg, "BLSTM", np.random.default_rng(7), scheme="he")[0]
    layout = _layout(cfg)
    w_i = next(b for b in layout if (b["stack"], b["layer"], b["name"]) == ("forward", 0, "w.i"))
    r_i = next(b for b in layout if (b["stack"], b["layer"], b["name"]) == ("forward", 0, "r.i"))
    assert (w_i["fan_in"], w_i["fan_out"]) == (64 + 128, 128) == (r_i["fan_in"], r_i["fan_out"])  # ONE combined map
    block_x = np.concatenate([_slice(flat_x, w_i), _slice(flat_x, r_i)])
    block_h = np.concatenate([_slice(flat_h, w_i), _slice(flat_h, r_i)])
    assert block_x.size >= 10_000  # this really is "a large block".

    combined_fan_in, out = 64 + 128, 128
    var_x, var_h = float(np.var(block_x)), float(np.var(block_h))
    expect_var_x = 2.0 / (combined_fan_in + out)  # Glorot uniform: (2*limit)**2/12 == 2/(fan_in+fan_out).
    expect_var_h = 2.0 / combined_fan_in  # He normal: std**2 == 2/fan_in.

    assert var_h > var_x  # guaranteed by construction whenever fan_out > 0 -- not just "close".
    assert 0.5 * expect_var_x < var_x < 2.0 * expect_var_x
    assert 0.5 * expect_var_h < var_h < 2.0 * expect_var_h


def test_peepholes_are_small_and_nonconstant() -> None:
    """The 12-column bundle gets a small nonzero draw (documented as recurrent-diagonal init,
    not Xavier/He-scaled -- see the `init_weights` module docstring), independent of `scheme`,
    in every stack and layer."""
    for scheme in ("xavier", "he"):
        flat = init_weights(TUPLE_A, "BLSTM", np.random.default_rng(3), scheme=scheme)[0]  # type: ignore[arg-type]
        peeps = [b for b in _layout(TUPLE_A) if b["kind"] == "peephole"]
        assert len(peeps) == 4
        for b in peeps:
            peep = _slice(flat, b)
            assert np.all(np.abs(peep) <= 0.1)
            assert np.unique(peep).size > 1  # not a degenerate constant fill.
            assert b["columns"][:3] == ["i.cell_prev", "f.cell_prev", "o.cell"]
