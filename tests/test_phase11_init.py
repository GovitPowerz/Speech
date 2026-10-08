"""Phase 11 Task 3: the transformer geometry reader, seeded init builder, and the SIZED d_ff.

THE LAYOUT CONTRACT IS THE RUST WALK (`nn/cells/transformer.rs::for_each_slot`, spec S1.2).
Every offset asserted here is recomputed from the spec block order INDEPENDENTLY of
`init_weights.py`'s own construction -- a THIRD derivation, alongside the walk itself and the
test-local offsets table in `transformer.rs`'s unit tests. A Python builder that agrees with
itself but not with the Rust `set_weights` walk still fails here.

    W_a (H x in) | b_a (H) | g_1 (H)                       adapter + RMSNorm_1 gain
    W_qkv (3H x H) | b_qkv (3H)                            the COMBINED [q | k | v] projection
    W_o (H x H) | b_o (H) | g_2 (H)                        attention output + RMSNorm_2 gain
    W_1 (d_ff x H) | b_1 (d_ff) | W_2 (H x d_ff) | b_2 (H) the FFN

    nb_of_weights = H*(in+1) + H + 3H*(H+1) + H*(H+1) + H + d_ff*(H+1) + H*(d_ff+1)

Every matrix is ROW-MAJOR at its MATH shape (output unit outer, source index inner); the Rust
stores most of them transposed and the walk absorbs that, so the flat order is the math order.

The bit-identical `speech_rs.Engine` round trip (the seam-level proof that these packs are what
`BlstmNetwork::set_weights` consumes, and the only CROSS-LANGUAGE length pin there is) lives in
`tests/pyo3/test_phase11_init_pyo3.py` -- every `importorskip("speech_rs")` in this repo lives
under `tests/pyo3/`, which is the directory the dedicated `python-pyo3` CI job runs.


THE SIZING PROCEDURE (spec S3) -- how `TRANSFORMER_DEFAULT_D_FF = 64` was confirmed
===================================================================================

`window` and `heads` are PARAMETER-FREE (ALiBi's slopes are constants; `heads` only reshapes
the same `W_qkv`), so `d_ff` is the cell's ONE sized knob. The rule: prefer an integer inside
BOTH lineage bands (+-15% of the SAME lineage's LSTM pack, the phase-9 S8.2 band); if the two
lineages conflict, v2 -- the defect-free lineage -- wins and the conflict is RECORDED.

Per-layer count at hidden `H = 24` (both lineages), from the closed form above:

    N(in) = 24*in + 2496 + 49*d_ff

Both packs share the output MLP and the normalize tail, so only the recurrent stacks move.

v1 lineage (`lre_sad.toml`: LSTMNeuronNb 23,24,24 / SubSampling 4,1 / MLP 48,12,1 / tail 46)
  layer 0 fan-in 23*4 = 92, layer 1 fan-in 24, bidirectional -> 2 stacks.
    LSTM  = 2*(11520 + 4992) + 601 + 46                = 33671   (the committed tuple-A length)
    TRANS = 2*((4704 + 49 d) + (3072 + 49 d)) + 601 + 46 = 16199 + 196 d

v2 lineage (`lre_sad_v2.toml`: LSTMNeuronNb 11,24,24, everything else identical -> tail 22)
  layer 0 fan-in 11*4 = 44 (the honest feature width; v1's 92 is the 2015 dead-column defect).
    LSTM  = 2*(6912 + 4992) + 601 + 22                 = 24431
    TRANS = 2*((3552 + 49 d) + (3072 + 49 d)) + 601 + 22 = 13871 + 196 d

    d_ff   v2 pack     v2 dev   v1 pack     v1 dev   verdict
      36     20927    -14.34%     23255    -30.93%   v2 in / v1 OUT   (v2 band floor)
      53     24259     -0.70%     26587    -21.04%   v2 in / v1 OUT
      54     24455     +0.10%     26783    -20.46%   v2 in / v1 OUT   <- the v2 ARGMIN
      63     26219     +7.32%     28547    -15.22%   v2 in / v1 OUT   (v1 misses by 0.22 pt)
      64     26415     +8.12%     28743    -14.64%   BOTH             <- THE DEFAULT
      72     27983    +14.54%     30311     -9.98%   BOTH             (v2 band ceiling)
      73     28179    +15.34%     30507     -9.40%   v2 OUT / v1 in
      89     31315    +28.18%     33643     -0.08%   v2 OUT / v1 in   <- the v1 argmin
     114     36215    +48.23%     38543    +14.47%   v2 OUT / v1 in   (v1 band ceiling)

Bands: v2 `d_ff in [36, 72]`, v1 `[64, 114]`, intersection `[64, 72]` -- NON-EMPTY, so the
primary rule decides and no tiebreak is needed. Within the intersection v2's deviation grows
monotonically (its argmin 54 lies below the floor), so 64 is simultaneously the smallest
integer inside both bands AND the v2-best point of the feasible set: the tiebreak lineage would
pick it too.

THE CONFLICT, recorded because it is exactly what the spec asked to be told about: unlike CfC
-- where `B = 45` was the v2 argmin AND the smallest v1-tolerable integer at once -- the two
CfC criteria DIVERGE here. The v2 argmin is 54 and it sits outside v1's band at -20.46%.

AND IT IS AN ARTEFACT OF COUNTING DEAD WEIGHT. Subtract v1's structurally-dead layer-0 `W_a`
columns (`2*24*48 = 2304`) and each lineage's normalize tail, and both lineages have the SAME
live closed form `13849 + 196*d_ff` against the SAME live LSTM target `24409`: at `d_ff = 64`
both are IDENTICALLY 26393 live weights, +8.13%. The phase-10 live-count identity extends to
the transformer intact, and under a live-count band there is no conflict at all -- one band
[36, 72], one argmin 54. PACK LENGTH is the repo's stated convention, so it is what the default
is sized against; the two conventions happen to agree at 64.

The underlying finding, stated rather than hidden: windowed attention at width 24 is
parameter-CHEAP next to a peephole LSTM (a whole cell layer costs `2496 + 49 d_ff` plus the
fan-in term, against the LSTM's `4*H*(fin + H + 4)`). At the textbook `4*H = 96` the v2 pack is
32687 (+33.79%, outside v2's band) while v1 is +3.99% and inside its own -- the textbook default
fails exactly one lineage. 64 therefore lands well above the naive "small cell -> small FFN"
instinct but comfortably BELOW the transformer-literature default.

Forward-only runs move both packs the same way (one stack, and `cell_overlay` resizes the MLP
input from 2H to H), so the verdict does not flip: v2 fwd LSTM 12239 vs TRANS 13231 (+8.11%),
v1 fwd LSTM 16871 vs TRANS 14407 (-14.60%).
"""

import math
import re
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from speech import config_bridge
from speech.drivers import baseline
from speech.init_weights import init_transformer_flat, init_weights, transformer_geometry

REF = Path("tests/reference_data/phase4d")

# The finalized geometry (spec S2/S3). Mirrored in THREE places -- `blstm.rs`'s
# `TRANSFORMER_DEFAULT_*`, `config_bridge`'s, and these literals -- and
# `test_the_rust_and_python_defaults_agree` pins the first two together.
SIZED_D_FF = 64
DEFAULT_WINDOW = 64
DEFAULT_HEADS = 4

# (out, fin, d_ff): the degenerate 1x1x1, a square-ish shape, a wide fan-in, and a d_ff SMALLER
# than H (so `W_1` and `W_2` are non-square in opposite directions, which is what makes a
# transposed-shape bug visible rather than absorbed).
SHAPES: tuple[tuple[int, int, int], ...] = ((1, 1, 1), (4, 5, 4), (2, 3, 7), (3, 7, 2), (6, 11, 5))


# ------------------------------------------------------------------------------------- #
# The Rust walk, transcribed independently (NOT imported from the builder module)
# ------------------------------------------------------------------------------------- #


def transformer_blocks(out: int, fin: int, d_ff: int) -> list[tuple[str, int]]:
    """`transformer.rs::for_each_slot`, transcribed: (name, start) per S1.2 block plus a
    trailing ("END", total)."""
    lens: list[tuple[str, int]] = [
        ("W_a", out * fin),
        ("b_a", out),
        ("g_1", out),
        ("W_qkv", 3 * out * out),
        ("b_qkv", 3 * out),
        ("W_o", out * out),
        ("b_o", out),
        ("g_2", out),
        ("W_1", d_ff * out),
        ("b_1", d_ff),
        ("W_2", out * d_ff),
        ("b_2", out),
    ]
    blocks: list[tuple[str, int]] = []
    pos = 0
    for name, length in lens:
        blocks.append((name, pos))
        pos += length
    blocks.append(("END", pos))
    return blocks


def transformer_block(out: int, fin: int, d_ff: int, name: str) -> tuple[int, int]:
    blocks = transformer_blocks(out, fin, d_ff)
    k = [n for n, _ in blocks].index(name)
    return blocks[k][1], blocks[k + 1][1]


def transformer_nb(out: int, fin: int, d_ff: int) -> int:
    """The S1.2 CLOSED formula, written out separately from the block walk above so the two
    derivations cross-check each other (`test_layer_length_is_the_rust_formula`)."""
    return out * (fin + 1) + out + 3 * out * (out + 1) + out * (out + 1) + out + d_ff * (out + 1) + out * (d_ff + 1)


def net_length(spec: dict[str, object], d_ff: int) -> int:
    """Full-net pack length: cell stacks (twice when bidirectional) + output MLP + tail."""
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    outn = cast(list[int], spec["OutputNeuronNb"])
    stacks = 1 if spec.get("Direction", "bidirectional") == "forward" else 2
    total = sum(stacks * transformer_nb(lstm[i + 1], lstm[i] * lsub[i], d_ff) for i in range(len(lstm) - 1))
    for i in range(len(outn) - 1):
        total += outn[i + 1] * outn[i] + outn[i + 1]
    return total + 2 * lstm[0]


def cell_spec(direction: str = "bidirectional", d_ff: int = 5) -> dict[str, object]:
    """A small synthetic net spec: 2 recurrent layers, a 2-layer output MLP. `hidden = 4` is
    divisible by the default 4 heads, so this spec is also engine-constructible."""
    hidden = 4
    return {
        "LSTMNeuronNb": [5, 4, hidden],
        "LSTMSubSampling": [2, 1],
        "OutputNeuronNb": [hidden if direction == "forward" else 2 * hidden, 3, 1],
        "OutputSubSampling": [1, 1],
        "CellType": "transformer",
        "Direction": direction,
        "Transformer": {"window": 8, "heads": 2, "d_ff": d_ff},
    }


# ------------------------------------------------------------------------------------- #
# Layer-level: lengths, block values
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_layer_length_is_the_rust_formula(out: int, fin: int, d_ff: int) -> None:
    assert transformer_blocks(out, fin, d_ff)[-1][1] == transformer_nb(out, fin, d_ff)  # walk == closed form
    flat = init_transformer_flat(np.random.default_rng(0), out, fin, d_ff)
    assert flat.dtype == np.float64
    assert flat.shape == (transformer_nb(out, fin, d_ff),)


@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_every_bias_block_is_exactly_zero(out: int, fin: int, d_ff: int) -> None:
    """Spec S8: biases 0 EVERYWHERE. The transformer has no forget-gate analogue to open (no
    memory valve at all), so unlike the LSTM/sLSTM builders there is no `forget_bias_one`
    exception -- and `b_k`, the STRUCTURALLY DEAD middle third of `b_qkv`, is covered by this
    same sweep rather than by a special case, because 0 is what every bias gets anyway."""
    flat = init_transformer_flat(np.random.default_rng(1), out, fin, d_ff)
    for name in ("b_a", "b_qkv", "b_o", "b_1", "b_2"):
        lo, hi = transformer_block(out, fin, d_ff, name)
        assert hi > lo, name
        assert np.all(flat[lo:hi] == 0.0), name


@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_the_two_rmsnorm_gains_are_exactly_one(out: int, fin: int, d_ff: int) -> None:
    """A gain is a SCALE, so its identity seed is 1.0, not 0.0 (the Mamba `g = 1` convention).
    Seeding these 0 would zero both sub-block inputs and stall the net at init -- the one place
    in this pack where the wrong structural constant is silently catastrophic rather than
    merely suboptimal."""
    flat = init_transformer_flat(np.random.default_rng(2), out, fin, d_ff)
    for name in ("g_1", "g_2"):
        lo, hi = transformer_block(out, fin, d_ff, name)
        assert hi - lo == out, name
        assert np.all(flat[lo:hi] == 1.0), name


@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_every_weight_block_is_drawn_and_finite(out: int, fin: int, d_ff: int) -> None:
    """Non-vacuity for the zero-bias / one-gain pins above: a zero-filled projection would make
    the first pass trivially and hide a builder that never draws at all."""
    flat = init_transformer_flat(np.random.default_rng(3), out, fin, d_ff)
    for name in ("W_a", "W_qkv", "W_o", "W_1", "W_2"):
        lo, hi = transformer_block(out, fin, d_ff, name)
        block = flat[lo:hi]
        assert np.all(np.isfinite(block)), name
        assert np.all(block != 0.0), name


def test_xavier_and_he_scale_the_same_block_differently() -> None:
    """The scheme reaches the blocks: He-normal (`sqrt(2/fan_in)`) and Xavier-uniform
    (`sqrt(6/(fan_in+fan_out))`, std = limit/sqrt(3)) have different standard deviations at the
    same fans, so a builder ignoring `scheme` would show one ratio of 1.0."""
    out, fin, d_ff = 16, 24, 32
    xav = init_transformer_flat(np.random.default_rng(4), out, fin, d_ff, "xavier")
    he = init_transformer_flat(np.random.default_rng(4), out, fin, d_ff, "he")
    lo, hi = transformer_block(out, fin, d_ff, "W_qkv")
    fan_in, fan_out = out, 3 * out
    want = math.sqrt(2.0 / fan_in) / (math.sqrt(6.0 / (fan_in + fan_out)) / math.sqrt(3.0))
    assert np.std(he[lo:hi]) / np.std(xav[lo:hi]) == pytest.approx(want, rel=0.25)


def test_forget_bias_one_is_not_a_parameter() -> None:
    """Deferred F3 (phase-11 T9): the first landed signature accepted-and-ignored
    `forget_bias_one` for call-site uniformity with the sLSTM/LSTM builders `_init_cell_pack`
    dispatches beside this one -- but `init_cfc_flat` (also forget-gate-free) never carried the
    flag at all, so the accept-and-ignore choice was the inconsistent one. The signature now
    has no such parameter, matching the `init_cfc_flat` precedent exactly: a caller passing it
    gets a loud `TypeError`, not a silently discarded flag."""
    with pytest.raises(TypeError):
        init_transformer_flat(np.random.default_rng(5), 4, 5, 6, "xavier", True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        init_transformer_flat(np.random.default_rng(5), 4, 5, 6, forget_bias_one=True)  # type: ignore[call-arg]


# ------------------------------------------------------------------------------------- #
# THE WHOLE-PACK RECONSTRUCTION PIN (mandatory, the phase-9/10 shear lesson)
# ------------------------------------------------------------------------------------- #
#
# Fingerprint, length and band pins are PERMUTATION-BLIND, and a flat `set_weights`/`weights`
# round trip through the Engine cannot see a permutation either (both directions drive the same
# walk). This pack admits two such permutations:
#
#   * the q / k / v ROW FAMILIES inside `W_qkv` -- same shape, same fans, one draw;
#   * `W_1` (d_ff x H) and `W_2` (H x d_ff) -- the SAME element count, and under Xavier the
#     SAME limit, because `sqrt(6/(fan_in+fan_out))` is SYMMETRIC in the two fans. Swapping
#     them is invisible to every distributional check in this file.
#
# The test below reconstructs the ENTIRE pack from a fresh `default_rng(seed)`, drawing block by
# block in spec order at the documented per-block shape/fans, and asserts BIT-identity: a
# permutation moves a draw to a different position in the stream, so position, scale, shape and
# draw order are all pinned at once. The draw helpers are written out here rather than imported
# from `init_weights`, so this is an independent transcription, not a tautology.
#
# WHAT IT CANNOT PIN, stated rather than implied: the q/k/v BIAS thirds. `b_qkv` is all zeros,
# so no test can tell `b_q` from `b_k` from `b_v` by value -- which is also why `b_k`'s
# structural deadness is pinned Rust-side, on the GRADIENT, not here on the seed.


def _xavier(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(rng.uniform(-limit, limit, size=shape), dtype=np.float64).reshape(-1)


def _he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    return np.asarray(rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape), dtype=np.float64).reshape(-1)


def _blk(rng: np.random.Generator, scheme: str, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    return _xavier(rng, shape, fan_in, fan_out) if scheme == "xavier" else _he(rng, shape, fan_in)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
@pytest.mark.parametrize(("out", "fin", "d_ff"), SHAPES)
def test_pack_is_reproducible_block_by_block(out: int, fin: int, d_ff: int, scheme: str) -> None:
    """S1.2 order at the S8 fans, each read straight off the block's own shape: `W_a` is
    `in -> H`; `W_qkv` is ONE combined `H -> 3H` matrix (so ONE fan pair, the sLSTM/CfC
    combined-block precedent -- NOT three `H -> H` draws); `W_o` is `H -> H`; the FFN is
    `H -> d_ff` then `d_ff -> H`. The two gains are 1.0 and every bias is 0.0."""
    rng = np.random.default_rng(37)
    want: list[np.ndarray] = [
        _blk(rng, scheme, (out, fin), fin, out),  # W_a
        np.zeros(out),  # b_a
        np.ones(out),  # g_1
        _blk(rng, scheme, (3 * out, out), out, 3 * out),  # W_qkv
        np.zeros(3 * out),  # b_qkv
        _blk(rng, scheme, (out, out), out, out),  # W_o
        np.zeros(out),  # b_o
        np.ones(out),  # g_2
        _blk(rng, scheme, (d_ff, out), out, d_ff),  # W_1
        np.zeros(d_ff),  # b_1
        _blk(rng, scheme, (out, d_ff), d_ff, out),  # W_2
        np.zeros(out),  # b_2
    ]
    got = init_transformer_flat(np.random.default_rng(37), out, fin, d_ff, scheme)  # type: ignore[arg-type]
    assert np.array_equal(got, np.concatenate(want))


def test_reconstruction_rejects_a_qk_family_swap() -> None:
    """SHEAR 1, non-vacuity for the pin above: the q and k ROW FAMILIES of `W_qkv` are the same
    shape at the same fans, so swapping them is invisible to every length, band, shape and
    round-trip assertion. It is also the permutation with the most semantic bite -- the S1.2
    `[q | k | v]` order is what makes "the k rows" an addressable range at all (the `b_k`
    deadness proof and the FD tier's dead-block pins both index it)."""
    out, fin, d_ff = 3, 7, 2
    rng = np.random.default_rng(37)
    parts: list[np.ndarray] = [_xavier(rng, (out, fin), fin, out), np.zeros(out), np.ones(out)]
    qkv = _xavier(rng, (3 * out, out), out, 3 * out).reshape(3 * out, out)
    q, k, v = qkv[:out].copy(), qkv[out : 2 * out].copy(), qkv[2 * out :].copy()
    parts.append(np.concatenate([k, q, v]).reshape(-1))  # THE SHEAR: k emitted where q belongs.
    parts += [np.zeros(3 * out), _xavier(rng, (out, out), out, out), np.zeros(out), np.ones(out)]
    parts += [_xavier(rng, (d_ff, out), out, d_ff), np.zeros(d_ff), _xavier(rng, (out, d_ff), d_ff, out), np.zeros(out)]

    got = init_transformer_flat(np.random.default_rng(37), out, fin, d_ff)
    bad = np.concatenate(parts)
    assert got.shape == bad.shape  # same LENGTH -- which is why a length pin cannot see it.
    assert not np.array_equal(got, bad)


def test_reconstruction_rejects_an_ffn_matrix_swap() -> None:
    """SHEAR 2, the FFN family (the brief's "deeper-FFN variant" adapted to a cell whose FFN
    has no depth knob -- one hidden layer, so the available permutation is between its two
    matrices). `W_1` is `d_ff x H` and `W_2` is `H x d_ff`: the SAME element count, so the swap
    keeps the pack length and no shape or band assertion can see it.

    THE SWAP THAT MATTERS IS PLACEMENT, not draw order, and the distinction is measured rather
    than assumed (the probe for this test found it): Xavier's `sqrt(6/(fan_in+fan_out))` is
    SYMMETRIC in the fans, so `W_1` and `W_2` draw from an IDENTICAL distribution and consume
    the same count -- reversing the two `_draw` CALLS is then a bit-level no-op, while emitting
    the drawn values crosswise is not. That is exactly the CfC head-swap shape: identically
    distributed blocks whose only distinguishing fact is which stream position lands where."""
    out, fin, d_ff = 4, 5, 4  # d_ff == out, so the two blocks are even the same SHAPE.
    rng = np.random.default_rng(37)
    parts: list[np.ndarray] = [_xavier(rng, (out, fin), fin, out), np.zeros(out), np.ones(out)]
    parts += [_xavier(rng, (3 * out, out), out, 3 * out), np.zeros(3 * out)]
    parts += [_xavier(rng, (out, out), out, out), np.zeros(out), np.ones(out)]
    w1 = _xavier(rng, (d_ff, out), out, d_ff)  # drawn in SPEC order...
    w2 = _xavier(rng, (out, d_ff), d_ff, out)
    parts += [w2, np.zeros(d_ff), w1, np.zeros(out)]  # ...and emitted CROSSWISE: the shear.

    got = init_transformer_flat(np.random.default_rng(37), out, fin, d_ff)
    bad = np.concatenate(parts)
    assert got.shape == bad.shape  # same LENGTH -- which is why a length pin cannot see it.
    assert not np.array_equal(got, bad)


@pytest.mark.parametrize(("scheme", "same_bytes"), [("xavier", True), ("he", False)])
def test_the_ffn_draw_order_is_only_visible_under_he(scheme: str, same_bytes: bool) -> None:
    """The measured LIMIT of the pin above, recorded rather than left as a surprise (this
    file's own mutation probe found it). Reversing the two FFN `_draw` CALLS -- the shape a
    mis-ordered pair of source lines actually takes -- is a BIT-LEVEL NO-OP under Xavier, whose
    symmetric `sqrt(6/(fan_in+fan_out))` gives both blocks the same limit over the same element
    count, and IS visible under He, whose `sqrt(2/fan_in)` distinguishes `H` from `d_ff`.

    So the two schemes miss in OPPOSITE directions -- He cannot see a fan_out error at all,
    Xavier cannot see this draw-order variant -- which is what makes parametrizing the
    reconstruction pin over BOTH load-bearing rather than decorative."""
    out, fin, d_ff = 3, 7, 5  # out != d_ff, so He's fan_in genuinely differs between the two.
    rng = np.random.default_rng(41)
    parts: list[np.ndarray] = [_blk(rng, scheme, (out, fin), fin, out), np.zeros(out), np.ones(out)]
    parts += [_blk(rng, scheme, (3 * out, out), out, 3 * out), np.zeros(3 * out)]
    parts += [_blk(rng, scheme, (out, out), out, out), np.zeros(out), np.ones(out)]
    # The two FFN calls REVERSED, each result left in the slot its own line occupies.
    parts += [_blk(rng, scheme, (out, d_ff), d_ff, out), np.zeros(d_ff), _blk(rng, scheme, (d_ff, out), out, d_ff), np.zeros(out)]

    got = init_transformer_flat(np.random.default_rng(41), out, fin, d_ff, scheme)  # type: ignore[arg-type]
    assert np.array_equal(got, np.concatenate(parts)) is same_bytes


# ------------------------------------------------------------------------------------- #
# Net-level assembly + determinism
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("direction", ["bidirectional", "forward"])
@pytest.mark.parametrize("d_ff", [1, 5])
def test_net_pack_length(direction: str, d_ff: int) -> None:
    spec = cell_spec(direction, d_ff)
    packs = init_weights(spec, np.random.default_rng(0))
    assert len(packs) == 1
    assert packs[0].shape == (net_length(spec, d_ff),)


def test_window_and_heads_never_move_the_pack() -> None:
    """Spec S3's premise, pinned rather than asserted in prose: ALiBi's slopes are constants
    and `heads` only reshapes the same `W_qkv`, so `d_ff` is the ONLY sized knob. If this ever
    fails, the whole sizing procedure above is measuring the wrong thing."""
    base = init_weights(cell_spec(), np.random.default_rng(8))[0]
    for window, heads in ((1, 1), (4096, 4), (8, 4)):
        spec = cell_spec()
        spec["Transformer"] = {"window": window, "heads": heads, "d_ff": 5}
        assert np.array_equal(init_weights(spec, np.random.default_rng(8))[0], base), (window, heads)


def test_forward_direction_drops_exactly_one_stack() -> None:
    bi = init_weights(cell_spec("bidirectional"), np.random.default_rng(0))[0]
    fw = init_weights(cell_spec("forward"), np.random.default_rng(0))[0]
    spec = cell_spec("bidirectional")
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    stack = sum(transformer_nb(lstm[i + 1], lstm[i] * lsub[i], 5) for i in range(len(lstm) - 1))
    mlp_delta = 3 * lstm[-1]  # the MLP's first layer is `hidden`, not `2*hidden`, wide.
    assert bi.shape[0] - fw.shape[0] == stack + mlp_delta


def test_normalize_tail_is_identity() -> None:
    spec = cell_spec()
    flat = init_weights(spec, np.random.default_rng(5))[0]
    n = cast(list[int], spec["LSTMNeuronNb"])[0]
    assert np.all(flat[-2 * n : -n] == 0.0)
    assert np.all(flat[-n:] == 1.0)


def test_output_mlp_biases_are_zero() -> None:
    """The output MLP block is the SAME `[weights row-major | bias]` layout every other cell
    packs (`NeuronLayer::set_weights`), so its bias run is locatable from the tail backwards."""
    spec = cell_spec()
    flat = init_weights(spec, np.random.default_rng(6))[0]
    outn = cast(list[int], spec["OutputNeuronNb"])
    pos = flat.shape[0] - 2 * cast(list[int], spec["LSTMNeuronNb"])[0]
    for i in range(len(outn) - 1, 0, -1):
        pos -= outn[i]
        assert np.all(flat[pos : pos + outn[i]] == 0.0), i
        pos -= outn[i] * outn[i - 1]


def test_determinism_same_seed_is_bit_identical() -> None:
    spec = cell_spec()
    a = init_weights(spec, np.random.default_rng(11))[0]
    b = init_weights(spec, np.random.default_rng(11))[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, init_weights(spec, np.random.default_rng(12))[0])


def test_unknown_scheme_raises_in_the_builder_and_the_entry_point() -> None:
    with pytest.raises(ValueError, match="scheme"):
        init_transformer_flat(np.random.default_rng(0), 3, 3, 4, "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scheme"):
        init_weights(cell_spec(), np.random.default_rng(0), "bogus")  # type: ignore[arg-type]


def test_unknown_cell_type_still_raises() -> None:
    """The fifth cell widened the accepted set; the guard must still be a guard."""
    spec = {**cell_spec(), "CellType": "mamba2"}
    with pytest.raises(ValueError, match="unknown cell type"):
        init_weights(spec, np.random.default_rng(0))


# ------------------------------------------------------------------------------------- #
# Config bridge: the three `Transformer_*` keys reach the spec
# ------------------------------------------------------------------------------------- #


def _tuple_a_cfg() -> dict[str, str]:
    return config_bridge.parse_legacy_config((REF / "tupleA_1_worker_1.config").read_text())


def test_nnet_spec_defaults_to_the_sized_geometry() -> None:
    spec = config_bridge.nnet_spec(_tuple_a_cfg(), prefix="BLSTM")
    assert spec["Transformer"] == {"window": DEFAULT_WINDOW, "heads": DEFAULT_HEADS, "d_ff": SIZED_D_FF}
    assert spec["CellType"] == "lstm"  # the geometry is inert unless the cell is selected.
    assert transformer_geometry(spec) == (DEFAULT_WINDOW, DEFAULT_HEADS, SIZED_D_FF)


def test_nnet_spec_reads_the_port_only_keys() -> None:
    cfg = _tuple_a_cfg()
    cfg["BLSTM_Cell_Type"] = "transformer"
    cfg["Transformer_Window"] = "16"
    cfg["Transformer_Heads"] = "8"
    cfg["Transformer_D_Ff"] = "40"
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    assert spec["CellType"] == "transformer"
    # UNPREFIXED by design (one geometry per config, spec S2) -- read the same way for the LID
    # net, whose own cell key IS prefixed.
    assert spec["Transformer"] == {"window": 16, "heads": 8, "d_ff": 40}
    assert transformer_geometry(spec) == (16, 8, 40)


def test_transformer_geometry_falls_back_when_the_entry_is_absent() -> None:
    """A hand-built spec (no `nnet_spec` call) must still size: the reader defaults, it does
    not KeyError -- the `cfc_geometry`/`mamba_geometry` contract."""
    assert transformer_geometry({"LSTMNeuronNb": [1, 2]}) == (DEFAULT_WINDOW, DEFAULT_HEADS, SIZED_D_FF)


@pytest.mark.parametrize("key", ["Transformer_Window", "Transformer_Heads", "Transformer_D_Ff"])
def test_degenerate_transformer_geometry_is_rejected(key: str) -> None:
    """`TransformerParams::from_legacy` (blstm.rs) hard-bails below 1; mirrored here because
    the builders are reachable WITHOUT the engine, where a 0 `d_ff` would emit a silently
    FFN-less pack that no length assert could distinguish from a valid one."""
    cfg = _tuple_a_cfg()
    cfg[key] = "0"
    with pytest.raises(ValueError, match=key):
        config_bridge.nnet_spec(cfg, prefix="BLSTM")


def test_the_rust_and_python_defaults_agree() -> None:
    """The geometry lives in TWO languages: `blstm.rs::TRANSFORMER_DEFAULT_*` sizes the net the
    engine builds, `config_bridge` sizes the pack Python seeds it with. A `d_ff` drift is a
    length mismatch at `set_weights`; a `window`/`heads` drift is worse, because it changes the
    ARITHMETIC with no length to catch it -- so all three are pinned, not just the sized one."""
    src = Path("src/rust/src/nn/blstm.rs").read_text()
    for name, want, mirror in (
        ("TRANSFORMER_DEFAULT_WINDOW", DEFAULT_WINDOW, config_bridge.TRANSFORMER_DEFAULT_WINDOW),
        ("TRANSFORMER_DEFAULT_HEADS", DEFAULT_HEADS, config_bridge.TRANSFORMER_DEFAULT_HEADS),
        ("TRANSFORMER_DEFAULT_D_FF", SIZED_D_FF, config_bridge.TRANSFORMER_DEFAULT_D_FF),
    ):
        m = re.search(rf"pub const {name}: usize = (\d+);", src)
        assert m is not None, f"the Rust constant {name} moved or was renamed"
        assert int(m.group(1)) == want == mirror, name


# ------------------------------------------------------------------------------------- #
# THE SIZING (spec S3) -- the arithmetic and the candidate table are in this docstring
# ------------------------------------------------------------------------------------- #

_V1: dict[str, object] = {"LSTMNeuronNb": [23, 24, 24], "LSTMSubSampling": [4, 1], "OutputNeuronNb": [48, 12, 1], "OutputSubSampling": [1, 1]}
_V2: dict[str, object] = {**_V1, "LSTMNeuronNb": [11, 24, 24]}
_BAND = 0.15


def _lstm_len(spec: dict[str, object]) -> int:
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    outn = cast(list[int], spec["OutputNeuronNb"])
    stacks = 1 if spec.get("Direction", "bidirectional") == "forward" else 2
    total = 0
    for i in range(len(lstm) - 1):
        out, fin = lstm[i + 1], lstm[i] * lsub[i]
        total += stacks * (4 * out * fin + 4 * out * out + 12 * out + 4 * out)
    for i in range(len(outn) - 1):
        total += outn[i + 1] * outn[i] + outn[i + 1]
    return total + 2 * lstm[0]


def _dev(spec: dict[str, object], d_ff: int) -> float:
    lstm = _lstm_len(spec)
    return (net_length(spec, d_ff) - lstm) / lstm


def test_lineage_pack_lengths_are_the_documented_arithmetic() -> None:
    """The two closed forms quoted in the module docstring, MEASURED. `16199 + 196*d` (v1) and
    `13871 + 196*d` (v2) are what the sizing decision rests on, so they are pinned, not prose --
    and the LSTM targets are re-reached here independently of the committed pack files."""
    assert _lstm_len(_V1) == 33671  # the committed tuple-A pack length, independently reached.
    assert _lstm_len(_V2) == 24431
    for d in (1, 54, 64, 89):
        assert net_length(_V1, d) == 16199 + 196 * d
        assert net_length(_V2, d) == 13871 + 196 * d


def test_the_default_is_inside_both_lineage_bands() -> None:
    """THE SIZING RULE (spec S3): prefer an integer inside BOTH bands. 64 is, and it is the
    SMALLEST such integer -- 63 misses v1 by 0.22 percentage points, which is the non-vacuity
    half (without it "inside the band" could be satisfied by a wide range with no reason to
    pick this end of it)."""
    assert abs(_dev(_V2, SIZED_D_FF)) < _BAND
    assert abs(_dev(_V1, SIZED_D_FF)) < _BAND
    assert net_length(_V2, SIZED_D_FF) == 26415
    assert net_length(_V1, SIZED_D_FF) == 28743
    assert abs(_dev(_V1, SIZED_D_FF - 1)) > _BAND  # 63 -> v1 at -15.22%, OUT.
    assert _dev(_V2, SIZED_D_FF) == pytest.approx(0.0812, abs=5e-5)
    assert _dev(_V1, SIZED_D_FF) == pytest.approx(-0.1464, abs=5e-5)


def test_the_two_bands_and_their_intersection() -> None:
    """The feasible set, measured end to end: v2 tolerates [36, 72], v1 tolerates [64, 114], so
    the intersection is [64, 72] and is NON-EMPTY -- which is why the spec's v2-wins tiebreak
    never had to fire. Both endpoints of each band are pinned on BOTH sides, so a band that
    silently widened or narrowed fails here."""
    v2_band = [d for d in range(1, 200) if abs(_dev(_V2, d)) <= _BAND]
    v1_band = [d for d in range(1, 200) if abs(_dev(_V1, d)) <= _BAND]
    assert (v2_band[0], v2_band[-1]) == (36, 72)
    assert (v1_band[0], v1_band[-1]) == (64, 114)
    both = sorted(set(v2_band) & set(v1_band))
    assert both == list(range(64, 73))
    assert both[0] == SIZED_D_FF


def test_the_v2_argmin_conflicts_with_v1_and_the_default_resolves_it() -> None:
    """THE RECORDED CONFLICT (spec S3's tiebreak clause). Unlike CfC -- where B = 45 was the v2
    argmin AND the smallest v1-tolerable integer at once -- the two criteria DIVERGE here: v2's
    argmin is 54, which sits at -20.46% on v1, outside its band. The primary rule (inside both)
    decides; and within the feasible set v2's deviation is monotone increasing, so the v2
    tiebreak applied there picks the SAME 64. Both readings agree, which is the point."""
    argmin_v2 = min(range(1, 200), key=lambda d: abs(_dev(_V2, d)))
    assert argmin_v2 == 54
    assert net_length(_V2, argmin_v2) == 24455
    assert abs(_dev(_V1, argmin_v2)) > _BAND  # THE CONFLICT: the v2 optimum is out of v1's band.
    assert _dev(_V1, argmin_v2) == pytest.approx(-0.2046, abs=5e-5)
    feasible = [d for d in range(1, 200) if abs(_dev(_V2, d)) <= _BAND and abs(_dev(_V1, d)) <= _BAND]
    assert min(feasible, key=lambda d: abs(_dev(_V2, d))) == SIZED_D_FF  # v2's pick INSIDE the feasible set.
    assert min(range(1, 200), key=lambda d: abs(_dev(_V1, d))) == 89  # v1's own argmin, for the record.


def test_forward_lineages_keep_the_same_verdict() -> None:
    """`cell_overlay` resizes the MLP input from 2H to H under `forward`, so both packs shed the
    same MLP rows and one stack -- the band verdict must not flip with the direction."""
    for base, lstm_want, trans_want in ((_V2, 12239, 13231), (_V1, 16871, 14407)):
        spec = {**base, "Direction": "forward", "OutputNeuronNb": [24, 12, 1]}
        assert _lstm_len(spec) == lstm_want
        assert net_length(spec, SIZED_D_FF) == trans_want
        assert abs(_dev(spec, SIZED_D_FF)) < _BAND


# ------------------------------------------------------------------------------------- #
# The four landed cells are BYTE-UNTOUCHED
# ------------------------------------------------------------------------------------- #


def test_other_cells_are_unchanged_by_the_transformer_geometry_entry() -> None:
    """A spec carrying the new `Transformer` entry (every spec does now -- `nnet_spec` always
    emits it) produces the SAME bytes as one without it, for every pre-phase-11 cell."""
    for cell, length in (("lstm", 33671), ("slstm", None), ("mamba", None), ("cfc", None)):
        spec = {**config_bridge.nnet_spec(_tuple_a_cfg(), prefix="BLSTM"), "CellType": cell}
        bare = {k: v for k, v in spec.items() if k != "Transformer"}
        a = init_weights(spec, np.random.default_rng(99))[0]
        b = init_weights(bare, np.random.default_rng(99))[0]
        assert np.array_equal(a, b), cell
        if length is not None:
            assert a.shape == (length,)


# ------------------------------------------------------------------------------------- #
# Baseline arm knob (phase-11 T9 fix round 1, review F2 -- the phase-9/10 `cell_overlay`/
# `--cell-type` CI-coverage precedent, `test_phase9_init.py::test_cell_type_knob_overlays_
# the_s6_key_only` / `test_phase10_init.py::test_cfc_knob_overlays_the_cell_key_only` and
# siblings, which this task's own production knob change had NONE of: the existing loops
# there iterate hardcoded cell tuples and do not pick "transformer" up automatically)
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
    `cell_overlay`'s own "THE FIFTH CELL NEEDS NO THIRD DERIVED KEY" docstring paragraph,
    pinned here rather than left as prose."""
    assert baseline.cell_overlay(_SAD_FLAT, "transformer", "bidirectional") == {"BLSTM_Cell_Type": "transformer"}


def test_transformer_knob_composes_with_forward() -> None:
    """`--direction forward` forces the SAME two derived keys every cell gets (the output
    MLP resize + `BLSTM_window 0`), transformer included -- no cell-specific branch."""
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
    """End to end through the arm surface (the CfC precedent,
    `test_phase10_init.py::test_overlaid_config_seeds_through_the_matching_builder`): the
    overlay lands on the flat config, `nnet_spec` reads it back, and `init_weights` emits a
    transformer-sized pack at the v1-lineage bidirectional length this file's own
    `net_length` closed form independently predicts."""
    cfg = {**_SAD_FLAT, **baseline.cell_overlay(_SAD_FLAT, "transformer", "bidirectional")}
    cfg |= {
        "BLSTM_NNetInputSize": "23",
        "BLSTM_CostLawSpeech": "log",
        "BLSTM_CostLawNoSpeech": "log",
        **{f"BLSTM_{d}_Is{p}Active": "true" for d in ("Forward", "Backward") for p in ("CellsPeepholes", "GatesPeepholes", "GatesRecurrentPeepholes")},
    }
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    flat = init_weights(spec, np.random.default_rng(0))[0]
    assert flat.shape == (16199 + 196 * SIZED_D_FF,)
    assert flat.shape == (net_length(spec, SIZED_D_FF),)
