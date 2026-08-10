"""Phase 10 Task 2: the CfC geometry reader, seeded init builder, and the SIZED default.

THE LAYOUT CONTRACT IS THE RUST WALK (`nn/cells/cfc.rs::for_each_slot`, spec S1.2). Every
offset asserted here is recomputed from the spec block order INDEPENDENTLY of
`init_weights.py`'s own construction -- a THIRD derivation, alongside the walk itself and
the test-local `offsets()` table in `cfc.rs`'s unit tests. A Python builder that agrees
with itself but not with the Rust `set_weights` walk still fails here.

    W_bb (B x (in+h)) | b_bb (B)                      the layer-0 backbone
    per deeper layer:  W_l (B x B) | b_l (B)          l = 2..L
    W_1 (h x B) | b_1 (h) | W_2 | b_2 | W_t | b_t     the three heads, [ff1|ff2|gate]

    nb_of_weights = B*(in+h+1) + (L-1)*B*(B+1) + 3*h*(B+1)

Every matrix is ROW-MAJOR at its MATH shape (output unit outer, source index inner); the
Rust stores them transposed and the walk absorbs that, so the flat order is the math order.

The bit-identical `speech_rs.Engine` round trip (the seam-level proof that these packs are
what `BlstmNetwork::set_weights` consumes) belongs to Task 3's `tests/pyo3/` tier -- every
`importorskip("speech_rs")` in this repo lives there.


THE SIZING PROCEDURE (spec S1.4) -- how `CFC_DEFAULT_BACKBONE_UNITS = 45` was picked
=====================================================================================

The rule: pick `B` (at `Cfc_Backbone_Layers 1`) so the FULL-NET CfC pack lands within
+-15% of the SAME LINEAGE's LSTM pack (the phase-9 S8.2 band). Both packs share the output
MLP and the normalize tail, so only the recurrent stacks move.

Per-layer counts, at `hidden h = 24` throughout and `stacks = 2` (bidirectional):

    LSTM   4*h*fin + 4*h*h + 12*h + 4*h        (gates+cell, the 12-row peephole bundle, biases)
    CfC    B*(fin+h+1) + 3*h*(B+1)             (S1.2 at L = 1)

v1 lineage (`lre_sad.toml`: LSTMNeuronNb 23,24,24 / SubSampling 4,1 / MLP 48,12,1 / tail 46)
  layer 0 fan-in 23*4 = 92, layer 1 fan-in 24.
    LSTM  = 2*(11520 + 4992) + 601 + 46      = 33671   (the committed tuple-A pack length)
    CfC   = 2*((117B + 72B + 72) + (49B + 72B + 72)) + 601 + 46
          = 620*B + 935
    matched B = 53 -> 33795 (+0.37%); at B = 45 -> 28835 (-14.36%), still INSIDE the band.

v2 lineage (spec S3.1: LSTMNeuronNb 11,24,24, everything else identical -> tail 22)
  layer 0 fan-in 11*4 = 44 (the honest feature width; v1's 92 is the 2015 dead-column
  mismatch), layer 1 fan-in 24.
    LSTM  = 2*(6912 + 4992) + 601 + 22       = 24431
    CfC   = 2*((69B + 72B + 72) + (49B + 72B + 72)) + 601 + 22
          = 524*B + 911
    matched B = 45 -> 24491 (+0.25%).

THE DEFAULT IS SIZED AGAINST v2 (the lineage the phase actually targets), so `B = 45`.
The v1 band is `B in [45, 60]` and the v2 band is `B in [38, 51]`: 45 is the v2 OPTIMUM and
simultaneously the SMALLEST integer v1 tolerates, so one default serves both lineages and
NO per-config override is needed -- worth stating because the brief allowed v1 to override
(v1's CfC leg is MECHANICAL-only, it carries no param-match requirement at all).

Forward-only runs move both packs the same way (one stack instead of two, and
`cell_overlay` resizes the MLP's input from 2h to h), so the ratio barely shifts:
v2 fwd LSTM 12239 vs CfC 12269 (+0.25%), v1 fwd LSTM 16871 vs CfC 14453 (-14.33%).
"""

import math
import re
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from speech import config_bridge
from speech.drivers import baseline
from speech.init_weights import init_cfc_flat, init_weights

REF = Path("tests/reference_data/phase4d")

# The finalized default (spec S1.4). Mirrored in THREE places -- `blstm.rs`'s
# `CFC_DEFAULT_BACKBONE_UNITS`, `config_bridge.CFC_DEFAULT_BACKBONE_UNITS`, and this
# literal -- and `test_the_rust_and_python_defaults_agree` pins the first two together.
SIZED_BACKBONE_UNITS = 45

# (out, fin, B, L): the degenerate 1x1, the FD grid's two-layer shape, a wide fan-in, and a
# THREE-layer shape.
#
# The L=3 row is not decoration. Every other shape here stops at `backbone_layers <= 2`,
# where the deeper-layer chain is a ONE-element list and "reverse the deeper layers" is a
# no-op -- so the parametrized reconstruction pin below was structurally blind to that
# permutation, and `test_reconstruction_rejects_a_backbone_layer_swap` was its SOLE
# detector (the phase-10 mutation battery's item 2b: 1 failed / 46 passed). With this row
# the same mutation fails the pin too (MEASURED: 3 failed / 49 passed of the 52 this file
# collects -- an earlier revision of this line wrote "55 passed", which is impossible: it
# would need 58 tests), so the permutation has two independent catchers and neither is
# load-bearing alone.
SHAPES: tuple[tuple[int, int, int, int], ...] = ((1, 1, 1, 1), (4, 5, 8, 2), (2, 3, 4, 1), (3, 7, 8, 1), (3, 5, 4, 3))


# ------------------------------------------------------------------------------------- #
# The Rust walk, transcribed independently (NOT imported from the builder module)
# ------------------------------------------------------------------------------------- #


def cfc_blocks(out: int, fin: int, b: int, layers: int) -> list[tuple[str, int]]:
    """`cfc.rs`'s test-local `offsets()` helper, transcribed: (name, start) per S1.2 block
    plus a trailing ("END", total)."""
    lens: list[tuple[str, int]] = []
    for li in range(layers):
        fan_in = fin + out if li == 0 else b
        lens += [(f"W_bb{li}", b * fan_in), (f"b_bb{li}", b)]
    for name in ("W_1", "W_2", "W_t"):
        lens += [(name, out * b), (name.replace("W", "b"), out)]
    blocks: list[tuple[str, int]] = []
    pos = 0
    for name, length in lens:
        blocks.append((name, pos))
        pos += length
    blocks.append(("END", pos))
    return blocks


def cfc_block(out: int, fin: int, b: int, layers: int, name: str) -> tuple[int, int]:
    blocks = cfc_blocks(out, fin, b, layers)
    k = [n for n, _ in blocks].index(name)
    return blocks[k][1], blocks[k + 1][1]


def cfc_nb(out: int, fin: int, b: int, layers: int) -> int:
    """The S1.2 CLOSED formula, written out separately from the block walk above so the two
    derivations cross-check each other (`test_layer_length_is_the_rust_formula`)."""
    return b * (fin + out + 1) + (layers - 1) * b * (b + 1) + 3 * out * (b + 1)


def net_length(spec: dict[str, object], b: int, layers: int) -> int:
    """Full-net pack length: cell stacks (twice when bidirectional) + output MLP + tail."""
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    outn = cast(list[int], spec["OutputNeuronNb"])
    stacks = 1 if spec.get("Direction", "bidirectional") == "forward" else 2
    total = sum(stacks * cfc_nb(lstm[i + 1], lstm[i] * lsub[i], b, layers) for i in range(len(lstm) - 1))
    for i in range(len(outn) - 1):
        total += outn[i + 1] * outn[i] + outn[i + 1]
    return total + 2 * lstm[0]


def cell_spec(direction: str = "bidirectional", b: int = 4, layers: int = 1) -> dict[str, object]:
    """A small synthetic net spec: 2 recurrent layers, a 2-layer output MLP."""
    hidden = 4
    return {
        "LSTMNeuronNb": [5, 3, hidden],
        "LSTMSubSampling": [2, 1],
        "OutputNeuronNb": [hidden if direction == "forward" else 2 * hidden, 3, 1],
        "OutputSubSampling": [1, 1],
        "CellType": "cfc",
        "Direction": direction,
        "Cfc": {"backbone_units": b, "backbone_layers": layers},
    }


# ------------------------------------------------------------------------------------- #
# Layer-level: lengths, block values
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize(("out", "fin", "b", "layers"), SHAPES)
def test_layer_length_is_the_rust_formula(out: int, fin: int, b: int, layers: int) -> None:
    assert cfc_blocks(out, fin, b, layers)[-1][1] == cfc_nb(out, fin, b, layers)  # walk == closed form
    flat = init_cfc_flat(np.random.default_rng(0), out, fin, b, layers)
    assert flat.dtype == np.float64
    assert flat.shape == (cfc_nb(out, fin, b, layers),)


@pytest.mark.parametrize(("out", "fin", "b", "layers"), SHAPES)
def test_every_bias_block_is_exactly_zero(out: int, fin: int, b: int, layers: int) -> None:
    """Spec S1.5: biases 0 EVERYWHERE. CfC has no forget-gate analogue to seed at 1 (the
    interpolation gate is a plain logistic between two live heads, not a memory valve), and
    no reference-lineage constants at all -- unlike Mamba's S3.4 block."""
    flat = init_cfc_flat(np.random.default_rng(1), out, fin, b, layers)
    names = [f"b_bb{li}" for li in range(layers)] + ["b_1", "b_2", "b_t"]
    for name in names:
        lo, hi = cfc_block(out, fin, b, layers, name)
        assert hi > lo, name
        assert np.all(flat[lo:hi] == 0.0), name


@pytest.mark.parametrize(("out", "fin", "b", "layers"), SHAPES)
def test_every_weight_block_is_drawn_and_finite(out: int, fin: int, b: int, layers: int) -> None:
    """Non-vacuity for the zero-bias pins above: a zero-filled projection would make them
    pass trivially."""
    flat = init_cfc_flat(np.random.default_rng(2), out, fin, b, layers)
    names = [f"W_bb{li}" for li in range(layers)] + ["W_1", "W_2", "W_t"]
    for name in names:
        lo, hi = cfc_block(out, fin, b, layers, name)
        block = flat[lo:hi]
        assert np.all(np.isfinite(block)), name
        assert np.all(block != 0.0), name


def test_xavier_and_he_scale_the_same_block_differently() -> None:
    """The scheme reaches the blocks: He-normal (`sqrt(2/fan_in)`) and Xavier-uniform
    (`sqrt(6/(fan_in+fan_out))`, std = limit/sqrt(3)) have different standard deviations at
    the same fans, so a builder ignoring `scheme` would show one ratio of 1.0."""
    out, fin, b, layers = 8, 8, 32, 1
    xav = init_cfc_flat(np.random.default_rng(3), out, fin, b, layers, "xavier")
    he = init_cfc_flat(np.random.default_rng(3), out, fin, b, layers, "he")
    lo, hi = cfc_block(out, fin, b, layers, "W_bb0")
    fan_in, fan_out = fin + out, b
    want = math.sqrt(2.0 / fan_in) / (math.sqrt(6.0 / (fan_in + fan_out)) / math.sqrt(3.0))
    assert np.std(he[lo:hi]) / np.std(xav[lo:hi]) == pytest.approx(want, rel=0.25)


# ------------------------------------------------------------------------------------- #
# THE WHOLE-PACK RECONSTRUCTION PIN (mandatory, the T4/phase-9 shear lesson)
# ------------------------------------------------------------------------------------- #
#
# Fingerprint and band pins are PERMUTATION-BLIND. CfC's three head matrices `W_1`, `W_2`,
# `W_t` are all `h x B` drawn from the SAME distribution at the SAME fans, so swapping any
# two of them is invisible to every length, band, shape and round-trip assertion in this
# file -- and a flat `set_weights`/`get_weights` round trip through the Engine cannot see it
# either, since both directions drive the same walk. The same is true of two deeper backbone
# layers (`B x B` each).
#
# The test below reconstructs the ENTIRE pack from a fresh `default_rng(seed)`, drawing block
# by block in spec order at the documented per-block shape/fans, and asserts BIT-identity: a
# permutation moves a draw to a different position in the stream, so position, scale, shape
# and draw order are all pinned at once. The draw helpers are written out here rather than
# imported from `init_weights`, so this is an independent transcription, not a tautology.


def _xavier(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(rng.uniform(-limit, limit, size=shape), dtype=np.float64).reshape(-1)


def _he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    return np.asarray(rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape), dtype=np.float64).reshape(-1)


def _blk(rng: np.random.Generator, scheme: str, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    return _xavier(rng, shape, fan_in, fan_out) if scheme == "xavier" else _he(rng, shape, fan_in)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
@pytest.mark.parametrize(("out", "fin", "b", "layers"), SHAPES)
def test_pack_is_reproducible_block_by_block(out: int, fin: int, b: int, layers: int, scheme: str) -> None:
    """S1.2 order at the S1.5 fans: the layer-0 backbone reads the CONCATENATED `[x | h]`
    vector through ONE matrix, so its `fan_in` is `in + h` -- the same combined value the
    LSTM/sLSTM paths reach by adding two separately-stored halves, here simply the matrix's
    own width. Deeper backbone layers are `B -> B`; each head is `B -> h` (three separate
    pre-activations, each fed by ONE product, so they do NOT share a combined fan)."""
    want: list[np.ndarray] = []
    rng = np.random.default_rng(31)
    for li in range(layers):
        fan_in = fin + out if li == 0 else b
        want.append(_blk(rng, scheme, (b, fan_in), fan_in, b))
        want.append(np.zeros(b))
    for _ in range(3):  # [ff1 | ff2 | gate]
        want.append(_blk(rng, scheme, (out, b), b, out))
        want.append(np.zeros(out))
    got = init_cfc_flat(np.random.default_rng(31), out, fin, b, layers, scheme)  # type: ignore[arg-type]
    assert np.array_equal(got, np.concatenate(want))


def test_reconstruction_rejects_a_head_swap() -> None:
    """Non-vacuity for the pin above: `W_1` and `W_2` are same-shape, same-fan, same-scheme
    ADJACENT DRAWN blocks -- the exact permutation no other assertion in this file can see."""
    out, fin, b, layers = 3, 7, 8, 1
    sheared: list[np.ndarray] = []
    rng = np.random.default_rng(31)
    sheared.append(_xavier(rng, (b, fin + out), fin + out, b))
    sheared.append(np.zeros(b))
    heads = [_xavier(rng, (out, b), b, out) for _ in range(3)]
    heads[0], heads[1] = heads[1], heads[0]  # THE SHEAR: ff2's draw emitted as ff1's block.
    for head in heads:
        sheared.append(head)
        sheared.append(np.zeros(out))

    got = init_cfc_flat(np.random.default_rng(31), out, fin, b, layers)
    bad = np.concatenate(sheared)
    assert got.shape == bad.shape  # same LENGTH -- which is why a length pin cannot see it.
    assert not np.array_equal(got, bad)


def test_reconstruction_rejects_a_backbone_layer_swap() -> None:
    """The second permutation the flat pack admits: two deeper backbone layers are both
    `B x B` at identical fans, so their order is only visible in the draw stream."""
    out, fin, b, layers = 4, 5, 8, 3
    rng = np.random.default_rng(31)
    parts: list[np.ndarray] = [_xavier(rng, (b, fin + out), fin + out, b), np.zeros(b)]
    deeper = [_xavier(rng, (b, b), b, b) for _ in range(layers - 1)]
    deeper.reverse()  # THE SHEAR: layer 2's draw emitted as layer 1's block.
    for w in deeper:
        parts += [w, np.zeros(b)]
    for _ in range(3):
        parts += [_xavier(rng, (out, b), b, out), np.zeros(out)]

    got = init_cfc_flat(np.random.default_rng(31), out, fin, b, layers)
    bad = np.concatenate(parts)
    assert got.shape == bad.shape
    assert not np.array_equal(got, bad)


# ------------------------------------------------------------------------------------- #
# Net-level assembly + determinism
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("direction", ["bidirectional", "forward"])
@pytest.mark.parametrize("layers", [1, 2])
def test_net_pack_length(direction: str, layers: int) -> None:
    spec = cell_spec(direction, b=6, layers=layers)
    packs = init_weights(spec, np.random.default_rng(0))
    assert len(packs) == 1
    assert packs[0].shape == (net_length(spec, 6, layers),)


def test_forward_direction_drops_exactly_one_stack() -> None:
    bi = init_weights(cell_spec("bidirectional", b=6), np.random.default_rng(0))[0]
    fw = init_weights(cell_spec("forward", b=6), np.random.default_rng(0))[0]
    spec = cell_spec("bidirectional", b=6)
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    stack = sum(cfc_nb(lstm[i + 1], lstm[i] * lsub[i], 6, 1) for i in range(len(lstm) - 1))
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
        init_cfc_flat(np.random.default_rng(0), 3, 3, 4, 1, "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scheme"):
        init_weights(cell_spec(), np.random.default_rng(0), "bogus")  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------- #
# Config bridge: the two `Cfc_*` keys reach the spec
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
    """`CfcParams::from_legacy` (blstm.rs) hard-bails below 1; mirrored here because the
    builders are reachable WITHOUT the engine, where a 0 would emit a silently degenerate
    pack (a head-only or backbone-less net) that no length assert could distinguish."""
    cfg = _tuple_a_cfg()
    cfg[key] = "0"
    with pytest.raises(ValueError, match=key):
        config_bridge.nnet_spec(cfg, prefix="BLSTM")


def test_the_rust_and_python_defaults_agree() -> None:
    """The sizing lives in TWO languages: `blstm.rs::CFC_DEFAULT_BACKBONE_UNITS` sizes the
    net the engine builds, `config_bridge` sizes the pack Python seeds it with. A drift
    between them is a length mismatch at `set_weights`, so pin them to each other -- the one
    cross-language constant this task introduces."""
    src = Path("src/rust/src/nn/blstm.rs").read_text()
    m = re.search(r"pub const CFC_DEFAULT_BACKBONE_UNITS: usize = (\d+);", src)
    assert m is not None, "the Rust default constant moved or was renamed"
    assert int(m.group(1)) == SIZED_BACKBONE_UNITS == config_bridge.CFC_DEFAULT_BACKBONE_UNITS


# ------------------------------------------------------------------------------------- #
# THE SIZING (spec S1.4) -- the arithmetic is in this module's docstring
# ------------------------------------------------------------------------------------- #

_V1: dict[str, object] = {"LSTMNeuronNb": [23, 24, 24], "LSTMSubSampling": [4, 1], "OutputNeuronNb": [48, 12, 1], "OutputSubSampling": [1, 1]}
_V2: dict[str, object] = {**_V1, "LSTMNeuronNb": [11, 24, 24]}


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


def test_lineage_pack_lengths_are_the_documented_arithmetic() -> None:
    """The two closed forms quoted in the module docstring, MEASURED. `620*B + 935` (v1) and
    `524*B + 911` (v2) are what the sizing decision rests on, so they are pinned, not prose."""
    assert _lstm_len(_V1) == 33671  # the committed tuple-A pack length, independently reached.
    assert _lstm_len(_V2) == 24431
    for b in (1, 24, 45, 53):
        assert net_length(_V1, b, 1) == 620 * b + 935
        assert net_length(_V2, b, 1) == 524 * b + 911


def test_the_default_matches_the_v2_lstm_pack_within_15_percent() -> None:
    """THE SIZING RULE (spec S1.4/S8.2). The default is sized against the v2 lineage: at
    B = 45 the CfC pack is +0.25% of v2's LSTM pack, and the neighbouring integers are
    strictly worse, so this is the optimum and not merely an in-band value."""
    lstm = _lstm_len(_V2)
    at = {b: net_length(_V2, b, 1) for b in (44, 45, 46)}
    rel = {b: abs(v - lstm) / lstm for b, v in at.items()}
    assert at[SIZED_BACKBONE_UNITS] == 24491
    assert rel[SIZED_BACKBONE_UNITS] < 0.01
    assert rel[45] < rel[44] and rel[45] < rel[46]


def test_the_default_also_leaves_the_v1_lineage_in_band() -> None:
    """v1 CfC is a MECHANICAL leg with no param-match requirement, so this is a bonus, not a
    gate -- but it is the reason no per-config override of the default is needed anywhere:
    B = 45 is simultaneously v2's optimum and the SMALLEST integer inside v1's band."""
    lstm = _lstm_len(_V1)
    assert abs(net_length(_V1, SIZED_BACKBONE_UNITS, 1) - lstm) / lstm < 0.15
    assert abs(net_length(_V1, SIZED_BACKBONE_UNITS - 1, 1) - lstm) / lstm > 0.15
    # v1's own optimum, quoted in the docstring, for the record.
    assert net_length(_V1, 53, 1) == 33795


def test_forward_lineages_keep_the_same_ratios() -> None:
    """`cell_overlay` resizes the MLP input from 2h to h under `forward`, so both packs shed
    the same MLP rows and one stack -- the band verdict must not flip with the direction."""
    for base, want_ratio in ((_V2, 0.01), (_V1, 0.15)):
        spec = {**base, "Direction": "forward", "OutputNeuronNb": [24, 12, 1]}
        lstm = _lstm_len(spec)
        assert abs(net_length(spec, SIZED_BACKBONE_UNITS, 1) - lstm) / lstm < want_ratio


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
    defaults live Rust-side and Python mirrors them, so omitting them keeps the config text
    minimal and the two sides agreeing by construction."""
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
    args = baseline.build_parser().parse_args(["sad", "--corpus-root", "/tmp/c", "--out-dir", "/tmp/o", "--cell-type", "cfc"])
    assert args.cell_type == "cfc"


def test_overlaid_config_seeds_through_the_matching_builder() -> None:
    """End to end through the arm surface: the overlay lands on the flat config, `nnet_spec`
    reads BOTH the cell type and the geometry back, and `init_weights` emits a CfC-sized pack
    at the SIZED default -- the v1-lineage number quoted in the docstring."""
    cfg = {**_SAD_FLAT, **baseline.cell_overlay(_SAD_FLAT, "cfc", "bidirectional")}
    cfg |= {
        "BLSTM_NNetInputSize": "23",
        "BLSTM_CostLawSpeech": "log",
        "BLSTM_CostLawNoSpeech": "log",
        **{f"BLSTM_{d}_Is{p}Active": "true" for d in ("Forward", "Backward") for p in ("CellsPeepholes", "GatesPeepholes", "GatesRecurrentPeepholes")},
    }
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    flat = init_weights(spec, np.random.default_rng(0))[0]
    assert flat.shape == (620 * SIZED_BACKBONE_UNITS + 935,)
    assert flat.shape == (net_length(spec, SIZED_BACKBONE_UNITS, 1),)


# ------------------------------------------------------------------------------------- #
# The three landed cells are BYTE-UNTOUCHED
# ------------------------------------------------------------------------------------- #


def test_other_cells_are_unchanged_by_the_cfc_geometry_entry() -> None:
    """A spec carrying the new `Cfc` entry (every spec does now -- `nnet_spec` always emits
    it) produces the SAME bytes as one without it, for every pre-phase-10 cell."""
    for cell, length in (("lstm", 33671), ("slstm", None), ("mamba", None)):
        spec = {**config_bridge.nnet_spec(_tuple_a_cfg(), prefix="BLSTM"), "CellType": cell}
        bare = {k: v for k, v in spec.items() if k != "Cfc"}
        a = init_weights(spec, np.random.default_rng(99))[0]
        b = init_weights(bare, np.random.default_rng(99))[0]
        assert np.array_equal(a, b), cell
        if length is not None:
            assert a.shape == (length,)
