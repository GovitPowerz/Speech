"""Phase 9 Task 4: per-cell seeded init builders + the baseline arm cell/direction knobs.

THE LAYOUT CONTRACT IS THE RUST WALK. Every offset asserted here is recomputed from the
spec block order (S2.2 / S3.2) INDEPENDENTLY of `init_weights.py`'s own construction --
the same one-hot-probe arithmetic `nn/cells/slstm.rs::flat_layout_positions_are_the_spec_order`
and `nn/cells/mamba.rs::flat_layout_positions_are_the_spec_order` (plus its `offsets()`
helper) pin on the Rust side. A Python builder that agrees with itself but not with the
Rust `set_weights` walk would still fail here.

The bit-identical `speech_rs.Engine` round trip (the seam-level proof that these packs
really are what `BlstmNetwork::set_weights` consumes) belongs to Task 5's
`tests/pyo3/test_phase9_seam.py` -- every `importorskip("speech_rs")` in this repo lives
under `tests/pyo3/`, which is the only tree the `python-pyo3` CI job runs.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from speech import config_bridge
from speech import weight_bridge as wb
from speech.drivers import baseline
from speech.drivers.spec import BaselineSpec
from speech.init_weights import MambaGeometry, init_mamba_flat, init_slstm_flat, init_weights

REF = Path("tests/reference_data/phase4d")


def _spec(config_name: str) -> dict[str, object]:
    cfg = config_bridge.parse_legacy_config((REF / config_name).read_text())
    return config_bridge.nnet_spec(cfg, prefix="BLSTM")


TUPLE_A_SPEC = _spec("tupleA_1_worker_1.config")

# Three shapes per cell: square (no mamba adapter), wide fan-in, and the degenerate 1x1.
SHAPES: tuple[tuple[int, int], ...] = ((3, 3), (7, 2), (1, 1))


# ------------------------------------------------------------------------------------- #
# The Rust formulas, transcribed independently (NOT imported from the builder module)
# ------------------------------------------------------------------------------------- #


def slstm_nb(out: int, fin: int) -> int:
    """`SlstmLayer::nb_of_weights` (S2.2): 4 gates x [R (out x out) | W (out x in) | b]."""
    return 4 * out * (out + fin + 1)


def mamba_blocks(out: int, fin: int, geom: MambaGeometry) -> list[tuple[str, int]]:
    """`mamba.rs`'s test-local `offsets()` helper, transcribed: (name, start) per S3.2 block
    plus a trailing ("END", total)."""
    di = geom.expand * out
    dr = geom.dt_rank if geom.dt_rank else max(1, -(-out // 16))
    lens: list[tuple[str, int]] = []
    if fin != out:
        lens += [("P", out * fin), ("p", out)]
    lens += [
        ("g", out),
        ("W_in", 2 * di * out),
        ("conv", di * geom.d_conv),
        ("b_conv", di),
        ("W_x", (dr + 2 * geom.d_state) * di),
        ("W_dt", di * dr),
        ("b_dt", di),
        ("A_log", di * geom.d_state),
        ("D", di),
        ("W_out", out * di),
    ]
    blocks: list[tuple[str, int]] = []
    pos = 0
    for name, length in lens:
        blocks.append((name, pos))
        pos += length
    blocks.append(("END", pos))
    return blocks


def mamba_block(out: int, fin: int, geom: MambaGeometry, name: str) -> tuple[int, int]:
    blocks = mamba_blocks(out, fin, geom)
    k = [n for n, _ in blocks].index(name)
    return blocks[k][1], blocks[k + 1][1]


def mamba_nb(out: int, fin: int, geom: MambaGeometry) -> int:
    return mamba_blocks(out, fin, geom)[-1][1]


def net_length(spec: dict[str, object], cell_nb: Callable[[int, int], int]) -> int:
    """The full-net pack length: cell stacks (twice when bidirectional) + output MLP + tail.
    `cell_nb(out, fin)` is the per-layer count for the cell under test."""
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    outn = cast(list[int], spec["OutputNeuronNb"])
    dirs = 1 if spec.get("Direction", "bidirectional") == "forward" else 2
    total = 0
    for i in range(len(lstm) - 1):
        total += dirs * cell_nb(lstm[i + 1], lstm[i] * lsub[i])
    for i in range(len(outn) - 1):
        total += outn[i + 1] * outn[i] + outn[i + 1]
    return total + 2 * lstm[0]


def cell_spec(cell: str, direction: str, geom: MambaGeometry | None = None) -> dict[str, object]:
    """A small synthetic net spec: 2 recurrent layers, a 2-layer output MLP."""
    hidden = 4
    spec: dict[str, object] = {
        "LSTMNeuronNb": [5, 3, hidden],
        "LSTMSubSampling": [2, 1],
        "OutputNeuronNb": [hidden if direction == "forward" else 2 * hidden, 3, 1],
        "OutputSubSampling": [1, 1],
        "CellType": cell,
        "Direction": direction,
    }
    if geom is not None:
        spec["Mamba"] = {"d_state": geom.d_state, "d_conv": geom.d_conv, "expand": geom.expand, "dt_rank": geom.dt_rank}
    return spec


# ------------------------------------------------------------------------------------- #
# sLSTM (spec S2.2 layout / S2.4 init)
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize(("out", "fin"), SHAPES)
def test_slstm_layer_length_is_the_rust_formula(out: int, fin: int) -> None:
    flat = init_slstm_flat(np.random.default_rng(0), out, fin)
    assert flat.dtype == np.float64
    assert flat.shape == (slstm_nb(out, fin),)


@pytest.mark.parametrize(("out", "fin"), SHAPES)
def test_slstm_forget_bias_block_is_exactly_one_and_the_others_zero(out: int, fin: int) -> None:
    """S2.4: `b_f = +1`, every other bias 0. The block order is `[i|f|o|z]`, so the forget
    biases are gate block 1's trailing `out` run -- located by the S2.2 stride arithmetic,
    never by a hardcoded number."""
    flat = init_slstm_flat(np.random.default_rng(1), out, fin, forget_bias_one=True)
    stride = out * out + out * fin + out
    for gate, name in enumerate(("i", "f", "o", "z")):
        lo = gate * stride + out * out + out * fin
        want = 1.0 if name == "f" else 0.0
        assert np.all(flat[lo : lo + out] == want), f"gate {name} bias"


@pytest.mark.parametrize(("out", "fin"), SHAPES)
def test_slstm_forget_bias_one_false_zeroes_every_bias(out: int, fin: int) -> None:
    flat = init_slstm_flat(np.random.default_rng(1), out, fin, forget_bias_one=False)
    stride = out * out + out * fin + out
    for gate in range(4):
        lo = gate * stride + out * out + out * fin
        assert np.all(flat[lo : lo + out] == 0.0)


def test_slstm_weight_blocks_are_drawn_and_scaled_per_scheme() -> None:
    """The R/W halves of each gate block are non-degenerate and Xavier-bounded on the
    COMBINED fan_in (`fin + out`, the module's documented convention -- both halves feed one
    summed preactivation, exactly as for the legacy LSTM)."""
    out, fin = 24, 20
    flat = init_slstm_flat(np.random.default_rng(2), out, fin, scheme="xavier")
    limit = math.sqrt(6.0 / ((fin + out) + out))
    stride = out * out + out * fin + out
    for gate in range(4):
        weights = flat[gate * stride : gate * stride + out * out + out * fin]
        assert np.all(np.abs(weights) <= limit)
        assert np.unique(weights).size > 1


# ------------------------------------------------------------------------------------- #
# Mamba (spec S3.2 layout / S3.4 LOAD-BEARING init)
# ------------------------------------------------------------------------------------- #

GEOMS: tuple[MambaGeometry, ...] = (
    MambaGeometry(d_state=2, d_conv=2, expand=1, dt_rank=0),
    MambaGeometry(d_state=3, d_conv=4, expand=2, dt_rank=2),
    MambaGeometry(),  # the S6 defaults (16/4/2/0)
)


@pytest.mark.parametrize(("out", "fin"), SHAPES)
@pytest.mark.parametrize("geom", GEOMS)
def test_mamba_layer_length_is_the_rust_formula(out: int, fin: int, geom: MambaGeometry) -> None:
    flat = init_mamba_flat(np.random.default_rng(0), out, fin, geom)
    assert flat.dtype == np.float64
    assert flat.shape == (mamba_nb(out, fin, geom),)


def test_mamba_adapter_block_exists_only_when_widths_differ() -> None:
    geom = MambaGeometry(d_state=2, d_conv=2, expand=1, dt_rank=1)
    with_adapter = init_mamba_flat(np.random.default_rng(0), 3, 5, geom).shape[0]
    without = init_mamba_flat(np.random.default_rng(0), 3, 3, geom).shape[0]
    assert with_adapter - without == 3 * 5 + 3  # P (out x in) + p (out)


@pytest.mark.parametrize("geom", GEOMS)
def test_mamba_dt_rank_auto_resolves_like_the_rust_ctor(geom: MambaGeometry) -> None:
    """`dt_rank == 0` -> `ceil(d_model / 16).max(1)` (`MambaLayer::new`, spec S3.3)."""
    for out, want in ((1, 1), (16, 1), (17, 2), (64, 4)):
        expect = geom.dt_rank if geom.dt_rank else want
        assert geom.resolve_dt_rank(out) == expect


@pytest.mark.parametrize(("out", "fin"), SHAPES)
def test_mamba_spec_pinned_constants(out: int, fin: int) -> None:
    """S3.4 verbatim: `A_log[c,s] = log(s+1)`, `D = 1`, `g = 1`, `b_dt` in
    `softplus^-1([1e-3, 1e-1])`, `W_dt` inside `+-dt_rank^{-1/2}`; the two structural
    biases (`p`, `b_conv`) are 0."""
    geom = MambaGeometry(d_state=5, d_conv=3, expand=2, dt_rank=0)
    flat = init_mamba_flat(np.random.default_rng(3), out, fin, geom)
    di = geom.expand * out
    dr = geom.resolve_dt_rank(out)

    lo, hi = mamba_block(out, fin, geom, "A_log")
    a_log = flat[lo:hi].reshape(di, geom.d_state)
    want_row = np.log(np.arange(1, geom.d_state + 1, dtype=np.float64))
    assert np.array_equal(a_log, np.tile(want_row, (di, 1)))

    lo, hi = mamba_block(out, fin, geom, "D")
    assert np.all(flat[lo:hi] == 1.0)
    lo, hi = mamba_block(out, fin, geom, "g")
    assert np.all(flat[lo:hi] == 1.0)
    lo, hi = mamba_block(out, fin, geom, "b_conv")
    assert np.all(flat[lo:hi] == 0.0)
    if fin != out:
        lo, hi = mamba_block(out, fin, geom, "p")
        assert np.all(flat[lo:hi] == 0.0)

    # b_dt = softplus^-1(Delta_0), Delta_0 log-uniform in [1e-3, 1e-1]: assert on the
    # FORWARD image (softplus(b_dt) back in the band), so the test never re-derives the
    # inverse the builder used.
    lo, hi = mamba_block(out, fin, geom, "b_dt")
    b_dt = flat[lo:hi]
    assert b_dt.shape == (di,)
    delta0 = np.log1p(np.exp(b_dt))  # softplus, safe: b_dt is very negative here.
    assert np.all(delta0 >= 1e-3 - 1e-12)
    assert np.all(delta0 <= 1e-1 + 1e-12)

    lo, hi = mamba_block(out, fin, geom, "W_dt")
    w_dt = flat[lo:hi]
    assert w_dt.shape == (di * dr,)
    assert np.all(np.abs(w_dt) <= dr**-0.5)


# ------------------------------------------------------------------------------------- #
# WHOLE-PACK RECONSTRUCTION: the pin that catches a BLOCK SHEAR
# ------------------------------------------------------------------------------------- #
#
# THE GAP THIS CLOSES (T4 review, I1). Every other assertion in this file is blind to a
# permutation of two ADJACENT DRAWN blocks. The fingerprint pins locate only the CONSTANT
# blocks (`g`, `D`, `A_log`, the zero biases); the band checks cannot separate `W_x` from
# `W_dt` because Xavier's limit for `W_x` falls inside `W_dt`'s `+-dt_rank^{-1/2}` band; and
# a flat `set_weights`/`get_weights` round trip through the Engine is permutation-blind BY
# CONSTRUCTION, since both directions drive the same walk. A reviewer built exactly that
# shear (emit `W_dt` before `W_x`) and every committed assertion still passed.
#
# These two tests reconstruct the ENTIRE pack from a fresh `default_rng(seed)`, drawing block
# by block in spec order at the documented per-block shape/fans, and assert BIT-identity. A
# shear moves a draw to a different position in the stream AND gives it a different limit, so
# it cannot survive: position, scale, shape and draw order are all pinned at once. The draw
# helpers below are written out here rather than imported from `init_weights`, so this is an
# independent transcription, not a tautology.


def _xavier(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(rng.uniform(-limit, limit, size=shape), dtype=np.float64).reshape(-1)


def _he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    return np.asarray(rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape), dtype=np.float64).reshape(-1)


def _blk(rng: np.random.Generator, scheme: str, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    return _xavier(rng, shape, fan_in, fan_out) if scheme == "xavier" else _he(rng, shape, fan_in)


@pytest.mark.parametrize("scheme", ["xavier", "he"])
@pytest.mark.parametrize(("out", "fin"), SHAPES)
def test_slstm_pack_is_reproducible_block_by_block(out: int, fin: int, scheme: str) -> None:
    """S2.2 order `[i|f|o|z] x [R | W | b]`, `R`/`W` sharing the combined `fan_in = in + out`
    with `fan_out = out`, biases undrawn (0, or 1 for the forget gate)."""
    want: list[np.ndarray] = []
    rng = np.random.default_rng(21)
    for gate in range(4):
        want.append(_blk(rng, scheme, (out, out), fin + out, out))
        want.append(_blk(rng, scheme, (out, fin), fin + out, out))
        want.append(np.full(out, 1.0 if gate == 1 else 0.0))
    got = init_slstm_flat(np.random.default_rng(21), out, fin, scheme)  # type: ignore[arg-type]
    assert np.array_equal(got, np.concatenate(want))


@pytest.mark.parametrize("scheme", ["xavier", "he"])
@pytest.mark.parametrize(("out", "fin"), SHAPES)
def test_mamba_pack_is_reproducible_block_by_block(out: int, fin: int, scheme: str) -> None:
    """S3.2 order with the S3.4 constants, block by block. `W_x` and `W_dt` are ADJACENT
    DRAWN blocks with overlapping value ranges -- transposing them is precisely the shear no
    other assertion in this file can see."""
    geom = MambaGeometry(d_state=4, d_conv=3, expand=2, dt_rank=0)
    di, ds, dc = geom.d_inner(out), geom.d_state, geom.d_conv
    dr = geom.resolve_dt_rank(out)

    want: list[np.ndarray] = []
    rng = np.random.default_rng(22)
    if fin != out:
        want.append(_blk(rng, scheme, (out, fin), fin, out))  # P
        want.append(np.zeros(out))  # p
    want.append(np.ones(out))  # g
    want.append(_blk(rng, scheme, (2 * di, out), out, 2 * di))  # W_in
    want.append(_blk(rng, scheme, (di, dc), dc, dc))  # conv (depthwise: fan_in = fan_out = d_conv)
    want.append(np.zeros(di))  # b_conv
    want.append(_blk(rng, scheme, (dr + 2 * ds, di), di, dr + 2 * ds))  # W_x
    want.append(np.asarray(rng.uniform(-(dr**-0.5), dr**-0.5, size=(di, dr))).reshape(-1))  # W_dt
    delta0 = np.exp(rng.uniform(math.log(1e-3), math.log(1e-1), size=di))  # log-uniform per channel
    want.append(delta0 + np.log1p(-np.exp(-delta0)))  # b_dt = softplus^-1(Delta_0)
    want.append(np.tile(np.log(np.arange(1, ds + 1, dtype=np.float64)), (di, 1)).reshape(-1))  # A_log
    want.append(np.ones(di))  # D
    want.append(_blk(rng, scheme, (out, di), di, out))  # W_out

    got = init_mamba_flat(np.random.default_rng(22), out, fin, geom, scheme)  # type: ignore[arg-type]
    assert np.array_equal(got, np.concatenate(want))


def test_mamba_reconstruction_rejects_a_wx_wdt_shear() -> None:
    """Non-vacuity for the pin above, and the exact review finding: swapping the two adjacent
    drawn blocks `W_x`/`W_dt` must NOT reproduce the builder's pack."""
    out, fin = 3, 3
    geom = MambaGeometry(d_state=4, d_conv=3, expand=2, dt_rank=0)
    di, ds, dc = geom.d_inner(out), geom.d_state, geom.d_conv
    dr = geom.resolve_dt_rank(out)

    sheared: list[np.ndarray] = []
    rng = np.random.default_rng(22)
    sheared.append(np.ones(out))
    sheared.append(_xavier(rng, (2 * di, out), out, 2 * di))
    sheared.append(_xavier(rng, (di, dc), dc, dc))
    sheared.append(np.zeros(di))
    # THE SHEAR: W_dt emitted before W_x.
    sheared.append(np.asarray(rng.uniform(-(dr**-0.5), dr**-0.5, size=(di, dr))).reshape(-1))
    sheared.append(_xavier(rng, (dr + 2 * ds, di), di, dr + 2 * ds))
    delta0 = np.exp(rng.uniform(math.log(1e-3), math.log(1e-1), size=di))
    sheared.append(delta0 + np.log1p(-np.exp(-delta0)))
    sheared.append(np.tile(np.log(np.arange(1, ds + 1, dtype=np.float64)), (di, 1)).reshape(-1))
    sheared.append(np.ones(di))
    sheared.append(_xavier(rng, (out, di), di, out))

    got = init_mamba_flat(np.random.default_rng(22), out, fin, geom)
    bad = np.concatenate(sheared)
    assert got.shape == bad.shape  # same LENGTH -- which is why a length pin cannot see it.
    assert not np.array_equal(got, bad)


def test_mamba_projection_blocks_are_non_degenerate() -> None:
    """The Xavier/He-drawn blocks (`W_in`, `W_x`, `W_out`, `conv`) are actually drawn -- a
    zero-filled projection would make every S3.4 constant above pass vacuously."""
    geom = MambaGeometry(d_state=4, d_conv=3, expand=2, dt_rank=2)
    flat = init_mamba_flat(np.random.default_rng(4), 6, 6, geom)
    for name in ("W_in", "W_x", "W_out", "conv"):
        lo, hi = mamba_block(6, 6, geom, name)
        block = flat[lo:hi]
        assert np.unique(block).size > 1, name
        assert np.all(np.isfinite(block)), name


# ------------------------------------------------------------------------------------- #
# Net-level assembly: cell stacks x direction + output MLP + normalize tail
# ------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("direction", ["bidirectional", "forward"])
def test_net_pack_length_slstm(direction: str) -> None:
    spec = cell_spec("slstm", direction)
    packs = init_weights(spec, np.random.default_rng(0))
    assert len(packs) == 1
    assert packs[0].shape == (net_length(spec, slstm_nb),)


@pytest.mark.parametrize("direction", ["bidirectional", "forward"])
def test_net_pack_length_mamba(direction: str) -> None:
    geom = MambaGeometry(d_state=3, d_conv=2, expand=2, dt_rank=1)
    spec = cell_spec("mamba", direction, geom)
    packs = init_weights(spec, np.random.default_rng(0))
    assert packs[0].shape == (net_length(spec, lambda o, f: mamba_nb(o, f, geom)),)


@pytest.mark.parametrize("direction", ["bidirectional", "forward"])
def test_net_pack_length_lstm(direction: str) -> None:
    spec = cell_spec("lstm", direction)
    packs = init_weights(spec, np.random.default_rng(0))
    assert packs[0].shape == (wb.element_count(spec),)
    assert packs[0].shape == (net_length(spec, lambda o, f: 4 * o * f + 4 * o * o + 12 * o + 4 * o),)


def test_forward_direction_halves_the_recurrent_half_of_the_pack() -> None:
    """Non-vacuity for the direction knob: dropping the backward stack removes EXACTLY one
    copy of the cell layers, nothing else."""
    for cell, cell_nb in (("lstm", lambda o, f: 4 * o * f + 4 * o * o + 16 * o), ("slstm", slstm_nb)):
        bi = init_weights(cell_spec(cell, "bidirectional"), np.random.default_rng(0))[0]
        fw = init_weights(cell_spec(cell, "forward"), np.random.default_rng(0))[0]
        spec = cell_spec(cell, "bidirectional")
        lstm = cast(list[int], spec["LSTMNeuronNb"])
        lsub = cast(list[int], spec["LSTMSubSampling"])
        stack = sum(cell_nb(lstm[i + 1], lstm[i] * lsub[i]) for i in range(len(lstm) - 1))
        # The output MLP shrinks too (its first layer is `hidden`, not `2*hidden`, wide).
        hidden = lstm[-1]
        mlp_delta = 3 * hidden  # OutputNeuronNb[1] == 3 rows x hidden fewer weights.
        assert bi.shape[0] - fw.shape[0] == stack + mlp_delta, cell


def test_normalize_tail_is_identity_for_every_cell() -> None:
    for cell, geom in (("slstm", None), ("mamba", MambaGeometry(d_state=2, d_conv=2, expand=1, dt_rank=1))):
        spec = cell_spec(cell, "bidirectional", geom)
        flat = init_weights(spec, np.random.default_rng(5))[0]
        n = cast(list[int], spec["LSTMNeuronNb"])[0]
        assert np.all(flat[-2 * n : -n] == 0.0)
        assert np.all(flat[-n:] == 1.0)


def test_output_mlp_biases_are_zero_for_every_cell() -> None:
    """The output MLP block is the SAME `[weights row-major | bias]` layout the LSTM path
    packs (`NeuronLayer::set_weights`), so its bias run is locatable from the tail backwards."""
    for cell in ("slstm", "mamba"):
        spec = cell_spec(cell, "bidirectional", MambaGeometry(d_state=2, d_conv=2, expand=1, dt_rank=1))
        flat = init_weights(spec, np.random.default_rng(6))[0]
        outn = cast(list[int], spec["OutputNeuronNb"])
        n = cast(list[int], spec["LSTMNeuronNb"])[0]
        pos = flat.shape[0] - 2 * n
        for i in range(len(outn) - 1, 0, -1):
            pos -= outn[i]
            assert np.all(flat[pos : pos + outn[i]] == 0.0), (cell, i)
            pos -= outn[i] * outn[i - 1]


@pytest.mark.parametrize("cell", ["lstm", "slstm", "mamba"])
def test_determinism_same_seed_is_bit_identical(cell: str) -> None:
    spec = cell_spec(cell, "bidirectional", MambaGeometry(d_state=2, d_conv=2, expand=1, dt_rank=1))
    a = init_weights(spec, np.random.default_rng(11))[0]
    b = init_weights(spec, np.random.default_rng(11))[0]
    assert np.array_equal(a, b)
    c = init_weights(spec, np.random.default_rng(12))[0]
    assert not np.array_equal(a, c)


def test_unknown_scheme_raises_in_every_public_entry_point() -> None:
    """`_draw`'s else-branch is He, so an unrecognized string would otherwise be silently
    honoured as "he" in the per-cell builders (reachable from untyped callers)."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="scheme"):
        init_slstm_flat(rng, 3, 3, "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scheme"):
        init_mamba_flat(rng, 3, 3, MambaGeometry(2, 2, 1, 1), "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scheme"):
        init_weights(cell_spec("slstm", "bidirectional"), rng, "bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(("key", "value"), [("Mamba_D_State", "0"), ("Mamba_D_Conv", "0"), ("Mamba_Expand", "0")])
def test_degenerate_mamba_geometry_is_rejected(key: str, value: str) -> None:
    """`MambaParams::from_legacy` (blstm.rs:252-255) hard-bails below 1; mirrored here because
    the builders are reachable without the engine, where a 0 would emit a silently degenerate
    pack. `dt_rank = 0` stays legal -- it means auto."""
    cfg = config_bridge.parse_legacy_config((REF / "tupleA_1_worker_1.config").read_text())
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        config_bridge.nnet_spec(cfg, prefix="BLSTM")

    cfg2 = config_bridge.parse_legacy_config((REF / "tupleA_1_worker_1.config").read_text())
    cfg2["Mamba_Dt_Rank"] = "0"
    assert cast(dict[str, int], config_bridge.nnet_spec(cfg2, prefix="BLSTM")["Mamba"])["dt_rank"] == 0


def test_unknown_cell_type_raises() -> None:
    with pytest.raises(ValueError):
        init_weights(cell_spec("gru", "bidirectional"), np.random.default_rng(0))


def test_unknown_direction_raises() -> None:
    with pytest.raises(ValueError):
        init_weights(cell_spec("lstm", "backward"), np.random.default_rng(0))


# ------------------------------------------------------------------------------------- #
# The LSTM path is BYTE-UNTOUCHED (spec S7.1)
# ------------------------------------------------------------------------------------- #


def test_lstm_path_is_byte_identical_with_and_without_the_new_keys() -> None:
    """A spec carrying no cell/direction keys at all (every pre-phase-9 caller) and one
    carrying the explicit defaults produce the SAME bytes -- the new dispatch is inert."""
    bare = dict(TUPLE_A_SPEC)
    bare.pop("CellType", None)
    bare.pop("Direction", None)
    explicit = {**bare, "CellType": "lstm", "Direction": "bidirectional"}
    a = init_weights(bare, np.random.default_rng(99))[0]
    b = init_weights(explicit, np.random.default_rng(99))[0]
    assert np.array_equal(a, b)
    assert a.shape == (33671,)  # the tuple-A pin the phase-5 suite already carries.


def test_forward_only_lstm_pack_round_trips_through_the_packer_inverse() -> None:
    """`flat_to_nnet` follows the direction too, so a forward-only pack still unpacks into
    locatable gate matrices (no backward layers) -- the forget-bias assertion below is made
    through the packer's own inverse, never a hardcoded flat offset."""
    spec = cell_spec("lstm", "forward")
    flat = init_weights(spec, np.random.default_rng(13))[0]
    net = wb.unpack_weights(flat, spec)
    assert cast(list[object], net["backward"]) == []
    layers = cast(list[dict[str, np.ndarray]], net["forward"])
    assert len(layers) == len(cast(list[int], spec["LSTMNeuronNb"])) - 1
    for layer in layers:
        assert np.all(layer["forget"][:, -1] == 1.0)
        assert np.all(layer["input"][:, -1] == 0.0)
    assert np.array_equal(wb.pack_weights(net), flat)


def test_element_count_follows_the_direction() -> None:
    bi = cell_spec("lstm", "bidirectional")
    fw = cell_spec("lstm", "forward")
    lstm = cast(list[int], bi["LSTMNeuronNb"])
    lsub = cast(list[int], bi["LSTMSubSampling"])
    stack = sum(4 * lstm[i + 1] * (lstm[i] * lsub[i]) + 4 * lstm[i + 1] ** 2 + 16 * lstm[i + 1] for i in range(len(lstm) - 1))
    hidden = lstm[-1]
    assert wb.element_count(bi) - wb.element_count(fw) == stack + 3 * hidden
    # No key at all == bidirectional (byte-identical default for every pre-phase-9 caller).
    bare = {k: v for k, v in bi.items() if k != "Direction"}
    assert wb.element_count(bare) == wb.element_count(bi)


# ------------------------------------------------------------------------------------- #
# Config bridge: the S6 keys reach the spec
# ------------------------------------------------------------------------------------- #


def test_nnet_spec_defaults_to_the_legacy_shape() -> None:
    cfg = config_bridge.parse_legacy_config((REF / "tupleA_1_worker_1.config").read_text())
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    assert spec["CellType"] == "lstm"
    assert spec["Direction"] == "bidirectional"
    assert spec["Mamba"] == {"d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": 0}


def test_nnet_spec_reads_the_port_only_keys() -> None:
    cfg = config_bridge.parse_legacy_config((REF / "tupleA_1_worker_1.config").read_text())
    cfg["BLSTM_Cell_Type"] = "mamba"
    cfg["BLSTM_Direction"] = "forward"
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
    """End to end through the arm surface: the overlay lands on the flat config, `nnet_spec`
    reads it back, and `init_weights` emits an sLSTM-sized pack."""
    cfg = {**_SAD_FLAT, **baseline.cell_overlay(_SAD_FLAT, "slstm", "forward")}
    cfg |= {
        "BLSTM_NNetInputSize": "23",
        "BLSTM_CostLawSpeech": "log",
        "BLSTM_CostLawNoSpeech": "log",
        **{f"BLSTM_{d}_Is{p}Active": "true" for d in ("Forward", "Backward") for p in ("CellsPeepholes", "GatesPeepholes", "GatesRecurrentPeepholes")},
    }
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    flat = init_weights(spec, np.random.default_rng(0))[0]
    assert flat.shape == (net_length(spec, slstm_nb),)


def test_parser_exposes_the_knobs_with_legacy_defaults() -> None:
    args = baseline.build_parser().parse_args(["sad", "--corpus-root", "/tmp/c", "--out-dir", "/tmp/o"])
    assert args.cell == "lstm"
    assert args.direction == "bidirectional"
    args = baseline.build_parser().parse_args(["sad", "--corpus-root", "/tmp/c", "--out-dir", "/tmp/o", "--cell", "mamba", "--direction", "forward"])
    assert args.cell == "mamba"
    assert args.direction == "forward"


def test_non_default_knobs_are_rejected_on_the_lid_arms(tmp_path: Path) -> None:
    """The knobs target the `BLSTM_` (SAD) net. On a Twin arm that net is the FROZEN SAD
    gate (Mode 7 never runs it), so swapping its cell would change nothing that trains --
    a silent no-op is worse than a loud bail; the LID-net wiring is Task 5's job."""
    with pytest.raises(ValueError, match="cell|direction"):
        baseline.run_baseline(BaselineSpec(arm="lid-phseq", corpus_root=tmp_path, out_dir=tmp_path / "out", cell="slstm"))
    with pytest.raises(ValueError, match="cell|direction"):
        baseline.run_baseline(BaselineSpec(arm="lid-features", corpus_root=tmp_path, out_dir=tmp_path / "out", direction="forward"))
