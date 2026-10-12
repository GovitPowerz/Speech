"""Seeded weight initialization over the Rust weight layout (Phase 5 Task 3; issue #27).

The LAYOUT comes from the engine: `speech_rs.weight_layout(config, prefix)` lists every block
of the `prefix` net's pack in `BlstmNetwork::set_weights` order -- `stack`, `layer`, `name`,
`kind`, row-major storage `shape`, the structural `fan_in` / `fan_out` of the block's linear
map, `offset`, `len` -- so this module holds no offset arithmetic and no per-cell geometry
(ADR-0005: a new cell is a Rust `weight_layout` plus a draw rule per block kind here, exceptions
by name). What this module DOES own is the DISTRIBUTIONS: a draw rule per block kind, a short
name-keyed exception table, and the RNG consumption order.

Kind defaults (`_draw_block`):
  - `weight`: "xavier" = Glorot & Bengio (2010) UNIFORM, `U(+-sqrt(6/(fan_in+fan_out)))`;
    "he" = He et al. (2015) NORMAL, `N(0, sqrt(2/fan_in))` (fan_in only, no fan_out term).
    Both read the RECORD's fans: for the LSTM / sLSTM gate maps that is the combined
    `fan_in = in + out` (the input and recurrent halves feed ONE summed pre-activation), for
    the CfC backbone the concatenated `[x | h]` width, for the output MLP the SUB-SAMPLED
    input width, for the Mamba depthwise conv `(d_conv, d_conv)`.
  - `bias`: 0.0.  - `gain`: 1.0 (the RMSNorm `g` / `g_1` / `g_2`, the Mamba skip `D`).
  - `stat`: the normalize tail, `mean` 0.0 / `std` 1.0 (identity).
  - `peephole`: the legacy LSTM's 12-column bundle, `U(-0.1, 0.1)` -- every peephole term is
    an ELEMENTWISE product against a state or gate value, so there is no fan to scale by;
    drawn inside the LSTM order below, never through the kind default.

Exceptions by name (`_EXCEPTIONS`), every one a lineage constant copied, not re-derived:
  - `b.f` (LSTM) / `b_f` (sLSTM): 1.0 under `forget_bias_one` ("start the forget gate open",
    Jozefowicz et al. 2015), else 0.0.
  - Mamba (spec S3.4, LOAD-BEARING -- a from-scratch run without them is known-fragile):
    `W_dt` uniform `+-dt_rank^{-1/2}` (`dt_rank` is the record's fan_in); `b_dt =
    softplus^-1(Delta_0)` with `Delta_0` log-uniform in `[1e-3, 1e-1]` per channel; `A_log[c,
    s] = log(s + 1)` (S4D-real). `D = 1` and `g = 1` are the `gain` default.
  Nothing is zeroed beyond the kind defaults: the structurally dead blocks (sLSTM `b_i`, the
  transformer `b_k` third of `b_qkv`) seed 0 like every other bias and never move.

RNG ORDER. The four port-only cells, the output MLP and the tail draw in LAYOUT order, block by
block (there is no legacy trajectory to match; the least surprising order wins). The legacy
LSTM layer draws in the ORDER THE PHASE-5 BUILDER DREW -- per gate `i`, `f`, `o`: `r.g`, `w.g`,
the gate's three gate-peephole columns `(out, 3)`, its cell-peephole column `(out,)`; then
`r.c`, `w.c` -- which is NOT its layout order (`w.* | r.* | peep | b.*`), so `_LSTM_ORDER`
spells it out and the bundle is scattered by its named `columns`. Every committed seed pack
regenerates byte-identical through this (`tests/test_seed_pack_regeneration.py`, manifest-
driven); a change to any draw count or order fails there first.

Returns one flat pack per net (a 1-element list, matching `speech_rs.Engine.set_weights(pos,
nets)`'s `Vec<Vec<f64>>` contract); the algo-6 Twin's `[sad, lid]` pair is built by calling
`init_weights` once per prefix on one shared `Generator`, in that order.
"""

import math
from itertools import groupby
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

_PEEPHOLE_SCALE = 0.1

# Spec S3.4: Delta_0 log-uniform in [1e-3, 1e-1], seeded per channel.
_DELTA_MIN = 1e-3
_DELTA_MAX = 1e-1

Scheme = Literal["xavier", "he"]
Block = dict[str, Any]


def _check_scheme(scheme: str) -> None:
    """Validated at the public entry point, because `_draw`'s `else` branch is He: an
    unrecognized string would otherwise be silently honoured as "he" rather than rejected
    (`Scheme` is a `Literal`, so only a static checker catches it)."""
    if scheme not in ("xavier", "he"):
        raise ValueError(f"unknown scheme: {scheme!r}")


def _draw(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int, fan_out: int, scheme: Scheme) -> NDArray[np.float64]:
    if scheme == "xavier":
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return np.asarray(rng.uniform(-limit, limit, size=shape), dtype=np.float64).reshape(-1)
    return np.asarray(rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape), dtype=np.float64).reshape(-1)


def _softplus_inverse(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """`log(e^y - 1)`, written as the algebraically identical `y + log1p(-e^{-y})` for
    DOMAIN SAFETY (the direct form overflows `exp` above `y ~ 709.78`); on S3.4's
    `[1e-3, 1e-1]` band both forms agree to ~1e-13."""
    return cast(NDArray[np.float64], y + np.log1p(-np.exp(-y)))


def _forget_bias(block: Block, rng: np.random.Generator, forget_bias_one: bool) -> NDArray[np.float64]:
    return np.full(block["len"], 1.0 if forget_bias_one else 0.0, dtype=np.float64)


def _mamba_w_dt(block: Block, rng: np.random.Generator, forget_bias_one: bool) -> NDArray[np.float64]:
    limit = block["fan_in"] ** -0.5  # fan_in IS dt_rank
    return np.asarray(rng.uniform(-limit, limit, size=tuple(block["shape"])), dtype=np.float64).reshape(-1)


def _mamba_b_dt(block: Block, rng: np.random.Generator, forget_bias_one: bool) -> NDArray[np.float64]:
    delta0 = np.exp(rng.uniform(math.log(_DELTA_MIN), math.log(_DELTA_MAX), size=block["len"]))
    return _softplus_inverse(np.asarray(delta0, dtype=np.float64))


def _mamba_a_log(block: Block, rng: np.random.Generator, forget_bias_one: bool) -> NDArray[np.float64]:
    d_inner, d_state = block["shape"]
    return np.tile(np.log(np.arange(1, d_state + 1, dtype=np.float64)), (d_inner, 1)).reshape(-1)


_EXCEPTIONS = {
    "b.f": _forget_bias,
    "b_f": _forget_bias,
    "W_dt": _mamba_w_dt,
    "b_dt": _mamba_b_dt,
    "A_log": _mamba_a_log,
}


def _draw_block(block: Block, rng: np.random.Generator, scheme: Scheme, forget_bias_one: bool) -> NDArray[np.float64]:
    """One block's values, flat, in its storage order: the name-keyed exception if there is
    one, else the kind default."""
    if (special := _EXCEPTIONS.get(block["name"])) is not None:
        return special(block, rng, forget_bias_one)
    kind, n = block["kind"], block["len"]
    if kind == "weight":
        return _draw(rng, tuple(block["shape"]), block["fan_in"], block["fan_out"], scheme)
    if kind == "bias":
        return np.zeros(n, dtype=np.float64)
    if kind == "gain":
        return np.ones(n, dtype=np.float64)
    if kind == "stat":
        return np.full(n, 1.0 if block["name"] == "std" else 0.0, dtype=np.float64)
    raise ValueError(f"no draw rule for block {block['name']!r} of kind {kind!r}")


# The legacy LSTM layer's RNG order (the phase-5 builder's, byte-pinned by the regeneration
# test): per gate, the recurrent block, the input block, the gate's three gate-peephole columns,
# then its cell-peephole column; the cell input's recurrent and input blocks last.
_LSTM_ORDER: tuple[str, ...] = ("r.i", "w.i", "peep:i", "r.f", "w.f", "peep:f", "r.o", "w.o", "peep:o", "r.c", "w.c")


def _draw_lstm_layer(blocks: list[Block], rng: np.random.Generator, scheme: Scheme, forget_bias_one: bool) -> dict[str, NDArray[np.float64]]:
    by_name = {b["name"]: b for b in blocks}
    peep = by_name["peep"]
    out = peep["shape"][0]
    columns: list[str] = peep["columns"]
    bundle = np.zeros(tuple(peep["shape"]), dtype=np.float64)
    values: dict[str, NDArray[np.float64]] = {}
    for step in _LSTM_ORDER:
        if step.startswith("peep:"):
            gate = step.split(":")[1]
            gate_cols = [k for k, c in enumerate(columns) if c.startswith(f"{gate}.") and not c.startswith(f"{gate}.cell")]
            cell_col = next(k for k, c in enumerate(columns) if c.startswith(f"{gate}.cell"))
            bundle[:, gate_cols] = rng.uniform(-_PEEPHOLE_SCALE, _PEEPHOLE_SCALE, size=(out, len(gate_cols)))
            bundle[:, cell_col] = rng.uniform(-_PEEPHOLE_SCALE, _PEEPHOLE_SCALE, size=out)
        else:
            values[step] = _draw_block(by_name[step], rng, scheme, forget_bias_one)
    values["peep"] = bundle.reshape(-1)
    for b in blocks:
        if b["name"] not in values:  # the biases: no draw, kind default or the forget-gate exception
            values[b["name"]] = _draw_block(b, rng, scheme, forget_bias_one)
    return values


def init_weights(
    config: dict[str, str],
    prefix: str,
    rng: np.random.Generator,
    scheme: Scheme = "xavier",
    forget_bias_one: bool = True,
) -> list[NDArray[np.float64]]:
    """Seeded init for the `prefix` net's flat pack (see the module docstring).

    `config` is the flat legacy-key map the engine reads (`parse_legacy_config` /
    `load_toml_config`); the architecture, direction and per-cell geometry are resolved by
    the engine itself through `speech_rs.weight_layout`, so a config the engine refuses
    raises `ValueError` here too, never a default."""
    _check_scheme(scheme)
    import speech_rs  # local: the pyo3 module is only needed on the engine path

    layout = cast(list[Block], speech_rs.weight_layout(config, prefix))
    pack = np.zeros(sum(b["len"] for b in layout), dtype=np.float64)
    for _, group in groupby(layout, key=lambda b: (b["stack"], b["layer"])):
        blocks = list(group)
        if any(b["kind"] == "peephole" for b in blocks):
            values = _draw_lstm_layer(blocks, rng, scheme, forget_bias_one)
        else:
            values = {b["name"]: _draw_block(b, rng, scheme, forget_bias_one) for b in blocks}
        for b in blocks:
            pack[b["offset"] : b["offset"] + b["len"]] = values[b["name"]]
    return [pack]
