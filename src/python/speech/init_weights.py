"""Seeded weight initialization over the `weight_bridge` flat-packer spec (Phase 5 Task 3).

Builds an nnet-domain structure shaped EXACTLY like `weight_bridge.nnet_to_flat` expects
(`{"forward"/"backward": [layer -> {"input"/"forget"/"output"/"cell": 2D}], "output": [layer
-> 2D], "mean"/"std": 1D}`) and hands it to `nnet_to_flat` itself to flatten -- so every column
position (recurrent block, fan-in block, the 12-column peephole bundle, the per-gate bias) is
derived from the SAME packer code the rest of the port already golden-tests, never a
re-hardcoded offset. `spec` is the `NnetSpec`-shaped dict `config_bridge.nnet_spec` produces
(and `weight_bridge.element_count`/`unpack_weights`/`flat_to_nnet` already consume): only
`LSTMNeuronNb`/`LSTMSubSampling`/`OutputNeuronNb` are read (matching what the packer itself
reads for shapes -- `OutputSubSampling` only feeds `adim` scaling, irrelevant to init).

Sampling-variant choices ("xavier" and "he" both have more than one textbook variant; one
picked per name, documented here rather than re-litigated at each call site):
  - "xavier": Glorot & Bengio (2010) UNIFORM -- U(-sqrt(6/(fan_in+fan_out)), +same).
  - "he": He et al. (2015) NORMAL -- N(0, sqrt(2/fan_in)) -- fan_in-only, no fan_out term
    (the original derivation targets forward-pass variance under ReLU-family units, so it
    never symmetrizes over fan_out the way Xavier does).

Per-gate weight blocks (recurrent + fan-in): `LstmLayer::feed_forward` (nn/layers.rs) sums TWO
independent matmuls into ONE preactivation -- `matmul(input, input_weights) +
matmul(prev_out, feedback_weights)`. The packed row's recurrent(out) and fan-in(fin) column
ranges are therefore two halves of one combined linear map onto the same `out`-wide
preactivation, not two independently-scaled matrices: both halves are drawn from the SAME
distribution using a COMBINED fan_in = fin + out (fan_out = out throughout). This keeps the
summed preactivation variance calibrated as a single fan_in=(fin+out) affine map, and gives the
xavier-vs-he variance-ratio sanity test one homogeneous block to measure.

Peepholes (the 12-column bundle `_pack_lstm_layer` interleaves into the I/F/O rows' tail):
every peephole term in `feed_forward` is an ELEMENTWISE (diagonal) product against a state or
gate value -- e.g. `cell_states[t-1, j] * peep[k, j]` -- never a matrix product, so there is no
fan_in/fan_out to derive a Xavier/He scale from. Treated as small recurrent-diagonal weights:
drawn from a fixed U(-0.1, 0.1), independent of `scheme` (standard practice for diagonal/
peephole terms -- small enough that the peephole contribution does not dominate the gate
preactivation before training has adjusted it).

Biases: 0.0 everywhere except the LSTM forget gate, seeded at 1.0 when `forget_bias_one` (the
"start the forget gate open" trick, Jozefowicz et al. 2015) -- located by literally being the
forget-gate matrix's own last column inside the packer-shaped dict, never a flat-vector offset.

Normalize tail: mean 0.0 / std 1.0 (identity), each `LSTMNeuronNb[0]`-wide, matching
`nnet_to_flat`'s trailing `mean`/`std` append.

Returns one flat pack per net (a 1-element list for a single `spec`, matching
`speech_rs.Engine.set_weights(pos, nets)`'s `Vec<Vec<f64>>` contract directly); a multi-net
architecture (e.g. the algo-6 Twin's `[sad, lid]` pair) is built by calling `init_weights` once
per net spec and concatenating the returned lists.

## Phase 9 (Task 4): the sLSTM/Mamba builders emit the flat order DIRECTLY

A DELIBERATE DESIGN DIFFERENCE from the LSTM path above, not an oversight. The new cells have
NO packer-side structured/nnet form to build and flatten (spec S1.3: their weights exist only
as flat `.bin` packs / seam vectors -- `weight_bridge.nnet_to_flat` and its inverse are
LSTM-shaped and stay that way), so [`init_slstm_flat`] / [`init_mamba_flat`] write the flat
blocks themselves in the order the Rust `set_weights` walk consumes them
(`nn/cells/slstm.rs::for_each_slot` -- spec S2.2; `nn/cells/mamba.rs::for_each_slot` -- spec
S3.2). Both walks are the layout's SINGLE definition on the Rust side (`set_weights`,
`get_weights` and `get_weights_derivatives` all drive them), so mirroring the walk is
mirroring the contract; `tests/test_phase9_init.py` re-derives every offset independently, the
same one-hot-probe arithmetic the Rust `flat_layout_positions_are_the_spec_order` tests pin.

The NET-level assembly is shared by all three cells: recurrent stacks (twice for
`Direction bidirectional`, once for `forward` -- `weight_bridge.spec_directions`), then the
output MLP (`[weights row-major | bias]` per layer, the `NeuronLayer` layout), then the
`mean`/`std` normalize tail -- exactly `BlstmNetwork::set_weights`'s block order. The output
MLP's input width (`OutputNeuronNb[0]`: `2*hidden` bidirectional, `hidden` forward) is READ
from the spec, never derived here: `BlstmConfig::from_legacy` validates it engine-side, so a
mismatched config must fail there, loudly, rather than be silently papered over by the
initializer.

sLSTM init (spec S2.4): `b_f = +1` under `forget_bias_one` (block index 1 of `[i|f|o|z]`),
every other bias 0, `W`/`R` on the same combined-fan_in Xavier/He convention as the LSTM path.
NOTE `b_i`: the input-gate bias is STRUCTURALLY NON-IDENTIFIABLE -- shifting it scales the
sLSTM's `C` and `N` states equally, so `c/n` (and therefore the output) is invariant and
`dL/db_i` is EXACTLY zero (`nn/cells/slstm.rs`'s module doc proves it; the T2 FD tier scores
those weights on an absolute floor). It is seeded 0 and will never move in training, whatever
we put there.

Mamba init (spec S3.4, LOAD-BEARING and SPEC-PINNED -- these are the S4/Mamba-lineage values,
copied, not re-derived; a from-scratch run without them is known-fragile): `A_log[c,s] =
log(s+1)` (S4D-real, so `A = -exp(A_log) = -(s+1)`), `b_dt = softplus^-1(Delta_0)` with
`Delta_0` log-uniform in `[1e-3, 1e-1]` drawn PER CHANNEL, `W_dt` uniform `+-dt_rank^{-1/2}`,
`D = 1`, `g = 1`; the projections (`P`, `W_in`, `W_x`, `W_out`) and the depthwise conv follow
the repo Xavier/He convention, and the two structural biases (`p`, `b_conv`) are 0. The
depthwise conv's fans are `fan_in = fan_out = d_conv` (each output channel sees `d_conv` taps
of exactly one input channel -- the standard depthwise reading of
`kernel * in_channels/groups`). `softplus^-1(y) = log(e^y - 1)` is evaluated as the
algebraically identical `y + log1p(-e^{-y})` -- for DOMAIN SAFETY (the direct form overflows
`exp` above `y ~ 709.78`), not for accuracy on this band, where both agree to ~1e-13.

Draw ORDER for the new cells is the FLAT ORDER (block by block). There is no legacy trajectory
to match -- only same-seed reproducibility matters -- so the least surprising order wins.
"""

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from speech.weight_bridge import nnet_to_flat, spec_directions

_PEEPHOLE_SCALE = 0.1

# Spec S3.4: Delta_0 log-uniform in [1e-3, 1e-1], seeded per channel.
_DELTA_MIN = 1e-3
_DELTA_MAX = 1e-1

Scheme = Literal["xavier", "he"]
CellType = Literal["lstm", "slstm", "mamba"]


def _xavier_uniform(rng: np.random.Generator, shape: tuple[int, int], fan_in: int, fan_out: int) -> NDArray[np.float64]:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(rng.uniform(-limit, limit, size=shape), dtype=np.float64)


def _he_normal(rng: np.random.Generator, shape: tuple[int, int], fan_in: int) -> NDArray[np.float64]:
    std = math.sqrt(2.0 / fan_in)
    return np.asarray(rng.normal(0.0, std, size=shape), dtype=np.float64)


def _draw(rng: np.random.Generator, shape: tuple[int, int], fan_in: int, fan_out: int, scheme: Scheme) -> NDArray[np.float64]:
    if scheme == "xavier":
        return _xavier_uniform(rng, shape, fan_in, fan_out)
    return _he_normal(rng, shape, fan_in)


def _check_scheme(scheme: str) -> None:
    """Validated at every PUBLIC entry point, because `_draw`'s `else` branch is He: an
    unrecognized string would otherwise be silently honoured as "he" rather than rejected
    (`Scheme` is a `Literal`, so only a static checker catches it, and the per-cell builders
    are reachable from untyped callers)."""
    if scheme not in ("xavier", "he"):
        raise ValueError(f"unknown scheme: {scheme!r}")


def _init_lstm_gates(rng: np.random.Generator, out: int, fin: int, scheme: Scheme, forget_bias_one: bool) -> dict[str, np.ndarray]:
    """One direction's one LSTM layer: input/forget/output (width fin+out+5, peephole-bearing)
    + cell (width fin+out+1, no peepholes) -- the exact shapes `_pack_lstm_layer` reads."""
    ncols = fin + out + 5
    cell_cols = fin + out + 1
    combined_fan_in = fin + out  # recurrent + fan-in feed one summed preactivation -> one shared scale.

    gates: dict[str, np.ndarray] = {}
    for name in ("input", "forget", "output"):
        mat = np.zeros((out, ncols), dtype=np.float64)
        mat[:, 0:out] = _draw(rng, (out, out), combined_fan_in, out, scheme)
        mat[:, out : out + fin] = _draw(rng, (out, fin), combined_fan_in, out, scheme)
        mat[:, ncols - 5 : ncols - 2] = rng.uniform(-_PEEPHOLE_SCALE, _PEEPHOLE_SCALE, size=(out, 3))  # gate-peephole x3
        mat[:, ncols - 2] = rng.uniform(-_PEEPHOLE_SCALE, _PEEPHOLE_SCALE, size=out)  # recurrent-peephole
        gates[name] = mat
    if forget_bias_one:
        gates["forget"][:, ncols - 1] = 1.0  # all other biases (incl. forget when the flag is off) stay 0.0 from np.zeros.

    cell = np.zeros((out, cell_cols), dtype=np.float64)
    cell[:, 0:out] = _draw(rng, (out, out), combined_fan_in, out, scheme)
    cell[:, out : out + fin] = _draw(rng, (out, fin), combined_fan_in, out, scheme)
    gates["cell"] = cell
    return gates


def _init_output_layer(rng: np.random.Generator, out: int, inp: int, scheme: Scheme) -> NDArray[np.float64]:
    mat = np.zeros((out, inp + 1), dtype=np.float64)
    mat[:, :inp] = _draw(rng, (out, inp), inp, out, scheme)
    return mat  # bias (last column) stays 0.0.


@dataclass(frozen=True)
class MambaGeometry:
    """The four `Mamba_*` hyperparameters (spec S3.3/S6), defaults mirroring
    `blstm.rs::MambaParams::default`. ONE geometry per config -- a per-net override is an
    explicit non-goal this phase, which is why the Rust reads the keys UNPREFIXED."""

    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 0

    def resolve_dt_rank(self, d_model: int) -> int:
        """`dt_rank == 0` means auto: `ceil(d_model / 16)`, floored at 1 -- the resolution
        `MambaLayer::new` performs, mirrored here because the pack LENGTH depends on it."""
        return self.dt_rank if self.dt_rank else max(1, -(-d_model // 16))

    def d_inner(self, d_model: int) -> int:
        return self.expand * d_model


def mamba_geometry(spec: dict[str, object]) -> MambaGeometry:
    """The spec's `Mamba` entry (`config_bridge.nnet_spec`) as a typed geometry; absent means
    the S6 defaults, matching a config that never states the keys."""
    raw = cast(dict[str, int], spec.get("Mamba", {}))
    return MambaGeometry(**{k: int(v) for k, v in raw.items()})


def init_slstm_flat(
    rng: np.random.Generator,
    output_size: int,
    input_size: int,
    scheme: Scheme = "xavier",
    forget_bias_one: bool = True,
) -> NDArray[np.float64]:
    """ONE sLSTM cell layer's flat block, in the S2.2 order `for a in [i, f, o, z]:
    [R_a (out x out) | W_a (out x in) | b_a (out)]`, each matrix ROW-major -- the
    `nn/cells/slstm.rs::for_each_slot` walk, mirrored. Length `4*out*(out + in + 1)`.

    `R` and `W` share ONE combined `fan_in = in + out` (both halves feed a single summed
    pre-activation, exactly as for the legacy LSTM -- see the module docstring); `fan_out` is
    `out`. Biases are 0 except `b_f = 1` under `forget_bias_one` (spec S2.4). `b_i` stays 0
    and is non-identifiable (module docstring)."""
    _check_scheme(scheme)
    combined_fan_in = input_size + output_size
    parts: list[NDArray[np.float64]] = []
    for gate in range(4):  # [i | f | o | z]
        parts.append(_draw(rng, (output_size, output_size), combined_fan_in, output_size, scheme).reshape(-1))
        parts.append(_draw(rng, (output_size, input_size), combined_fan_in, output_size, scheme).reshape(-1))
        bias = np.zeros(output_size, dtype=np.float64)
        if gate == 1 and forget_bias_one:
            bias[:] = 1.0
        parts.append(bias)
    return np.concatenate(parts)


def _softplus_inverse(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """`log(e^y - 1)`, written as the algebraically identical `y + log1p(-e^{-y})`.

    The advantage is DOMAIN SAFETY, not accuracy on this band: the direct form overflows
    `exp` for `y > ~709.78` and returns `-inf`/`nan` where the true value is just `y`, while
    this one degrades gracefully (`e^{-y}` underflows harmlessly). On S3.4's actual
    `[1e-3, 1e-1]` draw both forms agree to ~1e-13, so nothing here rides on the choice --
    it simply removes a caveat, and mirrors the same guarded shape `mamba.rs::softplus` uses
    in the forward direction."""
    return cast(NDArray[np.float64], y + np.log1p(-np.exp(-y)))


def init_mamba_flat(
    rng: np.random.Generator,
    output_size: int,
    input_size: int,
    geom: MambaGeometry,
    scheme: Scheme = "xavier",
) -> NDArray[np.float64]:
    """ONE Mamba cell layer's flat block, in the S3.2 order -- the
    `nn/cells/mamba.rs::for_each_slot` walk, mirrored block for block:

        [P | p] (only when in != out), g, W_in, conv | b_conv, W_x, W_dt | b_dt, A_log, D, W_out

    each matrix ROW-major at its MATH shape (`P` is `out x in`, `W_in` is `2 d_inner x
    d_model`, `W_x` is `(dt_rank + 2 d_state) x d_inner`, `W_dt` is `d_inner x dt_rank`,
    `A_log` is `d_inner x d_state`, `W_out` is `d_model x d_inner`) -- the Rust stores most
    of them TRANSPOSED, which the walk absorbs, so the flat order is the math order.

    The S3.4 constants are spec text (see the module docstring): copied, not re-derived."""
    _check_scheme(scheme)
    d_model = output_size
    d_inner = geom.d_inner(d_model)
    d_state, d_conv = geom.d_state, geom.d_conv
    dt_rank = geom.resolve_dt_rank(d_model)
    parts: list[NDArray[np.float64]] = []

    if input_size != d_model:  # [P | p], the width adapter
        parts.append(_draw(rng, (d_model, input_size), input_size, d_model, scheme).reshape(-1))
        parts.append(np.zeros(d_model, dtype=np.float64))
    parts.append(np.ones(d_model, dtype=np.float64))  # g (RMSNorm gain) = 1
    parts.append(_draw(rng, (2 * d_inner, d_model), d_model, 2 * d_inner, scheme).reshape(-1))  # W_in
    # Depthwise conv: each output channel sees d_conv taps of ONE input channel.
    parts.append(_draw(rng, (d_inner, d_conv), d_conv, d_conv, scheme).reshape(-1))
    parts.append(np.zeros(d_inner, dtype=np.float64))  # b_conv
    parts.append(_draw(rng, (dt_rank + 2 * d_state, d_inner), d_inner, dt_rank + 2 * d_state, scheme).reshape(-1))  # W_x
    limit = dt_rank**-0.5
    parts.append(np.asarray(rng.uniform(-limit, limit, size=(d_inner, dt_rank)), dtype=np.float64).reshape(-1))  # W_dt
    delta0 = np.exp(rng.uniform(math.log(_DELTA_MIN), math.log(_DELTA_MAX), size=d_inner))  # log-uniform per channel
    parts.append(_softplus_inverse(np.asarray(delta0, dtype=np.float64)))  # b_dt
    a_log = np.log(np.arange(1, d_state + 1, dtype=np.float64))  # A_log[c, s] = log(s + 1)
    parts.append(np.tile(a_log, (d_inner, 1)).reshape(-1))
    parts.append(np.ones(d_inner, dtype=np.float64))  # D = 1
    parts.append(_draw(rng, (d_model, d_inner), d_inner, d_model, scheme).reshape(-1))  # W_out
    return np.concatenate(parts)


def _init_lstm_pack(
    spec: dict[str, object],
    rng: np.random.Generator,
    scheme: Scheme,
    forget_bias_one: bool,
    directions: tuple[str, ...],
) -> NDArray[np.float64]:
    """The legacy path, BYTE-UNTOUCHED at `directions == ("forward", "backward")`: build the
    packer-shaped nnet dict and let `nnet_to_flat` flatten it (zero layout reimplementation).
    A causal spec simply leaves `nnet["backward"]` empty -- `nnet_to_flat` walks the lists it
    is given, so no packer branch is needed."""
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    outn = cast(list[int], spec["OutputNeuronNb"])

    nnet: dict[str, object] = {"forward": [], "backward": [], "output": []}
    for direction in directions:
        layers = cast(list[dict[str, np.ndarray]], nnet[direction])
        for i in range(len(lstm) - 1):
            out, fin = lstm[i + 1], lstm[i] * lsub[i]
            layers.append(_init_lstm_gates(rng, out, fin, scheme, forget_bias_one))

    out_layers = cast(list[np.ndarray], nnet["output"])
    for i in range(len(outn) - 1):
        out_layers.append(_init_output_layer(rng, outn[i + 1], outn[i], scheme))

    nnet["mean"] = np.zeros(lstm[0], dtype=np.float64)
    nnet["std"] = np.ones(lstm[0], dtype=np.float64)
    return np.asarray(nnet_to_flat(nnet, spec), dtype=np.float64)


def _init_cell_pack(
    spec: dict[str, object],
    rng: np.random.Generator,
    scheme: Scheme,
    forget_bias_one: bool,
    directions: tuple[str, ...],
    cell_type: CellType,
) -> NDArray[np.float64]:
    """The sLSTM/Mamba path: emit the flat blocks directly (spec S1.3 -- these cells have no
    structured domain), then the SAME output-MLP + normalize-tail blocks the LSTM pack ends
    with, in `BlstmNetwork::set_weights`'s order."""
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    outn = cast(list[int], spec["OutputNeuronNb"])
    geom = mamba_geometry(spec)

    parts: list[NDArray[np.float64]] = []
    for _ in directions:
        for i in range(len(lstm) - 1):
            out, fin = lstm[i + 1], lstm[i] * lsub[i]
            if cell_type == "slstm":
                parts.append(init_slstm_flat(rng, out, fin, scheme, forget_bias_one))
            else:
                parts.append(init_mamba_flat(rng, out, fin, geom, scheme))
    for i in range(len(outn) - 1):
        mat = _init_output_layer(rng, outn[i + 1], outn[i], scheme)
        parts.append(mat[:, :-1].reshape(-1))
        parts.append(mat[:, -1].reshape(-1))
    parts.append(np.zeros(lstm[0], dtype=np.float64))
    parts.append(np.ones(lstm[0], dtype=np.float64))
    return np.concatenate(parts)


def init_weights(
    spec: dict[str, object],
    rng: np.random.Generator,
    scheme: Scheme = "xavier",
    forget_bias_one: bool = True,
) -> list[NDArray[np.float64]]:
    """Seeded init for one net's flat pack, shaped per `spec` (see module docstring).

    The CELL TYPE and DIRECTION come from the spec itself (`config_bridge.nnet_spec` reads
    `{prefix}_Cell_Type` / `{prefix}_Direction` / `Mamba_*`), so every caller learns the
    architecture from the same config the engine does. Absent entries mean the legacy shape
    and this function is byte-for-byte what it was before phase 9."""
    _check_scheme(scheme)
    cell_type = cast(str, spec.get("CellType", "lstm"))
    if cell_type not in ("lstm", "slstm", "mamba"):
        raise ValueError(f"unknown cell type: {cell_type!r} (expected 'lstm', 'slstm' or 'mamba')")
    directions = spec_directions(spec)

    if cell_type == "lstm":
        return [_init_lstm_pack(spec, rng, scheme, forget_bias_one, directions)]
    return [_init_cell_pack(spec, rng, scheme, forget_bias_one, directions, cast(CellType, cell_type))]
