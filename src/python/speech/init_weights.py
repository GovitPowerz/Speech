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
"""

import math
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from speech.weight_bridge import nnet_to_flat

_PEEPHOLE_SCALE = 0.1

Scheme = Literal["xavier", "he"]


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


def init_weights(
    spec: dict[str, object],
    rng: np.random.Generator,
    scheme: Scheme = "xavier",
    forget_bias_one: bool = True,
) -> list[NDArray[np.float64]]:
    """Seeded init for one net's flat pack, shaped per `spec` (see module docstring)."""
    if scheme not in ("xavier", "he"):
        raise ValueError(f"unknown scheme: {scheme!r}")

    lstm = cast(list[int], spec["LSTMNeuronNb"])
    lsub = cast(list[int], spec["LSTMSubSampling"])
    outn = cast(list[int], spec["OutputNeuronNb"])

    nnet: dict[str, object] = {"forward": [], "backward": [], "output": []}
    for direction in ("forward", "backward"):
        layers = cast(list[dict[str, np.ndarray]], nnet[direction])
        for i in range(len(lstm) - 1):
            out, fin = lstm[i + 1], lstm[i] * lsub[i]
            layers.append(_init_lstm_gates(rng, out, fin, scheme, forget_bias_one))

    out_layers = cast(list[np.ndarray], nnet["output"])
    for i in range(len(outn) - 1):
        out_layers.append(_init_output_layer(rng, outn[i + 1], outn[i], scheme))

    nnet["mean"] = np.zeros(lstm[0], dtype=np.float64)
    nnet["std"] = np.ones(lstm[0], dtype=np.float64)

    return [np.asarray(nnet_to_flat(nnet, spec), dtype=np.float64)]
