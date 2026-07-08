"""Neural-net weight (de)serialization: canonical flat vector <-> `.bin`.

Ported from legacy MATLAB: config2network*.m, network2config*.m,
config2weights*.m, nnet2MatFile.m, weights2nnet.m, Write/ReadMatrixFromBinary.m,
CombineNNets.m, ModifyOutputNetwork.m, ForceLSTMBiais.m. See design spec section 6.

High-risk: the column-major flat layout + adim_coeff scaling are the seam to the
Rust engine - round-trip property tests are mandatory when implemented (Phase 3).
"""

import json
import math
import struct
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray


def _spec_from_net(net: dict[str, object]) -> dict[str, object]:
    """Derive the minimal LSTM-shape spec `nnet_to_flat` needs, straight off `net`'s own matrices.

    `net["forward"]`/`net["backward"]` layers are self-describing 2D arrays (out x ncols), so
    unlike `flat_to_nnet` (which slices a shapeless flat vector and MUST be told the spec),
    `nnet_to_flat` only ever reads `spec["LSTMNeuronNb"][i+1]` (= layer i's row count "out") and
    the product `spec["LSTMNeuronNb"][i] * spec["LSTMSubSampling"][i]` (= layer i's fan-in
    "fin", from the Cell matrix's narrower `out+fin+1` column count - see `_pack_lstm_layer`).
    `LSTMNeuronNb[i+1]` for i>0 is shared with layer i-1's fan-in product, so it is NOT a free
    choice: it is pinned to layer i-1's own "out". `LSTMNeuronNb[0]` has no such constraint (no
    layer -1), so it is set to layer 0's fan-in directly with `LSTMSubSampling[0] = 1`.
    Output layers and mean/std need no spec at all (`nnet_to_flat` reads their shapes as-is).
    """
    forward = cast(list[dict[str, np.ndarray]], net["forward"])
    outs = [int(np.asarray(layer["cell"], dtype=np.float64).shape[0]) for layer in forward]
    fins = [int(np.asarray(layer["cell"], dtype=np.float64).shape[1]) - out - 1 for out, layer in zip(outs, forward, strict=True)]
    lstm_neuron_nb = [fins[0], *outs]
    lstm_subsampling = [1, *(fins[i] // lstm_neuron_nb[i] for i in range(1, len(outs)))]
    return {"LSTMNeuronNb": lstm_neuron_nb, "LSTMSubSampling": lstm_subsampling}


def pack_weights(net: dict[str, object]) -> NDArray[np.float64]:
    """Flatten a network struct into the canonical row-major flat vector.

    `net` is nnet-domain (`config_to_nnet`'s output shape: `{"forward"/"backward": [layer ->
    {"input"/"forget"/"output"/"cell": 2D}], "output": [layer -> 2D], "mean"/"std": 1D}` - see
    the module-level "Structured/Nnet" comment above `element_count`). This is a thin wrapper
    over `nnet_to_flat` (the Rust twin: `config.rs::nnet_to_flat`): it needs no separate `arch`
    because `net`'s matrices already carry their own shape, so `_spec_from_net` recovers the
    handful of scalars `nnet_to_flat` actually reads.
    """
    return np.asarray(nnet_to_flat(net, _spec_from_net(net)), dtype=np.float64)


def unpack_weights(flat: NDArray[np.float64], arch: dict[str, object]) -> dict[str, object]:
    """Inverse of `pack_weights`.

    `flat` is a shapeless vector, so - unlike `pack_weights` - the caller MUST supply `arch`: an
    `NnetSpec`-shaped dict (as produced by `config_bridge.nnet_spec`: `LSTMNeuronNb`,
    `LSTMSubSampling`, `OutputNeuronNb`, `OutputSubSampling`, `NNetInputSize`). Thin wrapper over
    `flat_to_nnet` (the Rust twin: `config.rs::flat_to_nnet`), which is where the real slicing
    logic lives.
    """
    return flat_to_nnet(flat, arch)


def read_bin(path: Path) -> tuple[int, int, NDArray[np.float64]]:
    """Read the custom .bin matrix; returns (rows, cols, column-major f64 vector)."""
    raw = Path(path).read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    if rows * cols <= 0:
        raise ValueError(f"empty/invalid .bin: rows={rows} cols={cols}")
    data = np.frombuffer(raw[16 : 16 + 8 * rows * cols], dtype="<f8")
    if data.shape[0] != rows * cols:
        raise ValueError("truncated .bin payload")
    return rows, cols, np.array(data, dtype=np.float64)


def write_bin(rows: int, cols: int, data: NDArray[np.float64], path: Path) -> None:
    """Write rows/cols as i64 LE then column-major f64 LE (data is already column-major)."""
    flat = np.ascontiguousarray(data, dtype="<f8").reshape(-1)
    if flat.shape[0] != rows * cols:
        raise ValueError("data length does not match rows*cols")
    with Path(path).open("wb") as fh:
        fh.write(struct.pack("<qq", rows, cols))
        fh.write(flat.tobytes())


def read_weight_vector(path: Path) -> NDArray[np.float64]:
    """Read an N x 1 weight .bin as a flat vector (asserts cols == 1)."""
    rows, cols, data = read_bin(path)
    if cols != 1:
        raise ValueError(f"expected a column vector, got cols={cols}")
    return data


# A "row" is a config-domain gate/neuron vector of length ncols.
# Structured = {"forward": [layer -> {"input"/"forget"/"output"/"cell": 2D (out x ncols)}],
#               "backward": [...same...],
#               "output": [layer -> 2D (out x in+1)],
#               "mean": 1D, "std": 1D}
# Nnet has the same shape but nnet-domain (adim applied) matrices.


def element_count(spec: dict) -> int:
    lstm = spec["LSTMNeuronNb"]
    lsub = spec["LSTMSubSampling"]
    outn = spec["OutputNeuronNb"]
    total = 0
    for i in range(len(lstm) - 1):
        out = lstm[i + 1]
        fin = lstm[i] * lsub[i]
        total += 2 * (4 * out * fin + 4 * out * out + 12 * out + 4 * out)
    for i in range(len(outn) - 1):
        total += outn[i + 1] * outn[i] + outn[i + 1]
    total += 2 * lstm[0]
    return int(total)


def _lstm_adim(spec: dict, i: int) -> float:
    return math.sqrt(spec["LSTMNeuronNb"][i] * spec["LSTMSubSampling"][i] + spec["LSTMNeuronNb"][i + 1])


def _out_adim(spec: dict, i: int) -> float:
    return math.sqrt(spec["OutputNeuronNb"][i] * spec["OutputSubSampling"][i])


def _apply_adim(mat: np.ndarray, adim: float) -> np.ndarray:
    # divide every column by adim EXCEPT the last (bias) column.
    out = mat / adim
    # Bias restored to its ORIGINAL value here. Legacy MATLAB config2network.m instead
    # recomputes it as (config/adim)*adim, which can differ from the original by up to
    # 1 ULP on ~34 biases. Harmless for Phase 0a (Rust and Python agree exactly); flagged
    # for the Phase-2 inference-vs-legacy golden.
    out[:, -1] = mat[:, -1]
    return out


def config_to_nnet(structured: dict, spec: dict) -> dict:
    """Config-domain structured rows -> nnet-domain matrices (adim decode)."""
    nnet: dict = {"forward": [], "backward": [], "output": [], "mean": structured["mean"], "std": structured["std"]}
    for direction in ("forward", "backward"):
        for i, layer in enumerate(structured[direction]):
            adim = _lstm_adim(spec, i)
            nnet[direction].append({g: _apply_adim(layer[g], adim) for g in ("input", "forget", "output", "cell")})
    for i, layer in enumerate(structured["output"]):
        nnet["output"].append(_apply_adim(layer, _out_adim(spec, i)))
    return nnet


def _pack_lstm_layer(gates: dict, out: int, input_size: int) -> list[np.ndarray]:
    # I/F/O gate rows are [recurrent(out) | fan-in(input_size) | gate-peephole(3) | recurrent-peephole(1) | bias(1)]
    # (width = out + input_size + 5). CellWeight rows have no peepholes: [recurrent | fan-in | bias] (width out + input_size + 1).
    # Fan-in and recurrent column ranges are identical across gates; only per-gate last column is the bias.
    ncols = input_size + out + 5  # I/F/O frame, used for the peephole indices only
    idx_in = list(range(out, out + input_size))
    idx_rec = list(range(0, out))
    iw, fw, ow, cw = gates["input"], gates["forget"], gates["output"], gates["cell"]
    parts: list = []
    # 1-4 fan-in blocks, row-major
    for m in (iw, fw, ow, cw):
        parts.append(m[:, idx_in].reshape(-1))
    # 5-8 recurrent blocks
    for m in (iw, fw, ow, cw):
        parts.append(m[:, idx_rec].reshape(-1))
    # 9 peephole bundle (out x 12), row-major; only I/F/O carry peepholes
    peep = np.empty((out, 12), dtype=np.float64)
    for r in range(out):
        peep[r] = [
            iw[r, ncols - 2],
            fw[r, ncols - 2],
            ow[r, ncols - 2],
            iw[r, ncols - 5],
            iw[r, ncols - 4],
            iw[r, ncols - 3],
            fw[r, ncols - 5],
            fw[r, ncols - 4],
            fw[r, ncols - 3],
            ow[r, ncols - 5],
            ow[r, ncols - 4],
            ow[r, ncols - 3],
        ]
    parts.append(peep.reshape(-1))
    # 10-13 biases (each gate's own last column; cell is narrower)
    for m in (iw, fw, ow, cw):
        parts.append(m[:, -1].reshape(-1))
    return parts


def nnet_to_flat(nnet: dict, spec: dict) -> np.ndarray:
    lstm = spec["LSTMNeuronNb"]
    parts: list = []
    for direction in ("forward", "backward"):
        for i, layer in enumerate(nnet[direction]):
            parts.extend(_pack_lstm_layer(layer, out=lstm[i + 1], input_size=lstm[i] * spec["LSTMSubSampling"][i]))
    for layer in nnet["output"]:
        parts.append(layer[:, :-1].reshape(-1))  # weights
        parts.append(layer[:, -1].reshape(-1))  # bias
    parts.append(np.asarray(nnet["mean"], dtype=np.float64).reshape(-1))
    parts.append(np.asarray(nnet["std"], dtype=np.float64).reshape(-1))
    return np.concatenate(parts)


def load_structured(manifest_path: Path, bin_path: Path) -> tuple[dict, dict]:
    """Load (structured, spec) from the committed manifest + concatenated-rows .bin."""
    manifest = json.loads(Path(manifest_path).read_text())
    spec = manifest["spec"]
    _, _, blob = read_bin(bin_path)
    named: dict[str, np.ndarray] = {}
    off = 0
    for r in manifest["rows"]:
        named[r["name"]] = blob[off : off + r["len"]]
        off += r["len"]
    lstm, outn = spec["LSTMNeuronNb"], spec["OutputNeuronNb"]
    structured: dict = {"forward": [], "backward": [], "output": [], "mean": named["mean"], "std": named["std"]}
    for direction in ("forward", "backward"):
        for i in range(len(lstm) - 1):
            out = lstm[i + 1]
            structured[direction].append(
                {g: np.stack([named[f"{direction}.L{i}.B{b}.{g}"] for b in range(out)], 0) for g in ("input", "forget", "output", "cell")}
            )
    for i in range(len(outn) - 1):
        structured["output"].append(np.stack([named[f"output.L{i}.N{nn}"] for nn in range(outn[i + 1])], 0))
    return structured, spec


def flat_to_nnet(flat: np.ndarray, spec: dict) -> dict:
    """Inverse of nnet_to_flat: slice the flat vector back into nnet-domain matrices."""
    lstm, lsub, outn = spec["LSTMNeuronNb"], spec["LSTMSubSampling"], spec["OutputNeuronNb"]
    pos = 0
    nnet: dict = {"forward": [], "backward": [], "output": []}

    def take(k: int) -> np.ndarray:
        nonlocal pos
        seg = flat[pos : pos + k]
        pos += k
        return seg

    for direction in ("forward", "backward"):
        for i in range(len(lstm) - 1):
            out, fin = lstm[i + 1], lstm[i] * lsub[i]
            ncols = fin + out + 5  # I/F/O gate frame; the cell matrix is narrower (fin+out+1, no peepholes).
            cell_cols = fin + out + 1
            gates = {g: np.zeros((out, ncols)) for g in ("input", "forget", "output")}
            gates["cell"] = np.zeros((out, cell_cols))
            for g in ("input", "forget", "output", "cell"):
                gates[g][:, out : out + fin] = take(out * fin).reshape(out, fin)
            for g in ("input", "forget", "output", "cell"):
                gates[g][:, 0:out] = take(out * out).reshape(out, out)
            peep = take(12 * out).reshape(out, 12)
            for r in range(out):
                gates["input"][r, ncols - 2], gates["forget"][r, ncols - 2], gates["output"][r, ncols - 2] = peep[r, 0], peep[r, 1], peep[r, 2]
                gates["input"][r, ncols - 5 : ncols - 2] = peep[r, 3:6]
                gates["forget"][r, ncols - 5 : ncols - 2] = peep[r, 6:9]
                gates["output"][r, ncols - 5 : ncols - 2] = peep[r, 9:12]
            for g in ("input", "forget", "output"):
                gates[g][:, ncols - 1] = take(out)
            gates["cell"][:, cell_cols - 1] = take(out)
            nnet[direction].append(gates)
    for i in range(len(outn) - 1):
        out, inp = outn[i + 1], outn[i]
        mat = np.zeros((out, inp + 1))
        mat[:, :inp] = take(out * inp).reshape(out, inp)
        mat[:, inp] = take(out)
        nnet["output"].append(mat)
    nnet["mean"] = take(lstm[0])
    nnet["std"] = take(lstm[0])
    return nnet
