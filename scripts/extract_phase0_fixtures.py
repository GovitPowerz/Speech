"""Extract Phase-0a golden fixtures from the legacy save_net .mat (scipy).

Loads BestConfigStruct (structured config-domain weight rows) and emits committed
fixtures that let CI re-check the adim + pack pipeline without scipy/the .mat:
  tests/reference_data/phase0/best_config_domain.bin   (1 x N config-domain rows, concatenated)
  tests/reference_data/phase0/best_config_manifest.json (spec + row names/lengths)
  tests/reference_data/phase0/best_net_flat.bin         (nnet-domain flat pack, N x 1)

IMPORTANT (pairing correction, verified 2026-07-01):
  The brief assumed `nnet_best` in this .mat is the nnet-domain flat pack paired with
  BestConfigStruct, with 33671 real elems + zero padding. That is FALSE for this file.
  `nnet_best` is the CMA-ES optimizer genome (column 22 of `nnet_in`, a 50000 x 25
  population), dense (49987/50000 nonzero). Its head decodes to SCALED config
  hyperparameters, not weights: nnet_best[0]=7.5181=decision_thresh_rising*10,
  nnet_best[1]=4.7726=decision_area_rising*1000, nnet_best[4]=4=LSTMSubSampling[0], etc.
  No offset (raw, adim'd, or scale-invariant) of pack(config_to_nnet(BestConfigStruct))
  occurs anywhere in nnet_best. The two artifacts are not a (structured, flat) pair, so
  the brief's assertion `pack(config_to_nnet(structured)) == nnet_best[:33671]` cannot hold.

  The packer itself is verified correct against the legacy MATLAB ground truth
  (config2weights.m / weights2nnet.m / config2network.m) and against a bijection
  round-trip on the real Task-1 flat vector NNweights_config1.bin. This script therefore
  emits self-consistent fixtures (best_net_flat.bin = pack(config_to_nnet(structured)))
  and asserts the pipeline's structural invariants (adim invertibility + flat<->nnet
  bijection) rather than a false equality against nnet_best.

Usage: uv run --with scipy python scripts/extract_phase0_fixtures.py
Requires the local git-ignored .mat.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from speech import weight_bridge

MAT = Path("/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/save_net/LSTM_15-Oct-2015_BLSTM_OpenSAD15.mat")
OUT = Path("tests/reference_data/phase0")
PREFIX = "AlgName"  # BestConfigStruct uses the literal AlgName_ prefix


def field(bcs: object, name: str) -> np.ndarray:
    return np.atleast_1d(np.asarray(getattr(bcs, name), dtype=np.float64)).reshape(-1)


def main() -> None:
    m = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)
    bcs = m["BestConfigStruct"]

    p = f"{PREFIX}_"
    lstm = [int(x) for x in np.atleast_1d(getattr(bcs, f"{p}LSTMNeuronNb")).reshape(-1)]
    lsub = [int(x) for x in np.atleast_1d(getattr(bcs, f"{p}LSTMSubSampling")).reshape(-1)]
    outn = [int(x) for x in np.atleast_1d(getattr(bcs, f"{p}OutputNeuronNb")).reshape(-1)]
    osub = [int(x) for x in np.atleast_1d(getattr(bcs, f"{p}OutputSubSampling")).reshape(-1)]
    spec = {"LSTMNeuronNb": lstm, "LSTMSubSampling": lsub, "OutputNeuronNb": outn, "OutputSubSampling": osub, "NNetInputSize": lstm[0]}

    n = weight_bridge.element_count(spec)

    # Assemble structured config-domain rows + record the manifest.
    rows: list[dict] = []
    blob: list[np.ndarray] = []

    def emit(name: str, vec: np.ndarray) -> None:
        rows.append({"name": name, "len": int(vec.shape[0])})
        blob.append(vec)

    structured: dict = {"forward": [], "backward": [], "output": []}
    for direction in ("Forward", "Backward"):
        key = direction.lower()
        for i in range(len(lstm) - 1):
            out = lstm[i + 1]
            gates: dict = {}
            for gate, gk in (("input", "InputGateWeights"), ("forget", "ForgetGateWeights"), ("output", "OutputGateWeights"), ("cell", "CellWeight")):
                mat_rows = [field(bcs, f"{p}{direction}_Layer_{i}_LSTMBlock_{b}_{gk}") for b in range(out)]
                mat = np.stack(mat_rows, axis=0)
                gates[gate] = mat
                for b in range(out):
                    emit(f"{key}.L{i}.B{b}.{gate}", mat_rows[b])
            structured[key].append(gates)
    for i in range(len(outn) - 1):
        neurons = [field(bcs, f"{p}Output_Layer_{i}_Neuron_{nn}_Weights") for nn in range(outn[i + 1])]
        mat = np.stack(neurons, axis=0)
        structured["output"].append(mat)
        for nn in range(outn[i + 1]):
            emit(f"output.L{i}.N{nn}", neurons[nn])
    structured["mean"] = field(bcs, f"{p}NormalizeInputMean")
    structured["std"] = field(bcs, f"{p}NormalizeInputStd")
    emit("mean", structured["mean"])
    emit("std", structured["std"])

    # GROUND TRUTH (structural, since nnet_best is not the paired weight pack - see module docstring):
    # (1) pack length matches element_count; (2) flat<->nnet is a bijection; (3) config<->nnet adim
    #     leaves the bias column identical and rescales the rest exactly (invertible).
    nnet = weight_bridge.config_to_nnet(structured, spec)
    flat = weight_bridge.nnet_to_flat(nnet, spec)
    assert flat.shape[0] == n, f"packed length {flat.shape[0]} != {n}"
    rt = weight_bridge.nnet_to_flat(weight_bridge.flat_to_nnet(flat, spec), spec)
    assert np.array_equal(rt, flat), "BIJECTION BROKEN: flat -> nnet -> flat is not identity"
    for direction in ("forward", "backward"):
        for i, layer in enumerate(structured[direction]):
            adim = weight_bridge._lstm_adim(spec, i)
            for g in ("input", "forget", "output", "cell"):
                cfg_row, nn_row = layer[g], nnet[direction][i][g]
                assert np.array_equal(cfg_row[:, -1], nn_row[:, -1]), f"adim touched bias {direction}.L{i}.{g}"
                assert np.allclose(nn_row[:, :-1] * adim, cfg_row[:, :-1]), f"adim rescale wrong {direction}.L{i}.{g}"

    OUT.mkdir(parents=True, exist_ok=True)
    concat = np.concatenate(blob)
    weight_bridge.write_bin(1, int(concat.shape[0]), concat, OUT / "best_config_domain.bin")
    weight_bridge.write_bin(n, 1, flat, OUT / "best_net_flat.bin")
    (OUT / "best_config_manifest.json").write_text(json.dumps({"spec": spec, "rows": rows}, indent=1))
    print(f"OK: packed {n} elems; bijection + adim invariants hold; wrote fixtures to {OUT}")
    print("NOTE: best_net_flat.bin is pack(config_to_nnet(BestConfigStruct)); nnet_best is the")
    print("      optimizer genome, not the paired weight pack (see module docstring).")


if __name__ == "__main__":
    main()
