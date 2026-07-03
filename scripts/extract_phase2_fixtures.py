"""Build and run the oracle harness, then regenerate the Phase 2 NN-probe manifest.

Rebuilds ``tools/oracle_harness`` (now linking the vendored legacy NN stack:
``BLSTMNeuralNetwork``/``LSTMLayer``/``NeuronLayer`` + the link-only
``SRNLayer``/``CWRNNLayer``/``CostLaw``/``Rprop``), runs it, and parses the Phase 2
``NN_REAL`` + ``NN_PROBE`` stdout lines into ``tests/reference_data/phase2/
manifest.json``. Those lines record (a) the real ``BLSTMNeuralNetwork<LSTMLayer>``
built from the vendored config + weights with ``getNbOfWeights() == 33671``, and
(b) the bit-for-bit product-order divergence between Eigen's blocked products and
the ascending ``matSeq`` accumulation at four NN sites -- the measurement Phase 2's
NN forward port depends on (extends the Phase 1 ``dct_gemm_substitution`` probe).

The harness reads five input fixtures from its output dir (``excerpt_2ch_8k.wav``
plus four ``variant_*.config``); those live in ``tests/reference_data/phase1/``.
To avoid mutating the committed Phase 1 dir, we run the harness in a throwaway temp
dir seeded with copies of those inputs, keep the parsed probe results AND the Task 2
``act_sweep.bin`` dump (copied out into the Phase 2 dir), and write the manifest to
the Phase 2 dir.

Usage: uv run python scripts/extract_phase2_fixtures.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from speech.weight_bridge import read_bin

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "tools" / "oracle_harness"
PHASE1_DIR = REPO_ROOT / "tests" / "reference_data" / "phase1"
PHASE2_DIR = REPO_ROOT / "tests" / "reference_data" / "phase2"
PHASE0_DIR = REPO_ROOT / "tests" / "reference_data" / "phase0"

NN_CONFIG = PHASE0_DIR / "1_worker_1.config"
NN_WEIGHTS = PHASE0_DIR / "NNweights_config1.bin"

# Dumps the harness produces into its output dir, with their expected (rows, cols).
# Task 2: act_sweep.bin (4 x N: row0 inputs, row1 GatesFunction, row2 Logistic,
# row3 Maxmin2/asinh). N = len(xs) in main.cpp's Task 2 sweep block (21 probes).
# Task 3: lstm_w_roundtrip_{in,out}.bin, one column vector each, length
# nb0 + nb1 where nb0 = LstmLayer(I=3,O=2).nb_of_weights() = 72 and
# nb1 = LstmLayer(I=2,O=1).nb_of_weights() = 28 -> 100 rows, 1 col.
EXPECTED_SHAPES = {
    "act_sweep.bin": (4, 21),
    "lstm_w_roundtrip_in.bin": (100, 1),
    "lstm_w_roundtrip_out.bin": (100, 1),
}

# Task 4: LSTM forward/reverse dumps. Grid = {I=3,O=2,T=7; I=5,O=4,T=12} x
# {allon, cells, gates, gatesrec, alloff} x {fwd, rev} + two width-mismatch cases
# (3x2_T7_allon, cols I+2 / I-1). lstm_fwd_<case>.bin is (T x O), lstm_gates_<case>.bin
# is (T x 4O). Kept in sync with the runCase grid in main.cpp's Task 4 block.
_LSTM_SHAPES = [(3, 2, 7), (5, 4, 12)]
_LSTM_FLAGS = ["allon", "cells", "gates", "gatesrec", "alloff"]
for _i, _o, _t in _LSTM_SHAPES:
    for _fl in _LSTM_FLAGS:
        for _dir in ("fwd", "rev"):
            _name = f"{_i}x{_o}_T{_t}_{_fl}_{_dir}"
            EXPECTED_SHAPES[f"lstm_fwd_{_name}.bin"] = (_t, _o)
            EXPECTED_SHAPES[f"lstm_gates_{_name}.bin"] = (_t, 4 * _o)
for _name in ("3x2_T7_allon_wideP2_fwd", "3x2_T7_allon_narrowM1_fwd"):
    EXPECTED_SHAPES[f"lstm_fwd_{_name}.bin"] = (7, 2)
    EXPECTED_SHAPES[f"lstm_gates_{_name}.bin"] = (7, 8)

# Task 5: NeuronLayer (dense) forward dumps. hidden/softmax/wide/narrow are I=4,O=3,
# T=6 (softmax lastLayer=true O=3; hidden/wide/narrow lastLayer=false, asinh); wide
# uses input cols I+2 (leftCols), narrow uses cols I-1 (topRows). logistic is
# I=4,O=1,T=6 (lastLayer=true O=1). Kept in sync with the runCase grid in main.cpp's
# Task 5 block.
EXPECTED_SHAPES["dense_hidden.bin"] = (6, 3)
EXPECTED_SHAPES["dense_softmax.bin"] = (6, 3)
EXPECTED_SHAPES["dense_logistic.bin"] = (6, 1)
EXPECTED_SHAPES["dense_wide.bin"] = (6, 3)
EXPECTED_SHAPES["dense_narrow.bin"] = (6, 3)

# Task 6: NeuralNetwork container dumps. LSTM net [3,4,2] sub [2,1] on T=11: after
# SubSample(2) the input drops to floor(11/2)=5 rows; final layer output is 5 x 2
# (forward AND reverse). Dense net [4,3,2] sub [1,2] on T=11: SubSample(2) between
# layers -> 5 x 2 (softmax O=2). feedForwardDouble net [10,4,2] sub [1,1] from two
# 5-col halves on T=11 -> 11 x 2. Chained set_weights round-trip: nbTotal x 1 (LSTM
# net getNbOfWeights() = 224 + 80 = 304). Kept in sync with main.cpp's Task 6 block.
_NET_LSTM_NBTOTAL = 304
EXPECTED_SHAPES["net_lstm_fwd.bin"] = (5, 2)
EXPECTED_SHAPES["net_lstm_rev.bin"] = (5, 2)
EXPECTED_SHAPES["net_dense.bin"] = (5, 2)
EXPECTED_SHAPES["net_double.bin"] = (11, 2)
EXPECTED_SHAPES["net_w_in.bin"] = (_NET_LSTM_NBTOTAL, 1)
EXPECTED_SHAPES["net_w_out.bin"] = (_NET_LSTM_NBTOTAL, 1)

# Input fixtures the harness reads from its output dir (see main.cpp:577,1233).
HARNESS_INPUTS = [
    "excerpt_2ch_8k.wav",
    "variant_mfcc_deltas.config",
    "variant_mfcc_sdc.config",
    "variant_logmel.config",
    "variant_rawband_ltsv.config",
]

# The NN weight count the harness aborts on if wrong (BLSTMNeuralNetwork ctor path).
EXPECTED_NB_WEIGHTS = 33671

# The four product sites the harness probes (spec S8). Shapes are documented here for
# the manifest; the actual diverged/mismatch numbers are MEASURED fresh every run.
PROBE_SITES = {
    "lstm_input_gemm": "(T x 23) * (23 x 96): forward LSTM layer-0 input weights (top 23 rows) on a 200-row deterministic input.",
    "lstm_recurrence_gemv": "(1 x 24) * (24 x 96) recurrent GEMV (LSTMLayer.cpp:351), accumulated over 150 timesteps of a bounded mock recurrence.",
    "dense_gemm": "(200 x 48) * (48 x 12): output-MLP first NeuronLayer weights (NeuronLayer.cpp:130) on a deterministic input.",
    "softmax_rowsum": "row-sum of exp(dense output) (NeuronLayer.cpp:138-139): Eigen rowwise().sum() vs ascending matSeq against a ones vector.",
}

PROBE_TEXT = (
    "Eigen blocked product vs the explicit ascending-loop accumulation (matSeq in "
    "main.cpp) at the NN product sites; measured fresh by the harness's NN_PROBE on "
    "every regeneration, not hardcoded. Extends the Phase 1 dct_gemm_substitution "
    "measurement to the LSTM/dense/softmax shapes the NN forward port will use."
)

NN_REAL_RE = re.compile(r"^NN_REAL ok weights=(?P<weights>\d+)$", re.MULTILINE)

# Task 3: LstmLayer nb_of_weights() for the two chained synthetic layers
# (I=3,O=2 -> 72; I=2,O=1 -> 28), printed by the harness's own getNbOfWeights().
LSTM_NB_RE = re.compile(r"^NN_REAL ok lstm_nb0=(?P<nb0>\d+) lstm_nb1=(?P<nb1>\d+)$", re.MULTILINE)
EXPECTED_LSTM_NB0 = 72
EXPECTED_LSTM_NB1 = 28

NN_PROBE_RE = re.compile(
    r"^NN_PROBE site=(?P<site>\w+) diverged=(?P<diverged>\d) mismatches=(?P<mismatches>\d+) "
    r"total=(?P<total>\d+) first=\((?P<row>-?\d+),(?P<col>-?\d+)\) "
    r"eigen=0x(?P<eigen>[0-9a-f]+) loop=0x(?P<loop>[0-9a-f]+)$",
    re.MULTILINE,
)

# Task 4: real LSTMLayer::feedForward vs the ascending-loop reimpl (lstmForwardLoop),
# max ULP/abs gap over the whole dump grid. max_ulp=0 => the reimpl reproduces Eigen's
# actual forward output bit-for-bit at these shapes (the op-order/grouping contract).
NN_TOL_RE = re.compile(
    r"^NN_TOL site=(?P<site>\w+) max_ulp=(?P<max_ulp>\d+) max_abs=(?P<max_abs>\S+)$",
    re.MULTILINE,
)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout


def _parse_nn_real(stdout: str) -> int:
    match = NN_REAL_RE.search(stdout)
    if match is None:
        raise SystemExit("NN_REAL line missing from harness stdout")
    weights = int(match["weights"])
    if weights != EXPECTED_NB_WEIGHTS:
        raise SystemExit(f"NN_REAL weights {weights} != {EXPECTED_NB_WEIGHTS}")
    return weights


def _parse_lstm_nb(stdout: str) -> tuple[int, int]:
    match = LSTM_NB_RE.search(stdout)
    if match is None:
        raise SystemExit("Task 3 lstm_nb0/lstm_nb1 line missing from harness stdout")
    nb0, nb1 = int(match["nb0"]), int(match["nb1"])
    if (nb0, nb1) != (EXPECTED_LSTM_NB0, EXPECTED_LSTM_NB1):
        raise SystemExit(f"lstm nb ({nb0},{nb1}) != expected ({EXPECTED_LSTM_NB0},{EXPECTED_LSTM_NB1})")
    return nb0, nb1


def _parse_nn_tol(stdout: str, site: str) -> dict[str, object]:
    for match in NN_TOL_RE.finditer(stdout):
        if match["site"] == site:
            return {
                "site": match["site"],
                "max_ulp": int(match["max_ulp"]),
                "max_abs": match["max_abs"],
            }
    raise SystemExit(f"NN_TOL line for site={site} missing from harness stdout")


def _parse_nn_probes(stdout: str) -> dict[str, dict[str, object]]:
    probes: dict[str, dict[str, object]] = {}
    for match in NN_PROBE_RE.finditer(stdout):
        site = match["site"]
        probes[site] = {
            "shape": PROBE_SITES.get(site, "unknown site"),
            "diverged": bool(int(match["diverged"])),
            "mismatches": int(match["mismatches"]),
            "total": int(match["total"]),
            "first_index": [int(match["row"]), int(match["col"])],
            "eigen_bits": match["eigen"],
            "loop_bits": match["loop"],
        }
    missing = set(PROBE_SITES) - set(probes)
    if missing:
        raise SystemExit(f"NN_PROBE lines missing from harness stdout: {sorted(missing)}")
    return probes


def _brew_version(dep: str) -> str:
    out = _run(["brew", "list", "--versions", dep]).strip()
    parts = out.split()
    return parts[1] if len(parts) > 1 else "unknown"


def main() -> None:
    # 1. Build the harness (now with the NN stack linked).
    _run(["bash", str(HARNESS_DIR / "build.sh")])

    # 2. Run it in a throwaway dir seeded with the Phase 1 harness inputs, so the
    #    Phase 1 fixture dir is not mutated. Pass the NN config + weights as absolute
    #    argv (main.cpp resolves argv[2]/argv[3] regardless of cwd).
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name in HARNESS_INPUTS:
            shutil.copy2(PHASE1_DIR / name, tmp_dir / name)
        stdout = _run(
            [
                str(HARNESS_DIR / "oracle_harness"),
                str(tmp_dir) + "/",
                str(NN_CONFIG),
                str(NN_WEIGHTS),
            ]
        )
        # Task 2 (act_sweep.bin) + Task 3 (lstm_w_roundtrip_{in,out}.bin): persist
        # these dumps into the committed Phase 2 dir (everything else the harness
        # writes into tmp_dir is feature-stage output already covered by Phase 1's
        # own fixtures, so only the EXPECTED_SHAPES entries are copied out).
        for name in EXPECTED_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE2_DIR / name)

    nb_weights = _parse_nn_real(stdout)
    lstm_nb0, lstm_nb1 = _parse_lstm_nb(stdout)
    probes = _parse_nn_probes(stdout)
    lstm_nn_tol = _parse_nn_tol(stdout, "lstm_forward")
    dense_nn_tol = _parse_nn_tol(stdout, "dense_forward")
    # Task 6: container NN_TOL sites (real NeuralNetwork<L> vs the netForwardLoop reimpl).
    net_lstm_fwd_tol = _parse_nn_tol(stdout, "net_lstm_forward_fwd")
    net_lstm_rev_tol = _parse_nn_tol(stdout, "net_lstm_forward_rev")
    net_dense_tol = _parse_nn_tol(stdout, "net_dense_forward")
    net_double_tol = _parse_nn_tol(stdout, "net_double")

    # 2b. Read each dump back, verify its shape, and build the inventory.
    dump_inventory: dict[str, dict[str, int]] = {}
    for name, (er, ec) in EXPECTED_SHAPES.items():
        rows, cols, _ = read_bin(PHASE2_DIR / name)
        if (rows, cols) != (er, ec):
            raise SystemExit(f"{name}: shape ({rows},{cols}) != expected ({er},{ec})")
        dump_inventory[name] = {"rows": rows, "cols": cols}

    # 3. Compiler version (same g++ selection as build.sh).
    gxx = _run(["bash", "-c", 'ls "$(brew --prefix)"/bin/g++-* | sort -V | tail -1']).strip()
    compiler = _run([gxx, "--version"]).splitlines()[0].strip()

    manifest = {
        "compiler": compiler,
        "brew_versions": {dep: _brew_version(dep) for dep in ["gcc", "eigen@3", "boost"]},
        "flags": ("-O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE -include boost/math/special_functions/round.hpp"),
        "nn_source": {
            "config": "tests/reference_data/phase0/1_worker_1.config",
            "weights": "tests/reference_data/phase0/NNweights_config1.bin",
            "weightsFile_cleared": True,
            "layer_type": "LSTMLayer",
            "lstm_neuron_nb": [23, 24, 24],
            "lstm_sub_sampling": [4, 1],
            "output_neuron_nb": [48, 12, 1],
            "output_sub_sampling": [1, 1],
        },
        "nb_of_weights": nb_weights,
        "lstm_layer_chain": {
            "text": (
                "Task 3: LstmLayer(conf,'SYNW',0,3,2,true) chained into "
                "LstmLayer(conf,'SYNW',1,2,1,true) over one synthetic flat vector "
                "(w[k] = ((k*11+3) % 97)/97.0 - 0.5). nb0/nb1 from getNbOfWeights()."
            ),
            "nb0": lstm_nb0,
            "nb1": lstm_nb1,
        },
        "nn_product_probes": {
            "text": PROBE_TEXT,
            "sites": probes,
        },
        "lstm_forward_tol": {
            "text": (
                "Task 4: max ULP/abs gap between the REAL LSTMLayer::feedForward and "
                "the ascending-loop reimpl (lstmForwardLoop) over the whole forward/"
                "reverse dump grid. max_ulp=0 means the reimpl reproduces Eigen's "
                "actual forward output bit-for-bit at these shapes -- the op-order + "
                "combined/separate peephole grouping contract (LSTMLayer.cpp:312-421)."
            ),
            **lstm_nn_tol,
        },
        "dense_forward_tol": {
            "text": (
                "Task 5: max ULP/abs gap between the REAL NeuronLayer::feedForward and "
                "the ascending-loop reimpl (denseForwardLoop) over the hidden/softmax/"
                "logistic/width-mismatch dump grid. max_ulp=0 means the reimpl "
                "reproduces Eigen's actual forward output bit-for-bit at these shapes -- "
                "the width-tolerant projection + UNSTABILIZED softmax (no max-subtraction "
                "overflow guard) + sequential row-sum contract (NeuronLayer.cpp:127-149)."
            ),
            **dense_nn_tol,
        },
        "network_container_tol": {
            "text": (
                "Task 6: max ULP/abs gap between the REAL NeuralNetwork<L>::feedForward / "
                "feedForwardReverse / feedForwardDouble and the container reimpl "
                "(netForwardLoop, wired to lstmForwardLoop/denseForwardLoop) over the "
                "stacked-net cases. max_ulp=0 means the reimpl reproduces the real "
                "container output bit-for-bit -- the SubSample dropped-tail floor + "
                "chained set_weights head/tail split + per-layer lastLayer flag + "
                "double-input hcat (NeuralNetwork.hpp:125-249)."
            ),
            "net_lstm_forward_fwd": net_lstm_fwd_tol,
            "net_lstm_forward_rev": net_lstm_rev_tol,
            "net_dense_forward": net_dense_tol,
            "net_double": net_double_tol,
        },
        "dumps": dump_inventory,
    }

    manifest_path = PHASE2_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    diverged = sum(1 for p in probes.values() if p["diverged"])
    print(f"OK: weights={nb_weights}, {len(probes)} probes ({diverged} diverged), manifest -> {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
