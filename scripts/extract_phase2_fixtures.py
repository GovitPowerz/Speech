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
dir seeded with copies of those inputs, keep only the parsed probe results, and
write the manifest to the Phase 2 dir. Task 1 is measurement-only, so the Phase 2
dir holds just the manifest (no ``.bin`` dumps yet).

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

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "tools" / "oracle_harness"
PHASE1_DIR = REPO_ROOT / "tests" / "reference_data" / "phase1"
PHASE2_DIR = REPO_ROOT / "tests" / "reference_data" / "phase2"
PHASE0_DIR = REPO_ROOT / "tests" / "reference_data" / "phase0"

NN_CONFIG = PHASE0_DIR / "1_worker_1.config"
NN_WEIGHTS = PHASE0_DIR / "NNweights_config1.bin"

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

NN_PROBE_RE = re.compile(
    r"^NN_PROBE site=(?P<site>\w+) diverged=(?P<diverged>\d) mismatches=(?P<mismatches>\d+) "
    r"total=(?P<total>\d+) first=\((?P<row>-?\d+),(?P<col>-?\d+)\) "
    r"eigen=0x(?P<eigen>[0-9a-f]+) loop=0x(?P<loop>[0-9a-f]+)$",
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

    nb_weights = _parse_nn_real(stdout)
    probes = _parse_nn_probes(stdout)

    # 3. Compiler version (same g++ selection as build.sh).
    gxx = _run(["bash", "-c", 'ls "$(brew --prefix)"/bin/g++-* | sort -V | tail -1']).strip()
    compiler = _run([gxx, "--version"]).splitlines()[0].strip()

    manifest = {
        "compiler": compiler,
        "brew_versions": {dep: _brew_version(dep) for dep in ["gcc", "eigen@3", "boost"]},
        "flags": (
            "-O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off "
            "-DEIGEN_DONT_VECTORIZE -include boost/math/special_functions/round.hpp"
        ),
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
        "nn_product_probes": {
            "text": PROBE_TEXT,
            "sites": probes,
        },
        "dumps": {},
    }

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PHASE2_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    diverged = sum(1 for p in probes.values() if p["diverged"])
    print(
        f"OK: weights={nb_weights}, {len(probes)} probes ({diverged} diverged), "
        f"manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
