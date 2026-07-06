"""Build and run the oracle harness for Phase 3, then write the backward NN-probe manifest.

Phase 3 lands network-level training (BPTT + iRPROP-). This extractor covers Task 1
(the harness-local ascending-loop BACKWARD reimpl family mirroring the Phase 2 forward
reimpl, + the NN_TOL/NN_PROBE backward probe stages) and Task 2 (CostLaw backward seam
VALIDATION: the REAL compiled `CostLaw::computeUnitaryDeltas`/`computeDeltas` probed
directly as the golden -- cost.rs was already ported in Phase 0b-i, so this is not a
transcription target), both already wired into ``tools/oracle_harness/main.cpp``. It:

  1. Rebuilds the harness (the backward reimpl statics + probe stage are already in it;
     the linked TUs are unchanged from Phase 2b -- BLSTMNeuralNetwork/LSTMLayer/
     NeuronLayer/CostLaw/Rprop were linked since Phase 2).
  2. Runs it in a throwaway dir seeded with the Phase 1 harness inputs (so no committed
     fixture dir is mutated), passing the same argv the Phase 2b extractor does.
  3. Parses the backward NN_TOL lines (site=lstm_backward|dense_backward|
     net_lstm_backward_{fwd,rev}|net_dense_backward|blstm_feedbackward|blstm_real_backward)
     and the NN_PROBE backward lines (bwd_outer_product, bwd_backproj) into manifest.json.
     ASSERTS every SYNTHETIC site is max_ulp=0 (the reimpl reproduces the real class
     bit-for-bit); RECORDS the real-net site (blstm_real_backward) nonzero delta as
     Phase 4 calibration (the goldens ARE the reimpl, not the real-Eigen derivs).
  4. Copies the reimpl-produced Nx2 deriv goldens (bwd_*.bin) into
     tests/reference_data/phase3/ (Tasks 3-6 of Phase 3 consume them).
  5. REGRESSION GUARD (copy the phase2b pattern): hashes every file under
     tests/reference_data/{phase1,phase2,phase2b} BEFORE and AFTER the harness run and
     SystemExit's on any drift (the Phase 3 stage must not perturb a prior-phase dump).

Run TWICE; the manifest + goldens must be byte-identical.

Usage: uv run python scripts/extract_phase3_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "tools" / "oracle_harness"
PHASE0_DIR = REPO_ROOT / "tests" / "reference_data" / "phase0"
PHASE1_DIR = REPO_ROOT / "tests" / "reference_data" / "phase1"
PHASE2_DIR = REPO_ROOT / "tests" / "reference_data" / "phase2"
PHASE2B_DIR = REPO_ROOT / "tests" / "reference_data" / "phase2b"
PHASE3_DIR = REPO_ROOT / "tests" / "reference_data" / "phase3"

NN_CONFIG = PHASE0_DIR / "1_worker_1.config"
NN_WEIGHTS = PHASE0_DIR / "NNweights_config1.bin"
TDC_CONFIG = PHASE2B_DIR / "tdc.config"
LTSV_CONFIG = PHASE2B_DIR / "ltsv.config"
LTSV_DCT_CONFIG = PHASE2B_DIR / "ltsv_dct.config"
LTSV_TINY_CONFIG = PHASE2B_DIR / "ltsv_tiny.config"
LTSV_POWERMEL_CONFIG = PHASE2B_DIR / "ltsv_powermel.config"
SIGNAL_CONFIG = PHASE2B_DIR / "signal.config"

# Input fixtures the harness reads from its output dir (same set as Phase 1/2/2b).
HARNESS_INPUTS = [
    "excerpt_2ch_8k.wav",
    "variant_mfcc_deltas.config",
    "variant_mfcc_sdc.config",
    "variant_logmel.config",
    "variant_rawband_ltsv.config",
]

# The reimpl-produced Nx2 deriv goldens (col0 = summed deriv, col1 = frame count),
# with their expected (rows, cols). Tasks 3-6 of Phase 3 consume these.
#   lstm_backward:     LSTMLayer(3,2) nb = 4*3*2+4*2*2+12*2+4*2 = 72
#   dense_backward:    NeuronLayer(4,3) = 3*(4+1) = 15; NeuronLayer(4,1) = 5
#   net_lstm_backward: net [3,4,2] sub [2,1] -> L0 LSTM(6,4)=224 + L1 LSTM(4,2)=80 = 304
#   net_dense_backward: net [4,3,2] sub [1,2] -> L0 dense(4,3)=15 + L1 dense(6,2)=14 = 29
#   blstm_feedbackward: synthetic BLSTM (LSTM [3,4,2] sub [2,1] fwd+bwd + output
#                       [4,5,2] sub [1,1] + 2*3 stats tail) = 651
#   blstm_real_backward: the REAL 33,671-weight net (1_worker_1.config + NNweights)
EXPECTED_SHAPES = {
    "bwd_lstm_derivs.bin": (72, 2),
    "bwd_dense_hidden_derivs.bin": (15, 2),
    "bwd_dense_last_derivs.bin": (15, 2),
    "bwd_dense_logistic_derivs.bin": (5, 2),
    "bwd_net_lstm_fwd_derivs.bin": (304, 2),
    "bwd_net_lstm_rev_derivs.bin": (304, 2),
    "bwd_net_dense_derivs.bin": (29, 2),
    "bwd_blstm_derivs.bin": (651, 2),
    "bwd_blstm_real_derivs.bin": (33671, 2),
}

# Task 3: dense backward deltas_out (T x I) per NeuronLayer grid variant, dumped
# alongside the existing Nx2 deriv goldens above (same harness stage).
EXPECTED_DELTASOUT_SHAPES = {
    "bwd_dense_hidden_deltasout.bin": (6, 4),
    "bwd_dense_last_deltasout.bin": (6, 4),
    "bwd_dense_logistic_deltasout.bin": (6, 4),
}

# Task 2: CostLaw backward REAL-probe goldens (dumped from the REAL compiled
# computeUnitaryDeltas/computeDeltas, NOT a reimpl -- CostLaw is tier-2 oracle
# per spec S2, probed directly). Nx2 [output | delta] for the scalar sweeps
# (65-point k/64 grid), n_frames x n_classes for the multiclass fusion dumps.
COST_SHAPES = {
    "cost_deriv_scalar_poly_speech.bin": (65, 2),
    "cost_deriv_scalar_poly_other.bin": (65, 2),
    "cost_deriv_scalar_log_speech.bin": (65, 2),
    "cost_deriv_scalar_log_other.bin": (65, 2),
    "cost_deriv_scalar_sqrt_speech.bin": (65, 2),
    "cost_deriv_scalar_sqrt_other.bin": (65, 2),
    "cost_deltas_multiclass.bin": (4, 3),
    "cost_deltas_multiclass_pond.bin": (4, 3),
    "cost_deltas_wer.bin": (3, 3),
    "cost_deltas_wer_pond.bin": (3, 3),
    # Fix-wave 1 (review finding 1): mid-thresh (0.5/0.5) above-branch coverage
    # for the cubic and sqrt laws, at points away from 0/1 (nonzero logistic fold).
    "cost_deriv_scalar_midthresh_cubic_speech.bin": (3, 2),
    "cost_deriv_scalar_midthresh_cubic_other.bin": (3, 2),
    "cost_deriv_scalar_midthresh_sqrt_speech.bin": (3, 2),
    "cost_deriv_scalar_midthresh_sqrt_other.bin": (3, 2),
}

# The brief's literal alias names (cost_deriv_scalar_{speech,other}.bin) are a byte
# copy of the poly-named dump -- fix-wave 1 (review finding 2) removed the second,
# independent C++ dump block that produced them (a manual lock-step trap) in favor
# of a single copy here. Both filenames stay on disk; the Rust golden test
# references the literal alias names.
ALIAS_COPIES = {
    "cost_deriv_scalar_speech.bin": "cost_deriv_scalar_poly_speech.bin",
    "cost_deriv_scalar_other.bin": "cost_deriv_scalar_poly_other.bin",
}

# NN_TOL backward sites. Synthetic sites MUST be max_ulp=0; the real-net site is
# recorded nonzero (Phase 4 calibration).
SYNTHETIC_TOL_SITES = [
    "lstm_backward",
    "dense_backward",
    "dense_backward_deltasout",
    "net_lstm_backward_fwd",
    "net_lstm_backward_rev",
    "net_dense_backward",
    "blstm_feedbackward",
]
REAL_TOL_SITE = "blstm_real_backward"

# NN_PROBE backward sites (pure-arithmetic outer product + back-projection at synthetic
# k -- Eigen vs matSeq must agree bit-for-bit, diverged=0).
PROBE_SITES = {
    "bwd_outer_product": "(4x1)*(1x3) weight-deriv outer product input.col(row)*deltas.row(row).",
    "bwd_backproj": "(6x3)*(3x4) deltas*W^T back-projection (k=O), the divergent GEMM at NN width.",
}

EXPECTED_NB_DERIVS = 33671

NN_TOL_RE = re.compile(
    r"^NN_TOL site=(?P<site>\w+) max_ulp=(?P<max_ulp>\d+) max_abs=(?P<max_abs>\S+)$",
    re.MULTILINE,
)
NN_PROBE_RE = re.compile(
    r"^NN_PROBE site=(?P<site>\w+) diverged=(?P<diverged>\d) mismatches=(?P<mismatches>\d+) "
    r"total=(?P<total>\d+) first=\((?P<row>-?\d+),(?P<col>-?\d+)\) "
    r"eigen=0x(?P<eigen>[0-9a-f]+) loop=0x(?P<loop>[0-9a-f]+)$",
    re.MULTILINE,
)
NN_REAL_DERIVS_RE = re.compile(r"^NN_REAL ok backward_nb_derivs=(?P<n>\d+)$", re.MULTILINE)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout


def _hash_tree(root: Path) -> dict[str, str]:
    """Map every file under `root` (relative posix path) to its sha256."""
    digests: dict[str, str] = {}
    if not root.is_dir():
        return digests
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _read_bin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        rows = int.from_bytes(f.read(8), "little", signed=True)
        cols = int.from_bytes(f.read(8), "little", signed=True)
    return rows, cols


def _parse_nn_tol(stdout: str) -> dict[str, dict[str, object]]:
    tols: dict[str, dict[str, object]] = {}
    for m in NN_TOL_RE.finditer(stdout):
        tols[m["site"]] = {"max_ulp": int(m["max_ulp"]), "max_abs": m["max_abs"]}
    for site in [*SYNTHETIC_TOL_SITES, REAL_TOL_SITE]:
        if site not in tols:
            raise SystemExit(f"NN_TOL line for site={site} missing from harness stdout")
    # Synthetic backward sites MUST be bit-exact vs the real compiled class.
    nonzero = [s for s in SYNTHETIC_TOL_SITES if tols[s]["max_ulp"] != 0]
    if nonzero:
        raise SystemExit(f"synthetic backward NN_TOL sites must be max_ulp=0 (the reimpl is wrong): {[(s, tols[s]['max_ulp']) for s in nonzero]}")
    return tols


def _parse_nn_probes(stdout: str) -> dict[str, dict[str, object]]:
    probes: dict[str, dict[str, object]] = {}
    for m in NN_PROBE_RE.finditer(stdout):
        if m["site"] not in PROBE_SITES:
            continue
        probes[m["site"]] = {
            "shape": PROBE_SITES[m["site"]],
            "diverged": bool(int(m["diverged"])),
            "mismatches": int(m["mismatches"]),
            "total": int(m["total"]),
        }
    missing = set(PROBE_SITES) - set(probes)
    if missing:
        raise SystemExit(f"NN_PROBE backward lines missing from harness stdout: {sorted(missing)}")
    diverged = [s for s, p in probes.items() if p["diverged"]]
    if diverged:
        raise SystemExit(f"synthetic-k backward NN_PROBE sites must NOT diverge: {diverged}")
    return probes


def main() -> None:
    # 1. Build the harness.
    _run(["bash", str(HARNESS_DIR / "build.sh")])

    # 2. Snapshot the committed prior-phase fixture dirs BEFORE the run.
    before = {p: _hash_tree(d) for p, d in (("phase1", PHASE1_DIR), ("phase2", PHASE2_DIR), ("phase2b", PHASE2B_DIR))}

    # 3. Run the harness in a throwaway dir seeded with the Phase 1 inputs.
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
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
                str(TDC_CONFIG),
                str(LTSV_CONFIG),
                str(LTSV_DCT_CONFIG),
                str(LTSV_TINY_CONFIG),
                str(LTSV_POWERMEL_CONFIG),
                str(SIGNAL_CONFIG),
            ]
        )
        # Copy the reimpl-produced Nx2 deriv goldens into the committed Phase 3 dir.
        for name in EXPECTED_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Task 3: dense backward deltas_out goldens (same harness stage).
        for name in EXPECTED_DELTASOUT_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Copy the Task 2 CostLaw REAL-probe goldens (dumped from the real class).
        for name in COST_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Produce the brief's literal alias filenames as a byte copy of the
        # poly-named dump (fix-wave 1, review finding 2) -- no second harness
        # dump block, single source of truth.
        for alias_name, source_name in ALIAS_COPIES.items():
            shutil.copy2(PHASE3_DIR / source_name, PHASE3_DIR / alias_name)

    # 4. Regression guard: the prior-phase dirs must be byte-identical after the run.
    after = {p: _hash_tree(d) for p, d in (("phase1", PHASE1_DIR), ("phase2", PHASE2_DIR), ("phase2b", PHASE2B_DIR))}
    for phase in ("phase1", "phase2", "phase2b"):
        b, a = before[phase], after[phase]
        drifted = sorted(name for name in set(b) | set(a) if b.get(name) != a.get(name))
        if drifted:
            raise SystemExit(f"REGRESSION: {phase} fixtures changed after harness run: {drifted}")

    # 5. Parse + validate the backward probe lines.
    tols = _parse_nn_tol(stdout)
    probes = _parse_nn_probes(stdout)
    real_match = NN_REAL_DERIVS_RE.search(stdout)
    if real_match is None:
        raise SystemExit("NN_REAL ok backward_nb_derivs line missing from harness stdout")
    nb_derivs = int(real_match["n"])
    if nb_derivs != EXPECTED_NB_DERIVS:
        raise SystemExit(f"real-net backward deriv count {nb_derivs} != {EXPECTED_NB_DERIVS}")

    # 6. Shape sync: every persisted golden must match EXPECTED_SHAPES.
    mismatches = []
    for name, expected in EXPECTED_SHAPES.items():
        got = _read_bin_shape(PHASE3_DIR / name)
        if got != expected:
            mismatches.append({"file": name, "expected": expected, "got": got})
    if mismatches:
        raise SystemExit(f"bwd_*.bin shape mismatch: {mismatches}")
    deltasout_mismatches = []
    for name, expected in EXPECTED_DELTASOUT_SHAPES.items():
        got = _read_bin_shape(PHASE3_DIR / name)
        if got != expected:
            deltasout_mismatches.append({"file": name, "expected": expected, "got": got})
    if deltasout_mismatches:
        raise SystemExit(f"bwd_dense_*_deltasout.bin shape mismatch: {deltasout_mismatches}")
    cost_mismatches = []
    for name, expected in COST_SHAPES.items():
        got = _read_bin_shape(PHASE3_DIR / name)
        if got != expected:
            cost_mismatches.append({"file": name, "expected": expected, "got": got})
    if cost_mismatches:
        raise SystemExit(f"cost_*.bin shape mismatch: {cost_mismatches}")
    alias_mismatches = []
    for alias_name, source_name in ALIAS_COPIES.items():
        expected = COST_SHAPES[source_name]
        got = _read_bin_shape(PHASE3_DIR / alias_name)
        if got != expected:
            alias_mismatches.append({"file": alias_name, "expected": expected, "got": got})
    if alias_mismatches:
        raise SystemExit(f"cost_*.bin alias shape mismatch: {alias_mismatches}")

    # 7. Compiler version (same g++ selection as build.sh).
    gxx = _run(["bash", "-c", 'ls "$(brew --prefix)"/bin/g++-* | sort -V | tail -1']).strip()
    compiler = _run([gxx, "--version"]).splitlines()[0].strip()

    manifest = {
        "compiler": compiler,
        "flags": ("-O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE -include boost/math/special_functions/round.hpp"),
        "backward_reimpl": {
            "text": (
                "Phase 3 Task 1: the harness-local ascending-loop BACKWARD reimpl family "
                "(lstmForwardLoopCache/lstmBackwardLoop/lstmBackwardReverseLoop, "
                "denseBackwardLoop, netBackwardLoop/netBackwardReverseLoop, and the "
                "LstmSubNet/DenseSubNet/BlstmBack container reimpls) mirrors the Phase 2 "
                "forward reimpl family and is built on matSeq + the same file-scope kernels. "
                "For each site the harness constructs the REAL compiled layer/net, runs its "
                "forward THEN backward on the same operands, harvests "
                "getWeightsDerivatives().col(0) (the summed derivative), and compares it "
                "element-by-element (ULP) against the reimpl's col(0). The reimpl-produced "
                "Nx2 derivs (col0 = summed deriv, col1 = _NbOfSeqFedBackward frame count "
                "replicated) ARE the goldens Tasks 3-6 consume -- NOT the real-Eigen derivs, "
                "because the backward contains Eigen GEMMs at NN-wide k (_FeedbackWeights*"
                "test LSTMLayer.cpp:703, _InputWeights*testM :707, NeuronLayer deltas*"
                "_Weights.transpose() :202) that diverge from ascending accumulation at the "
                "real net's k>=23 shapes, doubly non-local because the forward already seeds "
                "the caches. Derivatives are recomputed FROM the post-activation caches "
                "(_Gates/_CellStates/_CellsIn) via the activation deriv structs (GatesFunction/"
                "Maxmin2/Identity::deriv) -- NO forward change. The seed deltas for the BLSTM "
                "sites come from the REAL compiled CostLaw::computeDeltas (bit-portable: the "
                "synthetic site's MULTICLASS softmax+CE path is pure arithmetic; the real "
                "net's scalar log-law seed is the same-libm calibration path)."
            ),
        },
        "lstm_backward_tol": {
            "text": (
                "Standalone LSTMLayer::feedBackward (LSTMLayer.cpp:518-723) vs lstmBackwardLoop "
                "over {I=3,O=2,T=7; I=5,O=4,T=6}, all peepholes on, deterministic synthetic "
                "weights/input/deltas. max_ulp=0 means the reimpl reproduces the real layer's "
                "getWeightsDerivatives bit-for-bit at these shapes (the reverse-time loop, the "
                "O->state->C->F->I gate-derivative order, the 12-row peephole index map, the "
                "row==0 no-cellstate forget branch, the deltasForgetGatetmp save/read)."
            ),
            **tols["lstm_backward"],
        },
        "dense_backward_tol": {
            "text": (
                "Standalone NeuronLayer::feedBackward (NeuronLayer.cpp:151-207) vs "
                "denseBackwardLoop over {I=4,O=3 hidden; I=4,O=3 last; I=4,O=1 last/logistic}. "
                "max_ulp=0. The hidden and last sites share col0 derivs (the weight-deriv "
                "input.col*deltas.row is lastLayer-agnostic; lastLayer only changes deltas_out, "
                "where the softmax+CE fusion means the output layer applies NO activation "
                "Jacobian -- the fusion lives in CostLaw). The blstm_feedbackward site is the "
                "end-to-end proof of the no-double-count property. Task 3 adds the deltas_out "
                "(T x I) golden per variant (dense_backward_deltasout, also max_ulp=0): the "
                "last-layer variants prove deltas_out == deltas*W^T with NO activation Jacobian "
                "(double-count trap, spec S11.3); the hidden variant proves the asinh-deriv "
                "(Maxmin2::deriv) fold on the LAYER INPUT."
            ),
            **tols["dense_backward"],
            "deltasout": tols["dense_backward_deltasout"],
        },
        "net_backward_tol": {
            "text": (
                "NeuralNetwork<LSTMLayer>/<NeuronLayer>::feedBackward (NeuralNetwork.hpp:251-351) "
                "vs netBackwardLoop over the LSTM net [3,4,2] sub [2,1] (fwd + reverse) and the "
                "dense net [4,3,2] sub [1,2], T=11 (the SubSample interaction: decimated forward "
                "rows vs the InvSubSample delta inflation + running invSubSamplingRatio). "
                "max_ulp=0 all four. The reverse site pins the time-reversed cache consumption "
                "(feedBackwardReverse reverses input/output/deltas, runs the forward-order "
                "backward, reverses the returned deltas)."
            ),
            "net_lstm_backward_fwd": tols["net_lstm_backward_fwd"],
            "net_lstm_backward_rev": tols["net_lstm_backward_rev"],
            "net_dense_backward": tols["net_dense_backward"],
        },
        "blstm_backward_tol": {
            "text": (
                "BLSTMNeuralNetwork<LSTMLayer>::feedBackward (BLSTMNeuralNetwork.cpp:439-460) + "
                "getWeightsDerivatives (:255-276) vs BlstmBack. The synthetic site "
                "(blstm_feedbackward: LSTM [3,4,2] sub [2,1] fwd+bwd, output [4,5,2] sub [1,1], "
                "T=12, multiclass targets, backprop active) is max_ulp=0 -- the reimpl "
                "reproduces the whole fan-out (CostLaw seed -> feedBackwardDouble hcat -> "
                "fwd feedBackward + bwd feedBackwardReverse) bit-for-bit, proving the Nx2 flat "
                "deriv layout (fwd|bwd|output|[0|1|0|1] stats tail) matches "
                "getWeightsDerivatives element-for-element. The real-net site "
                "(blstm_real_backward: the 33,671-weight net on e2e_input.bin, "
                "InputNormalizationType -1, scalar-VAD targets, log-law CostLaw seed) is "
                "EXPECTED NONZERO -- k=23/24/48 GEMM divergence, RECORDED here as Phase 4 "
                "calibration (the reimpl derivs ARE the golden, bwd_blstm_real_derivs.bin)."
            ),
            "synthetic": tols["blstm_feedbackward"],
            "real_calibration": tols[REAL_TOL_SITE],
            "real_nb_derivs": nb_derivs,
        },
        "nn_backward_product_probes": {
            "text": (
                "Pure-arithmetic backward products at synthetic k: the weight-deriv outer "
                "product input.col(row)*deltas.row(row) (small k) and the deltas*W^T "
                "back-projection (k=O). Eigen's product vs the ascending matSeq must agree "
                "bit-for-bit (diverged=0) at these shapes -- the same NN_PROBE format as the "
                "Phase 2 forward product probes."
            ),
            "sites": probes,
        },
        "dumps": {
            **{name: {"rows": r, "cols": c} for name, (r, c) in EXPECTED_SHAPES.items()},
            **{name: {"rows": r, "cols": c} for name, (r, c) in EXPECTED_DELTASOUT_SHAPES.items()},
        },
        "costlaw_backward": {
            "text": (
                "Task 2: CostLaw backward seam VALIDATION (cost.rs was already ported in "
                "Phase 0b-i). The REAL compiled CostLaw::computeUnitaryDeltas/computeDeltas "
                "(CostLaw.cpp:278-445, CostLaw.h:6-213) is probed DIRECTLY as the golden (S2 "
                "tier 2) -- no transcription. Scalar VAD sweeps: output=k/64 for k in [0,64] "
                "(65 points), target=1.0 (speech) / target=0.0 (other), for the square "
                "(polynomial, strict-bits), log (libm, canary-gated), and sqrt (libm, "
                "canary-gated) laws -- exercises the below/above-thresh switch, the no-speech "
                "1-output flip + derivative negation, and the output*(1-output) logistic "
                "chain-rule fold (:344). cost_deriv_scalar_{speech,other}.bin are a byte copy "
                "(shutil.copy2, this extractor) of the square-law dump under the brief's "
                "literal names -- fix-wave 1 (review finding 2) removed the second, "
                "independent C++ dump block that produced them. FIX-WAVE 1 (review finding "
                "1): the real config's CostLawThreshSpeech=1/CostLawThreshNoSpeech=0 "
                "(1_worker_1.config:64-65) means the k/64 sweep above only ever exercises "
                "the above-thresh branch at output=1.0/0.0, where the logistic fold is "
                "exactly zero -- cost_deriv_scalar_midthresh_{cubic,sqrt}_{speech,other}.bin "
                "override BOTH thresholds to 0.5 and dump 3 explicit points landing in the "
                "above-thresh regime away from 0/1 (speech 0.6/0.75/0.9, no-speech "
                "0.4/0.25/0.1 -- CostLaw.cpp:282,297,308,324), for the cubic law "
                "(AboveThreshCubicLaw, CostLaw.h:84-117, strict bits) and the sqrt law "
                "(AboveThreshSqrtLaw, CostLaw.h:182-213, canary-gated). Multiclass fusion "
                "(cost_deltas_multiclass.bin, n_frames=4 x n_classes=3): row1 is FULLY "
                "ignore-masked (target<0 -> delta==0.0 exactly), rows 0/2/3 place the on-class "
                "at different columns; the _pond variant sets classes_ponderations=2,3,4 and "
                "scales the WHOLE on-class row by the on-class ponderation (:410-417). "
                "BackPropWER (cost_deltas_wer.bin, n_frames=3 x n_classes=3): SOFT targets "
                "(0.9/0.8 on-class, not exact 1.0) because the WER scaling `10*(1-target)` / "
                "`10*target` (:379-397) VANISHES at target==1.0/0.0 exactly -- soft targets are "
                "the engine's real BackPropWER convention (BLSTMNeuralNetwork.cpp:871, "
                "0.1*modifier); a one-hot-only golden would be vacuous. Row2 is fully masked "
                "(ignore composes with WER). The _pond variant adds classes_ponderations to the "
                "per-element WER arm. All multiclass/WER dumps are pure arithmetic (no libm) -> "
                "strict bits."
            ),
            "dumps": {name: {"rows": r, "cols": c} for name, (r, c) in COST_SHAPES.items()},
            "alias_dumps": {
                alias_name: {"copy_of": source_name, **{"rows": COST_SHAPES[source_name][0], "cols": COST_SHAPES[source_name][1]}}
                for alias_name, source_name in ALIAS_COPIES.items()
            },
        },
    }

    manifest_path = PHASE3_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    real_ulp = tols[REAL_TOL_SITE]["max_ulp"]
    print(
        f"OK: {len(SYNTHETIC_TOL_SITES)} synthetic backward sites max_ulp=0, "
        f"real-net calibration max_ulp={real_ulp}, "
        f"{len(EXPECTED_SHAPES) + len(EXPECTED_DELTASOUT_SHAPES) + len(COST_SHAPES)} goldens -> "
        f"{manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
