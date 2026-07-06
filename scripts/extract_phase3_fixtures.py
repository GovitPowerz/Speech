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
     net_lstm_backward_{fwd,rev}|net_dense_backward|
     net_single_layer_backward_{subsample,plain}|blstm_feedbackward|blstm_real_backward)
     and the NN_PROBE backward lines (bwd_outer_product, bwd_backproj) into manifest.json.
     ASSERTS every SYNTHETIC site is max_ulp=0 (the reimpl reproduces the real class
     bit-for-bit); RECORDS the real-net site (blstm_real_backward) nonzero delta as
     Phase 4 calibration (the goldens ARE the reimpl, not the real-Eigen derivs).
  4. Copies the reimpl-produced Nx2 deriv goldens (bwd_*.bin) into
     tests/reference_data/phase3/ (Tasks 3-6 of Phase 3 consume them).
  5. REGRESSION GUARD (copy the phase2b pattern): hashes every file under
     tests/reference_data/{phase1,phase2,phase2b} BEFORE and AFTER the harness run and
     SystemExit's on any drift (the Phase 3 stage must not perturb a prior-phase dump).
  6. Copies the Task 7 Rprop trajectory dumps (rprop_traj{A,B}_step<k>_*.bin, the REAL
     compiled Rprop::updateWeights -- pure scalar, no libm/GEMM, bit-portable) and
     computes a per-step/per-element BRANCH RECORD from the trajectory's own
     derivs/costs (independent re-derivation of the deriv-sign x prev-deriv-sign x
     cost-comparison state machine, NOT read back from the dumps) into manifest.json,
     asserting all 7 required branch kinds (S11.1) are covered.

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
    # Output sub-sampling ratio 2: LSTM [3,4,2] sub [2,1] fwd(304)+bwd(304) + output
    # [4,5,2] sub [2,1] (L0 dense(8,5)=45 + L1 dense(5,2)=12 = 57) + 2*3 tail = 671. The
    # fwd/bwd col0 is scaled by 2 (:266-269) -- pins the ratio-multiplier mutation.
    "bwd_blstm_outratio_derivs.bin": (671, 2),
    "bwd_blstm_real_derivs.bin": (33671, 2),
}

# Task 6: windowed BLSTM backward goldens (the REAL feedForwardBackward's
# getWeightsDerivatives per processing type), on the windowed synthetic net LSTM
# [2,2] sub [1] + output [4,2] sub [1] (sub_sampling_ratio == 1). N = fwd LSTM(2,2)=64
# + bwd LSTM(2,2)=64 + output(4,2)=10 + 2*inputSize(2) stats tail = 142.
#   plain:     T=8,  window 0            -> one pass, count 8
#   truncate:  T=12, window 4           -> 3 non-overlapping windows, count 12
#   twosweeps: T=12, window 4, 2 sweeps -> both sweeps back-propagate, count 34
#   overlap:   T=12, window 3, shift 3  -> multi-covering, count 27
#   enforce:   T=8,  TargetEnforcementStep -1 -> one pass, count 8 (enforcement moves
#              the deltas, not the count)
WINDOWED_BWD_SHAPES = {
    "blstm_bwd_plain_derivs.bin": (142, 2),
    "blstm_bwd_truncate_derivs.bin": (142, 2),
    "blstm_bwd_twosweeps_derivs.bin": (142, 2),
    "blstm_bwd_overlap_derivs.bin": (142, 2),
    "blstm_bwd_enforce_derivs.bin": (142, 2),
}

# Per-variant windowed count expectations (spec S11.4): the hand-computable col1
# trained-weight count + the driver params the Rust test replays. `trained_count` is
# the total frame count fed backward across all windows/sweeps for every trained
# weight (uniform since ssr == 1); the stats tail is always count 1. MEASURED from the
# harness dumps (not hardcoded blindly): the extractor reads col1 and asserts the
# recorded value matches, so a drift surfaces here.
WINDOWED_BWD_COUNTS = {
    "plain": {"t_len": 8, "window_size": 0, "window_shift": 0, "trained_count": 8},
    "truncate": {"t_len": 12, "window_size": 4, "window_shift": 0, "trained_count": 12},
    "twosweeps": {"t_len": 12, "window_size": 4, "window_shift": 0, "trained_count": 34},
    "overlap": {"t_len": 12, "window_size": 3, "window_shift": 3, "trained_count": 27},
    "enforce": {"t_len": 8, "window_size": 0, "window_shift": 0, "trained_count": 8},
}

# Task 3: dense backward deltas_out (T x I) per NeuronLayer grid variant, dumped
# alongside the existing Nx2 deriv goldens above (same harness stage).
EXPECTED_DELTASOUT_SHAPES = {
    "bwd_dense_hidden_deltasout.bin": (6, 4),
    "bwd_dense_last_deltasout.bin": (6, 4),
    "bwd_dense_logistic_deltasout.bin": (6, 4),
}

# Task 5: the Network<L>::feedBackward returned deltas_out (deltasPreviousLayer) at
# the two net sites, dumped alongside the existing Nx2 net-deriv goldens. The row
# count is floor(T/R)*R, NOT T: SubSample floors 11 -> 5 rows, InvSubSample restores
# 5 -> 10, and the dropped trailing row is never recovered (documented quirk).
#   net_lstm  [3,4,2] sub [2,1]: layer-0 InvSubSample(2) of a 5-row deltasPrev -> 10 x 3.
#   net_dense [4,3,2] sub [1,2]: layer-1 InvSubSample(2) -> 10 rows; layer 0 back-projects
#                                at deltas.rows()=10 -> 10 x 4 (fan-in of net input).
NET_BWD_DELTASOUT_SHAPES = {
    "bwd_net_lstm_fwd_deltasout.bin": (10, 3),
    "bwd_net_lstm_rev_deltasout.bin": (10, 3),
    "bwd_net_dense_deltasout.bin": (10, 4),
}

# Fix-wave 1 (review finding 1): the single-layer container backward branch
# (neuronNb.size()==2, NeuralNetwork.hpp:255-263) had ZERO golden coverage -- both
# nets above are neuronNb.size()==3 (multi-layer). LSTM [2,2] at T=7 (odd length):
# subsample variant (sub [2], the brief's dropped fixture) exercises the InvSubSample
# arm; plain variant (sub [1], legacy :262) exercises the no-subsample arm. Both are
# structurally distinct from the multi-layer jj==0 arm: the single-layer branch reads
# `deltas`/`deltas.rows()` (the SEED) directly, never the running `deltasOut`.
#   subsample: I=2*2=4 -> weights 4*4*2+4*2*2+12*2+4*2=80; deltas_out floor(7/2)*2=6 x 2.
#   plain:     I=2*1=2 -> weights 4*2*2+4*2*2+12*2+4*2=64; deltas_out T=7 x 2 (no floor).
NET_SINGLE_LAYER_DERIVS_SHAPES = {
    "bwd_net_single_subsample_derivs.bin": (80, 2),
    "bwd_net_single_plain_derivs.bin": (64, 2),
}
NET_SINGLE_LAYER_DELTASOUT_SHAPES = {
    "bwd_net_single_subsample_deltasout.bin": (6, 2),
    "bwd_net_single_plain_deltasout.bin": (7, 2),
}

# Task 4: LSTMLayer::feedBackward per-variant goldens (synthetic I=2,O=2,T=5, odd
# length). deltasPreviousLayer is T x I = 5 x 2; derivs are Nx2 with
# N = 4*I*O + 4*O*O + 12*O + 4*O = 16 + 16 + 24 + 8 = 64 (layer-native shape). The
# signal_width variant feeds a 1-col input into the I=2 layer (width tolerance) --
# its derivs stay at the layer-native N=64 with the dead weight rows exactly 0.
LSTM_BWD_DELTASPREV_SHAPES = {
    "lstm_bwd_deltasprev_peep_all.bin": (5, 2),
    "lstm_bwd_deltasprev_peep_none.bin": (5, 2),
    "lstm_bwd_deltasprev_reverse.bin": (5, 2),
    "lstm_bwd_deltasprev_subsample.bin": (5, 2),
}
LSTM_BWD_DERIVS_SHAPES = {
    "lstm_bwd_derivs_peep_all.bin": (64, 2),
    "lstm_bwd_derivs_peep_none.bin": (64, 2),
    "lstm_bwd_derivs_reverse.bin": (64, 2),
    "lstm_bwd_derivs_subsample.bin": (64, 2),
    "lstm_bwd_signal_derivs.bin": (64, 2),
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
    "lstm_backward_variants",
    "dense_backward",
    "dense_backward_deltasout",
    "net_lstm_backward_fwd",
    "net_lstm_backward_rev",
    "net_dense_backward",
    "net_single_layer_backward_subsample",
    "net_single_layer_backward_plain",
    "blstm_feedbackward",
    # Task 6 windowed backward sites (the reimpl reproduces the real net bit-for-bit
    # at the synthetic ssr==1 shapes -- all must be max_ulp=0).
    "blstm_bwd_plain",
    "blstm_bwd_truncate",
    "blstm_bwd_twosweeps",
    "blstm_bwd_overlap",
    "blstm_bwd_enforce",
    "blstm_bwd_outputonly",
]
REAL_TOL_SITE = "blstm_real_backward"

# Task 5: the net-container deltas_out is a deltas*W^T GEMM (k=O), so the REAL Eigen
# path diverges from the ascending reimpl in the last ULP -- recorded as calibration
# (the reimpl dump is the golden), NOT gated to 0 like the k=1 outer-product derivs.
DELTASOUT_CALIB_SITES = [
    "net_lstm_backward_fwd_deltasout",
    "net_lstm_backward_rev_deltasout",
    "net_dense_backward_deltasout",
    "net_single_layer_backward_subsample_deltasout",
    "net_single_layer_backward_plain_deltasout",
]

# NN_PROBE backward sites (pure-arithmetic outer product + back-projection at synthetic
# k -- Eigen vs matSeq must agree bit-for-bit, diverged=0).
PROBE_SITES = {
    "bwd_outer_product": "(4x1)*(1x3) weight-deriv outer product input.col(row)*deltas.row(row).",
    "bwd_backproj": "(6x3)*(3x4) deltas*W^T back-projection (k=O), the divergent GEMM at NN width.",
}

# Task 7: Rprop iRPROP- trainer trajectories (REAL compiled Rprop::updateWeights,
# tools/oracle_harness/main.cpp Phase 3 Task 7 stage). Defined here (not just read
# back from the dumps) so the branch record is an INDEPENDENT re-derivation of the
# deriv-sign x prev-deriv-sign x cost-comparison state machine -- the same inputs
# fed to the harness, mirrored byte-for-byte, so a copy/paste error in the harness's
# hardcoded arrays would surface as a manifest/dump mismatch, not be silently trusted.
RPROP_ETA_MIN = 0.5
RPROP_ETA_PLUS = 1.2
RPROP_MIN_DELTA = 1e-9
RPROP_MAX_DELTA = 0.2
RPROP_INIT_DELTA = 1e-2

TRAJ_A_DERIVS = [
    [1.0, -1.0, 1.0, 0.0, 1.0],
    [1.0, -1.0, 1.0, 1.0, 0.0],
    [1.0, -1.0, -1.0, 1.0, 0.0],
    [1.0, -1.0, -1.0, -1.0, 0.0],
    [1.0, -1.0, -1.0, -1.0, 0.0],
]
TRAJ_A_COSTS = [10.0, 9.0, 12.0, 8.0, 7.0]
TRAJ_A_N = 5
TRAJ_A_STEPS = 5

TRAJ_B_N = 2
TRAJ_B_STEPS = 48


def _traj_b_derivs_costs() -> tuple[list[list[float]], list[float]]:
    derivs = []
    costs = []
    prev_cost = 5.0
    for step in range(1, TRAJ_B_STEPS + 1):
        d1 = 1.0 if step == 1 else (1.0 if step % 2 == 1 else -1.0)
        derivs.append([1.0, d1])
        cost = prev_cost + (1.0 if step % 2 == 0 else -1.0)
        costs.append(cost)
        prev_cost = cost
    return derivs, costs


TRAJ_B_DERIVS, TRAJ_B_COSTS = _traj_b_derivs_costs()


def _rprop_branch_record(all_derivs: list[list[float]], all_costs: list[float], n: int) -> list[list[str]]:
    """Re-derive, per step and per element, which Rprop.cpp:9-59 branch fires --
    an independent state machine over the SAME trajectory inputs the harness
    consumes (not a read-back of the harness's own state). Branch tags:
      init_pos / init_neg / init_zero        (first call, per element sign)
      grow                                    (derivTimesPrev > 0, unclamped)
      grow_clamped                            (derivTimesPrev > 0, hit max_delta)
      shrink+backtrack / shrink+nobacktrack   (derivTimesPrev < 0, cost gate)
      shrink_clamped+backtrack/+nobacktrack   (derivTimesPrev < 0, hit min_delta)
      zero / zero_pos / zero_neg              (derivTimesPrev == 0: literal-zero
                                                current deriv, or the
                                                post-backtrack forced-zero path)
    """
    deltas = [None] * n
    prev_derivs = [None] * n
    prev_cost = None
    record: list[list[str]] = []
    for step in range(len(all_derivs)):
        derivs = all_derivs[step]
        cost = all_costs[step]
        branch = [""] * n
        if deltas[0] is None:
            deltas = [RPROP_INIT_DELTA] * n
            prev_derivs = list(derivs)
            for j in range(n):
                d = derivs[j]
                branch[j] = "init_zero" if d == 0.0 else ("init_pos" if d > 0.0 else "init_neg")
        else:
            dtp = [derivs[j] * prev_derivs[j] for j in range(n)]
            prev_derivs = list(derivs)
            for j in range(n):
                if dtp[j] > 0.0:
                    deltas[j] *= RPROP_ETA_PLUS
                    if deltas[j] > RPROP_MAX_DELTA:
                        deltas[j] = RPROP_MAX_DELTA
                        branch[j] = "grow_clamped"
                    else:
                        branch[j] = "grow"
                elif dtp[j] < 0.0:
                    deltas[j] *= RPROP_ETA_MIN
                    clamped = deltas[j] < RPROP_MIN_DELTA
                    if clamped:
                        deltas[j] = RPROP_MIN_DELTA
                    fired = prev_cost < cost
                    tag = "shrink_clamped" if clamped else "shrink"
                    branch[j] = f"{tag}+{'backtrack' if fired else 'nobacktrack'}"
                    prev_derivs[j] = 0.0
                else:
                    d = derivs[j]
                    branch[j] = "zero" if d == 0.0 else ("zero_pos" if d > 0.0 else "zero_neg")
        prev_cost = cost
        record.append(branch)
    return record


TRAJ_A_BRANCHES = _rprop_branch_record(TRAJ_A_DERIVS, TRAJ_A_COSTS, TRAJ_A_N)
TRAJ_B_BRANCHES = _rprop_branch_record(TRAJ_B_DERIVS, TRAJ_B_COSTS, TRAJ_B_N)

# S11.1 non-vacuity: the seven required branch KINDS (ignoring the +back/nobacktrack
# and _pos/_neg/_clamped element-sign suffixes) that must appear somewhere across the
# two trajectories' branch records.
REQUIRED_BRANCH_KINDS = {
    "init": lambda b: b.startswith("init_"),
    "grow": lambda b: b == "grow",
    "grow_clamped": lambda b: b == "grow_clamped",
    "shrink_backtrack": lambda b: b.startswith("shrink") and b.endswith("+backtrack"),
    "shrink_nobacktrack": lambda b: b.startswith("shrink") and b.endswith("+nobacktrack"),
    "shrink_clamped": lambda b: b.startswith("shrink_clamped"),
    "zero_post": lambda b: b.startswith("zero"),
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
NN_OUTPUTONLY_RE = re.compile(r"^NN_OUTPUTONLY lstm_nonzero=(?P<n>\d+)$", re.MULTILINE)


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


def _read_bin_col1(path: Path) -> list[float]:
    """Read column 1 (the replicated frame count) of an Nx2 .bin (column-major f64)."""
    import struct

    with path.open("rb") as f:
        rows = int.from_bytes(f.read(8), "little", signed=True)
        cols = int.from_bytes(f.read(8), "little", signed=True)
        data = struct.unpack(f"<{rows * cols}d", f.read(8 * rows * cols))
    return list(data[rows : 2 * rows])  # column-major: col1 is the second rows-block


def _parse_nn_tol(stdout: str) -> dict[str, dict[str, object]]:
    tols: dict[str, dict[str, object]] = {}
    for m in NN_TOL_RE.finditer(stdout):
        tols[m["site"]] = {"max_ulp": int(m["max_ulp"]), "max_abs": m["max_abs"]}
    for site in [*SYNTHETIC_TOL_SITES, REAL_TOL_SITE, *DELTASOUT_CALIB_SITES]:
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
        # Task 6: windowed BLSTM backward goldens (the REAL feedForwardBackward derivs).
        for name in WINDOWED_BWD_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Task 3: dense backward deltas_out goldens (same harness stage).
        for name in EXPECTED_DELTASOUT_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Task 5: net-container backward deltas_out goldens (same harness stage).
        for name in NET_BWD_DELTASOUT_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Fix-wave 1 (review finding 1): single-layer container backward goldens.
        for name in (*NET_SINGLE_LAYER_DERIVS_SHAPES, *NET_SINGLE_LAYER_DELTASOUT_SHAPES):
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Task 4: LSTM backward per-variant goldens (deltasPreviousLayer + derivs).
        for name in (*LSTM_BWD_DELTASPREV_SHAPES, *LSTM_BWD_DERIVS_SHAPES):
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Copy the Task 2 CostLaw REAL-probe goldens (dumped from the real class).
        for name in COST_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE3_DIR / name)
        # Produce the brief's literal alias filenames as a byte copy of the
        # poly-named dump (fix-wave 1, review finding 2) -- no second harness
        # dump block, single source of truth.
        for alias_name, source_name in ALIAS_COPIES.items():
            shutil.copy2(PHASE3_DIR / source_name, PHASE3_DIR / alias_name)
        # Task 7: Rprop trajectory dumps (rprop_traj{A,B}_step<k>_{weights,deltas,
        # deltaweights,prevderivs}.bin).
        for traj, n_steps in (("trajA", TRAJ_A_STEPS), ("trajB", TRAJ_B_STEPS)):
            for step in range(1, n_steps + 1):
                for kind in ("weights", "deltas", "deltaweights", "prevderivs"):
                    name = f"rprop_{traj}_step{step}_{kind}.bin"
                    shutil.copy2(tmp_dir / name, PHASE3_DIR / name)

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
    # Task 6: back_prop_output_network_only short-circuit -- the LSTM deriv blocks must
    # stay exactly 0 (only the output net trains).
    outputonly_match = NN_OUTPUTONLY_RE.search(stdout)
    if outputonly_match is None:
        raise SystemExit("NN_OUTPUTONLY lstm_nonzero line missing from harness stdout")
    if int(outputonly_match["n"]) != 0:
        raise SystemExit(f"back_prop_output_network_only leaked {outputonly_match['n']} nonzero LSTM derivs")

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
    lstm_bwd_mismatches = []
    for name, expected in (*LSTM_BWD_DELTASPREV_SHAPES.items(), *LSTM_BWD_DERIVS_SHAPES.items()):
        got = _read_bin_shape(PHASE3_DIR / name)
        if got != expected:
            lstm_bwd_mismatches.append({"file": name, "expected": expected, "got": got})
    if lstm_bwd_mismatches:
        raise SystemExit(f"lstm_bwd_*.bin shape mismatch: {lstm_bwd_mismatches}")
    net_single_mismatches = []
    for name, expected in (*NET_SINGLE_LAYER_DERIVS_SHAPES.items(), *NET_SINGLE_LAYER_DELTASOUT_SHAPES.items()):
        got = _read_bin_shape(PHASE3_DIR / name)
        if got != expected:
            net_single_mismatches.append({"file": name, "expected": expected, "got": got})
    if net_single_mismatches:
        raise SystemExit(f"bwd_net_single_*.bin shape mismatch: {net_single_mismatches}")
    windowed_mismatches = []
    for name, expected in WINDOWED_BWD_SHAPES.items():
        got = _read_bin_shape(PHASE3_DIR / name)
        if got != expected:
            windowed_mismatches.append({"file": name, "expected": expected, "got": got})
    if windowed_mismatches:
        raise SystemExit(f"blstm_bwd_*.bin shape mismatch: {windowed_mismatches}")
    # Task 6 count-vector VERIFICATION (spec S11.4): the col1 trained-weight count in
    # each windowed dump must match the recorded `trained_count`, and the stats tail
    # (last 2*inputSize=4 rows) must be count 1. Reading col1 back from the dump makes
    # the manifest counts MEASURED, not hardcoded-and-hoped.
    count_mismatches = []
    for variant, meta in WINDOWED_BWD_COUNTS.items():
        col1 = _read_bin_col1(PHASE3_DIR / f"blstm_bwd_{variant}_derivs.bin")
        head = len(col1) - 4  # trained head vs the 2*inputSize stats tail
        head_counts = {int(c) for c in col1[:head]}
        tail_counts = {int(c) for c in col1[head:]}
        if head_counts != {meta["trained_count"]}:
            count_mismatches.append({"variant": variant, "expected_trained": meta["trained_count"], "got": sorted(head_counts)})
        if tail_counts != {1}:
            count_mismatches.append({"variant": variant, "expected_tail": 1, "got_tail": sorted(tail_counts)})
    if count_mismatches:
        raise SystemExit(f"windowed backward count-vector mismatch: {count_mismatches}")
    # Non-vacuity of the double-count/overlap semantics: twosweeps + overlap counts must
    # STRICTLY exceed their single-pass baselines (a halved-deriv or single-sweep port
    # would collapse these).
    if WINDOWED_BWD_COUNTS["twosweeps"]["trained_count"] <= WINDOWED_BWD_COUNTS["truncate"]["trained_count"]:
        raise SystemExit("twosweeps count must exceed the single-sweep truncate count")
    if WINDOWED_BWD_COUNTS["overlap"]["trained_count"] <= WINDOWED_BWD_COUNTS["overlap"]["t_len"]:
        raise SystemExit("overlap count must exceed the single-pass frame count")
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
    rprop_mismatches = []
    for traj, n_steps, n in (("trajA", TRAJ_A_STEPS, TRAJ_A_N), ("trajB", TRAJ_B_STEPS, TRAJ_B_N)):
        for step in range(1, n_steps + 1):
            for kind in ("weights", "deltas", "deltaweights", "prevderivs"):
                name = f"rprop_{traj}_step{step}_{kind}.bin"
                got = _read_bin_shape(PHASE3_DIR / name)
                if got != (n, 1):
                    rprop_mismatches.append({"file": name, "expected": (n, 1), "got": got})
    if rprop_mismatches:
        raise SystemExit(f"rprop_traj*.bin shape mismatch: {rprop_mismatches}")

    # S11.1 non-vacuity: every required branch KIND must appear at least once across
    # the two trajectories' branch records.
    all_branches = [b for step in (*TRAJ_A_BRANCHES, *TRAJ_B_BRANCHES) for b in step]
    missing_kinds = [
        kind for kind, pred in REQUIRED_BRANCH_KINDS.items() if not any(pred(b) for b in all_branches)
    ]
    if missing_kinds:
        raise SystemExit(f"Rprop trajectory does not exercise required branch kinds: {missing_kinds}")

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
                "max_ulp=0 all four (now covering BOTH the Nx2 derivs AND the returned "
                "deltas_out per site -- Task 5). The reverse site pins the time-reversed cache "
                "consumption (feedBackwardReverse reverses input/output/deltas, runs the "
                "forward-order backward, reverses the returned deltas). deltas_out row count is "
                "floor(T/R)*R (10, not 11): SubSample floors the trailing row, InvSubSample "
                "never restores it."
            ),
            "net_lstm_backward_fwd": tols["net_lstm_backward_fwd"],
            "net_lstm_backward_rev": tols["net_lstm_backward_rev"],
            "net_dense_backward": tols["net_dense_backward"],
            "deltasout_calibration": {site: tols[site] for site in DELTASOUT_CALIB_SITES},
        },
        "net_single_layer_backward_tol": {
            "text": (
                "Fix-wave 1 (review finding 1): NeuralNetwork<LSTMLayer>::feedBackward on a "
                "SINGLE-layer net (neuronNb.size()==2, NeuralNetwork.hpp:255-263) vs "
                "netBackwardLoop's L==2 branch, LSTM [2,2] at T=7 (odd length). This branch is "
                "REACHABLE in a real config (configs/legacy/LID_BLSTM.config: "
                "BLSTM_LSTMNeuronNb 11,12 with BLSTM_LSTMSubSampling 4) and is structurally "
                "distinct from the multi-layer jj==0 arm (:268-276): the single-layer branch "
                "reads `deltas`/`deltas.rows()` (the SEED) directly every call, never the "
                "running `deltasOut` the multi-layer arm reads on all but the last iteration. "
                "subsample (sub [2]) exercises the SubSample/InvSubSample inversion "
                "(floor(7/2)=3 decimated rows, InvSubSample restores 3*2=6, NOT 7); plain "
                "(sub [1], legacy :262) exercises the no-subsample else-branch. Both max_ulp=0 "
                "(synthetic, k<23 -- below the Eigen-blocked-GEMM divergence threshold) over "
                "BOTH the Nx2 derivs and the returned deltas_out; deltasout is a separate "
                "calibration site per the k=4*O GEMM-order convention (see net_backward_tol), "
                "measured 0 at this shape."
            ),
            "subsample": tols["net_single_layer_backward_subsample"],
            "plain": tols["net_single_layer_backward_plain"],
            "deltasout_calibration": {
                "subsample": tols["net_single_layer_backward_subsample_deltasout"],
                "plain": tols["net_single_layer_backward_plain_deltasout"],
            },
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
            "outratio_calibration": {
                "text": (
                    "Output sub-sampling [2,1] (out ratio 2): getWeightsDerivatives "
                    "multiplies the fwd/bwd LSTM col0 by 2 (:266-269). The output-net "
                    "SubSample/InvSubSample GEMM in the backward diverges Eigen from the "
                    "ascending reimpl (like the real-net site), so the REIMPL derivs are "
                    "the golden (bwd_blstm_outratio_derivs.bin) and the real-vs-reimpl "
                    "gap is recorded here. The Rust test pins the x2 multiplier by "
                    "comparing the fwd/bwd blocks vs this dump AND vs the unscaled "
                    "single-ratio site."
                ),
                **tols["blstm_outratio"],
            },
        },
        "blstm_windowed_backward": {
            "text": (
                "Task 6: the WINDOWED BLSTMNeuralNetwork::feedForwardBackward backward "
                "(BLSTMNeuralNetwork.cpp:711-830) per processing type, on the windowed "
                "synthetic net LSTM [2,2] sub [1] + output [4,2] sub [1] "
                "(sub_sampling_ratio == 1: no grid snapping/decimation -> the col1 count "
                "vector is hand-computable). The golden is the REAL compiled "
                "getWeightsDerivatives (the per-window worker's backward is at synthetic "
                "k < 23 where Eigen == the ascending reimpl -- the plain site is "
                "max_ulp=0; the windowed drivers are pure structural loops over that "
                "worker). Each variant's NN_TOL (site=blstm_bwd_<variant>) records the "
                "reimpl vs the real net -- all max_ulp=0. The count vector (col1) proves "
                "spec S6/S11.4: TwoSweeps back-propagates BOTH sweeps so the trained-"
                "weight count is 34 (> the single-sweep truncate 12) with the derivs NOT "
                "halved (the count carries the /2 forward-average compensation); OverLap "
                "back-propagates each covering window so the count is 27 (> the "
                "single-pass 12); the mean/std stats tail is always count 1. The enforce "
                "variant (TargetEnforcementStep -1) rewrites the per-window target "
                "interior to -0.5 BEFORE feed_backward (risk R7); its count stays 8 (the "
                "enforcement moves the deltas, not the frame count). blstm_bwd_outputonly "
                "(BackPropOutputNetworkOnly true) is not dumped as a golden -- the harness "
                "asserts NN_OUTPUTONLY lstm_nonzero=0 (the LSTM deriv blocks stay 0) and "
                "the Rust test builds it directly."
            ),
            "variants": {
                variant: {
                    **meta,
                    "nn_tol": tols[f"blstm_bwd_{variant}"],
                }
                for variant, meta in WINDOWED_BWD_COUNTS.items()
            },
            "stats_tail_count": 1,
            "output_only_nn_tol": tols["blstm_bwd_outputonly"],
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
        "lstm_backward_variants_tol": {
            "text": (
                "Task 4: LSTMLayer::feedBackward / feedBackwardReverse (LSTMLayer.cpp:"
                "518-732) per-variant goldens at synthetic I=2,O=2,T=5 (odd length). "
                "Variants: peep_all (all three peephole flags on), peep_none (all off -- "
                "the peephole cross-terms drop entirely), reverse (feedForwardReverse then "
                "feedBackwardReverse: reverse input/output/deltas, run the forward-order "
                "backward, reverse returned deltas; the caches are stored time-reversed and "
                "consumed as-is), subsample (invSubSamplingRatio=2, asserting the *2 block "
                "scaling on all four deriv blocks, :710-715). Each dumps deltasPreviousLayer "
                "(T x I) + the Nx2 derivs (col0 summed deriv, col1 frame count). The signal "
                "variant feeds a 1-col input into the I=2 layer (width tolerance :534-543, "
                "zero-pad to I rows): dead weight rows accumulate exactly 0 (spec S11.5), "
                "gradient stays layer-native (lstm_bwd_signal_derivs.bin). max_ulp=0 vs the "
                "real compiled layer over BOTH derivs and deltasPreviousLayer."
            ),
            **tols["lstm_backward_variants"],
        },
        "dumps": {
            **{name: {"rows": r, "cols": c} for name, (r, c) in EXPECTED_SHAPES.items()},
            **{name: {"rows": r, "cols": c} for name, (r, c) in EXPECTED_DELTASOUT_SHAPES.items()},
            **{name: {"rows": r, "cols": c} for name, (r, c) in NET_BWD_DELTASOUT_SHAPES.items()},
            **{name: {"rows": r, "cols": c} for name, (r, c) in LSTM_BWD_DELTASPREV_SHAPES.items()},
            **{name: {"rows": r, "cols": c} for name, (r, c) in LSTM_BWD_DERIVS_SHAPES.items()},
            **{name: {"rows": r, "cols": c} for name, (r, c) in NET_SINGLE_LAYER_DERIVS_SHAPES.items()},
            **{name: {"rows": r, "cols": c} for name, (r, c) in NET_SINGLE_LAYER_DELTASOUT_SHAPES.items()},
            **{name: {"rows": r, "cols": c} for name, (r, c) in WINDOWED_BWD_SHAPES.items()},
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
        "rprop_trajectories": {
            "text": (
                "Task 7: the REAL compiled Rprop::updateWeights (Rprop.cpp:9-59, the "
                ":60-76 .mat dump + _Count are dead and not exercised) is the golden "
                "DIRECTLY -- pure scalar, no libm/GEMM, bit-portable on every platform "
                "(STRICT BITS everywhere, no canary gate). Trajectory A (5 elements, 5 "
                "steps) exercises every branch except the delta clamps: first-call init "
                "(deriv>0/deriv<0/deriv==0), eta+ growth, eta- shrink with the "
                "cost-gated backtrack BOTH firing (step3, cost rises 9->12) and NOT "
                "firing (step4, cost falls 12->8), the post-backtrack prev_derivs==0 "
                "path (both with a nonzero and a literal-zero current deriv), and a "
                "literal subsequent-call deriv==0. Trajectory B (2 elements, 48 steps) "
                "is a DEDICATED clamp trajectory (the brief's 'second trajectory with "
                "init_delta near a clamp' alternative): element 0's deriv never flips, "
                "growing 1e-2*1.2^k to the max_delta=0.2 clamp (hit at step 18); "
                "element 1 alternates sign every step, so only every OTHER call is a "
                "genuine shrink (the intervening calls are the forced-zero "
                "post-backtrack path), reaching the min_delta=1e-9 clamp at step 48. "
                "The branch record below is an INDEPENDENT re-derivation (this "
                "extractor script, not read back from the dumps) of the deriv-sign x "
                "prev-deriv-sign x cost-comparison state machine over the exact same "
                "trajectory inputs fed to the harness -- the Rust golden test asserts "
                "it covers all 7 required branch kinds (S11.1) and cross-checks a few "
                "spot elements/steps against the dumped numbers."
            ),
            "eta_min": RPROP_ETA_MIN,
            "eta_plus": RPROP_ETA_PLUS,
            "min_delta": RPROP_MIN_DELTA,
            "max_delta": RPROP_MAX_DELTA,
            "init_delta": RPROP_INIT_DELTA,
            "trajectory_a": {
                "n_elements": TRAJ_A_N,
                "n_steps": TRAJ_A_STEPS,
                "derivs": TRAJ_A_DERIVS,
                "costs": TRAJ_A_COSTS,
                "branches": TRAJ_A_BRANCHES,
            },
            "trajectory_b": {
                "n_elements": TRAJ_B_N,
                "n_steps": TRAJ_B_STEPS,
                "derivs": TRAJ_B_DERIVS,
                "costs": TRAJ_B_COSTS,
                "branches": TRAJ_B_BRANCHES,
                "max_delta_clamp_first_step": next(
                    step for step, b in enumerate(TRAJ_B_BRANCHES, start=1) if "grow_clamped" in b
                ),
                "min_delta_clamp_first_step": next(
                    step for step, b in enumerate(TRAJ_B_BRANCHES, start=1) if any("shrink_clamped" in x for x in b)
                ),
            },
            "required_branch_kinds_covered": sorted(REQUIRED_BRANCH_KINDS),
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
