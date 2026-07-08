"""Run the Phase 4c Octave harness stages and write the SMORMS3 + Rprop + batching +
CheckGrad + MaskingValidation bit-pin fixtures.

This is the MATLAB-side analog of `scripts/extract_phase4b_fixtures.py` (which drives
the C++ oracle harness): instead of compiling and running vendored C++, it locates a
local `octave-cli`, runs `tools/octave_harness/run_stage.m` over each stage into a
throwaway tempdir, and converts the Octave `.mat` dumps to the io::binary `.bin` format
the Rust/Python codec reads. GNU Octave is a LOCAL-ONLY dependency (like libmatio for
the C++ harness): `brew install octave`. CI never runs this extractor -- it consumes
only the committed `.bin`/`manifest.json` fixtures.

Stages (`tools/octave_harness/stage_*.m`), each driving the REAL vendored .m unchanged
(TIER 1: real classdef / real function; see the harness README + IMPROVEMENTS for the
one Octave-compat accommodation -- the SMORMS3 empty-varargin lvalue miscount):

  smorms3 : SMORMS3.m (classdef) per-step state trajectories over a scripted gradient
            sequence. Main run: 2-cell theta (M=7), 12 steps -- proves the lrate x10
            warmup (1e-9 -> 1e-1 by step 8) AND the 1e-1 cap (steps 9-12 plateau).
            Round-trip run: f_df returns a MODIFIED theta_out (nonzero offset), pinning
            SMORMS3.m:355's `theta = theta_out + dtheta`.
            Eps-guard run: a near-zero gradient (M=1, 3 steps, grad=1e-8 constant) makes
            step-1 MMS ~ eps=1e-16 (same order, not rounding noise), genuinely
            exercising the two `+eps` guards (SMORMS3.m:335) that the main/round-trip
            sequences leave inactive (MMS >> eps there).
  rprop   : Rprop.m (function) per-step state over a crafted derivative/cost sequence
            exercising every branch (grow / shrink+backtrack / plain) -- pure arithmetic,
            STRICT bits everywhere.
  batching: CreateBatches.m + GetNewBatch.m (TIER 1) + the FALLBACK-TIER
            getworstandbest_transcribed.m (verbatim transcription of getCases.m's private
            getWorstAndBest subfunction -- unreachable standalone). randperm is SHADOWED
            with a fixed reverse-order permutation (batching_shadow/randperm.m) so
            CreateBatches's shuffle is reproducible AND trivially replicable in Python.
            Five CreateBatches cases (single / multi_nb / clobber / sub / degenerate) plus
            rotation traces and two getWorstAndBest cases -- PURE integer arithmetic
            throughout (no libm), STRICT bits everywhere.
  checkgrad: CheckGrad.m (TIER 1) driven with a harness-local quadratic-surrogate
            CostFunction.m shadow (re-deriving the real flat NN weight vector from param
            via the REAL vec2struct + nnet2MatFile on every call) plus no-op
            figure/subplot/semilogy/hold/grid shadows (CheckGrad's diagnostic plot errors
            under this Octave/graphics-toolkit combination). Surfaces a genuine legacy bug
            (weights2nnet.m never writes back the normalize mean/std tail of the `weights`
            vector nnet2MatFile.m appends -- CheckGrad's last 2 numeric derivatives are
            always exactly 0 regardless of the analytic value).
  masking : MaskingValidation.m + vec2struct.m (TIER 1), a PASS case (a well-formed scalar
            mask) and a FAIL case (masking a `_padding_block`-family vector field with a
            negative component, exploiting that field's encode/decode asymmetry).
  qpso    : QuantumPSO.m (845 LOC) via a MODIFIED-COPY (tools/octave_harness/qpso_modified/)
            whose rand/randperm/stblrnd are substituted by reads of a shared committed random
            table (generated here, fixed numpy seed) and whose CostFunction is a quadratic
            surrogate; the wall-clock reseed is removed. Dumps the post-init state + per-epoch
            pos/pbest/gbest trajectory + a standalone Levy (Chambers-Mallows-Stuck) sample. The
            Python port (speech.optimizers.quantum_pso) replays the SAME table. init_* is STRICT
            (pure arithmetic); the trajectory is canary-gated (transcendental via log(1/u)+Levy).

Determinism: only `.bin` + `manifest.json` are committed (the `.mat` MAT-v7 header
carries a churning timestamp; we never copy it). The `.bin` payloads and the manifest
(MEASURED octave version + step counts + shapes, no wall-clock) are deterministic, so
running this extractor TWICE yields byte-identical committed output. A prior-phase hash
guard (phase0..phase4b) SystemExits on any regression.

Usage: uv run python scripts/extract_phase4c_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "tools" / "octave_harness"
REF_DIR = REPO_ROOT / "tests" / "reference_data"
PHASE4C_DIR = REF_DIR / "phase4c"

# Prior-phase fixture dirs the extractor must NOT perturb (regression guard).
PRIOR_PHASES = ["phase0", "phase0b", "phase0bii", "phase1", "phase2", "phase2b", "phase3", "phase4a", "phase4b"]

# Per-stage: the octave var -> expected (rows, cols). The `.bin` name is f"{stage}_{var}.bin".
SMORMS3_SHAPES = {
    "grad_script": (7, 12),
    "theta0_flat": (7, 1),
    "theta_traj": (7, 12),
    "mms_traj": (7, 12),
    "steprate_traj": (7, 12),
    "delta_traj": (7, 12),
    "lrate_traj": (1, 12),
    "cost_traj": (1, 12),
    "theta_final_cell0": (3, 1),
    "theta_final_cell1": (2, 2),
    "rt_grad_script": (3, 4),
    "rt_theta0_flat": (3, 1),
    "rt_offset": (3, 1),
    "rt_theta_traj": (3, 4),
    "eps_grad_script": (1, 3),
    "eps_theta0_flat": (1, 1),
    "eps_theta_traj": (1, 3),
    "eps_mms_traj": (1, 3),
    "eps_steprate_traj": (1, 3),
    "eps_delta_traj": (1, 3),
    "eps_lrate_traj": (1, 3),
}
# The near-zero-gradient regime targets: SMORMS3.m:79/:335's guard value, and the
# "genuinely active" band the non-vacuity check below requires step-1 MMS to fall in.
SMORMS3_EPS = 1e-16
EPS_GUARD_RATIO_BAND = (1e-2, 1e2)  # within 2 orders of magnitude of SMORMS3_EPS
RPROP_SHAPES = {
    "deriv_script": (5, 6),
    "cost_script": (1, 6),
    "weights0": (5, 1),
    "weights_traj": (5, 6),
    "delta_traj": (5, 6),
    "deltaweight_traj": (5, 6),
    "derivout_traj": (5, 6),
}
BATCHING_SHAPES = {
    "single_index": (6, 1),
    "single_worst_index": (1, 1),
    "single_batch_traj": (2, 8),
    "single_validation_traj": (6, 8),
    "single_countpass_traj": (1, 8),
    "multi_nb_case1_index": (2, 1),
    "multi_nb_case2_index": (2, 1),
    "multi_nb_case3_index": (2, 1),
    "multi_nb_worst1_index": (1, 1),
    "multi_nb_worst2_index": (1, 1),
    "multi_nb_worst3_index": (1, 1),
    "multi_nb_batch_traj": (2, 8),
    "multi_nb_validation_traj": (6, 8),
    "multi_nb_countpass_traj": (1, 8),
    "clobber_case3_index": (2, 1),
    "sub_case1_index": (3, 1),
    "sub_case2_index": (3, 1),
    "sub_case3_sub1_index": (3, 1),
    "sub_case3_sub2_index": (3, 1),
    "sub_worst1_index": (1, 1),
    "sub_worst2_index": (1, 1),
    "sub_worst3_index": (1, 1),
    "sub_batch_traj": (2, 10),
    "sub_validation_traj": (12, 10),
    "sub_countpass_traj": (1, 10),
    "degenerate_cases": (3, 1),
    "degenerate_batch": (3, 1),
    "degenerate_validation": (3, 1),
    "gwb_basic_indBest": (2, 1),
    "gwb_basic_indMiddle": (1, 1),
    "gwb_basic_indWorst": (3, 1),
    "gwb_basic_scoreBest": (2, 1),
    "gwb_basic_scoreMiddle": (1, 1),
    "gwb_basic_scoreWorst": (3, 1),
    "gwb_edge_indBest": (0, 0),
    "gwb_edge_indMiddle": (0, 0),
    "gwb_edge_indWorst": (0, 0),
}
CHECKGRAD_SHAPES = {
    "nnet_in": (107, 1),
    "MultiWeights": (53, 1),
    "MultiDeriv_BackProp": (53, 1),
    "MultiDeriv_Num": (53, 1),
}
MASKING_SHAPES = {
    "pass_hasFailed": (1, 1),
    "pass_count": (1, 1),
    "pass_vector": (265, 1),
    "fail_hasFailed": (1, 1),
    "fail_count": (1, 1),
    "fail_vector": (265, 1),
}
# --- Task 11: QuantumPSO shared random table + trajectory ---
# The table is generated ONCE with a FIXED numpy seed and committed. Regeneration must be
# byte-identical: numpy's default_rng (PCG64) is version-stable for a given seed, and
# `.random(N)` draws N uniforms in [0,1). BOTH the Octave modified-copy (qpso_modified/) and
# the Python port (speech.optimizers.TableRng) read this same table sequentially, column-major.
QPSO_TABLE_SEED = 20260709
QPSO_TABLE_N = 5000
# The stage's small-D/ps/me run and standalone Levy dump (stage_qpso.m). D=3, ps=4, me=3, K=8.
QPSO_D, QPSO_PS, QPSO_ME, QPSO_K = 3, 4, 3, 8
QPSO_SHAPES = {
    "levy_in": (2 * QPSO_K, 1),  # the V(8)+W(8) input slice table[0:16]
    "levy_out": (1, QPSO_K),
    "init_pos": (QPSO_PS, QPSO_D),
    "init_pbest": (QPSO_PS, QPSO_D),
    "init_pbestval": (QPSO_PS, 1),
    "init_gbest": (1, QPSO_D),
    "init_gbestval": (1, 1),
    "pos_traj": (QPSO_PS, QPSO_D * QPSO_ME),  # horzcat of per-epoch pos
    "pbest_traj": (QPSO_PS, QPSO_D * QPSO_ME),
    "pbestval_traj": (QPSO_PS, QPSO_ME),
    "gbest_traj": (QPSO_ME, QPSO_D),  # vertcat of per-epoch gbest rows
    "gbestval_traj": (1, QPSO_ME),
    "final_out": (QPSO_D + 1, 1),  # OUT = [gbest'; gbestval]
    "cursor_end": (1, 1),  # draws consumed (== Python TableRng.cursor)
}

STAGE_LINE_RE = re.compile(
    r"^OCTAVE_STAGE smorms3 M=(?P<M>\d+) num_steps=(?P<ns>\d+) rt_M=(?P<rtM>\d+) rt_num_steps=(?P<rtns>\d+) "
    r"eps_M=(?P<epsM>\d+) eps_num_steps=(?P<epsns>\d+)$",
    re.MULTILINE,
)
RPROP_LINE_RE = re.compile(r"^OCTAVE_STAGE rprop N=(?P<N>\d+) K=(?P<K>\d+)$", re.MULTILINE)
BATCHING_LINE_RE = re.compile(
    r"^OCTAVE_STAGE batching single_n=(?P<single_n>\d+) multi_nb_n=(?P<multi_nb_n>\d+) "
    r"sub_n=(?P<sub_n>\d+) degenerate_batch_len=(?P<deg_len>\d+)$",
    re.MULTILINE,
)
CHECKGRAD_LINE_RE = re.compile(r"^OCTAVE_STAGE checkgrad n=(?P<n>\d+) Kw=(?P<Kw>\d+)$", re.MULTILINE)
MASKING_LINE_RE = re.compile(r"^OCTAVE_STAGE masking pass_failed=(?P<pass_failed>\d+) fail_failed=(?P<fail_failed>\d+)$", re.MULTILINE)
QPSO_LINE_RE = re.compile(
    r"^OCTAVE_STAGE qpso D=(?P<D>\d+) ps=(?P<ps>\d+) me=(?P<me>\d+) K=(?P<K>\d+) "
    r"cursor_end=(?P<cursor>\d+) te=(?P<te>\d+)$",
    re.MULTILINE,
)


def _find_octave() -> str:
    octave = shutil.which("octave-cli") or shutil.which("octave")
    if octave is None:
        raise SystemExit("octave-cli not found on PATH -- install the local-only dep: brew install octave")
    return octave


def _octave_version(octave: str) -> str:
    out = subprocess.run([octave, "--version"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"version (\S+)", out)
    if not m:
        raise SystemExit(f"could not parse octave version from: {out!r}")
    return m.group(1)


def _run_stage(octave: str, stage: str, out_dir: Path) -> str:
    result = subprocess.run(
        [
            octave,
            "--no-gui",
            "--quiet",
            "--path",
            str(HARNESS_DIR),
            "--eval",
            f"run_stage('{stage}', '{out_dir}')",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"octave stage {stage} failed ({result.returncode})")
    return result.stdout


def _write_bin(path: Path, matrix: np.ndarray) -> None:
    """Write `matrix` as an io::binary .bin: i64 LE rows, i64 LE cols, f64 LE column-major."""
    mat = np.atleast_2d(np.ascontiguousarray(matrix, dtype="<f8"))
    rows, cols = mat.shape
    with path.open("wb") as f:
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        f.write(mat.flatten(order="F").tobytes())


def _read_bin(path: Path) -> tuple[int, int, list[float]]:
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    n = rows * cols
    data = list(struct.unpack(f"<{n}d", raw[16 : 16 + 8 * n]))
    return rows, cols, data


def _convert_stage(mat_path: Path, stage: str, shapes: dict[str, tuple[int, int]], out_dir: Path) -> None:
    """Convert every expected variable of a stage .mat to `<stage>_<var>.bin`, asserting
    each matches its expected shape. Writes into `out_dir` (the tempdir) -- the copy into
    the committed PHASE4C_DIR happens only after every check passes."""
    mat = scipy.io.loadmat(mat_path)
    for var, (er, ec) in shapes.items():
        if var not in mat:
            raise SystemExit(f"{stage}: variable {var} missing from {mat_path.name}")
        m = np.atleast_2d(np.asarray(mat[var], dtype=np.float64))
        if m.shape != (er, ec):
            raise SystemExit(f"{stage}: {var} shape {m.shape} != expected {(er, ec)}")
        _write_bin(out_dir / f"{stage}_{var}.bin", m)


def _hash_tree(root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    if not root.is_dir():
        return digests
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def main() -> None:
    octave = _find_octave()
    octave_version = _octave_version(octave)

    before = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
    PHASE4C_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # The QuantumPSO stage reads a shared random table from its out_dir; generate it FIRST
        # (fixed seed -> byte-identical on every run) so the Octave stage can consume it.
        qpso_table = np.random.default_rng(QPSO_TABLE_SEED).random(QPSO_TABLE_N)
        _write_bin(tmp_dir / "qpso_random_table.bin", qpso_table.reshape(QPSO_TABLE_N, 1))
        smorms3_stdout = _run_stage(octave, "smorms3", tmp_dir)
        rprop_stdout = _run_stage(octave, "rprop", tmp_dir)
        batching_stdout = _run_stage(octave, "batching", tmp_dir)
        checkgrad_stdout = _run_stage(octave, "checkgrad", tmp_dir)
        masking_stdout = _run_stage(octave, "masking", tmp_dir)
        qpso_stdout = _run_stage(octave, "qpso", tmp_dir)

        sm = STAGE_LINE_RE.search(smorms3_stdout)
        if not sm:
            raise SystemExit("OCTAVE_STAGE smorms3 line missing from stdout")
        rm = RPROP_LINE_RE.search(rprop_stdout)
        if not rm:
            raise SystemExit("OCTAVE_STAGE rprop line missing from stdout")
        bm = BATCHING_LINE_RE.search(batching_stdout)
        if not bm:
            raise SystemExit("OCTAVE_STAGE batching line missing from stdout")
        cgm = CHECKGRAD_LINE_RE.search(checkgrad_stdout)
        if not cgm:
            raise SystemExit("OCTAVE_STAGE checkgrad line missing from stdout")
        mkm = MASKING_LINE_RE.search(masking_stdout)
        if not mkm:
            raise SystemExit("OCTAVE_STAGE masking line missing from stdout")
        qm = QPSO_LINE_RE.search(qpso_stdout)
        if not qm:
            raise SystemExit("OCTAVE_STAGE qpso line missing from stdout")
        if (int(qm["D"]), int(qm["ps"]), int(qm["me"]), int(qm["K"])) != (QPSO_D, QPSO_PS, QPSO_ME, QPSO_K):
            raise SystemExit(f"qpso stage dims {(qm['D'], qm['ps'], qm['me'], qm['K'])} != {(QPSO_D, QPSO_PS, QPSO_ME, QPSO_K)}")
        sm_dims = (int(sm["M"]), int(sm["ns"]), int(sm["rtM"]), int(sm["rtns"]), int(sm["epsM"]), int(sm["epsns"]))
        if sm_dims != (7, 12, 3, 4, 1, 3):
            raise SystemExit(f"smorms3 stage dims {sm_dims} != (7,12,3,4,1,3)")
        if (int(rm["N"]), int(rm["K"])) != (5, 6):
            raise SystemExit(f"rprop stage dims {(rm['N'], rm['K'])} != (5,6)")
        if (int(cgm["n"]), int(cgm["Kw"])) != (107, 53):
            raise SystemExit(f"checkgrad stage dims {(cgm['n'], cgm['Kw'])} != (107,53)")

        _convert_stage(tmp_dir / "smorms3.mat", "smorms3", SMORMS3_SHAPES, tmp_dir)
        _convert_stage(tmp_dir / "rprop.mat", "rprop", RPROP_SHAPES, tmp_dir)
        _convert_stage(tmp_dir / "batching.mat", "batching", BATCHING_SHAPES, tmp_dir)
        _convert_stage(tmp_dir / "checkgrad.mat", "checkgrad", CHECKGRAD_SHAPES, tmp_dir)
        _convert_stage(tmp_dir / "masking.mat", "masking", MASKING_SHAPES, tmp_dir)
        _convert_stage(tmp_dir / "qpso.mat", "qpso", QPSO_SHAPES, tmp_dir)

        # --- Non-vacuity: the goldens must actually exercise the pinned behaviors. ---
        # SMORMS3 lrate x10 warmup + 1e-1 cap: 1e-8..1e-1 then plateau.
        _, _, lrate = _read_bin(tmp_dir / "smorms3_lrate_traj.bin")
        expected_lrate = []
        _lr = 1e-9
        for _ in range(12):  # the iterated min(lrate*10, 1e-1) -- matches SMORMS3.m:347 bit-for-bit
            _lr = min(_lr * 10.0, 1e-1)
            expected_lrate.append(_lr)
        if lrate != expected_lrate:
            raise SystemExit(f"smorms3 lrate_traj {lrate} != expected x10-warmup/cap {expected_lrate}")
        if lrate[7] != 1e-1 or lrate[8] != 1e-1:
            raise SystemExit("smorms3 lrate must reach the 1e-1 cap by step 8 and plateau")
        # Round-trip offset is nonzero (else theta_out == theta_in and the pin is vacuous).
        _, _, rt_offset = _read_bin(tmp_dir / "smorms3_rt_offset.bin")
        if all(v == 0.0 for v in rt_offset):
            raise SystemExit("smorms3 rt_offset all-zero -- the theta_out round-trip pin would be vacuous")
        # Per-dim MMS distinct at step 1 (the vectorized EMA is genuinely per-dimension).
        mr, mc, mms = _read_bin(tmp_dir / "smorms3_mms_traj.bin")
        step1_mms = [mms[c * mr + r] for r in range(mr) for c in [0]]
        if len(set(step1_mms)) < mr:
            raise SystemExit("smorms3 step-1 MMS not per-dim distinct -- gradient script too uniform")
        # Eps-guard non-vacuity: step-1 MMS must be genuinely the same order as
        # SMORMS3_EPS (not rounding noise) for the eps=1e-16 guards to actually engage.
        _, _, eps_mms = _read_bin(tmp_dir / "smorms3_eps_mms_traj.bin")
        eps_step1_mms = eps_mms[0]
        eps_ratio = eps_step1_mms / SMORMS3_EPS
        if not (EPS_GUARD_RATIO_BAND[0] <= eps_ratio <= EPS_GUARD_RATIO_BAND[1]):
            raise SystemExit(
                f"smorms3 eps-guard step-1 MMS={eps_step1_mms} not within 2 orders of magnitude of "
                f"eps={SMORMS3_EPS} (ratio={eps_ratio}) -- eps-guard fixture would be vacuous"
            )

        # RPROP: both the etap-grow (>0.01) and etam-shrink (<0.01) branches fired, and the
        # sign-flip derivative-zeroing produced a 0 where the input derivative was nonzero.
        dr, dc, delta = _read_bin(tmp_dir / "rprop_delta_traj.bin")
        if not (max(delta) > 0.01 and min(delta) < 0.01):
            raise SystemExit(f"rprop delta_traj [{min(delta)}, {max(delta)}] did not span the 0.01 delta0 both ways")
        _, _, deriv_in = _read_bin(tmp_dir / "rprop_deriv_script.bin")
        _, _, deriv_out = _read_bin(tmp_dir / "rprop_derivout_traj.bin")
        if not any(o == 0.0 and i != 0.0 for i, o in zip(deriv_in, deriv_out, strict=True)):
            raise SystemExit("rprop derivout never zeroed a nonzero derivative -- the <0 sign-flip branch never fired")

        # BATCHING: the clobber case really did lose class-0's aggregate (Cases(3) ends up
        # holding class-2's data, [6,5] 1-based -- NOT class-0's [2,1]).
        _, _, clobber = _read_bin(tmp_dir / "batching_clobber_case3_index.bin")
        if sorted(clobber) != [5.0, 6.0]:
            raise SystemExit(f"batching clobber non-vacuity: clobber_case3_index {clobber} != class-2's [5,6] -- the clobber quirk did not fire")
        # Rotation traces must be non-trivial (the batch actually varies across steps, not
        # stuck repeating the same pair every call).
        for case, ncols in (("single", 8), ("multi_nb", 8), ("sub", 10)):
            r, c, traj = _read_bin(tmp_dir / f"batching_{case}_batch_traj.bin")
            cols = [tuple(traj[col * r : (col + 1) * r]) for col in range(ncols)]
            if len(set(cols)) < 2:
                raise SystemExit(f"batching {case} non-vacuity: batch_traj never changes across {ncols} steps")
        # Degenerate: Batch must equal Cases verbatim (GetNewBatch.m:3-5), not a rotation.
        _, _, deg_cases = _read_bin(tmp_dir / "batching_degenerate_cases.bin")
        _, _, deg_batch = _read_bin(tmp_dir / "batching_degenerate_batch.bin")
        if deg_cases != deg_batch:
            raise SystemExit(f"batching degenerate non-vacuity: batch {deg_batch} != cases {deg_cases}")

        # CHECKGRAD: the analytic/numeric agreement is non-vacuous (both nonzero, close)
        # for the well-behaved prefix, AND the discovered weights2nnet.m normalize-tail bug
        # (last 2 entries) genuinely shows numeric==0 with analytic!=0 -- else the pinned
        # quirk would be silently absent from the fixture.
        _, _, cg_analytic = _read_bin(tmp_dir / "checkgrad_MultiDeriv_BackProp.bin")
        _, _, cg_numeric = _read_bin(tmp_dir / "checkgrad_MultiDeriv_Num.bin")
        if not all(abs(a - n) < 1e-3 for a, n in zip(cg_analytic[:-2], cg_numeric[:-2], strict=True)):
            raise SystemExit("checkgrad non-vacuity: the well-behaved prefix's analytic/numeric derivatives do not agree")
        if cg_numeric[-1] != 0.0 or cg_numeric[-2] != 0.0:
            raise SystemExit(f"checkgrad non-vacuity: expected the normalize-tail bug (numeric==0) at the last 2 entries, got {cg_numeric[-2:]}")
        if cg_analytic[-1] == 0.0 or cg_analytic[-2] == 0.0:
            raise SystemExit("checkgrad non-vacuity: analytic tail is zero -- the quirk contrast (nonzero analytic vs zero numeric) would be vacuous")

        # MASKING: pass must not fail, fail must fail (else both cases collapse to the same
        # outcome and the golden pins nothing).
        if (int(mkm["pass_failed"]), int(mkm["fail_failed"])) != (0, 1):
            raise SystemExit(f"masking non-vacuity: (pass_failed, fail_failed) = {(mkm['pass_failed'], mkm['fail_failed'])} != (0, 1)")

        # QPSO: the trajectory must genuinely exercise the operators, not sit on a fixed point.
        cursor_line = int(qm["cursor"])
        _, _, cursor_bin = _read_bin(tmp_dir / "qpso_cursor_end.bin")
        if int(cursor_bin[0]) != cursor_line:
            raise SystemExit(f"qpso cursor_end .bin {cursor_bin[0]} != stdout {cursor_line}")
        # Base draws with NO DE/Levy candidates: init (6*ps*D) + me*(8 dead banks + phi/u/dead-sign
        # (3) + live-sign (1) = 12*ps*D, plus ps roulette wheels). DE recombination and the Levy
        # kick MUST have fired at least once (extra draws beyond base), else those arms are vacuous.
        base_draws = 6 * QPSO_PS * QPSO_D + QPSO_ME * (12 * QPSO_PS * QPSO_D + QPSO_PS)
        if cursor_line <= base_draws:
            raise SystemExit(f"qpso non-vacuity: cursor_end {cursor_line} <= base {base_draws} -- DE/Levy arms never fired")
        # gbest must EVOLVE across epochs (else the seed sits on the surrogate optimum and the
        # gbest pin is trivial), and every epoch's gbestval must be finite.
        gr, gc, gbest_traj = _read_bin(tmp_dir / "qpso_gbest_traj.bin")  # me x D column-major
        gbest_rows = [tuple(gbest_traj[c * gr + i] for c in range(gc)) for i in range(gr)]
        if len(set(gbest_rows)) < 2:
            raise SystemExit("qpso non-vacuity: gbest never changed across epochs (seed on the optimum?)")
        _, _, gbestval_traj = _read_bin(tmp_dir / "qpso_gbestval_traj.bin")
        if not all(np.isfinite(v) for v in gbestval_traj):
            raise SystemExit(f"qpso gbestval_traj not all finite: {gbestval_traj}")
        # Ranking robustness off the oracle libm: the per-epoch (gbestval + pbestval) values must
        # be well-separated (min gap >> any plausible libm ULP perturbation), so no sortrows /
        # min tie can flip discretely under a different libm. Measured gaps here are O(0.1)-O(1).
        pr, pc, pbestval_traj = _read_bin(tmp_dir / "qpso_pbestval_traj.bin")  # ps x me
        min_gap = float("inf")
        for k in range(QPSO_ME):
            vals = sorted([gbestval_traj[k]] + [pbestval_traj[k * pr + i] for i in range(pr)])
            gaps = [b - a for a, b in zip(vals[:-1], vals[1:], strict=True) if b - a > 0]
            if gaps:
                min_gap = min(min_gap, min(gaps))
        if min_gap < 1e-3:
            raise SystemExit(f"qpso ranking-robustness: min (gbestval+pbestval) gap {min_gap} < 1e-3 -- a libm flip could reorder the swarm")
        # Levy: the standalone CMS sample must be nonzero + finite (not a degenerate all-zero draw).
        _, _, levy_out = _read_bin(tmp_dir / "qpso_levy_out.bin")
        if not (all(np.isfinite(v) for v in levy_out) and any(v != 0.0 for v in levy_out)):
            raise SystemExit(f"qpso levy_out degenerate (all-zero or non-finite): {levy_out}")

        # Regression guard.
        after = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
        for ph in PRIOR_PHASES:
            drifted = sorted(n for n in set(before[ph]) | set(after[ph]) if before[ph].get(n) != after[ph].get(n))
            if drifted:
                raise SystemExit(f"REGRESSION: {ph} fixtures changed after the octave run: {drifted}")

        manifest = {
            "text": (
                "Phase 4c Task 6: the Octave harness foundation + the SMORMS3/Rprop bit-pins. "
                "GNU Octave executes the VENDORED legacy .m sources (legacy/Optimizer_V6.2.2/"
                "functions/{SMORMS3,Rprop}.m) to dump per-step optimizer state trajectories -- the "
                "MATLAB-side analog of the C++ oracle harness. LOCAL-ONLY dep (brew install octave); "
                "CI consumes only these committed fixtures. TIER 1 (real classdef / real function) for "
                "BOTH stages; the sole Octave accommodation is one value-neutral dummy varargin passed "
                "to SMORMS3 (the empty-cell lvalue cs-list miscount at SMORMS3.m:313; the gradient "
                "depends only on eval_count and the update math is untouched)."
            ),
            "octave_version": octave_version,
            "smorms3": {
                "text": (
                    "SMORMS3.m per-step state (:324-363). Main run: 2-cell theta0 "
                    "(3x1 vector + 2x2 matrix, M=7), 12 steps, pass-through theta_out; scripted "
                    "grad_script (per-dim-distinct, step-to-step sign flips). Per-field libm status "
                    "(the CROSS-LIBM rule): mms/steprate/delta/lrate are PURE arithmetic (EMA + mul + "
                    "add + min/max) -> STRICT bits on every platform; theta/dtheta traverse "
                    "sqrt(MMS) (a libm call, :335) -> canary-gated off the oracle env "
                    "(tests/_libm_gate.py). lrate_traj proves the x10 geometric warmup (1e-9 -> 1e-1 "
                    "by step 8) capped at 1e-1 (:347). Round-trip run (rt_*): f_df returns "
                    "theta_out = theta + rt_offset (nonzero), pinning `theta = theta_out + dtheta` "
                    "(:355) -- the update adds dtheta to the RETURNED theta, not the input. "
                    "theta_final_cell{0,1} pin the column-major flat->original reconstruction "
                    "(:286-303)."
                ),
                "num_steps": int(sm["ns"]),
                "M": int(sm["M"]),
                "rt_num_steps": int(sm["rtns"]),
                "rt_M": int(sm["rtM"]),
                "lrate_traj": lrate,
                "shapes": {f"smorms3_{v}.bin": list(s) for v, s in SMORMS3_SHAPES.items()},
                "eps_guard": {
                    "text": (
                        "Near-zero-gradient run (M=1, 3 steps, grad=1e-8 constant), added per review "
                        "finding: the main/round-trip sequences above leave both eps=1e-16 guards "
                        "(SMORMS3.m:335 sqrt(MMS)+eps and the MMS+eps min-cap denominator, shared by "
                        "the dtheta cap and the delta update) INACTIVE (MMS >> eps there, so an "
                        "eps-placement mutation is a no-op). Here step-1 MMS = 0.5*grad^2 is the SAME "
                        "ORDER as eps -- MMS+eps ~= 3*MMS, a measured effect, not rounding noise -- "
                        "genuinely exercising both guards. delta_traj is PURE arithmetic (the MMS+eps "
                        "term feeds it directly) -> STRICT bits; theta_traj is sqrt-bearing -> "
                        "canary-gated."
                    ),
                    "eps": SMORMS3_EPS,
                    "step1_mms": eps_step1_mms,
                    "step1_mms_over_eps_ratio": eps_ratio,
                    "M": int(sm["epsM"]),
                    "num_steps": int(sm["epsns"]),
                },
            },
            "rprop": {
                "text": (
                    "Rprop.m (iRPROP-, 38 LOC) per-step state over a crafted 5-weight x 6-step "
                    "derivative/cost sequence. etap 1.2 / etam 0.5 / delta0 0.01 (the :5 rand is DEAD "
                    "-- overwritten at :6, see IMPROVEMENTS) / deltamin 1e-9 / deltamax 1.0. All "
                    "branches exercised: same-sign delta*=etap grow (delta_traj max 0.020736 = "
                    "0.01*1.2^4), sign-flip delta*=etam shrink (delta_traj min 0.0025 = 0.01*0.5^2) "
                    "+ the cost-gated backtrack (prev_cost<cost) + derivative-zeroing (derivout has "
                    "0 where deriv_script was nonzero), and the plain ==0 step. PURE arithmetic (sign, "
                    "min/max, mul, add -- no libm) -> STRICT bits on every platform."
                ),
                "N": int(rm["N"]),
                "K": int(rm["K"]),
                "shapes": {f"rprop_{v}.bin": list(s) for v, s in RPROP_SHAPES.items()},
            },
            "batching": {
                "text": (
                    "Task 10: CreateBatches.m (158 LOC) + GetNewBatch.m (94 LOC), TIER 1 real "
                    "functions, plus the FALLBACK-TIER getworstandbest_transcribed.m (verbatim "
                    "transcription of getCases.m:116-165's private getWorstAndBest subfunction -- "
                    "unreachable standalone, see that file's header). randperm is SHADOWED "
                    "(batching_shadow/randperm.m) with a fixed reverse-order permutation (p=n:-1:1), "
                    "reproducible AND trivially replicable in Python (tests/test_phase4c_batching.py), "
                    "so create_batches's own shuffle is pinned, not just GetNewBatch's rotation. Cases: "
                    "single (nbOfTargetClasses<=1), multi_nb (nbOfTargetClasses>1, non-multilingual, "
                    "classes {1,2,5} -> clean aggregate), clobber (classes {0,1,2} -- the natural "
                    "contiguous labeling silently OVERWRITES the aggregate slot with the top target "
                    "class's data, losing the aggregate entirely -- structural-only, no rotation), sub "
                    "(nbOfTargetClasses>1, multilingual, classes {1,2,5,8} -> two SubCases groups), "
                    "degenerate (nbOfTargetClasses*nbOfWorstCases+nbOfCasesPerBatch > file_nb -- "
                    "GetNewBatch takes the ~isfield(Batches,'currentClass') branch, Batch=Cases "
                    "verbatim). PURE integer-cursor arithmetic throughout (no libm) -> STRICT bits "
                    "everywhere."
                ),
                "single_n": int(bm["single_n"]),
                "multi_nb_n": int(bm["multi_nb_n"]),
                "sub_n": int(bm["sub_n"]),
                "degenerate_batch_len": int(bm["deg_len"]),
                "shapes": {f"batching_{v}.bin": list(s) for v, s in BATCHING_SHAPES.items()},
            },
            "checkgrad": {
                "text": (
                    "Task 10: CheckGrad.m (163 LOC), TIER 1 real function, driven with a "
                    "harness-local quadratic-surrogate CostFunction.m shadow (checkgrad_shadow/, "
                    "f(w)=0.5*sum(c.*(w-target).^2) over the REAL flat NN weight vector, re-derived "
                    "from param via the REAL vec2struct + nnet2MatFile on every call -- so a weight-kk "
                    "epsilon nudge applied by CheckGrad's own network2config/weights2nnet/vec2struct "
                    "round trip is visible here) plus no-op figure/subplot/semilogy/hold/grid shadows "
                    "(CheckGrad's diagnostic plot uses the old-style `subplot 211` form, which errors "
                    "under this Octave/graphics-toolkit combination independent of headlessness). "
                    "epsilon=1e-5 (:12, unconditional); the surrogate is an EXACT quadratic form, so "
                    "central diff carries no truncation error. Surfaces a genuine legacy bug (not a "
                    "harness artifact): weights2nnet.m (:150-186) never writes back the normalize "
                    "mean/std tail nnet2MatFile.m (:140-141) appends to `weights` -- CheckGrad's last "
                    "2 (2*len(normalize.mean)) numeric derivatives are always exactly 0 regardless of "
                    "the analytic value, an asymmetry vs the LID branch (:109), which explicitly trims "
                    "that tail before its own loop. See IMPROVEMENTS.md."
                ),
                "n": int(cgm["n"]),
                "Kw": int(cgm["Kw"]),
                "shapes": {f"checkgrad_{v}.bin": list(s) for v, s in CHECKGRAD_SHAPES.items()},
            },
            "masking": {
                "text": (
                    "Task 10: MaskingValidation.m (54 LOC) + vec2struct.m, TIER 1 real functions, no "
                    "shadow needed. PASS: a well-formed scalar mask (AlgName_decision_thresh_rising="
                    "0.7, the same case Task 8 bit-pins). FAIL: masking AlgName_min_speech (a "
                    "_padding_block-family vector field) with a negative last component -- the "
                    "encode ((fv+0.1)*adim, using the raw masked fv) is NOT the algebraic inverse of "
                    "the decode (-0.1+abs(p/adim)) for fv<0 (abs() destroys the sign asymmetrically), "
                    "so the round trip genuinely diverges and MaskingValidation correctly flags it -- "
                    "discovered by direct execution, not guessed. See IMPROVEMENTS.md."
                ),
                "pass_failed": int(mkm["pass_failed"]),
                "fail_failed": int(mkm["fail_failed"]),
                "shapes": {f"masking_{v}.bin": list(s) for v, s in MASKING_SHAPES.items()},
            },
            "qpso": {
                "text": (
                    "Task 11: QuantumPSO.m (845 LOC) bit-pinned against a MODIFIED-COPY "
                    "(tools/octave_harness/qpso_modified/QuantumPSO.m) whose rand/randperm/stblrnd are "
                    "substituted by reads of a shared committed random table (qpso_random_table.bin, "
                    f"{QPSO_TABLE_N} uniforms from numpy default_rng(seed={QPSO_TABLE_SEED}), column-major "
                    "stream) and whose CostFunction is the quadratic surrogate sum_j (x_j-center_j)^2 "
                    "(center=[3.5,4.5,6.5]); the wall-clock reseed (:91) is removed. The legacy path is "
                    "nondeterministic BY DESIGN, so no single RUN is reproducible -- the table makes the "
                    "OPERATOR STRUCTURE pinnable. Small run: D=3, ps=4, me=3, PSOseed=1 (seed overwrites "
                    "the first ps rows), trelea=3 (Clerc, needed for the dead vel3 bank's chi), ergrd=1e-99 "
                    "ergrdep=500 errgoal=NaN. Structure exercised: init + opposition-based learning "
                    "(:257-327); the 4 DEAD velocity banks that DRAW 8*ps*D per epoch but are never applied "
                    "(:375-411, gate :447 unreachable); the QDPSO ranking-operator roulette + the log(1/u) "
                    "jump (:414-443); DE recombination (randperm neighbours) + a Chambers-Mallows-Stuck "
                    "Levy kick (:465-479); pbest/gbest updates (:623-686); stall gate (:751); final "
                    "re-generation (:799-843). Per-value comparator split: init_* is PURE arithmetic "
                    "(normmat division + surrogate + min/sortrows) -> STRICT bits; every *_traj is "
                    "transcendental from epoch 1 (log(1/u) + the Levy tan/atan/sin/cos/log/pow) -> "
                    "canary-gated (bit-exact on the oracle libm, measured; the ranking is libm-robust "
                    "because gbestval+pbestval gaps are O(0.1)-O(1) >> ULP). cursor_end pins the exact "
                    "draw count (== Python TableRng.cursor). levy_in/levy_out pin the standalone CMS "
                    "sampler over table[0:16]."
                ),
                "table_seed": QPSO_TABLE_SEED,
                "table_n": QPSO_TABLE_N,
                "D": QPSO_D,
                "ps": QPSO_PS,
                "me": QPSO_ME,
                "K": QPSO_K,
                "cursor_end": cursor_line,
                "te": int(qm["te"]),
                "min_pbestval_gap": min_gap,
                "params": {
                    "ac1": 2.1,
                    "ac2": 2.1,
                    "iw1": 0.9,
                    "iw2": 0.6,
                    "iwe": 300,
                    "ergrd": 1e-99,
                    "ergrdep": 500,
                    "errgoal": "NaN",
                    "trelea": 3,
                    "pso_seed": 1,
                    "minmax": 0,
                    "adim": 10.0,
                    "center": [3.5, 4.5, 6.5],
                    "seed_value": [[5, 4, 6], [3, 5, 7], [6, 2, 4], [4, 6, 5]],
                },
                "shapes": {f"qpso_{v}.bin": list(s) for v, s in QPSO_SHAPES.items()},
            },
        }

        # Commit: every check passed -> copy the .bin fixtures into PHASE4C_DIR.
        for stage, shapes in (
            ("smorms3", SMORMS3_SHAPES),
            ("rprop", RPROP_SHAPES),
            ("batching", BATCHING_SHAPES),
            ("checkgrad", CHECKGRAD_SHAPES),
            ("masking", MASKING_SHAPES),
            ("qpso", QPSO_SHAPES),
        ):
            for var in shapes:
                name = f"{stage}_{var}.bin"
                shutil.copy2(tmp_dir / name, PHASE4C_DIR / name)
        # The shared QuantumPSO random table (the Python port + the Octave copy both read it).
        shutil.copy2(tmp_dir / "qpso_random_table.bin", PHASE4C_DIR / "qpso_random_table.bin")
        manifest_path = PHASE4C_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"OK: phase4c fixtures (octave {octave_version}; smorms3 M=7 steps=12 lrate cap@step8, "
        f"rt offset nonzero; rprop 5x6 delta span [{min(delta)}, {max(delta)}]; "
        f"batching single/multi_nb/sub/clobber/degenerate + getWorstAndBest; "
        f"checkgrad Kw=53 (2 normalize-tail entries always 0); masking pass=0 fail=1; "
        f"qpso D=3 ps=4 me=3 cursor_end={cursor_line} min_gap={min_gap:.3g}), "
        f"manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
