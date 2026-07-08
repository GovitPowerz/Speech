"""Run the Phase 4c Octave harness stages and write the SMORMS3 + Rprop bit-pin
fixtures.

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
  rprop   : Rprop.m (function) per-step state over a crafted derivative/cost sequence
            exercising every branch (grow / shrink+backtrack / plain) -- pure arithmetic,
            STRICT bits everywhere.

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
}
RPROP_SHAPES = {
    "deriv_script": (5, 6),
    "cost_script": (1, 6),
    "weights0": (5, 1),
    "weights_traj": (5, 6),
    "delta_traj": (5, 6),
    "deltaweight_traj": (5, 6),
    "derivout_traj": (5, 6),
}

STAGE_LINE_RE = re.compile(
    r"^OCTAVE_STAGE smorms3 M=(?P<M>\d+) num_steps=(?P<ns>\d+) rt_M=(?P<rtM>\d+) rt_num_steps=(?P<rtns>\d+)$",
    re.MULTILINE,
)
RPROP_LINE_RE = re.compile(r"^OCTAVE_STAGE rprop N=(?P<N>\d+) K=(?P<K>\d+)$", re.MULTILINE)


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
        smorms3_stdout = _run_stage(octave, "smorms3", tmp_dir)
        rprop_stdout = _run_stage(octave, "rprop", tmp_dir)

        sm = STAGE_LINE_RE.search(smorms3_stdout)
        if not sm:
            raise SystemExit("OCTAVE_STAGE smorms3 line missing from stdout")
        rm = RPROP_LINE_RE.search(rprop_stdout)
        if not rm:
            raise SystemExit("OCTAVE_STAGE rprop line missing from stdout")
        if (int(sm["M"]), int(sm["ns"]), int(sm["rtM"]), int(sm["rtns"])) != (7, 12, 3, 4):
            raise SystemExit(f"smorms3 stage dims {(sm['M'], sm['ns'], sm['rtM'], sm['rtns'])} != (7,12,3,4)")
        if (int(rm["N"]), int(rm["K"])) != (5, 6):
            raise SystemExit(f"rprop stage dims {(rm['N'], rm['K'])} != (5,6)")

        _convert_stage(tmp_dir / "smorms3.mat", "smorms3", SMORMS3_SHAPES, tmp_dir)
        _convert_stage(tmp_dir / "rprop.mat", "rprop", RPROP_SHAPES, tmp_dir)

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

        # RPROP: both the etap-grow (>0.01) and etam-shrink (<0.01) branches fired, and the
        # sign-flip derivative-zeroing produced a 0 where the input derivative was nonzero.
        dr, dc, delta = _read_bin(tmp_dir / "rprop_delta_traj.bin")
        if not (max(delta) > 0.01 and min(delta) < 0.01):
            raise SystemExit(f"rprop delta_traj [{min(delta)}, {max(delta)}] did not span the 0.01 delta0 both ways")
        _, _, deriv_in = _read_bin(tmp_dir / "rprop_deriv_script.bin")
        _, _, deriv_out = _read_bin(tmp_dir / "rprop_derivout_traj.bin")
        if not any(o == 0.0 and i != 0.0 for i, o in zip(deriv_in, deriv_out, strict=True)):
            raise SystemExit("rprop derivout never zeroed a nonzero derivative -- the <0 sign-flip branch never fired")

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
        }

        # Commit: every check passed -> copy the .bin fixtures into PHASE4C_DIR.
        for stage, shapes in (("smorms3", SMORMS3_SHAPES), ("rprop", RPROP_SHAPES)):
            for var in shapes:
                name = f"{stage}_{var}.bin"
                shutil.copy2(tmp_dir / name, PHASE4C_DIR / name)
        manifest_path = PHASE4C_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"OK: phase4c fixtures (octave {octave_version}; smorms3 M=7 steps=12 lrate cap@step8, "
        f"rt offset nonzero; rprop 5x6 delta span [{min(delta)}, {max(delta)}]), "
        f"manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
