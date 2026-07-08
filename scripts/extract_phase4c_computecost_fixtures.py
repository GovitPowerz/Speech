"""Run the Phase 4c Octave `computecost` stage and write the ComputeCost-assembly goldens.

The MATLAB-side analog of the C++ oracle-harness extractors: GNU Octave runs the
`tools/octave_harness/stage_computecost.m` transcription of the VENDORED
`legacy/Optimizer_V6.2.2/functions/ComputeCost.m` cost-assembly block (the pure lines
:285-652 -- sortrows aggregation, deriv averaging, pooled input stats, L2, and the
balance-law 0/3/4/5/10 error + cost) over the COMMITTED phase4a/4b MultiConfigResults
fixtures + crafted per-balance variants, dumping the goldens the Python port
(`src/python/speech/engine.py`) is bit-pinned against.

TIER: FALLBACK (stage-local transcription; adjudicated in IMPROVEMENTS). ComputeCost.m's
top half shells out to the engine (`system('python RunFsp.py ...')`, :173-191) and cannot
run in Octave, and its `!`-escape cleaning (:37) deletes any pre-injected worker files, so
there is no injection point that leaves the vendored .m unmodified. The stage transcribes
the pure ASSEMBLY lines line-for-line (`% legacy:` provenance) and Octave executes them
with real MATLAB semantics (sortrows stable-ascending, median, hist center-binning, std
ddof=1, cumsum, exp/log). The port must match.

STRICT (pure arithmetic, bit-exact everywhere): agg (sortrows), avg (col0/max(1,col1)),
l2, pooled mean/nb, and the crafted integer balance 0/3/4/5. CANARY-gated (libm-bearing,
see tests/_libm_gate.py): pooled std (sqrt), the balance-10 LID calibration
(hist/cumsum/std/exp/log), and the committed-fixture realistic cases.

GNU Octave is a LOCAL-ONLY dep (brew install octave); CI consumes only these committed
fixtures. Running this extractor TWICE yields byte-identical committed output; a
prior-phase (and prior-phase4c) hash guard SystemExits on any regression.

Usage: uv run python scripts/extract_phase4c_computecost_fixtures.py
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

# Every octave var the stage dumps -> the `computecost_<var>.bin` fixture. The input
# matrices (crafted MCRs, deriv/pool/l2 inputs) let the port re-run the assembly; the
# `_error`/`_cost`/`_nnseg`/... outputs are the goldens it is pinned against.
VARS = [
    # aggregate_workers (sortrows [1 2 3])
    "agg_w1", "agg_w2", "agg_out",
    # average_derivs (col0/max(1,col1))
    "avg_in", "avg_out",
    # l2_penalty
    "l2_w", "l2_isBias", "l2_cost", "l2_grad",
    # pool_input_stats
    "pool_nb", "pool_mean", "pool_std", "pool_out_nb", "pool_out_mean", "pool_out_std",
    # crafted integer balance 0/3/4/5 (STRICT)
    "cb_mcr",
    "cb0_error", "cb0_cost", "cb0_nnseg", "cb0_cpumean",
    "cb3_error", "cb3_cost", "cb3_nnseg", "cb3_cpumean",
    "cb4_error", "cb4_cost", "cb4_nnseg", "cb4_cpumean",
    "cb5_error", "cb5_cost", "cb5_nnseg", "cb5_cpumean",
    # committed tier2 (algo 3) realistic (CANARY)
    "tier2_b0_error", "tier2_b0_cost", "tier2_b0_nnseg", "tier2_b0_cpumean",
    "tier2_b5_error", "tier2_b5_cost", "tier2_b5_nnseg", "tier2_b5_cpumean",
    # committed twin (algo 6) raw pieces (CANARY)
    "twin_nnseg", "twin_nnlid",
    # crafted balance-10 2-class cutoff-search branch (CANARY)
    "b10a_mcr",
    "b10a_error", "b10a_cost", "b10a_nnseg", "b10a_cpumean", "b10a_nnlid", "b10a_cutoff",
    # crafted balance-10 3-class else branch (CANARY)
    "b10b_mcr",
    "b10b_error", "b10b_cost", "b10b_nnseg", "b10b_cpumean", "b10b_nnlid", "b10b_cutoff",
]

STAGE_LINE_RE = re.compile(
    r"^OCTAVE_STAGE computecost cb0=(?P<cb0>\S+) cb5=(?P<cb5>\S+) b10a=(?P<b10a>\S+) "
    r"b10b=(?P<b10b>\S+) agg_rows=(?P<agg>\d+) pool_nb=(?P<pool>\d+)$",
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


def _run_stage(octave: str, out_dir: Path) -> str:
    result = subprocess.run(
        [octave, "--no-gui", "--quiet", "--path", str(HARNESS_DIR), "--eval", f"run_stage('computecost', '{out_dir}')"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"octave stage computecost failed ({result.returncode})")
    return result.stdout


def _write_bin(path: Path, matrix: np.ndarray) -> None:
    """Write `matrix` as an io::binary .bin: i64 LE rows, i64 LE cols, f64 LE column-major.
    A saved row vector (1xN, N>1) is normalized to a column so the codec is unambiguous."""
    mat = np.atleast_2d(np.ascontiguousarray(matrix, dtype="<f8"))
    if mat.shape[0] == 1 and mat.shape[1] > 1:
        mat = mat.T
    rows, cols = mat.shape
    with path.open("wb") as f:
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        f.write(mat.flatten(order="F").tobytes())


def _read_bin(path: Path) -> tuple[int, int, np.ndarray]:
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    n = rows * cols
    data = np.frombuffer(raw[16 : 16 + 8 * n], dtype="<f8").reshape((rows, cols), order="F")
    return rows, cols, np.array(data)


def _hash_tree(root: Path, skip_prefix: str | None = None) -> dict[str, str]:
    digests: dict[str, str] = {}
    if not root.is_dir():
        return digests
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if skip_prefix is not None and rel.startswith(skip_prefix):
                continue
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def main() -> None:
    octave = _find_octave()
    octave_version = _octave_version(octave)

    # Guard prior phases AND the existing phase4c fixtures (smorms3/rprop/genome) -- we only
    # write computecost_* files here.
    before = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
    before["phase4c"] = _hash_tree(PHASE4C_DIR, skip_prefix="computecost_")
    PHASE4C_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        stdout = _run_stage(octave, tmp_dir)

        sm = STAGE_LINE_RE.search(stdout)
        if not sm:
            raise SystemExit("OCTAVE_STAGE computecost line missing from stdout")
        if int(sm["agg"]) != 4:
            raise SystemExit(f"agg rows {sm['agg']} != 4")
        if int(sm["pool"]) != 400:
            raise SystemExit(f"pool nb {sm['pool']} != 400")

        mat = scipy.io.loadmat(tmp_dir / "computecost.mat")
        shapes: dict[str, list[int]] = {}
        for var in VARS:
            if var not in mat:
                raise SystemExit(f"variable {var} missing from computecost.mat")
            m = np.atleast_2d(np.asarray(mat[var], dtype=np.float64))
            _write_bin(tmp_dir / f"computecost_{var}.bin", m)
            r, c, _ = _read_bin(tmp_dir / f"computecost_{var}.bin")
            shapes[f"computecost_{var}.bin"] = [r, c]

        # --- Non-vacuity: the goldens must actually exercise the pinned behaviors. ---
        # balance 3 vs 4: the over-90 saturation (0* vs 1*) must make the error vectors DIVERGE.
        _, _, cb3 = _read_bin(tmp_dir / "computecost_cb3_error.bin")
        _, _, cb4 = _read_bin(tmp_dir / "computecost_cb4_error.bin")
        if np.array_equal(cb3, cb4):
            raise SystemExit("balance 3 == balance 4 error -- the over-90 saturation coefficient is not exercised")
        # sortrows actually reordered the concatenated workers.
        _, _, agg = _read_bin(tmp_dir / "computecost_agg_out.bin")
        if list(agg[:, 0]) != [1.0, 1.0, 2.0, 2.0]:
            raise SystemExit(f"agg_out not sorted ascending by col1: {list(agg[:, 0])}")
        # average_derivs exercised the count==0 -> max(1,0) divide (row 1: 10/1 = 10).
        _, _, avg = _read_bin(tmp_dir / "computecost_avg_out.bin")
        if avg[1, 0] != 10.0:
            raise SystemExit(f"average_derivs count==0 divide not exercised (avg_out[1]={avg[1, 0]}, want 10)")
        # l2 excluded the biases (positions 1,3 zeroed in the gradient).
        _, _, l2g = _read_bin(tmp_dir / "computecost_l2_grad.bin")
        if l2g[1, 0] != 0.0 or l2g[3, 0] != 0.0:
            raise SystemExit("l2_grad did not zero the bias positions -- is_bias exclusion not exercised")
        # balance-10 2-class cutoff search landed a finite interior cutoff (not clamped to 0.1/99.9).
        _, _, cut = _read_bin(tmp_dir / "computecost_b10a_cutoff.bin")
        if not (0.1 < float(cut[0, 0]) < 99.9):
            raise SystemExit(f"b10a cutoff {float(cut[0, 0])} clamped -- the cutoff search is vacuous")
        # balance-10 3-class took the else branch (per-file cutoff vector -> -1 sentinel).
        _, _, cutb = _read_bin(tmp_dir / "computecost_b10b_cutoff.bin")
        if float(cutb[0, 0]) != -1.0:
            raise SystemExit(f"b10b cutoff sentinel {float(cutb[0, 0])} != -1 (else branch not taken)")

        # Regression guard.
        after = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
        after["phase4c"] = _hash_tree(PHASE4C_DIR, skip_prefix="computecost_")
        for ph in [*PRIOR_PHASES, "phase4c"]:
            drifted = sorted(n for n in set(before[ph]) | set(after[ph]) if before[ph].get(n) != after[ph].get(n))
            if drifted:
                raise SystemExit(f"REGRESSION: {ph} fixtures changed after the octave run: {drifted}")

        manifest = {
            "text": (
                "Phase 4c Task 9: the ComputeCost/ComputeGradient cost-assembly bit-pins. GNU Octave runs "
                "tools/octave_harness/stage_computecost.m -- a FALLBACK-TIER stage-local transcription of the "
                "VENDORED legacy/Optimizer_V6.2.2/functions/ComputeCost.m pure ASSEMBLY lines (:285-652: sortrows "
                "[1 2 3] aggregation, deriv averaging col0/max(1,col1), pooled input stats, L2, and the balance-law "
                "0/3/4/5/10 error + cost) over the committed phase4a/4b MultiConfigResults fixtures + crafted "
                "per-balance variants. ComputeCost.m's top half shells out to the engine (system RunFsp) and cannot "
                "run in Octave, and its !-escape cleaning (:37) deletes pre-injected worker files, so no injection "
                "point leaves the vendored .m unmodified; the transcription is line-for-line with `% legacy:` "
                "provenance and Octave executes the real MATLAB semantics (sortrows stable-ascending, median, hist "
                "center-binning, std ddof=1, cumsum, exp/log). Column schema (Error_vad = MCR(:,4:end), 1-based): "
                "1 Pfa, 2 Pmiss, 3 (100-success), 4 cpu, 5 seg-num, 15 LID-num, 16 flag, 17:end-2 per-class "
                "(>150/+200 in-band), end-1 LID-denom, end seg-denom. STRICT: agg/avg/l2/pooled-mean/crafted-integer "
                "balance 0/3/4/5. CANARY (tests/_libm_gate.py): pooled std (sqrt), balance-10 (hist/std/exp/log), "
                "committed-fixture realistic cases."
            ),
            "tier": "fallback (stage-local transcription of ComputeCost.m:285-652)",
            "octave_version": octave_version,
            "l2_regul": 0.1,
            "balance_backprop": 0.5,
            "mode": 0,
            "measured": {
                "cb0_cost": float(sm["cb0"]),
                "cb5_cost": float(sm["cb5"]),
                "b10a_cost": float(sm["b10a"]),
                "b10b_cost": float(sm["b10b"]),
                "b10a_cutoff": float(cut[0, 0]),
            },
            # Three comparators (see the test):
            #   strict -- bit-exact on every platform (pure arithmetic; single-op /100; the
            #             per-file ERROR vectors + integer-error `mean` costs cb0/cb5).
            #   canary -- bit-exact on the oracle libm, hybrid ULP/abs elsewhere (a libm
            #             transcendental or sqrt in the chain: pooled std, balance-10 exp/log
            #             error vectors, the realistic committed-fixture element-wise errors).
            #   close  -- always a tight hybrid ULP/abs bound, EVEN on the oracle env: a `mean`/
            #             `mean(.^2)` reduction over FRACTIONAL errors differs numpy-vs-Octave by
            #             ~1 ULP by summation order (NOT libm), so exact-on-oracle is unattainable.
            "strict": [
                "computecost_agg_out.bin", "computecost_avg_out.bin", "computecost_l2_cost.bin",
                "computecost_l2_grad.bin", "computecost_pool_out_nb.bin", "computecost_pool_out_mean.bin",
                "computecost_cb0_error.bin", "computecost_cb0_cost.bin", "computecost_cb0_nnseg.bin",
                "computecost_cb3_error.bin", "computecost_cb3_nnseg.bin",
                "computecost_cb4_error.bin", "computecost_cb4_nnseg.bin",
                "computecost_cb5_error.bin", "computecost_cb5_cost.bin", "computecost_cb5_nnseg.bin",
            ],
            "canary": [
                "computecost_pool_out_std.bin",
                "computecost_tier2_b0_error.bin", "computecost_tier2_b5_error.bin",
                "computecost_twin_nnseg.bin", "computecost_twin_nnlid.bin",
                "computecost_b10a_error.bin", "computecost_b10a_cutoff.bin", "computecost_b10a_nnlid.bin",
                "computecost_b10b_error.bin",
            ],
            "close": [
                "computecost_cb3_cost.bin", "computecost_cb4_cost.bin",
                "computecost_tier2_b0_cost.bin", "computecost_tier2_b5_cost.bin",
                "computecost_b10a_cost.bin", "computecost_b10b_cost.bin",
            ],
            "shapes": shapes,
        }

        for var in VARS:
            shutil.copy2(tmp_dir / f"computecost_{var}.bin", PHASE4C_DIR / f"computecost_{var}.bin")
        manifest_path = PHASE4C_DIR / "computecost_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"OK: phase4c computecost fixtures (octave {octave_version}; cb0={sm['cb0']} cb5={sm['cb5']} "
        f"b10a={sm['b10a']} b10b={sm['b10b']}), manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
