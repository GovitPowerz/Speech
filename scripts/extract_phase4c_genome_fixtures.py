"""Run the Phase 4c Octave `vec2struct` stage and write the genome-bijection goldens.

The MATLAB-side analog of the C++ oracle-harness extractors: GNU Octave runs the VENDORED
`legacy/Optimizer_V6.2.2/functions/vec2struct.m` (+ the `printConfig.m` it calls at mode==1)
over seven crafted param/mask/PS cases and dumps, per case:

  genome_<case>_param.bin      the INPUT param vector (io::binary .bin, f64 LE column-major)
  genome_<case>_out_param.bin  the returned out_param (mask/sortrows/clamp re-encoding)
  genome_<case>.config         the printConfig-serialized engine-facing key->value dict
                               (weights excluded -- they go via the .bin codec)
and records count_param (the genome-length walk invariant, risk R3) in genome_manifest.json.
A single extra artifact, not per-case, closes the Task 12 T8 debt's second half:

  fmt_scalar_boundaries.config  a battery of `_fmt_scalar` %d/%15.15e boundary values run
                                 through the REAL printConfig.m directly (TIER 1)

Cases: algo0 VRCTS, tdc (algo1), calib (algo2 LTSV -- calibration-law decode + clamps, no
NN), spectral (algo3 -- full NN weight walk), twin (algo6 -- SAD + LID mirror), masked
(algo3 + Force_Symetry/Force_Identical_Rows block ties + a scalar-field mask write-back),
vecmask (Task 12: algo3 + the mask-VECTOR arms -- per-block LSTM weight masks incl. the
narrow CellWeight layout, an output-layer neuron mask, and the NormalizeInputMean/Std
masks incl. the Std abs() asymmetry -- isolated from Force_Symetry/Force_Identical_Rows so
each arm is exercised standalone, closing the 4c T8 accepted-debt item).

vec2struct is PURE arithmetic (rem/round/abs/min/max/sortrows -- no libm), so param is read
byte-for-byte from the octave dump and the Python port reproduces out_param bit-exactly and
the config string-exactly on every platform. GNU Octave is a LOCAL-ONLY dep (brew install
octave); CI consumes only these committed fixtures.

Usage: uv run python scripts/extract_phase4c_genome_fixtures.py
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

CASES = ["algo0", "tdc", "calib", "spectral", "twin", "masked", "vecmask"]

STAGE_LINE_RE = re.compile(
    r"^OCTAVE_STAGE vec2struct algo0=(?P<algo0>\d+) tdc=(?P<tdc>\d+) calib=(?P<calib>\d+) "
    r"spectral=(?P<spectral>\d+) twin=(?P<twin>\d+) masked=(?P<masked>\d+) vecmask=(?P<vecmask>\d+)$",
    re.MULTILINE,
)

# fmt_scalar_boundaries.config field order (name -> the Python literal `_fmt_scalar` must
# reproduce); mirrors stage_vec2struct.m::probe_fmt_scalar's `names`/`vals` exactly.
FMT_SCALAR_BOUNDARIES: list[tuple[str, float]] = [
    ("b_zero", 0.0),
    ("b_neg_zero", -0.0),
    ("b_one", 1.0),
    ("b_neg_one", -1.0),
    ("b_int_1e5", 100000.0),
    ("b_neg_int_1e5", -100000.0),
    ("b_frac_above_1em5", 0.000015),
    ("b_neg_frac_above_1em5", -0.000015),
    ("b_frac_below_1em5", 0.0000099),
    ("b_frac_at_1em5", 0.00001),
    ("b_frac_above_1e5", 123456.789),
    ("b_frac_below_1e5", 99999.99999),
    ("b_long_precision", 0.123456789012345),
    ("b_long_precision_neg", -3.14159265358979),
    ("b_near_integer_boundary", 2.9999999999999996),
    ("b_big_int", 1234567890123.0),
    ("b_five", 5.0),
    ("b_neg_five", -5.0),
]


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
        [octave, "--no-gui", "--quiet", "--path", str(HARNESS_DIR), "--eval", f"run_stage('vec2struct', '{out_dir}')"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"octave stage vec2struct failed ({result.returncode})")
    return result.stdout


def _write_bin(path: Path, matrix: np.ndarray) -> None:
    """Write `matrix` as an io::binary .bin: i64 LE rows, i64 LE cols, f64 LE column-major."""
    mat = np.atleast_2d(np.ascontiguousarray(matrix, dtype="<f8"))
    if mat.shape[0] == 1 and mat.shape[1] > 1:
        mat = mat.T  # normalize a saved row vector to a column
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


def _hash_tree(root: Path, skip_prefixes: tuple[str, ...] = ()) -> dict[str, str]:
    digests: dict[str, str] = {}
    if not root.is_dir():
        return digests
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if any(rel.startswith(p) for p in skip_prefixes):
                continue
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def main() -> None:
    octave = _find_octave()
    octave_version = _octave_version(octave)

    # Guard prior phases AND the existing phase4c fixtures (smorms3/rprop/manifest) -- we only
    # write genome_*/fmt_scalar_boundaries files here.
    PHASE4C_SKIP = ("genome_", "fmt_scalar_boundaries")
    before = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
    before["phase4c"] = _hash_tree(PHASE4C_DIR, skip_prefixes=PHASE4C_SKIP)
    PHASE4C_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        stdout = _run_stage(octave, tmp_dir)

        sm = STAGE_LINE_RE.search(stdout)
        if not sm:
            raise SystemExit("OCTAVE_STAGE vec2struct line missing from stdout")
        counts = {c: int(sm[c]) for c in CASES}

        mat = scipy.io.loadmat(tmp_dir / "vec2struct.mat")
        shapes: dict[str, list[int]] = {}
        for case in CASES:
            param = np.asarray(mat[f"{case}_param"], dtype=np.float64).reshape(-1)
            out_param = np.asarray(mat[f"{case}_out_param"], dtype=np.float64).reshape(-1)
            count = int(np.asarray(mat[f"{case}_count"]).reshape(-1)[0])
            n = param.size
            if count != counts[case]:
                raise SystemExit(f"{case}: .mat count {count} != stdout count {counts[case]}")
            if count != n + 1:
                raise SystemExit(f"{case}: count_param {count} != len(param)+1 ({n + 1})")
            if out_param.size != n:
                raise SystemExit(f"{case}: out_param size {out_param.size} != param size {n}")
            _write_bin(tmp_dir / f"genome_{case}_param.bin", param)
            _write_bin(tmp_dir / f"genome_{case}_out_param.bin", out_param)
            shapes[f"genome_{case}_param.bin"] = [n, 1]
            shapes[f"genome_{case}_out_param.bin"] = [n, 1]
            conf_src = tmp_dir / f"vec2struct_{case}.config"
            if not conf_src.is_file():
                raise SystemExit(f"{case}: printConfig did not write {conf_src.name}")
            shutil.copy2(conf_src, tmp_dir / f"genome_{case}.config")

        fmt_scalar_src = tmp_dir / "fmt_scalar_boundaries.config"
        if not fmt_scalar_src.is_file():
            raise SystemExit("probe_fmt_scalar did not write fmt_scalar_boundaries.config")

        # --- Non-vacuity: the goldens must actually exercise the pinned behaviors. ---
        # calib: the crafted param selects the non-default sqrt/cubic laws + both [0,1] clamps.
        calib_text = (tmp_dir / "genome_calib.config").read_text()
        for token in ("LTSV_CostLawSpeech sqrt", "LTSV_CostLawNoSpeech cubic", "LTSV_CostLawParamSpeech 1", "LTSV_CostLawParamNoSpeech 0"):
            if token not in calib_text:
                raise SystemExit(f"calib non-vacuity: '{token}' missing -- law decode / clamp fixture is vacuous")
        # masked: Force_Symetry/Force_Identical_Rows re-encoded blocks -> out_param diverges
        # from the input param; and the scalar-field mask wrote back position 1 = 0.7*adim.
        _, _, m_param = _read_bin(tmp_dir / "genome_masked_param.bin")
        _, _, m_out = _read_bin(tmp_dir / "genome_masked_out_param.bin")
        if all(a == b for a, b in zip(m_param, m_out, strict=True)):
            raise SystemExit("masked non-vacuity: out_param == param -- Force_Symetry/Force_Identical_Rows tied nothing")
        if m_out[0] != 7.0:
            raise SystemExit(f"masked non-vacuity: out_param[0]={m_out[0]} != 7.0 (mask decision_thresh_rising write-back)")
        # twin: the LID falling = -rising negate quirk + BackPropOutputNetworkOnly both fired.
        twin_text = (tmp_dir / "genome_twin.config").read_text()
        for token in ("BLSTM_LID_BackPropOutputNetworkOnly true", "BLSTM_BackPropOutputNetworkOnly true", "BLSTM_LID_Mode 7"):
            if token not in twin_text:
                raise SystemExit(f"twin non-vacuity: '{token}' missing")
        # vecmask (Task 12): the six mask-VECTOR arms wrote back their inverse-encoded value
        # at their exact genome slice, not just "some" divergence from the raw param -- each
        # (0-based slice, expected out_param) pair below is independently hand-derived from
        # the vec2struct.m walk order for base_ps(3) and cross-checked bit-exact against this
        # fixture before being hardcoded (see task-12-report.md). A coincidental match on a
        # wrong slice is astronomically unlikely across six independently-shaped checks.
        _, _, vm_out_l = _read_bin(tmp_dir / "genome_vecmask_out_param.bin")
        vm_out = np.asarray(vm_out_l, dtype=np.float64)
        _VECMASK_SLICES: list[tuple[str, slice, list[float]]] = [
            ("Forward_Layer_0_LSTMBlock_0_InputGateWeights", slice(47, 60), [x / 2 + 5 for x in [-2.5, -1.25, -0.625, 0, 0.625, 1.25, 2.5, 3.75, -3.75, 5, -5, 0.3125, -0.3125]]),
            ("Forward_Layer_0_LSTMBlock_1_CellWeight", slice(134, 143), [x / 2 + 5 for x in [1.5, -1.5, 2.25, -2.25, 0, 4.5, -4.5, 6.75, -6.75]]),
            ("Backward_Layer_0_LSTMBlock_0_OutputGateWeights", slice(169, 182), [x / 2 + 5 for x in [-4.5, 4.5, -0.75, 0.75, 8.5, -8.5, 1.125, -1.125, 2.75, -2.75, 0, 9.25, -9.25]]),
            ("Output_Layer_1_Neuron_0_Weights", slice(249, 252), [x / 2 + 5 for x in [-1.0, 2.5, -3.25]]),
            ("NormalizeInputMean", slice(252, 255), [(x + 1) / 2 for x in [0.5, -0.25, 0.125]]),
            ("NormalizeInputStd", slice(255, 258), [abs(x) - 1e-3 for x in [-2.0, 3.0, -0.5]]),
        ]
        for arm_name, sl, expected in _VECMASK_SLICES:
            got = vm_out[sl]
            if not np.allclose(got, expected):
                raise SystemExit(f"vecmask non-vacuity: {arm_name} slice {sl} = {got.tolist()} != expected {expected}")

        # fmt_scalar_boundaries: both the %d and %15.15e branches actually fired (a vacuous
        # probe would mean every value collapsed to one format).
        fmt_text = fmt_scalar_src.read_text()
        for token in (
            "b_int_1e5 100000",  # a large exact integer must NOT flip to e-notation
            "b_big_int 1234567890123",
            "b_frac_above_1em5 1.500000000000000e-05",
            "b_frac_above_1e5 1.234567890000000e+05",
            "b_near_integer_boundary 3.000000000000000e+00",  # round(v)==v decode edge, non-integer branch
        ):
            if token not in fmt_text:
                raise SystemExit(f"fmt_scalar_boundaries non-vacuity: '{token}' missing")

        # Regression guard.
        after = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
        after["phase4c"] = _hash_tree(PHASE4C_DIR, skip_prefixes=PHASE4C_SKIP)
        for ph in [*PRIOR_PHASES, "phase4c"]:
            drifted = sorted(n for n in set(before[ph]) | set(after[ph]) if before[ph].get(n) != after[ph].get(n))
            if drifted:
                raise SystemExit(f"REGRESSION: {ph} fixtures changed after the octave run: {drifted}")

        manifest = {
            "text": (
                "Phase 4c Task 8 (+ Phase 4d Task 12 closure): the vec2struct genome<->config "
                "bijection bit-pins. GNU Octave runs the VENDORED "
                "legacy/Optimizer_V6.2.2/functions/vec2struct.m (+ printConfig.m at mode==1) over "
                "seven algo/mask/PS cases, dumping the INPUT param, the returned out_param (the mask/"
                "sortrows/clamp re-encoding), count_param (the genome-length walk invariant), and the "
                "printConfig engine-facing key->value dict (weights excluded, emitted via the .bin "
                "codec). TIER 1 (real function). vec2struct is PURE arithmetic (rem/round/abs/min/max/"
                "sortrows -- no libm), so param is read byte-for-byte and the port reproduces out_param "
                "bit-exactly + the config string-exactly on every platform. Cases: algo0 VRCTS, tdc "
                "(algo1), calib (algo2 -- law decode + [0,1] clamps in isolation), spectral (algo3 -- "
                "full NN weight walk), twin (algo6 -- SAD + LID mirror incl. the LID falling=-rising "
                "negate quirk), masked (algo3 + Force_Symetry/Force_Identical_Rows block ties + a "
                "scalar-field mask write-back), vecmask (Task 12 -- algo3 + the mask-VECTOR arms: "
                "per-block LSTM weight masks incl. the narrow CellWeight layout, an output-layer "
                "neuron mask, and the NormalizeInputMean/Std masks incl. the Std abs() asymmetry, each "
                "isolated from Force_Symetry/Force_Identical_Rows). A separate artifact, "
                "fmt_scalar_boundaries.config, pins genome.py's _fmt_scalar %d/%15.15e boundary "
                "formatting via a direct real-printConfig.m call over a battery of boundary values "
                "(integers incl. -0 and a big exact integer, the 1e-5/1e+5 magnitude boundaries on "
                "both sides, a round(v)==v near-integer decode edge, long-precision fractions)."
            ),
            "octave_version": octave_version,
            "counts": counts,
            "shapes": shapes,
            "configs": [f"genome_{c}.config" for c in CASES],
            "fmt_scalar_boundaries": {
                "file": "fmt_scalar_boundaries.config",
                "names": [name for name, _ in FMT_SCALAR_BOUNDARIES],
            },
        }

        for case in CASES:
            for suffix in ("_param.bin", "_out_param.bin"):
                name = f"genome_{case}{suffix}"
                shutil.copy2(tmp_dir / name, PHASE4C_DIR / name)
            shutil.copy2(tmp_dir / f"genome_{case}.config", PHASE4C_DIR / f"genome_{case}.config")
        shutil.copy2(fmt_scalar_src, PHASE4C_DIR / "fmt_scalar_boundaries.config")
        manifest_path = PHASE4C_DIR / "genome_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"OK: phase4c genome fixtures (octave {octave_version}; counts {counts}), "
        f"manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
