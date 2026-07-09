"""Run the Phase 4c Octave `vec2struct` stage and write the genome-bijection goldens.

The MATLAB-side analog of the C++ oracle-harness extractors: GNU Octave runs the VENDORED
`legacy/Optimizer_V6.2.2/functions/vec2struct.m` (+ the `printConfig.m` it calls at mode==1)
over six crafted param/mask/PS cases and dumps, per case:

  genome_<case>_param.bin      the INPUT param vector (io::binary .bin, f64 LE column-major)
  genome_<case>_out_param.bin  the returned out_param (mask/sortrows/clamp re-encoding)
  genome_<case>.config         the printConfig-serialized engine-facing key->value dict
                               (weights excluded -- they go via the .bin codec)
and records count_param (the genome-length walk invariant, risk R3) in genome_manifest.json.

Cases: algo0 VRCTS, tdc (algo1), calib (algo2 LTSV -- calibration-law decode + clamps, no
NN), spectral (algo3 -- full NN weight walk), twin (algo6 -- SAD + LID mirror), masked
(algo3 + Force_Symetry/Force_Identical_Rows block ties + a scalar-field mask write-back).

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

CASES = ["algo0", "tdc", "calib", "spectral", "twin", "masked"]

STAGE_LINE_RE = re.compile(
    r"^OCTAVE_STAGE vec2struct algo0=(?P<algo0>\d+) tdc=(?P<tdc>\d+) calib=(?P<calib>\d+) "
    r"spectral=(?P<spectral>\d+) twin=(?P<twin>\d+) masked=(?P<masked>\d+)$",
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

    # Guard prior phases AND the existing phase4c fixtures (smorms3/rprop/manifest) -- we only
    # write genome_* files here.
    before = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
    before["phase4c"] = _hash_tree(PHASE4C_DIR, skip_prefix="genome_")
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

        # Regression guard.
        after = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
        after["phase4c"] = _hash_tree(PHASE4C_DIR, skip_prefix="genome_")
        for ph in [*PRIOR_PHASES, "phase4c"]:
            drifted = sorted(n for n in set(before[ph]) | set(after[ph]) if before[ph].get(n) != after[ph].get(n))
            if drifted:
                raise SystemExit(f"REGRESSION: {ph} fixtures changed after the octave run: {drifted}")

        manifest = {
            "text": (
                "Phase 4c Task 8: the vec2struct genome<->config bijection bit-pins. GNU Octave runs the "
                "VENDORED legacy/Optimizer_V6.2.2/functions/vec2struct.m (+ printConfig.m at mode==1) over "
                "six algo/mask/PS cases, dumping the INPUT param, the returned out_param (the mask/sortrows/"
                "clamp re-encoding), count_param (the genome-length walk invariant), and the printConfig "
                "engine-facing key->value dict (weights excluded, emitted via the .bin codec). TIER 1 (real "
                "function). vec2struct is PURE arithmetic (rem/round/abs/min/max/sortrows -- no libm), so "
                "param is read byte-for-byte and the port reproduces out_param bit-exactly + the config "
                "string-exactly on every platform. Cases: algo0 VRCTS, tdc (algo1), calib (algo2 -- law "
                "decode + [0,1] clamps in isolation), spectral (algo3 -- full NN weight walk), twin (algo6 -- "
                "SAD + LID mirror incl. the LID falling=-rising negate quirk), masked (algo3 + Force_Symetry/"
                "Force_Identical_Rows block ties + a scalar-field mask write-back)."
            ),
            "octave_version": octave_version,
            "counts": counts,
            "shapes": shapes,
            "configs": [f"genome_{c}.config" for c in CASES],
        }

        for case in CASES:
            for suffix in ("_param.bin", "_out_param.bin"):
                name = f"genome_{case}{suffix}"
                shutil.copy2(tmp_dir / name, PHASE4C_DIR / name)
            shutil.copy2(tmp_dir / f"genome_{case}.config", PHASE4C_DIR / f"genome_{case}.config")
        manifest_path = PHASE4C_DIR / "genome_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"OK: phase4c genome fixtures (octave {octave_version}; counts {counts}), "
        f"manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
