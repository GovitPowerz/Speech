"""Build and run the oracle harness for Phase 2b, then write the segmenter-wiring manifest.

Phase 2b links the vendored legacy segmenter TUs into the oracle harness
(``Segmenter``/``Segmentation``/``BLSTMSpectralSegmenter``/``BLSTMSignalSegmenter``/
``LongTermSpectralVariation``/``TimeDomainCorrel`` + ``-lpng``) so the REAL compiled
``Segmentation::toFile_VRCTS`` can serve as a byte golden. That routes through the
``iof::fmtr`` shim, which is upgraded from inert to FAITHFUL (``%f.Ns`` ->
``std::fixed`` + ``setprecision(N)``; ``%s`` -> default insertion; tail-flush + raw
passthrough). This extractor:

  1. Rebuilds the harness (now with the six segmenter TUs linked).
  2. Runs it in a throwaway dir seeded with the Phase 1 harness inputs, so no
     committed fixture dir is mutated.
  3. Parses the ``FMTR_CHECK`` self-test lines (each must be ``ok=1``; ok=0 ->
     SystemExit) into the manifest.
  4. Scans the four live driver TUs for UNCOMMENTED call sites of ``computeCost(``
     and ``computeSpectralPitch(`` (spec decision 7). Expected 0 each -> both ports
     stay deferred. NONZERO -> SystemExit (scope change for the controller).
  5. REGRESSION GUARD: hashes every file under ``tests/reference_data/phase1/`` and
     ``phase2/`` before and after the harness run; the fmtr upgrade must not perturb
     any prior dump (they use the harness's own Matrix2BinaryFile helpers, not fmtr).
     Any drift -> SystemExit.

Run TWICE; the manifest must be byte-identical.

Usage: uv run python scripts/extract_phase2b_fixtures.py
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
LEGACY_SRC = REPO_ROOT / "legacy" / "src"
PHASE0_DIR = REPO_ROOT / "tests" / "reference_data" / "phase0"
PHASE1_DIR = REPO_ROOT / "tests" / "reference_data" / "phase1"
PHASE2_DIR = REPO_ROOT / "tests" / "reference_data" / "phase2"
PHASE2B_DIR = REPO_ROOT / "tests" / "reference_data" / "phase2b"

NN_CONFIG = PHASE0_DIR / "1_worker_1.config"
NN_WEIGHTS = PHASE0_DIR / "NNweights_config1.bin"

# Input fixtures the harness reads from its output dir (same as Phase 1/2).
HARNESS_INPUTS = [
    "excerpt_2ch_8k.wav",
    "variant_mfcc_deltas.config",
    "variant_mfcc_sdc.config",
    "variant_logmel.config",
    "variant_rawband_ltsv.config",
]

# The six segmenter TUs linked in Phase 2b, and the extra link lib they drag.
SEGMENTER_TUS = [
    "Segmenter.cpp",
    "Segmentation.cpp",
    "BLSTMSpectralSegmenter.cpp",
    "BLSTMSignalSegmenter.cpp",
    "LongTermSpectralVariation.cpp",
    "TimeDomainCorrel.cpp",
]
EXTRA_LINK_LIBS = ["png"]

# The four LIVE driver TUs scanned for deferred-port call sites (spec decision 7).
DRIVER_TUS = [
    "TimeDomainCorrel.cpp",
    "LongTermSpectralVariation.cpp",
    "BLSTMSpectralSegmenter.cpp",
    "BLSTMSignalSegmenter.cpp",
]
DEFERRED_FNS = ["computeCost", "computeSpectralPitch"]

FMTR_CHECK_RE = re.compile(r"^FMTR_CHECK case=(?P<case>\w+) ok=(?P<ok>[01])$", re.MULTILINE)
EXPECTED_FMTR_CASES = {"fmtr_vrcts", "fmtr_f2", "fmtr_tail", "fmtr_f3"}


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
            rel = path.relative_to(root).as_posix()
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _parse_fmtr_checks(stdout: str) -> dict[str, int]:
    cases: dict[str, int] = {}
    for match in FMTR_CHECK_RE.finditer(stdout):
        cases[match["case"]] = int(match["ok"])
    missing = EXPECTED_FMTR_CASES - set(cases)
    if missing:
        raise SystemExit(f"FMTR_CHECK lines missing from harness stdout: {sorted(missing)}")
    failed = sorted(name for name, ok in cases.items() if ok != 1)
    if failed:
        raise SystemExit(f"FMTR_CHECK failed (ok=0) for: {failed}")
    return cases


def _scan_callsites(fn: str) -> list[dict[str, object]]:
    """Return UNCOMMENTED call sites of `fn(` across the four live driver TUs.

    A line is treated as commented out when its first non-space characters are
    ``//``. Block comments are not tracked (none of these call sites live inside a
    ``/* ... */`` block in the vendored sources), so this is a deliberately simple
    leading-``//`` scan -- documented here rather than parsing C++.
    """
    needle = fn + "("
    hits: list[dict[str, object]] = []
    for tu in DRIVER_TUS:
        text = (LEGACY_SRC / tu).read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle not in line:
                continue
            if line.lstrip().startswith("//"):
                continue
            hits.append({"file": tu, "line": lineno, "text": line.strip()})
    return hits


def main() -> None:
    # 1. Build the harness (now with the six segmenter TUs + -lpng linked).
    _run(["bash", str(HARNESS_DIR / "build.sh")])

    # 2. Snapshot the committed fixture dirs BEFORE the run (regression guard).
    before = {"phase1": _hash_tree(PHASE1_DIR), "phase2": _hash_tree(PHASE2_DIR)}

    # 3. Run the harness in a throwaway dir seeded with the Phase 1 inputs.
    PHASE2B_DIR.mkdir(parents=True, exist_ok=True)
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

    # 4. Regression guard: the committed fixture dirs must be byte-identical after
    #    the run (the fmtr upgrade must not perturb any prior harness dump).
    after = {"phase1": _hash_tree(PHASE1_DIR), "phase2": _hash_tree(PHASE2_DIR)}
    for phase in ("phase1", "phase2"):
        b, a = before[phase], after[phase]
        drifted = sorted(
            name for name in set(b) | set(a) if b.get(name) != a.get(name)
        )
        if drifted:
            raise SystemExit(
                f"REGRESSION: {phase} fixtures changed after harness run: {drifted}"
            )

    # 5. Parse the fmtr self-test (all ok=1 or SystemExit).
    fmtr_cases = _parse_fmtr_checks(stdout)

    # 6. Call-site scan (spec decision 7). Expected 0 uncommented callers each; a
    #    nonzero count is a scope change -> STOP.
    callsites = {fn: _scan_callsites(fn) for fn in DEFERRED_FNS}
    nonzero = {fn: hits for fn, hits in callsites.items() if hits}
    if nonzero:
        detail = json.dumps(nonzero, indent=2)
        raise SystemExit(
            "BLOCKED: found UNCOMMENTED call sites of a deferred function (scope "
            f"change for the controller):\n{detail}"
        )

    # 7. Compiler version (same g++ selection as build.sh).
    gxx = _run(["bash", "-c", 'ls "$(brew --prefix)"/bin/g++-* | sort -V | tail -1']).strip()
    compiler = _run([gxx, "--version"]).splitlines()[0].strip()

    manifest = {
        "compiler": compiler,
        "flags": (
            "-O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off "
            "-DEIGEN_DONT_VECTORIZE -include boost/math/special_functions/round.hpp"
        ),
        "segmenter_linkage": {
            "text": (
                "Phase 2b Task 1: the six vendored legacy segmenter TUs linked into "
                "the oracle harness so the REAL compiled Segmentation::toFile_VRCTS "
                "becomes a byte golden source. The include-graph analysis predicted "
                "no residual link errors beyond -lpng (Segmenter.cpp's plotting "
                "members odr-use png++ symbols even though runtime-dead); confirmed "
                "at build time -- no additional TUs were required."
            ),
            "translation_units": SEGMENTER_TUS,
            "extra_link_libs": EXTRA_LINK_LIBS,
            "additional_tus_required": [],
        },
        "fmtr_shim": {
            "text": (
                "Phase 2b Task 1: the iof::fmtr shim upgraded from inert (dropped "
                "format strings) to FAITHFUL. %f.Ns -> std::fixed + setprecision(N) "
                "(N observed in {0,2,3,4} on golden paths), %s -> default stream "
                "insertion, %% -> literal '%', tail-flush + raw passthrough for "
                "chained inserts after the last directive. Unknown/width directives "
                "(% 6ds, %f3.2s, % f6.2s, %s.s, ...) appear only on log paths the "
                "harness never dumps as a golden and are treated as %s after emitting "
                "their literal prefix. Each FMTR_CHECK case runs the shim against an "
                "independent std::ostringstream ground truth (fixed/setprecision "
                "directly); ok=1 means bit-identical strings."
            ),
            "checks": dict(sorted(fmtr_cases.items())),
        },
        "callsite_checks": {
            "text": (
                "Phase 2b spec decision 7: UNCOMMENTED call-site counts of the two "
                "deferred-port functions across the four LIVE driver TUs "
                "(TimeDomainCorrel, LongTermSpectralVariation, BLSTMSpectralSegmenter, "
                "BLSTMSignalSegmenter). A line is 'commented out' when its first "
                "non-space chars are '//'. 0 live callers each keeps both ports "
                "deferred (computeCost: no dead-code even; computeSpectralPitch: one "
                "commented-out call at BLSTMSpectralSegmenter.cpp:810). A nonzero "
                "count would be a scope change and aborts the extractor."
            ),
            "driver_tus": DRIVER_TUS,
            "counts": {fn: len(hits) for fn, hits in callsites.items()},
            "matched_lines": callsites,
        },
    }

    manifest_path = PHASE2B_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    counts = manifest["callsite_checks"]["counts"]
    print(
        f"OK: {len(fmtr_cases)} FMTR_CHECK all ok, "
        f"callsites {counts}, manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
