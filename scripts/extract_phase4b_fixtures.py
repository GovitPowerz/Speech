"""Build and run the oracle harness's phase4b_confusion stage, then write the
LID confusion-core fixtures.

Phase 4b Task 1 ports `BagOfProcessors::PrintConfusionMatrix`'s sentinel-decode
+ best-non-target argmax accumulation (`BagOfProcessors.cpp:501-535`) and
`Confusion2String`'s returned row-normalized error aggregate
(`Helpers.hpp:365-438`). Both are pure arithmetic (no libm), so every fixture
here is STRICT BITS on every platform.

The oracle harness's `phase4b_confusion` stage (appended to
`tools/oracle_harness/main.cpp`, unconditional -- no corpus workdir needed)
dumps three `.bin` files for one crafted 4-row x 3-class `results_lid` matrix
(a target-wins row, a target-loses row, a no-target row, and an ambiguous
exact-tie row):

  - confusion_input.bin  : the crafted input (4x3).
  - confusion_matrix.bin : the harness's OWN transcription of the accumulation
    loop (:501-535 -- the real matrix is a function-LOCAL inside the
    protected `PrintConfusionMatrix` and is never returned, even via a probe
    subclass, so it cannot be read back directly).
  - confusion_error.bin  : 1x2 = [error1, error2], where error1 is the REAL
    free `Confusion2String` called on the transcribed matrix, and error2 is
    the REAL (protected, via a `BagProbe` subclass) `PrintConfusionMatrix`
    itself called end-to-end on the SAME input. error1 == error2 bit-exact is
    the non-vacuous cross-check that the transcribed matrix matches what the
    real method built internally.

This extractor:

  1. Rebuilds the harness (BagOfProcessors.cpp was already linked in Phase 4a;
     no new translation units).
  2. Snapshots every prior-phase fixture dir BEFORE the run (regression guard).
  3. Runs the harness in a throwaway dir seeded with the standard Phase 1
     inputs (no corpus workdir arg -- the phase4a tier1/tier2 block is gated
     on that arg being present and is skipped here).
  4. Parses the `PHASE4B_CONFUSION ok=<0|1> error1=<f> error2=<f>` stdout line,
     asserting ok=1 (SystemExit otherwise) and cross-checking the echoed
     doubles against the committed `confusion_error.bin`.
  5. Copies the three `.bin` dumps into `tests/reference_data/phase4b/` and
     writes `manifest.json` with MEASURED shapes/values.
  6. REGRESSION GUARD: hashes every prior-phase fixture dir before/after; any
     drift -> SystemExit.

Run TWICE; the `.bin` dumps + manifest must be byte-identical (pure arithmetic,
no wall-clock or libm inputs).

Usage: uv run python scripts/extract_phase4b_fixtures.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "tools" / "oracle_harness"
REF_DIR = REPO_ROOT / "tests" / "reference_data"
PHASE0_DIR = REF_DIR / "phase0"
PHASE1_DIR = REF_DIR / "phase1"
PHASE2B_DIR = REF_DIR / "phase2b"
PHASE4B_DIR = REF_DIR / "phase4b"

NN_CONFIG = PHASE0_DIR / "1_worker_1.config"
NN_WEIGHTS = PHASE0_DIR / "NNweights_config1.bin"
TDC_CONFIG = PHASE2B_DIR / "tdc.config"
LTSV_CONFIG = PHASE2B_DIR / "ltsv.config"
LTSV_DCT_CONFIG = PHASE2B_DIR / "ltsv_dct.config"
LTSV_TINY_CONFIG = PHASE2B_DIR / "ltsv_tiny.config"
LTSV_POWERMEL_CONFIG = PHASE2B_DIR / "ltsv_powermel.config"
SIGNAL_CONFIG = PHASE2B_DIR / "signal.config"

# Input fixtures the harness reads from its argv[1] output dir (earlier phase
# 1/2/2b stages still run every invocation and expect these seeded).
HARNESS_INPUTS = [
    "excerpt_2ch_8k.wav",
    "variant_mfcc_deltas.config",
    "variant_mfcc_sdc.config",
    "variant_logmel.config",
    "variant_rawband_ltsv.config",
]

# Prior-phase fixture dirs the extractor must NOT perturb (regression guard;
# spans every phase committed so far, phase0..phase4a).
PRIOR_PHASES = ["phase0", "phase0b", "phase0bii", "phase1", "phase2", "phase2b", "phase3", "phase4a"]

CONFUSION_BINS = ["confusion_input.bin", "confusion_matrix.bin", "confusion_error.bin"]

PHASE4B_RE = re.compile(
    r"^PHASE4B_CONFUSION ok=(?P<ok>\d) error1=(?P<e1>[0-9.eE+-]+) error2=(?P<e2>[0-9.eE+-]+)$",
    re.MULTILINE,
)


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


def _read_bin(path: Path) -> tuple[int, int, list[float]]:
    """Read an io::binary .bin (i64 LE rows, i64 LE cols, f64 LE column-major)."""
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    n = rows * cols
    data = list(struct.unpack(f"<{n}d", raw[16 : 16 + 8 * n]))
    return rows, cols, data


def main() -> None:
    # 1. Build the harness.
    _run(["bash", str(HARNESS_DIR / "build.sh")])

    # 2. Snapshot the prior-phase fixture dirs BEFORE the run.
    before = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}

    PHASE4B_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Run the harness in a throwaway dir (no corpus workdir arg -> the
    #    phase4a tier1/tier2 block is skipped; phase4b_confusion is unconditional).
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

        for name in CONFUSION_BINS:
            src = tmp_dir / name
            if not src.is_file():
                raise SystemExit(f"harness did not produce {name}")
            shutil.copy2(src, PHASE4B_DIR / name)

    # 4. Parse + validate the PHASE4B_CONFUSION stdout line.
    m = PHASE4B_RE.search(stdout)
    if not m:
        raise SystemExit("PHASE4B_CONFUSION line missing from harness stdout")
    if m["ok"] != "1":
        raise SystemExit(
            f"PHASE4B_CONFUSION ok={m['ok']}: the transcribed confusion matrix's error "
            "(via the REAL Confusion2String) does not match the REAL PrintConfusionMatrix's "
            "own return on the same input -- the transcription is wrong."
        )
    error1 = float(m["e1"])
    error2 = float(m["e2"])
    if error1 != error2:
        raise SystemExit(f"error1 {error1!r} != error2 {error2!r} despite ok=1 (regex/float parse bug?)")

    err_rows, err_cols, err_data = _read_bin(PHASE4B_DIR / "confusion_error.bin")
    if (err_rows, err_cols) != (1, 2):
        raise SystemExit(f"confusion_error.bin shape {(err_rows, err_cols)} != (1, 2)")
    # The .bin is the SOURCE OF TRUTH (full double precision); the stdout echo
    # (17 significant digits via std::setprecision) is cross-checked against it.
    for label, exact, echoed in [("error1", err_data[0], error1), ("error2", err_data[1], error2)]:
        if exact != echoed:
            raise SystemExit(f"confusion_error.bin {label}={exact!r} != stdout echo {echoed!r}")

    in_rows, in_cols, _ = _read_bin(PHASE4B_DIR / "confusion_input.bin")
    mat_rows, mat_cols, _ = _read_bin(PHASE4B_DIR / "confusion_matrix.bin")
    if (in_rows, in_cols) != (4, 3):
        raise SystemExit(f"confusion_input.bin shape {(in_rows, in_cols)} != (4, 3)")
    if (mat_rows, mat_cols) != (5, 5):
        raise SystemExit(f"confusion_matrix.bin shape {(mat_rows, mat_cols)} != (5, 5)")

    # 5. Regression guard: the prior-phase fixture dirs must be byte-identical after.
    after = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
    for ph in PRIOR_PHASES:
        b, a = before[ph], after[ph]
        drifted = sorted(name for name in set(b) | set(a) if b.get(name) != a.get(name))
        if drifted:
            raise SystemExit(f"REGRESSION: {ph} fixtures changed after harness run: {drifted}")

    manifest = {
        "text": (
            "Phase 4b Task 1: the LID confusion core (BagOfProcessors::PrintConfusionMatrix "
            "sentinel-decode + argmax accumulation, Confusion2String's row-normalized error "
            "aggregate). Both are pure integer-threshold/division arithmetic -- no libm -- so "
            "every fixture here is asserted STRICT BITS on every platform."
        ),
        "confusion_input": {
            "text": (
                "4 crafted 3-class rows: row A's target (col1, value 280=100*0.8+200) beats "
                "its best competitor (50) -> WIN. Row B's target (col0, value 220=100*0.2+200, "
                "score 20) loses to its best competitor (90) -> MISS. Row C has NO target "
                "sentinel at all (every value <= 150) -- STICKY QUIRK: posTarget/"
                "posBestNotTarget are declared OUTSIDE the row loop in the legacy and reset to "
                "0 only ONCE, before row 0 (BagOfProcessors.cpp:509-510), NOT at every row "
                "boundary (:533-534 resets only scoreTarget/maxScoreNotTarget) -- so row C's "
                "miss is credited to posTarget=1, inherited from row B's sentinel, not to the "
                "index-header slot 0. Row D's target (col2, value 250=100*0.5+200, score 50) "
                "TIES its best competitor (50) exactly -> the strict `>` comparison sends a "
                "tie to the miss branch too."
            ),
            "shape": [in_rows, in_cols],
        },
        "confusion_matrix": {
            "text": (
                "The harness's OWN transcription of BagOfProcessors.cpp:501-535 (the real "
                "matrix is a function-local inside the protected PrintConfusionMatrix and is "
                "never returned, even via a probe subclass), cross-validated two independent "
                "ways against the REAL compiled code on the SAME input (see confusion_error "
                "below) -- not merely asserted to be right by inspection."
            ),
            "shape": [mat_rows, mat_cols],
        },
        "confusion_error": {
            "text": (
                "1x2 = [error1, error2]. error1: the transcribed confusion_matrix fed to the "
                "REAL free Confusion2String (Helpers.hpp:365-438, static -- directly callable). "
                "error2: the REAL (protected, via a BagProbe subclass) PrintConfusionMatrix "
                "itself, called end-to-end on confusion_input directly (BagOfProcessors.cpp:"
                "474-598). error1 == error2 bit-exact (harness-asserted via the ok= stdout "
                "flag, SystemExit on failure) is the non-vacuous cross-check that the "
                "transcribed matrix matches what the real method built internally -- the only "
                "way to validate it, since PrintConfusionMatrix never returns its matrix."
            ),
            "shape": [err_rows, err_cols],
            "error1": error1,
            "error2": error2,
        },
        "dead_code_not_ported": {
            "text": (
                "Two legacy blocks are commented out and NOT ported: the classNb==2 binary "
                "ROC-curve variant (BagOfProcessors.cpp:477-499, a 10001-step threshold-sweep "
                "negative/positive histogram) -- classNb==2 falls straight into the SAME "
                "general accumulation as any other classNb>=2, pinned by the Rust "
                "dead_binary_variant_not_ported test; and a duplicate normalization+print "
                "block (:541-593) that re-derives (via cout, not error +=) exactly what the "
                "LIVE Confusion2String call at :540 already computes."
            ),
        },
    }

    manifest_path = PHASE4B_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"OK: phase4b confusion fixtures (input={in_rows}x{in_cols}, "
        f"matrix={mat_rows}x{mat_cols}, error1=error2={error1!r}), "
        f"manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
