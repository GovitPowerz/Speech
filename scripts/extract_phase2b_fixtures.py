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
TDC_CONFIG = PHASE2B_DIR / "tdc.config"
LTSV_CONFIG = PHASE2B_DIR / "ltsv.config"
LTSV_DCT_CONFIG = PHASE2B_DIR / "ltsv_dct.config"
LTSV_TINY_CONFIG = PHASE2B_DIR / "ltsv_tiny.config"
LTSV_POWERMEL_CONFIG = PHASE2B_DIR / "ltsv_powermel.config"
SIGNAL_CONFIG = PHASE2B_DIR / "signal.config"

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

# Task 2: SegProbe results2segmentation dumps (conv coeff, convolved results, the
# channel-0 hypothesis boundary walk, and the conv=none variant). Persisted into
# the committed Phase 2b dir like Phase 2's EXPECTED_SHAPES.
EXPECTED_SHAPES = {
    "r2s_conv_coeff.bin": (1, 19),
    "r2s_convolved.bin": (1, 60),
    "r2s_boundaries.bin": (2, 2),
    "r2s_boundaries_noconv.bin": (2, 2),
}

# Task 4: TdcSegmenter (Algo 1) dumps -- the REAL TimeDomainCorrel::getSegmentation
# run on the excerpt under tdc.config (both channels), plus the two-files-in-
# sequence boundary golden and the compute_errors-vs-reference scores.
TDC_EXPECTED_SHAPES = {
    "tdc_result_chan1.bin": (1, 201),
    "tdc_result_chan2.bin": (1, 201),
    "tdc_convolved_chan1.bin": (1, 201),
    "tdc_convolved_chan2.bin": (1, 201),
    "tdc_boundaries_chan1.bin": (3, 2),
    "tdc_boundaries_chan2.bin": (2, 2),
    "tdc_boundaries_file2_chan1.bin": (3, 2),
    "tdc_scores.bin": (2, 3),
}
TDC_EXTRA_FILES = ["tdc_vrcts_chan1.xml"]

# Task 5: LtsvSegmenter (Algo 2) dumps -- the REAL LongTermSpectralVariation::
# getSegmentation run on the excerpt under ltsv.config (primary, nb_DCT=0, GEMM-free
# so the real getSegmentation is the bit-golden), ltsv_dct.config (secondary,
# nb_DCT=4, applyDCT swapped for the ascending-loop applyDCTLoop), ltsv_tiny.config
# (LTSVwindow 0.001, exercises the floor-to-1 quirk), and ltsv_powermel.config
# (Task 5 review Finding 1: is_log_mel=false + tuned padding/min_speech/min_silence,
# the NON-VACUOUS decision-layer golden -- see the harness's own comment block and
# the IMPROVEMENTS.md log-mel-blowup entry for why the other three configs never
# exercise a real hysteresis/smoothing crossing).
LTSV_EXPECTED_SHAPES = {
    "ltsv_result_chan1.bin": (1, 51),
    "ltsv_result_chan2.bin": (1, 51),
    "ltsv_convolved_chan1.bin": (1, 51),
    "ltsv_convolved_chan2.bin": (1, 51),
    "ltsv_boundaries_chan1.bin": (2, 2),
    "ltsv_boundaries_chan2.bin": (2, 2),
    "ltsv_boundaries_file2_chan1.bin": (2, 2),
    "ltsv_scores.bin": (2, 3),
    "ltsv_dct_result_chan1.bin": (1, 51),
    "ltsv_dct_result_chan2.bin": (1, 51),
    "ltsv_dct_convolved_chan1.bin": (1, 51),
    "ltsv_dct_convolved_chan2.bin": (1, 51),
    "ltsv_dct_boundaries_chan1.bin": (2, 2),
    "ltsv_tiny_boundaries_chan1.bin": (2, 2),
    "ltsv_powermel_result_chan1.bin": (1, 51),
    "ltsv_powermel_result_chan2.bin": (1, 51),
    "ltsv_powermel_convolved_chan1.bin": (1, 51),
    "ltsv_powermel_convolved_chan2.bin": (1, 51),
    "ltsv_powermel_boundaries_chan1.bin": (4, 2),
    "ltsv_powermel_boundaries_chan2.bin": (2, 2),
}
LTSV_EXTRA_FILES = ["ltsv_vrcts_chan1.xml"]
LTSV_CONFIG_FILES = ["ltsv.config", "ltsv_dct.config", "ltsv_tiny.config", "ltsv_powermel.config"]

# Task 6: BlstmSignalSegmenter (Algo 4) dumps -- the FIRST driver with the NN in the
# chain. Each variant's PRIMARY golden dumps come from a TRANSCRIPTION of
# getSegmentation (BLSTMSignalSegmenter.cpp:93-398) with ONLY the feedForwardBackward
# call swapped for the ascending-loop reimpl family (signalReimplFFB); a SECONDARY
# probe runs the REAL getSegmentation beside it (segment count + types match exactly
# or abort, boundary max-dt recorded in SEG_STRUCT). Real net (1_worker_1.config
# topology, 33671 weights) on the shared 2-channel excerpt. Three variants:
#   window0   -- BLSTM_window 0        (full-sequence, plain FFB; real=frameCount/ssr)
#   overlap   -- BLSTM_window 0.01, shift 0.0005 (overlap FFB; window_shift 4 == ssr
#                is the ONLY non-broken regime -- see the signal-overlap-oob note)
#   noOverlap -- BLSTM_window 0.5, shift 0 (truncate FFB; the =0.0 poisoning + the
#                ssr-division sizing gate)
# frame_count=16001, ssr=4. window0/noOverlap real_vec_size=4000; overlap=4001.
SIGNAL_EXPECTED_SHAPES = {
    "signal_window0_signalraw_chan1.bin": (1, 16001),
    "signal_window0_signal_chan1.bin": (1, 16001),
    "signal_window0_result_chan1.bin": (1, 4000),
    "signal_window0_convolved_chan1.bin": (1, 4000),
    "signal_window0_boundaries_chan1.bin": (2, 2),
    "signal_window0_scores.bin": (2, 3),
    "signal_overlap_signalraw_chan1.bin": (1, 16001),
    "signal_overlap_signal_chan1.bin": (1, 16001),
    "signal_overlap_result_chan1.bin": (1, 4001),
    "signal_overlap_convolved_chan1.bin": (1, 4001),
    "signal_overlap_boundaries_chan1.bin": (2, 2),
    "signal_overlap_scores.bin": (2, 3),
    "signal_noOverlap_signalraw_chan1.bin": (1, 16001),
    "signal_noOverlap_signal_chan1.bin": (1, 16001),
    "signal_noOverlap_result_chan1.bin": (1, 4000),
    "signal_noOverlap_convolved_chan1.bin": (1, 4000),
    "signal_noOverlap_boundaries_chan1.bin": (2, 2),
    "signal_noOverlap_scores.bin": (2, 3),
    "signal_noOverlap_boundaries_file1_chan1.bin": (2, 2),
    "signal_noOverlap_boundaries_file2_chan1.bin": (2, 2),
}
SIGNAL_EXTRA_FILES = [
    "signal_window0_vrcts_chan1.xml",
    "signal_overlap_vrcts_chan1.xml",
    "signal_noOverlap_vrcts_chan1.xml",
]
SIGNAL_CONFIG_FILES = ["signal.config"]

# Task 7: BlstmSpectralSegmenter (Algo 3, no pitch pass) dumps -- the REAL
# 1_worker_1.config's algorithm. Like the signal driver, the spectral result_vec
# comes from feedForwardBackward (BLSTMSpectralSegmenter.cpp:740); the PRIMARY golden
# dumps come from a TRANSCRIPTION of getSegmentation (:593-887 MINUS the pitch second
# pass :757-805, which the Task-7 configs short-circuit via TDCwindow 0) with ONLY
# that FFB call swapped for signalReimplFFB (the Task 6 reimpl family, reused). A
# SECONDARY probe runs the REAL getSegmentation beside it (structural equality or
# abort, SEG_STRUCT max-dt recorded). Real net (33671 weights), input width D=11
# (< 23 -> the LSTM input topRows tolerance). Three variants (BLSTM_shift overridden):
#   real      -- 1_worker_1.config AS-IS (BLSTM_shift 0.8 -> overlap window_shift 80)
#   overlap   -- BLSTM_shift 0.01 (heavier overlap, window_shift 1; the cross-channel
#                reuse golden -- chan-2 result seeded by chan-1 via the reused buffer)
#   noOverlap -- BLSTM_shift 0 (truncate FFB; window_size 324, window_shift 1)
# frame_count 16001, ssr 4, spectrum_shift_in_frames 80, periodogram rows T=201,
# real_vec_size = 201/4/1/1/1 = 50 for all three variants.
SPECTRAL_EXPECTED_SHAPES = {
    "spectral_real_inputseq_chan1.bin": (201, 11),
    "spectral_real_result_chan1.bin": (1, 50),
    "spectral_real_convolved_chan1.bin": (1, 50),
    "spectral_real_boundaries_chan1.bin": (3, 2),
    "spectral_real_scores.bin": (2, 3),
    "spectral_overlap_inputseq_chan1.bin": (201, 11),
    "spectral_overlap_result_chan1.bin": (1, 50),
    "spectral_overlap_result_chan2.bin": (1, 50),
    "spectral_overlap_convolved_chan1.bin": (1, 50),
    "spectral_overlap_convolved_chan2.bin": (1, 50),
    "spectral_overlap_boundaries_chan1.bin": (3, 2),
    "spectral_overlap_boundaries_chan2.bin": (2, 2),
    "spectral_overlap_scores.bin": (2, 3),
    "spectral_noOverlap_inputseq_chan1.bin": (201, 11),
    "spectral_noOverlap_result_chan1.bin": (1, 50),
    "spectral_noOverlap_convolved_chan1.bin": (1, 50),
    "spectral_noOverlap_boundaries_chan1.bin": (3, 2),
    "spectral_noOverlap_scores.bin": (2, 3),
    "spectral_noOverlap_boundaries_file1_chan1.bin": (3, 2),
    "spectral_noOverlap_boundaries_file2_chan1.bin": (3, 2),
    "spectral_noOverlap_params_file2.bin": (1, 4),
}
SPECTRAL_EXTRA_FILES = [
    "spectral_real_vrcts_chan1.xml",
    "spectral_overlap_vrcts_chan1.xml",
    "spectral_noOverlap_vrcts_chan1.xml",
]
# Task 7 reuses 1_worker_1.config (nn_config, already committed under phase0); no new
# config file is added under phase2b.
SPECTRAL_CONFIG_FILES: list[str] = []

# Task 8: BlstmSpectralSegmenter pitch-homothety SECOND pass (Algo 3 complete). The
# "pitch" variant = the real spectral base (BLSTM_shift 0.8 -> overlap) + TDC keys that
# activate the pitch pass (BLSTMSpectralSegmenter.cpp:757-805): TDCwindow 0.032 ->
# TDC_window_size 128 > 0 (gate on), TDCshift 0.01, lags (0.002,0.016), balance 0.7,
# hamming-257 window param 0.8. getPitch walks the pass-1 SPEECH segment ([0, ~1.18s]),
# returns pitch ~250 Hz (nonzero -> the warp does real work); the warp -> refilterbank/DCT
# -> re-forward shifts the boundary from pass-1 1.1831 to pass-2 1.2597. DUMP QUIRK
# (:848-850): the pass-1 result/convolved dumps are preserved (result_chan1 == the T7 real
# result, byte-for-byte) while the pass-2 result/inputseq/boundaries are dumped separately;
# the final boundaries dump reflects pass-2. The pass-1 result_chan1/inputseq_chan1 are the
# same recipe as T7 real (proves pass-1 is untouched); the pass-2 dumps pin the second
# pass. SEG_STRUCT site=spectral_pitch (reimpl pass-2 structure == real Eigen pass-2 or
# abort). The measured chan-1 pitch is printed as SPECTRAL_PITCH measured_pitch_chan1=<v>
# and recorded in the manifest.
SPECTRAL_PITCH_EXPECTED_SHAPES = {
    "spectral_pitch_pitch_chan1.bin": (1, 1),
    "spectral_pitch_result_chan1.bin": (1, 50),
    "spectral_pitch_convolved_chan1.bin": (1, 50),
    "spectral_pitch_inputseq_chan1.bin": (201, 11),
    "spectral_pitch_result_pass2_chan1.bin": (1, 50),
    "spectral_pitch_convolved_pass2_chan1.bin": (1, 50),
    "spectral_pitch_inputseq_pass2_chan1.bin": (201, 11),
    "spectral_pitch_boundaries_chan1.bin": (3, 2),
}
SPECTRAL_PITCH_EXTRA_FILES = ["spectral_pitch_vrcts_chan1.xml"]
# Reuses 1_worker_1.config with in-harness set_val TDC overrides; no new config file.
SPECTRAL_PITCH_CONFIG_FILES: list[str] = []

SPECTRAL_PITCH_RE = re.compile(r"^SPECTRAL_PITCH measured_pitch_chan1=(?P<pitch>[-0-9.eE+]+)$", re.MULTILINE)

# SEG_STRUCT self-test lines from the SECONDARY real-forward probe (spec decision 3):
# reimpl-driven segment structure (count + types) must equal the real Eigen path's,
# with the boundary max-delta recorded. ok=0 (or a missing line) -> SystemExit.
SEG_STRUCT_RE = re.compile(
    r"^SEG_STRUCT site=(?P<site>\w+) ok=(?P<ok>[01]) max_dt=(?P<max_dt>[-0-9.eE+]+)$",
    re.MULTILINE,
)
EXPECTED_SEG_STRUCT_SITES = {
    "signal_window0",
    "signal_overlap",
    "signal_noOverlap",
    "spectral_real",
    "spectral_overlap",
    "spectral_noOverlap",
    "spectral_pitch",
}


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


def _parse_seg_struct(stdout: str) -> dict[str, dict[str, object]]:
    """Parse the SEG_STRUCT secondary-probe lines; ok=0 or missing site -> SystemExit."""
    sites: dict[str, dict[str, object]] = {}
    for match in SEG_STRUCT_RE.finditer(stdout):
        sites[match["site"]] = {"ok": int(match["ok"]), "max_dt": float(match["max_dt"])}
    missing = EXPECTED_SEG_STRUCT_SITES - set(sites)
    if missing:
        raise SystemExit(f"SEG_STRUCT lines missing from harness stdout: {sorted(missing)}")
    failed = sorted(name for name, rec in sites.items() if rec["ok"] != 1)
    if failed:
        raise SystemExit(f"SEG_STRUCT failed (ok=0, structural mismatch) for: {failed}")
    return sites


def _read_bin_shape(path: Path) -> tuple[int, int]:
    """Read just the (rows, cols) header of a legacy `.bin` (i64 LE, i64 LE, ...)."""
    with path.open("rb") as f:
        rows = int.from_bytes(f.read(8), "little", signed=True)
        cols = int.from_bytes(f.read(8), "little", signed=True)
    return rows, cols


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
                str(TDC_CONFIG),
                str(LTSV_CONFIG),
                str(LTSV_DCT_CONFIG),
                str(LTSV_TINY_CONFIG),
                str(LTSV_POWERMEL_CONFIG),
                str(SIGNAL_CONFIG),
            ]
        )
        # Task 2 (SegProbe results2segmentation dumps): persist these into the
        # committed Phase 2b dir -- everything else the harness writes into tmp_dir
        # is prior-phase output already covered by Phase 1/2's own fixtures.
        for name in EXPECTED_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        # Task 4 (TdcSegmenter dumps): result/convolved/boundaries/scores + the
        # zero-offset VRCTS xml (0b-ii byte-equivalence closure golden).
        for name in TDC_EXPECTED_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        for name in TDC_EXTRA_FILES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        # Task 5 (LtsvSegmenter dumps): result/convolved/boundaries/scores (primary +
        # DCT secondary + tiny-window variants) + the zero-offset VRCTS xml.
        for name in LTSV_EXPECTED_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        for name in LTSV_EXTRA_FILES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        # Task 6 (BlstmSignalSegmenter dumps): signalraw/signal/result/convolved/
        # boundaries/scores per variant + the zero-offset VRCTS xml + the two-files
        # noOverlap lifecycle boundaries.
        for name in SIGNAL_EXPECTED_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        for name in SIGNAL_EXTRA_FILES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        # Task 7 (BlstmSpectralSegmenter dumps): inputseq/result/convolved/boundaries/
        # scores per variant (chan 2 for the overlap cross-channel reuse golden) + the
        # zero-offset VRCTS xml + the two-files noOverlap lifecycle boundaries + params.
        for name in SPECTRAL_EXPECTED_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        for name in SPECTRAL_EXTRA_FILES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        # Task 8 (BlstmSpectralSegmenter pitch second pass): pass-1 (preserved) +
        # pass-2 result/inputseq/convolved/boundaries + the pitch scalar + the
        # pass-2-driven VRCTS.
        for name in SPECTRAL_PITCH_EXPECTED_SHAPES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)
        for name in SPECTRAL_PITCH_EXTRA_FILES:
            shutil.copy2(tmp_dir / name, PHASE2B_DIR / name)

    # 4. Regression guard: the committed fixture dirs must be byte-identical after
    #    the run (the fmtr upgrade must not perturb any prior harness dump).
    after = {"phase1": _hash_tree(PHASE1_DIR), "phase2": _hash_tree(PHASE2_DIR)}
    for phase in ("phase1", "phase2"):
        b, a = before[phase], after[phase]
        drifted = sorted(name for name in set(b) | set(a) if b.get(name) != a.get(name))
        if drifted:
            raise SystemExit(f"REGRESSION: {phase} fixtures changed after harness run: {drifted}")

    # 5. Parse the fmtr self-test (all ok=1 or SystemExit).
    fmtr_cases = _parse_fmtr_checks(stdout)

    # 5b. Parse the Task 6 SEG_STRUCT secondary-probe lines (all ok=1 or SystemExit;
    #     a structural mismatch would have already aborted the harness itself).
    seg_struct = _parse_seg_struct(stdout)

    # 5c. Parse the Task 8 measured pitch (SPECTRAL_PITCH line; recorded in the manifest
    #     and pinned by the oracle-gated pitch scalar test).
    pitch_match = SPECTRAL_PITCH_RE.search(stdout)
    if not pitch_match:
        raise SystemExit("SPECTRAL_PITCH line missing from harness stdout")
    measured_pitch = float(pitch_match["pitch"])

    # 6. Call-site scan (spec decision 7). Expected 0 uncommented callers each; a
    #    nonzero count is a scope change -> STOP.
    callsites = {fn: _scan_callsites(fn) for fn in DEFERRED_FNS}
    nonzero = {fn: hits for fn, hits in callsites.items() if hits}
    if nonzero:
        detail = json.dumps(nonzero, indent=2)
        raise SystemExit(f"BLOCKED: found UNCOMMENTED call sites of a deferred function (scope change for the controller):\n{detail}")

    # 7. Task 2 shape sync: the persisted r2s_*.bin dumps must match EXPECTED_SHAPES.
    shape_mismatches = []
    for name, expected in EXPECTED_SHAPES.items():
        got = _read_bin_shape(PHASE2B_DIR / name)
        if got != expected:
            shape_mismatches.append({"file": name, "expected": expected, "got": got})
    if shape_mismatches:
        raise SystemExit(f"r2s_*.bin shape mismatch: {shape_mismatches}")

    # 7b. Task 4 shape sync: the persisted tdc_*.bin dumps must match TDC_EXPECTED_SHAPES.
    tdc_shape_mismatches = []
    for name, expected in TDC_EXPECTED_SHAPES.items():
        got = _read_bin_shape(PHASE2B_DIR / name)
        if got != expected:
            tdc_shape_mismatches.append({"file": name, "expected": expected, "got": got})
    if tdc_shape_mismatches:
        raise SystemExit(f"tdc_*.bin shape mismatch: {tdc_shape_mismatches}")

    # 7c. Task 5 shape sync: the persisted ltsv_*.bin dumps must match LTSV_EXPECTED_SHAPES.
    ltsv_shape_mismatches = []
    for name, expected in LTSV_EXPECTED_SHAPES.items():
        got = _read_bin_shape(PHASE2B_DIR / name)
        if got != expected:
            ltsv_shape_mismatches.append({"file": name, "expected": expected, "got": got})
    if ltsv_shape_mismatches:
        raise SystemExit(f"ltsv_*.bin shape mismatch: {ltsv_shape_mismatches}")

    # 7d. Task 6 shape sync: the persisted signal_*.bin dumps must match SIGNAL_EXPECTED_SHAPES.
    signal_shape_mismatches = []
    for name, expected in SIGNAL_EXPECTED_SHAPES.items():
        got = _read_bin_shape(PHASE2B_DIR / name)
        if got != expected:
            signal_shape_mismatches.append({"file": name, "expected": expected, "got": got})
    if signal_shape_mismatches:
        raise SystemExit(f"signal_*.bin shape mismatch: {signal_shape_mismatches}")

    # 7e. Task 7 shape sync: the persisted spectral_*.bin dumps must match SPECTRAL_EXPECTED_SHAPES.
    spectral_shape_mismatches = []
    for name, expected in SPECTRAL_EXPECTED_SHAPES.items():
        got = _read_bin_shape(PHASE2B_DIR / name)
        if got != expected:
            spectral_shape_mismatches.append({"file": name, "expected": expected, "got": got})
    if spectral_shape_mismatches:
        raise SystemExit(f"spectral_*.bin shape mismatch: {spectral_shape_mismatches}")

    # 7f. Task 8 shape sync: the pitch second-pass dumps must match their shapes, and
    #     the preserved pass-1 result/inputseq must be BYTE-IDENTICAL to the T7 real
    #     variant (the dump quirk: the externalized pass-1 result survives the pitch pass).
    spectral_pitch_shape_mismatches = []
    for name, expected in SPECTRAL_PITCH_EXPECTED_SHAPES.items():
        got = _read_bin_shape(PHASE2B_DIR / name)
        if got != expected:
            spectral_pitch_shape_mismatches.append({"file": name, "expected": expected, "got": got})
    if spectral_pitch_shape_mismatches:
        raise SystemExit(f"spectral_pitch_*.bin shape mismatch: {spectral_pitch_shape_mismatches}")
    for pass1, real in (
        ("spectral_pitch_result_chan1.bin", "spectral_real_result_chan1.bin"),
        ("spectral_pitch_inputseq_chan1.bin", "spectral_real_inputseq_chan1.bin"),
    ):
        if (PHASE2B_DIR / pass1).read_bytes() != (PHASE2B_DIR / real).read_bytes():
            raise SystemExit(f"pitch pass-1 dump {pass1} must be byte-identical to the T7 {real} (the dump quirk: pass 1 is preserved, pass 2 differs)")

    # 8. Compiler version (same g++ selection as build.sh).
    gxx = _run(["bash", "-c", 'ls "$(brew --prefix)"/bin/g++-* | sort -V | tail -1']).strip()
    compiler = _run([gxx, "--version"]).splitlines()[0].strip()

    manifest = {
        "compiler": compiler,
        "flags": ("-O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE -include boost/math/special_functions/round.hpp"),
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
        "results_to_segmentation": {
            "text": (
                "Phase 2b Task 2: SegProbe (a Segmenter subclass exposing the "
                "protected buildFromConf/results2segmentation/updateSegmentation via "
                "`using`) built from the REAL 1_worker_1.config (BLSTM prefix): "
                "BLSTM_convolution_window_size=9 -> the real _ConvolutionCoeff is a "
                "19-tap hHCw kernel (r2s_conv_coeff.bin, 1x19, sums to 1.0). A "
                "deterministic synthetic 1x60 result row (r[k] = 0.2 + "
                "0.7*((k*13) mod 17)/16) spanning the real decision thresholds "
                "(rising~0.752, falling~0.374) is run through the REAL "
                "results2segmentation(seg, 0.04, 0.0, results, targets, chan=0, "
                "SPEECH) against a REAL Segmentation built from the harness's own "
                "AudioStruct (Segmentation has no scalar-duration ctor -- only "
                "Segmentation(AudioStruct&, double), so 'Segmentation seg(6.0)' in "
                "the task brief is shorthand, not a literal signature). "
                "r2s_convolved.bin (1x60) is the POST-convolution results (in-place "
                "mutation observed through the same Eigen::Ref); r2s_boundaries.bin "
                "(Nx2 begin/type) walks the resulting seg._Classification[0] "
                "hypothesis list. The conv=none variant re-reads the config with "
                'BLSTM_convolution_window_size forced to "0" via the established '
                "_Params erase/set pattern (here: assignment, since the key is "
                "present) -- _ConvolutionCoeff collapses to 0 cols, the "
                "`.cols() > 1` gate in results2segmentation is false, and the "
                "harness asserts in-process that the noconv results are BIT-IDENTICAL "
                "to the pre-call copy (SystemExit if not) before dumping "
                "r2s_boundaries_noconv.bin."
            ),
            "expected_shapes": {k: list(v) for k, v in EXPECTED_SHAPES.items()},
        },
        "tdc_segmenter": {
            "text": (
                "Phase 2b Task 4: TdcSegmenter (Algo 1). CHECK-FIRST outcome: "
                "TimeDomainCorrel::getSegmentation (TimeDomainCorrel.cpp:93-259) calls "
                "no NN (autocorrelation is cwise+sum; the only libm is fmath::log/cos "
                "via classifySequence + the hamming window coefficients), so the REAL "
                "compiled getSegmentation is used AS-IS as the golden source -- no "
                "transcription at all, the strongest oracle of the phase. TdcProbe "
                "(harness-local, derives TimeDomainCorrel) exposes the protected "
                "Segmenter fields needed to independently replicate the pre-convolution "
                "result_vec (a getSegmentation-local variable, not otherwise "
                "observable): _MinMaxLag/_Balance are PRIVATE even to a derived class, "
                "so the harness re-reads TDC_lags from the same ConfigFile and "
                "reproduces the ctor's >=0.0 clamp (TDC_balance is unclamped and unused "
                "by the replication, since classifySequence takes it as an implicit "
                "member read inside the real, unmodified method). tdc.config: window "
                "0.032s/shift 0.01s -> window_size=128, full_window=257 (odd), "
                "min_lag=16, max_lag=128, window_shift=80 frames (0.01s, already >= "
                "1/rate so the floor is a no-op); vec_size=201 "
                "(16001 frames / 80, ceil since not exact). Dumps run on the shared "
                "2-channel excerpt AudioStruct (offset 0.35, dur 2.0, rate 8000): "
                "tdc_result_chan{1,2}.bin (pre-conv, 1x201), tdc_convolved_chan{1,2}.bin "
                "(post results2segmentation, 1x201), tdc_boundaries_chan{1,2}.bin "
                "(Nx2 begin/type hypothesis walk). tdc_vrcts_chan1.xml is the REAL "
                "toFile_VRCTS output on a SEPARATE zero-offset AudioStruct (same wav, "
                "audio_offset=0.0): the Rust Segmentation container has no "
                "_AudioOffset field (folds to 0.0 in to_vrcts_string), so byte "
                "equivalence on 'identical Segmentations' (spec S10) requires the "
                "harness side to also be offset-free, sidestepping the +0.35 that "
                "would otherwise bake into sigdur/stime/etime with no Rust-side "
                "equivalent. tdc_scores.bin (2x3 Pfa/Pmiss/ErrorRate for SPEECH) is "
                "compute_errors() against a PROGRAMMATIC reference built via direct "
                "label_segment calls on a fresh Segmentation._Reference (public "
                "member): two SPEECH spans [0.4,0.9) and [1.2,1.6) per channel, "
                "_WordErrorRate left at its default NbWords=-1 (no WER pass, "
                "matching the Rust nb_words<0 -> wer=None contract). "
                "tdc_boundaries_file2_chan1.bin is the TWO-FILES golden: the SAME "
                "TdcProbe instance runs getSegmentation a second time on the excerpt "
                "into a fresh Segmentation, pinning the _WindowShift quantization "
                "lifecycle (idempotent once quantized: round(x*rate)/rate on an "
                "already-quantized value is a fixed point, so file 2's boundaries "
                "differ from file 1's only through the hysteresis decision's own "
                "state, not a shift drift -- both are dumped so the Rust test can "
                "assert this explicitly rather than assume it)."
            ),
            "constants": {
                "rate": 8000,
                "window_size_frames": 128,
                "full_window_frames": 257,
                "min_lag_frames": 16,
                "max_lag_frames": 128,
                "window_shift_frames": 80,
                "window_shift_sec": 0.01,
                "vec_size": 201,
                "reference_spans_sec": [[0.4, 0.9], [1.2, 1.6]],
            },
            "expected_shapes": {k: list(v) for k, v in TDC_EXPECTED_SHAPES.items()},
            "extra_files": TDC_EXTRA_FILES,
        },
        "ltsv_segmenter": {
            "text": (
                "Phase 2b Task 5: LtsvSegmenter (Algo 2). PRIMARY config (ltsv.config, "
                "nb_DCT=0): LongTermSpectralVariation::getSegmentation "
                "(LongTermSpectralVariation.cpp:130-407) calls applyFilterBank only (no "
                "DCT -> no Eigen GEMM), so the REAL compiled getSegmentation is the "
                "bit-golden AS-IS, same strength oracle as TdcSegmenter. LtsvProbe "
                "(harness-local, derives LongTermSpectralVariation) exposes the "
                "protected fields needed to independently replicate the pre-convolution "
                "decimated result_vec (a getSegmentation-local, not otherwise "
                "observable) via the REAL public classifySequence -- same pattern as "
                "TdcProbe. ltsv.config: spectrum_order=8 (window_size=256, "
                "periodogram_length=129), spectrum_shift=0.01s (80 frames), nb_bins=26 "
                "mel filters (is_log_mel=true), nb_DCT=0, LTSVwindow=0.3s -> half-window "
                "15 periodogram frames, LTSVshift=0.04s -> shift 4 periodogram frames. "
                "The driver's own periodogram span uses begin_frame=0/end_frame= "
                "audio.getFrameCount() (LongTermSpectralVariation.cpp:293-294) -- ONE "
                "MORE than build_input_sequence's end=ncols()-1 call, the frameCount+1 "
                "vec_size quirk (spec S4.2): periodogram vec_size = ceil((frameCount+1)/"
                "spectrum_shift) = 201 rows, then real_vec_size = ceil(201/4) = 51 "
                "result columns. SECONDARY config (ltsv_dct.config, nb_DCT=4): "
                "applyDCT's Eigen GEMM diverges from the ascending-loop port at this "
                "shape (same family the Phase 1 GEMM_CHECK pre-check flagged), so this "
                "variant runs getSegmentation's body as an in-process reimplementation "
                "with ONLY the applyDCT call swapped for applyDCTLoop (T7's ascending- "
                "loop port, already defined for the Phase 1 DCT golden) -- periodogram, "
                "mel filterbank, LTSV classifySequence, and results2segmentation all "
                "stay the REAL compiled machinery, called through the same LtsvProbe "
                "surface. ltsv_tiny.config (LTSVwindow=0.001s): round(0.001*8000/2/80) "
                "== 0, floored to 1 (LongTermSpectralVariation.cpp:257-258) -- the "
                "LTSV-standalone floor-to-1 quirk, which DIFFERS from the BLSTM "
                "spectral segmenter's floor-to-0-disables (BLSTMSpectralSegmenter.cpp), "
                "ported as written. ltsv_vrcts_chan1.xml is the REAL toFile_VRCTS "
                "output on a SEPARATE zero-offset AudioStruct (same rationale as "
                "tdc_vrcts_chan1.xml). ltsv_scores.bin is compute_errors() against the "
                "same programmatic two-span reference recipe as TdcSegmenter. "
                "ltsv_boundaries_file2_chan1.bin is the TWO-FILES golden: the SAME "
                "LtsvProbe instance runs getSegmentation a second time on the excerpt, "
                "pinning the _WindowShift/_SpectrumShift quantization lifecycle (both "
                "are round(x*rate/N)*N/rate re-quantizations of an already-quantized "
                "value -- idempotent, same determinism-not-statefulness scope as the "
                "TDC two-files golden). "
                "POWER-SCALE variant (ltsv_powermel.config, Task 5 review Finding 1): "
                "the PRIMARY/SECONDARY/TINY configs above all share is_log_mel=true, "
                "which drives ltsv_classify_sequence's score to ~1e51 (see the "
                "IMPROVEMENTS.md log-mel-blowup entry) -- every column sits far above "
                "_DecisionThreshRising (0.6), so update_segmentation/smooth_segmentation "
                "never see a real threshold crossing and all three goldens collapse to "
                "one always-SPEECH span: VACUOUS for the decision layer (hysteresis, "
                "area gates, suppression, padding are never exercised). "
                "ltsv_powermel.config is byte-identical to ltsv.config except "
                "is_log_mel=false (restores the power-scale periodogram the LTSV "
                "formula's 1e-12 mean floor assumes) and smaller-but-still-nonzero "
                "speech_padding/min_speech/min_silence (0.05/0.1/0.1 vs 0.1/0.2/0.2 -- "
                "config keys only, tuned so a genuine ~0.56s silence gap between two "
                "real threshold crossings survives smoothing instead of being padded "
                "shut). Under this config the REAL compiled getSegmentation produces, "
                "on channel 1, the non-vacuous structure SPEECH[0,0.9892)/OTHER["
                "0.9892,1.3472)/SPEECH[1.3472,2.0) -- two genuine rising/falling "
                "hysteresis crossings with the smoothing pipeline (sanitize, "
                "suppress_short x3 per side, add_padding x2) actively shaping the "
                "final boundaries, not a no-op. Channel 2 stays a single always-SPEECH "
                "span under the SAME config (its own periodogram content never drops "
                "far enough below _DecisionThreshFalling to accrue a qualifying ending "
                "area) -- dumped honestly alongside channel 1 rather than omitted."
            ),
            "constants": {
                "rate": 8000,
                "spectrum_order": 8,
                "periodogram_length": 129,
                "spectrum_shift_frames": 80,
                "spectrum_shift_sec": 0.01,
                "periodogram_vec_size": 201,
                "ltsv_half_window_frames": 15,
                "ltsv_shift_frames": 4,
                "real_vec_size": 51,
                "ltsv_tiny_window_half_frames": 1,
                "reference_spans_sec": [[0.4, 0.9], [1.2, 1.6]],
                "powermel_chan1_boundaries_sec": [[0.0, "SPEECH"], [0.9892, "OTHER"], [1.3472, "SPEECH"], [2.0, "END"]],
                "powermel_chan2_boundaries_sec": [[0.0, "SPEECH"], [2.0, "END"]],
            },
            "expected_shapes": {k: list(v) for k, v in LTSV_EXPECTED_SHAPES.items()},
            "extra_files": LTSV_EXTRA_FILES,
            "config_files": LTSV_CONFIG_FILES,
        },
        "signal_segmenter": {
            "text": (
                "Phase 2b Task 6: BlstmSignalSegmenter (Algo 4), the FIRST driver with "
                "the NN in the chain. BLSTMSignalSegmenter::getSegmentation "
                "(BLSTMSignalSegmenter.cpp:93-398) produces its result_vec via "
                "_BLSTMNeuralNetwork.feedForwardBackward (:260), whose real Eigen GEMMs "
                "diverge from the ascending-loop port at the real net's k>=23 shapes "
                "(Phase 2 NN_PROBE) and the layer-0 GEMM seeds the recurrence, so the "
                "divergence propagates non-locally. SignalProbe (harness-local, derives "
                "BLSTMSignalSegmenter) therefore transcribes getSegmentation LIVE code "
                "(the param math :96-108/:221-254, results2segmentation, compute_errors) "
                "faithfully but swaps ONLY the FFB call for signalReimplFFB -- the "
                "ascending-loop reimpl family (t6MakeLstmSteps/t6MakeDenseSteps + "
                "blstmFeedForwardT6 + the plain/truncate/overlap windowed drivers) built "
                "on the file-scope kernel statics reused from Phase 2. SECONDARY probe "
                "(spec decision 3): the REAL compiled getSegmentation runs beside the "
                "transcription on the same inputs; the reimpl-driven and real Segmentation "
                "must match in segment count + types EXACTLY (a mismatch aborts the "
                "harness), and the boundary max-delta is recorded as "
                "SEG_STRUCT site=signal_<variant> ok=1 max_dt=<measured> (all three "
                "measured 0.0 here -- the real net's posteriors on this excerpt are "
                "~0.002-0.03, far BELOW _DecisionThreshRising (0.6), so NO speech is "
                "ever detected and all variants collapse to the untouched always-Other "
                "SEED hypothesis ([Other@0, End@dur]; VRCTS SegmentList empty, "
                "spdur=0.00) insensitive to the posterior jitter; the result_vec "
                "goldens, pinned bit-exact via BlstmSignalSegmenter::last_result_rows "
                "(a port-side observation point, no legacy counterpart), NOT the "
                "boundary structure, are what pin the NN chain bit-exactly). The real "
                "net (1_worker_1.config topology, 33671 weights) "
                "expects 23 inputs; signal mode feeds a 1-column input "
                "(audio._Data.row(chan).transpose()), which the LSTM input projection "
                "accepts via inputW.topRows(cols) width tolerance -- the legacy's own "
                "behavior, kept. THREE variants (real net + real config keys, overridden "
                "window/shift): window0 (BLSTM_window 0 -> full-sequence plain FFB; "
                "real_vec_size = frameCount/ssr = 16001/4 = 4000), overlap (BLSTM_window "
                "0.01, shift 0.0005 -> overlap FFB; window_size 40, window_shift 4; "
                "real_vec_size = ceil(16001/4) = 4001), noOverlap (BLSTM_window 0.5, "
                "shift 0 -> truncate FFB via the noOverlap poisoning; window_size floored "
                "to (round(0.5*8000)/4)*4 = 4000, window_shift clamped to 1, "
                "real_vec_size = ceil(16001/1)/4/1/1/1 = 4000). The brief's ORIGINAL "
                "overlap params (window 0.5, shift 0.1) are a broken-as-committed legacy "
                "path: the overlap result-vec sizing ceil(frameCount/shift) is far too "
                "small for the OverLap FFB driver's write index (begin/ssr + lengthShort "
                "~ frameCount/ssr), so the REAL getSegmentation aborts on it under Eigen "
                "assertions (verified) and heap-corrupts under release -DNDEBUG; the "
                "OverLap path only survives when window_shift <= ssr (== 4 samples, i.e. "
                "shift <= 0.0005s), the degenerate regime used above. See the "
                "IMPROVEMENTS.md signal-overlap-oob entry. Dumps per variant, chan 1: "
                "signalraw (PRE-preemph, the timing-divergence golden), signal (POST -- "
                "differs since BLSTM_preemph_ratio 0.97 > 0), result (pre-conv row), "
                "convolved (post results2segmentation row), boundaries (Nx2 begin/type), "
                "vrcts (REAL toFile_VRCTS on the reimpl-driven Segmentation, zero-offset "
                "audio, 0b-ii byte-equivalence closure), scores (2x3 Pfa/Pmiss/ErrorRate "
                "vs the same two-span programmatic reference as TDC/LTSV). The noOverlap "
                "variant gets the TWO-FILES golden: unlike TDC/LTSV (idempotent shift "
                "re-quantization), signal noOverlap assigns _WindowShift = 0.0 mid-run "
                "(:376) -- so file 2 (SAME probe) re-enters getSegmentation with "
                "_WindowShift == 0.0, :99 rounds 0.0*rate -> 0 -> shift<1 -> noOverlap "
                "re-triggers, :107-108 re-clamp to 1/rate. This is the REAL lifecycle "
                "re-entry (a genuine round trip through mutated state), pinned by "
                "signal_noOverlap_boundaries_file{1,2}_chan1.bin -- both are dumped so "
                "the Rust test COMPARES rather than assumes; here they are equal (the "
                "output is the same untouched always-Other SEED hypothesis on both "
                "files), which the test asserts and its doc-comment explains honestly; "
                "it additionally asserts the discriminating result-vec length/"
                "window_shift_sec observable (4000/0.0 on both files, not the 16001 "
                "no-reset overlap-branch counterfactual) so the round trip through the "
                "mutated state is pinned, not merely the coincidental boundary equality."
            ),
            "constants": {
                "rate": 8000,
                "frame_count": 16001,
                "sub_sampling_ratio": 4,
                "window0_real_vec_size": 4000,
                "overlap_window_size_samples": 40,
                "overlap_window_shift_samples": 4,
                "overlap_real_vec_size": 4001,
                "nooverlap_window_size_samples": 4000,
                "nooverlap_window_shift_samples": 1,
                "nooverlap_real_vec_size": 4000,
                "reference_spans_sec": [[0.4, 0.9], [1.2, 1.6]],
                "broken_overlap_params_sec": {"window": 0.5, "shift": 0.1},
            },
            "seg_struct": {k: v for k, v in sorted(seg_struct.items()) if k.startswith("signal_")},
            "expected_shapes": {k: list(v) for k, v in SIGNAL_EXPECTED_SHAPES.items()},
            "extra_files": SIGNAL_EXTRA_FILES,
            "config_files": SIGNAL_CONFIG_FILES,
        },
        "spectral_segmenter": {
            "text": (
                "Phase 2b Task 7: BlstmSpectralSegmenter (Algo 3, no pitch pass), the "
                "REAL 1_worker_1.config's algorithm. BLSTMSpectralSegmenter::"
                "getSegmentation (BLSTMSpectralSegmenter.cpp:593-887) produces its "
                "result_vec via _BLSTMNeuralNetwork.feedForwardBackward (:740), whose "
                "real Eigen GEMMs diverge from the ascending-loop port. SpectralProbe "
                "(harness-local, derives BLSTMSpectralSegmenter) transcribes the "
                "getSegmentation body MINUS the pitch second pass (:757-805, which the "
                "Task-7 configs short-circuit -- TDCwindow 0 -> TDC_window_size 0 -> the "
                "`if (TDC_window_size > 0)` gate at :757 is false) with ONLY the FFB call "
                "swapped for signalReimplFFB (the Task 6 reimpl family, reused: the "
                "plain/truncate/overlap windowed drivers are input-column-agnostic and "
                "the leftCols crop stays FALSE at D=11<23); the spectral param "
                "derivation (initSpectralAnalysis/getWindowingCoeff/getTemporalConvolution"
                "/getLTSVParam/getTDCParam/getBLSTMParam/getLTSV/getBLSTMInputSequence) "
                "calls the REAL compiled protected helpers. SECONDARY probe (spec "
                "decision 3): the REAL compiled getSegmentation runs beside the "
                "transcription; segment count + types must match EXACTLY (abort on "
                "mismatch), boundary max-delta recorded as SEG_STRUCT site="
                "spectral_<variant> ok=1 max_dt=<measured> (all three measured 0.0 -- the "
                "reimpl and real Eigen paths agree structurally). UNLIKE the signal "
                "driver, the real trained net on the REAL spectral features DOES cross "
                "the rising threshold (0.752): chan-1 detects speech [0, ~1.18s] "
                "([SPEECH@0, OTHER@~1.183, END@2.0]), so the decision layer (rising "
                "crossing + linear interpolation + hysteresis area) is genuinely "
                "exercised by these goldens, not collapsed to the seed. CROSS-CHANNEL "
                "REUSE QUIRK (the load-bearing layout fact): result_vec is allocated ONCE "
                "by getBLSTMParam (:631) and REUSED across channels (:740); under the "
                "overlap FFB (accumulate-in-place :664 += then /= count) channel 2 SEEDS "
                "from channel 1's post-division contents -- reproduced by holding one "
                "result_vec across the channel loop; the overlap variant dumps BOTH "
                "channels (chan-1 result[:5] ~0.997 vs chan-2 ~0.973, chan-2 result "
                "contaminated by chan-1's carry-over) so the Rust cross-channel golden "
                "pins it (a fresh-buffer chan-2 run must DIFFER). THREE variants (real "
                "net, BLSTM_shift overridden): real (1_worker_1.config AS-IS, BLSTM_shift "
                "0.8 -> overlap window_shift 80, 3 windows), overlap (BLSTM_shift 0.01 -> "
                "window_shift 1, heavier overlap, the cross-channel golden -- verified NO "
                "OOB: max overlap write index 50 == result_vec rows), noOverlap "
                "(BLSTM_shift 0 -> truncate FFB via the noOverlap poisoning; window_size "
                "floored to (round(3.25*8000/80)/4)*4 = 324, window_shift clamped to 1). "
                "All three: T=201 periodogram rows, real_vec_size = 201/4/1/1/1 = 50 "
                "(getBLSTMParam does the ssr-division UNCONDITIONALLY -- the `if noOverlap"
                "` guard at :487 is commented out, unlike the signal driver's gated "
                "version at BLSTMSignalSegmenter.cpp:239). Dumps per variant, chan 1 (+ "
                "chan 2 for overlap): inputseq (201x11, re-validates build_input_sequence "
                "under driver conditions), result (pre-conv row 1x50), convolved (post "
                "results2segmentation), boundaries (Nx2 begin/type), vrcts (REAL "
                "toFile_VRCTS on the reimpl-driven Segmentation, zero-offset audio, 0b-ii "
                "byte-equivalence), scores (2x3 Pfa/Pmiss/ErrorRate vs the same two-span "
                "programmatic reference as TDC/LTSV/signal). The noOverlap variant gets "
                "the TWO-FILES golden: spectral noOverlap assigns _WindowShift = 0.0 at "
                ":885 (post-dump). So file 2 (SAME probe) re-enters getBLSTMParam with "
                "_WindowShift == 0.0: :445 rounds 0.0 -> 0, :446 -> noOverlap re-triggers, "
                ":452 clamps shift to 1, :453 rewrites _WindowShift = 1*80/rate. Pinned "
                "by spectral_noOverlap_boundaries_file{1,2}_chan1.bin (both dumped, the "
                "test COMPARES rather than assumes) + the discriminating file2 params row "
                "spectral_noOverlap_params_file2.bin (1x4: window_shift=1, window_size="
                "324, spectrum_shift_in_frames=80, real_vec_size=50) so the "
                "quantization-lifecycle round trip is pinned, not merely coincidental "
                "boundary equality."
            ),
            "constants": {
                "rate": 8000,
                "frame_count": 16001,
                "sub_sampling_ratio": 4,
                "spectrum_shift_in_frames": 80,
                "periodogram_rows": 201,
                "input_width_d": 11,
                "real_vec_size": 50,
                "real_window_size_periodogram_frames": 163,
                "real_window_shift_periodogram_frames": 80,
                "overlap_window_shift_periodogram_frames": 1,
                "nooverlap_window_size_periodogram_frames": 324,
                "nooverlap_window_shift_periodogram_frames": 1,
                "file2_params_row": {
                    "window_shift": 1,
                    "window_size": 324,
                    "spectrum_shift_in_frames": 80,
                    "real_vec_size": 50,
                },
                "reference_spans_sec": [[0.4, 0.9], [1.2, 1.6]],
            },
            "seg_struct": {k: v for k, v in sorted(seg_struct.items()) if k.startswith("spectral_") and k != "spectral_pitch"},
            "expected_shapes": {k: list(v) for k, v in SPECTRAL_EXPECTED_SHAPES.items()},
            "extra_files": SPECTRAL_EXTRA_FILES,
            "config_files": SPECTRAL_CONFIG_FILES,
        },
        "spectral_pitch_pass": {
            "text": (
                "Phase 2b Task 8: the pitch-homothety SECOND pass completes Algo 3 "
                "(BLSTMSpectralSegmenter.cpp:757-805). SpectralProbe transcribes the pitch "
                "block gated on TDC_window_size > 0: getTDCParam REAL derives the TDC "
                "params (TDCwindow 0.032 -> TDC_window_size 128, full 257, shift 80, "
                "min_lag 16, max_lag 128, hamming-257 param 0.8); getPitch REAL (NN-free: "
                "getSequence + computePitch, an autocorrelation argmax) walks the PASS-1 "
                "SPEECH segment ([0, ~1.18s]) and returns the measured chan-1 pitch; the "
                "periodogram is warped by coeff = pitch/300 via applyHomothety (a pure "
                "linear-interpolation column warp, bit-exact off-libm); the mel filterbank "
                "+ DCT are re-run on the WARPED periodogram (applyDCTLoop, same ascending "
                "recipe as pass 1); the input sequence is rebuilt WITH THE OLD (pass-1) "
                "LTSV column (:792 -- NOT recomputed on the warp, a load-bearing quirk); "
                "the reimpl FFB re-forwards; clearClassification (:800) + "
                "results2segmentation (:801) OVERWRITE the pass-1 boundaries; the error/"
                "classif slots are OVERWRITTEN (:802-803). The re-seg is guarded on "
                "pitch > 0 (:791). NON-VACUITY (the brief's mandate): measured pitch ~250 "
                "Hz > 0, the warped periodogram differs, the pass-2 result differs from "
                "pass-1 at ALL 50 positions, and the boundary crossing MOVES from pass-1 "
                "1.1831 to pass-2 1.2597 -- the pitch pass does real work on this excerpt. "
                "DUMP QUIRK reproduced (:848-850): the externalized result golden preserves "
                "PASS-1 (spectral_pitch_result_chan1.bin is byte-identical to the T7 "
                "spectral_real_result_chan1.bin) even though the final boundaries "
                "(spectral_pitch_boundaries_chan1.bin) are PASS-2; the pass-2 result/"
                "inputseq/convolved are dumped separately (spectral_pitch_*_pass2_chan1). "
                "SECONDARY probe: the REAL Eigen getSegmentation runs its OWN pitch pass "
                "beside the transcription; SEG_STRUCT site=spectral_pitch pins pass-2 "
                "structural equality (count + types) with the measured boundary max-dt. "
                "The VRCTS golden (spectral_pitch_vrcts_chan1.xml) is the REAL toFile_VRCTS "
                "on the pass-2-driven Segmentation."
            ),
            "measured_pitch_chan1": measured_pitch,
            "pass1_boundary_crossing_sec": 1.1831,
            "pass2_boundary_crossing_sec": 1.2597,
            "constants": {
                "tdc_window_size": 128,
                "tdc_full_window_size": 257,
                "tdc_window_shift": 80,
                "min_lag": 16,
                "max_lag": 128,
                "balance": 0.7,
                "tdc_windowing_type": "hamming",
                "tdc_windowing_param": 0.8,
                "homothety_coeff": measured_pitch / 300.0,
            },
            "seg_struct": {k: v for k, v in sorted(seg_struct.items()) if k == "spectral_pitch"},
            "expected_shapes": {k: list(v) for k, v in SPECTRAL_PITCH_EXPECTED_SHAPES.items()},
            "extra_files": SPECTRAL_PITCH_EXTRA_FILES,
            "config_files": SPECTRAL_PITCH_CONFIG_FILES,
        },
    }

    manifest_path = PHASE2B_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    counts = manifest["callsite_checks"]["counts"]
    print(
        f"OK: {len(fmtr_cases)} FMTR_CHECK all ok, {len(seg_struct)} SEG_STRUCT all ok, "
        f"callsites {counts}, r2s shapes ok, tdc shapes ok, ltsv shapes ok, "
        f"signal shapes ok, spectral shapes ok, pitch shapes ok "
        f"(pitch_chan1={measured_pitch:.6g}), manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
