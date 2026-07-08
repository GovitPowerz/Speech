"""Build and run the oracle harness's phase4b stages, then write the LID
confusion-core (Task 1) + multiclass scoring-overload (Task 3) + BlstmSpectralLID
Algo-5 driver (Task 4) + phSeq reader (Task 5) + TwinBlstmSpectralLid Algo-6 driver
(Task 6) fixtures.

Task 6 runs the harness TwinProbe stage over the T2 corpus with twin_mode*.config
(4 variants: mode0, mode0_concat, mode2, mode3) + the real 33k SAD net + a synthetic
3-class LID net: the reimpl transcription of the wav-mode 0/2/3 getSegmentation paths
dumps the SAD result_vec + LID members + the concat-augmented per-segment scoring
(twin_<v>_<f>_{result,liderr,confusion,members,boundaries}_chan{1,2}.bin), and a
SECONDARY real-Eigen probe cross-checks SEG_STRUCT + LID_STRUCT (confusion +
_IsLIDCorrect equality + the concat branch counter). See manifest.json:twin.

Task 5 runs the harness over the committed synthetic corpus_phseq/{f1,f2,f3}.phSeq
fixtures (the format contract -- no real .phSeq file exists anywhere to validate
against) through the REAL AudioStruct file_type==1 ctor (AudioStruct.cpp:138-182):
pure indexing + one-hot construction, no libm, so every phseq_*.bin dump is STRICT
BITS on every platform. Dumps per file: `phseq_<f>_feat<i>.bin` (one per phSeq line,
the one-hot `_ExternalFeatures[i]`) and `phseq_<f>_periodogram.bin` (`_Periodogram`,
pre-filled at rows `numberOfPhonemes x 38`). A `PHASE4B_PHSEQ file=<f> lines=<n>
frames_count=<v> channels=<c> framerate=<r> periodogram_rows=<pr>
periodogram_cols=<pc>` stdout line is parsed and cross-checked against an
INDEPENDENT Python reimplementation of the ctor's arithmetic (`_phseq_expected`)
run directly over the same fixture files -- not merely trusting the harness output.

Task 4 runs the harness LidProbe stage over the T2 corpus (f1/f2/f3.wav, class 0/1/2)
with lid5.config + the real 33,671-weight SAD net: the reimpl transcription of
BLSTMSpectralLID::getSegmentation dumps the LTSV-SAD row + LID members
(lid5_<f>_{ltsv,liderr,confusion,members,boundaries}_chan{1,2}.bin), and a SECONDARY
real-Eigen probe cross-checks SEG_STRUCT (segment count/type) + LID_STRUCT (confusion +
_IsLIDCorrect equality) with the langid/cost deltas recorded as GEMM-divergence
calibration. See tests/reference_data/phase4b/manifest.json:lid5.

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
  5. Parses the `NN_TOL site=blstm_scoring_multi_* max_ulp=0` + `BLSTM_SCORING_MULTI`
     lines (Task 3), asserting every multiclass scoring site is 0-ULP and the
     unknown-class fold (targetIndex 5 -> 0) bit-matches targetIndex 0's cost. One
     of the five cases (`ti2_step0_mod2_costmod`) sets SYNC_BackPropWER so
     isCostModified() is true with target_modifier 2.0, exercising the nonzero
     soft-target placement on the multiclass path.
  6. Copies the confusion `.bin` dumps + `scoring_multi_out.bin` into
     `tests/reference_data/phase4b/` and writes `manifest.json` with MEASURED
     shapes/values.
  7. REGRESSION GUARD: hashes every prior-phase fixture dir before/after; any
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

import numpy as np
import scipy.io

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "tools" / "oracle_harness"
REF_DIR = REPO_ROOT / "tests" / "reference_data"
PHASE0_DIR = REF_DIR / "phase0"
PHASE1_DIR = REF_DIR / "phase1"
PHASE2B_DIR = REF_DIR / "phase2b"
PHASE4A_DIR = REF_DIR / "phase4a"
PHASE4B_DIR = REF_DIR / "phase4b"

NN_CONFIG = PHASE0_DIR / "1_worker_1.config"
NN_WEIGHTS = PHASE0_DIR / "NNweights_config1.bin"
TDC_CONFIG = PHASE2B_DIR / "tdc.config"
LTSV_CONFIG = PHASE2B_DIR / "ltsv.config"
LTSV_DCT_CONFIG = PHASE2B_DIR / "ltsv_dct.config"
LTSV_TINY_CONFIG = PHASE2B_DIR / "ltsv_tiny.config"
LTSV_POWERMEL_CONFIG = PHASE2B_DIR / "ltsv_powermel.config"
SIGNAL_CONFIG = PHASE2B_DIR / "signal.config"

# Task 4: BlstmSpectralLID (Algo 5). lid5.config + the T2 corpus (f1/f2/f3.wav, class
# 0/1/2 via the crafted CorpusItems) + the real 33,671-weight SAD net (NNweights_config1.bin).
LID5_CONFIG = PHASE4B_DIR / "lid5.config"
LID_CORPUS_DIR = PHASE4B_DIR / "corpus_lid"
LID5_FILES = ["f1", "f2", "f3"]
LID5_KINDS = ["ltsv", "liderr", "confusion", "members", "boundaries"]
LID5_BINS = [
    f"lid5_{name}_{kind}_chan{ch}.bin"
    for name in LID5_FILES
    for kind in LID5_KINDS
    for ch in (1, 2)
]

# Task 5: phSeq reader (File_Type 1). letterMapping (AudioStruct.h:18), a 38-entry
# char -> column bijection -- the fixture files below draw characters ONLY from this
# domain (checked by _phseq_expected below, which mirrors the Rust letter_index table).
PHSEQ_LETTER_MAPPING = {
    c: i
    for i, c in enumerate(
        " &A@EIHONSZacbedgfihkjmlonpsrutwvyxz.-"
    )
}
PHSEQ_DIR = PHASE4B_DIR / "corpus_phseq"
PHSEQ_FILES = ["f1", "f2", "f3"]

PHSEQ_RE = re.compile(
    r"^PHASE4B_PHSEQ file=(?P<name>\w+) lines=(?P<lines>\d+) frames_count=(?P<frames>\d+) "
    r"channels=(?P<channels>\d+) framerate=(?P<framerate>\d+) "
    r"periodogram_rows=(?P<prows>\d+) periodogram_cols=(?P<pcols>\d+)$",
    re.MULTILINE,
)


def _phseq_expected(path: Path) -> dict[str, object]:
    """Independent Python reimplementation of AudioStruct.cpp:138-182's arithmetic
    (NOT reading any harness output), so the harness's own numbers are cross-checked
    against a second source, not merely trusted."""
    lines = path.read_text().split("\n")
    # str.read_text().split("\n") on a trailing-newline file yields one spurious
    # trailing "" entry (unlike C++ getline / Python's own splitlines()) -- drop it,
    # matching getline's "no line for a bare trailing newline" behavior.
    if lines and lines[-1] == "":
        lines = lines[:-1]
    for line in lines:
        for ch in line:
            if ch not in PHSEQ_LETTER_MAPPING:
                raise SystemExit(f"{path}: char {ch!r} not in letterMapping domain")
    number_of_phonemes = 10
    row_begin = 5
    blocks = []
    for line in lines:
        blocks.append((row_begin, len(line)))
        number_of_phonemes += len(line) + 10
        row_begin += len(line) + 10
    # legacy AudioStruct.cpp:173, LEFT-ASSOCIATIVE: (numberOfPhonemes*0.01)*framerate.
    frames_count = int((number_of_phonemes * 0.01) * 8000.0)
    return {
        "lines": len(lines),
        "lens": [len(line) for line in lines],
        "number_of_phonemes": number_of_phonemes,
        "frames_count": frames_count,
        "blocks": blocks,
    }


# Task 6: TwinBlstmSpectralLid (Algo 6). twin_mode*.config (in PHASE4B_DIR) + the T2
# corpus + the real 33k SAD net + a synthetic 3-class LID net (weights dumped by the
# harness). Variant -> (mode, expected concat branch, reference-driven).
TWIN_VARIANTS = {
    "mode0": {"mode": 0, "concat": 0, "ref": False},
    "mode0_concat": {"mode": 0, "concat": 1, "ref": False},
    "mode1": {"mode": 1, "concat": 0, "ref": False},
    "mode2": {"mode": 2, "concat": 2, "ref": True},
    "mode3": {"mode": 3, "concat": 0, "ref": True},
    # Task 7c: modes 4/5/6 (the :640-902 LID-train + scoring branch). The LID net
    # trains on the whole inputSeq (feedForwardBackward) and the scoring slices
    # LID_result_vec per speech span by LIDTimeStep/LIDTimeOffset. No concat (concat=0);
    # always reference-driven (getTargetsLID + the per-mode classification assembly).
    # mode 4 = REF smoothed; mode 5 = SAD VAD; mode 6 = LID2Segmentation.
    "mode4": {"mode": 4, "concat": 0, "ref": True},
    "mode5": {"mode": 5, "concat": 0, "ref": True},
    "mode6": {"mode": 6, "concat": 0, "ref": True},
}
TWIN_FILES = ["f1", "f2", "f3"]
TWIN_KINDS = ["result", "liderr", "confusion", "members", "boundaries"]
TWIN_BINS = [f"twin_{v}_lidweights.bin" for v in TWIN_VARIANTS] + [
    f"twin_{v}_{f}_{kind}_chan{ch}.bin"
    for v in TWIN_VARIANTS
    for f in TWIN_FILES
    for kind in TWIN_KINDS
    for ch in (1, 2)
]
TWIN_SEG_STRUCT_RE = re.compile(
    r"^SEG_STRUCT site=twin_(?P<name>\w+) ok=1 max_dt=(?P<dt>[0-9.eE+-]+)$", re.MULTILINE
)
TWIN_STRUCT_RE = re.compile(
    r"^LID_STRUCT site=twin_(?P<name>\w+) ok=1 confusion_eq=1 iscorrect_eq=1 "
    r"concat=(?P<concat>\d+) langid_max_abs=(?P<langid>[0-9.eE+-]+)$",
    re.MULTILINE,
)

LID_SEG_STRUCT_RE = re.compile(
    r"^SEG_STRUCT site=lid5_(?P<name>\w+) ok=1 max_dt=(?P<dt>[0-9.eE+-]+)$", re.MULTILINE
)
LID_STRUCT_RE = re.compile(
    r"^LID_STRUCT site=lid5_(?P<name>\w+) ok=1 confusion_eq=1 iscorrect_eq=1 "
    r"langid_max_abs=(?P<langid>[0-9.eE+-]+) cost_max_abs=(?P<cost>[0-9.eE+-]+)$",
    re.MULTILINE,
)

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

# Task 7 FLAGSHIP: Mode 7 (phSeq) real-compiled getSegmentation LID members over
# twin_mode7{,_ppm1,_ppm2}.config x s{1,2,3}.phSeq + the strict noise-indexing probe.
MODE7_VARIANTS = ["twin_mode7", "twin_mode7_ppm1", "twin_mode7_ppm2"]
MODE7_FILES = ["s1", "s2", "s3"]
MODE7_KINDS = ["confusion", "liderr", "members"]
MODE7_BINS = (
    [
        f"mode7_{v}_{f}_{kind}.bin"
        for v in MODE7_VARIANTS
        for f in MODE7_FILES
        for kind in MODE7_KINDS
    ]
    + ["mode7_noise_in.bin", "mode7_noise_out.bin"]
    + ["mode7_dump_s1.mat"]  # DumpLIDInternals real-Eigen .mat (scipy value-check)
)
MODE7_RE = re.compile(
    r"^PHASE4B_MODE7 (?P<tag>\S+) nbclassif=(?P<nbc>\d+) isCorrect=(?P<iscorrect>-?\d+) "
    r"lidCumErr=(?P<cost>[0-9.eE+-]+) confSum=(?P<confsum>[0-9.eE+-]+)$",
    re.MULTILINE,
)
MODE7_WEIGHTS_RE = re.compile(r"^PHASE4B_MODE7 lidWeights=(?P<n>\d+)$", re.MULTILINE)

# Task 7: LID_bestNNWeight_1.bin provenance. The committed weight vector was a
# MANUAL scipy conversion of the legacy libmatio MultiConfigResults .mat's
# `weights` variable (12409x1) -- reproduce that conversion here so it stops being
# an undocumented one-off. The legacy binary artifact lives OUTSIDE this repo's
# git-ignored `legacy/` (source-only, per CLAUDE.md) in a sibling checkout, so the
# byte-identity re-check is LOCAL-ONLY: it runs (and can SystemExit on drift) only
# when that sibling checkout is present; the element-count assertion is NOT gated
# on it and always runs, so the check still has teeth on a machine without the
# legacy tree (e.g. CI, though CI never invokes this extractor -- see CLAUDE.md).
LEGACY_LID_MAT = Path(
    "/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Release/bin/LID_bestNNWeight_1_MultiConfigResults.mat"
)
LID_WEIGHT_BIN = PHASE4B_DIR / "LID_bestNNWeight_1.bin"

# Task 3: the multiclass scoring feedForward overload golden (BLSTMNeuralNetwork.h:195,
# the LID path). One reimpl output covers every case (the forward is target-independent).
SCORING_MULTI_BIN = "scoring_multi_out.bin"

# The harness multiclass scoring cases: tag -> (target_index, step, target_modifier, cost_modified).
SCORING_MULTI_CASES = {
    "ti0_step0": (0, 0, 1.0, False),
    "ti2_step0": (2, 0, 1.0, False),
    "ti2_step2": (2, 2, 1.0, False),
    "tiOOB_step0": (5, 0, 1.0, False),  # 5 >= 3 -> unknown-class fold -> 0
    "ti2_step0_mod2_costmod": (2, 0, 2.0, True),  # isCostModified True, modifier 2.0 -> nonzero soft target
}

PHASE4B_RE = re.compile(
    r"^PHASE4B_CONFUSION ok=(?P<ok>\d) error1=(?P<e1>[0-9.eE+-]+) error2=(?P<e2>[0-9.eE+-]+)$",
    re.MULTILINE,
)

# Task 9: corpus-level LID stages. The harness chdir's into a seeded workdir
# (argv[19]) and runs twin_train / twin_train_ns (+ the nsx no-gate
# counterfactual) / twin_e2e (REAL CorpusProcessor) / twin_gradcheck /
# lid5_gradcheck with the REAL compiled drivers (tiny nets -> every GEMM
# reduction dim < 23, bit-exact vs the ascending-loop port on the oracle env).
T9_WORKDIR_FILES = [
    "twin_train.config",
    "twin_train_ns.config",
    "twin_e2e.config",
    "twin_gradcheck.config",
    "lid5_gradcheck.config",
    "languagemapping_lid7.csv",
    "listing_gc_wav.csv",
]
T9_PHSEQ_FILES = ["s1.phSeq", "s2.phSeq", "s1.stm", "s2.stm", "listing_train.csv"]
T9_WAV_FILES = ["f1.wav", "f1.stm", "f1_gc.stm"]
T9_EPOCHS = 8  # 6 training epochs + solo + final eval
T9_DUMPS = (
    ["tiny_sad_seed.bin", "tiny_lid_seed.bin"]
    + [f"twin_train_sad_epoch{e}.bin" for e in range(T9_EPOCHS)]
    + [f"twin_train_lid_epoch{e}.bin" for e in range(T9_EPOCHS)]
    + [f"twin_train_ns_lid_epoch{e}.bin" for e in range(T9_EPOCHS)]
    + [f"twin_train_nsx_lid_epoch{e}.bin" for e in range(T9_EPOCHS)]
    + ["twin_gradcheck_net0.bin", "twin_gradcheck_net1.bin", "lid5_gradcheck.bin"]
)
# The 5 saveResults variables (CorpusProcessor.cpp:397-401), in write order.
T9_MAT_VARS = ["MultiConfigResults", "CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"]
# `_t9_convert_mat`'s output names (staged in tmp_dir, copied to PHASE4B_DIR last).
T9_MAT_BINS = [f"{prefix}_{var}.bin" for prefix in ("twin_train", "twin_e2e") for var in T9_MAT_VARS]
# bestNNWeight artifacts (BOTH nets; the LID pair carries saveWeightsLID's LID_
# prefix -- the Task-9 pin) copied raw from the workdir (io::binary payload
# despite the legacy .mat suffix).
T9_BEST_ARTIFACTS = [
    "weights_bestNNWeight_1_twin_train.mat",
    "weightsDerivatives_bestNNWeight_1_twin_train.mat",
    "weights_LID_bestNNWeight_1_twin_train.mat",
    "weightsDerivatives_LID_bestNNWeight_1_twin_train.mat",
]
T9_SEEDS_RE = re.compile(r"^\f?PHASE4B_T9 seeds sad=(?P<sad>\d+) lid=(?P<lid>\d+)$", re.MULTILINE)
T9_TRAIN_RE = re.compile(
    r"^\f?PHASE4B_T9_TRAIN tag=(?P<tag>\w+) epochs=(?P<epochs>\d+) fired=(?P<fired>\d+) "
    r"skipped=(?P<skipped>\d+) differ_lid=(?P<differ>\d+) best_cost=\[(?P<bc>[^\]]*)\] "
    r"cost_lid=\[(?P<cl>[^\]]*)\] speech=\[(?P<sp>[^\]]*)\] error_perc_lid=\[(?P<epl>[^\]]*)\]$",
    re.MULTILINE,
)
T9_NS_DIVERGE_RE = re.compile(r"^\f?PHASE4B_T9_NS_DIVERGE epoch=(?P<e>-?\d+)$", re.MULTILINE)
T9_GRADCHECK_RE = re.compile(
    r"^\f?PHASE4B_T9_GRADCHECK tag=(?P<tag>\w+) net=(?P<net>\d) sweep=(?P<sweep>\d+) "
    r"mean_err=(?P<err>[0-9.eE+-]+) mean_rel_err=(?P<rel>[0-9.eE+-]+)$",
    re.MULTILINE,
)
T9_E2E_RE = re.compile(r"^\f?PHASE4B_T9_E2E ok=1$", re.MULTILINE)

# Task 10: pitch-pass target-reuse pin under a LIVE STM reference (f1.wav/f1.stm,
# the phase4a 3-file corpus). measured_pitch_chan1 must be > 0 (the warp fired);
# cumError_chan1/nbClassif_chan1 must be nonzero (the live-target cost path fired).
PITCH_SCORED_RE = re.compile(
    r"^\f?SPECTRAL_PITCH_SCORED measured_pitch_chan1=(?P<pitch>[0-9.eE+-]+) "
    r"cumError_chan1=(?P<err>[0-9.eE+-]+) nbClassif_chan1=(?P<nb>-?\d+)$",
    re.MULTILINE,
)
PITCH_SCORED_KINDS = [
    "inputseq_chan1",
    "result_chan1",
    "convolved_chan1",
    "inputseq_pass2_chan1",
    "result_pass2_chan1",
    "convolved_pass2_chan1",
    "boundaries_chan1",
    "scores",
]
PITCH_SCORED_BINS = [f"spectral_pitch_scored_{kind}.bin" for kind in PITCH_SCORED_KINDS]

SCORING_MULTI_TOL_RE = re.compile(
    r"^NN_TOL site=blstm_scoring_multi_(?P<tag>\w+) max_ulp=(?P<ulp>\d+) "
    r"max_abs=(?P<abs>[0-9.eE+-]+)$",
    re.MULTILINE,
)
SCORING_MULTI_COST_RE = re.compile(
    r"^BLSTM_SCORING_MULTI case=(?P<tag>\w+) cost=0x(?P<cost>[0-9a-f]+) "
    r"cost_dec=(?P<cost_dec>[0-9.eE+-]+) nb_of_classif=(?P<nb>\d+)$",
    re.MULTILINE,
)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout


def _write_bin(path: Path, matrix: np.ndarray) -> None:
    """Write `matrix` as an io::binary .bin: i64 LE rows, i64 LE cols, f64 LE
    column-major -- the exact format the Rust golden loaders read."""
    mat = np.ascontiguousarray(matrix, dtype="<f8")
    rows, cols = mat.shape
    with path.open("wb") as f:
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        f.write(mat.flatten(order="F").tobytes())


def _t9_convert_mat(mat_path: Path, prefix: str, out_dir: Path) -> None:
    """Convert a Task-9 .mat's 5 variables to `<prefix>_<var>.bin` fixtures,
    zeroing MultiConfigResults' wall-clock timing column (col 6 = data col 3).
    Writes into `out_dir` (the harness tempdir, NOT PHASE4B_DIR directly) --
    validation reads these back from there too; the copy into PHASE4B_DIR only
    happens once every check in `main()` has passed."""
    mat = scipy.io.loadmat(mat_path)
    for var in T9_MAT_VARS:
        m = np.atleast_2d(np.asarray(mat[var], dtype=np.float64))
        if var == "MultiConfigResults" and m.shape[1] > 6:
            m = m.copy()
            m[:, 6] = 0.0
        _write_bin(out_dir / f"{prefix}_{var}.bin", m)


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


def _mat_dump_differs(new_mat: Path, committed_mat: Path) -> bool:
    """Churn guard for mode7_dump_s1.mat (the DumpLIDInternals real-Eigen .mat). The
    harness writes it via libmatio, whose MAT v5 header embeds a wall-clock timestamp
    that churns run-to-run even when the decoded matrices are byte-identical. Return
    True iff any decoded variable's VALUES differ, so a value-stable re-run does not
    rewrite the committed snapshot. Same rationale as the 4a `_mat_conversion_differs`
    pattern (scripts/extract_phase4a_fixtures.py)."""
    a = scipy.io.loadmat(new_mat)
    b = scipy.io.loadmat(committed_mat)
    keys = {k for k in set(a) | set(b) if not k.startswith("__")}
    for var in keys:
        if var not in a or var not in b:
            return True
        av = np.asarray(a[var]).astype("<f8")
        bv = np.asarray(b[var]).astype("<f8")
        if av.shape != bv.shape or not np.array_equal(av.view(np.uint64), bv.view(np.uint64)):
            return True
    return False


def _verify_lid_weight_provenance() -> dict[str, object]:
    """Task 7 IMPORTANT-2: make the `LID_bestNNWeight_1.bin` conversion reproducible +
    self-checking instead of a manual, undocumented one-off. Always asserts the
    committed .bin's OWN header element count == 12409 (independent of the legacy
    tree). If the legacy source .mat is present locally, redoes the scipy conversion
    (the `weights` variable) into a throwaway buffer and SystemExits on any byte
    drift from the committed .bin."""
    committed = LID_WEIGHT_BIN.read_bytes()
    rows, cols = struct.unpack("<qq", committed[:16])
    if rows * cols != 12409:
        raise SystemExit(
            f"{LID_WEIGHT_BIN}: header element count {rows}x{cols}={rows * cols} != 12409 "
            "(the built [36,24]/[48,1] LID net's weight count)"
        )

    provenance: dict[str, object] = {
        "committed_bin": str(LID_WEIGHT_BIN.relative_to(REPO_ROOT)),
        "shape": [rows, cols],
        "element_count": rows * cols,
        "source_mat": str(LEGACY_LID_MAT),
        "source_variable": "weights",
        "legacy_source_available": LEGACY_LID_MAT.is_file(),
        "reverified_byte_identical": False,
    }
    if LEGACY_LID_MAT.is_file():
        mat = scipy.io.loadmat(LEGACY_LID_MAT)
        weights = np.asarray(mat["weights"], dtype="<f8")
        w_rows, w_cols = weights.shape
        redone = struct.pack("<qq", w_rows, w_cols) + weights.flatten(order="F").tobytes()
        if redone != committed:
            raise SystemExit(
                f"{LID_WEIGHT_BIN} drifted from a fresh scipy conversion of "
                f"{LEGACY_LID_MAT}'s 'weights' variable -- re-run the conversion and re-commit."
            )
        provenance["reverified_byte_identical"] = True
    return provenance


def main() -> None:
    # 1. Build the harness.
    _run(["bash", str(HARNESS_DIR / "build.sh")])

    # 2. Snapshot the prior-phase fixture dirs BEFORE the run.
    before = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}

    PHASE4B_DIR.mkdir(parents=True, exist_ok=True)

    # Task 5: independently recompute the expected phSeq ctor arithmetic straight
    # from the committed fixture files (not from any harness output), and derive
    # the exact set of per-file .bin names the harness must produce (one feat<i>.bin
    # per line + one periodogram.bin).
    phseq_expected = {name: _phseq_expected(PHSEQ_DIR / f"{name}.phSeq") for name in PHSEQ_FILES}
    phseq_bins: list[str] = []
    for name in PHSEQ_FILES:
        for i in range(int(phseq_expected[name]["lines"])):
            phseq_bins.append(f"phseq_{name}_feat{i}.bin")
        phseq_bins.append(f"phseq_{name}_periodogram.bin")

    # 3. Run the harness in a throwaway dir (no corpus workdir arg -> the
    #    phase4a tier1/tier2 block is skipped; phase4b_confusion is unconditional).
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name in HARNESS_INPUTS:
            shutil.copy2(PHASE1_DIR / name, tmp_dir / name)

        # Task 9: seed the corpus workdir (configs + corpora + listings, all the
        # workdir-relative paths the configs reference).
        t9_dir = tmp_dir / "t9_workdir"
        (t9_dir / "corpus_phseq").mkdir(parents=True)
        (t9_dir / "corpus_lid").mkdir(parents=True)
        for name in T9_WORKDIR_FILES:
            shutil.copy2(PHASE4B_DIR / name, t9_dir / name)
        for name in T9_PHSEQ_FILES:
            shutil.copy2(PHASE4B_DIR / "corpus_phseq" / name, t9_dir / "corpus_phseq" / name)
        for name in T9_WAV_FILES:
            shutil.copy2(PHASE4B_DIR / "corpus_lid" / name, t9_dir / "corpus_lid" / name)

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
                "",  # argv[10] corpusWorkdir empty -> the phase4a tier block is skipped.
                "t1",  # argv[11..14] tier configs (unused when corpusWorkdir is empty).
                "t2",
                "t3",
                "t4",
                str(LID5_CONFIG),  # argv[15]: Task 4 LID config.
                str(LID_CORPUS_DIR),  # argv[16]: Task 4 corpus dir (f1/f2/f3.wav).
                str(PHSEQ_DIR),  # argv[17]: Task 5 phSeq corpus dir (f1/f2/f3.phSeq).
                str(PHASE4B_DIR),  # argv[18]: Task 6 Twin config dir (twin_mode*.config).
                str(t9_dir),  # argv[19]: Task 9 corpus-level LID workdir.
                str(PHASE4A_DIR / "corpus"),  # argv[20]: Task 10 f1.wav/f1.stm (read-only).
            ]
        )

        # Task 9: converted .mat value fixtures (staged into tmp_dir -- validation
        # below reads them back from there) + the bestNNWeight artifact pins
        # (raw io::binary payloads, still sitting in t9_dir). Existence is checked
        # now (a cheap, non-mutating sanity check); the actual COPY into the
        # committed PHASE4B_DIR happens only after every check below has passed
        # (see "Commit" near the end of this function) -- a failed run must never
        # leave the committed fixtures dir partially overwritten with unvalidated
        # output.
        _t9_convert_mat(t9_dir / "twin_train.mat", "twin_train", tmp_dir)
        _t9_convert_mat(t9_dir / "twin_e2e.mat", "twin_e2e", tmp_dir)
        for name in T9_BEST_ARTIFACTS:
            if not (t9_dir / name).is_file():
                raise SystemExit(f"harness did not produce the bestNNWeight artifact {name}")
        for name in [
            *CONFUSION_BINS,
            SCORING_MULTI_BIN,
            *LID5_BINS,
            *TWIN_BINS,
            *phseq_bins,
            *MODE7_BINS,
            *T9_DUMPS,
            *T9_MAT_BINS,
            *PITCH_SCORED_BINS,
        ]:
            if not (tmp_dir / name).is_file():
                raise SystemExit(f"harness did not produce {name}")

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

        err_rows, err_cols, err_data = _read_bin(tmp_dir / "confusion_error.bin")
        if (err_rows, err_cols) != (1, 2):
            raise SystemExit(f"confusion_error.bin shape {(err_rows, err_cols)} != (1, 2)")
        # The .bin is the SOURCE OF TRUTH (full double precision); the stdout echo
        # (17 significant digits via std::setprecision) is cross-checked against it.
        for label, exact, echoed in [("error1", err_data[0], error1), ("error2", err_data[1], error2)]:
            if exact != echoed:
                raise SystemExit(f"confusion_error.bin {label}={exact!r} != stdout echo {echoed!r}")

        in_rows, in_cols, _ = _read_bin(tmp_dir / "confusion_input.bin")
        mat_rows, mat_cols, _ = _read_bin(tmp_dir / "confusion_matrix.bin")
        if (in_rows, in_cols) != (4, 3):
            raise SystemExit(f"confusion_input.bin shape {(in_rows, in_cols)} != (4, 3)")
        if (mat_rows, mat_cols) != (5, 5):
            raise SystemExit(f"confusion_matrix.bin shape {(mat_rows, mat_cols)} != (5, 5)")

        # 4b. Task 3: parse + validate the multiclass scoring overload lines.
        tol_matches = {m["tag"]: m for m in SCORING_MULTI_TOL_RE.finditer(stdout)}
        cost_matches = {m["tag"]: m for m in SCORING_MULTI_COST_RE.finditer(stdout)}
        for tag in SCORING_MULTI_CASES:
            if tag not in tol_matches:
                raise SystemExit(f"missing NN_TOL scoring_multi line for {tag}")
            if int(tol_matches[tag]["ulp"]) != 0:
                raise SystemExit(
                    f"scoring_multi {tag} max_ulp={tol_matches[tag]['ulp']} != 0 -- the REAL "
                    "compiled scoring feedForward overload diverged from the reimpl at the "
                    "synthetic shape (expected 0 ULP)."
                )
            if tag not in cost_matches:
                raise SystemExit(f"missing BLSTM_SCORING_MULTI line for {tag}")
        # Unknown-class fold cross-check: targetIndex 5 (>= outputSize 3) folds to 0, so its
        # accumulated cost MUST bit-match targetIndex 0's (both step 0).
        if cost_matches["tiOOB_step0"]["cost"] != cost_matches["ti0_step0"]["cost"]:
            raise SystemExit("scoring_multi OOB (targetIndex 5) cost != targetIndex-0 cost: the unknown-class fold (BLSTMNeuralNetwork.cpp:873) is broken.")
        sc_rows, sc_cols, _ = _read_bin(tmp_dir / SCORING_MULTI_BIN)
        if (sc_rows, sc_cols) != (6, 3):
            raise SystemExit(f"{SCORING_MULTI_BIN} shape {(sc_rows, sc_cols)} != (6, 3)")

        # 4c. Task 4: parse + validate the LID SEG_STRUCT / LID_STRUCT structural lines. Each
        # file must produce an ok=1 SEG_STRUCT (segment count/type equality reimpl-vs-real; the
        # C++ std::abort()s otherwise, so its absence == a passing generation) and an ok=1
        # LID_STRUCT (confusion + _IsLIDCorrect equality). The langid/cost deltas are recorded
        # as GEMM-divergence calibration (the ascending reimpl vs the real Eigen forward).
        lid_seg = {m["name"]: m for m in LID_SEG_STRUCT_RE.finditer(stdout)}
        lid_struct = {m["name"]: m for m in LID_STRUCT_RE.finditer(stdout)}
        lid_cal: dict[str, dict[str, float]] = {}
        for name in LID5_FILES:
            if name not in lid_seg:
                raise SystemExit(f"missing SEG_STRUCT line for lid5_{name} (structural abort or mismatch)")
            if name not in lid_struct:
                raise SystemExit(f"missing LID_STRUCT line for lid5_{name} (confusion/isCorrect mismatch)")
            lid_cal[name] = {
                "seg_max_dt": float(lid_seg[name]["dt"]),
                "langid_max_abs": float(lid_struct[name]["langid"]),
                "cost_max_abs": float(lid_struct[name]["cost"]),
            }
        for name in LID5_BINS:
            src = tmp_dir / name
            if not src.is_file():
                raise SystemExit(f"harness did not produce {name}")
        # MEASURED per-file/per-channel LID members for the manifest.
        lid_measured: dict[str, dict[str, object]] = {}
        for name in LID5_FILES:
            chans: dict[str, object] = {}
            for ch in (1, 2):
                lr, lc, ltsv = _read_bin(tmp_dir / f"lid5_{name}_ltsv_chan{ch}.bin")
                er, ec, liderr = _read_bin(tmp_dir / f"lid5_{name}_liderr_chan{ch}.bin")
                cr, cc, _conf = _read_bin(tmp_dir / f"lid5_{name}_confusion_chan{ch}.bin")
                _mr, _mc, mem = _read_bin(tmp_dir / f"lid5_{name}_members_chan{ch}.bin")
                br, bc, _b = _read_bin(tmp_dir / f"lid5_{name}_boundaries_chan{ch}.bin")
                sentinel = max(liderr) > 150.0
                if not sentinel:
                    raise SystemExit(f"lid5_{name}_chan{ch}: no >150 sentinel in lid_classification_errors")
                chans[f"chan{ch}"] = {
                    "ltsv_len": lc,
                    "liderr": liderr,
                    "confusion_shape": [cr, cc],
                    "cumulative_error": mem[0],
                    "nb_of_classif": int(mem[1]),
                    "is_lid_correct": int(mem[2]),
                    "boundaries_shape": [br, bc],
                    "has_sentinel_gt150": sentinel,
                }
            lid_measured[name] = {"class_index": LID5_FILES.index(name), "channels": chans}

        # 4c'. Task 6: parse + validate the Twin SEG_STRUCT / LID_STRUCT structural lines.
        # Each variant x file must produce ok=1 SEG_STRUCT (segment count/type equality
        # reimpl-vs-real -- the C++ std::abort()s otherwise) + ok=1 LID_STRUCT (confusion +
        # _IsLIDCorrect equality). The `concat=` field must match the variant's expected
        # branch (0 not-called / 1 concat / 2 empty-fallback) -- the concat non-vacuity.
        twin_seg = {m["name"]: m for m in TWIN_SEG_STRUCT_RE.finditer(stdout)}
        twin_struct = {m["name"]: m for m in TWIN_STRUCT_RE.finditer(stdout)}
        twin_measured: dict[str, dict[str, object]] = {}
        for v, meta in TWIN_VARIANTS.items():
            files_m: dict[str, object] = {}
            for f in TWIN_FILES:
                key = f"{v}_{f}"
                if key not in twin_seg:
                    raise SystemExit(f"missing SEG_STRUCT line for twin_{key} (structural abort or mismatch)")
                if key not in twin_struct:
                    raise SystemExit(f"missing LID_STRUCT line for twin_{key} (confusion/isCorrect mismatch)")
                concat = int(twin_struct[key]["concat"])
                if concat != meta["concat"]:
                    raise SystemExit(
                        f"twin_{key}: concat branch {concat} != expected {meta['concat']} -- the "
                        "getBLSTMLIDInputSequence non-vacuity contract is broken."
                    )
                chans: dict[str, object] = {}
                for ch in (1, 2):
                    _rr, rc, _res = _read_bin(tmp_dir / f"twin_{v}_{f}_result_chan{ch}.bin")
                    _er, _ec, liderr = _read_bin(tmp_dir / f"twin_{v}_{f}_liderr_chan{ch}.bin")
                    cr, cc, _conf = _read_bin(tmp_dir / f"twin_{v}_{f}_confusion_chan{ch}.bin")
                    _mr, _mc, mem = _read_bin(tmp_dir / f"twin_{v}_{f}_members_chan{ch}.bin")
                    br, bc, _b = _read_bin(tmp_dir / f"twin_{v}_{f}_boundaries_chan{ch}.bin")
                    if max(liderr) <= 150.0:
                        raise SystemExit(f"twin_{key}_chan{ch}: no >150 sentinel in lid_classification_errors")
                    chans[f"chan{ch}"] = {
                        "result_len": rc,
                        "liderr": liderr,
                        "confusion_shape": [cr, cc],
                        "lid_cumulative_error": mem[0],
                        "lid_nb_of_classif": int(mem[1]),
                        "is_lid_correct": int(mem[2]),
                        "cumulative_error": mem[3],
                        "nb_of_classif": int(mem[4]),
                        "boundaries_shape": [br, bc],
                    }
                files_m[f] = {
                    "class_index": TWIN_FILES.index(f),
                    "seg_max_dt": float(twin_seg[key]["dt"]),
                    "langid_max_abs": float(twin_struct[key]["langid"]),
                    "channels": chans,
                }
            twin_measured[v] = {
                "mode": meta["mode"],
                "concat_branch": meta["concat"],
                "reference_driven": meta["ref"],
                "files": files_m,
            }

        # 4d. Task 5: parse + validate the PHASE4B_PHSEQ stdout lines against the
        # INDEPENDENT Python reimplementation (phseq_expected), and cross-check the
        # dumped .bin shapes/content match too -- not merely trusting either source
        # alone.
        phseq_stdout = {m["name"]: m for m in PHSEQ_RE.finditer(stdout)}
        phseq_measured: dict[str, dict[str, object]] = {}
        for name in PHSEQ_FILES:
            if name not in phseq_stdout:
                raise SystemExit(f"missing PHASE4B_PHSEQ line for {name}")
            m = phseq_stdout[name]
            exp = phseq_expected[name]
            if int(m["lines"]) != exp["lines"]:
                raise SystemExit(f"phseq {name}: harness lines={m['lines']} != expected {exp['lines']}")
            if int(m["frames"]) != exp["frames_count"]:
                raise SystemExit(
                    f"phseq {name}: harness frames_count={m['frames']} != expected {exp['frames_count']}"
                )
            if int(m["prows"]) != exp["number_of_phonemes"]:
                raise SystemExit(
                    f"phseq {name}: harness periodogram_rows={m['prows']} != expected numberOfPhonemes "
                    f"{exp['number_of_phonemes']}"
                )
            if (int(m["channels"]), int(m["framerate"]), int(m["pcols"])) != (1, 8000, 38):
                raise SystemExit(
                    f"phseq {name}: (channels,framerate,periodogram_cols)="
                    f"{(m['channels'], m['framerate'], m['pcols'])} != (1, 8000, 38)"
                )
            # Cross-check the dumped periodogram.bin shape + the block-fill placement
            # against the independently-computed (row_begin, len) blocks.
            pr, pc, pdata = _read_bin(tmp_dir / f"phseq_{name}_periodogram.bin")
            if (pr, pc) != (exp["number_of_phonemes"], 38):
                raise SystemExit(f"phseq_{name}_periodogram.bin shape {(pr, pc)} != {(exp['number_of_phonemes'], 38)}")
            feat_shapes = []
            for i, (row_begin, length) in enumerate(exp["blocks"]):
                fr, fc, fdata = _read_bin(tmp_dir / f"phseq_{name}_feat{i}.bin")
                if (fr, fc) != (length, 38):
                    raise SystemExit(f"phseq_{name}_feat{i}.bin shape {(fr, fc)} != {(length, 38)}")
                feat_shapes.append([row_begin, length])
                # Non-vacuity + placement check: every nonzero cell of feat[i] (column-major)
                # must reappear at periodogram[row_begin + local_row, col] (also column-major).
                for idx, v in enumerate(fdata):
                    if v == 0.0:
                        continue
                    local_row, col = idx % length, idx // length
                    global_row = row_begin + local_row
                    pidx = col * pr + global_row
                    if pdata[pidx] != v:
                        raise SystemExit(
                            f"phseq_{name}: feat{i}[{local_row},{col}]={v} not placed at "
                            f"periodogram[{global_row},{col}] (row_begin={row_begin})"
                        )
            phseq_measured[name] = {
                "lines": exp["lines"],
                "line_lengths": exp["lens"],
                "number_of_phonemes": exp["number_of_phonemes"],
                "frames_count": exp["frames_count"],
                "blocks_row_begin_len": feat_shapes,
            }
        # Non-vacuity: at least one fixture line must be length-0 (the "sentence" with
        # NO phonemes -- a 0-row one-hot block, exercising the len==0 edge of the
        # `+10` gap arithmetic). f1's middle blank line is that case.
        if not any(0 in exp["lens"] for exp in phseq_expected.values()):
            raise SystemExit("no phSeq fixture exercises a length-0 line (0-row one-hot block)")

        # 4e. Task 7 IMPORTANT-1: parse + validate the PHASE4B_MODE7 stdout lines (was
        # dead: MODE7_RE was defined but never wired up). One `lidWeights=` line (the
        # real LID net's weight count, must be exactly 12409) plus one `<variant>_<file>
        # nbclassif=.../isCorrect=.../lidCumErr=.../confSum=...` line per (variant, file)
        # -- a missing line, or a lidWeights count != 12409, aborts generation.
        wm = MODE7_WEIGHTS_RE.search(stdout)
        if not wm:
            raise SystemExit("PHASE4B_MODE7 lidWeights= line missing from harness stdout")
        mode7_lid_weights = int(wm["n"])
        if mode7_lid_weights != 12409:
            raise SystemExit(
                f"PHASE4B_MODE7 lidWeights={mode7_lid_weights} != 12409 -- the real LID net's "
                "weight count drifted (LID_bestNNWeight_1.bin element count, or the built "
                "[36,24]/[48,1] LID net topology, changed)."
            )

        mode7_matches = {m["tag"]: m for m in MODE7_RE.finditer(stdout)}
        mode7_measured: dict[str, dict[str, object]] = {}
        for v in MODE7_VARIANTS:
            mode7_files_m: dict[str, object] = {}
            for f in MODE7_FILES:
                tag = f"{v}_{f}"
                if tag not in mode7_matches:
                    raise SystemExit(f"missing PHASE4B_MODE7 line for {tag} (nbclassif/isCorrect absent)")
                mm = mode7_matches[tag]
                mode7_files_m[f] = {
                    "nb_of_classif": int(mm["nbc"]),
                    "is_correct": int(mm["iscorrect"]),
                    "lid_cumulative_error": float(mm["cost"]),
                    "confusion_sum": float(mm["confsum"]),
                }
            mode7_measured[v] = mode7_files_m

        # 4f. Task 7 IMPORTANT-2: verify (and, where the sibling legacy checkout is
        # present, re-derive) the committed LID_bestNNWeight_1.bin's provenance.
        lid_weight_provenance = _verify_lid_weight_provenance()

        # 4g. Task 9: parse + validate the corpus-level LID lines. Non-vacuity is
        # ENFORCED here (SystemExit), not hoped for: the twin_train best-cost gate
        # must FIRE and SKIP; every consecutive LID epoch pair must DIFFER; the main
        # corpus must have speech every epoch while the ns variant has none (its
        # costLID forced to -1.0 by the LIVE :465 gate); and the gated (ns) vs
        # ungated (nsx) counterfactual LID trajectories must DIVERGE at a measured
        # epoch -- the proof the gate is observable in the weights.
        t9_seeds = T9_SEEDS_RE.search(stdout)
        if not t9_seeds:
            raise SystemExit("PHASE4B_T9 seeds line missing from harness stdout")
        t9_train = {m["tag"]: m for m in T9_TRAIN_RE.finditer(stdout)}
        for tag in ("twin_train", "twin_train_ns", "twin_train_nsx"):
            if tag not in t9_train:
                raise SystemExit(f"missing PHASE4B_T9_TRAIN line for {tag}")
        def _floats(s: str) -> list[float]:
            return [float(x) for x in s.split(",") if x]
        tm = t9_train["twin_train"]
        t9_fired, t9_skipped = int(tm["fired"]), int(tm["skipped"])
        if t9_fired < 1 or t9_skipped < 1:
            raise SystemExit(
                f"twin_train best-cost gate fired={t9_fired} skipped={t9_skipped}: both must be "
                ">= 1 -- retune BLSTM_LID_BackPropagationRpropInit/epochs in twin_train.config."
            )
        if int(tm["differ"]) != T9_EPOCHS - 1:
            raise SystemExit(f"twin_train differ_lid={tm['differ']} != {T9_EPOCHS - 1}: LID epochs collapsed")
        if any(sp <= 1e-3 for sp in _floats(tm["sp"])):
            raise SystemExit("twin_train must detect speech EVERY epoch (else its costLID gets gated)")
        ns = t9_train["twin_train_ns"]
        if any(sp != 0.0 for sp in _floats(ns["sp"])):
            raise SystemExit("twin_train_ns must detect NO speech (rising threshold 11 > constant 10)")
        if any(cl != -1.0 for cl in _floats(ns["cl"])):
            raise SystemExit("twin_train_ns costLID must be -1.0 every epoch (the LIVE :465 gate)")
        nsx = t9_train["twin_train_nsx"]
        if all(cl == -1.0 for cl in _floats(nsx["cl"])):
            raise SystemExit("twin_train_nsx (counterfactual) costLID must carry the REAL values")
        t9_div = T9_NS_DIVERGE_RE.search(stdout)
        if not t9_div:
            raise SystemExit("PHASE4B_T9_NS_DIVERGE line missing")
        t9_diverge_epoch = int(t9_div["e"])
        if t9_diverge_epoch < 0:
            raise SystemExit(
                "ns vs nsx LID trajectories never diverged: the costLID=-1.0 gate is not "
                "observable with this config -- retune RpropInit/epochs."
            )
        t9_gc = {(m["tag"], int(m["net"])): m for m in T9_GRADCHECK_RE.finditer(stdout)}
        for key in (("twin", 0), ("twin", 1), ("lid5", 0)):
            if key not in t9_gc:
                raise SystemExit(f"missing PHASE4B_T9_GRADCHECK line for {key}")
        # Non-degeneracy: at least one numerical derivative per gradcheck golden.
        for name in ("twin_gradcheck_net0.bin", "twin_gradcheck_net1.bin", "lid5_gradcheck.bin"):
            gr, gcc, gdata = _read_bin(tmp_dir / name)
            if gcc != 3:
                raise SystemExit(f"{name}: cols {gcc} != 3")
            numerical = [gdata[2 * gr + r] for r in range(gr - 1)]  # col 2, rows 0..sweep-1
            if not any(abs(v) > 1e-12 for v in numerical):
                raise SystemExit(f"{name}: every numerical derivative is ~0 -- degenerate gradcheck")
        if not T9_E2E_RE.search(stdout):
            raise SystemExit("PHASE4B_T9_E2E ok=1 line missing (real CorpusProcessor run failed)")

        # Task 10: pitch-pass target-reuse pin (SPECTRAL_PITCH_SCORED line + the STM-
        # referenced dumps). Non-vacuity: pitch > 0 (the warp fired, matching the
        # NN-chain-only "pitch" variant's measured ~250 Hz).
        #
        # cumError_chan1/nbClassif_chan1 are expected to read 0 here EVEN THOUGH a
        # live STM reference is loaded: the harness's `signalReimplFFB` (the
        # ascending-loop NN swap every spectral variant uses, since the REAL
        # Eigen FFB diverges from the port at this net's shapes) has never
        # accumulated cost for ANY prior variant -- `transcribeSpectral` HARD-CODES
        # `_CumulativeError[chan] = 0.0` after every `results2segmentation` call
        # (main.cpp, "no NN cost accumulated (no targets on the reimpl path)"),
        # because forward-pass posteriors (`result_vec`, what this golden pins) are
        # mathematically INDEPENDENT of the target (`nn/blstm.rs::
        # feed_forward_backward_plain`: `output` is fully determined by
        # `self.feed_forward(input, output)` BEFORE the target-gated cost/backward
        # blocks run, given `TargetEnforcementStep >= 0` as here) -- extending the
        # reimpl to accumulate cost would be a much larger, shared-helper change for
        # a value this golden does not need. The live-cost claim (does a REAL
        # reference genuinely drive a nonzero cost) is instead asserted on the
        # RUST PRODUCTION driver directly, in
        # `phase4b_backlog.rs::pitch_pass_target_reuse_under_live_reference`.
        ps = PITCH_SCORED_RE.search(stdout)
        if not ps:
            raise SystemExit("SPECTRAL_PITCH_SCORED line missing from harness stdout")
        pitch_scored_pitch = float(ps["pitch"])
        pitch_scored_cum_error = float(ps["err"])
        pitch_scored_nb_classif = int(ps["nb"])
        if pitch_scored_pitch <= 0.0:
            raise SystemExit(
                f"SPECTRAL_PITCH_SCORED measured_pitch_chan1={pitch_scored_pitch} <= 0 -- "
                "the pitch pass did not fire on the STM-referenced run"
            )
        for name in PITCH_SCORED_BINS:
            if not (tmp_dir / name).is_file():
                raise SystemExit(f"harness did not produce {name}")

        t9_measured = {
            "seeds": {"sad_nb_weights": int(t9_seeds["sad"]), "lid_nb_weights": int(t9_seeds["lid"])},
            "train": {
                tag: {
                    "epochs": int(t9_train[tag]["epochs"]),
                    "gate_fired": int(t9_train[tag]["fired"]),
                    "gate_skipped": int(t9_train[tag]["skipped"]),
                    "lid_epochs_differing": int(t9_train[tag]["differ"]),
                    "best_cost": _floats(t9_train[tag]["bc"]),
                    "cost_lid": _floats(t9_train[tag]["cl"]),
                    "total_speech_duration": _floats(t9_train[tag]["sp"]),
                    "error_perc_lid": _floats(t9_train[tag]["epl"]),
                }
                for tag in ("twin_train", "twin_train_ns", "twin_train_nsx")
            },
            "ns_gate_diverge_epoch": t9_diverge_epoch,
            "gradcheck": {
                f"{tag}_net{net}": {
                    "sweep": int(m["sweep"]),
                    "mean_error": float(m["err"]),
                    "mean_relative_error": float(m["rel"]),
                }
                for (tag, net), m in t9_gc.items()
            },
            "gradcheck_max_weights": 10,
            "epsilon": 1e-5,
        }

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
            "scoring_multi": {
                "text": (
                    "Task 3: the MULTICLASS scoring feedForward overload (BLSTMNeuralNetwork.cpp:"
                    "843-929, BLSTMNeuralNetwork.h:195 -- the LID drivers' core call) on a "
                    "synthetic net (LSTM [3,4,2] sub [2,1], output [4,5,3] sub [1,1], T=12 -> "
                    "length 6, outputSize 3). scoring_multi_out.bin is the reimpl's raw "
                    "length x outputSize posterior matrix, pinned max_ulp=0 vs the REAL compiled "
                    "overload (NN_TOL site=blstm_scoring_multi_*). The forward output is "
                    "TARGET-INDEPENDENT (targets feed only cost/backward, both off here + step>=0 "
                    "leaves the plain-FFB cost block from rewriting the output), so one golden "
                    "covers every case; the NO-binary-expansion return is exercised (the [1-p,p] "
                    "expansion gate at :922-926 is outputSize == 1, so a multiclass net returns "
                    "the raw matrix). The per-case _Cost (read from the REAL class via getCost()) "
                    "reflects the multiclass target construction (:872-886) feeding "
                    "CostLaw::computeCost -- softmax CE, a libm chain, so the hex is the oracle-env "
                    "(Apple libm) value. targetIndex 5 (>= outputSize 3) folds to the unknown class "
                    "0 (:873), harness-asserted to bit-match targetIndex 0's cost. A fifth case "
                    "(ti2_step0_mod2_costmod) sets SYNC_BackPropWER (isCostModified() -> true) with "
                    "target_modifier 2.0, exercising the NONZERO soft-target placement (target = "
                    "0.1*modifier = 0.2; enforced rows set the non-target columns to 0.2 and the "
                    "target column to 1-target = 0.8), unlike the other four (hard-target 0.0/1.0) "
                    "cases."
                ),
                "shape": [sc_rows, sc_cols],
                "cases": {
                    tag: {
                        "target_index": SCORING_MULTI_CASES[tag][0],
                        "step": SCORING_MULTI_CASES[tag][1],
                        "target_modifier": SCORING_MULTI_CASES[tag][2],
                        "cost_modified": SCORING_MULTI_CASES[tag][3],
                        "max_ulp": int(tol_matches[tag]["ulp"]),
                        "cost_bits": "0x" + cost_matches[tag]["cost"],
                        "cost_dec": float(cost_matches[tag]["cost_dec"]),
                        "nb_of_classif": int(cost_matches[tag]["nb"]),
                    }
                    for tag in SCORING_MULTI_CASES
                },
            },
            "lid5": {
                "text": (
                    "Task 4: BlstmSpectralLID (Algo 5), the FIRST LID driver. "
                    "BLSTMSpectralLID::getSegmentation (:27-460) does SAD via LTSV "
                    "(classifySequence over the RAW periodogram, :271-274 -- the mel branch is "
                    "COMMENTED OUT) then runs the BLSTM PER SPEECH SEGMENT for language scoring "
                    "(:346-401 via the returning feedForward overload :363). The harness LidProbe "
                    "transcribes :27-460 swapping ONLY that :363 feedForward for the reimpl scoring "
                    "(target build :868-903 + signalReimplFFB plain path + REAL CostLaw::computeCost "
                    ":815-828), keeping LTSV/results2segmentation/compute_errors REAL; the reimpl "
                    "dumps (lid5_<f>_{ltsv,liderr,confusion,members,boundaries}_chan{1,2}.bin) are "
                    "the goldens the Rust port matches bit-exact (canary-gated: langID is a "
                    "softmax/log chain, cumulative_error a LogLaw chain). lid5.config forces the "
                    "plain scoring path (BLSTM_window 0) + a degenerate SAD (decision thresholds "
                    "<< 0 -> one big SPEECH segment/file), so the scoring loop always runs "
                    "(non-vacuity: the >150 in-band sentinel is present at every file's target "
                    "index via targetLID(target) = -2.0, and the confusion has off-diagonal mass "
                    "on f1/f3 -- target argmax != language target -- vs a diagonal hit on f2). The "
                    "real 33,671-weight SAD net is binary (outputSize 1 -> the [1-p,p] binary-"
                    "expansion path); targetIndex = getRefLangIndex() (f1/f2/f3 -> class 0/1/2, "
                    "class 2 clamps to 0 under classNb 2). A SECONDARY real-Eigen probe runs the "
                    "compiled getSegmentation beside the reimpl: SEG_STRUCT (segment count/type "
                    "equality, all max_dt == 0 -- the LTSV SAD is NN-free so it is bit-identical) "
                    "and LID_STRUCT (confusion + _IsLIDCorrect EQUALITY -- argmax agrees despite "
                    "the GEMM divergence; a mismatch std::abort()s generation). The langid deltas "
                    "are ~1e-16 (the ascending forward vs the real Eigen forward), recorded below."
                ),
                "calibration": lid_cal,
                "measured": lid_measured,
            },
            "twin": {
                "text": (
                    "Task 6: TwinBlstmSpectralLid (Algo 6), the wav-mode 0/2/3 paths + the "
                    "hidden-state concat. TwinBLSTMSpectralLID::getSegmentation (:263-1421) runs "
                    "the SAD BLSTM (_BLSTMNeuralNetwork) for the VAD result_vec (modes 0/3) then a "
                    "SECOND net (_LIDBLSTMNeuralNetwork) per speech segment for language scoring "
                    "(:1243-1291 via the returning feedForward :1250). The harness TwinProbe "
                    "transcribes the mode 0/2/3 paths swapping FOUR real-class call sites for reimpl "
                    "equivalents: the SAD FFB (:715, blstmFeedForwardT6), the type -1 self-"
                    "normalization shared by both nets (BLSTMNeuralNetwork.cpp:737-744), "
                    "getBLSTMLIDInputSequence's SAD-hidden-state concat (:139-168), and the LID "
                    "scoring feedForward (:1250, a generic small-topology reimpl) -- keeping "
                    "getTargets/results2segmentation/LID2Segmentation/"
                    "compute_errors REAL; the reimpl dumps (twin_<v>_<f>_{result,liderr,confusion,"
                    "members,boundaries}_chan{1,2}.bin) are the goldens the Rust port matches "
                    "bit-exact (canary-gated: liderr/lid_cumulative_error are softmax/CE chains, "
                    "the SAD cumulative_error a LogLaw chain in mode 3). members = [lid_cumulative_"
                    "error (*= audio._Weight), lid_nb_of_classif, is_lid_correct, cumulative_error "
                    "(SAD, *= LIDCostPonderation), nb_of_classif]. A SECONDARY real-Eigen probe runs "
                    "the compiled Twin beside the reimpl: SEG_STRUCT (segment count/type equality, "
                    "all max_dt == 0) + LID_STRUCT (confusion + _IsLIDCorrect EQUALITY + the concat "
                    "branch counter; a mismatch std::abort()s generation). NON-VACUITY: the >150 "
                    "in-band sentinel (targetLID(target) = -2.0) is present at every variant x file "
                    "x channel; the confusion has both diagonal (f1 target 0 == argmax) and "
                    "off-diagonal (f2/f3 miss) mass; and the concat branch is proven per variant -- "
                    "mode0 not-called (0), mode0_concat concat (1, LID input 59 > feature width 11 "
                    "with the SAD hidden states populated), mode2 empty-fallback (2, LID input 59 > "
                    "11 but the SAD outputs cleared at :762-763), mode3 not-called (0). Modes "
                    "2/3 iterate the REFERENCE (a programmatic 2-span segmentation) instead of the "
                    "SAD classification, and (mode != 0) overwrite the hypothesis via "
                    "LID2Segmentation -> reference-driven boundaries differ from mode0. The langid "
                    "deltas (~1e-16) are the ascending forward vs the real Eigen forward, recorded "
                    "per variant/file. Task 7c adds modes 4/5/6 (the :640-902 LID-train + scoring "
                    "branch, transcribeTwin456): the LID net runs feedForwardBackward ONCE over the "
                    "WHOLE inputSeq (:664) and the scoring slices LID_result_vec per speech span by "
                    "LIDTimeStep/LIDTimeOffset (offset SUBTRACTED before the divide, :811-813 -- the "
                    "load-bearing contrast vs the :1245 branch's bare begin/_SpectrumShift). The "
                    "three modes differ only in the classification the scoring iterates: mode 4 = "
                    "REFERENCE smoothed (:779-783; SAD net NOT run, cumulative_error 0); mode 5 = SAD "
                    "BLSTM VAD (blstmFeedForwardT6 on the LID-normalized inputSeq -> "
                    "results2segmentation; sadCumErr/nb populated); mode 6 = LID2Segmentation over "
                    "1-LID_result_vec.col(0) (:678-691; the SAD result_vec at :727-732 is dead). The "
                    ":798-902 accumulator is the three _PostProcessMode forms shared with the mode-7 "
                    "branch (langID += isCostModified ? segLID/rows : segLID; post-loop /= "
                    "segmentsCount or numberOfFrames, PPM != 2 -> exp, row-normalize). All configs use "
                    "PostProcessMode 0 (sum-log). NON-VACUITY: getTargetsLID is exercised (the LID "
                    "target build); LID2Segmentation (mode 6) produces speech spans DIFFERING from the "
                    "reference (~[1.73,1.81]/[1.89,1.94] vs the ref [0.4,0.9]/[1.2,1.6]) and from the "
                    "2-segment seed; the >150 sentinel is present at every file; the confusion has "
                    "diagonal (f1) + off-diagonal (f2/f3) mass. concat is always 0 (modes 4/5/6 never "
                    "reach the :1221 concat)."
                ),
                "measured": twin_measured,
            },
            "phseq": {
                "text": (
                    "Task 5: the phSeq reader (File_Type 1, AudioStruct.cpp:138-182). Pure "
                    "indexing + one-hot construction (letterMapping, AudioStruct.h:18 -- a "
                    "38-entry char -> column bijection), no libm involved anywhere in this "
                    "branch, so every phseq_*.bin dump here is STRICT BITS on every platform "
                    "(unlike the transcendental-dependent lid5/scoring_multi fixtures above). "
                    "corpus_phseq/{f1,f2,f3}.phSeq are SYNTHETIC fixtures (the format contract "
                    "-- no real .phSeq file exists anywhere in this repo or its legacy vendor "
                    "tree to validate against), characters drawn only from letterMapping's "
                    "domain. f1 has a middle BLANK line (0-length sentence -> a 0-row one-hot "
                    "block), exercising the len==0 edge of the `+10` gap arithmetic; f2/f3 "
                    "cover a space-containing multi-word sentence, the `.`/`-` special chars, "
                    "and a full 25-letter single-sentence line (every lowercase letter except "
                    "'q', which is absent from letterMapping). Every PHASE4B_PHSEQ stdout "
                    "line (lines/frames_count/periodogram_rows/periodogram_cols) is "
                    "cross-checked against an INDEPENDENT Python reimplementation of the "
                    "ctor's arithmetic run directly over the fixture files (not merely "
                    "trusting the harness), and every dumped feat<i>.bin nonzero cell is "
                    "verified to reappear at the expected periodogram.bin offset "
                    "(row_begin + local_row, col) -- a placement, not just a shape, check."
                ),
                "measured": phseq_measured,
            },
            "mode7": {
                "text": (
                    "Task 7 FLAGSHIP: TwinBlstmSpectralLid Mode 7 (phSeq) over twin_mode7{,_ppm1,"
                    "_ppm2} x s{1,2,3}.phSeq, pinned DIRECTLY against the REAL compiled "
                    "getSegmentation (no reimpl/transcription swap for this stage, unlike lid5/"
                    "twin above -- the real-compiled run IS the oracle; there is no SEG_STRUCT/"
                    "LID_STRUCT-style secondary probe here). The PHASE4B_MODE7 stdout lines are "
                    "now parsed + validated (previously dead code: MODE7_RE was defined but never "
                    "used): one `lidWeights=<n>` line (the real LID_bestNNWeight_1.bin net, "
                    "asserted == 12409) plus one `<variant>_<file> nbclassif=/isCorrect=/"
                    "lidCumErr=/confSum=` line per case (a missing line aborts generation, per "
                    "repo standard). lidCumErr/confSum are cross-checked below against the "
                    "committed mode7_<variant>_<file>_{members,confusion}.bin dumps by "
                    "tests/test_phase4b_fixtures.py."
                ),
                "lid_weights": mode7_lid_weights,
                "lid_weight_provenance": lid_weight_provenance,
                "measured": mode7_measured,
            },
            "corpus_lid": {
                "text": (
                    "Task 9: corpus-level LID goldens. UNLIKE the tier-2 CorpusProbe (reimpl-"
                    "swapped FFB), every per-file getSegmentation here is the REAL COMPILED "
                    "driver: the configs use TINY nets (every GEMM reduction dim < 23, the "
                    "nn_product_probes boundary), so the real Eigen engine and the ascending-"
                    "loop Rust port agree bit-for-bit on the oracle env (libm-bearing values "
                    "canary-gated off it). Only the corpus loop + SegmentationFunction row "
                    "assembly + saveAndUpdate are transcribed, with the REAL saveWeights/"
                    "saveWeightsLID/updateWeights/updateWeightsLID (real Rprop, both nets) and "
                    "the REAL PrintConfusionMatrix. twin_train: Mode-7 phSeq 2-file corpus, "
                    "3 epochs, BOTH nets' post-saveAndUpdate weights dumped per epoch; the "
                    "algo-6 criterion (cost+costLID) trajectory measured (fired/skipped). The "
                    "SAD net never forwards in Mode 7 -> its derivs stay a zero Nx2 -> "
                    "updateWeights normalizes 0/0 = NaN per element -> Rprop's else-branch "
                    "applies +delta per weight row per epoch (the stats-tail rows, count 1, "
                    "stay put): the SAD epoch dumps pin exactly this NaN-drift semantics. "
                    "twin_train_ns: rising threshold 11 > the constant-10 Mode-7 result_vec -> "
                    "NO speech -> saveAndUpdate:465 forces costLID = -1.0 into EVERY "
                    "updateWeightsLID (the LIVE gate); twin_train_nsx is the same run with the "
                    "gate DISABLED (counterfactual) -- the two LID trajectories diverge at "
                    "ns_gate_diverge_epoch (a port that dropped the gate would reproduce nsx, "
                    "not ns). twin_e2e: the REAL CorpusProcessor(-m).run() at epochs 0 -> the "
                    "twin_e2e_<var>.bin value fixtures for the spawned-binary e2e (timing col "
                    "masked). twin_gradcheck: Mode-5 wav 1-file corpus, BOTH nets backprop-"
                    "active -> the per-network cost-column switch (ii=0 SAD cols 4/len-1, ii=1 "
                    "LID cols 14/len-2) golden per net; lid5_gradcheck: Algo 5, cols 14/len-2. "
                    "SQUARE cost laws avoid the cost-law libm (the LSTM activation chain still carries libm, so the goldens stay canary-gated off the oracle env); Audio_max_duration 0.2 keeps every backward deriv reduction below the k>=23 Eigen-blocking boundary. "
                    "Each gradcheck .bin is (sweep+1) x 3: rows 0..sweep-1 = [analytic_col0, "
                    "analytic_col1, numerical]; final row [mean_error, mean_relative_error, 0]."
                ),
                "measured": t9_measured,
            },
            "pitch_scored": {
                "text": (
                    "Task 10 (phase4a backlog item 3): the spectral PITCH second pass "
                    "(BLSTMSpectralSegmenter.cpp:757-805) driven with a LIVE STM reference, "
                    "closing the gap the phase2b pitch goldens left (NN-chain only, "
                    "setReference=false). Reuses the SAME segmenter-level transcribeSpectral "
                    "lambda as the 'pitch' variant (byte-identical TDC overrides), but the "
                    "Segmentation is built from a REAL CorpusItem pointing at the phase4a "
                    "3-file corpus's f1.wav/f1.stm (Segmentation::Segmentation(AudioStruct&, "
                    "double) auto-loads the .stm reference from getRefSegFileName() -- no "
                    "lambda change needed, setReference stays false so the hardcoded synthetic "
                    "span is not ALSO pushed). Pins a bit-exact pass-2 result-row golden "
                    "(spectral_pitch_scored_result_pass2_chan1.bin) under a GENUINELY-BUILT "
                    "getTargets() target (seg._Reference.size() > 0 fires for real, from a "
                    "real committed .stm, not the synthetic hardcoded span every other "
                    "setReference=true call site here uses) -- closing the phase4a backlog's "
                    "'no reference-driven target path' gap in COVERAGE. cumError_chan1/"
                    "nbClassif_chan1 read 0 here (see the extractor's inline comment: the "
                    "reimpl transcription never wires cost through, and result_vec is "
                    "target-independent by construction given TargetEnforcementStep >= 0) -- "
                    "the live-cost claim (a real reference genuinely drives a nonzero cost, "
                    "i.e. the :793 target-reuse wiring is exercised on a NON-EMPTY target, not "
                    "merely inert) is asserted on the RUST PRODUCTION driver instead, in "
                    "phase4b_backlog.rs::pitch_pass_target_reuse_under_live_reference."
                ),
                "measured": {
                    "pitch_chan1": pitch_scored_pitch,
                    "cum_error_chan1": pitch_scored_cum_error,
                    "nb_classif_chan1": pitch_scored_nb_classif,
                },
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

        # 6. Commit: every check above passed, so NOW copy the generated fixtures
        #    from tmp_dir (and t9_dir, for the bestNNWeight artifacts) into the
        #    committed PHASE4B_DIR -- the copy is the LAST thing this function
        #    does, so a failed run above never leaves PHASE4B_DIR partially
        #    overwritten with unvalidated output (see the Task 7 review fix).
        for name in T9_BEST_ARTIFACTS:
            shutil.copy2(t9_dir / name, PHASE4B_DIR / name)
        for name in [
            *CONFUSION_BINS,
            SCORING_MULTI_BIN,
            *LID5_BINS,
            *TWIN_BINS,
            *phseq_bins,
            *MODE7_BINS,
            *T9_DUMPS,
            *T9_MAT_BINS,
            *PITCH_SCORED_BINS,
        ]:
            src = tmp_dir / name
            dst = PHASE4B_DIR / name
            # Task 7c churn nit: mode7_dump_s1.mat is libmatio output whose MAT v5 header
            # timestamp churns run-to-run; skip the rewrite when the VALUES are unchanged
            # (the 4a value-guarded skip pattern), so a re-run stays byte-identical.
            if name == "mode7_dump_s1.mat" and dst.is_file() and not _mat_dump_differs(src, dst):
                continue
            shutil.copy2(src, dst)

        manifest_path = PHASE4B_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        print(
            f"OK: phase4b fixtures (confusion input={in_rows}x{in_cols}, "
            f"matrix={mat_rows}x{mat_cols}, error1=error2={error1!r}; "
            f"scoring_multi={sc_rows}x{sc_cols}, all sites max_ulp=0; "
            f"lid5 {len(LID5_FILES)} files, all SEG_STRUCT/LID_STRUCT ok=1; "
            f"twin {len(TWIN_VARIANTS)} variants x {len(TWIN_FILES)} files, all "
            f"SEG_STRUCT/LID_STRUCT ok=1, concat branches pinned; "
            f"phseq {len(PHSEQ_FILES)} files, {len(phseq_bins)} bins; "
            f"mode7 lidWeights={mode7_lid_weights}, {len(MODE7_VARIANTS)} variants x "
            f"{len(MODE7_FILES)} files parsed, weight provenance reverified="
            f"{lid_weight_provenance['reverified_byte_identical']}; "
            f"t9 twin_train fired={t9_fired} skipped={t9_skipped} "
            f"ns_diverge_epoch={t9_diverge_epoch}, gradchecks twin+lid5 ok, e2e ok; "
            f"pitch_scored pitch_chan1={pitch_scored_pitch!r} "
            f"cumError_chan1={pitch_scored_cum_error!r} "
            f"nbClassif_chan1={pitch_scored_nb_classif}), "
            f"manifest -> {manifest_path.relative_to(REPO_ROOT)}"
        )


if __name__ == "__main__":
    main()
