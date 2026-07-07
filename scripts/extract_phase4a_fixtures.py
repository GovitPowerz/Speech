"""Build and run the oracle harness tier-1 stage, then write the corpus-driver fixtures.

Phase 4a Task 8 links the REAL compiled CorpusProcessor stack (CorpusProcessor /
BagOfProcessors / Corpus + VRCTSpart / BLSTMSpectralLID / TwinBLSTMSpectralLID +
FileDispatcher / ConvolutionalNeuralNetwork / ConvolutionalLayer link-only) into the
oracle harness. The phase4a_tier1 stage chdir's into a FIXED corpus workdir and runs
CorpusProcessor(configs, mode).run() end to end for three modes:

  - solo (-s, TDC/Algo 1, epochs -> 0)          -> solo_tdc.mat
  - train (-m, LTSV powermel/Algo 2, epochs 2)  -> train_ltsv.mat
  - multiconfig (-m, both configs, epochs 0)    -> multiconfig.mat

This extractor:

  1. Rebuilds the harness (now with the corpus-driver TUs linked).
  2. Snapshots the committed prior-phase fixture dirs BEFORE the run (regression guard).
  3. Seeds a FIXED corpus path (/tmp/speech_phase4a_corpus) with the committed corpora
     + configs, runs the harness (earlier stages into a throwaway dir, the tier-1 stage
     into the fixed path).
  4. Converts every harness-written .mat into per-variable .bin (io::binary format:
     i64 LE rows, i64 LE cols, f64 LE column-major) via scipy.io.loadmat, ZEROING the
     wall-clock timing column of MultiConfigResults (col 6 = the 4th data column after
     [file, conf, chan]); records masked_cols: [6] in the manifest.
  5. Copies the multi-channel VRCTS xml (the carried-over multi-channel byte oracle)
     and the RAW .mat outputs into the committed phase4a dir (the raw .mat is committed
     so the pytest guard can re-verify the scipy conversion).
  6. REGRESSION GUARD: hashes every prior-phase fixture dir before/after; any drift ->
     SystemExit.

Run TWICE; the .bin dumps + manifest must be byte-identical (the timing column, the
only wall-clock source, is masked to 0.0, so the conversion is deterministic).

Usage: uv run python scripts/extract_phase4a_fixtures.py
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
from typing import cast

import numpy as np
import scipy.io

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "tools" / "oracle_harness"
REF_DIR = REPO_ROOT / "tests" / "reference_data"
PHASE0_DIR = REF_DIR / "phase0"
PHASE1_DIR = REF_DIR / "phase1"
PHASE2B_DIR = REF_DIR / "phase2b"
PHASE4A_DIR = REF_DIR / "phase4a"

NN_CONFIG = PHASE0_DIR / "1_worker_1.config"
NN_WEIGHTS = PHASE0_DIR / "NNweights_config1.bin"
TDC_CONFIG = PHASE2B_DIR / "tdc.config"
LTSV_CONFIG = PHASE2B_DIR / "ltsv.config"
LTSV_DCT_CONFIG = PHASE2B_DIR / "ltsv_dct.config"
LTSV_TINY_CONFIG = PHASE2B_DIR / "ltsv_tiny.config"
LTSV_POWERMEL_CONFIG = PHASE2B_DIR / "ltsv_powermel.config"
SIGNAL_CONFIG = PHASE2B_DIR / "signal.config"

# Committed corpora + configs the tier-1 stage runs on.
TIER1_TDC_CONFIG = PHASE4A_DIR / "tier1_tdc.config"
TIER1_LTSV_CONFIG = PHASE4A_DIR / "tier1_ltsv_powermel.config"
CORPUS_DIR = PHASE4A_DIR / "corpus"
FILESLISTING = PHASE4A_DIR / "fileslisting.csv"
MAPPING = PHASE4A_DIR / "language2classmapping.csv"

# Task 9 tier-2 configs + listings (CorpusProbe train + gradCheck).
TIER2_SPECTRAL_CONFIG = PHASE4A_DIR / "tier2_spectral.config"
TIER2_GRADCHECK_CONFIG = PHASE4A_DIR / "tier2_gradcheck.config"
TIER2_FILESLISTING = PHASE4A_DIR / "tier2_fileslisting.csv"
TIER2_GC_FILESLISTING = PHASE4A_DIR / "tier2_gc_fileslisting.csv"

# Task 9 tier-2 artifacts. Epoch count = Neural_Networks_BackPropagation_Epochs(3)+2.
TIER2_EPOCHS = 5
# The gradCheck sweep cap (DEVIATION from the legacy full sweep; manifest key).
TIER2_GRADCHECK_MAX_WEIGHTS = 10
# bestNNWeight artifact basename (bestNNWeight_<pos+1>_<multiConfigResultsOutputFile>).
TIER2_BEST_BASE = "bestNNWeight_1_tier2_spectral.mat"
# The saveWeights .mat variables (BLSTMNeuralNetwork::saveWeights; the weights/derivs
# go to SEPARATE io::binary files, the .mat carries only the input statistics).
TIER2_BEST_MAT_VARS = ["nbOfInputs", "meanInputs", "stdInputs"]
# VRCTS from the tier-2 train (final epoch's bytes; committed with a tier2_ prefix so
# the names do not collide with the tier-1 vrcts_solo dumps).
TIER2_VRCTS = ["f1_chan_1.xml", "f1_chan_2.xml", "f2_chan_1.xml", "f2_chan_2.xml"]

# The FIXED corpus workdir the harness chdir's into (spec 2b path-stability convention:
# VRCTS AudioDoc path=/name= attrs are byte-reproducible only at a fixed path).
FIXED_CORPUS = Path("/tmp/speech_phase4a_corpus")

# Input fixtures the harness reads from its argv[1] output dir (earlier phase 1/2/2b
# stages still run every invocation and expect these seeded).
HARNESS_INPUTS = [
    "excerpt_2ch_8k.wav",
    "variant_mfcc_deltas.config",
    "variant_mfcc_sdc.config",
    "variant_logmel.config",
    "variant_rawband_ltsv.config",
]

# Prior-phase fixture dirs the extractor must NOT perturb (regression guard).
PRIOR_PHASES = ["phase0", "phase0b", "phase0bii", "phase1", "phase2", "phase2b", "phase3"]

# The 5 saveResults variables (CorpusProcessor.cpp:397-401), in write order.
MAT_VARS = [
    "MultiConfigResults",
    "CostMem",
    "BadClassifMem",
    "CostLIDMem",
    "BadClassifLIDMem",
]

# The three tier-1 runs: (mat basename, list of expected MultiConfigResults rows,
# CostMem rows, nb confs). Rows measured below; these are the SHAPE expectations.
RUNS = ["solo_tdc", "train_ltsv", "multiconfig"]

# MultiConfigResults masked column: col 6 (0-based) = data col 3 = time_per_hour
# (assemble_scored_row / assemble_unscored_row col 3; wall-clock -> zeroed).
MASKED_COLS = [6]

# The multi-channel VRCTS dumps to commit (the carried-over multi-channel byte oracle):
# the solo run's per-channel documents for each file. Solo (-s) is unscored so VRCTS is
# always written; the 2-channel excerpt yields _chan_1/_chan_2 per file.
VRCTS_DUMPS = [
    ("vrcts_solo", "f1_chan_1.xml"),
    ("vrcts_solo", "f1_chan_2.xml"),
    ("vrcts_solo", "f2_chan_1.xml"),
    ("vrcts_solo", "f2_chan_2.xml"),
    ("vrcts_solo", "f3_chan_1.xml"),
    ("vrcts_solo", "f3_chan_2.xml"),
]

NB_FILES_RE = re.compile(r"^PHASE4A_TIER1 nb_files=(?P<n>\d+)$", re.MULTILINE)
# \f? -- the legacy iof::fmtr prints end in "\n\f", so a form feed can precede the
# next line's text; tolerate it before each tier-2 marker.
TIER2_TRAIN_RE = re.compile(
    r"^\f?PHASE4A_TIER2_TRAIN epochs=(?P<epochs>\d+) differ=(?P<differ>\d+) "
    r"fired=(?P<fired>\d+) skipped=(?P<skipped>\d+) best_cost=\[(?P<costs>[^\]]+)\]$",
    re.MULTILINE,
)
TIER2_SEG_STRUCT_RE = re.compile(
    r"^\f?SEG_STRUCT site=tier2_train ok=1 max_dt=(?P<dt>[0-9.eE+-]+)$", re.MULTILINE
)
TIER2_GRADCHECK_RE = re.compile(
    r"^\f?PHASE4A_TIER2_GRADCHECK sweep=(?P<sweep>\d+) mean_err=(?P<err>[0-9.eE+-]+) "
    r"mean_rel_err=(?P<rel>[0-9.eE+-]+)$",
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


def _read_bin(path: Path) -> np.ndarray:
    """Read an io::binary .bin (i64 LE rows, i64 LE cols, f64 LE column-major)."""
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    return np.frombuffer(raw[16:], dtype="<f8").reshape((rows, cols), order="F")


def _write_bin(path: Path, matrix: np.ndarray) -> None:
    """Write `matrix` as an io::binary .bin: i64 LE rows, i64 LE cols, f64 LE
    column-major -- the exact format the Rust `load_bin_phase4a` reads."""
    mat = np.ascontiguousarray(matrix, dtype="<f8")
    rows, cols = mat.shape
    with path.open("wb") as f:
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        # column-major: iterate columns, write each column's rows contiguously.
        f.write(mat.flatten(order="F").tobytes())


def _mask_timing(name: str, matrix: np.ndarray) -> np.ndarray:
    """Zero the wall-clock timing column of MultiConfigResults (col 6); other
    variables pass through untouched. Returns a copy."""
    if name != "MultiConfigResults":
        return matrix
    masked = matrix.copy()
    for c in MASKED_COLS:
        if c < masked.shape[1]:
            masked[:, c] = 0.0
    return masked


def _mat_conversion_differs(new_mat: Path, committed_mat: Path) -> bool:
    """True iff the two .mat files disagree on ANY variable AFTER masking the timing
    column (the only wall-clock source). Used so a re-run does not churn the committed
    .mat snapshot when only the (masked-away) timing column moved."""
    a = scipy.io.loadmat(new_mat)
    b = scipy.io.loadmat(committed_mat)
    for var in MAT_VARS:
        if var not in a or var not in b:
            return True
        av = _mask_timing(var, cast(np.ndarray, a[var]).astype("<f8"))
        bv = _mask_timing(var, cast(np.ndarray, b[var]).astype("<f8"))
        if av.shape != bv.shape or not np.array_equal(av.view(np.uint64), bv.view(np.uint64)):
            return True
    return False


def _best_mat_differs(new_mat: Path, committed_mat: Path) -> bool:
    """Churn guard for the bestNNWeight stats .mat (nbOfInputs/meanInputs/stdInputs):
    True iff any variable's values differ. Same rationale as _mat_conversion_differs
    (matio zlib bytes churn run-to-run; the decoded values are the invariant)."""
    a = scipy.io.loadmat(new_mat)
    b = scipy.io.loadmat(committed_mat)
    for var in TIER2_BEST_MAT_VARS:
        if var not in a or var not in b:
            return True
        av = cast(np.ndarray, a[var]).astype("<f8")
        bv = cast(np.ndarray, b[var]).astype("<f8")
        if av.shape != bv.shape or not np.array_equal(av.view(np.uint64), bv.view(np.uint64)):
            return True
    return False


def _convert_mat(mat_path: Path, out_prefix: str) -> dict[str, list[int]]:
    """Load a harness .mat, write each variable to `<out_prefix>_<var>.bin` (timing
    masked), and return the per-variable (rows, cols) shapes."""
    mat = scipy.io.loadmat(mat_path)
    shapes: dict[str, list[int]] = {}
    for var in MAT_VARS:
        if var not in mat:
            raise SystemExit(f"{mat_path.name}: missing variable {var}")
        arr = cast(np.ndarray, mat[var]).astype("<f8")
        masked = _mask_timing(var, arr)
        out = PHASE4A_DIR / f"{out_prefix}_{var}.bin"
        _write_bin(out, masked)
        shapes[var] = [int(masked.shape[0]), int(masked.shape[1])]
    return shapes


def main() -> None:
    # 1. Build the harness (now with the corpus-driver TUs linked).
    _run(["bash", str(HARNESS_DIR / "build.sh")])

    # 2. Snapshot the prior-phase fixture dirs BEFORE the run (regression guard).
    before = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}

    PHASE4A_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Seed the FIXED corpus workdir (idempotent; cleared each run for byte-repro).
    if FIXED_CORPUS.exists():
        shutil.rmtree(FIXED_CORPUS)
    FIXED_CORPUS.mkdir(parents=True)
    shutil.copytree(CORPUS_DIR, FIXED_CORPUS / "corpus")
    shutil.copy2(FILESLISTING, FIXED_CORPUS / "fileslisting.csv")
    shutil.copy2(MAPPING, FIXED_CORPUS / "language2classmapping.csv")
    shutil.copy2(TIER1_TDC_CONFIG, FIXED_CORPUS / "tier1_tdc.config")
    shutil.copy2(TIER1_LTSV_CONFIG, FIXED_CORPUS / "tier1_ltsv_powermel.config")
    # Task 9 tier-2 seeds: configs + listings + the phase0 weight pack the
    # tier2_spectral.config's relative BLSTM_weightsFile resolves to.
    shutil.copy2(TIER2_SPECTRAL_CONFIG, FIXED_CORPUS / "tier2_spectral.config")
    shutil.copy2(TIER2_GRADCHECK_CONFIG, FIXED_CORPUS / "tier2_gradcheck.config")
    shutil.copy2(TIER2_FILESLISTING, FIXED_CORPUS / "tier2_fileslisting.csv")
    shutil.copy2(TIER2_GC_FILESLISTING, FIXED_CORPUS / "tier2_gc_fileslisting.csv")
    shutil.copy2(NN_WEIGHTS, FIXED_CORPUS / "NNweights_config1.bin")

    # 4. Run the harness: earlier stages into a throwaway dir, the tier-1 stage into
    #    the fixed corpus path (argv[10..12]).
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
                str(FIXED_CORPUS),
                str(FIXED_CORPUS / "tier1_tdc.config"),
                str(FIXED_CORPUS / "tier1_ltsv_powermel.config"),
                str(FIXED_CORPUS / "tier2_spectral.config"),
                str(FIXED_CORPUS / "tier2_gradcheck.config"),
            ]
        )

        # Task 9 tier-2 .bin dumps land in the harness OUTPUT dir (argv[1], this
        # throwaway tempdir) -- collect them before it closes. Byte-deterministic
        # (io::binary from the ascending-loop reimpl + real Rprop), committed as-is.
        tier2_bins = [f"tier2_weights_epoch{e}.bin" for e in range(TIER2_EPOCHS)]
        tier2_bins += ["tier2_gradcheck.bin", "tier2_gradcheck_seed.bin"]
        for name in tier2_bins:
            src = tmp_dir / name
            if not src.is_file():
                raise SystemExit(f"harness did not produce tier-2 dump {name}")
            shutil.copy2(src, PHASE4A_DIR / name)

    nb_match = NB_FILES_RE.search(stdout)
    if not nb_match:
        raise SystemExit("PHASE4A_TIER1 nb_files line missing from harness stdout")
    nb_files = int(nb_match["n"])

    # 5. Convert each run's .mat -> per-variable .bin (timing masked) + copy the raw
    #    .mat (committed for the pytest conversion guard). The .bin dumps are
    #    deterministic (the timing column, the only wall-clock source, is masked to
    #    0.0), but the raw .mat carries the live timing column + matio zlib state, so
    #    it churns every run. Only (re)write the committed .mat when it is ABSENT or
    #    its masked conversion actually DIFFERS from the committed one -- keeping the
    #    committed .mat a stable snapshot across regenerations (git-noise-free) while
    #    still catching a genuine conversion change.
    run_shapes: dict[str, dict[str, list[int]]] = {}
    for run in RUNS:
        mat_src = FIXED_CORPUS / f"{run}.mat"
        if not mat_src.is_file():
            raise SystemExit(f"harness did not produce {mat_src}")
        run_shapes[run] = _convert_mat(mat_src, run)
        committed = PHASE4A_DIR / f"{run}.mat"
        if not committed.is_file() or _mat_conversion_differs(mat_src, committed):
            shutil.copy2(mat_src, committed)

    # 6. Copy the multi-channel VRCTS xml (the carried-over byte oracle).
    for subdir, fname in VRCTS_DUMPS:
        src = FIXED_CORPUS / subdir / fname
        if not src.is_file():
            raise SystemExit(f"harness did not produce VRCTS {src}")
        shutil.copy2(src, PHASE4A_DIR / fname)

    # 6b. Task 9 tier-2 artifacts (train stage, CWD = the fixed corpus workdir).
    # Result .mat -> per-variable .bin (timing col already 0.0 in the transcription;
    # the mask is a no-op kept for pipeline symmetry) + the churn-guarded raw .mat.
    tier2_mat = FIXED_CORPUS / "tier2_spectral.mat"
    if not tier2_mat.is_file():
        raise SystemExit("harness did not produce tier2_spectral.mat")
    tier2_shapes = _convert_mat(tier2_mat, "tier2_spectral")
    committed = PHASE4A_DIR / "tier2_spectral.mat"
    if not committed.is_file() or _mat_conversion_differs(tier2_mat, committed):
        shutil.copy2(tier2_mat, committed)
    # bestNNWeight artifact split (BLSTMNeuralNetwork::saveWeights): the weights +
    # derivs are io::binary files (misleading .mat suffix -- the legacy composes
    # "weights_"+filename where filename ends in .mat); byte-deterministic, copied
    # as-is. The stats .mat (nbOfInputs/meanInputs/stdInputs; EMPTY under the real
    # config's InputNormalizationType -1, which never accumulates stats) is matio
    # zlib -> churn-guarded via its scipy conversion.
    for name in [f"weights_{TIER2_BEST_BASE}", f"weightsDerivatives_{TIER2_BEST_BASE}"]:
        src = FIXED_CORPUS / name
        if not src.is_file():
            raise SystemExit(f"harness did not produce bestNNWeight artifact {name}")
        shutil.copy2(src, PHASE4A_DIR / name)
    best_mat_src = FIXED_CORPUS / TIER2_BEST_BASE
    if not best_mat_src.is_file():
        raise SystemExit(f"harness did not produce {TIER2_BEST_BASE}")
    best_committed = PHASE4A_DIR / TIER2_BEST_BASE
    if not best_committed.is_file() or _best_mat_differs(best_mat_src, best_committed):
        shutil.copy2(best_mat_src, best_committed)
    # tier-2 VRCTS (final epoch's bytes), committed with a tier2_ prefix.
    for fname in TIER2_VRCTS:
        src = FIXED_CORPUS / "vrcts_tier2" / fname
        if not src.is_file():
            raise SystemExit(f"harness did not produce tier-2 VRCTS {src}")
        shutil.copy2(src, PHASE4A_DIR / f"tier2_{fname}")

    # 6c. Task 9 stdout probes: the trajectory (non-vacuity), the SEG_STRUCT
    # secondary probe (ABORT on absence -- a structural mismatch already aborted the
    # harness itself), and the gradCheck means.
    m_train = TIER2_TRAIN_RE.search(stdout)
    if not m_train:
        raise SystemExit("PHASE4A_TIER2_TRAIN line missing from harness stdout")
    if int(m_train["epochs"]) != TIER2_EPOCHS:
        raise SystemExit(f"tier-2 train epochs {m_train['epochs']} != {TIER2_EPOCHS}")
    differ, fired, skipped = int(m_train["differ"]), int(m_train["fired"]), int(m_train["skipped"])
    # NON-VACUITY (measured, not assumed): every consecutive epoch pair differs, and
    # the best-cost gate FIRED at least once AND SKIPPED at least once.
    if differ != TIER2_EPOCHS - 1:
        raise SystemExit(f"tier-2 non-vacuity: only {differ}/{TIER2_EPOCHS - 1} epoch pairs differ")
    if fired < 1 or skipped < 1:
        raise SystemExit(f"tier-2 non-vacuity: gate fired={fired} skipped={skipped} (need both >= 1)")
    best_cost_trajectory = [float(x) for x in m_train["costs"].split(",")]
    m_seg = TIER2_SEG_STRUCT_RE.search(stdout)
    if not m_seg:
        raise SystemExit("SEG_STRUCT site=tier2_train line missing from harness stdout")
    tier2_seg_max_dt = float(m_seg["dt"])
    m_gc = TIER2_GRADCHECK_RE.search(stdout)
    if not m_gc:
        raise SystemExit("PHASE4A_TIER2_GRADCHECK line missing from harness stdout")
    if int(m_gc["sweep"]) != TIER2_GRADCHECK_MAX_WEIGHTS:
        raise SystemExit(f"tier-2 gradcheck sweep {m_gc['sweep']} != {TIER2_GRADCHECK_MAX_WEIGHTS}")
    # The stdout echo carries only 6 significant digits; the manifest records the
    # EXACT doubles from the .bin's means row (row sweep = [mean_err, mean_rel, 0]),
    # cross-checked against the echo.
    gc_bin = _read_bin(PHASE4A_DIR / "tier2_gradcheck.bin")
    if gc_bin.shape != (TIER2_GRADCHECK_MAX_WEIGHTS + 1, 3):
        raise SystemExit(f"tier2_gradcheck.bin shape {gc_bin.shape} unexpected")
    gc_mean_err = float(gc_bin[TIER2_GRADCHECK_MAX_WEIGHTS, 0])
    gc_mean_rel_err = float(gc_bin[TIER2_GRADCHECK_MAX_WEIGHTS, 1])
    for exact, echoed in [(gc_mean_err, float(m_gc["err"])), (gc_mean_rel_err, float(m_gc["rel"]))]:
        if abs(exact - echoed) > 1e-5 * abs(echoed):
            raise SystemExit(f"tier-2 gradcheck .bin mean {exact} != stdout echo {echoed}")
    if not gc_mean_rel_err < 5e-4:
        raise SystemExit(f"tier-2 gradcheck mean_rel_err {gc_mean_rel_err} >= 5e-4")

    # 7. Regression guard: the prior-phase fixture dirs must be byte-identical after.
    after = {ph: _hash_tree(REF_DIR / ph) for ph in PRIOR_PHASES}
    for ph in PRIOR_PHASES:
        b, a = before[ph], after[ph]
        drifted = sorted(name for name in set(b) | set(a) if b.get(name) != a.get(name))
        if drifted:
            raise SystemExit(f"REGRESSION: {ph} fixtures changed after harness run: {drifted}")

    # 8. Compiler version (same g++ selection as build.sh) + libmatio version.
    gxx = _run(["bash", "-c", 'ls "$(brew --prefix)"/bin/g++-* | sort -V | tail -1']).strip()
    compiler = _run([gxx, "--version"]).splitlines()[0].strip()
    libmatio_version = _run(["bash", "-c", "brew list --versions libmatio"]).strip()

    manifest = {
        "compiler": compiler,
        "flags": (
            "-O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off "
            "-DEIGEN_DONT_VECTORIZE -include boost/math/special_functions/round.hpp"
        ),
        "libmatio_version": libmatio_version,
        "corpus_driver_linkage": {
            "text": (
                "Phase 4a Task 8: the REAL CorpusProcessor stack linked into the oracle "
                "harness (CorpusProcessor.cpp, BagOfProcessors.cpp, Corpus.cpp; CorpusItem.cpp "
                "already linked for AudioStruct). BagOfProcessors's member vectors force "
                "VRCTSpart.cpp / BLSTMSpectralLID.cpp / TwinBLSTMSpectralLID.cpp (constructed/"
                "destructed even for algo 1/2). TwinBLSTMSpectralLID drags "
                "ConvolutionalNeuralNetwork.cpp / ConvolutionalLayer.cpp (broken-as-committed "
                "at runtime, but they link and are never constructed here). CorpusProcessor.cpp's "
                "free CalcThread() (the commented-out threading path) odr-uses "
                "FileDispatcher.cpp. The omp.h no-op shim keeps the #pragma omp parallel-for "
                "sequential (NO -fopenmp) == the N=1 golden parity mode. -lmatio was already "
                "linked (Phase 2b Helpers.hpp/AudioStruct)."
            ),
            "translation_units": [
                "CorpusProcessor.cpp",
                "BagOfProcessors.cpp",
                "Corpus.cpp",
                "VRCTSpart.cpp",
                "BLSTMSpectralLID.cpp",
                "TwinBLSTMSpectralLID.cpp",
                "FileDispatcher.cpp",
                "ConvolutionalNeuralNetwork.cpp",
                "ConvolutionalLayer.cpp",
            ],
        },
        "corpus": {
            "text": (
                "Committed 3-file 2-channel corpus (3 copies of the phase1 "
                "excerpt_2ch_8k.wav) with crafted 2-channel STMs (one SPEECH span per "
                "channel inside 2.0s). fileslisting.csv exercises the weight column and "
                "the defaults: row 1 full (eng;us;0.5;2), row 2 present-weight with empty "
                "lang/dial siblings (;;0.75), row 3 defaults-only (wav;stm). Every row is "
                "referenced (STM) so the -m scored runs never hit the mandatory-reference "
                "bail. language2classmapping.csv maps 2 languages (eng;us->0, unk;unk->1) "
                "to 2 classes."
            ),
            "nb_files": nb_files,
            "class_index": {"f1": 0, "f2": 1, "f3": 1},
            "weights": {"f1": 0.5, "f2": 0.75, "f3": 1.0},
            "file_ids": {"f1": 2, "f2": 1, "f3": 1},
            "language_count": {"eng": 1, "unk": 2},
            "class_count": {"0": 1, "1": 2},
        },
        "runs": {
            "solo_tdc": {
                "mode": "-s",
                "algo": 1,
                "config": "tier1_tdc.config",
                "epochs": 0,
                "text": (
                    "solo (-s) is UNSCORED (scored branch is m/M/t/T only): runSolo -> "
                    "CostMem 1 x nbConf. The MultiConfigResults rows are unscored (error "
                    "cols 0/1/2 = 0), 3 files x 2 channels = 6 rows. VRCTS is always "
                    "written (unscored branch) -> the multi-channel byte oracle."
                ),
                "shapes": run_shapes["solo_tdc"],
            },
            "train_ltsv": {
                "mode": "-m",
                "algo": 2,
                "config": "tier1_ltsv_powermel.config",
                "epochs": 2,
                "text": (
                    "train (-m) with epochs 2: train() runs epoch 0 (solo) then 2 inner "
                    "epochs then a final eval at epoch 3, each saveResults writing "
                    "topRows(epoch+1); the last writes topRows(4) so CostMem is 4 x 1. "
                    "LTSV/TDC have NO weight updates, so CostMem rows are EXPECTED "
                    "IDENTICAL across epochs (tier-1 pins the loop/aggregation plumbing; "
                    "weight evolution is tier-2)."
                ),
                "shapes": run_shapes["train_ltsv"],
            },
            "multiconfig": {
                "mode": "-m",
                "algo": [1, 2],
                "config": ["tier1_tdc.config", "tier1_ltsv_powermel.config"],
                "epochs": 0,
                "text": (
                    "multiconfig (-m) with both configs, epochs 0 (overridden on "
                    "configs[0]): a single scored pass. MultiConfigResults interleaves "
                    "(file, conf, chan): 3 files x 2 confs x 2 channels = 12 rows, "
                    "ordered file-ascending then conf-ascending then chan-ascending "
                    "(transformResults BTreeMap iteration). CostMem is 1 x 2."
                ),
                "shapes": run_shapes["multiconfig"],
            },
        },
        "mat_conversion": {
            "text": (
                "Each harness .mat (saveResults writes 5 vars: MultiConfigResults, "
                "CostMem, BadClassifMem, CostLIDMem, BadClassifLIDMem via "
                "Matrix2MatFile, MAT_COMPRESSION_ZLIB) is converted to per-variable .bin "
                "(io::binary: i64 LE rows, i64 LE cols, f64 LE column-major) via "
                "scipy.io.loadmat. The wall-clock timing column of MultiConfigResults "
                "(col 6 = the 4th data column after [file, conf, chan] = time_per_hour) "
                "is ZEROED so the dump is byte-reproducible; the Rust golden masks the "
                "same column. The RAW .mat outputs are committed too so the pytest guard "
                "re-reads them with scipy and checks the conversion (with the same col-6 "
                "mask)."
            ),
            "vars": MAT_VARS,
            "masked_cols": MASKED_COLS,
        },
        "vrcts": {
            "text": (
                "The carried-over multi-channel VRCTS byte oracle (IMPROVEMENTS "
                "'Single-channel VRCTS write on the corpus path'). The REAL compiled "
                "Segmentation::toFile_VRCTS fans out one <basefilename>_chan_<n>.xml per "
                "channel for the 2-channel excerpt; name= is the audio basename minus "
                "extension, path= is the relative audio path (corpus/fN.wav), both from a "
                "FIXED corpus workdir so the attrs are byte-reproducible. The Rust "
                "write_vrcts_multichannel output must byte-match these."
            ),
            "dumps": [f"{sub}/{f}" for sub, f in VRCTS_DUMPS],
            "committed_names": [f for _, f in VRCTS_DUMPS],
        },
        "tier2": {
            "train": {
                "text": (
                    "Task 9 tier-2 CorpusProbe train (Algo 3 reimpl-swap): the harness "
                    "transcribes CorpusProcessor::train/run(epoch) + BagOfProcessors::"
                    "SegmentationFunction/saveAndUpdate around the spectral reimpl-swap "
                    "(ascending-loop forward+backward via BlstmBack + REAL CostLaw), "
                    "feeding the reimpl derivs into the REAL compiled updateWeights/"
                    "saveWeights (real Rprop) through their external-derivs arguments. "
                    "Config: tier2_spectral.config = the phase0 1_worker_1.config "
                    "re-pointed at the 2-file corpus (f1/f2), BackPropagationActivated "
                    "true, epochs 3, RpropInit 0.05 (chosen so the measured best-cost "
                    "gate trajectory both FIRES and SKIPS -- 1e-2 default fires every "
                    "epoch), numOuterThreads 1 (== the Rust static-lane N=1 parity "
                    "mode). Dumps the post-saveAndUpdate weight vector per epoch (0 = "
                    "post-solo .. 4 = post-final). The transcription's time_per_hour "
                    "result column is hard-zeroed (the Task 8 mask value), so the .mat "
                    "conversion mask is a no-op kept for pipeline symmetry."
                ),
                "config": "tier2_spectral.config",
                "epochs": TIER2_EPOCHS,
                "epoch_weight_files": [f"tier2_weights_epoch{e}.bin" for e in range(TIER2_EPOCHS)],
                "rprop_init": 0.05,
                "non_vacuity": {
                    "text": (
                        "MEASURED (harness stdout): every consecutive epoch weight pair "
                        "differs; the bestNNWeight save gate FIRED and SKIPPED at least "
                        "once each. The Rust golden re-asserts the same pattern from its "
                        "own trace."
                    ),
                    "epochs_differing": differ,
                    "gate_fired": fired,
                    "gate_skipped": skipped,
                    "best_cost_trajectory": best_cost_trajectory,
                },
                "seg_struct": {
                    "text": (
                        "SECONDARY structural probe (2b convention): the REAL Eigen "
                        "getSegmentation ran beside the reimpl per file at the initial "
                        "weights; segment count/types matched exactly (mismatch aborts "
                        "fixture generation), max boundary dt recorded."
                    ),
                    "site": "tier2_train",
                    "ok": 1,
                    "max_dt": tier2_seg_max_dt,
                },
                "shapes": tier2_shapes,
                "best_artifacts": [
                    f"weights_{TIER2_BEST_BASE}",
                    f"weightsDerivatives_{TIER2_BEST_BASE}",
                    TIER2_BEST_BASE,
                ],
                "vrcts": [f"tier2_{f}" for f in TIER2_VRCTS],
            },
            "gradcheck": {
                "text": (
                    "Task 9 tier-2 corpus-level gradCheck (CorpusProcessor.cpp:237-340) "
                    "on the synthetic signal net ([2,2] LSTM + [4,1] output, 137 "
                    "weights, algo 4) over the 1-file corpus (f1), seeded with the "
                    "phase3 synth_flat pattern (dumped as tier2_gradcheck_seed.bin -- "
                    "the single source of truth the Rust golden loads). "
                    "tier2_gradcheck.bin rows 0..sweep-1 = [analytic_col0, "
                    "analytic_col1, numerical] per weight; final row = [mean_error, "
                    "mean_relative_error, 0]."
                ),
                "config": "tier2_gradcheck.config",
                "gradcheck_max_weights": TIER2_GRADCHECK_MAX_WEIGHTS,
                "epsilon": 1e-5,
                "mean_error": gc_mean_err,
                "mean_relative_error": gc_mean_rel_err,
            },
        },
    }

    manifest_path = PHASE4A_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"OK: phase4a tier-1+2 fixtures (nb_files={nb_files}, "
        f"runs={RUNS}, masked_cols={MASKED_COLS}, "
        f"vrcts={len(VRCTS_DUMPS)} xml; tier2 fired={fired} skipped={skipped} "
        f"differ={differ} seg_dt={tier2_seg_max_dt} gc_rel={gc_mean_rel_err}), "
        f"manifest -> {manifest_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
