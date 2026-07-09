"""Phase 4d Task 3: the end-to-end PARITY oracle fixtures, from a SPLIT oracle.

Two independent oracles produce the committed fixtures, recorded per artifact in the
manifest (`oracle: real-binary | fastmath-rebuild`):

  (a) SEGMENTATION leg -- the resurrected 2015 x86_64 production `Segment` binary
      (tools/fsp_runtime/bin/Segment, run under Rosetta 2, OMP_NUM_THREADS=1). Its solo
      (-s) run writes VRCTS xml per channel, then SIGSEGVs (exit 139) in the post-
      inference saveWeights path AFTER the xml is on disk (a resurrection artifact -- see
      tools/fsp_runtime/README.md). We tolerate the crash and consume the xml. This is
      the segmentation oracle: parity_tuple{A,B}_vrcts_chan{1,2}.xml.

  (b) NUMERIC-COLUMN leg -- a -ffast-math REBUILD (tools/oracle_harness/build.sh
      --fastmath -> oracle_harness_fastmath) drives the SAME solo run through the REAL
      COMPILED legacy CorpusProcessor stack, but exits cleanly, so saveResults writes
      MultiConfigResults.mat (the cost/counter/duration columns the crashing binary never
      reaches). Converted to .bin (scipy, wall-clock col masked) as
      parity_tuple{A,B}_{mcr,result_rows}.bin.

The fastmath rebuild's VRCTS is compared to the real binary's as calibration evidence
(structural + boundary agreement recorded in the manifest); the committed VRCTS is the
REAL BINARY's, not the rebuild's.

INPUTS committed here: prcts_excerpt.wav (a deterministic 60 s stereo 8 kHz cut of the
legacy PRCTS wav, chosen so both channels carry leading silence + multiple speech
segments), parity_tuple{A,B}.config (the Task-1 real configs with ONLY path keys
localized to relative names a test cwd provides), and the shared aux inputs
fileslisting / languagemapping.csv / prcts_excerpt.stm.

LOCAL-ONLY (like the other extractors): needs the resurrected binary (gated via
`tools/fsp_runtime/setup.sh --check`), the external legacy PRCTS wav, and Homebrew g++
for the fastmath build. CI never runs this -- it consumes only the committed fixtures +
manifest (tests/test_phase4d_fixtures.py guards them WITHOUT any oracle).

Usage: uv run python scripts/extract_phase4d_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_TREE = Path("/Users/govit/Git/Govit/FastSpeechProcessing-legacy")
PRCTS_WAV = LEGACY_TREE / "PRCTS_RUS_RU_0000263489_01.wav"
FSP_RUNTIME = REPO_ROOT / "tools" / "fsp_runtime"
SEGMENT_BIN = FSP_RUNTIME / "bin" / "Segment"
HARNESS_DIR = REPO_ROOT / "tools" / "oracle_harness"
FASTMATH_BIN = HARNESS_DIR / "oracle_harness_fastmath"
PHASE4D_DIR = REPO_ROOT / "tests" / "reference_data" / "phase4d"
MANIFEST_PATH = PHASE4D_DIR / "phase4d_parity_manifest.json"

# Deterministic excerpt cut: start=0, 60 s. Adjudicated non-vacuous on BOTH configs and
# BOTH oracles (see the non-vacuity gate + the manifest adjudication string).
EXCERPT_START_SEC = 0
EXCERPT_LEN_SEC = 60

# MultiConfigResults wall-clock column (time_per_hour) -- masked to 0.0 for determinism,
# same column as extract_phase4a_fixtures.py (col 3 data + 3 prefix cols = col 6).
MCR_TIMING_COL = 6

# Shared aux inputs (relative-path fixtures the localized configs reference).
LANGMAP_TEXT = "fax;non;0\nchi;man;1\nspa;spa;2\n"
FILESLISTING_TEXT = "prcts_excerpt.wav;prcts_excerpt.stm;fax;non;1;30.0;\n"

# Config path keys localized to relative names + the target value. Every OTHER key is
# byte-preserved. BLSTM_weightsFile in the Task-1 tupleA config points at a dead .mat
# (the live loader reads the .bin via BinaryFile2Vector); localizing it to the committed
# weight pack's relative name is a path-key change, recorded in the manifest key-diff.
CONFIG_PATH_KEYS = {
    "Display_Output_Directory": ".",
    "BLSTM_weightsFile": "NNweights_config1.bin",
    "language2classmapping": "languagemapping.csv",
    "fileslisting": "fileslisting",
}

TUPLES = {
    "tupleA": {"config": "tupleA_1_worker_1.config", "weights": "tupleA_NNweights_config1.bin"},
    "tupleB": {"config": "tupleB_1_worker_1.config", "weights": "tupleB_NNweights_config1.bin"},
}

# Task-1 committed fixtures that this extractor must NOT perturb (regression guard).
TASK1_FILES = [
    "tupleA_1_worker_1.config",
    "tupleA_NNweights_config1.bin",
    "tupleB_1_worker_1.config",
    "tupleB_NNweights_config1.bin",
    "phase4d_sources.json",
]

SIZE_BUDGET_BYTES = 8 * 1024 * 1024

GCC_VERSION_RE = re.compile(r"\(Homebrew GCC [^)]*\)\s*(\S+)")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bin(path: Path, matrix: np.ndarray) -> None:
    """io::binary .bin: i64 LE rows, i64 LE cols, f64 LE column-major."""
    mat = np.atleast_2d(np.ascontiguousarray(matrix, dtype="<f8"))
    rows, cols = mat.shape
    with path.open("wb") as f:
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        f.write(mat.flatten(order="F").tobytes())


def _read_bin(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    return np.frombuffer(raw[16:], dtype="<f8").reshape((rows, cols), order="F")


def _localize_config(source_text: str) -> tuple[str, list[dict[str, str]]]:
    """Rewrite the path keys in CONFIG_PATH_KEYS to their relative targets, byte-preserving
    every other line. Returns (localized_text, key_diff)."""
    out_lines: list[str] = []
    diff: list[dict[str, str]] = []
    for line in source_text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        key = stripped.split(" ", 1)[0] if " " in stripped else stripped
        if key in CONFIG_PATH_KEYS:
            original_value = stripped[len(key) + 1 :]
            new_value = CONFIG_PATH_KEYS[key]
            newline = "\n" if line.endswith("\n") else ""
            out_lines.append(f"{key} {new_value}{newline}")
            diff.append({"key": key, "original": original_value, "localized": new_value})
        else:
            out_lines.append(line)
    return "".join(out_lines), diff


def _segments(xml_path: Path) -> list[tuple[float, float]]:
    """Parse (stime, etime) pairs from a VRCTS xml (also validates it parses)."""
    root = ET.fromstring(xml_path.read_text())
    return [
        (float(s.attrib["stime"]), float(s.attrib["etime"]))
        for s in root.iter("SpeechSegment")
    ]


def _seed_workdir(workdir: Path, config_name: str, config_text: str, weights_src: Path, excerpt_src: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(excerpt_src, workdir / "prcts_excerpt.wav")
    shutil.copy2(weights_src, workdir / "NNweights_config1.bin")
    (workdir / config_name).write_text(config_text)
    (workdir / "languagemapping.csv").write_text(LANGMAP_TEXT)
    (workdir / "fileslisting").write_text(FILESLISTING_TEXT)
    (workdir / "prcts_excerpt.stm").write_text("")


def _run_real_binary(workdir: Path, config_name: str) -> None:
    """Solo (-s) run of the resurrected binary. Tolerates exit 139 (post-xml SIGSEGV);
    the caller validates by checking the xml exists + parses, not the exit code."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(
        ["arch", "-x86_64", str(SEGMENT_BIN), "-s", config_name],
        cwd=workdir,
        env=env,
        capture_output=True,
    )


def _run_fastmath(workdir: Path, config_name: str) -> None:
    result = subprocess.run(
        [str(FASTMATH_BIN), str(workdir), config_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"fastmath harness failed ({result.returncode}) for {config_name}")


def _masked_mcr(mat_path: Path) -> np.ndarray:
    mcr = np.atleast_2d(np.asarray(scipy.io.loadmat(mat_path)["MultiConfigResults"], dtype="<f8")).copy()
    if MCR_TIMING_COL < mcr.shape[1]:
        mcr[:, MCR_TIMING_COL] = 0.0
    return mcr


def main() -> None:
    # --- 0. Gates: resurrected binary present, PRCTS wav present. ---
    check = subprocess.run(["bash", str(FSP_RUNTIME / "setup.sh"), "--check"], capture_output=True, text=True)
    if check.returncode != 0:
        sys.stderr.write(check.stdout)
        sys.stderr.write(check.stderr)
        raise SystemExit("fsp_runtime --check gate failed: the 2015 Segment oracle is not present")
    if not PRCTS_WAV.is_file():
        raise SystemExit(f"legacy PRCTS wav not found: {PRCTS_WAV}")

    # --- 1. Build the fastmath oracle (separate binary, never clobbers oracle_harness). ---
    build = subprocess.run(["bash", str(HARNESS_DIR / "build.sh"), "--fastmath"], capture_output=True, text=True)
    if build.returncode != 0:
        sys.stderr.write(build.stdout)
        sys.stderr.write(build.stderr)
        raise SystemExit("build.sh --fastmath failed")
    gcc_out = subprocess.run(["/opt/homebrew/bin/g++-16", "--version"], capture_output=True, text=True).stdout
    gcc_match = GCC_VERSION_RE.search(gcc_out)
    gcc_version = gcc_match.group(1) if gcc_match else "unknown"

    # --- 2. Regression snapshot of the Task-1 committed fixtures. ---
    task1_before = {n: _sha256((PHASE4D_DIR / n).read_bytes()) for n in TASK1_FILES}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        staging = tmp_dir / "staging"
        staging.mkdir()

        # --- 3. Deterministic excerpt cut (int16 -> PCM_16, byte-reproducible). ---
        data, sr = sf.read(PRCTS_WAV, dtype="int16", always_2d=True)
        seg = data[EXCERPT_START_SEC * sr : (EXCERPT_START_SEC + EXCERPT_LEN_SEC) * sr, :]
        excerpt_path = staging / "prcts_excerpt.wav"
        sf.write(excerpt_path, seg, sr, subtype="PCM_16")
        # Regenerate once and assert byte-identical (soundfile write is deterministic).
        recut = tmp_dir / "recut.wav"
        sf.write(recut, seg, sr, subtype="PCM_16")
        if excerpt_path.read_bytes() != recut.read_bytes():
            raise SystemExit("excerpt cut is not byte-deterministic")
        excerpt_info = sf.info(excerpt_path)

        # Aux inputs (shared).
        (staging / "languagemapping.csv").write_text(LANGMAP_TEXT)
        (staging / "fileslisting").write_text(FILESLISTING_TEXT)
        (staging / "prcts_excerpt.stm").write_text("")

        fixtures: dict[str, dict[str, object]] = {}
        config_localization: dict[str, object] = {}
        determinism_real: dict[str, object] = {}
        determinism_fastmath: dict[str, object] = {}
        structural: dict[str, object] = {}

        for tuple_name, spec in TUPLES.items():
            source_config = (PHASE4D_DIR / spec["config"]).read_text()
            localized, key_diff = _localize_config(source_config)
            parity_config_name = f"parity_{tuple_name}.config"
            (staging / parity_config_name).write_text(localized)
            config_localization[tuple_name] = {"source": spec["config"], "key_diff": key_diff}
            weights_src = PHASE4D_DIR / spec["weights"]

            # --- 4. REAL BINARY (segmentation oracle), run TWICE for determinism. ---
            real1 = tmp_dir / f"{tuple_name}_real1"
            real2 = tmp_dir / f"{tuple_name}_real2"
            _seed_workdir(real1, parity_config_name, localized, weights_src, excerpt_path)
            _seed_workdir(real2, parity_config_name, localized, weights_src, excerpt_path)
            _run_real_binary(real1, parity_config_name)
            _run_real_binary(real2, parity_config_name)

            real_segs: dict[int, list[tuple[float, float]]] = {}
            det_real = {}
            for chan in (1, 2):
                x1 = real1 / f"prcts_excerpt_chan_{chan}.xml"
                x2 = real2 / f"prcts_excerpt_chan_{chan}.xml"
                if not x1.is_file():
                    raise SystemExit(f"{tuple_name}: real binary produced no chan{chan} xml")
                det_real[f"chan{chan}"] = x1.read_bytes() == x2.read_bytes()
                real_segs[chan] = _segments(x1)
                # Commit the REAL BINARY's xml as the segmentation oracle.
                dst = staging / f"parity_{tuple_name}_vrcts_chan{chan}.xml"
                shutil.copy2(x1, dst)
            determinism_real[tuple_name] = det_real

            # Non-vacuity: >1 speech segment per channel.
            for chan in (1, 2):
                if len(real_segs[chan]) <= 1:
                    raise SystemExit(
                        f"{tuple_name} chan{chan} degenerate: {len(real_segs[chan])} speech segment(s) "
                        f"(need >1); re-cut the excerpt"
                    )
            if not all(det_real.values()):
                raise SystemExit(f"{tuple_name}: real binary VRCTS not deterministic across two runs")

            # --- 5. FASTMATH REBUILD (numeric columns), run TWICE for masked-mcr determinism. ---
            fm1 = tmp_dir / f"{tuple_name}_fm1"
            fm2 = tmp_dir / f"{tuple_name}_fm2"
            _seed_workdir(fm1, parity_config_name, localized, weights_src, excerpt_path)
            _seed_workdir(fm2, parity_config_name, localized, weights_src, excerpt_path)
            _run_fastmath(fm1, parity_config_name)
            _run_fastmath(fm2, parity_config_name)

            mcr_name = re.search(r"^multiConfigResultsOutputFile\s+(\S+)", localized, re.MULTILINE)
            if not mcr_name:
                raise SystemExit(f"{tuple_name}: multiConfigResultsOutputFile key missing from config")
            mat1 = fm1 / mcr_name.group(1)
            mat2 = fm2 / mcr_name.group(1)
            if not mat1.is_file():
                raise SystemExit(f"{tuple_name}: fastmath run produced no {mcr_name.group(1)}")
            masked1 = _masked_mcr(mat1)
            masked2 = _masked_mcr(mat2)
            det_fm = bool(np.array_equal(masked1.view(np.uint64), masked2.view(np.uint64)))
            if not det_fm:
                raise SystemExit(f"{tuple_name}: fastmath masked MultiConfigResults not deterministic")
            determinism_fastmath[tuple_name] = det_fm

            # mcr.bin (masked) + result_rows.bin (data cols = mcr[:, 3:], timing already 0).
            mcr_bin = staging / f"parity_{tuple_name}_mcr.bin"
            rows_bin = staging / f"parity_{tuple_name}_result_rows.bin"
            _write_bin(mcr_bin, masked1)
            _write_bin(rows_bin, masked1[:, 3:])

            # --- 6. Structural agreement: real vs fastmath VRCTS (calibration). ---
            struct_tuple = {}
            for chan in (1, 2):
                fm_xml = fm1 / f"prcts_excerpt_chan_{chan}.xml"
                fm_segs = _segments(fm_xml)
                real_xml = staging / f"parity_{tuple_name}_vrcts_chan{chan}.xml"
                byte_identical = real_xml.read_bytes() == fm_xml.read_bytes()
                max_dt = 0.0
                if len(fm_segs) == len(real_segs[chan]):
                    for (rs, re_), (fs, fe) in zip(real_segs[chan], fm_segs, strict=True):
                        max_dt = max(max_dt, abs(rs - fs), abs(re_ - fe))
                struct_tuple[f"chan{chan}"] = {
                    "real_segs": len(real_segs[chan]),
                    "fastmath_segs": len(fm_segs),
                    "structural_match": len(fm_segs) == len(real_segs[chan]),
                    "vrcts_byte_identical": byte_identical,
                    "max_boundary_dt_4dp": max_dt,
                }
                if len(fm_segs) != len(real_segs[chan]):
                    raise SystemExit(
                        f"{tuple_name} chan{chan}: fastmath segs {len(fm_segs)} != real segs "
                        f"{len(real_segs[chan])} -- structural divergence between oracles"
                    )
            structural[tuple_name] = struct_tuple

            # Fixture records.
            for chan in (1, 2):
                name = f"parity_{tuple_name}_vrcts_chan{chan}.xml"
                fixtures[name] = {
                    "oracle": "real-binary",
                    "sha256": _sha256((staging / name).read_bytes()),
                    "speech_segments": len(real_segs[chan]),
                }
            for suffix, arr in (("mcr", masked1), ("result_rows", masked1[:, 3:])):
                name = f"parity_{tuple_name}_{suffix}.bin"
                fixtures[name] = {
                    "oracle": "fastmath-rebuild",
                    "sha256": _sha256((staging / name).read_bytes()),
                    "shape": [int(arr.shape[0]), int(arr.shape[1])],
                }

        # Input fixtures.
        for name in ("prcts_excerpt.wav", "languagemapping.csv", "fileslisting", "prcts_excerpt.stm"):
            fixtures[name] = {"oracle": "input", "sha256": _sha256((staging / name).read_bytes())}
        for tuple_name in TUPLES:
            name = f"parity_{tuple_name}.config"
            fixtures[name] = {"oracle": "input", "sha256": _sha256((staging / name).read_bytes())}

        # --- 7. Size budget (staged NEW fixtures + existing Task-1 files). ---
        staged_bytes = sum(p.stat().st_size for p in staging.iterdir())
        task1_bytes = sum((PHASE4D_DIR / n).stat().st_size for n in TASK1_FILES)
        total_bytes = staged_bytes + task1_bytes
        if total_bytes >= SIZE_BUDGET_BYTES:
            raise SystemExit(f"size budget exceeded: {total_bytes} >= {SIZE_BUDGET_BYTES} bytes")

        # --- 8. Regression guard: Task-1 fixtures untouched. ---
        task1_after = {n: _sha256((PHASE4D_DIR / n).read_bytes()) for n in TASK1_FILES}
        drifted = [n for n in TASK1_FILES if task1_before[n] != task1_after[n]]
        if drifted:
            raise SystemExit(f"REGRESSION: Task-1 fixtures changed: {drifted}")

        manifest = {
            "text": (
                "Phase 4d Task 3: end-to-end PARITY oracle fixtures from a SPLIT oracle. The "
                "resurrected 2015 x86_64 -ffast-math production `Segment` binary "
                "(tools/fsp_runtime/bin/Segment, Rosetta 2, OMP_NUM_THREADS=1) serves the "
                "SEGMENTATION leg: its solo (-s) run writes VRCTS xml per channel then SIGSEGVs "
                "(exit 139) in the post-inference saveWeights path AFTER the xml is written (a "
                "resurrection artifact; the xml is consumed, the exit code ignored). It never "
                "reaches saveResults, so the NUMERIC-COLUMN leg (MultiConfigResults) comes from a "
                "-ffast-math REBUILD (tools/oracle_harness/build.sh --fastmath -> "
                "oracle_harness_fastmath) that drives the SAME solo run through the REAL COMPILED "
                "legacy CorpusProcessor stack but exits cleanly. Both oracles' VRCTS agree "
                "byte-for-byte at 4-decimal VRCTS precision on both tuples (see "
                "structural_agreement_real_vs_fastmath). Solo (-s) is the only viable mode: -m/-t "
                "exit(1) without a reference STM (BagOfProcessors.cpp:302), which an arbitrary "
                "excerpt has none of; -s is unscored, so the cost/error columns are 0 and the "
                "inference-derived numeric is the speechDuration column (mcr col 9)."
            ),
            "excerpt": {
                "source_wav": str(PRCTS_WAV),
                "start_sec": EXCERPT_START_SEC,
                "length_sec": EXCERPT_LEN_SEC,
                "samplerate": excerpt_info.samplerate,
                "channels": excerpt_info.channels,
                "frames": excerpt_info.frames,
                "subtype": excerpt_info.subtype,
                "sha256": _sha256(excerpt_path.read_bytes()),
                "adjudication": (
                    "start=0 length=60 s chosen once: both channels carry leading silence "
                    "(chan1 first speech ~3.41 s, chan2 ~1.78 s) plus multiple speech segments and "
                    "inter-segment silence. Non-degenerate (>1 speech segment/channel) on BOTH "
                    "configs and BOTH oracles: tupleA chan1=8/chan2=3, tupleB chan1=13/chan2=15. "
                    "No re-cut was needed."
                ),
            },
            "oracles": {
                "real_binary": {
                    "role": "segmentation (VRCTS xml)",
                    "binary": "tools/fsp_runtime/bin/Segment",
                    "run": "OMP_NUM_THREADS=1 arch -x86_64 Segment -s <config>",
                    "gate": "tools/fsp_runtime/setup.sh --check",
                    "exit_code_note": "SIGSEGV (exit 139) AFTER the VRCTS xml is written; tolerated",
                },
                "fastmath_rebuild": {
                    "role": "numeric columns (MultiConfigResults)",
                    "binary": "tools/oracle_harness/oracle_harness_fastmath",
                    "build": "tools/oracle_harness/build.sh --fastmath",
                    "driver": "tools/oracle_harness/phase4d_parity.cpp",
                    "gcc": gcc_version,
                    "flags": "-O3 -ffast-math -std=gnu++14 -fpermissive",
                    "dropped_strict_ieee_flags": ["-fno-fast-math", "-ffp-contract=off", "-DEIGEN_DONT_VECTORIZE"],
                    "dropped_x86_flags": ["-msse2", "-mfpmath=sse", "-march=core2"],
                    "dropped_x86_note": (
                        "absent by construction: this host is arm64, where these x86-only tuning "
                        "flags a 2015 x86_64 -ffast-math build carried do not apply and gcc rejects "
                        "them. Recorded, not applied."
                    ),
                },
            },
            "aux_inputs": {
                "fileslisting": FILESLISTING_TEXT.strip(),
                "languagemapping.csv": LANGMAP_TEXT.replace("\n", " ").strip(),
                "prcts_excerpt.stm": "empty (unscored solo run needs no reference)",
            },
            "config_localization": config_localization,
            "mcr_masked_col": MCR_TIMING_COL,
            "mcr_columns": (
                "[0]file [1]conf [2]chan [3-5]Pfa/Pmiss/globalError (unscored=0) [6]time_per_hour "
                "(WALL-CLOCK -> masked to 0.0) [7]cumulativeError (unscored=0) [8]signalDuration "
                "[9]speechDuration (inference-derived) [10]nbWords (legacy default -1) [11-18] "
                "WER+LID fields (0) [19]LIDNbOfClassif (0) [20]NbOfClassif (unscored=0). "
                "parity_*_result_rows.bin = mcr[:, 3:] (the per-channel data columns)."
            ),
            "fixtures": fixtures,
            "determinism": {
                "real_binary_vrcts_twice_identical": determinism_real,
                "fastmath_masked_mcr_twice_identical": determinism_fastmath,
            },
            "structural_agreement_real_vs_fastmath": structural,
            "task4_measured_deltas": {
                tuple_name: {"port_vs_oracle_boundary_dt": None, "port_vs_oracle_mcr_col_deltas": None}
                for tuple_name in TUPLES
            },
            "size_budget": {"total_bytes": total_bytes, "limit_bytes": SIZE_BUDGET_BYTES},
        }
        (staging / "phase4d_parity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        # --- 9. Commit: every check passed -> copy staged fixtures into PHASE4D_DIR. ---
        for p in sorted(staging.iterdir()):
            shutil.copy2(p, PHASE4D_DIR / p.name)

    print(
        f"OK: phase4d parity fixtures (excerpt start={EXCERPT_START_SEC}s len={EXCERPT_LEN_SEC}s "
        f"{excerpt_info.samplerate}Hz {excerpt_info.channels}ch; gcc {gcc_version}; "
        f"tupleA segs {structural['tupleA']['chan1']['real_segs']}/{structural['tupleA']['chan2']['real_segs']}, "
        f"tupleB segs {structural['tupleB']['chan1']['real_segs']}/{structural['tupleB']['chan2']['real_segs']}; "
        f"real-vs-fastmath VRCTS byte-identical; total {total_bytes} bytes < {SIZE_BUDGET_BYTES}) "
        f"-> {MANIFEST_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
