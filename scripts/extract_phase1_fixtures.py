"""Build and run the Phase 1 oracle harness, then regenerate the fixture manifest.

Compiles ``tools/oracle_harness`` against the vendored legacy feature sources with
strict IEEE floating-point flags, runs it to (re)produce the ``.bin`` dumps under
``tests/reference_data/phase1/``, and writes ``manifest.json`` recording the exact
toolchain, flags, harness constants, dump inventory (with shapes), and an anchor
value so later tasks can detect drift. The ``.bin`` codec is the phase-0a format
(i64 rows, i64 cols, f64 column-major), read here via ``speech.weight_bridge``.

Usage: uv run python scripts/extract_phase1_fixtures.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from speech.weight_bridge import read_bin

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "tools" / "oracle_harness"
FIXTURE_DIR = REPO_ROOT / "tests" / "reference_data" / "phase1"

# Harness constants ALL later tasks reuse (kept in sync with main.cpp).
CONSTANTS = {
    "offset_sec": 0.35,
    "max_dur_sec": 2.0,
    "preemph": 0.97,
    "noise_ratio": 0.001,
}

# The exact compile flag string; the presence test asserts the parity flags are here.
FLAGS = (
    "-O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off "
    "-DEIGEN_DONT_VECTORIZE -include boost/math/special_functions/round.hpp"
)

# Homebrew deps whose versions we record for reproducibility.
BREW_DEPS = ["gcc", "eigen@3", "libsndfile", "libmatio", "boost", "png++", "sse2neon"]

# Dumps the harness produces, with their expected (rows, cols).
EXPECTED_SHAPES = {
    "sig_norm.bin": (2, 16001),
    "sig_preemph.bin": (2, 16001),
    "sig_noise.bin": (2, 16001),
    "win_hamming_257.bin": (1, 257),
    "win_hann_257.bin": (1, 257),
    "win_hhcw_257.bin": (1, 257),
    "win_conv_norm_7.bin": (1, 7),
    "synth_20x50.bin": (20, 50),
    "fft_in_2.bin": (1, 4),
    "fft_out_2.bin": (1, 4),
    "fft_in_4.bin": (1, 8),
    "fft_out_4.bin": (1, 8),
    "fft_in_8.bin": (1, 16),
    "fft_out_8.bin": (1, 16),
    "fft_in_256.bin": (1, 512),
    "fft_out_256.bin": (1, 512),
    # Two-real periodogram (p=8 -> length 129; full end=16000 -> frameNb=201).
    "perio_p8_s80_chan1.bin": (201, 129),
    "perio_p8_s80_chan2.bin": (201, 129),
    "perio_conv_p8_s80_chan1.bin": (201, 129),
    "perio_conv_p8_s80_chan2.bin": (201, 129),
    # Odd-frame-count variant (end=399 -> frameNb=5).
    "perio_odd_chan1.bin": (5, 129),
    # Mel filterbank dumps. spectrum_size 128 -> 129 bins -> 29 mel filters (NOT
    # 26: the triangle walk yields end-beg overlapping filters). log-mel: 201x29;
    # +deltas(3)+dd(3): 201x87 ([static|delta|dd] with static overwritten by
    # delta); non-log mel: 201x29; synth (spectrum_size 49): 20x29.
    "logmel_26_chan1.bin": (201, 29),
    "logmel_deltas_chan1.bin": (201, 87),
    "mel_26_chan1.bin": (201, 29),
    "logmel_synth.bin": (20, 29),
    # DCT / MFCC / deltas / SDC dumps (Task 7). All from the chan-1 log-mel (T=201,
    # F=29) with nb_dct=13. Widths follow getNbDCT (MelFilterBank.h:45-89): plain
    # 13; +deltas(3)+dd(3) 39; +ignoreFirst 38 (drops c0); SDC 13+7*13=104; SDC
    # +ignoreFirst 12+7*13=103 (the SDC col 0 clobbers the last static c12).
    "mfcc_chan1.bin": (201, 13),
    "mfcc_deltas_chan1.bin": (201, 39),
    "mfcc_deltas_if_chan1.bin": (201, 38),
    "mfcc_sdc_chan1.bin": (201, 104),
    "mfcc_sdc_if_chan1.bin": (201, 103),
    # LTSV dumps (Task 8). Column over chan-1 periodogram (T=201, freq_beg=0,
    # freq_end=128, R=15, shift=4); per-frame scores over synth_20x50 (R=3, band
    # 0..49, shift=1). Both single-column (T x 1).
    "ltsv_chan1.bin": (201, 1),
    "ltsv_synth.bin": (20, 1),
    # Task 9 dumps. fmath::log sweep (2 x 20: row0 input, row1 fmath::log(input));
    # TDC score column over the chan-1 excerpt (1 x 201, TdcParams half=128 shift=80
    # min_lag=16 max_lag=128 balance=0.7); pitch scalar over the hand-built middle-1s
    # SPEECH segmentation [0.5s,1.5s) (1 x 1); homothety warp of the chan-1
    # periodogram with coeff = pitch/300 (201 x 129).
    "fmath_log_sweep.bin": (2, 20),
    "tdc_chan1.bin": (1, 201),
    "pitch_chan1.bin": (1, 1),
    "perio_homothety_chan1.bin": (201, 129),
    # Task 10: InputStatistics (real compiled legacy TU). Batch stats over the
    # full synth_20x50 matrix (1x50 mean/std, 1x1 n); merged stats from two
    # batches over row splits [0,7) and [7,20), update()'d together.
    "stats_batch_mean.bin": (1, 50),
    "stats_batch_std.bin": (1, 50),
    "stats_batch_n.bin": (1, 1),
    "stats_merged_mean.bin": (1, 50),
    "stats_merged_std.bin": (1, 50),
    "stats_merged_n.bin": (1, 1),
}

# The DCT matrix product melPeriodogram*_CoeffsDCT is the feature path's one real
# Eigen GEMM. Per the Task 7 brief the Eigen product is replaced by the explicit
# ascending triple loop (the portable/deterministic parity target the Rust
# apply_dct reproduces), so the DCT dumps are produced by the harness's
# applyDCTLoop, NOT the legacy mel.applyDCT. This substitution is LOCKED. The
# harness re-measures the divergence bit-for-bit on every regeneration (see the
# GEMM_CHECK line parsed below) so the numbers here are never a stale, hardcoded
# claim -- they are refreshed every time this script runs.
DCT_GEMM_SUBSTITUTION_TEXT = (
    "product melPeriodogram * _CoeffsDCT (legacy/src/MelFilterBank.cpp:223,267,309,314,356) "
    "replaced with the explicit ascending triple loop (dctProduct in main.cpp) because Eigen's "
    "blocked gebp kernel does not accumulate in plain ascending order; the fields below are "
    "measured fresh by the harness's GEMM_CHECK on every fixture regeneration, not hardcoded."
)

GEMM_CHECK_RE = re.compile(
    r"^GEMM_CHECK diverged=(?P<diverged>\d) mismatches=(?P<mismatches>\d+) total=(?P<total>\d+) "
    r"first=\((?P<row>-?\d+),(?P<col>-?\d+)\) eigen=0x(?P<eigen>[0-9a-f]+) loop=0x(?P<loop>[0-9a-f]+)$",
    re.MULTILINE,
)


def _parse_gemm_check(harness_stdout: str) -> dict[str, object]:
    match = GEMM_CHECK_RE.search(harness_stdout)
    if match is None:
        raise SystemExit("GEMM_CHECK line missing from harness stdout")
    return {
        "text": DCT_GEMM_SUBSTITUTION_TEXT,
        "diverged": bool(int(match["diverged"])),
        "mismatches": int(match["mismatches"]),
        "total": int(match["total"]),
        "first_index": [int(match["row"]), int(match["col"])],
        "eigen_bits": match["eigen"],
        "loop_bits": match["loop"],
    }

# Periodogram framing params + odd-variant end (kept in sync with main.cpp). The
# Rust golden reads perio_odd_end so both sides agree on the odd-frame end value.
PERIODOGRAM = {
    "p": 8,
    "shift": 80,
    "dc_offset": True,
    "window": "hamming_257",
    "temporal_conv": "hann_norm_7",
    "begin": 0,
    "end_full": 16000,
    "perio_odd_end": 399,
    "perio_odd_frame_nb": 5,
}

# Verified against legacy/src/BLSTMSpectralSegmenter.cpp:199-203 (clamp is Max-1)
# with Max=20 from the factory at :598-599. Value 19 (NOT 14); later tasks consume it.
SPECTRUM_ORDER_CLAMP = {
    "value": 19,
    "source": "legacy/src/BLSTMSpectralSegmenter.cpp:199-203 (Max=20 at :598-599)",
}

# Task 9 TDC/pitch/homothety params (kept in sync with main.cpp). Derived inline from
# the brief config (TDC_window=0.032, TDC_shift=0.01, lags=(0.002,0.016), balance=0.7
# @ rate 8000) per TimeDomainCorrel.cpp:100-109. The pitch dump uses a hand-built
# segmentation: a single SPEECH segment covering the middle 1.0s of the excerpt,
# [0.5s, 1.5s) (boundary list Other@0.0, Speech@0.5, Other@1.5, End@duration). The
# TDC score column + pitch both use DC-offset FALSE (a deterministic synthetic
# choice; the harness and Rust port agree on it). homothety coeff = pitch/300.
TDC = {
    "half_window": 128,
    "full_window": 257,
    "shift": 80,
    "min_lag": 16,
    "max_lag": 128,
    "balance": 0.7,
    "window": "hamming_257_param_0.8",
    "dc_offset": False,
    "pitch_segment_sec": [0.5, 1.5],
    "pitch_segment_class": "SPEECH",
    "homothety_coeff": "pitch/300",
}


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout


def _brew_version(dep: str) -> str:
    out = _run(["brew", "list", "--versions", dep]).strip()
    parts = out.split()
    return parts[1] if len(parts) > 1 else "unknown"


def main() -> None:
    # 1. Build the harness.
    _run(["bash", str(HARNESS_DIR / "build.sh")])

    # 2. Run it into the fixture directory.
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    harness_stdout = _run([str(HARNESS_DIR / "oracle_harness"), str(FIXTURE_DIR) + "/"])
    dct_gemm_substitution = _parse_gemm_check(harness_stdout)

    # 3. Read each dump back, verify its shape, and build the inventory.
    inventory: dict[str, dict[str, int]] = {}
    for name, (er, ec) in EXPECTED_SHAPES.items():
        rows, cols, _ = read_bin(FIXTURE_DIR / name)
        if (rows, cols) != (er, ec):
            raise SystemExit(f"{name}: shape ({rows},{cols}) != expected ({er},{ec})")
        inventory[name] = {"rows": rows, "cols": cols}

    # 4. Anchor value: sig_norm[0][0] as a round-trippable hex float.
    _, _, sig_norm = read_bin(FIXTURE_DIR / "sig_norm.bin")
    anchor = {"dump": "sig_norm.bin", "index": [0, 0], "value_hex": float(sig_norm[0]).hex()}

    # 5. Compiler version (same g++ selection as build.sh).
    gxx = _run(["bash", "-c", 'ls "$(brew --prefix)"/bin/g++-* | sort -V | tail -1']).strip()
    compiler = _run([gxx, "--version"]).splitlines()[0].strip()

    manifest = {
        "compiler": compiler,
        "brew_versions": {dep: _brew_version(dep) for dep in BREW_DEPS},
        "flags": FLAGS,
        "constants": CONSTANTS,
        "source_wav": {
            "path": "PRCTS_RUS_RU_0000263489_01.wav",
            "excerpt_frames": [240000, 264000],
            "channels": 2,
            "samplerate": 8000,
            "subtype": "PCM_16",
        },
        "dumps": inventory,
        "anchor": anchor,
        "spectrum_order_clamp": SPECTRUM_ORDER_CLAMP,
        "periodogram": PERIODOGRAM,
        "tdc": TDC,
        "dct_gemm_substitution": dct_gemm_substitution,
    }

    manifest_path = FIXTURE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"OK: {len(inventory)} dumps, manifest -> {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
