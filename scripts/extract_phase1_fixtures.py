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
    _run([str(HARNESS_DIR / "oracle_harness"), str(FIXTURE_DIR) + "/"])

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
    }

    manifest_path = FIXTURE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"OK: {len(inventory)} dumps, manifest -> {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
