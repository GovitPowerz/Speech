"""Audio corpus augmentation. Ported from legacy `AugmentCorpus.py` (Python 2, 73 LOC,
vendored read-only at `legacy/.../AugmentCorpus.py`).

Per listing line (`;`-separated, `elems[0]` the audio filename): first a silence-strip
DURATION PROBE (`:39-41`) -- `sox <file> noise.wav silence -l 1 0.1 1% -1 0.1 1%` then
`soxi -D noise.wav` -- gates on the SILENCE-STRIPPED duration (not the raw file duration)
against `durMin` (2.0, `:43`). Only on a pass: the original line is echoed to the
augmented listing, then 5 noise/pitch/tempo variants are drawn and rendered via 3 more
sox invocations each (`:58-68`), and a new listing row is appended (`:69`).

RNG CONVENTION, NOT PARITY: the legacy draws from `numpy.random` seeded off the wall
clock (Python 2, `:1-8`) -- no run is reproducible even against itself, let alone
cross-checkable from this environment (no Python 2 interpreter exists here; same
no-live-oracle situation as `opensad15.py`). This port injects a `numpy.random.Generator`
instead and preserves the legacy's DRAW ORDER verbatim (noise, noisetype, pitch, tempo --
`:46-57`) as a documented convention so a seeded run is at least reproducible in THIS
port, not as a bit-exact-vs-legacy claim (none is possible). See IMPROVEMENTS.md.

SCRATCH-FILE COLLISION HAZARD: `noise.wav`/`mod.wav` are HARD-CODED relative filenames
(`:39,58,60,63,66`), written and read in the current working directory across every sox
invocation for every file/variant. The legacy relies on running strictly SEQUENTIALLY in
one process/one cwd for this to be safe; nothing here changes that assumption (no unique
temp names, no scratch directory injection) -- reproduced as-is. See IMPROVEMENTS.md.

`str(pitch)`/`str(tempo)`/`str(noise)` inside the sox COMMAND strings (`:60,63`) are
CPython 2's `str(float)` (`%.12g`-equivalent), NOT Python 3's shortest-round-trip
`repr`-backed `str` -- reused from `speech.dataprep.opensad15.py2_str_float` (Task 5).
The FILENAME's `%.3f` fields (`:58`) are unaffected by the py2/py3 divergence: fixed-
precision `%f`/`.Nf` formatting agrees between Python 2 and 3 (only `str`/`repr`-style
shortest-round-trip formatting diverges), so plain `f"{x:.3f}"` is used there.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from speech.dataprep.opensad15 import py2_str_float

# Hard-coded scratch filenames, matching the legacy verbatim (see the module docstring's
# collision-hazard note): both are read/written relative to the current working directory.
_NOISE_SCRATCH = "noise.wav"
_MOD_SCRATCH = "mod.wav"

_DUR_MIN_DEFAULT = 2.0


class SoxRunner(Protocol):
    """Injection seam standing in for the legacy's `os.system(command)` shell-outs
    (`:40,62,65,68`) plus the `soxi -D` duration probe (`:41`)."""

    def run(self, cmd: list[str]) -> None:
        """Execute one sox command, given as its whitespace-split argv tokens."""
        ...

    def duration(self, path: Path) -> float:
        """Return `soxi -D <path>`'s duration reading, in seconds."""
        ...


class RealSoxRunner:
    """Default `SoxRunner`: executes real subprocesses.

    Port-side hardening, NOT a parity requirement (documented per the task brief): the
    legacy shells out via `os.system(command)` on a single space-joined string, which
    goes through `/bin/sh`. This default implementation instead runs the SAME tokens
    (`augment_corpus` already `.split()`s the pinned command string before calling
    `run`) via `subprocess.run(cmd, shell=False)` -- no shell in the loop. None of the
    pinned command strings ever contain shell metacharacters or quoting, so this is
    observably equivalent for well-formed inputs; unlike `os.system`, a missing
    `sox`/`soxi` binary raises `FileNotFoundError` instead of a shell error swallowed
    into a nonzero (and, for `os.system`, entirely ignored) exit status.
    """

    def run(self, cmd: list[str]) -> None:
        subprocess.run(cmd, check=False)

    def duration(self, path: Path) -> float:
        out = subprocess.check_output(["soxi", "-D", str(path)])
        return float(out)


@dataclass(frozen=True)
class Variant:
    """One augmentation variant's drawn parameters (`AugmentCorpus.py:46-57`)."""

    noise: float
    noisetype_str: str
    pitch: float
    tempo: float


def draw_variant(rng: np.random.Generator) -> Variant:
    """Port of `AugmentCorpus.py:46-57`. Draw order preserved verbatim: noise, noisetype,
    pitch, tempo -- see the module docstring's RNG-convention note."""
    noise = abs(0.005 * rng.standard_normal())
    noisetype = 4 * rng.random()
    if noisetype < 1:
        noisetype_str = "whitenoise"
    elif noisetype < 2:
        noisetype_str = "pinknoise"
    elif noisetype < 3:
        noisetype_str = "tpdfnoise"
    else:
        # Legacy's fourth rung is `elif (noisetype <= 4)`; unreachable-as-anything-else
        # here since rng.random() is half-open [0, 1) so `4*rng.random()` never reaches
        # 4.0 -- a plain `else` is behaviorally identical and keeps this total.
        noisetype_str = "brownnoise"
    pitch = 400 * rng.random() - 200
    tempo = 0.4 * rng.random() + 0.8
    return Variant(noise=noise, noisetype_str=noisetype_str, pitch=pitch, tempo=tempo)


def variant_filename(filename: str, variant: Variant) -> str:
    """Port of `AugmentCorpus.py:58`: `%.3f`-formatted noise/pitch/tempo (fixed
    precision -- no py2/py3 `str(float)` divergence here, unlike the sox commands
    below)."""
    return f"{filename[:-4]}_mod_{variant.noise:.3f}_{variant.noisetype_str}_{variant.pitch:.3f}_{variant.tempo:.3f}.wav"


def probe_command(filename: str) -> str:
    """Port of `AugmentCorpus.py:39`: the silence-strip duration probe, writing to the
    hard-coded `noise.wav` scratch file (see the module docstring)."""
    return f"sox {filename} {_NOISE_SCRATCH} silence -l 1 0.1 1% -1 0.1 1%"


def pitch_tempo_command(filename: str, pitch: float, tempo: float) -> str:
    """Port of `AugmentCorpus.py:60`. `pitch`/`tempo` are `str(float)`-formatted
    (CPython 2 semantics, via `py2_str_float`) -- NOT the `%.3f` used for the filename."""
    return f"sox {filename} {_MOD_SCRATCH} pitch {py2_str_float(pitch)} tempo {py2_str_float(tempo)}"


def noise_synth_command(noisetype_str: str, noise: float) -> str:
    """Port of `AugmentCorpus.py:63`. `noise` is `str(float)`-formatted (py2 semantics)."""
    return f"sox {_MOD_SCRATCH} {_NOISE_SCRATCH} synth {noisetype_str} vol {py2_str_float(noise)}"


def mix_command(new_filename: str) -> str:
    """Port of `AugmentCorpus.py:66`: mix the pitched/tempo-shifted signal with the
    synthesized noise into the final variant file."""
    return f"sox -m {_MOD_SCRATCH} {_NOISE_SCRATCH} {new_filename}"


def augment_corpus(
    listing: Path,
    rng: np.random.Generator,
    runner: SoxRunner,
    dur_min: float = _DUR_MIN_DEFAULT,
) -> Path:
    """Port of `AugmentCorpus.py`'s per-listing loop (`:31-70`).

    Reads `listing` (one `;`-separated row per line, `elems[0]` the audio filename),
    probes each file's silence-stripped duration, and for every file whose probed
    duration exceeds `dur_min` (`2.0` in the legacy, `:29,43`): echoes the ORIGINAL line
    to the augmented listing, then emits 5 noise/pitch/tempo variants (3 sox commands
    each) and appends one listing row per variant (`filename[1:6]` echoed verbatim with a
    trailing `;`, `:69`).

    Writes `<listing's path with its last 4 chars replaced>_addednoise.csv` (`:33`,
    matching the legacy's raw string-slice, not an extension-aware split) and returns
    that path. No guard against short/empty lines (`elems[0]`/`elems[1..5]` index
    directly, matching the legacy's unguarded `line.split(";")` -- a malformed row raises
    `IndexError`, same as the legacy would fail its own indexing).
    """
    out_listing = Path(str(listing)[:-4] + "_addednoise.csv")
    lines = listing.read_text(encoding="ascii").splitlines()

    with out_listing.open("wb") as out_f:
        for line in lines:
            elems = line.split(";")
            filename = elems[0]

            runner.run(probe_command(filename).split())
            duration = runner.duration(Path(_NOISE_SCRATCH))

            if duration > dur_min:
                out_f.write((line + "\n").encode("ascii"))
                for _ in range(5):
                    variant = draw_variant(rng)
                    new_filename = variant_filename(filename, variant)

                    runner.run(pitch_tempo_command(filename, variant.pitch, variant.tempo).split())
                    runner.run(noise_synth_command(variant.noisetype_str, variant.noise).split())
                    runner.run(mix_command(new_filename).split())

                    listing_row = f"{new_filename};{elems[1]};{elems[2]};{elems[3]};{elems[4]};{elems[5]};\n"
                    out_f.write(listing_row.encode("ascii"))

    return out_listing
