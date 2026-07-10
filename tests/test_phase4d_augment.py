"""Phase 4d Task 6: corpus augmentation (`speech.dataprep.augment`).

RNG CONVENTION, NOT PARITY (see the module docstring): the legacy seeds `numpy.random`
off the wall clock in Python 2, so no run -- including this port's -- can be cross-checked
against a live legacy trajectory. What IS pinned here: the injected `Generator`'s draw
order (noise, noisetype, pitch, tempo), the exact sox command strings (incl. the
`str(float)`-vs-`%.3f` split), and the augmented listing's byte layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from speech.dataprep.augment import (
    Variant,
    augment_corpus,
    draw_variant,
    mix_command,
    noise_synth_command,
    pitch_tempo_command,
    probe_command,
    variant_filename,
)
from speech.dataprep.opensad15 import py2_str_float


class RecordingRunner:
    """Test double for `SoxRunner`: records every `run()` call's argv tokens and
    returns pre-scripted `duration()` readings in call order (one per probe -- the
    scratch `noise.wav` file is overwritten on every call, so there is no way to key a
    canned reading by path; see the module docstring's collision-hazard note)."""

    def __init__(self, durations: list[float]) -> None:
        self.commands: list[list[str]] = []
        self._durations = iter(durations)

    def run(self, cmd: list[str]) -> None:
        self.commands.append(cmd)

    def duration(self, path: Path) -> float:
        assert path == Path("noise.wav")
        return next(self._durations)


# --------------------------------------------------------------------------- #
# Pure command-string builders.
# --------------------------------------------------------------------------- #


def test_probe_command_exact_string() -> None:
    assert probe_command("audio/one.wav") == "sox audio/one.wav noise.wav silence -l 1 0.1 1% -1 0.1 1%"


def test_pitch_tempo_command_uses_py2_str_float_not_python3_str() -> None:
    """The task brief's py2-str-float-sensitive case: a pitch value whose Python 3
    `str()` leaks fp noise past 12 significant digits, which Python 2's `str(float)`
    (and this port's `py2_str_float`) truncates."""
    pitch = 16.366300000000003
    tempo = 0.8
    cmd = pitch_tempo_command("audio/one.wav", pitch, tempo)
    assert cmd == "sox audio/one.wav mod.wav pitch 16.3663 tempo 0.8"
    assert str(pitch) not in cmd
    assert py2_str_float(pitch) in cmd


def test_noise_synth_command_exact_string() -> None:
    assert noise_synth_command("pinknoise", 0.010204595606925913) == "sox mod.wav noise.wav synth pinknoise vol 0.0102045956069"


def test_mix_command_exact_string() -> None:
    assert mix_command("audio/one_mod_0.010_pinknoise_-183.611_0.807.wav") == "sox -m mod.wav noise.wav audio/one_mod_0.010_pinknoise_-183.611_0.807.wav"


def test_variant_filename_uses_fixed_point_not_py2_str_float() -> None:
    """The filename's `%.3f` fields are unaffected by the py2/py3 `str(float)` split --
    fixed-precision formatting agrees between Python 2 and 3."""
    variant = Variant(noise=0.0051234, noisetype_str="pinknoise", pitch=-186.7, tempo=0.9123)
    assert variant_filename("corpus/foo.wav", variant) == "corpus/foo_mod_0.005_pinknoise_-186.700_0.912.wav"


# --------------------------------------------------------------------------- #
# draw_variant: RNG draw order + the noisetype ladder.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("seed", "expected_noisetype_str"),
    [
        (3, "whitenoise"),  # 4*rand() == 0.9472... < 1
        (0, "pinknoise"),  # 4*rand() == 1.0791... in [1, 2)
        (4, "tpdfnoise"),  # 4*rand() == 2.0453... in [2, 3)
        (1, "brownnoise"),  # 4*rand() == 3.8019... in [3, 4]
    ],
)
def test_draw_variant_noisetype_ladder(seed: int, expected_noisetype_str: str) -> None:
    """Seeds hand-picked (by brute-force search, not tuned to the implementation) so the
    SECOND draw off `default_rng(seed)` lands in each of the ladder's four buckets."""
    rng = np.random.default_rng(seed)
    variant = draw_variant(rng)
    assert variant.noisetype_str == expected_noisetype_str


def test_draw_variant_order_is_noise_then_noisetype_then_pitch_then_tempo() -> None:
    """Cross-check draw_variant's internal draw order against an independently
    re-derived sequence off a freshly seeded generator with the identical seed."""
    seed = 777
    variant = draw_variant(np.random.default_rng(seed))

    check_rng = np.random.default_rng(seed)
    expected_noise = abs(0.005 * check_rng.standard_normal())
    expected_noisetype = 4 * check_rng.random()
    expected_pitch = 400 * check_rng.random() - 200
    expected_tempo = 0.4 * check_rng.random() + 0.8

    assert variant.noise == expected_noise
    assert variant.pitch == expected_pitch
    assert variant.tempo == expected_tempo
    assert 0 <= expected_noisetype < 4


def test_draw_variant_value_ranges_hold_over_many_seeds() -> None:
    for seed in range(200):
        variant = draw_variant(np.random.default_rng(seed))
        assert variant.noise >= 0.0
        assert -200.0 <= variant.pitch < 200.0
        assert 0.8 <= variant.tempo < 1.2
        assert variant.noisetype_str in {"whitenoise", "pinknoise", "tpdfnoise", "brownnoise"}


# --------------------------------------------------------------------------- #
# augment_corpus: the end-to-end driver.
# --------------------------------------------------------------------------- #


def _expected_variant_block(filename: str, elems_tail: str, seed: int) -> tuple[list[list[str]], list[str]]:
    """Independently re-derive the 5-variant command sequence + listing rows for one
    passing file, off a freshly seeded generator sharing the same seed as the call under
    test -- NOT a call into draw_variant/the command builders, so this is a real
    cross-check of augment_corpus's wiring, not a tautology."""
    rng = np.random.default_rng(seed)
    commands: list[list[str]] = []
    listing_rows: list[str] = []
    for _ in range(5):
        noise = abs(0.005 * rng.standard_normal())
        noisetype = 4 * rng.random()
        if noisetype < 1:
            noisetype_str = "whitenoise"
        elif noisetype < 2:
            noisetype_str = "pinknoise"
        elif noisetype < 3:
            noisetype_str = "tpdfnoise"
        else:
            noisetype_str = "brownnoise"
        pitch = 400 * rng.random() - 200
        tempo = 0.4 * rng.random() + 0.8
        new_filename = f"{filename[:-4]}_mod_{noise:.3f}_{noisetype_str}_{pitch:.3f}_{tempo:.3f}.wav"

        commands.append(f"sox {filename} mod.wav pitch {py2_str_float(pitch)} tempo {py2_str_float(tempo)}".split())
        commands.append(f"sox mod.wav noise.wav synth {noisetype_str} vol {py2_str_float(noise)}".split())
        commands.append(f"sox -m mod.wav noise.wav {new_filename}".split())
        listing_rows.append(f"{new_filename};{elems_tail};")
    return commands, listing_rows


def test_augment_corpus_two_file_listing_one_passes_one_fails_gate(tmp_path: Path) -> None:
    listing = tmp_path / "in.csv"
    line1 = "audio/one.wav;langA;non;1.0;10.0;5.0"
    line2 = "audio/two.wav;langB;non;1.0;8.0;3.0"
    listing.write_text(line1 + "\n" + line2, encoding="ascii")

    runner = RecordingRunner(durations=[2.5, 1.0])  # file 1 passes (>2.0), file 2 fails
    seed = 1234
    rng = np.random.default_rng(seed)

    out_path = augment_corpus(listing, rng, runner, dur_min=2.0)

    assert out_path == tmp_path / "in_addednoise.csv"
    assert out_path.is_file()

    # -- Command vectors --
    assert runner.commands[0] == probe_command("audio/one.wav").split()

    expected_commands, expected_rows = _expected_variant_block("audio/one.wav", "langA;non;1.0;10.0;5.0", seed)
    assert runner.commands[1:16] == expected_commands

    # File 2's probe runs, but the gate fails so no variant commands follow.
    assert runner.commands[16] == probe_command("audio/two.wav").split()
    assert len(runner.commands) == 17

    # -- Output listing bytes --
    expected_text = "\n".join([line1, *expected_rows]) + "\n"
    assert out_path.read_text(encoding="ascii") == expected_text


def test_augment_corpus_dur_min_gate_is_strict_greater_than(tmp_path: Path) -> None:
    """`duration > durMin` (`AugmentCorpus.py:43`) -- exactly-equal duration must fail
    the gate, not pass it."""
    listing = tmp_path / "in.csv"
    listing.write_text("audio/one.wav;langA;non;1.0;10.0;5.0", encoding="ascii")
    runner = RecordingRunner(durations=[2.0])  # exactly dur_min

    out_path = augment_corpus(listing, np.random.default_rng(0), runner, dur_min=2.0)

    assert out_path.read_bytes() == b""
    assert len(runner.commands) == 1  # only the probe; no variant commands


def test_augment_corpus_default_dur_min_is_two_seconds(tmp_path: Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text("audio/one.wav;langA;non;1.0;10.0;5.0", encoding="ascii")
    runner = RecordingRunner(durations=[2.0])

    out_path = augment_corpus(listing, np.random.default_rng(0), runner)  # dur_min omitted

    assert out_path.read_bytes() == b""  # 2.0 is not > 2.0 (the default)


def test_augment_corpus_original_line_written_verbatim_before_variants(tmp_path: Path) -> None:
    """The augmented listing echoes the RAW original line (`AugmentCorpus.py:44`), not a
    field reconstructed from `elems` -- exercised with trailing-whitespace-sensitive
    content to prove it is copied untouched."""
    listing = tmp_path / "in.csv"
    raw_line = "audio/one.wav;langA;non;1.0;10.0;5.0"
    listing.write_text(raw_line, encoding="ascii")
    runner = RecordingRunner(durations=[3.0])

    out_path = augment_corpus(listing, np.random.default_rng(0), runner, dur_min=2.0)

    first_line = out_path.read_bytes().split(b"\n", 1)[0].decode("ascii")
    assert first_line == raw_line


def test_augment_corpus_short_line_raises_indexerror_on_gate_pass(tmp_path: Path) -> None:
    """No guard on `elems[1..5]` (`AugmentCorpus.py:69`) -- a passing-gate line with
    fewer than 6 `;`-fields raises `IndexError`, matching the legacy's own unguarded
    indexing."""
    listing = tmp_path / "in.csv"
    listing.write_text("audio/one.wav;langA", encoding="ascii")  # only 2 fields
    runner = RecordingRunner(durations=[3.0])  # passes the gate

    with pytest.raises(IndexError):
        augment_corpus(listing, np.random.default_rng(0), runner, dur_min=2.0)


def test_augment_corpus_out_listing_path_for_csv_extension(tmp_path: Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text("audio/one.wav;langA;non;1.0;10.0;5.0", encoding="ascii")
    runner = RecordingRunner(durations=[1.0])

    out_path = augment_corpus(listing, np.random.default_rng(0), runner, dur_min=2.0)

    assert out_path == tmp_path / "in_addednoise.csv"


def test_augment_corpus_out_listing_path_slices_last_four_chars_not_extension_aware(tmp_path: Path) -> None:
    """`listing[:-4]+"_addednoise.csv"` (`:33`) is a raw string slice on the FULL path,
    not an extension-aware split -- a 5-char extension (`.flst`) loses only its last 4
    chars, leaving the dot behind (`"in.flst"[:-4]` == `"in."`), NOT the extension-free
    stem a naive reader might expect. Reproduced verbatim, matching the legacy's own
    `.csv`-only assumption (see the parallel quirk in `opensad15.py`'s `[:-5]` strip)."""
    listing = tmp_path / "in.flst"
    listing.write_text("audio/one.wav;langA;non;1.0;10.0;5.0", encoding="ascii")
    runner = RecordingRunner(durations=[1.0])

    out_path = augment_corpus(listing, np.random.default_rng(0), runner, dur_min=2.0)

    assert out_path == tmp_path / "in._addednoise.csv"
