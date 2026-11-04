"""`Engine.channel_results()` cross-pinned against the wire (issue #23).

The seam's named view and the `results_matrix()` wire view are two renderings of the same
`ChannelResult` values. `tests/_result_rows.py` is the only Python decoder of the wire
layout (for the Octave goldens, which the pure-Python CI job compares without `speech_rs`);
this test is what keeps that decoder honest against the Rust writer, on the two committed
fixtures `test_seam_replay.py` already drives: `tier2_spectral` (algo 3, 18-wide rows) and
`twin_train` (algo 6, 18 + N columns, in-band targets)."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from speech.engine import ChannelResults
from speech.evaluate import hyp_intervals, load_vrcts_hyp

from tests._result_rows import channel_results_from_matrix

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"


@contextmanager
def chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _seed_tier2_spectral(dst: Path) -> None:
    """Mirrors `test_seam_replay._seed_tier2_spectral`: the 2-file algo-3 corpus."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in ("f1", "f2"):
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, dst / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", dst / "NNweights_config1.bin")
    (dst / "vrcts_tier2").mkdir()


def _seed_twin_train(dst: Path) -> None:
    """Mirrors `test_seam_replay._seed_twin_train`: the Mode-7 2-file phSeq Twin corpus."""
    for f in ("twin_train.config", "languagemapping_lid7.csv", "tiny_sad_seed.bin", "tiny_lid_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    phseq = dst / "corpus_phseq"
    phseq.mkdir(parents=True, exist_ok=True)
    for f in ("s1.phSeq", "s2.phSeq", "s1.stm", "s2.stm", "listing_train.csv"):
        shutil.copy(PHASE4B / "corpus_phseq" / f, phseq / f)


def _run(tmp_path: Path, seed: Callable[[Path], None], config: str) -> tuple[ChannelResults, ChannelResults, NDArray[np.float64]]:
    seed(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine([config], "-m")
        assert len(eng.channel_results()["file"]) == 0, "empty before the first run()"
        eng.run()
        named = ChannelResults.from_seam(eng.channel_results())
        mcr = np.asarray(eng.results_matrix(), dtype=np.float64)
    return named, channel_results_from_matrix(mcr), mcr


def _assert_same(named: ChannelResults, decoded: ChannelResults) -> None:
    """The WIRE fields: the matrix cannot carry `hypothesis`, whose pin is the dump below."""
    for f in fields(ChannelResults):
        if f.name == "hypothesis":
            continue
        a, b = getattr(named, f.name), getattr(decoded, f.name)
        assert a.dtype == b.dtype, f"{f.name}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{f.name}: shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b), f"{f.name}: seam {a!r} != decoded wire {b!r}"


def test_tier2_spectral_names_match_the_wire(tmp_path: Path) -> None:
    named, decoded, _mcr = _run(tmp_path, _seed_tier2_spectral, "tier2_spectral.config")
    _assert_same(named, decoded)
    assert len(named) == 4, "2 files x 2 channels"
    assert sorted(set(named.file.tolist())) == [0, 1] and set(named.conf.tolist()) == {0}
    assert named.lid_scores.shape == (4, 0) and set(named.lid_target.tolist()) == {-1}
    assert np.all(named.seg_count > 0), "the scored SAD rows carry a live counter"


def test_tier2_spectral_hypothesis_reads_as_the_dumped_vrcts(tmp_path: Path) -> None:
    """The two doors of the held-out scorer (issue #86): per row, `hyp_intervals` over the
    seam's `hypothesis` equals `load_vrcts_hyp` over the xml the same run dumped
    (`Dump_Directory vrcts_tier2`, the `_chan_<n>` fan-out), bit for bit. The audio cap is
    overlaid to a value with more than two decimals at which every decision is a speech run
    then an `Other` tail (probed: 1.2345 s decides all-speech, 2.4321 s all-other), so the
    `sigdur` sliver exists on the xml door, a dropped rounding is caught, and an `Other` row
    counted as speech is caught."""
    _seed_tier2_spectral(tmp_path)
    cfg = speech_rs.parse_legacy_config((tmp_path / "tier2_spectral.config").read_text())
    cfg["Audio_max_duration"] = "1.7654"
    with chdir(tmp_path):
        eng = speech_rs.Engine.from_map([cfg], "-m")
        eng.run()
        named = ChannelResults.from_seam(eng.channel_results())
    assert len(named) == len(named.hypothesis) == 4
    for i in range(len(named)):
        duration, rows = named.hypothesis[i]
        assert duration == named.audio_duration[i] and round(duration, 2) != duration, "the cap must leave a sigdur sliver"
        assert rows[0][0] == 0.0 and rows[-1] == (duration, "End")
        xml = tmp_path / "vrcts_tier2" / f"f{named.file[i] + 1}_chan_{named.chan[i] + 1}.xml"
        assert hyp_intervals(duration, rows) == load_vrcts_hyp(xml), f"row {i}: the two doors differ"
    assert all([c for _, c in rows] == ["Speech", "Other", "End"] for _, rows in named.hypothesis), "the cap must decide speech then an Other tail"


def test_twin_train_names_match_the_wire(tmp_path: Path) -> None:
    named, decoded, mcr = _run(tmp_path, _seed_twin_train, "twin_train.config")
    _assert_same(named, decoded)
    # N from an independent source: the class ids of the fixture's language mapping.
    mapping = (PHASE4B / "languagemapping_lid7.csv").read_text().splitlines()
    n_classes = len({line.split(";")[2] for line in mapping if line.strip()})
    assert n_classes >= 2 and mcr.shape[1] == 3 + 18 + n_classes
    assert named.lid_scores.shape[1] == n_classes
    assert np.all(named.lid_target >= 0), "every twin_train row carries an in-band target"
    assert np.all(named.lid_scores <= 150.0), "the target column is decoded, nothing is left in-band"
    # The decoded target re-encodes bit-exactly: the wire holds `score + 200` in the target
    # column (3 id columns, then result column 16 + target), and nothing else is in-band.
    for i, t in enumerate(named.lid_target.tolist()):
        wire_row = mcr[i, 3 + 16 : 3 + 16 + n_classes]
        assert wire_row[t] == named.lid_scores[i, t] + 200.0
        assert np.sum(wire_row > 150.0) == 1
