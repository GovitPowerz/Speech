"""Phase 5 Task 3: `init_weights` -> `Engine.set_weights` -> `Engine.weights()` round trip.

Lives under `tests/pyo3/` (not alongside the rest of the Task 3 suite in
`tests/test_phase5_init_weights.py`) because the dedicated `python-pyo3` CI job runs only
`pytest tests/pyo3`: every existing `importorskip("speech_rs")` test in this repo already
lives here (`test_engine_smoke.py` / `test_seam_replay.py` / `test_exit_gate.py`), so
putting a `speech_rs`-touching test here is what makes it actually execute in CI instead
of import-skipping in both jobs forever.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from speech import config_bridge
from speech.init_weights import init_weights

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"


@contextmanager
def chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _seed_tier2_spectral(dst: Path) -> None:
    """Mirrors `tests/pyo3/test_seam_replay.py::_seed_tier2_spectral`: the 2-file corpus +
    config + the phase0 weight pack the config's relative `BLSTM_weightsFile` resolves to
    (its content is irrelevant here -- `set_weights` overwrites it before any `run()`),
    plus the `Dump_Directory` the config points VRCTS writes at."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in ("f1", "f2"):
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, dst / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", dst / "NNweights_config1.bin")
    (dst / "vrcts_tier2").mkdir()


@pytest.mark.parametrize("scheme", ["xavier", "he"])
def test_init_weights_round_trips_through_engine_set_weights(tmp_path: Path, scheme: str) -> None:
    """`init_weights` -> `Engine.set_weights(0, packs)` -> `Engine.weights(0)` must be a
    bit-exact round trip (a pure copy in, a pure copy out -- no arithmetic on the PyO3
    boundary; mirrors `test_seam_replay.py::test_set_weights_roundtrip`'s hand-crafted-
    pattern version, but exercising a real `init_weights` pack for both schemes)."""
    _seed_tier2_spectral(tmp_path)
    cfg = config_bridge.parse_legacy_config((tmp_path / "tier2_spectral.config").read_text())
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")

    packs = init_weights(spec, np.random.default_rng(123), scheme=scheme)  # type: ignore[arg-type]
    assert len(packs) == 1
    assert packs[0].shape == (33671,)

    with chdir(tmp_path):
        eng = speech_rs.Engine(["tier2_spectral.config"], "-m")
        eng.set_weights(0, packs)
        got = eng.weights(0)

    assert len(got) == 1
    assert np.array_equal(got[0].view(np.uint64), packs[0].view(np.uint64)), "set_weights -> weights must be a bit-exact round trip"


def test_init_weights_pack_is_the_engine_length_at_output_subsampling_2(tmp_path: Path) -> None:
    """#61: at `BLSTM_OutputSubSampling 2,1` output layer 0 reads 48*2 inputs, so the net
    takes 33671 + 12*48 = 34247 weights and `set_weights` accepts nothing else."""
    _seed_tier2_spectral(tmp_path)
    cfg = config_bridge.parse_legacy_config((tmp_path / "tier2_spectral.config").read_text())
    cfg["BLSTM_OutputSubSampling"] = "2,1"
    cfg["BLSTM_weightsFile"] = ""  # the committed 33671 pack is short for this net.
    packs = init_weights(config_bridge.nnet_spec(cfg, prefix="BLSTM"), np.random.default_rng(123))
    assert packs[0].shape == (34247,)

    with chdir(tmp_path):
        eng = speech_rs.Engine.from_map([cfg], "-m")
        eng.set_weights(0, packs)
        got = eng.weights(0)

    assert np.array_equal(got[0].view(np.uint64), packs[0].view(np.uint64))
