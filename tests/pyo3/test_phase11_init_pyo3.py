"""Phase 11 Task 3: `init_transformer_flat` -> `Engine.set_weights` -> `Engine.weights()`.

THE ONLY CROSS-LANGUAGE LENGTH PIN there is. `tests/test_phase11_init.py` proves the Python
builder agrees with an independent transcription of `transformer.rs::for_each_slot`; this file
proves it agrees with the ACTUAL Rust walk, by handing the pack to an Engine built from a
config that selects the cell and reading it back.

Both failure directions are covered by the round trip, which is why it is worth its runtime:
`BlstmNetwork::set_weights` BAILS on a pack SHORTER than `nb_of_weights()` (so a too-small
Python length raises here), and silently ignores an over-long tail (so a too-large one is
caught by the SHAPE of what `weights()` returns, which the Rust side sizes).

Lives under `tests/pyo3/` for the phase-5 reason, unchanged: the dedicated `python-pyo3` CI job
runs `pytest tests/pyo3`, so a `speech_rs`-touching test placed anywhere else import-skips in
both jobs forever.
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

# `tier2_spectral.config` carries the v1 SAD topology exactly (LSTMNeuronNb 23,24,24 /
# SubSampling 4,1 / MLP 48,12,1 / NNetInputSize 23), so at the DEFAULT geometry its transformer
# pack is the v1 closed form `16199 + 196*d_ff` evaluated at 64 -- i.e. this file also measures
# the sizing arithmetic through the engine rather than only through Python.
V1_DEFAULT_PACK = 28743


@contextmanager
def chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _seed_tier2_transformer(dst: Path, extra: dict[str, str]) -> dict[str, str]:
    """Mirrors `test_phase5_init_weights_pyo3.py::_seed_tier2_spectral`, then APPENDS the cell
    keys (last-wins parsing, so appending is the documented way to overlay). Returns the parsed
    flat config, which is what `nnet_spec` reads -- so Python and the engine are seeded from
    the SAME text, not from two hand-kept copies."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in ("f1", "f2"):
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, dst / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", dst / "NNweights_config1.bin")
    (dst / "vrcts_tier2").mkdir()

    cfg_path = dst / "tier2_spectral.config"
    lines = [f"{k} {v}" for k, v in {"BLSTM_Cell_Type": "transformer", **extra}.items()]
    cfg_path.write_text(cfg_path.read_text() + "\n" + "\n".join(lines) + "\n")
    return config_bridge.parse_legacy_config(cfg_path.read_text())


@pytest.mark.parametrize("scheme", ["xavier", "he"])
def test_transformer_pack_round_trips_through_engine_set_weights(tmp_path: Path, scheme: str) -> None:
    """The default geometry: the Python pack length IS `nb_of_weights()` and the bytes survive
    the crossing untouched (a pure copy in, a pure copy out -- no arithmetic on the PyO3
    boundary)."""
    cfg = _seed_tier2_transformer(tmp_path, {})
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    assert spec["CellType"] == "transformer"

    packs = init_weights(spec, np.random.default_rng(123), scheme=scheme)  # type: ignore[arg-type]
    assert len(packs) == 1
    assert packs[0].shape == (V1_DEFAULT_PACK,)  # the v1 closed form at d_ff = 64.

    with chdir(tmp_path):
        eng = speech_rs.Engine(["tier2_spectral.config"], "-m")
        eng.set_weights(0, packs)
        got = eng.weights(0)

    assert len(got) == 1
    assert got[0].shape == packs[0].shape, "the Rust net's nb_of_weights disagrees with the Python builder"
    assert np.array_equal(got[0].view(np.uint64), packs[0].view(np.uint64)), "set_weights -> weights must be a bit-exact round trip"


@pytest.mark.parametrize(("keys", "d_ff"), [({"Transformer_D_Ff": "8"}, 8), ({"Transformer_D_Ff": "40", "Transformer_Heads": "8"}, 40)])
def test_the_geometry_keys_resize_both_sides_together(tmp_path: Path, keys: dict[str, str], d_ff: int) -> None:
    """A NON-DEFAULT `d_ff` must move the Rust net and the Python pack by the SAME amount --
    the pin that would catch a mirrored-constant drift or a key one side reads and the other
    ignores. `Transformer_Heads 8` rides along on the second row: it changes the attention
    arithmetic and must NOT change any length (24 % 8 == 0, so the engine accepts it)."""
    cfg = _seed_tier2_transformer(tmp_path, keys)
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    packs = init_weights(spec, np.random.default_rng(7))
    assert packs[0].shape == (16199 + 196 * d_ff,)

    with chdir(tmp_path):
        eng = speech_rs.Engine(["tier2_spectral.config"], "-m")
        eng.set_weights(0, packs)
        got = eng.weights(0)

    assert got[0].shape == packs[0].shape
    assert np.array_equal(got[0].view(np.uint64), packs[0].view(np.uint64))
