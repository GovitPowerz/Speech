"""Phase 5 Task 8: the modern training loop pyo3 SMOKE.

Exercises the ENGINE-BACKED default hooks of `train_modern` end to end on a tiny committed
corpus: from-scratch seeded init (`init_weights`) -> `steps_per_epoch` SMORMS3 steps over
`forward_backward` (backprop ON) -> per-epoch FORWARD-ONLY validation (a backprop-OFF
engine, `compute_cost` on the results) -> best/last checkpoints. 2 epochs, no early stop.
The unit tests (`tests/test_phase5_train_modern.py`) cover the state machine against stubs;
this proves the real seam runs (init pack sizes match the engine, the backprop-off
validation config yields a finite cost, both checkpoints land).

Marked `slow` + pyo3 (module-level importorskip); runs in the CI python-pyo3 job.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from speech.drivers.init import init_run
from speech.drivers.state import ModernTrainParams
from speech.drivers.train import train_modern

pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"


def _seed_tier2_spectral(dst: Path) -> Path:
    """The Algo-3 (single-net SAD) 2-file spectral corpus + config + the phase0 weight pack
    the config's `BLSTM_weightsFile` resolves to (loaded at engine construction, then
    OVERRIDDEN by the from-scratch init). Mirrors `test_exit_gate::_seed_tier2_spectral`."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in ("f1", "f2"):
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, dst / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", dst / "NNweights_config1.bin")
    (dst / "vrcts_tier2").mkdir()
    return dst / "tier2_spectral.config"


def _seed_twin_train(dst: Path) -> Path:
    """The Mode-7 2-file phSeq Twin corpus (Algo 6) + config + both TINY seed packs (loaded
    at construction, overridden by init). Mirrors `test_exit_gate::_seed_twin_train`."""
    for f in ("twin_train.config", "languagemapping_lid7.csv", "tiny_sad_seed.bin", "tiny_lid_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    phseq = dst / "corpus_phseq"
    phseq.mkdir(parents=True, exist_ok=True)
    for f in ("s1.phSeq", "s2.phSeq", "s1.stm", "s2.stm", "listing_train.csv"):
        shutil.copy(PHASE4B / "corpus_phseq" / f, phseq / f)
    return dst / "twin_train.config"


@pytest.mark.slow
def test_train_modern_algo3_smoke(tmp_path: Path) -> None:
    config = _seed_tier2_spectral(tmp_path)
    state = init_run(config, tmp_path / "run")
    assert state.algo == 3 and state.ps.lid is None

    params = ModernTrainParams(epochs=2, patience=99, steps_per_epoch=2, init_scheme="xavier", init_seed=7)
    res = train_modern(state, seed=0, params=params)

    assert res.epochs_run == 2
    assert len(res.history) == 2
    for r in res.history:
        assert np.isfinite(r.val_cost), f"forward-only validation cost must be finite: {r}"
        assert np.isfinite(r.train_cost), f"train cost must be finite: {r}"
        assert r.confusion_error is None  # single-net SAD: no LID confusion

    ckpt = Path(res.checkpoint_dir)
    for name in ("best_sad.bin", "last_sad.bin", "train_history.json"):
        assert (ckpt / name).is_file(), f"missing checkpoint artifact {name}"
    assert not (ckpt / "best_lid.bin").exists(), "single-net run must not write a LID pack"
    # forward-only validation actually ran (a backprop-OFF config was written).
    assert (Path(state.config_path).parent / "_valid_modern.config").is_file()


@pytest.mark.slow
def test_train_modern_twin_smoke_writes_both_nets(tmp_path: Path) -> None:
    config = _seed_twin_train(tmp_path)
    state = init_run(config, tmp_path / "run")
    assert state.algo == 6 and state.ps.lid is not None

    params = ModernTrainParams(epochs=2, patience=99, steps_per_epoch=2, init_scheme="he", init_seed=3)
    res = train_modern(state, seed=0, params=params)

    assert res.epochs_run == 2
    for r in res.history:
        assert np.isfinite(r.val_cost), f"twin validation cost must be finite: {r}"
    # both nets checkpointed (best + last).
    ckpt = Path(res.checkpoint_dir)
    for name in ("best_sad.bin", "best_lid.bin", "last_sad.bin", "last_lid.bin"):
        assert (ckpt / name).is_file(), f"twin run must checkpoint {name}"


@pytest.mark.slow
def test_train_modern_resume_smoke(tmp_path: Path) -> None:
    # Run 1 epoch, then resume IN PLACE toward 2 -- the resumed run loads last_*.bin +
    # train_history.json and continues, ending with 2 epochs of history.
    config = _seed_tier2_spectral(tmp_path)
    state = init_run(config, tmp_path / "run")
    train_modern(state, seed=0, params=ModernTrainParams(epochs=1, patience=99, steps_per_epoch=2, init_seed=7))
    ckpt = str(Path(state.out_dir) / "checkpoint")
    res = train_modern(state, seed=0, params=ModernTrainParams(epochs=2, patience=99, steps_per_epoch=2, resume_from=ckpt))
    assert res.epochs_run == 2
    assert [r.epoch for r in res.history] == [0, 1]
