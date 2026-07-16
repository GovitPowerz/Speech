"""Phase 5 Task 8: the modern training loop STATE MACHINE (stub-engine unit tests).

These pin the epoch / validation / early-stop / best-checkpoint / resume logic of
`train_modern` WITHOUT any engine: the `train_epoch`/`validate` hooks are stubs and
`init_weights_override` injects a tiny controllable weight, so the whole loop runs in
pure Python (no pyo3, no config-derived init). The engine-backed DEFAULT hooks are
exercised separately by the pyo3 smoke (`tests/pyo3/test_phase5_train_modern_smoke.py`).

Stub convention: `train_epoch` returns weights that ENCODE the epoch (`[float(epoch)]`)
so `best_*.bin`/`last_*.bin` are byte-predictable, and `validate` returns a crafted
per-epoch cost sequence so patience/best-selection land at known epochs. The resume test
uses a weight-CARRYING stub (`weights + 1`) so continuation genuinely depends on reloading
`last_*.bin`.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from speech.drivers.init import init_run
from speech.drivers.state import ModernTrainParams, RunState
from speech.drivers.train import train_modern
from speech.weight_bridge import read_bin

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"


def _algo3_state(tmp: Path) -> RunState:
    """A real single-net (Algo 3) RunState via `init_run` (no pyo3): the committed spectral
    config + its listing/mapping parsed into a typed state. The stub hooks + init override
    mean only the config PARSE is exercised, never the engine."""
    tmp.mkdir(parents=True, exist_ok=True)
    for f in ("tier2_spectral.config", "language2classmapping.csv", "tier2_fileslisting.csv"):
        shutil.copy(PHASE4A / f, tmp / f)
    return init_run(tmp / "tier2_spectral.config", tmp / "run")


def _epoch_encoding_stub() -> Callable[[list[np.ndarray], int], tuple[list[np.ndarray], float]]:
    """train hook: weights ENCODE the epoch (ignores input) so checkpoints are predictable."""

    def train_epoch(weights: list[np.ndarray], epoch: int) -> tuple[list[np.ndarray], float]:
        return [np.array([float(epoch)], dtype=np.float64)], float(epoch)

    return train_epoch


def _val_stub(val_seq: list[float]) -> Callable[[list[np.ndarray], int], tuple[float, float | None]]:
    def validate(weights: list[np.ndarray], epoch: int) -> tuple[float, float | None]:
        return float(val_seq[epoch]), None

    return validate


def _ckpt_vec(ckpt_dir: str, name: str) -> list[float]:
    return [float(x) for x in read_bin(Path(ckpt_dir) / name)[2]]


def test_early_stop_patience_triggers_at_crafted_epoch(tmp_path: Path) -> None:
    state = _algo3_state(tmp_path)
    # val falls to its min at epoch 2, then rises; patience=2 -> stop after epoch 4
    # (epoch 4 - best_epoch 2 == 2). epochs 0..4 run, epochs 5,6 never do.
    val_seq = [5.0, 4.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    res = train_modern(
        state,
        seed=0,
        params=ModernTrainParams(epochs=7, patience=2),
        train_epoch=_epoch_encoding_stub(),
        validate=_val_stub(val_seq),
        init_weights_override=[np.array([0.0])],
    )
    assert res.stopped_early is True
    assert res.best_epoch == 2
    assert res.best_val_cost == 3.0
    assert res.epochs_run == 5
    assert [r.epoch for r in res.history] == [0, 1, 2, 3, 4]
    assert res.history[2].is_best is True
    assert res.history[3].is_best is False
    assert res.history[4].val_cost == 5.0
    assert res.history[0].confusion_error is None  # single-net: no LID confusion recorded
    # best checkpoint holds epoch-2 weights, last holds epoch-4 weights.
    assert _ckpt_vec(res.checkpoint_dir, "best_sad.bin") == [2.0]
    assert _ckpt_vec(res.checkpoint_dir, "last_sad.bin") == [4.0]
    # single-net run must not write a LID pack.
    assert not (Path(res.checkpoint_dir) / "best_lid.bin").exists()


def test_no_early_stop_runs_every_epoch(tmp_path: Path) -> None:
    state = _algo3_state(tmp_path)
    val_seq = [5.0, 4.0, 3.0, 2.0]  # monotonically improving -> best updates every epoch
    res = train_modern(
        state,
        seed=0,
        params=ModernTrainParams(epochs=4, patience=2),
        train_epoch=_epoch_encoding_stub(),
        validate=_val_stub(val_seq),
        init_weights_override=[np.array([0.0])],
    )
    assert res.stopped_early is False
    assert res.epochs_run == 4
    assert res.best_epoch == 3
    assert res.best_val_cost == 2.0
    assert _ckpt_vec(res.checkpoint_dir, "best_sad.bin") == [3.0]
    assert _ckpt_vec(res.checkpoint_dir, "last_sad.bin") == [3.0]


def test_best_checkpoint_is_argmin_not_last(tmp_path: Path) -> None:
    state = _algo3_state(tmp_path)
    # min at epoch 1; epochs 2,3 worse but within patience -> full run, best != last.
    val_seq = [5.0, 2.0, 3.0, 4.0]
    res = train_modern(
        state,
        seed=0,
        params=ModernTrainParams(epochs=4, patience=10),
        train_epoch=_epoch_encoding_stub(),
        validate=_val_stub(val_seq),
        init_weights_override=[np.array([0.0])],
    )
    assert res.stopped_early is False
    assert res.best_epoch == 1
    assert _ckpt_vec(res.checkpoint_dir, "best_sad.bin") == [1.0]
    assert _ckpt_vec(res.checkpoint_dir, "last_sad.bin") == [3.0]
    # train_history.json is written and round-trips.
    from speech.drivers.state import ModernTrainResult

    reloaded = ModernTrainResult.model_validate_json((Path(res.checkpoint_dir) / "train_history.json").read_text())
    assert reloaded.best_epoch == 1
    assert [r.model_dump() for r in reloaded.history] == [r.model_dump() for r in res.history]


def test_resume_from_last_equivalence(tmp_path: Path) -> None:
    # A weight-CARRYING stub: continuation genuinely depends on reloading last_sad.bin.
    def train_epoch(weights: list[np.ndarray], epoch: int) -> tuple[list[np.ndarray], float]:
        return [np.asarray(weights[0], dtype=np.float64) + 1.0], float(epoch)

    def validate(weights: list[np.ndarray], epoch: int) -> tuple[float, float | None]:
        return float(weights[0][0]), None  # depends on the CARRIED weights

    init = [np.array([0.0])]

    # continuous 5 epochs (patience high -> no early stop)
    cont_state = _algo3_state(tmp_path / "cont")
    cont = train_modern(
        cont_state, seed=0, params=ModernTrainParams(epochs=5, patience=99), train_epoch=train_epoch, validate=validate, init_weights_override=init
    )

    # split: 3 epochs, then resume IN PLACE (same out_dir) toward the total of 5
    split_state = _algo3_state(tmp_path / "split")
    train_modern(split_state, seed=0, params=ModernTrainParams(epochs=3, patience=99), train_epoch=train_epoch, validate=validate, init_weights_override=init)
    resume_ckpt = str(Path(split_state.out_dir) / "checkpoint")
    split = train_modern(
        split_state, seed=0, params=ModernTrainParams(epochs=5, patience=99, resume_from=resume_ckpt), train_epoch=train_epoch, validate=validate
    )

    # same total, same trajectory, same checkpoints -- resuming is transparent.
    assert split.epochs_run == cont.epochs_run == 5
    assert (split.best_epoch, split.best_val_cost, split.stopped_early) == (cont.best_epoch, cont.best_val_cost, cont.stopped_early)
    assert [r.model_dump() for r in split.history] == [r.model_dump() for r in cont.history]
    for name in ("last_sad.bin", "best_sad.bin"):
        assert _ckpt_vec(split.checkpoint_dir, name) == _ckpt_vec(cont.checkpoint_dir, name)
    # sanity: the loop actually progressed weights across the boundary (final != init).
    assert _ckpt_vec(cont.checkpoint_dir, "last_sad.bin") == [5.0]


def test_resume_toward_already_reached_total_is_a_noop(tmp_path: Path) -> None:
    # Resuming with epochs <= epochs already run runs no new epoch (range is empty).
    state = _algo3_state(tmp_path)
    train_modern(
        state,
        seed=0,
        params=ModernTrainParams(epochs=3, patience=99),
        train_epoch=_epoch_encoding_stub(),
        validate=_val_stub([3.0, 2.0, 1.0]),
        init_weights_override=[np.array([0.0])],
    )
    ckpt = str(Path(state.out_dir) / "checkpoint")
    res = train_modern(
        state,
        seed=0,
        params=ModernTrainParams(epochs=3, patience=99, resume_from=ckpt),
        train_epoch=_epoch_encoding_stub(),
        validate=_val_stub([3.0, 2.0, 1.0]),
    )
    assert res.epochs_run == 3
    assert [r.epoch for r in res.history] == [0, 1, 2]
    assert _ckpt_vec(res.checkpoint_dir, "last_sad.bin") == [2.0]  # epoch 2's weights, unchanged


def test_missing_hook_without_engine_is_the_only_engine_dependency(tmp_path: Path) -> None:
    # Guard: with BOTH hooks stubbed, train_modern never imports speech_rs (the state
    # machine is engine-free). If a default hook were built, this would ImportError only
    # when speech_rs is absent -- here it must simply not be reached.
    state = _algo3_state(tmp_path)
    res = train_modern(
        state,
        seed=0,
        params=ModernTrainParams(epochs=2, patience=99),
        train_epoch=_epoch_encoding_stub(),
        validate=_val_stub([1.0, 0.5]),
        init_weights_override=[np.array([0.0])],
    )
    assert res.epochs_run == 2
