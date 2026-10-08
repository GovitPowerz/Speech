"""Phase 5 Task 8: the modern training loop pyo3 SMOKE.

Exercises the ENGINE-BACKED default hooks of `train_modern` end to end on a tiny committed
corpus: from-scratch seeded init (`init_weights`) -> `steps_per_epoch` SMORMS3 steps over
`forward_backward` (backprop ON) -> per-epoch FORWARD-ONLY validation (a backprop-OFF
engine, `compute_cost` on the results) -> best/last checkpoints. 2 epochs, no early stop.
The unit tests (`tests/test_phase5_train_modern.py`) cover the state machine against stubs;
this proves the real seam runs (init pack sizes match the engine, the backprop-off
validation fold yields a finite cost, both checkpoints land).

The fixtures' listing rows are relative to the fixture directory, so every run keeps the
process cwd there (the fold run resolves the config's path keys, not the listing's rows).

Marked `slow` + pyo3 (module-level importorskip); runs in the CI python-pyo3 job.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from speech.drivers.init import init_run
from speech.drivers.state import ModernTrainParams, ModernTrainResult, RunState
from speech.drivers.train import train_modern

pytest.importorskip("speech_rs")

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
    with chdir(tmp_path):
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


@pytest.mark.slow
def test_train_modern_twin_smoke_writes_both_nets(tmp_path: Path) -> None:
    config = _seed_twin_train(tmp_path)
    state = init_run(config, tmp_path / "run")
    assert state.algo == 6 and state.ps.lid is not None

    params = ModernTrainParams(epochs=2, patience=99, steps_per_epoch=2, init_scheme="he", init_seed=3)
    with chdir(tmp_path):
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
    with chdir(tmp_path):
        train_modern(state, seed=0, params=ModernTrainParams(epochs=1, patience=99, steps_per_epoch=2, init_seed=7))
        ckpt = str(Path(state.out_dir) / "checkpoint")
        res = train_modern(state, seed=0, params=ModernTrainParams(epochs=2, patience=99, steps_per_epoch=2, resume_from=ckpt))
    assert res.epochs_run == 2
    assert [r.epoch for r in res.history] == [0, 1]


# ---- Phase 6 Task 6: the NNCostSeg validation signal (carry-forward) --------------------
#
# The Phase-5 closeout finding: `train_modern`'s balance-5 forward "validation" cost is a
# useless from-scratch early-stop signal -- on the tier2 SAD fixture it collapses to the
# discrete `100 - success` error rate, STUCK at 30.0 (the seeded net's posteriors never
# cross the decision threshold), so it never strictly improves and best-selection freezes at
# epoch 0. The differentiable NNCostSeg (the SAME quantity `forward_backward` descends --
# `f = NNCostSeg (+ NNCostLID)`, F10-fixed) is CONTINUOUS and moves from scratch. Task 6
# makes it the default validation metric (`ModernTrainParams.val_metric`), so early-stop runs
# on the moving signal. These smoke tests exercise the REAL default validate hook end to end
# (no stubs), which is where the moving-vs-stuck contrast only exists.


def _sad_state(dst: Path) -> RunState:
    """Seed + `init_run` the tier2 (algo-3 SAD) fixture into `dst`; returns the RunState."""
    config = _seed_tier2_spectral(dst)
    return init_run(config, dst / "run")


def _train_sad(dst: Path, params: ModernTrainParams) -> ModernTrainResult:
    """`train_modern` on a fresh tier2 state in `dst`, with the cwd there (relative rows)."""
    state = _sad_state(dst)
    with chdir(dst):
        return train_modern(state, seed=0, params=params)


@pytest.mark.slow
def test_val_metric_nn_cost_seg_moves_where_balance_plateaus(tmp_path: Path) -> None:
    """RED/GREEN: the SAME fixture, SAME budget -- the DEFAULT (`nn_cost_seg`) validation cost
    MOVES across epochs while the explicit `balance` cost is a FROZEN plateau (stuck at 30.0,
    best_epoch pinned at 0). Both runs use `patience=99` so all epochs run (no early stop) and
    the raw per-epoch validation signal is what is compared.

    Before Task 6 lands this is RED two ways: `ModernTrainParams(val_metric="balance")` raises
    (extra="forbid", no such field), and the default run scores the OLD stuck balance cost, so
    the `default MOVES` assertion fails. It also mutation-guards the DEFAULT: flip the state.py
    default back to `balance` and the default arm freezes -> `len(set(default_costs)) > 1` fails."""
    default_res = _train_sad(  # no val_metric -> the new nn_cost_seg default
        tmp_path / "default",
        ModernTrainParams(epochs=4, patience=99, steps_per_epoch=2, init_scheme="xavier", init_seed=7),
    )
    balance_res = _train_sad(
        tmp_path / "balance",
        ModernTrainParams(epochs=4, patience=99, steps_per_epoch=2, init_scheme="xavier", init_seed=7, val_metric="balance"),
    )

    bal_costs = [r.val_cost for r in balance_res.history]
    default_costs = [r.val_cost for r in default_res.history]

    # balance: the discrete `100 - success` error rate is a frozen plateau; best-selection can
    # never advance off epoch 0 (no strict improvement ever).
    assert len(set(bal_costs)) == 1, f"balance validation must be a frozen plateau, got {bal_costs}"
    assert balance_res.best_epoch == 0, f"the balance plateau makes epoch 0 the (only) best, got {balance_res.best_epoch}"

    # nn_cost_seg (the default): a continuous signal that MOVES -- more than one distinct value,
    # and best-selection advances off epoch 0 (structurally impossible under the frozen signal).
    assert len(set(default_costs)) > 1, f"the default (nn_cost_seg) validation must MOVE, got {default_costs}"
    assert default_res.best_epoch > 0, f"a moving signal must let best-selection advance past epoch 0, got {default_res.best_epoch}"
    assert all(np.isfinite(c) for c in default_costs)


@pytest.mark.slow
def test_val_metric_both_deterministic(tmp_path: Path) -> None:
    """The both-metrics deterministic smoke: each metric, run twice at a fixed seed, produces a
    bit-identical per-epoch validation trajectory AND bit-identical `last_sad.bin`. Proves both
    validation paths are deterministic (the property the modern loop's resume/checkpoint story
    relies on)."""
    for metric in ("nn_cost_seg", "balance"):
        params = ModernTrainParams(epochs=2, patience=99, steps_per_epoch=2, init_scheme="xavier", init_seed=7, val_metric=metric)  # type: ignore[arg-type]
        a = _train_sad(tmp_path / f"{metric}_a", params)
        b = _train_sad(tmp_path / f"{metric}_b", params)
        assert [r.val_cost for r in a.history] == [r.val_cost for r in b.history], f"{metric}: validation trajectory not deterministic"
        assert np.isfinite([r.val_cost for r in a.history]).all(), f"{metric}: validation cost must be finite"
        assert (Path(a.checkpoint_dir) / "last_sad.bin").read_bytes() == (Path(b.checkpoint_dir) / "last_sad.bin").read_bytes(), (
            f"{metric}: trained weights not bit-identical across two fixed-seed runs"
        )


@pytest.mark.slow
def test_early_stop_triggers_on_moving_nn_cost_seg(tmp_path: Path) -> None:
    """The early-stop machinery re-pinned against the MOVING signal: with `nn_cost_seg` and a
    step budget large enough to overshoot the minimum (`steps_per_epoch=8`), the validation cost
    genuinely improves for several epochs, reaches a real best at an epoch > 0, then rises --
    and early-stop fires on `patience`. This is the NON-DEGENERATE early-stop the balance
    plateau could never produce (there best_epoch is frozen at 0 and any "stop" is an artifact
    of the frozen signal). Distinct from the balance-plateau early-stop gate
    (`test_exit_gate.py::test_early_stop_triggers`, deliberately pinned to `val_metric=balance`).

    Assertions are qualitative (stopped_early, best_epoch > 0, ran under budget), not an exact
    overshoot epoch, so they survive cross-libm jitter -- the overshoot span here is ~0.2, far
    above any libm noise floor."""
    res = _train_sad(tmp_path, ModernTrainParams(epochs=16, patience=2, steps_per_epoch=8, init_scheme="xavier", init_seed=7, val_metric="nn_cost_seg"))
    assert res.stopped_early is True, "early-stop must fire on the moving nn_cost_seg signal's post-improvement overshoot"
    assert res.best_epoch > 0, f"the moving signal must reach its best at a genuine epoch > 0 (not the frozen-plateau epoch 0), got {res.best_epoch}"
    assert res.epochs_run < 16, f"early-stop must halt before the 16-epoch budget, ran {res.epochs_run}"
