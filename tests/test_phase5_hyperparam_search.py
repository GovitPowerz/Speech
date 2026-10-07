"""Phase 5 Task 9 fix wave: penalty-region observability + the me>=2 QPSO guard.

`train_hyperparam_search`'s per-candidate eval (`cost_fn`, train.py) catches ANY exception
`score_hyperparam_genome` raises (an invalid DSP subregion -- a feature-dim mismatch against the
fixed base net, a typed-bailed engine path) and assigns `_HYPERPARAM_PENALTY`. Before this fix, a
genuine engine defect was indistinguishable from a legitimately-invalid region (both silently
became `1e6`), and a fully-penalized run reported that `1e6` gbest with no diagnostic -- worse,
the un-guarded gbest-refetch used to CRASH instead of reporting, since re-scoring a fully-penalized
gbest hits the identical failure a second time (see `train_hyperparam_search`'s checkpoint step).

These tests pin the fix WITHOUT any engine: `train_hyperparam_search` has no injectable scoring
hook (unlike `train_modern`'s `train_epoch`/`validate`, Task 8's stub-testability convention), so
`monkeypatch` stands in for one, replacing the module-level `score_hyperparam_genome` that
`cost_fn` looks up by bare name at call time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from speech.drivers import train as train_module
from speech.drivers.state import RunState
from speech.drivers.train import _HYPERPARAM_PENALTY, train_hyperparam_search
from speech.genome import weight_block_mask

from tests.test_phase5_train_modern import _algo3_state


def test_qpso_epochs_below_two_raises(tmp_path: Path) -> None:
    """The QDPSO exploration/contraction schedule divides by `me - 1`
    (`coef_exp_contr = 0.75 - 0.5*(i-1)/(me-1)`); `qpso_epochs < 2` must be rejected before
    ever reaching `quantum_pso`, not surface as a bare `ZeroDivisionError`."""
    state = _algo3_state(tmp_path)
    with pytest.raises(ValueError, match="qpso_epochs must be >= 2"):
        train_hyperparam_search(state, seed=1, qpso_particles=2, qpso_epochs=1)


def test_penalized_eval_recorded_with_exception_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One crafted penalized candidate (a synthetic `RuntimeError` standing in for either an
    invalid DSP region or a genuine engine defect -- indistinguishable without this fix) is
    counted on the returned result: `penalized_evals >= 1` and `penalized_types["RuntimeError"]
    >= 1`. Every other candidate succeeds with a cheap constant cost, so gbest stays well below
    the penalty (a partial-penalty run, not the fully-penalized case below)."""
    state = _algo3_state(tmp_path)
    _mask, searchable = weight_block_mask(state.ps)
    n = int(searchable.sum())

    calls = {"n": 0}

    def fake_score(
        state_: RunState, reduced: NDArray[np.float64], mask_: dict[str, object], searchable_: NDArray[np.bool_], workdir: Path
    ) -> tuple[float, NDArray[np.float64], str]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic invalid DSP region")
        return 1.0, np.zeros(n, dtype=np.float64), "cfg"

    monkeypatch.setattr(train_module, "score_hyperparam_genome", fake_score)
    result = train_hyperparam_search(state, seed=1, qpso_particles=2, qpso_epochs=2)

    assert result.penalized_evals >= 1
    assert result.penalized_types.get("RuntimeError", 0) >= 1
    assert result.penalized_evals == sum(result.penalized_types.values())
    assert result.gbestval < _HYPERPARAM_PENALTY, "only one candidate was penalized -- gbest must be a real (non-penalty) cost"


def test_fully_penalized_run_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every candidate penalized (a degenerate/all-invalid search): the run still COMPLETES (the
    gbest-refetch no longer re-raises on the same failure) and reports `gbestval ==
    _HYPERPARAM_PENALTY` with a loud warning -- the fully-penalized-run diagnostic."""
    state = _algo3_state(tmp_path)

    def always_fail(
        state_: RunState, reduced: NDArray[np.float64], mask_: dict[str, object], searchable_: NDArray[np.bool_], workdir: Path
    ) -> tuple[float, NDArray[np.float64], str]:
        raise RuntimeError("synthetic invalid DSP region")

    monkeypatch.setattr(train_module, "score_hyperparam_genome", always_fail)

    with pytest.warns(UserWarning, match="all-penalized"):
        result = train_hyperparam_search(state, seed=1, qpso_particles=2, qpso_epochs=2)

    assert result.gbestval == _HYPERPARAM_PENALTY
    assert result.penalized_evals > 0
    assert result.penalized_types == {"RuntimeError": result.penalized_evals}


def test_majority_penalized_run_warns_without_reaching_full_penalty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Over half the evals penalized but NOT every one (gbest still finds a real cost below the
    penalty): the run must still warn, on the penalized-FRACTION trigger rather than the
    all-penalized (`gbestval >= _HYPERPARAM_PENALTY`) one."""
    state = _algo3_state(tmp_path)
    _mask, searchable = weight_block_mask(state.ps)
    n = int(searchable.sum())

    # keyed by POSITION (not call count): the post-search gbest refetch re-queries the exact
    # array content of an already-seen candidate, and a call-count-only rule would assign it a
    # fresh (possibly different) outcome on that second lookup -- exactly the crash hazard this
    # fix wave closes for the real `score_hyperparam_genome`, which is deterministic per-position.
    # 1 in 4 DISTINCT positions succeeds; seed=1 with qpso_particles=2/qpso_epochs=2 is a
    # deterministic, empirically-checked run landing at 16/25 evals (64%) penalized, gbest real.
    outcomes: dict[tuple[float, ...], bool] = {}
    calls = {"n": 0}

    def mostly_fail(
        state_: RunState, reduced: NDArray[np.float64], mask_: dict[str, object], searchable_: NDArray[np.bool_], workdir: Path
    ) -> tuple[float, NDArray[np.float64], str]:
        key = tuple(np.round(np.asarray(reduced, dtype=np.float64), 9))
        if key not in outcomes:
            calls["n"] += 1
            outcomes[key] = calls["n"] % 4 == 0
        if outcomes[key]:
            return 1.0, np.zeros(n, dtype=np.float64), "cfg"
        raise RuntimeError("synthetic invalid DSP region")

    monkeypatch.setattr(train_module, "score_hyperparam_genome", mostly_fail)

    with pytest.warns(UserWarning, match="were penalized"):
        result = train_hyperparam_search(state, seed=1, qpso_particles=2, qpso_epochs=2)

    assert result.gbestval < _HYPERPARAM_PENALTY, "at least one real eval must have beaten the penalty"
    assert result.penalized_evals > 0, "the crafted stub must have penalized at least one candidate"
