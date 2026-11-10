"""Issue #149: the Modern loop's per-epoch LID error is `evaluate.lid_error` on config 0's
validation results, the argmax error (percent) the held-out score and the ledger use, not
the `confusionThresh.m` hit rule that credits only the first two classes. Driven through
`_make_default_validate` with the fold stubbed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray
from speech.drivers import train as T
from speech.drivers.state import ModernRun, ModernTrainParams

N = 12


class _Results:
    def __init__(self, scores: NDArray[np.float64], target: NDArray[np.int64]) -> None:
        self.row = SimpleNamespace(lid_scores=scores, lid_target=target)

    def __len__(self) -> int:
        return int(self.row.lid_target.shape[0])

    def for_config(self, pos: int) -> SimpleNamespace:
        assert pos == 0
        return self.row


def _val_lid_error(monkeypatch: pytest.MonkeyPatch, scores: NDArray[np.float64], target: NDArray[np.int64], algo: int = 6) -> float | None:
    class _Fold:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, weights: object) -> SimpleNamespace:
            return SimpleNamespace(nn_cost=1.0, results=_Results(scores, target))

    monkeypatch.setattr(T, "FoldRun", _Fold)
    run = ModernRun(cfg={"Algo_choice": str(algo), "fileslisting": "listing.csv"}, workdir=Path("."), out_dir=Path("."), listing=[])
    params = ModernTrainParams(epochs=1, patience=1, steps_per_epoch=1, minibatch=0, init_scheme="xavier", seed=0)
    _, err = T._make_default_validate(run, params)([], 0)
    return err


def _diagonal(n: int) -> NDArray[np.float64]:
    scores = np.full((n, n), 1.0)
    np.fill_diagonal(scores, 90.0)
    return scores


def test_every_class_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _val_lid_error(monkeypatch, _diagonal(N), np.arange(N, dtype=np.int64)) == 0.0


def test_three_misses_in_twelve_are_25_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = _diagonal(N)
    for row in (0, 5, 11):
        scores[row, (row + 1) % N] = 95.0
    assert _val_lid_error(monkeypatch, scores, np.arange(N, dtype=np.int64)) == 25.0


def test_a_tie_won_by_the_first_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = np.full((1, N), 1.0)
    scores[0, [3, 7]] = 90.0
    assert _val_lid_error(monkeypatch, scores, np.array([3], dtype=np.int64)) == 0.0
    assert _val_lid_error(monkeypatch, scores, np.array([7], dtype=np.int64)) == 100.0


def test_none_off_the_twin_and_on_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _val_lid_error(monkeypatch, _diagonal(N), np.arange(N, dtype=np.int64), algo=3) is None
    assert _val_lid_error(monkeypatch, np.zeros((0, N)), np.zeros(0, dtype=np.int64)) is None
