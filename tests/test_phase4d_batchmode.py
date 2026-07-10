"""Phase 4d Task 10: batch-mode wiring in the train driver.

Verifies the three load-bearing pieces of hard-example mini-batching going live:

  * the per-step batch listing BYTES (`_BatchRunner.next_listing` -> the Task-8
    `write_weighted_listing` output for a seeded `create_batches`/`get_new_batch`
    rotation), pinned exactly on a crafted 5-file corpus;
  * the rotation cursor advancing across steps (each `next_listing` consumes one
    `get_new_batch`, cycling the pool -- pure, no RNG);
  * `engine.forward_backward`'s now-live `listing_override`: when set it REBUILDS
    the engine on that listing via `make_engine` (the fresh-`fsp`-per-eval
    analogue), and when None it runs the passed engine unchanged (the full-corpus
    4c path -- the exit-gate-preserving default).

CADENCE (verified at legacy source, see task-10-report.md): `CreateBatches` is
called ONCE at run start (Train_BLSTM.m:71), NOT per epoch; `GetNewBatch` +
`WriteWeightedListing` live inside `ComputeGradient.m` (:53/:77) -- i.e. once per
gradient evaluation = once per inner SMORMS3 step. The brief's "epoch start" prose
is a documented correction: there is no epoch loop around `CreateBatches`.

No `speech_rs` needed here (the fresh-engine build is exercised through a fake in
`test_forward_backward_*`); the full determinism gate lives in
`tests/pyo3/test_exit_gate.py::test_batch_mode_deterministic`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from speech.batching import create_batches
from speech.drivers.train import _BatchRunner
from speech.engine import forward_backward


class _IdentityRng:
    """A `create_batches`-injectable RNG whose `permutation` is the identity -- so
    the shuffled pools stay in ascending file-index order and the rotation is fully
    hand-predictable (the same determinism-injection convention as
    `tests/test_phase4c_batching.py::_ReverseRng`, but identity for legibility)."""

    def permutation(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(a).copy()


def _records() -> list[dict[str, str]]:
    # 5-file synthetic corpus; file_id left at the read_listing default "1".
    return [
        {"filename": f"f{i}", "refseg": f"r{i}", "lang": "eng", "dial": "us", "weight": w, "file_id": "1"}
        for i, w in enumerate(("1.0", "0.5", "2.0", "3.0", "4.0"))
    ]


def _files_values() -> np.ndarray:
    # col0 = class index (single class here), col1 = per-file weight.
    weights = [1.0, 0.5, 2.0, 3.0, 4.0]
    return np.array([(0.0, w) for w in weights], dtype=np.float64)


def _runner(tmp_path: Path, minibatch: int, nb_worst: int) -> _BatchRunner:
    fv = _files_values()
    rng = cast(np.random.Generator, _IdentityRng())
    batches = create_batches(fv, minibatch, nb_worst, multilingual=False, nb_classes=1, rng=rng)
    return _BatchRunner(listing=_records(), files_values=fv, batches=batches, workdir=tmp_path)


def test_batch_listing_bytes_step0(tmp_path: Path) -> None:
    """minibatch=2, nb_worst=0, identity shuffle -> step 0 selects files [0,1]. The
    weighted listing is byte-exact: weight from files_values col1 (`%g`), file_id (1)
    in the 6th CSV field (the WriteWeightedListing duration slot the engine reads as
    file_id -- see IMPROVEMENTS)."""
    runner = _runner(tmp_path, minibatch=2, nb_worst=0)
    path = runner.next_listing()
    assert path.read_bytes() == b"f0;r0;eng;us;1;1;\nf1;r1;eng;us;0.5;1;\n"
    assert list(runner.last_index) == [0, 1]


def test_batch_listing_bytes_rotates_across_steps(tmp_path: Path) -> None:
    """The pure-rotation cursor advances: step 0 -> [0,1], step 1 -> [2,3], step 2
    wraps -> [4,0]. Each step overwrites the same `_batch.lst`, so the bytes differ
    step to step (the engine sees a genuinely different sub-corpus each inner step)."""
    runner = _runner(tmp_path, minibatch=2, nb_worst=0)

    runner.next_listing()
    assert list(runner.last_index) == [0, 1]
    step0 = (tmp_path / "_batch.lst").read_bytes()

    runner.next_listing()
    assert list(runner.last_index) == [2, 3]
    step1 = (tmp_path / "_batch.lst").read_bytes()

    runner.next_listing()
    assert list(runner.last_index) == [0, 4]  # np.unique sorts the wrapped [4,0]

    assert step0 != step1
    assert step1 == b"f2;r2;eng;us;2;1;\nf3;r3;eng;us;3;1;\n"


def test_batch_listing_includes_worst_cases(tmp_path: Path) -> None:
    """The written listing is `unique(new_batch U worstCases)` (ComputeGradient.m:55
    `count_unique([new_batch;worstCases])`): with nb_worst=1 the (identity-shuffled)
    worst file 0 is UNION-ed into every batch, so it always appears even when the
    rotation skips it."""
    runner = _runner(tmp_path, minibatch=1, nb_worst=1)
    # worst pool = file 0 (identity shuffle, first nb_worst). get_new_batch skips the
    # worst file when rotating, but next_listing unions it back in.
    runner.next_listing()
    assert 0 in set(runner.last_index)


def test_files_values_degenerate_gate_full_corpus(tmp_path: Path) -> None:
    """When the batch config is degenerate (`nb_classes*nb_worst+minibatch > file_nb`),
    create_batches returns the full-batch struct (`current_class is None`) and
    get_new_batch yields ALL files -- next_listing then writes the whole corpus
    (ComputeGradient.m's `~isempty(WorstCases)` else branch)."""
    runner = _runner(tmp_path, minibatch=99, nb_worst=0)  # 99 > 5 -> degenerate
    assert runner.batches.current_class is None
    runner.next_listing()
    assert list(runner.last_index) == [0, 1, 2, 3, 4]


# ---- forward_backward: the now-live listing_override ------------------------------------


class _FakeEngine:
    def __init__(self) -> None:
        self.ran = False
        self.set_weights_calls: list[tuple[int, list]] = []

    def set_weights(self, pos: int, nets: list) -> None:
        self.set_weights_calls.append((pos, nets))

    def run(self) -> None:
        self.ran = True

    def results_matrix(self) -> np.ndarray:
        # one row: [file, conf=1, chan, res...] with error_vad col4=10, col-1=2 -> nn_cost_seg=5.
        return np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0]], dtype=np.float64)

    def weights_derivatives(self, pos: int) -> list[np.ndarray]:
        return [np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float64)]


def test_forward_backward_listing_override_none_uses_passed_engine() -> None:
    """listing_override=None (the 4c full-corpus default): the passed engine is run
    as-is, make_engine is NEVER consulted -- this is the branch the twin/algo-3 exit
    gates ride, so it must stay byte-identical to pre-Task-10."""
    eng = _FakeEngine()
    calls: list[Path] = []

    def make_engine(p: Path) -> _FakeEngine:
        calls.append(p)
        return eng

    f, grads = forward_backward(eng, [np.ones(3)], None, make_engine=make_engine)

    assert eng.ran and calls == [], "the passed engine ran; make_engine was not called"
    assert f == pytest.approx(5.0)
    assert list(grads[0]) == pytest.approx([1.0, 1.0, 3.0])


def test_forward_backward_listing_override_rebuilds_via_make_engine(tmp_path: Path) -> None:
    """listing_override set: forward_backward REBUILDS the engine on that listing via
    make_engine (the fresh-fsp-per-eval contract) and runs the FRESH engine, not the
    passed one."""
    passed = _FakeEngine()
    fresh = _FakeEngine()
    got: list[Path] = []
    lst = tmp_path / "_batch.lst"

    def make_engine(p: Path) -> _FakeEngine:
        got.append(p)
        return fresh

    f, grads = forward_backward(passed, [np.ones(3)], lst, make_engine=make_engine)

    assert got == [lst], "make_engine called once with the batch listing path"
    assert fresh.ran and not passed.ran, "the fresh (rebuilt) engine ran; the passed one did not"
    assert f == pytest.approx(5.0)


def test_forward_backward_listing_override_requires_make_engine(tmp_path: Path) -> None:
    """A listing_override with no make_engine is a programming error (there is no way
    to build the fresh engine) -- surfaced as ValueError, not a silent full-corpus run."""
    with pytest.raises(ValueError, match="make_engine"):
        forward_backward(_FakeEngine(), [np.ones(3)], tmp_path / "_batch.lst")
