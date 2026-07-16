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
from speech.drivers.train import _BatchRunner, class_balance_values
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


def _runner(tmp_path: Path, minibatch: int, nb_worst: int, *, algo: int = 3) -> _BatchRunner:
    # algo=3 (a SAD algo, < 5) by default: ComputeGradient.m:74-76's gate then forces the
    # WRITTEN weight to a flat 1.0 for every row regardless of the raw listing weight, so
    # these rotation-focused tests don't need to carry the F3 rescale arithmetic -- see
    # test_batch_listing_bytes_class_balance_rescale below for that (a real algo>=5,
    # nb_target_classes<=2 corpus is required to observe the rescale surviving the gate).
    fv = _files_values()
    rng = cast(np.random.Generator, _IdentityRng())
    batches = create_batches(fv, minibatch, nb_worst, multilingual=False, nb_classes=1, rng=rng)
    mapping_path = tmp_path / "language2classmapping"
    if not mapping_path.exists():
        mapping_path.write_text("eng;us;1\n")
    return _BatchRunner(listing=_records(), files_values=fv, batches=batches, workdir=tmp_path, mapping_path=mapping_path, algo=algo)


def test_batch_listing_bytes_step0(tmp_path: Path) -> None:
    """minibatch=2, nb_worst=0, identity shuffle -> step 0 selects files [0,1]. algo=3
    (<5) trips the ComputeGradient.m:74-76 gate, so the weighted listing's weight field
    is a flat 1.0 for every row -- NOT files_values col1 (F3, IMPROVEMENTS.md: the raw
    listing weight is never what gets written once batch mode is on; see
    test_batch_listing_bytes_class_balance_rescale for the algo>=5 case where the
    rescale itself is observable). File_id (1) is in the 6th CSV field (the
    WriteWeightedListing duration slot the engine reads as file_id -- see IMPROVEMENTS).

    RE-PIN (Phase 5, F3): pre-fix this golden read
    `b"f0;r0;eng;us;1;1;\\nf1;r1;eng;us;0.5;1;\\n"` (the raw per-file weights 1/0.5) and
    went RED under the fix -- f1's raw weight 0.5 is no longer written verbatim once the
    gate always forces 1.0."""
    runner = _runner(tmp_path, minibatch=2, nb_worst=0)
    path = runner.next_listing()
    assert path.read_bytes() == b"f0;r0;eng;us;1;1;\nf1;r1;eng;us;1;1;\n"
    assert list(runner.last_index) == [0, 1]


def test_batch_listing_bytes_rotates_across_steps(tmp_path: Path) -> None:
    """The pure-rotation cursor advances: step 0 -> [0,1], step 1 -> [2,3], step 2
    wraps -> [4,0]. Each step overwrites the same `_batch.lst`, so the SELECTED FILES
    differ step to step even though (algo=3, gated) every row's weight is flat 1.0 --
    the file identities, not the weight field, are what distinguish step0 from step1
    here (see test_batch_listing_bytes_class_balance_rescale for weight-value coverage).

    RE-PIN (Phase 5, F3): pre-fix, step1 read
    `b"f2;r2;eng;us;2;1;\\nf3;r3;eng;us;3;1;\\n"` (raw weights 2/3); post-fix the gate
    forces both to 1."""
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
    assert step1 == b"f2;r2;eng;us;1;1;\nf3;r3;eng;us;1;1;\n"


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


# ---- F3 (Phase 5, IMPROVEMENTS.md): the per-eval class-balance rescale -------------------
#
# ComputeGradient.m:74-76 gates the whole :59-73 rescale block: it survives ONLY for
# algo>=5 (LID) with nbOfTargetClasses<=2 (a binary target/non-target split). Every test
# above uses algo=3 (<5), where the gate always wins and the rescale is unreachable -- these
# tests instead use algo=6 (Twin LID) to put the rescale itself on trial.


def _class_balance_records() -> list[dict[str, str]]:
    # Crafted 2-class corpus: f0/f1 map to classid 1 (in-class), f2 -> classid 2, f3 ->
    # classid 3 (both out-of-class; two DISTINCT non-1 classids so the mapping-file KEY
    # loop's per-key accumulation is genuinely exercised, not just a single else-branch key).
    return [
        {"filename": "f0", "refseg": "r0", "lang": "eng", "dial": "us", "weight": "1.0", "file_id": "1"},
        {"filename": "f1", "refseg": "r1", "lang": "eng", "dial": "us", "weight": "3.0", "file_id": "1"},
        {"filename": "f2", "refseg": "r2", "lang": "fra", "dial": "fr", "weight": "2.0", "file_id": "1"},
        {"filename": "f3", "refseg": "r3", "lang": "deu", "dial": "de", "weight": "6.0", "file_id": "1"},
    ]


def _write_class_balance_mapping(tmp_path: Path) -> Path:
    mapping_path = tmp_path / "language2classmapping"
    mapping_path.write_text("eng;us;1\nfra;fr;2\ndeu;de;3\n")
    return mapping_path


def test_class_balance_values_matches_hand_derivation(tmp_path: Path) -> None:
    """Direct pin of `class_balance_values` (F3), isolated from the batch-selection /
    listing-writer plumbing.

    Hand derivation (ComputeGradient.m:59-73, one term per mapping-file KEY):
        mapping: eng_us -> 1 (in-class), fra_fr -> 2, deu_de -> 3 (both out-of-class)
        batch:   f0(class1, w=1.0)  f1(class1, w=3.0)  f2(class2, w=2.0)  f3(class3, w=6.0)

        key eng_us (classNb=1): sumIn  += 1.0+3.0 = 4.0;  nbOfElem += 2
        key fra_fr (classNb=2): sumOut += 2.0
        key deu_de (classNb=3): sumOut += 6.0                 => sumOut = 8.0

        factor_in  = nbOfElem/sumIn  = 2/4.0 = 0.5
        factor_out = nbOfElem/sumOut = 2/8.0 = 0.25

        f0: 1.0*0.5  = 0.5      f1: 3.0*0.5  = 1.5
        f2: 2.0*0.25 = 0.5      f3: 6.0*0.25 = 1.5
    """
    mapping_path = _write_class_balance_mapping(tmp_path)
    out = class_balance_values(_class_balance_records(), mapping_path)
    assert out.tolist() == pytest.approx([0.5, 1.5, 0.5, 1.5])


def test_batch_listing_bytes_class_balance_rescale(tmp_path: Path) -> None:
    """End-to-end through `_BatchRunner.next_listing`: algo=6 (>=5), nb_classes=1 (so
    `Batches.nb_target_classes=1`, not >2) -- the :74-76 gate does NOT apply, so the
    written weight field is the class_balance_values rescale (see
    test_class_balance_values_matches_hand_derivation for the arithmetic), not a flat 1.0
    and not the raw listing weight either.

    RED (Phase 5, F3): before this fix, `_BatchRunner.next_listing` wrote
    `files_values[idx, 1]` (the RAW weight: 1/3/2/6) verbatim; this test's expected bytes
    are the FIXED (rescaled) values -- there is no "old" version of this specific test
    (it is new, added alongside the fix, since the pre-fix code had no algo/mapping_path
    concept at all -- see test_batch_listing_bytes_step0/_rotates_across_steps above for
    the re-pinned PRE-EXISTING goldens that went RED under this same fix)."""
    records = _class_balance_records()
    mapping_path = _write_class_balance_mapping(tmp_path)
    fv = np.array([(1.0, 1.0), (1.0, 3.0), (2.0, 2.0), (3.0, 6.0)], dtype=np.float64)
    rng = cast(np.random.Generator, _IdentityRng())
    batches = create_batches(fv, minibatch=4, nb_worst=0, multilingual=False, nb_classes=1, rng=rng)
    runner = _BatchRunner(listing=records, files_values=fv, batches=batches, workdir=tmp_path, mapping_path=mapping_path, algo=6)

    path = runner.next_listing()
    assert path.read_bytes() == (b"f0;r0;eng;us;0.5;1;\nf1;r1;eng;us;1.5;1;\nf2;r2;fra;fr;0.5;1;\nf3;r3;deu;de;1.5;1;\n")
    assert list(runner.last_index) == [0, 1, 2, 3]


def test_batch_listing_gate_forces_flat_weight_when_nb_target_classes_exceeds_two(tmp_path: Path) -> None:
    """Same crafted corpus, same algo=6 (>=5), but nb_classes=3 this time -> `Batches.
    nb_target_classes=3 > 2` trips the OTHER half of the :74-76 gate -- the rescale is
    computable (same law as above) but never written: every row gets a flat 1.0, exactly
    like the algo<5 case. Guards the `or` in `next_listing`'s gate against a mutation
    that drops the `nb_target_classes > 2` clause (which `test_batch_listing_bytes_
    class_balance_rescale` alone would not catch, since that test's nb_classes=1)."""
    records = _class_balance_records()
    mapping_path = _write_class_balance_mapping(tmp_path)
    fv = np.array([(1.0, 1.0), (1.0, 3.0), (2.0, 2.0), (3.0, 6.0)], dtype=np.float64)
    rng = cast(np.random.Generator, _IdentityRng())
    batches = create_batches(fv, minibatch=4, nb_worst=0, multilingual=False, nb_classes=3, rng=rng)
    assert batches.nb_target_classes == 3
    runner = _BatchRunner(listing=records, files_values=fv, batches=batches, workdir=tmp_path, mapping_path=mapping_path, algo=6)

    path = runner.next_listing()
    assert path.read_bytes() == (b"f0;r0;eng;us;1;1;\nf1;r1;eng;us;1;1;\nf2;r2;fra;fr;1;1;\nf3;r3;deu;de;1;1;\n")
    assert list(runner.last_index) == [0, 1, 2, 3]


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
