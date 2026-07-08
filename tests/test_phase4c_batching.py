"""Phase 4c Task 10: read_listing, create_batches/get_new_batch, get_worst_and_best.

The batching goldens (`tests/reference_data/phase4c/batching_*.bin`) are dumped by GNU
Octave running the REAL `legacy/Optimizer_V6.2.2/functions/{CreateBatches,GetNewBatch}.m`
(TIER 1) plus the FALLBACK-TIER `getworstandbest_transcribed.m` (verbatim transcription of
`getCases.m`'s private `getWorstAndBest` subfunction -- see
`tools/octave_harness/stage_batching.m`). `randperm` is shadowed with a fixed
reverse-order permutation (`batching_shadow/randperm.m`, `p = n:-1:1`), reproduced here via
a duck-typed `rng` stand-in -- both sides consume the SAME table-per-length, so
`create_batches`'s own shuffle is pinned, not just `get_new_batch`'s rotation.

`read_listing` has no Octave golden (it is a pure per-line record parser mirroring
`engine/corpus.rs`'s already-golden-tested listing-file branch) -- its tests are direct
unit tests over crafted listing text, mirroring `corpus.rs`'s own test names/cases.

Everything here is PURE integer-cursor arithmetic (no libm) -> STRICT bits everywhere.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from speech.batching import Batches, create_batches, get_new_batch, get_worst_and_best, read_listing
from speech.weight_bridge import read_bin

PHASE4C = Path(__file__).resolve().parent / "reference_data" / "phase4c"
MANIFEST = PHASE4C / "manifest.json"


def _load(name: str) -> NDArray[np.float64]:
    rows, cols, flat = read_bin(PHASE4C / f"batching_{name}.bin")
    return np.asarray(flat, dtype=np.float64).reshape((rows, cols), order="F")


def _load_i64_1based(name: str) -> NDArray[np.int64]:
    """Load a golden index vector and convert MATLAB's 1-based file indices to 0-based."""
    return (_load(name).astype(np.int64).reshape(-1)) - 1


class _ReverseRng:
    """Duck-typed `np.random.Generator` stand-in matching the Octave shadow
    `batching_shadow/randperm.m` (`p = n:-1:1`): `X(randperm(length(X)))` reverses `X`."""

    def permutation(self, x: NDArray[np.int64]) -> NDArray[np.int64]:
        return np.asarray(x)[::-1].copy()


def _rng() -> np.random.Generator:
    return cast(np.random.Generator, _ReverseRng())


# --------------------------------------------------------------------------- #
# read_listing
# --------------------------------------------------------------------------- #


def _write(tmp: Path, name: str, contents: str) -> Path:
    p = tmp / name
    p.write_text(contents)
    return p


def test_read_listing_basic_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "listing.csv", "a.wav;a.seg;eng;us;1.5;3\n")
        recs = read_listing(p)
        assert recs == [{"filename": "a.wav", "refseg": "a.seg", "lang": "eng", "dial": "us", "weight": "1.5", "file_id": "3"}]


def test_read_listing_nested_quirk() -> None:
    """Mirrors corpus.rs's `listing_nested_quirk`: an empty dial token (index 3) keeps
    its "unk" default but does NOT block weight parsing (tokens.len()>4 is a SIBLING of
    the tokens[3] emptiness check, not nested inside it)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "listing.csv", "f.wav;;eng;;0.5\n")
        recs = read_listing(p)
        assert recs[0]["dial"] == "unk"
        assert recs[0]["weight"] == "0.5"
        assert recs[0]["file_id"] == "1"


def test_read_listing_weight_parse_failure_keeps_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "listing.csv", "f.wav;;eng;;notanumber;7\n")
        recs = read_listing(p)
        assert recs[0]["weight"] == "1.0"
        assert recs[0]["file_id"] == "7"


def test_read_listing_trailing_delimiter_drop() -> None:
    """`splitstr`: "a;b;" -> ["a","b"] (one trailing empty dropped), but "a;b;;" keeps a
    genuinely-empty final token."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "listing.csv", "a.wav;seg;;;;\n")
        recs = read_listing(p)
        assert len(recs) == 1
        assert recs[0]["filename"] == "a.wav"
        assert recs[0]["refseg"] == "seg"


def test_read_listing_skips_blank_and_empty_filename_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "listing.csv", "a.wav;;eng;;1.0;1\n\n;b;c\nz.wav;;fra;;2.0;2\n")
        recs = read_listing(p)
        assert [r["filename"] for r in recs] == ["a.wav", "z.wav"]


# --------------------------------------------------------------------------- #
# create_batches / get_new_batch
# --------------------------------------------------------------------------- #


def test_batching_manifest_present() -> None:
    assert MANIFEST.is_file()
    m = json.loads(MANIFEST.read_text())
    assert "batching" in m
    assert m["batching"]["degenerate_batch_len"] == 3


def _rotate(batches: Batches, n: int) -> tuple[list[list[int]], list[NDArray[np.int64]], list[float]]:
    batch_traj: list[list[int]] = []
    val_traj: list[NDArray[np.int64]] = []
    cp_traj: list[float] = []
    for _ in range(n):
        batch, batches = get_new_batch(batches)
        batch_traj.append(batch)
        val_traj.append(batches.validation.copy())
        cp_traj.append(batches.count_pass)
    return batch_traj, val_traj, cp_traj


def _assert_rotation_matches(batches: Batches, case: str, n: int) -> None:
    batch_traj, val_traj, cp_traj = _rotate(batches, n)
    batch_golden = _load(f"{case}_batch_traj").astype(np.int64) - 1
    val_golden = _load(f"{case}_validation_traj").astype(np.int64)
    cp_golden = _load(f"{case}_countpass_traj").reshape(-1)
    for step in range(n):
        assert batch_traj[step] == batch_golden[:, step].tolist(), f"{case} step {step}: batch mismatch"
        assert np.array_equal(val_traj[step], val_golden[:, step]), f"{case} step {step}: validation mismatch"
        assert cp_traj[step] == cp_golden[step], f"{case} step {step}: count_pass mismatch"


def test_create_batches_single_and_rotation() -> None:
    """nbOfTargetClasses<=1, isMultiLingual<=0 (CreateBatches.m:19-27)."""
    fv = np.zeros((6, 1))
    batches = create_batches(fv, minibatch=2, nb_worst=1, multilingual=False, nb_classes=1, rng=_rng())
    assert np.array_equal(batches.cases[0].index, _load_i64_1based("single_index"))
    assert np.array_equal(batches.worst_cases[0].index, _load_i64_1based("single_worst_index"))
    _assert_rotation_matches(batches, "single", 8)


def test_create_batches_multi_nb_clean_aggregate_and_rotation() -> None:
    """nbOfTargetClasses>1, isMultiLingual<=0, classes {1,2,5} -> clean (non-clobbered)
    aggregate (CreateBatches.m:43-60)."""
    fv = np.array([1, 1, 2, 2, 5, 5], dtype=np.float64).reshape(-1, 1)
    batches = create_batches(fv, minibatch=2, nb_worst=1, multilingual=False, nb_classes=3, rng=_rng())
    assert np.array_equal(batches.cases[0].index, _load_i64_1based("multi_nb_case1_index"))
    assert np.array_equal(batches.cases[1].index, _load_i64_1based("multi_nb_case2_index"))
    assert np.array_equal(batches.cases[2].index, _load_i64_1based("multi_nb_case3_index"))
    assert np.array_equal(batches.worst_cases[0].index, _load_i64_1based("multi_nb_worst1_index"))
    assert np.array_equal(batches.worst_cases[1].index, _load_i64_1based("multi_nb_worst2_index"))
    assert np.array_equal(batches.worst_cases[2].index, _load_i64_1based("multi_nb_worst3_index"))
    _assert_rotation_matches(batches, "multi_nb", 8)


def test_create_batches_clobber_quirk() -> None:
    """The natural contiguous class labeling {0,1,2} with nb_classes=3: the LAST loop
    position (over sorted unique class values) is ITSELF a valid target class (value=2),
    so it silently OVERWRITES the aggregate slot (index nb_classes-1) reserved for
    non-target files -- class-0's indices are lost entirely. See IMPROVEMENTS.md."""
    fv = np.array([0, 0, 1, 1, 2, 2], dtype=np.float64).reshape(-1, 1)
    batches = create_batches(fv, minibatch=2, nb_worst=1, multilingual=False, nb_classes=3, rng=_rng())
    clobbered = _load_i64_1based("clobber_case3_index")
    assert np.array_equal(batches.cases[2].index, clobbered)
    # Non-vacuity: this is class-2's data (files 4,5, 0-based), NOT class-0's (files 0,1).
    assert set(batches.cases[2].index.tolist()) == {4, 5}


def test_create_batches_sub_and_rotation() -> None:
    """nbOfTargetClasses>1, isMultiLingual>0, classes {1,2,5,8} -> two SubCases groups
    under the aggregate class (CreateBatches.m:61-86)."""
    fv = np.array([1, 1, 1, 2, 2, 2, 5, 5, 5, 8, 8, 8], dtype=np.float64).reshape(-1, 1)
    batches = create_batches(fv, minibatch=2, nb_worst=1, multilingual=True, nb_classes=3, rng=_rng())
    assert np.array_equal(batches.cases[0].index, _load_i64_1based("sub_case1_index"))
    assert np.array_equal(batches.cases[1].index, _load_i64_1based("sub_case2_index"))
    assert len(batches.cases[2].sub_cases) == 2
    assert np.array_equal(batches.cases[2].sub_cases[0].index, _load_i64_1based("sub_case3_sub1_index"))
    assert np.array_equal(batches.cases[2].sub_cases[1].index, _load_i64_1based("sub_case3_sub2_index"))
    assert np.array_equal(batches.worst_cases[0].index, _load_i64_1based("sub_worst1_index"))
    assert np.array_equal(batches.worst_cases[1].index, _load_i64_1based("sub_worst2_index"))
    assert np.array_equal(batches.worst_cases[2].index, _load_i64_1based("sub_worst3_index"))
    _assert_rotation_matches(batches, "sub", 10)


def test_create_batches_degenerate_gate() -> None:
    """nbOfTargetClasses*nbOfWorstCases+nbOfCasesPerBatch > file_nb (CreateBatches.m:15-17):
    no shuffling at all -- degenerate_index is the flat, unshuffled (1:file_nb)'."""
    fv = np.array([1, 2, 0], dtype=np.float64).reshape(-1, 1)
    batches = create_batches(fv, minibatch=2, nb_worst=5, multilingual=False, nb_classes=3, rng=_rng())
    assert batches.current_class is None
    assert batches.cases == []
    assert batches.worst_cases == []
    assert batches.degenerate_index is not None
    assert np.array_equal(batches.degenerate_index, _load_i64_1based("degenerate_cases"))

    batch, batches = get_new_batch(batches)
    assert batch == _load_i64_1based("degenerate_batch").tolist()
    assert np.array_equal(batches.validation, _load("degenerate_validation").astype(np.int64).reshape(-1))


def test_create_batches_degenerate_batch_equals_cases_verbatim() -> None:
    """GetNewBatch.m:3-5: the degenerate branch returns Batches.Cases UNCHANGED (no
    rotation at all), not just a coincidentally-matching batch."""
    fv = np.array([1, 2, 0], dtype=np.float64).reshape(-1, 1)
    batches = create_batches(fv, minibatch=2, nb_worst=5, multilingual=False, nb_classes=3, rng=_rng())
    batch, _ = get_new_batch(batches)
    assert batches.degenerate_index is not None
    assert batch == batches.degenerate_index.tolist()


# --------------------------------------------------------------------------- #
# get_worst_and_best
# --------------------------------------------------------------------------- #


def test_get_worst_and_best_basic() -> None:
    idx = np.arange(1, 9)
    cost = np.array([8, 3, 6, 1, 7, 2, 5, 4])
    order = np.argsort(cost, kind="stable")
    sorting_table = np.column_stack([cost[order], idx[order] - 1]).astype(np.float64)

    r = get_worst_and_best(sorting_table, nb_worst=3, nb_best=2, nb_middle=2)
    assert r.ind_best == (_load_i64_1based("gwb_basic_indBest")).tolist()
    assert r.ind_middle == (_load_i64_1based("gwb_basic_indMiddle")).tolist()
    assert r.ind_worst == (_load_i64_1based("gwb_basic_indWorst")).tolist()
    assert r.score_best == _load("gwb_basic_scoreBest").reshape(-1).tolist()
    assert r.score_middle == _load("gwb_basic_scoreMiddle").reshape(-1).tolist()
    assert r.score_worst == _load("gwb_basic_scoreWorst").reshape(-1).tolist()


def test_get_worst_and_best_edge_zero_counts() -> None:
    """nbWorst<=0/nbBest<=0/nbMiddle<=0 all take the legacy `else` branches (widen worst
    to the whole table, reserve-but-do-not-fill one best slot, skip middle) -- nothing is
    appended to any output list."""
    idx = np.arange(1, 9)
    cost = np.array([8, 3, 6, 1, 7, 2, 5, 4])
    order = np.argsort(cost, kind="stable")
    sorting_table = np.column_stack([cost[order], idx[order] - 1]).astype(np.float64)

    r = get_worst_and_best(sorting_table, nb_worst=0, nb_best=0, nb_middle=0)
    assert r.ind_best == []
    assert r.ind_middle == []
    assert r.ind_worst == []
