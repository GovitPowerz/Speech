"""Issue #22: `FoldRun` against a fake seam -- the pure half of the fold-run pins.

What the engine receives is pinned here without the engine: a fake `speech_rs` whose
`Engine.from_map` records the map and whose `set_weights` records the packs. The real-engine
half (from_map vs path identity, the F11 rule behaviourally, the fast guard on a real fold)
is `tests/pyo3/test_fold_run.py`.
"""

from __future__ import annotations

import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from speech.fold_run import FoldResult, FoldRun, _config_text
from speech.weight_bridge import read_weight_vector

from tests._result_rows import channel_results_from_matrix

# A result row: `[file+1, conf+1, chan+1, res...]` with res[4] = seg_cost, res[14] = lid_cost,
# res[-2] = lid_count, res[-1] = seg_count (the 18-wide no-LID layout).
_ROW_WIDTH = 18


def _row(seg_cost: float, seg_count: float, lid_cost: float = 0.0, lid_count: float = 0.0) -> list[float]:
    res = [0.0] * _ROW_WIDTH
    res[4], res[14], res[16], res[17] = seg_cost, lid_cost, lid_count, seg_count
    return [1.0, 1.0, 1.0, *res]


def _fake_seam(monkeypatch: pytest.MonkeyPatch, rows: list[list[float]] | None = None) -> dict[str, Any]:
    """Install a fake `speech_rs`: records every `from_map` map and `set_weights` call."""
    seen: dict[str, Any] = {"maps": [], "set_weights": []}
    matrix = np.array(rows if rows is not None else [_row(10.0, 2.0)], dtype=np.float64)

    class Engine:
        def __init__(self) -> None:
            self.nets = [np.array([1.0, 2.0, 3.0]), np.array([4.0])]

        @staticmethod
        def from_map(configs: list[dict[str, str]], mode: str) -> Engine:
            seen["maps"].append(dict(configs[0]))
            seen["mode"] = mode
            # Like a fast processor, read the weight packs AT CONSTRUCTION (T6b).
            present = [k for k in ("BLSTM_weightsFile", "BLSTM_LID_weightsFile") if k in configs[0] and Path(configs[0][k]).is_file()]
            seen["loaded"] = {k: read_weight_vector(Path(configs[0][k])) for k in present}
            return Engine()

        def set_weights(self, pos: int, nets: list[Any]) -> None:
            seen["set_weights"].append([np.asarray(n, dtype=np.float64) for n in nets])
            self.nets = [np.asarray(n, dtype=np.float64) for n in nets]

        def run(self) -> None:
            seen["ran"] = seen.get("ran", 0) + 1

        def channel_results(self) -> dict[str, np.ndarray]:
            return asdict(channel_results_from_matrix(matrix))

        def weights(self, pos: int) -> list[np.ndarray]:
            return [w.copy() for w in self.nets]

        def weights_derivatives(self, pos: int) -> list[np.ndarray]:
            return [np.array([[2.0, 2.0], [0.0, 0.0], [6.0, 3.0]]), np.array([[1.0, 1.0]])]

    monkeypatch.setitem(sys.modules, "speech_rs", types.SimpleNamespace(Engine=Engine))
    return seen


def _base(algo: int = 3) -> dict[str, str]:
    cfg = {
        "Algo_choice": str(algo),
        "Neural_Networks_BackPropagation_Epochs": "3",
        "BLSTM_BackPropagationActivated": "false",
        "fileslisting": "listing.csv",
        "language2classmapping": "mapping.csv",
        "BLSTM_weightsFile": "sad.bin",
        "Dump_Directory": "dump",
    }
    if algo == 6:
        cfg["BLSTM_LID_BackPropagationActivated"] = "false"
        cfg["BLSTM_LID_weightsFile"] = "lid.bin"
    return cfg


# ---- the rendered map: the F11 rule and the backprop flags ------------------------------


def test_fold_run_forces_epochs_zero_on_every_fold(tmp_path: Path) -> None:
    """F11 (IMPROVEMENTS.md): a fold is ONE fold at theta, so `Epochs` is forced to 0 whatever
    the base carries, with backprop on or off -- `Epochs >= 1` would route `run()` through the
    engine-internal `train()` and measure the cost at engine-moved weights."""
    for backprop in (True, False):
        fold = FoldRun(_base(), tmp_path, backprop=backprop)
        assert fold.config["Neural_Networks_BackPropagation_Epochs"] == "0"
        assert "Neural_Networks_BackPropagation_Epochs 0\n" in fold.config_text


def test_fold_run_sets_the_backprop_flags_per_net(tmp_path: Path) -> None:
    """`backprop` is the only thing the parameter controls: `BLSTM_BackPropagationActivated`,
    plus `BLSTM_LID_BackPropagationActivated` on the Twin; a single-net config never gains a
    LID key."""
    on3 = FoldRun(_base(3), tmp_path, backprop=True).config
    off3 = FoldRun(_base(3), tmp_path, backprop=False).config
    assert on3["BLSTM_BackPropagationActivated"] == "true"
    assert off3["BLSTM_BackPropagationActivated"] == "false"
    assert "BLSTM_LID_BackPropagationActivated" not in on3

    on6 = FoldRun(_base(6), tmp_path, backprop=True).config
    off6 = FoldRun(_base(6), tmp_path, backprop=False).config
    assert on6["BLSTM_LID_BackPropagationActivated"] == "true"
    assert off6["BLSTM_LID_BackPropagationActivated"] == "false"


def test_fold_run_renders_through_the_one_config_text(tmp_path: Path) -> None:
    """`config_text` is `_config_text(config)`: byte-stable key order (the base's own order,
    an overwritten key keeps its position), so a hash of it is a hash of the map."""
    base = _base()
    fold = FoldRun(base, tmp_path, backprop=True)
    assert list(fold.config) == list(base), "no key moves; the overlay overwrites in place"
    assert fold.config_text == _config_text(fold.config)
    assert fold.config_text == "".join(f"{k} {v}\n" for k, v in fold.config.items())


def test_fold_run_listing_override(tmp_path: Path) -> None:
    fold = FoldRun(_base(), tmp_path, backprop=True, listing=tmp_path / "_batch.lst")
    assert fold.config["fileslisting"] == str(tmp_path / "_batch.lst")
    assert FoldRun(_base(), tmp_path, backprop=True, listing="valid.csv").config["fileslisting"] == "valid.csv"


def test_fold_run_never_mutates_the_base(tmp_path: Path) -> None:
    base = _base()
    before = dict(base)
    FoldRun(base, tmp_path, backprop=True, listing="x.csv")
    assert base == before


# ---- what the engine receives: absolute paths, no file, no cwd ---------------------------


def test_fold_run_hands_the_engine_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The six path keys reach `from_map` resolved against the workdir (the process cwd is not
    part of the contract); an absent `multiConfigResultsOutputFile` defaults to the workdir's
    `MultiConfigResults.mat` (where the chdir'd legacy flow put it), an absent `Dump_Directory`
    stays absent, an already-absolute value is left alone. `config` itself keeps the values as
    given, so its text is the same on every host."""
    seen = _fake_seam(monkeypatch)
    base = _base(6)
    del base["Dump_Directory"]
    base["BLSTM_LID_weightsFile"] = "/abs/lid.bin"
    fold = FoldRun(base, tmp_path, backprop=False)
    fold.run()

    assert seen["mode"] == "-m"
    (handed,) = seen["maps"]
    assert handed["fileslisting"] == str(tmp_path / "listing.csv")
    assert handed["language2classmapping"] == str(tmp_path / "mapping.csv")
    assert handed["BLSTM_weightsFile"] == str(tmp_path / "sad.bin")
    assert handed["BLSTM_LID_weightsFile"] == "/abs/lid.bin"
    assert handed["multiConfigResultsOutputFile"] == str(tmp_path / "MultiConfigResults.mat")
    assert "Dump_Directory" not in handed
    assert fold.config["fileslisting"] == "listing.csv", "the rendered map keeps the caller's relative value"
    assert "multiConfigResultsOutputFile" not in fold.config


def test_fold_run_leaves_an_empty_path_value_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty value is the engine's sentinel, not a path: `Dump_Directory ""` (the training
    TOMLs' "no dump") must reach the engine empty, not resolved to the workdir, or every fold
    writes one VRCTS xml per scored file into the run dir."""
    seen = _fake_seam(monkeypatch)
    base = _base()
    base["Dump_Directory"] = ""
    FoldRun(base, tmp_path, backprop=True).run()
    (handed,) = seen["maps"]
    assert handed["Dump_Directory"] == ""


def test_fold_run_rejects_a_pack_count_that_does_not_match_the_nets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `[sad]`-only list on the Twin would, on fast, repoint only the SAD key and score the
    config's own LID pack in silence (the #29 failure mode); it is refused before any engine."""
    seen = _fake_seam(monkeypatch)
    for fast in (False, True):
        base = _base(6)
        if fast:
            base["Inference_Path"] = "fast"
        with pytest.raises(ValueError, match="1 weight pack"):
            FoldRun(base, tmp_path, backprop=False).run([np.array([1.0, 2.0, 3.0])])
    with pytest.raises(ValueError, match="2 weight pack"):
        FoldRun(_base(3), tmp_path, backprop=False).run([np.array([1.0]), np.array([2.0])])
    assert seen["maps"] == [] and not list(tmp_path.glob("_fold_*"))


def test_fold_run_makes_a_relative_workdir_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative workdir (a CLI `out_dir`) is anchored to the cwd at construction, so the map
    the engine gets is absolute whatever the cwd is by then."""
    seen = _fake_seam(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    fold = FoldRun(_base(), Path("run"), backprop=False)
    monkeypatch.chdir(tmp_path.parent)
    fold.run()
    (handed,) = seen["maps"]
    assert handed["fileslisting"] == str(tmp_path / "run" / "listing.csv")


def test_fold_run_writes_no_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_seam(monkeypatch)
    before = set(tmp_path.iterdir())
    FoldRun(_base(), tmp_path, backprop=True).run([np.array([1.0, 2.0, 3.0])])
    assert set(tmp_path.iterdir()) == before


# ---- weights: set_weights on exact, a workdir pack on fast ------------------------------


def test_fold_run_exact_injects_through_set_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_seam(monkeypatch)
    fold = FoldRun(_base(), tmp_path, backprop=True)
    res = fold.run([np.array([5.0, 6.0, 7.0])])
    (call,) = seen["set_weights"]
    assert [list(n) for n in call] == [[5.0, 6.0, 7.0]]
    assert [list(w) for w in res.weights] == [[5.0, 6.0, 7.0]], "post-run weights read back off the engine"
    assert seen["ran"] == 1


def test_fold_run_reuses_one_engine_across_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The modern epoch calls `run` once per SMORMS3 step on the SAME engine."""
    seen = _fake_seam(monkeypatch)
    fold = FoldRun(_base(), tmp_path, backprop=True)
    fold.run([np.array([1.0, 1.0, 1.0])])
    fold.run([np.array([2.0, 2.0, 2.0])])
    fold.run()
    assert len(seen["maps"]) == 1
    assert len(seen["set_weights"]) == 2
    assert seen["ran"] == 3


def test_fold_run_fast_guard_writes_a_pack_and_repoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On `Inference_Path fast` the processor loads weights only at construction and
    `set_weights` bails (T6b): the arrays are written as `.bin` packs under the workdir, the
    weight keys repointed on a fresh engine, and the packs removed once it holds them;
    `set_weights` is never called."""
    seen = _fake_seam(monkeypatch)
    base = _base(6)
    base["Inference_Path"] = "fast"
    fold = FoldRun(base, tmp_path, backprop=False)
    sad, lid = np.array([0.5, -1.0, 2.0]), np.array([7.0])
    fold.run([sad, lid])

    assert seen["set_weights"] == []
    (handed,) = seen["maps"]
    sad_path, lid_path = Path(handed["BLSTM_weightsFile"]), Path(handed["BLSTM_LID_weightsFile"])
    assert sad_path.parent == tmp_path and lid_path.parent == tmp_path and sad_path != lid_path
    assert np.array_equal(seen["loaded"]["BLSTM_weightsFile"], sad)
    assert np.array_equal(seen["loaded"]["BLSTM_LID_weightsFile"], lid)
    assert not sad_path.exists() and not lid_path.exists(), "the packs are gone once the engine holds them"
    assert fold.config["BLSTM_weightsFile"] == "sad.bin", "the rendered map is not repointed"


def test_fold_run_fast_without_weights_runs_the_config_packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_seam(monkeypatch)
    base = _base()
    base["Inference_Path"] = "fast"
    FoldRun(base, tmp_path, backprop=False).run()
    (handed,) = seen["maps"]
    assert handed["BLSTM_weightsFile"] == str(tmp_path / "sad.bin")
    assert seen["set_weights"] == []
    assert not list(tmp_path.glob("*.bin"))


# ---- FoldResult: the costs over config 0 and the derivs ---------------------------------


def test_fold_result_costs_and_derivs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`nn_cost_seg = sum(seg_cost)/max(1e-6, sum(seg_count))`, `nn_cost_lid` likewise over the
    LID columns, `nn_cost` = seg plus lid on the Twin only (`ComputeGradient.m:110/:114`);
    `derivs` are the engine's per-net `Nx2` matrices on a gradient fold, empty on a
    forward-only one."""
    rows = [_row(10.0, 2.0, 3.0, 1.0), _row(5.0, 3.0, 1.0, 1.0)]
    _fake_seam(monkeypatch, rows)
    twin = FoldRun(_base(6), tmp_path, backprop=True).run()
    assert twin.nn_cost_seg == pytest.approx(15.0 / 5.0)
    assert twin.nn_cost_lid == pytest.approx(4.0 / 2.0)
    assert twin.nn_cost == pytest.approx(3.0 + 2.0)
    assert [d.shape for d in twin.derivs] == [(3, 2), (1, 2)]

    single = FoldRun(_base(3), tmp_path, backprop=False).run()
    assert single.nn_cost == single.nn_cost_seg == pytest.approx(3.0)
    assert single.derivs == []
    assert isinstance(single, FoldResult)
