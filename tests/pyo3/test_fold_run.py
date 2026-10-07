"""Issue #22: the fold run through the seam -- `Engine.from_map` and `FoldRun` on the engine.

The pure pins (the rendered map, the path keys, the fast guard's wiring) live in
`tests/test_fold_run.py` against a fake seam; these drive the real `speech_rs` module on the
committed tier-2 spectral (algo 3) and twin (algo 6) fixtures:

  * `from_map` vs the path constructor: the same map, the same fold, bit-identical results
    and weights -- the seam's two constructors are one engine.
  * The F11 rule behaviourally: a base config carrying `Epochs 3` runs ONE fold at theta
    through `FoldRun` (the pack comes back untouched; `train()` would have moved it).
  * The fast guard: on `Inference_Path fast` the weights reach the engine through a pack
    written under the workdir, never through `set_weights` (which bails there).
  * No process cwd: a fold run from an unrelated cwd on absolute listing rows matches the
    chdir'd path-constructor run.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from speech.config_bridge import parse_legacy_config
from speech.engine import ChannelResults
from speech.fold_run import FoldRun, _config_text

speech_rs = pytest.importorskip("speech_rs")

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


def _seed_tier2_spectral(dst: Path) -> dict[str, str]:
    """The algo-3 spectral 2-file corpus + config + the phase0 pack (mirrors
    `test_exit_gate._seed_tier2_spectral`). Returns the parsed config."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in ("f1", "f2"):
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, dst / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", dst / "NNweights_config1.bin")
    (dst / "vrcts_tier2").mkdir(exist_ok=True)
    return parse_legacy_config((dst / "tier2_spectral.config").read_text())


def _assert_same_results(a: ChannelResults, b: ChannelResults) -> None:
    """Every field bit-identical except `time_per_hour`, the one wall-clock column (it
    differs between two runs of the SAME engine)."""
    for f in fields(ChannelResults):
        if f.name == "time_per_hour":
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x.shape == y.shape, f.name
        if x.dtype.kind == "f":
            assert np.array_equal(x.view(np.uint64), y.view(np.uint64)), f"{f.name} differs"
        else:
            assert np.array_equal(x, y), f"{f.name} differs"


def test_from_map_matches_the_path_constructor(tmp_path: Path) -> None:
    """The same parsed map through `Engine.from_map` and the same file through the path
    constructor: identical seed weights, identical result matrix, identical channel results."""
    cfg = _seed_tier2_spectral(tmp_path)
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    with (tmp_path / "tier2_spectral.config").open("a") as fh:
        fh.write("\nNeural_Networks_BackPropagation_Epochs 0\n")

    with chdir(tmp_path):
        by_path = speech_rs.Engine(["tier2_spectral.config"], "-m")
        by_map = speech_rs.Engine.from_map([cfg], "-m")
        w_path = [np.asarray(w) for w in by_path.weights(0)]
        w_map = [np.asarray(w) for w in by_map.weights(0)]
        by_path.run()
        by_map.run()
        r_path = ChannelResults.from_seam(by_path.channel_results())
        r_map = ChannelResults.from_seam(by_map.channel_results())

    assert len(w_path) == len(w_map) == 1
    assert np.array_equal(w_path[0].view(np.uint64), w_map[0].view(np.uint64)), "the seed pack must load identically"
    assert len(r_path) == 4, "two stereo files, one config: four channel results"
    _assert_same_results(r_path, r_map)


def test_from_map_rejects_an_unknown_mode_flag(tmp_path: Path) -> None:
    cfg = _seed_tier2_spectral(tmp_path)
    with chdir(tmp_path), pytest.raises(RuntimeError, match="unknown mode flag"):
        speech_rs.Engine.from_map([cfg], "-z")


# ---- FoldRun on the real engine ---------------------------------------------------------


def test_fold_run_is_one_fold_at_theta_f11(tmp_path: Path) -> None:
    """F11 behaviourally: the committed tier-2 config carries `Epochs 3`. Through `FoldRun` a
    gradient fold measures its cost AT THETA -- bit-identical to a forward-only fold at the
    same pack -- and harvests a nonzero gradient. The foil runs the same map raw, where
    `Epochs 3` routes `run()` through the engine-internal `train()` and the cost comes back
    from engine-moved weights. Remove the Epochs-0 forcing and the cost assertion fails
    exactly like the foil.

    What a gradient fold does NOT leave untouched is the engine's own pack: `run_solo` ends
    with the bag's iRPROP- step (`save_and_update_epoch`), so `FoldResult.weights` is one
    engine step past theta on a gradient fold and theta itself on a forward-only one. Pinned
    here so nobody reads `weights` as "the input"."""
    cfg = _seed_tier2_spectral(tmp_path)
    assert cfg["Neural_Networks_BackPropagation_Epochs"] == "3", "the fixture must carry a training-shaped epoch count"
    with chdir(tmp_path):
        forward = FoldRun(cfg, tmp_path, backprop=False)
        w0 = forward.weights()
        at_theta = forward.run(w0)
        gradient = FoldRun(cfg, tmp_path, backprop=True).run(w0)

        raw = dict(FoldRun(cfg, tmp_path, backprop=True).config)
        raw["Neural_Networks_BackPropagation_Epochs"] = "3"
        foil = speech_rs.Engine.from_map([raw], "-m")
        foil.set_weights(0, [w0[0]])
        foil.run()
        foil_cost = ChannelResults.from_seam(foil.channel_results()).for_config(0)

    assert np.isfinite(at_theta.nn_cost)
    assert gradient.nn_cost == at_theta.nn_cost, "a gradient fold measures the cost at theta (F11)"
    assert len(gradient.derivs) == 1 and np.count_nonzero(gradient.derivs[0][:, 0]) > 0, "a gradient fold must harvest a nonzero gradient"
    assert at_theta.derivs == [], "a forward-only fold harvests nothing"
    assert np.array_equal(at_theta.weights[0].view(np.uint64), w0[0].view(np.uint64)), "a forward-only fold leaves the pack at theta"
    assert not np.array_equal(gradient.weights[0], w0[0]), "a gradient fold's post-run pack is one engine iRPROP- step past theta"
    foil_nn_cost = float(np.sum(foil_cost.seg_cost) / max(1e-6, float(np.sum(foil_cost.seg_count))))
    assert foil_nn_cost != at_theta.nn_cost, "the foil: Epochs 3 trains inside the engine and reports a cost at moved weights"


def test_fold_run_needs_no_process_cwd(tmp_path: Path) -> None:
    """A fold run from an unrelated cwd, on a listing with absolute rows, matches the
    chdir'd path-constructor run: the six path keys are resolved against the workdir, and
    the `.mat` the config names lands there too."""
    cfg = _seed_tier2_spectral(tmp_path)
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    listing = tmp_path / "tier2_fileslisting.csv"
    rows = listing.read_text().splitlines()
    listing.write_text("".join(";".join(str(tmp_path / c) if c.startswith("corpus/") else c for c in r.split(";")) + "\n" for r in rows))
    (tmp_path / "tier2_spectral.config").write_text(_config_text(cfg))

    with chdir(tmp_path):
        by_path = speech_rs.Engine(["tier2_spectral.config"], "-m")
        by_path.run()
        r_path = ChannelResults.from_seam(by_path.channel_results())

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with chdir(elsewhere):
        res = FoldRun(cfg, tmp_path, backprop=False).run()
    _assert_same_results(r_path, res.results)
    assert (tmp_path / cfg["multiConfigResultsOutputFile"]).is_file()
    assert not list(elsewhere.iterdir()), "nothing lands in the process cwd"


def test_fold_run_fast_guard_on_the_engine(tmp_path: Path) -> None:
    """`Inference_Path fast`: `run(weights)` with the config's own pack equals `run()` on the
    config's own pack (the pack went in through a workdir `.bin`, not `set_weights`, which
    bails on a fast processor), and a different pack changes the posteriors."""
    cfg = _seed_tier2_spectral(tmp_path)
    cfg["Inference_Path"] = "fast"
    with chdir(tmp_path):
        exact_pack = FoldRun(dict(cfg, Inference_Path="exact"), tmp_path, backprop=False).weights()
        own = FoldRun(cfg, tmp_path, backprop=False).run()
        injected = FoldRun(cfg, tmp_path, backprop=False).run(exact_pack)
        other = FoldRun(cfg, tmp_path, backprop=False).run([exact_pack[0] * 0.5])
    _assert_same_results(own.results, injected.results)
    assert not list(tmp_path.glob("_fold_*")), "the workdir packs are removed once the engine holds them"
    assert not np.array_equal(own.results.pfa, other.results.pfa), "the injected pack must be the one scored"
