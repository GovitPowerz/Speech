"""Issue #22: the fold run through the seam -- `Engine.from_map` and `FoldRun` on the engine.

The pure pins (the rendered map, the path keys, the injection's wiring) live in
`tests/test_fold_run.py` against a fake seam; these drive the real `speech_rs` module on the
committed tier-2 spectral (algo 3) and twin (algo 6) fixtures:

  * `from_map` vs the path constructor: the same map, the same fold, bit-identical results
    and weights -- the seam's two constructors are one engine.
  * The F11 rule behaviourally: a base config carrying `Epochs 3` runs ONE fold at theta
    through `FoldRun` (the pack comes back untouched; `train()` would have moved it).
  * The fast tree takes the packs the exact tree takes (issue #62): through `set_weights`
    on one reused engine, and a pack of the wrong length is refused on both trees in the
    same words (one leg per row of the issue's table, over the phase-9 fixtures).
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
from numpy.typing import NDArray
from speech.config_bridge import parse_legacy_config
from speech.engine import ChannelResults
from speech.fold_run import FoldRun, _config_text
from speech.weight_bridge import read_weight_vector

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"
PHASE9 = REPO_ROOT / "tests" / "reference_data" / "phase9"


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


def _seed_phase9_sad(dst: Path, name: str) -> dict[str, str]:
    """A phase-9 SAD config + its seed pack over the synthetic tier-2 corpus (one file)."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for ext in ("wav", "stm"):
        shutil.copy(PHASE4A / "corpus" / f"f1.{ext}", corpus / f"f1.{ext}")
    for f in ("language2classmapping.csv", "tier2_gc_fileslisting.csv"):
        shutil.copy(PHASE4A / f, dst / f)
    for f in (f"{name}.config", f"{name}_seed.bin"):
        shutil.copy(PHASE9 / f, dst / f)
    return parse_legacy_config((dst / f"{name}.config").read_text())


def _seed_phase9_twin(dst: Path, name: str) -> dict[str, str]:
    """A phase-9 Mode-7 Twin config + its LID seed over the phase-4b phSeq corpus and the
    unchanged phase-4b `tiny_sad_seed.bin`."""
    for f in ("languagemapping_lid7.csv", "tiny_sad_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    phseq = dst / "corpus_phseq"
    phseq.mkdir(parents=True, exist_ok=True)
    for f in ("s1.phSeq", "s2.phSeq", "s1.stm", "s2.stm", "listing_train.csv"):
        shutil.copy(PHASE4B / "corpus_phseq" / f, phseq / f)
    for f in (f"{name}.config", f"{name}_seed.bin"):
        shutil.copy(PHASE9 / f, dst / f)
    return parse_legacy_config((dst / f"{name}.config").read_text())


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


def test_fold_run_fast_injects_through_set_weights(tmp_path: Path) -> None:
    """`Inference_Path fast` (issue #62): the packs go in through `set_weights` on ONE engine,
    as on exact. The config's own pack injected scores as `run()` on it, a halved pack moves
    the posteriors, the first pack coming back scores bit-identically again (the reused fast
    engine carries nothing between runs), and no pack is written beside the run's own."""
    cfg = _seed_tier2_spectral(tmp_path)
    cfg["Inference_Path"] = "fast"
    with chdir(tmp_path):
        exact_pack = FoldRun(dict(cfg, Inference_Path="exact"), tmp_path, backprop=False).weights()
        own = FoldRun(cfg, tmp_path, backprop=False).run()
        fold = FoldRun(cfg, tmp_path, backprop=False)
        injected = fold.run(exact_pack)
        other = fold.run([exact_pack[0] * 0.5])
        again = fold.run(exact_pack)
    _assert_same_results(own.results, injected.results)
    _assert_same_results(injected.results, again.results)
    assert not np.array_equal(own.results.pfa, other.results.pfa), "the injected pack must be the one scored"
    assert [p.name for p in tmp_path.glob("*.bin")] == ["NNweights_config1.bin"], "nothing is written beside the run's own pack"


# One leg per row of issue #62's table: (fixture, packs from the seeded dir, the refusing net).
_WRONG_LENGTH = {
    # The sLSTM LID net (949) given its LSTM ancestor's pack (1093): the head ran on fast.
    "twin_lid_long": ("twin_mode7_lid_slstm", lambda d: [_pack(d, "tiny_sad_seed"), read_weight_vector(PHASE4B / "tiny_lid_seed.bin")]),
    # The SAD pack one too long: the fast Twin never read it.
    "twin_sad_long": ("twin_mode7_lid_slstm", lambda d: [np.append(_pack(d, "tiny_sad_seed"), 0.0), _pack(d, "twin_mode7_lid_slstm_seed")]),
    # One too short: refused on both trees, in two different wordings before the fix.
    "twin_lid_short": ("twin_mode7_lid_slstm", lambda d: [_pack(d, "tiny_sad_seed"), _pack(d, "twin_mode7_lid_slstm_seed")[:-1]]),
    # The sLSTM SAD net (1603) given the LSTM's pack (1651): the head ran on fast.
    "sad_long": ("slstm_forward", lambda d: [read_weight_vector(PHASE9 / "lstm_forward_seed.bin")]),
    "sad_short": ("slstm_forward", lambda d: [_pack(d, "slstm_forward_seed")[:-1]]),
}


def _pack(d: Path, stem: str) -> NDArray[np.float64]:
    return read_weight_vector(d / f"{stem}.bin")


@pytest.mark.parametrize("leg", sorted(_WRONG_LENGTH))
def test_fold_run_refuses_a_wrong_length_pack_on_both_trees(tmp_path: Path, leg: str) -> None:
    """Issue #62: a pack the exact tree refuses is refused on the fast tree too, with the
    identical message. Before the fix the fast tree wrote it to a workdir `.bin` and its file
    seam ran the head of an over-long pack, ignored a Twin's SAD pack, and worded a short one
    differently."""
    fixture, packs = _WRONG_LENGTH[leg]
    seed = _seed_phase9_twin if fixture.startswith("twin") else _seed_phase9_sad
    cfg = seed(tmp_path, fixture)
    nets = packs(tmp_path)
    messages = []
    with chdir(tmp_path):
        for path in ("exact", "fast"):
            with pytest.raises(RuntimeError, match="^The number of gains given is") as refused:
                FoldRun(dict(cfg, Inference_Path=path), tmp_path, backprop=False).run(nets)
            messages.append(str(refused.value))
    assert messages[0] == messages[1]
