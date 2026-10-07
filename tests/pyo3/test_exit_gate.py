"""Phase 4c Task 12: THE EXIT GATE -- the deterministic full train loop.

Runs `drivers.train.train` end-to-end on the committed 4b twin corpus (Algo 6,
Mode 7 phSeq, TINY nets): the QuantumPSO outer (ps=2, me=2) over the vec2struct
genome, the SMORMS3 inner loop (steps=4) honoring the BackPropagation.m contract
(2-cell `[sad, lid]` weights, normalize-tail strip) via `engine.forward_backward`
on `speech_rs.Engine`, and the balance-10 LID calibration cost as the outer
objective. The whole loop is driven by ONE fixed-seed numpy Generator (the
documented determinism deviation -- the legacy reseeds from the wall clock).

The gate: run the WHOLE thing twice with the same seed and assert the checkpoint
artifacts (gbest genome, cost history, final trained weights) are bit-identical.
This is the phase's end-to-end determinism contract; per-value legacy parity is
NOT the target for the optimizer path (impossible in principle -- S1 determinism
deviation).

Phase 5 Task 10 adds the FROM-SCRATCH exit gates (bottom of the file):
`test_from_scratch_sad_converges` (the full (a) convergence / (b) held-out / (c)
determinism gate, anchored on the modern loop's SMORMS3-over-`forward_backward`
core), `test_from_scratch_twin_mechanical` (the user-ratified REDUCED twin gate --
nonzero gradients + weight movement + determinism for BOTH nets; the Twin's
CONVERGENCE gate is deferred to Phase 6, see IMPROVEMENTS.md), and
`test_early_stop_triggers` (the real-path plateau early-stop).

The committed fixtures' listing rows are relative to the fixture directory, so every driver
call here keeps the process cwd there: the fold run (issue #22) resolves the config's path
keys against the workdir, and the engine resolves a listing ROW against the cwd, as the
binary does.

Marked `slow` + pyo3 (module-level importorskip); runs in the CI python-pyo3 job.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray
from speech.config_bridge import nnet_spec
from speech.drivers.init import init_run
from speech.drivers.state import ModernTrainParams
from speech.drivers.train import (
    _HYPERPARAM_PENALTY,
    _backprop_inner,
    _eval_overlay,
    _ponderations,
    _tail_lengths,
    score_hyperparam_genome,
    train,
    train_hyperparam_search,
    train_modern,
)
from speech.engine import forward_backward
from speech.fold_run import FoldRun
from speech.genome import genome_length, weight_block_mask
from speech.init_weights import init_weights
from speech.optimizers import Smorms3

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


def _seed_twin_train(dst: Path) -> Path:
    """Mirrors `tests/pyo3/test_seam_replay.py::_seed_twin_train`: the Mode-7 2-file
    phSeq corpus + config + both TINY-net seed packs. Returns the config path."""
    for f in ("twin_train.config", "languagemapping_lid7.csv", "tiny_sad_seed.bin", "tiny_lid_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    phseq = dst / "corpus_phseq"
    phseq.mkdir(parents=True, exist_ok=True)
    for f in ("s1.phSeq", "s2.phSeq", "s1.stm", "s2.stm", "listing_train.csv"):
        shutil.copy(PHASE4B / "corpus_phseq" / f, phseq / f)
    return dst / "twin_train.config"


def _run_once(tmp: Path, seed: int) -> dict[str, bytes]:
    config = _seed_twin_train(tmp)
    state = init_run(config, tmp / "run")
    with chdir(tmp):
        result = train(state, seed=seed, qpso_particles=2, qpso_epochs=2, inner_steps=4)
    ckpt = Path(result.checkpoint_dir)
    artifacts = {name: (ckpt / name).read_bytes() for name in ("gbest.bin", "cost_history.bin", "inner_cost_history.bin", "sad_weights.bin", "lid_weights.bin")}
    artifacts["_gbestval"] = np.float64(result.gbestval).tobytes()
    return artifacts


def _run_once_batch(tmp: Path, seed: int) -> dict[str, bytes]:
    """Phase 4d Task 10: the twin exit gate with hard-example mini-batching ON. minibatch=1
    (single-file batches rotating over the 2-file corpus), nb_worst=0, nb_classes=1 -- so the
    inner SMORMS3 loop draws a fresh batch listing + REBUILDS the engine on it per step
    (`ComputeGradient.m` cadence), instead of the full corpus. Determinism must survive the
    per-step engine rebuilds + the seeded `create_batches` shuffle (batch RNG = seed + 2)."""
    config = _seed_twin_train(tmp)
    state = init_run(config, tmp / "run")
    with chdir(tmp):
        result = train(state, seed=seed, qpso_particles=2, qpso_epochs=2, inner_steps=3, minibatch=1, nb_worst=0, nb_classes=1, multilingual=False)
    ckpt = Path(result.checkpoint_dir)
    artifacts = {name: (ckpt / name).read_bytes() for name in ("gbest.bin", "cost_history.bin", "inner_cost_history.bin", "sad_weights.bin", "lid_weights.bin")}
    artifacts["_gbestval"] = np.float64(result.gbestval).tobytes()
    return artifacts


@pytest.mark.slow
def test_batch_mode_deterministic(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The batch-mode analogue of the twin exit gate: mini-batching ON (a fresh gradient
    fold per inner step on the rotated weighted listing, `_BatchStep.next_fold`), run
    twice at the same fixed seed, must produce bit-identical checkpoints. Pins that the
    fresh-engine-per-step batch path + the seeded `create_batches`/`get_new_batch` rotation
    are deterministic end to end. Also asserts the run actually mini-batched (a batch listing
    was written to the workdir)."""
    a_tmp = tmp_path_factory.mktemp("exit_gate_batch_a")
    a = _run_once_batch(a_tmp, seed=20260709)
    b = _run_once_batch(tmp_path_factory.mktemp("exit_gate_batch_b"), seed=20260709)

    assert set(a) == set(b)
    for name in a:
        assert a[name] == b[name], f"batch-mode checkpoint artifact {name!r} is not bit-identical across two fixed-seed runs"

    # Non-vacuity: mini-batching actually engaged (a batch listing exists) and the loop ran.
    assert (a_tmp / "_batch.lst").is_file(), "batch mode must have written at least one weighted batch listing"
    inner_hist = np.frombuffer(a["inner_cost_history.bin"], dtype="<f8")[2:]
    assert inner_hist.size >= 3 and np.isfinite(inner_hist).all(), "the batched SMORMS3 inner loop (>=3 steps) must run + record finite costs"


def _seed_tier2_spectral(dst: Path) -> Path:
    """Mirrors `tests/pyo3/test_seam_replay.py::_seed_tier2_spectral`: the Algo-3 (spectral,
    single-net SAD) 2-file corpus + config + the phase0 33,671-weight pack the config's
    relative `BLSTM_weightsFile` resolves to, plus the `Dump_Directory`. Returns the config
    path. Chosen over `phase4d/parity_tupleA.config` for the algo-3 smoke because this corpus
    is the 4c seam-replay fixture (proven `Engine` compatibility) and one eval is ~0.2s."""
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


def _run_once_algo3(tmp: Path, seed: int) -> dict[str, bytes]:
    config = _seed_tier2_spectral(tmp)
    state = init_run(config, tmp / "run")
    assert state.algo == 3 and state.ps.lid is None, "the smoke must exercise the single-net path"
    with chdir(tmp):
        result = train(state, seed=seed, qpso_particles=2, qpso_epochs=2, inner_steps=2)
    ckpt = Path(result.checkpoint_dir)
    assert not (ckpt / "lid_weights.bin").exists(), "a single-net algo-3 run must NOT write a LID weight pack"
    artifacts = {name: (ckpt / name).read_bytes() for name in ("gbest.bin", "cost_history.bin", "inner_cost_history.bin", "sad_weights.bin")}
    artifacts["_gbestval"] = np.float64(result.gbestval).tobytes()
    return artifacts


@pytest.mark.slow
def test_full_train_loop_algo3_deterministic(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The single-net (Algo 3) analogue of the twin exit gate: the full outer+inner loop on
    the tier-2 spectral corpus (ps=2, me=2, SMORMS3 steps=2), run twice at the same fixed
    seed, must produce bit-identical checkpoint artifacts. Pins that the algo-3 driver
    generalization (`_tail_lengths`/`_ponderations`/`_backprop_inner` -> 1-cell `[sad]`)
    is deterministic and never touches the LID side."""
    a = _run_once_algo3(tmp_path_factory.mktemp("exit_gate_algo3_a"), seed=20260709)
    b = _run_once_algo3(tmp_path_factory.mktemp("exit_gate_algo3_b"), seed=20260709)

    assert set(a) == set(b)
    assert "lid_weights.bin" not in a, "the single-net checkpoint set carries no LID pack"
    for name in a:
        assert a[name] == b[name], f"algo-3 checkpoint artifact {name!r} is not bit-identical across two fixed-seed runs"

    # Non-vacuity: the loop actually ran (finite cost history + a real SMORMS3 inner trace).
    cost_hist = np.frombuffer(a["cost_history.bin"], dtype="<f8")[2:]  # skip the (rows, cols) header
    assert cost_hist.size > 0 and np.isfinite(cost_hist).all(), "QPSO cost history must be finite + non-empty"
    inner_hist = np.frombuffer(a["inner_cost_history.bin"], dtype="<f8")[2:]
    assert inner_hist.size >= 2 and np.isfinite(inner_hist).all(), "the SMORMS3 inner loop (>=2 steps) must run + record finite costs"


@pytest.mark.slow
def test_full_train_loop_deterministic(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The full outer+inner loop, run twice at the same fixed seed, must produce
    bit-identical checkpoint artifacts (gbest genome, QPSO cost history, inner
    SMORMS3 cost history, final trained SAD/LID weights) + gbestval."""
    a = _run_once(tmp_path_factory.mktemp("exit_gate_a"), seed=20260709)
    b = _run_once(tmp_path_factory.mktemp("exit_gate_b"), seed=20260709)

    assert set(a) == set(b)
    for name in a:
        assert a[name] == b[name], f"checkpoint artifact {name!r} is not bit-identical across two fixed-seed runs"

    # Non-vacuity: the artifacts are non-trivial (cost history has finite entries,
    # trained weights are real vectors), so the determinism claim is meaningful.
    cost_hist = np.frombuffer(a["cost_history.bin"], dtype="<f8")[2:]  # skip the (rows, cols) header
    assert cost_hist.size > 0 and np.isfinite(cost_hist).all(), "QPSO cost history must be finite + non-empty"
    inner_hist = np.frombuffer(a["inner_cost_history.bin"], dtype="<f8")[2:]
    assert inner_hist.size >= 4 and np.isfinite(inner_hist).all(), "the SMORMS3 inner loop (>=4 steps) must run + record finite costs"


def test_genome_ponderation_moves_gradient(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The genome->engine non-vacuity pin, F11-corrected (was `test_genome_ponderation_moves_cost`).

    The exit gate's determinism claim is only meaningful if the genome's decoded
    `CostPonderation` fields (`train.py`'s `_eval_overlay` injection) actually move the
    engine. F11 (phase 5) makes the single-eval `Epochs 0` (run_solo, forward-only scoring at
    the fixed base weights), which exposes the true role of the CostPonderation: it is a
    BACKWARD / cost-law WEIGHTING knob, NOT a forward-scoring knob. It does NOT move the
    forward-scoring cost on these fixtures -- balance-5 mode-0 zeroes the `nn_cost_seg` term
    and balance-10 scores the (ponderation-invariant) LID calibration columns -- so the old
    `score_genome` cost-delta assertion is invalid: it only ever passed via the pre-F11
    `Epochs 1` misroute that trained each candidate 3 folds + 2 Rprop before scoring.

    The ponderation's real, robust, non-vacuous effect is on the GRADIENT
    (`ponderate_weights_derivatives` scales col0 -- the summed derivative -- but NOT the col1
    frame count, so `col0/col1` moves with it), which is what drives the inner SMORMS3 that
    the exit gate's final BackPropagation runs. Two well-separated ponderation genomes ->
    robustly different LID gradients at the same theta. No exact float is pinned (libm-robust,
    no canary machinery)."""
    tmp = tmp_path_factory.mktemp("genome_grad")
    config = _seed_twin_train(tmp)
    state = init_run(config, tmp / "run")
    workdir = Path(state.config_path).parent
    d = genome_length(state.ps)

    genome_a = np.random.default_rng(1).uniform(0.0, state.ps.adim, size=d)
    genome_b = np.random.default_rng(2).uniform(0.0, state.ps.adim, size=d)

    def _lid_gradient(genome: np.ndarray) -> np.ndarray:
        ponds, _ = _ponderations(genome, None, state)
        with chdir(workdir):
            fold = FoldRun(_eval_overlay(state.base_config, ponds, state.ps.algo), workdir, backprop=True)
            _f, grads = forward_backward(fold, fold.weights())
        assert len(grads) == 2, "twin -> [sad, lid] gradients"
        return np.asarray(grads[1], dtype=np.float64)

    grad_a = _lid_gradient(genome_a)
    grad_b = _lid_gradient(genome_b)

    assert np.isfinite(grad_a).all() and np.isfinite(grad_b).all(), "gradients must be finite"
    lid_delta = float(np.linalg.norm(grad_a - grad_b))
    assert lid_delta > 1e-2, (
        f"distinct ponderation genomes produced near-identical LID gradients (delta {lid_delta}) -- the CostPonderation genome injection may be vacuous"
    )


# ---- Phase 5 Task 9: the NARROWED (non-weight) genome search ----------------------------
#
# The stronger sibling of `test_genome_ponderation_moves_cost`: with the weight blocks pinned OUT
# of the genome and the FULL non-weight DSP key set injected (not just the 2 CostPonderation
# fields), two candidates over the narrowed genome must produce DISTINCT engine configs AND
# distinct costs. The perturbation stays in the dimension-SAFE front-matter slice (decision
# thresholds/areas, windowing, DC/preemph/noise -- 12 DSP keys) so the FIXED base net's input
# dimension is preserved: freely searching the feature-DIMENSION keys against a fixed net is an
# invalid region the from-scratch search penalizes (exercised by the determinism gate below), not
# a per-candidate non-vacuity claim.
_SAFE_FRONT_MATTER_DIMS = 12


def test_hyperparam_search_narrowed_distinct_configs_and_costs(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The 4c non-vacuity pattern over the NARROWED genome + FULL key set: two searchable genomes
    (differing across the safe DSP front-matter slice) injected onto the base produce DISTINCT
    engine configs AND distinct finite costs. Far stronger than the old 2-key check -- the injected
    keys are genuine DSP hyperparameters (decision thresholds, windowing, preemph, noise) that
    reshape the SAD segmentation, not just the calibration ponderation."""
    tmp = tmp_path_factory.mktemp("hyperparam_nonvac")
    config = _seed_tier2_spectral(tmp)
    state = init_run(config, tmp / "run")
    assert state.algo == 3, "the non-vacuity smoke uses the single-net spectral corpus (real audio)"
    workdir = Path(state.config_path).parent

    mask, searchable = weight_block_mask(state.ps)
    n = int(searchable.sum())
    assert n < genome_length(state.ps) - 1, "the narrowed genome must be SMALLER than the full weight-carrying genome"

    def _front_matter_genome(seed: int) -> np.ndarray:
        g = np.zeros(n, dtype=np.float64)
        g[:_SAFE_FRONT_MATTER_DIMS] = np.random.default_rng(seed).uniform(0.0, state.ps.adim, size=_SAFE_FRONT_MATTER_DIMS)
        return g

    with chdir(workdir):
        cost_a, _out_a, text_a = score_hyperparam_genome(state, _front_matter_genome(1), mask, searchable, workdir)
        cost_b, _out_b, text_b = score_hyperparam_genome(state, _front_matter_genome(2), mask, searchable, workdir)

    assert np.isfinite(cost_a) and np.isfinite(cost_b), f"costs must be finite: {cost_a}, {cost_b}"
    assert text_a != text_b, "distinct hyperparameter genomes must inject DISTINCT engine configs"
    assert abs(cost_a - cost_b) > 1e-3, (
        f"distinct DSP hyperparameters decoded to near-identical costs ({cost_a} vs {cost_b}) -- the full-key-set injection may be vacuous"
    )
    # the injected configs are forward-only (fixed base weights) and carry no weight matrices.
    assert "Neural_Networks_BackPropagation_Epochs 0" in text_a
    assert not any(("_LSTMBlock_" in ln or "NormalizeInput" in ln) for ln in text_a.split("\n"))


@pytest.mark.slow
def test_hyperparam_search_runs_and_is_deterministic(tmp_path_factory: pytest.TempPathFactory) -> None:
    """`train_hyperparam_search` (QuantumPSO over the narrowed non-weight genome, ps=2/me=2, no
    inner SMORMS3) runs end to end on the single-net spectral corpus and is DETERMINISTIC: run
    twice at the same fixed seed, the checkpoints (gbest searchable genome, cost history) are
    bit-identical. Pins that (a) the narrowed search is runnable despite invalid DSP subregions
    (penalized, not fatal) and (b) it finds a VALID gbest (below the invalid-region penalty)."""

    def _run_once(tmp: Path) -> tuple[dict[str, bytes], float, int]:
        config = _seed_tier2_spectral(tmp)
        state = init_run(config, tmp / "run")
        _mask, searchable = weight_block_mask(state.ps)
        with chdir(tmp):
            result = train_hyperparam_search(state, seed=20260716, qpso_particles=2, qpso_epochs=2)
        ckpt = Path(result.checkpoint_dir)
        arts = {name: (ckpt / name).read_bytes() for name in ("gbest.bin", "cost_history.bin")}
        arts["_best_config"] = (ckpt / "best_hyperparam.config").read_bytes()
        return arts, result.gbestval, int(searchable.sum())

    a, gbestval_a, n_search = _run_once(tmp_path_factory.mktemp("hyperparam_search_a"))
    b, gbestval_b, _ = _run_once(tmp_path_factory.mktemp("hyperparam_search_b"))

    assert set(a) == set(b)
    for name in a:
        assert a[name] == b[name], f"narrowed-search checkpoint {name!r} is not bit-identical across two fixed-seed runs"
    assert gbestval_a == gbestval_b

    # the search ran over the narrowed genome and found a VALID config (below the invalid penalty).
    gbest = np.frombuffer(a["gbest.bin"], dtype="<f8")[2:]  # skip the (rows, cols) header
    assert gbest.size == n_search, f"gbest must be the narrowed genome ({gbest.size} != {n_search})"
    assert np.isfinite(gbestval_a) and gbestval_a < _HYPERPARAM_PENALTY, f"the search must find a valid (non-penalty) gbest, got {gbestval_a}"


# ---- Phase 5 Task 10: the from-scratch exit gates ---------------------------------------
#
# The dual-head convergence adjudication (user-ratified, 2026-07-16): SAD carries the FULL
# convergence gate; the Twin's CONVERGENCE gate is DEFERRED to Phase 6, so the Twin keeps a
# MECHANICAL gate only. The Twin's committed corpus is 1 file per language, so any disjoint
# train/held-out split puts a DIFFERENT language in the held-out than in training -- LID
# generalization is structurally unsatisfiable on it -- and the near-saturated tiny net's train
# improvement is a negligible ~2%. See IMPROVEMENTS.md `[phase5]` (Task 10 twin deferred
# convergence) for the full spec-deviation record.
#
# Both gates are anchored on the MODERN loop's core -- SMORMS3 over `engine.forward_backward`
# (F11 single-eval-at-theta, Epochs 0; F10 folded gradient), the exact inner primitive
# `train_modern`'s `_backprop_inner` rides -- NOT the legacy `train`'s QuantumPSO surface (which
# is flat on these fixtures: a forward-only score at the fixed base weights).

# Margins are HALF the measured from-scratch improvement (~2x honest headroom), NOT tuned to
# green -- TRUNCATED to 3 decimals (0.02640 -> 0.026, 0.01070 -> 0.010), so each margin sits
# fractionally BELOW true half (by 0.0004 / 0.0007): direction-safe (the coded bound is
# marginally easier to clear than exact-half phrasing implies, never harder), not a precision
# bug. Measured (seed 7, xavier, the 8-step engine-reused trajectory below):
#   SAD train cost-at-theta: 0.34742 -> 0.29462   improvement 0.05280 -> margin 0.026
#   SAD held-out (f2, forward NNCostSeg): 0.35065 -> 0.32925   improvement 0.02140 -> margin 0.010
# The run is BOUNDED at 8 SMORMS3 steps because the trajectory OVERSHOOTS from step 9: the
# cost-at-theta series 0.34742 0.34742 0.34742 0.34742 0.34741 0.34737 0.34692 0.34248 0.29462
# (steps 0..8) is monotone down, then step 9 jumps to 0.45633 and step 10 to 3.59834 (SMORMS3's
# lrate has ramped 1e-9 -> 1e-1 by then and steps past the minimum). The monotone assertion in
# gate (a) would FAIL if the budget reached step 9 -- the bound is load-bearing, not cosmetic.
_SAD_TRAIN_MARGIN = 0.026
_SAD_HELDOUT_MARGIN = 0.010
_SAD_STEPS = 8


class _SadRun(NamedTuple):
    trace: NDArray[np.float64]  # cost-at-theta, f@theta_0 .. f@theta_{_SAD_STEPS}
    trained: NDArray[np.float64]
    init: NDArray[np.float64]
    ho_init: float  # held-out (f2) forward NNCostSeg at the init weights
    ho_best: float  # held-out (f2) forward NNCostSeg at the trained weights


class _TwinRun(NamedTuple):
    nonzero: list[int]  # per-net nonzero-gradient count [sad, lid]
    move: list[float]  # per-net ||trained - init|| [sad, lid]
    trained: list[NDArray[np.float64]]


def _seed_twin_gradcheck(dst: Path) -> Path:
    """The Mode-5 Twin corpus (1 wav file, vie/vie) + config + both TINY seed packs. Mirrors
    `tests/pyo3/test_seam_replay.py::_seed_twin_gradcheck`. Chosen over `_seed_twin_train`
    (Mode 7) for the MECHANICAL gate because Mode 7's committed config has `BLSTM_Back
    PropagationActivated false` -- the SAD net is a FROZEN feature extractor feeding the LID
    net, so it takes no gradient and never moves (a "both nets move" gate is unsatisfiable on
    it). Mode 5 is the committed Twin config where BOTH nets are backprop-active (the same
    fixture `test_grad_check_seam` pins as having nonzero per-net gradients)."""
    for f in ("twin_gradcheck.config", "languagemapping_lid7.csv", "listing_gc_wav.csv", "tiny_sad_seed.bin", "tiny_lid_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    wav = dst / "corpus_lid"
    wav.mkdir(parents=True, exist_ok=True)
    for f in ("f1.wav", "f1.stm", "f1_gc.stm"):
        shutil.copy(PHASE4B / "corpus_lid" / f, wav / f)
    return dst / "twin_gradcheck.config"


def _sad_from_scratch(tmp: Path, seed: int) -> _SadRun:
    """One from-scratch SAD (algo 3) training run through the modern loop's core: seeded Xavier
    init -> `_SAD_STEPS` SMORMS3 steps over `forward_backward` (backprop ON, Epochs 0) on the
    TRAIN listing (f1 only) with the engine REUSED across steps (the `_backprop_inner`
    semantics) -> a forward NNCostSeg readout of the trained weights on the HELD-OUT file (f2).

    Splits the committed 2-file listing into `train_f1.csv` / `heldout_f2.csv` in the workdir
    so training never sees f2 (gate (b) is a genuine held-out generalization check). Returns
    the cost-at-theta trace (`f@theta_0 .. f@theta_{_SAD_STEPS}`, the last from a forward-only
    readout at the trained weights -- NOT a further step, so the run stops at-or-before step 8),
    the trained + init weight vectors, and the two held-out costs."""
    corpus = tmp / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in ("f1", "f2"):
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, tmp / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", tmp / "NNweights_config1.bin")
    (tmp / "vrcts_tier2").mkdir()
    lines = (tmp / "tier2_fileslisting.csv").read_text().splitlines()
    (tmp / "train_f1.csv").write_text(lines[0] + "\n")  # f1: eng/us, class 0
    (tmp / "heldout_f2.csv").write_text(lines[1] + "\n")  # f2: unmapped (held out of training)

    state = init_run(tmp / "tier2_spectral.config", tmp / "run")
    assert state.algo == 3 and state.ps.lid is None, "the SAD gate must exercise the single-net path"
    base = state.base_config
    workdir = Path(state.config_path).parent
    init_pack = init_weights(nnet_spec(base, "BLSTM"), np.random.default_rng(seed), "xavier", True)

    with chdir(workdir):
        fold = FoldRun(base, workdir, backprop=True, listing="train_f1.csv")

        def f_df(theta: list[np.ndarray], _ec: int) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
            cost, grads = forward_backward(fold, theta)
            return cost, grads, theta

        opt = Smorms3(f_df, [init_pack[0].copy()])
        trained = opt.optimize(_SAD_STEPS)  # _SAD_STEPS evals at theta_0..theta_{S-1}, final = theta_S
        best_cost, _g = forward_backward(fold, [trained[0]])  # forward readout at theta_S (NOT a step)
        trace = np.array([*opt.hist_f_flat, best_cost], dtype=np.float64)  # f@theta_0 .. f@theta_S

        heldout = FoldRun(base, workdir, backprop=True, listing="heldout_f2.csv")
        ho_init, _ = forward_backward(heldout, [init_pack[0]])
        ho_best, _ = forward_backward(heldout, [trained[0]])

    return _SadRun(trace=trace, trained=trained[0], init=init_pack[0], ho_init=float(ho_init), ho_best=float(ho_best))


@pytest.mark.slow
def test_from_scratch_sad_converges(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The SAD (algo 3) from-scratch convergence exit gate (a)/(b)/(c). From a seeded Xavier
    init, SMORMS3 over `forward_backward` (the modern loop's core) genuinely descends on the
    differentiable NNCostSeg objective, generalizes to a held-out file, and is deterministic.

    Anchored on the differentiable NNCostSeg (via `forward_backward`), NOT the balance-5 forward
    "validation" cost `train_modern` records per epoch -- that is a discrete error-rate step
    function stuck at 30.0 from scratch on this fixture (the net's posteriors never cross the
    decision threshold), a useless early-stop signal here (a separate finding; `test_early_stop
    _triggers` uses it deliberately as a plateau). The run is bounded at `_SAD_STEPS` steps
    because the trajectory overshoots from step 9 (see the module-level margin derivation)."""
    r = _sad_from_scratch(tmp_path_factory.mktemp("sad_conv_a"), seed=7)
    trace = r.trace
    init_cost, best_cost = float(trace[0]), float(trace[-1])

    # (a) monotone non-increasing within the 8-step budget: the measured trajectory descends
    # from step 0 to step 8; a tolerance of 1e-4 absorbs cross-libm jitter on the near-flat
    # early steps while still catching the step-9 overshoot (+0.16, far above 1e-4).
    for i in range(1, trace.size):
        assert trace[i] <= trace[i - 1] + 1e-4, f"cost-at-theta rose at step {i}: {trace[i]} > {trace[i - 1]} (overshoot inside the budget?)"
    # (a) convergence: the best cost (= trace[-1], the min by monotonicity) is strictly below
    # the epoch-0 cost by the stated margin (a truncated-to-3-decimals half of the measured
    # 0.05280 improvement -- see the module-level margin derivation, direction-safe).
    assert best_cost == float(np.min(trace)), f"the trained-weights cost must be the minimum of the bounded trace: {best_cost} vs {float(np.min(trace))}"
    assert init_cost - best_cost >= _SAD_TRAIN_MARGIN, f"SAD train did not converge: improvement {init_cost - best_cost:.5f} < margin {_SAD_TRAIN_MARGIN}"

    # (b) held-out generalization: the trained model beats the untrained init on f2 (forward
    # NNCostSeg), by a truncated-to-3-decimals half of the measured 0.02140 held-out improvement.
    ho_imp = r.ho_init - r.ho_best
    assert ho_imp >= _SAD_HELDOUT_MARGIN, f"SAD did not generalize: held-out improvement {ho_imp:.5f} < margin {_SAD_HELDOUT_MARGIN}"

    # (c) determinism: a second fixed-seed run reproduces the trajectory + trained weights bit-for-bit.
    r2 = _sad_from_scratch(tmp_path_factory.mktemp("sad_conv_b"), seed=7)
    assert np.array_equal(trace.view(np.uint64), r2.trace.view(np.uint64)), "the cost-at-theta trace must be bit-identical across two fixed-seed runs"
    assert np.array_equal(r.trained.view(np.uint64), r2.trained.view(np.uint64)), "the trained SAD weights must be bit-identical across two fixed-seed runs"


def _twin_mechanical(tmp: Path, seed: int, n_steps: int) -> _TwinRun:
    """One from-scratch Twin (algo 6) mechanical run on the Mode-5 corpus (both nets
    backprop-active): seeded He init of BOTH nets -> one `forward_backward` for the per-net
    gradients -> `n_steps` of `_backprop_inner` (the modern loop's inner primitive) for the
    per-net weight movement. Overrides `Gradient_Check_Epsilon` to 0 so `Engine.run()` takes
    the run_solo single-fold path (the config ships Epsilon 1e-5 for its gradCheck role, which
    would otherwise route `run()` to gradCheck instead of a forward/backward fold)."""
    config = _seed_twin_gradcheck(tmp)
    state = init_run(config, tmp / "run")
    assert state.algo == 6 and state.ps.lid is not None, "the twin gate must exercise the two-net path"
    base = dict(state.base_config)
    base["Neural_Networks_Gradient_Check_Epsilon"] = "0"  # run_solo (a fold at theta), not gradCheck
    workdir = Path(state.config_path).parent
    tails = _tail_lengths(base, 6)
    rng = np.random.default_rng(seed)
    init_pack = init_weights(nnet_spec(base, "BLSTM"), rng, "he", True) + init_weights(nnet_spec(base, "BLSTM_LID"), rng, "he", True)

    with chdir(workdir):
        _f, grads = forward_backward(FoldRun(base, workdir, backprop=True), [init_pack[0], init_pack[1]])
        nonzero = [int(np.count_nonzero(g)) for g in grads]

        trained, _hist = _backprop_inner(FoldRun(base, workdir, backprop=True), n_steps, tails, batch=None, seed_weights=init_pack)
        move = [float(np.linalg.norm(np.asarray(trained[k]) - init_pack[k])) for k in range(2)]

    return _TwinRun(nonzero=nonzero, move=move, trained=[np.asarray(w, dtype=np.float64) for w in trained])


@pytest.mark.slow
def test_from_scratch_twin_mechanical(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The Twin (algo 6) from-scratch MECHANICAL exit gate -- the user-ratified REDUCED gate.

    DEFERRED CONVERGENCE (spec deviation, user-ratified 2026-07-16; IMPROVEMENTS.md `[phase5]`
    Task 10 twin deferred convergence): a full dual-head convergence gate (train + held-out)
    is UNSATISFIABLE on the committed Twin corpus -- it holds exactly 1 file per language, so
    any disjoint train/held-out split trains on one language and validates on a DIFFERENT one
    (LID generalization is structurally impossible), and the near-saturated tiny net's train
    improvement is a negligible ~2%. The SAD head carries the full convergence gate
    (`test_from_scratch_sad_converges`); the Twin's convergence gate moves to Phase 6, gated on
    a real >= 2-file-per-language corpus.

    What this gate DOES prove (the foundation the Phase 6 convergence gate will ride): from a
    seeded He init, the modern-loop core drives BOTH nets -- nonzero per-net gradients flow
    (the F10 folded-gradient seam; a zero gradient here is exactly the pre-F10 no-op the T10
    discovery surfaced), the weights MOVE for both nets, and the whole run is deterministic
    (bit-identical trained weights across two fixed-seed runs). Uses the Mode-5 corpus, where
    both nets are backprop-active -- see `_seed_twin_gradcheck` for why Mode 7 is unusable."""
    r = _twin_mechanical(tmp_path_factory.mktemp("twin_mech_a"), seed=3, n_steps=5)
    assert r.nonzero[0] > 0 and r.nonzero[1] > 0, f"both nets must receive a NONZERO gradient (F10): sad/lid nonzero counts {r.nonzero}"
    assert r.move[0] > 0.0 and r.move[1] > 0.0, f"both nets' weights must MOVE off init: ||trained-init|| sad/lid {r.move}"

    r2 = _twin_mechanical(tmp_path_factory.mktemp("twin_mech_b"), seed=3, n_steps=5)
    for k, name in enumerate(("sad", "lid")):
        assert np.array_equal(r.trained[k].view(np.uint64), r2.trained[k].view(np.uint64)), (
            f"the trained {name} weights must be bit-identical across two fixed-seed runs"
        )


@pytest.mark.slow
def test_early_stop_triggers(tmp_path: Path) -> None:
    """The real-path early-stop gate on the LEGACY balance signal (explicitly `val_metric=
    "balance"`): `train_modern` with the engine-backed default hooks (no stubs) on the SAD
    fixture, `patience=1`, must STOP before the epoch budget. The forward-only balance-5
    validation cost is stuck at 30.0 from scratch on this fixture (the net's posteriors never
    cross the decision threshold), so it never strictly improves after epoch 0 -- a genuine
    plateau -- and early-stop fires at epoch 1 (`epochs_run == 2 < epochs == 4`). Distinct from
    the stub-driven state-machine unit tests (`tests/test_phase5_train_modern.py`): this drives
    the REAL engine loop end to end.

    Phase 6 Task 6 flipped the DEFAULT validation metric to `nn_cost_seg` (a continuous signal
    that does NOT plateau here), so this gate now pins `val_metric="balance"` to keep exercising
    the phase-5 balance-plateau behavior deliberately; the honest early-stop on the moving
    nn_cost_seg signal is pinned separately by
    `test_phase5_train_modern_smoke.py::test_early_stop_triggers_on_moving_nn_cost_seg`."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in ("f1", "f2"):
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, tmp_path / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", tmp_path / "NNweights_config1.bin")
    (tmp_path / "vrcts_tier2").mkdir()

    state = init_run(tmp_path / "tier2_spectral.config", tmp_path / "run")
    params = ModernTrainParams(epochs=4, patience=1, steps_per_epoch=2, init_scheme="xavier", init_seed=7, val_metric="balance")
    with chdir(tmp_path):
        res = train_modern(state, seed=0, params=params)

    assert res.stopped_early is True, "early-stop must fire on the stuck-validation plateau"
    assert res.epochs_run < params.epochs, f"early-stop must halt before the {params.epochs}-epoch budget, ran {res.epochs_run}"
    # the plateau was detected AT the best epoch (no strict improvement afterward).
    assert res.best_epoch == 0, f"the stuck-at-30.0 validation makes epoch 0 the (only) best, got best_epoch={res.best_epoch}"
