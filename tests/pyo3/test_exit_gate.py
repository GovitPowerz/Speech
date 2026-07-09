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

Marked `slow` + pyo3 (module-level importorskip); runs in the CI python-pyo3 job.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from speech.drivers.init import init_run
from speech.drivers.train import score_genome, train
from speech.genome import genome_length

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"


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
    result = train(state, seed=seed, qpso_particles=2, qpso_epochs=2, inner_steps=3, minibatch=1, nb_worst=0, nb_classes=1, multilingual=False)
    ckpt = Path(result.checkpoint_dir)
    artifacts = {name: (ckpt / name).read_bytes() for name in ("gbest.bin", "cost_history.bin", "inner_cost_history.bin", "sad_weights.bin", "lid_weights.bin")}
    artifacts["_gbestval"] = np.float64(result.gbestval).tobytes()
    return artifacts


@pytest.mark.slow
def test_batch_mode_deterministic(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The batch-mode analogue of the twin exit gate: mini-batching ON (per-inner-step
    weighted-listing engine rebuilds via `forward_backward`'s live `listing_override`), run
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


def test_genome_ponderation_moves_cost(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The genome->engine non-vacuity pin. The exit gate's determinism claim is only
    meaningful if the genome's decoded `CostPonderation` fields (`train.py`'s
    `_eval_config_text` injection) actually move the engine cost -- otherwise QPSO would
    be exploring a flat cost surface. Scores the SAME seeded twin state (via
    `score_genome`, the `cost_fn` per-candidate body, no inner SMORMS3 refinement) at two
    well-separated random genomes and asserts a healthy cost delta. No exact float is
    pinned (libm-robust, no canary machinery)."""
    tmp = tmp_path_factory.mktemp("genome_cost")
    config = _seed_twin_train(tmp)
    state = init_run(config, tmp / "run")
    workdir = Path(state.config_path).parent
    d = genome_length(state.ps)

    genome_a = np.random.default_rng(1).uniform(0.0, state.ps.adim, size=d)
    genome_b = np.random.default_rng(2).uniform(0.0, state.ps.adim, size=d)

    cost_a, _ = score_genome(state, genome_a, None, workdir)
    cost_b, _ = score_genome(state, genome_b, None, workdir)

    assert np.isfinite(cost_a), f"cost_a is not finite: {cost_a}"
    assert np.isfinite(cost_b), f"cost_b is not finite: {cost_b}"
    assert abs(cost_a - cost_b) > 1e-3, (
        f"distinct genomes decoded to near-identical costs ({cost_a} vs {cost_b}) -- the CostPonderation genome injection may be vacuous"
    )
