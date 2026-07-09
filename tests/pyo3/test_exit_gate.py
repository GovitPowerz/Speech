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
