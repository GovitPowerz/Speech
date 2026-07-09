"""Training driver: the QuantumPSO outer + SMORMS3 inner loop over the in-process engine.

Ported from `Train_BLSTM.m` (the outer driver: mask build, genome sizing via the
vec2struct count, `MaskingValidation` gate, population init, the `QuantumPSO` call at
:819, the post-QPSO `BackPropagation` at :973) + `BackPropagation.m` (the SMORMS3 inner
loop contract). The legacy seam -- write a `.config` per candidate, shell out to `fsp`,
read the `.mat`/`.bin` back -- becomes the in-process `speech_rs.Engine` (Phase 4c
Task 3) driven through `engine.forward_backward` (ComputeGradient's contract).

The genome<->engine binding (documented deviation, IMPROVEMENTS.md `phase4c-exit-gate`):
the vec2struct genome sizes/validates the search (`genome_length` + `masking_validation`
exactly as the legacy), and its two `CostPonderation` fields are surgically injected onto
the committed base `.config` per candidate (the rest of the config stays byte-identical to
the known-good base, so the engine never sees a malformed genome-derived config). The
`[sad, lid]` NN weights are seeded from the config's committed `.bin` packs -- the legacy
also decodes the initial weights from the genome, which this port does NOT wire (the
trained weights are re-seeded from the config each eval, not written back into the genome).
This keeps the exit gate a faithful full-loop DETERMINISM contract without the
config2weights round trip; per-value legacy parity of the optimizer path is impossible in
principle anyway (the clock-reseed determinism deviation, S1).

The BackPropagation.m inner-loop contract IS reproduced faithfully: a 2-cell `[sad, lid]`
SMORMS3 optimizer (`weightsIni = cell(2,1)`, :12) over the NORMALIZE-TAIL-STRIPPED weights
(`weights(1:end-2*length(normalize.mean))`, :29/:57) -- the mean/std tail is folded back
before every `Engine.set_weights` (the Rust seam demands the full vector) and dropped from
the returned gradient, so SMORMS3 only ever steps the trainable head.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from speech.drivers.state import RunState, TrainResult
from speech.engine import CostParams, compute_cost, forward_backward
from speech.genome import genome_length, vec2struct
from speech.optimizers import QpsoParams, Smorms3, quantum_pso
from speech.scoring import masking_validation
from speech.weight_bridge import write_bin

F64 = np.float64


@contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _tail_lengths(cfg: dict[str, str]) -> tuple[int, int]:
    """The normalize mean/std tail length per net = `2 * NNetInputSize`
    (`BLSTMNeuralNetwork::setWeights` :225-227 consumes mean then std, each inputSize)."""
    sad = 2 * int(cfg["BLSTM_NNetInputSize"])
    lid = 2 * int(cfg.get("BLSTM_LID_NNetInputSize", "0"))
    return sad, lid


def _eval_config_text(base: dict[str, str], sad_pond: str, lid_pond: str) -> str:
    """The committed base config with the genome's two `CostPonderation` fields injected
    and BackPropagation forced on for a single-eval gradient (`Epochs 1`) -- the T9
    single-eval semantics so `Engine.run()` is one forward/backward the Python loop owns."""
    cfg = dict(base)
    cfg["BLSTM_BackPropagationActivated"] = "true"
    cfg["BLSTM_LID_BackPropagationActivated"] = "true"
    cfg["BLSTM_CostPonderation"] = sad_pond
    cfg["BLSTM_LID_CostPonderation"] = lid_pond
    cfg["Neural_Networks_BackPropagation_Epochs"] = "1"
    return "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"


def _backprop_inner(
    engine: object,
    inner_steps: int,
    sad_tail_len: int,
    lid_tail_len: int,
) -> tuple[list[NDArray[np.float64]], list[float]]:
    """`BackPropagation.m`'s inner SMORMS3 loop over the 2-cell `[sad, lid]` weights.

    Reads the full config-seeded weights, strips each net's normalize mean/std tail
    (`weights(1:end-2*length(normalize.mean))`), runs `inner_steps` SMORMS3 steps over
    `forward_backward` (folding the tail back for `set_weights`, dropping it from the
    gradient), and returns the FULL trained `[sad, lid]` weights + the inner cost trace."""
    full = engine.weights(0)  # type: ignore[attr-defined]
    sad_full = np.asarray(full[0], dtype=F64)
    lid_full = np.asarray(full[1], dtype=F64)
    sad_tail = sad_full[len(sad_full) - sad_tail_len :].copy()
    lid_tail = lid_full[len(lid_full) - lid_tail_len :].copy()
    theta0 = [sad_full[: len(sad_full) - sad_tail_len].copy(), lid_full[: len(lid_full) - lid_tail_len].copy()]

    def f_df(theta: list[NDArray[np.float64]], _ec: int) -> tuple[float, list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        sf = np.concatenate([theta[0], sad_tail])
        lf = np.concatenate([theta[1], lid_tail])
        cost, grads = forward_backward(engine, [sf, lf], None)  # type: ignore[arg-type]
        gs = grads[0][: theta[0].shape[0]]
        gl = grads[1][: theta[1].shape[0]]
        return cost, [gs, gl], theta  # theta_out is a pass-through (SMORMS3 adds dtheta)

    opt = Smorms3(f_df, theta0)
    trained = opt.optimize(inner_steps)
    sad_out = np.concatenate([trained[0], sad_tail])
    lid_out = np.concatenate([trained[1], lid_tail])
    return [sad_out, lid_out], list(opt.hist_f_flat)


def _score_engine(
    config_text: str,
    workdir: Path,
    balance: int,
    algo: int,
    balance_backprop: float,
    inner_steps: int | None,
    sad_tail_len: int,
    lid_tail_len: int,
) -> tuple[float, list[NDArray[np.float64]], list[float]]:
    """Build a fresh engine from `config_text`, optionally run the SMORMS3 inner loop,
    then score the corpus fold via the balance-law cost (`ComputeCost.m`). Returns
    (outer cost, final `[sad, lid]` weights, inner cost trace)."""
    import speech_rs  # local: the pyo3 module is only needed on the engine path

    (workdir / "_eval.config").write_text(config_text)
    engine = speech_rs.Engine(["_eval.config"], "-m")
    inner_hist: list[float] = []
    if inner_steps is not None:
        trained, inner_hist = _backprop_inner(engine, inner_steps, sad_tail_len, lid_tail_len)
        engine.set_weights(0, [list(trained[0]), list(trained[1])])
    engine.run()
    results = np.asarray(engine.results_matrix(), dtype=F64)
    cost, _ = compute_cost(results, 1, balance, CostParams(mode=0, balance_backprop=balance_backprop, algo=algo))
    return cost, [np.asarray(w, dtype=F64) for w in engine.weights(0)], inner_hist


def _ponderations(genome: NDArray[np.float64], mask: dict[str, object] | None, state: RunState) -> tuple[str, str, NDArray[np.float64]]:
    """vec2struct the genome, returning the two engine ponderation strings + out_param."""
    cfg, out_param, _ = vec2struct(genome, mask, state.ps, 0)
    return cfg["BLSTM_CostPonderation"], cfg["BLSTM_LID_CostPonderation"], out_param


def score_genome(
    state: RunState,
    genome: NDArray[np.float64],
    mask: dict[str, object] | None,
    workdir: Path,
) -> tuple[float, NDArray[np.float64]]:
    """One QPSO candidate's cost (`cost_fn`'s per-particle body, no inner SMORMS3
    refinement): vec2struct-decode the genome's two `CostPonderation` fields, inject
    them onto the committed base config, and score the engine fold. Factored out of
    `train` so it is independently callable -- e.g. the genome->engine non-vacuity pin
    in `tests/pyo3/test_exit_gate.py`. Self-contained (chdirs into `workdir` itself),
    so it is safe to call from outside `train`'s own `_chdir(workdir)` block."""
    sad_pond, lid_pond, out_param = _ponderations(genome, mask, state)
    sad_tail_len, lid_tail_len = _tail_lengths(state.base_config)
    with _chdir(workdir):
        cost, _w, _h = _score_engine(
            _eval_config_text(state.base_config, sad_pond, lid_pond),
            workdir,
            state.balance,
            state.ps.algo,
            state.ps.BalanceBackProp,
            None,
            sad_tail_len,
            lid_tail_len,
        )
    return cost, out_param


def train(
    state: RunState,
    seed: int,
    *,
    qpso_particles: int = 24,
    qpso_epochs: int = 100,
    inner_steps: int = 20,
    seed_value: NDArray[np.float64] | None = None,
) -> TrainResult:
    """The full outer+inner training loop (`Train_BLSTM.m`), deterministic under a single
    fixed-seed numpy `Generator` (the S1 determinism deviation). QuantumPSO (`ps`/`me`)
    over the vec2struct genome; per candidate the balance-law cost of the engine fold; the
    `backprop_refine` hook + the post-QPSO step run the SMORMS3 inner loop honoring the
    BackPropagation contract; the gbest + cost histories + trained weights are checkpointed.
    `seed_value` (ReTrain's `nnet_best`) seeds the QPSO population's first rows."""
    ps = state.ps
    base = state.base_config
    balance = state.balance
    algo = ps.algo
    bbp = ps.BalanceBackProp
    sad_tail_len, lid_tail_len = _tail_lengths(base)

    mask: dict[str, object] | None = None  # full-DSP-free: the genome drives every field
    d = genome_length(ps)

    # MaskingValidation gate (Train_BLSTM.m:621-626): a representative genome must
    # round-trip through the mask. Uses its OWN generator so it never perturbs the QPSO
    # stream (both deterministic in `seed`).
    gate_rng = np.random.default_rng(seed + 1)
    if masking_validation(gate_rng.uniform(0.0, ps.adim, size=d), mask, ps):
        raise ValueError("MaskingValidation failed: the mask does not round-trip on this ps")

    workdir = Path(state.config_path).parent
    ckpt_dir = Path(state.out_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def cost_fn(positions: NDArray[np.float64], _mode: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pos = np.atleast_2d(np.asarray(positions, dtype=F64))
        costs = np.empty(pos.shape[0], dtype=F64)
        out = np.empty_like(pos)
        for i in range(pos.shape[0]):
            cost, out_param = score_genome(state, pos[i], mask, workdir)
            costs[i] = cost
            out[i] = out_param
        return costs, out

    def backprop_refine(positions: NDArray[np.float64], _epoch: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # legacy: QuantumPSO.m:564-593 -- the Rprop-refinement hook. Here it runs the
        # SMORMS3 inner loop (BackPropagation.m) on each candidate, improving its cost.
        pos = np.atleast_2d(np.asarray(positions, dtype=F64))
        costs = np.empty(pos.shape[0], dtype=F64)
        for i in range(pos.shape[0]):
            sad_pond, lid_pond, _out = _ponderations(pos[i], mask, state)
            cost, _w, _h = _score_engine(_eval_config_text(base, sad_pond, lid_pond), workdir, balance, algo, bbp, inner_steps, sad_tail_len, lid_tail_len)
            costs[i] = cost
        return costs, pos

    params = QpsoParams(
        ps=qpso_particles,
        me=qpso_epochs,
        ac1=2.1,
        ac2=2.1,
        iw1=0.9,
        iw2=0.6,
        iwe=qpso_epochs,
        ergrd=1e-99,
        ergrdep=500,
        errgoal=float("nan"),
        trelea=3,
        pso_seed=1 if seed_value is not None else 0,
        minmax=0,
        vr=np.column_stack([np.zeros(d), np.full(d, ps.adim)]),
        mv=np.full(d, ps.adim / 2.0),
        seed_value=seed_value,
        backprop_activated=1,
    )

    with _chdir(workdir):
        result = quantum_pso(cost_fn, params, d, np.random.default_rng(seed), backprop_refine=backprop_refine)
        # legacy: Train_BLSTM.m:973 -- BackPropagation on the QPSO winner. This produces
        # the final trained weights (and guarantees the inner loop runs at least once).
        sad_pond, lid_pond, _out = _ponderations(result.gbest, mask, state)
        _final_cost, final_w, inner_hist = _score_engine(
            _eval_config_text(base, sad_pond, lid_pond), workdir, balance, algo, bbp, inner_steps, sad_tail_len, lid_tail_len
        )

    gbest = np.asarray(result.gbest, dtype=F64)
    cost_hist = np.asarray(result.gbestval_traj, dtype=F64)
    inner_arr = np.asarray(inner_hist, dtype=F64)
    write_bin(gbest.shape[0], 1, gbest, ckpt_dir / "gbest.bin")
    write_bin(cost_hist.shape[0], 1, cost_hist, ckpt_dir / "cost_history.bin")
    write_bin(inner_arr.shape[0], 1, inner_arr, ckpt_dir / "inner_cost_history.bin")
    write_bin(final_w[0].shape[0], 1, final_w[0], ckpt_dir / "sad_weights.bin")
    write_bin(final_w[1].shape[0], 1, final_w[1], ckpt_dir / "lid_weights.bin")

    tr = TrainResult(
        gbest=[float(x) for x in gbest],
        gbestval=float(result.gbestval),
        cost_history=[float(x) for x in cost_hist],
        inner_cost_history=[float(x) for x in inner_hist],
        checkpoint_dir=str(ckpt_dir),
    )
    (ckpt_dir / "checkpoint.json").write_text(tr.model_dump_json(indent=2))
    return tr
