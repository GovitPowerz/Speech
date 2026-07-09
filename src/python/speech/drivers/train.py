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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from speech.batching import Batches, create_batches, get_new_batch, write_weighted_listing
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


def _tail_lengths(cfg: dict[str, str], algo: int) -> list[int]:
    """The normalize mean/std tail length per net = `2 * NNetInputSize`
    (`BLSTMNeuralNetwork::setWeights` :225-227 consumes mean then std, each inputSize).

    Returns one entry per BackPropagation net: `[sad]` for the single-net algos (3/4/5),
    `[sad, lid]` only for the algo-6 Twin -- the `BackPropagation.m:11-13` cell contract
    (`weightsIni{2}` populated iff `algo == 6`)."""
    tails = [2 * int(cfg["BLSTM_NNetInputSize"])]
    if algo == 6:
        tails.append(2 * int(cfg["BLSTM_LID_NNetInputSize"]))
    return tails


def _eval_config_text(base: dict[str, str], ponds: list[str], algo: int, fileslisting: str | None = None) -> str:
    """The committed base config with the genome's `CostPonderation` field(s) injected and
    BackPropagation forced on for a single-eval gradient (`Epochs 1`) -- the T9 single-eval
    semantics so `Engine.run()` is one forward/backward the Python loop owns. Only the algo-6
    Twin gets the `BLSTM_LID_*` injection; a single-net config's LID side is never touched
    (`BackPropagation.m` never builds the LID cell for algo != 6).

    `fileslisting` (Phase 4d Task 10): when given, overrides the corpus listing key so the
    engine folds over the per-step batch listing instead of the committed full corpus -- the
    `PS.FS.listing` re-point `ComputeGradient.m` does per gradient eval. The key already
    exists in `base`, so `dict.__setitem__` preserves its position (byte-stable ordering)."""
    cfg = dict(base)
    cfg["BLSTM_BackPropagationActivated"] = "true"
    cfg["BLSTM_CostPonderation"] = ponds[0]
    if algo == 6:
        cfg["BLSTM_LID_BackPropagationActivated"] = "true"
        cfg["BLSTM_LID_CostPonderation"] = ponds[1]
    cfg["Neural_Networks_BackPropagation_Epochs"] = "1"
    if fileslisting is not None:
        cfg["fileslisting"] = fileslisting
    return "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"


# ---- Hard-example mini-batching (Phase 4d Task 10) ---------------------------------------
#
# CADENCE (verified at legacy source -- see .superpowers/sdd/task-10-report.md):
#   * `CreateBatches` (Train_BLSTM.m:71) is called ONCE at run start, before the optimizer
#     -- NOT per epoch. The `Batches` struct persists for the whole run; `GetNewBatch` only
#     rotates its cursors. (The brief's "epoch start" prose is a documented correction.)
#   * `GetNewBatch` + `WriteWeightedListing` live inside `ComputeGradient.m` (:53/:77) --
#     the live gradient path is the ONLY caller (Train_BLSTM.m's own GetNewBatch is a
#     commented-out validation probe). ComputeGradient is one gradient eval = one inner
#     SMORMS3 step, so a fresh batch + fresh engine is drawn PER INNER STEP.
#   * The engine consumes the batch listing by reading `PS.FS.listing` (`CostFunction`
#     :104); the in-process analogue is a fresh `speech_rs.Engine` on a config whose
#     `fileslisting` points at the just-written batch listing (forward_backward's
#     `listing_override`/`make_engine`).


def _read_class_mapping(state: RunState) -> dict[tuple[str, str], int]:
    """Parse the `language2classmapping` CSV into `(lang, dial) -> class` (the
    `PS.Corpora.langMap` the corpus builds). Resolved relative to the config's own dir,
    like every other corpus path."""
    path = Path(state.config_path).parent / state.base_config["language2classmapping"]
    mapping: dict[tuple[str, str], int] = {}
    for line in path.read_text().splitlines():
        toks = line.split(";")
        if len(toks) >= 3 and toks[0]:
            mapping[(toks[0], toks[1])] = int(float(toks[2]))
    return mapping


def _files_values(state: RunState) -> NDArray[np.float64]:
    """Build `Corpora.filesValues`'s two batch-relevant columns from the listing records:
    col0 = class index (mapping lookup, legacy unknown -> -1), col1 = per-file weight
    (`filesValues(:,2)`). `create_batches` reads col0; `_BatchRunner.next_listing` reads
    col1 for the weighted listing's weight field.

    DEVIATION (IMPROVEMENTS): the port does NOT reproduce `ComputeGradient.m`'s per-eval
    class-balance RESCALE of col1 (:72-96, `nbOfElem/sumInClassIndex` etc.) -- that needs
    the `langMapConf`/in-class (`classNb == 1`) model the port's corpus does not carry. The
    raw listing weight is written instead; deterministic, and the cost impact is a per-file
    ponderation, not a determinism concern."""
    mapping = _read_class_mapping(state)
    rows = [(float(mapping.get((rec["lang"], rec["dial"]), -1)), float(rec["weight"])) for rec in state.listing]
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2), dtype=np.float64)


@dataclass
class _BatchRunner:
    """The persistent hard-example scheduler for one training run. Holds the shuffled-once
    `Batches` struct (its cursors mutate across steps) + the corpus records + `filesValues`.
    `next_listing` is the per-inner-step draw: `get_new_batch` -> union with the fixed worst
    cases -> `write_weighted_listing` to the workdir batch listing."""

    listing: list[dict[str, str]]
    files_values: NDArray[np.float64]
    batches: Batches
    workdir: Path
    last_index: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def next_listing(self) -> Path:
        """Draw one mini-batch and write it as a weighted listing to `workdir/_batch.lst`
        (overwriting the prior step's -- read immediately by the fresh engine, no race).

        Selection = `unique(new_batch U worstCases)` (`ComputeGradient.m:55`
        `count_unique([new_batch;worstCases])`). ORDER is ascending file index
        (`np.unique`), a documented simplification of the legacy's `sortrows(filesValues,
        -3)` re-sort -- the aggregate corpus cost is order-independent (compute_cost sums
        over the config's rows; `aggregate_workers` re-sorts by id).

        The weighted listing's two `%g` columns: weight = `filesValues(:,2)`; the 6th CSV
        field is the record's file_id, NOT a duration -- `WriteWeightedListing.m` writes
        `listing{i}.duration` there but `Corpus::from_config` reads field 6 as file_id (a
        legacy field-role mismatch, IMPROVEMENTS), so the port round-trips file_id and the
        engine sees the same file_id the full corpus would."""
        batch, _ = get_new_batch(self.batches)
        parts = [np.asarray(batch, dtype=np.int64)]
        parts += [w.index for w in self.batches.worst_cases if w.index.size]
        idx = np.unique(np.concatenate(parts)) if any(p.size for p in parts) else np.zeros(0, dtype=np.int64)

        items = [self.listing[int(i)] for i in idx]
        weight_col = self.files_values[idx, 1] if idx.size else np.zeros(0, dtype=np.float64)
        file_id_col = np.array([float(self.listing[int(i)]["file_id"]) for i in idx], dtype=np.float64)
        values = np.column_stack([weight_col, file_id_col]) if idx.size else np.zeros((0, 2), dtype=np.float64)

        path = self.workdir / "_batch.lst"
        write_weighted_listing(path, items, values)
        self.last_index = idx
        return path


@dataclass
class _BatchStep:
    """A per-candidate binding of the shared `_BatchRunner` to one genome's config: the
    inner loop calls `next_listing` (mutating the shared cursors) then `make_engine` to
    rebuild the corpus-scoped engine on that listing (the ponds/net structure fixed for the
    candidate, only the fileslisting rotating)."""

    runner: _BatchRunner
    base: dict[str, str]
    ponds: list[str]
    algo: int

    def next_listing(self) -> Path:
        return self.runner.next_listing()

    def make_engine(self, listing_path: Path) -> object:
        import speech_rs  # local: the pyo3 module is only needed on the engine path

        text = _eval_config_text(self.base, self.ponds, self.algo, fileslisting=Path(listing_path).name)
        (self.runner.workdir / "_batch_eval.config").write_text(text)
        return speech_rs.Engine(["_batch_eval.config"], "-m")


def _backprop_inner(
    engine: object,
    inner_steps: int,
    tails: list[int],
    batch: _BatchStep | None = None,
) -> tuple[list[NDArray[np.float64]], list[float]]:
    """`BackPropagation.m`'s inner SMORMS3 loop over the `[sad]` (algo 3/4/5) or `[sad, lid]`
    (algo 6) weight cells.

    Reads the full config-seeded weights, strips each net's normalize mean/std tail
    (`weights(1:end-2*length(normalize.mean))`), runs `inner_steps` SMORMS3 steps over
    `forward_backward` (folding the tail back for `set_weights`, dropping it from the
    gradient), and returns the FULL trained weights + the inner cost trace. The net count
    is `len(tails)` -- 1 or 2 -- so `engine.weights(0)[1]` is never indexed on a single-net
    engine (the RED IndexError).

    `batch` (Phase 4d): when set, each SMORMS3 step draws a fresh mini-batch listing and
    rebuilds the engine on it (`ComputeGradient.m`'s per-gradient GetNewBatch + fresh-fsp);
    when None, all steps run the passed engine's fixed full corpus (the 4c path)."""
    full = engine.weights(0)  # type: ignore[attr-defined]
    n = len(tails)
    fulls = [np.asarray(full[k], dtype=F64) for k in range(n)]
    net_tails = [fulls[k][len(fulls[k]) - tails[k] :].copy() for k in range(n)]
    theta0 = [fulls[k][: len(fulls[k]) - tails[k]].copy() for k in range(n)]

    def f_df(theta: list[NDArray[np.float64]], _ec: int) -> tuple[float, list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        packed = [np.concatenate([theta[k], net_tails[k]]) for k in range(n)]
        if batch is None:
            cost, grads = forward_backward(engine, packed, None)  # type: ignore[arg-type]
        else:
            listing = batch.next_listing()
            cost, grads = forward_backward(engine, packed, listing, make_engine=batch.make_engine)  # type: ignore[arg-type]
        gs = [grads[k][: theta[k].shape[0]] for k in range(n)]
        return cost, gs, theta  # theta_out is a pass-through (SMORMS3 adds dtheta)

    opt = Smorms3(f_df, theta0)
    trained = opt.optimize(inner_steps)
    out = [np.concatenate([trained[k], net_tails[k]]) for k in range(n)]
    return out, list(opt.hist_f_flat)


def _score_engine(
    config_text: str,
    workdir: Path,
    balance: int,
    algo: int,
    balance_backprop: float,
    inner_steps: int | None,
    tails: list[int],
    batch: _BatchStep | None = None,
) -> tuple[float, list[NDArray[np.float64]], list[float]]:
    """Build a fresh engine from `config_text`, optionally run the SMORMS3 inner loop,
    then score the corpus fold via the balance-law cost (`ComputeCost.m`). Returns
    (outer cost, final `[sad]`/`[sad, lid]` weights, inner cost trace).

    `batch` threads the hard-example scheduler into the inner loop (the gradient path); the
    OUTER scoring (`engine.run()` below) always folds the FULL committed corpus, matching
    the legacy -- GetNewBatch lives only in `ComputeGradient.m` (the gradient), never in the
    plain `CostFunction` scoring the post-backprop `feval(...,-5)` runs on the full listing."""
    import speech_rs  # local: the pyo3 module is only needed on the engine path

    (workdir / "_eval.config").write_text(config_text)
    engine = speech_rs.Engine(["_eval.config"], "-m")
    inner_hist: list[float] = []
    if inner_steps is not None:
        trained, inner_hist = _backprop_inner(engine, inner_steps, tails, batch)
        engine.set_weights(0, [list(w) for w in trained])
    engine.run()
    results = np.asarray(engine.results_matrix(), dtype=F64)
    cost, _ = compute_cost(results, 1, balance, CostParams(mode=0, balance_backprop=balance_backprop, algo=algo))
    return cost, [np.asarray(w, dtype=F64) for w in engine.weights(0)], inner_hist


def _ponderations(genome: NDArray[np.float64], mask: dict[str, object] | None, state: RunState) -> tuple[list[str], NDArray[np.float64]]:
    """vec2struct the genome, returning the engine ponderation string(s) + out_param. The
    algo-6 Twin adds the `BLSTM_LID_CostPonderation` field; single-net algos read the SAD
    `BLSTM_CostPonderation` alone (reading the absent LID field is the RED KeyError)."""
    cfg, out_param, _ = vec2struct(genome, mask, state.ps, 0)
    ponds = [cfg["BLSTM_CostPonderation"]]
    if state.ps.algo == 6:
        ponds.append(cfg["BLSTM_LID_CostPonderation"])
    return ponds, out_param


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
    ponds, out_param = _ponderations(genome, mask, state)
    tails = _tail_lengths(state.base_config, state.ps.algo)
    with _chdir(workdir):
        cost, _w, _h = _score_engine(
            _eval_config_text(state.base_config, ponds, state.ps.algo),
            workdir,
            state.balance,
            state.ps.algo,
            state.ps.BalanceBackProp,
            None,
            tails,
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
    minibatch: int = 0,
    nb_worst: int = 0,
    nb_classes: int = 1,
    multilingual: bool = False,
) -> TrainResult:
    """The full outer+inner training loop (`Train_BLSTM.m`), deterministic under a single
    fixed-seed numpy `Generator` (the S1 determinism deviation). QuantumPSO (`ps`/`me`)
    over the vec2struct genome; per candidate the balance-law cost of the engine fold; the
    `backprop_refine` hook + the post-QPSO step run the SMORMS3 inner loop honoring the
    BackPropagation contract; the gbest + cost histories + trained weights are checkpointed.
    `seed_value` (ReTrain's `nnet_best`) seeds the QPSO population's first rows.

    Hard-example mini-batching (Phase 4d Task 10) is OFF by default (`minibatch == 0`) --
    the full-corpus 4c path, byte-preserving the twin/algo-3 exit gates. The four knobs are
    the legacy `Train_BLSTM.m:67-70` LOCAL script variables (there are NO config keys for
    them): `minibatch` = `nbOfCasesPerBatch`, `nb_worst` = `nbOfWorstCases`, `nb_classes` =
    `nbOfTargetClasses`, `multilingual` = `isMultiLingual`. When `minibatch > 0` the inner
    SMORMS3 gradient loop draws a fresh batch listing + rebuilds the engine per step
    (`ComputeGradient.m` cadence); the outer QPSO scoring stays full-corpus."""
    ps = state.ps
    base = state.base_config
    balance = state.balance
    algo = ps.algo
    bbp = ps.BalanceBackProp
    tails = _tail_lengths(base, algo)

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

    # CreateBatches (Train_BLSTM.m:71): ONCE at run start, before the optimizer. Its own
    # generator (seed + 2) keeps the shuffle off both the QPSO (`seed`) and gate (`seed +
    # 1`) streams. `runner is None` <=> batch mode off <=> the 4c full-corpus path.
    runner: _BatchRunner | None = None
    if minibatch > 0:
        fv = _files_values(state)
        batch_rng = np.random.default_rng(seed + 2)
        batches = create_batches(fv, minibatch, nb_worst, multilingual, nb_classes, batch_rng)
        runner = _BatchRunner(state.listing, fv, batches, workdir)

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
            ponds, _out = _ponderations(pos[i], mask, state)
            batch = _BatchStep(runner, base, ponds, algo) if runner is not None else None
            cost, _w, _h = _score_engine(_eval_config_text(base, ponds, algo), workdir, balance, algo, bbp, inner_steps, tails, batch)
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
        ponds, _out = _ponderations(result.gbest, mask, state)
        final_batch = _BatchStep(runner, base, ponds, algo) if runner is not None else None
        _final_cost, final_w, inner_hist = _score_engine(_eval_config_text(base, ponds, algo), workdir, balance, algo, bbp, inner_steps, tails, final_batch)

    gbest = np.asarray(result.gbest, dtype=F64)
    cost_hist = np.asarray(result.gbestval_traj, dtype=F64)
    inner_arr = np.asarray(inner_hist, dtype=F64)
    write_bin(gbest.shape[0], 1, gbest, ckpt_dir / "gbest.bin")
    write_bin(cost_hist.shape[0], 1, cost_hist, ckpt_dir / "cost_history.bin")
    write_bin(inner_arr.shape[0], 1, inner_arr, ckpt_dir / "inner_cost_history.bin")
    write_bin(final_w[0].shape[0], 1, final_w[0], ckpt_dir / "sad_weights.bin")
    if len(final_w) >= 2:
        # legacy: only the algo-6 Twin trains a second (LID) net -- BackPropagation.m:11-13.
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
