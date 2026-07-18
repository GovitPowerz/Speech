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
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from speech.batching import Batches, create_batches, get_new_batch, write_weighted_listing
from speech.config_bridge import nnet_spec
from speech.drivers.state import EpochRecord, ModernTrainParams, ModernTrainResult, RunState, TrainResult
from speech.engine import CostParams, _error_vad, _nn_cost_lid, _nn_cost_seg, compute_cost, forward_backward
from speech.genome import RunConfig, genome_length, vec2struct, weight_block_mask
from speech.init_weights import init_weights
from speech.optimizers import QpsoParams, Smorms3, quantum_pso
from speech.scoring import confusion_matrix, masking_validation
from speech.weight_bridge import read_bin, write_bin

# Modern-loop hook types (Phase 5 Task 8): the engine-backed defaults are injectable so the
# state machine is unit-testable against a stub -- see `train_modern`.
TrainEpochFn = Callable[[list[NDArray[np.float64]], int], tuple[list[NDArray[np.float64]], float]]
ValidateFn = Callable[[list[NDArray[np.float64]], int], tuple[float, float | None]]

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
    BackPropagation forced on for a single-eval gradient (`Epochs 0`) -- the F11 single-eval
    semantics so `Engine.run()` is ONE forward/backward at the input theta that the Python
    loop owns. Only the algo-6 Twin gets the `BLSTM_LID_*` injection; a single-net config's
    LID side is never touched (`BackPropagation.m` never builds the LID cell for algo != 6).

    F11 (phase 5): `Epochs 0`, not `1`. `Epochs >= 1` routes `Engine.run()` through the
    engine-internal `train()` (3 folds + 2 Rprop updates), so the seam's cost/gradient are
    measured at engine-MOVED weights, not the input theta -- a 4c misroute. `Epochs 0` takes
    the `run_solo` path (one fold at theta; backprop is gated on `BackPropagationActivated`,
    not on Epochs, so the fold still harvests the gradient into the F10 seam stash). This
    matches the LEGACY: `ComputeGradient.m -> CostFunction.m -> ComputeCost.m` shells `fsp`
    with the base config's `Neural_Networks_BackPropagation_Epochs` UNCHANGED, and the real
    production `1_worker_1.config` has NO such key (default 0), so the legacy `fsp` ran
    `runSolo` per gradient call -- the inner Rprop loop lived in MATLAB
    (`CostFunction.m:248-291`), re-shelling `fsp` per step, exactly as this port's
    `_backprop_inner` owns the inner SMORMS3 loop. See IMPROVEMENTS.md `[phase5] F11`.

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
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
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


def _mapping_path(state: RunState) -> Path:
    """Resolve the `language2classmapping` CSV path, like every other corpus path
    (relative to the config's own dir)."""
    return Path(state.config_path).parent / state.base_config["language2classmapping"]


def _parse_class_mapping(path: Path) -> dict[tuple[str, str], int]:
    """Parse a `language2classmapping` CSV into `(lang, dial) -> class` (`PS.Corpora.
    langMap`, `processListing.m:8-10`): last-line-wins on a duplicate `(lang, dial)` key,
    matching MATLAB `containers.Map` assignment semantics (a later `langMap(key) = ...`
    overwrites silently, same as a plain dict literal loop)."""
    mapping: dict[tuple[str, str], int] = {}
    for line in Path(path).read_text().splitlines():
        toks = line.split(";")
        if len(toks) >= 3 and toks[0]:
            mapping[(toks[0], toks[1])] = int(float(toks[2]))
    return mapping


def _read_class_mapping(state: RunState) -> dict[tuple[str, str], int]:
    """`PS.Corpora.langMap` for `state`'s corpus -- see `_mapping_path`/`_parse_class_mapping`."""
    return _parse_class_mapping(_mapping_path(state))


def _files_values(state: RunState) -> NDArray[np.float64]:
    """Build `Corpora.filesValues`'s two batch-relevant columns from the listing records:
    col0 = class index (mapping lookup, legacy unknown -> -1), col1 = per-file RAW weight
    (`filesValues(:,2)` as loaded by `processListing.m`, pre-rescale). `create_batches`
    reads col0 to stratify the batch pools; col1 is no longer what ends up in the weighted
    listing (Phase 5, F3, IMPROVEMENTS.md -- FIXED, this commit): `_BatchRunner.next_listing`
    now derives the WRITTEN weight per eval via `class_balance_values` + the algo/
    nb_target_classes gate, matching `ComputeGradient.m:59-76`, instead of reading this raw
    column directly."""
    mapping = _read_class_mapping(state)
    rows = [(float(mapping.get((rec["lang"], rec["dial"]), -1)), float(rec["weight"])) for rec in state.listing]
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2), dtype=np.float64)


def class_balance_values(listing_records: list[dict[str, str]], mapping_path: Path) -> NDArray[np.float64]:
    """Port of `ComputeGradient.m:59-73` -- the per-eval class-balance rescale (Phase 5,
    F3, IMPROVEMENTS.md). `listing_records` is the CURRENT BATCH's selection (`tmp` =
    `count_unique([new_batch;worstCases])`, `:53-58`) -- the sums below run over the batch,
    not the whole corpus (the port carries no persistent corpus-wide `filesValues` state;
    see the docstring of `_BatchRunner.next_listing` for why that is the correct scope).

    TEMPORAL scope (Phase 5 T5 review): the legacy `filesValues(:,2)` DOES persist across
    optimizer calls within one run (`SMORMS3.m:306-321` threads `PS`), but
    `ComputeGradient.m:279-280` resets the whole column to 1 at the end of EVERY call
    where the periodic re-evaluation block (`:170-278`) does not fire -- which is every
    call under `Train_BLSTM.m`'s own hardcoded defaults (`adjustFileImportance = -1`).
    Recomputing from the raw CSV weight on every call therefore reproduces the canonical
    every-call-behaves-like-call-1 trace. The `Train_BLSTM_LIDSeg.m` variant
    (`adjustFileImportance = 1, RunFull = 50`), where the reset skips every 50th call and
    weights compound, is NOT modeled -- a documented residual, not an oversight.

    Law, read directly off the source (one term per KEY in the mapping file, matching
    `keySet = keys(PS.Corpora.Train.langMapConf)` / `PS.Corpora.langMap(keySet{ii})` --
    `langMapConf`'s keyset is `langMap`'s keyset plus any listing-only unmapped keys that
    would themselves error on the `langMap` lookup, so a well-formed listing -- every
    lang/dial covered by the mapping file -- makes the two keysets coincide; the port reads
    the mapping file's own keys directly, per the brief's documented scope, spec R2):

        for (lang, dial), classNb in mapping.items():
            rows = <batch records whose OWN classNb (via the SAME mapping) == this classNb>
            if classNb == 1:  sumIn  += sum(rows' raw weight); nbOfElem += len(rows)
            else:             sumOut += sum(rows' raw weight)
        weight[classNb == 1]  *= nbOfElem / sumIn
        weight[classNb != 1]  *= nbOfElem / sumOut

    Iterating per MAPPING KEY (not per distinct classid) is load-bearing and reproduced
    VERBATIM: if two keys ever shared a classid, that classid's batch contribution would be
    summed once PER KEY -- a legacy double-count quirk, not reachable by any 1-key-per-
    classid fixture (every committed corpus is 1:1) but deliberately not "cleaned up" to a
    per-unique-classid loop, since that would silently change the law on a corpus where it
    matters.

    Division follows plain IEEE-754 double semantics (0/0 -> nan, x/0 -> inf, matching
    MATLAB exactly) via `numpy` scalars, not Python floats (which raise ZeroDivisionError on
    a literal `0.0/0.0`) -- a batch with zero representation on one side of the split
    produces a nan/inf factor for that side, but it is applied only to that side's row
    selection, which is then necessarily EMPTY, so the factor is computed but never actually
    written into `out` (no observable effect, matching how legacy's own equivalent
    zero-batch-representation case behaves on the port's per-batch scope).

    Does NOT apply the `ComputeGradient.m:74-76` algo/`nbOfTargetClasses` gate
    (`filesValues(:,2) = 1` override for `algo < 5` or `nbOfTargetClasses > 2`) -- that
    decision needs `algo`/`nb_target_classes`, outside this function's 2-argument contract;
    `_BatchRunner.next_listing` applies it."""
    mapping = _parse_class_mapping(mapping_path)
    class_of = np.array([float(mapping.get((r["lang"], r["dial"]), -1)) for r in listing_records], dtype=np.float64)
    raw_weight = np.array([float(r["weight"]) for r in listing_records], dtype=np.float64)

    sum_in = np.float64(0.0)
    sum_out = np.float64(0.0)
    nb_elem = np.float64(0.0)
    for classid in mapping.values():
        rows = raw_weight[class_of == float(classid)]
        if classid == 1:
            sum_in += rows.sum()
            nb_elem += rows.size
        else:
            sum_out += rows.sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        factor_in = nb_elem / sum_in
        factor_out = nb_elem / sum_out

    out = raw_weight.copy()
    out[class_of == 1] *= factor_in
    out[class_of != 1] *= factor_out
    return out


@dataclass
class _BatchRunner:
    """The persistent hard-example scheduler for one training run. Holds the shuffled-once
    `Batches` struct (its cursors mutate across steps) + the corpus records + `filesValues`.
    `next_listing` is the per-inner-step draw: `get_new_batch` -> union with the fixed worst
    cases -> `write_weighted_listing` to the workdir batch listing.

    `files_values` is retained for parity with `create_batches`'s own input (constructed
    once, upstream, from the same `_files_values` call) but its col1 (raw weight) is no
    longer read by `next_listing` post-Phase-5-F3 -- `class_balance_values` re-derives the
    per-file weight straight from `listing`'s own records, which is the same source col1
    was built from in the first place."""

    listing: list[dict[str, str]]
    files_values: NDArray[np.float64]
    batches: Batches
    workdir: Path
    mapping_path: Path
    algo: int
    last_index: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def next_listing(self) -> Path:
        """Draw one mini-batch and write it as a weighted listing to `workdir/_batch.lst`
        (overwriting the prior step's -- read immediately by the fresh engine, no race).

        Selection = `unique(new_batch U worstCases)` (`ComputeGradient.m:55`
        `count_unique([new_batch;worstCases])`). ORDER is ascending file index
        (`np.unique`), a documented simplification of the legacy's `sortrows(filesValues,
        -3)` re-sort -- the aggregate corpus cost is order-independent (compute_cost sums
        over the config's rows; `aggregate_workers` re-sorts by id).

        WEIGHT column (Phase 5, F3, IMPROVEMENTS.md -- FIXED, this commit):
        `ComputeGradient.m:74-76` gates the whole `:59-73` rescale block -- `algo < 5`
        (every non-LID algo) or `nbOfTargetClasses > 2` (a >2-class LID split) OVERRIDES
        `filesValues(:,2)` to a flat 1.0 for EVERY selected row, discarding both the raw
        listing weight and the rescale; only `algo >= 5` (LID) with `nbOfTargetClasses <= 2`
        (a binary target/non-target split) lets `class_balance_values`'s rescale survive
        into the written listing. This is a hard override, not a fallback: the raw listing
        weight is NEVER what gets written once batch mode is on, in either branch.

        The weighted listing's OTHER `%g` column (the 6th CSV field) is the record's
        file_id, NOT a duration -- `WriteWeightedListing.m` writes `listing{i}.duration`
        there but `Corpus::from_config` reads field 6 as file_id (a legacy field-role
        mismatch, IMPROVEMENTS), so the port round-trips file_id and the engine sees the
        same file_id the full corpus would."""
        batch, _ = get_new_batch(self.batches)
        parts = [np.asarray(batch, dtype=np.int64)]
        parts += [w.index for w in self.batches.worst_cases if w.index.size]
        idx = np.unique(np.concatenate(parts)) if any(p.size for p in parts) else np.zeros(0, dtype=np.int64)

        items = [self.listing[int(i)] for i in idx]
        if self.algo < 5 or self.batches.nb_target_classes > 2:
            weight_col = np.ones(idx.size, dtype=np.float64)
        else:
            weight_col = class_balance_values(items, self.mapping_path)
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
    batch: _BatchStep | _ModernBatchStep | None = None,
    seed_weights: list[NDArray[np.float64]] | None = None,
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
    when None, all steps run the passed engine's fixed full corpus (the 4c path).

    `seed_weights` (Phase 5 Task 8): when set, SMORMS3 starts from THESE weights instead of
    the engine's config-seeded pack -- the modern loop threads the current epoch's weights
    in (init or the previous epoch's trained weights) so each epoch continues from where the
    last left off. `None` preserves the 4c/4d behavior exactly (read the pack off `engine`)."""
    full = seed_weights if seed_weights is not None else engine.weights(0)  # type: ignore[attr-defined]
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
        runner = _BatchRunner(state.listing, fv, batches, workdir, _mapping_path(state), algo)

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


# ---- The modern training loop (Phase 5 Task 8) ------------------------------------------
#
# Distinct from `train` (the legacy QuantumPSO-outer + SMORMS3-inner over the vec2struct
# genome): `train_modern` is a from-scratch SEEDED-init loop -- SMORMS3 epochs over the
# (batch-mode) training corpus, per-epoch FORWARD-ONLY validation on a held-out listing,
# early-stop on validation-cost patience, best+last checkpoints. NO genome, NO QuantumPSO,
# NO LR schedule; each epoch runs a FRESH SMORMS3 from the current weights (so the weights
# fully capture the trainable state, which is what makes resume-from-`last_*.bin` clean).


def _net_names(algo: int) -> list[str]:
    """The per-net checkpoint suffixes: `[sad]` for the single-net algos, `[sad, lid]` for
    the algo-6 Twin (same net count as `_tail_lengths`)."""
    return ["sad", "lid"] if algo == 6 else ["sad"]


def _save_ckpt(ckpt_dir: Path, prefix: str, weights: list[NDArray[np.float64]], net_names: list[str]) -> None:
    """Write one `<prefix>_<net>.bin` per net (the column-major `.bin` codec, same as
    `train`'s `sad_weights.bin`)."""
    for name, w in zip(net_names, weights, strict=True):
        arr = np.asarray(w, dtype=F64)
        write_bin(arr.shape[0], 1, arr, ckpt_dir / f"{prefix}_{name}.bin")


def _resume(resume_from: Path, algo: int) -> tuple[list[NDArray[np.float64]], list[EpochRecord], float, int, int]:
    """Reload the checkpoint written by a prior `train_modern`: the `last_<net>.bin` weights
    + the `train_history.json` bookkeeping (history, best cost/epoch, epochs already run).
    Returns `(weights, history, best_val, best_epoch, start_epoch)`; `start_epoch` is the
    number of epochs already run, so the loop continues toward `params.epochs` (the TOTAL)."""
    rd = Path(resume_from)
    prev = ModernTrainResult.model_validate_json((rd / "train_history.json").read_text())
    weights = [np.asarray(read_bin(rd / f"last_{name}.bin")[2], dtype=F64) for name in _net_names(algo)]
    return weights, list(prev.history), prev.best_val_cost, prev.best_epoch, prev.epochs_run


def _init_weights_from_scratch(state: RunState, params: ModernTrainParams) -> list[NDArray[np.float64]]:
    """Seeded Xavier/He init (Task 3) for each net's flat pack -- `[sad]` for single-net
    algos, `[sad, lid]` for the Twin. One shared `Generator(init_seed)` draws both nets in
    order (deterministic)."""
    rng = np.random.default_rng(params.init_seed)
    prefixes = ["BLSTM"] + (["BLSTM_LID"] if state.ps.algo == 6 else [])
    weights: list[NDArray[np.float64]] = []
    for prefix in prefixes:
        weights += init_weights(nnet_spec(state.base_config, prefix), rng, params.init_scheme, params.forget_bias_one)
    return weights


def _modern_config_text(base: dict[str, str], algo: int, *, backprop: bool, fileslisting: str | None = None) -> str:
    """The base config with backprop toggled (both nets) -- ALWAYS `Epochs 0`, whether
    backprop is ON (one forward/backward at theta the Python SMORMS3 loop owns; the fold
    harvests the gradient into the F10 seam stash) or OFF (a pure forward scoring pass for
    validation). The `BackPropagationActivated` flag -- NOT Epochs -- gates whether the fold
    computes a gradient, so `engine.run()` fills the MultiConfigResults cost columns either
    way. UNLIKE `_eval_config_text`, this does NOT inject `CostPonderation` -- the modern
    loop has no genome, so the config's own cost law stands (or the engine's default when the
    key is absent).

    F11 (phase 5): `Epochs 0`, not `1`, for the backprop-ON case. `Epochs >= 1` routes
    `run()` through the engine-internal `train()` (3 folds + 2 Rprop), so `forward_backward`
    reported cost/gradient at engine-moved weights, not the input theta -- the modern loop
    then trained on a wrongly-anchored gradient (T10 discovery). See IMPROVEMENTS.md
    `[phase5] F11`."""
    cfg = dict(base)
    flag = "true" if backprop else "false"
    cfg["BLSTM_BackPropagationActivated"] = flag
    if algo == 6:
        cfg["BLSTM_LID_BackPropagationActivated"] = flag
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    if fileslisting is not None:
        cfg["fileslisting"] = fileslisting
    return "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"


@dataclass
class _ModernBatchStep:
    """The modern loop's per-step batch binding: `next_listing` rotates the shared
    `_BatchRunner`'s cursors, `make_engine` rebuilds a backprop-ON engine on that listing
    WITHOUT the genome ponderation injection (unlike `_BatchStep`). Threaded into
    `_backprop_inner` exactly like `_BatchStep` (`.next_listing()` + `.make_engine`)."""

    runner: _BatchRunner
    base: dict[str, str]
    algo: int

    def next_listing(self) -> Path:
        return self.runner.next_listing()

    def make_engine(self, listing_path: Path) -> object:
        import speech_rs  # local: the pyo3 module is only needed on the engine path

        text = _modern_config_text(self.base, self.algo, backprop=True, fileslisting=Path(listing_path).name)
        (self.runner.workdir / "_train_modern_batch.config").write_text(text)
        return speech_rs.Engine(["_train_modern_batch.config"], "-m")


def _confusion_error(results: NDArray[np.float64], algo: int) -> float | None:
    """The FIXED (F5) confusion misclassification rate on the validation results, Twin-only
    (single-net SAD algos carry no LID confusion -> None). Extracts the per-file LID score
    block (`Error_vad[:, 16:-2]`, the `>150`/`-200` in-band columns) and derives
    `1 - hits/trials` from `scoring.confusion_matrix`. Returns None on any degeneracy
    (algo != 6, <2 classes, empty/ill-formed block) -- it is a recorded side-metric, never
    the early-stop signal."""
    if algo != 6:
        return None
    sel = results[results[:, 1] == 1]
    if sel.shape[0] == 0 or sel.shape[1] <= 3:
        return None
    scores = np.asarray(sel[:, 3:][:, 16:-2], dtype=F64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        return None
    try:
        conf = confusion_matrix(scores, 50.0)
    except ValueError:
        return None
    class_nb = scores.shape[1]
    hits = float(sum(conf[k, k] for k in range(1, class_nb + 1)))
    trials = float(sum(conf[k, class_nb + 1] for k in range(1, class_nb + 1)))
    return None if trials <= 0.0 else 1.0 - hits / trials


def _make_default_train_epoch(state: RunState, params: ModernTrainParams, workdir: Path, tails: list[int], seed: int) -> TrainEpochFn:
    """The engine-backed per-epoch trainer: `steps_per_epoch` SMORMS3 steps over
    `forward_backward` (backprop ON), seeded from the CURRENT weights. When `minibatch > 0`
    a persistent `_BatchRunner` (built ONCE here, cursors rotating across epochs) drives the
    per-step hard-example batch listing; else every step folds the full training corpus.
    The `create_batches` shuffle uses the master `seed + 2` (matching `train`); weight init
    uses `params.init_seed` -- two independent streams."""
    base = state.base_config
    algo = state.ps.algo

    runner: _BatchRunner | None = None
    if params.minibatch > 0:
        fv = _files_values(state)
        batch_rng = np.random.default_rng(seed + 2)
        batches = create_batches(fv, params.minibatch, params.nb_worst, params.multilingual, params.nb_classes, batch_rng)
        runner = _BatchRunner(state.listing, fv, batches, workdir, _mapping_path(state), algo)

    def train_epoch(weights: list[NDArray[np.float64]], epoch: int) -> tuple[list[NDArray[np.float64]], float]:
        import speech_rs  # local: the pyo3 module is only needed on the engine path

        # chdir into the workdir: the engine reads its config + corpus paths relative to CWD
        # (the same seam `train`/`score_genome` use). Checkpoints are absolute, unaffected.
        with _chdir(workdir):
            batch = _ModernBatchStep(runner, base, algo) if runner is not None else None
            engine: object | None = None
            if batch is None:
                (workdir / "_train_modern.config").write_text(_modern_config_text(base, algo, backprop=True))
                engine = speech_rs.Engine(["_train_modern.config"], "-m")
            trained, hist = _backprop_inner(engine, params.steps_per_epoch, tails, batch=batch, seed_weights=weights)
        return trained, (float(hist[-1]) if hist else float("nan"))

    return train_epoch


def _make_default_validate(state: RunState, params: ModernTrainParams, workdir: Path) -> ValidateFn:
    """The engine-backed forward-only validator: build a backprop-OFF engine on
    `valid_listing` (defaults to the training listing when unset), set the current weights,
    run ONE scoring pass, and read back the validation cost + the FIXED confusion metric.

    `params.val_metric` selects the early-stop cost (Phase 6 Task 6, a Phase-5 carry-forward):

      * "nn_cost_seg" (DEFAULT): the forward-only NNCostSeg objective, assembled off the
        results matrix EXACTLY as `engine.forward_backward` assembles its `f`
        (`f = NNCostSeg (+ NNCostLID for the algo-6 Twin)`, engine.py:378-385) -- so the
        validation signal is the held-out value of the SAME quantity SMORMS3 descends in
        training. A CONTINUOUS signal that moves from scratch. No gradient is harvested
        (`weights_derivatives` is never read): forward-only means the seg/LID-cost columns
        the cost reads suffice, and the engine's cost block accumulates them on ANY forward
        fold with references present (`BLSTMNeuralNetwork.cpp` gates the cost on
        `target.rows() > 0`, NOT on backprop), so the backprop-OFF config here still fills
        them.
      * "balance": the phase-5 balance-law cost (`compute_cost`) -- preserved verbatim. On
        the from-scratch SAD fixture balance-5 mode-0 collapses to the discrete `100-success`
        error rate, stuck at 30.0 (the seeded net's posteriors never cross the decision
        threshold), a useless early-stop plateau -- which is why nn_cost_seg is the default.

    Reuses `_modern_config_text(..., backprop=False)` (the established forward-only builder) --
    NO third config builder: both metrics run the identical backprop-OFF fold on
    `valid_listing`; only the cost read off the results differs."""
    base = state.base_config
    algo = state.ps.algo
    balance = state.balance
    bbp = state.ps.BalanceBackProp
    valid_listing = params.valid_listing if params.valid_listing is not None else base["fileslisting"]
    metric = params.val_metric

    def validate(weights: list[NDArray[np.float64]], epoch: int) -> tuple[float, float | None]:
        import speech_rs  # local: the pyo3 module is only needed on the engine path

        with _chdir(workdir):
            (workdir / "_valid_modern.config").write_text(_modern_config_text(base, algo, backprop=False, fileslisting=valid_listing))
            engine = speech_rs.Engine(["_valid_modern.config"], "-m")
            engine.set_weights(0, [list(np.asarray(w, dtype=F64)) for w in weights])
            engine.run()
            results = np.asarray(engine.results_matrix(), dtype=F64)
        if metric == "nn_cost_seg":
            error_vad = _error_vad(results, 1)
            val_cost = _nn_cost_seg(error_vad)
            if algo == 6:
                val_cost += _nn_cost_lid(error_vad)  # mirror forward_backward's f = NNCostSeg + NNCostLID (Twin)
        else:
            val_cost, _ = compute_cost(results, 1, balance, CostParams(mode=0, balance_backprop=bbp, algo=algo))
        return float(val_cost), _confusion_error(results, algo)

    return validate


def train_modern(
    state: RunState,
    seed: int,
    params: ModernTrainParams,
    *,
    train_epoch: TrainEpochFn | None = None,
    validate: ValidateFn | None = None,
    init_weights_override: list[NDArray[np.float64]] | None = None,
) -> ModernTrainResult:
    """The MODERN training loop: seeded-init (or checkpoint-resume) -> SMORMS3 epochs over
    batch mode -> per-epoch forward-only validation -> early-stop on validation-cost
    patience -> best/last checkpoints (`best_<net>.bin`, `last_<net>.bin`, `train_history.json`).

    The engine work is behind two injectable hooks -- `train_epoch(weights, epoch) ->
    (new_weights, train_cost)` and `validate(weights, epoch) -> (val_cost, confusion_error)`
    -- each defaulting to an engine-backed implementation. The unit tests pass STUBS, so the
    epoch/validation/early-stop/checkpoint/resume STATE MACHINE is exercised with no engine.

    Init: `resume_from` (load `last_<net>.bin` + `train_history.json`) takes precedence, then
    `init_weights_override` (test injection), then from-scratch seeded init. `epochs` is the
    TOTAL target -- a resumed run continues toward it, appending epochs.

    Early-stop: after `patience` epochs with no strict improvement in `val_cost`, stop.
    `best_<net>.bin` holds the argmin-val_cost epoch's weights; `last_<net>.bin` the latest."""
    algo = state.ps.algo
    net_names = _net_names(algo)
    ckpt_dir = Path(state.out_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(state.config_path).parent

    if params.resume_from is not None:
        weights, history, best_val, best_epoch, start_epoch = _resume(Path(params.resume_from), algo)
    elif init_weights_override is not None:
        weights = [np.asarray(w, dtype=F64) for w in init_weights_override]
        history, best_val, best_epoch, start_epoch = [], float("inf"), -1, 0
    else:
        weights = _init_weights_from_scratch(state, params)
        history, best_val, best_epoch, start_epoch = [], float("inf"), -1, 0

    if train_epoch is None:
        train_epoch = _make_default_train_epoch(state, params, workdir, _tail_lengths(state.base_config, algo), seed)
    if validate is None:
        validate = _make_default_validate(state, params, workdir)

    def _write_history(stopped: bool) -> ModernTrainResult:
        res = ModernTrainResult(
            best_epoch=best_epoch,
            best_val_cost=float(best_val),
            epochs_run=len(history),
            stopped_early=stopped,
            history=history,
            checkpoint_dir=str(ckpt_dir),
        )
        (ckpt_dir / "train_history.json").write_text(res.model_dump_json(indent=2))
        return res

    stopped_early = False
    for epoch in range(start_epoch, params.epochs):
        weights, train_cost = train_epoch(weights, epoch)
        val_cost, conf_err = validate(weights, epoch)

        _save_ckpt(ckpt_dir, "last", weights, net_names)
        is_best = val_cost < best_val
        if is_best:
            best_val, best_epoch = val_cost, epoch
            _save_ckpt(ckpt_dir, "best", weights, net_names)

        history.append(EpochRecord(epoch=epoch, train_cost=float(train_cost), val_cost=float(val_cost), confusion_error=conf_err, is_best=is_best))
        _write_history(stopped=False)  # per-epoch, so a resume sees the latest bookkeeping

        # Early-stop: `patience` epochs since the best (best_epoch < 0 only if val never finite).
        if best_epoch >= 0 and epoch - best_epoch >= params.patience:
            stopped_early = True
            break

    return _write_history(stopped=stopped_early)


# ---- The narrowed outer search (Phase 5 Task 9) -----------------------------------------
#
# The modern regime (user-locked, 2026-07-10): network WEIGHTS train by gradient (the modern
# loop above), so the outer QuantumPSO search is NARROWED to the DSP/config hyperparameter
# genome ONLY -- the weight-block + normalize-tail dims are pinned OUT of the genome permanently
# by `build_hyperparam_mask`. This is a DELIBERATE, documented break from the legacy genome (which
# carried the weights in-band by design); the legacy `train` above is byte-untouched as the
# legacy-regime driver. See IMPROVEMENTS.md `phase5-qpso-nonweight-genome`.


def build_hyperparam_mask(ps: RunConfig) -> dict[str, object]:
    """The PERMANENT weight-block + normalize-tail vec2struct mask: every network weight/bias
    matrix (`_nn_block`/`_output_neuron`) and the NormalizeInputMean/Std tail pinned out of the
    QPSO genome, so the outer search touches ONLY DSP/config hyperparameters. Derived from the
    walk (`genome.weight_block_mask`) -- no magic dim counts, so it tracks the net architecture
    automatically. Consumed by `masking_validation` (the round-trip gate) and by vec2struct (the
    per-candidate decode)."""
    mask, _searchable = weight_block_mask(ps)
    return mask


def _hyperparam_config_text(base: dict[str, str], config_struct: dict[str, str], algo: int) -> str:
    """Overlay the FULL vec2struct-decoded non-weight config (`config_struct` -- printConfig has
    already stripped the weight/normalize matrices) onto the byte-known-good base config,
    FORWARD-ONLY (backprop OFF, `Epochs 0`) so the search scores the FIXED base weights against
    the genome's DSP hyperparameters (no engine-internal training). The generalized sibling of
    `_eval_config_text` (which injects only the 2 `CostPonderation` fields): here EVERY searchable
    DSP key -- freq bands, windows, LTSV/TDC, decision thresholds, calibration laws, ponderations
    -- is injected. `dict(base)` preserves base key positions (byte-stable ordering, the same trick
    `_eval_config_text` uses); decoded keys overwrite in place / append. The weights reach the
    engine through the base config's committed `.bin` pack (`weightsFile`), untouched."""
    cfg = dict(base)
    for k, v in config_struct.items():
        cfg[k] = v
    cfg["BLSTM_BackPropagationActivated"] = "false"
    if algo == 6:
        cfg["BLSTM_LID_BackPropagationActivated"] = "false"
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    return "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"


# A candidate whose decoded DSP config drives the FIXED base net into an invalid region (a
# feature dimension the net was not sized for, or an engine path the port typed-bails, e.g. the
# Twin's Mode-7 pitch pass) is PENALIZED, not fatal -- the from-scratch search must steer away
# from invalid subregions of the DSP space, not crash on them. The penalty exceeds any real
# balance-law cost (~O(100)), so an invalid particle can never become the QPSO gbest.
_HYPERPARAM_PENALTY = 1.0e6


def _decode_hyperparam(
    state: RunState,
    reduced: NDArray[np.float64],
    mask: dict[str, object],
    searchable: NDArray[np.bool_],
) -> tuple[str, NDArray[np.float64]]:
    """The PURE (engine-free, never-crashing) half of a candidate eval: scatter the searchable-space
    genome `reduced` into the full vec2struct genome (masked weight dims filled with 0 -- the mask
    overrides them regardless), decode WITH the permanent weight mask, and inject the full non-weight
    config onto the base. Returns `(injected_config_text, reduced_out_param)` -- the searchable-space
    out_param (the sortrows/clamp write-back at the searchable positions; the masked positions carry
    only the mask inverse-encode and are dropped)."""
    full = np.zeros(searchable.shape[0], dtype=F64)
    full[searchable] = np.asarray(reduced, dtype=F64)
    config_struct, out_param, _ = vec2struct(full, mask, state.ps, 0)
    text = _hyperparam_config_text(state.base_config, config_struct, state.ps.algo)
    return text, out_param[searchable].copy()


def score_hyperparam_genome(
    state: RunState,
    reduced: NDArray[np.float64],
    mask: dict[str, object],
    searchable: NDArray[np.bool_],
    workdir: Path,
) -> tuple[float, NDArray[np.float64], str]:
    """One narrowed-QPSO candidate's forward-only cost (the `cost_fn` per-particle body). Decode the
    searchable-space genome (`_decode_hyperparam`), then score the engine fold with FIXED base weights
    (no inner SMORMS3, no engine-internal training -- backprop OFF, `Epochs 0`). Returns
    `(cost, reduced_out_param, injected_config_text)`; the config text is returned for the non-vacuity
    pin (distinct hyperparameters -> distinct engine configs). Self-contained (chdirs into `workdir`
    itself), like `score_genome`. Raises through any engine failure -- `train_hyperparam_search`'s
    cost_fn is the layer that penalizes an invalid candidate."""
    import speech_rs  # local: the pyo3 module is only needed on the engine path

    text, reduced_out = _decode_hyperparam(state, reduced, mask, searchable)
    with _chdir(workdir):
        (workdir / "_hyperparam_eval.config").write_text(text)
        engine = speech_rs.Engine(["_hyperparam_eval.config"], "-m")
        engine.run()
        results = np.asarray(engine.results_matrix(), dtype=F64)
    cost, _ = compute_cost(results, 1, state.balance, CostParams(mode=0, balance_backprop=state.ps.BalanceBackProp, algo=state.ps.algo))
    return float(cost), reduced_out, text


def train_hyperparam_search(
    state: RunState,
    seed: int,
    *,
    qpso_particles: int = 24,
    qpso_epochs: int = 100,
    seed_value: NDArray[np.float64] | None = None,
) -> TrainResult:
    """The NARROWED outer search (Phase 5 Task 9): QuantumPSO over the DSP/config hyperparameter
    genome ONLY -- network weights pinned out by `build_hyperparam_mask`, trained separately by
    gradient (`train_modern`). Each candidate is scored FORWARD-ONLY against the fixed base weights
    (`score_hyperparam_genome`); there is NO inner SMORMS3 and NO `backprop_refine` (the modern
    regime does not train weights inside the search, `backprop_activated=0`). Deterministic under a
    single fixed-seed numpy `Generator`. Checkpoints the gbest searchable genome + the QPSO cost
    history + the best decoded engine config (feed it to `train_modern` for the weight training).

    Distinct from the legacy `train` (byte-untouched, `mask=None` -> the full weight-carrying genome
    + 2-key `CostPonderation` injection): this searches the PERMANENTLY narrowed genome.

    `qpso_epochs` must be >= 2: the QDPSO exploration/contraction coefficient schedule
    (`QuantumPSO.m:415`, `coef_exp_contr = 0.75 - 0.5*(i-1)/(me-1)`, transcribed in
    `optimizers.quantum_pso`) divides by `me - 1`, so `qpso_epochs == 1` would raise a bare
    `ZeroDivisionError` deep inside the QPSO loop; guarded here with a clear message instead.

    Penalized-eval observability (Phase 5 Task 9 fix wave): a per-candidate eval that hits an
    invalid DSP subregion (feature-dim mismatch against the fixed base net, a typed-bailed engine
    path) is caught and penalized rather than raised -- see `cost_fn` below -- but the SAME catch
    also swallows a genuine engine defect, so the returned `TrainResult.penalized_evals`/
    `penalized_types` (a `type(exc).__name__` breakdown) let a caller tell the two apart. A run
    that ends with `gbestval >= _HYPERPARAM_PENALTY` (every eval penalized) or with over half its
    evals penalized emits a `warnings.warn`."""
    if qpso_epochs < 2:
        raise ValueError("qpso_epochs must be >= 2 (the legacy (i-1)/(me-1) schedule)")
    ps = state.ps
    mask, searchable = weight_block_mask(ps)
    n_search = int(searchable.sum())

    # MaskingValidation gate on the narrowed mask (Train_BLSTM.m:621-626 analogue): the mask must
    # round-trip. Full-length param (matching `train`'s size=genome_length convention), its own rng.
    gate_rng = np.random.default_rng(seed + 1)
    if masking_validation(gate_rng.uniform(0.0, ps.adim, size=genome_length(ps)), mask, ps):
        raise ValueError("MaskingValidation failed: the narrowed hyperparameter mask does not round-trip on this ps")

    workdir = Path(state.config_path).parent
    ckpt_dir = Path(state.out_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Penalized-eval observability (fix wave): counts + exception-type breakdown across every
    # candidate this run scores, surfaced on the returned TrainResult so a genuine engine defect
    # can be told apart from a legitimately-invalid DSP subregion (both currently land here).
    penalized_counts: dict[str, int] = {}
    total_evals = 0

    def cost_fn(positions: NDArray[np.float64], _mode: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        nonlocal total_evals
        pos = np.atleast_2d(np.asarray(positions, dtype=F64))
        costs = np.empty(pos.shape[0], dtype=F64)
        out = np.empty_like(pos)
        for i in range(pos.shape[0]):
            total_evals += 1
            try:
                cost, reduced_out, _text = score_hyperparam_genome(state, pos[i], mask, searchable, workdir)
            except BaseException as exc:  # noqa: BLE001 -- must catch the pyo3 PanicException (a direct BaseException subclass)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                # invalid DSP config (feature-dim mismatch / a typed-bailed engine path): penalize so
                # QPSO steers away. out_param is the PURE vec2struct write-back (decode never crashes).
                cost = _HYPERPARAM_PENALTY
                _text, reduced_out = _decode_hyperparam(state, pos[i], mask, searchable)
                penalized_counts[type(exc).__name__] = penalized_counts.get(type(exc).__name__, 0) + 1
            costs[i] = cost
            out[i] = reduced_out
        return costs, out

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
        vr=np.column_stack([np.zeros(n_search), np.full(n_search, ps.adim)]),
        mv=np.full(n_search, ps.adim / 2.0),
        seed_value=seed_value,
        backprop_activated=0,  # no weight refinement in the narrowed search
    )

    with _chdir(workdir):
        result = quantum_pso(cost_fn, params, n_search, np.random.default_rng(seed))
        gbest = np.asarray(result.gbest, dtype=F64)
        if result.gbestval >= _HYPERPARAM_PENALTY:
            # a fully-penalized run: gbest sits in the SAME invalid region every candidate did
            # (gbestval can only reach the penalty if NO eval ever beat it), so re-scoring it
            # through the engine would raise the identical way -- skip straight to the pure,
            # never-crashing decode instead of crashing on the checkpoint step.
            best_text, _out = _decode_hyperparam(state, gbest, mask, searchable)
        else:
            # the best decoded engine config (feed to train_modern for the weight training)
            _cost, _out, best_text = score_hyperparam_genome(state, gbest, mask, searchable, workdir)

    penalized_evals = sum(penalized_counts.values())
    penalized_fraction = (penalized_evals / total_evals) if total_evals else 0.0
    if result.gbestval >= _HYPERPARAM_PENALTY:
        warnings.warn(
            f"train_hyperparam_search: gbestval ({result.gbestval:g}) >= the penalty ({_HYPERPARAM_PENALTY:g}) -- "
            f"an all-penalized/degenerate search, {penalized_evals}/{total_evals} evals penalized "
            f"(types={penalized_counts!r}). This may be a genuine engine defect rather than an "
            "invalid DSP region -- inspect penalized_types before trusting this run.",
            stacklevel=2,
        )
    elif penalized_fraction > 0.5:
        warnings.warn(
            f"train_hyperparam_search: {penalized_evals}/{total_evals} evals ({penalized_fraction:.0%}) "
            f"were penalized (types={penalized_counts!r}) -- over half the searched DSP hyperparameter "
            "space was invalid; inspect penalized_types before trusting this run.",
            stacklevel=2,
        )

    cost_hist = np.asarray(result.gbestval_traj, dtype=F64)
    write_bin(gbest.shape[0], 1, gbest, ckpt_dir / "gbest.bin")
    write_bin(cost_hist.shape[0], 1, cost_hist, ckpt_dir / "cost_history.bin")
    (ckpt_dir / "best_hyperparam.config").write_text(best_text)

    tr = TrainResult(
        gbest=[float(x) for x in gbest],
        gbestval=float(result.gbestval),
        cost_history=[float(x) for x in cost_hist],
        inner_cost_history=[],
        checkpoint_dir=str(ckpt_dir),
        penalized_evals=penalized_evals,
        penalized_types=dict(penalized_counts),
    )
    (ckpt_dir / "checkpoint.json").write_text(tr.model_dump_json(indent=2))
    return tr
