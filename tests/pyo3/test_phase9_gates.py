"""Phase 9 Task 8 (spec S8.2): the FOUR from-scratch subset gates -- {sLSTM, Mamba} x
{bidirectional, forward} on the SAD arm, through the T4 `--cell-type`/`--direction` knobs.

Same protocol as the phase-6 SAD gate (`test_phase6_gates.py::test_sad_*`, the template):
corpus-gated + local-only, seeded, < 10 min each, run-twice bit-identical, and the HARD leg
is the TASK metric -- the trained model's pooled held-out DCF must beat its OWN from-scratch
init's on the identical test set. Every gate here uses the phase-6 SAD recipe VERBATIM
(subset 10 / valid 8 / test 24, 3 epochs x 10 steps, 20 s audio cap, seed 0) so the
vs-BLSTM comparison below is apples-to-apples: only the cell and the direction differ.

=====================================================================================
PARAM MATCH (spec S8.2 "+-15%"): ALREADY HOLDS AT THE S6 DEFAULTS -- no resizing needed
=====================================================================================
The arm's topology is fixed by `lre_sad.toml` (`LSTMNeuronNb 23,24,24` +
`LSTMSubSampling 4,1` -> TWO cell layers, layer 0 fed `4*23 = 92` stacked frames;
`OutputNeuronNb 48,12,1`; a `2*23 = 46`-element normalize tail). Swapping the cell keeps
that topology and only changes the per-layer block, so the totals fall out as:

    per-direction stack   = sum over the 2 cell layers of the cell's own block
    output MLP            = 48*12 + 12 + 12*1 + 1 = 601   (bidirectional, input 2*24)
                          = 24*12 + 12 + 12*1 + 1 = 313   (forward, input 1*24 -- the
                            `cell_overlay` output-width rewrite 48,12,1 -> 24,12,1)
    pack                  = D * stack + MLP + 46,   D = 2 (bidirectional) or 1 (forward)

    LSTM  layer = 4*out*(out+in) + 12*out + 4*out   (4 gate matrices, the 12-row peephole
                  bundle, 4 biases)      -> 11520 + 4992 = 16512 per direction
    sLSTM layer = 4*out*(out+in+1)                  (S2.2: [R_a | W_a | b_a], a in i,f,o,z)
                  4*24*117 + 4*24*49    -> 11232 + 4704 = 15936 per direction
    Mamba layer = [P|p iff in!=out] + g + W_in + conv + b_conv + W_x + W_dt + b_dt
                  + A_log + D + W_out   (S3.2, d_inner = 2*24 = 48, d_state 16, d_conv 4,
                  dt_rank auto = ceil(24/16) = 2)
                  -> 8544 (layer 0, carries the 24x92 input projection) + 6312 = 14856

| cell  | bidirectional | vs BLSTM | forward | vs BLSTM |
|-------|---------------|----------|---------|----------|
| LSTM  | 33671         | --       | 16871   | --       |
| sLSTM | 32519         | -3.42%   | 16295   | -3.41%   |
| Mamba | 30359         | -9.84%   | 15215   | -9.82%   |

Both cells land INSIDE +-15% at the spec's own default geometry, so this file pins the
lengths (T4 cross-checked each against the Rust `nb_of_weights()`) rather than resizing
anything. 33671 is also the 2015 production tuple-A pack size -- an independent anchor.

CAVEAT, measured (RESULTS.md carries the arithmetic): these are NOMINAL lengths. The arm
inherits a block of structurally DEAD layer-0 fan-in weights from the 2015 config -- fan-in
is sized `4*NNetInputSize = 92` while the DSP front-end produces an 11-wide feature vector,
so 48 of every 92 columns are never read -- and the three cells do NOT share it
proportionally (it is a `4 gates x out x 48` block in the gate cells, a single `out x 48`
input projection in Mamba). On LIVE (nonzero-gradient) weights the ordering inverts:
LSTM 24409, sLSTM 23257, Mamba 28009. Pre-existing, not a phase-9 regression, untouched.

=====================================================================================
THE PREFLIGHT LEG (`test_init_is_trainable`): the T5 log-law hazard, asserted
=====================================================================================
This arm trains under `CostLaw log/log`, and `Law::Log`'s forward is
`b + a*ln(clamp(y/adim, 1e-24, 1))` -- CONSTANT wherever that argument clamps, so F7's
consistent derivative there is EXACTLY ZERO. A from-scratch net whose output saturates is
therefore a PERMANENT STALL under this law, not noisy descent, and it would look like a
flat cost curve rather than a crash. The preflight leg asserts, at the from-scratch theta
BEFORE any training: the pack length, a finite initial NNCostSeg sitting in the log law's
INTERIOR (measured 0.269-0.320, i.e. ~0.5% of the saturated-forward constant
`-ln(1e-24) = 55.262`, worst 0.58%) both in aggregate and for every INDIVIDUAL file (worst
per-file normalized cost 0.442, 0.80% of the clamp -- the aggregate is frame-weighted, so it
alone could hide a saturated minority), and a strictly POSITIVE analytic gradient L2 norm at
epoch 0 (measured 0.357-0.722). A saturated init would trip an interiority pin or the
gradient pin. Whole-vector norms only: sLSTM's `b_i` is structurally non-identifiable
(analytically zero gradient by construction, proven in T2; the COMPUTED value is a
cancellation residue -- `slstm.rs:1023` measures max |dL/db_i| = 6.25e-17, not a bit-exact
zero, which is why `b_i`'s slots are EXCLUDED from the `_CONFIGS` dead-count floors rather
than counted into them), so no per-element gradient assert is meaningful.

=====================================================================================
vs BLSTM -- RECORDED, NOT GATED (spec R5, verbatim)
=====================================================================================
"R5 honest comparison: subset vs-BLSTM numbers are reported as-is with the thinness
caveat; no architecture-superiority claim is made from subset scale."

All four cells land at the SAME held-out DCF as the phase-6 BLSTM arm (0.2500 at every
collar, vs the 0.7500 all-non-speech init) because on a 10-file subset every from-scratch
SAD net mode-collapses to the window majority -- the documented phase-6 behavior ("the net
jumps from all-non-speech straight to all-speech"). A tie at 0.2500 is what an identical
collapse looks like, NOT evidence of architectural equivalence, so this file asserts the
beat-init direction and RECORDS the tie in `RESULTS.md` without gating on it. The
full-corpus user-fired runs are the referee.

Corpus-gated (`requires_corpus`) + pyo3 (`importorskip`): runs locally for implementers AND
reviewers, skips in CI. Measured whole-file runtime ~3.3 min on the dev box (2026-07-28,
Apple M4 Pro): 0.7 s for the four preflight legs, 146 s for the four hard gates, 48 s for
the four determinism legs. The slowest single gate is 67 s -- an order of magnitude inside
the < 10 min per-gate budget (spec R3).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import CORPUS_ROOT, GateRecorder, requires_corpus

pytest.importorskip("speech_rs")

from speech.drivers import baseline as B  # noqa: E402 -- after importorskip, matching the pyo3-suite convention
from speech.drivers.spec import BaselineSpec  # noqa: E402
from speech.drivers.state import ModernTrainParams, RunState  # noqa: E402
from speech.drivers.train import _init_weights_from_scratch  # noqa: E402
from speech.engine import average_derivs  # noqa: E402
from speech.fold_run import FoldRun  # noqa: E402

# (cell, direction, expected pack length, structurally-dead weight count) -- the S8.2 matrix.
# Lengths: T4 cross-pinned each against the Rust `BlstmNetwork::nb_of_weights()`.
# Dead counts: `D * blocks * out * 48 + 46` (see the dead-column caveat above) -- `D` = 2
# bidirectional / 1 forward, `blocks` = 4 gate matrices per sLSTM layer but a SINGLE input
# projection per Mamba layer, `out` = 24, 48 = the never-read fan-in columns, 46 = the frozen
# normalize tail. Only LAYER 0 contributes (layer 1's fan-in is the previous layer's output,
# fully live). This is the committed guard for the phase's dead-column claim.
_CONFIGS = [
    ("slstm", "bidirectional", 32519, 2 * 4 * 24 * 48 + 46),  # 9262
    ("slstm", "forward", 16295, 1 * 4 * 24 * 48 + 46),  # 4654
    ("mamba", "bidirectional", 30359, 2 * 1 * 24 * 48 + 46),  # 2350
    ("mamba", "forward", 15215, 1 * 1 * 24 * 48 + 46),  # 1198
]
_IDS = [f"{cell}-{direction}" for cell, direction, _, _ in _CONFIGS]
# Slack above the structural dead-count floor. The count is a FLOOR, not an equality: a live
# weight's gradient can land on exactly 0.0 by coincidence, and one measurably does -- the
# sLSTM/forward pack reads 4654 (== the floor, extra 0 / missing 0 against the predicted index
# SET) at this test's probe recipe but 4655 at a wider one (subset 4 / 20 s cap). So an `==`
# pin would be a data- and libm-dependent flake. The floor is the load-bearing direction
# (every predicted-dead weight IS dead); the slack keeps the guard two-sided so a pathological
# jump in dead weight still fails. 64 is 64x the one coincidence ever observed, but it is a
# LOOSE bound, not a tight one: on the smallest floor (mamba/forward, 1198) it is 5.3%, and it
# exceeds that config's per-column granularity of 24 -- so it would absorb up to two whole dead
# input columns there before firing. It catches a pathological jump, not a subtle one.
_DEAD_SLACK = 64

# The phase-6 SAD subset-gate recipe, VERBATIM (`test_phase6_gates.py::test_sad_subset_*`)
# -- shared so the vs-BLSTM comparison differs only in cell/direction.
_GATE = dict(subset=10, valid_size=8, test_size=24, epochs=3, steps_per_epoch=10, patience=99, seed=0, audio_max_duration=20.0)
# The phase-6 determinism recipe, VERBATIM (determinism is a pipeline property, provable on
# a short run; the full-recipe run-twice bit-identity was measured separately, see RESULTS).
_DET = dict(subset=8, valid_size=6, test_size=8, epochs=1, steps_per_epoch=8, patience=99, seed=0, audio_max_duration=15.0)

# `Law::Log`'s saturated-forward constant: `cost = b + a*ln(1e-24)` once the clamp bites.
# The initial cost must sit in the law's INTERIOR, orders of magnitude below this.
_LOG_CLAMP = float(-np.log(1e-24))  # 55.26204...


# ------------------------------------------------------------------------------------- #
# The preflight probe: one forward+backward at the from-scratch theta, no training.
# ------------------------------------------------------------------------------------- #


@dataclass
class _Rec:
    val_cost: float
    train_cost: float


@dataclass
class _Shim:
    """The minimal `ModernTrainResult` surface `run_baseline` reads back (it logs the
    costs and takes `checkpoint_dir`); the probe never trains, so the history is a
    single record carrying the measured init cost."""

    checkpoint_dir: str
    history: list[_Rec] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = -1
    best_val_cost: float = float("nan")


def _probe(out: dict[str, float]) -> Callable[[RunState, int, ModernTrainParams], _Shim]:
    """A `run_baseline(_train_fn=...)` stub that measures the from-scratch init INSTEAD of
    training: it re-uses the training loop's OWN seeded-init and config builders
    (`_init_weights_from_scratch`, a gradient `FoldRun` on the base config), so what it probes
    is exactly the theta and the fold epoch 0 would start from -- not a re-derivation that
    could silently drift from the real path."""

    def probe(state: RunState, seed: int, params: ModernTrainParams) -> _Shim:
        weights = _init_weights_from_scratch(state, params)
        # ONE gradient fold at the init: `nn_cost` is forward_backward's `f` on this single-net
        # arm and `average_derivs` its count normalization; the channel results come with it.
        res = FoldRun(state.base_config, Path(state.config_path).parent, backprop=True).run(weights)
        cost = res.nn_cost
        grads = [average_derivs(d) for d in res.derivs]
        results = res.results.for_config(0)

        grad = np.asarray(grads[0], dtype=np.float64)
        out["pack_len"] = float(len(weights[0]))
        out["init_cost"] = float(cost)
        out["grad_l2"] = float(np.linalg.norm(grad))
        out["grad_linf"] = float(np.max(np.abs(grad)))
        out["dead"] = float(np.count_nonzero(grad == 0.0))
        # PER-FILE normalized cost: the `seg_cost` / `seg_count` pair `_nn_cost_seg` sums,
        # read by name off the seam (issue #23). This file once read the wrong column off
        # the prefixed matrix (100*Pmiss); the `mean <= max` invariant in the test below is
        # what catches that class of slip.
        out["per_file_cost_max"] = float(np.max(results.seg_cost / np.maximum(1.0, results.seg_count)))
        ckpt = Path(state.out_dir) / "checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        return _Shim(checkpoint_dir=str(ckpt), history=[_Rec(float(cost), float(cost))], best_val_cost=float(cost))

    return probe


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize(("cell", "direction", "pack_len", "dead_expected"), _CONFIGS, ids=_IDS)
def test_init_is_trainable(tmp_path: Path, cell: str, direction: str, pack_len: int, dead_expected: int) -> None:
    """PREFLIGHT (the T5 log-law hazard + the T6 output-saturation hazard): before any
    training, the from-scratch init must be param-matched, cost-INTERIOR under the log law
    BOTH in aggregate and per file, and carry a NONZERO analytic gradient at epoch 0.
    Measured 2026-07-28, seed 0, AT THIS LEG'S OWN RECIPE (subset 2, 10 s cap -- an earlier
    revision published a mix of this leg's numbers and a wider exploratory run's; every value
    below now comes from the run this test performs). `%clamp` is the column left of it over
    `-ln(1e-24) = 55.262`:

    | config             | pack  | init NNCostSeg | %clamp | per-file max | %clamp | grad L2 | grad Linf | dead |
    |--------------------|-------|----------------|--------|--------------|--------|---------|-----------|------|
    | lstm-bi (baseline) | 33671 | 0.30034        | 0.543% | 0.42557      | 0.770% | 0.46050 | 0.20256   | 9262 |
    | lstm-forward       | 16871 | 0.26899        | 0.487% | 0.38775      | 0.702% | 0.43706 | 0.18776   | 4654 |
    | slstm-bi           | 32519 | 0.26942        | 0.488% | 0.37362      | 0.676% | 0.35736 | 0.18957   | 9262 |
    | slstm-forward      | 16295 | 0.27500        | 0.498% | 0.40237      | 0.728% | 0.37443 | 0.19126   | 4654 |
    | mamba-bi           | 30359 | 0.31969        | 0.578% | 0.41293      | 0.747% | 0.72178 | 0.18946   | 2350 |
    | mamba-forward      | 15215 | 0.31930        | 0.578% | 0.44149      | 0.799% | 0.49350 | 0.19512   | 1198 |

    No config's init sits near the saturated-forward constant (all ~0.5% of it, worst 0.58%),
    no INDIVIDUAL file does either (worst per-file 0.442, 0.80% of the clamp), and every
    gradient norm is far from zero -- so none of the four starts in the zero-gradient death the
    log law would otherwise make permanent. The `dead` column is the structurally-dead weight
    count, guarded here against the `_CONFIGS` prediction (the committed check on the
    dead-column finding). Cheap (~0.2 s per config: no valid/test split, no training, one fold
    over 2 files; ~0.7 s for all four)."""
    out: dict[str, float] = {}
    spec = BaselineSpec(
        arm="sad",
        corpus_root=CORPUS_ROOT,
        out_dir=tmp_path / "probe",
        subset=2,
        valid_size=0,
        test_size=0,  # no held-out scoring: this leg is about the init, not the task metric
        epochs=1,
        steps_per_epoch=1,
        patience=99,
        seed=0,
        audio_max_duration=10.0,
        cell=cell,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
    )
    B.run_baseline(spec, _train_fn=_probe(out))

    # (0) param match: the pack the arm actually seeds is the S8.2-sized one.
    assert out["pack_len"] == pack_len, f"{cell}/{direction} pack length {out['pack_len']:.0f} != the pinned {pack_len}"

    # (a) the initial cost is finite AND in the log law's interior, not pinned at its clamp --
    #     in AGGREGATE and, separately, for EVERY file (the frame-weighted aggregate could
    #     otherwise hide a saturated minority behind a healthy majority).
    assert np.isfinite(out["init_cost"]), f"init cost must be finite, got {out['init_cost']}"
    assert out["init_cost"] > 0.0, f"init cost must be positive, got {out['init_cost']}"
    # measured <= 0.320, i.e. <= 0.58% of the clamp constant; pin 5% -> ~8.6x headroom.
    assert out["init_cost"] < 0.05 * _LOG_CLAMP, f"init cost {out['init_cost']:.5f} sits near the log-law clamp {_LOG_CLAMP:.3f} (saturated init)"
    # SELF-CHECK FIRST: `init_cost` is a count-weighted MEAN of the per-file normalized costs,
    # so it can never exceed their max. If this fires, the two quantities are not the pair they
    # claim to be -- which is exactly how the first version of this pin shipped broken (it read
    # the prefixed matrix's column 4 = `res[1]` = 100*Pmiss, giving a "max" BELOW the mean and
    # an assert bounded by 100/count that could not fail).
    assert out["per_file_cost_max"] >= out["init_cost"] - 1e-9, (
        f"{cell}/{direction} per-file max {out['per_file_cost_max']:.6f} < aggregate mean {out['init_cost']:.6f}: "
        "the per-file quantity is not the cost (wrong channel_results field?)"
    )
    # per-file normalized worst case: measured <= 0.442 (0.80% of the clamp); same 5% pin
    # -> 6.3x headroom on the worst config (mamba/forward).
    assert out["per_file_cost_max"] < 0.05 * _LOG_CLAMP, (
        f"{cell}/{direction} worst per-file cost {out['per_file_cost_max']:.5f} sits near the log-law clamp {_LOG_CLAMP:.3f} (a saturated file)"
    )

    # (b) the analytic gradient at epoch 0 is NONZERO -- the zero-gradient death assert.
    #     Whole-vector norm (sLSTM's b_i is structurally non-identifiable, T2).
    #     Measured 0.356-0.727; pin 1e-3 -> ~350x headroom.
    assert out["grad_l2"] > 1e-3, f"{cell}/{direction} epoch-0 gradient norm {out['grad_l2']:.6f} is ~zero: the log-law saturation stall"
    assert np.isfinite(out["grad_linf"])

    # (c) the DEAD-COLUMN prediction (the header caveat's arithmetic), guarded. A FLOOR plus
    #     bounded slack, not an equality -- see `_DEAD_SLACK`. Falling BELOW the floor means
    #     fewer weights are dead than the fan-in-width arithmetic predicts; running far ABOVE
    #     it means weights died that the arithmetic does not account for.
    #     BLIND SPOTS, named rather than implied: this is a CARDINALITY check, so it cannot see
    #     (i) a pure block-order PERMUTATION of the pack (the dead set moves, the count does
    #     not -- S8.6 mutation 6 owns that), nor (ii) a compensating pair (n weights coming
    #     alive while n others die). The index-SET check that does see both was run during
    #     development (measured extra 0 / missing 0 for sLSTM in both directions, recorded in
    #     RESULTS + the task report) but is NOT committed here -- it needs a per-cell flat
    #     layout model, which would duplicate the packer.
    dead = int(out["dead"])
    assert dead >= dead_expected, f"{cell}/{direction} dead-weight count {dead} is below the structural floor {dead_expected}: the dead block moved"
    assert dead <= dead_expected + _DEAD_SLACK, f"{cell}/{direction} dead-weight count {dead} far exceeds the structural floor {dead_expected}"


# ------------------------------------------------------------------------------------- #
# The HARD gate: trained held-out DCF beats the model's OWN from-scratch init.
# ------------------------------------------------------------------------------------- #


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize(("cell", "direction", "pack_len", "dead_expected"), _CONFIGS, ids=_IDS)
def test_subset_gate_beats_own_init(tmp_path: Path, cell: str, direction: str, pack_len: int, dead_expected: int, gate_record: GateRecorder) -> None:
    """THE HARD LEG (spec S8.2 / S9.2): train this cell x direction from scratch on the
    10-file subset, then score a disjoint 24-file held-out slice END TO END through the T4
    DCF harness -- the trained pooled DCF must beat its own untrained init's, deterministically.
    Measured 2026-07-28, seed 0 (all five collars identical per row):

    | config        | trained DCF | Pmiss / Pfa     | init DCF | wall  |
    |---------------|-------------|-----------------|----------|-------|
    | slstm-bi      | 0.250000    | 0.0000 / 1.0000 | 0.750000 | ~41 s |
    | slstm-forward | 0.250000    | 0.0000 / 1.0000 | 0.750000 | ~16 s |
    | mamba-bi      | 0.250000    | 0.0000 / 1.0000 | 0.750000 | ~65 s |
    | mamba-forward | 0.249625    | 0.0000 / 0.9985 | 0.750000 | ~19 s |

    (phase-6 BLSTM, same recipe: 0.2500 vs 0.7500 -- RECORDED in RESULTS.md, NOT gated, R5.)
    The wall column is NOT a speed comparison: `--direction forward` also switches the
    windowing regime (`cell_overlay` forces `BLSTM_window 0`, the plain whole-sequence run,
    vs the bidirectional rows' windowed overlap), so the forward rows differ in TWO ways.

    The pins are the phase-6 SAD gate's, unchanged: both endpoints are degenerate
    (all-one-class collapse), so the DCF is essentially exactly 0.25/0.75 with no
    libm-sensitive boundary jitter. mamba-forward is the one row that is NOT exactly
    degenerate (Pfa 0.998498 -- it correctly rejects a sliver of non-speech), recorded as
    measured rather than rounded into the collapse story.

    The train/validation CE is NOT the signal here (the phase-6 lesson, and it bites harder
    on Mamba: its train cost ASCENDS 1.09 -> 2.07 across the three bidirectional epochs while
    the held-out DCF still lands at 0.25). Only the held-out TASK metric is gated."""
    t0 = time.time()
    res = B.run_baseline(BaselineSpec(arm="sad", corpus_root=CORPUS_ROOT, out_dir=tmp_path / "run", cell=cell, direction=direction, score_init=True, **_GATE))  # type: ignore[arg-type]
    wall = time.time() - t0

    # --- both DCF reports landed, valid ranges ---
    assert res.dcf is not None and res.init_dcf is not None
    tr = res.dcf.by_collar(0.5)
    ini = res.init_dcf.by_collar(0.5)
    assert np.isfinite(tr.dcf) and 0.0 <= tr.dcf <= 1.0 and 0.0 <= ini.dcf <= 1.0

    # --- the net learned to FIRE: trained detects held-out speech, the untrained init misses ~all ---
    assert tr.pmiss < 0.5, f"{cell}/{direction} trained must detect held-out speech (Pmiss {tr.pmiss:.3f})"
    assert ini.pmiss > 0.9, f"{cell}/{direction} untrained init misses ~all speech (Pmiss {ini.pmiss:.3f})"

    # --- THE HARD LEG: trained held-out DCF beats its OWN init by a margin, at EVERY collar ---
    #     (measured gap 0.50 at all five collars; pin a conservative 0.2.)
    for collar in (0.0, 0.25, 0.5, 1.0, 2.0):
        a, b = res.dcf.by_collar(collar), res.init_dcf.by_collar(collar)
        assert a.dcf < b.dcf - 0.2, f"{cell}/{direction} trained DCF {a.dcf:.4f} must beat init {b.dcf:.4f} at collar {collar}"
    assert tr.dcf <= 0.5, f"trained DCF {tr.dcf:.4f} must beat the all-non-speech baseline (0.75) with headroom"
    assert ini.dcf >= 0.6, f"untrained init should sit near the all-non-speech baseline, got {ini.dcf:.4f}"

    # --- the architecture knobs really took effect (not a silently-defaulted BLSTM run) ---
    assert (res.checkpoint_dir / "best_sad.bin").is_file() and (res.checkpoint_dir / "last_sad.bin").is_file()
    packs = {(res.checkpoint_dir / n).stat().st_size for n in ("best_sad.bin", "last_sad.bin")}
    assert packs == {16 + 8 * pack_len}, f"{cell}/{direction} checkpoint is not the {pack_len}-element pack: {packs}"

    # --- self-describing run: metadata + the dumped VRCTS hyps landed ---
    assert res.metadata_path.is_file()
    assert res.scores_dir is not None and len(list(res.scores_dir.glob("*.xml"))) == res.n_test
    assert wall < 600.0, f"{cell}/{direction} gate took {wall:.0f}s, over the 10 min budget"
    gate_record(res)


# ------------------------------------------------------------------------------------- #
# Determinism: run twice at a fixed seed -> bit-identical checkpoints.
# ------------------------------------------------------------------------------------- #


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize(("cell", "direction", "pack_len", "dead_expected"), _CONFIGS, ids=_IDS)
def test_deterministic(tmp_path: Path, cell: str, direction: str, pack_len: int, dead_expected: int) -> None:
    """Run-twice determinism at a fixed seed: bit-identical trained weight BYTES + identical
    pooled held-out DCF, per cell x direction. A SHORT config (the phase-6 pattern --
    determinism is a pipeline property, provable cheaply); the FULL-recipe run-twice
    bit-identity was measured separately (2026-07-28, mamba/bidirectional, identical
    `best_sad.bin` bytes and DCF)."""
    a = B.run_baseline(BaselineSpec(arm="sad", corpus_root=CORPUS_ROOT, out_dir=tmp_path / "a", cell=cell, direction=direction, **_DET))  # type: ignore[arg-type]
    b = B.run_baseline(BaselineSpec(arm="sad", corpus_root=CORPUS_ROOT, out_dir=tmp_path / "b", cell=cell, direction=direction, **_DET))  # type: ignore[arg-type]

    for name in ("last_sad.bin", "best_sad.bin"):
        pa, pb = (a.checkpoint_dir / name).read_bytes(), (b.checkpoint_dir / name).read_bytes()
        assert pa == pb, f"{cell}/{direction} {name} not bit-identical across two fixed-seed runs"
        assert len(pa) == 16 + 8 * pack_len, f"{name} is not the {pack_len}-element pack"
    assert a.dcf is not None and b.dcf is not None
    assert a.dcf.by_collar(0.5).dcf == b.dcf.by_collar(0.5).dcf, f"{cell}/{direction} pooled held-out DCF not reproducible"
