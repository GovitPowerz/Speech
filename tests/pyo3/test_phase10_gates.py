"""Phase 10 Task 4 (spec S3.2/S3.3): the from-scratch subset gates of the
`lre_sad_v2` lineage -- {LSTM, sLSTM, Mamba, CfC} x {bidirectional, forward} on the
`sad-v2` arm, through the `--cell-type`/`--direction` knobs. Phase 11 Task 9 (spec S8)
grew the matrix EIGHT -> TEN, adding the fifth cell, Transformer -- see "THE TRANSFORMER
ROWS" below for its own table, the init-quality anomaly it surfaced, and the user-ratified
`COLLAPSE_FLOOR_DCF` amendment (2026-08-12) that resolves it.

Same protocol as the phase-9 matrix (`test_phase9_gates.py`, itself the phase-6 SAD
gate's): corpus-gated + local-only, seeded, < 10 min each, run-twice bit-identical, and
the HARD leg is the TASK metric -- the trained model's pooled held-out DCF must beat its
OWN from-scratch init's on the identical test set, at EVERY collar. The phase-6 SAD recipe
is used VERBATIM (subset 10 / valid 8 / test 24, 3 epochs x 10 steps, 20 s audio cap,
seed 0), so every cross-row and cross-lineage comparison is apples-to-apples: only the
cell, the direction, and (vs phase 9) the CONFIG LINEAGE differ.

THE V1 FAMILY IS FROZEN AND STAYS GREEN. `test_phase9_gates.py` is untouched by this file;
`lre_sad.toml` gained a comment-only pointer to v2 (verified config_hash-safe -- the hash
reads the PARSED map, which TOML comments never enter).

=====================================================================================
THE LINEAGE, AND WHAT THE ONE VARIABLE BUYS (spec S3.1)
=====================================================================================
`lre_sad_v2.toml` is `lre_sad.toml` byte-for-byte except `nnet_input_size` 23 -> 11 and
`lstm_neuron_nb` `23,24,24` -> `11,24,24` (and the normalize tail those entail: 2*23 = 46
-> 2*11 = 22, written nowhere -- it self-sizes off the config on BOTH sides of the seam).
v1 declares a 23-wide input while its own DSP front-end emits 11 columns, so with
`lstm_sub_sampling 4` only 44 of layer 0's 92 fan-in columns carry data and the trailing 48
are structurally gradient-dead -- the 2015-inherited defect phase 9 measured. Here the
declared width IS the produced width: layer-0 fan-in is exactly `11*4 = 44`, all live.

MEASURED pack lengths (2026-08-01, LSTM/sLSTM/Mamba/CfC; 2026-08-12, Transformer --
phase-11 T9; every number below produced by `len(init_weights(...))` at the arm's own
overlaid config -- spec R4, no survey estimate survives, and BOTH Transformer rows agree
digit-for-digit with `tests/test_phase11_init.py`'s independent closed-form derivation at
the sized `d_ff = 64`: bidirectional `13871 + 196*d_ff`, forward `6959 + 98*d_ff` -- the
`d_ff`-dependent blocks (`W_1`/`b_1`/`W_2`/`b_2`) are purely per-stack, so the coefficient
exactly halves with the stack count while the constant term does not). The phase-9 v1
column is carried for contrast, NOT re-measured here:

| cell        | v2 bi | v2 fwd | v1 bi | v1 fwd | v2 bi vs v2 LSTM |
|-------------|-------|--------|-------|--------|------------------|
| LSTM        | 24431 | 12239  | 33671 | 16871  | --               |
| sLSTM       | 23279 | 11663  | 32519 | 16295  | -4.72%           |
| Mamba       | 28031 | 14039  | 30359 | 15215  | +14.74%          |
| CfC         | 24491 | 12269  | 28835 | 14453  | +0.25%           |
| Transformer | 26415 | 13231  | 28743 | 14407  | +8.12%           |

All five land inside the S8.2-style +-15% band of the same lineage's LSTM pack -- Mamba
only just (+14.74%), and that near-miss is the POINT: Mamba's layer-0 input projection is a
single `out x fin` adapter while the gate cells carry `4 x out x fin` (and CfC `B x fin`),
so halving `fin` shrinks the gate cells much harder than it shrinks Mamba. On v1 the same
ordering inverts once the dead block is discounted (phase 9's LIVE table: LSTM 24409 /
sLSTM 23257 / Mamba 28009). v2 makes NOMINAL == LIVE, which is the whole point of the fork.
Transformer's `W_a` width adapter is UNCONDITIONAL (unlike Mamba's `P`, present only when
`fin != out`), so it carries no v1-style dead-weight artefact of its own to discount --
its v1/v2 pack lengths are already directly comparable, nominal == live on BOTH lineages.

POSTURE (spec S3.4 / R5, stated so no reader infers more than was measured): v1 remains
the ONLY 2015-capacity-comparable lineage (its 33671 LSTM pack IS the production tuple-A
size). v2 is a NEW lineage with no 2015 counterpart. The corrected Xavier fan-in scaling
it enables (layer-0 `fan_in` 68 instead of 116, half of which carried no data) is a
MEASURABLE difference for the full-corpus launchers to adjudicate, NOT a
promised win, and nothing here is evidence of one -- see the vs-BLSTM note below.

=====================================================================================
THE INVERSE GUARD (spec S3.2): zero structurally-dead layer-0 INPUT columns
=====================================================================================
v1's gates pin dead-count FLOORS. v2's pin their mirror image: EVERY one of layer 0's 44
input columns must carry a nonzero gradient, in EVERY stack, for EVERY cell.

SCOPE, stated by BLOCK not by slack: this guard reads the layer-0 INPUT-PROJECTION block
only -- the weights that multiply input column `k`. Cell-level gradient-dead blocks that
are NOT input columns (sLSTM's non-identifiable `b_i`; Mamba's `A_log` at `T = 1`; CfC's
`W_bb` state columns at `T = 1` -- all three inert at this leg's `T`) are a separate,
already-pinned phenomenon and are excluded by never being addressed, not by a tolerance.

The per-cell offset arithmetic is derived in [`_input_column_offsets`] and its non-vacuity
is proven by [`test_v1_cfc_mechanical`], which runs the SAME machinery on the v1 arm and
finds exactly the 48 dead columns `[44, 92)` it must.

TWO-SIDED, because "nonzero" alone is too weak here: a structurally dead weight is EXACTLY
`0.0` (nothing is ever accumulated into it), but an analytically-zero-yet-computed weight
lands at cancellation scale -- measured, sLSTM's `b_i` reads ~1e-19 at this seam rather
than a bit-exact zero. So the guard asserts both the exact-zero property (the structural
claim) AND a magnitude floor `_LIVE_FLOOR = 1e-8`: ~1e11 above the cancellation class it
must reject, and ~2.9e4 below the smallest per-column magnitude ever measured here
(2.854e-04, lstm/bidirectional). It is a STRUCTURAL guard, not a tight numeric pin.

GUARD STRUCTURE, stated precisely so the leg count is not mistaken for independence: the
exact-zero leg is SUBSUMED by the floor leg (a column at exactly 0.0 has max|grad| 0.0 <
`_LIVE_FLOOR`, so `dead` is a strict SUBSET of `weak` and the floor leg alone would catch
everything the exact-zero leg catches). It is kept for its PRECISE failure message --
"structurally dead" and "gradient-inert" are different defects and should not report as one
-- not because it adds coverage. The genuinely independent third leg is the WHOLE-PACK
zero count (`_V2_FROZEN_TAIL`), which sees a dead block that MOVED somewhere the per-column
offsets never address.

=====================================================================================
THE PREFLIGHT LEG: the log-law hazard, asserted (phase-9 discipline, verbatim)
=====================================================================================
This arm trains under `CostLaw log/log`, and `Law::Log`'s forward is
`b + a*ln(clamp(y/adim, 1e-24, 1))` -- CONSTANT wherever that argument clamps, so F7's
consistent derivative there is EXACTLY ZERO. A from-scratch net whose output saturates is a
PERMANENT STALL under this law, not noisy descent, and it looks like a flat cost curve
rather than a crash. The preflight asserts, at the from-scratch theta BEFORE any training:
the pack length, a finite initial NNCostSeg in the log law's INTERIOR both in aggregate AND
for every INDIVIDUAL file (the frame-weighted aggregate alone could hide a saturated
minority), and a strictly positive analytic gradient L2 at epoch 0.

Per-file costs are read by slicing `results_matrix()`'s PREFIX off first (`[:, 3:]`,
exactly as `engine.py::_error_vad` does) and only then applying the per-res column
convention -- a `mean <= max` self-check precedes the bound so the pair cannot silently be
the wrong two quantities (the phase-9 file shipped that bug once).

=====================================================================================
vs BLSTM, and vs v1 -- RECORDED, NOT GATED (spec R5, verbatim)
=====================================================================================
"R5 honest comparison: subset vs-BLSTM numbers are reported as-is with the thinness
caveat; no architecture-superiority claim is made from subset scale."

On a 10-file subset every from-scratch SAD net mode-collapses to the window majority (the
documented phase-6 behavior: all-non-speech init -> all-speech trained), so rows tie at the
degenerate DCF and a tie is what an identical collapse looks like -- NOT evidence of
architectural or lineage equivalence. This file gates the beat-init DIRECTION only and
records the rest in `RESULTS.md`. The full-corpus user-fired runs are the referee, for the
cell comparison AND for the v1-vs-v2 lineage question.

=====================================================================================
THE TRANSFORMER ROWS (phase-11 T9, spec S8): ONE CLEAN, ONE RATIFIED VIA THE COLLAPSE FLOOR
=====================================================================================
`transformer/bidirectional` clears every leg cleanly, matching the pattern every other row
in this file sets: preflight interior (0.728% of the log-law clamp), zero dead/weak layer-0
columns, run-twice bit-identical, and THE HARD LEG passes at every collar with a healthy
~0.50 margin (trained DCF@0.5 0.250000 vs init 0.747406 -- see `test_subset_gate_beats_own_
init`'s table for the full five-collar breakdown).

`transformer/forward` does NOT clear the ORIGINAL margin-only criterion at two collars, and
this is recorded here in full rather than argued away or quietly widened (measured
2026-08-12, seed 0, the phase-6 recipe VERBATIM, run-twice bit-identical so the numbers
below are not noise):

| collar | trained dcf | init dcf | margin  | required | margin verdict | reaches floor? |
|--------|-------------|----------|---------|----------|-----------------|-----------------|
| 0.0    | 0.250000    | 0.476457 | 0.226457| >= 0.2   | OK              | yes (0.25<=0.2501)|
| 0.25   | 0.250000    | 0.465929 | 0.215929| >= 0.2   | OK              | yes             |
| 0.5    | 0.250000    | 0.455959 | 0.205959| >= 0.2   | OK              | yes             |
| 1.0    | 0.250000    | 0.444262 | 0.194262| >= 0.2   | FAIL            | yes -- PASSES via the floor |
| 2.0    | 0.250000    | 0.435836 | 0.185836| >= 0.2   | FAIL            | yes -- PASSES via the floor |

THE HEADLINE FINDING IS AN INIT-QUALITY ANOMALY, not a training failure: the TRAINED model
reaches DCF 0.250000 at every collar (Pmiss 0.0, Pfa 1.0) -- the exact all-speech collapse
point SEVEN of the other nine rows' trained models also reach (the remaining two, slstm-fwd
0.252430 and mamba-fwd 0.248770 at collar 0.5, are genuinely non-degenerate on the trained
side too but still clear their own margin comfortably, 0.494 and 0.489). Training is not in
question for THIS row: its trained result is indistinguishable from the typical row's. The
INIT side is where this row diverges from its nine siblings: every
other row's from-scratch Xavier init on a GATED RECURRENT cell collapses toward the
all-non-speech baseline (Pmiss > 0.9, most exactly 1.0, DCF ~0.75), which is WHY the
original `- 0.2` margin (sized against a ~0.50 measured gap on those rows) carried so much
headroom everywhere else. `transformer/forward`'s untrained init does NOT collapse that way
(Pmiss 0.581, Pfa 0.080, DCF 0.4560 at collar 0.5 -- confirmed by BOTH the pytest run and an
independent standalone `run_baseline` call, byte-identical to the printed precision). THE
MECHANISM (offered as the best available explanation, not independently proven by a
dedicated probe): an UNTRAINED windowed-attention layer at near-zero logits (Xavier draws
everything near 0, and `W_qkv`'s query/key rows start uncorrelated with each other) computes
attention scores `q . k / sqrt(d) + ALiBi bias` that are themselves near-zero and dominated
by the ALiBi term alone, so the softmax over the window is close to UNIFORM regardless of
the actual per-frame content -- i.e. the untrained cell behaves like a fixed windowed
AVERAGE of its recent input/value projections, which still carries real signal VARIANCE
through to the output. A fresh gated-recurrent cell (LSTM/sLSTM/Mamba/CfC), by contrast, has
no such "fall back to averaging" degenerate mode at init -- its output is whatever the
untrained gates happen to produce, and empirically that collapses hard toward one class.
`test_init_is_trainable[transformer-forward]` independently rules out a saturated-forward /
dead-gradient pathology as the explanation (init cost 0.863% of the log-law clamp, grad L2
2.24457, zero dead/weak columns): the net is trainable and DOES train, non-pathologically,
the same way every other row does; the init baseline is simply less degenerate, for
cell-architectural reasons distinct from anything remaining trainability-related.

THE RATIFICATION (2026-08-12, user-ratified via the coordinator, mirroring the phase-9
Twin convergence-gate deferral precedent -- a hard target that could not be met exactly as
originally stated was surfaced and resolved by explicit sign-off, not a unilateral agent
relaxation): THE HARD LEG's criterion is amended to an OR -- MARGIN >= 0.2 at the collar
(the original criterion, UNCHANGED), OR the trained model reaches `COLLAPSE_FLOOR_DCF =
0.2501` at that collar (the known all-speech operating point eight of the ten from-scratch
SAD rows on this subset converge to exactly). Semantics, verbatim: "training moved the
model a lot, or it reached the best-known subset operating point." THE NINE
ALREADY-PASSING ROWS' EVIDENCE IS UNCHANGED bit-for-bit: `margin_ok` alone is True at every
one of their collars, so `margin_ok or floor_ok` is True regardless of what `floor_ok`
says -- NOT because evaluation is "skipped" (`floor_ok` is an eagerly-assigned local,
computed unconditionally on every iteration; only the boolean `or` inside the `assert`
short-circuits, and even that costs nothing observable since `floor_ok` is already a plain
bool by the time it is read).

THE TWO INIT-DEGENERACY SANITY CHECKS (`ini.pmiss > 0.9`, `ini.dcf >= 0.6`) took a
DIFFERENT path, and the honest trail is the live record here rather than a superseded
mechanism: the FIRST committed form applied the IDENTICAL `or tr.dcf <= COLLAPSE_FLOOR_DCF`
disjunct to both (a necessary extension beyond the originally-ratified single assert, found
while implementing -- pytest evaluates `ini.pmiss > 0.9` BEFORE the per-collar loop even
runs, so leaving it untouched would still fail the row before the margin-or-floor logic is
ever reached). Review (fix round 1, F1) found that form UNFALSIFIABLE on nine of ten rows
(every row's TRAINED side reaches the floor regardless of what its INIT looked like), with
ZERO detection power against a future builder bug producing an accidentally non-degenerate
init, or `transformer/forward` itself regressing back to degenerate. It was STRENGTHENED to
a two-sided COMMITTED membership table instead -- `NON_DEGENERATE_INIT_ROWS` above,
mirroring `EXPECT_TIGHT_COHORT` (`src/rust/tests/phase9_stream_causal.rs:1388`): listed rows
MUST fail the degenerate characterization (`Pmiss > 0.9 AND DCF >= 0.6`), unlisted rows MUST
pass it, and membership moving is an ADJUDICATION, never a silent widen. THIS is the
CURRENT, committed form (see the assert in `test_subset_gate_beats_own_init` below the "THE
INIT-DEGENERACY CHARACTERIZATION" comment) -- THE HARD LEG's own margin-or-floor loop is the
ONLY site that still carries the original OR-disjunct design.

No training parameter was tuned to chase a pass at any point, across either round (seed,
epochs, steps_per_epoch, subset size are byte-identical to every other row's recipe); the
only change is the pass CRITERION itself.

Corpus-gated (`requires_corpus`) + pyo3 (`importorskip`): runs locally for implementers AND
reviewers, skips in CI. Measured whole-file runtime on the dev box (2026-08-01, Apple M4
Pro, LSTM/sLSTM/Mamba/CfC): 1.4 s for the ten preflight legs (eight v2 + two v1 CfC), 227 s
for the eight hard gates, 76 s for the eight determinism legs -- ~5.1 min total. The slowest
single gate is 62 s (mamba/bidirectional), an order of magnitude inside the < 10 min
per-gate budget the test asserts (spec R3). Phase-11 T9 (2026-08-12, same box) added the
fifth cell: +0.31 s for the two new preflight legs (0.22 s bi + 0.09 s fwd), +71.9 s for the
two new hard-gate legs (54.4 s bi + ~18.5 s fwd -- the training + scoring cost is identical
before and after the collapse-floor amendment, since only the pass CRITERION changed, not
the run itself), +24.6 s for the two new determinism legs (17.85 s bi + 6.32 s fwd) -- ~97 s
additional across the pair, well inside the same per-gate budget individually. Post-amendment
re-run (`-k transformer`, all six selected sub-legs): 96.43 s, exit 0, 6/6 PASSED.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.conftest import CORPUS_ROOT, GateRecorder, requires_corpus

pytest.importorskip("speech_rs")

from speech.config_bridge import nnet_spec  # noqa: E402 -- after importorskip, matching the pyo3-suite convention
from speech.drivers import baseline as B  # noqa: E402
from speech.drivers.state import ModernTrainParams, RunState  # noqa: E402
from speech.drivers.train import _init_weights_from_scratch, _modern_config_text  # noqa: E402
from speech.engine import forward_backward  # noqa: E402

# (cell, direction, MEASURED pack length) -- the S3.3 matrix, phase-11 T9 grown 8 -> 10 with
# the fifth cell. Each length is `init_weights`' own output at the arm's overlaid config,
# cross-checked against the closed forms in `_layer_length` below (`test_init_is_trainable`
# asserts both, so the literal and the arithmetic cannot drift apart silently). The two
# transformer rows were MEASURED (not estimated) via `len(init_weights(...))` on the real
# `sad-v2`-overlaid config -- see `tests/test_phase11_init.py`'s sizing docstring for the
# closed-form derivation at the sized `d_ff = 64` (bidirectional `13871 + 196*d_ff`, forward
# `6959 + 98*d_ff`); both rows agree exactly.
_CONFIGS = [
    ("lstm", "bidirectional", 24431),
    ("lstm", "forward", 12239),
    ("slstm", "bidirectional", 23279),
    ("slstm", "forward", 11663),
    ("mamba", "bidirectional", 28031),
    ("mamba", "forward", 14039),
    ("cfc", "bidirectional", 24491),
    ("cfc", "forward", 12269),
    ("transformer", "bidirectional", 26415),
    ("transformer", "forward", 13231),
]
_IDS = [f"{cell}-{direction}" for cell, direction, _ in _CONFIGS]

# The phase-6 SAD subset-gate recipe, VERBATIM (`test_phase6_gates.py::test_sad_subset_*`,
# and phase 9's `_GATE`) -- shared so every comparison differs only in cell/direction/lineage.
_GATE = dict(subset=10, valid_size=8, test_size=24, epochs=3, steps_per_epoch=10, patience=99, seed=0, audio_max_duration=20.0)
# The phase-6 determinism recipe, VERBATIM (determinism is a pipeline property, provable on
# a short run).
_DET = dict(subset=8, valid_size=6, test_size=8, epochs=1, steps_per_epoch=8, patience=99, seed=0, audio_max_duration=15.0)

# `Law::Log`'s saturated-forward constant: `cost = b + a*ln(1e-24)` once the clamp bites.
_LOG_CLAMP = float(-np.log(1e-24))  # 55.26204...

# THE COLLAPSE FLOOR (phase-11 T9, user-ratified 2026-08-12 -- mirroring the phase-9 Twin
# convergence-gate deferral precedent): the known all-speech operating point EIGHT of the
# ten from-scratch SAD rows in this matrix converge to EXACTLY (Pmiss 0, Pfa 1, DCF 0.25 at
# every collar); the remaining two (slstm-forward, mamba-forward) land near it without ever
# NEEDING this floor, since they clear the ORIGINAL margin criterion comfortably regardless
# -- see "THE TRANSFORMER ROWS" below for the full per-row table. The `+0.0001`
# absorbs the VRCTS `%f.4s` 4-decimal write quantum (the engine's hyp/ref timestamps are
# rounded to 4 decimals before scoring, so a DCF a hair above the mathematical 0.25 is the
# SAME operating point, not a worse one). `test_subset_gate_beats_own_init`'s hard leg
# passes a row either by the MARGIN (the original criterion: trained beats its own init by
# >= 0.2 at the collar) OR by reaching this floor (the model already found the best-known
# subset operating point, so a thin margin against an unusually non-degenerate init is not
# evidence of a worse outcome). See "THE TRANSFORMER ROWS" below for why this was needed:
# `transformer/forward`'s from-scratch init does not collapse degenerate the way every
# other cell's does, thinning its margin below 0.2 at the two widest collars even though
# its TRAINED model reaches DCF 0.250000 at every collar -- the SAME collapse point seven
# of the other nine rows' trained models also reach exactly.
COLLAPSE_FLOOR_DCF = 0.2501

# THE INIT-DEGENERACY COMMITTED MEMBERSHIP TABLE (phase-11 T9 fix round 1, review F1 -- the
# `EXPECT_TIGHT_COHORT` precedent, `src/rust/tests/phase9_stream_causal.rs:1388`). A row's
# from-scratch init is either DEGENERATE (`Pmiss > 0.9 AND DCF >= 0.6` -- the all-non-speech
# collapse every gated-recurrent cell's init reaches) or it is the ONE named exception,
# `transformer/forward` (Pmiss 0.581, DCF 0.456 -- "THE TRANSFORMER ROWS" below).
#
# THE COLLAPSE-FLOOR DISJUNCT (`COLLAPSE_FLOOR_DCF` above) was the WRONG instrument for this
# characterization, even though it is the RIGHT one for THE HARD LEG's own margin loop:
# `tr.dcf <= COLLAPSE_FLOOR_DCF` is TRUE on nine of ten rows regardless of what their init
# looked like (every row's TRAINED side reaches the floor), so an `or`-with-the-floor form on
# an INIT-side check is UNFALSIFIABLE for those nine -- it has ZERO detection power against
# (a) a future builder bug producing an accidentally non-degenerate init on any of them, or
# (b) `transformer/forward` itself regressing BACK to degenerate. A two-sided COMMITTED list
# is the right instrument: listed rows MUST fail the degenerate characterization, unlisted
# rows MUST pass it, and MEMBERSHIP MOVING IS AN ADJUDICATION -- re-list only after deciding
# the anomaly changed, never a silent widen to make a new number fit.
NON_DEGENERATE_INIT_ROWS = ("transformer-forward",)

# The inverse guard's magnitude floor (see the header). Rejects the ~1e-19 cancellation
# class that an analytically-zero-but-computed weight lands in; ~2.9e4 below the smallest
# per-column magnitude measured across all eight rows (2.854e-04).
_LIVE_FLOOR = 1e-8

# v2's frozen normalize mean/std tail: `2 * NNetInputSize` = 22 elements, seeded identity
# and never updated, so it is the ONLY structurally dead block a v2 pack has. Asserted as an
# EXACT total (not a floor plus slack, as v1's guards use) -- the inverse of v1's floors and
# the sharpest statement of the zero-dead-columns claim: 22 dead weights and not one more.
#
# THE RISK THIS TAKES, named: phase 9 recorded that a LIVE weight's gradient can land on
# exactly 0.0 by coincidence on some data, which is why its counts are floors with slack. An
# exact ceiling here would fail on such a coincidence. It is taken deliberately: this leg's
# recipe is fixed (subset 2, 10 s cap, seed 0) and deterministic, the measured value is 22 on
# all eight rows, and the whole point of the v2 lineage is that there is nothing else to be
# dead. A failure here means "investigate what became inert", not "widen this number" --
# extra zeros are exactly what a dead block MOVING somewhere the offsets do not address
# would look like, and that is the case the column guard alone cannot see.
_V2_FROZEN_TAIL = 22


# ------------------------------------------------------------------------------------- #
# Flat-layout arithmetic: where layer 0's input columns live, per cell.
#
# Transcribed from the SPECS (phase-9 S2.2 sLSTM / S3.2 Mamba, phase-10 S1.2 CfC) and from
# `nn/layers.rs::LstmLayer::set_weights` for the legacy cell -- NOT imported from
# `init_weights` or read back from Rust, the same independence `test_phase9_seam.py`'s
# block maps have. If either side drifts, these legs fail.
#
# WHOLE-PACK ORDER (`BlstmNetwork::set_weights`): `fwd stack | bwd stack (bidirectional
# only) | output MLP | mean | std`, each stack being `layer 0 | layer 1`. Only LAYER 0 has
# input columns worth guarding -- layer 1's fan-in is layer 0's output, live by construction.
# ------------------------------------------------------------------------------------- #


def _mamba_layer_length(fin: int, out: int, g: dict[str, int]) -> int:
    """One Mamba layer, spec S3.2: `[P | p] (iff fin != out), g, W_in, conv | b_conv, W_x,
    W_dt | b_dt, A_log, D, W_out`."""
    d_inner = g["expand"] * out
    dt_rank = g["dt_rank"] or max(1, -(-out // 16))  # `Mamba_Dt_Rank 0` -> auto `ceil(d_model / 16)`
    adapter = out * fin + out if fin != out else 0
    return (
        adapter
        + out  # g
        + 2 * d_inner * out  # W_in
        + d_inner * g["d_conv"]  # conv
        + d_inner  # b_conv
        + (dt_rank + 2 * g["d_state"]) * d_inner  # W_x
        + d_inner * dt_rank  # W_dt
        + d_inner  # b_dt
        + d_inner * g["d_state"]  # A_log
        + d_inner  # D
        + out * d_inner  # W_out
    )


def _layer_length(cell: str, fin: int, out: int, g: dict[str, int]) -> int:
    """One recurrent layer's flat length, per cell."""
    if cell == "lstm":  # 4 gate matrices (input + recurrent), the 12-row peephole bundle, 4 biases
        return 4 * out * (fin + out) + 12 * out + 4 * out
    if cell == "slstm":  # S2.2: `[R_a | W_a | b_a]` per gate a in i,f,o,z
        return 4 * out * (out + fin + 1)
    if cell == "mamba":
        return _mamba_layer_length(fin, out, g)
    if cell == "cfc":  # S1.2 closed form
        b, layers = g["backbone_units"], g["backbone_layers"]
        return b * (fin + out + 1) + (layers - 1) * b * (b + 1) + 3 * out * (b + 1)
    if cell == "transformer":  # phase-11 S1.2 closed form: [W_a|b_a|g_1|W_qkv|b_qkv|W_o|b_o|g_2|W_1|b_1|W_2|b_2]
        d_ff = g["d_ff"]
        return out * (fin + 1) + out + 3 * out * (out + 1) + out * (out + 1) + out + d_ff * (out + 1) + out * (d_ff + 1)
    raise AssertionError(f"unknown cell {cell!r}")


def _input_column_offsets(cell: str, fin: int, out: int, g: dict[str, int]) -> list[list[int]]:
    """`offsets[k]` = every flat position, RELATIVE TO THE LAYER-0 BASE, holding a weight
    that multiplies layer-0 input column `k`. One entry per input column.

    THE ARITHMETIC, per cell (each derived from that cell's own `for_each_slot` walk /
    `set_weights` order, and each restricted to the INPUT projection -- recurrent/state
    columns, biases and every downstream block are deliberately not addressed):

    - `lstm`: `input_weights` is `(fin x 4*out)` and is the layer's FIRST block, filled
      COLUMN-major (`LSTMLayer::setWeights`, `fill_col_major`), so `[k, c] -> c*fin + k`.
      Input column `k` is the k-th ROW of that matrix -> `{c*fin + k : c < 4*out}`.
    - `slstm`: per gate `a` the layer holds `[R_a (out x out) | W_a (out x fin) | b_a]`,
      each ROW-major, so gate `a`'s block starts at `a*out*(out+fin+1)` and `W_a[j, k]` is
      at `+ out*out + j*fin + k`. Column `k` -> `{a-block + out*out + j*fin + k : a < 4,
      j < out}` (4*out entries).
    - `mamba`: the ONLY place raw input enters is the width adapter `P (out x fin)`,
      row-major, the layer's first block when `fin != out` (it is, at layer 0: 44 != 24) --
      everything downstream sees `out`-wide. `P[j, k] -> j*fin + k`, so column `k` ->
      `{j*fin + k : j < out}` (out entries).
    - `cfc`: `W_bb0` is `(B x (fin + out))` row-major and is the layer's first block, but
      its fan-in is the CONCATENATION `z_t = [x_t | h_{t-1}]` -- so its columns are MIXED,
      and only `k < fin` are input columns (`k >= fin` are STATE columns, which are a
      different, already-pinned phenomenon: exactly dead at T=1, live beyond). Row-major
      `W_bb0[j, k] -> j*(fin + out) + k`, so input column `k` -> `{j*(fin+out) + k :
      j < B}` (B entries).
    - `transformer`: `W_a (H x in)` is the width ADAPTER and is the layer's first block,
      row-major -- UNLIKE Mamba's `P`, it is emitted UNCONDITIONALLY (`init_transformer_flat`
      draws it regardless of `fin == out`; there is no shape-closure special case here), and
      unlike CfC's `W_bb0` its fan-in is pure input with no state columns mixed in (the
      recurrence enters only through attention over PAST OUTPUT ROWS, never through a
      concatenated hidden vector). `W_a[j, k] -> j*fin + k`, so input column `k` ->
      `{j*fin + k : j < out}` (out entries) -- the same formula as Mamba's `P`, minus the
      `fin != out` precondition.
    """
    if cell == "lstm":
        return [[c * fin + k for c in range(4 * out)] for k in range(fin)]
    if cell == "slstm":
        block = out * (out + fin + 1)
        return [[a * block + out * out + j * fin + k for a in range(4) for j in range(out)] for k in range(fin)]
    if cell == "mamba":
        assert fin != out, "the Mamba width adapter is absent when fin == out, so there is no input-projection block to guard"
        return [[j * fin + k for j in range(out)] for k in range(fin)]
    if cell == "cfc":
        b = g["backbone_units"]
        return [[j * (fin + out) + k for j in range(b)] for k in range(fin)]
    if cell == "transformer":
        return [[j * fin + k for j in range(out)] for k in range(fin)]
    raise AssertionError(f"unknown cell {cell!r}")


@dataclass(frozen=True)
class _Arch:
    """The layer-0 geometry + stack bases a row's guard needs, read back out of the SAME
    overlaid config the run builds (never re-hardcoded)."""

    fin: int
    out: int
    pack_len: int
    stack_bases: tuple[int, ...]
    offsets: list[list[int]]


def _arch(arm: str, cell: str, direction: str) -> _Arch:
    import speech_rs

    repo_root = Path(__file__).resolve().parents[2]
    flat = {k: str(v) for k, v in speech_rs.load_toml_config(str(repo_root / B._ARM_CONFIGS[arm])).items()}
    flat.update(B.cell_overlay(dict(flat), cell, direction))
    spec = nnet_spec(flat, "BLSTM")
    lstm = cast(list[int], spec["LSTMNeuronNb"])
    sub = cast(list[int], spec["LSTMSubSampling"])
    outn = cast(list[int], spec["OutputNeuronNb"])
    g = cast(dict[str, int], spec["Mamba"] if cell == "mamba" else spec["Transformer"] if cell == "transformer" else spec["Cfc"])

    fin, out = lstm[0] * sub[0], lstm[1]
    len0 = _layer_length(cell, fin, out, g)
    len1 = _layer_length(cell, lstm[1] * sub[1], lstm[2], g)
    stacks = 1 if direction == "forward" else 2
    stack_len = len0 + len1
    mlp = sum(outn[i + 1] * outn[i] + outn[i + 1] for i in range(len(outn) - 1))
    pack = stacks * stack_len + mlp + 2 * lstm[0]
    return _Arch(fin, out, pack, tuple(s * stack_len for s in range(stacks)), _input_column_offsets(cell, fin, out, g))


def _dead_columns(grad: NDArray[np.float64], arch: _Arch) -> dict[int, list[int]]:
    """Per stack: the input columns whose ENTIRE input-projection block is exactly 0.0."""
    return {si: [k for k in range(arch.fin) if all(grad[base + p] == 0.0 for p in arch.offsets[k])] for si, base in enumerate(arch.stack_bases)}


def _weak_columns(grad: NDArray[np.float64], arch: _Arch) -> dict[int, list[tuple[int, float]]]:
    """Per stack: input columns whose largest |gradient| sits under `_LIVE_FLOOR` -- the
    cancellation-scale catch the exact-zero test alone would miss."""
    weak: dict[int, list[tuple[int, float]]] = {}
    for si, base in enumerate(arch.stack_bases):
        weak[si] = [(k, m) for k in range(arch.fin) if (m := max(abs(grad[base + p]) for p in arch.offsets[k])) < _LIVE_FLOOR]
    return weak


# ------------------------------------------------------------------------------------- #
# The preflight probe: one forward+backward at the from-scratch theta, no training.
# ------------------------------------------------------------------------------------- #


@dataclass
class _Rec:
    val_cost: float
    train_cost: float


@dataclass
class _Shim:
    """The minimal `ModernTrainResult` surface `run_baseline` reads back; the probe never
    trains, so the history is a single record carrying the measured init cost."""

    checkpoint_dir: str
    history: list[_Rec] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = -1
    best_val_cost: float = float("nan")


def _probe(out: dict[str, Any]) -> Callable[[RunState, int, ModernTrainParams], _Shim]:
    """A `run_baseline(_train_fn=...)` stub that MEASURES the from-scratch init instead of
    training: it re-uses the training loop's OWN seeded-init and config builders
    (`_init_weights_from_scratch`, `_modern_config_text(backprop=True)`), so what it probes
    is exactly the theta and the fold epoch 0 would start from -- not a re-derivation that
    could silently drift from the real path. (Phase 9's `_probe`, plus the raw gradient so
    the inverse guard can slice it.)"""

    def probe(state: RunState, seed: int, params: ModernTrainParams) -> _Shim:
        import speech_rs

        algo = state.ps.algo
        weights = _init_weights_from_scratch(state, params)
        workdir = Path(state.config_path).parent
        prev = Path.cwd()
        os.chdir(workdir)
        try:
            (workdir / "_preflight.config").write_text(_modern_config_text(state.base_config, algo, backprop=True))
            engine = speech_rs.Engine(["_preflight.config"], "-m")
            cost, grads = forward_backward(engine, weights)
            results = np.asarray(engine.results_matrix(), dtype=np.float64)
        finally:
            os.chdir(prev)

        grad = np.asarray(grads[0], dtype=np.float64)
        out["grad"] = grad
        out["pack_len"] = int(len(weights[0]))
        out["init_cost"] = float(cost)
        out["grad_l2"] = float(np.linalg.norm(grad))
        out["grad_linf"] = float(np.max(np.abs(grad)))
        out["dead"] = int(np.count_nonzero(grad == 0.0))
        # PER-FILE normalized cost. `results_matrix()` rows are PREFIXED
        # (`[file+1, conf+1, chan+1, res...]`, corpus_processor.rs), so the res block must be
        # sliced off exactly as `engine.py::_error_vad` does (`[:, 3:]`) BEFORE applying the
        # per-res column convention -- `res[4]` is the cumulative error and `res[-1]` the
        # counter, the same pair `_nn_cost_seg` sums. Indexing the prefixed matrix directly
        # would read `res[1]` (100*Pmiss) instead: the bug the phase-9 file shipped once, and
        # the `mean <= max` invariant below is what catches it.
        error_vad = results[results[:, 1] == 1][:, 3:]
        out["per_file_cost_max"] = float(np.max(error_vad[:, 4] / np.maximum(1.0, error_vad[:, -1])))
        ckpt = Path(state.out_dir) / "checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        return _Shim(checkpoint_dir=str(ckpt), history=[_Rec(float(cost), float(cost))], best_val_cost=float(cost))

    return probe


def _run_probe(tmp_path: Path, arm: str, cell: str, direction: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    B.run_baseline(
        arm,
        CORPUS_ROOT,
        tmp_path / "probe",
        subset=2,
        valid_size=0,
        test_size=0,  # no held-out scoring: this leg is about the init, not the task metric
        epochs=1,
        steps_per_epoch=1,
        patience=99,
        seed=0,
        audio_max_duration=10.0,
        cell_type=cell,
        direction=direction,
        _train_fn=_probe(out),
    )
    return out


# ------------------------------------------------------------------------------------- #
# PREFLIGHT + THE INVERSE GUARD
# ------------------------------------------------------------------------------------- #


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize(("cell", "direction", "pack_len"), _CONFIGS, ids=_IDS)
def test_init_is_trainable(tmp_path: Path, cell: str, direction: str, pack_len: int) -> None:
    """PREFLIGHT (the log-law hazard) + THE INVERSE GUARD (spec S3.2): before any training,
    the from-scratch v2 init must be the pinned size, cost-INTERIOR under the log law both in
    aggregate and per file, carry a NONZERO epoch-0 gradient, and have ZERO structurally-dead
    layer-0 input columns. Measured 2026-08-01, seed 0, AT THIS LEG'S OWN RECIPE (subset 2,
    10 s cap). `%clamp` is the column left of it over `-ln(1e-24) = 55.262`; `min col` is the
    smallest per-input-column max|grad| over every stack (the inverse guard's own margin):

    | config     | pack  | init NNCostSeg | %clamp | per-file max | %clamp | grad L2 | grad Linf | dead | min col  |
    |------------|-------|----------------|--------|--------------|--------|---------|-----------|------|----------|
    | lstm-bi    | 24431 | 0.28007        | 0.507% | 0.40669      | 0.736% | 0.39567 | 0.19439   | 22   | 2.854e-4 |
    | lstm-fwd   | 12239 | 0.27649        | 0.500% | 0.40178      | 0.727% | 0.37588 | 0.19127   | 22   | 3.493e-4 |
    | slstm-bi   | 23279 | 0.28918        | 0.523% | 0.40714      | 0.737% | 0.38630 | 0.19917   | 22   | 6.011e-4 |
    | slstm-fwd  | 11663 | 0.24322        | 0.440% | 0.33209      | 0.601% | 0.45236 | 0.17433   | 22   | 1.311e-3 |
    | mamba-bi   | 28031 | 0.28676        | 0.519% | 0.40283      | 0.729% | 0.54261 | 0.18486   | 22   | 2.960e-3 |
    | mamba-fwd  | 14039 | 0.29903        | 0.541% | 0.46646      | 0.844% | 0.50035 | 0.18487   | 22   | 3.856e-3 |
    | cfc-bi     | 24491 | 0.29857        | 0.540% | 0.41060      | 0.743% | 0.72674 | 0.19351   | 22   | 3.269e-3 |
    | cfc-fwd    | 12269 | 0.27922        | 0.505% | 0.39550      | 0.716% | 0.41801 | 0.19030   | 22   | 2.487e-3 |

    No row's init sits near the saturated-forward constant (all ~0.5% of it, worst 0.54%),
    no INDIVIDUAL file does either (worst 0.466, 0.84%), and every gradient norm is far from
    zero -- so none of the eight starts in the zero-gradient death the log law would make
    permanent. `dead` is 22 for EVERY row: exactly the frozen normalize tail, nothing else.
    Cheap (~0.15 s per row: no valid/test split, no training, one fold over 2 files)."""
    out = _run_probe(tmp_path, "sad-v2", cell, direction)
    arch = _arch("sad-v2", cell, direction)

    # (0) param match, asserted TWICE against independent derivations: the literal in
    #     `_CONFIGS` (what the phase pins) and the closed-form block arithmetic in
    #     `_layer_length` (what the guard's offsets are derived from). A drift in either
    #     invalidates the offsets, so both must agree with what `init_weights` produced.
    assert out["pack_len"] == pack_len, f"{cell}/{direction} pack length {out['pack_len']} != the pinned {pack_len}"
    assert arch.pack_len == pack_len, f"{cell}/{direction} block arithmetic gives {arch.pack_len}, the pinned length is {pack_len}"
    assert arch.fin == 44, f"v2's layer-0 fan-in must be 11*4 = 44 live columns, got {arch.fin}"

    # (a) the initial cost is finite AND in the log law's interior, in AGGREGATE and for
    #     EVERY file (the frame-weighted aggregate could hide a saturated minority).
    assert np.isfinite(out["init_cost"]) and out["init_cost"] > 0.0, f"init cost must be finite and positive, got {out['init_cost']}"
    # measured <= 0.299, i.e. <= 0.54% of the clamp constant; pin 5% -> ~9.2x headroom.
    assert out["init_cost"] < 0.05 * _LOG_CLAMP, f"init cost {out['init_cost']:.5f} sits near the log-law clamp {_LOG_CLAMP:.3f} (saturated init)"
    # SELF-CHECK FIRST: `init_cost` is a count-weighted MEAN of the per-file normalized
    # costs, so it can never exceed their max. If this fires, the two quantities are not the
    # pair they claim to be (wrong `results_matrix` column -- see `_probe`).
    assert out["per_file_cost_max"] >= out["init_cost"] - 1e-9, (
        f"{cell}/{direction} per-file max {out['per_file_cost_max']:.6f} < aggregate mean {out['init_cost']:.6f}: "
        "the per-file quantity is not the cost (wrong results_matrix column?)"
    )
    # per-file worst case: measured <= 0.466 (0.84% of the clamp); same 5% pin -> 5.9x headroom.
    assert out["per_file_cost_max"] < 0.05 * _LOG_CLAMP, (
        f"{cell}/{direction} worst per-file cost {out['per_file_cost_max']:.5f} sits near the log-law clamp {_LOG_CLAMP:.3f} (a saturated file)"
    )

    # (b) the analytic gradient at epoch 0 is NONZERO -- the zero-gradient-death assert.
    #     Whole-vector norm; measured 0.376-0.727, pin 1e-3 -> ~376x headroom.
    assert out["grad_l2"] > 1e-3, f"{cell}/{direction} epoch-0 gradient norm {out['grad_l2']:.6f} is ~zero: the log-law saturation stall"
    assert np.isfinite(out["grad_linf"])

    # (c) THE INVERSE GUARD (spec S3.2): zero structurally-dead layer-0 input columns, in
    #     every stack. Two-sided -- exact-zero for the structural claim, `_LIVE_FLOOR` for
    #     the cancellation class an exact-zero test alone would pass.
    dead = _dead_columns(out["grad"], arch)
    assert all(not v for v in dead.values()), f"{cell}/{direction} has structurally DEAD layer-0 input columns (v2 must have none): {dead}"
    weak = _weak_columns(out["grad"], arch)
    assert all(not v for v in weak.values()), f"{cell}/{direction} layer-0 input columns are gradient-inert (< {_LIVE_FLOOR:g}): {weak}"

    # (d) ... and the pack as a WHOLE carries exactly the frozen normalize tail as dead
    #     weight and nothing else. The sharpest form of the zero-dead-columns claim: (c)
    #     alone would pass if a dead block moved somewhere the offsets do not address.
    assert out["dead"] == _V2_FROZEN_TAIL, (
        f"{cell}/{direction} has {out['dead']} exactly-zero weights; v2 should carry only the "
        f"{_V2_FROZEN_TAIL}-element frozen normalize tail (a dead block appeared or the tail moved)"
    )


# ------------------------------------------------------------------------------------- #
# The HARD gate: trained held-out DCF beats the model's OWN from-scratch init.
# ------------------------------------------------------------------------------------- #


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize(("cell", "direction", "pack_len"), _CONFIGS, ids=_IDS)
def test_subset_gate_beats_own_init(tmp_path: Path, cell: str, direction: str, pack_len: int, gate_record: GateRecorder) -> None:
    """THE HARD LEG (spec S3.3): train this cell x direction from scratch on the v2 lineage's
    10-file subset, then score a disjoint 24-file held-out slice END TO END through the DCF
    harness -- the trained pooled DCF must beat its own untrained init's, at every collar,
    deterministically. Measured 2026-08-01, seed 0 (LSTM/sLSTM/Mamba/CfC) and 2026-08-12,
    seed 0 (Transformer, phase-11 T9), at collar 0.5 with the min/max over all five collars
    alongside (rounded where a row is exactly degenerate, printed in full where it is not --
    rounding a non-degenerate row into the collapse story would be a lie the table is here to
    prevent):

    | config          | trained DCF@0.5 | Pmiss / Pfa       | trained collar range  | init DCF@0.5 | init collar range     | wall   | best ep |
    |-----------------|-----------------|-------------------|-----------------------|--------------|-----------------------|--------|---------|
    | lstm-bi         | 0.250000        | 0.000000/1.000000 | 0.250000 (all)        | 0.750000     | 0.750000 (all)        | ~37 s  | 1       |
    | lstm-fwd        | 0.250000        | 0.000000/1.000000 | 0.250000 (all)        | 0.750000     | 0.750000 (all)        | ~15 s  | 1       |
    | slstm-bi        | 0.250000        | 0.000000/1.000000 | 0.250000 (all)        | 0.750000     | 0.750000 (all)        | ~34 s  | 0       |
    | slstm-fwd       | 0.252430        | 0.026415/0.930474 | [0.248021, 0.255545]  | 0.750000     | 0.750000 (all)        | ~15 s  | 0       |
    | mamba-bi        | 0.250000        | 0.000000/1.000000 | 0.250000 (all)        | 0.750000     | 0.750000 (all)        | ~62 s  | 1       |
    | mamba-fwd       | 0.248770        | 0.000000/0.995078 | [0.248770, 0.250000]  | 0.737710     | [0.736969, 0.738062]  | ~19 s  | 1       |
    | cfc-bi          | 0.250000        | 0.000000/1.000000 | 0.250000 (all)        | 0.750000     | 0.750000 (all)        | ~33 s  | 1       |
    | cfc-fwd         | 0.250000        | 0.000000/1.000000 | 0.250000 (all)        | 0.750000     | 0.750000 (all)        | ~15 s  | 2       |
    | transformer-bi  | 0.250000        | 0.000000/1.000000 | 0.250000 (all)        | 0.747406     | [0.747045, 0.748523]  | ~54 s  | 2       |
    | transformer-fwd | 0.250000        | 0.000000/1.000000 | 0.250000 (all)        | 0.455959     | [0.435836, 0.476457]  | ~19 s  | 1       |

    SIX of the ten rows are degenerate at BOTH endpoints (all-one-class collapse), so their
    DCF is exactly 0.25/0.75 with no libm-sensitive boundary jitter, and the six-way TIE is
    what an identical collapse looks like on a 10-file subset -- RECORDED not gated (R5,
    header). EIGHT of the ten rows' TRAINED side reaches EXACTLY the collapse point at every
    collar (DCF 0.250000, Pmiss 0, Pfa 1). The remaining two are genuinely non-degenerate on
    the TRAINED side too (NOT a contradiction -- see below): `slstm-forward` (0.252430 at
    collar 0.5, ranging [0.248021, 0.255545]) and `mamba-forward` (0.248770 at collar 0.5,
    ranging [0.248770, 0.250000]) both still clear their own untrained-vs-trained MARGIN
    comfortably (0.494 and 0.489 respectively), so training is never genuinely in doubt for
    any of the ten rows; only the INIT side varies in a way that matters. FOUR rows' init is
    not exactly degenerate, and they fall into two different categories. THREE are the
    honest-but-still-clearing-the-original-margin rows: `slstm-forward` genuinely rejects
    ~7% of held-out non-speech at a 2.6% miss cost
    (its WORST collar, 0.255545, is still 0.494 below its init); `mamba-forward`'s trained
    model rejects a 0.49% sliver while its init is also not fully degenerate (Pmiss 0.982625,
    which is what the `> 0.9` init pin below is sized for); and `transformer-bidirectional`'s
    init similarly sits just off the exact 0.75 baseline (Pmiss 0.996061, DCF range
    [0.747045, 0.748523]) while still clearing every collar with a ~0.50 margin, the same
    healthy margin the six exactly-degenerate rows carry. THE FOURTH, `transformer-forward`,
    is DIFFERENT IN KIND, not degree: its init DCF sits at 0.435836-0.476457 -- a whole
    quarter below every other row's init, reflecting a Pmiss of 0.581 (not >0.9) -- and this
    genuinely MISSES the original `- 0.2` per-collar margin at collars 1.0 and 2.0 (measured
    margins 0.194262 and 0.185836), an INIT-QUALITY ANOMALY rather than a training failure
    (its trained side reaches DCF 0.250000 at every collar, the same collapse point SEVEN of
    the other nine rows' trained sides also reach).
    It PASSES the user-ratified 2026-08-12 amendment at every collar via the collapse-floor
    disjunct instead (`COLLAPSE_FLOOR_DCF`; see "THE TRANSFORMER ROWS" in the module
    docstring for the full breakdown, the likely mechanism, and the ratification). The
    phase-9 v1 rows land at the same 0.2500 (bar mamba-forward's 0.249625), so this run says
    nothing about v1 vs v2 either; the full-corpus launchers are the referee.

    REPRODUCIBILITY of the table itself: every DCF@0.5 above was produced twice, in two
    independent processes (this test, and a standalone driver run), identical to the printed
    precision (the transformer pair additionally cross-checked against `test_deterministic`'s
    own bit-identical-weights run). `test_deterministic` carries the bit-level claim on the
    weight BYTES.

    The wall column is NOT a speed comparison: `--direction forward` also switches the
    windowing regime (`cell_overlay` forces `BLSTM_window 0`, the plain whole-sequence run,
    vs the bidirectional rows' windowed overlap), so the forward rows differ in TWO ways.

    The train/validation CE is NOT the signal here (the phase-6 lesson): only the held-out
    TASK metric is gated."""
    t0 = time.time()
    res = B.run_baseline("sad-v2", CORPUS_ROOT, tmp_path / "run", cell_type=cell, direction=direction, score_init=True, **_GATE)  # type: ignore[arg-type]
    wall = time.time() - t0

    # --- both DCF reports landed, valid ranges ---
    assert res.dcf is not None and res.init_dcf is not None
    tr = res.dcf.by_collar(0.5)
    ini = res.init_dcf.by_collar(0.5)
    assert np.isfinite(tr.dcf) and 0.0 <= tr.dcf <= 1.0 and 0.0 <= ini.dcf <= 1.0

    # --- the net learned to FIRE: trained detects held-out speech ---
    assert tr.pmiss < 0.5, f"{cell}/{direction} trained must detect held-out speech (Pmiss {tr.pmiss:.3f})"

    # --- THE INIT-DEGENERACY CHARACTERIZATION (two-sided, COMMITTED -- fix round 1, review
    #     F1; see NON_DEGENERATE_INIT_ROWS's doc above for why a collapse-floor disjunct was
    #     the wrong instrument here, even though it is the right one for THE HARD LEG below).
    #     Every row's from-scratch init is EITHER the all-non-speech collapse every
    #     gated-recurrent cell's init reaches (Pmiss > 0.9 AND DCF >= 0.6) OR the one named
    #     exception -- a row moving between the two is an ADJUDICATION, never a number to
    #     silently widen.
    row_id = f"{cell}-{direction}"
    init_degenerate = ini.pmiss > 0.9 and ini.dcf >= 0.6
    assert init_degenerate == (row_id not in NON_DEGENERATE_INIT_ROWS), (
        f"{cell}/{direction} init-degeneracy characterization moved (Pmiss {ini.pmiss:.3f}, DCF {ini.dcf:.4f}) -- adjudicate, do not re-list"
    )

    # --- THE HARD LEG: trained held-out DCF beats its OWN init by a margin, at EVERY collar,
    #     OR the trained model reached the collapse floor at that collar (user-ratified
    #     2026-08-12, mirroring the phase-9 Twin convergence-gate deferral precedent -- see
    #     COLLAPSE_FLOOR_DCF's doc above and "THE TRANSFORMER ROWS" in the module docstring).
    #     THE NINE ALREADY-PASSING ROWS' EVIDENCE IS UNCHANGED bit-for-bit: `margin_ok` alone
    #     is True at every one of their collars, so `margin_ok or floor_ok` is True regardless
    #     of what `floor_ok` says -- NOT because evaluation is "skipped" (`floor_ok` is an
    #     eagerly-assigned local, computed unconditionally on every iteration; only the
    #     boolean `or` inside the `assert` short-circuits, and even that costs nothing
    #     observable here since `floor_ok` is already a plain bool by the time it is read).
    #     (measured gap 0.50 at all five collars on nine of ten rows; pin a conservative 0.2.)
    for collar in (0.0, 0.25, 0.5, 1.0, 2.0):
        a, b = res.dcf.by_collar(collar), res.init_dcf.by_collar(collar)
        margin_ok = a.dcf < b.dcf - 0.2
        floor_ok = a.dcf <= COLLAPSE_FLOOR_DCF
        assert margin_ok or floor_ok, (
            f"{cell}/{direction} trained DCF {a.dcf:.4f} must beat init {b.dcf:.4f} by >= 0.2 at collar "
            f"{collar}, or reach the collapse floor {COLLAPSE_FLOOR_DCF} (trained {a.dcf:.4f})"
        )
    assert tr.dcf <= 0.5, f"trained DCF {tr.dcf:.4f} must beat the all-non-speech baseline (0.75) with headroom"

    # --- the lineage AND the architecture knobs really took effect (not a silently-defaulted
    #     v1/BLSTM run): the checkpoint is the v2-sized pack for THIS cell x direction ---
    assert (res.checkpoint_dir / "best_sad.bin").is_file() and (res.checkpoint_dir / "last_sad.bin").is_file()
    packs = {(res.checkpoint_dir / n).stat().st_size for n in ("best_sad.bin", "last_sad.bin")}
    assert packs == {16 + 8 * pack_len}, f"{cell}/{direction} checkpoint is not the {pack_len}-element v2 pack: {packs}"

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
@pytest.mark.parametrize(("cell", "direction", "pack_len"), _CONFIGS, ids=_IDS)
def test_deterministic(tmp_path: Path, cell: str, direction: str, pack_len: int) -> None:
    """Run-twice determinism at a fixed seed: bit-identical trained weight BYTES + identical
    pooled held-out DCF, per cell x direction. A SHORT config (the phase-6/9 pattern --
    determinism is a pipeline property, provable cheaply)."""
    a = B.run_baseline("sad-v2", CORPUS_ROOT, tmp_path / "a", cell_type=cell, direction=direction, **_DET)  # type: ignore[arg-type]
    b = B.run_baseline("sad-v2", CORPUS_ROOT, tmp_path / "b", cell_type=cell, direction=direction, **_DET)  # type: ignore[arg-type]

    for name in ("last_sad.bin", "best_sad.bin"):
        pa, pb = (a.checkpoint_dir / name).read_bytes(), (b.checkpoint_dir / name).read_bytes()
        assert pa == pb, f"{cell}/{direction} {name} not bit-identical across two fixed-seed runs"
        assert len(pa) == 16 + 8 * pack_len, f"{name} is not the {pack_len}-element pack"
    assert a.dcf is not None and b.dcf is not None
    assert a.dcf.by_collar(0.5).dcf == b.dcf.by_collar(0.5).dcf, f"{cell}/{direction} pooled held-out DCF not reproducible"


# ------------------------------------------------------------------------------------- #
# The v1 CfC MECHANICAL leg (spec S3.3) -- and the inverse guard's non-vacuity proof.
# ------------------------------------------------------------------------------------- #

# (direction, MEASURED v1 pack length, predicted dead-weight floor).
#
# THE DERIVATION. v1's layer-0 fan-in is `NNetInputSize * sub_sampling = 23*4 = 92` while the
# DSP front-end emits `3*nb_dct - ignore_first_dct = 11` columns, stacked 4-deep = 44 -- so
# input columns `[44, 92)`, 48 of them, are never read and every weight addressing them is
# exactly zero forever. The CfC layer-0 block that addresses them is `W_bb0`, `B x (92+24)`
# row-major, so each of its `B` rows contributes 48 dead entries:
#
#     dead(W_bb0) = B * 48                 per stack
#     dead(pack)  = D * B * 48 + 46        D = 2 bidirectional / 1 forward, 46 = 2*23, the
#                                          frozen normalize tail
#
# at the sized default `B = 45` (`CFC_DEFAULT_BACKBONE_UNITS`): 4366 / 2206. This is the
# cell-DEPENDENT shape the phase-9 T8 pattern predicts per cell -- `B*48` here where the gate
# cells give `4*out*48` and Mamba a single `out*48`.
#
# B IS THE DEFAULT 45, NOT the v1-matched 53: spec S3.3 gives CfC on v1 a MECHANICAL leg
# only -- construction plus its own dead-count floor -- with NO param-match requirement, and
# `Cfc_Backbone_Units 53` (T2's v1-matched value, 33795 vs LSTM's 33671) would only matter
# for a fair-size comparison this leg does not make. At 45, v1 CfC sits at -14.36% of the v1
# LSTM pack: in the +-15% band, but that is a coincidence of the sizing, not a claim.
_V1_CFC = [("bidirectional", 28835, 2 * 45 * 48 + 46), ("forward", 14453, 1 * 45 * 48 + 46)]
# Slack above the floor (the phase-9 `_DEAD_SLACK` pattern): the count is a FLOOR, since a
# LIVE weight can land on exactly 0.0 by coincidence. 64 is phase 9's value, kept for
# consistency; both rows measured EXACTLY at the floor (4366 / 2206), so the slack is
# currently unused headroom, not absorbed error.
_DEAD_SLACK = 64


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize(("direction", "pack_len", "dead_expected"), _V1_CFC, ids=[f"v1-cfc-{d}" for d, _, _ in _V1_CFC])
def test_v1_cfc_mechanical(tmp_path: Path, direction: str, pack_len: int, dead_expected: int) -> None:
    """CfC on the FROZEN v1 lineage: MECHANICAL only (spec S3.3) -- it constructs, seeds at
    the derived size, produces a finite cost and a nonzero gradient, and carries EXACTLY the
    dead block the `_V1_CFC` arithmetic above predicts. No convergence gate here (CfC's
    convergence gate IS its v2 rows) and no param-match requirement.

    This leg is ALSO the inverse guard's non-vacuity proof: it runs the SAME
    `_input_column_offsets` machinery `test_init_is_trainable` runs, on the same cell, and
    must find exactly the 48 dead columns `[44, 92)`. If the offset arithmetic were wrong --
    addressing biases, a recurrent block, or nothing at all -- it would find 0 dead columns
    here and the v2 guard would be silently vacuous. Measured 2026-08-01, seed 0:

    | config          | pack  | init NNCostSeg | per-file max | grad L2 | dead | dead cols/stack |
    |-----------------|-------|----------------|--------------|---------|------|-----------------|
    | v1 cfc-bi       | 28835 | 0.34644        | 0.47582      | 0.97583 | 4366 | 48 (== [44,92)) |
    | v1 cfc-fwd      | 14453 | 0.29140        | 0.41856      | 0.55960 | 2206 | 48 (== [44,92)) |
    """
    out = _run_probe(tmp_path, "sad", "cfc", direction)
    arch = _arch("sad", "cfc", direction)

    # construction + sizing, from two independent derivations (as in the v2 preflight)
    assert out["pack_len"] == pack_len, f"v1 cfc/{direction} pack length {out['pack_len']} != the pinned {pack_len}"
    assert arch.pack_len == pack_len, f"v1 cfc/{direction} block arithmetic gives {arch.pack_len}, pinned {pack_len}"
    assert arch.fin == 92, f"v1's layer-0 fan-in must be 23*4 = 92 declared columns, got {arch.fin}"

    # it runs: finite interior cost, nonzero gradient (the same hazards, not re-argued)
    assert np.isfinite(out["init_cost"]) and 0.0 < out["init_cost"] < 0.05 * _LOG_CLAMP
    assert out["per_file_cost_max"] >= out["init_cost"] - 1e-9 and out["per_file_cost_max"] < 0.05 * _LOG_CLAMP
    assert out["grad_l2"] > 1e-3, f"v1 cfc/{direction} epoch-0 gradient norm {out['grad_l2']:.6f} is ~zero"

    # THE DEAD BLOCK: exactly the 48 columns [44, 92), in every stack -- the v2 guard's mirror
    # image, and its non-vacuity proof.
    dead = _dead_columns(out["grad"], arch)
    for si, cols in dead.items():
        assert cols == list(range(44, 92)), f"v1 cfc/{direction} stack {si}: dead columns {cols[:3]}...{cols[-3:]} != [44, 92)"
    # ... and the complement is LIVE in EVERY stack, not just the forward one -- the dead
    # check above loops all stacks, so this one must too or the bidirectional row's reverse
    # half would go unchecked on the liveness side.
    for si, base in enumerate(arch.stack_bases):
        live = [k for k in range(44) if any(abs(out["grad"][base + p]) >= _LIVE_FLOOR for p in arch.offsets[k])]
        assert live == list(range(44)), f"v1 cfc/{direction} stack {si}: columns [0, 44) must all be live, got {len(live)}"

    # ... and the WHOLE-PACK dead count against the derived floor (phase-9 `_DEAD_SLACK` pattern)
    assert out["dead"] >= dead_expected, f"v1 cfc/{direction} dead-weight count {out['dead']} is below the structural floor {dead_expected}"
    assert out["dead"] <= dead_expected + _DEAD_SLACK, f"v1 cfc/{direction} dead-weight count {out['dead']} far exceeds the floor {dead_expected}"
