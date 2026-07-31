# Phase 10 -- CfC + fast-tree completion + the v2 lineage -- Design

Date: 2026-07-31
Status: approved (design sections 1-4 user-approved in-conversation)
Branch: `feature/phase-10-cfc-fast-completion`

## Goal

Three threads, one phase. (1) A THIRD causal architecture -- the CfC cell (closed-form
continuous-time lineage) -- lands behind the proven `CellLayer` seam: exact f64 forward +
hand-derived backward, FD/grad-check pinned, from-scratch gated, fast twin + streaming
inherited through the existing cell-generic machinery. (2) The fast tree COMPLETES its
(cell x direction) matrix: bidirectional f32 twins for the new cells, a forward-only LSTM
twin, the full-f32 mel front-end (retiring the ~23.5 MB widen tax), and Mamba exact-path
retention gating (retiring the ~2x inference RSS penalty). (3) A clean `lre_sad_v2`
config lineage fixes the 2015-inherited dead-input-columns defect (declared 23-wide vs
DSP-produced 11) BEFORE the full-corpus launchers make that fix expensive -- fresh
subset baselines for every cell, like-for-like live capacity for the referee runs.

User-ratified scope decisions (2026-07-31):

1. **Phase themes**: CfC + fast-tree completion AND the v2 config lineage. The
   streaming endgame (provisional-begin, holdback tightening) and the transformer are
   EXCLUDED -- each named a later-phase candidate (see Deferred).
2. **v2 scope**: MINIMAL -- the 11-wide fix only, isolating exactly one variable for
   the v1-vs-v2 referee attribution. A streaming-native lineage is a later, separate
   decision.
3. **Approach A**: parallel bi-cell fast driver against the shared window-span helpers;
   `FastBlstm`/`overlap_window_step` byte-untouched (the phase-8 bit-identity kernel is
   never moved); dedupe-later accepted deliberately over unification churn.

## Non-goals

- Transformer encoder (its own phase: the O(T^2)-at-corpus-scale + bounded-KV streaming
  definition problem -- see Deferred for the design trap on record).
- Provisional-begin emission, global-holdback tightening, the `raw_segments` quadratic
  replay fix (the streaming-endgame theme, not selected -- Deferred, with the T9-updated
  design notes so nothing re-derives them from scratch).
- A streaming-native SAD lineage (norm-type/window changes beyond the 11-wide fix).
- sLSTM heads/mLSTM; Mamba recompute-on-backward; per-net Mamba geometry overrides;
  QPSO over new-cell hyperparameters (all carried forward unchanged).
- Any full-corpus training run (user-fired; the launchers gain a v2 recipe, nothing
  fires in-phase).

## S1 -- The CfC cell contract

**S1.1 Forward** (per timestep; the ncps-lineage GATED CfC under the constant-Delta-t
reduction -- uniform frame times make the paper's `t_a*Dt + t_b` gate affine in a
constant, so the two time heads collapse into ONE gate head; documented in the module
doc as the reduction, keeping the continuous-time lineage legible):

```
z_t = [x_t | h_{t-1}]                                   (width in + h)
bb  = lecun_tanh(W_bb z_t + b_bb)                       (backbone; width B, L layers --
                                                         layers 2..L are B -> B;
                                                         lecun_tanh(x) = 1.7159 * tanh(2x/3))
ff1 = tanh(W_1 bb + b_1)      ff2 = tanh(W_2 bb + b_2)  (h each)
g   = sigmoid(W_t bb + b_t)                             (h; the time-interpolation gate)
h_t = ff1 * (1 - g) + ff2 * g
```

Output is `h_t`. State zero-init (`h_0 = 0`). Natively causal, fixed-size state, O(T).
`sigmoid` is the plain logistic (`activations.rs::logistic_fn`), NOT the 0.1-prescaled
`GatesFunction`.

**S1.2 Flat layout** (row-major per matrix, the single `for_each_slot` walk per the
house pattern): `W_bb (B x (in+h)) | b_bb (B)` then per deeper backbone layer
`W (B x B) | b (B)`, then `W_1 (h x B) | b_1 (h) | W_2 | b_2 | W_t | b_t`.
`nb_of_weights = B*(in+h+1) + (L-1)*B*(B+1) + 3*h*(B+1)`.

**S1.3 Backward.** Plain reverse-time BPTT: the single recurrence path is `h_{t-1}`
entering through `z_t`'s tail slice; adjoints are tanh'/sigmoid'/lecun_tanh'
(`1.7159 * (2/3) * (1 - tanh^2(2x/3))`) locals + matmul transposes; `dz` splits into
`dx_t` (returned deltas) and the carried `dh_{t-1}`. Caches: per-timestep `z` (or
equivalently the `h` sequence), backbone pre-activations per layer, the three head
pre-activations. Any structurally gradient-dead block the derivation finds is pinned
exactly `== 0.0` in the cell's own unit tests (the phase-9 pattern); none is currently
predicted.

**S1.4 Hyperparams + sizing.** Port-only keys, section `[nn_cfc]` (flat:
`Cfc_Backbone_Units`, `Cfc_Backbone_Layers`), defaults chosen at implementation time by
the SIZING PROCEDURE: pick `Cfc_Backbone_Units` (at `Cfc_Backbone_Layers 1`) so the
full-net pack lands within the S8.2-style +-15% band of the same lineage's LSTM pack,
per lineage (v1 and v2 differ in layer-0 width, so the matched B may differ; the task
records the arithmetic). One geometry per config (the `Mamba_*` precedent); the
`MambaParams`-style `from_legacy` validation (`>= 1`, mirrored in Python
`config_bridge`).

**S1.5 Init.** Xavier/He per repo convention on every block (backbone + three heads;
`fan_in`/`fan_out` from the block shapes; biases 0). STATED OPENLY: CfC has no
reference-lineage magic constants (no S4D-real analogue) -- the blocks are plain dense
layers and the phase-5 variance conventions apply as-is. Builder emits the S1.2 flat
order directly; whole-pack block-by-block reconstruction pins (the T4 shear lesson)
are MANDATORY.

**S1.6 The 4th-variant touch list** is phase 9's verbatim (compile-error-enforced):
`CellType` variant + parse/as_str; the exhaustive `from_config` arm; `CellLayer` variant
+ 10 delegation arms; `nn/cells/cfc.rs`; FD-tier Cases in the cell-agnostic harness;
`FastCfc` step kernel + `FastCell` arm (`classify_fast_shape`'s causal arm auto-admits
any non-LSTM cell -- ZERO streaming-layer change); `KEY_TABLE` rows; Python geometry
reader + init builder + `baseline.py` choices tuples.

## S2 -- Config vocabulary

| Flat key | TOML | Values / default |
|---|---|---|
| `BLSTM_Cell_Type` (existing) | `[nn] cell_type` | gains the value `cfc` |
| `BLSTM_LID_Cell_Type` (existing) | `[nn_lid] cell_type` | gains `cfc` (mechanical wiring only) |
| `Cfc_Backbone_Units` | `[nn_cfc] backbone_units` | int, default = the sized value (S1.4) |
| `Cfc_Backbone_Layers` | `[nn_cfc] backbone_layers` | int, 1 |

Absent keys mean the defaults; every pre-phase-10 config decodes unchanged.

## S3 -- The `lre_sad_v2` lineage

**S3.1 The file.** `configs/training/lre_sad_v2.toml` = v1 byte-for-byte EXCEPT:
`nnet_input_size = 11`, `lstm_neuron_nb = "11,24,24"` (entry 0 is the input width;
hidden layers unchanged), and the entailed normalize tail (2*11 = 22, self-sizing --
`train.py:76` reads the config). Headers: v2's states the lineage relationship + points
at RESULTS' dead-columns section; v1's gains a pointer to v2. NOTHING else differs --
one variable, clean attribution.

**S3.2 The inverse guard.** v1's gates pin dead-count FLOORS; v2's gates pin ZERO
structurally-dead layer-0 INPUT COLUMNS per cell: every column of the layer-0
input-projection block carries nonzero gradient. Cell-level dead blocks that are NOT
input columns (sLSTM's `b_i`; Mamba's T=1-only cases) are a separate, already-pinned
phenomenon and stay out of this guard's scope. Pack-length pins recomputed per cell at
implementation (the survey's ~24431-class estimates are back-of-envelope, MEASURED
values go in).

**S3.3 The gate matrix.** A new corpus-gated file beside the v1 family (v1 files stay
frozen AND green): {lstm, slstm, mamba, cfc} x {bidirectional, forward} = 8 subset
gates on v2, the phase-6 protocol verbatim (seeded init, `train_modern` with
`nn_cost_seg`, HARD beat-init on held-out DCF at every collar, run-twice byte-identical,
<10 min each, budget asserted). CfC's convergence gate IS its v2 rows; on v1 CfC gets a
MECHANICAL leg only (construction + its own predicted dead-count floor -- the
dead-block shape is cell-dependent).

**S3.4 Surface + posture.** New arm key `sad-v2` in `_ARM_CONFIGS`
(`speech baseline sad-v2 --cell-type ... --direction ...`). RESULTS gains: the v2
section (gate table, pack table, the like-for-like LIVE-capacity comparison), the v2
launcher recipe rows (TBD until user-fired), and the explicit framing notes: v1 remains
the ONLY 2015-capacity-comparable lineage; the corrected Xavier fan-in scaling is a
MEASURABLE difference for the launchers to adjudicate, not a promised win.

## S4 -- Full-f32 mel (lands FIRST among the fast changes)

f32 twins of `apply_filter_bank` / `apply_dct` / the deltas + delta-deltas + SDC
assembly + `assemble_input_sequence`, transcribing the quirk list `mel.rs`'s module doc
names (the deltas-no-DCT static-block OVERWRITE layout; SDC's unconditional n=3
regression kernel; the ignoreFirst branch-B first-column drop). Lands at BOTH widen
sites ATOMICALLY -- offline (`FastPipeline::build_input_sequence`) and the streaming
window variant (`assemble_perio_window`) -- or offline-vs-streamed fast diverges and
the phase-8/9 bit-equal gates fail by construction. The exact `mel.rs` stays
byte-untouched; the f64-widen path is DELETED from the fast tree, not kept as a mode.
EVERY fast-vs-exact pin re-measured-then-pinned in this task's single sweep
(phase7_parity_{sad,lid}, phase9_fast_parity, phase9_stream_causal posterior pins, the
corpus metric tiers) -- boundary/argmax identity must hold; ANY flip is an R1 STOP.
RSS re-benched on the phase-7 recipe: the ~68 MB fast plateau is expected to drop
toward ~56 MB (the ~23.5 MB per-channel widen buffer dies) -- measured, recorded.

## S5 -- Bidirectional f32 cell twins

`FastBiCell`: forward step loop + reversed step loop (the exact tree's
flip-rows-around-forward semantics) + hcat + the shared per-row `DenseRowChain`, driven
by a FRESH windowed-overlap accumulate/average loop written against the already-shared
span helpers (`window_begin`/`window_end`). `FastBlstm` + `overlap_window_step` are
BYTE-UNTOUCHED -- the phase-7/8 parity suites passing WITHOUT EDITS is the
behavior-preservation proof (approach A, ratified). Dispatch: `classify_fast_shape`'s
bidirectional-new-cell bail flips to `BiCell(cell)` for {slstm, mamba, cfc}; the
windowed regime follows the shape (BiCell runs OVERLAP or plain per the config, exactly
as the exact tree does). Parity per cell: boundary count/type identity, max_dt exactly
0.0, posterior deltas measured-then-pinned, R1 on flips; bench rows per cell. NO
streaming leg -- bidirectional is unstreamable by construction (whole-sequence
lookahead), stated in module docs. The Twin's frozen-SAD-net conservative gate
(`bail_unsupported_shape`) is revisited ONLY if a covering gate exists; otherwise it
stays conservative with a doc note.

## S6 -- Forward-only LSTM fast twin

`FastCell::Lstm`: a per-step peephole-LSTM kernel -- `dot_f32` per-step from day one
(the DenseRowChain lesson: the batched `lstm_layer_forward` differs in the last f32 ULP
at re-chunked granularities; NEVER reuse it for a streaming-shared kernel), the exact
`layers.rs` per-timestep body transcribed op-for-op (12-row peephole bundle, gate
order, `expLimit` guards). `FastCausalNet` + `StreamCausal` + the CLI/PyO3 inherit it
with zero new surface; `classify_fast_shape` (lstm, forward) flips from bail to
`Causal(Lstm)`. Parity legs (exact fwd-LSTM vs fast, streamed vs offline) + the bench
matrix's n/a cell filled -- the windowed-vs-causal regime decomposition gains its LSTM
control at the fast tier.

## S7 -- Mamba exact-path retention gating

An inference-only retention flag threaded at CONSTRUCTION time from the same condition
that gates the backward (backprop active AND references present -- the `blstm.rs:1452`
clone-gate precedent; NEVER keyed off `Epochs`, the F11 lesson: `Epochs 0` +
`BackPropagationActivated` still harvests gradients). When inference-only: skip the
`MambaCache` fill and `Network::layers_output` retention (both are backward-only
consumers -- verified by the survey: zero non-backward readers). SANCTIONED-CLASS
ADJUDICATION UP FRONT: this touches `nn/` outside `cells/`; it is declared here as
phase 10's ONE new sanctioned behavior-free touch class -- behavior-free because the
skipped structures have no forward-path reader -- with the committed golden suite
byte-green as arbiter, an explicit F11-seam regression leg (the seam still returns
nonzero finite gradients under `Epochs 0` + backprop), a run-twice pin, and a
re-benched exact-Mamba RSS row (107.8 -> ~55 MB expectation, measured). The R6
static-lane clone note: a retained cache must not be duplicated across per-lane bag
clones when inference-only (the flag makes the cache empty, so the clone cost dies
with it -- asserted, not assumed).

## S8 -- Stream-LID surface

`speech stream-lid` PORT-ONLY subcommand (the `stream`/`bench` precedent) replaying
per-utterance entries through `StreamingLidSession`, one line per utterance + a final
summary; a PyO3 `StreamingLidSession` binding (thin wrapper, the phase-8 T6 pattern:
push_utterance -> score tuple, finish -> aggregate; every numpy crossing a copy).
Pinned by CLI + binding smoke legs against the committed phase-8 stream-lid fixtures.
This closes the deferral that fell off the phase-9 spec's Deferred list (its only
record was stale phase-8 wording).

## S9 -- Verification protocol

**S9.1 Ordering rule:** S4 (f32 mel) lands BEFORE S5/S6 so every new twin pins against
the final front-end; the mel task owns the one re-pin sweep of existing fast numbers.

**S9.2 CfC tiers** (phase-9 verbatim): unit FD (shapes x seeds, per-shape
measured-then-pinned, in-test STOP over 1e-4, resolvable-count/structural asserts);
seam-tier committed synthetic fixtures ({bi, fwd}) + Engine `grad_check` at
measured*10; init reconstruction pins; the v2 gate matrix (S3.3) as convergence; fast
parity + causal streaming gate rows (cell-generic machinery, new rows only); LID
mechanical wiring (`BLSTM_LID_Cell_Type cfc` constructs + round-trips + grad_checks).

**S9.3 Standing suites:** every committed gate suite green at every task boundary; the
S4 sweep is the ONE sanctioned pin re-measurement; goldens byte-green throughout (the
exact tree's only touch is S7's adjudicated class + the CfC cell additions inside
`cells/`).

**S9.4 Mutation battery** (~8, apply-FAIL-revert-PASS, tree clean between):
(1) CfC gate inversion (swap `1-g`/`g`) -> FD + forward pins; (2) backbone block-order
swap in the Python builder -> init reconstruction pins; (3) v2 regression (11 -> 23 in
v2's config) -> the zero-dead-columns inverse guard; (4) f32-mel ignoreFirst-branch
drop -> the re-pinned parity legs; (5) bi-twin reverse-pass skip (forward-only both
halves) -> bi parity legs; (6) fwd-LSTM batched-projection substitution -> streaming
bit-identity legs (arithmetic, so the parity tier owns it -- the phase-9
threading-vs-arithmetic doctrine says the streaming gate alone would be blind; verify
the doctrine holds); (7) retention flag inverted (retain in inference / skip in
training) -> the F11 regression leg + goldens; (8) stream-lid session state leak across
utterances -> the prefix/aggregate legs. Honest gaps recorded.

**S9.5 Docs:** CLAUDE.md module rows (cfc.rs, fast/cells.rs, fast driver/mel rows, the
v2 lineage in the baseline row + Roadmap bullet), README (phase bullet + v2 launcher
recipe), RESULTS (v2 tables, re-pinned fast numbers, RSS rows, bench), IMPROVEMENTS
(battery only; port-only divergences stay in module docs + RESULTS).

## S10 -- Exit gate

1. CfC passes both grad tiers at pinned tolerances, both directions; param-matched per
   lineage; its v2 gates pass hard beat-init deterministically.
2. The v2 lineage is baselined: 8/8 subset gates, the zero-dead-columns guard, the
   like-for-like live-capacity table, the `sad-v2` launcher recipe in RESULTS.
3. The (cell x direction) fast matrix is RESOLVED: every combination either implemented
   with pinned parity (bi x {slstm,mamba,cfc}, fwd x {lstm,slstm,mamba,cfc}, bi-LSTM =
   the existing FastBlstm) or deliberately bailed with the reason documented (none is
   expected to remain bailed).
4. RSS wins measured on the bench recipe: the fast plateau post-mel, exact-Mamba
   post-gating; RTF rows for the new twins.
5. Goldens byte-green; the battery recorded; docs refreshed; the stream-lid surface
   landed.

## R -- Standing rules

- **R1 STOP-and-adjudicate** on any decision flip or nonzero boundary max_dt in any
  parity/streaming gate; never silently widen (the S4 sweep re-pins ONCE, by design,
  with flips still forbidden).
- **R2 exact-tree byte-freeze** outside: (a) the CfC additions inside `cells/` + the
  compile-enforced dispatch arms, (b) S7's adjudicated retention class. Goldens arbiter.
- **R3 license hygiene**: unchanged (corpus-gated tests, synthetic fixtures, no corpus
  identifiers committed).
- **R4 measure-then-pin**: every tolerance/budget/RSS/latency number measured first;
  sizing (S1.4) and pack lengths (S3.2) measured at implementation, never transcribed
  from survey estimates.
- **R5 honest comparison**: subset numbers carry the thinness caveat; the Xavier-scaling
  payoff is framed as measurable-not-promised; no architecture claim from subset scale.
- **R6 memory-pressure protocol** (standing, from the phase-9 host incidents): every
  cargo build/test/clippy capped `-j 4`, test execution `--test-threads=4`, no
  concurrent compile-heavy commands, targeted test invocations over full suites, at
  most one full-suite run per task.

## Deferred (named, not lost)

- **Transformer encoder** -- with the design trap ON RECORD: whole-sequence f64
  attention is 1.8-16 GB/layer at full-corpus lengths, so the exact cell must be
  DEFINED as windowed/bounded-KV attention from day one or streaming bit-identity is
  unsatisfiable; param budget at width 24 constrains d_ff. Its own phase.
- **The streaming endgame**: provisional-begin (T9 UPDATED the design -- full-segment
  retraction semantics required, not right-trim-only; six pinned finality surfaces to
  re-contract), global-holdback tightening (phase-level: derive code-side, pin,
  re-prove leftward dominance; buys 0.60447 s off every bound), and the `raw_segments`
  quadratic-replay fix (O(segments) smoothing replay per push on unbounded streams --
  the online posture's real scaling gap).
- Streaming-native SAD lineage (norm/window changes beyond 11-wide).
- sLSTM heads/block-diagonal + mLSTM; Mamba recompute-on-backward; per-net Mamba
  overrides; QPSO over new-cell hyperparameters.
- The stale IMPROVEMENTS provisional-begin line-ref (`fast/stream.rs:685-697` now
  points at StreamCausal) -- fix rides whichever phase takes the streaming endgame.
