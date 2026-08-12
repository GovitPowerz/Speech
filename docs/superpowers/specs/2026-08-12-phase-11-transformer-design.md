# Phase 11 -- The Transformer Encoder (windowed causal attention behind the CellLayer seam)

Date: 2026-08-12. Status: user-approved design (sections 1-3 approved in conversation).
Branch: `feature/phase-11-transformer` off `main@f0783d5` (PR #17 merged, the resmooth
interstitial included).

## Goal

Land the FIFTH `CellLayer` variant -- a pre-norm transformer block over a CAUSAL
SLIDING-WINDOW of bounded width `W`, with ALiBi relative positions -- completing the
user's named architecture list (mamba, xLSTM/sLSTM, CfC all landed; transformers open).
The phase runs the proven per-cell template end to end: exact f64 cell + hand-derived
backward (FD + seam tiers), config vocabulary, Python init builder + sizing, BOTH f32
fast twins (the total-matrix invariant), the streaming row, from-scratch subset gates on
`lre_sad_v2`, retention gating for the attention cache DAY ONE, a mutation battery, and
the docs refresh. One early behavior-free task consolidates the `fast/cells.rs` /
`fast/bicell.rs` shared scaffolding BEFORE the fifth cell multiplies the duplication
(approach A, user-selected).

THE DESIGN TRAP, answered by definition (phase-10 Deferred, on record): whole-sequence
f64 attention is 1.8-16 GB/layer at full-corpus lengths, so the exact cell is DEFINED as
windowed/bounded-KV attention from day one. The exact tree's backward cache is then
`T x W`-bounded, and streaming bit-identity is satisfiable because the streamed KV ring
computes the SAME windowed attention the offline forward does.

## Non-goals

- Whole-sequence (unwindowed) attention in any tree, at any precision.
- Cross-attention, decoder shapes, KV-cache quantization, flash-style kernel fusion.
- RoPE / learned positions (ALiBi user-selected; irregular-dt is a different cell,
  the CfC precedent).
- Training on the fast path (posture unchanged: exact f64 only), multi-channel
  streaming, the streaming endgame (provisional-begin + holdback tightening -- phase-12
  candidate), mLSTM / sLSTM heads (still Deferred).
- Full-corpus headline runs (user-fired launchers, as in phases 6/9/10).

## S1 -- the exact cell (`nn/cells/transformer.rs`, enum variant `Transformer`)

S1.1 FORWARD, per timestep `t`, with `H = out`, `A = heads`, `d = H/A`, window `W`
(edge-truncating at the sequence start: frame `t` attends over
`j in [max(0, t-W+1), t]`, the house edge-truncating-convolution convention):

1. Width adapter (the Mamba shape-closure precedent; the residuals add the ADAPTED
   input): `x'_t = W_a x_t + b_a`, `in -> H`.
2. MHA sub-block, pre-norm: `u_t = RMSNorm_1(x'_t)` (the in-tree Mamba RMSNorm,
   `eps 1e-5`, learned gain `g_1`); combined projection `[q|k|v]_t = W_qkv u_t + b_qkv`
   (ONE `3H x H` matrix, the sLSTM/CfC combined-block precedent); per head `h`, logits
   over the window
       `l_{t,j} = (q_t . k_j) / sqrt(d) + m_h * (j - t)`
   with `m_h` the standard ALiBi geometric slopes `2^(-8h/A)` (h = 1..A), ZERO learned
   parameters; softmax over the window row (see S1.3); `attn_t = sum_j p_{t,j} v_j`
   per head, heads concatenated -> `H`; residual 1: `s_t = x'_t + W_o attn_t + b_o`.
3. FFN sub-block, pre-norm: `v_t = RMSNorm_2(s_t)` (gain `g_2`);
   `h_t = s_t + W_2 silu(W_1 v_t + b_1) + b_2`, `d_ff` the sized width (S3), SiLU
   reused from the Mamba cell.

Output `h_t` (width `H`). No output softmax in the cell (the downstream output MLP owns
classification). Under `Direction::Bidirectional` the reverse stack runs the SAME
back-looking cell on the time-reversed sequence -- exactly how every other cell gets its
bidirectionality; no cell-side branch.

S1.2 FLAT LAYOUT, walked once by `for_each_slot` (set/get/derivatives cannot drift):
`[W_a | b_a | g_1 | W_qkv | b_qkv | W_o | b_o | g_2 | W_1 | b_1 | W_2 | b_2]`, matrices
row-major. `nb_of_weights = H*(in+1) + H + 3H*(H+1) + H*(H+1) + H + d_ff*(H+1)
+ H*(d_ff+1)`.

S1.3 SOFTMAX CONVENTION, a DECLARED divergence from the F8 conditional guard: the
attention softmax ALWAYS subtracts the window-row max. Justification: there are no
legacy bytes to match (port-only cell), and the f32 twin requires the headroom
(`ln(f32::MAX) ~ 88.72` vs f64's ~709.78; untrained logits spike). Op order pinned:
ascending loops, max-then-exp-then-normalize, one convention in both trees so the f32
twin is a precision-only divergence.

S1.4 DERIVED DEAD BLOCKS, both pinned `== 0.0` in the cell's own unit tests:
- `b_k` (the k rows of `b_qkv`) is gradient-dead ALWAYS: `q_t . b_k` is constant across
  the window row, annihilated by softmax shift-invariance. This cell's `b_i` analogue
  (sLSTM precedent): seeded 0, never moves, excluded from FD majors, pinned for every
  `T` (no contrast needed -- dead unconditionally).
- The q and k rows of `W_qkv` (and `b_q`) are dead at `T = 1`: every window has exactly
  one element and a one-element softmax is constantly 1, so `dq = dk = 0` exactly. The
  Mamba-`A_log`-at-`T=1` analogue, pinned with a `T = 2` contrast.
Nothing else is derived dead: `b_q` is LIVE for `T >= 2` (`b_q . k_j` varies in `j`),
RMSNorm gains are live, `v`/`W_o`/FFN blocks are live.

S1.5 BACKWARD, hand-derived reverse walk: FFN adjoints (SiLU derivative as the Mamba
cell already states it), RMSNorm backward (the projection-minus-mean-free form, reusing
the Mamba derivation shape), residual fan-ins, per-window-row softmax Jacobian
(`dp = p .* (dl - sum(p .* dl))`), the `1/sqrt(d)` and ALiBi terms (ALiBi contributes
NO parameter gradient -- slopes are constants -- only the additive logit path), and the
q/k/v scatter back through the window (a value at `j` receives contributions from every
`t >= j` whose window covers it: the reverse-time fold mirrors Mamba's conv-fold
ordering constraint). Backward caches: per-row attention weights (`T x W` per head),
`u`/`s`/`v_t`/FFN pre-activations -- all `T x O(W + d_ff)`-bounded, never `T^2`.

S1.6 TIERS, both phase-9 harnesses reused UNCHANGED (they are cell-agnostic; the seam's
claim cashed again):
- FD tier (`tests/phase9_cell_grad.rs`): shapes covering `t in {1, 2, W-1, W+3, 23}` at
  minimum, with a SMALL fixture window (e.g. `W = 4`) so the `t <= W` and `t > W`
  regimes BOTH appear at FD-tractable lengths -- the window edge is the new structural
  axis -- x 3 seeds; eps MEASURED by sweep before pinning; per-shape pins at
  measured*10; `max_rel_major` under the `1e-4` STOP (R4); resolvable-count asserts
  carrying the S1.4 structural claims (predicted == measured dead counts).
- Seam tier (`tests/pyo3/test_phase9_seam.py`): committed synthetic fixtures
  `{transformer} x {bi, fwd}` SAD + a mode-7 Twin carrying
  `BLSTM_LID_Cell_Type transformer`; corpus-level `grad_check` through
  `speech_rs.Engine`, pins measured*10, in-test assert no pin exceeds `1e-4`.

## S2 -- config vocabulary (KEY_TABLE 187 -> 190)

Three port-only keys on the Mamba/CfC UNPREFIXED precedent (one transformer geometry
per config, shared by whichever net selects the cell), section `[nn_transformer]`:
- `Transformer_Window` (`window`): `W` in CELL rows (post-subsampling), default 64.
- `Transformer_Heads` (`heads`): `A`, default 4; validation `A >= 1` and `H % A == 0`
  at `from_legacy` (a non-dividing head count is a config error, not a truncation).
- `Transformer_D_Ff` (`d_ff`): default = the S3 sized constant; validation `>= 1`.
`BLSTM{,_LID}_Cell_Type` gains the value `transformer`. Absent keys mean the defaults,
so every pre-phase-11 config decodes byte-unchanged (pinned, the standing rule).
`config_bridge.py::_transformer_geometry` mirrors the reads with the raise-on-invalid
convention; the Rust and Python defaults are pinned against each other by the CfC
parse-the-Rust-source mechanism.

## S3 -- sizing (the CfC dual-lineage procedure)

Heads and window are PARAM-FREE (ALiBi has no weights; `A` only reshapes); the sized
constant is `d_ff`. Record the closed form per lineage from S1.2 summed over the stack
(+ output MLP + normalize tail), then choose `d_ff` preferring an integer inside BOTH
lineage bands (the CfC convention: v2 argmin AND v1's +-15% band if satisfiable; if the
two conflict, v2 -- the defect-free lineage -- wins the tiebreak and the conflict is
RECORDED).

PROVISIONAL arithmetic (the sizing task re-derives and pins; these numbers are
design-time estimates, not commitments): v2 bidirectional cells
`2 * [(3552 + 49 d_ff) + (3072 + 49 d_ff)]` + output `601` + tail `22`
`= 13871 + 196 d_ff` vs the v2 LSTM pack 24431 -> argmin near `d_ff ~ 54`; v1's band
needs `d_ff >= ~64`, and `d_ff = 64` sits inside BOTH (v2 ~ +8%, v1 ~ -15%). If the
re-derivation confirms, the default is 64; the finding to DOCUMENT either way: windowed
attention at width 24 is parameter-cheap next to a peephole LSTM, so `d_ff` lands well
above the textbook heuristic to hold capacity comparability -- state it, do not hide it.

## S4 -- the dedupe task (early, behavior-free; the phase-10 named follow-on)

Consolidate what `fast/cells.rs` and `fast/bicell.rs` duplicate -- the per-cell stack
driving loop, state-vector plumbing, and dense-tail wiring -- into ONE generic driver
both consume, BEFORE the fifth cell lands. Constraints:
- PURE CODE MOTION: zero arithmetic reorder/hoist/fusion; every reduction stays the
  ascending contiguous loop it was.
- THE ARBITER is the committed suite UNEDITED: every phase-7/8/9/10 parity + streaming
  leg and the resmooth oracle pass with their EXISTING pin values. Bit-identity, not
  tolerance: parity pins do not move, streamed-vs-offline stays `to_bits`-equal.
- `FastBlstm` / `overlap_window_step` stay BYTE-UNTOUCHED (the phase-8 sacred kernels;
  the dedupe scope is the CELL scaffolding, not the windowed-BLSTM path).
If the extraction cannot be made behavior-free for some fragment, that fragment stays
duplicated and the report says why -- partial dedupe is acceptable, silent arithmetic
drift is not (R1).

## S5 -- the f32 twins (the total-matrix invariant holds)

S5.1 `FastTransformer` (in `fast/cells.rs`, post-dedupe): the causal step kernel.
STATE = the bounded KV ring -- the last `min(t+1, W)` (k, v) rows per head -- plus
nothing else; ALiBi is relative, so absolute stream position never enters the state.
`step(x_t, state)`: adapter -> RMSNorm -> qkv row -> attend over the ring (a ring
shorter than `W` at stream start IS the offline edge-truncation, which is what makes
offline-vs-streamed bit-identity hold by construction) -> residual -> FFN -> push
`(k_t, v_t)`, evicting the oldest at capacity. The whole-sequence `feed_forward` LOOPS
this step (the phase-9 kernel-sharing contract).

S5.2 DIVERGENCE DISCIPLINE, the four phase-9 classes carried verbatim: f32 arithmetic
with op order preserved element-for-element; EVERY projection and attention reduction a
per-step `dot_f32` over contiguous rows -- NO batched faer inside the step (the
battery-item-6 design-time control, restated in the module doc where the violating edit
would be made; the dense output MLP stays on the shared per-row `DenseRowChain`); the
f32 exp guards at `ln(f32::MAX)`; no `t == 0` special-casing that a resumed session
would break. Softmax max-subtract matches the exact cell exactly (S1.3), so the twin is
precision-only.

S5.3 THE MATRIX STAYS TOTAL: `classify_fast_shape` gains `Causal(Transformer)` and
`BiCell(Transformer)`; `cell_weight_count`'s exhaustive match makes a missing kernel a
COMPILE error (the phase-10 invariant). `FastBiCell` drives the same back-looking
kernel on the reversed sequence for the reverse stack -- no transformer-specific bicell
code beyond the enum arm.

S5.4 PARITY (`tests/phase9_fast_parity.rs` + `tests/phase10_bicell_parity.rs` extended,
or phase-11 siblings if cleaner): boundary count/types IDENTICAL, `max_dt` EXACTLY 0.0,
argmax zero flips (R1 STOP on any flip); posterior deltas measured-then-pinned at the
house convention with the crossing legs made non-vacuous by the existing `Lever` ladder
(gain-1-first, so prior cells' rows stay byte-identical).

## S6 -- streaming (the fifth `StreamCausal` row)

`StreamCausal` drives `FastCell`/`FastCellState` generically -- the transformer row
lands with ZERO machinery change (the S5.2-phase-9 claim, cashed a third time; phase 10
cashed it for cfc and lstm). Gate (`tests/phase9_stream_causal.rs` extended): streamed
`finish()` BIT-FOR-BIT equal to the offline fast causal run, `max_dt` EXACTLY 0.0 over
a committed `>= 2` interior-boundary floor (the crossing sweep manufactures real
boundaries; aim for the sibling-strength floors, the phase-10 rider-9 standard),
posteriors bit-equal, chunk-invariant at 20/100/1000/7 ms by `to_bits`, prefix zero
retractions. LATENCY: a causal windowed-attention cell has NO lookahead -- the derived
bound is UNCHANGED at the fixture's `1.73400 s` (and the real-arm `2.82357 s`
derivation), pinned tight at `bound + PUSH_CHUNK_S`. The KV ring adds `W x 2H` f32 per
layer of session state -- constant, stated in the module doc.

## S7 -- retention gating, day one (the phase-9 Mamba lesson applied proactively)

The attention cache (`T x W` weights per head + sub-block intermediates) is the largest
backward cache of any cell yet, so `CellLayer::set_retain_cache`'s transformer arm
GATES from day one (the phase-10 T9 mechanism: inference-only =
`!BackPropagationActivated` at construction, the loud-panic choke point on a dropped
buffer, structural soundness argued from the private-backward funnel). Measured
before/after peak RSS lands in RESULTS.md with the bench rows. The other three cells'
retention arms stay no-op follow-ons (unchanged scope).

## S8 -- the Python side + gates

- `init_weights.py::init_transformer_flat` + `transformer_geometry(spec)`: the S1.2
  flat order emitted directly (no nnet domain), Xavier/He per block with fans from the
  block shapes, `b_k` seeded 0 (S1.4), RMSNorm gains seeded 1.0, biases 0. NO magic
  constants (stated openly, the CfC precedent): every block is a plain dense shape.
  MANDATORY whole-pack block-by-block reconstruction pin + at least two shear
  companions (a q/k-family permutation; a deeper-FFN variant), all self-mutation-probed.
- `drivers/baseline.py`: `transformer` joins `--cell-type`; `cell_overlay` writes the
  same forward-forced derived keys as today; knobs stay SAD-arm-only (the frozen-SAD
  Twin reason, unchanged).
- THE GATE MATRIX grows 8 -> 10 (`tests/pyo3/test_phase10_gates.py` extended or a
  phase-11 sibling): `transformer x {bidirectional, forward}` on `lre_sad_v2`, HARD
  beat-init at every collar, first-run, run-twice byte-identical, the zero-dead-columns
  inverse guard running unchanged. Corpus-gated, local-only, license hygiene absolute.

## S9 -- verification standards + battery + docs

Everything measure-then-pin; no pin widens silently (R1/R3). The battery (S9.4-style,
>= 8 items, each named to the leg that must catch it): candidates include an ALiBi
slope sign flip, a window off-by-one at the edge (`t-W` vs `t-W+1`), removing the
softmax max-subtract in the f32 twin, a KV-ring eviction-order swap, a `b_k`-liveness
mutation (must break the dead-block pin, not a behavioral leg), a dedupe-regression
probe (re-introduce one duplicated divergent copy), a reconstruction-pin shear, a
`T=1` q/k-dead contrast break. Honest gaps recorded, not argued away. Docs refresh:
CLAUDE.md rows (`nn/cells/*`, `fast/cells.rs`, `fast/driver.rs`, `fast/stream.rs`
matrix mentions, KEY_TABLE count, Roadmap 2 bullet), RESULTS.md phase-11 section,
IMPROVEMENTS.md UNTOUCHED (port-only divergence lives in module docs + RESULTS, the
standing rule). Memory + ledger per the house loop.

## Risks

- R1 STOP-AND-ADJUDICATE: any decision flip (boundary, argmax, gate regression) or pin
  breach stops the task; never widen silently.
- R2 EXACT-TREE DISCIPLINE: sanctioned touches outside `cells/` are ONLY the enum arm,
  the KEY_TABLE rows, and the retention arm (already-adjudicated class); the committed
  golden suite byte-green is the proof, checked at every task boundary.
- R3 MEASURE-THEN-PIN: eps by sweep, pins at measured*10, bench budgets wide-CI +
  local bounds.
- R4 THE 1e-4 STOP on every gradient pin (FD + seam), `max_rel_major` discriminating.
- R5 REPORT-NEVER-GATE for vs-other-cells comparisons (subset ties are recorded; the
  launcher runs referee).
- R6 MEMORY PROTOCOL (standing, two host reboots on record): every cargo `-j 4`, every
  test run `--test-threads=4`, never two compile-heavy commands concurrently, targeted
  runs while iterating, at most ONE full-suite run per task, FOREGROUND-only subagent
  commands.

## Exit gate

1. FD + seam tiers green with every pin under the 1e-4 STOP; the S1.4 dead blocks
   pinned `== 0.0` with their contrasts.
2. Both f32 twins: boundary identity + zero flips, posteriors measured-then-pinned;
   `classify_fast_shape` still total (compile-enforced).
3. Streaming: the fifth causal row bit-for-bit, chunk-invariant, >= 2 real interior
   boundaries, latency bound unchanged.
4. The 10-gate matrix passes HARD beat-init, run-twice byte-identical.
5. The dedupe proven by the UNEDITED suite at existing pins.
6. Retention measured (exact-tree transformer RSS with/without backprop).
7. Battery >= 8 named items; docs refreshed; suite counts recorded.

## Deferred (named, not lost)

- The streaming endgame (provisional-begin full-retraction semantics + global-holdback
  tightening; the stale IMPROVEMENTS provisional-begin line-ref rides it) -- phase-12
  candidate.
- RoPE / irregular-dt attention variants; mLSTM; sLSTM heads/block-diagonal; Mamba
  recompute-on-backward; per-net geometry overrides; QPSO over new-cell hyperparams.
- Remaining phase-10 follow-ons not taken here: BiCell corpus tier, the one-hot
  bank-table value pin, the L >= 2 CfC fixture, retention for the other three cells,
  the sub-unit gain rung, symbol citations, the Mamba triple-defaults source pin, the
  tagged pack header.
- Full-corpus launcher runs (v1 + sad-v2 + the new cells) -- user-fired, post-phase.
