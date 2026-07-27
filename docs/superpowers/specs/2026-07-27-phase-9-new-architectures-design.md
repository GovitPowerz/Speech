# Phase 9 -- New Architectures (Mamba + sLSTM behind a cell abstraction) -- Design

Date: 2026-07-27
Status: approved (design sections 1-4 user-approved in-conversation)
Branch: `feature/phase-9-new-architectures`

## Goal

Land two modern recurrent architectures -- Mamba (S6, recurrent form) and sLSTM (the xLSTM
paper's scalar-memory cell) -- as first-class citizens of the existing engine: trainable FROM
SCRATCH through the phase-5/6 machinery (seeded init, SMORMS3 via the seam, `train_modern`,
the moving `NNCostSeg` validation signal), hand-derived analytic backward passes verified by
`grad_check`, and, for the causal (forward-only) direction, a streaming path that drops the
phase-8 windowed-lookahead term entirely -- the roadmap's named "truly causal" upgrade.

User-ratified scope decisions (2026-07-27):

1. **Training path**: hand-derived forward + analytic backward in Rust, plugged into the
   existing `Layer` trait and the SMORMS3 seam; `grad_check` pins every backward. No autodiff
   dependency, no torch-train/rust-infer split.
2. **Architectures this phase**: Mamba (S6) + xLSTM (sLSTM variant). Transformer/CfC follow
   behind the then-proven abstraction in a later phase.
3. **Task scope**: SAD gated, LID wired -- full from-scratch subset gates on the SAD arm for
   both cells; the Twin's LID net becomes config-selectable to the new cells with mechanical
   tests (shapes, weight round-trip, grad_check) but no from-scratch LID convergence gate.
4. **Streaming**: causal streaming included -- exact f64 tree (train) + f32 fast twins
   (infer) + a causal streaming path; latency re-derived and pinned. The provisional-begin
   silence-class refinement stays DEFERRED (IMPROVEMENTS `[phase8 -> phase9]` entry unchanged).

## Non-goals

- Transformer encoder and CfC (phase 10+ behind the abstraction).
- Bidirectional f32 fast twins for the new cells (offline bi-direction quality arms run in
  the exact tree; a bidirectional fast twin is a mechanical follow-on -- named, not landed).
- Provisional-begin (revisable-emission) streaming semantics.
- Block-diagonal/multi-head sLSTM recurrence and mLSTM (matrix memory).
- Recompute-on-backward memory trimming for the Mamba BPTT cache (named trim, subset-scale
  training does not need it).
- Any full-corpus training run (user-fired post-phase, the phase-6 posture).
- QPSO-searchable new-cell hyperparameters (the DSP/config genome does not grow this phase;
  new-cell hyperparams are config keys only).

## S1 -- The cell abstraction (exact f64 tree)

**S1.1 The cut is at the layer, not the network.** `Network<L>` is already generic over the
10-method `Layer` trait (whole-sequence `Array2<f64>` in/out: `feed_forward`,
`feed_forward_reverse`, `feed_backward`, `feed_backward_reverse`, `get_weights_derivatives`,
`reset_weights_derivatives`, `ponderate_weights_derivatives`, `set_weights`, `get_weights`,
`nb_of_weights`). New module family:

```
src/rust/src/nn/cells/mod.rs      -- enum CellLayer { Lstm(LstmLayer), Slstm(SlstmLayer), Mamba(MambaLayer) }
src/rust/src/nn/cells/slstm.rs    -- SlstmLayer (forward + analytic backward)
src/rust/src/nn/cells/mamba.rs    -- MambaLayer (forward + analytic backward)
```

`CellLayer` implements `Layer` by 3-arm match delegation (the `Processor`-enum precedent:
closed set, static dispatch, concrete methods reachable). `Blstm`'s two recurrent stacks
change type `Network<LstmLayer> -> Network<CellLayer>`; `Network` itself and the output
`Network<NeuronLayer>` are untouched. Everything above the cell -- the four windowed drivers,
input normalization, scoring/cost accumulation, the flat-weight seam, `train_modern`,
batching -- never sees the cell type. The existing `impl Layer for LstmLayer` block in
`network.rs` stays (unit tests use it); `CellLayer::Lstm` wraps the same struct.

**S1.2 Direction.** A port-only structural mode, per net: `bidirectional` (default; the
legacy shape) or `forward` (causal: only the forward stack is built, `backward_network =
None`, and the output MLP consumes `hidden` columns instead of the hcat'd `2*hidden`). The
forward-only branch lives in `feed_forward_double`/`feed_backward_double` (skip the hcat and
the reverse stack), which makes it mechanically available under every windowed driver; it is
DEAD on every legacy config. Windowed modes with a causal cell are pointless-but-defined
(window boundaries reset state); the phase's causal configs use window 0 (plain FFB) and the
streaming session REQUIRES window 0 (S5.3). Bidirectional new cells come free via the
wrapper's existing reverse trick (flip input -> forward -> flip output -- cell-agnostic).

**S1.3 Weight seam.** Each cell defines its flat layout ONCE, by its own
`set_weights`/`get_weights` order (S2.2/S3.2); `nb_of_weights` and the `[deriv | count]`
Nx2 assembly follow the same order, exactly as `LstmLayer` does today. New-cell weights live
ONLY in `.bin` packs / seam vectors -- never config-text weight blocks, never the genome
(weights are permanently masked out of it since phase 5) -- so the `adim_coeff` seam is a
non-concern for new cells BY CONSTRUCTION. The normalize mean/std tail stays a `Blstm`-level
concern (unchanged). `init_weights.py` gains per-cell builders emitting the Rust order,
pinned by cross-language length + bit-identical `speech_rs.Engine` round-trip tests (the
phase-0a/phase-5 pattern).

**S1.4 Exact-tree discipline.** The committed golden suite (948 tests) must stay byte-green
at every commit. THREE sanctioned touch classes, all behavior-free on LSTM configs:

- (a) the `CellLayer` enum wrap (LSTM f64 arithmetic untouched -- an extra match indirection
  only; the golden suite is the proof),
- (b) the `Direction` forward-only branches (dead on every legacy/default config),
- (c) the `KEY_TABLE`/config rows (S6).

Anything else touching `nn/` outside `cells/` must be justified against this list in review.
Numeric divergence of the new cells is BY DESIGN (port-only code, no legacy source) and is
documented in module docs + `RESULTS.md`, NEVER `IMPROVEMENTS.md` (the phase-7 rule).

## S2 -- sLSTM contract

**S2.1 Forward** (per timestep `t`; pre-activations from full matrices, single head):

```
a~ = W_a x_t + R_a h_{t-1} + b_a        for a in {i, f, z, o}
m_t = max(f~ + m_{t-1}, i~)             (stabilizer; m_0 per S2.4, so m_1 = i~_1)
i' = exp(i~ - m_t)        f' = exp(f~ + m_{t-1} - m_t)
c_t = f' c_{t-1} + i' tanh(z~)          n_t = f' n_{t-1} + i'
h_t = sigmoid(o~) * (c_t / n_t)
```

Reverse-direction execution reuses the wrapper's flip trick (no cell-side reverse state).

**S2.2 Flat layout** (mirrors the LSTM convention where it applies): gate blocks in the
order `[i | f | o | z]` (echoing the LSTM `[i | f | o | g]`), each block row-major
`[R (out x out) | W (out x in) | b (out)]`. NO peepholes, NO cell-narrow asymmetry -- all
four blocks are the same width. `nb_of_weights = 4 * out * (out + in + 1)`.

**S2.3 Backward.** Reverse-time BPTT with the stabilizer `m` held CONSTANT -- and that is
EXACT, not an approximation: by induction `c_t = C_t e^{-m_t}` and `n_t = N_t e^{-m_t}` for
the unstabilized `C, N`, so `c_t/n_t = C_t/N_t` is analytically m-invariant and
`dh/dm = 0`. This also removes the max() kink from the gradient path entirely (no
subgradient choice exists to make). Adjoint flows: `dh_t -> do~` (sigmoid') and
`d(c_t/n_t) -> dc_t, dn_t` (quotient rule); the twin adjoint recurrences carry
`dc`/`dn` backward through `f'` (adding each step's local contributions `di'`, `df'`,
`dz~` via `tanh'` and the as-computed `i'`/`f'` exp values); gate pre-activation deltas
fold into the four `[R | W | b]` accumulations with the `[deriv | count]` col1 =
frame-count convention. Caches for backward: per-timestep gate values (or
pre-activations), `c`, `n`, `m`, `h` -- O(T * out) each, negligible.

**S2.4 Init + numerics.** State zero-init (`c_0 = n_0 = 0`, `h_0 = 0`) and
`m_0 = -1e30` (documented constant standing in for -inf): at t=1 the max resolves to
`m_1 = i~_1` and `f'` multiplies the zero state, so the m_0 convention is value-inert
beyond selecting the official xLSTM `m_1 = i~_1`. `n >= i' > 0` for all t (exp is positive),
so `c/n` never divides by zero. Seeded init (`init_weights.py`): forget-gate bias `b_f = +1`
(the `forget_bias_one` echo), other biases 0, `W`/`R` per the repo Xavier/He convention.

## S3 -- Mamba contract

**S3.1 Forward** (one `MambaLayer(in, out)`, recurrent form; `d_model = out`,
`d_inner = expand * d_model`):

```
x' = P x_t + p                    ONLY when in != out (width adapter, out x in + bias); else x' = x_t
xn = RMSNorm(x')                  (learned weight g, eps 1e-5: xn = x' / sqrt(mean(x'^2) + eps) * g)
[u | res] = W_in xn               (2*d_inner x d_model, no bias)
uc_t = SiLU( causal_conv1d(u)_t + b_conv )   (depthwise, kernel d_conv, left-padded)
[dt~_t | B_t | C_t] = W_x uc_t    (dt_rank + 2*d_state rows, no bias -- B_t/C_t are
                                   PER-TIMESTEP functions of the input: the selectivity)
Delta_t = softplus( W_dt dt~_t + b_dt )      (d_inner)
Abar_t[c,s] = exp( Delta_t[c] * A[c,s] ),  A = -exp(A_log)
h_t[c,s] = Abar_t[c,s] * h_{t-1}[c,s] + Delta_t[c] * uc_t[c] * B_t[s]
y_t[c]   = sum_s C_t[s] * h_t[c,s] + D[c] * uc_t[c]
out_t = W_out ( y_t * SiLU(res_t) ) + x'     (d_model x d_inner, no bias; residual adds x')
```

`A < 0` and `Delta_t > 0` give `Abar_t in (0,1)`: the recurrence is stable by construction.

**S3.2 Flat layout** (row-major per matrix, in this order): `[P | p]` (only when
`in != out`), `g` (RMSNorm), `W_in`, `conv (d_inner x d_conv) | b_conv`, `W_x`,
`W_dt | b_dt`, `A_log (d_inner x d_state)`, `D (d_inner)`, `W_out`. The Rust
`set_weights` order IS the layout (S1.3).

**S3.3 Hyperparameters** (config keys, S6): `Mamba_D_State` (default 16), `Mamba_D_Conv`
(4), `Mamba_Expand` (2), `Mamba_Dt_Rank` (0 = auto `ceil(d_model/16)`). Applied per
mamba-cell net; layer width `d_model` comes from the existing `Layer_%d` size keys.

**S3.4 Init (LOAD-BEARING, spec-pinned).** `A_log[c,s] = log(s+1)` (S4D-real), `b_dt =
softplus^-1(Delta_0)` with `Delta_0` log-uniform in `[1e-3, 1e-1]` (seeded draw per
channel), `W_dt` uniform `+-dt_rank^{-1/2}`, `D = 1`, `g = 1`, projections + conv per the
repo Xavier/He convention. These are the S4/Mamba lineage inits; a from-scratch run without
them is known-fragile, which is why they are spec text and not implementation choice.

**S3.5 Backward.** Reverse-time through the elementwise recurrence (state adjoint):
`dh_t[c,s] = dy_t[c] * C_t[s] + Abar_{t+1}[c,s] * dh_{t+1}[c,s]`, then
`dAbar_t = dh_t . h_{t-1} -> dDelta_t, dA_log` via the exp product rule;
`d(Delta_t * uc_t * B_t)` fans to `dDelta_t`, `duc_t`, `dB_t`; `dC_t`, `dD` from the
output sum; softplus'
into `W_dt`; conv1d backward = correlation with the flipped kernel + per-channel bias sums;
SiLU' = `sigmoid(x)(1 + x(1 - sigmoid(x)))`; RMSNorm backward standard; matmul transposes
for the projections; residual adds `dx'` straight through; the width adapter transposes
when present. Caches: the `h_t` sequence (T x d_inner x d_state) + block intermediates --
~94 MB f64 transient for a 60 s file at `d_model 64 / expand 2 / d_state 16`; acceptable at
subset scale (recompute-on-backward is the named trim, a non-goal).

## S4 -- f32 fast twins (causal only)

**S4.1 Step kernel.** `fast/` gains `FastSlstm`/`FastMamba` (sibling module(s) beside
`fast/nn.rs`; exact file split is implementer latitude):
forward-only f32, built around an explicit `step(x_t, &mut State)` kernel -- carried state
`(h, c, n, m)` for sLSTM, `(conv ring, h)` for Mamba -- plus a whole-sequence wrapper that
loops the step. ONE kernel, two drivers: the offline fast path and the streaming session
run the SAME step function, so offline-vs-streamed bit-identity holds BY CONSTRUCTION (the
phase-7/8 shared-kernel precedent). Weights narrow f64 -> f32 once at construction,
post-pack, exactly like `FastBlstm::from_flat`.

**S4.2 Scope.** CAUSAL (Direction=forward) only. `Inference_Path fast` with a bidirectional
new-cell config typed-bails (pinned); bidirectional new-cell inference stays on the exact
tree this phase. The fast SAD driver (`FastSpectralSegmenter`) dispatches on cell type +
direction at construction; the output MLP forward reuses the existing fast dense/softmax
path on `hidden` (not `2*hidden`) columns.

**S4.3 Parity.** Exact-causal vs fast-causal offline: tolerance measured-then-pinned
(committed synthetic fixture legs + the real-checkpoint corpus tier), boundary count/types
IDENTITY, max_dt 0.0, argmax zero flips -- R1 STOP on any flip (the phase-7 rule verbatim).

## S5 -- Causal streaming

**S5.1 `StreamCausal`.** A new layer in `fast/stream.rs` replacing `StreamOverlap` in the
four-layer composition. Front-end (gain/preemph-carry/dither-index + reach-padded feature
rows), frozen type-1 norm, and `StreamDecision` (19-tap edge-truncating conv + latched
hysteresis + derived holdback + the consumed-frontier trigger) are ALL UNCHANGED. Per
finalized feature row: frozen-norm affine -> cell `step` -> output MLP row -> posterior ->
decision push. No window machinery, no output-row averaging, no EOS window flush (EOS
reduces to flushing the front-end's reach tail).

**S5.2 Session dispatch.** `StreamingSession::new(map, rate, channels)` reads cell type +
direction from the config and constructs the windowed-BLSTM path (phase 8, unchanged) or
the causal path. Same `push`/`finish` API -> the `speech stream` CLI and the PyO3
`StreamingSession` gain the causal mode with ZERO new surface.

**S5.3 Validation bails** (each pinned): causal streaming requires `Direction forward`
AND a NEW cell (`slstm`/`mamba` -> `FastSlstm`/`FastMamba`; a forward-only LSTM fast twin
is out of scope, same named-follow-on status as the bidirectional fast twins), window 0
(`window_size` resolves 0), MONO, `Audio_fixed_gain` present, `InputNormalizationType 1`
frozen tail, `output_size 1`, algo 3. Everything else typed-bails with the phase-8 wording
precedent.

**S5.4 Frozen-norm contract.** Identical to phase 8: `Audio_fixed_gain` (per-sample fixed
gain) + the pack-carried type-1 tail are the causality cut; streaming changes TIMING only.
The phase-8 caveats carry forward verbatim: gain-inertness on all-DCT/`IgnoreFirstDCT`
configs (any positive gain -> bit-identical posterior), and the identity-tail regime on
checkpoints whose normalize tail never trained.

**S5.5 Latency.** Structural bound = `feature_reach + conv_delay + holdback` (the
`nn_window` term DIES; no overlap lookahead exists). On the gate config's parameters that
is `0.14400 + 0.36000 + 2.28957 ~= 2.79 s` -- but the bound is DERIVED in-phase from the
actual config (the phase-8 derivation discipline), measured, and pinned per-class: SPEECH
within the bound; the OTHER (silence) commit-wait AREA term remains (inherent, per-class
pinned, provisional-begin deferred).

**S5.6 The streaming gate** (the phase-8 pattern): streamed `finish()` `Segmentation` +
posterior history BIT-FOR-BIT equal to the offline fast causal run on the same config;
boundary max_dt EXACTLY 0.0 (R1 STOP on nonzero); chunk-invariant (20/100/1000/7 ms);
prefix zero retractions.

## S6 -- Config vocabulary (port-only keys)

Flat keys + `KEY_TABLE` rows (the `Inference_Path`/`Audio_fixed_gain` precedent; TOML
canonical, `.config` accepted):

| Flat key | TOML | Values / default |
|---|---|---|
| `BLSTM_Cell_Type` | `[nn] cell_type` | `lstm` (default) / `slstm` / `mamba` |
| `BLSTM_Direction` | `[nn] direction` | `bidirectional` (default) / `forward` |
| `BLSTM_LID_Cell_Type` | `[nn.lid] cell_type` | same, for the Twin's LID net |
| `BLSTM_LID_Direction` | `[nn.lid] direction` | same, for the Twin's LID net |
| `Mamba_D_State` | `[nn.mamba] d_state` | int, 16 |
| `Mamba_D_Conv` | `[nn.mamba] d_conv` | int, 4 |
| `Mamba_Expand` | `[nn.mamba] expand` | int, 2 |
| `Mamba_Dt_Rank` | `[nn.mamba] dt_rank` | int, 0 = auto `ceil(d_model/16)` |

(Exact TOML section spelling settles at implementation against `toml_config.rs`'s existing
section conventions; the flat keys are the contract.) `NnetSpec`/construction reads the
cell/direction keys; absent keys mean the legacy shape (every existing config untouched).
The LID-prefixed pair exists for the mechanical LID wiring; Mamba hyperparams apply to
whichever net(s) select `mamba` (a per-net override is NOT provided this phase -- one
mamba geometry per config, documented).

## S7 -- Python-side extensions

**S7.1 `init_weights.py`.** Per-cell builders (`slstm`, `mamba`) emitting the Rust flat
order (S2.2/S3.2), seeded, with the S2.4/S3.4 scheme constants; the existing LSTM path is
byte-untouched. Pinned by length + bit-identical Engine round-trip per cell.

**S7.2 Baseline arm knobs.** The SAD arm (`drivers/baseline.py`) gains `--cell-type`
{lstm,slstm,mamba} and `--direction` {bidirectional,forward} knobs that overlay the S6 keys
onto the arm config + route init through the per-cell builders. Default = today's BLSTM arm
byte-identical. The subset gates (S8.2) run through this surface -- no new driver.

**S7.3 `train_modern`.** UNCHANGED by design (the seam is architecture-agnostic: flat
vectors + `forward_backward`). Any change requested by implementation reality is a spec
deviation to surface in review, not to absorb silently.

## S8 -- Verification protocol

**S8.1 Backward verification (the phase's load-bearing instrument).** Per cell x direction:

- UNIT tier (CI-visible, committed): central-diff vs analytic at tiny synthetic shapes
  (several seeds, several T incl. T=1 and odd T; tolerance measured-then-pinned, expected
  ~1e-6 relative on well-scaled points). Runs on every `cargo test`.
- SEAM tier: corpus-level `grad_check` through `speech_rs.Engine` on the gate configs
  (the analytic fold vs FD at the seam, the phase-4c/5 pattern), causal + bidirectional.

**S8.2 From-scratch subset gates** (phase-6 protocol: corpus-gated, local-only, <10 min
each, run-twice bit-identical, seeded): bi-sLSTM SAD + bi-Mamba SAD, sized param-matched to
the BLSTM arm (+-15%): HARD gate = trained beats own init on held-out DCF, deterministically.
The vs-BLSTM comparison (subset DCF 0.25) is RECORDED in `RESULTS.md`, NOT hard-gated (the
subset is thin and mode-collapsed; a cell could tie 0.25 by collapsing identically; the
full-corpus user-fired runs are the referee). The CAUSAL variants (fwd-sLSTM, fwd-Mamba)
train from scratch too -- beat-init HARD (the streaming artifact must converge honestly).

**S8.3 LID mechanical wiring**: cell-swapped Twin LID net passes construction/shape tests,
weight set/get round-trip, and grad_check; no convergence gate (scope decision 3).

**S8.4 Parity + streaming gates**: S4.3 + S5.6 as committed CI legs (synthetic fixtures) +
the corpus tier (real checkpoint, corpus-gated). Latency S5.5 derived-measured-pinned.

**S8.5 Exact-tree regression**: the full committed golden suite byte-green at every task
boundary; the S1.4 sanctioned-touch list is the review checklist for any `nn/` diff.

**S8.6 Mutation battery** (~8, apply-FAIL-revert-PASS, tree clean between cycles): (1)
stabilizer dropped (`m_t = 0`) -> sLSTM forward/grad pins; (2) `A_log` sign flip -> Mamba
pins (recurrence explodes); (3) conv-ring off-by-one -> chunk-invariance legs; (4)
step-state leak across files/utterances -> per-file gates + streaming prefix; (5)
`Direction` hcat misroute (feed 2*hidden on forward) -> shape/golden pins; (6) packer
block-order swap -> cross-language round-trip; (7) f32 streaming state not reset at
construction -> streaming gate; (8) init scheme swapped (A_log uniform) -> seeded-init pins
+ the from-scratch gate's determinism leg. Honest gaps recorded.

**S8.7 Bench.** `speech bench` rows for the new causal cells (fast path): RTF + peak RSS
vs the BLSTM fast baseline, measure-then-pin local budgets + the wide CI smoke. The
linear-time/no-window expectation (causal cells beat windowed-BLSTM RTF) is a HYPOTHESIS to
measure, not an assumption.

## S9 -- Exit gate

Phase 9 is done when, at branch end:

1. Both cells pass S8.1 (unit + seam grad checks, pinned tolerances) in both directions.
2. All four from-scratch subset gates (bi/fwd x sLSTM/Mamba) pass their HARD beat-init
   legs deterministically; the vs-BLSTM numbers are recorded in `RESULTS.md`.
3. The causal streaming gate (S5.6) passes bit-for-bit with the latency bound derived,
   measured, and pinned (S5.5); fast-vs-exact parity (S4.3) pinned with zero flips.
4. The exact tree's committed golden suite is byte-green; the LID mechanical wiring (S8.3)
   is pinned; the battery (S8.6) is run with results recorded.
5. Docs refreshed: CLAUDE.md module rows (cells, fast twins, StreamCausal, keys), README,
   `RESULTS.md` (comparison + latency + bench tables); port-only design notes in module
   docs + RESULTS, never IMPROVEMENTS.

## R -- Standing rules

- **R1 STOP-and-adjudicate**: any decision flip (argmax, boundary count/type) or nonzero
  boundary max_dt in a parity/streaming gate STOPS the phase for adjudication -- never
  silently widen a tolerance (the phase-7/8 rule verbatim).
- **R2 exact-tree byte-freeze**: outside the S1.4 sanctioned classes, no `nn/` behavior
  change; the golden suite is the arbiter.
- **R3 license hygiene**: no corpus path, filename, or content-derived value committed;
  corpus tests gated `requires_corpus`/`corpus_root_or_skip`; synthetic fixtures only in CI.
- **R4 measure-then-pin**: every tolerance, budget, and latency number is measured first,
  pinned second, with headroom stated.
- **R5 honest comparison**: subset vs-BLSTM numbers are reported as-is with the thinness
  caveat; no architecture-superiority claim is made from subset scale.

## Deferred (named, not lost)

- Transformer encoder + CfC behind `CellLayer` (phase 10 candidates; the whole-sequence
  `Layer` trait fits an attention block).
- Bidirectional f32 fast twins for the new cells; a forward-only LSTM fast twin.
- Provisional-begin streaming emission (IMPROVEMENTS `[phase8 -> phase9]` -- stays open).
- sLSTM heads/block-diagonal recurrence, mLSTM.
- Mamba recompute-on-backward; per-net Mamba hyperparam overrides.
- QPSO search over new-cell hyperparameters.
