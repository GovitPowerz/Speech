# Phase 3: Training (network-level BPTT + iRPROP-) - Design

Date: 2026-07-06. Branch: `feature/phase-3-training` (off main 999d58a, contains Phases 0a-2b).
Survey basis: scratchpad `p3_explore.md` (all legacy line numbers below re-verified there).
Parity target: bit-exact vs the strict-IEEE oracle harness (`tools/oracle_harness/`), which
already links every TU this phase needs (`BLSTMNeuralNetwork.cpp LSTMLayer.cpp NeuronLayer.cpp
CostLaw.cpp Rprop.cpp`, build.sh:95-96). Legacy quirks are load-bearing: reproduce exactly,
log each in IMPROVEMENTS.md, never fix.

## S1. Scope

IN: the complete network-level training chain -
`CostLaw::computeDeltas`/`computeUnitaryDeltas`/per-law `deriv()` (cost.rs);
`NeuronLayer::feedBackward` and `LSTMLayer::feedBackward` (nn/layers.rs);
`Network<L>::feedBackward` reverse-layer loop incl. the SubSample interaction
(nn/network.rs); `BlstmNetwork` `feed_backward` + `feedBackwardDouble` fan-out, the
backward halves of the windowed drivers, `get_weights_derivatives` (Nx2 flat assembly),
`reset_weights_derivatives` (real body), `update_weights` (normalize + trainer)
(nn/blstm.rs); `Rprop` iRPROP- (nn/train.rs); a network-level central-difference grad
check; the engine-semantics backward oracle in nn_reference.py.

OUT (Phase 4): corpus epoch loop + cross-file deriv reduction (CorpusProcessor::run/train,
the OpenMP-critical sum - the Rust reduction MUST be deterministic when it lands; noted in
S11.7), BagOfProcessors::saveAndUpdate + best-weight save lifecycle, corpus-level gradCheck
mode (CorpusProcessor:237-340 - the result-column indexing 4/14/.size()-1/-2 is documented
there for Phase 4), CLI Train/GradCheck modes, PyO3 seam, Python optimizer zoo
(optimizers.py SMORMS3 stays a stub), scoring.py CheckGrad.

DEAD - do not port: LSTMLayer.cpp:423-516 + :734-913 (commented alternate
feedForwardReverse/feedBackwardReverse bodies), all `_MaxSaturation` logic, Rprop.cpp:60-76
(.mat dump) + `_Count`, SRN/CWRNN backward (link-only), ConvolutionalLayer backward
(broken-as-committed), the tthread CalcThread path.

## S2. Oracle strategy (four tiers, strongest first)

1. **iRPROP-: the REAL compiled `Rprop::updateWeights` is the golden directly.**
   Rprop.cpp:9-77 is pure scalar (no libm, no GEMM, the :25 cwiseProduct is element-wise) -
   bit-portable; goldens assert STRICT BITS on every platform, no canary gate. Strongest
   oracle of the phase; no transcription.
2. **CostLaw backward: probe the REAL compiled `computeDeltas`/`computeUnitaryDeltas`/
   `deriv()`.** Polynomial laws (linear/square/cubic) bit-exact everywhere; LogLaw/Sqrt
   laws are libm-dependent - canary-gated per the binding cross-libm rule.
3. **Layer/container backward: harness ascending-loop reimpl is the golden source +
   NN_TOL probes of the real classes.** The backward contains Eigen GEMMs at NN-wide k
   (`_FeedbackWeights*test` LSTMLayer.cpp:703, `_InputWeights*testM` :707, NeuronLayer
   `deltas*_Weights.transpose()` :202) - same blocked-vs-ascending divergence as the
   forward, and doubly non-local because the forward already seeds the caches. Identical
   Phase 2 pattern: goldens dumped from the reimpl; NN_TOL records real-class deltas
   (synthetic small shapes expected 0 ULP; real-net sites recorded as Phase 4
   calibration).
4. **Python nn_reference.py backward oracle as independent arbiter**: scalar-loop engine
   semantics, plain-loop sums (numpy pairwise hazard), cross-checked bit-for-bit against
   the harness dumps AND the Rust layers on synthetic shapes (same triangulation as the
   Phase 2 forward).

NaN policy: any NaN in a fixture (e.g. count-0 normalization, LogLaw at 0) compares
bit-exact on the oracle env, by `is_nan` off it (arch-defined quiet-NaN sign; CLAUDE.md
rule extended in P2b).

## S3. The gradient object: Nx2, layout == weight-packer layout

`getWeightsDerivatives()` returns Nx2: col0 = SUMMED derivative, col1 =
`_NbOfSeqFedBackward` (frame count) replicated per element (LSTMLayer.cpp:251-295,
NeuronLayer.cpp:101-114, NeuralNetwork.hpp:98-111 vertical hcat).
`BLSTMNeuralNetwork::getWeightsDerivatives` (:255-276) hcats fwd|bwd|output then appends
the mean/std tail as [deriv 0 | count 1] (stats never move). If OutputSubSamplingRatio>1
the fwd/bwd LSTM col0 is multiplied by it (:266-269). The flat ordering (gate blocks,
12-row peephole bundle rows 0-11 with the SAME index map as the forward - survey S1.5 -
4 biases, tail) matches the Phase 0a flat weight packer element-for-element; a
pack/unpack/repack property test pins the agreement.

Per-element counts DIFFER by region: mean/std tail count=1; LSTM counts accumulate
frames x sweeps x per-window coverings. Normalization is `col0 cwiseQuotient col1` -
element-wise, AT UPDATE TIME ONLY (BLSTMNeuralNetwork.cpp:303-310), never inside the
backward. A scalar /nframes is WRONG (risk R1).

Per-file lifecycle: `reset_weights_derivatives` zeroes the accumulators + count (called
at segmenter entry in legacy, e.g. BLSTMSpectralSegmenter.cpp:473); derivs then
accumulate across every window/sweep of the file until harvested.

## S4. CostLaw backward seam (cost.rs)

- Each of the 7 ported VAD laws gains `deriv()` EXACTLY per CostLaw.h:6-213 (Linear: _A;
  Square: _B+2A y; BelowThreshCubic: 2B y+3A y^2; AboveThreshCubic with the 1-yVal flip;
  Log: _A/y with its clamps; the two Sqrt laws). costPonderation is baked into _A/_B at
  construction - verify the ponderation plumbing already ported forward covers this.
- `computeUnitaryDeltas` (CostLaw.cpp:278-345, scalar VAD path): select law deriv, apply
  BackPropWER scaling (`*10*(1-target)` speech / `*10*target` other), then multiply by
  `output*(1-output)` - the LOGISTIC output-activation derivative folded in (the scalar
  output head uses Logistic, NeuronLayer.cpp:144).
- `computeDeltas` (CostLaw.cpp:278-445): scalar path per row when targetSeq.cols()==1;
  multiclass path (:403-418) is the SOFTMAX+CE FUSION: `target<0 -> 0` (ignore mask),
  `target>0.5 -> output-1.0`, else `output`. BackPropWER (:372-401) and
  ClassesPonderations (:410-417, whole row scaled by target-class ponderation) branches
  scale it. NO Jacobian anywhere else - see S5 double-count trap.

## S5. Layer backward (nn/layers.rs)

- **LSTMLayer::feedBackward (LSTMLayer.cpp:518-723)**: reverse-time loop, gate-derivative
  order O -> state -> C -> F -> I exactly as survey S1.4 items 1-10 (epsilonCells,
  deltasOutputGate with prior-step peephole cross-terms rows 5/8/11, epsilonForget from
  NEXT-step forget gate, epsilonState with cells_peep rows 0/1/2, deltasCells,
  deltasForgetGate with the row==0 no-cellstate branch, deltasInputGate, stacked-testM
  deltasPreviousLayer, deltasFeedback). Derivatives recomputed FROM post-activation
  caches (`_Gates`/`_CellStates`/`_CellsIn` - the Rust fields already exist and suffice;
  no forward change). End of call: multiply all 4 deriv blocks by invSubSamplingRatio if
  >1 (:710-715), ACCUMULATE into the four deriv members, `_NbOfSeqFedBackward += linesNb`
  (:716-720). All products via the measured ascending-loop contract (`matmul_seq`).
- **Width tolerance in backward** (:534-543): input cols > _InputSize -> leftCols
  truncate; cols < _InputSize -> ZERO-PAD input rows. Gradient shapes stay layer-native;
  the P2b signal driver's dead weight rows accumulate exactly 0 (S11 test).
- **Reverse layer**: `feedBackwardReverse` (:725-732) = reverse input/output/deltas, call
  the forward-order feedBackward, reverse returned deltas. The caches are STORED
  time-reversed after `feedForwardReverse` and consumed as-is - do NOT un-reverse
  (layers.rs:331-335 doc-comments already pin this).
- **NeuronLayer::feedBackward (NeuronLayer.cpp:151-207)**: `weightsDerivatives +=
  input.col(jj)*deltas.row(jj)` per row, bias += colwise sum (:178-183); lastLayer
  deltas_out = `deltas*W'` with NO activation derivative (fusion is in CostLaw - adding
  a softmax Jacobian here double-counts, risk R3); hidden deltas_out multiplied by
  asinh-deriv ON THE LAYER INPUT (:204). No retained cache; input passed in from
  `layers_output`.

## S6. Container + BLSTM backward (nn/network.rs, nn/blstm.rs)

- `Network<L>::feedBackward` mirrors NeuralNetwork.hpp:251-300: reverse-order layer loop
  feeding each layer its retained `layers_output` input. The SubSample interaction
  (decimated forward rows vs backward delta rows) is NOT summarized in the survey - the
  implementer transcribes :251-300 exactly and documents what the legacy does with
  sub-sampled deltas (crux verification point; do not invent semantics).
- `feedBackwardDouble` (NeuralNetwork.hpp:353-362): hcat `_OutputForward|_OutputBackward`
  as MLP input, run the MLP backward, return deltas split left/right to the fwd/rev
  stacks (BLSTMNeuralNetwork.cpp:439-460). `_BackPropOutputNetworkOnly` short-circuits
  the LSTM backward.
- Per-window worker (BLSTMNeuralNetwork.cpp:776-830): forward, then (if backprop active
  and targets present) the `_TargetEnforcementStep<0` interior-row rewrite to -0.5
  applied to target AND (for cost) OUTPUT - caller-visible outputSeq mutation (:817-822,
  risk R7) - then feedBackward, then cost/_NbOfClassif accumulation.
- Windowed drivers backprop INSIDE EVERY WINDOW (Truncate :538, OverLap :656, MLPOverLap
  :703). TwoSweeps: both sweeps backprop; derivs and counts both double (forward output
  averaged /2, derivs NOT halved - :548-590). OverLap: overlap-region frames backprop
  once per covering window; forward output divided by coverage, derivs NOT - the col1
  count carries the compensation (:592-681). The NaN coverage rows are forward-only
  artifacts (fresh targetSeqShort per window :649).
- Input normalization (types 1/-1/-2) applied IN PLACE before dispatch exactly as the
  forward already does (:715-751; -2 inside the non-windowed overload :777-786).
- `update_weights(cost)`: normalize Nx2 (S3) then call the trainer
  (BLSTMNeuralNetwork.cpp:303-310). The trainer is a LONG-LIVED per-network member.

## S7. iRPROP- (nn/train.rs)

`Rprop` struct replacing the stub. Trainer.h: single virtual
`updateWeights(derivs, weights, cost)` - in Rust a plain struct + method; no trait until
a second trainer exists (YAGNI). Ctor defaults (Rprop.cpp:6): eta 0.5/1.2, delta clamps
1e-9/0.2, init delta from config `_BackPropagationRpropInit` (default 1e-2,
BLSTMNeuralNetwork.cpp:151). State: `deltas`, `prev_derivs`, `delta_weights`,
`prev_cost` - persists across calls for the network's lifetime.

Exact per-element semantics (Rprop.cpp:9-77), INVERTED SIGN CONVENTION throughout
(deriv>0 -> dw = -delta; deriv<0 -> dw = +delta; deriv==0 -> dw = 0; applied
`weights += dw`):
- First call (empty state): init deltas=const(init_delta), prev_derivs=derivs,
  delta_weights=0; apply dw per sign.
- Subsequent: `derivTimesPrev = derivs .* prev_derivs`; then per element:
  - `>0`: delta *= 1.2 clamped to max; recompute dw; apply.
  - `<0`: delta *= 0.5 clamped to min; COST-GATED BACKTRACK: only if `prev_cost < cost`
    (cost went UP) `weights -= delta_weights` element-wise; then `prev_derivs = 0` for
    that element (forces next step into the ==0 branch); NO dw applied this element.
  - `==0`: recompute dw; apply.
- `prev_cost = cost` at the end. `delta_weights` records the last applied dw per element.
Copy verbatim - the inverted sign and the cost gate are CLAUDE.md-listed hazards.

## S8. Network-level grad check

Phase 3 lands the central-difference METHOD at network granularity (the corpus-level
mode with its result-column indexing is Phase 4): for a fixture (real or synthetic net +
input + targets), perturb each selected flat weight +eps / -2eps (net -eps) with full
state restore between runs, cost from one `feed_forward_backward`, numerical grad =
`(costPlus-costMinus)/(2 eps)`, compared against analytic `col0/col1`
(CorpusProcessor:237-340 documents the legacy shape: same central difference, eps from
`Neural_Networks_Gradient_Check_Epsilon`, relative error floored at 1e-24). This is a
validation harness, not a byte-golden: assert the analytic/numerical relative error
under a documented bound on a spot-checked weight subset (full sweep is O(N) forwards).
It is the built-in non-vacuity layer for the analytic chain.

## S9. Python backward oracle (nn_reference.py)

`lstm_backward_oracle`, `dense_backward_oracle`, `blstm_backward_oracle` with ENGINE
semantics (sigmoid(0.1z) gates, asinh, fusion in the cost seam). `BLSTM_Backward.m` is
structure-reference ONLY - it pairs with the divergent MATLAB forward (sigmoid(x), no
softmax, internal /ncols normalization) and must NOT be ported (P2b T11 verdict). The
net-level forward helpers already return (y, gates, cells); thread them through. Plain
ascending loops only. Cross-checks: vs harness dumps and vs the Rust layers, bit-for-bit
on synthetic shapes, canary-gated where transcendental.

## S10. Fixtures & harness work

New harness stage(s): backward reimpl (ascending-loop) mirroring the P2 reimpl family,
NN_TOL backward probes (synthetic + real-net), REAL CostLaw computeDeltas dumps, REAL
Rprop::updateWeights trajectory dumps (multi-step, see S11.1). The grad check itself is
bound-based per S8 (no byte-golden); it needs no fixtures beyond the nets and inputs
already dumped. Fixture pipeline unchanged: build.sh local-only, extractor with hash guard,
MEASURED manifest values, regenerate twice byte-identical. Real-net fixtures reuse
NNweights_config1.bin + 1_worker_1.config + the e2e input.

## S11. Testing requirements (non-vacuity is PROVEN, not assumed - P2b lesson)

1. **iRPROP- branch coverage**: the Rprop golden trajectory must exercise ALL branches -
   eta+ growth, eta- shrink, both clamps, the cost-gated backtrack FIRING (cost rise)
   and NOT firing (cost fall), the post-backtrack prev_derivs==0 path, and first-call
   init - asserted structurally (which branch fired per step is derivable from the
   trajectory; assert it, don't hope).
2. **Gradient non-triviality**: real-net gradient fixtures must be non-zero and
   non-saturated (assert min/max/nonzero-count structurally).
3. **Double-count trap**: a test proving the output layer applies NO activation
   derivative (fusion only in CostLaw) - e.g. dense-layer backward golden with deltas
   whose fused form is distinguishable.
4. **Count vector**: TwoSweeps and OverLap fixtures assert col1 per region (frames x
   sweeps, per-window coverage counts, mean/std tail == 1) by hand-computed expectation.
5. **Width tolerance**: signal-style 1-col input backward - dead weight rows accumulate
   exactly 0.0 (bit), gradient shapes layer-native.
6. **Layout agreement**: pack/unpack/repack property test between the Nx2 flat deriv
   layout and the P0a weight packer ordering.
7. **Determinism note**: all Phase 3 accumulation is single-file/single-network -
   sequential by construction. The cross-file reduction hazard (OpenMP critical,
   order-nondeterministic last-ULP) is documented for Phase 4 in the handoff.
8. **Statefulness**: trainer state across >=3 update calls golden-pinned (the P2b
   two-files lesson: pin lifecycle via discriminating observables, honestly scoped).
9. Comparator discipline: strict-bits for Rprop everywhere; canary-gated assert_oracle_eq
   for everything downstream of libm; structural values exact. NaN per the arch-sign rule.

## S12. IMPROVEMENTS.md candidates (log during implementation)

Inverted iRPROP- sign convention; cost-gated backtrack + prev_derivs zeroing; TwoSweeps
deriv double-count (compensated only via col1); OverLap deriv-vs-output normalization
asymmetry; the enforcement-step OUTPUT mutation (:817-822); logistic-deriv fold in
computeUnitaryDeltas; BackPropWER 10x asymmetric scaling; the count-1 mean/std tail;
whatever the SubSample backward transcription reveals (S6).

## S13. Risks

R1 per-element deriv/count normalization (the layout trap). R2 stateful cost-gated
backtracking across epochs. R3 softmax+CE fusion double-count. R4
ponderations/BackPropWER entering cost AND gradient. R5 grad-check cost-column indexing
(deferred with Phase 4's corpus mode). R6 cross-file reduction determinism (Phase 4,
documented). R7 enforcement-step output mutation. R8 masked-target width (cost operand
fixed in P2 to target.ncols - exercise masked multiclass in a golden). R9 reverse-layer
time-reversed caches (do not un-reverse).
