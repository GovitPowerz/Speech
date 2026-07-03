# Phase 2 - NN Forward Design Spec

Date: 2026-07-02
Status: approved (design), pending implementation plan
Branch: `feature/phase-2-nn-forward` (off `main`, which contains merged Phase 1)

## 1. Goal and scope

Port the NN forward pass: `nn/activations.rs` (the overridden activation set), `nn/layers.rs` (LSTMLayer forward incl. reverse + NeuronLayer dense forward), `nn/network.rs` (the `NeuralNetwork` stack container: sub-sample decimation, chained weight (de)serialization, forward drivers), and `nn/blstm.rs` (the `BLSTMNeuralNetwork` wrapper: construction from config, flat-weight seam, input-normalization modes, processing-mode dispatch incl. the windowed drivers, the top-level scoring forward, per-file reset semantics). `nn/train.rs` remains a stub (Phase 3: BPTT + iRPROP-).

**Acceptance:** the real trained SAD network (`tests/reference_data/phase0/1_worker_1.config`: LSTMNeuronNb [23,24,24], sub [4,1], OutputNeuronNb [48,12,1], InputNormalizationType -1, Algo 3; weights `NNweights_config1.bin`, 33,671 elements, byte-validated in Phase 0a) runs forward on real features produced by the Phase 1 pipeline under that config's DSP params and matches the oracle bit-for-bit (canary-gated hybrid comparator elsewhere): final posteriors, `_OutputForward`, `_OutputBackward`. Plus synthetic per-unit goldens covering every structural branch, hand-computed anchors, and an independent numpy oracle.

Phase 2b (separate): segmenter wiring (`tasks/sad.rs`, run_pipeline lift, result-vec sizing, `results2segmentation` glue). Phase 3: training. Phase 4: PyO3 + end-to-end.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Layer scope | LSTMLayer + NeuronLayer only. Layer choice is COMPILE-TIME in the legacy (`BLSTMNeuralNetwork<LayerType>`); every live constructor is `<LSTMLayer>` (BLSTMSpectralSegmenter.cpp:20, BLSTMSignalSegmenter.cpp:19/26, BLSTMSpectralLID.cpp:13, TwinBLSTMSpectralLID.cpp:13/21). SRN/CWRNN exist only as explicit instantiations (BLSTMNeuralNetwork.cpp:960-962) + commented-out uses - dead code, stay stubs. CNN is live-but-broken (UB with the real lid.config: empty `_Layers` indexed at ConvolutionalNeuralNetwork.cpp:265/228; weights never serialized; backward commented out) - stays a stub, IMPROVEMENTS-documented. |
| 2 | Wiring scope | Pure NN forward. The windowed/overlap/TwoSweeps drivers ARE ported (they live inside BLSTMNeuralNetwork.cpp), the Segmenter-level wiring is not. |
| 3 | Oracle | Extend `tools/oracle_harness/`: link the real NN TUs (BLSTMNeuralNetwork.cpp, LSTMLayer.cpp, NeuronLayer.cpp, SRNLayer.cpp, CWRNNLayer.cpp [link-required by the explicit instantiations], CostLaw.cpp, Rprop.cpp - include graph verified free of Segmenter/Segmentation drag; the inert iof shim compiles all NN uses). Goldens come from a harness-local, branch-for-branch REIMPLEMENTATION of the forward path with ascending-loop matrix products, because the layer-0 GEMMs seed the recurrence: the Phase 1 DCT-substitution problem recurs non-locally (LSTMLayer.cpp:314/316/318/351, NeuronLayer.cpp:130/132/134) and no post-hoc substitution is possible. The REAL compiled classes stay linked and are probed at every fixture regeneration: (a) bitwise GEMM_CHECK-style probes on each product kind recorded in the manifest, (b) a full-forward tolerance comparison (reimpl vs real class) asserting agreement within a measured ULP/absolute envelope. Rust matches the reimplementation bit-for-bit. |
| 4 | numpy oracle | `src/python/speech/nn_reference.py` implements the ENGINE semantics (0.1-prescaled gates with expLimit saturation, asinh cell activations, the 12-row peephole map, exact op order, plain-loop accumulation per the Phase 1 numpy lesson). `BLSTM_Forward.m` is NOT ported: it uses `sigmoid(x)` gates (no 0.1 pre-scale, no clamping; .m lines 96/106/123) and no softmax - it cannot match the engine and is documented as a semantic cross-reference only. |
| 5 | Weight seam | Rust `nn` consumes the flat vector via legacy-style chained `set_weights` (each unit eats its head, returns the tail) - NOT via `config.rs`'s packer. The flat order (Forward net | Backward net | Output net | mean | std, tail length = LSTMNeuronNb[0]) agrees with Phase 0a `config.rs` (verified: config.rs:259-278 pack / :367-387 unpack vs BLSTMNeuralNetwork.cpp:209-225). `config.rs` does not model the `_IsMLP` layout (output-net-only + tail of outputNeuronNb[0]) - Python-seam gap, IMPROVEMENTS-noted, not a Phase 2 blocker. |
| 6 | Comparator | Phase 1's canary-gated hybrid comparator infrastructure is reused as-is (exp/asinh are libm; goldens are oracle-environment products). |

## 3. Activations (exact) - `nn/activations.rs`

Legacy: `ActivationFunctions.h`, `Log.hpp:193-195`.

- `exp_limit = f64::MAX.ln()` (~709.782712893384). Computed once, same expression.
- `gates_fn(x)` (GatesFunction, h:229-238): `if 0.1*x < exp_limit { if 0.1*x > -exp_limit { 1.0/(1.0+exp(-0.1*x)) } else { 0.0 } } else { 1.0 }`. Boundary EXCLUSIVE: at exactly +-exp_limit returns saturated 1/0. The 0.1 pre-scale is intentional (deriv `0.1*y*(1-y)` at h:241 confirms).
- `logistic_fn(x)` (h:40-49): strict guards `x >= exp_limit -> 1`, `x <= -exp_limit -> 0`, else `1/(1+exp(-x))` (note: Logistic saturates INCLUSIVE at the boundary, GatesFunction EXCLUSIVE - port both literally).
- `maxmin2_fn(x) = asinh(x)` (h:158-160) and `identity_fn(x) = asinh(x)` (h:206-208): both asinh; keep distinct names (Phase 3 derivatives differ only in name; cell-input g uses Maxmin2, cell-output squash uses Identity, hidden dense layers use Maxmin2).
- `asinh_fn(x) = asinh(x)` (the `Asinh` struct used by input normalization).
- Softmax is NOT a struct - it is inline in NeuronLayer (section 5). `Logarithm` (fmath) is already ported (Phase 1 `fmath_log`); `InvLogistic`/`ReLU` are unused by the LSTM+dense forward - omit.

## 4. LSTMLayer forward (exact) - `nn/layers.rs`

Legacy: `LSTMLayer.h` (49 lines), `LSTMLayer.cpp:6-159` (ctor), `:162-249` (weights), `:312-421` (forward). BPTT at :518-914 is Phase 3.

### 4.1 Layout

- `_InputWeights` I x 4O, `_FeedbackWeights` O x 4O, `_PeepWeight` 12 x O, `_Biaises` 1 x 4O (the header size comments at h:33-36 are stale lies; trust the resizes at cpp:19-22).
- Gate column blocks in all 4O-wide structures: `[i | f | o | g]` = cols [0,O), [O,2O), [2O,3O), [3O,4O).
- Forward state kept for Phase 3: `_Gates` (T x 4O, post-activation in place), `_CellStates`, `_CellsIn` (T x O). `feedForwardReverse` leaves them in reversed-time order.
- Config keys: `<prep>_MaxSaturation` (int, default -1, DEAD - all saturation logic commented out; read the key, ignore it), `<prep>_IsCellsPeepholesActive` / `_IsGatesPeepholesActive` / `_IsGatesRecurrentPeepholesActive` (bool, default true; the legacy member is misspelled `Reccurent`, the KEY is `Recurrent`).

### 4.2 Flat weight block (per layer; `set_weights` eats the head, returns the tail)

`nb = 4*I*O + 4*O*O + 12*O + 4*O` (cpp:205-207). Order (cpp:162-203): `_InputWeights` COLUMN-major (col j outer, row i inner), `_FeedbackWeights` column-major, `_PeepWeight` column-major (12 peep weights contiguous per output neuron), `_Biaises`. `getWeights` is the exact mirror. Values pass verbatim (no adim division - that lives in the config-text ctor branch, cpp:126-130, which Phase 2 does not take: production uses `weightsSetExternally=true` + setWeights).

### 4.3 Recurrence (cpp:312-413) - operation ORDER is the contract

Input projection, whole sequence (313-319): width-tolerant - `cols > I` -> `leftCols(I)*W`; `cols < I` -> `X*W.topRows(cols)`; else `X*W`; then `.rowwise() + biases`.

t = 0 (325-348): activate i,f (gates_fn) and g (maxmin2/asinh); `c_0 = i_0.*g_0` (NO forget term - zero initial state by omission); o pre-activation extras in order: `+ c_0.*P[2]` (Cells flag), `+ i_0.*P[9]`, `+ f_0.*P[10]` (Gates flag; two separate adds); activate o; `cells_in_0 = asinh(c_0)` (identity_fn); `y_0 = o_0.*cells_in_0`. No row-11 term at t=0.

t >= 1 (350-412), exact order:
1. `gates.row(t) += y_{t-1} * FeedbackWeights` (all four blocks at once, g included).
2. i/f extras: cells `i += c_{t-1}.*P[0]`, `f += c_{t-1}.*P[1]`; gates-recurrent `i += i_{t-1}.*P[3]`, `f += f_{t-1}.*P[7]`; gates - each ONE combined expression: `i += (f_{t-1}.*P[4] + o_{t-1}.*P[5])`, `f += (i_{t-1}.*P[6] + o_{t-1}.*P[8])`. All `t-1` gate reads are POST-activation values (activated in place).
3. Activate i,f (gates_fn over [0,2O)).
4. Activate g (asinh over [3O,4O)); g gets recurrent feedback but never peepholes.
5. `c_t = i_t.*g_t + c_{t-1}.*f_t` - single combined expression (cpp:387).
6. o extras in order: `+ c_t.*P[2]` (CURRENT); `+ i_t.*P[9]`; `+ f_t.*P[10]` (two separate adds); `+ o_{t-1}.*P[11]` (gates-recurrent).
7. Activate o; `cells_in_t = asinh(c_t)`; `y_t = o_t.*cells_in_t`.

12-row peephole map (row -> source -> target, flag): 0: c_{t-1}->i (Cells); 1: c_{t-1}->f (Cells); 2: c_t->o (Cells, all t); 3: i_{t-1}->i (GatesRec); 4: f_{t-1}->i (Gates); 5: o_{t-1}->i (Gates); 6: i_{t-1}->f (Gates); 7: f_{t-1}->f (GatesRec); 8: o_{t-1}->f (Gates); 9: i_t->o (Gates, all t); 10: f_t->o (Gates, all t); 11: o_{t-1}->o (GatesRec).

`feedForwardReverse` (415-421): reverse rows of the input, run forward, reverse rows of the output. Do NOT port from the commented hand-unrolled variant (423-516) - its peephole grouping differs.

`lastLayer` parameter: UNUSED in the LSTM forward (both directions) - keep in the signature for the Layer contract.

## 5. NeuronLayer + container (exact) - `nn/layers.rs` + `nn/network.rs`

Legacy: `NeuronLayer.{h,cpp}`, `NeuralNetwork.hpp` (365 lines, header-only).

### 5.1 NeuronLayer

- `_Weights` I x O, `_Biaises` 1 x O. Flat block (cpp:69-99): weights column-major (per-neuron fan-in blocks contiguous), then all biases; `nb = O*(I+1)`.
- Forward (cpp:127-149): width-tolerant projection (same three branches as LSTM); then bias `.rowwise()` BEFORE activation; `lastLayer && O > 1` -> UNSTABILIZED softmax: `exp(a+b)`, per-row sum (sequential left-to-right), per-column quotient - no max subtraction; `lastLayer && O == 1` -> logistic_fn; `!lastLayer` -> asinh (maxmin2_fn).
- No reverse variant (never run reversed).
- Config-text weight branch (cpp:40-50) divides weights AND bias by `_InputSize` at read time (the dense-layer analog of the adim seam: flat-vector values are the divided ones); the missing-key random branch (cpp:28-39) uses `_RandomVector` with `posIni = I+O`, row-major walk, `2r-1`, `/= I`, and the whole-matrix fill sits inside the per-neuron loop with a shadowed index (idempotent when ALL keys missing; pathological when mixed - reproduce all-missing only, document mixed as out of contract). Phase 2 production path is `weightsSetExternally` + set_weights; the config-text branch is ported for completeness of the ctor but exercised only by unit tests.

### 5.2 NeuralNetwork container

- Ctor (hpp:25-37): layer jj input = `NeuronNb[jj]*SubSampling[jj]`, output = `NeuronNb[jj+1]`; cumulative `_SubSamplingRatios`; `getSubSamplingRatio()` = last cumulative product. The copy ctor (hpp:39-51) is ill-formed C++ that compiles only because it is never instantiated - do NOT port it.
- `SubSample(R, X)` (hpp:125-134): T x C -> floor(T/R) x C*R; trailing `T mod R` frames DROPPED; source frame `jj*R+kk` lands at cols [kk*C,(kk+1)*C) (frame-contiguous temporal stacking). `Repeat` (hpp:147-156): each row duplicated R times consecutively (consumer-side upsampling; port for Phase 2b callers). `InvSubSample` is Phase 3.
- Chained weights (hpp:69-96): layer-0-first consume-head/return-tail; getWeights mirrors.
- `feedForward` (hpp:158-198): empty input (0 rows) = silent no-op. Single-layer: optional subsample then layer forward with lastLayer=true. Multi-layer ascending: layer 0 from (optionally subsampled) input into `_LayersOutput[0]` (resized T_sub x NeuronNb[1], lastLayer=false); middle layers likewise; final layer writes the CALLER-ALLOCATED `outputSeq` (never resized by the container) with lastLayer=true. `_LayersOutput` caches are retained (Phase 3 needs them). Row counts cascade per-layer floors - never one floor of the product.
- `feedForwardReverse` (hpp:200-240): identical driver dispatching each layer's reverse; ascending layer order (reversal is inside the layer).
- `feedForwardDouble(A, B, out)` (hpp:242-249): `Input << A, B` (hcat to exactly NeuronNb[0] cols; forward half LEFT, backward half RIGHT - confirmed by the Phase 3 delta split at BLSTMNeuralNetwork.cpp:453-457) then normal forward.

## 6. BLSTM wrapper (exact) - `nn/blstm.rs`

Legacy: `BLSTMNeuralNetwork.h` (211 lines; h:35-145 is a stale commented ctor - trust the cpp), `BLSTMNeuralNetwork.cpp` (962 lines).

### 6.1 Construction (cpp:26-153)

Config keys (prefix from caller): `_LSTMNeuronNb` (list, >= 2 else err), `_LSTMSubSampling` (len = LSTMNeuronNb-1), `_OutputNeuronNb` (>= 2), `_OutputSubSampling` (len-1), `_InputNormalizationType` (short), `_TwoSweeps` (bool), `_BackPropagationActivated` (false), `_BackPropOutputNetworkOnly` (false), CostLaw keys (Phase 0b-i), `_TargetEnforcementStep` (0), `_weightsFile` ("" -> optional `.bin` load via the Phase 0a codec; size < needed -> err, > needed -> warn + set anyway), `_BackPropagationRpropInit` (1e-2, Phase 3). Constraint: non-MLP requires `outputNeuronNb[0] == 2*LSTMNeuronNb.back()`. `_IsMLP = (LSTMNeuronNb[0] == 0)` - in-band signal. Sub-network prefixes: `{prefix}_Forward`, `{prefix}_Backward`, `{prefix}_Output`. Mean/std tail init Zero/Ones of the input size.

### 6.2 Flat seam (cpp:209-253)

set_weights: Forward net -> Backward net -> Output net -> mean(inputSize) -> std(inputSize), inputSize = LSTMNeuronNb[0] (MLP: output net only, tail = outputNeuronNb[0]). get_weights mirrors. `getNbOfWeights` = sum + 2*inputSize. Driver asserts 33,671 for the real config.

### 6.3 Input normalization (cpp:711-751, 777-781)

Windowed entry `feedForwardBackward(inputSeq mutable, window_size, window_shift, outputSeq, targetSeq)`: zeroes `_Cost`/`_NbOfClassif` (712-713), then by type:
- Type 1 (715-728): per column jj < min(cols, tail len): `(x - mean_j)/max(1e-12, std_j)` IN PLACE; columns beyond untouched; then `analyseInputSeq` (InputStatistics, Phase 1) on the normalized input - ONLY this branch.
- Type -1 (729-751): whole-sequence self-normalization IN PLACE: mean = colSum/rows; center row-by-row (explicit row loop); std = `sqrt((colSum(x^2) + 1e-32)/rows)` - the 1e-32 added to the SUM before dividing; then per row `asinh(x/std)`. Keep the row-loop op order.
- Type -2 (777-781): same formula PER WINDOW into a copy (input not mutated), whole-matrix expressions with replicate - match as written.
- Other values: no normalization.

### 6.4 Core forward (cpp:419-437, 462-469)

`feedForward(input, output)`: outputLength = rows divided SEQUENTIALLY by each LSTM ratio; resize `_OutputForward`/`_OutputBackward` (outputLength x lstmOut); leftCols-trim gated on `ratios[0] > 1 && inputWider`; forward net forward + backward net reverse; `output_net.feedForwardDouble(_OutputForward, _OutputBackward, outputSeq)` (forward LEFT, backward RIGHT). `feedForwardMLP` analogous.

### 6.5 Processing-mode dispatch + windowed drivers (cpp:364-367, 753-773, 488-709)

`setProcessingType(truncate, overlap)` sets `_TruncatesSequence`/`_Overlaps`. Dispatch: MLP -> `MLPOverLap` iff truncate && overlap else plain MLP (window args ignored - quirk); BLSTM -> OverLap / Truncate / plain.
- Plain FFB (776-830): type-2 normalization branch; cost accumulation; `_TargetEnforcementStep < 0` QUIRK: the caller-visible outputSeq INTERIOR rows [1, rows-1) are overwritten with -0.5 before computeCost (821); `_NbOfClassif += rows`.
- TruncateSweep (488-546): non-overlapping windows `jj += window_size`; per-window sequential-floor lengths; trailing chunk shorter than the subsampling ratio silently dropped (output rows keep prior contents); stitched `_OutputForward`/`_OutputBackward` written back at the end.
- Truncate + TwoSweeps (548-590): `shiftShort = (window_size/2)/ratio` (division order), edge-replication padding, two sweeps offset by half a window, `output = (sweep1 + sweep2)/2`, and `_OutputForward`/`_OutputBackward` become the HCAT of both sweeps (double width) - quirk.
- OverLap (592-681): `jj += window_shift`; begin/end SNAPPED to the subsampling grid (`begin = (begin/ratio)*ratio` at 620, `end = ((end+1)/ratio)*ratio - 1` at 626); accumulate sums + counts; final per-column quotient; rows never covered divide 0/0 -> NaN (reproduce, no guard); `_NbOfClassif` double-counts overlapped frames by design.
- MLPOverLap (683-709): IGNORES the passed window_size (`= ratio/2`); only exact full windows produce one scalar each at row jj/window_shift; edges skipped; `_OutputForward`/`_OutputBackward` emptied.

### 6.6 Top-level scoring forward (cpp:843-929)

`feedForward(input, window_size, window_shift, targetIndex, targetModifier) -> MatrixXd`: rows < subsampling ratio -> returns Zero(1, outputSize). Target construction (targetIndex >= 0): soft target `0.1*targetModifier` if the cost law is modified else 0.0; multiclass unknown class (`targetIndex >= outputSize`) -> index 0; enforcement counter: rows enforced every `_TargetEnforcementStep+1` rows (row 0 always), others = -0.5 ignore marker. Binary expansion (922-926): outputSize 1 && targetIndex >= 0 -> 2 columns `[1-p, p]`. Input is a mutable ref throughout (normalization mutates the caller's matrix).

### 6.7 Reset semantics (cpp:278-287) and copy behavior (cpp:155-173)

`resetWeightsDerivatives`: resets the nets' gradient accumulators + fresh `_InputStatistics`; does NOT touch `_Cost`/`_NbOfClassif` (zeroed per windowed call) nor `_OutputForward`/`_OutputBackward`. Copy ctor drops `_Trainer`, `_InputStatistics`, `_OutputForward`, `_OutputBackward` (matters for the per-thread copies in the Phase 4 driver; document on the Rust Clone impl).

## 7. Module interfaces

```rust
// nn/activations.rs
pub fn exp_limit() -> f64;                          // ln(f64::MAX), computed once
pub fn gates_fn(x: f64) -> f64;                     // sigmoid(0.1x), exclusive saturation
pub fn logistic_fn(x: f64) -> f64;                  // sigmoid(x), inclusive saturation
pub fn maxmin2_fn(x: f64) -> f64;                   // asinh
pub fn identity_fn(x: f64) -> f64;                  // asinh (distinct symbol)
pub fn asinh_fn(x: f64) -> f64;

// nn/layers.rs
pub struct LstmLayer { /* input_weights (I x 4O), feedback_weights (O x 4O), peep (12 x O), biases (4O),
                          gates/cell_states/cells_in caches, peephole flags, in/out sizes */ }
impl LstmLayer {
    pub fn new(input_size: usize, output_size: usize, cells_peep: bool, gates_peep: bool, gates_rec_peep: bool) -> LstmLayer;
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64];   // eats head, returns tail
    pub fn get_weights(&self, out: &mut Vec<f64>);
    pub fn nb_of_weights(&self) -> usize;
    pub fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool);
    pub fn feed_forward_reverse(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool);
}
pub struct NeuronLayer { /* weights (I x O), biases (O), derivative state stub */ }
impl NeuronLayer {
    pub fn new(input_size: usize, output_size: usize) -> NeuronLayer;
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64];
    pub fn get_weights(&self, out: &mut Vec<f64>);
    pub fn nb_of_weights(&self) -> usize;
    pub fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool);
}
pub(crate) fn matmul_seq(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64>;  // ascending triple loop, the shared product

// nn/network.rs
pub struct Network<L> { /* neuron_nb, sub_sampling, cumulative ratios, layers, layers_output caches */ }
impl<L: Layer> Network<L> {
    pub fn new(neuron_nb: Vec<usize>, sub_sampling: Vec<usize>, mk: impl FnMut(usize, usize, usize) -> L) -> Network<L>;
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64];
    pub fn get_weights(&self, out: &mut Vec<f64>);
    pub fn nb_of_weights(&self) -> usize;
    pub fn input_size(&self) -> usize; pub fn output_size(&self) -> usize;
    pub fn sub_sampling_ratio(&self) -> usize;                          // last cumulative
    pub fn sub_samplings(&self) -> &[usize];
    pub fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>);
    pub fn feed_forward_reverse(&mut self, input: &Array2<f64>, output: &mut Array2<f64>);
    pub fn feed_forward_double(&mut self, a: &Array2<f64>, b: &Array2<f64>, output: &mut Array2<f64>);
}
pub fn sub_sample(ratio: usize, input: &Array2<f64>) -> Array2<f64>;
pub fn repeat_rows(input: &Array2<f64>, n: usize) -> Array2<f64>;
pub trait Layer { /* feed_forward, feed_forward_reverse (recurrent only), set_weights, get_weights, nb_of_weights */ }

// nn/blstm.rs
pub struct BlstmConfig { /* lstm_neuron_nb, lstm_subsampling, output_neuron_nb, output_subsampling,
                            input_normalization_type: i16, two_sweeps, target_enforcement_step: i32,
                            peephole flags per direction, cost law params */ }
impl BlstmConfig { pub fn from_legacy(map: &IndexMap<String,String>, prefix: &str) -> anyhow::Result<BlstmConfig>; }
pub struct BlstmNetwork { /* forward: Network<LstmLayer>, backward: Network<LstmLayer>, output: Network<NeuronLayer>,
                             mean/std tails, output_forward/output_backward, cost, nb_of_classif, is_mlp,
                             truncates, overlaps, input_statistics */ }
impl BlstmNetwork {
    pub fn from_config(cfg: &BlstmConfig) -> anyhow::Result<BlstmNetwork>;
    pub fn set_weights(&mut self, flat: &[f64]) -> anyhow::Result<()>;  // asserts exact length
    pub fn get_weights(&self) -> Vec<f64>;
    pub fn nb_of_weights(&self) -> usize;
    pub fn set_processing_type(&mut self, truncates: bool, overlaps: bool);
    pub fn feed_forward_backward(&mut self, input: &mut Array2<f64>, window_size: usize, window_shift: usize,
                                 output: &mut Array2<f64>, target: &Array2<f64>);
    pub fn feed_forward_scoring(&mut self, input: &mut Array2<f64>, window_size: usize, window_shift: usize,
                                target_index: i64, target_modifier: f64) -> Array2<f64>;
    pub fn output_forward(&self) -> &Array2<f64>; pub fn output_backward(&self) -> &Array2<f64>;
    pub fn cost(&self) -> f64; pub fn nb_of_classif(&self) -> i64;
    pub fn reset_weights_derivatives(&mut self);
}
```

Python oracle (tests only): `nn_reference.py` - `lstm_forward_oracle(weights..., input, flags) -> (y, gates, cells)`, `dense_forward_oracle`, `blstm_forward_oracle` (full wrapper for the synthetic nets), plain loops, engine semantics.

## 8. Oracle harness plan

- build.sh adds TUs: BLSTMNeuralNetwork.cpp, LSTMLayer.cpp, NeuronLayer.cpp, SRNLayer.cpp, CWRNNLayer.cpp, CostLaw.cpp, Rprop.cpp (link closure verified; no new deps).
- Driver constructs the REAL `BLSTMNeuralNetwork<LSTMLayer>` from the vendored `1_worker_1.config` via the legacy ConfigFile (override `BLSTM_weightsFile` to "" via `conf.set_val`; `weightsSetExternally=true`; `setWeights` from `NNweights_config1.bin`; assert 33,671).
- Harness-local forward reimplementation (`nnForwardLoop` family): LSTMLayer::feedForward (~100 lines), NeuronLayer::feedForward (~20), the container drivers and SubSample, the BLSTM wrapper forward + normalization + windowed drivers - branch-for-branch with `// legacy: file:lines` provenance, ascending-loop products via a shared `matSeq` helper. Diff-reviewed like all transcriptions.
- Probes at every regeneration, recorded in the manifest: (a) bitwise GEMM/GEMV checks per product site (input projection GEMM, recurrence GEMV at LSTMLayer.cpp:351, dense GEMM, softmax row-sum) - real Eigen vs ascending loop on the real data shapes; (b) full-forward comparison real-class vs reimpl on the real net (tolerance: the Phase 1 hybrid bound; measured max ULP/abs recorded).
- Dumps (via Matrix2BinaryFile): real-net end-to-end (features from the Phase 1 pipeline under the real config's DSP params -> `nn_input_real.bin`, posteriors `nn_out_real.bin`, `nn_outfwd_real.bin`, `nn_outbwd_real.bin`); synthetic nets (tiny dims, closed-form deterministic weights fed via setWeights): single LSTM layer fwd/rev x peephole-flag combos x width-mismatch branches; NeuronLayer softmax/logistic/hidden-asinh; sub-sampled stacks (incl. a T not divisible by R case); each normalization mode (1 / -1 / -2 / none); windowed modes (Truncate, Truncate+TwoSweeps, OverLap incl. an uncovered-rows NaN case, MLPOverLap); the scoring forward with targets (binary expansion + enforcement-step markers); chained set_weights round-trip vectors.

## 9. Golden fixtures, strategy, acceptance

`tests/reference_data/phase2/` (committed): all dumps above + `nn_cases.json` (numpy-oracle cases) + manifest additions (probe results). Tests:
1. Activations: unit tests incl. the exclusive-vs-inclusive saturation boundaries at exactly +-exp_limit (bit-pinned), 0.1 pre-scale.
2. LstmLayer: bit-exact vs synthetic goldens (all flag combos, both directions, width branches); hand-computed O=1/T=2 recurrence anchor (arithmetic in comments); numpy-oracle JSON cross-check.
3. NeuronLayer: softmax/logistic/hidden goldens; unstabilized-softmax overflow behavior documented (no guard).
4. Network container: sub_sample indexing (hand cases incl. dropped tail frames), cascaded floor lengths, chained set_weights round-trip vs golden vectors, feed_forward_double hcat order.
5. Blstm wrapper: normalization modes bit-exact (incl. in-place mutation semantics); windowed drivers vs goldens (incl. the TwoSweeps double-width _OutputForward and the OverLap NaN rows); scoring forward (binary expansion, -0.5 markers, interior overwrite quirk).
6. END-TO-END GATE: real net + real features -> posteriors + output_forward/output_backward bit-exact vs `nn_*_real.bin` (canary-gated hybrid off-oracle).
7. Flat-seam agreement: BlstmNetwork::set_weights/get_weights round-trip == config.rs flat vectors on the real net (cross-module consistency test).
8. Fixture-presence pytest + probe-fields manifest assertions.

## 10. Risks and pitfalls (pinned by tests / logged)

- Reimplementation drift (the transcription surface is the largest yet): pinned by diff-review + the real-class tolerance probe + numpy oracle + hand anchors, all independent.
- GEMV at LSTMLayer.cpp:351 may or may not diverge like the GEMMs (different Eigen kernel): the per-site bitwise probes decide empirically; the reimpl uses ascending loops everywhere regardless.
- The stale commented ctor in BLSTMNeuralNetwork.h:35-145 and the commented reverse-loop LSTM variant (LSTMLayer.cpp:423-516): implementers must port from the LIVE code only (both flagged in the plan).
- Uninitialized peephole rows 3-11 in the config-text branch (LSTMLayer.cpp:54-124): the port zeroes them (deviation from UB, documented); production path unaffected.
- Softmax overflow (unstabilized exp): reproduce; the real net's logits are small; a synthetic golden pins the formula, not the overflow.
- iof-shim name collisions in toMat/saveWeights: harness never calls them; dumps use the harness's own helper on the public members.
- `_NbOfClassif`/`_Cost` semantics interact with CostLaw (Phase 0b-i): the scoring-forward goldens exercise computeCost through the already-ported laws - any disagreement is a cross-phase seam bug, surface loudly.

## 11. Definition of done

- `./check_all.sh`, `./lint_code.sh`, `uv run pytest tests` green; both comparator modes green (SPEECH_ORACLE_LIBM unset + =ulp).
- All 8 test groups passing; the end-to-end real-net gate bit-exact on the oracle machine.
- `nn/activations.rs`, `nn/layers.rs`, `nn/network.rs`, `nn/blstm.rs` no longer stubs; `nn/train.rs` stays a stub with an accurate doc-comment.
- CLAUDE.md: nn rows flip to Implemented (Phase 2) with accurate one-liners; SRN/CWRNN/Conv rows annotated dead-code/broken-in-legacy; module map notes the harness NN extension.
- README roadmap: Phase 2 done paragraph (oracle strategy incl. the recurrence-GEMM finding, what is compiled-TU-probed vs transcription-pinned vs oracle/hand-tested).
- IMPROVEMENTS.md entries: peephole rows 3-11 uninit (+ the port's zeroing deviation), Recurrent/Reccurent key/member spelling, dead _MaxSaturation, _NbOfSeqFedBackward uninit, exclusive-vs-inclusive saturation asymmetry, t=0 missing forget term (by-omission zero state), width-tolerance silent truncation, TwoSweeps double-width outputs, OverLap NaN rows + double-counted _NbOfClassif, MLPOverLap window_size override, scoring-forward -0.5 interior overwrite, config.rs MLP-layout gap, CNN broken-as-committed, copy-ctor dropped members.
- Final step: smart-commit over the whole branch. No push.

## 12. Out of scope

BPTT/feedBackward/Rprop/trainer state (Phase 3); SRN/CWRNN/Conv ports (dead/broken legacy code - stubs remain); segmenter wiring incl. run_pipeline lift (Phase 2b); `config.rs` MLP flat-layout variant (IMPROVEMENTS-noted); `toMat`/`saveWeights` (.mat diagnostics, Phase 4); the `BLSTM_Forward.m` MATLAB port (semantics mismatch documented in the spec, section 2 decision 4).
