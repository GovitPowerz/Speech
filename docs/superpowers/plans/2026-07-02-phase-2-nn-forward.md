# Phase 2 - NN Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the NN forward pass (activations, LSTM layer, dense layer, network container, BLSTM wrapper incl. windowed processing) bit-exactly against a strict-IEEE oracle, gated by a real trained network running on real Phase 1 features.

**Architecture:** The existing oracle harness (tools/oracle_harness/) links the real legacy NN translation units for probing, while goldens come from a harness-local ascending-loop forward reimplementation (the layer-0 GEMMs seed the recurrence, so the Phase 1 DCT-substitution problem recurs non-locally and must be baked in before any golden is cut). Rust matches the reimplementation bit-for-bit; the real compiled classes verify it at tolerance on every regeneration. An independent numpy oracle (engine semantics, NOT BLSTM_Forward.m) triple-pins the math.

**Tech Stack:** Rust (ndarray, existing io::binary/legacy_config/constants + Phase 1 comparator infra), C++ harness (Homebrew g++ strict-IEEE flags), Python (numpy plain-loop oracles, pytest).

## Global Constraints

- Branch: `feature/phase-2-nn-forward` (already created off `main`). NEVER commit to main. No rebase.
- Spec: `docs/superpowers/specs/2026-07-02-phase-2-nn-forward-design.md` - referenced as "spec S<n>". Legacy sources (git-ignored): `/Users/govit/Git/Govit/Speech/legacy/src/`.
- Commit trailer (every commit): `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Legacy quirks are load-bearing. Do NOT fix them (IMPROVEMENTS.md entries, Task 11 consolidates). NEVER edit files under `legacy/`.
- Port from LIVE legacy code only: BLSTMNeuralNetwork.h:35-145 is a STALE COMMENTED ctor; LSTMLayer.cpp:423-516 is a DEAD hand-unrolled reverse variant with different peephole grouping. Both are traps.
- All Rust matrix products/reductions: hand-written sequential ascending loops (the shared `matmul_seq`). ndarray Array2<f64>, rows = time frames.
- Golden asserts via the Phase 1 comparator: `assert_bits_eq` for portable-arithmetic chains, `assert_oracle_eq` (canary-gated hybrid) for chains reaching exp/asinh/ln - classify per test with a one-line justification comment.
- Harness flags/build conventions unchanged (build.sh, strict-IEEE, byte-identical double regeneration, manifest sync). Fixtures: `tests/reference_data/phase2/` (committed). Rust tests reach them via `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase2/...")`.
- Run tests via `cargo test --test <file>` from src/rust/; pytest/uv from repo root. Lint gates: cargo fmt --check, clippy --all-targets -- -D warnings, ./lint_code.sh. ASCII-only committed files.
- numpy oracles use PLAIN PYTHON LOOPS for accumulations (numpy .sum() is pairwise, ULP-drifts - Phase 1 lesson, documented in each oracle docstring).
- Real-net constants: `tests/reference_data/phase0/1_worker_1.config` (LSTMNeuronNb 23,24,24; sub 4,1; OutputNeuronNb 48,12,1; InputNormalizationType -1; prefix BLSTM), weights `tests/reference_data/phase0/NNweights_config1.bin` (33,671 elements). Synthetic-fixture inputs use the closed-form `((i*7 + j*13) % 100)/100.0`; synthetic weights use `((k*11 + 3) % 97)/97.0 - 0.5` (deterministic, no random).

---

### Task 1: Harness NN linkage + product probes

**Files:**
- Modify: `tools/oracle_harness/build.sh`
- Modify: `tools/oracle_harness/main.cpp`
- Modify: `scripts/extract_phase1_fixtures.py` (rename-scope note: it becomes the shared extractor; add phase2 dir handling) or Create: `scripts/extract_phase2_fixtures.py` (preferred - separate script, same conventions incl. GEMM_CHECK-style parsing)
- Create (generated): `tests/reference_data/phase2/manifest.json`
- Create: `tests/test_phase2_fixtures.py`

**Interfaces:**
- Produces: harness builds with the NN TUs linked; a real `BLSTMNeuralNetwork<LSTMLayer>` constructed from the vendored real config + weights; `matSeq(A, B)` ascending-loop product helper in main.cpp; per-site product probes (parseable `NN_PROBE site=<name> diverged=<0|1> mismatches=<n> total=<n> first=(r,c) eigen=0x... loop=0x...` lines) parsed into `manifest.json:nn_product_probes`; `tests/reference_data/phase2/` established with the manifest + presence pytest.
- Probe sites (spec S8): `lstm_input_gemm` (T x 23 times 23 x 96 on real data), `lstm_recurrence_gemv` (1 x 24 times 24 x 96, accumulated over >= 100 timesteps), `dense_gemm` (48 x 12 shapes), `softmax_rowsum` (row sums of exp on the dense output shape).

- [ ] **Step 1: Extend build.sh**

Append to the compile line: `"$SRC/BLSTMNeuralNetwork.cpp" "$SRC/LSTMLayer.cpp" "$SRC/NeuronLayer.cpp" "$SRC/SRNLayer.cpp" "$SRC/CWRNNLayer.cpp" "$SRC/CostLaw.cpp" "$SRC/Rprop.cpp"` (SRN/CWRNN are link-required by the explicit instantiations at BLSTMNeuralNetwork.cpp:960-962). Build. If the linker demands further legacy TUs, add them and record in the manifest; if a compile error hits the iof shim, extend the shim inertly (spec S8 verified the insertion forms compile - a failure here means an unanticipated form; report it, don't guess).

- [ ] **Step 2: Real-net construction in main.cpp**

New stage (provenance comments): read `tests/reference_data/phase0/1_worker_1.config` via the legacy `ConfigFile(path, '_')`; `conf.set_val<std::string>("BLSTM_weightsFile", "")`; construct `BLSTMNeuralNetwork<LSTMLayer> nn(conf, "BLSTM", true)`; load `NNweights_config1.bin` via the legacy `BinaryFile2Vector`; `if (nn.getNbOfWeights() != 33671) abort()`; `nn.setWeights(flat)`. Print `NN_REAL ok weights=33671`.

- [ ] **Step 3: matSeq + probes**

`matSeq(const MatrixXd& A, const MatrixXd& B)`: explicit `for i / for j / accumulate k ascending` product. Probes: for each site, compute the Eigen product AND matSeq on REAL data (synthetic-shaped slices of the real weights/inputs are fine for dense/softmax; the lstm sites use the real net's actual first-layer weights and a 200-row slice of a deterministic input), compare all elements bit-for-bit (memcpy to uint64_t), print the `NN_PROBE` line per site. No abort - the reimpl (later tasks) uses matSeq unconditionally; probes are measurement (Phase 1 GEMM_CHECK precedent).

- [ ] **Step 4: Extractor + manifest + presence test**

`scripts/extract_phase2_fixtures.py` (docstring with Usage line): builds harness, runs it with the phase2 output dir, parses NN_PROBE + NN_REAL lines (missing -> SystemExit), writes `tests/reference_data/phase2/manifest.json` (flags, versions, probe results, inventory - initially just the manifest). Run TWICE, byte-identical. `tests/test_phase2_fixtures.py`: manifest exists, probe fields present, weights=33671 recorded.

- [ ] **Step 5: Verify + commit**

Run: `uv run python scripts/extract_phase2_fixtures.py && uv run pytest tests/test_phase2_fixtures.py -v && ./lint_code.sh` - all green; `LC_ALL=C grep -n '[^ -~]'` on touched files empty.
```bash
git add tools/oracle_harness scripts/extract_phase2_fixtures.py tests/reference_data/phase2 tests/test_phase2_fixtures.py
git commit -m "feat(phase2): link legacy NN stack into harness; product-order probes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: nn/activations.rs

**Files:**
- Modify: `src/rust/src/nn/activations.rs` (replace stub)
- Modify: `tools/oracle_harness/main.cpp` (activation sweep dump)
- Create: `src/rust/tests/phase2_activations_golden.rs`

**Interfaces:**
- Consumes: real `ActivationFunctions.h` static fns (header-only, callable directly in the harness).
- Produces: `pub fn exp_limit() -> f64; pub fn gates_fn(x: f64) -> f64; pub fn logistic_fn(x: f64) -> f64; pub fn maxmin2_fn(x: f64) -> f64; pub fn identity_fn(x: f64) -> f64; pub fn asinh_fn(x: f64) -> f64;` per spec S3. Fixture `act_sweep.bin` (4 x N: row0 inputs, row1 GatesFunction, row2 Logistic, row3 asinh).

- [ ] **Step 1: Harness sweep dump**

Inputs: `{0.0, +-1.0, +-10.0, +-100.0, +-7097.827128933841 (exp_limit/0.1), +-7097.827128933842, +-709.782712893384, +-709.7827128933841, +-1e4, 0.5, -0.5, 42.0}` plus the exact f64s `+-exp_limit` and `+-exp_limit/0.1` computed in-harness as `std::log(std::numeric_limits<double>::max())` expressions (do NOT hardcode decimals for the boundary probes - compute them). Dump 4 x N via Matrix2BinaryFile calling the REAL `GatesFunction::fn`, `Logistic::fn`, `Maxmin2::fn`. Regenerate (twice, byte-identical), manifest sync.

- [ ] **Step 2: Failing tests**

```rust
mod common; // reuse the Phase 1 helpers via a path attribute or a copied module - check how Phase 1 test binaries mount common/ and do the same
use speech::nn::activations::*;

#[test]
fn activation_sweep_matches_oracle() {
    let dump = common::load_bin_phase2("act_sweep.bin");
    for k in 0..dump.ncols() {
        let x = dump[[0, k]];
        common::assert_oracle_eq_f64(gates_fn(x), dump[[1, k]], &format!("gates[{k}]")); // exp = libm -> oracle mode
        common::assert_oracle_eq_f64(logistic_fn(x), dump[[2, k]], &format!("logistic[{k}]"));
        common::assert_oracle_eq_f64(maxmin2_fn(x), dump[[3, k]], &format!("asinh[{k}]"));
    }
}
#[test]
fn saturation_boundaries_exact() {
    let el = exp_limit();
    assert_eq!(gates_fn(10.0 * el), 1.0);      // 0.1*x == el exactly -> EXCLUSIVE -> saturated
    assert_eq!(gates_fn(-10.0 * el), 0.0);
    assert_eq!(logistic_fn(el), 1.0);           // INCLUSIVE at boundary
    assert_eq!(logistic_fn(-el), 0.0);
    assert!(logistic_fn(el - 1.0) < 1.0);
}
```
(`load_bin_phase2` + phase2 fixture path helper: add to tests/common/mod.rs in this task - Consumes note for later tasks.)

- [ ] **Step 3: Run FAIL, implement (spec S3, ActivationFunctions.h:40-49, 158-160, 206-208, 229-238; Log.hpp:193-195), run PASS**

- [ ] **Step 4: Lint + commit** (message: `feat(phase2): activation set (0.1-gates, exclusive/inclusive saturation, asinh)`)

---

### Task 3: LstmLayer struct + weight chaining

**Files:**
- Modify: `src/rust/src/nn/layers.rs` (LstmLayer struct, set/get weights, nb)
- Modify: `tools/oracle_harness/main.cpp`
- Create: `src/rust/tests/phase2_layers_golden.rs`

**Interfaces:**
- Produces: `pub struct LstmLayer` with `new(input_size, output_size, cells_peep, gates_peep, gates_rec_peep) -> LstmLayer`, `set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64]`, `get_weights(&self, out: &mut Vec<f64>)`, `nb_of_weights(&self) -> usize` (= 4*I*O + 4*O*O + 12*O + 4*O). Internal layout per spec S4.1/S4.2 (gate blocks [i|f|o|g]; peep 12 x O). Fixtures `lstm_w_roundtrip_in.bin` / `lstm_w_roundtrip_out.bin`.

- [ ] **Step 1: Harness dump** - construct a real `LSTMLayer(conf, "SYNW", 0, /*in*/3, /*out*/2, /*weightsSetExternally*/ true)` (needs a minimal ConfigFile: reuse the real config object - the SYNW-prefixed keys all have defaults); synthetic flat vector of length nb (deterministic formula), `setWeights`, dump the input vector and the `getWeights()` output (must be identical - pins the mirror) plus a SECOND layer chained on the leftover tail (I=2,O=1) to pin head/tail semantics. Regenerate.

- [ ] **Step 2: Failing tests** - Rust round-trip == both dumps bit-exact (`assert_bits_eq` - pure copy logic, portable); chaining test (two layers consume the same concatenated vector, tail length asserted); nb_of_weights formula table for (I,O) in {(3,2),(23,24),(96,24)}.

- [ ] **Step 3-4: Implement (spec S4.1/S4.2; LSTMLayer.cpp:162-249: column-major per block, order InputWeights/FeedbackWeights/PeepWeight/Biaises), PASS, lint + commit** (message: `feat(phase2): LstmLayer weight layout + chained (de)serialization`)

---

### Task 4: LstmLayer forward + reverse (CRUX)

**Files:**
- Modify: `src/rust/src/nn/layers.rs`
- Modify: `tools/oracle_harness/main.cpp` (lstmForwardLoop reimpl + dumps)
- Modify: `src/rust/tests/phase2_layers_golden.rs`
- Create: `src/python/speech/nn_reference.py` (replace stub; `lstm_forward_oracle`)
- Create: `scripts/extract_phase2_oracle_cases.py`
- Create (generated): `tests/reference_data/phase2/lstm_cases.json`
- Modify: `tests/test_features_oracle.py` or Create `tests/test_nn_oracle.py` (anchor tests)

**Interfaces:**
- Consumes: activations (Task 2), LstmLayer struct (Task 3), matSeq (harness).
- Produces: `pub fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool)`, `pub fn feed_forward_reverse(...)` same signature; `pub(crate) fn matmul_seq(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64>` in layers.rs; Python `lstm_forward_oracle(input_w, feedback_w, peep, biases, x, flags) -> (y, gates, cells)`.

- [ ] **Step 1: Harness reimpl + dumps**

Transcribe `lstmForwardLoop` from LSTMLayer.cpp:312-413 (provenance comments; matSeq for the input projection AND the recurrence row-product; every element-wise expression in the legacy's exact combined/separate grouping per spec S4.3). Probe: run the REAL `LSTMLayer::feedForward` beside it on each dump input; print `NN_TOL site=lstm_forward max_ulp=<n> max_abs=<x>` (parsed into the manifest). Dumps (synthetic weights via setWeights on both real layer and reimpl - they share the flat vector): grid over {I=3 O=2 T=7; I=5 O=4 T=12} x {all flags on; cells only; gates only; gates-rec only; all off} x {forward, reverse} -> `lstm_fwd_<case>.bin` (T x O outputs) + `lstm_gates_<case>.bin` (T x 4O, pins in-place activation states); width-mismatch cases: input cols I+2 and I-1 (forward only). Regenerate twice.

- [ ] **Step 2: Failing Rust tests**

Goldens vs every dump (`assert_oracle_eq` - exp/asinh chains). Hand anchor (arithmetic in comments, expression-coded):
```rust
#[test]
fn lstm_hand_case_o1_t2() {
    // I=1, O=1, all weights distinct small rationals, flags all on. t=0: c0 = sig(0.1*(w_i*x0+b_i)) * asinh(w_g*x0+b_g)
    // o0_pre = w_o*x0 + b_o + c0*p2 + i0*p9 + f0*p10; y0 = sig(0.1*o0_pre) * asinh(c0)
    // t=1 adds feedback + all 12 peep terms in the spec S4.3 order - code the expected value via the same
    // sequence of f64 ops written out longhand (the golden dumps arbitrate the order; this pins structure).
    ...
}
```
Plus: reverse == flip(forward(flip)) property on a golden input; t=0 has no forget contribution (weights chosen so a wrong forget term shifts the value: f-gate bias large, c_{-1} would matter if read).

- [ ] **Step 3: Python oracle + JSON**

`lstm_forward_oracle` (plain loops, engine semantics per spec S3/S4.3; independent coding - per-timestep scalar loops, NOT a transliteration of the Rust block structure). Anchor test = the same hand case. `scripts/extract_phase2_oracle_cases.py` -> `lstm_cases.json` (deterministic weight/input formulas, the flag grid, T in {1, 2, 7}); Rust cross-check test replays via LstmLayer and asserts bit-equality (`to_bits`) - both sides are same-libm local computations.

- [ ] **Step 4: Run FAIL -> implement (spec S4.3 verbatim order) -> PASS both languages -> lint + commit** (message: `feat(phase2): LSTM forward/reverse (12-row peepholes, exact recurrence order)`)

---

### Task 5: NeuronLayer forward + weights

**Files:**
- Modify: `src/rust/src/nn/layers.rs`
- Modify: `tools/oracle_harness/main.cpp` (denseForwardLoop + dumps)
- Modify: `src/rust/tests/phase2_layers_golden.rs`
- Modify: `src/python/speech/nn_reference.py` (+ `dense_forward_oracle`), `scripts/extract_phase2_oracle_cases.py` (+ `dense_cases.json`), oracle tests

**Interfaces:**
- Produces: `pub struct NeuronLayer` with `new(input_size, output_size)`, `set_weights/get_weights/nb_of_weights` (= O*(I+1); layout: weights column-major then all biases, NeuronLayer.cpp:69-99), `feed_forward(&mut self, input, output, last_layer)` per spec S5.1.

- [ ] **Step 1: Harness reimpl (denseForwardLoop, NeuronLayer.cpp:127-149, matSeq product, sequential row-sum softmax) + NN_TOL probe vs real class + dumps: {hidden asinh (lastLayer=false)}, {softmax O=3}, {logistic O=1}, width-mismatch both directions - `dense_<case>.bin`. Regenerate.**
- [ ] **Step 2: Failing tests: goldens (`assert_oracle_eq`); hand softmax case O=2/T=1 with expression-coded expected (exp/sum/quotient order); weight round-trip bits.**
- [ ] **Step 3: Implement + oracle (`dense_forward_oracle`, plain loops incl. sequential row-sum) + `dense_cases.json` + cross-check.**
- [ ] **Step 4: PASS both -> lint + commit** (message: `feat(phase2): dense layer forward (unstabilized softmax, logistic, asinh hidden)`)

---

### Task 6: Network container

**Files:**
- Modify: `src/rust/src/nn/network.rs`
- Modify: `tools/oracle_harness/main.cpp` (container reimpl plumbing + stacked dumps)
- Create: `src/rust/tests/phase2_network_golden.rs`

**Interfaces:**
- Consumes: LstmLayer, NeuronLayer (their `Layer`-shaped methods).
- Produces (spec S7): `pub trait Layer { fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool); fn feed_forward_reverse(...); fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64]; fn get_weights(&self, out: &mut Vec<f64>); fn nb_of_weights(&self) -> usize; }` (implement for both layer types; NeuronLayer::feed_forward_reverse panics with "never run reversed" - legacy never instantiates it); `pub struct Network<L: Layer>` with `new(neuron_nb: Vec<usize>, sub_sampling: Vec<usize>, mk: impl FnMut(usize, usize, usize) -> L)`, `set_weights/get_weights/nb_of_weights`, `input_size/output_size/sub_sampling_ratio/sub_samplings`, `feed_forward/feed_forward_reverse/feed_forward_double`; `pub fn sub_sample(ratio, input) -> Array2<f64>`; `pub fn repeat_rows(input, n) -> Array2<f64>`.

- [ ] **Step 1: Harness stacked dumps** - reimpl the container driver (NeuralNetwork.hpp:125-249 SubSample + feedForward/Reverse/Double, provenance comments) wired to lstmForwardLoop/denseForwardLoop; dumps: 2-layer LSTM net [3,4,2] sub [2,1] on T=11 (odd - pins dropped-tail floor), forward and reverse -> `net_lstm_<dir>.bin`; a dense net [4,3,2] sub [1,2]; `feedForwardDouble` case (two 5-col halves -> [10,4,2]) -> `net_double.bin`; chained set_weights golden across the stack. NN_TOL probe vs the real `NeuralNetwork<LSTMLayer>` on the same inputs. Regenerate.
- [ ] **Step 2: Failing tests**: sub_sample hand cases (T=11 R=2 -> 5 rows, frame jj*R+kk at cols [kk*C,(kk+1)*C), tail frame 10 dropped - exact expected matrix written out); repeat_rows; cascaded floors (T=11, subs [2,2] -> 5 -> 2, NOT 11/4=2 coincidence - pick T=13 subs [2,3]: 13->6->2 vs 13/6=2; add a case where they DIFFER: T=11 subs [2,3]: 11->5->1 vs 11/6=1... choose T=15 subs [2,3]: 15->7->2 vs 15/6=2; and T=17 subs [3,2]: 17->5->2 vs 17/6=2. Hmm - use T=11 subs [3,2]: 11->3->1 while 11/6=1; the distinguishing case is T=14 subs [2,3]: 14->7->2 vs floor(14/6)=2... Compute one that differs: T=10 subs [3,2]: 10->3->1, product floor 10/6=1. T=16 subs [3,2]: 16->5->2, 16/6=2. T=22 subs [4,3]: 22->5->1, 22/12=1. T=9 subs [2,4]: 9->4->1, 9/8=1. T=15 subs [4,2]: 15->3->1, 15/8=1. T=14 subs [4,2]: 14->3->1 vs 14/8=1. T=23 subs [4,3]: 23->5->1 vs 23/12=1. T=27 subs [4,3]: 27->6->2 vs 27/12=2. Sequential-vs-product differ when floor(floor(T/a)/b) != floor(T/(a*b)): impossible - floor division nests exactly. THEY NEVER DIFFER for two levels (floor(floor(x/a)/b) == floor(x/(ab)) is an identity). The legacy's sequential division still must be ported as written; test the identity case and note the identity in a comment instead of hunting a phantom counterexample); container goldens (`assert_oracle_eq`); double hcat order (forward LEFT half - construct asymmetric halves so a swap fails).
- [ ] **Step 3: Implement (spec S5.2) -> PASS -> lint + commit** (message: `feat(phase2): network container (sub-sample stacking, chained weights, double-input MLP entry)`)

---

### Task 7: BlstmConfig + BlstmNetwork construction + flat seam

**Files:**
- Modify: `src/rust/src/nn/blstm.rs`
- Create: `src/rust/tests/phase2_blstm_golden.rs`

**Interfaces:**
- Consumes: `speech::legacy_config::parse_legacy_config`, Network/layers, `speech::io::binary::read_matrix` (or the vector variant - check io/binary.rs), `speech::config` (element_count for the cross-check).
- Produces (spec S6.1/S6.2/S7): `pub struct BlstmConfig { ... } impl BlstmConfig { pub fn from_legacy(map, prefix) -> Result<BlstmConfig> }`; `pub struct BlstmNetwork` with `from_config`, `set_weights(&mut self, flat: &[f64]) -> Result<()>`, `get_weights() -> Vec<f64>`, `nb_of_weights()`, accessors. MLP mode (`lstm_neuron_nb[0] == 0`) supported structurally.

- [ ] **Step 1: Failing tests** - from_legacy on the real 1_worker_1.config (typed asserts: [23,24,24], [4,1], [48,12,1], [1... wait output sub has len 2], norm type -1, TwoSweeps value as in the file - read the file in the test); validation errors (short lists, the 2*last constraint per BLSTMNeuralNetwork.cpp:43-46); nb_of_weights == 33671 for the real net; set_weights(NNweights bin) -> get_weights round-trip bit-identical to the file; cross-check `speech::config::element_count(&spec) == nb_of_weights()` (the Phase 0a seam agreement, spec decision 5); MLP-mode structural test (lstm[0]=0 -> nets absent, tail = output input size).
- [ ] **Step 2-3: Implement (spec S6.1/S6.2; BLSTMNeuralNetwork.cpp:26-153, 209-253) -> PASS -> lint + commit** (message: `feat(phase2): BLSTM wrapper construction + flat weight seam (33,671 round-trip)`)

---

### Task 8: Input normalization + core forward

**Files:**
- Modify: `src/rust/src/nn/blstm.rs`
- Modify: `tools/oracle_harness/main.cpp` (blstm wrapper reimpl: normalization + core forward; dumps)
- Modify: `src/rust/tests/phase2_blstm_golden.rs`
- Modify: `src/python/speech/nn_reference.py` (+ `blstm_forward_oracle` for the synthetic full wrapper), cases script + JSON

**Interfaces:**
- Consumes: everything above; `speech::features::stats::InputStatistics` (type-1 branch calls it).
- Produces: normalization (types 1/-1/-2/other per spec S6.3 - types 1 and -1 mutate the caller's matrix), `feed_forward` core (spec S6.4: sequential-division output length, leftCols gate `ratios[0] > 1 && wider`, forward-net forward + backward-net reverse, feedForwardDouble), plain `feed_forward_backward` (no windowing; cost accumulation incl. the `_TargetEnforcementStep < 0` interior -0.5 overwrite quirk, spec S6.5 plain-FFB).

- [ ] **Step 1: Harness reimpl + dumps**: blstm wrapper forward (BLSTMNeuralNetwork.cpp:419-437, 711-751, 776-830 transcribed with provenance; NN_TOL probe vs the real class); dumps on a synthetic net [3,4,2]/[2,1]/[16,5,2]/[1,1] (16 = 2*2*4? no - output[0] must equal 2*lstm.back() = 4; use [4,5,2]... set lstm [3,4,2] -> output [4,5,3] sub [1,1]) with T=12: one per normalization type (1 with a nonzero mean/std tail in the flat vector; -1; -2; 0) -> `blstm_norm<type>_out.bin` + the mutated input dumps `blstm_norm<type>_input_after.bin` (pins in-place semantics); real-net full-sequence dump: excerpt features are NOT yet needed - use a deterministic synthetic 200 x 23 input -> `blstm_real_fullseq_{out,fwd,bwd}.bin` (the real net, plain FFB, no targets). Regenerate.
- [ ] **Step 2: Failing tests**: goldens (oracle_eq); in-place mutation asserted (input compared to the after-dump); type-1 columns-beyond-tail untouched; cost-quirk unit test (enforcement step -1: output interior == -0.5 after the call, expression-coded).
- [ ] **Step 3: Implement + `blstm_forward_oracle` (synthetic net sizes, plain loops) + JSON cross-check -> PASS -> lint + commit** (message: `feat(phase2): input normalization modes + core bidirectional forward`)

---

### Task 9: Windowed drivers (CRUX)

**Files:**
- Modify: `src/rust/src/nn/blstm.rs`
- Modify: `tools/oracle_harness/main.cpp`
- Modify: `src/rust/tests/phase2_blstm_golden.rs`

**Interfaces:**
- Produces: `set_processing_type(truncates, overlaps)`, dispatch + the four windowed drivers per spec S6.5: TruncateSweep (BLSTMNeuralNetwork.cpp:488-546), Truncate+TwoSweeps (:548-590), OverLap (:592-681), MLPOverLap (:683-709). Full `feed_forward_backward(input, window_size, window_shift, output, target)` signature final.

- [ ] **Step 1: Harness reimpl + dumps** (transcribed with provenance; NN_TOL probes): on the synthetic net - Truncate window 6 over T=20 (dropped trailing chunk: T=20 window 6 sub-ratio 2 -> last chunk 2 rows... choose T=21 so the tail chunk of 3 < ratio*? construct a case where lengthShort==0 documented in a comment); TwoSweeps (window 8; assert double-width _OutputForward in-harness); OverLap (window_size 4 shift 3 with grid snapping exercised + an uncovered-rows NaN case: shift > 2*window+1); MLPOverLap on an MLP-mode net. Dumps: outputs + _OutputForward/_OutputBackward per mode -> `blstm_<mode>_{out,fwd,bwd}.bin`. Regenerate.
- [ ] **Step 2: Failing tests**: goldens per mode (oracle_eq; the NaN rows asserted via is_nan on the exact expected rows); TwoSweeps output width == 2x; MLPOverLap ignores the passed window_size (call with a wrong one, same result); dispatch table unit test.
- [ ] **Step 3: Implement (spec S6.5 verbatim: begin/end snapping formulas, count averaging, `(window_size/2)/ratio` division order, edge replication) -> PASS -> lint + commit** (message: `feat(phase2): windowed forward drivers (truncate, two-sweeps, overlap, MLP-overlap)`)

---

### Task 10: Scoring forward + END-TO-END real-net gate (CRUX)

**Files:**
- Modify: `src/rust/src/nn/blstm.rs`
- Modify: `tools/oracle_harness/main.cpp`
- Create: `src/rust/tests/phase2_e2e_gate.rs`

**Interfaces:**
- Consumes: the whole Phase 1 feature pipeline (audio, windowing, periodogram, mel/DCT, LTSV, pipeline derive/assemble) + the whole Phase 2 nn stack.
- Produces: `feed_forward_scoring(input, window_size, window_shift, target_index, target_modifier) -> Array2<f64>` per spec S6.6 (soft targets 0.1*modifier, unknown-class -> 0, enforcement counter with row 0 always enforced, -0.5 markers, binary [1-p, p] expansion, rows < ratio -> Zero(1, out)); the E2E gate fixtures.

- [ ] **Step 1: Harness scoring dumps** (transcribed :843-929): synthetic binary net with targets (enforcement steps 0 and 2; targetModifier cases; cost-modified vs not - read isCostModified from the CostLaw port semantics, cite CostLaw source line in a comment) -> `blstm_scoring_<case>.bin` + recorded `_Cost`/`_NbOfClassif` values in the manifest.
- [ ] **Step 2: E2E harness leg**: run the Phase 1 feature pipeline transcriptions ALREADY IN main.cpp under the REAL config's BLSTM_* DSP keys on the excerpt wav (chan 1) -> inputSeq; then the real-net blstm reimpl forward, full-sequence (setProcessingType(false,false) equivalent) AND one OverLap variant (window_size 25, shift 12 - explicit constants recorded in the manifest, since getBLSTMParam derivation is Phase 2b) -> `e2e_input.bin`, `e2e_{out,fwd,bwd}_{full,overlap}.bin`. NN_TOL probe vs the real class on both. Regenerate twice.
- [ ] **Step 3: Failing Rust tests**: scoring unit goldens + cost/classif assertions; THE GATE in phase2_e2e_gate.rs - full Rust chain: read_audio(excerpt, 0.35, 2.0) -> preemph per config -> SpectralParams::derive(BLSTM keys) -> periodogram -> mel/DCT -> LTSV -> assemble_input_sequence -> BlstmNetwork(from 1_worker config + NNweights bin) -> feed_forward_backward -> assert_oracle_eq vs all e2e dumps (input asserted first - it re-validates Phase 1 under the REAL config for free).
- [ ] **Step 4: Implement scoring -> PASS everything -> lint + commit** (message: `feat(phase2): scoring forward; end-to-end real-net gate green (Phase1 features -> posteriors)`)

---

### Task 11: Docs, IMPROVEMENTS, verification, smart-commit prep

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `IMPROVEMENTS.md`, `src/rust/src/nn/mod.rs`, `src/rust/src/nn/train.rs` (doc-comment accuracy only)

- [ ] **Step 1: IMPROVEMENTS.md** - add the spec S11 list: peephole rows 3-11 uninit in the config-text branch (+ the port's zeroing deviation); Recurrent/Reccurent key-vs-member spelling; dead `_MaxSaturation`; `_NbOfSeqFedBackward` uninit; GatesFunction-exclusive vs Logistic-inclusive saturation; t=0 missing forget term; width-tolerance silent truncation both directions; TwoSweeps double-width outputs; OverLap NaN rows + double-counted `_NbOfClassif`; MLPOverLap window_size override; scoring-forward -0.5 interior overwrite; config.rs MLP flat-layout gap; CNN broken-as-committed (UB indexing, orphaned weights); copy-ctor dropped members; the NN product-order substitution (manifest-measured, references nn_product_probes). No duplicates with existing entries.
- [ ] **Step 2: CLAUDE.md** - nn rows -> Implemented (Phase 2) one-liners (blstm/layers/network/activations); SRN/CWRNN annotated "dead code in legacy - not ported (stubs)"; conv row annotated broken-as-committed; harness note extended (NN TUs linked + probes).
- [ ] **Step 3: README roadmap** - Phase 2 done paragraph: oracle strategy (recurrence-GEMM finding, reimpl-with-probes), the E2E gate (Phase1 features -> real net -> posteriors), what is probe-verified vs transcription-pinned vs oracle/hand-tested, BLSTM_Forward.m mismatch note.
- [ ] **Step 4: Full verification** - `./check_all.sh && ./lint_code.sh && uv run pytest tests -v`; both comparator modes (`SPEECH_ORACLE_LIBM=ulp` and unset) for cargo + pytest; ASCII sweep. Commit:
```bash
git add -A   # verify git status first
git commit -m "docs(phase2): mark NN forward implemented; IMPROVEMENTS consolidation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
- [ ] **Step 5: smart-commit** - controller-run over the whole branch (doc sync), then final whole-branch review + finishing-a-development-branch.

---

## Self-Review

**1. Spec coverage:** S3 -> Task 2; S4.1/4.2 -> Task 3; S4.3 -> Task 4; S5.1 -> Task 5; S5.2 -> Task 6; S6.1/6.2 -> Task 7; S6.3/6.4 + plain FFB of S6.5 -> Task 8; S6.5 windowed -> Task 9; S6.6 + acceptance gate -> Task 10; S8 harness plan -> Tasks 1,4,5,6,8,9,10 incrementally; S9 tests 1-8 -> mapped (test 7 flat-seam agreement in Task 7; test 8 presence in Task 1); S10 risks: stale-code traps (Global Constraints), GEMV probe (Task 1 sites + Task 4 NN_TOL), softmax overflow (Task 5 comment), iof collisions (Task 1 Step 2 avoids toMat), CostLaw seam (Task 10 cost assertions); S11 -> Task 11. Reset semantics (S6.7): `reset_weights_derivatives` lands in Task 7 (struct-level state) with a unit test - covered by the S6.7 citation in Task 7's implement step. Gap check: none found.

**2. Placeholder scan:** Task 6 Step 2 contains an explicit worked-through note that sequential-vs-product floor division is an identity for nested floors (the test pins the sequential form + documents the identity) - deliberate, not a placeholder. Intricate port bodies follow the established RE-with-golden pattern (spec section + legacy lines + shown anchors as arbiter). No TBD/TODO.

**3. Type consistency:** LstmLayer/NeuronLayer set_weights slice-chaining signature identical across Tasks 3/5/6/7; Layer trait (Task 6) matches the methods Tasks 3-5 defined; BlstmNetwork method names consistent across Tasks 7-10 and match spec S7; fixture names consistent between producing harness steps and consuming tests; `load_bin_phase2`/phase2 path helper introduced in Task 2 and consumed thereafter; matSeq (harness) vs matmul_seq (Rust) naming split is intentional and documented in Task 4's Interfaces.
