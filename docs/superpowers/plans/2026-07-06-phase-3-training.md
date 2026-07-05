# Phase 3 - Training (network-level BPTT + iRPROP-) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the complete network-level training chain - `CostLaw` backward seam, `NeuronLayer`/`LSTMLayer` `feed_backward`, `Network` container backward (incl. the `SubSample` inversion), the `BlstmNetwork` windowed-backward fan-out + `get_weights_derivatives` (Nx2) + `update_weights`, the `Rprop` iRPROP- trainer, a network-level grad check, and the Python engine-semantics backward oracle - bit-exact against a strict-IEEE oracle harness whose backward machinery is either the REAL compiled legacy code (Rprop, CostLaw deltas) or a harness-local ascending-loop reimpl (layer/container/BLSTM backward) with NN_TOL calibration against the real classes.

**Architecture:** Four oracle tiers, strongest first (spec S2): (1) iRPROP- probes the REAL compiled `Rprop::updateWeights` directly, strict bits everywhere (pure scalar, no libm/GEMM); (2) CostLaw backward probes the REAL compiled `computeDeltas`/`computeUnitaryDeltas`/`deriv()` (polynomial laws bit-exact, log/sqrt canary-gated); (3) layer/container/BLSTM backward goldens are DUMPED from a harness-local ascending-loop reimpl (the backward contains Eigen GEMMs at NN-wide `k` - same blocked-vs-ascending divergence as the Phase 2 forward, doubly non-local because the forward already seeds the caches), with NN_TOL lines recording the real-class deltas (synthetic sites 0 ULP; real-net sites Phase 4 calibration); (4) the Python `nn_reference.py` backward oracle is an independent scalar-loop arbiter cross-checked bit-for-bit vs the harness dumps AND the Rust layers on synthetic shapes. The Rust ports live in `cost.rs`, `nn/{layers,network,blstm,train}.rs`; the grad check is a bound-based validation harness (not a byte-golden).

**Tech Stack:** Rust (existing ports: `cost.rs`, `nn/*`, `features/stats.rs`), C++ harness (Homebrew g++ strict-IEEE, links `BLSTMNeuralNetwork LSTMLayer NeuronLayer CostLaw Rprop` already - `build.sh:95-96`), Python (`nn_reference.py` oracle + pytest cross-checks).

## Global Constraints

- Branch: `feature/phase-3-training` (created off `main` at `999d58a`, contains Phases 0a-2b; the working head is `1340719`). NEVER commit to `main`. No rebase.
- Spec: `docs/superpowers/specs/2026-07-06-phase-3-training-design.md` ("spec S<n>"). Legacy sources (git-ignored): `/Users/govit/Git/Govit/Speech/legacy/src/`. NEVER edit `legacy/`.
- Commit trailer (every commit), verbatim: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- NEVER push. NEVER use the `gh` CLI. Commit only; the controller runs smart-commit + finishing-a-development-branch at the end (Task 10).
- Legacy quirks are load-bearing - PORT, do not fix. Every reproduced quirk gets an IMPROVEMENTS.md entry (Task 10 consolidates; log candidates as you go). The spec S12 list is the checklist.
- **Bit-exact contract**: goldens assert STRICT BITS where the value is pure arithmetic or pure scalar (Rprop everywhere; polynomial CostLaw derivs; the Nx2 layout/count columns; structural min/max/nonzero counts). Everything downstream of libm (asinh/exp/log/sqrt/cos in the layer/BLSTM backward, LogLaw/SqrtLaw derivs, the grad-check cost) is CANARY-GATED via `assert_oracle_eq`/`assert_oracle_eq_f64` (Rust) and `tests/_libm_gate.py` (Python).
- **CROSS-LIBM RULE (binding, 4th-occurrence lesson)**: any assert comparing a committed fixture/JSON/manifest value against a value recomputed through libm goes through the canary gate. "Both sides local" counts ONLY if both are computed in the same run.
- **NaN policy (arch-sign clause)**: any NaN a training path can produce (count-0 normalization `0/0`, LogLaw at 0) compares bit-exact on the oracle env; OFF the oracle env compare by `is_nan` (the arch-defined quiet-NaN sign differs; the CLAUDE.md NaN-sign rule extended in P2b covers this).
- **Ascending-loop product contract**: every new numeric product uses the measured ascending-loop order - `matmul_seq` (Rust, `nn/layers.rs:16`) / `matSeq` (harness, `main.cpp:216`) / `_matmul_seq_row` (Python, `nn_reference.py:61`). Eigen's blocked GEMM diverges at NN `k`; do NOT use `ndarray::dot`/`np.dot`/Eigen `*` for any golden-bearing product.
- ASCII-only in all files (source, tests, docs, commit messages). Plain hyphens and straight quotes.
- **Fixtures**: `tests/reference_data/phase3/`. Harness build.sh is LOCAL-ONLY (CI consumes the committed fixtures, never builds the harness). Regenerate fixtures TWICE byte-identical. Manifest values are MEASURED (read from the harness run), not hardcoded. Hash-guard: the extractor asserts no `phase1/phase2/phase2b` fixture drifts after the run. `// legacy: file:lines` provenance comment on every transcription.
- Run Rust tests via `cargo test --test <file>` from `src/rust/`; pytest/uv from repo root. Lint gates: `cargo fmt --check`, `clippy --all-targets -- -D warnings`, `./lint_code.sh`. Full gate: `./check_all.sh && ./lint_code.sh && uv run pytest tests`.
- **Subagents do the work themselves** - no delegation-of-delegation, no worktrees spawned from within a task, no background children for the implementation itself.

### Real-config facts that bind this phase

- `tests/reference_data/phase0/1_worker_1.config` = Algo 3, prefix `BLSTM`, 33,671-weight net, `NNweights_config1.bin`. Input width D=11, net input 23 (D<23 -> LSTM `topRows` width tolerance). `sub_sampling_ratio()` = 4 (LSTM `[4,1]` x output `[1,1]`). `InputNormalizationType -1`.
- **`BLSTM_BackPropagationActivated false`** in the real config, and no `BLSTM_BackPropagationRpropInit` key (default 1e-2). The E2E gradient gate (Task 8) and every backward-bearing harness stage MUST override `BackPropagationActivated -> true` via the established `conf.set_val<bool>(...)` / `conf._Params.erase(...)` pattern (`main.cpp:2100,2109,2432`), exactly as the Phase 2 forward probes flipped it to `false`. `BackPropOutputNetworkOnly` default `false`; `TargetEnforcementStep` default `0`.
- Excerpt wav: `tests/reference_data/phase1/excerpt_2ch_8k.wav` at offset 0.35 dur 2.0, rate 8000. The Phase 2 e2e assembled input (`e2e_input.bin`, width 11) is the real-net feature fixture; reuse it.

### Deriv-object layout (spec S3) - the single biggest trap (risk R1)

`get_weights_derivatives()` returns Nx2: **col0 = SUMMED derivative, col1 = `_NbOfSeqFedBackward` (frame count) REPLICATED per element** (`LSTMLayer.cpp:251-295`, `NeuronLayer.cpp:101-114`, `NeuralNetwork.hpp:98-111` vertical hcat). `BlstmNetwork::get_weights_derivatives` (`:255-276`) hcats fwd|bwd|output col0/col1 then appends the mean/std tail as `[Zero(mean) | Ones | Zero(std) | Ones]` (4 tail blocks - stats never move, count 1). If `OutputSubSamplingRatio > 1` the fwd/bwd LSTM col0 is multiplied by it (`:266-269`). The flat ordering matches the P0a weight packer element-for-element. Normalization is `col0 cwiseQuotient col1` - ELEMENT-WISE, at UPDATE TIME ONLY (`BLSTMNeuralNetwork.cpp:306`), never inside the backward. Different regions carry different counts (mean/std tail count=1; LSTM counts accumulate frames x sweeps x per-window coverings). A scalar `/nframes` is WRONG.

### SubSample-backward resolution (spec S6 crux, RESOLVED from `NeuralNetwork.hpp:251-300`)

The legacy `feedBackward` does NOT drop decimated rows: for a layer with `_SubSampling[jj] > 1` it (1) re-`SubSample`s the layer's stored input (`Input` for layer 0, `_LayersOutput[jj-1]` otherwise) to reconstruct the decimated forward input, truncated to the incoming delta row count via `.topRows(deltas.rows())`/`.topRows(deltas_out.rows())`; (2) calls the layer's `feedBackward` passing the running `invSubSamplingRatio` (the layer scales its four deriv blocks by it, `LSTMLayer.cpp:710-715`); (3) `InvSubSample`s the RETURNED `deltas_out` back to the pre-decimation temporal resolution (`NeuralNetwork.hpp:136-145`: each decimated delta row `jj` is split across `ratio` output rows, output row `jj*ratio+kk` = `Input.block(jj, kk*cols, 1, cols)` - the inverse of the forward's column-block stacking); (4) multiplies `invSubSamplingRatio *= _SubSampling[jj]` so the NEXT (earlier) layer scales by the accumulated ratio. The `.topRows(...)` on the reconstructed input aligns it to the delta row count (`deltas` for the last/first-sub layer, `deltas_out` for interior layers - note the two different bounds at `:271` vs `:280`). Layer 0's `deltas_out` is `InvSubSample`d and returned to the BLSTM wrapper. This is transcribed faithfully in Task 5; the `[4,1]` real net exercises exactly the layer-0 `_SubSampling[0]=4 > 1` arm (Task 8 E2E covers it non-vacuously).

### What is ALREADY ported (do NOT rewrite - verify + golden-test only)

- **`cost.rs` backward is DONE**: `Law::deriv()` (all 7 laws, `cost.rs:71-108`), `compute_unitary_delta` (`:290-316`), `compute_deltas` (`:385-442`) already exist and match `CostLaw.h:6-213` + `CostLaw.cpp:278-420` on inspection. Task 2 is REAL-probe golden VALIDATION of the existing code (not a rewrite), plus wiring `computeDeltas`/`compute_unitary_delta` into the BLSTM `feed_backward` seam and confirming the ponderation plumbing (`classes_ponderations()` present; `getCostPonderation` is a SCORING/LID path, NOT a training-gradient input - verified, no plumbing gap). This is a spec deviation from S4's "add deriv()" framing - see Self-Review.

---

### Task 1: Harness backward reimpl foundation + NN_TOL backward probes + extractors + fixture scaffolding

**Files:**
- Modify: `tools/oracle_harness/main.cpp` (backward reimpl family + NN_TOL/NN_PROBE backward probe stages)
- Create: `scripts/extract_phase3_fixtures.py` (`.bin`/manifest goldens; mirrors `extract_phase2_fixtures.py`)
- Create: `scripts/extract_phase3_oracle_cases.py` (JSON cross-check cases for `nn_reference.py`; mirrors `extract_phase2_oracle_cases.py`)
- Create (generated): `tests/reference_data/phase3/manifest.json`
- Create: `tests/test_phase3_fixtures.py`
- Modify: `src/rust/tests/common/mod.rs` (add `fixture_phase3`/`load_bin_phase3`)

**Interfaces:**
- Produces: a harness-local ascending-loop backward reimpl family mirroring the Phase 2 forward reimpl (`lstmBackwardLoop`, `denseBackwardLoop`, `netBackwardLoop`/`netBackwardReverseLoop`, `blstmFeedBackwardLoop` + `blstmFeedBackwardDoubleLoop`), built on `matSeq` and the existing forward reimpl statics; NN_TOL lines `site=lstm_backward|dense_backward|net_lstm_backward_{fwd,rev}|net_dense_backward|blstm_feedbackward|blstm_real_backward` (synthetic sites `max_ulp=0`; real-net site nonzero, recorded); the phase3 fixture dir + presence pytest; the `fixture_phase3`/`load_bin_phase3` helpers.
- Consumes: `matSeq` (`main.cpp:216`), the Phase 2 forward reimpl statics (`lstmForwardLoop`/`denseForwardLoop`/`blstmFeedForwardLoop`), the real 33,671-weight net (`1_worker_1.config` + `NNweights_config1.bin`), `e2e_input.bin`.

- [ ] **Step 1: Backward reimpl family in main.cpp**

Add a harness-local ascending-loop backward reimpl mirroring the Phase 2 forward reimpl structure. Transcribe from the legacy with `// legacy: file:lines` provenance on each block:
- `lstmBackwardLoop(...)` from `LSTMLayer.cpp:518-723`: reverse-time loop, the gate-derivative order O -> state -> C -> F -> I, the 12-row peephole index map (rows 0,1,2 cells; 4,5,6,8,9,10 gates; 3,7,11 gates-rec - SAME map as the forward, survey S1.5), the `row==0` no-cellstate forget branch (`:648-657`), the `deltasForgetGatetmp` save (`:624`, read at `:660` for the input gate), the width tolerance (`:534-543`: `cols > _InputSize` -> `leftCols` transpose; `cols < _InputSize` -> ZERO-PAD to `_InputSize` rows), the `testM`/`test` 4O-stacking, `deltasPreviousLayer = (_InputWeights*testM).transpose()` and `deltasFeedback = (_FeedbackWeights*test).transpose()` via `matSeq`, the `invSubSamplingRatio > 1` block scaling (`:710-715`), accumulate into the four deriv members, `_NbOfSeqFedBackward += linesNb` (`:716-720`). Derivatives recomputed FROM the post-activation caches (`_Gates`/`_CellStates`/`_CellsIn`) via `CwiseActFunctionDeriv<>` - NO forward change.
- `denseBackwardLoop(...)` from `NeuronLayer.cpp:151-207`: `weightsDerivatives += input.col(jj)*deltas.row(jj)` per row, `biaisesDerivatives += deltas.colwise().sum()` (`:178-183`); `lastLayer` -> `deltas*_Weights.transpose()` with NO activation deriv; hidden -> `* Maxmin2'(InputSeq)` (`:204`); `invSubSamplingRatio` scaling (`:192-195`); `_NbOfSeqFedBackward += InputSeq.rows()` (`:199` - note `InputSeq.rows()`, not `deltas.rows()`).
- `netBackwardLoop`/`netBackwardReverseLoop` from `NeuralNetwork.hpp:251-351`: the reverse-order layer loop with the SubSample/InvSubSample inversion + running `invSubSamplingRatio` (per the resolution above). `netBackwardReverseLoop` = reverse input/output/deltas, call `netBackwardLoop`, reverse the returned deltas (`feedBackwardReverse` `:302-351` / `LSTMLayer.cpp:725-732`).
- `blstmFeedBackwardLoop`/`blstmFeedBackwardDoubleLoop` from `BLSTMNeuralNetwork.cpp:439-460` + `NeuralNetwork.hpp:353-362`: seed deltas via the REAL compiled `computeDeltas` (call it - CostLaw is bit-portable for polynomial laws; the golden config uses a polynomial law), `feedBackwardDouble` (hcat `_OutputForward|_OutputBackward`, MLP backward, split left/right), then fwd `feedBackward` + bwd `feedBackwardReverse` with the `leftCols(inputSize)` crop gate (`:452-454`). `_BackPropOutputNetworkOnly` short-circuit.

DEAD - do NOT port: `LSTMLayer.cpp:423-516` + `:734-913` (commented alternate reverse bodies), `_MaxSaturation` logic (`:693-705` commented), `Rprop.cpp:60-76` (.mat dump).

- [ ] **Step 2: NN_TOL/NN_PROBE backward probe stages**

Mirror the Phase 2 probe pattern (`main.cpp:2373-2566` forward): construct the REAL compiled layer/net, run BOTH the real `feedBackward` and `lstmBackwardLoop`/etc on the same operands, harvest `getWeightsDerivatives()` from each, compare every element by bit (ULP), print `NN_TOL site=<name> max_ulp=<n>`. Synthetic small shapes (O <= 3, k < 23) MUST be `max_ulp=0` (assert in the extractor). The real-net site (`blstm_real_backward`, O=24/k=23-48) records nonzero as Phase 4 calibration. Also add an NN_PROBE-style 0-ULP structural cross-check on the pure-arithmetic backward products at synthetic k (the `input.col(row)*deltas` outer products are small-k, expected 0 ULP - assert). Dump the reimpl-produced Nx2 derivs as the goldens (Tasks 3-6 consume them), not the real-Eigen derivs.

- [ ] **Step 3: Extractors + presence pytest**

`scripts/extract_phase3_fixtures.py` (docstring Usage; builds the harness; runs it in a throwaway dir seeded with the Phase 1/2 inputs; parses all NN_TOL/NN_PROBE lines into `manifest.json` with the synthetic-0-ULP assertion + real-net calibration record; copies the Tasks 2-8 `.bin` goldens into `tests/reference_data/phase3/`; run TWICE byte-identical). CRITICAL regression guard (copy the phase2b pattern, `extract_phase2b_fixtures.py:294-303,428-435`): hash `phase1`/`phase2`/`phase2b` before/after and `SystemExit` on drift. `scripts/extract_phase3_oracle_cases.py`: dumps the JSON backward oracle-cases (weights + inputs + targets + expected Nx2 derivs) that `nn_reference.py`'s cross-check (Task 9) consumes, mirroring `extract_phase2_oracle_cases.py`. `tests/test_phase3_fixtures.py`: manifest present, all synthetic NN_TOL `max_ulp == 0`, the real-net calibration line present + recorded, expected shapes match.

- [ ] **Step 4: Rust fixture helpers**

Add `fixture_phase3(name)`/`load_bin_phase3(name)` to `src/rust/tests/common/mod.rs` (mirror `fixture_phase2b`/`load_bin_phase2b` `:40-50`).

- [ ] **Step 5: Verify + commit**

Run: `uv run python scripts/extract_phase3_fixtures.py && uv run python scripts/extract_phase3_oracle_cases.py && uv run pytest tests/test_phase3_fixtures.py -v && ./lint_code.sh`; `cd src/rust && cargo test` (unaffected, green); ASCII sweep; regenerate-twice check.
```bash
git add tools/oracle_harness scripts/extract_phase3_fixtures.py scripts/extract_phase3_oracle_cases.py tests/reference_data/phase3 tests/test_phase3_fixtures.py src/rust/tests/common/mod.rs
git commit -m "feat(phase3): harness backward reimpl + NN_TOL backward probes + phase3 fixtures

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: CostLaw backward seam validation + REAL-probe goldens + BLSTM deltas wiring

**Files:**
- Modify: `src/rust/src/cost.rs` (only if a probe reveals a divergence; the deriv chain is already ported - primary work is validation)
- Modify: `tools/oracle_harness/main.cpp` (CostLaw backward probe stage)
- Create: `src/rust/tests/phase3_costlaw_backward_golden.rs`
- Modify: `scripts/extract_phase3_fixtures.py` (CostLaw deriv dumps)

**Interfaces:**
- Consumes: `CostLaw::compute_unitary_delta(output, target) -> f64` (`cost.rs:290`), `CostLaw::compute_deltas(&[f64], &[f64], n_classes, &mut [f64])` (`cost.rs:385`), `CostLaw::from_config(&IndexMap, prefix)` (`cost.rs:207`), `assert_oracle_eq_f64`/`assert_bits_eq` (`common/mod.rs`).
- Produces: `cost_deriv_scalar_speech.bin` / `cost_deriv_scalar_other.bin` (Nx2: `[output | delta]` sweeps for a polynomial law + a log law + a sqrt law), `cost_deltas_multiclass.bin` (softmax+CE fusion, incl. an ignore-masked row and a ponderation variant), `cost_deltas_wer.bin` (the BackPropWER-scaled multiclass path) - all DUMPED from the REAL compiled `computeUnitaryDeltas`/`computeDeltas`. The `feed_backward` seam in `nn/blstm.rs` calls `compute_deltas` (multiclass) / `compute_unitary_delta` (scalar VAD) to seed the output-layer deltas (delivered in Task 6; the deltas function is verified here).

- [ ] **Step 1: Harness CostLaw backward probe + dumps**

New stage: construct a REAL `CostLaw` from three configs (a polynomial law - `square`; a `log` law; a `sqrt` law) via the same `ConfigFile` + `set_val` override pattern. Dump:
- Scalar VAD: sweep `output` over `[0,1]` (deterministic grid, e.g. `k/64`), for `target=1.0` (speech) and `target=0.0` (other), calling the REAL `computeUnitaryDeltas` -> `cost_deriv_scalar_speech.bin`/`cost_deriv_scalar_other.bin` (Nx2 `[output | delta]`). This exercises the below/above switching, the `-deriv` no-speech negation, the `output*(1-output)` logistic fold (`CostLaw.cpp:344`), and the WER scaling when the config sets `BackPropWER >= 0`.
- Multiclass: a deterministic `n_frames x n_classes` output + one-hot target incl. (a) an ignore-masked row (`target < 0`), (b) an on-class `target > 0.5` row, via the REAL `computeDeltas` -> `cost_deltas_multiclass.bin`. Add a `classes_ponderations`-set variant and a `BackPropWER`-set variant -> `cost_deltas_wer.bin`.

The polynomial-law dumps assert STRICT BITS (Rust); the `log`/`sqrt` law dumps are CANARY-GATED (`assert_oracle_eq_f64`). Provenance comments cite `CostLaw.cpp:278-420` + `CostLaw.h:6-213`.

- [ ] **Step 2: Failing Rust tests**

`phase3_costlaw_backward_golden.rs`:
- `scalar_delta_polynomial_bit_exact`: replay the speech/other sweeps through `compute_unitary_delta`, `assert_bits_eq` vs `cost_deriv_scalar_*.bin` for the `square` config (strict - polynomial + `output*(1-output)` is pure arithmetic). Comparator: strict bits.
- `scalar_delta_log_sqrt_canary`: same sweeps for the `log` and `sqrt` configs, `assert_oracle_eq_f64` per element (canary-gated - `ln`/`sqrt`). Comparator: canary.
- `multiclass_deltas_fusion`: replay through `compute_deltas`, `assert_bits_eq` vs `cost_deltas_multiclass.bin` (the `output-1`/`output`/`0` fusion is pure arithmetic - strict). Comparator: strict bits.
- `multiclass_deltas_wer_and_pond`: the WER + ponderation variants vs `cost_deltas_wer.bin`. Comparator: strict bits (multiplies of pure arithmetic).
- `ignore_mask_zeroes_row` (structural): assert the masked row's deltas are exactly `0.0` (bit) - the ignore path (risk R8, masked-target width).
- `ponderation_plumbing` (structural): a config with `classes_ponderations` set produces a row scaled by the on-class ponderation; a config without produces the unscaled row; assert they differ by exactly the ponderation factor. Proves the class-balance coefficient enters the gradient (risk R4).

- [ ] **Step 3: Implement/validate -> PASS -> lint + commit**

If any probe reveals a divergence in the already-ported `cost.rs` backward, fix it against the probe (the probe is the arbiter) and log the reason. Otherwise the code is unchanged and only the goldens + tests land.
```bash
git commit -m "feat(phase3): CostLaw backward REAL-probe goldens (deriv/computeDeltas/WER/ponderation)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: NeuronLayer::feed_backward

**Files:**
- Modify: `src/rust/src/nn/layers.rs` (add `feed_backward` + deriv-accumulator fields to `NeuronLayer`)
- Modify: `tools/oracle_harness/main.cpp` (dense backward dumps - reuse the Task 1 `denseBackwardLoop`)
- Create: `src/rust/tests/phase3_neuron_backward_golden.rs`

**Interfaces:**
- Consumes: `matmul_seq` (`layers.rs:16`), `maxmin2_fn` (`activations`), `load_bin_phase3`, `assert_oracle_eq`, `assert_bits_eq`.
- Produces on `NeuronLayer`:
  - fields `weights_derivatives: Array2<f64>` (I x O), `biases_derivatives: Array2<f64>` (1 x O), `nb_of_seq_fed_backward: i64`.
  - `pub fn feed_backward(&mut self, input: &Array2<f64>, deltas: &Array2<f64>, inv_sub_sampling_ratio: usize, last_layer: bool) -> Array2<f64>` (returns `deltas_out`, T x I). Legacy `NeuronLayer::feedBackward` (`:151-207`): `weights_derivatives += input.col(jj)*deltas.row(jj)` per row (via `matmul_seq` on the row-slices), `biases_derivatives += deltas.colwise().sum()`, `inv_sub_sampling_ratio > 1` scaling, `_NbOfSeqFedBackward += input.nrows()` (NOT `deltas.nrows()`), `last_layer` -> `deltas*W'` (NO activation deriv), hidden -> `(deltas*W') .* Maxmin2'(input)`. Width tolerance (`:157-166`): `cols > I` -> `leftCols(I)`; `cols < I` -> ZERO-PAD to I. No retained cache; input passed in.
  - `pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>)` mirroring `NeuronLayer.cpp:101-114`: Nx2, col0 = deriv (column-major: weights then bias), col1 = `nb_of_seq_fed_backward` replicated. (The `Vec<[f64;2]>` accumulator threads through `Network`/`BlstmNetwork`; final assembly to `Array2` in Task 6.)
  - `pub fn reset_weights_derivatives(&mut self)` (`:116-120`): zero the two deriv matrices + count.

- [ ] **Step 1: Harness dense backward dumps**

Reuse `denseBackwardLoop` (Task 1). Construct a REAL `NeuronLayer` (synthetic small shapes: hidden `I=3,O=4`, last-layer `I=4,O=3`), run backward with a deterministic delta + input, dump: `neuron_bwd_hidden_deltasout.bin` (T x I), `neuron_bwd_hidden_derivs.bin` (Nx2), `neuron_bwd_last_deltasout.bin`, `neuron_bwd_last_derivs.bin`. The last-layer case dumps deltas whose FUSED form is distinguishable from a spuriously-applied softmax Jacobian (the double-count trap, spec S11.3): craft the incoming deltas as the pre-fused `output-target` values and assert the returned `deltas_out` is exactly `deltas*W'` (NO extra `output*(1-output)` factor). NN_TOL `site=dense_backward max_ulp=0` (synthetic, strict).

- [ ] **Step 2: Failing Rust tests**

`phase3_neuron_backward_golden.rs`:
- `hidden_backward`: `deltas_out` vs `neuron_bwd_hidden_deltasout.bin` (canary - the `Maxmin2'` asinh-deriv on input), derivs Nx2 vs `neuron_bwd_hidden_derivs.bin` (col0 canary via `assert_oracle_eq`, col1 strict-int-equal). Comparator: deltas_out + col0 canary; col1 structural exact.
- `last_backward_no_activation_deriv` (double-count trap, S11.3): `deltas_out` vs `neuron_bwd_last_deltasout.bin` and a HAND assertion that `deltas_out == deltas.dot(W')` with NO `output*(1-output)` fold - constructed so a wrongly-added softmax Jacobian would visibly differ. Comparator: strict bits (the last-layer `deltas*W'` is pure arithmetic at these shapes; use `assert_oracle_eq` if the harness records nonzero ULP, else strict).
- `count_is_input_rows` (structural): assert `nb_of_seq_fed_backward == input.nrows()` after one call (NOT `deltas.nrows()` - they differ under sub-sampling), and `== 2*input.nrows()` after two calls (accumulation). Comparator: structural exact.
- `reset_zeroes_all` (structural): after `reset_weights_derivatives`, both matrices are all-zero and count is 0.

- [ ] **Step 3: Implement -> PASS -> lint + commit**
```bash
git commit -m "feat(phase3): NeuronLayer feed_backward (no output-activation deriv; input-rows count)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: LSTMLayer::feed_backward (CRUX)

**Files:**
- Modify: `src/rust/src/nn/layers.rs` (add `feed_backward`/`feed_backward_reverse` + deriv fields to `LstmLayer`)
- Modify: `tools/oracle_harness/main.cpp` (LSTM backward dumps + NN_TOL, reuse Task 1 `lstmBackwardLoop`)
- Create: `src/rust/tests/phase3_lstm_backward_golden.rs`

**Interfaces:**
- Consumes: `matmul_seq`, `gates_fn`/`identity_fn`/`maxmin2_fn` and their DERIVATIVES (add `gates_deriv`/`maxmin2_deriv` to `nn/activations.rs`, transcribing `ActivationFunctions.h:162-165,211-214,240-242` EXACTLY - all three take the POST-activation value `y`: `GatesFunction::deriv(y) = 0.1*y*(1-y)` (the activated sigmoid output times the 0.1 pre-scale - a plain sigmoid-derivative-of-output, NOT `f'(f^{-1}(y))`); `Maxmin2::deriv(y)` == `Identity::deriv(y)` = `1/sqrt(1+sinh(y)*sinh(y))` where `y` is the asinh output (so `Identity'(_CellsIn)` recovers `1/sqrt(1+cellstate^2)` since `_CellsIn=asinh(cellstate)`; `Maxmin2'(input)` where `input` is the prev-layer asinh output). Cite the exact header lines), the `gates`/`cells_in`/`cell_states` caches (`layers.rs:317-329`).
- Produces on `LstmLayer`:
  - fields `input_weights_derivatives` (I x 4O), `feedback_weights_derivatives` (O x 4O), `peep_weight_derivatives` (12 x O), `biases_derivatives` (1 x 4O), `nb_of_seq_fed_backward: i64`.
  - `pub fn feed_backward(&mut self, input: &Array2<f64>, output: &Array2<f64>, deltas: &Array2<f64>, inv_sub_sampling_ratio: usize, last_layer: bool) -> Array2<f64>` (returns `deltas_previous_layer`, T x I). Full transcription of `LSTMLayer.cpp:518-723` - the reverse-time loop, gate-derivative order O -> state -> C -> F -> I, peephole cross-terms (12-row map), the `row==0` no-cellstate forget branch (`:648-657`), the `deltasForgetGatetmp` save/read (`:624`/`:660`), the `testM`/`test` 4O-stacking with `deltasPreviousLayer=(InputW*testM)'` and `deltasFeedback=(FeedbackW*test)'` via `matmul_seq`, width tolerance (`:534-543`), `inv_sub_sampling_ratio` block scaling (`:710-715`), accumulate + `nb_of_seq_fed_backward += deltas.nrows()` (`:720` - `linesNb = deltas.rows()`). READ the legacy lines, transcribe with provenance comments, the harness `lstmBackwardLoop` dumps are the arbiter.
  - `pub fn feed_backward_reverse(...)` (`:725-732`): reverse input/output/deltas columns, call `feed_backward`, reverse the returned deltas. The caches are consumed in their reversed-storage order - do NOT un-reverse (the `feed_forward_reverse` doc at `layers.rs:331-335` already pins this).
  - `pub fn get_weights_derivatives(&self, out: &mut Vec<[f64;2]>)` (`LSTMLayer.cpp:251-295`): Nx2, block order InputWeights -> FeedbackWeights -> PeepWeight -> Biaises, each column-major, col1 = count replicated.
  - `pub fn reset_weights_derivatives(&mut self)` (`:297-...`).

- [ ] **Step 1: Harness LSTM backward dumps + NN_TOL**

Reuse `lstmBackwardLoop` (Task 1). Construct a REAL `LSTMLayer` with all three peephole flags ON (synthetic `I=2,O=2`, T=5 incl. odd length) and a deterministic input/output/delta. Dump: `lstm_bwd_deltasprev_<variant>.bin` (T x I), `lstm_bwd_derivs_<variant>.bin` (Nx2), for variants: `peep_all` (all flags on), `peep_none` (all off), `reverse` (via `feed_backward_reverse`), `subsample` (`inv_sub_sampling_ratio=2`, asserts the `*2` block scaling). NN_TOL `site=lstm_backward max_ulp=0` for the synthetic shapes (strict). Width-tolerance variant: `signal_width` (1-col input into an I=2 layer -> zero-pad; the P2b signal driver's dead weight rows accumulate exactly 0 - spec S11.5) -> `lstm_bwd_signal_derivs.bin` with the dead-row-zero assertion.

- [ ] **Step 2: Failing Rust tests**

`phase3_lstm_backward_golden.rs`:
- `backward_peep_all` / `backward_peep_none`: `deltas_previous_layer` vs `lstm_bwd_deltasprev_<v>.bin` (canary - gate/cell derivs traverse asinh/sigmoid), derivs Nx2 vs `lstm_bwd_derivs_<v>.bin` (col0 canary, col1 strict-int). Comparator: canary + structural.
- `backward_reverse_caches_not_unreversed`: the `reverse` variant vs its dumps; a structural assertion that flipping the input WITHOUT flipping the caches would give a different result (proves the reversed-storage consumption, risk R9). Comparator: canary + structural.
- `subsample_scales_derivs`: the `subsample` variant - assert col0 equals the `peep_all` col0 times 2 (bit-exact ratio on the scaled block) and col1 unchanged. Comparator: strict bits on the ratio.
- `width_tolerance_dead_rows_zero` (S11.5): the `signal_width` variant - the input rows beyond the 1-col signal accumulate EXACTLY `0.0` (bit) in `input_weights_derivatives`; gradient shape stays layer-native (I x 4O). Comparator: strict bits (the zero rows are exact).
- `row_zero_forget_branch` (structural): a T=1 case exercises ONLY the `row==0` forget branch (no `_CellStates.row(row-1)` term); assert it does not read out-of-bounds and matches the dump. Comparator: canary.
- `count_is_deltas_rows` (structural): `nb_of_seq_fed_backward == deltas.nrows()` after one call, accumulates on two.

- [ ] **Step 3: Implement -> PASS -> lint + commit**
```bash
git commit -m "feat(phase3): LSTMLayer feed_backward (peephole BPTT; reverse wrapper; invSubSampling scaling)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Network container backward + feedBackwardDouble + the SubSample resolution (CRUX)

**Files:**
- Modify: `src/rust/src/nn/network.rs` (add `feed_backward`/`feed_backward_reverse`/`feed_backward_double` + `get_weights_derivatives`/`reset_weights_derivatives` + `inv_sub_sample` + the `Layer` trait backward methods)
- Modify: `tools/oracle_harness/main.cpp` (container backward dumps, reuse Task 1 `netBackwardLoop`)
- Create: `src/rust/tests/phase3_network_backward_golden.rs`

**Interfaces:**
- Consumes: `Layer` trait (extend), `sub_sample` (`network.rs:319`), the `layers_output` cache (`network.rs:97`), the layer `feed_backward`/`get_weights_derivatives`/`reset_weights_derivatives` (Tasks 3-4).
- Produces:
  - extend `trait Layer`: `fn feed_backward(&mut self, input, output, deltas, inv_sub_sampling_ratio, last_layer) -> Array2<f64>`, `fn feed_backward_reverse(...) -> Array2<f64>`, `fn get_weights_derivatives(&self, out: &mut Vec<[f64;2]>)`, `fn reset_weights_derivatives(&mut self)`. `NeuronLayer::feed_backward_reverse` panics (never run reversed, mirroring the forward `network.rs:65-72`).
  - `pub fn inv_sub_sample(ratio: usize, input: &Array2<f64>) -> Array2<f64>` (`NeuralNetwork.hpp:136-145`): `T x C -> T*ratio x C/ratio`, output row `jj*ratio+kk` = `input.block(jj, kk*(C/ratio), 1, C/ratio)` - the inverse of `sub_sample`'s column-block stacking.
  - `Network::feed_backward(&mut self, input, output_seq, deltas) -> Array2<f64>` and `feed_backward_reverse(...)`: the reverse-order layer loop `NeuralNetwork.hpp:251-300`/`:302-351` verbatim - single-layer vs multi-layer, the running `inv_sub_sampling_ratio` starting at 1, the `SubSample(...).topRows(delta_rows)` reconstruction (`.topRows(deltas.rows())` for the last/first-sub layer, `.topRows(deltas_out.rows())` for interior - the two bounds differ), the `InvSubSample` of the returned `deltas_out` when `_SubSampling[jj] > 1`, and `inv_sub_sampling_ratio *= _SubSampling[jj]` AFTER. Layer 0's `deltas_out` (InvSubSampled) is returned. Transcribe faithfully - the SubSample-backward resolution in Global Constraints is the semantics; the harness `netBackwardLoop` dumps are the arbiter.
  - `Network::feed_backward_double(&mut self, first, second, output_seq, deltas) -> Array2<f64>` (`NeuralNetwork.hpp:353-362`): hcat `first|second` (first on LEFT), run `feed_backward`, return `deltas_out`. Empty `first` -> empty.
  - `Network::get_weights_derivatives(&self, out: &mut Vec<[f64;2]>)` (`NeuralNetwork.hpp:98-111`): concat each layer's Nx2 (layer-0-first, `jj==0` seeds). `reset_weights_derivatives` (`:113-117`).

- [ ] **Step 1: Harness container backward dumps**

Reuse `netBackwardLoop`/`netBackwardReverseLoop` (Task 1). Two synthetic nets: (a) multi-layer LSTM stack `[2,3,2]` sub `[1,1]` (no decimation - exercises the plain interior loop); (b) LSTM stack `[2,2]` sub `[2]` (single-layer, `_SubSampling[0]=2 > 1` - exercises the SubSample/InvSubSample inversion + `inv_sub_sampling_ratio` running). Dump `net_bwd_<variant>_deltasout.bin` and `net_bwd_<variant>_derivs.bin` for forward + reverse drivers. A `double` variant exercises `feed_backward_double` (dense output net `[4,3]`, first|second each 2-wide). NN_TOL `site=net_lstm_backward_{fwd,rev}|net_dense_backward|net_double max_ulp=0` synthetic.

- [ ] **Step 2: Failing Rust tests**

`phase3_network_backward_golden.rs`:
- `multilayer_no_subsample`: `deltas_out` + Nx2 derivs vs the `[2,3,2]` dumps (canary + structural col1). Comparator: canary + structural.
- `single_layer_subsample_inversion` (SubSample crux): the `[2,2]` sub `[2]` net - `deltas_out` vs the dump AND a structural assertion that its row count is `input.nrows()` (InvSubSampled back to full temporal resolution, not the decimated `input.nrows()/2`); assert the layer's col0 is scaled by the running ratio. Comparator: canary + structural row-count.
- `feed_backward_double_split`: `feed_backward_double` matches the manual hcat + `feed_backward` (the forward-LEFT split). Comparator: strict bits (composition identity).
- `reverse_driver`: the reverse variant vs its dumps. Comparator: canary.
- `get_derivatives_layout` (structural): the concatenated Nx2 has the exact row count `nb_of_weights()` and layer-0-first ordering (assert a known-index element maps to layer 0's first weight). Comparator: structural exact.

- [ ] **Step 3: Implement -> PASS -> lint + commit**
```bash
git commit -m "feat(phase3): Network container backward incl. SubSample/InvSubSample inversion + feedBackwardDouble

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: BLSTM windowed backward + get_weights_derivatives (Nx2) + update_weights normalization

**Files:**
- Modify: `src/rust/src/nn/blstm.rs` (add `feed_backward`, wire the backward into `feed_forward_backward_plain`/`_mlp` + the windowed drivers, `get_weights_derivatives`, real `reset_weights_derivatives`, `update_weights`)
- Modify: `tools/oracle_harness/main.cpp` (BLSTM backward dumps, reuse Task 1 `blstmFeedBackwardLoop`)
- Create: `src/rust/tests/phase3_blstm_backward_golden.rs`

**Interfaces:**
- Consumes: Tasks 2-5 products, `CostLaw::compute_deltas`/`compute_unitary_delta` (Task 2), the existing windowed forward drivers + `output_forward`/`output_backward` (`blstm.rs:225-226`), `Rprop` (Task 7 - `update_weights` depends on it; sequence Task 7 before this task's `update_weights` step OR stub the trainer call and complete in Task 7's wake - see ordering note).
- Produces on `BlstmNetwork`:
  - `fn feed_backward(&mut self, input: &Array2<f64>, output_seq: &Array2<f64>, target_seq: &Array2<f64>)` (`BLSTMNeuralNetwork.cpp:439-460`): seed `deltas = Zero(target.rows, target.cols)`, `cost_law.compute_deltas(output, target, deltas)` (scalar path when `target.ncols()==1` uses `compute_unitary_delta` per row - mirror the `compute_cost` dispatch at `blstm.rs:839-858`), `output_network.feed_backward_double(output_forward, output_backward, output, deltas)` returns split deltas, then (unless `back_prop_output_network_only`) `forward_network.feed_backward(input, output_forward, deltas.leftCols(cols/2))` + `backward_network.feed_backward_reverse(input, output_backward, deltas.rightCols(cols/2))` with the `leftCols(inputSize)` crop gate (`:452-454`).
  - WIRE the backward into the drivers (replace the "Backward is Phase 3" gates): `feed_forward_backward_plain` (`blstm.rs:771-817`) - after forward, when `back_propagation_activated && target.nrows()>0`, apply the `_TargetEnforcementStep < 0` interior-row rewrite to the TARGET (`:803-808`), call `feed_backward`, THEN the existing cost path (which does its own separate enforcement rewrite of target AND output at `:816-822`, risk R7 - already ported at `blstm.rs:788-808`, keep it). `feed_forward_backward_mlp` similarly (no enforcement branch, `:832-841`). The windowed drivers (Truncate/TwoSweeps/OverLap/MLPOverLap) call `feed_forward_backward_plain`/`_mlp` per window (already the structure at `blstm.rs`), so the backward now runs INSIDE every window automatically - derivs accumulate across windows/sweeps, counts double for TwoSweeps and multiply per covering for OverLap (the `col1` count carries the compensation; derivs NOT halved - spec S6).
  - `pub fn get_weights_derivatives(&self) -> Array2<f64>` (`:255-276`): gated on `back_propagation_activated` (empty otherwise); hcat fwd|bwd|output Nx2 (from Tasks 4-5's `Vec<[f64;2]>` harvest) + the mean/std tail `[Zero | Ones | Zero | Ones]` (count 1); `output_network.sub_sampling_ratio() > 1` -> multiply fwd/bwd col0 by it (`:266-269`). Layout matches the P0a packer.
  - `pub fn reset_weights_derivatives(&mut self)` - REPLACE the `input_statistics`-only stub (`blstm.rs:423`) with the real body (`:278-287`): reset the three sub-networks' deriv accumulators + `input_statistics`.
  - `pub fn update_weights(&mut self, weights_derivatives: &Array2<f64>, cost: f64)` (`:303-310`): `weights_derivatives.size() > 0` guard; `get_weights()`; `norm = col0 cwiseQuotient col1` (element-wise); `trainer.update_weights(&norm, &mut weights, cost)`; `set_weights(&weights)`. The `Rprop` trainer is a LONG-LIVED per-network member (add `trainer: Rprop` field, constructed from `_BackPropagationRpropInit` default 1e-2 - Task 7).

- [ ] **Step 1: Harness BLSTM backward dumps**

Reuse `blstmFeedBackwardLoop` (Task 1). Construct a REAL small BLSTM (synthetic, LSTM `[2,2]` sub `[1]`, output `[4,2]` sub `[1]`, `BackPropagationActivated true`, a polynomial cost law so `computeDeltas` is bit-portable) with deterministic weights, run `feed_forward_backward` (plain + overlap + truncate/two-sweeps variants) with a deterministic target, harvest `getWeightsDerivatives()`. Dump `blstm_bwd_<variant>_derivs.bin` (Nx2) per variant. The mean/std tail assertion: dump the last `2*inputSize` rows and assert `[0,1,0,1]` structure (deriv 0, count 1). NN_TOL `site=blstm_feedbackward|blstm_real_backward` (synthetic 0; real-net nonzero calibration). The count-vector goldens (spec S11.4): for TwoSweeps and OverLap, dump col1 and record the per-region expected counts in the manifest (frames x sweeps; per-window coverage) for the hand-computed test.

- [ ] **Step 2: Failing Rust tests**

`phase3_blstm_backward_golden.rs`:
- `plain_backward_derivs`: `get_weights_derivatives()` col0 vs `blstm_bwd_plain_derivs.bin` (canary), col1 strict-int. Comparator: canary + structural.
- `overlap_count_vector` / `twosweeps_count_vector` (S11.4): assert col1 per region by HAND-COMPUTED expectation (from the manifest counts) - frames x sweeps for TwoSweeps, per-window coverage for OverLap, mean/std tail == 1. Comparator: structural exact (integer counts).
- `meanstd_tail_structure` (S11.4): the last `2*inputSize` rows are exactly `[0,1,0,1]` (bit). Comparator: strict bits.
- `layout_matches_packer` (S11.6, risk R1): a pack/unpack/repack property test - the Nx2 col0 ordering matches the P0a flat weight packer element-for-element (harvest derivs, treat col0 as a flat weight vector, round-trip through the packer, assert byte-identical ordering). Comparator: structural exact.
- `gradient_non_trivial` (S11.2): the real-net gradient fixture (Task 8 provides the real net; here on the synthetic net) is non-zero and non-saturated - assert `min < max`, `nonzero_count > 0`, no `NaN`/`Inf`. Comparator: structural.
- `output_only_shortcircuit` (structural): `back_prop_output_network_only true` -> the LSTM deriv blocks stay zero after backward; only the output-net block is nonzero. Comparator: structural.
- `reset_zeroes_subnetworks` (structural): after `reset_weights_derivatives`, all three sub-networks' Nx2 col0 are zero and counts 0.

- [ ] **Step 3: Implement -> PASS -> lint + commit**

Ordering note: `update_weights` calls `Rprop` (Task 7). Land `feed_backward`/`get_weights_derivatives`/`reset` + the driver wiring in this commit; add `update_weights` + the `trainer` field in Task 7's wake (or land Task 7 first if the executor prefers - the two are independent except for `update_weights`). If `update_weights` is deferred, its test moves to Task 7.
```bash
git commit -m "feat(phase3): BlstmNetwork feed_backward + windowed backward + Nx2 get_weights_derivatives

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Rprop iRPROP- trainer (REAL compiled golden, strict bits)

**Files:**
- Modify: `src/rust/src/nn/train.rs` (replace the bare stub with the `Rprop` struct + `update_weights`)
- Modify: `src/rust/src/nn/blstm.rs` (add the `trainer: Rprop` field + `update_weights`, if deferred from Task 6)
- Modify: `tools/oracle_harness/main.cpp` (Rprop trajectory dumps - the REAL compiled `Rprop::updateWeights`)
- Create: `src/rust/tests/phase3_rprop_golden.rs`

**Interfaces:**
- Consumes: `assert_bits_eq` (strict everywhere - Rprop is pure scalar, no libm/GEMM), `load_bin_phase3`.
- Produces `pub struct Rprop`:
  - fields `deltas: Vec<f64>`, `prev_derivs: Vec<f64>`, `delta_weights: Vec<f64>`, `prev_cost: f64`, `eta_min: f64` (0.5), `eta_plus: f64` (1.2), `min_delta: f64` (1e-9), `max_delta: f64` (0.2), `init_delta: f64`. `pub fn new(init_delta: f64) -> Rprop` (ctor defaults `Rprop.cpp:6`; `prev_cost` UNINITIALIZED in legacy but never read before it is written - init to `0.0` and document it is unused on the first call).
  - `pub fn update_weights(&mut self, weights_derivatives: &[f64], weights: &mut [f64], cost: f64)` - the exact per-element `Rprop.cpp:9-77` with the INVERTED SIGN CONVENTION (`deriv > 0 -> dw = -delta`; `deriv < 0 -> dw = +delta`; `deriv == 0 -> dw = 0`; `weights += dw`): first call (`deltas.is_empty()`) inits `deltas = const(init_delta)`, `prev_derivs = derivs.to_vec()`, `delta_weights = 0`, applies dw per sign; subsequent computes `derivTimesPrev = derivs .* prev_derivs`, `prev_derivs = derivs`, then per element `>0` -> `delta *= 1.2` clamp `max`, recompute dw, apply; `<0` -> `delta *= 0.5` clamp `min`, COST-GATED BACKTRACK (`if prev_cost < cost { weights[j] -= delta_weights[j] }`), `prev_derivs[j] = 0`, NO dw applied; `==0` -> recompute dw, apply. `prev_cost = cost` at end. Copy verbatim - inverted sign + cost gate are CLAUDE.md hazards. No `Trainer` trait (YAGNI - a plain struct until a second trainer exists; the Python SMORMS3 is a DIFFERENT, Python-side optimizer).
- Wire `update_weights` on `BlstmNetwork` (if deferred from Task 6): the `trainer: Rprop` field (init from `_BackPropagationRpropInit` default 1e-2), the `col0/col1` normalize + trainer call + `set_weights`.

- [ ] **Step 1: Harness Rprop trajectory dumps (multi-step, ALL branches)**

Construct the REAL compiled `Rprop` (`new(1e-2)`). Design a MULTI-STEP deterministic trajectory (>= 5 steps) over a small weight vector that PROVABLY exercises EVERY branch (spec S11.1): first-call init; eta+ growth (same-sign deriv); eta- shrink (sign flip); the `max_delta`/`min_delta` clamps (drive a delta past each bound); the cost-gated backtrack FIRING (feed a `cost` GREATER than `prev_cost` on a sign-flip element) and NOT firing (feed `cost` LESS); the post-backtrack `prev_derivs==0` path (the step after a backtrack). Dump, per step: `rprop_step<k>_weights.bin`, `rprop_step<k>_deltas.bin`, `rprop_step<k>_deltaweights.bin`, `rprop_step<k>_prevderivs.bin` (harvested from the REAL trainer via a probe subclass exposing the protected members). Record in the manifest, per step, WHICH branch fired per element (derivable from the deriv-sign x prev-deriv-sign x cost trajectory) so the test asserts it structurally. NO NN_TOL / no canary - Rprop is bit-portable; STRICT BITS.

- [ ] **Step 2: Failing Rust tests**

`phase3_rprop_golden.rs`:
- `trajectory_bit_exact`: replay the EXACT same trajectory through the Rust `Rprop`, `assert_bits_eq` on `weights`/`deltas`/`delta_weights`/`prev_derivs` at EVERY step vs the dumps. Comparator: STRICT BITS (every platform).
- `all_branches_fired` (S11.1 non-vacuity): assert STRUCTURALLY, per step, which branch fired per element (from the manifest branch record) - eta+ growth, eta- shrink, both clamps, backtrack FIRING, backtrack NOT firing, post-backtrack `prev_derivs==0`, first-call init. Not "hope they fired" - assert the branch record covers all seven. Comparator: structural exact.
- `inverted_sign` (structural): a single positive-deriv element with `deriv > 0` moves the weight DOWN (`dw < 0`); a negative-deriv element moves it UP. Comparator: strict bits on the sign.
- `stateful_across_calls` (S11.8): the trainer state (`deltas`/`prev_derivs`/`delta_weights`/`prev_cost`) persists across >= 3 calls - a fresh trainer per call would give different weights; assert the 3-call trajectory differs from three independent first-calls. Comparator: strict bits (discriminating trajectory).
- `bp_update_weights_normalizes` (if `update_weights` landed here): a BLSTM `update_weights` with a known Nx2 (distinct col1 per region) applies `col0/col1` element-wise (NOT a scalar), then the trainer step; assert the mean/std tail (count 1) and an LSTM region (count > 1) move by the correct per-element normalized gradient. Comparator: strict bits.

- [ ] **Step 3: Implement -> PASS -> lint + commit**
```bash
git commit -m "feat(phase3): Rprop iRPROP- trainer (inverted sign; cost-gated backtrack; stateful) + BLSTM update_weights

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Network-level grad check + real-net E2E gradient gate (phase acceptance crux)

**Files:**
- Create: `src/rust/tests/phase3_gradcheck.rs` (central-difference validation)
- Create: `src/rust/tests/phase3_e2e_grad_gate.rs` (real net + real config + e2e input)
- Modify: `tools/oracle_harness/main.cpp` (real-net Nx2 gradient dump + one-iRPROP-step dump)
- Modify: `scripts/extract_phase3_fixtures.py`

**Interfaces:**
- Consumes: `BlstmNetwork::feed_forward_backward`/`feed_forward_scoring`/`get_weights_derivatives`/`update_weights`/`set_weights`/`get_weights` (Tasks 6-7), the real `1_worker_1.config` + `NNweights_config1.bin` + `e2e_input.bin`, `assert_oracle_eq`/`assert_oracle_eq_f64`.
- Produces:
  - Grad check (spec S8): for a fixture (synthetic net + input + targets), perturb each selected flat weight `+eps`/`-eps` with full state restore between runs (`set_weights` + fresh net), cost from ONE `feed_forward_backward`, numerical grad = `(costPlus - costMinus)/(2*eps)`, compared vs analytic `col0/col1` under a DOCUMENTED relative-error bound (ref floored at 1e-24, per `CorpusProcessor:237-340`). A spot-checked weight SUBSET (full sweep is O(N) forwards). This is a VALIDATION harness, not a byte-golden - the built-in non-vacuity layer for the analytic chain.
  - E2E gate: the real 33,671-weight net under `1_worker_1.config` (with `BackPropagationActivated` OVERRIDDEN to `true` - the config has it `false`), `InputNormalizationType -1`, e2e input width 11 (< 23 -> the LSTM `topRows` width tolerance), a synthetic-but-deterministic target (or a scoring-derived target via `feed_forward_scoring`): `feed_forward_backward` -> `get_weights_derivatives()` (Nx2) compared BIT-EXACT (canary-gated) vs the harness reimpl Nx2 dump; then ONE `update_weights` step compared bit-exact (STRICT - Rprop is portable) vs the REAL `Rprop::updateWeights` on the same normalized gradient, INCLUDING a fired-backtrack case (a second step with a cost rise).

- [ ] **Step 1: Harness real-net gradient + iRPROP-step dumps**

Extend the Task 1/6 real-net stage: run the REAL config's net through the reimpl `blstmFeedBackwardLoop` (backprop active), harvest the reimpl Nx2 -> `e2e_grad_nx2.bin`. Separately, apply ONE REAL `Rprop::updateWeights` to the normalized `col0/col1` gradient -> `e2e_weights_after_step.bin`; a SECOND step with a deliberately-raised `cost` (backtrack fires) -> `e2e_weights_after_backtrack.bin`. NN_TOL `site=blstm_real_backward` records the reimpl-vs-real-Eigen delta (Phase 4 calibration). The grad-check needs NO new fixture beyond the nets/inputs already dumped (bound-based, S10).

- [ ] **Step 2: Failing Rust tests**

`phase3_gradcheck.rs`:
- `central_difference_matches_analytic` (S8 non-vacuity): on a synthetic net + target with NON-TRIVIAL gradients (backprop active, cost varying - assert the analytic grads are not all zero first), spot-check a weight subset; numerical vs analytic relative error under the documented bound. Comparator: relative-error bound (canary-gated, documented threshold), NOT a byte-golden.
- `gradcheck_state_restore` (structural): assert the net's weights are IDENTICAL before and after each perturbed run (full restore). Comparator: strict bits.

`phase3_e2e_grad_gate.rs`:
- `real_net_gradient_bit_exact`: the real-net `get_weights_derivatives()` Nx2 col0 vs `e2e_grad_nx2.bin` (canary - the k=23 GEMM diverges, the golden IS the reimpl), col1 strict-int. Comparator: canary + structural.
- `real_net_one_irprop_step`: one `update_weights` -> weights vs `e2e_weights_after_step.bin`. Comparator: STRICT BITS (Rprop portable; the normalized gradient is the canary-gated input, but the trainer step on a FIXED gradient is bit-portable - if the input gradient itself is canary-only, gate the whole assert; document the choice).
- `real_net_backtrack_fires`: the second step (cost rise) -> weights vs `e2e_weights_after_backtrack.bin`, and a structural assertion that the backtrack branch fired (weights moved back on the sign-flip elements). Comparator: canary (downstream of the canary-gated gradient) + structural branch assertion.
- `gradient_non_trivial_real` (S11.2): the real-net Nx2 col0 is non-zero, non-saturated, no NaN/Inf. Comparator: structural.

- [ ] **Step 3: Implement -> PASS -> lint + commit**
```bash
git commit -m "feat(phase3): network grad check + real-net E2E gradient gate (Nx2 + iRPROP- step + backtrack)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Python backward oracle (nn_reference.py, independent arbiter)

**Files:**
- Modify: `src/python/speech/nn_reference.py` (add `lstm_backward_oracle`, `dense_backward_oracle`, `blstm_backward_oracle`)
- Create: `tests/test_phase3_nn_reference.py`

**Interfaces:**
- Consumes: the existing `_gates_fn`/`_asinh`/`_logistic_fn`/`_matmul_seq_row` helpers (`nn_reference.py:34-72`), the forward oracles that RETURN `(y, gates, cells)` (`lstm_forward_oracle:75`, thread the caches through - the net-level `_lstm_net_forward` currently discards them at `:321,324`), the Task 1 `extract_phase3_oracle_cases.py` JSON cases (harness dumps: weights + inputs + targets + expected Nx2 derivs), the Rust layers (cross-check via the JSON cases the Rust tests also consume), `tests/_libm_gate.py` canary.
- Produces: `lstm_backward_oracle(...)`, `dense_backward_oracle(...)`, `blstm_backward_oracle(...)` with ENGINE semantics (sigmoid(0.1z) gates, asinh, the softmax+CE fusion in the cost seam - NOT `BLSTM_Backward.m`, which pairs with the divergent MATLAB forward and MUST NOT be ported, per the P2b T11 verdict). Plain ascending Python loops only (numpy pairwise summation drifts). Add `deriv` helpers `_gates_deriv(y)=0.1*y*(1-y)` and `_asinh_deriv(y)=1/sqrt(1+sinh(y)^2)` (from `ActivationFunctions.h:162-165,240-242`, taking the post-activation cached value `y` as the engine does).

- [ ] **Step 1: Thread caches + write the backward oracles**

Make `_lstm_net_forward` retain `(y, gates, cells)` per layer (a `caches` list) for the backward. Write the three backward oracles transcribing the ENGINE backward (same source lines as the Rust: `LSTMLayer.cpp:518-723`, `NeuronLayer.cpp:151-207`, `BLSTMNeuralNetwork.cpp:439-460`) as scalar loops. Provenance docstrings citing the legacy lines + the P2b T11 no-port verdict for `BLSTM_Backward.m`.

- [ ] **Step 2: Failing tests**

`tests/test_phase3_nn_reference.py`:
- `lstm_backward_matches_harness`: the Python `lstm_backward_oracle` on the Task 1 JSON cases (synthetic shapes) - Nx2 derivs + deltas_out bit-for-bit vs the harness dump (canary-gated via `tests/_libm_gate.py` - gates/cells traverse asinh/sigmoid). Comparator: canary.
- `dense_backward_matches_harness`: same for `dense_backward_oracle` (incl. the no-output-activation-deriv contract). Comparator: canary (hidden asinh-deriv) / strict (last-layer pure arithmetic).
- `blstm_backward_matches_harness`: `blstm_backward_oracle` end-to-end on a synthetic BLSTM JSON case vs the harness `blstm_bwd_*_derivs.bin`. Comparator: canary.
- `triangulation_vs_rust` (independent arbiter): assert the Python oracle Nx2 == the Rust-produced Nx2 on the SAME synthetic case (the JSON case both consume) - the third-codebase cross-check (as done for the P2 forward). Comparator: canary.
- `not_blstm_backward_m` (doc/structural): a comment-anchored assertion that the gate activation used is `sigmoid(0.1z)` not `sigmoid(z)` (probe `_gates_fn` at a known input), pinning the engine-not-MATLAB semantics.

- [ ] **Step 3: Implement -> PASS -> lint + commit**
```bash
git commit -m "feat(phase3): Python backward oracle (engine semantics, triangulated vs harness + Rust)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Docs, IMPROVEMENTS consolidation, full verification, smart-commit

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `IMPROVEMENTS.md`, `src/rust/src/nn/mod.rs` (doc refresh if present)

- [ ] **Step 1: IMPROVEMENTS.md** - new `[phase3]` entries per spec S12: the inverted iRPROP- sign convention; the cost-gated backtrack + `prev_derivs` zeroing; the TwoSweeps deriv double-count (compensated only via col1); the OverLap deriv-vs-output normalization asymmetry (derivs NOT divided by coverage, only col1 counts); the enforcement-step OUTPUT mutation (`:816-822`, risk R7); the logistic-deriv fold in `computeUnitaryDeltas` (`:344`); the BackPropWER 10x asymmetric scaling; the count-1 mean/std tail; the SubSample-backward `InvSubSample` + running-`invSubSamplingRatio` inversion (the S6 resolution - decimated rows are re-expanded, not dropped); the `Rprop` uninitialized-`_PrevCost`-on-first-call subtlety (never read before written).
- [ ] **Step 2: CLAUDE.md** - rows: `nn/train.rs` -> Implemented (Phase 3) with the iRPROP- description; `nn/layers.rs` (+`feed_backward` for both layers); `nn/network.rs` (+container backward + SubSample inversion); `nn/blstm.rs` (+`feed_backward`/windowed backward/`get_weights_derivatives`/`update_weights`); `cost.rs` (note the backward seam is golden-validated against REAL probes); `nn_reference.py` (+backward oracles). Update the Phase-status prose ("training ... typed stubs" -> Phase 3 lands the network-level training chain; the corpus epoch loop / PyO3 seam remain Phase 4). Harness note: the backward reimpl family + NN_TOL backward probes + REAL Rprop/CostLaw probes.
- [ ] **Step 3: README roadmap** - a Phase 3 done paragraph: the four-tier oracle (REAL Rprop/CostLaw probes, harness reimpl for layer/BLSTM backward with NN_TOL, Python triangulation), the SubSample-backward resolution, the real-net E2E gradient gate (Nx2 bit-exact + iRPROP- step + fired backtrack), what is real-compiled vs reimpl-dumped vs Rust-only. Note the deferred Phase 4 handoff (corpus reduction determinism, R6).
- [ ] **Step 4: Full verification** - `./check_all.sh && ./lint_code.sh && uv run pytest tests -v`; both comparator modes (strict + canary env); ASCII sweep; regenerate-fixtures-twice final check; confirm `phase1/phase2/phase2b` fixture hashes unchanged.
```bash
git add -A   # verify scope first
git commit -m "docs(phase3): mark training implemented; IMPROVEMENTS consolidation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
- [ ] **Step 5: smart-commit** - invoke the `smart-commit` skill, telling it to take the WHOLE git branch into account; then the final whole-branch review + `finishing-a-development-branch`.

---

## Self-Review

**1. Spec coverage (S1-S13 -> task mapping):**

- **S1 (scope IN)**: CostLaw backward -> T2; NeuronLayer backward -> T3; LSTMLayer backward -> T4; Network container backward + SubSample -> T5; BLSTM windowed backward + Nx2 + reset + update_weights -> T6; Rprop -> T7; grad check -> T8; Python oracle -> T9. **S1 (scope OUT/DEAD)**: corpus epoch loop, `saveAndUpdate`, corpus grad-check column indexing, CLI modes, PyO3 seam, SMORMS3 -> explicitly Phase 4 (Global Constraints + T10 handoff); dead reverse bodies / `_MaxSaturation` / Rprop .mat dump / SRN-CWRNN-Conv backward -> "DEAD - do NOT port" callouts in T1/T4.
- **S2 (four oracle tiers)**: tier 1 (REAL Rprop) -> T7; tier 2 (REAL CostLaw) -> T2; tier 3 (reimpl + NN_TOL) -> T1 foundation + T3-T6 consumers; tier 4 (Python arbiter) -> T9. **S2 NaN policy** -> Global Constraints NaN clause.
- **S3 (Nx2 layout)** -> the dedicated Global Constraints block + T6 `get_weights_derivatives` + T6 `layout_matches_packer` (S11.6) + T4/T5 `get_weights_derivatives`.
- **S4 (CostLaw seam)** -> T2. **Deviation**: the deriv chain is ALREADY ported in `cost.rs` (verified against `CostLaw.h`/`.cpp`); T2 is REAL-probe VALIDATION + BLSTM wiring, not a from-scratch add. Called out in "What is ALREADY ported" + here.
- **S5 (layer backward)** -> T3 (Neuron) + T4 (LSTM). The double-count trap (no output-activation deriv) -> T3 `last_backward_no_activation_deriv`; width tolerance -> T4 `width_tolerance_dead_rows_zero`; reverse caches -> T4 `backward_reverse_caches_not_unreversed`.
- **S6 (container + BLSTM backward + SubSample crux)** -> T5 (container + `inv_sub_sample` + the resolution, now CONCRETE with citations, replacing the spec's transcribe-placeholder) + T6 (BLSTM fan-out, feedBackwardDouble, windowed backprop, enforcement rewrite, update_weights).
- **S7 (iRPROP-)** -> T7, verbatim per-element semantics + no-Trainer-trait YAGNI.
- **S8 (grad check)** -> T8 `phase3_gradcheck.rs`, bound-based.
- **S9 (Python oracle)** -> T9, engine semantics + no `BLSTM_Backward.m`.
- **S10 (fixtures/harness)** -> T1 (reimpl + extractors + hash guard + regenerate-twice + measured manifest); harness local-only in Global Constraints.
- **S11 (non-vacuity, 1-9)**: (1) iRPROP- all-branches -> T7 `all_branches_fired`; (2) gradient non-triviality -> T6 `gradient_non_trivial` + T8 `gradient_non_trivial_real`; (3) double-count trap -> T3 `last_backward_no_activation_deriv`; (4) count vector -> T6 `overlap/twosweeps_count_vector`; (5) width tolerance -> T4 `width_tolerance_dead_rows_zero`; (6) layout agreement -> T6 `layout_matches_packer`; (7) determinism note -> T10 handoff (R6); (8) statefulness -> T7 `stateful_across_calls`; (9) comparator discipline -> per-test comparator declarations throughout + Global Constraints.
- **S12 (IMPROVEMENTS candidates)** -> T10 Step 1 (the full list, incl. the SubSample-backward resolution).
- **S13 (risks R1-R9)**: R1 (per-element norm) -> S3 block + T6 `layout_matches_packer`/`update_weights`; R2 (stateful backtrack) -> T7 `stateful_across_calls`; R3 (fusion double-count) -> T3; R4 (ponderations in gradient) -> T2 `ponderation_plumbing`; R5 (grad-check column indexing) -> deferred to Phase 4 (Global Constraints + S1 OUT); R6 (cross-file reduction determinism) -> T10 handoff; R7 (enforcement output mutation) -> T6 wiring; R8 (masked-target width) -> T2 `ignore_mask_zeroes_row`; R9 (reverse caches) -> T4 `backward_reverse_caches_not_unreversed`.

Gap check: none found. Every S-section maps to at least one task; every S11 non-vacuity obligation maps to a named test with a declared comparator.

**2. Placeholder scan:** No TBD/TODO. T2 carries a conditional ("if a probe reveals a divergence in the already-ported code, fix against the probe") - deliberate, both arms specified (the code is verified-present, the probe is the arbiter). T6 carries an ordering note for `update_weights`-depends-on-Rprop with both sequencings specified - deliberate, not a placeholder. The SubSample-backward semantics are CONCRETE (Global Constraints resolution + T5 interface), replacing the spec's "transcribe, do not invent" placeholder with the actual `InvSubSample` + running-ratio inversion and line citations. Config literals: the golden configs reuse `1_worker_1.config` with documented `set_val` overrides (`BackPropagationActivated -> true`); synthetic net shapes are given inline per task.

**3. Type consistency (verified against the real Rust sources):**
- `matmul_seq` (`layers.rs:16`), `matSeq` (`main.cpp:216`), `_matmul_seq_row` (`nn_reference.py:61`) - the ascending-loop contract, cited by real names.
- `assert_bits_eq`/`assert_oracle_eq`/`assert_oracle_eq_f64`/`oracle_mode`/`assert_vrcts_eq` - real `common/mod.rs` names; `fixture_phase3`/`load_bin_phase3` added in T1 mirroring `fixture_phase2b`/`load_bin_phase2b:40-50`.
- `Layer` trait method names (`feed_forward`/`feed_forward_reverse`/`set_weights`/`get_weights`/`nb_of_weights`, `network.rs:23-34`) - T5 extends with `feed_backward`/`feed_backward_reverse`/`get_weights_derivatives`/`reset_weights_derivatives`, consistent signatures.
- `BlstmNetwork` field names (`output_forward`/`output_backward`/`cost`/`nb_of_classif`, `blstm.rs:225-229`; `forward_network`/`backward_network`/`output_network`; `cfg.back_propagation_activated`/`back_prop_output_network_only`/`target_enforcement_step`, `blstm.rs:129-132`) - consumed by name in T6.
- `CostLaw::compute_unitary_delta`/`compute_deltas`/`classes_ponderations`/`from_config` (`cost.rs:290,385,331,207`) - the ALREADY-PORTED backward, consumed by T2/T6 with real names.
- `feed_backward(&mut self, input, output/output_seq, deltas, inv_sub_sampling_ratio: usize, last_layer: bool) -> Array2<f64>` - IDENTICAL signature shape on `LstmLayer` (T4), `NeuronLayer` (T3, with `deltas` in place of `(output, deltas)` since NeuronLayer takes only input+deltas per `NeuronLayer.cpp:151`), and the `Layer` trait (T5); the `Vec<[f64;2]>` deriv accumulator threads T3->T4->T5->T6 to the final `Array2<f64>` in `BlstmNetwork::get_weights_derivatives`. Note: `NeuronLayer::feed_backward` omits the `output` arg (legacy passes only `InputSeq`+`deltas`); `LstmLayer::feed_backward` takes `output` (the `outputSeq` for the `output.col(row-1)` recurrent-deriv reads) - this asymmetry matches the legacy signatures and is intentional.
- `Rprop::new(init_delta) -> Rprop` + `update_weights(&[f64], &mut [f64], f64)` (T7) consumed by `BlstmNetwork::update_weights` (T6/T7); `Vec<f64>` state fields, no `Trainer` trait (YAGNI).
- Fixture names consistent between producing (harness/extractor) and consuming (Rust/Python test) steps: `cost_deriv_scalar_*`, `neuron_bwd_*`, `lstm_bwd_*`, `net_bwd_*`, `blstm_bwd_*`, `rprop_step*`, `e2e_grad_nx2`/`e2e_weights_after_*`.

10 tasks; ~P2-sized each; every task reviewer-gateable (failing-test-first, declared comparators, exact commit messages).
