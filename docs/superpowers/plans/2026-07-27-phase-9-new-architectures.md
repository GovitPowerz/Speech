# Phase 9 -- New Architectures (Mamba + sLSTM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two modern recurrent cells (sLSTM, Mamba/S6) behind a closed `CellLayer` enum in
the exact f64 tree -- trainable from scratch through the untouched phase-5/6 machinery with
hand-derived analytic backwards pinned by grad checks -- plus causal (forward-only) f32 fast
twins and a `StreamCausal` streaming path that drops the windowed-lookahead latency term.

**Architecture:** Spec `docs/superpowers/specs/2026-07-27-phase-9-new-architectures-design.md`
(S-references below point there). The cut is at the layer: `Network<L>` is already generic,
so `nn/cells/` adds `CellLayer { Lstm, Slstm, Mamba }` implementing the existing 10-method
`Layer` trait; `Blstm`'s stacks re-type to `Network<CellLayer>`; everything above the cell
(drivers, normalization, scoring, seam, `train_modern`) is untouched. Port-only config keys
select cell + direction. Fast twins are step-kernel-based (one kernel shared by offline fast
and streaming, bit-identity by construction).

**Tech Stack:** Rust (ndarray exact tree, faer/f32 fast tree), PyO3 (`speech_rs`), Python
3.14 (numpy, pytest), uv, cargo.

## Global Constraints

- Branch `feature/phase-9-new-architectures`; never commit to main; never push; never `gh`.
- Exact-tree discipline (S1.4): outside the three sanctioned classes -- (a) the enum wrap,
  (b) dead-by-default `Direction` branches, (c) `KEY_TABLE`/config rows -- no `nn/` behavior
  change. The full committed golden suite must be byte-green at EVERY task boundary.
- R1 STOP-and-adjudicate: any argmax/boundary flip or nonzero boundary max_dt in a parity or
  streaming gate stops the phase; never silently widen a tolerance.
- R3 license hygiene: no corpus path/filename/content-derived value committed; corpus tests
  behind `requires_corpus` (Python) / `common::corpus_root_or_skip` (Rust); CI sees only
  synthetic committed fixtures.
- R4 measure-then-pin: every tolerance/budget/latency number is measured first, pinned with
  stated headroom second.
- Port-only design notes go in module docs + `RESULTS.md`, NEVER `IMPROVEMENTS.md`.
- `train_modern` is UNCHANGED by design (S7.3): any change implementation reality demands
  is a spec deviation to surface in review, never to absorb silently.
- Python via `uv run ...`; ASCII-only output; plain hyphens/straight quotes.
- Subagent runs are FOREGROUND-only (no run_in_background, no Monitor waits).
- Gate command set ("Gates") unless a task narrows it:
  `cd src/rust && cargo fmt --check && cargo clippy --release --all-targets -- -D warnings && cargo test --release`
  then from repo root `uv run pytest tests -q -m "not slow"` and `./lint_code.sh`.

## File Structure (locked)

```
src/rust/src/nn/cells/mod.rs        CellLayer enum + Layer delegation (T1; variants T2/T3)
src/rust/src/nn/cells/slstm.rs      SlstmLayer forward+backward (T2)
src/rust/src/nn/cells/mamba.rs      MambaLayer forward+backward (T3)
src/rust/src/nn/blstm.rs            stacks -> Network<CellLayer>; Direction; ctor dispatch (T1-T3)
src/rust/src/nn/network.rs          forward-only branches in feed_*_double (T1)
src/rust/src/toml_config.rs         8 KEY_TABLE rows (T1, T3)
src/rust/src/fast/cells.rs          FastSlstm/FastMamba step kernels + seq wrappers (T6)
src/rust/src/fast/driver.rs         causal dispatch in FastSpectralSegmenter (T6)
src/rust/src/fast/stream.rs         StreamCausal + session dispatch (T7)
src/python/speech/init_weights.py   per-cell builders (T4)
src/python/speech/drivers/baseline.py  --cell-type/--direction knobs (T4)
src/rust/tests/phase9_cell_grad.rs  unit FD tier, both cells x directions (T2/T3)
src/rust/tests/phase9_fast_parity.rs   exact-causal vs fast-causal legs (T6)
src/rust/tests/phase9_stream_causal.rs streaming gate (T7)
tests/test_phase9_init.py           builder length/layout pins (T4)
tests/pyo3/test_phase9_seam.py      seam grad_check + LID mechanical wiring (T5)
tests/pyo3/test_phase9_gates.py     4 from-scratch subset gates, corpus-gated (T8)
tests/pyo3/test_phase9_parity.py    corpus streaming/metric tier, corpus-gated (T9)
tests/reference_data/phase9/        synthetic committed fixture configs + weight seeds
```

Dependency order: T1 -> T2 -> T3 -> T4 -> T5 -> {T6, T8} -> T7 (after T6) -> T9 (after
T7+T8) -> T10 -> T11 -> T12. T6/T7 (Rust `fast/`) and T8 (Python corpus gates) touch
disjoint files and may pipeline per the sdd skill's disjoint-files rule.

---

### Task 1: the cell seam -- `CellLayer`, `Direction`, config keys (behavior-free)

**Files:**
- Create: `src/rust/src/nn/cells/mod.rs`
- Modify: `src/rust/src/nn/mod.rs` (declare `cells`), `src/rust/src/nn/blstm.rs`
  (stack types + ctor dispatch + `Direction`), `src/rust/src/nn/network.rs`
  (forward-only branches), `src/rust/src/toml_config.rs` (4 of the 8 rows),
  `src/python/speech/config_bridge.py` ONLY if `nnet_spec` needs the new keys surfaced
  (read the module first; do not restructure)
- Test: golden suite (the proof of behavior-freeness) + new unit tests in `cells/mod.rs`
  and `src/rust/tests/phase9_cell_grad.rs` is NOT this task (T2 creates it)

**Interfaces:**
- Produces: `pub enum CellLayer { Lstm(LstmLayer) }` (variants grow in T2/T3) with
  `impl Layer for CellLayer` delegating all 10 methods; `Blstm` fields
  `forward_network/backward_network: Option<Network<CellLayer>>`;
  `pub enum Direction { Bidirectional, Forward }` parsed from `BLSTM_Direction`
  (absent -> `Bidirectional`); ctor reads `BLSTM_Cell_Type` (absent -> `lstm`; values
  `slstm`/`mamba` typed-bail `anyhow!("cell type '{v}' not yet implemented")` until T2/T3);
  `Network::feed_forward_double`/`feed_backward_double` gain a forward-only path (no
  hcat, no reverse stack; output width = hidden). Flat keys + TOML rows:
  `BLSTM_Cell_Type` -> `[nn] cell_type`, `BLSTM_Direction` -> `[nn] direction`,
  `BLSTM_LID_Cell_Type` -> `[nn.lid] cell_type`, `BLSTM_LID_Direction` -> `[nn.lid] direction`
  (spell sections per `toml_config.rs`'s existing conventions; flat keys are the contract).

- [ ] **Step 1**: Read `nn/network.rs` (the `Layer` trait + `feed_forward_double`/
  `feed_backward_double`), `nn/blstm.rs` (ctor + stack usage sites), `toml_config.rs`
  (KEY_TABLE shape). Map every `Network<LstmLayer>` mention.
- [ ] **Step 2**: Create `cells/mod.rs`: the enum (Lstm variant only), the 10-method match
  delegation, a module doc naming the phase-9 spec + the three sanctioned touch classes.
- [ ] **Step 3**: Re-type `Blstm`'s stacks; fix construction (wrap `LstmLayer::new` results
  in `CellLayer::Lstm`); `cargo test --release` -- the FULL suite must pass byte-green.
  Any golden diff here is a defect in the wrap, full stop.
- [ ] **Step 4**: Add `Direction`: key parse, ctor plumb (`Forward` -> `backward_network =
  None` + output net sized `hidden`), the forward-only branches in both `*_double`
  methods. Unit-test the forward-only shape path with a tiny in-test LSTM net (T=4,
  in=3, hidden=2: output rows have `hidden` columns pre-MLP; `feed_backward_double`
  routes deltas only to the forward stack).
- [ ] **Step 5**: Add the 4 KEY_TABLE rows + round-trip test (toml -> flat -> toml keeps
  the keys); `BLSTM_Cell_Type slstm` on the tier-2 fixture config typed-bails with the
  not-yet-implemented wording (pinned test, removed by T2).
- [ ] **Step 6**: Gates. Commit `feat(phase9): CellLayer seam + Direction + config keys
  (behavior-free, goldens byte-green)`.

---

### Task 2: `SlstmLayer` -- forward + analytic backward + unit FD tier

**Files:**
- Create: `src/rust/src/nn/cells/slstm.rs`, `src/rust/tests/phase9_cell_grad.rs`
- Modify: `src/rust/src/nn/cells/mod.rs` (variant + delegation arms),
  `src/rust/src/nn/blstm.rs` (ctor arm: `slstm` builds `CellLayer::Slstm`, removes the T1
  bail for slstm)
- Test: `phase9_cell_grad.rs` + inline `#[cfg(test)]`

**Interfaces:**
- Consumes: T1's enum/ctor seam.
- Produces: `SlstmLayer::new(input_size: usize, output_size: usize) -> Self`;
  `Layer`-conforming methods; flat layout per spec S2.2 -- gate blocks `[i|f|o|z]`, each
  row-major `[R (out x out) | W (out x in) | b (out)]`, `nb_of_weights = 4*out*(out+in+1)`.
  State zero-init per call; `m_0 = -1e30` (S2.4). `forget` block is index 1 (`f`).

- [ ] **Step 1**: Write the FD harness FIRST in `phase9_cell_grad.rs` (shared by T3 --
  design it cell-agnostic over `&mut dyn Layer`):

```rust
/// Central-diff vs analytic on L = sum(G .* Y) where G is a fixed pseudo-random
/// matrix. Analytic: feed_forward, then feed_backward with deltas = G; col0 of
/// get_weights_derivatives IS dL/dw (col1 is the count, unused here).
/// last_layer = false everywhere (cells never sit at the output; NeuronLayer does).
fn fd_check(layer: &mut dyn Layer, t: usize, input_size: usize, eps: f64) -> f64 {
    // seed weights + input + G from a deterministic LCG (no rand dep), run FD per
    // weight: (L(w+eps) - L(w-eps)) / (2*eps), return max relative error
    // rel = |fd - analytic| / max(1e-8, |fd|.max(|analytic|))
}
```

  Failing first: `SlstmLayer` does not exist -- the test file does not compile. That IS the
  red step for a new-module task.
- [ ] **Step 2**: Implement the forward exactly per S2.1 (stabilized exp gating, twin
  `c`/`n` recurrences, `h = sigmoid(o~) * c/n`), `feed_forward_reverse` NOT needed
  cell-side (the wrapper flips) -- implement it as the same forward (the trait requires
  it; `Blstm` handles reversal outside, mirroring how `LstmLayer` receives pre-reversed
  input). Cache per-timestep gate values, `c`, `n`, `m`, `h` for backward.
- [ ] **Step 3**: Forward sanity units: shapes at T=1/T=7/odd T; run-twice bit-identical;
  a hand-computed 1-unit/1-input single-step value (compute i',f',c,n,h by hand in the
  test with literal constants); stabilizer non-overflow probe (pre-activations ~ +-800
  stay finite -- the unstabilized exp would be inf).
- [ ] **Step 4**: Implement the backward per S2.3 (m CONSTANT -- exact by the S2.3
  invariance; quotient rule; twin adjoint recurrences; `[R|W|b]` accumulation with col1 =
  frame count) + `get/reset/ponderate_weights_derivatives` + `set/get_weights` in the
  S2.2 order.
- [ ] **Step 5**: FD-pin: `fd_check` at (t, in, out) in {(1,3,2), (7,3,2), (11,5,4),
  (23,7,3)} x 3 seeds, eps 1e-6 -> measure the max rel error, then PIN at measured*10
  (expected ~1e-6..1e-7 band; if any point exceeds 1e-4 STOP and debug -- do not widen).
  Also pin: derivative accumulation across two feed_backward calls sums (reset clears).
- [ ] **Step 6**: Wire the enum variant + ctor arm; the T1 slstm-bail test flips to a
  construction-succeeds test. Golden suite still byte-green (LSTM paths untouched).
- [ ] **Step 7**: Gates. Commit `feat(phase9): SlstmLayer forward + analytic backward,
  FD-pinned`.

---

### Task 3: `MambaLayer` -- forward + analytic backward + unit FD tier

**Files:**
- Create: `src/rust/src/nn/cells/mamba.rs`
- Modify: `src/rust/src/nn/cells/mod.rs` (variant), `src/rust/src/nn/blstm.rs` (ctor arm +
  hyperparam reads), `src/rust/src/toml_config.rs` (4 rows: `Mamba_D_State` ->
  `[nn.mamba] d_state` 16, `Mamba_D_Conv` -> `d_conv` 4, `Mamba_Expand` -> `expand` 2,
  `Mamba_Dt_Rank` -> `dt_rank` 0=auto), `src/rust/tests/phase9_cell_grad.rs` (mamba legs
  reusing the T2 harness)
- Test: same files

**Interfaces:**
- Consumes: T1 seam, T2's `fd_check` harness.
- Produces: `MambaLayer::new(input_size, output_size, d_state, d_conv, expand, dt_rank)`
  (`dt_rank == 0` -> `ceil(out/16)`); flat layout per S3.2 IN THIS ORDER: `[P|p]` (only
  when `in != out`), `g`, `W_in`, `conv|b_conv`, `W_x`, `W_dt|b_dt`, `A_log`, `D`,
  `W_out`; forward per S3.1 (width adapter -> RMSNorm(eps 1e-5) -> in_proj -> depthwise
  causal conv1d + SiLU -> selective SSM recurrence with per-timestep `Delta_t/B_t/C_t` ->
  `y * SiLU(res)` -> out_proj -> +residual).

- [ ] **Step 1**: Extend `phase9_cell_grad.rs` with mamba cases (compile-fail red).
- [ ] **Step 2**: Implement the forward per S3.1. Left-pad the conv with `d_conv - 1`
  zeros (causal); `A = -exp(A_log)` computed per call; cache `uc`, `Delta`, `B`, `C`,
  `h_t` sequence, RMSNorm inverse-rms, SiLU pre-activations for backward.
- [ ] **Step 3**: Forward units: shapes (in==out pure-block AND in!=out adapter cases);
  T=1; run-twice; stability probe (`Abar in (0,1)` -- assert every computed `Abar` is in
  (0,1) on random weights); a literal hand-computed single-step at d_model=1, d_state=1,
  d_conv=1, expand=1 (degenerate but exercises every op incl. softplus/exp/RMSNorm).
- [ ] **Step 4**: Implement the backward per S3.5 (state adjoint `dh_t = dy_t*C_t +
  Abar_{t+1}.dh_{t+1}`, exp product rule to `dDelta/dA_log`, conv backward as flipped
  correlation + bias sums, SiLU'/softplus'/RMSNorm backward, projection transposes,
  residual pass-through, adapter transpose when present).
- [ ] **Step 5**: FD-pin at (t, in, out, d_state, d_conv, expand) in {(1,3,3,2,2,1),
  (7,3,3,4,3,2), (9,4,6,4,4,2) adapter case, (23,6,6,8,4,2)} x 3 seeds; same
  measure-then-pin protocol as T2 Step 5 (STOP over 1e-4).
- [ ] **Step 6**: Enum variant + ctor arm + hyperparam key reads + the 4 KEY_TABLE rows
  (+ round-trip test); the T1 mamba-bail test flips to construction-succeeds. Goldens
  byte-green.
- [ ] **Step 7**: Gates. Commit `feat(phase9): MambaLayer forward + analytic backward,
  FD-pinned`.

---

### Task 4: Python init builders + baseline arm knobs

**Files:**
- Modify: `src/python/speech/init_weights.py` (per-cell builders),
  `src/python/speech/drivers/baseline.py` (`--cell-type`, `--direction` knobs)
- Create: `tests/test_phase9_init.py`
- Test: `tests/test_phase9_init.py` (the pyo3 Engine round-trip leg belongs to T5, which
  runs after this task and owns `tests/pyo3/test_phase9_seam.py`)

**Interfaces:**
- Consumes: S2.2/S3.2 flat layouts (READ the Rust `set_weights` implementations first --
  the Rust order IS the contract).
- Produces: `init_weights.py`: `init_slstm_flat(spec, rng) -> NDArray` and
  `init_mamba_flat(spec, rng, d_state, d_conv, expand, dt_rank) -> NDArray` (exact
  argument spellings may follow the module's existing `init_weights(spec, rng, scheme,
  forget_bias_one)` conventions -- extend that entry point with a `cell_type` parameter
  rather than free functions IF it reads cleaner in the module; the CONTRACT is: seeded,
  per-cell scheme constants per S2.4/S3.4, emitting the Rust flat order, LSTM path
  byte-untouched). `baseline.py`: `--cell-type {lstm,slstm,mamba}` + `--direction
  {bidirectional,forward}` overlay the S6 keys onto the arm config and route init through
  the per-cell builders; DEFAULT invocation byte-identical to today (pinned).

- [ ] **Step 1**: Write failing tests: builder length == the Rust `nb_of_weights` formula
  for 3 shapes per cell; sLSTM forget-bias block slice == 1.0 exactly; mamba `A_log`
  rows == `log(s+1)`; `D` == 1; `g` == 1; `b_dt` in `softplus^-1([1e-3, 1e-1])`;
  same-seed determinism; LSTM path unchanged (existing pins keep passing).
- [ ] **Step 2**: Implement the builders (S2.4/S3.4 constants are SPEC text -- copy them,
  do not re-derive). Run: `uv run pytest tests/test_phase9_init.py -v` -> PASS.
- [ ] **Step 3**: Add the baseline knobs: parser args, config-key overlay, init routing,
  and a default-run byte-identity pin (build the arm config text with and without the
  new args at defaults -> identical string).
- [ ] **Step 4**: Gates. Commit `feat(phase9): per-cell init builders + baseline arm
  cell/direction knobs`.

---

### Task 5: seam tier -- committed gate configs, `grad_check` legs, LID mechanical wiring

**Files:**
- Create: `tests/reference_data/phase9/` (synthetic fixture configs: the tier-2 spectral
  fixture cloned x4 -- `{slstm,mamba} x {bidirectional,forward}` -- plus one Twin config
  with `BLSTM_LID_Cell_Type slstm`; seeded weight `.bin`s generated by the T4 builders
  via a `scripts/`-side generator committed alongside), `tests/pyo3/test_phase9_seam.py`
- Modify: `src/rust/src/tasks/lid.rs` ONLY if the Twin's LID-net construction does not
  already flow through the T1-T3 ctor dispatch (read first; expected: it constructs
  `Blstm` from `BLSTM_LID_*` keys, so the dispatch is already live)
- Test: `tests/pyo3/test_phase9_seam.py`

**Interfaces:**
- Consumes: T1-T4. `speech_rs.Engine.grad_check(max_weights)` + `weights_derivatives()`
  (the existing seam surface, unchanged).
- Produces: the committed phase-9 fixture family (CI-visible, synthetic, license-clean)
  later tasks reuse; `test_phase9_seam.py` assertions later gates import.

- [ ] **Step 1**: Build the 4 SAD fixture configs (tier-2 clone + `BLSTM_Cell_Type` /
  `BLSTM_Direction` + for mamba small hyperparams `d_state 4, d_conv 3, expand 2`) +
  seeded weight packs via the T4 builders. Keep nets TINY (grad_check runtime).
- [ ] **Step 2**: Failing test first: `Engine(config).grad_check(cap)` per fixture -- FD vs
  analytic max rel error, measure across the 4 configs, THEN pin at measured*10 (STOP
  over 1e-4). Also: `set_weights(pack)` then `weights()` returns the pack BIT-identical
  per cell (the S1.3 cross-language round-trip pin); `weights_derivatives` after `run`
  is nonzero and finite (the F10 regression class); run-twice determinism bit-identical.
- [ ] **Step 3**: LID mechanical wiring: the Twin fixture constructs, `set_weights`/
  `get_weights` round-trips both nets, `grad_check` on the Twin passes at the pinned
  tolerance. If construction does NOT flow through the dispatch, fix minimally in
  `tasks/lid.rs` (surface in the ledger as a spec-expected-vs-found note).
- [ ] **Step 4**: Gates. Commit `test(phase9): seam grad_check legs + LID mechanical
  wiring on committed synthetic fixtures`.

---

### Task 6: f32 fast twins (causal only) + driver dispatch + parity legs

**Files:**
- Create: `src/rust/src/fast/cells.rs`, `src/rust/tests/phase9_fast_parity.rs`
- Modify: `src/rust/src/fast/mod.rs` (declare), `src/rust/src/fast/driver.rs`
  (`FastSpectralSegmenter` construction dispatch on cell+direction; typed bails),
  `src/rust/src/engine/bag_of_processors.rs` ONLY if `Inference_Path fast` dispatch
  needs a new arm (read first)
- Test: `phase9_fast_parity.rs`

**Interfaces:**
- Consumes: T2/T3 exact cells (the parity oracle), T5 fixtures.
- Produces: `FastSlstm::from_flat(flat: &[f64], input_size, output_size) -> Self` with
  `pub struct SlstmState { h, c, n, m: Vec<f32> }`, `fn state(&self) -> SlstmState`
  (zeroed, m = -1e30f32), `fn step(&self, x: &[f32], s: &mut SlstmState, out: &mut [f32])`;
  `FastMamba::from_flat(flat, input_size, output_size, d_state, d_conv, expand, dt_rank)`
  with `MambaState { conv_ring: Vec<f32>, ring_pos: usize, h: Vec<f32> }` and the same
  `state()/step()` shape; both with `fn feed_forward(&self, seq) -> FastMatrix` wrappers
  that LOOP `step` (bit-identity by construction). Driver: causal fast = stacked cell
  steps + the existing fast dense/softmax on `hidden` columns. Typed bails (each pinned):
  `Inference_Path fast` + bidirectional new cell; causal fast + `sub_sampling != 1`;
  causal fast + LSTM cell (named follow-on).

- [ ] **Step 1**: Failing tests: construction from the T5 fixture packs (length checks);
  step-loop == whole-sequence wrapper BIT-identical; state carried across two
  half-sequences == one full sequence BIT-identical (the streaming precondition).
- [ ] **Step 2**: Implement both fast cells (f32 transcriptions of the T2/T3 forwards;
  narrow once at `from_flat` post-pack, `flat[i] as f32`; preallocated workspaces,
  lazy-grow, run-twice bit-identical).
- [ ] **Step 3**: Parity legs vs the exact tree: on the T5 forward fixtures (fwd-slstm,
  fwd-mamba), exact f64 posterior vs fast f32 posterior -- measure max rel delta, pin at
  measured*10 (phase-7 expectation band ~1e-6..1e-5); boundary count/types IDENTICAL
  through the shared f64 decision layer; argmax zero flips. R1 STOP on any flip.
- [ ] **Step 4**: Driver dispatch + the three typed bails, each with a pinned test.
- [ ] **Step 5**: Gates. Commit `feat(phase9): causal f32 fast twins + driver dispatch,
  parity pinned`.

---

### Task 7: `StreamCausal` + session dispatch + the streaming gate

**Files:**
- Modify: `src/rust/src/fast/stream.rs` (`StreamCausal` layer + `StreamingSession`
  dispatch windowed-vs-causal + S5.3 validation bails)
- Create: `src/rust/tests/phase9_stream_causal.rs`
- Test: `phase9_stream_causal.rs` + the existing `phase8_gate.rs` must stay green
  (windowed path untouched)

**Interfaces:**
- Consumes: T6 step kernels + states; phase-8 `StreamFrontEnd`/`StreamDecision`/frozen-norm
  UNCHANGED.
- Produces: `StreamingSession::new(map, rate, channels)` unchanged signature, internal
  dispatch on cell+direction; per finalized feature row: frozen type-1 affine -> cell
  step stack -> fast dense/softmax -> posterior -> `StreamDecision` push. EOS = front-end
  reach-tail flush only (no window flush). S5.3 bails pinned: causal streaming requires
  fwd+new-cell, window 0, MONO, `Audio_fixed_gain`, `InputNormalizationType 1`,
  `output_size 1`, algo 3, `sub_sampling 1`.

- [ ] **Step 1**: Failing gate first (the phase-8 gate pattern, on the T5 fwd fixtures +
  synthetic wav): streamed `finish()` `Segmentation` + posterior history BIT-FOR-BIT ==
  the offline fast causal bag run; boundary max_dt EXACTLY 0.0 (R1 STOP on nonzero);
  chunk-invariant at 20/100/1000/7 ms; prefix zero-retractions.
- [ ] **Step 2**: Implement `StreamCausal` + the dispatch + bails (each bail pinned).
- [ ] **Step 3**: Latency: DERIVE the structural bound from the gate config
  (`feature_reach + conv_delay + holdback` -- the nn_window term is gone; show the
  arithmetic in the test comment), measure per-class push-lag, PIN: speech-class max lag
  <= bound; the silence-class commit-wait pinned per-class as in phase 8.
- [ ] **Step 4**: `phase8_gate.rs` + full suite green (windowed path byte-untouched).
- [ ] **Step 5**: Gates. Commit `feat(phase9): StreamCausal streaming path -- bit-equal
  gate + derived latency bound`.

---

### Task 8: from-scratch subset gates (corpus-gated) + RESULTS comparison

**Files:**
- Create: `tests/pyo3/test_phase9_gates.py`
- Modify: `RESULTS.md` (phase-9 section: the 4-gate table + vs-BLSTM comparison),
  `tests/reference_data/phase9/` (the two ARM configs if the baseline knobs need
  committed variants; runtime placeholder listing keys per the phase-6 license pattern)
- Test: `tests/pyo3/test_phase9_gates.py`

**Interfaces:**
- Consumes: T4 knobs (`speech baseline sad --cell-type ... --direction ...` or the
  equivalent `run_baseline` API), the phase-6 SAD subset protocol
  (`tests/pyo3/test_phase6_gates.py` is the template -- READ IT FIRST), `evaluate.dcf`.
- Produces: 4 corpus-gated gates -- `{slstm,mamba} x {bidirectional,forward}` -- each:
  <10 min, run-twice bit-identical checkpoints, HARD leg = trained held-out DCF beats
  own-init DCF deterministically; the vs-BLSTM subset number (0.2500) RECORDED in
  RESULTS.md, NOT hard-gated (R5 wording).

- [ ] **Step 1**: Param-match sizing: compute hidden/d_model per cell so total weights ~=
  the BLSTM arm pack +-15% (record the arithmetic in the test file docstring; sLSTM
  `4*out*(out+in+1)` + MLP; mamba per the S3.2 blocks).
- [ ] **Step 2**: Write the 4 gates (phase-6 template: corpus-gated skip, seeded init via
  T4, `train_modern` with `val_metric="nn_cost_seg"`, held-out DCF via the engine VRCTS
  dumps + `_pool_dcf`). Run locally against the corpus. Any gate that does NOT beat init:
  investigate lr/epochs within the phase-6 knob envelope FIRST; if a cell genuinely
  cannot beat init on the subset, STOP and surface (do not soften the hard leg).
- [ ] **Step 3**: Record the 4 results + the vs-BLSTM comparison + the sizing table in
  `RESULTS.md` with the R5 thinness caveat verbatim.
- [ ] **Step 4**: Gates (non-slow suite untouched by the corpus-gated file). Commit
  `test(phase9): four from-scratch subset gates - both cells, both directions`.

---

### Task 9: corpus streaming/metric tier + bench rows

**Files:**
- Create: `tests/pyo3/test_phase9_parity.py`
- Modify: `RESULTS.md` (streaming-on-real-data + bench tables; local budget pins in
  `src/rust/tests/phase9_fast_parity.rs` ONLY if a bench-regression bound is warranted --
  follow the phase-7 local-vs-CI budget split)
- Test: `tests/pyo3/test_phase9_parity.py`

**Interfaces:**
- Consumes: T7 session, T8's fwd checkpoints (the causal artifacts), the phase-8 corpus
  tier (`tests/pyo3/test_phase8_parity.py` is the template), `speech bench`.
- Produces: corpus-gated legs -- (a) streamed vs offline-fast-causal on one real train wav
  (sorted-first) with a T8 fwd checkpoint: boundary within the VRCTS write quantum,
  in-process max_dt 0.0, R1 STOP; (b) fast-vs-exact causal metric deltas on the T8
  checkpoints (DCF float-equal or measured-then-pinned); (c) `speech bench` RTF/RSS rows
  for both causal cells vs the BLSTM fast baseline, recorded in RESULTS (the linear-time
  hypothesis measured, not assumed).

- [ ] **Step 1**: Write + run the corpus legs locally (license hygiene: no corpus name
  committed; the phase-6/7 cache recipe for checkpoints).
- [ ] **Step 2**: Bench runs (sorted-first selection, `--repeat=3`), record the table.
- [ ] **Step 3**: Gates. Commit `test(phase9): corpus streaming parity + causal bench
  rows`.

---

### Task 10: mutation battery

**Files:** `IMPROVEMENTS.md` only ("Mutation battery (Phase 9)"; tree clean between
cycles; FOREGROUND).

- [ ] The list (apply-FAIL-revert-PASS): (1) sLSTM stabilizer dropped (`m_t = 0`) -> the
  T2 overflow probe + FD pins; (2) `A_log` sign flip (`A = +exp`) -> the T3 stability
  probe (`Abar in (0,1)`) + FD; (3) mamba conv-ring off-by-one -> the T6 split-sequence
  bit-identity + T7 chunk-invariance; (4) step-state not reset between files -> the T6
  state tests + T7 gate; (5) `Direction` hcat misroute (2*hidden on forward) -> T1 shape
  units + T5 seam; (6) packer block-order swap (sLSTM `[i|f|z|o]`) -> T4 layout pins +
  T5 round-trip; (7) fast f32 state seeded m=0 -> T6 parity vs exact; (8) mamba init
  `A_log` uniform (scheme swap) -> T4 init pins + the T8 determinism leg (corpus-gated
  catcher recorded honestly if that is the only catcher). Honest gaps recorded.
- [ ] Full final pass (non-slow + per-file slow + cargo + lint). Commit `test(phase9):
  mutation battery results recorded`.

---

### Task 11: docs refresh

**Files:** `README.md`, `CLAUDE.md`, `RESULTS.md`, `IMPROVEMENTS.md` consistency.

- [ ] CLAUDE.md: new module rows (`nn/cells/*`, `fast/cells.rs`), row updates where T1-T7
  extended existing modules (`nn/blstm.rs`, `nn/network.rs`, `fast/driver.rs`,
  `fast/stream.rs`, `toml_config.rs` KEY_TABLE count, `init_weights.py`, `baseline.py`);
  the Roadmap 2 Phase 9 line flips to done-with-numbers (grad-check tolerances, the 4
  gate verdicts, the latency bound, bench); a "Since Phase 9" convention bullet if
  warranted. README: phase bullet + causal quick-start. Every claim code-verified; hashes
  literal. Ledgered review Minors folded in.
- [ ] Gates + commit `docs(phase9): README/CLAUDE.md/RESULTS refresh - new architectures
  land`.

---

### Task 12: final review + smart-commit + finish

- [ ] Whole-branch final review (the sdd final-review pattern: package the branch diff,
  fresh reviewer, fix-now items resolved before proceeding).
- [ ] Invoke the `smart-commit` skill, telling it to take the WHOLE git branch into
  account.
- [ ] Invoke `superpowers:finishing-a-development-branch` (verify tests, present options;
  the user's standing choice is keep-branch-as-is, he merges via PR himself; NEVER push).
