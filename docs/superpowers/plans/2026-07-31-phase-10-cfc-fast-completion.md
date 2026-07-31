# Phase 10 -- CfC + Fast-Tree Completion + v2 Lineage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The CfC cell end-to-end (exact f64 train + f32 causal fast + streaming, behind
the `CellLayer` seam), the fast tree's (cell x direction) matrix completed (bi twins,
fwd-LSTM twin, full-f32 mel, Mamba retention gating, stream-lid surface), and the clean
11-wide `lre_sad_v2` lineage baselined before the full-corpus launchers fire.

**Architecture:** Spec `docs/superpowers/specs/2026-07-31-phase-10-cfc-fast-completion-design.md`
(S-references point there). CfC follows the phase-9 4th-variant touch list verbatim
(compile-error-enforced). The fast completion follows ratified approach A: a parallel
`FastBiCell` driver against shared span helpers, `FastBlstm`/`overlap_window_step`
byte-untouched. The f32 mel lands FIRST among fast changes and owns the one sanctioned
re-pin sweep (S9.1). v2 is a one-variable config fork with an inverse (zero-dead-columns)
guard and a fresh 8-gate matrix.

**Tech Stack:** Rust (ndarray exact tree, faer/realfft f32 fast tree), PyO3 (`speech_rs`),
Python 3.14 (numpy, pytest), uv, cargo.

## Global Constraints

- Branch `feature/phase-10-cfc-fast-completion`; never commit to main; never push; never `gh`.
- **R6 MEMORY-PRESSURE PROTOCOL (standing, from the phase-9 host reboots): every cargo
  build/test/clippy invocation carries `-j 4`; test execution carries `-- --test-threads=4`;
  never two compile-heavy commands concurrently (no cargo + maturin overlap); targeted
  single-file test runs over full suites; at most ONE full-suite run per task.**
- R2 exact-tree byte-freeze outside: (a) the CfC additions inside `nn/cells/` + the
  compile-enforced dispatch arms, (b) S7's adjudicated retention class. The committed
  golden suite byte-green at EVERY task boundary is the arbiter.
- R1 STOP-and-adjudicate: any boundary/argmax flip or nonzero boundary max_dt in a
  parity/streaming gate stops the phase (BLOCKED with numbers); never silently widen.
  The T5 mel sweep re-pins tolerance MAGNITUDES once, by design -- flips stay forbidden.
- R3 license hygiene: corpus tests behind `requires_corpus`; no corpus path/filename/
  content-derived value committed; synthetic fixtures only in CI.
- R4 measure-then-pin: every tolerance/budget/RSS number measured first; the CfC sizing
  (S1.4) and v2 pack lengths (S3.2) are measured at implementation, never transcribed
  from survey estimates.
- R5 honest comparison: subset numbers carry the thinness caveat; the v2 Xavier-scaling
  payoff is framed measurable-not-promised.
- Port-only design notes go in module docs + `RESULTS.md`, NEVER `IMPROVEMENTS.md`.
- Python via `uv run ...`; ASCII-only output; plain hyphens/straight quotes.
- Subagent runs are FOREGROUND-only (no run_in_background, no Monitor waits).
- Gate command set ("Gates") unless a task narrows it:
  `cd src/rust && cargo fmt --check && cargo clippy --release -j 4 --all-targets -- -D warnings && cargo test --release -j 4 -- --test-threads=4`
  then from repo root `uv run pytest tests -q -m "not slow"` and
  `uv run pytest tests/pyo3 -q` and `./lint_code.sh` (sequenced, never concurrent).

## File Structure (locked)

```
src/rust/src/nn/cells/cfc.rs           CfcLayer forward+backward (T1)
src/rust/src/nn/cells/mod.rs           Cfc variant + 10 delegation arms (T1)
src/rust/src/nn/blstm.rs               CellType::Cfc + parse/as_str + ctor arm + CfcParams (T1)
src/rust/src/toml_config.rs            2 KEY_TABLE rows [nn_cfc] (T1)
src/rust/tests/phase9_cell_grad.rs     CfC FD Cases (T1; the cell-agnostic harness grows rows)
src/python/speech/config_bridge.py     Cfc geometry reader (T2)
src/python/speech/init_weights.py      init_cfc_flat + _init_cell_pack branch (T2)
src/python/speech/drivers/baseline.py  cfc in choices tuples + sad-v2 arm key (T2, T4)
tests/test_phase10_init.py             CfC builder pins + reconstruction (T2)
scripts/extract_phase9_fixtures.py     +2 CfC seam fixtures (T3; same generator, new rows)
tests/reference_data/phase9/           cfc_{bidirectional,forward}.config + packs (T3)
tests/pyo3/test_phase9_seam.py         CfC grad_check + LID-wiring rows (T3)
configs/training/lre_sad_v2.toml       the one-variable v2 fork (T4)
configs/training/lre_sad.toml          header pointer to v2 (T4, comment-only)
tests/pyo3/test_phase10_gates.py       the v2 8-gate matrix + inverse guard (T4)
src/rust/src/fast/mel32.rs             f32 mel/DCT/deltas/SDC kernels (T5)
src/rust/src/fast/pipeline.rs          both widen sites -> mel32 (T5)
src/rust/tests/phase7_parity_*.rs      re-measured pins (T5, pin values only)
src/rust/tests/phase9_fast_parity.rs   re-measured pins (T5) + FastCfc causal legs (T6)
src/rust/src/fast/cells.rs             FastCfc step kernel + FastCell::Cfc (T6); FastCell::Lstm (T8)
src/rust/tests/phase9_stream_causal.rs CfC streaming rows (T6); fwd-LSTM rows (T8)
src/rust/src/fast/bicell.rs            FastBiCell + the fresh overlap loop (T7)
src/rust/src/fast/driver.rs            dispatch flips: BiCell + Causal(Lstm) (T7, T8)
src/rust/tests/phase10_bicell_parity.rs  bi-twin parity legs, 3 cells (T7)
src/rust/src/nn/network.rs             layers_output retention gate (T9, adjudicated class)
src/rust/src/nn/cells/mamba.rs         MambaCache retention gate (T9)
src/rust/src/nn/blstm.rs               set_inference_only thread (T9)
src/rust/src/stream_cli.rs (or sibling) speech stream-lid subcommand (T10)
src/rust/speech-py/src/lib.rs          StreamingLidSession pyclass (T10)
tests/pyo3/test_phase10_stream_lid.py  binding smokes (T10)
```

Dependency order: T1 -> T2 -> {T3, T4} and independently T5 -> T6 -> T7 -> T8; T9 and
T10 are independent of everything after T1; T11 (battery) -> T12 (docs) -> T13 (final)
close. Pipelining for sdd: {T3, T4} may pipeline with {T5, T6} (disjoint files:
python/configs vs fast Rust); T9/T10 slot wherever a lane is free; ONE committer at a
time as always.

---

### Task 1: the CfC exact cell -- forward + analytic backward + FD tier + dispatch

**Files:**
- Create: `src/rust/src/nn/cells/cfc.rs`
- Modify: `src/rust/src/nn/cells/mod.rs` (variant + 10 arms), `src/rust/src/nn/blstm.rs`
  (CellType::Cfc + parse/as_str + the exhaustive `from_config` arm + `CfcParams`
  {backbone_units, backbone_layers} read like `MambaParams` with `>= 1` validation),
  `src/rust/src/toml_config.rs` (2 rows: `Cfc_Backbone_Units` -> `[nn_cfc]
  backbone_units`, `Cfc_Backbone_Layers` -> `[nn_cfc] backbone_layers`),
  `src/rust/tests/phase9_cell_grad.rs` (CfC Cases), `src/rust/tests/phase9_cell_config.rs`
  (round-trip rows)
- Test: FD harness Cases + inline `#[cfg(test)]` in cfc.rs

**Interfaces:**
- Consumes: the `Layer` trait (nn/network.rs:25-67), the `for_each_slot` single-walk
  pattern (slstm.rs is the template), the cell-agnostic FD harness (`sweep(label, make,
  cases, ...)` with per-Case pins).
- Produces: `CfcLayer::new(input_size, output_size, backbone_units, backbone_layers)`;
  S1.1 forward EXACTLY (`z = [x|h_prev]`, `lecun_tanh(x) = 1.7159*tanh(2x/3)` backbone,
  tanh'd heads ff1/ff2, `sigmoid` gate g via `logistic_fn`, `h = ff1*(1-g) + ff2*g`,
  `h_0 = 0`); S1.2 flat order `W_bb|b_bb | [W|b per deeper layer] | W_1|b_1 | W_2|b_2 |
  W_t|b_t`, `nb = B*(in+h+1) + (L-1)*B*(B+1) + 3*h*(B+1)`; reverse methods via the
  wrapper flip (implement as same-forward, the sLSTM precedent); default
  `Cfc_Backbone_Units` = the T2-sized value -- T1 lands the key with a PROVISIONAL
  default 24 and T2's sizing task finalizes it (one-line change, noted in both tasks).

- [ ] **Step 1**: Read slstm.rs in full (the house cell-module template: for_each_slot,
  module doc, test organization, hand-computed single-step, mutation-probe docs) + the
  FD harness's Case/sweep contract.
- [ ] **Step 2**: RED -- add the CfC Cases to phase9_cell_grad.rs at shapes
  `(t, in, out, B, L)` in {(1,3,2,4,1), (7,3,2,4,1), (11,5,4,8,2), (23,7,3,8,1)} x 3
  seeds; file fails to compile (CfcLayer absent).
- [ ] **Step 3**: Implement the forward per S1.1. Cache per-timestep: z (input+state
  concat), per-backbone-layer pre-activations, the three head pre-activations, g and
  the heads' post-activations. Forward units: shapes, T=1, run-twice bit-identical, a
  literal hand-computed single-step at (in=1, h=1, B=1, L=1), lecun_tanh constant pins.
- [ ] **Step 4**: Implement the backward: reverse-time loop; adjoints
  `dh -> dff1 = dh*(1-g)`, `dff2 = dh*g`, `dg = dh*(ff2-ff1)`; head locals
  `tanh' = 1-tanh^2`, `sigmoid' = g*(1-g)`; backbone `lecun_tanh'(x) =
  1.7159*(2/3)*(1-tanh^2(2x/3))`; matmul transposes; `dz` splits into returned `dx_t`
  and carried `dh_{t-1}`; `[deriv|count]` accumulation col1 = frame count; accumulate/
  reset/ponderate per the trait.
- [ ] **Step 5**: FD-pin: measure per shape, pin at measured*10, in-test assert every
  pin < 1e-4 (STOP over 1e-4 -- BLOCKED with numbers, never widen). Structural
  resolvable-count asserts (predicted == measured expected: no dead blocks predicted --
  assert the count equals the full pack minus any T=1 structural exceptions the
  derivation actually finds; document the arithmetic).
- [ ] **Step 6**: Wire: CellType::Cfc + parse/as_str, the exhaustive from_config arm
  (compile error resolves), CellLayer::Cfc + 10 delegation arms, CfcParams from_legacy
  (defaults provisional-24/1, `>= 1` bail), the 2 KEY_TABLE rows + round-trip test rows.
  The fast/stream typed bails auto-reject cfc (cell != Lstm at the bidirectional bail;
  the causal arm auto-ADMITS cfc but FastCell::from_flat has no Cfc arm yet -- verify
  what actually happens on `Inference_Path fast` + cfc/forward TODAY and pin the
  current behavior (a typed error, not a panic; if it panics, add a bail in
  classify_fast_shape's causal arm gated on FastCell support, removed by T6).
- [ ] **Step 7**: Gates (R6-capped). Goldens byte-green. Commit
  `feat(phase10): CfcLayer forward + analytic backward, FD-pinned`.

---

### Task 2: CfC Python -- geometry, init builder, sizing, knobs

**Files:**
- Modify: `src/python/speech/config_bridge.py` (`_cfc_geometry` reader, the Mamba
  precedent: present-but-invalid raises, absent -> defaults), `src/python/speech/
  init_weights.py` (`init_cfc_flat(rng, out, fin, backbone_units, backbone_layers)` +
  the `_init_cell_pack` dispatch branch), `src/python/speech/drivers/baseline.py`
  (`cfc` in the --cell-type choices tuples), `src/rust/src/nn/blstm.rs` +
  `src/rust/src/toml_config.rs` (finalize the Cfc_Backbone_Units default per the sizing)
- Create: `tests/test_phase10_init.py`
- Test: `tests/test_phase10_init.py`

**Interfaces:**
- Consumes: T1's flat order (the Rust `for_each_slot` IS the contract -- read it first);
  `weight_bridge.spec_directions`; the `_init_cell_pack` assembly.
- Produces: `init_cfc_flat` emitting the S1.2 order, Xavier/He per repo convention on
  every block, biases 0 (S1.5 -- no reference-lineage constants, stated in the
  docstring); the SIZED `Cfc_Backbone_Units` default: compute B such that the v2-lineage
  full-net CfC pack lands within +-15% of the v2 LSTM pack (v2 layer-0 in=44 after
  sub-sampling, layers 24,24 - do the arithmetic against T1's nb formula, record it in
  the test docstring; v1's CfC mechanical leg may override B per-config if the default
  breaks v1's band -- v1 CfC has no param-match requirement, note it).

- [ ] **Step 1**: RED -- length pins (== the Rust nb formula for 3 shapes), block-value
  pins (biases exactly 0), determinism, LSTM/sLSTM/Mamba paths byte-untouched.
- [ ] **Step 2**: Implement builder + geometry reader + dispatch branch. PASS.
- [ ] **Step 3**: The whole-pack block-by-block reconstruction pin (MANDATORY, the T4
  shear lesson): reconstruct the full pack from a fresh `default_rng(seed)` drawing
  block by block in spec order with the documented fans, assert BIT-identity; plus a
  shear companion (swap two adjacent same-distribution blocks in a copy of the
  reconstruction -> must differ from the builder output).
- [ ] **Step 4**: Sizing: compute + record the B arithmetic for both lineages; finalize
  the Rust-side default (one-line change in blstm.rs; KEY_TABLE row comment updated);
  add the baseline choices `cfc`; pin default-invocation byte-identity still holds.
- [ ] **Step 5**: Gates (R6). Commit
  `feat(phase10): CfC init builder + geometry + sized default`.

---

### Task 3: CfC seam fixtures + grad_check + LID wiring

**Files:**
- Modify: `scripts/extract_phase9_fixtures.py` (2 new rows: cfc_bidirectional,
  cfc_forward -- same tier-2 clone recipe, tiny geometry e.g. B=6/L=1; regenerate),
  `tests/pyo3/test_phase9_seam.py` (CfC rows in the parametrized legs + a
  `BLSTM_LID_Cell_Type cfc` mechanical leg on the mode-7 Twin fixture),
  `tests/reference_data/phase9/` (the 2 new configs + packs + manifest)
- Test: `tests/pyo3/test_phase9_seam.py`

**Interfaces:**
- Consumes: T1 cell, T2 builder; the seam file's existing parametrized structure
  (grad_check measured*10 pins under the in-test 1e-4 STOP assert, Engine round-trips,
  block probes, structural legs).
- Produces: the committed cfc fixture pair later tasks reuse (T6 parity legs).

- [ ] **Step 1**: Extend the generator (strip-list discipline: the epsilon-key lesson --
  verify no inherited `Neural_Networks_Gradient_Check_Epsilon` or `Epochs` survives in
  the new configs); regenerate with `--check` byte-reproducibility; manifest sha updated
  in-test.
- [ ] **Step 2**: RED -> measure -> pin the CfC grad_check rows (measured*10, < 1e-4);
  bit-identical Engine round-trip; a CfC STRUCTURAL leg (non-permutation-blind: zero a
  known block via its offset -> outputs must differ; the T5-phase-9 pattern); run-twice.
- [ ] **Step 3**: The LID mechanical leg: `BLSTM_LID_Cell_Type cfc` on the mode-7 Twin
  constructs, round-trips both packs, grad_checks at the pinned tolerance.
- [ ] **Step 4**: Gates (R6; the pyo3 file is CI-visible synthetic -- keep it fast).
  Commit `test(phase10): CfC seam fixtures + grad_check + LID wiring`.

---

### Task 4: the lre_sad_v2 lineage + the 8-gate matrix

**Files:**
- Create: `configs/training/lre_sad_v2.toml`, `tests/pyo3/test_phase10_gates.py`
- Modify: `configs/training/lre_sad.toml` (header pointer, comment-only --
  config_hash-safe, verified the phase-9 way), `src/python/speech/drivers/baseline.py`
  (`sad-v2` in `_ARM_CONFIGS`), `RESULTS.md` (the v2 section: gate table, pack table,
  like-for-like live-capacity table, the `sad-v2` launcher recipe rows TBD)
- Test: `tests/pyo3/test_phase10_gates.py` (corpus-gated, slow-marked)

**Interfaces:**
- Consumes: T1/T2 (cfc rows in the matrix); the phase-9 gate file as the protocol
  template (tests/pyo3/test_phase9_gates.py: _CONFIGS rows, preflight, hard legs,
  determinism, budget asserts).
- Produces: the v2 lineage + its guards; the launcher recipe
  `speech baseline sad-v2 --cell-type {lstm,slstm,mamba,cfc} --direction {bi,fwd} ...`.

- [ ] **Step 1**: Write v2 per S3.1: v1 byte-for-byte except `nnet_input_size = 11`,
  `lstm_neuron_nb = "11,24,24"`; headers per spec. Verify the diff is exactly those
  lines (+ headers).
- [ ] **Step 2**: The gate file: 8 rows {lstm,slstm,mamba,cfc} x {bi,fwd}, phase-6
  protocol verbatim (seeded, hard beat-init per collar, run-twice byte-identical,
  <10 min asserted, budget asserted). Preflight per row: init cost interior + per-file
  max (the corrected column discipline: slice as `_error_vad` does), epoch-0 grad norm
  > 0, and THE INVERSE GUARD: zero structurally-dead layer-0 input columns (assert
  every column of the layer-0 input-projection gradient block carries nonzero sum;
  cell-level dead blocks like sLSTM b_i are out of scope, excluded by block not by
  slack). Pack-length pins per cell MEASURED (record; replace survey estimates).
- [ ] **Step 3**: Run the matrix locally (corpus-gated; ~8 x <10 min budget; run rows
  sequentially -- R6). Any row failing beat-init: investigate within the phase-6 knob
  envelope; genuine failure = STOP and surface (R5; do not soften).
- [ ] **Step 4**: The v1 CfC MECHANICAL leg (S3.3): on the v1 arm, cfc constructs +
  its own PREDICTED dead-count floor (derive the cfc layer-0 dead-block arithmetic for
  the 48-of-92 dead columns -- the cell-dependent shape, the phase-9 T8 pattern) --
  no convergence gate, no param-match requirement on v1 (note it; override
  Cfc_Backbone_Units per-config only if construction demands).
- [ ] **Step 5**: RESULTS: the v2 section per S3.4 (framing notes verbatim-ish: v1 =
  the only 2015-capacity-comparable lineage; Xavier scaling measurable-not-promised).
- [ ] **Step 5**: Gates (R6; non-slow suites untouched). Commit
  `test(phase10): lre_sad_v2 lineage - 8-gate matrix + inverse guard`.

---

### Task 5: full-f32 mel -- both widen sites + the one re-pin sweep

**Files:**
- Create: `src/rust/src/fast/mel32.rs` (f32 mel bank apply + DCT + deltas/dd + SDC +
  assemble, transcribed from features/mel.rs + features/pipeline.rs with the quirk list:
  the deltas-no-DCT static-block OVERWRITE layout, SDC's unconditional n=3 kernel, the
  ignoreFirst branch-B first-column drop; banks built once at construction)
- Modify: `src/rust/src/fast/pipeline.rs` (BOTH widen sites -- build_input_sequence and
  assemble_perio_window -- route through mel32; the f64-widen path DELETED),
  `src/rust/tests/phase7_fast_pipeline.rs` + `phase7_parity_sad.rs` +
  `phase7_parity_lid.rs` + `phase9_fast_parity.rs` + `phase9_stream_causal.rs`
  (re-measured pin VALUES only -- no assertion structure changes), `RESULTS.md` (the
  re-pinned table + RSS rows)
- Test: mel32 inline units + the existing parity suites

**Interfaces:**
- Consumes: the exact features/mel.rs (byte-untouched, the transcription oracle);
  FastPipeline's bank-construction pattern.
- Produces: the final fast front-end every later twin pins against (S9.1).

- [ ] **Step 1**: Transcribe the f32 kernels op-for-op (ascending loops, same order;
  each quirk carried with a doc note naming its mel.rs source line). Inline units:
  run-twice, shape/quirk pins (ignoreFirst drop, static-block overwrite layout), a
  small-shape f32-vs-f64-exact comparison at measured tolerance.
- [ ] **Step 2**: Swap BOTH widen sites in one commit-unit; delete the widen buffers.
  Offline-vs-streamed remains bit-identical BY CONSTRUCTION (same kernel both sites) --
  the phase-8/9 streaming gates must stay green UNEDITED (their pins compare fast-vs-
  fast).
- [ ] **Step 3**: THE SWEEP: run every fast parity suite; MEASURE the new deltas;
  re-pin each at measured*10-class headroom (record old -> new in a RESULTS table);
  boundary count/type identity and argmax zero-flips MUST hold everywhere -- any flip
  is R1 STOP. Corpus tiers re-run locally (metric deltas expected to stay exactly 0.0;
  if a metric moves, STOP and adjudicate).
- [ ] **Step 4**: RSS: re-bench the phase-7 recipe rows (`speech bench --repeat=1`
  fresh-process, R6-capped build); record the plateau change (expectation ~68 -> ~56 MB,
  publish measured).
- [ ] **Step 5**: Gates (R6, ONE full cargo run). Commit
  `feat(phase10): full-f32 mel at both fast sites - re-pinned sweep + RSS`.

---

### Task 6: FastCfc causal kernel + streaming rows

**Files:**
- Modify: `src/rust/src/fast/cells.rs` (FastCfc step kernel + state {h: Vec<f32>} +
  from_flat + FastCell::Cfc arms), `src/rust/tests/phase9_fast_parity.rs` (cfc causal
  parity legs vs the exact tree on the T3 fwd fixture),
  `src/rust/tests/phase9_stream_causal.rs` (cfc rows: bit-equal gate, chunk-invariance,
  latency legs -- the cell-generic machinery grows rows only)
- Test: those suites + inline cells.rs units

**Interfaces:**
- Consumes: T1's exact forward (the transcription oracle), T3's cfc fixtures, T5's
  final front-end.
- Produces: FastCell::Cfc -- `classify_fast_shape`'s causal arm now genuinely admits
  cfc end to end (CLI + PyO3 streaming inherit with zero new surface).

- [ ] **Step 1**: Transcribe the step kernel f32 op-for-op (dot_f32 per projection,
  same term order, logistic/tanh f32 guards); state() zeroed; step-loop == wrapper,
  split-state == unsplit, run-twice -- all BIT-identical pins (the T6-phase-9 suite of
  pins, cfc rows).
- [ ] **Step 2**: Causal parity legs: exact-vs-fast posterior measured-then-pinned,
  boundary identity, max_dt exactly 0.0, output-bias crossing sweep for non-vacuity.
- [ ] **Step 3**: Streaming rows: streamed finish() bit-equal to offline fast causal,
  chunk-invariant (20/100/1000/7 ms), prefix zero-retraction, the derived latency bound
  re-checked for the cfc fixture config (same structural terms -- assert the derivation,
  measure the lag, pin bound + PUSH_CHUNK_S).
- [ ] **Step 4**: Remove any T1 interim bail for cfc-causal-fast; the bail test flips
  to construction-succeeds.
- [ ] **Step 5**: Gates (R6). Commit
  `feat(phase10): FastCfc causal kernel - parity + streaming rows`.

---

### Task 7: bidirectional f32 cell twins

**Files:**
- Create: `src/rust/src/fast/bicell.rs` (FastBiCell: fwd + reversed step loops + hcat +
  DenseRowChain; the FRESH windowed-overlap accumulate/average loop against
  window_begin/window_end; the plain whole-sequence path too -- regime follows config),
  `src/rust/tests/phase10_bicell_parity.rs`
- Modify: `src/rust/src/fast/mod.rs` (declare), `src/rust/src/fast/driver.rs`
  (classify_fast_shape: bidirectional x {slstm,mamba,cfc} -> BiCell(cell), bail removed;
  FastSpectralSegmenter construction arm)
- Test: `phase10_bicell_parity.rs` + inline units

**Interfaces:**
- Consumes: FastCell (T6 complete: Slstm/Mamba/Cfc), the shared span helpers, the T3/T5
  fixtures + front-end.
- Produces: the bi fast twins; `FastBlstm` + `overlap_window_step` BYTE-UNTOUCHED (the
  phase-7/8 suites green WITHOUT EDITS is the proof -- assert this by running them
  unmodified).

- [ ] **Step 1**: RED -- parity test: exact bi (Network<CellLayer> Direction
  bidirectional) vs FastBiCell on the T3 bi fixtures, per cell: posterior
  measured-then-pinned, boundary identity, max_dt 0.0, crossing non-vacuity; plain AND
  overlap regimes (in-test window overlays, the phase-9 pattern).
- [ ] **Step 2**: Implement FastBiCell: reversed pass = the step loop over reversed
  rows (the exact flip semantics), hcat column order matching the exact
  feed_forward_double, the fresh overlap accumulate/average loop (jj-ascending f32 +=
  order matching FastBlstm's documented convention -- same arithmetic contract, new
  code), sub-sampling via the shared sub_sample_into.
- [ ] **Step 3**: Dispatch flips + the removed-bail tests flip to construction; the
  phase-7/8 suites run UNMODIFIED and green (the byte-untouched proof); bench rows per
  cell (RTF + RSS, R6-capped). NO streaming leg -- bidirectional is unstreamable by
  construction (whole-sequence lookahead); state it in bicell.rs's module doc. The
  Twin's frozen-SAD-net conservative gate (bail_unsupported_shape) stays conservative
  with a doc note UNLESS a covering gate exists (S5's condition -- expected: it stays).
- [ ] **Step 4**: Gates (R6). Commit
  `feat(phase10): bidirectional f32 cell twins - parity pinned, FastBlstm untouched`.

---

### Task 8: forward-only LSTM fast twin

**Files:**
- Modify: `src/rust/src/fast/cells.rs` (FastCell::Lstm: per-step peephole kernel --
  12-row peephole bundle, [i|f|o|g] order, expLimit f32 guards, dot_f32 per projection
  FROM DAY ONE -- never the batched lstm_layer_forward), `src/rust/src/fast/driver.rs`
  (classify_fast_shape: (lstm, forward) -> Causal(Lstm), bail removed),
  `src/rust/tests/phase9_fast_parity.rs` + `phase9_stream_causal.rs` (lstm-fwd rows),
  `RESULTS.md` (the bench matrix n/a cell filled)
- Test: those suites

**Interfaces:**
- Consumes: the exact layers.rs per-timestep body (the oracle), FastCausalNet (generic,
  inherits), cell_overlay (writes window 0 + OutputNeuronNb for forward -- already
  cell-agnostic).
- Produces: the complete causal set {lstm, slstm, mamba, cfc}.

- [ ] **Step 1**: Transcribe the step kernel; step-loop/split-state/run-twice bit pins.
- [ ] **Step 2**: Parity + streaming rows (a seeded lstm-fwd pack via the T2-era
  builders -- init_weights already handles lstm+forward from phase 9); measured pins.
- [ ] **Step 3**: Bench: the n/a cell + the fast-tier windowed-vs-causal LSTM control
  row; RESULTS updated.
- [ ] **Step 4**: Gates (R6). Commit
  `feat(phase10): forward-only LSTM fast twin - the causal set complete`.

---

### Task 9: Mamba exact-path retention gating (the adjudicated class)

**Files:**
- Modify: `src/rust/src/nn/cells/mamba.rs` (a retain_caches flag; when false, skip the
  MambaCache fill), `src/rust/src/nn/network.rs` (a retain_layers_output flag; when
  false, skip the layers_output retention -- the ONE adjudicated touch outside cells/),
  `src/rust/src/nn/blstm.rs` (set_inference_only(bool) threading both), the bag/driver
  construction site that knows backward-will-never-run (read engine/bag_of_processors.rs
  + tasks/sad.rs: the condition is backprop active AND references present -- the
  blstm.rs:1452 clone-gate precedent; NEVER Epochs)
- Test: inline units + a new F11-seam regression leg in tests/pyo3/test_phase9_seam.py
  + goldens

**Interfaces:**
- Consumes: the survey-verified fact that MambaCache + layers_output have ZERO
  non-backward readers (re-verify with rg before implementing -- if a reader appeared,
  STOP and surface).
- Produces: inference-only exact runs skip both retentions; default (training/backward)
  behavior BYTE-identical.

- [ ] **Step 1**: Re-verify the zero-non-backward-readers claim (rg; record in report).
- [ ] **Step 2**: Implement the flags default-ON (retain) so every existing path is
  byte-identical; thread inference-only from the construction condition; a
  backward-after-no-retain forward must PANIC with a clear message (never silently
  wrong gradients).
- [ ] **Step 3**: The F11 leg: Epochs 0 + BackPropagationActivated through the seam
  still returns nonzero finite gradients (retention ON on that path -- assert). The R6
  clone note: assert the per-lane bag clones carry empty caches under inference-only.
- [ ] **Step 4**: Goldens byte-green (ONE full cargo run, R6); re-bench the exact-Mamba
  RSS row (expectation 107.8 -> ~55 MB, publish measured); RESULTS row.
- [ ] **Step 5**: Gates. Commit
  `feat(phase10): inference-only retention gating - exact-Mamba RSS closed`.

---

### Task 10: the stream-lid surface

**Files:**
- Modify: `src/rust/src/stream_cli.rs` (or a sibling `stream_lid_cli.rs` + cli.rs
  wiring -- follow the `stream`/`bench` subcommand precedent): `speech stream-lid
  <config> <input>` replaying per-utterance entries through StreamingLidSession, one
  line per utterance + a final summary; `src/rust/speech-py/src/lib.rs` (StreamingLidSession
  pyclass: new(config_path, rate, lang_index) -- read the Rust session's actual ctor
  signature and mirror it; push_utterance -> score tuple; finish -> aggregate; numpy
  copies; GIL released)
- Create: `tests/pyo3/test_phase10_stream_lid.py`
- Test: CLI smoke in the existing stream-lid Rust suite + the new pyo3 file

**Interfaces:**
- Consumes: fast/stream_lid.rs::StreamingLidSession (lib-only since phase 8; the
  phase-8 fixtures under tests/reference_data/ are the smoke inputs).
- Produces: the closed deferral -- CLI + binding, each pinned.

- [ ] **Step 1**: CLI subcommand + a spawned-binary smoke against the committed
  phase-8 stream-lid fixture (utterance lines + summary line format pinned).
- [ ] **Step 2**: PyO3 class + smokes (push/finish tuples match the Rust session on the
  same fixture; the phase-8 duplicate-catch pattern).
- [ ] **Step 3**: Gates (R6; maturin rebuild sequenced after cargo, never concurrent).
  Commit `feat(phase10): stream-lid CLI + PyO3 binding - the deferral closed`.

---

### Task 11: mutation battery

**Files:** `IMPROVEMENTS.md` only ("Mutation battery (Phase 10)"; tree clean between
cycles; FOREGROUND; R6 caps on every rebuild).

- [ ] The list (apply-FAIL-revert-PASS): (1) CfC gate inversion (swap `1-g`/`g`) -> FD
  + forward pins; (2) backbone block-order swap in init_cfc_flat -> the reconstruction
  pins; (3) v2 regression (nnet_input_size 11 -> 23 in v2) -> the zero-dead-columns
  inverse guard (corpus-gated catcher, run it); (4) mel32 ignoreFirst-branch drop ->
  the re-pinned parity legs; (5) bi-twin reverse-pass skip (both halves forward) ->
  phase10_bicell_parity; (6) fwd-LSTM batched-projection substitution -> streaming
  bit-identity legs (the threading-vs-arithmetic doctrine says parity owns arithmetic
  -- verify the doctrine: the streaming gate alone should NOT catch it if the
  substitution is applied to BOTH offline and streamed sides; record what actually
  catches); (7) retention flag inverted -> the F11 leg + goldens; (8) stream-lid
  session state leak across utterances -> the prefix/aggregate legs. Honest gaps
  recorded.
- [ ] Full final pass (R6: one full cargo + non-slow + per-file slow + lint,
  sequenced). Commit `test(phase10): mutation battery results recorded`.

---

### Task 12: docs refresh

**Files:** `README.md`, `CLAUDE.md`, `RESULTS.md`, `IMPROVEMENTS.md` consistency.

- [ ] CLAUDE.md: new rows (nn/cells/cfc.rs, fast/mel32.rs, fast/bicell.rs); updated
  rows (cells/mod, blstm, toml_config KEY_TABLE count -- COUNT it, fast/cells,
  fast/driver, fast/pipeline, network.rs retention class, stream_cli, speech-py,
  init_weights, config_bridge, baseline incl. sad-v2); the Roadmap 2 Phase 10 bullet
  done-with-numbers; the "Since Phase 10" convention bullet if warranted (candidate:
  the completed (cell x direction) fast matrix as the new invariant + R6 as standing).
  README: phase bullet, v2 launcher recipe, stream-lid usage line. Every claim
  code-verified (re-derive, never transcribe -- the phase-9 lesson); stale-number
  sweeps BY GREP with per-hit adjudication.
- [ ] Gates + commit `docs(phase10): README/CLAUDE.md/RESULTS refresh - phase 10 lands`.

---

### Task 13: final review + smart-commit + finish

- [ ] Whole-branch final review (the sdd final-review pattern: package the branch diff,
  fresh reviewer on the most capable model, R6-rationed compile budget, fix-now items
  resolved before proceeding).
- [ ] Invoke the `smart-commit` skill, telling it to take the WHOLE git branch into
  account.
- [ ] Invoke `superpowers:finishing-a-development-branch` (verify tests R6-capped,
  present options; the user's standing choice is keep-branch-as-is; NEVER push).
