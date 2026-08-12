# Phase 11 -- Transformer Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the fifth `CellLayer` variant -- a pre-norm transformer block over a causal
sliding window with ALiBi positions -- end to end: exact f64 cell + FD/seam gradient
tiers, config keys, Python init + dual-lineage sizing, both f32 fast twins, the fifth
streaming row, day-one retention gating, the 10-gate `lre_sad_v2` matrix, battery, docs.

**Architecture:** The cut is at the layer (the phase-9 seam): a new enum variant + a
`Layer` impl inside `nn/cells/`, three `KEY_TABLE` rows, a Python init builder emitting
the same flat order, and per-cell arms in the fast tree. One early behavior-free task
consolidates the three duplicated `fast/cells.rs`/`fast/bicell.rs` fragments before the
fifth cell multiplies them. Spec:
`docs/superpowers/specs/2026-08-12-phase-11-transformer-design.md` (SOURCE-GOVERNS: where
this plan and the spec disagree, the spec wins; where either disagrees with the tree,
read the tree and say so in the report).

**Tech Stack:** Rust (edition 2024, ndarray exact tree, hand-rolled f32 fast tree),
PyO3/`speech_rs`, Python 3.14 + uv, pytest, the phase-9 FD/seam harnesses reused.

## Global Constraints (every task; copy into every dispatch)

- R6 MEMORY PROTOCOL (two host reboots on record): every cargo invocation `-j 4`; every
  test execution `-- --test-threads=4`; NEVER two compile-heavy commands concurrently
  (no cargo+maturin overlap); targeted single-file test runs while iterating; at most
  ONE full-suite run per task; ALL commands FOREGROUND (never `run_in_background`, never
  a Monitor; never read exit codes through a pipe -- `cmd > log 2>&1; echo $?`).
- EXACT-TREE DISCIPLINE (spec R2): outside `nn/cells/` the ONLY sanctioned exact-tree
  touches are the `CellType`/enum arms, the `KEY_TABLE` rows, and the transformer
  retention arm. The committed golden suite byte-green is the proof, checked before
  every commit.
- MEASURE-THEN-PIN (R3/R4): eps by sweep; pins at measured*10; every gradient pin under
  the `1e-4` STOP; decision flips (boundary/argmax) STOP the task (R1) -- adjudicate,
  never widen.
- IMPROVEMENTS.md UNTOUCHED: port-only divergence lives in module docs + RESULTS.md.
- Git: never commit to main; this branch is `feature/phase-11-transformer`; never push;
  never use `gh`. Commit trailer EXACTLY:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- ASCII only in all authored text (plain hyphens, straight quotes).
- License hygiene: nothing from `data/LRE03-LRE07/` or `data/Scoring_LRE15/` in any
  tracked file; corpus tests behind `requires_corpus`/`corpus_root_or_skip`.
- Python via `uv run`; lint gates `cargo fmt --all -- --check` +
  `cargo clippy -j 4 --all-targets --all-features` (zero warnings) per task.

---

### Task 1: The cells/bicell dedupe (behavior-free, suite-arbitrated)

**Files:**
- Modify: `src/rust/src/fast/cells.rs` (host the shared fragments)
- Modify: `src/rust/src/fast/bicell.rs` (consume them)
- Test: NO test file edits -- the arbiter is the existing suite UNEDITED.

**Interfaces:**
- Consumes: the already-shared `cell_weight_count`, `build_cell`, `cell_stack_forward`,
  `DenseRowChain`, `window_begin`/`window_end` (all in place since phase 10).
- Produces: shared helpers for the THREE fragments RESULTS.md:2018 names as duplicated
  (~55 lines): (a) the `element_count` walk (`FastBiCell::element_count` at
  `bicell.rs:126` vs the causal twin's), (b) the dense-MLP construction inside the two
  `from_flat`s, (c) the per-row output drive. Suggested shapes (implementer may adjust
  names, MUST keep `pub(crate)` scope):
  `fn stack_element_count(spec: &NnetSpec, cell: CellType, stacks: usize) -> usize`,
  `fn build_dense_tail(flat: &mut &[f64], sizes: &[usize], fan_in: usize) -> Vec<FastDenseLayer>`,
  `fn drive_output_rows(chain: &mut DenseRowChain, ...) -> ...` -- read both copies
  first; extract ONLY what is token-for-token the same arithmetic.

- [ ] **Step 1: Survey.** Read both files' copies of the three fragments side by side.
  Write the survey into the task report: for each fragment, IDENTICAL / DIVERGENT and
  why. A fragment with any arithmetic divergence stays duplicated (spec S4: partial
  dedupe acceptable, silent drift is not).
- [ ] **Step 2: Extract.** Pure code motion into `cells.rs` `pub(crate)` helpers; both
  callers consume them. Zero reorder/hoist/fusion: every reduction stays the ascending
  contiguous loop it was. `FastBlstm`/`overlap_window_step` (`fast/nn.rs`) BYTE-UNTOUCHED
  (`git diff --stat` must not list `fast/nn.rs`).
- [ ] **Step 3: Gates.** Run, R6-capped, each: `cargo test -j 4 --test
  phase9_fast_parity --test phase10_bicell_parity --test phase9_stream_causal --test
  phase8_gate --test stream_incremental_resmooth -- --test-threads=4`. ALL green with
  ZERO edits to any test file, at existing pin values.
- [ ] **Step 4: fmt + clippy** (commands in Global Constraints), zero warnings.
- [ ] **Step 5: Commit** `refactor(fast): dedupe cells/bicell scaffolding -- element
  count, dense tail, output drive` (+ trailer).

### Task 2: The exact cell + config keys + FD tier

**Files:**
- Create: `src/rust/src/nn/cells/transformer.rs`
- Modify: `src/rust/src/nn/cells/mod.rs` (enum variant + match delegation + the
  `set_retain_cache` arm as a no-op THIS task; Task 8 makes it gate)
- Modify: `src/rust/src/nn/blstm.rs` (`CellType::Transformer`, `TransformerParams`,
  `from_legacy` reads + validation, `from_config` exhaustive-match arm)
- Modify: `src/rust/src/toml_config.rs` (three `KEY_TABLE` rows, section
  `[nn_transformer]`; count comment 187 -> 190)
- Modify: `src/rust/src/fast/cells.rs` + `src/rust/src/fast/driver.rs` (INTERIM ARMS
  ONLY -- see the ordering note below)
- Test: `src/rust/tests/phase9_cell_grad.rs` (transformer legs), unit tests inside
  `transformer.rs`, `src/rust/tests/phase9_cell_config.rs`-style decode pins.

**ORDERING NOTE (compile-enforced totality vs task order):** adding the
`CellType::Transformer` variant makes every exhaustive fast-tree match
(`cell_weight_count`, `build_cell`, `classify_fast_shape`, and any sibling the compiler
names) a COMPILE ERROR until kernels exist. THIS task adds those arms as LOUD
`unimplemented!("transformer fast twin lands in phase-11 T5/T6")` panics -- compiling,
unreachable by any committed config, each REPLACED by Tasks 5/6. None may survive to
branch end (Task 11 verifies `rg -n "unimplemented" src/rust/src/fast` is clean).

**Interfaces:**
- Produces: `TransformerParams { window: usize, heads: usize, d_ff: usize }` with
  `TRANSFORMER_DEFAULT_WINDOW: usize = 64`, `TRANSFORMER_DEFAULT_HEADS: usize = 4`,
  `TRANSFORMER_DEFAULT_D_FF: usize = 64` (PROVISIONAL -- Task 3's sizing confirms or
  corrects this ONE constant in both languages); `TransformerLayer::new(input_size,
  output_size, params: &TransformerParams)`; `nb_of_weights()`; the S1.2 flat order
  `[W_a | b_a | g_1 | W_qkv | b_qkv | W_o | b_o | g_2 | W_1 | b_1 | W_2 | b_2]` walked
  ONCE by `for_each_slot` (copy the CfC walk pattern); keys `Transformer_Window`,
  `Transformer_Heads`, `Transformer_D_Ff` (rows on the `("Cfc_Backbone_Units",
  "nn_cfc", "backbone_units")` template at `toml_config.rs:212`).
- Validation in `TransformerParams::from_legacy`: `window >= 1`, `heads >= 1`,
  `d_ff >= 1`, and `output_size % heads == 0` checked at `TransformerLayer::new`
  (a non-dividing head count is a hard error naming both numbers).

- [ ] **Step 1: Forward, RED-first where feasible.** Implement S1.1 exactly: adapter ->
  RMSNorm_1 (copy the Mamba RMSNorm op order, eps 1e-5) -> combined `W_qkv` ->
  per-head windowed logits `(q_t . k_j)/sqrt(d) + m_h*(j - t)` over
  `j in [max(0, t-W+1), t]`, ALiBi slopes `m_h = 2^(-8h/A)` for `h = 1..A` computed
  once at construction -> softmax ALWAYS max-subtract (S1.3, ascending loops) ->
  weighted-v fold -> `s = x' + W_o attn + b_o` -> RMSNorm_2 -> FFN with `silu` (reuse
  the Mamba transcription) -> `h = s + ffn`. Backward caches: per-row attention
  weights, `u`, `s`, `v_t` (RMSNorm_2 output), FFN pre-activations.
- [ ] **Step 2: Backward, hand-derived** (S1.5): FFN adjoints, RMSNorm backward (mirror
  the Mamba derivation shape), residual fan-ins, per-row softmax Jacobian
  `dp = p .* (dl - sum(p .* dl))`, the `1/sqrt(d)` scale, and the reverse scatter (a
  `(k_j, v_j)` receives contributions from every `t >= j` with `t - j < W`; fold
  reverse-time like Mamba's conv fold). ALiBi contributes NO parameter gradient.
- [ ] **Step 3: Dead-block unit pins** in `transformer.rs`: `b_k` gradient EXACTLY 0.0
  at T in {1, 3, 7} (dead ALWAYS, S1.4 -- softmax shift-invariance); q/k rows of
  `W_qkv` + `b_q` EXACTLY 0.0 at `T = 1` with a `T = 2` NONZERO contrast.
- [ ] **Step 4: FD tier.** Extend `tests/phase9_cell_grad.rs` on the mamba pattern
  (geometry beyond `(in, out)`): fixture `W = 4`, heads 2, `d_ff = 8`; shapes
  `(t, in, out)` in `{(1,3,2), (2,3,2), (3,3,4), (7,5,4), (23,7,6)}` x 3 seeds (t
  spans both `t <= W` and `t > W` regimes); MEASURE eps by the existing `sweep`
  (8-point), then pin per-shape at measured*10; `max_rel_major` must sit under `1e-4`
  (STOP if not -- adjudicate the derivation, do not widen); resolvable-count asserts
  carrying the S1.4 dead counts (predicted == measured).
- [ ] **Step 5: Config pins.** Decode pins: absent keys -> defaults (a pre-phase-11
  config decodes byte-unchanged); present-invalid -> error; `[nn_transformer]`
  round-trips through `--convert-config`.
- [ ] **Step 6: Targeted tests** (`cargo test -j 4 --test phase9_cell_grad -- --test-threads=4`
  + the new unit tests + `cargo test -j 4 toml -- --test-threads=4`), then ONE full
  `cargo test -j 4 -- --test-threads=4` (this task's full-suite run) -- committed
  goldens must be byte-green (R2 proof).
- [ ] **Step 7: fmt + clippy; commit** `feat(nn): windowed causal attention cell
  (ALiBi) -- the fifth CellLayer variant` (+ trailer).

### Task 3: Python init builder + the dual-lineage sizing

**Files:**
- Modify: `src/python/speech/init_weights.py` (`transformer_geometry`,
  `init_transformer_flat` -- mirror `cfc_geometry`/`init_cfc_flat` at :294/:306)
- Modify: `src/python/speech/config_bridge.py` (`_transformer_geometry` +
  `TRANSFORMER_DEFAULT_WINDOW/HEADS/D_FF` mirrored constants, the `_cfc_geometry`
  pattern at :64; `nnet_spec` gains the `"Transformer"` slot)
- Test: `tests/test_phase11_init.py` (new), extending the phase-9/10 init-test patterns.

**Interfaces:**
- Consumes: Task 2's flat order + `TransformerParams` defaults.
- Produces: `transformer_geometry(spec) -> tuple[int, int, int]` (window, heads, d_ff);
  `init_transformer_flat(spec, rng, scheme, forget_bias_one) -> list[NDArray]` (the
  signature shape of `init_cfc_flat`; `forget_bias_one` accepted-and-ignored for
  call-site uniformity, documented); THE SIZED `d_ff` default, corrected in BOTH
  `blstm.rs` and `config_bridge.py` if it moves off 64.

- [ ] **Step 1: The sizing (spec S3).** Re-derive the closed form per lineage from the
  S1.2 formula summed over each stack + output MLP + normalize tail (v2: layers
  (44->24), (24->24); v1: (92->24), (24->24); bidirectional = 2 stacks + the 48-input
  output MLP; forward = 1 stack + 24-input). Compute the v2 argmin vs pack 24431 and
  v1's +-15% band vs 33671; choose per the spec rule (inside BOTH if satisfiable, else
  v2 wins, conflict RECORDED). Record both closed forms + the chosen `d_ff` in the
  task report AND as test asserts. If the choice is not 64, update
  `TRANSFORMER_DEFAULT_D_FF` in `blstm.rs` + `config_bridge.py` in THIS task.
- [ ] **Step 2: Builders.** Xavier/He per block (fans from block shapes), `b_k` seeded
  0 (S1.4), RMSNorm gains 1.0, other biases 0. Emit the Rust flat order DIRECTLY.
- [ ] **Step 3: The reconstruction pin + shears.** Whole-pack block-by-block
  reconstruction (independent in-test transcription of every block boundary), plus TWO
  shear companions: a q/k-family permutation and a deeper-FFN variant; each
  self-mutation-probed (temporarily mis-order two blocks in a LOCAL copy -> the pin
  must fail; both schemes parametrized -- He's asymmetric fans make fans pinnable).
- [ ] **Step 4: Cross-language pins.** Pack-length equality vs Rust
  (`speech_rs.Engine` round-trip bit-identical on a tiny synthetic config, the
  phase-9/10 pattern); the defaults parse-pin against the Rust source (copy the CfC
  `test_the_rust_and_python_defaults_agree` mechanism for all three constants).
- [ ] **Step 5:** `uv run pytest tests/test_phase11_init.py -v` green;
  `./lint_code.sh` green. Commit `feat(python): transformer init builder + the sized
  d_ff (dual-lineage)` (+ trailer).

### Task 4: Seam fixtures + LID wiring

**Files:**
- Create: committed synthetic fixture configs under
  `tests/reference_data/phase9/seam/` (the existing naming scheme) for
  `{transformer} x {bi, fwd}` SAD + one mode-7 Twin with
  `BLSTM_LID_Cell_Type transformer`
- Test: `tests/pyo3/test_phase9_seam.py` (extend the parametrization)

**Interfaces:**
- Consumes: Tasks 2-3 (cell + builder). Produces: the corpus-level `grad_check`
  evidence the exit gate requires; the LID mechanical wiring proof.

- [ ] **Step 1:** Author the three fixtures on the existing seam-fixture template
  (tiny nets, synthetic listings, no corpus tokens).
- [ ] **Step 2:** Extend the seam test's case list; run
  `uv run pytest tests/pyo3/test_phase9_seam.py -v -k transformer` FOREGROUND
  (maturin build first if needed -- NEVER concurrent with cargo). MEASURE the
  grad_check max-rel per fixture, pin at measured*10, in-test assert every pin
  `<= 1e-4` (R4 STOP).
- [ ] **Step 3:** Engine weight round-trip bit-identity pins (the existing leg,
  extended). Commit `test(seam): transformer seam fixtures + LID wiring pins`
  (+ trailer).

### Task 5: FastTransformer -- the causal f32 twin

**Files:**
- Modify: `src/rust/src/fast/cells.rs` (`TransformerState` -- the KV ring --,
  `FastTransformer`, `FastCell::Transformer` + `FastCellState::Transformer` arms,
  `cell_weight_count`/`build_cell` arms)
- Modify: `src/rust/src/fast/driver.rs` (`classify_fast_shape`:
  `(transformer, forward) -> Causal(Transformer)`; the bidirectional arm lands in
  Task 6 -- until then it may return `BiCell(Transformer)` only if Task 6's kernel
  exists, so THIS task keeps the exhaustive match compiling by pointing the
  bidirectional row at `BiCell(Transformer)` and Task 6 supplies the kernel; if that
  ordering is uncomfortable, swap Tasks 5/6's driver edits -- the compile-enforced
  totality must hold at EVERY commit boundary)
- Test: `src/rust/tests/phase9_fast_parity.rs` (transformer causal legs, the cfc-leg
  template: in-test seeded packs, output-bias `Lever` crossing sweep gain-1-first)

**Interfaces:**
- Consumes: Task 2's exact cell (the parity oracle), Task 1's deduped scaffolding.
- Produces: `FastTransformer::step(&self, x_t: &[f32], state: &mut TransformerState)`
  with `TransformerState` = per-head KV ring (capacity `W`, oldest-evicted) --
  construct via `state()` only (the Mamba private-scratch precedent);
  `from_flat` narrowing f64->f32 once post-pack.

- [ ] **Step 1: Transcribe op-for-op** from `TransformerLayer::feed_forward` (S5.2),
  REPLACING Task 2's `unimplemented!` interim arms in `cell_weight_count`/`build_cell`
  (Task 6 replaces any bicell-side remainder):
  per-step `dot_f32` everywhere INCLUDING q.k and the weighted-v fold (NO faer inside
  the step -- restate the battery-item-6 rule in the module doc); f32 exp guard at
  `ln(f32::MAX)`; softmax max-subtract identical to exact; ring shorter than `W` at
  start == offline edge-truncation.
- [ ] **Step 2: Contract pins** (the phase-9 set): wrapper == step-loop bit-identical;
  split-state == unsplit; run-twice; weight count vs the exact cell; a ring-eviction
  pin (a `T > W` sequence where evicting the WRONG row moves the output -- assert the
  correct one, measured non-vacuous).
- [ ] **Step 3: Parity legs** (`phase9_fast_parity.rs`): boundary count/types
  IDENTICAL, `max_dt` EXACTLY 0.0, argmax zero flips; posterior `max_rel` MEASURED
  then pinned (expect the existing `1.0e-4` causal pin to hold; if measured*10
  exceeds it, STOP and adjudicate); crossing legs non-vacuous via the `Lever` ladder,
  gain-1-first so sibling rows stay byte-identical.
- [ ] **Step 4:** Targeted runs (`cargo test -j 4 --test phase9_fast_parity --
  --test-threads=4` + cells unit tests); fmt + clippy; commit `feat(fast): causal f32
  transformer twin -- KV-ring step kernel` (+ trailer).

### Task 6: The bidirectional arm

**Files:**
- Modify: `src/rust/src/fast/bicell.rs` (the `Transformer` arm -- post-Task-1 this is
  the enum arm + element-count row, no fresh scaffolding)
- Modify: `src/rust/src/fast/driver.rs` (finalize both transformer rows of
  `classify_fast_shape`)
- Test: `src/rust/tests/phase10_bicell_parity.rs` (transformer x {plain, overlap} x
  {base, crossing} legs on the existing template)

**Interfaces:** Consumes Task 5's kernel (`cell_stack_forward(.., reverse = true)`
drives it for the reverse stack -- no transformer-specific bicell code).

- [ ] **Step 1:** Enum arms + element count via Task 1's shared walk.
- [ ] **Step 2:** Parity legs: boundary identity, `max_dt` 0.0, measured-then-pinned
  posteriors (the bicell tier's `2.0e-4` pin is the ceiling; STOP if measured*10
  exceeds); the overlap-vs-plain distinctness guard extended; crossing sweep with the
  two-knob ladder as slstm needed.
- [ ] **Step 3:** `cargo test -j 4 --test phase10_bicell_parity -- --test-threads=4`;
  fmt + clippy; commit `feat(fast): bidirectional transformer arm -- the matrix stays
  total` (+ trailer).

### Task 7: The fifth streaming row

**Files:**
- Test: `src/rust/tests/phase9_stream_causal.rs` (the transformer row; ZERO
  `src/fast/stream.rs` machinery edits expected -- `StreamCausal` drives
  `FastCell`/`FastCellState` generically; if an edit IS needed, STOP and report why
  before making it)

**Interfaces:** Consumes Tasks 5's kernel + the existing gate machinery.

- [ ] **Step 1:** Add the transformer fixture config (cell keys only differ -- the
  phase-10 precedent) + the crossing sweep to manufacture `>= 2` committed interior
  boundaries (aim for the rider-9 sibling-strength floors; record the measured count).
- [ ] **Step 2:** The gate legs: streamed `finish()` bit-for-bit vs the offline fast
  causal run, `max_dt` EXACTLY 0.0, posteriors `to_bits`-equal, chunk-invariant at
  20/100/1000/7 ms, prefix zero retractions; the latency legs at the UNCHANGED
  `1.73400 s` bound + `PUSH_CHUNK_S` (no lookahead in a causal windowed cell).
- [ ] **Step 3:** `cargo test -j 4 --test phase9_stream_causal -- --test-threads=4`
  green; ALSO re-run `phase8_gate` + `stream_incremental_resmooth` (must be untouched
  green). Commit `test(stream): transformer streaming row -- bit-equal, fifth cell,
  bound unchanged` (+ trailer).

### Task 8: Retention gating day one + bench rows

**Files:**
- Modify: `src/rust/src/nn/cells/transformer.rs` + `nn/cells/mod.rs` (the
  `set_retain_cache` transformer arm GATES: when not retaining, release the attention
  weights + intermediates after forward; backward panics LOUDLY through the existing
  choke point if caches were dropped)
- Test: unit pins in `transformer.rs`; a bench row measurement (local, printed).

**Interfaces:** Consumes Task 2's cell; the phase-10 T9 mechanism
(`BlstmNetwork::set_inference_only` already fans to `CellLayer::set_retain_cache`).

- [ ] **Step 1:** Gate the caches; unit-pin: forward-only run with retention off ->
  backward panics with the named message; retention on -> backward bit-identical to a
  never-gated run.
- [ ] **Step 2:** Measure exact-tree peak RSS with/without backprop on a synthetic
  long-T transformer config via `speech bench` (R6: one run each); record both numbers
  for RESULTS.md (Task 11 lands them).
- [ ] **Step 3:** Full-suite run if not already spent this task; fmt + clippy; commit
  `feat(nn): transformer retention gating -- attention cache released inference-only`
  (+ trailer).

### Task 9: Baseline knob + the 10-gate matrix

**Files:**
- Modify: `src/python/speech/drivers/baseline.py` (`transformer` joins the
  `cell_overlay` choices at :391 and the CLI `--cell-type`; the overlay writes
  `BLSTM_Cell_Type transformer` + the same forward-forced derived keys)
- Test: `tests/pyo3/test_phase10_gates.py` (two `_CONFIGS` rows:
  `transformer x {bidirectional, forward}` with pack lengths measured via
  `len of init_transformer_flat` and pinned as literals)

**Interfaces:** Consumes Tasks 2/3/5/6 (exact cell through both twins -- the gates run
the exact tree; the knobs must produce configs the engine builds).

- [ ] **Step 1:** The knob + overlay; a config-hash pin that the DEFAULT run's config
  text is byte-unchanged (the phase-9 mechanism).
- [ ] **Step 2:** The two gate rows: HARD beat-init at every collar, first-run,
  run-twice byte-identical, the zero-dead-columns inverse guard unchanged. Corpus-gated
  (`requires_corpus`), local-only, run FOREGROUND per-file:
  `uv run pytest tests/pyo3/test_phase10_gates.py -v -k transformer`. Record wall
  times + DCF numbers for RESULTS.
- [ ] **Step 3:** `./lint_code.sh`; commit `feat(baseline): transformer cell knob + the
  10-gate sad-v2 matrix` (+ trailer).

### Task 10: Mutation battery

**Files:** ephemeral mutations (apply -> observe -> REVERT each; tree clean after);
report appended to the task ledger.

- [ ] Run >= 8 items, each naming its expected catcher FIRST, then measured: (1) ALiBi
  slope sign flip -> FD tier / parity; (2) window off-by-one (`t-W` for `t-W+1`) ->
  exact-vs-fast parity or the FD window-edge shapes; (3) f32 softmax max-subtract
  removed -> the f32 exp guard pin / parity under the Lever's large-logit leg; (4)
  KV-ring eviction-order swap -> Task 5's eviction pin; (5) `b_k` made live (add a
  spurious gradient path) -> the dead-block pin; (6) reconstruction-pin shear (swap
  `W_1`/`W_2` emission order in a LOCAL builder copy) -> Task 3's pin; (7) re-duplicate
  one deduped fragment WITH a one-ULP divergence -> the parity legs at existing pins;
  (8) `T = 1` q/k-dead contrast break (skip the window clamp at t=0) -> the T=2
  contrast / FD counts. Honest gaps RECORDED with the phase-9/10 doctrine language
  (self-consistency pins threading; parity owns decisions + arithmetic at pin
  resolution).
- [ ] Commit the battery report notes with the docs task if any doc line changes;
  otherwise ledger-only.

### Task 11: Docs refresh

**Files:** `CLAUDE.md` (the `nn/cells/*`, `fast/cells.rs`, `fast/bicell.rs`,
`fast/driver.rs`, `fast/stream.rs` rows; the KEY_TABLE count 187 -> 190 in the Locked
decisions bullet; a Roadmap 2 Phase 11 bullet), `README.md` (if it lists cells),
`RESULTS.md` (the phase-11 section: FD/seam pins, parity numbers, streaming row,
retention RSS, gate numbers, sizing closed forms + the d_ff decision, battery table,
follow-ons), memory + ledger.

- [ ] Sweep BARE NUMERALS tree-wide for every stale count this phase moves (187, the
  8-gate matrix, "four cells", 155-254 stays); per-hit adjudication (process lesson
  27/31). Cite symbols, not line numbers, where possible (lesson 32).
- [ ] Commit `docs: phase 11 refresh -- five cells, 190 keys, the 10-gate matrix`
  (+ trailer).

### Task 12: Final review + smart-commit + finishing

- [ ] Whole-branch final review (fresh reviewer, the house two-verdict contract) over
  `main..HEAD`; fix wave if needed.
- [ ] Invoke the `smart-commit` skill, telling it to take the WHOLE git branch into
  account (per the user's global planning rule).
- [ ] ONE R6-capped full verification (`cargo test -j 4 -- --test-threads=4` +
  `uv run pytest tests -x -q` non-slow + the pyo3 file) -- counts recorded.
- [ ] Invoke `superpowers:finishing-a-development-branch`; present the standard menu
  (the user handles push/PR themselves -- never push, never gh).
