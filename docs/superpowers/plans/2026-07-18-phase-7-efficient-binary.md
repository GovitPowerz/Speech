# Phase 7: the efficient binary (offline) - implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runtime-selectable f32 fast inference path (faer matmuls, realfft FFT,
preallocated workspaces) beside the byte-untouched exact f64 path, proven quality-parity
against it on the committed 2015 packs + the phase-6 subset checkpoints, with a
measure-then-pin RTF/peak-RSS benchmark suite.

**Architecture:** Approach A from the spec (docs/superpowers/specs/
2026-07-18-phase-7-efficient-binary-design.md - READ IT FIRST, S0-S4 govern): a new
`src/rust/src/fast/` module tree (`pipeline` / `nn` / `driver`) sharing the config,
weight-pack, and f64 decision/serialization layers with the exact path; a new
`Inference_Path` config key (default `exact`) dispatched at bag construction; the exact
tree is UNTOUCHABLE except the single dispatch site and Task 8's golden-re-verified
neutral hoists.

**Tech Stack:** Rust (edition 2024), `faer` (pure-Rust SIMD matmuls), `realfft` (real
FFT, already declared), `criterion` (micro-benches), `libc` (ru_maxrss). No system BLAS.

## Global Constraints

- The EXACT TREE IS UNTOUCHABLE: no file under `features/`, `nn/`, `tasks/`, `audio.rs`,
  `cost.rs` changes behavior; the only exact-tree edits are Task 4's single dispatch site
  in `engine/bag_of_processors.rs`, Task 1's bench plumbing outside the compute path, and
  Task 8's hoists (each verified byte-identical on the FULL golden suite).
- Fast-path numeric divergence from exact is BY DESIGN: documented in module docs +
  RESULTS.md, never IMPROVEMENTS.md (spec S4). No contorting the fast path toward
  bit-parity (spec R3).
- Measure-then-pin: every tolerance and budget is MEASURED first, pinned with headroom,
  direction-safe; a knife-edge boundary flip is a finding to adjudicate openly (spec R1).
- Corpus gating exactly as phase 6: `requires_corpus`/`corpus_root_or_skip`, CI sees
  committed fixtures only; the license bright line holds (no real corpus identifier or
  per-file content-derived value in tracked files; runtime sorted-first selection).
- FOREGROUND everything; never run_in_background/Monitor; never read exit codes through
  pipes; slow pyo3/corpus runs per-file/per-test (the phase-6 pattern).
- ASCII only; cargo fmt/clippy -D warnings/mypy/ruff clean; commit trailer exactly
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; never push, never gh, never
  main; `.superpowers/` stays untracked.
- CI runs no corpus, no perl, no Octave: every new CI-visible test uses committed
  fixtures only; bench assertions in CI are wide-headroom smokes (spec R5).

---

### Task 1: the bench harness + the exact-path baseline

**Files:**
- Create: `src/rust/benches/kernels.rs` (criterion micro-benches)
- Create: `src/rust/src/bench.rs` (the `speech bench` end-to-end harness)
- Modify: `src/rust/src/cli.rs` (+ `main.rs`, `lib.rs`) - the `bench` subcommand
- Modify: `src/rust/Cargo.toml` (criterion dev-dep + `[[bench]]`, `libc` dep)
- Modify: `RESULTS.md` (the phase-7 baseline table)
- Test: `src/rust/tests/phase7_bench.rs`

**Interfaces:**
- Produces: `speech bench [--repeat=N] [--path=exact] <config>` - runs the corpus config
  end to end (the `-i` image-mode compute path, unscored), prints ONE parseable line per
  run: `BENCH path=exact wall_s=<f> audio_s=<f> rtf=<f> maxrss_mb=<f> files=<n>`;
  `audio_s` = the summed per-file processed durations (min(file duration, Audio_max_duration));
  `maxrss_mb` via `libc::getrusage(RUSAGE_SELF)` (documented: bytes on macOS, KB on
  Linux, normalized to MB). `--path` accepts only `exact` until Task 4 wires `fast`
  (unknown value = usage error).
- Produces: `bench.rs::run_bench(configs, repeat, path) -> BenchReport` (pub, consumed by
  the CLI + tests); criterion targets `matmul_seq_92x96`, `gfft_1024`, `mel_apply_513x20`
  named for Task 7's fast twins.
- CLI note: `bench` is a PORT-ONLY subcommand (the `--convert-config` precedent),
  doc-commented as such; the legacy flag modes are untouched.

- [ ] RED: `phase7_bench.rs` - `bench_line_parses` (spawn the binary: `speech bench
  --repeat=1 tests/reference_data/phase4a/tier2_spectral.config` staged into a tempdir
  with the committed 60 s `phase4d/prcts_excerpt.wav` and the tuple-A pack; assert the
  BENCH line parses, `rtf > 0`, `audio_s` within 1% of 120.0 (60 s x 2 ch), maxrss sane)
  + `bench_rejects_unknown_path`. Run: FAIL (no subcommand).
- [ ] Implement `bench.rs` + the CLI wiring + the criterion bench file (micro-benches
  call the EXISTING exact kernels on fixture-shaped inputs; no fast path yet).
- [ ] GREEN + `cargo bench -- --test` (criterion smoke, no timing assertions).
- [ ] Measure the baseline: the 60 s fixture (CI workhorse) + CORPUS-GATED one real 600 s
  wav (runtime-selected, never named) x {SAD tier2 config, twin_mode7 config}; record
  wall/RTF/maxrss into RESULTS.md's new "Phase 7 - performance" table (hardware named:
  Apple Silicon dev box; the session's scouting numbers RTF ~0.004 / ~0.6 MB per
  audio-second are the sanity cross-check).
- [ ] Gates: cargo test + fmt + clippy; uv run pytest tests -q -m "not slow" (untouched).
- [ ] Commit `feat(phase7): bench harness - criterion micros + speech bench RTF/RSS + the exact baseline`.

### Task 2: fast::nn - the f32 BLSTM forward core

**Files:**
- Create: `src/rust/src/fast/mod.rs`, `src/rust/src/fast/nn.rs`
- Modify: `src/rust/Cargo.toml` (add `faer`)
- Test: `src/rust/tests/phase7_fast_nn.rs`

**Interfaces:**
- Consumes: `config::NnetSpec` (the existing spec extraction) and the flat f64 pack
  layout (`nn/blstm.rs::set_weights` order - the packer contract, adim ALREADY APPLIED
  by the existing config-reader path; cite the layout source in module docs).
- Produces: `fast::nn::FastBlstm` with `from_flat(spec: &NnetSpec, flat: &[f64]) ->
  Result<FastBlstm>` (narrows f64->f32 ONCE, after adim; SoA per-gate blocks
  `[i|f|o|g]`), `feed_forward(&mut self, input: &FastMatrix) -> &FastMatrix` (posteriors,
  frames x classes; `FastMatrix` = the tree's row-major f32 buffer type, `fast::nn::
  FastMatrix { data: Vec<f32>, rows, cols }`), `workspace` preallocated at construction
  from the config's max sequence hint and grown geometrically if exceeded (never
  per-frame). Batched input projection via `faer::linalg::matmul` (whole-sequence
  frames x I * I x 4O), per-timestep recurrence as a faer matvec into the workspace;
  fwd + reverse passes + HCAT + the dense/softmax output MLP; SubSample (integer-floor,
  matching exact); the f32 transcriptions of `Maxmin2`/`Identity` (asinh),
  `GatesFunction` (sigmoid(0.1z) with the exact expLimit guards), the 700.0-guarded
  softmax. FORWARD ONLY - no backward, no trainer.
- Produces (for Tasks 4/5): the width-tolerance crop (input cols > I -> topRows crop,
  matching `layers.rs:181-187` semantics) and peephole handling per the exact layout
  (cell rows narrow, no peepholes - the parity hazard list governs).

- [ ] RED: synthetic-net tolerance tests - build a small exact `BlstmNetwork` (the
  test-support hooks / the phase-2 fixture pattern) and a `FastBlstm` from the SAME flat
  pack; on random f64 inputs (seeded), assert per-frame posterior max relative delta
  under a PLACEHOLDER 1e-2 (tightened to measured in the same task), argmax identical;
  plus unit pins for each activation vs its f64 twin at grid points incl. the expLimit
  saturation edges and the softmax guard threshold. Run: FAIL (module absent).
- [ ] Implement `fast/nn.rs`. GREEN. MEASURE the synthetic-net delta distribution +
  the real tuple-A pack loaded through `from_flat` (construction-only pin: narrowing is
  lossless-roundtrip observable - `flat[i] as f32` bit-pins on a sample), tighten the
  tolerance pins with >=10x headroom over measured, record measured values in the test
  docstrings.
- [ ] Workspace-hygiene pin: `feed_forward` twice on the same input -> bit-identical f32
  output (workspace reuse is clean); an allocation-count smoke via a debug counter is
  NOT required (keep it simple - the bench, Task 7, is the allocation referee).
- [ ] Gates + commit `feat(phase7): fast::nn - f32 BLSTM forward on faer with preallocated workspaces`.

### Task 3: fast::pipeline - f32 features on realfft

**Files:**
- Create: `src/rust/src/fast/pipeline.rs`
- Test: `src/rust/tests/phase7_fast_pipeline.rs`

**Interfaces:**
- Consumes: `features::pipeline::FeatureConfig`/`SpectralParams` (the EXISTING derived
  params - derivation is config arithmetic, shared, not duplicated); the audio samples.
- Produces: `fast::pipeline::FastPipeline` with `new(params: &SpectralParams, cfg:
  &FeatureConfig) -> FastPipeline` (precomputes ONCE: the realfft plan
  (`RealFftPlanner::<f32>::plan_fft_forward(window_size)`), window coefficients f32, the
  mel bank + DCT table f32) and `build_input_sequence(&mut self, samples: &[f32]) ->
  FastMatrix` (windowing -> ONE real FFT per frame (no two-real packing - realfft is
  natively real; the exact path's packing is a GFFT-era trick, documented divergence) ->
  periodogram with the exact path's Welch-style normalization transcribed f32 -> mel ->
  log/DCT -> optional LTSV column, assembled per `assemble_input_sequence`'s priority
  rules). Also `samples_f32(audio bytes/path) -> Vec<f32>` decode helper OR consume the
  existing `read_audio` output narrowed once - implementer picks the seam that avoids
  duplicating symphonia plumbing; the audio is held ONCE in f32 on this path (no
  data/data_raw pair).
- Produces (for Task 4): periodogram access sufficient for the algo-3 driver's LTSV/mel
  needs (mirror `build_input_sequence_parts`' return shape in f32).

- [ ] RED: tolerance tests vs the exact pipeline on the committed 3 s + 60 s wavs at the
  tier2 config: per-element relative delta of the assembled input sequence under a
  placeholder tolerance, tightened to measured (expect f32 + FFT-algorithm differences;
  the MEASURED distribution governs the pin; record it). A windowing-coefficient unit pin
  (f32 vs f64 coeffs) and a mel-bank single-construction assertion (constructor counts
  via a test hook are overkill - assert instead that two `build_input_sequence` calls on
  the same `FastPipeline` are bit-identical and reuse the plan). Run: FAIL.
- [ ] Implement `fast/pipeline.rs`. GREEN with measured-then-pinned tolerances.
- [ ] Gates + commit `feat(phase7): fast::pipeline - f32 features on realfft, banks built once`.

### Task 4: the algo-3 fast SAD driver + Inference_Path selection

**Files:**
- Create: `src/rust/src/fast/driver.rs`
- Modify: `src/rust/src/engine/bag_of_processors.rs` (THE single dispatch site),
  `src/rust/src/toml_config.rs` (KEY_TABLE row: `[engine] inference_path <->
  Inference_Path`)
- Test: `src/rust/tests/phase7_parity_sad.rs` (+ a dispatch unit in the bag's tests)

**Interfaces:**
- Consumes: `FastPipeline`, `FastBlstm`, the shared decision layer
  (`tasks/segmenter.rs::results_to_segmentation`, `tasks/segmentation*`), the config map.
- Produces: `fast::driver::FastSpectralSegmenter` with `from_legacy(map, weights:
  Option<&[f64]>) -> Result<Self>` and `impl Segmenter` (the same trait the exact
  drivers implement - `get_segmentation(audio, refs, ...)`: fast features -> fast
  forward -> the f64 result-rows handoff (posteriors widened f32->f64 at the seam) ->
  the SHARED `results_to_segmentation`/scoring, so cols/`compute_errors` flow
  unchanged). PITCH PASS: `TDCwindow > 0` in fast mode -> typed bail (spec R4), pinned.
- Produces: `Inference_Path` config key (values `exact`|`fast`, ABSENT = exact; any
  other value = a construction error), read in `BagOfProcessors::from_configs`: algo 3 +
  `fast` -> `Processor::FastSpectral(FastSpectralSegmenter)` (a new enum variant; the
  match arms delegate identically to the other variants). Algo != 3 + `fast` -> a typed
  bail until Task 5 adds the Twin (then algo 6 Mode 7 joins; everything else stays
  bailed, pinned).

- [ ] RED: the dispatch unit (`Inference_Path fast` + algo 3 -> FastSpectral; absent ->
  the exact Spectral; junk value -> error; algo 4 + fast -> bail) + THE CI PARITY LEG
  (spec S1.6a): stage tier2_spectral.config + the tuple-A pack + prcts_excerpt.wav; run
  BOTH paths through the bag (image mode, both channels); assert (placeholders ->
  measured): posterior/result-row max relative delta; segment COUNT AND TYPES IDENTICAL
  per channel; boundary max-dt <= pin; scored-column relative delta (run once scored
  against a crafted reference to exercise compute_errors). Run: FAIL (no fast driver).
- [ ] Implement driver + key + dispatch. GREEN; MEASURE all deltas (print them in the
  test with --nocapture once), pin with >=10x headroom; if a boundary flips (R1), STOP,
  adjudicate openly in the ledger/report, then pin what is defensible.
- [ ] The exact-path-untouched proof: `git diff <base> -- src/rust/src/features src/rust/src/nn
  src/rust/src/tasks src/rust/src/audio.rs src/rust/src/cost.rs` is EMPTY (paste in the
  report); full cargo test green (every committed golden byte-stable).
- [ ] Gates + commit `feat(phase7): the fast algo-3 SAD driver behind Inference_Path - CI parity pinned`.

### Task 5: the Mode-7 fast LID driver (phSeq + cep)

**Files:**
- Modify: `src/rust/src/fast/driver.rs` (add `FastTwinLid`),
  `src/rust/src/engine/bag_of_processors.rs` (the algo-6 fast dispatch joins the Task-4
  site)
- Test: `src/rust/tests/phase7_parity_lid.rs`

**Interfaces:**
- Consumes: `FastBlstm` (the LID net; NO fast pipeline needed - Mode 7 consumes
  `external_features` one-hot/cep matrices read by the EXACT `audio.rs` readers,
  narrowed f32 at the driver seam), the shared `.scr`/result assembly.
- Produces: `fast::driver::FastTwinLid` (`from_legacy(map, sad_weights, lid_weights)`,
  `impl Segmenter`): Mode 7 ONLY (other modes + the wav arm + the reference-pitch pass
  typed-bail, each pinned); the SAD net is NEVER RUN (the synthesized-constant
  result_vec, transcribed from the exact `:1407` semantics); the LID forward per
  external_features entry; the in-band LID encoding + epilogue ponderation transcribed;
  posteriors widened f64 at the result seam so the shared `.scr` writer + confusion
  columns flow unchanged. `Processor::FastTwinLid` variant.
- Produces: the CI parity leg (spec S1.6b): the committed real 12,409-weight Mode-7 LID
  net + twin_mode7.config + the committed phSeq fixtures - `.scr`-level score deltas
  measured-then-pinned + ARGMAX DECISIONS IDENTICAL per file; a synthetic-cep leg (the
  8 committed phase6 cep fixtures through File_Type 2 fast vs exact, same assertions).

- [ ] RED (both parity legs + the bail pins). Run: FAIL.
- [ ] Implement. GREEN; measure; pin with headroom; argmax identity is a HARD assert
  (a flip = STOP and adjudicate - it would mean f32 moves a decision on committed
  fixtures).
- [ ] The exact-tree diff-empty proof again (the Task-4 command, now also excluding the
  shared dispatch file which DOES change - name the allowed file explicitly).
- [ ] Gates + commit `feat(phase7): the fast Mode-7 LID driver - phSeq + cep parity pinned`.

### Task 6: the corpus-gated metric-parity tier

**Files:**
- Create: `tests/pyo3/test_phase7_parity.py` (corpus-gated, local-only)
- Modify: `src/python/speech/drivers/baseline.py` ONLY IF a tiny hook is needed to point
  scoring at a config overlay (prefer zero changes: the eval configs are dicts -
  overlay `Inference_Path` via the existing `extra=` seam)
- Test: (the file IS the test)

**Interfaces:**
- Consumes: the phase-6 subset-gate recipe (`run_baseline` with the gate's exact
  subset/seed params) to produce checkpoints IF absent on disk (a session-local cache
  dir under the test's tmp area or `data/`-adjacent gitignored path - NEVER committed);
  `evaluate.dcf/lid_error/cavg`; the `Inference_Path` key via `assemble_flat_config`'s
  `extra=`.
- Produces: for each arm (sad, lid-features, lid-phseq): score the SAME held-out slice
  with the SAME checkpoint under `exact` and `fast`; assert metric deltas
  (DCF/lid_error/cavg) below measured-then-pinned epsilons AND per-file argmax/decision
  agreement counted (LID: identical argmax per file expected - HARD assert unless
  adjudicated; SAD: per-file DCF deltas + pooled delta pinned).

- [ ] RED (the file with skips verified clean sans corpus, then the real assertions
  against a stubbed fast path if the checkpoints exist... in practice: write the test,
  run corpus-gated locally, measure, pin). Runtime budget: reuse ~subset-gate scale
  (<10 min total; scoring only, no training unless checkpoints are absent).
- [ ] GREEN locally with measured pins; runtimes + deltas in the report + RESULTS.md's
  phase-7 parity paragraph.
- [ ] Gates (the non-slow suite untouched; lint) + commit
  `test(phase7): corpus-gated metric parity - fast vs exact on the subset checkpoints`.

### Task 7: the fast-path bench + pinned budgets

**Files:**
- Modify: `src/rust/src/bench.rs` + `src/rust/src/cli.rs` (`--path=fast` live),
  `src/rust/benches/kernels.rs` (fast-kernel twins: faer matmul at 92x96, realfft 1024,
  fast mel apply), `RESULTS.md` (the exact-vs-fast table + the PINNED budgets),
  `src/rust/tests/phase7_bench.rs` (the CI budget smoke)
- Test: extend `phase7_bench.rs`

**Interfaces:**
- Consumes: Tasks 1/4/5. Produces: `speech bench --path=fast <config>`; the CI smoke
  assertion `fast wall <= exact wall * 1.5` on the 60 s fixture (WIDE headroom - CI
  timing is flaky, spec R5; the real numbers are local); the RESULTS.md table: exact vs
  fast x {SAD 60 s, SAD 600 s (corpus-gated), LID cep, LID phSeq} - wall, RTF, maxrss,
  MB per audio-second, speedup; the measured ratios pinned as local regression bounds
  (documented as this-box numbers, hardware named).

- [ ] RED (the --path=fast rejection test flips to acceptance; the CI smoke). FAIL.
- [ ] Implement + measure the full matrix + pin + fill RESULTS.md.
- [ ] Gates + commit `feat(phase7): fast-path benchmarks - the measured exact-vs-fast budgets pinned`.

### Task 8: shared-code hygiene (behaviorally-neutral, golden-re-verified)

**Files:**
- Modify: `src/rust/src/features/pipeline.rs` (the mel-bank double-construction hoist:
  build the `MelFilterBank` once in `build_input_sequence[_parts]` and pass it through
  to `assemble_from_periodogram` instead of reconstructing; the pitch-pass site in
  `tasks/sad.rs:1571` reuses the hoisted bank ONLY if provably identical - the warp
  path rebuilds mel on the SAME params, verify at source and document)
- Adjudicate (modify only if provably safe): `src/rust/src/audio.rs` `data`/`data_raw`
- Test: the EXISTING golden suites are the test

**Interfaces:** none new - this task is defined by observable NO-CHANGE.

- [ ] Read the exact usage of every `MelFilterBank::new` site + `data_raw` reader; write
  the adjudication (hoist-safe? drop-safe? keep-documented?) in the report BEFORE
  editing.
- [ ] Apply the safe hoists; run the FULL cargo suite + the phase-1/2b feature goldens
  explicitly: zero diffs, all green, byte-stable. If any golden moves: revert that
  hoist, record why.
- [ ] Re-run `speech bench` exact on the 60 s fixture; record the (likely small) delta
  in RESULTS.md.
- [ ] Gates + commit `perf(phase7): behaviorally-neutral exact-path hoists - goldens byte-stable`.

### Task 9: dependency hygiene

**Files:**
- Modify: `src/rust/Cargo.toml` (+ `speech-py/Cargo.toml` if it re-declares): REMOVE
  `nalgebra`; REMOVE the direct `rustfft` declaration iff `realfft` covers every use
  (realfft re-exports/depends on rustfft - keep only what compiles); keep `faer`,
  `realfft`, `criterion`, `libc`.
- Test: the build is the test.

- [ ] `rg -n 'nalgebra|rustfft' src/rust/src src/rust/speech-py/src` -> zero direct uses
  (paste in the report); edit Cargo.tomls; `cargo build --release && cargo test` +
  maturin rebuild + `uv run pytest tests/pyo3 -q` green; binary size before/after noted.
- [ ] Commit `chore(phase7): drop unused nalgebra/rustfft direct deps`.

### Task 10: mutation battery

**Files:** `IMPROVEMENTS.md` only (a "Mutation battery (Phase 7)" subsection; the tree
clean between and after every cycle; FOREGROUND; corpus-gated catchers run locally).

- [ ] The list (apply-FAIL-revert-PASS, one line each, honest gaps recorded):
  (1) skip the adim application before f32 narrowing (from_flat consumes a pre-adim
  pack) -> the Task-2 construction/parity pins; (2) transpose the faer input-projection
  operands -> the synthetic tolerance pins; (3) break the realfft normalization (drop
  the Welch-style scale) -> the Task-3 pipeline tolerance; (4) flip the Inference_Path
  default to fast -> the dispatch default-exact unit; (5) poison the workspace between
  calls (skip the reset) -> the run-twice bit-identity pin; (6) run the Twin's SAD net
  in fast Mode 7 (replace the synthesized constant with a forward) -> the LID parity
  leg (cost columns move); (7) widen the CI budget smoke's operand (fast = exact*10) ->
  the smoke itself (sanity that it CAN fail); (8) un-hoist Task 8's mel bank (restore
  the double construction) -> observable-no-change means NO catcher fires - record as
  the honest expected-gap demonstrating the hoist's neutrality (the bench delta is the
  only observable).
- [ ] Full final pass: non-slow pytest + per-file slow + cargo test + lint. Commit
  `test(phase7): mutation battery results recorded`.

### Task 11: docs refresh

**Files:** `README.md`, `CLAUDE.md`, `RESULTS.md`, `IMPROVEMENTS.md` (consistency).

- [ ] CLAUDE.md: the `fast/` module rows (pipeline/nn/driver + bench.rs); the
  conventions line "nalgebra/ndarray for linear algebra" corrected (ndarray exact +
  faer fast; nalgebra gone); the Roadmap 2 Phase 7 line flips to done-with-numbers;
  the Inference_Path key documented in the config section; the bench subcommand noted
  as port-only. README: the phase-7 bullet + the measured table pointer. Every claim
  code-verified; every hash literal + verified at git log. RESULTS.md/IMPROVEMENTS
  consistency pass. Ledger Minors folded in.
- [ ] Gates + commit `docs(phase7): README/CLAUDE.md/RESULTS refresh - the fast path lands`.

### Task 12: final review + smart-commit + finish

- [ ] Final whole-branch review (most capable model; package `main@<base>..HEAD`): the
  exact-tree-untouchable audit (the diff-empty proofs re-run), parity-pin honesty
  (measured values vs pins, headroom real), the two-trees drift surfaces (module-doc
  cross-cites present), bench methodology (audio_s accounting, maxrss normalization),
  license hygiene (the phase-6 sweep method re-applied), corpus-gating uniformity,
  ledger Minor triage. Fix-now items -> a fix wave.
- [ ] Invoke the `smart-commit` skill taking the WHOLE git branch into account, then
  `superpowers:finishing-a-development-branch` (the user merges via PR; never push).

## Self-Review

1. **Spec coverage:** S1.1->T3; S1.2->T2; S1.3->T4+T5; S1.4->T2 (from_flat); S1.5->T4;
   S1.6->T4(a)+T5(b); S1.7->T6; S1.8->T1+T7; S1.9->T8; S1.10->T9; S1.11->T11(+T12);
   S2 tiers->T4/T5/T6/T7; R1->T4/T5's STOP-and-adjudicate steps; R2->the module-doc
   cross-cites (T2-T5) + T12's audit; R3->the Global Constraints line + T3's
   documented no-packing divergence; R4->T4/T5's pinned bails; R5->T7's wide CI smoke.
2. **Placeholders:** the tolerance values are deliberately "placeholder -> measured ->
   pinned" per the measure-then-pin decision - that is the method, not a gap; no TBDs
   remain.
3. **Type consistency:** `FastMatrix` defined in T2, consumed in T3/T4; `FastBlstm::
   from_flat` (T2) consumed in T4/T5; `FastPipeline::build_input_sequence` (T3)
   consumed in T4; `Processor::FastSpectral`/`FastTwinLid` (T4/T5) dispatch at one
   site; `run_bench`/the BENCH line format (T1) consumed in T7's smoke.
