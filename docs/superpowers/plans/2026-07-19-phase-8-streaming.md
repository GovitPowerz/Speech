# Phase 8: online/streaming mode - implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A chunked-input SAD streaming session on the phase-7 fast kernels (bounded
~4.3 s lookahead, frozen normalization) proven equivalent to the offline fast path under
identical frozen stats, plus per-utterance LID chunking, a `speech stream` CLI, and a
PyO3 binding.

**Architecture:** Approach A from the spec (docs/superpowers/specs/
2026-07-19-phase-8-streaming-design.md - READ IT FIRST, S0-S4 govern): a stateful
`fast::stream::StreamingSession` owning the existing `FastPipeline`/`FastBlstm` plus
four new state pieces (frozen-norm values, sample/feature ring, overlap accumulator +
per-row counts, latched decision state + pending smoothing tail); the ONE sanctioned
exact-tree touch is the `Audio_fixed_gain` Option-thread through `read_audio`
(inert-by-default, golden-proven). The offline fast path under the same frozen stats is
the oracle.

**Tech Stack:** Rust (the fast/ tree), the existing realfft/faer kernels, PyO3, no new
dependencies.

## Global Constraints

- The EXACT TREE IS UNTOUCHABLE except the S1.1 `Audio_fixed_gain` thread through
  `audio.rs::read_audio` + its bag/driver threading (Option<f64>, `None` = byte-identical
  legacy path; the FULL golden suite green is the proof) and the one KEY_TABLE row.
- R1 discipline: the design predicts EXACT streamed-vs-offline-frozen agreement; any
  boundary delta is STOP-and-adjudicate with a root-cause note, never a silent tolerance.
- Chunking must change TIMING only, never arithmetic: streamed outputs at different
  chunk sizes are BIT-IDENTICAL to each other (the R3 catcher).
- Measure-then-pin for the latency pins (the structural floor named separately from the
  data-dependent area term); the causality-cost leg is REPORTED in RESULTS.md, never
  gated (S1.7).
- Corpus gating + the license bright line exactly as established; CI sees committed
  fixtures only (the 60 s fixture STAGED MONO - channel 0 - per the spec).
- FOREGROUND everything; never run_in_background/Monitor; never read exit codes through
  pipes; slow corpus runs per-test.
- ASCII; fmt/clippy -D warnings/mypy/ruff clean; the commit trailer exactly
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; never push, never gh, never
  main.

---

### Task 1: `Audio_fixed_gain` + the frozen-norm reference mode

**Files:**
- Modify: `src/rust/src/audio.rs` (the sanctioned thread: `read_audio` gains
  `fixed_gain: Option<f64>`; `normalize_channels` skipped when `Some(g)`, each channel
  divided by `g.max(1e-3)` instead - mirror the legacy floor), `src/rust/src/engine/
  bag_of_processors.rs` (read the key, thread to the `read_audio` call),
  `src/rust/src/toml_config.rs` (KEY_TABLE row `("Audio_fixed_gain","audio","fixed_gain")`)
- Create: `tests/reference_data/phase8/` staging helpers as needed (a frozen-variant
  config: the tier2 config + `Audio_fixed_gain <measured>` + `BLSTM_InputNormalizationType 1`)
- Test: `src/rust/tests/phase8_frozen_norm.rs`

**Interfaces:**
- Produces: the offline fast path honoring frozen stats (the gate's REFERENCE mode);
  the frozen gate config staged by a shared test helper `common::stage_frozen_tier2`
  (tempdir: fixture wav mono-extracted channel 0, tuple-A pack, the frozen config) -
  consumed by Tasks 2/3/5/6.
- The fixed-gain VALUE for the gate config: measure the 60 s fixture's channel-0
  `(2*RMS+max)/2` once and bake that number into the staged config (so frozen and
  self-norm modes coincide numerically on THIS fixture - making later equivalence
  comparisons maximally interpretable); document the choice.

- [ ] RED: `fixed_gain_inert_when_absent` (a config WITHOUT the key -> `read_audio`
  output bit-identical to before - assert against a pre-recorded hash or direct
  comparison on the 3 s fixture) + `fixed_gain_replaces_normalization` (WITH the key ->
  the channel equals raw/gain exactly; the (2RMS+max)/2 path not taken). Run: FAIL.
- [ ] Implement the thread + the key row. GREEN. FULL cargo suite byte-stable (the
  inertness proof - name the golden binaries in the report).
- [ ] The CI causality-cost leg: offline-frozen vs offline-self-norm on the staged mono
  fixture - segment boundaries + posterior deltas measured and RECORDED (report +
  RESULTS.md placeholder row; with the gain baked to the fixture's own statistic the
  audio-norm halves coincide and the remaining delta isolates the type-1-vs-self-norm
  input change - state it).
- [ ] Gates + commit `feat(phase8): Audio_fixed_gain - the frozen-norm reference mode`.

### Task 2: the streaming front-end (per-range features + carried state)

**Files:**
- Modify: `src/rust/src/fast/pipeline.rs` (a per-range extension: compute feature rows
  `[from, to)` from the session's sample ring - reusing the existing kernels/banks; NO
  behavior change to the whole-sequence path)
- Create: `src/rust/src/fast/stream.rs` (the front-end half: `StreamFrontEnd` - the
  sample ring, pre-emphasis 1-sample carry, dither table-index carry, fixed-gain
  application per-sample, frame-completion arithmetic `frames_ready(samples_pushed)`)
- Test: `src/rust/tests/phase8_stream_frontend.rs`

**Interfaces:**
- Consumes: `FastPipeline` (T3's per-range API), the staged frozen config (T1).
- Produces: `fast::stream::StreamFrontEnd::new(params, cfg, rate, fixed_gain, noise_magnitude,
  preemph) -> Result`, `push(&mut self, samples: &[f32]) -> ()` (buffers + applies gain),
  `take_ready_rows(&mut self, pipeline: &mut FastPipeline) -> FastMatrix-slice/rows`
  (the newly completable feature rows), `flush(...)` (EOS: the tail frames incl. the
  right-edge zero-pad semantics). Exact-equivalence contract: the concatenation of all
  emitted rows == the offline `build_input_sequence` on the same (frozen-gain) audio,
  BIT-IDENTICAL, at any chunking.

- [ ] RED: `frontend_rows_bit_equal_offline` (the staged mono fixture pushed at 100 ms
  chunks -> concatenated rows vs the offline pipeline build: bit-equal) +
  `frontend_chunk_invariance` (20 ms vs 1 s vs 7 ms pushes -> bit-identical rows) +
  `preemph_carry_across_chunks` (a crafted 2-chunk signal where the boundary sample
  matters - the carry vs a no-carry mutation differs). Run: FAIL.
- [ ] Implement. GREEN (bit-equal, all chunkings).
- [ ] Gates + commit `feat(phase8): the streaming front-end - per-range features, carried state, bit-equal`.

### Task 3: the streaming overlap engine (window firing + row finalization)

**Files:**
- Modify: `src/rust/src/fast/stream.rs` (the overlap half: `StreamOverlap` - window
  firing when the input span exists, the accumulator ring + per-row counts, row
  finalization when the covering count completes, the EOS partial-tail flush replicating
  the offline clamp+snap), `src/rust/src/fast/nn.rs` ONLY if a windowed-single-call
  entry is needed (prefer driving the existing per-window body; the fast tree is open)
- Test: `src/rust/tests/phase8_stream_overlap.rs`

**Interfaces:**
- Consumes: `FastBlstm` (the per-window forward), T2's row stream.
- Produces: `StreamOverlap::new(net_window, net_shift, ssr, output_size)`,
  `push_rows(&mut self, rows, net: &mut FastBlstm) -> finalized posterior rows`,
  `flush(net) -> the tail rows`. Contract: the concatenation of finalized rows ==
  the offline `feed_forward_overlap` output on the full sequence, BIT-IDENTICAL
  (incl. the 0/0=NaN uncovered rows), at any chunking; a row finalizes no later than
  input-frame `row + 2*window_size` (the structural bound, asserted).

- [ ] RED: `overlap_rows_bit_equal_offline` + `overlap_chunk_invariance` +
  `finalization_bound_holds` (each row's finalization input-index <= row + 2*window + ssr
  slack) + `eos_tail_matches_offline` (the last ~window of rows, where the offline
  clamp+snap semantics live). Run: FAIL.
- [ ] Implement. GREEN bit-equal.
- [ ] Gates + commit `feat(phase8): the streaming overlap engine - windows fire, rows finalize, bit-equal`.

### Task 4: the incremental decision layer

**Files:**
- Modify: `src/rust/src/fast/stream.rs` (`StreamDecision` - the 19-tap convolution on
  the finalized-row stream with its half-width delay; the persistent hysteresis state
  machine replicating `update_segmentation_raw`'s loop state EXACTLY (begin/end/
  begin_area/end_area/has_begun/has_ended - transcribe from `segmenter.rs:290-415`,
  cite); re-smooth-and-emit-stable-prefix: maintain the raw segment list, on each new
  raw segment re-run the SHARED `smooth_segmentation` on a clone and emit segments whose
  end < (last_raw_boundary - HOLDBACK) where HOLDBACK = the derived maximum smoothing
  reach (padding-before 0.68 + final suppress 0.34 -> derive precisely from the config
  params at construction, do not hardcode); EOS flush)
- Test: `src/rust/tests/phase8_stream_decision.rs`

**Interfaces:**
- Consumes: T3's finalized posterior rows; the SHARED `tasks/segmenter.rs` +
  `tasks/segmentation.rs` functions (read-only reuse - the smoothing runs the existing
  f64 code on cloned lists).
- Produces: `StreamDecision::new(driver_cfg, seg_cfg, time_step)`,
  `push_rows(rows) -> Vec<EmittedSegment>` (`EmittedSegment { begin_s: f64, end_s: f64,
  class: SegClass, emitted_at_audio_s: f64 }`), `flush(audio_duration) -> (Vec<EmittedSegment>,
  Segmentation)`. Contract: flush's `Segmentation` == the offline
  `results_to_segmentation` + `smooth_segmentation` output on the same rows,
  BIT-IDENTICAL boundaries; every EmittedSegment appears UNCHANGED in the final list
  (prefix consistency - R2's catcher).

- [ ] RED: `decision_final_equals_offline` (drive with the offline-computed posterior
  rows from the fixture; compare flush vs the offline decision pipeline) +
  `prefix_consistency_holds` (collect every mid-stream emission; assert each appears
  unchanged finally) + `holdback_derived_not_hardcoded` (a config with different padding
  params -> the holdback moves accordingly). Run: FAIL.
- [ ] Implement. GREEN.
- [ ] Gates + commit `feat(phase8): the incremental decision layer - latched hysteresis, stable-prefix smoothing`.

### Task 5: StreamingSession assembly + THE CI GATE + latency pins

**Files:**
- Modify: `src/rust/src/fast/stream.rs` (`StreamingSession` composing T2+T3+T4:
  `new(map: &IndexMap<String,String>) -> Result` with the validation bails (algo 3 only,
  mono only, `Audio_fixed_gain` present, `InputNormalizationType 1` - each typed +
  pinned), `push(&[f32]) -> Vec<EmittedSegment>`, `finish() -> (Vec<EmittedSegment>,
  Segmentation)`, the per-emission lag tracking)
- Test: `src/rust/tests/phase8_gate.rs` (THE PHASE GATE)

**Interfaces:**
- Consumes: everything above + `common::stage_frozen_tier2`.
- Produces: the public session API (T6/T7 bind it); the gate results (the pins).

- [ ] RED: the gate legs - (a) `stream_finish_equals_offline_frozen` (the staged mono
  fixture: session at 100 ms vs the offline fast bag run on the frozen config: segment
  count/types IDENTICAL, boundary max_dt measured -> pin; the design predicts 0.0 - a
  nonzero is R1 STOP), (b) `chunking_bit_invariance` (20 ms/100 ms/1 s/7 ms: the four
  finish Segmentations + posterior histories BIT-IDENTICAL to each other), (c)
  `prefix_consistency_e2e`, (d) `validation_bails` (stereo config, missing gain, type -1,
  algo 4 - four bails), (e) `latency_bounds` (max emission lag <= window_s + conv_s +
  holdback_s + AREA_MEASURED with the measured area term recorded in the docstring and
  pinned with headroom; the structural components derived from the config, not
  hardcoded). Run: FAIL.
- [ ] Implement the session. GREEN; MEASURE + pin; any boundary delta -> STOP and
  adjudicate in the ledger/report before pinning.
- [ ] Gates + commit `feat(phase8): StreamingSession - the CI equivalence/invariance/prefix gate + latency pins`.

### Task 6: the `speech stream` CLI + the PyO3 binding

**Files:**
- Modify: `src/rust/src/cli.rs` + `src/rust/src/main.rs` (the `stream` subcommand, the
  bench template: `speech stream [--chunk-ms N] <config> <wav>` - replays the wav mono
  (bail on stereo), prints one parseable `SEG begin=<f> end=<f> class=<s> lag_s=<f>`
  line per emission + a `STREAM chunks=<n> max_lag_s=<f> mean_lag_s=<f> rtf=<f>` summary),
  `src/rust/speech-py/src/lib.rs` (`StreamingSession` pyclass: `new(config_path,
  chunk_defaults?)`, `push(numpy f32 1-D) -> list[tuple]`, `finish() -> (list[tuple],
  the segmentation rows)`; COPY semantics per the seam convention)
- Test: `src/rust/tests/phase8_cli.rs` (spawned-binary: the SEG/STREAM lines parse, the
  summary consistent with the session) + `tests/pyo3/test_phase8_stream.py` (the binding
  replays the fixture, equals the Rust gate's segments; importorskip)

**Interfaces:**
- Consumes: T5's session. Produces: the user-facing surfaces; the STREAM line format
  (T8's corpus instrument).

- [ ] RED (both test files) -> implement -> GREEN -> maturin rebuild + the pyo3 test ->
  gates + commit `feat(phase8): the stream subcommand + the PyO3 StreamingSession`.

### Task 7: StreamingLidSession (per-utterance)

**Files:**
- Modify: `src/rust/src/fast/stream.rs` (or a sibling `fast/stream_lid.rs` if cleaner):
  `StreamingLidSession::new(map) -> Result` (Mode-7 fast Twin validation),
  `push_utterance(features: &FastMatrix or the f64 rows) -> UtteranceScore { scores,
  argmax, running_aggregate }`, `finish() -> the final result rows` (== the offline
  FastTwinLid rows on the same entries); PyO3 + CLI NOT required (lib-only this phase -
  the LID streaming consumer is phase-9; document)
- Test: `src/rust/tests/phase8_stream_lid.rs`

**Interfaces:**
- Consumes: `FastTwinLid`'s per-entry machinery (refactor the per-entry loop body into a
  callable shared by the offline driver and the session - fast-tree internal, the
  offline parity legs must stay green).
- Produces: per-utterance streaming LID, utterance-granular by design.

- [ ] RED: `per_utterance_equals_offline` (push the committed phSeq fixtures one
  utterance at a time; each UtteranceScore == the offline per-entry values; finish rows
  == the offline driver's rows bit-identical) + `running_aggregate_is_prefix_correct`
  (after k pushes == the offline run on the first k entries). Run: FAIL.
- [ ] Implement (the loop-body refactor + the session). GREEN; the phase-7 LID parity
  legs re-run green (no offline regression).
- [ ] Gates + commit `feat(phase8): StreamingLidSession - per-utterance scores + running aggregate`.

### Task 8: the corpus tier + the causality-cost report

**Files:**
- Create: `tests/pyo3/test_phase8_parity.py` (corpus-gated: one real ~600 s train wav
  (sorted-first), the phase-6 SAD subset checkpoint (the T6-phase-7 cache recipe),
  streamed via the PyO3 session at 100 ms vs the offline-frozen fast run - the S1.6
  assertions on real data + trained weights; timings recorded)
- Modify: `RESULTS.md` (the phase-8 section: the equivalence/latency table + THE
  CAUSALITY-COST numbers - offline-frozen vs offline-self-norm on the corpus file +
  the held-out DCF delta on the phase-6 SAD test slice, both modes scored, the delta
  REPORTED with the mechanism named)
- Test: (the file is the test)

**Interfaces:** consumes T5/T6 + the phase-7 parity-cache recipe + `evaluate.dcf`.

- [ ] Write + run corpus-gated locally; measure; record; any real-data boundary delta
  between streamed and offline-frozen -> R1 STOP-and-adjudicate.
- [ ] Gates (non-slow suite untouched) + commit
  `test(phase8): corpus streaming parity + the causality cost reported`.

### Task 9: mutation battery

**Files:** `IMPROVEMENTS.md` only ("Mutation battery (Phase 8)"; tree clean between
cycles; FOREGROUND).

- [ ] The list (apply-FAIL-revert-PASS): (1) drop the pre-emphasis carry -> the T2
  carry test; (2) break the ring wraparound (off-by-one on the modular index) -> the T2
  bit-equal legs; (3) emit a row one window early (count-1) -> the T3 bit-equal/
  finalization tests; (4) skip the EOS tail flush -> the T3 eos test + the T5 gate; (5)
  emit past the smoothing holdback (holdback/2) -> the T4/T5 prefix-consistency legs;
  (6) leak chunk size into arithmetic (process a partial frame early) -> the
  chunk-invariance legs; (7) fixed-gain misapplied (skip the floor or apply twice) ->
  the T1 tests + the gate; (8) the LID running aggregate off-by-one-entry -> the T7
  prefix test. Honest gaps recorded.
- [ ] Full final pass (non-slow + per-file slow + cargo + lint). Commit
  `test(phase8): mutation battery results recorded`.

### Task 10: docs refresh

**Files:** `README.md`, `CLAUDE.md`, `RESULTS.md`, `IMPROVEMENTS.md` consistency.

- [ ] CLAUDE.md: the `fast/stream.rs` row (+ the pipeline/nn/driver row updates where
  T2/T3/T7 extended them); the `Audio_fixed_gain` key documented; the `stream`
  subcommand; the Roadmap 2 Phase 8 line flips to done-with-numbers (the equivalence
  verdict, the latency bound, the causality cost); the "Since Phase 8" convention bullet
  if warranted. README: the phase bullet + the streaming quick-start line. Every claim
  code-verified; hashes literal. The ledgered review Minors folded in.
- [ ] Gates + commit `docs(phase8): README/CLAUDE.md/RESULTS refresh - streaming lands`.

### Task 11: final review + smart-commit + finish

- [ ] Final whole-branch review (most capable model; package `main@bfd8657..HEAD`): the
  exact-tree audit (the ONE sanctioned thread), the bit-invariance honesty, the
  prefix/holdback derivation, the latency-pin split (structural vs data-dependent), the
  ledger Minor triage, license/corpus-gating uniformity.
- [ ] Invoke the `smart-commit` skill taking the WHOLE git branch into account, then
  `superpowers:finishing-a-development-branch` (the user merges via PR; never push).

## Self-Review

1. **Spec coverage:** S1.1->T1; S1.2->T2+T3+T5; S1.3->T4; S1.4->T7; S1.5->T6;
   S1.6->T5; S1.7->T1 (CI leg) + T8 (corpus + DCF); S1.8->T5; S1.9->T8; S1.10->T9+T10+T11;
   R1->T5/T8 STOP steps; R2->T4/T5 prefix legs; R3->the chunk-invariance legs + T9
   items 1-6; R4->T1's baked-gain note + T8 using one pack both sides; R5->T5's pin
   split.
2. **Placeholders:** none - the bit-equal contracts and the holdback derivation are
   specified as testable properties with the deriving rule named; the measured values
   are measure-then-pin by design.
3. **Type consistency:** `EmittedSegment` defined in T4, consumed T5/T6/T8;
   `stage_frozen_tier2` defined T1, consumed T2/T3/T5/T6; `StreamFrontEnd`/`StreamOverlap`/
   `StreamDecision` compose into T5's `StreamingSession`; the STREAM/SEG line formats
   defined T6, consumed T8.
