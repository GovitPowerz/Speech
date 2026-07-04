# Phase 2b - Segmenter Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the per-file segmentation drivers for Algo 1-4 (TDC, LTSV, BLSTM spectral incl. the pitch second pass, BLSTM signal) into the ported layers, bit-exact against a harness oracle whose decision machinery is the REAL compiled legacy code.

**Architecture:** The harness gains the segmenter TUs and a faithful mini-fmtr iof shim; probe subclasses call the real compiled protected machinery (buildFromConf, updateSegmentation, smoothSegmentation, results2segmentation, param derivation, toFile_VRCTS) and transcribe only the getSegmentation wiring with the NN swapped for the bit-stable reimpl. The real-Eigen-forward path runs as a secondary structural probe that ABORTS generation on a segment-structure mismatch. Rust drivers in tasks/sad.rs reproduce the wiring incl. the cross-file mutable-state lifecycle; the run_pipeline composition is lifted into the library.

**Tech Stack:** Rust (existing ports: features/*, nn/*, tasks/{segmenter,segmentation,segmentation_io}), C++ harness (Homebrew g++ strict-IEEE + upgraded iof shim), Python (pytest fixture checks).

## Global Constraints

- Branch: `feature/phase-2b-segmenter-wiring` (created off `main`). NEVER commit to main. No rebase.
- Spec: `docs/superpowers/specs/2026-07-03-phase-2b-segmenter-wiring-design.md` ("spec S<n>"). Legacy sources (git-ignored): `/Users/govit/Git/Govit/Speech/legacy/src/`.
- Commit trailer (every commit): `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Legacy quirks are load-bearing - port, don't fix (IMPROVEMENTS entries; Task 9 consolidates). NEVER edit legacy/.
- Port from LIVE code only. Known dead-code traps: BLSTMSignalSegmenter.cpp is ~55% commented (TwoSweeps :282-305, feedBackward :264-314); updateSegmentation's :846-996 block is dead; LID2Segmentation's threshMin arm is dead.
- CROSS-LIBM RULE (3rd-occurrence lesson, binding): any assert comparing a committed fixture/JSON/manifest value against a value recomputed through libm goes through the canary gate (Rust assert_oracle_eq*, Python tests/_libm_gate.py). "Both sides local" counts ONLY if both are computed in the same run. Everything downstream of features/NN traverses libm.
- All sec->sample conversions round-half-away-from-zero (f64::round == boost::math::round). Sequential ascending loops for any new numeric code (there should be almost none - this phase is wiring).
- Harness conventions unchanged: build.sh strict-IEEE flags, regenerate fixtures TWICE byte-identical, manifest measured-not-hardcoded, `// legacy: file:lines` provenance on every transcription. Fixtures: `tests/reference_data/phase2b/`.
- Run Rust tests via `cargo test --test <file>` from src/rust/; pytest/uv from repo root. Lint gates: cargo fmt --check, clippy --all-targets -- -D warnings, ./lint_code.sh. ASCII-only.
- Real config facts: `tests/reference_data/phase0/1_worker_1.config` = Algo 3, prefix BLSTM, decision thresholds rising 7.518e-1/4.77e-3, falling 3.742e-1/5.82e-2, `BLSTM_convolution_window_size 9`, LTSVwindow 0, TDCwindow 0 (no pitch pass), InputNormalizationType -1. `tests/reference_data/phase0bii/seg.config` + `lid.config` also vendored. Excerpt wav: `tests/reference_data/phase1/excerpt_2ch_8k.wav` at offset 0.35 dur 2.0.
- The Rust `Segmentation` is single-channel (spec S6): drivers hold `Vec<Segmentation>` per channel; per-channel scalars live on the driver.

---

### Task 1: Harness segmenter linkage + faithful mini-fmtr + call-site checks

**Files:**
- Modify: `tools/oracle_harness/build.sh`
- Modify: `tools/oracle_harness/shims/iof/io.hpp` (inert -> faithful)
- Modify: `tools/oracle_harness/main.cpp` (fmtr self-test stage)
- Create: `scripts/extract_phase2b_fixtures.py`
- Create (generated): `tests/reference_data/phase2b/manifest.json`
- Create: `tests/test_phase2b_fixtures.py`

**Interfaces:**
- Produces: harness links `Segmenter.cpp Segmentation.cpp BLSTMSpectralSegmenter.cpp BLSTMSignalSegmenter.cpp LongTermSpectralVariation.cpp TimeDomainCorrel.cpp` + `-lpng`; the faithful fmtr (spec S8.2); `FMTR_CHECK` parseable self-test lines in the manifest; call-site verdicts for `computeCost`/`computeSpectralPitch` recorded in the manifest (spec decision 7); the phase2b extractor + fixture dir + presence pytest (same conventions as phase2's).

- [ ] **Step 1: Faithful mini-fmtr**

Replace the inert shim per spec S8.2: ctors `fmtr(const char*)` AND `fmtr(const std::string&)`; `operator<<(std::ostream&, const fmtr&)` returns a proxy holding the ostream + format string + cursor; each chained `operator<<(proxy&, T value)` emits literal text up to the next `%` directive then formats the value (`%s` -> default insertion; `%f.Ns` -> save flags/precision, `std::fixed << std::setprecision(N)`, insert, restore); after consuming the LAST directive, flush the remaining literal tail immediately; further chained inserts pass through raw. Unknown/width directives (`% 6ds`, `%f3.2s` etc., log-only paths) must not crash: treat them as `%s` after emitting their literal prefix (document the simplification - they never reach a golden). Keep it header-only, ASCII, ~80 lines.

- [ ] **Step 2: build.sh + link**

Append the six TUs and `-lpng` (add `libpng` presence to the dependency check; it is a png++ dependency and likely already installed - `brew list libpng || brew install libpng`). Build. Resolve any residual link errors with more legacy TUs (record in the manifest) - the include-graph analysis predicts none beyond these.

- [ ] **Step 3: fmtr self-test + call-site checks in main.cpp**

New stage printing parseable lines:
- `FMTR_CHECK case=<name> ok=<0|1>` for: `fmtr("<x a=\"%s\" b=\"%f.4s\"/>") << 3 << 1.25` == `<x a="3" b="1.2500"/>`; `fmtr("%f.2s") << 120.0` == `120.00`; the tail-flush + raw-passthrough case (`fmtr("warn %s") << "t" << "u"` == `warn tu`); a `%f.3s` case. Compare against std::ostringstream ground truth built with fixed/setprecision directly; ok=0 on any mismatch -> extractor SystemExit.
- `CALLSITE_CHECK fn=computeCost callers=<n>` and `fn=computeSpectralPitch callers=<n>`: implemented as a build-time grep is fine - simpler: run `rg -c` from the EXTRACTOR (python) over the four live driver TUs (TimeDomainCorrel.cpp, LongTermSpectralVariation.cpp, BLSTMSpectralSegmenter.cpp, BLSTMSignalSegmenter.cpp) counting UNCOMMENTED call sites of `computeCost(` and `computeSpectralPitch(`; record counts + the matched lines in the manifest. Expected per the reports: 0 live callers each -> both stay deferred (spec decision 7). If nonzero: STOP and report to the controller (scope change).

- [ ] **Step 4: Extractor + presence pytest**

`scripts/extract_phase2b_fixtures.py` (docstring Usage; builds the harness; runs it with the phase2b out dir; parses FMTR_CHECK (all ok) + the callsite results into `tests/reference_data/phase2b/manifest.json`; run TWICE byte-identical). CRITICAL regression check inside the script: after the harness run, verify NO file under `tests/reference_data/phase1/` or `phase2/` changed (the fmtr upgrade must not perturb existing dumps - prior stages use the harness's own dump helpers, not fmtr; hash-compare before/after and SystemExit on drift). `tests/test_phase2b_fixtures.py`: manifest present, all FMTR_CHECK ok, callsite counts == 0.

- [ ] **Step 5: Verify + commit**

Run: `uv run python scripts/extract_phase2b_fixtures.py && uv run pytest tests/test_phase2b_fixtures.py -v && ./lint_code.sh`; `cd src/rust && cargo test` (unaffected, green); ASCII sweep.
```bash
git add tools/oracle_harness scripts/extract_phase2b_fixtures.py tests/reference_data/phase2b tests/test_phase2b_fixtures.py
git commit -m "feat(phase2b): link segmenter TUs; faithful iof fmtr shim; call-site checks

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: DriverConfig + results_to_segmentation + Segmenter trait + clear_hypothesis

**Files:**
- Modify: `src/rust/src/tasks/segmenter.rs`
- Modify: `src/rust/src/tasks/segmentation.rs` (clear_hypothesis)
- Modify: `tools/oracle_harness/main.cpp` (SegProbe base + results2segmentation goldens)
- Create: `src/rust/tests/phase2b_wiring_golden.rs`

**Interfaces:**
- Consumes: `windowing_coefficients`, `convolution_horiz` (audio.rs), `update_segmentation`, `SegmenterConfig` (segmenter.rs), `Segmentation` (segmentation.rs).
- Produces (spec S3/S7):
  - `pub struct DriverConfig { pub dump_dir: String, pub window_size_sec: f64, pub window_shift_sec: f64, pub conv_coeff: Option<Vec<f64>>, pub back_prop_wer: f64 }` with `from_config(m: &IndexMap<String,String>, prefix: &str) -> anyhow::Result<DriverConfig>` - keys per spec S3.1: `Dump_Directory` (global, default ""), `{prefix}_window` (default? legacy: required via get<double> with NO default - check Segmenter.cpp:90; port as required -> Err if missing; clamp < 0 -> 0), `{prefix}_shift` (same, clamp), `{prefix}_convolution_window_size` (usize; > 0 -> read `{prefix}_convolution_window_type` + verify, else "none"), then `conv_coeff = windowing_coefficients(ty, true, 2*size+1, 0.83333)`, `{prefix}_BackPropWER` (default -1.0). Plus `pub fn verify_windowing_type(ty: &str) -> Option<String>` (warning-only; reproduces the by-value no-fix quirk by NOT mutating anything; returns the legacy warning text).
  - `pub fn results_to_segmentation(seg: &mut Segmentation, time_step: f64, time_offset: f64, results: &mut Vec<f64>, class: SegClass, conv: Option<&[f64]>, cfg: &SegmenterConfig)` - conv Some && len > 1 -> convolution_horiz IN PLACE (adapt: convolution_horiz takes Array2 1xN; either add a slice variant `pub fn convolution_horiz_slice(row: &mut [f64], coeffs: &[f64])` in audio.rs reusing conv_taps, or wrap - pick the slice variant, it is the natural fit and spectral/LTSV/TDC all hold Vec results) then `update_segmentation(seg, results, class, time_offset, time_step, cfg)`.
  - `pub trait Segmenter { fn get_segmentation(&mut self, audio: &mut Audio, seg_per_chan: &mut [Segmentation]) -> anyhow::Result<()>; fn set_weights(&mut self, _flat: &[f64]) -> anyhow::Result<()> { Ok(()) } fn get_weights(&self) -> Vec<f64> { Vec::new() } fn get_weights_derivatives(&self) -> Array2<f64> { Array2::zeros((0,0)) } }` (spec S3.3; seg_per_chan pre-sized by the caller, one per channel).
  - `impl Segmentation { pub fn clear_hypothesis(&mut self) }` - port `Segmentation::clearClassification` (READ Segmentation.cpp for the exact body - the plan's implementer cites the lines; semantics: reset the hypothesis boundary list to the seeded `[Other@0, End@audio_duration]` state; the reference list is per-channel in the legacy and lives OUTSIDE the Rust container - nothing else changes).

- [ ] **Step 1: Harness SegProbe base + goldens**

`struct SegProbe : Segmenter { void getSegmentation(AudioStruct&, Segmentation&) override {} using Segmenter::buildFromConf; using Segmenter::results2segmentation; using Segmenter::updateSegmentation; ... }` (expose the protected surface). Stage: construct from the real 1_worker_1.config (`buildFromConf(conf, "BLSTM", false, false)` - the REAL compiled reader incl. `_ConvolutionCoeff` construction, window size 9 -> 19-tap normalized coeffs); build a deterministic synthetic result row (1 x 60, closed-form values spanning the real thresholds: `0.2 + 0.7*((k*13)%17)/16.0`); create a legacy `Segmentation seg(6.0)`; call the REAL `results2segmentation(seg, 0.04, 0.0, results, targets, 0, SPEECH)`; dump: the post-convolution results (`r2s_convolved.bin` 1x60), the boundary list (`r2s_boundaries.bin` Nx2: begin/type via a boundary-dump helper walking the legacy list), and the config-built conv coeffs (`r2s_conv_coeff.bin`). Also a conv=none variant (`BLSTM_convolution_window_size` overridden to 0 via `conf._Params` erase+set - the Task-1-P2 erase precedent; dump `r2s_boundaries_noconv.bin`). Regenerate twice.

- [ ] **Step 2: Failing Rust tests**

`phase2b_wiring_golden.rs`: DriverConfig::from_config on the real config (window/shift/conv values asserted; conv_coeff bit-compare vs `r2s_conv_coeff.bin` via assert_oracle_eq - cos chain); results_to_segmentation replay: same synthetic row + SegmenterConfig::from_config(real config) -> convolved results vs dump (oracle_eq), boundary tuples vs dump (f64 bits via oracle-gated compare; type column exact ints); the noconv variant (results untouched - bits equal to the input); in-place-mutation visibility test (caller's Vec changed); clear_hypothesis unit test (label two segments, clear, assert seeded state, audio_duration preserved).

- [ ] **Step 3: Implement -> PASS -> lint + commit** (message: `feat(phase2b): DriverConfig, results_to_segmentation, Segmenter trait, clear_hypothesis`)

---

### Task 3: build_input_sequence lift (zero-churn refactor)

**Files:**
- Modify: `src/rust/src/features/pipeline.rs`
- Modify: `src/rust/tests/phase1_pipeline_golden.rs` (re-point run_pipeline)
- Modify: `src/rust/tests/phase2_e2e_gate.rs` (re-point assemble_real_input)

**Interfaces:**
- Produces (spec S5): `pub fn build_input_sequence(audio: &Audio, cfg: &FeatureConfig, s: &SpectralParams, chan: usize, temporal_conv: Option<&[f64]>) -> Array2<f64>` - the exact composition quoted in the spec's source reports (window coeffs from cfg.win_type/win_param over s.buffer_size unnormalized; compute_segment_periodogram_estimates over 0..=ncols-1 with temporal_conv; mel bank from the SNAPPED band with spectrum_size = s.bins - 1 when nb_bins > 0; DCT when nb_dct > 0; spectral_cols resolution; LTSV band asymmetry - nb_bins > 0 -> (0, spectral_cols-1) else (s.freq_beg, s.freq_end); get_ltsv when s.ltsv_half_window >= 1; assemble_input_sequence). Preemph/noise NOT applied here (drivers own audio mutation).

- [ ] **Step 1: Implement the function** (move the body from `phase1_pipeline_golden.rs::run_pipeline:351-426`, generalizing the two call sites' differences - there are none beyond config source and the real config's LTSV being disabled by R==0, which the R >= 1 gate already handles).
- [ ] **Step 2: Re-point both tests** - `run_pipeline` becomes `build_input_sequence(&audio, &c, &s, chan, None)` + the audio/config setup; `assemble_real_input` likewise. Delete the duplicated composition bodies.
- [ ] **Step 3: Zero-churn proof** - Run: `cargo test --test phase1_pipeline_golden && cargo test --test phase2_e2e_gate` then FULL `cargo test` (both comparator modes). Expected: all green, no fixture or expectation changes anywhere in the diff (test-code-only). Any golden churn = bug in the lift.
- [ ] **Step 4: Lint + commit** (message: `refactor(phase2b): lift run_pipeline into features::pipeline::build_input_sequence`)

---

### Task 4: TdcSegmenter (Algo 1)

**Files:**
- Create driver code in: `src/rust/src/tasks/sad.rs` (replace stub)
- Modify: `tools/oracle_harness/main.cpp` (TdcProbe + dumps)
- Create: `src/rust/tests/phase2b_tdc_golden.rs`

**Interfaces:**
- Consumes: DriverConfig/SegmenterConfig/results_to_segmentation/Segmenter trait (Task 2), `get_sequence`/`windowing_coefficients` (audio.rs), `tdc_classify_sequence` (ltsv_tdc.rs), `compute_errors`/`to_vrcts_string` (segmentation_io.rs).
- Produces: `pub struct TdcSegmenter { /* driver_cfg, seg_cfg, lags: Vec<f64>, balance: f64, windowing type/param, stateful window_shift_sec */ }` with `pub fn from_legacy(map: &IndexMap<String,String>) -> anyhow::Result<TdcSegmenter>` (prefix "TDC"; TDC_lags >= 2 else Err, ALL clamped >= 0; TDC_balance unclamped) and `impl Segmenter` per spec S4.1. Per-channel outputs via `pub fn cumulative_error(&self) -> &[f64]` etc. (zeros for TDC - no cost path).

- [ ] **Step 1: Harness TdcProbe + dumps**

`struct TdcProbe : TimeDomainCorrel` exposing the machinery; transcribe getSegmentation (TimeDomainCorrel.cpp:93-259, provenance comments) - it calls NO NN, so the transcription swaps NOTHING; instead call the REAL `TimeDomainCorrel::getSegmentation` DIRECTLY as the oracle... CHECK FIRST: the real getSegmentation is fully deterministic (getSequence + classifySequence + results2segmentation - the only libm is fmath_log/cos, same on the oracle machine; NO Eigen GEMM in the chain - the autocorrelation is cwise+sum). If a quick NN_TOL-style probe of `classifySequence` vs the Phase-1-ported semantics confirms bit-equality (it should - Phase 1's tdc golden came from a transcription matched to the real class), use the REAL getSegmentation as the golden source directly (no transcription at all - strongest oracle of the phase). Construct from a TDC-prefixed config (write `tests/reference_data/phase2b/tdc.config`: TDC_window 0.032, TDC_shift 0.01, TDC_lags 0.002,0.016, TDC_balance 0.7, TDC_windowing_type hamming, TDC_windowing_param 0.8, TDC_flag_DCOffset false, TDC_preemph_ratio 0.97, TDC_noise_seed 0, TDC_noise_ratio 0, decision thresholds rising 0.6/0.05 falling 0.3/0.1, TDC_convolution_window_size 4, TDC_convolution_window_type hann, speech_padding 0.1,0.1,0.1,0.1, min_speech 0.2,0.2,0.2, min_silence 0.2,0.2, Dump_Directory <out>); run on the excerpt (both channels); dumps: `tdc_result_chan{1,2}.bin` (the PRE-convolution row - dump before calling results2segmentation by replicating the loop, or expose post-hoc; simplest: probe subclass copies result_vec before the r2s call), `tdc_convolved_chan{1,2}.bin`, `tdc_boundaries_chan{1,2}.bin`, `tdc_vrcts_chan1.xml` (REAL toFile_VRCTS bytes via the faithful fmtr), `tdc_scores.bin` (compute_errors vs a programmatic reference built by direct `label_segment` calls - constants in the manifest). TWO-FILES golden: run the SAME probe object twice on the excerpt; dump the second run's boundaries (`tdc_boundaries_file2_chan1.bin`) - pins the `_WindowShift` quantization lifecycle. Secondary probe N/A (no NN). Regenerate twice.

- [ ] **Step 2: Failing Rust tests** - from_legacy asserts; goldens: result rows (oracle_eq - fmath/cos chains), convolved rows, boundary lists (oracle-gated f64 + exact types), VRCTS bytes vs `to_vrcts_string` output AND vs the dumped file (byte-equal on oracle env - this is the S10 equivalence pin, canary-gated re-parse comparison off-env), compute_errors vs `tdc_scores.bin`; the two-files test (same TdcSegmenter instance, run twice, second boundaries == file2 dump); hand tests: shift floor at 1/rate, window ODD (2*round(w*rate/2)+1).

- [ ] **Step 3: Implement TdcSegmenter -> PASS -> lint + commit** (message: `feat(phase2b): TdcSegmenter driver (Algo 1) + VRCTS byte equivalence`)

---

### Task 5: LtsvSegmenter (Algo 2)

**Files:**
- Modify: `src/rust/src/tasks/sad.rs`
- Modify: `src/rust/src/features/pipeline.rs` (the LTSV.cpp freq-band clamp-order VARIANT)
- Modify: `tools/oracle_harness/main.cpp` (LtsvProbe + dumps)
- Create: `src/rust/tests/phase2b_ltsv_golden.rs`

**Interfaces:**
- Consumes: Task 2/3 products, `compute_segment_periodogram_estimates`, `MelFilterBank`, `ltsv_classify_sequence`, `FeatureConfig`/`SpectralParams`.
- Produces: `pub struct LtsvSegmenter { ... }` + `from_legacy(map)` (prefix "LTSV") + `impl Segmenter` per spec S4.2. In pipeline.rs: `pub fn derive_freq_band_ltsv_variant(cfg: &FeatureConfig, rate: f64, bins: usize) -> (usize, usize, f64, f64)` (the LTSV.cpp:200-209 clamp order: floor, ceil, only `freq_beg > freq_end -> freq_beg = freq_end`, snap; flagged doc comment naming the divergence vs the BLSTM variant in derive).

- [ ] **Step 1: Harness LtsvProbe + dumps** - like Task 4: the real `LongTermSpectralVariation::getSegmentation` is NN-free and (with the DCT caveat: its mel/DCT branch calls the REAL applyFilterBank/applyDCT - applyDCT contains the Eigen GEMM! Config choice: use nb_DCT 0 in the primary golden config (`ltsv.config`: LTSV_spectrum_order 8, LTSV_spectrum_shift 0.01, nb_bins 26, is_log_mel true, nb_DCT 0, LTSVwindow 0.3, LTSVshift 0.04, minFreq 64, maxFreq 3800, minMelFreq 64, maxMelFreq 3800, temporal conv 0/none, windowing hamming/0.83333, flag_DCOffset true, preemph 0.97, noise 0, decision/padding/min lists as Task 4, conv_window_size 4/hann) so the whole chain is GEMM-free and the REAL getSegmentation is the bit-golden. Add a SECONDARY nb_DCT 4 variant where the probe transcribes ONLY the applyDCT call to the loop product (the T7 applyDCTLoop already in main.cpp) - document the substitution). Dumps per channel: decimated result row, convolved row, boundaries, VRCTS bytes, compute_errors, + two-files golden. Regenerate twice.
- [ ] **Step 2: Failing Rust tests** - band-variant unit tests (crafted min/max where the missing second reclamp DIFFERS from the BLSTM variant: minFreq > maxFreq post-snap crafted case - derive both variants, assert they differ where the legacy differs); goldens incl. the frameCount+1 vec_size quirk and the floor-to-1 LTSV window (window sec small enough that round(...) == 0 -> floored to 1: assert via a dedicated config variant `ltsv_tiny.config` with LTSVwindow 0.001); the standard golden set + two-files.
- [ ] **Step 3: Implement -> PASS -> lint + commit** (message: `feat(phase2b): LtsvSegmenter driver (Algo 2) + LTSV-variant freq band`)

---

### Task 6: BlstmSignalSegmenter (Algo 4)

**Files:**
- Modify: `src/rust/src/tasks/sad.rs`
- Modify: `tools/oracle_harness/main.cpp` (SignalProbe + dumps)
- Create: `src/rust/tests/phase2b_signal_golden.rs`

**Interfaces:**
- Consumes: Task 2 products, `BlstmNetwork` (nn/blstm.rs: from_config/set_weights/set_processing_type/feed_forward_backward/cost/nb_of_classif), `get_targets`.
- Produces: `pub struct BlstmSignalSegmenter { /* driver_cfg, seg_cfg, net: BlstmNetwork, two_sweeps: bool (dead, parsed), stateful window_shift_sec */ }` + `from_legacy(map, weights: Option<&[f64]>)` + `impl Segmenter` per spec S4.4 (set_weights/get_weights delegate to the net). Per-channel `cumulative_error`/`nb_of_classif` from the NN.

- [ ] **Step 1: Harness SignalProbe + dumps**

`struct SignalProbe : BLSTMSignalSegmenter` - transcribe getSegmentation (BLSTMSignalSegmenter.cpp:93-398 LIVE code only, provenance comments), swapping `_BLSTMNeuralNetwork.feedForwardBackward` (:260) for the reimpl blstm family; the REAL compiled param math is inline in getSegmentation (not helper methods) so the transcription carries it - it is small (:96-108, :221-254) and the signal report's 12-divergence list is the checklist. SECONDARY probe: run the REAL `BLSTMSignalSegmenter::getSegmentation` beside it; compare segment structure (count + types EXACT, abort on mismatch per spec decision 3); record boundary max-delta into the manifest (`SEG_STRUCT site=signal ok=1 max_dt=<x>`). Config `tests/reference_data/phase2b/signal.config`: BLSTM_* namespace - a small synthetic net is NOT possible (the real 33,671 net expects 23 inputs; signal mode feeds 1 col - width tolerance topRows(1)! Fine: the real net RUNS on 1-col input via the tolerance; use the real net + real config keys BUT overridden window/shift for three variants: window==0 (full-sequence), overlapping (BLSTM_window 0.5, BLSTM_shift 0.1), noOverlap (BLSTM_window 0.5, BLSTM_shift 0)). Dumps per variant, chan 1: `signalraw` (PRE-preemph - the timing divergence golden), `signal` (post), result row, convolved row, boundaries, VRCTS bytes, errors; the noOverlap variant gets the TWO-FILES golden (pins the `= 0.0` reset: file 2 re-triggers noOverlap deterministically). Regenerate twice.

- [ ] **Step 2: Failing Rust tests** - from_legacy (TwoSweeps parsed-required: missing key -> Err; value unused - assert field exists and is dead via a doc test note); sizing hand tests (unconditional shift ceil-division; ssr-division GATED - all three variant cases with hand-computed expected sizes for frameCount=16001; the full=1-when-window-0 quirk: windowing coeffs computed over len 1 -> None? windowing_coefficients returns None for size <= 1 - assert the driver still proceeds, matching the legacy's empty-coeff no-op); timeStep branches (default/noOverlap: `shift*ssr` & `step/2 - shift/2`; overlap: `shift` & 0.0); goldens per variant (result rows oracle_eq; boundaries; VRCTS; errors; signalraw vs signal differ - bits of both pinned); two-files noOverlap golden; DC-offset-is-log-only test (flag true vs false -> identical outputs).

- [ ] **Step 3: Implement -> PASS -> lint + commit** (message: `feat(phase2b): BlstmSignalSegmenter driver (Algo 4) incl. stateful noOverlap lifecycle`)

---

### Task 7: BlstmSpectralSegmenter core (Algo 3, no pitch pass)

**Files:**
- Modify: `src/rust/src/tasks/sad.rs`
- Modify: `tools/oracle_harness/main.cpp` (SpectralProbe + dumps)
- Create: `src/rust/tests/phase2b_spectral_golden.rs`

**Interfaces:**
- Consumes: everything incl. `build_input_sequence` (Task 3), `get_ltsv`, `get_targets`, `load_ref_stm` (for a targets-active golden).
- Produces: `pub struct BlstmSpectralSegmenter { /* configs, net, stateful: window_shift_sec, spectrum_shift_sec, spectrum_shift_in_frames, ltsv_shift_sec */ }` + `from_legacy(map, weights)` + `impl Segmenter` per spec S4.3 (WITHOUT the pitch pass - Task 8 adds it; gate `tdc.half_window > 0` short-circuits false for the Task-7 configs). `pub(crate) fn get_blstm_param(...)` internal helper mirroring BLSTMSpectralSegmenter.cpp:439-500 (window/shift in periodogram frames, ssr floor, noOverlap incl. 10*ssr minimum, result-vec length: unconditional ceil-division by shift_in_frames then UNCONDITIONAL sequential ssr-division).

- [ ] **Step 1: Harness SpectralProbe + dumps**

`struct SpectralProbe : BLSTMSpectralSegmenter` - the REAL compiled helpers are protected member functions (initSpectralAnalysis, getWindowingCoeff, getTemporalConvolution, getLTSVParam, getTDCParam, getBLSTMParam, getLTSV, getBLSTMInputSequence): call them REAL; transcribe only the getSegmentation body (:593-887 minus the pitch block :757-805 for this task - configs keep TDCwindow 0), swapping the feedForwardBackward call (:740) for the reimpl. CRITICAL layout facts to preserve: result_vec allocated ONCE (returned by the real getBLSTMParam) and REUSED across channels - the overlap in-place accumulation makes channel 2 seed from channel 1 (the cross-channel golden: dump BOTH channels' results under the overlap variant; the Rust driver must reproduce by holding one buffer across the channel loop). Targets-active golden: build a reference from `tests/reference_data/phase0bii/ref.stm` via the REAL loadReference path? - the real loader needs the extraction proxy (NOT built); instead build the reference programmatically from the STM's known spans (constants in the manifest) via label_segment on `seg._Reference` - the probe sets it directly (public member). Configs: the REAL 1_worker_1.config as-is (window/shift from its BLSTM_window/BLSTM_shift values - READ them from the file) + two variants (overlap: BLSTM_shift small > 0; noOverlap: BLSTM_shift 0). Dumps per variant chan{1,2}: inputseq (re-validates build_input_sequence under the driver), result rows pre/post conv, boundaries, VRCTS bytes, errors, targets row (when reference set). SECONDARY real-forward structural probe + max_dt records per variant. TWO-FILES golden on the noOverlap variant + the spectrum-shift-quantization lifecycle (dump file2 params row). Regenerate twice.

- [ ] **Step 2: Failing Rust tests** - get_blstm_param hand tests (all clamps; sizing for 201 frames, sub ratios [4,1]/[2,1] cases incl. the unconditional-ssr-division contrast with signal's gated version - cite both legacy lines in comments); timeStep overlap-branch override (`spectrum_shift_sec*ssr` & `step/2 - spectrum_shift/2`); cross-channel reuse golden (chan 2's dump only matches if the buffer carries over - assert a fresh-buffer run DIFFERS, proving the quirk is pinned); full golden set per variant; targets golden; two-files golden; the non-wav 80-fallback documented as unit-test-only (construct the state mutation directly - no non-wav fixture).

- [ ] **Step 3: Implement -> PASS -> lint + commit** (message: `feat(phase2b): BlstmSpectralSegmenter core driver (Algo 3)`)

---

### Task 8: Spectral pitch-homothety second pass

**Files:**
- Modify: `src/rust/src/tasks/sad.rs`
- Modify: `tools/oracle_harness/main.cpp`
- Modify: `src/rust/tests/phase2b_spectral_golden.rs`

**Interfaces:**
- Consumes: `get_pitch`/`apply_homothety` (ltsv_tdc.rs), `clear_hypothesis` (Task 2), the Task 7 driver.
- Produces: the pitch block inside `BlstmSpectralSegmenter::get_segmentation` per spec S4.3 (BLSTMSpectralSegmenter.cpp:757-805): gated `tdc.half_window > 0`; `get_pitch` (dc_offset from cfg.flag_dc_offset) over the FIRST-pass segmentation; `coeff = pitch/300.0`; `apply_homothety` on the periodogram; re-run filterbank/DCT; rebuild inputSeq WITH THE OLD LTSV column; re-forward (reimpl path); `clear_hypothesis`; re-`results_to_segmentation`; OVERWRITE the error/classif slots.

- [ ] **Step 1: Harness pitch-variant golden** - config `spectral_pitch.config` = the Task 7 base + `BLSTM_TDCwindow 0.032, BLSTM_TDCshift 0.01, BLSTM_TDC_lags 0.002,0.016, BLSTM_TDC_balance 0.7, BLSTM_TDC_windowing_type hamming, BLSTM_TDC_windowing_param 0.8`; the probe transcribes the pitch block (the real getPitch helper called REAL; homothety + refilterbank via the existing loop-product path). Dumps: first-pass boundaries (`pitch_boundaries_pass1.bin` - the legacy dump quirk preserves pass-1 result_vec), the measured pitch (manifest), warped-periodogram slice hash, second-pass boundaries + VRCTS + errors. Regenerate twice.
- [ ] **Step 2: Failing tests** - both-pass goldens; the pass-1-dump-preserved quirk (result dump == pass 1 even though boundaries == pass 2); pitch value vs manifest (oracle-gated scalar); gate-off regression (Task 7 goldens unchanged - full rerun).
- [ ] **Step 3: Implement -> PASS -> lint + commit** (message: `feat(phase2b): pitch-homothety second pass (Algo 3 complete)`)

---

### Task 9: Docs, IMPROVEMENTS, accumulated minors, verification

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `IMPROVEMENTS.md`, `src/rust/src/tasks/mod.rs` (doc refresh)

- [ ] **Step 1: IMPROVEMENTS.md** - new [phase2b] entries per spec S11: dead write-only base members + the `_CostLawParamSpeech` default collision (0.5 vs CostLaw's 0.0); verifyWindowingType double quirk (by-value no-fix + _LogStream clobber); the stateful `_WindowShift` lifecycle map (incl. the noOverlap 0.0 poisoning and the signal-vs-spectral reset-order divergence); sizing-gate asymmetry (signal gated / spectral unconditional, with the commented-gate line refs); timeStep/timeOffset asymmetry; spectral cross-channel result_vec reuse x overlap accumulation; signal signalRaw-timing divergence; signal full=1-when-window-0; LTSV standalone floor-to-1 vs spectral floor-to-0; the single-channel container restructure (behavior-preserving deviation); UPDATE the 0b-ii deferred VRCTS-writer entry to CLOSED (byte-equivalence established) with the new evidence cite.
- [ ] **Step 2: CLAUDE.md** - rows: tasks/segmenter.rs (+DriverConfig/results_to_segmentation/trait), tasks/sad.rs -> Implemented (Phase 2b) with the four drivers, tasks/segmentation.rs (+clear_hypothesis), features/pipeline.rs (+build_input_sequence, gates re-pointed); harness note (segmenter TUs + faithful fmtr + SEG_STRUCT probes); check stale overview text.
- [ ] **Step 3: README roadmap** - Phase 2b done paragraph: the SegProbe real-compiled-machinery oracle, the faithful fmtr closing the VRCTS writer golden, the FP-flip analysis and the structural secondary probe, what is real-compiled vs transcribed vs Rust-only.
- [ ] **Step 4: Full verification** - `./check_all.sh && ./lint_code.sh && uv run pytest tests -v`; both comparator modes; ASCII sweep; regenerate-twice final check.
```bash
git add -A   # verify scope first
git commit -m "docs(phase2b): mark segmenter wiring implemented; IMPROVEMENTS consolidation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
- [ ] **Step 5: smart-commit** - controller-run over the whole branch, then final whole-branch review + finishing-a-development-branch.

---

## Self-Review

**1. Spec coverage:** S3.1/3.2/3.3 + S6 -> Task 2; S5 -> Task 3; S4.1 -> Task 4; S4.2 (+the band variant) -> Task 5; S4.4 -> Task 6; S4.3 core -> Task 7; S4.3 pitch -> Task 8; S8.1/8.2 + decision 7 call-site checks -> Task 1; S8.3 SegProbe/goldens -> Tasks 2,4-8 incrementally; S9 tests 1-8 -> mapped (1-2 Task 2; 3 per-driver; 4 Tasks 6/7; 5 per-driver two-files; 6 Task 3; 7 Task 8; 8 Task 1); S10 risks: fmtr equivalence (Task 4 Step 2 byte pin + Task 1 self-test), knife-edge abort (Tasks 6/7 secondary probes), cross-channel reuse (Task 7), single-channel restructure (Task 2 + per-driver aggregation in goldens), dead-code traps (Global Constraints); S11 -> Task 9; S12 exclusions hold (computeCost/computeSpectralPitch gated by Task 1's call-site check with a STOP-on-nonzero escalation). Gap check: none found.

**2. Placeholder scan:** Task 4 Step 1 contains a conditional oracle choice (real getSegmentation directly IF the NN-free chain probes bit-equal) with both arms specified - deliberate, not a placeholder. Config literals are given inline for every new .config. Intricate transcriptions follow the established RE-with-golden pattern (legacy lines + probes as arbiter). No TBD/TODO.

**3. Type consistency:** `results_to_segmentation(seg, time_step, time_offset, results: &mut Vec<f64>, class, conv, cfg)` consistent Tasks 2/4/5/6/7 (drivers hold Vec<f64> rows); `convolution_horiz_slice` introduced Task 2, consumed by results_to_segmentation only; the Segmenter trait's `seg_per_chan: &mut [Segmentation]` consistent across drivers; `build_input_sequence` signature identical Tasks 3/7; fixture names consistent between producing and consuming steps; DriverConfig fields match spec S7 verbatim.
