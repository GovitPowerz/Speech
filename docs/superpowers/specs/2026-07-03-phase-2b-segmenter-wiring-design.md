# Phase 2b - Segmenter Wiring Design Spec

Date: 2026-07-03
Status: approved (design), pending implementation plan
Branch: `feature/phase-2b-segmenter-wiring` (off `main`, which contains merged Phases 0a/0b/1/2)

## 1. Goal and scope

Wire the per-file segmentation drivers for Algo 1-4 into the already-ported layers (features, NN forward, decision primitives, segmentation I/O). Deliverables: the real `Segmenter` trait; four drivers in `tasks/sad.rs` (`TdcSegmenter` Algo 1, `LtsvSegmenter` Algo 2, `BlstmSpectralSegmenter` Algo 3 incl. the pitch-homothety second pass, `BlstmSignalSegmenter` Algo 4); the Segmenter-base residue (`results2segmentation`, the `{prefix}_window`/`{prefix}_shift`/`{prefix}_convolution_window_*` reads, `Dump_Directory`); the **run_pipeline lift** (`features::pipeline::build_input_sequence`, re-pointing both existing gate tests at the production path); VRCTS write-out through `tasks/segmentation_io`; and reproduction of the cross-file mutable-state semantics.

**Acceptance:** for each algo, wav -> `Segmentation`, bit-exact vs the harness oracle on the excerpt under real/variant configs: the (convolved) result vector, the boundary list, the **VRCTS XML bytes** (closing the Phase 0b-ii deferred writer golden), and `compute_errors` metrics vs a constructed reference. Two-files-in-sequence tests pin the stateful-member behavior.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Algo scope | 1-4 (TDC, LTSV, BLSTM spectral, BLSTM signal). LID (5/6) and VRCTSpart (0) defer. |
| 2 | Oracle | The SegProbe pattern: harness compiles the real segmenter TUs (`Segmenter.cpp`, `Segmentation.cpp`, `BLSTMSpectralSegmenter.cpp`, `BLSTMSignalSegmenter.cpp`, `LongTermSpectralVariation.cpp` [base-class link-required], `TimeDomainCorrel.cpp`) and probe subclasses call the REAL compiled protected machinery (`buildFromConf`, `updateSegmentation`, `smoothSegmentation`, `results2segmentation`, the param-derivation family, `getLTSV`, `getBLSTMInputSequence`, `getPitch`, `toFile_VRCTS`); only the `getSegmentation` bodies are transcribed, with the two `feedForwardBackward` calls swapped for the existing bit-stable reimpl. Rationale: a real-Eigen-forward oracle is structure/tolerance-only - boundary times are linear interpolations of posteriors (jitter ~1e-15 s per crossing) and hysteresis/area comparisons can flip discretely (measured posterior gap 2-4e-16/frame vs thresholds at Segmenter.cpp:734-817). |
| 3 | Real-forward probe | The compiled real-Eigen path runs as a SECONDARY probe per golden: segment count + types must match the reimpl-driven result exactly; boundary-time deltas measured and recorded in the manifest (expected <= ~1e-12 s). A structural mismatch aborts fixture generation (it would mean a knife-edge flip - surface, don't absorb). |
| 4 | iof shim | Upgraded from inert to a FAITHFUL mini-fmtr (spec in S8.2). This makes the real compiled `toFile_VRCTS` the byte-golden source, closing the 0b-ii deferral. The extraction proxy is NOT built - references are constructed programmatically. |
| 5 | run_pipeline lift | `build_input_sequence(audio, &FeatureConfig, &SpectralParams, chan, conv: Option<&[f64]>) -> Array2<f64>` in `features/pipeline.rs`, owning the mel-variant LTSV band reset (the `(0, spectral_cols-1)` vs `(freq_beg, freq_end)` asymmetry currently duplicated in `phase1_pipeline_golden.rs:351-426` and `phase2_e2e_gate.rs:65-145` - the two are structurally identical). Both gate tests re-pointed; goldens unchanged. |
| 6 | Stateful members | Rust drivers are mutable structs reproducing the legacy per-file mutations exactly (S3.4); pinned by two-files-in-sequence tests. |
| 7 | Deferred pieces | Base `computeCost` (Segmenter.cpp:640-657, transposed `targetSeq(jj,kk)` indexing) defers unless the plan's call-site check finds a live Algo 1-4 caller (Algo 3/4 use the NN's `getCost`; 1/2 accumulate no cost). `computeSpectralPitch` (cpp:1128-1242) defers pending the same check. Plotting/toPNG never ported (compiled into the harness for link-closure only). |

## 3. Segmenter base residue (exact) - `tasks/segmenter.rs` additions

Legacy: `Segmenter.{h,cpp}`.

### 3.1 Config residue -> `DriverConfig` (new struct alongside `SegmenterConfig`)

From `buildFromConf` (cpp:73-148), the keys NOT yet ported anywhere: `Dump_Directory` (global, default ""); `{prefix}_window` -> `window_size_sec` (clamp `< 0 -> 0`, cpp:90-91); `{prefix}_shift` -> `window_shift_sec` (clamp `< 0 -> 0`, cpp:92-93; THE stateful member seed); `{prefix}_convolution_window_size` (usize; if `> 0` also read `{prefix}_convolution_window_type` and verify, else type = "none", cpp:105-111); then UNCONDITIONALLY `conv_coeff = windowing_coefficients(conv_type, /*normalized*/true, 2*size+1, 0.83333)` (cpp:113; `Option<Vec<f64>>`, None = no smoothing); `{prefix}_BackPropWER` (default -1.0, no clamp, cpp:147). The write-only members `_CostPonderation`/`_CostLaw`/`_CostLawParam` (cpp:139-145) are NOT ported - IMPROVEMENTS entry notes them dead, incl. the same-key-different-default collision (`_CostLawParamSpeech` default 0.5 here vs 0.0 in CostLaw.cpp:14). Display_* plotting keys not ported. `verifyWindowingType` (cpp:61-71) ported as a warning-only check reproducing BOTH quirks: by-value sanitization that never fixes the member, and unconditional `_LogStream` overwrite (Rust: a returned warning string; the driver stores the LAST call's result only).

### 3.2 `results2segmentation` (cpp:1113-1126)

`pub fn results_to_segmentation(seg, time_step, time_offset, results: &mut Array2<f64>, chan, class, conv: Option<&[f64]>, cfg: &SegmenterConfig)`: if conv is Some with len > 1 -> `convolution_horiz(results, conv)` IN PLACE (later consumers see convolved values - load-bearing); then `update_segmentation(seg, results-as-row-slice, class, time_offset, time_step, cfg)`. The legacy targets param is unused by live code - dropped with a doc note. Unit-test dump `convoluted_output_chan_%d` is a harness-side concern.

### 3.3 Trait

```rust
pub trait Segmenter {
    fn get_segmentation(&mut self, audio: &mut Audio, seg: &mut Segmentation) -> anyhow::Result<()>;
    fn set_weights(&mut self, _flat: &[f64]) -> anyhow::Result<()> { Ok(()) }        // base no-op (cpp:150)
    fn get_weights(&self) -> Vec<f64> { Vec::new() }                                  // cpp:152-154
    fn get_weights_derivatives(&self) -> Array2<f64> { Array2::zeros((0, 0)) }        // cpp:156-158
}
```
`audio` is `&mut` because preemph/noise mutate it in place (legacy semantics). Copy-ctor semantics (everything except `_MatFilePtr`, cpp:23-57) documented on the drivers' `Clone` impls (Phase 4's per-thread copies).

### 3.4 Cross-file mutable state (reproduce exactly; two-files tests)

- `window_shift_sec` (`_WindowShift`): re-quantized per call USING ITS CURRENT VALUE - TDC: floor at `1/rate` then `round(x*rate)/rate` (TimeDomainCorrel.cpp:103-105); signal: `round(x*rate)/rate` (BLSTMSignalSegmenter.cpp:99-108) + `= 0.0` when noOverlap, reset BEFORE `compute_errors` (:376); spectral: `round(x*rate/ssif)*ssif/rate` (BLSTMSpectralSegmenter.cpp:445-453) + `= 0.0` when noOverlap AFTER the dumps (:885 - the reset-order divergence is cosmetic but ported as written); LTSV: `round(x*rate/spectrum_shift)*spectrum_shift/rate` (LongTermSpectralVariation.cpp:261-263).
- `spectrum_shift_sec` (`_SpectrumShift`): self-quantizing per call; spectral's non-wav fallback `_SpectrumShiftInFrames = 80` PERSISTS into the member (BLSTMSpectralSegmenter.cpp:208-210).
- `ltsv_shift_sec`: re-quantized per call (BLSTMSpectralSegmenter.cpp:302-306).
- Also mutated per file: the audio buffer (preemph/noise), NN derivative accumulators (reset once per file pre-channel-loop), NN cost/classif, the log string. NOT mutated (config-time only): conv_coeff, thresholds, padding/min lists, `window_size_sec` (read-only after the ctor clamp).

## 4. The four drivers (exact) - `tasks/sad.rs`

All flows are fully mapped in the Phase 1/2b reports; the plan cites the exact legacy lines per task. Summary contracts:

### 4.1 `TdcSegmenter` (Algo 1; TimeDomainCorrel.cpp:93-259)

Ctor: `buildFromConf(conf, "TDC", ...)` + `TDC_lags` (>= 2 else Err, ALL entries clamped >= 0) + `TDC_balance` (unclamped). Per file: window/shift derivation in samples (`half = round(w*rate/2)`, full = 2*half+1 ODD; shift floored at 1/rate then quantized - the stateful mutation); preemph -> noise; window coeffs (unnormalized); `vec_size = ceil(frameCount/shift)`; per channel: ROW result_vec, frames at `jj += shift` via `get_sequence` (dc_offset flag APPLIED here) + `tdc_classify_sequence`; `results_to_segmentation(seg, window_shift_sec /*post-quantization*/, 0.0, ...)`; `seg.compute_errors()` after the loop. Uses the Phase 1 ports (`get_sequence`, `tdc_classify_sequence`, `windowing_coefficients`) verbatim.

### 4.2 `LtsvSegmenter` (Algo 2; LongTermSpectralVariation.cpp:130-407)

Ctor: `buildFromConf(conf, "LTSV", ...)` + `FeatureConfig::from_legacy(map, "LTSV")` for the spectral keys. Per file: spectrum-order clamp 19; shift quantization (stateful); preemph -> noise; freq band (the LTSV.cpp clamp-order VARIANT: floor, ceil, only `freq_beg > freq_end -> freq_beg = freq_end` - no second reclamp; differs from the BLSTM variant already in `SpectralParams::derive` - port as a flagged variant); mel/DCT branch incl. `_NbDCT` clamp to nb_filters and the band reset to the mel output size; temporal-convolution kernel (normalized); LTSV window quantization with the LID-differing floors (`window < 1 -> 1`? NO - in the standalone LTSV: `LTSV_window_size < 1 -> 1` per :258 - NOTE this differs from the spectral segmenter's `< 1 -> 0` disable; port as written); `vec_size = ceil((frameCount+1)/spectrum_shift)` (the +1 quirk) then `real = ceil(vec_size/ltsv_shift)`; per channel: `compute_segment_periodogram_estimates` then DECIMATED row result_vec `result_vec(0, jj/shift) = ltsv_classify_sequence(...)` (no interpolation backfill - that is the spectral segmenter's getLTSV, not this driver); `results_to_segmentation(seg, window_shift_sec, 0.0, ...)`; `compute_errors`.

### 4.3 `BlstmSpectralSegmenter` (Algo 3; BLSTMSpectralSegmenter.cpp:593-887; the Phase 1 wiring report's step map is authoritative)

Ctor: `buildFromConf(conf, "BLSTM", ...)` + `FeatureConfig` + `BlstmConfig` + the LTSVshift-gate deviation (read unconditionally; already IMPROVEMENTS'd). Per file, per the wiring report steps 1-13: initSpectralAnalysis (order clamp, shift-in-frames incl. non-wav 80 fallback persisting, preemph/noise, freq band BLSTM-variant, mel bank from SNAPPED band, the `freq_beg/end` reset to the deltas-EXPANDED output size); window coeffs + temporal conv; `getBLSTMParam` (window/shift in periodogram-frame units, ssr floor, noOverlap incl. the `10*ssr` minimum, result-vec sizing: unconditional ceil-division by `_SpectrumShiftInFrames`, then UNCONDITIONAL sequential ssr-division - result_vec allocated ONCE and REUSED across channels incl. its contents: with the overlap driver accumulating into the caller's buffer, channel 2 seeds from channel 1's averaged results - load-bearing quirk, ported); per channel: `build_input_sequence` (the lift) + LTSV via `get_ltsv` (interpolated) with the reset band; targets via `get_targets` when a reference exists; `setProcessingType((window>0), !noOverlap)`; `feed_forward_backward`; timeStep/timeOffset: default `window_shift_sec*ssr` & `step/2 - shift/2`; overlap branch OVERRIDE `spectrum_shift_sec*ssr` & `step/2 - spectrum_shift/2` (the asymmetry vs signal); `results_to_segmentation` with `targetSeqTmp = result_vec` copy quirk; `_CumulativeError/_NbOfClassif` from the NN. PITCH SECOND PASS (:757-805) when `tdc.half_window > 0`: `get_pitch` (dc_offset threaded from cfg) over the FIRST-pass segmentation, `coeff = pitch/300`, `apply_homothety` on the periodogram, re-run filterbank/DCT, rebuild inputSeq WITH THE OLD LTSV column, re-forward, `seg.clear_classification(chan)` (NEW method on Segmentation - port from Segmentation.cpp), re-`results_to_segmentation`, overwrite the error/classif slots. Post-loop: `compute_errors`; noOverlap shift reset (:885).
NOTE: `seg._CumulativeError`/`_NbOfClassif`/`clearClassification` are per-channel fields on the legacy Segmentation not yet in the Rust container - S6 adds them.

### 4.4 `BlstmSignalSegmenter` (Algo 4; BLSTMSignalSegmenter.cpp; the signal report's 12-divergence list is authoritative)

Same "BLSTM" key namespace as Algo 3 (shared config surface) + `BLSTM_TwoSweeps` (required bool, parsed-but-dead - the parse itself is load-bearing). Per file: window/shift in SIGNAL samples (no `/ssif` anywhere); NO `full = 0` when window == 0 (stays 1 - windowing coeffs computed over length 1, dump-only); `_FlagDCOffset` log-only (never applied); preemph -> noise; per channel: `inputSeq = audio.data.row(chan)` as an Nx1 column - NO feature extraction; result-vec sizing: unconditional ceil-division by window_shift then ssr-division GATED on `(window==0 || noOverlap)`; fresh result_vec/targetSeq PER CHANNEL (allocation-scope divergence vs spectral); timeStep overlap branch `window_shift_sec` & 0.0; `feed_forward_backward`; `results_to_segmentation` with the same targetSeqTmp quirk; noOverlap shift reset BEFORE `compute_errors`. No pitch pass, no TwoSweeps (dead), no NN dumps.

## 5. run_pipeline lift (exact) - `features/pipeline.rs`

```rust
pub fn build_input_sequence(audio: &Audio, cfg: &FeatureConfig, s: &SpectralParams, chan: usize,
                            temporal_conv: Option<&[f64]>) -> Array2<f64>
```
Body = the (identical) composition currently in `phase1_pipeline_golden.rs::run_pipeline` and `phase2_e2e_gate.rs::assemble_real_input`: window coeffs, `compute_segment_periodogram_estimates(0..=ncols-1, temporal_conv)`, mel bank from the SNAPPED band with `spectrum_size = s.bins - 1`, optional DCT, spectral_cols resolution, the LTSV band asymmetry (`nb_bins > 0 -> (0, spectral_cols-1)` else `(freq_beg, freq_end)`), `get_ltsv` when `ltsv_half_window >= 1`, `assemble_input_sequence`. Preemph/noise stay OUTSIDE (the drivers own audio mutation). Both gate tests re-pointed at it; all goldens must stay green unchanged (the lift is a pure refactor - any golden churn is a bug). The spectral driver consumes it; the harness E2E leg comment updated to note the production path now exists.

## 6. Segmentation container additions - `tasks/segmentation.rs`

Per-channel bookkeeping the drivers write (legacy `Segmentation.h` public fields): `cumulative_error: Vec<f64>`, `nb_of_classif: Vec<i64>`, `classification: Vec<Vec<...>>`-equivalent... The legacy stores per-channel hypothesis lists; the Rust container is single-channel (0b-ii decision). RESOLUTION: the Rust `Segmentation` stays single-channel; drivers hold one `Segmentation` per channel (`Vec<Segmentation>`) and the per-channel scalars live on the DRIVER (`cumulative_error[chan]`, `nb_of_classif[chan]`); `clear_classification` becomes `Segmentation::clear_hypothesis()` (reset segs to the seeded `[Other@0, End@dur]` state, preserving audio_duration and the reference) - port from `Segmentation::clearClassification` (Segmentation.cpp - the plan cites exact lines). `compute_errors` is already per-channel-agnostic (takes one hyp + one ref). The multi-channel aggregation (`seg.compute_errors()` looping channels) lives in the driver. This is a structural deviation from the legacy's channel-indexed container, documented in the spec and IMPROVEMENTS (behavior-preserving: all goldens are per-channel).

## 7. Module interfaces

```rust
// tasks/segmenter.rs (additions)
pub struct DriverConfig { pub dump_dir: String, pub window_size_sec: f64, pub window_shift_sec: f64,
    pub conv_coeff: Option<Vec<f64>>, pub back_prop_wer: f64 }
impl DriverConfig { pub fn from_config(m: &IndexMap<String,String>, prefix: &str) -> anyhow::Result<DriverConfig>; }
pub fn results_to_segmentation(seg: &mut Segmentation, time_step: f64, time_offset: f64,
    results: &mut [f64], class: SegClass, conv: Option<&[f64]>, cfg: &SegmenterConfig);
pub trait Segmenter { /* S3.3 */ }

// tasks/sad.rs
pub struct TdcSegmenter { /* configs + stateful window_shift_sec + balance/lags */ }
pub struct LtsvSegmenter { /* configs + stateful shifts */ }
pub struct BlstmSpectralSegmenter { /* configs + BlstmNetwork + stateful shifts + spectrum_shift_in_frames */ }
pub struct BlstmSignalSegmenter { /* configs + BlstmNetwork + two_sweeps (dead) + stateful shift */ }
impl each: pub fn from_legacy(map, /*weights*/ Option<&[f64]>) -> anyhow::Result<Self>;
impl Segmenter for each (get_segmentation per S4; set/get_weights delegate to the NN for 3/4);
// per-channel outputs exposed:
pub fn segmentations(&self) -> &[Segmentation]; pub fn cumulative_error(&self) -> &[f64];
pub fn nb_of_classif(&self) -> &[i64]; pub fn score(&mut self, reference: ...) -> Vec<ScoreReport>;

// features/pipeline.rs (addition)
pub fn build_input_sequence(audio: &Audio, cfg: &FeatureConfig, s: &SpectralParams, chan: usize,
    temporal_conv: Option<&[f64]>) -> Array2<f64>;

// tasks/segmentation.rs (addition)
impl Segmentation { pub fn clear_hypothesis(&mut self); }
```
(Exact driver-struct fields and the `score` signature are fixed in the plan; the trait + lift signatures above are binding.)

## 8. Oracle harness plan

### 8.1 New TUs + link

build.sh adds: `Segmenter.cpp`, `Segmentation.cpp`, `BLSTMSpectralSegmenter.cpp`, `BLSTMSignalSegmenter.cpp`, `LongTermSpectralVariation.cpp`, `TimeDomainCorrel.cpp`, and `-lpng` (plotting symbols odr-used though runtime-dead). `exclude_nontrans` global: Segmentation.cpp:13 DEFINES it (zero-init false; normally set by FastSpeechProcessing.cpp:68) - nothing needed in the harness; the false default gates the STM excluded_region path off, matching the Rust loader's explicit parameter.

### 8.2 Faithful mini-fmtr (replaces the inert shim)

Semantics pinned in Phase 0b-ii (`%f.Ns` == `std::fixed << setprecision(N)`; IMPROVEMENTS.md + segmentation_io.rs:331,:567): ctors `fmtr(const char*)` AND `fmtr(const std::string&)` (the string ctor is a COMPILE BLOCKER today - Segmenter.cpp:366 etc.); `operator<<(ostream&, fmtr)` returns a proxy; each chained value emits literal text up to the next `%` directive then formats: `%s` = default stream insertion; `%f.Ns` = fixed/setprecision(N) with flag save/restore (N in {2,3,4} on golden paths); after the LAST directive the remaining literal tail flushes immediately and further chained inserts pass through raw (Segmenter.cpp:66-67 relies on this). Width/flag forms (`% 6ds` etc.) are log-only - pass through unformatted without crashing. NO extraction proxy. The upgrade changes .mat var names and dump filenames the Phase 1/2 stages produce ONLY if those stages used fmtr - they did not (they use the harness's own helpers); regenerate-twice must show zero .bin churn.

### 8.3 SegProbe pattern + goldens

Probe subclasses (harness-local, deriving the real classes) expose the protected machinery; `getSegmentation` bodies transcribed with provenance comments, NN calls swapped for the reimpl family. Per algo x config variant, dumps: the post-convolution result vector (`convoluted_output` equivalent), the boundary list (begin/type rows via a boundary-dump helper), the VRCTS bytes (real compiled `toFile_VRCTS` on the reimpl-driven Segmentation - THE 0b-ii closure golden), `compute_errors` metrics vs a programmatically-built reference, and the legacy unit-test dump inventory equivalents (signal: signalRaw PRE-preemph vs signal POST - the timing divergence is part of the golden; spectral: the full dump set). Configs: the real `1_worker_1.config` (Algo 3) + `seg.config` + variants covering: signal mode, noOverlap, overlap, window==0 full-sequence, LTSV standalone, TDC standalone, pitch-pass-active (TDCwindow > 0). Two-files-in-sequence golden: run the SAME probe object on the excerpt twice (second run inherits mutated state) - dump both files' boundary lists (pins the `_WindowShift` lifecycle incl. the noOverlap 0.0 reset).
Secondary real-forward probe per S2 decision 3 (structure equal or abort; boundary deltas to the manifest).

## 9. Fixtures, tests, acceptance

`tests/reference_data/phase2b/` (committed). Tests:
1. `DriverConfig::from_config` + conv-kernel construction (real configs; "uniform"+normalized quirk already covered - reference it).
2. `results_to_segmentation`: convolution-gate cases (None, len 1, len > 1 in-place mutation visible to caller), dispatch into update_segmentation (row orientation).
3. Per-driver goldens: result vectors + boundary lists + VRCTS bytes + compute_errors, per config variant (canary-gated where the chain traverses libm - i.e. everything downstream of features/NN; boundary lists compared as f64 bits under the gate; VRCTS bytes exact on the oracle env, hybrid-value re-parse comparison off-env).
4. Wiring hand tests: both result-vec sizing variants (incl. the gate asymmetry and the spectral cross-channel buffer reuse), timeStep/timeOffset all branches (default/noOverlap/overlap x spectral/signal), the noOverlap `10*ssr` floor, the signal full=1-when-window-0 quirk.
5. Stateful tests: two-files-in-sequence per driver (golden-matched); the spectral non-wav 80-fallback persistence documented (unit test only - no non-wav fixture).
6. Lift regression: both existing gates green UNCHANGED after re-pointing at `build_input_sequence`.
7. Pitch-pass golden: TDCwindow-active variant - first-pass and second-pass boundary lists both dumped (the first-pass dump preserved per the legacy dump quirk).
8. Fixture-presence pytest + manifest probe-field assertions (incl. the real-forward structural-equality records).

## 10. Risks and pitfalls (pinned by tests / logged)

- Transcription drift in the getSegmentation bodies: smallest-yet surface (wiring only), pinned by the real-compiled machinery underneath + the secondary real-forward structural probe + hand tests.
- The mini-fmtr is now load-bearing for golden BYTES: its `%f.Ns` must match the 0b-ii-pinned semantics exactly; pinned by comparing `toFile_VRCTS` output against `to_vrcts_string` (the Rust writer) on identical Segmentations - the two must agree byte-for-byte on the oracle env (this equivalence IS acceptance).
- Knife-edge hysteresis flips between reimpl and real-forward paths: decision 3 aborts generation on structural mismatch (measured, not assumed).
- The spectral cross-channel result_vec reuse interacting with the overlap in-place accumulation (channel 2 seeded by channel 1): reproduced deliberately; a dedicated golden on the 2-channel excerpt pins it.
- `Segmentation` single-channel restructure (S6): behavior-preserving but structural - the per-channel aggregation must match the legacy loop order in compute_errors dumps.
- Stale legacy traps: BLSTMSignalSegmenter is ~55% commented dead code (TwoSweeps block :282-305, feedBackward :264-314) - port from live code only.

## 11. Definition of done

- `./check_all.sh`, `./lint_code.sh`, `uv run pytest tests` green; both comparator modes green.
- All 8 test groups passing; VRCTS byte-equivalence (real toFile_VRCTS == Rust to_vrcts_string) established - the 0b-ii IMPROVEMENTS deferral entry updated to CLOSED.
- `tasks/sad.rs` + trait + lift no longer stubs; both gate tests re-pointed with zero golden churn.
- CLAUDE.md rows (tasks/segmenter.rs, tasks/sad.rs, features/pipeline.rs, tasks/segmentation.rs) updated; README roadmap Phase 2b paragraph; IMPROVEMENTS entries (dead write-only members + default collision, verifyWindowingType double quirk, the stateful lifecycle map incl. reset-order divergence, sizing-gate asymmetry, timeStep asymmetry, spectral buffer reuse, signal signalRaw-timing divergence, full=1 quirk, LTSV standalone floor-to-1 vs spectral floor-to-0, single-channel container restructure).
- Final step: smart-commit over the whole branch. No push.

## 12. Out of scope

LID (Algo 5/6) + TwinBLSTM + base `computeCost` (unless the call-site check flips it in) + `computeSpectralPitch` (same); VRCTSpart (Algo 0); BagOfProcessors/corpus driver/rayon (Phase 4); plotting; TRS loader; the iof extraction proxy; MLP-mode drivers (no live config).
