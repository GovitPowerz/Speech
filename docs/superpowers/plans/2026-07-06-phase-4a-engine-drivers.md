# Phase 4a - Engine Drivers (Corpus / BagOfProcessors / CorpusProcessor / CLI / .mat) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the corpus-level engine drivers - `engine/corpus.rs` (Corpus/CorpusItem CSV parsing), `engine/bag_of_processors.rs` (Algo dispatch, per-file `SegmentationFunction`, `saveAndUpdate` best-weight lifecycle), `engine/corpus_processor.rs` (mode dispatch, epoch loop with the static-lane deterministic reduction, corpus-level gradCheck, `.mat` results), `io/matfile.rs` (v5 writer), and the full legacy CLI - bit-exact against a two-tier oracle: the REAL compiled legacy driver stack on NN-free Algo 1/2 corpora (zero transcription) plus a CorpusProbe reimpl-swap for Algo 3 with real Rprop epoch chaining.

**Architecture:** The bag holds `Vec<Processor>` (enum over the concrete 2b drivers, NOT `Box<dyn Segmenter>`); the epoch loop uses static lanes (file `j` -> lane `j mod N`, per-lane bag clones chaining state, results folded in ascending file order) so N=1 is byte-identical to the legacy single-thread run == the omp-shimmed harness == the goldens. `.mat` parity is value-level via scipy-converted `.bin` dumps (legacy matio writes zlib-compressed vars); Rust ships an uncompressed-v5 WRITER only, no reader. Spec: `docs/superpowers/specs/2026-07-06-phase-4a-engine-drivers-design.md` ("spec S<n>").

**Tech Stack:** Rust (existing ports: `tasks/sad.rs` drivers, `tasks/segmentation{,_io}.rs`, `nn/blstm.rs`, `audio.rs`, `legacy_config.rs`, `io/binary.rs`), C++ harness (`tools/oracle_harness/`, Homebrew g++ strict-IEEE, six segmenter TUs + NN TUs already linked; adds `CorpusProcessor BagOfProcessors Corpus CorpusItem` + Homebrew libmatio), Python (scipy `.mat` conversion + pytest cross-checks).

## Global Constraints

- Branch: `feature/phase-4a-engine-drivers` (off `main` at `8efcbfc`; spec committed at `e077500`). NEVER commit to `main`. No rebase. NEVER push. NEVER use `gh`.
- Commit trailer (every commit), verbatim: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Legacy sources (git-ignored): `/Users/govit/Git/Govit/Speech/legacy/src/`. NEVER edit `legacy/`. `// legacy: file:lines` provenance comment on every transcription.
- Legacy quirks are load-bearing - PORT, do not fix. Every reproduced quirk gets an IMPROVEMENTS.md entry (Task 12 consolidates; spec S8 is the checklist; log candidates as you go).
- **Bit-exact contract**: strict bits for pure arithmetic/scalar values (corpus CSV parse, result-vector assembly given driver outputs, `saveAndUpdate` sums/means, costMem, Rprop-updated weights at synthetic shapes, `.bin` bytes). Everything downstream of libm (the NN/feature chains feeding cumulative_error, LTSV/TDC scores) is CANARY-GATED via `assert_oracle_eq`/`assert_oracle_eq_f64` (`src/rust/tests/common/mod.rs`). The CROSS-LIBM RULE from the Phase 3 plan is binding: any assert comparing a committed fixture value against a value recomputed through libm goes through the canary gate.
- **NaN policy**: fixture NaNs compare bit-exact on the oracle env, by `is_nan` off it (arch-defined quiet-NaN sign; CLAUDE.md rule).
- **Timing mask (spec S6)**: result-vector col 3 (`time_per_hour`, wall clock) = `MultiConfigResults` col 6 (after the 3 id cols) is masked in EVERY comparison; the extractor zeroes it in converted dumps and records the mask in the manifest.
- **Ascending-loop product contract** unchanged (`matmul_seq`/`matSeq`); no new goldens may use `ndarray::dot`/Eigen `*`.
- ASCII-only in all files. Plain hyphens, straight quotes.
- Fixtures: `tests/reference_data/phase4a/`. Harness build is LOCAL-ONLY (CI consumes committed fixtures). Regenerate TWICE byte-identical. Manifest values MEASURED, never hardcoded. Extractor hash-guards `phase1/phase2/phase2b/phase3` fixtures against drift (copy the `extract_phase2b_fixtures.py:294-303,428-435` pattern).
- Run Rust tests via `cargo test --test <file>` from `src/rust/`; pytest/uv from repo root. Gates: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `./lint_code.sh`, full `./check_all.sh && ./lint_code.sh && uv run pytest tests`.
- Subagents do the work themselves - no delegation-of-delegation, no worktrees from within a task.

### Legacy source map (binding line references)

- `CorpusProcessor.cpp`: ctor `:48-72` (output-file TRUNCATION in m/t modes `:53-59`, epochs clamp `:60-62`, epsilon `:63`, `_BestCost=1e20` `:64-66`, `isWorkDistributed` kill `:67-71`), `runSolo` `:74-81`, `train` `:83-111` (epoch-0 solo, inner mode `'m'`, final eval at `epoch=N+1`), `run()` dispatch `:113-137`, `run(epoch,...)` `:140-235` (omp loop `:172-203`, critical accumulation `:178-202`, `saveAndUpdate`+`saveResults` gating `:215-234`), `gradCheck` `:237-340` (bag snapshot `:238`, `_TrainingEpochs=0` `:243`, cost columns `:275-289`, relative floor 1e-24 `:329-331`), `transformResults` `:342-389`, `saveResults` `:391-404`.
- `BagOfProcessors.cpp`: ctor `:9-51` (key clears `:21-23`, per-algo push + `_ConfigIndex`), copy `:53-71`, `isBackPropActivated` `:73-86`, `getWeights` `:88-101`, `setWeights` `:103-114`, `getInputStatistics` `:116-130`, `getWeightsDerivatives` `:132-146`, `saveWeights` `:148-181` (per-algo criteria), `updateWeights` `:183-205` (`costLID=-1` arrives from `saveAndUpdate:465`), `SegmentationFunction` `:207-407` (lock block `:214-250` STUBBED, scored/unscored result assembly `:311-402`), `saveAndUpdate` `:409-471`, `PrintConfusionMatrix` `:473-598` (STUB in 4a).
- `Corpus.cpp`: ctor `:7-131` (mapping parse `:16-39`, listing parse `:58-101` with the NESTED only-if-previous-token-non-empty quirk `:74-96`, no-listing fallback keys `:43-57`), `addItem` `:164-194` (unknown lang/dial -> classIndex -1 + mapping MUTATION).
- `FastSpeechProcessing.cpp`: `main` `:31-75` (mode parse, `--key=val` overrides via `set_val(..., false)` where empty value REMOVES the key, `exclude_nontrans` global `:68`), usage `:78-87`.
- `BLSTMNeuralNetwork.cpp`: `weightsFile` ctor load `:122-150` (`<prefix>_weightsFile` via `BinaryFile2Vector`, too-few -> error, too-many -> warning), `resetInputStatistics`/`getInputStatistics` `:286-300`, `_InputStatistics` accumulation `:380-415` (from_matrix on first, `update` after; `leftCols` width per branch), `saveWeights` `:312-331` (`weights_<filename>` + `weightsDerivatives_<filename>` as `.bin` via `Matrix2BinaryFile`; `.mat` gets `nbOfInputs` scalar + `meanInputs`/`stdInputs`).
- `Helpers.hpp`: `Value2MatFile` `:441-454`, `Matrix2MatFile` `:456-470` - both `Mat_VarCreate(MAT_C_DOUBLE)` + `Mat_VarWrite(..., MAT_COMPRESSION_ZLIB)`.
- `Segmentation.cpp`: ctor `:42-113` (duration `(frames-1)/rate`, per-chan seed, reference load dispatch by extension), `compute_errors` `:288` (Rust port: `segmentation_io::compute_errors`).

### The 18-column result vector (Algo 1-4, no LID; binding for Tasks 5-9)

| col | value | source (Rust) |
|-----|-------|---------------|
| 0 | `100*Pfa(SPEECH)` | `ScoreReport.per_class[SPEECH].pfa` (field names per `segmentation_io.rs:45-49` + `ErrorStats`) |
| 1 | `100*Pmiss(SPEECH)` | same report |
| 2 | `100*sum(per_class[OTHER..EXCLUDED].error_rate)` | same report (classes `OTHER`..`EXCLUDED` exclusive, `BagOfProcessors.cpp:317-319`) |
| 3 | `time_per_hour` (WALL CLOCK - masked) | `std::time::Instant` measurement, formula `BagOfProcessors.cpp:309` |
| 4 | `cumulative_error[chan]` | driver accessor (`sad.rs`) |
| 5 | `(frame_count-1)/rate` | audio |
| 6 | speech-duration walk over hyp segments | `BagOfProcessors.cpp:324-330` |
| 7-13 | WER: nb_words, corrects, subs, ins, dels, coverage, delay | `ScoreReport.wer` |
| 14,15 | `0.0, 0.0` (no LID in 4a - the `_IsLIDCorrect.empty()` branch) | constants |
| 16 | `lid_nb_of_classif` = 0 | constant in 4a |
| 17 | `nb_of_classif[chan]` | driver accessor |

`saveAndUpdate` reads: cost = `sums(4)/sums(17)` (guard `>0`), badClassif = `means(2)`, costLID = `sums(14)/sums(16)` (guard `>0`), badLIDClassif = `100-means(15)`, WER over cols 7-13. gradCheck reads cost col 4 / counter col 17 (algo 3/4).

### Mode semantics (binding)

Mode letters carry CASE: uppercase = verbose logging. Extend `cli::Mode` to `struct Mode { kind: ModeKind, verbose: bool }` with `ModeKind { Solo, Image, UnitTest, Multi }`. Letter gates transcribed exactly: log emission gates on UPPERCASE `S/T/M/I` (`BagOfProcessors.cpp:257,262,...`); the scored-results branch gates on `m/M/t/T` OR (`i/I` AND reference non-empty) (`:311`); training/gradCheck allowed for `m/M/t/T/i/I` (`CorpusProcessor.cpp:116-133`); `train()`'s inner epochs use lowercase non-verbose `m` (`:96-97`); gradCheck permanently flips the processor mode to `m` (`:123`).

### What already exists (do NOT rewrite)

- Drivers + `Segmenter` trait (`tasks/segmenter.rs:24-42`), driver `from_legacy` ctors (weights via `Option<&[f64]>`), `score` statics, `cumulative_error()`/`nb_of_classif()` accessors, `last_result_rows` observation points.
- `segmentation_io.rs`: `load_ref_stm(text, chan, off, dur, exclude_nontrans)`, `load_ref_csv(text, off, dur, pruning_thresh) -> (Segmentation, i64)`, `compute_errors(hyp, Option<&Segmentation>, nb_words) -> ScoreReport`, `to_vrcts_string`/`write_vrcts`.
- `audio.rs::read_audio(path, offset_sec, max_duration_sec) -> Result<Audio>`; `nn/blstm.rs`: `set_weights/get_weights/get_weights_derivatives/reset_weights_derivatives/update_weights` (Rprop state inside); `io/binary.rs::{read_matrix, write_matrix, read_weight_vector}`; `legacy_config.rs::parse_legacy_config`.
- Harness: `SpectralProbe`/`SignalProbe`/`TdcProbe`/`LtsvProbe` + reimpl families + `iof::fmtr` faithful shim + omp no-op shim + fixture extractor patterns (`scripts/extract_phase2b_fixtures.py`, `extract_phase3_fixtures.py`).

---

### Task 1: `io/matfile.rs` - MAT v5 writer (matrix + scalar)

**Files:**
- Modify: `src/rust/src/io/matfile.rs` (replace the stub)
- Create: `src/rust/tests/phase4a_matfile.rs`
- Create: `tests/test_phase4a_matfile.py`

**Interfaces:**
- Produces: `pub struct MatWriter` with `pub fn create(path: &Path) -> anyhow::Result<MatWriter>`, `pub fn write_matrix(&mut self, name: &str, rows: usize, cols: usize, data_col_major: &[f64]) -> anyhow::Result<()>`, `pub fn write_scalar(&mut self, name: &str, value: f64) -> anyhow::Result<()>` (1x1 matrix), `pub fn finish(self) -> anyhow::Result<()>`. Consumed by Tasks 6-7 (`saveResults`, driver `save_weights`).
- Consumes: nothing new. NO new crate deps (spec S6: no `.mat` reader; the `matfile` crate stays out - CLAUDE.md conventions note amended in Task 12).

- [ ] **Step 1: Failing tests**

`phase4a_matfile.rs` (byte-structure unit tests - the format is fixed, uncompressed v5):
```rust
// header_is_v5: 128-byte header; bytes 0..4 == b"MATL"; u16 at 124 == 0x0100; bytes 126..128 == b"IM".
// matrix_element_layout: write 2x3; assert miMATRIX(14) tag, array-flags miUINT32 class mxDOUBLE_CLASS(6),
//   dims miINT32 [2,3], name miINT8 "A" (8-byte padded), data miDOUBLE 6*8 bytes column-major, total 8-aligned.
// scalar_is_1x1: write_scalar -> dims [1,1], one f64.
// two_variables_sequential: offsets chain correctly (each element 8-byte aligned).
```
`test_phase4a_matfile.py` (pytest cannot call Rust - no PyO3 yet - so the seam is a COMMITTED sample): the pytest loads `tests/reference_data/phase4a/matwriter_sample.mat` with `scipy.io.loadmat` and asserts variables `A` (2x3, fixed known constants) and `s` (1.5) match bit-exact. The sample is written by an `#[ignore]`d Rust test (`cargo test --test phase4a_matfile -- --ignored write_sample`) and committed; it is byte-reproducible because the writer uses a FIXED header text string (`"MATLAB 5.0 MAT-file, written by speech-rs"`, space-padded - no timestamp).

- [ ] **Step 2: Run tests, verify failure** - `cargo test --test phase4a_matfile` fails to compile (`MatWriter` missing).

- [ ] **Step 3: Implement `MatWriter`**

Uncompressed MAT v5 per the MathWorks spec: 116-byte text header (FIXED string, space-padded), 8 zero bytes subsys, u16 `0x0100`, `b"IM"`; per variable one miMATRIX element containing sub-elements [array flags (miUINT32, 8 bytes: `mxDOUBLE_CLASS = 6`, flags 0), dimensions (miINT32, [rows, cols] as i32), name (miINT8, bytes, 8-padded; use the LONG format unconditionally for simplicity - scipy accepts both), real data (miDOUBLE, column-major f64)]; every sub-element padded to 8-byte boundary; the miMATRIX tag's size = total payload bytes. `write_scalar(name, v)` = `write_matrix(name, 1, 1, &[v])`. Doc-comment: value-level parity contract, legacy writes zlib via matio (`Helpers.hpp:450,469`) - byte parity is out of scope by design (spec S6).

- [ ] **Step 4: Generate + commit the sample, run all tests**

Run: `cd src/rust && cargo test --test phase4a_matfile -- --include-ignored` then `uv run pytest tests/test_phase4a_matfile.py -v`. Expected: PASS both; scipy loads the sample.

- [ ] **Step 5: Commit**
```bash
git add src/rust/src/io/matfile.rs src/rust/tests/phase4a_matfile.rs tests/test_phase4a_matfile.py tests/reference_data/phase4a/matwriter_sample.mat
git commit -m "feat(phase4a): MAT v5 writer (matrix + scalar), scipy-validated

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `engine/corpus.rs` - Corpus + CorpusItem

**Files:**
- Modify: `src/rust/src/engine/corpus.rs`
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Produces:
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct CorpusItem { pub file_name: String, pub ref_seg: String, pub language: String,
    pub dialect: String, pub class_index: i32, pub file_id: i32, pub weight: f64 }  // adds file_id to the stub
#[derive(Debug, Clone, Default)]
pub struct Corpus { /* items, language_count: BTreeMap<String,i32>, dialect_count: BTreeMap<String,i32>,
    class_count: BTreeMap<i32,i32>, mapping: BTreeMap<String, BTreeMap<String,i32>> */ }
impl Corpus {
    pub fn from_config(map: &IndexMap<String, String>) -> anyhow::Result<Corpus>;
    pub fn nb_of_files(&self) -> usize;
    pub fn item(&self, pos: usize) -> &CorpusItem;
    pub fn class_count(&self) -> &BTreeMap<i32, i32>;
    pub fn language_count(&self) -> &BTreeMap<String, i32>;
}
```
- Consumes: `legacy_config`-parsed `IndexMap` (keys `language2classmapping`, `fileslisting`, fallback `files`/`refsegfiles`/`reflangfiles`/`refdialfiles` comma-lists).

- [ ] **Step 1: Failing tests** (inline): `mapping_parse` (crafted `lang;dial;3` lines incl. blank + missing-fields lines); `listing_nested_quirk` - line `f.wav;;eng;;0.5` must yield weight **1.0** (the nested parse never reaches token 4 because token 1 (`refseg`) is empty ONLY gates its own field - CAREFUL, transcribe `:74-96` exactly: each level nests inside `tokens[k].size() > 0` of the PREVIOUS field, so an empty `dial` (token 3) means weight/fileId are never parsed); `unknown_language_mutates_mapping` (second file with same unknown lang hits the now-cached `-1`); `no_listing_fallback` (`files`/`refsegfiles` comma-lists, defaults `lang="unk"`, weight 1.0, file_id 1); `class_count_accumulates`.
- [ ] **Step 2: Verify failure** - `cargo test corpus` (lib tests) fails to compile.
- [ ] **Step 3: Implement** - direct transcription of `Corpus.cpp:7-131,164-194` with `// legacy:` provenance; missing mapping/listing file -> `anyhow::bail!` (legacy `exit(1)` - the error surfaces to `main` in Task 10). Weight parse via `str::parse::<f64>` mirroring the `istringstream >> double` accept-on-success semantics (parse failure -> keep 1.0). Console prints dropped (display-only), NOTE the "normalizing coefficients" print (`:123-130`) is log-only, not state - drop with a doc-comment.
- [ ] **Step 4: `cargo test` green; clippy/fmt clean.**
- [ ] **Step 5: Commit** (`feat(phase4a): Corpus/CorpusItem CSV parsing incl. nested-token quirk`).

---

### Task 3: NN lifecycle surface - `BlstmNetwork` InputStatistics + weightsFile load + `save_weights`; driver delegates

**Files:**
- Modify: `src/rust/src/nn/blstm.rs`
- Modify: `src/rust/src/tasks/sad.rs` (spectral + signal driver delegates; `#[derive(Clone)]` where missing)
- Modify: `src/rust/src/tasks/segmenter.rs` (nothing unless a driver lacks `Clone` via contained types)
- Test: inline unit tests + `src/rust/tests/phase4a_lifecycle.rs`

**Interfaces:**
- Produces (on `BlstmNetwork`): `pub fn input_statistics(&self) -> &InputStatistics`, `pub fn reset_input_statistics(&mut self)`, `pub fn accumulate_input_statistics(...)` wired INSIDE the existing scoring forward at the exact `BLSTMNeuralNetwork.cpp:380-415` sites (transcribe the enclosing branch conditions + the `leftCols(getInputSize())` width per branch; first call `from_matrix`, later `update`), `pub fn save_weights(&self, filename: &str, weights_derivatives: &Array2<f64>, stats: &InputStatistics) -> anyhow::Result<()>` (writes `weights_<filename>` + `weightsDerivatives_<filename>` via `io::binary::write_matrix`, then a `.mat` at `<filename>` with `nbOfInputs`/`meanInputs`/`stdInputs` via `MatWriter` - `BLSTMNeuralNetwork.cpp:312-331`), and `pub fn load_weights_file(map, prefix, net) -> Result<()>`-style handling of `<prefix>_weightsFile` (`:122-150`: empty -> skip; too-few -> error; too-many -> warning + truncate per the exact legacy semantics at `:132-150` - read them before coding).
- Produces (on `BlstmSpectralSegmenter` + `BlstmSignalSegmenter`): `pub fn is_back_prop_activated(&self) -> bool`, `pub fn update_weights(&mut self, derivs: &Array2<f64>, cost: f64)`, `pub fn save_weights(&self, filename: &str, derivs: &Array2<f64>, stats: &InputStatistics) -> anyhow::Result<()>`, `pub fn input_statistics(&self) -> &InputStatistics`, `pub fn reset_weights_derivatives(&mut self)` - thin delegates to the net (`BLSTMSpectralSegmenter.cpp:111-113` pattern).
- Consumes: Task 1 `MatWriter`; existing net methods.

- [ ] **Step 1: Failing tests** - `phase4a_lifecycle.rs`: `input_stats_accumulate_across_files` (two synthetic forwards -> stats equal a hand-chained `from_matrix`+`update` on the same matrices - reuse the Phase 1 `InputStatistics` fixtures' semantics; STRICT bits, pure arithmetic); `save_weights_writes_three_artifacts` (tempdir; `.bin` round-trips via `io::binary::read_matrix`; `.mat` structurally valid via the Task 1 byte checks); `weights_file_too_many_truncates` / `too_few_errors` (synthetic config + crafted `.bin`).
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement.** Add `#[derive(Clone)]` to every driver + contained type that lacks it (`BagOfProcessors` cloning in Tasks 4/7 requires it; `Rprop` state must clone - verify `nn/train.rs`). The accumulation gating conditions come from reading `BLSTMNeuralNetwork.cpp:370-415` in full - the branch structure (normalization type / MLP vs BLSTM) decides which `leftCols` width feeds the stats; transcribe with provenance.
- [ ] **Step 4: Tests green; clippy/fmt; existing phase2/2b/3 suites still green (`cargo test`).**
- [ ] **Step 5: Commit** (`feat(phase4a): BlstmNetwork InputStatistics + weightsFile load + save_weights; driver lifecycle delegates`).

---

### Task 4: `Processor` enum + `BagOfProcessors` construction/dispatch

**Files:**
- Modify: `src/rust/src/engine/bag_of_processors.rs`
- Modify: `src/rust/src/cli.rs` (the `Mode { kind: ModeKind, verbose: bool }` struct replaces the bare enum HERE - the bag ctor needs mode letters; Task 10 only adds arg parsing on top)
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Produces: `cli::Mode`/`cli::ModeKind` per the "Mode semantics" binding section, plus:
```rust
#[derive(Clone)]
pub enum Processor { Tdc(TdcSegmenter), Ltsv(LtsvSegmenter),
    Spectral(BlstmSpectralSegmenter), Signal(BlstmSignalSegmenter) }
#[derive(Clone)]
pub struct BagOfProcessors { /* nb_of_conf, num_outer_threads: i32, offset_begin: f64,
    duration_max: f64, file_type: i32, lock_files_dir: String, lock_files_prefix: String,
    algo_types: Vec<i32>, pruning_thresholds: Vec<f64>, processors: Vec<Processor>,
    exclude_nontrans: bool */ }
impl BagOfProcessors {
    pub fn from_configs(configs: &mut [IndexMap<String, String>], mode: Mode) -> anyhow::Result<BagOfProcessors>;
    pub fn nb_of_conf(&self) -> usize;
    pub fn nb_of_threads(&self) -> i32;
    pub fn algo_type(&self, pos: usize) -> i32;
    pub fn is_work_distributed(&self) -> bool;           // lock_files_dir non-empty (BagOfProcessors.h:25)
    pub fn is_back_prop_activated(&self, pos: usize) -> Vec<bool>;
    pub fn get_weights(&self, pos: usize) -> Vec<Vec<f64>>;
    pub fn set_weights(&mut self, pos: usize, new_weights: &[Vec<f64>]) -> anyhow::Result<()>;
    pub fn get_input_statistics(&self, pos: usize) -> Vec<InputStatistics>;
    pub fn get_weights_derivatives(&self, pos: usize) -> Vec<Array2<f64>>;
}
```
(`_ConfigIndex` disappears: `processors[pos]` is already per-config - the enum replaces the typed-vector indirection; doc-comment records the equivalence.)
- Consumes: Task 3 delegates; driver `from_legacy` ctors; `<prefix>_weightsFile` loading from Task 3.
- Note: config defaults/keys from `BagOfProcessors.cpp:10-27`: `numOuterThreads` (REQUIRED, no default), `Audio_offset` 0.0, `Audio_max_duration` 3.6e6, `File_Type` 0, `LockFilesDir`/`LockFilesPrefix` "", per-config `Algo_choice` (required), `Pruning_Threshold` 0.0; the ctor SETS `files`/`refsegfiles`/`reflangfiles` to `""` in each config map BEFORE constructing drivers (`:21-23`, memory quirk - reproduce; `refdialfiles` NOT cleared). Algo 0/5/6 -> `anyhow::bail!("Algo {n} not ported (Phase 4b)")`. `File_Type != 0` -> bail (non-wav paths unported; IMPROVEMENTS entry).

- [ ] **Step 1: Failing tests**: `constructs_algo_1_through_4` (four single-config bags from the committed phase2b fixture configs + the phase0 real config; weights key pointing at the phase0 `NNweights_config1.bin` for algo 3); `algo_5_bails`; `key_clears_applied` (map mutated); `dispatch_vec_shapes` (`get_weights` len 0 for algo 1/2, len 1 for 3/4; `is_back_prop_activated` == `vec![false]` default for algo 1/2 per `:85`).
- [ ] **Step 2: Verify failure.** **Step 3: Implement** (transcribe `:9-146`). **Step 4: Green + lints.**
- [ ] **Step 5: Commit** (`feat(phase4a): BagOfProcessors construction + Processor enum dispatch`).

---

### Task 5: `SegmentationFunction` - per-file flow + result assembly

**Files:**
- Modify: `src/rust/src/engine/bag_of_processors.rs`
- Test: `src/rust/tests/phase4a_segfn.rs`

**Interfaces:**
- Produces: `pub fn segmentation_function(&mut self, item: &CorpusItem, mode: Mode) -> anyhow::Result<BTreeMap<usize, BTreeMap<usize, Vec<f64>>>>` (config -> channel -> 18-col result vector; the legacy `jj` param only feeds the lock-file name - stubbed with the lock block, doc-comment + IMPROVEMENTS).
- Consumes: `audio::read_audio(item.file_name, offset_begin, duration_max)`; `Segmentation::new((frames-1)/rate)` per channel; reference load dispatch by `item.ref_seg` extension - `.stm` -> `load_ref_stm(text, chan, off, dur, self.exclude_nontrans)` per channel, `.csv` -> `load_ref_csv(text, off, dur, pruning_thresholds[pos])`, `.trs` -> bail (unported, IMPROVEMENTS), none/other -> no reference; `Segmenter::get_segmentation`; `compute_errors(hyp, ref, nb_words)`; `to_vrcts_string`/`write_vrcts`; driver `cumulative_error()`/`nb_of_classif()`.

- [ ] **Step 1: Failing tests** (use `tests/reference_data/phase1/excerpt_2ch_8k.wav` + the phase2b TDC config + a crafted 2-channel STM in a tempdir):
```rust
// scored_mode_result_columns: mode -m; assert cols 14,15,16 == 0.0 and col 17 == nb_of_classif; col 3 is
//   NONZERO (timing) but only asserted finite; cols 0-2 match a direct compute_errors call on the same hyp/ref.
// missing_reference_in_scored_mode_errors: mode -m, item.ref_seg="" -> Err (legacy exit(1), :302-305).
// unscored_mode_zero_columns_and_vrcts: mode -s, no ref; cols 0,1,2,4 == 0.0; VRCTS written NEXT TO the
//   audio (tempdir copy of the wav; legacy :394-401 no-dumpDir branch strips the 4-char extension).
// dump_dir_vrcts: driver_cfg dump dir set -> VRCTS under dumpDir with basename quirk (:352-356).
// speech_duration_walk: col 6 equals the manual walk over hyp segments (:324-330).
```
- [ ] **Step 2: Verify failure.** **Step 3: Implement** (transcribe `:207-407` minus the lock block; keep the `audio.reset()` per-config call `:403`; the per-config `Segmentation` seeding happens INSIDE the config loop `:260`). The mandatory-reference check `:302-305` maps to "scored mode AND `compute_errors` would get no reference" -> bail with the legacy message text.
- [ ] **Step 4: Green + lints.** **Step 5: Commit** (`feat(phase4a): per-file SegmentationFunction + 18-col result assembly`).

---

### Task 6: `saveAndUpdate` + bag `saveWeights`/`updateWeights`

**Files:**
- Modify: `src/rust/src/engine/bag_of_processors.rs`
- Test: `src/rust/tests/phase4a_save_update.rs`

**Interfaces:**
- Produces:
```rust
pub fn save_and_update(&mut self, filename: &str, results_per_conf: &[Array2<f64>],
    best_cost: &mut BTreeMap<usize, f64>, derivs: &BTreeMap<usize, Vec<Array2<f64>>>,
    stats: &BTreeMap<usize, Vec<InputStatistics>>,
    cost_mem_row: &mut [f64], bad_classif_row: &mut [f64],
    cost_lid_row: &mut [f64], bad_classif_lid_row: &mut [f64]) -> anyhow::Result<()>;
```
plus private `save_weights(pos, ...)` (`:148-181`: criteria - algo 3/4 `cost`, algo 5 `badClassifLID+costLID`, algo 6 `cost+costLID`; gate `best_cost[pos] > criterion`; filename `bestNNWeight_<pos+1>_<filename>`) and `update_weights(pos, ...)` (`:183-205`; algo 3/4 criterion `cost`). The `costLID = -1.0` when `totalSpeechDuration < 1e-3` happens in `save_and_update` (`:465`) BEFORE `update_weights` and AFTER `save_weights` - order is load-bearing.
- Consumes: Task 3 driver delegates; Task 1 `MatWriter` (via driver save). Column sums/means via explicit ascending loops over rows (NOT `ndarray::sum_axis` - product contract; sums are the golden-bearing values).
- `PrintConfusionMatrix` (`:473-598`): stub `fn print_confusion_matrix(..) -> f64 { unreachable!("Algo 5/6 - Phase 4b") }` with a doc-comment carrying the `>150`/`-200` in-band decode summary for 4b.

- [ ] **Step 1: Failing tests** (crafted `results_per_conf` matrices, algo-1 bag - update is a no-op there, so the aggregation math is isolated):
```rust
// aggregation_bit_exact: hand-computed sums/means on a 4x18 crafted matrix; cost normalization guard
//   (counter 0 -> cost NOT divided); WER percent scaling + nb_words guard (:426-433).
// best_cost_gate_fires_and_skips: algo-3 bag (real net, tempdir): epoch A cost 5.0 -> saved (files exist),
//   best updated; epoch B cost 7.0 -> NOT saved (mtime/content unchanged), best unchanged. STRICT.
// cost_mem_rows_written: the four row slices receive cost/badClassif/costLID/badLIDClassif.
// update_called_with_neg_costlid: speechDuration sum < 1e-3 -> the update path receives costLID == -1.0
//   (observable on algo 3 via unchanged... NOT observable in 4a since algo 3 update ignores costLID ->
//   assert via a test-only #[cfg(test)] hook recording the passed value; doc-comment notes 4b makes it live).
```
- [ ] **Step 2: Verify failure.** **Step 3: Implement** (transcribe `:409-471`). **Step 4: Green + lints.** **Step 5: Commit** (`feat(phase4a): saveAndUpdate aggregation + best-weight lifecycle`).

---

### Task 7: `CorpusProcessor` - mode dispatch, epoch loop with static lanes, transformResults, saveResults, gradCheck

**Files:**
- Modify: `src/rust/src/engine/corpus_processor.rs`
- Test: `src/rust/tests/phase4a_corpus_processor.rs`

**Interfaces:**
- Produces:
```rust
pub struct CorpusProcessor { /* training_epochs: usize, epsilon: f64, corpus: Corpus,
    processors: BagOfProcessors, mode: Mode, output_file_name: String,
    results: BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, Vec<f64>>>>,  // file -> conf -> chan
    res_per_conf: Vec<Array2<f64>>, results_e: Array2<f64>,
    cost_mem/bad_classif_mem/cost_lid_mem/bad_classif_lid_mem: Array2<f64>,
    best_cost: BTreeMap<usize, f64> */ }
impl CorpusProcessor {
    pub fn new(configs: Vec<IndexMap<String, String>>, mode: Mode) -> anyhow::Result<CorpusProcessor>;
    pub fn run(&mut self) -> anyhow::Result<()>;         // :113-137 dispatch
    // private: run_solo :74-81, train :83-111, run_epoch(epoch, mode, derivs, is_log) :140-235,
    //          grad_check(epsilon) :237-340, transform_results :342-389, save_results(epoch) :391-404
}
```
- Consumes: Tasks 2/4/5/6 + Task 1 (`save_results` writes the 5 named matrices: `MultiConfigResults`, `CostMem`, `BadClassifMem`, `CostLIDMem`, `BadClassifLIDMem`, each `topRows(epoch+1)`).

- [ ] **Step 1: Static-lane `run_epoch`** - the core (spec S3). Implementation shape:
```rust
let n = self.processors.nb_of_threads().clamp(1, nb_files as i32) as usize;
let lane_of = |j: usize| j % n;
// Per lane: clone the epoch-start bag ONCE, walk its files in ascending j, collect (j, results_j) plus,
// for each conf in the derivs map request, the post-file get_weights_derivatives + get_input_statistics.
// rayon::scope: one spawn per lane writing into a per-lane Vec (no shared mutation).
// After the scope: fold ascending j (BTreeMap insert + derivs[ii][kk] += tmp[kk] elementwise; stats
// first-in from_copy then .update()) - transcribing the critical-section body :178-202 in FILE ORDER.
```
The legacy first-assignment-vs-`+=` map semantics (`:184-191`: first file CREATES the entry by move, later files `+=`) is reproduced by the ascending fold. `saveAndUpdate`+`saveResults` gating per `:215-234` (incl. the results-EMPTY branch `:232-234` that still calls `save_results`).
- [ ] **Step 2: `train`/`run_solo`/`run` dispatch + ctor** - transcribe `:48-137` (output-file truncation for m/t at ctor; epochs clamp; `isWorkDistributed` kill; `_BestCost` 1e20 seeds; the epsilon>0 train-then-gradCheck path; mode-letter gates with the exact legacy `cerr` messages as `anyhow` errors where legacy only prints - legacy CONTINUES after the cerr at `:119` (no exit); mirror: log via `eprintln!` and continue, matching observable behavior).
- [ ] **Step 3: `transform_results` + `save_results`** - transcribe `:342-404`. Row layout `[file+1, conf+1, chan+1, res...]`; per-conf matrices sized `2*nb_files` rows then truncated (conservative resize == truncate to counters).
- [ ] **Step 4: `grad_check`** - transcribe `:237-340` (bag snapshot clone `:238`; `_TrainingEpochs = 0`; per-weight restore-perturb-run; cost columns 4/17 for algo 3/4 via the binding table; relative floor 1e-24; mean abs + rel errors returned as `pub struct GradCheckReport { pub mean_error: f64, pub mean_relative_error: f64, pub per_weight: Vec<(f64, f64, f64)> }` (backprop-normalized, numerical, abs-diff) so Task 9's golden can replay it).
- [ ] **Step 5: Failing-then-green tests**: `mode_dispatch_matrix` (epsilon/epochs/mode-letter combinations -> which private paths run; use a 1-file TDC corpus in tempdir); `transform_results_ordering` (crafted results map with a GAP file index -> ascending iteration, conservative resize); `lanes_n1_equals_sequential` (TDC 3-file corpus: `run_epoch` with n=1 == a hand-sequential loop, STRICT bits); `save_results_variables` (scipy-checkable via the Task 1 sample pattern: read back the written `.mat` bytes with the unit-test parser helpers); `grad_check_synthetic` (the 142-weight synthetic config from `phase3_gradcheck.rs` reused on a 1-file corpus; assert `mean_relative_error < 5e-4` - the corpus-level analog of the Phase 3 bound; canary-gated values).
- [ ] **Step 6: Commit** (`feat(phase4a): CorpusProcessor epoch loop with static lanes + transformResults + saveResults + corpus gradCheck`).

---

### Task 8: Harness tier-1 (REAL compiled driver stack) + fixtures + Rust goldens

**Files:**
- Modify: `tools/oracle_harness/main.cpp` + `tools/oracle_harness/build.sh` (link `CorpusProcessor.cpp BagOfProcessors.cpp Corpus.cpp CorpusItem.cpp VRCTSpart.cpp` if not already + Homebrew libmatio `-lmatio`; VRCTSpart links but is never constructed - Algo 0 unused)
- Create: `scripts/extract_phase4a_fixtures.py`
- Create (generated): `tests/reference_data/phase4a/` (corpora + configs + converted `.bin` dumps + `manifest.json`)
- Create: `src/rust/tests/phase4a_tier1_e2e.rs`, `tests/test_phase4a_fixtures.py`
- Modify: `src/rust/tests/common/mod.rs` (`fixture_phase4a`/`load_bin_phase4a`)

**Interfaces:**
- Produces: committed corpora (3 wav copies of `excerpt_2ch_8k.wav` under `corpus/f{1,2,3}.wav` + crafted 2-channel STMs with SPEECH/OTHER spans inside 2.0s + `fileslisting.csv` (exercising the weight column AND one nested-quirk row) + `language2classmapping.csv`); config variants `tier1_tdc.config` (Algo 1) + `tier1_ltsv_powermel.config` (Algo 2, the real-crossings variant) with `numOuterThreads 1`, `Neural_Networks_BackPropagation_Epochs 2`, `multiConfigResultsOutputFile <name>.mat`, relative corpus paths; harness stage `phase4a_tier1` that runs the REAL `CorpusProcessor(configs, mode).run()` for: solo (`-s`, epochs overridden to 0), train (`-m`, epochs 2), multiconfig (`-m` with both configs); the extractor converts every produced `.mat` via `scipy.io.loadmat` into per-variable `.bin` (f64 column-major, i64 dims header - reuse the `io::binary` codec format so `load_bin_phase4a` reads them), ZEROES the timing column of `MultiConfigResults` (records `masked_cols: [6]` in the manifest), copies VRCTS dumps, and hash-guards prior phases.
- Consumes: the omp no-op shim (already forces sequential == N=1), the `iof::fmtr` shim, existing excerpt + STM fixture patterns.

- [ ] **Step 1: Harness stage + corpora + configs.** The corpora/configs are AUTHORED (committed) files; the harness consumes them from the fixture dir via relative paths. Mode strings passed as writable `char` buffers (legacy mutates `mode`). Assert in-harness that `Mat_Open` on each output succeeds before exiting (fail fast on matio issues).
- [ ] **Step 2: Extractor** (mirror `extract_phase2b_fixtures.py`): build harness, run stage in a throwaway dir seeded with the corpora, scipy-convert, mask, manifest (`fixture list + masked_cols + run modes + libmatio version`), TWICE-regenerate byte-check, prior-phase hash guard.
- [ ] **Step 3: Rust goldens** `phase4a_tier1_e2e.rs`:
```rust
// solo_tdc_matches: CorpusProcessor::new(parse(tier1_tdc.config with epochs->0), Mode::solo()) .run();
//   read back our .mat via the byte-parser helper; compare MultiConfigResults (mask col 6),
//   CostMem/BadClassifMem vs the converted fixtures. TDC chain is libm-bearing -> canary-gated compare.
// train_ltsv_two_epochs_matches: epochs=2 run; CostMem is (epochs+2) x 1; all rows compared.
// multiconfig_matches: both configs; ResultsE row interleaving (file,conf,chan) pinned.
// vrcts_byte_equal: our write_vrcts output == the harness dump (byte-exact; fmtr shim proven in 2b).
// listing_parse_vs_fixture: Corpus::from_config on the committed listing == manifest counts (STRICT).
```
- [ ] **Step 4: `tests/test_phase4a_fixtures.py`** - manifest present, masked_cols recorded, every referenced `.bin` exists + shapes parse, scipy re-reads the RAW harness `.mat` outputs identically to the converted `.bin` (guards the conversion itself).
- [ ] **Step 5: Non-vacuity checks (in-test, per the Phase 3 standard)**: `CostMem` rows differ across epochs OR are proven constant-by-construction (TDC/LTSV have no weight updates - the EXPECTED result is identical rows; assert identity EXPLICITLY and document: tier-1 pins the loop/aggregation plumbing, not weight evolution - that is tier-2's job).
- [ ] **Step 6: Commit** (`feat(phase4a): tier-1 real-CorpusProcessor fixtures + corpus-level e2e goldens`).

---

### Task 9: Harness tier-2 CorpusProbe (Algo 3, reimpl-swap) + train/gradCheck goldens

**Files:**
- Modify: `tools/oracle_harness/main.cpp` (CorpusProbe stage), `scripts/extract_phase4a_fixtures.py`
- Create: `src/rust/tests/phase4a_train_golden.rs`, `src/rust/tests/phase4a_gradcheck_golden.rs`
- Create (generated): fixtures under `tests/reference_data/phase4a/`

**Interfaces:**
- Produces: harness-local `runCorpusTrain(...)`/`runCorpusGradCheck(...)` transcribing `CorpusProcessor::train/run(epoch)/gradCheck` + `BagOfProcessors::SegmentationFunction/saveAndUpdate` (provenance comments), operating on REAL compiled `SpectralProbe` objects with ONLY the `getSegmentation` call swapped for the existing 2b `SpectralProbe` reimpl entry (the real `updateWeights`/Rprop/`saveWeights` machinery runs REAL). Configs: `tier2_spectral.config` = the phase0 real config re-pointed at a 2-file corpus (f1/f2 wavs + STMs), `BLSTM_BackPropagationActivated -> true` override (the established `set_val` pattern), epochs 3. Dumps per epoch: the full 33,671 weight vector (`tier2_weights_epoch{0..4}.bin` - epoch 0 = post-solo, 4 = post-final), `CostMem` etc. converted as in Task 8, the `bestNNWeight_*` artifacts, VRCTS. GradCheck stage: `tier2_gradcheck.config` = the 142-weight synthetic net config (reuse the phase3 synthetic-config generator in the harness) on a 1-file corpus, `maxWeights=10` cap parameter (DEVIATION from the legacy full sweep, recorded in the manifest as `gradcheck_max_weights: 10`), dumping per-weight `[analytic_col0, analytic_col1, numerical]` rows + the two mean errors -> `tier2_gradcheck.bin`.
- Consumes: 2b `SpectralProbe`, phase0 real config + `NNweights_config1.bin`, Task 8 extractor plumbing.
- SECONDARY structural probe (2b convention): run the same train once with the REAL Eigen forward; assert segment count/type equality per epoch vs the reimpl run, record `SEG_STRUCT` max_dt lines in the manifest; ABORT generation on structural mismatch.

- [ ] **Step 1: Harness stages + regenerate fixtures** (twice, byte-identical; prior-phase hash guard).
- [ ] **Step 2: Rust goldens**:
```rust
// phase4a_train_golden: build CorpusProcessor from tier2_spectral.config (weights loaded via the
//   Task 3 weightsFile path re-pointed at NNweights_config1.bin), run(); after each epoch capture
//   processors.get_weights(0)[0] via a test-observation hook (pub(crate) accessor or re-run epoch
//   boundaries) and assert vs tier2_weights_epoch*.bin - canary-gated per element (NN chain).
//   NON-VACUITY: consecutive epoch vectors DIFFER (assert), and the bestNNWeight gate fired on at
//   least one epoch AND skipped on at least one (read the manifest's recorded best-cost trajectory;
//   if the natural trajectory does not skip, adjust epochs/config in Task 9 Step 1 until it does -
//   MEASURED, not assumed).
// phase4a_gradcheck_golden: run grad_check on the same synthetic config/corpus; replay the 10 weights;
//   assert per-weight [analytic, numerical] and the two means vs tier2_gradcheck.bin (canary-gated);
//   assert mean_relative_error < 5e-4 (bound doc-commented as in phase3_gradcheck.rs).
```
- [ ] **Step 3: pytest additions** - fixture presence + `gradcheck_max_weights` + SEG_STRUCT records asserted `max_dt == 0.0`.
- [ ] **Step 4: Commit** (`feat(phase4a): tier-2 CorpusProbe train/gradCheck fixtures + epoch-chained weight goldens`).

---

### Task 10: CLI + `main` wiring + integration test

**Files:**
- Modify: `src/rust/src/cli.rs` (Mode struct + arg parsing), `src/rust/src/main.rs`, `src/rust/src/lib.rs` (re-export `engine::corpus_processor::CorpusProcessor`)
- Test: `src/rust/tests/phase4a_cli.rs`

**Interfaces:**
- Produces:
```rust
pub struct Mode { pub kind: ModeKind, pub verbose: bool }   // replaces the bare enum; from_flag keeps case
pub enum ModeKind { Solo, Image, UnitTest, Multi }
pub struct CliInvocation { pub mode: Mode, pub configs: Vec<IndexMap<String, String>> }
pub fn parse_cli(args: &[String]) -> anyhow::Result<CliInvocation>;
// Single-config modes: LAST arg is the config; middle args are --key=val overrides applied via
// set-or-remove (empty val REMOVES the key; FastSpeechProcessing.cpp:47-52 + usage :85).
// Multi mode: every arg after the flag is a config path. exclude_nontrans read from configs[0]
// (default false) and stored INTO the parsed maps for the bag (threaded, not a global).
```
`main.rs`: parse -> `CorpusProcessor::new(configs, mode)?.run()` -> exit code (usage + code 2 on parse failure, matching the current stub's contract; engine errors -> eprintln + code 1). Usage text mirrors `:78-87` with `.config` wording.
- Consumes: Tasks 2-9.

- [ ] **Step 1: Failing tests**: `override_applied` / `override_empty_removes_key` / `multi_collects_all_configs` / `case_sets_verbose` (parse-level, no engine run); `binary_end_to_end` - `std::process::Command::new(env!("CARGO_BIN_EXE_speech"))` with `-s <tier1_tdc.config epochs->0, tempdir cwd seeded with the corpus>`; assert exit 0 + the output `.mat` exists + `MultiConfigResults` values match the tier-1 fixture (masked col; canary-gated).
- [ ] **Step 2: Verify failure. Step 3: Implement. Step 4: Green + lints + full `./check_all.sh`.**
- [ ] **Step 5: Commit** (`feat(phase4a): full legacy CLI + main wiring + binary e2e test`).

---

### Task 11: Mutation battery (spec S7) - execute + record

**Files:**
- Modify: `IMPROVEMENTS.md` (measurements), plan checkboxes; NO production code changes survive this task.

Each mutation: apply, run the named suite, confirm FAIL, revert, confirm PASS. Record each outcome (suite + failing assert) in the task log and the relevant IMPROVEMENTS entries:

- [ ] Reduction fold order reversed (descending file index in `run_epoch`'s fold) -> `phase4a_train_golden` must FAIL. If it does NOT (sums bit-insensitive on this corpus), record the measurement and add a crafted-divergence note per spec S7.
- [ ] Cost column 4 -> 5 in `save_and_update` -> `phase4a_tier1_e2e::train_ltsv_two_epochs_matches` FAILS (CostMem).
- [ ] Counter-normalization guard dropped (`always divide`) -> `phase4a_save_update::aggregation_bit_exact` FAILS.
- [ ] Best-cost gate `>` -> `>=` -> `phase4a_train_golden` non-vacuity (fire+skip) FAILS.
- [ ] gradCheck epsilon sign flipped -> `phase4a_gradcheck_golden` FAILS.
- [ ] Lanes N=2 vs N=1 divergence measured: run the Rust engine (no fixture) on a 3-file corpus of the 2b `noOverlap` signal config at N=1 and N=2; record whether outputs differ (EXPECTED: differ - state chaining). Document in the IMPROVEMENTS lane-model entry. If they do NOT differ, investigate before recording (the 2b two-files golden says cross-file state is real).
- [ ] `costLID = -1.0` gate removal - UNREACHABLE observable in 4a (algo 3/4 update ignores costLID); record as deferred-to-4b in IMPROVEMENTS.
- [ ] Commit (`test(phase4a): mutation battery results recorded`).

---

### Task 12: Docs - README, CLAUDE.md, IMPROVEMENTS.md

**Files:**
- Modify: `README.md` (Roadmap: Phase 4 decomposition 4a-4d + Phase 4a done-paragraph in the established voice; module table rows for `engine/*`, `io/matfile.rs`, `cli.rs`)
- Modify: `CLAUDE.md` (module table: `bag_of_processors.rs` enum-over-concrete-types CORRECTION replacing the `Vec<Box<dyn Segmenter>>` sketch; `engine/*`/`matfile`/`cli` rows -> Implemented (Phase 4a); conventions: amend "matfile lands in Phase 4" to "hand-rolled v5 writer, no reader (spec S6)"; Project Overview stub list update)
- Modify: `IMPROVEMENTS.md` (spec S8 checklist: timing column; ctor output truncation; gradCheck mode-flip + epochs zeroing + no-mat-during-gradCheck; "N files" = file x channel rows; costLID -1 gate; lock-file stub; lane-model N-dependence; unscored-mode VRCTS next to audio; config key clears; listing nested-token parse quirk; File_Type != 0 unported; TRS reference loader unported; plus anything logged during Tasks 1-11)

- [ ] Write all three; cross-check every claim against the landed code (no aspirational statements).
- [ ] `./check_all.sh && ./lint_code.sh && uv run pytest tests` full green.
- [ ] Commit (`docs(phase4a): README/CLAUDE.md/IMPROVEMENTS refresh`).

---

### Task 13: smart-commit + finish

- [ ] Invoke the `smart-commit` skill, telling it to take the WHOLE `feature/phase-4a-engine-drivers` branch into account (docs/CLAUDE.md sync + final commit).
- [ ] Invoke `superpowers:finishing-a-development-branch` (user merges via PR himself; expect "keep branch as-is" or PR per his workflow). NEVER push.

## Self-Review

1. **Spec coverage**: S1 scope -> Tasks 1-10; S2 architecture (enum bag, Vec-valued API) -> Task 4; S3 lanes -> Task 7 (+11 measurement); S4 quirks -> Task 7; S5 gradCheck -> Tasks 7/9; S6 two tiers + .mat contract -> Tasks 8/9 (+1); S7 battery -> Task 11 (+ non-vacuity inline in 8/9); S8 -> Task 12; S9 risks: R1 matio (Task 8 fail-fast + shim fallback), R2 deriv-reset semantics (Task 9 fixtures arbitrate; Task 7 transcribes whatever `:178-202` implies), R3 seeding/ref-load (Task 5), R4 lanes (Tasks 7/11), R5 resolved (scipy conversion, Task 8). S10 process -> Task 13.
2. **Placeholders**: none - every "read legacy lines X" instruction names exact ranges the implementer transcribes (the repo's established transcription pattern), not open questions. Two deliberate measurement points (Task 9 best-cost trajectory, Task 11 fold-order sensitivity) specify the decision rule.
3. **Type consistency**: `Mode`/`ModeKind` (Task 10) is consumed by Tasks 4/5/7 - those tasks compile against the struct introduced there; to avoid a forward dependency, Task 4 Step 3 introduces the `Mode` struct in `cli.rs` as part of its implementation if Task 10 has not run (the bag ctor needs mode letters on day one). Order note added: **Task 4 carries the `Mode` struct change; Task 10 only adds parsing on top.** `Vec<Vec<f64>>` weights (bag) vs `Vec<f64>` (net) - conversion at the enum arms. `BTreeMap` keys `usize` everywhere.
