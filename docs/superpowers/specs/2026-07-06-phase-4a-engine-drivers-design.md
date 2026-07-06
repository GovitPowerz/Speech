# Phase 4a: Engine drivers (Corpus / BagOfProcessors / CorpusProcessor / CLI / .mat) - Design

Date: 2026-07-06
Status: approved (brainstorm 2026-07-06)
Branch: `feature/phase-4a-engine-drivers` (off `main`, which contains Phases 0-3 via PRs #2/#5/#6)

Phase 4 as listed in the README spans six subsystems. It is decomposed into
sub-phases; this spec covers **4a only**. The agreed sequence:

- **4a (this spec)**: Rust engine drivers - `engine/corpus.rs`,
  `engine/bag_of_processors.rs`, `engine/corpus_processor.rs`, `io/matfile.rs`,
  the full CLI, ending at a corpus-level E2E parity gate.
- **4b**: LID - `tasks/lid.rs` (Algo 5 `BLSTMSpectralLID`, Algo 6
  `TwinBLSTMSpectralLID`), `tasks/vrcts.rs` (Algo 0 external-tool adapter), and
  the LID-only branches of the bag (confusion print, paired weight lifecycle).
- **4c**: PyO3 seam (`speech-py` `forward_backward`, zero-copy numpy) + Python
  optimizer zoo / engine driver / batching / CheckGrad; TOML canonical config
  lands here (first real consumer - the Python side starts WRITING configs).
- **4d**: dataprep + held-out-corpus SAD/LID score parity + the tolerance
  comparison vs the original `-ffast-math` fsp binary (never bit-matchable;
  and the 2015 Linux binary likely does not run on this Darwin/arm64 machine,
  so 4d decides how/whether to execute it).

## S1. Scope

IN (all Rust; Python only as a test cross-check):

- `engine/corpus.rs` - `Corpus`/`CorpusItem`: parse the `fileslisting` and
  `language2classmapping` CSVs (`;`-separated), language/dialect/class counts,
  per-item weight. Ported from `Corpus.{h,cpp}` (225 LOC) + `CorpusItem.*` (82).
- `engine/bag_of_processors.rs` - ported from `BagOfProcessors.{h,cpp}` (659):
  construction from `Vec<parsed config>` (per-config `Algo_choice` dispatch,
  `Pruning_Threshold`, the `files`/`refsegfiles`/`reflangfiles` key clears,
  audio offset/duration/file-type from config 0), `SegmentationFunction`
  (per-file audio load, per-config `Segmentation` seeding + reference check,
  driver dispatch, the ~18+-column result-vector assembly incl. WER/LID slots
  and the timing column, VRCTS dump calls incl. the no-dumpDir
  write-next-to-audio branch), `saveAndUpdate` (colwise sums/means, cost =
  sums(4)/sums(last-col) and costLID = sums(14)/sums(last-1) with their >0
  guards, WER aggregation, `costMem` row writes, best-cost-gated `saveWeights`
  with per-algo criteria, `updateWeights` incl. the `costLID = -1.0`
  no-speech gate), and the `getWeights`/`setWeights`/`getWeightsDerivatives`/
  `getInputStatistics`/`isBackPropActivated` dispatch (Vec-valued per network,
  matching legacy).
- `engine/corpus_processor.rs` - ported from `CorpusProcessor.{h,cpp}` (443):
  `run()` mode dispatch (incl. the `_Epsilon>0` train-then-gradCheck path and
  its mode-flip to `m`), `runSolo`, `train` (epoch 0 solo -> N training epochs
  with mode forced to `m` -> final eval; `_CostMem`/`_BadClassifMem`/
  `_CostLIDMem`/`_BadClassifLIDMem` shaped `(epochs+2) x nConfs`), the
  parallel `run(epoch, mode, derivs, isLog)` with the static-lane model (S3),
  `gradCheck` (S5), `transformResults`, `saveResults`.
- `io/matfile.rs` - `.mat` v5 writer for f64 dense matrices (matio-compatible
  element encoding) + reader for tests (`matfile` crate).
- `cli.rs` / `main.rs` - full legacy CLI: `-s/-S/-i/-I/-t/-T` single-config
  modes with `--key=val` overrides (empty value REMOVES the key; overrides
  apply between the mode flag and the last arg, which is the config), `-m/-M`
  multi-config (all remaining args are configs), `exclude_nontrans` read from
  config 0 and threaded as a parameter to the STM reference loader (NOT a
  global), `CorpusProcessor` construction + `run()`, nonzero exit on error.
  Usage text corrected to `.config` (TOML deferred to 4c).
- Driver-level `save_weights` on the NN drivers (Algo 3/4): compose
  `bestNNWeight_<n>_<outfile>` and write the weight `.bin` incl. the mean/std
  tail from `InputStatistics` - the Phase 0a packer seam finally exercised by
  a full engine run (the deferred Phase 0a milestone).

Dispatch coverage: Algo 1-4 run for real. Algo 0/5/6 arms exist as typed
"not yet ported (Phase 4b)" errors; `PrintConfusionMatrix` (only reachable
from algo 5/6) stays a documented stub.

OUT: LID (4b), VRCTS adapter (4b), PyO3 + optimizer zoo + CheckGrad (4c),
dataprep + held-out score parity + fsp-binary comparison (4d), TOML (4c),
lock-file distributed mode (the `isWorkDistributed` guard that zeroes
`_TrainingEpochs`/`_Epsilon` IS ported; the lock-file create/skip logic itself
is cluster plumbing - stub + IMPROVEMENTS entry).

## S2. Architecture

**Bag representation.** `Vec<Processor>` where `Processor` is an enum over the
concrete driver types (`Tdc(TdcSegmenter)`, `Ltsv(LtsvSegmenter)`,
`Spectral(BlstmSpectralSegmenter)`, `Signal(BlstmSignalSegmenter)`, plus
stub arms `Vrcts`/`Lid`/`TwinLid`). This deviates from the
`Vec<Box<dyn Segmenter>>` sketch in CLAUDE.md deliberately: the legacy
dispatch is per-concrete-type (7 typed vectors + `_ConfigIndex`), lifecycle
methods and save criteria differ per algo, and 4b's TwinBLSTM needs paired
LID getters that a shared trait would bloat. The `Segmenter` trait keeps the
per-file `get_segmentation` role. The enum makes the CLAUDE.md module-table
entry for `bag_of_processors.rs` stale - update it in the docs task.

Bag-level lifecycle API mirrors legacy signatures: `get_weights(pos) ->
Vec<Vec<f64>>`, `set_weights(pos, &[Vec<f64>])`, `get_weights_derivatives(pos)
-> Vec<Array2<f64>>`, `get_input_statistics(pos) -> Vec<InputStatistics>`,
`is_back_prop_activated(pos) -> Vec<bool>` (one element per network; two for
algo 6 in 4b; empty for NN-free algos).

**Result vector.** `SegmentationFunction` per (config, channel):
`[100*Pfa_SPEECH, 100*Pmiss_SPEECH, 100*globalErrorRate(sum OTHER..EXCLUDED),
time_per_hour, cumulativeError, (frames-1)/rate, speechDuration, WER x7
(nbWords, corrects, subs, ins, dels, coverage, delay), LID slots (or 0.0,0.0),
[LID confusion cols], LIDNbOfClassif, NbOfClassif]`. Column 3 is WALL-CLOCK
time - it flows into the `.mat` by legacy design and is masked in every
comparison (S6). The unscored branch (`-s`/`-i` without reference) zeroes the
score columns and STILL writes VRCTS (next to the audio when no dumpDir);
reproduce both branches.

**Cross-epoch state flows through weights only**: the epoch loop processes
files on clones of the epoch-start bag; the member bag advances only via
`saveAndUpdate` -> `updateWeights` (long-lived Rprop state lives in the
member's nets). Reproduce exactly: per-file churn is discarded at epoch end.

## S3. Determinism: the static-lane model (closes risk R6)

Legacy: `#pragma omp parallel for firstprivate(processors) schedule(dynamic,1)`
+ `#pragma omp critical` accumulation. Two nondeterminism sources: (a)
segmenter state chains across files WITHIN a thread (proven real by the 2b
`noOverlap` two-files golden) under a dynamic schedule; (b) the gradient/stats
`+=` runs in completion order.

Port: file `j` -> lane `j mod N` (N = `numOuterThreads`, clamped to
`[1, nbFiles]`). Each lane clones the epoch-start bag once and chains state
sequentially through its files (rayon scoped spawn - one task per lane,
NOT per file; rayon is the locked convention). Per-file results and per-file derivative/stats contributions
are collected indexed by file; after the parallel section they are folded in
ASCENDING FILE ORDER (derivatives `+=` per element; `InputStatistics::update`
merge in the same order). Properties:

- N=1 is byte-identical to the legacy single-thread run == the omp-shimmed
  harness == the goldens. **N=1 is the parity mode.**
- N>1 is deterministic for fixed N but N-dependent (state chaining per lane),
  a strict improvement over legacy (nondeterministic even at fixed N).
  IMPROVEMENTS entry; measured in the mutation battery (S7).

Per-file derivative contribution note: `getWeightsDerivatives` is read from
the lane bag after each file; the drivers reset per file (verify against
`SegmentationFunction`'s reset semantics during implementation - Phase 3
landed `reset_weights_derivatives`; the legacy resets inside the driver's
`getSegmentation`/FFB path). If legacy does NOT reset between files within a
thread, the accumulation semantics differ - transcribe whatever the legacy
actually does; the tier-2 fixtures will catch a mismatch either way.

## S4. CorpusProcessor flow quirks (reproduce verbatim)

- Constructor: `multiConfigResultsOutputFile` default
  `MultiConfigResults.mat`; in m/t modes the output file is TRUNCATED (a
  single space written) at construction. `_BestCost[ii] = 1e20` seeds.
  Negative `Neural_Networks_BackPropagation_Epochs` clamps to 0.
  `isWorkDistributed()` (lock-files configured) zeroes epochs + epsilon.
- `run()` dispatch: epsilon>0 AND epochs>0 -> `train()` THEN `gradCheck`;
  gradCheck path permanently flips `_Mode[1]` to `m`. Mode letters gate
  everything (`m/M/t/T/i/I` may train; `s/S` may not).
- `train()`: `run(0, _Mode, .., true)` solo pass; epochs `1..=N` with a local
  mode `'m'` and `isLog=false`; final `run(N+1, _Mode, .., true)`.
  `_CostMem.block(epoch, ..)` row indexing; `saveAndUpdate` is called when
  `_TrainingEpochs > 0` for every epoch, else only when
  `epoch <= _TrainingEpochs` (i.e. epoch 0 for solo runs).
- `saveResults(epoch)` writes 5 named matrices: `MultiConfigResults`
  (`_ResultsE`), `CostMem`/`BadClassifMem`/`CostLIDMem`/`BadClassifLIDMem`
  truncated to `topRows(epoch+1)`. Note: during gradCheck's inner runs
  (`_TrainingEpochs` zeroed, epoch=1) NEITHER `saveAndUpdate` nor
  `saveResults` fires - no `.mat` churn mid-gradCheck.
- `transformResults`: iterate `_Results` (map keyed by file index) in
  ascending key order -> `BTreeMap` (or `Vec<Option<..>>`); rows are
  `[file+1, conf+1, chan+1, res...]`; per-conf matrices in the same order;
  conservative-resize to the actual row count.

## S5. gradCheck (corpus-level; the Phase 3 deferral)

Snapshot the whole bag (`Clone`), then for each network `ii` of config 0 with
backprop active: one analytic run `run(1, _Mode, derivs, true)`, then per
weight `k`: restore bag, perturb `+eps`, `run(1, 'm', .., false)`, extract
cost from `_Results` with the algo-dependent columns - algo 3/4: cost col 4,
counter last col; algo 5: col 14, counter last-1; algo 6: by network index -
then `-eps` likewise; numerical = `(c+ - c-)/(2 eps)`; compare against
`derivs[0][ii](k,0)/derivs[0][ii](k,1)` (the Nx2 normalize-at-read
convention from Phase 3); accumulate mean absolute + relative error
(relative floor `ref = max(|numerical|, 1e-24)`). Checks CONFIG 0 ONLY
(legacy `getWeights(0)`), reproduce. In 4a only algo 3/4 arms are live.

## S6. Oracle strategy (two tiers) + .mat parity

**Tier 1 - real compiled driver stack (strongest, zero transcription).**
Harness links `CorpusProcessor.cpp`, `BagOfProcessors.cpp`, `Corpus.cpp`,
`CorpusItem.cpp` (+ the already-linked segmenter/NN/segmentation TUs). The
existing no-op `omp.h` shim makes the real `run()` sequential == the N=1
parity mode. New harness dep: matio (`Mat_Create`/`Matrix2MatFile`) via
Homebrew `libmatio` (harness is local-only; CI consumes fixtures only);
fallback if linking fights back: a faithful mini-shim writing the same v5
byte layout. Corpora: 3-4 files (both channels) built from the committed
excerpt WAVs + STM references; configs Algo 1 (TDC) and Algo 2 (LTSV
powermel - the variant with real threshold crossings). Runs dumped: solo,
train (epochs=2: full epoch loop, aggregation, costMem - weight updates are
no-ops for NN-free algos, which is exactly why the ENTIRE driver logic can be
pinned against real compiled code), and a two-config `-m` run. The
matio-written `.mat` files are the value-parity source; VRCTS dumps from the
corpus path are additional byte goldens (exercises the `size()-4` basename
truncation).

**Tier 2 - CorpusProbe (transcription + reimpl swap).** Algo 3 under the
real `1_worker_1.config` + real 33,671-weight net: transcribe
`run`/`train`/`gradCheck`/`SegmentationFunction`/`saveAndUpdate` in the
harness, swapping ONLY the `getSegmentation` calls for the existing 2b
`SpectralProbe` reimpl family. Fixtures: 2-file corpus, 2-3 training epochs
with real Rprop chaining (per-epoch weight vectors bit-exact), final `.mat`
values, and a corpus-level gradCheck fixture on a tiny crafted config (the
142-weight Phase 3 synthetic shape) sweeping ~10 weights (analytic,
numerical, error accumulators, column indexing all pinned). Secondary
real-Eigen structural probe records SEG_STRUCT-style agreement per the 2b
convention (abort fixture generation on structural mismatch).

**.mat parity contract is value-level, never byte-level** (matio header
carries a timestamp): parse both sides, compare f64 payloads bit-exact,
EXCEPT the timing column (col 3 of the result vector = col 6 of
`MultiConfigResults` after the 3 id columns), which is masked everywhere.

## S7. Testing requirements (non-vacuity PROVEN, per the Phase 3 standard)

Suites: `phase4a_corpus_golden` (CSV parsing vs real `ConfigFiles/`
listings), `phase4a_matfile` (writer/reader round-trip property tests +
value parity vs matio fixtures + scipy cross-read in pytest),
`phase4a_tier1_e2e` (solo/train/multiconfig vs tier-1 fixtures),
`phase4a_train_golden` (tier-2 epoch-chained weights),
`phase4a_gradcheck_golden`, CLI integration test (in-process `cli::run` +
one spawned-binary smoke test).

Non-vacuity: the train fixture's cost trajectory must make the best-cost
gate FIRE on at least one epoch and SKIP on at least one; per-epoch weights
must actually differ; the gradCheck fixture's analytic and numerical columns
must be nonzero.

Mutation battery (each must break a golden; measure, don't assume):
- reduction fold order reversed (if bit-insensitive on the fixture, craft a
  corpus where it isn't, or document the measurement);
- cost column 4 -> 5; counter normalization dropped;
- best-cost gate `>` -> `>=`;
- `costLID = -1.0` no-speech gate removed (once observable; may defer to 4b
  if unreachable without LID - document);
- N=2 lanes vs N=1: measured divergence on a corpus containing the stateful
  `noOverlap` signal config (proves lane semantics are load-bearing). This
  corpus is Rust-side only (no harness fixture needed - the measurement
  compares the Rust engine against itself at two N values, reusing the 2b
  signal-noOverlap config + excerpt files);
- gradCheck epsilon sign flipped.

## S8. IMPROVEMENTS.md candidates (log during implementation)

Timing (wall-clock) column inside results/`.mat`; constructor truncating the
output file in m/t modes; gradCheck permanently flipping `_Mode` and zeroing
`_TrainingEpochs`; "N files processed" log counting file x channel rows;
`costLID = -1.0` gate semantics; lock-file distributed skip (stubbed);
lane-model N-dependence (deterministic improvement over legacy);
unscored-mode VRCTS write next to the audio file; the `files`/`refsegfiles`/
`reflangfiles` config-key clears in the bag constructor. Plus whatever
implementation surfaces.

## S9. Risks

- R1: matio linking on the harness env (fallback shim planned, S6).
- R2: per-file derivative reset semantics inside `SegmentationFunction`
  (S3 note) - tier-2 fixtures arbitrate.
- R3: `Segmentation` seeding + reference loading order in the bag (the 2b
  drivers were tested per-file with pre-seeded segmentations; the bag now
  owns seeding + STM/CSV reference load + the `exclude_nontrans` flag) -
  verify against `Segmentation(audio, pruning)` + reference-load call sites
  during implementation.
- R4: stateful-driver cross-file chaining vs lane cloning - pinned by the
  two-files-style fixtures; any mismatch is a bug in the lane model, not the
  drivers.
- R5: `.mat` reader crate fidelity - guarded by the scipy cross-read.

## S10. Process

superpowers loop: this spec -> writing-plans
(`docs/superpowers/plans/2026-07-06-phase-4a-engine-drivers.md`) ->
subagent-driven development (fresh implementer + reviewer per task; mutation
batteries re-run by reviewers) -> docs task (README roadmap + CLAUDE.md
module table incl. the bag-enum correction + IMPROVEMENTS.md) -> smart-commit
(whole branch) -> finishing-a-development-branch (user merges via PR).
