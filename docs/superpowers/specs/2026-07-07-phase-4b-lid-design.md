# Phase 4b: LID (Algo 5/6 + VRCTS adapter + LID corpus integration) - Design

Date: 2026-07-07
Status: approved (brainstorm 2026-07-07)
Branch: `feature/phase-4b-lid` (off `feature/phase-4a-engine-drivers` at `ab6a23d` -
4a is unmerged pending the user's PR, so 4b stacks per the established pattern)

Phase 4 decomposition recap: 4a engine drivers (DONE) -> **4b LID (this spec)** ->
4c PyO3 + Python optimizer + TOML -> 4d dataprep + held-out score parity + the
fsp-binary tolerance comparison.

## S1. Scope

IN (all Rust + harness; staged as a LADDER - each rung reuses the prior rung's
fixtures):

1. **LID scaffolding**: driver-side LID scoring state - `lid_cumulative_error`,
   `is_lid_correct` (100/0/-1), `lid_classification_errors` (the row vector with
   the `targetLID(targetIndex) = -2.0` construction: target column becomes
   `100*langID + 200` (range 200-300, always > 150), non-target columns
   `100*langID` in [0,100] - `TwinBLSTMSpectralLID.cpp:1318-1337`,
   `BLSTMSpectralLID.cpp:412-426`), `lid_nb_of_classif`, per-segment confusion.
   Placement: ACCESSORS ON THE DRIVER (the 2b convention - `cumulative_error`/
   `nb_of_classif` precedent), NOT fields on `Segmentation` (deviation from the
   legacy struct placement, consistent with the ported architecture; the
   `compute_errors` LID logging block incl. its `Confusion2String` call is
   display-only and dropped). Shared scoring core: `confusion_to_string`-
   equivalent pure function returning the aggregate error + normalized matrix
   (ANSI rendering dropped; the RETURNED ERROR is load-bearing -
   `Helpers.hpp:365-436`).
2. **`tasks/lid.rs::BlstmSpectralLid`** (Algo 5, `BLSTMSpectralLID.cpp` 460 LOC):
   single net (prefix `BLSTM`), no modes/CNN/LID2Segmentation; inline spectral
   setup + LTSV-driven `result_vec` + per-speech-segment scoring loop
   (`:346-401`) + the LID member writes (`:412-426`).
3. **phSeq reader**: `File_Type 1` in `audio.rs` - one-hot `external_features:
   Vec<Array2<f64>>` per `AudioStruct.cpp:163-172`; `read_audio` grows a
   file-type parameter (wav path unchanged); the bag's `File_Type` gate lifts
   for 1; cep (2) still bails.
4. **`tasks/lid.rs::TwinBlstmSpectralLid`** (Algo 6, `TwinBLSTMSpectralLID.cpp`
   1,422 LOC): SAD net (prefix `BLSTM`) + LID net (prefix `BLSTM_LID`), ALL 8
   modes (`_Mode` 0-7 incl. negative-mode branches transcribed as-is), all 3
   `_PostProcessMode` accumulators (sum-log / entropy-weighted / vote),
   `getTargetsLID` (`:170-219`), `LID2Segmentation` (Mode 6),
   `getBLSTMLIDInputSequence` (`:139-168` - the Modes-0/3 SAD-hidden-state
   concat with z-normalization, nearest-index resampling, empty fallback),
   `getLIDBLSTMParam`, `_MinNbOfFrames` guard, Mode-7 Gaussian noise via the
   fixed `constants.rs` random tables, `_DumpLIDInternals` (Mode-7 `.mat`
   dumps), the LID weight lifecycle (`updateWeightsLID`/`saveWeightsLID` with
   the `LID_` filename prefix, `setWeightsLID`/`getWeightsLID`/
   `getWeightsDerivativesLID`/`getInputStatisticsLID`/`isBackPropActivatedLID`
   - `:59-85`), the epilogue ponderation
   (`ponderateWeightsDerivatives(LIDCostPonderation)`, `_LIDCumulativeError *=
   audio._Weight`, `_CumulativeError *= LIDCostPonderation` - `:1318-1421`).
5. **`tasks/vrcts.rs::VrctsPart`** (Algo 0, `VRCTSpart.cpp` 71 LOC): faithful
   skip-if-output-XML-exists semantics; the `system()` shell-out ports as a
   real process spawn of the legacy path (`/usr/local/vrcts/...` - fails
   without the binary, IMPROVEMENTS'd); parse-back via the already-ported
   `load_vrcts`; tests exercise the skip path with pre-seeded XML.
6. **Bag/corpus LID arms live**: `Processor::Lid`/`TwinLid`; 2-element dispatch
   vecs (`[sad, lid]` order) in all five Vec-valued bag methods + `set_weights`
   consuming both; `saveWeights` criteria (algo 5 `badClassifLID+costLID`,
   algo 6 `cost+costLID`); `updateWeights` (algo 5 `costLID`, algo 6 `cost`
   then `costLID` - the `costLID = -1.0` no-speech gate finally LIVE);
   `PrintConfusionMatrix` implemented per its 4a doc-comment (>150 sentinel,
   `-200` decode, best-non-target argmax); result cols 14-16 + confusion cols
   live from driver accessors; gradCheck cost columns per algo AND network
   index (algo 5: cols 14/len-2; algo 6 net 0: 4/last, net 1: 14/len-2 -
   `CorpusProcessor.cpp:275-289`).
7. **The 4a test backlog** (IMPROVEMENTS.md:1679-1684): best-cost TIE golden;
   3+-file fold-order-divergence golden (MEASURED non-commutative); pitch-pass
   target-reuse pin (`TDCwindow > 0` + reference-loaded spectral config).

OUT: the CNN (`_LIDConvNeuralNetwork` - dead under the committed phSeq config
since the CNN block is gated on `abs(_Mode)==7 && audio.hasReadWavFile()`
(`:922-963`) and `File_Type 1` never sets `hasReadWavFile`; broken-as-committed
otherwise per the locked Phase 2 exclusion - the ctor ports as the degenerate
no-op the committed config produces, and the Mode-7-WAV branch bails with a
typed error instead of reproducing UB, IMPROVEMENTS'd), cep ingestion
(`File_Type 2`), the TRS loader (no corpus needs it), PyO3/Python (4c), the
`compute_errors` LID display block, SRN/CWRNN LID net variants (commented out
in legacy).

Exit gate: the committed `LID_BLSTM.config` (Mode 7, `File_Type 1`) runs
end-to-end through `./speech -m` on a synthetic-phSeq multi-class corpus with
the real 95k LID weights, bit-exact vs the harness oracle; a corpus-level
Algo-6 train golden chains BOTH nets' weights through real Rprop epochs.

## S2. Architecture

- **No inheritance**: legacy's Twin inherits `BLSTMSpectralSegmenter` for
  protected feature helpers; the 2b port externalized those into
  `features::pipeline` free functions - both LID drivers call them directly.
- `TwinBlstmSpectralLid { sad_net: BlstmNetwork, lid_net: BlstmNetwork, /* mode,
  post_process_mode, lid_window_size/shift, detection/training-pruning
  thresholds, decision rising/falling, min_nb_of_frames, noise_magnitude,
  dump_lid_internals */ }` + the `driver_cfg`/`seg_cfg`/`feature_cfg` trio.
  Ctor reads the `BLSTM_LID_*` key family (window, shift, DetectionThreshold
  -1.0, TrainingPruningThreshold -1.0, decision_thresh_rising/falling, Mode 0,
  PostProcessMode 0, MinNbOfFrames 0, NoiseMagnitude 0.0, DumpInternals false
  - `TwinBLSTMSpectralLID.cpp:10-39`) + both nets via the established
  `from_legacy` + `<prefix>_weightsFile` loading (Task-3 seam).
- `BlstmSpectralLid` = the spectral driver's shape + the per-segment scoring
  loop; single net, no LID-specific keys.
- **Target-index plumbing**: `Audio` gains `lang_index: i32` (-1 default) and
  `weight: f64` (1.0 default), set by the bag from the `CorpusItem` after
  `read_audio` (legacy `AudioStruct._LangIndex`/`_Weight`). `targetIndex` is
  clamped into `[0, classNb)` with `classNb = max(lid output size, 2)`
  (`:621-639`).
- **`nn/blstm.rs` additions**: `ponderate_weights_derivatives(factor)` +
  `get_cost_ponderation(class_index)` (transcribed; `getCostPonderation` reads
  `classes_ponderations` - `BLSTMNeuralNetwork.cpp:333-345`); verify
  `feed_forward_scoring` against the returning
  `feedForward(inputSeq, w, s, targetIndex, targetModifier)` overload
  (`BLSTMNeuralNetwork.h:195`) and adapt if the signature differs.
  `output_forward`/`output_backward` are already public.
- **Dispatch/gradCheck** per S1.6. `save_and_update`'s algo-5/6 confusion
  block (`BagOfProcessors.cpp:438-441`) goes live: confusion columns sliced
  from result cols `16..len-2`.
- Error handling mirrors 4a: construction failures bail; the Mode-7-wav CNN
  branch bails typed.

## S3. Oracle strategy

**Primary: SegProbe pattern, NN-swap only.** The three LID TUs are already
compiled+linked (4a Task 8 build.sh). `LidProbe` (Algo 5) + `TwinProbe`
(Algo 6) transcribe the `getSegmentation` bodies; ONLY the NN calls (the
returning `feedForward(..., targetIndex, targetModifier)` overload +
`feedForwardBackward`) swap for the harness reimpl family (extended with the
scoring-overload variant if not already present); `getTargetsLID`,
`getLIDBLSTMParam`, `getBLSTMLIDInputSequence`, `LID2Segmentation`,
`Confusion2String`, `results2segmentation` are NN-free and called REAL (the
probe feeds the reimpl's hidden outputs into the real helpers where they
consume `_OutputForward`/`_OutputBackward`).

**Secondary: real-Eigen structural probes** per config variant - SEG_STRUCT
plus a new **LID_STRUCT** record (confusion-matrix equality, argmax-language
equality, boundary max-dt); generation ABORTS on structural mismatch; value
deltas recorded as calibration.

**Corpus-level: tier-2 style only** (LID always has an NN in-chain, so there
is no tier-1 real-compiled leg): the 4a CorpusProbe extends to Algo 5/6 -
train golden chaining BOTH nets' per-epoch weights through the REAL Rprop
(fire+skip best-cost trajectory MEASURED and locked, as in 4a Task 9),
`saveAndUpdate` with live confusion/costLID (the `-1.0` gate observable
end-to-end at last), gradCheck goldens per network index, and a `./speech`
binary e2e on the committed Mode-7 config. `PrintConfusionMatrix` pins the
returned error + matrix values against the real compiled `Confusion2String`.

## S4. Fixtures & mode coverage

- **Multi-class corpus**: existing excerpt wavs RELABELED across the real
  3-class `languagemapping.csv` (`fax;non;0`, `chi;man;1`, `spa;spa;2` -
  committed as a phase4b fixture). `targetIndex` flows from the listing, not
  the audio content, so relabeling is fully non-degenerate for parity.
- **Real LID weights**: `LID_bestNNWeight_1_MultiConfigResults.mat` (95k, from
  `FastSpeechProcessing-legacy/Release/bin/`) committed as the LID weight
  fixture (a `.bin` payload with a `.mat` name, per the saveWeights quirk).
- **Synthetic phSeq files** for Mode 7, written per the reader transcription.
- **Config variants** derived from the committed `configs/legacy/
  LID_BLSTM.config` (Algo 6, Mode 7, `File_Type 1`, SAD `[11,12]/[24,1]` +
  LID `[36,24]/[48,1]` nets) with re-pointed paths.
- **Mode coverage**: flagship golden Mode 7 (real config + real weights).
  Crafted-config goldens: Mode 0 (default; plus a crafted-LID-input-width
  variant FORCING the `getBLSTMLIDInputSequence` hidden-state concat), Modes
  2/3 (reference-driven), 5, 6 (`LID2Segmentation`). Modes 1/4: transcription
  + harness unit probes if fixture crafting is disproportionate (decide by
  measurement during planning/implementation; document either way). All three
  `_PostProcessMode` values covered across the variants. EVERY golden carries
  a PROVEN non-vacuity check: the `>150` sentinel present in the results,
  confusion off-diagonals nonzero where designed, per-mode branch-taken
  counters from the probe.
- Algo 5 golden: its own config variant over the multi-class wav corpus.
- VRCTS: pre-seeded fixture XML exercises the skip+parse path; the spawn path
  is pinned by a does-it-attempt test (command line captured, not executed).

## S5. The 4a backlog (own task)

1. Best-cost TIE golden: craft an epoch pair with byte-equal costs; the `>`
   gate must SKIP (a `>=` mutation then fires - closing the battery gap).
2. Fold-order-divergence golden: 3+-file corpus with MEASURED non-commutative
   derivative sums (craft until reversal breaks bits; record the measurement).
3. Pitch-pass target-reuse pin: `TDCwindow > 0` + reference-loaded spectral
   config; pins the pass-1-target-reuse quirk (`BLSTMSpectralSegmenter.cpp:793`)
   under live targets.

## S6. Testing requirements

Suites: `phase4b_lid5_golden`, `phase4b_twin_golden` (per-mode),
`phase4b_phseq` (reader unit + golden), `phase4b_vrcts`,
`phase4b_confusion_golden`, `phase4b_corpus_lid` (train/gradCheck/binary e2e),
`phase4b_backlog` (the three S5 items), pytest fixture guards. Mutation
battery (each must break a named golden): in-band offset 200 -> 100; sentinel
150 flip; `targetLID = -2.0` sign flip; post-process-mode swap; `LID_` prefix
drop; `[sad, lid]` dispatch order swap; costLID `-1.0` gate removal (now
live); `/= 56.0` constant change. Comparators: canary-gated for everything
downstream of the NN/libm; strict for structural/integer values. Fixture
rules, hash guards, twice-regeneration, manifest-measured values: identical
to 4a.

## S7. IMPROVEMENTS.md candidates (log during implementation)

The `segmentationLID /= 56.0` magic constant (`:1292`); the `_Mode < 0`
mostly-dead branches; the double `language2classmapping` key (last-wins) in
the committed config; the LID output-size force-to->=2; the CNN-dead-under-
phSeq gate; the VRCTS hard-coded binary path; `is_lid_correct` -1-when-no-
target; the `getTargetsLID` -0.5 dont-care rows; the `LID_` filename prefix
composing onto the already-prefixed `bestNNWeight_` name; whatever
implementation surfaces.

## S8. Risks

- R1: the returning-feedForward overload vs `feed_forward_scoring` signature
  mismatch - adapter, verified in the ladder's first NN-touching task.
- R2: Mode-7 `.mat` internals dumps (`_DumpLIDInternals`) need `MatWriter`
  compatibility with per-segment variable names - writer exists (4a T1),
  variable-name scheme transcribed.
- R3: phSeq format details (frame rate/times semantics) - transcription +
  synthetic-fixture round-trip pins it; no on-disk phSeq exists, so the
  synthetic files ARE the format contract (documented in the manifest).
- R4: the Twin's 8-mode surface is large - the ladder + per-mode probes keep
  each rung reviewable; Modes 1/4 fixture-vs-probe decided by measurement.
- R5: two-net gradCheck restore semantics (bag snapshot must restore BOTH
  nets incl. Rprop state) - covered by the 4a Clone chain; pinned by the
  gradCheck golden.

## S9. Process

superpowers loop: this spec -> writing-plans
(`docs/superpowers/plans/2026-07-07-phase-4b-lid.md`) -> subagent-driven
development (fresh implementer + reviewer per task; mutation batteries re-run
by reviewers; briefs state LEGACY SOURCE GOVERNS over brief prose) -> docs
task (README roadmap 4b done-bullet + CLAUDE.md rows for lid.rs/vrcts.rs/
audio phSeq + IMPROVEMENTS) -> smart-commit (whole branch) ->
finishing-a-development-branch (user merges via PR himself; 4b PR stacks on
the 4a PR).
