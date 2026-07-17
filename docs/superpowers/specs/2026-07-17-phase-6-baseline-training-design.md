# Phase 6: baseline training on real data -- design spec

Date: 2026-07-17
Branch: `feature/phase-6-baseline-training` (base main@7e4dcb6, post-phase-5-merge; the
corpus-gitignore commit 4ba7c20 is this branch's first commit)
Approach: A ("three-track: ingestion -> metrics -> training"), user-approved.

Phase 6 of Roadmap 2: SAD + LID trained FROM SCRATCH on real data via the phase-5
machinery, with a DCF/EER/Cavg eval harness. Exit: the baseline PROTOCOL + machinery,
subset-proven -- the full-corpus numbers land when the user fires the offline launchers.

## S0. Facts the design rests on (verified 2026-07-17, do not re-derive)

- THE CORPUS: `data/LRE03-LRE07/` (27 GB, user-provided, GITIGNORED -- licensed
  LDC-derived material, NOTHING from it is ever committed): `train/{audio/{LRE03,LRE07}
  (2,066 wav + 2,068 xml refs), LID_Features/ (plp8f0mvsdd + sdc56f0 + plp8mvs* +
  cmllr variants), phSeq/ (+ listings with 2015-era absolute paths), phSeq_conf/,
  seg.xml}`; `eval/{audio/ (3,840 sph + 3,840 xml + 3 ndx), LID_Features/, phSeq/
  (per-condition 3/10/30), phSeq_conf/}`. Plus `data/Scoring_LRE15/` (315 MB, gitignored):
  the user's IS2016-era LRE15 scoring/analysis toolkit (Cavg/LER PDFs, dev/eval cuts,
  conf CSVs) -- the published-numbers comparability reference.
- THE 2015 NUMBERS DO NOT SURVIVE AS TABLES: the SAD Results_Stats PDFs are graphical
  dashboards (visual reads: Pmiss ~0-2%, Pfa ~15-22%, training-cost floor ~0.05-0.06;
  the .mat "CostMem 0.123" is a single-epoch snapshot); the LID error was only ever
  computed by `compare_scores.m`, whose inputs (`refs.mat`, the 17-Apr LIDscoreDet
  dirs) are LOST. Both baselines must be RECOMPUTED under the phase's protocol.
- THE METRICS: SAD = DCF `0.75*Pmiss + 0.25*Pfa` at 5 collar sizes (no-collar, 1/4s,
  1/2s, 1s, 2s), per `scoreFile_SAD.pl` v2.1 (NIST OpenSAD, Greg Sanders, vendored at
  `Optimizer_V6.2.2/scoreFile_SAD.pl`; CLI `-r ref -h hyp -s/-e/-g ref cols -t/-f/-u
  hyp cols`, TAB-delimited, zero-based; ref classes S/NS/NT/RI/RS/RX with RI=speech;
  v2.1 returns 0 not NaN on degenerate files; invoked by the Python-2 `ComputeDCF.py`
  driver with cols `-s 2 -e 3 -g 4 -t 5 -f 6 -u 7`). LID = 12-way argmax error
  (`compare_scores.m`: argmax across per-language detector scores, error % =
  mismatches/total; its patternnet fusion stage is OUT of the port's scope) + Cavg
  (NIST LRE convention, cross-checked against Scoring_LRE15 where verifiable).
- THE 2015 LID REGIME: `File_Type 2` -- the custom binary cep reader
  (`AudioStruct.cpp:183-256`: int32 nbRecords, int16 vectorSize, int16 magic, then
  records; 8 kHz hardcoded) over `.plp8f0mvsdd` feature files; 15,402 LRE03 (12
  languages: spa 2168, eng 2153, chi 2002, vie 1235, fre 1099, hin 1037, fas 1027,
  ger 1006, jap 994, ara 951, kor 907, tam 823) + 14,782 LRE07 (9 languages -- vie,
  fre, fas absent). 12 one-vs-rest detectors in 2015, each with a per-run
  `langmapping.csv` remapping the target to class 1. The port's cep path is currently
  a typed bail ("complete-as-portable, no data") -- the data now exists and the format
  is documented: the bail flips to a real reader this phase.
- THE EVAL LISTINGS: raw `.sph` per condition (`listing_eval_{3,10,30}s.csv`, 1200
  each) AND `_vtln` variants pointing at PRECOMPUTED FEATURE files (1189-1199 each) --
  the features-regime eval runs with ZERO audio decoding via the vtln listings; sph
  needs conversion (sph2pipe) for the wav-eval arm + nothing in the port reads SPHERE
  (symphonia lacks it; the legacy used libsndfile's built-in NIST support).
- LISTING SCHEMA (6 fields, `;`): `path;ref-xml;lang;class/cond;weight;duration`.
  Durations: eval fixed 3/10/30 s; training variable (28-196 s observed).
- USER DECISIONS (this brainstorm): corpus at `data/LRE03-LRE07` (gitignored, done);
  BOTH LID arms, acoustic first; the acoustic arm = the FEATURES regime (File_Type 2,
  the 2015-comparable path) with the wav+own-front-end regime deferred; subset CI
  gates + full runs as user-fired offline launchers.
- PHASE-5 CARRY-FORWARDS absorbed here: NNCostSeg as the validation signal (the
  balance-5 signal is a stuck step function -- train_modern is certified piecewise
  until this lands); the cost-law correctness pass (AboveThreshCubic + the
  softmax-ponderation derivative items -- live in SAD training's laws). Deferred and
  documented unless tripped over: per-candidate net re-init (hyperparam-search
  invalid regions stay penalized+observable), the legacy-regime `train()` retirement
  decision, the single-layer init path.

## S1. Deliverables

### Track 1 -- data + ingestion

- **S1.1 The cep reader (File_Type 2 un-bail).** `audio.rs` gains the reader per
  `AudioStruct.cpp:183-256` (LEGACY SOURCE GOVERNS the record layout -- the scout's
  int32/int16/int16 header summary is the prior, the source is the truth), filling
  `external_features` the way the phSeq path does, 8 kHz hardcoded. The
  `bag_of_processors` File_Type 2+ pre-gate opens for type 2 (types 3/4 stay bailed --
  no data, out of scope, documented). Pinned: committed SYNTHETIC cep fixtures crafted
  to the format (CI; incl. header-edge cases: zero records, vectorSize mismatches ->
  typed errors); corpus-gated real-file cross-checks (vectorSize x nbRecords
  consistency vs the listing's duration field at the known frame rate); the
  complete-as-portable IMPROVEMENTS entry flips per the phase-5 protocol (a RED is
  impossible for a bail -> pin the NEW reader's behavior; the old bail's pinning test
  re-pins to the opened gate).
- **S1.2 Listing localization.** A dataprep tool (`dataprep/lre.py` or similar; plan
  decides) rewriting the 2015 listings' absolute `/Users/Govit/...` paths against the
  local corpus root, verifying per-row file EXISTENCE (missing files reported, not
  silently dropped -- the 2015 listings may reference files the archive lacks), and
  emitting the phase's canonical train/eval listings + the 12-language
  `language2classmapping` (the single 12-class mapping -- see S1.7). Corpus-gated
  tests; the tool itself is pure-logic-testable with synthetic trees.
- **S1.3 SAD listings.** Derived from `train/audio`'s wav+xml pairs (fileslisting +
  refsegfiles form; the xml refs load via the already-ported VRCTS loader -- verify
  one real file corpus-gated). A seeded held-out split (train/valid/test) with the
  split recorded in the emitted listings.
- **S1.4 sph2pipe conversion.** `dataprep/` gains the sph->wav step behind an
  injectable runner (the sox pattern: command strings pinned, execution injected;
  sph2pipe presence documented as a local prerequisite -- brew/source, never vendored).
  Feeds the wav-eval arm when wanted; NOT on the features-regime critical path.

### Track 2 -- the eval harness

- **S1.5 `speech/evaluate.py`.** `dcf(ref_intervals, hyp_intervals, collars) ->
  DcfReport` porting `scoreFile_SAD.pl`'s interval logic (the 6 ref classes, the
  5-collar table, v2.1's degenerate-file zero semantics) -- VALUE-PINNED against the
  REAL vendored perl scorer via `/usr/bin/perl` (crafted tab fixtures,
  extractor-committed goldens -- the 4d STM-normalizer pattern; `tools/perl_oracle/`
  grows a second runner). `lid_error(scores, refs) -> float` (argmax semantics per
  `compare_scores.m`, fusion excluded). `cavg(...)` per the NIST LRE convention --
  implemented against the formula, cross-checked against `data/Scoring_LRE15`'s
  artifacts where a verifiable input/output pair exists (adjudicated at
  implementation; if the toolkit yields no runnable pair, the formula + crafted-case
  tests stand alone and the limitation is documented). The harness consumes ENGINE
  OUTPUTS: VRCTS xml (SAD hyps), `.scr` files (LID scores).

### Track 3 -- training

- **S1.6 The validation signal (carry-forward).** `train_modern`'s validate hook gains
  the metric choice (`val_metric: "nn_cost_seg" | "balance"`), defaulting to
  NNCostSeg -- the F10-fixed forward cost, a continuous signal. RED: the phase-5
  stuck-at-30.0 demonstration (the plateau test re-pins to exercise BOTH metrics).
  The piecewise-certification note in the docs flips to end-to-end once the subset
  gates train through the real signal.
- **S1.7 The LID features arm.** A single 12-CLASS Twin run (the port-native
  multiclass head + the FIXED F5 confusion machinery), NOT 2015's 12 one-vs-rest
  detectors (argmax error stays directly comparable; the one-vs-rest replication is
  documented optional legacy-comparability work). Configs: File_Type 2 + the
  LRE-derived TOML/legacy config pair seeded from the 2015 best-configs' DSP keys
  (verified against `ConfigFiles/`/the run dirs at implementation -- incl. the
  vtln-vs-plain train/eval pairing question, adjudicated there and documented).
  Deliverables: the stratified-subset CI gate (corpus-gated: bounded epochs on a
  seeded ~stratified subset; from-scratch improvement per the phase-5 gate pattern +
  the harness scores the trained model end to end + determinism) and the FULL-RUN
  LAUNCHER (`speech.cli train-baseline lid-features ...` or a script -- plan decides:
  config, resume-from-checkpoint, N-lane engine parallelism, progress/ETA logging, a
  1-step dry-run smoke mode the gate exercises).
- **S1.8 The SAD arm.** Same shape on the S1.3 listings: subset gate + launcher; the
  harness scores DCF on the held-out split.
- **S1.9 The phonotactic arm (second).** Mode-7 on the localized phSeqbis listings
  (File_Type 1 -- fully ported, bit-exact lineage), same subset-gate + launcher
  shape. NOTE the Mode-7 SAD net is a frozen feature extractor (phase-5 finding) --
  the arm trains the LID net only, stated plainly in its gate.
- **S1.10 The cost-law pass (carry-forward).** The scoped-out derivative-consistency
  items (AboveThreshCubic and the softmax-ponderation family) adjudicated and fixed
  where wrong-gradient-on-live-paths, per the phase-5 fix protocol (RED -> re-pin ->
  mutation -> flip), sized by a mini-sweep of the fixlist's "CONSIDERED, SCOPED OUT"
  entries against the SAD/LID configs this phase actually trains.
- **S1.11 `RESULTS.md`** (committed): the baseline protocol (splits, conditions,
  metrics, seeds, configs) + a results table template the full runs fill in; the
  IS2016/Scoring_LRE15 published numbers cited as the comparability reference.

## S2. Oracle/verification tiers

| Artifact | Oracle | Tier |
|---|---|---|
| dcf() | the REAL vendored scoreFile_SAD.pl via /usr/bin/perl | LIVE value-oracle |
| cep reader | AudioStruct.cpp source + synthetic fixtures + corpus cross-checks | transcription + corpus-gated consistency |
| lid_error/cavg | compare_scores.m semantics + the NIST formula + Scoring_LRE15 cross-check where possible | transcription (+ published-number sanity) |
| subset training gates | the phase-5 improvement-gate pattern (margins measured-then-pinned) | improvement-shaped, corpus-gated |
| launchers | 1-step dry-run smokes | mechanical |
| listings/localization | synthetic-tree unit tests + corpus-gated existence audits | mixed |

CI discipline: everything committed is synthetic/crafted; every real-data test is
corpus-gated (skipif `data/LRE03-LRE07` absent) and runs locally only. The phase-5 fix
protocol governs every behavior change (S1.1's bail flip, S1.6's signal, S1.10's laws).

## S3. Risks

- R1 cep record-layout surprises (dtype/endianness/the magic's meaning): the source
  governs; real-file duration cross-checks catch a wrong stride immediately;
  corpus-gated.
- R2 the vtln-vs-plain pairing (train on plain, eval on vtln-warped features may be
  the 2015 intent or a mismatch): adjudicated against the 2015 best-configs at
  implementation; whichever pairing 2015 used is the baseline protocol; documented.
- R3 subset-gate runtime on the loaded box: measured first, bounded honestly
  (target: each corpus-gated gate < ~10 min local); the full runs are launcher-side
  by design.
- R4 corpus-license hygiene: gitignore (done) + the synthetic-fixtures-only rule +
  review attention on every commit touching tests/fixtures.
- R5 Cavg verifiability against Scoring_LRE15: adjudicate-at-implementation with an
  honest fallback (formula + crafted cases only, limitation documented).
- R6 the 12-class-vs-one-vs-rest comparability: argmax error is regime-agnostic;
  Cavg conventions may differ (documented per the S1.7 note); the optional
  replication path recorded.
- R7 missing corpus files vs the 2015 listings (the archive may be partial): S1.2
  reports per-row existence; the canonical listings are the EXISTING intersection,
  counts recorded in RESULTS.md.

## S4. Determinism / deviations

Seeded splits + seeded subset selection (recorded in the emitted listings); the
launchers checkpoint/resume (the phase-5 resume-equivalence machinery); N-lane
parallelism is deterministic-but-N-dependent (the R6-4a model) -- the LAUNCHER RECORDS
N in the run metadata and RESULTS.md, and the baseline protocol fixes N per run.
Python-2 ComputeDCF.py is NOT ported (a driver, not logic -- the harness replaces it);
the perl scorer is the oracle, not a runtime dependency.

## S5. Mutation battery (plan finalizes; candidates)

dcf collar arithmetic off-by-one -> the perl-oracle goldens; the RI-is-speech class
flip -> same; lid_error argmax tie-handling -> the crafted cases; the cep header
stride misread -> the synthetic fixtures; the localization existence-check dropped ->
the missing-file report test; the NNCostSeg signal wired back to balance -> the
re-pinned plateau test; the subset gate's margin zeroed -> the stagnant-trainer
demonstration; Cavg prior/cost constants perturbed -> the formula cases.

## S6. Documentation

Per-task IMPROVEMENTS entries (the S1.1 flip; S1.6/S1.10 protocol fixes; the S1.7
regime note); README/CLAUDE.md rows for evaluate.py + the dataprep additions + the
cep reader; RESULTS.md as the living baseline record; the Roadmap 2 section updated
at phase end.

## S7. Finish

The per-phase loop unchanged: subagent-driven (fresh implementer + reviewer per task,
opus for the cep/harness/training-gate tasks), corpus-gated tests exercised locally by
implementers AND reviewers, mutation battery, docs refresh, final whole-branch review
(most capable model), smart-commit (whole branch), finishing-a-development-branch
(user merges via PR; never push). The full baseline runs are explicitly POST-PHASE
user-fired jobs; RESULTS.md receives their numbers whenever they complete.
