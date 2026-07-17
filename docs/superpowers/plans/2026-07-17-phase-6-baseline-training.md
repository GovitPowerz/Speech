# Phase 6: baseline training on real data -- implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The from-scratch SAD + LID baseline machinery on the real LRE03/07 corpus -- ingestion (the cep un-bail), a perl-oracle-pinned DCF/EER/Cavg harness, and three training arms as subset CI gates + user-fired full-run launchers.

**Architecture:** Spec `docs/superpowers/specs/2026-07-17-phase-6-baseline-training-design.md` (approach A, three tracks: ingestion -> metrics -> training). The corpus lives at `data/LRE03-LRE07/` (27 GB, GITIGNORED, licensed) with `data/Scoring_LRE15/` as the published-numbers reference; every real-data test is CORPUS-GATED (skipif absent -- local-only, CI keeps synthetic fixtures). The phase-5 fix protocol governs every behavior change.

**Tech Stack:** Rust edition 2024 (the cep reader), Python >=3.14 (evaluate.py, dataprep, drivers), /usr/bin/perl (the vendored scoreFile_SAD.pl as a live value-oracle), sph2pipe (a documented local prerequisite, injectable-runner pattern).

## Global Constraints

- LICENSE HYGIENE: NOTHING from `data/LRE03-LRE07/` or `data/Scoring_LRE15/` is ever committed -- committed fixtures are synthetic/crafted only; reviewers verify this on every fixture-touching commit.
- CORPUS-GATING: every real-data test uses a single shared helper/skipif keyed on the corpus root's existence (establish it in Task 1, reuse everywhere); such tests run locally (implementers AND reviewers exercise them) and skip in CI.
- THE FIX PROTOCOL (IMPROVEMENTS.md preamble) governs S1.1's bail flip, S1.6's signal change, and S1.10's law fixes: RED (or pin-new for a bail) -> re-pin -> revert-mutation -> the entry flip with the literal hash.
- LEGACY SOURCE GOVERNS: the cep record layout comes from `AudioStruct.cpp:183-256`, the DCF semantics from the vendored `scoreFile_SAD.pl`, the argmax error from `compare_scores.m` -- read them before writing a line; never modify legacy trees.
- Branch `feature/phase-6-baseline-training`; per-task commits with trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; never push; never gh.
- FOREGROUND everything -- NEVER run_in_background or Monitor; never read exit codes through pipes (log-capture); generous timeouts (the box is loaded; full cargo ~15 min).
- Gates per task: `uv run pytest tests` + `./lint_code.sh` (+ `uv run ruff check --config=pyproject.toml scripts` when touching scripts/); `cd src/rust && cargo test` + `cargo fmt --check` + `cargo clippy --workspace --all-targets -- -D warnings` when Rust changes; corpus-gated tests run explicitly and their results reported.
- Subset-gate runtime target: each corpus-gated training gate < ~10 min local (spec R3) -- measure first, bound honestly.
- ASCII only.

---

### Task 1: the cep reader (File_Type 2 un-bail)

**Files:** Modify `src/rust/src/audio.rs` (the `read_cep` path per `AudioStruct.cpp:183-256`; the `read_audio` file_type-2 bail replaced), `src/rust/src/engine/bag_of_processors.rs` (the File_Type pre-gate opens for 2; 3/4 stay bailed, documented); Create committed synthetic fixtures `tests/reference_data/phase6/cep/{tiny_ok.plp,zero_records.plp,truncated.plp,...}` + a fixture-writer helper in the test; Create the shared corpus-gate helper (`src/rust/tests/common/` for Rust, `tests/conftest.py` addition for Python: `CORPUS_ROOT = repo/data/LRE03-LRE07`, skip-if-absent); Test `src/rust/tests/phase6_cep.rs` + a corpus-gated real-file consistency test.
**Interfaces:** Produces cep ingestion: `File_Type 2` configs load `.plp8f0mvsdd`-family files into `external_features` (the phSeq pattern -- one row per record, vectorSize columns), 8 kHz hardcoded per the source. THE RECORD LAYOUT COMES FROM THE SOURCE (int32 nbRecords, int16 vectorSize, int16 magic, then records -- the scout's header summary is the prior; endianness, the record dtype, and the magic's semantics are read off `AudioStruct.cpp:183-256`, and any surprise is documented). Also produces the corpus-gate helpers every later task reuses.
- [ ] Pin-new (a bail has no RED): the OLD bail's pinning test re-pins to the opened gate; synthetic-fixture tests (a hand-built tiny cep -> exact expected feature matrix; header-edge cases -> typed errors: zero records, truncation, vectorSize/byte-length mismatch); the corpus-gated consistency test (a sampled real file: nbRecords x frame-rate ~ the listing's duration field; vectorSize matches the config's NNetInputSize expectations) -> implement -> GREEN -> the IMPROVEMENTS complete-as-portable entry flips per protocol -> full Rust + Python gates -> commit `feat(phase6): the File_Type 2 cep reader - the bail becomes a real ingestion path`.

### Task 2: listing localization + SAD listings + the 12-class mapping

**Files:** Create `src/python/speech/dataprep/lre.py`; Test `tests/test_phase6_lre_listings.py` (synthetic-tree units + corpus-gated audits).
**Interfaces:** Produces `localize_listing(src: Path, corpus_root: Path, out: Path) -> LocalizeReport` (rewrites the 2015 absolute paths against the local root; per-row file-EXISTENCE verification -- missing rows land in the report AND an emitted `<out>.missing` sidecar, never silently dropped, spec R7; the canonical output = the existing intersection), `derive_sad_listings(corpus_root, out_dir, seed) -> SadSplit` (from `train/audio`'s wav+xml pairs: fileslisting + refsegfiles form, a SEEDED train/valid/test split recorded in the emitted listings; the xml refs verified loadable via the ported VRCTS loader on a corpus-gated sample), `write_lre_mapping(out: Path)` (the single 12-class `language2classmapping` -- the 12 LRE03 languages, alphabetical class order per the ported convention). Reuses `batching.read_listing`/`write_listing` conventions.
- [ ] RED (synthetic trees: localization rewrites + missing-row reporting; the split's seeding + disjointness; the mapping's 12 rows) -> implement -> GREEN -> corpus-gated: localize the real `listing_training_comb_plp8f0mvsdd_LRE03.csv` (+LRE07) and REPORT the existence counts (how much of the 2015 corpus the archive actually holds -- the numbers go in the report + later RESULTS.md) -> gates -> commit `feat(phase6): LRE listing localization + SAD listings + the 12-class mapping`.

### Task 3: sph2pipe conversion

**Files:** Modify `src/python/speech/dataprep/lre.py` (add the conversion step); Test extend `tests/test_phase6_lre_listings.py`.
**Interfaces:** Produces `convert_sph_listing(listing: Path, out_dir: Path, runner: SphRunner) -> Path` -- the sox/augment injectable-runner pattern (`SphRunner.run(cmd: list[str])`; the pinned command surface: `sph2pipe -f wav <in> <out>`); emits a rewritten listing pointing at the wavs. sph2pipe presence = a documented local prerequisite (README note in Task 12); the recorded-runner tests pin the command vectors, a corpus-gated test converts ONE real sph and reads it back via the port's wav reader (if sph2pipe is installed locally; skip-with-reason otherwise).
- [ ] RED (recorder tests) -> implement -> GREEN -> gates -> commit `feat(phase6): sph-to-wav conversion behind an injectable runner`.

### Task 4: the DCF harness + the perl oracle

**Files:** Create `src/python/speech/evaluate.py` (the `dcf` half); extend `tools/perl_oracle/` (a `run_sad_scorer.sh` runner driving the vendored `Optimizer_V6.2.2/scoreFile_SAD.pl` via /usr/bin/perl); extend `scripts/` with a phase-6 extractor stage (crafted ref/hyp tab fixtures -> committed goldens under `tests/reference_data/phase6/dcf/`; twice-run determinism; SKIP without perl/legacy); Test `tests/test_phase6_evaluate.py`.
**Interfaces:** Produces `dcf(ref: list[Interval], hyp: list[Interval], collars: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0)) -> DcfReport` (per-collar Pmiss/Pfa/DCF with DCF = 0.75*Pmiss + 0.25*Pfa; `Interval = (start, end, kind)` with ref kinds S/NS/NT/RI/RS/RX -- RI counts as speech, NS/NT/RX/RS as non-speech; hyp kinds speech/non-speech; v2.1's degenerate-file ZERO-not-NaN semantics) + `load_vrcts_hyp(xml_path) -> list[Interval]` and `load_tab_ref(tab_path, cols) -> list[Interval]` adapters (the engine's VRCTS xml is the hyp source; the NIST tab format the ref source, the ComputeDCF.py column convention -s 2 -e 3 -g 4). VALUE-PINNED against the real perl scorer on crafted fixtures covering: plain overlaps, every ref class, collar boundary hits, the degenerate all-speech/all-nonspeech files, sub-collar segments.
- [ ] Extractor perl stage (local) -> committed goldens -> RED -> implement -> GREEN value-exact per collar -> gates -> commit `feat(phase6): the DCF scorer pinned against the vendored NIST perl scorer`.

### Task 5: lid_error + cavg

**Files:** Modify `src/python/speech/evaluate.py`; Test extend `tests/test_phase6_evaluate.py`.
**Interfaces:** Produces `lid_error(scores: NDArray (n_files, n_langs), refs: NDArray (n_files,)) -> float` (argmax semantics per `compare_scores.m` -- error % = mismatches/total; ties resolved per numpy argmax-first, documented -- the .m's max() takes the first maximum too, verify at source; the patternnet fusion stage EXCLUDED, documented) + `read_scr_scores(scores_dir, class_keys) -> (scores, files)` (the adapter over the port's .scr outputs -- mind the alphabetical class order + the softmax the writer applied; decide and DOCUMENT whether lid_error consumes the softmaxed .scr values or re-reads raw scores -- argmax is monotone-invariant under softmax, state it) + `cavg(scores, refs, ...) -> float` per the NIST LRE convention (the standard pair-wise Cavg formula; constants documented). Scoring_LRE15 cross-check: ADJUDICATE at implementation -- inspect `data/Scoring_LRE15/` (corpus-gated) for a runnable input/output pair (e.g. dev_conf.csv + a published Cavg value reproducible from it); if one exists, add the corpus-gated cross-check; if not, the formula + crafted-case tests stand and the limitation is documented (spec R5) -- state the adjudication in the report.
- [ ] RED (crafted cases: known confusion -> hand-computed error; argmax ties; a tiny synthetic Cavg case hand-derived from the formula) -> implement -> GREEN -> gates -> commit `feat(phase6): lid_error + cavg per the legacy and NIST conventions`.

### Task 6: the NNCostSeg validation signal (carry-forward)

**Files:** Modify `src/python/speech/drivers/train.py` (`_make_default_validate` gains the metric choice), `src/python/speech/drivers/state.py` (`ModernTrainParams.val_metric: Literal["nn_cost_seg","balance"] = "nn_cost_seg"`); Test extend `tests/test_phase5_train_modern.py` + the pyo3 smoke.
**Interfaces:** Produces the continuous validation signal: `val_metric="nn_cost_seg"` computes the F10-fixed forward NNCostSeg on the valid listing (forward-only, `_nn_cost_seg`-family assembly per engine.py) instead of the stuck balance cost; `"balance"` preserves the phase-5 behavior. Per the fix protocol: RED = the phase-5 stuck-at-30.0 plateau demonstration (the early-stop gate's test re-pins to exercise BOTH metrics -- the balance path still plateaus, the nn_cost_seg path yields a MOVING signal on the same fixture); the piecewise-certification docs flip in Task 12 once the arms train through the real signal.
- [ ] RED -> implement -> GREEN -> the pyo3 smoke re-pinned (both metrics) -> gates -> commit `fix(phase6): NNCostSeg as the default validation signal (the stuck-balance carry-forward)`.

### Task 7: the cost-law correctness pass (carry-forward)

**Files:** The mini-sweep first: read the fixlist's "CONSIDERED, SCOPED OUT" entries (docs/superpowers/plans/2026-07-10-phase5-fixlist.md) + the cost.rs laws the phase-6 configs actually train (the SAD configs' CostLaw/AboveThresh laws; the LID balance-10 path) -- emit the adjudicated fix list into the report + the ledger. Then per adjudicated fix: Modify `src/rust/src/cost.rs`; re-pin the affected phase3/phase0b golden families; Test additions.
**Interfaces:** Produces derivative-consistent cost laws on every live training path (the F7 pattern: the deriv matches the clamped/branched forward everywhere; expected members: the AboveThreshCubic derivative arm, the softmax-ponderation family -- THE SWEEP GOVERNS; a law with no live phase-6 path stays documented-not-fixed).
- [ ] Sweep -> per-fix protocol (RED via off-grid directed tests where the goldens mask, the F7 lesson) -> re-pin -> mutations -> flips -> full Rust gates -> commit `fix(phase6): cost-law derivatives consistent on the live training paths`.

### Task 8: the LID features arm (+ RESULTS.md)

**Files:** Create `configs/training/lre03_lid_features.toml` (+ the flat-config seam pieces the driver synthesizes -- seeded from the 2015 best-configs' DSP keys, VERIFIED against `ConfigFiles/`/the 03-Jun run dirs at implementation; the vtln-vs-plain train/eval pairing adjudicated HERE against the 2015 best-configs and documented -- spec R2), `src/python/speech/drivers/baseline.py` (the launcher: `run_baseline(arm, corpus_root, out_dir, resume, lanes, subset)` + a CLI entry `speech.cli baseline lid-features ...` -- config assembly, the localized listings from Task 2, from-scratch init or resume, train_modern with val_metric=nn_cost_seg, progress/ETA logging, run-metadata (seed, N lanes, subset spec) recorded, a `--dry-run` 1-step smoke), `RESULTS.md` (the baseline protocol: splits, conditions, metrics, seeds, configs, the N-lane determinism note; the results table template; the IS2016/Scoring_LRE15 citation); Test `tests/test_phase6_baseline_arms.py` (launcher units against stubs) + the corpus-gated subset gate `tests/pyo3/test_phase6_gates.py::test_lid_features_subset_trains_and_scores`.
**Interfaces:** Consumes Tasks 1/2/5/6 (+7 transitively). Produces the arm: a stratified seeded subset (per-language proportional, size chosen to fit the <10 min budget -- measure) trains from scratch (12-class Twin, File_Type 2), improvement asserted per the phase-5 gate pattern (margins measured-then-pinned), then `evaluate` scores the trained model on a held-out subset slice via `.scr` -> `lid_error` (+cavg recorded) -- the END-TO-END proof; run-twice determinism; the dry-run smoke exercised in the same gate file (corpus-gated).
- [ ] RED (launcher units; the gate against a stub then the real path) -> implement -> GREEN locally (the corpus-gated gate's measured runtime + margins in the report) -> gates -> commit `feat(phase6): the LID features arm - subset gate + the full-run launcher + RESULTS.md`.

### Task 9: the SAD arm

**Files:** Create `configs/training/lre_sad.toml` (+ seam pieces; algo-3 spectral on the train wavs); Modify `src/python/speech/drivers/baseline.py` (the `sad` arm), `RESULTS.md` (the SAD section); Test extend the unit + gate files (`test_sad_subset_trains_and_scores`).
**Interfaces:** Consumes Tasks 2 (the SAD listings + split)/4 (dcf)/6. Produces: the subset gate -- from-scratch algo-3 training on the seeded subset, improvement + the harness scoring DCF on the held-out split (the engine's VRCTS hyps vs the xml-derived refs via the Task-4 adapters) + determinism + the dry-run smoke; the launcher arm.
- [ ] RED -> implement -> GREEN locally (runtime + margins + the first real DCF numbers on the subset -- recorded) -> gates -> commit `feat(phase6): the SAD arm - subset gate + launcher`.

### Task 10: the phonotactic arm

**Files:** Create `configs/training/lre03_lid_phseq.toml` (Mode-7, File_Type 1, seeded from the committed twin_mode7 lineage + the 2015 phSeq configs); Modify `baseline.py` (the `lid-phseq` arm), `RESULTS.md`; Test extend (`test_lid_phseq_subset_trains_and_scores`).
**Interfaces:** Consumes Tasks 2 (localized phSeq listings)/5/6. Produces the same gate + launcher shape on the phSeqbis data. THE ARM TRAINS THE LID NET ONLY (the Mode-7 SAD net is a frozen feature extractor -- the phase-5 finding, stated in the gate's docstring); improvement + lid_error scoring + determinism.
- [ ] RED -> implement -> GREEN locally -> gates -> commit `feat(phase6): the phonotactic Mode-7 arm - subset gate + launcher`.

### Task 11: mutation battery

**Files:** `IMPROVEMENTS.md` only (apply-fail-revert-pass; FOREGROUND; tree clean between/after; corpus-gated catchers run locally).
- [ ] The S5 list: dcf collar arithmetic off-by-one -> the perl goldens; the RI-is-speech flip -> same; lid_error argmax tie-handling -> the crafted cases; the cep header stride misread -> the synthetic fixtures; the localization existence-check dropped -> the missing-row report test; the NNCostSeg signal wired back to balance-only -> the re-pinned plateau test; a subset gate's margin zeroed -> the stagnant-trainer demonstration (the item-7-4d pattern); a Cavg constant perturbed -> the formula cases. One line each in a "Mutation battery (Phase 6)" subsection; honest gaps recorded. Full pytest + cargo once at the end. Commit `test(phase6): mutation battery results recorded`.

### Task 12: docs refresh

**Files:** `README.md` (the Phase 6 bullet + Roadmap 2 update; the sph2pipe local-prerequisite note; the corpus-layout note), `CLAUDE.md` (evaluate.py + dataprep/lre.py + baseline.py rows; the cep reader's audio.rs row update -- the complete-as-portable claim flips; the corpus-gating convention in the testing narration; the piecewise-certification note flips to end-to-end), `IMPROVEMENTS.md` consistency (hashes literal, the S1.1/S1.6/S1.10 flips verified), `RESULTS.md` final consistency.
- [ ] Raw material: the phase-6 ledger + landed code (every claim code-verified). Full gates. Commit `docs(phase6): README/CLAUDE.md/IMPROVEMENTS/RESULTS refresh - the baseline machinery lands`.

### Task 13: final review + smart-commit + finish

- [ ] Final whole-branch review (most capable model; package `main@7e4dcb6..HEAD`; the ledger Minor triage; special attention: license hygiene across every committed fixture, the corpus-gating uniformity, the perl-oracle goldens' crafted-not-copied provenance, the subset gates' margin honesty).
- [ ] Invoke the `smart-commit` skill taking the WHOLE git branch into account, then `superpowers:finishing-a-development-branch` (user merges via PR; never push). The full baseline runs remain post-phase user-fired jobs; RESULTS.md receives their numbers whenever they complete.

## Self-Review

1. **Spec coverage**: S1.1->T1; S1.2->T2; S1.3->T2; S1.4->T3; S1.5->T4+T5; S1.6->T6; S1.7->T8; S1.8->T9; S1.9->T10; S1.10->T7; S1.11->T8 (created) + T9/T10 (extended) + T12 (finalized); S2 tiers per task; S3: R1->T1's source-governs + cross-checks, R2->T8's pairing adjudication, R3->the runtime budget constraint + per-gate measurement, R4->the license-hygiene constraint + T13's sweep, R5->T5's adjudication, R6->T8's regime note, R7->T2's existence reporting; S4->T2 (seeded splits), T8 (run metadata/N), the ComputeDCF-not-ported note lives in T4's module doc; S5->T11; S6->T12; S7->T13.
2. **Placeholders**: none -- the four adjudicate-at-implementation points (the cep record layout, the vtln pairing, the Cavg cross-check, the .scr-vs-raw scores question) each carry explicit decision rules and a documentation obligation.
3. **Type consistency**: the corpus-gate helpers (T1) reused by T2/T3/T5/T8/T9/T10; `localize_listing`/`derive_sad_listings`/`write_lre_mapping` (T2) consumed by T8/T9/T10; `SphRunner`/`convert_sph_listing` (T3) self-contained; `dcf`/`Interval`/`load_vrcts_hyp`/`load_tab_ref` (T4) consumed by T9; `lid_error`/`read_scr_scores`/`cavg` (T5) consumed by T8/T10; `val_metric` (T6) consumed by T8/T9/T10; `run_baseline`/the CLI arm names (T8) extended by T9/T10.
