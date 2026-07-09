# Phase 4d: dataprep + end-to-end parity -- implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last stub area (`dataprep/`), execute both deferred parity milestones against the resurrected 2015 production binary, close the Phase-0a packer/adim golden, port the feasible backlog four, and declare the roadmap complete.

**Architecture:** Spec `docs/superpowers/specs/2026-07-09-phase-4d-dataprep-endtoend-design.md` (approach A, fresh-oracle-first). Two new local-only oracle surfaces join the harness family: `/usr/bin/perl` executing the real `.pl` normalizers, and `tools/fsp_runtime/` running the real 2015 `-ffast-math` Mach-O binary under Rosetta. CI consumes committed `tests/reference_data/phase4d/` fixtures only, never any oracle.

**Tech Stack:** Python >=3.14 (uv, pydantic, numpy, soundfile), Rust edition 2024, Octave 11.3 harness (local), perl 5.34 (local), Rosetta 2 + `install_name_tool` (local).

## Global Constraints

- LEGACY SOURCE GOVERNS over spec/plan prose; every reproduced quirk gets an IMPROVEMENTS.md entry with a real Pinned-by citation. NEVER modify `legacy/` or `/Users/govit/Git/Govit/FastSpeechProcessing-legacy` (the binary is COPIED out, never touched in place).
- Branch `feature/phase-4d-dataprep-endtoend`; per-task commits with trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; never push; never gh.
- FOREGROUND test runs only. Gates per task: `uv run pytest tests` + `./lint_code.sh`; `cd src/rust && cargo test` when Rust changes.
- ASCII only. Cross-libm rule: strict pins only on pure-arithmetic/table-driven values; libm-bearing values through `assert_f64_close`/canary comparators; parity-gate bounds are MEASURED-THEN-PINNED manifest values, never invented.
- Committed phase4d fixtures total < 8 MB (extractor asserts the budget; the PRCTS excerpt + 2 weight packs dominate).

---

### Task 1: Phase-0a closure -- the packer/adim byte golden against the real weight packs

**Files:** Create `tests/reference_data/phase4d/{tupleA_NNweights_config1.bin,tupleA_1_worker_1.config,tupleB_NNweights_config1.bin,tupleB_1_worker_1.config}` (copied verbatim from `Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/` and `Optimizer_V6.2.1/Executables/14-Oct-2015_BLSTM_OpenSAD15/` in the legacy copy -- record source paths + SHA-256 in a `phase4d_sources.json` manifest); Test `src/rust/tests/phase4d_packer_golden.rs`, `tests/test_phase4d_packer.py`; Modify `CLAUDE.md` locked-decisions row ONLY if the wording needs the closure note now (else defer to T15).
**Interfaces:** Consumes `config.rs::NnetSpec`/`nnet_to_flat`/`flat_to_nnet`, `io/binary.rs::read_bin/write_bin`, `weight_bridge.pack_weights/unpack_weights`. Produces the committed real-pack fixtures ALSO consumed by Tasks 3/4 (the parity runs load `tupleA_NNweights_config1.bin`).
- [ ] Failing tests: Rust -- parse each tuple's real `.config` (legacy parser) -> `NnetSpec` -> `read_bin` the real pack -> element count matches spec (tuple A should be the familiar 33,671-shape family; MEASURE tuple B) -> `flat_to_nnet` -> `nnet_to_flat` -> repack -> BYTE-IDENTICAL to the file; adim invariants hold. Python -- same round trip through `weight_bridge` (Rust-pack == Python-pack on the real packs, extending the 0a cross-language check from synthetic to real). -> implement (fixture copy + manifest; no production-code change expected -- if a real pack EXPOSES a packer bug, that is a finding to fix with its own test) -> GREEN -> gates -> commit `test(phase4d): packer/adim byte golden closed against the real 2015 weight packs`.

### Task 2: tools/fsp_runtime/ -- resurrect the 2015 binary (BOUNDED; fallback decision recorded)

**Files:** Create `tools/fsp_runtime/{setup.sh,README.md,.gitignore}` (gitignore `bin/`, `prefix/`, `work/`); the script copies `Segment` from the legacy Executables dir into `tools/fsp_runtime/bin/`, builds x86_64 deps into `prefix/` (libogg, libvorbis, FLAC, libsndfile, libpng, matio -- source builds under `arch -x86_64`, matio at a 1.3.x-era tag first, newer tag + `libmatio.2.dylib` symlink only if the old tag will not build and the symbol check passes), extracts `libstdc++.6.dylib`/`libgomp.1.dylib`/`libgcc_s.1.dylib` from a Homebrew x86_64 gcc BOTTLE tarball (no brew install), rewrites all 9 install names via `install_name_tool -change`, re-signs `codesign -f -s -`, and self-tests.
**Interfaces:** Produces a runnable `tools/fsp_runtime/bin/Segment` (local-only) + `setup.sh --check` exit-0 gate consumed by Task 3's extractor. SUCCESS CRITERION: (a) `arch -x86_64 bin/Segment` with argc<3 prints the usage text; (b) a solo `-s` run on a local wav under a localized config COMPLETES writing parseable outputs. FALLBACK (if blocked after a genuine bounded attempt -- document each failed avenue): add a `-ffast-math` build mode to `tools/oracle_harness/build.sh` (`-O3 -ffast-math -msse2 -mfpmath=sse` where the arch allows; on arm64 drop the x86-only flags and record that) and expose the same run surface; `phase4d_sources.json` gains an `oracle: real-binary | fastmath-rebuild` field every downstream fixture carries.
- [ ] Implement `setup.sh` incrementally (deps -> dylib rewiring -> usage smoke -> solo run on the committed phase-1 fixture wav or the PRCTS wav) -> record the outcome + `otool -L` of the final binary in `tools/fsp_runtime/README.md` -> commit `feat(phase4d): fsp_runtime resurrects the 2015 Segment binary under Rosetta` (or the fallback variant + IMPROVEMENTS deviation entry).

### Task 3: parity extractor -- committed inputs + oracle outputs

**Files:** Create `scripts/extract_phase4d_fixtures.py` (the established extractor pattern: tempdir, validate, size-budget assert, twice-run determinism where the oracle is deterministic, SHA manifest); committed inputs `tests/reference_data/phase4d/{prcts_excerpt.wav,parity_tupleA.config,parity_tupleB.config}`; committed oracle outputs `parity_tupleA_{vrcts_chan1.xml,vrcts_chan2.xml,mcr.bin,result_rows.bin}` (+ tuple B set) + `phase4d_parity_manifest.json` (measured boundary/cost deltas real-vs-port recorded here in Task 4).
**Interfaces:** Consumes Task 2's `Segment` (or fallback oracle). Produces: `prcts_excerpt.wav` = a DETERMINISTIC soundfile cut of the PRCTS stereo wav (start/length chosen once so the excerpt contains real speech AND silence on both channels; assert non-trivial segment structure in the oracle output -- non-vacuity); `parity_tupleA.config` = the real `1_worker_1.config` with ONLY path keys localized (files listing -> the excerpt, output dirs -> cwd; every other key byte-preserved; the diff vs the original is printed into the manifest); oracle VRCTS XML + MultiConfigResults-as-.bin per the S6/scipy precedent.
- [ ] Implement the extractor -> run it (FOREGROUND, local) -> verify the oracle outputs are non-degenerate (segment count > 1 per channel; else re-cut deterministically and record the adjudication) -> commit inputs + outputs + manifest `test(phase4d): parity oracle fixtures from the real 2015 binary`.

### Task 4: the port-side parity gates (pytest + cargo)

**Files:** Create `src/rust/tests/phase4d_parity.rs`, `tests/test_phase4d_parity.py`; Modify `scripts/extract_phase4d_fixtures.py` (adds the measured-delta recording pass), `tests/reference_data/phase4d/phase4d_parity_manifest.json`.
**Interfaces:** Consumes Tasks 1-3 fixtures + `speech` CLI / `speech_rs.Engine`. Produces the HEADLINE gates: run the port on `parity_tupleA.config` (+B) -> (a) segment counts + types EXACT vs the oracle XML (hard fail on structural mismatch), (b) per-boundary |dt| <= manifest bound, (c) result-row cost/score columns within manifest relative tolerances (timing col masked per the col-6 precedent). Bounds are MEASURED first (the extractor's delta pass runs the port locally, records max deltas), then pinned with a small headroom factor recorded in the manifest (`measured`, `pinned`, `headroom` fields per bound -- the T15 reviewer checks the headroom is honest, e.g. <= 4x measured). The audio-gated TUPLE-A REPLAY test lands here too: `tests/test_phase4d_parity.py::test_tupleA_replay_vs_saved_2015_results`, `@pytest.mark.skipif(not OPENSAD15_AUDIO_ROOT.exists(), ...)` with the documented activation path in the module docstring (user decision: maybe-later).
- [ ] Failing gates (fixtures exist, port comparison unwritten) -> implement -> GREEN on this host -> verify CI-safety (all comparisons vs committed fixtures; tolerances not libm-fragile) -> gates -> commit `test(phase4d): end-to-end parity gates vs the 2015 -ffast-math binary at measured-then-pinned tolerances`.

### Task 5: dataprep/opensad15.py

**Files:** Modify `src/python/speech/dataprep/opensad15.py`; Create `tests/test_phase4d_opensad15.py` + crafted fixtures `tests/reference_data/phase4d/opensad15/{case1.tab,case1_expected.xml,case1_expected.stm,...}` (>= 4 cases: plain multi-segment, overlapping-collar merge, first/last clamp edges, an `RI` + non-S filter mix).
**Interfaces:** Produces `convert_tab_file(audio_path: str, tab_lines: list[str]) -> tuple[bytes, list[str], str]` (xml bytes, stm lines, listing line -- the pure core; duration/lang extraction INSIDE per the legacy line reads) and `process_opensad15(listing: Path, out_listing: Path) -> None` (sequential; Pool(20) -> documented determinism deviation). Consumed by nothing downstream (leaf) but the XML must parse under `tasks/segmentation_io.rs::load_vrcts` -- add a Rust cross-parse test case in `src/rust/tests/` (fixture-shared).
- [ ] RED (hand-verified fixtures written FIRST, from a line-by-line read of `ProcessOpenSAD15Corpus.py:24-108` -- the S/RI field-4 filter, the SegsExcl 2.0-collar/0.1-snap merge incl. the `SegsExcl[-1][1] > max(minVal, begS-2.0-0.1)` branch, the 0.1 first/last clamps, `path_leaf(audiofile[:-5])`, `toto toto`, the lang fallback `split('_')[-2]`) -> implement -> GREEN -> the `load_vrcts` cross-parse -> gates -> IMPROVEMENTS (the `[:-5]` slice quirk; the no-live-oracle tier statement; Pool deviation) -> commit `feat(phase4d): OpenSAD15 corpus converter with hand-verified fixtures + load_vrcts cross-parse`.

### Task 6: dataprep/augment.py

**Files:** Modify `src/python/speech/dataprep/augment.py`; Create `tests/test_phase4d_augment.py`.
**Interfaces:** Produces `SoxRunner = Callable[[list[str]], None]`-style injection (a small protocol: `run(cmd: list[str]) -> None` + `duration(path: Path) -> float` for the soxi read), `augment_corpus(listing: Path, rng: np.random.Generator, runner: SoxRunner, dur_min: float = 2.0) -> Path` (writes `<listing stem>_addednoise.csv`, returns it). Draw order PER FILE preserved verbatim from `AugmentCorpus.py:44-70`: noise=abs(0.005*randn()), noisetype=4*rand() ladder (<1 white, <2 pink, <3 tpdf, <=4 brown), pitch=400*rand()-200, tempo=0.4*rand()+0.8, x5 variants; filename `%.3f` formats exact; the augmented listing row echoes fields 1-5 + `;`-terminator per the source.
- [ ] RED (recording-runner tests: exact command vectors for a 2-file listing with a seeded Generator; the dur_min gate; the listing bytes) -> implement -> GREEN -> gates -> IMPROVEMENTS (RNG convention-not-parity note) -> commit `feat(phase4d): corpus augmentation with injected RNG + recorded sox commands`.

### Task 7: dataprep/stm_normalize.py + the perl live oracle

**Files:** Modify `src/python/speech/dataprep/stm_normalize.py`; Create `tools/perl_oracle/run_norm.sh` (tiny: pipes fixture stdin through the REAL `.pl` in the legacy copy via `/usr/bin/perl`), extend `scripts/extract_phase4d_fixtures.py` (perl stage: crafted STM inputs -> committed input/output pairs `tests/reference_data/phase4d/stm/{light_caseN.in,light_caseN.out,full_caseN.in,full_caseN.out}`); Test `tests/test_phase4d_stm.py`.
**Interfaces:** Produces `normalize_stm_light(lines: list[str]) -> list[str]` (full transcription of `norm_stm_pkt_light.pl`) and `normalize_stm_full(lines: list[str]) -> list[str]` (the 03a regex core; the `norm-tagger|norm-parser` number arm raises `NormalizerUnavailable` -- typed bail, lost LIMSI binaries). Crafted inputs must exercise: comments, blanks, `ignore_`, every filler class, partial-word `(-x)`/`(x-)` rewrites, punctuation strip, the empty->`ignore_time_segment_in_scoring` branch; watch perl-vs-python regex semantics (`\s`, anchors, the `/o` flag is inert, iteration order of the `s///g` cascade is the LINE of the script -- transcribe in order).
- [ ] Extractor perl stage runs (local) -> committed byte pairs -> RED (python vs the committed .out bytes) -> implement -> GREEN byte-for-byte -> gates -> IMPROVEMENTS (the full normalizer's bailed arm; the .sh driver not-ported) -> commit `feat(phase4d): STM normalizers pinned byte-exact vs the real perl scripts`.

### Task 8: listing writers (batching.py) + Octave Tier-1 stage

**Files:** Modify `src/python/speech/batching.py`; Create `tools/octave_harness/stage_writelisting.m` (Tier 1: drives the REAL vendored `WriteListing.m`/`WriteWeightedListing.m` with a crafted PS struct); extend the phase4c-or-4d extractor for the stage (committed `tests/reference_data/phase4d/listing/{plain.flst,plain_worker_N.flst,weighted.lst}` goldens); Test `tests/test_phase4d_listing_writers.py`.
**Interfaces:** Produces `write_listing(base: Path, items: list[dict[str, str]], nb_workers: int) -> None` (emits `<base>.flst` + `<base>_worker_<j>.flst`, the `fliplr(length(index)-jj+1:-nbworker:1)` interleave reproduced index-for-index -- note MATLAB 1-based `jj`) and `write_weighted_listing(path: Path, items: list[dict[str, str]], values: NDArray) -> None` (6-field `%g` rows -- `%g` fixtures must probe integer/6-sig-fig/exponent boundaries per spec R5, port matches OCTAVE bytes). Consumed by Task 10.
- [ ] Octave stage -> committed goldens -> RED -> implement -> GREEN byte-exact -> gates -> commit `feat(phase4d): listing writers pinned vs the real vendored .m via Octave`.

### Task 9: algo-3 driver generalization

**Files:** Modify `src/python/speech/drivers/{train.py,state.py,test.py}` (single-net paths: `ps_from_config` lid=None arms, `_tail_lengths` -> 1-element, `_backprop_inner`/`score_genome`/`_ponderations` -> no LID cell/field, `evaluate` single-net scores); Test `tests/test_phase4d_algo3_drivers.py` + a pyo3 smoke in `tests/pyo3/test_exit_gate.py` (an algo-3 train smoke on the committed 4a tier-2 spectral fixture corpus, small steps, determinism-checked run-twice).
**Interfaces:** Consumes the 4c driver internals; produces `train`/`score_genome`/`evaluate` accepting algo-3 (`Algo_choice 3`) configs without KeyError/IndexError (the 4c-review Info finding). The `[sad]` 1-cell BackPropagation contract per `BackPropagation.m:13` (cell 1 present iff SAD backprop on; no LID cell for algo != 6).
- [ ] RED (unit: ps_from_config/tails/ponderations on the tuple-A-style config; pyo3: the smoke) -> implement -> GREEN -> full gates incl. the UNCHANGED twin exit gate -> commit `feat(phase4d): drivers generalized to single-net algo-3 configs`.

### Task 10: batch-wise inner rebuilds (listing_override goes live)

**Files:** Modify `src/python/speech/drivers/train.py` (batch mode: `create_batches` at epoch start per the legacy cadence in `Train_BLSTM.m` -- READ THE SOURCE for where GetNewBatch/WriteWeightedListing sit in the inner loop -- then per inner step `get_new_batch` -> `write_weighted_listing` to a workdir batch listing -> fresh `Engine` on a config whose files-listing key points at it -> `forward_backward(..., listing_override=...)` consumed for real), `src/python/speech/engine.py` (listing_override doc de-advisory); Test `tests/test_phase4d_batchmode.py` + extend `tests/pyo3/test_exit_gate.py::test_batch_mode_deterministic` (twin fixture, batch mode on, run twice, bit-identical).
**Interfaces:** Consumes Task 8's `write_weighted_listing`, Task 9's generalized internals, 4c's `create_batches`/`get_new_batch` (pure rotation). Produces `TrainParams`-level batch knobs (minibatch size, nb_worst -- named per the legacy config keys) defaulting OFF (full-corpus mode = the 4c behavior, unchanged exit gate).
- [ ] RED (batch listing bytes per step for a seeded run; rotation cursor advance across steps; the determinism gate) -> implement -> GREEN -> gates -> IMPROVEMENTS (any Train_BLSTM cadence quirks found) -> commit `feat(phase4d): batch-wise engine rebuilds via weighted listings (listing_override live)`.

### Task 11: .scr Octave golden + alphabetical class order

**Files:** Create `tools/octave_harness/stage_scr.m` (drives `Test_BLSTM.m`'s `.scr` writer block `:252-266` -- Tier 1 if the block is injectable behind shadow stand-ins, else the fallback transcription tier with `% legacy:` provenance; adjudicate at implementation and document which); extractor stage; committed `tests/reference_data/phase4d/scr/expected.scr`; Modify `src/python/speech/drivers/test.py` (`_class_keys` -> legacy ALPHABETICAL `keys(langMapConf)` order); Test extend `tests/test_phase4c_drivers.py` or new `tests/test_phase4d_scr.py`.
**Interfaces:** Produces the byte-pinned `.scr` law (softmax `exp(s/100)/max(1e-3,sum)`, sort desc stable, `filename lang-dial score` with the `%.15f`-vs-`num2str('%15.15f')` equivalence re-verified against Octave's actual bytes) and closes the T12 ordering divergence.
- [ ] Octave stage -> golden -> RED (the port's current class-id order should FAIL the byte comparison -- the non-vacuity proof of the order fix) -> implement (alphabetical) -> GREEN byte-exact -> gates -> commit `feat(phase4d): .scr writer pinned vs Octave + legacy alphabetical class order`.

### Task 12: T8-debt closure -- mask-vector golden arms + _fmt_scalar table

**Files:** Modify `tools/octave_harness/stage_vec2struct.m` (+ extractor) adding mask-vector cases (per-block weight masks, output mask, normalize mask -- the arms transcribed-but-uncovered since 4c T8); committed new vec2struct case fixtures; Test extend `tests/test_phase4c_genome.py` (new cases) + a `_fmt_scalar` boundary unit table (integers, negatives, 1e-5/1e+5 exponent switches, the exact `%.15g`-family behavior vs Octave).
**Interfaces:** Consumes `genome.vec2struct`; produces the closed T8 accepted-debt items (ledger Minor -> covered).
- [ ] Octave cases -> RED if any arm mismatches (a mismatch is a FINDING: fix the port arm, IMPROVEMENTS if a quirk) -> GREEN -> gates -> commit `test(phase4d): vec2struct mask-vector arms + _fmt_scalar boundaries golden-covered`.

### Task 13: blocked-four closure

**Files:** Modify `IMPROVEMENTS.md` (a "Complete-as-portable closures (Phase 4d)" subsection: balances 6-9 xml2wer-source-LOST entry with the ComputeCost.m:392-504 ssh/shell evidence; Twin pitch second pass what-it-would-take; Mode-7 WAV/CNN broken-as-committed cross-ref; cep no-data cross-ref), verify each typed bail EXISTS in code (`compute_cost` ValueError for 6-9; `bag_of_processors`/`audio.rs` cep bail; the Twin WAV/CNN bail) and cite the pinning tests; Test: only if a bail lacks a test -- add the missing pin.
**Interfaces:** Documentation + verification; produces the complete-as-portable declaration the T15 docs task and final review lean on.
- [ ] Verify bails + citations (grep real test names) -> write entries -> gates -> commit `docs(phase4d): blocked-four complete-as-portable closures`.

### Task 14: Mutation battery

**Files:** `IMPROVEMENTS.md` only (apply-fail-revert-pass; FOREGROUND; tree clean between and after).
- [ ] The S5 list: opensad15 collar 2.0->1.0; the S/RI filter widened; `[:-5]`->`[:-4]`; light-normalizer filler regex dropped; write_listing fliplr stride off-by-one; augment noisetype ladder boundary moved; parity structural assert weakened (craft a segment-type flip in a COPY of the fixture -- the gate must fail); batch-mode rotation wiring bypassed. One line each in a "Mutation battery (Phase 4d)" subsection; honest gaps recorded. Full pytest + cargo once at the end. Commit `test(phase4d): mutation battery results recorded`.

### Task 15: Docs refresh

**Files:** `README.md` (Phase 4d done-bullet; ROADMAP COMPLETE statement), `CLAUDE.md` (dataprep row -> Implemented (Phase 4d); project overview de-stubbed; the "2015 Linux binary" ERROR corrected to Mach-O/Rosetta; conventions gain the fsp_runtime + perl-oracle narration in the "Since Phase N" style; locked-decisions packer-golden deferral marked CLOSED), `pyproject.toml` (`allow_empty_bodies` REMOVED -- verify mypy passes; if anything still needs it, that is a finding), `IMPROVEMENTS.md` consistency pass.
- [ ] Raw material: the 4d ledger section + landed code (verify every claim; the T14 lesson: brief claims are code-verified before docs). Full gates. Commit `docs(phase4d): README/CLAUDE.md/IMPROVEMENTS refresh + roadmap complete`.

### Task 16: Final review + smart-commit + finish

- [ ] Final whole-branch review (most capable model; package `main@07b479b..HEAD`; ledger Minor triage incl. the parity-manifest headroom honesty check), fix wave if needed.
- [ ] Invoke the `smart-commit` skill taking the WHOLE git branch into account, then `superpowers:finishing-a-development-branch` (user merges via PR). NEVER push.

## Self-Review

1. **Spec coverage**: S1.1->T5; S1.2->T6; S1.3->T7; S1.4->T8; S1.5->T2; S1.6->T3+T4 (+T1 the packer closure; the audio-gated replay inside T4); S1.7->T10/T9/T11/T12; S1.8->T13; S2 tiers embedded per task; S3 risks: R1 T2 fallback field, R2 T3 non-degeneracy + T4 measured bounds, R3 T5 cross-parse, R4 T3/T4 adjudication notes, R5 T8 boundary fixtures, R6 extractor budget; S4 deviations logged in T5/T6/T2; S5->T14; S6->T15; S7->T16.
2. **Placeholders**: none -- the two adjudicate-at-implementation notes (T11 Tier-1-vs-transcription, T2 matio tag) carry explicit decision rules per the established pattern.
3. **Type consistency**: `write_weighted_listing` (T8) consumed by T10 with the same signature; T9's generalized internals consumed by T10; T1's fixture packs consumed by T3/T4 under the same paths; `SoxRunner` local to T6; extractor filenames consistent across T3/T4/T7/T8/T11/T12.
