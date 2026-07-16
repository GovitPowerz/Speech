# Phase 5: un-quirk + training foundation -- implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip the golden regime to port-truth (tag + retire the legacy-parity gates), fix the correctness-critical legacy bugs, and land the modern from-scratch training loop -- exit-gated on dual-head from-scratch convergence.

**Architecture:** Spec `docs/superpowers/specs/2026-07-10-phase-5-unquirk-training-foundation-design.md` (approach A, two-track with the tag first). Track 1 = the un-quirk sweep (fix -> RED -> re-pin -> mutation -> IMPROVEMENTS flip, per the S2 protocol). Track 2 = the modern loop (Xavier/He init, epochs/validation/early-stop, QPSO narrowed to the non-weight genome). The S1.2 adjudication sweep is the fix-list source of truth -- tasks 4-7's expected membership below yields to it.

**Tech Stack:** Python >=3.14 (uv, numpy, pydantic), Rust edition 2024, the existing PyO3 seam (`speech_rs.Engine`), TOML canonical config. No new dependencies.

## Global Constraints

- THE FIX PROTOCOL (spec S2) governs every un-quirk task: RED (the old golden fails under the fix -- if nothing catches it, pin the OLD behavior first, then fix) -> re-pin to port-truth -> revert-mutation check recorded -> IMPROVEMENTS entry flips to "FIXED (phase 5, commit <hash>)" keeping the legacy description + the oracle-divergence note.
- The S1.2 sweep's adjudication GOVERNS fix-task membership over this plan's expected lists; a fix with no observable consequence drops back to documented.
- Branch `feature/phase-5-unquirk-training-foundation`; per-task commits with trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; never push; never gh; never modify `legacy/` or `/Users/govit/Git/Govit/FastSpeechProcessing-legacy`.
- FOREGROUND everything -- NEVER use run_in_background or Monitor; never read exit codes through pipes (log-capture instead).
- Gates per task: `uv run pytest tests` + `./lint_code.sh` (+ `uv run ruff check --config=pyproject.toml scripts` when touching scripts/); `cd src/rust && cargo test` + `cargo fmt --check` + `cargo clippy --workspace --all-targets -- -D warnings` when Rust changes.
- ASCII only. Oracle harnesses are NOT regenerated for fixed sites (the oracle describes the legacy; divergence is documented per site).

---

### Task 1: the tag + gate retirement + regime-change docs

**Files:** Tag `legacy-parity-v1` = main@74284d6 (annotated). Delete: `src/rust/tests/phase4d_parity.rs`, `tests/test_phase4d_parity.py`; Modify `.github/workflows/ci.yml` (drop the `tests/test_phase4d_parity.py` arg from the python-pyo3 job), `README.md` + `CLAUDE.md` (the regime-change note: goldens assert PORT-TRUTH from this commit; the parity proof lives at `legacy-parity-v1`; a short "Roadmap 2" section pointing at the phase-5 spec), `IMPROVEMENTS.md` (the "Phase 5 fix protocol" preamble stating S2 verbatim).
**Interfaces:** Produces the tag (create locally: `git tag -a legacy-parity-v1 74284d6 -m "..."`; the user pushes tags himself -- note it in the report) and the retired-gate state every later task builds on. KEEP `tests/reference_data/phase4d/` entirely (the packer golden, algo-3 smokes, `tests/test_phase4d_fixtures.py` integrity guards all stay green).
- [ ] Verify the survivors still pass after the deletion (`uv run pytest tests/test_phase4d_fixtures.py tests/test_phase4d_packer.py` + `cargo test --test phase4d_packer_golden --test phase4d_opensad15_vrcts`) -> full gates -> commit `chore(phase5): tag legacy-parity-v1 + retire the 2015-parity gates (port-truth regime)`.

### Task 2: the adjudication sweep (the fix-list source of truth)

**Files:** Read `IMPROVEMENTS.md` end to end; write the fix list into `.superpowers/sdd/progress.md` (the phase-5 section) AND commit a compact table into `docs/superpowers/plans/2026-07-10-phase5-fixlist.md` (entry, wrong-result/UB evidence, golden families re-pinned, batch/order, the observable-consequence test per fix).
**Interfaces:** Produces the DEFINITIVE membership + ordering for tasks 4-7. Criterion (spec S0.3): wrong results or UB only. Expected members (this plan's guess, the sweep governs): training-path -- CreateBatches clobber, per-eval class-balance rescale, get_new_batch no-cap, weights2nnet tail-drop (verify fixed-by-design, flip); inference/eval -- LTSVshift UB, confusion sticky-index (both language sides), stereo-CSV reference load. Explicitly kept-documented: in-band `>150/-200`, display quirks, dead-code reproductions, cost-law double-read, `nb_words` -1.
- [ ] Sweep -> the committed fixlist doc + ledger entry -> commit `docs(phase5): the correctness-critical fix list adjudicated`.

### Task 3: weight init (modern-loop piece 1 -- independent of the fixes, can land early)

**Files:** Create `src/python/speech/init_weights.py`; Test `tests/test_phase5_init_weights.py`.
**Interfaces:** Produces `init_weights(spec, rng: np.random.Generator, scheme: Literal["xavier","he"] = "xavier", forget_bias_one: bool = True) -> list[NDArray[np.float64]]` where `spec` is the same structured description `weight_bridge.pack_weights` consumes (reuse its spec type -- read `weight_bridge.py` first; do NOT invent a parallel spec). Per-layer fan-in/fan-out from the spec shapes; gate/cell blocks scaled per scheme; biases 0.0 except LSTM forget-gate bias = 1.0 when flagged; the mean/std normalize tail = (0.0, 1.0) identity. Returns flat packs (one per net) valid for `speech_rs.Engine.set_weights` AND `weight_bridge.write_bin` seeding.
- [ ] RED (shape/adim invariants: pack length == the spec-derived count for the tuple-A spec fixture; forget-bias positions == 1.0 -- derive positions from the packer layout, do not hardcode magic offsets; determinism: same seed -> bit-identical; xavier-vs-he variance ratio sanity on a big block) -> implement -> GREEN -> the round trip: `init_weights` -> `set_weights` -> `weights()` -> bit-identical (pyo3 test, importorskip) -> gates -> commit `feat(phase5): seeded Xavier/He weight init over the packer spec`.

### Task 4: training-path fixes, batching family (sweep-governed)

**Files:** Modify `src/python/speech/batching.py` (the CreateBatches aggregate-slot clobber -- index `Cases`/`WorstCases` by CLASS VALUE not loop position; the `get_new_batch` iteration cap -- a bounded-walk guard raising a typed error instead of spinning); re-pin `tests/test_phase4c_batching.py` goldens (the Octave-pinned rotation goldens that encode the clobber); Test additions for the fixed semantics + the cap.
**Interfaces:** Consumes Task 2's adjudication. Produces the FIXED `create_batches` (aggregate slot preserved alongside contiguous class labels) and a `get_new_batch` that raises `BatchRotationStuck` (new typed exception, exported) after `2 * len(index)` fruitless advances instead of spinning -- the T13-4d latent hazard closed.
- [ ] Protocol: RED (the existing clobber-reproducing golden fails under the fix -- it exists, `test_phase4c_batching.py` pinned the clobber; the no-cap hazard has NO pin: add the would-spin case against OLD behavior first via a bounded harness, then fix) -> re-pin -> revert-mutation -> IMPROVEMENTS flips (2 entries) -> gates -> commit `fix(phase5): CreateBatches class-value indexing + bounded batch rotation`.

### Task 5: training-path fixes, rescale + tail (sweep-governed)

**Files:** Modify `src/python/speech/drivers/train.py` (the per-eval class-balance rescale: build the in-class model from the mapping file -- `nbOfElem/sumInClassIndex` per `ComputeGradient.m:72-96` semantics, applied to the weighted-listing values per gradient eval when batch mode is ON), `src/python/speech/batching.py` if the model helper belongs there; verify-and-flip the weights2nnet tail-drop entry (the port's `pack/unpack` round trip carries the tail -- prove it with a directed test if none exists, then flip the entry to FIXED-BY-DESIGN); Test extend `tests/test_phase4d_batchmode.py` (the rescaled listing bytes for a crafted 2-class corpus) + `tests/test_phase4c_weight_bridge.py` (the tail round-trip pin if missing).
**Interfaces:** Consumes Task 4's fixed batching. Produces `class_balance_values(listing_records, mapping_path) -> NDArray` (the rescale model, driver-consumed) wired into `_BatchRunner.next_listing`.
- [ ] Protocol per fix -> gates -> commit `fix(phase5): per-eval class-balance rescale + the tail-drop entry flipped`.

### Task 6: inference/eval fixes, confusion + reference-load family (sweep-governed)

**Files:** Modify `src/rust/src/engine/confusion.rs` + `src/python/speech/scoring.py` (the sticky `posTarget`/`posBestNotTarget`: declare/reset PER ROW in both languages -- the two ports share the quirk and must fix identically), `src/rust/src/tasks/segmentation_io.rs` or the loader call site (the stereo-CSV-reference-never-loads bug -- the channel-gate condition that silently drops the reference for stereo CSV refs); re-pin `src/rust/tests/phase4b_confusion_golden.rs` fixtures + the scoring pytest goldens + any corpus-level goldens the reference fix shifts (the sweep's cascade analysis governs the batch).
**Interfaces:** Produces per-row-correct confusion decoding (directly consumed by phase 6's EER/DCF) and stereo CSV references that actually load (scored runs on stereo corpora stop being silently unscored).
- [ ] Protocol per fix (the confusion fix's RED = the sticky-quirk goldens fail; the CSV fix's RED = a stereo-CSV corpus test currently pins EMPTY references -- it exists per the 4a ledger, find it) -> re-pin -> mutations -> IMPROVEMENTS flips -> full Rust + Python gates -> commit `fix(phase5): per-row confusion indices + stereo CSV reference loading`.

### Task 7: inference fixes, LTSVshift UB (sweep-governed)

**Files:** Modify `src/rust/src/tasks/lid.rs` (the algo-5 uninitialized-LTSVshift: initialize the member properly at construction -- read the 4b IMPROVEMENTS entry for the pinned deterministic stand-in the port chose, and what the legacy UB did); re-pin the affected algo-5 goldens (`tests/reference_data/phase4b/lid5_*` family via the port-side dumps -- NOT the C++ harness).
**Interfaces:** Produces a properly-initialized algo-5 first-construction path.
- [ ] Protocol (RED = the lid5 goldens pinning the UB-era value fail) -> re-pin -> mutation -> IMPROVEMENTS flip -> gates -> commit `fix(phase5): algo-5 LTSVshift initialized (UB-era behavior retired)`.

### Task 8: the modern training loop (modern-loop piece 2)

**Files:** Modify `src/python/speech/drivers/train.py` (add `train_modern(state: RunState, seed: int, params: ModernTrainParams) -> TrainResult`; the 4c/4d `train` stays untouched), `src/python/speech/drivers/state.py` (`ModernTrainParams` pydantic: epochs, patience, valid_listing, minibatch knobs, init scheme/seed), `src/rust/src/toml_config.rs` KEY_TABLE only if a new engine-side key is genuinely needed (expect NONE -- training params are driver-side; document); Create the `[training]` TOML section convention in `configs/` (a commented example); Test `tests/test_phase5_train_modern.py` (unit: epoch/validation/early-stop state machine against a stub engine callable) + a pyo3 smoke.
**Interfaces:** Consumes Task 3's `init_weights`, Tasks 4-5's fixed batching/rescale, the existing `forward_backward`/`Smorms3`/`make_engine` seams. Produces: from-scratch init or checkpoint resume; epochs over batch mode; per-epoch forward-only validation on `valid_listing` (engine run with backprop flags off, cost + the FIXED confusion metrics recorded); early-stop on validation-cost patience; best+last checkpoints (`best_*.bin`, `last_*.bin`, `train_history.json`). SMORMS3 the trainer; no LR schedule.
- [ ] RED (the state-machine unit tests: patience triggers at the crafted epoch; best-checkpoint selection; resume-from-last equivalence) -> implement -> GREEN -> the pyo3 smoke (tiny corpus, 2 epochs, validation path exercised) -> gates -> commit `feat(phase5): the modern training loop - epochs, validation, early-stop, checkpoints`.

### Task 9: QPSO narrowing to the non-weight genome (modern-loop piece 3)

**Files:** Modify `src/python/speech/drivers/train.py` (the genome mask/injection: from the 2 CostPonderation keys to the FULL non-weight key set -- every vec2struct-emitted key EXCEPT the weight blocks (`*_Layer_*` weight/bias matrices) and the normalize tail; `genome_length` recomputed accordingly via the masked walk), `src/python/speech/genome.py` ONLY if a mask-construction helper is missing (vec2struct itself MUST NOT change -- its 4c/4d goldens stay byte-green as the regression sentinel); Test `tests/test_phase5_genome_narrowing.py` + extend the pyo3 exit-gate file (a 2-candidate QPSO run over the narrowed genome -> distinct engine configs AND distinct costs -- the 4c non-vacuity pattern over the full key set).
**Interfaces:** Consumes the 4c `quantum_pso`/`vec2struct`/`masking_validation`. Produces `build_hyperparam_mask(ps) -> mask` (weight blocks masked out permanently) + the generalized `_eval_config_text` injection (all non-weight decoded keys onto the base config). The genome shrink is a DELIBERATE documented break from the legacy genome (IMPROVEMENTS entry, not a fix flip -- a design divergence record).
- [ ] RED (the narrowing tests: mask excludes exactly the weight/tail dims -- derive from the walk, no magic counts; injection covers the decoded non-weight keys; vec2struct goldens untouched) -> implement -> GREEN -> gates -> commit `feat(phase5): QPSO narrowed to the non-weight hyperparameter genome`.

### Task 10: the dual-head from-scratch exit gates

**Files:** Extend `tests/pyo3/test_exit_gate.py`: `test_from_scratch_sad_converges` (algo-3, the 4a tier-2 spectral corpus fixtures) + `test_from_scratch_twin_converges` (algo-6, the 4b tiny-net corpus) + `test_early_stop_triggers` (the crafted-patience case).
**Interfaces:** Consumes Tasks 3/8 (+4/5 transitively). Asserts per head (spec S1.7): (a) margin-based improvement (best cost strictly below the epoch-0 cost by a stated margin; monotone best-cost series); (b) the trained model beats the untrained init on a HELD-OUT fixture file (split the fixture listing; forward-only eval both models, compare costs); (c) run-twice bit-identity at a fixed seed (checkpoints + history). CI-runnable: tiny nets, bounded epochs (<=4), target < ~60s total.
- [ ] RED (gates fail against a stubbed no-op trainer or before Task 8 lands -- sequence ensures real RED) -> GREEN on the real loop -> gates -> commit `test(phase5): dual-head from-scratch convergence exit gates`.

### Task 11: mutation battery

**Files:** `IMPROVEMENTS.md` only (apply-fail-revert-pass; FOREGROUND; tree clean between and after).
- [ ] The battery: every un-quirk fix's revert-mutation RE-RUN from the committed state (tasks 4-7 recorded them; re-verify post-integration); plus the loop pieces: init forget-bias flag dropped -> the bias-position test fails; the validation cadence bypassed (early-stop never consulted) -> `test_early_stop_triggers` fails; the rescale skipped -> the rescaled-listing-bytes golden fails; the genome mask widened to include a weight dim -> the narrowing test fails; the convergence gate's margin zeroed -> a crafted stagnant trainer passes (proving the margin bites -- test-side demonstration, the item-7-4d pattern). One line each in a "Mutation battery (Phase 5)" subsection; honest gaps recorded. Full pytest + cargo once at the end. Commit `test(phase5): mutation battery results recorded`.

### Task 12: docs refresh

**Files:** `README.md` (the Phase 5 done-bullet opening the Roadmap 2 section; the regime-change note finalized), `CLAUDE.md` (module rows touched by fixes updated -- batching/scoring/confusion/lid/train rows note the FIXED-at-phase-5 divergences; the "Since Phase 5" conventions sentence: port-truth goldens, the fix protocol, oracle harnesses describe the legacy only), `IMPROVEMENTS.md` consistency pass (every flip has the commit hash + the oracle-divergence note; the kept-documented set explicitly marked "kept by decision, phase 5").
- [ ] Raw material: the phase-5 ledger + landed code (verify every claim; brief claims are code-verified before docs). Full gates. Commit `docs(phase5): README/CLAUDE.md/IMPROVEMENTS refresh - the port-truth era opens`.

### Task 13: Final review + smart-commit + finish

- [ ] Final whole-branch review (most capable model; package `main@74284d6..HEAD`; ledger Minor triage; special attention: every re-pinned golden's mutation evidence, the vec2struct-untouched sentinel, the exit gates' honesty).
- [ ] Invoke the `smart-commit` skill taking the WHOLE git branch into account, then `superpowers:finishing-a-development-branch` (user merges via PR). NEVER push.

## Self-Review

1. **Spec coverage**: S1.1->T1; S1.2->T2; S1.3->T4/T5/T6/T7 (membership sweep-governed); S1.4->T3; S1.5->T8; S1.6->T9; S1.7->T10; S2 protocol->the Global Constraints block + every fix task; S3 risks: R1 the sweep's cascade ordering (T2), R2 documents-and-descopes (T5), R3 margins/seeds (T10), R4 gates retire in T1 before any fix, R5 improvement-not-quality (T10); S4->T12/T13.
2. **Placeholders**: none -- the sweep-governed caveat is an explicit governance rule, not a TBD; every expected fix names its file, RED source, and golden family.
3. **Type consistency**: `init_weights` (T3) consumed by T8/T10 with the same signature; `ModernTrainParams`/`train_modern` (T8) consumed by T10; `build_hyperparam_mask` (T9) self-contained; `BatchRotationStuck` (T4) named once; `class_balance_values` (T5) consumed by the T8 loop via `_BatchRunner`.
