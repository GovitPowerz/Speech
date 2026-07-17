# Phase 5: un-quirk + training foundation -- design spec (+ the Roadmap 2 record)

Date: 2026-07-10
Branch: `feature/phase-5-unquirk-training-foundation` (base main@74284d6, post-4d-merge)
Approach: A ("two-track with the tag first"), user-approved.

This spec opens the POST-PORT ERA and doubles as the durable Roadmap 2 record. The port
era (phases 0a-4d) ended with the roadmap complete and parity proven against the
resurrected 2015 production binary. Roadmap 2's ultimate goal (user, 2026-07-10): a very
efficient (fast, low-memory) SAD + LID Rust binary taking audio in both OFFLINE and
ONLINE (streaming chunk) modes, then newer neural architectures (mamba, transformers,
CfC, xLSTM, ...) -- with the tool properly trainable FROM SCRATCH first, to build a
strong baseline to improve on. Available data: the user holds the LRE03 + LRE07 corpora
(the legacy tree's `Listings/` reference them); SAD corpora to be re-sourced (OpenSAD15
would re-activate the dormant tuple-A replay).

## R2. Roadmap 2 (user-approved ordering: 5 -> 6 -> 7 -> 8 -> 9)

- **Phase 5 (THIS SPEC) -- un-quirk + training foundation.** The golden-regime change
  (port-truth replaces legacy-truth), the correctness-critical bug sweep, and the modern
  from-scratch training loop. Exit: from-scratch convergence on committed fixtures,
  deterministic, both heads (SAD + Twin LID).
- **Phase 6 -- baseline training on real data.** LRE03/07 re-attached, augmentation live,
  SAD corpus when sourced; SAD + LID trained from scratch; a DCF/EER eval harness. Exit:
  a reproducible baseline meeting/beating the 2015 numbers -- the reference every later
  phase measures against.
- **Phase 7 -- the efficient binary (offline).** A fast inference path beside the exact
  one: f32, `rustfft`/`realfft`, BLAS/SIMD matmuls, allocation/memory hygiene, an RTF +
  peak-RSS benchmark suite. Gate: quality-parity at tolerance vs the exact path on the
  phase-6 baseline models + measured speed/memory budgets.
- **Phase 8 -- online/streaming mode.** Chunked-input API (lib + CLI), streaming feature
  extraction, windowed-BLSTM-with-bounded-lookahead as the first causal cut. Gate:
  bounded-latency streaming output equivalent-at-tolerance to offline on the same audio.
- **Phase 9 -- new architectures.** Mamba/transformer/CfC/xLSTM behind a layer
  abstraction, trained via the phase-5/6 machinery against the phase-6 baseline; the
  causal-friendly ones upgrade phase 8 from windowed-lookahead to truly causal.

Ordering rationale (user-arbitrated): training before performance, because the phase-7/8
quality gates need the phase-6 eval harness + a trusted baseline, and phase-5 fixes
change outputs that perf work would otherwise re-baseline twice.

## S0. Locked decisions (user-approved this brainstorm)

1. **Modern training regime**: standard init (Xavier/He) + gradient training (SMORMS3
   stays the trainer -- bit-pinned, adaptive; optimizer experiments belong to phase 9);
   QPSO narrows to DSP/config hyperparameters ONLY. The legacy genome-weight encode
   (`coeff_NN*(2p/adim-1)`) stays ported-and-pinned but permanently unused by the new
   path -- the 4c binding deviation becomes the design.
2. **Tag + retire**: the roadmap-complete merge commit is tagged `legacy-parity-v1` --
   the permanent, runnable parity proof. The 2015-binary parity gates leave HEAD in the
   same change (deleted, not skipped; README/CLAUDE.md point at the tag). No
   legacy-compat dual paths anywhere.
3. **Correctness-critical sweep scope**: fix what produces wrong results or UB. Inert
   quirks (display, dead-code reproductions, working in-band conventions like the
   `>150/-200` signaling, the cost-law double-read family) stay documented, not fixed.

## S1. Deliverables

### S1.1 The tag + gate retirement (the phase's task 1)
Tag `legacy-parity-v1` = main@74284d6 (annotated, message naming the proof surface: the
4d parity gates, fixtures, manifest, and the fsp_runtime/oracle-harness family). At HEAD:
delete `src/rust/tests/phase4d_parity.rs` + `tests/test_phase4d_parity.py` (and their
ci.yml arg), KEEP every fixture (`tests/reference_data/phase4d/` feeds the packer golden,
the algo-3 smokes, `test_phase4d_fixtures.py`'s integrity guards -- those stay). README +
CLAUDE.md gain the regime-change note: goldens assert PORT-TRUTH from this commit on;
the parity proof lives at the tag. IMPROVEMENTS.md gains a "Phase 5 fix protocol"
preamble (S2).

### S1.2 The adjudication sweep (task 2, the fix-list source of truth)
Walk every IMPROVEMENTS.md entry against the S0.3 criterion; emit the definitive fix
list INTO the ledger with per-fix: the entry, the wrong-result/UB evidence, the golden
families it re-pins, dependencies/batching order (worst-cascade first), and the
observable-consequence test that will prove the fix (each fix must have one -- a fix
with no observable consequence is by definition not correctness-critical and drops back
to documented). Expected members from session knowledge (the sweep adjudicates, this
list does not bind it): training-path -- the CreateBatches aggregate-slot clobber, the
missing per-eval class-balance rescale (`ComputeGradient.m:72-96` semantics; the
in-class model built in the driver from the mapping file), the `get_new_batch`
no-iteration-cap hazard, the weights2nnet tail-drop (verify fixed-by-design in the
port's round trip, flip the entry); inference/eval-path -- the uninitialized-LTSVshift
UB (proper init), the confusion sticky `posTarget`/`posBestNotTarget` (per-row reset --
directly improves phase-6 EER/DCF fidelity), the stereo-CSV-reference-never-loads bug
(silent scoring corruption). Explicitly kept-documented: in-band signaling, display
quirks, dead-code faithfulness, cost-law double-read, `nb_words` default -1 (dead
surface -- xml2wer lost).

### S1.3 The un-quirk track (tasks batched per golden family)
Each fix follows the S2 protocol. Python-side fixes (batching, scoring, drivers) re-pin
pytest goldens; Rust-side fixes (LTSVshift, confusion core, segmentation_io) re-pin the
affected `tests/reference_data/` fixtures via the established extractors where one
exists, or direct re-dump where the golden is port-generated. The Octave/C++ oracle
harnesses are NOT regenerated for fixed sites -- each fixed site's IMPROVEMENTS entry
documents the now-deliberate oracle divergence (the oracle still describes the LEGACY;
the port has left it on purpose).

### S1.4 Weight init (the modern-loop track, piece 1)
`src/python/speech/init_weights.py` (or a weight_bridge extension -- plan decides by
size): `init_weights(spec: NnetSpec-equivalent, rng: np.random.Generator, scheme:
"xavier"|"he", forget_bias_one: bool = True) -> list[NDArray]` -- per-layer fan-in/out
from the spec shapes (the same shapes the packer validates), zero biases except the
LSTM forget-gate bias = 1.0 when flagged (standard trick, default on), the mean/std
normalize tail initialized to identity (mean 0 / std 1). Output = flat packs entering
the engine through `set_weights` / written as `.bin` seeds. Numpy-side only; NO Rust
changes. Seeded-deterministic (the exit gate depends on it).

### S1.5 The modern training loop (piece 2)
`drivers/train.py` grows `train_modern(state, seed, params) -> TrainResult` (the 4c/4d
`train` stays as the legacy-regime driver until phase 6 decides its fate): from-scratch
init (S1.4) or checkpoint resume; EPOCHS over the corpus with the now-fixed batch
machinery (create_batches once, get_new_batch + weighted listing + fresh engine per
step -- the T10 wiring, now on corrected batching); per-epoch VALIDATION on a held-out
listing (forward-only engine run, cost + the FIXED confusion metrics -- the role
`valid_batch.m` never got to play in the port); EARLY-STOP with patience on validation
cost; checkpointing best + last. SMORMS3 the trainer, no external LR schedule (its
adaptivity suffices; YAGNI). Training params live in a `[training]` TOML section
(canonical config side); the driver keeps synthesizing the flat engine config through
the existing seam.

### S1.6 QPSO narrowing (piece 3)
The genome binding generalizes from the two CostPonderation keys to the FULL NON-WEIGHT
key set: the vec2struct walk's DSP/config emissions (freq bands, windows, LTSV/TDC
params, decision thresholds, calibration laws) inject onto the base config per
candidate; the weight blocks are masked OUT of the genome permanently (genome_length
shrinks accordingly -- a deliberate, documented break from the legacy genome; the
legacy-faithful walk stays pinned by the existing 4c/4d goldens which do not change,
since vec2struct itself is untouched -- only the DRIVER's mask/injection changes).
This is the phase-6 hyperparameter-search surface, built now while the binding code is
warm. Exit-gate-adjacent check: a 2-candidate QPSO run over a non-weight genome
produces distinct engine configs and distinct costs (the 4c non-vacuity pattern,
now over the full key set).

### S1.7 The exit gate
`tests/pyo3/test_exit_gate.py` gains the phase-5 gates: FROM-SCRATCH CONVERGENCE on
committed fixtures, both heads -- algo-3 SAD on the 4a tier-2 spectral corpus and the
algo-6 Twin on the 4b tiny-net corpus. Asserts per head: (a) training cost strictly
improves over the run (monotone-improving best-cost, margin-based final < initial);
(b) the trained model beats the untrained init on a HELD-OUT fixture file by a margin;
(c) the validation/early-stop path exercises (a crafted stop case triggers patience);
(d) run-twice bit-identity at a fixed seed. CI-runnable (tiny nets, bounded steps).

## S2. The fix protocol (the golden-regime change, every un-quirk task follows it)

1. RED: the existing golden/test FAILS under the fix (proof the fix is observable; a
   fix nothing catches is either untested legacy surface -- add the pin FIRST against
   the old behavior, then fix -- or not correctness-critical, drop it back).
2. Re-pin: the golden regenerates/re-derives to the FIXED behavior.
3. Mutation: revert-the-fix breaks the new golden (recorded per fix, battery-style).
4. IMPROVEMENTS.md: the entry flips to "FIXED (phase 5, commit <hash>)" keeping the
   original legacy-behavior description for the record, plus the oracle-divergence
   note where an oracle harness still describes the legacy behavior.

## S3. Risks

- R1 re-pin cascades (one fix invalidates many goldens): the S1.2 sweep orders fixes to
  batch golden families, worst-cascade first; fixes within a family land as one task.
- R2 the rescale's class model: the mapping file is already a driver input; the in-class
  model is built driver-side (no engine change). If a semantic hole appears (the legacy
  read a struct the port lacks), the sweep documents-and-descopes rather than invents.
- R3 convergence-gate flakiness on tiny corpora: fixed seeds, margin asserts,
  improvement-not-absolute-quality criteria, deterministic engine (N=1 lanes).
- R4 half-fixed states: the gates retire at tag-time (S1.1) BEFORE any fix lands; no
  gate is ever red at HEAD.
- R5 the modern loop's from-scratch models may be poor on tiny fixtures (they will be):
  the exit gate measures IMPROVEMENT and DETERMINISM only; quality is phase 6's job on
  real data.

## S4. Documentation + finish

Per-fix IMPROVEMENTS flips with real citations (the protocol embeds it); README/CLAUDE.md
regime-change notes land with task 1 and get a final consistency pass at phase end;
the per-phase superpowers loop continues unchanged (subagent-driven, fresh
implementer + reviewer per task, mutation batteries, final whole-branch review on the
most capable model, smart-commit taking the whole branch into account,
finishing-a-development-branch; user merges via PR; never push).
