# Phase 4c: PyO3 seam + Python optimizer + TOML canonical config - Design

Date: 2026-07-08
Status: approved (brainstorm 2026-07-08)
Branch: `feature/phase-4c-pyo3-optimizer` (off `main` at `7add21f` - Phases 0-4b
merged via PRs #7/#8)

Phase 4 decomposition recap: 4a engine drivers (DONE) -> 4b LID (DONE) ->
**4c (this spec)** -> 4d dataprep + held-out score parity + the fsp-binary
tolerance comparison.

## S1. Scope

IN (ladder; each rung reuses the prior's fixtures):

1. **Rust API promotion**: the `_for_test` hooks on `CorpusProcessor`
   (`corpus_processor.rs:825-1000`) become real public methods -
   `results_matrix() -> Array2<f64>` (the post-run `MultiConfigResults`),
   `weights(pos) -> Vec<Vec<f64>>` / `set_weights(pos, &[Vec<f64>])` (the
   `[sad, lid]` pairing preserved), `weights_derivatives(pos) ->
   Vec<Array2<f64>>` (Nx2), `input_statistics(pos)`, `grad_check(epsilon,
   max_weights) -> GradCheckReport` (promoting `grad_check_all_for_test`'s
   per-network form). Thin promotions, doc-commented as the seam surface;
   epoch traces stay `test-support`; the full golden suites guard zero
   behavior change.
2. **`speech-py` binding (COARSE, corpus-level - the legacy per-evaluation
   contract)**: `speech_rs.Engine` (`#[pyclass]` over `CorpusProcessor`) -
   ctor from config paths (`.config`/`.toml` by extension) or pre-parsed
   dicts; `run()` releases the GIL; numpy f64 arrays out (results matrix,
   per-net weights list, per-net Nx2 derivatives list, stats); `anyhow` ->
   `RuntimeError` with chain. Module fns: `parse_legacy_config(text) ->
   dict`, `load_toml_config(path) -> dict`, `version()`. Nothing else.
   CI's `python-pyo3` job gains the seam pytest subset (it currently only
   builds + import-checks).
3. **TOML canonical config**: serde schema mapping DECLARATIVELY onto the
   legacy key space - sections/keys flatten through one conversion table
   into the same `IndexMap<String, String>` the engine consumes (ONE config
   semantics; `toml -> map == .config -> map` is a testable property, then
   bit-identical engine outputs on committed fixtures). `configs/lid/
   lid_blstm.toml` grows to full key coverage; a legacy->TOML converter in
   the CLI; typed getters = future work (documented). The unused `toml = 0.9`
   dep (`Cargo.toml:19`) finally gets wired.
4. **Python deterministic layer** (each vs Octave goldens, S3):
   - `weight_bridge.pack_weights`/`unpack_weights` (the last Phase-0a stubs;
     hypothesis round-trip property tests - the CLAUDE.md mandate).
   - `genome.py` (new module): the `vec2struct` bijection
     (`functions/vec2struct.m`, 1,693 LOC - the single largest port): the
     genome walk with `count_param`, mask semantics (masked field = fixed +
     `out_param` inverse write-back), the NN-weight encoding `coeff_NN *
     (2*param/adim - 1)`, `Force_Identical_Rows`/`Force_Symetry`, the
     calibration cost-law decode (`rem(round(5*|p|/adim),5)`), clamps.
   - `engine.py`: `ComputeCost`/`ComputeGradient` on the seam - the
     MultiConfigResults column schema (col5/last = seg cost num/denom,
     col15/end-1 = LID, cols 17:end-2 = per-class scores with the >150/+200
     encoding), `sortrows([1 2 3])` aggregation, `deriv ./ max(1,count)`
     averaging (`ComputeCost.m:357`), pooled input-stats recombination
     (`:284-353`), L2 penalties (`:554` cost / `:362` gradient,
     bias-excluded), the pure balance laws 0/3/4/5/10 (`:439-624` incl. the
     over-90 saturation penalties and the balance-10 LID calibration
     cutoff/relative-error laws).
   - `batching.py`: `CreateBatches` stratification (incl. the full-batch
     degenerate gate), `GetNewBatch` round-robin rotation (no RNG),
     `getCases`/`getWorstAndBest` best/middle/worst selection.
   - `scoring.py`: `check_grad` (central-difference assembly, diagnostic
     contract - no hard threshold, per `CheckGrad.m`), `masking_validation`
     (1e-12 round-trip), the completed `confusion_matrix`.
5. **Optimizers** (`optimizers.py`):
   - `Smorms3` (class): the exact update rule (`SMORMS3.m:324-363`) incl.
     `r = 1/(delta+1)`, the two `eps=1e-16` placements, the `min(lrate, ...)`
     per-param cap, the x10 geometric lrate warmup capped at 1e-1
     (`:347`), and the `theta = theta_out + dtheta` ROUND-TRIP subtlety
     (theta_out returned from f_df, not the input). Bit-pinned vs Octave.
   - `rprop()`: `functions/Rprop.m` (38 LOC; etap 1.2 / etam 0.5 / delta0
     0.01 / deltamin 1e-9 / deltamax 1.0, cost-gated backtrack; the `:5`
     rand is DEAD - overwritten at `:6`). Distinct from the engine's C++
     Rprop. Bit-pinned vs Octave.
   - `quantum_pso()`: full transcription of `QuantumPSO.m` - 2*ps uniform
     init + PSOseed overwrite + opposition-based learning (`:252-327`), the
     QDPSO update (`:414-443`: contraction-expansion 0.75->0.25 linear,
     ranking-operator roulette attractor, `log(1/u)` jump with the 1e-5/1e-32
     floors), DE recombination + Levy mutation (`:463-479`,
     Chambers-Mallows-Stuck for the alpha-stable), the backprop gate
     (`:480-485`), the DEAD-but-stream-consuming velocity banks
     (`:375-411` - they draw from the RNG, so the draw order must reproduce
     them), stall termination (`ergrdep`), the final-generation re-eval.
     Bit-pinned vs Octave GIVEN the shared random table (S3).
   - `cmaes`: thin `pycma` wrap of the fallback branch (`Train_BLSTM.m:1147`
     invocation shape: x0 = adim/2 * ones, sigma0 = adim/3). NOT bit-pinned
     (alternative branch; Hansen 3.61.beta vs pycma lineage documented).
6. **Drivers + CLI**: `drivers/{init,train,retrain,test}.py` with a typed
   pydantic `RunState` replacing the `PS` god-struct (Init: experiment dirs
   + listing processing + state save; Train: masks, genome sizing via the
   vec2struct count, `MaskingValidation`, the QPSO/SMORMS3 wiring, uniform
   `[0, adim]` population init through the fixed-seed Generator; ReTrain:
   resume from checkpoint incl. `nnet_best` seeding; Test: `.scr` score
   outputs - softmax `exp(score/100)/sum`, sorted). `cli.py` dispatch.

OUT (deferred/excluded): balance 6-9 WER laws (xml2wer shell-outs -> 4d);
Adam/RMSprop/sfo/pso_Trelea (NOT wired in legacy - excluded per YAGNI,
documented); dataprep (4d); held-out score parity + fsp comparison (4d);
distributed multi-worker execution (nbworker=1 in-process is the 4c
contract; the lock-file machinery stays stubbed); TOML typed getters.

**Determinism (the deliberate deviation, lane-model style)**: the legacy
optimizer path is nondeterministic BY DESIGN (`QuantumPSO.m:91` reseeds from
wall clock; no fixed seed anywhere). The port routes every stochastic draw
through one injected numpy `Generator` with a fixed per-run seed, preserving
each operator's draw structure and order. Bit-parity with legacy optimizer
RUNS is impossible in principle and is NOT the target; IMPROVEMENTS'd.

Exit gate: `drivers/train.py` runs the full loop (QuantumPSO outer at tiny
params, SMORMS3 inner, real engine in-process via the seam) on the committed
4b twin corpus with a fixed seed, end-to-end, checkpoint artifacts produced,
deterministically reproducible twice; AND the seam replays the committed
4a/4b train fixtures with `results_matrix` + final weights matching the
Rust-native goldens (canary-gated).

## S2. Architecture

Per the brainstorm (approved): coarse corpus-level seam (the legacy loop
reran the whole engine per SMORMS3 step, so coarse IS the faithful hot
loop); `engine.py::forward_backward(config, weights, batch) -> (cost,
gradient)` implements `ComputeGradient`'s contract on top of `Engine.run`;
TOML flattens to the legacy `IndexMap` (one semantics); `genome.py` is its
own module (file-size discipline); pydantic `RunState` replaces `PS`.
The `[sad, lid]` two-element list is the numpy pairing contract for algo 6.
GIL released during `run()`. The engine's own determinism (static lanes,
N=1 parity mode) is orthogonal and untouched.

## S3. Oracle: the Octave harness

`tools/octave_harness/` - per-stage `.m` scripts calling the VENDORED
`legacy/Optimizer_V6.2.2` sources (git-ignored, local-only; `brew install
octave` documented like libmatio; CI consumes committed fixtures only),
driven by `scripts/extract_phase4c_fixtures.py` (octave-cli, .mat->.bin
conversion, manifest MEASURED values, twice-regeneration, prior-phase hash
guard). Octave-vs-MATLAB compat quirks are absorbed per-fixture and
adjudicated in IMPROVEMENTS (the .m source is the contract, Octave the
executor).

Stages: SMORMS3 state trajectories (fixed gradient sequences; MMS/stepRate/
delta/lrate per step, bit-exact); Rprop.m sequences; vec2struct round-trips
(crafted genome vectors + masks over the committed configs, both directions
+ the out_param inverse + weight blocks); ComputeCost assembly fed the
COMMITTED 4a/4b MultiConfigResults fixtures + crafted per-balance-law
variants (0/3/4/5/10, L2, pooled stats); GetNewBatch rotation traces;
MaskingValidation pass/fail; CheckGrad's central-difference assembly with an
injected quadratic surrogate cost (identical on both sides - the real
CostFunction shells to the engine, which Octave cannot run); QuantumPSO for
a few iterations at small D with the SHARED RANDOM TABLE (a committed .bin
both sides consume, draw-order-faithful incl. the dead velocity banks) + the
surrogate cost - position/pbest/gbest trajectories bit-exact given identical
randomness.

## S4. Seam integration tests + exit gate

pytest (maturin-built; the CI `python-pyo3` job gains this subset):
`Engine` replays the committed 4a tier-2 + 4b twin configs end-to-end,
`results_matrix` + final weights vs the committed goldens (canary-gated via
`tests/_libm_gate.py`); `grad_check` through the seam vs the committed
gradCheck goldens; TOML-vs-.config engine-output bit-identity; same-seed
determinism (two runs, identical bits); hypothesis pack/unpack round-trips.
Exit-gate test per S1.

## S5. Mutation battery (4c flavor)

Each must break a named golden: SMORMS3 warmup x10 -> x2; an eps placement
moved; the theta_out round-trip dropped; vec2struct mask-inverse dropped;
`deriv/max(1,count)` -> `/count`; L2 bias-exclusion flipped; QPSO
contraction schedule inverted (0.25->0.75); batching rotation off-by-one.

## S6. IMPROVEMENTS.md candidates

Fixed-seed determinism deviation; balance 6-9 deferral; unwired-optimizer
exclusions (Adam/RMSprop/sfo/pso_Trelea); the SMORMS3 file's mislabeled
SFO header; the Rprop.m dead rand at `:5`; QPSO dead velocity banks that
still consume the stream; every Octave-compat adjudication; the two 4b
follow-ups CLOSED in this phase's harness-touching tasks (`cost_max_abs`
delta semantics; mode-7 dump filename legacy-faithful rename now that
`audio_file_name` exists).

## S7. Risks

- R1: Octave cannot run some vendored .m construct (classdef handle,
  sortrows variants) - absorb per-fixture: smallest-possible stage rewrite
  on the HARNESS side only, adjudicated + documented; the Python port
  always follows the .m source.
- R2: PyO3/numpy zero-copy vs the pairing contract - copies are acceptable
  (seconds-per-run amortizes); document where copies happen.
- R3: vec2struct's genome length (ncoef) depends on config fields walked -
  the count must match the legacy EXACTLY or every mask/genome index
  shifts; pinned by round-trip goldens on multiple configs incl. algo 6.
- R4: the alpha-stable Levy sampler - CMS transform must match stblrnd's
  parameterization (alpha=1.3, beta=1, gamma=0.5, delta=0); pinned via the
  shared-table QPSO trajectories.
- R5: `results_matrix` promotion exposes what save_results wrote - the
  timing column stays wall-clock (masked in tests, documented for Python
  consumers).

## S8. Process

superpowers loop: this spec -> writing-plans
(`docs/superpowers/plans/2026-07-08-phase-4c-pyo3-optimizer.md`) ->
subagent-driven development (fresh implementer + reviewer per task; briefs
state LEGACY SOURCE GOVERNS; FOREGROUND test runs; mutation batteries) ->
docs -> smart-commit (whole branch) -> finishing-a-development-branch (user
merges via PR).
