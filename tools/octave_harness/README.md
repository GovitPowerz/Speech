# Octave harness (Phase 4c)

The MATLAB-side analog of the C++ `tools/oracle_harness/`: GNU Octave executes the
**vendored** legacy `.m` sources (`legacy/Optimizer_V6.2.2/functions/*.m`) to dump golden
per-step optimizer state trajectories. The Rust/Python port is then bit-pinned against
those dumps.

## Local-only dependency

GNU Octave is a **local-only** developer dependency, exactly like `libmatio` for the C++
oracle harness. Install it once:

```bash
brew install octave        # 11.3.0 bottled at time of writing
```

CI never installs Octave and never runs this harness. CI consumes only the committed
`tests/reference_data/phase4c/*.bin` + `manifest.json` fixtures the extractor produces.
`legacy/` is git-ignored (source-only, present locally as the oracle); the harness `.m`
here `addpath` the vendored sources and **call** them, never modify them.

## Layout

| File | Role |
|------|------|
| `run_stage.m` | Dispatcher: `run_stage(stage, out_dir)` -- adds this dir + the vendored `functions/` dir to the path, then runs the named stage into `out_dir`. |
| `stage_smorms3.m` | Drives the REAL `SMORMS3.m` classdef over scripted gradient sequences; dumps per-step `theta/MMS/stepRate/delta/lrate` + the round-trip variant. |
| `stage_rprop.m` | Drives the REAL `Rprop.m` function over a crafted derivative/cost sequence; dumps per-step `weights/delta/deltaweight/derivatives`. |
| `smorms3_f_df.m` | The injected SMORMS3 objective (scripted gradient + `theta_out` round-trip offset). |
| `stage_vec2struct.m` | Drives the REAL `vec2struct.m` (+ `printConfig.m`) over six algo/mask/PS cases; dumps the genome<->config bijection goldens. |
| `stage_computecost.m` | **FALLBACK-TIER** transcription of the `ComputeCost.m` cost-assembly lines (`:285-652`); drives the balance-law 0/3/4/5/10 error + cost, deriv averaging, pooled stats, and L2 over the committed 4a/4b MultiConfigResults fixtures + crafted variants. |
| `stage_qpso.m` + `qpso_modified/` | **SUBSTITUTION-COPY TIER** (Phase 4c Task 11): drives a MODIFIED-COPY of `QuantumPSO.m` (`qpso_modified/QuantumPSO.m`) whose rand/randperm/stblrnd are substituted by reads of a shared committed random table (`tbl_rand`/`tbl_randperm`/`tbl_stblrnd`, a stage-global cursor) and whose `CostFunction` is the quadratic `qpso_surrogate.m`; the wall-clock reseed (`:91`) is removed. Every substitution is a `% HARNESS-SUB` comment quoting the original line. Dumps the post-init state + per-epoch pos/pbest/gbest trajectory + a standalone Levy sample; the Python `speech.optimizers.quantum_pso` replays the SAME table (`tests/reference_data/phase4c/qpso_random_table.bin`). |
| `stage_writelisting.m` (Phase 4d Task 8) | **TIER 1**: drives the REAL, unmodified `WriteListing.m`/`WriteWeightedListing.m` -- 7-file/3-worker `fliplr` shard interleave (plain) plus 8 crafted `%g`-boundary values (weighted). Both write plain text directly (no `.mat` conversion step, unlike every other stage). |
| `stage_scr.m` (Phase 4d Task 11) | **HYBRID TIER**: `PS.Corpora.Train.langMapConf`/`.listing` come from a REAL Tier-1 call to `processListing.m` (so the alphabetical class-key order is never hand-simulated); `Test_BLSTM.m:251-269`'s `.scr`-writer loop body is FALLBACK-TIER transcription (`Test_BLSTM.m` is a top-level script wired to a `CostFunction.m` engine shell-out, no callable function boundary exists), with every builtin/vendored call inside the loop (`textscan`, `sortrows`, `keys`, `num2str`, `fprintf`) real and unmodified. |

Run a stage by hand:

```bash
octave-cli --no-gui --quiet --path tools/octave_harness \
  --eval "run_stage('smorms3', '/tmp/out')"
```

The extractors (`scripts/extract_phase4c_fixtures.py` for smorms3/rprop,
`scripts/extract_phase4c_genome_fixtures.py` for vec2struct,
`scripts/extract_phase4c_computecost_fixtures.py` for computecost) run the stages into a
tempdir, convert the `.mat` dumps to `.bin` (dropping the MAT-v7 timestamp header, so
re-runs are byte-identical), validate shapes/step-counts/non-vacuity, hash-guard the
prior-phase (and prior-phase4c) fixtures, and commit the `.bin` + `manifest.json`.

## Fidelity tiers

The smorms3/rprop/vec2struct stages are **TIER 1**: the real vendored classdef/function is
exercised unchanged (real ctor, real `optimization_step`, real `f_df_wrapper`, real update
math for SMORMS3; the real function for Rprop; the real `vec2struct`/`printConfig`).

`stage_computecost.m` is the **FALLBACK TIER** (the MATLAB analog of the C++ harness's
"reimpl-swap"): `ComputeCost.m`'s top half shells out to the engine
(`system('python RunFsp.py ...')`, `:173-191`) after a config-write loop and then LOADS the
worker `.mat`/`.bin` the shell-out wrote, and its `!`-escape cleaning (`:37`) deletes any
pre-injected worker files before the shell-out -- so there is no injection point that leaves
the vendored `.m` unmodified. The stage transcribes only the PURE assembly lines
(`:285-652`) line-for-line with `% legacy:` provenance and lets Octave execute the real
MATLAB builtins (median/hist/std/cumsum/exp/log). See `IMPROVEMENTS.md` (phase4c
ComputeCost fallback-tier entry).

`stage_qpso.m` is the **SUBSTITUTION-COPY TIER**: `QuantumPSO.m:91` reseeds `rand` from the
wall clock, so no legacy RUN is reproducible and there is no injection point that leaves the
`.m` unmodified while making it deterministic. `qpso_modified/QuantumPSO.m` is a near-verbatim
copy of `QuantumPSO.m:252-843` with ONLY the RNG and cost calls substituted (every one a
`% HARNESS-SUB` comment quoting the original), so the algorithm structure -- init + OBL, the
dead-but-drawing velocity banks, the QDPSO ranking-operator roulette, DE + Levy, pbest/gbest,
stall, final generation -- runs unchanged over the shared table. The parity target is the
port's OPERATOR STRUCTURE, not a wall-clock run; the copy shadows the vendored `QuantumPSO.m`
via `addpath(...,'-begin')`. See `IMPROVEMENTS.md` (the phase4c determinism/dead-banks/
substitution-adjudication entries).

The **one** Octave-compat accommodation is `stage_smorms3.m` passing a single dummy
`varargin` to `SMORMS3(...)`: Octave 11.3.0 miscounts the empty-cell lvalue cs-list at
`SMORMS3.m:313` (`[f, df, theta_out, obj.varargin_stored{:}] = obj.f_df(...)`) and raises
"function called with too many outputs" when `varargin_stored` is `{}`. Passing one dummy
varargin makes the expansion well-defined; it is value-neutral (the gradient depends only
on `eval_count`; the update math never reads the varargin). The `.m` source is untouched --
it stays the contract, Octave is only the executor. See `IMPROVEMENTS.md` (phase4c
Octave-compat entry).
