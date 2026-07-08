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

Run a stage by hand:

```bash
octave-cli --no-gui --quiet --path tools/octave_harness \
  --eval "run_stage('smorms3', '/tmp/out')"
```

The extractor `scripts/extract_phase4c_fixtures.py` runs the stages into a tempdir,
converts the `.mat` dumps to `.bin` (dropping the MAT-v7 timestamp header, so re-runs are
byte-identical), validates shapes/step-counts/non-vacuity, hash-guards the prior-phase
fixtures, and commits the `.bin` + `manifest.json`.

## Fidelity tiers

Both current stages are **TIER 1**: the real vendored classdef/function is exercised
unchanged (real ctor, real `optimization_step`, real `f_df_wrapper`, real update math for
SMORMS3; the real function for Rprop). The C++ harness's "reimpl-swap" fallback tier has
no analog here yet.

The **one** Octave-compat accommodation is `stage_smorms3.m` passing a single dummy
`varargin` to `SMORMS3(...)`: Octave 11.3.0 miscounts the empty-cell lvalue cs-list at
`SMORMS3.m:313` (`[f, df, theta_out, obj.varargin_stored{:}] = obj.f_df(...)`) and raises
"function called with too many outputs" when `varargin_stored` is `{}`. Passing one dummy
varargin makes the expansion well-defined; it is value-neutral (the gradient depends only
on `eval_count`; the update math never reads the varargin). The `.m` source is untouched --
it stays the contract, Octave is only the executor. See `IMPROVEMENTS.md` (phase4c
Octave-compat entry).
