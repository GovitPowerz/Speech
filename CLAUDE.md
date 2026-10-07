# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech is a Speech Activity Detection (SAD) and spoken Language Identification (LID) engine with its training loop. It began as a port of the legacy `FastSpeechProcessing` system (a C++ forward+gradient engine of ~21.5k LOC plus a 200k-line constants table, and a MATLAB optimizer of 110 `.m` files): the C++ engine became the **Rust** crate + binary `speech`, the MATLAB optimizer the **Python** package `speech`, and the legacy file/subprocess seam became an in-process **PyO3** API (`speech-py` -> module `speech_rs`), with the legacy formats (flat `.config`, little-endian `.bin` weight vectors, listing CSVs, `.mat` results) kept as a byte-compatible serialization layer. The flat-weight layout and `adim_coeff` scaling live once, in Rust.

**Roadmap 1 (the port) is closed and tagged `legacy-parity-v1`**: the port matches the resurrected 2015 `-ffast-math` production binary (VRCTS byte-identical, scored result columns within ~1e-15), and every rung below it (config, codecs, features, NN forward, BPTT, iRPROP-, segmentation, scoring, drivers, LID, the seam, dataprep) is golden-tested against a legacy oracle. **Roadmap 2 (phases 5-11) is also done**: goldens flipped to port-truth with eleven legacy bugs fixed under a protocol, a from-scratch training loop with real-data baselines (subset-proven; the full-corpus runs are user-fired and land in `RESULTS.md`), a runtime-selectable f32 fast tree, a bit-identical streaming mode, and four new recurrent cells (sLSTM, Mamba, CfC, a windowed transformer) behind one seam, each with both gradient tiers and both f32 twins. No stub remains anywhere in `src/`. The per-phase record is `docs/ROADMAP.md`; the numbers are `RESULTS.md`.

The ultimate goal (user, 2026-07-10): a very efficient SAD + LID Rust binary in both offline and online (streaming) modes, then newer architectures, trainable from scratch to build a strong baseline to improve on.

## Where things live

| question | file |
|---|---|
| How a run and a training step flow through the code | `docs/ARCHITECTURE.md` |
| The vocabulary (and the synonyms to avoid) | `CONTEXT.md` |
| The decisions that constrain new work | `docs/adr/` (9 ADRs) |
| What each phase delivered, proved, and left named | `docs/ROADMAP.md`; specs and plans indexed in `docs/superpowers/README.md` |
| Every measured number, bench table, gate and mutation battery | `RESULTS.md`; its ledger-owned tables are rendered from `ledger/` (ADR-0009) |
| Every reproduced legacy quirk, kept or fixed, and the fix protocol | `IMPROVEMENTS.md` |
| The oracle tiers and the standing gates, in full | `docs/validation.md` |
| Per-module responsibilities and legacy sources, row by row | `src/rust/README.md`, `src/python/speech/README.md` |
| Who decides what, and what evidence a change needs | `DEVELOPMENT.md` |

## Build & Development Commands

```bash
# -- Rust engine --
cd src/rust
cargo build --release              # Build optimized binary `speech`
cargo test                         # Run unit + integration tests
# Run from repo root:
./src/rust/target/release/speech -m configs/lid/lid_blstm.toml

# -- PyO3 Bindings --
# Builds + installs the speech_rs module from the speech-py crate:
uv run maturin develop --release --manifest-path src/rust/speech-py/Cargo.toml

# -- Python optimizer/orchestrator --
uv sync                            # Install dependencies (Python >=3.14)
uv sync --group dev                # Include dev tools (pytest, ruff, mypy, hypothesis, maturin)
uv run pytest tests                # Run all tests
uv run pytest tests/test_smoke.py::test_package_imports -v

# -- Utility Scripts (from repo root) --
./build.sh                         # Build Rust binary + PyO3 bindings
./setup_env.sh                     # Create fresh .venv + uv sync + PYTHONPATH
./lint_code.sh                     # ruff (imports, format, check) + mypy
./check_all.sh                     # Rust: test + fmt --check + clippy + release build
./upgrade_dependencies.sh          # uv sync --upgrade

# -- Measurement ledger (ADR-0009) --
uv run python -m speech.ledger add runs/<run>/record.json   # validate + promote a run's record, then render
uv run python -m speech.ledger render --check               # RESULTS.md holds what ledger/ renders (CI)
uv run pytest tests/pyo3/test_phase10_gates.py --ledger-stage /tmp/stage   # gates copy their records out of tmp_path
./src/rust/target/release/speech bench --json --label=<name> --path=fast <config>   # the bench payload
uv run python -m speech.ledger bench --leg phase7_60s                   # a staged Phase 7 leg: 3 fresh processes x {exact, fast}, promoted + rendered
```

Suite sizes at the Phase 11 close: cargo 1283, pytest 896 non-slow (`uv run pytest tests -q -m "not slow"`), pyo3 203 (`uv run pytest tests/pyo3 -q`, corpus present, ~20 min).

## Architecture

The Rust crate (`src/rust/`) is the engine: a Cargo workspace with the core `speech` crate (lib + bin) and the `speech-py` PyO3 binding crate (module `speech_rs`). TOML is the canonical config; a legacy `.config` importer exists for validation runs against goldens. The Python package (`src/python/speech/`) is the optimizer and orchestrator, driving the engine in-process via PyO3.

Every forward sits on three axes, each pair held together by a gate (`docs/ARCHITECTURE.md`): **precision** (the behaviour-frozen exact f64 tree `nn/`/`features/`/`tasks/` vs the f32 `fast/` tree, selected by `Inference_Path` at one dispatch site), **timing** (offline vs the streamed `fast/stream.rs` / `fast/stream_lid.rs` sessions), and **cell x direction** (`{lstm, slstm, mamba, cfc, transformer}` x `{bidirectional, forward}` behind the closed `CellLayer` enum in `nn/cells/`, with f32 twins in `fast/cells.rs` and `fast/bicell.rs`). Only the exact tree trains; only a forward net streams; the fast matrix is total and compile-enforced.

The module maps (one row per file: legacy source, responsibility, what landed when, every reproduced quirk) are `src/rust/README.md` and `src/python/speech/README.md`. Read the row before touching a module; the quirks recorded there are load-bearing for the goldens.

## Key Decisions & Pitfalls

### Locked decisions

- **Naming**: Rust core crate `speech` (lib + bin), PyO3 crate `speech-py` -> module `speech_rs`, Python package `speech`, CLI binary `speech`. Mirrors the sibling `aerocapture`/`aerocapture-py`/`aerocapture_rs`.
- **Legacy source**: vendored source-only under `legacy/` (C++ `src/` + `Optimizer_V6.2.2` `.m`, no data/figures/executables). `legacy/` is **git-ignored** -- present locally as a golden oracle, not committed.
- **Config format** (ADR-0006): TOML is canonical (`toml_config.rs`, a 190-row bidirectional `KEY_TABLE`; both directions read the same table so encode/decode cannot drift; an unmapped key round-trips through `[legacy.raw]`). The legacy importer (`legacy_config.rs`) is retained as the validation path and `.config` is accepted everywhere `.toml` is. Port-only keys (`Inference_Path`, `Audio_fixed_gain`, `BLSTM{,_LID}_Cell_Type`/`_Direction`, the unprefixed `Mamba_*`/`Cfc_*`/`Transformer_*` geometry) default to the legacy shape, so every older config decodes byte-unchanged. Gain-inertness finding: on an all-DCT / `IgnoreFirstDCT` config any positive `Audio_fixed_gain` yields the same posterior (the AC DCT basis and the deltas annihilate a uniform log-shift); the gain is decision-relevant only for raw-band / mel-input configs.
- **Two SAD lineages** (ADR-0008): `lre_sad.toml` (v1) is the only 2015-capacity-comparable one; `lre_sad_v2.toml` differs in exactly two value lines and removes v1's 48 structurally dead layer-0 input columns. Live parameter counts are identical per cell. Quote the lineage with every SAD number; keep v1's gates frozen when touching v2's.
- **Roadmap order**: lead with the pure-logic + I/O parity layer, golden-tested bit-for-bit before any NN. Both roadmaps are closed; the Phase 0a packer seam was closed for real in Phase 4d (two real 2015 packs repack byte-identical, mutation-probed since no independent structured/`.bin` pair exists to drive a classic RED).

### Parity hazards (port carefully, golden-test each)

- **Weight-vector flat layout ordering**: input/recurrent gate blocks, then the 12-row peephole bundle, then 4 biases, then the mean/std tail. Get the ordering wrong and every downstream number is garbage.
- **`adim_coeff` scaling**: `adim_coeff = sqrt(fan_in*sub + fan_out)`, bias-excepted. In the legacy this scaling lives **only in the config reader**, NOT in the flat packer -- so a naive round-trip through the packer drops it. In this port it must live once in Rust.
- **CellWeight-narrow layout**: unlike the Input/Forget/Output gate matrices, the Cell matrix carries no peepholes -- its rows are `out + in + 1` (recurrent + fan-in + bias only), one column-triplet shorter than the gate matrices. Easy to over-generalize and get wrong.
- **Activations**: non-standard. `Maxmin2`/`Identity` are `asinh` (cell + output), `GatesFunction` is `sigmoid(0.1z)` (note the 0.1 pre-scale), output is softmax.
- **iRPROP- inverted sign convention**: the legacy update uses an inverted sign; copy it exactly, do not "fix" it.
- **FFT two-real packing + custom Welch normalization**: two real signals packed into one complex FFT, with a bespoke normalization -- not the textbook periodogram.
- **Non-standard LTSV**: score is `-sum r(r-1)`, and it uses variance, NOT its square root.
- **Fixed 100000-entry random tables**: `Constants.h` bakes in two 100000-entry random tables for determinism; reproduce them (and the modular indexing) faithfully.
- **Little-endian assumption**: `.bin` and `.mat` are little-endian; the codec hard-assumes LE.
- **Layer naming off-by-one**: MATLAB is 1-based, the config `Layer_%d` keys are 0-based. Mind the offset when bridging.
- **In-band signaling offsets**: `score > 150` and `val - 200` encode signals in-band in the legacy numeric streams; these magic offsets are load-bearing, not noise.
- **The peephole bundle is mixed-kind**: rows 0-2 peep cell states, rows 3-11 peep gate values, several from the previous step. A streaming kernel that carries only `h`/`c` computes a correct whole-sequence output and a wrong streamed one.
- **A declared input width is not a produced one**: `NNetInputSize` must equal `3*nb_dct - ignore_first_dct` (the delta keys are regression orders, not counts); v1 inherits a 23-vs-11 mismatch from 2015 and the dead columns are pinned, not fixed (ADR-0008).

### Scaffolding-phase relaxations (CLOSED as of Phase 4d)

Historical note: the repo started as all-stubs with a crate-root `#![allow(dead_code, unused_variables)]` (removed in Phase 0a) and `allow_empty_bodies = true` in the mypy config (removed in Phase 4d Task 15). Any narrow `#[allow(...)]` left is on a specific item, not crate-wide; no module in `src/` has a stub body.

### Legacy quirks & deferred fixes

Legacy bugs and oddities reproduced on purpose are tracked in **[IMPROVEMENTS.md](IMPROVEMENTS.md)**, each kept or fixed. Since Phase 5 the rule is port-truth (ADR-0001): a correctness-critical entry is fixed under the "Phase 5 fix protocol" preamble (RED -> re-pin -> mutation -> flip the entry to `FIXED (phase, F<n>, commit)`), a kept entry states why. Every phase that reproduces a legacy quirk adds an entry; every fix flips one.

## Conventions

- **Rust**: Edition 2024, `ndarray` on the exact path + `faer` (pure-Rust SIMD, no system BLAS) on the fast path, release profile with LTO. Deps: `realfft` (re-exports `rustfft`; the exact FFT is hand-rolled), `symphonia`, `rayon`, `byteorder`/`bytemuck`, `quick-xml`, `serde`/`toml`, `libc`; `criterion` dev-dep for `benches/kernels.rs`. `io/matfile.rs` is a hand-rolled v5 writer with no reader. `speech-py` uses `pyo3` + `numpy`.
- **Python**: Python >=3.14, ruff (line-length 160, target py314; select E,F,I,W,UP,B,SIM), mypy (`disallow_untyped_defs` + `warn_return_any`, not `--strict`), uv. Dev tools in `[dependency-groups]`. Core deps: numpy, scipy, pandas, pyarrow, pydantic, matplotlib, rich, soundfile, cma.
- **Testing (Python)**: pytest + hypothesis. Goldens under `tests/reference_data/`, added per phase. Round-trip property tests are mandatory for the weight bridge.
- **Testing (Rust)**: inline unit tests + proptest, integration tests in `src/rust/tests/`. A `test-support` Cargo feature exposes observation hooks with nothing production-reachable. Validation is golden-file bit-for-bit against the git-ignored `legacy/` oracle via the LOCAL-ONLY harness family (`tools/oracle_harness/` strict-IEEE C++ incl. the real compiled NN/segmenter/LID translation units; `tools/octave_harness/`; `tools/perl_oracle/`; `tools/fsp_runtime/`, the resurrected 2015 binary); CI consumes only committed fixtures. Bit-exactness holds on the oracle environment (Apple libm, canaries in `tests/reference_data/phase1/libm_canaries.bin`); elsewhere transcendental-dependent goldens assert a hybrid bound (`<= 4` ULP or `512*eps` scaled to the fixture), canary-gated, and NaN elements compare by `is_nan` (the quiet-NaN sign bit is architecture-defined). Mutation standard: a load-bearing quirk is paired with a "revert it, the golden must fail" check. The full tier-by-tier history is `docs/validation.md`.
- **Standing rules, by the phase that introduced them** (the full record with its evidence is `docs/validation.md`; the decisions are the ADRs):
  - **Since Phase 5**: goldens assert PORT-TRUTH. A legacy bug is fixed under the fix protocol, never silently; the oracle harnesses are never regenerated for a fixed site (ADR-0001).
  - **Since Phase 6**: real-data tests are CORPUS-GATED and local-only (`requires_corpus` / `corpus_root_or_skip`; CI runs synthetic fixtures only). LICENSE HYGIENE is a hard rule: no corpus path, filename or feature table is committed; fixtures use synthetic pattern-preserving names. The rule is executable for paths: `tests/test_license_hygiene.py` fails on any tracked text file naming a file by a path through the corpus root directory (directory globs pass; bare basenames, binaries and the 2015 host-prefixed paths stay review-only).
  - **Since Phase 7**: two inference trees. The exact tree is frozen (five sanctioned behaviour-free touch classes, the golden suite byte-green as the proof); fast-path divergence is BY DESIGN and lives in `fast/` module docs + `RESULTS.md`, NEVER `IMPROVEMENTS.md`. Parity gates assert decision IDENTITY (the SAD tier owns boundary count/type/`max_dt == 0.0`; the LID tier alone owns argmax zero-flips) and pin numbers measure-then-pin at measured*10; a breach is a STOP, never a widening (ADR-0002, ADR-0003).
  - **Since Phase 8**: streaming is bit-equal to the offline run under the frozen norm, chunk-invariant, with zero retractions; the emission trigger is the CONSUMED frontier (never the received one) clamped by the last raw boundary and the hysteresis's `pending_begin`. Training and bidirectional streaming are out by construction (ADR-0004).
  - **Since Phase 9**: `CellLayer` is the extension seam -- a new cell is a variant, a `Layer` impl inside `cells/`, an `init_weights.py` builder, its `KEY_TABLE` rows and both f32 twins; a diff to `nn/` outside `cells/` must be justified against the touch classes. THE BATTERY LESSON: self-consistent legs (split-state, chunk-invariance, the streaming gate) compare two runs of ONE kernel and pin state threading only; parity legs own arithmetic only at their pin's resolution; a SYMMETRIC kernel substitution inside a shared step is owned by nothing but the design-time rule in that module's doc (per-step `dot_f32`, no batched projection inside a step), so review the rule (ADR-0005).
  - **Since Phase 10**: the (cell x direction) fast matrix is TOTAL (`classify_fast_shape` returns no `Result`; a kernel-less cell is a compile error at `cell_weight_count`). A new cell lands both twins, or a documented test-pinned bail, in the phase it lands its exact cell. Two SAD lineages, not interchangeable (ADR-0008).
  - **Since the ledger (issue #20, ADR-0009)**: a measured number is a RECORD under `ledger/<kind>/<id>.json` (recipe = identity, SHA/build/host = provenance), promoted only through `python -m speech.ledger add` (schema + hygiene validator, no dirty tree without `--allow-dirty`, release builds only, never a resumed run, one live record per (source, recipe) unless `--supersede REASON`, a bench label naming exactly one (lanes, lineage)); the RESULTS.md tables between `<!-- ledger:table <name> -->` markers are RENDERED, never hand-edited (`render --check` in CI); a `TBD` row is filled by appending a record; supersession is append-only with a reason. The record is not the pin: pins stay in the tests. Historic numbers were re-run, not transcribed; a re-run metric that moves is a STOP. A schema bump migrates the committed records through a script in the same commit and `load` reads one version (issue #38). The four Phase 7 bench legs are staged from the repo (`speech.ledger.stage`) and run by `python -m speech.ledger bench --leg`; the README / ARCHITECTURE sentences quoting the speedups and the SAD peak RSS are asserted against the latest bench pair (`speech.ledger.prose`), so a prose number changes only with a record.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) -- Rust (fmt, clippy, test), Python (ruff lint, ruff format, mypy, pytest), and PyO3 (maturin build + targeted pytest) run on PRs to `main` and manual dispatch.

## Tone

Be a **quirky friendly but critical peer reviewer**. Think of yourself as a quirky senior developer doing a code review: helpful, but holding me to high standards. Always **challenge inefficiencies**: if I'm doing something the hard way, call it out. Given the bit-exact-against-legacy target, treat every parity hazard above as a place where "close enough" is a bug.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, operated via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: the five canonical role names are the label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`); categories are `bug` / `enhancement`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the repo root + ADRs in `docs/adr/`. See `docs/agents/domain.md`.
