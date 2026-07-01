# Speech Repo Setup — Design Spec

Date: 2026-07-01
Status: approved (design), pending implementation plan

## 1. Goal and scope

Stand up the `Speech` repository as the home for porting the legacy `FastSpeechProcessing`
system (C++ engine -> Rust, MATLAB optimizer -> Python), mirroring the structure and
conventions of the sibling `Aerocapture` project.

**This deliverable is bounded to two things:**

1. A **green-compiling development environment**: the full module tree wired and building,
   with `cargo build` + `cargo clippy -D warnings` clean, `cargo test` green on trivial
   tests, `pytest` green, and `ruff` + `mypy --strict` clean. Modules are typed stubs with
   doc-comments and signatures that match the target architecture — navigable scaffolding,
   **no ported algorithm logic**.
2. **`CLAUDE.md`** and **`README.md`** in the Aerocapture style, describing the real system
   and the phased port roadmap.

**Explicitly out of scope:** porting any actual algorithm (features, NN forward/backward,
optimizers, scoring, decision laws). Each becomes a roadmap phase with its own spec -> plan.

## 2. Source of truth

The legacy system was mapped by an 8-agent analysis pass (see session history). Summary:

- **C++ `src/` (~21.5k LOC logic + a 200k-line `Constants.h` data table) -> Rust.** A fast
  forward+gradient engine: audio I/O (libsndfile/ogg/vorbis/FLAC), acoustic features
  (radix-2 FFT periodogram with two-real packing, triangular Mel filterbank with 1125/700
  scaling, DCT-II MFCCs + deltas/SDC, LTSV, time-domain pitch/autocorrelation), from-scratch
  bidirectional recurrent nets (peephole BLSTM, plus SRN and Clockwork-RNN variants) + output
  MLP, SAD segmentation (hysteresis-with-area double threshold + 7-step smoothing), and
  TwinBLSTM LID (LID net consumes the SAD net's hidden states). Also computes analytic BPTT
  gradients and an in-engine iRPROP- update. OpenMP parallel-for over a corpus. Binary is
  `fsp`.
- **MATLAB `Optimizer_V6.2.2` (110 `.m` files) -> Python.** The outer optimization loop and
  data bookkeeping: global search over DSP hyperparameters (bespoke QuantumPSO, or CMA-ES) +
  gradient training of NN weights (SMORMS3 wired; Adam/RMSprop/RProp/SFO present),
  cross-entropy losses (SAD + LID), calibration cost laws, EER-cutoff confusion scoring,
  hard-example mini-batching, per-language net combination.
- **The seam (today):** files — a flat `.config`, little-endian `.bin` weight/gradient
  vectors, `fileslisting`/`language2classmapping` CSVs, `.mat` results — with the engine
  launched per evaluation via `RunFsp.py`.

## 3. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Naming | Rust core crate `speech` (lib + bin), PyO3 crate `speech-py` -> module `speech_rs`, Python package `speech`, CLI binary `speech`. Mirrors `aerocapture`/`aerocapture-py`/`aerocapture_rs`. |
| 2 | Legacy source | Vendored **source-only** under `legacy/` (C++ `src/` + `Optimizer_V6.2.2` `.m`, no data/figures/executables). `legacy/` is **git-ignored** — present locally as a golden oracle, not committed. |
| 3 | Config format | **TOML-first** (Aerocapture philosophy: config is the only input), plus a legacy `.config` compatibility importer for validation runs against goldens. |
| 4 | Roadmap order | Lead with the **pure-logic + I/O parity layer** (config parser, weight `.bin` codec, segmentation/smoothing/scoring, feature extraction), golden-tested bit-for-bit before any NN. |

Additional design stances carried from the analysis:

- **Interface:** replace the file/subprocess seam with an in-process PyO3 API for the hot
  training loop, while keeping the legacy `.config`/`.bin`/`.mat` formats as a byte-compatible
  serialization layer for validation and cluster drop-in. The flat-weight layout + `adim_coeff`
  scaling live **once** in Rust (`io::binary` + `nn`), exposed to Python via PyO3 — a single
  source of truth, not duplicated across two languages as in the legacy.
- **Parity target:** bit-exact-where-feasible against the legacy (Aerocapture precedent).
  Faithful port of the fixed 100000-entry random tables (`Constants.h`) and the `fmath`
  fast-log/exp is a documented, deferrable decision made per-module during the port, not now.
- **Optimizers:** QuantumPSO is hand-ported (bespoke ~900-line search), not swapped for pymoo.
  CMA-ES may use `cma`/`pycma`.

## 4. Repository layout

```
Speech/
  pyproject.toml  uv.lock
  setup_env.sh  build.sh  check_all.sh  lint_code.sh  upgrade_dependencies.sh
  .github/workflows/ci.yml
  .zed/debug.json   speech.code-workspace
  .gitignore        CLAUDE.md   README.md
  configs/{sad,lid,training}/*.toml        # TOML canonical
  configs/legacy/                          # sample legacy .config for the importer
  data/                                    # random_tables.bin (from Constants.h), language maps, fixtures
  output/.gitkeep
  legacy/                                  # git-ignored: C++ src/ + Optimizer_V6.2.2 .m
  docs/superpowers/specs/                  # this spec + future phase specs
  src/rust/
    Cargo.toml                             # workspace: speech (lib+bin) + speech-py
    Cargo.lock
    src/
      lib.rs  main.rs  config.rs  legacy_config.rs  constants.rs
      engine/{mod,corpus_processor,bag_of_processors,corpus}.rs
      audio.rs
      features/{mod,fft,mel,ltsv_tdc,stats}.rs
      nn/{mod,network,blstm,layers,activations,train}.rs
      cost.rs
      tasks/{mod,segmenter,sad,lid,segmentation_io,vrcts}.rs
      io/{mod,binary,matfile}.rs
    tests/                                 # cargo integration + e2e skeleton
    speech-py/
      Cargo.toml  pyproject.toml
      src/lib.rs                           # PyO3 module speech_rs
  src/python/speech/
    __init__.py  cli.py  config_bridge.py  weight_bridge.py  engine.py
    optimizers.py  scoring.py  batching.py  nn_reference.py
    drivers/{__init__,init,train,retrain,test}.py
    dataprep/{__init__,augment,opensad15,stm_normalize}.py
  tests/
    __init__.py  conftest.py  test_smoke.py
    reference_data/                        # golden fixtures (added per phase)
```

## 5. Rust module map (stub-level provenance)

Each module ships as a compiling stub: module doc-comment stating its legacy source and
responsibility, key public types/signatures with `todo!()` or trivial bodies, warning-clean.

| Rust module | Legacy source | Responsibility |
|-------------|---------------|----------------|
| `lib.rs` / `main.rs` | `FastSpeechProcessing.cpp`, `CorpusProcessor.cpp` | Crate root + engine API (`run_solo`/`run_train`/`run_grad_check`); CLI mirroring `fsp` modes (`-s/-i/-t/-m` + `--key=val`) as a typed `Mode` enum. |
| `config.rs` | `ConfigFile.*`, `String.hpp` | TOML canonical config (serde) + typed getters. Highest-priority exact-match target. |
| `legacy_config.rs` | `ConfigFile.*` | Importer for the legacy whitespace `.config` (`_`-rest-of-line continuation, last-wins, `val*count` repeat). |
| `constants.rs` | `Constants.h` | Loads two 100000-entry tables (`_RandomGaussVector`, `_RandomVector`) from `data/random_tables.bin` via `include_bytes!`; modular indexing preserved. |
| `engine/corpus_processor.rs` | `CorpusProcessor.*` | Driver: build corpus + processors, rayon `par_iter` over files (replaces OpenMP), epoch training loop, finite-diff grad check, result reduction + `.mat` write. |
| `engine/bag_of_processors.rs` | `BagOfProcessors.*` | `Vec<Box<dyn Segmenter>>` keyed by `Algo` enum (replaces if/else ladder), per-file dispatch, weight save/update, metric reduction + confusion matrix. |
| `engine/corpus.rs` | `Corpus.*`, `CorpusItem.*` | Parse `language2classmapping` + `fileslisting` CSVs, `CorpusItem` record, class-balance coefficients. |
| `audio.rs` | `AudioStruct.*` | Audio container + I/O (symphonia), normalization, dither, pre-emphasis, framing/windowing. |
| `features/fft.rs` | `fft.hpp` | FFT (rustfft/realfft) reproducing the two-real packing + custom Welch normalization. |
| `features/mel.rs` | `MelFilterBank.*` | Triangular Mel filterbank (1125/700, unit-peak), DCT-II MFCCs, deltas/delta-deltas, SDC. |
| `features/ltsv_tdc.rs` | `LongTermSpectralVariation.*`, `TimeDomainCorrel.*` | LTSV entropy-variance score, time-domain autocorrelation pitch features. |
| `features/stats.rs` | `InputStatistics.*`, `fmath.hpp` | Streaming per-dim mean/std with parallel merge; fast log/exp policy. |
| `nn/network.rs` | `NeuralNetwork.hpp` | Generic stacked-layer container over a `Layer` trait; sub-sample decimation; weight (de)serialization. |
| `nn/blstm.rs` | `BLSTMNeuralNetwork.*` | Bidirectional wrapper (forward + reverse + output MLP), input normalization, BPTT scheduling, iRPROP- call. |
| `nn/layers.rs` | `LSTMLayer.*`, `SRNLayer.*`, `CWRNNLayer.*`, `NeuronLayer.*`, `Convolutional*.*` | `Layer` impls: peephole LSTM (12-row peephole, `[i\|f\|o\|g]` gates), SRN, Clockwork-RNN, dense, forward-only conv. |
| `nn/activations.rs` | `ActivationFunctions.h` | Active (overridden) activations: asinh cell/output, `sigmoid(0.1z)` gates, softmax. |
| `nn/train.rs` | `Rprop.cpp`, `Trainer.h` | iRPROP- on flat weight vector (inverted sign convention, weight-backtracking). |
| `cost.rs` | `CostLaw.*` | Piecewise speech/no-speech cost primitives + multiclass softmax cross-entropy with ignore-mask. |
| `tasks/segmenter.rs` | `Segmenter.*` | `Segmenter` trait + decision logic (hysteresis-with-area, single-threshold LID, 7-step smoothing). Pure logic, golden-test first. |
| `tasks/sad.rs` | `BLSTMSignalSegmenter.*`, `BLSTMSpectralSegmenter.*`, `BLSTMSpectralLID.*` | SAD segmenters: BLSTM over signal/spectral/LTSV features -> per-frame posterior. |
| `tasks/lid.rs` | `TwinBLSTMSpectralLID.*` | Twin/Siamese LID: SAD + LID BLSTM (+optional CNN), per-segment language accumulation, argmax, confusion. |
| `tasks/segmentation_io.rs` | `Segmentation.*` | Segment container + primitives (sanitize 1e-4s, suppress-short, padding), error/WER/LID scoring, VRCTS XML + STM/TRS loaders. |
| `tasks/vrcts.rs` | `VRCTSpart.*` | External-tool adapter behind the `Segmenter` trait. |
| `io/binary.rs` | `Helpers.hpp` (`BinaryFile2Vector`/`Matrix2BinaryFile`) | Weight `.bin` codec (i64 LE rows, i64 LE cols, f64 LE column-major). The MATLAB seam. |
| `io/matfile.rs` | `Helpers.hpp` (`Matrix2MatFile`), `Timer.cpp` | `.mat` v5 read/write for diagnostics/legacy parity. |

## 6. Python module map (stub-level provenance)

| Python module | Legacy source | Responsibility |
|---------------|---------------|----------------|
| `cli.py` + `drivers/` | `Init_BLSTM.m`, `Train_BLSTM*.m`, `ReTrain_BLSTM.m`, `Test_BLSTM.m`, `valid_batch.m` | Run lifecycle drivers; typed pydantic state replaces the mutable `PS` god-struct. |
| `config_bridge.py` | `vec2struct.m`, `printConfig.m`, `readConfig.m`, `SanitizeConfigStruct.m` | Param-vector + mask + arch <-> config struct <-> `.config` text; weight-key exclusion as an explicit denylist. |
| `weight_bridge.py` | `config2network*.m`, `network2config*.m`, `config2weights*.m`, `nnet2MatFile.m`, `weights2nnet.m`, `Write/ReadMatrixFromBinary.m`, `CombineNNets.m`, `ModifyOutputNetwork.m`, `ForceLSTMBiais.m` | High-risk weight serialization: canonical column-major flat vector + `adim_coeff` scaling + `.bin` codec. Round-trip property tests mandatory. |
| `engine.py` | `ComputeCost.m`, `ComputeGradient*.m`, `CostFunction*.m`, `RunFsp.py` | Engine driver: call the Rust engine via PyO3 (replaces subprocess), normalize gradients by count, fold input stats into layer 1, assemble cost (CE + WER + L2 + penalties). |
| `optimizers.py` | `QuantumPSO.m`, `pso_Trelea_vectorized.m`, `cmaes.m`, `SMORMS3.m`, `Adam.m`, `RMSprop.m`, `Rprop.m`, `sfo.m`, `BackPropagation.m` | Optimizer zoo. SMORMS3 wired; QuantumPSO faithful reimplementation; CMA-ES via `cma`. |
| `scoring.py` | `confusion*.m`, `CheckGrad.m`, `MaskingValidation.m` | LID confusion matrices, EER cutoff, analytic-vs-numeric grad check, config round-trip masking. |
| `batching.py` | `CreateBatches.m`, `GetNewBatch.m`, `getCases.m`, `processListing.m` | Hard-example mini-batch scheduler, PID importance weighting, listing parsing. |
| `nn_reference.py` | `BLSTM_Forward.m`, `BLSTM_Backward.m`, `Hz2Mel.m`, `Mel2Hz.m` | Pure-Python BLSTM forward/backward oracle for gradient-check fixtures only. |
| `dataprep/` | `AugmentCorpus.py`, `ProcessOpenSAD15Corpus.py`, STM normalizers, `Write*Listing.m` | Corpus prep (Python 3): augmentation, NIST OpenSAD15 conversion, STM normalization, listing writers. |

## 7. Toolchain specifics

- **`pyproject.toml`** (retarget Aerocapture's): `name = "speech"`, `requires-python = ">=3.14"`,
  `[tool.hatch.build.targets.wheel] packages = ["src/python/speech"]`, ruff line-length 160
  target py314 (select E,F,I,W,UP,B,SIM), mypy strict (`mypy_path = src/python`), pytest
  `pythonpath = ["src/python", "."]`. Core deps: `numpy`, `scipy`, `pandas`, `pyarrow`,
  `pydantic`, `matplotlib`, `rich`, `soundfile`, `cma`. Dev group: `pytest`, `ruff`, `mypy`,
  `hypothesis`, `maturin`, plus type stubs.
- **`Cargo.toml`** workspace `speech` + `speech-py`; edition 2024, release profile LTO. Deps:
  `ndarray`, `nalgebra`, `rustfft`, `realfft`, `symphonia`, `rayon`, `byteorder`, `bytemuck`,
  `matfile`, `quick-xml`, `serde`, `serde_json`, `toml`, `indexmap`, `anyhow`/`thiserror`.
  Dev: `approx`, `proptest`, `rstest`, `tempfile`. `speech-py`: `pyo3`, `numpy`.
- **Scripts** adapted 1:1: `setup_env.sh` (fresh `.venv` + `uv sync` + PYTHONPATH), `build.sh`
  (`cargo build --release` + `maturin develop` for `speech-py`), `check_all.sh` (cargo
  test/fmt/clippy + build), `lint_code.sh` (ruff imports/format/check + mypy),
  `upgrade_dependencies.sh`.
- **CI** `.github/workflows/ci.yml`: rust (fmt, clippy, test) + python (ruff lint, ruff format,
  mypy, pytest) + pyo3 (maturin build + targeted pytest). PRs to `main` + `workflow_dispatch`.
- **`.zed/debug.json`** + **`speech.code-workspace`** retargeted (binary `speech`, crate paths).
- **`.gitignore`** extend the current Rust-only file with: Python (`.venv/`, `__pycache__/`,
  `.mypy_cache/`, `.pytest_cache/`, `.hypothesis/`, `.ruff_cache/`, `*.egg-info/`), MATLAB/data
  (`*.mat`), `output/*` (keep `.gitkeep`), `legacy/`, `.claude/`, `.DS_Store`.

## 8. Documentation content

- **`CLAUDE.md`**: dense Project Overview (LID+SAD, the C++<->MATLAB seam, the PyO3 replacement),
  Build & Development Commands, Architecture (the module maps of sections 5-6 with
  responsibilities + provenance), a **Key Decisions & Pitfalls** section seeded with the parity
  hazards below, Conventions, Tone (quirky critical peer reviewer, per Aerocapture).
- **`README.md`**: what-it-is paragraph, Quick Start, Project Structure, the SAD/LID + feature/NN
  domain explanation, TOML config format, the validation strategy, and the phased Roadmap.

**Parity hazards to document (from the analysis):** weight-vector flat layout ordering
(input/recurrent gate blocks, 12-row peephole bundle, 4 biases, mean/std tail); `adim_coeff =
sqrt(fan_in*sub + fan_out)` scaling (bias-excepted) living only in the config reader, not the
flat packer; activations (`Maxmin2`/`Identity` = asinh, `GatesFunction` = `sigmoid(0.1z)`);
iRPROP- inverted sign convention; FFT two-real packing + custom Welch normalization;
non-standard LTSV (`-sum r(r-1)`, variance not sqrt); fixed 100000-entry random tables for
determinism; little-endian assumption in `.bin`/`.mat`; MATLAB 1-based vs config 0-based
`Layer_%d` naming and the `score>150`/`val-200` in-band signaling offsets.

## 9. Roadmap (documented in README, not built here)

- **Phase 0 — pure-logic + I/O parity:** `config`/`legacy_config`, `io::binary`/`matfile`,
  `tasks::segmenter`/`segmentation_io`, cost primitives. Golden-tested bit-for-bit. (First.)
- **Phase 1 — features:** audio, FFT, Mel/MFCC/deltas/SDC, LTSV, TDC, stats. Golden vs legacy
  `.mat` dumps.
- **Phase 2 — NN forward:** layers, BLSTM, network, activations. Forward-parity vs
  `BLSTM_Forward` oracle.
- **Phase 3 — training/optimizers:** BPTT gradients, iRPROP-, the Python optimizer zoo. Gradient
  parity vs `BLSTM_Backward` + finite diff.
- **Phase 4 — PyO3 + end-to-end:** `speech-py` bridge, engine driver, drivers/dataprep,
  end-to-end LID/SAD score parity on a held-out corpus.

## 10. Definition of done (this deliverable)

- `cargo build --release` and `cargo clippy --all-targets -- -D warnings` succeed in `src/rust`.
- `cargo test` passes (trivial per-module + one integration smoke test).
- `uv sync --group dev` succeeds; `ruff check`, `ruff format --check`, `mypy --strict`, and
  `pytest tests` all pass (a smoke test asserts the package imports).
- `./build.sh`, `./check_all.sh`, `./lint_code.sh` run green.
- `CLAUDE.md` and `README.md` present and accurate to sections 5-9.
- Final step: invoke the `smart-commit` skill over the whole branch.
