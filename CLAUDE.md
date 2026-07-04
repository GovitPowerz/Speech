# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech is a port of the legacy `FastSpeechProcessing` system: a fast forward+gradient engine for Speech Activity Detection (SAD) and spoken Language Identification (LID). The original splits across two languages -- a C++ engine (~21.5k LOC of logic plus a 200k-line `Constants.h` data table) that does audio I/O, acoustic feature extraction (FFT periodogram, Mel/MFCC/deltas/SDC, LTSV, time-domain pitch), from-scratch bidirectional recurrent nets (peephole BLSTM, SRN, Clockwork-RNN) + output MLP, analytic BPTT gradients, an in-engine iRPROP- update, SAD segmentation, and a TwinBLSTM LID head that consumes the SAD net's hidden states; and a MATLAB `Optimizer_V6.2.2` (110 `.m` files) that runs the outer optimization loop -- global DSP-hyperparameter search (bespoke QuantumPSO or CMA-ES) plus gradient NN training (SMORMS3 and friends), cross-entropy losses, calibration cost laws, EER-cutoff confusion scoring, and hard-example mini-batching. The port maps the C++ engine to **Rust** (crate + binary `speech`) and the MATLAB optimizer to **Python** (package `speech`). The legacy seam is a file/subprocess boundary -- a flat `.config`, little-endian `.bin` weight/gradient vectors, `fileslisting`/`language2classmapping` CSVs, and `.mat` results, with the engine launched per evaluation via `RunFsp.py`. This repo replaces that seam with an **in-process PyO3 API** (`speech-py` -> module `speech_rs`) for the hot training loop, while keeping the legacy formats as a byte-compatible serialization layer for validation and cluster drop-in. The flat-weight layout and `adim_coeff` scaling live **once** in Rust (`io::binary` + `nn`) exposed to Python, not duplicated across two languages as in the legacy. The parity target is bit-exact-where-feasible against the legacy oracle: the roadmap led with a pure-logic + I/O layer (config parser, weight `.bin` codec, segmentation/smoothing/scoring, features) golden-tested bit-for-bit before any NN landed. The phases marked done in the Roadmap / architecture table (pure logic + I/O, features, NN forward) are real, tested logic -- the remaining modules (training, engine drivers, PyO3 seam) are typed stubs with target-accurate signatures and doc-comments, no ported algorithm logic yet.

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
```

## Architecture

The Rust crate (`src/rust/`) is the engine: a Cargo workspace with the core `speech` crate (lib + bin) and the `speech-py` PyO3 binding crate (module `speech_rs`). TOML is the canonical config; a legacy `.config` importer exists for validation runs against goldens. The Python package (`src/python/speech/`) is the optimizer and orchestrator, driving the engine in-process via PyO3.

### Rust module map (`src/rust/src/`)

Each module ships as a compiling stub: a doc-comment naming its legacy source and responsibility, plus key public signatures with `todo!()` or trivial bodies (the modules marked Implemented (Phase ...) below are the exception - they have real, golden-tested bodies).

| Rust module | Legacy source | Responsibility |
|-------------|---------------|----------------|
| `lib.rs` / `main.rs` / `cli.rs` | `FastSpeechProcessing.cpp`, `CorpusProcessor.cpp` | Crate root + engine API (`run_solo`/`run_train`/`run_grad_check`); CLI mirroring the legacy `fsp` modes (`-s`/`-i`/`-t`/`-m` + `--key=val`) as a typed `Mode` enum. |
| `config.rs` | `ConfigFile.*`, `String.hpp` | Implemented (Phase 0a): `NnetSpec` extraction from a parsed legacy config, plus the config<->nnet `adim_coeff` seam and flat weight packer (row-major matrices; cell rows are narrow, no peepholes). TOML canonical config (serde) + typed getters remain the highest-priority exact-match target for later phases. |
| `legacy_config.rs` | `ConfigFile.*` | Implemented (Phase 0a): last-wins KEY-value line parser for the legacy whitespace `.config`, golden-tested against the real `1_worker_1.config`. |
| `constants.rs` | `Constants.h` | Loads two 100000-entry tables (`_RandomGaussVector`, `_RandomVector`) from `data/random_tables.bin` via `include_bytes!`; modular indexing preserved. |
| `engine/corpus_processor.rs` | `CorpusProcessor.*` | Driver: build corpus + processors, rayon `par_iter` over files (replaces OpenMP), epoch training loop, finite-diff grad check, result reduction + `.mat` write. |
| `engine/bag_of_processors.rs` | `BagOfProcessors.*` | `Vec<Box<dyn Segmenter>>` keyed by an `Algo` enum (replaces the if/else ladder), per-file dispatch, weight save/update, metric reduction + confusion matrix. |
| `engine/corpus.rs` | `Corpus.*`, `CorpusItem.*` | Parse `language2classmapping` + `fileslisting` CSVs, `CorpusItem` record, class-balance coefficients. |
| `audio.rs` | `AudioStruct.*` | Implemented (Phase 1): audio container + I/O (symphonia) with (2*RMS+max)/2 normalization, dither/noise, pre-emphasis, windowing coefficients + edge-truncating convolutions, exact GFFT, and the two-real periodogram/framing driver -- golden-tested bit-for-bit against the strict-IEEE oracle harness. |
| `features/fft.rs` | `fft.hpp` | Implemented (Phase 1): exact GFFT port (Taylor-twiddle-seeded radix-2 DIT + Numerical-Recipes running recurrence) reproducing the two-real packing + custom Welch normalization, bit-exact vs the oracle harness. `rustfft`/`realfft` are NOT used here -- library FFTs cannot match the oracle's exact rounding; both remain declared-but-unused deps pending a later cleanup. |
| `features/mel.rs` | `MelFilterBank.*` | Implemented (Phase 1): triangular Mel filterbank (1125/700, unit-peak, whole-bank empty-filter fallback), log/mel `apply_filter_bank`, DCT-II MFCCs (`apply_dct`, explicit ascending-loop product -- the legacy Eigen GEMM order is non-portable), deltas/delta-deltas, and SDC (incl. the `ignoreFirst` last-static-clobber quirk), golden-tested bit-for-bit plus a numpy cross-check. |
| `features/ltsv_tdc.rs` | `LongTermSpectralVariation.*`, `TimeDomainCorrel.*`, `fmath.hpp` | Implemented (Phase 1): LTSV half -- `ltsv_classify_sequence` (biased-variance-not-std spectral-variation score over an inclusive, edge-shrinking context window, mean-only 1e-12 floor) and `get_ltsv` (strided score + linear-interpolation backfill, trailing zeros). TDC half -- `fmath_log` (the 2048-entry f32 bit-trick table log, no domain guard: `log(0)` finite), `tdc_classify_sequence` (MaxPeak + zero-crossing cross-correlation score, `(1-MaxPeak)` narrowed to f32 through the log), `compute_pitch` (argmax-lag estimate via the legacy `long/long` INTEGER division, strict acceptance band), `get_pitch` (SPEECH-segment walk, zero-count -> divide-by-1), `apply_homothety` (freq-bin warp by `pitch/300`). All golden-tested bit-for-bit plus numpy cross-checks (fmath_log double-pinned: harness dump == Rust == numpy oracle). |
| `features/pipeline.rs` | `BLSTMSpectralSegmenter.*` (param derivation + input assembly), `TimeDomainCorrel.*` | Implemented (Phase 1): `FeatureConfig::from_legacy` (config key reads + sanitization order: negative-zero, swap-if-reversed, span-widen), `SpectralParams::derive` (spectrum-order clamp, freq-band snap, LTSV/TDC param derivation, `TdcParams`), and `assemble_input_sequence` (DCT/mel/raw-band priority + LTSV hcat) -- golden-tested bit-for-bit end to end (`inputseq_<variant>_chan{1,2}.bin`, 4 config variants x 2 channels). |
| `features/stats.rs` | `InputStatistics.*`, `fmath.hpp` | Implemented (Phase 1): `InputStatistics` streaming per-dim mean/std -- `from_matrix` (population mean/std over sequential per-column sums, no `n == 0` guard: empty batch yields NaN by design) and `update` (merge: `n == 0` -> plain copy, else pooled merge re-squaring the STORED std in a pinned per-element op order, else no-op), golden-tested against the REAL compiled legacy `InputStatistics.cpp` (linked into the oracle harness) plus a numpy cross-check over chained merges incl. empty chunks. |
| `nn/network.rs` | `NeuralNetwork.hpp` | Implemented (Phase 2): generic `Network<L>` stacked-layer container -- construction, `SubSample` decimation (integer-floor row count, trailing `T mod R` frames silently dropped), per-layer forward chaining, and the head-eating flat weight (de)serialization seam. Golden-tested against the harness reimpl incl. odd-length (T=11) container runs. |
| `nn/blstm.rs` | `BLSTMNeuralNetwork.*` | Implemented (Phase 2, forward only): bidirectional wrapper -- construction from legacy config, flat weight seam, the input-normalization types, core forward (fwd+rev LSTM -> HCAT -> output MLP), the 4 windowed drivers (truncate, two-sweeps, overlap, MLP-overlap) with their quirks reproduced, and the top-level scoring forward incl. cost/`_NbOfClassif` accumulation. E2E gate green: real config + real 33,671-weight net -> posteriors bit-exact vs the strict-IEEE reimpl. BPTT + the iRPROP- call are Phase 3. |
| `nn/layers.rs` | `LSTMLayer.*`, `NeuronLayer.*` (`SRNLayer.*`, `CWRNNLayer.*`, `Convolutional*.*` NOT ported) | Implemented (Phase 2): peephole LSTM forward (12-row peephole bundle, `[i|f|o|g]` gate blocks, reverse pass as reverse-input -> forward -> reverse-output) + dense `NeuronLayer` (asinh hidden, unstabilized softmax output), all products via the measured ascending-loop contract (`matmul_seq`). SRN/CWRNN forward NOT ported -- dead code in legacy (compile-time instantiations only, zero live constructors). Conv NOT ported -- broken-as-committed in legacy (UB with the real lid.config, orphaned weights; see IMPROVEMENTS.md). |
| `nn/activations.rs` | `ActivationFunctions.h` | Implemented (Phase 2): the active overrides -- asinh cell/output (`Maxmin2`/`Identity`), `sigmoid(0.1z)` gates (`GatesFunction`) and plain `Logistic`, with their exact `expLimit` saturation guards; softmax is inline in `NeuronLayer` per the legacy. |
| `nn/train.rs` | `Rprop.cpp`, `Trainer.h` | Stub (Phase 3): iRPROP- on the flat weight vector (inverted sign convention, weight-backtracking). |
| `cost.rs` | `CostLaw.*` | Implemented (Phase 0b-i): 7 piecewise VAD cost laws (linear/square/cubic/log/sqrt, below+above-thresh) + multiclass softmax cross-entropy with ignore-mask, ported bit-exactly incl. the double-read and LogLaw quirks (see IMPROVEMENTS.md). |
| `tasks/segmenter.rs` | `Segmenter.*` | Implemented (Phase 0b-ii; Phase 2b added the real `Segmenter` trait + driver residue): `SegmenterConfig` + the decision primitives -- `update_segmentation` (hysteresis-with-area, row vector), `lid_to_segmentation` (single-threshold, col vector -- the orientation asymmetry is load-bearing), `smooth_segmentation` (fixed 8-step pipeline), `get_targets`; plus `DriverConfig` (dump dir, window/shift, convolution kernel, `BackPropWER`) and `results_to_segmentation` (in-place convolution gate + dispatch). LTSV/BLSTM-spectral/BLSTM-signal segmenters (`tasks/sad.rs`) remain stubs pending later Phase 2b tasks. |
| `tasks/sad.rs` | `BLSTMSignalSegmenter.*`, `BLSTMSpectralSegmenter.*`, `BLSTMSpectralLID.*`, `TimeDomainCorrel.*` | Implemented (Phase 2b, Algo 1 only): `TdcSegmenter` (time-domain-correlation SAD, no NN) -- `from_legacy` + `impl Segmenter`, golden-tested bit-exact against the real compiled `TimeDomainCorrel::getSegmentation` (no transcription needed: the chain has no NN, only `fmath::log`/cos). LTSV/BLSTM-spectral/BLSTM-signal segmenters (Algo 2-4) remain stubs. |
| `tasks/lid.rs` | `TwinBLSTMSpectralLID.*` | Twin/Siamese LID: SAD + LID BLSTM (+ optional CNN), per-segment language accumulation, argmax, confusion. |
| `tasks/segmentation.rs` | `Segmentation.*` | Implemented (Phase 0b-i): boundary-list `Segmentation` container (`SegClass` + `Segment`) with `label_segment`/`sanitize`/`suppress_short`/`add_padding`/`modify_type`/`update_count`, ported from `Segmentation.cpp`. |
| `tasks/segmentation_io.rs` | `Segmentation.*` | Implemented (Phase 0b-ii): VRCTS XML write/load (`%f.4s` 4-decimal times, byte-exact round-trip against the real fixtures for the loader), STM/CSV reference loaders, ASCII writer, and `compute_errors` (per-class Pfa/Pmiss/ErrorRate Pass 2 + WER/coverage/delay Pass 1), ported from `Segmentation.cpp`. TRS loader + LID scoring not yet ported. |
| `tasks/vrcts.rs` | `VRCTSpart.*` | External-tool adapter behind the `Segmenter` trait. |
| `io/binary.rs` | `Helpers.hpp` (`BinaryFile2Vector`/`Matrix2BinaryFile`) | Implemented (Phase 0a): weight `.bin` codec (i64 LE rows, i64 LE cols, f64 LE column-major), byte-exact-validated against the real `NNweights_config1.bin`. The MATLAB seam. |
| `io/matfile.rs` | `Helpers.hpp` (`Matrix2MatFile`), `Timer.cpp` | `.mat` v5 read/write for diagnostics/legacy parity. |

### Python module map (`src/python/speech/`)

| Python module | Legacy source | Responsibility |
|---------------|---------------|----------------|
| `cli.py` + `drivers/` | `Init_BLSTM.m`, `Train_BLSTM*.m`, `ReTrain_BLSTM.m`, `Test_BLSTM.m`, `valid_batch.m` | Run-lifecycle drivers; typed pydantic state replaces the mutable `PS` god-struct. |
| `config_bridge.py` | `vec2struct.m`, `printConfig.m`, `readConfig.m`, `SanitizeConfigStruct.m` | Implemented (Phase 0a): legacy `.config` <-> parsed-config bridge, golden-tested against the real `1_worker_1.config`. Param-vector + mask + arch <-> config struct still pending; weight-key exclusion as an explicit denylist. |
| `weight_bridge.py` | `config2network*.m`, `network2config*.m`, `config2weights*.m`, `nnet2MatFile.m`, `weights2nnet.m`, `Write/ReadMatrixFromBinary.m`, `CombineNNets.m`, `ModifyOutputNetwork.m`, `ForceLSTMBiais.m` | Implemented (Phase 0a): flat weight packer + `adim_coeff` seam, cross-language-validated (Rust-pack == Python-pack) and source-verified against `config2weights.m`; end-to-end golden vs a legacy pack deferred to the end-to-end inference-output parity milestone (no independent legacy golden exists yet). `.bin` codec round-trip property-tested. |
| `engine.py` | `ComputeCost.m`, `ComputeGradient*.m`, `CostFunction*.m`, `RunFsp.py` | Engine driver: call the Rust engine via PyO3 (replaces the subprocess), normalize gradients by count, fold input stats into layer 1, assemble cost (CE + WER + L2 + penalties). |
| `optimizers.py` | `QuantumPSO.m`, `pso_Trelea_vectorized.m`, `cmaes.m`, `SMORMS3.m`, `Adam.m`, `RMSprop.m`, `Rprop.m`, `sfo.m`, `BackPropagation.m` | Optimizer zoo. SMORMS3 wired; QuantumPSO faithful reimplementation; CMA-ES via `cma`. |
| `scoring.py` | `confusion*.m`, `CheckGrad.m`, `MaskingValidation.m` | LID confusion matrices, EER cutoff, analytic-vs-numeric grad check, config round-trip masking. |
| `batching.py` | `CreateBatches.m`, `GetNewBatch.m`, `getCases.m`, `processListing.m` | Hard-example mini-batch scheduler, PID importance weighting, listing parsing. |
| `nn_reference.py` | `BLSTM_Forward.m`, `BLSTM_Backward.m`, `Hz2Mel.m`, `Mel2Hz.m` | Partly implemented (Phase 2): independent scalar-loop LSTM/dense forward oracles with ENGINE semantics, cross-checked bit-exact against the Rust layers; gradient-check fixtures only. `BLSTM_Forward.m` itself was NOT ported -- it implements different semantics than the engine (`sigmoid(x)` gates, no output softmax) and cannot match it. Backward oracle lands in Phase 3. |
| `dataprep/` | `AugmentCorpus.py`, `ProcessOpenSAD15Corpus.py`, STM normalizers, `Write*Listing.m` | Corpus prep: augmentation, NIST OpenSAD15 conversion, STM normalization, listing writers. |

## Key Decisions & Pitfalls

### Locked decisions

- **Naming**: Rust core crate `speech` (lib + bin), PyO3 crate `speech-py` -> module `speech_rs`, Python package `speech`, CLI binary `speech`. Mirrors the sibling `aerocapture`/`aerocapture-py`/`aerocapture_rs`.
- **Legacy source**: vendored source-only under `legacy/` (C++ `src/` + `Optimizer_V6.2.2` `.m`, no data/figures/executables). `legacy/` is **git-ignored** -- present locally as a golden oracle, not committed.
- **Config format**: **TOML-first** (config is the only input), plus a legacy `.config` compatibility importer for validation runs against goldens.
- **Roadmap order**: lead with the **pure-logic + I/O parity layer** (config parser, weight `.bin` codec, segmentation/smoothing/scoring, feature extraction), golden-tested bit-for-bit before any NN.
- **Phase 0a status (complete)**: the `.bin` codec is byte-exact-validated against the real `NNweights_config1.bin`, and the legacy `.config` parser against the real `1_worker_1.config`. The `adim_coeff` seam + flat weight packer is validated by source cross-check vs `config2weights.m`, the exact element count (33,671), the mutual-inverse bijection, adim invariants, the normalize-tail matching a real `.bin`, and Rust-pack == Python-pack cross-language agreement -- it is NOT validated against an independent legacy golden pack, because none exists (`nnet_best` is the CMA-ES optimizer genome, not a weight-pack). The bit-exact packer/adim golden is **deferred to the end-to-end inference-output parity milestone** (matching a real `.bin` against the legacy VRCTS segmentations; README Roadmap Phase 4).

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

### Scaffolding-phase relaxations (remove as modules land)

This repo started as all-stubs with two warning/typecheck suppressions in place. Remove the rest as real logic replaces the remaining stubs:

- Rust: the crate-root `#![allow(dead_code, unused_variables)]` in `src/rust/src/lib.rs` was removed in Phase 0a -- the crate now builds clean under `-D warnings` with `io/binary.rs`, `legacy_config.rs`, and `config.rs` implemented. If a future stub needs suppression, add a narrow `#[allow(...)]` on that item, not a crate-wide blanket.
- Python: `allow_empty_bodies = true` in the mypy config (`pyproject.toml`).

### Legacy quirks & deferred fixes

Legacy bugs/oddities are reproduced bit-exactly on purpose (they are load-bearing for the goldens) and tracked in **[IMPROVEMENTS.md](IMPROVEMENTS.md)** to revisit after end-to-end parity. Do NOT "fix" them mid-port. **Every phase that reproduces a legacy quirk/bug adds an entry to IMPROVEMENTS.md.**

## Conventions

- **Rust**: Edition 2024, `nalgebra`/`ndarray` for linear algebra, release profile with LTO. Engine deps include `rustfft`/`realfft`, `symphonia`, `rayon`, `byteorder`/`bytemuck`, `quick-xml`, `serde`/`toml`; `matfile` lands in Phase 4 when `io/matfile.rs` is implemented. `speech-py` uses `pyo3` + `numpy`.
- **Python**: Python >=3.14, ruff (line-length 160, target py314; select E,F,I,W,UP,B,SIM), mypy strict mode, uv package manager. Dev tools in `[dependency-groups]` (not `[project.optional-dependencies]`). Core deps: numpy, scipy, pandas, pyarrow, pydantic, matplotlib, rich, soundfile, cma.
- **Testing (Python)**: pytest, hypothesis (property-based). Golden reference fixtures under `tests/reference_data/`, added per roadmap phase. Round-trip property tests are mandatory for the weight bridge.
- **Testing (Rust)**: unit tests (inline `#[cfg(test)]`, proptest property tests), integration tests (`src/rust/tests/`). Dev-deps: `approx`, `proptest`, `rstest`, `tempfile`. Validation is golden-file bit-for-bit against the git-ignored `legacy/` oracle, cross-checked via `tools/oracle_harness/` (a local-only strict-IEEE rebuild of vendored legacy sources; CI consumes only the committed fixtures it produces, never builds the harness itself). Since Phase 2 the harness also LINKS the vendored NN translation units (`BLSTMNeuralNetwork`/`LSTMLayer`/`NeuronLayer` + link-only `SRNLayer`/`CWRNNLayer`/`CostLaw`/`Rprop`) and probes the REAL compiled classes against a harness-local ascending-loop reimplementation: NN goldens are dumped from the reimpl, because Eigen's blocked GEMM diverges from ascending accumulation at the k>=23 NN shapes (`manifest.json:nn_product_probes`) and the layer-0 GEMM seeds the recurrence, so the divergence propagates non-locally; NN_TOL probes record the real-class deltas (all synthetic sites 0 ULP; real-net sites small nonzero, kept as Phase 4 calibration data). Bit-exactness holds on the oracle environment (Apple libm; canaries in `tests/reference_data/phase1/libm_canaries.bin`); on other libms (e.g. CI glibc) the transcendental-dependent goldens assert a hybrid bound -- `<= 4` ULP or an absolute tolerance of `512*eps` scaled to the fixture magnitude (libm differences propagate absolutely, so near-zero outputs can be many ULP off) -- canary-gated (`common::oracle_mode`), while the pure-arithmetic goldens stay bit-exact everywhere.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) -- Rust (fmt, clippy, test), Python (ruff lint, ruff format, mypy, pytest), and PyO3 (maturin build + targeted pytest) run on PRs to `main` and manual dispatch (`workflow_dispatch`).

## Tone

Be a **quirky friendly but critical peer reviewer**. Think of yourself as a quirky senior developer doing a code review: helpful, but holding me to high standards. Always **challenge inefficiencies**: if I'm doing something the hard way, call it out. Given the bit-exact-against-legacy target, treat every parity hazard above as a place where "close enough" is a bug.
