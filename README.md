# Speech

Speech Activity Detection (SAD) and spoken Language Identification (LID) engine. It is a port of the legacy `FastSpeechProcessing` system: a C++ forward+gradient engine becomes a **Rust engine** (crate + binary `speech`), and a MATLAB optimizer becomes a **Python optimizer/orchestrator** (package `speech`). The two talk in-process through a PyO3 bridge (`speech_rs`). The port targets bit-level parity against the legacy oracle, validated layer by layer.

**Status: scaffold, phases landing incrementally.** The full module tree compiles and is wired end to end. The phases marked done in the Roadmap / architecture table are real, tested logic; the remaining modules are typed stubs with target-accurate signatures and doc-comments, not ported algorithms. The SAD/LID/feature/NN behavior described below is the target the Roadmap builds toward, not what runs today. Each algorithm lands as its own roadmap phase, golden-tested against the legacy before it is trusted.

## Quick Start

```bash
# Build the Rust engine (binary `speech`)
cd src/rust && cargo build --release && cd ../..

# Run on a TOML config
# Scaffold today: prints "engine not yet implemented"; real LID/SAD lands per-phase (see Roadmap)
./src/rust/target/release/speech -m configs/lid/lid_blstm.toml

# Build the PyO3 bindings (module speech_rs)
uv run maturin develop --release --manifest-path src/rust/speech-py/Cargo.toml

# Set up the Python environment (Python >=3.14)
uv sync --group dev

# Run tests
cd src/rust && cargo test && cd ../..
uv run pytest tests
```

## Project Structure

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
  docs/superpowers/specs/                  # design spec + future phase specs
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

## What it does

Two tasks, built on a shared feature front-end and neural-net zoo (target behavior; see Roadmap - not what runs today).

- **SAD (Speech Activity Detection).** A recurrent net emits a per-frame speech posterior; segmentation turns that posterior into speech/non-speech segments via a hysteresis-with-area double threshold plus a 7-step smoothing pass. Pure decision logic, golden-tested first.
- **LID (spoken Language Identification).** A TwinBLSTM head consumes the SAD net's hidden states (the "twin" seam), accumulates per-segment evidence, and produces a per-file language posterior (argmax + confusion). LID is stacked on top of SAD, not run independently.

Underneath both:

- **Feature front-end:** framing/windowing, radix-2 FFT periodogram (two-real packing, custom Welch normalization), triangular Mel filterbank (1125/700 scaling) with DCT-II MFCCs and deltas/delta-deltas/SDC, LTSV (entropy-variance), and time-domain pitch/autocorrelation (TDC).
- **NN zoo:** from-scratch bidirectional recurrent nets over a `Layer` trait: peephole BLSTM (12-row peephole, `[i|f|o|g]` gates), plus SRN and Clockwork-RNN (CWRNN) variants, forward-only convolution (CNN), and a dense output MLP. Non-standard activations throughout (asinh cell/output, `sigmoid(0.1z)` gates, softmax).

## Configuration

TOML is the canonical and only input format. A run config names the algorithm and its DSP/decision parameters; the engine is launched on a single `.toml` file (`speech -m <config.toml>`).

```toml
# configs/lid/lid_blstm.toml  (Twin-BLSTM LID, legacy Algo_choice 6)
[engine]
algo = "twin_blstm_lid"
num_outer_threads = 1
num_inner_threads = 1
results_file = "MultiConfigResults.mat"

[audio]
offset = 0.0
max_duration = 120.0

[decision]
thresh_rising = 0.6
area_rising = 0.0
thresh_falling = 0.3795068189162301
area_falling = 0.0

[spectrum]
order = 9
shift = 0.025
temporal_convolution_type = "hamming"

[preprocess]
preemph_ratio = -0.97
noise_seed = 3
noise_ratio = 0.02921428128844371
```

A legacy `.config` importer (`legacy_config.rs`, `configs/legacy/`) parses the original flat whitespace format (with its `_`-continuation, last-wins, and `val*count` repeat quirks) so validation runs can be driven from the exact goldens the legacy consumed. TOML is the format going forward; the importer exists only for parity.

## Validation strategy

Parity is verified as a layered ladder, each rung golden-tested against the legacy oracle before the next is trusted:

1. **Golden features** - feature extraction bit-for-bit against legacy `.mat` dumps.
2. **Weight roundtrip** - flat weight-vector `.bin` codec (ordering + `adim_coeff` scaling) survives a Rust encode/decode without drift; property-tested.
3. **Forward parity** - NN forward matches the `BLSTM_Forward` oracle.
4. **Gradient parity** - analytic BPTT matches `BLSTM_Backward` and finite differences.
5. **Optimizer-step parity** - a single iRPROP-/optimizer update matches the legacy step.
6. **Decision/scoring parity** - segmentation, smoothing, WER/error/LID scoring, EER-cutoff confusion.
7. **End-to-end LID/SAD** - full-pipeline score parity on a held-out corpus.

The flat-weight layout and `adim_coeff` scaling live once in Rust (not duplicated across two languages as in the legacy), so the codec has a single source of truth to validate.

## Roadmap

The port lands in phases, leading with the lowest-risk, highest-leverage layer (pure logic + I/O) so everything downstream stands on golden-tested ground.

- **Phase 0 - pure-logic + I/O parity (first).**
  - **Phase 0a - byte-parity foundation (done).** The `io::binary` `.bin` codec and the `legacy_config`/`config_bridge` parser are golden-tested bit-for-bit against the real `NNweights_config1.bin` and `1_worker_1.config`. The `config`/`weight_bridge` adim seam + flat weight packer is cross-language-validated (Rust-pack == Python-pack) and source-verified against `config2weights.m` (element count, bijection, adim invariants, normalize-tail vs a real `.bin`); an independent legacy golden pack does not exist yet (`nnet_best` is a CMA-ES genome, not a weight-pack), so the end-to-end bit-exact packer/adim golden is deferred to the end-to-end inference-output parity milestone (inference on a real `.bin` matched against the legacy VRCTS segmentations; see Phase 4 below).
  - **Phase 0b-i - cost laws + Segmentation container (done).** `cost.rs` ports the 7 piecewise VAD cost laws (linear/square/cubic/log/sqrt, below+above-threshold) plus the multiclass softmax cross-entropy; the scalar VAD path is golden-tested bit-for-bit against the legacy `test_costlaws.mat` (linear law, `CostPonderation=0.2`), while the other laws and the softmax path are unit-tested from the exact `CostLaw.{h,cpp}` coefficients (no external `.mat` golden for those yet). `tasks::segmentation` ports the boundary-list `Segmentation` container (`SegClass`/`Segment`, `label_segment`/`sanitize`/`suppress_short`/`add_padding`/`modify_type`/`update_count`), unit-tested against hand-derived boundary lists plus a `sanitize` proptest; no external golden yet (segmentation-output goldens land in 0b-ii via VRCTS XML).
  - **Phase 0b-ii - segmenter decision logic + scoring (done).** `tasks::segmenter` ports the decision primitives (`update_segmentation` hysteresis-with-area over a row vector, `lid_to_segmentation` single-threshold over a column vector, the fixed 8-step `smooth_segmentation`, `get_targets`); `tasks::segmentation_io` ports the VRCTS XML writer/loader, the STM/CSV reference loaders, the ASCII writer, and `compute_errors` (per-class Pfa/Pmiss/ErrorRate Pass 2 plus the WER/coverage/delay pass). Validation: the VRCTS loader round-trips byte-exact against the real fixtures and the writer is pinned by a constructed 4-decimal (`%f.4s`) golden (the committed fixtures are external `vrcts_part` reference input, so the writer-vs-real-engine byte golden defers to Phase 4); the segmenter decision and `compute_errors` Pass 2 are hand-computed-unit-tested **and** cross-checked bit-for-bit against a numpy differential oracle (Rust == Python); WER (Pass 1/3) is unit-tested only (no `.csv` reference in the tree yet). This **closes the Phase 0/0b pure-logic + I/O parity layer** -- next is Phase 1 (features).
- **Phase 1 - features (in progress).** Audio I/O, FFT, Mel/MFCC/deltas/SDC, LTSV, TDC, streaming stats, golden-tested bit-for-bit against a strict-IEEE oracle harness (vendored legacy sources compiled with `-fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE`, dumped via the legacy's own `.bin` writer). Done so far: audio decode/normalization/pre-emphasis/noise, windowing coefficients + edge-truncating convolutions, the exact GFFT port, the two-real periodogram/framing driver, the Mel filterbank (log/mel, whole-bank fallback, DCT-II MFCCs, deltas/delta-deltas, SDC), and the LTSV score (`ltsv_classify_sequence`) + interpolated column (`get_ltsv`) -- the LTSV class itself can't be compiled standalone (it drags the whole `Segmenter` hierarchy), so the oracle harness carries a verbatim transcription instead of a direct legacy build. Remaining: time-domain autocorrelation pitch (TDC), streaming stats.
- **Phase 2 - NN forward.** Layers, BLSTM, network container, activations. Forward-parity vs the `BLSTM_Forward` oracle.
- **Phase 3 - training/optimizers.** BPTT gradients, iRPROP-, and the Python optimizer zoo (SMORMS3, QuantumPSO, CMA-ES, and friends). Gradient parity vs `BLSTM_Backward` + finite diff.
- **Phase 4 - PyO3 + end-to-end.** The `speech-py` bridge, engine driver, run drivers and dataprep, then end-to-end LID/SAD score parity on a held-out corpus.

Each phase carries its own spec under `docs/superpowers/specs/`.

## Testing / CI

```bash
./check_all.sh          # Rust: cargo test + fmt --check + clippy + release build
./lint_code.sh          # Python: ruff (imports, format, check) + mypy --strict
uv run pytest tests     # Python test suite
```

GitHub Actions (`.github/workflows/ci.yml`) runs on PRs to `main` and manual dispatch (`workflow_dispatch`), with four jobs:

- **rust** - `cargo fmt --check`, `cargo clippy`, `cargo test`.
- **python-lint** - `ruff check`, `ruff format --check`, `mypy --strict`.
- **python-test** - `pytest`.
- **python-pyo3** - `maturin develop --release` for `speech-py`, then an import check of `speech_rs`.
