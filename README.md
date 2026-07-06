# Speech

Speech Activity Detection (SAD) and spoken Language Identification (LID) engine. It is a port of the legacy `FastSpeechProcessing` system: a C++ forward+gradient engine becomes a **Rust engine** (crate + binary `speech`), and a MATLAB optimizer becomes a **Python optimizer/orchestrator** (package `speech`). The two talk in-process through a PyO3 bridge (`speech_rs`). The port targets bit-level parity against the legacy oracle, validated layer by layer.

**Status: phases landing incrementally.** The full module tree compiles and is wired end to end. The phases marked done in the Roadmap / architecture table are real, tested logic (pure logic + I/O, features, the NN forward, the four per-file SAD segmenter drivers as of Phase 2b -- a real wav now flows wav -> features -> NN posteriors -> hysteresis decision -> `Segmentation` -> VRCTS XML bytes in-process, golden-tested against the oracle -- and, as of Phase 3, the network-level BPTT + iRPROP- training chain); the remaining modules (the corpus/engine drivers, LID, PyO3 seam) are typed stubs with target-accurate signatures and doc-comments, not ported algorithms. The corpus-level SAD/LID runs described below are the target the Roadmap builds toward, not what runs today (per-file SAD segmentation is real; the CLI/corpus driver on top of it is not). Each algorithm lands as its own roadmap phase, golden-tested against the legacy before it is trusted.

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
3. **Forward parity** - NN forward matches the strict-IEEE harness oracle (real compiled legacy TUs probed against an ascending-loop reimpl; MATLAB's `BLSTM_Forward.m` has divergent semantics and was never the oracle).
4. **Gradient parity** - analytic BPTT matches the same harness oracle bit-exactly plus central finite differences (MATLAB's `BLSTM_Backward.m` shares the divergent semantics; the Python `nn_reference.py` oracle implements ENGINE semantics instead).
5. **Optimizer-step parity** - a single iRPROP- update matches the REAL compiled legacy `Rprop::updateWeights` bit-for-bit, including a fired cost-gated backtrack.
6. **Decision/scoring parity** - segmentation, smoothing, WER/error/LID scoring, EER-cutoff confusion.
7. **End-to-end LID/SAD** - full-pipeline score parity on a held-out corpus.

The flat-weight layout and `adim_coeff` scaling live once in Rust (not duplicated across two languages as in the legacy), so the codec has a single source of truth to validate.

## Roadmap

The port lands in phases, leading with the lowest-risk, highest-leverage layer (pure logic + I/O) so everything downstream stands on golden-tested ground.

- **Phase 0 - pure-logic + I/O parity (first).**
  - **Phase 0a - byte-parity foundation (done).** The `io::binary` `.bin` codec and the `legacy_config`/`config_bridge` parser are golden-tested bit-for-bit against the real `NNweights_config1.bin` and `1_worker_1.config`. The `config`/`weight_bridge` adim seam + flat weight packer is cross-language-validated (Rust-pack == Python-pack) and source-verified against `config2weights.m` (element count, bijection, adim invariants, normalize-tail vs a real `.bin`); an independent legacy golden pack does not exist yet (`nnet_best` is a CMA-ES genome, not a weight-pack), so the end-to-end bit-exact packer/adim golden is deferred to the end-to-end inference-output parity milestone (inference on a real `.bin` matched against the legacy VRCTS segmentations; see Phase 4 below).
  - **Phase 0b-i - cost laws + Segmentation container (done).** `cost.rs` ports the 7 piecewise VAD cost laws (linear/square/cubic/log/sqrt, below+above-threshold) plus the multiclass softmax cross-entropy; the scalar VAD path is golden-tested bit-for-bit against the legacy `test_costlaws.mat` (linear law, `CostPonderation=0.2`), while the other laws and the softmax path are unit-tested from the exact `CostLaw.{h,cpp}` coefficients (no external `.mat` golden for those yet). `tasks::segmentation` ports the boundary-list `Segmentation` container (`SegClass`/`Segment`, `label_segment`/`sanitize`/`suppress_short`/`add_padding`/`modify_type`/`update_count`), unit-tested against hand-derived boundary lists plus a `sanitize` proptest; no external golden yet (segmentation-output goldens land in 0b-ii via VRCTS XML).
  - **Phase 0b-ii - segmenter decision logic + scoring (done).** `tasks::segmenter` ports the decision primitives (`update_segmentation` hysteresis-with-area over a row vector, `lid_to_segmentation` single-threshold over a column vector, the fixed 8-step `smooth_segmentation`, `get_targets`); `tasks::segmentation_io` ports the VRCTS XML writer/loader, the STM/CSV reference loaders, the ASCII writer, and `compute_errors` (per-class Pfa/Pmiss/ErrorRate Pass 2 plus the WER/coverage/delay pass). Validation: the VRCTS loader round-trips byte-exact against the real fixtures and the writer is pinned by a constructed 4-decimal (`%f.4s`) golden (at this point the committed fixtures were external `vrcts_part` reference input, so the writer-vs-real-engine byte golden deferred; **this deferral was CLOSED in Phase 2b** -- see below -- once a faithful in-harness formatter made the real compiled writer itself a byte-golden source); the segmenter decision and `compute_errors` Pass 2 are hand-computed-unit-tested **and** cross-checked bit-for-bit against a numpy differential oracle (Rust == Python); WER (Pass 1/3) is unit-tested only (no `.csv` reference in the tree yet). This **closes the Phase 0/0b pure-logic + I/O parity layer** -- next is Phase 1 (features).
- **Phase 1 - features (done).** Audio I/O, FFT, Mel/MFCC/deltas/SDC, LTSV, TDC, streaming stats, and feature-pipeline param derivation/assembly, golden-tested bit-for-bit against a strict-IEEE oracle harness (`tools/oracle_harness/`: vendored legacy sources rebuilt with Homebrew g++, `-O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE`, no OpenMP -- a deliberate deviation from the original `fsp` binary's `-ffast-math`/gcc-4.9 build; comparing against that original binary's tolerance is deferred to Phase 4). Bit-exactness is the contract on the oracle environment (Apple libm; libm canaries recorded in the fixtures as `libm_canaries.bin`); on other libms the transcendental-dependent goldens (cos/ln/exp chains) assert a hybrid bound instead of exact bits -- `<= 4` ULP or an absolute tolerance of `512*eps` scaled to the fixture magnitude (a 1-ULP libm difference propagates as an absolute error, so near-zero outputs like a Hann window tail can be many ULP off) -- canary-gated, so CI on glibc stays green without weakening the pure-arithmetic goldens. The **end-to-end feature parity gate** (`phase1_pipeline_golden.rs`) reproduces the full legacy chain -- excerpt -> pre-emphasis -> windowing -> periodogram -> mel/DCT per config -> LTSV -> `assemble_input_sequence` -- bit-exact vs `inputseq_<variant>_chan{1,2}.bin` across 4 config variants (mfcc+deltas, mfcc+SDC, log-mel, raw-band+LTSV) x 2 channels. `features/fft.rs` is an EXACT GFFT port (Taylor-twiddle-seeded radix-2 DIT + Numerical-Recipes running recurrence, the DL&lt;4&gt; fused `-i` butterfly, 1-based NR bit-reversal) -- `rustfft`/`realfft` are NOT used (library FFTs cannot match the oracle's exact rounding; both remain declared-but-unused deps). One real divergence surfaced and was absorbed rather than hidden: the mandatory Task-7 harness pre-check found Eigen's scalar GEMM for the DCT projection (`melPeriodogram * _CoeffsDCT`) does not accumulate in ascending order even fully devectorized (measured: 1511/2613 elements ~1 ULP off an explicit ascending loop), so both the harness and `features/mel.rs` use the explicit ascending-loop product as the portable parity target (`manifest.json:dct_gemm_substitution`) -- DCT-path goldens are therefore ~1 ULP off a hypothetical real-Eigen legacy run, an acceptable/expected gap absorbed by Phase 4. Provenance varies by module: `audio.rs`/`features/{fft,mel,stats}.rs` are pinned against COMPILED legacy translation units linked into the harness (`AudioStruct`, `MelFilterBank` construction + filter-bank apply, `InputStatistics`); `features/ltsv_tdc.rs` and the pipeline's param-derivation/assembly logic are pinned against a diff-reviewed VERBATIM TRANSCRIPTION in the harness (the LTSV/TDC classes can't compile standalone -- they drag in the whole `Segmenter` hierarchy), cross-checked by numpy oracles independent of the harness; symphonia's WAV decode was verified bit-exact against libsndfile (PCM16 x/32768) and is pinned by the `sig_norm.bin` harness dump golden (`decode_normalize_matches_oracle`, `phase1_audio_golden.rs`). See `IMPROVEMENTS.md` for every reproduced legacy quirk (dropped N+1th window sample, DC-bin `/n` vs AC `/4n` periodogram normalization, odd-tail double write, `get_sequence` stale-buffer/no-op/DC-offset-over-padding, unnormalized-uniform-window fallthrough, `round(rate*dur+1)` clip length, convolution edge truncation without renormalization, LTSV variance-not-std, `[delta|delta|dd]` static overwrite, SDC `ignoreFirst` last-static clobber, `fmath_log` finite at 0, `InputStatistics` std-resqrt merge order, and more) -- none of them were "fixed" mid-port; they are load-bearing for the goldens.
- **Phase 2 - NN forward (done).** `nn/{activations,layers,network,blstm}.rs`: the exact activation overrides (asinh cell/output, `sigmoid(0.1z)` gates with their `expLimit` saturation guards), the peephole LSTM forward (12-row peephole bundle, reverse pass as reverse-input -> forward -> reverse-output), the dense `NeuronLayer` (asinh hidden, unstabilized softmax output), the `Network<L>` container (`SubSample` integer-floor decimation, head-eating flat weight seam), and the BLSTM wrapper (input normalization, core fwd+rev -> HCAT -> output-MLP forward, the 4 windowed drivers -- truncate, two-sweeps, overlap, MLP-overlap -- and the top-level scoring forward with cost/`_NbOfClassif` accumulation). Scope is LSTM + dense only: SRN/CWRNN are dead code in the legacy (compile-time instantiations, zero live constructors) and the conv layer is broken-as-committed (UB with the real lid.config, orphaned weights) -- both stay stubs, documented in `IMPROVEMENTS.md`; `nn/train.rs` (iRPROP-) is Phase 3. The Phase 1 GEMM finding recurs, non-locally this time: Eigen's blocked products diverge from ascending-loop accumulation at the NN shapes (`lstm_input_gemm` k=23: 8750/19200 elements ~1 ULP off; `dense_gemm` k=48: 1890/2400; the recurrence GEMV and softmax row-sum are bit-exact), and the layer-0 input GEMM seeds the recurrence, so the divergence propagates through every timestep. The oracle strategy is therefore **reimpl-with-probes**: the harness LINKS the real vendored NN translation units (`BLSTMNeuralNetwork`/`LSTMLayer`/`NeuronLayer` + link-only `SRNLayer`/`CWRNNLayer`/`CostLaw`/`Rprop`), dumps the goldens from a diff-reviewed harness-local ascending-loop reimplementation of the forward chain, and PROBES the real compiled classes against it at every site (`manifest.json:nn_product_probes` + the NN_TOL records): all synthetic-net sites match at 0 ULP (incl. hidden states, double-width two-sweeps outputs, and the OverLap NaN rows), while the real-net sites show the expected small GEMM-order deltas (fullseq output 7 ULP, hidden states 735/2076 ULP at ~2e-15 absolute, E2E posteriors 15/5 ULP, E2E hidden states up to 32792 ULP at <= 2.554e-15 absolute for `e2e_full_bwd`) -- recorded in the manifest as Phase 4 calibration data, the same absorb-don't-hide posture as the Phase 1 DCT substitution. The **end-to-end real-net gate** (`phase2_e2e_gate.rs`) chains the Phase 1 feature pipeline under the REAL `1_worker_1.config` keys (a `201 x 11` input: `3*nb_DCT - 1` with `IgnoreFirstDCT`, no LTSV column) into the real 33,671-weight net and asserts the posteriors bit-exact vs the reimpl -- surfacing one real finding: the config's 11-wide features feed a net trained for 23 inputs, silently tolerated by the legacy width-tolerance branch (12 dead layer-0 weight rows; see `IMPROVEMENTS.md`). Provenance: the LSTM/dense/container/BLSTM forwards are compiled-TU-probed as above; the cost-quirk and in-place-mutation semantics are expression-coded unit tests; activations and the weight seam are pinned by harness dumps plus hand-computed boundary tests; a scalar-loop Python oracle (`nn_reference.py`) cross-checks the Rust layers bit-exactly from a structurally different implementation. Note: MATLAB's `BLSTM_Forward.m` was NOT the oracle and was not ported -- it implements different semantics than the engine (`sigmoid(x)` gates without the 0.1 pre-scale, no output softmax) and cannot match it; `nn_reference.py` implements ENGINE semantics instead.
- **Phase 2b - segmenter wiring (done).** Wires the per-file segmentation drivers into the ported layers: the real `Segmenter` trait, the `Segmenter`-base residue (`DriverConfig`, `results_to_segmentation`), the `run_pipeline` lift into `features::pipeline::build_input_sequence`, and all four drivers in `tasks/sad.rs` (Algo 1-4). Oracle strategy is the **SegProbe pattern**: the harness now LINKS six real segmenter translation units (`Segmenter.cpp`, `Segmentation.cpp`, `BLSTMSpectralSegmenter.cpp`, `BLSTMSignalSegmenter.cpp`, `LongTermSpectralVariation.cpp`, `TimeDomainCorrel.cpp`) and harness-local probe subclasses call the REAL compiled protected machinery (`buildFromConf`, `updateSegmentation`, `smoothSegmentation`, `results2segmentation`, param derivation, `getLTSV`, `getBLSTMInputSequence`, `getPitch`, `toFile_VRCTS`) -- only the `getSegmentation` bodies are transcribed, and only where an NN sits in the chain are the `feedForwardBackward` calls swapped for the existing bit-stable reimpl family. Why the swap instead of a real-Eigen bit-golden: the NN posteriors sit within 2-4e-16/frame of the decision thresholds on real data, boundary times are LINEAR INTERPOLATIONS of those posteriors (~1e-15 s jitter per crossing), and the hysteresis/area comparisons are DISCRETE -- a sub-ULP forward difference can flip a whole segment into or out of existence, so the real-Eigen forward is only trustworthy as a STRUCTURE oracle, never a bit one. `TdcSegmenter` (Algo 1) and `LtsvSegmenter` (Algo 2) have no NN in the chain at all, so their goldens run against the REAL compiled `getSegmentation` directly (LTSV's `nb_DCT=4` branch swaps only `applyDCT` for the ascending-loop `applyDCTLoop`, the same Eigen-GEMM-divergence workaround from Phase 1/2). `BlstmSignalSegmenter` (Algo 4) is the FIRST driver with the NN in the chain -- the raw 1-column signal feeds `nn/blstm.rs`'s `feed_forward_backward` (the real 23-input net runs on it via the LSTM input-projection `topRows` width tolerance) -- so its oracle is a REIMPL-SWAP (`SignalProbe`): `getSegmentation` transcribed faithfully, only the FFB call swapped, plus a SECONDARY real-forward structural probe (reimpl vs real-Eigen segment count+types must match exactly or the harness aborts; boundary max-dt recorded -- all three variants measured `0.0`). On this excerpt the real net's posteriors stay sub-threshold (~0.002-0.03 vs a 0.6 rising threshold), so every signal variant collapses to the untouched always-Other seed hypothesis `[Other@0, End@dur]` -- the NN chain itself is still pinned bit-exact via `last_result_rows`, but the decision layer is NOT exercised by these fixtures; the result-row pins carry the parity load here, not the boundaries. Three variants: `window0`, `overlap` (the only non-broken degenerate regime -- signal overlap mode is broken-as-committed for `window_shift > ssr`, see `IMPROVEMENTS.md` `signal-overlap-oob`), and `noOverlap` (the genuinely-stateful `_WindowShift = 0.0` poisoning, pinned by a two-files golden). `LtsvSegmenter`'s `is_log_mel=true` configs (the primary `ltsv.config` and its DCT/tiny siblings) are similarly decision-layer-blind on this excerpt -- the LTSV mean-floor assumes a power-scale periodogram, so log-scale input blows the score up to ~1e51 and every frame reads as SPEECH (see `IMPROVEMENTS.md`); the power-scale `ltsv_powermel` variant exists specifically to exercise real threshold crossings (a genuine SPEECH/OTHER/SPEECH structure on channel 1). `BlstmSpectralSegmenter` (Algo 3, the REAL `1_worker_1.config`'s algorithm) is COMPLETE including the pitch-homothety second pass (`BLSTMSpectralSegmenter.cpp:757-805`): `get_pitch` over the pass-1 segmentation (NN-free), `apply_homothety` warping the periodogram by `pitch/300`, re-filterbank/DCT, and a re-forward that rebuilds the input with the OLD (pass-1) LTSV column (a load-bearing quirk -- WIRED but unexercised on this config, since the real config's `LTSVwindow 0` means no LTSV column exists in either pass; see `IMPROVEMENTS.md`), then `clear_hypothesis` + re-segment. Its oracle `SpectralProbe` transcribes the full `getSegmentation` (calling `getTDCParam`/`getPitch` REAL, both NN-free) and swaps only the FFB calls for the reimpl, plus the same SECONDARY real-forward structural probe (all four variants `SEG_STRUCT max_dt=0.0`). UNLIKE signal, the real net genuinely crosses the rising threshold here, so the full decision layer (rising crossing + interpolation + hysteresis area) is exercised non-trivially, including the pitch pass: the `pitch` variant (TDCwindow 0.032, measured pitch ~250 Hz) is NON-VACUOUS -- the warp moves the chan-1 boundary from pass-1 1.1831 s to pass-2 1.2597 s, pinned by `spectral_pitch_*` goldens. The legacy dump quirk (externalized result stays PASS-1 while the boundaries are PASS-2) is reproduced and pinned. Separately, the Phase 0b-ii VRCTS-writer byte-golden deferral is now CLOSED: the harness's `iof::fmtr` shim was upgraded from inert to a FAITHFUL mini-formatter (`%f.Ns` == `std::fixed` + `setprecision(N)`), making the real compiled `toFile_VRCTS` itself a byte-golden source -- `to_vrcts_string` (the Rust writer) is asserted byte-for-byte equal to it across all four drivers' dumps (nine `.xml` fixtures total), closing the deferral without waiting on an end-to-end legacy run. Algo 1-4 are all landed; LID (Algo 5/6) + `TwinBLSTM` defer to Phase 4.
- **Phase 3 - network-level training (done).** The complete network-level BPTT + iRPROP- chain: `CostLaw` backward (`deriv()` per law, `computeUnitaryDeltas`/`computeDeltas` incl. the softmax+CE fusion and the `BackPropWER`/ponderation scaling -- already ported alongside the Phase 0b-i forward, now VALIDATED), `LstmLayer`/`NeuronLayer::feed_backward` (peephole BPTT with the O->state->C->F->I gate-derivative order; dense weight/bias deriv accumulation with NO output-activation Jacobian -- the fusion lives in `cost.rs`, proving the double-count trap doesn't bite), the `Network<L>` container backward (reverse-order layer loop, `InvSubSample` decimated-row re-expansion -- the SubSample crux resolved: rows are RE-EXPANDED, not dropped, though the floor's trailing `T mod R` frames stay gone), and the full `BlstmNetwork` backward (`feed_backward`/`feedBackwardDouble` fan-out, all 4 windowed drivers' backward halves, `get_weights_derivatives` Nx2 assembly, `reset_weights_derivatives`, `update_weights`). Oracle strategy is a **four-tier ladder, strongest first**: (1) `Rprop::updateWeights` probed against the REAL compiled class directly, strict bits on every platform, no canary (pure scalar, no libm/GEMM); (2) `CostLaw` backward probed against the REAL compiled class (polynomial laws bit-exact everywhere, Log/Sqrt canary-gated); (3) layer/container/BLSTM backward via a harness ascending-loop reimpl + NN_TOL probes of the real classes (all synthetic-shape sites `max_ulp=0`; real-net/output-ratio sites recorded as Phase 4 GEMM-divergence calibration, same blocked-vs-ascending story as the Phase 2 forward); (4) an independent Python engine-semantics backward oracle (`nn_reference.py`'s `lstm_backward_oracle`/`dense_backward_oracle`/`blstm_backward_oracle`) cross-checked bit-for-bit against the same harness fixtures the Rust suite consumes -- transitively triangulated through the shared fixtures, not a live PyO3 cross-call (the `speech-py` bridge is still an 18-line stub; that binding is Phase 4). The **real-net E2E gradient gate** (`phase3_e2e_grad_gate.rs`) chains the real 33,671-weight net's Nx2 gradient bit-exact vs the reimpl, then drives two `Rprop::update_weights` calls: the first step matches the real compiled trainer bit-exact, the second (cost negated) fires the cost-gated backtrack on 24409 of 33671 elements -- a genuine, non-vacuous majority, not a corner case. The network-level central-difference **grad check** (`phase3_gradcheck.rs`) spot-checks the real net's analytic gradient against numerical central differences on 10 weights spanning every sub-network region: measured relative errors run `3.8e-11` to `1.4e-7`, comfortably inside the documented `5e-4` bound (mean-vs-max bound choice noted in the doc-comment). What's still deferred to **Phase 4**: the corpus epoch loop and cross-file gradient reduction (the legacy's OpenMP-critical sum is order-nondeterministic -- the Rust reduction must be deterministic when it lands, tracked in `IMPROVEMENTS.md`), `BagOfProcessors::saveAndUpdate`'s best-weight lifecycle, the corpus-level grad-check CLI mode (with its documented result-column indexing hazard), the PyO3 seam, and the Python optimizer zoo (SMORMS3/QuantumPSO/CMA-ES stay stubs).
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
