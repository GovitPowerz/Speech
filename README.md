# Speech

[![CI](https://github.com/GovitPowerz/Speech/actions/workflows/ci.yml/badge.svg)](https://github.com/GovitPowerz/Speech/actions/workflows/ci.yml)

A Speech Activity Detection (SAD) and spoken Language Identification (LID) engine, with its training loop. A **Rust engine** takes audio (or precomputed features) through a feature front-end, a recurrent net and a hysteresis decision layer to speech segments and language decisions, offline or streamed chunk by chunk; a **Python package** trains it from scratch and evaluates it on a real corpus, driving the engine in-process through one PyO3 seam. The engine began as a bit-exact port of a 2015 C++/MATLAB production system, was proven against the resurrected original, and was then taken past it: a fast f32 inference tree, a streaming mode whose output is bit-identical to the offline run, and four new recurrent cells (sLSTM, Mamba, CfC, a windowed transformer) behind one extension seam, each gradient-checked and trained from scratch.

## Highlights

- **A numerical port proven, then frozen.** The Rust engine reproduces the 2015 production binary: segment boundaries byte-identical in the legacy XML format, scored result columns within a maximum relative delta of about 1e-15, two real 2015 weight packs repacked byte-for-byte. The proof is the annotated tag `legacy-parity-v1`; from there the goldens flipped to the port's own truth and eleven legacy bugs were fixed under a stated protocol, each with a mutation that reverts it ([ADR-0001](docs/adr/0001-port-truth-after-the-parity-tag.md), [IMPROVEMENTS.md](IMPROVEMENTS.md)).
- **Two implementations of the same arithmetic, held together by identity gates.** An f32 fast tree (`faer` SIMD matmuls, a native real FFT, no system BLAS) runs SAD 4.9x and LID 3.4x faster end to end with every segment boundary, every argmax and every real-data metric identical to the exact f64 tree; posterior deltas are measured, then pinned at ten times the measurement, and a breach stops the work rather than widening the pin ([ADR-0002](docs/adr/0002-the-exact-tree-is-frozen.md), [ADR-0003](docs/adr/0003-decisions-are-gated-on-identity-numbers-are-pinned.md)).
- **Streaming that changes timing, never arithmetic.** The two whole-file statistics that block causality are frozen rather than approximated, offline and streamed runs drive the same step kernel, and a settled-prefix emission rule delivers segments mid-stream with zero retractions. A causal cell removes the lookahead term and takes the derived speech latency bound from 6.05 s to 2.82 s on the same config ([ADR-0004](docs/adr/0004-streaming-changes-timing-never-arithmetic.md)).
- **Five recurrent cells behind one seam, each with the same evidence.** The legacy peephole LSTM, sLSTM, Mamba (S6 in recurrent form), CfC and a windowed causal transformer with ALiBi are one `CellLayer` enum; adding a cell touches nothing outside it, and forgetting its f32 twin is a compile error. Every cell lands with a hand-derived backward checked by central differences per shape and seed, a corpus-level gradient check through the seam, derived-then-pinned dead weight blocks, exact-vs-fast parity, a streaming row and a from-scratch training gate ([ADR-0005](docs/adr/0005-the-cell-is-the-extension-seam.md)).

## What the gates measure

| claim | evidence | where |
|---|---|---|
| Port parity with the 2015 production binary | VRCTS boundaries byte-identical; result columns max relative delta ~1e-15; packer byte-identical on two real 2015 packs (33,671 and 42,911 weights) | `legacy-parity-v1`, `tests/reference_data/phase4d/` |
| Fast tree, quality | boundary count / type identical, `max_dt` exactly 0.0, argmax zero flips; DCF / LID error / Cavg deltas exactly 0.0 on three real-data checkpoints | `src/rust/tests/phase7_parity_*.rs`, `tests/pyo3/test_phase7_parity.py` |
| Fast tree, cost (Apple M4 Pro) | SAD 4.93x, LID 3.43-3.63x end to end; `realfft` 23.4x per call, `faer` projection 12.4x; fast SAD peak RSS 43.5 MB vs 57.2 exact | [RESULTS.md](RESULTS.md), the Phase 7 bench matrix and micro-benches |
| Streaming equivalence | streamed `finish()` bit-for-bit equal to the offline frozen run on every cell, chunk-invariant at 20 / 100 / 1000 / 7 ms, zero retractions; a real retraction found on corpus data and fixed | `src/rust/tests/phase8_gate.rs`, `phase9_stream_causal.rs`, `tests/pyo3/test_phase8_parity.py` |
| Cell gradients | central differences vs the analytic backward per shape x 3 seeds, worst discriminating residual 7.4e-6 (Mamba), 2.9e-7 (transformer), 5.5e-8 (CfC), all pinned under a 1e-4 ceiling; dead blocks pinned exactly | `src/rust/tests/phase9_cell_grad.rs`, `tests/pyo3/test_phase9_seam.py` |
| From-scratch training | 10 of 10 cell x direction rows on the v2 SAD lineage beat their untrained init at every collar, run-twice byte-identical; the 12-class LID arms beat chance and init on held-out subsets; the LID cell knob is probed over all 20 (arm x cell x direction) legs and trained on `slstm / forward` per arm | `tests/pyo3/test_phase10_gates.py`, `test_phase6_gates.py`, `test_lid_cells_gates.py` |
| Every test claims what it catches | a mutation battery per phase, scored catcher by catcher, gaps measured and recorded (Phase 11: 8 of 10 named catchers fire; the two misses are sub-pin symmetric substitutions) | [RESULTS.md](RESULTS.md), the battery sections |
| Every gate number has a record | 19 ledger records (one per subset-gate row: SHA, build, host, recipe, metrics), each produced by re-running its gate, none transcribed; the eight RESULTS.md gate tables are rendered from them and CI fails if the file and the ledger disagree; a `TBD` full-run row is filled by a committed runner under `experiments/` | `ledger/`, `tests/test_ledger.py`, [ADR-0009](docs/adr/0009-measurements-are-ledger-records.md) |

The from-scratch numbers are subset-scale and said so: on ten files a SAD net collapses from all-non-speech to all-speech (DCF exactly 0.75 to 0.25) with no partial-discrimination point between, and the 12-class LID arms reach 73% (acoustic) and 84% (phonotactic) held-out error against 92% chance. Genuine discrimination is the job of the full-corpus launcher runs, whose rows [RESULTS.md](RESULTS.md) holds open; nothing corpus-derived is committed.

## Where to start

1. **How it works**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Two languages and one seam, a SAD run in ten lines, a training step in eight, the three implementation axes and the gate that holds each pair together.
2. **The vocabulary**: [CONTEXT.md](CONTEXT.md). Raw vs smoothed segments, the exact and fast trees, the causality cut, pins and STOPs.
3. **The decisions**: [docs/adr/](docs/adr/). Port-truth after the parity tag, the frozen exact tree, identity gates and measured pins, streaming without arithmetic change, the cell seam, config keys declared once, deterministic lanes, two SAD lineages, measurements as ledger records.
4. **The evidence**: [docs/validation.md](docs/validation.md) for the oracle tiers and the standing gates; [RESULTS.md](RESULTS.md) for every number, bench table and mutation battery, its gate tables rendered from the records under [ledger/](ledger/).
5. **The code, module by module**: [src/rust/README.md](src/rust/README.md) and [src/python/speech/README.md](src/python/speech/README.md), one row per file with its legacy source and every reproduced quirk.
6. **How it was built**: [docs/ROADMAP.md](docs/ROADMAP.md) phase by phase; the dated specs and plans in [docs/superpowers/](docs/superpowers/README.md); [IMPROVEMENTS.md](IMPROVEMENTS.md) for every legacy behaviour kept or fixed; [CLAUDE.md](CLAUDE.md), the agent brief.

The code is written largely with coding agents; [DEVELOPMENT.md](DEVELOPMENT.md) says which decisions stay the author's, what evidence a change needs, and how merges stay human-only.

## Quick Start

```bash
# Build the Rust engine (binary `speech`)
cd src/rust && cargo build --release && cd ../..

# Run a corpus through a TOML config (SAD: Algo 1-4; LID: Algo 5-6)
./src/rust/target/release/speech -m configs/lid/lid_blstm.toml

# Stream a MONO wav through the online SAD session; one `SEG` line per finalized
# segment, flushed at emission time, then a `STREAM` summary. The config carries the
# causality cut ([audio] fixed_gain) and, for a causal cell, [nn] direction = "forward"
# with [spectrum] frame_window = 0.
./src/rust/target/release/speech stream --chunk-ms=100 <config.toml> <mono.wav>

# Stream a phSeq/cep utterances file through the online LID session (utterance-granular).
./src/rust/target/release/speech stream-lid <config.toml> <utterances>

# Benchmark a config on either tree (one parseable BENCH line per run)
./src/rust/target/release/speech bench --repeat=3 --path=fast <config.toml>

# Build the PyO3 bindings (module speech_rs)
uv run maturin develop --release --manifest-path src/rust/speech-py/Cargo.toml

# Set up the Python environment (Python >=3.14)
uv sync --group dev

# Train a SAD net from scratch (needs the local corpus; every size derives from the config)
uv run python -m speech.drivers.baseline sad-v2 --cell cfc --direction forward --corpus-root <corpus> --out-dir runs/cfc_fwd

# Fire a full-corpus row of a RESULTS.md table and promote its record (the runners carry the recipes)
experiments/05_lid_cells.sh lid-phseq/mamba/forward

# Run tests
cd src/rust && cargo test && cd ../..
uv run pytest tests -m "not slow"
uv run pytest tests/pyo3          # the seam suites; corpus-gated legs skip without the corpus
```

## What it does

Two tasks, built on a shared feature front-end and neural-net zoo.

- **SAD (Speech Activity Detection).** A recurrent net emits a per-frame speech posterior; segmentation turns that posterior into speech/non-speech segments via a hysteresis-with-area double threshold plus an 8-step smoothing pass. Pure decision logic, golden-tested first.
- **LID (spoken Language Identification).** A TwinBLSTM head consumes the SAD net's hidden states (the "twin" seam), accumulates per-segment evidence, and produces a per-file language posterior (argmax + confusion). LID is stacked on top of SAD, not run independently.

Underneath both:

- **Feature front-end:** framing/windowing, radix-2 FFT periodogram (two-real packing, custom Welch normalization), triangular Mel filterbank (1125/700 scaling) with DCT-II MFCCs and deltas/delta-deltas/SDC, LTSV (entropy-variance), and time-domain pitch/autocorrelation (TDC).
- **NN zoo:** from-scratch recurrent nets over a `Layer` trait, with the cell selected by config (`CellLayer`, Phase 9) and the direction bidirectional or forward/causal: the legacy peephole LSTM (12-row peephole, `[i|f|o|g]` gates), **sLSTM** (exponential gating with a running-max stabilizer and a twin `(c, n)` normalizer), **Mamba/S6** in recurrent form (selective per-timestep `Delta`/`B`/`C`, depthwise causal conv, RMSNorm), as of Phase 10 **CfC** (the closed-form continuous-time cell: a `lecun_tanh` backbone feeding two candidate heads blended by a learned time-interpolation gate) and, as of Phase 11, a **transformer** (a pre-norm block over a causal sliding window with ALiBi relative positions -- no learned position parameters, no whole-sequence attention), all feeding a dense output MLP. Non-standard activations throughout (asinh cell/output, `sigmoid(0.1z)` gates, softmax). The legacy's SRN, Clockwork-RNN and convolution layers were NOT ported -- SRN/CWRNN are dead code in the original (compile-time instantiations only) and the conv layer is broken-as-committed; see `IMPROVEMENTS.md`.

## Configuration

TOML is the canonical input format. A run config names the algorithm and its DSP/decision parameters; the engine is launched on a single config file (`speech -m <config.toml>`).

```toml
# configs/lid/lid_blstm.toml  (Twin-BLSTM LID, legacy Algo_choice 6)
[engine]
algo_choice = 6
num_outer_threads = 1
num_inner_threads = 1
results_file = "MultiConfigResults.mat"

[audio]
offset = 0
max_duration = 120

[decision]
thresh_rising = "0.6"
area_rising = "0.0"
thresh_falling = "3.795068189162301e-01"
area_falling = "0.0"

[spectrum]
order = 9
shift = "2.500000000000000e-02"
temporal_convolution_type = "hamming"

[preprocess]
preemph_ratio = "-9.700000000000000e-01"
noise_seed = 3
noise_ratio = "2.921428128844371e-02"
```

Integers and booleans are native TOML; every other value (floats, comma lists, words, paths) is a quoted string carrying the legacy decimal text verbatim, and a native TOML float is rejected so no value is silently re-rounded.

A legacy `.config` importer (`legacy_config.rs`, `configs/legacy/`) parses the original flat whitespace format (with its `_`-continuation, last-wins, and `val*count` repeat quirks) so validation runs can be driven from the exact goldens the legacy consumed; a `.config` is accepted everywhere a `.toml` is ([ADR-0006](docs/adr/0006-config-keys-are-declared-once.md)). TOML is the format going forward; the importer exists for parity.

## Validation

Parity against the legacy was established as a ladder, each rung golden-tested bit-for-bit against an oracle before the next was trusted: features, the weight codec, the NN forward, the analytic gradient, one optimizer step, the decision and scoring layer, then whole-corpus runs against the resurrected 2015 binary. Since then every change is held to the standing gates: decisions identical, numbers pinned, streaming bit-equal, gradients finite-difference-checked, and a mutation battery per phase. [docs/validation.md](docs/validation.md) has the oracle tiers and the gates in full; [RESULTS.md](RESULTS.md) the measurements.

## Roadmap

Both roadmaps are closed: the port (phases 0a to 4d, tag `legacy-parity-v1`) and the post-port era (phases 5 to 11: port-truth fixes, from-scratch training on real data, the fast tree, streaming, four new cells and a second SAD lineage). [docs/ROADMAP.md](docs/ROADMAP.md) records what each phase delivered and what it left named. The one open item is the user-fired full-corpus baseline runs, whose numbers land in [RESULTS.md](RESULTS.md) as they complete. Open work is tracked in GitHub Issues.

## Project Structure

```
Speech/
  README.md  CLAUDE.md  CONTEXT.md  DEVELOPMENT.md  CITATION.cff  LICENSE
  RESULTS.md  IMPROVEMENTS.md
  ledger/{baseline,bench}/*.json          # the measurement ledger: one record per measured number (ADR-0009)
  experiments/{00..05}_*.sh  lib.sh        # the runners: one per RESULTS.md table with a TBD full-run row, recipe hard-coded, record promoted
  pyproject.toml  uv.lock
  setup_env.sh  build.sh  check_all.sh  lint_code.sh  upgrade_dependencies.sh
  .github/workflows/ci.yml
  .claude/settings.json                    # the agent permission policy (DEVELOPMENT.md)
  docs/
    ARCHITECTURE.md  ROADMAP.md  validation.md
    adr/                                   # the nine decision records
    agents/                                # issue-tracker, triage-label and domain-doc contracts
    superpowers/{specs,plans}/             # one dated design spec + plan per phase (README.md indexes them)
  configs/{lid,training}/*.toml            # TOML canonical; configs/legacy/ holds a sample .config
  data/                                    # random_tables.bin (from Constants.h); the corpus is local and git-ignored
  legacy/                                  # git-ignored: the vendored C++ src/ + Optimizer_V6.2.2 .m
  tools/                                   # local-only oracle harnesses (C++, Octave, Perl, the 2015 binary)
  scripts/                                 # the fixture extractors that drive the harnesses
  src/rust/                                # the engine (README.md: the module map)
    Cargo.toml  Cargo.lock  build.rs       # workspace: speech (lib+bin) + speech-py; build.rs bakes in the build provenance
    src/
      lib.rs  main.rs  cli.rs  config.rs  legacy_config.rs  toml_config.rs  constants.rs
      engine/{corpus_processor,bag_of_processors,channel_result,corpus,confusion}.rs
      audio.rs  cost.rs  bench.rs  stream_cli.rs  stream_lid_cli.rs
      features/{fft,mel,ltsv_tdc,stats,pipeline}.rs
      nn/{network,blstm,layers,activations,train}.rs
      nn/cells/{slstm,mamba,cfc,transformer}.rs   # the CellLayer seam
      tasks/{segmenter,sad,lid,segmentation,segmentation_io,vrcts}.rs
      fast/{nn,mel32,pipeline,driver,cells,bicell,stream,stream_lid}.rs   # f32 fast + streaming
      io/{mod,binary,matfile}.rs
    tests/                                 # cargo integration suites, one file per phase gate
    benches/kernels.rs                     # criterion micro-benches
    speech-py/src/lib.rs                   # the PyO3 module speech_rs
  src/python/speech/                       # the optimizer / orchestrator (README.md: the module map)
    cli.py  config_bridge.py  weight_bridge.py  init_weights.py  engine.py  genome.py
    evaluate.py  optimizers.py  scoring.py  batching.py  nn_reference.py  features_oracle.py
    drivers/{state,init,train,retrain,test,baseline,spec}.py
    dataprep/{augment,opensad15,stm_normalize,lre}.py
    ledger/{schema,tables,render,cli,stage,bench,prose}.py   # the record schema, the RESULTS.md renderers, the staged bench legs, `python -m speech.ledger`
  tests/                                   # pytest suites + tests/pyo3/ (the seam) + reference_data/ (goldens)
```

## Testing / CI

```bash
./check_all.sh          # Rust: cargo test + fmt --check + clippy + release build
./lint_code.sh          # Python: ruff (imports, format, check) + mypy
uv run pytest tests     # Python test suite
```

GitHub Actions (`.github/workflows/ci.yml`) runs on PRs to `main` and manual dispatch (`workflow_dispatch`), with four jobs:

- **rust** - `cargo fmt --check`, `cargo clippy`, `cargo test`.
- **python-lint** - `ruff check`, `ruff format --check`, `mypy`.
- **python-test** - `pytest`, including the ledger gates: every record under `ledger/` validates and RESULTS.md holds exactly what the ledger renders (`python -m speech.ledger render --check`).
- **python-pyo3** - `maturin develop --release` for `speech-py`, an import check of `speech_rs`, then `pytest tests/pyo3` (the in-process seam-replay + exit-gate suites). The Phase 4d PyO3-seam parity gate against the committed 2015-binary fixtures was retired from HEAD in Phase 5 ([ADR-0001](docs/adr/0001-port-truth-after-the-parity-tag.md), [docs/ROADMAP.md](docs/ROADMAP.md)); its proof lives at the `legacy-parity-v1` tag.

## Author

Grégory Gelly. The engine ports a 2015 C++/MATLAB speech-processing system the author co-developed, then extends it with the inference, streaming and architecture work above. Contact: gregory.gelly@gmail.com.

## License

Apache-2.0, see [LICENSE](LICENSE). The license is also declared in both Cargo manifests, `pyproject.toml` and `CITATION.cff`.
