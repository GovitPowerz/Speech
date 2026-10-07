# Architecture

One page for a reader who has built the binary and wants to know how a run and a training step actually flow through this code. Vocabulary is [CONTEXT.md](../CONTEXT.md)'s; the decisions this page rests on are the [ADRs](adr/); per-module detail is the README next to each module ([src/rust/README.md](../src/rust/README.md), [src/python/speech/README.md](../src/python/speech/README.md)); the evidence is [docs/validation.md](validation.md) and the numbers [RESULTS.md](../RESULTS.md).

## Two languages, one seam

- **Rust** (`src/rust/`) is the engine: audio I/O and the feature front-end, the recurrent nets and their analytic backward, the per-file SAD/LID drivers, the decision layer, the corpus fold, the file formats, and the whole fast and streaming tree. It is a library crate plus a CLI (`speech -m <config>`, `speech stream`, `speech stream-lid`, `speech bench`).
- **Python** (`src/python/speech/`) is everything around it: seeded weight init, the training loop, the legacy outer search, cost assembly, batching, the DCF/LID evaluation harness, corpus preparation, the lifecycle CLI.
- They meet at **one seam**, the PyO3 crate `speech-py` (module `speech_rs`): `Engine` (construct from a config, `run`, read results and per-net gradients, `set_weights`, `grad_check`), `StreamingSession` (chunked SAD) and `StreamingLidSession` (per-utterance LID). Every numpy crossing is a copy; the GIL is released around every `run`. The flat weight layout and the `adim` scaling live once, in Rust, and Python consumes them through the seam rather than re-deriving them.
- The legacy's own seam (a flat `.config`, little-endian `.bin` weight vectors, listing CSVs, `.mat` results) is kept as a byte-compatible serialization layer, which is what lets the committed 2015 fixtures drive the port.

## A SAD run in ten lines

`speech -m config.toml` builds a `CorpusProcessor` and calls `run()`:

1. `toml_config` flattens the TOML through the one `KEY_TABLE` into the legacy key shape (ADR-0006); `engine::corpus` parses the listing and the class mapping into corpus items.
2. `engine::bag_of_processors` builds one driver per config (the `Processor` enum: seven exact drivers, two fast ones), dispatching on `Algo` and on `Inference_Path` at one site (ADR-0002).
3. Per file, per channel: `audio::read_audio` decodes and normalizes (`(2*RMS+max)/2`, or the frozen `Audio_fixed_gain`), applies pre-emphasis, dither and windowing coefficients.
4. `features::pipeline::build_input_sequence_parts`: framing, the exact two-real-packed FFT periodogram (`features::fft`), the Mel bank, DCT-II MFCCs and regression deltas (`features::mel`), optionally the LTSV column; the fast path runs the same stages in f32 on `realfft` and `faer` (`fast::pipeline`, `fast::mel32`).
5. `nn::blstm::BlstmNetwork::feed_forward_backward` under one of the four windowed drivers (or the plain whole-sequence forward for a causal net): type-1 input normalization, the two `Network<CellLayer>` stacks (ADR-0005), the HCAT, the output MLP (asinh hidden, softmax out), the cost law and, when a reference is attached and backprop is on, the analytic BPTT.
6. `tasks::segmenter::results_to_segmentation`: the temporal convolution gate, then the hysteresis-with-area double threshold (`update_segmentation`) producing raw segments, then the fixed 8-step smoothing.
7. `tasks::segmentation_io`: the VRCTS XML dump, the reference load (STM / CSV / VRCTS), `compute_errors` (Pfa / Pmiss / error rate per class, coverage, delay).
8. The per-channel result (`engine::channel_result::ChannelResult`, the typed value the 18+N result row is rendered from) joins the lane's fold (ADR-0007); at epoch end the ascending-index fold sums its fields per config, the best-weight save gate fires, and in training mode the in-engine iRPROP- update runs on the count-normalized gradient.
9. `io::matfile` writes the `MultiConfigResults.mat` (a hand-rolled v5 writer; there is no reader).
10. The LID Twin (Algo 6) wraps steps 3-7 twice: the SAD net's hidden states feed the LID net per speech segment, or in Mode 7 the LID net scores precomputed phSeq/cep utterances with the SAD net frozen; the per-file language scores carry their target in-band (the `>150` sentinel, the `-200` offset), decoded once at the bag into `LidResult` and re-encoded only for the `.mat`.

The SAD pitch second pass (Algo 3, `TDCwindow > 0`) repeats steps 4-6 on a periodogram warped by the pass-1 pitch estimate.

## A training step in eight lines

`uv run python -m speech.drivers.baseline sad --cell-type cfc --direction forward ...` (`drivers/baseline.py`) or the modern loop directly (`drivers/train.py::train_modern`):

1. `dataprep/lre.py` derives the listings, the 12-class mapping and the split from the local corpus; nothing corpus-derived is committed.
2. `config_bridge.nnet_spec` reads the architecture (sizes, cell, direction, per-cell geometry) out of the same config the engine will read.
3. `init_weights.py` seeds one flat pack per net (Xavier or He through the packer the goldens already test for the LSTM; the Rust flat order emitted directly for the four new cells, pinned block by block).
4. `engine.forward_backward` is `ComputeGradient.m`'s contract over `speech_rs.Engine`: set weights, `run` one fold at `Epochs 0`, read the channel results by name (`Engine.channel_results()`) and the per-net derivatives, assemble `f = NNCostSeg (+ NNCostLID)` and the count-normalized gradient (plus the LID-only L2 term).
5. `optimizers.Smorms3` takes the step (Octave-pinned bit-for-bit); optionally `batching` rotates a hard-example mini-batch listing per evaluation and rebuilds the engine against it.
6. Per epoch, a forward-only validation on the held-out split with the moving `NNCostSeg` signal; early stop on patience; `best_<net>.bin` / `last_<net>.bin` / `train_history.json`.
7. Held-out scoring: for SAD the engine dumps one VRCTS hypothesis per file and `evaluate.dcf` pools them per collar (the NIST perl scorer, ported value-for-value); for LID the `.scr` score files feed `evaluate.lid_error` and `evaluate.cavg`.
8. `run_metadata.json` records seed, lane count, subset spec, config hash, and the git SHA + dirty flag of the tree the run started on (stamped before any listing is written, so a mid-run edit or commit never becomes the record's provenance and a git failure stops the run before its directory exists); a second run at the same seed is byte-identical, which the subset gates assert.

The legacy outer loop (`quantum_pso` over the vec2struct genome, with the inner SMORMS3 hook) is kept as a second driver; since Phase 5 the genome carries DSP hyperparameters only, weights masked out permanently.

## The three implementation axes

Every forward can be placed on three axes, and a gate holds each pair together:

| axis | values | what holds them together |
|---|---|---|
| precision | exact f64 (`nn/`, `features/`, `tasks/`) vs fast f32 (`fast/`) | boundary count / type identity and `max_dt == 0.0`, argmax zero flips, metric deltas exactly `0.0` on real data; posterior deltas measured then pinned (`phase7_parity_*.rs`, `phase9_fast_parity.rs`, `phase10_bicell_parity.rs`) |
| timing | offline whole-file vs streamed chunks (`fast/stream.rs`, `fast/stream_lid.rs`) | streamed `finish()` bit-for-bit equal to the offline run under the same frozen norm, chunk-invariant, zero retractions (`phase8_gate.rs`, `phase9_stream_causal.rs`, `stream_incremental_resmooth.rs`) |
| cell x direction | `{lstm, slstm, mamba, cfc, transformer}` x `{bidirectional, forward}` (`nn/cells/`, `fast/cells.rs`, `fast/bicell.rs`) | a hand-derived backward against central differences per shape and seed, a corpus-level gradient check through the seam, and a from-scratch subset gate per row (`phase9_cell_grad.rs`, `tests/pyo3/test_phase9_seam.py`, `test_phase10_gates.py`) |

Only the exact tree trains; the fast tree is inference-only and bails loudly otherwise. Only a forward (causal) net streams; a bidirectional net is refused at session construction. The fast matrix is total: `classify_fast_shape` has no bail arm left, and a future cell without a kernel is a compile error.

## Where the time goes

`speech bench [--repeat=N] [--path=exact|fast] [--json] [--label=NAME] <config>` prints one line per run (`wall_s`, `audio_s`, `rtf`, `maxrss_mb`), or with `--json` the one document per invocation that `python -m speech.ledger bench` wraps into a ledger record (ADR-0009); `benches/kernels.rs` times the three exact kernels against their fast twins. On the committed fixtures and the corpus-gated legs (one machine, named in `RESULTS.md`'s Phase 7 table) the fast tree runs SAD 4.9x and LID 3.4x faster end to end at a real-time factor near `5e-4`, the `realfft` periodogram 23x per call and the `faer` projection 12x; the fast SAD path peaks at 44 MB of RSS against the exact tree's 57. A streamed SAD run's cost per push is bounded (the incremental re-smooth replays a window of 3-6 segments regardless of stream length).

## Where the numbers come from

`RESULTS.md` is the living record: the baseline protocol (corpus, metrics, seeds, lane count), the subset-gate rows per arm with their untrained-init baselines, the launcher commands for the full-corpus runs and the rows waiting for them, then one section per phase from 7 on (bench tables, latency decompositions, gradient-check residuals, memory before/after, every mutation battery's catcher map with its gaps). Every number names the config lineage it was measured on (ADR-0008) and, where a pin exists, the measurement the pin was set from.
