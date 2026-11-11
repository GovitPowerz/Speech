# ADR-0002: The exact f64 tree is behaviour-frozen; the f32 fast tree lives beside it, selected at one dispatch site

**Status:** accepted | **Date:** 2026-07-19 (Phase 7, PR #13); touch classes extended 2026-07-31 (Phase 9), 2026-08-11 (Phase 10), 2026-08-12 (Phase 11), 2026-10-06 (issues #32, #47, #50), 2026-10-07 (issue #23), 2026-10-09 (issue #60), 2026-10-12 (issues #26, #27), 2026-10-13 (issue #25), 2026-10-29 (issues #90, #94), 2026-11-02 (issue #121), 2026-11-03 (issue #83), 2026-11-05 (issue #87), 2026-11-06 (issues #134, #136), 2026-11-07 (issue #135), 2026-11-09 (issue #140), 2026-11-10 (issue #146), 2026-11-11 (issue #154)

## Context

The exact tree (`nn/`, `features/`, `tasks/`, `engine/`) is the port's proof: every committed
golden was dumped from a legacy oracle against it, and the committed suite passing byte-for-byte
is what the parity claim rests on. Phase 7 needed a fast inference path (SIMD matmuls, a native
real FFT, f32 throughout), and any arithmetic change inside the exact tree would have moved the
goldens it is proven by.

## Decision

The fast path is a **separate tree** (`src/rust/src/fast/`): f32 twins of the exact kernels,
selected by the `Inference_Path` config key (default `exact`) at **one** bag-construction site.
The exact tree stays frozen: its only sanctioned edits are named **touch classes**, each
behaviour-free on every exact-path run and proven so by the committed golden suite staying
byte-green (the bench plumbing, the dispatch site and its key row, the training and seam guards
on fast variants (one site, every mode, since issue #110, 2026-10-28), golden-re-verified hoists; later the `CellLayer` enum wrap, the forward-only
branches, inference-only retention gating, the exact-length pack guard, the malformed-value
guard on `nn/blstm.rs`'s defaulting config getters and on `cost.rs`'s `CostLaw::from_config`
reads: a present-but-unparseable key errors instead of silently taking the default, as the legacy
`read<T>` does for a value with no parseable prefix; the port is stricter on a parseable prefix
with trailing junk (`1O`, `false # c`), which the legacy's `ss >> val` reads as `1` / `false`,
issue #32; both refuse a non-finite value, and `cost.rs` also parses each `classes_ponderations`
entry strictly instead of dropping a bad one and shifting the later classes, and errors on an
unknown cost-law name instead of panicking, issue #47; the same guard extended to every f64
and list config read in the exact tree -- `features/pipeline.rs`, `tasks/sad.rs`,
`tasks/segmenter.rs`, `tasks/lid.rs`, `config.rs`, `engine/bag_of_processors.rs`,
`engine/corpus.rs` -- by routing them through the one `legacy_config.rs` reader family, so a
non-finite value errors naming the key everywhere and every list takes the legacy
`split_with_repeat` grammar (one trailing comma dropped, the `*` repeater) with three deliberate
tightenings, a bad repeat count, a third `*` part and whitespace inside a repeat form, all
errors where the legacy aborted, silently dropped or truncated, issue #50, and then every i32 / bool / String / usize / i16 / i64 scalar read -- the 24 private copies in `engine/bag_of_processors.rs`, `nn/blstm.rs`, `tasks/segmenter.rs`, `tasks/sad.rs`, `tasks/lid.rs`, `fast/plan.rs`, `features/pipeline.rs`, `tasks/vrcts.rs`, `engine/corpus.rs`, `config.rs`, `cost.rs` -- through the one generic `legacy_config::get<T>` / `get_default<T>` over `parse_token<T>` (trimmed, whole-string, one error wording naming the type, the key and the text), the one visible contract change being that the untrimmed copies now TRIM (`features/pipeline.rs`'s four getters, `nn/blstm.rs`'s `get_bool` / `get_i16` and its `_weightsFile` / `_Cell_Type` / `_Direction` reads (the last two through `FromStr` on `CellType` / `Direction`), the `_weightsFile` copy in `fast/plan.rs`, the string readers in the bag, `tasks/segmenter.rs`, `tasks/sad.rs`, `engine/corpus.rs` and `engine/corpus_processor.rs`'s `multiConfigResultsOutputFile`), observable only on a padded value through the in-memory `Engine.from_map` seam or a padded quoted string in a `.toml` (`parse_legacy_config` trims every `.config` value; `toml_to_map` copies a TOML string as written), behaviour-free on every committed config, issue #87; the typed channel
result: the result row's 18+N layout owned by one `engine/` type, `engine/channel_result.rs`,
with `to_row()` the only writer and the in-band LID target decoded once at the bag, proven
byte-identical by the `.mat` goldens and the seam cross-pin, issue #23; the pack-count guard: `engine/bag_of_processors.rs::set_weights` sizes the list against the arm's pack count (the driver's `pack_count` since issue #25) before indexing it, so a short list is a refusal naming both counts instead of a panic and a long one is no longer truncated in silence, algo 0/1/2 inert at any length, issue #60; the stepwise hoist: a per-step loop body moved into a
state struct (or a stateless per-output function) in the same module, the batch function becoming the fold over it, proven by the
committed goldens byte-green AND the pre-existing bit-equality leg passing UNMODIFIED through the
move -- `tasks/segmenter.rs::HystStep` under `update_segmentation_raw` (commit `476ffec`) and
`audio.rs::conv_at` under `convolution_horiz_slice` (commit `582b8a7`), the streaming session
then driving the same struct and the same function, issue #26; introspection: a method that
reads layer state and writes nothing -- the `weight_layout()` of every layer type in
`nn/layers.rs` and `nn/cells/`, `Network::layers()`, and `BlstmNetwork::weight_layout`, the
block list the `speech_rs.weight_layout` seam exports, proven behaviour-free by the goldens
byte-green and pinned to `set_weights` by one-hot probes per cell, issue #27, and the decided
segmentation retained on `engine/channel_result.rs::ChannelResult::hypothesis` (the sanitized
list `compute_errors` left, cloned in at the row's assembly, not on the wire) and read by the
`channel_results()` seam's `hypothesis` key, proven behaviour-free by the goldens byte-green
and pinned to the VRCTS dump by a byte-identical reprint, with `tasks/segmentation.rs::Segmentation`
deriving `Debug` and `PartialEq` for it (a derive, no body), issue #86, and the exact mel bank's
tables read once, `features/mel.rs::MelFilterBank::tables() -> MelTables<'_>` (`pub(crate)`, a
borrowed view of the private tables and their indexing scalars, nothing computed), the one
reader being `fast/mel32.rs::FastMelBank::from_exact`, which replaces the fast tree's f64
re-transcription of the constructor, with `features/pipeline.rs::build_mel_bank` widened to
`pub(crate)` so the fast pipeline builds the exact bank at the exact path's one site, proven
behaviour-free by the goldens byte-green and the two constructors measured bit-equal before
the deletion, issue #88; the driver
interface: `tasks/segmenter.rs::Segmenter` as the one surface the bag reaches every driver
through -- methods returning state a driver already holds (its trainable nets, its pack count,
one channel's result members, its posteriors) plus `set_packs`, the pass-through it replaces --
the three LID drivers' five member vectors regrouped into one `LidMembers` with every write site
renamed, the drivers' pass-through delegators, `score` copies and write-only `channels`
fields deleted, and `engine/bag_of_processors.rs` reading every driver through the trait:
the weight, statistics, derivative and back-prop reads iterate the trainable nets, save and
update take one `training_policy` keyed on the algo number as the legacy is (the same
criteria, update costs, `LID_` prefix and order); nothing recomputed, proven by the goldens
byte-green, issue #25; the same class, 2026-10-28: `Segmenter::is_fast_tree` deleted with
its default and both overrides, fast or exact being the bag's one stored fact, issue #110;
the same class, 2026-10-29: the Bag's and the drivers' caller-less getters, four
never-read fields (the Bag's `lock_files_prefix` with its `LockFilesPrefix` read, the Twin's
two threshold fields and the fast Twin's `mode`, whose keys stay parsed), the Bag's discarded WER percent block, two
copies of the speech-duration walk and the corpus processor's `top_rows` alias deleted,
nothing recomputed, the `.mat` goldens byte-identical, issue #94; the same class, 2026-10-29:
`BlstmConfig::input_size()`, the legacy `getInputSize` rule stated on the parsed config and
pinned equal to the built net's, read by the `speech_rs.blstm_config` seam where a config
declares no `NNetInputSize`, the net's constructor and getter untouched, issue #90; the
stepwise hoist again, 2026-11-02: the exact spectral preamble -- the stateful `_SpectrumShift` /
`_SpectrumShiftInFrames` / SAD `_WindowShift` re-quantization, `getBLSTMParam` and the SAD
time axis -- is ONE body, `tasks/sad.rs::SpectralCarry::advance`, folded by the spectral SAD
driver, the Algo 5 driver (its shared half: the signal-pattern scoring axis and the LTSV sizing
stay its own) and the Twin's three modes (the `!hasReadWavFile -> 80` override an argument
every caller keys on the audio, `non_wav_ssif`), the input gates one free function, the
epilogue reset one method, and the Twin's inline `getLIDBLSTMParam`, written three times,
`tasks/lid.rs::derive_lid_window` with the `&mut` write-back `get_blstm_param` has, which
`fast/plan.rs` now imports instead of its copy (the #26 direction: the fast tree calls the exact
body, never the reverse); the signal driver shares none of it and keeps its inline block;
proven by the goldens byte-green and the four phase2b lifecycle pins passing UNMODIFIED through
the spectral driver's delegating getters, issue #121 (the non-wav read those drivers never
reproduced is no hoist: it is a port fix recorded in IMPROVEMENTS.md, Algo 5 now taking the
legacy 80 and the spectral SAD driver and the Twin's modes 0-6 refusing it); and the
caller-less class the same day: the write-only `ltsv_shift_sec` deleted on both drivers with
its four per-file writes, the port's LTSV shift having derived from the config value since
Phase 2b, issue #121; the stepwise hoist a third time, 2026-11-03: the per-channel LID
scoring accumulator -- the confusion header seed, the per-block score, the strict `>` argmax
with its confusion update, the `langID` weighting, `segmentsCount`, the normalization and the
member row -- written four times on the exact tree (Algo 5, the Twin's modes 0-3, 4-6 and 7)
and once more on the fast tree, is ONE struct, `tasks/lid.rs::LidChannelAcc` (`push` per scored
block, `finalize` read-only), its three variation points enums (`BlockScore`, `BlockWeight`,
`LidNorm`), each `LidNorm` variant carrying its copy's guard literally (Algo 5 and modes 0-3
never tested the sum; a unification would be a fix-protocol entry, not a hoist); the fast Twin's
per-entry body and the utterance stream drive the same struct on the widened posteriors, so
`fast/driver.rs` lost its copy (the #26 direction again), and the target clamp is one body,
`clamp_target_index`, read by the four drivers and the fast plan; proven by the four LID goldens
byte-green and `phase7_parity_lid` / `phase8_stream_lid` passing with their assertions unmodified
(one doc comment in the latter renamed to the moved symbol), with the corollary
the `issue83` battery measures: once both trees run the one accumulator, the fast-vs-exact
parity legs are self-consistent for it and its arithmetic is owned by the exact LID goldens
alone, issue #83; and a read-only accessor over existing state, 2026-11-06: the four per-epoch
result arrays `CorpusProcessor` already held become one value, `engine/corpus_processor.rs::EpochMems`,
sized and filled where they were and handed out by `epoch_mems()` (the `speech_rs.Engine.epoch_mems()`
dict reads it under the `.mat` variable names), no exact-path number moved, proven by the seven
in-process goldens re-pointed from the `.mat` to it staying byte-green against the unchanged `.bin`
fixtures and by the bridge test pinning the accessor bit-identical to the `.mat` the same run
wrote, issue #134 (the bridge test retired with the `.mat` in issue #135); and a file side effect removed, 2026-11-06: the best-weight save
(`nn/blstm.rs::save_weights`'s `weights_` / `weightsDerivatives_` `.bin` pair and its stats
`.mat`, the `bestNNWeight_<n>_` / `LID_` naming, `run_epoch`'s input-statistics fold that fed only
the stats file) is deleted while the decision stays: the bag's `best_cost_gate` keeps the
per-algo criterion, the strict `>` and the order before the update, and a fire captures each
net's pre-update pack and folded derivatives in memory under `test-support` (so in a production
build the gate's only effect is the running `bestCost`, the accepted state: the Modern loop's
`Checkpoint` is the production writer); every exact-path number unchanged, proven by the two training goldens re-pointed to that capture staying
byte-green against the unchanged `.bin` fixtures, issue #136; and the same class again, 2026-11-07:
the results `.mat` (`CorpusProcessor::save_results`, the `multiConfigResultsOutputFile` key and
its m/M/t/T truncation) and the Twin's Mode 7 `_DumpLIDInternals` `.mat` (`tasks/lid.rs`) are
deleted with `io/matfile.rs`, every exact-path number unchanged, proven by the goldens #134
re-pointed to `results_matrix()` / `epoch_mems()` staying byte-green and the two binary
end-to-end tests reading the same values through `--results-json`; two refusals go with the
reads, so a malformed `BLSTM_LID_DumpInternals` and a results path in a missing directory no
longer stop a run, and `BLSTM_LID_DumpInternals true` is inert without a warning, issue #135).
The caller-less class once more, 2026-11-09, extended to a write-only accumulator deleted with
the computation that only fed it: `BlstmNetwork`'s input-statistics accumulator (the field, the
type-1 branch's snapshot clone and `analyse_input_seq` fold, its reset, the two net accessors,
`BagOfProcessors::get_input_statistics` and `CorpusProcessor::input_statistics`) had no reader
once #136 retired the stats `.mat`, and goes with `features/stats.rs` and its phase-1 goldens.
The committed goldens run no `InputNormalizationType 1` config, so byte-green says nothing about
the edited branch; the proof is a bit-exact pin of the exact type-1 forward measured on the
pre-cut commit and green unchanged on the post-cut one
(`phase8_frozen_norm.rs::exact_type1_posteriors_are_port_truth_pinned` against
`tests/reference_data/phase8/`), issue #140.
The inference-only retention gating class once more, 2026-11-10, extended to a value the one
production caller discards: the stacks' layer-0 input deltas. A port-only gate,
`CellLayer::set_input_deltas` (inherent, like `set_retain_cache`; default on, which is the legacy
behaviour and the contract the container and layer goldens read), is switched off by
`BlstmNetwork::from_config` on every cell of its stacks. A gated cell skips the deltas it would
propagate only when it runs as the container's last backward layer (`last_layer`, which
`nn/network.rs` passes `true` at layer 0 only) and returns a `0 x 0` array: the whole-sequence
`(_InputWeights * testM)^T` product on the LSTM (`nn/layers.rs`) and the sLSTM, the adapter
back-projection on Mamba and the transformer, a copy on the CfC. `nn/network.rs` and the `Layer`
trait are untouched, and MLP mode is out of scope. This is not the caller-less class: nothing is
deleted. Proven by the goldens byte-green with the container and layer goldens unmodified, by
eight net-level weight-derivative pins measured on the pre-gate commit
(`stack_derivative_pins.rs` against `tests/reference_data/issue146/`) and green unchanged after
it, and by a permanent differential leg (the same net with the gate back on folds
`to_bits`-identical weight derivatives, five cells x two directions x stack sub-sampling `1,1`,
`4,1` and `1,4`, the last running the gated layer at an inverse sub-sampling ratio of 4); the gate's state is asserted, since no golden can see a lost skip, issue #146.
The stepwise hoist a fourth time, 2026-11-11: the corpus fold's per-file step -- the first
file's matrices moved in, every later file added -- written in `run_epoch` and again in the
fold-order probe, is ONE function, `engine/corpus_processor.rs::fold_file_derivs`, with the add
in place (`+=`, the same per-element f64 add as the `&a + &b` it replaces) and `run_epoch`'s
`derivs` parameter gone: the fold ends in the seam stash by move, and the gradient check clones
the stash once after its analytic run. The fold then runs as files complete, behind one bounded
channel per lane, which is a scheduling change ADR-0007 owns (its #154 amendment), with one
contract change on a failed fold: the error is the first failing file's in ascending order,
where it was the lowest failing lane's. Proven by the goldens byte-green with
`lanes_n1_equals_sequential` and `fold_order_ascending_golden` passing with their bodies
unmodified, and by pins
of the fold at two and three lanes measured on the pre-change commit
(`issue154_lane_fold.rs` against `tests/reference_data/issue154/`) and green unchanged after
it, issue #154.
Training is exact-f64 only and bails loudly on the fast variants: since issue #110 (2026-10-28)
at the one bag-construction site, for every mode, on any of four config facts (the config-0
epoch count and gradient-check epsilon, every config's two `BackPropagationActivated` flags),
so neither `CorpusProcessor::train` nor `run()`'s gradient check is reachable on a fast bag
(the seam's explicit `grad_check` there answers `[]`: no net is active).

Fast-path numeric divergence is **by design**: it is documented in the `fast/` module docs and
`RESULTS.md`, never in `IMPROVEMENTS.md`, which tracks legacy-quirk debt only. Its only bound is
the parity gates of ADR-0003.

## Consequences

- An exact-tree diff outside `nn/cells/` must be justified in review against the touch classes;
  "it does not change the goldens" is the proof, and it is checked, not assumed.
- Streaming (ADR-0004) and every new cell's f32 twin (ADR-0005) are fast-tree work; the exact
  tree only ever gains a reference implementation.
- The two trees share no kernel; the fast tree reuses the exact tree's **decision layer** (the
  hysteresis, smoothing and scoring run in f64 on the widened posteriors), which is what makes
  boundary identity a meaningful gate.
