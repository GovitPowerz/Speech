# Speech

Speech Activity Detection and spoken Language Identification: a Rust engine that turns audio (or precomputed features) into speech segments and language decisions, and a Python optimizer that trains it. The engine is a port of a 2015 C++/MATLAB system, proven against the original and then taken past it.

## Language

### Tasks and signals

**SAD** (Speech Activity Detection):
The per-frame speech / non-speech task. Its output is a Segmentation.
_Avoid_: VAD (the legacy cost-law names use it; the task name here is SAD)

**LID** (spoken Language Identification):
The per-file language task. Always stacked on a SAD net (the Twin); never run alone.

**Posterior**:
A net's per-frame output row before any decision: one speech probability (SAD), one score per language (LID).

**Segmentation**:
The engine's decision output: an ordered boundary list seeded `[Other@0, End@duration]`, carrying its own reference when one is attached.
_Avoid_: mask, labels

**Boundary**:
One entry of a Segmentation: a class (`Speech`, `Other`, `End`) and a begin time.

**Hysteresis**:
The rising / falling double threshold with an area condition that turns posteriors into raw segments.

**Raw segment**:
A hysteresis output, final once closed. The smoothing pipeline reads raw segments and never revises them.

**Smoothing**:
The fixed 8-step pipeline (suppress short, pad, sanitize) that turns raw segments into the emitted Segmentation.

**Reference**:
The ground-truth segmentation attached to a file (STM, CSV or VRCTS). Its presence, not the backprop flag, is what gates target generation.

**VRCTS**:
The XML segmentation format the legacy dumped and read back; times are written to four decimals, the VRCTS write quantum.

**Frame**:
One periodogram / feature row at the spectrum shift (10 ms on the committed configs).
_Avoid_: window (a window is a span of frames or samples, see below)

**Utterance**:
One `external_features` entry: one phSeq line or one cep record. The LID streaming granule.

**Twin**:
The Algo-6 driver: a SAD net plus a LID net that consumes the SAD net's hidden states.

**Mode 7**:
The Twin arm that scores precomputed features (phSeq or cep) per utterance under the frozen-SAD contract.

**Frozen-SAD contract**:
In Mode 7 the SAD net never runs: its result is synthesized, its gradient is structurally zero, only the LID net trains.

**Algo**:
The legacy algorithm number a config selects: 0 VRCTS adapter, 1 TDC, 2 LTSV, 3 spectral BLSTM SAD, 4 signal BLSTM SAD, 5 BLSTM LID, 6 Twin LID.

### Engine and seam

**Engine**:
The Rust crate `speech` and its binary.

**Seam**:
The PyO3 crate `speech-py`, module `speech_rs`: `Engine`, `StreamingSession`, `StreamingLidSession`. Every numpy crossing is a copy.

**Driver**:
A per-file `Segmenter` implementation for one Algo. The nine drivers live in a closed `Processor` enum.
_Avoid_: processor for the engine as a whole

**Bag**:
`BagOfProcessors`: the per-config set of drivers plus the result assembly and best-weight save gate.

**Corpus**:
The `fileslisting` plus `language2classmapping` pair a run reads; a Corpus item is one listing row.

**Lane**:
One of N static parallel folds (file `j` goes to lane `j mod N`). N=1 is the deterministic parity mode; N>1 is deterministic but N-dependent.

**Fold**:
The ascending-file-index reduction of per-file results and gradients at the end of an epoch.

**Weight pack**:
The flat f64 vector of a net's weights in the legacy order (forward, backward, output MLP, normalize tail), post-adim. A `.bin` is its file form.
_Avoid_: checkpoint (a checkpoint is a directory of packs plus a history)

**adim coefficient**:
`sqrt(fan_in*sub + fan_out)`, the per-block scaling applied once in Rust at the config seam, bias excepted.

**Normalize tail**:
The trailing mean / std block of a pack, the type-1 input-normalization statistics.

**KEY_TABLE**:
The one bidirectional table mapping TOML `[section] key` to the legacy flat key. Config keys are declared once (ADR-0006).

**Legacy config**:
The flat whitespace `.config` the 2015 system read. Still importable, used for validation runs only; TOML is canonical.

### Neural networks

**Cell**:
A recurrent layer kind behind the `CellLayer` enum: `lstm`, `slstm`, `mamba`, `cfc`, `transformer`. The extension seam (ADR-0005).

**Direction**:
`bidirectional` (two stacks, forward and reversed, concatenated) or `forward` (one causal stack, no lookahead).

**Stack**:
One direction's `Network<CellLayer>`.

**Output MLP**:
The dense layers after the recurrent stacks: asinh hidden units, softmax output.

**Peephole bundle**:
The LSTM's 12-row peephole matrix. Mixed-kind: some rows read cell states, others the previous step's gate values.

**Windowed driver**:
One of the four regimes a bidirectional net runs under (truncate, two-sweeps, overlap, MLP-overlap).

**Plain regime**:
The whole-sequence forward (`window 0`), the only regime a causal net runs and the only one streaming accepts for it.

**Sub-sampling**:
Per-layer integer decimation of the time axis; the trailing `T mod R` frames are dropped, never restored.

**Lineage**:
A committed SAD config family. **v1** (`lre_sad.toml`) is the only 2015-capacity-comparable one; **v2** (`lre_sad_v2.toml`) removes v1's 48 structurally dead input columns and nothing else.

**Live weight**:
A weight with a nonzero gradient. v1 and v2 have identical live counts per cell; only their nominal pack sizes differ.

**Dead block**:
A weight block whose gradient is structurally zero (sLSTM `b_i`, transformer `b_k`, v1's dead input columns). Derived, then pinned.

### The three implementation axes

**Exact tree**:
The f64 reference implementation (`nn/`, `features/`, `tasks/`, `engine/`). Behaviour-frozen since Phase 7 (ADR-0002); the only path that trains.

**Fast tree**:
The f32 inference path (`fast/`), selected by `Inference_Path = "fast"` at one dispatch site.
_Avoid_: fast for anything merely quick

**Twin kernel**:
A fast-tree cell's f32 counterpart of an exact cell (`FastLstm`, `FastSlstm`, `FastMamba`, `FastCfc`, `FastTransformer`).

**Step kernel**:
The explicit per-timestep `step(x_t, &mut State)` every fast cell exposes. The offline forward loops it; the streaming session drives it; offline and streamed are therefore the same arithmetic.

**Parity leg**:
A test comparing the fast tree against the exact tree on one fixture. Owns arithmetic, at its pin's resolution.

**Self-consistent leg**:
A test comparing two runs of the same kernel (split state, chunk sizes, run-twice). Owns state threading; blind to arithmetic.

**Streaming session**:
The chunked-input SAD path (`StreamingSession`) or the per-utterance LID path (`StreamingLidSession`).

**Causality cut**:
Freezing the two whole-file statistics (the audio gain via `Audio_fixed_gain`, the type-1 normalize tail) so that streaming changes timing only, never arithmetic (ADR-0004).

**Emission**:
A segment finalized and released mid-stream.

**Retraction**:
Any change to an already-emitted boundary. Forbidden; the standing gate asserts zero.

**Holdback**:
The derived smoothing reach the decision layer waits for before emitting, a sum of the config's minimum durations and paddings.

**Consumed frontier**:
How far the hysteresis has actually consumed the input, the first term of the emission trigger; the received frontier is never used.

**Pending begin**:
The begin time of a raw segment the hysteresis has opened but not closed, the third term of the emission trigger.

**Latency bound**:
The derived structural bound on a Speech emission: feature reach, NN window (zero for a causal cell), convolution delay, holdback.

**Commit-wait ceiling**:
The data-dependent bound an Other segment inherits: its right edge is the next speech's onset, so it waits for that speech to close.

### Training and evaluation

**Modern loop**:
`train_modern`: seeded init, SMORMS3 epochs through the seam, per-epoch validation, early stop, best / last checkpoints. Gradient-only.

**Legacy regime**:
The 2015 outer loop: QuantumPSO over a genome that carried the weights in-band. Kept byte-untouched as a driver; weights are now masked out of the genome.

**Genome**:
The vec2struct parameter vector the outer search decodes into a config.

**Arm**:
A from-scratch baseline recipe behind `speech baseline <arm>`: `sad`, `sad-v2`, `lid-features`, `lid-phseq`.

**Subset gate**:
A corpus-gated test that trains an arm from scratch on a stratified subset and asserts beat-init (and, for LID, beat-chance), run-twice bit-identical.

**Full run**:
The user-fired full-corpus launcher. Its numbers land in `RESULTS.md` as they complete; the subset gates are not a substitute for them.

**Collapse**:
The degenerate all-one-class operating point a from-scratch SAD net reaches on a tiny subset (DCF exactly 0.25 all-speech or 0.75 all-non-speech).

**DCF**:
`0.75*Pmiss + 0.25*Pfa` per collar, the NIST OpenSAD scorer ported value-for-value from the vendored perl.

**Collar**:
The scoring tolerance around each reference boundary: 0, 0.25, 0.5, 1 and 2 s.

**Validation signal**:
The per-epoch held-out metric early stopping reads. `NNCostSeg` moves from scratch; the balance cost plateaus.

**Corpus-gated**:
A test that skips cleanly when `data/LRE03-LRE07/` is absent. CI never has it.

### Parity and evidence

**Parity**:
Agreement with a reference: bit-exact where feasible, otherwise measured, then pinned.

**Golden**:
A committed fixture an oracle produced, compared bit-for-bit (or canary-gated) by a test.

**Oracle**:
The harness that produced a golden: the strict-IEEE C++ rebuild, Octave, Perl, or the resurrected 2015 binary. Local-only; CI consumes only their committed output.

**Legacy-truth**:
The golden regime of the port (Roadmap 1): the legacy's output is right by definition, bugs included. Frozen at tag `legacy-parity-v1`.

**Port-truth**:
The golden regime since Phase 5: the port fixes legacy bugs on purpose, each under the fix protocol (ADR-0001).

**Quirk**:
A reproduced legacy behaviour, tracked in `IMPROVEMENTS.md` as kept or fixed.

**Fix protocol**:
RED, re-pin, mutation, flip the entry: the procedure every port-truth fix follows.

**Pin**:
A numeric test threshold set at measured times ten. Never widened to pass.

**STOP**:
The rule that a decision-changing divergence (a boundary, an argmax, a count) halts the work for adjudication instead of widening a pin (ADR-0003).

**Mutation battery**:
The per-phase set of deliberate code mutations, each expected to break a named test. Scored in `RESULTS.md`, gaps included.

**Catcher**:
The test a mutation breaks.

**Touch class**:
A sanctioned category of edit to the exact tree outside `nn/cells/`; anything else needs the golden suite byte-green as proof it is behaviour-free.

**Canary gating**:
Committed libm canaries decide whether a transcendental-dependent golden compares bit-exact (the oracle machine) or within a derived bound (any other).
