# Phase 8: online/streaming mode - design

Roadmap 2, phase 8. Chunked-input API (lib + CLI), streaming feature extraction,
windowed-BLSTM-with-bounded-lookahead as the first causal cut. Gate: bounded-latency
streaming output equivalent-at-tolerance to offline on the same audio.

User decisions (2026-07-19, this session): FROZEN STATS + FROZEN REFERENCE (streaming
uses frozen normalization statistics - a config-fixed audio gain + the pack-carried
type-1 input stats; the equivalence gate compares streaming vs an OFFLINE run under the
SAME frozen stats, and the offline-frozen vs offline-self-norm delta is measured and
REPORTED as the quantified cost of causality, not gated); scope = SAD FRAME-STREAMING +
LID PER-UTTERANCE (no live-wav LID arm; LID stays utterance-granular, honest about
Mode 7); API = PUSH + `speech stream` SUBCOMMAND (lib StreamingSession, a thin PyO3
binding, a chunked-wav-replay CLI - deterministic, no live-audio dependency);
construction = APPROACH A (a stateful session in fast/ reusing the phase-7 kernels;
mono-first, multi-channel typed-bails with the cross-channel-seeding question deferred).

## S0. Surveyed facts (the streaming-seam survey, 2026-07-19; file:line cites in the
session ledger)

Gate config: `tier2_spectral.config` derived constants - BLSTM window 3.25 s = 327
feature frames (half 163), shift 0.8 s = 80 frames, spectrum shift 0.01 s, ssr 4
(time_step 0.04 s), InputNormalizationType -1, convolution_window_size 9 (19-tap kernel),
TDCwindow 0, LTSVwindow 0.

- EXACTLY TWO whole-file statistics block causality: (a) the `(2*RMS+max)/2` per-channel
  audio normalization (`audio.rs::normalize_channels` :463-482, called once at :906 on
  the whole post-truncation waveform); (b) the type -1 per-sequence self-normalization
  (exact `blstm.rs::self_normalize` :769-801 at :1027; fast twin `fast/nn.rs::
  self_normalize_f32` :117-160 called at `fast/driver.rs:370` once per channel before
  windowing). InputNormalizationType 1 (pack-carried frozen mean/std, `blstm.rs:1008-1024`)
  ALREADY EXISTS and is streaming-friendly.
- Everything else on the SAD fast path streams or is bounded: pre-emphasis is a causal
  1-tap FIR; dither a per-frame table walk; framing/periodogram/mel/DCT row-independent;
  the overlap forward is accumulate-into-buffer + per-row counts (`fast/nn.rs:594-689`)
  - a row is FINAL once its last covering window fires, an intrinsic lookahead of the
  FULL window (~3.26 s), NOT window-shift; the hysteresis scan is CAUSAL WITH LATCHING
  (boundaries never revised by later frames; `segmenter.rs:290-415`); the 19-tap
  convolution adds 0.36 s; the 8-step smoothing reaches back <= 0.68 s (padding) with a
  0.34 s final suppress; the falling-area accumulation adds a DATA-DEPENDENT emission
  delay (small in practice, unbounded when hovering at threshold). Total structural
  emission lookahead ~4.3 s + the area term, dominated by the BLSTM window.
- End-of-stream: the grid-snapped partial tail windows and the open hysteresis segment
  flush at EOS (`fast/nn.rs:629-650`, `segmenter.rs:409-412`).
- The cross-channel `result_vec` seeding is a real cross-channel dependency
  (`fast/driver.rs:358`, never re-zeroed between channels) - a non-issue mono (channel 0
  starts from zeros); multi-channel streaming defers.
- LID has NO frame-online path: Mode 7 consumes whole-utterance `external_features`
  matrices; each entry is scored as a block (truncate/TwoSweeps needs the whole block);
  the answer aggregates across entries. The natural streaming unit is the utterance.
- State objects are lifetime-stable already: `FastPipeline` (banks/plan/coeffs/buffers
  built once) and `FastBlstm` (weights + lazy-grow workspaces) carry across calls; the
  new session state is exactly: the frozen-norm values, a rolling sample/feature ring
  (~one window), the overlap accumulator + per-row counts, the latched decision state +
  the pending smoothing tail.

## S1. Deliverables

### Track 1 - the streaming engine

- **S1.1 `Audio_fixed_gain`** (the one new config surface): a port-only key (TOML
  `[audio] fixed_gain`, one KEY_TABLE row). When present, `read_audio` divides the
  channel by this constant INSTEAD of computing `(2*RMS+max)/2`; threaded as
  `Option<f64>` with `None` = the legacy path BYTE-IDENTICAL (the sanctioned
  inert-by-default guard class; no committed config sets it; the full golden suite is
  the proof). The offline fast path honors it - that is what makes the frozen reference
  runnable. Streaming REQUIRES it + InputNormalizationType 1 (typed bail otherwise).
- **S1.2 `fast::stream::StreamingSession`**: `new(config) -> Result` (validates algo 3,
  mono - multi-channel typed-bails - and the frozen-norm keys); owns the phase-7
  `FastPipeline` + `FastBlstm` unchanged plus the four state pieces (S0). `push(&[f32])
  -> Vec<EmittedSegment>` (segments finalized by this call; `EmittedSegment` = begin/end
  seconds + class, plus the emission lag); `finish() -> (Vec<EmittedSegment>,
  Segmentation)` (the tail flush + the complete segmentation). Pre-emphasis carries one
  sample across chunks; dither carries its table index; frames complete as their sample
  spans arrive (a small per-range `FastPipeline` extension - a fast-tree addition);
  windows fire when their input span exists; rows emit when their covering-window count
  completes.
- **S1.3 The incremental decision layer**: the 19-tap convolution on the emitted-row
  stream (0.36 s half-width delay); the hysteresis scan as a PERSISTENT state machine
  replicating `update_segmentation_raw`'s loop state exactly (begin/end/areas/latches);
  smoothing via RE-SMOOTH-AND-EMIT-STABLE-PREFIX: on each new raw segment, re-run the
  cheap whole-list `smooth_segmentation` and emit segments beyond the maximum smoothing
  reach (<= 0.68 s padding + 0.34 s suppress) that provably cannot change - no online
  re-derivation of the 8-step pipeline, equivalence at EOS by construction. `finish()`
  output must equal the offline-frozen run's `Segmentation` (the gate).
- **S1.4 `StreamingLidSession`** (per-utterance): `push_utterance(features) ->
  per-utterance score + the running aggregate`, `finish() -> the final result rows` -
  wrapping the existing Mode-7 per-entry loop. Utterance-granular by design; documented
  as such.
- **S1.5 Surfaces**: the PyO3 `StreamingSession` binding (numpy f32 chunks in, segment
  tuples out, the finish rows; COPY semantics per the seam convention); the
  `speech stream <config> [--chunk-ms N] <wav>` PORT-ONLY subcommand (the bench
  precedent): replays the wav in chunks, prints parseable timestamped SEG lines as
  segments finalize + a `STREAM path=fast chunks=<n> max_lag_s=<f> mean_lag_s=<f>
  rtf=<f>` summary line.

### Track 2 - proof

- **S1.6 The equivalence gate, CI tier** (the committed 60 s fixture STAGED MONO -
  channel 0 extracted at test-staging time, since the session is mono-first and the
  fixture wav is stereo - + the tuple-A pack + the tier2 config flipped to frozen norm;
  the offline-frozen reference runs on the SAME staged mono wav): (a) OFFLINE EQUIVALENCE - streaming
  `finish()` vs the offline fast run under identical frozen stats: segment count/types
  IDENTICAL, boundary max_dt measured-then-pinned (the design predicts EXACT agreement -
  same kernels, same normalized values, deferral not approximation; any delta is a
  finding to adjudicate); (b) CHUNKING INVARIANCE - streamed at 20 ms / 100 ms / 1 s /
  7 ms (non-divisor): all four BIT-IDENTICAL to each other (chunking changes timing,
  never arithmetic); (c) PREFIX CONSISTENCY - every mid-stream emission appears
  unchanged in the final output (no retractions), asserted across the whole gate corpus.
- **S1.7 The causality-cost leg** (REPORTED, not gated): offline-frozen vs
  offline-self-norm on the committed fixtures + a corpus-gated real file: boundary
  deltas + held-out DCF delta recorded in RESULTS.md. A mode difference, not a defect;
  sanity-checked (same order of magnitude), never pass/fail beyond that.
- **S1.8 Latency**: the session tracks per-emission lag (audio-time pushed minus the
  emitted boundary time); pins = the STRUCTURAL bound (window 3.26 s + convolution
  0.36 s + smoothing holdback <= 1.02 s) with the data-dependent area term MEASURED on
  the fixtures and pinned with headroom (measure-then-pin; the structural floor and the
  data tail separately named); per-chunk RTF confirms real-time margin (the phase-7
  fast-path RTF ~0.0005 gives ~3 orders of headroom).
- **S1.9 The corpus tier** (local, corpus-gated): one real ~600 s file streamed vs
  offline-frozen with the phase-6 SAD subset checkpoint (trained weights, not seeds) -
  the same S1.6 assertions.

### Track 3 - docs + closeout

- **S1.10** Mutation battery (streaming-shaped: e.g. drop the pre-emphasis carry, break
  the ring wraparound, emit before the count completes, skip the EOS flush, chunk-size
  arithmetic leak); docs refresh (CLAUDE.md `fast/stream.rs` row + the key + the
  subcommand + the Roadmap 2 flip; README; RESULTS.md the causality-cost + latency
  tables); final whole-branch review; smart-commit; finish. The house shape.

## S2. Gate/oracle tiers

The OFFLINE FAST PATH UNDER FROZEN STATS is the oracle (same kernels, only chunking
differs). Tiers: (1) CI equivalence + chunking-invariance + prefix consistency (S1.6);
(2) corpus-gated real-data equivalence (S1.9); (3) the latency pins (S1.8); (4) the
reported causality cost (S1.7). No legacy oracle exists for streaming - this is
port-only surface, PORT-TRUTH by construction.

## S3. Risks

- **R1 the exact-agreement prediction fails** (a boundary differs between streamed and
  offline-frozen): STOP and adjudicate openly - the likely causes are ordering (a
  deferred window computed on a different workspace state) or the EOS tail handling;
  never silently widen to a tolerance without a root-cause note.
- **R2 the stable-prefix bound is wrong** (a smoothing step reaches further than the
  derived 0.68+0.34 s under some segment pattern): the prefix-consistency assertion
  catches it as a retraction; the fix is a corrected holdback derivation, not a wider
  emission delay by fiat.
- **R3 chunk-boundary arithmetic leaks** (pre-emphasis carry, dither index, ring
  wraparound, partial-frame handling): the chunking-invariance bit-identity leg is the
  designed catcher; the battery mutates each mechanism.
- **R4 the type-1 stats in real packs**: the tuple-A pack carries a trained normalize
  tail; from-scratch packs carry the 0/1 identity - both are valid frozen stats, but the
  gate configs must state WHICH pack they use and the causality-cost leg must use the
  same pack on both sides.
- **R5 latency-pin brittleness in CI**: the structural bound is deterministic (frame
  arithmetic), safe to pin tight; the area term is data-dependent - pinned only on the
  committed fixtures with headroom, documented as fixture-specific.

## S4. Conventions

- The exact tree is UNTOUCHABLE except the S1.1 `Audio_fixed_gain` thread through
  `read_audio` (inert-by-default, golden-proven) - the phase-7 sanctioned-guard class.
- Streaming numeric behavior = the offline fast path's (frozen-stats mode); divergence
  from the SELF-NORM mode is by design and lives in RESULTS.md (the causality cost),
  never IMPROVEMENTS.md.
- Corpus gating, license bright line, ASCII, foreground, the commit trailer, never
  push, feature branch only - all as established.
