# ADR-0004: Streaming changes timing, never arithmetic: the frozen-norm causality cut, shared step kernels, and emission with zero retractions

**Status:** accepted | **Date:** 2026-07-20 (Phase 8, PR #14); the `pending_begin` term added 2026-07-31 (Phase 9 Task 9); the bidirectional refusal scoped to the frame stream 2026-10-09 (issue #57); the decision layer brought under the kernel rule 2026-10-12 (issue #26); the setup rule extended to the Twin 2026-10-31 (issue #84)

## Context

The offline engine computes two whole-file statistics before any frame is scored: the audio
gain `(2*RMS+max)/2` and the type-1 input self-normalization. Both are acausal. A streaming
mode could either approximate them (and then stream a *different* model) or remove them from
the arithmetic entirely. The decision layer is acausal too: smoothing pads and merges segments
with knowledge of what follows.

## Decision

**The causality cut freezes, it does not approximate.** `Audio_fixed_gain` replaces the audio
gain with a constant; the pack-carried normalize tail replaces the self-norm with a per-column
affine. An offline run under the same frozen pair is the oracle, and the streamed output must be
**bit-for-bit** equal to it: the posterior history `to_bits`-equal, the final `Segmentation`
identical, chunk-invariant at 20 / 100 / 1000 / 7 ms.

**One kernel, two drivers.** The offline forward and the streaming session run the same step
kernel (`overlap_window_step` for the windowed BLSTM; the per-cell `step` for every causal
cell), so equality holds by construction rather than by tolerance. A performance edit that
batches one side and not the other is caught by the bit-equality gate.
**Amendment (2026-10-10, issue #24):** the two drivers also share their SETUP -- one
`fast::plan::FastSadPlan` (rate-free) and one `SadTimeline` (at the rate) are the shared input, the
step kernel the shared arithmetic; a divergence between the offline and streamed setups is now a
single-site change, and the gates above are the proof that moving the setup moved no arithmetic.
**Amendment (2026-10-12, issue #26):** the rule now covers the DECISION layer too. The hysteresis
is one step kernel, `tasks::segmenter::HystStep` (`push` per convolved value, `flush_tail` at the
end, `pending_begin` read by the stream alone), which the offline `update_segmentation_raw` folds
over and `fast::stream::StreamDecision` drives value by value; the edge-truncating convolution is
one per-output body, `audio::conv_at`, which `convolution_horiz_slice` calls on its snapshot and
`ConvStream` on its ring. Before #26 the stream carried loop-state-for-loop-state transcriptions of
both, held equal by one test; now the self-consistent leg
(`phase8_stream_decision.rs::raw_hysteresis_matches_offline`) pins state threading only, per
ADR-0005's battery lesson, and the arithmetic has exactly the owners the exact tree always had (the
Phase 0b-ii golden, the numpy oracle, the Phase 1 audio goldens), except the two terms only the
stream reads: `pending_begin`'s `+ off` (pinned by
`phase0bii_segmenter.rs::hyst_step_pending_begin_is_the_labeled_begin_in_audio_time`) and
`conv_at`'s `src_base` (the batch passes 0; the trimming stream legs own it). The move is an
ADR-0002 touch class, the stepwise hoist.

**Amendment (2026-10-31, issue #84):** the SETUP rule covers the Twin too -- one
`fast::plan::FastLidPlan` (rate-free) and one `LidTimeline` (at the rate and target) are the
shared input of the offline fast Twin and the Utterance stream, which used to build a whole
offline Twin to borrow its net and its params; the per-prefix net reading and the SAD-axis body
are shared with the SAD plan, and the Twin's two stateful shifts are dropped as the SAD plan's
was (a fixed point at one rate, reset included; `RESULTS.md`, issue #24's section).
**Emission is settled-prefix only, with zero retractions.** The decision layer emits a segment
only once nothing in the future can change it: the frontier is bounded by the last raw boundary,
the **consumed** frontier minus one step (never the received one), and the hysteresis's
**pending begin** (an opened, unclosed raw segment that lies behind the other two). The derived
`holdback` dominates the smoothing reach. A retraction anywhere is a STOP.

## Consequences

- Latency is derived per class from the config and pinned, not promised: a Speech segment lands
  within `feature_reach + nn_window + conv_delay + holdback` unless a raw segment reopens inside
  the holdback, in which case it inherits the Other class's commit-wait ceiling (the Phase 9
  corpus tier found this case; the claim was rescoped, not the gate loosened).
- A causal cell removes the `nn_window` term (6.05 s to 2.82 s on the same config lineage); a
  bidirectional net is unstreamable by construction and is refused at session construction.
  **Amendment (2026-10-09, issue #57):** that refusal is a statement about the FRAME stream
  (`fast/stream.rs`, one chunk of samples per push) and a bidirectional NEW cell; the
  bidirectional LSTM streams there through `overlap_window_step`, bounded by `nn_window`.
  The UTTERANCE stream (`fast/stream_lid.rs`) is the exception: one utterance per push, scored
  whole through the same block forward the offline Twin uses, so any direction streams there,
  prefix-correct and bit-equal to the offline fast run on `finish`. A frame-granular LID
  emission model would be its own issue.
- The causality cost (frozen norm vs native self-norm) is reported per net in `RESULTS.md`,
  never gated: on the committed subset nets it is noise around a degenerate operating point.
