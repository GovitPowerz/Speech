# ADR-0004: Streaming changes timing, never arithmetic: the frozen-norm causality cut, shared step kernels, and emission with zero retractions

**Status:** accepted · **Date:** 2026-07-20 (Phase 8, PR #14); the `pending_begin` term added 2026-07-31 (Phase 9 Task 9)

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
- The causality cost (frozen norm vs native self-norm) is reported per net in `RESULTS.md`,
  never gated: on the committed subset nets it is noise around a degenerate operating point.
