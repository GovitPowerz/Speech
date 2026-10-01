# ADR-0008: Two SAD config lineages coexist and are not interchangeable; v1 is the only 2015-capacity-comparable one

**Status:** accepted · **Date:** 2026-08-01 (Phase 10 Task 4, commit `074e392`)

## Context

While explaining the gradient-check counts of the Phase 9 gates, a defect inherited verbatim
from the 2015 production config surfaced: `lre_sad.toml` declares a 23-column NN input, but its
own DSP front-end produces 11 (`3*nb_dct - ignore_first_dct`; the delta keys are regression
orders, not counts). With sub-sampling 4, only 44 of layer 0's 92 fan-in columns carry data and
48 are structurally dead: about 27% of the BLSTM arm's weights can never train. Fixing it in
place would resize every committed pack and invalidate the Phase 6 baseline numbers.

## Decision

Fork, do not fix. `lre_sad_v2.toml` is `lre_sad.toml` byte-for-byte except two value lines
(`nnet_input_size 23 -> 11`, `lstm_neuron_nb "23,24,24" -> "11,24,24"`); the normalize tail
self-sizes on both sides of the seam. v1 stays unchanged, guarded by committed dead-column
floors; v2 is guarded by the inverse (zero dead layer-0 columns per cell, proven non-vacuous
by running the same check on v1 and finding exactly `[44, 92)`).

The deliverable is an identity: v2's **live** (nonzero-gradient) parameter count equals v1's
exactly, per cell, in both directions (LSTM 24409, sLSTM 23257, Mamba 28009, CfC 24469), so the
fork removes dead weight and nothing else.

## Consequences

- Quote the lineage with every SAD number. v1 (33671-weight LSTM pack, the production tuple-A
  size) is the only lineage comparable to the 2015 capacity; v2 has no 2015 counterpart.
- A v1-vs-v2 comparison isolates to pack size (-27.4% on the LSTM) and Xavier fan-in rescaling
  (a 1.19-1.28x wider init), never capacity. It is a measurable question, not a promised win.
- New cells are sized against both lineages' +-15% pack bands on live counts; the CfC backbone
  width (45) and the transformer FFN width (64) were chosen that way, and the one apparent
  conflict was an artefact of counting v1's dead columns.
- v1's gates stay frozen when v2's are touched.
