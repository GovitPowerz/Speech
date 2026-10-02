# ADR-0003: Decisions are gated on identity; numbers are measured, then pinned at ten times the measurement; a breach is a STOP

**Status:** accepted | **Date:** 2026-07-19 (Phase 7 spec, rules R1/R2); applied to every parity and streaming gate since

## Context

Once a second implementation of the same arithmetic exists (the fast tree, a streamed path, a
new cell's twin), "close enough" needs a definition. A tolerance chosen in advance is either too
loose to catch a wrong kernel or too tight for cross-platform SIMD variance, and a tolerance
widened to make a test pass has no evidentiary value.

## Decision

Two tiers, with different rules.

**Decisions are gated on identity.** A segment count, a boundary type, a boundary time
(`max_dt == 0.0` exactly), an argmax, a metric delta on real data (DCF, LID error, Cavg at
exactly `0.0`): these are asserted equal. Any difference stops the work for adjudication; the
test is never relaxed to a tolerance.

**Numbers are measured, then pinned.** A posterior delta, a gradient-check residual, a benchmark
budget: run the leg, record the measurement in the test and in `RESULTS.md`, and pin it at
**measured x 10** per shape. A pin is never widened to pass. When a legitimate change moves a
measurement (one sanctioned re-measurement: the full-f32 mel in Phase 10), every affected pin is
re-measured in one sweep and the old numbers are kept as superseded, not edited away. An
absolute ceiling sits above the pins where one is derivable (the `1e-4` gradient-check STOP:
a pin above it is a defect, not a widening).

## Consequences

- A crossing fixture that saturates is fixed by widening the **search** for interior boundaries
  (a gain or offset ladder), never the pin.
- Self-consistency legs (two runs of one kernel) own state threading only; parity legs own
  arithmetic only down to the pin's resolution. Below that resolution a symmetric kernel edit is
  invisible by construction, so the control is a design-time rule in the module doc, not a test.
- Every phase ends with a mutation battery that applies the edits the gates claim to catch and
  records which test actually fired, gaps included (`RESULTS.md`).
