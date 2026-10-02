# ADR-0001: Goldens assert port-truth after the parity tag; legacy-truth is frozen at `legacy-parity-v1`

**Status:** accepted | **Date:** 2026-07-16 (commit `6c6acdf`, the Phase 5 opening)

## Context

Roadmap 1 ported a 2015 C++/MATLAB system bit-for-bit, bugs included: every legacy quirk was
reproduced on purpose because the goldens were dumped from the legacy and a "fix" would have
broken parity before parity was proven. By the end of Phase 4d the port matched the resurrected
2015 production binary (VRCTS byte-identical, scored result columns within ~1e-15), and the
reproduced bugs (`IMPROVEMENTS.md`) included wrong-result ones: a batching clobber, a sticky
confusion index, a gradient that was `A/y` where the cost was constant.

## Decision

The parity proof is frozen as the annotated tag `legacy-parity-v1` (`main@74284d6`), reproducible
with `git checkout legacy-parity-v1`, gates included. From Phase 5 on, HEAD's goldens assert
**port-truth**: correctness-critical legacy behaviours are fixed, each under the fix protocol in
`IMPROVEMENTS.md` (RED against the legacy value, re-pin, a mutation that reverts the fix and must
fail, flip the entry to `FIXED (phase, F<n>, commit)`). The 2015-parity gates were deleted from
HEAD in the tagging commit, not skipped. The oracle harness family (`tools/`) is never regenerated
for a fixed site: it describes the legacy forever and stays the oracle only for entries kept.

## Consequences

- A kept quirk needs a stated reason (a load-bearing convention, no clear alternative, dead
  code); the Phase 5 sweep recorded one per entry.
- Numbers quoted from the legacy era (the 2015 dashboards) are comparability references, not
  targets; the baselines are recomputed under the port's own protocol (`RESULTS.md`).
- Any new divergence from the legacy is either a documented fix or a bug; "the legacy did it"
  no longer settles a review.
