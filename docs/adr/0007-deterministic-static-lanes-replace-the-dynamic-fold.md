# ADR-0007: A deterministic static-lane fold replaces the legacy's dynamic OpenMP reduction; N=1 is the parity mode

**Status:** accepted · **Date:** 2026-07-07 (Phase 4a, PR #7, spec rule R6)

## Context

The legacy corpus loop was `#pragma omp parallel for schedule(dynamic,1)` with an
`omp critical` fold: the order in which per-file results and gradients were accumulated
depended on thread scheduling, so two runs of the same corpus could differ in the last bits
of a summed gradient, and no run was reproducible in principle. The port's goldens, the
run-twice gates and the fix protocol all need a reproducible fold.

## Decision

Files are assigned to lanes **statically**: file `j` goes to lane `j mod N`, with `N` clamped to
`[1, nb_files]`. Each lane clones the bag at epoch start, processes its files in order, and the
per-lane results are folded in **ascending file index**. `N = 1` is byte-identical to a
sequential loop and is the golden-pinned parity mode; `N > 1` is deterministic but
`N`-dependent (per-lane state chains), so a run records its `N` and the baseline protocol
fixes it per run.

The release seam reads its gradient from the **stashed ascending fold** after the epoch
(`seam_derivs`), the same fold code the in-engine gradient check uses, so the two agree
bit-for-bit by construction. (Before that stash existed the seam read a never-updated
accumulator and returned an exactly zero gradient; the Phase 5 exit gate found it, F10.)

## Consequences

- Every number in `RESULTS.md` names its lane count; comparing across lane counts is not a
  parity comparison.
- The oracle harness's corpus-level tier runs the legacy drivers over nets small enough that
  every reduction is below the GEMM divergence threshold, which is what lets the legacy's
  sequential fold be matched at all.
- Work distribution is a performance knob, never a numerics knob; a future dynamic scheduler
  would have to fold in file order to keep the gates.
