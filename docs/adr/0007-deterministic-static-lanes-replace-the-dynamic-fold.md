# ADR-0007: A deterministic static-lane fold replaces the legacy's dynamic OpenMP reduction; N=1 is the parity mode

**Status:** accepted | **Date:** 2026-07-07 (Phase 4a, PR #7, spec rule R6); amended 2026-11-11 (issue #154)

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

## Amendment (issue #154): the fold runs as files complete

The fold of the Decision held every file's per-net derivative matrices until the last lane had
finished, then sorted and summed them, so a gradient fold's memory grew with its file count
(about 0.5 MiB per file for the v1 SAD net). Since issue #154 each lane sends its files down its
own bounded channel and the fold, on the calling thread, receives from lane `j mod N` for
`j = 0, 1, 2, ...`. File `j` is always the next thing that lane sends, so the cyclic receive is
the ascending fold itself, with no reorder buffer and no sort. A lane that gets ahead of the
fold blocks on its send: at channel capacity `C` (a constant, 1) at most `N * (C + 1) + 1`
per-file entries are alive. The capacity is a performance knob; no number depends on it. The
lanes are scoped OS threads, because a pool worker blocked on a receive would starve a lane
once `N` reaches the pool size, and `rayon` left the workspace with its one use site.

One behaviour changed: a failed fold reports the first failing file in ascending order (it was
the lowest failing lane's; the two agree at `N = 1`), the other lanes stop at their next send,
and neither the results (the result matrix is reset with them) nor the seam stash is committed. A lane's panic is still re-raised
with its own payload, the fold joining each lane itself. Nothing else did: the assignment of
files to lanes, the per-lane bag clone and its state chain, the order of the sum and every
summed value, pinned at two and three lanes on the commit before the change
(`src/rust/tests/issue154_lane_fold.rs`). The per-file step is one function,
`fold_file_derivs`: the first file's matrices are moved in, so a negative zero keeps its sign,
and later files add in place.
