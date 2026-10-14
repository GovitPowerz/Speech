# ADR-0011: Mutation batteries are live patches; a verdict is an expectation, not a record

**Status:** accepted | **Date:** 2026-10-13 (issue #33, grilling outcome recorded there)

## Context

Every phase, and since issue #22 every architecture issue, closed with a mutation battery: the
edits a gate claims to catch, applied by hand, the catcher's failure observed, the source
reverted. The scorecard went into `IMPROVEMENTS.md` (phases 4a to 10) or `RESULTS.md` (since),
the diffs into the gitignored task ledger, and nothing under `src/` or `tests/` was committed.
A reader could read the catcher map and could not re-apply one mutation. The code also moved
under the scorecards: by the issue #33 survey, #27 had deleted the only catcher of a Phase 10
item and the site and catcher of a Phase 11 one, and #62 one of #22's two catchers, with
nothing failing anywhere.

## Decision

A mutation is a committed patch, `tests/mutations/<battery>/<id>.patch`, whose header (text
`git apply` ignores before the first `diff --git`) names the mutation, each catcher with its
expected verdict, and any precondition (`requires: corpus`, `requires: oracle-libm`). One script,
`tests/mutations/run.py`, checks and runs them.

**Live, not archival.** Every committed mutation applies to the current tree and holds its
expected verdicts there. A change that moves a mutated site re-targets the patch; a change that
removes the site deletes it (a retired mutation, archived by git history). The rejected
alternative, each patch pinned to its base SHA and replayed in a worktree, needs no upkeep but
proves only history and cannot see a catcher rot.

**Named catchers only.** A run executes the catchers the header names, one verdict per
(mutation, catcher): `CAUGHT` (the catcher fails), `MISSED` (it passes), `SKIPPED` (the catcher
did not run: the test skipped itself, or a declared precondition is absent, since a Rust corpus
gate passes silently and the runner cannot see that skip without the declaration), `INVALID`
(the mutated tree does not build, which is not a catch), `STALE` (the patch does not apply, or a
catcher no longer names exactly one test; a test filter matching nothing exits 0 and would read
as `MISSED`). Gaps are committed as expected `MISSED`. An observed verdict that differs from the
expected one is a STOP (ADR-0003), a skip included: a claim the run could not verify is not
green, so a full run needs the corpus. A whole-suite sweep per mutation was rejected: it answers
"what catches this" at suite cost per item, not the scorecard's claim.

**Not ledger records.** ADR-0009 deferred battery rows to this issue. A verdict is an
expectation committed beside its mutation and re-checked by the runner; a record of it could
only equal its source, or differ and be a STOP that is never promoted.

**Cheap half in CI, costly half at close.** CI checks both staleness halves on every pull
request (`git apply --check`; each catcher resolved against `cargo test --release -- --list` or
pytest collection), with no rebuild. A run costs a release (LTO) rebuild per mutation before any
catcher runs, plus two `maturin develop` builds when a pytest catcher reads a Rust mutation, so
an issue close runs its new battery plus the mutations whose patched or catcher files the branch
touched (`--changed-since`), and the full battery runs before a tag or on demand.

## Consequences

- A refactor's pull request can fail on a `.patch` file. That is the rule working: the refactor
  re-targets or retires the mutation in the same change.
- The scorecards in `IMPROVEMENTS.md` and `RESULTS.md` stay dated history (what was observed at
  that close); the patch header is the current claim. A verdict change edits the header, never
  the historic scorecard.
- What the scheme still cannot see between full runs: a verdict flip caused from outside both
  the patched file and the catcher file (a caller that stops reaching the site).

## Amendment (issue #78): the second precondition is the host's libm

Two catches hold only where the canary-gated comparators run strict: `phase4a/1` (one ULP on
the raw derivative through `common::assert_oracle_eq`) and `phase4c/2` (one ULP on theta
through `tests/_libm_gate.assert_f64_close`). Off the oracle libm the comparators accept the
hybrid bound (`docs/validation.md`), both read `MISSED`, and a full run reports a STOP for a
host, not a regression. A second precondition, `requires: oracle-libm`, is read exactly as
`requires: corpus` is: per mutation, every catcher `SKIPPED` when the host lacks it. The runner
recomputes the same fixture the comparators recompute (`tests/reference_data/phase1/libm_canaries.bin`,
10 cos / 5 log / 3 exp columns) under the same `SPEECH_ORACLE_LIBM` override, in stdlib (the
`--check --kind cargo` CI job runs on a bare `python3`), so the runner and the catchers cannot
disagree about the host; a test pins the runner's verdict against `oracle_strict()`. Any other
override value is refused before a catcher runs (exit 2), whatever the selection declares:
`common::oracle_mode` panics on it, so every Rust catcher reaching a comparator would read
`CAUGHT`, and `oracle_strict()` raises on it too. A skip is
still a STOP, as it is for an absent corpus: an off-oracle full run exits non-zero on these two,
since a claim it could not verify is not green, so a full run needs the corpus AND the oracle
libm. `SPEECH_ORACLE_LIBM=ulp` runs the whole battery as an off-oracle host would, which is how
the sweep for further host-dependent catches was made (the #78 outcome: none beyond the two).
