# ADR-0002: The exact f64 tree is behaviour-frozen; the f32 fast tree lives beside it, selected at one dispatch site

**Status:** accepted | **Date:** 2026-07-19 (Phase 7, PR #13); touch classes extended 2026-07-31 (Phase 9), 2026-08-11 (Phase 10), 2026-08-12 (Phase 11), 2026-10-06 (issues #32, #47, #50)

## Context

The exact tree (`nn/`, `features/`, `tasks/`, `engine/`) is the port's proof: every committed
golden was dumped from a legacy oracle against it, and the committed suite passing byte-for-byte
is what the parity claim rests on. Phase 7 needed a fast inference path (SIMD matmuls, a native
real FFT, f32 throughout), and any arithmetic change inside the exact tree would have moved the
goldens it is proven by.

## Decision

The fast path is a **separate tree** (`src/rust/src/fast/`): f32 twins of the exact kernels,
selected by the `Inference_Path` config key (default `exact`) at **one** bag-construction site.
The exact tree stays frozen: its only sanctioned edits are named **touch classes**, each
behaviour-free on every exact-path run and proven so by the committed golden suite staying
byte-green (the bench plumbing, the dispatch site and its key row, the training and seam guards
on fast variants, golden-re-verified hoists; later the `CellLayer` enum wrap, the forward-only
branches, inference-only retention gating, the exact-length pack guard, the malformed-value
guard on `nn/blstm.rs`'s defaulting config getters and on `cost.rs`'s `CostLaw::from_config`
reads: a present-but-unparseable key errors instead of silently taking the default, as the legacy
`read<T>` does for a value with no parseable prefix; the port is stricter on a parseable prefix
with trailing junk (`1O`, `false # c`), which the legacy's `ss >> val` reads as `1` / `false`,
issue #32; both refuse a non-finite value, and `cost.rs` also parses each `classes_ponderations`
entry strictly instead of dropping a bad one and shifting the later classes, and errors on an
unknown cost-law name instead of panicking, issue #47; the same guard extended to every f64
and list config read in the exact tree -- `features/pipeline.rs`, `tasks/sad.rs`,
`tasks/segmenter.rs`, `tasks/lid.rs`, `config.rs` -- by routing them through the one
`legacy_config.rs` reader family, so a non-finite value errors naming the key everywhere and
every list takes the legacy `split_with_repeat` grammar (one trailing comma dropped, the `*`
repeater) with three deliberate tightenings (an inner empty piece, a bad repeat count, a third
`*` part), issue #50). Training is exact-f64 only and bails
loudly on the fast variants.

Fast-path numeric divergence is **by design**: it is documented in the `fast/` module docs and
`RESULTS.md`, never in `IMPROVEMENTS.md`, which tracks legacy-quirk debt only. Its only bound is
the parity gates of ADR-0003.

## Consequences

- An exact-tree diff outside `nn/cells/` must be justified in review against the touch classes;
  "it does not change the goldens" is the proof, and it is checked, not assumed.
- Streaming (ADR-0004) and every new cell's f32 twin (ADR-0005) are fast-tree work; the exact
  tree only ever gains a reference implementation.
- The two trees share no kernel; the fast tree reuses the exact tree's **decision layer** (the
  hysteresis, smoothing and scoring run in f64 on the widened posteriors), which is what makes
  boundary identity a meaningful gate.
