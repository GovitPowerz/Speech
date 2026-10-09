# ADR-0005: The recurrent cell is the extension seam; a new architecture is one enum variant and nothing outside it

**Status:** accepted | **Date:** 2026-07-31 (Phase 9, PR #15); totality of the fast matrix 2026-08-11 (Phase 10); fifth cell 2026-08-12 (Phase 11); totality extended to the Twin's LID net 2026-10-09 (issue #57)

## Context

The port's network container was already generic (`Network<L: Layer>`, a 10-method trait), but
the only `L` was the legacy peephole LSTM. The goal of adding sLSTM, Mamba, CfC and a windowed
transformer could have been met by a trait object, a per-cell network type, or a new container.
Each of those would have touched the windowed drivers, the input normalization, the flat-weight
seam, scoring and the training loop, all golden-pinned.

## Decision

**The cut is at the layer.** `nn/cells/` holds a **closed enum** `CellLayer` implementing
`Layer` by match delegation; `BlstmNetwork`'s two stacks are `Network<CellLayer>`. A new cell is:
a variant, a `Layer` impl inside `cells/`, an `init_weights.py` builder emitting the same flat
order, its `KEY_TABLE` rows, and both f32 twins (`fast/cells.rs` causal, `fast/bicell.rs`
bidirectional). The exhaustive `match` in `from_config` and in `cell_weight_count` makes a
missing arm a **compile error**, so the fast (cell x direction) matrix is total by construction.
`Direction` (`bidirectional` / `forward`) is the one companion axis, read at the same seam.
Since 2026-10-09 (issue #57) totality covers the Twin's LID net too: `FastLidNet` classifies
the `BLSTM_LID` prefix through the same function and reuses the same two cell twins.

Every cell lands with the same evidence: a hand-derived backward checked by central differences
per shape and seed (pinned at measured x 10, under a `1e-4` STOP), a corpus-level gradient check
through the PyO3 seam, derived-then-pinned dead blocks, exact-vs-fast parity at boundary
identity, a streaming row at bit-equality, and a from-scratch subset gate.

## Consequences

- `train_modern`, batching, the windowed drivers and the flat-weight seam never learn the cell
  type; Phase 9 needed zero changes to the training loop, and the fifth cell's streaming row
  needed zero `src/` edits.
- Per-cell numeric conventions are declared, not hidden: the softmax in the transformer always
  max-subtracts (the f32 twin needs the headroom), CfC runs under a constant-step reduction, the
  sLSTM running-max stabilizer is held constant in the backward because the output is invariant
  to it.
- Retention gating (releasing backward caches on an inference-only net) is an inherent method on
  the enum, not an eleventh trait method; it is applied where a cache was measured expensive
  (Mamba, the transformer's attention cache) and is a named follow-on elsewhere.
