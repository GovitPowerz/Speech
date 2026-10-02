# ADR-0009: Measurements are ledger records; RESULTS.md tables are rendered from them

**Status:** accepted | **Date:** 2026-10-02 (issue #20, grilling outcome recorded there)

## Context

`RESULTS.md` was the only home of most measured numbers: about 1800 number tokens, a few
percent pinned by a test, a few percent gated by a margin, the rest prose, none machine-written.
A run printed its metrics to the console, `speech bench` printed one text line per run, a gate
printed what it measured and the author copied it into a docstring and a table. Three files
quoted the same speedup independently and nothing failed when one copy drifted. Delete
`RESULTS.md` and nothing regenerates it; the complexity lived in the author's transcription.

## Decision

Every measured number is a **record** in the **ledger**, `ledger/<kind>/<id>.json`, one JSON
file per record. A record is an envelope (schema version, when, git SHA and dirty flag, the
build's profile, target and rustc, the host's hardware class, never a hostname) around a
**recipe** (what was run: for a baseline the arm, lineage, cell, direction, split spec,
training budget, seed and lane count; for a bench the label, path, lanes and lineage) and a
**payload** (what it measured). The recipe is the identity of a measurement; the config hash,
SHA, build and host are provenance. Two kinds exist, `baseline` and `bench`; mutation-battery
rows wait for #33 to settle their shape.

Two tiers. A producer (`run_baseline`, a subset gate through the `gate_record` fixture,
`speech bench --json`) writes its record where it runs; `python -m speech.ledger add` is the
one choke point into the committed tree: it validates the schema, runs the license-hygiene
validator (no corpus file, no absolute path, in any string field), refuses a dirty-tree record
unless told otherwise and a non-release build always, copies the record, then renders.

The ledger-owned tables of `RESULTS.md` sit between `<!-- ledger:table <name> -->` and
`<!-- ledger:end -->` markers and are rewritten from the records by a named renderer; CI runs
`render --check`. A seed is part of the recipe, so seeds are rows, never averaged; a recipe with
no full-run record renders a `TBD` row, so filling one is appending a record, not editing prose.
Supersession is append-only: the new record names the old one and the reason; the renderer
hides the old row and footnotes the chain. Python owns the schema; Rust emits payloads only.

The record is not the pin. ADR-0003's pins stay in the tests at measured times ten; a record
says what was measured, a test says what is asserted, and the gate's node id links them.

## Consequences

- A number quoted in `RESULTS.md`'s rendered tables has a record naming its SHA, build and
  host; the "this box" of the prose is now a hardware class in a field.
- No historic number was transcribed into a record. Every gate was re-run on `main` to produce
  its records; a metric that differed from the printed value would have been a STOP under
  ADR-0003, and wall differences are not metrics.
- `run_metadata.json` stays the local run manifest (it may hold the corpus root; it never leaves
  the gitignored `runs/`); `record.json` is the promotable subset.
- The bench stagers, the `bench --processes` ingestion, the Phase 7 bench table and the prose
  assertion over the quoted speedups are the second half of #20.
