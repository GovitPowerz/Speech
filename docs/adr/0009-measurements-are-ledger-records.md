# ADR-0009: Measurements are ledger records; RESULTS.md tables are rendered from them

**Status:** accepted | **Date:** 2026-10-02 (issue #20, grilling outcome recorded there); the one-live-record rule and schema evolution added 2026-10-02 (issue #38, grilling outcome recorded there); the bench label rule added 2026-10-02 (issue #39)

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
training budget, seed, lane count and the listing the split was drawn from; for a bench the
label, path, lanes and lineage) and a
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

One live record per (source, recipe) (issue #38). `add` refuses a baseline record whose
(source, recipe) already has a non-superseded record unless `--supersede REASON` says why the
copy replaces it; the predecessor is looked up per record and stamped on the copy, a record
with no predecessor is added plainly, a measurement already in the ledger (plainly or as a
stamped copy) is skipped, and a record carrying its own `supersedes` must name the live record
of its key. The renderer never chooses: two live records with one key
fail `render`. Bench records repeat a recipe by protocol (three processes per path, pooled by
the renderer) and are outside this rule; their rule is that a label names exactly one
(lanes, lineage): `add` refuses a bench record whose label is already in the ledger with
another recipe, the caller picks a new label or reruns with the recipe it names (usually the
missing `--lineage`), and a ledger holding two recipes under one label
fails `render` rather than pooling them (issue #39). A resumed baseline
run is not a measurement (its wall covers one segment, its batch cursors restarted from the
seed, its SHA names only the last tree) and is refused at `add` with no override; a per-segment
provenance chain in the envelope is the additive design if one is ever needed. A localized 2015
listing enters the recipe as the blake2b-8 of its bytes (`listing`; `None` for a split derived
from the corpus tree), never as a path, which the hygiene validator forbids.

Schema evolution. `load` reads exactly one schema version. A bump migrates every committed
record through a script in the same commit: a JSON-level transform, validated by the new model,
written under the new content-derived id with `recorded_at` kept and `supersedes` targets
remapped, the old files removed, the tables re-rendered. Ids change, numbers never, and the
script is deleted in the next commit. Schema 2 (issue #38) added `recipe.listing` and
`payload.resumed` to the baseline record and re-keyed the 41 records then committed.

The record is not the pin. ADR-0003's pins stay in the tests at measured times ten; a record
says what was measured, a test says what is asserted, and the gate's node id links them.

A RESULTS.md row cannot flip silently between two measurements: a second record for a live
(source, recipe) names what it replaces and why, or is refused.

## Consequences

- A number quoted in `RESULTS.md`'s rendered tables has a record naming its SHA, build and
  host; the "this box" of the prose is now a hardware class in a field.
- No historic number was transcribed into a record. Every gate was re-run on `main` to produce
  its records; a metric that differed from the printed value would have been a STOP under
  ADR-0003, and wall differences are not metrics.
- `run_metadata.json` stays the local run manifest (it may hold the corpus root; it never leaves
  the gitignored `runs/`); `record.json` is the promotable subset.
- The four Phase 7 bench legs are staged from the repository (`speech.ledger.stage`: the 60 s
  fixture and the three sorted-first corpus files, no filename recorded) and run by
  `python -m speech.ledger bench --leg <name>` as the Phase 7 protocol (N fresh processes per
  path, one record each); the exact-vs-fast matrix is rendered per host, and the speedups the
  README and ARCHITECTURE quote in prose are asserted against the latest bench pair per leg
  (`speech.ledger.prose`), so the three independent "4.6x" copies cannot drift again.
