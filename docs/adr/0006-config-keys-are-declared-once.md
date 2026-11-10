# ADR-0006: TOML is the canonical config and every key is declared once in a bidirectional table; the legacy `.config` importer is retained for validation

**Status:** accepted | **Date:** 2026-07-09 (Phase 4c, PR #9); the table has grown from 175 to 190 rows since, one block per phase, and lost two with the features they configured (188 since issue #135: a `.config` still carrying `multiConfigResultsOutputFile` or `BLSTM_LID_DumpInternals` loads with the key unread, while a TOML still carrying `[engine] results_file` or `[nn_lid] dump_internals`, as every pre-#135 `--convert-config` output of such a config does, is refused as an unknown key; delete the line or move it under `[legacy.raw]`); the Python mirror of the sized defaults retired 2026-10-12 (issue #27); the `config_bridge.nnet_spec` wrapper deleted 2026-10-29 (issue #90); one config-file reader since 2026-11-10 (issue #150): `toml_config::load_config` (`.toml` by extension, anything else through the legacy importer) behind the CLI and the seam's `speech_rs.load_config`, so no production Python parses config text

## Context

The 2015 engine read a flat whitespace `.config` with `_`-continuation, last-wins and
`val*count` repeat quirks, and every golden fixture names its keys in that shape. A typed config
is wanted going forward, but a second vocabulary maintained by hand would drift from the first,
and the drift would be invisible until a golden moved.

## Decision

`toml_config.rs` holds **one** `KEY_TABLE`: each row maps a TOML `[section] key` to its legacy
flat key, and both directions (TOML to legacy for the engine, legacy to TOML for
`speech --convert-config`) read the same row, so encode and decode cannot drift apart. A key
with no row round-trips losslessly through `[legacy.raw]`; nothing is silently dropped. The
legacy parser (`legacy_config.rs`) is kept as the validation path, and `.config` is accepted
everywhere `.toml` is.

Port-only keys (the fast-path selector, the streaming gain, the cell and direction, the per-cell
geometry) are added as rows with the legacy shape as their default, so every pre-existing config
decodes byte-unchanged; a geometry block is unprefixed (one geometry per config) by precedent.
The Python side reads the same keys through the seam call `speech_rs.blstm_config` (the engine's
own `BlstmConfig::from_legacy`, its defaults filled in; the `config_bridge.nnet_spec` wrapper over
it, test-only since issue #27, was deleted in issue #90), so a present-but-invalid geometry raises
on both sides rather than defaulting,
because a silently defaulted geometry changes a weight pack's length with nothing to catch it.

## Consequences

- A new config key is one table row plus its default; a mismatch between the two sides of the
  seam is a test failure in `phase4c_toml.rs`, not a runtime surprise.
- Sized defaults (`CFC_DEFAULT_BACKBONE_UNITS`, the transformer `window` / `heads` / `d_ff`,
  the `Mamba_*` geometry) exist in Rust alone; Python holds no numeric mirror and reads the
  resolved values through the seam, so the source-parsing pins the mirror needed are gone
  (issue #27).
- The config hash a training run records reads the parsed map, so comments and key order never
  enter it.
