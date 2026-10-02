"""One-shot migration of the committed ledger from schema 1 to schema 2 (issue #38).

    python -m speech.ledger.migrate_v2 [--root DIR]

A JSON-level transform, never a hand edit: every record gets `schema_version: 2`; a baseline
record gets `recipe.listing: null` (every v1 run derived its split from the corpus tree) and
`payload.resumed: false` (no v1 run was resumed). Each result is validated by the v2 model and
written under its new content-derived id with `recorded_at` kept; `supersedes` targets are
remapped from old ids to new ones; the old files are removed. Deleted in the commit after the
one that applied it, as ADR-0009's schema-evolution rule says.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech.ledger.render import render_file
from speech.ledger.schema import LEDGER_DIR, RECORD, write_record


def transform(doc: dict[str, object]) -> dict[str, object]:
    if doc["schema_version"] != 1:
        raise ValueError(f"expected a schema 1 record, got {doc['schema_version']}")
    out = dict(doc)
    out["schema_version"] = 2
    if doc["kind"] == "baseline":
        out["recipe"] = {**doc["recipe"], "listing": None}  # type: ignore[dict-item]
        out["payload"] = {**doc["payload"], "resumed": False}  # type: ignore[dict-item]
    return out


def migrate(root: Path) -> dict[str, str]:
    """Rewrite every record under `root`; returns the old-id -> new-id map."""
    paths = sorted(root.glob("*/*.json"))
    docs = {p: transform(json.loads(p.read_text())) for p in paths}
    # Pass 1: the new ids with `supersedes` still naming old ids (the id covers every field, so
    # a remapped target changes the id again; the map is fixed-point after the second pass only
    # if chains are resolved in order, which `sorted` by recorded_at prefix guarantees).
    ids: dict[str, str] = {}
    for p in paths:
        old_id = p.stem
        doc = docs[p]
        if doc.get("supersedes") is not None:
            doc["supersedes"] = ids[doc["supersedes"]]  # type: ignore[index]
        ids[old_id] = RECORD.validate_python(doc).id
    for p in paths:
        rec = RECORD.validate_python(docs[p])
        write_record(rec, p.parent / f"{rec.id}.json")
        if p.stem != rec.id:
            p.unlink()
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m speech.ledger.migrate_v2")
    parser.add_argument("--root", type=Path, default=LEDGER_DIR)
    args = parser.parse_args(argv)
    ids = migrate(args.root)
    for old, new in ids.items():
        print(f"{old} -> {new}")
    render_file(root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
