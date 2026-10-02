"""`python -m speech.ledger`: promote records into the ledger and render RESULTS.md (issue #20).

    python -m speech.ledger add [--allow-dirty] <record.json>...
    python -m speech.ledger render [--check]

`add` is the one choke point between a run directory and the committed tree: it validates each
record (schema and license hygiene), refuses a dirty-tree record unless told otherwise and a
non-release build always, copies the record to `ledger/<kind>/<id>.json`, then renders so the
record and its table land in one commit. `render --check` exits 1 with a diff when RESULTS.md
does not hold what the ledger renders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from speech.ledger.render import RESULTS, render_file
from speech.ledger.schema import LEDGER_DIR, canonical_json, read_record, write_record


def add(paths: list[Path], root: Path, *, allow_dirty: bool) -> list[Path]:
    added: list[Path] = []
    for path in paths:
        rec = read_record(path)
        if rec.git_dirty and not allow_dirty:
            raise SystemExit(f"{path}: recorded on a dirty tree ({rec.git_sha[:12]}); pass --allow-dirty to promote it anyway")
        if rec.build.profile != "release":
            raise SystemExit(f"{path}: a {rec.build.profile!r} build is not a measurement; only release-profile records are promoted")
        dest = root / rec.kind / f"{rec.id}.json"
        if dest.exists():
            if canonical_json(read_record(dest)) != canonical_json(rec):
                raise SystemExit(f"{dest} exists with different content")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        added.append(write_record(rec, dest))
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m speech.ledger", description="the measurement ledger: promote records, render RESULTS.md")
    parser.add_argument("--root", type=Path, default=LEDGER_DIR, help="the ledger directory")
    parser.add_argument("--results", type=Path, default=RESULTS, help="the file holding the ledger tables")
    sub = parser.add_subparsers(dest="command", required=True)
    p_add = sub.add_parser("add", help="validate record files and copy them into the ledger, then render")
    p_add.add_argument("records", type=Path, nargs="+")
    p_add.add_argument("--allow-dirty", action="store_true", help="promote a record taken on a dirty tree")
    p_add.add_argument("--no-render", action="store_true")
    p_render = sub.add_parser("render", help="rewrite the ledger tables (or --check that they are current)")
    p_render.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "add":
        for dest in add(args.records, args.root, allow_dirty=args.allow_dirty):
            print(f"added {dest.relative_to(args.root.parent) if dest.is_relative_to(args.root.parent) else dest}")
        if args.no_render:
            return 0
        render_file(args.results, args.root)
        return 0

    diff = render_file(args.results, args.root, check=args.check)
    if args.check and diff:
        sys.stdout.write(diff)
        print(f"{args.results} is not what the ledger renders; run: python -m speech.ledger render")
        return 1
    return 0
