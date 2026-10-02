"""`python -m speech.ledger`: promote records into the ledger and render RESULTS.md (issue #20).

    python -m speech.ledger add [--allow-dirty] [--supersede REASON] <record.json>...
    python -m speech.ledger render [--check]
    python -m speech.ledger bench (--leg NAME | --config PATH --label L) [--path exact|fast|both] [--processes N]

`add` is the one choke point between a run directory and the committed tree: it validates each
record (schema and license hygiene), refuses a dirty-tree record unless told otherwise, a
non-release build always and a resumed baseline run always, refuses a baseline record whose
(source, recipe) already has a live record unless `--supersede REASON` names why the copy
replaces it (the predecessor is looked up per record, in argument order within one batch, and
stamped on the copy, which therefore gets a new id; a record with no predecessor is added
plainly, one already in the ledger, plainly or as a stamped copy, is skipped, and one carrying
its own `supersedes` must name the live record of its key), copies the record to
`ledger/<kind>/<id>.json`, then renders
so the record and its table land in one commit. Bench records repeat a recipe by protocol
(three processes per path) and are outside the one-live-record rule; their label rule is issue
#39. `render --check` exits 1 with a diff when RESULTS.md does not hold what the ledger renders.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from speech.ledger.bench import BINARY, bench_json, lanes_of, wrap
from speech.ledger.render import RESULTS, render_file
from speech.ledger.schema import LEDGER_DIR, BaselineRecord, RecordBase, git_state, load, read_record, write_record
from speech.ledger.stage import CORPUS_ROOT, LEGS
from speech.ledger.tables import current_baselines


def add(paths: list[Path], root: Path, *, allow_dirty: bool, supersede: str | None = None) -> list[Path]:
    # Validate the whole batch before copying any of it, so a refusal leaves the ledger untouched.
    if supersede is not None and not supersede.strip():
        raise SystemExit("--supersede needs a reason")
    records = load(root) if root.is_dir() else []
    live = {r.key(): r for r in current_baselines(records)}
    # `load` has refused a file whose content is not its name, so a matching unstamped id is the
    # same measurement, whether it went in plainly or as a `--supersede` copy.
    promoted = {_unstamped_id(r) for r in records}
    pending = []
    for path in paths:
        rec = read_record(path)
        if rec.git_dirty and not allow_dirty:
            raise SystemExit(f"{path}: recorded on a dirty tree ({rec.git_sha[:12]}); pass --allow-dirty to promote it anyway")
        if rec.build.profile != "release":
            raise SystemExit(f"{path}: a {rec.build.profile!r} build is not a measurement; only release-profile records are promoted")
        if isinstance(rec, BaselineRecord) and rec.payload.resumed:
            raise SystemExit(
                f"{path}: a resumed run is not a measurement (one segment's wall, restarted batch cursors, the last tree's SHA); re-run from scratch"
            )
        if _unstamped_id(rec) in promoted:
            continue
        if isinstance(rec, BaselineRecord):
            prev = live.get(rec.key())
            if rec.supersedes is not None:
                if prev is None or rec.supersedes != prev.id:
                    raise SystemExit(f"{path}: supersedes {rec.supersedes}, which is not the live record of its (source, recipe)")
            elif prev is not None:
                if supersede is None:
                    raise SystemExit(f"{path}: its (source, recipe) already has a live record {prev.id}; pass --supersede REASON to replace it")
                # Through the validator, not `model_copy`: the reason is free text and the hygiene rule applies to it.
                rec = BaselineRecord.model_validate({**rec.model_dump(mode="json"), "supersedes": prev.id, "reason": supersede})
            live[rec.key()] = rec
        promoted.add(_unstamped_id(rec))
        pending.append((rec, root / rec.kind / f"{rec.id}.json"))
    added: list[Path] = []
    for rec, dest in pending:
        dest.parent.mkdir(parents=True, exist_ok=True)
        added.append(write_record(rec, dest))
    return added


def _unstamped_id(rec: RecordBase) -> str:
    return rec.model_copy(update={"supersedes": None, "reason": None}).id


def bench(args: argparse.Namespace) -> list[Path]:
    """Stage (a named leg, or the given config), run N fresh processes per path, wrap and write
    one record each into the stage directory; the caller promotes them."""
    if not args.allow_dirty and git_state()[1]:
        raise SystemExit("the tree is dirty, so `add` would refuse every record; commit first or pass --allow-dirty")
    stage_dir = Path(tempfile.mkdtemp(prefix="speech-bench-")) if args.stage_dir is None else args.stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    if args.leg is not None:
        leg = LEGS[args.leg]
        config, label, lineage = leg.stage(stage_dir, args.corpus_root), leg.label, leg.lineage
    else:
        config, label, lineage = args.config, args.label, args.lineage
    if not args.binary.is_file():
        raise SystemExit(f"{args.binary} is not built (cargo build --release)")
    lanes = lanes_of(config)
    paths = ("exact", "fast") if args.path == "both" else (args.path,)
    written: list[Path] = []
    for path in paths:
        for _ in range(args.processes):
            rec = wrap(bench_json(args.binary, config, path, label), lineage=lineage, lanes=lanes)
            written.append(write_record(rec, stage_dir / f"{rec.id}.json"))
            run = rec.payload.runs[0]
            print(f"{label} {path}: wall_s={run.wall_s:.4f} audio_s={run.audio_s:.2f} rtf={run.rtf:.6f} maxrss_mb={run.maxrss_mb:.3f}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m speech.ledger", description="the measurement ledger: promote records, render RESULTS.md")
    parser.add_argument("--root", type=Path, default=LEDGER_DIR, help="the ledger directory")
    parser.add_argument("--results", type=Path, default=RESULTS, help="the file holding the ledger tables")
    sub = parser.add_subparsers(dest="command", required=True)
    p_add = sub.add_parser("add", help="validate record files and copy them into the ledger, then render")
    p_add.add_argument("records", type=Path, nargs="+")
    p_add.add_argument("--allow-dirty", action="store_true", help="promote a record taken on a dirty tree")
    p_add.add_argument("--supersede", metavar="REASON", default=None, help="a baseline record whose (source, recipe) is live replaces it, for this reason")
    p_add.add_argument("--no-render", action="store_true")
    p_render = sub.add_parser("render", help="rewrite the ledger tables (or --check that they are current)")
    p_render.add_argument("--check", action="store_true")
    p_bench = sub.add_parser("bench", help="run the Phase 7 bench protocol (N fresh processes per path) and promote the records")
    which = p_bench.add_mutually_exclusive_group(required=True)
    which.add_argument("--leg", choices=sorted(LEGS), help="a staged leg (the Phase 7 recipes, written down once)")
    which.add_argument("--config", type=Path, help="an arbitrary config (then --label is required)")
    p_bench.add_argument("--label", help="the recipe's name in the ledger (never a config path)")
    p_bench.add_argument("--lineage", choices=("v1", "v2"), default=None, help="the SAD lineage of an arbitrary config (ADR-0008)")
    p_bench.add_argument("--path", choices=("exact", "fast", "both"), default="both")
    p_bench.add_argument("--processes", type=int, default=3, help="fresh processes per path (the Phase 7 protocol: 3)")
    p_bench.add_argument("--binary", type=Path, default=BINARY)
    p_bench.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    p_bench.add_argument("--stage-dir", type=Path, default=None, help="where the staged config and the records go (default: a fresh temp dir)")
    p_bench.add_argument("--allow-dirty", action="store_true")
    p_bench.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "bench":
        if args.config is not None and not args.label:
            parser.error("--config requires --label")
        if args.processes < 1:
            parser.error("--processes must be >= 1")
        for dest in add(bench(args), args.root, allow_dirty=args.allow_dirty):
            print(f"added {dest.relative_to(args.root.parent) if dest.is_relative_to(args.root.parent) else dest}")
        if not args.no_render:
            render_file(args.results, args.root)
        return 0

    if args.command == "add":
        for dest in add(args.records, args.root, allow_dirty=args.allow_dirty, supersede=args.supersede):
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
