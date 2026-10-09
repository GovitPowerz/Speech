"""The RESULTS.md tables rendered from the ledger (issue #20).

Each renderer is a function over the loaded records, registered by the name a
`<!-- ledger:table <name> -->` marker in RESULTS.md uses. Shared rules: a table takes the one
live (non-superseded) record per (source, recipe), two being an error, never a choice (issue
#38); a subset-gate row is the latest `gate` record of its (arm, lineage, cell, direction); a full-run row is one row per non-superseded full record (a
seed is part of the recipe, so seeds are rows, never averaged) and a single `TBD` row when none
exists; a bench label names one (lanes, lineage), two being an error, never a pooled cell
(issue #39); every table ends with the hosts its rows came from and one footnote line per
supersession chain. Prose around a table stays hand-written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from speech.config_bridge import CELL_TYPES, DIRECTIONS
from speech.ledger.schema import BaselineRecord, BenchRecord, Lineage, superseded_ids
from speech.ledger.stage import LEGS

Record = BaselineRecord | BenchRecord
Renderer = Callable[[list[Record]], list[str]]
TABLES: dict[str, Renderer] = {}

CHANCE = f"{100.0 * (1.0 - 1.0 / 12.0):.2f}"
COLLARS = (0.0, 0.25, 0.5, 1.0, 2.0)
CELL_NAME = {"lstm": "LSTM", "slstm": "sLSTM", "mamba": "Mamba", "cfc": "CfC", "transformer": "Transformer"}


def table(name: str) -> Callable[[Renderer], Renderer]:
    def register(fn: Renderer) -> Renderer:
        TABLES[name] = fn
        return fn

    return register


# --------------------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------------------- #


def current_baselines(records: Iterable[Record]) -> list[BaselineRecord]:
    """The one non-superseded baseline record per (source, recipe), in recorded order. Two live
    records with one key is a ledger that bypassed `add` (issue #38): the renderer never
    chooses between them, it raises."""
    base = [r for r in records if isinstance(r, BaselineRecord)]
    dead = superseded_ids(base)
    live: dict[tuple[object, ...], BaselineRecord] = {}
    for r in sorted(base, key=lambda r: (r.recorded_at, r.id)):
        if r.id in dead:
            continue
        key = r.key()
        if key in live:
            raise ValueError(f"two live records for one (source, recipe): {live[key].id} and {r.id}; the later one must supersede the earlier")
        live[key] = r
    return list(live.values())


def gate_row(current: list[BaselineRecord], arm: str, lineage: str | None, cell: str, direction: str) -> BaselineRecord | None:
    rows = [
        r
        for r in current
        if r.payload.source == "gate"
        and r.recipe.arm == arm
        and r.recipe.lineage == lineage
        and r.recipe.cell == cell
        and r.recipe.direction == direction
        and not r.recipe.full
    ]
    return max(rows, key=lambda r: (r.recorded_at, r.id)) if rows else None


def full_rows(current: list[BaselineRecord], arm: str, lineage: str | None, cells: Iterable[tuple[str, str]]) -> list[BaselineRecord]:
    wanted = set(cells)
    return [r for r in current if r.recipe.full and r.recipe.arm == arm and r.recipe.lineage == lineage and (r.recipe.cell, r.recipe.direction) in wanted]


# --------------------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------------------- #


def _date(r: BaselineRecord) -> str:
    return r.recorded_at.date().isoformat()


def _files(r: BaselineRecord) -> str:
    p = r.payload
    return f"{p.n_train} / {p.n_valid} / {p.n_test}"


def _budget(r: BaselineRecord) -> str:
    rc = r.recipe
    s = f"{rc.epochs} ep x {rc.steps_per_epoch} steps"
    if rc.audio_max_duration is not None:
        s += f", {rc.audio_max_duration:g} s cap"
    return s


def _config(r: BaselineRecord) -> str:
    return f"`{r.payload.config_name.rsplit('/', 1)[-1]}`, {_budget(r)}"


def _cell(r: BaselineRecord) -> str:
    return f"{CELL_NAME.get(r.recipe.cell, r.recipe.cell)} / {r.recipe.direction}"


def _opt(x: float | None, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def _collar_range(r: BaselineRecord, *, init: bool = False) -> str:
    scores = [r.payload.collar(c, init=init) for c in COLLARS]
    if any(s is None for s in scores):
        return "n/a"
    values = [s.dcf for s in scores if s is not None]
    lo, hi = min(values), max(values)
    return f"{lo:.6f} (all 5)" if lo == hi else f"[{lo:.6f}, {hi:.6f}]"


def _dcf_cells(r: BaselineRecord, *, init: bool = False) -> list[str]:
    return [_opt(s.dcf if (s := r.payload.collar(c, init=init)) else None, ".4f") for c in COLLARS]


def _pmiss_pfa(r: BaselineRecord, fmt: str, *, init: bool = False) -> str:
    s = r.payload.collar(0.5, init=init)
    return "n/a" if s is None else f"{s.pmiss:{fmt}} / {s.pfa:{fmt}}"


def _gain(r: BaselineRecord) -> str:
    tr, ini = r.payload.collar(0.5), r.payload.collar(0.5, init=True)
    return "n/a" if tr is None or ini is None else f"{ini.dcf - tr.dcf:+.6f}"


def _wall(r: BaselineRecord) -> str:
    return f"{r.payload.wall_s:.0f} s"


def _tbd(label: str, n: int) -> str:
    return "| " + " | ".join([label, *(["TBD"] * n)]) + " |"


def host_key(r: Record) -> str:
    return f"{r.host.chip} ({r.host.arch}, {r.host.cores} cores, {r.host.os})"


def _footer(rows: Sequence[Record], all_records: Iterable[Record]) -> list[str]:
    """The hosts the rows came from, then one line per supersession link in the chains touching
    them, each naming the record that did the superseding and its reason."""
    lines: list[str] = []
    hosts = sorted({host_key(r) for r in rows})
    if hosts:
        lines += ["", "Hosts: " + "; ".join(hosts) + "."]
    by_id = {r.id: r for r in all_records}
    for r in sorted(rows, key=lambda r: r.id):
        cur: Record | None = r
        while cur is not None and cur.supersedes is not None:
            lines.append(f"Superseded: `{cur.supersedes}` by `{cur.id}` ({cur.reason}).")
            cur = by_id.get(cur.supersedes)
    return lines


# --------------------------------------------------------------------------------------- #
# Phase 6: the three Results tables
# --------------------------------------------------------------------------------------- #


def _lid_table(records: list[Record], arm: str) -> list[str]:
    current = current_baselines(records)
    gate = gate_row(current, arm, None, "lstm", "bidirectional")
    fulls = full_rows(current, arm, None, [("lstm", "bidirectional")])
    lines = [
        "| split | files (train/valid/test) | LID error % | Cavg | chance % | config | seed |",
        "|---|---|---|---|---|---|---|",
    ]
    rows: list[BaselineRecord] = []
    if gate is None:
        lines.append(_tbd("subset gate", 6))
    else:
        p = gate.payload
        lines.append(
            f"| subset gate ({_date(gate)}) | {_files(gate)} | {_opt(p.lid_error, '.2f')} | {_opt(p.cavg, '.2f')} | {CHANCE} "
            f"| {_config(gate)} | {gate.recipe.seed} |"
        )
        lines.append(
            f"| subset gate -- untrained init baseline | (same test set) | {_opt(p.init_lid_error, '.2f')} | {_opt(p.init_cavg, '.2f')} | {CHANCE} "
            f"| (seed packs, no training) | {gate.recipe.seed} |"
        )
        rows.append(gate)
    for r in fulls:
        p = r.payload
        lines.append(
            f"| full run ({_date(r)}) | {_files(r)} | {_opt(p.lid_error, '.2f')} | {_opt(p.cavg, '.2f')} | {CHANCE} | {_config(r)} | {r.recipe.seed} |"
        )
        rows.append(r)
    if not fulls:
        lines.append(_tbd("full run", 6))
    return lines + _footer(rows, records)


@table("phase6_lid_features")
def phase6_lid_features(records: list[Record]) -> list[str]:
    return _lid_table(records, "lid-features")


@table("phase6_lid_phseq")
def phase6_lid_phseq(records: list[Record]) -> list[str]:
    return _lid_table(records, "lid-phseq")


@table("phase6_sad_v1")
def phase6_sad_v1(records: list[Record]) -> list[str]:
    current = current_baselines(records)
    gate = gate_row(current, "sad", "v1", "lstm", "bidirectional")
    fulls = full_rows(current, "sad", "v1", [("lstm", "bidirectional")])
    lines = [
        "| split | files (train/valid/test) | DCF@0 | DCF@0.25 | DCF@0.5 | DCF@1 | DCF@2 | Pmiss/Pfa @0.5 | config | seed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows: list[BaselineRecord] = []
    if gate is None:
        lines.append(_tbd("subset gate", 9))
    else:
        lines.append(
            f"| subset gate ({_date(gate)}) | {_files(gate)} | {' | '.join(_dcf_cells(gate))} | {_pmiss_pfa(gate, '.2f')} "
            f"| {_config(gate)} | {gate.recipe.seed} |"
        )
        lines.append(
            f"| subset gate -- untrained init baseline | (same test set) | {' | '.join(_dcf_cells(gate, init=True))} | {_pmiss_pfa(gate, '.2f', init=True)} "
            f"| (seed pack, no training) | {gate.recipe.seed} |"
        )
        rows.append(gate)
    for r in fulls:
        lines.append(f"| full run ({_date(r)}) | {_files(r)} | {' | '.join(_dcf_cells(r))} | {_pmiss_pfa(r, '.2f')} | {_config(r)} | {r.recipe.seed} |")
        rows.append(r)
    if not fulls:
        lines.append(_tbd("full run", 9))
    return lines + _footer(rows, records)


# --------------------------------------------------------------------------------------- #
# Phases 9, 10, 11: the (cell x direction) SAD gate matrices
# --------------------------------------------------------------------------------------- #


def _matrix(
    records: list[Record], arm: str, lineage: str | None, cells: list[tuple[str, str]], scope: str, header: str, row: Callable[[BaselineRecord], str]
) -> list[str]:
    """One (cell x direction) table: a gate row per cell (TBD when none), then the full-run rows
    (one per record, TBD when none), the metric columns coming from `row`."""
    current = current_baselines(records)
    n = header.count("|") - 2
    lines = [header, "|" + "---|" * (n + 1)]
    rows: list[BaselineRecord] = []
    for cell, direction in cells:
        r = gate_row(current, arm, lineage, cell, direction)
        if r is None:
            lines.append(_tbd(f"{CELL_NAME[cell]} / {direction}", n))
            continue
        lines.append(f"| {_cell(r)} | {row(r)} |")
        rows.append(r)
    fulls = full_rows(current, arm, lineage, cells)
    for r in fulls:
        lines.append(f"| full run: {_cell(r)} ({_date(r)}, seed {r.recipe.seed}) | {row(r)} |")
        rows.append(r)
    if not fulls:
        lines.append(_tbd(f"full-corpus runs ({scope})", n))
    return lines + _footer(rows, records)


_DCF_HEADER = (
    "| cell / direction | files (train/valid/test) | trained DCF@0.5 | Pmiss / Pfa @0.5 | trained collar range "
    "| init DCF@0.5 | init collar range | gain@0.5 | wall |"
)


def _dcf_row(r: BaselineRecord) -> str:
    tr, ini = r.payload.collar(0.5), r.payload.collar(0.5, init=True)
    return (
        f"{_files(r)} | {_opt(tr.dcf if tr else None, '.6f')} | {_pmiss_pfa(r, '.6f')} | {_collar_range(r)} "
        f"| {_opt(ini.dcf if ini else None, '.6f')} | {_collar_range(r, init=True)} | {_gain(r)} | {_wall(r)}"
    )


def _cell_matrix(records: list[Record], arm: str, lineage: str, cells: list[tuple[str, str]], scope: str) -> list[str]:
    return _matrix(records, arm, lineage, cells, scope, _DCF_HEADER, _dcf_row)


_BOTH = ("bidirectional", "forward")


@table("phase9_v1_cells")
def phase9_v1_cells(records: list[Record]) -> list[str]:
    cells = [("lstm", "bidirectional"), *[(c, d) for c in ("slstm", "mamba") for d in _BOTH]]
    return _cell_matrix(records, "sad", "v1", cells, "all cells")


@table("phase10_v2_cells")
def phase10_v2_cells(records: list[Record]) -> list[str]:
    cells = [(c, d) for c in ("lstm", "slstm", "mamba", "cfc") for d in _BOTH]
    return _cell_matrix(records, "sad-v2", "v2", cells, "any cell x direction")


@table("phase11_v2_cells")
def phase11_v2_cells(records: list[Record]) -> list[str]:
    return _cell_matrix(records, "sad-v2", "v2", [("transformer", d) for d in _BOTH], "transformer")


# --------------------------------------------------------------------------------------- #
# Issue #21: the (cell x direction) LID matrices, one per LID arm (exact-tree numbers; the
# fast Twin LID runs the same matrix since #57, parity-pinned, not re-measured here). The
# LSTM / bidirectional row is the Phase-6 gate's.
# --------------------------------------------------------------------------------------- #

_ALL_CELLS = [(c, d) for c in CELL_TYPES for d in DIRECTIONS]
_LID_HEADER = "| cell / direction | files (train/valid/test) | trained LID error % | init LID error % | gain (pt) | Cavg | chance % | wall |"


def _lid_row(r: BaselineRecord) -> str:
    p = r.payload
    gain = "n/a" if p.lid_error is None or p.init_lid_error is None else f"{p.init_lid_error - p.lid_error:+.2f}"
    return f"{_files(r)} | {_opt(p.lid_error, '.2f')} | {_opt(p.init_lid_error, '.2f')} | {gain} | {_opt(p.cavg, '.2f')} | {CHANCE} | {_wall(r)}"


def _lid_cell_matrix(records: list[Record], arm: str) -> list[str]:
    return _matrix(records, arm, None, _ALL_CELLS, "any cell x direction", _LID_HEADER, _lid_row)


@table("lid_features_cells")
def lid_features_cells(records: list[Record]) -> list[str]:
    return _lid_cell_matrix(records, "lid-features")


@table("lid_phseq_cells")
def lid_phseq_cells(records: list[Record]) -> list[str]:
    return _lid_cell_matrix(records, "lid-phseq")


# --------------------------------------------------------------------------------------- #
# Phase 7: the exact-vs-fast bench matrix
# --------------------------------------------------------------------------------------- #

LEG_ORDER = tuple(LEGS)


def current_benches(records: Iterable[Record]) -> list[BenchRecord]:
    bench = [r for r in records if isinstance(r, BenchRecord)]
    dead = superseded_ids(bench)
    return [r for r in bench if r.id not in dead]


def bench_labels(records: Iterable[Record]) -> dict[str, tuple[int, Lineage | None]]:
    """`label -> (lanes, lineage)` over every bench record. A label names exactly one recipe
    (issue #39): `add` refuses the second, and a ledger holding two is an error here, never a
    cell pooling two lane counts or a speedup dividing one recipe's wall by another's."""
    labels: dict[str, tuple[int, Lineage | None]] = {}
    for r in records:
        if isinstance(r, BenchRecord):
            recipe = (r.recipe.lanes, r.recipe.lineage)
            if labels.setdefault(r.recipe.label, recipe) != recipe:
                raise ValueError(f"bench label {r.recipe.label!r} names two recipes, {labels[r.recipe.label]} and {recipe}; a label is one recipe")
    return labels


@dataclass(frozen=True)
class BenchCell:
    """One (label, path, host) cell: the runs of every record sharing the latest SHA and build."""

    label: str
    path: str
    host: str
    git_sha: str
    audio_s: float
    files: int
    walls: tuple[float, ...]
    maxrss: tuple[float, ...]
    records: tuple[BenchRecord, ...]

    @property
    def wall_mean(self) -> float:
        return sum(self.walls) / len(self.walls)

    @property
    def maxrss_mean(self) -> float:
        return sum(self.maxrss) / len(self.maxrss)


def bench_cells(records: Iterable[Record]) -> list[BenchCell]:
    """Group the current bench records by (label, path, host), keep the (SHA, build) group a
    cell's most recent record belongs to, and pool that group's runs. A label is one recipe
    (`bench_labels`), so the pool never mixes lane counts or lineages."""
    records = list(records)
    bench_labels(records)
    by_cell: dict[tuple[str, str, str], list[BenchRecord]] = {}
    for r in current_benches(records):
        by_cell.setdefault((r.recipe.label, r.recipe.path, host_key(r)), []).append(r)
    cells: list[BenchCell] = []
    for (label, path, host), recs in by_cell.items():
        newest = max(recs, key=lambda r: (r.recorded_at, r.id))
        group = [r for r in recs if (r.git_sha, r.build) == (newest.git_sha, newest.build)]
        runs = [run for r in group for run in r.payload.runs]
        cells.append(
            BenchCell(
                label=label,
                path=path,
                host=host,
                git_sha=newest.git_sha,
                audio_s=runs[0].audio_s,
                files=runs[0].files,
                walls=tuple(run.wall_s for run in runs),
                maxrss=tuple(run.maxrss_mb for run in runs),
                records=tuple(sorted(group, key=lambda r: (r.recorded_at, r.id))),
            )
        )
    return cells


def speedups(cells: Iterable[BenchCell]) -> dict[tuple[str, str], float]:
    """`(label, host) -> exact mean wall / fast mean wall` where both paths exist."""
    by_key: dict[tuple[str, str], dict[str, BenchCell]] = {}
    for c in cells:
        by_key.setdefault((c.label, c.host), {})[c.path] = c
    return {k: v["exact"].wall_mean / v["fast"].wall_mean for k, v in by_key.items() if "exact" in v and "fast" in v}


def latest_host(cells: Iterable[BenchCell]) -> str | None:
    """The host of the most recently recorded bench cell: the machine the prose speaks for."""
    cells = list(cells)
    if not cells:
        return None
    return max(cells, key=lambda c: max((r.recorded_at, r.id) for r in c.records)).host


def _leg_name(label: str) -> str:
    return LEGS[label].display if label in LEGS else label


@table("phase7_bench_matrix")
def phase7_bench_matrix(records: list[Record]) -> list[str]:
    cells = bench_cells(records)
    ratio = speedups(cells)
    labels = [*LEG_ORDER, *sorted({c.label for c in cells} - set(LEG_ORDER))]
    hosts = sorted({c.host for c in cells})
    header = [
        "| leg | audio_s | path | wall_s (mean [range], n) | rtf | maxrss_mib | MiB/audio-s | speedup (wall, fast vs exact) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines: list[str] = []
    rows: list[BenchRecord] = []
    for host in hosts or [None]:  # type: ignore[list-item]
        if host is not None and len(hosts) > 1:
            lines += [f"Host: {host}.", ""]
        lines += header
        for label in labels:
            for path in ("exact", "fast"):
                cell = next((c for c in cells if c.label == label and c.path == path and c.host == host), None)
                if cell is None:
                    lines.append(f"| {_leg_name(label)} | TBD | {path} | TBD | TBD | TBD | TBD | TBD |")
                    continue
                rows.extend(cell.records)
                speed = "baseline" if path == "exact" else (f"{ratio[(label, host)]:.2f}x" if (label, host) in ratio else "n/a")
                wall = f"{cell.wall_mean:.4f} [{min(cell.walls):.4f}-{max(cell.walls):.4f}] (n={len(cell.walls)})"
                lines.append(
                    f"| {_leg_name(label)} | {cell.audio_s:.2f} | {path} | {wall} | {cell.wall_mean / cell.audio_s:.6f} "
                    f"| {cell.maxrss_mean:.3f} | {cell.maxrss_mean / cell.audio_s:.4f} | {speed} |"
                )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines + _footer(rows, records)
