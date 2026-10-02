"""The RESULTS.md tables rendered from the ledger (issue #20).

Each renderer is a function over the loaded records, registered by the name a
`<!-- ledger:table <name> -->` marker in RESULTS.md uses. Shared rules: a table takes the latest
non-superseded record per recipe; a subset-gate row is the latest `gate` record of its
(arm, lineage, cell, direction); a full-run row is one row per non-superseded full record (a
seed is part of the recipe, so seeds are rows, never averaged) and a single `TBD` row when none
exists; every table ends with the hosts its rows came from and one footnote line per
supersession chain. Prose around a table stays hand-written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from speech.ledger.schema import BaselineRecord, BenchRecord, superseded_ids

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
    """The latest non-superseded baseline record per recipe, in recorded order."""
    base = [r for r in records if isinstance(r, BaselineRecord)]
    dead = superseded_ids(base)
    latest: dict[tuple[object, ...], BaselineRecord] = {}
    for r in sorted(base, key=lambda r: (r.recorded_at, r.id)):
        if r.id not in dead:
            latest[r.recipe.key()] = r
    return list(latest.values())


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


def _footer(rows: list[BaselineRecord], all_records: Iterable[Record]) -> list[str]:
    """The hosts the rows came from, then one line per supersession chain touching them."""
    lines: list[str] = []
    hosts = sorted({f"{r.host.chip} ({r.host.arch}, {r.host.cores} cores, {r.host.os})" for r in rows})
    if hosts:
        lines += ["", "Hosts: " + "; ".join(hosts) + "."]
    by_id = {r.id: r for r in all_records}
    for r in sorted(rows, key=lambda r: r.id):
        old_id = r.supersedes
        while old_id is not None:
            lines.append(f"Superseded: `{old_id}` by `{r.id}` ({r.reason}).")
            old = by_id.get(old_id)
            old_id = old.supersedes if old is not None else None
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


def _cell_matrix(records: list[Record], arm: str, lineage: str, cells: list[tuple[str, str]], scope: str) -> list[str]:
    current = current_baselines(records)
    lines = [
        "| cell / direction | files (train/valid/test) | trained DCF@0.5 | Pmiss / Pfa @0.5 | trained collar range "
        "| init DCF@0.5 | init collar range | gain@0.5 | wall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows: list[BaselineRecord] = []
    for cell, direction in cells:
        r = gate_row(current, arm, lineage, cell, direction)
        if r is None:
            lines.append(_tbd(f"{CELL_NAME[cell]} / {direction}", 8))
            continue
        tr, ini = r.payload.collar(0.5), r.payload.collar(0.5, init=True)
        lines.append(
            f"| {_cell(r)} | {_files(r)} | {_opt(tr.dcf if tr else None, '.6f')} | {_pmiss_pfa(r, '.6f')} | {_collar_range(r)} "
            f"| {_opt(ini.dcf if ini else None, '.6f')} | {_collar_range(r, init=True)} | {_gain(r)} | {_wall(r)} |"
        )
        rows.append(r)
    fulls = full_rows(current, arm, lineage, cells)
    for r in fulls:
        tr, ini = r.payload.collar(0.5), r.payload.collar(0.5, init=True)
        lines.append(
            f"| full run: {_cell(r)} ({_date(r)}, seed {r.recipe.seed}) | {_files(r)} | {_opt(tr.dcf if tr else None, '.6f')} | {_pmiss_pfa(r, '.6f')} "
            f"| {_collar_range(r)} | {_opt(ini.dcf if ini else None, '.6f')} | {_collar_range(r, init=True)} | {_gain(r)} | {_wall(r)} |"
        )
        rows.append(r)
    if not fulls:
        lines.append(_tbd(f"full-corpus runs ({scope})", 8))
    return lines + _footer(rows, records)


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
