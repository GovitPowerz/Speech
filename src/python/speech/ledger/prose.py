"""The numbers the live documents quote in prose, asserted against the ledger (issue #20).

A table marker cannot own a number inside a sentence, so the README and ARCHITECTURE sentences
that quote the Phase 7 speedups are registered here: file, a regex with one capture group per
quoted number, and the derived values they must equal. The values come from the latest bench
pair per leg on the most recently recording host. A one-decimal claim ("SAD 4.6x") is the
lower of that task's two legs at one decimal (the conservative reading of "runs 4.6x faster");
a range ("4.58-4.60x") is min-max over the legs at two decimals, collapsing to one value when
both legs round the same. The SAD peak RSS (issue #41) is the `phase7_60s` cells' `maxrss_mean`
on that host, fast and exact, at one decimal in the README and rounded to the MB in
ARCHITECTURE (Python's format rounding, half to even on the binary float). RESULTS.md's Phase 7 prose is history and is not asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from speech.ledger.schema import REPO
from speech.ledger.tables import Record, bench_cells, latest_host, speedups

SAD_LEGS = ("phase7_60s", "phase7_sad_corpus")
LID_LEGS = ("phase7_lid_phseq", "phase7_lid_cep")
RSS_LEG = "phase7_60s"


@dataclass(frozen=True)
class Quote:
    file: str
    pattern: str
    values: tuple[str, ...]  # one derived-value name per capture group


QUOTES: tuple[Quote, ...] = (
    Quote("README.md", r"runs SAD (\d+\.\d)x and LID (\d+\.\d)x faster end to end", ("sad_1dp", "lid_1dp")),
    Quote("README.md", r"\| SAD (\d+\.\d\d(?:-\d+\.\d\d)?)x, LID (\d+\.\d\d(?:-\d+\.\d\d)?)x end to end", ("sad_range", "lid_range")),
    Quote("README.md", r"fast SAD peak RSS (\d+\.\d) MB vs (\d+\.\d) exact", ("sad_rss_fast_1dp", "sad_rss_exact_1dp")),
    Quote("docs/ARCHITECTURE.md", r"runs SAD (\d+\.\d)x and LID (\d+\.\d)x faster end to end", ("sad_1dp", "lid_1dp")),
    Quote("docs/ARCHITECTURE.md", r"fast SAD path peaks at (\d+) MB of RSS against the exact tree's (\d+)", ("sad_rss_fast_0dp", "sad_rss_exact_0dp")),
)


def derived(records: list[Record]) -> dict[str, str]:
    """The quotable values, or an empty dict when a leg has no bench pair yet."""
    cells = bench_cells(records)
    host = latest_host(cells)
    ratio = speedups(cells)
    out: dict[str, str] = {}
    for task, legs in (("sad", SAD_LEGS), ("lid", LID_LEGS)):
        values = [ratio[(leg, host)] for leg in legs if host is not None and (leg, host) in ratio]
        if len(values) != len(legs):
            return {}
        lo, hi = f"{min(values):.2f}", f"{max(values):.2f}"
        out[f"{task}_1dp"] = f"{min(values):.1f}"
        out[f"{task}_range"] = lo if lo == hi else f"{lo}-{hi}"
    rss = {c.path: c.maxrss_mean for c in cells if c.label == RSS_LEG and c.host == host}
    for path in ("fast", "exact"):
        out[f"sad_rss_{path}_1dp"] = f"{rss[path]:.1f}"
        out[f"sad_rss_{path}_0dp"] = f"{rss[path]:.0f}"
    return out


def matches(repo: Path = REPO) -> list[tuple[Quote, re.Match[str] | None, int]]:
    """Each quote with its match in its file and the number of matches found."""
    out = []
    for q in QUOTES:
        text = (repo / q.file).read_text(encoding="utf-8")
        found = list(re.finditer(q.pattern, text))
        out.append((q, found[0] if found else None, len(found)))
    return out


def check(records: list[Record], repo: Path = REPO) -> list[str]:
    """Every registered quote matches exactly once and equals its ledger-derived value."""
    want = derived(records)
    problems: list[str] = []
    if not want:
        return ["the ledger holds no complete bench pair for the four Phase 7 legs"]
    for q, m, n in matches(repo):
        if m is None or n != 1:
            problems.append(f"{q.file}: pattern {q.pattern!r} matched {n} times, expected 1")
            continue
        for group, name in zip(m.groups(), q.values, strict=True):
            if group != want[name]:
                problems.append(f"{q.file}: quotes {group} for {name}, the ledger says {want[name]}")
    return problems
