"""Rewrite the ledger-owned tables of RESULTS.md between their markers (issue #20).

A block is `<!-- ledger:table <name> -->` ... `<!-- ledger:end -->`; everything between the two
markers is replaced by the named renderer's output over the loaded records. `check` renders in
memory and reports whether the file already holds that output, which is what CI runs.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from speech.ledger.schema import LEDGER_DIR, REPO, load
from speech.ledger.tables import TABLES, Record

RESULTS = REPO / "RESULTS.md"
BEGIN = re.compile(r"^<!-- ledger:table (?P<name>[a-z0-9_]+) -->$")
END = "<!-- ledger:end -->"


def render_text(text: str, records: list[Record]) -> str:
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = BEGIN.match(lines[i])
        if m is None:
            out.append(lines[i])
            i += 1
            continue
        name = m.group("name")
        if name not in TABLES:
            raise ValueError(f"line {i + 1}: no ledger table named {name!r} (known: {sorted(TABLES)})")
        out.append(lines[i])
        out.extend(TABLES[name](records))
        i += 1
        while i < len(lines) and lines[i] != END:
            if BEGIN.match(lines[i]):
                raise ValueError(f"line {i + 1}: ledger table {name!r} has no end marker before the next table")
            i += 1
        if i == len(lines):
            raise ValueError(f"ledger table {name!r} has no end marker")
        out.append(END)
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def table_names(text: str) -> list[str]:
    return [m.group("name") for line in text.splitlines() if (m := BEGIN.match(line))]


def render_file(path: Path = RESULTS, root: Path = LEDGER_DIR, *, check: bool = False) -> str:
    """Rewrite `path` in place (or, with `check`, leave it alone) and return the unified diff
    between the file and its rendering: empty when the file was already current."""
    before = path.read_text(encoding="utf-8")
    after = render_text(before, load(root))
    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), str(path), f"{path} (rendered)"))
    if diff and not check:
        path.write_text(after, encoding="utf-8")
    return diff
