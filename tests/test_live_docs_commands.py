"""The commands the live documents tell a reader to run exist and parse (issue #30).

RESULTS.md and the module READMEs named a `speech baseline` subcommand for three phases; the Rust
binary `speech` has `bench`, `stream` and `stream-lid` only, and no Python console script is
installed. The working entry is `python -m speech.drivers.baseline`. The first test keeps the stale
spelling out of every live document; the second feeds each RESULTS.md launcher recipe through the
real parser, so a recipe that names a flag the parser does not have fails here, not on the reader.
The dated specs and plans under docs/superpowers/ are historical records and are not scanned.
"""

import re
import shlex
from pathlib import Path

from speech.drivers import baseline as B

REPO = Path(__file__).resolve().parents[1]
LIVE_DOCS = (
    "README.md",
    "RESULTS.md",
    "CLAUDE.md",
    "DEVELOPMENT.md",
    "CONTEXT.md",
    "docs/ROADMAP.md",
    "docs/ARCHITECTURE.md",
    "docs/validation.md",
    "src/rust/README.md",
    "src/python/speech/README.md",
)
LAUNCHER = "uv run python -m speech.drivers.baseline "


def results_md_recipes() -> list[list[str]]:
    """Every fenced launcher command in RESULTS.md, backslash continuations joined, as argv."""
    lines = (REPO / "RESULTS.md").read_text(encoding="utf-8").splitlines()
    recipes: list[list[str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(LAUNCHER):
            cmd = lines[i]
            while cmd.rstrip().endswith("\\"):
                i += 1
                cmd = cmd.rstrip()[:-1] + " " + lines[i].strip()
            recipes.append(shlex.split(cmd[len(LAUNCHER) :]))
        i += 1
    return recipes


def test_no_live_document_names_the_stale_launcher() -> None:
    stale = re.compile(r"\bspeech baseline\b")
    hits = [f"{doc}:{n}" for doc in LIVE_DOCS for n, line in enumerate((REPO / doc).read_text(encoding="utf-8").splitlines(), 1) if stale.search(line)]
    assert hits == []


def test_results_md_launcher_recipes_parse() -> None:
    recipes = results_md_recipes()
    assert len(recipes) >= 5, recipes  # the three Phase 6 arms plus the per-cell recipes of later phases
    for argv in recipes:
        args = B.build_parser().parse_args(argv)
        assert args.arm in B._ARM_CONFIGS
