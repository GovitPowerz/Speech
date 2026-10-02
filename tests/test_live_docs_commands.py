"""The commands the live documents tell a reader to run exist, parse and run (issue #30).

RESULTS.md and the module READMEs named a `speech` subcommand `baseline` for three phases; the
Rust binary `speech` has `bench`, `stream` and `stream-lid` only, and no Python console script is
installed. The working entries are `python -m speech.drivers.baseline` and `python -m speech.cli`.
The first test keeps the stale spelling out of every tracked text file (the dated specs and plans
under docs/superpowers/ are historical records and are not scanned); the second feeds each
RESULTS.md and README.md launcher recipe through the real parser and checks its output directory
is gitignored; the third runs both module entries, since a module without a `__main__` guard
imports and exits 0 without doing anything.
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from speech.drivers import baseline as B

from tests.test_license_hygiene import tracked_text_files

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = "uv run python -m speech.drivers.baseline "
# The removed launcher, not the prose "all-non-speech baseline".
STALE = re.compile(r"(?<![\w-])speech baseline\b")


def test_no_tracked_file_names_the_stale_launcher() -> None:
    historical, this_file = REPO / "docs" / "superpowers", Path(__file__).resolve()
    scanned = [p for p in tracked_text_files(REPO) if not p.is_symlink() and not p.is_relative_to(historical) and p != this_file]
    hits = [f"{p}:{n}" for p in scanned for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1) if STALE.search(line)]
    assert hits == []


def test_launcher_recipes_parse_and_write_to_ignored_dirs() -> None:
    recipes = [
        shlex.split(line[len(LAUNCHER) :])
        for doc in ("RESULTS.md", "README.md")
        for line in (REPO / doc).read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
        if line.startswith(LAUNCHER)
    ]
    assert len(recipes) >= 7, recipes  # the six RESULTS.md recipes plus the README quick start
    for argv in recipes:
        out_dir = B.build_parser().parse_args(argv).out_dir
        assert subprocess.run(["git", "check-ignore", "-q", str(out_dir / "base.config")], cwd=REPO).returncode == 0, argv


@pytest.mark.parametrize("module", ["speech.drivers.baseline", "speech.cli"])
def test_module_entry_runs(module: str) -> None:
    res = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO, capture_output=True, text=True)
    assert res.returncode == 0 and res.stdout.startswith("usage: ") and f"-m {module} " in res.stdout.splitlines()[0], res.stdout
