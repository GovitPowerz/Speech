"""The committed runners under `experiments/` (issue #21) parse, and every row they name is an arm,
a cell and a direction the launcher knows. No corpus: a runner is only ever fired by hand."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from speech.config_bridge import CELL_TYPES, DIRECTIONS
from speech.drivers.baseline import build_parser
from speech.drivers.spec import ARM_CONFIG, BaselineSpec
from speech.ledger.tables import TABLES

REPO = Path(__file__).resolve().parents[1]
RUNNERS = sorted((REPO / "experiments").glob("*.sh"))
_ROW = re.compile(r"\b(" + "|".join(re.escape(arm) for arm in ARM_CONFIG) + r")/([a-z]+)/([a-z]+)\b")


@pytest.mark.parametrize("script", RUNNERS, ids=lambda p: p.name)
def test_runner_parses(script: Path) -> None:
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_runners_exist_for_every_tbd_table() -> None:
    assert [p.name for p in RUNNERS] == [
        "00_prereqs.sh",
        "01_phase6_full_runs.sh",
        "02_v1_cells.sh",
        "03_v2_cells.sh",
        "04_v2_transformer.sh",
        "05_lid_cells.sh",
        "lib.sh",
    ]
    # Every rendered table carrying a TBD full-run row is named by a runner's header.
    text = "\n".join(p.read_text() for p in RUNNERS)
    tbd = [name for name, render in TABLES.items() if any("TBD" in line and "full" in line for line in render([]))]
    assert tbd and [name for name in tbd if name not in text] == []


@pytest.mark.parametrize("script", RUNNERS, ids=lambda p: p.name)
def test_every_named_row_is_in_the_launchers_vocabulary(script: Path) -> None:
    rows = _ROW.findall(script.read_text())
    for arm, cell, direction in rows:
        assert arm in ARM_CONFIG and cell in CELL_TYPES and direction in DIRECTIONS, (script.name, arm, cell, direction)
    # Every `for cell in ...` loop draws its cells from the same vocabulary.
    for cell in re.findall(r"for cell in ([a-z ]+);", script.read_text()):
        assert set(cell.split()) <= set(CELL_TYPES), (script.name, cell)


def _fired(tmp_path: Path, script: Path) -> list[list[str]]:
    """Fire a runner from a copy of `experiments/` against a fake `uv` that logs each launcher
    argv (and accepts every `ledger add`) instead of training."""
    exp = tmp_path / "experiments"
    shutil.copytree(REPO / "experiments", exp)
    log = tmp_path / "argv.log"
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(f'#!/bin/sh\n[ "$4" = speech.drivers.baseline ] && echo "$@" >> {log}\nexit 0\n')
    uv.chmod(0o755)
    res = subprocess.run(["bash", str(exp / script.name)], env={**os.environ, "PATH": f"{uv.parent}:{os.environ['PATH']}"}, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return [line.split()[4:] for line in log.read_text().splitlines()]


@pytest.mark.parametrize("script", [p for p in RUNNERS if p.name[:2].isdigit() and p.name != "00_prereqs.sh"], ids=lambda p: p.name)
def test_every_runner_row_is_a_valid_full_run(tmp_path: Path, script: Path) -> None:
    """The recipes live in `lib.sh`, not in prose a docs test parses, so a flag typo there would
    surface only when a multi-hour row is fired: each row's argv must build a full-run spec."""
    calls = _fired(tmp_path, script)
    assert calls
    for argv in calls:
        spec = BaselineSpec.from_args(build_parser().parse_args(argv))
        assert spec.recipe(listing=None).full and (spec.epochs, spec.steps_per_epoch, spec.patience, spec.seed) == (40, 25, 6, 0), argv
        assert (spec.audio_max_duration == 120.0) == (spec.arm in ("sad", "sad-v2")), argv
