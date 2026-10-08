"""The committed runners under `experiments/` (issue #21) parse, and every row they name is an arm,
a cell and a direction the launcher knows. No corpus: a runner is only ever fired by hand."""

import re
import subprocess
from pathlib import Path

import pytest
from speech.config_bridge import CELL_TYPES, DIRECTIONS
from speech.drivers.spec import ARM_CONFIG

REPO = Path(__file__).resolve().parents[1]
RUNNERS = sorted((REPO / "experiments").glob("*.sh"))
_ROW = re.compile(r"\b(sad(?:-v2)?|lid-features|lid-phseq)/([a-z]+)/([a-z]+)\b")


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


@pytest.mark.parametrize("script", RUNNERS, ids=lambda p: p.name)
def test_every_named_row_is_in_the_launchers_vocabulary(script: Path) -> None:
    rows = _ROW.findall(script.read_text())
    for arm, cell, direction in rows:
        assert arm in ARM_CONFIG and cell in CELL_TYPES and direction in DIRECTIONS, (script.name, arm, cell, direction)
    # Every `ROWS+=("$arm/$cell/$direction")` loop draws its cells from the same vocabulary.
    for cell in re.findall(r"for cell in ([a-z ]+);", script.read_text()):
        assert set(cell.split()) <= set(CELL_TYPES), (script.name, cell)
