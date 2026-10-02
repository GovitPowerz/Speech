"""Shared pytest fixtures for the speech test suite."""

import os
import pathlib
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from hypothesis import settings

if TYPE_CHECKING:
    from speech.drivers.baseline import BaselineResult
else:
    BaselineResult = object

# Licensed LRE03/07 corpus root (27 GB, gitignored, absent in CI). Phase-6 corpus-gated
# tests mark themselves `@requires_corpus` so they skip cleanly when it is missing.
CORPUS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "LRE03-LRE07"

requires_corpus = pytest.mark.skipif(
    not CORPUS_ROOT.is_dir(),
    reason=f"licensed LRE03/07 corpus absent at {CORPUS_ROOT}",
)

# Derandomized profile for CI reproducibility (a flaky property-test failure with no
# replayable seed is useless in CI logs). Local runs default to "dev" (randomized,
# fresh examples every run); set HYPOTHESIS_PROFILE=ci to match CI locally.
settings.register_profile("ci", derandomize=True)
settings.register_profile("dev", derandomize=False)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


def pytest_addoption(parser: pytest.Parser) -> None:
    # The ledger (issue #20): a subset gate's run directory is a `tmp_path`, gone after the
    # test, so a gate copies its record here for `python -m speech.ledger add <dir>/*.json`.
    parser.addoption("--ledger-stage", type=pathlib.Path, default=None, help="directory the subset gates copy their ledger records into")


GateRecorder = Callable[[BaselineResult], pathlib.Path]


@pytest.fixture
def gate_record(request: pytest.FixtureRequest) -> GateRecorder:
    """Turn a passing gate's `BaselineResult` into its `source="gate"` ledger record: written
    over the run directory's `record.json`, and copied to `--ledger-stage` when given."""
    from speech.ledger.schema import write_record

    stage: pathlib.Path | None = request.config.getoption("--ledger-stage")

    def record(res: BaselineResult) -> pathlib.Path:
        rec = res.to_record("gate", test=request.node.nodeid)
        path = write_record(rec, res.out_dir / "record.json")
        if stage is not None:
            stage.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, stage / f"{rec.id}.json")
        return path

    return record
