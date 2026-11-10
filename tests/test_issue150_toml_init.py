"""Issue #150: `speech init` reads a canonical TOML config, not only a legacy `.config`
(ADR-0006). `RunState.from_config` reads every config through the engine's one loader,
`speech_rs.load_config`."""

from __future__ import annotations

from pathlib import Path

import pytest
from speech.cli import main
from speech.drivers.state import RunState

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    ("config", "algo"),
    [(CONFIGS / "lid" / "lid_blstm.toml", 6), (CONFIGS / "training" / "lre_sad.toml", 3)],
    ids=["lid_blstm", "lre_sad"],
)
def test_init_reads_a_toml_config(config: Path, algo: int, tmp_path: Path) -> None:
    assert main(["init", str(config), str(tmp_path)]) == 0
    state = RunState.load(tmp_path / "run_state.json")
    assert state.ps.algo == algo
    assert state.config_path == str(config.resolve())
