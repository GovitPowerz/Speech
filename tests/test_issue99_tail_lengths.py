"""Issue #99: the normalize tail lengths come from the engine's weight layout, not from a
config key read in Python.

`_tail_lengths` feeds `_backprop_inner`'s strip of each net's mean/std tail. Issue #27 moved
every layout fact to Rust (`speech_rs.weight_layout`); this one stayed as `2 * NNetInputSize`
arithmetic on the config map, which under ADR-0008 is exactly the number that can disagree with
the width the engine builds, and which the committed LID configs do not even carry. Pinned on
both SAD lineages, the committed LID config and the two legacy-format fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import speech_rs
from speech.config_bridge import parse_legacy_config
from speech.drivers.train import _tail_lengths

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"
PHASE4D = REPO_ROOT / "tests" / "reference_data" / "phase4d"

CASES: dict[str, tuple[Path, int]] = {
    "lre_sad_v1": (CONFIGS / "training" / "lre_sad.toml", 3),
    "lre_sad_v2": (CONFIGS / "training" / "lre_sad_v2.toml", 3),
    "lid_blstm": (CONFIGS / "lid" / "lid_blstm.toml", 6),
    "twin_train": (PHASE4B / "twin_train.config", 6),
    "parity_tupleA": (PHASE4D / "parity_tupleA.config", 3),
}


def _cfg(path: Path) -> dict[str, str]:
    if path.suffix == ".toml":
        return cast(dict[str, str], speech_rs.load_toml_config(str(path)))
    return parse_legacy_config(path.read_text())


def _engine_tail(cfg: dict[str, str], prefix: str) -> int:
    """The length of the `tail` stack the engine's layout ends with: the pack's last blocks."""
    layout = cast(list[dict[str, Any]], speech_rs.weight_layout(cfg, prefix))
    tail = [b for b in layout if b["stack"] == "tail"]
    assert tail and tail[-1]["offset"] + tail[-1]["len"] == layout[-1]["offset"] + layout[-1]["len"]
    return int(sum(b["len"] for b in tail))


@pytest.mark.parametrize("name", sorted(CASES))
def test_tail_lengths_are_the_engine_tail_stack(name: str) -> None:
    path, algo = CASES[name]
    cfg = _cfg(path)
    prefixes = ["BLSTM", "BLSTM_LID"] if algo == 6 else ["BLSTM"]
    assert _tail_lengths(cfg, algo) == [_engine_tail(cfg, p) for p in prefixes]


def test_tail_lengths_need_no_input_size_key() -> None:
    """The committed LID config declares no `NNetInputSize` on either net; the engine derives
    the widths (11 and 36) and so must the helper."""
    cfg = _cfg(CONFIGS / "lid" / "lid_blstm.toml")
    assert "BLSTM_NNetInputSize" not in cfg and "BLSTM_LID_NNetInputSize" not in cfg
    assert _tail_lengths(cfg, 6) == [22, 72]


def test_drivers_hold_no_input_size_arithmetic() -> None:
    """ADR-0005: no Python geometry. The key never appears in the drivers' source."""
    hits = sorted(p.name for p in (REPO_ROOT / "src" / "python" / "speech" / "drivers").glob("*.py") if "NNetInputSize" in p.read_text())
    assert hits == []
