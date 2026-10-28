"""Issue #110: the fast tree's training refusal through the seam, one site and mode-free.

  * `Engine.from_map` under `Inference_Path fast` refuses a training-shaped map at
    construction in Image mode as in Multi mode, with one wording naming the four facts
    (epochs, gradient-check epsilon, both `BackPropagationActivated` flags); the exact tree
    builds the same map.
  * A gradient check is training-shaped: `Neural_Networks_Gradient_Check_Epsilon > 0` on a
    fast map refuses in UnitTest mode, before the mode's output-file truncation.
  * `Engine.grad_check` on a fast engine answers `[]`, the exact engine's answer with backprop
    off (decision 4: after the refusal a fast engine always has backprop off, so the empty
    list is the truthful zero-active-nets answer, not a silent no-op).

Rows run on the phase-9 forward sLSTM SAD fixture (its config carries
`BLSTM_BackPropagationActivated true`), seeded into a tmp dir.
"""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path
from typing import Any

import pytest
from speech.config_bridge import parse_legacy_config

speech_rs = pytest.importorskip("speech_rs")

from tests.pyo3.test_phase9_seam import seed_sad  # noqa: E402

FIXTURE = "slstm_forward"
REFUSAL = "Inference_Path fast is inference-only"


def _cfg(tmp_path: Path, path: str, **overlay: str) -> dict[str, str]:
    seed_sad(tmp_path, FIXTURE)
    cfg = parse_legacy_config((tmp_path / f"{FIXTURE}.config").read_text())
    cfg["Inference_Path"] = path
    cfg.update(overlay)
    return cfg


def _inference_only(cfg: dict[str, str]) -> dict[str, str]:
    return {**cfg, "Neural_Networks_BackPropagation_Epochs": "0", "BLSTM_BackPropagationActivated": "false"}


def _build(tmp_path: Path, cfg: dict[str, str], mode: str) -> Any:
    with chdir(tmp_path):
        return speech_rs.Engine.from_map([cfg], mode)


@pytest.mark.parametrize("mode", ["-i", "-m"])
def test_fast_refuses_training_shaped_map_at_construction(tmp_path: Path, mode: str) -> None:
    cfg = _cfg(tmp_path, "fast", Neural_Networks_BackPropagation_Epochs="2")
    assert cfg["BLSTM_BackPropagationActivated"] == "true", "the fixture must carry the flag on"
    with pytest.raises(RuntimeError, match=REFUSAL) as refused:
        _build(tmp_path, cfg, mode)
    msg = str(refused.value)
    for fact in ("Neural_Networks_BackPropagation_Epochs 2", "Neural_Networks_Gradient_Check_Epsilon 0", "BackPropagationActivated SAD=true/LID=false"):
        assert fact in msg, msg
    assert "mode" not in msg.lower(), f"the wording names the facts, not the mode: {msg}"
    # The exact tree builds the same map (training stays exact f64).
    _build(tmp_path, {**cfg, "Inference_Path": "exact"}, mode)


def test_fast_refuses_gradient_check_epsilon_in_unit_test_mode(tmp_path: Path) -> None:
    cfg = _inference_only(_cfg(tmp_path, "fast", Neural_Networks_Gradient_Check_Epsilon="1e-4"))
    with pytest.raises(RuntimeError, match=REFUSAL) as refused:
        _build(tmp_path, cfg, "-t")
    assert "Neural_Networks_Gradient_Check_Epsilon 0.0001" in str(refused.value)
    assert not (tmp_path / "MultiConfigResults.mat").exists(), "the refusal precedes the UnitTest-mode output-file truncation"
    _build(tmp_path, {**cfg, "Inference_Path": "exact"}, "-t")


def test_fast_grad_check_is_the_exact_backprop_off_answer(tmp_path: Path) -> None:
    fast = _build(tmp_path, _inference_only(_cfg(tmp_path, "fast")), "-m")
    exact = _build(tmp_path, _inference_only(_cfg(tmp_path, "exact")), "-m")
    assert fast.grad_check(1e-4, 1) == [] == exact.grad_check(1e-4, 1)
