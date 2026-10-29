"""Issue #90: `speech_rs.blstm_config(config, prefix)` accepts every config the engine accepts.

`NNetInputSize` is the one entry the engine does not parse for itself: the exact net is built
`LSTMNeuronNb[0]` wide (`OutputNeuronNb[0]` in MLP mode) and never reads the key, and the
committed LID Twin config declares it on neither net. The seam reports the declared value when
the key is present (malformed -> `ValueError` naming the key, ADR-0006) and the engine's width
when it is absent, so the call succeeds whenever `BlstmConfig::from_legacy` does.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from speech.config_bridge import parse_legacy_config

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
LID_TWIN = REPO_ROOT / "configs" / "legacy" / "LID_BLSTM.config"
WORKER = REPO_ROOT / "tests" / "reference_data" / "phase0" / "1_worker_1.config"


def _config(flat: dict[str, str], prefix: str) -> dict[str, Any]:
    return cast(dict[str, Any], speech_rs.blstm_config(flat, prefix))


@pytest.mark.parametrize(("prefix", "width"), [("BLSTM_LID", 36), ("BLSTM", 11)])
def test_the_committed_lid_twin_declares_no_input_size_and_both_nets_report_the_engines_width(prefix: str, width: int) -> None:
    flat = parse_legacy_config(LID_TWIN.read_text())
    assert f"{prefix}_NNetInputSize" not in flat
    spec = _config(flat, prefix)
    assert spec["NNetInputSize"] == width
    assert spec["LSTMNeuronNb"][0] == width


def test_a_declared_input_size_is_returned_as_declared() -> None:
    flat = parse_legacy_config(WORKER.read_text())
    assert flat["BLSTM_NNetInputSize"] == "23"
    assert _config(flat, "BLSTM")["NNetInputSize"] == 23


def test_a_malformed_declared_input_size_raises_naming_the_key() -> None:
    flat = parse_legacy_config(WORKER.read_text())
    flat["BLSTM_NNetInputSize"] = "2E"
    with pytest.raises(ValueError, match="BLSTM_NNetInputSize"):
        speech_rs.blstm_config(flat, "BLSTM")
