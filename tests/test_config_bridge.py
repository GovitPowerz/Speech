"""Legacy .config parsing tests (Python side)."""

from pathlib import Path

from speech import config_bridge

FIX = Path("tests/reference_data/phase0/1_worker_1.config")


def test_parse_last_wins() -> None:
    cfg = config_bridge.parse_legacy_config("a 1\n# c 2\na 3\n")
    assert cfg["a"] == "3"
    assert "c" not in cfg


def test_nnet_spec_from_real_config() -> None:
    cfg = config_bridge.parse_legacy_config(FIX.read_text())
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    assert spec["LSTMNeuronNb"] == [23, 24, 24]
    assert spec["LSTMSubSampling"] == [4, 1]
    assert spec["OutputNeuronNb"] == [48, 12, 1]
    assert spec["OutputSubSampling"] == [1, 1]
    assert spec["NNetInputSize"] == 23
    assert spec["peepholes"] == [True, True, True, True, True, True]
    assert spec["CostLawSpeech"] == "log"
    assert spec["CostLawNoSpeech"] == "log"
