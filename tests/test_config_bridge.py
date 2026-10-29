"""Legacy .config parsing tests (Python side)."""

from pathlib import Path

import speech_rs
from speech import config_bridge

FIX = Path("tests/reference_data/phase0/1_worker_1.config")


def test_parse_last_wins() -> None:
    cfg = config_bridge.parse_legacy_config("a 1\n# c 2\na 3\n")
    assert cfg["a"] == "3"
    assert "c" not in cfg


def test_blstm_config_from_real_config() -> None:
    cfg = config_bridge.parse_legacy_config(FIX.read_text())
    spec = speech_rs.blstm_config(cfg, "BLSTM")
    assert spec["LSTMNeuronNb"] == [23, 24, 24]
    assert spec["LSTMSubSampling"] == [4, 1]
    assert spec["OutputNeuronNb"] == [48, 12, 1]
    assert spec["OutputSubSampling"] == [1, 1]
    assert spec["NNetInputSize"] == 23
    assert spec["peepholes"] == [True, True, True, True, True, True]
    assert spec["CostLawSpeech"] == "log"
    assert spec["CostLawNoSpeech"] == "log"


def test_the_cell_matrix_is_declared_once() -> None:
    """Issue #21: the (cell x direction) matrix has one declaration, here, and every Python
    consumer reads it from here. The names are the ones `nn/blstm.rs` parses. Since issue #27
    `init_weights` reads NO cell vocabulary at all: it draws per block kind off the layout."""
    from typing import get_args

    from speech.ledger.tables import CELL_NAME

    assert config_bridge.CELL_TYPES == ("lstm", "slstm", "mamba", "cfc", "transformer")
    assert config_bridge.DIRECTIONS == ("bidirectional", "forward")
    assert get_args(config_bridge.CellType) == config_bridge.CELL_TYPES
    assert get_args(config_bridge.Direction) == config_bridge.DIRECTIONS
    # the LID cell tables iterate CELL_TYPES and label each row through CELL_NAME
    assert set(CELL_NAME) == set(config_bridge.CELL_TYPES)
