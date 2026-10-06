"""Strict `true`/`false` config flags: a missing key takes the default, `true`/`false` read as
written, anything else raises naming the key -- the rule the engine applies (issue #32)."""

import re
from pathlib import Path

import numpy as np
import pytest
from speech import config_bridge
from speech.drivers.state import ps_from_config
from speech.genome import genome_length, vec2struct

TWIN_CONFIG = Path("tests/reference_data/phase4b/twin_train.config")
PEEPHOLE_KINDS = ("IsCellsPeepholesActive", "IsGatesPeepholesActive", "IsGatesRecurrentPeepholesActive")


def _twin() -> dict[str, str]:
    return config_bridge.parse_legacy_config(TWIN_CONFIG.read_text())


def _with_peepholes(prefix: str, value: str | None) -> dict[str, str]:
    cfg = _twin()
    for kind in PEEPHOLE_KINDS:
        for side in ("Forward", "Backward"):
            key = f"{prefix}_{side}_{kind}"
            if value is None:
                cfg.pop(key, None)
            else:
                cfg[key] = value
    return cfg


@pytest.mark.parametrize(("text", "default", "expected"), [("true", False, True), ("false", True, False), (" true ", False, True), ("false\t", True, False)])
def test_read_flag_reads_true_and_false_as_written(text: str, default: bool, expected: bool) -> None:
    assert config_bridge.read_flag({"k": text}, "k", default) is expected


@pytest.mark.parametrize("default", [True, False])
def test_read_flag_missing_key_takes_the_default(default: bool) -> None:
    assert config_bridge.read_flag({}, "k", default) is default


@pytest.mark.parametrize("text", ["True", "FALSE", "1", "0", "yes", "", "ture", "true false"])
def test_read_flag_malformed_raises_naming_key_and_value(text: str) -> None:
    with pytest.raises(ValueError, match=r"'k'.*" + re.escape(repr(text))):
        config_bridge.read_flag({"k": text}, "k", True)


@pytest.mark.parametrize(("value", "expected"), [("false", 0), ("true", 1), (None, 1)])
def test_ps_from_config_peepholes(value: str | None, expected: int) -> None:
    ps = ps_from_config(_with_peepholes("BLSTM", value))
    assert [getattr(ps, f) for f in PEEPHOLE_KINDS] == [expected] * 3


@pytest.mark.parametrize(("value", "expected"), [("false", 0), ("true", 1), (None, 1)])
def test_ps_from_config_lid_peepholes(value: str | None, expected: int) -> None:
    lid = ps_from_config(_with_peepholes("BLSTM_LID", value)).lid
    assert lid is not None
    assert [getattr(lid, f) for f in PEEPHOLE_KINDS] == [expected] * 3


@pytest.mark.parametrize("side", ["Forward", "Backward"])
@pytest.mark.parametrize("prefix", ["BLSTM", "BLSTM_LID"])
def test_ps_from_config_malformed_peephole_raises(prefix: str, side: str) -> None:
    cfg = _twin()
    cfg[f"{prefix}_{side}_IsGatesPeepholesActive"] = "True"
    with pytest.raises(ValueError, match=f"'{prefix}_{side}_IsGatesPeepholesActive'"):
        ps_from_config(cfg)


@pytest.mark.parametrize(("forward", "backward"), [("true", "false"), ("false", None)])
@pytest.mark.parametrize("prefix", ["BLSTM", "BLSTM_LID"])
def test_ps_from_config_asymmetric_peephole_raises(prefix: str, forward: str, backward: str | None) -> None:
    cfg = _twin()
    cfg[f"{prefix}_Forward_IsCellsPeepholesActive"] = forward
    if backward is None:
        cfg.pop(f"{prefix}_Backward_IsCellsPeepholesActive")
    else:
        cfg[f"{prefix}_Backward_IsCellsPeepholesActive"] = backward
    with pytest.raises(ValueError, match=f"'{prefix}_Forward_IsCellsPeepholesActive' and '{prefix}_Backward_IsCellsPeepholesActive' differ"):
        ps_from_config(cfg)


def test_ps_from_config_forward_net_ignores_backward_peephole() -> None:
    cfg = _with_peepholes("BLSTM", "false")
    cfg["BLSTM_Direction"] = "forward"
    cfg.pop("BLSTM_Backward_IsCellsPeepholesActive")
    assert ps_from_config(cfg).IsCellsPeepholesActive == 0


@pytest.mark.parametrize("prefix", ["BLSTM", "BLSTM_LID"])
def test_vec2struct_carries_a_disabled_peephole_to_both_directions(prefix: str) -> None:
    ps = ps_from_config(_with_peepholes(prefix, "false"))
    decoded, _, _ = vec2struct(np.zeros(genome_length(ps) - 1), None, ps, 0)
    for kind in PEEPHOLE_KINDS:
        for side in ("Forward", "Backward"):
            assert decoded[f"{prefix}_{side}_{kind}"] == "false"


@pytest.mark.parametrize(("value", "expected"), [("false", 0), ("true", 1), (None, 0)])
def test_ps_from_config_exclude_nontrans(value: str | None, expected: int) -> None:
    cfg = _twin()
    cfg.pop("exclude_nontrans", None)
    if value is not None:
        cfg["exclude_nontrans"] = value
    assert ps_from_config(cfg).exclude_nontrans == expected


def test_ps_from_config_malformed_exclude_nontrans_raises() -> None:
    cfg = _twin()
    cfg["exclude_nontrans"] = "1"
    with pytest.raises(ValueError, match="'exclude_nontrans'"):
        ps_from_config(cfg)


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("true", True), (None, True)])
@pytest.mark.parametrize("prefix", ["BLSTM", "BLSTM_LID"])
def test_nnet_spec_peepholes(prefix: str, value: str | None, expected: bool) -> None:
    assert config_bridge.nnet_spec(_with_peepholes(prefix, value), prefix)["peepholes"] == [expected] * 6


def test_nnet_spec_malformed_peephole_raises() -> None:
    cfg = _twin()
    cfg["BLSTM_Backward_IsCellsPeepholesActive"] = "1"
    with pytest.raises(ValueError, match="'BLSTM_Backward_IsCellsPeepholesActive'"):
        config_bridge.nnet_spec(cfg, "BLSTM")
