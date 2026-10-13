"""Issue #27: `speech_rs.weight_layout(config, prefix)` -- the block list of a pack, from Rust.

The record is what `init_weights.py` draws per block kind from, so this file pins its SHAPE at
the seam: the fields, the stack order, contiguous offsets tiling exactly the committed packs'
lengths (every Phase 9 fixture through its manifest, both Phase 4d tuples), the peephole
bundle's 12 column names on the legacy LSTM alone, the sub-sampled output fan, the MLP-mode
tail, and a `ValueError` (never a default) on a config the engine refuses (ADR-0006).
"""

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from speech.config_bridge import parse_legacy_config

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE9 = REPO_ROOT / "tests" / "reference_data" / "phase9"
PHASE4D = REPO_ROOT / "tests" / "reference_data" / "phase4d"
MANIFEST = cast(dict[str, Any], json.loads((PHASE9 / "manifest.json").read_text()))
FIXTURES = cast(dict[str, dict[str, Any]], MANIFEST["fixtures"])

FIELDS = {"stack", "layer", "name", "kind", "shape", "fan_in", "fan_out", "offset", "len"}
KINDS = {"weight", "bias", "gain", "peephole", "stat"}


def _layout(path: Path, prefix: str) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], speech_rs.weight_layout(parse_legacy_config(path.read_text()), prefix))


def _assert_well_formed(layout: list[dict[str, Any]], pack_length: int) -> None:
    offset = 0
    for b in layout:
        assert set(b) >= FIELDS, b
        assert set(b) - FIELDS <= {"columns"}, b
        assert b["kind"] in KINDS, b
        assert b["offset"] == offset, b
        assert b["len"] > 0 and b["len"] == math.prod(b["shape"]), b
        assert len(b["shape"]) in (1, 2), b
        assert ("columns" in b) == (b["kind"] == "peephole"), b
        offset += b["len"]
    assert offset == pack_length
    stacks = [b["stack"] for b in layout]
    runs = [s for i, s in enumerate(stacks) if i == 0 or stacks[i - 1] != s]
    assert runs in (["forward", "backward", "output", "tail"], ["forward", "output", "tail"], ["output", "tail"]), runs
    assert [b["name"] for b in layout if b["stack"] == "tail"] == ["mean", "std"]
    assert all(b["kind"] == "stat" for b in layout if b["stack"] == "tail")


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_phase9_fixture_layout_tiles_its_committed_pack(name: str) -> None:
    info = FIXTURES[name]
    pack_length = info["lid_pack_length"] if info["nets"] == 2 else info["pack_length"]
    layout = _layout(PHASE9 / info["config"], info["prefix"])
    _assert_well_formed(layout, pack_length)
    cell = info.get("lid_cell_type", info["cell_type"])
    direction = info.get("lid_direction", info["direction"])
    assert ("backward" in {b["stack"] for b in layout}) == (direction == "bidirectional")
    assert any(b["kind"] == "peephole" for b in layout) == (cell == "lstm")


@pytest.mark.parametrize(("config", "pack_length"), [("tupleA_1_worker_1.config", 33671), ("tupleB_1_worker_1.config", 42911)])
def test_the_real_2015_shapes_tile_their_packs(config: str, pack_length: int) -> None:
    layout = _layout(PHASE4D / config, "BLSTM")
    _assert_well_formed(layout, pack_length)
    peeps = [b for b in layout if b["kind"] == "peephole"]
    assert len(peeps) == 4  # two layers x two stacks
    for p in peeps:
        assert p["name"] == "peep"
        assert p["shape"] == [24, 12]
        assert len(p["columns"]) == 12 and len(set(p["columns"])) == 12
        assert p["columns"][:3] == ["i.cell_prev", "f.cell_prev", "o.cell"]
    layer0 = [b for b in layout if b["stack"] == "forward" and b["layer"] == 0]
    assert [b["name"] for b in layer0] == ["w.i", "w.f", "w.o", "w.c", "r.i", "r.f", "r.o", "r.c", "peep", "b.i", "b.f", "b.o", "b.c"]
    # The gate map's combined fan: in + out, with layer 0's fan-in sub-sampled (x4).
    fin = {"tupleA_1_worker_1.config": 23, "tupleB_1_worker_1.config": 35}[config] * 4
    assert all((b["fan_in"], b["fan_out"]) == (fin + 24, 24) for b in layer0)
    assert layer0[0]["shape"] == [24, fin]


def test_output_sub_sampling_widens_the_output_fan() -> None:
    cfg = parse_legacy_config((PHASE4D / "tupleA_1_worker_1.config").read_text())
    base = speech_rs.weight_layout(cfg, "BLSTM")
    cfg["BLSTM_OutputSubSampling"] = "2,1"
    wide = speech_rs.weight_layout(cfg, "BLSTM")
    out0 = [b for b in wide if b["stack"] == "output" and b["layer"] == 0]
    assert [(b["name"], b["shape"], b["fan_in"], b["fan_out"]) for b in out0] == [("W", [12, 96], 96, 12), ("b", [12], 96, 12)]
    assert sum(b["len"] for b in wide) - sum(b["len"] for b in base) == 12 * 48


def test_mlp_mode_keeps_the_tail_at_the_output_input_width() -> None:
    cfg = parse_legacy_config((PHASE4D / "tupleA_1_worker_1.config").read_text())
    cfg["BLSTM_LSTMNeuronNb"] = "0,24"
    cfg["BLSTM_LSTMSubSampling"] = "1"
    layout = speech_rs.weight_layout(cfg, "BLSTM")
    assert [b["stack"] for b in layout] == ["output", "output", "output", "output", "tail", "tail"]
    assert [b["len"] for b in layout if b["stack"] == "tail"] == [48, 48]
    _assert_well_formed(layout, 12 * 49 + 1 * 13 + 96)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("BLSTM_Cell_Type", "gru", "unknown cell type"),
        ("Mamba_D_State", "0", "Mamba_D_State"),
        ("BLSTM_OutputNeuronNb", "47,12,1", "twice the number of LSTM cells"),
    ],
)
def test_a_config_the_engine_refuses_raises_value_error(key: str, value: str, match: str) -> None:
    cfg = parse_legacy_config((PHASE4D / "tupleA_1_worker_1.config").read_text())
    cfg[key] = value
    with pytest.raises(ValueError, match=match):
        speech_rs.weight_layout(cfg, "BLSTM")


def test_the_twin_prefix_selects_the_lid_net() -> None:
    info = FIXTURES["twin_mode7_lid_slstm"]
    cfg = parse_legacy_config((PHASE9 / info["config"]).read_text())
    sad = speech_rs.weight_layout(cfg, "BLSTM")
    lid = speech_rs.weight_layout(cfg, "BLSTM_LID")
    assert sum(b["len"] for b in sad) == info["sad_pack_length"]
    assert sum(b["len"] for b in lid) == info["lid_pack_length"]
    assert any(b["kind"] == "peephole" for b in sad) and not any(b["kind"] == "peephole" for b in lid)
