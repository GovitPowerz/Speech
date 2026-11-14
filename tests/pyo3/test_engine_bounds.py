"""Issue #60: the `speech_rs.Engine` binding refuses an out-of-range config position and a
wrong-length pack list as Python exceptions, never a `pyo3_runtime.PanicException`.

  * `weights(pos)`, `set_weights(pos, nets)`, `weights_derivatives(pos)` with `pos ==
    len(configs)`: `IndexError` naming the position and the config count.
  * `set_weights(pos, nets)` with `len(nets)` different from config `pos`'s net count (1 for
    a single-net config, 2 for the Twin): `RuntimeError` naming both counts, raised before any
    pack is applied (the exact engine's packs are byte-equal before and after).
  * The exact and fast trees word the same misuse identically.

Issue #153: a pack crosses as a contiguous 1-D float64 array; a list of floats or another
dtype is a `TypeError` raised on the `nets` argument, a strided view a `RuntimeError`, and the
engine's pack is unchanged after either.

The Twin rows run on the phase-9 Mode-7 sLSTM LID fixture, the single-net rows on the phase-9
forward sLSTM SAD fixture, both seeded into a tmp dir and built through `Engine.from_map`
with the backprop flags off (the fast bag refuses a training-shaped map).
"""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from speech.weight_bridge import read_weight_vector

speech_rs = pytest.importorskip("speech_rs")

from tests.pyo3.test_phase9_seam import seed_sad, seed_twin_mode7  # noqa: E402

TWIN = "twin_mode7_lid_slstm"
SINGLE = "slstm_forward"


def _engine(tmp_path: Path, fixture: str, path: str) -> Any:
    cfg = dict(speech_rs.load_config(str(tmp_path / f"{fixture}.config")))
    cfg["Inference_Path"] = path
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    cfg["BLSTM_BackPropagationActivated"] = "false"
    if fixture == TWIN:
        cfg["BLSTM_LID_BackPropagationActivated"] = "false"
    with chdir(tmp_path):
        return speech_rs.Engine.from_map([cfg], "-m")


def _packs(tmp_path: Path, fixture: str) -> list[np.ndarray]:
    if fixture == TWIN:
        return [read_weight_vector(tmp_path / "tiny_sad_seed.bin"), read_weight_vector(tmp_path / f"{TWIN}_seed.bin")]
    return [read_weight_vector(tmp_path / f"{SINGLE}_seed.bin")]


def _seed(tmp_path: Path, fixture: str) -> None:
    (seed_twin_mode7 if fixture == TWIN else seed_sad)(tmp_path, fixture)


@pytest.mark.parametrize("path", ["exact", "fast"])
@pytest.mark.parametrize("method", ["weights", "set_weights", "weights_derivatives"])
def test_out_of_range_pos_raises_index_error(tmp_path: Path, method: str, path: str) -> None:
    _seed(tmp_path, TWIN)
    eng = _engine(tmp_path, TWIN, path)
    args = (1, _packs(tmp_path, TWIN)) if method == "set_weights" else (1,)
    with pytest.raises(IndexError, match=r"config position 1 is out of range: this engine holds 1 config\(s\)"):
        getattr(eng, method)(*args)


# (fixture, the wrong list). The packs are the config's own HALVED, so a list applied head-first
# (the pre-fix long-list behaviour) would move `weights(0)` and fail the before/after pin.
_WRONG_COUNT = {
    "twin_empty": (TWIN, lambda p: []),
    "twin_one": (TWIN, lambda p: [p[0] * 0.5]),
    "twin_three": (TWIN, lambda p: [p[0] * 0.5, p[1] * 0.5, p[1]]),
    "single_empty": (SINGLE, lambda p: []),
    "single_two": (SINGLE, lambda p: [p[0] * 0.5, p[0]]),
}


@pytest.mark.parametrize("leg", sorted(_WRONG_COUNT))
def test_wrong_pack_count_raises_before_applying_on_both_trees(tmp_path: Path, leg: str) -> None:
    fixture, wrong = _WRONG_COUNT[leg]
    _seed(tmp_path, fixture)
    nets = wrong(_packs(tmp_path, fixture))
    expected = 2 if fixture == TWIN else 1
    messages = []
    for path in ("exact", "fast"):
        eng = _engine(tmp_path, fixture, path)
        before = [w.copy() for w in eng.weights(0)]
        with pytest.raises(RuntimeError) as refused:
            eng.set_weights(0, nets)
        msg = str(refused.value)
        assert f"given is {len(nets)} but config 0 holds {expected} network(s)" in msg, msg
        after = eng.weights(0)
        assert len(after) == len(before)
        for a, b in zip(after, before, strict=True):
            assert np.array_equal(a, b), "a refused list leaves the engine's packs unchanged"
        messages.append(msg)
    assert messages[0] == messages[1]


def test_right_pack_count_still_sets(tmp_path: Path) -> None:
    _seed(tmp_path, TWIN)
    eng = _engine(tmp_path, TWIN, "exact")
    sad, lid = _packs(tmp_path, TWIN)
    eng.set_weights(0, [sad * 0.5, lid])
    assert np.array_equal(eng.weights(0)[0], sad * 0.5)


# Issue #153: a pack crosses the seam as a contiguous float64 array, copied once. Anything the
# binding would have to convert element by element is refused, and the engine's pack is unmoved.
_REFUSED_PACK = {
    "float_list": (lambda p: p.tolist(), TypeError, "argument 'nets'"),
    "float32": (lambda p: p.astype(np.float32), TypeError, "argument 'nets'"),
    "strided": (lambda p: np.repeat(p, 2)[::2], RuntimeError, "contiguous 1-D float64"),
}


@pytest.mark.parametrize("leg", sorted(_REFUSED_PACK))
def test_set_weights_takes_contiguous_float64_arrays_only(tmp_path: Path, leg: str) -> None:
    convert, error, match = _REFUSED_PACK[leg]
    _seed(tmp_path, SINGLE)
    eng = _engine(tmp_path, SINGLE, "exact")
    (pack,) = _packs(tmp_path, SINGLE)
    before = eng.weights(0)[0].copy()
    with pytest.raises(error, match=match):
        eng.set_weights(0, [convert(pack * 0.5)])
    assert np.array_equal(eng.weights(0)[0], before), "a refused pack leaves the engine's pack unchanged"
