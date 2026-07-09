"""Phase 4c Task 8: the vec2struct genome<->config bijection, bit-pinned vs Octave.

The goldens (`tests/reference_data/phase4c/genome_*`) are dumped by GNU Octave running the
REAL `legacy/Optimizer_V6.2.2/functions/vec2struct.m` (+ `printConfig.m`) over six algo/mask/PS
cases (see `scripts/extract_phase4c_genome_fixtures.py`). Each case pins, against the Python
port (`speech.genome`):
  * count_param (STRICT -- one miscount shifts every downstream genome index; risk R3),
  * the printConfig configStruct (every emitted key/value string-exact, in insertion order),
  * out_param (bit-exact -- vec2struct is pure arithmetic, no libm), and
  * the mask fix + inverse write-back / Force_Symetry-Force_Identical_Rows block ties.

RunConfig / mask per case mirror the Octave stage's PS (`tools/octave_harness/stage_vec2struct.m`);
param is read byte-for-byte from the committed `.bin` (never recomputed), so the comparison is
platform-independent.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from speech.config_bridge import parse_legacy_config
from speech.genome import LidNetSpec, RunConfig, genome_length, vec2struct

PHASE4C = Path(__file__).resolve().parent / "reference_data" / "phase4c"
GENOME_MANIFEST = PHASE4C / "genome_manifest.json"
CASES = ["algo0", "tdc", "calib", "spectral", "twin", "masked"]


def _read_bin(path: Path) -> NDArray[np.float64]:
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    n = rows * cols
    return np.asarray(struct.unpack(f"<{n}d", raw[16 : 16 + 8 * n]), dtype=np.float64).reshape(-1)


def _manifest() -> dict[str, object]:
    return cast(dict[str, object], json.loads(GENOME_MANIFEST.read_text()))


def _base(algo: int, balance: int = 6) -> RunConfig:
    """Mirror stage_vec2struct.m's base_ps (tiny nets)."""
    return RunConfig(
        numOuterThreads=7,
        numInnerThreads=1,
        OutputFile="MultiConfigResults.mat",
        Display_MillisecondsPerPixel=64,
        name_dir_fig="/tmp/fsp/Figures",
        name_dir="/tmp/fsp",
        offset=0,
        durmax=240,
        algo=algo,
        nbworker=1,
        epoch=0,
        adim=10,
        coeff_NN=10,
        VRCTS_isFast=1,
        VRCTS_force=0,
        balance=balance,
        exclude_nontrans=0,
        useVRCTSFeatures=0,
        nnmatfile="NNweights.mat",
        minSegmentLength=0,
        addNoise=0,
        mappingFile="/tmp/fsp/languagemapping.csv",
        listing="/tmp/fsp/fileslisting",
        BackPropagationActivated=1,
        NNType=0,
        LSTM_net_size=[3, 2],
        LSTMSubSampling=[2],
        Output_net_size=[4, 2, 1],
        OutputSubSampling=[1, 1],
        LSTM_MaxSat=-10,
        BackPropWER=-1.0,
        BalanceBackProp=0.0,
        InputNormalizationType=-1,
        IsCellsPeepholesActive=1,
        IsGatesPeepholesActive=1,
        IsGatesRecurrentPeepholesActive=1,
    )


def _lid() -> LidNetSpec:
    return LidNetSpec(
        NNType=0,
        BackPropagationActivated=0,
        LSTM_net_size=[2, 3],
        LSTMSubSampling=[1],
        Output_net_size=[6, 1],
        OutputSubSampling=[1],
        Mode=7,
        PostProcessMode=0,
        TargetEnforcementStep=0,
        BackPropWER=-1.0,
        classes_ponderations=[],
        InputNormalizationType=-1,
        IsCellsPeepholesActive=1,
        IsGatesPeepholesActive=1,
        IsGatesRecurrentPeepholesActive=1,
        LSTM_MaxSat=-10,
        nnmatfile="LIDNNweights.mat",
    )


def _case(name: str) -> tuple[RunConfig, dict[str, object] | None]:
    if name == "algo0":
        return _base(0), None
    if name == "tdc":
        return _base(1), None
    if name == "calib":
        return _base(2), None
    if name == "spectral":
        return _base(3), None
    if name == "twin":
        return _base(6).model_copy(update={"lid": _lid(), "mask_has_nnet": True, "mask_has_nnetLID": True}), None
    if name == "masked":
        mask: dict[str, object] = {"Force_Symetry": 1, "Force_Identical_Rows": 1, "AlgName_decision_thresh_rising": 0.7}
        return _base(3), mask
    raise ValueError(name)


def _run(name: str) -> tuple[dict[str, str], NDArray[np.float64], int, NDArray[np.float64]]:
    ps, mask = _case(name)
    param = _read_bin(PHASE4C / f"genome_{name}_param.bin")
    cfg, out, count = vec2struct(param, mask, ps, 0)
    return cfg, out, count, param


def test_genome_manifest_present() -> None:
    assert GENOME_MANIFEST.is_file(), "genome_manifest.json must be committed"
    m = _manifest()
    assert set(cast(dict[str, int], m["counts"])) == set(CASES)
    version = cast(str, m["octave_version"])
    assert version and version[0].isdigit()


@pytest.mark.parametrize("name", CASES)
def test_count_param_matches(name: str) -> None:
    """STRICT: count_param == Octave's, and genome_length (the zero-genome walk) agrees."""
    ps, mask = _case(name)
    _, _, count, param = _run(name)
    expected = cast(dict[str, int], _manifest()["counts"])[name]
    assert count == expected, f"{name}: count_param {count} != manifest {expected}"
    assert count == param.size + 1, f"{name}: count_param {count} != len(param)+1"
    assert genome_length(ps) == expected, f"{name}: genome_length != count_param"


@pytest.mark.parametrize("name", CASES)
def test_configstruct_fields_match(name: str) -> None:
    """Every emitted key/value string-exact vs the printConfig golden, in insertion order."""
    cfg, _, _, _ = _run(name)
    golden = parse_legacy_config((PHASE4C / f"genome_{name}.config").read_text())
    assert list(cfg.items()) == list(golden.items()), f"{name}: config dict mismatch"


@pytest.mark.parametrize("name", CASES)
def test_out_param_inverse_matches(name: str) -> None:
    """out_param bit-exact (pure arithmetic -> STRICT bits on every platform)."""
    _, out, _, _ = _run(name)
    golden = _read_bin(PHASE4C / f"genome_{name}_out_param.bin")
    assert out.shape == golden.shape
    assert out.view(np.uint64).tobytes() == golden.view(np.uint64).tobytes(), f"{name}: out_param bit mismatch"


def test_mask_fixes_field_and_writes_back() -> None:
    """The masked case pins AlgName_decision_thresh_rising = 0.7: the config carries the fixed
    value and out_param[0] carries its inverse encode (0.7 * adim = 7.0)."""
    cfg, out, _, _ = _run("masked")
    assert cfg["BLSTM_decision_thresh_rising"] == "7.000000000000000e-01"
    assert out[0] == 7.0


def test_force_symetry_and_identical_rows_tie_blocks() -> None:
    """Non-vacuity: Force_Symetry ties each backward LSTM block to its forward twin, and
    Force_Identical_Rows ties block jj>1 to block 0. For the tiny net LSTM=[3,2] sub=[2] the
    layer-0 InputGate blocks span (0-based) forward-block0 [47:60], forward-block1 [95:108],
    backward-block0 [143:156]; the ties re-encode out_param to the FORWARD block-0 input
    params (up to the coeff_NN round-trip)."""
    ps, mask = _case("masked")
    param = _read_bin(PHASE4C / "genome_masked_param.bin")
    _, out, _ = vec2struct(param, mask, ps, 0)
    _, out_plain, _ = vec2struct(param, None, ps, 0)
    fwd0, fwd1, bwd0 = slice(47, 60), slice(95, 108), slice(143, 156)
    # non-vacuous: the input params of the three blocks genuinely differ
    assert not np.allclose(param[bwd0], param[fwd0])
    assert not np.allclose(param[fwd1], param[fwd0])
    # Force_Symetry: backward block tied to forward block (and it actually changed out_param)
    assert np.allclose(out[bwd0], param[fwd0])
    assert not np.array_equal(out[bwd0], out_plain[bwd0])
    assert np.array_equal(out_plain[bwd0], param[bwd0])  # no write-back without the flags
    # Force_Identical_Rows: forward block jj=2 tied to block 0
    assert np.allclose(out[fwd1], param[fwd0])


def test_twin_lid_falling_negate_quirk() -> None:
    """The algo-6 LID head stores decision_thresh_falling = -decision_thresh_rising
    unconditionally (vec2struct.m:964), regardless of the param/mask/clamp branch above it."""
    cfg, _, _, _ = _run("twin")
    rising = float(cfg["BLSTM_LID_decision_thresh_rising"])
    falling = float(cfg["BLSTM_LID_decision_thresh_falling"])
    assert falling == -rising
    # BackPropOutputNetworkOnly toggles (PS.VP.mask nnet/nnetLID) both fired, Mode 7 present.
    assert cfg["BLSTM_BackPropOutputNetworkOnly"] == "true"
    assert cfg["BLSTM_LID_BackPropOutputNetworkOnly"] == "true"
    assert cfg["BLSTM_LID_Mode"] == "7"


def test_calib_law_decode_and_clamps() -> None:
    """The algo-2 calibration case pins the rem(round(5*|p|/adim),5) law decode + the [0,1]
    param/thresh clamps (crafted param -> sqrt/cubic + clamp-high/clamp-low)."""
    cfg, _, _, _ = _run("calib")
    assert cfg["LTSV_CostLawSpeech"] == "sqrt"
    assert cfg["LTSV_CostLawNoSpeech"] == "cubic"
    assert cfg["LTSV_CostLawParamSpeech"] == "1"  # clamped up from 2.0
    assert cfg["LTSV_CostLawParamNoSpeech"] == "0"  # clamped down from -0.5
    assert cfg["LTSV_CostLawThreshNoSpeech"] == "1"  # clamped up from 2.5
