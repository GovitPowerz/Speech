"""Phase 4c Task 8 (+ Phase 4d Task 12 closure): the vec2struct genome<->config bijection,
bit-pinned vs Octave.

The goldens (`tests/reference_data/phase4c/genome_*`) are dumped by GNU Octave running the
REAL `legacy/Optimizer_V6.2.2/functions/vec2struct.m` (+ `printConfig.m`) over seven algo/mask/PS
cases (see `scripts/extract_phase4c_genome_fixtures.py`). Each case pins, against the Python
port (`speech.genome`):
  * count_param (STRICT -- one miscount shifts every downstream genome index; risk R3),
  * the printConfig configStruct (every emitted key/value string-exact, in insertion order),
  * out_param (bit-exact -- vec2struct is pure arithmetic, no libm), and
  * the mask fix + inverse write-back / Force_Symetry-Force_Identical_Rows block ties.

Task 12 closes the two accepted-debt items from the 4c T8 final review (ledger Minor
roll-up): the "vecmask" case exercises the MASK-VECTOR arms (per-block LSTM weight masks
incl. the narrow CellWeight layout, an output-layer neuron mask, and the NormalizeInputMean/
Std masks incl. the Std abs() asymmetry) that "masked" left uncovered -- FS/FIR tie the
blocks together but never exercise a raw `isfield(maskStruct, fieldName)` vector arm
standalone. `fmt_scalar_boundaries.config` (a direct real-printConfig.m probe, not a
vec2struct case) makes `_fmt_scalar`'s %d/%15.15e boundary formatting an explicit unit
table instead of an incidental byproduct of the other cases' config-string comparisons.

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
from speech.genome import LidNetSpec, RunConfig, _fmt_scalar, genome_length, vec2struct

PHASE4C = Path(__file__).resolve().parent / "reference_data" / "phase4c"
GENOME_MANIFEST = PHASE4C / "genome_manifest.json"
FMT_SCALAR_BOUNDARIES_CONFIG = PHASE4C / "fmt_scalar_boundaries.config"
CASES = ["algo0", "tdc", "calib", "spectral", "twin", "masked", "vecmask"]

# fmt_scalar_boundaries.config field order -- MUST mirror
# tools/octave_harness/stage_vec2struct.m::probe_fmt_scalar's `names`/`vals` exactly.
FMT_SCALAR_BOUNDARIES: list[tuple[str, float]] = [
    ("b_zero", 0.0),
    ("b_neg_zero", -0.0),
    ("b_one", 1.0),
    ("b_neg_one", -1.0),
    ("b_int_1e5", 100000.0),
    ("b_neg_int_1e5", -100000.0),
    ("b_frac_above_1em5", 0.000015),
    ("b_neg_frac_above_1em5", -0.000015),
    ("b_frac_below_1em5", 0.0000099),
    ("b_frac_at_1em5", 0.00001),
    ("b_frac_above_1e5", 123456.789),
    ("b_frac_below_1e5", 99999.99999),
    ("b_long_precision", 0.123456789012345),
    ("b_long_precision_neg", -3.14159265358979),
    ("b_near_integer_boundary", 2.9999999999999996),
    ("b_big_int", 1234567890123.0),
    ("b_five", 5.0),
    ("b_neg_five", -5.0),
]


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
    if name == "vecmask":
        return _base(3), dict(_VECMASK_MASK)
    raise ValueError(name)


# Task 12: the mask-VECTOR arms, isolated from Force_Symetry/Force_Identical_Rows (neither
# flag is set in the "vecmask" case) so each `isfield(maskStruct, fieldName)` vector branch
# fires standalone. Numeric literals MUST mirror
# tools/octave_harness/stage_vec2struct.m::build_case's 'vecmask' case exactly.
_VECMASK_MASK: dict[str, object] = {
    "AlgName_Forward_Layer_0_LSTMBlock_0_InputGateWeights": [-2.5, -1.25, -0.625, 0, 0.625, 1.25, 2.5, 3.75, -3.75, 5, -5, 0.3125, -0.3125],
    "AlgName_Forward_Layer_0_LSTMBlock_1_CellWeight": [1.5, -1.5, 2.25, -2.25, 0, 4.5, -4.5, 6.75, -6.75],
    "AlgName_Backward_Layer_0_LSTMBlock_0_OutputGateWeights": [-4.5, 4.5, -0.75, 0.75, 8.5, -8.5, 1.125, -1.125, 2.75, -2.75, 0, 9.25, -9.25],
    "AlgName_Output_Layer_1_Neuron_0_Weights": [-1.0, 2.5, -3.25],
    "AlgName_NormalizeInputMean": [0.5, -0.25, 0.125],
    "AlgName_NormalizeInputStd": [-2.0, 3.0, -0.5],
}


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


# ---- Task 12: mask-VECTOR arms (4c T8 accepted debt, closed) ----

# (arm label, 0-based out_param slice, expected write-back formula over the crafted mask
# value from _VECMASK_MASK) -- coeff=10/adim=10 for every base_ps(3) case, so the weight-
# block/output-neuron inverse encode is `field/2 + 5`; NormalizeInputMean is `(field+1)/2`
# (no coeff/adim scaling); NormalizeInputStd is `abs(field) - 1e-3` (the abs() asymmetry --
# the crafted mask is deliberately signed to exercise it).
_VECMASK_ARMS: list[tuple[str, slice, list[float]]] = [
    (
        "Forward_Layer_0_LSTMBlock_0_InputGateWeights",
        slice(47, 60),
        [x / 2 + 5 for x in cast(list[float], _VECMASK_MASK["AlgName_Forward_Layer_0_LSTMBlock_0_InputGateWeights"])],
    ),
    (
        "Forward_Layer_0_LSTMBlock_1_CellWeight",
        slice(134, 143),
        [x / 2 + 5 for x in cast(list[float], _VECMASK_MASK["AlgName_Forward_Layer_0_LSTMBlock_1_CellWeight"])],
    ),
    (
        "Backward_Layer_0_LSTMBlock_0_OutputGateWeights",
        slice(169, 182),
        [x / 2 + 5 for x in cast(list[float], _VECMASK_MASK["AlgName_Backward_Layer_0_LSTMBlock_0_OutputGateWeights"])],
    ),
    ("Output_Layer_1_Neuron_0_Weights", slice(249, 252), [x / 2 + 5 for x in cast(list[float], _VECMASK_MASK["AlgName_Output_Layer_1_Neuron_0_Weights"])]),
    ("NormalizeInputMean", slice(252, 255), [(x + 1) / 2 for x in cast(list[float], _VECMASK_MASK["AlgName_NormalizeInputMean"])]),
    ("NormalizeInputStd", slice(255, 258), [abs(x) - 1e-3 for x in cast(list[float], _VECMASK_MASK["AlgName_NormalizeInputStd"])]),
]


@pytest.mark.parametrize("arm_name,sl,expected", _VECMASK_ARMS, ids=[a[0] for a in _VECMASK_ARMS])
def test_vecmask_arm_writes_expected_slice(arm_name: str, sl: slice, expected: list[float]) -> None:
    """Each mask-vector arm's out_param slice matches its inverse-encode formula exactly
    (cross-derived independently of genome.py -- see task-12-report.md), pinning the six
    arms individually rather than relying only on the whole-vector bit-exact golden check."""
    _, out, _, _ = _run("vecmask")
    assert np.allclose(out[sl], expected), f"{arm_name}: out_param{sl} = {out[sl].tolist()} != {expected}"


def test_vecmask_arms_are_non_vacuous_vs_unmasked() -> None:
    """Non-vacuity: an unmasked run at the same six slices produces DIFFERENT out_param
    values (the param-derived decode, not the mask's crafted constants) -- so each arm's
    golden match in test_vecmask_arm_writes_expected_slice is because the mask fired, not
    because the unmasked decode happened to coincide."""
    ps, mask = _case("vecmask")
    param = _read_bin(PHASE4C / "genome_vecmask_param.bin")
    _, out_masked, _ = vec2struct(param, mask, ps, 0)
    _, out_plain, _ = vec2struct(param, None, ps, 0)
    for arm_name, sl, _expected in _VECMASK_ARMS:
        assert not np.allclose(out_masked[sl], out_plain[sl]), f"{arm_name}: masked/unmasked out_param coincide -- mask arm is vacuous"
        # unmasked: none of the six arms' write-back (`_setv`) ever fires without a mask hit,
        # so the unmasked out_param is untouched at these slices (== the raw input param).
        assert np.array_equal(param[sl], out_plain[sl]), f"{arm_name}: unmasked out_param unexpectedly diverges from raw param"


def test_normalize_std_mask_applies_abs_asymmetry() -> None:
    """The NormalizeInputStd mask arm's crafted value is SIGNED ([-2.0, 3.0, -0.5]) but
    both the config field AND the out_param write-back store abs(mask) -- unlike
    NormalizeInputMean, which passes its (equally signed) mask through unmodified aside
    from the (field+1)/2 encode. Isolates the negative-mask asymmetry vec2struct.m:933-938
    applies only to Std, mirroring the asymmetry already known from 4c scoring."""
    _, out, _, _ = _run("vecmask")
    mean_mask = cast(list[float], _VECMASK_MASK["AlgName_NormalizeInputMean"])
    std_mask = cast(list[float], _VECMASK_MASK["AlgName_NormalizeInputStd"])
    assert any(v < 0 for v in std_mask), "fixture must carry a negative Std mask value to exercise abs()"
    assert np.allclose(out[slice(252, 255)], [(x + 1) / 2 for x in mean_mask])  # Mean: no abs
    assert np.allclose(out[slice(255, 258)], [abs(x) - 1e-3 for x in std_mask])  # Std: abs() applied


def test_fmt_scalar_boundaries_manifest_present() -> None:
    assert FMT_SCALAR_BOUNDARIES_CONFIG.is_file(), "fmt_scalar_boundaries.config must be committed"
    m = _manifest()
    assert "fmt_scalar_boundaries" in m
    fsb = cast(dict[str, object], m["fmt_scalar_boundaries"])
    assert fsb["file"] == "fmt_scalar_boundaries.config"
    assert cast(list[str], fsb["names"]) == [name for name, _ in FMT_SCALAR_BOUNDARIES]


@pytest.mark.parametrize("name,value", FMT_SCALAR_BOUNDARIES)
def test_fmt_scalar_boundary_matches_octave(name: str, value: float) -> None:
    """_fmt_scalar's %d/%15.15e boundary formatting, pinned string-exact against a direct
    real-printConfig.m call (TIER 1) over integers (incl. -0 and a big exact integer that
    must not flip to e-notation), negatives, the 1e-5/1e+5 magnitude boundaries on both
    sides, a round(v)==v near-integer decode edge, and long-precision fractions."""
    cfg = parse_legacy_config(FMT_SCALAR_BOUNDARIES_CONFIG.read_text())
    assert _fmt_scalar(value) == cfg[name], f"{name}: _fmt_scalar({value!r}) = {_fmt_scalar(value)!r} != octave {cfg[name]!r}"
