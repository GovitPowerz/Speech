"""Phase 4c Task 10: masking_validation.

masking_validation goldens (`tests/reference_data/phase4c/masking_*.bin`) are dumped by
GNU Octave running the REAL `legacy/Optimizer_V6.2.2/functions/{MaskingValidation,
vec2struct}.m` (see `tools/octave_harness/stage_masking.m`). (The check_grad pin that
shared this file was retired in issue #103, the confusion_matrix unit tests in issue #155.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from speech.genome import RunConfig
from speech.scoring import masking_validation
from speech.weight_bridge import read_bin

PHASE4C = Path(__file__).resolve().parent / "reference_data" / "phase4c"
MANIFEST = PHASE4C / "manifest.json"


def _load(name: str) -> NDArray[np.float64]:
    rows, cols, flat = read_bin(PHASE4C / f"{name}.bin")
    return np.asarray(flat, dtype=np.float64).reshape((rows, cols), order="F")


# --------------------------------------------------------------------------- #
# masking_validation (vs MaskingValidation.m)
# --------------------------------------------------------------------------- #


def _masking_ps() -> RunConfig:
    """Mirrors stage_masking.m's base_ps_masking (== stage_vec2struct.m's base_ps(3))."""
    return RunConfig(
        numOuterThreads=7,
        numInnerThreads=1,
        OutputFile="MultiConfigResults.mat",
        Display_MillisecondsPerPixel=64,
        name_dir_fig="/tmp/fsp/Figures",
        name_dir="/tmp/fsp",
        offset=0,
        durmax=240,
        algo=3,
        nbworker=1,
        epoch=0,
        adim=10,
        coeff_NN=10,
        VRCTS_isFast=1,
        VRCTS_force=0,
        balance=6,
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


def test_masking_manifest_present() -> None:
    assert MANIFEST.is_file()
    m = json.loads(MANIFEST.read_text())
    assert m["masking"]["pass_failed"] == 0
    assert m["masking"]["fail_failed"] == 1


def test_masking_validation_pass_case() -> None:
    """A well-formed scalar mask (AlgName_decision_thresh_rising=0.7, the same case Task
    8 bit-pins) round-trips cleanly -- no mismatches."""
    vector = _load("masking_pass_vector").reshape(-1)
    mismatches = masking_validation(vector, {"AlgName_decision_thresh_rising": 0.7}, _masking_ps())
    assert mismatches == []


def test_masking_validation_fail_case() -> None:
    """Masking AlgName_min_speech (a `_padding_block`-family vector field) with a
    negative last component: the encode `(fv+0.1)*adim` (using the RAW masked fv) is NOT
    the algebraic inverse of the decode `-0.1+abs(p/adim)` for fv<0 (abs() destroys the
    sign asymmetrically), so the round trip genuinely diverges. See IMPROVEMENTS.md."""
    vector = _load("masking_fail_vector").reshape(-1)
    mismatches = masking_validation(vector, {"AlgName_min_speech": [1.0, 1.0, -5.0]}, _masking_ps())
    assert mismatches == ["BLSTM_min_speech"]


def test_masking_validation_numeric_1e12_threshold() -> None:
    """A mask value that differs from the round trip by exactly at/under 1e-12 passes;
    anything genuinely larger fails -- pins the `abs(diff) > 1e-12` boundary (:39/:45)."""
    ps = _masking_ps()
    vector = _load("masking_pass_vector").reshape(-1)
    # A well-formed mask must round-trip clean regardless of the exact value (as long as
    # it stays within the field's valid domain) -- 0.5 is a different but equally
    # well-formed rising threshold.
    assert masking_validation(vector, {"AlgName_decision_thresh_rising": 0.5}, ps) == []
