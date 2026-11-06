"""Phase 4c Task 10: masking_validation, confusion_matrix.

masking_validation goldens (`tests/reference_data/phase4c/masking_*.bin`) are dumped by
GNU Octave running the REAL `legacy/Optimizer_V6.2.2/functions/{MaskingValidation,
vec2struct}.m` (see `tools/octave_harness/stage_masking.m`). confusion_matrix has no Octave
golden; its tests are direct unit tests hand-verified against `confusionThresh.m`'s logic.
(The check_grad pin that shared this file was retired in issue #103.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from speech.genome import RunConfig
from speech.scoring import confusion_matrix, masking_validation
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


# --------------------------------------------------------------------------- #
# confusion_matrix (vs confusionThresh.m, direct unit tests -- no Octave golden)
# --------------------------------------------------------------------------- #


def test_confusion_matrix_labels_and_shape() -> None:
    cm = confusion_matrix(np.array([0]), np.array([[-49.0, 10.0]]), thresh=10.0)
    assert cm.shape == (4, 4)
    assert cm[0, :3].tolist() == [0.0, 1.0, 2.0]
    assert cm[:3, 0].tolist() == [0.0, 1.0, 2.0]
    assert cm[3, 0] == 0.0 and cm[0, 3] == 0.0  # the totals row/col is unlabeled


def test_confusion_matrix_hit_when_target_clears_threshold() -> None:
    """Target in slot 1 (0-based; legacy posTarget=2) with score > thresh -> a HIT on
    the diagonal, not charged to the best non-target class."""
    # class1 is the target (wire 260 -> decoded 60)
    cm = confusion_matrix(np.array([1]), np.array([[10.0, 60.0]]), thresh=10.0)
    assert cm[2, 2] == 1.0  # pos_target = kk(1)+1 = 2, diagonal hit
    assert cm[2, 3] == 1.0 and cm[3, 2] == 1.0


def test_confusion_matrix_miss_charges_best_non_target() -> None:
    # target=class0 (wire 151 -> decoded -49), best-not-target=class1
    cm = confusion_matrix(np.array([0]), np.array([[-49.0, 10.0]]), thresh=10.0)
    # pos_target = 0+1 = 1, pos_best_not_target = 1+1 = 2 (miss: -49 > 100-10=90 is false)
    assert cm[1, 2] == 1.0
    assert cm[1, 3] == 1.0 and cm[3, 2] == 1.0
    assert cm[1, 1] == 0.0  # not a diagonal hit


def test_confusion_matrix_no_target_row_skipped() -> None:
    """FIXED (phase 5, F5): `posTarget`/`posBestNotTarget` are declared PER ROW, so a file
    with no target (legacy: no score > 150) no longer inherits the previous row's
    posTarget -- it is an out-of-set trial with no true class in the closed set, skipped
    entirely. Pre-fix (sticky) the 2nd row reused row0's posTarget=1/posBestNotTarget=2 and
    double-charged cm[1,2] to 2.0; post-fix only row0's miss counts, so cm[1,2]==1.0. The
    Octave oracle still describes the sticky behavior; the port diverges by design. Fixed
    identically to the Rust engine/confusion.rs::confusion_from_results."""
    scores = np.array(
        [
            [-49.0, 10.0],  # row0: target=class0 (wire 151 -> -49), best-not-target=class1 -> miss
            [20.0, 30.0],  # row1: no target -> SKIPPED (not sticky-charged)
        ]
    )
    cm = confusion_matrix(np.array([0, -1]), scores, thresh=10.0)
    assert cm[1, 2] == 1.0
    assert cm[1, 3] == 1.0
    assert cm[3, 2] == 1.0


def test_confusion_cross_language_identity_no_target_skip() -> None:
    """BOTH-LANGUAGE IDENTITY (F5): byte-identical mirror of the Rust
    `engine::confusion::tests::cross_language_identity_no_target_skip`. The two confusion
    ports are different legacy sources (confusionThresh.m's asymmetric threshold vs
    PrintConfusionMatrix's max-competitor win), so they only produce the same matrix when
    the win/hit decisions coincide -- this crafted 2-class input is engineered so both
    agree: row0's col-1 target (score 60) clears BOTH the thresh=10 hit and the
    max-competitor win, and row1 has no target so BOTH skip it. There is no PyO3 seam for
    confusion (it is not on the speech_rs.Engine surface), so the identity is pinned by
    this test plus its Rust twin hardcoding the SAME expected matrix."""
    scores = np.array([[5.0, 60.0], [20.0, 30.0]])  # row0's target col 1 (wire 260 -> 60); row1 none
    cm = confusion_matrix(np.array([1, -1]), scores, thresh=10.0)
    want = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],  # labels clean (row1 no-target -> skipped)
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 1.0],  # row0 win at (2,2)
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert np.array_equal(cm, want), cm


def test_confusion_matrix_class_nb_le_1_raises() -> None:
    with pytest.raises(ValueError, match="class_nb"):
        confusion_matrix(np.array([0, 0]), np.array([[1.0], [2.0]]), thresh=10.0)
