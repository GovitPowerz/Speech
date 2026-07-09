"""Phase 4c Task 10: check_grad, masking_validation, confusion_matrix.

check_grad/masking_validation goldens (`tests/reference_data/phase4c/{checkgrad,
masking}_*.bin`) are dumped by GNU Octave running the REAL `legacy/Optimizer_V6.2.2/
functions/{CheckGrad,MaskingValidation,vec2struct}.m` (see
`tools/octave_harness/stage_{checkgrad,masking}.m`). confusion_matrix has no Octave
golden (the brief scopes only GetNewBatch/CheckGrad/MaskingValidation as Octave stages);
its tests are direct unit tests hand-verified against `confusionThresh.m`'s logic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from speech.genome import RunConfig
from speech.scoring import check_grad, confusion_matrix, masking_validation
from speech.weight_bridge import read_bin

from tests._libm_gate import assert_f64_matrix_close

PHASE4C = Path(__file__).resolve().parent / "reference_data" / "phase4c"
MANIFEST = PHASE4C / "manifest.json"

CostFn = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


def _load(name: str) -> NDArray[np.float64]:
    rows, cols, flat = read_bin(PHASE4C / f"{name}.bin")
    return np.asarray(flat, dtype=np.float64).reshape((rows, cols), order="F")


# --------------------------------------------------------------------------- #
# check_grad (vs the checkgrad_shadow/CostFunction.m quadratic surrogate golden)
# --------------------------------------------------------------------------- #


def _surrogate_cost_fn(kw: int) -> tuple[NDArray[np.float64], NDArray[np.float64], CostFn]:
    """Reproduces checkgrad_shadow/CostFunction.m's quadratic surrogate exactly:
    f(w) = 0.5*sum(c.*(w-target)^2), df/dw = c.*(w-target), with
    target = 0.1*sin(0.37*idx), c = 1+0.05*mod(idx,5), idx 1-based 1..Kw."""
    idx = np.arange(1, kw + 1, dtype=np.float64)
    target = 0.1 * np.sin(0.37 * idx)
    c = 1.0 + 0.05 * np.mod(idx, 5)

    def cost_fn(w: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        d = w - target
        return float(0.5 * np.sum(c * d * d)), c * d

    return target, c, cost_fn


def test_checkgrad_manifest_present() -> None:
    assert MANIFEST.is_file()
    m = json.loads(MANIFEST.read_text())
    assert m["checkgrad"]["Kw"] == 53
    assert m["checkgrad"]["n"] == 107


def test_check_grad_analytic_matches_golden() -> None:
    """The analytic gradient (one cost_fn call, mirroring CheckGrad.m:36) matches the
    real CheckGrad.m's MultiDeriv_BackProp -- traverses `sin`, so canary-gated close."""
    weights0 = _load("checkgrad_MultiWeights").reshape(-1)
    golden_analytic = _load("checkgrad_MultiDeriv_BackProp").reshape(-1)
    _, _, cost_fn = _surrogate_cost_fn(weights0.size)

    report = check_grad(cost_fn, weights0, epsilon=1e-5)
    assert_f64_matrix_close(report.analytic, golden_analytic, "check_grad analytic vs CheckGrad.m MultiDeriv_BackProp")


def test_check_grad_numeric_matches_golden_well_behaved_prefix() -> None:
    """The central-diff numeric derivative matches CheckGrad.m's MultiDeriv_Num for the
    51 weight entries the legacy round trip actually perturbs.

    ALWAYS-TOLERANT (not the libm canary comparator): Octave's numeric column is a
    central diff over `weights` re-derived through the FULL legacy
    vec2struct/nnet2MatFile/weights2nnet/network2config round trip for every perturbed
    point, while this port's `cost_fn` perturbs `weights` directly -- two genuinely
    different computational paths, not a same-path libm split, so bit/ULP exactness is
    not the right bar even on the oracle env. Measured max abs diff ~1e-8 (float
    round-trip noise through the adim/coeff scale-unscale chain, not truncation error --
    the surrogate is an exact quadratic form)."""
    weights0 = _load("checkgrad_MultiWeights").reshape(-1)
    golden_numeric = _load("checkgrad_MultiDeriv_Num").reshape(-1)
    _, _, cost_fn = _surrogate_cost_fn(weights0.size)

    report = check_grad(cost_fn, weights0, epsilon=1e-5)
    assert np.allclose(report.numeric[:-2], golden_numeric[:-2], atol=1e-6, rtol=1e-6), (
        f"check_grad numeric[:-2] vs CheckGrad.m MultiDeriv_Num[:-2]: max diff {np.max(np.abs(report.numeric[:-2] - golden_numeric[:-2]))}"
    )


def test_check_grad_normalize_tail_quirk_documented_not_reproduced() -> None:
    """`weights2nnet.m` (:150-186) never writes back the normalize mean/std tail
    `nnet2MatFile.m` (:140-141) appends to `weights` -- so the REAL CheckGrad.m's last 2
    numeric derivatives are always exactly 0 (golden), despite a nonzero analytic value.
    This port's `check_grad` perturbs `weights` DIRECTLY (no config round trip), so it
    does NOT reproduce that bug: both tail entries get a proper nonzero numeric gradient,
    matching the analytic one almost exactly (central diff on an exact quadratic form
    carries no truncation error). See IMPROVEMENTS.md."""
    weights0 = _load("checkgrad_MultiWeights").reshape(-1)
    golden_numeric = _load("checkgrad_MultiDeriv_Num").reshape(-1)
    golden_analytic = _load("checkgrad_MultiDeriv_BackProp").reshape(-1)
    assert golden_numeric[-2:].tolist() == [0.0, 0.0], "golden fixture must exhibit the quirk (numeric==0)"
    assert golden_analytic[-2] != 0.0 and golden_analytic[-1] != 0.0

    _, _, cost_fn = _surrogate_cost_fn(weights0.size)
    report = check_grad(cost_fn, weights0, epsilon=1e-5)
    assert np.all(report.numeric[-2:] != 0.0), "the generic port must NOT reproduce the weights2nnet.m tail-drop bug"
    assert np.allclose(report.numeric[-2:], report.analytic[-2:], atol=1e-6)


def test_check_grad_denom_floor_matches_lid_guard_formula() -> None:
    """CheckGrad.m:148's LID-branch divisor `max(1e-24, 2*epsilon)`; for epsilon=1e-5 the
    floor never actually binds (2e-5 > 1e-24), so denom_floor is a documented no-op here
    -- this pins the FORMULA, not a distinguishing numeric outcome."""

    def cost_fn(w: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        return float(0.5 * np.sum(w * w)), w

    w0 = np.array([1.0, -2.0, 3.0])
    plain = check_grad(cost_fn, w0, epsilon=1e-5)
    floored = check_grad(cost_fn, w0, epsilon=1e-5, denom_floor=1e-24)
    assert np.allclose(plain.numeric, floored.numeric)


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
    scores = np.array([[151.0, 10.0]])
    cm = confusion_matrix(scores, thresh=10.0)
    assert cm.shape == (4, 4)
    assert cm[0, :3].tolist() == [0.0, 1.0, 2.0]
    assert cm[:3, 0].tolist() == [0.0, 1.0, 2.0]
    assert cm[3, 0] == 0.0 and cm[0, 3] == 0.0  # the totals row/col is unlabeled


def test_confusion_matrix_hit_when_target_clears_threshold() -> None:
    """Target in slot 1 (0-based; legacy posTarget=2) with score > thresh -> a HIT on
    the diagonal, not charged to the best non-target class."""
    scores = np.array([[10.0, 260.0]])  # class1 signaled target, score = 260-200 = 60
    cm = confusion_matrix(scores, thresh=10.0)
    assert cm[2, 2] == 1.0  # pos_target = kk(1)+1 = 2, diagonal hit
    assert cm[2, 3] == 1.0 and cm[3, 2] == 1.0


def test_confusion_matrix_miss_charges_best_non_target() -> None:
    scores = np.array([[151.0, 10.0]])  # target=class0 (score=-49), best-not-target=class1
    cm = confusion_matrix(scores, thresh=10.0)
    # pos_target = 0+1 = 1, pos_best_not_target = 1+1 = 2 (miss: -49 > 100-10=90 is false)
    assert cm[1, 2] == 1.0
    assert cm[1, 3] == 1.0 and cm[3, 2] == 1.0
    assert cm[1, 1] == 0.0  # not a diagonal hit


def test_confusion_matrix_sticky_pos_target_quirk() -> None:
    """`posTarget`/`posBestNotTarget` are declared ONCE outside the file loop and never
    reset per row (confusionThresh.m has no per-row reset, only maxScoreNotTarget/
    scoreTarget are reset) -- a row with NO score > 150 silently reuses the PREVIOUS
    row's posTarget/posBestNotTarget, decided against the CURRENT row's threshold check
    (using the stale scoreTarget=-1.0, reset at the end of every row)."""
    scores = np.array(
        [
            [151.0, 10.0],  # row0: target=class0, best-not-target=class1
            [20.0, 30.0],  # row1: no target signaled -> posTarget/posBestNotTarget STICKY
        ]
    )
    cm = confusion_matrix(scores, thresh=10.0)
    # Both rows charge pos_target=1 (sticky), pos_best_not_target=2: cm[1,2] accumulates 2.
    assert cm[1, 2] == 2.0
    assert cm[1, 3] == 2.0
    assert cm[3, 2] == 2.0


def test_confusion_matrix_class_nb_le_1_raises() -> None:
    with pytest.raises(ValueError, match="class_nb"):
        confusion_matrix(np.array([[1.0], [2.0]]), thresh=10.0)
