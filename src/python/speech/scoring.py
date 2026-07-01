"""LID confusion matrices, EER cutoff, gradient checks, masking sanity.

Ported from legacy MATLAB: confusion*.m, CheckGrad.m, MaskingValidation.m.
See design spec section 6.

Also hosts the numpy differential oracle for the Rust segmenter decision
(`update_segmentation_oracle`), used only in tests to cross-check Rust ==
Python bit-for-bit on random inputs.
"""

import numpy as np
from numpy.typing import NDArray

# SegClass code for SPEECH, matching the Rust `SegClass::Speech as i32`
# (`segmentation.rs`) and the legacy `segment_class` enum.
SPEECH_CODE = 1


def confusion_matrix(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> NDArray[np.int64]:
    """Per-language confusion matrix with in-band target signaling (Phase 4)."""
    ...


def update_segmentation_oracle(
    results_row: list[float],
    rising: float,
    area_rising: float,
    falling: float,
    area_falling: float,
    dt: float,
    off: float,
    class_code: int = SPEECH_CODE,
) -> list[tuple[float, float, int]]:
    """Independent numpy port of the RAW segmenter decision (spec 3.2, NO smoothing).

    Faithful to `Segmenter::updateSegmentation` (`Segmenter.cpp:725-845`), the
    decision body only: the INIT rising check, the rising/falling hysteresis with
    area gating, the re-run-rising-after-label, and the `results.size()` tail.
    Returns the pre-smoothing, pre-sanitize labeled segments as
    `(begin + off, end + off, class_code)`, one per legacy `label_segment` call,
    in emission order. This isolates the decision math for the Rust cross-check
    (`update_segmentation_raw`).

    Every arithmetic expression, comparison direction (`>=`/`<`/`<=`/`>`), and the
    negative-by-construction area term `(r_prev - t) * (t - r_prev)` are kept in the
    exact legacy form so f64 rounding matches Rust bit-for-bit. `np.float64` scalars
    are IEEE-754 doubles, so this reproduces the same rounding as the Rust `f64` port.
    """
    r = np.asarray(results_row, dtype=np.float64)
    length = r.shape[0]

    t_r = np.float64(rising)
    a_r = np.float64(area_rising)
    t_f = np.float64(falling)
    a_f = np.float64(area_falling)
    dt_f = np.float64(dt)
    off_f = np.float64(off)

    segments: list[tuple[float, float, int]] = []

    begin = np.float64(-1.0)
    end = np.float64(-1.0)
    begin_area = np.float64(-1.0)
    end_area = np.float64(-1.0)
    has_begun = False
    has_ended = False

    if length > 0 and r[0] >= t_r:
        begin = np.float64(0.0)
        begin_area = np.float64(0.0)

    for ii in range(1, length):
        rv = r[ii]
        r_prev = r[ii - 1]
        ii_f = np.float64(ii)

        if not has_begun:
            if begin < 0.0:
                if rv >= t_r and r_prev < t_r:
                    begin = dt_f * (ii_f - (rv - t_r) / (rv - r_prev))
                    begin_area = (ii_f - begin / dt_f) * (rv - t_r) / np.float64(2.0)
                    if begin_area >= a_r:
                        has_begun = True
            elif rv >= t_r:
                begin_area += (rv + r_prev - t_r * np.float64(2.0)) / np.float64(2.0)
                if begin_area >= a_r:
                    has_begun = True
            elif rv < t_r and r_prev >= t_r:
                begin_area += (r_prev - t_r) * (t_r - r_prev) / (rv - r_prev) / np.float64(2.0)
                if begin_area >= a_r:
                    has_begun = True
                else:
                    begin = np.float64(-1.0)
                    begin_area = np.float64(-1.0)
                    has_begun = False
            else:
                begin = np.float64(-1.0)
                begin_area = np.float64(-1.0)
                has_begun = False

        if has_begun:
            if end < 0.0:
                if rv <= t_f and r_prev > t_f:
                    end = dt_f * (ii_f - (rv - t_f) / (rv - r_prev))
                    end_area = (ii_f - end / dt_f) * (t_f - rv) / np.float64(2.0)
                    if end_area >= a_f:
                        has_ended = True
            elif rv <= t_f:
                end_area += (np.float64(2.0) * t_f - rv - r_prev) / np.float64(2.0)
                if end_area >= a_f:
                    has_ended = True
            elif rv > t_f and r_prev <= t_f:
                end_area += (t_f - r_prev) * (t_f - r_prev) / (rv - r_prev) / np.float64(2.0)
                if end_area >= a_f:
                    has_ended = True
                else:
                    end = np.float64(-1.0)
                    end_area = np.float64(-1.0)
                    has_ended = False
            else:
                end = np.float64(-1.0)
                end_area = np.float64(-1.0)
                has_ended = False

            if has_ended:
                if begin < end:
                    segments.append((float(begin + off_f), float(end + off_f), class_code))
                    begin = np.float64(-1.0)
                    begin_area = np.float64(-1.0)
                    has_begun = False
                    end = np.float64(-1.0)
                    end_area = np.float64(-1.0)
                    has_ended = False

                    if rv >= t_r and r_prev < t_r:
                        begin = dt_f * (ii_f - (rv - t_r) / (rv - r_prev))
                        begin_area = (ii_f - begin / dt_f) * (rv - t_r) / np.float64(2.0)
                        if begin_area >= a_r:
                            has_begun = True
                else:
                    begin = np.float64(-1.0)
                    begin_area = np.float64(-1.0)
                    has_begun = False
                    end = np.float64(-1.0)
                    end_area = np.float64(-1.0)
                    has_ended = False

    if has_begun:
        end = dt_f * np.float64(length)
        segments.append((float(begin + off_f), float(end + off_f), class_code))

    return segments
