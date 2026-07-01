"""LID confusion matrices, EER cutoff, gradient checks, masking sanity.

Ported from legacy MATLAB: confusion*.m, CheckGrad.m, MaskingValidation.m.
See design spec section 6.

Also hosts the numpy differential oracle for the Rust segmenter decision
(`update_segmentation_oracle`), used only in tests to cross-check Rust ==
Python bit-for-bit on random inputs.
"""

import numpy as np
from numpy.typing import NDArray

# SegClass codes, matching the Rust `SegClass as i32` (`segmentation.rs`) and the
# legacy `segment_class` enum (`Segmentation.h`).
OTHER_CODE = 0
SPEECH_CODE = 1
INSERTION_CODE = 19
SUBSTITUTION_CODE = 20
EXCLUDED_CODE = 21
END_CODE = 22
N_CLASSES = 23


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


def _sanitize_bl(bl: list[list[np.float64 | int]]) -> None:
    """In-place port of `Segmentation::sanitize` (`Segmentation.cpp:176-194`).

    Snap each non-terminal boundary to the 1e-4 grid (round-half-away-from-zero,
    matching `boost::math::round` and Rust `f64::round`) and merge adjacent
    same-type segments. `bl` is `[[begin, code], ...]` ending with the End
    sentinel, which is never rounded (the loop stops at the sentinel pair).
    """
    i = 0
    while i < len(bl) and i + 1 < len(bl):
        scaled = np.float64(bl[i][0]) * np.float64(1.0e4)
        # `np.round` is round-half-to-even; Rust `f64::round` (and
        # `boost::math::round`) is round-half-AWAY-from-zero. Reproduce the latter
        # so ties snap identically. `copysign(floor(|x| + 0.5), x)` is the exact
        # half-away-from-zero rule for either sign.
        rounded = np.copysign(np.floor(np.abs(scaled) + np.float64(0.5)), scaled)
        bl[i][0] = rounded / np.float64(1.0e4)
        if bl[i][1] == bl[i + 1][1]:
            del bl[i + 1]
        else:
            i += 1


def _modify_type_bl(bl: list[list[np.float64 | int]], before: int, after: int) -> None:
    """In-place port of `modifySegmentsType` (`Segmentation.cpp:196-204`).

    Retype every non-sentinel segment from `before` to `after`, then sanitize
    (the legacy calls `sanitize` at the tail; the Rust `modify_type` matches).
    """
    i = 0
    while i + 1 < len(bl):
        if bl[i][1] == before:
            bl[i][1] = after
        i += 1
    _sanitize_bl(bl)


def _update_count_bl(bl: list[list[np.float64 | int]]) -> NDArray[np.float64]:
    """Port of `update_count` (`Segmentation.cpp:206-214`) + Rust `update_count`.

    Sanitize, then accumulate each segment's duration `(next.begin - begin)` into
    a per-class table indexed by code. The sentinel is never counted (loop stops
    at the sentinel pair), so `count[End]` stays 0.0.
    """
    _sanitize_bl(bl)
    count = np.zeros(N_CLASSES, dtype=np.float64)
    i = 0
    while i + 1 < len(bl):
        count[int(bl[i][1])] += np.float64(bl[i + 1][0]) - np.float64(bl[i][0])
        i += 1
    return count


def compute_errors_pass2_oracle(
    ref_bl: list[tuple[float, int]],
    hyp_bl: list[tuple[float, int]],
    audio_duration: float,
) -> list[list[float]]:
    """Independent numpy port of `compute_errors` Pass 2 (`Segmentation.cpp:410-476`).

    Operates on boundary lists `ref_bl`/`hyp_bl` = `list[(begin, code)]`, each
    ending with an `End` sentinel at `audio_duration`. Returns
    `[[pmiss, pfa, error_rate], ...]` (index = class code, length 23), matching
    the Rust `compute_errors` `per_class` field bit-for-bit.

    Steps, in Rust `compute_errors` order (Pass 2 path, `nb_words < 0`):
      1. `hyp.sanitize()` (`Segmentation.cpp:410`).
      2. `modify_type` on the ref clone: `Substitution -> Speech`, then
         `Insertion -> Other` (`Segmentation.cpp:411-412`); each remap sanitizes.
      3. `update_count` on the modified ref -> `label_counts`
         (`Segmentation.cpp:413`); also sanitizes.
      4. Two-pointer walk (`itRef` idx 0, `it` idx 1) accumulating raw-seconds
         Pfa/Pmiss/ErrorRate (`Segmentation.cpp:426-454`).
      5. Normalize per class `j in [Other, Excluded)` with the zero-guards
         (`Segmentation.cpp:456-476`).

    All arithmetic uses `np.float64` (IEEE-754 doubles), so the additions and
    divisions round identically to the Rust `f64` port.
    """
    dur = np.float64(audio_duration)

    hyp: list[list[np.float64 | int]] = [[np.float64(b), int(c)] for (b, c) in hyp_bl]
    _sanitize_bl(hyp)

    refc: list[list[np.float64 | int]] = [[np.float64(b), int(c)] for (b, c) in ref_bl]
    _modify_type_bl(refc, SUBSTITUTION_CODE, SPEECH_CODE)
    _modify_type_bl(refc, INSERTION_CODE, OTHER_CODE)
    label_counts = _update_count_bl(refc)

    pmiss = np.zeros(N_CLASSES, dtype=np.float64)
    pfa = np.zeros(N_CLASSES, dtype=np.float64)
    error_rate = np.zeros(N_CLASSES, dtype=np.float64)

    r = refc
    h = hyp
    end_ref = len(r)
    end_hyp = len(h)
    it_ref = 0
    it = 1
    while it_ref != end_ref and it != end_hyp:
        r_begin = np.float64(r[it_ref][0])
        h_begin = np.float64(h[it][0])
        if r_begin <= h_begin:
            ref_ty = int(r[it_ref][1])
            prev_hyp = int(h[it - 1][1])
            scorable = ref_ty != END_CODE and ref_ty != EXCLUDED_CODE and ref_ty != prev_hyp and not (prev_hyp == SPEECH_CODE and ref_ty == SUBSTITUTION_CODE)
            if scorable:
                r_next = np.float64(r[it_ref + 1][0])
                error = r_next - r_begin if h_begin > r_next else h_begin - r_begin
                pmiss[ref_ty] += error
                error_rate[prev_hyp] += error
                pfa[prev_hyp] += error
            it_ref += 1
        else:
            hyp_ty = int(h[it][1])
            prev_ref = int(r[it_ref - 1][1])
            scorable = hyp_ty != END_CODE and prev_ref != EXCLUDED_CODE and prev_ref != hyp_ty
            if scorable:
                h_next = np.float64(h[it + 1][0])
                error = h_next - h_begin if r_begin > h_next else r_begin - h_begin
                pmiss[prev_ref] += error
                error_rate[hyp_ty] += error
                pfa[hyp_ty] += error
            it += 1

    count_excluded = label_counts[EXCLUDED_CODE]
    for j in range(EXCLUDED_CODE):
        count = label_counts[j]
        count_others = dur - count - count_excluded
        if dur - count_excluded > np.float64(0.0):
            error_rate[j] = error_rate[j] / (dur - count_excluded)
        else:
            error_rate[j] = np.float64(0.0)
        if count > np.float64(0.0):
            pmiss[j] = pmiss[j] / count
        else:
            pmiss[j] = np.float64(0.0)
        if count_others > np.float64(0.0):
            pfa[j] = pfa[j] / count_others
        else:
            pfa[j] = np.float64(0.0)

    return [[float(pmiss[j]), float(pfa[j]), float(error_rate[j])] for j in range(N_CLASSES)]
