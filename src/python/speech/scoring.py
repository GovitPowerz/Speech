"""LID confusion matrices, EER cutoff, masking sanity.

Ported from legacy MATLAB: confusion*.m, MaskingValidation.m. See design spec section 6.
(The CheckGrad.m port, `check_grad`, was retired in issue #103: the port's gradient check
is the engine's own `Engine.grad_check`, pinned through the seam. The numpy oracles of the
segmenter decision and of `compute_errors` Pass 2 moved to `tests/oracles/scoring.py` in
issue #133.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from speech.genome import RunConfig, vec2struct


def confusion_matrix(lid_target: NDArray[np.int64], lid_scores: NDArray[np.float64], thresh: float) -> NDArray[np.float64]:
    """Port of legacy `confusionThresh.m` (`Optimizer_V6.2.2/functions/confusionThresh.m`).

    `lid_scores` is one row per scored file, one column per language class, the TARGET
    column already decoded; `lid_target` names that column per row (`-1`: no target).
    The legacy read the target in-band (`score > 150`, real score `= value - 200`), the
    same convention the Rust `engine/confusion.rs::confusion_from_results` consumed (a
    different legacy source, `BagOfProcessors.cpp`'s `PrintConfusionMatrix`); since issue
    #23 both take the decoded form. Returns a `(class_nb+2) x (class_nb+2)` matrix:
    row/col 0 carry the `0..class_nb` axis labels, the last row/col accumulate per-row/
    col totals. A file scores a HIT only if its target lands in matrix position 1 or 2
    (0-based; the legacy's 1-based `posTarget` 2 or 3) AND clears the asymmetric
    threshold (`thresh` at position 2, `100-thresh` at position 1); otherwise it is
    charged against the best-scoring non-target class.

    PER-ROW DECODE (FIXED in phase 5; see IMPROVEMENTS.md): the competitor accumulators
    (`maxScoreNotTarget`/`posBestNotTarget`) are declared INSIDE the file loop, so a row
    can never inherit a prior row's target/competitor index. A file with no target
    (`lid_target == -1`, legacy: no score `> 150`, `pos_target == 0`) is an out-of-set
    trial with no true class in the closed set, so it is SKIPPED entirely -- credited to no
    cell and never to the index-header slot, making the matrix ORDER-INDEPENDENT. The
    legacy `confusionThresh.m` instead declared `posTarget`/`posBestNotTarget` ONCE outside
    the loop (only `maxScoreNotTarget`/`scoreTarget` reset per row), so a no-target file
    silently reused the PREVIOUS row's `posTarget`; the Octave oracle still describes that
    sticky behavior, the port deliberately diverges. Fixed identically to the Rust
    `engine/confusion.rs::confusion_from_results` (F5, both languages).

    `class_nb <= 1` raises: the legacy's `if (classNb > 1)` has no `else`, so `confusion`
    is never assigned and MATLAB itself would error "output argument not assigned".
    """
    lid_scores = np.asarray(lid_scores, dtype=np.float64)
    lid_target = np.asarray(lid_target, dtype=np.int64)
    class_nb = lid_scores.shape[1] if lid_scores.ndim == 2 else 0
    if class_nb <= 1:
        raise ValueError(f"confusion_matrix: class_nb={class_nb} <= 1 -- legacy confusionThresh.m leaves `confusion` unassigned here")

    confusion = np.zeros((class_nb + 2, class_nb + 2), dtype=np.float64)
    for kk in range(class_nb + 1):
        confusion[0, kk] = kk
        confusion[kk, 0] = kk

    for jj in range(lid_scores.shape[0]):
        t = int(lid_target[jj])
        if t < 0:
            # No target this row: out-of-set trial, credited to no class.
            continue
        pos_target = t + 1
        score_target = float(lid_scores[jj, t])
        max_score_not_target = -1.0
        pos_best_not_target = 0
        for kk in range(class_nb):
            v = float(lid_scores[jj, kk])
            if kk != t and v > max_score_not_target:
                max_score_not_target = v
                pos_best_not_target = kk + 1

        if (pos_target == 2 and score_target > thresh) or (pos_target == 1 and score_target > 100.0 - thresh):
            confusion[pos_target, pos_target] += 1.0
            confusion[pos_target, class_nb + 1] += 1.0
            confusion[class_nb + 1, pos_target] += 1.0
        else:
            confusion[pos_target, pos_best_not_target] += 1.0
            confusion[pos_target, class_nb + 1] += 1.0
            confusion[class_nb + 1, pos_best_not_target] += 1.0

    return confusion


def _is_numeric_field(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def masking_validation(param: NDArray[np.float64], mask: dict[str, object] | None, ps: RunConfig) -> list[str]:
    """Port of `MaskingValidation.m` (`Optimizer_V6.2.2/functions/MaskingValidation.m`).

    Double `vec2struct` round trip (:6-7): `vec2struct(param, mask, ps)` -> `configStruct`
    (the mask-forced config) + `out_param` (the write-back), then
    `vec2struct(out_param, None, ps)` -> `maskedConfigStruct` (out_param DECODED FRESH,
    no mask). Every field present in `configStruct` is compared against the same field in
    `maskedConfigStruct`: string fields via exact match (:31), numeric fields via
    `abs(diff) > 1e-12` (:39/:45) -- `speech.genome.vec2struct` already returns the
    printConfig-formatted STRING dict, so numeric-vs-string is decided by whether the
    value parses as a float (matching the legacy `ischar` branch in spirit: `%d`/
    `%15.15e`-formatted numbers always parse, genuine string fields like window-type
    names or "true"/"false" never do).

    Returns the list of mismatched field names (empty = pass, matching `hasFailed`
    collection order -- `configStruct`'s field order, i.e. insertion order).
    """
    config, out_param, _ = vec2struct(param, mask, ps, 0)
    masked_config, _, _ = vec2struct(out_param, None, ps, 0)

    mismatches: list[str] = []
    for name, v1 in config.items():
        v2 = masked_config[name]
        f1 = _is_numeric_field(v1)
        if f1 is not None:
            f2 = _is_numeric_field(v2)
            if f2 is None or abs(f1 - f2) > 1e-12:
                mismatches.append(name)
        elif v1 != v2:
            mismatches.append(name)
    return mismatches
