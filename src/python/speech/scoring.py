"""Masking sanity.

Ported from legacy MATLAB: MaskingValidation.m. See design spec section 6.
(The CheckGrad.m port, `check_grad`, was retired in issue #103: the port's gradient check
is the engine's own `Engine.grad_check`, pinned through the seam. The confusionThresh.m
port, `confusion_matrix`, was retired in issue #155: it had no caller once the per-epoch
LID error became `evaluate.lid_error`, #149.) The numpy oracles of the segmenter decision
and of `compute_errors` Pass 2 are `tests/oracles/scoring.py` since issue #133.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from speech.genome import RunConfig, vec2struct


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
