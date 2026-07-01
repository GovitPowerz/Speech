"""Numpy oracle for the shared regression-deltas kernel (Task 6).

Independent (vectorized) coding of the same clamped-formula kernel that the Rust
`speech::features::mel::regression_deltas` implements sequentially. The two are
cross-checked bit-for-bit via `tests/reference_data/phase1/deltas_cases.json`
(emitted by `scripts/extract_phase1_oracle_cases.py`); the hand anchor in
`tests/test_features_oracle.py` is the arbiter.

Legacy source: MelFilterBank.cpp:153-170 (the deltas block-op sequence). The
kernel is

    D[t] = sum_{j=1..n} j * (B[min(t+j, T-1)] - B[max(t-j, 0)]) / (2 * sum_{j} j^2)

where the denominator sums j^2 over ALL j in 1..n (the legacy `adim` accumulates
j*j even when T <= j and the shifted copy is skipped -- the clamp saturates but
the term still contributes).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def regression_deltas_oracle(base: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    t = base.shape[0]
    acc = np.zeros_like(base)
    denom = 0.0
    for j in range(1, n + 1):
        plus = base[np.minimum(np.arange(t) + j, t - 1)]
        minus = base[np.maximum(np.arange(t) - j, 0)]
        acc += j * (plus - minus)
        denom += j * j
    return acc / (2.0 * denom)
