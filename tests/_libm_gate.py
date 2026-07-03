"""Shared canary-gated cross-libm comparator for Python oracle tests.

Phase 1/2 oracle dumps were produced on the oracle env (Apple libm). Any oracle
chain that traverses a libm transcendental (``exp``/``log``/``asinh``/``cos``)
rounds 1 ULP differently under a different libm (e.g. CI glibc), which downstream
arithmetic can amplify to a few ULP. Bit-exactness holds ON THE ORACLE ENV;
elsewhere these helpers relax to a tight hybrid ULP-or-scaled-absolute bound,
gated by the same libm canary fixture the Rust side uses
(``src/rust/tests/common/mod.rs::oracle_mode()``, same 10 cos / 5 log / 3 exp
column layout in ``tests/reference_data/phase1/libm_canaries.bin``).

``SPEECH_ORACLE_LIBM=strict|ulp`` forces a path (mirrors the Rust env override).
"""

from __future__ import annotations

import math
import os
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from speech.weight_bridge import read_bin

_CANARY_PATH = Path(__file__).resolve().parent / "reference_data" / "phase1" / "libm_canaries.bin"

_HYBRID_ABS_FACTOR = 512.0


@cache
def oracle_strict() -> bool:
    """True if THIS platform's libm matches the recorded oracle canaries bit-for-bit."""
    forced = os.environ.get("SPEECH_ORACLE_LIBM")
    if forced == "strict":
        return True
    if forced == "ulp":
        return False
    rows, _, flat = read_bin(_CANARY_PATH)  # 2 x N column-major
    n = flat.size // rows
    for k in range(n):
        x = float(flat[k * rows + 0])
        want = float(flat[k * rows + 1])
        got = math.cos(x) if k < 10 else math.log(x) if k < 15 else math.exp(x)
        if got.hex() != want.hex():
            return False
    return True


def assert_f32_close(got: np.float32, want: np.float32, label: str) -> None:
    """Exact on the oracle libm, else hybrid: <=2 f32 ULP or an absolute tolerance
    of ``512 * 2**-23 * max(|expected|, 1.0)`` (for f32-valued oracles, e.g. fmath).

    Mirrors the Rust ``assert_oracle_eq_f32`` hybrid comparator.
    """
    if oracle_strict():
        assert got == want, f"{label}: {got!r} != {want!r}"
        return
    assert np.isfinite(got) and np.isfinite(want), f"{label}: non-finite {got!r} {want!r}"
    a = int(np.float32(got).view(np.int32))
    b = int(np.float32(want).view(np.int32))
    ulp = abs(a - b) if np.signbit(got) == np.signbit(want) else None
    abs_diff = abs(float(got) - float(want))
    abs_tol = _HYBRID_ABS_FACTOR * 2.0**-23 * max(abs(float(want)), 1.0)
    assert (ulp is not None and ulp <= 2) or abs_diff <= abs_tol, (
        f"{label}: f32 ULP {'sign' if ulp is None else ulp} > 2 and |diff|={abs_diff!r} > abs_tol={abs_tol!r} ({got!r} vs {want!r})"
    )


def assert_f64_close(got: float, want: float, label: str) -> None:
    """Exact on the oracle libm, else hybrid: <=4 f64 ULP or an absolute tolerance
    of ``512 * 2**-52 * max(|expected|, 1.0)``.

    Mirrors the Rust ``assert_oracle_eq_f64`` scalar hybrid comparator.
    """
    if oracle_strict():
        assert float(got).hex() == float(want).hex(), f"{label}: {got!r} != {want!r}"
        return
    assert math.isfinite(got) and math.isfinite(want), f"{label}: non-finite {got!r} {want!r}"
    a = int(np.float64(got).view(np.int64))
    b = int(np.float64(want).view(np.int64))
    ulp = abs(a - b) if np.signbit(got) == np.signbit(want) else None
    abs_diff = abs(float(got) - float(want))
    abs_tol = _HYBRID_ABS_FACTOR * 2.0**-52 * max(abs(float(want)), 1.0)
    assert (ulp is not None and ulp <= 4) or abs_diff <= abs_tol, (
        f"{label}: f64 ULP {'sign' if ulp is None else ulp} > 4 and |diff|={abs_diff!r} > abs_tol={abs_tol!r} ({got!r} vs {want!r})"
    )


def assert_f64_matrix_close(got: NDArray[np.float64], want: NDArray[np.float64], label: str) -> None:
    """Elementwise matrix variant of `assert_f64_close`. `scale` (Ulp-mode absolute
    tolerance) is `max(|want|)` over the WHOLE matrix, mirroring the Rust
    `assert_oracle_eq` matrix comparator (not per-element `max(|want_i|, 1.0)`)."""
    assert got.shape == want.shape, f"{label}: shape mismatch {got.shape} != {want.shape}"
    if oracle_strict():
        assert got.tobytes() == np.ascontiguousarray(want, dtype=np.float64).tobytes(), f"{label}: bit-exact mismatch"
        return
    scale = float(np.max(np.abs(want))) if want.size else 0.0
    abs_tol = _HYBRID_ABS_FACTOR * 2.0**-52 * scale
    got_flat = got.reshape(-1)
    want_flat = want.reshape(-1)
    for idx in range(got_flat.size):
        a = float(got_flat[idx])
        b = float(want_flat[idx])
        assert math.isfinite(a) and math.isfinite(b), f"{label}: non-finite at flat index {idx} ({a!r} vs {b!r})"
        ai = int(np.float64(a).view(np.int64))
        bi = int(np.float64(b).view(np.int64))
        ulp = abs(ai - bi) if np.signbit(a) == np.signbit(b) else None
        abs_diff = abs(a - b)
        assert (ulp is not None and ulp <= 4) or abs_diff <= abs_tol, (
            f"{label}: at flat index {idx} ULP={'sign' if ulp is None else ulp} > 4 and |diff|={abs_diff!r} > abs_tol={abs_tol!r} ({a!r} vs {b!r})"
        )
