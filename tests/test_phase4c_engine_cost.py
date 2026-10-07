"""Phase 4c Task 9: the ComputeCost cost-assembly, bit-pinned vs Octave.

The goldens (`tests/reference_data/phase4c/computecost_*`) are dumped by GNU Octave running
`tools/octave_harness/stage_computecost.m` -- a FALLBACK-TIER stage-local transcription of
the VENDORED `legacy/Optimizer_V6.2.2/functions/ComputeCost.m` pure ASSEMBLY lines
(:285-652), since ComputeCost.m's top half shells out to the engine and cannot run in
Octave (see the stage header + `scripts/extract_phase4c_computecost_fixtures.py`). Each
group pins the Python port (`speech.engine`) against Octave's real MATLAB semantics
(sortrows stable-ascending, median, hist center-binning, std ddof=1, cumsum, exp/log):

  * average_derivs (col0/max(1,count)), l2_penalty,
    pooled mean/nb, the crafted-integer balance 0/3/4/5, and every cpu_mean (median) golden
    -> STRICT (bit-exact; median is a single sort + at-most-one /2, not a multi-term
    reduction, so it never picks up the `close` group's accumulation gap below).
  * pooled std (sqrt), the balance-10 LID calibration (incl. the b10c zero-zero
    interior-cutoff variant, :571-572), the committed-fixture realistic cases, and the
    fractional-`mean` cost scalars measured 0-ULP-exact on the oracle env
    (tier2_b0/tier2_b5/b10b/b10c) -> CANARY (the hybrid ULP-or-scaled-abs comparator,
    `tests/_libm_gate.py`; the manifest's `canary` list).
  * the fractional-`mean` cost scalars measured genuinely 1-ULP-off Octave even on the
    oracle env (cb3/cb4/b10a) -> CLOSE (`_assert_close_always` below; NOT libm-gated,
    since the gap is not a libm split -- see that function's docstring and the
    IMPROVEMENTS entry for the measured evidence).

The manifest's `strict`/`canary`/`close` lists are the single source of truth for which
comparator each `.bin` gets.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from speech.engine import (
    ChannelResults,
    CostBreakdown,
    CostParams,
    average_derivs,
    compute_cost,
    l2_penalty,
    pool_input_stats,
)

from tests._libm_gate import assert_f64_close
from tests._result_rows import channel_results_from_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE4C = REPO_ROOT / "tests" / "reference_data" / "phase4c"
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"
MANIFEST = cast(dict[str, Any], json.loads((PHASE4C / "computecost_manifest.json").read_text()))
STRICT: set[str] = set(cast(list[str], MANIFEST["strict"]))
CANARY: set[str] = set(cast(list[str], MANIFEST["canary"]))
CLOSE: set[str] = set(cast(list[str], MANIFEST["close"]))


def _assert_close_always(got: float, want: float, label: str) -> None:
    """A tight hybrid ULP-or-scaled-abs bound applied on EVERY platform (unlike
    `assert_f64_close`, which is exact on the oracle libm). For a `mean`/`mean(.^2)`
    reduction over fractional errors (cb3/cb4/b10a), numpy and Octave disagree by exactly
    1 ULP regardless of libm -- and NOT because of summation order: a plain sequential
    Python loop (`s=0.0; for v in x: s+=v; s/len(x)`) reproduces `np.mean`'s bit pattern
    exactly for these small vectors (order is invariant in f64 at this n), yet still
    differs from Octave by 1 ULP. The real cause is that Octave accumulates `mean`/`sum`
    internally in EXTENDED precision (x87/long-double), a gap no f64 loop order can close,
    so exact-on-oracle is unattainable -- but the gap is bounded the same way a libm gap
    would be: <=4 f64 ULP or abs <= 512*2^-52*max(|want|,1). See the IMPROVEMENTS entry for
    the measured evidence table (numpy == sequential-loop, both != Octave, on every case)."""
    assert math.isfinite(got) and math.isfinite(want), f"{label}: non-finite {got!r} {want!r}"
    a = int(np.float64(got).view(np.int64))
    b = int(np.float64(want).view(np.int64))
    ulp = abs(a - b) if np.signbit(got) == np.signbit(want) else (1 << 62)
    abs_tol = 512.0 * 2.0**-52 * max(abs(want), 1.0)
    assert ulp <= 4 or abs(got - want) <= abs_tol, f"{label}: {got!r} != {want!r} ({ulp} ULP, |diff|={abs(got - want)!r})"


def _read_bin_2d(path: Path) -> NDArray[np.float64]:
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    n = rows * cols
    return np.frombuffer(raw[16 : 16 + 8 * n], dtype="<f8").reshape((rows, cols), order="F").astype(np.float64)


def _cc(name: str) -> NDArray[np.float64]:
    return _read_bin_2d(PHASE4C / f"computecost_{name}.bin")


def _assert_against(got: NDArray[np.float64] | float, name: str, label: str | None = None) -> None:
    """Compare `got` against the `computecost_<name>.bin` golden using the comparator the
    manifest assigns it (STRICT bit-exact, else canary-gated hybrid bound)."""
    label = label or name
    bin_name = f"computecost_{name}.bin"
    want = _cc(name).reshape(-1)
    g = np.atleast_1d(np.asarray(got, dtype=np.float64)).reshape(-1)
    assert g.shape == want.shape, f"{label}: shape {g.shape} != {want.shape}"
    if bin_name in STRICT:
        for i in range(g.size):
            # `==` (not `.hex()`) so 0.0 and -0.0 -- e.g. l2_grad's masked bias slots -- compare equal.
            assert g[i] == want[i], f"{label}[{i}] strict: {g[i]!r} != {want[i]!r}"
    elif bin_name in CANARY:
        for i in range(g.size):
            assert_f64_close(float(g[i]), float(want[i]), f"{label}[{i}]")
    elif bin_name in CLOSE:
        for i in range(g.size):
            _assert_close_always(float(g[i]), float(want[i]), f"{label}[{i}]")
    else:
        raise AssertionError(f"{bin_name} not classified strict/canary/close in the manifest")


def _mcr(subdir: Path, name: str) -> NDArray[np.float64]:
    return _read_bin_2d(subdir / name)


def _rows(name: str) -> ChannelResults:
    """An Octave `MultiConfigResults` golden, decoded to names (`tests/_result_rows.py`).

    The crafted `cb_mcr` carries only the columns balances 0/3/4/5 read (legacy
    `Error_vad(:,1:5)` and `(:,end)`): a 6-wide result block. The decoder takes the full
    18-column row, so the block is widened with zeros between column 4 and the trailing
    counter, which keeps every column the Octave stage read where it read it."""
    mcr = _cc(name)
    width = mcr.shape[1] - 3
    if width < 18:
        pad = np.zeros((mcr.shape[0], 18 - width), dtype=np.float64)
        mcr = np.hstack([mcr[:, :-1], pad, mcr[:, -1:]])
    return channel_results_from_matrix(mcr)


# ==== aggregation / averaging / pooling / L2 (STRICT) ========================


def test_average_derivs() -> None:
    got = average_derivs(_cc("avg_in"))
    _assert_against(got, "avg_out")
    # Mutation guard: the count==0 row divides by max(1,0)==1 (not 0 -> inf/nan).
    assert np.isfinite(got).all() and got[1] == 10.0


def test_l2_penalty() -> None:
    w = _cc("l2_w").reshape(-1)
    is_bias = _cc("l2_isBias").reshape(-1)
    l2 = float(cast(float, MANIFEST["l2_regul"]))
    cost, grad = l2_penalty(w, is_bias, l2)
    _assert_against(cost, "l2_cost")
    _assert_against(grad, "l2_grad")
    # Mutation guard: the bias positions (is_bias==1) are zeroed in the gradient.
    assert grad[1] == 0.0 and grad[3] == 0.0


def test_pool_input_stats() -> None:
    nb = _cc("pool_nb").reshape(-1)
    mean = _cc("pool_mean")  # dim x worker
    std = _cc("pool_std")
    stats = [(int(nb[j]), mean[:, j], std[:, j]) for j in range(nb.shape[0])]
    total_nb, pooled_mean, pooled_std = pool_input_stats(stats)
    _assert_against(float(total_nb), "pool_out_nb")
    _assert_against(pooled_mean, "pool_out_mean")
    _assert_against(pooled_std, "pool_out_std")  # sqrt -> canary


# ==== balance laws 0/3/4/5 on crafted integer MCRs (STRICT error + cb0/cb5 cost) ==========


@pytest.mark.parametrize("balance", [0, 3, 4, 5])
def test_balance_crafted(balance: int) -> None:
    params = CostParams(
        mode=int(cast(int, MANIFEST["mode"])),
        balance_backprop=float(cast(float, MANIFEST["balance_backprop"])),
        algo=3,
    )
    cost, bd = compute_cost(_rows("cb_mcr"), 0, balance, params)
    _assert_against(bd.error, f"cb{balance}_error")
    _assert_against(cost, f"cb{balance}_cost")
    _assert_against(bd.nn_cost_seg, f"cb{balance}_nnseg")
    _assert_against(bd.cpu_mean, f"cb{balance}_cpumean")


def test_balance_3_vs_4_over90_saturation() -> None:
    """Mutation guard: the `0*` (balance 3) vs `1*` (balance 4) over-90 saturation
    coefficient MUST make the error vectors diverge on the Pfa/Pmiss>90 rows."""
    p = CostParams(mode=0, balance_backprop=0.5, algo=3)
    _, b3 = compute_cost(_rows("cb_mcr"), 0, 3, p)
    _, b4 = compute_cost(_rows("cb_mcr"), 0, 4, p)
    assert not np.array_equal(b3.error, b4.error), "over-90 saturation coefficient not exercised"


# ==== balance laws on the committed tier2 (algo 3) fixture (CANARY) =======================


@pytest.mark.parametrize("balance", [0, 5])
def test_balance_tier2_committed(balance: int) -> None:
    results = channel_results_from_matrix(_mcr(PHASE4A, "tier2_spectral_MultiConfigResults.bin"))
    params = CostParams(mode=0, balance_backprop=0.5, algo=3)
    cost, bd = compute_cost(results, 0, balance, params)
    _assert_against(bd.error, f"tier2_b{balance}_error")
    _assert_against(cost, f"tier2_b{balance}_cost")
    _assert_against(bd.cpu_mean, f"tier2_b{balance}_cpumean")


# ==== balance 10 (LID calibration) ========================================================


def test_balance10_two_class_cutoff_search() -> None:
    cost, bd = compute_cost(_rows("b10a_mcr"), 0, 10, CostParams(mode=0, algo=6))
    assert bd.cutoff is not None
    _assert_against(bd.cutoff, "b10a_cutoff")
    _assert_against(bd.error, "b10a_error")
    _assert_against(cost, "b10a_cost")
    _assert_against(bd.nn_cost_lid, "b10a_nnlid")
    _assert_against(bd.cpu_mean, "b10a_cpumean")
    # Mutation guard: the hist cutoff search landed a finite interior cutoff (not clamped).
    assert 0.1 < bd.cutoff < 99.9


def test_balance10_three_class_else_branch() -> None:
    cost, bd = compute_cost(_rows("b10b_mcr"), 0, 10, CostParams(mode=0, algo=6))
    # >2 classes -> the per-file cutoff-vector branch (no scalar cutoff).
    assert bd.cutoff is None
    _assert_against(bd.error, "b10b_error")
    _assert_against(cost, "b10b_cost")
    _assert_against(bd.cpu_mean, "b10b_cpumean")


def test_balance10_two_class_zero_zero_interior_cutoff() -> None:
    """The `(n(pos)==0)&&(n2(pos)==0)` midpoint branch (ComputeCost.m:571-572): b10c is
    crafted so class-1-fired scores (~290/295) and class-2-fired scores (~280/285) leave a
    wide interior span of the hist grid where BOTH cumulative curves are identically zero,
    so argmin(|n-n2|) alone is not enough -- the branch instead takes the midpoint between
    the nearest edges of the two distributions."""
    cost, bd = compute_cost(_rows("b10c_mcr"), 0, 10, CostParams(mode=0, algo=6))
    assert bd.cutoff is not None
    _assert_against(bd.cutoff, "b10c_cutoff")
    _assert_against(bd.error, "b10c_error")
    _assert_against(cost, "b10c_cost")
    _assert_against(bd.nn_cost_lid, "b10c_nnlid")
    _assert_against(bd.cpu_mean, "b10c_cpumean")
    # Mutation guard: the interior-midpoint branch lands near 45 (the midpoint between the
    # two clusters' nearest edges, ~10 and ~80). Reverting to the plain `cutoff = t(pos)`
    # argmin branch would instead land at the FIRST bin of the zero-zero plateau (~10.005) --
    # far outside this band.
    assert 30.0 < bd.cutoff < 60.0


def test_twin_committed_nn_costs() -> None:
    """NNCostSeg / NNCostLID raw pieces off the committed twin (algo 6) fixture."""
    from speech.engine import _nn_cost_lid, _nn_cost_seg

    r = channel_results_from_matrix(_mcr(PHASE4B, "twin_train_MultiConfigResults.bin")).for_config(0)
    _assert_against(_nn_cost_seg(r), "twin_nnseg")
    _assert_against(_nn_cost_lid(r), "twin_nnlid")


# ==== assembly-contract sanity ============================================================


def test_compute_cost_rejects_deferred_balances() -> None:
    for balance in (6, 7, 8, 9):
        with pytest.raises(ValueError, match="deferred to 4d"):
            compute_cost(_rows("cb_mcr"), 0, balance, CostParams())


def test_breakdown_is_the_dataclass() -> None:
    _, bd = compute_cost(_rows("cb_mcr"), 0, 0, CostParams())
    assert isinstance(bd, CostBreakdown)
    assert bd.cutoff is None and bd.nn_cost_lid == 0.0
