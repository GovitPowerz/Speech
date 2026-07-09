"""Phase 4c Task 6: SMORMS3 + Rprop bit-pins against the Octave-executed vendored .m.

The goldens (`tests/reference_data/phase4c/`) are per-step optimizer state trajectories
dumped by GNU Octave running the REAL `legacy/Optimizer_V6.2.2/functions/{SMORMS3,Rprop}.m`
(see `scripts/extract_phase4c_fixtures.py`). These tests replay the identical scripted
gradient/derivative sequences through the Python port and compare per-step state.

Per-field libm status (the CROSS-LIBM rule, `tests/_libm_gate.py`):
  - SMORMS3 `mms`/`step_rate`/`delta`/`lrate`: PURE arithmetic (EMA + mul + add + min/max),
    STRICT bits on every platform.
  - SMORMS3 `theta`/`dtheta`: traverse `sqrt(MMS)` (SMORMS3.m:335), so compared with the
    canary-gated `assert_f64_close` (exact on the oracle env; <=4 ULP off it).
  - Rprop: PURE arithmetic (sign, min/max, mul, add), STRICT bits everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from speech.optimizers import RpropState, Smorms3, rprop_step
from speech.weight_bridge import read_bin

from tests._libm_gate import assert_f64_close

PHASE4C = Path(__file__).resolve().parent / "reference_data" / "phase4c"


def _load(name: str) -> NDArray[np.float64]:
    """Load a phase4c `.bin` into an (rows, cols) column-major matrix."""
    rows, cols, flat = read_bin(PHASE4C / f"{name}.bin")
    return np.asarray(flat, dtype=np.float64).reshape((rows, cols), order="F")


def _assert_bits(got: NDArray[np.float64], want: NDArray[np.float64], label: str) -> None:
    """STRICT bit equality (pure-arithmetic fields only)."""
    g = np.ascontiguousarray(got, dtype=np.float64).reshape(-1)
    w = np.ascontiguousarray(want, dtype=np.float64).reshape(-1)
    assert g.shape == w.shape, f"{label}: shape {g.shape} != {w.shape}"
    assert g.view(np.uint64).tobytes() == w.view(np.uint64).tobytes(), f"{label}: bit mismatch got={g!r} want={w!r}"


def _cells_from_flat(flat: NDArray[np.float64], shapes: list[tuple[int, int]]) -> list[NDArray[np.float64]]:
    """Column-major (Fortran) unflatten into cells, mirroring SMORMS3.m:286-303."""
    cells = []
    i = 0
    for r, c in shapes:
        n = r * c
        cells.append(flat[i : i + n].reshape((r, c), order="F"))
        i += n
    return cells


def _build_f_df(
    grad_script: NDArray[np.float64], offset: NDArray[np.float64], shapes: list[tuple[int, int]]
) -> Callable[[list[NDArray[np.float64]], int], tuple[float, list[NDArray[np.float64]], list[NDArray[np.float64]]]]:
    """The Python analog of `tools/octave_harness/smorms3_f_df.m`: scripted gradient
    (grad_script column `eval_count`) reshaped into theta's cell structure, and
    theta_out = theta + offset (the round-trip)."""

    def f_df(theta_list: list[NDArray[np.float64]], eval_count: int) -> tuple[float, list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        g = grad_script[:, eval_count - 1]
        grads = _cells_from_flat(g, shapes)
        off_cells = _cells_from_flat(offset, shapes)
        outs = [th + off for th, off in zip(theta_list, off_cells, strict=True)]
        return float(np.sum(g * g)), grads, outs

    return f_df


def _main_run_setup() -> tuple[
    Callable[[list[NDArray[np.float64]], int], tuple[float, list[NDArray[np.float64]], list[NDArray[np.float64]]]],
    list[NDArray[np.float64]],
    list[tuple[int, int]],
    int,
]:
    grad_script = _load("smorms3_grad_script")  # (7, 12)
    theta0_flat = _load("smorms3_theta0_flat").reshape(-1)  # (7,)
    shapes = [_load("smorms3_theta_final_cell0").shape, _load("smorms3_theta_final_cell1").shape]  # (3,1), (2,2)
    theta0 = _cells_from_flat(theta0_flat, shapes)
    offset = np.zeros(7)  # pass-through
    return _build_f_df(grad_script, offset, shapes), theta0, shapes, grad_script.shape[1]


def test_smorms3_trajectory_bit_exact() -> None:
    """Per-step mms/step_rate/delta STRICT bits + lrate STRICT + theta canary-gated,
    against the real SMORMS3.m classdef driven step-by-step over the scripted gradients."""
    f_df, theta0, _shapes, num_steps = _main_run_setup()
    mms_g = _load("smorms3_mms_traj")
    sr_g = _load("smorms3_steprate_traj")
    delta_g = _load("smorms3_delta_traj")
    lrate_g = _load("smorms3_lrate_traj")
    theta_g = _load("smorms3_theta_traj")

    opt = Smorms3(f_df, theta0)
    for s in range(1, num_steps + 1):
        opt.optimization_step()
        col = s - 1
        _assert_bits(opt.mms, mms_g[:, col], f"mms step {s}")
        _assert_bits(opt.step_rate, sr_g[:, col], f"step_rate step {s}")
        _assert_bits(opt.delta, delta_g[:, col], f"delta step {s}")
        assert float(opt.lrate).hex() == float(lrate_g[0, col]).hex(), f"lrate step {s}"
        for d in range(theta_g.shape[0]):
            assert_f64_close(float(opt.theta[d]), float(theta_g[d, col]), f"theta[{d}] step {s}")


def _eps_run_setup() -> tuple[
    Callable[[list[NDArray[np.float64]], int], tuple[float, list[NDArray[np.float64]], list[NDArray[np.float64]]]],
    list[NDArray[np.float64]],
    list[tuple[int, int]],
    int,
]:
    grad_script = _load("smorms3_eps_grad_script")  # (1, 3)
    theta0_flat = _load("smorms3_eps_theta0_flat").reshape(-1)  # (1,)
    shapes = [(1, 1)]
    theta0 = _cells_from_flat(theta0_flat, shapes)
    offset = np.zeros(1)  # pass-through
    return _build_f_df(grad_script, offset, shapes), theta0, shapes, grad_script.shape[1]


def test_smorms3_eps_guard_active() -> None:
    """Near-zero-gradient regime (M=1, 3 steps, grad=1e-8 constant) that makes BOTH
    eps=1e-16 guards (SMORMS3.m:335 `sqrt(MMS)+eps` in the dtheta denominator, and the
    `MMS+eps` min-cap denominator shared by the dtheta cap and the delta update)
    genuinely active. Step-1 MMS = 0.5*grad^2 ~= 5e-17 -- the SAME ORDER as eps
    (MMS+eps ~= 3*MMS, a measured effect, not rounding noise). This is the counterpart
    to `test_smorms3_trajectory_bit_exact`'s main run, where MMS >> eps everywhere and
    an eps-placement mutation is a no-op (see IMPROVEMENTS.md phase4c eps-guard entry).

    delta_traj is PURE arithmetic (the MMS+eps term feeds it directly) -> STRICT bits;
    theta_traj is sqrt-bearing -> canary-gated.

    Mutation (local, not committed -- verified by hand, see the report): dropping
    `+eps` from `sqrt(MMS)+eps` (the dtheta denominator) changes `theta` at step 1 here
    while leaving the main/round-trip goldens bit-identical (confirming that guard is a
    no-op there, per IMPROVEMENTS). Dropping `+eps` from `MMS+eps` at its delta-update
    site changes `delta` at step 1 by ~22% (1.5 vs 1.8333) -- a real effect, not ULP
    noise. One nuance found and reported honestly: the OTHER `MMS+eps` occurrence (the
    dtheta min-cap ratio inside `min(lrate, stepRate^2/(MMS+eps))`) stays a no-op even
    on this fixture, because `lrate` is still 1e-9 (pre-warmup) at step 1 and saturates
    the `min()` regardless of the ratio's value -- that occurrence remains untested by
    any fixture in this suite.
    """
    f_df, theta0, _shapes, num_steps = _eps_run_setup()
    mms_g = _load("smorms3_eps_mms_traj")
    sr_g = _load("smorms3_eps_steprate_traj")
    delta_g = _load("smorms3_eps_delta_traj")
    lrate_g = _load("smorms3_eps_lrate_traj")
    theta_g = _load("smorms3_eps_theta_traj")

    # Non-vacuity: step-1 MMS is genuinely the same order of magnitude as eps (not
    # rounding noise) -- else this fixture wouldn't actually exercise the guards.
    eps = 1e-16
    step1_mms = float(mms_g[0, 0])
    ratio = step1_mms / eps
    assert 1e-2 <= ratio <= 1e2, f"eps-guard fixture vacuous: step1 mms={step1_mms} eps={eps} ratio={ratio}"

    opt = Smorms3(f_df, theta0)
    for s in range(1, num_steps + 1):
        opt.optimization_step()
        col = s - 1
        _assert_bits(opt.mms, mms_g[:, col], f"eps mms step {s}")
        _assert_bits(opt.step_rate, sr_g[:, col], f"eps step_rate step {s}")
        _assert_bits(opt.delta, delta_g[:, col], f"eps delta step {s}")
        assert float(opt.lrate).hex() == float(lrate_g[0, col]).hex(), f"eps lrate step {s}"
        for d in range(theta_g.shape[0]):
            assert_f64_close(float(opt.theta[d]), float(theta_g[d, col]), f"eps theta[{d}] step {s}")


def test_smorms3_lrate_warmup_x10_capped() -> None:
    """The lrate x10 geometric warmup (1e-9 -> 1e-1 by step 8) capped at 1e-1 (SMORMS3.m:347).
    STRICT bits: pure `min(lrate*10, lrate_max)`. Mutation guard: an x2 warmup, or moving/
    dropping the cap, breaks the plateau."""
    f_df, theta0, _shapes, num_steps = _main_run_setup()
    lrate_g = _load("smorms3_lrate_traj").reshape(-1)
    opt = Smorms3(f_df, theta0)
    got = []
    for _ in range(num_steps):
        opt.optimization_step()
        got.append(float(opt.lrate))
    for s in range(num_steps):
        assert got[s].hex() == float(lrate_g[s]).hex(), f"lrate step {s + 1}"
    # The x10 factor is pinned exactly by the iterated reference (a *2 mutation diverges):
    exp: list[float] = []
    lr = 1e-9
    for _ in range(num_steps):
        lr = min(lr * 10.0, 1e-1)
        exp.append(lr)
    assert [v.hex() for v in got] == [v.hex() for v in exp], "lrate != iterated min(lrate*10, 1e-1)"
    # Structural non-vacuity: strictly-increasing warmup through step 8, then a hard 1e-1 plateau.
    assert got[0] == 1e-8 and got[7] == 1e-1
    assert all(got[i] < got[i + 1] for i in range(7)), "x10 warmup must strictly increase"
    assert all(v == 1e-1 for v in got[7:]), "lrate must plateau at the 1e-1 cap from step 8"


def test_smorms3_theta_out_roundtrip_pinned() -> None:
    """SMORMS3.m:355 updates `theta = theta_out + dtheta`, where theta_out comes FROM f_df.
    The round-trip fixture's f_df returns theta_out = theta + rt_offset (nonzero), so a port
    that added dtheta to the INPUT theta (dropping the round-trip) would diverge by the
    accumulated offset. Canary-gated (theta is sqrt-bearing) but the offset is O(0.1) >> ULP."""
    grad_script = _load("smorms3_rt_grad_script")  # (3, 4)
    theta0_flat = _load("smorms3_rt_theta0_flat").reshape(-1)  # (3,)
    offset = _load("smorms3_rt_offset").reshape(-1)  # (3,)
    theta_g = _load("smorms3_rt_theta_traj")  # (3, 4)
    assert np.any(offset != 0.0), "round-trip offset must be nonzero (else the pin is vacuous)"

    shapes = [(3, 1)]
    theta0 = _cells_from_flat(theta0_flat, shapes)
    f_df = _build_f_df(grad_script, offset, shapes)
    opt = Smorms3(f_df, theta0)
    num_steps = grad_script.shape[1]
    for s in range(1, num_steps + 1):
        opt.optimization_step()
        for d in range(theta_g.shape[0]):
            assert_f64_close(float(opt.theta[d]), float(theta_g[d, s - 1]), f"rt theta[{d}] step {s}")


def test_smorms3_optimize_returns_original_cell_format() -> None:
    """optimize(num_passes) returns the original (cell) format; the returned cells pin the
    column-major flat->original reconstruction (SMORMS3.m:286-303) against Octave's dump."""
    f_df, theta0, shapes, num_steps = _main_run_setup()
    opt = Smorms3(f_df, theta0)
    result = opt.optimize(num_steps)
    assert [r.shape for r in result] == shapes
    cell0_g = _load("smorms3_theta_final_cell0")
    cell1_g = _load("smorms3_theta_final_cell1")
    for got, want, name in ((result[0], cell0_g, "cell0"), (result[1], cell1_g, "cell1")):
        for idx in range(got.size):
            assert_f64_close(float(got.flatten(order="F")[idx]), float(want.flatten(order="F")[idx]), f"{name}[{idx}]")


def test_rprop_sequence_bit_exact() -> None:
    """Rprop.m per-step weights/delta/deltaweight/derivatives over the crafted 5x6 sequence.
    STRICT bits everywhere (pure arithmetic). Exercises grow (delta*=etap), shrink+backtrack
    (delta*=etam, prev_cost<cost -> weight backtrack, derivative zeroing), and the plain step."""
    deriv = _load("rprop_deriv_script")  # (5, 6)
    cost = _load("rprop_cost_script").reshape(-1)  # (6,)
    w0 = _load("rprop_weights0").reshape(-1)  # (5,)
    w_g = _load("rprop_weights_traj")
    delta_g = _load("rprop_delta_traj")
    dw_g = _load("rprop_deltaweight_traj")
    dout_g = _load("rprop_derivout_traj")

    n = w0.shape[0]
    state = RpropState(
        weights=w0.copy(),
        delta=np.zeros(n),
        deltaweight=np.zeros(n),
        previous_derivatives=None,
        prev_cost=0.0,
    )
    for s in range(deriv.shape[1]):
        state = rprop_step(state, deriv[:, s], float(cost[s]))
        _assert_bits(state.weights, w_g[:, s], f"weights step {s + 1}")
        _assert_bits(state.delta, delta_g[:, s], f"delta step {s + 1}")
        _assert_bits(state.deltaweight, dw_g[:, s], f"deltaweight step {s + 1}")
        assert state.previous_derivatives is not None
        _assert_bits(state.previous_derivatives, dout_g[:, s], f"derivout step {s + 1}")
    # Non-vacuity: the sign-flip <0 branch zeroed at least one nonzero input derivative.
    assert np.any((dout_g == 0.0) & (deriv != 0.0)), "no derivative-zeroing observed -- <0 branch never fired"
