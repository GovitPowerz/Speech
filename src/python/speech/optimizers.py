"""Optimizer zoo for NN training and DSP-hyperparameter search.

Ported from legacy MATLAB: SMORMS3.m (wired), Rprop.m, Adam.m, RMSprop.m,
QuantumPSO.m (bespoke - hand-port, do NOT swap for pymoo), cmaes.m, sfo.m,
pso_Trelea_vectorized.m, BackPropagation.m. See design spec section 6.

Bit-pinned against GNU Octave running the vendored .m sources (Phase 4c Task 6;
`tools/octave_harness/`, goldens under `tests/reference_data/phase4c/`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# The injected objective's contract, mirroring SMORMS3.m's f_df (:306-321): given the
# parameters in ORIGINAL (cell/list) format and the current 1-based eval_count, return
# (cost, gradient-in-original-format, theta_out-in-original-format). theta_out is the
# ROUND-TRIP subtlety: the update adds dtheta to this RETURNED theta, not the input.
Smorms3Objective = Callable[
    [list[NDArray[np.float64]], int],
    tuple[float, list[NDArray[np.float64]], list[NDArray[np.float64]]],
]


class Smorms3:
    """SMORMS3 gradient descent (the wired NN optimizer).

    Faithful port of the vendored `SMORMS3.m` classdef (mislabeled "SFO" in its header
    -- see IMPROVEMENTS): the ctor (:151-187), `f_df_wrapper` (:306-321), the update
    `optimization_step` (:324-363), and `optimize` (:190-212). State is exposed flat
    (`mms`/`step_rate`/`delta`/`lrate`) for the goldens.
    """

    def __init__(self, f_df: Smorms3Objective, theta0: list[NDArray[np.float64]]) -> None:
        # legacy: SMORMS3.m:173-186
        self.f_df = f_df
        self.theta_original: list[NDArray[np.float64]] = [np.asarray(a, dtype=np.float64) for a in theta0]
        self.theta: NDArray[np.float64] = self._flatten(self.theta_original)
        self.theta_prior_step: NDArray[np.float64] = self.theta.copy()
        self.M: int = int(self.theta.shape[0])
        self.mms: NDArray[np.float64] = np.zeros(self.M)
        self.step_rate: NDArray[np.float64] = np.zeros(self.M)
        self.delta: NDArray[np.float64] = np.ones(self.M)
        self.lrate_max: float = 1e-1  # legacy: SMORMS3.m:72
        self.lrate: float = 1e-9  # legacy: SMORMS3.m:73
        self.eps: float = 1e-16  # legacy: SMORMS3.m:79
        self.eval_count: int = 1  # legacy: SMORMS3.m:119
        self.hist_f_flat: list[float] = []

    @staticmethod
    def _flatten(cells: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        # legacy: SMORMS3.m:261-283 -- column-major (Fortran) concatenation of each cell.
        return np.concatenate([np.asarray(c, dtype=np.float64).flatten(order="F") for c in cells])

    def _unflatten(self, flat: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        # legacy: SMORMS3.m:286-303 -- column-major reshape back into the original shapes.
        out: list[NDArray[np.float64]] = []
        i = 0
        for cell in self.theta_original:
            n = cell.size
            out.append(flat[i : i + n].reshape(cell.shape, order="F"))
            i += n
        return out

    def _f_df_wrapper(self, theta_in: NDArray[np.float64]) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        # legacy: SMORMS3.m:306-321
        theta_local = self._unflatten(theta_in)
        f, df_full, theta_local_out = self.f_df(theta_local, self.eval_count)
        df_flat = self._flatten(df_full)
        theta_out_flat = self._flatten(theta_local_out)
        self.hist_f_flat.append(f)
        self.eval_count += 1
        return f, df_flat, theta_out_flat

    def optimization_step(self) -> None:
        # legacy: SMORMS3.m:324-363 -- the exact update rule.
        _f, grad, theta_out = self._f_df_wrapper(self.theta)
        r = 1.0 / (self.delta + 1.0)
        self.step_rate = (1 - r) * self.step_rate + r * grad
        self.mms = (1 - r) * self.mms + r * grad * grad
        dtheta = -grad * np.minimum(self.lrate, self.step_rate * self.step_rate / (self.mms + self.eps)) / (np.sqrt(self.mms) + self.eps)
        self.delta = 1 + self.delta * (1 - self.step_rate * self.step_rate / (self.mms + self.eps))
        self.lrate = min(self.lrate * 10, self.lrate_max)  # legacy: SMORMS3.m:347
        self.theta_prior_step = self.theta
        self.theta = theta_out + dtheta  # legacy: SMORMS3.m:355 -- theta_out FROM f_df, not the input

    def optimize(self, num_passes: int) -> list[NDArray[np.float64]]:
        # legacy: SMORMS3.m:190-212 -- the loop range is fixed at entry (MATLAB semantics).
        num_steps = self.eval_count - 1 + math.ceil(num_passes)
        for _ in range(self.eval_count, num_steps + 1):
            self.optimization_step()
        return self._unflatten(self.theta)


@dataclass
class RpropState:
    """iRPROP- per-weight state carried call-to-call (Rprop.m's out-args fed back as
    in-args). `previous_derivatives` is None before the first step (the init branch,
    Rprop.m:10); `prev_cost` is last step's cost (the backtrack gate at :24)."""

    weights: NDArray[np.float64]
    delta: NDArray[np.float64]
    deltaweight: NDArray[np.float64]
    previous_derivatives: NDArray[np.float64] | None
    prev_cost: float


def rprop_step(state: RpropState, grad: NDArray[np.float64], cost: float) -> RpropState:
    """One iRPROP- step -- a faithful port of the vendored `Rprop.m` (38 LOC).

    etap 1.2 / etam 0.5 / delta0 0.01 (the `:5` rand is DEAD, overwritten at `:6` -- see
    IMPROVEMENTS) / deltamin 1e-9 / deltamax 1.0. Sign-step with a cost-gated backtrack:
    on a derivative sign flip the delta shrinks, the weight is rolled back iff the cost
    rose (prev_cost < cost), and the derivative is zeroed (so next step hits the ==0
    branch). Pure arithmetic -- no libm.
    """
    # legacy: Rprop.m:3-8
    etap = 1.2
    etam = 0.5
    delta0 = 0.01
    deltamin = 1e-9
    deltamax = 1.0

    weights = np.array(state.weights, dtype=np.float64, copy=True)
    derivatives = np.array(grad, dtype=np.float64, copy=True)
    prev = state.previous_derivatives

    if prev is None or prev.size == 0:
        # legacy: Rprop.m:10-15 (init branch)
        delta = delta0 * np.ones_like(weights)
        deltaweight = -np.sign(derivatives) * delta
        weights = weights + deltaweight
    else:
        # legacy: Rprop.m:16-33
        delta = np.array(state.delta, dtype=np.float64, copy=True)
        deltaweight = np.array(state.deltaweight, dtype=np.float64, copy=True)
        for nn in range(weights.size):
            product = derivatives[nn] * prev[nn]
            if product > 0:
                delta[nn] = min(delta[nn] * etap, deltamax)
                deltaweight[nn] = -np.sign(derivatives[nn]) * delta[nn]
                weights[nn] = weights[nn] + deltaweight[nn]
            elif product < 0:
                delta[nn] = max(delta[nn] * etam, deltamin)
                if state.prev_cost < cost:
                    weights[nn] = weights[nn] - deltaweight[nn]
                derivatives[nn] = 0.0
            else:
                deltaweight[nn] = -np.sign(derivatives[nn]) * delta[nn]
                weights[nn] = weights[nn] + deltaweight[nn]

    return RpropState(
        weights=weights,
        delta=delta,
        deltaweight=deltaweight,
        previous_derivatives=derivatives,
        prev_cost=cost,
    )
