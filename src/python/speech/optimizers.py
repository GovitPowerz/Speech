"""Optimizer zoo for NN training and DSP-hyperparameter search.

Ported from legacy MATLAB: SMORMS3.m (wired), Rprop.m (wired), QuantumPSO.m (wired --
bespoke hand-port, do NOT swap for pymoo), cmaes.m (via the `cma` package, alternative
branch). Adam.m/RMSprop.m/sfo.m/pso_Trelea_vectorized.m are EXCLUDED (never wired in
legacy, YAGNI per the Phase 4c spec); BackPropagation.m's inner-loop contract is
implemented by `drivers/train.py::_backprop_inner`. See design spec section 6.

SMORMS3/Rprop are bit-pinned against GNU Octave running the vendored .m sources unchanged
(Phase 4c Task 6). QuantumPSO is bit-pinned against a MODIFIED-COPY of QuantumPSO.m
(`tools/octave_harness/qpso_modified/`) whose rand/randperm/stblrnd are substituted by
reads of a shared committed random table -- the legacy path reseeds from the wall clock,
so no single legacy RUN is reproducible; the table makes the OPERATOR STRUCTURE pinnable.
cmaes_optimize is NOT bit-pinned (a different `cma`-package lineage). Goldens under
`tests/reference_data/phase4c/`.
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


# ============================================================================
# QuantumPSO -- the bespoke global DSP-hyperparameter optimizer
# ============================================================================
# Faithful transcription of legacy/Optimizer_V6.2.2/functions/QuantumPSO.m (845 LOC).
# The legacy path is nondeterministic BY DESIGN (`QuantumPSO.m:91` reseeds `rand` from
# the wall clock). This port routes EVERY stochastic draw through an injected source
# (`TableRng` for bit-pinning against the Octave modified-copy oracle, or a plain numpy
# `Generator` for production), so the trajectory is exactly reproducible given a table.
# The clock reseed (:91) is dropped -- see IMPROVEMENTS.md (the determinism deviation).
#
# The Chambers-Mallows-Stuck Levy sampler (`levy_stable_cms`) is <80 lines, so it lives
# here, not in a separate levy.py module (per the task brief's size heuristic).


class TableRng:
    """A table-backed random source: reads sequential f64s from a committed table and
    exposes the exact draw shapes QuantumPSO uses. BOTH the Octave modified-copy oracle
    (`tools/octave_harness/qpso_modified/`) and this port read the SAME committed table
    (`tests/reference_data/phase4c/qpso_random_table.bin`) through an identical cursor, so
    the trajectory is bit-pinned.

    Column-major fill is the load-bearing detail: MATLAB `rand([m,n])` lays out its stream
    column-major, so `rand(m,n)` here reads `m*n` values and reshapes with `order='F'` --
    element `(i,j)` gets `table[cursor + j*m + i]`. The Octave `tbl_rand` does the same via
    `reshape(vals,[m,n])` (Octave reshape is column-major). numpy's default row-major fill
    would silently transpose every matrix draw; do NOT use it here.
    """

    def __init__(self, table: NDArray[np.float64]) -> None:
        self.table: NDArray[np.float64] = np.asarray(table, dtype=np.float64).reshape(-1)
        self.cursor: int = 0

    def _take(self, count: int) -> NDArray[np.float64]:
        if self.cursor + count > self.table.size:
            raise IndexError(f"TableRng exhausted: need {count} at cursor {self.cursor}, size {self.table.size}")
        vals = self.table[self.cursor : self.cursor + count]
        self.cursor += count
        return vals

    def rand(self, *shape: int) -> float | NDArray[np.float64]:
        """`rand()` -> scalar; `rand(n)` -> length-n vector; `rand(m,n)` -> (m,n) column-major."""
        if len(shape) == 0:
            return float(self._take(1)[0])
        dims = [int(s) for s in shape]
        count = int(np.prod(dims))
        vals = self._take(count)
        if len(dims) == 1:
            return vals.copy()
        return vals.reshape(tuple(dims), order="F").copy()

    def randperm(self, n: int, k: int | None = None) -> NDArray[np.int64]:
        """Table-based Durstenfeld/Fisher-Yates over 1..n (MATLAB 1-based values), consuming
        `n-1` draws: `for i=n..2: j=floor(rand*i)+1; swap p[i],p[j]`. Returns the first `k`.
        Identical algorithm on the Octave side (`tbl_randperm.m`), so the selected indices
        match -- NOT MATLAB's builtin randperm (which is not table-reproducible)."""
        p = list(range(1, n + 1))
        for i in range(n, 1, -1):
            r = float(self._take(1)[0])
            j = int(math.floor(r * i)) + 1
            p[i - 1], p[j - 1] = p[j - 1], p[i - 1]
        return np.array(p if k is None else p[:k], dtype=np.int64)


class _GeneratorSource:
    """Adapts a numpy `Generator` to the TableRng surface for the PRODUCTION (non-pinned)
    path. Fill order is immaterial here (the values are random), so `rand(m,n)` uses numpy's
    native row-major fill; only TableRng needs the column-major discipline."""

    def __init__(self, gen: np.random.Generator) -> None:
        self.gen = gen

    def rand(self, *shape: int) -> float | NDArray[np.float64]:
        if len(shape) == 0:
            return float(self.gen.random())
        dims = tuple(int(s) for s in shape)
        arr = self.gen.random(dims)
        return arr if len(dims) > 1 else arr.reshape(-1)

    def randperm(self, n: int, k: int | None = None) -> NDArray[np.int64]:
        p = self.gen.permutation(n).astype(np.int64) + 1
        return p if k is None else p[:k]


_RandSource = TableRng | _GeneratorSource


def _as_rand_source(rng: np.random.Generator | TableRng) -> _RandSource:
    return rng if isinstance(rng, TableRng) else _GeneratorSource(rng)


# The injected objective (replaces `CostFunction` / the engine feval). Given a (n x D)
# stack of candidate positions (one particle per ROW) and the current eval mode
# (-2/-1 init, i iteration, 0 final), returns (costs[n], pos_out[n x D]). The surrogate
# passes positions through (pos_out == positions); the real engine may project them.
QpsoObjective = Callable[[NDArray[np.float64], int], tuple[NDArray[np.float64], NDArray[np.float64]]]


@dataclass
class QpsoParams:
    """QuantumPSO parameters (the `psoparams` vector of Train_BLSTM.m:902 plus the VR/mv/seed
    args). Names mirror the legacy locals: ps particles, me epochs, ac1/ac2 accel consts,
    iw1/iw2/iwe inertia schedule, ergrd/ergrdep stall gate, errgoal (NaN disables), trelea
    PSO model (3 = Clerc, required for the dead vel3 bank's `chi`), pso_seed (1 = seed the
    first rows), minmax (0 = minimize)."""

    ps: int
    me: int
    ac1: float
    ac2: float
    iw1: float
    iw2: float
    iwe: int
    ergrd: float
    ergrdep: int
    errgoal: float
    trelea: int
    pso_seed: int
    minmax: int
    vr: NDArray[np.float64]  # D x 2 [min, max] per dim
    mv: NDArray[np.float64]  # D-vector max velocity (dead: vels are never applied)
    seed_value: NDArray[np.float64] | None  # PSOseedValue (rows x cols), used iff pso_seed == 1
    backprop_activated: int = 0  # PS.NS.BackPropagationActivated (0 -> the :481 gate short-circuits)


@dataclass
class QpsoResult:
    """QuantumPSO output plus the per-iteration snapshots the goldens pin. `init_*` is the
    post-initialization state (pre-loop, pure arithmetic -> STRICT-comparable); the `*_traj`
    lists hold one entry per epoch (transcendental from epoch 1 via the `log(1/u)` QDPSO
    jump -> canary-comparable). `out` is the legacy `OUT = [gbest'; gbestval]`."""

    gbest: NDArray[np.float64]
    gbestval: float
    pbest: NDArray[np.float64]
    pbestval: NDArray[np.float64]
    tr: NDArray[np.float64]
    te: int
    init_pos: NDArray[np.float64]
    init_pbest: NDArray[np.float64]
    init_pbestval: NDArray[np.float64]
    init_gbest: NDArray[np.float64]
    init_gbestval: float
    pos_traj: list[NDArray[np.float64]]
    pbest_traj: list[NDArray[np.float64]]
    pbestval_traj: list[NDArray[np.float64]]
    gbest_traj: list[NDArray[np.float64]]
    gbestval_traj: list[float]
    out: NDArray[np.float64]


def _normmat_flag1(x: NDArray[np.float64], newminmax: NDArray[np.float64]) -> NDArray[np.float64]:
    """Port of `normmat.m` flag==1 (:64-97): normalize each COLUMN of `x` into its target
    range. `newminmax` is 2 x D (row 0 mins, row 1 maxs). The per-column reference min `a` and
    span `den` come from `x`'s OWN min/max (with the abs-based large/small pick, :68-74), NOT
    from the target range -- so `rand([2ps,D])` in [0,1) is stretched to [colmin_mapped, ...]
    where the column's actual min maps to `newminmax(1,i)` and max to `newminmax(2,i)`."""
    a = x.min(axis=0)  # legacy: a = min(x,[],1)
    b = x.max(axis=0)  # legacy: b = max(x,[],1)
    use_a = np.abs(a) > np.abs(b)  # legacy: if abs(a(i)) > abs(b(i))
    large = np.where(use_a, a, b)
    small = np.where(use_a, b, a)
    den = np.abs(large - small)  # legacy: den = abs(large - small)
    rng = newminmax[1, :] - newminmax[0, :]  # legacy: range = newminmaxA(2,:) - newminmaxA(1,:)
    z = (x - a) / np.where(den == 0.0, 1.0, den)  # guard only shapes the den==0 rows, replaced next
    out = z * rng + newminmax[0, :]
    zero_cols = den == 0.0  # legacy: if den(i)==0 -> out(j,i)=x(j,i)
    if np.any(zero_cols):
        out[:, zero_cols] = x[:, zero_cols]
    return out


def levy_stable_cms(
    src: _RandSource | TableRng,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    n: int,
) -> NDArray[np.float64]:
    """Chambers-Mallows-Stuck alpha-stable sampler -- port of `stblrnd.m`'s general
    `alpha != 1` branch (:75-82) + the `alpha != 1` scale/shift (:95-96). QuantumPSO calls
    `stblrnd(1.3,1,0.5,0,1,D)` (:475), which hits exactly this branch. Consumes TWO
    `rand([1,n])` draws in order: V first (:76), then W (:77). Transcendental
    (tan/atan/sin/cos/log/pow) -> canary-gated off the oracle libm."""
    v_draw = np.asarray(src.rand(1, n)).reshape(-1)  # legacy: rand(sizeOut) for V
    w_draw = np.asarray(src.rand(1, n)).reshape(-1)  # legacy: rand(sizeOut) for W
    v = math.pi / 2 * (2 * v_draw - 1)  # legacy: V = pi/2*(2*rand-1)
    w = -np.log(w_draw)  # legacy: W = -log(rand)
    const = beta * math.tan(math.pi * alpha / 2)  # legacy: const = beta*tan(pi*alpha/2)
    b_off = math.atan(const)  # legacy: B = atan(const)
    s = (1 + const * const) ** (1 / (2 * alpha))  # legacy: S = (1+const^2)^(1/(2*alpha))
    r = s * np.sin(alpha * v + b_off) / np.cos(v) ** (1 / alpha) * (np.cos((1 - alpha) * v - b_off) / w) ** ((1 - alpha) / alpha)
    return np.asarray(gamma * r + delta, dtype=np.float64)  # legacy: r = gamma*r + delta (alpha != 1)


def quantum_pso(
    cost_fn: QpsoObjective,
    params: QpsoParams,
    d: int,
    rng: np.random.Generator | TableRng,
    *,
    backprop_refine: Callable[[NDArray[np.float64], int], tuple[NDArray[np.float64], NDArray[np.float64]]] | None = None,
    _dead_banks: bool = True,
) -> QpsoResult:
    """QuantumPSO (the ranking-operator QDPSO variant) -- a faithful transcription of
    `QuantumPSO.m:252-843`. Every stochastic draw is routed through `rng` (a `TableRng` for
    bit-pinning or a numpy `Generator` for production).

    Structure (legacy line refs inline): init + opposition-based learning (:257-327); the
    four DEAD velocity banks that still DRAW the stream (:375-411); the QDPSO update with the
    ranking-operator roulette (:414-443); DE recombination + a Levy kick (:463-479); the
    backprop-refine gate (:480-485, a HOOK -- `backprop_refine`, off by default so the `&&`
    short-circuits and draws NOTHING); the cost eval + neighbour/pbest/gbest updates
    (:563-686); the stall-termination gate (:749-787); and the final re-generation (:799-843).

    `_dead_banks=False` skips the 8 per-epoch dead-velocity draws -- a test hook proving those
    draws shift the whole downstream stream (the non-vacuity mutation), NOT a production path.
    """
    src = _as_rand_source(rng)
    ps, me, dd = params.ps, params.me, d
    ergrd, ergrdep = params.ergrd, params.ergrdep
    minmax = params.minmax
    vr = np.asarray(params.vr, dtype=np.float64)  # D x 2
    backprop_activated = params.backprop_activated
    # ac1/ac2/iw1/iw2/iwe/errgoal (params.*) are read only inside code the port omits as DEAD:
    # the vel-bank arithmetic (:378-411, never applied) and the errgoal goal block (:766,
    # errgoal=NaN). The dead banks' DRAWS are still consumed below; their maths is not.

    # ---- INITIALIZE (:257) ----
    x0 = np.asarray(src.rand(2 * ps, dd), dtype=np.float64)  # legacy: rand([2*ps,D])
    pos = _normmat_flag1(x0, vr.T)  # legacy: normmat(rand([2*ps,D]), VR', 1)
    if params.pso_seed == 1 and params.seed_value is not None:  # legacy: :259-262
        sv = np.asarray(params.seed_value, dtype=np.float64)
        pos[: sv.shape[0], : sv.shape[1]] = sv
    for _ in range(4):  # legacy: :265-272 four initial velocities (DEAD, never applied)
        src.rand(ps, dd)

    # ---- Opposition-Based Learning (:279-288) ----
    out0, pos_out = cost_fn(pos, -2)  # legacy: feval(functname,pos,PS,-2)
    minbound = pos_out.min(axis=0)  # legacy: min(pos_out,[],1)
    maxbound = pos_out.max(axis=0)  # legacy: max(pos_out,[],1)
    pos_out_opp = (minbound + maxbound)[None, :] - pos_out  # legacy: repmat(minbound+maxbound,...) - pos_out
    out_opp, pos_out_opp = cost_fn(pos_out_opp, -2)
    out = np.concatenate([out0, out_opp])  # legacy: out = [out; out_opp]
    pos_out = np.vstack([pos_out, pos_out_opp])  # legacy: pos_out = [pos_out; pos_out_opp]

    idx1 = int(np.argmin(out))  # legacy: [iterbestval,idx1] = min(out)
    gbestval = float(out[idx1])
    gbest = pos_out[idx1].copy()

    out, pos_out = cost_fn(pos_out, -1)  # legacy: feval(functname,pos_out,PS,-1)
    order = np.lexsort((np.arange(1, out.size + 1), out))  # legacy: sortrows([out (1:numel(out))'])
    sel = order[:ps]  # legacy: tmp_sort(1:ps,2)
    pos = pos_out[sel].copy()  # legacy: pos = pos_out(tmp_sort(1:ps,2),:)
    out = out[sel].copy()  # legacy: out = out(tmp_sort(1:ps,2),:)

    pbest = pos.copy()  # legacy: pbest = pos
    pbestval = out.copy()  # legacy: pbestval = out

    xreal_cols = np.vstack([gbest[None, :], pbest])  # legacy: xreal = [gbest;pbest;GApos]'
    cout = np.concatenate([[gbestval], pbestval])  # legacy: cout = [gbestval;pbestval;outGA]'
    idx1 = int(np.argmin(cout))
    if gbestval >= cout[idx1]:  # legacy: :323-327
        gbestval = float(cout[idx1])
        gbest = xreal_cols[idx1].copy()

    init_pos = pos.copy()
    init_pbest = pbest.copy()
    init_pbestval = pbestval.copy()
    init_gbest = gbest.copy()
    init_gbestval = gbestval

    tr = [math.nan] * (me + 2)  # legacy: tr = ones(1,me)*NaN (1-based; auto-grows)
    cnt2 = 0
    te = 0
    pos_traj: list[NDArray[np.float64]] = []
    pbest_traj: list[NDArray[np.float64]] = []
    pbestval_traj: list[NDArray[np.float64]] = []
    gbest_traj: list[NDArray[np.float64]] = []
    gbestval_traj: list[float] = []

    for i in range(1, me + 1):  # legacy: for i=1:me
        # ---- DEAD velocity banks (:375-411): they DRAW but are never applied (gate :447 unreachable) ----
        if _dead_banks:
            src.rand(ps, dd)  # legacy: :375 rannum1 (bank 1)
            src.rand(ps, dd)  # legacy: :376 rannum2
            src.rand(ps, dd)  # legacy: :382 rannum1 (bank 2)
            src.rand(ps, dd)  # legacy: :383 rannum2
            src.rand(ps, dd)  # legacy: :389 rannum1 (bank 3)
            src.rand(ps, dd)  # legacy: :390 rannum2
            src.rand(ps, dd)  # legacy: :403 rannum1 (bank 4)
            src.rand(ps, dd)  # legacy: :404 rannum2

        # ---- QDPSO update (:414-443) ----
        coef_exp_contr = 0.75 - 0.5 * (i - 1) / (me - 1)  # legacy: :415
        phi = np.asarray(src.rand(ps, dd), dtype=np.float64)  # legacy: phi = rand(ps,D)
        u = np.maximum(1e-32, np.asarray(src.rand(ps, dd), dtype=np.float64))  # legacy: u = max(1e-32,rand(ps,D))
        src.rand(ps, dd)  # legacy: :420 signs = sign(rand(ps,D)-0.5) -- DEAD (overwritten at :442)

        # Ranking operator: roulette-wheel local attractor over pbest ranked by pbestval (:424-440)
        irank = np.argsort(pbestval, kind="stable")  # legacy: [rank,irank] = sortrows(pbestval)
        col2 = np.arange(ps, 0, -1)  # legacy: elem(:,2) = (ps:-1:1)'
        local_attractor = np.zeros((ps, dd))
        for ii in range(1, ps + 1):
            rank = col2 * (col2 > ii)  # legacy: rank = elem(:,2).*(elem(:,2) > ii)
            s = rank.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                sector = rank / s  # legacy: sector = rank/sum(rank) (0/0 -> NaN when s==0, matching Octave)
                sect_cs = np.cumsum(sector)
            wheel = float(src.rand())  # legacy: wheel = rand(1,1)
            tmp = np.where(sect_cs >= wheel)[0]  # legacy: tmp = find(sect_cs >= wheel)
            if tmp.size == 0:
                local_attractor[ps - ii, :] = gbest  # legacy: localAttractor(ps+1-ii,:) = gbest
            else:
                local_attractor[ps - ii, :] = pbest[irank[tmp[0]], :]  # legacy: pbest(irank(tmp(1)),:)

        attractor = phi * pbest + (1 - phi) * local_attractor  # legacy: :441
        signs = np.sign(np.asarray(src.rand(ps, dd), dtype=np.float64) - 0.5)  # legacy: :442
        pos = attractor + coef_exp_contr * signs * np.maximum(1e-5, np.abs(attractor - pos)) * np.log(1.0 / u)  # legacy: :443

        # ---- position update (:447) is DEAD: `i > 20*me/2` (== i > 10*me) is never true; no draws ----

        # ---- DE recombination + Levy kick (:465-479) ----
        xreal_new: list[NDArray[np.float64]] = []
        indexes: list[int] = []
        for ii in range(ps):  # legacy: for ii=1:size(pos,1)
            if float(src.rand()) < 0.2:  # legacy: :466 if (rand < 0.2)
                neighboors = src.randperm(ps, 2)  # legacy: randperm(size(pos,1),2)
                pond = np.asarray(src.rand(1, 3), dtype=np.float64).reshape(-1)  # legacy: rand(1,3)
                pond = pond / pond.sum()  # legacy: ponderations/sum(ponderations)
                n0, n1 = int(neighboors[0]) - 1, int(neighboors[1]) - 1
                new_pos = pond[0] * pos[ii] + pond[1] * (pbest[ii] - pos[ii]) + pond[2] * (pos[n0] - pos[n1])  # legacy: :470
                xreal_new.append(new_pos)
                indexes.append(ii)
            if float(src.rand()) < 0.2:  # legacy: :474 if (rand < 0.2)
                lv = levy_stable_cms(src, 1.3, 1, 0.5, 0, dd)  # legacy: stblrnd(1.3,1,0.5,0,1,D)
                sgn = np.sign(np.asarray(src.rand(1, dd), dtype=np.float64).reshape(-1) - 0.5)  # legacy: sign(rand(1,D)-0.5)
                xreal_new.append(pos[ii] + lv * sgn)  # legacy: :475
                indexes.append(ii)

        # ---- backprop-refine gate (:480-485): HOOK. `&&` short-circuits when activated<=0 (NO draw) ----
        cur_bp = 0
        if backprop_activated > 0:
            cur_bp = 1 if float(src.rand()) < 0.5 else 0  # legacy: :481 rand < 0.5

        # ---- cost eval (:563) ----
        if xreal_new:
            cand = np.vstack(xreal_new)
            stacked = np.vstack([pos, cand])
        else:
            stacked = pos
        out_full, pos_out_full = cost_fn(stacked, i)  # legacy: feval(functname,[pos;xreal_tmp'],PS,i)

        if cur_bp > 0 and backprop_refine is not None:  # legacy: :564-593 (T12 wires the real refinement)
            out_full, pos_out_full = backprop_refine(pos_out_full, i)

        n_cand = len(indexes)
        xreal_tmp = pos_out_full[ps : ps + n_cand]  # legacy: :595 pos_out(size(pos,1)+1:end,:)
        cout_tmp = out_full[ps : ps + n_cand]  # legacy: :596
        pos = pos_out_full[:ps].copy()  # legacy: :597
        out = out_full[:ps].copy()  # legacy: :598

        for k in range(n_cand):  # legacy: :600-606 neighbour update
            idx = indexes[k]
            if cout_tmp[k] < out[idx]:
                pos[idx] = xreal_tmp[k]
                out[idx] = cout_tmp[k]

        # ---- gbest update (:623-630) ----
        xreal_cols = np.vstack([gbest[None, :], pbest, pos])  # legacy: xreal = [gbest;pbest;pos;GApos]'
        cout = np.concatenate([[gbestval], pbestval, out])  # legacy: cout = [gbestval;pbestval;out;outGA]'
        idx1 = int(np.argmin(cout))
        if gbestval >= cout[idx1]:
            gbestval = float(cout[idx1])
            gbest = xreal_cols[idx1].copy()

        tr[i + 1] = gbestval  # legacy: tr(i+1) = gbestval
        te = i  # legacy: te = i

        # ---- pbest update (minmax==0, :652-657) ----
        if minmax == 0:
            tempi = np.where(pbestval >= out)[0]  # legacy: find(pbestval>=out)
            pbestval[tempi] = out[tempi]
            pbest[tempi] = pos[tempi]

        pos_traj.append(pos.copy())
        pbest_traj.append(pbest.copy())
        pbestval_traj.append(pbestval.copy())
        gbest_traj.append(gbest.copy())
        gbestval_traj.append(gbestval)

        # ---- stall termination (:749-763); errgoal=NaN disables the :766 block ----
        tmp1 = abs(tr[i] - gbestval)  # legacy: tmp1 = abs(tr(i) - gbestval); tr(1)=NaN for i=1
        if tmp1 > ergrd:  # legacy: :752
            cnt2 = 0
        elif tmp1 <= ergrd:  # legacy: :754 (NaN fails both -> cnt2 unchanged, matching Octave)
            cnt2 += 1
            if cnt2 >= ergrdep:
                break

    # ---- FINAL GENERATION (:799-843): re-eval [gbest;pbest] with RpropFinal (hook), pick the min ----
    stacked = np.vstack([gbest[None, :], pbest])  # legacy: [gbest;pbest;GApos]
    outfinal, pos_out = cost_fn(stacked, 0)  # legacy: feval(functname,[gbest;pbest;GApos],PS,0)
    gbest = pos_out[0].copy()  # legacy: :812
    gbestval = float(outfinal[0])  # legacy: :813
    pbest = pos_out[1 : 1 + ps].copy()  # legacy: :815
    pbestval = outfinal[1 : 1 + ps].copy()  # legacy: :816
    xreal_cols = np.vstack([gbest[None, :], pbest])  # legacy: xreal = [gbest;pbest;GApos]'
    cout = np.concatenate([[gbestval], pbestval])  # legacy: cout = [gbestval;pbestval;outGA]'
    idx1 = int(np.argmin(cout))
    if gbestval >= cout[idx1]:  # legacy: :832-835
        gbestval = float(cout[idx1])
        gbest = xreal_cols[idx1].copy()
    out_final = np.concatenate([gbest, [gbestval]])  # legacy: OUT = [gbest';gbestval]

    return QpsoResult(
        gbest=gbest,
        gbestval=gbestval,
        pbest=pbest,
        pbestval=pbestval,
        tr=np.array([v for v in tr if not math.isnan(v)], dtype=np.float64),
        te=te,
        init_pos=init_pos,
        init_pbest=init_pbest,
        init_pbestval=init_pbestval,
        init_gbest=init_gbest,
        init_gbestval=init_gbestval,
        pos_traj=pos_traj,
        pbest_traj=pbest_traj,
        pbestval_traj=pbestval_traj,
        gbest_traj=gbest_traj,
        gbestval_traj=gbestval_traj,
        out=out_final,
    )


@dataclass
class CmaesResult:
    """Best-so-far solution from `cmaes_optimize` (a thin `cma`-package wrap, NOT bit-pinned)."""

    x: NDArray[np.float64]
    fun: float
    evals: int
    stop: dict[str, object]


def cmaes_optimize(
    cost_fn: QpsoObjective,
    x0: NDArray[np.float64],
    sigma0: float,
    opts: dict[str, object] | None = None,
) -> CmaesResult:
    """CMA-ES via the `cma` pip package -- the alternative outer-optimizer branch of
    Train_BLSTM.m:1147 (`[X,F,E,STOP,OUT] = cmaes(functname, adim/2*ones(dims,1), adim/3,
    OPTS, PS, 1)` with `OPTS.EvalParallel='yes'`).

    NOT bit-pinned, and a DIFFERENT lineage than the vendored `cmaes.m`: the legacy `cmaes.m`
    is Hansen's older MATLAB reference implementation; the `cma` package is Hansen's maintained
    Python port. Their sampling/recombination RNG and default stop criteria differ, so this
    wrap targets the same INVOCATION SHAPE and search behaviour, not bit equality. Kept thin;
    validated by a smoke test, not a golden.

    `EvalParallel='yes'` is honoured by handing the whole `ask()` population to `cost_fn` as a
    (lambda x D) matrix per generation (one candidate per row)."""
    import cma  # type: ignore[import-untyped]  # lazy: core dep, but only this branch needs it

    x0_list = np.asarray(x0, dtype=np.float64).reshape(-1).tolist()
    es = cma.CMAEvolutionStrategy(x0_list, float(sigma0), opts or {})
    while not es.stop():
        solutions = es.ask()
        costs, _ = cost_fn(np.asarray(solutions, dtype=np.float64), 0)  # EvalParallel: whole population
        es.tell(solutions, [float(c) for c in np.asarray(costs).reshape(-1)])
    res = es.result
    return CmaesResult(
        x=np.asarray(res.xbest, dtype=np.float64),
        fun=float(res.fbest),
        evals=int(res.evaluations),
        stop=dict(es.stop()),
    )
