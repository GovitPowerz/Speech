"""Engine driver: invoke the Rust engine, assemble cost + gradients.

Ported from legacy MATLAB `Optimizer_V6.2.2/functions/ComputeCost.m` (782 LOC) +
`ComputeGradient.m` (336) + `RunFsp.py` (the subprocess seam -> in-process PyO3 call to
`speech_rs.Engine`). See design spec section 6.

Two layers:

  * The PURE cost assembly (`compute_cost` + the `aggregate_workers` / `average_derivs` /
    `pool_input_stats` / `l2_penalty` helpers): given a MultiConfigResults matrix (and, for
    the worker/deriv/stats helpers, the per-worker pieces), reproduce ComputeCost.m's
    `sortrows([1 2 3])` aggregation (:374), `MultiDeriv./max(1,count)` averaging (:357),
    pooled input-stats recombination (:285-300), L2 penalties (:362/:554), and the balance
    laws 0/3/4/5/10 (:428-652). Octave-pinned bit-for-bit (`tests/test_phase4c_engine_cost.py`).
  * The SEAM (`forward_backward`): ComputeGradient's contract over `speech_rs.Engine` --
    set weights -> run -> read results+derivs -> `f = NNCostSeg (+ NNCostLID)` (:110/:114)
    with the count-normalized (`+ L2` for the LID net) gradient.

MultiConfigResults column schema. The Rust `results_matrix()` row is
`[file+1, conf+1, chan+1, res...]`, so the 3 leading id columns (0-based 0/1/2) are the
legacy `MultiConfigResults(:,1:3)` and `Error_vad = MCR(MCR(:,2)==ii, 4:end)` becomes the
0-based slice `results[results[:,1]==config_idx][:, 3:]`. Within Error_vad, the legacy
1-based columns map to 0-based as:

  legacy Error_vad(:,k)  0-based Error_vad[:, k-1]  meaning
  --------------------------------------------------------------------------------
  1                       0                          Pfa
  2                       1                          Pmiss
  3                       2                          100 - success
  4                       3                          cpu (the wall-clock timing column)
  5                       4                          seg-cost numerator
  15                      14                         LID-cost numerator
  16                      15                         LID flag (recomputed in balance 10)
  17:end-2                16:-2                       per-class LID scores (>150 -> +200 in-band)
  end-1                   -2                         LID-cost denominator
  end                     -1                         seg-cost denominator

Legacy quirks reproduced on purpose are tracked in IMPROVEMENTS.md; the load-bearing ones
carry a `# legacy:` provenance comment at their site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import speech_rs

F64 = np.float64


@dataclass(frozen=True)
class CostParams:
    """The non-results inputs ComputeCost.m's balance assembly reads.

    `mode` gates the cpu-median (`mode >= 0`, :433) and the Rprop coefficient
    (`mode == 4` -> `coeffRprop = 0`, :445/:453/:461); `balance_backprop` is
    `PS.NS.BalanceBackProp` (the Pfa/Pmiss weighting in balance 3/4, :450/:458);
    `algo` is `PS.VP.algo` (the `(algo == 1)*cpu` term in balance 3/4).
    """

    mode: int = 0
    balance_backprop: float = 0.0
    algo: int = 3


@dataclass
class CostBreakdown:
    """Reachable intermediates of the cost assembly (the Octave-pinned goldens)."""

    nn_cost_seg: float
    nn_cost_lid: float
    cpu_mean: float
    error: NDArray[np.float64]
    cutoff: float | None = None


# ---- Pure results-derived pieces (shared by compute_cost + forward_backward) ------------


def _error_vad(results: NDArray[np.float64], config_idx: int) -> NDArray[np.float64]:
    """`Error_vad = MultiConfigResults(MultiConfigResults(:,2)==ii, 4:end)` (:429).

    The conf column is 0-based index 1 (legacy col 2); the res block starts at 0-based
    index 3 (legacy col 4)."""
    sel = results[results[:, 1] == config_idx]
    return np.asarray(sel[:, 3:], dtype=F64)


def _nn_cost_seg(error_vad: NDArray[np.float64]) -> float:
    # legacy: ComputeCost.m:430 NNcost = sum(Error_vad(:,5))/max(1e-6,sum(Error_vad(:,end)));
    return float(np.sum(error_vad[:, 4]) / max(1e-6, float(np.sum(error_vad[:, -1]))))


def _nn_cost_lid(error_vad: NDArray[np.float64]) -> float:
    # legacy: ComputeCost.m:552 NNCostLID = sum(Error_vad(:,15))/max(1e-6,sum(Error_vad(:,end-1)));
    return float(np.sum(error_vad[:, 14]) / max(1e-6, float(np.sum(error_vad[:, -2]))))


# ---- Aggregation / averaging / pooling / L2 ---------------------------------------------


def aggregate_workers(worker_results: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Concatenate the per-worker MultiConfigResults and `sortrows(...,[1 2 3])` (:219/:374).

    MATLAB `sortrows` is a STABLE ascending lexicographic sort on the given key columns;
    `np.lexsort` (also stable) with the keys in reverse priority order is the equivalent.
    Empty workers contribute nothing (legacy `[X;[]] == X`)."""
    non_empty = [np.atleast_2d(np.asarray(w, dtype=F64)) for w in worker_results if np.asarray(w).size > 0]
    if not non_empty:
        return np.zeros((0, 0), dtype=F64)
    stacked = np.vstack(non_empty)
    order = np.lexsort((stacked[:, 2], stacked[:, 1], stacked[:, 0]))
    return np.ascontiguousarray(stacked[order])


def average_derivs(derivs: NDArray[np.float64]) -> NDArray[np.float64]:
    """`MultiDeriv(:,ii) = MultiDeriv(:,ii)./max(1,MultiDerivCount(:,ii))` (:357).

    `derivs` is the engine's per-network `Nx2` matrix (col 0 summed deriv, col 1 count);
    a zero count divides by 1 (never by 0)."""
    d = np.asarray(derivs, dtype=F64)
    return np.asarray(d[:, 0] / np.maximum(1.0, d[:, 1]), dtype=F64)


def pool_input_stats(
    stats: list[tuple[int, NDArray[np.float64], NDArray[np.float64]]],
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """Pooled per-dim mean/std across workers (:285-300), same math as the Rust
    `io/stats.rs::InputStatistics::update` (the streaming pooled merge, re-squaring the
    stored std). `stats` is a per-worker list of `(nb, mean_vec, std_vec)` for one config.

    Transcribes the exact ascending-worker accumulation order so the STRICT pieces (nb,
    mean) are bit-exact; the std carries a `sqrt` (canary-gated off the oracle env)."""
    dim = int(np.asarray(stats[0][1]).shape[0])
    total_nb = 0
    mean = np.zeros(dim, dtype=F64)
    for nb, mean_w, _ in stats:
        total_nb += int(nb)  # legacy: :289
        mean += int(nb) * np.asarray(mean_w, dtype=F64)  # legacy: :290
    mean = mean / total_nb  # legacy: :292
    var_acc = np.zeros(dim, dtype=F64)
    for nb, mean_w, std_w in stats:
        # legacy: :297 nb*(std^2 + (pooled_mean - mean_w)^2)
        var_acc += int(nb) * (np.asarray(std_w, dtype=F64) ** 2 + (mean - np.asarray(mean_w, dtype=F64)) ** 2)
    std = np.sqrt(var_acc / total_nb)  # legacy: :299
    return total_nb, mean, std


def l2_penalty(
    weights: NDArray[np.float64],
    is_bias: NDArray[np.float64],
    l2: float,
) -> tuple[float, NDArray[np.float64]]:
    """L2 regularization on the non-bias weights: the cost term (:554) and the gradient
    term (:362).

    legacy: ComputeCost.m:554 `+= L2_regul*sum((w.*(1-isBias)).^2)/2`
    legacy: ComputeCost.m:362 `+= L2_regul*(w.*(1-isBias))`"""
    w = np.asarray(weights, dtype=F64)
    keep = 1.0 - np.asarray(is_bias, dtype=F64)
    masked = w * keep
    cost = float(l2 * np.sum(masked**2) / 2.0)
    grad = np.asarray(l2 * masked, dtype=F64)
    return cost, grad


# ---- The balance-law cost assembly ------------------------------------------------------


def _matlab_hist(y: NDArray[np.float64], centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """MATLAB/Octave `hist(y, centers)`: bins centered at `centers`, boundaries at the
    midpoints, right-closed (a value on a boundary falls in the LOWER bin), the first/last
    bins catching the -inf/+inf tails. Reproduced because balance 10's cutoff search
    (:561-576) depends on the exact binning."""
    if centers.shape[0] < 2:
        return np.array([float(y.shape[0])], dtype=F64)
    cutoffs = (centers[:-1] + centers[1:]) / 2.0
    counts = np.zeros(centers.shape[0], dtype=F64)
    if y.shape[0] == 0:
        return counts
    idx = np.searchsorted(cutoffs, y, side="left")
    for i in idx:
        counts[int(i)] += 1.0
    return counts


def _balance10_cutoff(error_vad: NDArray[np.float64]) -> float:
    """The 2-class cutoff search (:561-576). `Error_vad(:,17)`/`(:,18)` are the two
    per-class in-band score columns (0-based 16/17)."""
    c1 = error_vad[:, 16]
    c2 = error_vad[:, 17]
    centers = np.arange(10000, dtype=F64) * 0.01 + 0.005  # (0:0.01:99.99)+0.005
    tmp = c2[c2 > 150] - 200
    n = _matlab_hist(tmp, centers)
    n = np.cumsum(100 * n / np.sum(n))
    tmp2 = 300 - c1[c1 > 150]
    n2 = _matlab_hist(tmp2, centers)
    n2 = np.cumsum(100 * n2[::-1] / np.sum(n2))[::-1]
    pos = int(np.argmin(np.abs(n - n2)))
    if n[pos] == 0 and n2[pos] == 0:
        first_n = int(np.nonzero(n)[0][0])
        last_n2 = int(np.nonzero(n2)[0][-1])
        cutoff = (centers[first_n] + centers[last_n2]) / 2.0
    else:
        cutoff = float(centers[pos])
    return float(max(0.1, min(99.9, cutoff)))


def _balance10(error_vad: NDArray[np.float64]) -> tuple[NDArray[np.float64], float | None]:
    """The balance-10 LID calibration error vector (:560-616). Returns (error, cutoff);
    `cutoff` is `None` for the >2-class else branch (there it is a per-file vector)."""
    scores = error_vad[:, 16:-2]
    n_classes = scores.shape[1]
    if n_classes == 2:
        c1 = error_vad[:, 16]
        c2 = error_vad[:, 17]
        cutoff = _balance10_cutoff(error_vad)
        # legacy: ComputeCost.m:580-583
        rel1 = (c1 > 150) * (c1 - 200 - 100 + cutoff)
        rel1 = rel1 * ((rel1 >= 0) / cutoff + (rel1 < 0) / (100 - cutoff))
        rel2 = (c2 > 150) * (c2 - 200 - cutoff)
        rel2 = rel2 * ((rel2 >= 0) / (100 - cutoff) + (rel2 < 0) / cutoff)
        rel_error = rel1 + rel2
        lid_flag = 100 * (rel_error > 0)
        lid_score = (100 - lid_flag + 100 * np.exp(-rel_error)) / 100
        # legacy: ComputeCost.m:590
        stat_in = (c1 > 150) * (c1 - 200) + (c1 <= 150) * c1
        std_term = np.std(stat_in, ddof=1)
        error = lid_score - np.log(max(1e-24, 1 * min(1.0, float(std_term)))) + 10 * (cutoff <= 0.1) + 10 * (cutoff >= 99.9)
        return np.asarray(error, dtype=F64), cutoff
    # legacy: ComputeCost.m:598-612 (else branch; per-file cutoff vector)
    rel_error = np.zeros(scores.shape[0], dtype=F64)
    for mm in range(n_classes):
        col = error_vad[:, 16 + mm]
        tmp = scores.copy()
        tmp[:, mm] = 0.0
        cutoff_vec = np.maximum(0.01, np.minimum(99.9, (col > 150) * np.max(tmp, axis=1)))
        rel_tmp = (col > 150) * (col - 200 - cutoff_vec)
        rel_tmp = rel_tmp * ((rel_tmp >= 0) / (100 - cutoff_vec) + (rel_tmp < 0) / cutoff_vec)
        rel_error = rel_error + rel_tmp
    lid_flag = 100 * (rel_error > 0)
    lid_score = (100 - lid_flag + 100 * np.exp(-rel_error)) / 100
    return np.asarray(lid_score, dtype=F64), None


def compute_cost(
    results: NDArray[np.float64],
    config_idx: int,
    balance: int,
    params: CostParams,
) -> tuple[float, CostBreakdown]:
    """ComputeCost.m's PURE balance-law assembly for one config (:428-652).

    `results` is a (post-`sortrows`) MultiConfigResults matrix; `config_idx` selects the
    conf-column rows. Returns (balance cost, breakdown). Only balances 0/3/4/5/10 are
    ported (6-9 are the WER shell-out laws, deferred to 4d)."""
    error_vad = _error_vad(np.asarray(results, dtype=F64), config_idx)
    nn_cost_seg = _nn_cost_seg(error_vad)
    nn_cost_lid = 0.0
    cutoff: float | None = None

    # legacy: ComputeCost.m:432-436
    cpu_mean = float(np.median(error_vad[:, 3])) if params.mode >= 0 else 0.0
    # legacy: ComputeCost.m:445/:453/:461
    coeff_rprop = 0.0 if params.mode == 4 else 1.0

    pfa = error_vad[:, 0]
    pmiss = error_vad[:, 1]
    bbp = params.balance_backprop
    is_algo1 = 1.0 if params.algo == 1 else 0.0
    cpu_term = is_algo1 * (1 * cpu_mean * (cpu_mean > 100) + 0.01 * cpu_mean * (cpu_mean <= 100))

    if balance == 0:
        # legacy: ComputeCost.m:440
        error = pfa + pmiss + 3 * (cpu_mean / 5)
    elif balance == 3:
        # legacy: ComputeCost.m:450 (over-90 saturation OFF: the 0* coefficient)
        error = (
            100 * (1 - coeff_rprop) * nn_cost_seg
            + coeff_rprop * (bbp * (pfa + 0 * (pfa > 90) * ((pfa - 90) ** 2)) + (1 - bbp) * (pmiss + 0 * (pmiss > 90) * ((pmiss - 90) ** 2)) + cpu_term)
        ) / 100
    elif balance == 4:
        # legacy: ComputeCost.m:458 (over-90 saturation ON: the 1* coefficient)
        error = (
            100 * (1 - coeff_rprop) * nn_cost_seg
            + coeff_rprop * (bbp * (pfa + 1 * (pfa > 90) * ((pfa - 90) ** 2)) + (1 - bbp) * (pmiss + 1 * (pmiss > 90) * ((pmiss - 90) ** 2)) + cpu_term)
        ) / 100
    elif balance == 5:
        # legacy: ComputeCost.m:466
        error = 100 * (1 - coeff_rprop) * nn_cost_seg + coeff_rprop * (error_vad[:, 2] + 0.0 * cpu_mean)
    elif balance == 10:
        # legacy: ComputeCost.m:552 (raw NNCostLID; the weights-based L2 is applied in
        # forward_backward, which holds the LID weights + is_bias).
        nn_cost_lid = _nn_cost_lid(error_vad)
        error, cutoff = _balance10(error_vad)
    else:
        raise ValueError(f"compute_cost ports balances 0/3/4/5/10, got {balance} (6-9 are deferred to 4d)")

    error = np.atleast_1d(np.asarray(error, dtype=F64))
    # legacy: ComputeCost.m:630 (balance >= 10: cost = mean(error.^2)) vs :646-651 (balance
    # < 6, mode not 4/1: cost = mean(error)).
    cost = float(np.mean(error**2)) if balance == 10 else float(np.mean(error))

    breakdown = CostBreakdown(
        nn_cost_seg=nn_cost_seg,
        nn_cost_lid=nn_cost_lid,
        cpu_mean=cpu_mean,
        error=error,
        cutoff=cutoff,
    )
    return cost, breakdown


# ---- The seam: ComputeGradient's contract over speech_rs.Engine -------------------------


def forward_backward(
    engine: speech_rs.Engine,
    weights: list[NDArray[np.float64]],
    listing_override: Path | None = None,
    *,
    config_idx: int = 1,
    l2: float = 0.0,
    is_bias: list[NDArray[np.float64]] | None = None,
) -> tuple[float, list[NDArray[np.float64]]]:
    """ComputeGradient's contract (`ComputeGradient.m:104-115`) over the coarse
    corpus-level `speech_rs.Engine`: seed the net(s), run the corpus fold, then read the
    results + per-network derivatives back and assemble

        f = NNCostSeg (+ NNCostLID for algo 6)           (:110 / :114)
        dfdtheta{net} = MultiDeriv ./ max(1,count)       (:357, count-normalized)
                        (+ L2 for the LID net)            (:362, LID only)

    `weights` is the `[sad]` (algo 3/4/5) or `[sad, lid]` (algo 6) numpy pairing that
    `Engine.set_weights`/`Engine.weights` round-trip. The returned gradient list mirrors
    that pairing, each truncated to `len(weights[net])` (`MultiDeriv(1:length(weights),1)`,
    :109/:113 -- drops the normalize mean/std tail when the caller passes a tail-stripped
    theta).

    L2 (`l2`/`is_bias`) reproduces the legacy asymmetry: it regularizes the LID net ONLY
    (net index 1), never the SAD net -- ComputeCost.m applies L2 only inside the
    `algo == 6` LID block (:362/:554), leaving the seg side unregularized (IMPROVEMENTS).
    Defaults `l2 == 0` -> no L2 (the algo-3 seam path).

    `listing_override` is ADVISORY: the coarse seam builds a fresh `Engine` per gradient
    eval against the batch listing (the legacy `WriteWeightedListing` + engine rerun), so
    by the time an `Engine` reaches here its corpus is already fixed at construction. The
    per-batch rebuild is the T12 driver's job; this function runs whatever corpus the
    passed-in engine was built with.
    """
    nets = [np.ascontiguousarray(np.asarray(w, dtype=F64)) for w in weights]
    engine.set_weights(0, nets)
    engine.run()

    results = np.asarray(engine.results_matrix(), dtype=F64)
    derivs = engine.weights_derivatives(0)
    error_vad = _error_vad(results, config_idx)

    f = _nn_cost_seg(error_vad)
    if len(nets) == 2:
        # legacy: ComputeGradient.m:114 f = f + NNCostLID (algo 6).
        nn_lid = _nn_cost_lid(error_vad)
        if l2 > 0.0 and is_bias is not None:
            lid_cost, _ = l2_penalty(nets[1], is_bias[1], l2)
            nn_lid += lid_cost
        f = f + nn_lid

    gradients: list[NDArray[np.float64]] = []
    for k, dv in enumerate(derivs):
        grad = average_derivs(np.asarray(dv, dtype=F64))[: nets[k].shape[0]]
        if k == 1 and l2 > 0.0 and is_bias is not None:
            # legacy: ComputeGradient.m:362 -- L2 gradient on the LID net only.
            _, lid_grad = l2_penalty(nets[k], is_bias[k], l2)
            grad = grad + lid_grad[: grad.shape[0]]
        gradients.append(np.ascontiguousarray(grad))
    return float(f), gradients
