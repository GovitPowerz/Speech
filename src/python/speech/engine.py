"""Engine driver: invoke the Rust engine, assemble cost + gradients.

Ported from legacy MATLAB `Optimizer_V6.2.2/functions/ComputeCost.m` (782 LOC) +
`ComputeGradient.m` (336) + `RunFsp.py` (the subprocess seam -> in-process PyO3 call to
`speech_rs.Engine`). See design spec section 6.

Two layers:

  * The PURE cost assembly (`compute_cost` + the `average_derivs` / `pool_input_stats` /
    `l2_penalty` helpers): given the engine's channel results (and, for the deriv/stats
    helpers, the per-worker pieces), reproduce ComputeCost.m's `MultiDeriv./max(1,count)`
    averaging (:357), pooled input-stats recombination (:285-300), L2 penalties
    (:362/:554), and the balance laws 0/3/4/5/10 (:428-652). Octave-pinned bit-for-bit
    (`tests/test_phase4c_engine_cost.py`).
  * The SEAM (`forward_backward`): ComputeGradient's contract over a `fold_run.FoldRun` --
    one gradient fold at the weights -> `f = NNCostSeg (+ NNCostLID)` (:110/:114) with the
    count-normalized (`+ L2` for the LID net) gradient. The engine mechanics (the config
    overlay, the F11 rule, the fast guard, the path resolution) are the fold run's (issue #22).

The results are read by NAME: `speech_rs.Engine.channel_results()` returns one array per
field, wrapped here as `ChannelResults` (issue #23). The legacy `Error_vad` column indices
(`Error_vad(:,5)`, `(:,15)`, `(:,17:end-2)`, ...) that `ComputeCost.m` reads are the
`seg_cost`, `lid_cost`, `lid_scores`, ... fields; the wire layout itself lives in Rust
(`engine/channel_result.rs`) and, for the Octave goldens only, in `tests/_result_rows.py`.
The in-band LID target (`> 150`, `- 200`) is decoded before it reaches Python:
`lid_target` names the class, `lid_scores` holds the decoded score.

Legacy quirks reproduced on purpose are tracked in IMPROVEMENTS.md; the load-bearing ones
carry a `# legacy:` provenance comment at their site.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from speech.fold_run import FoldRun

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


@dataclass(frozen=True, eq=False)
class ChannelResults:
    """The engine's channel results by name, one entry per (file, config, channel) in
    ascending order: the columnar view `speech_rs.Engine.channel_results()` returns.

    Ids are 0-based, like every `pos` on the seam. `lid_target` is `-1` for a row with no
    in-band target (an out-of-set trial), `lid_scores` is `n x N` with the target column
    already decoded (`N == 0` on a run without LID). `lid_correct` is the `_IsLIDCorrect`
    flag (100 / 0 on the wire). The wire layout itself lives in Rust
    (`engine/channel_result.rs`) and, for the Octave goldens only, in `tests/_result_rows.py`.
    """

    file: NDArray[np.int64]
    conf: NDArray[np.int64]
    chan: NDArray[np.int64]
    pfa: NDArray[np.float64]
    pmiss: NDArray[np.float64]
    error_rate: NDArray[np.float64]
    time_per_hour: NDArray[np.float64]
    seg_cost: NDArray[np.float64]
    audio_duration: NDArray[np.float64]
    speech_duration: NDArray[np.float64]
    lid_cost: NDArray[np.float64]
    lid_correct: NDArray[np.bool_]
    lid_target: NDArray[np.int64]
    lid_scores: NDArray[np.float64]
    lid_count: NDArray[np.int64]
    seg_count: NDArray[np.int64]

    @classmethod
    def from_seam(cls, d: Mapping[str, NDArray[Any]]) -> ChannelResults:
        """Wrap the seam's dict (one array per field; every crossing is already a copy)."""
        return cls(**{f.name: np.asarray(d[f.name]) for f in fields(cls)})

    def __len__(self) -> int:
        return int(self.file.shape[0])

    def for_config(self, pos: int) -> ChannelResults:
        """The rows of config `pos`. Raises on an empty selection: a silent empty
        selection is a silent 0.0 cost, the failure mode a 1-vs-0-based slip produces."""
        sel = self.conf == pos
        if not np.any(sel):
            raise ValueError(f"channel results: no row for config {pos} (configs present: {sorted(set(self.conf.tolist()))})")
        return ChannelResults(**{f.name: getattr(self, f.name)[sel] for f in fields(self)})


# ---- Pure results-derived pieces (shared by compute_cost + forward_backward) ------------


def _nn_cost_seg(r: ChannelResults) -> float:
    # legacy: ComputeCost.m:430 NNcost = sum(Error_vad(:,5))/max(1e-6,sum(Error_vad(:,end)));
    return float(np.sum(r.seg_cost) / max(1e-6, float(np.sum(r.seg_count))))


def _nn_cost_lid(r: ChannelResults) -> float:
    # legacy: ComputeCost.m:552 NNCostLID = sum(Error_vad(:,15))/max(1e-6,sum(Error_vad(:,end-1)));
    return float(np.sum(r.lid_cost) / max(1e-6, float(np.sum(r.lid_count))))


# ---- Averaging / pooling / L2 -----------------------------------------------------------


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


def _balance10_cutoff(z: NDArray[np.float64], t: NDArray[np.int64]) -> float:
    """The 2-class cutoff search (:561-576) on the decoded scores `z` (n x 2) and the
    target index `t`. `Error_vad(:,17)`/`(:,18)` were the two in-band columns: the legacy
    `c2(c2 > 150) - 200` is `z[:, 1][t == 1]`, and `300 - c1(c1 > 150)` is
    `100 - z[:, 0][t == 0]` (both exact on the wire's [200, 300] range, so bit-identical)."""
    centers = np.arange(10000, dtype=F64) * 0.01 + 0.005  # (0:0.01:99.99)+0.005
    tmp = z[:, 1][t == 1]
    n = _matlab_hist(tmp, centers)
    n = np.cumsum(100 * n / np.sum(n))
    tmp2 = 100 - z[:, 0][t == 0]
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


def _balance10(r: ChannelResults) -> tuple[NDArray[np.float64], float | None]:
    """The balance-10 LID calibration error vector (:560-616). Returns (error, cutoff);
    `cutoff` is `None` for the >2-class else branch (there it is a per-file vector).

    Written on the decoded scores: every legacy `(col > 150)` mask is `(t == k)`, every
    `col - 200` is the decoded column, with the subtraction order kept left to right so
    each expression is the legacy's bit for bit (the decode is exact on [200, 300])."""
    z = r.lid_scores
    t = r.lid_target
    n_classes = z.shape[1]
    if n_classes == 2:
        z1 = z[:, 0]
        z2 = z[:, 1]
        cutoff = _balance10_cutoff(z, t)
        # legacy: ComputeCost.m:580-583
        rel1 = (t == 0) * (z1 - 100 + cutoff)
        rel1 = rel1 * ((rel1 >= 0) / cutoff + (rel1 < 0) / (100 - cutoff))
        rel2 = (t == 1) * (z2 - cutoff)
        rel2 = rel2 * ((rel2 >= 0) / (100 - cutoff) + (rel2 < 0) / cutoff)
        rel_error = rel1 + rel2
        lid_flag = 100 * (rel_error > 0)
        lid_score = (100 - lid_flag + 100 * np.exp(-rel_error)) / 100
        # legacy: ComputeCost.m:590 -- `(c1 > 150)*(c1 - 200) + (c1 <= 150)*c1` is z1.
        std_term = np.std(z1, ddof=1)
        error = lid_score - np.log(max(1e-24, 1 * min(1.0, float(std_term)))) + 10 * (cutoff <= 0.1) + 10 * (cutoff >= 99.9)
        return np.asarray(error, dtype=F64), cutoff
    # legacy: ComputeCost.m:598-612 (else branch; per-file cutoff vector)
    rel_error = np.zeros(z.shape[0], dtype=F64)
    for mm in range(n_classes):
        col = z[:, mm]
        is_target = t == mm
        tmp = z.copy()
        tmp[:, mm] = 0.0
        cutoff_vec = np.maximum(0.01, np.minimum(99.9, is_target * np.max(tmp, axis=1)))
        rel_tmp = is_target * (col - cutoff_vec)
        rel_tmp = rel_tmp * ((rel_tmp >= 0) / (100 - cutoff_vec) + (rel_tmp < 0) / cutoff_vec)
        rel_error = rel_error + rel_tmp
    lid_flag = 100 * (rel_error > 0)
    lid_score = (100 - lid_flag + 100 * np.exp(-rel_error)) / 100
    return np.asarray(lid_score, dtype=F64), None


def compute_cost(
    results: ChannelResults,
    pos: int,
    balance: int,
    params: CostParams,
) -> tuple[float, CostBreakdown]:
    """ComputeCost.m's PURE balance-law assembly for one config (:428-652).

    `results` is the engine's channel results; `pos` selects the config (0-based; the
    legacy `Error_vad = MCR(MCR(:,2)==ii, 4:end)`). Returns (balance cost, breakdown).
    Only balances 0/3/4/5/10 are ported (6-9 are the WER shell-out laws, deferred to 4d)."""
    r = results.for_config(pos)
    nn_cost_seg = _nn_cost_seg(r)
    nn_cost_lid = 0.0
    cutoff: float | None = None

    # legacy: ComputeCost.m:432-436
    cpu_mean = float(np.median(r.time_per_hour)) if params.mode >= 0 else 0.0
    # legacy: ComputeCost.m:445/:453/:461
    coeff_rprop = 0.0 if params.mode == 4 else 1.0

    pfa = r.pfa
    pmiss = r.pmiss
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
        error = 100 * (1 - coeff_rprop) * nn_cost_seg + coeff_rprop * (r.error_rate + 0.0 * cpu_mean)
    elif balance == 10:
        # legacy: ComputeCost.m:552 (raw NNCostLID; the weights-based L2 is applied in
        # forward_backward, which holds the LID weights + is_bias).
        nn_cost_lid = _nn_cost_lid(r)
        error, cutoff = _balance10(r)
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


# ---- The seam: ComputeGradient's contract over a fold run ---------------------------------


def forward_backward(
    fold: FoldRun,
    weights: list[NDArray[np.float64]],
    *,
    l2: float = 0.0,
    is_bias: list[NDArray[np.float64]] | None = None,
) -> tuple[float, list[NDArray[np.float64]]]:
    """ComputeGradient's contract (`ComputeGradient.m:104-115`) over one gradient fold: seed
    the net(s), run the corpus fold at those weights, then read the results + per-network
    derivatives back and assemble

        f = NNCostSeg (+ NNCostLID for algo 6)           (:110 / :114)
        dfdtheta{net} = MultiDeriv ./ max(1,count)       (:357, count-normalized)
                        (+ L2 for the LID net)            (:362, LID only)

    `fold` is a `FoldRun` built with `backprop=True`; it owns the engine, the config overlay
    (the F11 `Epochs 0` rule) and the listing, so the hard-example mini-batch path is simply
    a fresh `FoldRun` per step on the batch listing (`drivers/train.py::_backprop_inner`),
    the in-process analogue of `ComputeGradient.m`'s per-gradient `WriteWeightedListing`
    (:53/:77) + fresh-`fsp`-per-eval (`CostFunction` :104).

    `weights` is the `[sad]` (algo 3/4/5) or `[sad, lid]` (algo 6) numpy pairing that
    `Engine.set_weights`/`Engine.weights` round-trip. The returned gradient list mirrors
    that pairing, each truncated to `len(weights[net])` (`MultiDeriv(1:length(weights),1)`,
    :109/:113 -- drops the normalize mean/std tail when the caller passes a tail-stripped
    theta).

    L2 (`l2`/`is_bias`) reproduces the legacy asymmetry: it regularizes the LID net ONLY
    (net index 1), never the SAD net -- ComputeCost.m applies L2 only inside the
    `algo == 6` LID block (:362/:554), leaving the seg side unregularized (IMPROVEMENTS).
    Defaults `l2 == 0` -> no L2 (the algo-3 seam path).
    """
    if not fold.backprop:
        raise ValueError("forward_backward needs a gradient fold (FoldRun(..., backprop=True)); a forward-only fold harvests no derivatives")

    nets = [np.ascontiguousarray(np.asarray(w, dtype=F64)) for w in weights]
    res = fold.run(nets)

    # Config 0 throughout: the cost is read off the same config the weights went into and
    # the derivatives come out of.
    f = res.nn_cost_seg
    if res.twin:
        # legacy: ComputeGradient.m:114 f = f + NNCostLID (algo 6). Kept as seg + (lid + L2),
        # the legacy's own order, rather than `res.nn_cost + L2`: the rounding differs.
        nn_lid = res.nn_cost_lid
        if l2 > 0.0 and is_bias is not None:
            lid_cost, _ = l2_penalty(nets[1], is_bias[1], l2)
            nn_lid += lid_cost
        f = f + nn_lid

    gradients: list[NDArray[np.float64]] = []
    for k, dv in enumerate(res.derivs):
        grad = average_derivs(dv)[: nets[k].shape[0]]
        if k == 1 and l2 > 0.0 and is_bias is not None:
            # legacy: ComputeGradient.m:362 -- L2 gradient on the LID net only.
            _, lid_grad = l2_penalty(nets[k], is_bias[k], l2)
            grad = grad + lid_grad[: grad.shape[0]]
        gradients.append(np.ascontiguousarray(grad))
    return float(f), gradients
