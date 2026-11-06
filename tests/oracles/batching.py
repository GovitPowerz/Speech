"""Oracle of `getCases.m`'s private `getWorstAndBest` subfunction (`getCases.m:116-165`).

Unreachable standalone in the legacy (a MATLAB file-local helper whose caller needs the
full `ComputeCost`/`CostFunction` engine chain); Octave-pinned through the FALLBACK tier.
No production caller: `drivers/train.py`'s batch rebuild draws through `create_batches`
and `get_new_batch`. Test-only since issue #133.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _mround(x: float) -> float:
    """MATLAB round: half away from zero (numpy's is banker's)."""
    return float(np.sign(x) * np.floor(np.abs(x) + 0.5))


@dataclass
class WorstAndBest:
    """`[indBest,indMiddle,indWorst,scoreBest,scoreMiddle,scoreWorst]` from
    `getCases.m`'s private `getWorstAndBest` subfunction."""

    ind_best: list[int]
    ind_middle: list[int]
    ind_worst: list[int]
    score_best: list[float]
    score_middle: list[float]
    score_worst: list[float]


def get_worst_and_best(sorting_table: NDArray[np.float64], nb_worst: int, nb_best: int, nb_middle: int) -> WorstAndBest:
    """Port of `getCases.m:116-165`'s private `getWorstAndBest` subfunction (unreachable
    standalone in the legacy -- it is a MATLAB file-local helper of `getCases.m`, which
    itself requires the full `ComputeCost`/`CostFunction` engine chain to invoke).

    `sorting_table` is `[cost, index]`, PRE-SORTED ascending by cost (mirrors the
    caller's `sortrows` contract in `getCases.m`); `index` values are opaque (typically
    0-based file indices) and are returned as-is, deduplicated by value. Worst = the
    `nb_worst` highest-cost rows (scanned from the tail, first-seen-wins dedup, order
    restored ascending-by-original-position via a final reverse); best = the `nb_best`
    lowest-cost rows excluding anything already claimed by worst; middle = `nb_middle`
    rows evenly step-spaced between best and worst (`round` half-away-from-zero, min
    step 1), excluding anything already claimed. `nb_worst<=0` widens worst to cover the
    whole table; `nb_best<=0` still reserves exactly one "best" slot count (row 1) even
    though nothing is appended -- both mirror the legacy's `else` branches verbatim,
    including their effect on the `step`/`count` bookkeeping the middle scan reuses.

    Internal bookkeeping (`count`/`nb_worst`/`nb_best`) is kept 1-based, mirroring the
    legacy row-position arithmetic exactly; only array reads are translated to 0-based.
    """
    file_nb = sorting_table.shape[0]

    ind_worst: list[int] = []
    score_worst: list[float] = []
    if nb_worst > 0:
        count = file_nb
        while len(ind_worst) < nb_worst and count > 0:
            idx = int(sorting_table[count - 1, 1])
            if not ind_worst or idx not in ind_worst:
                ind_worst.append(idx)
                score_worst.append(float(sorting_table[count - 1, 0]))
            count -= 1
        nb_worst = count + 1
    else:
        nb_worst = file_nb
    ind_worst.reverse()
    score_worst.reverse()

    ind_best: list[int] = []
    score_best: list[float] = []
    if nb_best > 0:
        count = 1
        while len(ind_best) < nb_best and count <= file_nb:
            idx = int(sorting_table[count - 1, 1])
            not_worst = idx not in ind_worst
            if (not ind_best and not_worst) or (idx not in ind_best and not_worst):
                ind_best.append(idx)
                score_best.append(float(sorting_table[count - 1, 0]))
            count += 1
        nb_best = count - 1
    else:
        nb_best = 1

    ind_middle: list[int] = []
    score_middle: list[float] = []
    if nb_middle > 0:
        step = int(max(1.0, _mround((nb_worst - nb_best + 1) / (nb_middle + 1))))
        count = nb_best + step
        while len(ind_middle) < nb_middle and count <= nb_worst:
            idx = int(sorting_table[count - 1, 1])
            not_taken = idx not in ind_best and idx not in ind_worst
            if (not ind_middle and not_taken) or (idx not in ind_middle and not_taken):
                ind_middle.append(idx)
                score_middle.append(float(sorting_table[count - 1, 0]))
            count += step

    return WorstAndBest(ind_best, ind_middle, ind_worst, score_best, score_middle, score_worst)
