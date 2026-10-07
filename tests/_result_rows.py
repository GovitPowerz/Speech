"""The matrix-to-names decoder for the Octave goldens (issue #23).

The pure-Python CI job has no `speech_rs`, and the `ComputeCost.m` goldens under
`tests/reference_data/phase4c/` are `MultiConfigResults` matrices the Octave stage dumped.
This is the ONE place in Python that knows the result row's column layout; production code
reads names through `speech_rs.Engine.channel_results()` and `speech.engine.ChannelResults`.
The Rust writer is `engine/channel_result.rs::ChannelResult::to_row`; `tests/pyo3/
test_channel_results.py` cross-pins this decoder against it on two committed fixtures.

The `MultiConfigResults` row is `[file+1, conf+1, chan+1, res...]`: the three leading id
columns are the legacy `MultiConfigResults(:,1:3)` (1-based), and `res` is the result row.
Within the result row, the legacy 1-based `Error_vad(:,k)` maps to 0-based `k-1`:

  legacy Error_vad(:,k)  0-based res[k-1]  field
  ------------------------------------------------------------------
  1                       0                 pfa
  2                       1                 pmiss
  3                       2                 error_rate (100 - success)
  4                       3                 time_per_hour (the wall-clock timing column)
  5                       4                 seg_cost (the seg-cost numerator)
  6, 7                    5, 6              audio_duration, speech_duration
  8:14                    7:13              WER tallies (dead surface, not decoded)
  15                      14                lid_cost (the LID-cost numerator)
  16                      15                lid_correct (100 / 0 on the wire)
  17:end-2                16:-2             lid_scores, the target in-band (> 150 -> v - 200)
  end-1                   -2                lid_count (the LID-cost denominator)
  end                     -1                seg_count (the seg-cost denominator)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from speech.engine import ChannelResults

F64 = np.float64


def channel_results_from_matrix(mcr: NDArray[np.float64]) -> ChannelResults:
    """Decode a `MultiConfigResults` matrix (1-based id prefix + result rows) to names.

    Ids come out 0-based, like every `pos` on the seam. An empty `(0, 0)` matrix decodes
    to zero-length fields with `lid_scores` of shape `(0, 0)`."""
    m = np.asarray(mcr, dtype=F64)
    if m.size == 0:
        return ChannelResults.from_seam(_empty())
    res = m[:, 3:]
    scores = np.array(res[:, 16:-2], dtype=F64)
    n_rows = scores.shape[0]
    lid_target = np.full(n_rows, -1, dtype=np.int64)
    for i in range(n_rows):
        above = np.flatnonzero(scores[i] > 150.0)
        if above.size:
            t = int(above[-1])  # the LAST column > 150 is the target (the legacy loop)
            lid_target[i] = t
            scores[i, t] = scores[i, t] - 200.0
    return ChannelResults(
        file=(m[:, 0] - 1).astype(np.int64),
        conf=(m[:, 1] - 1).astype(np.int64),
        chan=(m[:, 2] - 1).astype(np.int64),
        pfa=res[:, 0].copy(),
        pmiss=res[:, 1].copy(),
        error_rate=res[:, 2].copy(),
        time_per_hour=res[:, 3].copy(),
        seg_cost=res[:, 4].copy(),
        audio_duration=res[:, 5].copy(),
        speech_duration=res[:, 6].copy(),
        lid_cost=res[:, 14].copy(),
        lid_correct=res[:, 15] == 100.0,
        lid_target=lid_target,
        lid_scores=scores,
        lid_count=res[:, -2].astype(np.int64),
        seg_count=res[:, -1].astype(np.int64),
    )


def _empty() -> dict[str, NDArray[np.float64] | NDArray[np.int64] | NDArray[np.bool_]]:
    z = np.zeros(0, dtype=F64)
    return {
        "file": np.zeros(0, dtype=np.int64),
        "conf": np.zeros(0, dtype=np.int64),
        "chan": np.zeros(0, dtype=np.int64),
        "pfa": z,
        "pmiss": z,
        "error_rate": z,
        "time_per_hour": z,
        "seg_cost": z,
        "audio_duration": z,
        "speech_duration": z,
        "lid_cost": z,
        "lid_correct": np.zeros(0, dtype=np.bool_),
        "lid_target": np.zeros(0, dtype=np.int64),
        "lid_scores": np.zeros((0, 0), dtype=F64),
        "lid_count": np.zeros(0, dtype=np.int64),
        "seg_count": np.zeros(0, dtype=np.int64),
    }
