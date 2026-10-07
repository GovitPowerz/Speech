"""`tests/_result_rows.py`, the matrix-to-names decoder, and `ChannelResults` (issue #23).

Crafted rows only; the decoder is cross-pinned against the Rust writer in
`tests/pyo3/test_channel_results.py`."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
from speech.engine import ChannelResults

from tests._result_rows import channel_results_from_matrix


def _row(file: int, conf: int, chan: int, res: list[float]) -> list[float]:
    return [float(file), float(conf), float(chan), *res]


def _plain(seg_cost: float, seg_count: float) -> list[float]:
    res = [0.0] * 18
    res[0], res[1], res[2], res[3] = 1.5, 2.5, 3.5, 4.5
    res[4] = seg_cost
    res[5], res[6] = 100.0, 50.0
    res[7] = -1.0
    res[17] = seg_count
    return res


def test_plain_row_decodes_with_no_lid() -> None:
    mcr = np.array([_row(1, 1, 1, _plain(10.0, 2.0)), _row(1, 1, 2, _plain(20.0, 4.0))])
    r = channel_results_from_matrix(mcr)
    assert len(r) == 2
    assert r.file.tolist() == [0, 0] and r.conf.tolist() == [0, 0] and r.chan.tolist() == [0, 1]
    assert r.pfa.tolist() == [1.5, 1.5] and r.pmiss.tolist() == [2.5, 2.5] and r.error_rate.tolist() == [3.5, 3.5]
    assert r.time_per_hour.tolist() == [4.5, 4.5]
    assert r.seg_cost.tolist() == [10.0, 20.0] and r.seg_count.tolist() == [2, 4]
    assert r.audio_duration.tolist() == [100.0, 100.0] and r.speech_duration.tolist() == [50.0, 50.0]
    assert r.lid_cost.tolist() == [0.0, 0.0] and r.lid_count.tolist() == [0, 0]
    assert r.lid_correct.tolist() == [False, False]
    assert r.lid_target.tolist() == [-1, -1]
    assert r.lid_scores.shape == (2, 0)
    assert r.seg_count.dtype == np.int64 and r.lid_correct.dtype == np.bool_


def test_lid_row_decodes_the_target_column() -> None:
    res = [0.0] * 14 + [7.5, 100.0, 10.0, 260.0, 42.0, 9.0]  # N = 2, target in column 1
    r = channel_results_from_matrix(np.array([_row(3, 2, 1, res)]))
    assert r.file.tolist() == [2] and r.conf.tolist() == [1] and r.chan.tolist() == [0]
    assert r.lid_cost.tolist() == [7.5] and r.lid_correct.tolist() == [True]
    assert r.lid_target.tolist() == [1]
    assert r.lid_scores.tolist() == [[10.0, 60.0]]
    assert r.lid_count.tolist() == [42] and r.seg_count.tolist() == [9]


def test_no_target_row_is_minus_one_and_raw() -> None:
    res = [0.0] * 14 + [0.0, 0.0, 20.0, 30.0, 0.0, 0.0]
    r = channel_results_from_matrix(np.array([_row(1, 1, 1, res)]))
    assert r.lid_target.tolist() == [-1]
    assert r.lid_scores.tolist() == [[20.0, 30.0]]


def test_empty_matrix_decodes_to_empty_fields() -> None:
    r = channel_results_from_matrix(np.zeros((0, 0)))
    assert len(r) == 0 and r.lid_scores.shape == (0, 0)


def test_for_config_selects_and_raises_on_empty() -> None:
    mcr = np.array([_row(1, 1, 1, _plain(1.0, 1.0)), _row(1, 2, 1, _plain(2.0, 1.0)), _row(2, 2, 1, _plain(3.0, 1.0))])
    r = channel_results_from_matrix(mcr)
    c1 = r.for_config(1)
    assert len(c1) == 2 and c1.seg_cost.tolist() == [2.0, 3.0] and c1.file.tolist() == [0, 1]
    with pytest.raises(ValueError, match="no row for config 5"):
        r.for_config(5)


def test_from_seam_round_trips_the_dict() -> None:
    res = [0.0] * 14 + [7.5, 100.0, 10.0, 260.0, 42.0, 9.0]
    r = channel_results_from_matrix(np.array([_row(1, 1, 1, res)]))
    back = ChannelResults.from_seam(asdict(r))
    for k, v in asdict(r).items():
        assert np.array_equal(getattr(back, k), v), k
        assert getattr(back, k).dtype == v.dtype, k
