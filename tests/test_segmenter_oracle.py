"""Unit tests for the numpy segmenter-decision oracle (`speech.scoring`).

Pins the same hand-computed `clean_rising_falling_crossing` case as the Rust
Task-2 test (`begin=1.5`, `end=3.5`), so the oracle is anchored independently of
the fixture cross-check.
"""

from speech.scoring import SPEECH_CODE, update_segmentation_oracle


def test_clean_rising_falling_crossing() -> None:
    # dt=1.0, off=0. results cross up between idx1(0.0)->idx2(1.0) at 0.5*dt,
    # and back down between idx3(1.0)->idx4(0.0) at 0.5*dt past idx3. area=0.
    row = [0.0, 0.0, 1.0, 1.0, 0.0]
    segs = update_segmentation_oracle(row, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0)
    # rising: r(2)>=0.5 && r(1)<0.5 -> begin = 1*(2 - (1.0-0.5)/(1.0-0.0)) = 1.5
    # falling: r(4)<=0.5 && r(3)>0.5 -> end = 1*(4 - (0.0-0.5)/(0.0-1.0)) = 3.5
    assert segs == [(1.5, 3.5, SPEECH_CODE)]


def test_no_crossing_yields_nothing() -> None:
    # never reaches the rising threshold -> no segment.
    row = [0.0, 0.1, 0.2, 0.1, 0.0]
    assert update_segmentation_oracle(row, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0) == []


def test_offset_shifts_boundaries() -> None:
    row = [0.0, 0.0, 1.0, 1.0, 0.0]
    segs = update_segmentation_oracle(row, 0.5, 0.0, 0.5, 0.0, 1.0, 10.0)
    assert segs == [(11.5, 13.5, SPEECH_CODE)]
