"""Unit tests for the numpy `compute_errors` Pass 2 oracle (`speech.scoring`).

Pins the same hand-computed ref/hyp case as the Rust Task-7 test: ref speech-span
[2, 5], hyp speech-span [3, 6], audio_duration 10, giving
`per_class[Speech] = [1/3, 1/7, 0.1]` and `per_class[Other] = [1/7, 1/3, 0.1]`.
Anchors the oracle independently of the fixture cross-check.
"""

from speech.scoring import (
    END_CODE,
    OTHER_CODE,
    SPEECH_CODE,
    compute_errors_pass2_oracle,
)


def test_task7_hand_case() -> None:
    # ref: [Other@0, Speech@2, Other@5, End@10]; hyp: [Other@0, Speech@3, Other@6, End@10].
    ref = [(0.0, OTHER_CODE), (2.0, SPEECH_CODE), (5.0, OTHER_CODE), (10.0, END_CODE)]
    hyp = [(0.0, OTHER_CODE), (3.0, SPEECH_CODE), (6.0, OTHER_CODE), (10.0, END_CODE)]
    per_class = compute_errors_pass2_oracle(ref, hyp, 10.0)
    assert per_class[SPEECH_CODE] == [1.0 / 3.0, 1.0 / 7.0, 0.1]
    assert per_class[OTHER_CODE] == [1.0 / 7.0, 1.0 / 3.0, 0.1]


def test_identical_ref_hyp_is_zero_error() -> None:
    # perfect hypothesis: no scorable mismatch anywhere.
    ref = [(0.0, OTHER_CODE), (2.0, SPEECH_CODE), (5.0, OTHER_CODE), (10.0, END_CODE)]
    per_class = compute_errors_pass2_oracle(ref, list(ref), 10.0)
    assert per_class[SPEECH_CODE] == [0.0, 0.0, 0.0]
    assert per_class[OTHER_CODE] == [0.0, 0.0, 0.0]
