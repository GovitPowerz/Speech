"""Assert the Phase-0b-ii fixtures are present and shaped as expected."""

from pathlib import Path

REF = Path("tests/reference_data/phase0bii")


def test_vrcts_fixtures_present() -> None:
    assert (REF / "vrcts_1seg.xml").read_text().count("<SpeechSegment") == 1
    assert (REF / "vrcts_empty.xml").read_text().count("<SpeechSegment") == 0
    assert (REF / "vrcts_16seg.xml").read_text().count("<SpeechSegment") == 16


def test_stm_and_configs_present() -> None:
    assert (REF / "ref.stm").read_text().startswith(";;")
    assert "BLSTM_decision_thresh_rising" in (REF / "seg.config").read_text()
