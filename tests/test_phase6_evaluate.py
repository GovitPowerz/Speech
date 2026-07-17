"""Phase 6 Task 4: DCF scorer gates.

`dcf()` is pinned value-for-value against the REAL vendored NIST `scoreFile_SAD.pl` (v2.1)
via the committed `tests/reference_data/phase6/dcf/*.json` goldens -- each produced by
`scripts/extract_phase6_fixtures.py` running the crafted ref/hyp intervals through
`/usr/bin/perl` (`tools/perl_oracle/run_sad_scorer.sh`). The perl prints Prob_Miss /
Prob_FalseAlarm / DCF at `%7.5f` (5 decimals); the port reproduces the scorer's exact
double-precision segment accumulation, so pinning is 5-decimal string equality (`{:.5f}`)
-- the literal "float equality to the scorer's printed precision". CI never runs the perl;
it only consumes these goldens.

The `load_tab_ref` / `load_vrcts_hyp` adapters are pinned separately on crafted inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from speech.evaluate import CollarScore, DcfReport, Interval, dcf, load_tab_ref, load_vrcts_hyp

DCF_DIR = Path(__file__).resolve().parent / "reference_data" / "phase6" / "dcf"
GOLDENS = sorted(p for p in DCF_DIR.glob("*.json") if p.name != "dcf_manifest.json")


def _intervals(rows: list[list[Any]]) -> list[Interval]:
    return [(float(s), float(e), str(k)) for s, e, k in rows]


# --- value-oracle parity: dcf() vs the real perl scorer, per collar -------------------


@pytest.mark.parametrize("golden_path", GOLDENS, ids=lambda p: p.stem)
def test_dcf_matches_perl_oracle(golden_path: Path) -> None:
    golden = json.loads(golden_path.read_text())
    ref = _intervals(golden["ref"])
    hyp = _intervals(golden["hyp"])

    collars = [c["collar"] for c in golden["collars"]]
    report = dcf(ref, hyp, collars=collars)

    assert len(report.scores) == len(golden["collars"])
    for got, want in zip(report.scores, golden["collars"], strict=True):
        assert got.collar == want["collar"]
        # 5-decimal string equality == the scorer's printed %7.5f precision.
        for field in ("pmiss", "pfa", "dcf"):
            got_str = f"{getattr(got, field):.5f}"
            want_str = f"{want[field]:.5f}"
            assert got_str == want_str, f"{golden_path.stem} collar={want['collar']} {field}: port {got_str} != perl {want_str}"


def test_all_14_goldens_present() -> None:
    """Guard against a silently-empty golden dir (which would vacuously pass the
    parametrized gate)."""
    assert len(GOLDENS) == 14, f"expected 14 committed dcf goldens, found {len(GOLDENS)}"


# --- dcf() semantics (redundant with the goldens, but name the behavior each pins) ----


def test_dcf_report_by_collar_lookup() -> None:
    report = dcf([(0.0, 6.0, "S")], [(0.0, 6.0, "speech")])
    assert isinstance(report.by_collar(0.0), CollarScore)
    assert report.by_collar(2.0).dcf == 0.0
    with pytest.raises(KeyError):
        report.by_collar(0.75)


def test_dcf_ri_counts_as_speech_not_nonspeech() -> None:
    """RI folds into the speech region. Hyp calls the RI second wholly non-speech, so it
    reads as a miss (nonzero pmiss), NOT as a correct non-speech call (which would leave
    pmiss 0 and raise pfa)."""
    ref = [(0.0, 1.0, "S"), (1.0, 2.0, "RI"), (2.0, 4.0, "NS")]
    hyp = [(0.0, 1.0, "speech"), (1.0, 4.0, "non-speech")]
    score = dcf(ref, hyp).by_collar(0.0)
    assert score.pmiss == pytest.approx(1.0 / 2.0)  # 1s FN over the 2s S+RI speech region
    assert score.pfa == 0.0


@pytest.mark.parametrize("kind", ["non-speech", "nonspeech"])
def test_dcf_accepts_both_nonspeech_spellings(kind: str) -> None:
    """The perl accepts both `non-speech` and `nonspeech` (:326); the port must too."""
    ref = [(0.0, 4.0, "NS")]
    hyp = [(0.0, 4.0, kind)]
    score = dcf(ref, hyp).by_collar(0.0)
    assert score.pfa == 0.0
    assert score.dcf == 0.0


def test_dcf_degenerate_all_speech_is_zero_not_nan() -> None:
    """v2.1 zero-not-NaN: an all-speech ref with a perfect hyp has no nonspeech time; pfa
    must be 0.0 (guarded denominator), not NaN."""
    report = dcf([(0.0, 6.0, "S")], [(0.0, 6.0, "speech")])
    for s in report.scores:
        assert s.pmiss == 0.0 and s.pfa == 0.0 and s.dcf == 0.0


def test_dcf_hyp_padded_when_shorter_than_ref() -> None:
    """Hyp ends before the ref: the tail is padded to NonSpeech, becoming a miss."""
    score = dcf([(0.0, 4.0, "S")], [(0.0, 3.0, "speech")]).by_collar(0.0)
    assert score.pmiss == pytest.approx(1.0 / 4.0)  # 1s of padded FN over 4s speech


def test_dcf_report_type() -> None:
    assert isinstance(dcf([(0.0, 1.0, "S")], [(0.0, 1.0, "speech")]), DcfReport)


# --- load_tab_ref adapter -------------------------------------------------------------


def test_load_tab_ref_computedcf_columns(tmp_path: Path) -> None:
    """The -s 2 -e 3 -g 4 layout: cols 0,1 are ignored, 2/3/4 are start/end/type."""
    tab = tmp_path / "ref.tab"
    tab.write_text("file\t1\t0.0\t2.0\tS\nfile\t1\t2.0\t4.0\tNS\nfile\t1\t4.0\t6.0\tRI\n")
    assert load_tab_ref(tab) == [(0.0, 2.0, "S"), (2.0, 4.0, "NS"), (4.0, 6.0, "RI")]


def test_load_tab_ref_custom_columns(tmp_path: Path) -> None:
    tab = tmp_path / "ref.tab"
    tab.write_text("0.0\t2.0\tNT\n")
    assert load_tab_ref(tab, start_col=0, end_col=1, type_col=2) == [(0.0, 2.0, "NT")]


def test_load_tab_ref_skips_blank_lines(tmp_path: Path) -> None:
    tab = tmp_path / "ref.tab"
    tab.write_text("file\t1\t0.0\t2.0\tS\n\nfile\t1\t2.0\t4.0\tNS\n")
    assert load_tab_ref(tab) == [(0.0, 2.0, "S"), (2.0, 4.0, "NS")]


# --- load_vrcts_hyp adapter -----------------------------------------------------------

_VRCTS_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<AudioDoc name="ex" path="ex.wav">\n'
    '<ProcList>\n<Proc name="vrcts_part" version="1.3"/>\n</ProcList>\n'
    "<ChannelList>\n"
)


def _vrcts(sigdur: float, segs: list[tuple[float, float]]) -> str:
    lines = [_VRCTS_HEAD, f'<Channel num="1" sigdur="{sigdur:.2f}" spdur="0.00"/>\n', "</ChannelList>\n"]
    lines.append('<SpeakerList>\n<Speaker ch="1" dur="0.00" gender="1" spkid="1"/>\n</SpeakerList>\n')
    lines.append("<SegmentList>\n")
    for b, e in segs:
        lines.append(f'<SpeechSegment ch="1" sconf="1.00" stime="{b:.4f}" etime="{e:.4f}" spkid="1"/>\n')
    lines.append("</SegmentList>\n</AudioDoc>\n")
    return "".join(lines)


def test_load_vrcts_hyp_fills_gaps(tmp_path: Path) -> None:
    """Speech segments become speech; leading/inter/trailing gaps (up to sigdur) become
    non-speech, contiguously covering [0, sigdur]."""
    xml = tmp_path / "hyp.xml"
    xml.write_text(_vrcts(10.0, [(1.0, 3.0), (5.0, 7.0)]))
    assert load_vrcts_hyp(xml) == [
        (0.0, 1.0, "non-speech"),
        (1.0, 3.0, "speech"),
        (3.0, 5.0, "non-speech"),
        (5.0, 7.0, "speech"),
        (7.0, 10.0, "non-speech"),
    ]


def test_load_vrcts_hyp_no_leading_or_trailing_gap(tmp_path: Path) -> None:
    """Speech that starts at 0 and ends at sigdur leaves no zero-width padding segs."""
    xml = tmp_path / "hyp.xml"
    xml.write_text(_vrcts(4.0, [(0.0, 2.0), (2.0, 4.0)]))
    assert load_vrcts_hyp(xml) == [(0.0, 2.0, "speech"), (2.0, 4.0, "speech")]


def test_load_vrcts_hyp_no_speech_is_all_nonspeech(tmp_path: Path) -> None:
    xml = tmp_path / "hyp.xml"
    xml.write_text(_vrcts(5.0, []))
    assert load_vrcts_hyp(xml) == [(0.0, 5.0, "non-speech")]


def test_load_vrcts_hyp_feeds_dcf(tmp_path: Path) -> None:
    """End-to-end: adapter output is a valid hyp for dcf() (contiguous, starts at 0)."""
    xml = tmp_path / "hyp.xml"
    xml.write_text(_vrcts(6.0, [(1.0, 4.0)]))
    ref = [(0.0, 6.0, "S")]
    score = dcf(ref, load_vrcts_hyp(xml)).by_collar(0.0)
    # speech ref [0,6]; hyp speech only [1,4] -> 3s FN out of 6s.
    assert score.pmiss == pytest.approx(3.0 / 6.0)


def test_load_vrcts_hyp_on_real_engine_output() -> None:
    """Cross-check against a REAL engine-written VRCTS (the phase4d parity fixture the
    Rust `tasks/segmentation_io.rs` writer produced): 8 speech segments, sigdur 60s ->
    17 contiguous intervals covering [0, 60], the 8 speech spans verbatim."""
    fixture = Path(__file__).resolve().parent / "reference_data" / "phase4d" / "parity_tupleA_vrcts_chan1.xml"
    iv = load_vrcts_hyp(fixture)
    speech = [(b, e) for b, e, k in iv if k == "speech"]
    assert speech == [
        (3.4118, 9.7781),
        (10.7270, 24.0868),
        (28.8641, 31.5815),
        (33.0620, 36.2782),
        (37.7527, 40.0841),
        (41.3476, 43.9589),
        (50.4009, 55.0103),
        (56.9852, 59.9999),
    ]
    assert iv[0][0] == 0.0 and iv[-1][1] == 60.0
    assert all(iv[i][1] == iv[i + 1][0] for i in range(len(iv) - 1))  # contiguous
    assert [k for _, _, k in iv] == ["non-speech", "speech"] * 8 + ["non-speech"]
