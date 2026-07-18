"""Phase 6 Task 4: DCF scorer gates.

`dcf()` is pinned value-for-value against the REAL vendored NIST `scoreFile_SAD.pl` (v2.1)
via the committed `tests/reference_data/phase6/dcf/*.json` goldens -- each produced by
`scripts/extract_phase6_fixtures.py` running the crafted ref/hyp intervals through
`/usr/bin/perl` (`tools/perl_oracle/run_sad_scorer.sh`). The perl prints Prob_Miss /
Prob_FalseAlarm / DCF at `%7.5f` (5 decimals); the port reproduces the scorer's exact
double-precision segment accumulation, so pinning is 5-decimal string equality (`{:.5f}`)
-- a ~5e-6 string-equality CLASS, not literal float bit-equality, but safe here because the
underlying accumulation is bit-identical to the perl's (no value sits within that ~5e-6 window
of a different true result, so the rounding never masks a real difference). CI never runs the
perl; it only consumes these goldens.

The `load_tab_ref` / `load_vrcts_hyp` adapters are pinned separately on crafted inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from speech.drivers.test import write_scores
from speech.evaluate import CollarScore, DcfReport, Interval, cavg, dcf, lid_error, load_tab_ref, load_vrcts_hyp, load_vrcts_ref, read_scr_scores

DCF_DIR = Path(__file__).resolve().parent / "reference_data" / "phase6" / "dcf"
GOLDENS = sorted(p for p in DCF_DIR.glob("*.json") if p.name != "dcf_manifest.json")

# Licensed LRE15 scoring artifacts (315 MB, gitignored, absent in CI) -- a SEPARATE
# gate from `tests.conftest.CORPUS_ROOT` (which keys on the LRE03/07 audio corpus).
# The Cavg cross-check below reads a real 2015 confusion matrix from here at runtime;
# NOTHING from it is committed (no filenames, no counts, no Cavg values).
SCORING_LRE15_CONF = Path(__file__).resolve().parent.parent / "data" / "Scoring_LRE15" / "dev_conf.csv"
requires_scoring_lre15 = pytest.mark.skipif(
    not SCORING_LRE15_CONF.is_file(),
    reason=f"licensed Scoring_LRE15 artifacts absent at {SCORING_LRE15_CONF}",
)


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


def test_load_vrcts_ref_maps_to_nist_kinds(tmp_path: Path) -> None:
    """The SAD-arm reference adapter: `load_vrcts_hyp`'s contiguous intervals with the kinds
    remapped to the NIST ref labels (speech -> S, non-speech -> NS) -- the same span/gap
    structure, so it feeds dcf()'s reference side directly."""
    xml = tmp_path / "ref.xml"
    xml.write_text(_vrcts(10.0, [(1.0, 3.0), (5.0, 7.0)]))
    assert load_vrcts_ref(xml) == [
        (0.0, 1.0, "NS"),
        (1.0, 3.0, "S"),
        (3.0, 5.0, "NS"),
        (5.0, 7.0, "S"),
        (7.0, 10.0, "NS"),
    ]
    # feeds dcf's reference side (contiguous, starts at 0, raw NIST kinds).
    hyp = load_vrcts_hyp(xml)  # score the ref against itself -> zero error
    s = dcf(load_vrcts_ref(xml), hyp).by_collar(0.0)
    assert s.pmiss == 0.0 and s.pfa == 0.0


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


# --- lid_error: argmax error, compare_scores.m semantics ------------------------------
#
# SOURCE (legacy/Optimizer_V6.2.2/compare_scores.m:19-20):
#   [val,pos] = max(scores_test,[],2);
#   sum(ref_test ~= pos)/length(ref_test)*100
# The error is a PERCENTAGE (mismatches/total*100), not a fraction; MATLAB max(...,[],2)
# takes the FIRST maximum per row, which numpy argmax matches. The pso_out(ii) per-detector
# weighting (:10) is upstream of the argmax and the patternnet(40) fusion (:50-58) is out
# of scope -- lid_error consumes the already-assembled (n_files, n_langs) score matrix.


def test_lid_error_known_confusion_is_a_percentage() -> None:
    """A hand-built 5-file, 3-lang confusion: refs [0,0,1,1,2], argmax [0,0,1,0,1].
    Mismatches = files 3 (ref 1 -> pred 0) and 4 (ref 2 -> pred 1) = 2 of 5. The legacy
    formula is mismatches/total*100 = 2/5*100 = 40.0 (a PERCENT, not the fraction 0.4)."""
    scores = np.array(
        [
            [0.9, 0.1, 0.0],  # ref 0 -> argmax 0 (hit)
            [0.7, 0.2, 0.1],  # ref 0 -> argmax 0 (hit)
            [0.1, 0.8, 0.1],  # ref 1 -> argmax 1 (hit)
            [0.6, 0.4, 0.0],  # ref 1 -> argmax 0 (MISS)
            [0.2, 0.7, 0.1],  # ref 2 -> argmax 1 (MISS)
        ]
    )
    refs = np.array([0, 0, 1, 1, 2])
    assert lid_error(scores, refs) == pytest.approx(40.0)


def test_lid_error_perfect_is_zero() -> None:
    scores = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    refs = np.array([0, 1, 0])
    assert lid_error(scores, refs) == 0.0


def test_lid_error_all_wrong_is_hundred() -> None:
    scores = np.array([[0.1, 0.9], [0.9, 0.1]])
    refs = np.array([0, 1])
    assert lid_error(scores, refs) == pytest.approx(100.0)


def test_lid_error_tie_takes_first_maximum_matching_matlab_max() -> None:
    """MATLAB max(...,[],2) and numpy argmax both return the FIRST maximum on a tie.
    Row [0.5, 0.5, 0.1] -> argmax 0. So ref 0 is a hit (0% error) and ref 1 is a miss
    (100% error) -- the tie resolves to the lower column index, deterministically."""
    scores = np.array([[0.5, 0.5, 0.1]])
    assert lid_error(scores, np.array([0])) == 0.0
    assert lid_error(scores, np.array([1])) == pytest.approx(100.0)


def test_lid_error_index_base_agnostic() -> None:
    """The mismatch count is invariant to the indexing base as long as refs and argmax
    share it (the legacy uses 1-based for both; the port uses numpy's 0-based). Same
    scores, refs shifted by a constant that keeps them consistent -> same error."""
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.1, 0.8]])
    assert lid_error(scores, np.array([0, 2])) == 0.0  # 0-based hits


# --- read_scr_scores: adapter over the .scr writer ------------------------------------
#
# The port's drivers/test.py::write_scores emits, per file, one `filename lang-dial score`
# line per class, SOFTMAXED (exp(s/100)/max(1e-3,sum)) and sorted DESCENDING by score.
# read_scr_scores inverts it: map each line's `lang-dial` label back to its class-key
# COLUMN (labels are not in class order -- they are score-sorted), returning the softmax
# probabilities in class-key column order. lid_error consumes these softmaxed values
# directly: softmax is strictly monotone (exp increasing, the per-row sum -- clamped at
# 1e-3 -- a positive per-row constant), so argmax(softmax(s)) == argmax(s); the .scr
# values and the raw scores give the identical argmax and hence the identical lid_error.


def test_read_scr_scores_round_trips_write_scores(tmp_path: Path) -> None:
    class_keys = ["ara_non", "chi_non", "eng_non"]
    raw = np.array([10.0, 50.0, 30.0])  # raw LID scores, column order == class_keys order
    scr_dir = tmp_path / "scores"
    scr_dir.mkdir()
    (scr_dir / "f1.wav.scr").write_text(write_scores(raw, class_keys, "f1.wav"))

    scores, files = read_scr_scores(scr_dir, class_keys)

    assert files == ["f1.wav"]  # the .scr stem (name minus the ".scr" suffix)
    assert scores.shape == (1, 3)
    # read reconstructs the writer's softmax IN CLASS-KEY COLUMN ORDER (not score order).
    expected = np.exp(raw / 100.0)
    expected = expected / expected.sum()
    assert scores[0] == pytest.approx(expected)
    # monotone invariance: the softmaxed .scr argmax == the raw-score argmax (col 1, chi).
    assert int(scores[0].argmax()) == 1 == int(raw.argmax())


def test_read_scr_scores_multiple_files_sorted_by_name(tmp_path: Path) -> None:
    class_keys = ["ara_non", "chi_non"]
    scr_dir = tmp_path / "s"
    scr_dir.mkdir()
    (scr_dir / "b.scr").write_text(write_scores(np.array([9.0, 1.0]), class_keys, "b"))
    (scr_dir / "a.scr").write_text(write_scores(np.array([1.0, 9.0]), class_keys, "a"))

    scores, files = read_scr_scores(scr_dir, class_keys)

    assert files == ["a", "b"]  # sorted by .scr filename, deterministic
    assert int(scores[0].argmax()) == 1  # a -> chi
    assert int(scores[1].argmax()) == 0  # b -> ara


def test_read_scr_scores_two_char_dial_underscore_label(tmp_path: Path) -> None:
    """The writer's `key[-3:]` dial slice renders a 2-char dial with a leading underscore
    (e.g. 'aaa_11' -> label 'aaa-_11', the load-bearing quirk in _class_keys). read must
    invert the SAME label rule, or the column mapping breaks."""
    class_keys = ["aaa_11", "bbb_22"]
    scr_dir = tmp_path / "s"
    scr_dir.mkdir()
    (scr_dir / "x.scr").write_text(write_scores(np.array([9.0, 1.0]), class_keys, "x"))

    scores, files = read_scr_scores(scr_dir, class_keys)

    assert files == ["x"]
    assert int(scores[0].argmax()) == 0  # 'aaa-_11' maps back to column 0


def test_read_scr_scores_feeds_lid_error(tmp_path: Path) -> None:
    """End to end: write three .scr files, read them, score with lid_error. The third
    file's truth is ara (0) but it scored highest on chi (1) -> 1 of 3 wrong = 33.33%."""
    class_keys = ["ara_non", "chi_non", "eng_non"]
    scr_dir = tmp_path / "s"
    scr_dir.mkdir()
    (scr_dir / "0.scr").write_text(write_scores(np.array([9.0, 1.0, 1.0]), class_keys, "0"))
    (scr_dir / "1.scr").write_text(write_scores(np.array([1.0, 9.0, 1.0]), class_keys, "1"))
    (scr_dir / "2.scr").write_text(write_scores(np.array([1.0, 9.0, 1.0]), class_keys, "2"))

    scores, files = read_scr_scores(scr_dir, class_keys)
    refs = np.array([0, 1, 0])  # truths: ara, chi, ara

    assert lid_error(scores, refs) == pytest.approx(100.0 / 3.0)


def test_read_scr_scores_ambiguous_labels_raise(tmp_path: Path) -> None:
    """Two class keys that collapse to the same lang-dial label make the round trip
    lossy -- read cannot know which column a line belongs to, so it refuses."""
    scr_dir = tmp_path / "s"
    scr_dir.mkdir()
    with pytest.raises(ValueError, match="ambiguous"):
        read_scr_scores(scr_dir, ["abc_xyz", "abcxxyz"])  # both -> 'abc-xyz'


# --- cavg: NIST LRE closed-set identification Cavg ------------------------------------
#
# SOURCE (data/Scoring_LRE15/Scripts/ComputeCavgNIST.py:167-192, the 2015 oracle):
#   Cavg = mean_i [ 0.5*Pmiss(i) + (1/(N-1)) * sum_{j!=i} 0.5*Pfa(i,j) ]
# with max(1.0, count) denominator guards and P_target=0.5 (LRE convention, C_Miss=C_FA=1).
# cavg() implements the standard NIST pairwise form grouped by DETECTOR t (P_Fa(t,n) =
# #(ref n -> pred t)/#n); the script groups by TRUE language i (#(i->j)/#i). Both are the
# same complete pairwise sum over ordered pairs, so they yield the same scalar -- the
# corpus-gated test cross-checks exactly that on a real 2015 confusion matrix.


def test_cavg_hand_derived_three_language_case() -> None:
    """N=3, hand-derived. Confusion C (rows=true, cols=argmax), reconstructed below:
        ref 0: 2 files -> pred 0, 0            row 0 = [2, 0, 0], sum 2
        ref 1: 2 files -> pred 1, 0            row 1 = [1, 1, 0], sum 2
        ref 2: 1 file  -> pred 1               row 2 = [0, 1, 0], sum 1
    Pmiss(t) = (rowsum_t - C[t][t]) / rowsum_t:
        Pmiss(0)=0/2=0    Pmiss(1)=1/2=0.5    Pmiss(2)=1/1=1.0
    Pfa(t,n) = C[n][t]/rowsum_n, averaged over n!=t and /(N-1=2):
        t=0: (C[1][0]/2 + C[2][0]/1)/2 = (0.5 + 0)/2 = 0.25
        t=1: (C[0][1]/2 + C[2][1]/1)/2 = (0   + 1)/2 = 0.5
        t=2: (C[0][2]/2 + C[1][2]/2)/2 = (0   + 0)/2 = 0
    term(t) = 0.5*Pmiss(t) + 0.5*Pfa_avg(t):
        t=0: 0.5*0   + 0.5*0.25 = 0.125
        t=1: 0.5*0.5 + 0.5*0.5  = 0.5
        t=2: 0.5*1.0 + 0.5*0    = 0.5
    Cavg = (0.125 + 0.5 + 0.5)/3 = 1.125/3 = 0.375."""
    scores = np.array(
        [
            [0.9, 0.1, 0.0],  # ref 0 -> 0
            [0.8, 0.2, 0.0],  # ref 0 -> 0
            [0.1, 0.9, 0.0],  # ref 1 -> 1
            [0.6, 0.4, 0.0],  # ref 1 -> 0
            [0.2, 0.7, 0.1],  # ref 2 -> 1
        ]
    )
    refs = np.array([0, 0, 1, 1, 2])
    assert cavg(scores, refs) == pytest.approx(0.375)


def test_cavg_perfect_is_zero() -> None:
    scores = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    refs = np.array([0, 1, 0, 1])
    assert cavg(scores, refs) == pytest.approx(0.0)


def test_cavg_ptarget_constant_is_load_bearing() -> None:
    """Perturbing P_target off the LRE 0.5 convention changes the score (the mutation
    battery target): with an imperfect classifier the Cavg must move."""
    scores = np.array([[0.9, 0.1, 0.0], [0.6, 0.4, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    refs = np.array([0, 1, 2, 2])
    base = cavg(scores, refs)  # p_target=0.5
    assert cavg(scores, refs, p_target=0.9) != pytest.approx(base)


def test_cavg_zero_trial_language_contributes_zero_not_nan() -> None:
    """A language column with no reference trials (subset gates need not cover all 12)
    must not produce NaN: the max(1.0, count) guards give it a 0 contribution, and it
    still counts toward N in the outer 1/N average (matching ComputeCavgNIST.py)."""
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])  # lang 2 never a truth
    refs = np.array([0, 1])
    val = cavg(scores, refs)
    assert np.isfinite(val)
    assert val == pytest.approx(0.0)  # a perfect 2-of-3-language classifier


def _read_confusion_csv(path: Path) -> np.ndarray:
    rows = [[float(x) for x in line.split(",")] for line in path.read_text().splitlines() if line.strip()]
    c = np.array(rows, dtype=np.float64)
    assert c.shape[0] == c.shape[1], f"confusion matrix not square: {c.shape}"
    return c


def _nist_cavg_from_confusion(c: np.ndarray) -> float:
    """Faithful transcription of ComputeCavgNIST.py:167-192 ("Cavg max all dialects", the
    non-cluster path) for CLOSED-SET decisions, consuming the confusion matrix directly:
    InClass[i]=rowsum_i, Pmiss[i]=rowsum_i-C[i][i], Pfa[i][j]=C[i][j] (j!=i; the diagonal
    is a correct call, never a false alarm), Other[i][j]=rowsum_i. Grouped by TRUE
    language i -- a DIFFERENT grouping than cavg()'s per-detector one, so agreement is a
    real cross-check, not a shared path. The 0.5 factors are the script's literal
    constants (P_target and P_nontarget, both 0.5 at the LRE convention)."""
    n = c.shape[0]
    rowsum = c.sum(axis=1)
    cavg_all = 0.0
    for i in range(n):
        pmiss = 0.5 * (rowsum[i] - c[i, i]) / max(1.0, rowsum[i])
        pfa = 0.0
        for j in range(n):
            if j == i:
                continue  # Pfa[i][i] is never accumulated in the legacy (only j != i)
            pfa += 0.5 * c[i, j] / max(1.0, rowsum[i])
        pfa /= n - 1
        cavg_all += pmiss + pfa
    return float(cavg_all / n)


@requires_scoring_lre15
def test_cavg_matches_nist_oracle_on_real_dev_confusion() -> None:
    """Corpus-gated cross-check (Scoring_LRE15, gitignored -- NOTHING committed, the
    matrix and the expected value are both computed at runtime): cavg(), via argmax on
    reconstructed one-hot scores, reproduces the 2015 ComputeCavgNIST.py "Cavg max all
    dialects" number on the REAL dev confusion matrix. cavg() groups false alarms by
    DETECTOR and divides by the non-target's count; the oracle groups by TRUE language
    and divides by the true language's count -- the same complete pairwise sum, so
    equality validates the convention (P_target=0.5, the (N-1) normalization, the
    max(1,count) guards), not a shared code path."""
    c = _read_confusion_csv(SCORING_LRE15_CONF)
    n = c.shape[0]
    # reconstruct (scores, refs) whose argmax confusion == c: one file per count, a
    # one-hot score row peaking at the predicted column (unique argmax, no ties).
    scores_rows: list[np.ndarray] = []
    refs: list[int] = []
    for i in range(n):
        for j in range(n):
            row = np.zeros(n)
            row[j] = 1.0
            for _ in range(int(round(c[i, j]))):
                scores_rows.append(row)
                refs.append(i)
    scores = np.array(scores_rows)
    got = cavg(scores, np.array(refs))
    want = _nist_cavg_from_confusion(c)
    assert got == pytest.approx(want, abs=1e-12)
