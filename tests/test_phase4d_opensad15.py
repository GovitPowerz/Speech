"""Phase 4d Task 5: OpenSAD15 corpus converter (`speech.dataprep.opensad15`).

NO LIVE ORACLE -- see the module docstring for the tier statement. Every fixture under
`tests/reference_data/phase4d/opensad15/` is hand-computed from a line-by-line reading
of the legacy `ProcessOpenSAD15Corpus.py:23-108`, independently of this module's
implementation (a second arithmetic pass, not a dump from this code). The Rust suite
additionally cross-parses one emitted XML via `load_vrcts`
(`src/rust/tests/phase4d_opensad15_vrcts.rs`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from speech.dataprep.opensad15 import convert_tab_file, process_opensad15, py2_str_float

FIXTURES = Path(__file__).resolve().parent / "reference_data" / "phase4d" / "opensad15"

# case name -> (audio_path, tab filename used to derive the stm/xml paths)
CASES = {
    "case1": "corpus/spk1_eng_001.flac",
    "case2": "corpus/session02.flac",
    "case3": "corpus/interview03.flac",
    "case4": "corpus/meeting04.flac",
    "case5": "corpus/rec01.sph",
}


# --------------------------------------------------------------------------- #
# py2_str_float: the CPython 2.7 str(float) emulation (%.12g + ".0" completion).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (143.76000000000002, "143.76"),  # the task brief's worked example
        (16.366300000000003, "16.3663"),  # case1's actual spdur (fp noise past 12 sig figs)
        (8.579999999999998, "8.58"),  # case5's actual spdur
        (1.0, "1.0"),  # bare-integral completion
        (100.0, "100.0"),
        (0.1, "0.1"),
        (-0.0, "-0.0"),
        (0.0, "0.0"),
        (1e16, "1e+16"),  # already has an exponent: no ".0" appended
        (1e-5, "1e-05"),
    ],
)
def test_py2_str_float_pinned_cases(x: float, expected: str) -> None:
    assert py2_str_float(x) == expected


def test_py2_str_float_nan_inf() -> None:
    assert py2_str_float(float("nan")) == "nan"
    assert py2_str_float(float("inf")) == "inf"
    assert py2_str_float(float("-inf")) == "-inf"


def test_py2_str_float_differs_from_python3_str_on_noisy_case() -> None:
    """The whole point of the helper: Python 3's plain str() would leak the fp noise
    that Python 2's str() truncates at 12 significant digits."""
    x = 143.76000000000002
    assert str(x) != py2_str_float(x)
    assert str(x) == "143.76000000000002"
    assert py2_str_float(x) == "143.76"


# --------------------------------------------------------------------------- #
# convert_tab_file: the 5 hand-verified fixtures.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", sorted(CASES))
def test_convert_tab_file_matches_hand_computed_fixture(case: str) -> None:
    audio_path = CASES[case]
    tab_path = f"{case}.tab"
    tab_lines = (FIXTURES / tab_path).read_text(encoding="ascii").splitlines()

    xml_bytes, stm_lines, listing_line = convert_tab_file(audio_path, tab_path, tab_lines)

    expected_xml = (FIXTURES / f"{case}_expected.xml").read_text(encoding="ascii")
    expected_stm = (FIXTURES / f"{case}_expected.stm").read_text(encoding="ascii")
    expected_listing = (FIXTURES / f"{case}_expected_listing.txt").read_text(encoding="ascii").strip("\n")

    assert xml_bytes.decode("ascii") == expected_xml
    assert "\n".join(stm_lines) + "\n" == expected_stm
    assert listing_line == expected_listing


def test_case1_duration_and_lang_taken_from_last_row_regardless_of_filter() -> None:
    """case1's last tab row is an 'NS' row (filtered OUT of Segs) with an empty lang
    field -- duration ("50.0") and the lang-fallback path are both driven by that
    filtered-out row, not by the last INCLUDED (S) row (whose own end is "45.0")."""
    tab_lines = (FIXTURES / "case1.tab").read_text(encoding="ascii").splitlines()
    _xml, _stm, listing_line = convert_tab_file(CASES["case1"], "case1.tab", tab_lines)
    fields = listing_line.split(";")
    assert fields[2] == "eng"  # lang, via the split('_')[-2] fallback on "spk1_eng_001"
    assert fields[5] == "50.0"  # duration, the NS row's end -- not the last S row's 45.0


def test_case3_lang_from_last_row_overrides_earlier_explicit_lang() -> None:
    """case3's two rows carry DIFFERENT explicit lang codes ('eng' then 'spa'); the
    final lang must be 'spa' (the physically last row), not 'eng' (the only S row)."""
    tab_lines = (FIXTURES / "case3.tab").read_text(encoding="ascii").splitlines()
    _xml, _stm, listing_line = convert_tab_file(CASES["case3"], "case3.tab", tab_lines)
    assert listing_line.split(";")[2] == "spa"


def test_case4_filter_excludes_non_s_non_ri_labels() -> None:
    """SP (len2, not RI-prefixed), N (len1, not S), SIL (len3) are all excluded; only
    the S and RI rows contribute segments/spdur."""
    tab_lines = (FIXTURES / "case4.tab").read_text(encoding="ascii").splitlines()
    xml_bytes, _stm, _listing = convert_tab_file(CASES["case4"], "case4.tab", tab_lines)
    xml = xml_bytes.decode("ascii")
    assert xml.count("<SpeechSegment ") == 2
    assert 'stime="4.0"' not in xml  # the excluded SP row
    assert 'stime="23.0"' not in xml  # the excluded N row
    assert 'stime="24.0"' not in xml  # the excluded SIL row


def test_case5_extension_length_quirk_truncates_stem() -> None:
    """audio_path[:-5] assumes a 5-char extension (".flac"); a 4-char ".sph" loses the
    last real stem character ("rec01" -> "rec0"), reproduced verbatim."""
    tab_lines = (FIXTURES / "case5.tab").read_text(encoding="ascii").splitlines()
    xml_bytes, _stm, _listing = convert_tab_file(CASES["case5"], "case5.tab", tab_lines)
    xml = xml_bytes.decode("ascii")
    assert 'name="rec0"' in xml
    assert 'path="corpus/rec01.sph"' in xml  # the path attribute is NOT truncated


def test_lang_fallback_raises_indexerror_without_two_underscore_tokens() -> None:
    """Legacy `path_leaf(audiofile[:-5]).split('_')[-2]` (`:67`) has no guard: an empty
    lang field on a stem with fewer than 2 underscore-separated tokens indexes past the
    split result and raises, matching CPython 2's identical list-indexing semantics.
    Not exercised by any of the 5 golden cases (all give lang explicitly or use a
    2+-token stem) -- a latent bug reproduced on purpose, not a defensive addition."""
    tab_lines = ["u\t1\t1.0\t2.0\tS\t0\t0\t0\t"]  # empty lang field
    with pytest.raises(IndexError):
        convert_tab_file("corpus/onetoken.flac", "x.tab", tab_lines)


# --------------------------------------------------------------------------- #
# process_opensad15: the sequential file-I/O driver.
# --------------------------------------------------------------------------- #


def test_process_opensad15_writes_xml_stm_and_listing(tmp_path: Path) -> None:
    case = "case1"
    audio_path = CASES[case]
    tab_dst = tmp_path / f"{case}.tab"
    tab_dst.write_bytes((FIXTURES / f"{case}.tab").read_bytes())

    listing = tmp_path / "in.flst"
    listing.write_text(f"{audio_path};{tab_dst};extra;ignored\n", encoding="ascii")
    out_listing = tmp_path / "out.flst"

    process_opensad15(listing, out_listing)

    xml_path = tmp_path / f"{case}.xml"
    stm_path = tmp_path / f"{case}.stm"
    assert xml_path.read_text(encoding="ascii") == (FIXTURES / f"{case}_expected.xml").read_text(encoding="ascii")
    assert stm_path.read_text(encoding="ascii") == (FIXTURES / f"{case}_expected.stm").read_text(encoding="ascii")

    out_text = out_listing.read_text(encoding="ascii")
    assert out_text.count("\n") == 1  # exactly one listing line written
    fields = out_text.strip("\n").split(";")
    assert fields[0] == audio_path
    assert fields[1] == str(tab_dst)[:-4] + ".stm"


def test_process_opensad15_stm_field_is_tabfile_stem_plus_stm(tmp_path: Path) -> None:
    case = "case2"
    audio_path = CASES[case]
    tab_dst = tmp_path / f"{case}.tab"
    tab_dst.write_bytes((FIXTURES / f"{case}.tab").read_bytes())

    listing = tmp_path / "in.flst"
    listing.write_text(f"{audio_path};{tab_dst}\n", encoding="ascii")
    out_listing = tmp_path / "out.flst"

    process_opensad15(listing, out_listing)

    expected_stm_path = str(tab_dst)[:-4] + ".stm"
    fields = out_listing.read_text(encoding="ascii").strip("\n").split(";")
    assert fields[1] == expected_stm_path
    assert Path(expected_stm_path).is_file()


def test_process_opensad15_sequential_preserves_listing_order(tmp_path: Path) -> None:
    """Documented Pool(20) -> sequential deviation: output listing order must equal
    input listing order (multiple entries, order checked explicitly)."""
    out_listing = tmp_path / "out.flst"
    listing_lines = []
    for case in ("case3", "case1", "case4"):
        tab_dst = tmp_path / f"{case}.tab"
        tab_dst.write_bytes((FIXTURES / f"{case}.tab").read_bytes())
        listing_lines.append(f"{CASES[case]};{tab_dst}")
    listing = tmp_path / "in.flst"
    listing.write_text("\n".join(listing_lines) + "\n", encoding="ascii")

    process_opensad15(listing, out_listing)

    out_lines = out_listing.read_text(encoding="ascii").strip("\n").split("\n")
    assert len(out_lines) == 3
    assert [line.split(";")[0] for line in out_lines] == [CASES["case3"], CASES["case1"], CASES["case4"]]


def test_process_opensad15_blank_listing_line_raises_indexerror(tmp_path: Path) -> None:
    """The port's earlier `if not line: continue` guard (an undocumented defensive
    divergence, caught in review and removed for legacy parity) is gone: a blank listing
    line splits to `[""]`, so the `fields[1]` read raises IndexError exactly like the
    legacy's own unguarded `tmp[1]` on an empty `treat_file('')` row -- crash-equivalent,
    not silently skipped."""
    listing = tmp_path / "in.flst"
    listing.write_text("\n", encoding="ascii")
    out_listing = tmp_path / "out.flst"
    with pytest.raises(IndexError):
        process_opensad15(listing, out_listing)


def test_process_opensad15_empty_segments_writes_empty_stm(tmp_path: Path) -> None:
    """No S/RI rows at all -> Segs and SegsExcl both stay empty -> an empty .stm file
    (matching the legacy: the file is opened and closed with nothing ever written)."""
    tab_dst = tmp_path / "novoice.tab"
    tab_dst.write_text("u\t1\t1.0\t2.0\tNS\t0\t0\t0\teng\n", encoding="ascii")
    listing = tmp_path / "in.flst"
    listing.write_text(f"corpus/novoice.flac;{tab_dst}\n", encoding="ascii")
    out_listing = tmp_path / "out.flst"

    process_opensad15(listing, out_listing)

    stm_path = tmp_path / "novoice.stm"
    assert stm_path.read_bytes() == b""
    xml_bytes = (tmp_path / "novoice.xml").read_bytes()
    assert b"<SpeechSegment " not in xml_bytes


def test_case1_spdur_is_not_a_clean_decimal_in_double_precision() -> None:
    """Non-vacuity check for the py2_str_float pin: case1's raw spdur genuinely carries
    fp noise past 12 significant digits (this isn't a synthetic-only concern)."""
    total = (9.7781 - 3.4118) + (25.0 - 20.0) + (45.0 - 40.0)
    assert repr(total) == "16.366300000000003"  # python3 default str/repr leaks the noise
    assert py2_str_float(total) == "16.3663"
