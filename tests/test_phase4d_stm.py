"""Phase 4d Task 7: STM normalizer byte-oracle gates.

`normalize_stm_light` is pinned byte-for-byte against `light_*.{in,out}` -- both
committed by `scripts/extract_phase4d_fixtures.py --stm-fixtures`, run through the REAL
`/usr/bin/perl norm_stm_pkt_light.pl` via `tools/perl_oracle/run_norm.sh`. The `full_*`
pairs pin the comment/blank-line prefix of `norm_stm_pk_cts_all_trans_03a.pl` the same
way; `normalize_stm_full`'s content-line bail (`NormalizerUnavailable`, the lost
norm-tagger/norm-parser LIMSI binaries) has no live-oracle backing (see the module
docstring) and is pinned here by plain assertions instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from speech.dataprep.stm_normalize import NormalizerUnavailable, normalize_stm_full, normalize_stm_light

STM_DIR = Path(__file__).resolve().parent / "reference_data" / "phase4d" / "stm"

LIGHT_CASES = ["case1", "case2", "case3", "case4", "case5"]
FULL_CASES = ["case1", "case2", "case3"]


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


# --- byte-oracle parity ---------------------------------------------------------------


@pytest.mark.parametrize("case", LIGHT_CASES)
def test_normalize_stm_light_matches_perl_oracle(case: str) -> None:
    in_lines = _read_lines(STM_DIR / f"light_{case}.in")
    want = (STM_DIR / f"light_{case}.out").read_text()
    got = "".join(normalize_stm_light(in_lines))
    assert got == want, f"light_{case}: port diverges from the real perl script"


@pytest.mark.parametrize("case", FULL_CASES)
def test_normalize_stm_full_prefix_matches_perl_oracle(case: str) -> None:
    """The full_* fixtures cover ONLY the comment-passthrough + blank-line-drop prefix
    (see stm_normalize.py's module docstring for why content lines are excluded from
    this oracle)."""
    in_lines = _read_lines(STM_DIR / f"full_{case}.in")
    want = (STM_DIR / f"full_{case}.out").read_text()
    got = "".join(normalize_stm_full(in_lines))
    assert got == want, f"full_{case}: port diverges from the real perl script"


# --- light: targeted category checks (redundant with the fixtures above, but pin the
# specific behavior each fixture line is there to exercise, by name) ------------------


def test_light_comment_lines_passthrough_verbatim() -> None:
    lines = [";; a comment\n", ";;no-space comment\n"]
    assert normalize_stm_light(lines) == lines


def test_light_blank_line_dropped() -> None:
    assert normalize_stm_light(["   \n"]) == []
    assert normalize_stm_light(["\n"]) == []


def test_light_blank_line_without_trailing_newline_is_not_dropped() -> None:
    """Legacy quirk: `/^\\s*\\n/` requires an ACTUAL embedded newline to match, so a
    final unterminated all-whitespace "line" is NOT recognized as blank and falls
    through to the content path (producing a degenerate ignore_time_segment_in_scoring
    record, not a silent drop). Python's re.match reproduces this exactly (no special
    casing needed): `^\\s*\\n` cannot match a string with no `\\n` in it at all."""
    out = normalize_stm_light(["   "])
    assert out != [], "an unterminated blank line must NOT be silently dropped"


def test_light_ignore_line_passthrough_verbatim() -> None:
    line = "file1_ignore_ 1 spkA 1.00 2.00 <o,f0> ignore_ this stays\n"
    assert normalize_stm_light([line]) == [line]


def test_light_head_apostrophe_becomes_underscore() -> None:
    out = normalize_stm_light(["f1_1'2 1 spkA 0.00 1.00 <o,f0> mot\n"])
    assert out[0].startswith("f1_1_2 ")


def test_light_text_apostrophe_is_preserved() -> None:
    """Unlike the head, `light`'s text apostrophe substitution never fires -- it is
    only applied to `$head` (:22)."""
    out = normalize_stm_light(["f1 1 spkA 0.00 1.00 <o,f0> l'ami\n"])
    assert "l'ami" in out[0]


def test_light_filler_class1_adjacent_tokens_alternate() -> None:
    """perl s///g non-overlap: adjacent filler tokens alternate matched/unmatched
    because a match's trailing whitespace is consumed, denying the next token its own
    leading boundary."""
    out = normalize_stm_light(["f1 1 s 0.00 1.00 <o> hm mm eh huhum hum\n"])[0]
    assert "{fw} mm {fw} huhum {fw}" in out


def test_light_filler_class2_adjacent_tokens_alternate() -> None:
    out = normalize_stm_light(["f1 1 s 0.00 1.00 <o> ee oo ii aa m em d\n"])[0]
    assert "{fw} oo {fw} aa {fw} em {fw}" in out


def test_light_partial_word_both_directions() -> None:
    out = normalize_stm_light(["f1 1 s 0.00 1.00 <o> mot- suite -mot\n"])[0]
    assert "(mot-) suite (-mot)" in out


def test_light_lone_hyphen() -> None:
    out = normalize_stm_light(["f1 1 s 0.00 1.00 <o> a - b\n"])[0]
    assert "a (-) b" in out


def test_light_punctuation_stripped() -> None:
    out = normalize_stm_light(['f1 1 s 0.00 1.00 <o> a.b!c?d,e"f:g;h\\i/j#k*l^m\n'])[0]
    assert "abcdefghijklm" in out.replace(" ", "")


def test_light_empty_text_becomes_ignore_time_segment() -> None:
    out = normalize_stm_light(["f1 1 s 0.00 1.00 <o>\n"])[0]
    assert out == "f1 1 s 0.00 1.00 <o> ignore_time_segment_in_scoring\n"


def test_light_whitespace_only_text_becomes_ignore_time_segment() -> None:
    out = normalize_stm_light(["f1 1 s 0.00 1.00 <o> . ! ?\n"])[0]
    assert out == "f1 1 s 0.00 1.00 <o> ignore_time_segment_in_scoring\n"


# --- full: comment/blank prefix + the typed bail --------------------------------------


def test_full_comment_lines_passthrough_verbatim() -> None:
    lines = [";; a comment\n", ";;no-space comment\n"]
    assert normalize_stm_full(lines) == lines


def test_full_blank_lines_dropped() -> None:
    assert normalize_stm_full(["\n", "   \n"]) == []


def test_full_content_line_raises_normalizer_unavailable() -> None:
    """The number-normalization arm (norm-tagger|norm-parser) would engage for ANY
    content line -- the shell-out at 03a:109 is unconditional, no branch gates it."""
    with pytest.raises(NormalizerUnavailable):
        normalize_stm_full(["f1 1 spkA 0.00 1.00 <o,f0> hello world\n"])


def test_full_content_line_raise_carries_prefix_state() -> None:
    """White-box check (no live oracle -- see the module docstring): the exception
    carries the head/text as computed by the pure-regex prefix up to the bail point."""
    with pytest.raises(NormalizerUnavailable) as exc_info:
        normalize_stm_full(["f1 1 spkA 0.00 1.00 <o,f0> hello world\n"])
    assert exc_info.value.head == "f1 1 spkA 0.00 1.00 <o,f0>"
    # :20's DOUBLE-SPACE join divergence from light's single-space join survives (no
    # filler pattern here touches plain "hello"/"world"); :80's single-leading-space
    # strip (no `+`, unlike light's :47) removes exactly the ONE leading pad space,
    # leaving the trailing pad space untouched.
    assert exc_info.value.text == "hello  world "


def test_full_content_line_after_comments_still_raises() -> None:
    """A representative mixed-content line: comments pass, then the first genuine
    content line still bails (no partial output is returned)."""
    with pytest.raises(NormalizerUnavailable):
        normalize_stm_full([";; header\n", "f1 1 spkA 0.00 1.00 <o,f0> euh bonjour\n"])
