"""STM transcript normalization. Ported from two legacy Perl scripts (vendored read-only
at `legacy/.../norm_stm_pkt_light.pl` and `legacy/.../norm_stm_pk_cts_all_trans_03a.pl`),
pinned against a LIVE BYTE ORACLE (`/usr/bin/perl` running the REAL vendored scripts via
`tools/perl_oracle/run_norm.sh`) -- unlike `opensad15.py`/`augment.py` (Python 2, no
interpreter left), Perl 5.34 is present on this host, so this module gets the strongest
validation tier available in Phase 4d dataprep.

Both scripts share the same per-line shape: a `while (<STDIN>)` loop reading one STM
record per line, `head = fields[0:6]` / `text = fields[6:]` split, then a fixed cascade
of `s///g` regex substitutions applied to `text` IN SOURCE LINE ORDER (never reordered --
several rules depend on an earlier rule having already run, e.g. the paren-reattachment
rules consume `(-)` tokens the paren-collapse rule produced two lines earlier).

`normalize_stm_light` (`norm_stm_pkt_light.pl`, 57 live lines, no external dependency)
is fully self-contained and BYTE-ORACLE-VERIFIED end to end: every `light_*` fixture
under `tests/reference_data/phase4d/stm/` is the real script's stdout, run twice per
case to confirm determinism (`scripts/extract_phase4d_fixtures.py --stm-fixtures`).

`normalize_stm_full` (`norm_stm_pk_cts_all_trans_03a.pl`, 210 lines) is different: EVERY
content line (anything past the comment/blank checks) hits an UNCONDITIONAL shell-out at
line 109 -- `$text=`echo "$text" | norm-tagger | norm-parser --lang=spa`;` -- with no
branch between the `chomp` (:15) and that backtick call. The `norm-tagger`/`norm-parser`
LIMSI binaries are LOST (not merely absent from this host; the brief states they no
longer exist anywhere), so this arm can never run for real. Verified live: in an
environment where those binaries are simply missing (this one), perl's backticks do NOT
raise -- the shell prints "command not found" to stderr, stdout is empty, and the script
continues with `$text = ""`, silently producing a broken-environment artifact rather than
crashing. Treating that artifact as an oracle would pin an accident of this host, not
legacy behavior, so `normalize_stm_full` instead raises `NormalizerUnavailable` at the
point equivalent to line 109 -- a TYPED BAIL, not a reproduction of the accidental empty-
string fallback.

The split is CONTIGUOUS, not a prefix-plus-reachable-suffix: since the shell-out is
unconditional for every content line, lines 110-202 (the `$type=~/am/` punctuation branch,
the illegal-char/accented-letter cleanup, etc.) are unreachable from the port's
perspective and are NOT transcribed. The pure-regex PREFIX (:17-105 -- head/text split,
apostrophe substitution, the filler/bracket/punctuation cascade) IS transcribed
(`_full_prefix`, mirroring the brief's "port everything up to that point that is pure
regex"), and its result is attached to the raised exception (`.head`/`.text`) for
inspection -- but it has NO live-oracle backing: the real script never exposes that
intermediate value for a content line (it always reaches the shell-out first), so unlike
`normalize_stm_light` this transcription is verified only by hand against the Perl
source, not against a live run. See IMPROVEMENTS.md and `tests/test_phase4d_stm.py`.

Legacy quirks reproduced verbatim (see IMPROVEMENTS.md for the full list):
  * `light`'s two filler alternation rules (`(^|\\s)(euh|hm+|...)(\\s|$)` and the second
    hesitation class) exhibit perl `s///g`'s classic non-overlap skip: a match's trailing
    `\\s` is CONSUMED, so an immediately adjacent token loses its own leading boundary and
    survives unmatched -- every other token in a run of adjacent fillers alternates
    matched/unmatched. Python's `re.sub` has the identical non-overlap semantics (verified
    against the real perl output on `light_case2`), so a direct `re.sub` transcription
    reproduces it with no special-casing.
  * `full`'s filler line `\\[[rien]\\]` (:49) is a BUG: `[rien]` is a Perl CHARACTER CLASS
    (matches any ONE of r/i/e/n), not the literal word "rien" the author evidently meant.
    Reproduced as `r"\\[[rien]\\]"` (unreachable in practice, see above).
  * `full`'s head/text split uses a DOUBLE-SPACE join (`join '  ', ...`, :20) where
    `light` uses a single space (:20 in `light`) -- a genuine cross-script divergence,
    not a typo in this port.
  * `full`'s leading-space strip (`s/^ //`, :80) removes AT MOST ONE leading space
    (no `+` quantifier), where `light`'s (`s/^ +//`, :47) removes ALL of them -- another
    genuine cross-script divergence.
"""

from __future__ import annotations

import re

# --- normalize_stm_light (norm_stm_pkt_light.pl) -------------------------------------

_COMMENT_RE = re.compile(r"^;;")
_BLANK_RE = re.compile(r"^\s*\n")
_IGNORE_RE = re.compile(r"ignore_")
_APOSTROPHE_RE = re.compile(r"'")

# One (pattern, replacement) tuple per legacy `s///g` line, IN SOURCE ORDER (:26-48).
_LIGHT_SUBS: list[tuple[str, str]] = [
    (r"\[-?respiro-?\]", " {breath} "),  # :26
    (r"\[-?noise=breath-?\]", " {breath} "),  # :27
    (r"(^|\s)(euh|hm+|mm+|eh|huhum|hum)(\s|$)", " {fw} "),  # :29
    (r"(^|\s)(ee|oo|ii|aa|m|em|d)(\s|$)", " {fw} "),  # :31
    (r"%(e|o|i|a|m|em)(\s|$)", " {fw} "),  # :33
    (r"%[emoa]+( |$)", " {fw} "),  # :34
    (r"&[^ ]+( |$)", " {fw} "),  # :35
    (r"\[[^\]]+\]", " "),  # :37
    (r"\([^)]*\)", "(-)"),  # :39
    ('[.!?,":;\\\\/#*^]', ""),  # :41 -- literal (no backslash regex metachars needed)
    (r"([a-zA-Z]+-) ", r"(\1) "),  # :42
    (r" (-[a-zA-Z]+)", r" (\1)"),  # :43
    (r" - ", " (-) "),  # :44
    (r" \(-?\)([^ ]+) ", r" (-\1) "),  # :45
    (r" ([^( ]+)\(-?\) ", r" (\1-) "),  # :46
    (r"^ +", ""),  # :47
    (r" +", " "),  # :48
]
_LIGHT_SUBS_COMPILED = [(re.compile(p), r) for p, r in _LIGHT_SUBS]

_EMPTY_TEXT_RE = re.compile(r"^\s*$")


def normalize_stm_light(lines: list[str]) -> list[str]:
    """Port of `norm_stm_pkt_light.pl` (57 live lines), byte-oracle-verified against the
    real script (`tests/reference_data/phase4d/stm/light_*.{in,out}`).

    `lines` are raw records as read from a file object (`open(...).readlines()` /
    `list(f)` semantics) -- each entry carries its own trailing `\\n` except possibly the
    last, load-bearing: the legacy blank-line check (`/^\\s*\\n/`) requires an ACTUAL
    embedded newline character to match, so a final unterminated all-whitespace line is
    NOT treated as blank (falls through to the content path, matching the real script).
    Returned lines are similarly newline-terminated (comment/`ignore_` lines are echoed
    verbatim, preserving whatever newline -- or lack of one -- the input line carried;
    computed lines always get exactly one trailing `\\n`, per the legacy's `print "$head
    $text\\n"`).
    """
    out: list[str] = []
    for raw in lines:
        if _COMMENT_RE.match(raw):  # :11
            out.append(raw)
            continue
        if _BLANK_RE.match(raw):  # :12
            continue
        if _IGNORE_RE.search(raw):  # :13
            out.append(raw)
            continue

        line = raw[:-1] if raw.endswith("\n") else raw  # :15 chomp
        tokens = line.split()  # :17
        head = " ".join(tokens[0:6])  # :19
        text = " ".join(tokens[6:])  # :20

        head = _APOSTROPHE_RE.sub("_", head)  # :22

        text = " " + text + " "  # :25
        for pattern, repl in _LIGHT_SUBS_COMPILED:
            text = pattern.sub(repl, text)

        if _EMPTY_TEXT_RE.match(text):  # :50
            text = "ignore_time_segment_in_scoring"  # :52

        out.append(f"{head} {text}\n")  # :55
    return out


# --- normalize_stm_full (norm_stm_pk_cts_all_trans_03a.pl) ---------------------------


class NormalizerUnavailable(Exception):
    """Raised by `normalize_stm_full` when a content line reaches the legacy
    `norm-tagger`/`norm-parser` shell-out (03a:109) -- those LIMSI binaries are lost, so
    the number-normalization arm can never run. Carries the (`head`, `text`) computed by
    the pure-regex prefix (03a:17-105) up to that point -- NOT live-oracle-verified (see
    the module docstring): the real script never exposes this value for a content line.
    """

    def __init__(self, head: str, text: str) -> None:
        super().__init__("norm-tagger|norm-parser unavailable (03a:109 shell-out, lost LIMSI binaries)")
        self.head = head
        self.text = text


_BLANK_RE2 = re.compile(r"^\s*$")  # 03a's SECOND, redundant blank check (:14)

# One (pattern, replacement) tuple per legacy `s///g` line, IN SOURCE ORDER (:31-105).
# NO live oracle: see the module docstring -- this prefix is unreachable-in-practice
# (every content line bails right after it runs), transcribed by hand against the
# source, not cross-run.
_FULL_PREFIX_SUBS: list[tuple[str, str]] = [
    (r" &\w+ ", " {fw} "),  # :31
    (r" ah ", " {fw} "),  # :32
    (r" bah ", " {fw} "),  # :33
    (r" buah ", " {fw} "),  # :34
    (r" buf ", " {fw} "),  # :35
    (r" eh ", " {fw} "),  # :36
    (r" huy ", " {fw} "),  # :37
    (r" oh ", " {fw} "),  # :38
    (r" pum ", " {fw} "),  # :39
    (r" uf ", " {fw} "),  # :40
    (r" uh ", " {fw} "),  # :41
    (r"&", " "),  # :43
    # :45 -- perl's `(%hesitation)` is a CAPTURE GROUP around the literal text
    # "%hesitation" (unescaped parens are grouping syntax, not literal chars).
    (r"%hesitation", " {fw} "),
    (r"%%[%]*", " {fw} "),  # :46
    (r"%\w+", " {fw} "),  # :47
    (r"&\w+", " {fw} "),  # :48
    (r"\[[rien]\]", " {breath} "),  # :49 -- BUG: char class, not literal "rien"
    (r"\[noise=breath\]", " {breath} "),  # :50
    (r"\[-noise=breath\]", " {breath} "),  # :51
    (r"\[rire\]", " {laugh} "),  # :52
    (r"\{lipsmack\}", " "),  # :53
    (r"\{laughter\}", " "),  # :54
    (r"\{laugh\}", " "),  # :55
    (r"\{cough\}", " "),  # :56
    (r"\^", " "),  # :57
    (r"\x20\xa0", ""),  # :60 -- literal SPACE + NBSP two-char sequence
    (r"<s>", ""),  # :63
    (r"</s>", ""),  # :64
    (r"<s/>", ""),  # :65
    (r"\[[^\]-]+\]", " "),  # :68
    (r"\[-?[^\]-]+-?\]", " "),  # :69
    (r"\[[^\]]+\]", " "),  # :70
    (r"\[\]", " "),  # :72
    (r"([a-zA-Z]+)- ", r"\1 "),  # :77 -- drops the hyphen (no parens, unlike light)
    (r" -([a-zA-Z]+)", r" \1"),  # :78 -- drops the hyphen
    (r" - ", " (-) "),  # :79
    (r"^ ", ""),  # :80 -- ONE leading space only (no +), unlike light's :47
    (r"\+", " "),  # :81
    (r"\}\)", "}"),  # :82
    (r"profe\(sor\]", "profe(sor)"),  # :85 -- corpus-specific mismatched-bracket fix
    (r"cons\(\)trucción\)", "cons(trucción)"),  # :86 -- ditto
    (r"\x01\x53", ""),  # :89 -- illegal control-char sequence
    ("ĺ", "l"),  # :93 -- literal 'ĺ'
    (r"ĺ", "l"),  # :94 -- \x{013A}, same codepoint as :93 (redundant, harmless)
    ("ŕ", "r"),  # :96 -- literal 'ŕ'
    (r"ŕ", "r"),  # :97 -- \x{0155}, redundant with :96
    ("ń", "n"),  # :99 -- literal 'ń'
    (r"ń", "n"),  # :100 -- \x{0144}, redundant with :99
    ("[\";!?.']", " "),  # :105 -- apostrophe IS stripped from full's text (unlike light)
]
_FULL_PREFIX_SUBS_COMPILED = [(re.compile(p), r) for p, r in _FULL_PREFIX_SUBS]


def _full_prefix(text: str) -> str:
    """The pure-regex portion of 03a's per-line body (:31-105), applied to the already
    head/text-split, `" "`-padded `text`. See the module docstring: unreachable output,
    no live-oracle backing."""
    for pattern, repl in _FULL_PREFIX_SUBS_COMPILED:
        text = pattern.sub(repl, text)
    return text


def normalize_stm_full(lines: list[str]) -> list[str]:
    """Port of `norm_stm_pk_cts_all_trans_03a.pl`'s comment/blank-line prefix
    (:10-16), byte-oracle-verified (`tests/reference_data/phase4d/stm/full_*.{in,out}`).
    Any content line reaches the unconditional `norm-tagger`/`norm-parser` shell-out
    (:109) and raises [`NormalizerUnavailable`][speech.dataprep.stm_normalize.NormalizerUnavailable]
    instead of producing output -- see the module docstring for why this is a typed bail
    rather than a reproduction of the shell-out's (environment-dependent, non-legacy)
    failure behavior.

    Same `lines` newline convention as [`normalize_stm_light`]
    [speech.dataprep.stm_normalize.normalize_stm_light].
    """
    out: list[str] = []
    for raw in lines:
        if _COMMENT_RE.match(raw):  # :12
            out.append(raw)
            continue
        if _BLANK_RE.match(raw) or _BLANK_RE2.match(raw):  # :13-14
            continue

        line = raw[:-1] if raw.endswith("\n") else raw  # :15 chomp
        tokens = line.split()  # :17
        head = " ".join(tokens[0:6])  # :19
        text = "  ".join(tokens[6:])  # :20 -- DOUBLE-SPACE join, unlike light's single

        text = " " + text + " "  # :22
        head = _APOSTROPHE_RE.sub("_", head)  # :23
        text = _full_prefix(text)  # :31-105

        raise NormalizerUnavailable(head, text)  # :109 equivalent
    return out
