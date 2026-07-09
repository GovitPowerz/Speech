"""NIST OpenSAD 2015 corpus converter. Ported from legacy `ProcessOpenSAD15Corpus.py`
(Python 2, ~145 LOC, vendored read-only at `legacy/.../ProcessOpenSAD15Corpus.py`).

VALIDATION TIER (read this before trusting the goldens): this is the phase's weakest
tier -- there is NO live oracle. The legacy script is Python 2 and no Python 2
interpreter exists in this environment (or reasonably obtainable one), so nothing here
can be cross-run against the original. Every fixture under
`tests/reference_data/phase4d/opensad15/` is HAND-COMPUTED from a line-by-line reading
of the legacy source (not dumped from a live run), then cross-checked by a second,
independent-of-this-module arithmetic pass (recorded in the Task 5 report). The emitted
XML is additionally cross-parsed by the Rust engine's own `load_vrcts`
(`src/rust/tests/phase4d_opensad15_vrcts.rs`) to prove it is actually consumable, which
is real (if narrow) validation beyond pure hand-transcription.

`treat_file(line)` (`:23-108`) is split into a PURE core, `convert_tab_file`, and the
file-I/O driver, `process_opensad15`. Quirks reproduced verbatim (see IMPROVEMENTS.md
for the `[4d]` entries):

  * `duration`/`lang` are read from `tmp[3]`/`tmp[8]` on EVERY tab row, unconditional on
    the S/RI filter (`:37-38` run before the `:39` filter `if`) -- both end up holding
    whatever the LAST physical row in the file had, whether or not that row passed the
    filter. Two independent instances of the same "last-row-wins" bug, not one.
  * `audiofile[:-5]` (`:48`, `:67`, `:100`) is a hard-coded 5-character strip, not an
    extension-aware split -- correct only when the audio extension is exactly 5 chars
    incl. the dot (e.g. `.flac`); any other extension length silently corrupts the
    stem. Reproduced as-is (`_path_leaf(audio_path[:-5])`, no extension detection).
  * `str(float)` on segment/collar times and `spdur` (`:53`, `:57`, `:61`, `:100-102`,
    `:106`) uses CPython 2's `str(float)` formatting, NOT Python 3's (which is `repr`,
    shortest-round-trip). Python 2.7's `float.__str__` calls `PyOS_double_to_string(v,
    'g', 12, 0, ...)` -- i.e. C `%.12g` -- while `float.__repr__` (and Python 3's
    `str`==`repr`) uses the shortest round-trippable digit string. The two diverge
    exactly when a float carries sub-ULP noise past 12 significant digits (e.g. a
    segment-duration sum landing on `16.366300000000003`): Python 2 prints `16.3663`,
    Python 3's plain `str()` would print `16.366300000000003`. `py2_str_float` below
    reproduces the Python 2 behavior via `f"{x:.12g}"` (equivalent to `'%.12g' % x`) plus
    the `.0`-completion Python 2 always applies to bare integral results.
  * `multiprocessing.Pool(20).imap` (`:131-135`) is replaced by a SEQUENTIAL loop in
    `process_opensad15` -- a documented determinism deviation (Pool result ORDER is not
    guaranteed to match input order under `imap` in the presence of uneven per-file
    work, whereas the sequential port always preserves listing order).

`convert_tab_file`'s signature adds a `tab_path` parameter beyond `audio_path`/
`tab_lines`: the emitted listing line embeds the STM output filename, which is derived
from the TAB file's path (`tabfile[:-4]+".stm"`, `:96`), not the audio file's -- there is
no way to assemble the full listing line without it.
"""

from __future__ import annotations

import ntpath
from dataclasses import dataclass
from pathlib import Path


def py2_str_float(x: float) -> str:
    """Emulate CPython 2.7's `str(float)`: `%.12g` formatting (12 significant digits,
    NOT the shortest round-trip repr Python 3's `str`/`repr` use), with Python's
    "always show a decimal point or exponent" completion for bare-integral/nan/inf
    results (`'%.12g' % 1.0` is the C-style `'1'`; Python 2's `str(1.0)` is `'1.0'`).

    No live Python 2 interpreter exists to validate this against; derived from the
    documented CPython 2.7 `float_str`/`PyOS_double_to_string` behavior (precision-12
    'g' format) and pinned by worked cases in `tests/test_phase4d_opensad15.py`,
    including the `143.76000000000002` -> `'143.76'` example from the task brief.
    """
    s = f"{x:.12g}"
    if "." not in s and "e" not in s and "n" not in s:  # bare int, and not nan/inf
        s += ".0"
    return s


def _path_leaf(path: str) -> str:
    """Port of the legacy `path_leaf` (`:18-20`): `ntpath.split` (Windows-style, but
    parses forward slashes too) with a `ntpath.basename(head)` fallback for a
    trailing-separator path whose `tail` is empty."""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


@dataclass
class _ExclSeg:
    """One `SegsExcl` row (`:69-93`): a `[begin, end)` span plus its `kind` -- `1` for
    a real speech segment, `0` for the surrounding collar/excluded region."""

    beg: float
    end: float
    kind: int


def convert_tab_file(audio_path: str, tab_path: str, tab_lines: list[str]) -> tuple[bytes, list[str], str]:
    """Port of `treat_file`'s pure core (`ProcessOpenSAD15Corpus.py:23-108`), split from
    its file I/O. `tab_lines` are the already-`splitlines()`-ed contents of the `.tab`
    file named by `tab_path` (read by the caller); `audio_path` is the listing line's
    first field.

    Returns `(xml_bytes, stm_lines, listing_line)`:
      * `xml_bytes`: the full VRCTS-style XML document, ASCII-encoded, trailing newline.
      * `stm_lines`: one STM reference row per `SegsExcl` entry, NOT newline-terminated
        (the caller joins them) -- a port-only return-shape choice, not a legacy quirk.
      * `listing_line`: the complete `audiofile;stmfile;lang;non;1.0;duration;spdur;`
        row (`:106`), also NOT newline-terminated.
    """
    segs: list[tuple[str, str]] = []
    spdur = 0.0
    duration = ""
    lang = ""
    for line in tab_lines:
        fields = line.split("\t")
        beg, end_s, label, lang = fields[2], fields[3], fields[4], fields[8]
        # Legacy quirk: duration/lang are overwritten on EVERY row, unconditional on
        # the S/RI filter below -- both end up as the LAST physical row's values.
        duration = end_s
        if (len(label) == 1 and label.startswith("S")) or (len(label) == 2 and label.startswith("RI")):
            segs.append((beg, end_s))
            spdur += float(end_s) - float(beg)

    stem = _path_leaf(audio_path[:-5])  # [:-5] quirk: correct only for a 5-char ext.

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<AudioDoc name="{stem}" path="{audio_path}">',
        "<ProcList>",
        '<Proc name="vrcts_part" version="1.3"/>',
        "</ProcList>",
        "<ChannelList>",
        f'<Channel num="1" sigdur="{duration}" spdur="{py2_str_float(spdur)}"/>',
        "</ChannelList>",
        "<SpeakerList>",
        f'<Speaker ch="1" dur="{py2_str_float(spdur)}" gender="1" spkid="1"/>',
        "</SpeakerList>",
        "<SegmentList>",
    ]
    for beg, end_s in segs:
        xml_lines.append(f'<SpeechSegment ch="1" sconf="1.00" stime="{beg}" etime="{end_s}" spkid="1"/>')
    xml_lines.append("</SegmentList>")
    xml_lines.append("</AudioDoc>")
    xml_bytes = ("\n".join(xml_lines) + "\n").encode("ascii")

    if len(lang) == 0:
        lang = _path_leaf(audio_path[:-5]).split("_")[-2]

    duration_f = float(duration)
    segs_excl: list[_ExclSeg] = []
    for beg, end_s in segs:
        beg_f = float(beg)
        end_f = float(end_s)
        if len(segs_excl) == 0:
            min_val = 0.0
            segs_excl.append(_ExclSeg(max(min_val, beg_f - 2.0), beg_f, 0))
            segs_excl.append(_ExclSeg(beg_f, end_f, 1))
            segs_excl.append(_ExclSeg(end_f, min(end_f + 2.0, duration_f), 0))
        else:
            min_val = segs_excl[-2].end
            if segs_excl[-1].end > beg_f or segs_excl[-1].end > max(min_val, beg_f - 2.0 - 0.1):
                segs_excl[-1].end = beg_f
                segs_excl.append(_ExclSeg(beg_f, end_f, 1))
                segs_excl.append(_ExclSeg(end_f, min(end_f + 2.0, duration_f), 0))
            else:
                segs_excl.append(_ExclSeg(max(min_val, beg_f - 2.0), beg_f, 0))
                segs_excl.append(_ExclSeg(beg_f, end_f, 1))
                segs_excl.append(_ExclSeg(end_f, min(end_f + 2.0, duration_f), 0))
    if len(segs_excl) > 0:
        if segs_excl[0].beg < 0.1:
            segs_excl[0].beg = 0.0
        if segs_excl[-1].end > duration_f - 0.1:
            segs_excl[-1].end = duration_f

    stm_lines: list[str] = []
    for seg in segs_excl:
        beg_s = py2_str_float(seg.beg)
        end_s2 = py2_str_float(seg.end)
        if seg.kind == 1:
            stm_lines.append(f"{stem} 1 {stem} {beg_s} {end_s2} toto toto")
        else:
            stm_lines.append(f"{stem} 1 excluded_region {beg_s} {end_s2} toto toto")

    stm_filename = tab_path[:-4] + ".stm"
    listing_line = f"{audio_path};{stm_filename};{lang};non;1.0;{duration};{py2_str_float(spdur)};"

    return xml_bytes, stm_lines, listing_line


def process_opensad15(listing: Path, out_listing: Path) -> None:
    """Port of `main()`'s per-listing loop (`:121-141`): read `listing` (one
    `audiofile;tabfile;...` row per line), convert each referenced `.tab` file via
    [`convert_tab_file`][speech.dataprep.opensad15.convert_tab_file], write the `.xml`/
    `.stm` next to the `.tab` (`tabfile[:-4]+".xml"`/`".stm"`), and append the resulting
    listing line to `out_listing` (truncated at the start, matching the legacy's
    `open(outputListing,'wb+')`).

    SEQUENTIAL, not `multiprocessing.Pool(20).imap` (`:131`) -- a documented determinism
    deviation (see the module docstring): output order always matches listing order.
    The legacy's two bare `print` debug lines (`:127-129`) are dropped (dead output, no
    file-visible effect).
    """
    lines = listing.read_text(encoding="ascii").splitlines()
    with out_listing.open("wb") as out_f:
        for line in lines:
            if not line:
                continue
            fields = line.split(";")
            audio_path, tab_path = fields[0], fields[1]
            tab_lines = Path(tab_path).read_text(encoding="ascii").splitlines()
            xml_bytes, stm_lines, listing_line = convert_tab_file(audio_path, tab_path, tab_lines)

            Path(tab_path[:-4] + ".xml").write_bytes(xml_bytes)
            stm_text = "\n".join(stm_lines) + ("\n" if stm_lines else "")
            Path(tab_path[:-4] + ".stm").write_bytes(stm_text.encode("ascii"))

            out_f.write((listing_line + "\n").encode("ascii"))
