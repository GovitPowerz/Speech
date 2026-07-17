"""SAD detection-cost (DCF) scoring -- a faithful port of the NIST OpenSAD scorer.

`dcf()` reproduces `Optimizer_V6.2.2/scoreFile_SAD.pl` (v2.1, Greg Sanders, public
domain) segment-for-segment: the reference ingestion (splicing consecutive same-class
segments, exnihilating a leading NonSpeech when the ref starts after 0), the five collar
constructions (no-collar + 0.25/0.5/1.0/2.0s), the hypothesis trim/pad to the reference
span, and the interval sweep that accumulates speech/nonspeech time into
miss/false-alarm sums. Per collar it reports Prob_Miss, Prob_FalseAlarm, and
DCF = 0.75*Pmiss + 0.25*Pfa, including v2.1's zero-not-NaN semantics on degenerate
(all-speech or all-nonspeech) files.

The port accumulates the same segment durations in the same sweep order as the perl, so
the sums are bit-identical doubles; `tests/test_phase6_evaluate.py` pins the reported
values against the real perl scorer (via `scripts/extract_phase6_fixtures.py`) to the
scorer's printed 5-decimal precision.

Reference kinds are the raw NIST tab kinds (S / RI count as speech; NS / NT / RS / RX as
non-speech); hypothesis kinds are `speech` / `non-speech` (`nonspeech` also accepted, as
the perl does). `Interval` is `(start, end, kind)`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# (start, end, kind). Ref kinds: S/NS/NT/RI/RS/RX. Hyp kinds: speech/non-speech.
Interval = tuple[float, float, str]

# scoreFile_SAD.pl's "< 0.00001 means actually 0.0" guard (:551, :561, :575).
_EPS = 0.00001

# Ref-side kind partitions (:99 non-speech; :107 speech, RI folded into S per :108).
_REF_NONSPEECH = frozenset({"NS", "NT", "RX", "RS"})
_REF_SPEECH = frozenset({"S", "RI"})

# Hyp-side kind spellings (:326 accepts both non-speech spellings; :334 speech).
_HYP_NONSPEECH = frozenset({"non-speech", "nonspeech"})
_HYP_SPEECH = frozenset({"speech"})

_SIGDUR_RE = re.compile(r'<Channel\b[^>]*\bsigdur="([^"]*)"')
_SEG_RE = re.compile(r'<SpeechSegment\b[^>]*\bstime="([^"]*)"[^>]*\betime="([^"]*)"')


@dataclass(frozen=True)
class CollarScore:
    """Per-collar scores. `dcf` == 0.75*`pmiss` + 0.25*`pfa` (with the degenerate branches)."""

    collar: float
    pmiss: float
    pfa: float
    dcf: float


@dataclass(frozen=True)
class DcfReport:
    """The five per-collar scores, in the order the `collars` argument was given."""

    scores: list[CollarScore]

    def by_collar(self, collar: float) -> CollarScore:
        for s in self.scores:
            if s.collar == collar:
                return s
        raise KeyError(f"no score for collar {collar}")


def dcf(
    ref: list[Interval],
    hyp: list[Interval],
    collars: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> DcfReport:
    """Score a hypothesis against a reference, per collar. `ref` carries raw NIST kinds
    (S/NS/NT/RI/RS/RX), `hyp` carries speech/non-speech; both must be contiguous (a gap in
    the ref past time 0 is an error, as in the perl). A collar of 0.0 is the no-collar
    pass."""
    no_collar = _build_ref_no_collar(ref)
    hyp_segs = _fit_hyp(_build_hyp_segs(hyp), no_collar[0][0], no_collar[-1][1])
    scores = []
    for c in collars:
        ref_segs = list(no_collar) if c == 0.0 else _build_collar_segs(no_collar, c)
        scores.append(_score_collar(ref_segs, hyp_segs, c))
    return DcfReport(scores)


def load_tab_ref(path: Path, start_col: int = 2, end_col: int = 3, type_col: int = 4) -> list[Interval]:
    """Read a TAB-delimited NIST ref file into raw intervals. Column defaults are the
    ComputeDCF.py convention (-s 2 -e 3 -g 4); kinds are returned verbatim (S/NS/NT/RI/RS/RX)
    for `dcf()` to interpret."""
    out: list[Interval] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        out.append((float(fields[start_col]), float(fields[end_col]), fields[type_col]))
    return out


def load_vrcts_hyp(path: Path) -> list[Interval]:
    """Read the engine's VRCTS xml (the AudioDoc/Channel/SpeechSegment family
    `tasks/segmentation_io.rs` writes) into contiguous speech/non-speech intervals: the
    `<SpeechSegment>` spans are speech, the gaps between them (and the leading gap from 0,
    the trailing gap up to the channel `sigdur`) are non-speech. Mirrors ComputeDCF.py's
    gap-filling, with the channel `sigdur` playing the role of its ref-tab `duration`."""
    text = path.read_text()
    speech = sorted((float(a), float(b)) for a, b in _SEG_RE.findall(text))

    sig = _SIGDUR_RE.search(text)
    sigdur = float(sig.group(1)) if sig else (speech[-1][1] if speech else 0.0)

    out: list[Interval] = []
    cursor = 0.0
    for begin, end in speech:
        if begin > cursor:
            out.append((cursor, begin, "non-speech"))
        out.append((begin, end, "speech"))
        cursor = end
    if sigdur > cursor:
        out.append((cursor, sigdur, "non-speech"))
    return out


# --- reference ingestion + collar construction ---------------------------------------


def _build_ref_no_collar(ref: list[Interval]) -> list[Interval]:
    """Port of scoreFile_SAD.pl:85-126 -- fold raw ref rows into a contiguous
    Speech/NonSpeech segment list, splicing consecutive same-class rows and exnihilating a
    leading NonSpeech when the first row starts after 0."""
    segs: list[Interval] = []
    curr = 0.0
    prev = ""
    for s, e, g in ref:
        if s > curr and curr == 0.0:
            segs.append((0.0, s, "NonSpeech"))
            prev = "NonSpeech"
            curr = s
        if s > curr and curr > 0.0:
            raise ValueError(f"un-annotated ref interval from {curr} to {s}")
        elif g in _REF_NONSPEECH:
            if prev == "NonSpeech" and curr > 0.0:
                s = segs.pop()[0]  # splice into the preceding NonSpeech
            segs.append((s, e, "NonSpeech"))
            prev = "NonSpeech"
            curr = e
        elif g in _REF_SPEECH and s == curr:
            if prev == "Speech" and curr > 0.0:
                s = segs.pop()[0]  # splice into the preceding Speech
            segs.append((s, e, "Speech"))
            prev = "Speech"
            curr = e
        elif g not in _REF_SPEECH and g not in _REF_NONSPEECH:
            raise ValueError(f"unexpected ref segment type {g!r}")
        else:
            raise ValueError(f"overlapping ref segs: prev end {curr}, new begin {s}")
    return segs


def _build_collar_segs(no_collar: list[Interval], c: float) -> list[Interval]:
    """Port of scoreFile_SAD.pl:128-268 -- carve collars of width `c` out of the nonspeech
    segments. The leading nonspeech gets one trailing collar (threshold c+0.1); interior
    nonspeech gets a collar on each side (threshold 2c+0.1); if a segment is too short to
    leave >= 0.1s of scored nonspeech it becomes a whole collar. Speech is unchanged."""
    out: list[Interval] = []
    start, end, kind = no_collar[0]
    if kind == "NonSpeech":
        if end - start >= c + 0.1:
            out.append((start, end - c, "NonSpeech"))
            out.append((end - c, end, "Collar"))
        else:
            out.append((start, end, "Collar"))
    else:
        out.append((0.0, end, "Speech"))

    for start, end, kind in no_collar[1:]:
        if kind == "Speech":
            out.append((start, end, "Speech"))
        elif end - start >= 2.0 * c + 0.1:
            out.append((start, start + c, "Collar"))
            out.append((start + c, end - c, "NonSpeech"))
            out.append((end - c, end, "Collar"))
        else:
            out.append((start, end, "Collar"))
    return out


# --- hypothesis ingestion + fit ------------------------------------------------------


def _build_hyp_segs(hyp: list[Interval]) -> list[Interval]:
    """Port of scoreFile_SAD.pl:318-347 -- fold raw hyp rows into a contiguous
    Speech/NonSpeech list, splicing consecutive same-class rows. The first row must start
    at 0 and rows must be contiguous (the perl dies otherwise)."""
    segs: list[Interval] = []
    curr = 0.0
    prev = ""
    for t, f, u in hyp:
        if t != curr:
            raise ValueError(f"non-contiguous hyp at {curr} -> {t}")
        if u in _HYP_NONSPEECH:
            if prev == "NonSpeech" and curr > 0.0:
                t = segs.pop()[0]
            segs.append((t, f, "NonSpeech"))
            prev = "NonSpeech"
            curr = f
        elif u in _HYP_SPEECH:
            if prev == "Speech" and curr > 0.0:
                t = segs.pop()[0]
            segs.append((t, f, "Speech"))
            prev = "Speech"
            curr = f
        else:
            raise ValueError(f"unexpected hyp segment type {u!r}")
    return segs


def _fit_hyp(hyp_segs: list[Interval], ref_start: float, ref_end: float) -> list[Interval]:
    """Port of scoreFile_SAD.pl:350-395 -- trim/pad the hyp so it starts at `ref_start` and
    ends at `ref_end`. A hyp seg that ENCLOSES an endpoint is left long (the scoring sweep
    bounds it); a hyp that falls short is padded with NonSpeech."""
    segs = list(hyp_segs)

    if segs[0][0] < ref_start:
        while segs[0][1] <= ref_start:
            segs.pop(0)
    if segs[0][0] > ref_start:
        if segs[0][2] == "NonSpeech":
            segs[0] = (ref_start, segs[0][1], "NonSpeech")
        else:
            segs.insert(0, (ref_start, segs[0][0], "NonSpeech"))

    if segs[-1][1] > ref_end:
        while segs[-1][0] >= ref_end:
            segs.pop()
    if segs[-1][1] < ref_end:
        if segs[-1][2] == "NonSpeech":
            segs[-1] = (segs[-1][0], ref_end, "NonSpeech")
        else:
            segs.append((segs[-1][1], ref_end, "NonSpeech"))
    return segs


# --- scoring sweep -------------------------------------------------------------------


def _score_collar(ref_segs: list[Interval], hyp_segs: list[Interval], collar: float) -> CollarScore:
    """Port of scoreFile_SAD.pl's computeAndPrintScores (:424-591): sweep the two
    contiguous segment lists, accumulating scored speech/nonspeech time into miss/false-
    alarm sums, then apply the v2.1 zero-guarded rate + DCF formulas."""
    speech_sum = 0.0
    fn = 0.0
    nonspeech_sum = 0.0
    fp = 0.0

    curr_t = 0.0
    ref_state = "undefined"
    hyp_state = "undefined"
    end_scored = ref_segs[-1][1]
    hyp_idx = 0
    max_hyp = len(hyp_segs) - 1
    ref_idx = 0
    max_ref = len(ref_segs) - 1

    while curr_t < end_scored:
        prev_t = curr_t
        hs = hyp_segs[hyp_idx]
        rs = ref_segs[ref_idx]
        # Whichever seg starts strictly earlier keeps its state sticky; the interval [prev,
        # curr] ends at the nearer of the two current seg ends (:476-513). NB the perl's two
        # `$currScoringTime == ...` lines (:478, :491) are `==` not `=` -- inert comparisons,
        # not assignments; curr_t is always (re)set by the end-advance below, so they are
        # correctly omitted here (dropping them changes nothing).
        if hs[0] < rs[0]:
            ref_state = rs[2]
        elif hs[0] > rs[0]:
            hyp_state = hs[2]
        else:
            ref_state = rs[2]
            hyp_state = hs[2]
        if hs[1] <= rs[1]:
            curr_t = hs[1]
            if hyp_idx < max_hyp:
                hyp_idx += 1
        else:
            curr_t = rs[1]
            if ref_idx < max_ref:
                ref_idx += 1

        if ref_state == "Collar":
            continue
        seg_dur = curr_t - prev_t
        if ref_state == "Speech":
            speech_sum += seg_dur
            if hyp_state == "NonSpeech":
                fn += seg_dur
        elif ref_state == "NonSpeech":
            nonspeech_sum += seg_dur
            if hyp_state == "Speech":
                fp += seg_dur

    pmiss = fn / (speech_sum if speech_sum >= _EPS else 1.0)
    pfa = fp / (nonspeech_sum if nonspeech_sum >= _EPS else 1.0)

    if speech_sum < _EPS and nonspeech_sum < _EPS:
        dcf_val = 0.0
    elif speech_sum < _EPS:
        dcf_val = 0.25 * (fp / nonspeech_sum)
    elif nonspeech_sum < _EPS:
        dcf_val = 0.75 * (fn / speech_sum)
    else:
        dcf_val = 0.75 * (fn / speech_sum) + 0.25 * (fp / nonspeech_sum)

    return CollarScore(collar=collar, pmiss=pmiss, pfa=pfa, dcf=dcf_val)
