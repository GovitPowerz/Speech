"""Phase 6 Task 4: the DCF value-oracle fixtures, from the REAL vendored NIST scorer.

Drives `scripts/`-crafted ref/hyp SAD interval pairs through the read-only legacy
`Optimizer_V6.2.2/scoreFile_SAD.pl` (v2.1, Greg Sanders, public domain) via
`tools/perl_oracle/run_sad_scorer.sh`, captures the scorer's per-collar Prob_Miss /
Prob_FalseAlarm / DCF table (five collar sizes: no-collar, 0.25s, 0.5s, 1.0s, 2.0s), and
commits the parsed values as per-case JSON goldens under
`tests/reference_data/phase6/dcf/`. `src/python/speech/evaluate.py`'s `dcf()` is pinned
value-for-value against these (to the scorer's printed 5-decimal precision).

Each committed golden is self-contained: the crafted ref intervals (raw NIST kinds
S/NS/NT/RI/RS/RX), the crafted hyp intervals (speech/non-speech), and the five per-collar
{pmiss, pfa, dcf} triples the perl printed. The crafted `.tab` byte pairs are ephemeral
(a tempdir) -- they are exactly reconstructible from the committed intervals via the same
`_ref_tab`/`_hyp_tab` writers (the ComputeDCF.py:80/83 column layout), so committing them
would only invite drift.

Each case is scored TWICE (same files, same paths) and the parsed tables must be
identical before anything is committed (the perl is deterministic; this guards against a
runner/parse regression).

LOCAL-ONLY, like the other extractors: needs /usr/bin/perl and the legacy tree checked
out at LEGACY_TREE. SKIPS gracefully (prints and returns) when either is absent -- CI
never runs this stage, it only consumes the committed goldens (tests/test_phase6_evaluate.py
guards them WITHOUT any oracle).

Usage: uv run python scripts/extract_phase6_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_TREE = Path("/Users/govit/Git/Govit/FastSpeechProcessing-legacy")
PERL_BIN = Path("/usr/bin/perl")
SCORER = LEGACY_TREE / "Optimizer_V6.2.2" / "scoreFile_SAD.pl"
RUN_SCORER = REPO_ROOT / "tools" / "perl_oracle" / "run_sad_scorer.sh"
DCF_DIR = REPO_ROOT / "tests" / "reference_data" / "phase6" / "dcf"
MANIFEST_PATH = DCF_DIR / "dcf_manifest.json"

# The five collar sizes scoreFile_SAD.pl reports, in stdout order, and the exact block
# labels it prints (computeAndPrintScores' $collarSize arg, :413-417).
COLLAR_LABELS = ["No Collar", "QuarterSecond Collar", "HalfSecond Collar", "OneSecond Collar", "TwoSecond Collar"]
COLLAR_SIZES = [0.0, 0.25, 0.5, 1.0, 2.0]

LABEL_RE = re.compile(r"Scores with (.+)")
PMISS_RE = re.compile(r"Prob_Miss == (\d+\.\d+)")
PFA_RE = re.compile(r"Prob_FalseAlarm == (\d+\.\d+)")
DCF_RE = re.compile(r"DCF == (\d+\.\d+)")

Interval = tuple[float, float, str]

# --- Crafted cases: (description, ref intervals [S/NS/NT/RI/RS/RX], hyp intervals
# [speech/non-speech]). Every ref is contiguous; a ref whose first seg starts > 0 is the
# exnihilate case (the scorer fabricates a leading NonSpeech, :89-95). Every hyp starts at
# 0 (the scorer dies otherwise, :324). Boundaries are chosen so speech/nonspeech time sums
# are exact in double precision, and the collar-arithmetic thresholds (leading c+0.1,
# interior 2c+0.1) land on both sides of segment durations to exercise carve-vs-merge.
CASES: dict[str, tuple[str, list[Interval], list[Interval]]] = {
    "01_plain_overlap": (
        "Baseline TP/FN/FP/TN over a speech/nonspeech/speech ref with a hyp that "
        "misaligns both boundaries -- exercises the multi-segment scoring sweep.",
        [(0.0, 2.0, "S"), (2.0, 4.0, "NS"), (4.0, 6.0, "S")],
        [(0.0, 1.5, "speech"), (1.5, 4.5, "non-speech"), (4.5, 6.0, "speech")],
    ),
    "02_ri_counts_as_speech": (
        "RI merges into the speech region (S then RI -> speech [0,2]); a partial-hit hyp "
        "makes pmiss/pfa sensitive to the RI classification (RI-as-nonspeech would flip "
        "the 0.5s of FN into 0.5s of FP with a different denominator).",
        [(0.0, 1.0, "S"), (1.0, 2.0, "RI"), (2.0, 4.0, "NS")],
        [(0.0, 1.5, "speech"), (1.5, 4.0, "non-speech")],
    ),
    "03_nonspeech_classes": (
        "NS/NT/RS/RX all merge into one nonspeech region [1,5]; a hyp that calls the "
        "whole middle speech makes pfa (and its denominator) sensitive to every one of "
        "those four kinds being treated as non-speech.",
        [(0.0, 1.0, "S"), (1.0, 2.0, "NS"), (2.0, 3.0, "NT"), (3.0, 4.0, "RS"), (4.0, 5.0, "RX"), (5.0, 6.0, "S")],
        [(0.0, 0.5, "non-speech"), (0.5, 5.5, "speech"), (5.5, 6.0, "non-speech")],
    ),
    "04_collar_boundary_fp": (
        "Interior nonspeech of duration 2.1s (== the one-second interior threshold "
        "2*1.0+0.1) with an FP straddling both collar edges: the FP falls inside the "
        "collar (excluded) once the collar grows past 0.25s, so pfa/dcf step across "
        "collar sizes; the 2.0s collar swallows the whole segment (nonspeech sum -> 0, "
        "the pfa zero-guard + the 'nonspeech==0' DCF branch).",
        [(0.0, 2.0, "S"), (2.0, 4.1, "NS"), (4.1, 6.0, "S")],
        [(0.0, 2.25, "speech"), (2.25, 3.85, "non-speech"), (3.85, 6.0, "speech")],
    ),
    "05_all_speech_perfect": (
        "Degenerate all-speech ref, perfectly hypothesized: nonspeech sum == 0 (pfa "
        "zero-guard) and FN == 0, so every collar is 0/0/0 -- the v2.1 zero-not-NaN "
        "semantics on the speech-only side.",
        [(0.0, 6.0, "S")],
        [(0.0, 6.0, "speech")],
    ),
    "06_all_speech_miss": (
        "All-speech ref with a trailing miss: pmiss > 0, pfa == 0 (no nonspeech), and DCF "
        "takes the 'nonspeech sum == 0' branch (0.75*pmiss only).",
        [(0.0, 6.0, "S")],
        [(0.0, 4.0, "speech"), (4.0, 6.0, "non-speech")],
    ),
    "07_all_nonspeech_perfect": (
        "Degenerate all-nonspeech ref, perfectly hypothesized: speech sum == 0 (pmiss "
        "zero-guard), FP == 0, every collar 0/0/0. Also exercises the LEADING-nonspeech "
        "collar carve (one trailing collar of width c off the single segment).",
        [(0.0, 6.0, "NS")],
        [(0.0, 6.0, "non-speech")],
    ),
    "08_all_nonspeech_fp": (
        "All-nonspeech ref with a leading FP far from the trailing collar: pfa > 0 and DCF "
        "takes the 'speech sum == 0' branch (0.25*pfa only); the pfa denominator shrinks "
        "as the trailing collar grows, stepping pfa across collar sizes.",
        [(0.0, 6.0, "NS")],
        [(0.0, 2.0, "speech"), (2.0, 6.0, "non-speech")],
    ),
    "09_sub_collar_micro": (
        "A 0.3s interior nonspeech island -- below every interior collar threshold "
        "(smallest is 2*0.25+0.1=0.6) -- so it is wholly a collar (excluded) for all four "
        "nonzero collars; only the no-collar pass scores it (pfa 1.0 there). Sub-collar "
        "micro-segment coverage.",
        [(0.0, 2.0, "S"), (2.0, 2.3, "NS"), (2.3, 5.0, "S")],
        [(0.0, 5.0, "speech")],
    ),
    "10_hyp_shorter_pad": (
        "Hyp ends at 3.0 but the ref runs to 4.0: the scorer pads the trailing Speech hyp "
        "with a fabricated NonSpeech [3,4] (:390-393), turning the uncovered tail into FN.",
        [(0.0, 4.0, "S")],
        [(0.0, 3.0, "speech")],
    ),
    "11_hyp_longer_truncate": (
        "Hyp runs to 6.0 but the ref ends at 4.0: the scorer pops the fully-past-refEnd "
        "trailing Speech seg (:381-383) and the scoring sweep bounds the rest at refEnd, "
        "so the [4,6] hyp tail never scores.",
        [(0.0, 2.0, "S"), (2.0, 4.0, "NS")],
        [(0.0, 3.0, "speech"), (3.0, 4.5, "non-speech"), (4.5, 6.0, "speech")],
    ),
    "12_ref_starts_late_exnihilate": (
        "Ref first seg starts at 1.0 (> 0): the scorer exnihilates a leading NonSpeech "
        "[0,1] (:89-95). That fabricated seg then drives the leading collar carve, and the "
        "trailing NonSpeech [3,5] drives an interior carve -- a hyp of all-speech makes "
        "both nonspeech regions FP so the collar denominators are visible.",
        [(1.0, 3.0, "S"), (3.0, 5.0, "NS")],
        [(0.0, 5.0, "speech")],
    ),
    "13_exact_threshold_boundary": (
        "Interior nonspeech of duration exactly 0.6s == the quarter-second interior "
        "threshold (2*0.25+0.1): the >= boundary is inclusive so the 0.25s collar CARVES "
        "(0.1s scored middle) while the 0.5s collar (threshold 1.1) MERGES it to a whole "
        "collar -- pins the exact float comparison at the carve/merge boundary.",
        [(0.0, 1.0, "S"), (1.0, 1.6, "NS"), (1.6, 3.0, "S")],
        [(0.0, 3.0, "speech")],
    ),
    "14_all_collar_dcf_zero": (
        "A 0.3s all-nonspeech ref: below the leading threshold 0.25+0.1=0.35 for every "
        "nonzero collar, so the whole ref becomes a single collar -> both speech and "
        "nonspeech sums are 0 -> the hardcoded 'DCF == 0.00000' both-zero branch (:575-576).",
        [(0.0, 0.3, "NS")],
        [(0.0, 0.3, "non-speech")],
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fmt(x: float) -> str:
    """Shortest round-trip decimal so the perl's strtod recovers the identical double."""
    return repr(float(x))


def _ref_tab(intervals: list[Interval]) -> str:
    """Ref .tab: cols 0=file 1=chan [2]=start [3]=end [4]=type (the -s2 -e3 -g4 layout)."""
    return "".join(f"crafted\t1\t{_fmt(s)}\t{_fmt(e)}\t{g}\n" for s, e, g in intervals)


def _hyp_tab(intervals: list[Interval]) -> str:
    """Hyp .tab: cols 0-4 dummy, [5]=start [6]=end [7]=type (ComputeDCF.py:80 / -t5 -f6 -u7)."""
    return "".join(f"testDef\tex\tds\tSAD\tcrafted\t{_fmt(s)}\t{_fmt(e)}\t{u}\n" for s, e, u in intervals)


def _run_scorer(ref_path: Path, hyp_path: Path) -> str:
    result = subprocess.run(
        ["bash", str(RUN_SCORER), str(ref_path), str(hyp_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"run_sad_scorer.sh failed (exit {result.returncode}): {result.stderr}")
    return result.stdout


def _parse_table(stdout: str) -> list[dict[str, float]]:
    """Parse the five per-collar blocks from scoreFile_SAD.pl's STDOUT into
    [{collar, pmiss, pfa, dcf}, ...] in the fixed COLLAR_SIZES order."""
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in stdout.splitlines():
        label = LABEL_RE.search(line)
        if label:
            if current is not None:
                blocks.append(current)
            current = {"label": label.group(1).strip()}
            continue
        if current is None:
            continue
        for key, pat in (("pmiss", PMISS_RE), ("pfa", PFA_RE), ("dcf", DCF_RE)):
            m = pat.search(line)
            if m:
                current[key] = float(m.group(1))
                break
    if current is not None:
        blocks.append(current)

    labels = [b["label"] for b in blocks]
    if labels != COLLAR_LABELS:
        raise SystemExit(f"unexpected collar block labels: {labels}")
    out: list[dict[str, float]] = []
    for size, b in zip(COLLAR_SIZES, blocks, strict=True):
        missing = {"pmiss", "pfa", "dcf"} - b.keys()
        if missing:
            raise SystemExit(f"collar {size}: missing {missing} in scorer output")
        out.append({"collar": size, "pmiss": b["pmiss"], "pfa": b["pfa"], "dcf": b["dcf"]})
    return out


def main() -> None:
    if not PERL_BIN.is_file():
        print(f"SKIP: {PERL_BIN} not found -- dcf fixtures not (re)generated")
        return
    if not SCORER.is_file():
        print(f"SKIP: NIST scorer not found at {SCORER} -- dcf fixtures not (re)generated")
        return

    DCF_DIR.mkdir(parents=True, exist_ok=True)
    cases_manifest: dict[str, dict[str, object]] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name, (description, ref, hyp) in CASES.items():
            ref_path = tmp_dir / f"{name}.ref.tab"
            hyp_path = tmp_dir / f"{name}.hyp.tab"
            ref_path.write_text(_ref_tab(ref))
            hyp_path.write_text(_hyp_tab(hyp))

            table1 = _parse_table(_run_scorer(ref_path, hyp_path))
            table2 = _parse_table(_run_scorer(ref_path, hyp_path))
            if table1 != table2:
                raise SystemExit(f"{name}: scorer not deterministic across two runs")

            golden = {
                "name": name,
                "description": description,
                "ref": [[s, e, g] for s, e, g in ref],
                "hyp": [[s, e, u] for s, e, u in hyp],
                "collars": table1,
            }
            golden_bytes = (json.dumps(golden, indent=2) + "\n").encode()
            (DCF_DIR / f"{name}.json").write_bytes(golden_bytes)
            cases_manifest[name] = {"sha256": _sha256(golden_bytes), "bytes": len(golden_bytes)}

    perl_version = subprocess.run([str(PERL_BIN), "-e", "print $^V"], capture_output=True, text=True).stdout.strip()

    manifest = {
        "text": (
            "Phase 6 Task 4: live value-oracle for the DCF SAD scorer. Each <case>.json is "
            "the REAL vendored NIST scoreFile_SAD.pl (v2.1) run over the crafted ref/hyp "
            "intervals via /usr/bin/perl through tools/perl_oracle/run_sad_scorer.sh (never "
            "modifies the read-only legacy tree), with the five per-collar Prob_Miss / "
            "Prob_FalseAlarm / DCF values parsed from its STDOUT at the scorer's printed "
            "5-decimal precision. src/python/speech/evaluate.py::dcf is pinned value-for-value "
            "against these; each case is scored twice per extraction to confirm determinism. "
            "Column convention is ComputeDCF.py's (-s 2 -e 3 -g 4 -t 5 -f 6 -u 7)."
        ),
        "perl_bin": str(PERL_BIN),
        "perl_version": perl_version,
        "scorer": str(SCORER.relative_to(LEGACY_TREE.parent)),
        "runner": str(RUN_SCORER.relative_to(REPO_ROOT)),
        "collar_sizes": COLLAR_SIZES,
        "cases": cases_manifest,
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    MANIFEST_PATH.write_bytes(manifest_bytes)
    print(f"OK: dcf fixtures ({len(cases_manifest)} cases, perl {perl_version}) -> {DCF_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
