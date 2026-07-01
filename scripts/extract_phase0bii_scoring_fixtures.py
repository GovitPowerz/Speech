"""Emit deterministic cross-language fixtures for the `compute_errors` Pass 2 oracle.

Generates DETERMINISTIC ref/hyp boundary-list pairs from fixed closed-form spans
(NOT `random`/`np.random`), each with grid-aligned begins/ends so they survive the
1e-4-grid snap bit-for-bit. Runs each pair through `compute_errors_pass2_oracle`
(the numpy port of `Segmentation.cpp:410-476`, Pass 2 only) and writes
`tests/reference_data/phase0bii/compute_errors_cases.json`:

    [{"ref_spans": [[b,e,code],...], "hyp_spans": [[b,e,code],...],
      "audio_duration": dur, "expected": [[pmiss,pfa,error_rate], ...x23]}, ...]

The Rust cross-check (`src/rust/tests/phase0bii_scoring.rs`) loads this, rebuilds
ref/hyp via `Segmentation::new(dur)` + `label_segment` per span, runs
`compute_errors(&mut hyp, Some(&ref), -1)`, and asserts the per-class Pfa/Pmiss/
ErrorRate match the oracle's `expected` bit-for-bit.

The span codes cycle through Speech(1), Substitution(20), Insertion(19),
Excluded(21) so the run exercises: Sub->Speech and Ins->Other folding, Excluded
(exemption + normalization skip), both two-pointer branches (ref-advance /
hyp-advance), and both arms of the min-overlap ternary.

Usage: uv run python scripts/extract_phase0bii_scoring_fixtures.py
"""

import json
from pathlib import Path

from speech.scoring import (
    EXCLUDED_CODE,
    INSERTION_CODE,
    OTHER_CODE,
    SPEECH_CODE,
    SUBSTITUTION_CODE,
    compute_errors_pass2_oracle,
)

OUT = Path("tests/reference_data/phase0bii/compute_errors_cases.json")

Span = tuple[float, float, int]
Case = tuple[list[Span], list[Span], float]

# Deterministic ref/hyp span sets. Every span is [begin, end, code] with begins/
# ends on a 0.1 grid, sorted and non-overlapping within each side, all inside
# [0, audio_duration]. Other is the implicit background. Codes are chosen to cover
# the folding + Excluded + both pointer branches + both min-overlap arms.
CASES: list[Case] = [
    # 1. Task-7 hand case: single Speech ref vs shifted Speech hyp.
    ([(2.0, 5.0, SPEECH_CODE)], [(3.0, 6.0, SPEECH_CODE)], 10.0),
    # 2. Substitution ref folds to Speech; hyp Speech partially overlaps.
    ([(1.0, 4.0, SUBSTITUTION_CODE)], [(2.0, 5.0, SPEECH_CODE)], 8.0),
    # 3. Insertion ref folds to Other (so ref is all-Other after folding); a
    #    Speech hyp inside it is pure false-alarm.
    ([(1.0, 3.0, INSERTION_CODE)], [(1.5, 2.5, SPEECH_CODE)], 6.0),
    # 4. Excluded ref span: exempt from scoring and removed from the
    #    normalization denominators.
    ([(2.0, 4.0, EXCLUDED_CODE), (5.0, 7.0, SPEECH_CODE)], [(3.0, 6.0, SPEECH_CODE)], 10.0),
    # 5. Ref Speech fully contains hyp Speech (hyp shorter) -> ref-advance branch
    #    with the `h.begin <= r_next` min-overlap arm and a trailing miss.
    ([(1.0, 8.0, SPEECH_CODE)], [(3.0, 5.0, SPEECH_CODE)], 10.0),
    # 6. Hyp Speech fully contains ref Speech (hyp longer) -> hyp-advance branch
    #    and the `r.begin > h_next`/`else` arms both fire.
    ([(4.0, 6.0, SPEECH_CODE)], [(2.0, 9.0, SPEECH_CODE)], 12.0),
    # 7. Two disjoint ref Speech spans vs one wide hyp Speech: multiple crossings,
    #    both branches, min-overlap ternary both arms.
    (
        [(1.0, 3.0, SPEECH_CODE), (6.0, 8.0, SPEECH_CODE)],
        [(2.0, 7.0, SPEECH_CODE)],
        10.0,
    ),
    # 8. Mixed codes: Substitution + Insertion + Excluded in ref, two hyp Speech
    #    spans -> folding + Excluded + both branches together.
    (
        [
            (1.0, 3.0, SUBSTITUTION_CODE),
            (4.0, 6.0, INSERTION_CODE),
            (7.0, 9.0, EXCLUDED_CODE),
            (10.0, 12.0, SPEECH_CODE),
        ],
        [(2.0, 5.0, SPEECH_CODE), (8.0, 11.0, SPEECH_CODE)],
        14.0,
    ),
    # 9. Empty hyp (all-Other): every scorable ref span is a pure miss.
    ([(2.0, 5.0, SPEECH_CODE), (6.0, 9.0, SUBSTITUTION_CODE)], [], 10.0),
    # 10. Empty ref (all-Other): every hyp Speech span is a pure false-alarm.
    ([], [(1.0, 3.0, SPEECH_CODE), (5.0, 8.0, SPEECH_CODE)], 10.0),
]


def spans_to_bl(spans: list[Span], dur: float) -> list[tuple[float, int]]:
    """Build a boundary list `[(begin, code), ...]` from sorted non-overlapping
    labeled spans, Other as background, ending with the End sentinel. Rounds each
    begin to the 1e-4 grid and drops any zero-length duplicate boundary."""
    bl: list[tuple[float, int]] = [(0.0, OTHER_CODE)]
    for begin, end, code in spans:
        bl.append((round(begin, 4), code))
        bl.append((round(end, 4), OTHER_CODE))
    bl.append((round(dur, 4), 22))  # End sentinel.
    # Drop zero-length boundaries (a later boundary at the same time wins).
    dedup: list[tuple[float, int]] = []
    for b, c in bl:
        if dedup and dedup[-1][0] == b:
            dedup[-1] = (b, c)
        else:
            dedup.append((b, c))
    return dedup


def main() -> None:
    cases = []
    for ref_spans, hyp_spans, dur in CASES:
        ref_bl = spans_to_bl(ref_spans, dur)
        hyp_bl = spans_to_bl(hyp_spans, dur)
        expected = compute_errors_pass2_oracle(ref_bl, hyp_bl, dur)
        cases.append(
            {
                "ref_spans": [[b, e, c] for (b, e, c) in ref_spans],
                "hyp_spans": [[b, e, c] for (b, e, c) in hyp_spans],
                "audio_duration": dur,
                "expected": expected,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1) + "\n")
    print(f"OK: wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
