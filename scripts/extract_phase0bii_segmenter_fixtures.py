"""Emit deterministic cross-language fixtures for the segmenter decision oracle.

Generates DETERMINISTIC pseudo-random results rows from a fixed closed-form
formula (`((i*7 + j*13) % 100) / 100.0`) - NOT `random`/`np.random` - so the
cases are reproducible bit-for-bit. Runs each row through `update_segmentation_oracle`
(the numpy port of spec 3.2, decision only, no smoothing) under a few
(rising, area_rising, falling, area_falling, dt, off) parameter settings, and
writes `tests/reference_data/phase0bii/update_seg_cases.json`:

    [{"row": [...], "params": {...}, "expected": [[begin, end, code], ...]}, ...]

The Rust cross-check (`src/rust/tests/phase0bii_segmenter.rs`) loads this and
asserts `update_segmentation_raw` produces the same segments bit-for-bit.

Usage: uv run python scripts/extract_phase0bii_segmenter_fixtures.py
"""

import json
from pathlib import Path

from speech.scoring import update_segmentation_oracle

OUT = Path("tests/reference_data/phase0bii/update_seg_cases.json")

N_ROWS = 20
ROW_LEN = 50

# Parameter settings chosen to exercise: a clean single crossing, area gating
# (non-trigger + accumulation), the falling reset/re-run path, a tail-open
# segment, and a nonzero offset. The values span the deterministic row range
# ([0.00, 0.99]) so crossings actually happen.
PARAM_SETS = [
    {"rising": 0.6, "area_rising": 0.05, "falling": 0.3, "area_falling": 0.03, "dt": 0.02, "off": 0.0},
    {"rising": 0.5, "area_rising": 0.0, "falling": 0.5, "area_falling": 0.0, "dt": 1.0, "off": 0.0},
    {"rising": 0.8, "area_rising": 0.2, "falling": 0.2, "area_falling": 0.1, "dt": 0.1, "off": 5.0},
    {"rising": 0.4, "area_rising": 0.5, "falling": 0.4, "area_falling": 0.4, "dt": 0.5, "off": -1.0},
    # falling > rising with a positive falling area: the ONLY regime that reaches
    # the re-run-rising-after-label branch. A segment can then end via falling
    # ACCUMULATION at an index where r is simultaneously a fresh rising up-cross
    # (r in [rising, falling], r_prev < rising). This param set is intentionally
    # NOT config-clamped (buildFromConf forces falling <= rising); it feeds the
    # raw decision directly, exactly as update_segmentation_raw does, so it pins
    # Rust == Python on that branch too. With falling <= rising the re-run branch
    # is unreachable; see the EXTRA_ROWS comment.
    {"rising": 0.3, "area_rising": 0.0, "falling": 0.7, "area_falling": 0.5, "dt": 0.1, "off": 2.0},
    {"rising": 0.3, "area_rising": 0.0, "falling": 0.7, "area_falling": 1.5, "dt": 1.0, "off": 0.0},
]

# Hand-crafted deterministic rows that (paired with the falling>rising param
# sets above) exercise the re-run-rising-after-label branch: a segment ends via
# falling accumulation at index ii where r in [rising, falling] and r_prev <
# rising, so the same ii is also a fresh rising up-cross and the re-run seeds a
# new begin. The two reset_else branches (`begin>=0 && !hasBegun && r<tR &&
# r_prev<tR`, and its falling twin) are provably unreachable: the accum/init
# invariants force r_prev onto the crossed side, so a non-crossing drop while
# begin>=0 cannot occur. They are left uncovered by design (dead legacy code).
EXTRA_ROWS = [
    [0.0, 1.0, 0.6, 0.1, 0.5, 0.1, 0.6, 0.5, 0.1, 1.0, 0.0],
    [0.0, 1.0, 0.5, 0.1, 0.4, 0.1, 0.5, 0.4, 0.1, 0.9, 0.0],
    [0.2, 0.9, 0.6, 0.1, 0.55, 0.1, 0.6, 0.1, 0.65, 0.1, 0.0],
]


def make_row(i: int) -> list[float]:
    return [((i * 7 + j * 13) % 100) / 100.0 for j in range(ROW_LEN)]


def main() -> None:
    cases = []
    rows = [make_row(i) for i in range(N_ROWS)] + EXTRA_ROWS
    for row in rows:
        for params in PARAM_SETS:
            expected = update_segmentation_oracle(
                row,
                params["rising"],
                params["area_rising"],
                params["falling"],
                params["area_falling"],
                params["dt"],
                params["off"],
            )
            cases.append(
                {
                    "row": row,
                    "params": params,
                    "expected": [[b, e, c] for (b, e, c) in expected],
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1) + "\n")
    n_seg = sum(len(c["expected"]) for c in cases)
    print(f"OK: wrote {len(cases)} cases ({n_seg} labeled segments total) to {OUT}")


if __name__ == "__main__":
    main()
