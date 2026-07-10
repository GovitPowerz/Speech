"""Phase 4d Task 8: listing writers pinned vs the real vendored WriteListing.m /
WriteWeightedListing.m via Octave.

Goldens (`tests/reference_data/phase4d/listing/`) are the RAW TEXT files Octave itself
wrote by driving the vendored `.m` unchanged (`tools/octave_harness/stage_writelisting.m`,
TIER 1) -- unlike every other Octave-fixture-consuming test in this repo, there is no
`.bin` conversion step here: `WriteListing`/`WriteWeightedListing` already write plain
text, so the goldens ARE that text, byte for byte. The `PLAIN_ITEMS`/`WEIGHTED_ITEMS`/
`WEIGHTED_VALUES` fed to `write_listing`/`write_weighted_listing` below are a literal
transcription of the crafted case in `stage_writelisting.m` (same file names, langs,
dials, weights, durations) -- this is a byte-oracle golden, not a black-box parity
test, so the two sides' inputs must match by construction, the same convention
`stage_batching.m` / `test_phase4c_batching.py` already establish.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from speech.batching import _worker_positions, write_listing, write_weighted_listing

LISTING_DIR = Path(__file__).resolve().parent / "reference_data" / "phase4d" / "listing"

# --- plain: mirrors stage_writelisting.m's 7-file / 3-worker case ---
PLAIN_ITEMS = [
    {
        "filename": f"audio_{i:02d}.wav",
        "refseg": f"ref_{i:02d}.xml",
        "lang": f"lang_{i:02d}",
        "dial": f"dial_{i:02d}",
    }
    for i in range(1, 8)
]
PLAIN_NB_WORKERS = 3

# --- weighted: mirrors stage_writelisting.m's 8-file %g-boundary probe case ---
_WEIGHTED_LANG_DIAL = [("eng", "us"), ("fra", "ca")] * 4
WEIGHTED_ITEMS = [
    {"filename": f"wfile_{i:02d}.wav", "refseg": f"wseg_{i:02d}.xml", "lang": lang, "dial": dial} for i, (lang, dial) in enumerate(_WEIGHTED_LANG_DIAL, start=1)
]
# weight (filesValues(:,2)), duration -- see stage_writelisting.m's header comment for
# which %g boundary each row probes.
WEIGHTED_WEIGHTS = [1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 1234567.0, -0.0]
WEIGHTED_DURATIONS = [30.0, 123456.789, 1000000.0, 100000.0, 0.0001, 0.00001, 999999.5, 0.0]
WEIGHTED_VALUES = np.array(list(zip(WEIGHTED_WEIGHTS, WEIGHTED_DURATIONS, strict=True)), dtype=np.float64)


def test_write_listing_plain_matches_golden(tmp_path: Path) -> None:
    base = tmp_path / "plain"
    write_listing(base, PLAIN_ITEMS, PLAIN_NB_WORKERS)
    got = (tmp_path / "plain.flst").read_bytes()
    want = (LISTING_DIR / "plain.flst").read_bytes()
    assert got == want


def test_write_listing_worker_shards_match_golden(tmp_path: Path) -> None:
    base = tmp_path / "plain"
    write_listing(base, PLAIN_ITEMS, PLAIN_NB_WORKERS)
    for jj in (1, 2, 3):
        got = (tmp_path / f"plain_worker_{jj}.flst").read_bytes()
        want = (LISTING_DIR / f"plain_worker_{jj}.flst").read_bytes()
        assert got == want, f"worker {jj} shard mismatch"


def test_write_listing_worker_shards_partition_all_items_exactly_once() -> None:
    """Non-vacuity: 7 items / 3 workers (7=3+2+2) -- every item must appear in exactly
    one shard, proving the fliplr interleave is a genuine partition, not an artifact of
    a degenerate small case."""
    seen: list[int] = []
    for jj in (1, 2, 3):
        seen.extend(_worker_positions(len(PLAIN_ITEMS), jj, PLAIN_NB_WORKERS))
    assert sorted(seen) == list(range(len(PLAIN_ITEMS)))


def test_worker_positions_hand_derived() -> None:
    """Pins the brief's by-hand derivation directly: jj=1 -> [1,4,7], jj=2 -> [3,6],
    jj=3 -> [2,5] (1-based MATLAB positions), 0-based here."""
    assert _worker_positions(7, 1, 3) == [0, 3, 6]
    assert _worker_positions(7, 2, 3) == [2, 5]
    assert _worker_positions(7, 3, 3) == [1, 4]


def test_write_listing_nb_workers_capped_at_item_count(tmp_path: Path) -> None:
    """`PS.VP.nbworker = min(PS.VP.nbworker, length(index))` -- requesting more workers
    than items must not create empty/out-of-range shard files."""
    base = tmp_path / "small"
    items = PLAIN_ITEMS[:2]
    write_listing(base, items, nb_workers=5)
    assert (tmp_path / "small_worker_1.flst").is_file()
    assert (tmp_path / "small_worker_2.flst").is_file()
    assert not (tmp_path / "small_worker_3.flst").is_file()


def test_write_weighted_listing_matches_golden(tmp_path: Path) -> None:
    path = tmp_path / "weighted.lst"
    write_weighted_listing(path, WEIGHTED_ITEMS, WEIGHTED_VALUES)
    got = path.read_bytes()
    want = (LISTING_DIR / "weighted.lst").read_bytes()
    assert got == want


def test_write_weighted_listing_no_suffix_appended(tmp_path: Path) -> None:
    """Unlike write_listing, write_weighted_listing writes `path` AS GIVEN -- no
    `.flst` suffix appended (the worker-shard block that would append one is commented
    out in the legacy source)."""
    path = tmp_path / "custom_name_no_ext"
    write_weighted_listing(path, WEIGHTED_ITEMS[:1], WEIGHTED_VALUES[:1])
    assert path.is_file()
    assert not (tmp_path / "custom_name_no_ext.flst").is_file()
    assert not (tmp_path / "custom_name_no_ext.lst").is_file()


# --- %g formatting: targeted boundary checks (redundant with the golden above, but pin
# each style-switch boundary by name) ---------------------------------------------------


def test_weighted_listing_g_format_integer() -> None:
    lines = _rows_only(WEIGHTED_ITEMS[:1], WEIGHTED_VALUES[:1])
    assert lines[0].endswith(";1;30;\n")


def test_weighted_listing_g_format_six_sig_fig_rounding() -> None:
    lines = _rows_only(WEIGHTED_ITEMS[1:2], WEIGHTED_VALUES[1:2])
    assert ";0.5;123457;\n" in lines[0]


def test_weighted_listing_g_format_exponent_switch_at_1e6() -> None:
    lines = _rows_only(WEIGHTED_ITEMS[2:3], WEIGHTED_VALUES[2:3])
    assert ";2;1e+06;\n" in lines[0]


def test_weighted_listing_g_format_stays_decimal_at_1e5() -> None:
    lines = _rows_only(WEIGHTED_ITEMS[3:4], WEIGHTED_VALUES[3:4])
    assert ";3;100000;\n" in lines[0]


def test_weighted_listing_g_format_stays_decimal_at_1e_minus4() -> None:
    lines = _rows_only(WEIGHTED_ITEMS[4:5], WEIGHTED_VALUES[4:5])
    assert ";4;0.0001;\n" in lines[0]


def test_weighted_listing_g_format_exponent_switch_at_1e_minus5() -> None:
    lines = _rows_only(WEIGHTED_ITEMS[5:6], WEIGHTED_VALUES[5:6])
    assert ";5;1e-05;\n" in lines[0]


def test_weighted_listing_g_format_rounding_carries_across_exponent_boundary() -> None:
    lines = _rows_only(WEIGHTED_ITEMS[6:7], WEIGHTED_VALUES[6:7])
    assert ";1.23457e+06;1e+06;\n" in lines[0]


def test_weighted_listing_g_format_negative_zero() -> None:
    lines = _rows_only(WEIGHTED_ITEMS[7:8], WEIGHTED_VALUES[7:8])
    assert ";-0;0;\n" in lines[0]


def _rows_only(items: list[dict[str, str]], values: np.ndarray) -> list[str]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.lst"
        write_weighted_listing(path, items, values)
        return path.read_text().splitlines(keepends=True)
