"""Phase 4d Task 3 parity-fixture guards.

The extractor (`scripts/extract_phase4d_fixtures.py`) drives the SPLIT oracle -- the
resurrected 2015 `Segment` binary for the VRCTS segmentation legs and a -ffast-math
rebuild of the legacy CorpusProcessor for the MultiConfigResults numeric columns -- and
commits the excerpt + localized configs + oracle outputs + manifest. These tests guard
the committed fixtures WITHOUT any oracle (both are local-only): the manifest is present
and records the excerpt cut + per-artifact oracle provenance, every committed file's
sha256 matches the manifest, every VRCTS xml parses to its recorded (non-vacuous) segment
count, the .bin numeric fixtures parse to their recorded shapes with the wall-clock column
masked, and the whole phase4d dir stays under the 8 MB budget.
"""

from __future__ import annotations

import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

PHASE4D = Path(__file__).resolve().parent / "reference_data" / "phase4d"
MANIFEST = PHASE4D / "phase4d_parity_manifest.json"
TUPLES = ("tupleA", "tupleB")
SIZE_BUDGET_BYTES = 8 * 1024 * 1024


def _manifest() -> dict[str, object]:
    return cast(dict[str, object], json.loads(MANIFEST.read_text()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_bin(path: Path) -> tuple[int, int, list[float]]:
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    n = rows * cols
    return rows, cols, list(struct.unpack(f"<{n}d", raw[16 : 16 + 8 * n]))


def test_manifest_present_and_records_excerpt() -> None:
    assert MANIFEST.is_file(), "phase4d_parity_manifest.json must be committed"
    m = _manifest()
    ex = cast(dict[str, object], m["excerpt"])
    assert ex["start_sec"] == 0 and ex["length_sec"] == 60
    assert ex["samplerate"] == 8000 and ex["channels"] == 2
    assert cast(str, ex["adjudication"]).strip(), "excerpt adjudication must be recorded"


def test_manifest_records_split_oracle() -> None:
    m = _manifest()
    oracles = cast(dict[str, object], m["oracles"])
    assert "real_binary" in oracles and "fastmath_rebuild" in oracles
    fm = cast(dict[str, object], oracles["fastmath_rebuild"])
    assert "-ffast-math" in cast(str, fm["flags"])
    # The strict-IEEE flags must be recorded as DROPPED (this is the whole point).
    assert "-fno-fast-math" in cast(list[str], fm["dropped_strict_ieee_flags"])
    assert "-DEIGEN_DONT_VECTORIZE" in cast(list[str], fm["dropped_strict_ieee_flags"])
    assert cast(list[str], fm["dropped_x86_flags"]), "dropped x86-only flags must be recorded"


def test_every_fixture_sha_matches_manifest() -> None:
    m = _manifest()
    fixtures = cast(dict[str, dict[str, object]], m["fixtures"])
    assert fixtures, "manifest lists no fixtures"
    for name, rec in fixtures.items():
        path = PHASE4D / name
        assert path.is_file(), f"committed fixture {name} missing"
        assert _sha256(path) == rec["sha256"], f"{name}: sha256 != manifest"


def test_vrcts_parses_and_is_non_vacuous() -> None:
    """Every VRCTS xml parses and carries >1 speech segment per channel (non-vacuity),
    matching the manifest's recorded count."""
    m = _manifest()
    fixtures = cast(dict[str, dict[str, object]], m["fixtures"])
    seen = 0
    for name, rec in fixtures.items():
        if not name.endswith(".xml"):
            continue
        seen += 1
        assert rec["oracle"] == "real-binary", f"{name}: VRCTS oracle must be the real binary"
        root = ET.fromstring((PHASE4D / name).read_text())
        segs = list(root.iter("SpeechSegment"))
        assert len(segs) == rec["speech_segments"], f"{name}: segment count != manifest"
        assert len(segs) > 1, f"{name}: degenerate ({len(segs)} speech segment(s))"
    assert seen == 4, f"expected 4 VRCTS fixtures (2 tuples x 2 channels), found {seen}"


def test_mcr_bins_shape_and_timing_masked() -> None:
    """mcr.bin parses to its recorded shape, the wall-clock column (col 6) is masked to
    0.0, and result_rows.bin equals mcr[:, 3:]."""
    m = _manifest()
    fixtures = cast(dict[str, dict[str, object]], m["fixtures"])
    masked_col = cast(int, m["mcr_masked_col"])
    for tuple_name in TUPLES:
        mcr_rows, mcr_cols, mcr = _read_bin(PHASE4D / f"parity_{tuple_name}_mcr.bin")
        assert [mcr_rows, mcr_cols] == fixtures[f"parity_{tuple_name}_mcr.bin"]["shape"]
        assert cast(str, fixtures[f"parity_{tuple_name}_mcr.bin"]["oracle"]) == "fastmath-rebuild"
        # column-major: col c occupies mcr[c*rows : (c+1)*rows].
        timing = mcr[masked_col * mcr_rows : (masked_col + 1) * mcr_rows]
        assert all(v == 0.0 for v in timing), f"{tuple_name}: mcr timing col not masked to 0"
        rr_rows, rr_cols, rows_data = _read_bin(PHASE4D / f"parity_{tuple_name}_result_rows.bin")
        assert rr_rows == mcr_rows and rr_cols == mcr_cols - 3, "result_rows must be mcr[:, 3:]"
        # result_rows == mcr with the first 3 (prefix) columns dropped.
        assert rows_data == mcr[3 * mcr_rows :], f"{tuple_name}: result_rows != mcr[:, 3:]"


def test_config_localization_recorded() -> None:
    """Each localized parity config is present and its key-diff (path keys only) is
    recorded; every localized key resolves to a relative (test-cwd) name."""
    m = _manifest()
    loc = cast(dict[str, dict[str, object]], m["config_localization"])
    for tuple_name in TUPLES:
        cfg = PHASE4D / f"parity_{tuple_name}.config"
        assert cfg.is_file() and cfg.read_text().strip()
        key_diff = cast(list[dict[str, str]], loc[tuple_name]["key_diff"])
        assert key_diff, f"{tuple_name}: no key-diff recorded"
        for entry in key_diff:
            assert not entry["localized"].startswith("/"), f"{tuple_name}: {entry['key']} not localized"


def test_structural_agreement_real_vs_fastmath() -> None:
    """The two oracles agree structurally (equal segment counts) on both channels of both
    tuples -- the calibration evidence the split oracle rests on."""
    m = _manifest()
    struct_ = cast(dict[str, dict[str, dict[str, object]]], m["structural_agreement_real_vs_fastmath"])
    for tuple_name in TUPLES:
        for chan in ("chan1", "chan2"):
            rec = struct_[tuple_name][chan]
            assert rec["structural_match"] is True, f"{tuple_name} {chan}: oracles disagree structurally"
            assert rec["real_segs"] == rec["fastmath_segs"]


def test_task4_placeholders_present_and_empty() -> None:
    m = _manifest()
    t4 = cast(dict[str, dict[str, object]], m["task4_measured_deltas"])
    for tuple_name in TUPLES:
        assert tuple_name in t4
        assert t4[tuple_name]["port_vs_oracle_boundary_dt"] is None
        assert t4[tuple_name]["port_vs_oracle_mcr_col_deltas"] is None


def test_size_budget() -> None:
    total = sum(p.stat().st_size for p in PHASE4D.iterdir() if p.is_file())
    assert total < SIZE_BUDGET_BYTES, f"phase4d fixtures {total} bytes >= {SIZE_BUDGET_BYTES} budget"
    # The manifest records the staged total (computed before the manifest itself is
    # written), so it must be <= the on-disk dir total (which includes the manifest) and
    # under budget.
    recorded = cast(dict[str, int], _manifest()["size_budget"])["total_bytes"]
    assert recorded < SIZE_BUDGET_BYTES
    assert recorded <= total
