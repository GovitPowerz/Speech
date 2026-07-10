"""Phase 4d Task 3 parity-fixture guards.

The extractor (`scripts/extract_phase4d_fixtures.py`) drives the SPLIT, DUAL-MODE oracle
-- the resurrected 2015 `Segment` binary for the VRCTS segmentation leg (-s) and a
-ffast-math rebuild of the legacy CorpusProcessor for the MultiConfigResults numeric
columns, run in TWO modes (-s for VRCTS structural calibration, -m for the SCORED numeric
columns) -- and commits the excerpt wav + a real derived reference .stm + localized
configs + oracle outputs + manifest. These tests guard the committed fixtures WITHOUT any
oracle (both are local-only): the manifest is present and records the excerpt cut +
per-artifact oracle provenance, every committed file's sha256 matches the manifest, every
VRCTS xml parses to its recorded (non-vacuous) segment count, the .bin numeric fixtures
parse to their recorded shapes with the wall-clock column masked, the -m leg's mcr columns
are genuinely SCORED (non-vacuous Pfa/NbOfClassif -- the Task 3 fix-wave's core guard,
correcting the original writeup's exit(1) rationale), and the whole phase4d dir stays
under the 8 MB budget.
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


def test_mcr_scored_leg_is_non_vacuous() -> None:
    """Task 3 fix-wave non-vacuity guard: the -m leg is genuinely SCORED via the excerpt
    reference .stm, so Pfa/globalError/cumulativeError/NbOfClassif must not be all-zero
    (the near-vacuous state the T3 review flagged and this fix-wave corrected)."""
    m = _manifest()
    fixtures = cast(dict[str, dict[str, object]], m["fixtures"])
    for tuple_name in TUPLES:
        assert fixtures[f"parity_{tuple_name}_mcr.bin"]["mode"] == "m"
        assert fixtures[f"parity_{tuple_name}_mcr.bin"]["scored"] is True
        mcr_rows, mcr_cols, mcr = _read_bin(PHASE4D / f"parity_{tuple_name}_mcr.bin")
        # column-major: col c occupies mcr[c*rows : (c+1)*rows]. col 3 = Pfa, last col =
        # NbOfClassif (see the manifest's mcr_columns doc).
        pfa = mcr[3 * mcr_rows : 4 * mcr_rows]
        nb_of_classif = mcr[(mcr_cols - 1) * mcr_rows : mcr_cols * mcr_rows]
        assert any(v != 0.0 for v in pfa), f"{tuple_name}: Pfa column is still all-zero (vacuous)"
        assert all(v > 0.0 for v in nb_of_classif), f"{tuple_name}: NbOfClassif column is still zero (vacuous)"


def test_excerpt_stm_derivation_recorded_and_ascii() -> None:
    """The reference .stm fixture is a real (non-empty) derived excerpt: ASCII-only,
    committed with its derivation recorded in the manifest (source .stm, window, the
    placeholder token that replaced the transcript field)."""
    m = _manifest()
    stm_path = PHASE4D / "prcts_excerpt.stm"
    assert stm_path.is_file()
    text = stm_path.read_text()
    assert text.isascii(), "prcts_excerpt.stm must be ASCII-only"
    assert text.strip(), "prcts_excerpt.stm must not be empty (it must enable scoring)"
    ex = cast(dict[str, object], m["excerpt_stm"])
    assert ex["window_sec"] == 60.0
    assert cast(str, ex["placeholder_token"]).strip()
    assert cast(int, ex["lines_kept"]) > 0
    fixtures = cast(dict[str, dict[str, object]], m["fixtures"])
    assert fixtures["prcts_excerpt.stm"]["sha256"] == hashlib.sha256(stm_path.read_bytes()).hexdigest()


def test_stm_leg_vrcts_mode_invariance_recorded() -> None:
    """The manifest records the corrected exit(1) rationale and the mode-invariance
    verification (the -s VRCTS leg is unaffected by the now-real reference .stm)."""
    m = _manifest()
    stm_leg = cast(dict[str, str], m["stm_leg"])
    assert "FAILED TO OPEN" in stm_leg["exit1_rationale_corrected"]
    assert stm_leg["vrcts_mode_invariance"].strip()


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


def test_task4_measured_deltas_filled() -> None:
    """Task 4 filled the placeholders (measured-then-pinned tolerances the parity
    gates read). Guard the shape + the load-bearing invariants: boundary dt measured
    0.0 pinned to one display quantum, mcr measured at fp bit-noise (well below the
    1e-3 STOP line) pinned to the cross-libm floor params."""
    m = _manifest()
    t4 = cast(dict[str, dict[str, object]], m["task4_measured_deltas"])
    for tuple_name in TUPLES:
        assert tuple_name in t4
        dt = cast(dict[str, object], t4[tuple_name]["port_vs_oracle_boundary_dt"])
        assert dt["measured_max_s"] == 0.0, f"{tuple_name}: VRCTS not byte-identical"
        assert dt["pinned_abs_s"] == 1e-4, f"{tuple_name}: dt pin must be one display quantum"

        mcr = cast(dict[str, object], t4[tuple_name]["port_vs_oracle_mcr_col_deltas"])
        assert cast(float, mcr["measured_max_rel"]) < 1e-3, f"{tuple_name}: mcr rel past STOP line"
        assert cast(float, mcr["measured_max_abs"]) < 1e-6, f"{tuple_name}: mcr abs past STOP line"
        assert mcr["pinned_ulp"] == 4
        assert mcr["pinned_abs_factor"] == 512.0
        assert cast(float, mcr["eps"]) > 0.0


def test_size_budget() -> None:
    total = sum(p.stat().st_size for p in PHASE4D.iterdir() if p.is_file())
    assert total < SIZE_BUDGET_BYTES, f"phase4d fixtures {total} bytes >= {SIZE_BUDGET_BYTES} budget"
    # The manifest records the staged total (computed before the manifest itself is
    # written), so it must be <= the on-disk dir total (which includes the manifest) and
    # under budget.
    recorded = cast(dict[str, int], _manifest()["size_budget"])["total_bytes"]
    assert recorded < SIZE_BUDGET_BYTES
    assert recorded <= total
