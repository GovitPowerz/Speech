"""Phase 4c Task 6 fixture guards.

The extractor (`scripts/extract_phase4c_fixtures.py`) runs the Octave harness stages and
commits per-step SMORMS3/Rprop state trajectories. These tests guard the committed
fixtures WITHOUT Octave (which is a local-only dep): the manifest is present and records
the measured octave version + step counts, every referenced `.bin` exists and parses to
its recorded shape, and the pinned non-vacuities (lrate warmup+cap, nonzero round-trip
offset, rprop branch coverage) hold.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

PHASE4C = Path(__file__).resolve().parent / "reference_data" / "phase4c"
MANIFEST = PHASE4C / "manifest.json"


def _read_bin(path: Path) -> tuple[int, int, list[float]]:
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    n = rows * cols
    data = list(struct.unpack(f"<{n}d", raw[16 : 16 + 8 * n]))
    return rows, cols, data


def _manifest() -> dict[str, object]:
    return cast(dict[str, object], json.loads(MANIFEST.read_text()))


def test_manifest_present_and_records_octave_version() -> None:
    assert MANIFEST.is_file(), "phase4c manifest.json must be committed"
    m = _manifest()
    assert "smorms3" in m and "rprop" in m
    version = cast(str, m["octave_version"])
    assert version and version[0].isdigit(), f"octave_version {version!r} not recorded/parsed"


def test_smorms3_bins_exist_and_match_manifest_shapes() -> None:
    m = _manifest()
    shapes = cast(dict[str, list[int]], cast(dict[str, object], m["smorms3"])["shapes"])
    for name, shape in shapes.items():
        rows, cols, _ = _read_bin(PHASE4C / name)
        assert (rows, cols) == tuple(shape), f"{name}: shape ({rows},{cols}) != manifest {shape}"


def test_rprop_bins_exist_and_match_manifest_shapes() -> None:
    m = _manifest()
    shapes = cast(dict[str, list[int]], cast(dict[str, object], m["rprop"])["shapes"])
    for name, shape in shapes.items():
        rows, cols, _ = _read_bin(PHASE4C / name)
        assert (rows, cols) == tuple(shape), f"{name}: shape ({rows},{cols}) != manifest {shape}"


def test_smorms3_lrate_warmup_and_cap_recorded() -> None:
    """The manifest's lrate_traj and the committed .bin both show the x10 warmup capped at
    1e-1: 1e-8..1e-1 (step 8) then a plateau."""
    m = _manifest()
    lrate_manifest = cast(list[float], cast(dict[str, object], m["smorms3"])["lrate_traj"])
    _, _, lrate_bin = _read_bin(PHASE4C / "smorms3_lrate_traj.bin")
    assert lrate_manifest == lrate_bin, "manifest lrate_traj != committed .bin"
    # The iterated min(lrate*10, 1e-1) reference (SMORMS3.m:347): step 4 rounds to
    # 9.999999999999999e-06, NOT the literal 1e-5, so compare against the iterated form.
    exp: list[float] = []
    lr = 1e-9
    for _ in range(len(lrate_bin)):
        lr = min(lr * 10.0, 1e-1)
        exp.append(lr)
    assert [v.hex() for v in lrate_bin] == [v.hex() for v in exp], "lrate_traj != iterated x10 warmup/cap"
    assert lrate_bin[0] == 1e-8 and lrate_bin[7] == 1e-1
    assert all(lrate_bin[i] < lrate_bin[i + 1] for i in range(7)), "x10 warmup must strictly increase"
    assert all(v == 1e-1 for v in lrate_bin[7:]), "lrate must plateau at the 1e-1 cap from step 8"


def test_smorms3_roundtrip_offset_nonzero() -> None:
    _, _, offset = _read_bin(PHASE4C / "smorms3_rt_offset.bin")
    assert any(v != 0.0 for v in offset), "round-trip offset must be nonzero"


def test_rprop_delta_spans_delta0_both_ways() -> None:
    """delta0 = 0.01; the etap-grow branch pushes some delta > 0.01 and the etam-shrink
    branch some delta < 0.01 -- both must be present."""
    _, _, delta = _read_bin(PHASE4C / "rprop_delta_traj.bin")
    assert max(delta) > 0.01, "no delta grew above delta0 (etap branch never fired)"
    assert min(delta) < 0.01, "no delta shrank below delta0 (etam branch never fired)"


def test_rprop_derivative_zeroing_present() -> None:
    dr, dc, deriv_in = _read_bin(PHASE4C / "rprop_deriv_script.bin")
    _, _, deriv_out = _read_bin(PHASE4C / "rprop_derivout_traj.bin")
    assert any(o == 0.0 and i != 0.0 for i, o in zip(deriv_in, deriv_out, strict=True)), "no derivative was zeroed -- the <0 sign-flip branch never fired"


# --- Task 8: vec2struct genome-bijection fixtures ---
GENOME_MANIFEST = PHASE4C / "genome_manifest.json"
GENOME_CASES = ["algo0", "tdc", "calib", "spectral", "twin", "masked"]


def _genome_manifest() -> dict[str, object]:
    return cast(dict[str, object], json.loads(GENOME_MANIFEST.read_text()))


def test_genome_manifest_present_and_records_octave_version() -> None:
    assert GENOME_MANIFEST.is_file(), "phase4c genome_manifest.json must be committed"
    m = _genome_manifest()
    version = cast(str, m["octave_version"])
    assert version and version[0].isdigit(), f"octave_version {version!r} not recorded/parsed"
    assert set(cast(dict[str, int], m["counts"])) == set(GENOME_CASES)


def test_genome_bins_exist_and_match_manifest_shapes() -> None:
    m = _genome_manifest()
    shapes = cast(dict[str, list[int]], m["shapes"])
    for name, shape in shapes.items():
        rows, cols, _ = _read_bin(PHASE4C / name)
        assert (rows, cols) == tuple(shape), f"{name}: shape ({rows},{cols}) != manifest {shape}"


def test_genome_configs_and_count_consistency() -> None:
    """Each case's .config is present and non-empty, and count_param == len(param)+1."""
    m = _genome_manifest()
    counts = cast(dict[str, int], m["counts"])
    for case in GENOME_CASES:
        conf = PHASE4C / f"genome_{case}.config"
        assert conf.is_file() and conf.read_text().strip(), f"{conf.name} missing/empty"
        rows, _, _ = _read_bin(PHASE4C / f"genome_{case}_param.bin")
        assert counts[case] == rows + 1, f"{case}: count {counts[case]} != len(param)+1 ({rows + 1})"
