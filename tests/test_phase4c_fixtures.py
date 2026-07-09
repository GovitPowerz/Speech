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


# --- Task 10: batching + CheckGrad + MaskingValidation fixtures ---


def test_batching_manifest_present() -> None:
    m = _manifest()
    assert "batching" in m, "phase4c manifest.json missing the batching section"
    b = cast(dict[str, object], m["batching"])
    assert b["degenerate_batch_len"] == 3


def test_batching_bins_exist_and_match_manifest_shapes() -> None:
    m = _manifest()
    shapes = cast(dict[str, list[int]], cast(dict[str, object], m["batching"])["shapes"])
    for name, shape in shapes.items():
        rows, cols, _ = _read_bin(PHASE4C / name)
        assert (rows, cols) == tuple(shape), f"{name}: shape ({rows},{cols}) != manifest {shape}"


def test_batching_clobber_quirk_recorded() -> None:
    _, _, clobber = _read_bin(PHASE4C / "batching_clobber_case3_index.bin")
    assert sorted(clobber) == [5.0, 6.0], "clobber_case3_index must be class-2's data, not class-0's (the quirk)"


def test_checkgrad_manifest_present_and_kw() -> None:
    m = _manifest()
    assert "checkgrad" in m, "phase4c manifest.json missing the checkgrad section"
    cg = cast(dict[str, object], m["checkgrad"])
    assert cg["Kw"] == 53 and cg["n"] == 107


def test_checkgrad_normalize_tail_quirk_recorded() -> None:
    """The last 2 (2*len(normalize.mean)) numeric derivatives are always exactly 0
    (weights2nnet.m never writes back the normalize tail), while the analytic ones are
    genuinely nonzero -- else the quirk fixture would be vacuous."""
    _, _, analytic = _read_bin(PHASE4C / "checkgrad_MultiDeriv_BackProp.bin")
    _, _, numeric = _read_bin(PHASE4C / "checkgrad_MultiDeriv_Num.bin")
    assert numeric[-2:] == [0.0, 0.0]
    assert analytic[-2] != 0.0 and analytic[-1] != 0.0


def test_masking_manifest_present_and_pass_fail_split() -> None:
    m = _manifest()
    assert "masking" in m, "phase4c manifest.json missing the masking section"
    mk = cast(dict[str, object], m["masking"])
    assert (mk["pass_failed"], mk["fail_failed"]) == (0, 1)


# --- Task 11: QuantumPSO shared-random-table fixtures ---


def test_qpso_manifest_present_and_records_params() -> None:
    m = _manifest()
    assert "qpso" in m, "phase4c manifest.json missing the qpso section"
    q = cast(dict[str, object], m["qpso"])
    assert (q["D"], q["ps"], q["me"], q["K"]) == (3, 4, 3, 8)
    assert q["table_seed"] == 20260709 and q["table_n"] == 5000


def test_qpso_bins_exist_and_match_manifest_shapes() -> None:
    m = _manifest()
    shapes = cast(dict[str, list[int]], cast(dict[str, object], m["qpso"])["shapes"])
    for name, shape in shapes.items():
        rows, cols, _ = _read_bin(PHASE4C / name)
        assert (rows, cols) == tuple(shape), f"{name}: shape ({rows},{cols}) != manifest {shape}"


def test_qpso_random_table_present_and_sized() -> None:
    m = _manifest()
    q = cast(dict[str, object], m["qpso"])
    rows, cols, _ = _read_bin(PHASE4C / "qpso_random_table.bin")
    assert (rows, cols) == (q["table_n"], 1), f"random table shape ({rows},{cols}) != ({q['table_n']}, 1)"


def test_qpso_cursor_and_dedraw_nonvacuity() -> None:
    """cursor_end (the .bin, the manifest, and the base-draw floor all agree): DE/Levy fired
    (cursor beyond the no-candidate base), and gbest genuinely evolved across epochs."""
    m = _manifest()
    q = cast(dict[str, object], m["qpso"])
    _, _, cursor = _read_bin(PHASE4C / "qpso_cursor_end.bin")
    assert int(cursor[0]) == q["cursor_end"]
    ps, d, me = cast(int, q["ps"]), cast(int, q["D"]), cast(int, q["me"])
    base_draws = 6 * ps * d + me * (12 * ps * d + ps)
    assert int(cursor[0]) > base_draws, "cursor_end at/below base -- DE/Levy arms never fired"
    gr, gc, gbest = _read_bin(PHASE4C / "qpso_gbest_traj.bin")  # me x D column-major
    rows = {tuple(gbest[c * gr + i] for c in range(gc)) for i in range(gr)}
    assert len(rows) >= 2, "gbest never changed across epochs (the pin would be trivial)"


def test_qpso_levy_nonzero_and_input_is_table_prefix() -> None:
    _, _, levy_out = _read_bin(PHASE4C / "qpso_levy_out.bin")
    assert any(v != 0.0 for v in levy_out) and all(v == v for v in levy_out), "levy_out degenerate"
    _, _, levy_in = _read_bin(PHASE4C / "qpso_levy_in.bin")
    _, _, table = _read_bin(PHASE4C / "qpso_random_table.bin")
    assert levy_in == table[:16], "levy_in must be exactly table[0:16] (V=table[0:8], W=table[8:16])"


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
