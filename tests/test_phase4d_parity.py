"""Phase 4d Task 4: the PyO3-seam end-to-end PARITY gates.

The Rust twin (`src/rust/tests/phase4d_parity.rs`) replays the committed Phase 4d
oracle fixtures through `CorpusProcessor` directly; THIS leg drives the SAME
fixtures through the in-process PyO3 seam (`speech_rs.Engine`), proving the seam
end to end against the 2015 `-ffast-math` production build. Both assert:

  (a) STRUCTURAL EXACT per channel -- speech-segment count + ordered boundary
      sequence vs the oracle VRCTS xml (a mismatch is a HARD failure, the parity
      floor);
  (b) per-boundary |dt| <= the manifest's pinned `pinned_abs_s` (one 4-decimal
      VRCTS display quantum; measured 0.0 -- byte-identical VRCTS);
  (c) `MultiConfigResults` columns vs `parity_*_mcr.bin`: id/count columns EXACT,
      wall-clock timing (col 6) masked, WER/LID columns (11..=19) asserted 0, and
      the scored data columns within `ULP<=4 OR |diff|<=512*eps*max(|oracle|,1)`
      (the cross-libm floor -- the `-ffast-math` oracle is not bit-matchable).

Tolerances are READ FROM THE MANIFEST (`task4_measured_deltas`, filled by
`scripts/extract_phase4d_fixtures.py --measure-deltas`). CI-safe: needs only the
committed fixtures + the built `speech_rs` module (`importorskip` skips cleanly
where it is absent), never the oracle / `fsp_runtime` / fastmath build / legacy
tree.

The audio-gated `test_tupleA_replay_vs_saved_2015_results` is LOCAL-ONLY by
construction (the OpenSAD15 corpus is not committed). Activation procedure:

  1. Point `OPENSAD15_AUDIO_ROOT` at a directory containing the tuple-A original
     inputs -- the un-localized 2015 config `1_worker_1.config`, its `fileslisting`
     + `language2classmapping.csv`, the referenced wav/stm corpus, the weight pack
     `NNweights_config1.bin`, and the SAVED 2015 `MultiConfigResults_worker_1.mat`.
  2. `OPENSAD15_AUDIO_ROOT=/path/to/tupleA uv run pytest
     tests/test_phase4d_parity.py::test_tupleA_replay_vs_saved_2015_results`.

It is NOT a phase exit criterion (user decision: audio maybe-later).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE4D = REPO_ROOT / "tests" / "reference_data" / "phase4d"

LANGMAP_TEXT = "fax;non;0\nchi;man;1\nspa;spa;2\n"
FILESLISTING_TEXT = "prcts_excerpt.wav;prcts_excerpt.stm;fax;non;1;30.0;\n"

# (weights pack, expected [chan1, chan2] speech-segment counts) per tuple -- the
# manifest `fixtures.*.speech_segments`, also the extractor's non-vacuity floor.
TUPLES = {
    "tupleA": ("tupleA_NNweights_config1.bin", (8, 3)),
    "tupleB": ("tupleB_NNweights_config1.bin", (13, 15)),
}


@contextmanager
def chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def manifest() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads((PHASE4D / "phase4d_parity_manifest.json").read_text()))


def pinned(m: dict[str, Any], tuple_name: str, leg: str, field: str) -> float:
    return float(m["task4_measured_deltas"][tuple_name][leg][field])


def read_bin(path: Path) -> np.ndarray:
    """io::binary .bin: i64 LE rows, i64 LE cols, f64 LE column-major."""
    raw = path.read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    return np.frombuffer(raw[16:], dtype="<f8").reshape((rows, cols), order="F")


def seg_pairs(xml_path: Path) -> list[tuple[float, float]]:
    """Ordered (stime, etime) pairs from a VRCTS xml (also validates it parses)."""
    root = ET.fromstring(xml_path.read_text())
    return [(float(s.attrib["stime"]), float(s.attrib["etime"])) for s in root.iter("SpeechSegment")]


def with_dump_dir(config_text: str, dump_dir: str) -> str:
    out, seen = [], False
    for line in config_text.splitlines():
        if line.startswith("Dump_Directory "):
            out.append(f"Dump_Directory {dump_dir}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"Dump_Directory {dump_dir}")
    return "\n".join(out) + "\n"


def seed(workdir: Path, tuple_name: str, config_text: str, weights_name: str) -> str:
    shutil.copy2(PHASE4D / "prcts_excerpt.wav", workdir / "prcts_excerpt.wav")
    shutil.copy2(PHASE4D / "prcts_excerpt.stm", workdir / "prcts_excerpt.stm")
    shutil.copy2(PHASE4D / weights_name, workdir / "NNweights_config1.bin")
    (workdir / "languagemapping.csv").write_text(LANGMAP_TEXT)
    (workdir / "fileslisting").write_text(FILESLISTING_TEXT)
    config_name = f"parity_{tuple_name}.config"
    (workdir / config_name).write_text(config_text)
    return config_name


def _ulp_dist(a: float, b: float) -> float:
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or (np.signbit(a) != np.signbit(b)):
        return float("inf")
    ai = int(np.array(a, dtype="<f8").view("<i8"))
    bi = int(np.array(b, dtype="<f8").view("<i8"))
    return float(abs(ai - bi))


def _hybrid_ok(a: float, b: float, ulp_bound: float, abs_tol: float) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return _ulp_dist(a, b) <= ulp_bound or abs(a - b) <= abs_tol


@pytest.mark.parametrize("tuple_name", list(TUPLES))
def test_vrcts_structural_and_boundaries(tuple_name: str, tmp_path: Path) -> None:
    """(a)+(b): the port's -s VRCTS matches the oracle's structurally (exact speech
    count + ordered boundaries) and per-boundary within the pinned display quantum."""
    weights_name, expected = TUPLES[tuple_name]
    m = manifest()
    pinned_dt = pinned(m, tuple_name, "port_vs_oracle_boundary_dt", "pinned_abs_s")

    config_text = with_dump_dir((PHASE4D / f"parity_{tuple_name}.config").read_text(), "vrcts_out")
    config_name = seed(tmp_path, tuple_name, config_text, weights_name)
    (tmp_path / "vrcts_out").mkdir(exist_ok=True)
    with chdir(tmp_path):
        eng = speech_rs.Engine([config_name], "-s")
        eng.run()

    for chan0, expected_segs in enumerate(expected):
        chan = chan0 + 1
        port_xml = tmp_path / "vrcts_out" / f"prcts_excerpt_chan_{chan}.xml"
        oracle_xml = PHASE4D / f"parity_{tuple_name}_vrcts_chan{chan}.xml"
        assert port_xml.is_file(), f"{tuple_name} chan{chan}: port wrote no VRCTS xml"

        port = seg_pairs(port_xml)
        oracle = seg_pairs(oracle_xml)

        # Non-vacuity: oracle carries the manifest speech count (>1 -> the decision
        # layer is genuinely exercised).
        assert len(oracle) == expected_segs, f"{tuple_name} chan{chan}: oracle speech count"
        assert expected_segs > 1, f"{tuple_name} chan{chan}: excerpt must be non-degenerate"

        # (a) STRUCTURAL EXACT: identical ordered speech-segment sequence. HARD.
        assert len(port) == len(oracle), f"{tuple_name} chan{chan}: STRUCTURAL count mismatch port={len(port)} oracle={len(oracle)}"
        # (b) per-boundary |dt| <= the pinned display quantum (both stime and etime).
        for i, ((ps, pe), (os_, oe)) in enumerate(zip(port, oracle, strict=True)):
            assert abs(ps - os_) <= pinned_dt, f"{tuple_name} chan{chan}: seg {i} stime |dt|={abs(ps - os_):.3e} > {pinned_dt:.3e}"
            assert abs(pe - oe) <= pinned_dt, f"{tuple_name} chan{chan}: seg {i} etime |dt|={abs(pe - oe):.3e} > {pinned_dt:.3e}"


@pytest.mark.parametrize("tuple_name", list(TUPLES))
def test_mcr_columns(tuple_name: str, tmp_path: Path) -> None:
    """(c): the port's -m MultiConfigResults matches the oracle's -- id/count columns
    EXACT, timing masked, WER/LID columns 0, scored columns within the pinned floor."""
    weights_name, _ = TUPLES[tuple_name]
    m = manifest()
    ulp_bound = pinned(m, tuple_name, "port_vs_oracle_mcr_col_deltas", "pinned_ulp")
    abs_factor = pinned(m, tuple_name, "port_vs_oracle_mcr_col_deltas", "pinned_abs_factor")
    eps = pinned(m, tuple_name, "port_vs_oracle_mcr_col_deltas", "eps")

    config_text = (PHASE4D / f"parity_{tuple_name}.config").read_text()
    config_name = seed(tmp_path, tuple_name, config_text, weights_name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([config_name], "-m")
        eng.run()
        got = np.atleast_2d(np.asarray(eng.results_matrix(), dtype="<f8"))

    want = read_bin(PHASE4D / f"parity_{tuple_name}_mcr.bin")
    assert got.shape == want.shape == (2, 21), f"{tuple_name}: MCR shape {got.shape} vs {want.shape}"

    for r in range(2):
        for c in range(21):
            g, w = float(got[r, c]), float(want[r, c])
            if c == 6:
                continue  # wall-clock timing: masked.
            if c in (0, 1, 2, 10, 20):
                assert g == w, f"{tuple_name}: MCR[{r},{c}] id/count must be EXACT: {g} vs {w}"
            elif 11 <= c <= 19:
                assert w == 0.0, f"{tuple_name}: fixture MCR[{r},{c}] expected 0"
                assert g == 0.0, f"{tuple_name}: MCR[{r},{c}] WER/LID must be 0 in port, got {g}"
            else:
                abs_tol = abs_factor * eps * max(abs(w), 1.0)
                assert _hybrid_ok(g, w, ulp_bound, abs_tol), (
                    f"{tuple_name}: MCR[{r},{c}] scored column outside pinned floor: |diff|={abs(g - w):.3e} ULP={_ulp_dist(g, w)} got {g} want {w}"
                )


# --- Audio-gated tuple-A replay against the SAVED 2015 results (local-only) ------

OPENSAD15_AUDIO_ROOT = Path(os.environ.get("OPENSAD15_AUDIO_ROOT", "/nonexistent-opensad15-root"))


@pytest.mark.skipif(
    not OPENSAD15_AUDIO_ROOT.exists(),
    reason="OPENSAD15_AUDIO_ROOT not set/present (local-only replay; see module docstring)",
)
def test_tupleA_replay_vs_saved_2015_results(tmp_path: Path) -> None:
    """Run the port over tuple A's ORIGINAL OpenSAD15 listing and compare against the
    SAVED 2015 `MultiConfigResults_worker_1.mat` (converted via scipy at runtime -- the
    only place this module touches a `.mat`, since it is local-only by construction).
    Not a phase exit criterion; see the module docstring for the activation procedure."""
    import scipy.io  # local-only import: never needed on the committed CI path.

    config = OPENSAD15_AUDIO_ROOT / "1_worker_1.config"
    saved_mat = OPENSAD15_AUDIO_ROOT / "MultiConfigResults_worker_1.mat"
    for required in (config, saved_mat):
        if not required.is_file():
            pytest.skip(f"replay input missing: {required}")

    with chdir(OPENSAD15_AUDIO_ROOT):
        eng = speech_rs.Engine([config.name], "-m")
        eng.run()
        got = np.atleast_2d(np.asarray(eng.results_matrix(), dtype="<f8"))

    want = np.atleast_2d(np.asarray(scipy.io.loadmat(saved_mat)["MultiConfigResults"], dtype="<f8"))
    assert got.shape == want.shape, f"replay MCR shape {got.shape} vs saved {want.shape}"

    m = manifest()
    abs_factor = pinned(m, "tupleA", "port_vs_oracle_mcr_col_deltas", "pinned_abs_factor")
    eps = pinned(m, "tupleA", "port_vs_oracle_mcr_col_deltas", "eps")
    ulp_bound = pinned(m, "tupleA", "port_vs_oracle_mcr_col_deltas", "pinned_ulp")
    for r in range(want.shape[0]):
        for c in range(want.shape[1]):
            if c == 6:  # wall-clock timing.
                continue
            g, w = float(got[r, c]), float(want[r, c])
            abs_tol = abs_factor * eps * max(abs(w), 1.0)
            assert _hybrid_ok(g, w, ulp_bound, abs_tol), f"replay MCR[{r},{c}] outside pinned floor: |diff|={abs(g - w):.3e} got {g} want {w}"
