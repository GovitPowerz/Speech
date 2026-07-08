"""Phase 4c Task 3: `speech_rs.Engine` coarse corpus-level binding smoke tests.

`importorskip` guards the whole module: the plain `python-test` CI job never
builds `speech_rs`, so it SKIPS these; only the `python-pyo3` job (which runs
`maturin develop`) exercises them.

Each Engine construction resolves RELATIVE corpus/weight paths from the CWD, so
every test seeds a tempdir with a byte-copy of the committed fixtures and chdirs
into it -- mirroring `src/rust/tests/phase4a_tier1_e2e.rs::seed_corpus`.
"""

import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"
RUST_MANIFEST = REPO_ROOT / "src" / "rust" / "Cargo.toml"


@contextmanager
def chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _copy_corpus(dst: Path, files: tuple[str, ...]) -> None:
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")


def _seed_tier1(dst: Path) -> None:
    """Algo-1 (TDC) 3-file corpus + config, epochs overridden to 0 so `-m`
    dispatches a single solo eval (no training loop)."""
    _copy_corpus(dst, ("f1", "f2", "f3"))
    for f in ("fileslisting.csv", "language2classmapping.csv", "tier1_tdc.config"):
        shutil.copy(PHASE4A / f, dst / f)
    # Legacy parser is last-wins: appending overrides the committed `Epochs 2`.
    with (dst / "tier1_tdc.config").open("a") as fh:
        fh.write("\nNeural_Networks_BackPropagation_Epochs 0\n")
    (dst / "vrcts").mkdir(exist_ok=True)  # the config's Dump_Directory.


def _seed_tier2(dst: Path) -> None:
    """Algo-3 (spectral) config: construction loads NNweights_config1.bin
    (33,671 weights). The pairing test never run()s, so only the CSVs + weight
    pack must resolve; the wavs are copied for parity with the Rust seed."""
    _copy_corpus(dst, ("f1", "f2"))
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, dst / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", dst / "NNweights_config1.bin")


def test_import_and_version() -> None:
    v = speech_rs.version()
    assert isinstance(v, str)
    assert v
    assert hasattr(speech_rs, "Engine")


def test_engine_runs_tier1_tdc(tmp_path: Path) -> None:
    _seed_tier1(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["tier1_tdc.config"], "-m")
        eng.run()
        res = eng.results_matrix()
    assert isinstance(res, np.ndarray)
    assert res.dtype == np.float64
    # 3 files x 1 conf x 2 channels; ResultsE row = [file+1, conf+1, chan+1, res...].
    assert res.shape == (6, 21)
    assert set(res[:, 0].astype(int)) == {1, 2, 3}  # file ids
    assert np.all(res[:, 1] == 1.0)  # single config
    assert set(res[:, 2].astype(int)) == {1, 2}  # channels


def test_weights_pairing_contract(tmp_path: Path) -> None:
    # Algo 3: single-net -> a 1-element list of the 33,671-weight flat vector.
    tier2 = tmp_path / "tier2"
    tier2.mkdir()
    _seed_tier2(tier2)
    with chdir(tier2):
        eng = speech_rs.Engine(["tier2_spectral.config"], "-m")
        w = eng.weights(0)
    assert isinstance(w, list)
    assert len(w) == 1
    assert isinstance(w[0], np.ndarray)
    assert w[0].shape == (33671,)

    # Algo 1 (TDC): no NN -> empty list.
    tier1 = tmp_path / "tier1"
    tier1.mkdir()
    _seed_tier1(tier1)
    with chdir(tier1):
        eng = speech_rs.Engine(["tier1_tdc.config"], "-m")
        w = eng.weights(0)
    assert w == []


def test_error_chain_surfaces(tmp_path: Path) -> None:
    # A present config whose fileslisting points at a missing file: the anyhow
    # chain from Corpus::from_config must surface as a RuntimeError naming the path.
    (tmp_path / "mapping.csv").write_text("unk;unk;0\n")
    (tmp_path / "broken.config").write_text("Algo_choice 1\nlanguage2classmapping mapping.csv\nfileslisting does_not_exist_listing.csv\n")
    with chdir(tmp_path):  # noqa: SIM117
        with pytest.raises(RuntimeError, match="does_not_exist_listing.csv"):
            speech_rs.Engine(["broken.config"], "-m")


# === Phase 4c Task 5: TOML canonical config ==================================
# The Rust-side conversion table (`speech::toml_config`) lives only in Rust, so
# these tests drive the actual `speech --convert-config` CLI tool (via `cargo
# run`, which builds it if needed) to produce a real `.toml` fixture, then
# exercise the `speech_rs` seam (`load_toml_config` + `Engine(["*.toml"], ...)`)
# against it -- an end-to-end proof that the PyO3 seam's `.toml` acceptance
# (not just the pure Rust `toml_config` module) is wired correctly.


def _convert_config_to_toml(config_path: Path, out_path: Path) -> None:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(RUST_MANIFEST),
            "--bin",
            "speech",
            "--",
            "--convert-config",
            str(config_path),
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"--convert-config failed: stdout={result.stdout}\nstderr={result.stderr}"
    assert out_path.exists(), "--convert-config did not write the .toml output"


def test_load_toml_config_matches_legacy_parse(tmp_path: Path) -> None:
    config_path = PHASE4A / "tier1_tdc.config"
    toml_path = tmp_path / "tier1_tdc.toml"
    _convert_config_to_toml(config_path, toml_path)

    got = speech_rs.load_toml_config(str(toml_path))
    want = speech_rs.parse_legacy_config(config_path.read_text())
    assert got == want


def test_engine_runs_tier1_tdc_toml(tmp_path: Path) -> None:
    # Mirrors test_engine_runs_tier1_tdc, but the config is converted to TOML
    # first and Engine() is constructed from the .toml path -- exercising the
    # `.toml` extension-dispatch branch inside `Engine::new` (speech-py/src/lib.rs).
    _seed_tier1(tmp_path)
    toml_path = tmp_path / "tier1_tdc.toml"
    _convert_config_to_toml(tmp_path / "tier1_tdc.config", toml_path)

    with chdir(tmp_path):
        eng_config = speech_rs.Engine(["tier1_tdc.config"], "-m")
        eng_config.run()
        res_config = eng_config.results_matrix()

        eng_toml = speech_rs.Engine(["tier1_tdc.toml"], "-m")
        eng_toml.run()
        res_toml = eng_toml.results_matrix()

    assert res_config.shape == res_toml.shape == (6, 21)
    # Column 6 is the wall-clock timing column (masked in every Rust-side golden
    # comparison too, e.g. phase4a_tier1_e2e.rs); every other column is a pure
    # function of the (equivalent) config content and must match bit-for-bit.
    mask = np.ones(res_config.shape[1], dtype=bool)
    mask[6] = False
    np.testing.assert_array_equal(res_config[:, mask], res_toml[:, mask])
