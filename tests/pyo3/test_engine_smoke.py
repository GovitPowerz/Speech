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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"


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
