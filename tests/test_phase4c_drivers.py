"""Phase 4c Task 12: lifecycle-driver unit tests (Octave-free, no engine).

The engine-in-the-loop exit gate lives in `tests/pyo3/test_exit_gate.py` (pyo3 +
slow marked). These pin the pure driver pieces: `RunState` construction/round-trip
(the typed PS replacement), the genome sizing + `MaskingValidation` gate on the twin
`RunConfig`, the `.scr` softmax law (`Test_BLSTM.m:252-266`), the retrain `nnet_best`
seeding, and the `cli.main` dispatch.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from speech.cli import main as cli_main
from speech.drivers.init import init_run
from speech.drivers.retrain import seed_from_checkpoint
from speech.drivers.state import RunConfig, RunState
from speech.drivers.test import write_scores
from speech.genome import genome_length
from speech.scoring import masking_validation
from speech.weight_bridge import write_bin

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"


def _seed_init_inputs(dst: Path) -> Path:
    """The minimal files `init_run` reads: the twin config + its listing + mapping
    (no audio/phSeq/stm -- init does not touch the corpus payload)."""
    for f in ("twin_train.config", "languagemapping_lid7.csv"):
        (dst / f).write_bytes((PHASE4B / f).read_bytes())
    phseq = dst / "corpus_phseq"
    phseq.mkdir(parents=True, exist_ok=True)
    (phseq / "listing_train.csv").write_bytes((PHASE4B / "corpus_phseq" / "listing_train.csv").read_bytes())
    return dst / "twin_train.config"


def test_run_state_json_roundtrip(tmp_path: Path) -> None:
    """`init_run` parses the twin config into a typed `RunState`, and the JSON save
    (the documented deviation from ParamStruct.mat) round-trips bit-for-bit."""
    config = _seed_init_inputs(tmp_path)
    out_dir = tmp_path / "run"
    state = init_run(config, out_dir)

    assert isinstance(state.ps, RunConfig)
    assert state.ps.algo == 6
    assert state.balance == 10
    assert len(state.listing) == 2, "the twin listing has 2 files"
    assert state.base_config["Algo_choice"] == "6"
    state_file = out_dir / "run_state.json"
    assert state_file.exists(), "init_run persists the state as JSON"

    reloaded = RunState.load(state_file)
    assert reloaded == state, "RunState JSON save/load must round-trip"


def test_twin_genome_length_and_masking(tmp_path: Path) -> None:
    """The twin `RunConfig` sizes a 1698-coeff genome and a full-DSP-free mask
    round-trips cleanly through `MaskingValidation` (empty = pass)."""
    config = _seed_init_inputs(tmp_path)
    state = init_run(config, tmp_path / "run")
    d = genome_length(state.ps)
    assert d == 1698, "twin genome length (pinned so a miscount is caught)"

    rng = np.random.default_rng(7)
    genome = rng.uniform(0.0, state.ps.adim, size=d)
    assert masking_validation(genome, None, state.ps) == [], "empty mask must pass the gate on the twin ps"


def test_scr_format_hand_computed() -> None:
    """`write_scores` reproduces `Test_BLSTM.m:252-266`: softmax `exp(s/100)/sum`,
    sort DESC, `filename lang-dial %15.15f` lines. Golden computed independently."""
    scores_row = np.array([0.0, 100.0], dtype=np.float64)
    keys = ["engeng", "vievie"]
    text = write_scores(scores_row, keys, "file1")

    denom = 1.0 + math.exp(1.0)
    hi = math.exp(1.0) / denom  # class 1 (vievie), higher score -> first
    lo = 1.0 / denom  # class 0 (engeng)
    expected = f"file1 vie-vie {hi:.15f}\nfile1 eng-eng {lo:.15f}\n"
    assert text == expected


def test_scr_sum_floor() -> None:
    """The `max(1e-3, sum)` denominator floor (`Test_BLSTM.m:253`) kicks in when the
    softmax sum underflows below 1e-3 -- here two hugely-negative scores."""
    scores_row = np.array([-100000.0, -100000.0], dtype=np.float64)
    keys = ["engeng", "vievie"]
    text = write_scores(scores_row, keys, "f")
    # exp(-1000) underflows to 0; sum 0 -> floored to 1e-3; each val = 0/1e-3 = 0.
    for line in text.strip().split("\n"):
        assert line.endswith(f"{0.0:.15f}"), line


def test_retrain_seed_from_checkpoint(tmp_path: Path) -> None:
    """`seed_from_checkpoint` loads a saved gbest genome (`ReTrain_BLSTM.m`'s
    `nnet_best` seeding) as the QPSO seed vector, bit-for-bit."""
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    gbest = np.array([((k * 13 + 1) % 91) / 91.0 for k in range(1698)], dtype=np.float64)
    write_bin(gbest.shape[0], 1, gbest, ckpt / "gbest.bin")

    got = seed_from_checkpoint(ckpt)
    assert got.shape == gbest.shape
    assert np.array_equal(got.view(np.uint64), gbest.view(np.uint64)), "seed must be the saved gbest bit-for-bit"


def test_cli_init_dispatch(tmp_path: Path) -> None:
    """`cli.main(["init", config, out_dir])` runs `init_run` and persists the state;
    an unknown subcommand returns nonzero without raising."""
    config = _seed_init_inputs(tmp_path)
    out_dir = tmp_path / "run"
    rc = cli_main(["init", str(config), str(out_dir)])
    assert rc == 0
    assert (out_dir / "run_state.json").exists()

    assert cli_main(["bogus-command"]) != 0
    assert cli_main([]) != 0
