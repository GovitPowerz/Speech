"""Phase 4c Task 12: lifecycle-driver unit tests (Octave-free, no engine).

The engine-in-the-loop exit gate lives in `tests/pyo3/test_exit_gate.py` (pyo3 +
slow marked). These pin the pure driver pieces: `RunState` construction/round-trip
(the typed PS replacement), the genome sizing + `MaskingValidation` gate on the twin
`RunConfig`, the `.scr` softmax law (`Test_BLSTM.m:252-266`), the retrain `nnet_best`
seeding, the `cli.main` dispatch, and (#29) `evaluate`'s checkpoint pack resolution.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from speech.cli import main as cli_main
from speech.drivers.init import init_run
from speech.drivers.retrain import seed_from_checkpoint
from speech.drivers.state import RunConfig, RunState
from speech.drivers.test import evaluate, resolve_checkpoint_packs, write_scores
from speech.genome import genome_length
from speech.scoring import masking_validation
from speech.weight_bridge import read_weight_vector, write_bin

from tests._result_rows import channel_results_from_matrix

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


# --------------------------------------------------------------------------------------- #
# #29: evaluate must not fall through to the config's seed pack
# --------------------------------------------------------------------------------------- #


def test_resolve_checkpoint_packs_accepts_both_naming_schemes(tmp_path: Path) -> None:
    """`train` writes `<net>_weights.bin`; `train_modern` writes `best_<net>.bin`. Either
    complete scheme resolves, and the legacy scheme wins when both are complete."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "sad_weights.bin").write_bytes(b"")
    (legacy / "lid_weights.bin").write_bytes(b"")
    assert resolve_checkpoint_packs(legacy, 6) == [legacy / "sad_weights.bin", legacy / "lid_weights.bin"]

    modern = tmp_path / "modern"
    modern.mkdir()
    (modern / "best_sad.bin").write_bytes(b"")
    (modern / "best_lid.bin").write_bytes(b"")
    (modern / "last_sad.bin").write_bytes(b"")
    assert resolve_checkpoint_packs(modern, 6) == [modern / "best_sad.bin", modern / "best_lid.bin"]
    assert resolve_checkpoint_packs(modern, 3) == [modern / "best_sad.bin"]

    both = tmp_path / "both"
    both.mkdir()
    (both / "sad_weights.bin").write_bytes(b"")
    (both / "best_sad.bin").write_bytes(b"")
    assert resolve_checkpoint_packs(both, 3) == [both / "sad_weights.bin"]


def test_resolve_checkpoint_packs_never_mixes_schemes(tmp_path: Path) -> None:
    """A legacy SAD pack beside a `train_modern` LID pack is two runs, not one checkpoint."""
    (tmp_path / "sad_weights.bin").write_bytes(b"")
    (tmp_path / "best_lid.bin").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="no complete weight-pack set"):
        resolve_checkpoint_packs(tmp_path, 6)


def test_resolve_checkpoint_packs_missing_net_raises(tmp_path: Path) -> None:
    """A checkpoint with the SAD pack but no LID pack (algo 6) names the missing net and
    both accepted filenames; `last_<net>.bin` alone is not a scoring checkpoint."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "sad_weights.bin").write_bytes(b"")
    (ckpt / "last_lid.bin").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"lid_weights\.bin.*best_lid\.bin") as exc:
        resolve_checkpoint_packs(ckpt, 6)
    assert str(ckpt) in str(exc.value)


def test_cli_test_missing_pack_fails_loudly(tmp_path: Path) -> None:
    """The #29 regression at the CLI: `test` on a checkpoint directory that lacks its packs
    used to skip `set_weights` and score the config's seed pack silently. The resolution is
    the subcommand's, before `evaluate` (which takes packs) touches anything on disk."""
    config = _seed_init_inputs(tmp_path)
    out_dir = tmp_path / "run"
    init_run(config, out_dir)
    empty_ckpt = tmp_path / "ckpt"
    empty_ckpt.mkdir()

    with pytest.raises(FileNotFoundError, match=r"sad_weights\.bin \+ lid_weights\.bin or best_sad\.bin \+ best_lid\.bin"):
        cli_main(["test", str(out_dir), str(empty_ckpt)])
    assert not (out_dir / "scores").exists(), "must raise before the scores dir is created"


def _fake_speech_rs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for the pyo3 module: record the map `evaluate`'s fold run hands `from_map`
    and the weights handed to `set_weights`; score no files."""
    seen: dict[str, Any] = {}

    class Engine:
        @staticmethod
        def from_map(configs: list[dict[str, str]], mode: str) -> Engine:
            seen["config"] = dict(configs[0])
            # Like a fast processor, read the weight packs AT CONSTRUCTION (T6b).
            present = [k for k in ("BLSTM_weightsFile", "BLSTM_LID_weightsFile") if Path(configs[0][k]).is_file()]
            seen["loaded"] = {k: list(read_weight_vector(Path(configs[0][k]))) for k in present}
            return Engine()

        def set_weights(self, conf: int, nets: list[list[float]]) -> None:
            seen["weights"] = [list(n) for n in nets]

        def run(self) -> None:
            pass

        def channel_results(self) -> dict[str, np.ndarray]:
            return asdict(channel_results_from_matrix(np.zeros((0, 0))))

        def weights(self, conf: int) -> list[np.ndarray]:
            return []

    monkeypatch.setitem(sys.modules, "speech_rs", types.SimpleNamespace(Engine=Engine))
    return seen


def _twin_state_and_ckpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RunState, Path]:
    """A twin run state whose config dir differs from the cwd, plus a `best_*` checkpoint
    passed RELATIVE to that cwd (the CLI shape)."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    state = init_run(_seed_init_inputs(cfg_dir), tmp_path / "run")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    write_bin(2, 1, np.array([1.0, 2.0]), ckpt / "best_sad.bin")
    write_bin(1, 1, np.array([3.0]), ckpt / "best_lid.bin")
    monkeypatch.chdir(tmp_path)
    return state, Path("ckpt")


def test_evaluate_exact_injects_resolved_packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact path: the resolved packs, read against the caller's cwd (not the config dir the
    fold resolves its paths against), go through `set_weights`; the map is inference-shaped
    and its paths are the config dir's, not the cwd's."""
    seen = _fake_speech_rs(monkeypatch)
    state, ckpt = _twin_state_and_ckpt(tmp_path, monkeypatch)

    evaluate(state, resolve_checkpoint_packs(ckpt, state.algo))

    assert seen["weights"] == [[1.0, 2.0], [3.0]]
    assert seen["config"]["Neural_Networks_BackPropagation_Epochs"] == "0"
    assert seen["config"]["BLSTM_BackPropagationActivated"] == "false"
    assert seen["config"]["BLSTM_LID_BackPropagationActivated"] == "false"
    assert seen["config"]["fileslisting"] == str(tmp_path / "cfg" / "corpus_phseq" / "listing_train.csv")


def test_evaluate_fast_injects_through_workdir_packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast path: `set_weights` bails there (T6b), so the checkpoint packs are injected by
    writing them under the config dir and repointing the config's weight keys at them for
    the engine's construction -- not ignored."""
    seen = _fake_speech_rs(monkeypatch)
    state, ckpt = _twin_state_and_ckpt(tmp_path, monkeypatch)
    state.base_config["Inference_Path"] = "fast"

    evaluate(state, resolve_checkpoint_packs(ckpt, state.algo))

    assert "weights" not in seen
    sad, lid = Path(seen["config"]["BLSTM_weightsFile"]), Path(seen["config"]["BLSTM_LID_weightsFile"])
    assert sad.parent == lid.parent == tmp_path / "cfg"
    assert seen["loaded"] == {"BLSTM_weightsFile": [1.0, 2.0], "BLSTM_LID_weightsFile": [3.0]}
    assert not sad.exists() and not lid.exists(), "the packs are removed once the engine holds them"
