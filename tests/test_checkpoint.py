"""Issue #82: the checkpoint directory as a value, `drivers/checkpoint.py::Checkpoint`.

One module owns the two naming schemes (`<net>_weights.bin` from the legacy `train`,
`best_<net>.bin` / `last_<net>.bin` from `train_modern`) and the Modern loop's
`train_history.json`; every consumer used to rebuild the file names as literals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from speech.drivers.checkpoint import Checkpoint, EpochRecord, ModernTrainResult, net_names
from speech.weight_bridge import read_bin


def test_net_names_is_the_backprop_cell_contract() -> None:
    assert net_names(6) == ["sad", "lid"]
    assert net_names(3) == ["sad"]
    assert net_names(5) == ["sad"]


def test_best_and_last_name_one_pack_per_net(tmp_path: Path) -> None:
    ckpt = Checkpoint(tmp_path, ["sad", "lid"])
    assert ckpt.best() == [tmp_path / "best_sad.bin", tmp_path / "best_lid.bin"]
    assert ckpt.last() == [tmp_path / "last_sad.bin", tmp_path / "last_lid.bin"]
    assert Checkpoint(tmp_path, ["sad"]).best() == [tmp_path / "best_sad.bin"]


def test_save_writes_the_bin_codec_under_the_tag(tmp_path: Path) -> None:
    ckpt = Checkpoint(tmp_path / "checkpoint", ["sad", "lid"])
    ckpt.save("best", [np.array([1.0, 2.0]), np.array([3.0])])
    assert [float(x) for x in read_bin(ckpt.dir / "best_sad.bin")[2]] == [1.0, 2.0]
    assert [float(x) for x in read_bin(ckpt.dir / "best_lid.bin")[2]] == [3.0]
    with pytest.raises(ValueError):
        ckpt.save("last", [np.array([1.0])])


def test_history_round_trips_through_train_history_json(tmp_path: Path) -> None:
    ckpt = Checkpoint(tmp_path, ["sad"])
    res = ModernTrainResult(
        best_epoch=1,
        best_val_cost=0.5,
        epochs_run=2,
        stopped_early=False,
        history=[
            EpochRecord(epoch=0, train_cost=1.0, val_cost=0.9, lid_error=None, is_best=True),
            EpochRecord(epoch=1, train_cost=0.8, val_cost=0.5, lid_error=None, is_best=True),
        ],
        checkpoint_dir=str(tmp_path),
    )
    ckpt.write_history(res)
    assert (tmp_path / "train_history.json").is_file()
    assert ckpt.history() == res


def test_resume_reloads_last_packs_and_continues_at_epochs_run(tmp_path: Path) -> None:
    ckpt = Checkpoint(tmp_path, ["sad"])
    ckpt.save("last", [np.array([4.0])])
    ckpt.save("best", [np.array([2.0])])
    hist = [EpochRecord(epoch=e, train_cost=float(e), val_cost=float(3 - e), lid_error=None, is_best=True) for e in range(3)]
    ckpt.write_history(ModernTrainResult(best_epoch=2, best_val_cost=1.0, epochs_run=3, stopped_early=False, history=hist, checkpoint_dir=str(tmp_path)))
    weights, history, best_val, best_epoch, start_epoch = ckpt.resume()
    assert [float(x) for x in weights[0]] == [4.0]
    assert history == hist and best_val == 1.0 and best_epoch == 2
    assert start_epoch == 3


# --------------------------------------------------------------------------------------- #
# #29: evaluate must not fall through to the config's seed pack (moved from test_phase4c_drivers)
# --------------------------------------------------------------------------------------- #


def test_packs_accepts_both_naming_schemes(tmp_path: Path) -> None:
    """`train` writes `<net>_weights.bin`; `train_modern` writes `best_<net>.bin`. Either
    complete scheme resolves, and the legacy scheme wins when both are complete."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "sad_weights.bin").write_bytes(b"")
    (legacy / "lid_weights.bin").write_bytes(b"")
    assert Checkpoint(legacy, ["sad", "lid"]).packs() == [legacy / "sad_weights.bin", legacy / "lid_weights.bin"]

    modern = tmp_path / "modern"
    modern.mkdir()
    (modern / "best_sad.bin").write_bytes(b"")
    (modern / "best_lid.bin").write_bytes(b"")
    (modern / "last_sad.bin").write_bytes(b"")
    assert Checkpoint(modern, ["sad", "lid"]).packs() == [modern / "best_sad.bin", modern / "best_lid.bin"]
    assert Checkpoint(modern, ["sad"]).packs() == [modern / "best_sad.bin"]

    both = tmp_path / "both"
    both.mkdir()
    (both / "sad_weights.bin").write_bytes(b"")
    (both / "best_sad.bin").write_bytes(b"")
    assert Checkpoint(both, ["sad"]).packs() == [both / "sad_weights.bin"]


def test_packs_never_mixes_schemes(tmp_path: Path) -> None:
    """A legacy SAD pack beside a `train_modern` LID pack is two runs, not one checkpoint."""
    (tmp_path / "sad_weights.bin").write_bytes(b"")
    (tmp_path / "best_lid.bin").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="no complete weight-pack set"):
        Checkpoint(tmp_path, ["sad", "lid"]).packs()


def test_packs_missing_net_raises(tmp_path: Path) -> None:
    """A checkpoint with the SAD pack but no LID pack (algo 6) names the missing net and
    both accepted filenames; `last_<net>.bin` alone is not a scoring checkpoint."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "sad_weights.bin").write_bytes(b"")
    (ckpt / "last_lid.bin").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"lid_weights\.bin.*best_lid\.bin") as exc:
        Checkpoint(ckpt, ["sad", "lid"]).packs()
    assert str(ckpt) in str(exc.value)
