"""The checkpoint directory as a value (issue #82).

CONTEXT.md: a checkpoint is a directory of packs plus a history. Two writers, two naming
schemes: the legacy `train` driver writes `<net>_weights.bin`, the Modern loop writes
`best_<net>.bin` / `last_<net>.bin` and `train_history.json`. This module is the one place
those names exist; `packs()` is the resolver the `test` subcommand and the Launcher score
through (issue #29: a scheme resolves only when it holds EVERY net, never a mix).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from speech.weight_bridge import read_bin, write_bin

F64 = np.float64
HISTORY = "train_history.json"


def net_names(algo: int) -> list[str]:
    """The per-net checkpoint suffixes: `[sad]` for the single-net algos, `[sad, lid]` for
    the algo-6 Twin (`BackPropagation.m:11-13`'s cell contract, same count as `_tail_lengths`)."""
    return ["sad", "lid"] if algo == 6 else ["sad"]


class EpochRecord(BaseModel):
    """One epoch's recorded metrics in `train_history.json`. `confusion_error` is the
    FIXED (F5) confusion misclassification rate on the validation set (algo 6 only; `None`
    for the single-net SAD algos, which carry no LID confusion)."""

    model_config = ConfigDict(extra="forbid")

    epoch: int
    train_cost: float
    val_cost: float
    confusion_error: float | None
    is_best: bool


class ModernTrainResult(BaseModel):
    """`train_modern`'s outcome + the checkpoint location (also serialized to
    `train_history.json` for resume). Distinct from `TrainResult` (the QPSO-shaped legacy
    outcome): the modern loop has no genome, so `gbest`/`gbestval` would be meaningless --
    this carries the best-epoch selection, the per-epoch history, and the early-stop flag."""

    model_config = ConfigDict(extra="forbid")

    best_epoch: int
    best_val_cost: float
    epochs_run: int
    stopped_early: bool
    history: list[EpochRecord]
    checkpoint_dir: str


@dataclass(frozen=True)
class Checkpoint:
    """`dir` holds one pack per net in `nets` order under a tag (`best`, `last`), plus the
    Modern loop's history. Construct it from the run (`ModernRun.checkpoint`), never from
    a rebuilt path."""

    dir: Path
    nets: list[str]

    def pack(self, tag: str, net: str) -> Path:
        return self.dir / f"{tag}_{net}.bin"

    def best(self) -> list[Path]:
        return [self.pack("best", n) for n in self.nets]

    def last(self) -> list[Path]:
        return [self.pack("last", n) for n in self.nets]

    def save(self, tag: str, weights: list[NDArray[np.float64]]) -> None:
        """Write one `<tag>_<net>.bin` per net (the column-major `.bin` codec, same as the
        legacy `train`'s `sad_weights.bin`)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, w in zip(self.nets, weights, strict=True):
            arr = np.asarray(w, dtype=F64)
            write_bin(arr.shape[0], 1, arr, self.pack(tag, name))

    def history(self) -> ModernTrainResult:
        return ModernTrainResult.model_validate_json((self.dir / HISTORY).read_text())

    def write_history(self, res: ModernTrainResult) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / HISTORY).write_text(res.model_dump_json(indent=2))

    def resume(self) -> tuple[list[NDArray[np.float64]], list[EpochRecord], float, int, int]:
        """Reload a prior `train_modern`'s `last_<net>.bin` weights + the history bookkeeping.
        Returns `(weights, history, best_val, best_epoch, start_epoch)`; `start_epoch` is the
        number of epochs already run, so the loop continues toward `params.epochs` (the TOTAL)."""
        prev = self.history()
        weights = [np.asarray(read_bin(p)[2], dtype=F64) for p in self.last()]
        return weights, list(prev.history), prev.best_val_cost, prev.best_epoch, prev.epochs_run

    def packs(self) -> list[Path]:
        """The scoring packs, one per net: a complete `<net>_weights.bin` set (the legacy
        `train`) or a complete `best_<net>.bin` set (`train_modern`; its `last_<net>.bin` is a
        resume point, not a scoring checkpoint). The legacy scheme wins when both are
        complete; a `[sad, lid]` pair is never assembled from two schemes. Raises
        `FileNotFoundError` naming both accepted sets when neither is complete (#29: the old
        `all(exists)` guard skipped `set_weights` and scored the config's seed pack in silence)."""
        schemes = [[self.dir / f"{net}_weights.bin" for net in self.nets], self.best()]
        for packs in schemes:
            if all(p.exists() for p in packs):
                return packs
        expected = " or ".join(" + ".join(p.name for p in packs) for packs in schemes)
        raise FileNotFoundError(f"checkpoint {self.dir} has no complete weight-pack set: expected {expected}")
