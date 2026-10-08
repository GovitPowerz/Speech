"""The baseline spec (issue #21): ONE typed declaration of a launcher run.

`BaselineSpec` is what `python -m speech.drivers.baseline` parses into and what
`run_baseline` takes. Its defaults are the launcher's one set (the parser reads them off the
fields, so a flag and a direct call train the same run); its validators refuse an invalid run
at construction, before any directory exists; `recipe()` projects the measurement's identity
into the ledger's `BaselineRecipe` (ADR-0009). Frozen: a derived run (the dry-run clamp) is a
`model_copy`, never a mutation.

Seam-free by design (no `speech_rs`, no engine import): the plain-Python CI job and the ledger
validate specs without the corpus or the PyO3 build. The (cell x direction) vocabulary is
`config_bridge`'s, declared once.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from speech.config_bridge import CellType, Direction
from speech.ledger.schema import Arm, BaselineRecipe, Lineage

#: The committed canonical TOML for each arm, relative to the repo root.
ARM_CONFIG: dict[Arm, str] = {
    "sad": "configs/training/lre_sad.toml",
    "sad-v2": "configs/training/lre_sad_v2.toml",
    "lid-features": "configs/training/lre03_lid_features.toml",
    "lid-phseq": "configs/training/lre03_lid_phseq.toml",
}
#: The SAD lineage each arm trains (ADR-0008); a LID arm has none. Declared, and pinned against
#: `lineage_of(ARM_CONFIG[arm])` by a test, so the parse is a check rather than the source.
ARM_LINEAGE: dict[Arm, Lineage | None] = {"sad": "v1", "sad-v2": "v2", "lid-features": None, "lid-phseq": None}
SAD_ARMS: frozenset[str] = frozenset(arm for arm, lineage in ARM_LINEAGE.items() if lineage is not None)
LID_ARMS: frozenset[str] = frozenset(ARM_LINEAGE) - SAD_ARMS


def listing_hash(path: Path) -> str:
    """The recipe's `listing` (issue #38): the blake2b-8 of the source listing's bytes, the same
    digest family as the record id, so a localized 2015 listing is identified without its path."""
    return hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest()


class BaselineSpec(BaseModel):
    """One launcher run. Every field is a CLI flag of the same name (`-` for `_`); the recipe
    fields (ADR-0009) are the measurement's identity, the rest the run site."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: Arm
    corpus_root: Path
    out_dir: Path
    seed: int = 0
    lanes: PositiveInt = 1
    subset: PositiveInt | None = None
    epochs: PositiveInt = 40
    patience: PositiveInt = 6
    steps_per_epoch: PositiveInt = 8
    init_scheme: Literal["xavier", "he"] = "xavier"
    valid_size: NonNegativeInt = 12
    test_size: NonNegativeInt = 48
    minibatch: NonNegativeInt = 0
    audio_max_duration: PositiveFloat | None = None
    cell: CellType = "lstm"
    direction: Direction = "bidirectional"
    lre_listing: Path | None = None
    score_init: bool = False
    resume: bool = False
    dry_run: bool = False

    @model_validator(mode="after")
    def _arm_knobs(self) -> BaselineSpec:
        if self.lre_listing is not None and self.arm in SAD_ARMS:
            raise ValueError(f"lre_listing is a LID-arm knob; the {self.arm} arm derives its split from the corpus tree and would ignore it")
        if self.audio_max_duration is not None and self.arm in LID_ARMS:
            raise ValueError(f"audio_max_duration is a SAD-arm knob; the {self.arm} arm scores precomputed features, not audio")
        return self

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BaselineSpec:
        """The one builder behind both CLI entries: every parser dest is a field of the same name."""
        return cls(**vars(args))

    def recipe(self, *, listing: str | None) -> BaselineRecipe:
        """The run's ledger identity. `listing` is the hash of `lre_listing`'s bytes as stamped
        when the run started (`listing_hash`), never recomputed here: a listing edited during
        a run must not rename the record of the bytes that trained."""
        if (listing is None) != (self.lre_listing is None):
            raise ValueError(f"listing hash {listing!r} disagrees with lre_listing {self.lre_listing!r}: a localized listing has a hash, a derived split none")
        return BaselineRecipe(
            arm=self.arm,
            lineage=ARM_LINEAGE[self.arm],
            cell=self.cell,
            direction=self.direction,
            subset=self.subset,
            valid_size=self.valid_size,
            test_size=self.test_size,
            audio_max_duration=self.audio_max_duration,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            patience=self.patience,
            minibatch=self.minibatch,
            init_scheme=self.init_scheme,
            seed=self.seed,
            lanes=self.lanes,
            listing=listing,
        )
