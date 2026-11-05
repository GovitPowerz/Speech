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
from types import UnionType
from typing import Annotated, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from speech.config_bridge import CellType, Direction
from speech.init_weights import Scheme
from speech.ledger.schema import Arm, BaselineRecipe, Lineage, lineage_of

#: The committed canonical TOML for each arm, relative to the repo root.
ARM_CONFIG: dict[Arm, str] = {
    "sad": "configs/training/lre_sad.toml",
    "sad-v2": "configs/training/lre_sad_v2.toml",
    "lid-features": "configs/training/lre03_lid_features.toml",
    "lid-phseq": "configs/training/lre03_lid_phseq.toml",
}
#: The SAD lineage each arm trains (ADR-0008); a LID arm has none. Read off the arm's config
#: name by the ledger's one lineage rule, `lineage_of` (issue #85): the parse is the source.
ARM_LINEAGE: dict[Arm, Lineage | None] = {arm: lineage_of(config) for arm, config in ARM_CONFIG.items()}
SAD_ARMS: frozenset[str] = frozenset(arm for arm, lineage in ARM_LINEAGE.items() if lineage is not None)
LID_ARMS: frozenset[str] = frozenset(ARM_LINEAGE) - SAD_ARMS
#: The recipe fields `recipe()` derives rather than reads off a spec field of the same name.
RECIPE_DERIVED: frozenset[str] = frozenset({"lineage", "listing"})


def listing_hash(path: Path) -> str:
    """The recipe's `listing` (issue #38): the blake2b-8 of the source listing's bytes, the same
    digest family as the record id, so a localized 2015 listing is identified without its path."""
    return hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest()


class BaselineSpec(BaseModel):
    """One launcher run. Every field is a CLI flag of the same name (`-` for `_`); the recipe
    fields (ADR-0009) are the measurement's identity, the rest the run site."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: Arm = Field(description="the training arm")
    corpus_root: Path = Field(description="the LRE03/07 corpus root (data/LRE03-LRE07)")
    out_dir: Path = Field(description="run directory for listings/config/checkpoints")
    seed: NonNegativeInt = 0
    lanes: PositiveInt = Field(default=1, description="engine numOuterThreads fold width (N=1 is the parity mode)")
    subset: PositiveInt | None = Field(default=None, description="stratified subset size (omit for the full corpus)")
    epochs: PositiveInt = 40
    patience: PositiveInt = 6
    steps_per_epoch: PositiveInt = 8
    init_scheme: Scheme = "xavier"
    # #98: a held-out size of 0 is no split on any arm
    valid_size: NonNegativeInt = Field(
        default=12,
        description="held-out validation slice (files; LID allocates per language, min 1 each for a positive size); 0 validates on the train listing",
    )
    test_size: NonNegativeInt = Field(
        default=48, description="held-out test slice (files; LID: min 1 per language for a positive size; 0 scores nothing) the final metric is scored on"
    )
    minibatch: NonNegativeInt = Field(
        default=0, description="training files drawn per SMORMS3 step, class-stratified round-robin (0 = the whole train listing)"
    )
    audio_max_duration: PositiveFloat | None = Field(
        default=None, description="SAD arm: cap Audio_max_duration (s); the corpus wavs are 576-1800 s (median ~600 s)"
    )
    cell: CellType = Field(default="lstm", description="the recurrent cell of the net that trains (default = the peephole BLSTM)")
    direction: Direction = Field(default="bidirectional", description="forward drops the backward stack (and halves the output MLP's input width)")
    lre_listing: Path | None = Field(default=None, description="LID arm: localize this 2015 listing instead of deriving from the corpus tree")
    score_init: bool = Field(default=False, description="also score the from-scratch seed pack on the test slice (the init-baseline row)")
    resume: bool = Field(default=False, description="continue from out_dir/checkpoint")
    dry_run: bool = Field(default=False, description="1-step smoke: tiny subset, 1 epoch, 1 step, still scored")

    @model_validator(mode="after")
    def _arm_knobs(self) -> BaselineSpec:
        if self.lre_listing is not None and self.arm in SAD_ARMS:
            raise ValueError(f"lre_listing is a LID-arm knob; the {self.arm} arm derives its split from the corpus tree and would ignore it")
        if self.audio_max_duration is not None and self.arm in LID_ARMS:
            raise ValueError(f"audio_max_duration is a SAD-arm knob; the {self.arm} arm scores precomputed features, not audio")
        return self

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """The launcher's flags, generated from the fields (issue #85): `arm` is the positional,
        every other field is `--<name>` with `_` as `-`, its default the field's, its help the
        field's description, a `bool` a `store_true`, a `Literal` its choices, and a field with no
        default required. `from_args` is the inverse, so the CLI and a direct call train the same
        run by construction. Only the annotation kinds the spec uses are mapped; a new kind
        raises here rather than parse as a string."""
        for name, field in cls.model_fields.items():
            if name == "arm":
                parser.add_argument("arm", choices=sorted(ARM_CONFIG), help=field.description)
                continue
            flag = f"--{name.replace('_', '-')}"
            kind, choices = _flag_kind(field.annotation)
            if kind is bool:
                if field.default is not False:
                    raise TypeError(f"a bool spec field is a store_true flag, so its default is False: {name}={field.default!r}")
                parser.add_argument(flag, action="store_true", help=field.description)
            elif field.is_required():
                parser.add_argument(flag, type=kind, choices=choices, required=True, help=field.description)
            else:
                parser.add_argument(flag, type=kind, choices=choices, default=field.default, help=field.description)
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BaselineSpec:
        """The one builder behind both CLI entries: every parser dest is a field of the same name."""
        return cls(**vars(args))

    def recipe(self, *, listing: str | None) -> BaselineRecipe:
        """The run's ledger identity. The recipe model is the tag (issue #85): every
        `BaselineRecipe` field but the two derived here (`lineage` off the arm, `listing`) is
        read off the spec by name, so a knob is identity exactly when the ledger schema says
        so (ADR-0009) and a spec field the schema does not name is run-site by construction.
        `listing` is the hash of `lre_listing`'s bytes as stamped when the run started
        (`listing_hash`), never recomputed here: a listing edited during a run must not rename
        the record of the bytes that trained."""
        return BaselineRecipe(
            **{name: getattr(self, name) for name in BaselineRecipe.model_fields if name not in RECIPE_DERIVED},
            lineage=ARM_LINEAGE[self.arm],
            listing=listing,
        )


def _flag_kind(annotation: object) -> tuple[type, tuple[str, ...] | None]:
    """The argparse `type` and `choices` of a spec field's annotation: `Annotated[X, ...]` and
    `X | None` unwrap to `X`; a `Literal` of strings is `str` with its values as choices; `int`, `float`,
    `Path` and `bool` are themselves. Anything else is refused."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return _flag_kind(get_args(annotation)[0])
    if origin is UnionType:
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) != 1:
            raise TypeError(f"no CLI flag kind for a spec field annotated {annotation!r}: one member besides None")
        return _flag_kind(members[0])
    if origin is Literal:
        values = get_args(annotation)
        if not all(isinstance(v, str) for v in values):
            raise TypeError(f"no CLI flag kind for a spec field annotated {annotation!r}: a Literal flag parses as str, so its values must be")
        return str, values
    if isinstance(annotation, type) and annotation in (int, float, Path, bool):
        return annotation, None
    raise TypeError(f"no CLI flag kind for a spec field annotated {annotation!r}")
