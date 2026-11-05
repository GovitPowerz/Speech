"""Issue #21: the launcher spec. One typed declaration of a baseline run (`drivers/spec.py`,
seam-free): the fields and their one set of defaults, the validators that refuse an invalid
run before any directory exists, and the projection to the ledger recipe. Engine-free."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError
from speech.drivers.spec import ARM_CONFIG, ARM_LINEAGE, LID_ARMS, RECIPE_DERIVED, SAD_ARMS, BaselineSpec, _flag_kind
from speech.drivers.state import ModernTrainParams
from speech.ledger.schema import BaselineRecipe


def _spec(**over: object) -> BaselineSpec:
    return BaselineSpec(**{"arm": "sad", "corpus_root": Path("/c"), "out_dir": Path("/o"), **over})  # type: ignore[arg-type]


def test_defaults_are_the_one_set() -> None:
    s = _spec()
    assert (s.seed, s.lanes, s.subset) == (0, 1, None)
    assert (s.epochs, s.steps_per_epoch, s.patience) == (40, 8, 6)
    assert (s.valid_size, s.test_size, s.minibatch, s.init_scheme) == (12, 48, 0, "xavier")
    assert (s.cell, s.direction) == ("lstm", "bidirectional")
    assert s.audio_max_duration is None and s.lre_listing is None
    assert (s.score_init, s.resume, s.dry_run) == (False, False, False)


def test_spec_is_frozen_and_closed() -> None:
    s = _spec()
    with pytest.raises(ValidationError):
        s.epochs = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _spec(cell_type="lstm")
    assert s.model_copy(update={"epochs": 1}).epochs == 1 and s.epochs == 40


@pytest.mark.parametrize(
    "bad",
    [
        dict(seed=-1),
        dict(epochs=0),
        dict(steps_per_epoch=0),
        dict(patience=0),
        dict(lanes=0),
        dict(subset=0),
        dict(valid_size=-1),
        dict(test_size=-1),
        dict(minibatch=-1),
        dict(audio_max_duration=0.0),
        dict(cell="gru"),
        dict(direction="backward"),
        dict(arm="lid"),
        dict(init_scheme="orthogonal"),
    ],
    ids=lambda d: next(iter(d)),
)
def test_counts_and_vocabulary_are_validated(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _spec(**bad)


def test_arm_cross_checks(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="lre_listing"):
        _spec(arm="sad", lre_listing=tmp_path / "l.csv")
    with pytest.raises(ValidationError, match="audio_max_duration"):
        _spec(arm="lid-phseq", audio_max_duration=20.0)
    assert _spec(arm="lid-features", lre_listing=tmp_path / "l.csv").lre_listing == tmp_path / "l.csv"
    assert _spec(arm="sad-v2", audio_max_duration=20.0).audio_max_duration == 20.0


def test_the_recipe_model_is_the_tag() -> None:
    """Issue #85: a recipe field is a spec field of the same name unless `recipe()` derives it,
    so identity is declared once, in the ledger schema, and the projection cannot drift."""
    assert {"lineage", "listing"} == RECIPE_DERIVED
    projected = set(BaselineRecipe.model_fields) - RECIPE_DERIVED
    assert projected <= set(BaselineSpec.model_fields)
    s = _spec(arm="lid-features", cell="mamba", direction="forward", seed=5, lanes=3, subset=7, valid_size=2, test_size=4, minibatch=6)
    s = s.model_copy(update={"epochs": 11, "steps_per_epoch": 13, "patience": 17, "init_scheme": "he"})
    recipe = s.recipe(listing="fedcba9876543210").model_dump()
    assert {k: recipe[k] for k in projected} == {k: getattr(s, k) for k in projected}


def test_spec_named_train_params_are_required_and_take_the_spec_values() -> None:
    """Issue #85: `ModernTrainParams` is the Modern loop's input, not a spec copy, so a knob the
    spec declares under the same name carries no default there (the spec is the one default
    set) and the spec's value validates into it unchanged."""
    shared = set(ModernTrainParams.model_fields) & set(BaselineSpec.model_fields)
    assert shared == {"epochs", "patience", "steps_per_epoch", "minibatch", "init_scheme", "seed"}
    assert all(ModernTrainParams.model_fields[name].is_required() for name in shared)
    for name in set(ModernTrainParams.model_fields) - shared:
        assert not ModernTrainParams.model_fields[name].is_required(), name
    s = _spec(seed=5, minibatch=6).model_copy(update={"epochs": 11, "steps_per_epoch": 13, "patience": 17, "init_scheme": "he"})
    params = ModernTrainParams(**{name: getattr(s, name) for name in shared})
    assert {name: getattr(params, name) for name in shared} == {name: getattr(s, name) for name in shared}


@pytest.mark.parametrize("annotation", [str, int | str, Path | None | int, list[int], Literal[1, 2], Literal["a", 1]], ids=repr)
def test_the_flag_generator_refuses_an_annotation_kind_the_spec_does_not_use(annotation: object) -> None:
    with pytest.raises(TypeError, match="no CLI flag kind"):
        _flag_kind(annotation)


def test_the_flag_generator_unwraps_what_the_spec_uses() -> None:
    assert _flag_kind(BaselineSpec.model_fields["subset"].annotation) == (int, None)
    assert _flag_kind(BaselineSpec.model_fields["audio_max_duration"].annotation) == (float, None)
    assert _flag_kind(BaselineSpec.model_fields["lre_listing"].annotation) == (Path, None)
    assert _flag_kind(BaselineSpec.model_fields["cell"].annotation) == (str, ("lstm", "slstm", "mamba", "cfc", "transformer"))
    assert _flag_kind(Literal["a"] | None) == (str, ("a",))
    assert _flag_kind(BaselineSpec.model_fields["dry_run"].annotation) == (bool, None)


def test_the_flag_generator_refuses_a_true_bool_default() -> None:
    """A bool flag is a `store_true`, so a `True` default could not round-trip: refused at generation."""

    class Spec(BaselineSpec):
        verbose: bool = True

    with pytest.raises(TypeError, match="store_true"):
        Spec.add_arguments(argparse.ArgumentParser())


def test_recipe_projection() -> None:
    s = _spec(arm="sad-v2", cell="cfc", direction="forward", seed=3, subset=10, valid_size=8, test_size=24, audio_max_duration=20.0)
    s = s.model_copy(update={"epochs": 3, "steps_per_epoch": 10, "patience": 99})
    assert s.recipe(listing=None) == BaselineRecipe(
        arm="sad-v2",
        lineage="v2",
        cell="cfc",
        direction="forward",
        subset=10,
        valid_size=8,
        test_size=24,
        audio_max_duration=20.0,
        epochs=3,
        steps_per_epoch=10,
        patience=99,
        minibatch=0,
        init_scheme="xavier",
        seed=3,
        lanes=1,
        listing=None,
    )
    assert _spec(arm="sad").recipe(listing=None).lineage == "v1"
    localized = _spec(arm="lid-phseq", lre_listing=Path("/l/lre03_train.csv")).recipe(listing="0123456789abcdef")
    assert localized.lineage is None and localized.listing == "0123456789abcdef"
    assert _spec(arm="lid-features").recipe(listing=None).full and not _spec(arm="lid-features", subset=5).recipe(listing=None).full


def test_every_arm_names_a_committed_config_and_the_arm_sets_follow_the_lineage() -> None:
    assert set(ARM_CONFIG) == set(ARM_LINEAGE) == {"sad", "sad-v2", "lid-features", "lid-phseq"}
    for config in ARM_CONFIG.values():
        assert (Path(__file__).resolve().parents[1] / config).is_file(), config
    assert {"sad", "sad-v2"} == SAD_ARMS and {"lid-features", "lid-phseq"} == LID_ARMS


@pytest.mark.parametrize("arm", ["sad", "lid-phseq"])
def test_a_zero_held_out_size_is_one_contract_on_every_arm(arm: str) -> None:
    """Issue #98: `0` is accepted on both arm kinds and means no held-out split (the arm's
    floor applies to positive sizes only, pinned on the arms in `test_phase6_baseline_arms`)."""
    spec = BaselineSpec(arm=arm, corpus_root=Path("/c"), out_dir=Path("/o"), valid_size=0, test_size=0)  # type: ignore[arg-type]
    assert (spec.valid_size, spec.test_size) == (0, 0)
    assert (spec.recipe(listing=None).valid_size, spec.recipe(listing=None).test_size) == (0, 0)
