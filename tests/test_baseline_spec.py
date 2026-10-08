"""Issue #21: the launcher spec. One typed declaration of a baseline run (`drivers/spec.py`,
seam-free): the fields and their one set of defaults, the validators that refuse an invalid
run before any directory exists, and the projection to the ledger recipe. Engine-free."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from speech.drivers.spec import ARM_CONFIG, ARM_LINEAGE, BaselineSpec
from speech.ledger.schema import BaselineRecipe, lineage_of


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


def test_every_arm_declares_the_lineage_its_config_name_carries() -> None:
    assert set(ARM_CONFIG) == set(ARM_LINEAGE) == {"sad", "sad-v2", "lid-features", "lid-phseq"}
    for arm, config in ARM_CONFIG.items():
        assert ARM_LINEAGE[arm] == lineage_of(config), arm
        assert (Path(__file__).resolve().parents[1] / config).is_file(), config
