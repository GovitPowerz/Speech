"""Assert the full module tree imports (architecture is wired)."""

import importlib

import pytest

MODULES = [
    "speech.cli",
    "speech.config_bridge",
    "speech.weight_bridge",
    "speech.engine",
    "speech.fold_run",
    "speech.optimizers",
    "speech.scoring",
    "speech.batching",
    "speech.nn_reference",
    "speech.drivers.init",
    "speech.drivers.train",
    "speech.drivers.retrain",
    "speech.drivers.test",
    "speech.dataprep.augment",
    "speech.dataprep.opensad15",
    "speech.dataprep.stm_normalize",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None
