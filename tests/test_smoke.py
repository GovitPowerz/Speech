"""Smoke tests: the package and its modules import and expose a version."""

import speech


def test_package_imports() -> None:
    assert speech.__version__ == "0.1.0"
