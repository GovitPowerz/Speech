"""Shared pytest fixtures for the speech test suite."""

import os
import pathlib

import pytest
from hypothesis import settings

# Licensed LRE03/07 corpus root (27 GB, gitignored, absent in CI). Phase-6 corpus-gated
# tests mark themselves `@requires_corpus` so they skip cleanly when it is missing.
CORPUS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "LRE03-LRE07"

requires_corpus = pytest.mark.skipif(
    not CORPUS_ROOT.is_dir(),
    reason=f"licensed LRE03/07 corpus absent at {CORPUS_ROOT}",
)

# Derandomized profile for CI reproducibility (a flaky property-test failure with no
# replayable seed is useless in CI logs). Local runs default to "dev" (randomized,
# fresh examples every run); set HYPOTHESIS_PROFILE=ci to match CI locally.
settings.register_profile("ci", derandomize=True)
settings.register_profile("dev", derandomize=False)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
