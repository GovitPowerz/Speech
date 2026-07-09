"""Shared pytest fixtures for the speech test suite."""

import os

from hypothesis import settings

# Derandomized profile for CI reproducibility (a flaky property-test failure with no
# replayable seed is useless in CI logs). Local runs default to "dev" (randomized,
# fresh examples every run); set HYPOTHESIS_PROFILE=ci to match CI locally.
settings.register_profile("ci", derandomize=True)
settings.register_profile("dev", derandomize=False)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
