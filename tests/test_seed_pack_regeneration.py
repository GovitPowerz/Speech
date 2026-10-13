"""Issue #27: every committed Phase 9 seed pack regenerates from `init_weights`, BYTE-IDENTICAL on the oracle env.

Manifest-driven: `tests/reference_data/phase9/manifest.json` records, per fixture, the config
text, the init prefix (`BLSTM` for a SAD net, `BLSTM_LID` for the Twin's LID net), the RNG seed,
the scheme and the forget-bias rule the pack was drawn with. The test replays exactly that and
compares the bytes to the committed `.bin`, so the RNG CONSUMPTION ORDER per block is pinned
through any rewrite of the builders (the issue's constraint: same-seed packs and the run-twice
byte-identity gates must reproduce bit-for-bit).

Byte identity holds on the environment the packs were drawn on (macOS arm64, the oracle env);
elsewhere the comparison is `tests/_libm_gate.py`'s canary-gated hybrid bound. Measured: the
arm64 numpy wheel computes `Generator.uniform` as ONE fused `fma(high - low, u, low)` (100000 of
100000 draws match the fused form, 48% match the unfused `low + (high - low) * u`), so an x86_64
host lands 1 ULP off on about half the elements; the Mamba `exp`/`log` draws add libm rounding.
A changed consumption order or draw count moves values by O(limit), far outside the bound.

Written first and green before the migration; it must stay green through every step. The
Phase 4a/4b seed packs have no recorded provenance and stay length-pinned by their own tests.
"""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from speech.config_bridge import nnet_spec, parse_legacy_config
from speech.init_weights import init_weights
from speech.weight_bridge import read_bin

from tests._libm_gate import assert_f64_matrix_close

PHASE9 = Path("tests/reference_data/phase9")
MANIFEST = cast(dict[str, Any], json.loads((PHASE9 / "manifest.json").read_text()))
FIXTURES = cast(dict[str, dict[str, Any]], MANIFEST["fixtures"])


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_committed_seed_pack_regenerates_byte_identical(name: str) -> None:
    info = FIXTURES[name]
    flat = parse_legacy_config((PHASE9 / info["config"]).read_text())
    rng = np.random.default_rng(info["seed"])
    packs = init_weights(nnet_spec(flat, info["prefix"]), rng, info["init_scheme"], info["forget_bias_one"])
    assert len(packs) == 1

    rows, cols, committed = read_bin(PHASE9 / info["weights"])
    assert (rows, cols) == (len(committed), 1)
    assert packs[0].dtype == np.float64
    assert packs[0].shape == committed.shape, f"{name}: regenerated {packs[0].shape[0]} elements, the committed pack holds {committed.shape[0]}"
    assert_f64_matrix_close(packs[0], committed, f"{name}: the regenerated pack vs the committed bytes")


def test_every_fixture_records_its_provenance() -> None:
    """The regeneration leg is only as complete as the manifest: every fixture carries the
    five fields the replay needs, and the prefix names the net the pack seeds."""
    for name, info in FIXTURES.items():
        for key in ("config", "prefix", "seed", "init_scheme", "forget_bias_one", "weights"):
            assert key in info, f"{name}: manifest lacks {key!r}"
        assert info["prefix"] == ("BLSTM_LID" if info["nets"] == 2 else "BLSTM"), name
