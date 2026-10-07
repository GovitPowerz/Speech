"""Phase 4d Task 9: the single-net (algo-3) driver generalization.

The 4c lifecycle drivers shipped ALGO-6-ONLY: `_backprop_inner` indexes `full[1]`,
`_ponderations` reads `BLSTM_LID_CostPonderation`, `_tail_lengths` returns a 2-tuple,
so a single-net algo-3 config KeyErrors/IndexErrors (the 4c-review Info finding). This
pins the generalized contract against the committed REAL algo-3 config
(`parity_tupleA.config`) + the algo-6 twin config as the unchanged-behaviour foil.

The governing legacy source is `BackPropagation.m:11-13`: `weightsIni = cell(2,1)` with
cell 1 present iff `PS.NS.BackPropagationActivated == 1` and cell 2 present iff
`(PS.VP.algo == 6) && (PS.NS.LID.BackPropagationActivated == 1)` -- i.e. a 1-cell `[sad]`
contract for algo 3, a 2-cell `[sad, lid]` contract for algo 6.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
from speech.config_bridge import parse_legacy_config
from speech.drivers import train as T
from speech.drivers.state import ps_from_config
from speech.genome import genome_length

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE4D = REPO_ROOT / "tests" / "reference_data" / "phase4d"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"

ALGO3_CONFIG = PHASE4D / "parity_tupleA.config"
ALGO6_CONFIG = PHASE4B / "twin_train.config"


def _cfg(path: Path) -> dict[str, str]:
    return parse_legacy_config(path.read_text())


# ---- ps_from_config: the lid=None arm for algo != 6 -------------------------------------


def test_ps_from_config_algo3_is_single_net() -> None:
    """algo 3 -> `lid is None`, balance 5 (the non-LID default); the LID sub-spec is only
    built for algo 6. This arm already exists in state.py -- pinned here as the contract."""
    ps = ps_from_config(_cfg(ALGO3_CONFIG))
    assert ps.algo == 3
    assert ps.lid is None
    assert ps.balance == 5


def test_ps_from_config_algo6_still_builds_lid() -> None:
    """The algo-6 foil: `lid` is populated, balance 10 -- unchanged behaviour."""
    ps = ps_from_config(_cfg(ALGO6_CONFIG))
    assert ps.algo == 6
    assert ps.lid is not None
    assert ps.balance == 10


# ---- _tail_lengths: 1-element for algo 3, 2-element for algo 6 ---------------------------


def test_tail_lengths_algo3_single_element() -> None:
    """algo 3 -> a 1-element tail list `[2*BLSTM_NNetInputSize]`, NOT a `(sad, 0)` 2-tuple.
    `BLSTM_NNetInputSize 23` -> tail 46."""
    tails = T._tail_lengths(_cfg(ALGO3_CONFIG), 3)
    assert tails == [46]


def test_tail_lengths_algo6_two_elements() -> None:
    """algo 6 -> `[2*BLSTM_NNetInputSize, 2*BLSTM_LID_NNetInputSize]`. The twin config's
    `BLSTM_NNetInputSize 8` / `BLSTM_LID_NNetInputSize 12` -> `[16, 24]`."""
    tails = T._tail_lengths(_cfg(ALGO6_CONFIG), 6)
    assert tails == [16, 24]


# ---- _ponderations: no LID field for algo != 6 ------------------------------------------


def test_ponderations_algo3_no_lid_field() -> None:
    """algo 3 -> `_ponderations` returns a 1-element ponderation list (the SAD
    `BLSTM_CostPonderation`), NO `BLSTM_LID_CostPonderation` read (which KeyErrors on an
    algo-3 vec2struct -- the RED failure)."""
    ps = ps_from_config(_cfg(ALGO3_CONFIG))
    state = types.SimpleNamespace(ps=ps)
    genome = np.zeros(genome_length(ps))
    ponds, out_param = T._ponderations(genome, None, state)  # type: ignore[arg-type]
    assert len(ponds) == 1
    assert out_param.shape == genome.shape


def test_ponderations_algo6_two_fields() -> None:
    """The algo-6 foil: `_ponderations` returns a 2-element list `[sad_pond, lid_pond]`."""
    ps = ps_from_config(_cfg(ALGO6_CONFIG))
    state = types.SimpleNamespace(ps=ps)
    genome = np.zeros(genome_length(ps))
    ponds, _out = T._ponderations(genome, None, state)  # type: ignore[arg-type]
    assert len(ponds) == 2


# ---- The 1-cell BackPropagation contract (BackPropagation.m:11-13) -----------------------


def test_backpropagation_cell_count_contract() -> None:
    """`BackPropagation.m` populates `weightsIni{1}` (SAD) always and `weightsIni{2}` (LID)
    iff `algo == 6`. The port's cell count is driven by `_tail_lengths` / `_ponderations`
    length: 1 cell for algo 3, 2 cells for algo 6 -- pinned jointly so the two length
    signals can never drift apart."""
    ps3 = ps_from_config(_cfg(ALGO3_CONFIG))
    ps6 = ps_from_config(_cfg(ALGO6_CONFIG))
    st3 = types.SimpleNamespace(ps=ps3)
    st6 = types.SimpleNamespace(ps=ps6)

    n_tail3 = len(T._tail_lengths(_cfg(ALGO3_CONFIG), 3))
    n_tail6 = len(T._tail_lengths(_cfg(ALGO6_CONFIG), 6))
    n_pond3 = len(T._ponderations(np.zeros(genome_length(ps3)), None, st3)[0])  # type: ignore[arg-type]
    n_pond6 = len(T._ponderations(np.zeros(genome_length(ps6)), None, st6)[0])  # type: ignore[arg-type]

    assert n_tail3 == n_pond3 == 1, "algo 3 -> a single [sad] cell"
    assert n_tail6 == n_pond6 == 2, "algo 6 -> the [sad, lid] pair"


# ---- _eval_overlay: no spurious LID injection for algo != 6 ------------------------------
#
# The backprop flags and the F11 `Epochs 0` rule left these builders for `fold_run.FoldRun`
# (issue #22); their pins are `tests/test_fold_run.py`. What stays here is the overlay itself.


def test_eval_overlay_algo3_no_lid_keys() -> None:
    """algo 3 -> only `BLSTM_CostPonderation` injected; NO `BLSTM_LID_*` keys added to a
    single-net base config (the legacy never touches the LID side for algo != 6), and
    nothing else moves (the flags are the fold run's)."""
    base = _cfg(ALGO3_CONFIG)
    cfg = T._eval_overlay(base, ["0.5"], 3)
    assert cfg["BLSTM_CostPonderation"] == "0.5"
    assert "BLSTM_LID_CostPonderation" not in cfg
    assert "BLSTM_LID_BackPropagationActivated" not in cfg
    assert {k: v for k, v in cfg.items() if k != "BLSTM_CostPonderation"} == {k: v for k, v in base.items() if k != "BLSTM_CostPonderation"}


def test_eval_overlay_algo6_injects_lid() -> None:
    """The algo-6 foil: both SAD + LID ponderations injected, in place (byte-stable order)."""
    base = _cfg(ALGO6_CONFIG)
    cfg = T._eval_overlay(base, ["0.5", "0.7"], 6)
    assert cfg["BLSTM_CostPonderation"] == "0.5"
    assert cfg["BLSTM_LID_CostPonderation"] == "0.7"
    assert list(cfg) == list(base)
