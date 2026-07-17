"""Phase 5 Task 9: QPSO narrowed to the non-weight (DSP/config hyperparameter) genome.

The modern regime (user-locked, 2026-07-10): network WEIGHTS train by gradient, so the outer
QuantumPSO search is NARROWED to the DSP/config hyperparameters -- the weight-block +
normalize-tail dims are pinned OUT of the genome PERMANENTLY. These tests pin, WITHOUT any
engine (pure genome/mask logic):

  * `build_hyperparam_mask` / `genome.weight_block_mask` mask EXACTLY the weight/normalize dims
    -- every masked dim's owning field is a printConfig-dropped (weight/normalize) key and vice
    versa, cross-checked against the walk's own `_MaskTraceWalk` ranges AND an independent
    net-structure arithmetic count (NO magic dim numbers);
  * the narrowed genome_length shrinks by exactly the masked-dim count;
  * the generalized injection (`_hyperparam_config_text`) covers exactly the decoded non-weight
    keys (every searchable DSP key, no weight/normalize key), forward-only, base preserved;
  * `masking_validation` passes on the narrowed mask;
  * vec2struct's 4c goldens stay BYTE-GREEN (the regression sentinel -- vec2struct itself is
    untouched).

The engine-backed non-vacuity (distinct hyperparameters -> distinct engine configs AND distinct
costs) lives in `tests/pyo3/test_exit_gate.py`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from speech.config_bridge import parse_legacy_config
from speech.drivers.train import _hyperparam_config_text, build_hyperparam_mask
from speech.genome import (
    RunConfig,
    _MaskTraceWalk,
    _printconfig_written,
    genome_length,
    vec2struct,
    weight_block_mask,
)
from speech.scoring import masking_validation

# The committed vec2struct RunConfig builders + golden loaders -- reused as the regression sentinel.
from tests.test_phase4c_genome import (
    CASES,
    PHASE4C,
    _base,
    _lid,
    _manifest,
    _read_bin,
    _run,
)


def _twin_ps() -> RunConfig:
    """Algo-6 Twin (both SAD + LID nets) -- exercises both weight/normalize mask arms."""
    return _base(6).model_copy(update={"lid": _lid(), "mask_has_nnet": True, "mask_has_nnetLID": True})


def _net_weight_normalize_dims(lstm: list[int], lsub: list[int], out_net: list[int], out_sub: list[int]) -> int:
    """INDEPENDENT (no-walk) count of a single net's weight + normalize genome dims, straight from
    the net sizes -- the derived expectation the mask must match. Mirrors `_lstm_and_output`'s own
    shapes: 3 gate matrices of `base+5` + 1 (narrow) cell matrix of `base+1` per LSTM block, then
    `prev*sub+1` per output neuron, then `2*inputSize` normalize (mean + std)."""
    total = 0
    if lstm[0] != 0:
        for _direction in range(2):  # Forward + Backward
            for ii in range(1, len(lstm)):
                base = lstm[ii - 1] * lsub[ii - 1] + lstm[ii]
                per_block = 3 * (base + 5) + (base + 1)  # gate_nb = base+5, cell_nb = base+1
                total += lstm[ii] * per_block
    for ii in range(1, len(out_net)):
        total += out_net[ii] * (out_net[ii - 1] * out_sub[ii - 1] + 1)
    input_size = lstm[0] if lstm[0] != 0 else out_net[0]
    total += 2 * input_size  # NormalizeInputMean + NormalizeInputStd
    return total


def _expected_masked_dims(ps: RunConfig) -> int:
    """The derived weight/normalize dim count for `ps` (SAD net, + LID net iff algo 6)."""
    masked = _net_weight_normalize_dims(ps.LSTM_net_size, ps.LSTMSubSampling, ps.Output_net_size, ps.OutputSubSampling)
    if ps.algo == 6:
        assert ps.lid is not None
        lid = ps.lid
        masked += _net_weight_normalize_dims(lid.LSTM_net_size, lid.LSTMSubSampling, lid.Output_net_size, lid.OutputSubSampling)
    return masked


_PS_CASES = {"twin": _twin_ps(), "spectral": _base(3), "signal": _base(4)}


# ---- the mask carves EXACTLY the weight/normalize dims -----------------------------------


@pytest.mark.parametrize("name", list(_PS_CASES))
def test_masked_dims_match_derived_net_structure(name: str) -> None:
    """The searchable count is a DERIVED expectation (net-structure arithmetic), not a magic
    number: total genome dims - the independently-computed weight/normalize dim count."""
    ps = _PS_CASES[name]
    _mask, searchable = weight_block_mask(ps)
    total = genome_length(ps) - 1
    assert searchable.shape[0] == total
    n_masked = int((~searchable).sum())
    assert n_masked == _expected_masked_dims(ps), f"{name}: masked dims {n_masked} != derived {_expected_masked_dims(ps)}"
    # the narrowed genome shrinks by exactly the weight/normalize dims (and only for netted algos).
    assert int(searchable.sum()) == total - n_masked
    assert n_masked > 0, f"{name}: a netted algo must mask a non-empty weight/normalize block"


@pytest.mark.parametrize("name", list(_PS_CASES))
def test_every_masked_dim_is_a_weight_normalize_key(name: str) -> None:
    """Every masked dim's owning FIELD is a printConfig-dropped (weight/normalize) key, and every
    searchable dim is NOT masked -- derived from the walk's own `_MaskTraceWalk` ranges, then
    classified by printConfig's OWN rule (`_printconfig_written`), an oracle independent of the
    mask-build path."""
    ps = _PS_CASES[name]
    _mask, searchable = weight_block_mask(ps)
    tw = _MaskTraceWalk(ps)
    tw.run()

    covered = np.zeros(searchable.shape[0], dtype=bool)
    for start, stop, key in tw.masked_ranges:
        assert not _printconfig_written(key), f"{name}: masked range key {key!r} is printConfig-WRITTEN (not a weight/normalize key)"
        assert stop > start
        assert not covered[start:stop].any(), f"{name}: overlapping masked range at {key!r}"
        covered[start:stop] = True
    # the traced ranges are EXACTLY the mask (the ~searchable set), no more, no less.
    assert np.array_equal(covered, ~searchable), f"{name}: traced weight/normalize ranges != mask complement"


@pytest.mark.parametrize("name", list(_PS_CASES))
def test_mask_dict_keys_are_exactly_the_printconfig_dropped_keys(name: str) -> None:
    """`build_hyperparam_mask` pins EXACTLY the printConfig-dropped fields -- the walk's own
    weight/normalize classification -- and every pinned key is a real dim-consuming
    `_MaskTraceWalk` range (no structural/header key sneaks in)."""
    ps = _PS_CASES[name]
    mask = build_hyperparam_mask(ps)
    tw = _MaskTraceWalk(ps)
    tw.run()
    range_keys = {key for _s, _e, key in tw.masked_ranges}

    for key in mask:
        assert not _printconfig_written(key), f"{name}: mask key {key!r} is printConfig-written (would reach the engine config)"
    assert set(mask) == range_keys, f"{name}: mask keys != the traced weight/normalize range keys"
    # every masked field carries a decoded value (weight matrices are arrays; normalize is an array).
    for key, val in mask.items():
        arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
        assert arr.size >= 1 and np.isfinite(arr).all(), f"{name}: mask value for {key!r} is empty/non-finite"


def test_non_netted_algo_masks_nothing() -> None:
    """Algo 0 (VRCTS, no NN): there is no weight/normalize block, so the mask is empty and the
    whole genome stays searchable -- the narrowing is a no-op where there are no weights."""
    ps = _base(0)
    mask, searchable = weight_block_mask(ps)
    assert mask == {}
    assert bool(searchable.all())


# ---- masking_validation passes on the narrowed mask -------------------------------------


@pytest.mark.parametrize("name", list(_PS_CASES))
def test_masking_validation_passes_on_narrowed_mask(name: str) -> None:
    """The narrowed weight/normalize mask round-trips: `masking_validation` (the double vec2struct
    round trip) reports NO mismatch on a representative genome. printConfig never emits the pinned
    weight/normalize fields, so the mask only ever touches dropped keys -- the searchable fields
    (the only ones compared) round-trip exactly as under `mask=None`."""
    ps = _PS_CASES[name]
    mask = build_hyperparam_mask(ps)
    rng = np.random.default_rng(1234)
    param = rng.uniform(0.0, ps.adim, size=genome_length(ps))
    assert masking_validation(param, mask, ps) == [], f"{name}: narrowed mask failed masking_validation"


# ---- the generalized injection covers exactly the non-weight decoded keys ----------------


@pytest.mark.parametrize("name", list(_PS_CASES))
def test_injection_covers_nonweight_keys_only(name: str) -> None:
    """`_hyperparam_config_text` overlays EVERY decoded non-weight key onto the base and NO
    weight/normalize key (printConfig already stripped them), forward-only (backprop OFF,
    Epochs 0), with the base's own extra keys preserved (byte-stable overlay)."""
    ps = _PS_CASES[name]
    mask, searchable = weight_block_mask(ps)
    rng = np.random.default_rng(7)
    full = np.zeros(searchable.shape[0])
    full[searchable] = rng.uniform(0.0, ps.adim, size=int(searchable.sum()))
    config_struct, _out, _c = vec2struct(full, mask, ps, 0)

    # configStruct already carries no weight/normalize key (printConfig drops them).
    assert not any(("_LSTMBlock_" in k or "_Output_Layer_" in k or "NormalizeInput" in k) for k in config_struct)

    base = {"BLSTM_decision_thresh_rising": "0.5", "Dump_Directory": "dump", "fileslisting": "x.csv", "language2classmapping": "m.csv"}
    text = _hyperparam_config_text(base, config_struct, ps.algo)
    lines = [ln for ln in text.strip().split("\n") if ln]
    keys = {ln.split(" ", 1)[0] for ln in lines}

    # every decoded (non-weight) key is injected ...
    for k in config_struct:
        assert k in keys, f"{name}: decoded key {k!r} missing from the injected config"
    # ... and no weight/normalize key appears in the emitted config ...
    assert not any(("_LSTMBlock_" in k or "_Output_Layer_" in k or "NormalizeInput" in k) for k in keys)
    # ... the base's own non-decoded keys survive (byte-stable overlay) ...
    assert "Dump_Directory" in keys
    # ... and the eval is forward-only (fixed base weights, no engine-internal training).
    assert "BLSTM_BackPropagationActivated false" in text
    assert "Neural_Networks_BackPropagation_Epochs 0" in text
    if ps.algo == 6:
        assert "BLSTM_LID_BackPropagationActivated false" in text


def test_injection_is_far_stronger_than_the_2key_check() -> None:
    """The generalized injection moves MANY DSP keys, not just the 2 legacy CostPonderation fields:
    two genomes differing across the searchable space produce injected configs differing in a broad
    set of DSP keys (freq bands, windows, decision thresholds, calibration, ponderations)."""
    ps = _base(3)
    mask, searchable = weight_block_mask(ps)
    n = int(searchable.sum())
    ga = np.zeros(searchable.shape[0])
    ga[searchable] = np.random.default_rng(1).uniform(0.0, ps.adim, size=n)
    gb = np.zeros(searchable.shape[0])
    gb[searchable] = np.random.default_rng(2).uniform(0.0, ps.adim, size=n)
    cfg_a, _oa, _ca = vec2struct(ga, mask, ps, 0)
    cfg_b, _ob, _cb = vec2struct(gb, mask, ps, 0)
    differing = {k for k in cfg_a if cfg_a[k] != cfg_b.get(k)}
    assert len(differing) > 5, f"only {len(differing)} keys differ -- injection is not broader than the 2-key check"
    # the differing set is DSP hyperparameters, never a weight/normalize key.
    assert not any(("_LSTMBlock_" in k or "NormalizeInput" in k) for k in differing)


# ---- the regression sentinel: vec2struct's 4c goldens stay byte-green --------------------


@pytest.mark.parametrize("name", CASES)
def test_vec2struct_4c_goldens_still_byte_green(name: str) -> None:
    """vec2struct itself is UNTOUCHED by the narrowing -- its 4c Octave goldens (count_param, the
    printConfig configStruct, and the bit-exact out_param) all still match. This is the regression
    sentinel: build_hyperparam_mask only READS the walk, never changes it."""
    cfg, out, count, _param = _run(name)
    expected = cast(dict[str, int], _manifest()["counts"])[name]
    assert count == expected, f"{name}: count_param regressed ({count} != {expected})"
    golden = parse_legacy_config((PHASE4C / f"genome_{name}.config").read_text())
    assert list(cfg.items()) == list(golden.items()), f"{name}: configStruct regressed"
    gout = _read_bin(PHASE4C / f"genome_{name}_out_param.bin")
    assert out.shape == gout.shape
    assert out.view(np.uint64).tobytes() == gout.view(np.uint64).tobytes(), f"{name}: out_param regressed"
