"""Phase 4d Task 1: packer/adim byte golden closed against the real 2015 weight packs.

Closes the Phase-0a deferral (CLAUDE.md "Phase 0a status"): the flat weight packer +
`adim_coeff` seam was never validated against an INDEPENDENT legacy golden pack, because none
was known to exist at Phase 0a time. Two real production tuples (real `.config` + real
`NNweights_config1.bin`) surfaced in the full legacy copy (see
`tests/reference_data/phase4d/phase4d_sources.json` for provenance + SHA-256):
  - Tuple A: Optimizer_V6.2.2/15-Oct-2015_BLSTM_OpenSAD15 (byte-identical to the
    already-committed `tests/reference_data/phase0` fixtures - the familiar 33,671 shape).
  - Tuple B: Optimizer_V6.2.1/14-Oct-2015_BLSTM_OpenSAD15 (genuinely new: a wider
    NNetInputSize=35 front end, MEASURED element count 42,911).

Cross-language agreement (per `phase0_packer.rs`'s header convention): "Rust-pack ==
Python-pack" is established by BOTH languages independently repacking the real `.bin` back to
BYTE-IDENTICAL committed fixture bytes, rather than a literal cross-process call - see
`src/rust/tests/phase4d_packer_golden.rs` for the Rust half of this same pair of assertions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from speech import config_bridge
from speech import weight_bridge as wb

REF = Path("tests/reference_data/phase4d")


@dataclass(frozen=True)
class FixtureTuple:
    tag: str
    config_name: str
    bin_name: str
    expected_element_count: int


TUPLE_A = FixtureTuple(tag="A", config_name="tupleA_1_worker_1.config", bin_name="tupleA_NNweights_config1.bin", expected_element_count=33671)
TUPLE_B = FixtureTuple(tag="B", config_name="tupleB_1_worker_1.config", bin_name="tupleB_NNweights_config1.bin", expected_element_count=42911)


def _spec_for(t: FixtureTuple) -> dict[str, object]:
    cfg = config_bridge.parse_legacy_config((REF / t.config_name).read_text())
    return config_bridge.nnet_spec(cfg, prefix="BLSTM")


def test_tuple_a_element_count_is_the_familiar_33671_shape() -> None:
    spec = _spec_for(TUPLE_A)
    assert spec["LSTMNeuronNb"] == [23, 24, 24]
    assert spec["NNetInputSize"] == 23
    assert wb.element_count(spec) == TUPLE_A.expected_element_count
    rows, cols, data = wb.read_bin(REF / TUPLE_A.bin_name)
    assert cols == 1
    assert rows == TUPLE_A.expected_element_count
    assert data.shape[0] == TUPLE_A.expected_element_count


def test_tuple_b_element_count_is_measured_at_42911() -> None:
    spec = _spec_for(TUPLE_B)
    assert spec["LSTMNeuronNb"] == [35, 24, 24]
    assert spec["NNetInputSize"] == 35
    assert wb.element_count(spec) == TUPLE_B.expected_element_count
    rows, cols, data = wb.read_bin(REF / TUPLE_B.bin_name)
    assert cols == 1
    assert rows == TUPLE_B.expected_element_count
    assert data.shape[0] == TUPLE_B.expected_element_count

    # Cross-check independent of NnetSpec/element_count entirely: the raw file size alone
    # pins the element count (16-byte header + 8 bytes/element).
    size = (REF / TUPLE_B.bin_name).stat().st_size
    assert (size - 16) // 8 == TUPLE_B.expected_element_count


def _assert_repack_byte_identical(t: FixtureTuple, tmp_path: Path) -> None:
    spec = _spec_for(t)
    bin_path = REF / t.bin_name
    rows, cols, flat = wb.read_bin(bin_path)
    assert flat.shape[0] == wb.element_count(spec), t.tag

    net = wb.flat_to_nnet(flat, spec)
    repacked = wb.nnet_to_flat(net, spec)
    assert np.array_equal(repacked, flat), f"tuple {t.tag}: vector-level bijection"

    out_path = tmp_path / t.bin_name
    wb.write_bin(rows, cols, repacked, out_path)

    original_bytes = bin_path.read_bytes()
    repacked_bytes = out_path.read_bytes()
    assert repacked_bytes == original_bytes, f"tuple {t.tag}: repacked file bytes must be byte-identical to the committed fixture"


def test_tuple_a_repack_is_byte_identical_to_the_committed_file(tmp_path: Path) -> None:
    _assert_repack_byte_identical(TUPLE_A, tmp_path)


def test_tuple_b_repack_is_byte_identical_to_the_committed_file(tmp_path: Path) -> None:
    _assert_repack_byte_identical(TUPLE_B, tmp_path)


def test_tuple_a_pack_weights_matches_rust_verified_byte_golden() -> None:
    """`pack_weights`/`unpack_weights` (the Phase 4c public wrapper) close the same loop."""
    spec = _spec_for(TUPLE_A)
    flat = wb.read_weight_vector(REF / TUPLE_A.bin_name)
    net = wb.unpack_weights(flat, spec)
    assert np.array_equal(wb.pack_weights(net), flat)


def test_tuple_b_pack_weights_matches_rust_verified_byte_golden() -> None:
    spec = _spec_for(TUPLE_B)
    flat = wb.read_weight_vector(REF / TUPLE_B.bin_name)
    net = wb.unpack_weights(flat, spec)
    assert np.array_equal(wb.pack_weights(net), flat)


def _assert_adim_invariants(t: FixtureTuple) -> None:
    """The `adim_coeff` invariant (CLAUDE.md locked decision: `sqrt(fan_in*sub + fan_out)`,
    bias-excepted) holds structurally for both tuples' REAL architectures: every non-bias
    column of every gate/output row is scaled by exactly `1/adim`, and every bias column (the
    row's last element) passes through `config_to_nnet` untouched. `flat_to_nnet`'s output is
    fed in as the `structured` (config-domain) input purely as a shape-compatible probe -- this
    checks the SCALING BEHAVIOR of `config_to_nnet`, not a value-level domain claim.
    """
    spec = _spec_for(t)
    flat = wb.read_weight_vector(REF / t.bin_name)
    probe = wb.flat_to_nnet(flat, spec)
    scaled = wb.config_to_nnet(probe, spec)

    lstm, lsub = cast(list[int], spec["LSTMNeuronNb"]), cast(list[int], spec["LSTMSubSampling"])
    for direction in ("forward", "backward"):
        for i, (layer_probe, layer_scaled) in enumerate(zip(probe[direction], scaled[direction], strict=True)):
            out = lstm[i + 1]
            fin = lstm[i] * lsub[i]
            adim = float(np.sqrt(lstm[i] * lsub[i] + lstm[i + 1]))
            assert np.isfinite(adim) and adim > 0.0, f"tuple {t.tag}: adim must be finite positive"
            for gate, ncols in (("input", fin + out + 5), ("forget", fin + out + 5), ("output", fin + out + 5), ("cell", fin + out + 1)):
                row_probe = layer_probe[gate]
                row_scaled = layer_scaled[gate]
                assert row_probe.shape == (out, ncols)
                # Bias (last column) is exempt: passes through unscaled, bit-exact.
                assert np.array_equal(row_scaled[:, -1], row_probe[:, -1]), f"tuple {t.tag} layer {i} gate {gate}: bias must pass through"
                # Every other column is scaled by exactly 1/adim.
                assert np.array_equal(row_scaled[:, :-1], row_probe[:, :-1] / adim), f"tuple {t.tag} layer {i} gate {gate}: body scaling"

    outn, osub = cast(list[int], spec["OutputNeuronNb"]), cast(list[int], spec["OutputSubSampling"])
    for i, (mat_probe, mat_scaled) in enumerate(zip(probe["output"], scaled["output"], strict=True)):
        adim = float(np.sqrt(outn[i] * osub[i]))
        assert np.isfinite(adim) and adim > 0.0, f"tuple {t.tag}: output adim must be finite positive"
        assert np.array_equal(mat_scaled[:, -1], mat_probe[:, -1]), f"tuple {t.tag} output layer {i}: bias must pass through"
        assert np.array_equal(mat_scaled[:, :-1], mat_probe[:, :-1] / adim), f"tuple {t.tag} output layer {i}: body scaling"


def test_tuple_a_adim_invariants_hold() -> None:
    _assert_adim_invariants(TUPLE_A)


def test_tuple_b_adim_invariants_hold() -> None:
    _assert_adim_invariants(TUPLE_B)
