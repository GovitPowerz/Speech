"""Assert the Phase-0b-i cost-sweep fixtures are well-formed."""

from pathlib import Path

from speech import weight_bridge

REF = Path("tests/reference_data/phase0b")


def test_sweep_fixture_shapes() -> None:
    for name in ["sweep_output", "sweep_cost_speech", "sweep_cost_nospeech", "sweep_deriv_speech", "sweep_deriv_nospeech"]:
        _, cols, data = weight_bridge.read_bin(REF / f"{name}.bin")
        assert cols == 1000
        assert data.shape == (1000,)


def test_output_column_monotone() -> None:
    _, _, out = weight_bridge.read_bin(REF / "sweep_output.bin")
    assert (out[1:] >= out[:-1]).all()  # output sweep is a monotone grid
    assert out.min() >= 0.0 and out.max() <= 1.0
