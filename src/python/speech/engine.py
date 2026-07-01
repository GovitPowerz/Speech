"""Engine driver: invoke the Rust engine, assemble cost + gradients.

Ported from legacy MATLAB: ComputeCost.m, ComputeGradient*.m, CostFunction*.m,
and RunFsp.py (subprocess seam -> in-process PyO3 call to `speech_rs`).
See design spec section 6.
"""

from numpy.typing import NDArray


def forward_backward(config_path: str, weights: NDArray, batch: list[str]) -> tuple[float, NDArray]:
    """Run the Rust engine over a batch, returning (cost, gradient) (Phase 4)."""
    ...
