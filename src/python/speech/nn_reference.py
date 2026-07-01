"""Pure-Python BLSTM forward/backward oracle for gradient-check fixtures only.

Ported from legacy MATLAB: BLSTM_Forward.m, BLSTM_Backward.m, Hz2Mel.m, Mel2Hz.m.
NOT a production path - used to validate the Rust engine (Phase 2-3).
"""

import numpy as np
from numpy.typing import NDArray


def blstm_forward(inputs: NDArray[np.float64], weights: dict[str, object]) -> NDArray[np.float64]:
    """Reference bidirectional-LSTM forward pass (Phase 2)."""
    ...
