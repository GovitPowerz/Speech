"""Optimizer zoo for NN training and DSP-hyperparameter search.

Ported from legacy MATLAB: SMORMS3.m (wired), Adam.m, RMSprop.m, Rprop.m,
QuantumPSO.m (bespoke - hand-port, do NOT swap for pymoo), cmaes.m, sfo.m,
pso_Trelea_vectorized.m, BackPropagation.m. See design spec section 6.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def smorms3(grad_fn: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]], theta0: NDArray[np.float64]) -> NDArray[np.float64]:
    """SMORMS3 gradient descent (the wired NN optimizer) (Phase 3)."""
    ...
