"""LID confusion matrices, EER cutoff, gradient checks, masking sanity.

Ported from legacy MATLAB: confusion*.m, CheckGrad.m, MaskingValidation.m.
See design spec section 6.
"""

import numpy as np
from numpy.typing import NDArray


def confusion_matrix(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> NDArray[np.int64]:
    """Per-language confusion matrix with in-band target signaling (Phase 4)."""
    ...
