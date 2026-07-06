"""Assert the committed MAT v5 sample (Rust MatWriter output) loads bit-exact via scipy.

pytest cannot call Rust directly (no PyO3 seam yet), so the seam is a committed sample
written by `cargo test --test phase4a_matfile -- --ignored write_sample`
(`src/rust/tests/phase4a_matfile.rs`) and re-generated only if the writer changes.
"""

from pathlib import Path
from typing import cast

import numpy as np
import scipy.io

SAMPLE = Path("tests/reference_data/phase4a/matwriter_sample.mat")


def test_matwriter_sample_loads_bit_exact() -> None:
    mat = scipy.io.loadmat(SAMPLE)
    a = cast(np.ndarray, mat["A"])
    s = cast(np.ndarray, mat["s"])

    expected_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert a.shape == (2, 3)
    assert a.dtype == np.float64
    assert np.array_equal(a, expected_a)

    assert s.shape == (1, 1)
    assert s.dtype == np.float64
    assert s[0, 0] == 1.5
