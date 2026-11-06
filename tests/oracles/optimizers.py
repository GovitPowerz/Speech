"""The QPSO oracle's table reader, a `speech.optimizers.RandSource` (issue #133: test-only,
out of the shipped package; production passes a numpy `Generator`)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


class TableRng:
    """A table-backed random source: reads sequential f64s from a committed table and
    exposes the exact draw shapes QuantumPSO uses. BOTH the Octave modified-copy oracle
    (`tools/octave_harness/qpso_modified/`) and this port read the SAME committed table
    (`tests/reference_data/phase4c/qpso_random_table.bin`) through an identical cursor, so
    the trajectory is bit-pinned.

    Column-major fill is the load-bearing detail: MATLAB `rand([m,n])` lays out its stream
    column-major, so `rand(m,n)` here reads `m*n` values and reshapes with `order='F'` --
    element `(i,j)` gets `table[cursor + j*m + i]`. The Octave `tbl_rand` does the same via
    `reshape(vals,[m,n])` (Octave reshape is column-major). numpy's default row-major fill
    would silently transpose every matrix draw; do NOT use it here.
    """

    def __init__(self, table: NDArray[np.float64]) -> None:
        self.table: NDArray[np.float64] = np.asarray(table, dtype=np.float64).reshape(-1)
        self.cursor: int = 0

    def _take(self, count: int) -> NDArray[np.float64]:
        if self.cursor + count > self.table.size:
            raise IndexError(f"TableRng exhausted: need {count} at cursor {self.cursor}, size {self.table.size}")
        vals = self.table[self.cursor : self.cursor + count]
        self.cursor += count
        return vals

    def rand(self, *shape: int) -> float | NDArray[np.float64]:
        """`rand()` -> scalar; `rand(n)` -> length-n vector; `rand(m,n)` -> (m,n) column-major."""
        if len(shape) == 0:
            return float(self._take(1)[0])
        dims = [int(s) for s in shape]
        count = int(np.prod(dims))
        vals = self._take(count)
        if len(dims) == 1:
            return vals.copy()
        return vals.reshape(tuple(dims), order="F").copy()

    def randperm(self, n: int, k: int | None = None) -> NDArray[np.int64]:
        """Table-based Durstenfeld/Fisher-Yates over 1..n (MATLAB 1-based values), consuming
        `n-1` draws: `for i=n..2: j=floor(rand*i)+1; swap p[i],p[j]`. Returns the first `k`.
        Identical algorithm on the Octave side (`tbl_randperm.m`), so the selected indices
        match -- NOT MATLAB's builtin randperm (which is not table-reproducible)."""
        p = list(range(1, n + 1))
        for i in range(n, 1, -1):
            r = float(self._take(1)[0])
            j = int(math.floor(r * i)) + 1
            p[i - 1], p[j - 1] = p[j - 1], p[i - 1]
        return np.array(p if k is None else p[:k], dtype=np.int64)
