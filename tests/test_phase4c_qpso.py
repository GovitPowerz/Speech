"""Phase 4c Task 11: QuantumPSO bit-pin against the shared-random-table Octave oracle.

The goldens (`tests/reference_data/phase4c/qpso_*.bin`) are dumped by GNU Octave running a
MODIFIED-COPY of the vendored `QuantumPSO.m` (`tools/octave_harness/qpso_modified/`) whose
rand/randperm/stblrnd are substituted by reads of a committed random table
(`qpso_random_table.bin`) and whose CostFunction is a quadratic surrogate; the wall-clock
reseed is removed. This test replays the IDENTICAL table through the Python port
(`speech.optimizers.quantum_pso`) and compares.

Per-value comparator split (`tests/_libm_gate.py`):
  - `init_*` (post-init state): PURE arithmetic (normmat division + the surrogate + min/sortrows)
    -> STRICT bits on every platform.
  - every `*_traj` + `final_out`: transcendental from epoch 1 (the QDPSO `log(1/u)` jump and the
    Levy tan/atan/sin/cos/log/pow) -> canary-gated (bit-exact on the oracle libm, measured;
    <=4 ULP off it). The ranking is libm-robust: the extractor guards a min gbestval+pbestval
    gap >> ULP, so no sortrows/min tie can flip discretely under a different libm.
  - `cursor_end`: exact integer draw count (== `TableRng.cursor`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from speech.optimizers import QpsoObjective, QpsoParams, QpsoResult, TableRng, cmaes_optimize, levy_stable_cms, quantum_pso
from speech.weight_bridge import read_bin

from tests._libm_gate import assert_f64_close

PHASE4C = Path(__file__).resolve().parent / "reference_data" / "phase4c"
MANIFEST = PHASE4C / "manifest.json"


def _surrogate_quadratic_factory(center: NDArray[np.float64]) -> QpsoObjective:
    """The quadratic cost surrogate replacing the engine `CostFunction` in BOTH the Octave
    oracle stage (`qpso_surrogate.m`) and these tests: `cost(row) = sum_j (row_j - center_j)^2`,
    pass-through positions. The sum is an EXPLICIT left-to-right fold to match Octave's
    `sum(.,2)` order bit-for-bit. Pure arithmetic (no libm) -- the trajectory's transcendentals
    come only from the QDPSO `log(1/u)` and the Levy sampler, not the surrogate."""
    c = np.asarray(center, dtype=np.float64).reshape(-1)

    def cost_fn(positions: NDArray[np.float64], mode: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        p = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        sq = (p - c) ** 2
        out = sq[:, 0].copy()
        for j in range(1, sq.shape[1]):
            out = out + sq[:, j]
        return out, p.copy()

    return cost_fn


def _load(name: str) -> NDArray[np.float64]:
    rows, cols, flat = read_bin(PHASE4C / f"{name}.bin")
    return np.asarray(flat, dtype=np.float64).reshape((rows, cols), order="F")


def _qpso_meta() -> dict[str, object]:
    return cast(dict[str, object], cast(dict[str, object], json.loads(MANIFEST.read_text()))["qpso"])


def _build_params() -> tuple[QpsoParams, int, NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct the exact stage params from the manifest (single source of truth)."""
    meta = _qpso_meta()
    p = cast(dict[str, object], meta["params"])
    d = int(cast(int, meta["D"]))
    adim = float(cast(float, p["adim"]))
    center = np.asarray(cast(list[float], p["center"]), dtype=np.float64)
    seed_value = np.asarray(cast(list[list[float]], p["seed_value"]), dtype=np.float64)
    vr = np.column_stack([np.zeros(d), adim * np.ones(d)])
    mv = (vr[:, 1] - vr[:, 0]) / 2
    params = QpsoParams(
        ps=int(cast(int, meta["ps"])),
        me=int(cast(int, meta["me"])),
        ac1=float(cast(float, p["ac1"])),
        ac2=float(cast(float, p["ac2"])),
        iw1=float(cast(float, p["iw1"])),
        iw2=float(cast(float, p["iw2"])),
        iwe=int(cast(int, p["iwe"])),
        ergrd=float(cast(float, p["ergrd"])),
        ergrdep=int(cast(int, p["ergrdep"])),
        errgoal=float("nan"),
        trelea=int(cast(int, p["trelea"])),
        pso_seed=int(cast(int, p["pso_seed"])),
        minmax=int(cast(int, p["minmax"])),
        vr=vr,
        mv=mv,
        seed_value=seed_value,
        backprop_activated=0,
    )
    table = read_bin(PHASE4C / "qpso_random_table.bin")[2]
    return params, d, center, table


def _run_port(dead_banks: bool = True) -> tuple[QpsoResult, QpsoParams, int]:
    params, d, center, table = _build_params()
    cost_fn = _surrogate_quadratic_factory(center)
    res = quantum_pso(cost_fn, params, d, TableRng(table), _dead_banks=dead_banks)
    return res, params, d


def _assert_bits(got: NDArray[np.float64], want: NDArray[np.float64], label: str) -> None:
    g = np.ascontiguousarray(got, dtype=np.float64).reshape(-1)
    w = np.ascontiguousarray(want, dtype=np.float64).reshape(-1)
    assert g.shape == w.shape, f"{label}: shape {g.shape} != {w.shape}"
    assert g.view(np.uint64).tobytes() == w.view(np.uint64).tobytes(), f"{label}: bit mismatch got={g!r} want={w!r}"


def _assert_close(got: NDArray[np.float64], want: NDArray[np.float64], label: str) -> None:
    g = np.asarray(got, dtype=np.float64).reshape(-1)
    w = np.asarray(want, dtype=np.float64).reshape(-1)
    assert g.shape == w.shape, f"{label}: shape {g.shape} != {w.shape}"
    for i in range(g.size):
        assert_f64_close(float(g[i]), float(w[i]), f"{label}[{i}]")


def test_qpso_trajectory_bit_exact_given_table() -> None:
    """The full QuantumPSO trajectory replayed through the port over the shared table.
    init_* STRICT (pure arithmetic); the per-epoch trajectory + final_out canary-gated
    (transcendental via log(1/u) + Levy). The draw count (cursor_end) is pinned exactly."""
    r, params, d = _run_port()

    # --- init: STRICT bits ---
    _assert_bits(r.init_pos, _load("qpso_init_pos"), "init_pos")
    _assert_bits(r.init_pbest, _load("qpso_init_pbest"), "init_pbest")
    _assert_bits(r.init_pbestval, _load("qpso_init_pbestval").reshape(-1), "init_pbestval")
    _assert_bits(r.init_gbest, _load("qpso_init_gbest").reshape(-1), "init_gbest")
    _assert_bits(np.array([r.init_gbestval]), _load("qpso_init_gbestval").reshape(-1), "init_gbestval")

    # --- per-epoch trajectory: canary-gated ---
    me, ps = params.me, params.ps
    pos_traj = _load("qpso_pos_traj")  # ps x D*me
    pbest_traj = _load("qpso_pbest_traj")
    pbestval_traj = _load("qpso_pbestval_traj")  # ps x me
    gbest_traj = _load("qpso_gbest_traj")  # me x D
    gbestval_traj = _load("qpso_gbestval_traj").reshape(-1)  # me
    for k in range(me):
        _assert_close(r.pos_traj[k], pos_traj[:, k * d : (k + 1) * d], f"pos_traj[{k}]")
        _assert_close(r.pbest_traj[k], pbest_traj[:, k * d : (k + 1) * d], f"pbest_traj[{k}]")
        _assert_close(r.pbestval_traj[k], pbestval_traj[:, k], f"pbestval_traj[{k}]")
        _assert_close(r.gbest_traj[k], gbest_traj[k, :], f"gbest_traj[{k}]")
        assert_f64_close(float(r.gbestval_traj[k]), float(gbestval_traj[k]), f"gbestval_traj[{k}]")
    _assert_close(r.out, _load("qpso_final_out").reshape(-1), "final_out")
    assert ps == r.pbest.shape[0]

    # --- exact draw count ---
    cursor_end = int(_load("qpso_cursor_end").reshape(-1)[0])
    table = read_bin(PHASE4C / "qpso_random_table.bin")[2]
    src = TableRng(table)
    quantum_pso(_surrogate_quadratic_factory(_build_params()[2]), params, d, src)
    assert src.cursor == cursor_end, f"port consumed {src.cursor} draws, oracle {cursor_end}"


def test_levy_cms_matches_stblrnd() -> None:
    """The standalone Chambers-Mallows-Stuck Levy sample over table[0:16] (V=table[0:8],
    W=table[8:16]) vs the Octave `tbl_stblrnd(1.3,1,0.5,0,1,8)` dump. MEASURED bit-for-bit on
    the oracle libm (assert_f64_close is hex-exact there); canary-gated (<=4 ULP) off it, since
    the transform traverses tan/atan/sin/cos/log/pow. Also pins the exact 16-draw consumption."""
    table = read_bin(PHASE4C / "qpso_random_table.bin")[2]
    want = _load("qpso_levy_out").reshape(-1)
    levy_in = _load("qpso_levy_in").reshape(-1)
    assert np.array_equal(levy_in, table[:16]), "levy_in must be exactly table[0:16]"
    src = TableRng(table)
    got = levy_stable_cms(src, 1.3, 1, 0.5, 0, 8)
    assert src.cursor == 16, f"levy consumed {src.cursor} draws, expected 16 (2*8: V then W)"
    _assert_close(got, want, "levy_cms")
    assert np.all(np.isfinite(got)) and np.any(got != 0.0), "levy sample degenerate"


def test_dead_banks_consume_stream() -> None:
    """Non-vacuity of the DEAD velocity banks (QuantumPSO.m:375-411): they draw 8*ps*D uniforms
    per epoch even though the apply gate (:447 `i > 20*me/2`) is unreachable. Removing those
    draws (the `_dead_banks=False` mutation -- a local variant, NOT a committed source edit)
    shifts the ENTIRE downstream stream, so the position trajectory diverges from the pinned run.
    This proves the banks are load-bearing for stream alignment, not ignorable."""
    r, params, d = _run_port(dead_banks=True)
    m, _, _ = _run_port(dead_banks=False)
    # The pinned run matches the oracle (bit-exact epoch 0 pos); the mutated run must NOT.
    pos0_oracle = _load("qpso_pos_traj")[:, 0:d]
    _assert_close(r.pos_traj[0], pos0_oracle, "dead_banks real pos_traj[0]")
    diverged = any(not np.allclose(r.pos_traj[k], m.pos_traj[k]) for k in range(params.me))
    assert diverged, "dead-bank draws removed but the trajectory is unchanged -- they would be vacuous"
    # And specifically epoch 0 already shifts (the very first per-epoch draws are the dead banks).
    assert not np.allclose(r.pos_traj[0], m.pos_traj[0]), "epoch-0 pos unaffected by the dead banks"


def test_cmaes_optimize_smoke() -> None:
    """cmaes_optimize is NOT bit-pinned (a different `cma`-package lineage than the vendored
    cmaes.m); a smoke test only. On a separable quadratic it must converge near the optimum,
    driving cost_fn with the whole ask() population (EvalParallel='yes' shape)."""
    center = np.array([2.0, -3.0, 5.0])
    cost_fn = _surrogate_quadratic_factory(center)
    res = cmaes_optimize(cost_fn, x0=np.zeros(3), sigma0=2.0, opts={"seed": 7, "maxfevals": 3000, "verbose": -9})
    assert np.allclose(res.x, center, atol=1e-2), f"cmaes did not converge: {res.x} vs {center}"
    assert res.fun < 1e-3 and res.evals > 0
