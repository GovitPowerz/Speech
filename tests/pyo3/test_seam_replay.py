"""Phase 4c Task 4: seam integration goldens.

Replays the committed Phase 4a/4b train + gradCheck fixtures IN-PROCESS through
`speech_rs.Engine` (Phase 4c Task 3) instead of the spawned-binary/harness route
those fixtures were originally dumped from. The calling convention (how to seed
a tempdir, which config keys matter, which comparator tolerance regime applies
to which fixture) is mirrored from the Rust golden tests that replay the SAME
fixtures against `CorpusProcessor` directly:

- `src/rust/tests/phase4a_train_golden.rs::tier2_train_epoch_weights_golden`
- `src/rust/tests/phase4a_gradcheck_golden.rs::tier2_gradcheck_golden`
- `src/rust/tests/phase4b_corpus_lid.rs::twin_train_epoch_weights_golden` /
  `twin_gradcheck_golden`

Two DIFFERENT tolerance regimes are load-bearing here, not one:

- tier2 fixtures (phase4a) are dumped from a harness reimpl chain that is
  bit-exact-or-canary-gated against THIS port's own ascending-loop arithmetic
  (`assert_oracle_eq_f64`'s libm-canary axis) -> mirrored via
  `tests/_libm_gate.py::assert_f64_close`.
- twin/lid5 gradcheck fixtures (phase4b) are dumped from the REAL COMPILED
  Eigen engine, which the Rust golden documents as hitting occasional 1-3 ULP
  real-Eigen-vs-ascending-loop rounding splits even at the shrunk TINY-net
  shapes (`phase4b_corpus_lid.rs::assert_gradcheck`'s doc comment) -- a GEMM
  block-order divergence, not a libm one, so the libm-canary comparator does
  not apply; mirrored here via `_assert_gradcheck_report` using the same
  measured ULP/abs bounds as the Rust `assert_gradcheck` helper, including the
  Task-9 1e-6 tightness bound on mean_relative_error for the twin leg.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import scipy.io
from numpy.typing import NDArray
from speech.config_bridge import parse_legacy_config as parse_legacy_config_py
from speech.engine import forward_backward
from speech.optimizers import Smorms3
from speech.weight_bridge import read_bin

from tests._libm_gate import assert_f64_close

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"


@contextmanager
def chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# ==== Corpus seeding (mirrors the Rust golden tests' seed_* helpers) =========


def _seed_tier2_spectral(dst: Path) -> None:
    """Mirrors `phase4a_train_golden.rs::seed_tier2`: 2-file corpus + config +
    the phase0 weight pack the config's relative `BLSTM_weightsFile` resolves
    to, plus the `Dump_Directory` the config points VRCTS writes at."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for f in ("f1", "f2"):
        shutil.copy(PHASE4A / "corpus" / f"{f}.wav", corpus / f"{f}.wav")
        shutil.copy(PHASE4A / "corpus" / f"{f}.stm", corpus / f"{f}.stm")
    for f in ("language2classmapping.csv", "tier2_fileslisting.csv", "tier2_spectral.config"):
        shutil.copy(PHASE4A / f, dst / f)
    shutil.copy(PHASE0 / "NNweights_config1.bin", dst / "NNweights_config1.bin")
    (dst / "vrcts_tier2").mkdir()


def _seed_tier2_gradcheck(dst: Path) -> None:
    """Mirrors `phase4a_gradcheck_golden.rs::seed_gradcheck`: the 1-file
    synthetic-net corpus. No weight pack -- the net is seeded via `set_weights`
    from the committed `tier2_gradcheck_seed.bin` pattern."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for ext in ("wav", "stm"):
        shutil.copy(PHASE4A / "corpus" / f"f1.{ext}", corpus / f"f1.{ext}")
    for f in ("language2classmapping.csv", "tier2_gc_fileslisting.csv", "tier2_gradcheck.config"):
        shutil.copy(PHASE4A / f, dst / f)


def _seed_twin_train(dst: Path) -> None:
    """Mirrors `phase4b_corpus_lid.rs::seed_t9`'s phSeq leg: the Mode-7
    2-file phSeq corpus + config + both TINY-net seed packs."""
    for f in ("twin_train.config", "languagemapping_lid7.csv", "tiny_sad_seed.bin", "tiny_lid_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    phseq = dst / "corpus_phseq"
    phseq.mkdir(parents=True, exist_ok=True)
    for f in ("s1.phSeq", "s2.phSeq", "s1.stm", "s2.stm", "listing_train.csv"):
        shutil.copy(PHASE4B / "corpus_phseq" / f, phseq / f)


def _seed_twin_gradcheck(dst: Path) -> None:
    """Mirrors `phase4b_corpus_lid.rs::seed_t9`'s wav leg: the 1-file wav
    corpus + config + both TINY-net seed packs (Algo 6, Mode 5)."""
    for f in ("twin_gradcheck.config", "languagemapping_lid7.csv", "listing_gc_wav.csv", "tiny_sad_seed.bin", "tiny_lid_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    wav = dst / "corpus_lid"
    wav.mkdir(parents=True, exist_ok=True)
    for f in ("f1.wav", "f1.stm", "f1_gc.stm"):
        shutil.copy(PHASE4B / "corpus_lid" / f, wav / f)


# ==== Comparators ============================================================


def _load_bin_matrix(path: Path) -> NDArray[np.float64]:
    """`.bin` (io::binary column-major) -> a row-major (rows, cols) array, the
    same shape scipy hands back for a `.mat` variable."""
    rows, cols, flat = read_bin(path)
    return flat.reshape((rows, cols), order="F")


def _assert_weight_vector(got: NDArray[np.float64], want_path: Path, label: str) -> None:
    """Per-element canary-gated compare of a flat weight vector against an
    `N x 1` `.bin` golden (mirrors the Rust golden's per-weight loop over
    `assert_oracle_eq_f64`)."""
    rows, cols, flat = read_bin(want_path)
    assert cols == 1, f"{label}: golden must be a column vector"
    assert got.shape == (rows,), f"{label}: shape {got.shape} != ({rows},)"
    for k in range(rows):
        assert_f64_close(float(got[k]), float(flat[k]), f"{label}[{k}]")


def _assert_matrix(
    got: NDArray[np.float64],
    want: NDArray[np.float64],
    strict_cols: tuple[int, ...],
    masked_cols: tuple[int, ...],
    label: str,
) -> None:
    """Mirrors the Rust `common::assert_matrix`: `strict_cols` bit-exact
    (id/count columns), `masked_cols` skipped (timing), the rest canary-gated."""
    assert got.shape == want.shape, f"{label}: shape mismatch {got.shape} != {want.shape}"
    rows, cols = got.shape
    for r in range(rows):
        for c in range(cols):
            if c in masked_cols:
                continue
            g = float(got[r, c])
            w = float(want[r, c])
            if c in strict_cols:
                assert g == w, f"{label}[{r},{c}] strict: got {g} want {w}"
            else:
                assert_f64_close(g, w, f"{label}[{r},{c}]")


def _mat_var(mat_path: Path, name: str) -> NDArray[np.float64]:
    mat = scipy.io.loadmat(mat_path)
    return cast(NDArray[np.float64], mat[name]).astype("<f8")


def _ulp_diff(a: float, b: float) -> int:
    """Mirrors the Rust `ulp_diff`: bit-identical -> 0; opposite sign -> huge
    (always beyond any tolerance); else the signed-bit-pattern distance."""
    a_bits = int(np.float64(a).view(np.uint64))
    b_bits = int(np.float64(b).view(np.uint64))
    if a_bits == b_bits:
        return 0
    ai = int(np.float64(a).view(np.int64))
    bi = int(np.float64(b).view(np.int64))
    if (ai < 0) != (bi < 0):
        return 1 << 62
    return abs(ai - bi)


def _assert_gradcheck_report(report: dict[str, Any], golden: NDArray[np.float64], label: str, is_twin: bool = False) -> None:
    """Mirrors `phase4b_corpus_lid.rs::assert_gradcheck`'s MEASURED tolerances
    (NOT the libm-canary standard): backprop column <= 16 ULP, numerical column
    <= 1e-8 absolute, means <= 1e-8 / 1e-5 absolute. For tier2 (is_twin=False),
    mean_relative_error < 5e-4; for twin (is_twin=True), mean_relative_error < 1e-6
    (the Task-9 measured bound: src/rust/tests/phase4b_corpus_lid.rs:460-464)."""
    per_weight = cast(NDArray[np.float64], report["per_weight"])
    sweep = golden.shape[0] - 1
    assert per_weight.shape == (sweep, 3), f"{label}: sweep length"
    for k in range(sweep):
        want_back = float(golden[k, 0] / golden[k, 1])
        got_back = float(per_weight[k, 0])
        assert _ulp_diff(got_back, want_back) <= 16, f"{label} weight {k} backprop: got {got_back!r} want {want_back!r} (> 16 ULP)"
        want_num = float(golden[k, 2])
        got_num = float(per_weight[k, 1])
        assert abs(got_num - want_num) <= 1e-8, f"{label} weight {k} numerical: got {got_num!r} want {want_num!r} (> 1e-8 abs)"
    mean_error = float(cast(float, report["mean_error"]))
    mean_rel = float(cast(float, report["mean_relative_error"]))
    assert abs(mean_error - float(golden[sweep, 0])) <= 1e-8, f"{label}: mean_error mismatch"
    assert abs(mean_rel - float(golden[sweep, 1])) <= 1e-5, f"{label}: mean_relative_error mismatch"
    if is_twin:
        # Task-9 measured bound: analytic/numerical agreement is tight at ~1e-9 scale.
        # See phase4b_corpus_lid.rs:460-464 for the Rust golden assertion.
        assert mean_rel < 1e-6, f"{label}: mean_relative_error {mean_rel} not at the measured ~1e-9 scale"
    else:
        assert mean_rel < 5e-4, f"{label}: mean_relative_error {mean_rel} exceeds the 5e-4 analytic-vs-numeric bound"
    assert any(abs(float(t)) > 1e-12 for t in per_weight[:, 1]), f"{label}: all numerical derivatives ~0 (degenerate check)"


# ==== Task 4 brief: the four named replay tests ==============================


def test_tier2_spectral_train_replay(tmp_path: Path) -> None:
    """Algo 3 (spectral), 2-file corpus, 3 training epochs (Multi mode): the
    final weights and the 5 `.mat` result variables must match the committed
    tier-2 train fixtures (timing column masked, id columns strict)."""
    _seed_tier2_spectral(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["tier2_spectral.config"], "-m")
        eng.run()
        weights = eng.weights(0)
        assert len(weights) == 1, "algo 3 -> a single flat net"
        _assert_weight_vector(weights[0], PHASE4A / "tier2_weights_epoch4.bin", "tier2 final weights")

        mat_path = Path("tier2_spectral.mat")
        got_mcr = _mat_var(mat_path, "MultiConfigResults")
        want_mcr = _load_bin_matrix(PHASE4A / "tier2_spectral_MultiConfigResults.bin")
        _assert_matrix(got_mcr, want_mcr, (0, 1, 2), (6,), "tier2 MultiConfigResults")
        for name in ("CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"):
            got = _mat_var(mat_path, name)
            want = _load_bin_matrix(PHASE4A / f"tier2_spectral_{name}.bin")
            _assert_matrix(got, want, (), (), f"tier2 {name}")

    manifest = cast(dict[str, Any], json.loads((PHASE4A / "manifest.json").read_text()))
    epoch_files = cast(list[str], manifest["tier2"]["train"]["epoch_weight_files"])
    assert epoch_files[-1] == "tier2_weights_epoch4.bin", "manifest confirms epoch4 is the final saveAndUpdate"


def test_twin_train_replay(tmp_path: Path) -> None:
    """Algo 6 (Twin SAD+LID), Mode 7 phSeq, 2-file corpus, 6 training epochs +
    solo + final (8 saveAndUpdate rows): BOTH nets' final weights and the 5
    `.mat` result variables must match the committed twin-train fixtures."""
    _seed_twin_train(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["twin_train.config"], "-m")
        eng.run()
        weights = eng.weights(0)
        assert len(weights) == 2, "algo 6 -> [sad, lid] pair"
        _assert_weight_vector(weights[0], PHASE4B / "twin_train_sad_epoch7.bin", "twin sad final weights")
        _assert_weight_vector(weights[1], PHASE4B / "twin_train_lid_epoch7.bin", "twin lid final weights")

        mat_path = Path("twin_train.mat")
        got_mcr = _mat_var(mat_path, "MultiConfigResults")
        want_mcr = _load_bin_matrix(PHASE4B / "twin_train_MultiConfigResults.bin")
        assert got_mcr.shape[1] == 23, "3 id cols + the 20-col algo-6 row"
        _assert_matrix(got_mcr, want_mcr, (0, 1, 2), (6,), "twin_train MultiConfigResults")
        for name in ("CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"):
            got = _mat_var(mat_path, name)
            want = _load_bin_matrix(PHASE4B / f"twin_train_{name}.bin")
            _assert_matrix(got, want, (), (), f"twin_train {name}")

    manifest = cast(dict[str, Any], json.loads((PHASE4B / "manifest.json").read_text()))
    epochs = cast(int, manifest["corpus_lid"]["measured"]["train"]["twin_train"]["epochs"])
    assert epochs == 8, "epoch indices 0..7, final = epoch7"


def test_grad_check_seam(tmp_path_factory: pytest.TempPathFactory) -> None:
    """`Engine.grad_check(epsilon, max_weights)` against two corpus-level
    gradCheck fixtures with DIFFERENT origin (see module docstring): tier2
    (phase4a, algo 4 synthetic net, harness-vs-port canary-gated) and twin
    (phase4b, algo 6, both nets, real-Eigen-vs-port measured-tolerance)."""
    # --- tier2: canary-gated (mirrors phase4a_gradcheck_golden.rs). ----------
    manifest_a = cast(dict[str, Any], json.loads((PHASE4A / "manifest.json").read_text()))
    gc_a = cast(dict[str, Any], manifest_a["tier2"]["gradcheck"])
    max_weights_a = int(cast(int, gc_a["gradcheck_max_weights"]))
    epsilon_a = float(cast(float, gc_a["epsilon"]))
    assert max_weights_a == 10, "manifest gradcheck_max_weights"

    dst_a = tmp_path_factory.mktemp("gc_tier2")
    _seed_tier2_gradcheck(dst_a)
    with chdir(dst_a):
        eng = speech_rs.Engine(["tier2_gradcheck.config"], "-m")
        _, cols, seed_flat = read_bin(PHASE4A / "tier2_gradcheck_seed.bin")
        assert cols == 1 and seed_flat.shape[0] == 137, "synthetic [2,2]+[4,1] net has 137 weights"
        expect_seed = np.array([((k * 11 + 3) % 97) / 97.0 - 0.5 for k in range(137)], dtype=np.float64)
        assert np.array_equal(seed_flat.view(np.uint64), expect_seed.view(np.uint64)), "seed fixture must be the phase3 synth_flat pattern"
        eng.set_weights(0, [seed_flat])
        reports = eng.grad_check(epsilon_a, max_weights_a)
        assert len(reports) == 1, "one backprop-active network"
        net_idx, report = reports[0]
        assert net_idx == 0

        golden = _load_bin_matrix(PHASE4A / "tier2_gradcheck.bin")
        assert golden.shape == (11, 3), "10 per-weight rows + the means row"
        per_weight = cast(NDArray[np.float64], report["per_weight"])
        assert per_weight.shape == (10, 3)
        for k in range(10):
            want_backprop = float(golden[k, 0] / golden[k, 1])
            assert_f64_close(float(per_weight[k, 0]), want_backprop, f"tier2 weight {k} backprop")
            assert_f64_close(float(per_weight[k, 1]), float(golden[k, 2]), f"tier2 weight {k} numerical")
        assert_f64_close(float(cast(float, report["mean_error"])), float(golden[10, 0]), "tier2 mean_error")
        mean_rel_a = float(cast(float, report["mean_relative_error"]))
        assert_f64_close(mean_rel_a, float(golden[10, 1]), "tier2 mean_relative_error")
        assert mean_rel_a < 5e-4, f"tier2: analytic vs numeric must agree to < 5e-4, got {mean_rel_a}"

    # --- twin: measured ULP/abs bound (mirrors phase4b_corpus_lid.rs). -------
    manifest_b = cast(dict[str, Any], json.loads((PHASE4B / "manifest.json").read_text()))
    measured_b = cast(dict[str, Any], manifest_b["corpus_lid"]["measured"])
    max_weights_b = int(cast(int, measured_b["gradcheck_max_weights"]))
    epsilon_b = float(cast(float, measured_b["epsilon"]))
    assert max_weights_b == 10, "manifest gradcheck_max_weights"

    dst_b = tmp_path_factory.mktemp("gc_twin")
    _seed_twin_gradcheck(dst_b)
    with chdir(dst_b):
        eng = speech_rs.Engine(["twin_gradcheck.config"], "-m")
        reports = eng.grad_check(epsilon_b, max_weights_b)
        assert len(reports) == 2, "both nets backprop-active -> two reports"
        assert reports[0][0] == 0, "network index 0 (SAD) first"
        assert reports[1][0] == 1, "network index 1 (LID) second"
        for net_idx, report in reports:
            golden = _load_bin_matrix(PHASE4B / f"twin_gradcheck_net{net_idx}.bin")
            _assert_gradcheck_report(report, golden, f"twin net {net_idx}", is_twin=True)

        n0 = [float(t) for t in cast(NDArray[np.float64], reports[0][1]["per_weight"])[:, 1]]
        n1 = [float(t) for t in cast(NDArray[np.float64], reports[1][1]["per_weight"])[:, 1]]
        assert n0 != n1, "net-0 vs net-1 numerical gradients must differ (the column switch is load-bearing)"


def test_same_seed_determinism(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Two full Engine runs on the SAME seeded config (twin_train, N=1) must
    produce bit-identical `results_matrix` + final weights -- pinning that the
    PyO3 seam itself adds no nondeterminism on top of the 4a N=1 guarantee.

    `results_matrix()` IS the pre-`.mat`-write `ResultsE` (the same array
    `MultiConfigResults` is written from), so it carries the SAME wall-clock
    timing column at index 6 (`assemble_scored_row`'s `time_per_hour`, offset
    by the 3 leading id columns) -- genuinely non-deterministic (real elapsed
    time per run) and NOT part of the seam's own determinism contract. This
    was caught RED on the first run of this test (column 6 differed, every
    other column matched bit-for-bit) -- masked here the same way every other
    `MultiConfigResults` comparison in this repo masks it."""
    TIMING_COL = 6
    results: list[NDArray[np.float64]] = []
    sad_weights: list[NDArray[np.float64]] = []
    lid_weights: list[NDArray[np.float64]] = []
    for i in range(2):
        dst = tmp_path_factory.mktemp(f"determinism_{i}")
        _seed_twin_train(dst)
        with chdir(dst):
            eng = speech_rs.Engine(["twin_train.config"], "-m")
            eng.run()
            results.append(np.array(eng.results_matrix(), dtype=np.float64, copy=True))
            w = eng.weights(0)
            sad_weights.append(np.array(w[0], dtype=np.float64, copy=True))
            lid_weights.append(np.array(w[1], dtype=np.float64, copy=True))

    assert results[0].shape == results[1].shape and results[0].shape[0] > 0
    masked = [r.copy() for r in results]
    for r in masked:
        r[:, TIMING_COL] = 0.0
    assert np.array_equal(masked[0].view(np.uint64), masked[1].view(np.uint64)), "results_matrix must be bit-identical across runs (timing column masked)"
    assert np.array_equal(sad_weights[0].view(np.uint64), sad_weights[1].view(np.uint64)), "SAD final weights must be bit-identical across runs"
    assert np.array_equal(lid_weights[0].view(np.uint64), lid_weights[1].view(np.uint64)), "LID final weights must be bit-identical across runs"


# ==== Task 9: engine.forward_backward over the seam ==========================


def _seed_tier2_single_epoch(dst: Path) -> None:
    """`_seed_tier2_spectral` + override the config's `Epochs 3` down to 1 (last-wins),
    so `Engine.run()` is a SINGLE forward-backward + deriv dump -- the gradient eval
    `forward_backward` needs, not the full 3-epoch train."""
    _seed_tier2_spectral(dst)
    with (dst / "tier2_spectral.config").open("a") as fh:
        fh.write("\nNeural_Networks_BackPropagation_Epochs 1\n")


def test_forward_backward_tier2_determinism(tmp_path_factory: pytest.TempPathFactory) -> None:
    """`engine.forward_backward` (ComputeGradient's contract) on the tier-2 spectral fixture:
    two evals of the SAME seed weights on FRESH engines (the coarse-seam rebuild-per-eval
    pattern) must return a bit-identical finite cost + a single-net gradient paired and
    shaped to the input weights. The FULL epoch-0->1 equivalence lands in T12's exit gate."""
    results: list[tuple[float, NDArray[np.float64]]] = []
    seed: NDArray[np.float64] | None = None
    for i in range(2):
        dst = tmp_path_factory.mktemp(f"fb_tier2_{i}")
        _seed_tier2_single_epoch(dst)
        with chdir(dst):
            eng = speech_rs.Engine(["tier2_spectral.config"], "-m")
            if seed is None:
                seed = np.array(eng.weights(0)[0], dtype=np.float64, copy=True)
            f, grads = forward_backward(eng, [seed], None)
            assert len(grads) == 1, "algo 3 -> a single [sad] gradient"
            assert grads[0].shape == seed.shape, "gradient must be paired + shaped to the input weights"
            assert math.isfinite(f), "cost must be finite"
            assert np.isfinite(grads[0]).all(), "gradient must be finite"
            results.append((f, np.array(grads[0], dtype=np.float64, copy=True)))

    assert results[0][0] == results[1][0], "forward_backward cost must be deterministic across fresh engines"
    assert np.array_equal(results[0][1].view(np.uint64), results[1][1].view(np.uint64)), "forward_backward gradient must be bit-identical across fresh engines"


def test_forward_backward_drives_smorms3(tmp_path: Path) -> None:
    """One `Smorms3` step wrapping `forward_backward` (the inner training move) stays finite
    and shape-stable -- the seam feeds the optimizer, not just the goldens."""
    _seed_tier2_single_epoch(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["tier2_spectral.config"], "-m")
        theta0 = [np.array(eng.weights(0)[0], dtype=np.float64, copy=True)]

        def f_df(theta: list[NDArray[np.float64]], _ec: int) -> tuple[float, list[NDArray[np.float64]], list[NDArray[np.float64]]]:
            cost, grads = forward_backward(eng, theta, None)
            return cost, grads, theta  # theta_out is a pass-through (SMORMS3 adds dtheta to it)

        opt = Smorms3(f_df, theta0)
        opt.optimization_step()
        assert opt.theta.shape == theta0[0].shape, "theta stays the flat net length"
        assert np.isfinite(opt.theta).all(), "one SMORMS3 step must keep theta finite"
        assert opt.hist_f_flat and math.isfinite(opt.hist_f_flat[-1]), "the recorded cost must be finite"


# ==== Task 3 minors folded in =================================================


def test_set_weights_roundtrip(tmp_path: Path) -> None:
    """`set_weights` then `weights` back through the PyO3 boundary must be a
    bit-exact round trip (pure copy in, pure copy out, no arithmetic) -- the
    boundary pin flagged as missing in the Task 3 report's Concerns section."""
    _seed_tier2_spectral(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["tier2_spectral.config"], "-m")
        n = eng.weights(0)[0].shape[0]
        assert n == 33671
        pattern = np.array([((k * 7 + 5) % 101) / 101.0 - 0.5 for k in range(n)], dtype=np.float64)
        eng.set_weights(0, [pattern])
        got = eng.weights(0)
        assert len(got) == 1
        assert np.array_equal(got[0].view(np.uint64), pattern.view(np.uint64)), "set_weights -> weights must be a bit-exact round trip"


def test_parse_legacy_config_binding() -> None:
    """`speech_rs.parse_legacy_config` (the Rust binding) must agree key-for-key
    (and in insertion order) with the Python mirror `config_bridge.parse_legacy_config`
    on the same last-wins/comment-skipping semantics."""
    text = "\n".join(
        [
            "# a comment line, skipped",
            "Algo_choice 3",
            "",
            "BLSTM_window 3.25",
            "Algo_choice 4",  # duplicate key: last-wins.
            "key_with_no_value",
            "  BLSTM_shift   0.1  ",
        ]
    )
    got = dict(speech_rs.parse_legacy_config(text))
    want = parse_legacy_config_py(text)
    assert got == want
    assert list(got.keys()) == list(want.keys()), "insertion order must match"
    assert got["Algo_choice"] == "4", "last-wins duplicate key"
    assert got["key_with_no_value"] == ""
