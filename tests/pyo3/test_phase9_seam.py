"""Phase 9 Task 5: the SEAM tier for the new cells (spec S8.1 SEAM, S8.3, S1.3).

The phase-9 counterpart of `test_seam_replay.py`: everything here drives the committed
SYNTHETIC phase-9 gate fixtures (`tests/reference_data/phase9/`, written by
`scripts/extract_phase9_fixtures.py`) through `speech_rs.Engine` in-process. No corpus, no
oracle, no gating -- CI's `python-pyo3` job runs the whole file.

WHAT THE TIER PINS, and why each leg exists:

- **`grad_check`** (S8.1 SEAM): the engine's own corpus-level central difference vs its
  analytic fold, per cell x direction, plus both Twins. This is the leg the spec names. Its
  one weakness is that `max_weights` caps a PREFIX of the flat pack, and at these geometries
  the first 12 weights all live in ONE block (sLSTM `R_i`, mamba `P`).

- **block probe** (this task's addition, covering that weakness): for EVERY named block of
  each cell's flat layout (spec S2.2 / S3.2) it finite-differences one representative index
  against `weights_derivatives`. Two things fall out at once -- the backward is checked
  block by block instead of prefix-deep, and the Nx2 derivative row order is pinned against
  the `set_weights` order by physical measurement (a permuted harvest would report block
  `j`'s derivative at block `k`'s index and the FD at `k` would disagree).

- **structural legs** (`..._is_output_inert`, `..._gates_the_recurrent_blocks`): the
  round-trip `set_weights(p) -> weights() == p` is PERMUTATION-BLIND (the getter mirrors the
  setter through the same walk, so any self-consistent permutation passes it). These two
  legs are not: they compute a block's offset HERE, from the spec arithmetic, and assert a
  DECODED behavioural consequence of that block landing exactly there -- sLSTM's `b_i` is
  output-invariant while `b_f` is not; zeroing mamba's `W_out` makes the whole recurrent
  half of its pack inert while `P` stays live. Move a block and they fail.

- **F10 regression class**: `weights_derivatives` after `run()` must be finite and NOT the
  all-zero pre-F10 gradient. Asserted on BLOCK SUMS / whole-vector aggregates, NEVER per
  row: sLSTM's `b_i` rows are EXACTLY zero by construction (`nn/cells/slstm.rs`), so a
  per-row nonzero assert would be wrong rather than strict.

THE ERROR METRIC ([`scaled_errors`]) is a scale-floored relative error, NOT the raw
`|a-b|/|b|` the engine's own `mean_relative_error` reports. These nets span ~5 decades of
gradient magnitude at init (sLSTM's swept `R_i` block runs from 4e-4 down to 4e-20), and a
raw relative error on a 1e-20 component divides FD ROUNDOFF by ~nothing. The floor is
`1e-3 * max|gradient| over the compared set`, i.e. "components below a thousandth of the
largest gradient are compared against that thousandth" -- the standard gradient-check
criterion, and the same near-zero partition (for the same reason) the unit tier introduced
in `src/rust/tests/phase9_cell_grad.rs`. The unit tier is what pins the cells' backward
per-weight; this tier pins that the cell is WIRED correctly through the whole corpus stack.

MEASURE-THEN-PIN (spec R4): every tolerance below is `measured * 10` with the measured value
beside it, and every pin is < 1e-4. Measured on this box (M4 Pro / macOS 25.5 / Apple libm);
FD-vs-analytic AGREEMENT is a difference of two quantities computed through the same libm,
so it carries far less platform sensitivity than an absolute golden -- but a CI failure here
should be read as "re-measure", never as "widen".

The SAD fixture configs use SQUARE cost laws, matching the phase-4b `twin_gradcheck.config`
precedent. Under the live `log` pair the corpus cost saturates and analytic-vs-FD agreement
collapses -- CELL-AGNOSTICALLY, reproduced on the legacy LSTM (see the generator docstring
for the numbers). A gradcheck fixture has to measure the cell backward, not that regime.
"""

from __future__ import annotations

import hashlib
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
from numpy.typing import NDArray
from speech.config_bridge import nnet_spec
from speech.init_weights import init_weights
from speech.weight_bridge import read_weight_vector, write_bin

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"
PHASE9 = REPO_ROOT / "tests" / "reference_data" / "phase9"
MANIFEST = cast(dict[str, Any], json.loads((PHASE9 / "manifest.json").read_text()))
FIXTURES = cast(dict[str, Any], MANIFEST["fixtures"])
GEOMETRY = cast(dict[str, int], MANIFEST["geometry"])

SAD_FIXTURES = ("slstm_bidirectional", "slstm_forward", "mamba_bidirectional", "mamba_forward")
SAD_EPSILON = float(cast(float, MANIFEST["measured"]["sad_epsilon"]))
TWIN_EPSILON = float(cast(float, MANIFEST["measured"]["twin_epsilon"]))
TWIN_MODE7_EPSILON = float(cast(float, MANIFEST["measured"]["twin_mode7_epsilon"]))
MAX_WEIGHTS = int(cast(int, MANIFEST["measured"]["grad_check_max_weights"]))

# The scale floor, as a fraction of the largest gradient in the compared set (see the module
# docstring). Deliberately NOT a tunable per fixture: one number, stated once.
FLOOR_FRACTION = 1e-3

# sLSTM's `b_i` gradient is zero by construction (spec S2.1: shifting the input-gate bias
# scales the stabilizer's C and N equally, so h = sigmoid(o~) * C/N is invariant). The
# CANCELLATION IS ARITHMETIC, not a structural skip in the code, so what survives a corpus
# fold is floating-point residue, not a hard zero. MEASURED worst ratio of a `b_i` block sum
# to the pack's largest gradient: 1.033e-16 (slstm_bidirectional); pinned at measured*10.
BIAS_I_RESIDUE_PIN = 1.1e-15


@contextmanager
def chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def scaled_errors(analytic: Any, numerical: Any) -> NDArray[np.float64]:
    """Per-element `|a - n| / max(|a|, |n|, FLOOR_FRACTION * max|.|)` -- see the module
    docstring for why the floor is there and what it costs."""
    a = np.asarray(analytic, dtype=np.float64)
    n = np.asarray(numerical, dtype=np.float64)
    floor = FLOOR_FRACTION * max(float(np.abs(a).max(initial=0.0)), float(np.abs(n).max(initial=0.0)))
    denominator = np.maximum(np.maximum(np.abs(a), np.abs(n)), floor)
    return cast(NDArray[np.float64], np.abs(a - n) / denominator)


# ==== Corpus seeding =========================================================


def seed_sad(dst: Path, name: str) -> None:
    """The phase-9 SAD gate config + its seeded pack, over the committed synthetic tier-2
    corpus (mirrors `test_seam_replay._seed_tier2_gradcheck`: 1 file, 2 s, wav + stm)."""
    corpus = dst / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for ext in ("wav", "stm"):
        shutil.copy(PHASE4A / "corpus" / f"f1.{ext}", corpus / f"f1.{ext}")
    for f in ("language2classmapping.csv", "tier2_gc_fileslisting.csv"):
        shutil.copy(PHASE4A / f, dst / f)
    for f in (f"{name}.config", f"{name}_seed.bin"):
        shutil.copy(PHASE9 / f, dst / f)


def seed_twin(dst: Path) -> None:
    """The Mode-5 wav Twin: phase-9 config + sLSTM LID pack over the committed phase-4b wav
    corpus and the UNCHANGED phase-4b `tiny_sad_seed.bin` (the SAD net stays legacy LSTM)."""
    for f in ("languagemapping_lid7.csv", "listing_gc_wav.csv", "tiny_sad_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    wav = dst / "corpus_lid"
    wav.mkdir(parents=True, exist_ok=True)
    for f in ("f1.wav", "f1.stm", "f1_gc.stm"):
        shutil.copy(PHASE4B / "corpus_lid" / f, wav / f)
    for f in ("twin_lid_slstm.config", "twin_lid_slstm_seed.bin"):
        shutil.copy(PHASE9 / f, dst / f)


def seed_twin_mode7(dst: Path) -> None:
    """The Mode-7 phSeq Twin (the live phase-6 LID training regime), over the committed
    phase-4b phSeq corpus."""
    for f in ("languagemapping_lid7.csv", "tiny_sad_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    phseq = dst / "corpus_phseq"
    phseq.mkdir(parents=True, exist_ok=True)
    for f in ("s1.phSeq", "s2.phSeq", "s1.stm", "s2.stm", "listing_train.csv"):
        shutil.copy(PHASE4B / "corpus_phseq" / f, phseq / f)
    for f in ("twin_mode7_lid_slstm.config", "twin_mode7_lid_slstm_seed.bin"):
        shutil.copy(PHASE9 / f, dst / f)


# ==== Flat-layout block maps (spec S2.2 / S3.2, stated INDEPENDENTLY here) ====
#
# Transcribed from the SPEC, not read back from Rust or from `init_weights`. That
# independence is the point: the block probe and the two structural legs assert that the
# engine agrees with this table, so a layout drift on either side fails loudly.


def slstm_blocks(out: int, fin: int) -> dict[str, tuple[int, int]]:
    """One sLSTM layer: gates `[i | f | o | z]`, each `[R (out x out) | W (out x fin) | b]`."""
    block = out * (out + fin + 1)
    blocks: dict[str, tuple[int, int]] = {}
    for g, gate in enumerate(("i", "f", "o", "z")):
        base = g * block
        blocks[f"R_{gate}"] = (base, out * out)
        blocks[f"W_{gate}"] = (base + out * out, out * fin)
        blocks[f"b_{gate}"] = (base + out * out + out * fin, out)
    return blocks


def mamba_blocks(out: int, fin: int, d_state: int, d_conv: int, expand: int) -> dict[str, tuple[int, int]]:
    """One Mamba layer, spec S3.2 order:
    `[P | p] (iff fin != out), g, W_in, conv | b_conv, W_x, W_dt | b_dt, A_log, D, W_out`."""
    d_inner = expand * out
    dt_rank = max(1, -(-out // 16))  # `Mamba_Dt_Rank 0` -> auto `ceil(d_model / 16)`
    sizes: list[tuple[str, int]] = []
    if fin != out:
        sizes += [("P", out * fin), ("p", out)]
    sizes += [
        ("g", out),
        ("W_in", 2 * d_inner * out),
        ("conv", d_inner * d_conv),
        ("b_conv", d_inner),
        ("W_x", (dt_rank + 2 * d_state) * d_inner),
        ("W_dt", d_inner * dt_rank),
        ("b_dt", d_inner),
        ("A_log", d_inner * d_state),
        ("D", d_inner),
        ("W_out", out * d_inner),
    ]
    blocks: dict[str, tuple[int, int]] = {}
    position = 0
    for label, n in sizes:
        blocks[label] = (position, n)
        position += n
    return blocks


def fixture_blocks(name: str) -> dict[str, tuple[int, int]]:
    """The FULL flat pack of a SAD fixture as named blocks, in `Blstm::set_weights` order:
    `fwd stack | bwd stack (bidirectional only) | output MLP | mean | std`."""
    info = cast(dict[str, Any], FIXTURES[name])
    out = GEOMETRY["sad_hidden"]
    fin = GEOMETRY["sad_input"] * GEOMETRY["sad_sub_sampling"]
    if info["cell_type"] == "slstm":
        layer = slstm_blocks(out, fin)
    else:
        layer = mamba_blocks(out, fin, GEOMETRY["mamba_d_state"], GEOMETRY["mamba_d_conv"], GEOMETRY["mamba_expand"])
    layer_len = sum(n for _, n in layer.values())

    blocks: dict[str, tuple[int, int]] = {}
    directions = ("fwd", "bwd") if info["direction"] == "bidirectional" else ("fwd",)
    for d, tag in enumerate(directions):
        for label, (start, n) in layer.items():
            blocks[f"{tag}.{label}"] = (d * layer_len + start, n)
    base = len(directions) * layer_len
    out_in = out * len(directions)
    blocks["out.W"] = (base, out_in)  # the single output layer: 1 neuron x out_in inputs
    blocks["out.b"] = (base + out_in, 1)
    blocks["norm.mean"] = (base + out_in + 1, GEOMETRY["sad_input"])
    blocks["norm.std"] = (base + out_in + 1 + GEOMETRY["sad_input"], GEOMETRY["sad_input"])
    assert blocks["norm.std"][0] + blocks["norm.std"][1] == info["pack_length"], f"{name}: the block map does not tile the pack"
    return blocks


# ==== Engine helpers =========================================================


def load_pack(name: str) -> NDArray[np.float64]:
    return read_weight_vector(PHASE9 / f"{name}_seed.bin")


def corpus_cost(eng: Any) -> float:
    """The scalar `grad_check` differentiates, recomputed from the seam.

    `CorpusProcessor::grad_check_cost` (algo 3/4) accumulates `row[4] / row[len-1]` over the
    per-file result rows; `results_matrix` is those rows with 3 leading id columns
    (`[file+1, conf+1, chan+1, ...]`), so the cost column is `3 + 4` and the counter is the
    last. That identification is not asserted anywhere directly -- it is VALIDATED by the
    block probe agreeing with the analytic fold to ~1e-7, which it could not do if this were
    the wrong column.
    """
    eng.run()
    r = np.asarray(eng.results_matrix(), dtype=np.float64)
    return float(r[:, 3 + 4].sum() / r[:, -1].sum())


def analytic_gradient(eng: Any, net: int = 0) -> NDArray[np.float64]:
    """The folded analytic gradient at the CURRENT weights, normalized exactly as
    `grad_check` normalizes it (`derivs[k,0] / derivs[k,1]`). Requires a prior `run()`."""
    dv = np.asarray(eng.weights_derivatives(0)[net], dtype=np.float64)
    assert dv.size > 0, "the seam returned an empty derivative matrix (backprop off for this net?)"
    assert float(np.abs(dv[:, 1]).min()) > 0.0, "the derivative frame counts (col 1) contain a zero -- the fold did not run for this net"
    return cast(NDArray[np.float64], dv[:, 0] / dv[:, 1])


def central_difference(eng: Any, pack: NDArray[np.float64], index: int, eps: float) -> float:
    plus, minus = pack.copy(), pack.copy()
    plus[index] += eps
    minus[index] -= eps
    eng.set_weights(0, [plus])
    cost_plus = corpus_cost(eng)
    eng.set_weights(0, [minus])
    cost_minus = corpus_cost(eng)
    return (cost_plus - cost_minus) / (2.0 * eps)


def grad_check_columns(report: dict[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`(analytic backprop column, numerical column)` of one `grad_check` report."""
    per_weight = np.asarray(cast(NDArray[np.float64], report["per_weight"]), dtype=np.float64)
    return per_weight[:, 0], per_weight[:, 1]


# ==== S8.1 SEAM tier: grad_check ============================================

# MEASURED worst scaled error over the 12-weight sweep at eps 1e-5, pinned at measured*10.
GRAD_CHECK_PINS = {
    "slstm_bidirectional": (6.792e-6, 6.8e-5),
    "slstm_forward": (3.197e-6, 3.2e-5),
    "mamba_bidirectional": (6.593e-9, 6.6e-8),
    "mamba_forward": (7.130e-9, 7.2e-8),
}


@pytest.mark.parametrize("name", SAD_FIXTURES)
def test_grad_check_seam(name: str, tmp_path: Path) -> None:
    """S8.1 SEAM tier: the corpus-level analytic fold vs the engine's own central difference,
    per cell x direction. NON-VACUITY is asserted on the numerical column as an aggregate
    (its largest magnitude), never per row."""
    measured, pin = GRAD_CHECK_PINS[name]
    assert pin < 1e-4, "spec R4: a grad_check pin above 1e-4 is a STOP, not a widening"
    seed_sad(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        reports = eng.grad_check(SAD_EPSILON, MAX_WEIGHTS)
        assert len(reports) == 1, "algo 3 -> one backprop-active network"
        net_index, report = reports[0]
        assert net_index == 0
        analytic, numerical = grad_check_columns(report)
        assert numerical.shape == (MAX_WEIGHTS,)
        assert float(np.abs(numerical).max()) > 1e-12, f"{name}: the whole numerical sweep is ~0 (degenerate check)"
        worst = float(scaled_errors(analytic, numerical).max())
        assert worst <= pin, f"{name}: worst scaled error {worst:.4e} > pin {pin:.1e} (measured {measured:.4e})"


# ==== The block probe: every named block, not just the swept prefix ==========

# MEASURED worst scaled error across the whole block map at eps 1e-5, pinned at measured*10.
BLOCK_PROBE_PINS = {
    "slstm_bidirectional": (1.258e-6, 1.3e-5),
    "slstm_forward": (8.754e-8, 8.8e-7),
    "mamba_bidirectional": (2.015e-7, 2.1e-6),
    "mamba_forward": (7.110e-8, 7.2e-7),
}


@pytest.mark.parametrize("name", SAD_FIXTURES)
def test_block_probe_matches_finite_difference(name: str, tmp_path: Path) -> None:
    """One representative index per NAMED flat-layout block, analytic vs central difference.

    Covers what `grad_check`'s prefix cap cannot reach (at these geometries its 12 weights
    all sit inside a single block) and, because the probe index comes from the block map
    stated in THIS file, it simultaneously pins the Nx2 derivative row order against the
    `set_weights` order: a permuted harvest reports block `j`'s derivative at block `k`'s
    index, and the physical FD at `k` disagrees.
    """
    measured, pin = BLOCK_PROBE_PINS[name]
    assert pin < 1e-4, "spec R4: a pin above 1e-4 is a STOP"
    blocks = fixture_blocks(name)
    pack = load_pack(name)
    seed_sad(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        eng.set_weights(0, [pack])
        corpus_cost(eng)
        analytic_all = analytic_gradient(eng)
        assert analytic_all.shape == pack.shape
        labels = list(blocks)
        analytic = np.array([analytic_all[blocks[label][0]] for label in labels], dtype=np.float64)
        numerical = np.array([central_difference(eng, pack, blocks[label][0], SAD_EPSILON) for label in labels], dtype=np.float64)
    errors = scaled_errors(analytic, numerical)
    worst_index = int(errors.argmax())
    worst = float(errors[worst_index])
    detail = f"block {labels[worst_index]} (index {blocks[labels[worst_index]][0]}): analytic {analytic[worst_index]!r} vs FD {numerical[worst_index]!r}"
    assert worst <= pin, f"{name}: worst scaled error {worst:.4e} > pin {pin:.1e} (measured {measured:.4e}) at {detail}"


# ==== S1.3 weight seam ======================================================


@pytest.mark.parametrize("name", SAD_FIXTURES)
def test_engine_loads_the_committed_pack_bit_exactly(name: str, tmp_path: Path) -> None:
    """The config's `BLSTM_weightsFile` is the committed seeded pack: the engine must accept
    its length (a wrong per-cell layout length fails construction loudly) and hold it
    bit-for-bit, which is the cross-language half of S1.3 -- `init_weights.py` emitted these
    bytes from the Rust `set_weights` order."""
    pack = load_pack(name)
    assert pack.shape[0] == FIXTURES[name]["pack_length"]
    seed_sad(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        weights = eng.weights(0)
        assert len(weights) == 1
        assert np.array_equal(weights[0].view(np.uint64), pack.view(np.uint64)), f"{name}: the engine did not load the pack bit-exactly"


@pytest.mark.parametrize("name", SAD_FIXTURES)
def test_set_weights_roundtrip_is_bit_exact(name: str, tmp_path: Path) -> None:
    """S1.3: `set_weights(p)` then `weights()` returns `p` bit-for-bit, per cell.

    NOTE this leg is PERMUTATION-BLIND on its own (the getter walks the same order the
    setter does) -- `test_slstm_input_gate_bias_block_is_output_inert`,
    `test_mamba_output_projection_gates_the_recurrent_blocks` and the block probe are the
    legs that are not.
    """
    n = int(FIXTURES[name]["pack_length"])
    pattern = np.array([((k * 13 + 7) % 103) / 103.0 - 0.5 for k in range(n)], dtype=np.float64)
    seed_sad(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        eng.set_weights(0, [pattern])
        got = eng.weights(0)
        assert len(got) == 1
        assert np.array_equal(got[0].view(np.uint64), pattern.view(np.uint64)), f"{name}: set_weights -> weights is not a bit-exact round trip"


# ==== F10 regression class + determinism ====================================


@pytest.mark.parametrize("name", SAD_FIXTURES)
def test_weights_derivatives_are_finite_and_block_wise_alive(name: str, tmp_path: Path) -> None:
    """F10 (phase 5): the release seam returns the FOLDED gradient, not the pre-F10 all-zero
    accumulator.

    Asserted on BLOCK SUMS, never per row. Two blocks are dead BY DESIGN and are pinned dead
    here rather than being allowed to hide inside a whole-vector nonzero check: sLSTM's `b_i`
    (invariant up to floating-point residue, see [`BIAS_I_RESIDUE_PIN`]) and the
    `norm.mean`/`norm.std` tail (exactly zero -- inert at `InputNormalizationType 0`).
    """
    blocks = fixture_blocks(name)
    pack = load_pack(name)
    seed_sad(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        eng.set_weights(0, [pack])
        corpus_cost(eng)
        analytic = analytic_gradient(eng)
    assert np.isfinite(analytic).all(), f"{name}: the folded gradient is not finite"
    assert int(np.count_nonzero(analytic)) > 0, f"{name}: F10 regression -- the seam returned an all-zero gradient"
    scale = float(np.abs(analytic).max())
    for label, (start, n) in blocks.items():
        block_sum = float(np.abs(analytic[start : start + n]).sum())
        if label.startswith("norm."):
            assert block_sum == 0.0, f"{name} block {label}: the normalize tail must be exactly inert, got sum |g| = {block_sum!r}"
        elif label.endswith(".b_i"):
            ratio = block_sum / scale
            assert ratio <= BIAS_I_RESIDUE_PIN, (
                f"{name} block {label}: sum |g| / scale = {ratio:.4e} > {BIAS_I_RESIDUE_PIN:.1e} -- b_i is no longer output-invariant"
            )
        else:
            assert block_sum > 0.0, f"{name} block {label}: sum |g| == 0, the block receives no gradient at all"


@pytest.mark.parametrize("name", SAD_FIXTURES)
def test_run_twice_is_bit_identical(name: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Two independent Engines on the same fixture produce bit-identical results + gradients
    (the timing column, genuinely wall-clock, is masked exactly as `test_seam_replay` masks
    it)."""
    timing_col = 6
    pack = load_pack(name)
    results: list[NDArray[np.float64]] = []
    gradients: list[NDArray[np.float64]] = []
    for i in range(2):
        dst = tmp_path_factory.mktemp(f"p9_det_{name}_{i}")
        seed_sad(dst, name)
        with chdir(dst):
            eng = speech_rs.Engine([f"{name}.config"], "-m")
            eng.set_weights(0, [pack])
            eng.run()
            r = np.array(eng.results_matrix(), dtype=np.float64, copy=True)
            r[:, timing_col] = 0.0
            results.append(r)
            gradients.append(np.array(analytic_gradient(eng), dtype=np.float64, copy=True))
    assert results[0].shape == results[1].shape and results[0].shape[0] > 0
    assert np.array_equal(results[0].view(np.uint64), results[1].view(np.uint64)), f"{name}: results_matrix is not deterministic"
    assert np.array_equal(gradients[0].view(np.uint64), gradients[1].view(np.uint64)), f"{name}: the folded gradient is not deterministic"


# ==== The two structural (NOT permutation-blind) legs ========================


@pytest.mark.parametrize("name", ("slstm_bidirectional", "slstm_forward"))
def test_slstm_input_gate_bias_block_is_output_inert(name: str, tmp_path: Path) -> None:
    """sLSTM's `b_i` block is output-INVARIANT and `b_f` is not -- both located by the block
    map stated in this file.

    Why it discriminates: shifting the input-gate bias scales the sLSTM stabilizer's `C` and
    `N` by the same factor, so `h = sigmoid(o~) * C/N` is unchanged (spec S2.1, proven in
    `nn/cells/slstm.rs`). No other block of this pack has that property. Overwrite `b_i` with
    a large constant and the corpus cost must be BIT-identical; do the same to `b_f` (the
    very next `out` entries) and it must move. Under any block-order permutation the slice
    called `b_i` here is some other parameter, and the bit-identity fails.
    """
    blocks = fixture_blocks(name)
    pack = load_pack(name)
    seed_sad(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")

        def cost_with(label: str, value: float) -> float:
            start, n = blocks[label]
            candidate = pack.copy()
            candidate[start : start + n] = value
            eng.set_weights(0, [candidate])
            return corpus_cost(eng)

        eng.set_weights(0, [pack])
        base = corpus_cost(eng)
        assert math.isfinite(base)
        assert cost_with("fwd.b_i", 3.5) == base, f"{name}: rewriting fwd.b_i moved the cost -- that slice is not the input-gate bias"
        assert cost_with("fwd.b_f", 3.5) != base, f"{name}: rewriting fwd.b_f left the cost identical -- the probe is vacuous"


@pytest.mark.parametrize("name", ("mamba_bidirectional", "mamba_forward"))
def test_mamba_output_projection_gates_the_recurrent_blocks(name: str, tmp_path: Path) -> None:
    """Zeroing mamba's `W_out` block must make the ENTIRE recurrent half of the layer inert
    while the `[P | p]` adapter stays live.

    Why it discriminates: with `W_out = 0` the layer collapses to its residual,
    `out_t = x'_t = P x_t + p` (spec S3.1), which depends on NO other block. So rewriting
    `A_log` / `W_x` / `conv` / `D` must leave the corpus cost BIT-identical, rewriting `P`
    must move it, and -- the non-vacuity half -- with `W_out` restored, rewriting `A_log`
    must move it again. Four decoded consequences of three block offsets; a permuted layout
    breaks at least one of them.
    """
    blocks = fixture_blocks(name)
    pack = load_pack(name)
    seed_sad(tmp_path, name)

    def rewrite(base_pack: NDArray[np.float64], label: str, value: float) -> NDArray[np.float64]:
        start, n = blocks[label]
        out = base_pack.copy()
        out[start : start + n] = value
        return out

    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")

        def cost_of(candidate: NDArray[np.float64]) -> float:
            eng.set_weights(0, [candidate])
            return corpus_cost(eng)

        gated = rewrite(pack, "fwd.W_out", 0.0)
        if "bwd.W_out" in blocks:
            gated = rewrite(gated, "bwd.W_out", 0.0)
        residual_cost = cost_of(gated)
        assert math.isfinite(residual_cost)

        for label in ("fwd.A_log", "fwd.W_x", "fwd.conv", "fwd.D"):
            assert cost_of(rewrite(gated, label, 0.25)) == residual_cost, f"{name}: with W_out zeroed, {label} still moved the cost -- a block offset is wrong"

        assert cost_of(rewrite(gated, "fwd.P", 0.05)) != residual_cost, f"{name}: with W_out zeroed, the residual adapter P is dead -- the probe is vacuous"

        live_cost = cost_of(pack)
        assert cost_of(rewrite(pack, "fwd.A_log", 0.25)) != live_cost, f"{name}: A_log is inert even with W_out live -- the gating leg proves nothing"


# ==== S8.3: the Twin's LID mechanical wiring =================================
#
# FINDING (brief step 3): `tasks/lid.rs` needed NO change. `TwinBlstmSpectralLid::from_legacy`
# already builds its LID net through `BlstmConfig::from_legacy(map, "BLSTM_LID")`, which is
# the same T1 constructor that reads `{prefix}_Cell_Type` / `{prefix}_Direction` -- so
# `BLSTM_LID_Cell_Type slstm` dispatches with zero driver-side work. These tests pin that.
#
# TWO fixtures, because neither covers the other:
#   * Mode 5 (wav) is the gradcheck regime -- BOTH nets backprop-active, so `grad_check`
#     visits the legacy-LSTM SAD net AND the swapped sLSTM LID net.
#   * Mode 7 (phSeq) is the LIVE regime `speech baseline lid-phseq` trains, and the only one
#     where the LID net's `weights_derivatives` fold is readable at the seam. On Mode 5 that
#     fold comes back all-zero after a `run()` -- PRE-EXISTING and cell-agnostic (reproduced
#     on the committed phase-4b `twin_gradcheck.config` with its legacy LSTM LID net), so
#     the F10 leg for the Twin lives on the Mode-7 fixture, not this one.

# MEASURED worst scaled error over the 12-weight sweep at eps 1e-4, pinned at measured*10.
TWIN_GRAD_CHECK_PINS = {0: (5.390e-9, 5.4e-8), 1: (3.585e-6, 3.6e-5)}
# Mode 7, MEASURED at its own eps 1e-3 (see the manifest note: its LID gradient is ~9e-7,
# three decades below the mode-5 one, so its FD optimum sits at a wider step).
TWIN_MODE7_GRAD_CHECK_PIN = (9.197e-7, 9.2e-6)


def test_twin_lid_cell_swap_constructs_and_roundtrips(tmp_path: Path) -> None:
    """S8.3: the Twin builds with an sLSTM LID net beside its legacy-LSTM SAD net, loads both
    committed packs bit-exactly, and round-trips BOTH through `set_weights`/`weights`."""
    info = cast(dict[str, Any], FIXTURES["twin_lid_slstm"])
    lid_pack = load_pack("twin_lid_slstm")
    sad_pack = read_weight_vector(PHASE4B / "tiny_sad_seed.bin")
    assert lid_pack.shape[0] == info["lid_pack_length"]
    assert sad_pack.shape[0] == info["sad_pack_length"]
    seed_twin(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["twin_lid_slstm.config"], "-m")
        weights = eng.weights(0)
        assert len(weights) == 2, "algo 6 -> [sad, lid]"
        assert np.array_equal(weights[0].view(np.uint64), sad_pack.view(np.uint64)), "SAD net: the legacy pack was not loaded bit-exactly"
        assert np.array_equal(weights[1].view(np.uint64), lid_pack.view(np.uint64)), "LID net: the sLSTM pack was not loaded bit-exactly"

        sad_pattern = np.array([((k * 5 + 1) % 89) / 89.0 - 0.5 for k in range(sad_pack.shape[0])], dtype=np.float64)
        lid_pattern = np.array([((k * 13 + 7) % 103) / 103.0 - 0.5 for k in range(lid_pack.shape[0])], dtype=np.float64)
        eng.set_weights(0, [sad_pattern, lid_pattern])
        got = eng.weights(0)
        assert np.array_equal(got[0].view(np.uint64), sad_pattern.view(np.uint64)), "SAD round trip is not bit-exact"
        assert np.array_equal(got[1].view(np.uint64), lid_pattern.view(np.uint64)), "LID round trip is not bit-exact"


def test_twin_grad_check_both_nets(tmp_path: Path) -> None:
    """S8.3: `grad_check` on the cell-swapped Mode-5 Twin -- net 0 (legacy LSTM SAD, cost
    columns `4 / len-1`) and net 1 (sLSTM LID, columns `14 / len-2`) both agree with the
    central difference at their pinned tolerances. The SAD leg doubles as the control: a
    regression moving BOTH would be the shared fold, not the cell."""
    seed_twin(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["twin_lid_slstm.config"], "-m")
        reports = eng.grad_check(TWIN_EPSILON, MAX_WEIGHTS)
        assert len(reports) == 2, "both nets backprop-active -> two reports"
        assert [net for net, _ in reports] == [0, 1], "ascending network index"
        numericals = []
        for net_index, report in reports:
            measured, pin = TWIN_GRAD_CHECK_PINS[net_index]
            assert pin < 1e-4, "spec R4: a grad_check pin above 1e-4 is a STOP"
            analytic, numerical = grad_check_columns(report)
            assert float(np.abs(numerical).max()) > 1e-12, f"twin net {net_index}: the whole numerical sweep is ~0"
            worst = float(scaled_errors(analytic, numerical).max())
            assert worst <= pin, f"twin net {net_index}: worst scaled error {worst:.4e} > pin {pin:.1e} (measured {measured:.4e})"
            numericals.append(numerical)
        assert not np.array_equal(numericals[0], numericals[1]), "the two nets' numerical gradients must differ (the cost-column switch is load-bearing)"


def test_twin_run_twice_is_bit_identical(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Determinism on the two-net bag: results AND both `grad_check` reports bit-identical
    across independent Engines."""
    timing_col = 6
    results: list[NDArray[np.float64]] = []
    sweeps: list[list[NDArray[np.float64]]] = []
    for i in range(2):
        dst = tmp_path_factory.mktemp(f"p9_twin_det_{i}")
        seed_twin(dst)
        with chdir(dst):
            eng = speech_rs.Engine(["twin_lid_slstm.config"], "-m")
            sweeps.append(
                [np.array(cast(NDArray[np.float64], rep["per_weight"]), dtype=np.float64, copy=True) for _, rep in eng.grad_check(TWIN_EPSILON, MAX_WEIGHTS)]
            )
            eng2 = speech_rs.Engine(["twin_lid_slstm.config"], "-m")
            eng2.run()
            r = np.array(eng2.results_matrix(), dtype=np.float64, copy=True)
            r[:, timing_col] = 0.0
            results.append(r)
    assert np.array_equal(results[0].view(np.uint64), results[1].view(np.uint64)), "twin results_matrix is not deterministic"
    for net in (0, 1):
        assert np.array_equal(sweeps[0][net].view(np.uint64), sweeps[1][net].view(np.uint64)), f"twin net {net} grad_check sweep is not deterministic"


def test_twin_mode7_lid_grad_check(tmp_path: Path) -> None:
    """S8.3 in the LIVE regime: the same cell swap under Mode 7 phSeq (what
    `speech baseline lid-phseq` trains). The Mode-7 contract freezes the SAD net, so
    `grad_check` visits the sLSTM LID net ALONE -- exactly one report, which is itself part
    of the pin (a second report would mean the frozen-SAD contract moved)."""
    measured, pin = TWIN_MODE7_GRAD_CHECK_PIN
    assert pin < 1e-4, "spec R4: a grad_check pin above 1e-4 is a STOP"
    seed_twin_mode7(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["twin_mode7_lid_slstm.config"], "-m")
        reports = eng.grad_check(TWIN_MODE7_EPSILON, MAX_WEIGHTS)
        assert len(reports) == 1 and reports[0][0] == 1, "Mode 7 freezes the SAD net -> the LID net (index 1) is the only backprop-active one"
        analytic, numerical = grad_check_columns(reports[0][1])
        assert float(np.abs(numerical).max()) > 1e-12, "mode-7 twin: the whole numerical sweep is ~0"
        worst = float(scaled_errors(analytic, numerical).max())
        assert worst <= pin, f"mode-7 twin LID: worst scaled error {worst:.4e} > pin {pin:.1e} (measured {measured:.4e})"


def test_twin_mode7_lid_gradient_is_finite_and_block_wise_alive(tmp_path: Path) -> None:
    """F10 regression class on the swapped LID net, in the only Twin regime where the seam's
    folded gradient is readable (see the section comment). Block sums, never per row: `b_i`
    is output-invariant for the same sLSTM reason as on the SAD side."""
    lid_pack = load_pack("twin_mode7_lid_slstm")
    # `BLSTM_LID_LSTMNeuronNb 12,6` with `BLSTM_LID_LSTMSubSampling 1` -> out 6, fan-in 12.
    blocks = slstm_blocks(6, 12)
    seed_twin_mode7(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["twin_mode7_lid_slstm.config"], "-m")
        eng.run()
        gradient = analytic_gradient(eng, 1)
    assert gradient.shape == lid_pack.shape
    assert np.isfinite(gradient).all(), "the LID gradient is not finite"
    assert int(np.count_nonzero(gradient)) > 0, "F10 regression -- the LID seam returned an all-zero gradient"
    scale = float(np.abs(gradient).max())
    for label, (start, n) in blocks.items():
        block_sum = float(np.abs(gradient[start : start + n]).sum())
        if label == "b_i":
            ratio = block_sum / scale
            assert ratio <= BIAS_I_RESIDUE_PIN, (
                f"LID block {label}: sum |g| / scale = {ratio:.4e} > {BIAS_I_RESIDUE_PIN:.1e} -- b_i is no longer output-invariant"
            )
        else:
            assert block_sum > 0.0, f"LID block {label}: sum |g| == 0, the block receives no gradient at all"


# ==== The real committed SAD config still defaults to the legacy shape =======


def test_real_lre_sad_config_defaults_to_lstm_bidirectional(tmp_path: Path) -> None:
    """The phase-9 keys are ADDITIVE: `configs/training/lre_sad.toml` (the real, committed
    phase-6 SAD arm config) states neither, so both sides must decode the legacy shape.

    This closes T4's default-identity pin, which used a synthetic stand-in. Asserted at
    THREE levels on the real file: the flattened key map carries no phase-9 key at all;
    `nnet_spec` reports `lstm`/`bidirectional`; and the engine BUILDS against the pack
    `init_weights` produces from that spec -- i.e. the default Python init still fits the
    default Rust architecture, byte for byte.

    Engine CONSTRUCTION (not `run`) is the level: `lre_sad.toml`'s `[corpus]` keys are
    runtime placeholders `drivers.baseline` rewrites, so the listing/mapping are staged here
    from the committed synthetic tier-2 corpus rather than anything corpus-derived.
    """
    flat = dict(speech_rs.load_toml_config(str(REPO_ROOT / "configs" / "training" / "lre_sad.toml")))
    assert "BLSTM_Cell_Type" not in flat, "lre_sad.toml gained a cell-type key -- this default-identity pin needs revisiting"
    assert "BLSTM_Direction" not in flat, "lre_sad.toml gained a direction key -- this default-identity pin needs revisiting"

    spec = nnet_spec(flat)
    assert spec["CellType"] == "lstm"
    assert spec["Direction"] == "bidirectional"
    pack = init_weights(spec, np.random.default_rng(920507), "xavier", True)[0]

    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for ext in ("wav", "stm"):
        shutil.copy(PHASE4A / "corpus" / f"f1.{ext}", corpus / f"f1.{ext}")
    shutil.copy(PHASE4A / "language2classmapping.csv", tmp_path / flat["language2classmapping"])
    shutil.copy(PHASE4A / "tier2_gc_fileslisting.csv", tmp_path / flat["fileslisting"])
    write_bin(pack.shape[0], 1, pack, tmp_path / flat["BLSTM_weightsFile"])
    (tmp_path / "lre_sad.config").write_text("".join(f"{k} {v}\n" for k, v in flat.items()))

    with chdir(tmp_path):
        eng = speech_rs.Engine(["lre_sad.config"], "-m")
        weights = eng.weights(0)
    assert len(weights) == 1
    assert weights[0].shape[0] == pack.shape[0], "the default Python init pack no longer matches the real config's engine-side length"
    assert np.array_equal(weights[0].view(np.uint64), pack.view(np.uint64)), "the real config's engine did not load the default-init pack bit-exactly"


# ==== Fixture integrity =====================================================


def test_committed_fixture_digests_match_the_manifest() -> None:
    """The generator records a sha256 per emitted file; a fixture edited by hand (rather than
    by re-running `scripts/extract_phase9_fixtures.py`) fails here first."""
    digests = cast(dict[str, str], MANIFEST["files"])
    on_disk = {p.name for p in PHASE9.iterdir() if p.name != "manifest.json"}
    assert on_disk == set(digests), f"the phase9 fixture set drifted: {on_disk ^ set(digests)}"
    for name, want in digests.items():
        got = hashlib.sha256((PHASE9 / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: sha256 {got} != manifest {want}"
