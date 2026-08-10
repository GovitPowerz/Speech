"""Phase 9 Task 5 + Phase 10 Task 3: the SEAM tier for the port-only cells (phase-9 spec
S8.1 SEAM / S8.3 / S1.3; phase-10 spec S9.2 for the CfC rows).

The phase-9 counterpart of `test_seam_replay.py`: everything here drives the committed
SYNTHETIC gate fixtures (`tests/reference_data/phase9/`, written by
`scripts/extract_phase9_fixtures.py`) through `speech_rs.Engine` in-process. No corpus, no
oracle, no gating -- CI's `python-pyo3` job runs the whole file.

PHASE 10 added three rows on exactly the phase-9 recipe -- `cfc_{bidirectional,forward}`
(SAD) and `twin_mode7_lid_cfc` (the LID mechanical wiring) -- plus one new structural leg
([`test_cfc_time_gate_saturation_selects_one_head`]). They also carry a COVERAGE CLOSURE the
CfC cell's own unit + FD tiers structurally cannot. Those tiers drive `CfcLayer` DIRECTLY,
and `phase9_cell_config.rs` reaches only the WEIGHT-SEAM arms of the enum
(`set_weights`/`get_weights`/`nb_of_weights`, via driver construction), so `CellLayer::Cfc`'s
FORWARD, BACKWARD and DERIVATIVE-HARVEST arms were compiled-but-never-executed: a mis-wired
one (`feed_forward` delegating to `feed_forward_reverse`, say) would have passed every test
in the repo. The Engine path here runs a real corpus forward AND backward THROUGH the enum,
and [`test_grad_check_seam`] is the leg that discriminates -- it compares the engine's own
central difference of the FORWARD against the analytic fold from the BACKWARD, so any arm
swap that desynchronizes the two fails it loudly. Verified by mutation, not asserted: each
of the two swaps above fails 5 legs (both `grad_check` rows, both block probes, the mode-7
Twin). The Task 3 report records them, plus the one class this is blind to -- a CONSISTENT
double swap of forward AND backward, which is self-consistent (the phase-9 battery lesson
verbatim: a leg comparing two runs of the same kernel cannot see inside it).

WHAT THE TIER PINS, and why each leg exists:

- **`grad_check`** (S8.1 SEAM): the engine's own corpus-level central difference vs its
  analytic fold, per cell x direction, plus both Twins. This is the leg the spec names. Its
  one weakness is that `max_weights` caps a PREFIX of the flat pack, and at these geometries
  the first 12 weights all live in ONE block (sLSTM `R_i`, mamba `P`, cfc `W_bb0`).

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
  row: sLSTM's `b_i` rows carry no gradient by construction (`nn/cells/slstm.rs`; residue
  only, see [`BIAS_I_RESIDUE_PIN`]), so a per-row nonzero assert would be wrong rather than
  strict.

THE ERROR METRIC ([`scaled_errors`]) is a scale-floored relative error, NOT the raw
`|a-b|/|b|` the engine's own `mean_relative_error` reports. These nets span ~5 decades of
gradient magnitude at init (sLSTM's swept `R_i` block runs from 4e-4 down to 4e-20), and a
raw relative error on a 1e-20 component divides FD ROUNDOFF by ~nothing. The floor is
`1e-3 * max|gradient| over the compared set`, i.e. "components below a thousandth of the
largest gradient are compared against that thousandth" -- the standard gradient-check
criterion, and the same near-zero partition (for the same reason) the unit tier introduced
in `src/rust/tests/phase9_cell_grad.rs`. ITS BLIND SPOT, stated plainly: a component whose
true derivative sits below ~1e-3 of the largest one can be 100% WRONG and still pass, since
the floor caps the denominator. What bounds that is the block-sum legs, which require every
named block to receive a nonzero gradient (and the two blocks that must NOT are named and
pinned dead), plus the unit tier, which pins the cells' backward per-weight. This tier's job
is to pin that the cell is WIRED correctly through the whole corpus stack, not to re-prove
the math.

MEASURE-THEN-PIN (spec R4): every tolerance below is `measured * 10` with the measured value
beside it, and every pin is < 1e-4. Measured on this box (M4 Pro / macOS 25.5 / Apple libm);
FD-vs-analytic AGREEMENT is a difference of two quantities computed through the same libm,
so it carries far less platform sensitivity than an absolute golden -- but a CI failure here
should be read as "re-measure", never as "widen".

The SAD fixture configs use SQUARE cost laws rather than the live `log` pair -- NOT because
`log` gives a wrong answer at these weights (with only the law names flipped, the committed
`mamba_bidirectional` reads cost 1.105 and a worst scaled error of 1.26e-8 with a textbook
U-curve), but because `log` stops being a valid gradcheck INSTRUMENT once the outputs
approach `Law::Log`'s clamps: the central difference loses its convergence plateau, and
further in the clamp-consistent derivative (F7) is exactly zero. `square` is smooth
everywhere on this corpus. See `scripts/extract_phase9_fixtures.py` and the Task 5 report
for the recipes, numbers, and the training-side (T8) consequence.
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
from speech.config_bridge import nnet_spec, parse_legacy_config
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
# The manifest carries the epsilons + `max_weights` every pin below was measured at, and it
# is the one committed file its own per-file digest table cannot cover. Pinned here instead
# (T5 review I3) so a manifest edit cannot silently move a tolerance's operating point.
MANIFEST_SHA256 = "0684c8a33cb3a70e5ba48c828c78bf155c70142ab7832f513a91de416faa915d"

# The port-only cells' SAD fixtures -- the ones this file's `grad_check` tier exists for.
# `lstm_forward` (phase-10 Task 8) is IN the manifest but deliberately NOT here: the
# legacy peephole LSTM's backward already has a real oracle (the phase-2/3 goldens against
# the compiled `LSTMLayer`), and that row was generated for the RUST fast-parity/streaming
# tiers. The whole-manifest legs below (digests, the gradcheck-reroute strip guard) still
# cover it. See `scripts/extract_phase9_fixtures.py`'s PROVENANCE note.
SAD_FIXTURES = ("slstm_bidirectional", "slstm_forward", "mamba_bidirectional", "mamba_forward", "cfc_bidirectional", "cfc_forward")
MODE7_TWINS = ("twin_mode7_lid_slstm", "twin_mode7_lid_cfc")
SAD_EPSILON = float(cast(float, MANIFEST["measured"]["sad_epsilon"]))
TWIN_EPSILON = float(cast(float, MANIFEST["measured"]["twin_epsilon"]))
TWIN_MODE7_EPSILON = float(cast(float, MANIFEST["measured"]["twin_mode7_epsilon"]))
TWIN_MODE7_CFC_EPSILON = float(cast(float, MANIFEST["measured"]["twin_mode7_cfc_epsilon"]))
MAX_WEIGHTS = int(cast(int, MANIFEST["measured"]["grad_check_max_weights"]))

# The scale floor, as a fraction of the largest gradient in the compared set (see the module
# docstring). Deliberately NOT a tunable per fixture: one number, stated once.
FLOOR_FRACTION = 1e-3

# sLSTM's `b_i` gradient is zero by construction (spec S2.1: shifting the input-gate bias
# scales the stabilizer's C and N equally, so h = sigmoid(o~) * C/N is invariant). The
# CANCELLATION IS ARITHMETIC, not a structural skip in the code, so what survives a corpus
# fold is floating-point residue, not a hard zero. MEASURED worst ratio of a `b_i` block sum
# to the pack's largest gradient, over all four sites that check it: 1.796e-16 (the mode-5
# Twin's LID net; slstm_bidirectional reads 1.033e-16). Pinned at measured*10.
BIAS_I_RESIDUE_PIN = 1.8e-15


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


def seed_twin_mode7(dst: Path, name: str = "twin_mode7_lid_slstm") -> None:
    """The Mode-7 phSeq Twin (the live phase-6 LID training regime), over the committed
    phase-4b phSeq corpus. `name` selects the LID cell (`..._slstm` / `..._cfc`) -- both
    fixtures share this corpus and the UNCHANGED phase-4b `tiny_sad_seed.bin`."""
    for f in ("languagemapping_lid7.csv", "tiny_sad_seed.bin"):
        shutil.copy(PHASE4B / f, dst / f)
    phseq = dst / "corpus_phseq"
    phseq.mkdir(parents=True, exist_ok=True)
    for f in ("s1.phSeq", "s2.phSeq", "s1.stm", "s2.stm", "listing_train.csv"):
        shutil.copy(PHASE4B / "corpus_phseq" / f, phseq / f)
    for f in (f"{name}.config", f"{name}_seed.bin"):
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


def cfc_blocks(out: int, backbone_units: int, backbone_layers: int, fin: int) -> dict[str, tuple[int, int]]:
    """One CfC layer, phase-10 spec S1.2 order: the backbone stack `W_bb{l} | b_bb{l}`
    (`l = 0` is `B x (in+H)`, the deeper ones `B x B`), then the three heads
    `W_ff1 | b_ff1 | W_ff2 | b_ff2 | W_g | b_g`, each `H x B` plus `H`.

    Transcribed from the spec, not from Rust and not from `init_weights` -- the same
    independence the two sibling maps above have, and what makes the block probe and
    [`test_cfc_time_gate_saturation_selects_one_head`] able to disagree with the engine."""
    b, h = backbone_units, out
    sizes: list[tuple[str, int]] = []
    for layer in range(backbone_layers):
        fan_in = fin + h if layer == 0 else b
        sizes += [(f"W_bb{layer}", b * fan_in), (f"b_bb{layer}", b)]
    for head in ("ff1", "ff2", "g"):
        sizes += [(f"W_{head}", h * b), (f"b_{head}", h)]
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
    elif info["cell_type"] == "cfc":
        layer = cfc_blocks(out, GEOMETRY["cfc_backbone_units"], GEOMETRY["cfc_backbone_layers"], fin)
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


def corpus_cost(eng: Any, net: int = 0) -> float:
    """The scalar `grad_check` differentiates for network `net`, recomputed from the seam.

    `CorpusProcessor::grad_check_cost` accumulates `row[cost] / row[counter]` over the
    per-file result rows, with the column pair switching on the algo AND the network index:
    algo 3/4 and algo-6 net 0 -> `row[4] / row[len-1]`; algo-6 net 1 (the LID net) ->
    `row[14] / row[len-2]`. `results_matrix` is those rows with 3 leading id columns
    (`[file+1, conf+1, chan+1, ...]`), so the cost column is `3 + 4` or `3 + 14` and the
    counter is `-1` or `-2`.

    That identification is cross-checked directly by
    `test_corpus_cost_columns_match_the_engine_gradcheck` (this helper's FD against
    `grad_check(eps, 1)`'s own numerical entry, both nets) as well as implicitly by the block
    probe agreeing with the analytic fold to ~1e-7.
    """
    eng.run()
    r = np.asarray(eng.results_matrix(), dtype=np.float64)
    cost_col, counter_col = (3 + 4, -1) if net == 0 else (3 + 14, -2)
    return float(r[:, cost_col].sum() / r[:, counter_col].sum())


def analytic_gradient(eng: Any, net: int = 0) -> NDArray[np.float64]:
    """The folded analytic gradient at the CURRENT weights, normalized exactly as
    `grad_check` normalizes it (`derivs[k,0] / derivs[k,1]`). Requires a prior `run()`."""
    dv = np.asarray(eng.weights_derivatives(0)[net], dtype=np.float64)
    assert dv.size > 0, "the seam returned an empty derivative matrix (backprop off for this net?)"
    assert float(np.abs(dv[:, 1]).min()) > 0.0, "the derivative frame counts (col 1) contain a zero -- the fold did not run for this net"
    return cast(NDArray[np.float64], dv[:, 0] / dv[:, 1])


def central_difference(eng: Any, packs: list[NDArray[np.float64]], net: int, index: int, eps: float) -> float:
    """Central difference of network `net`'s corpus cost w.r.t. its flat weight `index`.
    `packs` is the FULL per-network weight list the bag holds (one entry for algo 3, the
    `[sad, lid]` pair for algo 6) -- only `packs[net]` is perturbed."""
    plus = [p.copy() for p in packs]
    minus = [p.copy() for p in packs]
    plus[net][index] += eps
    minus[net][index] -= eps
    eng.set_weights(0, plus)
    cost_plus = corpus_cost(eng, net)
    eng.set_weights(0, minus)
    cost_minus = corpus_cost(eng, net)
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
    # The phase-10 CfC rows run at the SAME `sad_epsilon` rather than getting one of their
    # own, and that reuse is MEASURED, not assumed: a 5-point sweep (1e-7 .. 1e-3) reads
    #   bi   5.92e-6 / 1.70e-7 / 3.14e-7 / 3.13e-5 / 3.13e-3
    #   fwd  2.81e-6 / 3.41e-7 / 1.17e-8 / 1.17e-6 / 1.17e-4
    # i.e. a textbook U bottoming at 1e-6 (bi) and 1e-5 (fwd) with the right half the clean
    # `eps^2` truncation ramp. 1e-5 sits at or beside both minima, ~2 decades under the STOP.
    "cfc_bidirectional": (3.138e-7, 3.2e-6),
    "cfc_forward": (1.169e-8, 1.2e-7),
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
    "cfc_bidirectional": (1.417e-8, 1.5e-7),
    "cfc_forward": (1.027e-8, 1.1e-7),
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
        numerical = np.array([central_difference(eng, [pack], 0, blocks[label][0], SAD_EPSILON) for label in labels], dtype=np.float64)
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

    The CfC rows take NO exemption and that is the claim: phase-10 spec S1.3 predicts no
    dead block at `T > 1` (its one structural zero, `W_bb`'s state columns, is a `T = 1`
    artefact and this corpus is ~50 timesteps deep), so all eight CfC blocks go through the
    `> 0.0` branch. MEASURED: only `norm.*` comes back zero on either CfC fixture.
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


# `Logistic::fn`'s INCLUSIVE saturation bound (`nn/activations.rs::exp_limit`, `ln(f64::MAX)
# ~ 709.78`): a gate pre-activation at or beyond it returns EXACTLY 1.0 / 0.0, no rounding
# argument needed. 800 clears it with room and is nowhere near overflowing anything.
CFC_GATE_SATURATION = 800.0


@pytest.mark.parametrize("name", ("cfc_bidirectional", "cfc_forward"))
def test_cfc_time_gate_saturation_selects_one_head(name: str, tmp_path: Path) -> None:
    """Saturating CfC's time gate must make EXACTLY ONE of the two candidate heads inert,
    and WHICH one flips with the sign -- three block offsets, four decoded consequences.

    Why it discriminates (phase-10 spec S1.1: `h = ff1 (1 - g) + ff2 g`, `g = sigmoid(.)`):
    zero the `W_g` block and drive `b_g` past the logistic's saturation bound, and `g` is
    EXACTLY 1.0 (or 0.0) at every timestep, so `h` collapses to `ff2` (or `ff1`) bit-for-bit.
    Rewriting the collapsed-away head's weight AND bias blocks must then leave the corpus
    cost BIT-identical, while rewriting the surviving head's must move it -- and the whole
    picture must MIRROR when the sign flips. That is what pins the three head blocks in the
    S1.2 order: a `ff1`/`ff2` swap flips which regime is inert (both asserts fail), and any
    permutation that puts something else where `b_g` is stops saturating the gate at all
    (the "dead" head stays live). The `norm`-tail and backbone blocks are pinned by the block
    probe instead; this leg is about the head triple.

    NOTE the bidirectional fixture needs BOTH stacks saturated -- the reverse half feeds the
    same output MLP, so the survivor-head liveness assert cannot be drowned by a live reverse
    stack.
    """
    blocks = fixture_blocks(name)
    pack = load_pack(name)
    stacks = ("fwd", "bwd") if any(label.startswith("bwd.") for label in blocks) else ("fwd",)
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

        for sign, survivor, collapsed in ((+1.0, "ff2", "ff1"), (-1.0, "ff1", "ff2")):
            saturated = pack
            for stack in stacks:
                saturated = rewrite(saturated, f"{stack}.W_g", 0.0)
                saturated = rewrite(saturated, f"{stack}.b_g", sign * CFC_GATE_SATURATION)
            gated_cost = cost_of(saturated)
            assert math.isfinite(gated_cost)
            for block in (f"fwd.W_{collapsed}", f"fwd.b_{collapsed}"):
                assert cost_of(rewrite(saturated, block, 0.3)) == gated_cost, (
                    f"{name}: with the gate saturated to {survivor}, rewriting {block} moved the cost -- a head block offset is wrong"
                )
            assert cost_of(rewrite(saturated, f"fwd.W_{survivor}", 0.3)) != gated_cost, (
                f"{name}: with the gate saturated to {survivor}, that head is inert too -- the probe is vacuous"
            )

        live_cost = cost_of(pack)
        assert cost_of(rewrite(pack, "fwd.b_g", 1.5)) != live_cost, (
            f"{name}: the gate bias is inert at the committed weights -- the saturation leg proves nothing"
        )


# ==== S8.3: the Twin's LID mechanical wiring =================================
#
# FINDING (brief step 3): `tasks/lid.rs` needed NO change. `TwinBlstmSpectralLid::from_legacy`
# already builds its LID net through `BlstmConfig::from_legacy(map, "BLSTM_LID")`, which is
# the same T1 constructor that reads `{prefix}_Cell_Type` / `{prefix}_Direction` -- so
# `BLSTM_LID_Cell_Type slstm` dispatches with zero driver-side work. These tests pin that.
#
# TWO REGIMES, because neither covers the other:
#   * Mode 5 (wav) is the two-nets-live regime -- BOTH nets backprop-active, so `grad_check`
#     visits the legacy-LSTM SAD net AND the swapped sLSTM LID net, and the block probe can
#     finite-difference the LID net's own cost columns (`14 / len-2`).
#   * Mode 7 (phSeq) is the LIVE regime `speech baseline lid-phseq` trains: one-hot
#     `external_features` instead of wav, and a FROZEN SAD net, so `grad_check` visits the
#     LID net alone.
#
# The phase-10 CfC row (spec S9.2) takes MODE 7 ONLY -- the live regime, deliberately, not
# for want of a mode-5 clone: what the LID leg pins is the CTOR DISPATCH plus the flat-pack
# length agreement between `init_weights.py` and `CfcLayer::nb_of_weights`, and mode 5 would
# re-run the identical wiring against a second corpus. The two-nets-live gradient path is
# already pinned cell-agnostically by the sLSTM mode-5 rows below.
#
# A T5-review CORRECTION lives here: an earlier revision claimed Mode 5's
# `weights_derivatives` came back all-zero after a `run()` as a "pre-existing, mode-specific"
# property. It was not. The Mode-5 fixture had INHERITED
# `Neural_Networks_Gradient_Check_Epsilon 1e-5` from its phase-4b source, which routes
# `run()` into `grad_check_full` (`corpus_processor.rs:208`), whose epilogue restores
# `seam_derivs` -- so the seam read the reset accumulator. The "control" that appeared to
# confirm the diagnosis (the unmodified phase-4b config) carries the SAME key, so it
# confirmed the key, not the mode. The generator now strips the line, and the Mode-5 fold is
# live: the F10 + block-probe legs below are the ones that were wrongly thought impossible.

# MEASURED worst scaled error over the 12-weight sweep at eps 1e-4, pinned at measured*10.
TWIN_GRAD_CHECK_PINS = {0: (5.390e-9, 5.4e-8), 1: (3.585e-6, 3.6e-5)}
# Mode 7, per fixture: `(epsilon, measured, pin)`. Each epsilon is the MEASURED U-curve
# minimum for THAT net -- the sLSTM Twin's LID gradient is ~9e-7 (three decades below the
# mode-5 one, so its FD optimum is at a wide 1e-3), the phase-10 CfC Twin's is ~5.6e-5, ~60x
# bigger, and its optimum accordingly sits one decade lower (sweep 1e-7 .. 1e-3:
# 2.04e-6 / 6.67e-8 / 1.74e-8 / 2.63e-9 / 3.29e-7).
TWIN_MODE7_ROWS = {
    "twin_mode7_lid_slstm": (TWIN_MODE7_EPSILON, 9.197e-7, 9.2e-6),
    "twin_mode7_lid_cfc": (TWIN_MODE7_CFC_EPSILON, 2.630e-9, 2.7e-8),
}


def mode7_lid_blocks(name: str) -> dict[str, tuple[int, int]]:
    """The Mode-7 Twin's LID-net block map: `BLSTM_LID_LSTMNeuronNb 12,6` with
    `BLSTM_LID_LSTMSubSampling 1` -> out 6, fan-in 12, in whichever cell's layout the
    fixture's `BLSTM_LID_Cell_Type` selects."""
    if FIXTURES[name]["lid_cell_type"] == "cfc":
        return cfc_blocks(6, GEOMETRY["cfc_backbone_units"], GEOMETRY["cfc_backbone_layers"], 12)
    return slstm_blocks(6, 12)


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
    """Determinism on the two-net bag, across independent Engines: the forward `results_matrix`
    AT THETA, BOTH nets' folded gradients at theta, and both `grad_check` sweeps.

    The forward is measured on a SEPARATE Engine from the sweep on purpose:
    `grad_check_capped` restores `processors` and `seam_derivs` at its epilogue but NOT
    `self.results`, so a `results_matrix()` read after a sweep reports the last `-eps`
    perturbation rather than theta. (Before the T5-review C1 fix this leg was worse than
    that: the fixture's inherited `Gradient_Check_Epsilon` made even the "forward" Engine's
    `run()` a full uncapped gradCheck, which is where ~4.7 s of this file's runtime went.)
    """
    timing_col = 6
    results: list[NDArray[np.float64]] = []
    gradients: list[list[NDArray[np.float64]]] = []
    sweeps: list[list[NDArray[np.float64]]] = []
    for i in range(2):
        dst = tmp_path_factory.mktemp(f"p9_twin_det_{i}")
        seed_twin(dst)
        with chdir(dst):
            forward = speech_rs.Engine(["twin_lid_slstm.config"], "-m")
            forward.run()
            r = np.array(forward.results_matrix(), dtype=np.float64, copy=True)
            r[:, timing_col] = 0.0
            results.append(r)
            gradients.append([np.array(analytic_gradient(forward, net), dtype=np.float64, copy=True) for net in (0, 1)])

            sweeper = speech_rs.Engine(["twin_lid_slstm.config"], "-m")
            sweeps.append(
                [
                    np.array(cast(NDArray[np.float64], rep["per_weight"]), dtype=np.float64, copy=True)
                    for _, rep in sweeper.grad_check(TWIN_EPSILON, MAX_WEIGHTS)
                ]
            )
    assert np.array_equal(results[0].view(np.uint64), results[1].view(np.uint64)), "twin results_matrix is not deterministic"
    for net in (0, 1):
        assert np.array_equal(gradients[0][net].view(np.uint64), gradients[1][net].view(np.uint64)), f"twin net {net} folded gradient is not deterministic"
        assert np.array_equal(sweeps[0][net].view(np.uint64), sweeps[1][net].view(np.uint64)), f"twin net {net} grad_check sweep is not deterministic"


# MEASURED worst scaled error across the mode-5 LID net's block map at eps 1e-4 (the same
# epsilon its grad_check leg uses), pinned at measured*10.
TWIN_LID_BLOCK_PROBE_PIN = (3.876e-7, 3.9e-6)


def test_twin_lid_gradient_is_finite_and_block_wise_alive(tmp_path: Path) -> None:
    """F10 regression class on the swapped sLSTM LID net in the MODE-5 (two-nets-live) regime
    -- the leg the pre-C1 fixture made look impossible (see the section comment). Block sums,
    never per row; `b_i` is output-invariant for the same sLSTM reason as on the SAD side.

    The legacy-LSTM SAD net's own fold is checked alive as an aggregate too: it is the control
    that says the bag folded BOTH nets, not just the swapped one.
    """
    lid_pack = load_pack("twin_lid_slstm")
    # `BLSTM_LID_LSTMNeuronNb 12,6` with `BLSTM_LID_LSTMSubSampling 1` -> out 6, fan-in 12.
    blocks = slstm_blocks(6, 12)
    seed_twin(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["twin_lid_slstm.config"], "-m")
        eng.run()
        sad_gradient = analytic_gradient(eng, 0)
        gradient = analytic_gradient(eng, 1)
    assert np.isfinite(sad_gradient).all() and int(np.count_nonzero(sad_gradient)) > 0, (
        "the legacy-LSTM SAD net's fold is dead -- the bag did not fold both nets"
    )
    assert gradient.shape == lid_pack.shape
    assert np.isfinite(gradient).all(), "the LID gradient is not finite"
    assert int(np.count_nonzero(gradient)) > 0, "F10 regression -- the LID seam returned an all-zero gradient"
    scale = float(np.abs(gradient).max())
    for label, (start, n) in blocks.items():
        block_sum = float(np.abs(gradient[start : start + n]).sum())
        if label == "b_i":
            ratio = block_sum / scale
            assert ratio <= BIAS_I_RESIDUE_PIN, (
                f"mode-5 LID block {label}: sum |g| / scale = {ratio:.4e} > {BIAS_I_RESIDUE_PIN:.1e} -- b_i is no longer output-invariant"
            )
        else:
            assert block_sum > 0.0, f"mode-5 LID block {label}: sum |g| == 0, the block receives no gradient at all"


def test_twin_lid_block_probe_matches_finite_difference(tmp_path: Path) -> None:
    """The block probe on the SWAPPED LID net: one representative index per named sLSTM block,
    the folded analytic gradient vs a central difference of the LID net's OWN corpus cost
    (`row[14] / row[len-2]` -- the algo-6 net-1 column pair). Only possible because C1 made
    the mode-5 fold live; it is what carries per-block backward coverage for the LID cell,
    which `grad_check`'s 12-weight prefix (all inside `R_i`) cannot reach."""
    measured, pin = TWIN_LID_BLOCK_PROBE_PIN
    assert pin < 1e-4, "spec R4: a pin above 1e-4 is a STOP"
    blocks = slstm_blocks(6, 12)
    lid_pack = load_pack("twin_lid_slstm")
    sad_pack = read_weight_vector(PHASE4B / "tiny_sad_seed.bin")
    packs = [sad_pack, lid_pack]
    seed_twin(tmp_path)
    with chdir(tmp_path):
        eng = speech_rs.Engine(["twin_lid_slstm.config"], "-m")
        eng.run()
        analytic_all = analytic_gradient(eng, 1)
        labels = list(blocks)
        analytic = np.array([analytic_all[blocks[label][0]] for label in labels], dtype=np.float64)
        numerical = np.array([central_difference(eng, packs, 1, blocks[label][0], TWIN_EPSILON) for label in labels], dtype=np.float64)
    errors = scaled_errors(analytic, numerical)
    worst_index = int(errors.argmax())
    worst = float(errors[worst_index])
    detail = f"block {labels[worst_index]}: analytic {analytic[worst_index]!r} vs FD {numerical[worst_index]!r}"
    assert worst <= pin, f"mode-5 LID: worst scaled error {worst:.4e} > pin {pin:.1e} (measured {measured:.4e}) at {detail}"


@pytest.mark.parametrize("name", MODE7_TWINS)
def test_twin_mode7_lid_packs_load_and_roundtrip(name: str, tmp_path: Path) -> None:
    """S8.3 / phase-10 S9.2 step 1: the Twin BUILDS with the swapped LID cell beside its
    legacy-LSTM SAD net, loads BOTH committed packs bit-exactly, and round-trips both.

    The length check is the load-bearing half for a NEW cell: `BlstmNetwork::set_weights`
    consumes `nb_of_weights()` per layer head-first, so a Python builder that emitted the
    wrong CfC block sizes would either fail construction here or (worse) silently eat the
    output MLP's bytes -- which is exactly what `spec_directions`/`init_cfc_flat` agreeing
    with `CfcLayer::nb_of_weights` prevents."""
    info = cast(dict[str, Any], FIXTURES[name])
    lid_pack = load_pack(name)
    sad_pack = read_weight_vector(PHASE4B / "tiny_sad_seed.bin")
    assert lid_pack.shape[0] == info["lid_pack_length"]
    assert sad_pack.shape[0] == info["sad_pack_length"]
    seed_twin_mode7(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        weights = eng.weights(0)
        assert len(weights) == 2, "algo 6 -> [sad, lid]"
        assert np.array_equal(weights[0].view(np.uint64), sad_pack.view(np.uint64)), f"{name}: the SAD pack was not loaded bit-exactly"
        assert np.array_equal(weights[1].view(np.uint64), lid_pack.view(np.uint64)), f"{name}: the LID pack was not loaded bit-exactly"

        sad_pattern = np.array([((k * 5 + 1) % 89) / 89.0 - 0.5 for k in range(sad_pack.shape[0])], dtype=np.float64)
        lid_pattern = np.array([((k * 13 + 7) % 103) / 103.0 - 0.5 for k in range(lid_pack.shape[0])], dtype=np.float64)
        eng.set_weights(0, [sad_pattern, lid_pattern])
        got = eng.weights(0)
        assert np.array_equal(got[0].view(np.uint64), sad_pattern.view(np.uint64)), f"{name}: SAD round trip is not bit-exact"
        assert np.array_equal(got[1].view(np.uint64), lid_pattern.view(np.uint64)), f"{name}: LID round trip is not bit-exact"


@pytest.mark.parametrize("name", MODE7_TWINS)
def test_twin_mode7_lid_grad_check(name: str, tmp_path: Path) -> None:
    """S8.3 in the LIVE regime: the same cell swap under Mode 7 phSeq (what
    `speech baseline lid-phseq` trains). The Mode-7 contract freezes the SAD net, so
    `grad_check` visits the swapped LID net ALONE -- exactly one report, which is itself part
    of the pin (a second report would mean the frozen-SAD contract moved).

    THE PHASE-10 FINDING, pinned rather than assumed: `tasks/lid.rs` needed NO change for the
    CfC row either. `TwinBlstmSpectralLid::from_legacy` builds its LID net through
    `BlstmConfig::from_legacy(map, "BLSTM_LID")`, whose EXHAUSTIVE `match cell_type` gained
    the `Cfc` arm in T1 -- so a fourth cell is a config key and a seeded pack, nothing else.
    This leg is what turns that from a code reading into a measurement."""
    epsilon, measured, pin = TWIN_MODE7_ROWS[name]
    assert pin < 1e-4, "spec R4: a grad_check pin above 1e-4 is a STOP"
    seed_twin_mode7(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        reports = eng.grad_check(epsilon, MAX_WEIGHTS)
        assert len(reports) == 1 and reports[0][0] == 1, "Mode 7 freezes the SAD net -> the LID net (index 1) is the only backprop-active one"
        analytic, numerical = grad_check_columns(reports[0][1])
        assert float(np.abs(numerical).max()) > 1e-12, f"{name}: the whole numerical sweep is ~0"
        worst = float(scaled_errors(analytic, numerical).max())
        assert worst <= pin, f"{name} LID: worst scaled error {worst:.4e} > pin {pin:.1e} (measured {measured:.4e})"


@pytest.mark.parametrize("name", MODE7_TWINS)
def test_twin_mode7_lid_gradient_is_finite_and_block_wise_alive(name: str, tmp_path: Path) -> None:
    """F10 regression class on the swapped LID net, in the only Twin regime where the seam's
    folded gradient is readable (see the section comment). Block sums, never per row: sLSTM's
    `b_i` is output-invariant for the same reason as on the SAD side, and the CfC row takes
    no exemption at all (no dead block at `T > 1`)."""
    lid_pack = load_pack(name)
    blocks = mode7_lid_blocks(name)
    seed_twin_mode7(tmp_path, name)
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        eng.run()
        gradient = analytic_gradient(eng, 1)
    assert gradient.shape == lid_pack.shape
    assert np.isfinite(gradient).all(), f"{name}: the LID gradient is not finite"
    assert int(np.count_nonzero(gradient)) > 0, f"{name}: F10 regression -- the LID seam returned an all-zero gradient"
    scale = float(np.abs(gradient).max())
    for label, (start, n) in blocks.items():
        block_sum = float(np.abs(gradient[start : start + n]).sum())
        if label == "b_i":
            ratio = block_sum / scale
            assert ratio <= BIAS_I_RESIDUE_PIN, (
                f"{name} LID block {label}: sum |g| / scale = {ratio:.4e} > {BIAS_I_RESIDUE_PIN:.1e} -- b_i is no longer output-invariant"
            )
        else:
            assert block_sum > 0.0, f"{name} LID block {label}: sum |g| == 0, the block receives no gradient at all"


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
    by re-running `scripts/extract_phase9_fixtures.py`) fails here first.

    `manifest.json` cannot self-digest, and it is TOLERANCE-AFFECTING test input (the three
    epsilons and `max_weights` are read from it, and a changed epsilon silently invalidates
    every measured pin below). So its own sha256 is pinned HERE instead -- a T5-review item.
    Re-running the generator is what legitimately changes it; update this constant in the
    same commit, having RE-MEASURED the pins.
    """
    manifest_sha = hashlib.sha256((PHASE9 / "manifest.json").read_bytes()).hexdigest()
    assert manifest_sha == MANIFEST_SHA256, f"manifest.json changed (sha256 {manifest_sha}) -- re-measure the pins, then update MANIFEST_SHA256"

    digests = cast(dict[str, str], MANIFEST["files"])
    on_disk = {p.name for p in PHASE9.iterdir() if p.name != "manifest.json"}
    assert on_disk == set(digests), f"the phase9 fixture set drifted: {on_disk ^ set(digests)}"
    for name, want in digests.items():
        got = hashlib.sha256((PHASE9 / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: sha256 {got} != manifest {want}"


def test_fixture_configs_leave_mamba_dt_rank_at_the_auto_default() -> None:
    """[`mamba_blocks`] recomputes `Mamba_Dt_Rank 0`'s auto rule (`ceil(d_model/16)`)
    INDEPENDENTLY of Rust, so every block offset after `W_x` silently depends on the key
    being absent. Pin that it is -- a stated `Mamba_Dt_Rank` in a fixture would shift the
    whole tail of the block map with nothing else complaining.

    The three geometry keys the mamba fixtures DO state are pinned against the manifest at
    the same time, since `fixture_blocks` reads them from there rather than from the config.
    """
    for name in ("mamba_bidirectional", "mamba_forward"):
        text = (PHASE9 / f"{name}.config").read_text()
        assert "Mamba_Dt_Rank" not in text, f"{name}.config states Mamba_Dt_Rank -- mamba_blocks' auto-rule arithmetic no longer applies"
        for key, value in (
            ("Mamba_D_State", GEOMETRY["mamba_d_state"]),
            ("Mamba_D_Conv", GEOMETRY["mamba_d_conv"]),
            ("Mamba_Expand", GEOMETRY["mamba_expand"]),
        ):
            assert f"\n{key} {value}\n" in text, f"{name}.config: {key} does not match the manifest geometry ({value})"


def test_fixture_configs_state_the_cfc_geometry() -> None:
    """[`cfc_blocks`] sizes every block off the manifest geometry, so a fixture that dropped
    the `Cfc_*` keys would silently fall back to the SIZED phase-10 default (B=45) on BOTH
    sides and shift the whole block map with nothing else complaining. Pin that all three CfC
    fixtures state them, and that they match the manifest -- the `Mamba_Dt_Rank` precedent.
    """
    for name in ("cfc_bidirectional", "cfc_forward", "twin_mode7_lid_cfc"):
        text = (PHASE9 / f"{name}.config").read_text()
        for key, value in (
            ("Cfc_Backbone_Units", GEOMETRY["cfc_backbone_units"]),
            ("Cfc_Backbone_Layers", GEOMETRY["cfc_backbone_layers"]),
        ):
            assert f"\n{key} {value}\n" in text, f"{name}.config: {key} does not match the manifest geometry ({value})"


def test_no_fixture_config_reroutes_run_into_the_engine_gradcheck() -> None:
    """THE STRIP-LIST GUARD, asserted at PARSE level over every committed fixture (the T5
    review's C1 lesson, generalized so a future clone cannot re-import the hazard).

    Two keys decide what `Engine.run()` actually does, and every leg in this file assumes
    the same answer -- one forward+backward fold at theta:
      * `Neural_Networks_Gradient_Check_Epsilon` present -> `CorpusProcessor::run()` takes
        the `grad_check_full` branch, whose epilogue RESETS `seam_derivs`, so
        `weights_derivatives` reads an all-zero gradient and the F10/block-probe legs
        silently measure nothing.
      * `Neural_Networks_BackPropagation_Epochs != 0` -> `run()` is the engine-internal
        training loop, so the reported cost/gradient are at engine-MOVED weights, not theta
        (the phase-5 F11 convention).
    Parse level, not `grep`: the parser is last-wins, and the two Mode-7 Twins inherit an
    `Epochs 6` from `twin_train.config` that an appended `0` overrides -- a text scan would
    read whichever line it hit first.
    """
    for name in sorted(cast(dict[str, Any], MANIFEST["fixtures"])):
        flat = parse_legacy_config((PHASE9 / f"{name}.config").read_text())
        assert "Neural_Networks_Gradient_Check_Epsilon" not in flat, f"{name}.config carries a gradient-check epsilon -- run() is no longer a forward fold"
        assert flat.get("Neural_Networks_BackPropagation_Epochs") == "0", (
            f"{name}.config: effective Epochs is {flat.get('Neural_Networks_BackPropagation_Epochs')!r}, not 0"
        )


@pytest.mark.parametrize("name,net", [("slstm_bidirectional", 0), ("twin_lid_slstm", 0), ("twin_lid_slstm", 1)])
def test_corpus_cost_columns_match_the_engine_gradcheck(name: str, net: int, tmp_path: Path) -> None:
    """[`corpus_cost`]'s column identification, cross-checked against the ENGINE's own
    arithmetic rather than only against the analytic fold.

    `grad_check(eps, 1)` differentiates flat weight 0 of each backprop-active net using
    `grad_check_cost`'s internal column pair; this helper's own central difference of the
    same weight must reproduce that numerical entry. If `corpus_cost` read the wrong column
    (say the SAD pair for the LID net) the two would diverge grossly -- so this pins the
    `4 / len-1` vs `14 / len-2` switch, which otherwise only shows up indirectly.
    """
    if name == "twin_lid_slstm":
        seed_twin(tmp_path)
        packs = [read_weight_vector(PHASE4B / "tiny_sad_seed.bin"), load_pack(name)]
        eps = TWIN_EPSILON
    else:
        seed_sad(tmp_path, name)
        packs = [load_pack(name)]
        eps = SAD_EPSILON
    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        engine_numerical = {idx: float(grad_check_columns(rep)[1][0]) for idx, rep in eng.grad_check(eps, 1)}
        assert net in engine_numerical, f"{name}: network {net} is not backprop-active"
        mine = central_difference(eng, packs, net, 0, eps)
    want = engine_numerical[net]
    assert abs(want) > 1e-12, f"{name} net {net}: the engine's own numerical derivative is ~0 (degenerate cross-check)"
    assert mine == want, f"{name} net {net}: corpus_cost FD {mine!r} != the engine's grad_check numerical {want!r} -- wrong cost/counter columns"


# ==== Phase 10 Task 9 (spec S7): the inference-only retention gating =========
#
# T9 gates the two backward-only retentions (`MambaCache`, `Network::layers_output`) on
# `BackPropagationActivated`, decided ONCE at `BlstmNetwork::from_config`. These two legs
# are the seam-level halves of that claim: the gradient still flows where a backward can
# run, and the forward is untouched where one cannot.


@pytest.mark.parametrize("name", ("mamba_bidirectional", "mamba_forward"))
def test_epochs_zero_with_backprop_on_still_returns_a_real_gradient(name: str, tmp_path: Path) -> None:
    """THE F11 REGRESSION LEG (phase-10 spec S7).

    `Epochs 0` + `BackPropagationActivated true` is the modern training loop's seam shape --
    one forward+backward fold at theta, backprop harvested, no engine-internal weight move
    (phase-5 F11, `drivers/train.py::_modern_config_text`). It is EXACTLY the configuration a
    naive reading of "inference-only" would mistake for inference and gate the retention off,
    which would silently zero every gradient the Python SMORMS3 loop reads -- the same class
    of failure F10 and F11 already cost this repo twice.

    The condition therefore keys off `BackPropagationActivated` ALONE, and this leg pins the
    consequence on the two MAMBA fixtures (the cell whose cache the gating drops): `Epochs 0`
    written EXPLICITLY into the config text, gradient finite and non-zero. Under the inverted
    gating (retain in training / skip in inference, S9.4 mutation 7) the mamba backward
    asserts and this leg fails loudly rather than reporting zeros.
    """
    pack = load_pack(name)
    seed_sad(tmp_path, name)
    cfg = tmp_path / f"{name}.config"
    flat = parse_legacy_config(cfg.read_text())
    assert flat["BLSTM_BackPropagationActivated"] == "true", f"{name}: this leg needs a backprop-ON config"
    # Last-wins parser: appending re-states Epochs 0 in this config's own text, so the leg
    # does not lean on the fixture generator having written it.
    cfg.write_text(cfg.read_text() + "Neural_Networks_BackPropagation_Epochs 0\n")

    with chdir(tmp_path):
        eng = speech_rs.Engine([f"{name}.config"], "-m")
        eng.set_weights(0, [pack])
        eng.run()
        dv = np.asarray(eng.weights_derivatives(0)[0], dtype=np.float64)

    assert dv.size > 0, f"{name}: the seam returned an EMPTY gradient under Epochs 0 + backprop on"
    assert np.isfinite(dv).all(), f"{name}: the folded gradient is not finite"
    assert int(np.count_nonzero(dv[:, 0])) > 0, f"{name}: F11 regression -- the retention gating zeroed the gradient"


@pytest.mark.parametrize("name", ("mamba_bidirectional", "mamba_forward"))
def test_turning_backprop_off_does_not_move_the_forward(name: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    """THE BEHAVIOUR-FREE CLAIM at the seam, on the corpus rather than on a synthetic layer.

    `BackPropagationActivated false` is the exact condition that switches the retention off,
    so this compares the two regimes' `results_matrix` BIT for bit (timing column masked, as
    every determinism leg here masks it). The forward cost/counter columns are computed from
    the same posteriors either way -- the backward runs BEFORE the cost block and touches
    neither -- so any difference here would mean the retention gating changed a number, which
    is precisely what spec S7 claims it cannot.

    It also pins that an inference-only run COMPLETES: the two loud bails T9 adds
    (`MambaLayer::feed_backward`, `Network::feed_backward`) must be unreachable on a
    backprop-off corpus run, and a mis-scoped condition would surface here as a panic rather
    than as a wrong number.
    """
    timing_col = 6
    pack = load_pack(name)
    rows: list[NDArray[np.float64]] = []
    for i, backprop in enumerate(("true", "false")):
        dst = tmp_path_factory.mktemp(f"p10_t9_{name}_{i}")
        seed_sad(dst, name)
        cfg = dst / f"{name}.config"
        cfg.write_text(cfg.read_text() + f"BLSTM_BackPropagationActivated {backprop}\n")
        with chdir(dst):
            eng = speech_rs.Engine([f"{name}.config"], "-m")
            eng.set_weights(0, [pack])
            eng.run()
            r = np.array(eng.results_matrix(), dtype=np.float64, copy=True)
            r[:, timing_col] = 0.0
            rows.append(r)
            if backprop == "false":
                dv = np.asarray(eng.weights_derivatives(0)[0], dtype=np.float64)
                assert dv.size == 0, f"{name}: a backprop-off net reported a gradient ({dv.shape})"

    assert rows[0].shape == rows[1].shape and rows[0].shape[0] > 0
    assert np.array_equal(rows[0].view(np.uint64), rows[1].view(np.uint64)), (
        f"{name}: the retention gating moved the forward -- results_matrix differs between backprop on and off"
    )
