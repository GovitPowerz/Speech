"""Issue #21: the LID cell knob. `--cell`/`--direction` on `lid-features` and `lid-phseq`
target the LID net (the one that trains under the frozen-SAD contract) through the same
overlay the SAD arms use, under the `BLSTM_LID` prefix. Two legs, the Phase-9 pattern:

* THE PREFLIGHT PROBE, every (arm x cell x direction): one gradient fold at the from-scratch
  theta, no training. What it pins is the SEAM AGREEMENT the acceptance names: the LID pack
  `init_weights` seeds under the selected cell is the length the engine builds for
  `BLSTM_LID_Cell_Type`/`_Direction` (a mismatch is a `set_weights` error inside the fold),
  the SAD pack is the legacy LSTM's and byte-identical across the cells, the init cost is
  finite and interior, the LID gradient is nonzero and the SAD gradient structurally zero
  (Mode 7 never runs the SAD net). Seconds per leg.
* ONE TRAINED LEG per arm on `slstm / forward`: the two derived keys of a forward LID net
  (`BLSTM_LID_OutputNeuronNb 24,12`, `BLSTM_LID_window 0`) exercised end to end at the
  phase-6 LID gate recipe, the frozen-SAD contract asserted, the held-out argmax error
  beating the model's own init (and chance where the margin exists, `_CHANCE_MARGIN`),
  run-twice bit-identical on a short recipe.
  Records its gate row (`gate_record`) for the LID cell tables.

Corpus-gated + pyo3, local-only. Every recorded number is an exact-tree number, like every SAD
cell row before it; since #57 the trained slstm/forward leg ALSO scores its held-out slice on
the fast tree (`Inference_Path fast`, the plain-regime causal LID twin) and asserts argmax
identity per file and a LID error delta of exactly 0.0 -- the real-data companion of
`src/rust/tests/fast_twin_lid_matrix.rs`.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import CORPUS_ROOT, GateRecorder, requires_corpus

pytest.importorskip("speech_rs")

from speech.batching import read_listing  # noqa: E402
from speech.config_bridge import CELL_TYPES, DIRECTIONS, parse_legacy_config  # noqa: E402
from speech.drivers import baseline as B  # noqa: E402
from speech.drivers.spec import BaselineSpec  # noqa: E402
from speech.drivers.state import ModernTrainParams, RunState  # noqa: E402
from speech.drivers.test import _class_keys  # noqa: E402
from speech.drivers.train import _init_weights_from_scratch  # noqa: E402
from speech.engine import average_derivs  # noqa: E402
from speech.evaluate import read_scr_scores  # noqa: E402
from speech.fold_run import FoldRun  # noqa: E402

_CHANCE = 100.0 * (1.0 - 1.0 / 12.0)
_ARMS = ("lid-features", "lid-phseq")
_MATRIX = [(arm, cell, direction) for arm in _ARMS for cell in CELL_TYPES for direction in DIRECTIONS]
_IDS = [f"{arm}-{cell}-{direction}" for arm, cell, direction in _MATRIX]
# `Law::Log`'s saturated-forward constant (the Phase-9 preflight hazard).
_LOG_CLAMP = float(-np.log(1e-24))
# The Twin's SAD net on both LID arms: the legacy LSTM pack the committed TOMLs size (measured; the
# seam-free unit test pins its bytes identical across LID cells).
_SAD_LEN = 5807
# The LID pack length per (arm, cell, direction): what `init_weights` seeds under the selected
# cell AND what the engine's `set_weights` accepted inside the probe's fold (measured 2026-10-08).
# lid-features: LID input 23 (cep), lid-phseq: 38 (phSeq one-hot); both `lstm_neuron_nb *,24`,
# `output_neuron_nb 48,12` (24,12 forward).
_LID_LEN: dict[tuple[str, str, str], int] = {
    ("lid-features", "lstm", "bidirectional"): 10426,
    ("lid-features", "lstm", "forward"): 5242,
    ("lid-features", "slstm", "bidirectional"): 9850,
    ("lid-features", "slstm", "forward"): 4954,
    ("lid-features", "mamba", "bidirectional"): 14410,
    ("lid-features", "mamba", "forward"): 7234,
    ("lid-features", "cfc", "bidirectional"): 11578,
    ("lid-features", "cfc", "forward"): 5818,
    ("lid-features", "transformer", "bidirectional"): 13002,
    ("lid-features", "transformer", "forward"): 6530,
    ("lid-phseq", "lstm", "bidirectional"): 13336,
    ("lid-phseq", "lstm", "forward"): 6712,
    ("lid-phseq", "slstm", "bidirectional"): 12760,
    ("lid-phseq", "slstm", "forward"): 6424,
    ("lid-phseq", "mamba", "bidirectional"): 15160,
    ("lid-phseq", "mamba", "forward"): 7624,
    ("lid-phseq", "cfc", "bidirectional"): 12958,
    ("lid-phseq", "cfc", "forward"): 6523,
    ("lid-phseq", "transformer", "bidirectional"): 13752,
    ("lid-phseq", "transformer", "forward"): 6920,
}

# The phase-6 LID gate recipes, VERBATIM (`test_phase6_gates.py`), so the slstm/forward rows
# compare to the LSTM rows on the same split and budget.
_GATE: dict[str, dict[str, object]] = {
    "lid-features": dict(subset=36, valid_size=12, test_size=48, epochs=2, steps_per_epoch=25, patience=99, seed=0),
    "lid-phseq": dict(subset=16, valid_size=12, test_size=48, epochs=2, steps_per_epoch=12, patience=99, seed=0),
}
# The beat-chance margin per arm, MEASURED 2026-10-08 (seed 0, the recipes above, slstm/forward):
#   lid-features: trained 83.33% vs init 93.75% (chance 91.67%): +8.33 pt over chance, +10.42 over init.
#   lid-phseq:    trained 91.11% vs init 100.00%: +8.89 pt over init but only +0.56 over chance, the
#                 thin from-scratch phSeq regime RESULTS.md records for the LSTM row too (a 1-epoch cut
#                 went negative there). Pinned as measured: beat-init on both arms, beat-chance only
#                 where the margin exists; the phSeq row renders its number rather than a widened pin.
_CHANCE_MARGIN: dict[str, float | None] = {"lid-features": 2.0, "lid-phseq": None}
_DET: dict[str, dict[str, object]] = {
    "lid-features": dict(subset=24, valid_size=12, test_size=12, epochs=1, steps_per_epoch=6, patience=99, seed=0),
    "lid-phseq": dict(subset=12, valid_size=8, test_size=12, epochs=1, steps_per_epoch=6, patience=99, seed=0),
}


@dataclass
class _Rec:
    val_cost: float
    train_cost: float


@dataclass
class _Shim:
    checkpoint_dir: str
    history: list[_Rec] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = -1
    best_val_cost: float = float("nan")


def _probe(out: dict[str, float]) -> Callable[[RunState, int, ModernTrainParams], _Shim]:
    """A `run_baseline(_train_fn=...)` stub measuring the from-scratch init instead of
    training, through the loop's OWN seeded init and a gradient `FoldRun` on the base config
    (the Phase-9 probe). The seed packs stand in as the checkpoint so the held-out scoring
    that follows runs on the init (LID allocates at least one test file per language)."""

    def probe(state: RunState, seed: int, params: ModernTrainParams) -> _Shim:
        weights = _init_weights_from_scratch(state, params)
        res = FoldRun(state.base_config, Path(state.config_path).parent, backprop=True).run(weights)
        grads = [average_derivs(d) for d in res.derivs]
        out["sad_len"], out["lid_len"] = float(len(weights[0])), float(len(weights[1]))
        out["init_cost"] = float(res.nn_cost)
        out["sad_grad_linf"] = float(np.max(np.abs(grads[0])))
        out["lid_grad_l2"] = float(np.linalg.norm(grads[1]))
        ckpt = Path(state.out_dir) / "checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        for net in ("sad", "lid"):
            shutil.copyfile(Path(state.out_dir) / f"{net}_seed.bin", ckpt / f"best_{net}.bin")
        return _Shim(checkpoint_dir=str(ckpt), history=[_Rec(float(res.nn_cost), float(res.nn_cost))], best_val_cost=float(res.nn_cost))

    return probe


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize(("arm", "cell", "direction"), _MATRIX, ids=_IDS)
def test_lid_init_is_trainable(tmp_path: Path, arm: str, cell: str, direction: str) -> None:
    """Measured 2026-10-08, seed 0, subset 12 (one file per language), the LID net's cost being
    the 12-way softmax cross-entropy (uniform = ln 12 = 2.485):

    | arm          | cell / direction      | LID pack | init cost | LID grad L2 |
    |--------------|-----------------------|----------|-----------|-------------|
    | lid-features | lstm / bi, fwd        | 10426, 5242  | 2.532, 2.536 | 0.154, 0.141 |
    | lid-features | slstm / bi, fwd       | 9850, 4954   | 2.512, 2.524 | 0.133, 0.275 |
    | lid-features | mamba / bi, fwd       | 14410, 7234  | 2.915, 2.879 | 0.617, 0.500 |
    | lid-features | cfc / bi, fwd         | 11578, 5818  | 2.592, 2.558 | 0.290, 0.237 |
    | lid-features | transformer / bi, fwd | 13002, 6530  | 3.204, 3.052 | 1.501, 0.942 |
    | lid-phseq    | lstm / bi, fwd        | 13336, 6712  | 2.486, 2.485 | 0.146, 0.146 |
    | lid-phseq    | slstm / bi, fwd       | 12760, 6424  | 2.485, 2.496 | 0.133, 0.176 |
    | lid-phseq    | mamba / bi, fwd       | 15160, 7624  | 2.518, 2.525 | 0.205, 0.230 |
    | lid-phseq    | cfc / bi, fwd         | 12958, 6523  | 2.498, 2.494 | 0.195, 0.186 |
    | lid-phseq    | transformer / bi, fwd | 13752, 6920  | 2.724, 2.694 | 1.370, 1.776 |

    Every init sits at or just above uniform (worst 3.204, 5.8% of the log law's saturated
    constant 55.26; the bound below is 20%, 3.4x headroom over the worst) and every LID
    gradient is far from zero (worst 0.133 against a 1e-3 pin). The SAD gradient is exactly
    zero on all 20 legs: Mode 7 never runs that net."""
    out: dict[str, float] = {}
    spec = BaselineSpec(
        arm=arm,  # type: ignore[arg-type]
        corpus_root=CORPUS_ROOT,
        out_dir=tmp_path / "probe",
        subset=12,
        valid_size=0,
        test_size=0,
        epochs=1,
        steps_per_epoch=1,
        patience=99,
        seed=0,
        cell=cell,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
    )
    B.run_baseline(spec, _train_fn=_probe(out))
    base = (tmp_path / "probe" / "base.config").read_text()

    # The overlay landed under the LID prefix and nowhere else.
    assert (f"BLSTM_LID_Cell_Type {cell}" in base) == (cell != "lstm") and "BLSTM_Cell_Type" not in base
    assert (f"BLSTM_LID_Direction {direction}" in base) == (direction != "bidirectional") and "BLSTM_Direction" not in base
    # The SAD pack is the legacy LSTM's whatever the LID cell; the fold accepted both packs, so
    # the LID pack is the length the engine builds for the selected cell.
    assert out["sad_len"] == _SAD_LEN, f"{arm} {cell}/{direction}: SAD pack {out['sad_len']:.0f} != the legacy LSTM's {_SAD_LEN}"
    want = _LID_LEN[arm, cell, direction]
    assert out["lid_len"] == want, f"{arm} {cell}/{direction}: LID pack {out['lid_len']:.0f} != pinned {want}"
    # Finite, interior init cost; the LID net has a gradient, the frozen SAD net has none.
    assert np.isfinite(out["init_cost"]) and 0.0 < out["init_cost"] < 0.2 * _LOG_CLAMP, out
    assert out["lid_grad_l2"] > 1e-3, f"{arm} {cell}/{direction}: epoch-0 LID gradient is ~zero"
    assert out["sad_grad_linf"] == 0.0, f"{arm} {cell}/{direction}: the frozen SAD net carries a gradient"


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize("arm", _ARMS)
def test_lid_slstm_forward_trains_and_scores(tmp_path: Path, arm: str, gate_record: GateRecorder) -> None:
    spec = BaselineSpec(arm=arm, corpus_root=CORPUS_ROOT, out_dir=tmp_path / "run", cell="slstm", direction="forward", score_init=True, **_GATE[arm])  # type: ignore[arg-type]
    res = B.run_baseline(spec)

    base = (res.out_dir / "base.config").read_text()
    assert "BLSTM_LID_Cell_Type slstm" in base and "BLSTM_LID_Direction forward" in base
    assert "BLSTM_LID_OutputNeuronNb 24,12" in base and "BLSTM_LID_window 0" in base

    # The frozen-SAD contract: the SAD net never moves; the LID net does.
    seed_sad = (res.out_dir / "sad_seed.bin").read_bytes()
    assert (res.checkpoint_dir / "best_sad.bin").read_bytes() == seed_sad
    assert (res.checkpoint_dir / "best_lid.bin").read_bytes() != (res.out_dir / "lid_seed.bin").read_bytes()

    assert res.lid_error is not None and res.init_lid_error is not None and np.isfinite(res.lid_error)
    margin = _CHANCE_MARGIN[arm]
    if margin is not None:
        assert res.lid_error <= _CHANCE - margin, f"{arm} slstm/forward held-out error {res.lid_error:.2f}% must beat chance {_CHANCE:.2f}%"
    assert res.lid_error < res.init_lid_error - 2.0, f"{arm} slstm/forward trained {res.lid_error:.2f}% must beat init {res.init_lid_error:.2f}%"
    assert res.cavg is not None and 0.0 <= res.cavg <= 1.0
    assert res.scores_dir is not None and len(list(res.scores_dir.glob("*.scr"))) == res.n_test
    gate_record(res)

    # #57: the SAME trained pair scored on the fast tree (the plain-regime causal LID twin,
    # `BLSTM_LID_window 0`) over the SAME held-out listing -- argmax identity per file and a
    # LID error delta of exactly 0.0 (ADR-0003: the LID tier owns argmax zero-flips).
    arm_obj = B.ARMS[arm]  # type: ignore[index]
    assert isinstance(arm_obj, B.LidArm)
    test_name = f"{arm_obj.stem}_test.flst"
    test_records = read_listing(res.out_dir / test_name)
    cfg = parse_legacy_config((res.out_dir / "base.config").read_text())
    scored: dict[str, tuple[float | None, Path]] = {}
    for path, extra in (("score_exact", {}), ("score_fast", {"Inference_Path": "fast"})):
        eval_cfg = {**cfg, "fileslisting": test_name, **extra}
        state = RunState.from_parsed(eval_cfg, res.out_dir / "base.config", res.out_dir)
        packs = (res.checkpoint_dir / "best_sad.bin", res.checkpoint_dir / "best_lid.bin")
        err, _cavg, sdir = B._score_packs_on_test(state, *packs, tmp_path / path, test_records)
        scored[path] = (err, sdir)
    (err_exact, dir_exact), (err_fast, dir_fast) = scored["score_exact"], scored["score_fast"]
    assert err_exact == res.lid_error, f"{arm}: the re-scored exact error must reproduce the run's"
    assert err_fast is not None and err_exact is not None and err_fast - err_exact == 0.0, f"{arm}: fast LID error {err_fast} vs exact {err_exact}"
    class_keys = _class_keys(res.out_dir / cfg["language2classmapping"])
    s_exact, f_exact = read_scr_scores(dir_exact, class_keys)
    s_fast, f_fast = read_scr_scores(dir_fast, class_keys)
    assert f_exact == f_fast and len(f_exact) == res.n_test
    flips = int(np.sum(np.argmax(s_exact, axis=1) != np.argmax(s_fast, axis=1)))
    assert flips == 0, f"{arm}: {flips} per-file argmax flips fast vs exact (R1 STOP)"


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize("arm", _ARMS)
def test_lid_slstm_forward_deterministic(tmp_path: Path, arm: str) -> None:
    kw = dict(arm=arm, corpus_root=CORPUS_ROOT, cell="slstm", direction="forward", **_DET[arm])
    a = B.run_baseline(BaselineSpec(out_dir=tmp_path / "a", **kw))  # type: ignore[arg-type]
    b = B.run_baseline(BaselineSpec(out_dir=tmp_path / "b", **kw))  # type: ignore[arg-type]
    for name in ("last_lid.bin", "best_lid.bin", "last_sad.bin", "best_sad.bin"):
        assert (a.checkpoint_dir / name).read_bytes() == (b.checkpoint_dir / name).read_bytes(), f"{arm}: {name} not bit-identical"
    assert a.lid_error == b.lid_error
