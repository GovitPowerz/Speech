"""Phase 6 Task 8: the LID FEATURES arm subset gate (corpus-gated, local-only, pyo3).

THE flagship end-to-end proof: a stratified seeded subset of the LRE03 `.plp8f0mvsdd` cep
features (File_Type 2) trains the 12-class Twin (Algo 6, Mode 7) FROM SCRATCH via
`train_modern` (val_metric=nn_cost_seg), then the trained model is scored on a disjoint
held-out slice end to end -- `drivers/test.py::evaluate` -> `.scr` -> `read_scr_scores` ->
`lid_error` + `cavg` (Task 5). These are the phase's FIRST real headline numbers.

WHY THE IMPROVEMENT SIGNAL IS ARGMAX ERROR, NOT THE CE VALIDATION COST (a Task-8 finding):
from scratch the softmax cross-entropy VALIDATION cost can RISE while the argmax accuracy
improves -- the net grows confident on the (majority-leaning) classes it learns first, so
its held-out CE overfits even as it starts getting languages right. So the honest,
direction-safe improvement here is the held-out LID argmax error: it must (a) beat the
12-way chance error (91.67%) with headroom and (b) beat the model's OWN untrained-init error
on the identical test set. The TRAIN cost descending is the separate machinery-works signal.
The net mode-collapses toward the majority classes on this tiny subset (an inherent property
of from-scratch 12-way LID on a few files per language, NOT a bug -- the full-corpus launcher
run is where genuine 12-way discrimination lands); the 48-file held-out set makes the
beat-chance count (measured 13 correct vs chance ~4) robust to that collapse and to cross-libm
descent jitter. Margins were MEASURED on this machine (2026-07-18) and PINNED with headroom.

Corpus-gated (`requires_corpus`) + pyo3 (`importorskip`): runs locally for implementers AND
reviewers, skips in CI. Measured whole-file runtime ~6 min on the dev box (main gate ~4.7 min
+ the short determinism run + the dry-run smoke), inside the < 10 min budget (spec R3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import CORPUS_ROOT, requires_corpus

pytest.importorskip("speech_rs")

from speech.drivers import baseline as B  # noqa: E402 -- after importorskip, matching the pyo3-suite convention

_CHANCE = 100.0 * (1.0 - 1.0 / 12.0)  # 12-way argmax chance error = 91.666...%


@pytest.mark.slow
@requires_corpus
def test_lid_features_subset_trains_and_scores(tmp_path: Path) -> None:
    """Train the 12-class Twin from scratch on a stratified cep-features subset, then score a
    disjoint 48-file held-out slice end to end. Pins (measured 2026-07-18, seed 0):
    lid_error 72.92%, init 93.75%, cavg 0.46, train_cost 2.368 -> 2.216; ~282 s."""
    res = B.run_baseline(
        "lid-features",
        CORPUS_ROOT,
        tmp_path / "run",
        subset=36,  # ~3 train files/language (per-language proportional, seeded)
        valid_size=12,
        test_size=48,  # ~4 held-out files/language -- a stable argmax-error denominator
        epochs=2,
        steps_per_epoch=25,  # past SMORMS3's ~6-step warm-up so each epoch genuinely descends
        patience=99,
        seed=0,
        score_init=True,  # also score the untrained init on the SAME test set (the improvement baseline)
    )

    # --- the machinery works: the SMORMS3 train objective descends across epochs ---
    assert len(res.train_costs) == 2, f"expected 2 epochs of history, got {res.train_costs}"
    assert res.train_costs[-1] < res.train_costs[0] - 0.02, f"train cost must descend (machinery), got {res.train_costs}"

    # --- the end-to-end held-out LID error is real, beats chance, and beats its own init ---
    assert res.lid_error is not None and res.init_lid_error is not None
    assert np.isfinite(res.lid_error)
    # (a) beats 12-way chance with headroom (measured 72.92%; pin <= 84% -> ~11 pt below measured,
    #     ~8 pt below chance -- survives cross-libm descent jitter on the 48-file test set).
    assert res.lid_error <= _CHANCE - 7.0, f"held-out LID error {res.lid_error:.2f}% must beat chance {_CHANCE:.2f}% with margin"
    # (b) beats the model's OWN untrained-init error on the identical test set (measured 20.8 pt gap).
    assert res.lid_error < res.init_lid_error - 5.0, f"trained {res.lid_error:.2f}% must beat init {res.init_lid_error:.2f}% by margin"

    # --- cavg RECORDED (a valid closed-set Cavg in [0, 1]); no hard pin (not stable at this scale) ---
    assert res.cavg is not None and 0.0 <= res.cavg <= 1.0, f"cavg must be a valid closed-set value, got {res.cavg}"

    # --- the run is self-describing: metadata + both checkpoints + the .scr scores landed ---
    assert res.metadata_path.is_file()
    assert (res.checkpoint_dir / "best_lid.bin").is_file() and (res.checkpoint_dir / "best_sad.bin").is_file()
    assert res.scores_dir is not None and len(list(res.scores_dir.glob("*.scr"))) == res.n_test


@pytest.mark.slow
@requires_corpus
def test_lid_features_deterministic(tmp_path: Path) -> None:
    """Run-twice determinism at a fixed seed: bit-identical trained weights + identical
    held-out error. A SHORT config (determinism is a pipeline property, provable cheaply);
    the full gate's determinism was measured separately (2026-07-18) and holds identically."""
    kw = dict(subset=24, valid_size=12, test_size=12, epochs=1, steps_per_epoch=6, patience=99, seed=0)
    a = B.run_baseline("lid-features", CORPUS_ROOT, tmp_path / "a", **kw)  # type: ignore[arg-type]
    b = B.run_baseline("lid-features", CORPUS_ROOT, tmp_path / "b", **kw)  # type: ignore[arg-type]

    for name in ("last_lid.bin", "best_lid.bin", "last_sad.bin", "best_sad.bin"):
        assert (a.checkpoint_dir / name).read_bytes() == (b.checkpoint_dir / name).read_bytes(), f"{name} not bit-identical across two fixed-seed runs"
    assert a.lid_error == b.lid_error, f"held-out error not reproducible: {a.lid_error} vs {b.lid_error}"


@pytest.mark.slow
@requires_corpus
def test_lid_features_dry_run_smoke(tmp_path: Path) -> None:
    """The `--dry-run` 1-step smoke: init + 1 epoch on a tiny slice + score, no long training.
    Proves the whole arm wiring (listings, config, seeded init, engine, scoring) runs fast."""
    res = B.run_baseline("lid-features", CORPUS_ROOT, tmp_path / "dry", dry_run=True, seed=0)
    assert res.epochs_run == 1, "dry-run must run exactly 1 epoch"
    assert res.n_train > 0 and res.n_test > 0
    assert res.metadata_path.is_file()
    assert res.lid_error is not None and np.isfinite(res.lid_error), "dry-run must still score the held-out slice end to end"
    # the dry-run metadata records the forced 1-step regime.
    import json

    meta = json.loads(res.metadata_path.read_text())
    assert meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1
