"""Phase 6 Task 8: the LID FEATURES arm subset gate (corpus-gated, local-only, pyo3).

THE flagship end-to-end proof: a stratified seeded subset of the LRE03 `.plp8f0mvsdd` cep
features (File_Type 2) trains the 12-class Twin (Algo 6, Mode 7) FROM SCRATCH via
`train_modern` (val_metric=nn_cost_seg), then the trained model is scored on a disjoint
held-out slice end to end -- `drivers/test.py::evaluate` -> `.scr` -> `read_scr_scores` ->
`lid_error` + `cavg` (Task 5). These are the phase's FIRST real headline numbers.

WHY THE IMPROVEMENT SIGNAL IS ARGMAX ERROR, NOT THE CE VALIDATION COST (a Task-8 finding):
from scratch the softmax cross-entropy VALIDATION cost CAN rise while the argmax accuracy
improves -- the net grows confident on the (majority-leaning) classes it learns first, so
its held-out CE can overfit even as it starts getting languages right. This is a GENERAL
caution (observed in exploratory/longer runs), NOT necessarily this committed 2-epoch run,
whose val cost actually DECREASED (best_epoch=1); either way the gate keys off the TASK
metric regardless. So the honest, direction-safe improvement here is the held-out LID argmax
error: it must (a) beat the 12-way chance error (91.67%) with headroom and (b) beat the
model's OWN untrained-init error on the identical test set. The TRAIN cost descending is the
separate machinery-works signal.
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

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import CORPUS_ROOT, GateRecorder, requires_corpus

pytest.importorskip("speech_rs")

from speech.drivers import baseline as B  # noqa: E402 -- after importorskip, matching the pyo3-suite convention
from speech.drivers.spec import BaselineSpec  # noqa: E402

_CHANCE = 100.0 * (1.0 - 1.0 / 12.0)  # 12-way argmax chance error = 91.666...%


@pytest.mark.slow
@requires_corpus
def test_lid_features_subset_trains_and_scores(tmp_path: Path, gate_record: GateRecorder) -> None:
    """Train the 12-class Twin from scratch on a stratified cep-features subset, then score a
    disjoint 48-file held-out slice end to end. Pins (measured 2026-07-18, seed 0):
    lid_error 72.92%, init 93.75%, cavg 0.46, train_cost 2.368 -> 2.216; ~282 s."""
    spec = BaselineSpec(
        arm="lid-features",
        corpus_root=CORPUS_ROOT,
        out_dir=tmp_path / "run",
        subset=36,  # ~3 train files/language (per-language proportional, seeded)
        valid_size=12,
        test_size=48,  # ~4 held-out files/language -- a stable argmax-error denominator
        epochs=2,
        steps_per_epoch=25,  # past SMORMS3's ~6-step warm-up so each epoch genuinely descends
        patience=99,
        seed=0,
        score_init=True,  # also score the untrained init on the SAME test set (the improvement baseline)
    )
    res = B.run_baseline(spec)

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
    gate_record(res)


@pytest.mark.slow
@requires_corpus
def test_lid_features_deterministic(tmp_path: Path) -> None:
    """Run-twice determinism at a fixed seed: bit-identical trained weights + identical
    held-out error. A SHORT config (determinism is a pipeline property, provable cheaply);
    the full gate's determinism was measured separately (2026-07-18) and holds identically."""
    kw = dict(arm="lid-features", corpus_root=CORPUS_ROOT, subset=24, valid_size=12, test_size=12, epochs=1, steps_per_epoch=6, patience=99, seed=0)
    a = B.run_baseline(BaselineSpec(out_dir=tmp_path / "a", **kw))  # type: ignore[arg-type]
    b = B.run_baseline(BaselineSpec(out_dir=tmp_path / "b", **kw))  # type: ignore[arg-type]

    for name in ("last_lid.bin", "best_lid.bin", "last_sad.bin", "best_sad.bin"):
        assert (a.checkpoint_dir / name).read_bytes() == (b.checkpoint_dir / name).read_bytes(), f"{name} not bit-identical across two fixed-seed runs"
    assert a.lid_error == b.lid_error, f"held-out error not reproducible: {a.lid_error} vs {b.lid_error}"


@pytest.mark.slow
@requires_corpus
def test_lid_features_dry_run_smoke(tmp_path: Path) -> None:
    """The `--dry-run` 1-step smoke: init + 1 epoch on a tiny slice + score, no long training.
    Proves the whole arm wiring (listings, config, seeded init, engine, scoring) runs fast."""
    res = B.run_baseline(BaselineSpec(arm="lid-features", corpus_root=CORPUS_ROOT, out_dir=tmp_path / "dry", dry_run=True, seed=0))
    assert res.epochs_run == 1, "dry-run must run exactly 1 epoch"
    assert res.n_train > 0 and res.n_test > 0
    assert res.metadata_path.is_file()
    assert res.lid_error is not None and np.isfinite(res.lid_error), "dry-run must still score the held-out slice end to end"
    # the dry-run metadata records the forced 1-step regime.
    meta = json.loads(res.metadata_path.read_text())
    assert meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1


# ============================== SAD ARM (Task 9) =============================================
#
# From-scratch algo-3 spectral SAD (File_Type 0 wav) on the real corpus wav/xml pairs, scored
# END TO END with the T4 DCF harness -- the engine's dumped VRCTS hyps vs the `.part.xml`
# references (windowed to the capped-audio span) -> pooled `dcf` -> the FIRST REAL DCF NUMBERS.
#
# WHY THE SIGNAL IS THE TRAINED-VS-INIT DCF, NOT A BEAT-TRIVIAL DCF (the honest caveat, the SAD
# analogue of the LID mode-collapse note above): the corpus wavs cap to ~53% speech windows, so
# on a tiny subset the from-scratch net mode-collapses toward the window-majority -- the seeded
# init sits BELOW the rising threshold everywhere (all non-speech, DCF ~0.75), and a few SMORMS3
# epochs drive it ABOVE the threshold everywhere (all speech, DCF ~0.25). There is no
# partial-discrimination sweet spot at this scale (measured: the net jumps from all-non-speech
# straight to all-speech). The direction-safe TASK metric is therefore the trained-vs-init
# held-out DCF (init all-non-speech ~0.75 -> trained fires ~0.25, +0.50 at every collar), plus
# "the net learned to fire" (trained Pmiss ~0 vs init Pmiss ~1); genuine speech/non-speech
# discrimination (a DCF below the all-speech 0.25 baseline) is the FULL-RUN launcher's job (more
# data, more epochs). Because both endpoints are degenerate (all-one-class), the DCF is exactly
# 0.25/0.75 with no libm-sensitive boundary jitter -- rock-stable cross-machine. Margins were
# MEASURED on this box (2026-07-18, seed 0) and PINNED with generous headroom.


@pytest.mark.slow
@requires_corpus
def test_sad_subset_trains_and_scores(tmp_path: Path, gate_record: GateRecorder) -> None:
    """Train algo-3 SAD from scratch on a 10-file subset (20 s cap), then score a disjoint
    24-file held-out slice END TO END via the T4 DCF harness. Pins (measured 2026-07-18, seed
    0, ~80 s): held-out DCF@0.5 trained 0.2500 (Pmiss 0.0 Pfa 1.0) vs init 0.7500 (Pmiss 1.0
    Pfa 0.0), +0.50 at every collar; the trained net fires (all-speech collapse), the init
    does not (all-non-speech)."""
    spec = BaselineSpec(
        arm="sad",
        corpus_root=CORPUS_ROOT,
        out_dir=tmp_path / "run",
        subset=10,  # 10 train wavs (seeded first-N of the 70/15/15 SAD split)
        valid_size=8,
        test_size=24,  # a stable pooled-DCF denominator, disjoint from train/valid
        epochs=3,
        steps_per_epoch=10,  # ~30 SMORMS3 steps: past warm-up, into the all-speech attractor
        patience=99,
        seed=0,
        audio_max_duration=20.0,  # cap the 576-1800 s CallFriend recordings (median ~600 s); the ref windows to match
        score_init=True,  # also score the untrained init on the SAME test set (the DCF baseline)
    )
    res = B.run_baseline(spec)

    # --- both DCF reports landed, valid ranges ---
    assert res.dcf is not None and res.init_dcf is not None
    tr = res.dcf.by_collar(0.5)
    ini = res.init_dcf.by_collar(0.5)
    assert np.isfinite(tr.dcf) and 0.0 <= tr.dcf <= 1.0 and 0.0 <= ini.dcf <= 1.0

    # --- the net learned to FIRE: trained detects speech, the untrained init misses ~all ---
    assert tr.pmiss < 0.5, f"trained must detect held-out speech (Pmiss {tr.pmiss:.3f})"
    assert ini.pmiss > 0.9, f"the untrained init misses ~all speech (Pmiss {ini.pmiss:.3f})"

    # --- the direction-safe task metric: trained held-out DCF beats its OWN init by a margin ---
    #     (measured trained 0.25 vs init 0.75, gap 0.50; pin a conservative 0.2 gap).
    assert tr.dcf < ini.dcf - 0.2, f"trained DCF {tr.dcf:.4f} must beat init {ini.dcf:.4f} by margin"
    # trained beats the all-non-speech baseline (0.75) with headroom; init sits near it.
    assert tr.dcf <= 0.5, f"trained DCF {tr.dcf:.4f} must beat the all-non-speech baseline (0.75) with headroom"
    assert ini.dcf >= 0.6, f"untrained init should sit near the all-non-speech baseline, got {ini.dcf:.4f}"

    # --- self-describing run: metadata + checkpoint + the dumped VRCTS hyps landed ---
    assert res.metadata_path.is_file()
    assert (res.checkpoint_dir / "best_sad.bin").is_file() and (res.checkpoint_dir / "last_sad.bin").is_file()
    assert res.scores_dir is not None and len(list(res.scores_dir.glob("*.xml"))) == res.n_test
    gate_record(res)


@pytest.mark.slow
@requires_corpus
def test_sad_deterministic(tmp_path: Path) -> None:
    """Run-twice determinism at a fixed seed: bit-identical trained weights + identical pooled
    held-out DCF. A SHORT config (determinism is a pipeline property, provable cheaply); the
    full gate's determinism was measured separately (2026-07-18, bit-identical best_sad.bin)."""
    kw = dict(arm="sad", corpus_root=CORPUS_ROOT, subset=8, valid_size=6, test_size=8, epochs=1, steps_per_epoch=8, patience=99, seed=0)
    kw["audio_max_duration"] = 15.0
    a = B.run_baseline(BaselineSpec(out_dir=tmp_path / "a", **kw))  # type: ignore[arg-type]
    b = B.run_baseline(BaselineSpec(out_dir=tmp_path / "b", **kw))  # type: ignore[arg-type]

    for name in ("last_sad.bin", "best_sad.bin"):
        assert (a.checkpoint_dir / name).read_bytes() == (b.checkpoint_dir / name).read_bytes(), f"{name} not bit-identical across two fixed-seed runs"
    assert a.dcf is not None and b.dcf is not None
    assert a.dcf.by_collar(0.5).dcf == b.dcf.by_collar(0.5).dcf, "pooled held-out DCF not reproducible"


@pytest.mark.slow
@requires_corpus
def test_sad_dry_run_smoke(tmp_path: Path) -> None:
    """The `--dry-run` 1-step smoke for the SAD arm: init + 1 epoch on a tiny slice at a short
    audio cap + DCF scoring, no long training. Proves the whole arm wiring (derive_sad_listings,
    config, seeded init, engine VRCTS dump, dcf) runs fast end to end."""
    res = B.run_baseline(BaselineSpec(arm="sad", corpus_root=CORPUS_ROOT, out_dir=tmp_path / "dry", dry_run=True, seed=0))
    assert res.epochs_run == 1, "dry-run must run exactly 1 epoch"
    assert res.n_train > 0 and res.n_test > 0
    assert res.metadata_path.is_file()
    assert res.dcf is not None, "dry-run must still score the held-out slice end to end (a valid DcfReport)"
    assert np.isfinite(res.dcf.by_collar(0.5).dcf)

    meta = json.loads(res.metadata_path.read_text())
    assert meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1
    assert meta["audio_max_duration"] == 10.0, "the SAD dry-run forces a short audio cap"


# ============================== LID PHONOTACTIC ARM (Task 10) ================================
#
# The 2015 FLAGSHIP regime: the 12-class Twin (Algo 6) in Mode 7 over File_Type 1 phSeq (the
# fully-ported, bit-exact-lineage phSeq reader), trained FROM SCRATCH via train_modern, then
# scored on a disjoint held-out slice -> `.scr` -> lid_error + cavg (Task 5). Same shape as the
# LID FEATURES arm above; the DISTINCTIVE proof here is THE FROZEN-SAD CONTRACT, asserted
# byte-for-byte.
#
# THE FROZEN-SAD CONTRACT (the phase-5 finding, plan/spec S1.9, VERIFIED at the seam for this
# task -- see .superpowers/sdd/task-10-report.md). In Mode 7 the SAD net (BLSTM_*) is NEVER run:
# `tasks/lid.rs::get_segmentation_mode7` synthesizes its result_vec constant 10.0 (:1407) and
# feeds the LID net the phSeq one-hot DIRECTLY (:1448-1476) -- the SAD hidden states never enter
# the LID input (unlike modes 0-3). The SAD net's weight derivatives are RESET (:1352) then only
# SCALED at the epilogue (:1643), NEVER accumulated -> `weights_derivatives(sad)` is structurally
# ZERO. Through the seam (`engine.forward_backward` -> SMORMS3, `optimizers.py:93` dtheta = -grad*
# ... = 0 with `_backprop_inner`'s pass-through theta) the SAD net stays EXACTLY at its from-scratch
# seed while ONLY the LID net descends. Both nets ARE seeded (the engine needs a loadable Twin);
# ONLY the LID net trains. The gate proves this: best_sad.bin == last_sad.bin == sad_seed.bin
# byte-for-byte (sad_seed.bin equals train_modern's own from-scratch SAD init byte-for-byte), while
# best_lid.bin != lid_seed.bin. The Twin validation metric (nn_cost_seg = NNCostSeg + NNCostLID) is
# still right for a LID-only arm: the frozen SAD's Mode-7 NNCostSeg contribution is weight-
# independent (a constant offset on the validation curve); only NNCostLID moves.
#
# WHY THE SIGNAL IS THE HELD-OUT ARGMAX, NOT THE TRAIN/CE COST (the T8 lesson, and SHARPER here):
# from scratch the per-epoch train cost can RISE while held-out argmax accuracy IMPROVES (measured:
# this config's train_costs ASCEND 2.41 -> 2.93 across the two epochs, yet the held-out error DROPS
# 6.8 pt below chance). So the honest, direction-safe improvement is the held-out LID argmax error:
# it must (a) beat 12-way chance (91.67%) and (b) beat the model's OWN untrained-init error on the
# identical test set. MODE-COLLAPSE-HONEST FRAMING (spec S1.9, "no overclaim"): phonotactic LID from
# scratch on a tiny per-language subset is HARD and UNSTABLE -- the 2-epoch structure reliably beats
# init (measured +4.4 pt at subset 24, +6.7 pt at subset 16), but a single-epoch cut can go NEGATIVE
# (measured -3.0 pt). The margins here are THIN vs the acoustic arm's ~20 pt: the 2015 story was
# phonotactic >> acoustic, but only WITH FULL DATA -- on this subset phonotactic is WEAKER, exactly
# the "needs more data than acoustic" flip side, honestly noted. Margins MEASURED (2026-07-18,
# seed 0) + PINNED with headroom; run-twice determinism proves reproducibility on this box.


@pytest.mark.slow
@requires_corpus
def test_lid_phseq_subset_trains_and_scores(tmp_path: Path, gate_record: GateRecorder) -> None:
    """Train the 12-class Twin (Mode 7, File_Type 1 phSeq) from scratch on a stratified subset,
    then score a disjoint held-out 45-file slice end to end. Pins (measured 2026-07-18, seed 0):
    lid_error 84.44%, init 91.11% (+6.67 pt), cavg 0.49, train_costs 2.41 -> 2.93 (ascending;
    argmax still improves), ~232 s; the SAD net frozen (byte-identical to seed), only the LID
    net trains."""
    spec = BaselineSpec(
        arm="lid-phseq",
        corpus_root=CORPUS_ROOT,
        out_dir=tmp_path / "run",
        subset=16,  # ~1-2 train files/language (per-language proportional, seeded); n_train ~15
        valid_size=12,
        test_size=48,  # a stable argmax-error denominator, disjoint from train/valid
        epochs=2,  # the 2-epoch structure reliably beats init; a 1-epoch cut can go negative
        steps_per_epoch=12,  # past SMORMS3's warm-up
        patience=99,
        seed=0,
        score_init=True,  # also score the untrained init on the SAME test set (the improvement baseline)
    )
    res = B.run_baseline(spec)

    # --- THE FROZEN-SAD CONTRACT (the crux): the SAD net NEVER moves; only the LID net trains ---
    seed_sad = (res.out_dir / "sad_seed.bin").read_bytes()
    assert (res.checkpoint_dir / "best_sad.bin").read_bytes() == seed_sad, "best_sad must equal the from-scratch seed (SAD frozen in Mode 7)"
    assert (res.checkpoint_dir / "last_sad.bin").read_bytes() == seed_sad, "last_sad must equal the from-scratch seed (SAD frozen in Mode 7)"
    assert (res.checkpoint_dir / "best_lid.bin").read_bytes() != (res.out_dir / "lid_seed.bin").read_bytes(), "the LID net must train (move off seed)"

    # --- the end-to-end held-out LID error is real, beats chance, and beats its own init ---
    assert res.lid_error is not None and res.init_lid_error is not None
    assert np.isfinite(res.lid_error)
    # (a) beats 12-way chance (measured 84.44%, +7.2 pt; pin <= chance - 2.0 -> ~5 pt headroom,
    #     robust to descent jitter on the 45-file test set).
    assert res.lid_error <= _CHANCE - 2.0, f"held-out LID error {res.lid_error:.2f}% must beat chance {_CHANCE:.2f}% with margin"
    # (b) beats the model's OWN untrained-init error on the identical test set (measured +6.67 pt).
    assert res.lid_error < res.init_lid_error - 2.0, f"trained {res.lid_error:.2f}% must beat init {res.init_lid_error:.2f}% by margin"

    # --- cavg RECORDED (a valid closed-set Cavg in [0, 1]); no hard pin (not stable at this scale) ---
    assert res.cavg is not None and 0.0 <= res.cavg <= 1.0, f"cavg must be a valid closed-set value, got {res.cavg}"

    # --- the run is self-describing: metadata + both checkpoints + the .scr scores landed ---
    assert res.metadata_path.is_file()
    assert (res.checkpoint_dir / "best_lid.bin").is_file() and (res.checkpoint_dir / "best_sad.bin").is_file()
    assert res.scores_dir is not None and len(list(res.scores_dir.glob("*.scr"))) == res.n_test
    gate_record(res)


@pytest.mark.slow
@requires_corpus
def test_lid_phseq_deterministic(tmp_path: Path) -> None:
    """Run-twice determinism at a fixed seed: bit-identical trained weights (incl. the frozen SAD
    net) + identical held-out error. A SHORT config (determinism is a pipeline property, provable
    cheaply)."""
    kw = dict(arm="lid-phseq", corpus_root=CORPUS_ROOT, subset=12, valid_size=8, test_size=12, epochs=1, steps_per_epoch=6, patience=99, seed=0)
    a = B.run_baseline(BaselineSpec(out_dir=tmp_path / "a", **kw))  # type: ignore[arg-type]
    b = B.run_baseline(BaselineSpec(out_dir=tmp_path / "b", **kw))  # type: ignore[arg-type]

    for name in ("last_lid.bin", "best_lid.bin", "last_sad.bin", "best_sad.bin"):
        assert (a.checkpoint_dir / name).read_bytes() == (b.checkpoint_dir / name).read_bytes(), f"{name} not bit-identical across two fixed-seed runs"
    assert a.lid_error == b.lid_error, f"held-out error not reproducible: {a.lid_error} vs {b.lid_error}"


@pytest.mark.slow
@requires_corpus
def test_lid_phseq_dry_run_smoke(tmp_path: Path) -> None:
    """The `--dry-run` 1-step smoke for the phonotactic arm: init + 1 epoch on a tiny slice +
    score, no long training. Proves the whole arm wiring (phSeq glob, config, seeded init, engine
    Mode-7 forward, scoring) runs fast -- and that the frozen-SAD contract holds even in 1 step."""
    res = B.run_baseline(BaselineSpec(arm="lid-phseq", corpus_root=CORPUS_ROOT, out_dir=tmp_path / "dry", dry_run=True, seed=0))
    assert res.epochs_run == 1, "dry-run must run exactly 1 epoch"
    assert res.n_train > 0 and res.n_test > 0
    assert res.metadata_path.is_file()
    assert res.lid_error is not None and np.isfinite(res.lid_error), "dry-run must still score the held-out slice end to end"
    # the frozen-SAD contract holds even in a 1-step smoke.
    assert (res.checkpoint_dir / "best_sad.bin").read_bytes() == (res.out_dir / "sad_seed.bin").read_bytes()
    meta = json.loads(res.metadata_path.read_text())
    assert meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1
