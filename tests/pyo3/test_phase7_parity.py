"""Phase 7 Task 6: the corpus-gated metric-parity tier (fast vs exact, real data).

The CI parity legs (`src/rust/tests/phase7_parity_{sad,lid}.rs`) proved the fast f32
inference path is BOUNDARY-exact and ARGMAX-exact against the exact f64 path on the
committed fixtures (SAD boundary max_dt = 0.0, count/types identical; LID argmax zero
flips). This tier lifts that proof onto REAL corpus data at the METRIC level: the same
phase-6 subset checkpoints are scored on the same held-out slice under BOTH
`Inference_Path` values, and the decisions + the reported metrics are compared.

THE PREDICTION (from the CI legs): identical per-file decisions => identical metrics, so
the metric deltas are EXACTLY 0.0. This file asserts that hard:
  - LID (features + phSeq): per-file argmax IDENTICAL fast-vs-exact (the drift detector);
    `lid_error` and `cavg` deltas EXACTLY 0.0 (both are argmax-only functions -- softmax is
    monotone -- so identical argmax => bit-identical metric, `evaluate.py` docstrings).
  - SAD: per-file VRCTS hypothesis boundaries IDENTICAL (count + types + times); the pooled
    held-out DCF delta EXACTLY 0.0 at every collar (identical boundaries => identical DCF).
The score/posterior VALUES still differ by the f32 tolerance (reported as a diagnostic); it
is the DECISIONS that must not move. If a genuine divergence ever appears, the R1 discipline
applies: the assertion fails, and the report records the file count + delta distribution for
open adjudication (do NOT silently widen an epsilon to make it pass).

CHECKPOINTS: reuse the EXACT phase-6 subset-gate recipe (`tests/pyo3/test_phase6_gates.py`)
to TRAIN the checkpoints -- training always runs on the EXACT f64 path (the fast path is
inference-only by design). Training is the expensive step (~1-5 min/arm); scoring is
seconds. So checkpoints are cached under a gitignored session-persistent dir
(`data/phase7_parity_cache/`, `.gitignore`d -- NEVER committed, it holds corpus-path
listings) keyed by arm, and a warm cache SKIPS training. `run_baseline` at a fixed seed is
deterministic (proven by the phase-6 determinism gates), so a cached checkpoint is
bit-identical to what a fresh run would produce -- the cache is a pure speedup, not a
correctness compromise.

Corpus-gated (`requires_corpus`) + pyo3 (`importorskip`): runs locally for implementers AND
reviewers, skips cleanly in CI. Each arm is its own test so a slow cold-cache run stays
inside a single per-test invocation (the pytest tool-timeout convention). ASCII only.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import CORPUS_ROOT, requires_corpus

pytest.importorskip("speech_rs")

from speech.batching import read_listing  # noqa: E402 -- after importorskip, matching the pyo3-suite convention
from speech.config_bridge import parse_legacy_config  # noqa: E402
from speech.drivers import baseline as B  # noqa: E402
from speech.drivers.state import RunState  # noqa: E402
from speech.drivers.test import _class_keys  # noqa: E402
from speech.evaluate import load_vrcts_hyp, read_scr_scores  # noqa: E402

# The five DCF collars the SAD arm pools over (baseline._DCF_COLLARS).
_COLLARS: tuple[float, ...] = B._DCF_COLLARS

# Session-persistent, gitignored checkpoint cache (holds corpus-path listings -> NEVER
# committed; `.gitignore` carries `/data/phase7_parity_cache/`). parents[2] = repo root.
_CACHE = Path(__file__).resolve().parents[2] / "data" / "phase7_parity_cache"

# The phase-6 subset-gate MAIN recipes, verbatim from tests/pyo3/test_phase6_gates.py --
# these produce the "phase-6 subset checkpoints" this tier scores. `score_init` is omitted
# (this tier re-scores both paths itself; run_baseline's own scoring is the one-time
# cold-cache exact pass we do not consume).
_RECIPES: dict[str, dict[str, object]] = {
    "lid-features": dict(subset=36, valid_size=12, test_size=48, epochs=2, steps_per_epoch=25, patience=99, seed=0),
    "lid-phseq": dict(subset=16, valid_size=12, test_size=48, epochs=2, steps_per_epoch=12, patience=99, seed=0),
    "sad": dict(subset=10, valid_size=8, test_size=24, epochs=3, steps_per_epoch=10, patience=99, seed=0, audio_max_duration=20.0),
}

# The test listing filename run_baseline writes under out_dir for each arm.
_TEST_LISTING: dict[str, str] = {
    "lid-features": "lre03_lid_features_test.flst",
    "lid-phseq": "lre03_lid_phseq_test.flst",
    "sad": "sad_test_subset.flst",
}

# Exact-path sanity pins (T6 review): `_RECIPES` + `seed=0` are deterministic (the phase-6
# determinism gates), so the EXACT-path metric on a freshly-trained checkpoint reproduces
# bit-for-bit run to run. Pinning it turns silent checkpoint-provenance drift (a stale cache,
# or a recipe/seed edit that changes the trained weights without updating this pin) into a
# loud failure here, instead of a fast-vs-exact parity check that keeps passing against the
# WRONG checkpoint. Values rounded to 4dp, matching the `:.4f`/`:.6f` print formatting below.
# Re-derive deliberately (never just widen/overwrite on a red run without checking WHY --
# `rm -rf data/phase7_parity_cache/<arm>` first, per `_ensure_arm`'s cache-staleness note).
_EXPECTED_EXACT_LID_ERROR: dict[str, float] = {
    "lid-features": 72.9167,
    "lid-phseq": 84.4444,
}
_EXPECTED_EXACT_SAD_DCF_AT_0_5 = 0.2500


@dataclass
class _Arm:
    """A trained arm's on-disk artefacts (the warm-cache handle)."""

    arm: str
    out_dir: Path
    ckpt: Path
    base_config: Path
    test_listing: Path
    test_records: list[dict[str, str]]
    train_s: float
    trained_now: bool
    timings: dict[str, float] = field(default_factory=dict)


def _ensure_arm(arm: str) -> _Arm:
    """Train the arm's checkpoint on the EXACT path (phase-6 subset recipe) if the cache is
    cold, else reuse it. The `.parity_ready.json` sentinel marks a COMPLETE run (written only
    after run_baseline returns), so an interrupted training leaves no sentinel and the next
    call retrains from a clean dir.

    CACHE-STALENESS CAVEAT (T6 review): the sentinel guards INTERRUPTED runs only -- it says
    nothing about whether `_RECIPES[arm]`, the training code (`run_baseline`/`train_modern`/
    the arm's TOML), or `_EXPECTED_EXACT_LID_ERROR`/`_EXPECTED_EXACT_SAD_DCF_AT_0_5` have
    drifted apart since the cache was written. A warm cache is reused as-is even if the recipe
    or training code changed underneath it. Touching training code or an arm's recipe requires
    `rm -rf data/phase7_parity_cache/<arm>` to force a clean retrain before trusting this
    tier's pins again."""
    out_dir = (_CACHE / arm).resolve()
    ckpt = out_dir / "checkpoint"
    sentinel = out_dir / ".parity_ready.json"

    trained_now = False
    if not sentinel.is_file():
        if out_dir.exists():
            shutil.rmtree(out_dir)  # clean any partial cache (specific cache subdir only)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        B.run_baseline(arm, CORPUS_ROOT, out_dir, **_RECIPES[arm])  # type: ignore[arg-type]
        train_s = time.time() - t0
        sentinel.write_text(json.dumps({"train_s": train_s, "recipe": {k: str(v) for k, v in _RECIPES[arm].items()}}))
        trained_now = True
    else:
        train_s = float(json.loads(sentinel.read_text()).get("train_s", 0.0))

    test_listing = out_dir / _TEST_LISTING[arm]
    records = read_listing(test_listing)
    assert (ckpt / "best_sad.bin").is_file(), f"{arm}: checkpoint missing (training failed?)"
    return _Arm(arm, out_dir, ckpt, out_dir / "base.config", test_listing, records, train_s, trained_now)


def _eval_config(a: _Arm, inference_path: str, tag: str, *, twin: bool) -> Path:
    """An inference-shaped eval config: the trained base config, repointed at the held-out
    test listing, with backprop OFF + `Neural_Networks_BackPropagation_Epochs 0` (so the
    fast+training construction bail never fires -- the config is genuinely inference-shaped)
    + the chosen `Inference_Path`. exact and fast configs are byte-identical except that one
    key, so the comparison isolates the f32 path.

    THE WEIGHT KEYS ARE REPOINTED AT THE TRAINED CHECKPOINT (not the base config's seed
    pack): the fast drivers load weights ONLY at construction via `load_weights_file`
    (`BLSTM_weightsFile`/`BLSTM_LID_weightsFile`) -- there is no settable-after-construction
    f64 weight surface, so `BagOfProcessors::set_weights` now BAILS LOUDLY on
    `Processor::FastSpectral`/`FastTwinLid` (T6b, `bag_of_processors.rs`'s `set_weights`
    match arms; pre-T6b it was a SILENT NO-OP, which is the failure mode this repoint
    guards against -- the seam would otherwise score the config's SEED weights while exact
    scores the injected TRAINED ones, a 100%-divergence artefact, NOT a real parity
    failure). `evaluate()`/`_score_packs_on_test`/`_score_sad_pack_on_test` all SKIP their
    `set_weights` call under `Inference_Path fast` (also T6b) specifically because this
    repoint already did the injection at config time -- the two mechanisms are the same
    injection, so skipping is semantics-preserving, not a workaround. Loading trained
    weights from the config is the fast path's only injection mechanism (and how a real
    fast deployment loads them), and it is identical for both paths -- so this is the
    apples-to-apples comparison. In Mode 7 the SAD net is frozen (`best_sad == sad_seed`),
    so `BLSTM_weightsFile` is weight-neutral there; `BLSTM_LID_weightsFile` carries the
    trained LID net."""
    cfg = dict(parse_legacy_config(a.base_config.read_text()))
    cfg["fileslisting"] = a.test_listing.name
    cfg["BLSTM_weightsFile"] = str((a.ckpt / "best_sad.bin").resolve())
    cfg["BLSTM_BackPropagationActivated"] = "false"
    if twin:
        cfg["BLSTM_LID_weightsFile"] = str((a.ckpt / "best_lid.bin").resolve())
        cfg["BLSTM_LID_BackPropagationActivated"] = "false"
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    cfg["Inference_Path"] = inference_path
    path = a.out_dir / f"parity_eval_{tag}.config"
    path.write_text(B._config_text(cfg))
    return path


# --------------------------------------------------------------------------------------- #
# LID arms (features + phSeq): score the Twin checkpoint under both paths, compare argmax
# --------------------------------------------------------------------------------------- #


def _score_lid(a: _Arm, inference_path: str, tag: str) -> tuple[float, float, np.ndarray, list[str]]:
    """Score the `[best_sad, best_lid]` checkpoint on the held-out slice under `inference_path`
    (`exact`/`fast`) end to end via baseline's own `_score_packs_on_test` (-> `.scr` ->
    `lid_error`/`cavg`), then read the `.scr` scores back for the per-file argmax."""
    eval_config = _eval_config(a, inference_path, tag, twin=True)
    eval_state = RunState.from_config(eval_config, a.out_dir)
    score_dir = a.out_dir / f"parity_lid_{tag}"
    if score_dir.exists():
        shutil.rmtree(score_dir)
    t0 = time.time()
    lid_err, cavg_val, scores_dir = B._score_packs_on_test(eval_state, a.ckpt / "best_sad.bin", a.ckpt / "best_lid.bin", score_dir, a.test_records)
    a.timings[f"score_{tag}_s"] = time.time() - t0
    assert lid_err is not None and cavg_val is not None, f"{a.arm}/{tag}: no .scr scores produced (empty held-out slice)"

    keys = _class_keys(a.out_dir / eval_state.base_config["language2classmapping"])
    scores, files = read_scr_scores(scores_dir, keys)
    return lid_err, cavg_val, scores, files


def _run_lid_parity(arm: str) -> None:
    a = _ensure_arm(arm)
    err_e, cavg_e, sc_e, files_e = _score_lid(a, "exact", "exact")
    err_f, cavg_f, sc_f, files_f = _score_lid(a, "fast", "fast")

    # Same held-out file set, same sorted order (read_scr_scores sorts by glob).
    assert files_e == files_f, f"{arm}: fast/exact scored different files ({len(files_e)} vs {len(files_f)})"

    argmax_e = sc_e.argmax(axis=1)
    argmax_f = sc_f.argmax(axis=1)
    disagree = np.flatnonzero(argmax_e != argmax_f)
    score_max_abs = float(np.max(np.abs(sc_e - sc_f))) if sc_e.size else 0.0

    print(
        f"\n[PARITY {arm}] files={len(files_e)} argmax_disagree={len(disagree)} "
        f"score_max_abs={score_max_abs:.3e} lid_error exact={err_e:.4f} fast={err_f:.4f} "
        f"d={abs(err_f - err_e):.3e} cavg exact={cavg_e:.6f} fast={cavg_f:.6f} d={abs(cavg_f - cavg_e):.3e} "
        f"train_s={a.train_s:.1f} (cold={a.trained_now}) score_exact_s={a.timings.get('score_exact_s', 0):.2f} "
        f"score_fast_s={a.timings.get('score_fast_s', 0):.2f}"
    )

    # HARD (T6 review): the EXACT-path lid_error matches the pinned phase-6 checkpoint value
    # -- catches checkpoint-provenance drift (stale/mismatched cache, a recipe/seed edit)
    # that would otherwise let the fast-vs-exact comparison below pass against the wrong
    # checkpoint silently. See `_EXPECTED_EXACT_LID_ERROR`'s note before touching this pin.
    expected = _EXPECTED_EXACT_LID_ERROR[arm]
    assert round(err_e, 4) == expected, f"{arm}: exact-path lid_error={err_e:.4f} != pinned {expected} (stale/mismatched checkpoint cache?)"

    # HARD (R1 drift detector): identical per-file argmax fast-vs-exact. If this ever fails,
    # STOP and adjudicate with the printed disagree count + files -- do NOT widen.
    assert len(disagree) == 0, f"{arm}: {len(disagree)} of {len(files_e)} files flipped argmax fast-vs-exact (indices {disagree.tolist()[:20]})"

    # Identical argmax => bit-identical argmax-only metrics: EXACTLY 0.0 delta (the prediction).
    assert err_f == err_e, f"{arm}: lid_error moved {err_e} -> {err_f} despite identical argmax"
    assert cavg_f == cavg_e, f"{arm}: cavg moved {cavg_e} -> {cavg_f} despite identical argmax"


@pytest.mark.slow
@requires_corpus
def test_lid_features_metric_parity() -> None:
    """LID FEATURES (Twin, Mode 7, File_Type 2 cep): fast vs exact on the phase-6 subset
    checkpoint. Per-file argmax IDENTICAL; lid_error + cavg deltas EXACTLY 0.0."""
    _run_lid_parity("lid-features")


@pytest.mark.slow
@requires_corpus
def test_lid_phseq_metric_parity() -> None:
    """LID PHONOTACTIC (Twin, Mode 7, File_Type 1 phSeq): fast vs exact on the phase-6 subset
    checkpoint. Per-file argmax IDENTICAL; lid_error + cavg deltas EXACTLY 0.0."""
    _run_lid_parity("lid-phseq")


# --------------------------------------------------------------------------------------- #
# SAD arm (algo-3 spectral): score the checkpoint under both paths, compare VRCTS boundaries
# --------------------------------------------------------------------------------------- #


def _score_sad(a: _Arm, inference_path: str, tag: str) -> tuple[B.DcfReport, Path]:
    """Score the `best_sad` checkpoint on the held-out slice under `inference_path` end to
    end via baseline's own `_score_sad_pack_on_test` (engine dumps one VRCTS hyp xml/file ->
    pooled `dcf` over the windowed `.part.xml` refs). Returns the pooled DcfReport + the dump
    dir (the per-file hyp xmls, for the boundary comparison)."""
    cfg = dict(parse_legacy_config(a.base_config.read_text()))
    # Point the weight-file key at the TRAINED checkpoint: the fast SAD driver loads weights
    # only at construction (`set_weights` now BAILS LOUDLY on `Processor::FastSpectral`,
    # T6b -- see `_eval_config`'s note), so both paths must load the trained pack from the
    # config to score the same net. `_score_sad_pack_on_test` also `set_weights`-injects the
    # same pack on exact (redundant there -- config-time injection already did it -- but
    # load-bearing for `run_baseline`'s OWN internal exact-only scoring calls, which do NOT
    # repoint BLSTM_weightsFile this way); on fast it SKIPS the call entirely (T6b) since
    # the config repoint above is that path's only, already-sufficient injection.
    cfg["BLSTM_weightsFile"] = str((a.ckpt / "best_sad.bin").resolve())
    cfg["Inference_Path"] = inference_path
    dump_dir = a.out_dir / f"parity_sad_{tag}"
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    t0 = time.time()
    report, dump = B._score_sad_pack_on_test(cfg, a.out_dir, a.ckpt / "best_sad.bin", dump_dir, a.test_records, a.test_listing.name)
    a.timings[f"score_{tag}_s"] = time.time() - t0
    assert report is not None, f"{a.arm}/{tag}: no VRCTS hyps produced (empty held-out slice)"
    return report, dump


@pytest.mark.slow
@requires_corpus
def test_sad_metric_parity() -> None:
    """SAD (algo-3 spectral, File_Type 0 wav): fast vs exact on the phase-6 subset checkpoint.
    Per-file VRCTS boundaries IDENTICAL (count + types + times); pooled DCF delta EXACTLY 0.0
    at every collar.

    T11 note: `n_scored` counts files that produced a hyp xml on BOTH paths (a file missing
    a hyp on either path is skipped via `continue`); it is a SCORED-file count, not an
    assertion that every `test_records` entry was covered -- the only coverage gate is
    `n_scored > 0`."""
    a = _ensure_arm("sad")
    rep_e, dump_e = _score_sad(a, "exact", "exact")
    rep_f, dump_f = _score_sad(a, "fast", "fast")

    n_scored = 0
    type_or_count_disagree = 0
    max_dt = 0.0
    for rec in a.test_records:
        xe = B._hyp_xml_for(rec["filename"], dump_e)
        xf = B._hyp_xml_for(rec["filename"], dump_f)
        if not (xe.is_file() and xf.is_file()):
            continue
        ie = load_vrcts_hyp(xe)
        if_ = load_vrcts_hyp(xf)
        n_scored += 1
        kinds_e = [k for _, _, k in ie]
        kinds_f = [k for _, _, k in if_]
        if kinds_e != kinds_f:
            type_or_count_disagree += 1
            continue
        for (se, ee, _), (sf, ef, _) in zip(ie, if_, strict=True):
            max_dt = max(max_dt, abs(se - sf), abs(ee - ef))

    dcf_deltas = {c: abs(rep_f.by_collar(c).dcf - rep_e.by_collar(c).dcf) for c in _COLLARS}
    print(
        f"\n[PARITY sad] files={n_scored} boundary_type_or_count_disagree={type_or_count_disagree} "
        f"max_boundary_dt={max_dt:.3e}s dcf@0.5 exact={rep_e.by_collar(0.5).dcf:.6f} fast={rep_f.by_collar(0.5).dcf:.6f} "
        f"dcf_deltas={ {c: f'{d:.3e}' for c, d in dcf_deltas.items()} } "
        f"train_s={a.train_s:.1f} (cold={a.trained_now}) score_exact_s={a.timings.get('score_exact_s', 0):.2f} "
        f"score_fast_s={a.timings.get('score_fast_s', 0):.2f}"
    )

    # HARD (T6 review): the EXACT-path DCF@0.5 matches the pinned phase-6 checkpoint value --
    # catches checkpoint-provenance drift (stale/mismatched cache, a recipe/seed edit) that
    # would otherwise let the fast-vs-exact comparison below pass against the wrong checkpoint
    # silently. See `_EXPECTED_EXACT_SAD_DCF_AT_0_5`'s note before touching this pin.
    exact_dcf_0_5 = rep_e.by_collar(0.5).dcf
    assert round(exact_dcf_0_5, 4) == _EXPECTED_EXACT_SAD_DCF_AT_0_5, (
        f"sad: exact-path DCF@0.5={exact_dcf_0_5:.6f} != pinned {_EXPECTED_EXACT_SAD_DCF_AT_0_5} (stale/mismatched checkpoint cache?)"
    )

    # HARD (R1 drift detector): identical per-file segment count+types, identical boundary
    # times, EXACTLY-0.0 pooled DCF delta at every collar. If any ever moves, STOP + adjudicate.
    assert n_scored > 0, "sad: no held-out hyp xmls to compare"
    assert type_or_count_disagree == 0, f"sad: {type_or_count_disagree}/{n_scored} files differ in segment count/types fast-vs-exact"
    assert max_dt == 0.0, f"sad: boundary times moved fast-vs-exact (max_dt={max_dt:.3e}s)"
    for c in _COLLARS:
        assert rep_f.by_collar(c).dcf == rep_e.by_collar(c).dcf, f"sad: DCF@{c} moved {rep_e.by_collar(c).dcf} -> {rep_f.by_collar(c).dcf}"
