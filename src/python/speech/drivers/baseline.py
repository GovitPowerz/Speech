"""Phase 6 baseline-training arms: the from-scratch SAD/LID launchers + subset gates.

Design spec `docs/superpowers/specs/2026-07-17-phase-6-baseline-training-design.md`
(S1.7) + plan Task 8. This module is the ORCHESTRATOR that stitches the landed Phase 6
seams into a runnable arm: the cep ingestion (Task 1, `audio.rs::read_cep`), the listing
tooling (Task 2, `dataprep/lre.py`), the modern training loop (Phase 5 + Task 6,
`drivers/train.py::train_modern` with `val_metric="nn_cost_seg"`), the seeded init (Task 3,
`init_weights.py`), and the LID metrics (Task 5, `evaluate.py`). It invents NO new engine
path -- every heavy step delegates to an already-golden-tested seam.

THE LID FEATURES ARM (`arm="lid-features"`): a 12-class Twin (Algo 6) in Mode 7 over the
File_Type-2 `.plp8f0mvsdd` cep features. Mode 7 is the ONLY twin mode that consumes
`audio.external_features` directly (every other mode recomputes a periodogram from the
zeroed cep waveform); the SAD net is a FROZEN feature gate there (constant result_vec), so
ONLY the LID net trains -- the same structural finding the phonotactic Mode-7 arm carries.
The cep files are pre-VAD'd whole-utterance speech (verified: the archive's own
`.plp8f0m.xml` refs carry one SpeechSegment spanning `sigdur`), so a synthesized
whole-file-speech STM reference satisfies the engine's `-m`-mode mandatory-reference gate;
the LID training target is the listing's language label, not the seg reference.

R2 (vtln vs plain): the archive's TRAIN side carries only plain `plp8f0mvsdd`; a
vtln/cmllr-train pairing is structurally impossible, so the baseline pairing is PLAIN
features on both train and eval (RESULTS.md carries the evidence).

LICENSE HYGIENE: nothing corpus-derived is committed. The listing, mapping, reference, and
seed weight packs are all synthesized at RUNTIME under `out_dir` from `corpus_root`; the
committed TOML carries only DSP hyperparameter numbers + placeholder path strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console

from speech.config_bridge import nnet_spec
from speech.dataprep.lre import LRE03_LANGUAGES, localize_listing
from speech.drivers.state import ModernTrainParams, RunState
from speech.drivers.train import train_modern
from speech.evaluate import cavg, lid_error, read_scr_scores
from speech.init_weights import init_weights
from speech.weight_bridge import write_bin

# The committed canonical TOML for each arm (relative to the repo root). Task 8 ships
# lid-features; Tasks 9/10 add the sad + lid-phseq entries.
_ARM_CONFIGS: dict[str, str] = {
    "lid-features": "configs/training/lre03_lid_features.toml",
}

# The single whole-file-speech STM reference the cep listings point at (synthesized under
# out_dir). One line, huge end time; `load_ref_stm` clamps it to each file's own duration
# via `Segmentation::new(dur)` + `sanitize()`, so it marks EVERY cep file as one speech
# segment regardless of length -- correct for the pre-VAD'd cep regime. The `s ... s`
# first/third tokens satisfy `load_ref_stm`'s `second.starts_with(first)` speech gate, and
# the trailing two tokens satisfy its >=7-token line requirement.
_SPEECH_STM = "s 1 s 0.0 100000 speech seg\n"

# The 12 LRE03 languages the mapping/head cover (dialect fixed "non"), alphabetical -- the
# order `dataprep.lre.write_lre_mapping` + `drivers/test.py::_class_keys` both use, so the
# mapping's class ids, the engine's lang_index, the `.scr` columns, and the argmax refs all
# share one index base.
_LANGS: tuple[str, ...] = tuple(sorted(LRE03_LANGUAGES))
_CLASS_OF: dict[str, int] = {lang: i for i, lang in enumerate(_LANGS)}


# --------------------------------------------------------------------------------------- #
# Stratified subset + split (pure, seeded, deterministic -- unit-tested with no corpus)
# --------------------------------------------------------------------------------------- #


def _by_language(records: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in records:
        groups[rec["lang"]].append(rec)
    return dict(groups)


def _per_language_counts(groups: dict[str, list[dict[str, str]]], total: int | None) -> dict[str, int]:
    """Per-language quota for a PROPORTIONAL draw of `total` files -- each language gets
    `round(total * n_lang / n_all)`, floored at 1 (min 1/language present) and capped at its
    own count, so the draw preserves the source's per-language proportions AND keeps every
    language (all 12 LID classes) represented even for a small `total`. `total is None` takes
    the FULL per-language count. Shared by `stratified_splits` so train/valid/test all use one
    proportional rule."""
    if total is None:
        return {lang: len(items) for lang, items in groups.items()}
    n_all = sum(len(v) for v in groups.values())
    return {lang: min(len(items), max(1, round(total * len(items) / n_all))) for lang, items in groups.items()}


def stratified_splits(
    records: Sequence[dict[str, str]],
    n_train: int | None,
    n_valid: int,
    n_test: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Draw three DISJOINT, per-language PROPORTIONAL, seeded samples (train, valid, test)
    from `records` -- by explicit target COUNTS, so training cost (~`n_train`) is decoupled
    from held-out size (a big, cheap-to-score `n_test` gives a stable LID error even though
    training stays small). Each split's size is allocated proportionally across languages
    (min 1/language present), then carved from a per-language seeded shuffle in
    train->valid->test order so the three never overlap. `n_train=None` takes ALL remaining
    files per language for train (the full-run launcher). A language with too few files
    fills train/valid/test in that priority order and simply runs out -- no row is
    fabricated or shared across splits. Deterministic (per-language derived seed)."""
    groups = _by_language(records)
    q_valid = _per_language_counts(groups, n_valid)
    q_test = _per_language_counts(groups, n_test)
    q_train = _per_language_counts(groups, n_train)
    train: list[dict[str, str]] = []
    valid: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for lang in sorted(groups):
        items = groups[lang]
        n = len(items)
        rng = np.random.default_rng([seed, _string_seed(lang)])
        order = [int(i) for i in rng.permutation(n)]
        # Priority: train first (it is the training data), then valid, then test -- each
        # bounded by what remains, so the three slices are disjoint by construction.
        cut1 = min(q_train[lang], n)
        cut2 = min(cut1 + q_valid[lang], n)
        cut3 = min(cut2 + q_test[lang], n)
        train.extend(items[i] for i in order[:cut1])
        valid.extend(items[i] for i in order[cut1:cut2])
        test.extend(items[i] for i in order[cut2:cut3])
    return train, valid, test


def _string_seed(s: str) -> int:
    """A stable 32-bit seed from a string (blake2b digest), so per-language shuffles are
    reproducible across runs/platforms without depending on Python's salted `hash()`."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big")


# --------------------------------------------------------------------------------------- #
# Listing derivation + config assembly
# --------------------------------------------------------------------------------------- #


def derive_lid_features_records(corpus_root: Path, ref_stm: Path) -> list[dict[str, str]]:
    """Build LID features listing records straight from the corpus tree (self-contained --
    no dependency on the local-only 2015 legacy listing). Globs
    `train/LID_Features/plp8f0mvsdd/LRE03/*.plp8f0mvsdd`, taking the language from the
    filename prefix (`xxx_0001.plp8f0mvsdd` -> `xxx`), keeping only the 12 known LRE03
    languages, and pointing every row's refseg at the synthesized whole-file-speech STM.
    Every path is absolute (glob results) and every file provably exists (it was globbed).
    Sorted by filename for determinism."""
    root = (corpus_root / "train" / "LID_Features" / "plp8f0mvsdd" / "LRE03").resolve()
    records: list[dict[str, str]] = []
    for path in sorted(root.glob("*.plp8f0mvsdd")):
        lang = path.name.split("_", 1)[0]
        if lang not in _CLASS_OF:
            continue
        records.append({"filename": str(path), "refseg": str(ref_stm), "lang": lang, "dial": "non", "weight": "1.0", "file_id": "1.0"})
    return records


def _records_from_localized(localized_csv: Path, ref_stm: Path) -> list[dict[str, str]]:
    """Read a `localize_listing` output back into records with the refseg REPLACED by the
    synthesized whole-file-speech STM (the 2015 listing's refseg is the empty shared
    `train/seg.xml`; LID training keys off the language label, not the seg reference)."""
    from speech.batching import read_listing

    out: list[dict[str, str]] = []
    for rec in read_listing(localized_csv):
        if rec["lang"] not in _CLASS_OF:
            continue
        out.append({**rec, "refseg": str(ref_stm), "dial": "non"})
    return out


def _write_listing_rows(path: Path, records: Sequence[dict[str, str]]) -> None:
    """Write records as the 6-field `;`-separated fileslisting the engine parses
    (`filename;refseg;lang;dial;weight;file_id;`)."""
    path.write_text("".join(f"{r['filename']};{r['refseg']};{r['lang']};{r['dial']};{r.get('weight', '1.0')};{r.get('file_id', '1.0')};\n" for r in records))


def write_lre_mapping_12(path: Path) -> None:
    """The 12-class `language2classmapping` (alphabetical, class 0..11, dialect `non`) --
    identical output to `dataprep.lre.write_lre_mapping`, inlined here against the same
    `_LANGS` order so the driver has no import-order coupling to that module's constant."""
    path.write_text("".join(f"{lang};non;{i}\n" for i, lang in enumerate(_LANGS)))


def assemble_flat_config(
    flat: dict[str, str],
    *,
    fileslisting: str,
    mapping: str,
    sad_seed: str,
    lid_seed: str,
    lanes: int,
) -> dict[str, str]:
    """Overlay the runtime corpus/weight/lane keys onto the TOML-flattened base config
    (`speech_rs.load_toml_config` output). PURE -- no engine, no I/O -- so the unit tests
    exercise it with a synthetic flat dict. `dict(flat)` preserves the base key order
    (byte-stable), and each overwritten key already exists in the base so its position is
    kept. `numOuterThreads` = `lanes` records the N-lane fold width in the config the engine
    reads (deterministic-but-N-dependent, the R6-4a model; RESULTS.md fixes N per run)."""
    cfg = dict(flat)
    cfg["fileslisting"] = fileslisting
    cfg["language2classmapping"] = mapping
    cfg["BLSTM_weightsFile"] = sad_seed
    cfg["BLSTM_LID_weightsFile"] = lid_seed
    cfg["numOuterThreads"] = str(lanes)
    return cfg


def _config_text(cfg: dict[str, str]) -> str:
    return "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"


def _generate_seed_packs(flat: dict[str, str], out_dir: Path, seed: int, init_scheme: str, forget_bias_one: bool) -> None:
    """Write valid seed weight packs the engine loads at construction (`BLSTM_weightsFile`/
    `BLSTM_LID_weightsFile`). `train_modern` re-inits from scratch and overrides these, but
    the engine still needs a loadable, correctly-sized pack to build the net -- so this
    mirrors the phase-5 twin smoke's committed-seed-pack pattern, generated at runtime here
    (no corpus-derived bytes) via the SAME `init_weights` the training loop uses. Both nets
    are drawn from one shared `Generator` in `[sad, lid]` order (deterministic).

    `init_scheme`/`forget_bias_one` MUST match the `ModernTrainParams` the training call
    actually uses (Task 8 review fix -- previously hardcoded "xavier"/True regardless): the
    seed pack doubles as the SCORED `score_init` baseline, so a mismatched scheme here would
    silently compare a trained net (e.g. a `--init-scheme he` run) against an init baseline
    drawn under a different scheme than the one training actually started from."""
    rng = np.random.default_rng(seed)
    for prefix, name in (("BLSTM", "sad_seed.bin"), ("BLSTM_LID", "lid_seed.bin")):
        pack = init_weights(nnet_spec(flat, prefix), rng, init_scheme, forget_bias_one)[0]  # type: ignore[arg-type]
        write_bin(pack.shape[0], 1, pack, out_dir / name)


# --------------------------------------------------------------------------------------- #
# Scoring (delegates to evaluate.py's .scr writer + the Task-5 LID metrics)
# --------------------------------------------------------------------------------------- #


def _score_packs_on_test(
    eval_state: RunState,
    sad_pack: Path,
    lid_pack: Path,
    score_dir: Path,
    test_records: Sequence[dict[str, str]],
) -> tuple[float | None, float | None, Path]:
    """Score one `[sad, lid]` weight-pack pair on the held-out test split end to end:
    `evaluate` -> per-file `.scr` -> `read_scr_scores` -> `lid_error` + `cavg`. `evaluate`
    expects the legacy `sad_weights.bin`/`lid_weights.bin` pack names in a checkpoint dir, so
    the two packs (a trained `best_<net>.bin` or the untrained `<net>_seed.bin`) are copied
    in under those names first -- letting the SAME scorer measure both the trained model and
    its own from-scratch init on the identical test set (the direction-safe improvement).

    Returns `(lid_error_pct, cavg, scores_dir)`; the metrics are `None` only if the test
    split produced no `.scr` files (a structurally empty held-out set)."""
    from speech.drivers.test import _class_keys, evaluate

    score_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(sad_pack, score_dir / "sad_weights.bin")
    shutil.copy(lid_pack, score_dir / "lid_weights.bin")
    # Nested under score_dir (already distinct per pass -- "score_trained" vs "score_init")
    # so the trained and init `.scr` outputs never share one directory (Task 8 review fix:
    # `evaluate`'s default output dir is fixed per `eval_state`, so two calls against the
    # SAME eval_state used to clobber each other's `.scr` files on disk).
    scores_dir = evaluate(eval_state, score_dir, scores_dir=score_dir / "scores")

    mapping_path = Path(eval_state.config_path).parent / eval_state.base_config["language2classmapping"]
    class_keys = _class_keys(mapping_path)
    scores, files = read_scr_scores(scores_dir, class_keys)
    if scores.shape[0] == 0:
        return None, None, scores_dir

    # Align each scored file (`.scr` stem = the cep basename) to its true class via the test
    # records, so `refs[i]` shares the alphabetical class base with the score columns.
    lang_by_name = {Path(r["filename"]).name: r["lang"] for r in test_records}
    refs = np.array([_CLASS_OF[lang_by_name[f]] for f in files], dtype=np.int_)
    return lid_error(scores, refs), cavg(scores, refs), scores_dir


# --------------------------------------------------------------------------------------- #
# Run metadata
# --------------------------------------------------------------------------------------- #


def write_run_metadata(out_dir: Path, meta: dict[str, object]) -> Path:
    """Record the run's reproducibility metadata (seed, lanes, subset spec, config hash,
    split counts, arm) as `out_dir/run_metadata.json`. Returns the path. Pure I/O -- unit
    tested directly."""
    path = out_dir / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def _config_hash(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


# --------------------------------------------------------------------------------------- #
# The arm launcher
# --------------------------------------------------------------------------------------- #


@dataclass
class BaselineResult:
    """The outcome of one `run_baseline` call -- the metrics + the artefact locations the
    gate asserts on and the launcher logs."""

    arm: str
    out_dir: Path
    checkpoint_dir: Path
    metadata_path: Path
    epochs_run: int
    best_epoch: int
    best_val_cost: float
    first_val_cost: float
    val_costs: list[float] = field(default_factory=list)
    train_costs: list[float] = field(default_factory=list)
    lid_error: float | None = None
    cavg: float | None = None
    init_lid_error: float | None = None  # untrained-init held-out error (scored iff score_init)
    init_cavg: float | None = None
    scores_dir: Path | None = None
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0


def run_baseline(
    arm: str,
    corpus_root: Path,
    out_dir: Path,
    *,
    resume: bool = False,
    lanes: int = 1,
    subset: int | None = None,
    dry_run: bool = False,
    seed: int = 0,
    epochs: int = 20,
    patience: int = 6,
    steps_per_epoch: int = 25,
    init_scheme: str = "xavier",
    lre_listing: Path | None = None,
    valid_size: int = 12,
    test_size: int = 48,
    minibatch: int = 0,
    score_init: bool = False,
    console: Console | None = None,
    _train_fn: Callable[..., object] | None = None,
) -> BaselineResult:
    """Run one baseline training arm end to end: prepare the corpus listings, assemble the
    config, seeded from-scratch init (or resume), `train_modern` with the moving NNCostSeg
    validation signal, then score a held-out slice (LID: `.scr` -> `lid_error` + `cavg`).

    `subset`: the TRAIN sample size (per-language proportional, seeded) -- the CI-gate regime
    (< 10 min); `None` trains on the whole corpus (the launcher). `valid_size`/`test_size`
    are SEPARATE, disjoint held-out sizes -- decoupled from `subset` so a big, cheap-to-score
    test set gives a stable LID error while training stays small (scoring is
    O(test x forward); training is O(subset x steps x epochs)). `dry_run`: a 1-step smoke (a
    tiny train subset, 1 epoch, 1 step) still scored end to end. `resume`: continue from
    `out_dir/checkpoint`'s `last_*.bin`. `lanes`: the engine's `numOuterThreads` fold width
    (recorded in metadata; N=1 is the deterministic parity mode). `lre_listing`: localize
    this 2015 listing instead of deriving from the corpus tree. `_train_fn` injects a stub
    `train_modern` for tests."""
    if arm not in _ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm!r}; known arms: {sorted(_ARM_CONFIGS)}")
    if arm != "lid-features":
        raise NotImplementedError(f"arm {arm!r} lands in a later task (only lid-features is wired in Task 8)")

    console = console or Console()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # __file__ = <repo>/src/python/speech/drivers/baseline.py -> parents[4] = <repo>.
    repo_root = Path(__file__).resolve().parents[4]
    toml_path = repo_root / _ARM_CONFIGS[arm]

    if dry_run:
        subset, epochs, steps_per_epoch, patience, test_size, valid_size = (subset or 24), 1, 1, 99, min(test_size, 24), min(valid_size, 12)

    t0 = time.time()
    console.log(f"[bold]baseline {arm}[/bold]: corpus={corpus_root} out={out_dir} subset={subset} lanes={lanes} seed={seed} dry_run={dry_run}")

    # --- 1. listings + mapping + reference (synthesized under out_dir) --------------------
    ref_stm = out_dir / "ref_speech.stm"
    ref_stm.write_text(_SPEECH_STM)
    if lre_listing is not None:
        localized = out_dir / "localized_lre03.csv"
        rep = localize_listing(Path(lre_listing), corpus_root, localized)
        console.log(f"localized {rep.rows_found}/{rep.rows_total} rows ({rep.rows_missing} missing)")
        records = _records_from_localized(localized, ref_stm)
    else:
        records = derive_lid_features_records(corpus_root, ref_stm)
    if not records:
        raise RuntimeError(f"no LID features found under {corpus_root} (expected train/LID_Features/plp8f0mvsdd/LRE03/*.plp8f0mvsdd)")
    console.log(f"corpus records: {len(records)}")

    train_rec, valid_rec, test_rec = stratified_splits(records, subset, valid_size, test_size, seed)
    console.log(f"split: train={len(train_rec)} valid={len(valid_rec)} test={len(test_rec)}")

    train_lst = out_dir / "lre03_lid_features_train.flst"
    valid_lst = out_dir / "lre03_lid_features_valid.flst"
    test_lst = out_dir / "lre03_lid_features_test.flst"
    _write_listing_rows(train_lst, train_rec)
    _write_listing_rows(valid_lst, valid_rec)
    _write_listing_rows(test_lst, test_rec)
    mapping = out_dir / "language2classmapping_lre12.csv"
    write_lre_mapping_12(mapping)

    # --- 2. training params + config assembly + seed packs -------------------------------
    # Built here (not down in step 4) so `_generate_seed_packs` can be handed the SAME
    # init_scheme/forget_bias_one the training call below will use (Task 8 review fix).
    params = ModernTrainParams(
        epochs=epochs,
        patience=patience,
        steps_per_epoch=steps_per_epoch,
        valid_listing=valid_lst.name if valid_rec else None,
        val_metric="nn_cost_seg",
        minibatch=minibatch,
        nb_classes=len(_LANGS),
        multilingual=minibatch > 0,
        init_scheme=init_scheme,  # type: ignore[arg-type]
        init_seed=seed,
        resume_from=str(out_dir / "checkpoint") if resume else None,
    )

    import speech_rs  # local: the pyo3 module is only needed on the engine path

    flat = {k: str(v) for k, v in speech_rs.load_toml_config(str(toml_path)).items()}
    cfg = assemble_flat_config(flat, fileslisting=train_lst.name, mapping=mapping.name, sad_seed="sad_seed.bin", lid_seed="lid_seed.bin", lanes=lanes)
    cfg_text = _config_text(cfg)
    base_config = out_dir / "base.config"
    base_config.write_text(cfg_text)
    _generate_seed_packs(cfg, out_dir, seed, params.init_scheme, params.forget_bias_one)

    # --- 3. run metadata -----------------------------------------------------------------
    metadata_path = write_run_metadata(
        out_dir,
        {
            "arm": arm,
            "seed": seed,
            "lanes": lanes,
            "subset": subset,
            "dry_run": dry_run,
            "epochs": epochs,
            "patience": patience,
            "steps_per_epoch": steps_per_epoch,
            "minibatch": minibatch,
            "init_scheme": init_scheme,
            "val_metric": "nn_cost_seg",
            "config_hash": _config_hash(cfg_text),
            "config_toml": str(toml_path.relative_to(repo_root)),
            "corpus_root": str(corpus_root),
            "n_train": len(train_rec),
            "n_valid": len(valid_rec),
            "n_test": len(test_rec),
            "resume": resume,
        },
    )

    # --- 4. train (from scratch or resume) -----------------------------------------------
    state = RunState.from_config(base_config, out_dir)
    train = _train_fn if _train_fn is not None else train_modern
    console.log(f"training: {epochs} epochs x {steps_per_epoch} steps (patience {patience})")
    res = train(state, seed, params)  # type: ignore[operator]
    val_costs = [r.val_cost for r in res.history]  # type: ignore[attr-defined]
    train_costs = [r.train_cost for r in res.history]  # type: ignore[attr-defined]
    console.log(f"trained {res.epochs_run} epochs; best_epoch={res.best_epoch} best_val={res.best_val_cost:.5f}")  # type: ignore[attr-defined]

    # --- 5. score the held-out split (trained; optionally the untrained init too) ---------
    lid_err: float | None = None
    cavg_val: float | None = None
    init_err: float | None = None
    init_cavg_val: float | None = None
    scores_dir: Path | None = None
    ckpt = Path(res.checkpoint_dir)  # type: ignore[attr-defined]
    if test_rec:
        eval_base = out_dir / "eval_base.config"
        eval_cfg = dict(cfg)
        eval_cfg["fileslisting"] = test_lst.name
        eval_base.write_text(_config_text(eval_cfg))
        eval_state = RunState.from_config(eval_base, out_dir)
        lid_err, cavg_val, scores_dir = _score_packs_on_test(eval_state, ckpt / "best_sad.bin", ckpt / "best_lid.bin", out_dir / "score_trained", test_rec)
        chance = 100.0 * (1.0 - 1.0 / len(_LANGS))
        if lid_err is not None and cavg_val is not None:
            console.log(f"held-out: lid_error={lid_err:.2f}% (chance {chance:.2f}%) cavg={cavg_val:.4f}")
        else:
            console.log("held-out: no .scr scores produced (empty test split)")
        if score_init:
            init_err, init_cavg_val, _ = _score_packs_on_test(eval_state, out_dir / "sad_seed.bin", out_dir / "lid_seed.bin", out_dir / "score_init", test_rec)
            if init_err is not None and lid_err is not None:
                console.log(f"init baseline: lid_error={init_err:.2f}% (improvement {init_err - lid_err:+.2f}pt)")

    console.log(f"[green]done[/green] in {time.time() - t0:.1f}s")
    return BaselineResult(
        arm=arm,
        out_dir=out_dir,
        checkpoint_dir=ckpt,
        metadata_path=metadata_path,
        epochs_run=res.epochs_run,  # type: ignore[attr-defined]
        best_epoch=res.best_epoch,  # type: ignore[attr-defined]
        best_val_cost=res.best_val_cost,  # type: ignore[attr-defined]
        first_val_cost=val_costs[0] if val_costs else float("nan"),
        val_costs=val_costs,
        train_costs=train_costs,
        lid_error=lid_err,
        cavg=cavg_val,
        init_lid_error=init_err,
        init_cavg=init_cavg_val,
        scores_dir=scores_dir,
        n_train=len(train_rec),
        n_valid=len(valid_rec),
        n_test=len(test_rec),
    )


# --------------------------------------------------------------------------------------- #
# CLI: `python -m speech.drivers.baseline lid-features --corpus-root ... --out-dir ...`
# --------------------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """The `baseline <arm>` argument parser -- also mounted as the `baseline` subcommand in
    `speech.cli`. Unit-tested for arg wiring (no run)."""
    parser = argparse.ArgumentParser(prog="baseline", description="Phase 6 from-scratch baseline training arms")
    parser.add_argument("arm", choices=sorted(_ARM_CONFIGS), help="the training arm")
    parser.add_argument("--corpus-root", type=Path, required=True, help="the LRE03/07 corpus root (data/LRE03-LRE07)")
    parser.add_argument("--out-dir", type=Path, required=True, help="run directory for listings/config/checkpoints/scores")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1, help="engine numOuterThreads fold width (N=1 is the parity mode)")
    parser.add_argument("--subset", type=int, default=None, help="stratified subset size (omit for the full corpus)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    parser.add_argument("--init-scheme", choices=("xavier", "he"), default="xavier")
    parser.add_argument("--lre-listing", type=Path, default=None, help="localize this 2015 listing instead of deriving from the corpus tree")
    parser.add_argument("--resume", action="store_true", help="continue from out_dir/checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="1-step smoke: tiny subset, 1 epoch, 1 step, still scored")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args -> `run_baseline`. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    run_baseline(
        args.arm,
        args.corpus_root,
        args.out_dir,
        resume=args.resume,
        lanes=args.lanes,
        subset=args.subset,
        dry_run=args.dry_run,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        steps_per_epoch=args.steps_per_epoch,
        init_scheme=args.init_scheme,
        lre_listing=args.lre_listing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
