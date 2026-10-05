"""Evaluation driver: the `.scr` score outputs.

Ported from `Test_BLSTM.m:246-270`: run the trained net over the corpus, then for each
file emit a `.scr` file of per-language softmax scores. The score law (`:252-266`) is:
`exp(score/100)`, normalize by `max(1e-3, sum)`, sort DESCENDING, and write one
`filename lang-dial %15.15f` line per class (lang = key[:3], dial = key[-3:]).

`write_scores` is the pure, golden-tested core; `evaluate` is the thin engine-driving
orchestrator around it (not golden-pinned -- the exit gate does not exercise Test).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from speech.drivers.state import RunState
from speech.drivers.train import _net_names
from speech.weight_bridge import read_weight_vector

F64 = np.float64


def write_scores(scores_row: NDArray[np.float64], class_keys: list[str], filename: str) -> str:
    """The `.scr` text for one file (`Test_BLSTM.m:252-266`). `scores_row[k]` is the raw
    LID score for class `k`; `class_keys[k]` is its language key (lang = first 3 chars,
    dial = last 3). Softmax `exp(s/100)/max(1e-3, sum)`, sort DESC, one line per class."""
    s = np.exp(np.asarray(scores_row, dtype=F64) / 100.0)
    s = s / max(1e-3, float(np.sum(s)))
    order = np.argsort(-s, kind="stable")  # sortrows(...,-1): descending
    lines = []
    for k in order:
        key = class_keys[int(k)]
        lang = key[:3]
        dial = key[-3:]
        lines.append(f"{filename} {lang}-{dial} {float(s[int(k)]):.15f}")
    return "\n".join(lines) + "\n"


def _decode_lid_scores(error_vad_row: NDArray[np.float64]) -> NDArray[np.float64]:
    """The per-class LID scores from an algo-6 result row: columns `16:-2`, in-band
    decoded (`>150 -> value - 200` marks the target class; others are raw).

    DEGRADES SILENTLY on a single-net result row (algo 3/4/5 -- no confusion columns
    inserted, `engine/bag_of_processors.rs` fixed 18-column width): `[16:-2]` on an
    18-wide row is `[16:16]`, an empty slice, not an `IndexError`. `.scr` scoring is a
    Twin-only (algo 6) artifact of `Test_BLSTM.m`; nothing here guards against calling
    it on a non-Twin checkpoint."""
    scores = np.asarray(error_vad_row[16:-2], dtype=F64)
    return np.where(scores > 150.0, scores - 200.0, scores)


def _class_keys(mapping_path: Path) -> list[str]:
    """Class keys in legacy `keys(langMapConf)` order, from the `lang;dial;classid`
    language mapping: the composed `lang_dial` key (`processListing.m:10`, note the
    underscore -- it is what `write_scores`' `key[-3:]` dial slice sees for a 2-char
    dial, e.g. 'aaa_11' -> 'aaa-_11'), unique (containers.Map), sorted ALPHABETICALLY
    (MATLAB `keys()` = ASCII byte order; Python's code-point sort agrees on ASCII keys).
    The mapping's class-id column is IGNORED, exactly like the legacy writer:
    `processListing.m:85-88` overwrites langMapConf's values with alphabetical
    positions, so `Test_BLSTM.m:249/:263` labels score column ii with the ii-th
    alphabetical key regardless of the file's ids. Pinned byte-exact vs the Octave
    golden (`tests/test_phase4d_scr.py`); closes the Phase 4c class-id-order
    divergence."""
    keys: set[str] = set()
    for line in mapping_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 3:
            continue
        keys.add(parts[0] + "_" + parts[1])
    return sorted(keys)


def resolve_checkpoint_packs(checkpoint: Path, algo: int) -> list[Path]:
    """The weight packs `evaluate` injects, one per net (`[sad]`, or `[sad, lid]` for the
    algo-6 Twin). Two writers, two naming schemes: the legacy `train` driver writes
    `<net>_weights.bin`, `train_modern` writes `best_<net>.bin` (its `last_<net>.bin` is a
    resume point, not a scoring checkpoint). The legacy name wins when both exist.
    Raises `FileNotFoundError` naming the net and both accepted filenames when neither is
    there (#29: the old `all(exists)` guard skipped `set_weights` and scored the config's
    seed pack in silence)."""
    packs = []
    for net in _net_names(algo):
        candidates = [checkpoint / f"{net}_weights.bin", checkpoint / f"best_{net}.bin"]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            names = " or ".join(c.name for c in candidates)
            raise FileNotFoundError(f"checkpoint {checkpoint} has no {net} weight pack: expected {names}")
        packs.append(found)
    return packs


def evaluate(state: RunState, checkpoint: Path, scores_dir: Path | None = None) -> Path:
    """Score the corpus with the checkpoint weights and write per-file `.scr` outputs to
    `scores_dir` (default `<out_dir>/scores/`). Returns the scores directory.

    `scores_dir`: override the output directory (Task 8 review fix). Callers that
    `evaluate` more than one checkpoint against the SAME `state` -- e.g. a trained pack and
    its own untrained-init baseline -- must pass DISTINCT dirs, or the second call's `.scr`
    files silently overwrite the first's on disk (the default `<out_dir>/scores/` is fixed
    per `state`, not per call).

    Only meaningful for the Twin (algo 6): `.scr` is a LID artifact, and a single-net
    checkpoint (algo 3/4/5, post-Task-9 generalized drivers) degrades SILENTLY, not with
    an error -- `_decode_lid_scores` slices an empty LID-score column range out of the
    18-wide non-LID result row, so `write_scores` gets an empty `scores_row` and writes a
    near-blank `.scr` file (zero score lines, just a trailing newline) per file. No gate
    stops `evaluate` from being pointed at a non-algo-6 checkpoint."""
    workdir = Path(state.config_path).parent
    cfg = dict(state.base_config)
    cfg["BLSTM_BackPropagationActivated"] = "false"
    if state.algo == 6:
        cfg["BLSTM_LID_BackPropagationActivated"] = "false"
    config_text = "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"

    # Fast processors load weights only at construction from the config's weight keys and
    # `set_weights` bails on them (T6b, see `_score_sad_pack_on_test`), so injection is
    # exact-only; a missing pack raises here, before the engine (or anything on disk) exists.
    inject = cfg.get("Inference_Path", "exact") != "fast"
    packs = resolve_checkpoint_packs(Path(checkpoint), state.algo) if inject else []

    import speech_rs  # local: the pyo3 module is only needed on the engine path

    scores_dir = Path(scores_dir) if scores_dir is not None else Path(state.out_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    prev = Path.cwd()
    os.chdir(workdir)
    try:
        (workdir / "_eval.config").write_text(config_text)
        engine = speech_rs.Engine(["_eval.config"], "-m")
        if inject:
            engine.set_weights(0, [list(read_weight_vector(p)) for p in packs])
        engine.run()
        results = np.asarray(engine.results_matrix(), dtype=F64)
    finally:
        os.chdir(prev)

    keys = _class_keys(workdir / state.base_config["language2classmapping"])
    for row in results:
        file_idx = int(row[0]) - 1
        filename = Path(state.listing[file_idx]["filename"]).name if 0 <= file_idx < len(state.listing) else f"file{file_idx}"
        scores = _decode_lid_scores(row[3:])
        (scores_dir / f"{filename}.scr").write_text(write_scores(scores, keys, filename))
    return scores_dir
