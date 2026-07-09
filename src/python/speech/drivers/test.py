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
    decoded (`>150 -> value - 200` marks the target class; others are raw)."""
    scores = np.asarray(error_vad_row[16:-2], dtype=F64)
    return np.where(scores > 150.0, scores - 200.0, scores)


def _class_keys(mapping_path: Path) -> list[str]:
    """Class keys ordered by class id, from the `lang;dial;classid` language mapping."""
    rows: list[tuple[int, str]] = []
    for line in mapping_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 3:
            continue
        rows.append((int(parts[2]), parts[0] + parts[1]))
    return [key for _cid, key in sorted(rows)]


def evaluate(state: RunState, checkpoint: Path) -> Path:
    """Score the corpus with the checkpoint weights and write per-file `.scr` outputs to
    `<out_dir>/scores/`. Returns the scores directory."""
    import speech_rs  # local: the pyo3 module is only needed on the engine path

    workdir = Path(state.config_path).parent
    ckpt = Path(checkpoint)
    scores_dir = Path(state.out_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(state.base_config)
    cfg["BLSTM_BackPropagationActivated"] = "false"
    cfg["BLSTM_LID_BackPropagationActivated"] = "false"
    config_text = "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"

    prev = Path.cwd()
    os.chdir(workdir)
    try:
        (workdir / "_eval.config").write_text(config_text)
        engine = speech_rs.Engine(["_eval.config"], "-m")
        if (ckpt / "sad_weights.bin").exists() and (ckpt / "lid_weights.bin").exists():
            sad = read_weight_vector(ckpt / "sad_weights.bin")
            lid = read_weight_vector(ckpt / "lid_weights.bin")
            engine.set_weights(0, [list(sad), list(lid)])
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
