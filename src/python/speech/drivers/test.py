"""Evaluation driver: the `.scr` score outputs.

Ported from `Test_BLSTM.m:246-270`: run the trained net over the corpus, then for each
file emit a `.scr` file of per-language softmax scores. The score law (`:252-266`) is:
`exp(score/100)`, normalize by `max(1e-3, sum)`, sort DESCENDING, and write one
`filename lang-dial %15.15f` line per class (lang = key[:3], dial = key[-3:]).

`write_scores` is the pure, golden-tested core; `evaluate` is the thin orchestrator around
it (not golden-pinned -- the exit gate does not exercise Test): one forward-only `FoldRun`
at the checkpoint's packs, then one `.scr` per channel result.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from speech.drivers.state import RunState
from speech.drivers.train import _net_names
from speech.fold_run import FoldRun
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
    """A checkpoint directory's weight packs, one per net (`[sad]`, or `[sad, lid]` for the
    algo-6 Twin) -- what the `test` subcommand hands `evaluate`. Two writers, two naming
    schemes: the legacy `train` driver writes `<net>_weights.bin`, `train_modern` writes
    `best_<net>.bin` (its `last_<net>.bin` is a resume point, not a scoring checkpoint). A
    scheme resolves only when it holds EVERY net, so a `[sad, lid]` pair is never assembled
    from two different runs; the legacy scheme wins when both are complete. Raises
    `FileNotFoundError` naming both accepted pack sets when neither is complete (#29: the old
    `all(exists)` guard skipped `set_weights` and scored the config's seed pack in silence)."""
    nets = _net_names(algo)
    schemes = [[checkpoint / f"{net}_weights.bin" for net in nets], [checkpoint / f"best_{net}.bin" for net in nets]]
    for packs in schemes:
        if all(p.exists() for p in packs):
            return packs
    expected = " or ".join(" + ".join(p.name for p in packs) for packs in schemes)
    raise FileNotFoundError(f"checkpoint {checkpoint} has no complete weight-pack set: expected {expected}")


def evaluate(state: RunState, packs: Sequence[Path], scores_dir: Path | None = None) -> Path:
    """Score the corpus at `packs` (`[sad]`, or `[sad, lid]` for the Twin -- a checkpoint
    directory resolves to them through `resolve_checkpoint_packs`) and write per-file `.scr`
    outputs to `scores_dir` (default `<out_dir>/scores/`). Returns the scores directory.

    `scores_dir`: override the output directory (Task 8 review fix). Callers that
    `evaluate` more than one pack set against the SAME `state` -- e.g. a trained pack and
    its own untrained-init baseline -- must pass DISTINCT dirs, or the second call's `.scr`
    files silently overwrite the first's on disk (the default `<out_dir>/scores/` is fixed
    per `state`, not per call).

    Only meaningful for the Twin (algo 6): `.scr` is a LID artifact, and a single-net
    checkpoint (algo 3/4/5, post-Task-9 generalized drivers) degrades SILENTLY, not with
    an error -- a run without LID has an `n x 0` `lid_scores` block, so `write_scores`
    gets an empty `scores_row` and writes a near-blank `.scr` file (zero score lines, just
    a trailing newline) per file. No gate stops `evaluate` from being pointed at a
    non-algo-6 checkpoint."""
    workdir = Path(state.config_path).parent
    # The packs are read first, against the caller's cwd, so a missing one raises before the
    # engine (or anything on disk) exists. The fold run owns the fast-path injection.
    nets = [read_weight_vector(p) for p in packs]
    fold = FoldRun(state.base_config, workdir, backprop=False)

    scores_dir = Path(scores_dir) if scores_dir is not None else Path(state.out_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    results = fold.run(nets).results

    keys = _class_keys(workdir / state.base_config["language2classmapping"])
    for i in range(len(results)):
        file_idx = int(results.file[i])
        filename = Path(state.listing[file_idx]["filename"]).name if 0 <= file_idx < len(state.listing) else f"file{file_idx}"
        (scores_dir / f"{filename}.scr").write_text(write_scores(results.lid_scores[i], keys, filename))
    return scores_dir
