"""The bench legs, staged from the repository instead of a paragraph of RESULTS.md prose.

The four Phase 7 legs, two recipes. The SAD legs are `tests/reference_data/phase4a/tier2_spectral.config`
with the tuple-A pack and a one-file unscored listing (the `tests/phase7_bench.rs::
stage_bench_config` recipe): the committed 60 s stereo fixture, or the sorted-first corpus wav.
The LID legs are `tests/reference_data/phase4b/twin_mode7.config` (the Phase 4b flagship Twin,
Mode 7, backprop already off) over the sorted-first corpus phSeq or cep file. Corpus files are
selected at runtime as the lexicographically first match under their subtree (a sort over a
glob, no content-based picking) and no filename ever leaves the staging directory: the ledger
record carries the leg's label and the engine's `audio_s`, nothing else about the file.

The two gradient-fold legs (issue #154) are exact-only: one exact gradient fold of the SAD v1 arm
as a training step runs it, at one lane and at four. The launcher's own `prepare` builds the run
directory (the first `FOLD_FILES` rows of the seeded train split, the 120 s cap of the full-run
protocol, the from-scratch seed pack) and `FoldRun` supplies the gradient overlay (backprop on,
`Epochs 0`) with its path keys resolved, so the staged config is the map the seam gets. They run
with `LIVE_PEAK_ENV`, so their `maxrss_mb` is the fold's live peak; a third leg is the one-lane
fold with the allocator's cache set on (`LARGE_CACHE_ENV`), the record of what that switch removes.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

from speech.ledger.schema import CORPUS_ROOT_NAME, REPO, Lineage

FIXTURES = REPO / "tests" / "reference_data"
CORPUS_ROOT = REPO / "data" / CORPUS_ROOT_NAME


BenchPath = Literal["exact", "fast"]
BOTH_PATHS: tuple[BenchPath, ...] = ("exact", "fast")


@dataclass(frozen=True)
class Leg:
    label: str
    display: str
    lineage: Lineage | None
    stage: Callable[[Path, Path], Path]
    paths: tuple[BenchPath, ...] = BOTH_PATHS
    env: tuple[tuple[str, str], ...] = ()


def _sad_config(stage_dir: Path, audio: Path) -> Path:
    """`tier2_spectral.config` + the tuple-A pack over one unscored file, the cap lifted so the
    whole file is processed. Every path key is absolute, the one OUTPUT key (`Dump_Directory`)
    included, so the binary runs from any cwd."""
    pack = stage_dir / "NNweights_config1.bin"
    shutil.copyfile(FIXTURES / "phase0" / "NNweights_config1.bin", pack)
    listing = stage_dir / "bench_listing.csv"
    listing.write_text(f"{audio}\n")
    mapping = stage_dir / "bench_mapping.csv"
    mapping.write_text("")
    (stage_dir / "vrcts_bench").mkdir(exist_ok=True)
    base = (FIXTURES / "phase4a" / "tier2_spectral.config").read_text()
    tail = (
        "\n# ==== bench staging (speech.ledger.stage, last-wins) ====\n"
        "Audio_max_duration 3600000\n"
        f"fileslisting {listing}\n"
        f"language2classmapping {mapping}\n"
        f"BLSTM_weightsFile {pack}\n"
        f"Dump_Directory {stage_dir / 'vrcts_bench'}\n"
        "Neural_Networks_BackPropagation_Epochs 0\n"
        "BLSTM_BackPropagationActivated false\n"
    )
    cfg = stage_dir / "bench_sad.config"
    cfg.write_text(base + tail)
    return cfg


def _twin_config(stage_dir: Path, features: Path, file_type: int) -> Path:
    """`twin_mode7.config` over one precomputed-feature file (phSeq `File_Type 1`, cep
    `File_Type 2`), every path key absolute (the committed LID pack and mapping, the output
    key); the listing row names the mapping's first language so the Twin has a class to score
    against."""
    mapping = FIXTURES / "phase4b" / "languagemapping_lid7.csv"
    lang, dial = mapping.read_text().splitlines()[0].split(";")[:2]
    listing = stage_dir / "bench_listing.csv"
    listing.write_text(f"{features};;{lang};{dial};1.0\n")
    (stage_dir / "vrcts_bench").mkdir(exist_ok=True)
    base = (FIXTURES / "phase4b" / "twin_mode7.config").read_text()
    tail = (
        "\n# ==== bench staging (speech.ledger.stage, last-wins) ====\n"
        "numOuterThreads 1\n"
        f"BLSTM_LID_weightsFile {FIXTURES / 'phase4b' / 'LID_bestNNWeight_1.bin'}\n"
        f"language2classmapping {mapping}\n"
        f"fileslisting {listing}\n"
        f"File_Type {file_type}\n"
        f"Dump_Directory {stage_dir / 'vrcts_bench'}\n"
    )
    cfg = stage_dir / "bench_twin.config"
    cfg.write_text(base + tail)
    return cfg


def sorted_first(root: Path, pattern: str) -> Path:
    """The lexicographically first file matching `pattern` under `root` (recursive), the
    Phase 7 selection rule. Raises when the subtree is absent or empty."""
    if not root.is_dir():
        raise FileNotFoundError(f"corpus subtree {root.relative_to(root.anchor)} is absent (the corpus is local and licensed)")
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} under {root.name}/")
    return matches[0]


def stage_fixture_60s(stage_dir: Path, corpus_root: Path) -> Path:
    wav = stage_dir / "prcts_excerpt.wav"
    shutil.copyfile(FIXTURES / "phase4d" / "prcts_excerpt.wav", wav)
    return _sad_config(stage_dir, wav)


def stage_corpus_sad(stage_dir: Path, corpus_root: Path) -> Path:
    return _sad_config(stage_dir, sorted_first(corpus_root / "train" / "audio", "*.wav"))


def stage_corpus_phseq(stage_dir: Path, corpus_root: Path) -> Path:
    return _twin_config(stage_dir, sorted_first(corpus_root / "train" / "phSeq", "*.phSeqbis"), 1)


def stage_corpus_cep(stage_dir: Path, corpus_root: Path) -> Path:
    return _twin_config(stage_dir, sorted_first(corpus_root / "train" / "LID_Features", "*.plp8f0mvsdd"), 2)


FOLD_FILES = 128
FOLD_AUDIO_CAP_S = 120.0
#: macOS libmalloc keeps freed large blocks resident, so over many files `maxrss_mb` reads the
#: allocator's cache of per-file transients, not live memory (the `fold_grad_sad_n1_cached` leg
#: against `fold_grad_sad_n1` is the measured gap). A multi-file leg switches the cache off. The
#: variable is ignored off macOS.
LIVE_PEAK_ENV = (("MallocLargeCache", "0"),)
#: The cache on by name (`1` is libmalloc's default), so a caller's exported `0` cannot turn it off.
LARGE_CACHE_ENV = (("MallocLargeCache", "1"),)


def stage_grad_fold(stage_dir: Path, corpus_root: Path, *, lanes: int) -> Path:
    """One exact gradient fold of the SAD v1 arm over `FOLD_FILES` train files at `lanes` lanes
    (issue #154): `prepare` under a fresh `stage_dir/run-*` (it refuses a directory holding a run,
    so a reused stage directory gets a new one), then the `FoldRun` gradient map, written as the
    config the binary reads. The seed pack is the launcher's own from-scratch init (seed 0)."""
    from rich.console import Console  # local: the launcher's imports stay off the ledger's import path

    from speech.drivers.baseline import prepare
    from speech.drivers.spec import BaselineSpec
    from speech.fold_run import FoldRun, _config_text

    spec = BaselineSpec(
        arm="sad",
        corpus_root=corpus_root,
        out_dir=Path(tempfile.mkdtemp(prefix="run-", dir=stage_dir)),
        lanes=lanes,
        subset=FOLD_FILES,
        valid_size=0,
        test_size=0,
        audio_max_duration=FOLD_AUDIO_CAP_S,
    )
    prepared = prepare(spec, console=Console(quiet=True))
    cfg = stage_dir / "bench_fold.config"
    cfg.write_text(_config_text(FoldRun(prepared.cfg, prepared.out_dir, backprop=True).engine_map()))
    return cfg


#: The gradient-fold legs (issue #154), rendered by their own table, never the Phase 7 matrix.
_FOLD = (
    Leg("fold_grad_sad_n1", f"SAD v1 gradient fold ({FOLD_FILES} files, 1 lane)", "v1", partial(stage_grad_fold, lanes=1), ("exact",), LIVE_PEAK_ENV),
    Leg("fold_grad_sad_n4", f"SAD v1 gradient fold ({FOLD_FILES} files, 4 lanes)", "v1", partial(stage_grad_fold, lanes=4), ("exact",), LIVE_PEAK_ENV),
    Leg(
        "fold_grad_sad_n1_cached",
        f"SAD v1 gradient fold ({FOLD_FILES} files, 1 lane, allocator cache on)",
        "v1",
        partial(stage_grad_fold, lanes=1),
        ("exact",),
        LARGE_CACHE_ENV,
    ),
)
FOLD_LEGS = tuple(leg.label for leg in _FOLD)

LEGS: dict[str, Leg] = {
    leg.label: leg
    for leg in (
        Leg("phase7_60s", "SAD 60 s fixture (stereo)", "v1", stage_fixture_60s),
        Leg("phase7_sad_corpus", "SAD corpus-gated (mono)", "v1", stage_corpus_sad),
        Leg("phase7_lid_phseq", "LID phSeq corpus-gated (Twin M7)", None, stage_corpus_phseq),
        Leg("phase7_lid_cep", "LID cep corpus-gated (Twin M7)", None, stage_corpus_cep),
        *_FOLD,
    )
}
