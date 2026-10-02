"""The Phase 7 bench legs, staged from the repository instead of a paragraph of RESULTS.md prose.

Four legs, two recipes. The SAD legs are `tests/reference_data/phase4a/tier2_spectral.config`
with the tuple-A pack and a one-file unscored listing (the `tests/phase7_bench.rs::
stage_bench_config` recipe): the committed 60 s stereo fixture, or the sorted-first corpus wav.
The LID legs are `tests/reference_data/phase4b/twin_mode7.config` (the Phase 4b flagship Twin,
Mode 7, backprop already off) over the sorted-first corpus phSeq or cep file. Corpus files are
selected at runtime as the lexicographically first match under their subtree (a sort over a
glob, no content-based picking) and no filename ever leaves the staging directory: the ledger
record carries the leg's label and the engine's `audio_s`, nothing else about the file.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from speech.ledger.schema import CORPUS_ROOT_NAME, REPO, Lineage

FIXTURES = REPO / "tests" / "reference_data"
CORPUS_ROOT = REPO / "data" / CORPUS_ROOT_NAME


@dataclass(frozen=True)
class Leg:
    label: str
    display: str
    lineage: Lineage | None
    corpus: bool
    stage: Callable[[Path, Path], Path]


def _sad_config(stage_dir: Path, audio: Path) -> Path:
    """`tier2_spectral.config` + the tuple-A pack over one unscored file, the cap lifted so the
    whole file is processed. Input paths are absolute; the two OUTPUT keys (`Dump_Directory`,
    `multiConfigResultsOutputFile`) stay relative, as in the Rust recipe, because the engine
    fails on an absolute value there, so the binary runs with the staging directory as cwd."""
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
        "multiConfigResultsOutputFile bench_result.mat\n"
        "Dump_Directory vrcts_bench\n"
        "Neural_Networks_BackPropagation_Epochs 0\n"
        "BLSTM_BackPropagationActivated false\n"
    )
    cfg = stage_dir / "bench_sad.config"
    cfg.write_text(base + tail)
    return cfg


def _twin_config(stage_dir: Path, features: Path, file_type: int) -> Path:
    """`twin_mode7.config` over one precomputed-feature file (phSeq `File_Type 1`, cep
    `File_Type 2`), the committed LID pack and mapping made absolute; the listing row names the
    mapping's first language so the Twin has a class to score against."""
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
        "multiConfigResultsOutputFile bench_result.mat\n"
        "Dump_Directory vrcts_bench\n"
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


LEGS: dict[str, Leg] = {
    leg.label: leg
    for leg in (
        Leg("phase7_60s", "SAD 60 s fixture (stereo)", "v1", False, stage_fixture_60s),
        Leg("phase7_sad_corpus", "SAD corpus-gated (mono)", "v1", True, stage_corpus_sad),
        Leg("phase7_lid_phseq", "LID phSeq corpus-gated (Twin M7)", None, True, stage_corpus_phseq),
        Leg("phase7_lid_cep", "LID cep corpus-gated (Twin M7)", None, True, stage_corpus_cep),
    )
}
