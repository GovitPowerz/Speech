"""The Phase 6 license-hygiene rule, made executable: no tracked file names a corpus file.

The rule (CLAUDE.md "Since Phase 6", docs/validation.md): no corpus path, filename or feature
table is committed; fixtures use synthetic pattern-preserving names. Until now it was enforced
by review only. The scanner here reads every tracked text file (a symlink as the target path
git stores for it) and flags a path through the corpus root directory that ends in a concrete
filename. Directory-level globs (`data/LRE03-LRE07/train/audio/**`) and the bare root name are
allowed: RESULTS.md uses both to describe the protocol.

Scope: only paths spelled through the root directory name are caught. Bare corpus basenames,
the 2015 host-prefixed listing paths the legacy goldens carry, binaries, and the scoring
toolkit root (`data/Scoring_LRE15/`, cited by name for its scorer scripts but also holding the
LRE15 per-segment score tables) stay review-only.
"""

import os
import re
import subprocess
from pathlib import Path

from tests.conftest import CORPUS_ROOT

REPO = Path(__file__).resolve().parents[1]

# A path through the corpus root whose last segment is a concrete filename: a stem of word
# characters (so `*`, `**` and `{...}` globs never match) followed by a dot and an extension.
CORPUS_FILE = re.compile(re.escape(CORPUS_ROOT.name) + r"/(?:[\w.-]+/)*[\w-]*\w\.\w+")


def tracked_text_files(repo: Path) -> list[Path]:
    """Every file `git ls-files` reports, minus binaries (git's own heuristic: a NUL in the first 8000 bytes)."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True).stdout
    paths = [repo / rel.decode("utf-8") for rel in out.split(b"\0") if rel]
    text: list[Path] = []
    for p in paths:
        if p.is_symlink():
            text.append(p)
            continue
        if not p.is_file():
            continue
        with p.open("rb") as fh:
            if b"\0" not in fh.read(8000):
                text.append(p)
    return text


def corpus_file_mentions(paths: list[Path]) -> list[str]:
    """`path:line: text` for every line that names a concrete file under the corpus root."""
    hits: list[str] = []
    for p in paths:
        content = os.readlink(p) if p.is_symlink() else p.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(content.splitlines(), start=1):
            if CORPUS_FILE.search(line):
                hits.append(f"{p}:{lineno}: {line.strip()[:120]}")
    return hits


def test_no_tracked_file_names_a_corpus_file() -> None:
    assert corpus_file_mentions(tracked_text_files(REPO)) == []


def test_scanner_flags_a_listing_row(tmp_path: Path) -> None:
    # A fileslisting row as `_write_listing_rows` emits it, with synthetic names: audio path first,
    # reference second. Built from the root constant so this test file does not itself trip the scan.
    listing = tmp_path / "fileslisting.csv"
    listing.write_text(f"{CORPUS_ROOT}/train/audio/ara/ab12cd.wav;{CORPUS_ROOT}/train/ref/ab12cd.part.xml;ara;;1.0;1.0;\n")
    assert len(corpus_file_mentions([listing])) == 1


def test_scanner_flags_a_metadata_path(tmp_path: Path) -> None:
    meta = tmp_path / "run_metadata.json"
    meta.write_text(f'{{"fileslisting": "{CORPUS_ROOT}/train/LID_Features/eng/xy98zw.plp8f0mvsdd"}}\n')
    assert len(corpus_file_mentions([meta])) == 1


def test_scanner_flags_a_long_feature_extension(tmp_path: Path) -> None:
    # The eval-side features carry 14- and 16-character extensions.
    notes = tmp_path / "notes.md"
    notes.write_text(f"{CORPUS_ROOT}/eval/LID_Features/30/xy98zw.plp8f0mvscmllrdd\n")
    assert len(corpus_file_mentions([notes])) == 1


def test_scanner_reads_a_symlink_as_its_target_path(tmp_path: Path) -> None:
    # Git commits a symlink as its target path; a dangling link (the CI case) must still be scanned.
    link = tmp_path / "clip.wav"
    link.symlink_to(f"{CORPUS_ROOT}/train/audio/ara/ab12cd.wav")
    assert len(corpus_file_mentions([link])) == 1


def test_scanner_allows_directory_globs_and_the_bare_root(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text(
        f"repointed at the sorted-first `*.wav` under `{CORPUS_ROOT}/train/audio/**`\n"
        f"the corpus lives at `{CORPUS_ROOT}/` (27 GB, gitignored)\n"
        f"`*.phSeqbis` under `{CORPUS_ROOT}/train/phSeq/**`\n"
    )
    assert corpus_file_mentions([notes]) == []


def test_runs_dir_is_ignored() -> None:
    # The launcher recipes write to runs/<arm>_full/; its listings and metadata carry corpus paths.
    # Unanchored, so a run launched from a subdirectory is ignored too.
    for path in ("runs/sad_full/base.config", "src/rust/runs/sad_full/base.config"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO).returncode == 0, path
