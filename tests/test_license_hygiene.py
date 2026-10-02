"""The Phase 6 license-hygiene rule, made executable: no tracked file names a corpus file.

The rule (CLAUDE.md "Since Phase 6", docs/validation.md): no corpus path, filename or feature
table is committed; fixtures use synthetic pattern-preserving names. Until now it was enforced
by review only. The scanner here reads every tracked text file and flags a path under the corpus
root that ends in a concrete filename. Directory-level globs (`data/LRE03-LRE07/train/audio/**`)
and the bare root name are allowed: RESULTS.md uses both to describe the protocol.

The scoring toolkit root (`data/Scoring_LRE15/`) is deliberately not scanned: the tree cites its
scorer scripts by name as the oracle source, and those are code, not licensed audio.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS_ROOT = "data/LRE03-LRE07"

# A path under the corpus root whose last segment is a concrete filename: a stem of word
# characters (so `*`, `**` and `{...}` globs never match) followed by a dot and an extension.
CORPUS_FILE = re.compile(r"LRE03-LRE07/(?:[\w.-]+/)*[\w-]*\w\.[A-Za-z0-9]{1,12}\b")


def tracked_text_files(repo: Path) -> list[Path]:
    """Every file `git ls-files` reports, minus binaries (git's own heuristic: a NUL in the first 8000 bytes)."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True).stdout
    paths = [repo / rel.decode("utf-8") for rel in out.split(b"\0") if rel]
    text: list[Path] = []
    for p in paths:
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
        for lineno, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
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
    rc = subprocess.run(["git", "check-ignore", "-q", "runs/sad_full/base.config"], cwd=REPO).returncode
    assert rc == 0
