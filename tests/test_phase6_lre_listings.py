"""Phase 6 Task 2: LRE03/07 listing localization + SAD listings + the 12-class
`language2classmapping`. Synthetic-tree unit tests (pure logic, no corpus needed)
plus corpus-gated audits (the real archive at `tests.conftest.CORPUS_ROOT` and the
legacy `Listings/` sibling repo, read-only, holding the real 2015 fileslisting CSVs
this codebase never vendors).
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest
import soundfile as sf
from speech.batching import read_listing
from speech.dataprep.lre import (
    LRE03_LANGUAGES,
    RealSphRunner,
    _localize_path,
    _wav_xml_pairs,
    convert_sph_listing,
    derive_sad_listings,
    localize_listing,
    parse_vrcts_structural,
    sph2wav_command,
    write_lre_mapping,
)

from tests.conftest import CORPUS_ROOT, requires_corpus

# Read-only sibling repo (NOT part of this checkout) holding the real 2015
# fileslisting CSVs this codebase never vendors -- a second, independent local
# prerequisite from the corpus itself, gated separately.
LEGACY_LISTINGS_DIR = pathlib.Path("/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Listings")
requires_legacy_listings = pytest.mark.skipif(
    not LEGACY_LISTINGS_DIR.is_dir(),
    reason=f"legacy Listings sibling repo absent at {LEGACY_LISTINGS_DIR}",
)


def _toy_vrcts_xml() -> bytes:
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<AudioDoc name="x" path="/tmp/x.wav">\n'
        b'<ChannelList><Channel num="1" sigdur="10.0" spdur="5.0"/></ChannelList>\n'
        b"<SegmentList>\n"
        b'<SpeechSegment ch="1" sconf="1.00" stime="1.0" etime="2.5" spkid="1"/>\n'
        b'<SpeechSegment ch="1" sconf="1.00" stime="3.0" etime="4.0" spkid="1"/>\n'
        b"</SegmentList>\n</AudioDoc>\n"
    )


# --------------------------------------------------------------------------------- #
# _localize_path
# --------------------------------------------------------------------------------- #


def test_localize_path_strips_2015_prefix_at_train_anchor(tmp_path: pathlib.Path) -> None:
    corpus_root = tmp_path / "corpus"
    raw = "/Users/Govit/Documents/AudioFiles/LRE03/train/LID_Features/plp8f0mvsdd/LRE03/xxx_0001.plp8f0mvsdd"
    got = _localize_path(raw, corpus_root)
    assert got == corpus_root / "train" / "LID_Features" / "plp8f0mvsdd" / "LRE03" / "xxx_0001.plp8f0mvsdd"


def test_localize_path_strips_2015_prefix_at_eval_anchor(tmp_path: pathlib.Path) -> None:
    corpus_root = tmp_path / "corpus"
    raw = "/Users/Govit/Documents/AudioFiles/LRE03/eval/audio/LRE07/foo.wav"
    got = _localize_path(raw, corpus_root)
    assert got == corpus_root / "eval" / "audio" / "LRE07" / "foo.wav"


def test_localize_path_no_anchor_falls_back_to_basename(tmp_path: pathlib.Path) -> None:
    corpus_root = tmp_path / "corpus"
    got = _localize_path("/some/weird/tree/no_anchor_here.wav", corpus_root)
    assert got == corpus_root / "no_anchor_here.wav"


# --------------------------------------------------------------------------------- #
# localize_listing
# --------------------------------------------------------------------------------- #


def _make_localize_corpus_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "corpus_root"
    (root / "train" / "feats").mkdir(parents=True)
    (root / "train" / "feats" / "a.plp").write_bytes(b"AAAA")
    (root / "train" / "seg.xml").write_bytes(b"")  # shared ref, present but empty (matches the real corpus)
    return root


def test_localize_listing_found_and_missing_rows(tmp_path: pathlib.Path) -> None:
    root = _make_localize_corpus_tree(tmp_path)
    src = tmp_path / "src_listing.csv"
    src.write_text(
        "/Users/Govit/Documents/AudioFiles/X/train/feats/a.plp;/Users/Govit/Documents/AudioFiles/X/train/seg.xml;eng;non;1.0;12.3;\n"
        "/Users/Govit/Documents/AudioFiles/X/train/feats/MISSING.plp;/Users/Govit/Documents/AudioFiles/X/train/seg.xml;chi;non;1.0;5.0;\n"
    )
    out = tmp_path / "out_listing.csv"

    report = localize_listing(src, root, out)

    assert report.rows_total == 2
    assert report.rows_found == 1
    assert report.rows_missing == 1
    assert report.missing_by_reason == {"primary": 1}

    rows = read_listing(out)
    assert len(rows) == 1
    assert rows[0]["filename"] == str(root / "train" / "feats" / "a.plp")
    assert rows[0]["refseg"] == str(root / "train" / "seg.xml")
    assert rows[0]["lang"] == "eng"
    # 6th column (duration in the real LRE listings, "file_id" in the generic
    # schema) is preserved verbatim, not reinterpreted -- see the module docstring.
    assert rows[0]["file_id"] == "12.3"

    missing_text = report.missing_path.read_text()
    assert "MISSING.plp" in missing_text
    assert "primary" in missing_text


def test_localize_listing_missing_refseg_reported(tmp_path: pathlib.Path) -> None:
    root = _make_localize_corpus_tree(tmp_path)
    src = tmp_path / "src2.csv"
    src.write_text("/Users/Govit/Documents/AudioFiles/X/train/feats/a.plp;/Users/Govit/Documents/AudioFiles/X/train/NOPE.xml;eng;non;1.0;1.0;\n")
    out = tmp_path / "out2.csv"

    report = localize_listing(src, root, out)

    assert report.rows_found == 0
    assert report.rows_missing == 1
    assert report.missing_by_reason == {"refseg": 1}


def test_localize_listing_missing_both_columns_reported_combined(tmp_path: pathlib.Path) -> None:
    root = _make_localize_corpus_tree(tmp_path)
    src = tmp_path / "src5.csv"
    src.write_text("/Users/Govit/Documents/AudioFiles/X/train/feats/NOPE.plp;/Users/Govit/Documents/AudioFiles/X/train/NOPE.xml;eng;non;1.0;1.0;\n")
    out = tmp_path / "out5.csv"

    report = localize_listing(src, root, out)

    assert report.rows_missing == 1
    assert report.missing_by_reason == {"primary+refseg": 1}


def test_localize_listing_empty_refseg_column_does_not_require_a_file(tmp_path: pathlib.Path) -> None:
    root = _make_localize_corpus_tree(tmp_path)
    src = tmp_path / "src3.csv"
    src.write_text("/Users/Govit/Documents/AudioFiles/X/train/feats/a.plp;;eng;non;1.0;1.0;\n")
    out = tmp_path / "out3.csv"

    report = localize_listing(src, root, out)

    assert report.rows_found == 1
    assert report.rows_missing == 0


def test_localize_listing_missing_sidecar_always_written_even_when_empty(tmp_path: pathlib.Path) -> None:
    root = _make_localize_corpus_tree(tmp_path)
    src = tmp_path / "src4.csv"
    src.write_text("/Users/Govit/Documents/AudioFiles/X/train/feats/a.plp;;eng;non;1.0;1.0;\n")
    out = tmp_path / "out4.csv"

    report = localize_listing(src, root, out)

    assert report.missing_path.is_file()
    assert report.missing_path.read_text() == ""
    assert report.missing_path == out.parent / f"{out.name}.missing"


def test_localize_listing_blank_lines_excluded_from_rows_total(tmp_path: pathlib.Path) -> None:
    root = _make_localize_corpus_tree(tmp_path)
    src = tmp_path / "src6.csv"
    src.write_text("/Users/Govit/Documents/AudioFiles/X/train/feats/a.plp;;eng;non;1.0;1.0;\n\n")
    out = tmp_path / "out6.csv"

    report = localize_listing(src, root, out)

    assert report.rows_total == 1  # the blank line is not a parsed record


# --------------------------------------------------------------------------------- #
# write_lre_mapping
# --------------------------------------------------------------------------------- #


def test_write_lre_mapping_12_rows_alphabetical_sequential_classindex(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "mapping.csv"
    write_lre_mapping(out)

    lines = out.read_text().splitlines()
    assert len(lines) == 12
    langs = []
    for i, line in enumerate(lines):
        lang, dial, class_index = line.split(";")
        assert dial == "non"
        assert int(class_index) == i
        langs.append(lang)
    assert langs == sorted(langs)
    assert set(langs) == set(LRE03_LANGUAGES)


def test_write_lre_mapping_parses_like_corpus_rs(tmp_path: pathlib.Path) -> None:
    """Independent reimplementation of `Corpus::from_config`'s mapping-parse rule
    (`engine/corpus.rs`'s `language2classmapping` load): semicolon-split (no
    trailing-empty handling needed, these lines carry no trailing ';'), class index
    via a plain int cast (the real parser's `atof(...) as i32`, a no-op on a bare
    digit string). A second-language cross-check, not a byte round-trip through
    Rust."""
    out = tmp_path / "mapping2.csv"
    write_lre_mapping(out)

    mapping: dict[tuple[str, str], int] = {}
    for line in out.read_text().splitlines():
        lang, dial, class_index = line.split(";")
        mapping[(lang, dial)] = int(class_index)

    assert mapping[("ara", "non")] == 0
    assert mapping[("vie", "non")] == 11
    assert len(mapping) == 12


# --------------------------------------------------------------------------------- #
# _wav_xml_pairs / derive_sad_listings
# --------------------------------------------------------------------------------- #


def _make_sad_audio_tree(tmp_path: pathlib.Path, n: int = 12) -> pathlib.Path:
    root = tmp_path / "corpus_root"
    audio = root / "train" / "audio" / "C1"
    audio.mkdir(parents=True)
    for i in range(n):
        (audio / f"spk_{i}.MT1.mp1.wav").write_bytes(b"RIFF....")
        (audio / f"spk_{i}.part.xml").write_bytes(_toy_vrcts_xml())
    # one orphan wav (no matching xml) and one orphan xml (no matching wav)
    (audio / "orphan_wav.MT1.mp1.wav").write_bytes(b"RIFF")
    (audio / "orphan_xml.part.xml").write_bytes(_toy_vrcts_xml())
    return root


def test_wav_xml_pairs_matches_by_first_dot_stem_and_counts_orphans(tmp_path: pathlib.Path) -> None:
    root = _make_sad_audio_tree(tmp_path, n=5)
    pairs, n_orphan_wav, n_orphan_xml = _wav_xml_pairs(root / "train" / "audio")

    assert len(pairs) == 5
    assert n_orphan_wav == 1
    assert n_orphan_xml == 1
    for wav, xml in pairs:
        assert wav.name.split(".", 1)[0] == xml.name.split(".", 1)[0]


def test_wav_xml_pairs_no_middle_segment_naming_also_matches(tmp_path: pathlib.Path) -> None:
    """LRE07-style naming (no `.MT1.mp1` middle segment): `XX-000001-A-con.wav` <->
    `XX-000001-A-con.part.xml` -- the same first-dot rule as LRE03's naming."""
    root = tmp_path / "corpus_root"
    audio = root / "train" / "audio" / "LRE07"
    audio.mkdir(parents=True)
    (audio / "XX-000001-A-con.wav").write_bytes(b"RIFF")
    (audio / "XX-000001-A-con.part.xml").write_bytes(_toy_vrcts_xml())

    pairs, n_orphan_wav, n_orphan_xml = _wav_xml_pairs(audio)

    assert len(pairs) == 1
    assert n_orphan_wav == 0
    assert n_orphan_xml == 0


def test_derive_sad_listings_split_is_seeded_disjoint_and_covers_all(tmp_path: pathlib.Path) -> None:
    root = _make_sad_audio_tree(tmp_path, n=40)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    split = derive_sad_listings(root, out_dir, seed=1234)

    assert split.n_total == 40
    assert split.n_orphan_wav == 1
    assert split.n_orphan_xml == 1
    assert split.n_train + split.n_valid + split.n_test == split.n_total

    train_rows = read_listing(split.train_path)
    valid_rows = read_listing(split.valid_path)
    test_rows = read_listing(split.test_path)
    assert len(train_rows) == split.n_train
    assert len(valid_rows) == split.n_valid
    assert len(test_rows) == split.n_test

    train_files = {r["filename"] for r in train_rows}
    valid_files = {r["filename"] for r in valid_rows}
    test_files = {r["filename"] for r in test_rows}
    assert train_files.isdisjoint(valid_files)
    assert train_files.isdisjoint(test_files)
    assert valid_files.isdisjoint(test_files)
    assert len(train_files | valid_files | test_files) == 40

    # no worker-shard duplicate emitted (the nb_workers=0 bypass)
    assert not (out_dir / "sad_train_worker_1.flst").exists()

    split_json = json.loads(split.split_json_path.read_text())
    assert split_json["seed"] == 1234
    assert split_json["n_total"] == 40
    assert len(split_json["train_files"]) == split.n_train


def test_derive_sad_listings_same_seed_reproducible(tmp_path: pathlib.Path) -> None:
    root = _make_sad_audio_tree(tmp_path, n=30)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    split_a = derive_sad_listings(root, out_a, seed=7)
    split_b = derive_sad_listings(root, out_b, seed=7)

    files_a = [r["filename"] for r in read_listing(split_a.train_path)]
    files_b = [r["filename"] for r in read_listing(split_b.train_path)]
    assert files_a == files_b


def test_derive_sad_listings_different_seed_differs(tmp_path: pathlib.Path) -> None:
    root = _make_sad_audio_tree(tmp_path, n=30)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    split_a = derive_sad_listings(root, out_a, seed=1)
    split_b = derive_sad_listings(root, out_b, seed=2)

    files_a = [r["filename"] for r in read_listing(split_a.train_path)]
    files_b = [r["filename"] for r in read_listing(split_b.train_path)]
    assert files_a != files_b


def test_derive_sad_listings_paths_are_absolute_and_lang_is_unk(tmp_path: pathlib.Path) -> None:
    root = _make_sad_audio_tree(tmp_path, n=6)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    split = derive_sad_listings(root, out_dir, seed=0)

    rows = read_listing(split.train_path) + read_listing(split.valid_path) + read_listing(split.test_path)
    assert rows
    for r in rows:
        assert pathlib.Path(r["filename"]).is_absolute()
        assert pathlib.Path(r["refseg"]).is_absolute()
        assert r["lang"] == "unk"
        assert r["dial"] == "unk"


# --------------------------------------------------------------------------------- #
# parse_vrcts_structural
# --------------------------------------------------------------------------------- #


def test_parse_vrcts_structural_reads_speech_segments(tmp_path: pathlib.Path) -> None:
    xml_path = tmp_path / "x.part.xml"
    xml_path.write_bytes(_toy_vrcts_xml())

    segs = parse_vrcts_structural(xml_path)

    assert segs == [(1.0, 2.5), (3.0, 4.0)]


def test_parse_vrcts_structural_no_segments_is_empty_not_an_error(tmp_path: pathlib.Path) -> None:
    xml_path = tmp_path / "empty.xml"
    xml_path.write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n<AudioDoc name="x" path="/tmp/x.wav">\n<SegmentList>\n</SegmentList>\n</AudioDoc>\n')

    assert parse_vrcts_structural(xml_path) == []


# --------------------------------------------------------------------------------- #
# convert_sph_listing / sph2wav_command
# --------------------------------------------------------------------------------- #


class RecordingSphRunner:
    """Test double for `SphRunner`: records every `run()` call's argv tokens, no real
    subprocess and no real file written -- mirrors `test_phase4d_augment.py`'s
    `RecordingRunner` idiom exactly. Because no file is actually created, a target
    wav's `is_file()` check stays False across repeated calls within one test unless
    the test itself pre-creates the target (see the idempotence test below)."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, cmd: list[str]) -> None:
        self.commands.append(cmd)


def test_sph2wav_command_exact_string() -> None:
    assert sph2wav_command("corpus/a.sph", pathlib.Path("/out/a.wav")) == ["sph2pipe", "-f", "wav", "corpus/a.sph", "/out/a.wav"]


def test_convert_sph_listing_pins_command_vector_for_sph_row(tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text("corpus/audio/a.sph;corpus/ref/a.xml;eng;non;1.0;10.0;\n")
    out_dir = tmp_path / "wavs"
    runner = RecordingSphRunner()

    convert_sph_listing(listing, out_dir, runner)

    expected_wav = out_dir.resolve() / "a.wav"
    assert runner.commands == [["sph2pipe", "-f", "wav", "corpus/audio/a.sph", str(expected_wav)]]


def test_convert_sph_listing_mixed_sph_and_non_sph_rows(tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text(
        "corpus/a.sph;corpus/a.xml;eng;non;1.0;10.0;\ncorpus/b.wav;corpus/b.xml;chi;non;1.0;8.0;\n",
    )
    out_dir = tmp_path / "wavs"
    runner = RecordingSphRunner()

    out_listing = convert_sph_listing(listing, out_dir, runner)

    rows = read_listing(out_listing)
    assert len(rows) == 2
    assert rows[0]["filename"] == str(out_dir.resolve() / "a.wav")
    assert rows[1]["filename"] == "corpus/b.wav"  # non-sph row untouched
    assert len(runner.commands) == 1  # only the sph row triggers a conversion


def test_convert_sph_listing_preserves_non_path_fields_byte_for_byte(tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text("corpus/a.sph;corpus/a.xml;eng;non;0.500;07.0;\n")
    out_dir = tmp_path / "wavs"
    runner = RecordingSphRunner()

    out_listing = convert_sph_listing(listing, out_dir, runner)

    rows = read_listing(out_listing)
    assert rows[0]["lang"] == "eng"
    assert rows[0]["dial"] == "non"
    assert rows[0]["weight"] == "0.500"  # original token text, not reformatted to "0.5"
    assert rows[0]["file_id"] == "07.0"  # original token text, not reformatted to "7.0"
    assert rows[0]["refseg"] == "corpus/a.xml"  # refseg is not itself a .sph -- untouched


def test_convert_sph_listing_creates_out_dir(tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text("corpus/a.sph;;eng;non;1.0;1.0;\n")
    out_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not out_dir.exists()
    runner = RecordingSphRunner()

    convert_sph_listing(listing, out_dir, runner)

    assert out_dir.is_dir()


def test_convert_sph_listing_returns_rewritten_listing_under_out_dir(tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text("corpus/a.sph;;eng;non;1.0;1.0;\n")
    out_dir = tmp_path / "wavs"
    runner = RecordingSphRunner()

    out_listing = convert_sph_listing(listing, out_dir, runner)

    assert out_listing == out_dir.resolve() / "in.csv"
    assert out_listing.is_file()


def test_convert_sph_listing_skips_conversion_when_target_wav_already_exists(tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text("corpus/a.sph;;eng;non;1.0;1.0;\n")
    out_dir = tmp_path / "wavs"
    out_dir.mkdir()
    (out_dir / "a.wav").write_bytes(b"already-converted")
    runner = RecordingSphRunner()

    out_listing = convert_sph_listing(listing, out_dir, runner)

    assert runner.commands == []  # no conversion command issued
    rows = read_listing(out_listing)
    assert rows[0]["filename"] == str(out_dir.resolve() / "a.wav")  # listing still rewritten


def test_convert_sph_listing_basename_collision_raises(tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text(
        "corpus/3/x.sph;;eng;non;1.0;1.0;\ncorpus/10/x.sph;;eng;non;1.0;1.0;\n",
    )
    out_dir = tmp_path / "wavs"
    runner = RecordingSphRunner()

    with pytest.raises(ValueError, match="collision"):
        convert_sph_listing(listing, out_dir, runner)


def test_convert_sph_listing_same_source_repeated_is_not_a_collision(tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "in.csv"
    listing.write_text("corpus/a.sph;;eng;non;1.0;1.0;\ncorpus/a.sph;;chi;non;1.0;1.0;\n")
    out_dir = tmp_path / "wavs"
    runner = RecordingSphRunner()

    out_listing = convert_sph_listing(listing, out_dir, runner)  # must not raise

    rows = read_listing(out_listing)
    assert len(rows) == 2
    assert rows[0]["filename"] == rows[1]["filename"]


# --------------------------------------------------------------------------------- #
# Corpus-gated audits (local-only; skip cleanly in CI / without the corpus)
# --------------------------------------------------------------------------------- #


@pytest.mark.corpus
@requires_corpus
def test_derive_sad_listings_real_corpus(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "sad"
    out_dir.mkdir()

    split = derive_sad_listings(CORPUS_ROOT, out_dir, seed=0)

    # Measured (Task 2 audit): 598 LRE03 + 1468 LRE07 wav/xml pairs, all matched;
    # the only orphans are LRE03's 2 reference-only files (illustrative naming:
    # xx_0099_{a,b}.part.xml, no matching wav in the archive).
    assert split.n_total == 2066
    assert split.n_orphan_wav == 0
    assert split.n_orphan_xml == 2
    assert split.n_train + split.n_valid + split.n_test == 2066


@pytest.mark.corpus
@requires_corpus
def test_derive_sad_listings_xml_sample_parses_structurally(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "sad"
    out_dir.mkdir()
    split = derive_sad_listings(CORPUS_ROOT, out_dir, seed=0)

    rows = read_listing(split.train_path)
    sample = pathlib.Path(rows[0]["refseg"])
    segs = parse_vrcts_structural(sample)

    assert len(segs) > 0
    for stime, etime in segs:
        assert etime > stime >= 0.0
    # This proves the sampled xml is loadable-SHAPED only; the byte-faithful
    # load_vrcts_hyp cross-check against the Rust `load_vrcts` port lands with
    # Task 4 (speech.evaluate) -- see parse_vrcts_structural's docstring.


@pytest.mark.corpus
@requires_corpus
@requires_legacy_listings
def test_localize_real_lre03_listing_existence_counts(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "lre03_localized.csv"
    report = localize_listing(LEGACY_LISTINGS_DIR / "listing_training_comb_plp8f0mvsdd_LRE03.csv", CORPUS_ROOT, out)

    assert report.rows_total == 15402
    # measured-then-pinned (Task 2 audit) -- see the task report for the exact
    # per-language existence breakdown and archive-completeness discussion.
    assert report.rows_found == 15402
    assert report.rows_missing == 0


@pytest.mark.corpus
@requires_corpus
@requires_legacy_listings
def test_localize_real_lre07_listing_existence_counts(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "lre07_localized.csv"
    report = localize_listing(LEGACY_LISTINGS_DIR / "listing_training_comb_plp8f0mvsdd_LRE07.csv", CORPUS_ROOT, out)

    assert report.rows_total == 14782
    assert report.rows_found == 14782
    assert report.rows_missing == 0


@pytest.mark.corpus
@requires_corpus
@requires_legacy_listings
def test_write_lre_mapping_covers_real_listing_languages(tmp_path: pathlib.Path) -> None:
    langs_lre03 = {r["lang"] for r in read_listing(LEGACY_LISTINGS_DIR / "listing_training_comb_plp8f0mvsdd_LRE03.csv")}
    langs_lre07 = {r["lang"] for r in read_listing(LEGACY_LISTINGS_DIR / "listing_training_comb_plp8f0mvsdd_LRE07.csv")}

    assert langs_lre03 <= set(LRE03_LANGUAGES)
    assert langs_lre07 <= set(LRE03_LANGUAGES)
    assert langs_lre03 == set(LRE03_LANGUAGES)  # LRE03 uses all 12


@pytest.mark.corpus
@requires_corpus
def test_convert_sph_listing_real_sph_converts_to_valid_wav(tmp_path: pathlib.Path) -> None:
    """Glob-discovered (never hardcoded, license hygiene): find any one real `.sph`
    under the corpus, convert it with the REAL subprocess-backed runner, and read the
    result back via soundfile to prove it is a valid wav. Skips with a named reason if
    either precondition (a `.sph` in the archive, `sph2pipe` on PATH) is unmet."""
    sph_files = sorted(CORPUS_ROOT.rglob("*.sph"))
    if not sph_files:
        pytest.skip(f"no .sph files under {CORPUS_ROOT}")
    if shutil.which("sph2pipe") is None:
        pytest.skip("sph2pipe not installed (a documented local prerequisite, not vendored)")

    sph_path = sph_files[0]
    listing = tmp_path / "real_sph.csv"
    listing.write_text(f"{sph_path};;eng;non;1.0;1.0;\n")
    out_dir = tmp_path / "wavs"

    out_listing = convert_sph_listing(listing, out_dir, RealSphRunner())

    rows = read_listing(out_listing)
    wav_path = pathlib.Path(rows[0]["filename"])
    assert wav_path.is_file()

    info = sf.info(str(wav_path))
    assert info.frames > 0
    assert info.samplerate > 0
