"""LRE03/07 corpus listing tooling: localize the 2015 absolute-path listings against
the local archive (`data/LRE03-LRE07/`, gitignored, licensed), derive SAD (wav+xml)
train/valid/test listings from `train/audio`, and write the 12-class LRE
`language2classmapping`. Phase 6 Task 2 (S1.2/S1.3 in the design spec) -- port-only,
no direct legacy source: the legacy never localized its own listings, they were
authored once, in place, on the original 2015 collection machine.

ABSOLUTE VS RELATIVE PATHS (decided here; applies to every path this module writes
INTO a listing row -- not to the listing FILES themselves, see below): every row path
is written ABSOLUTE, resolved against `corpus_root`. The rest of this codebase's
convention -- `language2classmapping`/`fileslisting` config KEYS, and the small
COMMITTED listing fixtures those keys point at, resolve relative to wherever the
driver `_chdir`s (`drivers/train.py`'s `_mapping_path`: "relative to the config's own
dir"; `tests/reference_data/phase4a/fileslisting.csv`'s `corpus/f1.wav` rows are a
clean example of why -- the fixture and its config travel together in the repo). That
convention exists to keep small, portable, COMMITTED fixtures working from any
checkout. `data/LRE03-LRE07/` does not travel with anything: it is a 27 GB,
gitignored, user-supplied tree with no fixed offset from wherever a future training
config (`configs/training/*.toml`, Task 8/9/10) ends up living -- its own location is
already an effectively-fixed local absolute fact (see `tests/conftest.py`'s
`CORPUS_ROOT`). Writing corpus-referencing ROW paths as absolute makes every listing
this module emits correct regardless of whatever directory a driver later `_chdir`s
into -- Task 8/9/10 never need to compute a relative offset back to the corpus. (Where
the listing FILE ITSELF lives, and what a future config's `fileslisting` key says to
find it, is a separate, smaller decision left to whichever task writes that config.)
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from speech.batching import read_listing, write_listing

# 2015 absolute listing paths are rooted under an opaque host-specific prefix
# (`/Users/Govit/Documents/AudioFiles/LRE03/...`); the corpus-relevant tail always
# starts at the first `train`/`eval` split component -- verified empirically against
# the real `listing_training_comb_plp8f0mvsdd_{LRE03,LRE07}.csv` (Task 2 audit, see
# the task report): every sampled row's tail lands directly under `corpus_root` with
# no further renesting. A path with neither anchor falls back to its bare filename so
# the row still gets probed for existence (and reported if absent) rather than being
# silently mis-resolved.
_SPLIT_ANCHORS = ("train", "eval")


def _localize_path(raw: str, corpus_root: Path) -> Path:
    """Map one 2015 absolute path onto `corpus_root`'s local layout: find the first
    `train`/`eval` path component and keep everything from there on."""
    posix = raw.replace("\\", "/")
    parts = [p for p in posix.split("/") if p]
    for i, part in enumerate(parts):
        if part in _SPLIT_ANCHORS:
            return corpus_root.joinpath(*parts[i:])
    return corpus_root / PurePosixPath(posix).name


@dataclass
class LocalizeReport:
    """`localize_listing`'s result. `missing_by_reason` counts rows by WHICH
    localized column(s) failed the existence probe (`"primary"`, `"refseg"`, or
    `"primary+refseg"`) -- `rows_missing` is a row count (a row missing both columns
    counts once there, but under one combined key here, not double-counted across
    two keys)."""

    rows_total: int
    rows_found: int
    rows_missing: int
    out_path: Path
    missing_path: Path
    missing_by_reason: dict[str, int]


def localize_listing(src: Path, corpus_root: Path, out: Path) -> LocalizeReport:
    """Rewrite a 2015-era fileslisting's absolute paths (`filename`/`refseg`
    columns) against the local `corpus_root`, verifying per row that BOTH the
    localized primary file and (if the source row carries one) the localized refseg
    file actually exist on disk. A row where either localized path is missing is
    EXCLUDED from `out` and recorded in `<out>.missing` alongside its ORIGINAL
    (pre-localization) columns and a reason -- never silently dropped (design spec
    R7). `<out>.missing` is ALWAYS written, even with zero missing rows (an empty
    file), so a caller never has to branch on its existence before reading it.

    `out`'s rows preserve every original field (lang/dial/weight/6th-column)
    UNCHANGED except the two path columns -- including when the 6th column is
    semantically a duration, not the generic schema's `file_id` (the real LRE
    listings are `path;ref-xml;lang;class;weight;duration`, six fields like the
    engine's own `fileslisting` grammar, but with a different meaning in the last
    slot; `read_listing`/`write_listing`'s dict schema calls it `file_id` regardless
    -- this function never interprets that column, only carries its original string
    through, so the mismatch is harmless here).

    Parses via `batching.read_listing` (reused, not reimplemented), so
    `rows_total` is a PARSED-RECORD count: a blank line or a line whose first token
    is empty is not a row to begin with, matching the engine's own listing
    semantics. Neither `src`'s nor `out`'s parent directory is created -- caller's
    responsibility, matching `batching.write_listing`'s own no-mkdir convention.
    """
    records = read_listing(src)
    corpus_root = corpus_root.resolve()

    found_rows: list[dict[str, str]] = []
    missing_lines: list[str] = []
    missing_by_reason: dict[str, int] = {}

    for rec in records:
        local_filename = _localize_path(rec["filename"], corpus_root)
        primary_ok = local_filename.is_file()

        has_refseg = rec["refseg"] != ""
        refseg_str = rec["refseg"]
        refseg_ok = True
        if has_refseg:
            local_refseg = _localize_path(rec["refseg"], corpus_root)
            refseg_ok = local_refseg.is_file()
            refseg_str = str(local_refseg)

        if primary_ok and refseg_ok:
            new_rec = dict(rec)
            new_rec["filename"] = str(local_filename)
            new_rec["refseg"] = refseg_str
            found_rows.append(new_rec)
        else:
            reasons = []
            if not primary_ok:
                reasons.append("primary")
            if not refseg_ok:
                reasons.append("refseg")
            reason = "+".join(reasons)
            missing_by_reason[reason] = missing_by_reason.get(reason, 0) + 1
            missing_lines.append(f"{rec['filename']};{rec['refseg']};{reason}\n")

    out.write_text("".join(f"{r['filename']};{r['refseg']};{r['lang']};{r['dial']};{r['weight']};{r['file_id']};\n" for r in found_rows))
    missing_path = out.parent / f"{out.name}.missing"
    missing_path.write_text("".join(missing_lines))

    return LocalizeReport(
        rows_total=len(records),
        rows_found=len(found_rows),
        rows_missing=len(records) - len(found_rows),
        out_path=out,
        missing_path=missing_path,
        missing_by_reason=missing_by_reason,
    )


# --------------------------------------------------------------------------------- #
# SAD listings: derived fresh from train/audio's wav+xml pairs (no 2015 listing to
# localize -- the legacy never shipped a fileslisting for raw-audio SAD training over
# this corpus, only for the precomputed LID features).
# --------------------------------------------------------------------------------- #

# 70/15/15 train/valid/test, hardcoded (not a legacy value -- this is new tooling, no
# parity target). `round()` here is plain Python round-half-to-even; NOT
# `batching._mround` (MATLAB round-half-away-from-zero) -- that convention exists only
# where this codebase reproduces MATLAB arithmetic, which this module never does.
_TRAIN_RATIO = 0.70
_VALID_RATIO = 0.15
# test = remainder (n_total - n_train - n_valid), not a third ratio constant --
# guarantees the three counts sum to n_total exactly regardless of rounding.


def _wav_xml_pairs(train_audio_root: Path) -> tuple[list[tuple[Path, Path]], int, int]:
    """Pair each `*.wav` under `train_audio_root` with its `*.xml` sibling in the
    SAME directory, matched by "filename up to (not including) the first '.'" --
    observed naming (Task 2 audit): LRE03 wavs carry a `.MT1.mp1.wav` middle segment
    the xml drops entirely (`ar_4240_a.MT1.mp1.wav` <-> `ar_4240_a.part.xml`), LRE07
    wavs have no middle segment at all (`AE-001244-A-con.wav` <->
    `AE-001244-A-con.part.xml`) -- both fall out of the SAME first-dot rule, so
    there is no need to hardcode the `.part.xml` suffix specifically (any same-stem
    `*.xml` sibling matches).

    Returns `(pairs, n_orphan_wav, n_orphan_xml)`: a wav with no matching xml, or an
    xml with no matching wav (2 real cases in the archive:
    `sp_4379_{a,b}.part.xml`), are EXCLUDED from `pairs` and counted separately --
    never silently merged into a mismatched pair.
    """

    def stem(p: Path) -> str:
        return p.name.split(".", 1)[0]

    wavs = sorted(train_audio_root.rglob("*.wav"))
    xmls = sorted(train_audio_root.rglob("*.xml"))
    xml_by_key = {(x.parent, stem(x)): x for x in xmls}

    pairs: list[tuple[Path, Path]] = []
    matched_xml_keys: set[tuple[Path, str]] = set()
    n_orphan_wav = 0
    for w in wavs:
        key = (w.parent, stem(w))
        xml = xml_by_key.get(key)
        if xml is None:
            n_orphan_wav += 1
            continue
        pairs.append((w, xml))
        matched_xml_keys.add(key)

    n_orphan_xml = len(xml_by_key) - len(matched_xml_keys)
    return pairs, n_orphan_wav, n_orphan_xml


@dataclass
class SadSplit:
    """`derive_sad_listings`'s result. `n_orphan_wav`/`n_orphan_xml` are the
    unmatched wav/xml counts `_wav_xml_pairs` excluded (see its docstring) --
    reported, not silently dropped, mirroring `LocalizeReport`'s existence-audit
    stance even though design spec R7 names `localize_listing` specifically."""

    seed: int
    n_total: int
    n_train: int
    n_valid: int
    n_test: int
    n_orphan_wav: int
    n_orphan_xml: int
    train_path: Path
    valid_path: Path
    test_path: Path
    split_json_path: Path


def derive_sad_listings(corpus_root: Path, out_dir: Path, seed: int) -> SadSplit:
    """Build a SAD fileslisting (`batching.write_listing`'s convention --
    `filename;refseg;lang;dial;`) from every `train/audio/**/*.wav` + same-stem
    `*.xml` pair under `corpus_root` (see `_wav_xml_pairs`), then a SEEDED, disjoint
    70/15/15 train/valid/test split (`np.random.default_rng(seed).permutation`)
    recorded BOTH in the emitted listings' filenames (`sad_train.flst`/
    `sad_valid.flst`/`sad_test.flst`, written via `write_listing(..., nb_workers=0)`
    -- `nb_workers=0` skips `write_listing`'s worker-shard files entirely, since
    `min(0, n) == 0` empties its shard loop; worker-sharding and this
    train/valid/test split are different axes, and reusing `write_listing` with a
    real worker count would silently also emit a redundant `_worker_1` duplicate of
    the base listing per split) AND in a `split.json` sidecar (seed, ratios,
    per-split counts, orphan counts, and each split's file list -- reproducible
    audit independent of re-parsing the listings).

    SAD has no per-file language target: `lang`/`dial` are written `"unk"`/`"unk"`
    for every row -- language is the LID arm's concern, carried instead by the
    localized LRE listings `localize_listing` produces.

    Every emitted path is ABSOLUTE (see the module docstring). Neither `out_dir`
    nor `corpus_root` is created -- caller's responsibility.
    """
    corpus_root = corpus_root.resolve()
    pairs, n_orphan_wav, n_orphan_xml = _wav_xml_pairs(corpus_root / "train" / "audio")

    n_total = len(pairs)
    order = np.random.default_rng(seed).permutation(n_total).tolist()
    n_train = round(n_total * _TRAIN_RATIO)
    n_valid = round(n_total * _VALID_RATIO)
    n_test = n_total - n_train - n_valid

    split_indices = {
        "train": order[:n_train],
        "valid": order[n_train : n_train + n_valid],
        "test": order[n_train + n_valid :],
    }

    split_paths: dict[str, Path] = {}
    split_files: dict[str, list[str]] = {}
    for name, idx in split_indices.items():
        items = [{"filename": str(pairs[i][0]), "refseg": str(pairs[i][1]), "lang": "unk", "dial": "unk"} for i in idx]
        base = out_dir / f"sad_{name}"
        write_listing(base, items, nb_workers=0)
        split_paths[name] = base.parent / f"{base.name}.flst"
        split_files[name] = [it["filename"] for it in items]

    split_json_path = out_dir / "split.json"
    split_json_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "train_ratio": _TRAIN_RATIO,
                "valid_ratio": _VALID_RATIO,
                "test_ratio": round(1.0 - _TRAIN_RATIO - _VALID_RATIO, 10),
                "n_total": n_total,
                "n_train": n_train,
                "n_valid": n_valid,
                "n_test": n_test,
                "n_orphan_wav": n_orphan_wav,
                "n_orphan_xml": n_orphan_xml,
                "train_files": split_files["train"],
                "valid_files": split_files["valid"],
                "test_files": split_files["test"],
            },
            indent=2,
        )
    )

    return SadSplit(
        seed=seed,
        n_total=n_total,
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        n_orphan_wav=n_orphan_wav,
        n_orphan_xml=n_orphan_xml,
        train_path=split_paths["train"],
        valid_path=split_paths["valid"],
        test_path=split_paths["test"],
        split_json_path=split_json_path,
    )


def parse_vrcts_structural(path: Path) -> list[tuple[float, float]]:
    """Structural-only VRCTS reference check: parses `path` as well-formed XML and
    extracts every `<SpeechSegment stime=".." etime=".."/>` as a `(stime, etime)`
    pair, document order. NOT a port of `tasks::segmentation_io::load_vrcts`
    (Rust) -- no offset/duration windowing, no `sanitize()`, no in-band signal
    handling; it only proves a sampled reference is loadable-SHAPED, which is all
    Task 2's corpus-gated audit needs. The byte-faithful Python-side adapter
    (`load_vrcts_hyp`) is Task 4's deliverable (`speech.evaluate`), deliberately
    not duplicated here.

    Raises `xml.etree.ElementTree.ParseError` on malformed XML, `KeyError` on a
    `SpeechSegment` missing `stime`/`etime`, `ValueError` if either isn't a float.
    """
    root = ET.parse(path).getroot()
    return [(float(seg.attrib["stime"]), float(seg.attrib["etime"])) for seg in root.iter("SpeechSegment")]


# The 12 LRE03 languages (design spec S0/S1.7); LRE07 trains a 9-language SUBSET of
# these (vie/fre/fas absent -- Task 2 audit). Already alphabetical; `write_lre_mapping`
# still sorts defensively rather than assuming this literal stays that way.
LRE03_LANGUAGES: tuple[str, ...] = ("ara", "chi", "eng", "fas", "fre", "ger", "hin", "jap", "kor", "spa", "tam", "vie")


def write_lre_mapping(out: Path) -> None:
    """Write the single 12-class `language2classmapping` (`lang;dial;classIndex`
    per one line, matching `Corpus::from_config`'s parser -- `engine/corpus.rs`)
    for [`LRE03_LANGUAGES`][speech.dataprep.lre.LRE03_LANGUAGES]. `dial` is fixed
    `"non"` (the only dialect tag the real LRE03/07 listings ever carry -- Task 2
    audit). `classIndex` runs 0..11 in ALPHABETICAL language order, matching the
    port's `_class_keys` (`drivers/test.py`) convention -- but note the engine's OWN
    mapping-file parser (unlike `_class_keys`, which derives its key order from
    `keys(langMapConf)` and ignores this column) assigns `class_index` from this
    column DIRECTLY, so the column must actually BE alphabetical for the two to
    agree, not just for cosmetic consistency with `_class_keys`.
    """
    langs = sorted(LRE03_LANGUAGES)
    out.write_text("".join(f"{lang};non;{i}\n" for i, lang in enumerate(langs)))
