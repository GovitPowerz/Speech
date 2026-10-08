"""Phase 6 Task 8: launcher UNIT tests for `speech.drivers.baseline` (no engine, no corpus).

These pin the PURE, engine-free surface of the baseline arm: stratified subset/split
sampling (per-language proportional, seeded, deterministic), config assembly, run-metadata
recording, corpus-listing derivation over a SYNTHETIC tree, and CLI arg wiring / flag
plumbing (with `run_baseline` monkeypatched). The engine-backed end-to-end path is the
corpus-gated gate `tests/pyo3/test_phase6_gates.py`.

License hygiene: every fixture here is synthetic (hand-built tiny records / a fake corpus
tree of empty `.plp8f0mvsdd` files) -- nothing from `data/LRE03-LRE07/` is read or committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from speech.batching import create_batches, get_new_batch
from speech.config_bridge import CELL_TYPES, DIRECTIONS
from speech.drivers import baseline as B
from speech.drivers.spec import ARM_CONFIG, LID_ARMS, SAD_ARMS, BaselineSpec
from speech.drivers.state import ModernTrainParams

# --------------------------------------------------------------------------------------- #
# synthetic records
# --------------------------------------------------------------------------------------- #


def _records(counts: dict[str, int]) -> list[dict[str, str]]:
    """Synthetic listing records: `counts[lang]` files per language, unique filenames."""
    out: list[dict[str, str]] = []
    for lang, n in counts.items():
        for i in range(n):
            out.append({"filename": f"/x/{lang}_{i}.plp8f0mvsdd", "refseg": "/x/ref.stm", "lang": lang, "dial": "non", "weight": "1.0", "file_id": "1.0"})
    return out


# --------------------------------------------------------------------------------------- #
# stratified disjoint splits (subset stratification: per-language proportional, seeded)
# --------------------------------------------------------------------------------------- #


def test_stratified_splits_disjoint_and_proportional() -> None:
    recs = _records({"ara": 100, "chi": 100, "tam": 100})  # abundant per language
    train, valid, test = B.stratified_splits(recs, n_train=30, n_valid=6, n_test=12, seed=0)
    tr, va, te = ({r["filename"] for r in s} for s in (train, valid, test))
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), "the three splits must be disjoint"
    # proportional per language: train 10/lang, valid 2/lang, test 4/lang.
    assert len(train) == 30 and len(valid) == 6 and len(test) == 12
    for lang in ("ara", "chi", "tam"):
        assert sum(1 for r in test if r["lang"] == lang) == 4


def test_stratified_splits_proportional_to_source_and_floors_rare_language() -> None:
    # imbalanced source: the train quota tracks source proportion (ara >> chi) BUT a language
    # too rare for its proportional quota to round up still survives at >= 1 (all classes kept).
    recs = _records({"ara": 1000, "chi": 500, "tam": 30})  # total 1530
    train, _valid, _test = B.stratified_splits(recs, n_train=60, n_valid=0, n_test=0, seed=0)
    per = {lang: sum(1 for r in train if r["lang"] == lang) for lang in ("ara", "chi", "tam")}
    assert per["ara"] > per["chi"] > per["tam"], f"train quotas must track source proportion, got {per}"
    assert per["tam"] >= 1, "a rare language must not vanish from the training split"


def test_stratified_splits_seed_sensitive() -> None:
    recs = _records({"ara": 100, "chi": 100})
    a = [r["filename"] for r in B.stratified_splits(recs, 40, 0, 0, seed=0)[0]]
    c = [r["filename"] for r in B.stratified_splits(recs, 40, 0, 0, seed=1)[0]]
    assert a != c and len(a) == len(c), "a different seed must draw a different sample of the same size"


def test_stratified_splits_none_train_fills_heldout() -> None:
    recs = _records({"ara": 20, "chi": 20})
    train, valid, test = B.stratified_splits(recs, n_train=None, n_valid=4, n_test=4, seed=0)
    # n_train=None (the full-run path): valid/test are carved to their full quotas FIRST and
    # train takes the per-language REMAINDER -- so with plenty of files EVERY split is filled
    # and the three stay disjoint. Regression pin: the pre-fix code gave train ALL files and
    # left valid/test EMPTY (empirically 40/0/0).
    assert len(train) + len(valid) + len(test) <= len(recs)
    tr, va, te = ({r["filename"] for r in s} for s in (train, valid, test))
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(train) > 0 and len(valid) > 0 and len(test) > 0, "n_train=None must still fill the held-out valid AND test splits"


def test_stratified_splits_deterministic() -> None:
    recs = _records({"ara": 50, "chi": 50})
    a = B.stratified_splits(recs, 20, 4, 8, seed=3)
    b = B.stratified_splits(recs, 20, 4, 8, seed=3)
    assert [r["filename"] for r in a[0]] == [r["filename"] for r in b[0]]
    assert [r["filename"] for r in a[2]] == [r["filename"] for r in b[2]]


# --------------------------------------------------------------------------------------- #
# config assembly + metadata + hashing
# --------------------------------------------------------------------------------------- #


def test_assemble_flat_config_overrides_only_runtime_keys() -> None:
    base = {
        "Algo_choice": "6",
        "fileslisting": "PLACEHOLDER",
        "language2classmapping": "PH",
        "BLSTM_weightsFile": "x",
        "BLSTM_LID_weightsFile": "y",
        "numOuterThreads": "1",
        "BLSTM_LID_Mode": "7",
    }
    cfg = B.assemble_flat_config(base, fileslisting="train.flst", mapping="map.csv", sad_seed="s.bin", lid_seed="l.bin", lanes=4)
    assert cfg["fileslisting"] == "train.flst"
    assert cfg["language2classmapping"] == "map.csv"
    assert cfg["BLSTM_weightsFile"] == "s.bin"
    assert cfg["BLSTM_LID_weightsFile"] == "l.bin"
    assert cfg["numOuterThreads"] == "4"
    # untouched keys survive verbatim; base is not mutated in place.
    assert cfg["Algo_choice"] == "6" and cfg["BLSTM_LID_Mode"] == "7"
    assert base["fileslisting"] == "PLACEHOLDER", "assemble_flat_config must not mutate its input"


def test_write_run_metadata_roundtrip(tmp_path: Path) -> None:
    meta = {"arm": "lid-features", "seed": 0, "lanes": 1, "subset": 36, "config_hash": "abc"}
    path = B.write_run_metadata(tmp_path, meta)
    assert path == tmp_path / "run_metadata.json"
    loaded = json.loads(path.read_text())
    assert loaded == meta


def test_config_hash_stable_and_sensitive() -> None:
    assert B._config_hash("a b\n") == B._config_hash("a b\n")
    assert B._config_hash("a b\n") != B._config_hash("a c\n")


# --------------------------------------------------------------------------------------- #
# corpus-listing derivation over a synthetic tree + the 12-class mapping
# --------------------------------------------------------------------------------------- #


def test_derive_lid_features_records_from_synthetic_tree(tmp_path: Path) -> None:
    root = tmp_path / "train" / "LID_Features" / "plp8f0mvsdd" / "LRE03"
    root.mkdir(parents=True)
    for name in ("ara_0001.plp8f0mvsdd", "spa_0001.plp8f0mvsdd", "vie_0001.plp8f0mvsdd", "zzz_0001.plp8f0mvsdd"):
        (root / name).write_bytes(b"")  # empty stand-in, synthetic ids (no real corpus filename committed); derivation only reads names
    ref = tmp_path / "ref.stm"
    recs = B.derive_lid_features_records(tmp_path, ref)
    langs = sorted(r["lang"] for r in recs)
    assert langs == ["ara", "spa", "vie"], "only the 12 known LRE03 languages survive (zzz dropped)"
    assert all(r["refseg"] == str(ref) and r["dial"] == "non" for r in recs)
    assert all(Path(r["filename"]).is_absolute() for r in recs)


def test_derive_lid_phseq_records_from_synthetic_tree(tmp_path: Path) -> None:
    root = tmp_path / "train" / "phSeq"
    root.mkdir(parents=True)
    # SYNTHETIC names (fake IDs -- no real corpus filename is committed): a 2-letter language
    # prefix + underscore + the rest, matching the CallFriend `<lang>_<id>...file.phSeqbis`
    # SHAPE without being a real file. Covers: three known prefixes, a numeric no-prefix file
    # (dropped), an unknown prefix (dropped), and the non-`.file` variant (not globbed).
    for name in (
        "ar_0.file.phSeqbis",  # ara
        "ma_0.file.phSeqbis",  # chi (Mandarin)
        "vi_0.file.phSeqbis",  # vie
        "lid0.file.phSeqbis",  # numeric, no underscore -> prefix not in the map -> dropped
        "zz_0.file.phSeqbis",  # unknown 2-letter prefix -> dropped
        "ar_0.phSeqbis",  # the per-sentence variant (NOT .file) -> not globbed
    ):
        (root / name).write_text(".a.\n")
    ref = tmp_path / "ref.stm"
    recs = B.derive_lid_phseq_records(tmp_path, ref)
    langs = sorted(r["lang"] for r in recs)
    assert langs == ["ara", "chi", "vie"], "only .file.phSeqbis with a known 2-letter prefix survive"
    assert all(r["refseg"] == str(ref) and r["dial"] == "non" for r in recs)
    assert all(Path(r["filename"]).is_absolute() and r["filename"].endswith(".file.phSeqbis") for r in recs)


def test_phseq_prefix_map_bijects_onto_the_12_classes() -> None:
    # every 2-letter prefix maps to a distinct one of the 12 alphabetical LRE03 classes.
    assert sorted(B._PHSEQ_PREFIX_TO_LANG.values()) == sorted(B._LANGS)
    assert len(set(B._PHSEQ_PREFIX_TO_LANG.values())) == 12
    assert B._PHSEQ_PREFIX_TO_LANG["ma"] == "chi", "Mandarin prefix 'ma' maps to the LRE03 'chi' tag"


def test_write_lre_mapping_12_alphabetical(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    B.write_lre_mapping_12(path)
    lines = [ln for ln in path.read_text().splitlines() if ln]
    assert len(lines) == 12
    assert lines[0] == "ara;non;0" and lines[-1] == "vie;non;11"
    # class ids are 0..11 in alphabetical language order (the _class_keys / engine convention).
    for i, ln in enumerate(lines):
        assert ln.split(";")[2] == str(i)


# --------------------------------------------------------------------------------------- #
# CLI arg wiring + flag plumbing (run_baseline monkeypatched -- no engine)
# --------------------------------------------------------------------------------------- #


def test_build_parser_arg_wiring() -> None:
    args = B.build_parser().parse_args(
        ["lid-features", "--corpus-root", "/c", "--out-dir", "/o", "--subset", "36", "--lanes", "2", "--seed", "7", "--dry-run", "--resume"]
    )
    spec = BaselineSpec.from_args(args)
    assert spec.arm == "lid-features"
    assert spec.corpus_root == Path("/c") and spec.out_dir == Path("/o")
    assert spec.subset == 36 and spec.lanes == 2 and spec.seed == 7
    assert spec.dry_run is True and spec.resume is True


def _forwarded(monkeypatch: pytest.MonkeyPatch, entry: Callable[[list[str]], int], argv: list[str]) -> BaselineSpec:
    seen: list[BaselineSpec] = []
    monkeypatch.setattr(B, "run_baseline", lambda spec, **kw: seen.append(spec))
    assert entry(argv) == 0
    assert len(seen) == 1
    return seen[0]


def test_both_entries_build_the_same_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #28: the `speech.cli` mount once dropped --cell-type/--direction. Both entries hand
    the parsed namespace to the one builder, `BaselineSpec.from_args`, so the parser is the one
    place a flag is defined and the spec the one place it is validated."""
    import speech.cli as cli

    argv = ["sad", "--corpus-root", "/c", "--out-dir", "/o", "--cell", "cfc", "--direction", "forward"]
    argv += ["--score-init", "--valid-size", "5", "--test-size", "7", "--minibatch", "3", "--lanes", "2", "--audio-max-duration", "30.0"]
    expected = BaselineSpec(
        arm="sad",
        corpus_root=Path("/c"),
        out_dir=Path("/o"),
        cell="cfc",
        direction="forward",
        score_init=True,
        valid_size=5,
        test_size=7,
        minibatch=3,
        lanes=2,
        audio_max_duration=30.0,
    )
    assert _forwarded(monkeypatch, B.main, argv) == expected
    assert _forwarded(monkeypatch, cli.main, ["baseline", *argv]) == expected


def test_parser_dests_and_defaults_are_the_specs() -> None:
    """Both ways: every parser dest is a spec field and every field a dest, and an all-default
    parse equals an all-default spec, so the CLI and a direct call train the same run."""
    ns = vars(B.build_parser().parse_args(["sad", "--corpus-root", "/c", "--out-dir", "/o"]))
    assert set(ns) == set(BaselineSpec.model_fields)
    assert BaselineSpec.from_args(argparse.Namespace(**ns)) == BaselineSpec(arm="sad", corpus_root=Path("/c"), out_dir=Path("/o"))


@pytest.mark.parametrize("argv", [["--epochs", "0"], ["--valid-size", "-1"], ["--lanes", "0"], ["--lre-listing", "/l.csv"]], ids=lambda a: a[0])
def test_an_invalid_run_is_a_usage_error_before_any_directory_exists(tmp_path: Path, argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "never"
    with pytest.raises(SystemExit) as e:
        B.main(["sad", "--corpus-root", "/c", "--out-dir", str(out), *argv])
    assert e.value.code == 2 and argv[0].lstrip("-").replace("-", "_") in capsys.readouterr().err
    assert not out.exists()


def test_listing_hash_is_blake2b_8_of_the_bytes(tmp_path: Path) -> None:
    p = tmp_path / "listing.csv"
    p.write_bytes(b"x,y\n1,2\n")
    assert B.listing_hash(p) == hashlib.blake2b(b"x,y\n1,2\n", digest_size=8).hexdigest()
    assert len(B.listing_hash(p)) == 16


def test_all_four_arms_wired() -> None:
    assert set(ARM_CONFIG) == {"lid-features", "sad", "sad-v2", "lid-phseq"}
    assert set(LID_ARMS) == {"lid-features", "lid-phseq"}
    assert set(SAD_ARMS) == {"sad", "sad-v2"}
    # Every declared arm has an instance, and the SAD class is exactly the SAD set: a new arm
    # cannot be silently mis-classified (the partition guard of the old frozensets).
    assert set(B.ARMS) == set(ARM_CONFIG)
    assert {arm for arm, obj in B.ARMS.items() if isinstance(obj, B.SadArm)} == set(SAD_ARMS)
    assert all(B.ARMS[arm].config == ARM_CONFIG[arm] for arm in ARM_CONFIG)


def test_lid_phseq_dispatches_into_lid_path(tmp_path: Path) -> None:
    # On an empty corpus tree the phonotactic arm proceeds into the LID record-derivation path
    # and raises the "no LID records" RuntimeError with the phSeq-specific hint.
    with pytest.raises(RuntimeError, match=r"no LID records.*phSeq"):
        B.run_baseline(_spec("lid-phseq", tmp_path, tmp_path / "out"))


def _spec(arm: str, corpus_root: Path, out_dir: Path, **over: object) -> BaselineSpec:
    return BaselineSpec(**{"arm": arm, "corpus_root": corpus_root, "out_dir": out_dir, **over})  # type: ignore[arg-type]


def _stub_run_state() -> type:
    """A `RunState` stand-in for the engine-free plumbing tests: both constructors return a bare
    object (nothing downstream reads it once the trainer and the scorers are stubbed)."""
    return type("RS", (), {"from_config": staticmethod(lambda *a, **k: object()), "from_parsed": staticmethod(lambda *a, **k: object())})


def test_run_baseline_flag_plumbing_with_stub_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The dry_run OVERRIDES (subset/epochs/steps/patience) + metadata recording, exercised
    with a synthetic corpus tree + stubbed engine boundary: `speech_rs.load_toml_config`,
    `train_modern` (via `_train_fn`), and the held-out scorer are all replaced, so no engine
    or corpus is touched. Proves run_baseline wires the flags into the metadata + params."""
    root = tmp_path / "corpus"
    lre = root / "train" / "LID_Features" / "plp8f0mvsdd" / "LRE03"
    lre.mkdir(parents=True)
    for lang in B._LANGS:
        for i in range(4):
            (lre / f"{lang}_{i}.plp8f0mvsdd").write_bytes(b"")

    # stub the engine boundary: a fake speech_rs module + a stub trainer + a no-op scorer.
    import sys
    import types

    fake_rs = types.ModuleType("speech_rs")
    fake_flat = {
        "Algo_choice": "6",
        "BLSTM_NNetInputSize": "11",
        "BLSTM_LID_NNetInputSize": "23",
        "fileslisting": "x",
        "language2classmapping": "y",
        "BLSTM_weightsFile": "s",
        "BLSTM_LID_weightsFile": "l",
        "numOuterThreads": "1",
    }
    setattr(fake_rs, "load_toml_config", lambda p: fake_flat)  # noqa: B010 -- dynamic attr on a fake module
    monkeypatch.setitem(sys.modules, "speech_rs", fake_rs)
    monkeypatch.setattr(B, "_generate_seed_packs", lambda flat, out, seed, init_scheme, forget_bias_one: None)
    monkeypatch.setattr(B, "RunState", _stub_run_state())

    class _Res:
        checkpoint_dir = str(tmp_path / "out" / "checkpoint")
        epochs_run, best_epoch, best_val_cost = 1, 0, 1.0
        history: list[object] = []

    monkeypatch.setattr(B, "_score_packs_on_test", lambda *a, **k: (None, None, None))

    out = tmp_path / "out"
    (tmp_path / "out" / "checkpoint").mkdir(parents=True)
    seen: list[ModernTrainParams] = []

    def stub_train(state: object, seed: int, params: ModernTrainParams) -> _Res:
        seen.append(params)
        return _Res()

    res = B.run_baseline(_spec("lid-features", root, out, dry_run=True, seed=5, lanes=2, minibatch=4, score_init=True), _train_fn=stub_train)

    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1
    assert meta["seed"] == 5 and meta["lanes"] == 2
    assert meta["minibatch"] == 4 and meta["valid_size"] == 12 and meta["test_size"] == 24 and meta["score_init"] is True
    assert meta["lre_listing"] is None and meta["lre_listing_hash"] is None and meta["resume"] is False
    assert res.n_train > 0  # a tiny subset was drawn from the synthetic tree

    # A localized listing: the manifest keeps its path (local only) and its content hash (the recipe's `listing`).
    listing = tmp_path / "lre03_train.csv"
    listing.write_text("a,b\n")
    monkeypatch.setattr(B, "localize_listing", lambda src, corpus_root, out: types.SimpleNamespace(rows_found=1, rows_total=1, rows_missing=0))
    monkeypatch.setattr(B, "_records_from_localized", lambda localized, ref_stm: B.derive_lid_features_records(root, ref_stm))
    B.run_baseline(_spec("lid-features", root, tmp_path / "out2", dry_run=True, lre_listing=listing), _train_fn=stub_train)
    meta2 = json.loads((tmp_path / "out2" / "run_metadata.json").read_text())
    assert meta2["lre_listing"] == str(listing) and meta2["lre_listing_hash"] == B.listing_hash(listing)

    # The mini-batch rotation built from these params draws every one of the 12 classes (the
    # multilingual layout parked class 11 in a never-read aggregate slot: `vie` never trained).
    params = seen[0]
    fv = np.repeat(np.arange(len(B._LANGS), dtype=np.float64), 3).reshape(-1, 1)
    batches = create_batches(fv, params.minibatch, params.nb_worst, params.multilingual, params.nb_classes, np.random.default_rng(0))
    drawn: set[int] = set()
    for _ in range(20):
        idx, batches = get_new_batch(batches)
        drawn.update(int(fv[i, 0]) for i in idx)
    assert drawn == set(range(len(B._LANGS)))


def _stub_phseq_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, type]:
    """A synthetic phSeq tree plus the engine boundary stubbed (a fake `speech_rs`, no seed packs, no
    RunState, no scorer): `(corpus root, out dir, a train-result class for a `_train_fn` stub)`."""
    import sys
    import types

    root = tmp_path / "corpus"
    phseq = root / "train" / "phSeq"
    phseq.mkdir(parents=True)
    for prefix in ("ar", "ma", "en", "sp", "vi", "ta", "ko", "ja", "hi", "ge", "fr", "fa"):
        for i in range(4):
            (phseq / f"{prefix}_{i}.file.phSeqbis").write_text(".a.\n")  # synthetic (fake id), no real corpus filename

    fake_rs = types.ModuleType("speech_rs")
    fake_flat = {
        "Algo_choice": "6",
        "File_Type": "1",
        "BLSTM_NNetInputSize": "11",
        "BLSTM_LID_NNetInputSize": "38",
        "BLSTM_LID_LSTMNeuronNb": "38,24",
        "BLSTM_LID_OutputNeuronNb": "48,12",
        "fileslisting": "x",
        "language2classmapping": "y",
        "BLSTM_weightsFile": "s",
        "BLSTM_LID_weightsFile": "l",
        "numOuterThreads": "1",
    }
    setattr(fake_rs, "load_toml_config", lambda p: fake_flat)  # noqa: B010 -- dynamic attr on a fake module
    monkeypatch.setitem(sys.modules, "speech_rs", fake_rs)
    monkeypatch.setattr(B, "_generate_seed_packs", lambda flat, out, seed, init_scheme, forget_bias_one: None)
    monkeypatch.setattr(B, "RunState", _stub_run_state())
    monkeypatch.setattr(B, "_score_packs_on_test", lambda *a, **k: (None, None, None))

    class _Res:
        checkpoint_dir = str(tmp_path / "out" / "checkpoint")
        epochs_run, best_epoch, best_val_cost = 1, 0, 1.0
        history: list[object] = []

    out = tmp_path / "out"
    (out / "checkpoint").mkdir(parents=True)
    return root, out, _Res


def test_run_baseline_lid_phseq_flag_plumbing_with_stub_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The lid-phseq dispatch (phSeq tree glob, 12-class Twin, File_Type 1, LID input 38) +
    dry_run overrides + metadata, with the engine boundary stubbed. Proves run_baseline routes
    the phonotactic arm through the shared LID path (both nets seeded, `.scr` scorer) without
    touching the engine/corpus, and that the config it assembles carries File_Type 1."""
    root, out, _Res = _stub_phseq_engine(tmp_path, monkeypatch)
    res = B.run_baseline(_spec("lid-phseq", root, out, dry_run=True, seed=5, lanes=2), _train_fn=lambda state, seed, params: _Res())

    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["arm"] == "lid-phseq" and meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1
    assert meta["seed"] == 5 and meta["lanes"] == 2
    assert meta["config_toml"] == "configs/training/lre03_lid_phseq.toml"
    assert res.n_train > 0
    # the phonotactic arm is a Twin (both nets seeded) at File_Type 1.
    cfg_text = (out / "base.config").read_text()
    assert "File_Type 1" in cfg_text and "BLSTM_LID_weightsFile lid_seed.bin" in cfg_text
    # listings carry the phseq stem.
    assert (out / "lre03_lid_phseq_train.flst").is_file()


def test_run_baseline_records_the_provenance_of_the_tree_it_started_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #40: a commit made while the run trains must not become the record's SHA. The stub
    trainer moves the tree's state mid-run; `record.json` and the manifest keep the start's."""
    root, out, _Res = _stub_phseq_engine(tmp_path, monkeypatch)
    tree = {"state": ("a" * 40, False)}
    monkeypatch.setattr(B, "git_state", lambda repo: tree["state"])

    def train_then_commit(state: object, seed: int, params: object) -> object:
        tree["state"] = ("b" * 40, True)
        return _Res()

    res = B.run_baseline(_spec("lid-phseq", root, out, subset=12, valid_size=0, test_size=0, epochs=1, steps_per_epoch=1), _train_fn=train_then_commit)
    record = json.loads((out / "record.json").read_text())
    meta = json.loads(res.metadata_path.read_text())
    assert (record["git_sha"], record["git_dirty"]) == ("a" * 40, False)
    assert (meta["git_sha"], meta["git_dirty"]) == ("a" * 40, False)


def test_run_baseline_fails_on_git_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #40: a tree whose provenance cannot be read stops the run before any training, and
    before the run directory exists (no half-started run on disk)."""
    root, _, _Res = _stub_phseq_engine(tmp_path, monkeypatch)

    def no_git(repo: Path) -> tuple[str, bool]:
        raise RuntimeError("git unavailable")

    trained: list[int] = []
    monkeypatch.setattr(B, "git_state", no_git)
    out = tmp_path / "never_created"
    with pytest.raises(RuntimeError, match="git unavailable"):
        B.run_baseline(_spec("lid-phseq", root, out, dry_run=True), _train_fn=lambda state, seed, params: trained.append(seed))
    assert trained == [] and not out.exists()


def test_a_dry_run_writes_its_manifest_but_no_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run is a smoke, not a measurement: the ledger has no dry-run field, so the only way
    a 1-step run cannot be promoted as a launcher row is for it to leave no `record.json`."""
    root, out, _Res = _stub_phseq_engine(tmp_path, monkeypatch)
    train = lambda state, seed, params: _Res()  # noqa: E731
    B.run_baseline(_spec("lid-phseq", root, out, dry_run=True), _train_fn=train)
    assert (out / "run_metadata.json").is_file() and not (out / "record.json").exists()
    out2 = tmp_path / "out2"
    B.run_baseline(_spec("lid-phseq", root, out2, subset=12, valid_size=0, test_size=0, epochs=1, steps_per_epoch=1), _train_fn=train)
    assert (out2 / "record.json").is_file()


def test_an_out_dir_holding_a_run_is_refused_unless_resuming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An overwritten full run is a lost record: the launcher refuses a run directory that
    already holds a run's manifest unless the call resumes from it."""
    root, out, _Res = _stub_phseq_engine(tmp_path, monkeypatch)
    train = lambda state, seed, params: _Res()  # noqa: E731
    B.run_baseline(_spec("lid-phseq", root, out, dry_run=True), _train_fn=train)
    trained: list[int] = []
    with pytest.raises(FileExistsError, match="out_dir|resume"):
        B.run_baseline(_spec("lid-phseq", root, out, dry_run=True), _train_fn=lambda state, seed, params: trained.append(seed))
    assert trained == []
    B.run_baseline(_spec("lid-phseq", root, out, dry_run=True, resume=True), _train_fn=train)


# --------------------------------------------------------------------------------------- #
# The LID cell knob (issue #21): `--cell`/`--direction` on a LID arm target the net that
# trains, through the same overlay under the `BLSTM_LID` prefix; the frozen SAD net stays
# the legacy LSTM at its seed, byte-identical across every LID cell.
# --------------------------------------------------------------------------------------- #

_TWIN_FLAT: dict[str, str] = {
    "BLSTM_LSTMNeuronNb": "23,24,24",
    "BLSTM_LSTMSubSampling": "4,1",
    "BLSTM_OutputNeuronNb": "48,12,1",
    "BLSTM_OutputSubSampling": "1,1",
    "BLSTM_NNetInputSize": "23",
    "BLSTM_CostLawSpeech": "log",
    "BLSTM_CostLawNoSpeech": "log",
    "BLSTM_LID_LSTMNeuronNb": "38,24",
    "BLSTM_LID_LSTMSubSampling": "1",
    "BLSTM_LID_OutputNeuronNb": "48,12",
    "BLSTM_LID_OutputSubSampling": "1",
    "BLSTM_LID_NNetInputSize": "38",
    "BLSTM_LID_CostLawSpeech": "log",
    "BLSTM_LID_CostLawNoSpeech": "log",
}


def test_cell_overlay_targets_the_lid_net_under_its_prefix() -> None:
    """The committed LID TOMLs declare `lstm_neuron_nb 38,24` (phSeq) and `output_neuron_nb
    48,12`; a forward LID net needs the output MLP's first width at the last hidden width (24)
    and the plain whole-utterance regime (`window 0`), the two derived keys the SAD overlay
    already writes, under the LID prefix. Bidirectional keeps the TOML's 0.25 s window."""
    assert B.cell_overlay(_TWIN_FLAT, "slstm", "bidirectional", prefix="BLSTM_LID") == {"BLSTM_LID_Cell_Type": "slstm"}
    assert B.cell_overlay(_TWIN_FLAT, "mamba", "forward", prefix="BLSTM_LID") == {
        "BLSTM_LID_Cell_Type": "mamba",
        "BLSTM_LID_Direction": "forward",
        "BLSTM_LID_OutputNeuronNb": "24,12",
        "BLSTM_LID_window": "0",
    }
    assert B.cell_overlay(_TWIN_FLAT, "lstm", "bidirectional", prefix="BLSTM_LID") == {}
    # the SAD overlay is untouched by the parameter's default.
    assert B.cell_overlay(_TWIN_FLAT, "lstm", "forward") == {"BLSTM_Direction": "forward", "BLSTM_OutputNeuronNb": "24,12,1", "BLSTM_window": "0"}


def test_lid_arm_knob_lands_on_the_lid_net_and_leaves_the_sad_net_lstm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, out, _Res = _stub_phseq_engine(tmp_path, monkeypatch)
    res = B.run_baseline(_spec("lid-phseq", root, out, dry_run=True, cell="mamba", direction="forward"), _train_fn=lambda state, seed, params: _Res())
    cfg_text = (out / "base.config").read_text()
    assert "BLSTM_LID_Cell_Type mamba" in cfg_text and "BLSTM_LID_Direction forward" in cfg_text
    assert "BLSTM_LID_OutputNeuronNb 24,12" in cfg_text and "BLSTM_LID_window 0" in cfg_text
    assert "BLSTM_Cell_Type" not in cfg_text and "BLSTM_Direction" not in cfg_text and "BLSTM_OutputNeuronNb 48,12,1" not in cfg_text
    meta = json.loads(res.metadata_path.read_text())
    assert meta["cell"] == "mamba" and meta["direction"] == "forward"


def test_sad_seed_pack_is_byte_identical_across_lid_cells(tmp_path: Path) -> None:
    """Both nets are drawn from one generator in `[sad, lid]` order, so the SAD seed never
    sees the LID cell: every LID row of a given seed starts from the same SAD pack, which is
    what makes the rows comparable. The LID pack itself changes with the cell."""
    sad_packs: dict[tuple[str, str], bytes] = {}
    lid_lengths: dict[tuple[str, str], int] = {}
    for cell in CELL_TYPES:
        for direction in DIRECTIONS:
            cfg = {**_TWIN_FLAT, **B.cell_overlay(_TWIN_FLAT, cell, direction, prefix="BLSTM_LID")}
            out = tmp_path / f"{cell}_{direction}"
            out.mkdir()
            B._generate_seed_packs(cfg, out, 0, "xavier", True)
            sad_packs[cell, direction] = (out / "sad_seed.bin").read_bytes()
            lid_lengths[cell, direction] = len((out / "lid_seed.bin").read_bytes())
    assert len(set(sad_packs.values())) == 1
    assert len(set(lid_lengths.values())) == len(lid_lengths), lid_lengths


def test_speech_cli_mounts_baseline_lid_phseq(monkeypatch: pytest.MonkeyPatch) -> None:
    import speech.cli as cli

    spec = _forwarded(monkeypatch, cli.main, ["baseline", "lid-phseq", "--corpus-root", "/c", "--out-dir", "/o", "--subset", "24"])
    assert spec.arm == "lid-phseq" and spec.subset == 24


# --------------------------------------------------------------------------------------- #
# SAD arm units (Task 9): single-net config assembly, the VRCTS ref adapter + pooled DCF,
# split plumbing over a synthetic wav/xml tree, CLI wiring. Every fixture synthetic.
# --------------------------------------------------------------------------------------- #


def _vrcts_xml(sigdur: float, segs: list[tuple[float, float]]) -> str:
    """A minimal VRCTS xml (the AudioDoc/Channel/SpeechSegment shape `load_vrcts_hyp` reads)."""
    lines = "".join(f'<SpeechSegment ch="1" sconf="1.00" stime="{s:.2f}" etime="{e:.2f}" spkid="1"/>' for s, e in segs)
    return f'<AudioDoc name="x"><ChannelList><Channel num="1" sigdur="{sigdur:.2f}" spdur="0.00"/></ChannelList><SegmentList>{lines}</SegmentList></AudioDoc>'


def test_assemble_flat_config_single_net_no_lid_seed() -> None:
    base = {"fileslisting": "P", "language2classmapping": "P", "BLSTM_weightsFile": "x", "numOuterThreads": "1", "Audio_max_duration": "120"}
    cfg = B.assemble_flat_config(base, fileslisting="t.flst", mapping="m.csv", sad_seed="s.bin", lanes=2, extra={"Audio_max_duration": "20.0"})
    # single-net: NO BLSTM_LID_weightsFile is invented (that key would dangle on an algo-3 config).
    assert cfg["BLSTM_weightsFile"] == "s.bin" and "BLSTM_LID_weightsFile" not in cfg
    assert cfg["fileslisting"] == "t.flst" and cfg["language2classmapping"] == "m.csv" and cfg["numOuterThreads"] == "2"
    # extra overlays the audio cap in place (the key already existed, so its position is kept).
    assert cfg["Audio_max_duration"] == "20.0"
    assert base["Audio_max_duration"] == "120", "assemble_flat_config must not mutate its input"


def test_write_sad_mapping(tmp_path: Path) -> None:
    p = tmp_path / "sad_mapping.csv"
    B.write_sad_mapping(p)
    assert p.read_text() == "unk;unk;0\n"


def test_hyp_xml_for_strips_extension() -> None:
    d = Path("/scores")
    # strip_last_4 (drop the 4-char .wav), then .xml -- the engine's own dump-target rule.
    # (synthetic names: multi-dot LRE03-style + single-dot LRE07-style, no real corpus file.)
    assert B._hyp_xml_for("/a/b/zz_0000_a.MT1.mp1.wav", d) == d / "zz_0000_a.MT1.mp1.xml"
    assert B._hyp_xml_for("ZZ-000000-A-con.wav", d) == d / "ZZ-000000-A-con.xml"


def test_windowed_vrcts_ref_clips_and_pads(tmp_path: Path) -> None:
    x = tmp_path / "ref.xml"
    x.write_text(_vrcts_xml(100.0, [(10.0, 20.0), (30.0, 40.0)]))
    # clip mid-gap: the ref ends EXACTLY at the window (no trailing pad needed).
    assert B._windowed_vrcts_ref(x, 25.0) == [(0.0, 10.0, "NS"), (10.0, 20.0, "S"), (20.0, 25.0, "NS")]
    # window past sigdur: the ref pads with NS to the window end (so pooling has no gap).
    r = B._windowed_vrcts_ref(x, 120.0)
    assert r[0] == (0.0, 10.0, "NS") and r[-1] == (100.0, 120.0, "NS")
    # a window before the first segment is one whole NS span.
    assert B._windowed_vrcts_ref(x, 5.0) == [(0.0, 5.0, "NS")]


def test_pool_dcf_perfect_and_false_alarm() -> None:
    # file1: speech, correctly detected. file2: nonspeech, all flagged speech.
    # pooled -> speech_sum=10 fn=0, nonspeech_sum=10 fp=10 -> Pmiss 0, Pfa 1, DCF=0.25 (no-collar).
    f1 = ([(0.0, 10.0, "S")], [(0.0, 10.0, "speech")])
    f2 = ([(0.0, 10.0, "NS")], [(0.0, 10.0, "speech")])
    c = B._pool_dcf([f1, f2], (0.0,)).by_collar(0.0)
    assert abs(c.pmiss - 0.0) < 1e-9 and abs(c.pfa - 1.0) < 1e-9 and abs(c.dcf - 0.25) < 1e-9


def _sad_corpus_tree(root: Path, n: int) -> None:
    audio = root / "train" / "audio" / "LRE03"
    audio.mkdir(parents=True)
    for i in range(n):
        (audio / f"xx_{i:03d}.MT1.mp1.wav").write_bytes(b"")
        (audio / f"xx_{i:03d}.part.xml").write_text(_vrcts_xml(60.0, [(1.0, 5.0)]))


def test_prepare_sad_listings_split_plumbing(tmp_path: Path) -> None:
    from rich.console import Console

    root = tmp_path / "corpus"
    _sad_corpus_tree(root, 40)  # derive_sad_listings 70/15/15 -> 28/6/6
    out = tmp_path / "out"
    out.mkdir()
    sp = B._prepare_sad_listings(root, out, seed=0, subset=8, valid_size=3, test_size=4, console=Console())
    tr, va, te = sp.train, sp.valid, sp.test
    assert len(tr) == 8 and len(va) == 3 and len(te) == 4  # first-N subsets of each split
    assert (sp.train_name, sp.valid_name, sp.test_name, sp.mapping_name) == (
        "sad_train_subset.flst",
        "sad_valid_subset.flst",
        "sad_test_subset.flst",
        "sad_mapping.csv",
    )
    assert (out / sp.train_name).is_file() and (out / sp.mapping_name).read_text() == "unk;unk;0\n" and sp.n_classes == 1
    trf, vaf, tef = ({r["filename"] for r in s} for s in (tr, va, te))
    assert trf.isdisjoint(vaf) and trf.isdisjoint(tef) and vaf.isdisjoint(tef), "the subsets stay disjoint (the base split is)"
    assert all(r["refseg"].endswith(".part.xml") for r in tr), "SAD refs are the corpus VRCTS .part.xml files"


def test_run_baseline_sad_flag_plumbing_with_stub_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SAD arm dispatch (single-net config, no LID seed, audio cap) + dry_run overrides +
    metadata, with the engine boundary stubbed (`speech_rs`, seed-pack init, `RunState`, the
    DCF scorer). Proves run_baseline routes the sad arm without touching the engine/corpus."""
    import sys
    import types

    root = tmp_path / "corpus"
    _sad_corpus_tree(root, 12)

    fake_rs = types.ModuleType("speech_rs")
    fake_flat = {
        "Algo_choice": "3",
        "BLSTM_NNetInputSize": "23",
        "fileslisting": "x",
        "language2classmapping": "y",
        "BLSTM_weightsFile": "s",
        "numOuterThreads": "1",
        "Audio_max_duration": "120",
    }
    setattr(fake_rs, "load_toml_config", lambda p: fake_flat)  # noqa: B010 -- dynamic attr on a fake module
    monkeypatch.setitem(sys.modules, "speech_rs", fake_rs)
    monkeypatch.setattr(B, "_generate_sad_seed_pack", lambda flat, out, seed, scheme, forget: None)
    monkeypatch.setattr(B, "RunState", _stub_run_state())
    monkeypatch.setattr(B, "_score_sad_pack_on_test", lambda *a, **k: (None, None))

    class _Res:
        checkpoint_dir = str(tmp_path / "out" / "checkpoint")
        epochs_run, best_epoch, best_val_cost = 1, 0, 1.0
        history: list[object] = []

    out = tmp_path / "out"
    (out / "checkpoint").mkdir(parents=True)
    res = B.run_baseline(_spec("sad", root, out, dry_run=True, seed=5, lanes=2, audio_max_duration=15.0), _train_fn=lambda state, seed, params: _Res())

    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["arm"] == "sad" and meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1
    assert meta["seed"] == 5 and meta["lanes"] == 2 and meta["audio_max_duration"] == 15.0
    assert res.n_train > 0
    # single-net config: no LID weight-file key, and the audio cap threaded into base.config.
    cfg_text = (out / "base.config").read_text()
    assert "BLSTM_LID_weightsFile" not in cfg_text and "Audio_max_duration 15.0" in cfg_text


def test_build_parser_sad_audio_cap_wiring() -> None:
    spec = BaselineSpec.from_args(B.build_parser().parse_args(["sad", "--corpus-root", "/c", "--out-dir", "/o", "--audio-max-duration", "20.0"]))
    assert spec.arm == "sad" and spec.audio_max_duration == 20.0


def test_baseline_main_sad_forwards_audio_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _forwarded(monkeypatch, B.main, ["sad", "--corpus-root", "/c", "--out-dir", "/o", "--audio-max-duration", "30.0"])
    assert spec.arm == "sad" and spec.audio_max_duration == 30.0


def test_speech_cli_mounts_baseline_sad(monkeypatch: pytest.MonkeyPatch) -> None:
    import speech.cli as cli

    spec = _forwarded(monkeypatch, cli.main, ["baseline", "sad", "--corpus-root", "/c", "--out-dir", "/o", "--audio-max-duration", "25.0"])
    assert spec.arm == "sad" and spec.audio_max_duration == 25.0
