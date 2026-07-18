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

import json
from pathlib import Path

import pytest
from speech.drivers import baseline as B

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


def test_stratified_splits_full_train_when_none() -> None:
    recs = _records({"ara": 20, "chi": 20})
    train, valid, test = B.stratified_splits(recs, n_train=None, n_valid=4, n_test=4, seed=0)
    # n_train=None: train takes ALL remaining after valid+test are carved (priority train
    # first -> valid -> test); with plenty of files every split is filled and disjoint.
    assert len(train) + len(valid) + len(test) <= len(recs)
    tr, va, te = ({r["filename"] for r in s} for s in (train, valid, test))
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(train) > 0


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
    assert args.arm == "lid-features"
    assert args.corpus_root == Path("/c") and args.out_dir == Path("/o")
    assert args.subset == 36 and args.lanes == 2 and args.seed == 7
    assert args.dry_run is True and args.resume is True


def test_baseline_main_flag_plumbing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(arm: str, corpus_root: Path, out_dir: Path, **kw: object) -> None:
        captured["arm"], captured["corpus_root"], captured["out_dir"] = arm, corpus_root, out_dir
        captured.update(kw)

    monkeypatch.setattr(B, "run_baseline", fake_run)
    rc = B.main(["lid-features", "--corpus-root", "/c", "--out-dir", "/o", "--subset", "50", "--dry-run", "--resume", "--lanes", "3"])
    assert rc == 0
    assert captured["arm"] == "lid-features"
    assert captured["subset"] == 50
    assert captured["dry_run"] is True and captured["resume"] is True and captured["lanes"] == 3


def test_speech_cli_mounts_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    import speech.cli as cli

    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_baseline", lambda arm, cr, od, **kw: captured.update({"arm": arm, **kw}) or None)
    rc = cli.main(["baseline", "lid-features", "--corpus-root", "/c", "--out-dir", "/o", "--subset", "12"])
    assert rc == 0 and captured["arm"] == "lid-features" and captured["subset"] == 12


def test_unknown_arm_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        B.run_baseline("nonsense", tmp_path, tmp_path)


def test_all_three_arms_wired() -> None:
    # Task 10 completes the arm set: lid-features, sad, lid-phseq are all registered AND
    # dispatched (none raise NotImplementedError anymore). A registered arm that reached the
    # NotImplementedError guard would be a wiring gap; there are none left.
    assert set(B._ARM_CONFIGS) == {"lid-features", "sad", "lid-phseq"}
    assert set(B._LID_ARMS) == {"lid-features", "lid-phseq"}


def test_lid_phseq_dispatches_into_lid_path(tmp_path: Path) -> None:
    # lid-phseq is WIRED (Task 10): it no longer hits the NotImplementedError guard. On an
    # empty corpus tree it proceeds into the LID record-derivation path and raises the
    # "no LID records" RuntimeError with the phSeq-specific hint -- proving it dispatched.
    with pytest.raises(RuntimeError, match=r"no LID records.*phSeq"):
        B.run_baseline("lid-phseq", tmp_path, tmp_path)


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
    monkeypatch.setattr(B, "RunState", type("RS", (), {"from_config": staticmethod(lambda *a, **k: object())}))

    class _Res:
        checkpoint_dir = str(tmp_path / "out" / "checkpoint")
        epochs_run, best_epoch, best_val_cost = 1, 0, 1.0
        history: list[object] = []

    monkeypatch.setattr(B, "_score_packs_on_test", lambda *a, **k: (None, None, None))

    out = tmp_path / "out"
    (tmp_path / "out" / "checkpoint").mkdir(parents=True)
    res = B.run_baseline("lid-features", root, out, dry_run=True, seed=5, lanes=2, _train_fn=lambda state, seed, params: _Res())

    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1
    assert meta["seed"] == 5 and meta["lanes"] == 2
    assert res.n_train > 0  # a tiny subset was drawn from the synthetic tree


def test_run_baseline_lid_phseq_flag_plumbing_with_stub_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The lid-phseq dispatch (phSeq tree glob, 12-class Twin, File_Type 1, LID input 38) +
    dry_run overrides + metadata, with the engine boundary stubbed. Proves run_baseline routes
    the phonotactic arm through the shared LID path (both nets seeded, `.scr` scorer) without
    touching the engine/corpus, and that the config it assembles carries File_Type 1."""
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
        "fileslisting": "x",
        "language2classmapping": "y",
        "BLSTM_weightsFile": "s",
        "BLSTM_LID_weightsFile": "l",
        "numOuterThreads": "1",
    }
    setattr(fake_rs, "load_toml_config", lambda p: fake_flat)  # noqa: B010 -- dynamic attr on a fake module
    monkeypatch.setitem(sys.modules, "speech_rs", fake_rs)
    monkeypatch.setattr(B, "_generate_seed_packs", lambda flat, out, seed, init_scheme, forget_bias_one: None)
    monkeypatch.setattr(B, "RunState", type("RS", (), {"from_config": staticmethod(lambda *a, **k: object())}))
    monkeypatch.setattr(B, "_score_packs_on_test", lambda *a, **k: (None, None, None))

    class _Res:
        checkpoint_dir = str(tmp_path / "out" / "checkpoint")
        epochs_run, best_epoch, best_val_cost = 1, 0, 1.0
        history: list[object] = []

    out = tmp_path / "out"
    (out / "checkpoint").mkdir(parents=True)
    res = B.run_baseline("lid-phseq", root, out, dry_run=True, seed=5, lanes=2, _train_fn=lambda state, seed, params: _Res())

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


def test_speech_cli_mounts_baseline_lid_phseq(monkeypatch: pytest.MonkeyPatch) -> None:
    import speech.cli as cli

    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_baseline", lambda arm, cr, od, **kw: captured.update({"arm": arm, **kw}) or None)
    rc = cli.main(["baseline", "lid-phseq", "--corpus-root", "/c", "--out-dir", "/o", "--subset", "24"])
    assert rc == 0 and captured["arm"] == "lid-phseq" and captured["subset"] == 24


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
    tr, va, te, tn, vn, ten, mn = B._prepare_sad_listings(root, out, seed=0, subset=8, valid_size=3, test_size=4, console=Console())
    assert len(tr) == 8 and len(va) == 3 and len(te) == 4  # first-N subsets of each split
    assert (tn, vn, ten, mn) == ("sad_train_subset.flst", "sad_valid_subset.flst", "sad_test_subset.flst", "sad_mapping.csv")
    assert (out / tn).is_file() and (out / mn).read_text() == "unk;unk;0\n"
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
    monkeypatch.setattr(B, "RunState", type("RS", (), {"from_config": staticmethod(lambda *a, **k: object())}))
    monkeypatch.setattr(B, "_score_sad_pack_on_test", lambda *a, **k: (None, None))

    class _Res:
        checkpoint_dir = str(tmp_path / "out" / "checkpoint")
        epochs_run, best_epoch, best_val_cost = 1, 0, 1.0
        history: list[object] = []

    out = tmp_path / "out"
    (out / "checkpoint").mkdir(parents=True)
    res = B.run_baseline("sad", root, out, dry_run=True, seed=5, lanes=2, audio_max_duration=15.0, _train_fn=lambda state, seed, params: _Res())

    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["arm"] == "sad" and meta["dry_run"] is True and meta["epochs"] == 1 and meta["steps_per_epoch"] == 1
    assert meta["seed"] == 5 and meta["lanes"] == 2 and meta["audio_max_duration"] == 15.0
    assert res.n_train > 0
    # single-net config: no LID weight-file key, and the audio cap threaded into base.config.
    cfg_text = (out / "base.config").read_text()
    assert "BLSTM_LID_weightsFile" not in cfg_text and "Audio_max_duration 15.0" in cfg_text


def test_build_parser_sad_audio_cap_wiring() -> None:
    args = B.build_parser().parse_args(["sad", "--corpus-root", "/c", "--out-dir", "/o", "--audio-max-duration", "20.0"])
    assert args.arm == "sad" and args.audio_max_duration == 20.0


def test_baseline_main_sad_forwards_audio_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(arm: str, corpus_root: Path, out_dir: Path, **kw: object) -> None:
        captured["arm"] = arm
        captured.update(kw)

    monkeypatch.setattr(B, "run_baseline", fake_run)
    rc = B.main(["sad", "--corpus-root", "/c", "--out-dir", "/o", "--audio-max-duration", "30.0"])
    assert rc == 0 and captured["arm"] == "sad" and captured["audio_max_duration"] == 30.0


def test_speech_cli_mounts_baseline_sad(monkeypatch: pytest.MonkeyPatch) -> None:
    import speech.cli as cli

    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_baseline", lambda arm, cr, od, **kw: captured.update({"arm": arm, **kw}) or None)
    rc = cli.main(["baseline", "sad", "--corpus-root", "/c", "--out-dir", "/o", "--audio-max-duration", "25.0"])
    assert rc == 0 and captured["arm"] == "sad" and captured["audio_max_duration"] == 25.0
