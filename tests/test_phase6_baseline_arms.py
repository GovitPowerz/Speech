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
    for name in ("ara_1.plp8f0mvsdd", "spa_9.plp8f0mvsdd", "vie_3.plp8f0mvsdd", "zzz_1.plp8f0mvsdd"):
        (root / name).write_bytes(b"")  # empty stand-in; derivation only reads names
    ref = tmp_path / "ref.stm"
    recs = B.derive_lid_features_records(tmp_path, ref)
    langs = sorted(r["lang"] for r in recs)
    assert langs == ["ara", "spa", "vie"], "only the 12 known LRE03 languages survive (zzz dropped)"
    assert all(r["refseg"] == str(ref) and r["dial"] == "non" for r in recs)
    assert all(Path(r["filename"]).is_absolute() for r in recs)


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


def test_unimplemented_arm_raises(tmp_path: Path) -> None:
    # sad / lid-phseq are declared later (Tasks 9/10); only lid-features is wired now.
    B._ARM_CONFIGS["sad"] = "configs/training/lre_sad.toml"  # simulate a later-task entry
    try:
        with pytest.raises(NotImplementedError):
            B.run_baseline("sad", tmp_path, tmp_path)
    finally:
        del B._ARM_CONFIGS["sad"]


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
    monkeypatch.setattr(B, "_generate_seed_packs", lambda flat, out, seed: None)
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
