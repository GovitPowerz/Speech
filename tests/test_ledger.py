"""The measurement ledger (issue #20, ADR-0009): schema, hygiene, identity, selection, the
RESULTS.md renderers and the `add` / `render --check` commands, on synthetic records; plus the
two standing gates over the committed tree: every record under `ledger/` validates and names
a release build, and RESULTS.md holds exactly what the ledger renders.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from speech.drivers import baseline as B
from speech.evaluate import CollarScore as EvalCollar
from speech.evaluate import DcfReport
from speech.ledger import cli, prose, schema
from speech.ledger.bench import BINARY, bench_json, lanes_of, wrap
from speech.ledger.render import RESULTS, render_file, render_text, table_names
from speech.ledger.schema import LEDGER_DIR, BaselinePayload, BaselineRecipe, BaselineRecord, BenchPayload, BenchRecipe, BenchRecord, Build, CollarScore, Host
from speech.ledger.stage import LEGS, sorted_first, stage_fixture_60s
from speech.ledger.tables import TABLES, bench_cells, current_baselines, speedups

from tests.conftest import CORPUS_ROOT, requires_corpus

requires_binary = pytest.mark.skipif(not BINARY.is_file(), reason=f"release binary absent at {BINARY}")

REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)
BUILD = Build(profile="release", target="aarch64-apple-darwin", rustc="rustc 1.99.0 (b940084d7 2026-09-28)")
HOST = Host(chip="Apple M4 Pro", arch="arm64", cores=14, os="Darwin 25.6.0")


def _collars(dcf: float, pmiss: float, pfa: float) -> list[CollarScore]:
    return [CollarScore(collar=c, dcf=dcf, pmiss=pmiss, pfa=pfa) for c in (0.0, 0.25, 0.5, 1.0, 2.0)]


def sad_gate(cell: str = "lstm", direction: str = "bidirectional", *, lineage: str = "v1", at: datetime = T0, **over: object) -> BaselineRecord:
    arm = "sad" if lineage == "v1" else "sad-v2"
    recipe = dict(
        arm=arm,
        lineage=lineage,
        cell=cell,
        direction=direction,
        subset=10,
        valid_size=8,
        test_size=24,
        audio_max_duration=20.0,
        epochs=3,
        steps_per_epoch=10,
        patience=99,
        minibatch=0,
        init_scheme="xavier",
        seed=0,
        lanes=1,
    )
    recipe.update({k: v for k, v in over.items() if k in recipe})
    payload = BaselinePayload(
        source="gate",
        test=f"tests/pyo3/test_gates.py::test_gate[{cell}-{direction}]",
        config_name=f"configs/training/{'lre_sad' if lineage == 'v1' else 'lre_sad_v2'}.toml",
        config_hash="0123456789abcdef",
        n_train=10,
        n_valid=8,
        n_test=24,
        val_metric="nn_cost_seg",
        epochs_run=3,
        best_epoch=1,
        wall_s=37.2,
        first_val_cost=0.5,
        best_val_cost=0.1,
        first_train_cost=0.6,
        last_train_cost=0.05,
        dcf=_collars(0.25, 0.0, 1.0),
        init_dcf=_collars(0.75, 1.0, 0.0),
        lid_error=None,
        cavg=None,
        init_lid_error=None,
        init_cavg=None,
    )
    payload = payload.model_copy(update={k: v for k, v in over.items() if k in BaselinePayload.model_fields})
    envelope = dict(recorded_at=at, git_sha="a" * 40, git_dirty=False, build=BUILD, host=HOST)
    envelope.update({k: v for k, v in over.items() if k in ("recorded_at", "git_sha", "git_dirty", "build", "host", "supersedes", "reason")})
    return BaselineRecord(recipe=BaselineRecipe(**recipe), payload=payload, **envelope)  # type: ignore[arg-type]


def lid_gate(arm: str = "lid-features", **over: object) -> BaselineRecord:
    r = sad_gate(**over)  # type: ignore[arg-type]
    recipe = r.recipe.model_copy(
        update={"arm": arm, "lineage": None, "subset": 36, "valid_size": 12, "test_size": 48, "audio_max_duration": None, "epochs": 2, "steps_per_epoch": 25}
    )
    payload = r.payload.model_copy(
        update={
            "config_name": f"configs/training/lre03_{arm.replace('-', '_')}.toml",
            "n_train": 35,
            "n_valid": 15,
            "n_test": 48,
            "dcf": None,
            "init_dcf": None,
            "lid_error": 72.92,
            "cavg": 0.46,
            "init_lid_error": 93.75,
            "init_cavg": 0.53,
        }
    )
    return r.model_copy(update={"recipe": recipe, "payload": payload})


def bench_record(label: str = "phase7_60s", path: str = "fast", wall: float = 0.06, *, at: datetime = T0, sha: str = "b" * 40, **over: object) -> BenchRecord:
    audio = {"phase7_60s": 120.0, "phase7_sad_corpus": 75.0, "phase7_lid_phseq": 42.54, "phase7_lid_cep": 32.65}.get(label, 10.0)
    rec = BenchRecord(
        recorded_at=at,
        git_sha=sha,
        git_dirty=False,
        build=BUILD,
        host=HOST,
        recipe=BenchRecipe(label=label, path=path, lanes=1, lineage="v1"),  # type: ignore[arg-type]
        payload=BenchPayload(repeat=1, config_hash="00ff00ff00ff00ff", runs=[dict(wall_s=wall, audio_s=audio, rtf=wall / audio, maxrss_mb=46.0, files=1)]),  # type: ignore[list-item]
    )
    return rec.model_copy(update=over) if over else rec


def _pair(label: str, exact: float, fast: float, **over: object) -> list[BenchRecord]:
    return [bench_record(label, "exact", exact, **over), bench_record(label, "fast", fast, **over)]  # type: ignore[arg-type]


def _four_legs(**over: object) -> list[BenchRecord]:
    return [
        *_pair("phase7_60s", 0.2638, 0.0573, **over),
        *_pair("phase7_sad_corpus", 0.1679, 0.03667, **over),
        *_pair("phase7_lid_phseq", 0.0445, 0.0126, **over),
        *_pair("phase7_lid_cep", 0.0339, 0.0095, **over),
    ]


# --------------------------------------------------------------------------------------- #
# Schema: hygiene, identity, supersession
# --------------------------------------------------------------------------------------- #


def test_corpus_root_name_matches_the_conftest_constant() -> None:
    assert CORPUS_ROOT.name == schema.CORPUS_ROOT_NAME


def test_record_rejects_a_corpus_file_in_any_string_field() -> None:
    with pytest.raises(ValidationError, match="corpus file"):
        sad_gate(config_name=f"{CORPUS_ROOT.name}/train/audio/ara/ab12cd.wav")
    with pytest.raises(ValidationError, match="corpus file"):
        sad_gate(reason=f"re-run after {CORPUS_ROOT.name}/eval/LID_Features/30/xy98zw.plp8f0mvscmllrdd moved", supersedes="x")


def test_record_rejects_an_absolute_path() -> None:
    with pytest.raises(ValidationError, match="absolute path"):
        sad_gate(config_name="/Users/someone/Speech/configs/training/lre_sad.toml")
    with pytest.raises(ValidationError, match="absolute path"):
        sad_gate(test=r"C:\\runs\\gate.py::test")


def test_record_rejects_naive_time_and_half_supersession() -> None:
    with pytest.raises(ValidationError, match="UTC"):
        sad_gate(recorded_at=datetime(2026, 10, 2, 12, 0, 0))
    with pytest.raises(ValidationError, match="together"):
        sad_gate(supersedes="20261001T000000Z_0000000000000000")
    with pytest.raises(ValidationError, match="together"):
        sad_gate(reason="orphan reason")


def test_extra_fields_are_rejected() -> None:
    data = sad_gate().model_dump(mode="json")
    data["payload"]["bounds"] = []
    with pytest.raises(ValidationError):
        schema.RECORD.validate_python(data)


def test_id_is_content_derived_and_the_filename_must_match(tmp_path: Path) -> None:
    r = sad_gate()
    assert r.id == sad_gate().id
    assert r.id != sad_gate(wall_s=38.0).id
    assert r.id.startswith("20261002T120000Z_") and len(r.id) == len("20261002T120000Z_") + 16

    root = tmp_path / "ledger"
    (root / "baseline").mkdir(parents=True)
    path = schema.write_record(r, root / "baseline" / f"{r.id}.json")
    assert schema.read_record(path) == r
    assert schema.load(root) == [r]

    path.rename(root / "baseline" / "renamed.json")
    with pytest.raises(ValueError, match="does not match the record id"):
        schema.load(root)


def test_a_record_under_the_wrong_kind_directory_fails(tmp_path: Path) -> None:
    r = bench_record()
    (tmp_path / "baseline").mkdir()
    schema.write_record(r, tmp_path / "baseline" / f"{r.id}.json")
    with pytest.raises(ValueError, match="bench record under baseline"):
        schema.load(tmp_path)


def test_bench_record_round_trips(tmp_path: Path) -> None:
    r = bench_record()
    path = schema.write_record(r, tmp_path / "b.json")
    assert schema.read_record(path) == r
    assert json.loads(path.read_text())["kind"] == "bench"


def test_git_state_ignores_the_ledger_outputs_only() -> None:
    # Only meaningful on a clean tree (CI, or a developer who committed); a dirty tree stays dirty either way.
    sha, dirty_before = schema.git_state(REPO)
    assert len(sha) == 40
    probe = schema.LEDGER_DIR / "bench" / "_probe_untracked.json"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("{}")
    try:
        assert schema.git_state(REPO)[1] == dirty_before, "an untracked file under ledger/ must not flip the dirty flag"
    finally:
        probe.unlink()
    outside = REPO / "_probe_untracked.txt"
    outside.write_text("")
    try:
        assert schema.git_state(REPO)[1] is True, "an untracked file outside the ledger outputs is dirt"
    finally:
        outside.unlink()


def test_lineage_of_config_names() -> None:
    assert schema.lineage_of("configs/training/lre_sad.toml") == "v1"
    assert schema.lineage_of("configs/training/lre_sad_v2.toml") == "v2"
    assert schema.lineage_of("configs/training/lre03_lid_features.toml") is None


# --------------------------------------------------------------------------------------- #
# Selection and rendering
# --------------------------------------------------------------------------------------- #


def test_current_baselines_takes_latest_per_recipe_and_drops_superseded() -> None:
    old = sad_gate(at=T0)
    newer = sad_gate(at=T0.replace(hour=13), wall_s=40.0)
    replaced = sad_gate(at=T0.replace(hour=14), wall_s=41.0, supersedes=newer.id, reason="re-measured after the mel fix")
    other = sad_gate("slstm", "forward")
    current = current_baselines([old, newer, replaced, other])
    assert {r.id for r in current} == {replaced.id, other.id}


def test_phase6_lid_table_renders_gate_init_and_tbd_rows() -> None:
    lines = TABLES["phase6_lid_features"]([lid_gate()])
    assert lines[0].startswith("| split | files (train/valid/test) | LID error % | Cavg | chance %")
    assert lines[2] == "| subset gate (2026-10-02) | 35 / 15 / 48 | 72.92 | 0.46 | 91.67 | `lre03_lid_features.toml`, 2 ep x 25 steps | 0 |"
    assert lines[3] == "| subset gate -- untrained init baseline | (same test set) | 93.75 | 0.53 | 91.67 | (seed packs, no training) | 0 |"
    assert lines[4] == "| full run | TBD | TBD | TBD | TBD | TBD | TBD |"
    assert lines[6] == "Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0)."
    # The phseq table does not pick up the features gate.
    assert TABLES["phase6_lid_phseq"]([lid_gate()])[2] == "| subset gate | TBD | TBD | TBD | TBD | TBD | TBD |"


def test_phase6_sad_table_and_a_full_run_row() -> None:
    gate = sad_gate()
    full = sad_gate(at=T0.replace(day=3), subset=None, epochs=40, steps_per_epoch=8, source="launcher", test=None, n_train=1446, wall_s=7200.0)
    lines = TABLES["phase6_sad_v1"]([gate, full])
    assert lines[2].startswith(
        "| subset gate (2026-10-02) | 10 / 8 / 24 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.00 / 1.00 | `lre_sad.toml`, 3 ep x 10 steps, 20 s cap | 0 |"
    )
    assert lines[3].startswith("| subset gate -- untrained init baseline | (same test set) | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 1.00 / 0.00 |")
    assert lines[4].startswith("| full run (2026-10-03) | 1446 / 8 / 24 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.00 / 1.00 |")
    assert lines[4].endswith("| `lre_sad.toml`, 40 ep x 8 steps, 20 s cap | 0 |")


def test_cell_matrix_renders_tbd_for_missing_cells_and_a_collar_range() -> None:
    slstm = sad_gate(
        "slstm",
        "forward",
        lineage="v2",
        dcf=[
            CollarScore(collar=c, dcf=d, pmiss=0.026415, pfa=0.930474)
            for c, d in zip((0.0, 0.25, 0.5, 1.0, 2.0), (0.248021, 0.25, 0.25243, 0.255, 0.255545), strict=True)
        ],
    )
    lines = TABLES["phase10_v2_cells"]([slstm])
    assert lines[2] == "| LSTM / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |"
    assert (
        lines[5] == "| sLSTM / forward | 10 / 8 / 24 | 0.252430 | 0.026415 / 0.930474 | [0.248021, 0.255545] | 0.750000 | 0.750000 (all 5) | +0.497570 | 37 s |"
    )
    assert lines[10] == "| full-corpus runs (any cell x direction) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |"
    # The v1 matrix does not pick up a v2 record; the transformer table only its own cell.
    assert all("TBD" in line for line in TABLES["phase9_v1_cells"]([slstm])[2:8])
    assert len(TABLES["phase11_v2_cells"]([slstm])) == 5  # header, rule, 2 TBD rows, 1 TBD full row


def test_supersession_footnote_hides_the_old_row() -> None:
    old = sad_gate()
    new = sad_gate(at=T0.replace(hour=13), wall_s=39.0, supersedes=old.id, reason="re-run on main after the fixture change")
    lines = TABLES["phase6_sad_v1"]([old, new])
    assert sum(line.startswith("| subset gate (") for line in lines) == 1
    assert lines[-1] == f"Superseded: `{old.id}` by `{new.id}` (re-run on main after the fixture change)."


def test_supersession_footnote_names_each_link_of_a_chain() -> None:
    a = sad_gate()
    b = sad_gate(at=T0.replace(hour=13), wall_s=39.0, supersedes=a.id, reason="reason-b")
    c = sad_gate(at=T0.replace(hour=14), wall_s=40.0, supersedes=b.id, reason="reason-c")
    assert TABLES["phase6_sad_v1"]([a, b, c])[-2:] == [f"Superseded: `{b.id}` by `{c.id}` (reason-c).", f"Superseded: `{a.id}` by `{b.id}` (reason-b)."]


def test_a_launcher_record_with_a_gates_recipe_keeps_the_gate_row() -> None:
    gate = sad_gate()
    launcher = sad_gate(at=T0.replace(hour=13), source="launcher", test=None)
    assert TABLES["phase6_sad_v1"]([gate, launcher])[2].startswith("| subset gate (2026-10-02) |")


def test_render_text_rewrites_between_markers_and_rejects_bad_markers() -> None:
    text = "intro\n<!-- ledger:table phase6_lid_features -->\n| stale |\n<!-- ledger:end -->\noutro\n"
    out = render_text(text, [lid_gate()])
    assert out.startswith("intro\n<!-- ledger:table phase6_lid_features -->\n| split |") and out.endswith("<!-- ledger:end -->\noutro\n")
    assert "| stale |" not in out
    assert table_names(text) == ["phase6_lid_features"]
    with pytest.raises(ValueError, match="no ledger table named"):
        render_text("<!-- ledger:table nope -->\n<!-- ledger:end -->\n", [])
    with pytest.raises(ValueError, match="no end marker"):
        render_text("<!-- ledger:table phase6_sad_v1 -->\n| x |\n", [])


# --------------------------------------------------------------------------------------- #
# The producer side: BaselineResult.to_record without an engine
# --------------------------------------------------------------------------------------- #


def test_to_record_reads_the_recipe_from_run_metadata(tmp_path: Path) -> None:
    meta = {
        "arm": "sad-v2",
        "seed": 3,
        "lanes": 1,
        "subset": 10,
        "epochs": 3,
        "patience": 99,
        "steps_per_epoch": 10,
        "minibatch": 0,
        "valid_size": 8,
        "test_size": 24,
        "init_scheme": "xavier",
        "val_metric": "nn_cost_seg",
        "audio_max_duration": 20.0,
        "cell_type": "cfc",
        "direction": "forward",
        "config_hash": "feedface01234567",
        "config_toml": "configs/training/lre_sad_v2.toml",
        "corpus_root": str(CORPUS_ROOT),  # the manifest may hold it; the record must not
        "git_sha": "c" * 40,
        "git_dirty": False,
    }
    path = B.write_run_metadata(tmp_path, meta)
    rep = DcfReport(scores=[EvalCollar(collar=c, pmiss=0.0, pfa=1.0, dcf=0.25) for c in (0.0, 0.25, 0.5, 1.0, 2.0)])
    res = B.BaselineResult(
        arm="sad-v2",
        out_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoint",
        metadata_path=path,
        epochs_run=3,
        best_epoch=2,
        best_val_cost=0.1,
        first_val_cost=float("nan"),
        val_costs=[float("nan"), 0.2, 0.1],
        train_costs=[0.6, 0.3, 0.1],
        dcf=rep,
        init_dcf=None,
        n_train=10,
        n_valid=8,
        n_test=24,
        wall_s=15.5,
    )
    rec = res.to_record("gate", test="tests/pyo3/test_phase10_gates.py::test_subset_gate_beats_own_init[cfc-forward]")
    assert rec.recipe == BaselineRecipe(
        arm="sad-v2",
        lineage="v2",
        cell="cfc",
        direction="forward",
        subset=10,
        valid_size=8,
        test_size=24,
        audio_max_duration=20.0,
        epochs=3,
        steps_per_epoch=10,
        patience=99,
        minibatch=0,
        init_scheme="xavier",
        seed=3,
        lanes=1,
    )
    assert rec.payload.source == "gate" and rec.payload.first_val_cost is None and rec.payload.best_val_cost == 0.1
    assert rec.payload.first_train_cost == 0.6 and rec.payload.last_train_cost == 0.1 and rec.payload.wall_s == 15.5
    assert rec.payload.collar(0.5) == CollarScore(collar=0.5, dcf=0.25, pmiss=0.0, pfa=1.0) and rec.payload.init_dcf is None
    assert rec.git_sha == "c" * 40 and not rec.git_dirty, "the provenance is the run's start, read from the manifest"
    assert rec.host.cores > 0 and rec.build.profile in ("release", "debug", "unavailable")
    assert str(CORPUS_ROOT) not in schema.canonical_json(rec)


# --------------------------------------------------------------------------------------- #
# The commands
# --------------------------------------------------------------------------------------- #


def _results_with_markers(path: Path) -> Path:
    path.write_text("# Results\n\n<!-- ledger:table phase6_lid_features -->\n| stale |\n<!-- ledger:end -->\n\ntail\n")
    return path


def test_add_validates_copies_and_renders(tmp_path: Path) -> None:
    root, results = tmp_path / "ledger", _results_with_markers(tmp_path / "RESULTS.md")
    rec = lid_gate()
    src = schema.write_record(rec, tmp_path / "record.json")
    assert cli.main(["--root", str(root), "--results", str(results), "add", str(src)]) == 0
    assert (root / "baseline" / f"{rec.id}.json").is_file()
    assert "| subset gate (2026-10-02) | 35 / 15 / 48 | 72.92 |" in results.read_text()
    # Idempotent on the same record, loud on a different one with the same id.
    assert cli.main(["--root", str(root), "--results", str(results), "add", str(src), "--no-render"]) == 0
    (root / "baseline" / f"{rec.id}.json").write_text(schema.write_record(lid_gate(wall_s=1.0), tmp_path / "other.json").read_text())
    with pytest.raises(SystemExit, match="different content"):
        cli.main(["--root", str(root), "--results", str(results), "add", str(src), "--no-render"])


def test_add_refuses_dirty_and_non_release_records(tmp_path: Path) -> None:
    root, results = tmp_path / "ledger", _results_with_markers(tmp_path / "RESULTS.md")
    dirty = schema.write_record(lid_gate(git_dirty=True), tmp_path / "dirty.json")
    with pytest.raises(SystemExit, match="dirty tree"):
        cli.main(["--root", str(root), "--results", str(results), "add", str(dirty)])
    assert cli.main(["--root", str(root), "--results", str(results), "add", str(dirty), "--allow-dirty"]) == 0
    debug = schema.write_record(lid_gate(build=Build(profile="debug", target="x", rustc="rustc 1.0")), tmp_path / "debug.json")
    with pytest.raises(SystemExit, match="not a measurement"):
        cli.main(["--root", str(root), "--results", str(results), "add", str(debug), "--allow-dirty"])


def test_render_check_fails_on_a_stale_table_and_passes_after_render(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, results = tmp_path / "ledger", _results_with_markers(tmp_path / "RESULTS.md")
    (root / "baseline").mkdir(parents=True)
    rec = lid_gate()
    schema.write_record(rec, root / "baseline" / f"{rec.id}.json")
    assert cli.main(["--root", str(root), "--results", str(results), "render", "--check"]) == 1
    assert "-| stale |" in capsys.readouterr().out
    assert cli.main(["--root", str(root), "--results", str(results), "render"]) == 0
    assert cli.main(["--root", str(root), "--results", str(results), "render", "--check"]) == 0
    assert render_file(results, root, check=True) == ""


def test_module_entry_runs() -> None:
    res = subprocess.run([sys.executable, "-m", "speech.ledger", "render", "--help"], cwd=REPO, capture_output=True, text=True)
    assert res.returncode == 0 and "--check" in res.stdout


# --------------------------------------------------------------------------------------- #
# The standing gates over the committed tree (CI)
# --------------------------------------------------------------------------------------- #


def test_every_committed_record_validates_and_is_a_release_build() -> None:
    records = schema.load(LEDGER_DIR)
    assert records, "the ledger must hold the re-run gate records (ADR-0009)"
    for r in records:
        assert r.build.profile == "release", r.id
        assert len(r.git_sha) == 40, r.id
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))
    for r in records:
        if r.supersedes is not None:
            assert r.supersedes in ids, f"{r.id} supersedes an unknown record {r.supersedes}"


def test_results_md_holds_what_the_ledger_renders() -> None:
    names = table_names(RESULTS.read_text(encoding="utf-8"))
    assert set(names) == set(TABLES), f"every renderer has a marker and every marker a renderer: {sorted(set(names) ^ set(TABLES))}"
    assert render_file(RESULTS, LEDGER_DIR, check=True) == "", "RESULTS.md is stale; run: python -m speech.ledger render"


def test_seam_build_info_has_the_three_fields() -> None:
    speech_rs = pytest.importorskip("speech_rs")
    info = speech_rs.build_info()
    assert set(info) == {"profile", "target", "rustc"}
    assert info["profile"] in ("release", "debug") and info["target"] and info["rustc"].startswith("rustc ")
    assert schema.seam_build_info() == Build(**info)


# --------------------------------------------------------------------------------------- #
# Bench: stagers, the wrapper, the matrix renderer, the prose registry
# --------------------------------------------------------------------------------------- #


def test_stage_fixture_60s_writes_an_absolute_path_config(tmp_path: Path) -> None:
    cfg = stage_fixture_60s(tmp_path, tmp_path / "no-corpus")
    text = cfg.read_text()
    assert (tmp_path / "prcts_excerpt.wav").is_file() and (tmp_path / "NNweights_config1.bin").is_file()
    assert (tmp_path / "bench_listing.csv").read_text() == f"{tmp_path / 'prcts_excerpt.wav'}\n"
    assert f"\nDump_Directory {tmp_path / 'vrcts_bench'}\n" in text and (tmp_path / "vrcts_bench").is_dir()
    assert f"\nmultiConfigResultsOutputFile {tmp_path / 'bench_result.mat'}\n" in text
    assert f"\nBLSTM_weightsFile {tmp_path / 'NNweights_config1.bin'}\n" in text
    assert lanes_of(cfg) == 1


def test_sorted_first_is_lexicographic_and_loud_when_absent(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "a.wav").write_bytes(b"")
    (tmp_path / "a" / "z.wav").write_bytes(b"")
    assert sorted_first(tmp_path, "*.wav") == tmp_path / "a" / "z.wav"
    with pytest.raises(FileNotFoundError, match="no \\*.phSeqbis"):
        sorted_first(tmp_path, "*.phSeqbis")
    with pytest.raises(FileNotFoundError, match="absent"):
        sorted_first(tmp_path / "missing", "*.wav")


def test_lanes_of_is_last_wins(tmp_path: Path) -> None:
    cfg = tmp_path / "x.config"
    cfg.write_text("numOuterThreads 8\nfoo 1\nnumOuterThreads 3\n")
    assert lanes_of(cfg) == 3
    cfg.write_text("foo 1\n")
    assert lanes_of(cfg) == 1
    toml = tmp_path / "x.toml"
    toml.write_text("[engine]\nnum_outer_threads = 4\n")
    assert lanes_of(toml) == 4


def test_wrap_reads_the_build_from_the_document_and_strips_the_run_path() -> None:
    doc = {
        "label": "phase7_60s",
        "path": "fast",
        "repeat": 2,
        "config_hash": "0123456789abcdef",
        "build": {"profile": "release", "target": "aarch64-apple-darwin", "rustc": "rustc 1.99.0"},
        "runs": [
            {"path": "fast", "wall_s": 0.05, "audio_s": 120.0, "rtf": 0.0004, "maxrss_mb": 42.0, "files": 1},
            {"path": "fast", "wall_s": 0.06, "audio_s": 120.0, "rtf": 0.0005, "maxrss_mb": 43.0, "files": 1},
        ],
    }
    rec = wrap(doc, lineage="v1", lanes=1)
    assert rec.kind == "bench" and rec.build.profile == "release" and rec.recipe == BenchRecipe(label="phase7_60s", path="fast", lanes=1, lineage="v1")
    assert rec.payload.repeat == 2 and [r.wall_s for r in rec.payload.runs] == [0.05, 0.06]
    assert len(rec.git_sha) == 40 and rec.host.cores > 0


@requires_binary
def test_fixture_leg_runs_through_the_binary(tmp_path: Path) -> None:
    cfg = stage_fixture_60s(tmp_path, tmp_path / "no-corpus")
    doc = bench_json(BINARY, cfg, "fast", "phase7_60s")
    rec = wrap(doc, lineage="v1", lanes=lanes_of(cfg))
    run = rec.payload.runs[0]
    assert rec.recipe.label == "phase7_60s" and rec.recipe.path == "fast" and run.files == 1
    assert abs(run.audio_s - 120.0) < 1.2 and run.wall_s > 0 and run.maxrss_mb > 1
    assert str(tmp_path) not in schema.canonical_json(rec)


@requires_binary
@requires_corpus
@pytest.mark.parametrize("label", ["phase7_sad_corpus", "phase7_lid_phseq", "phase7_lid_cep"])
def test_corpus_legs_stage_and_run(tmp_path: Path, label: str) -> None:
    leg = LEGS[label]
    cfg = leg.stage(tmp_path, CORPUS_ROOT)
    doc = bench_json(BINARY, cfg, "exact", label)
    rec = wrap(doc, lineage=leg.lineage, lanes=lanes_of(cfg))
    assert rec.payload.runs[0].audio_s > 0 and rec.payload.runs[0].files == 1
    assert CORPUS_ROOT.name not in schema.canonical_json(rec)


def test_bench_matrix_pairs_speedup_and_renders_tbd_for_missing_cells() -> None:
    records = [*_pair("phase7_60s", 0.2638, 0.0573), bench_record("phase7_lid_cep", "exact", 0.0339)]
    lines = TABLES["phase7_bench_matrix"](records)  # type: ignore[arg-type]
    assert lines[0].startswith("| leg | audio_s | path | wall_s (mean [range], n) |")
    assert lines[2] == "| SAD 60 s fixture (stereo) | 120.00 | exact | 0.2638 [0.2638-0.2638] (n=1) | 0.002198 | 46.000 | 0.3833 | baseline |"
    assert lines[3] == "| SAD 60 s fixture (stereo) | 120.00 | fast | 0.0573 [0.0573-0.0573] (n=1) | 0.000477 | 46.000 | 0.3833 | 4.60x |"
    assert lines[4] == "| SAD corpus-gated (mono) | TBD | exact | TBD | TBD | TBD | TBD | TBD |"
    assert lines[8].startswith("| LID cep corpus-gated (Twin M7) | 32.65 | exact |")
    assert lines[9] == "| LID cep corpus-gated (Twin M7) | TBD | fast | TBD | TBD | TBD | TBD | TBD |"
    assert lines[-1] == "Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0)."


def test_bench_cells_pool_the_processes_of_the_latest_sha_only() -> None:
    old = [bench_record("phase7_60s", "exact", 0.9, sha="c" * 40, at=T0.replace(day=1))]
    new = [bench_record("phase7_60s", "exact", w, at=T0.replace(minute=i)) for i, w in enumerate((0.26, 0.27, 0.28))]
    [cell] = bench_cells([*old, *new])
    assert cell.git_sha == "b" * 40 and cell.walls == (0.26, 0.27, 0.28) and len(cell.records) == 3
    assert speedups([cell]) == {}


def test_prose_registry_regexes_each_match_once_in_the_live_documents() -> None:
    assert [(q.file, n) for q, _, n in prose.matches(REPO) if n != 1] == []


def test_prose_derived_values_follow_the_stated_rounding() -> None:
    want = prose.derived(_four_legs())  # type: ignore[arg-type]
    assert want == {"sad_1dp": "4.6", "sad_range": "4.58-4.60", "lid_1dp": "3.5", "lid_range": "3.53-3.57"}
    same = prose.derived(
        [*_pair("phase7_60s", 0.3, 0.1), *_pair("phase7_sad_corpus", 0.6, 0.2), *_pair("phase7_lid_phseq", 0.3, 0.1), *_pair("phase7_lid_cep", 0.3, 0.1)]
    )  # type: ignore[arg-type]
    assert same["sad_range"] == "3.00" and same["lid_range"] == "3.00"
    assert prose.derived([bench_record()]) == {}  # type: ignore[list-item]


def test_prose_quotes_match_the_ledger() -> None:
    assert prose.check(schema.load(LEDGER_DIR)) == []
