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
from speech.ledger import cli, schema
from speech.ledger.render import RESULTS, render_file, render_text, table_names
from speech.ledger.schema import LEDGER_DIR, BaselinePayload, BaselineRecipe, BaselineRecord, BenchPayload, BenchRecipe, BenchRecord, Build, CollarScore, Host
from speech.ledger.tables import TABLES, current_baselines

from tests.conftest import CORPUS_ROOT

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


def bench_record() -> BenchRecord:
    return BenchRecord(
        recorded_at=T0,
        git_sha="b" * 40,
        git_dirty=False,
        build=BUILD,
        host=HOST,
        recipe=BenchRecipe(label="phase7_60s", path="fast", lanes=1, lineage="v1"),
        payload=BenchPayload(repeat=1, config_hash="00ff00ff00ff00ff", runs=[dict(wall_s=0.06, audio_s=120.0, rtf=0.0005, maxrss_mb=46.0, files=1)]),  # type: ignore[list-item]
    )


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
    assert len(rec.git_sha) == 40 and rec.host.cores > 0 and rec.build.profile in ("release", "debug", "unavailable")
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
