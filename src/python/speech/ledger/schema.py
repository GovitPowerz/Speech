"""The ledger's record schema and the provenance it stamps on every record (issue #20).

One record is one measurement: an envelope (schema version, kind, when, which commit, which
build, which host) around a `recipe` (the identity of the measurement: what was run) and a
`payload` (what it measured). Two kinds exist: `baseline` (a `run_baseline` run, from the
launcher or a subset gate) and `bench` (one `speech bench --json` invocation). The recipe is the
identity: two records with one recipe are the same measurement taken twice, which is what
`supersedes` is for; the config hash, SHA, build and host say how it was run, never which.

Schema evolution (issue #38): `load` reads exactly one `SCHEMA_VERSION`. A bump migrates the
committed records through a script in the same commit (a JSON-level transform, validated by the
new model, written under the new content-derived id with `recorded_at` kept and `supersedes`
targets remapped); ids change, numbers never, and the script does not outlive its commit.

License hygiene is a validator, not a review item: a record whose string fields name a file
under the corpus root, or carry any absolute path, is rejected at construction. The record is
the promotable subset of a run; `run_metadata.json` (which may hold `corpus_root`) never is.

This module imports nothing from the seam at import time, so the plain Python CI job can
validate every committed record and render RESULTS.md without the corpus or the PyO3 build.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

SCHEMA_VERSION = 2
REPO = Path(__file__).resolve().parents[4]
LEDGER_DIR = REPO / "ledger"
# The corpus root's directory name (`tests/conftest.CORPUS_ROOT.name`, asserted equal there) and
# the corpus-file pattern of `tests/test_license_hygiene.py`: a path through the root whose last
# segment is a concrete filename.
CORPUS_ROOT_NAME = "LRE03-LRE07"
CORPUS_FILE = re.compile(re.escape(CORPUS_ROOT_NAME) + r"/(?:[\w.-]+/)*[\w-]*\w\.\w+")
ABSOLUTE_PATH = re.compile(r"^(/|[A-Za-z]:\\)")

Lineage = Literal["v1", "v2"]
Arm = Literal["sad", "sad-v2", "lid-features", "lid-phseq"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Build(_Strict):
    """`speech_rs.build_info()` / the bench document's `build`: the binary that measured."""

    profile: str
    target: str
    rustc: str


class Host(_Strict):
    """The hardware class, never a hostname: what RESULTS.md's "this box" has always meant."""

    chip: str
    arch: str
    cores: int
    os: str


class BaselineRecipe(_Strict):
    """What a `run_baseline` measurement is: the arm, lineage, cell and direction, the split
    spec, the training budget, the seed, the lane count and the listing the split was drawn
    from (`listing`: the blake2b-8 of a localized 2015 listing's bytes, never its path; `None`
    when the records were derived from the corpus tree). `subset=None` is the full run."""

    arm: Arm
    lineage: Lineage | None
    cell: str
    direction: str
    subset: int | None
    valid_size: int
    test_size: int
    audio_max_duration: float | None
    epochs: int
    steps_per_epoch: int
    patience: int
    minibatch: int
    init_scheme: str
    seed: int
    lanes: int
    listing: str | None

    def key(self) -> tuple[object, ...]:
        return tuple(self.model_dump().values())

    @property
    def full(self) -> bool:
        return self.subset is None


class CollarScore(_Strict):
    collar: float
    dcf: float
    pmiss: float
    pfa: float


class BaselinePayload(_Strict):
    """What a `run_baseline` run measured. `source` says who ran it (the launcher or a subset
    gate, with the gate's pytest node id); the pin stays in the test (ADR-0003). `resumed` says
    the call continued from `out_dir/checkpoint`: its `wall_s` covers one segment, its batch
    cursors restarted and its provenance names only the last tree, so `add` refuses it."""

    source: Literal["launcher", "gate"]
    test: str | None
    resumed: bool
    config_name: str
    config_hash: str
    n_train: int
    n_valid: int
    n_test: int
    val_metric: str
    epochs_run: int
    best_epoch: int
    wall_s: float
    first_val_cost: float | None
    best_val_cost: float | None
    first_train_cost: float | None
    last_train_cost: float | None
    dcf: list[CollarScore] | None
    init_dcf: list[CollarScore] | None
    lid_error: float | None
    cavg: float | None
    init_lid_error: float | None
    init_cavg: float | None

    def collar(self, c: float, *, init: bool = False) -> CollarScore | None:
        scores = self.init_dcf if init else self.dcf
        if scores is None:
            return None
        for s in scores:
            if s.collar == c:
                return s
        raise KeyError(f"no score for collar {c}")


class BenchRecipe(_Strict):
    """What a bench measurement is: the label (never a config path), the compute path, the lane
    count and, for a SAD config, its lineage."""

    label: str
    path: Literal["exact", "fast"]
    lanes: int
    lineage: Lineage | None


class BenchRun(_Strict):
    wall_s: float
    audio_s: float
    rtf: float
    maxrss_mb: float
    files: int


class BenchPayload(_Strict):
    """One `speech bench --json` document: `repeat` runs of one process. The Phase 7 protocol
    (three independent `--repeat=1` processes) is three records; the renderer aggregates."""

    repeat: int
    config_hash: str
    runs: list[BenchRun]


class RecordBase(_Strict):
    schema_version: Literal[2] = 2
    recorded_at: datetime
    git_sha: str
    git_dirty: bool
    build: Build
    host: Host
    supersedes: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _hygiene(self) -> RecordBase:
        hits = [s for s in _strings(self.model_dump(mode="json")) if CORPUS_FILE.search(s) or ABSOLUTE_PATH.match(s)]
        if hits:
            raise ValueError(f"a record may not name a corpus file or carry an absolute path: {hits}")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("recorded_at must be UTC")
        if (self.supersedes is None) != (self.reason is None):
            raise ValueError("supersedes and reason come together")
        return self

    @property
    def id(self) -> str:
        """`<recorded_at>_<blake2b-8 of the canonical JSON>`: derived from the content, so the
        filename and the record cannot disagree without `load` noticing."""
        digest = hashlib.blake2b(canonical_json(self).encode(), digest_size=8).hexdigest()
        return f"{self.recorded_at:%Y%m%dT%H%M%SZ}_{digest}"


class BaselineRecord(RecordBase):
    kind: Literal["baseline"] = "baseline"
    recipe: BaselineRecipe
    payload: BaselinePayload


class BenchRecord(RecordBase):
    kind: Literal["bench"] = "bench"
    recipe: BenchRecipe
    payload: BenchPayload


Record = Annotated[BaselineRecord | BenchRecord, Field(discriminator="kind")]
RECORD = TypeAdapter[BaselineRecord | BenchRecord](Record)


def _strings(obj: object) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def canonical_json(record: RecordBase) -> str:
    return json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def write_record(record: RecordBase, path: Path) -> Path:
    path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return path


def read_record(path: Path) -> BaselineRecord | BenchRecord:
    return RECORD.validate_json(path.read_text())


def load(root: Path = LEDGER_DIR) -> list[BaselineRecord | BenchRecord]:
    """Every record under `root`, sorted by `recorded_at` then id. A file whose name is not its
    record's id is a corrupted or hand-edited record and fails loudly."""
    records = []
    for path in sorted(root.glob("*/*.json")):
        rec = read_record(path)
        if path.stem != rec.id:
            raise ValueError(f"{path}: filename does not match the record id {rec.id}")
        if path.parent.name != rec.kind:
            raise ValueError(f"{path}: a {rec.kind} record under {path.parent.name}/")
        records.append(rec)
    return sorted(records, key=lambda r: (r.recorded_at, r.id))


def superseded_ids(records: list[BaselineRecord] | list[BenchRecord] | list[BaselineRecord | BenchRecord]) -> set[str]:
    return {r.supersedes for r in records if r.supersedes is not None}


# --------------------------------------------------------------------------------------- #
# Provenance collectors: what the envelope says about the code, the build and the machine.
# --------------------------------------------------------------------------------------- #


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


# The ledger's own outputs: a record under `ledger/` or a rendered RESULTS.md changes no
# measurement, so promoting one leg must not make the next leg's record "dirty".
LEDGER_OUTPUTS = ("ledger", "RESULTS.md")


def git_state(repo: Path = REPO) -> tuple[str, bool]:
    """`(HEAD sha, dirty)`, where dirty means any tracked change or untracked file outside the
    ledger's own outputs (`LEDGER_OUTPUTS`). A failing git call raises `RuntimeError` carrying
    git's own reason (the captured stderr would otherwise be lost behind the exit status)."""
    pathspec = [".", *(f":(exclude){p}" for p in LEDGER_OUTPUTS)]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", *pathspec], cwd=repo, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git failed in {repo}: {e.stderr.strip() or e}") from e
    return sha, bool(status.strip())


def host_info() -> Host:
    return Host(chip=_chip(), arch=platform.machine(), cores=os.cpu_count() or 0, os=f"{platform.system()} {platform.release()}")


def _chip() -> str:
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def seam_build_info() -> Build:
    """The PyO3 module's build provenance. Imported here, not at module level: the plain
    Python CI job renders without the seam. Without the module (an engine-free stubbed run) the
    profile reads `unavailable`, which `add` refuses like any non-release build."""
    try:
        import speech_rs

        info = speech_rs.build_info()
    except ImportError, AttributeError:  # absent, or an engine-free test's stub module
        return Build(profile="unavailable", target="", rustc="")
    return Build(**info)


def lineage_of(config_name: str) -> Lineage | None:
    """The SAD lineage a config name belongs to (ADR-0008); a LID config has none."""
    stem = Path(config_name).stem
    if stem == "lre_sad":
        return "v1"
    if stem == "lre_sad_v2":
        return "v2"
    return None
