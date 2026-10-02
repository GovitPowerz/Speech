"""`python -m speech.ledger bench`: the Phase 7 bench protocol as a command (issue #20).

N fresh `speech bench --json --repeat=1` processes per (leg, path), each wrapped into a
`BenchRecord` (git state and host from here, the build from the document the binary printed) and
promoted through `add`. One record per process is the protocol: `maxrss_mb` is a whole-process
high-water mark, so a clean per-run peak needs a fresh process, and the renderer aggregates the
records that share (label, path, host, SHA, build).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from speech.ledger.schema import REPO, BenchPayload, BenchRecipe, BenchRecord, BenchRun, Build, Lineage, git_state, host_info, now_utc

BINARY = REPO / "src" / "rust" / "target" / "release" / "speech"
# The legacy `numOuterThreads N` line or the TOML `num_outer_threads = N` key.
_LANES = re.compile(r"^\s*(?:numOuterThreads\s+|num_outer_threads\s*=\s*)(\d+)\s*$", re.MULTILINE)


def lanes_of(config: Path) -> int:
    """The config's `numOuterThreads` (`.config`) or `num_outer_threads` (`.toml`), last-wins like
    the legacy reader; 1 when unset."""
    found = _LANES.findall(config.read_text())
    return int(found[-1]) if found else 1


def bench_json(binary: Path, config: Path, path: str, label: str, *, repeat: int = 1) -> dict[str, object]:
    """One `speech bench --json` process; the parsed document."""
    cmd = [str(binary), "bench", f"--repeat={repeat}", f"--path={path}", "--json", f"--label={label}", str(config)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"speech bench failed ({out.returncode}): {out.stderr.strip()}")
    doc: dict[str, object] = json.loads(out.stdout)
    return doc


def wrap(doc: dict[str, object], *, lineage: Lineage | None, lanes: int) -> BenchRecord:
    """The document as a ledger record: the envelope stamped here, the build read from the
    document (the binary knows its own profile; this process knows the tree and the machine)."""
    sha, dirty = git_state()
    build = doc["build"]
    runs = doc["runs"]
    assert isinstance(build, dict) and isinstance(runs, list)
    return BenchRecord(
        recorded_at=now_utc(),
        git_sha=sha,
        git_dirty=dirty,
        build=Build(**build),
        host=host_info(),
        recipe=BenchRecipe(label=str(doc["label"]), path=str(doc["path"]), lanes=lanes, lineage=lineage),  # type: ignore[arg-type]
        payload=BenchPayload(
            repeat=int(doc["repeat"]),  # type: ignore[call-overload]
            config_hash=str(doc["config_hash"]),
            runs=[BenchRun(**{k: v for k, v in run.items() if k != "path"}) for run in runs],
        ),
    )
