"""Network/run initialization driver.

Ported from `Init_BLSTM.m`: the experiment-directory + listing-processing + state-save
front end. The legacy builds `PS.Corpora` via `processListing`, creates the
`Executables/<name_dir>` tree, copies the `fsp` binary + `RunFsp.py`, and saves
`ParamStruct.mat`. Here the engine is in-process (no binary copy, no `RunFsp.py` seam),
`processListing` is `batching.read_listing`, and `ParamStruct.mat` becomes a typed
`RunState` JSON (the documented deviation, see `drivers/state.py`).
"""

from __future__ import annotations

from pathlib import Path

from speech.drivers.state import RunState


def init_run(config: Path, out_dir: Path) -> RunState:
    """Parse the engine `.config`, read its corpus listing, build the typed `RunState`,
    create `out_dir`, and persist the state as `out_dir/run_state.json`. The config's
    OWN directory stays the corpus root (the listing/weight paths are resolved relative
    to it at train time); `out_dir` only holds the orchestrator's checkpoints."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = RunState.from_config(Path(config), out)
    state.save(out / "run_state.json")
    return state
