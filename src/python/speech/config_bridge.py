"""Config writer/reader: legacy .config <-> typed spec.

Ported from legacy MATLAB: readConfig.m / printConfig.m and C++ ConfigFile.cpp.
The .config is plain-text KEY value, last-wins, comma-vectors, # comments,
_-in-key rest-of-line continuation. See design spec section 4.
"""

from typing import Literal, cast, get_args

#: The (cell x direction) matrix, declared ONCE (issue #21): the names `nn/blstm.rs` parses for
#: `{prefix}_Cell_Type` / `{prefix}_Direction`. The seed builders (`init_weights`), the launcher
#: overlay, the launcher spec (`drivers/spec.py`) and the ledger recipe (`ledger/schema.py`) all
#: read the vocabulary from here; a sixth cell is one entry here plus its builder and its seam
#: gate, nowhere else in Python.
CellType = Literal["lstm", "slstm", "mamba", "cfc", "transformer"]
Direction = Literal["bidirectional", "forward"]
CELL_TYPES: tuple[str, ...] = get_args(CellType)
DIRECTIONS: tuple[str, ...] = get_args(Direction)


def parse_legacy_config(text: str) -> dict[str, str]:
    """Parse legacy KEY value config text; last duplicate key wins, # lines skipped."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        name = parts[0]
        if name.startswith("#"):
            continue
        out[name] = parts[1] if len(parts) > 1 else ""
    return out


def read_flag(cfg: dict[str, str], key: str, default: bool) -> bool:
    """A boolean config key: missing -> `default`; exactly `true`/`false` (trimmed) otherwise,
    anything else raises -- the engine's rule (`bag_of_processors.rs::get_bool_default`, issue
    #32), so a `True`/`1`/typo cannot read as one value here and fail or flip in the engine."""
    if key not in cfg:
        return default
    value = cfg[key].strip()
    if value not in ("true", "false"):
        raise ValueError(f"'{key}' must be 'true' or 'false' (got {cfg[key]!r})")
    return value == "true"


def nnet_spec(cfg: dict[str, str], prefix: str = "BLSTM") -> dict[str, object]:
    """The resolved network spec of the `prefix` net, read by the ENGINE (issue #27):
    `speech_rs.blstm_config` parses the same keys `BlstmConfig::from_legacy` parses and returns
    them under the names this function has always used (`LSTMNeuronNb`, `LSTMSubSampling`,
    `OutputNeuronNb`, `OutputSubSampling`, `NNetInputSize`, `peepholes`, `CostLawSpeech`,
    `CostLawNoSpeech`, `CellType`, `Direction`, `Mamba`, `Cfc`, `Transformer`), the per-cell
    geometry carrying its RUST defaults when the keys are absent -- so no numeric default is
    mirrored here (ADR-0006). A present-but-invalid key raises `ValueError` with the engine's
    message, never a default: a silently defaulted geometry would change a pack's length with
    nothing to catch it."""
    import speech_rs  # local: `import speech` stays free of the extension for the modules that never touch it

    return cast(dict[str, object], speech_rs.blstm_config(cfg, prefix))
