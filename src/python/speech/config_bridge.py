"""What Python declares about a config: the (cell x direction) vocabulary and the boolean
flag rule, both over an already-parsed KEY -> value map.

Config text and config files are parsed by the engine alone, on the seam:
`speech_rs.load_config(path)` for a file (`.toml` or legacy `.config`, issue #150) and
`speech_rs.parse_legacy_config(text)` for legacy text. The Python `.config` parser that
lived here (the `readConfig.m` / `ConfigFile.cpp` mirror) was retired in issue #155.
"""

from typing import Literal, get_args

#: The (cell x direction) matrix, declared ONCE (issue #21): the names `nn/blstm.rs` parses for
#: `{prefix}_Cell_Type` / `{prefix}_Direction`. The seed builders (`init_weights`), the launcher
#: overlay, the launcher spec (`drivers/spec.py`) and the ledger recipe (`ledger/schema.py`) all
#: read the vocabulary from here; a sixth cell is one entry here plus its builder and its seam
#: gate, nowhere else in Python.
CellType = Literal["lstm", "slstm", "mamba", "cfc", "transformer"]
Direction = Literal["bidirectional", "forward"]
CELL_TYPES: tuple[str, ...] = get_args(CellType)
DIRECTIONS: tuple[str, ...] = get_args(Direction)


def read_flag(cfg: dict[str, str], key: str, default: bool) -> bool:
    """A boolean config key: missing -> `default`; exactly `true`/`false` (trimmed) otherwise,
    anything else raises -- the engine's rule (`legacy_config.rs::get_default::<bool>`, issues
    #32 and #87), so a `True`/`1`/typo cannot read as one value here and fail or flip in the engine."""
    if key not in cfg:
        return default
    value = cfg[key].strip()
    if value not in ("true", "false"):
        raise ValueError(f"'{key}' must be 'true' or 'false' (got {cfg[key]!r})")
    return value == "true"
