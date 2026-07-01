"""Config writer/reader: param-vector <-> config struct <-> `.config` text.

Ported from legacy MATLAB: vec2struct.m, printConfig.m, readConfig.m,
SanitizeConfigStruct.m, cell2str.m. See design spec section 6.
"""

from pathlib import Path


def write_config(params: dict[str, object], path: Path) -> None:
    """Serialize a config struct to legacy `.config` text (Phase 0)."""
    ...


def read_config(path: Path) -> dict[str, str]:
    """Parse a legacy `.config` file into a flat key/value map (Phase 0)."""
    ...
