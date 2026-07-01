"""Extract the two fixed random tables from the legacy Constants.h.

`_RandomGaussVector[100000]` (N(0,1), weight init) and `_RandomVector[100000]`
(U[0,1), dither/augmentation) are the source of determinism in the legacy engine.
This writes them as one little-endian f64 blob (gauss then uniform) at
data/random_tables.bin so the Rust `constants` module can `include_bytes!` it.

Usage: uv run python scripts/extract_random_tables.py
Requires the git-ignored legacy/ copy present locally.
"""

import re
import struct
from pathlib import Path

CONSTANTS = Path("legacy/src/Constants.h")
OUT = Path("data/random_tables.bin")
N = 100000


def extract_array(text: str, name: str) -> list[float]:
    start = text.index(f"{name}[{N}]")
    open_brace = text.index("{", start)
    close_brace = text.index("}", open_brace)
    body = text[open_brace + 1 : close_brace]
    values = [float(tok) for tok in re.split(r"[,\s]+", body.strip()) if tok]
    if len(values) != N:
        raise ValueError(f"{name}: expected {N} values, got {len(values)}")
    return values


def main() -> None:
    text = CONSTANTS.read_text()
    gauss = extract_array(text, "_RandomGaussVector")
    uniform = extract_array(text, "_RandomVector")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as fh:
        fh.write(struct.pack(f"<{2 * N}d", *gauss, *uniform))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
