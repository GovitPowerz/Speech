"""Shared helpers for the init suites (issue #27): every block is located through the engine's
weight layout (`speech_rs.weight_layout`), never a Python re-derivation, and the RNG-order
reconstruction pins draw with the Xavier / He helpers written out here rather than imported from
`init_weights` (an independent transcription, not a tautology)."""

import math
from typing import Any, cast

import numpy as np
import speech_rs
from numpy.typing import NDArray

Block = dict[str, Any]


def layout_of(cfg: dict[str, str], prefix: str = "BLSTM") -> list[Block]:
    return cast(list[Block], speech_rs.weight_layout(cfg, prefix))


def block_of(blocks: list[Block], stack: str, layer: int, name: str) -> Block:
    return next(b for b in blocks if (b["stack"], b["layer"], b["name"]) == (stack, layer, name))


def block_slice(pack: NDArray[np.float64], b: Block) -> NDArray[np.float64]:
    return pack[b["offset"] : b["offset"] + b["len"]]


def pack_length(cfg: dict[str, str], prefix: str = "BLSTM") -> int:
    return int(sum(b["len"] for b in layout_of(cfg, prefix)))


def layer0(cfg: dict[str, str], pack: NDArray[np.float64]) -> NDArray[np.float64]:
    """The forward stack's layer-0 run -- the FIRST thing drawn, so a fresh `default_rng(seed)`
    reconstruction lines up with it from the first draw."""
    blocks = [b for b in layout_of(cfg) if (b["stack"], b["layer"]) == ("forward", 0)]
    return pack[blocks[0]["offset"] : blocks[-1]["offset"] + blocks[-1]["len"]]


def xavier(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(rng.uniform(-limit, limit, size=shape), dtype=np.float64).reshape(-1)


def he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    return np.asarray(rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape), dtype=np.float64).reshape(-1)


def draw(rng: np.random.Generator, scheme: str, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    return xavier(rng, shape, fan_in, fan_out) if scheme == "xavier" else he(rng, shape, fan_in)
