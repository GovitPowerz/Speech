"""The fold run: one engine construction over a listing and the single fold it runs.

Port-only (issue #22; no legacy source). The legacy optimizer ran one fold per gradient
evaluation by writing a `.config`, shelling `fsp` in the config's directory and reading the
`.mat` back (`ComputeGradient.m -> CostFunction.m -> ComputeCost.m`, `RunFsp.py`). The port
replaced the shell-out with the in-process `speech_rs.Engine`, and the six-step recipe around
it (render the config, write it, chdir, force the flags, guard the fast path, read the results)
was repeated at every driver site. This module is that recipe, written once:

  * `FoldRun(cfg, workdir, *, backprop, listing=None)` overlays the base config: the backprop
    flags (SAD, plus LID on the Twin), `Neural_Networks_BackPropagation_Epochs 0`
    unconditionally (the F11 rule, IMPROVEMENTS.md: a fold is ONE fold at theta, never the
    engine-internal `train()`), and the listing override. `config` is that map and
    `config_text` its rendering through the one `_config_text`, workdir-independent.
  * The engine is built through `speech_rs.Engine.from_map` on `config` with the six path
    keys resolved against `workdir` (ADR-0010: the process cwd is not part of the Python-side
    contract; the engine still resolves a relative LISTING ROW against its cwd, as the binary
    does, so a production listing carries absolute rows). An absent
    `multiConfigResultsOutputFile` lands in the workdir (where the chdir'd flow put it); an
    absent or empty `Dump_Directory` stays as given (empty is the engine's "no dump", so it is
    never resolved to the workdir). No config file is ever written.
  * `run(weights=None) -> FoldResult` is one fold at `weights` (or at the config's own packs);
    a pack count that does not match the config's nets is refused. The engine is built once,
    on the config's own packs, and `set_weights` injects per run (the modern loop reuses it
    across every epoch's SMORMS3 steps), on both trees: since issue #62 a fast processor
    rebuilds its f32 net from the pack, and both trees refuse a pack of the wrong length in
    the same words (the exact-length contract; only a weight FILE is read head-first).
  * `FoldResult` carries the channel results of every config, the per-net derivatives (empty
    on a forward-only fold), the post-run weights, and over config 0 the costs
    `ComputeGradient.m:110/:114` assemble: `nn_cost_seg`, `nn_cost_lid`, and `nn_cost` (seg,
    plus lid on the Twin). L2 stays with `engine.forward_backward`.

Checkpoints are outside this module: it takes weights, and a checkpoint is a directory of
packs (CONTEXT.md) resolved by `drivers/test.py::resolve_checkpoint_packs`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from speech.engine import ChannelResults, _nn_cost_lid, _nn_cost_seg

if TYPE_CHECKING:
    import speech_rs

F64 = np.float64

# The config keys that name a file or directory the engine opens, resolved against the
# workdir before the map crosses the seam.
_PATH_KEYS = (
    "fileslisting",
    "language2classmapping",
    "BLSTM_weightsFile",
    "BLSTM_LID_weightsFile",
    "Dump_Directory",
    "multiConfigResultsOutputFile",
)


def _config_text(cfg: Mapping[str, str]) -> str:
    """The flat legacy rendering, one `KEY value` line per entry in map order."""
    return "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"


@dataclass
class FoldResult:
    """One fold's readout. `results` covers every config of the run (`for_config(pos)` selects
    one); the cost properties read config 0, the config the weights went into."""

    results: ChannelResults
    derivs: list[NDArray[np.float64]]
    weights: list[NDArray[np.float64]]
    twin: bool

    @property
    def nn_cost_seg(self) -> float:
        return _nn_cost_seg(self.results.for_config(0))

    @property
    def nn_cost_lid(self) -> float:
        return _nn_cost_lid(self.results.for_config(0))

    @property
    def nn_cost(self) -> float:
        # legacy: ComputeGradient.m:110 f = NNCostSeg; :114 f = f + NNCostLID (algo 6 only).
        return self.nn_cost_seg + self.nn_cost_lid if self.twin else self.nn_cost_seg


class FoldRun:
    """One engine over one listing, run at the caller's weights. See the module doc."""

    def __init__(self, cfg: Mapping[str, str], workdir: Path, *, backprop: bool, listing: Path | str | None = None) -> None:
        self.workdir = Path(workdir).absolute()  # a relative workdir would hand the cwd back to the engine
        self.backprop = backprop
        self.twin = int(cfg.get("Algo_choice", "0")) == 6
        config = dict(cfg)
        flag = "true" if backprop else "false"
        config["BLSTM_BackPropagationActivated"] = flag
        if self.twin:
            config["BLSTM_LID_BackPropagationActivated"] = flag
        # legacy: F11 (IMPROVEMENTS.md) -- `Epochs 0` takes `run_solo`, one fold at theta; the
        # gradient is gated on `BackPropagationActivated`, not on Epochs.
        config["Neural_Networks_BackPropagation_Epochs"] = "0"
        if listing is not None:
            config["fileslisting"] = str(listing)
        self.config: dict[str, str] = config
        self._engine: speech_rs.Engine | None = None

    @property
    def config_text(self) -> str:
        return _config_text(self.config)

    def _engine_map(self) -> dict[str, str]:
        """`config` with the path keys resolved against the workdir: the map the seam gets."""
        out = dict(self.config)
        for key in _PATH_KEYS:
            if out.get(key):  # an empty value is a sentinel (`Dump_Directory ""` = no dump), not a path
                out[key] = str(self.workdir / out[key])
        out.setdefault("multiConfigResultsOutputFile", str(self.workdir / "MultiConfigResults.mat"))
        return out

    @staticmethod
    def _build(config: dict[str, str]) -> speech_rs.Engine:
        import speech_rs  # local: the pyo3 module is only needed on the engine path

        return speech_rs.Engine.from_map([config], "-m")

    def _shared_engine(self) -> speech_rs.Engine:
        """The engine on the config's own packs, built once."""
        if self._engine is None:
            self._engine = self._build(self._engine_map())
        return self._engine

    def weights(self) -> list[NDArray[np.float64]]:
        """The per-net packs the engine holds now: the config's own on a fresh fold, the last
        injected pack (one engine step past it after a gradient fold) once `run` has been
        called with weights. Exact tree only: a fast processor exposes no weights (an empty
        list)."""
        return [np.asarray(w, dtype=F64) for w in self._shared_engine().weights(0)]

    def run(self, weights: Sequence[NDArray[np.float64]] | None = None) -> FoldResult:
        """One fold at `weights` (`[sad]` or `[sad, lid]`); `None` runs the engine's current
        pack (the config's own on a fresh fold)."""
        nets = None if weights is None else [np.ascontiguousarray(np.asarray(w, dtype=F64)) for w in weights]
        if nets is not None and len(nets) != (2 if self.twin else 1):
            # The bag indexes the list unchecked: a short one panics, a long one's extra is
            # ignored (#29).
            raise ValueError(f"FoldRun.run: {len(nets)} weight pack(s) for a {'Twin' if self.twin else 'single-net'} config")
        engine = self._shared_engine()
        if nets is not None:
            engine.set_weights(0, nets)
        engine.run()
        results = ChannelResults.from_seam(engine.channel_results())
        derivs = [np.asarray(d, dtype=F64) for d in engine.weights_derivatives(0)] if self.backprop else []
        post = [np.asarray(w, dtype=F64) for w in engine.weights(0)]
        return FoldResult(results=results, derivs=derivs, weights=post, twin=self.twin)
