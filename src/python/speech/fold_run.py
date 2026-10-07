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
    absent `Dump_Directory` stays absent. No config file is ever written.
  * `run(weights=None) -> FoldResult` is one fold at `weights` (or at the config's own packs).
    On the exact tree the engine is built once and `set_weights` injects per run (the modern
    epoch reuses it across SMORMS3 steps); on `Inference_Path fast` the processor loads weights
    only at construction and `set_weights` bails (T6b), so the arrays are written as `.bin`
    packs under the workdir, the weight keys repointed, a fresh engine built on them, and the
    packs removed again (the engine holds the weights; nothing is left beside the run's own).
  * `FoldResult` carries the channel results of every config, the per-net derivatives (empty
    on a forward-only fold), the post-run weights, and over config 0 the costs
    `ComputeGradient.m:110/:114` assemble: `nn_cost_seg`, `nn_cost_lid`, and `nn_cost` (seg,
    plus lid on the Twin). L2 stays with `engine.forward_backward`.

Checkpoints are outside this module: it takes weights, and a checkpoint is a directory of
packs (CONTEXT.md) resolved by `drivers/test.py::resolve_checkpoint_packs`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from speech.engine import ChannelResults, _nn_cost_lid, _nn_cost_seg
from speech.weight_bridge import write_bin

if TYPE_CHECKING:
    import speech_rs

F64 = np.float64

# The config keys that name a file or directory the engine opens, resolved against the
# workdir before the map crosses the seam. The two weight keys are also the fast guard's
# repoint targets, in net order (`[sad]`, or `[sad, lid]` on the Twin).
_WEIGHT_KEYS = ("BLSTM_weightsFile", "BLSTM_LID_weightsFile")
_PATH_KEYS = ("fileslisting", "language2classmapping", *_WEIGHT_KEYS, "Dump_Directory", "multiConfigResultsOutputFile")


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
        self.fast = cfg.get("Inference_Path", "exact") == "fast"
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
            if key in out:
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

    def _fast_engine(self, nets: list[NDArray[np.float64]]) -> speech_rs.Engine:
        """The fast guard: write each pack under the workdir, repoint the weight keys, build,
        then remove the packs (a fast processor reads them at construction, T6b)."""
        config = self._engine_map()
        packs: list[Path] = []
        for key, name, w in zip(_WEIGHT_KEYS, ("sad", "lid"), nets, strict=False):
            fd, tmp = tempfile.mkstemp(dir=self.workdir, prefix=f"_fold_{name}_", suffix=".bin")
            os.close(fd)
            path = Path(tmp)
            write_bin(w.shape[0], 1, w, path)
            config[key] = str(path)
            packs.append(path)
        try:
            return self._build(config)
        finally:
            for path in packs:
                path.unlink()

    def weights(self) -> list[NDArray[np.float64]]:
        """The per-net packs the engine holds now: the config's own on a fresh fold, the last
        injected pack (one engine step past it after a gradient fold) once `run` has been
        called with weights."""
        return [np.asarray(w, dtype=F64) for w in self._shared_engine().weights(0)]

    def run(self, weights: Sequence[NDArray[np.float64]] | None = None) -> FoldResult:
        """One fold at `weights` (`[sad]` or `[sad, lid]`); `None` runs the engine's current
        pack (the config's own on a fresh fold)."""
        if weights is None:
            engine = self._shared_engine()
        else:
            nets = [np.ascontiguousarray(np.asarray(w, dtype=F64)) for w in weights]
            if self.fast:
                engine = self._fast_engine(nets)
            else:
                engine = self._shared_engine()
                engine.set_weights(0, nets)
        engine.run()
        results = ChannelResults.from_seam(engine.channel_results())
        derivs = [np.asarray(d, dtype=F64) for d in engine.weights_derivatives(0)] if self.backprop else []
        post = [np.asarray(w, dtype=F64) for w in engine.weights(0)]
        return FoldResult(results=results, derivs=derivs, weights=post, twin=self.twin)
