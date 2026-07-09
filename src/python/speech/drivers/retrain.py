"""Resume/retrain driver.

Ported from `ReTrain_BLSTM.m`: reload a prior run's `nnet_best` and seed the QuantumPSO
population with it (`PSOseedparam = [nnet_best ...]`, ReTrain_BLSTM.m's reuse branch), then
run the same outer+inner loop. Here `nnet_best` is the checkpoint's saved gbest genome
(`gbest.bin`), fed to `train` as the QPSO `seed_value`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from speech.drivers.state import RunState, TrainResult
from speech.drivers.train import train
from speech.weight_bridge import read_weight_vector


def seed_from_checkpoint(checkpoint: Path) -> NDArray[np.float64]:
    """Load a saved gbest genome (`checkpoint/gbest.bin`) as the QPSO seed vector -- the
    `nnet_best` seeding of `ReTrain_BLSTM.m`."""
    return read_weight_vector(Path(checkpoint) / "gbest.bin")


def retrain(
    state: RunState,
    checkpoint: Path,
    seed: int,
    *,
    qpso_particles: int = 24,
    qpso_epochs: int = 100,
    inner_steps: int = 20,
) -> TrainResult:
    """Resume training, seeding the QPSO population's first rows with the checkpoint's
    gbest genome (`ReTrain_BLSTM.m`'s `nnet_best` reuse).

    Deviation: legacy `ReTrain_BLSTM.m:803` seeds `nnet_best` PLUS half the prior
    population (`nnet_in(:,1:ceil(N/2))`); this port seeds `nnet_best` ONLY, since the
    JSON checkpoint stores the gbest genome, not the full population (see IMPROVEMENTS.md).
    """
    nnet_best = seed_from_checkpoint(checkpoint)
    return train(
        state,
        seed,
        qpso_particles=qpso_particles,
        qpso_epochs=qpso_epochs,
        inner_steps=inner_steps,
        seed_value=nnet_best.reshape(1, -1),
    )
