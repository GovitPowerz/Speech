"""Extract Phase-0b-i cost-law golden fixtures from the legacy .mat sweeps (scipy).

test_costlaws.mat holds cost_speech / cost_other (each 1000x2: output=col0, cost=col1);
test_costlaws_deriv.mat holds the derivative sweeps. This emits the five sweep columns as
1x1000 .bin fixtures plus costlaw_config.json (the legacy config that generated them), so the
Rust cost test reproduces them bit-for-bit without scipy.

Usage: uv run --with scipy python scripts/extract_phase0b_costlaw_fixtures.py
Requires the local git-ignored .mat files.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from speech import weight_bridge

BIN = Path("/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Release/bin")
OUT = Path("tests/reference_data/phase0b")
# Sweep-generating config, deduced from the anchor output=0.001,target=1 -> 0.1997999997999998
# = LinearLaw(P=cp=0.2, q=0, t=1-1e-6): 0.2 + (-0.2*(1-0)/(1-1e-6))*0.001. Speech law = linear.
# The Rust cost test is the arbiter: if a value diverges, adjust these keys against the sweep.
CONFIG = {
    "prefix": "BLSTM",
    "keys": {
        "BLSTM_CostLawSpeech": "linear",
        "BLSTM_CostLawNoSpeech": "linear",
        "BLSTM_CostPonderation": "0.2",
        "BLSTM_CostLawParamSpeech": "0",
        "BLSTM_CostLawParamNoSpeech": "0",
        "BLSTM_CostLawThreshSpeech": "1",
        "BLSTM_CostLawThreshNoSpeech": "0",
    },
}


def col(mat: dict, var: str, c: int) -> np.ndarray:
    a = np.asarray(mat[var], dtype=np.float64)
    return np.ascontiguousarray(a[:, c])


def main() -> None:
    m = sio.loadmat(BIN / "test_costlaws.mat")
    md = sio.loadmat(BIN / "test_costlaws_deriv.mat")
    OUT.mkdir(parents=True, exist_ok=True)
    output = col(m, "cost_speech", 0)  # output column (same for all sweeps)
    n = output.shape[0]
    weight_bridge.write_bin(1, n, output, OUT / "sweep_output.bin")
    weight_bridge.write_bin(1, n, col(m, "cost_speech", 1), OUT / "sweep_cost_speech.bin")
    weight_bridge.write_bin(1, n, col(m, "cost_other", 1), OUT / "sweep_cost_nospeech.bin")
    weight_bridge.write_bin(1, n, col(md, "cost_deriv_speech", 1), OUT / "sweep_deriv_speech.bin")
    weight_bridge.write_bin(1, n, col(md, "cost_deriv_other", 1), OUT / "sweep_deriv_nospeech.bin")
    (OUT / "costlaw_config.json").write_text(json.dumps(CONFIG, indent=1))
    # Anchor is row 1 (output=0.001), not row 0 (output=0.0), per the deduced-config comment above.
    print(f"OK: wrote {n}-row sweeps + config to {OUT}; anchor cost_speech[output~0.001]={col(m, 'cost_speech', 1)[1]!r}")


if __name__ == "__main__":
    main()
