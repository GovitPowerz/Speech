"""Config writer/reader: legacy .config <-> typed spec.

Ported from legacy MATLAB: readConfig.m / printConfig.m and C++ ConfigFile.cpp.
The .config is plain-text KEY value, last-wins, comma-vectors, # comments,
_-in-key rest-of-line continuation. See design spec section 4.
"""


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


def _ints(cfg: dict[str, str], key: str) -> list[int]:
    return [int(x) for x in cfg[key].split(",")]


def nnet_spec(cfg: dict[str, str], prefix: str = "BLSTM") -> dict[str, object]:
    """Extract the NNType-0 network spec from a parsed legacy config."""
    p = f"{prefix}_"
    peephole_keys = [
        f"{p}Forward_IsCellsPeepholesActive",
        f"{p}Backward_IsCellsPeepholesActive",
        f"{p}Forward_IsGatesPeepholesActive",
        f"{p}Backward_IsGatesPeepholesActive",
        f"{p}Forward_IsGatesRecurrentPeepholesActive",
        f"{p}Backward_IsGatesRecurrentPeepholesActive",
    ]
    return {
        "LSTMNeuronNb": _ints(cfg, f"{p}LSTMNeuronNb"),
        "LSTMSubSampling": _ints(cfg, f"{p}LSTMSubSampling"),
        "OutputNeuronNb": _ints(cfg, f"{p}OutputNeuronNb"),
        "OutputSubSampling": _ints(cfg, f"{p}OutputSubSampling"),
        "NNetInputSize": int(cfg[f"{p}NNetInputSize"]),
        "peepholes": [cfg[k] == "true" for k in peephole_keys],
        "CostLawSpeech": cfg[f"{p}CostLawSpeech"],
        "CostLawNoSpeech": cfg[f"{p}CostLawNoSpeech"],
    }
