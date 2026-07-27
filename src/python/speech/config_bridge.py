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


def _mamba_geometry(cfg: dict[str, str]) -> dict[str, int]:
    """The four `Mamba_*` keys (spec S3.3/S6), UNPREFIXED by design -- one mamba geometry per
    config, shared by whichever net(s) select `mamba`, exactly as `blstm.rs::MambaParams`
    reads them. Absent keys mean the Rust defaults (16/4/2/0); a present-but-unparseable
    value raises, mirroring `MambaParams::from_legacy`'s hard error (these keys have no
    legacy source, so there is nothing to stay bug-compatible with, and a silently-defaulted
    geometry would change the weight-pack LENGTH without telling anyone)."""
    defaults = {"d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": 0}
    keys = {"d_state": "Mamba_D_State", "d_conv": "Mamba_D_Conv", "expand": "Mamba_Expand", "dt_rank": "Mamba_Dt_Rank"}
    return {name: int(cfg[key]) if key in cfg else defaults[name] for name, key in keys.items()}


def nnet_spec(cfg: dict[str, str], prefix: str = "BLSTM") -> dict[str, object]:
    """Extract the NNType-0 network spec from a parsed legacy config.

    Phase 9 (spec S6) added three port-only entries -- `CellType`, `Direction` and `Mamba` --
    read from `{prefix}_Cell_Type` / `{prefix}_Direction` / the unprefixed `Mamba_*`, so the
    seeded-init side (`init_weights.py`) and the pack-length side (`weight_bridge.py`) learn
    the architecture from the SAME config the engine reads, with no call-site churn. Absent
    keys mean the legacy shape (`lstm` / `bidirectional`), so every pre-phase-9 config
    decodes exactly as before.
    """
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
        "CellType": cfg.get(f"{p}Cell_Type", "lstm"),
        "Direction": cfg.get(f"{p}Direction", "bidirectional"),
        "Mamba": _mamba_geometry(cfg),
    }
