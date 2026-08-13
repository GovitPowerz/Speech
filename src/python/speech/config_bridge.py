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
    # `d_state`/`d_conv`/`expand` must be >= 1; `dt_rank` may be 0 (= auto). The SAME
    # two-sided check `MambaParams::from_legacy` (blstm.rs:252-255) makes -- mirrored rather
    # than left to the Rust because the Python builders are reachable WITHOUT the engine
    # (`init_weights` straight off a spec), where a 0 would silently emit a degenerate pack
    # (empty conv/state blocks) that no length assert could distinguish from a valid one.
    out: dict[str, int] = {}
    for name, key in keys.items():
        value = int(cfg[key]) if key in cfg else defaults[name]
        minimum = 0 if name == "dt_rank" else 1
        if value < minimum:
            raise ValueError(f"'{key}' must be >= {minimum} (got {value})")
        out[name] = value
    return out


#: The SIZED CfC backbone width (phase-10 spec S1.4), mirroring
#: `blstm.rs::CFC_DEFAULT_BACKBONE_UNITS`. It is a MIRROR, not an independent choice: the
#: engine builds the net from the Rust constant and Python seeds the pack from this one, so
#: a drift between them is a length mismatch at `set_weights` (pinned against the Rust
#: source by `tests/test_phase10_init.py::test_the_rust_and_python_defaults_agree`).
#: 45 is the value that puts the v2-lineage CfC pack within +0.25% of the same lineage's
#: LSTM pack -- the full sizing arithmetic for BOTH lineages is in that test module's
#: docstring.
CFC_DEFAULT_BACKBONE_UNITS = 45
CFC_DEFAULT_BACKBONE_LAYERS = 1


def _cfc_geometry(cfg: dict[str, str]) -> dict[str, int]:
    """The two `Cfc_*` keys (phase-10 spec S1.4/S2), UNPREFIXED by design -- one CfC geometry
    per config, shared by whichever net(s) select `cfc`, exactly as `blstm.rs::CfcParams`
    reads them (the `Mamba_*` precedent verbatim). Absent keys mean the Rust defaults; a
    present-but-unparseable or `< 1` value raises, mirroring `CfcParams::from_legacy`'s hard
    error -- these keys have no legacy source to stay bug-compatible with, the Python
    builders are reachable WITHOUT the engine, and a silently-defaulted geometry would change
    the weight-pack LENGTH with nothing to catch it."""
    defaults = {"backbone_units": CFC_DEFAULT_BACKBONE_UNITS, "backbone_layers": CFC_DEFAULT_BACKBONE_LAYERS}
    keys = {"backbone_units": "Cfc_Backbone_Units", "backbone_layers": "Cfc_Backbone_Layers"}
    out: dict[str, int] = {}
    for name, key in keys.items():
        value = int(cfg[key]) if key in cfg else defaults[name]
        if value < 1:
            raise ValueError(f"'{key}' must be >= 1 (got {value})")
        out[name] = value
    return out


#: The transformer geometry defaults (phase-11 spec S2/S3), mirroring
#: `blstm.rs::TRANSFORMER_DEFAULT_{WINDOW,HEADS,D_FF}`. MIRRORS, not independent choices --
#: the engine builds the net from the Rust constants and Python seeds the pack from these, so
#: a drift is a length mismatch at `set_weights` (pinned against the Rust source by
#: `tests/test_phase11_init.py::test_the_rust_and_python_defaults_agree`).
#:
#: Only `d_ff` is SIZED: `window` and `heads` are parameter-FREE (ALiBi's slopes carry no
#: weights and `heads` only reshapes the same `W_qkv`), so neither moves the pack length.
#: `d_ff = 64` is the smallest integer inside BOTH lineage bands -- the full dual-lineage
#: arithmetic is in that test module's docstring.
TRANSFORMER_DEFAULT_WINDOW = 64
TRANSFORMER_DEFAULT_HEADS = 4
TRANSFORMER_DEFAULT_D_FF = 64


def _transformer_geometry(cfg: dict[str, str]) -> dict[str, int]:
    """The three `Transformer_*` keys (phase-11 spec S2), UNPREFIXED by design -- one
    transformer geometry per config, shared by whichever net(s) select `transformer`, exactly
    as `blstm.rs::TransformerParams` reads them (the `Mamba_*`/`Cfc_*` precedent verbatim).
    Absent keys mean the Rust defaults; a present-but-unparseable or `< 1` value raises,
    mirroring `TransformerParams::from_legacy`'s hard error.

    `H % heads == 0` is NOT checked here, matching the Rust split: this reader never sees the
    layer width. `BlstmConfig::from_legacy` owns that check engine-side (and `init_weights`
    does not need it -- `heads` never enters the pack)."""
    defaults = {"window": TRANSFORMER_DEFAULT_WINDOW, "heads": TRANSFORMER_DEFAULT_HEADS, "d_ff": TRANSFORMER_DEFAULT_D_FF}
    keys = {"window": "Transformer_Window", "heads": "Transformer_Heads", "d_ff": "Transformer_D_Ff"}
    out: dict[str, int] = {}
    for name, key in keys.items():
        value = int(cfg[key]) if key in cfg else defaults[name]
        if value < 1:
            raise ValueError(f"'{key}' must be >= 1 (got {value})")
        out[name] = value
    return out


def nnet_spec(cfg: dict[str, str], prefix: str = "BLSTM") -> dict[str, object]:
    """Extract the NNType-0 network spec from a parsed legacy config.

    Phase 9 (spec S6) added three port-only entries -- `CellType`, `Direction` and `Mamba` --
    read from `{prefix}_Cell_Type` / `{prefix}_Direction` / the unprefixed `Mamba_*`, so the
    seeded-init side (`init_weights.py`) and the pack-length side (`weight_bridge.py`) learn
    the architecture from the SAME config the engine reads, with no call-site churn. Absent
    keys mean the legacy shape (`lstm` / `bidirectional`), so every pre-phase-9 config
    decodes exactly as before.

    Phase 10 (spec S1.4/S2) added a fourth, `Cfc`, on the same terms: the unprefixed
    `Cfc_Backbone_Units` / `Cfc_Backbone_Layers`, defaulting to the SIZED Rust values. It is
    emitted unconditionally like `Mamba` and is inert unless `CellType` is `cfc`.

    Phase 11 (spec S2) added a fifth, `Transformer` (the unprefixed `Transformer_Window` /
    `Transformer_Heads` / `Transformer_D_Ff`), identically: emitted unconditionally, inert
    unless `CellType` is `transformer`. Every consumer reads these entries BY NAME, so an
    additive slot cannot disturb a spec built for any other cell.
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
        "Cfc": _cfc_geometry(cfg),
        "Transformer": _transformer_geometry(cfg),
    }
