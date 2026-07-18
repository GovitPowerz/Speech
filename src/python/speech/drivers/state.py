"""Typed run state -- the pydantic replacement for the legacy `PS` god-struct.

Ported from the `PS.*` field usage across `Init_BLSTM.m` / `Train_BLSTM.m` /
`Test_BLSTM.m`. The legacy `PS` is a free-form MATLAB struct persisted as
`ParamStruct.mat`; here it is a typed `RunState` persisted as JSON -- a documented
deviation from the `.mat` seam (the engine consumes the flat `.config` + `.bin`
weights, never `ParamStruct.mat`, so the outer orchestrator's own state format is
free to be a plain, diffable, deterministic JSON blob). See IMPROVEMENTS.md.

`RunConfig`/`LidNetSpec` are RE-EXPORTED from `speech.genome` (not moved): `genome.py`
and `scoring.py` already import them there, and moving would churn both for zero
benefit (the task brief's stated preference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from speech.batching import read_listing
from speech.config_bridge import parse_legacy_config
from speech.genome import LidNetSpec, RunConfig

__all__ = [
    "EpochRecord",
    "LidNetSpec",
    "ModernTrainParams",
    "ModernTrainResult",
    "RunConfig",
    "RunState",
    "TrainResult",
    "ps_from_config",
]


def _ints(cfg: dict[str, str], key: str) -> list[int]:
    return [int(x) for x in cfg[key].split(",")]


def _flag(cfg: dict[str, str], key: str, default: int = 0) -> int:
    return 1 if cfg.get(key, "false") == "true" else default


def _lid_from_config(cfg: dict[str, str]) -> LidNetSpec:
    """PS.NS.LID.* from the `BLSTM_LID_*` engine keys (algo 6 only). Optimizer-only
    fields (NNType, backprop flag) default to the Train_BLSTM.m algo-6 values."""
    return LidNetSpec(
        NNType=0,
        BackPropagationActivated=1,
        LSTM_net_size=_ints(cfg, "BLSTM_LID_LSTMNeuronNb"),
        LSTMSubSampling=_ints(cfg, "BLSTM_LID_LSTMSubSampling"),
        Output_net_size=_ints(cfg, "BLSTM_LID_OutputNeuronNb"),
        OutputSubSampling=_ints(cfg, "BLSTM_LID_OutputSubSampling"),
        Mode=int(cfg["BLSTM_LID_Mode"]),
        PostProcessMode=int(cfg.get("BLSTM_LID_PostProcessMode", "0")),
        TargetEnforcementStep=int(cfg.get("BLSTM_LID_TargetEnforcementStep", "0")),
        BackPropWER=float(cfg.get("BLSTM_LID_BackPropWER", "-1.0")),
        classes_ponderations=[],
        InputNormalizationType=int(cfg.get("BLSTM_LID_InputNormalizationType", "0")),
        IsCellsPeepholesActive=_flag(cfg, "BLSTM_LID_Forward_IsCellsPeepholesActive", 1),
        IsGatesPeepholesActive=_flag(cfg, "BLSTM_LID_Forward_IsGatesPeepholesActive", 1),
        IsGatesRecurrentPeepholesActive=_flag(cfg, "BLSTM_LID_Forward_IsGatesRecurrentPeepholesActive", 1),
        LSTM_MaxSat=float(cfg.get("BLSTM_LID_Forward_MaxSaturation", "-10.0")),
        nnmatfile=cfg.get("BLSTM_LID_weightsFile", "LIDNNweights.mat"),
    )


def ps_from_config(
    cfg: dict[str, str],
    *,
    adim: float = 10.0,
    coeff_nn: float = 5.0,
    balance: int | None = None,
) -> RunConfig:
    """Build the vec2struct `RunConfig` (the fields `vec2struct` reads) from a parsed
    engine `.config`. The DSP/net-size/flag fields come from the config; the outer
    optimizer knobs (`adim`, `coeff_NN`, `balance`, `BalanceBackProp`) are NOT engine
    config keys -- they carry the Train_BLSTM.m defaults (algo 6 -> balance 10, the LID
    calibration objective). NNType 0 only (SRN/CWRNN unported)."""
    algo = int(cfg["Algo_choice"])
    if balance is None:
        balance = 10 if algo in (5, 6) else 5
    lid = _lid_from_config(cfg) if algo == 6 else None
    return RunConfig(
        numOuterThreads=int(cfg.get("numOuterThreads", "1")),
        numInnerThreads=int(cfg.get("numInnerThreads", "1")),
        OutputFile=cfg.get("multiConfigResultsOutputFile", "MultiConfigResults.mat"),
        Display_MillisecondsPerPixel=float(cfg.get("Display_MillisecondsPerPixel", "64")),
        name_dir_fig=cfg.get("Display_Output_Directory", "figs"),
        offset=float(cfg.get("Audio_offset", "0")),
        durmax=float(cfg.get("Audio_max_duration", "120")),
        algo=algo,
        nbworker=1,
        name_dir="run",
        epoch=0,
        adim=adim,
        coeff_NN=coeff_nn,
        VRCTS_isFast=1,
        VRCTS_force=0,
        balance=balance,
        exclude_nontrans=1 if cfg.get("exclude_nontrans", "false") == "true" else 0,
        useVRCTSFeatures=int(cfg.get("BLSTM_use_cep_files", "0")),
        nnmatfile=cfg.get("BLSTM_weightsFile", "NNweights.mat"),
        minSegmentLength=0.0,
        addNoise=0.0,
        mappingFile=cfg["language2classmapping"],
        listing=cfg["fileslisting"],
        BackPropagationActivated=1,
        NNType=0,
        LSTM_net_size=_ints(cfg, "BLSTM_LSTMNeuronNb"),
        LSTMSubSampling=_ints(cfg, "BLSTM_LSTMSubSampling"),
        Output_net_size=_ints(cfg, "BLSTM_OutputNeuronNb"),
        OutputSubSampling=_ints(cfg, "BLSTM_OutputSubSampling"),
        LSTM_MaxSat=float(cfg.get("BLSTM_Forward_MaxSaturation", "-10.0")),
        BackPropWER=float(cfg.get("BLSTM_BackPropWER", "-1.0")),
        BalanceBackProp=0.5,
        InputNormalizationType=int(cfg.get("BLSTM_InputNormalizationType", "0")),
        IsCellsPeepholesActive=_flag(cfg, "BLSTM_Forward_IsCellsPeepholesActive", 1),
        IsGatesPeepholesActive=_flag(cfg, "BLSTM_Forward_IsGatesPeepholesActive", 1),
        IsGatesRecurrentPeepholesActive=_flag(cfg, "BLSTM_Forward_IsGatesRecurrentPeepholesActive", 1),
        lid=lid,
    )


class RunState(BaseModel):
    """The typed run state: the parsed engine config, the derived vec2struct `RunConfig`,
    the corpus listing records, and the run directories. Persisted as JSON."""

    model_config = ConfigDict(extra="forbid")

    config_path: str  # absolute path to the engine .config (its dir holds the corpus)
    out_dir: str  # where checkpoints land
    algo: int
    balance: int
    base_config: dict[str, str]  # the parsed engine config (the byte-known-good base)
    ps: RunConfig  # the vec2struct genome spec
    listing: list[dict[str, str]]  # read_listing records

    def save(self, path: Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> RunState:
        return cls.model_validate_json(Path(path).read_text())

    @classmethod
    def from_config(cls, config: Path, out_dir: Path) -> RunState:
        config = Path(config).resolve()
        cfg = parse_legacy_config(config.read_text())
        ps = ps_from_config(cfg)
        listing_path = config.parent / cfg["fileslisting"]
        listing = read_listing(listing_path) if listing_path.exists() else []
        return cls(
            config_path=str(config),
            out_dir=str(Path(out_dir).resolve()),
            algo=ps.algo,
            balance=ps.balance,
            base_config=cfg,
            ps=ps,
            listing=listing,
        )


class TrainResult(BaseModel):
    """The outer-loop outcome + checkpoint location. `gbest` is the QPSO genome,
    `cost_history` the per-epoch gbestval trajectory, `inner_cost_history` the final
    SMORMS3 inner-loop cost trace (BackPropagation on the gbest). `penalized_evals`/
    `penalized_types` are `train_hyperparam_search`-only observability (Phase 5 Task 9 fix
    wave): a count of per-candidate evals that hit the `_HYPERPARAM_PENALTY` catch, broken
    down by `type(exc).__name__`, so a genuine engine defect can be told apart from a
    legitimately-invalid DSP subregion. The legacy `train` never penalizes (no catch on its
    eval path), so both default to empty/zero for that path."""

    model_config = ConfigDict(extra="forbid")

    gbest: list[float]
    gbestval: float
    cost_history: list[float]
    inner_cost_history: list[float]
    checkpoint_dir: str
    penalized_evals: int = 0
    penalized_types: dict[str, int] = Field(default_factory=dict)


class ModernTrainParams(BaseModel):
    """Driver-side knobs for `drivers.train.train_modern` -- the MODERN loop (from-scratch
    seeded init + SMORMS3 epochs + forward-only validation + early-stop). These are ALL
    driver-side (there are NO engine `.config`/TOML keys for them; the `[training]` section
    in `configs/lid/lid_blstm.toml` is a commented CONVENTION example only, never parsed by
    the engine). `epochs` is the TOTAL target (resume continues toward it, not `+epochs`)."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = 50  # total epochs (resume continues toward this)
    patience: int = 5  # early-stop after this many epochs with no validation-cost improvement
    steps_per_epoch: int = 10  # SMORMS3 steps per epoch (no LR schedule; fresh optimizer per epoch)
    valid_listing: str | None = None  # validation fileslisting (rel to config dir); None -> reuse the training listing
    # The forward-only validation cost the early-stop follows (Phase 6 Task 6, a Phase-5 carry-forward).
    # "nn_cost_seg" (DEFAULT): the CONTINUOUS NNCostSeg objective (the same `f` forward_backward descends,
    # F10-fixed) -- it MOVES from scratch. "balance": the phase-5 balance-law cost, which on the SAD
    # from-scratch fixture collapses to the discrete `100 - success` rate STUCK at 30.0 -- a useless
    # early-stop plateau (why nn_cost_seg is now the default; see IMPROVEMENTS.md / the pyo3 smoke).
    val_metric: Literal["nn_cost_seg", "balance"] = "nn_cost_seg"
    # hard-example mini-batching (Task 4/5 fixed batching); minibatch == 0 -> full-corpus training per epoch.
    minibatch: int = 0
    nb_worst: int = 0
    nb_classes: int = 1
    multilingual: bool = False
    # from-scratch seeded init (Task 3 `init_weights`); ignored when `resume_from` is set.
    init_scheme: Literal["xavier", "he"] = "xavier"
    init_seed: int = 0
    forget_bias_one: bool = True
    resume_from: str | None = None  # checkpoint dir to resume from (loads last_*.bin + train_history.json); None -> from-scratch


class EpochRecord(BaseModel):
    """One epoch's recorded metrics in `train_history.json`. `confusion_error` is the
    FIXED (F5) confusion misclassification rate on the validation set (algo 6 only; `None`
    for the single-net SAD algos, which carry no LID confusion)."""

    model_config = ConfigDict(extra="forbid")

    epoch: int
    train_cost: float
    val_cost: float
    confusion_error: float | None
    is_best: bool


class ModernTrainResult(BaseModel):
    """`train_modern`'s outcome + the checkpoint location (also serialized to
    `train_history.json` for resume). Distinct from `TrainResult` (the QPSO-shaped legacy
    outcome): the modern loop has no genome, so `gbest`/`gbestval` would be meaningless --
    this carries the best-epoch selection, the per-epoch history, and the early-stop flag."""

    model_config = ConfigDict(extra="forbid")

    best_epoch: int
    best_val_cost: float
    epochs_run: int
    stopped_early: bool
    history: list[EpochRecord]
    checkpoint_dir: str
