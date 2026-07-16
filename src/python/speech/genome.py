"""The vec2struct genome<->config bijection.

Ported from legacy MATLAB `Optimizer_V6.2.2/functions/vec2struct.m` (1693 LOC) + the
`printConfig.m` serialization it calls. This is the single function mapping the optimizer's
flat parameter vector (the "genome") to a full engine config and, via `out_param`, back --
the inverse write-back that mask-fixed / clamped / sorted fields impose.

Two outputs matter for parity:
  * `configStruct` (returned here as the flat legacy KEY->value STRING dict, exactly the shape
    `config_bridge.parse_legacy_config` produces): the engine-facing `.config`. printConfig
    EXCLUDES the NN weight / output / normalization fields (they reach the engine through the
    `.bin` weight codec, not the text config) and renames the `AlgName_` prefix to the algo tag
    (`BLSTM_`/`LTSV_`/`TDC_`/`VRCTS_`). Numeric values are `%d` when integer-valued, `%15.15e`
    otherwise (per element for vectors).
  * `out_param`: `param` with the mask inverse-encodes, the `sortrows` re-orderings, and the
    calibration clamps written back -- so a masked/sorted genome round-trips.

`count_param` (the genome length) is a pure function of the PS spec (algo + net sizes +
balance + flags), independent of the param values or the mask; `genome_length` runs the same
walk with a zero genome to return it.

vec2struct is PURE arithmetic (rem/round/abs/min/max/sortrows -- no libm), so out_param is
bit-exact and the config string-exact on every platform (bit-pinned vs Octave in
tests/test_phase4c_genome.py). Legacy quirks reproduced on purpose are tracked in
IMPROVEMENTS.md; the load-bearing ones are commented `# legacy quirk:` at their site.

RunConfig carries ONLY the PS fields vec2struct reads; drivers/state.py may re-export it
later (T12) if it needs the same shape.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

# rem(round(5*|p|/adim),5) -> window / cost-law tables; 2-way for the boolean fields.
_WINDOW_TYPES: list[object] = ["hamming", "hann", "hHCw", "uniform", "none"]
_COST_LAWS: list[object] = ["log", "linear", "square", "sqrt", "cubic"]
_BOOL: list[object] = ["false", "true"]
_NOISE_SEED: list[object] = [-3, 3]


class LidNetSpec(BaseModel):
    """PS.NS.LID.* + PS.VP.LID.nnmatfile -- read only when algo == 6 (NNType 0 only)."""

    model_config = ConfigDict(extra="forbid")

    NNType: int  # PS.NS.LID.NNType
    BackPropagationActivated: int  # PS.NS.LID.BackPropagationActivated
    LSTM_net_size: list[int]  # PS.NS.LID.LSTM_net_size
    LSTMSubSampling: list[int]  # PS.NS.LID.LSTMSubSampling
    Output_net_size: list[int]  # PS.NS.LID.Output_net_size
    OutputSubSampling: list[int]  # PS.NS.LID.OutputSubSampling
    Mode: int  # PS.NS.LID.Mode
    PostProcessMode: int  # PS.NS.LID.PostProcessMode
    TargetEnforcementStep: int  # PS.NS.LID.TargetEnforcementStep
    BackPropWER: float  # PS.NS.LID.BackPropWER
    classes_ponderations: list[float]  # PS.NS.LID.classes_ponderations ([] -> field omitted)
    InputNormalizationType: int  # PS.NS.LID.InputNormalizationType
    IsCellsPeepholesActive: int  # PS.NS.LID.IsCellsPeepholesActive
    IsGatesPeepholesActive: int  # PS.NS.LID.IsGatesPeepholesActive
    IsGatesRecurrentPeepholesActive: int  # PS.NS.LID.IsGatesRecurrentPeepholesActive
    LSTM_MaxSat: float  # PS.NS.LID.LSTM.MaxSat
    nnmatfile: str  # PS.VP.LID.nnmatfile


class RunConfig(BaseModel):
    """The PS fields vec2struct reads (Train_BLSTM_Seg.m:126-345 provenance)."""

    model_config = ConfigDict(extra="forbid")

    # PS.VP.* front matter + PS.FS paths
    numOuterThreads: int  # PS.VP.nbproc.outer
    numInnerThreads: int  # PS.VP.nbproc.inner
    OutputFile: str  # PS.VP.OutputFile -> multiConfigResultsOutputFile
    Display_MillisecondsPerPixel: float  # PS.VP.Display_MillisecondsPerPixel
    name_dir_fig: str  # PS.FS.name_dir_fig -> Display_Output_Directory
    offset: float  # PS.VP.offset -> Audio_offset
    durmax: float  # PS.VP.durmax -> Audio_max_duration
    algo: int  # PS.VP.algo -> Algo_choice
    nbworker: int  # PS.VP.nbworker (nbworker>1 -> LockFilesDir/Prefix)
    name_dir: str  # PS.FS.name_dir (LockFilesDir base)
    epoch: int  # PS.VP.epoch (LockFilesPrefix)
    adim: float  # PS.VP.adim
    coeff_NN: float  # PS.NS.coeff_NN
    VRCTS_isFast: int  # PS.VP.VRCTS_isFast (algo 0)
    VRCTS_force: int  # PS.VP.VRCTS_force (algo 0)
    balance: int  # PS.VP.balance
    exclude_nontrans: int  # PS.VP.exclude_nontrans
    useVRCTSFeatures: int  # PS.VP.useVRCTSFeatures
    nnmatfile: str  # PS.VP.nnmatfile -> AlgName_weightsFile
    minSegmentLength: float  # PS.VP.minSegmentLength (algo 6)
    addNoise: float  # PS.VP.addNoise (algo 6)
    mappingFile: str  # PS.Corpora.mappingFile -> language2classmapping
    listing: str  # PS.FS.listing -> fileslisting
    # isfield(PS.VP.mask, 'nnet'/'nnetLID') -> BackPropOutputNetworkOnly toggles
    mask_has_nnet: bool = False
    mask_has_nnetLID: bool = False
    # PS.NS.* SAD net
    BackPropagationActivated: int  # PS.NS.BackPropagationActivated
    NNType: int  # PS.NS.NNType
    LSTM_net_size: list[int]  # PS.NS.LSTM_net_size
    LSTMSubSampling: list[int]  # PS.NS.LSTMSubSampling
    Output_net_size: list[int]  # PS.NS.Output_net_size
    OutputSubSampling: list[int]  # PS.NS.OutputSubSampling
    LSTM_MaxSat: float  # PS.NS.LSTM.MaxSat
    BackPropWER: float  # PS.NS.BackPropWER
    BalanceBackProp: float  # PS.NS.BalanceBackProp
    InputNormalizationType: int  # PS.NS.InputNormalizationType
    IsCellsPeepholesActive: int  # PS.NS.IsCellsPeepholesActive
    IsGatesPeepholesActive: int  # PS.NS.IsGatesPeepholesActive
    IsGatesRecurrentPeepholesActive: int  # PS.NS.IsGatesRecurrentPeepholesActive
    lid: LidNetSpec | None = None  # PS.NS.LID.* (+ PS.VP.LID); required for algo == 6


def _mround_s(x: float) -> float:
    """MATLAB round: half away from zero (numpy's is banker's)."""
    return float(np.sign(x) * np.floor(np.abs(x) + 0.5))


class _Walk:
    """One vec2struct traversal. `param is None` = the count-only (genome_length) walk:
    param slices read as zeros and out_param writes no-op, so only `count` is meaningful."""

    def __init__(self, param: NDArray[np.float64] | None, mask: dict[str, object] | None, ps: RunConfig) -> None:
        self.param = param
        self.out: NDArray[np.float64] | None = None if param is None else param.astype(np.float64).copy()
        self.mask = mask
        self.ps = ps
        self.adim = float(ps.adim)
        self.coeff = float(ps.coeff_NN)
        self.count = 1
        self.cfg: dict[str, object] = {}
        self.algname = "AlgName_"

    # --- param / out_param access (1-based count, mirroring MATLAB) ---
    def _p1(self) -> float:
        return 0.0 if self.param is None else float(self.param[self.count - 1])

    def _pv(self, n: int) -> NDArray[np.float64]:
        if self.param is None:
            return np.zeros(n, dtype=np.float64)
        return np.asarray(self.param[self.count - 1 : self.count - 1 + n], dtype=np.float64).copy()

    def _set(self, k: int, val: float) -> None:
        if self.out is not None:
            self.out[self.count - 1 + k] = val

    def _setv(self, vals: NDArray[np.float64]) -> None:
        if self.out is not None:
            self.out[self.count - 1 : self.count - 1 + len(vals)] = vals

    # --- mask access ---
    def _mhas(self, name: str) -> bool:
        return self.mask is not None and name in self.mask

    def _mget(self, name: str) -> object:
        assert self.mask is not None
        return self.mask[name]

    def _mget_f(self, name: str) -> float:
        return float(self._mget(name))  # type: ignore[arg-type]

    def _mget_v(self, name: str) -> NDArray[np.float64]:
        return np.atleast_1d(np.asarray(self._mget(name), dtype=np.float64)).reshape(-1)

    def _fir(self) -> bool:
        return self._mhas("Force_Identical_Rows") and self._mget("Force_Identical_Rows") == 1

    def _fs(self) -> bool:
        return self._mhas("Force_Symetry") and self._mget("Force_Symetry") == 1

    # --- reusable field kinds ---
    def _choice(self, name: str, tables: list[object], radix: int) -> None:
        """rem(round(radix*|p|/adim), radix) -> tables[k]; mask fixes + inverse-encodes."""
        tmp = int(_mround_s(radix * abs(self._p1() / self.adim))) % radix
        fieldValue = tables[tmp]
        if self._mhas(name):
            fieldValue = self._mget(name)
            tmp = tables.index(fieldValue)
            self._set(0, tmp / radix * self.adim)
        self.cfg[name] = fieldValue
        self.count += 1

    def _nn_block(self, name: str, nb_need: int, fir_ref: str | None, fs_ref: str | None) -> None:
        """A weight block: coeff_NN*(2p/adim-1); mask, then Force_Identical_Rows (fir_ref set
        only when jj>1), then Force_Symetry (fs_ref set only for backward blocks). Last wins."""
        coeff, adim = self.coeff, self.adim
        field = coeff * (2.0 * self._pv(nb_need) / adim - 1.0)
        if self._mhas(name):
            field = self._mget_v(name)
            self._setv((field / coeff + 1.0) / 2.0 * adim)
        if fir_ref is not None and self._fir():
            field = np.atleast_1d(np.asarray(self.cfg[fir_ref], dtype=np.float64)).reshape(-1)
            self._setv((field / coeff + 1.0) / 2.0 * adim)
        if fs_ref is not None and self._fs():
            field = np.atleast_1d(np.asarray(self.cfg[fs_ref], dtype=np.float64)).reshape(-1)
            self._setv((field / coeff + 1.0) / 2.0 * adim)
        self.cfg[name] = field
        self.count += nb_need

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        ps = self.ps
        cfg = self.cfg

        # ---- non-genome header (vec2struct.m:5-27) ----
        cfg["numOuterThreads"] = int(ps.numOuterThreads)
        cfg["numInnerThreads"] = int(ps.numInnerThreads)
        cfg["multiConfigResultsOutputFile"] = ps.OutputFile
        cfg["Display_Height"] = 0
        cfg["Display_MillisecondsPerPixel"] = ps.Display_MillisecondsPerPixel
        cfg["Display_WaveAll_Height"] = 50
        cfg["Display_WaveAll_Width"] = 1000
        cfg["Display_Output_Directory"] = ps.name_dir_fig
        cfg["Audio_offset"] = ps.offset
        cfg["Audio_max_duration"] = ps.durmax
        cfg["Algo_choice"] = int(ps.algo)
        if ps.nbworker > 1:  # legacy: a pwd-dependent LockFilesDir -- tests keep nbworker == 1
            cfg["LockFilesDir"] = f"{ps.name_dir}/LockFiles"
            cfg["LockFilesPrefix"] = str(ps.epoch)

        self._front_matter()
        self._algo_branch()
        self._calibration()
        self._tail()

    # ---- front matter (:32-243) ----
    def _front_matter(self) -> None:
        adim = self.adim
        cfg = self.cfg

        # decision_thresh_rising
        fv = abs(self._p1()) / adim
        if self._mhas("AlgName_decision_thresh_rising"):
            fv = self._mget_f("AlgName_decision_thresh_rising")
            self._set(0, fv * adim)
        cfg["AlgName_decision_thresh_rising"] = fv
        self.count += 1

        # decision_area_rising
        fv = 0.01 * abs(self._p1()) / adim
        if self._mhas("AlgName_decision_area_rising"):
            fv = self._mget_f("AlgName_decision_area_rising")
            self._set(0, fv / 0.01 * adim)
        cfg["AlgName_decision_area_rising"] = fv
        self.count += 1

        # decision_thresh_falling (clamped to rising)
        fv = abs(self._p1()) / adim
        if self._mhas("AlgName_decision_thresh_falling"):
            fv = self._mget_f("AlgName_decision_thresh_falling")
            self._set(0, fv * adim)
        if fv > float(cfg["AlgName_decision_thresh_rising"]):  # type: ignore[arg-type]
            fv = float(cfg["AlgName_decision_thresh_rising"])  # type: ignore[arg-type]
            self._set(0, fv * adim)
        cfg["AlgName_decision_thresh_falling"] = fv
        self.count += 1

        # decision_area_falling
        fv = 0.01 * abs(self._p1()) / adim
        if self._mhas("AlgName_decision_area_falling"):
            fv = self._mget_f("AlgName_decision_area_falling")
            self._set(0, fv / 0.01 * adim)
        cfg["AlgName_decision_area_falling"] = fv
        self.count += 1

        self._choice("AlgName_windowing_type", _WINDOW_TYPES, 5)

        # windowing_param (min 100 AFTER out_param encode)
        fv = abs(self._p1()) / adim
        if self._mhas("AlgName_windowing_param"):
            fv = self._mget_f("AlgName_windowing_param")
            self._set(0, fv * adim)
        cfg["AlgName_windowing_param"] = min(100.0, fv)
        self.count += 1

        self._choice("AlgName_convolution_window_type", _WINDOW_TYPES, 5)

        # convolution_window_size (round(abs), min 100)
        fv = _mround_s(abs(self._p1()))
        if self._mhas("AlgName_convolution_window_size"):
            fv = self._mget_f("AlgName_convolution_window_size")
            self._set(0, fv)
        cfg["AlgName_convolution_window_size"] = min(100.0, fv)
        self.count += 1

        self._choice("AlgName_flag_DCOffset", _BOOL, 2)

        # preemph_ratio (NO abs; min 2)
        fv = self._p1() / adim
        if self._mhas("AlgName_preemph_ratio"):
            fv = self._mget_f("AlgName_preemph_ratio")
            self._set(0, fv * adim)
        cfg["AlgName_preemph_ratio"] = min(2.0, fv)
        self.count += 1

        self._choice("AlgName_noise_seed", _NOISE_SEED, 2)

        # noise_ratio (min 1)
        fv = abs(self._p1()) / adim
        if self._mhas("AlgName_noise_ratio"):
            fv = self._mget_f("AlgName_noise_ratio")
            self._set(0, fv * adim)
        cfg["AlgName_noise_ratio"] = min(1.0, fv)
        self.count += 1

        # speech_padding (4), min_silence (2), min_speech (3): fv = -0.1 + |p/adim|, min 10
        self._padding_block("AlgName_speech_padding", 4)
        self._padding_block("AlgName_min_silence", 2)
        fvv = self._padding_block("AlgName_min_speech", 3, defer_store=True)
        if self.ps.algo < 5:  # legacy: last min_speech element floored at 0 only below algo 5
            fvv[2] = max(0.0, fvv[2])
        self.cfg["AlgName_min_speech"] = fvv

    def _padding_block(self, name: str, n: int, defer_store: bool = False) -> NDArray[np.float64]:
        adim = self.adim
        fv = -0.1 + np.abs(self._pv(n) / adim)
        if self._mhas(name):
            fv = self._mget_v(name)
            self._setv((fv + 0.1) * adim)
        fv = np.minimum(10.0, fv)
        if not defer_store:
            self.cfg[name] = fv
        self.count += n
        return fv

    # ---- algo branch (:247-1290) ----
    def _algo_branch(self) -> None:
        algo = self.ps.algo
        if algo == 0:
            self._algo_vrcts()
        elif algo == 1:
            self._algo_tdc()
        elif algo in (2, 3, 4, 5, 6):
            self._algo_spectral()

    def _algo_vrcts(self) -> None:
        self.algname = "VRCTS_"
        self.cfg["AlgName_window"] = 0
        self.cfg["AlgName_shift"] = 1
        self.cfg["AlgName_isFast"] = int(self.ps.VRCTS_isFast)
        self.cfg["AlgName_force"] = int(self.ps.VRCTS_force)

    def _algo_tdc(self) -> None:
        adim = self.adim
        self.algname = "TDC_"
        c = self.count
        param_tmp = 0.001 + 0.1 * np.sort(np.abs(self._pv(3) / adim))
        self._setv((param_tmp - 0.001) / 0.1 * adim)

        fv = float(param_tmp[2])
        if self._mhas("AlgName_window"):
            fv = self._mget_f("AlgName_window")
            self._set(2, (fv - 0.001) / 0.1 * adim)
        fv = min(10.0, fv)
        self.cfg["AlgName_window"] = fv
        thresh = fv

        lags = np.minimum(param_tmp[0:2], thresh)
        self._setv_at(c, (lags - 0.001) / 0.1 * adim)
        if self._mhas("AlgName_lags"):
            lags = np.minimum(self._mget_v("AlgName_lags"), thresh)
            self._setv_at(c, (lags - 0.001) / 0.1 * adim)
        self.cfg["AlgName_lags"] = lags
        self.count += 3

        fv = self._p1() / adim
        if self._mhas("AlgName_balance"):
            fv = self._mget_f("AlgName_balance")
            self._set(0, fv * adim)
        self.cfg["AlgName_balance"] = fv
        self.count += 1

        fv = 0.001 + abs(self._p1() / adim)
        self._set(0, (fv - 0.001) * adim)
        if self._mhas("AlgName_shift"):
            fv = self._mget_f("AlgName_shift")
            self._set(0, (fv - 0.001) * adim)
        self.cfg["AlgName_shift"] = fv
        self.count += 1

    def _setv_at(self, count1: int, vals: NDArray[np.float64]) -> None:
        if self.out is not None:
            self.out[count1 - 1 : count1 - 1 + len(vals)] = vals

    def _algo_spectral(self) -> None:
        ps = self.ps
        algo = ps.algo
        adim = self.adim
        cfg = self.cfg
        if algo == 2:
            self.algname = "LTSV_"

        if algo != 4:
            # spectrum_order
            fv = float(5 + int(_mround_s(7 * abs(self._p1() / adim))) % 7)
            if self._mhas("AlgName_spectrum_order"):
                fv = self._mget_f("AlgName_spectrum_order")
                self._set(0, (fv - 5) / 7 * adim)
            cfg["AlgName_spectrum_order"] = fv
            self.count += 1

            # spectrum_shift (min 0.5, offset 0.005)
            fv = min(0.5, 0.005 + abs(self._p1() / adim))
            if self._mhas("AlgName_spectrum_shift"):
                fv = min(0.5, self._mget_f("AlgName_spectrum_shift"))
                self._set(0, (fv - 0.005) * adim)
            cfg["AlgName_spectrum_shift"] = fv
            self.count += 1
            spectrum_shift = fv

            self._choice("AlgName_spectrum_temporal_convolution_type", _WINDOW_TYPES, 5)

            fv = _mround_s(abs(self._p1()))
            if self._mhas("AlgName_spectrum_temporal_convolution_size"):
                fv = self._mget_f("AlgName_spectrum_temporal_convolution_size")
                self._set(0, fv)
            cfg["AlgName_spectrum_temporal_convolution_size"] = min(100.0, fv)
            self.count += 1

            # AlgName_window: spectrum_shift * |param| (raw abs, not /adim); mask sign quirk
            self._shift_scaled("AlgName_window", spectrum_shift, neg_mask=True, clamp10=False)
            self._shift_scaled("AlgName_shift", spectrum_shift, neg_mask=False, clamp10=False)

            if algo in (3, 5, 6):
                self._ltsv_tdc_sub(spectrum_shift)

            # minFreq
            fv = abs(500.0 * (self._p1() / adim))
            if self._mhas("AlgName_minFreq"):
                fv = self._mget_f("AlgName_minFreq")
                self._set(0, fv / 500.0 * adim)
            cfg["AlgName_minFreq"] = fv
            self.count += 1

            # maxFreq
            fv = abs(8000.0 - 1000.0 * (self._p1() / adim))
            if self._mhas("AlgName_maxFreq"):
                fv = self._mget_f("AlgName_maxFreq")
                self._set(0, -(fv - 8000.0) / 1000.0 * adim)
            cfg["AlgName_maxFreq"] = fv
            self.count += 1

            # min/max Mel (sortrows * 4000)
            param_tmp = 4000.0 * np.sort(np.abs(self._pv(2) / adim))
            self._setv(param_tmp / 4000.0 * adim)
            fv = float(param_tmp[0])
            if self._mhas("AlgName_minMelFreq"):
                fv = self._mget_f("AlgName_minMelFreq")
                self._set(0, fv / 4000.0 * adim)
            cfg["AlgName_minMelFreq"] = fv
            fv2 = float(param_tmp[1])
            if self._mhas("AlgName_maxMelFreq"):
                fv2 = self._mget_f("AlgName_maxMelFreq")
                self._set(1, fv2 / 4000.0 * adim)
            cfg["AlgName_maxMelFreq"] = fv2
            self.count += 2

            fv = _mround_s(abs(self._p1()))
            if self._mhas("AlgName_nb_bins"):
                fv = self._mget_f("AlgName_nb_bins")
                self._set(0, fv)
            cfg["AlgName_nb_bins"] = fv
            self.count += 1

            self._choice("AlgName_is_log_mel", _BOOL, 2)

            fv = _mround_s(abs(self._p1()))
            if self._mhas("AlgName_nb_DCT"):
                fv = self._mget_f("AlgName_nb_DCT")
                self._set(0, fv)
            cfg["AlgName_nb_DCT"] = fv
            self.count += 1

            # ComputeDeltasNb: round(param) with NO abs (legacy), min 100
            fv = _mround_s(self._p1())
            if self._mhas("AlgName_ComputeDeltasNb"):
                fv = self._mget_f("AlgName_ComputeDeltasNb")
                self._set(0, fv)
            cfg["AlgName_ComputeDeltasNb"] = min(100.0, fv)
            self.count += 1

            fv = _mround_s(abs(self._p1()))
            if self._mhas("AlgName_ComputeDeltaDeltasNb"):
                fv = self._mget_f("AlgName_ComputeDeltaDeltasNb")
                self._set(0, fv)
            cfg["AlgName_ComputeDeltaDeltasNb"] = min(100.0, fv)
            self.count += 1

            self._choice("AlgName_IgnoreFirstDCT", _BOOL, 2)
        else:
            # algo == 4 (signal): window / shift only
            fv = abs(self._p1() / adim)
            if self._mhas("AlgName_window"):
                fv = self._mget_f("AlgName_window")
                self._set(0, fv * adim)
            cfg["AlgName_window"] = fv
            self.count += 1
            fv = abs(self._p1() / adim)
            self._set(0, fv * adim)
            if self._mhas("AlgName_shift"):
                fv = self._mget_f("AlgName_shift")
                self._set(0, fv * adim)
            cfg["AlgName_shift"] = fv
            self.count += 1

        if algo in (3, 4, 5, 6):
            self._blstm(spectrum_shift if algo != 4 else 0.0)

    def _shift_scaled(self, name: str, spectrum_shift: float, neg_mask: bool, clamp10: bool) -> None:
        """spectrum_shift * |param(count)| (raw abs); mask inverse = value/spectrum_shift.
        neg_mask: a negative mask value is re-signed to -spectrum_shift*mask (window/LTSVwindow/
        LID_window). clamp10: min(10, .) after the encode (LTSVwindow)."""
        fv = spectrum_shift * abs(self._p1())
        if self._mhas(name):
            m = self._mget_f(name)
            fv = -spectrum_shift * m if neg_mask and m < 0 else m
            self._set(0, fv / spectrum_shift)
        if clamp10:
            fv = min(10.0, fv)
        self.cfg[name] = fv
        self.count += 1

    def _ltsv_tdc_sub(self, spectrum_shift: float) -> None:
        adim = self.adim
        cfg = self.cfg
        self._shift_scaled("AlgName_LTSVwindow", spectrum_shift, neg_mask=True, clamp10=True)

        # LTSVshift: spectrum_shift * (1 + |param|); mask inverse = value/spectrum_shift - 1
        fv = spectrum_shift * (1.0 + abs(self._p1()))
        if self._mhas("AlgName_LTSVshift"):
            fv = self._mget_f("AlgName_LTSVshift")
            self._set(0, fv / spectrum_shift - 1.0)
        cfg["AlgName_LTSVshift"] = fv
        self.count += 1

        # TDC window/lags (sortrows * 0.1)
        c = self.count
        param_tmp = 0.1 * np.sort(np.abs(self._pv(3) / adim))
        self._setv(param_tmp / 0.1 * adim)
        fv = float(param_tmp[2])
        if self._mhas("AlgName_TDCwindow"):
            fv = self._mget_f("AlgName_TDCwindow")
            self._set(2, fv / 0.1 * adim)
        fv = min(10.0, fv)
        cfg["AlgName_TDCwindow"] = fv
        thresh = fv
        lags = np.minimum(param_tmp[0:2], thresh)
        self._setv_at(c, lags / 0.1 * adim)
        if self._mhas("AlgName_TDC_lags"):
            lags = np.minimum(self._mget_v("AlgName_TDC_lags"), thresh)
            self._setv_at(c, lags / 0.1 * adim)
        cfg["AlgName_TDC_lags"] = lags
        self.count += 3

        fv = self._p1() / adim
        if self._mhas("AlgName_TDC_balance"):
            fv = self._mget_f("AlgName_TDC_balance")
            self._set(0, fv * adim)
        cfg["AlgName_TDC_balance"] = fv
        self.count += 1

        fv = 0.001 + abs(self._p1() / adim)
        self._set(0, (fv - 0.001) * adim)
        if self._mhas("AlgName_TDCshift"):
            fv = self._mget_f("AlgName_TDCshift")
            self._set(0, (fv - 0.001) * adim)
        cfg["AlgName_TDCshift"] = fv
        self.count += 1

        self._choice("AlgName_TDC_windowing_type", _WINDOW_TYPES, 5)

        fv = abs(self._p1()) / adim
        if self._mhas("AlgName_TDC_windowing_param"):
            fv = self._mget_f("AlgName_TDC_windowing_param")
            self._set(0, fv * adim)
        cfg["AlgName_TDC_windowing_param"] = min(100.0, fv)
        self.count += 1

    def _blstm(self, spectrum_shift: float) -> None:
        ps = self.ps
        cfg = self.cfg
        self.algname = "BLSTM_"
        cfg["AlgName_BackPropagationActivated"] = "true" if ps.BackPropagationActivated == 1 else "false"
        self._choice("AlgName_TwoSweeps", _BOOL, 2)
        cfg["AlgName_LSTMNeuronNb"] = list(ps.LSTM_net_size)
        cfg["AlgName_LSTMSubSampling"] = list(ps.LSTMSubSampling)
        self._lstm_and_output(
            "AlgName",
            ps.NNType,
            ps.LSTM_net_size,
            ps.LSTMSubSampling,
            ps.Output_net_size,
            ps.OutputSubSampling,
        )
        cfg["AlgName_weightsFile"] = ps.nnmatfile
        if ps.algo == 6:
            self._lid(spectrum_shift)

    def _lstm_and_output(
        self,
        pfx: str,
        nntype: int,
        lstm: list[int],
        lsub: list[int],
        out_net: list[int],
        out_sub: list[int],
    ) -> None:
        """The LSTM weight blocks + output layer, shared by the SAD (pfx='AlgName') and LID
        (pfx='AlgName_LID') nets. NNType 0 only (SRN/CWRNN weights are dead in legacy)."""
        cfg = self.cfg
        is_lid = pfx == "AlgName_LID"
        if lstm[0] == 0:
            cfg[f"{pfx}_NNetInputSize"] = out_net[0]
        else:
            if not is_lid:
                cfg["AlgName_NNType"] = nntype
                if nntype == 0:
                    cfg["AlgName_NNetInputSize"] = lstm[0]
                else:
                    raise NotImplementedError(f"vec2struct: NNType {nntype} (SRN/CWRNN) not ported")
            else:
                cfg["AlgName_LID_NNType"] = nntype
                cfg["AlgName_LID_NNetInputSize"] = lstm[0]
                if nntype != 0:
                    raise NotImplementedError(f"vec2struct: LID NNType {nntype} (SRN/CWRNN) not ported")

            for direction in ("Forward", "Backward"):
                for ii in range(1, len(lstm)):  # 1-based layer
                    for jj in range(1, lstm[ii] + 1):  # 1-based block
                        base = lstm[ii - 1] * lsub[ii - 1] + lstm[ii]
                        gate_nb = base + 2 + 3
                        cell_nb = base + 1
                        for gate, nb in (
                            ("InputGateWeights", gate_nb),
                            ("ForgetGateWeights", gate_nb),
                            ("OutputGateWeights", gate_nb),
                            ("CellWeight", cell_nb),
                        ):
                            name = f"{pfx}_{direction}_Layer_{ii - 1}_LSTMBlock_{jj - 1}_{gate}"
                            fir_ref = f"{pfx}_{direction}_Layer_{ii - 1}_LSTMBlock_0_{gate}" if jj > 1 else None
                            fs_ref = f"{pfx}_Forward_Layer_{ii - 1}_LSTMBlock_{jj - 1}_{gate}" if direction == "Backward" else None
                            self._nn_block(name, nb, fir_ref, fs_ref)

        cfg[f"{pfx}_OutputNeuronNb"] = list(out_net)
        cfg[f"{pfx}_OutputSubSampling"] = list(out_sub)
        for ii in range(1, len(out_net)):  # 1-based
            for jj in range(1, out_net[ii] + 1):
                nb = out_net[ii - 1] * out_sub[ii - 1] + 1
                name = f"{pfx}_Output_Layer_{ii - 1}_Neuron_{jj - 1}_Weights"
                self._output_neuron(name, nb, ii, out_net, out_sub)

        # normalization (NO adim scaling; excluded from the emitted config)
        nb = out_net[0] if lstm[0] == 0 else lstm[0]  # == NNetInputSize
        field = 2.0 * self._pv(nb) - 1.0
        if self._mhas(f"{pfx}_NormalizeInputMean"):
            field = self._mget_v(f"{pfx}_NormalizeInputMean")
            self._setv((field + 1.0) / 2.0)
        cfg[f"{pfx}_NormalizeInputMean"] = field
        self.count += nb
        field = 1e-3 + np.abs(self._pv(nb))
        if self._mhas(f"{pfx}_NormalizeInputStd"):
            field = np.abs(self._mget_v(f"{pfx}_NormalizeInputStd"))
            self._setv(field - 1e-3)
        cfg[f"{pfx}_NormalizeInputStd"] = field
        self.count += nb

    def _output_neuron(self, name: str, nb: int, ii: int, out_net: list[int], out_sub: list[int]) -> None:
        coeff, adim = self.coeff, self.adim
        field = coeff * (2.0 * self._pv(nb) / adim - 1.0)
        if self._mhas(name):
            field = self._mget_v(name)
            self._setv((field / coeff + 1.0) / 2.0 * adim)
        if ii == 1 and self._fir():
            # legacy repmat tie: [block-1 head] ++ [block-2 head] ++ bias, halving OutSize*OutSub
            half = (out_net[ii - 1] * out_sub[ii - 1]) // 2
            mid = out_net[ii - 1] // 2  # 0-based index of fieldValue(OutSize/2+1)
            field = np.concatenate([np.full(half, field[0]), np.full(half, field[mid]), field[-1:]])
            self._setv((field / coeff + 1.0) / 2.0 * adim)
        self.cfg[name] = field
        self.count += nb

    def _lid(self, spectrum_shift: float) -> None:
        ps = self.ps
        lid = ps.lid
        assert lid is not None, "algo 6 requires RunConfig.lid"
        adim = self.adim
        cfg = self.cfg

        # LID_decision_thresh_rising
        fv = self._p1() / adim
        if self._mhas("AlgName_LID_decision_thresh_rising"):
            fv = self._mget_f("AlgName_LID_decision_thresh_rising")
            self._set(0, fv * adim)
        cfg["AlgName_LID_decision_thresh_rising"] = fv
        self.count += 1

        # LID_decision_thresh_falling: computed/clamped, then OVERWRITTEN with -rising (quirk)
        fv = self._p1() / adim
        if self._mhas("AlgName_LID_decision_thresh_falling"):
            fv = self._mget_f("AlgName_LID_decision_thresh_falling")
            self._set(0, fv * adim)
        rising = float(cfg["AlgName_LID_decision_thresh_rising"])  # type: ignore[arg-type]
        if fv > rising:
            fv = rising
            self._set(0, fv * adim)
        fv = -rising  # legacy quirk: stored value is unconditionally -rising
        cfg["AlgName_LID_decision_thresh_falling"] = fv
        self.count += 1

        self._shift_scaled("AlgName_LID_window", spectrum_shift, neg_mask=True, clamp10=False)
        self._shift_scaled("AlgName_LID_shift", spectrum_shift, neg_mask=False, clamp10=False)

        cfg["AlgName_LID_BackPropagationActivated"] = "true" if lid.BackPropagationActivated == 1 else "false"
        self._choice("AlgName_LID_TwoSweeps", _BOOL, 2)
        cfg["AlgName_LID_LSTMNeuronNb"] = list(lid.LSTM_net_size)
        cfg["AlgName_LID_LSTMSubSampling"] = list(lid.LSTMSubSampling)
        self._lstm_and_output(
            "AlgName_LID",
            lid.NNType,
            lid.LSTM_net_size,
            lid.LSTMSubSampling,
            lid.Output_net_size,
            lid.OutputSubSampling,
        )
        cfg["AlgName_LID_weightsFile"] = lid.nnmatfile

    # ---- calibration cost laws (:1324-1421) ----
    def _calibration(self) -> None:
        self._choice("AlgName_CostLawSpeech", _COST_LAWS, 5)
        self._choice("AlgName_CostLawNoSpeech", _COST_LAWS, 5)
        self._clamped01("AlgName_CostLawParamSpeech")
        self._clamped01("AlgName_CostLawParamNoSpeech")
        self._clamped01("AlgName_CostLawThreshSpeech")
        self._clamped01("AlgName_CostLawThreshNoSpeech")

    def _clamped01(self, name: str) -> None:
        """max(0, min(1, p/adim)); out_param encoded UNCONDITIONALLY (not just under a mask)."""
        adim = self.adim
        fv = max(0.0, min(1.0, self._p1() / adim))
        self._set(0, fv * adim)
        if self._mhas(name):
            fv = self._mget_f(name)
            self._set(0, fv * adim)
        self.cfg[name] = fv
        self.count += 1

    # ---- tail: MaxSat, balance, corpus, feature flags, LID extras (:1423-1692) ----
    def _tail(self) -> None:
        ps = self.ps
        adim = self.adim
        cfg = self.cfg

        if ps.LSTM_MaxSat > 0:
            cfg["AlgName_Forward_MaxSaturation"] = ps.LSTM_MaxSat
            cfg["AlgName_Backward_MaxSaturation"] = ps.LSTM_MaxSat

        if ps.balance >= 9:
            if ps.BackPropWER >= 0:
                cfg["AlgName_BackPropWER"] = ps.BackPropWER
            fv = max(0.0, min(0.25, 0.25 * self._p1() / adim))
            self._set(0, fv * adim / 0.25)
            if self._mhas("Pruning_Threshold"):
                fv = self._mget_f("Pruning_Threshold")
                self._set(0, fv * adim / 0.25)
            cfg["Pruning_Threshold"] = fv
            self.count += 1

        if ps.balance == 4 or ps.balance >= 6:
            fv = max(0.01, min(0.99, self._p1() / adim))
            self._set(0, fv * adim)
            if self._mhas("AlgName_CostPonderation"):
                fv = self._mget_f("AlgName_CostPonderation")
                self._set(0, fv * adim)
            cfg["AlgName_CostPonderation"] = fv
            self.count += 1
        elif ps.balance == 5:
            cfg["AlgName_CostPonderation"] = ps.BalanceBackProp

        cfg["language2classmapping"] = ps.mappingFile
        if ps.exclude_nontrans == 1:
            cfg["exclude_nontrans"] = "true"

        if ps.algo in (3, 5, 6):
            cfg["AlgName_use_cep_files"] = int(ps.useVRCTSFeatures)
            if ps.useVRCTSFeatures == 5:
                cfg["File_Type"] = 1
            elif ps.useVRCTSFeatures == 6:
                cfg["File_Type"] = 4
            elif ps.useVRCTSFeatures > 0:
                cfg["File_Type"] = 2

        if ps.algo in (3, 4, 5, 6):
            if ps.mask_has_nnet:
                cfg["AlgName_BackPropOutputNetworkOnly"] = "true"
            cfg["AlgName_InputNormalizationType"] = ps.InputNormalizationType if ps.InputNormalizationType in (1, -1, -2) else 0
            self._peephole_flags("AlgName", ps.IsCellsPeepholesActive, ps.IsGatesPeepholesActive, ps.IsGatesRecurrentPeepholesActive)

        if ps.algo == 6:
            self._lid_tail()

        cfg["fileslisting"] = ps.listing

    def _peephole_flags(self, pfx: str, cells: int, gates: int, gates_rec: int) -> None:
        cfg = self.cfg
        for kind, val in (("IsCellsPeepholesActive", cells), ("IsGatesPeepholesActive", gates), ("IsGatesRecurrentPeepholesActive", gates_rec)):
            s = "false" if val == 0 else "true"
            cfg[f"{pfx}_Forward_{kind}"] = s
            cfg[f"{pfx}_Backward_{kind}"] = s

    def _lid_tail(self) -> None:
        ps = self.ps
        lid = ps.lid
        assert lid is not None
        adim = self.adim
        cfg = self.cfg

        if ps.mask_has_nnetLID:
            cfg["AlgName_LID_BackPropOutputNetworkOnly"] = "true"
        if ps.minSegmentLength > 0:
            cfg["AlgName_LID_MinNbOfFrames"] = ps.minSegmentLength
        if ps.addNoise > 0:
            cfg["AlgName_LID_NoiseMagnitude"] = ps.addNoise
        cfg["AlgName_LID_TargetEnforcementStep"] = lid.TargetEnforcementStep
        if lid.PostProcessMode > 0:
            cfg["AlgName_LID_PostProcessMode"] = lid.PostProcessMode
        cfg["AlgName_LID_Mode"] = lid.Mode
        cfg["AlgName_LID_BackPropWER"] = lid.BackPropWER

        fv = max(0.01, min(0.99, self._p1() / adim))
        self._set(0, fv * adim)
        if self._mhas("AlgName_LID_CostPonderation"):
            fv = self._mget_f("AlgName_LID_CostPonderation")
            self._set(0, fv * adim)
        cfg["AlgName_LID_CostPonderation"] = fv
        self.count += 1

        if len(lid.classes_ponderations) > 0:
            cfg["AlgName_LID_classes_ponderations"] = list(lid.classes_ponderations)

        self._choice("AlgName_LID_CostLawSpeech", _COST_LAWS, 5)
        self._choice("AlgName_LID_CostLawNoSpeech", _COST_LAWS, 5)
        self._clamped01("AlgName_LID_CostLawParamSpeech")
        self._clamped01("AlgName_LID_CostLawParamNoSpeech")
        self._clamped01("AlgName_LID_CostLawThreshSpeech")
        self._clamped01("AlgName_LID_CostLawThreshNoSpeech")

        if lid.LSTM_MaxSat > 0:
            cfg["AlgName_LID_Forward_MaxSaturation"] = lid.LSTM_MaxSat
            cfg["AlgName_LID_Backward_MaxSaturation"] = lid.LSTM_MaxSat

        cfg["AlgName_LID_InputNormalizationType"] = lid.InputNormalizationType if lid.InputNormalizationType in (1, -1, -2) else 0
        self._peephole_flags("AlgName_LID", lid.IsCellsPeepholesActive, lid.IsGatesPeepholesActive, lid.IsGatesRecurrentPeepholesActive)


# ---- printConfig serialization (printConfig.m ACTIVE inline condition, :8-9) ----
def _printconfig_written(name: str) -> bool:
    """Whether printConfig emits `name`. Mirrors the ACTIVE inline boolean (:8-9), which
    DIFFERS from the commented-out isNotExcluded: the Forward_/Backward_/LID peephole-flag +
    MaxSaturation fields are KEPT (whitelisted), only the weight/output/normalize matrices are
    dropped (they reach the engine via the .bin codec). Verified against the real 1_worker_1
    config (peephole flags present, weights absent)."""
    if name == "algName":
        return False
    if name.startswith("AlgName_NormalizeInput"):
        return False
    if name.startswith("AlgName_Output_"):
        return False
    for group in ("Forward", "Backward"):
        base = f"AlgName_{group}_"
        if name.startswith(base):
            keep = (
                name.endswith("_ActivationClocks")
                or name.startswith(f"{base}MaxSaturation")
                or name.startswith(f"{base}IsCellsPeepholesActive")
                or name.startswith(f"{base}IsGatesPeepholesActive")
                or name.startswith(f"{base}IsGatesRecurrentPeepholesActive")
            )
            if not keep:
                return False
    if name.startswith("AlgName_LID_NormalizeInput"):
        return False
    if name.startswith("AlgName_LID_Output_"):
        return False
    for group in ("Forward", "Backward"):
        base = f"AlgName_LID_{group}_"
        if name.startswith(base):
            keep = (
                name.endswith("_ActivationClocks")
                or name.startswith(f"{base}MaxSaturation")
                or name.startswith(f"{base}IsCellsPeepholesActive")
                or name.startswith(f"{base}IsGatesPeepholesActive")
                or name.startswith(f"{base}IsGatesRecurrentPeepholesActive")
            )
            if not keep:
                return False
    return True


def _fmt_scalar(v: object) -> str:
    if isinstance(v, str):
        return v
    fv = float(v)  # type: ignore[arg-type]
    if fv.is_integer():
        return f"{int(fv):d}"
    return f"{fv:15.15e}"


def _fmt_value(v: object) -> str:
    """printConfig numeric formatting: %d if integer-valued else %15.15e, per element for
    vectors (length > 1) joined by ','."""
    if isinstance(v, str):
        return v
    arr = np.atleast_1d(np.asarray(v, dtype=np.float64)).reshape(-1)
    if arr.size > 1:
        return ",".join(_fmt_scalar(float(x)) for x in arr)
    return _fmt_scalar(float(arr[0]))


def _serialize(cfg: dict[str, object], algname: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in cfg.items():
        if not _printconfig_written(key):
            continue
        out_key = key.replace("AlgName_", algname, 1) if key.startswith("AlgName_") else key
        out[out_key] = _fmt_value(value)
    return out


def vec2struct(
    param: NDArray[np.float64],
    mask: dict[str, object] | None,
    ps: RunConfig,
    mode: int,
) -> tuple[dict[str, str], NDArray[np.float64], int]:
    """Map the genome `param` to (configStruct, out_param, count_param).

    configStruct is the flat legacy KEY->value STRING dict printConfig would write (weights
    excluded; keys renamed AlgName_->algo tag; %d/%15.15e formatting). out_param is param with
    the mask inverse / sortrows / clamp write-backs. `mode` is accepted for legacy call-site
    parity (in the legacy it only gates the file write); the dict is always returned.
    """
    _ = mode
    w = _Walk(np.asarray(param, dtype=np.float64), mask, ps)
    w.run()
    assert w.out is not None
    return _serialize(w.cfg, w.algname), w.out, w.count


def genome_length(ps: RunConfig) -> int:
    """count_param -- the number of genome coefficients + 1 (the 1-based cursor's final
    value, matching what vec2struct returns). A pure function of the PS spec: run the walk
    over a zero genome (mask-independent)."""
    w = _Walk(None, None, ps)
    w.run()
    return w.count


# ---- Phase 5 Task 9: the non-weight (DSP/config hyperparameter) genome ------------------
#
# The modern regime (user-locked, 2026-07-10): network WEIGHTS train by gradient (the modern
# SMORMS3 loop), so the outer QuantumPSO search must NOT carry them -- it searches DSP/config
# hyperparameters ONLY. `weight_block_mask` carves the weight/normalize dims out of the genome
# by DERIVING them from the walk itself (no hardcoded dim counts), so the carve-out tracks the
# net architecture automatically. This is a DELIBERATE, documented break from the legacy genome
# (which carried the weights in-band by design) -- see IMPROVEMENTS.md `phase5-qpso-nonweight-genome`.


class _MaskTraceWalk(_Walk):
    """A count-only walk (`param is None`) that records the genome dim ranges consumed by the
    weight/normalize arms -- the `_nn_block` / `_output_neuron` matrices and the NormalizeInput
    Mean/Std tail, all of which live inside `_lstm_and_output`. The count advances are param-
    independent (they depend only on the net sizes), so the zero-arg walk traces the exact same
    ranges any real genome would consume. Used by `weight_block_mask` to build the searchable-dim
    selector structurally."""

    def __init__(self, ps: RunConfig) -> None:
        super().__init__(None, None, ps)
        self.masked_ranges: list[tuple[int, int, str]] = []  # (start0, stop0, field_key)

    def _nn_block(self, name: str, nb_need: int, fir_ref: str | None, fs_ref: str | None) -> None:
        start = self.count - 1
        super()._nn_block(name, nb_need, fir_ref, fs_ref)
        self.masked_ranges.append((start, self.count - 1, name))

    def _output_neuron(self, name: str, nb: int, ii: int, out_net: list[int], out_sub: list[int]) -> None:
        start = self.count - 1
        super()._output_neuron(name, nb, ii, out_net, out_sub)
        self.masked_ranges.append((start, self.count - 1, name))

    def _lstm_and_output(self, pfx: str, nntype: int, lstm: list[int], lsub: list[int], out_net: list[int], out_sub: list[int]) -> None:
        super()._lstm_and_output(pfx, nntype, lstm, lsub, out_net, out_sub)
        # The `_nn_block`/`_output_neuron` sub-ranges are recorded above; the NormalizeInputMean/Std
        # tail is the LAST 2*inputSize dims of this call (mean then std, each == NNetInputSize), the
        # only weight/normalize consumption not routed through the two overridden primitives.
        nb = out_net[0] if lstm[0] == 0 else lstm[0]  # == NNetInputSize, mirroring _lstm_and_output
        stop = self.count - 1
        self.masked_ranges.append((stop - 2 * nb, stop - nb, f"{pfx}_NormalizeInputMean"))
        self.masked_ranges.append((stop - nb, stop, f"{pfx}_NormalizeInputStd"))


def weight_block_mask(ps: RunConfig) -> tuple[dict[str, object], NDArray[np.bool_]]:
    """The permanent weight/normalize mask + the searchable-dim selector, both derived from the
    vec2struct walk (NO hardcoded dim counts). Returns `(mask, searchable)`:

      * `mask` -- a vec2struct field-name mask pinning every weight-block / normalize-tail FIELD
        (exactly the cfg keys `printConfig` drops: the `_nn_block`/`_output_neuron` matrices +
        NormalizeInputMean/Std) to its zero-genome-decoded value. Passed as vec2struct's `mask`
        arg it forces those fields regardless of the (placeholder) weight dims, so the outer search
        never touches a network weight. The pinned VALUES are natural decode outputs, so the mask
        round-trips (`masking_validation` passes); they are irrelevant to the engine anyway -- the
        weights reach it through the committed `.bin` pack, and printConfig never writes these keys.
      * `searchable` -- a bool array over the genome's coefficients (length `genome_length-1`), True
        where the dim feeds a searchable DSP/config hyperparameter (freq bands, windows, LTSV/TDC,
        decision thresholds, calibration laws, ponderations), False where it feeds a masked
        weight/normalize field.

    The `mask` key set is `{k for k in cfg if not _printconfig_written(k)}` -- printConfig's OWN
    weight/normalize classification -- and the `searchable` False-set is the union of the traced
    `_MaskTraceWalk` ranges; the two agree by construction (every dropped key is a dim-consuming
    weight/normalize arm, and every such arm's key is dropped), which the T9 tests cross-check."""
    total = genome_length(ps) - 1
    w0 = _Walk(np.zeros(total, dtype=np.float64), None, ps)
    w0.run()
    mask: dict[str, object] = {k: v for k, v in w0.cfg.items() if not _printconfig_written(k)}

    tw = _MaskTraceWalk(ps)
    tw.run()
    searchable = np.ones(total, dtype=bool)
    for start, stop, _key in tw.masked_ranges:
        searchable[start:stop] = False
    return mask, searchable
