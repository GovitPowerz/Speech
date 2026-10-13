//! `Segmenter` decision logic (hysteresis-with-area) + the fixed smoothing pipeline.
//!
//! Ported from legacy C++ `Segmenter.cpp`: `buildFromConf` (:73-148),
//! `updateSegmentation` (:725-845), `smoothSegmentation` (:709-723),
//! `results2segmentation` (:1113-1126). Pure logic, validated by hand-computed unit
//! tests plus the harness `SegProbe` goldens (`phase2b_wiring_golden.rs`).

use anyhow::{Result, anyhow};
use indexmap::IndexMap;

use crate::audio::{Audio, convolution_horiz_slice, windowing_coefficients};
use crate::legacy_config::{get_f64, get_f64_default, get_f64_list};
use crate::nn::blstm::BlstmNetwork;

use super::lid::LidChannel;
use super::segmentation::{SegClass, Segmentation};

/// Produces a [`Segmentation`](super::segmentation_io) from features/audio.
///
/// `get_segmentation` mirrors the legacy pure-virtual `Segmenter::getSegmentation`
/// (`Segmenter.h:25`): `seg_per_chan` is pre-sized by the caller, one entry per
/// audio channel.
///
/// The trait is the driver interface (issue #25): the bag reaches every driver
/// through it. Two weight surfaces: the PACK surface ([`Self::pack_count`],
/// [`Self::set_packs`]) spans both trees, the TRAINING surface ([`Self::nets`],
/// [`Self::nets_mut`]) is the exact tree's f64 nets only. Their defaults are the
/// legacy base-class no-op bodies (`Segmenter::setWeights`/`getWeights`/
/// `getWeightsDerivatives`, `Segmenter.cpp:107-115`): a plain (non-BLSTM) segmenter
/// has no trainable weights, so the defaults are the correct behaviour, not a stub.
///
/// `refs` is the per-channel REFERENCE segmentation (the legacy `seg._Reference`,
/// which the Rust [`Segmentation`] does NOT carry internally): `Some(&[Segmentation])`
/// with one entry per channel when a reference was loaded, `None` otherwise. The two
/// NN drivers (spectral/signal) build per-frame training targets from it
/// (`Segmenter::getTargets`, gated on `seg._Reference.size() > 0`); the non-NN
/// drivers (TDC/LTSV) ignore it. With `refs = None` the drivers run the no-target
/// forward path, byte-identical to the pre-target behaviour.
pub trait Segmenter {
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        refs: Option<&[Segmentation]>,
    ) -> Result<()>;

    /// The driver's `_DumpDir`: gates the bag's scored-branch VRCTS write.
    fn dump_dir(&self) -> &str;

    /// How many weight packs the driver exchanges through [`Self::set_packs`]: 2 for
    /// the Twin (`[sad, lid]`, exact or fast), 1 for a single-net driver (exact or
    /// fast), 0 for a driver with no NN (algo 0/1/2).
    fn pack_count(&self) -> usize {
        0
    }

    /// Inject one pack per net, in [`Self::pack_count`] order. The bag sizes the list
    /// first (issue #60), so a driver indexes it freely. Algo 0/1/2 keep the legacy
    /// no-op (no `else` branch in `BagOfProcessors::setWeights`).
    fn set_packs(&mut self, _packs: &[Vec<f64>]) -> Result<()> {
        Ok(())
    }

    /// The trainable f64 nets, in pack order (`[sad, lid]` for the Twin). Empty on the
    /// fast drivers (inference-only, training stays exact f64) and on algo 0/1/2.
    fn nets(&self) -> Vec<&BlstmNetwork> {
        Vec::new()
    }

    /// [`Self::nets`], mutably.
    fn nets_mut(&mut self) -> Vec<&mut BlstmNetwork> {
        Vec::new()
    }

    /// Channel `chan`'s result members from the last `get_segmentation` call.
    fn channel_members(&self, chan: usize) -> ChannelMembers<'_>;

    /// Port-side observation point (no legacy counterpart): the per-channel
    /// PRE-convolution result rows of the last call. Empty for a driver with no posterior.
    fn last_result_rows(&self) -> &[Vec<f64>] {
        &[]
    }
}

/// One channel's driver-side result members, the bag's input to the channel result.
#[derive(Debug, Clone, Copy)]
pub struct ChannelMembers<'a> {
    /// `seg._CumulativeError[chan]` (result col 4): 0 on a driver with no SAD net
    /// harvest (algo 0/1/2/5, the fast drivers).
    pub cumulative_error: f64,
    /// `seg._NbOfClassif[chan]` (result col `len-1`).
    pub nb_of_classif: i64,
    /// The channel's LID members (still in-band encoded) on a LID driver, `None` on
    /// algo 0-4: the legacy `!seg._IsLIDCorrect.empty()` gate (`:338`/`:380`).
    pub lid: Option<LidChannel<'a>>,
}

/// Segmenter decision parameters, parsed from the legacy config (`buildFromConf`).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmenterConfig {
    pub rising: f64,
    pub area_rising: f64,
    pub falling: f64,
    pub area_falling: f64,
    pub padding: [f64; 4],
    pub min_speech: [f64; 3],
    pub min_silence: [f64; 2],
}

impl SegmenterConfig {
    /// Port of `Segmenter::buildFromConf` (:85-138): the `falling > rising` clamp,
    /// the length checks (`speech_padding` >= 4, `min_speech` >= 3, `min_silence`
    /// >= 2, else error), and the per-element negative -> 0 clamp.
    pub fn from_config(m: &IndexMap<String, String>, prefix: &str) -> Result<SegmenterConfig> {
        let rising = get_f64(m, &format!("{prefix}_decision_thresh_rising"))?;
        let area_rising = get_f64(m, &format!("{prefix}_decision_area_rising"))?;
        let mut falling = get_f64(m, &format!("{prefix}_decision_thresh_falling"))?;
        if falling > rising {
            falling = rising;
        }
        let area_falling = get_f64(m, &format!("{prefix}_decision_area_falling"))?;

        let padding_raw = get_f64_list(m, &format!("{prefix}_speech_padding"))?;
        if padding_raw.len() < 4 {
            return Err(anyhow!("{prefix}_speech_padding must contain 4 values"));
        }
        let min_speech_raw = get_f64_list(m, &format!("{prefix}_min_speech"))?;
        if min_speech_raw.len() < 3 {
            return Err(anyhow!("{prefix}_min_speech must contain 3 values"));
        }
        let min_silence_raw = get_f64_list(m, &format!("{prefix}_min_silence"))?;
        if min_silence_raw.len() < 2 {
            return Err(anyhow!("{prefix}_min_silence must contain 2 values"));
        }

        let clamp0 = |v: f64| if v < 0.0 { 0.0 } else { v };
        Ok(SegmenterConfig {
            rising,
            area_rising,
            falling,
            area_falling,
            padding: [
                clamp0(padding_raw[0]),
                clamp0(padding_raw[1]),
                clamp0(padding_raw[2]),
                clamp0(padding_raw[3]),
            ],
            min_speech: [
                clamp0(min_speech_raw[0]),
                clamp0(min_speech_raw[1]),
                clamp0(min_speech_raw[2]),
            ],
            min_silence: [clamp0(min_silence_raw[0]), clamp0(min_silence_raw[1])],
        })
    }
}

/// Read a scalar `usize` from a config value (legacy `conf.get<vector<double>::
/// size_type>(name)`, no default).
fn parse_usize(m: &IndexMap<String, String>, key: &str) -> Result<usize> {
    m.get(key)
        .ok_or_else(|| anyhow!("param '{key}' not found in config"))?
        .trim()
        .parse::<usize>()
        .map_err(|e| anyhow!("`{key}`: cannot parse: {e}"))
}

/// Read a string from a config value, with a default for a missing key (legacy
/// `conf.get<string>(name, default)`).
fn parse_string_default(m: &IndexMap<String, String>, key: &str, default: &str) -> String {
    m.get(key).cloned().unwrap_or_else(|| default.to_string())
}

/// The legacy warning text for an unsupported windowing type
/// (`Segmenter::verifyWindowingType`, `Segmenter.cpp:61-70`).
fn windowing_type_warning(parameter_name: &str) -> String {
    format!(
        "\n!!! WARNING !!!\nThe parameter given for {parameter_name} is not supported. The acceptable choices are \"uniform\", \"hamming\", \"hann\", \"hHCw\" or \"none\".\n{parameter_name}The parameter is set to none.\n"
    )
}

const VALID_WINDOWING_TYPES: [&str; 5] = ["hamming", "hann", "hHCw", "uniform", "none"];

/// Port of `Segmenter::verifyWindowingType` (`Segmenter.cpp:61-70`).
///
/// Legacy quirk reproduced FAITHFULLY: the legacy parameter is passed BY VALUE, so
/// the `windowing_type = "none"` reassignment inside the function is a local-only
/// mutation that is never written back to the caller's stored `_WindowingType`/
/// `_ConvolutionType` field -- an invalid type is logged but the invalid string is
/// still stored and used downstream. This port has nothing to mutate (it takes a
/// borrowed `&str`), which is the same "no fix" behaviour by construction: it
/// returns `Some(warning)` for the caller to log, and the caller's own stored type
/// string is left untouched either way.
pub fn verify_windowing_type(windowing_type: &str) -> Option<String> {
    if VALID_WINDOWING_TYPES.contains(&windowing_type) {
        None
    } else {
        Some(windowing_type_warning(windowing_type))
    }
}

/// The residual `Segmenter::buildFromConf` keys NOT already covered by
/// [`SegmenterConfig`] (decision/padding/min lists) or `FeatureConfig`
/// (windowing/preemph/noise): the dump directory, window/shift, the convolution
/// kernel, and the WER back-prop toggle.
#[derive(Debug, Clone, PartialEq)]
pub struct DriverConfig {
    pub dump_dir: String,
    pub window_size_sec: f64,
    pub window_shift_sec: f64,
    pub conv_coeff: Option<Vec<f64>>,
    pub back_prop_wer: f64,
}

impl DriverConfig {
    /// Port of the residual `Segmenter::buildFromConf` reads (`Segmenter.cpp:73-148`):
    /// `Dump_Directory` (GLOBAL key, default `""`); `{prefix}_window`/`{prefix}_shift`
    /// (required, no default -- `Err` on missing key, negative -> 0 clamp, :90-93);
    /// `{prefix}_convolution_window_size` (required `usize`; `> 0` -> read
    /// `{prefix}_convolution_window_type` + verify, else `"none"`, :100-108); the
    /// convolution coefficients (`windowing_coefficients(ty, true, 2*size+1,
    /// 0.83333)`, the legacy `getWindowingCoefficients` default extra param,
    /// `Helpers.hpp:222`); `{prefix}_BackPropWER` (default `-1.0`).
    pub fn from_config(m: &IndexMap<String, String>, prefix: &str) -> Result<DriverConfig> {
        let dump_dir = parse_string_default(m, "Dump_Directory", "");

        let mut window_size_sec = get_f64(m, &format!("{prefix}_window"))?;
        if window_size_sec < 0.0 {
            window_size_sec = 0.0;
        }
        let mut window_shift_sec = get_f64(m, &format!("{prefix}_shift"))?;
        if window_shift_sec < 0.0 {
            window_shift_sec = 0.0;
        }

        let conv_window_size = parse_usize(m, &format!("{prefix}_convolution_window_size"))?;
        let conv_type = if conv_window_size > 0 {
            let ty = m
                .get(&format!("{prefix}_convolution_window_type"))
                .ok_or_else(|| {
                    anyhow!("param '{prefix}_convolution_window_type' not found in config")
                })?
                .clone();
            // verifyWindowingType is called for its logging side effect only (the
            // by-value no-fix quirk): the stored type is used as-is either way.
            let _ = verify_windowing_type(&ty);
            ty
        } else {
            "none".to_string()
        };
        let conv_coeff =
            windowing_coefficients(&conv_type, true, 2 * conv_window_size + 1, 0.83333);

        let back_prop_wer = get_f64_default(m, &format!("{prefix}_BackPropWER"), -1.0)?;

        Ok(DriverConfig {
            dump_dir,
            window_size_sec,
            window_shift_sec,
            conv_coeff,
            back_prop_wer,
        })
    }
}

/// Port of `Segmenter::results2segmentation` (`Segmenter.cpp:1113-1126`).
///
/// `conv` is `Some(&coeffs)` with `len() > 1` -> convolves `results` IN PLACE via
/// [`convolution_horiz_slice`] (the legacy `_ConvolutionCoeff.cols() > 1` gate;
/// a `None`/single-tap kernel is a no-op, matching the legacy empty-matrix
/// `cols() == 0` case and any degenerate 1-tap kernel), THEN calls
/// [`update_segmentation`] (`updateSegmentation`, which itself runs
/// [`smooth_segmentation`]). `targets` is unused by the live `updateSegmentation`
/// body (every write site is commented out, `Segmenter.cpp:893-994`) and is
/// therefore dropped from this signature, with this doc note standing in for the
/// dead parameter.
pub fn results_to_segmentation(
    seg: &mut Segmentation,
    time_step: f64,
    time_offset: f64,
    results: &mut [f64],
    class: SegClass,
    conv: Option<&[f64]>,
    cfg: &SegmenterConfig,
) {
    if let Some(coeffs) = conv
        && coeffs.len() > 1
    {
        convolution_horiz_slice(results, coeffs);
    }
    update_segmentation(seg, results, class, time_offset, time_step, cfg);
}

/// The hysteresis-with-area decision as ONE STEP KERNEL (issue #26): the loop body of
/// the legacy `Segmenter::updateSegmentation` (:725-845) with every loop variable
/// (`begin/end/begin_area/end_area/has_begun/has_ended`, the running index `ii` and
/// `results[ii-1]`) persisted on the struct, fed one convolved posterior value at a
/// time.
///
/// Two drivers fold over it (ADR-0004, "one kernel, two drivers", extended to the decision
/// layer): the batch [`update_segmentation_raw`] feeds a whole row vector then
/// [`flush_tail`](Self::flush_tail); the streaming `fast::stream::StreamDecision` feeds the
/// convolved values as they finalize. Equality between the two holds by construction, so
/// the state-threading leg (`phase8_stream_decision.rs::raw_hysteresis_matches_offline`)
/// pins only that the stream hands the step the same values in the same order (ADR-0005).
///
/// [`push`](Self::push) runs one iteration: the INIT check for the first value (`begin`
/// and `begin_area` seeded WITHOUT `has_begun`, so the next value's rising branch keys off
/// `begin < 0` being false), the loop body for the rest, returning the raw segment
/// `(begin + off, end + off)` on the iteration that labels one (at most one per value).
/// [`flush_tail`](Self::flush_tail) is the tail force-emit: the legacy `dt *
/// results.size()`, which on a row vector is the count of values fed, `idx` (the
/// `results.size()` quirk, IMPROVEMENTS.md 0b-ii, is the same number here).
pub struct HystStep {
    t_r: f64,
    a_r: f64,
    t_f: f64,
    a_f: f64,
    off: f64,
    dt: f64,
    begin: f64,
    end: f64,
    begin_area: f64,
    end_area: f64,
    has_begun: bool,
    has_ended: bool,
    idx: usize, // number of values processed so far (== the batch loop's `ii`)
    prev: f64,  // the previous value (the batch `results[ii-1]`)
}

impl HystStep {
    pub fn new(cfg: &SegmenterConfig, off: f64, dt: f64) -> HystStep {
        HystStep {
            t_r: cfg.rising,
            a_r: cfg.area_rising,
            t_f: cfg.falling,
            a_f: cfg.area_falling,
            off,
            dt,
            begin: -1.0,
            end: -1.0,
            begin_area: -1.0,
            end_area: -1.0,
            has_begun: false,
            has_ended: false,
            idx: 0,
            prev: 0.0,
        }
    }

    /// Feed one convolved posterior value. Returns the raw segment labeled on this
    /// step, if any (at most one per value, exactly as the legacy loop labels at most once
    /// per `ii`).
    pub fn push(&mut self, r: f64) -> Option<(f64, f64)> {
        if self.idx == 0 {
            // INIT: begin/begin_area seed, WITHOUT has_begun; the ii>=1 body's rising
            // branch then keys off `begin < 0` being false.
            if r >= self.t_r {
                self.begin = 0.0;
                self.begin_area = 0.0;
            }
            self.prev = r;
            self.idx = 1;
            return None;
        }

        let ii = self.idx as f64;
        let r_prev = self.prev;
        let (t_r, a_r, t_f, a_f, dt) = (self.t_r, self.a_r, self.t_f, self.a_f, self.dt);
        let mut out = None;

        if !self.has_begun {
            if self.begin < 0.0 {
                if r >= t_r && r_prev < t_r {
                    self.begin = dt * (ii - (r - t_r) / (r - r_prev));
                    self.begin_area = (ii - self.begin / dt) * (r - t_r) / 2.0;
                    if self.begin_area >= a_r {
                        self.has_begun = true;
                    }
                }
            } else if r >= t_r {
                self.begin_area += (r + r_prev - t_r * 2.0) / 2.0;
                if self.begin_area >= a_r {
                    self.has_begun = true;
                }
            } else if r < t_r && r_prev >= t_r {
                self.begin_area += (r_prev - t_r) * (t_r - r_prev) / (r - r_prev) / 2.0;
                if self.begin_area >= a_r {
                    self.has_begun = true;
                } else {
                    self.begin = -1.0;
                    self.begin_area = -1.0;
                    self.has_begun = false;
                }
            } else {
                self.begin = -1.0;
                self.begin_area = -1.0;
                self.has_begun = false;
            }
        }

        if self.has_begun {
            if self.end < 0.0 {
                if r <= t_f && r_prev > t_f {
                    self.end = dt * (ii - (r - t_f) / (r - r_prev));
                    self.end_area = (ii - self.end / dt) * (t_f - r) / 2.0;
                    if self.end_area >= a_f {
                        self.has_ended = true;
                    }
                }
            } else if r <= t_f {
                self.end_area += (2.0 * t_f - r - r_prev) / 2.0;
                if self.end_area >= a_f {
                    self.has_ended = true;
                }
            } else if r > t_f && r_prev <= t_f {
                self.end_area += (t_f - r_prev) * (t_f - r_prev) / (r - r_prev) / 2.0;
                if self.end_area >= a_f {
                    self.has_ended = true;
                } else {
                    self.end = -1.0;
                    self.end_area = -1.0;
                    self.has_ended = false;
                }
            } else {
                self.end = -1.0;
                self.end_area = -1.0;
                self.has_ended = false;
            }

            if self.has_ended {
                if self.begin < self.end {
                    out = Some((self.begin + self.off, self.end + self.off));
                    self.begin = -1.0;
                    self.begin_area = -1.0;
                    self.has_begun = false;
                    self.end = -1.0;
                    self.end_area = -1.0;
                    self.has_ended = false;

                    // Re-run rising after label at the SAME ii.
                    if r >= t_r && r_prev < t_r {
                        self.begin = dt * (ii - (r - t_r) / (r - r_prev));
                        self.begin_area = (ii - self.begin / dt) * (r - t_r) / 2.0;
                        if self.begin_area >= a_r {
                            self.has_begun = true;
                        }
                    }
                } else {
                    self.begin = -1.0;
                    self.begin_area = -1.0;
                    self.has_begun = false;
                    self.end = -1.0;
                    self.end_area = -1.0;
                    self.has_ended = false;
                }
            }
        }

        self.prev = r;
        self.idx += 1;
        out
    }

    /// The audio-time BEGIN of a raw segment that is currently OPEN or TENTATIVELY open --
    /// i.e. a rising crossing has been recorded (`begin >= 0`) and not yet either closed
    /// into `raw_segments` or retracted by a failed area test. `None` when the hysteresis
    /// carries no candidate begin at all.
    ///
    /// THIS IS THE THIRD SOURCE OF FUTURE RAW STRUCTURE (Phase 9 Task 9 fix, found by the
    /// corpus streaming tier `tests/pyo3/test_phase9_parity.py`). A pending begin lies
    /// BEHIND the consumed frontier -- the hysteresis passed it and is still inside the
    /// segment -- so the streaming `StreamDecision::resmooth_and_emit`'s emission frontier
    /// must clamp to it; `max(last_raw_boundary, consumed_frontier - dt)` alone does NOT bound where the
    /// next raw segment begins. Both audio-time offsets match the closed-segment tuple
    /// (`begin + off`, `push`'s label site). The batch fold never reads it.
    pub fn pending_begin(&self) -> Option<f64> {
        if self.begin >= 0.0 {
            Some(self.begin + self.off)
        } else {
            None
        }
    }

    /// The tail force-emit: if a segment is still open at EOS, close it at
    /// `dt * results.len()` (== `dt * idx`, since every convolved value has been fed).
    /// EOS closes the open segment or drops a tentative one, so no candidate survives it:
    /// [`pending_begin`](Self::pending_begin) is `None` afterwards.
    pub fn flush_tail(&mut self) -> Option<(f64, f64)> {
        let tail = if self.has_begun {
            let end = self.dt * self.idx as f64;
            Some((self.begin + self.off, end + self.off))
        } else {
            None
        };
        self.begin = -1.0;
        self.begin_area = -1.0;
        self.has_begun = false;
        tail
    }
}

/// The RAW hysteresis-with-area decision over the ROW vector `results`, WITHOUT
/// smoothing. Returns the pre-smoothing, pre-sanitize labeled segments as
/// `(begin + off, end + off, class as i32)`, one entry per legacy
/// `label_segment` call, in emission order.
///
/// Direct port of the decision body of `Segmenter::updateSegmentation`
/// (:725-845): the INIT check, the rising/falling hysteresis with area gating,
/// the re-run-rising-after-label, and the `results.size()` tail. Since issue #26
/// it is the FOLD over [`HystStep`] (one `push` per value, then `flush_tail`),
/// the same kernel the streaming session drives. This isolates the decision math
/// so it can be cross-checked bit-for-bit against the numpy oracle
/// (`speech.scoring.update_segmentation_oracle`). `results[k]` is the
/// legacy `results(0, k)`; `dt` is `timeStep`, `off` is `timeOffset`. The TAIL
/// uses `results.len()` (the legacy `results.size()` = rows*cols, equal to
/// `length` only for a pure row vector - reproduced quirk, logged in
/// IMPROVEMENTS.md).
pub fn update_segmentation_raw(
    results: &[f64],
    class: SegClass,
    off: f64,
    dt: f64,
    cfg: &SegmenterConfig,
) -> Vec<(f64, f64, i32)> {
    let mut step = HystStep::new(cfg, off, dt);
    let mut segments: Vec<(f64, f64, i32)> = results
        .iter()
        .filter_map(|&r| step.push(r))
        .map(|(b, e)| (b, e, class as i32))
        .collect();
    if let Some((b, e)) = step.flush_tail() {
        segments.push((b, e, class as i32));
    }
    segments
}

/// Hysteresis-with-area decision over the ROW vector `results`, then smoothing.
///
/// Runs [`update_segmentation_raw`] (the exact legacy decision body) and replays
/// its labeled segments into `seg` via `label_segment` in emission order, then
/// applies the fixed smoothing pipeline. Behaviour-identical to the pre-refactor
/// monolithic port of `Segmenter::updateSegmentation` (:725-845).
pub fn update_segmentation(
    seg: &mut Segmentation,
    results: &[f64],
    class: SegClass,
    off: f64,
    dt: f64,
    cfg: &SegmenterConfig,
) {
    for (begin, end, code) in update_segmentation_raw(results, class, off, dt, cfg) {
        debug_assert_eq!(code, class as i32);
        seg.label_segment(begin, end, class);
    }
    smooth_segmentation(seg, cfg);
}

/// Single-threshold, COL-vector decision, then `sanitize` only (NO smoothing).
///
/// Direct port of `Segmenter::LID2Segmentation` (`Segmenter.cpp:999-1055`, the
/// active `threshMax` branch; the `threshMin` second pass is commented out in the
/// legacy and reproduced as such). `results[k]` is the legacy `results(k, 0)` -
/// the COL vector, `length = results.rows()`. This is the load-bearing
/// orientation asymmetry vs [`update_segmentation`], which reads the ROW vector
/// `results(0, k)`. Only `thresh_max` is used: no area gating, no hysteresis.
///
/// Rising: `r(ii) >= thresh_max && r(ii-1) < thresh_max` -> linear-interp
/// `begin`, `hasBegun` immediately (no area). Falling: `r(ii) <= thresh_max &&
/// r(ii-1) > thresh_max` -> linear-interp `end`. On `begin < end` label + reset
/// (NO re-run-rising-after-label, unlike `update_segmentation`); else reset. The
/// INIT `r(0) >= thresh_max` sets `begin = 0` but NOT `hasBegun`, and the loop's
/// rising is guarded by `begin < 0`, so a begin-at-0 that never re-crosses
/// produces no segment (nor a tail) - a legacy quirk reproduced verbatim. TAIL
/// uses `results.len()` (the legacy `results.size()` = rows*cols, equal to
/// `length` for a col vector).
pub fn lid_to_segmentation(
    seg: &mut Segmentation,
    results: &[f64],
    class: SegClass,
    off: f64,
    dt: f64,
    thresh_max: f64,
) {
    let mut begin = -1.0f64;
    let mut end = -1.0f64;
    let mut has_begun = false;
    let mut has_ended = false;
    let length = results.len();

    if length > 0 && results[0] >= thresh_max {
        begin = 0.0;
    }

    for ii in 1..length {
        let r = results[ii];
        let r_prev = results[ii - 1];

        if !has_begun && begin < 0.0 && r >= thresh_max && r_prev < thresh_max {
            begin = dt * (ii as f64 - (r - thresh_max) / (r - r_prev));
            has_begun = true;
        }

        if has_begun {
            if end < 0.0 && r <= thresh_max && r_prev > thresh_max {
                end = dt * (ii as f64 - (r - thresh_max) / (r - r_prev));
                has_ended = true;
            }
            if has_ended {
                if begin < end {
                    seg.label_segment(begin + off, end + off, class);
                }
                begin = -1.0;
                end = -1.0;
                has_begun = false;
                has_ended = false;
            }
        }
    }

    if has_begun {
        end = dt * results.len() as f64;
        seg.label_segment(begin + off, end + off, class);
    }

    seg.sanitize();
}

/// Per-frame training targets from a reference segmentation
/// (`Segmenter::getTargets`, `:659-707`). Returns `n_rows` targets (the legacy
/// `targets(ii, 0)`); the frame count is passed explicitly since the legacy sizes
/// `targets` externally to match `results.rows()` (there is no such matrix in the
/// Rust signature). `reference` maps to the legacy `seg._Reference.at(chan)` (a
/// boundary list ending with the `End` sentinel); `seg` (the hypothesis) is
/// carried only for signature parity with the spec and is unused here.
///
/// For each row `ii`, `t = ii*time_step + time_offset`; advance `it_ref` while
/// `it_ref+1` is not the sentinel-past-end AND `(it_ref+1).begin <= t`.
///
/// - `back_prop_wer >= 0`: SPEECH/SUBSTITUTION -> `1 - 0.1*time_step/dur_seg`
///   with `dur_seg = max(next-cur, time_step)` (only when `it_ref+1` exists);
///   EXCLUDED -> `-0.5`; INSERTION -> `0.1*time_step/dur_seg` (only when `it_ref+1`
///   exists); else the two neighbor-window branches (a following SPEECH/SUBST
///   window opening within `_BackPropWER`, or a preceding one still open), else
///   `time_step/100`.
/// - `back_prop_wer < 0`: `it_ref.ty == class || (class==SPEECH &&
///   it_ref.ty==SUBSTITUTION)` -> `1.0`; EXCLUDED -> `-0.5`; else `0.0`.
///
/// The default-initialized `targets(ii,0)` is `0.0` (Eigen zero-init); the two
/// `back_prop_wer >= 0` branches that guard on `it_ref+1 != end()` leave the entry
/// at `0.0` when the guard fails, which this port reproduces via a `0.0`-filled
/// buffer.
#[allow(clippy::too_many_arguments)]
pub fn get_targets(
    _seg: &Segmentation,
    reference: &Segmentation,
    time_step: f64,
    time_offset: f64,
    back_prop_wer: f64,
    class: SegClass,
    n_rows: usize,
) -> Vec<f64> {
    let mut targets = vec![0.0f64; n_rows];
    let refs = reference.segments();
    if refs.is_empty() {
        return targets;
    }
    let len = refs.len();
    let mut idx = 0usize;

    for (ii, target) in targets.iter_mut().enumerate() {
        let t = ii as f64 * time_step + time_offset;
        while idx + 1 < len && refs[idx + 1].begin <= t {
            idx += 1;
        }
        let ty = refs[idx].ty;

        if back_prop_wer >= 0.0 {
            if ty == SegClass::Speech || ty == SegClass::Substitution {
                if idx + 1 < len {
                    let mut dur_seg = refs[idx + 1].begin - refs[idx].begin;
                    if dur_seg < time_step {
                        dur_seg = time_step;
                    }
                    *target = 1.0 - 0.1 * time_step / dur_seg;
                }
            } else if ty == SegClass::Excluded {
                *target = -0.5;
            } else if ty == SegClass::Insertion {
                if idx + 1 < len {
                    let mut dur_seg = refs[idx + 1].begin - refs[idx].begin;
                    if dur_seg < time_step {
                        dur_seg = time_step;
                    }
                    *target = 0.1 * time_step / dur_seg;
                }
            } else if idx + 1 < len
                && idx + 2 < len
                && (refs[idx + 1].ty == SegClass::Speech
                    || refs[idx + 1].ty == SegClass::Substitution)
                && refs[idx + 1].begin - back_prop_wer <= t
            {
                let mut dur_seg = refs[idx + 2].begin - refs[idx + 1].begin;
                if dur_seg < time_step {
                    dur_seg = time_step;
                }
                *target = 1.0 - 0.1 * time_step / dur_seg;
            } else if idx != 0
                && (refs[idx - 1].ty == SegClass::Speech
                    || refs[idx - 1].ty == SegClass::Substitution)
                && refs[idx].begin + back_prop_wer >= t
            {
                let mut dur_seg = refs[idx].begin - refs[idx - 1].begin;
                if dur_seg < time_step {
                    dur_seg = time_step;
                }
                *target = 1.0 - 0.1 * time_step / dur_seg;
            } else {
                *target = time_step / 100.0;
            }
        } else if ty == class || (class == SegClass::Speech && ty == SegClass::Substitution) {
            *target = 1.0;
        } else if ty == SegClass::Excluded {
            *target = -0.5;
        } else {
            *target = 0.0;
        }
    }

    targets
}

/// The fixed 8-step smoothing pipeline (`Segmenter.cpp:709-723`).
pub fn smooth_segmentation(seg: &mut Segmentation, cfg: &SegmenterConfig) {
    seg.sanitize();
    seg.suppress_short(cfg.min_speech[0], SegClass::Speech);
    seg.add_padding(cfg.padding[0], cfg.padding[1], SegClass::Speech);
    seg.suppress_short(cfg.min_silence[0], SegClass::Other);
    seg.suppress_short(cfg.min_speech[1], SegClass::Speech);
    seg.add_padding(cfg.padding[2], cfg.padding[3], SegClass::Speech);
    seg.suppress_short(cfg.min_silence[1], SegClass::Other);
    seg.suppress_short(cfg.min_speech[2], SegClass::Speech);
}
