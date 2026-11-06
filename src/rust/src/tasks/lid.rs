//! Spectral LID driver (Algo 5; `BLSTMSpectralLID.{h,cpp}`) -- the FIRST LID driver.
//!
//! Ported from legacy C++: `BLSTMSpectralLID.cpp` (ctor `:10-14`, `getSegmentation`
//! `:27-460`). `BLSTMSpectralLID : BLSTMSpectralSegmenter : LongTermSpectralVariation
//! : Segmenter`; the ctor calls `BLSTMSpectralSegmenter::buildFromConf(conf, "BLSTM",
//! ...)` + a `BLSTMNeuralNetwork(conf, "BLSTM", false)`, so it reads the SAME `BLSTM_*`
//! config namespace as the spectral SAD driver (no `BLSTM_LID_*` keys).
//!
//! STRUCTURE (the crux, and why this is NOT the spectral SAD driver in disguise): the
//! LID `getSegmentation` inlines a COPY of the spectral setup (`:31-237`) that DIVERGES
//! from the base `BLSTMSpectralSegmenter::getSegmentation` in load-bearing ways -- the
//! Algo-5 lines govern where they differ (LEGACY SOURCE GOVERNS; diffed line by line):
//!
//! 1. **SAD via LTSV, not the BLSTM.** The result_vec is the LTSV spectral-variation
//!    score (`classifySequence` over the RAW periodogram, `:271-274`), NOT the NN
//!    posterior. The mel branch of that loop (`:252-269`) is COMMENTED OUT in the
//!    legacy, so the classify ALWAYS runs on `audio._Periodogram` (raw) with
//!    `freq_end` = the mel/DCT output width - 1. The BLSTM runs only PER SPEECH SEGMENT
//!    for language scoring (`:346-401`), never for the SAD result_vec.
//! 2. **LTSV window floors to 1, not 0** (`:157-158` `if (< 1) = 1`), UNLIKE the base
//!    `getLTSVParam` (`:303` `if (< 1) = 0`, disables). So LTSV SAD is ALWAYS active
//!    (even with `LTSVwindow 0`, `round(0) = 0 -> 1`). This is the STANDALONE
//!    `LtsvSegmenter`'s floor, not the spectral SAD's.
//! 3. **result_vec sized by `LTSV_window_shift`, not the ssr-division** (`:227-232`
//!    `ceil(vec_size / LTSV_window_shift)`), matching the LTSV driver -- NOT
//!    `getBLSTMParam`'s ssr-division (`:481-497`). The LTSV loop writes DECIMATED
//!    indices `result_vec[jj / LTSV_window_shift]` (NO interpolation backfill).
//! 4. **SAD `results2segmentation` timeStep = `_WindowShift`, offset 0.0** (`:302`) --
//!    the BLSTM-derived window shift used as the LTSV-SAD time step (a compression
//!    quirk: the result_vec is LTSV-decimated but stepped by `_WindowShift`, not
//!    `_WindowShift * LTSV_window_shift`). Reproduced verbatim.
//! 5. **scoring timeStep uses the SIGNAL pattern** (`:333-343`): the overlap branch
//!    keeps `timeStep = _WindowShift` (like `BlstmSignalSegmenter`), NOT
//!    `_SpectrumShift * ssr` (the base spectral driver's `:731`).
//! 6. **TDC params (`:162-173`) and `LTSV_freq_beg/end` (`:109-110`) are DEAD** --
//!    computed but never used (no pitch pass in the LID driver; the LTSV loop uses the
//!    post-mel `freq_beg/end`). Skipped here.
//! 7. **min_lag/max_lag derivation (`:170-172`) is SIMPLER than `getTDCParam`** (no
//!    `min_lag < 1 -> 1`, no `max_lag < min_lag` guard, no `_MinMaxLag` writeback) --
//!    moot since TDC is dead.
//!
//! The per-segment scoring loop (`:346-401`) drives [`BlstmNetwork::feed_forward_scoring`]
//! (the returning `feedForward` overload, `:363`): `segLID = colwise sum / rows`, the
//! `costLID`/`langID`/confusion accumulation, and the member writes (`:412-426`:
//! `targetLID(target) = -2.0`; `_LIDClassificationErrors = 100*(langID - targetLID)`;
//! `_IsLIDCorrect` 100/0/-1). `costLID` (`:320,376,405,410`) is computed but NEVER
//! stored (dead local) -- not reproduced. The `_CepstreCoefficients` writeback of the
//! normalized input (`:364-368`) is DEAD (segments read disjoint row blocks) -- skipped,
//! see IMPROVEMENTS.md.

use anyhow::{Result, anyhow};
use indexmap::IndexMap;
use ndarray::{Array2, ArrayView2};

use crate::audio::{Audio, compute_segment_periodogram_estimates, windowing_coefficients};
use crate::constants::random_gauss;
use crate::features::ltsv_tdc::ltsv_classify_sequence;
use crate::features::mel::MelFilterBank;
use crate::features::pipeline::{FeatureConfig, SpectralParams, build_input_sequence_parts};
use crate::legacy_config::{get_default, get_f64, get_f64_default};
use crate::nn::activations::inv_logistic_fn;
use crate::nn::blstm::{BlstmConfig, BlstmNetwork};
use crate::tasks::sad::{SpectralAxis, SpectralCarry, apply_input_gates, non_wav_ssif};
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::{compute_errors, speech_duration};
use crate::tasks::segmenter::{
    ChannelMembers, DriverConfig, LidChannel, Segmenter, SegmenterConfig, get_targets,
    lid_to_segmentation, results_to_segmentation, smooth_segmentation,
};

/// The per-channel LID members a LID driver (algo 5, the Twin, the fast Twin) writes on
/// each `get_segmentation` call, one entry per channel. The legacy writes them onto
/// `seg._LID*[chan]` (algo 5 `BLSTMSpectralLID.cpp:414-425`, the Twin
/// `TwinBLSTMSpectralLID.cpp:1320-1331`); the Rust `Segmentation` is single-channel (S6
/// restructure), so they live on the driver like the SAD driver's
/// `cumulative_error`/`nb_of_classif`.
#[derive(Clone, Debug, Default)]
pub struct LidMembers {
    /// `seg._LIDCumulativeError[chan]` (algo 5 `:414`, Twin `:1320`): the LID NN cost
    /// summed over speech segments; the Twin then scales it `*= audio._Weight` (`:1336`).
    pub cumulative_error: Vec<f64>,
    /// `seg._LIDNbOfClassif[chan]` (algo 5 `:415`, Twin `:1321`).
    pub nb_of_classif: Vec<i64>,
    /// `seg._LIDClassificationErrors[chan]` (algo 5 `:416`, Twin `:1322`): the `1 x classNb` row
    /// `100 * (langID - targetLID)`, with `targetLID[targetIndex] = -2.0` -- so the
    /// target entry carries the `>150` in-band sentinel (`100*langID + 200 >= 200`).
    pub classification_errors: Vec<Vec<f64>>,
    /// `seg._LIDSegmentsConfusion[chan]` (algo 5 `:418`, Twin `:1324`): the `(classNb+2) x (classNb+2)` per-file
    /// confusion matrix (label header row/col 0, per-class 1..classNb, totals classNb+1).
    pub segments_confusion: Vec<Array2<f64>>,
    /// `seg._IsLIDCorrect[chan]` (algo 5 `:420-425`, Twin `:1325-1331`): 100 (target is argmax) / 0 (not) / -1 (dead:
    /// `targetIndex` is clamped `>= 0`, so the -1 branch never fires).
    pub is_correct: Vec<i32>,
}

impl LidMembers {
    /// The per-call reset every LID `get_segmentation` starts with: zero cost and count,
    /// an empty score row, a `0 x 0` confusion and `0` per channel.
    pub fn zeroed(channels: usize) -> Self {
        LidMembers {
            cumulative_error: vec![0.0; channels],
            nb_of_classif: vec![0; channels],
            classification_errors: vec![Vec::new(); channels],
            segments_confusion: vec![Array2::<f64>::zeros((0, 0)); channels],
            is_correct: vec![0; channels],
        }
    }

    /// Channel `chan`'s members.
    pub fn channel(&self, chan: usize) -> LidChannel<'_> {
        LidChannel {
            cumulative_error: self.cumulative_error[chan],
            nb_of_classif: self.nb_of_classif[chan],
            is_correct: self.is_correct[chan],
            classification_errors: &self.classification_errors[chan],
        }
    }
}

// ===========================================================================
// The per-channel LID scoring accumulator -- ONE kernel (issue #83)
// ===========================================================================
//
// The legacy writes the per-channel LID scoring loop four times on the exact tree (algo 5
// `BLSTMSpectralLID.cpp:346-426`; the Twin's modes 0-3 `TwinBLSTMSpectralLID.cpp:1240-1337`,
// modes 4-6 `:798-902` + `:1318-1337`, mode 7 `:980-1179` + `:1318-1337`) and the fast Twin
// carried a fifth in f32-widened-to-f64. Token-identical across them: the confusion header
// seed, the strict `>` argmax with the six-line confusion update, the `segmentsCount`
// increment, the member-write block. What varies is three parameters, never a reorder:
// the per-block score ([`BlockScore`]), the `langID` weighting ([`BlockWeight`]) and the
// post-loop normalization ([`LidNorm`]). Each driver folds its loop through
// [`LidChannelAcc::push`] and reads the members off [`LidChannelAcc::finalize`]; the fast
// Twin and the utterance stream (`fast/driver.rs`, `fast/stream_lid.rs`) drive the SAME
// struct on the widened posteriors (ADR-0002's stepwise-hoist touch class, the #26 / #121
// direction: the fast tree imports the exact kernel). Sums stay f64 and ascending; the
// values are those of the copies, the goldens byte-green the proof.

/// How a scored block's posteriors (`rows x out_cols`) become its `segLID` score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockScore {
    /// `segLID = colwise sum / rows` over the first `class_nb` columns (algo 5
    /// `BLSTMSpectralLID.cpp:372`, Twin modes 0-3 `TwinBLSTMSpectralLID.cpp:1254`). Reads
    /// `class_nb` columns as the copies did, so `out.ncols() >= class_nb` is the contract:
    /// both callers score through `feed_forward_scoring` with the clamped `target_index >= 0`,
    /// which expands a width-1 net to `[1-p, p]`, and skip a block shorter than `ssr` (the
    /// one `(1, output_size)` zero return).
    ColumnMean,
    /// The three-arm `_PostProcessMode` accumulator over every row with `out[kk, 1] >= 0`
    /// (Twin mode 7 `:1073-1108`, modes 4-6 `:823-859`; the mode-7 `short_result_vec > 0.5`
    /// gate is `Ones(...)` there, always true): mode `1` entropy-normalized log posteriors,
    /// `2` a per-row argmax vote, else the plain `ln(max(1e-24, p))`. `out_cols` wide; each
    /// counted row increments the accumulator's `numberOfFrames`.
    PostProcess(i32),
}

/// How a block's `segLID` is added into the channel's `langID`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockWeight {
    /// `langID += segLID` (algo 5 `:391`).
    Plain,
    /// `langID += isCostModified ? segLID / rows : segLID` (Twin mode 7 `:1131-1135`, modes
    /// 4-6 `:875-879`), `rows` the scored block's own row count.
    MeanIfCostModified,
    /// `langID += isCostModified ? segLID : dur * segLID` (Twin modes 0-3 `:1275-1279`), `dur`
    /// the scoring segment's duration in seconds.
    DurationUnlessCostModified { dur: f64 },
}

/// The channel's post-loop `langID` normalization. Each variant carries its own guard
/// LITERALLY: the copies differ there (algo 5 and modes 0-3 do not test the sum) and the
/// hoist is behaviour-free; a unification would be an IMPROVEMENTS.md entry under the fix
/// protocol.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LidNorm {
    /// `segmentsCount > 0 ? langID /= segmentsCount : langID(0) = 1` (algo 5 `:403-411`).
    MeanOverSegments,
    /// Twin modes 7 / 4-6 (`:1163-1179`, `:887-902`): under `segmentsCount > 0 &&
    /// langID.sum() != 0`, divide by `segmentsCount` (cost-modified) or `numberOfFrames`,
    /// `exp` each entry when `exp` (`_PostProcessMode != 2`), then divide by the sum; else
    /// `langID(0) = 1`.
    PostProcess { exp: bool },
    /// Twin modes 0-3 (`:1299-1305`): under `segmentsCount > 0`, divide by `segmentsCount`
    /// (cost-modified) or `total_speech_duration`; else `langID(0) = 1`.
    Duration { total_speech_duration: f64 },
}

impl LidNorm {
    /// The `_PostProcessMode` normalization (Twin modes 7 / 4-6): `exp` unless the vote mode
    /// (`if (_PostProcessMode != 2)`, `:1169`, `:893`).
    pub fn post_process(mode: i32) -> LidNorm {
        LidNorm::PostProcess { exp: mode != 2 }
    }
}

/// One scored block's per-entry observables: its `segLID` (post-[`BlockScore`]) and its
/// strict-`>` argmax `j`. The utterance stream reports them per push.
#[derive(Clone, Debug, PartialEq)]
pub struct LidEntryOutcome {
    pub seg_lid: Vec<f64>,
    pub argmax: usize,
}

/// The finalized per-channel LID members: what every LID driver writes into its
/// [`LidMembers`] for one channel, and what the utterance stream returns as its running
/// aggregate. `classification_errors` / `confusion` / `is_lid_correct` are the
/// `classification_errors` / `segments_confusion` / `is_correct` members;
/// `predicted_language` / `segments_count` are the streaming-useful extras.
#[derive(Debug, Clone, PartialEq)]
pub struct LidAggregate {
    /// In-band `100*(langID - targetLID)` per class (the `.scr`/confusion column source) --
    /// normalized langID, the `LidMembers::classification_errors` row.
    pub classification_errors: Vec<f64>,
    /// The per-block argmax-count confusion matrix accumulated so far
    /// (`LidMembers::segments_confusion`; NOT normalized by the epilogue).
    pub confusion: Array2<f64>,
    /// `100` iff the normalized langID argmax is the target, else `0`
    /// (`LidMembers::is_correct`).
    pub is_lid_correct: i32,
    /// The predicted language: argmax of the normalized langID (first-max).
    pub predicted_language: usize,
    /// Scored (not skipped) blocks folded so far -- the legacy `segmentsCount`.
    pub segments_count: i32,
}

/// `targetIndex = clamp(lang_index, [0, classNb))` (algo 5 `BLSTMSpectralLID.cpp:321-323`,
/// the Twin `TwinBLSTMSpectralLID.cpp:626-628`): an out-of-range or negative reference
/// language index scores as class 0. The one body every LID driver and the fast plan read.
pub fn clamp_target_index(lang_index: i32, class_nb: usize) -> usize {
    if lang_index >= class_nb as i32 || lang_index < 0 {
        0
    } else {
        lang_index as usize
    }
}

/// The per-channel LID scoring accumulator. [`new`](Self::new) seeds the confusion index
/// header exactly as the copies do (algo 5 `:314-318`, Twin `:787-790`); [`push`](Self::push)
/// folds one scored block; [`finalize`](Self::finalize) is read-only, so the offline drivers
/// call it once and the utterance stream after every push.
#[derive(Clone, Debug)]
pub struct LidChannelAcc {
    langid: Vec<f64>,
    confusion: Array2<f64>,
    segments_count: i32,
    number_of_frames: i32,
    class_nb: usize,
    ti: usize,
    cost_modified: bool,
}

/// First-max argmax (strict `>`): the inline per-block argmax of every copy
/// (`TwinBLSTMSpectralLID.cpp:1113-1114`) and the `langID.maxCoeff()` pick (`:1325`).
fn argmax_f64(v: &[f64]) -> usize {
    let mut j = 0usize;
    let mut best = v[0];
    for (c, &x) in v.iter().enumerate().skip(1) {
        if x > best {
            best = x;
            j = c;
        }
    }
    j
}

impl LidChannelAcc {
    /// `class_nb = max(2, the LID output width)`; `ti` the clamped target index
    /// ([`clamp_target_index`]); `cost_modified` the LID net's cost law.
    pub fn new(class_nb: usize, ti: usize, cost_modified: bool) -> LidChannelAcc {
        let mut confusion = Array2::<f64>::zeros((class_nb + 2, class_nb + 2));
        for kk in 0..class_nb + 1 {
            confusion[[0, kk]] = kk as f64;
            confusion[[kk, 0]] = kk as f64;
        }
        LidChannelAcc {
            langid: vec![0.0f64; class_nb],
            confusion,
            segments_count: 0,
            number_of_frames: 0,
            class_nb,
            ti,
            cost_modified,
        }
    }

    /// Scored (not skipped) blocks folded so far -- the legacy `segmentsCount`.
    pub fn segments_count(&self) -> i32 {
        self.segments_count
    }

    /// Fold ONE scored block: its `segLID` per `score`, the strict `>` argmax `j`, the
    /// confusion update (`j == ti` on the diagonal, else `(ti, j)`, both marginals), then
    /// `langID` per `weight` over `seg_lid.len() <= class_nb` entries, and `segmentsCount++`.
    pub fn push(
        &mut self,
        out: ArrayView2<f64>,
        score: BlockScore,
        weight: BlockWeight,
    ) -> LidEntryOutcome {
        let rows = out.nrows();
        let seg_lid = match score {
            BlockScore::ColumnMean => {
                let out_rows = rows as f64;
                let mut seg_lid = vec![0.0f64; self.class_nb];
                for (c, sv) in seg_lid.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for r in 0..rows {
                        acc += out[[r, c]];
                    }
                    *sv = acc / out_rows;
                }
                seg_lid
            }
            BlockScore::PostProcess(mode) => {
                let out_cols = out.ncols();
                let mut seg_lid = vec![0.0f64; out_cols];
                match mode {
                    1 => {
                        for kk in 0..rows {
                            if out[[kk, 1]] >= 0.0 {
                                let mut entropy = 0.0;
                                for ll in 0..out_cols {
                                    let lo = f64::max(1e-24, out[[kk, ll]]).ln();
                                    entropy -= out[[kk, ll]] * lo;
                                }
                                entropy /= 2.0f64.ln();
                                if entropy < 1e-24 {
                                    entropy = 1e-24;
                                }
                                for (ll, sl) in seg_lid.iter_mut().enumerate() {
                                    *sl += f64::max(1e-24, out[[kk, ll]] / entropy).ln();
                                }
                                self.number_of_frames += 1;
                            }
                        }
                    }
                    2 => {
                        for kk in 0..rows {
                            if out[[kk, 1]] >= 0.0 {
                                let mut jm = 0usize;
                                let mut best = out[[kk, 0]];
                                for c in 1..out_cols {
                                    if out[[kk, c]] > best {
                                        best = out[[kk, c]];
                                        jm = c;
                                    }
                                }
                                seg_lid[jm] += 1.0;
                                self.number_of_frames += 1;
                            }
                        }
                    }
                    _ => {
                        for kk in 0..rows {
                            if out[[kk, 1]] >= 0.0 {
                                for (ll, sl) in seg_lid.iter_mut().enumerate() {
                                    *sl += f64::max(1e-24, out[[kk, ll]]).ln();
                                }
                                self.number_of_frames += 1;
                            }
                        }
                    }
                }
                seg_lid
            }
        };

        let j = argmax_f64(&seg_lid);
        let pos_target = self.ti + 1;
        if j == self.ti {
            self.confusion[[pos_target, pos_target]] += 1.0;
            self.confusion[[pos_target, self.class_nb + 1]] += 1.0;
            self.confusion[[self.class_nb + 1, pos_target]] += 1.0;
        } else {
            let pos_best = j + 1;
            self.confusion[[pos_target, pos_best]] += 1.0;
            self.confusion[[pos_target, self.class_nb + 1]] += 1.0;
            self.confusion[[self.class_nb + 1, pos_best]] += 1.0;
        }

        match weight {
            BlockWeight::Plain => {
                for (c, &v) in seg_lid.iter().enumerate() {
                    self.langid[c] += v;
                }
            }
            BlockWeight::MeanIfCostModified => {
                if self.cost_modified {
                    let rows = rows as f64;
                    for (c, &v) in seg_lid.iter().enumerate() {
                        self.langid[c] += v / rows;
                    }
                } else {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        self.langid[c] += v;
                    }
                }
            }
            BlockWeight::DurationUnlessCostModified { dur } => {
                if self.cost_modified {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        self.langid[c] += v;
                    }
                } else {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        self.langid[c] += dur * v;
                    }
                }
            }
        }

        self.segments_count += 1;
        LidEntryOutcome { seg_lid, argmax: j }
    }

    /// The finalized members: normalize a CLONE of `langID` per `norm` (the copies mutated
    /// in place and never read it afterwards; the values are identical, and the stream's
    /// mid-run snapshot must not consume the accumulator), then `targetLID(ti) = -2`,
    /// `100*(langID - targetLID)`, `is_correct = 100 iff langID(ti) == max` and the first-max
    /// predicted language.
    pub fn finalize(&self, norm: LidNorm) -> LidAggregate {
        let mut langid = self.langid.clone();
        match norm {
            LidNorm::MeanOverSegments => {
                if self.segments_count > 0 {
                    for v in langid.iter_mut() {
                        *v /= self.segments_count as f64;
                    }
                } else {
                    langid[0] = 1.0;
                }
            }
            LidNorm::PostProcess { exp } => {
                if self.segments_count > 0 && langid.iter().sum::<f64>() != 0.0 {
                    if self.cost_modified {
                        for v in langid.iter_mut() {
                            *v /= self.segments_count as f64;
                        }
                    } else {
                        for v in langid.iter_mut() {
                            *v /= self.number_of_frames as f64;
                        }
                    }
                    if exp {
                        for v in langid.iter_mut() {
                            *v = v.exp();
                        }
                    }
                    let adim: f64 = langid.iter().sum();
                    for v in langid.iter_mut() {
                        *v /= adim;
                    }
                } else {
                    langid[0] = 1.0;
                }
            }
            LidNorm::Duration {
                total_speech_duration,
            } => {
                if self.segments_count > 0 {
                    if self.cost_modified {
                        for v in langid.iter_mut() {
                            *v /= self.segments_count as f64;
                        }
                    } else {
                        for v in langid.iter_mut() {
                            *v /= total_speech_duration;
                        }
                    }
                } else {
                    langid[0] = 1.0;
                }
            }
        }

        let mut target_lid = vec![0.0f64; self.class_nb];
        target_lid[self.ti] = -2.0;
        let mut errors = vec![0.0f64; self.class_nb];
        for c in 0..self.class_nb {
            errors[c] = 100.0 * (langid[c] - target_lid[c]);
        }
        let max_langid = langid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let is_lid_correct = if langid[self.ti] == max_langid {
            100
        } else {
            0
        };
        let predicted_language = argmax_f64(&langid);

        LidAggregate {
            classification_errors: errors,
            confusion: self.confusion.clone(),
            is_lid_correct,
            predicted_language,
            segments_count: self.segments_count,
        }
    }
}

/// BLSTM spectral LID driver (Algo 5; `BLSTMSpectralLID.{h,cpp}`).
///
/// Holds the spectral/mel/LTSV config (via [`FeatureConfig`]), the SAD decision config
/// ([`SegmenterConfig`]/[`DriverConfig`]), and the LID BLSTM net. Stateful members
/// (`_SpectrumShift`, `_WindowShift`) self-quantize per call exactly like the spectral
/// driver's. [`Self::lid_members`] surfaces the per-channel LID members ([`LidMembers`]).
#[derive(Clone)]
pub struct BlstmSpectralLid {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    feature_cfg: FeatureConfig,
    net: BlstmNetwork,
    /// The stateful `_SpectrumShift` (`:53`) and `_WindowShift` (`:187/:458`), issue #121.
    /// The legacy's `_SpectrumShiftInFrames` is a LOCAL here (`:51`); the carry holds it
    /// as state, read by nothing between files (the next call recomputes it from
    /// `_SpectrumShift`), so the same struct serves.
    carry: SpectralCarry,
    /// Port-side observation point (NO legacy counterpart, same rationale as
    /// `BlstmSpectralSegmenter::last_result_rows`): the PRE-convolution LTSV-SAD
    /// result_vec per channel, captured before `results_to_segmentation` mutates it.
    last_result_rows: Vec<Vec<f64>>,
    /// The per-channel LID members (`seg._LID*[chan]`, `seg._IsLIDCorrect[chan]`).
    lid: LidMembers,
}

impl BlstmSpectralLid {
    /// Port of the `BLSTMSpectralLID(ConfigFile&, ...)` ctor (`BLSTMSpectralLID.cpp:
    /// 10-14`): `BLSTMSpectralSegmenter::buildFromConf(conf, "BLSTM", ...)` (the
    /// `Segmenter` + spectral residue via [`SegmenterConfig`]/[`DriverConfig`]/
    /// [`FeatureConfig`]) + `BLSTMNeuralNetwork(conf, "BLSTM", false)`. `weights`
    /// mirrors the two ctor overloads (`Some(flat)` -> `setWeights`, `None` -> the
    /// no-weights ctor), matching [`BlstmSpectralSegmenter::from_legacy`].
    pub fn from_legacy(
        map: &IndexMap<String, String>,
        weights: Option<&[f64]>,
    ) -> Result<BlstmSpectralLid> {
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;

        let blstm_cfg = BlstmConfig::from_legacy(map, "BLSTM")?;
        let mut net = BlstmNetwork::from_config(blstm_cfg)?;
        if let Some(flat) = weights {
            net.set_weights(flat)?;
        }

        let carry = SpectralCarry::new(feature_cfg.shift_sec, driver_cfg.window_shift_sec);

        Ok(BlstmSpectralLid {
            driver_cfg,
            seg_cfg,
            feature_cfg,
            net,
            carry,
            last_result_rows: Vec::new(),
            lid: LidMembers::default(),
        })
    }

    /// `<prefix>_weightsFile` load delegate (see `BlstmSpectralSegmenter::
    /// load_weights_file`): applies a `weightsFile` key AFTER building via
    /// `from_legacy(map, None)`, matching the legacy in-ctor load.
    pub fn load_weights_file(&mut self, map: &IndexMap<String, String>) -> Result<()> {
        self.net.load_weights_file(map, "BLSTM")
    }

    /// The per-channel LID members written by the last `get_segmentation` call.
    pub fn lid_members(&self) -> &LidMembers {
        &self.lid
    }
}

impl Segmenter for BlstmSpectralLid {
    /// Port of `BLSTMSpectralLID::getSegmentation` (`BLSTMSpectralLID.cpp:27-460`), the
    /// non-unit-test/non-plotting path (the `.mat`/PNG branches `:59-77,:207-219,
    /// :277-296,:303-305,:428-457` dropped: display/diagnostic-only, and the LID driver
    /// writes NO VRCTS in `getSegmentation`). `refs` is IGNORED: the LID scoring builds
    /// its own per-frame target internally from `audio.getRefLangIndex()` (`:363`), and
    /// the SAD is LTSV (NN-free) -- no `getTargets` call exists in this driver.
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        _refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        let rate = audio.sample_rate as f64;

        // Feature params (order clamp, pre-mel freq band, mel sizing) via the shared
        // derivation; `s.shift_frames` is not consumed (the LID uses the stateful
        // spectrum_shift below), but the band/order/bins are.
        let s = SpectralParams::derive(&self.feature_cfg, rate);

        // The input gates (`:78-89`) first -- they mutate the audio in place and read nothing
        // the carry writes -- then the carry's preamble (#121): spectrum_shift, a LOCAL in the
        // legacy (`:51`) forced to 80 on a non-wav read (`:52`, this driver's only non-wav
        // branch: the periodogram is computed off the waveform either way) and whose
        // re-quantized seconds `_SpectrumShift` (`:53`) persists, and getBLSTMParam
        // (`:175-190`), which mutates the stateful `_WindowShift`.
        // The carry's spectral time axis is NOT this driver's: the scoring axis below
        // follows the SIGNAL pattern (`:333-343`), and the returned real_vec_size (the
        // ssr-division) is IGNORED -- the LID sizes result_vec by LTSV_window_shift below
        // (divergence #3). The pure derivations between the two legacy sites (windowing,
        // mel bank, temporal convolution, LTSV) read neither the window shift nor the
        // audio, so they follow the call unchanged.
        apply_input_gates(audio, &self.feature_cfg);
        let ssr = self.net.sub_sampling_ratio();
        let SpectralAxis {
            ssif: spectrum_shift,
            spectrum_shift_sec,
            window_size,
            window_shift,
            no_overlap,
            ..
        } = self.carry.advance(
            rate,
            &self.driver_cfg,
            &self.net,
            audio.data.ncols(),
            non_wav_ssif(audio),
        );

        // Signal windowing coeffs (`:90`).
        let windowing_coeff = windowing_coefficients(
            &self.feature_cfg.win_type,
            false,
            s.buffer_size,
            self.feature_cfg.win_param,
        );

        // Mel bank + POST-mel freq band (`:97-146`). freq_beg/freq_end are RESET to
        // (0, mel/DCT width - 1) when mel is active (`:141-142`); else they stay the
        // pre-mel band (`:98-108`). The LTSV loop + the raw-path scoring block both read
        // these post-reset values (`:271-274`, `:360-361`).
        let (mel_bank, freq_beg, freq_end) = if self.feature_cfg.nb_bins > 0 {
            let bank = MelFilterBank::new(
                self.feature_cfg.min_mel,
                self.feature_cfg.max_mel,
                self.feature_cfg.nb_bins,
                s.min_freq,
                s.max_freq,
                rate,
                s.bins - 1,
                self.feature_cfg.is_log,
                self.feature_cfg.nb_dct,
                self.feature_cfg.ignore_first,
                self.feature_cfg.deltas_nb,
                self.feature_cfg.dd_nb,
            );
            let width = if self.feature_cfg.nb_dct > 0 {
                bank.nb_dct()
            } else {
                bank.nb_filters()
            };
            (Some(bank), 0usize, width - 1)
        } else {
            (None, s.freq_beg, s.freq_end)
        };

        // Temporal convolution of the periodogram (`:148-149`).
        let temporal_conv = windowing_coefficients(
            &self.feature_cfg.conv_type,
            true,
            2 * self.feature_cfg.conv_size as usize + 1,
            0.83333,
        );

        // LTSV window/shift (`:157-160`). FLOOR TO 1 (the divergence #2): even
        // `LTSVwindow 0 -> round(0) = 0 -> 1`, so LTSV SAD is always active.
        let mut ltsv_half_window =
            f64::round(self.feature_cfg.ltsv_window * rate / 2.0 / spectrum_shift as f64) as i64;
        if ltsv_half_window < 1 {
            ltsv_half_window = 1;
        }
        let ltsv_half_window = ltsv_half_window as usize;
        let mut ltsv_window_shift =
            f64::round(self.feature_cfg.ltsv_shift * rate / spectrum_shift as f64) as i64;
        if ltsv_window_shift < 1 {
            ltsv_window_shift = 1;
        }
        let ltsv_window_shift = ltsv_window_shift as usize;

        // TDC params (`:162-173`) are DEAD in the LID driver (no pitch pass) -- skipped.

        // vec_size / real_vec_size (`:221-232`): vec_size = ceil(frameCount /
        // spectrum_shift); real_vec_size = ceil(vec_size / LTSV_window_shift) (the LTSV
        // decimation, NOT the ssr-division).
        let frame_count = audio.data.ncols();
        let vec_size = if frame_count.is_multiple_of(spectrum_shift) {
            frame_count / spectrum_shift
        } else {
            frame_count / spectrum_shift + 1
        };
        let real_vec_size = if vec_size.is_multiple_of(ltsv_window_shift) {
            vec_size / ltsv_window_shift
        } else {
            vec_size / ltsv_window_shift + 1
        };

        // resetWeightsDerivatives ONCE (`:234`).
        self.net.reset_weights_derivatives();

        // scoring timeStep/timeOffset (`:333-343`, the SIGNAL pattern: overlap keeps
        // `_WindowShift`). window_shift_sec here is the POST-getBLSTMParam value.
        let window_shift_sec = self.carry.window_shift_sec();
        let mut time_step = window_shift_sec * ssr as f64;
        let mut time_offset = time_step / 2.0 - window_shift_sec / 2.0;
        if window_size > 0 {
            if no_overlap {
                time_step = window_shift_sec * ssr as f64;
                time_offset = time_step / 2.0 - window_shift_sec / 2.0;
            } else {
                time_step = window_shift_sec;
                time_offset = 0.0;
            }
        }
        let _ = time_offset; // SAD results2segmentation uses offset 0.0 (`:302`); scoring
        // needs only time_step (the ponderation modifier). timeOffset is computed for
        // parity but unused in the live path.

        // classNb = max(2, output_size) (`:309-310`).
        let class_nb = self.net.output_size().max(2);

        // setProcessingType((window > 0), !noOverlap) ONCE (`:344`).
        self.net.set_processing_type(window_size > 0, !no_overlap);

        let channels = audio.data.nrows();
        self.last_result_rows = Vec::with_capacity(channels);
        self.lid = LidMembers::zeroed(channels);

        // Stable per-file locals for the borrow checker (self.net is borrowed mut in the
        // scoring loop; everything else must be a local read).
        let conv_coeff = self.driver_cfg.conv_coeff.clone();

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // Periodogram (`:242-244`): begin 0, end frameCount-1 (NOT the LTSV
            // driver's +1 quirk). Raw periodogram (mel is applied separately below).
            let perio = compute_segment_periodogram_estimates(
                audio,
                s.order,
                spectrum_shift,
                chan,
                self.feature_cfg.flag_dc_offset,
                windowing_coeff.as_deref(),
                temporal_conv.as_deref(),
                0,
                frame_count - 1,
            );

            // Mel/DCT for the per-segment scoring input (`:354-359`). The RAW periodogram
            // above feeds the LTSV SAD; the mel/DCT feeds the NN.
            let (fb, cep) = match &mel_bank {
                Some(bank) => {
                    let fb = bank.apply_filter_bank(&perio);
                    let cep = if self.feature_cfg.nb_dct > 0 {
                        Some(bank.apply_dct(&fb))
                    } else {
                        None
                    };
                    (Some(fb), cep)
                }
                None => (None, None),
            };

            // --- LTSV SAD (`:271-274`): DECIMATED write over the RAW periodogram, NO
            // interpolation backfill. `result_vec` is fresh per channel: the legacy
            // reuses one buffer across channels, but every index is overwritten each
            // channel (the loop covers 0..real_vec_size), so fresh-per-channel is
            // byte-identical (unlike the spectral overlap-accumulation reuse quirk).
            let mut result_vec = vec![0.0f64; real_vec_size];
            let mut jj = 0usize;
            while jj < vec_size {
                result_vec[jj / ltsv_window_shift] =
                    ltsv_classify_sequence(&perio, jj, freq_beg, freq_end, ltsv_half_window);
                jj += ltsv_window_shift;
            }

            // Capture the PRE-convolution LTSV row (port-side observation point).
            self.last_result_rows.push(result_vec.clone());

            // results2segmentation (`:300-302`): timeStep = _WindowShift, offset 0.0.
            results_to_segmentation(
                seg,
                window_shift_sec,
                0.0,
                &mut result_vec,
                SegClass::Speech,
                conv_coeff.as_deref(),
                &self.seg_cfg,
            );

            // --- LID scoring loop (`:346-401`) --------------------------------------
            let mut nn_cost = 0.0f64;
            let mut nb_of_classif = 0i64;

            // targetIndex = clamp(audio.getRefLangIndex()) to [0, classNb) (`:321-323`).
            let ti = clamp_target_index(audio.lang_index, class_nb);
            let target_index = ti as i32;
            // The scoring accumulator (`:314-318` seed): the one kernel, `LidChannelAcc`.
            let mut acc = LidChannelAcc::new(class_nb, ti, self.net.is_cost_modified());

            // Snapshot the SAD boundaries (the scoring loop does not mutate `seg`).
            let segs = seg.segments().to_vec();
            for i in 0..segs.len().saturating_sub(1) {
                if segs[i].ty != SegClass::Speech {
                    continue;
                }
                // rowBegin = ceil(begin / _SpectrumShift) (`:349-350`, truncate then
                // bump-if-not-exact); rowEnd = floor(next.begin / _SpectrumShift)
                // (`:351`).
                let ratio = segs[i].begin / spectrum_shift_sec;
                let mut row_begin = ratio as usize;
                if (row_begin as f64) != ratio {
                    row_begin += 1;
                }
                let row_end = (segs[i + 1].begin / spectrum_shift_sec) as usize;

                // `rowEnd - rowBegin + 1 >= ssr` (`:352`). `row_end >= row_begin` guards
                // the unsigned underflow (the legacy UB-slices there; unreachable with
                // real min-length segments -- see IMPROVEMENTS.md).
                if row_end < row_begin || row_end - row_begin + 1 < ssr {
                    continue;
                }
                let n_rows = row_end - row_begin + 1;

                // inputSeq block (`:354-361`): DCT block, else filterbank block, else the
                // raw log-periodogram band block.
                let mut input_seq = if let Some(cep) = cep.as_ref() {
                    let cols = cep.ncols();
                    let mut m = Array2::<f64>::zeros((n_rows, cols));
                    for r in 0..n_rows {
                        for c in 0..cols {
                            m[[r, c]] = cep[[row_begin + r, c]];
                        }
                    }
                    m
                } else if let Some(fb) = fb.as_ref() {
                    let cols = fb.ncols();
                    let mut m = Array2::<f64>::zeros((n_rows, cols));
                    for r in 0..n_rows {
                        for c in 0..cols {
                            m[[r, c]] = fb[[row_begin + r, c]];
                        }
                    }
                    m
                } else {
                    // raw path: ln(perio.block(rowBegin, freq_beg, n_rows, freq_end-freq_beg+1) + 1e-24).
                    let width = freq_end - freq_beg + 1;
                    let mut m = Array2::<f64>::zeros((n_rows, width));
                    for r in 0..n_rows {
                        for c in 0..width {
                            m[[r, c]] = (perio[[row_begin + r, freq_beg + c]] + 1e-24).ln();
                        }
                    }
                    m
                };

                // feedForward scoring (`:363`): modifier = timeStep / segment-duration.
                let modifier = time_step / (segs[i + 1].begin - segs[i].begin);
                let output_seq = self.net.feed_forward_scoring(
                    &mut input_seq,
                    window_size,
                    window_shift,
                    target_index as i64,
                    modifier,
                );

                // The `_CepstreCoefficients` writeback (`:364-368`) is DEAD -- skipped.

                nn_cost += self.net.cost; // NNCost += getCost() (`:370`)
                nb_of_classif += self.net.nb_of_classif; // (`:371`)

                // segLID = colwise sum / rows (`:372`), argmax j (`:378`) + confusion
                // (`:379-389`; `costLID` `:376` is a dead local, not reproduced), langID +=
                // segLID (`:391`): the kernel.
                acc.push(
                    output_seq.view(),
                    BlockScore::ColumnMean,
                    BlockWeight::Plain,
                );
            }

            // Post-loop reductions (`:403-411`) + the member writes (`:412-426`).
            let agg = acc.finalize(LidNorm::MeanOverSegments);
            self.lid.cumulative_error[chan] = nn_cost;
            self.lid.nb_of_classif[chan] = nb_of_classif;
            self.lid.classification_errors[chan] = agg.classification_errors;
            self.lid.segments_confusion[chan] = agg.confusion;
            self.lid.is_correct[chan] = agg.is_lid_correct;
        }

        // seg.compute_errors() after the FULL channel loop (`:438`).
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }

        // noOverlap `_WindowShift = 0.0` reset (`:458`).
        self.carry.reset_after(no_overlap);

        Ok(())
    }

    /// The configured dump directory (see `TdcSegmenter::dump_dir`).
    fn dump_dir(&self) -> &str {
        &self.driver_cfg.dump_dir
    }

    /// PRE-convolution LTSV-SAD result rows (one per channel) from the last call.
    fn last_result_rows(&self) -> &[Vec<f64>] {
        &self.last_result_rows
    }

    fn pack_count(&self) -> usize {
        1
    }

    fn set_packs(&mut self, packs: &[Vec<f64>]) -> Result<()> {
        self.net.set_weights(&packs[0])
    }

    /// The bag's algo-5 arms (`BagOfProcessors.cpp:79,94,109,122,138,168,194`) resolve
    /// to the INHERITED `BLSTMSpectralSegmenter` methods on `_BLSTMNeuralNetwork`, and in
    /// `BLSTMSpectralLID` that base-class member IS the LID net (the ctor builds it from
    /// the same `BLSTM_*` namespace).
    fn nets(&self) -> Vec<&BlstmNetwork> {
        vec![&self.net]
    }

    fn nets_mut(&mut self) -> Vec<&mut BlstmNetwork> {
        vec![&mut self.net]
    }

    fn channel_members(&self, chan: usize) -> ChannelMembers<'_> {
        // The SAD is LTSV (NN-free): no SAD cost or count, only the LID members.
        ChannelMembers {
            cumulative_error: 0.0,
            nb_of_classif: 0,
            lid: Some(self.lid.channel(chan)),
        }
    }
}

// ===========================================================================
// Twin/Siamese LID driver (Algo 6; `TwinBLSTMSpectralLID.{h,cpp}`).
// ===========================================================================

/// Port of `TwinBLSTMSpectralLID.cpp:906-913`'s `_DumpDir.size() > 0` branch (the only
/// branch reachable here, since the call site already guards on `!dump_dir.is_empty()`):
///
/// ```cpp
/// basefilename = audio.getAudioFileName().substr(
///     audio.getAudioFileName().find_last_of('/')+1,
///     audio.getAudioFileName().size()-audio.getAudioFileName().find_last_of('/')-1-4);
/// basefilename = _DumpDir+"/"+basefilename;
/// ```
///
/// Strips the directory (the portion after the last `/`, or the whole string if there is
/// none) then ALWAYS drops exactly 4 trailing characters from that basename -- NOT an
/// extension-aware strip. For a `.wav` name this cleanly removes `.wav`; for the mode-7
/// phSeq arm's `.phSeq` (6 chars) it leaves a partial extension (`"s1.phSeq"` ->
/// `"s1.p"`), a load-bearing legacy quirk reproduced verbatim (see IMPROVEMENTS.md). When
/// the basename is shorter than 4 bytes, `std::string::substr`'s length-clamping (the
/// requested length underflows to a huge `size_t`, then gets clamped to the remaining
/// string) leaves it untouched rather than panicking -- well-defined in C++, reproduced
/// here with an explicit length check instead of relying on unsigned wraparound.
fn mode7_dump_basename(audio_file_name: &str, dump_dir: &str) -> String {
    let after_slash = match audio_file_name.rfind('/') {
        Some(i) => &audio_file_name[i + 1..],
        None => audio_file_name,
    };
    let stem = if after_slash.len() >= 4 {
        &after_slash[..after_slash.len() - 4]
    } else {
        after_slash
    };
    format!("{dump_dir}/{stem}")
}

/// Twin/Siamese spectral LID driver (Algo 6; `TwinBLSTMSpectralLID.{h,cpp}`).
///
/// UNLIKE Algo 5 (`BlstmSpectralLid`, whose SAD is LTSV-driven), Algo 6's SAD IS the
/// BLSTM spectral segmenter: `getSegmentation` (`:263-1421`) inlines the BASE
/// `BLSTMSpectralSegmenter` spectrum pipeline (`:349-619`, the same
/// `getBLSTMInputSequence` the base driver uses), runs the SAD BLSTM
/// (`_BLSTMNeuralNetwork`, `:715`) for the VAD result_vec, then a SECOND
/// (`_LIDBLSTMNeuralNetwork`) per-speech-segment for language scoring
/// (`:1243-1291`). The two nets read the `BLSTM` / `BLSTM_LID` config namespaces.
///
/// THIS DRIVER ports the wav-mode 0/2/3 paths + the hidden-state concat only. Modes
/// 4/5/6/7 (the `if ((_Mode==4)||(_Mode==5)||(_Mode==6))` LID-first block `:640-692`,
/// the `abs(_Mode)==7` CNN/noise block `:903-1193`) and the pitch second pass
/// (`:349-614` `TDC_window_size > 0`, a REFERENCE-based pre-forward warp that DIVERGES
/// from the base driver's post-forward pitch pass) BAIL typed, deferred to the next
/// task. The CNN member (`_LIDConvNeuralNetwork`, `:23`) is NOT ported (dead under the
/// port scope, only live in mode 7 -- see IMPROVEMENTS phase2 Conv exclusion).
///
/// MODE gating on the ported `:1194-1310` scoring branch (reached by modes 0/1/2/3):
/// - Mode 0/3 run the SAD net `feedForwardBackward` (`:713-717`), populating
///   `_OutputForward`/`_OutputBackward` (consumed by the concat).
/// - Mode 2 synthesizes `result_vec = LID_result_vec` (all-zero, `:719-724`) and CLEARS
///   the SAD net outputs (`:762-763`) -> the concat takes the empty fallback.
/// - Mode 1 fills `result_vec` with the constant 10.0 (`:726`) + clears outputs.
/// - The scoring loop iterates the REFERENCE for modes 2/3, the CLASSIFICATION for
///   modes 0/1 (`:1223-1239`). For modes != 0 the final classification is OVERWRITTEN by
///   `LID2Segmentation(segmentationLID)` (`:1294-1298`).
///
/// The `lid_*` accessors mirror [`BlstmSpectralLid`]'s; unlike Algo 5, the SAD-side
/// `cumulative_error`/`nb_of_classif` ARE written (`= NNCost`, then `*= LIDCostPonderation`
/// `:774,:1337`) since the SAD is a real NN here.
#[derive(Clone)]
pub struct TwinBlstmSpectralLid {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    feature_cfg: FeatureConfig,
    sad_net: BlstmNetwork,
    lid_net: BlstmNetwork,
    /// `_LIDWindowSize` (`:15-16`, clamped `< 0 -> 0`); constant.
    lid_window_size_sec: f64,
    /// `_LIDWindowShift` (`:17-18` seed, mutated by `getLIDBLSTMParam` `:102`, reset to
    /// `0.0` on LIDnoOverlap `:1420`); stateful.
    lid_window_shift_sec: f64,
    /// `_LIDDecisionThreshRising` (`:28`). Used by `LID2Segmentation` (`:1296`) + the
    /// `segmentationLID` midpoint seed (`:1242`).
    lid_decision_thresh_rising: f64,
    /// `_LIDDecisionThreshFalling` (`:29`). Midpoint seed only on the 0/2/3 path
    /// (`LID2Segmentation` ignores its `threshMin` arg -- see [`lid_to_segmentation`]).
    lid_decision_thresh_falling: f64,
    /// `_Mode` (`:33`, default 0). 0/1/2/3 ported; 4/5/6/7 bail.
    mode: i32,
    /// `_PostProcessMode` (`:35`, default 0). The 0/2/3 scoring branch always sums log
    /// posteriors via `segLID = colwise sum / rows` (the `_PostProcessMode` 1/2 variants
    /// live only in the mode 4/5/6/7 blocks) -- stored for the next task.
    post_process_mode: i32,
    /// `_MinNbOfFrames` (`:36`, default 0). Mode 7-only; stored for the next task.
    min_nb_of_frames: i32,
    /// `_NoiseMagnitude` (`:37`, default 0.0). Mode 7-only; stored for the next task.
    noise_magnitude: f64,
    /// `_DumpLIDInternals` (`:38`, default false). Mode 7-only; stored for the next task.
    dump_lid_internals: bool,
    /// The stateful `_SpectrumShift` / `_SpectrumShiftInFrames` / SAD `_WindowShift`
    /// (`:208-210`, `:453/:1419`), issue #121.
    carry: SpectralCarry,
    /// `seg._CumulativeError[chan]` (`:774`, then `*= LIDCostPonderation` `:1337`).
    cumulative_error: Vec<f64>,
    /// `seg._NbOfClassif[chan]` (`:775`).
    nb_of_classif: Vec<i64>,
    /// Port-side observation point (no legacy counterpart, same rationale as
    /// [`BlstmSpectralSegmenter::last_result_rows`]): the SAD `result_vec2` row per
    /// channel captured PRE-convolution (before `results_to_segmentation`).
    last_result_rows: Vec<Vec<f64>>,
    /// The per-channel LID members (`seg._LID*[chan]`, `seg._IsLIDCorrect[chan]`).
    lid: LidMembers,
    /// Port-side observation point (no legacy counterpart): which branch of
    /// `getBLSTMLIDInputSequence` (`:139-168`) the concat guard (`:1221`) took per
    /// channel -- `0` not called (LID input <= feature width), `1` concat (SAD
    /// `_OutputForward` populated), `2` empty fallback (outputs cleared/empty). Pins the
    /// concat non-vacuity.
    concat_branch: Vec<i32>,
}

impl TwinBlstmSpectralLid {
    /// Port of the `TwinBLSTMSpectralLID(ConfigFile&, ...)` ctor (`:10-39`):
    /// `BLSTMSpectralSegmenter::buildFromConf(conf, "BLSTM", ...)` (the `Segmenter` +
    /// spectral residue) + `BLSTMNeuralNetwork(conf, "BLSTM", false)` (the SAD net) +
    /// `BLSTMNeuralNetwork(conf, "BLSTM_LID", false)` (the LID net) + the `BLSTM_LID_*`
    /// scalars. The `_LIDConvNeuralNetwork(conf, "CNN", false)` (`:23`) is NOT ported.
    /// `weights`/`lid_weights` mirror the two ctor overloads: `Some(flat)` ->
    /// `setWeights`/`setWeightsLID`, `None` -> the no-weights ctor.
    pub fn from_legacy(
        map: &IndexMap<String, String>,
        weights: Option<&[f64]>,
        lid_weights: Option<&[f64]>,
    ) -> Result<TwinBlstmSpectralLid> {
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;

        let sad_cfg = BlstmConfig::from_legacy(map, "BLSTM")?;
        let mut sad_net = BlstmNetwork::from_config(sad_cfg)?;
        if let Some(flat) = weights {
            sad_net.set_weights(flat)?;
        }

        let lid_cfg = BlstmConfig::from_legacy(map, "BLSTM_LID")?;
        let mut lid_net = BlstmNetwork::from_config(lid_cfg)?;
        if let Some(flat) = lid_weights {
            lid_net.set_weights(flat)?;
        }

        let mut lid_window_size_sec = get_f64(map, "BLSTM_LID_window")?;
        if lid_window_size_sec < 0.0 {
            lid_window_size_sec = 0.0;
        }
        let mut lid_window_shift_sec = get_f64(map, "BLSTM_LID_shift")?;
        if lid_window_shift_sec < 0.0 {
            lid_window_shift_sec = 0.0;
        }

        // `_LIDDetectionThreshold` (`:25`) and `_LIDTrainingPruningThreshold` (`:26`),
        // both default -1.0: read by nothing in the port. The legacy `:1286` pruning
        // gate is NOT reproduced (it feeds only the skipped VRCTS dump, see the
        // `interestSegs` note in `get_segmentation_mode456` and IMPROVEMENTS.md).
        // Parsed for validation only (a malformed value still errors), the values
        // discarded (issue #94).
        get_f64_default(map, "BLSTM_LID_DetectionThreshold", -1.0)?;
        get_f64_default(map, "BLSTM_LID_TrainingPruningThreshold", -1.0)?;
        let lid_decision_thresh_rising = get_f64(map, "BLSTM_LID_decision_thresh_rising")?;
        let lid_decision_thresh_falling = get_f64(map, "BLSTM_LID_decision_thresh_falling")?;
        let mode = get_default::<i32>(map, "BLSTM_LID_Mode", 0)?;
        let post_process_mode = get_default::<i32>(map, "BLSTM_LID_PostProcessMode", 0)?;
        let min_nb_of_frames = get_default::<i32>(map, "BLSTM_LID_MinNbOfFrames", 0)?;
        let noise_magnitude = get_f64_default(map, "BLSTM_LID_NoiseMagnitude", 0.0)?;
        let dump_lid_internals = get_default::<bool>(map, "BLSTM_LID_DumpInternals", false)?;

        let carry = SpectralCarry::new(feature_cfg.shift_sec, driver_cfg.window_shift_sec);

        Ok(TwinBlstmSpectralLid {
            driver_cfg,
            seg_cfg,
            feature_cfg,
            sad_net,
            lid_net,
            lid_window_size_sec,
            lid_window_shift_sec,
            lid_decision_thresh_rising,
            lid_decision_thresh_falling,
            mode,
            post_process_mode,
            min_nb_of_frames,
            noise_magnitude,
            dump_lid_internals,
            carry,
            cumulative_error: Vec::new(),
            nb_of_classif: Vec::new(),
            last_result_rows: Vec::new(),
            lid: LidMembers::default(),
            concat_branch: Vec::new(),
        })
    }

    /// `<prefix>_weightsFile` load delegates for BOTH nets (`BLSTM` / `BLSTM_LID`),
    /// matching the legacy in-ctor loads.
    pub fn load_weights_file(&mut self, map: &IndexMap<String, String>) -> Result<()> {
        self.sad_net.load_weights_file(map, "BLSTM")?;
        self.lid_net.load_weights_file(map, "BLSTM_LID")
    }

    // -- scalar/config accessors --------------------------------------------
    pub fn post_process_mode(&self) -> i32 {
        self.post_process_mode
    }
    // -- LID-side outputs (mirror Algo 5) ------------------------------------
    /// The per-channel LID members written by the last `get_segmentation` call.
    pub fn lid_members(&self) -> &LidMembers {
        &self.lid
    }
    /// Per-channel concat branch taken (`0`/`1`/`2`, see the field doc) -- concat
    /// non-vacuity.
    pub fn concat_branch(&self) -> &[i32] {
        &self.concat_branch
    }

    /// Port of `getBLSTMLIDInputSequence` (`:139-168`): when the SAD net's
    /// `_OutputForward` is populated, hcat its z-normalized `[fwd | bwd]` hidden states
    /// onto `input_seq`, resampling by nearest index when the LSTM output row count
    /// differs from `input_seq`'s (`:150-163`); an EMPTY `_OutputForward` returns
    /// `input_seq` unchanged (`:164-166`). Returns `(augmented, branch)` with `branch`
    /// `1` = concat, `2` = empty fallback.
    fn get_blstm_lid_input_sequence(&self, input_seq: &Array2<f64>) -> (Array2<f64>, i32) {
        let fwd = &self.sad_net.output_forward;
        let bwd = &self.sad_net.output_backward;
        if fwd.nrows() == 0 {
            return (input_seq.clone(), 2);
        }
        // addedColumns = [fwd | bwd], then per-column z-normalize (mean/std over rows,
        // std with the +1e-32 floor), `:142-148`.
        let n = fwd.nrows();
        let fc = fwd.ncols();
        let bc = bwd.ncols();
        let added_cols = fc + bc;
        let mut added = Array2::<f64>::zeros((n, added_cols));
        for r in 0..n {
            for c in 0..fc {
                added[[r, c]] = fwd[[r, c]];
            }
            for c in 0..bc {
                added[[r, fc + c]] = bwd[[r, c]];
            }
        }
        for c in 0..added_cols {
            let mut sum = 0.0;
            for r in 0..n {
                sum += added[[r, c]];
            }
            let mean = sum / n as f64;
            let mut sq = 0.0;
            for r in 0..n {
                let d = added[[r, c]] - mean;
                added[[r, c]] = d;
                sq += d * d;
            }
            let std = ((sq + 1e-32) / n as f64).sqrt();
            for r in 0..n {
                added[[r, c]] /= std;
            }
        }

        let rows = input_seq.nrows();
        let cols = input_seq.ncols();
        let mut out = Array2::<f64>::zeros((rows, cols + added_cols));
        for r in 0..rows {
            for c in 0..cols {
                out[[r, c]] = input_seq[[r, c]];
            }
        }
        if n == rows {
            // Direct hcat (`:150-152`).
            for r in 0..rows {
                for c in 0..added_cols {
                    out[[r, cols + c]] = added[[r, c]];
                }
            }
        } else {
            // Nearest-index resample (`:154-162`): indexSpectrum = round((n-1)*frame/
            // (length-1)) via the `+0.5` truncation.
            let denom = (rows as f64 - 1.0).max(1.0);
            for frame in 0..rows {
                let idx = ((n as f64 - 1.0) * frame as f64 / denom + 0.5) as usize;
                for c in 0..added_cols {
                    out[[frame, cols + c]] = added[[idx, c]];
                }
            }
        }
        (out, 1)
    }

    /// Port of `getTargetsLID` (`:170-219`): the per-frame LID target from the
    /// (smoothed reference) classification `seg`. Multiclass (`output_size > 1`): a
    /// SPEECH row enforces `[target..]` with `target_index -> 1-target`; an OTHER row
    /// enforces `target_index -> target`, `col0 -> 1-target`; non-enforced rows are
    /// `-0.5`. Binary (`output_size <= 1`): SPEECH enforces `(target_index==1) ?
    /// 1-target : target`, OTHER is always `-0.5`. `target = isCostModified() ? 0.1/
    /// dur_seg : 0.0`. Enforcement uses `counter == step` (EQUALITY, `:182` -- NOT the
    /// scoring path's `>=`). Called by modes 4/5/6 (`:662`) to build the LID net's
    /// training target from the smoothed reference; exercised by a direct unit test too.
    fn get_targets_lid(
        &self,
        seg: &Segmentation,
        time_step: f64,
        time_offset: f64,
        target_index: usize,
        n_rows: usize,
    ) -> Array2<f64> {
        let out_size = self.lid_net.output_size();
        let cols = out_size.max(1);
        let mut targets = Array2::<f64>::zeros((n_rows, cols));
        let segs = seg.segments();
        if segs.is_empty() {
            return targets;
        }
        let cost_modified = self.lid_net.is_cost_modified();
        let step = self.lid_net.target_enforcement_step();
        let mut counter = 0i32;
        let mut idx = 0usize;
        for ii in 0..n_rows {
            let t = ii as f64 * time_step + time_offset;
            while idx + 1 < segs.len() && segs[idx + 1].begin <= t {
                idx += 1;
            }
            let target = if cost_modified && idx + 1 < segs.len() {
                0.1 / (segs[idx + 1].begin - segs[idx].begin)
            } else {
                0.0
            };
            if out_size > 1 {
                if segs[idx].ty == SegClass::Speech {
                    if counter == step {
                        for c in 0..cols {
                            targets[[ii, c]] = target;
                        }
                        targets[[ii, target_index]] = 1.0 - target;
                        counter = 0;
                    } else {
                        for c in 0..cols {
                            targets[[ii, c]] = -0.5;
                        }
                        counter += 1;
                    }
                } else if counter == step {
                    targets[[ii, target_index]] = target;
                    targets[[ii, 0]] = 1.0 - target;
                    counter = 0;
                } else {
                    for c in 0..cols {
                        targets[[ii, c]] = -0.5;
                    }
                    counter += 1;
                }
            } else if segs[idx].ty == SegClass::Speech {
                if counter == step {
                    counter = 0;
                    targets[[ii, 0]] = if target_index == 1 {
                        1.0 - target
                    } else {
                        target
                    };
                } else {
                    counter += 1;
                    targets[[ii, 0]] = -0.5;
                }
            } else {
                targets[[ii, 0]] = -0.5;
            }
        }
        targets
    }

    /// Override the DumpLIDInternals output directory (`_DumpDir`). The legacy derives
    /// the `.mat` filename from `audio.getAudioFileName()` (`:906-913`, see
    /// [`mode7_dump_basename`]) -- `<dir>/<audio basename minus 4 trailing chars>_chan<c>
    /// _lid_dump.mat`. Closed 4c-era fix: `Audio::audio_file_name` (landed in 4b Task 8,
    /// populated post-hoc by `engine::bag_of_processors::apply_corpus_item`, not by
    /// `read_audio`/`read_phseq`) is now threaded into this dump path.
    pub fn set_dump_dir(&mut self, dir: String) {
        self.driver_cfg.dump_dir = dir;
    }

    /// Port of the `abs(_Mode) == 7` branch (`:903-1193`), phSeq (File_Type 1) arm only.
    /// The SAD net is NEVER run (`result_vec` is synthesized constant 10.0 at `:726`);
    /// the LID net scores each `audio._ExternalFeatures[i]` block directly (`:980-1155`).
    /// The wav arm (`:922-963`) runs the CNN (`_LIDConvNeuralNetwork`, not ported) -> bail.
    fn get_segmentation_mode7(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        _refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        if audio.periodogram.is_none() {
            return Err(anyhow!(
                "TwinBlstmSpectralLid mode 7: wav arm runs the CNN (not ported); only File_Type 1 (phSeq) supported (legacy :922-963)"
            ));
        }
        let rate = audio.sample_rate as f64;
        let s = SpectralParams::derive(&self.feature_cfg, rate);
        if let Some(tdc) = s.tdc.as_ref()
            && tdc.half_window > 0
        {
            return Err(anyhow!(
                "TwinBlstmSpectralLid mode 7: pitch pass (TDCwindow > 0) not ported"
            ));
        }

        // The carry's preamble (#121): the stateful spectrum-shift quantization
        // (`initSpectralAnalysis` `:208-210`) with the `!hasReadWavFile()` override (`:209`)
        // forcing `_SpectrumShiftInFrames = 80` for phSeq (File_Type 1) -- LOAD-BEARING: it
        // drives every downstream window/shift derivation (getLIDBLSTMParam window 25, not
        // the 10 the raw `round(0.025*8000) = 200` would give); keyed on the audio by
        // `non_wav_ssif` as on every spectral driver, always `Some` past the wav bail above.
        // The input gates (harmless on the zeroed phSeq waveform) run first; getBLSTMParam
        // (the SAD result_vec sizing) and the SAD time axis (`:696-706`) follow inside
        // `advance`.
        apply_input_gates(audio, &self.feature_cfg);
        let SpectralAxis {
            ssif,
            no_overlap,
            real_vec_size,
            time_step,
            time_offset,
            ..
        } = self.carry.advance(
            rate,
            &self.driver_cfg,
            &self.sad_net,
            audio.data.ncols(),
            non_wav_ssif(audio),
        );

        // getLIDBLSTMParam; mutates lid_window_shift_sec.
        let lid_ssr = self.lid_net.sub_sampling_ratio();
        let (lid_window_size, lid_window_shift, lid_no_overlap) = derive_lid_window(
            self.lid_window_size_sec,
            &mut self.lid_window_shift_sec,
            lid_ssr,
            ssif,
            rate,
        );
        self.lid_net.reset_weights_derivatives();
        self.sad_net.reset_weights_derivatives();

        let class_nb = self.lid_net.output_size().max(2);
        let cost_modified = self.lid_net.is_cost_modified();

        let channels = audio.data.nrows();
        self.cumulative_error = vec![0.0; channels];
        self.nb_of_classif = vec![0; channels];
        self.last_result_rows = Vec::with_capacity(channels);
        self.lid = LidMembers::zeroed(channels);
        self.concat_branch = vec![0; channels];

        let audio_weight = audio.weight;
        let post_process_mode = self.post_process_mode;
        let min_nb_of_frames = self.min_nb_of_frames;
        let noise_magnitude = self.noise_magnitude;
        let dump_lid_internals = self.dump_lid_internals;
        let dump_dir = self.driver_cfg.dump_dir.clone();
        let conv_coeff = self.driver_cfg.conv_coeff.clone();
        let mut lid_cost_ponderation = 1.0f64;
        // External features are shared across channels (phSeq has _ChannelsCount = 1).
        let external_features = audio.external_features.clone();
        let audio_file_name = audio.audio_file_name.clone();

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // targetIndex = clamp(lang_index, [0, classNb)) (`:626-628`).
            let ti = clamp_target_index(audio.lang_index, class_nb);
            let target_index = ti as i32;

            // SAD else branch (`:725`): result_vec.setConstant(10.0); clear SAD outputs.
            let mut result_vec2 = vec![10.0f64; real_vec_size];
            self.sad_net.output_forward = Array2::<f64>::zeros((0, 0));
            self.sad_net.output_backward = Array2::<f64>::zeros((0, 0));
            self.last_result_rows.push(result_vec2.clone());
            // results2segmentation (`:773`, mode 7 runs it: _Mode != 4 && != 6).
            results_to_segmentation(
                seg,
                time_step,
                time_offset,
                &mut result_vec2,
                SegClass::Speech,
                conv_coeff.as_deref(),
                &self.seg_cfg,
            );
            self.cumulative_error[chan] = 0.0; // NNCost (SAD never run) (`:885`)
            self.nb_of_classif[chan] = 0;

            // --- abs(_Mode)==7 external-features loop (`:965-1179`) ------------
            self.lid_net
                .set_processing_type(lid_window_size > 0, !lid_no_overlap); // :971

            let mut dump = if dump_lid_internals && !dump_dir.is_empty() {
                let base = mode7_dump_basename(&audio_file_name, &dump_dir);
                let p = std::path::PathBuf::from(format!("{base}_chan{chan}_lid_dump.mat"));
                Some(crate::io::matfile::MatWriter::create(&p)?)
            } else {
                None
            };
            let mut count_dump = 0i64;

            // The scoring accumulator (`:787-790` seed; `:1113-1135` per entry; `:1163-1179`
            // + `:1318-1337` epilogue): the ONE kernel, `LidChannelAcc`.
            let mut acc = LidChannelAcc::new(class_nb, ti, cost_modified);
            let mut lid_nn_cost = 0.0f64;
            let mut lid_nb_classif = 0i64;

            for feat in &external_features {
                // _MinNbOfFrames + ssr guard (`:981`).
                if feat.nrows() < lid_ssr || (feat.nrows() as i32) < min_nb_of_frames {
                    continue;
                }
                // Gaussian noise (`:1022-1032`). PORT: random_init is wall-clock
                // (`:311`, non-deterministic/non-portable) -> the port fixes randinit = 0
                // (IMPROVEMENTS'd); table lookups are pure so the indexing is exact.
                let mut feat_noised = feat.clone();
                if noise_magnitude > 0.0 {
                    let cols = feat.ncols();
                    let randinit = 0usize;
                    for kk in 0..feat.nrows() {
                        for ll in 0..cols {
                            let m = random_gauss(kk * cols + ll + randinit) - 0.5;
                            feat_noised[[kk, ll]] += noise_magnitude * m;
                        }
                    }
                }

                // feedForward (`:1034`): modifier = 1/feat.rows().
                let modifier = 1.0 / feat.nrows() as f64;
                let output_seq = self.lid_net.feed_forward_scoring(
                    &mut feat_noised,
                    lid_window_size,
                    lid_window_shift,
                    target_index as i64,
                    modifier,
                );
                lid_nn_cost += self.lid_net.cost; // :1035
                lid_nb_classif += self.lid_net.nb_of_classif; // :1036

                // segLID per _PostProcessMode (`:1073-1108`; the `_Mode < 0` sub-branch
                // `:1040-1071` is dead for _Mode = +7), argmax + confusion (`:1113-1130`),
                // langID (`:1131-1135`): the kernel.
                acc.push(
                    output_seq.view(),
                    BlockScore::PostProcess(post_process_mode),
                    BlockWeight::MeanIfCostModified,
                );

                // DumpLIDInternals features_<n> = [oF | oB] (`:1136-1144`).
                if let Some(w) = dump.as_mut() {
                    let of = &self.lid_net.output_forward;
                    let ob = &self.lid_net.output_backward;
                    let n = of.nrows();
                    let (fc, bc) = (of.ncols(), ob.ncols());
                    let mut colmaj = Vec::with_capacity(n * (fc + bc));
                    for c in 0..fc {
                        for r in 0..n {
                            colmaj.push(of[[r, c]]);
                        }
                    }
                    for c in 0..bc {
                        for r in 0..n {
                            colmaj.push(ob[[r, c]]);
                        }
                    }
                    w.write_matrix(&format!("features_{count_dump}"), n, fc + bc, &colmaj)?;
                    count_dump += 1;
                }
            }

            if let Some(mut w) = dump.take() {
                w.write_scalar("matNb", count_dump as f64)?; // :1157-1158
                w.finish()?;
            }

            // langID normalization (`:1163-1179`) + member writes (`:1318-1337`).
            let agg = acc.finalize(LidNorm::post_process(post_process_mode));
            self.lid.cumulative_error[chan] = lid_nn_cost;
            self.lid.nb_of_classif[chan] = lid_nb_classif;
            self.lid.classification_errors[chan] = agg.classification_errors;
            self.lid.segments_confusion[chan] = agg.confusion;
            self.lid.is_correct[chan] = agg.is_lid_correct;

            lid_cost_ponderation = self.lid_net.get_cost_ponderation(target_index as i64); // :1334
            self.lid_net.ponderate_weights_derivatives(audio_weight); // :1335
            self.lid.cumulative_error[chan] *= audio_weight; // :1336
            self.cumulative_error[chan] *= lid_cost_ponderation; // :1337
        }

        // Epilogue (`:1393-1420`).
        self.sad_net
            .ponderate_weights_derivatives(lid_cost_ponderation);
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }
        self.carry.reset_after(no_overlap);
        if lid_no_overlap {
            self.lid_window_shift_sec = 0.0;
        }
        Ok(())
    }

    /// Port of the modes 4/5/6 wav path: the LID-train branch (`:640-692`), the SAD-side
    /// synthesis (`:694-796`), the `:798-902` scoring branch, and the shared member writes
    /// (`:1318-1337`) + epilogue (`:1393-1420`).
    ///
    /// UNLIKE modes 0/1/2/3, the LID net runs `feedForwardBackward` ONCE over the WHOLE
    /// inputSeq (`:664`), producing `LID_result_vec`; the scoring loop then reads
    /// `LID_result_vec` blocks sliced by `LIDTimeStep`/`LIDTimeOffset` (`:811-814`, offset
    /// SUBTRACTED before the division -- the load-bearing contrast vs the `:1245` branch's
    /// bare `begin/_SpectrumShift`), with NO further NN forward. Per-mode classification:
    /// mode 4 = REFERENCE smoothed on top (`:779-783`); mode 5 = SAD BLSTM VAD
    /// (`results2segmentation`, `:713-717,:773`); mode 6 = `LID2Segmentation` over
    /// `1 - LID_result_vec.col(0)` (`:678-691`). The `:798-902` accumulator matches the
    /// mode-7 branch's three `_PostProcessMode` forms (`langID += isCostModified ?
    /// segLID/rows : segLID`; post-loop `/= isCostModified ? segmentsCount : numberOfFrames`,
    /// `PPM != 2 -> exp`, row-normalize).
    fn get_segmentation_mode456(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        let rate = audio.sample_rate as f64;
        let s = SpectralParams::derive(&self.feature_cfg, rate);
        if let Some(tdc) = s.tdc.as_ref()
            && tdc.half_window > 0
        {
            return Err(anyhow!(
                "TwinBlstmSpectralLid mode {}: pitch second pass (TDCwindow > 0) not ported",
                self.mode
            ));
        }

        // The input gates first (they mutate the audio in place and read nothing the carry
        // writes), then the carry's preamble (#121): the stateful spectrum-shift quantization
        // (`:208-210`), getBLSTMParam (SAD window/shift + result-vec sizing) and the SAD time
        // axis (`:696-706`, overlap OVERRIDES with _SpectrumShift).
        apply_input_gates(audio, &self.feature_cfg);
        let SpectralAxis {
            ssif: spectrum_shift_in_frames,
            spectrum_shift_sec,
            window_size,
            window_shift,
            no_overlap,
            real_vec_size,
            time_step,
            time_offset,
        } = self.carry.advance(
            rate,
            &self.driver_cfg,
            &self.sad_net,
            audio.data.ncols(),
            non_wav_ssif(audio),
        );

        // getLIDBLSTMParam (LID window/shift + LID_result_vec sizing); mutates
        // lid_window_shift_sec + resets the LID net derivs (`:87-137`).
        let lid_ssr = self.lid_net.sub_sampling_ratio();
        let (lid_window_size, lid_window_shift, lid_no_overlap) = derive_lid_window(
            self.lid_window_size_sec,
            &mut self.lid_window_shift_sec,
            lid_ssr,
            spectrum_shift_in_frames,
            rate,
        );
        self.lid_net.reset_weights_derivatives();
        let frame_count = audio.data.ncols();
        let vec_size = if frame_count.is_multiple_of(spectrum_shift_in_frames) {
            frame_count / spectrum_shift_in_frames
        } else {
            frame_count / spectrum_shift_in_frames + 1
        };
        let mut lid_real_vec_size = vec_size;
        if lid_ssr > 1 {
            for r in self.lid_net.lstm_sub_sampling() {
                lid_real_vec_size /= r;
            }
            for r in self.lid_net.output_sub_sampling() {
                lid_real_vec_size /= r;
            }
        }
        self.sad_net.reset_weights_derivatives();

        // LIDTimeStep/LIDTimeOffset (`:629-638`): overlap keeps _SpectrumShift * lidSsr.
        // `lid_window_shift_sec` is the POST-getLIDBLSTMParam value.
        let lid_window_shift_sec = self.lid_window_shift_sec;
        let mut lid_time_step = lid_window_shift_sec * lid_ssr as f64;
        let mut lid_time_offset = lid_time_step / 2.0 - lid_window_shift_sec / 2.0;
        if lid_window_size > 0 {
            if lid_no_overlap {
                lid_time_step = lid_window_shift_sec * lid_ssr as f64;
                lid_time_offset = lid_time_step / 2.0 - lid_window_shift_sec / 2.0;
            } else {
                lid_time_step = spectrum_shift_sec * lid_ssr as f64;
                lid_time_offset = lid_time_step / 2.0 - spectrum_shift_sec / 2.0;
            }
        }

        let class_nb = self.lid_net.output_size().max(2);
        let lid_out_size = self.lid_net.output_size();
        let cost_modified = self.lid_net.is_cost_modified();

        let temporal_conv = windowing_coefficients(
            &self.feature_cfg.conv_type,
            true,
            2 * self.feature_cfg.conv_size as usize + 1,
            0.83333,
        );

        let channels = audio.data.nrows();
        self.cumulative_error = vec![0.0; channels];
        self.nb_of_classif = vec![0; channels];
        self.last_result_rows = Vec::with_capacity(channels);
        self.lid = LidMembers::zeroed(channels);
        // Modes 4/5/6 never reach the `:1221` concat (that lives in the `:1194` branch);
        // record 0 (not-called) so the observation point stays meaningful cross-mode.
        self.concat_branch = vec![0; channels];

        let audio_weight = audio.weight;
        let mode = self.mode;
        let post_process_mode = self.post_process_mode;
        let lid_decision_thresh_rising = self.lid_decision_thresh_rising;
        let seg_cfg = self.seg_cfg.clone();
        let conv_coeff = self.driver_cfg.conv_coeff.clone();
        let back_prop_wer = self.driver_cfg.back_prop_wer;
        let mut lid_cost_ponderation = 1.0f64;

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // SAD input (base spectral pipeline).
            let (mut input_seq, _perio, _ltsv1) = build_input_sequence_parts(
                audio,
                &self.feature_cfg,
                &s,
                chan,
                temporal_conv.as_deref(),
            );

            let ti = clamp_target_index(audio.lang_index, class_nb);
            let target_index = ti as i32;

            // --- LID-train branch (`:640-692`) ---------------------------------
            let mut lid_result_vec = Array2::<f64>::zeros((lid_real_vec_size, lid_out_size));

            // LIDtargetSeq (`:658-662`): built over the SMOOTHED reference classification
            // (`:660-661`), then the hypothesis is cleared again (`:663`). With no
            // reference the classification copy is empty, so getTargetsLID leaves zeros.
            let has_ref = matches!(refs, Some(rs) if !rs[chan].segments().is_empty());
            let lid_target_seq = if has_ref {
                let mut temp = refs.unwrap()[chan].clone();
                smooth_segmentation(&mut temp, &seg_cfg);
                self.get_targets_lid(&temp, lid_time_step, lid_time_offset, ti, lid_real_vec_size)
            } else {
                Array2::<f64>::zeros((lid_real_vec_size, lid_out_size.max(1)))
            };
            seg.clear_hypothesis(); // `:663`

            // feedForwardBackward on the WHOLE inputSeq (`:641,:664`). Normalizes inputSeq
            // in place (type -1) -- mode 5's SAD net below sees the LID-normalized input.
            self.lid_net
                .set_processing_type(lid_window_size > 0, !lid_no_overlap);
            self.lid_net.feed_forward_backward(
                &mut input_seq,
                lid_window_size,
                lid_window_shift,
                &mut lid_result_vec,
                &lid_target_seq,
            );
            let lid_nn_cost = self.lid_net.cost; // `:665`
            let lid_nb_classif = self.lid_net.nb_of_classif; // `:666`

            // mode 6: LID2Segmentation over 1 - LID_result_vec.col(0) (`:678-691`).
            if mode == 6 {
                seg.clear_hypothesis(); // `:679`
                let inv: Vec<f64> = (0..lid_result_vec.nrows())
                    .map(|r| 1.0 - lid_result_vec[[r, 0]])
                    .collect();
                lid_to_segmentation(
                    seg,
                    &inv,
                    SegClass::Speech,
                    lid_time_offset,
                    lid_time_step,
                    lid_decision_thresh_rising,
                );
            }

            // --- SAD side (`:694-796`) -----------------------------------------
            // SAD targetSeq (`:707-711`): computed when a reference exists (used only by
            // mode 5's SAD FFB; harmless for 4/6). getTargets reads the REFERENCE.
            let sad_target = if has_ref {
                let rs = refs.unwrap();
                let col = get_targets(
                    seg,
                    &rs[chan],
                    time_step,
                    time_offset,
                    back_prop_wer,
                    SegClass::Speech,
                    real_vec_size,
                );
                let mut t = Array2::from_shape_vec((real_vec_size, 1), col).unwrap();
                if lid_out_size == 1 && target_index != 1 {
                    t.fill(0.0);
                }
                t
            } else {
                Array2::<f64>::zeros((0, 0))
            };

            // result_vec + NNCost/nbOfClassif per mode (`:713-764`).
            let mut nn_cost = 0.0f64;
            let mut nb_of_classif = 0i64;
            let result_vec: Array2<f64> = if mode == 5 {
                // mode 0/3/5 branch (`:713-717`): SAD net FFB on the (LID-normalized)
                // inputSeq. topRows tolerance handles the 23-input net on narrow features.
                self.sad_net
                    .set_processing_type(window_size > 0, !no_overlap);
                let mut rv = Array2::<f64>::zeros((real_vec_size, 1));
                self.sad_net.feed_forward_backward(
                    &mut input_seq,
                    window_size,
                    window_shift,
                    &mut rv,
                    &sad_target,
                );
                nn_cost = self.sad_net.cost;
                nb_of_classif = self.sad_net.nb_of_classif;
                rv
            } else if mode == 4 {
                // `:719-724`: result_vec = LID_result_vec.col(targetIndex) (or the whole
                // single column when the LID net is binary). SAD outputs cleared (`:762`).
                self.sad_net.output_forward = Array2::<f64>::zeros((0, 0));
                self.sad_net.output_backward = Array2::<f64>::zeros((0, 0));
                let col = if lid_out_size == 1 { 0 } else { ti };
                let mut rv = Array2::<f64>::zeros((lid_result_vec.nrows(), 1));
                for r in 0..lid_result_vec.nrows() {
                    rv[[r, 0]] = lid_result_vec[[r, col]];
                }
                rv
            } else {
                // mode 6 (`:727-732`): result_vec = 1 - LID_result_vec.col(0). The timeStep/
                // threshold overwrites at `:728-732` are DEAD (results2segmentation is
                // skipped for mode 6, and the scoring branch reads LIDTimeStep, not
                // timeStep) -- reproduced only as the dumped result_vec row. SAD outputs
                // cleared (`:762`).
                self.sad_net.output_forward = Array2::<f64>::zeros((0, 0));
                self.sad_net.output_backward = Array2::<f64>::zeros((0, 0));
                let mut rv = Array2::<f64>::zeros((lid_result_vec.nrows(), 1));
                for r in 0..lid_result_vec.nrows() {
                    rv[[r, 0]] = 1.0 - lid_result_vec[[r, 0]];
                }
                rv
            };

            // result_vec2 = transpose -> ROW vector (`:766`); the port observation point.
            let mut result_vec2: Vec<f64> = result_vec.column(0).to_vec();
            self.last_result_rows.push(result_vec2.clone());

            // results2segmentation for mode != 4 && != 6 -> only mode 5 (`:773`).
            if mode == 5 {
                results_to_segmentation(
                    seg,
                    time_step,
                    time_offset,
                    &mut result_vec2,
                    SegClass::Speech,
                    conv_coeff.as_deref(),
                    &seg_cfg,
                );
            }
            self.cumulative_error[chan] = nn_cost; // `:774`
            self.nb_of_classif[chan] = nb_of_classif; // `:775`

            // mode 4: classification = REFERENCE smoothed (`:779-783`).
            if mode == 4 {
                seg.clear_hypothesis(); // `:780`
                if has_ref {
                    seg.set_segments_from(&refs.unwrap()[chan]); // `:781`
                }
                smooth_segmentation(seg, &seg_cfg); // `:782`
            }

            // --- scoring branch (`:798-902`) -----------------------------------
            // totalSpeechDuration (`:799-806`) is DEAD for this branch (the post-loop
            // normalizes by segmentsCount/numberOfFrames, `:888-892`) -- skipped.
            let segs = seg.segments().to_vec();
            // The scoring accumulator (`:787-790` seed): the one kernel, `LidChannelAcc`.
            let mut acc = LidChannelAcc::new(class_nb, ti, cost_modified);
            let lid_rows = lid_result_vec.nrows();

            for i in 0..segs.len().saturating_sub(1) {
                if segs[i].ty != SegClass::Speech {
                    continue;
                }
                // rowBegin = ceil((begin - LIDTimeOffset)/LIDTimeStep) (`:811-812`);
                // rowEnd = floor((next.begin - LIDTimeOffset)/LIDTimeStep) (`:813`).
                let ratio = (segs[i].begin - lid_time_offset) / lid_time_step;
                let mut row_begin = ratio as usize;
                if (row_begin as f64) != ratio {
                    row_begin += 1;
                }
                let mut row_end = ((segs[i + 1].begin - lid_time_offset) / lid_time_step) as usize;
                if row_end >= lid_rows {
                    row_end = lid_rows - 1; // `:814`
                }
                // `:815` guard (`rowEnd >= rowBegin`).
                if row_end < row_begin {
                    continue;
                }
                let n_rows = row_end - row_begin + 1;

                // outputSeq = LID_result_vec.block(rowBegin, 0, n_rows, cols) (`:816`), with
                // the cols==1 binary expansion (`:817-821`): col1 = p, col0 = 1-p.
                let out_cols = if lid_out_size == 1 { 2 } else { lid_out_size };
                let mut output_seq = Array2::<f64>::zeros((n_rows, out_cols));
                if lid_out_size == 1 {
                    for r in 0..n_rows {
                        let p = lid_result_vec[[row_begin + r, 0]];
                        output_seq[[r, 0]] = 1.0 - p;
                        output_seq[[r, 1]] = p;
                    }
                } else {
                    for r in 0..n_rows {
                        for c in 0..out_cols {
                            output_seq[[r, c]] = lid_result_vec[[row_begin + r, c]];
                        }
                    }
                }

                // segLID per _PostProcessMode (`:823-859`), argmax + confusion (`:862-874`;
                // `score` `:860-861` is a dead local, not reproduced), langID (`:875-879`):
                // the kernel.
                acc.push(
                    output_seq.view(),
                    BlockScore::PostProcess(post_process_mode),
                    BlockWeight::MeanIfCostModified,
                );
            }

            // `:885-886` (redundant with `:774-775`).
            self.cumulative_error[chan] = nn_cost;
            self.nb_of_classif[chan] = nb_of_classif;

            // post-loop reductions (`:887-902`) + member writes (`:1318-1337`).
            let agg = acc.finalize(LidNorm::post_process(post_process_mode));
            self.lid.cumulative_error[chan] = lid_nn_cost;
            self.lid.nb_of_classif[chan] = lid_nb_classif;
            self.lid.classification_errors[chan] = agg.classification_errors;
            self.lid.segments_confusion[chan] = agg.confusion;
            self.lid.is_correct[chan] = agg.is_lid_correct;

            lid_cost_ponderation = self.lid_net.get_cost_ponderation(target_index as i64);
            self.lid_net.ponderate_weights_derivatives(audio_weight);
            self.lid.cumulative_error[chan] *= audio_weight;
            self.cumulative_error[chan] *= lid_cost_ponderation;
        }

        // Epilogue (`:1393-1420`).
        self.sad_net
            .ponderate_weights_derivatives(lid_cost_ponderation);
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }
        self.carry.reset_after(no_overlap);
        if lid_no_overlap {
            self.lid_window_shift_sec = 0.0;
        }
        Ok(())
    }
}

/// `getLIDBLSTMParam` (`TwinBLSTMSpectralLID.cpp:87-137`), the LID window derivation:
/// `(lid_window_size, lid_window_shift, lid_no_overlap)` in periodogram frames, and the
/// stateful `_LIDWindowShift` written back through `shift_sec` as `shift * ssif / rate`
/// (`:102`), exactly like [`get_blstm_param`](crate::tasks::sad::get_blstm_param) writes
/// `_WindowShift`. The one body of the three Twin modes (issue #121; the fast tree's plan
/// calls it with a local it discards, the dropped carry of #84). The derivative reset the
/// legacy does inside (`:110`) stays with the caller.
pub fn derive_lid_window(
    size_sec: f64,
    shift_sec: &mut f64,
    lid_ssr: usize,
    ssif: usize,
    rate: f64,
) -> (usize, usize, bool) {
    let ssifd = ssif as f64;
    let mut lid_window_size = f64::round(size_sec * rate / 2.0 / ssifd) as usize;
    if lid_window_size != 0 && lid_window_size < lid_ssr {
        lid_window_size = lid_ssr;
    }
    let mut lid_window_shift = f64::round(*shift_sec * rate / ssifd) as i64;
    let mut lid_no_overlap = false;
    if lid_window_size != 0 && lid_window_shift < 1 {
        lid_no_overlap = true;
        let raw = f64::round(size_sec * rate / ssifd) as usize;
        lid_window_size = (raw / lid_ssr) * lid_ssr;
        if lid_window_size < 10 * lid_ssr {
            lid_window_size = 10 * lid_ssr;
        }
    }
    if lid_window_size == 0 || lid_window_shift < 1 {
        lid_window_shift = 1;
    }
    *shift_sec = (lid_window_shift * ssif as i64) as f64 / rate;
    (lid_window_size, lid_window_shift as usize, lid_no_overlap)
}

impl Segmenter for TwinBlstmSpectralLid {
    /// Port of `TwinBLSTMSpectralLID::getSegmentation` (`:263-1421`), the wav-mode
    /// 0/1/2/3 non-unit-test/non-plotting paths (the `.mat`/PNG/log branches dropped).
    /// Modes 4/5/6/7 and the pitch second pass BAIL typed (see the struct doc). `refs`
    /// maps to `seg._Reference`: required for modes 2/3 (the scoring loop iterates it)
    /// and, for any mode with a reference, builds the SAD net's `targetSeq` (`:707-711`).
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        // Mode 7 (the `abs(_Mode) == 7` branch, `:903-1193`): the phSeq external-features
        // LID loop. Only the File_Type-1 (phSeq) arm is ported -- the wav arm (`:922-963`)
        // runs the CNN (`:927`, NOT ported) so it bails. Handled in its own method.
        if self.mode == 7 {
            return self.get_segmentation_mode7(audio, seg_per_chan, refs);
        }
        // Modes 0-6 on a phSeq / cep read feed the precomputed features to the nets (`:354`,
        // `:642`), an arm not ported: refused before the carry moves (issue #121).
        if audio.periodogram.is_some() {
            return Err(anyhow!(
                "TwinBlstmSpectralLid mode {}: non-wav input (File_Type 1/2) not ported: the \
                 legacy feeds the precomputed features to the nets (legacy :354, :642); only \
                 Mode 7 runs a phSeq / cep read",
                self.mode
            ));
        }
        // Modes 4/5/6 (the LID-train branch `:640-692` + the `:798-902` scoring branch):
        // the LID net trains on the WHOLE inputSeq (`feedForwardBackward`, :664) and the
        // scoring slices LID_result_vec per speech segment (NOT a per-segment forward).
        if self.mode == 4 || self.mode == 5 || self.mode == 6 {
            return self.get_segmentation_mode456(audio, seg_per_chan, refs);
        }
        if !(self.mode == 0 || self.mode == 1 || self.mode == 2 || self.mode == 3) {
            return Err(anyhow!(
                "TwinBlstmSpectralLid: mode {} not ported",
                self.mode
            ));
        }
        let rate = audio.sample_rate as f64;

        // initSpectralAnalysis param derivation (`:281`) via the base spectral machinery.
        let s = SpectralParams::derive(&self.feature_cfg, rate);

        // The pitch second pass (`:349-614`, TDC_window_size > 0) is a REFERENCE-based
        // pre-forward warp that DIVERGES from the base driver's post-forward pitch pass --
        // deferred (bail typed). All ported configs use TDCwindow 0.
        if let Some(tdc) = s.tdc.as_ref()
            && tdc.half_window > 0
        {
            return Err(anyhow!(
                "TwinBlstmSpectralLid: pitch second pass (TDCwindow > 0) not ported (legacy :349-614)"
            ));
        }

        // The input gates (`initSpectralAnalysis` :216-227) first -- they mutate the audio in
        // place and read nothing the carry writes -- then the carry's preamble (#121): the
        // stateful spectrum-shift quantization (`:208-210`), getBLSTMParam (`:300` ->
        // `:439-500`, SAD window/shift + result-vec sizing) and the SAD time axis
        // (`:696-706`, overlap OVERRIDES with _SpectrumShift).
        apply_input_gates(audio, &self.feature_cfg);
        let SpectralAxis {
            ssif: spectrum_shift_in_frames,
            spectrum_shift_sec,
            window_size,
            window_shift,
            no_overlap,
            real_vec_size,
            time_step,
            time_offset,
        } = self.carry.advance(
            rate,
            &self.driver_cfg,
            &self.sad_net,
            audio.data.ncols(),
            non_wav_ssif(audio),
        );

        // getLIDBLSTMParam (`:306` -> `:87-137`): LID window/shift + LID_result_vec
        // sizing; mutates self.lid_window_shift_sec + resets the LID net derivs.
        let lid_ssr = self.lid_net.sub_sampling_ratio();
        let (lid_window_size, lid_window_shift, lid_no_overlap) = derive_lid_window(
            self.lid_window_size_sec,
            &mut self.lid_window_shift_sec,
            lid_ssr,
            spectrum_shift_in_frames,
            rate,
        );
        self.lid_net.reset_weights_derivatives(); // :110
        let frame_count = audio.data.ncols();
        let vec_size = if frame_count.is_multiple_of(spectrum_shift_in_frames) {
            frame_count / spectrum_shift_in_frames
        } else {
            frame_count / spectrum_shift_in_frames + 1
        };
        let mut lid_real_vec_size = vec_size;
        if lid_ssr > 1 {
            for r in self.lid_net.lstm_sub_sampling() {
                lid_real_vec_size /= r;
            }
            for r in self.lid_net.output_sub_sampling() {
                lid_real_vec_size /= r;
            }
        }

        // resetWeightsDerivatives on the SAD net (`:234` via getBLSTMParam already reset,
        // but the base getBLSTMParam :473 resets -- our get_blstm_param does NOT, so reset
        // here to match).
        self.sad_net.reset_weights_derivatives();

        let class_nb = self.lid_net.output_size().max(2);

        let temporal_conv = windowing_coefficients(
            &self.feature_cfg.conv_type,
            true,
            2 * self.feature_cfg.conv_size as usize + 1,
            0.83333,
        );

        let channels = audio.data.nrows();
        self.cumulative_error = vec![0.0; channels];
        self.nb_of_classif = vec![0; channels];
        self.last_result_rows = Vec::with_capacity(channels);
        self.lid = LidMembers::zeroed(channels);
        self.concat_branch = vec![0; channels];

        let audio_weight = audio.weight;
        let mut lid_cost_ponderation = 1.0f64;

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // SAD input (`:359-619`): the BASE spectral pipeline (periodogram -> mel/DCT
            // -> LTSV) -- identical to `BlstmSpectralSegmenter` when the pitch pass is off.
            let (mut input_seq, _perio, _ltsv1) = build_input_sequence_parts(
                audio,
                &self.feature_cfg,
                &s,
                chan,
                temporal_conv.as_deref(),
            );

            // targetIndex = clamp(audio.getRefLangIndex()) to [0, classNb) (`:626-628`).
            let ti = clamp_target_index(audio.lang_index, class_nb);
            let target_index = ti as i32;

            // SAD targetSeq (`:707-711`): built when a reference exists; :710 zeroes it
            // for a BINARY LID net when targetIndex != 1.
            let has_ref = matches!(refs, Some(rs) if !rs[chan].segments().is_empty());
            let target = if has_ref {
                let rs = refs.unwrap();
                let col = get_targets(
                    seg,
                    &rs[chan],
                    time_step,
                    time_offset,
                    self.driver_cfg.back_prop_wer,
                    SegClass::Speech,
                    real_vec_size,
                );
                let mut t = Array2::from_shape_vec((real_vec_size, 1), col).unwrap();
                if self.lid_net.output_size() == 1 && target_index != 1 {
                    t.fill(0.0);
                }
                t
            } else {
                Array2::<f64>::zeros((0, 0))
            };

            // SAD result_vec: modes 0/3 run the SAD net; else (modes 1/2) synthesize.
            let mut result_vec = Array2::<f64>::zeros((real_vec_size, 1));
            let mut nn_cost = 0.0f64;
            let mut nb_of_classif = 0i64;
            if self.mode == 0 || self.mode == 3 {
                // `:713-717` SAD net feedForwardBackward (populates _OutputForward/Backward).
                self.sad_net
                    .set_processing_type(window_size > 0, !no_overlap);
                self.sad_net.feed_forward_backward(
                    &mut input_seq,
                    window_size,
                    window_shift,
                    &mut result_vec,
                    &target,
                );
                nn_cost = self.sad_net.cost;
                nb_of_classif = self.sad_net.nb_of_classif;
            } else {
                // `:718-764` else branch. Mode 2: result_vec = LID_result_vec (all-zero,
                // `:719-724`). Mode 1: result_vec.setConstant(10.0) (`:726`). Both CLEAR
                // the SAD net outputs (`:762-763`).
                if self.mode == 2 {
                    // LID_result_vec is Zero(lid_real_vec_size, lid_output_size); col
                    // (targetIndex) or the whole matrix (cols==1) -> a zero column.
                    result_vec = Array2::<f64>::zeros((lid_real_vec_size, 1));
                } else {
                    // mode 1
                    result_vec.fill(10.0);
                }
                self.sad_net.output_forward = Array2::<f64>::zeros((0, 0));
                self.sad_net.output_backward = Array2::<f64>::zeros((0, 0));
            }

            // result_vec2 = transpose -> ROW vector (`:766`).
            let mut result_vec2: Vec<f64> = result_vec.column(0).to_vec();
            self.last_result_rows.push(result_vec2.clone());

            // results2segmentation (`:773`, for modes != 4,6 -> yes for 0/1/2/3).
            results_to_segmentation(
                seg,
                time_step,
                time_offset,
                &mut result_vec2,
                SegClass::Speech,
                self.driver_cfg.conv_coeff.as_deref(),
                &self.seg_cfg,
            );
            // :774-775.
            self.cumulative_error[chan] = nn_cost;
            self.nb_of_classif[chan] = nb_of_classif;

            // ---- LID scoring branch (`:1194-1310`) --------------------------------
            self.lid_net
                .set_processing_type(lid_window_size > 0, !lid_no_overlap); // :1206

            // Concat (`:1221`): LID input size > feature width -> augment.
            self.concat_branch[chan] = if self.lid_net.input_size() > input_seq.ncols() {
                let (aug, branch) = self.get_blstm_lid_input_sequence(&input_seq);
                input_seq = aug;
                branch
            } else {
                0
            };

            // Scoring iterator: REFERENCE for modes 2/3, CLASSIFICATION for 0/1 (`:1223-1239`).
            let scoring_seg: &Segmentation = if (self.mode == 2 || self.mode == 3) && has_ref {
                &refs.unwrap()[chan]
            } else {
                seg
            };
            let scoring_segs = scoring_seg.segments().to_vec();

            // totalSpeechDuration over the scoring iterator (`:1229-1234`).
            let total_speech_duration = speech_duration(scoring_seg);

            // interestSegs (`:1240`) + the TrainingPruning gate (`:1286`, mutating
            // `itInterest->_Type`) are NOT reproduced: `interestSegs` is consumed ONLY by
            // the VRCTS dump (skipped) + the mode!=0 replacement (`:1297`, itself
            // dump-only), so both the copy and the pruning are observably dead in this
            // port. The gate is config-gated on `_LIDTrainingPruningThreshold > 0`
            // (default -1.0) besides -- see IMPROVEMENTS.md.

            // segmentationLID (`:1242`): midpoint seed.
            let mut segmentation_lid =
                vec![
                    (self.lid_decision_thresh_rising + self.lid_decision_thresh_falling) / 2.0;
                    lid_real_vec_size
                ];

            // The scoring accumulator (`:787-790` seed): the one kernel, `LidChannelAcc`.
            let mut acc = LidChannelAcc::new(class_nb, ti, self.lid_net.is_cost_modified());
            let mut lid_nn_cost = 0.0f64;
            let mut lid_nb_classif = 0i64;

            for i in 0..scoring_segs.len().saturating_sub(1) {
                if scoring_segs[i].ty != SegClass::Speech {
                    continue;
                }
                // rowBegin = ceil(begin / _SpectrumShift), rowEnd = floor(next / _SpectrumShift)
                // (`:1245-1247`). NO offset.
                let ratio = scoring_segs[i].begin / spectrum_shift_sec;
                let mut row_begin = ratio as usize;
                if (row_begin as f64) != ratio {
                    row_begin += 1;
                }
                let row_end = (scoring_segs[i + 1].begin / spectrum_shift_sec) as usize;
                // `:1248` guard (unsigned; the row_end >= row_begin guard prevents underflow).
                if row_end < row_begin || row_end - row_begin + 1 < lid_ssr {
                    continue;
                }
                let n_rows = row_end - row_begin + 1;
                if row_begin + n_rows > input_seq.nrows() {
                    continue; // block would read past the input (legacy UB-slices; guarded).
                }

                // inputSeqLID = inputSeq.block(rowBegin, 0, n_rows, cols) (`:1249`).
                let cols = input_seq.ncols();
                let mut input_seq_lid = Array2::<f64>::zeros((n_rows, cols));
                for r in 0..n_rows {
                    for c in 0..cols {
                        input_seq_lid[[r, c]] = input_seq[[row_begin + r, c]];
                    }
                }

                // Scoring feedForward (`:1250`): modifier = timeStep / segment duration.
                let dur = scoring_segs[i + 1].begin - scoring_segs[i].begin;
                let output_seq = self.lid_net.feed_forward_scoring(
                    &mut input_seq_lid,
                    lid_window_size,
                    lid_window_shift,
                    target_index as i64,
                    time_step / dur,
                );

                // segmentationLID block = InvLogistic(outputSeq.col(1)) (`:1251`).
                let start = row_begin / lid_ssr;
                for r in 0..output_seq.nrows() {
                    if start + r < segmentation_lid.len() && output_seq.ncols() > 1 {
                        segmentation_lid[start + r] = inv_logistic_fn(output_seq[[r, 1]]);
                    }
                }

                lid_nn_cost += self.lid_net.cost; // :1252
                lid_nb_classif += self.lid_net.nb_of_classif; // :1253

                // segLID = colwise sum / rows (`:1254`), argmax j (`:1257-1258`) + confusion
                // (`:1265-1274`), langID (`:1275-1279`: cost-modified += segLID, else
                // += dur*segLID): the kernel.
                acc.push(
                    output_seq.view(),
                    BlockScore::ColumnMean,
                    BlockWeight::DurationUnlessCostModified { dur },
                );
            }

            // segmentationLID /= 56 (`:1292`).
            for v in segmentation_lid.iter_mut() {
                *v /= 56.0;
            }

            // mode != 0 -> clearClassification + LID2Segmentation (`:1294-1298`), inside the
            // `segmentsCount > 0` branch ahead of the normalization; it reads no langID, so
            // it runs before `finalize` with nothing observable reordered.
            if acc.segments_count() > 0 && self.mode != 0 {
                seg.clear_hypothesis();
                lid_to_segmentation(
                    seg,
                    &segmentation_lid,
                    SegClass::Speech,
                    time_offset,
                    time_step,
                    self.lid_decision_thresh_rising,
                );
            }

            // langID normalization (`:1299-1305`) + member writes (`:1318-1337`).
            let agg = acc.finalize(LidNorm::Duration {
                total_speech_duration,
            });
            self.lid.cumulative_error[chan] = lid_nn_cost; // :1320
            self.lid.nb_of_classif[chan] = lid_nb_classif; // :1321
            self.lid.classification_errors[chan] = agg.classification_errors; // :1322
            self.lid.segments_confusion[chan] = agg.confusion; // :1324
            self.lid.is_correct[chan] = agg.is_lid_correct; // :1325-1329

            lid_cost_ponderation = self.lid_net.get_cost_ponderation(target_index as i64); // :1334
            self.lid_net.ponderate_weights_derivatives(audio_weight); // :1335
            self.lid.cumulative_error[chan] *= audio_weight; // :1336
            self.cumulative_error[chan] *= lid_cost_ponderation; // :1337
        }

        // Epilogue (`:1393-1420`): SAD net ponderate by the LAST channel's LIDCostPonderation.
        self.sad_net
            .ponderate_weights_derivatives(lid_cost_ponderation); // :1393
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1); // :1394
        }
        self.carry.reset_after(no_overlap); // :1419
        if lid_no_overlap {
            self.lid_window_shift_sec = 0.0; // :1420
        }

        Ok(())
    }

    fn dump_dir(&self) -> &str {
        &self.driver_cfg.dump_dir
    }

    /// PRE-convolution SAD `result_vec2` rows (one per channel) from the last call.
    fn last_result_rows(&self) -> &[Vec<f64>] {
        &self.last_result_rows
    }

    fn pack_count(&self) -> usize {
        2
    }

    /// `setWeights` then `setWeightsLID` (`BagOfProcessors.cpp:110-113`).
    fn set_packs(&mut self, packs: &[Vec<f64>]) -> Result<()> {
        self.sad_net.set_weights(&packs[0])?;
        self.lid_net.set_weights(&packs[1])
    }

    fn nets(&self) -> Vec<&BlstmNetwork> {
        vec![&self.sad_net, &self.lid_net]
    }

    fn nets_mut(&mut self) -> Vec<&mut BlstmNetwork> {
        vec![&mut self.sad_net, &mut self.lid_net]
    }

    fn channel_members(&self, chan: usize) -> ChannelMembers<'_> {
        ChannelMembers {
            cumulative_error: self.cumulative_error[chan],
            nb_of_classif: self.nb_of_classif[chan],
            lid: Some(self.lid.channel(chan)),
        }
    }
}

#[cfg(test)]
mod twin_tests {
    use super::*;
    use crate::tasks::segmentation::Segmentation;

    /// The kernel's argmax is FIRST-max (strict `>`): on a tied `segLID` the lower class
    /// wins the confusion cell, as every legacy copy's inline loop did. No committed fixture
    /// ties exactly (the `issue83/5` battery row), so the fact needs its own pin.
    #[test]
    fn lid_channel_acc_argmax_is_first_max_on_a_tie() {
        let mut acc = LidChannelAcc::new(2, 1, false);
        // Vote mode: row 0 votes class 0, row 1 votes class 1 -> segLID = [1, 1], a tie.
        let out = ndarray::arr2(&[[0.9, 0.1], [0.1, 0.9]]);
        let o = acc.push(out.view(), BlockScore::PostProcess(2), BlockWeight::Plain);
        assert_eq!(o.seg_lid, vec![1.0, 1.0]);
        assert_eq!(o.argmax, 0, "first max, not last");
        // target 1 (row 2), predicted 0 (col 1): a miss cell, nothing on the diagonal.
        let agg = acc.finalize(LidNorm::PostProcess { exp: false });
        assert_eq!(agg.confusion[[2, 1]], 1.0);
        assert_eq!(agg.confusion[[2, 2]], 0.0);
        assert_eq!(agg.predicted_language, 0);
    }

    #[test]
    fn clamp_target_index_folds_out_of_range_and_negative_to_zero() {
        assert_eq!(clamp_target_index(0, 3), 0);
        assert_eq!(clamp_target_index(2, 3), 2);
        assert_eq!(clamp_target_index(3, 3), 0);
        assert_eq!(clamp_target_index(-1, 3), 0);
    }

    /// `getTargetsLID` (`:170-219`) enforcement pattern with `TargetEnforcementStep 2`:
    /// SPEECH rows are `-0.5` until `counter == step` (EQUALITY), then the enforced row
    /// sets every column to `target` (0.0, cost NOT modified) with `target_index -> 1-target`.
    /// Pins the `==`-not-`>=` asymmetry vs the scoring path (IMPROVEMENTS'd).
    #[test]
    fn get_targets_lid_enforcement() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/reference_data/phase4b/twin_mode0.config");
        let mut map =
            crate::legacy_config::parse_legacy_config(&std::fs::read_to_string(p).unwrap());
        map.insert(
            "BLSTM_LID_TargetEnforcementStep".to_string(),
            "2".to_string(),
        );
        let drv = TwinBlstmSpectralLid::from_legacy(&map, None, None).unwrap();

        // A single SPEECH span [0, 0.9) so all 6 rows (t = 0.0..0.5 at step 0.1) are SPEECH.
        let mut seg = Segmentation::new(1.0);
        seg.label_segment(0.0, 0.9, SegClass::Speech);
        let targets = drv.get_targets_lid(&seg, 0.1, 0.0, 1, 6);

        // step 2: rows 0,1 = -0.5; row 2 = enforced; rows 3,4 = -0.5; row 5 = enforced.
        for r in [0usize, 1, 3, 4] {
            for c in 0..3 {
                assert_eq!(targets[[r, c]], -0.5, "row {r} col {c} should be -0.5");
            }
        }
        for r in [2usize, 5] {
            assert_eq!(targets[[r, 0]], 0.0, "enforced row {r} col0");
            assert_eq!(
                targets[[r, 1]],
                1.0,
                "enforced row {r} target col (1-target)"
            );
            assert_eq!(targets[[r, 2]], 0.0, "enforced row {r} col2");
        }
    }
}
