//! `Segmenter` decision logic (hysteresis-with-area) + the fixed smoothing pipeline.
//!
//! Ported from legacy C++ `Segmenter.cpp`: `buildFromConf` (:85-138),
//! `updateSegmentation` (:725-845), `smoothSegmentation` (:709-723). Pure logic,
//! validated by hand-computed unit tests.

use anyhow::{Result, anyhow};
use indexmap::IndexMap;

use super::segmentation::{SegClass, Segmentation};

/// Produces a [`Segmentation`](super::segmentation_io) from features/audio.
pub trait Segmenter {}

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

/// Parse a comma-separated list of `f64` from a config value (legacy `n<double>`).
fn parse_list(m: &IndexMap<String, String>, key: &str) -> Result<Vec<f64>> {
    let raw = m
        .get(key)
        .ok_or_else(|| anyhow!("missing config key `{key}`"))?;
    raw.split(',')
        .map(|tok| {
            tok.trim()
                .parse::<f64>()
                .map_err(|e| anyhow!("`{key}`: cannot parse `{tok}`: {e}"))
        })
        .collect()
}

/// Parse a scalar `f64` from a config value.
fn parse_scalar(m: &IndexMap<String, String>, key: &str) -> Result<f64> {
    m.get(key)
        .ok_or_else(|| anyhow!("missing config key `{key}`"))?
        .trim()
        .parse::<f64>()
        .map_err(|e| anyhow!("`{key}`: cannot parse: {e}"))
}

impl SegmenterConfig {
    /// Port of `Segmenter::buildFromConf` (:85-138): the `falling > rising` clamp,
    /// the length checks (`speech_padding` >= 4, `min_speech` >= 3, `min_silence`
    /// >= 2, else error), and the per-element negative -> 0 clamp.
    pub fn from_config(m: &IndexMap<String, String>, prefix: &str) -> Result<SegmenterConfig> {
        let rising = parse_scalar(m, &format!("{prefix}_decision_thresh_rising"))?;
        let area_rising = parse_scalar(m, &format!("{prefix}_decision_area_rising"))?;
        let mut falling = parse_scalar(m, &format!("{prefix}_decision_thresh_falling"))?;
        if falling > rising {
            falling = rising;
        }
        let area_falling = parse_scalar(m, &format!("{prefix}_decision_area_falling"))?;

        let padding_raw = parse_list(m, &format!("{prefix}_speech_padding"))?;
        if padding_raw.len() < 4 {
            return Err(anyhow!("{prefix}_speech_padding must contain 4 values"));
        }
        let min_speech_raw = parse_list(m, &format!("{prefix}_min_speech"))?;
        if min_speech_raw.len() < 3 {
            return Err(anyhow!("{prefix}_min_speech must contain 3 values"));
        }
        let min_silence_raw = parse_list(m, &format!("{prefix}_min_silence"))?;
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

/// The RAW hysteresis-with-area decision over the ROW vector `results`, WITHOUT
/// smoothing. Returns the pre-smoothing, pre-sanitize labeled segments as
/// `(begin + off, end + off, class as i32)`, one entry per legacy
/// `label_segment` call, in emission order.
///
/// Direct port of the decision body of `Segmenter::updateSegmentation`
/// (:725-845): the INIT check, the rising/falling hysteresis with area gating,
/// the re-run-rising-after-label, and the `results.size()` tail. This isolates
/// the decision math so it can be cross-checked bit-for-bit against the numpy
/// oracle (`speech.scoring.update_segmentation_oracle`). `results[k]` is the
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
    let t_r = cfg.rising;
    let a_r = cfg.area_rising;
    let t_f = cfg.falling;
    let a_f = cfg.area_falling;

    let mut segments: Vec<(f64, f64, i32)> = Vec::new();

    let mut begin = -1.0f64;
    let mut end = -1.0f64;
    let mut begin_area = -1.0f64;
    let mut end_area = -1.0f64;
    let mut has_begun = false;
    let mut has_ended = false;
    let length = results.len();

    if length > 0 && results[0] >= t_r {
        begin = 0.0;
        begin_area = 0.0;
    }

    for ii in 1..length {
        let r = results[ii];
        let r_prev = results[ii - 1];

        if !has_begun {
            if begin < 0.0 {
                if r >= t_r && r_prev < t_r {
                    begin = dt * (ii as f64 - (r - t_r) / (r - r_prev));
                    begin_area = (ii as f64 - begin / dt) * (r - t_r) / 2.0;
                    if begin_area >= a_r {
                        has_begun = true;
                    }
                }
            } else if r >= t_r {
                begin_area += (r + r_prev - t_r * 2.0) / 2.0;
                if begin_area >= a_r {
                    has_begun = true;
                }
            } else if r < t_r && r_prev >= t_r {
                begin_area += (r_prev - t_r) * (t_r - r_prev) / (r - r_prev) / 2.0;
                if begin_area >= a_r {
                    has_begun = true;
                } else {
                    begin = -1.0;
                    begin_area = -1.0;
                    has_begun = false;
                }
            } else {
                begin = -1.0;
                begin_area = -1.0;
                has_begun = false;
            }
        }

        if has_begun {
            if end < 0.0 {
                if r <= t_f && r_prev > t_f {
                    end = dt * (ii as f64 - (r - t_f) / (r - r_prev));
                    end_area = (ii as f64 - end / dt) * (t_f - r) / 2.0;
                    if end_area >= a_f {
                        has_ended = true;
                    }
                }
            } else if r <= t_f {
                end_area += (2.0 * t_f - r - r_prev) / 2.0;
                if end_area >= a_f {
                    has_ended = true;
                }
            } else if r > t_f && r_prev <= t_f {
                end_area += (t_f - r_prev) * (t_f - r_prev) / (r - r_prev) / 2.0;
                if end_area >= a_f {
                    has_ended = true;
                } else {
                    end = -1.0;
                    end_area = -1.0;
                    has_ended = false;
                }
            } else {
                end = -1.0;
                end_area = -1.0;
                has_ended = false;
            }

            if has_ended {
                if begin < end {
                    segments.push((begin + off, end + off, class as i32));
                    begin = -1.0;
                    begin_area = -1.0;
                    has_begun = false;
                    end = -1.0;
                    end_area = -1.0;
                    has_ended = false;

                    if r >= t_r && r_prev < t_r {
                        begin = dt * (ii as f64 - (r - t_r) / (r - r_prev));
                        begin_area = (ii as f64 - begin / dt) * (r - t_r) / 2.0;
                        if begin_area >= a_r {
                            has_begun = true;
                        }
                    }
                } else {
                    begin = -1.0;
                    begin_area = -1.0;
                    has_begun = false;
                    end = -1.0;
                    end_area = -1.0;
                    has_ended = false;
                }
            }
        }
    }

    if has_begun {
        end = dt * results.len() as f64;
        segments.push((begin + off, end + off, class as i32));
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
