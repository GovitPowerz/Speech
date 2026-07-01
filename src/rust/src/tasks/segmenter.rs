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
