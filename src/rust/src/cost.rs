//! Piecewise VAD cost laws + multiclass softmax cross-entropy with ignore-mask.
//!
//! Ported bit-exactly from legacy C++: CostLaw.{h,cpp}. Preserves the double-read
//! quirk and the AboveThreshCubic name routing (see IMPROVEMENTS.md Legacy Quirks).
//! The legacy LogLaw cost/deriv Adim asymmetry (a wrong gradient in the saturated
//! region) was FIXED in phase5 (F7): `Law::deriv` for `Log` is now consistent with
//! the clamped forward. See design spec section 3.
//!
//! The scalar dispatch mirrors CostLaw.cpp:122-189 (cost) and 278-345 (delta):
//! the no-speech regime tests `output > switching_thresh_no_speech`, feeds the
//! flipped `1 - output` to its laws, negates the derivative, and the logistic
//! chain-rule factor multiplies the (mutated) output for both regimes.

use anyhow::{Result, bail};
use indexmap::IndexMap;

use crate::legacy_config::{get_f64_default, parse_finite_f64};

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// One piecewise law primitive; coefficients precomputed from (P, q, t) and name.
#[derive(Debug, Clone, Copy)]
enum Law {
    Linear { a: f64, b: f64 },
    Square { a: f64, b: f64, c: f64 },
    BelowCubic { a: f64, b: f64, d: f64 },
    AboveCubic { a: f64, b: f64 },
    Log { a: f64, b: f64, adim: f64 },
    BelowSqrt { a: f64, b: f64, adim: f64 },
    AboveSqrt { a: f64, b: f64, adim: f64 },
}

impl Law {
    // The Log/Sqrt COST arms use legacy sequential-if clamps (order + NaN semantics
    // differ from f64::clamp); do not "simplify". The Log DERIV arm was made consistent
    // with the clamped forward in phase5 (F7) -- see its arm below + IMPROVEMENTS.md.
    #[allow(clippy::manual_clamp)]
    fn cost(&self, y: f64) -> f64 {
        match *self {
            Law::Linear { a, b } => b + a * y,
            Law::Square { a, b, c } => c + b * y + a * y * y,
            Law::BelowCubic { a, b, d } => d + b * y * y + a * y * y * y,
            Law::AboveCubic { a, b } => {
                let y = 1.0 - y;
                b * y * y + a * y * y * y
            }
            Law::Log { a, b, adim } => {
                let mut y = y / adim;
                if y < 1e-24 {
                    y = 1e-24;
                }
                if y > 1.0 {
                    y = 1.0;
                }
                b + a * y.ln()
            }
            Law::BelowSqrt { a, b, adim } => {
                let mut y = y / adim;
                if y > 1.0 {
                    y = 1.0;
                }
                b + a * (1.0 - y).sqrt()
            }
            Law::AboveSqrt { a, b, adim } => {
                let mut y = 1.0 - y;
                y /= adim;
                if y > 1.0 {
                    y = 1.0;
                }
                b + a * (1.0 - y).sqrt()
            }
        }
    }

    // manual_range_contains: the explicit `z < 1e-24 || z > 1.0` mirrors the forward's
    // two separate clamp conditions (and keeps NaN -> NaN, not -> 0, exact); do not fold.
    #[allow(clippy::manual_range_contains)]
    fn deriv(&self, y: f64) -> f64 {
        match *self {
            Law::Linear { a, .. } => a,
            Law::Square { a, b, .. } => b + 2.0 * a * y,
            Law::BelowCubic { a, b, .. } => 2.0 * b * y + 3.0 * a * y * y,
            Law::AboveCubic { a, b } => {
                let y = 1.0 - y;
                -2.0 * b * y - 3.0 * a * y * y
            }
            Law::Log { a, adim, .. } => {
                // F7 (phase5; was IMPROVEMENTS [0b-i] LogLaw asymmetry): the derivative
                // CONSISTENT with the clamped forward. `cost()` computes
                // `b + a*ln(clamp(y/adim, 1e-24, 1))`; where that argument is clamped the
                // cost is CONSTANT, so the true gradient is 0. In the unclamped interior
                // `d/dy [b + a*ln(y/adim)] = a/y` (the adim cancels via the chain rule, so
                // the interior value is unchanged from the legacy). The legacy returned
                // `a/clamp(y,1e-24,1)` UNCONDITIONALLY -- a nonzero WRONG gradient in the
                // saturated region. On the committed VAD grids the divergence only lands at
                // output -> {0,1}, where the logistic fold `output*(1-output)` in
                // `compute_unitary_delta` zeros the contribution, so no existing golden
                // moves; the divergence is pinned off-grid by
                // `log_deriv_consistent_with_clamped_forward`. See IMPROVEMENTS.md.
                let z = y / adim;
                if z < 1e-24 || z > 1.0 { 0.0 } else { a / y }
            }
            Law::BelowSqrt { a, adim, .. } => {
                let mut y = 1.0 - y / adim;
                if y < 1e-24 {
                    y = 1e-24;
                }
                -a / 2.0 / y.sqrt()
            }
            Law::AboveSqrt { a, adim, .. } => {
                let mut y = 1.0 - y;
                y = 1.0 - y / adim;
                if y < 1e-24 {
                    y = 1e-24;
                }
                a / 2.0 / y.sqrt()
            }
        }
    }
}

fn below_law(name: &str, p: f64, q: f64, t: f64) -> Law {
    match name {
        "linear" => Law::Linear {
            a: -p * (1.0 - q) / t,
            b: p,
        },
        "square" => Law::Square {
            a: p * (1.0 - q) / t / t,
            b: -p * 2.0 * (1.0 - q) / t,
            c: p,
        },
        "cubic" => Law::BelowCubic {
            a: p * 2.0 * (1.0 - q) / t / t / t,
            b: -p * 3.0 * (1.0 - q) / t / t,
            d: p,
        },
        "log" => Law::Log {
            a: -p * (1.0 - q),
            b: p * q,
            adim: t,
        },
        other => unreachable!("cost law {other:?} passed get_law"),
    }
}

fn above_cubic(name: &str, p: f64, q: f64, t: f64) -> Law {
    let (a, b) = match name {
        "square" | "cubic" => (
            -p * 2.0 * q / (1.0 - t).powi(3),
            p * 3.0 * q / (1.0 - t).powi(2),
        ),
        "linear" | "log" => (
            p * ((1.0 - q) / t / (1.0 - t).powi(2) - 2.0 * q / (1.0 - t).powi(3)),
            p * (3.0 * q / (1.0 - t).powi(2) - (1.0 - q) / t / (1.0 - t)),
        ),
        other => unreachable!("cost law {other:?} passed get_law"),
    };
    Law::AboveCubic { a, b }
}

/// Built law pair for one regime (below-thresh + above-thresh), incl. the sqrt special case.
#[derive(Debug, Clone, Copy)]
struct RegimeLaws {
    below: Law,
    above: Law,
}

impl RegimeLaws {
    fn build(name: &str, p: f64, q: f64, t: f64) -> RegimeLaws {
        if name == "sqrt" {
            RegimeLaws {
                below: Law::BelowSqrt {
                    a: p * (1.0 - q),
                    b: p * q,
                    adim: t,
                },
                above: Law::AboveSqrt {
                    a: p * (-q),
                    b: p * q,
                    adim: 1.0 - t,
                },
            }
        } else {
            RegimeLaws {
                below: below_law(name, p, q, t),
                above: above_cubic(name, p, q, t),
            }
        }
    }
}

/// The VAD cost law (scalar path) + softmax config, ported from CostLaw.
#[derive(Debug, Clone)]
pub struct CostLaw {
    speech: RegimeLaws,
    no_speech: RegimeLaws,
    switching_thresh_speech: f64,
    switching_thresh_no_speech: f64,
    back_prop_wer: bool,
    classes_ponderations: Vec<f64>,
    /// `_CostPonderation` (`CostLaw.cpp:114-116`): the clamped [0.01, 0.99] scalar
    /// (config `<prefix>_CostPonderation`, default 0.5). Kept as a member solely to
    /// back `getCostPonderation()`, consumed by the BINARY branch of
    /// `BlstmNetwork::get_cost_ponderation`.
    cost_ponderation: f64,
    speech_name: String,
    no_speech_name: String,
}

/// `conf.get<string>(name, "log")` for a regime's law: an unknown name errors (the legacy
/// ctor `exit(1)`s) instead of reaching the panic in `below_law`.
fn get_law(m: &IndexMap<String, String>, key: &str) -> Result<String> {
    let name = m.get(key).map_or("log", |s| s.trim());
    match name {
        "log" | "linear" | "square" | "cubic" | "sqrt" => Ok(name.to_string()),
        _ => bail!("cost law '{name}' is not valid for '{key}'"),
    }
}

impl CostLaw {
    /// Build from a parsed legacy config, using the algName prefix (e.g. "BLSTM").
    ///
    /// Every numeric key defaults when MISSING and errors when present but malformed or
    /// non-finite; each `classes_ponderations` entry is parsed strictly, so a bad entry
    /// errors instead of dropping out and shifting later weights onto the wrong class.
    pub fn from_config(m: &IndexMap<String, String>, prefix: &str) -> Result<CostLaw> {
        let p = format!("{prefix}_");
        let cp = clamp(
            get_f64_default(m, &format!("{p}CostPonderation"), 0.5)?,
            0.01,
            0.99,
        );
        let ps = clamp(
            get_f64_default(m, &format!("{p}CostLawParamSpeech"), 0.0)?,
            0.0,
            1.0,
        );
        let pn = clamp(
            get_f64_default(m, &format!("{p}CostLawParamNoSpeech"), 0.0)?,
            0.0,
            1.0,
        );
        let th_s = clamp(
            get_f64_default(m, &format!("{p}CostLawThreshSpeech"), 1.0)?,
            1e-6,
            1.0 - 1e-6,
        );
        let th_n_law = 1.0
            - clamp(
                get_f64_default(m, &format!("{p}CostLawThreshNoSpeech"), 0.0)?,
                1e-6,
                1.0 - 1e-6,
            );
        let speech_name = get_law(m, &format!("{p}CostLawSpeech"))?;
        let no_speech_name = get_law(m, &format!("{p}CostLawNoSpeech"))?;
        // DECISIVE DOUBLE-READ: branch selectors re-read raw/unclamped (defaults 10.0 / -1.0).
        let switching_thresh_speech = get_f64_default(m, &format!("{p}CostLawThreshSpeech"), 10.0)?;
        let switching_thresh_no_speech =
            get_f64_default(m, &format!("{p}CostLawThreshNoSpeech"), -1.0)?;
        let back_prop_wer = get_f64_default(m, &format!("{p}BackPropWER"), -1.0)? >= 0.0;
        let pond_key = format!("{p}classes_ponderations");
        let cponds = match m.get(&pond_key) {
            None => Vec::new(),
            Some(v) => v
                .split(',')
                .map(|s| parse_finite_f64(s, &pond_key))
                .collect::<Result<Vec<_>>>()?,
        };
        let classes_ponderations = if !cponds.is_empty() && cponds[0] > 0.0 {
            cponds
        } else {
            Vec::new()
        };
        Ok(CostLaw {
            speech: RegimeLaws::build(&speech_name, cp, ps, th_s),
            no_speech: RegimeLaws::build(&no_speech_name, 1.0 - cp, pn, th_n_law),
            switching_thresh_speech,
            switching_thresh_no_speech,
            back_prop_wer,
            classes_ponderations,
            cost_ponderation: cp,
            speech_name,
            no_speech_name,
        })
    }

    /// Scalar VAD cost for one (output, target). target<0 => ignored (0).
    ///
    /// Mirrors CostLaw.cpp:122-189. No-speech tests `output > switching_thresh_no_speech`
    /// and feeds the flipped `1 - output` to its below/above laws.
    pub fn compute_unitary_cost(&self, output: f64, target: f64) -> f64 {
        if target < 0.0 {
            return 0.0;
        }
        let mut cost = if target > 0.5 {
            if output < self.switching_thresh_speech {
                self.speech.below.cost(output)
            } else {
                self.speech.above.cost(output)
            }
        } else {
            let flipped = 1.0 - output;
            if output > self.switching_thresh_no_speech {
                self.no_speech.below.cost(flipped)
            } else {
                self.no_speech.above.cost(flipped)
            }
        };
        if self.back_prop_wer {
            cost *= if target > 0.5 {
                10.0 * (1.0 - target)
            } else {
                10.0 * target
            };
        }
        cost
    }

    /// Scalar VAD delta = law derivative * logistic chain rule output*(1-output). target<0 => 0.
    ///
    /// Mirrors CostLaw.cpp:278-345. The no-speech regime feeds `1 - output` to its laws,
    /// negates the derivative, and the final chain-rule factor multiplies the flipped output.
    pub fn compute_unitary_delta(&self, output: f64, target: f64) -> f64 {
        if target < 0.0 {
            return 0.0;
        }
        let mut output = output;
        let mut delta = if target > 0.5 {
            if output < self.switching_thresh_speech {
                self.speech.below.deriv(output)
            } else {
                self.speech.above.deriv(output)
            }
        } else if output > self.switching_thresh_no_speech {
            output = 1.0 - output;
            -self.no_speech.below.deriv(output)
        } else {
            output = 1.0 - output;
            -self.no_speech.above.deriv(output)
        };
        if self.back_prop_wer {
            delta *= if target > 0.5 {
                10.0 * (1.0 - target)
            } else {
                10.0 * target
            };
        }
        delta * output * (1.0 - output)
    }

    /// Speech/no-speech law names (kept for legacy parity; unused by the softmax path).
    pub fn law_names(&self) -> (&str, &str) {
        (&self.speech_name, &self.no_speech_name)
    }

    /// `isCostModified` (`CostLaw.cpp:110-112`): returns `_BackPropWER`. The scoring
    /// `feedForward` reads this to decide the soft target (`0.1*modifier` when set,
    /// else `0.0`) at `BLSTMNeuralNetwork.cpp:871`.
    pub fn is_cost_modified(&self) -> bool {
        self.back_prop_wer
    }

    /// Class ponderations for the softmax path (empty when unset).
    pub fn classes_ponderations(&self) -> &[f64] {
        &self.classes_ponderations
    }

    /// `getCostPonderation` (`CostLaw.cpp:114-116`): the clamped scalar cost ponderation.
    pub fn cost_ponderation(&self) -> f64 {
        self.cost_ponderation
    }

    /// Multiclass softmax cross-entropy cost over a row-major `n_frames x n_classes` batch.
    ///
    /// Mirrors CostLaw.cpp:212-246 (the `targetSeq.cols() > 1` branch): sums
    /// `-ln(clamp(output, 1e-24, inf))` over the on-class (`target > 0.5`) entries.
    /// A frame whose target row is all `< 0` contributes nothing (no on-class).
    /// Each term is scaled by `classes_ponderations[kk]` (when present) and by
    /// `10 * (1 - target)` when `back_prop_wer` is set.
    ///
    /// Accumulation order is COLUMN-MAJOR (outer kk/classes, inner jj/frames) to
    /// match CostLaw.cpp:219-231 bit-exactly: f64 addition is not associative, so
    /// summing in a different order can diverge in the last bit(s) for batches
    /// with 2+ on-class terms across different columns.
    pub fn compute_cost(&self, outputs: &[f64], onehot_target: &[f64], n_classes: usize) -> f64 {
        let use_pond = !self.classes_ponderations.is_empty();
        if use_pond && self.classes_ponderations.len() < n_classes {
            panic!("Not enough ponderations given for the number of classes!");
        }
        let mut cost = 0.0;
        let n_frames = outputs.len() / n_classes;
        for kk in 0..n_classes {
            for jj in 0..n_frames {
                let idx = jj * n_classes + kk;
                let target = onehot_target[idx];
                if target > 0.5 {
                    let value = outputs[idx].max(1e-24);
                    let pond = if use_pond {
                        self.classes_ponderations[kk]
                    } else {
                        1.0
                    };
                    if self.back_prop_wer {
                        cost += -value.ln() * pond * 10.0 * (1.0 - target);
                    } else {
                        cost += -value.ln() * pond;
                    }
                }
            }
        }
        cost
    }

    /// Multiclass softmax cross-entropy deltas over a row-major `n_frames x n_classes` batch.
    ///
    /// Mirrors CostLaw.cpp:358-419 (the `targetSeq.cols() > 1` branch). Per element:
    /// `target < 0` => 0 (ignore); the core contract is `delta = output - onehot`,
    /// i.e. `output - 1` on the on-class (`target > 0.5`) and `output` elsewhere.
    /// When `back_prop_wer` is set the legacy bakes in the WER factors
    /// (`10*(1-target)` on-class, `10*target` off-class); class ponderations scale
    /// each element (WER path) or the whole frame row by the on-class ponderation
    /// (non-WER path).
    pub fn compute_deltas(
        &self,
        outputs: &[f64],
        onehot_target: &[f64],
        n_classes: usize,
        out: &mut [f64],
    ) {
        let use_pond = !self.classes_ponderations.is_empty();
        if use_pond && self.classes_ponderations.len() < n_classes {
            panic!("Not enough ponderations given for the number of classes!");
        }
        let n_frames = outputs.len() / n_classes;
        if self.back_prop_wer {
            for jj in 0..n_frames {
                for kk in 0..n_classes {
                    let idx = jj * n_classes + kk;
                    let target = onehot_target[idx];
                    let pond = if use_pond {
                        self.classes_ponderations[kk]
                    } else {
                        1.0
                    };
                    out[idx] = if target < 0.0 {
                        0.0
                    } else if target > 0.5 {
                        pond * 10.0 * (1.0 - target) * (outputs[idx] - 1.0)
                    } else {
                        pond * 10.0 * target * outputs[idx]
                    };
                }
            }
        } else {
            for jj in 0..n_frames {
                // Legacy captures the on-class ponderation for the frame, then scales
                // the whole row by it (CostLaw.cpp:404-417).
                let mut ponderation = 1.0;
                for kk in 0..n_classes {
                    let idx = jj * n_classes + kk;
                    let target = onehot_target[idx];
                    out[idx] = if target < 0.0 {
                        0.0
                    } else if target > 0.5 {
                        if use_pond {
                            ponderation = self.classes_ponderations[kk];
                        }
                        outputs[idx] - 1.0
                    } else {
                        outputs[idx]
                    };
                }
                if use_pond {
                    for kk in 0..n_classes {
                        out[jj * n_classes + kk] *= ponderation;
                    }
                }
            }
        }
    }
}

/// Issue #47: every numeric key distinguishes a MISSING key (default) from a PRESENT
/// but malformed or non-finite one (error naming the key and the text), as #32 does for
/// the `nn/blstm.rs` getters; `classes_ponderations` entries are parsed strictly.
#[cfg(test)]
mod config_read_tests {
    use super::*;

    const MALFORMED: [&str; 5] = ["1O", "O.5", "1e-x", "", "0.5,0.6"];
    const NON_FINITE: [&str; 6] = ["nan", "NaN", "inf", "-inf", "-infinity", "1e400"];

    fn map(key: &str, value: &str) -> IndexMap<String, String> {
        let mut m = IndexMap::new();
        m.insert(key.to_string(), value.to_string());
        m
    }

    fn assert_strict(suffix: &str) {
        let key = format!("BLSTM_{suffix}");
        assert!(CostLaw::from_config(&map(&key, " 0.3 "), "BLSTM").is_ok());
        for bad in MALFORMED.iter().chain(NON_FINITE.iter()) {
            let err = CostLaw::from_config(&map(&key, bad), "BLSTM")
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&key) && err.contains(&format!("'{bad}'")),
                "{key}={bad:?}: {err}"
            );
        }
    }

    #[test]
    fn cost_law_names_strict() {
        for suffix in ["CostLawSpeech", "CostLawNoSpeech"] {
            let key = format!("BLSTM_{suffix}");
            for good in ["log", "linear", "square", "cubic", "sqrt"] {
                assert!(CostLaw::from_config(&map(&key, good), "BLSTM").is_ok());
            }
            assert!(CostLaw::from_config(&map(&key, " log "), "BLSTM").is_ok());
            let err = CostLaw::from_config(&map(&key, "Log"), "BLSTM")
                .unwrap_err()
                .to_string();
            assert!(err.contains(&key) && err.contains("'Log'"), "{err}");
        }
    }

    #[test]
    fn missing_keys_take_the_defaults() {
        let law = CostLaw::from_config(&IndexMap::new(), "BLSTM").unwrap();
        assert_eq!(law.speech_name, "log");
        assert_eq!(law.no_speech_name, "log");
        assert_eq!(law.cost_ponderation(), 0.5);
        assert_eq!(law.switching_thresh_speech, 10.0);
        assert_eq!(law.switching_thresh_no_speech, -1.0);
        assert!(!law.is_cost_modified());
        assert!(law.classes_ponderations().is_empty());
    }

    #[test]
    fn cost_ponderation_strict() {
        assert_strict("CostPonderation");
    }

    #[test]
    fn cost_law_param_speech_strict() {
        assert_strict("CostLawParamSpeech");
    }

    #[test]
    fn cost_law_param_no_speech_strict() {
        assert_strict("CostLawParamNoSpeech");
    }

    #[test]
    fn cost_law_thresh_speech_strict() {
        assert_strict("CostLawThreshSpeech");
    }

    #[test]
    fn cost_law_thresh_no_speech_strict() {
        assert_strict("CostLawThreshNoSpeech");
    }

    #[test]
    fn back_prop_wer_strict() {
        assert_strict("BackPropWER");
    }

    /// A malformed middle entry used to drop out (`1,O.5,2` -> `[1, 2]`), shifting
    /// class 2's weight onto class 1.
    #[test]
    fn classes_ponderations_strict() {
        let key = "BLSTM_classes_ponderations";
        let law = CostLaw::from_config(&map(key, "2, 3 ,4"), "BLSTM").unwrap();
        assert_eq!(law.classes_ponderations(), &[2.0, 3.0, 4.0]);
        let law = CostLaw::from_config(&map(key, "0,3,4"), "BLSTM").unwrap();
        assert!(law.classes_ponderations().is_empty(), "first <= 0 disables");
        for bad in ["1,O.5,2", "1,,2", "1,2,", "1,nan,2", "1,inf"] {
            let err = CostLaw::from_config(&map(key, bad), "BLSTM")
                .unwrap_err()
                .to_string();
            assert!(err.contains(key), "{bad:?}: {err}");
        }
        let err = CostLaw::from_config(&map(key, "1,O.5,2"), "BLSTM")
            .unwrap_err()
            .to_string();
        assert!(err.contains("O.5"), "{err}");
    }
}
