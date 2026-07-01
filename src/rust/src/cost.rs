//! Piecewise VAD cost laws + multiclass softmax cross-entropy with ignore-mask.
//!
//! Ported bit-exactly from legacy C++: CostLaw.{h,cpp}. Preserves the double-read
//! quirk, the LogLaw cost/deriv Adim asymmetry, and the AboveThreshCubic name
//! routing (see IMPROVEMENTS.md Legacy Quirks). See design spec section 3.
//!
//! The scalar dispatch mirrors CostLaw.cpp:122-189 (cost) and 278-345 (delta):
//! the no-speech regime tests `output > switching_thresh_no_speech`, feeds the
//! flipped `1 - output` to its laws, negates the derivative, and the logistic
//! chain-rule factor multiplies the (mutated) output for both regimes.

use indexmap::IndexMap;

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
    // The LogLaw arms use legacy sequential-if clamps (order + NaN semantics differ
    // from f64::clamp, and the deriv clamps RAW y, not y/adim); do not "simplify".
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

    #[allow(clippy::manual_clamp)]
    fn deriv(&self, y: f64) -> f64 {
        match *self {
            Law::Linear { a, .. } => a,
            Law::Square { a, b, .. } => b + 2.0 * a * y,
            Law::BelowCubic { a, b, .. } => 2.0 * b * y + 3.0 * a * y * y,
            Law::AboveCubic { a, b } => {
                let y = 1.0 - y;
                -2.0 * b * y - 3.0 * a * y * y
            }
            Law::Log { a, .. } => {
                // ASYMMETRY: deriv clamps RAW y (not divided by adim). Legacy quirk, reproduced.
                let mut y = y;
                if y < 1e-24 {
                    y = 1e-24;
                }
                if y > 1.0 {
                    y = 1.0;
                }
                a / y
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
        other => panic!("cost law not valid: {other}"),
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
        other => panic!("above-thresh cubic name not valid: {other}"),
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
    speech_name: String,
    no_speech_name: String,
}

fn getf(m: &IndexMap<String, String>, key: &str, default: f64) -> f64 {
    m.get(key)
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

fn gets(m: &IndexMap<String, String>, key: &str, default: &str) -> String {
    m.get(key).cloned().unwrap_or_else(|| default.to_string())
}

impl CostLaw {
    /// Build from a parsed legacy config, using the algName prefix (e.g. "BLSTM").
    pub fn from_config(m: &IndexMap<String, String>, prefix: &str) -> CostLaw {
        let p = format!("{prefix}_");
        let cp = clamp(getf(m, &format!("{p}CostPonderation"), 0.5), 0.01, 0.99);
        let ps = clamp(getf(m, &format!("{p}CostLawParamSpeech"), 0.0), 0.0, 1.0);
        let pn = clamp(getf(m, &format!("{p}CostLawParamNoSpeech"), 0.0), 0.0, 1.0);
        let th_s = clamp(
            getf(m, &format!("{p}CostLawThreshSpeech"), 1.0),
            1e-6,
            1.0 - 1e-6,
        );
        let th_n_law = 1.0
            - clamp(
                getf(m, &format!("{p}CostLawThreshNoSpeech"), 0.0),
                1e-6,
                1.0 - 1e-6,
            );
        let speech_name = gets(m, &format!("{p}CostLawSpeech"), "log");
        let no_speech_name = gets(m, &format!("{p}CostLawNoSpeech"), "log");
        // DECISIVE DOUBLE-READ: branch selectors re-read raw/unclamped (defaults 10.0 / -1.0).
        let switching_thresh_speech = getf(m, &format!("{p}CostLawThreshSpeech"), 10.0);
        let switching_thresh_no_speech = getf(m, &format!("{p}CostLawThreshNoSpeech"), -1.0);
        let back_prop_wer = getf(m, &format!("{p}BackPropWER"), -1.0) >= 0.0;
        let cponds = m
            .get(&format!("{p}classes_ponderations"))
            .map(|v| {
                v.split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let classes_ponderations = if !cponds.is_empty() && cponds[0] > 0.0 {
            cponds
        } else {
            Vec::new()
        };
        CostLaw {
            speech: RegimeLaws::build(&speech_name, cp, ps, th_s),
            no_speech: RegimeLaws::build(&no_speech_name, 1.0 - cp, pn, th_n_law),
            switching_thresh_speech,
            switching_thresh_no_speech,
            back_prop_wer,
            classes_ponderations,
            speech_name,
            no_speech_name,
        }
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

    /// Speech/no-speech law names (for later multiclass wiring; keeps parity with legacy).
    pub fn law_names(&self) -> (&str, &str) {
        (&self.speech_name, &self.no_speech_name)
    }

    /// Class ponderations for the softmax path (empty when unset); wired in a later task.
    pub fn classes_ponderations(&self) -> &[f64] {
        &self.classes_ponderations
    }
}
