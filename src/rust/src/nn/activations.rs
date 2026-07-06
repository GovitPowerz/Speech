//! Activations honoring the legacy overrides: asinh cell/output, sigmoid(0.1z)
//! gates, softmax. Ported from legacy C++: ActivationFunctions.h. Phase 2.
//!
//! `exp_limit` mirrors `Log<double>::expLimit` (Log.hpp:193-195):
//! `std::log(std::numeric_limits<double>::max())`. `gates_fn`/`logistic_fn`
//! port `GatesFunction::fn`/`Logistic::fn` (ActivationFunctions.h:229-238,
//! 40-49) with their guard order preserved exactly:
//!
//! - `Logistic::fn`: INCLUSIVE saturation -- `x >= expLimit -> 1`,
//!   `x <= -expLimit -> 0` (the legacy tests `x < expLimit` / `x > -expLimit`
//!   as the pass-through branch, so the boundary itself saturates).
//! - `GatesFunction::fn`: a 0.1 pre-scale, EXCLUSIVE saturation on the SCALED
//!   value -- outer guard `0.1*x < expLimit`, inner guard `0.1*x > -expLimit`;
//!   at exactly `0.1*x == +-expLimit` it saturates (the guards are strict, so
//!   equality fails the pass-through branch on both sides).
//!
//! `Maxmin2`/`Identity` are both `std::asinh` (ActivationFunctions.h:158-160,
//! 206-208).

use std::sync::OnceLock;

/// `Log<double>::expLimit`: `std::log(f64::MAX)`, computed once.
pub fn exp_limit() -> f64 {
    static EXP_LIMIT: OnceLock<f64> = OnceLock::new();
    *EXP_LIMIT.get_or_init(|| f64::MAX.ln())
}

/// `GatesFunction::fn` (ActivationFunctions.h:230-237): sigmoid on `0.1*x`,
/// EXCLUSIVE saturation at `0.1*x == +-expLimit`.
pub fn gates_fn(x: f64) -> f64 {
    let scaled = 0.1 * x;
    if scaled < exp_limit() {
        if scaled > -exp_limit() {
            1.0 / (1.0 + (-scaled).exp())
        } else {
            0.0
        }
    } else {
        1.0
    }
}

/// `GatesFunction::deriv` (ActivationFunctions.h:240-242): `0.1*y*(1-y)`, taken on
/// the POST-activation value `y` (the activated sigmoid output). This is a plain
/// sigmoid-derivative-of-output times the 0.1 pre-scale (`d/dz sigmoid(0.1 z) =
/// 0.1 sigmoid(0.1 z)(1-sigmoid(0.1 z))`), NOT `f'(f^{-1}(y))`; the LSTM backward
/// (`LSTMLayer.cpp:585,629,653,662`) evaluates it on the cached gate values.
pub fn gates_deriv(y: f64) -> f64 {
    0.1 * y * (1.0 - y)
}

/// `Logistic::fn` (ActivationFunctions.h:41-48): plain sigmoid, INCLUSIVE
/// saturation at `x == +-expLimit`.
pub fn logistic_fn(x: f64) -> f64 {
    if x < exp_limit() {
        if x > -exp_limit() {
            1.0 / (1.0 + (-x).exp())
        } else {
            0.0
        }
    } else {
        1.0
    }
}

/// `Maxmin2::fn` (ActivationFunctions.h:158-160): `std::asinh`.
pub fn maxmin2_fn(x: f64) -> f64 {
    x.asinh()
}

/// `Identity::fn` (ActivationFunctions.h:207-209): `std::asinh` (same override).
pub fn identity_fn(x: f64) -> f64 {
    x.asinh()
}

/// `Asinh::fn` (ActivationFunctions.h:169-171): `std::asinh`.
pub fn asinh_fn(x: f64) -> f64 {
    x.asinh()
}

/// `Maxmin2::deriv` / `Asinh::deriv` (ActivationFunctions.h:162-165, 173-176):
/// `1/sqrt(1+sinh(y)^2)`, taken on `y` -- the pre-image at which `fn` was
/// evaluated (i.e. the raw layer input in `NeuronLayer::feedBackward:204`, NOT
/// the post-`asinh` output). Both structs share the identical body.
pub fn maxmin2_deriv(y: f64) -> f64 {
    let input = y.sinh();
    1.0 / (1.0 + input * input).sqrt()
}
