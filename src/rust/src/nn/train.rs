//! iRPROP- on a flat weight vector (inverted sign, weight-backtracking).
//!
//! Ported from legacy C++: Rprop.cpp, Trainer.h. Phase 3.
//!
//! No `Trainer` trait: Trainer.h declares a single-method virtual interface, but
//! Rprop is the only implementation ever instantiated (Rprop.cpp is the sole
//! non-header .cpp under the hierarchy) -- a trait here would be an abstraction
//! for one variant (YAGNI). The Python-side SMORMS3 (`optimizers.py`) is a
//! different, Python-side optimizer over the OUTER DSP-hyperparameter loop, not
//! a second Rust `Trainer`.
//!
//! DEAD - not ported: Rprop.cpp:60-76 (a commented-out per-call .mat dump of
//! weights/deltaweights/deltas/prevderivs) and the `_Count` member that fed its
//! filename counter (also commented out in Rprop.h) -- both inert in the
//! legacy source itself.

/// iRPROP- trainer state (`Rprop.cpp:6-77`). `deltas`/`prev_derivs`/
/// `delta_weights` are empty until the first [`Rprop::update_weights`] call,
/// which sizes them to the weight vector's length; `deltas.is_empty()` is the
/// legacy `_Deltas.size() == 0` first-call gate.
#[derive(Debug, Clone)]
pub struct Rprop {
    deltas: Vec<f64>,
    prev_derivs: Vec<f64>,
    delta_weights: Vec<f64>,
    /// `_PrevCost` (Rprop.h): UNINITIALIZED in the legacy ctor, but never read
    /// before the first call's tail (`_PrevCost = cost;`) writes it -- the
    /// per-element cost-gated backtrack branch (`_PrevCost < cost`) is
    /// unreachable on the first call (that call takes the empty-`_Deltas` init
    /// branch entirely, which never touches `_PrevCost`). Initialized to `0.0`
    /// here; the value is dead until the second call reads it.
    prev_cost: f64,
    eta_min: f64,
    eta_plus: f64,
    min_delta: f64,
    max_delta: f64,
    init_delta: f64,
}

impl Rprop {
    /// `Rprop(double initDelta)` (`Rprop.cpp:6`): eta 0.5/1.2, delta clamps
    /// 1e-9/0.2, `initDelta` from the caller (legacy default 1e-2,
    /// `_BackPropagationRpropInit`, `BLSTMNeuralNetwork.cpp:151`).
    pub fn new(init_delta: f64) -> Rprop {
        Rprop {
            deltas: Vec::new(),
            prev_derivs: Vec::new(),
            delta_weights: Vec::new(),
            prev_cost: 0.0,
            eta_min: 0.5,
            eta_plus: 1.2,
            min_delta: 1e-9,
            max_delta: 0.2,
            init_delta,
        }
    }

    /// `Rprop::updateWeights` (`Rprop.cpp:9-59`; `:60-76` is the dead .mat dump,
    /// not ported). INVERTED SIGN CONVENTION throughout: `deriv > 0 -> dw =
    /// -delta` (descend), `deriv < 0 -> dw = +delta` (ascend), `deriv == 0 -> dw
    /// = 0`; applied as `weights += dw`. Do not "fix" this -- it is the legacy's
    /// convention and every downstream golden is pinned against it.
    pub fn update_weights(&mut self, weights_derivatives: &[f64], weights: &mut [f64], cost: f64) {
        let n = weights_derivatives.len();
        if self.deltas.is_empty() {
            // First call (`_Deltas.size() == 0`, :10-22): const-init deltas,
            // snapshot derivs, zero delta_weights, then apply per element.
            self.deltas = vec![self.init_delta; weights.len()];
            self.prev_derivs = weights_derivatives.to_vec();
            self.delta_weights = vec![0.0; weights.len()];
            for j in 0..n {
                let d = weights_derivatives[j];
                self.delta_weights[j] = if d == 0.0 {
                    0.0
                } else if d > 0.0 {
                    -self.deltas[j]
                } else {
                    self.deltas[j]
                };
                weights[j] += self.delta_weights[j];
            }
        } else {
            // Subsequent call (:24-56). `derivTimesPrev` computed BEFORE
            // `_PrevDerivs` is overwritten (:25-26) -- order matters, it reads
            // the OLD prev_derivs.
            let deriv_times_prev: Vec<f64> = (0..n)
                .map(|j| weights_derivatives[j] * self.prev_derivs[j])
                .collect();
            self.prev_derivs = weights_derivatives.to_vec();
            for j in 0..n {
                let dtp = deriv_times_prev[j];
                if dtp > 0.0 {
                    // Same-sign two steps running: grow, clamp to max (:28-31).
                    self.deltas[j] *= self.eta_plus;
                    if self.deltas[j] > self.max_delta {
                        self.deltas[j] = self.max_delta;
                    }
                    let d = weights_derivatives[j];
                    self.delta_weights[j] = if d == 0.0 {
                        0.0
                    } else if d > 0.0 {
                        -self.deltas[j]
                    } else {
                        self.deltas[j]
                    };
                    weights[j] += self.delta_weights[j];
                } else if dtp < 0.0 {
                    // Sign flip: shrink, clamp to min (:38-41). COST-GATED
                    // BACKTRACK: only when cost ROSE since the last call
                    // (`_PrevCost < cost`) undo the LAST applied dw (:42-44) --
                    // do not "fix" this into an unconditional backtrack. NO dw
                    // is applied for this element this call either way. Force
                    // `prev_derivs[j] = 0` (:45) so the NEXT call's
                    // derivTimesPrev for this element is exactly 0 regardless
                    // of sign, routing it into the `== 0` branch below.
                    self.deltas[j] *= self.eta_min;
                    if self.deltas[j] < self.min_delta {
                        self.deltas[j] = self.min_delta;
                    }
                    if self.prev_cost < cost {
                        weights[j] -= self.delta_weights[j];
                    }
                    self.prev_derivs[j] = 0.0;
                } else {
                    // dtp == 0.0 (:47-55): either a genuinely zero current
                    // deriv, or the post-backtrack forced-zero prev_derivs from
                    // a prior call. Recompute + apply dw at the CURRENT delta
                    // (unchanged this call).
                    let d = weights_derivatives[j];
                    self.delta_weights[j] = if d == 0.0 {
                        0.0
                    } else if d > 0.0 {
                        -self.deltas[j]
                    } else {
                        self.deltas[j]
                    };
                    weights[j] += self.delta_weights[j];
                }
            }
        }
        self.prev_cost = cost; // :58
    }

    pub fn deltas(&self) -> &[f64] {
        &self.deltas
    }

    pub fn prev_derivs(&self) -> &[f64] {
        &self.prev_derivs
    }

    pub fn delta_weights(&self) -> &[f64] {
        &self.delta_weights
    }

    pub fn prev_cost(&self) -> f64 {
        self.prev_cost
    }
}
