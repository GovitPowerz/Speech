//! sLSTM: the xLSTM paper's scalar-memory cell -- exponential input/forget gating
//! with a running max stabilizer and a twin `(c, n)` normalizer recurrence.
//!
//! Port-only (Phase 9 Task 2) -- NO legacy source. Spec:
//! `docs/superpowers/specs/2026-07-27-phase-9-new-architectures-design.md`, S2.1
//! (forward), S2.2 (flat layout), S2.3 (backward + the m-invariance argument), S2.4
//! (init + numerics). Numeric divergence from anything legacy is BY DESIGN and is
//! documented here + in `RESULTS.md`, never `IMPROVEMENTS.md` (the phase-7 rule).
//!
//! What landed: the whole-sequence f64 forward (S2.1), a HAND-DERIVED analytic
//! backward (S2.3, pinned against central differences by
//! `tests/phase9_cell_grad.rs`), and the `Layer`-conforming weight/derivative seam in
//! the S2.2 flat order. Training-ready: the layer plugs into `Network<CellLayer>`
//! and therefore into every windowed driver, the flat-weight seam, and SMORMS3,
//! with no caller aware of the cell type.
//!
//! ## Forward (S2.1)
//!
//! ```text
//! a~_t = W_a x_t + R_a h_{t-1} + b_a          for a in {i, f, o, z}
//! m_t  = max(f~_t + m_{t-1}, i~_t)            (stabilizer; m_{-1} = M_INIT)
//! i'_t = exp(i~_t - m_t)     f'_t = exp(f~_t + m_{t-1} - m_t)
//! c_t  = f'_t c_{t-1} + i'_t tanh(z~_t)       n_t = f'_t n_{t-1} + i'_t
//! h_t  = sigmoid(o~_t) * (c_t / n_t)
//! ```
//!
//! State is zero-init per call (`c_{-1} = n_{-1} = h_{-1} = 0`) and `m_{-1} =
//! [`M_INIT`]`. At `t = 0` the max resolves to `m_0 = i~_0` (the sentinel is
//! absorbed: `f~_0 + (-1e30) == -1e30` for any `|f~_0| < ~1e14`), so `i'_0 = exp(0) =
//! 1` and `f'_0 = exp(-1e30 - i~_0)` UNDERFLOWS to exactly 0 -- which is what makes
//! the zero initial state inert rather than merely small. That is the official xLSTM
//! `m_1 = i~_1` convention (S2.4), reached without an `if t == 0` branch in the
//! recurrence.
//!
//! `max(f', i') = exp(0) = 1` by construction, so `n_t >= 1 > 0` for every `t` by
//! induction (`n_0 = i'_0 = 1`; if `f' = 1` then `n_t >= n_{t-1}`, else `i' = 1` and
//! `n_t >= 1`): the `c/n` division cannot divide by zero, and no `exp` can overflow
//! (both arguments are `<= 0`). The one assumption is `|f~| < ~1e30` -- an absurd
//! pre-activation would overrun the sentinel.
//!
//! ## Backward (S2.3): why `m` is held CONSTANT, and why that is EXACT
//!
//! By induction `c_t = C_t e^{-m_t}` and `n_t = N_t e^{-m_t}` for the UNSTABILIZED
//! `C_t = e^{f~_t} C_{t-1} + e^{i~_t} z_t`, `N_t = e^{f~_t} N_{t-1} + e^{i~_t}`:
//!
//! ```text
//! c_t = e^{f~_t + m_{t-1} - m_t} (C_{t-1} e^{-m_{t-1}}) + e^{i~_t - m_t} z_t
//!     = e^{-m_t} (e^{f~_t} C_{t-1} + e^{i~_t} z_t) = e^{-m_t} C_t
//! ```
//!
//! so `c_t/n_t = C_t/N_t` is analytically m-INVARIANT, and `h` depends on `m` only
//! through that ratio. Hence `dh/dm = 0` and the partial derivative taken at a FROZEN
//! `m` schedule IS the total derivative -- not an approximation. It also removes the
//! `max()` kink from the gradient path entirely: there is no subgradient choice to
//! make. Differentiating at frozen `m` gives `d i'/d i~ = i'` and `d f'/d f~ = f'`
//! (the cached exp values themselves), which is the whole reason the backward reads
//! `[`gates`]` and never `[`m_states`]`.
//!
//! Adjoint equations, reverse time (`da~` = the pre-activation delta of gate `a`):
//!
//! ```text
//! dh_t  = deltas_t + sum_a R_a^T da~_{t+1}            (second term absent at t = T-1)
//! do~_t = dh_t * y_t * o_t (1 - o_t)                  y_t = c_t/n_t, o_t = sigmoid(o~_t)
//! dy_t  = dh_t * o_t
//! dc_t  = dy_t / n_t          + f'_{t+1} dc_{t+1}     (quotient rule + the twin carry)
//! dn_t  = -dy_t * y_t / n_t   + f'_{t+1} dn_{t+1}
//! df~_t = (dc_t c_{t-1} + dn_t n_{t-1}) * f'_t
//! di~_t = (dc_t z_t + dn_t) * i'_t
//! dz~_t = dc_t i'_t (1 - z_t^2)
//! dW_a += da~_t (x) x_t   dR_a += da~_t (x) h_{t-1}   db_a += da~_t
//! dx_t  = sum_a W_a^T da~_t
//! ```
//!
//! ## A structural degeneracy the gradient reproduces exactly: `b_i`
//!
//! The same `c_t = C_t e^{-m_t}` identity says the INPUT-GATE BIAS has an EXACTLY
//! ZERO gradient. Shift `i~_t[j] -> i~_t[j] + b` for all `t` (which is precisely what
//! moving `b_i[j]` does): by induction `C_t -> e^b C_t` and `N_t -> e^b N_t`, so
//! `c_t/n_t` -- and therefore `h_t`, and therefore every downstream pre-activation --
//! is UNCHANGED. Only the RELATIVE size of `i~` across time matters; its level is
//! absorbed by the normalizer `n`. (`b_f` has no such invariance: shifting `f~`
//! does not scale the `t = 0` term, which carries no forget contribution.)
//!
//! Consequences, both intentional: `input_gate_bias_shift_leaves_the_output_invariant`
//! pins the forward invariance and the ~0 analytic gradient, and the FD tier
//! (`tests/phase9_cell_grad.rs`) scores those weights on an ABSOLUTE floor -- a
//! relative error is meaningless where the true derivative is zero. `b_i` is a
//! non-identifiable parameter: a from-scratch run will never move it, whatever the
//! seeded init puts there.
//!
//! ## Conventions inherited from the house (`nn::layers::LstmLayer`)
//!
//! - Weights live in COMBINED gate-block matrices (`input_weights` `in x 4O`,
//!   `feedback_weights` `O x 4O`, `biases` `1 x 4O`, column blocks `[i|f|o|z]`), so
//!   the projections are the same [`matmul_seq`] calls the LSTM makes; the S2.2 flat
//!   order is a (de)serialization concern only, walked ONCE by [`for_each_slot`].
//! - `get_weights_derivatives` returns Nx2 `[summed deriv | frame count]` in the flat
//!   order, `ponderate_weights_derivatives` scales col0 only, `set_weights` returns
//!   the unconsumed tail, `inv_sub_sampling_ratio > 1` scales the deriv blocks.
//! - `feed_forward_reverse` / `feed_backward_reverse` flip rows around the
//!   forward-order body and leave the caches in REVERSED order (consumed as-is by the
//!   reverse backward), exactly as `LstmLayer` does.
//! - Input WIDTH TOLERANCE: `cols > in` crops to the left `in` columns, `cols < in`
//!   zero-pads to `in` (identical to the LSTM's `leftCols`/`topRows` pair, since
//!   padding the input with zeros and dropping the unused weight rows compute the
//!   same product). The propagated deltas are always `T x in`.
//! - All summations are ASCENDING loops (or [`matmul_seq`], which is the same
//!   ascending contract): no reordering for style.

use ndarray::Array2;

use super::super::activations::logistic_fn;
use super::super::layers::matmul_seq;
use super::super::network::Layer;

/// `m_{-1}` (spec S2.4): a documented finite constant standing in for `-inf`. Chosen
/// so `f~ + M_INIT` is absorbed back to `M_INIT` for any sane pre-activation, making
/// `m_0 = i~_0` and `f'_0 = 0` exactly.
pub const M_INIT: f64 = -1e30;

/// Gate-block column order inside the combined matrices and the S2.2 flat layout:
/// `[i | f | o | z]`.
const GATE_NB: usize = 4;
const BLOCK_I: usize = 0;
const BLOCK_F: usize = 1;
const BLOCK_O: usize = 2;
const BLOCK_Z: usize = 3;

/// Which combined matrix a flat-layout slot addresses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotKind {
    /// `feedback_weights` (the `R_a` recurrent block).
    Recurrent,
    /// `input_weights` (the `W_a` input block).
    Input,
    /// `biases` (the `b_a` block).
    Bias,
}

/// Walk the S2.2 flat layout EXACTLY ONCE, in order, calling `f(kind, row, col)` with
/// the COMBINED-matrix indices of each successive flat element:
///
/// ```text
/// for a in [i, f, o, z]:  R_a (out x out) row-major, W_a (out x in) row-major, b_a (out)
/// ```
///
/// Row-major per matrix means the OUTPUT unit `j` is the outer index and the source
/// index `k` the inner one; the combined matrices are transposed relative to that
/// (`feedback_weights[[k, a*out + j]] == R_a[j, k]`), which is why this walk exists:
/// `set_weights`, `get_weights` and `get_weights_derivatives` all drive it, so the
/// three orders cannot drift apart (spec S1.3 -- the layout is defined once).
fn for_each_slot(output_size: usize, input_size: usize, mut f: impl FnMut(SlotKind, usize, usize)) {
    for a in 0..GATE_NB {
        for j in 0..output_size {
            for k in 0..output_size {
                f(SlotKind::Recurrent, k, a * output_size + j);
            }
        }
        for j in 0..output_size {
            for k in 0..input_size {
                f(SlotKind::Input, k, a * output_size + j);
            }
        }
        for j in 0..output_size {
            f(SlotKind::Bias, 0, a * output_size + j);
        }
    }
}

/// The sLSTM cell (spec S2). See the module doc for the math, the layout and the
/// house conventions it inherits.
#[derive(Clone)]
pub struct SlstmLayer {
    input_size: usize,
    output_size: usize,

    /// `in x 4O`, column blocks `[i|f|o|z]`: `[[k, a*O + j]] == W_a[j, k]`.
    input_weights: Array2<f64>,
    /// `O x 4O`, same blocks: `[[k, a*O + j]] == R_a[j, k]`.
    feedback_weights: Array2<f64>,
    /// `1 x 4O`, same blocks.
    biases: Array2<f64>,

    /// Forward cache, `T x 4O`, blocks `[i' | f' | o | z]`: the POST-gating values
    /// the backward multiplies -- the STABILIZED exps `i' = exp(i~ - m)` /
    /// `f' = exp(f~ + m_prev - m)` (whose derivative w.r.t. their pre-activation is
    /// themselves, at frozen `m`), `o = sigmoid(o~)` and `z = tanh(z~)` (whose
    /// derivatives `o(1-o)` / `1 - z^2` are functions of the post-activation too).
    /// No pre-activation is needed by the backward, so none is cached.
    gates: Array2<f64>,
    /// Forward cache, `T x O`: `c_t` (the numerator state).
    cell_states: Array2<f64>,
    /// Forward cache, `T x O`: `n_t` (the normalizer state), `>= 1` by construction.
    norm_states: Array2<f64>,
    /// Forward cache, `T x O`: `m_t`. The backward NEVER reads this (spec S2.3 --
    /// `m` is held constant, which is exact); it is kept for the stabilizer unit
    /// probes and as the state a future streaming session must carry.
    m_states: Array2<f64>,

    /// Deriv accumulators, shaped like their weight twins, plus the frame count
    /// (`col1` of the Nx2 harvest). Reset via `reset_weights_derivatives`,
    /// accumulated across `feed_backward` calls.
    input_weights_derivatives: Array2<f64>,
    feedback_weights_derivatives: Array2<f64>,
    biases_derivatives: Array2<f64>,
    nb_of_seq_fed_backward: i64,
}

impl SlstmLayer {
    /// New layer with zeroed weights (callers set them via `set_weights`, exactly
    /// like `LstmLayer::new` with `weightsSetExternally = true`). Seeded init lives
    /// Python-side (`init_weights.py`, spec S2.4/S7.1).
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let four = GATE_NB * output_size;
        SlstmLayer {
            input_size,
            output_size,
            input_weights: Array2::zeros((input_size, four)),
            feedback_weights: Array2::zeros((output_size, four)),
            biases: Array2::zeros((1, four)),
            gates: Array2::zeros((0, 0)),
            cell_states: Array2::zeros((0, 0)),
            norm_states: Array2::zeros((0, 0)),
            m_states: Array2::zeros((0, 0)),
            input_weights_derivatives: Array2::zeros((input_size, four)),
            feedback_weights_derivatives: Array2::zeros((output_size, four)),
            biases_derivatives: Array2::zeros((1, four)),
            nb_of_seq_fed_backward: 0,
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// `4 * out * (out + in + 1)` (spec S2.2): per gate, `out x out` recurrent +
    /// `out x in` input + `out` biases. NO peepholes, NO cell-narrow asymmetry --
    /// all four blocks are the same width (unlike the legacy LSTM's).
    pub fn nb_of_weights(&self) -> usize {
        GATE_NB * self.output_size * (self.output_size + self.input_size + 1)
    }

    /// Consume `flat[..nb_of_weights()]` in the S2.2 order and return the tail for the
    /// next layer in the stack (the chained head-eating contract).
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        let (needed, left) = flat.split_at(self.nb_of_weights());
        let (o, i) = (self.output_size, self.input_size);
        let (fb, iw, bs) = (
            &mut self.feedback_weights,
            &mut self.input_weights,
            &mut self.biases,
        );
        let mut p = 0usize;
        for_each_slot(o, i, |kind, r, c| {
            let m: &mut Array2<f64> = match kind {
                SlotKind::Recurrent => fb,
                SlotKind::Input => iw,
                SlotKind::Bias => bs,
            };
            m[[r, c]] = needed[p];
            p += 1;
        });
        left
    }

    /// The mirror of [`Self::set_weights`]: append this layer's weights to `out` in
    /// the same S2.2 order.
    pub fn get_weights(&self, out: &mut Vec<f64>) {
        let (fb, iw, bs) = (&self.feedback_weights, &self.input_weights, &self.biases);
        for_each_slot(self.output_size, self.input_size, |kind, r, c| {
            let m: &Array2<f64> = match kind {
                SlotKind::Recurrent => fb,
                SlotKind::Input => iw,
                SlotKind::Bias => bs,
            };
            out.push(m[[r, c]]);
        });
    }

    /// Nx2 `[summed deriv | frame count]` harvest in the SAME S2.2 order, so row `k`
    /// is the derivative of flat weight `k` (the seam's contract).
    pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        let count = self.nb_of_seq_fed_backward as f64;
        let (fb, iw, bs) = (
            &self.feedback_weights_derivatives,
            &self.input_weights_derivatives,
            &self.biases_derivatives,
        );
        for_each_slot(self.output_size, self.input_size, |kind, r, c| {
            let m: &Array2<f64> = match kind {
                SlotKind::Recurrent => fb,
                SlotKind::Input => iw,
                SlotKind::Bias => bs,
            };
            out.push([m[[r, c]], count]);
        });
    }

    /// Zero the deriv accumulators and the frame count.
    pub fn reset_weights_derivatives(&mut self) {
        self.nb_of_seq_fed_backward = 0;
        self.input_weights_derivatives.fill(0.0);
        self.feedback_weights_derivatives.fill(0.0);
        self.biases_derivatives.fill(0.0);
    }

    /// Scale the deriv accumulators (the harvested col0) in place; the frame count
    /// (col1) is deliberately untouched.
    pub fn ponderate_weights_derivatives(&mut self, factor: f64) {
        self.input_weights_derivatives *= factor;
        self.feedback_weights_derivatives *= factor;
        self.biases_derivatives *= factor;
    }

    /// Post-gating cache (`T x 4O`, blocks `[i'|f'|o|z]`) from the last forward.
    pub fn gates(&self) -> &Array2<f64> {
        &self.gates
    }

    /// `c_t` cache (`T x O`) from the last forward.
    pub fn cell_states(&self) -> &Array2<f64> {
        &self.cell_states
    }

    /// `n_t` cache (`T x O`) from the last forward.
    pub fn norm_states(&self) -> &Array2<f64> {
        &self.norm_states
    }

    /// `m_t` cache (`T x O`) from the last forward. Never read by the backward.
    pub fn m_states(&self) -> &Array2<f64> {
        &self.m_states
    }

    /// The input at the layer's own width: crop (`cols > in`) or zero-pad
    /// (`cols < in`). One representation used by BOTH passes, so the forward
    /// projection and the backward's `dW` accumulation can never disagree about
    /// which columns exist.
    fn reconcile_input(&self, input: &Array2<f64>) -> Array2<f64> {
        let (t, cols) = input.dim();
        if cols == self.input_size {
            return input.to_owned();
        }
        let mut out = Array2::<f64>::zeros((t, self.input_size));
        let keep = cols.min(self.input_size);
        for r in 0..t {
            for k in 0..keep {
                out[[r, k]] = input[[r, k]];
            }
        }
        out
    }

    /// Whole-sequence forward (spec S2.1). Fills `output` (`T x O`) and the four
    /// caches. `last_layer` is unused (signature parity with the trait).
    pub fn feed_forward(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        let _ = last_layer;
        let o = self.output_size;
        let recon = self.reconcile_input(input);
        let t_len = recon.dim().0;

        // Batched input projection + biases (the recurrent term is per-step: it
        // needs h_{t-1}, which does not exist yet).
        let mut pre = matmul_seq(&recon, &self.input_weights); // T x 4O
        for t in 0..t_len {
            for j in 0..GATE_NB * o {
                pre[[t, j]] += self.biases[[0, j]];
            }
        }

        let mut gates = Array2::<f64>::zeros((t_len, GATE_NB * o));
        let mut cell_states = Array2::<f64>::zeros((t_len, o));
        let mut norm_states = Array2::<f64>::zeros((t_len, o));
        let mut m_states = Array2::<f64>::zeros((t_len, o));
        output.fill(0.0);

        for t in 0..t_len {
            if t > 0 {
                let prev = output.slice(ndarray::s![t - 1..t, ..]).to_owned(); // 1 x O
                let rec = matmul_seq(&prev, &self.feedback_weights); // 1 x 4O
                for j in 0..GATE_NB * o {
                    pre[[t, j]] += rec[[0, j]];
                }
            }
            for j in 0..o {
                let (m_prev, c_prev, n_prev) = if t > 0 {
                    (
                        m_states[[t - 1, j]],
                        cell_states[[t - 1, j]],
                        norm_states[[t - 1, j]],
                    )
                } else {
                    (M_INIT, 0.0, 0.0)
                };

                let i_pre = pre[[t, BLOCK_I * o + j]];
                let fm = pre[[t, BLOCK_F * o + j]] + m_prev;
                // max(fm, i~) written as an explicit comparison (NOT `f64::max`,
                // whose NaN handling differs from a plain ternary).
                let m_t = if fm > i_pre { fm } else { i_pre };
                let ip = (i_pre - m_t).exp();
                let fp = (fm - m_t).exp();
                let z = pre[[t, BLOCK_Z * o + j]].tanh();
                let og = logistic_fn(pre[[t, BLOCK_O * o + j]]);

                let c_t = fp * c_prev + ip * z;
                let n_t = fp * n_prev + ip;

                m_states[[t, j]] = m_t;
                cell_states[[t, j]] = c_t;
                norm_states[[t, j]] = n_t;
                gates[[t, BLOCK_I * o + j]] = ip;
                gates[[t, BLOCK_F * o + j]] = fp;
                gates[[t, BLOCK_O * o + j]] = og;
                gates[[t, BLOCK_Z * o + j]] = z;
                output[[t, j]] = og * (c_t / n_t);
            }
        }

        self.gates = gates;
        self.cell_states = cell_states;
        self.norm_states = norm_states;
        self.m_states = m_states;
    }

    /// Reverse-direction forward: flip the input rows, run [`Self::feed_forward`],
    /// flip the output back. The caches are LEFT in reversed-input order (consumed
    /// as-is by [`Self::feed_backward_reverse`]) -- the `LstmLayer` convention.
    pub fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();
        let mut output_rev = Array2::<f64>::zeros(output.dim());
        self.feed_forward(&input_rev, &mut output_rev, last_layer);
        output.assign(&output_rev.slice(ndarray::s![..;-1, ..]));
    }

    /// Whole-sequence BPTT (spec S2.3, the adjoint equations in the module doc).
    /// Accumulates the three deriv blocks + the frame count, and returns the deltas
    /// propagated to the previous layer (`T x in`).
    ///
    /// `output` supplies `h_{t-1}` for the recurrent accumulation (the caller passes
    /// the retained layer output, exactly as `LstmLayer::feed_backward` reads it);
    /// `deltas` is `dL/dh`. `last_layer` is unused.
    pub fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let _ = last_layer;
        let o = self.output_size;
        let i = self.input_size;
        let lines_nb = deltas.dim().0;
        let recon_input = self.reconcile_input(input);

        let g = &self.gates;
        let cs = &self.cell_states;
        let ns = &self.norm_states;

        // Per-call accumulators, folded into the members at the end.
        let mut iw_d = Array2::<f64>::zeros((i, GATE_NB * o));
        let mut fw_d = Array2::<f64>::zeros((o, GATE_NB * o));
        let mut bs_d = Array2::<f64>::zeros((1, GATE_NB * o));

        // Carried backward across the reverse-time loop: the recurrent dh term from
        // the LATER step, and the twin state adjoints already multiplied by
        // `f'_{t+1}` (so at row `t` they are exactly the `+ f'_{t+1} d*_{t+1}` terms).
        let mut deltas_feedback = vec![0.0f64; o];
        let mut carried_dc = vec![0.0f64; o];
        let mut carried_dn = vec![0.0f64; o];

        // Per-row pre-activation deltas, stacked [i|f|o|z] into a 4O x T matrix for
        // the two GEMMs (the LSTM's `testM` pattern).
        let mut test_m = Array2::<f64>::zeros((GATE_NB * o, lines_nb));

        for row in (0..lines_nb).rev() {
            for j in 0..o {
                let ip = g[[row, BLOCK_I * o + j]];
                let fp = g[[row, BLOCK_F * o + j]];
                let og = g[[row, BLOCK_O * o + j]];
                let z = g[[row, BLOCK_Z * o + j]];
                let n_t = ns[[row, j]];
                let y_t = cs[[row, j]] / n_t;
                let (c_prev, n_prev) = if row > 0 {
                    (cs[[row - 1, j]], ns[[row - 1, j]])
                } else {
                    (0.0, 0.0)
                };

                // dh_t = incoming deltas + the recurrent term from step t+1.
                let dh = deltas[[row, j]] + deltas_feedback[j];
                // h = o * y  ->  do~ (sigmoid') and dy.
                let d_o_pre = dh * y_t * og * (1.0 - og);
                let dy = dh * og;
                // y = c/n (quotient rule) + the twin carries from step t+1.
                let dc = dy / n_t + carried_dc[j];
                let dn = -dy * y_t / n_t + carried_dn[j];
                // c = f' c_prev + i' z ; n = f' n_prev + i'  (m frozen: di'/di~ = i',
                // df'/df~ = f').
                let d_f_pre = (dc * c_prev + dn * n_prev) * fp;
                let d_i_pre = (dc * z + dn) * ip;
                let d_z_pre = dc * ip * (1.0 - z * z);
                // Carries for the next (earlier) row.
                carried_dc[j] = dc * fp;
                carried_dn[j] = dn * fp;

                test_m[[BLOCK_I * o + j, row]] = d_i_pre;
                test_m[[BLOCK_F * o + j, row]] = d_f_pre;
                test_m[[BLOCK_O * o + j, row]] = d_o_pre;
                test_m[[BLOCK_Z * o + j, row]] = d_z_pre;
            }

            // Weight-deriv accumulation for all four blocks at once (k = 1 outer
            // products, done as explicit ascending loops).
            for k in 0..i {
                for c in 0..GATE_NB * o {
                    iw_d[[k, c]] += recon_input[[row, k]] * test_m[[c, row]];
                }
            }
            if row > 0 {
                for k in 0..o {
                    for c in 0..GATE_NB * o {
                        fw_d[[k, c]] += output[[row - 1, k]] * test_m[[c, row]];
                    }
                }
            }
            for c in 0..GATE_NB * o {
                bs_d[[0, c]] += test_m[[c, row]];
            }

            // dh_{t-1} = sum_a R_a^T da~_t, for the next (earlier) iteration.
            let test_col = test_m.slice(ndarray::s![.., row..row + 1]).to_owned();
            let fb = matmul_seq(&self.feedback_weights, &test_col); // O x 1
            for j in 0..o {
                deltas_feedback[j] = fb[[j, 0]];
            }
        }

        // dx = (W * testM)^T -> T x in.
        let dpl = matmul_seq(&self.input_weights, &test_m); // in x T
        let deltas_previous_layer = dpl.t().to_owned();

        if inv_sub_sampling_ratio > 1 {
            let r = inv_sub_sampling_ratio as f64;
            iw_d.mapv_inplace(|v| v * r);
            fw_d.mapv_inplace(|v| v * r);
            bs_d.mapv_inplace(|v| v * r);
        }

        self.input_weights_derivatives += &iw_d;
        self.feedback_weights_derivatives += &fw_d;
        self.biases_derivatives += &bs_d;
        self.nb_of_seq_fed_backward += lines_nb as i64;

        deltas_previous_layer
    }

    /// Reverse-time BPTT: flip input/output/deltas, run the forward-order body, flip
    /// the returned deltas back. The caches are already time-reversed by
    /// [`Self::feed_forward_reverse`] and are consumed AS-IS (do NOT un-reverse).
    pub fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();
        let output_rev = output.slice(ndarray::s![..;-1, ..]).to_owned();
        let deltas_rev = deltas.slice(ndarray::s![..;-1, ..]).to_owned();
        let dpl = self.feed_backward(
            &input_rev,
            &output_rev,
            &deltas_rev,
            inv_sub_sampling_ratio,
            last_layer,
        );
        dpl.slice(ndarray::s![..;-1, ..]).to_owned()
    }
}

/// The trait impl is a thin forward to the inherent methods, mirroring
/// `impl Layer for LstmLayer` (`nn::network`). It lives HERE rather than in
/// `nn/network.rs` so the phase's new-cell code stays inside `nn/cells/` (spec S1.4);
/// `CellLayer`'s match arms call the INHERENT methods by UFCS, so the two paths
/// cannot diverge.
impl Layer for SlstmLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        SlstmLayer::feed_forward(self, input, output, last_layer);
    }
    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        SlstmLayer::feed_forward_reverse(self, input, output, last_layer);
    }
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        SlstmLayer::feed_backward(
            self,
            input,
            output,
            deltas,
            inv_sub_sampling_ratio,
            last_layer,
        )
    }
    fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        SlstmLayer::feed_backward_reverse(
            self,
            input,
            output,
            deltas,
            inv_sub_sampling_ratio,
            last_layer,
        )
    }
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        SlstmLayer::get_weights_derivatives(self, out);
    }
    fn reset_weights_derivatives(&mut self) {
        SlstmLayer::reset_weights_derivatives(self);
    }
    fn ponderate_weights_derivatives(&mut self, factor: f64) {
        SlstmLayer::ponderate_weights_derivatives(self, factor);
    }
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        SlstmLayer::set_weights(self, flat)
    }
    fn get_weights(&self, out: &mut Vec<f64>) {
        SlstmLayer::get_weights(self, out);
    }
    fn nb_of_weights(&self) -> usize {
        SlstmLayer::nb_of_weights(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: usize = 3;
    const O: usize = 2;
    const T: usize = 7;

    /// Deterministic, non-degenerate weight fill (no RNG dependency).
    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.31 - 0.013 * (k as f64) + 0.007 * ((k % 5) as f64))
            .collect()
    }

    fn seq(rows: usize, cols: usize, seed: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            seed + 0.17 * (r as f64) - 0.23 * (c as f64) + 0.011 * ((r * cols + c) as f64)
        })
    }

    fn loaded(i: usize, o: usize) -> SlstmLayer {
        let mut cell = SlstmLayer::new(i, o);
        cell.set_weights(&weights(cell.nb_of_weights()));
        cell
    }

    fn forward(cell: &mut SlstmLayer, input: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((input.nrows(), cell.output_size()));
        cell.feed_forward(input, &mut out, false);
        out
    }

    // --- Weight seam (spec S2.2) -------------------------------------------

    #[test]
    fn nb_of_weights_is_the_spec_formula() {
        for (i, o) in [(3usize, 2usize), (7, 3), (1, 1), (40, 64)] {
            assert_eq!(
                SlstmLayer::new(i, o).nb_of_weights(),
                4 * o * (o + i + 1),
                "in={i} out={o}"
            );
        }
    }

    /// The flat layout is `[i|f|o|z]` gate blocks, each `[R | W | b]` ROW-major --
    /// checked positionally, not just by round trip (a round trip alone would pass
    /// on any self-consistent permutation).
    #[test]
    fn flat_layout_positions_are_the_spec_order() {
        let (i, o) = (3usize, 2usize);
        let mut cell = SlstmLayer::new(i, o);
        let n = cell.nb_of_weights();
        // A one-hot probe per position tells us exactly which matrix cell it lands in.
        let block = o * o + o * i + o; // per-gate stride
        assert_eq!(n, 4 * block);
        for (gate, name) in [(0usize, "i"), (1, "f"), (2, "o"), (3, "z")] {
            let base = gate * block;
            // R_a row-major: element (j, k) at base + j*o + k -> feedback[[k, a*o+j]].
            for j in 0..o {
                for k in 0..o {
                    let mut flat = vec![0.0; n];
                    flat[base + j * o + k] = 1.0;
                    cell.set_weights(&flat);
                    assert_eq!(
                        cell.feedback_weights[[k, gate * o + j]],
                        1.0,
                        "gate {name}: R[{j},{k}] misplaced"
                    );
                }
            }
            // W_a row-major: element (j, k) at base + o*o + j*i + k -> input[[k, a*o+j]].
            for j in 0..o {
                for k in 0..i {
                    let mut flat = vec![0.0; n];
                    flat[base + o * o + j * i + k] = 1.0;
                    cell.set_weights(&flat);
                    assert_eq!(
                        cell.input_weights[[k, gate * o + j]],
                        1.0,
                        "gate {name}: W[{j},{k}] misplaced"
                    );
                }
            }
            // b_a: element j at base + o*o + o*i + j.
            for j in 0..o {
                let mut flat = vec![0.0; n];
                flat[base + o * o + o * i + j] = 1.0;
                cell.set_weights(&flat);
                assert_eq!(
                    cell.biases[[0, gate * o + j]],
                    1.0,
                    "gate {name}: b[{j}] misplaced"
                );
            }
        }
    }

    #[test]
    fn weight_seam_round_trips_and_returns_the_tail() {
        let mut cell = SlstmLayer::new(I, O);
        let nb = cell.nb_of_weights();
        let w = weights(nb + 5);
        let tail = cell.set_weights(&w);
        assert_eq!(tail.len(), 5);
        assert_eq!(tail, &w[nb..]);

        let mut back = Vec::new();
        cell.get_weights(&mut back);
        assert_eq!(back, w[..nb].to_vec());
    }

    // --- Forward (spec S2.1) ------------------------------------------------

    #[test]
    fn forward_shapes_and_caches_track_the_sequence_length() {
        for t in [1usize, 2, 7, 11] {
            let mut cell = loaded(I, O);
            let out = forward(&mut cell, &seq(t, I, 0.4));
            assert_eq!(out.dim(), (t, O));
            assert_eq!(cell.gates().dim(), (t, 4 * O));
            assert_eq!(cell.cell_states().dim(), (t, O));
            assert_eq!(cell.norm_states().dim(), (t, O));
            assert_eq!(cell.m_states().dim(), (t, O));
            assert!(out.iter().all(|v| v.is_finite()), "t={t}");
        }
    }

    /// The forward is a pure function of (weights, input): run twice, bit-identical
    /// (no state leaks across calls -- spec S2.4 zero-init per call).
    #[test]
    fn forward_is_run_twice_bit_identical_and_stateless() {
        let mut cell = loaded(I, O);
        let input = seq(T, I, 0.4);
        let first = forward(&mut cell, &input);
        let (g1, c1, n1, m1) = (
            cell.gates().clone(),
            cell.cell_states().clone(),
            cell.norm_states().clone(),
            cell.m_states().clone(),
        );
        let second = forward(&mut cell, &input);
        assert_eq!(first, second);
        assert_eq!(&g1, cell.gates());
        assert_eq!(&c1, cell.cell_states());
        assert_eq!(&n1, cell.norm_states());
        assert_eq!(&m1, cell.m_states());

        // A fresh layer with the same weights agrees too (no ctor-order dependency).
        let mut fresh = loaded(I, O);
        assert_eq!(forward(&mut fresh, &input), first);
    }

    /// HAND-COMPUTED single step (1 unit, 1 input): every quantity of S2.1 written
    /// out with literal constants. The algebra is exact in f64 (`0*0 = 0`,
    /// `1*x = x`, `x/1 = x`), so this asserts EQUALITY, not a tolerance.
    #[test]
    fn single_step_matches_the_hand_computed_value() {
        let (i, o) = (1usize, 1usize);
        let mut cell = SlstmLayer::new(i, o);
        // Layout per gate: [R (1x1) | W (1x1) | b (1)] for [i, f, o, z].
        let flat = vec![
            0.9, 0.3, 0.1, // i: R (unused at t=0), W, b
            0.7, -0.4, 0.2, // f
            -0.5, 0.6, -0.1, // o
            0.2, 0.8, 0.05, // z
        ];
        assert_eq!(flat.len(), cell.nb_of_weights());
        cell.set_weights(&flat);

        let x = 0.5;
        let input = Array2::from_shape_vec((1, 1), vec![x]).unwrap();
        let out = forward(&mut cell, &input);

        let i_pre = x * 0.3 + 0.1;
        let f_pre = x * -0.4 + 0.2;
        let o_pre = x * 0.6 + -0.1;
        let z_pre = x * 0.8 + 0.05;

        // m_0 = max(f~ + M_INIT, i~) = i~ (the sentinel absorbs f~).
        assert_eq!(f_pre + M_INIT, M_INIT, "the sentinel must absorb f~");
        assert_eq!(cell.m_states()[[0, 0]], i_pre);
        // i' = exp(0) = 1 exactly; f' underflows to exactly 0.
        assert_eq!(cell.gates()[[0, 0]], 1.0);
        assert_eq!(cell.gates()[[0, 1]], 0.0);
        // c = tanh(z~), n = 1, h = sigmoid(o~) * tanh(z~).
        assert_eq!(cell.cell_states()[[0, 0]], z_pre.tanh());
        assert_eq!(cell.norm_states()[[0, 0]], 1.0);
        let want = (1.0 / (1.0 + (-o_pre).exp())) * z_pre.tanh();
        assert_eq!(out[[0, 0]], want);
    }

    /// The stabilizer keeps `+-800` pre-activations finite where the UNSTABILIZED
    /// `exp` is `inf`: `max(i', f') = exp(0) = 1` and `n >= 1`, so nothing overflows
    /// and nothing divides by zero.
    #[test]
    fn stabilizer_survives_huge_pre_activations() {
        // Non-vacuity: this is the value the unstabilized cell would exponentiate.
        assert!(800.0_f64.exp().is_infinite());

        let (i, o) = (1usize, 1usize);
        for sign in [1.0f64, -1.0] {
            let mut cell = SlstmLayer::new(i, o);
            // W = 800*sign on every gate, x = +-1 -> pre-activations +-800.
            let flat = vec![
                0.0,
                800.0 * sign,
                0.0,
                0.0,
                800.0 * sign,
                0.0,
                0.0,
                800.0 * sign,
                0.0,
                0.0,
                800.0 * sign,
                0.0,
            ];
            cell.set_weights(&flat);
            let input = Array2::from_shape_vec((5, 1), vec![1.0, -1.0, 1.0, 1.0, -1.0]).unwrap();
            let out = forward(&mut cell, &input);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "sign={sign}: non-finite output {out:?}"
            );
            assert!(
                cell.norm_states().iter().all(|v| *v >= 1.0),
                "sign={sign}: n dropped below 1: {:?}",
                cell.norm_states()
            );
            assert!(cell.gates().iter().all(|v| v.is_finite()));
        }
    }

    /// `feed_forward_reverse` is the forward on flipped rows (and leaves the caches
    /// reversed, the `LstmLayer` convention the reverse backward depends on).
    #[test]
    fn reverse_forward_is_the_flipped_forward() {
        let input = seq(T, I, 0.4);
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();

        let mut a = loaded(I, O);
        let mut out_rev = Array2::<f64>::zeros((T, O));
        a.feed_forward_reverse(&input, &mut out_rev, false);

        let mut b = loaded(I, O);
        let plain_on_flipped = forward(&mut b, &input_rev);

        assert_eq!(
            out_rev,
            plain_on_flipped.slice(ndarray::s![..;-1, ..]).to_owned()
        );
        // Caches stay in reversed-input order.
        assert_eq!(a.gates(), b.gates());
    }

    // --- Backward (spec S2.3) ----------------------------------------------

    /// Derivatives ACCUMULATE across `feed_backward` calls (col0 sums, col1 counts
    /// frames) and `reset_weights_derivatives` clears both.
    #[test]
    fn derivatives_accumulate_across_calls_and_reset_clears() {
        let mut cell = loaded(I, O);
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);

        let out = forward(&mut cell, &input);
        cell.reset_weights_derivatives();
        let _ = cell.feed_backward(&input, &out, &deltas, 1, false);
        let mut once = Vec::new();
        cell.get_weights_derivatives(&mut once);
        assert!(
            once.iter().any(|r| r[0] != 0.0),
            "no derivative accumulated -- the test is vacuous"
        );
        assert!(once.iter().all(|r| r[1] == T as f64));

        let _ = cell.feed_backward(&input, &out, &deltas, 1, false);
        let mut twice = Vec::new();
        cell.get_weights_derivatives(&mut twice);
        for k in 0..once.len() {
            assert_eq!(twice[k][0], once[k][0] + once[k][0], "row {k} did not sum");
            assert_eq!(twice[k][1], 2.0 * T as f64);
        }

        cell.reset_weights_derivatives();
        let mut zeroed = Vec::new();
        cell.get_weights_derivatives(&mut zeroed);
        assert!(zeroed.iter().all(|r| r[0] == 0.0 && r[1] == 0.0));
    }

    /// THE `b_i` DEGENERACY (module doc): a uniform shift of the input-gate bias
    /// scales `C` and `N` equally, so the output is INVARIANT and the analytic
    /// gradient w.r.t. `b_i` is exactly zero. Pinned in both directions -- the
    /// forward invariance (so the zero is not an artifact of a broken backward) and
    /// the analytic zero -- against NON-ZERO gradients elsewhere in the same pack.
    #[test]
    fn input_gate_bias_shift_leaves_the_output_invariant() {
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);
        let base = weights(SlstmLayer::new(I, O).nb_of_weights());
        // b_i lives at [o*o + o*i, o*o + o*i + o) inside gate block 0.
        let bias_lo = O * O + O * I;

        let mut a = SlstmLayer::new(I, O);
        a.set_weights(&base);
        let out_a = forward(&mut a, &input);

        let mut shifted = base.clone();
        for j in 0..O {
            shifted[bias_lo + j] += 2.0;
        }
        let mut b = SlstmLayer::new(I, O);
        b.set_weights(&shifted);
        let out_b = forward(&mut b, &input);

        // Non-vacuity: the shift really did move the internal state (m and the exps
        // all change) -- only the c/n ratio, hence the output, is invariant.
        assert_ne!(
            a.m_states(),
            b.m_states(),
            "the shift was a no-op internally"
        );
        let max_dh = (0..T)
            .flat_map(|r| (0..O).map(move |c| (r, c)))
            .map(|(r, c)| (out_a[[r, c]] - out_b[[r, c]]).abs())
            .fold(0.0f64, f64::max);
        // MEASURED max |dh| over the 7x2 output = 5.55e-17 (pure rounding); pinned at
        // 1e-14.
        assert!(max_dh < 1e-14, "b_i shift moved the output by {max_dh:e}");

        a.reset_weights_derivatives();
        let _ = a.feed_backward(&input, &out_a, &deltas, 1, false);
        let mut rows = Vec::new();
        a.get_weights_derivatives(&mut rows);
        let b_i_grad = (0..O)
            .map(|j| rows[bias_lo + j][0].abs())
            .fold(0.0f64, f64::max);
        // MEASURED max |dL/db_i| = 6.25e-17 (cancellation of O(1) per-step terms);
        // pinned at 1e-14, against a max |grad| over the whole pack of ~2.8e-1.
        assert!(b_i_grad < 1e-14, "dL/db_i is not ~0: {b_i_grad:e}");
        let overall = rows.iter().map(|r| r[0].abs()).fold(0.0f64, f64::max);
        assert!(
            overall > 1e-3,
            "the rest of the gradient is ~0 too -- the contrast is vacuous"
        );
    }

    /// `ponderate_weights_derivatives` scales col0 only; `inv_sub_sampling_ratio`
    /// scales the accumulated blocks (the house conventions).
    #[test]
    fn ponderation_scales_col0_only_and_issr_scales_the_blocks() {
        let mut cell = loaded(I, O);
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);
        let out = forward(&mut cell, &input);

        cell.reset_weights_derivatives();
        let _ = cell.feed_backward(&input, &out, &deltas, 1, false);
        let mut base = Vec::new();
        cell.get_weights_derivatives(&mut base);
        cell.ponderate_weights_derivatives(0.25);
        let mut pond = Vec::new();
        cell.get_weights_derivatives(&mut pond);
        for k in 0..base.len() {
            assert_eq!(pond[k][0], base[k][0] * 0.25);
            assert_eq!(pond[k][1], base[k][1], "the frame count must not scale");
        }

        let mut scaled = loaded(I, O);
        let out2 = forward(&mut scaled, &input);
        scaled.reset_weights_derivatives();
        let _ = scaled.feed_backward(&input, &out2, &deltas, 3, false);
        let mut rows = Vec::new();
        scaled.get_weights_derivatives(&mut rows);
        for k in 0..base.len() {
            assert_eq!(rows[k][0], base[k][0] * 3.0, "row {k}");
        }
    }

    /// The reverse backward is the forward-order backward on flipped rows, with the
    /// propagated deltas flipped back.
    #[test]
    fn reverse_backward_is_the_flipped_backward() {
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);

        let mut a = loaded(I, O);
        let mut out_rev = Array2::<f64>::zeros((T, O));
        a.feed_forward_reverse(&input, &mut out_rev, false);
        let dpl_rev = a.feed_backward_reverse(&input, &out_rev, &deltas, 1, false);
        let mut da = Vec::new();
        a.get_weights_derivatives(&mut da);

        let mut b = loaded(I, O);
        let input_f = input.slice(ndarray::s![..;-1, ..]).to_owned();
        let out_f = forward(&mut b, &input_f);
        let dpl_f = b.feed_backward(
            &input_f,
            &out_f,
            &deltas.slice(ndarray::s![..;-1, ..]).to_owned(),
            1,
            false,
        );
        let mut db = Vec::new();
        b.get_weights_derivatives(&mut db);

        assert_eq!(dpl_rev, dpl_f.slice(ndarray::s![..;-1, ..]).to_owned());
        assert_eq!(da, db);
        assert!(da.iter().any(|r| r[0] != 0.0));
    }

    /// The propagated deltas are `T x in` (the layer's own width), even when the
    /// caller feeds a narrower or wider sequence -- the width-tolerance contract the
    /// forward and backward share through `reconcile_input`.
    #[test]
    fn width_tolerance_is_shared_by_both_passes() {
        let narrow = seq(T, 1, 0.4);
        let wide = seq(T, I + 2, 0.4);
        for input in [&narrow, &wide] {
            let mut cell = loaded(I, O);
            let out = forward(&mut cell, input);
            assert_eq!(out.dim(), (T, O));
            let dpl = cell.feed_backward(input, &out, &seq(T, O, -0.2), 1, false);
            assert_eq!(dpl.dim(), (T, I));
        }

        // Cropping is exactly "ignore the extra columns": a wide input agrees with
        // its own left `I` columns fed directly.
        let mut a = loaded(I, O);
        let mut b = loaded(I, O);
        assert_eq!(
            forward(&mut a, &wide),
            forward(&mut b, &wide.slice(ndarray::s![.., ..I]).to_owned())
        );
    }
}
