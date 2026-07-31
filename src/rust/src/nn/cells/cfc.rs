//! CfC: the closed-form continuous-time cell (the ncps lineage), in its GATED form
//! under the constant-`Delta t` reduction.
//!
//! Port-only (Phase 10 Task 1) -- NO legacy source. Spec:
//! `docs/superpowers/specs/2026-07-31-phase-10-cfc-fast-completion-design.md`, S1.1
//! (forward), S1.2 (flat layout), S1.3 (backward), S1.4-S1.6 (hyperparams, init, the
//! 4th-variant touch list). Numeric divergence from anything legacy is BY DESIGN and
//! is documented here + in `RESULTS.md`, never `IMPROVEMENTS.md` (the phase-7 rule).
//!
//! What landed: the whole-sequence f64 forward (S1.1), a HAND-DERIVED analytic
//! backward (S1.3, pinned against central differences by
//! `tests/phase9_cell_grad.rs`), and the `Layer`-conforming weight/derivative seam in
//! the S1.2 flat order. Training-ready: the layer plugs into `Network<CellLayer>` and
//! therefore into every windowed driver, the flat-weight seam, and SMORMS3, with no
//! caller aware of the cell type.
//!
//! ## Forward (S1.1)
//!
//! ```text
//! z_t    = [x_t | h_{t-1}]                            (width in + H)
//! a_1,t  = W_bb z_t + b_bb           bb_1,t = lt(a_1,t)             (width B)
//! a_l,t  = W_l bb_{l-1},t + b_l      bb_l,t = lt(a_l,t)   l = 2..L  (width B)
//! ff1_t  = tanh(W_1 bb_L,t + b_1)    ff2_t  = tanh(W_2 bb_L,t + b_2)      (H each)
//! g_t    = sigmoid(W_t bb_L,t + b_t)                                      (H)
//! h_t    = ff1_t (1 - g_t) + ff2_t g_t
//! ```
//!
//! with `lt` the LeCun tanh [`lecun_tanh`] (`1.7159 * tanh(2x/3)`) and `sigmoid` the
//! PLAIN logistic ([`logistic_fn`], the house `Logistic::fn` with its `expLimit`
//! saturation guards) -- NOT the 0.1-prescaled `GatesFunction` the legacy LSTM's gates
//! use. Output is `h_t`; the state is `h` alone, zero-init per call (`h_{-1} = 0`).
//! Natively causal, fixed-size state, O(T).
//!
//! ## THE CONSTANT-`Delta t` REDUCTION (why there is ONE gate head, not two)
//!
//! The published gated CfC interpolates between two heads with a time gate
//! `sigma(-t_a Delta t + t_b)` whose argument is affine in the elapsed time between
//! samples. Every consumer in this repo runs on UNIFORMLY framed audio: `Delta t` is
//! the frame shift, one constant for the whole corpus. An affine function of a
//! constant is a constant offset, so `-t_a Delta t + t_b` collapses to a single
//! learned pre-activation `W_t bb + b_t` -- the two time heads are not approximated
//! away, they are EXACTLY absorbed into one, with `b_t` free to represent
//! `t_b - t_a Delta t` at whatever `Delta t` the config framing implies. The
//! continuous-time lineage stays legible (this is still the closed-form gated
//! interpolation between two candidate states), and an irregular-`Delta t` variant --
//! which would need the elapsed time threaded in as a per-timestep INPUT -- is a
//! separate cell, not a config knob on this one.
//!
//! ## Backward (S1.3)
//!
//! Plain reverse-time BPTT: the ONLY recurrence path is `h_{t-1}` entering through
//! `z_t`'s tail slice, so the whole per-timestep block is an ordinary feed-forward
//! stack and its adjoints are activation locals times matmul transposes. `da~`
//! denotes the delta of a PRE-activation.
//!
//! ```text
//! dh_t     = deltas_t + carry_t            (carry_t = dz_{t+1}[in..], 0 at t = T-1)
//! dff1_t   = dh_t (1 - g_t)                dff2_t = dh_t g_t
//! dg_t     = dh_t (ff2_t - ff1_t)
//! du1_t    = dff1_t (1 - ff1_t^2)          du2_t  = dff2_t (1 - ff2_t^2)
//! dut_t    = dg_t g_t (1 - g_t)
//! dW_head += du_t (x) bb_L,t               db_head += du_t     (head in {1, 2, t})
//! dbb_L,t  = W_1^T du1_t + W_2^T du2_t + W_t^T dut_t
//!
//! for l = L .. 1:                          (in_l,t = bb_{l-1},t, and z_t at l = 1)
//!   da_l,t   = dbb_l,t * lt'(a_l,t)        lt'(v) = 1.7159 * (2/3) (1 - tanh^2(2v/3))
//!   dW_l    += da_l,t (x) in_l,t           db_l += da_l,t
//!   dbb_{l-1},t = W_l^T da_l,t
//!
//! dz_t     = W_bb^T da_1,t
//! dx_t     = dz_t[0 .. in]                 carry_{t-1} = dz_t[in .. in+H]
//! ```
//!
//! The head derivatives are taken on POST-activations (`1 - ff^2`, `g (1 - g)` -- the
//! house `LstmLayer`/`slstm` convention, which is why only `ff1`/`ff2`/`g` are
//! cached), the backbone derivative on the PRE-activation `a_l,t` (S1.3 spells `lt'`
//! that way), which is why BOTH backbone caches exist: the pre-activations feed `lt'`
//! and the post-activations are the inputs the `dW` outer products need.
//!
//! ## The one structurally gradient-dead block: `W_bb`'s state columns at `T = 1`
//!
//! `dW_bb[j, in+k] = sum_t da_1,t[j] * z_t[in+k] = sum_t da_1,t[j] * h_{t-1}[k]`, and
//! `h_{-1} = 0`. At `T = 1` the sum has exactly one term, against the zero initial
//! state, so those `B * H` weights have an EXACTLY zero derivative -- pinned as `==
//! 0.0` (not a tolerance) by `backbone_state_columns_are_gradient_dead_at_t1`, with a
//! `T = 2` contrast in the same test (the mamba `a_log_gradient_is_exactly_zero_at_t1`
//! precedent). It is a `T = 1` artefact, NOT a degeneracy: at any `T > 1` those
//! columns are live.
//!
//! NOTHING ELSE is dead. In particular there is no CfC analogue of the sLSTM `b_i`
//! non-identifiability: `tanh` and `sigmoid` are plain bounded activations with no
//! scale invariance for a bias shift to be absorbed into, and both heads feed the
//! output through live paths at every timestep.
//!
//! ## Conventions inherited from the house (`nn::layers::LstmLayer`, `cells::slstm`)
//!
//! - Every projection matrix is stored TRANSPOSED (`w[[k, j]] == W[j, k]`), so each
//!   product is a plain [`matmul_seq`] call -- the ascending accumulation contract,
//!   uniformly. The three heads live in ONE combined `B x 3H` matrix with column
//!   blocks `[ff1 | ff2 | gate]` (the sLSTM combined-gate-block precedent), so a
//!   timestep runs `L + 1` products and no more.
//! - The S1.2 flat order is a (de)serialization concern only, walked ONCE by
//!   [`for_each_slot`], so `set_weights`, `get_weights` and `get_weights_derivatives`
//!   cannot drift apart.
//! - `get_weights_derivatives` returns Nx2 `[summed deriv | frame count]` in the flat
//!   order, `ponderate_weights_derivatives` scales col0 only, `set_weights` returns
//!   the unconsumed tail, `inv_sub_sampling_ratio > 1` scales the deriv blocks.
//! - `feed_forward_reverse` / `feed_backward_reverse` flip rows around the
//!   forward-order body and leave the caches in REVERSED order (consumed as-is by the
//!   reverse backward). CfC is CAUSAL, so the reverse direction is a genuine
//!   anti-causal pass over the flipped sequence -- the same trick the wrapper uses to
//!   get a bidirectional stack out of a one-directional cell.
//! - Input WIDTH TOLERANCE: `cols > in` crops to the left `in` columns, `cols < in`
//!   zero-pads. The propagated deltas are always `T x in`.
//! - All summations are ASCENDING loops (or [`matmul_seq`], the same contract).

use ndarray::Array2;

use super::super::activations::logistic_fn;
use super::super::layers::matmul_seq;
use super::super::network::Layer;

/// LeCun-tanh amplitude (spec S1.1). The literal `1.7159` of the classic
/// `1.7159 tanh(2x/3)` parameterisation, chosen there so the map has unit gain and
/// `f(+-1) = +-1`; it is part of the cell definition, not a tunable.
pub const LECUN_SCALE: f64 = 1.7159;

/// LeCun-tanh input slope (spec S1.1): the `2/3` of `1.7159 tanh(2x/3)`. Written as a
/// division so the constant is the correctly-rounded f64 nearest `2/3`, not a
/// hand-typed decimal.
pub const LECUN_SLOPE: f64 = 2.0 / 3.0;

/// `lt(x) = 1.7159 tanh(2x/3)` (spec S1.1). No saturation guard is needed: `tanh` is
/// bounded and finite for every finite argument (and returns `+-1` at infinity).
fn lecun_tanh(x: f64) -> f64 {
    LECUN_SCALE * (LECUN_SLOPE * x).tanh()
}

/// `lt'(x) = 1.7159 * (2/3) * (1 - tanh^2(2x/3))`, taken on the PRE-activation `x`
/// (spec S1.3). Deliberately NOT expressed through the cached post-activation
/// `bb = 1.7159 tanh(2x/3)`: the algebraically equal `1.7159 * (2/3) * (1 -
/// (bb/1.7159)^2)` costs an extra division-and-rounding round trip for nothing, and
/// the pre-activations are cached anyway.
fn lecun_tanh_deriv(x: f64) -> f64 {
    let s = (LECUN_SLOPE * x).tanh();
    LECUN_SCALE * LECUN_SLOPE * (1.0 - s * s)
}

/// Column-block order inside the combined head matrix and the S1.2 flat layout:
/// `[ff1 | ff2 | gate]`.
const HEAD_NB: usize = 3;
const BLOCK_FF1: usize = 0;
const BLOCK_FF2: usize = 1;
const BLOCK_GATE: usize = 2;

/// Which matrix a flat-layout slot addresses. The backbone variants carry the LAYER
/// INDEX, since the backbone is a `Vec` of matrices rather than one combined block
/// (layer 0 is `(in+H) x B` and the deeper ones are `B x B`, so they cannot be
/// combined the way the three equal-shaped heads are).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotKind {
    BackboneWeight(usize),
    BackboneBias(usize),
    HeadWeight,
    HeadBias,
}

/// Walk the S1.2 flat layout EXACTLY ONCE, in order, calling `f(kind, row, col)` with
/// the STORED-matrix indices of each successive flat element:
///
/// ```text
/// W_bb (B x (in+H)) row-major | b_bb (B)
/// then per deeper backbone layer:  W_l (B x B) row-major | b_l (B)
/// then per head in [ff1, ff2, gate]:  W (H x B) row-major | b (H)
/// ```
///
/// Row-major per matrix means the OUTPUT unit `j` is the outer index and the source
/// index `k` the inner one; the stored matrices are transposed relative to that
/// (`backbone_weights[l][[k, j]] == W_l[j, k]`, `head_weights[[k, blk*H + j]] ==
/// W_blk[j, k]`), which is why this walk exists: `set_weights`, `get_weights` and
/// `get_weights_derivatives` all drive it, so the three orders cannot drift apart.
fn for_each_slot(
    input_size: usize,
    output_size: usize,
    backbone_units: usize,
    backbone_layers: usize,
    mut f: impl FnMut(SlotKind, usize, usize),
) {
    let b = backbone_units;
    for li in 0..backbone_layers {
        let fan_in = if li == 0 { input_size + output_size } else { b };
        for j in 0..b {
            for k in 0..fan_in {
                f(SlotKind::BackboneWeight(li), k, j);
            }
        }
        for j in 0..b {
            f(SlotKind::BackboneBias(li), 0, j);
        }
    }
    for blk in 0..HEAD_NB {
        for j in 0..output_size {
            for k in 0..b {
                f(SlotKind::HeadWeight, k, blk * output_size + j);
            }
        }
        for j in 0..output_size {
            f(SlotKind::HeadBias, 0, blk * output_size + j);
        }
    }
}

/// The CfC cell (spec S1). See the module doc for the math, the layout, the
/// constant-`Delta t` reduction and the house conventions it inherits.
#[derive(Clone)]
pub struct CfcLayer {
    input_size: usize,
    output_size: usize,
    backbone_units: usize,
    backbone_layers: usize,

    /// Per backbone layer, TRANSPOSED: `[0]` is `(in+H) x B`, the deeper ones `B x B`.
    backbone_weights: Vec<Array2<f64>>,
    /// Per backbone layer, `1 x B`.
    backbone_biases: Vec<Array2<f64>>,
    /// `B x 3H`, column blocks `[ff1|ff2|gate]`: `[[k, blk*H + j]] == W_blk[j, k]`.
    head_weights: Array2<f64>,
    /// `1 x 3H`, same blocks.
    head_biases: Array2<f64>,

    /// Forward cache, `T x (in+H)`: `z_t = [x_t | h_{t-1}]`. The backward's `dW_bb`
    /// outer product and the `dz` split both read it, so the two passes cannot
    /// disagree about what the first backbone layer was fed (which is why
    /// `feed_backward` ignores its `output` argument -- the sLSTM reads `h_{t-1}`
    /// back out of `output`, this cell keeps its own copy).
    z_cache: Array2<f64>,
    /// Forward cache, `L` entries of `T x B`: the backbone PRE-activations `a_l,t`
    /// ([`lecun_tanh_deriv`] takes them).
    backbone_pre: Vec<Array2<f64>>,
    /// Forward cache, `L` entries of `T x B`: the backbone POST-activations `bb_l,t`
    /// (the inputs the next layer's / the heads' `dW` outer products need).
    backbone_post: Vec<Array2<f64>>,
    /// Forward cache, `T x 3H`, blocks `[ff1|ff2|g]`: the head POST-activations,
    /// whose own derivatives (`1 - ff^2`, `g(1-g)`) are functions of themselves, so
    /// no head pre-activation is cached.
    heads: Array2<f64>,

    /// Deriv accumulators, shaped like their weight twins, plus the frame count
    /// (`col1` of the Nx2 harvest). Reset via `reset_weights_derivatives`,
    /// accumulated across `feed_backward` calls.
    backbone_weights_derivatives: Vec<Array2<f64>>,
    backbone_biases_derivatives: Vec<Array2<f64>>,
    head_weights_derivatives: Array2<f64>,
    head_biases_derivatives: Array2<f64>,
    nb_of_seq_fed_backward: i64,
}

impl CfcLayer {
    /// New layer with zeroed weights (callers set them via `set_weights`, exactly like
    /// `LstmLayer::new` with `weightsSetExternally = true`). Seeded init lives
    /// Python-side (`init_weights.py`, spec S1.5 / Task 2).
    ///
    /// `backbone_units` / `backbone_layers` are clamped to `>= 1` (the
    /// `MambaLayer::new` precedent): the config reader
    /// [`CfcParams`](super::super::blstm::CfcParams) rejects a `0` LOUDLY, and this
    /// clamp only stops a directly-constructed degenerate geometry from producing an
    /// empty backbone whose first index would panic.
    pub fn new(
        input_size: usize,
        output_size: usize,
        backbone_units: usize,
        backbone_layers: usize,
    ) -> CfcLayer {
        let b = backbone_units.max(1);
        let l = backbone_layers.max(1);
        let backbone_weights: Vec<Array2<f64>> = (0..l)
            .map(|li| {
                let fan_in = if li == 0 { input_size + output_size } else { b };
                Array2::zeros((fan_in, b))
            })
            .collect();
        let backbone_biases: Vec<Array2<f64>> = (0..l).map(|_| Array2::zeros((1, b))).collect();
        CfcLayer {
            input_size,
            output_size,
            backbone_units: b,
            backbone_layers: l,
            backbone_weights_derivatives: backbone_weights.clone(),
            backbone_biases_derivatives: backbone_biases.clone(),
            backbone_weights,
            backbone_biases,
            head_weights: Array2::zeros((b, HEAD_NB * output_size)),
            head_biases: Array2::zeros((1, HEAD_NB * output_size)),
            z_cache: Array2::zeros((0, 0)),
            backbone_pre: Vec::new(),
            backbone_post: Vec::new(),
            heads: Array2::zeros((0, 0)),
            head_weights_derivatives: Array2::zeros((b, HEAD_NB * output_size)),
            head_biases_derivatives: Array2::zeros((1, HEAD_NB * output_size)),
            nb_of_seq_fed_backward: 0,
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn backbone_units(&self) -> usize {
        self.backbone_units
    }

    pub fn backbone_layers(&self) -> usize {
        self.backbone_layers
    }

    /// `B*(in+H+1) + (L-1)*B*(B+1) + 3*H*(B+1)` (spec S1.2): the first backbone layer
    /// (`B x (in+H)` plus `B` biases), `L-1` deeper `B x B` layers plus their biases,
    /// and three `H x B` heads plus their biases.
    pub fn nb_of_weights(&self) -> usize {
        let (b, l, h) = (self.backbone_units, self.backbone_layers, self.output_size);
        b * (self.input_size + h + 1) + (l - 1) * b * (b + 1) + HEAD_NB * h * (b + 1)
    }

    /// Consume `flat[..nb_of_weights()]` in the S1.2 order and return the tail for the
    /// next layer in the stack (the chained head-eating contract).
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        let (needed, left) = flat.split_at(self.nb_of_weights());
        let (i, o, b, l) = (
            self.input_size,
            self.output_size,
            self.backbone_units,
            self.backbone_layers,
        );
        let (bw, bb, hw, hb) = (
            &mut self.backbone_weights,
            &mut self.backbone_biases,
            &mut self.head_weights,
            &mut self.head_biases,
        );
        let mut p = 0usize;
        for_each_slot(i, o, b, l, |kind, r, c| {
            match kind {
                SlotKind::BackboneWeight(li) => bw[li][[r, c]] = needed[p],
                SlotKind::BackboneBias(li) => bb[li][[r, c]] = needed[p],
                SlotKind::HeadWeight => hw[[r, c]] = needed[p],
                SlotKind::HeadBias => hb[[r, c]] = needed[p],
            }
            p += 1;
        });
        left
    }

    /// The mirror of [`Self::set_weights`]: append this layer's weights to `out` in
    /// the same S1.2 order.
    pub fn get_weights(&self, out: &mut Vec<f64>) {
        let (bw, bb, hw, hb) = (
            &self.backbone_weights,
            &self.backbone_biases,
            &self.head_weights,
            &self.head_biases,
        );
        for_each_slot(
            self.input_size,
            self.output_size,
            self.backbone_units,
            self.backbone_layers,
            |kind, r, c| {
                out.push(match kind {
                    SlotKind::BackboneWeight(li) => bw[li][[r, c]],
                    SlotKind::BackboneBias(li) => bb[li][[r, c]],
                    SlotKind::HeadWeight => hw[[r, c]],
                    SlotKind::HeadBias => hb[[r, c]],
                });
            },
        );
    }

    /// Nx2 `[summed deriv | frame count]` harvest in the SAME S1.2 order, so row `k`
    /// is the derivative of flat weight `k` (the seam's contract).
    pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        let count = self.nb_of_seq_fed_backward as f64;
        let (bw, bb, hw, hb) = (
            &self.backbone_weights_derivatives,
            &self.backbone_biases_derivatives,
            &self.head_weights_derivatives,
            &self.head_biases_derivatives,
        );
        for_each_slot(
            self.input_size,
            self.output_size,
            self.backbone_units,
            self.backbone_layers,
            |kind, r, c| {
                let v = match kind {
                    SlotKind::BackboneWeight(li) => bw[li][[r, c]],
                    SlotKind::BackboneBias(li) => bb[li][[r, c]],
                    SlotKind::HeadWeight => hw[[r, c]],
                    SlotKind::HeadBias => hb[[r, c]],
                };
                out.push([v, count]);
            },
        );
    }

    /// Zero the deriv accumulators and the frame count.
    pub fn reset_weights_derivatives(&mut self) {
        self.nb_of_seq_fed_backward = 0;
        for m in self.backbone_weights_derivatives.iter_mut() {
            m.fill(0.0);
        }
        for m in self.backbone_biases_derivatives.iter_mut() {
            m.fill(0.0);
        }
        self.head_weights_derivatives.fill(0.0);
        self.head_biases_derivatives.fill(0.0);
    }

    /// Scale the deriv accumulators (the harvested col0) in place; the frame count
    /// (col1) is deliberately untouched.
    pub fn ponderate_weights_derivatives(&mut self, factor: f64) {
        for m in self.backbone_weights_derivatives.iter_mut() {
            *m *= factor;
        }
        for m in self.backbone_biases_derivatives.iter_mut() {
            *m *= factor;
        }
        self.head_weights_derivatives *= factor;
        self.head_biases_derivatives *= factor;
    }

    /// `z_t = [x_t | h_{t-1}]` cache (`T x (in+H)`) from the last forward.
    pub fn z_cache(&self) -> &Array2<f64> {
        &self.z_cache
    }

    /// Backbone PRE-activation caches (`L` x `T x B`) from the last forward.
    pub fn backbone_pre(&self) -> &[Array2<f64>] {
        &self.backbone_pre
    }

    /// Backbone POST-activation caches (`L` x `T x B`) from the last forward.
    pub fn backbone_post(&self) -> &[Array2<f64>] {
        &self.backbone_post
    }

    /// Head post-activation cache (`T x 3H`, blocks `[ff1|ff2|g]`) from the last
    /// forward.
    pub fn heads(&self) -> &Array2<f64> {
        &self.heads
    }

    /// The input at the layer's own width: crop (`cols > in`) or zero-pad
    /// (`cols < in`). One representation used by BOTH passes, so the forward
    /// projection and the backward's `dW` accumulation can never disagree about which
    /// columns exist.
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

    /// Whole-sequence forward (spec S1.1). Fills `output` (`T x H`) and the four
    /// caches. `last_layer` is unused (signature parity with the trait).
    pub fn feed_forward(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        let _ = last_layer;
        let (i, h, b, l) = (
            self.input_size,
            self.output_size,
            self.backbone_units,
            self.backbone_layers,
        );
        let recon = self.reconcile_input(input);
        let t_len = recon.dim().0;

        let mut z_cache = Array2::<f64>::zeros((t_len, i + h));
        let mut pre: Vec<Array2<f64>> = (0..l).map(|_| Array2::zeros((t_len, b))).collect();
        let mut post: Vec<Array2<f64>> = (0..l).map(|_| Array2::zeros((t_len, b))).collect();
        let mut heads = Array2::<f64>::zeros((t_len, HEAD_NB * h));
        output.fill(0.0);

        for t in 0..t_len {
            // z_t = [x_t | h_{t-1}]; h_{-1} = 0, which the zeroed row already is.
            for k in 0..i {
                z_cache[[t, k]] = recon[[t, k]];
            }
            if t > 0 {
                for k in 0..h {
                    z_cache[[t, i + k]] = output[[t - 1, k]];
                }
            }

            // Backbone: L lecun_tanh layers, the first fed z_t and the rest chained.
            let mut cur = z_cache.slice(ndarray::s![t..t + 1, ..]).to_owned(); // 1 x (i+h)
            for li in 0..l {
                let a = matmul_seq(&cur, &self.backbone_weights[li]); // 1 x B
                for j in 0..b {
                    let v = a[[0, j]] + self.backbone_biases[li][[0, j]];
                    pre[li][[t, j]] = v;
                    post[li][[t, j]] = lecun_tanh(v);
                }
                cur = post[li].slice(ndarray::s![t..t + 1, ..]).to_owned(); // 1 x B
            }

            // The three heads share one product (combined column blocks).
            let u = matmul_seq(&cur, &self.head_weights); // 1 x 3H
            for j in 0..h {
                let (c1, c2, cg) = (BLOCK_FF1 * h + j, BLOCK_FF2 * h + j, BLOCK_GATE * h + j);
                let ff1 = (u[[0, c1]] + self.head_biases[[0, c1]]).tanh();
                let ff2 = (u[[0, c2]] + self.head_biases[[0, c2]]).tanh();
                let g = logistic_fn(u[[0, cg]] + self.head_biases[[0, cg]]);
                heads[[t, c1]] = ff1;
                heads[[t, c2]] = ff2;
                heads[[t, cg]] = g;
                output[[t, j]] = ff1 * (1.0 - g) + ff2 * g;
            }
        }

        self.z_cache = z_cache;
        self.backbone_pre = pre;
        self.backbone_post = post;
        self.heads = heads;
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

    /// Whole-sequence BPTT (spec S1.3, the adjoint equations in the module doc).
    /// Accumulates the weight-deriv blocks + the frame count, and returns the deltas
    /// propagated to the previous layer (`T x in`).
    ///
    /// `deltas` is `dL/dh`. `input`, `output` and `last_layer` are UNUSED: every
    /// quantity the adjoints need is in the forward caches ([`Self::z_cache`] already
    /// carries both the reconciled input and `h_{t-1}`), which is what makes it
    /// impossible for the two passes to disagree about the layer's own view of its
    /// input. The arguments stay for `Layer` signature parity.
    pub fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let _ = (input, output, last_layer);
        let (i, h, b, l) = (
            self.input_size,
            self.output_size,
            self.backbone_units,
            self.backbone_layers,
        );
        let lines_nb = deltas.dim().0;

        // Per-call accumulators, folded into the members at the end.
        let mut bw_d: Vec<Array2<f64>> = self
            .backbone_weights
            .iter()
            .map(|m| Array2::<f64>::zeros(m.dim()))
            .collect();
        let mut bb_d: Vec<Array2<f64>> = (0..l).map(|_| Array2::<f64>::zeros((1, b))).collect();
        let mut hw_d = Array2::<f64>::zeros((b, HEAD_NB * h));
        let mut hb_d = Array2::<f64>::zeros((1, HEAD_NB * h));

        // `dz_{t+1}[in..]`, the ONLY quantity carried backwards across timesteps.
        let mut carry = vec![0.0f64; h];
        let mut deltas_previous_layer = Array2::<f64>::zeros((lines_nb, i));

        // Head pre-activation deltas as a 3H x 1 column, so the `dbb_L` propagation is
        // one `matmul_seq` over the combined head matrix.
        let mut d_head = Array2::<f64>::zeros((HEAD_NB * h, 1));

        for row in (0..lines_nb).rev() {
            for j in 0..h {
                let (c1, c2, cg) = (BLOCK_FF1 * h + j, BLOCK_FF2 * h + j, BLOCK_GATE * h + j);
                let ff1 = self.heads[[row, c1]];
                let ff2 = self.heads[[row, c2]];
                let g = self.heads[[row, cg]];

                let dh = deltas[[row, j]] + carry[j];
                // h = ff1 (1-g) + ff2 g.
                let dff1 = dh * (1.0 - g);
                let dff2 = dh * g;
                let dg = dh * (ff2 - ff1);
                // tanh' = 1 - y^2 ; sigmoid' = y (1 - y), on the cached POST-values.
                d_head[[c1, 0]] = dff1 * (1.0 - ff1 * ff1);
                d_head[[c2, 0]] = dff2 * (1.0 - ff2 * ff2);
                d_head[[cg, 0]] = dg * g * (1.0 - g);
            }

            // Head weights/biases: outer product with the last backbone activation.
            for k in 0..b {
                let bb_last = self.backbone_post[l - 1][[row, k]];
                for c in 0..HEAD_NB * h {
                    hw_d[[k, c]] += bb_last * d_head[[c, 0]];
                }
            }
            for c in 0..HEAD_NB * h {
                hb_d[[0, c]] += d_head[[c, 0]];
            }

            // dbb_L = sum over the three heads of W_head^T du_head.
            let mut d_cur = matmul_seq(&self.head_weights, &d_head); // B x 1

            // Backbone, deepest layer first.
            for li in (0..l).rev() {
                let mut da = Array2::<f64>::zeros((b, 1));
                for j in 0..b {
                    da[[j, 0]] = d_cur[[j, 0]] * lecun_tanh_deriv(self.backbone_pre[li][[row, j]]);
                }
                // dW_l += da (x) in_l ; in_0 = z_t, in_l = bb_{l-1}.
                if li == 0 {
                    for k in 0..i + h {
                        let zk = self.z_cache[[row, k]];
                        for j in 0..b {
                            bw_d[0][[k, j]] += zk * da[[j, 0]];
                        }
                    }
                } else {
                    for k in 0..b {
                        let ik = self.backbone_post[li - 1][[row, k]];
                        for j in 0..b {
                            bw_d[li][[k, j]] += ik * da[[j, 0]];
                        }
                    }
                }
                for j in 0..b {
                    bb_d[li][[0, j]] += da[[j, 0]];
                }
                d_cur = matmul_seq(&self.backbone_weights[li], &da); // (in+H or B) x 1
            }

            // d_cur is now dz_t: the head is dx_t, the tail is the carry to t-1.
            for k in 0..i {
                deltas_previous_layer[[row, k]] = d_cur[[k, 0]];
            }
            for k in 0..h {
                carry[k] = d_cur[[i + k, 0]];
            }
        }

        if inv_sub_sampling_ratio > 1 {
            let r = inv_sub_sampling_ratio as f64;
            for m in bw_d.iter_mut() {
                m.mapv_inplace(|v| v * r);
            }
            for m in bb_d.iter_mut() {
                m.mapv_inplace(|v| v * r);
            }
            hw_d.mapv_inplace(|v| v * r);
            hb_d.mapv_inplace(|v| v * r);
        }

        for li in 0..l {
            self.backbone_weights_derivatives[li] += &bw_d[li];
            self.backbone_biases_derivatives[li] += &bb_d[li];
        }
        self.head_weights_derivatives += &hw_d;
        self.head_biases_derivatives += &hb_d;
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
/// `impl Layer for SlstmLayer` above it. It lives HERE rather than in
/// `nn/network.rs` so the new-cell code stays inside `nn/cells/` (spec S1.6);
/// `CellLayer`'s match arms call the INHERENT methods by UFCS, so the two paths
/// cannot diverge.
impl Layer for CfcLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        CfcLayer::feed_forward(self, input, output, last_layer);
    }
    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        CfcLayer::feed_forward_reverse(self, input, output, last_layer);
    }
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        CfcLayer::feed_backward(
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
        CfcLayer::feed_backward_reverse(
            self,
            input,
            output,
            deltas,
            inv_sub_sampling_ratio,
            last_layer,
        )
    }
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        CfcLayer::get_weights_derivatives(self, out);
    }
    fn reset_weights_derivatives(&mut self) {
        CfcLayer::reset_weights_derivatives(self);
    }
    fn ponderate_weights_derivatives(&mut self, factor: f64) {
        CfcLayer::ponderate_weights_derivatives(self, factor);
    }
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        CfcLayer::set_weights(self, flat)
    }
    fn get_weights(&self, out: &mut Vec<f64>) {
        CfcLayer::get_weights(self, out);
    }
    fn nb_of_weights(&self) -> usize {
        CfcLayer::nb_of_weights(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: usize = 3;
    const O: usize = 2;
    const B: usize = 4;
    const L: usize = 1;
    const T: usize = 7;

    /// Deterministic, non-degenerate weight fill (no RNG dependency).
    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.29 - 0.011 * (k as f64) + 0.006 * ((k % 7) as f64))
            .collect()
    }

    fn seq(rows: usize, cols: usize, seed: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            seed + 0.17 * (r as f64) - 0.23 * (c as f64) + 0.011 * ((r * cols + c) as f64)
        })
    }

    fn loaded(i: usize, o: usize, b: usize, l: usize) -> CfcLayer {
        let mut cell = CfcLayer::new(i, o, b, l);
        cell.set_weights(&weights(cell.nb_of_weights()));
        cell
    }

    fn forward(cell: &mut CfcLayer, input: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((input.nrows(), cell.output_size()));
        cell.feed_forward(input, &mut out, false);
        out
    }

    /// An INDEPENDENT transcription of the S1.2 block order: `(name, start offset)`
    /// pairs plus a terminal `END`, so a block's `[lo, hi)` range comes from this
    /// table and not from the walk under test.
    fn offsets(i: usize, o: usize, b: usize, l: usize) -> Vec<(String, usize)> {
        let mut v = Vec::new();
        let mut p = 0usize;
        for li in 0..l {
            let fan_in = if li == 0 { i + o } else { b };
            v.push((format!("W_bb{li}"), p));
            p += b * fan_in;
            v.push((format!("b_bb{li}"), p));
            p += b;
        }
        for name in ["W_1", "b_1", "W_2", "b_2", "W_t", "b_t"] {
            v.push((name.to_string(), p));
            p += if name.starts_with('W') { o * b } else { o };
        }
        v.push(("END".to_string(), p));
        v
    }

    fn block(blocks: &[(String, usize)], name: &str) -> (usize, usize) {
        let k = blocks.iter().position(|(n, _)| n == name).unwrap();
        (blocks[k].1, blocks[k + 1].1)
    }

    fn derivs(cell: &CfcLayer) -> Vec<f64> {
        let mut rows = Vec::new();
        cell.get_weights_derivatives(&mut rows);
        rows.iter().map(|r| r[0]).collect()
    }

    // --- Weight seam (spec S1.2) -------------------------------------------

    #[test]
    fn nb_of_weights_is_the_spec_formula() {
        for (i, o, b, l) in [
            (3usize, 2usize, 4usize, 1usize),
            (5, 4, 8, 2),
            (1, 1, 1, 1),
            (11, 24, 24, 3),
        ] {
            assert_eq!(
                CfcLayer::new(i, o, b, l).nb_of_weights(),
                b * (i + o + 1) + (l - 1) * b * (b + 1) + 3 * o * (b + 1),
                "in={i} out={o} B={b} L={l}"
            );
            // The independent offset table must agree with the formula.
            let blocks = offsets(i, o, b, l);
            assert_eq!(
                blocks.last().unwrap().1,
                CfcLayer::new(i, o, b, l).nb_of_weights()
            );
        }
    }

    /// The flat layout is `W_bb|b_bb`, per deeper layer `W|b`, then the three heads
    /// `W|b` -- each matrix ROW-major. Checked POSITIONALLY with a one-hot probe per
    /// slot (a round trip alone would pass on any self-consistent permutation).
    #[test]
    fn flat_layout_positions_are_the_spec_order() {
        let (i, o, b, l) = (3usize, 2usize, 4usize, 2usize);
        let mut cell = CfcLayer::new(i, o, b, l);
        let n = cell.nb_of_weights();
        let blocks = offsets(i, o, b, l);

        for li in 0..l {
            let fan_in = if li == 0 { i + o } else { b };
            let (wlo, _) = block(&blocks, &format!("W_bb{li}"));
            for j in 0..b {
                for k in 0..fan_in {
                    let mut flat = vec![0.0; n];
                    flat[wlo + j * fan_in + k] = 1.0;
                    cell.set_weights(&flat);
                    assert_eq!(
                        cell.backbone_weights[li][[k, j]],
                        1.0,
                        "backbone {li}: W[{j},{k}] misplaced"
                    );
                }
            }
            let (blo, _) = block(&blocks, &format!("b_bb{li}"));
            for j in 0..b {
                let mut flat = vec![0.0; n];
                flat[blo + j] = 1.0;
                cell.set_weights(&flat);
                assert_eq!(
                    cell.backbone_biases[li][[0, j]],
                    1.0,
                    "backbone {li}: b[{j}] misplaced"
                );
            }
        }

        for (blk, (wn, bn)) in [("W_1", "b_1"), ("W_2", "b_2"), ("W_t", "b_t")]
            .into_iter()
            .enumerate()
        {
            let (wlo, _) = block(&blocks, wn);
            for j in 0..o {
                for k in 0..b {
                    let mut flat = vec![0.0; n];
                    flat[wlo + j * b + k] = 1.0;
                    cell.set_weights(&flat);
                    assert_eq!(
                        cell.head_weights[[k, blk * o + j]],
                        1.0,
                        "head {wn}: W[{j},{k}] misplaced"
                    );
                }
            }
            let (blo, _) = block(&blocks, bn);
            for j in 0..o {
                let mut flat = vec![0.0; n];
                flat[blo + j] = 1.0;
                cell.set_weights(&flat);
                assert_eq!(
                    cell.head_biases[[0, blk * o + j]],
                    1.0,
                    "head {bn}: b[{j}] misplaced"
                );
            }
        }
    }

    #[test]
    fn weight_seam_round_trips_and_returns_the_tail() {
        let mut cell = CfcLayer::new(I, O, B, L);
        let nb = cell.nb_of_weights();
        let w = weights(nb + 5);
        let tail = cell.set_weights(&w);
        assert_eq!(tail.len(), 5);
        assert_eq!(tail, &w[nb..]);

        let mut back = Vec::new();
        cell.get_weights(&mut back);
        assert_eq!(back, w[..nb].to_vec());
    }

    // --- Activations (spec S1.1) --------------------------------------------

    /// The two LeCun-tanh constants and the derivative, pinned against literals so a
    /// typo in either cannot pass (`lt(0) = 0`, `lt(1) = 1.7159*tanh(2/3)`,
    /// `lt'(0) = 1.7159*2/3`), plus a central-difference cross-check of `lt'`.
    #[test]
    fn lecun_tanh_matches_its_definition() {
        assert_eq!(LECUN_SCALE, 1.7159);
        assert_eq!(LECUN_SLOPE, 2.0 / 3.0);
        assert_eq!(lecun_tanh(0.0), 0.0);
        assert_eq!(lecun_tanh(1.0), 1.7159 * (2.0f64 / 3.0).tanh());
        assert_eq!(lecun_tanh(-1.0), -lecun_tanh(1.0));
        assert_eq!(lecun_tanh_deriv(0.0), 1.7159 * (2.0 / 3.0));

        let eps = 1e-6;
        for x in [-2.3f64, -0.4, 0.0, 0.7, 3.1] {
            let fd = (lecun_tanh(x + eps) - lecun_tanh(x - eps)) / (2.0 * eps);
            // MEASURED max |fd - lt'| over these five points = 3.9e-11; pinned at 1e-9.
            assert!(
                (fd - lecun_tanh_deriv(x)).abs() < 1e-9,
                "lt'({x}) disagrees with the central difference"
            );
        }
    }

    // --- Forward (spec S1.1) ------------------------------------------------

    #[test]
    fn forward_shapes_and_caches_track_the_sequence_length() {
        for t in [1usize, 2, 7, 11] {
            for (b, l) in [(B, L), (8, 3)] {
                let mut cell = loaded(I, O, b, l);
                let out = forward(&mut cell, &seq(t, I, 0.4));
                assert_eq!(out.dim(), (t, O));
                assert_eq!(cell.z_cache().dim(), (t, I + O));
                assert_eq!(cell.backbone_pre().len(), l);
                assert_eq!(cell.backbone_post().len(), l);
                for li in 0..l {
                    assert_eq!(cell.backbone_pre()[li].dim(), (t, b));
                    assert_eq!(cell.backbone_post()[li].dim(), (t, b));
                }
                assert_eq!(cell.heads().dim(), (t, 3 * O));
                assert!(out.iter().all(|v| v.is_finite()), "t={t} B={b} L={l}");
            }
        }
    }

    /// The forward is a pure function of (weights, input): run twice, bit-identical
    /// (no state leaks across calls -- `h_{-1} = 0` per call, spec S1.1).
    #[test]
    fn forward_is_run_twice_bit_identical_and_stateless() {
        let mut cell = loaded(I, O, B, L);
        let input = seq(T, I, 0.4);
        let first = forward(&mut cell, &input);
        let (z1, p1, q1, h1) = (
            cell.z_cache().clone(),
            cell.backbone_pre().to_vec(),
            cell.backbone_post().to_vec(),
            cell.heads().clone(),
        );
        let second = forward(&mut cell, &input);
        assert_eq!(first, second);
        assert_eq!(&z1, cell.z_cache());
        assert_eq!(p1, cell.backbone_pre().to_vec());
        assert_eq!(q1, cell.backbone_post().to_vec());
        assert_eq!(&h1, cell.heads());

        // A fresh layer with the same weights agrees too (no ctor-order dependency).
        let mut fresh = loaded(I, O, B, L);
        assert_eq!(forward(&mut fresh, &input), first);
    }

    /// HAND-COMPUTED single step (1 input, 1 unit, 1 backbone unit, 1 layer): every
    /// quantity of S1.1 written out with literal constants. Asserts EQUALITY -- the
    /// port is expected to evaluate exactly these expressions, in this order.
    #[test]
    fn single_step_matches_the_hand_computed_value() {
        let (i, o, b, l) = (1usize, 1usize, 1usize, 1usize);
        let mut cell = CfcLayer::new(i, o, b, l);
        // Layout: W_bb (1x2) | b_bb (1) | W_1 (1x1) | b_1 | W_2 | b_2 | W_t | b_t.
        let flat = vec![
            0.4, -0.7, // W_bb: [x-column, h-column]
            0.15, // b_bb
            0.9, 0.05, // W_1, b_1
            -0.6, 0.2, // W_2, b_2
            0.3, -0.25, // W_t, b_t
        ];
        assert_eq!(flat.len(), cell.nb_of_weights());
        cell.set_weights(&flat);

        let x = 0.5;
        let input = Array2::from_shape_vec((1, 1), vec![x]).unwrap();
        let out = forward(&mut cell, &input);

        // h_{-1} = 0, so the h-column of W_bb contributes nothing at t = 0.
        let a = x * 0.4 + 0.0 * -0.7 + 0.15;
        let bb = 1.7159 * (2.0 / 3.0 * a).tanh();
        let ff1 = (bb * 0.9 + 0.05).tanh();
        let ff2 = (bb * -0.6 + 0.2).tanh();
        let g = 1.0 / (1.0 + (-(bb * 0.3 - 0.25)).exp());
        assert_eq!(cell.backbone_pre()[0][[0, 0]], a);
        assert_eq!(cell.backbone_post()[0][[0, 0]], bb);
        assert_eq!(cell.heads()[[0, 0]], ff1);
        assert_eq!(cell.heads()[[0, 1]], ff2);
        assert_eq!(cell.heads()[[0, 2]], g);
        assert_eq!(out[[0, 0]], ff1 * (1.0 - g) + ff2 * g);
        // The gate really interpolates: the output sits strictly between the heads.
        assert!(
            (out[[0, 0]] - ff1).abs() > 1e-3 && (out[[0, 0]] - ff2).abs() > 1e-3,
            "the interpolation is degenerate -- the check is vacuous"
        );
    }

    /// The gate is a genuine interpolation: at `g -> 0` the output IS `ff1`, at
    /// `g -> 1` it IS `ff2`. Driven by the gate BIAS alone (+-40, deep in the
    /// sigmoid's saturation), so the two heads are untouched between the two runs.
    #[test]
    fn the_gate_interpolates_between_the_two_heads() {
        let (i, o, b, l) = (2usize, 2usize, 3usize, 1usize);
        let blocks = offsets(i, o, b, l);
        let (gate_bias_lo, _) = block(&blocks, "b_t");
        let base = weights(CfcLayer::new(i, o, b, l).nb_of_weights());
        let input = seq(5, i, 0.3);

        for (bias, want_head) in [(-40.0f64, 0usize), (40.0, 1usize)] {
            let mut w = base.clone();
            for j in 0..o {
                w[gate_bias_lo + j] = bias;
            }
            let mut cell = CfcLayer::new(i, o, b, l);
            cell.set_weights(&w);
            let out = forward(&mut cell, &input);
            for r in 0..5 {
                for j in 0..o {
                    let head = cell.heads()[[r, want_head * o + j]];
                    assert!(
                        (out[[r, j]] - head).abs() < 1e-16,
                        "row {r} unit {j}: output {} is not the selected head {head}",
                        out[[r, j]]
                    );
                }
            }
        }
    }

    /// `feed_forward_reverse` is the forward on flipped rows (and leaves the caches
    /// reversed, the `LstmLayer` convention the reverse backward depends on).
    #[test]
    fn reverse_forward_is_the_flipped_forward() {
        let input = seq(T, I, 0.4);
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();

        let mut a = loaded(I, O, B, L);
        let mut out_rev = Array2::<f64>::zeros((T, O));
        a.feed_forward_reverse(&input, &mut out_rev, false);

        let mut b = loaded(I, O, B, L);
        let plain_on_flipped = forward(&mut b, &input_rev);

        assert_eq!(
            out_rev,
            plain_on_flipped.slice(ndarray::s![..;-1, ..]).to_owned()
        );
        // Caches stay in reversed-input order.
        assert_eq!(a.heads(), b.heads());
    }

    // --- Backward (spec S1.3) ----------------------------------------------

    /// Derivatives ACCUMULATE across `feed_backward` calls (col0 sums, col1 counts
    /// frames) and `reset_weights_derivatives` clears both.
    #[test]
    fn derivatives_accumulate_across_calls_and_reset_clears() {
        let mut cell = loaded(I, O, B, L);
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

    /// THE `T = 1` STRUCTURAL ZERO (module doc): the `h_{t-1}` column block of the
    /// FIRST backbone matrix multiplies `h_{-1} = 0` at a single timestep, so its
    /// `B * H` derivatives are EXACTLY zero -- asserted as `== 0.0`, not a tolerance,
    /// with a `T = 2` contrast in the same loop (the mamba
    /// `a_log_gradient_is_exactly_zero_at_t1` precedent).
    #[test]
    fn backbone_state_columns_are_gradient_dead_at_t1() {
        let (i, o, b, l) = (3usize, 2usize, 4usize, 1usize);
        let blocks = offsets(i, o, b, l);
        let (wlo, whi) = block(&blocks, "W_bb0");
        assert_eq!(whi - wlo, b * (i + o));
        // Row-major `W_bb[j, k]` -> flat `wlo + j*(i+o) + k`; the state columns are
        // `k in [i, i+o)`.
        let state_slots: Vec<usize> = (0..b)
            .flat_map(|j| (i..i + o).map(move |k| wlo + j * (i + o) + k))
            .collect();
        assert_eq!(state_slots.len(), b * o);

        for (t, want_zero) in [(1usize, true), (2, false)] {
            let mut cell = loaded(i, o, b, l);
            let input = seq(t, i, 0.4);
            let out = forward(&mut cell, &input);
            cell.reset_weights_derivatives();
            let _ = cell.feed_backward(&input, &out, &seq(t, o, -0.2), 1, false);
            let d = derivs(&cell);
            let worst = state_slots
                .iter()
                .map(|&s| d[s].abs())
                .fold(0.0f64, f64::max);
            if want_zero {
                for &s in state_slots.iter() {
                    assert_eq!(
                        d[s], 0.0,
                        "W_bb state column w[{s}] is not exactly zero at T=1"
                    );
                }
                // Contrast: the rest of the pack is very much alive at T=1 too.
                let overall = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                assert!(
                    overall > 1e-3,
                    "the whole gradient is ~0 at T=1 -- the zero claim is vacuous"
                );
            } else {
                assert!(
                    worst > 1e-6,
                    "the state columns are still dead at T=2 ({worst:e}) -- the T=1 claim \
                     is not about T at all"
                );
            }
        }
    }

    /// `ponderate_weights_derivatives` scales col0 only; `inv_sub_sampling_ratio`
    /// scales the accumulated blocks (the house conventions).
    #[test]
    fn ponderation_scales_col0_only_and_issr_scales_the_blocks() {
        let mut cell = loaded(I, O, B, L);
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

        let mut scaled = loaded(I, O, B, L);
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

        let mut a = loaded(I, O, B, L);
        let mut out_rev = Array2::<f64>::zeros((T, O));
        a.feed_forward_reverse(&input, &mut out_rev, false);
        let dpl_rev = a.feed_backward_reverse(&input, &out_rev, &deltas, 1, false);
        let mut da = Vec::new();
        a.get_weights_derivatives(&mut da);

        let mut b = loaded(I, O, B, L);
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
            let mut cell = loaded(I, O, B, L);
            let out = forward(&mut cell, input);
            assert_eq!(out.dim(), (T, O));
            let dpl = cell.feed_backward(input, &out, &seq(T, O, -0.2), 1, false);
            assert_eq!(dpl.dim(), (T, I));
        }

        // Cropping is exactly "ignore the extra columns": a wide input agrees with
        // its own left `I` columns fed directly.
        let mut a = loaded(I, O, B, L);
        let mut b = loaded(I, O, B, L);
        assert_eq!(
            forward(&mut a, &wide),
            forward(&mut b, &wide.slice(ndarray::s![.., ..I]).to_owned())
        );
    }
}
