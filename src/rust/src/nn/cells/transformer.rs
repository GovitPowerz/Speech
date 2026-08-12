//! Transformer: a PRE-NORM block over a CAUSAL SLIDING WINDOW of bounded width `W`,
//! with ALiBi relative positions and no learned position parameters at all.
//!
//! Port-only (Phase 11 Task 2) -- NO legacy source. Spec:
//! `docs/superpowers/specs/2026-08-12-phase-11-transformer-design.md`, S1.1 (forward),
//! S1.2 (flat layout), S1.3 (the softmax convention), S1.4 (the derived dead blocks),
//! S1.5 (backward), S2 (config vocabulary). Numeric divergence from anything legacy is
//! BY DESIGN and is documented here + in `RESULTS.md`, never `IMPROVEMENTS.md` (the
//! phase-7 rule).
//!
//! What landed: the whole-sequence f64 forward (S1.1), a HAND-DERIVED analytic backward
//! (S1.5, pinned against central differences by `tests/phase9_cell_grad.rs`), and the
//! `Layer`-conforming weight/derivative seam in the S1.2 flat order. Training-ready: the
//! layer plugs into `Network<CellLayer>` and therefore into every windowed driver, the
//! flat-weight seam, and SMORMS3, with no caller aware of the cell type.
//!
//! ## WHY WINDOWED, BY DEFINITION (the phase's design trap, answered up front)
//!
//! Whole-sequence f64 attention is `T^2` in both time and backward CACHE, which at
//! full-corpus lengths is gigabytes per layer. The cell is therefore DEFINED as
//! bounded-KV attention: frame `t` attends over `j in [max(0, t-W+1), t]`, the house
//! edge-truncating-convolution convention at the sequence start. Every cache below is
//! `T x O(W + d_ff)`-bounded, never `T^2`, which is also what makes the f32 streaming
//! twin's bounded KV ring bit-identical to this forward by construction (S5.1).
//!
//! ## Forward (S1.1), per timestep `t`, with `H = out`, `A = heads`, `d = H/A`
//!
//! ```text
//! x'_t   = W_a x_t + b_a                                   (width adapter, in -> H)
//! u_t    = RMSNorm_1(x'_t) = x'_t r1_t g_1                  r1 = (mean(x'^2)+eps)^-1/2
//! [q|k|v]_t = W_qkv u_t + b_qkv                             (ONE 3H x H matrix)
//! l^h_{t,j} = (q^h_t . k^h_j)/sqrt(d) + m_h (j - t)         j in [max(0,t-W+1), t]
//! p^h_{t,.} = softmax(l^h_{t,.})                            (S1.3, always max-subtract)
//! attn^h_t  = sum_j p^h_{t,j} v^h_j                         heads concatenated -> H
//! s_t    = x'_t + W_o attn_t + b_o                          (residual 1)
//! v~_t   = RMSNorm_2(s_t) = s_t r2_t g_2
//! h_t    = s_t + W_2 silu(W_1 v~_t + b_1) + b_2             (residual 2, FFN width d_ff)
//! ```
//!
//! `m_h = 2^(-8h/A)` for `h = 1..A` are the standard ALiBi geometric slopes, computed
//! ONCE at construction and carrying ZERO learned parameters. `j - t <= 0`, so the term
//! is a distance PENALTY. RMSNorm and `silu` are the in-tree Mamba transcriptions
//! ([`RMS_EPS`], [`silu`], [`silu_deriv`]), imported rather than re-derived so the two
//! cells cannot drift apart.
//!
//! NAME COLLISION, called out because the spec text carries it: `v` is the attention
//! VALUE vector, while the spec's `v_t` is the RMSNorm_2 OUTPUT. This module calls the
//! latter `v_norm` throughout and never abbreviates it.
//!
//! Output is `h_t` (width `H`); there is no output softmax in the cell (the downstream
//! output MLP owns classification). The cell is natively CAUSAL and stateless across
//! calls, so under `Direction::Bidirectional` the wrapper gets its reverse half by
//! running the SAME back-looking cell on the time-reversed sequence -- no cell-side
//! branch, exactly like every other cell here.
//!
//! ## Flat layout (S1.2), walked once by [`for_each_slot`]
//!
//! `[W_a | b_a | g_1 | W_qkv | b_qkv | W_o | b_o | g_2 | W_1 | b_1 | W_2 | b_2]`, each
//! matrix ROW-major (output unit outer, source index inner), and
//! `nb_of_weights = H(in+1) + H + 3H(H+1) + H(H+1) + H + d_ff(H+1) + H(d_ff+1)`.
//! `W_qkv`'s `3H` rows are the blocks `[q | k | v]` in that order, which is what makes
//! "the k rows" below an addressable range rather than a figure of speech.
//!
//! ## The softmax convention (S1.3): ALWAYS max-subtract
//!
//! A DECLARED divergence from the house F8 conditional guard (`nn::layers`' output
//! softmax subtracts the row max only above `EXP_OVERFLOW_GUARD`). Here the subtraction
//! is unconditional, for two reasons: there are no legacy bytes to match (this cell is
//! port-only), and the f32 twin needs the headroom -- `ln(f32::MAX) ~ 88.72` against
//! f64's `~709.78`, and untrained attention logits spike. ONE convention in both trees
//! makes the f32 twin a PRECISION-only divergence rather than a structural one. Op order
//! is pinned: ascending loops, max-then-exp-then-normalize.
//!
//! The subtraction is also gradient-FREE, and that is exact rather than an
//! approximation (the sLSTM `m`-invariance argument, in a cleaner form): `softmax(l - c)
//! == softmax(l)` for ANY constant `c`, so the composed map IS softmax and its Jacobian
//! is the standard one. The `max()` kink contributes nothing because the function is
//! constant along the direction the max moves in -- there is no subgradient to choose.
//!
//! ## Backward (S1.5), hand-derived; `da` denotes the delta of `a`
//!
//! ```text
//! FFN:    da1_t[r] = (sum_m W_2[m,r] dh_t[m]) silu'(a1_t[r])
//!         dW_2[m,r] += silu(a1_t[r]) dh_t[m]     db_2[m] += dh_t[m]
//!         dW_1[r,m] += v_norm_t[m] da1_t[r]      db_1[r] += da1_t[r]
//!         dv_norm_t[m] = sum_r W_1[r,m] da1_t[r]
//! norm2:  r = r2_t,  q = sum_j dv_norm_t[j] s_t[j] g_2[j]
//!         dg_2[m] += dv_norm_t[m] s_t[m] r
//!         ds_t[m]  = dh_t[m] + dv_norm_t[m] g_2[m] r - (r^3/H) s_t[m] q
//! res1:   dx'_t[m] += ds_t[m]   db_o[m] += ds_t[m]   dW_o[m,k] += attn_t[k] ds_t[m]
//!         dattn_t[k] = sum_m W_o[m,k] ds_t[m]
//! attn:   dp_{t,j} = sum_e dattn^h_t[e] v^h_j[e]
//!         dv^h_j[e] += p_{t,j} dattn^h_t[e]                    (REVERSE SCATTER)
//!         dl_{t,j}  = p_{t,j} (dp_{t,j} - sum_j' p_{t,j'} dp_{t,j'})
//!         dq^h_t[e] += (dl_{t,j}/sqrt(d)) k^h_j[e]
//!         dk^h_j[e] += (dl_{t,j}/sqrt(d)) q^h_t[e]             (REVERSE SCATTER)
//! qkv:    dW_qkv[c,m] += u_t[m] dqkv_t[c]   db_qkv[c] += dqkv_t[c]
//!         du_t[m] = sum_c W_qkv[c,m] dqkv_t[c]
//! norm1:  r = r1_t,  q = sum_j du_t[j] x'_t[j] g_1[j]
//!         dg_1[m] += du_t[m] x'_t[m] r
//!         dx'_t[m] += du_t[m] g_1[m] r - (r^3/H) x'_t[m] q
//! adapt:  dW_a[m,k] += x_t[k] dx'_t[m]   db_a[m] += dx'_t[m]
//!         dx_t[k] = sum_m W_a[m,k] dx'_t[m]
//! ```
//!
//! ALiBi contributes NO parameter gradient (the slopes are constants); it enters only
//! the additive logit path, whose derivative w.r.t. the logit is the identity.
//!
//! THE ONE STRUCTURAL ORDERING CONSTRAINT, the Mamba conv-fold shape: `k_j` and `v_j`
//! receive contributions from EVERY `t >= j` with `t - j < W`, so neither is complete
//! until the whole sequence has been walked. The pass is therefore SPLIT -- a
//! reverse-time loop down to `dq`/`dk`/`dv` (scattering as it goes), then a forward-time
//! loop for the qkv projection, RMSNorm_1 and the adapter. `dq_t` needs no scatter (a
//! query is used by its own row alone) and is complete inside the first loop.
//!
//! `dx'` accumulates TWO contributions -- the residual-1 pass-through in the reverse
//! loop and the RMSNorm_1 backward in the forward loop -- which is why it is a whole
//! `T x H` buffer rather than a per-step vector.
//!
//! ## Structurally gradient-dead weights (S1.4)
//!
//! - **`b_k` (the k rows of `b_qkv`) is dead ALWAYS.** Shifting `b_k` by `delta` shifts
//!   every `k_j` by the same `delta`, so every logit in a window ROW moves by the SAME
//!   `(q_t . delta)/sqrt(d)` -- annihilated by softmax shift-invariance. This cell's
//!   `b_i` analogue (the sLSTM precedent): seeded 0, never moves, non-identifiable.
//!   THE PIN IS TWO-REGIME, and the distinction is real rather than pedantic:
//!   * at `T = 1` it is EXACTLY `0.0`, because a one-element window makes `l - max = 0`,
//!     `exp(0) = 1`, `p = 1/1 = 1` and hence `dl = p (dp - p dp) = 0.0` bit-exactly;
//!   * at `T >= 2` it is zero in R but a CANCELLATION in f64 (`sum_j dl_{t,j} = 0`
//!     analytically; MEASURED `2.168e-17` at `T = 3` and `8.007e-17` at `T = 7`, against
//!     a whole-pack gradient maximum of ~5e1), so it is pinned on a MEASURED absolute
//!     floor exactly as the sLSTM `b_i` is. This is NOT the Mamba/CfC
//!     multiplies-an-exact-zero shape, and pretending otherwise would mean forcing the
//!     block to zero -- papering over, which this repo does not do.
//! - **The q and k ROWS of `W_qkv`, plus `b_q`, are dead at `T = 1`** (the Mamba
//!   `A_log`-at-`T=1` analogue): every window has exactly one element and a one-element
//!   softmax is constantly 1, so `dq = dk = 0.0` bit-exactly. Pinned with a `T = 2`
//!   contrast. It is a `T = 1` artefact, not a degeneracy.
//!
//! NOTHING ELSE is dead. `b_q` is LIVE for `T >= 2` (`b_q . k_j` varies with `j`), the
//! RMSNorm gains are live, and the `v`/`W_o`/FFN blocks are live at every length.
//!
//! ## Conventions inherited from the house (`nn::layers::LstmLayer`, `cells::*`)
//!
//! - Every projection matrix is stored TRANSPOSED (`w[[k, j]] == W[j, k]`), so each
//!   product is a plain [`matmul_seq`] call -- the ascending accumulation contract,
//!   uniformly. `q`, `k` and `v` share ONE combined `H x 3H` matrix (the sLSTM/CfC
//!   combined-block precedent), so a timestep runs four projections and no more.
//! - The S1.2 flat order is a (de)serialization concern only, walked ONCE by
//!   [`for_each_slot`], so `set_weights`, `get_weights` and `get_weights_derivatives`
//!   cannot drift apart -- and `nb_of_weights` COUNTS that same walk, so a layout edit
//!   cannot desynchronize the length from the order either.
//! - `get_weights_derivatives` returns Nx2 `[summed deriv | frame count]` in the flat
//!   order, `ponderate_weights_derivatives` scales col0 only, `set_weights` returns the
//!   unconsumed tail, `inv_sub_sampling_ratio > 1` scales the deriv blocks.
//! - `feed_forward_reverse` / `feed_backward_reverse` flip rows around the forward-order
//!   body and leave the caches in REVERSED order (consumed as-is by the reverse
//!   backward). The cell is CAUSAL, so the reverse direction is a genuine anti-causal
//!   pass over the flipped sequence.
//! - Input WIDTH TOLERANCE: `cols > in` crops to the left `in` columns, `cols < in`
//!   zero-pads. The propagated deltas are always `T x in`.
//! - All summations are ASCENDING loops (or [`matmul_seq`], the same contract).
//!
//! ## Retention (phase-11 S7, Task 8): GATED, day one
//!
//! The attention cache is the largest backward cache of any cell in the tree --
//! `attn_weights` (`T x A*wcap`) is the one field with no fixed bound on its own column
//! count (`A*W` grows with the config, unlike the `H`/`3H`/`d_ff`-bounded fields), and at
//! the default geometry (`A = 4`, `W = 64`) it is already wider on its own (256 columns)
//! than every other cached field combined. So `CellLayer::set_retain_cache`'s transformer
//! arm gates it from day one, on the phase-10 T9 mechanism (`MambaLayer`), rather than as
//! a discovered-later follow-on.
//!
//! The mechanism has Mamba's two parts, CLEAR and SHRINK. CLEAR: every other cache field
//! is computed in full regardless of `retain_cache` -- each is produced by ONE
//! whole-sequence [`matmul_seq`] call (there is no cheaper way to compute it without
//! restructuring the projection itself, out of scope here) -- and simply left OUT of
//! `self.cache` when not retaining: `self.cache = TransformerCache::default()` drops the
//! freshly-computed locals at the end of the call. SHRINK: `attn_weights` gets the
//! Mamba-`h` treatment, and an even simpler version of it -- it is written at row `t` and
//! read back ONLY within that SAME row's attention-output accumulation (no cross-row read
//! at all during forward; only the backward's softmax-Jacobian pass reads a DIFFERENT
//! row's weights). So under `retain_cache == false` it shrinks from `T x A*wcap` to `1 x
//! A*wcap`, and every access uses `ta = if retain { t } else { 0 }` in place of a bare
//! `t` -- ONE slot, no `t-1` alternation needed (unlike Mamba's rolling PAIR), because
//! forward never looks backward through this buffer at all. The retaining path is
//! untouched (`ta == t` always), so [`tests::forward_is_bit_identical_without_the_cache`]
//! is the pin that says the two indexings agree.
//!
//! A backward after a non-retaining forward PANICS with a named message
//! ([`Self::feed_backward`]) rather than folding an empty or stale buffer.
//! [`super::super::network::Network`]'s own `retain_layers_output` check
//! (`drive_backward`) is a SECOND, cell-agnostic choke point on the same call path --
//! unchanged by this task, and already sufficient on its own via the `Layer` trait funnel
//! (it does not inspect which `CellLayer` variant it is driving). `true` (retain) is the
//! default and the only behaviour of anything built outside `BlstmNetwork::from_config`,
//! so a direct construction is unaffected unless it opts in.

use ndarray::Array2;

use super::super::layers::matmul_seq;
use super::super::network::Layer;
use super::mamba::{RMS_EPS, silu, silu_deriv};
use crate::nn::blstm::TransformerParams;

/// The ALiBi slope base exponent: `m_h = 2^(-8h/A)` for `h = 1..A` (spec S1.1), the
/// standard geometric ladder from the ALiBi paper. Evaluated through `exp2` rather than
/// `powf(2.0, .)` so the common `A | 8` cases land on exact powers of two.
const ALIBI_EXPONENT: f64 = -8.0;

/// Column-block order inside the combined `W_qkv`/`b_qkv` and the S1.2 flat layout:
/// `[q | k | v]`. These index BLOCKS of `H` columns, not single columns.
const QKV_NB: usize = 3;
const BLOCK_Q: usize = 0;
const BLOCK_K: usize = 1;
const BLOCK_V: usize = 2;

/// Which weight block a flat-layout slot addresses (see [`for_each_slot`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotKind {
    /// `W_a` (width adapter), stored transposed `in x H`.
    AdapterW,
    /// `b_a`, `1 x H`.
    AdapterB,
    /// `g_1` (RMSNorm_1 gain), `1 x H`.
    Gain1,
    /// `W_qkv`, stored transposed `H x 3H`, column blocks `[q|k|v]`.
    Qkv,
    /// `b_qkv`, `1 x 3H`, same blocks.
    QkvB,
    /// `W_o` (attention out-projection), stored transposed `H x H`.
    OutW,
    /// `b_o`, `1 x H`.
    OutB,
    /// `g_2` (RMSNorm_2 gain), `1 x H`.
    Gain2,
    /// `W_1` (FFN up-projection), stored transposed `H x d_ff`.
    Ff1W,
    /// `b_1`, `1 x d_ff`.
    Ff1B,
    /// `W_2` (FFN down-projection), stored transposed `d_ff x H`.
    Ff2W,
    /// `b_2`, `1 x H`.
    Ff2B,
}

/// The twelve weight blocks in one struct, so the DERIVATIVE accumulator is literally
/// the same type (the [`super::mamba::MambaLayer`] precedent): the shapes cannot drift
/// apart, and `fill`/`scale`/`add_assign` are one loop each instead of twelve
/// statements.
#[derive(Clone)]
struct TransformerWeights {
    adapter_w: Array2<f64>,
    adapter_b: Array2<f64>,
    gain1: Array2<f64>,
    qkv: Array2<f64>,
    qkv_b: Array2<f64>,
    out_w: Array2<f64>,
    out_b: Array2<f64>,
    gain2: Array2<f64>,
    ff1_w: Array2<f64>,
    ff1_b: Array2<f64>,
    ff2_w: Array2<f64>,
    ff2_b: Array2<f64>,
}

impl TransformerWeights {
    fn zeros(input_size: usize, h: usize, d_ff: usize) -> TransformerWeights {
        TransformerWeights {
            adapter_w: Array2::zeros((input_size, h)),
            adapter_b: Array2::zeros((1, h)),
            gain1: Array2::zeros((1, h)),
            qkv: Array2::zeros((h, QKV_NB * h)),
            qkv_b: Array2::zeros((1, QKV_NB * h)),
            out_w: Array2::zeros((h, h)),
            out_b: Array2::zeros((1, h)),
            gain2: Array2::zeros((1, h)),
            ff1_w: Array2::zeros((h, d_ff)),
            ff1_b: Array2::zeros((1, d_ff)),
            ff2_w: Array2::zeros((d_ff, h)),
            ff2_b: Array2::zeros((1, h)),
        }
    }

    fn blocks(&self) -> [&Array2<f64>; 12] {
        [
            &self.adapter_w,
            &self.adapter_b,
            &self.gain1,
            &self.qkv,
            &self.qkv_b,
            &self.out_w,
            &self.out_b,
            &self.gain2,
            &self.ff1_w,
            &self.ff1_b,
            &self.ff2_w,
            &self.ff2_b,
        ]
    }

    fn blocks_mut(&mut self) -> [&mut Array2<f64>; 12] {
        [
            &mut self.adapter_w,
            &mut self.adapter_b,
            &mut self.gain1,
            &mut self.qkv,
            &mut self.qkv_b,
            &mut self.out_w,
            &mut self.out_b,
            &mut self.gain2,
            &mut self.ff1_w,
            &mut self.ff1_b,
            &mut self.ff2_w,
            &mut self.ff2_b,
        ]
    }

    fn slot(&self, kind: SlotKind) -> &Array2<f64> {
        match kind {
            SlotKind::AdapterW => &self.adapter_w,
            SlotKind::AdapterB => &self.adapter_b,
            SlotKind::Gain1 => &self.gain1,
            SlotKind::Qkv => &self.qkv,
            SlotKind::QkvB => &self.qkv_b,
            SlotKind::OutW => &self.out_w,
            SlotKind::OutB => &self.out_b,
            SlotKind::Gain2 => &self.gain2,
            SlotKind::Ff1W => &self.ff1_w,
            SlotKind::Ff1B => &self.ff1_b,
            SlotKind::Ff2W => &self.ff2_w,
            SlotKind::Ff2B => &self.ff2_b,
        }
    }

    fn slot_mut(&mut self, kind: SlotKind) -> &mut Array2<f64> {
        match kind {
            SlotKind::AdapterW => &mut self.adapter_w,
            SlotKind::AdapterB => &mut self.adapter_b,
            SlotKind::Gain1 => &mut self.gain1,
            SlotKind::Qkv => &mut self.qkv,
            SlotKind::QkvB => &mut self.qkv_b,
            SlotKind::OutW => &mut self.out_w,
            SlotKind::OutB => &mut self.out_b,
            SlotKind::Gain2 => &mut self.gain2,
            SlotKind::Ff1W => &mut self.ff1_w,
            SlotKind::Ff1B => &mut self.ff1_b,
            SlotKind::Ff2W => &mut self.ff2_w,
            SlotKind::Ff2B => &mut self.ff2_b,
        }
    }

    fn fill(&mut self, v: f64) {
        for b in self.blocks_mut() {
            b.fill(v);
        }
    }

    fn scale(&mut self, factor: f64) {
        for b in self.blocks_mut() {
            *b *= factor;
        }
    }

    fn add_assign(&mut self, other: &TransformerWeights) {
        for (dst, src) in self.blocks_mut().into_iter().zip(other.blocks()) {
            *dst += src;
        }
    }
}

/// Walk the S1.2 flat layout EXACTLY ONCE, in order, calling `f(kind, row, col)` with
/// the STORED-matrix indices of each successive flat element:
///
/// ```text
/// W_a (H x in) row-major | b_a (H) | g_1 (H)
/// W_qkv (3H x H) row-major | b_qkv (3H)          rows blocked [q | k | v]
/// W_o (H x H) row-major | b_o (H) | g_2 (H)
/// W_1 (d_ff x H) row-major | b_1 (d_ff) | W_2 (H x d_ff) row-major | b_2 (H)
/// ```
///
/// Row-major per matrix means the OUTPUT unit is the outer index and the source index
/// the inner one; the stored matrices are TRANSPOSED relative to that, which is exactly
/// why this walk exists rather than three hand-written index computations.
fn for_each_slot(
    input_size: usize,
    h: usize,
    d_ff: usize,
    mut f: impl FnMut(SlotKind, usize, usize),
) {
    // W_a (H x in) row-major, then b_a.
    for j in 0..h {
        for k in 0..input_size {
            f(SlotKind::AdapterW, k, j);
        }
    }
    for j in 0..h {
        f(SlotKind::AdapterB, 0, j);
    }
    // g_1
    for m in 0..h {
        f(SlotKind::Gain1, 0, m);
    }
    // W_qkv (3H x H) row-major, then b_qkv.
    for r in 0..QKV_NB * h {
        for m in 0..h {
            f(SlotKind::Qkv, m, r);
        }
    }
    for r in 0..QKV_NB * h {
        f(SlotKind::QkvB, 0, r);
    }
    // W_o (H x H) row-major, then b_o.
    for j in 0..h {
        for k in 0..h {
            f(SlotKind::OutW, k, j);
        }
    }
    for j in 0..h {
        f(SlotKind::OutB, 0, j);
    }
    // g_2
    for m in 0..h {
        f(SlotKind::Gain2, 0, m);
    }
    // W_1 (d_ff x H) row-major, then b_1.
    for r in 0..d_ff {
        for m in 0..h {
            f(SlotKind::Ff1W, m, r);
        }
    }
    for r in 0..d_ff {
        f(SlotKind::Ff1B, 0, r);
    }
    // W_2 (H x d_ff) row-major, then b_2.
    for j in 0..h {
        for r in 0..d_ff {
            f(SlotKind::Ff2W, r, j);
        }
    }
    for j in 0..h {
        f(SlotKind::Ff2B, 0, j);
    }
}

/// Everything the backward reads back from the forward (spec S1.5's cache note). Every
/// field is `T x O(W + d_ff)`-bounded; nothing here is `T^2`.
///
/// PRE- vs POST-activation is the classic trap, so each field says which it is:
/// `ffn_pre` is PRE-SiLU (the adjoint calls [`silu_deriv`], which takes the
/// pre-activation, and re-derives `silu` itself for the `dW_2` outer product), while
/// `attn_weights` are POST-softmax.
#[derive(Clone, Default)]
struct TransformerCache {
    /// The RECONCILED input (`T x in`) -- the `dW_a` outer-product factor. Cached rather
    /// than recomputed from the backward's `input` argument (the CfC contract: the
    /// backward reads only what the forward stored, so the two passes cannot disagree
    /// about the layer's own view of its input).
    x_in: Array2<f64>,
    /// `x'` -- the ADAPTED input (`T x H`). Read by the RMSNorm_1 backward and the
    /// `dg_1` accumulation.
    x_adapted: Array2<f64>,
    /// `1 / sqrt(mean(x'^2) + eps)` per timestep (`T`).
    rms1_inv: Vec<f64>,
    /// `u` (`T x H`), the RMSNorm_1 output -- the `dW_qkv` outer-product factor.
    u: Array2<f64>,
    /// `[q | k | v]` (`T x 3H`), post-bias. The backward reads `k_j` for `dq`, `q_t` for
    /// `dk` and `v_j` for `dp`, so all three blocks are live in the adjoint.
    qkv: Array2<f64>,
    /// Per-row attention weights (`T x A*wcap`), head-major: row `t`, head `hh`, window
    /// offset `jj` lives at `[[t, hh*wcap + jj]]` with `jj = j - max(0, t-W+1)`. Slots
    /// past a short row's window length stay zero.
    attn_weights: Array2<f64>,
    /// The window capacity actually allocated, `min(W, T)` (clamped to `>= 1`): the
    /// stride of [`Self::attn_weights`].
    wcap: usize,
    /// `attn` (`T x H`), heads concatenated -- the `dW_o` outer-product factor.
    attn_out: Array2<f64>,
    /// `s` (`T x H`), the residual-1 output. Read by the RMSNorm_2 backward.
    s: Array2<f64>,
    /// `1 / sqrt(mean(s^2) + eps)` per timestep (`T`).
    rms2_inv: Vec<f64>,
    /// `v_norm` (`T x H`), the RMSNorm_2 output -- the `dW_1` outer-product factor.
    /// (The spec calls this `v_t`; see the module doc's name-collision note.)
    v_norm: Array2<f64>,
    /// `a1` (`T x d_ff`), the FFN hidden PRE-activation.
    ffn_pre: Array2<f64>,
}

/// The windowed causal attention cell (spec S1). See the module doc for the math, the
/// layout, the softmax convention, the dead blocks and the house conventions.
#[derive(Clone)]
pub struct TransformerLayer {
    input_size: usize,
    output_size: usize,
    window: usize,
    heads: usize,
    d_ff: usize,
    /// `d = H / heads`, the per-head width (an exact division -- see [`Self::new`]).
    head_dim: usize,
    /// `sqrt(d)`, the logit scale DENOMINATOR. Stored as the square root and DIVIDED by
    /// (not stored as its reciprocal and multiplied), so the arithmetic is literally the
    /// spec's `(q . k)/sqrt(d)` and the f32 twin has one unambiguous op order to mirror.
    sqrt_d: f64,
    /// ALiBi slopes `m_h = 2^(-8h/A)`, `h = 1..A`, computed once at construction. NOT
    /// weights: they never enter the flat pack and take no gradient.
    slopes: Vec<f64>,

    weights: TransformerWeights,
    derivatives: TransformerWeights,
    cache: TransformerCache,
    /// Phase 11 spec S7 (Task 8, the phase-10 T9 mechanism applied to this cell). `true`
    /// (the default, and every pre-Task-8 behaviour) fills [`TransformerCache`] at the end
    /// of every forward; `false` drops it (see the module doc's "Retention" section for
    /// the clear+shrink mechanism). Flipped at CONSTRUCTION only
    /// (`BlstmNetwork::set_inference_only`, driven by `BackPropagationActivated`), never
    /// per call, so a net either retains for its whole life or never does.
    ///
    /// Measured on a long-T bidirectional SAD net (2 stacked layers, `H = 24`, default
    /// geometry, a 60 s stereo excerpt forced through the PLAIN whole-sequence driver via
    /// `BLSTM_window 0` so the cache scales with the full sequence): `BackPropagationActivated
    /// true` (retaining) peaks at 93.422 MB vs `false` (this flag off) at 61.359 MB,
    /// `speech bench --path=exact`, one run each -- 1.52x less peak RSS. See `RESULTS.md`'s
    /// Task 8 row for the full recipe.
    retain_cache: bool,
    nb_of_seq_fed_backward: i64,
}

impl TransformerLayer {
    /// New layer with zeroed weights (callers set them via `set_weights`, exactly like
    /// `LstmLayer::new` with `weightsSetExternally = true`). Seeded init lives
    /// Python-side (`init_weights.py`, spec S8).
    ///
    /// PANICS when `output_size % heads != 0`, NAMING BOTH NUMBERS: the head split is a
    /// structural property of the layout (`d = H/A` sizes every per-head slice), and a
    /// silent truncation would build a net whose attention quietly ignored the tail
    /// units. The config path catches this EARLIER and as a typed error --
    /// `BlstmConfig::from_legacy` validates every recurrent-layer width against
    /// `Transformer_Heads` (spec S2) -- so this panic is the last-resort guard for a
    /// direct construction, not the primary message a user sees.
    ///
    /// `window`/`heads`/`d_ff` are clamped to `>= 1` (the `MambaLayer::new` /
    /// `CfcLayer::new` precedent): [`TransformerParams::from_legacy`] rejects a `0`
    /// LOUDLY, and the clamp only stops a directly-constructed degenerate geometry from
    /// producing an empty window or a division by zero.
    pub fn new(input_size: usize, output_size: usize, params: &TransformerParams) -> Self {
        let heads = params.heads.max(1);
        let window = params.window.max(1);
        let d_ff = params.d_ff.max(1);
        assert!(
            output_size.is_multiple_of(heads),
            "transformer: the cell width ({output_size}) must be divisible by the head count \
             ({heads}) -- a non-dividing head count is a configuration error, not something to \
             truncate"
        );
        let head_dim = output_size / heads;
        let slopes: Vec<f64> = (1..=heads)
            .map(|h| (ALIBI_EXPONENT * h as f64 / heads as f64).exp2())
            .collect();
        let weights = TransformerWeights::zeros(input_size, output_size, d_ff);
        TransformerLayer {
            input_size,
            output_size,
            window,
            heads,
            d_ff,
            head_dim,
            sqrt_d: (head_dim as f64).sqrt(),
            slopes,
            derivatives: weights.clone(),
            weights,
            cache: TransformerCache::default(),
            retain_cache: true,
            nb_of_seq_fed_backward: 0,
        }
    }

    /// Phase 11 spec S7 (Task 8): turn the backward cache retention off (inference-only)
    /// or back on. See the module doc's "Retention" section for the clear+shrink
    /// mechanism this flips.
    ///
    /// Turning it OFF also drops whatever the last forward left behind, so a subsequent
    /// [`Self::feed_backward`] hits the loud bail rather than silently folding a stale
    /// cache from an earlier sequence into the derivative accumulators.
    pub fn set_retain_cache(&mut self, retain: bool) {
        self.retain_cache = retain;
        if !retain {
            self.cache = TransformerCache::default();
        }
    }

    /// Whether this layer retains its backward cache (spec S7; `true` by default).
    pub fn retains_cache(&self) -> bool {
        self.retain_cache
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn d_ff(&self) -> usize {
        self.d_ff
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The ALiBi slopes `m_h = 2^(-8h/A)`, `h = 1..A` (spec S1.1). Exposed for the unit
    /// pin and for the f32 twin to mirror rather than re-derive.
    pub fn slopes(&self) -> &[f64] {
        &self.slopes
    }

    /// `H(in+1) + H + 3H(H+1) + H(H+1) + H + d_ff(H+1) + H(d_ff+1)` (spec S1.2), COUNTED
    /// off [`for_each_slot`] rather than re-derived, so the length and the order have one
    /// source. (`nb_of_weights_is_the_spec_formula` pins it against the closed form.)
    pub fn nb_of_weights(&self) -> usize {
        let mut n = 0usize;
        for_each_slot(self.input_size, self.output_size, self.d_ff, |_, _, _| {
            n += 1
        });
        n
    }

    /// Consume `flat[..nb_of_weights()]` in the S1.2 order and return the tail for the
    /// next layer in the stack (the chained head-eating contract).
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        let (needed, left) = flat.split_at(self.nb_of_weights());
        let w = &mut self.weights;
        let mut p = 0usize;
        for_each_slot(
            self.input_size,
            self.output_size,
            self.d_ff,
            |kind, r, c| {
                w.slot_mut(kind)[[r, c]] = needed[p];
                p += 1;
            },
        );
        left
    }

    /// The mirror of [`Self::set_weights`]: append this layer's weights to `out` in the
    /// same S1.2 order.
    pub fn get_weights(&self, out: &mut Vec<f64>) {
        let w = &self.weights;
        for_each_slot(
            self.input_size,
            self.output_size,
            self.d_ff,
            |kind, r, c| {
                out.push(w.slot(kind)[[r, c]]);
            },
        );
    }

    /// Nx2 `[summed deriv | frame count]` harvest in the SAME S1.2 order, so row `k` is
    /// the derivative of flat weight `k` (the seam's contract).
    pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        let count = self.nb_of_seq_fed_backward as f64;
        let d = &self.derivatives;
        for_each_slot(
            self.input_size,
            self.output_size,
            self.d_ff,
            |kind, r, c| {
                out.push([d.slot(kind)[[r, c]], count]);
            },
        );
    }

    /// Zero the deriv accumulators and the frame count.
    pub fn reset_weights_derivatives(&mut self) {
        self.nb_of_seq_fed_backward = 0;
        self.derivatives.fill(0.0);
    }

    /// Scale the deriv accumulators (the harvested col0) in place; the frame count
    /// (col1) is deliberately untouched.
    pub fn ponderate_weights_derivatives(&mut self, factor: f64) {
        self.derivatives.scale(factor);
    }

    /// Per-row attention weights (`T x A*wcap`) from the last forward -- see
    /// [`TransformerCache::attn_weights`] for the indexing.
    pub fn attn_weights(&self) -> &Array2<f64> {
        &self.cache.attn_weights
    }

    /// The stride of [`Self::attn_weights`] (`min(W, T)`, clamped `>= 1`).
    pub fn window_capacity(&self) -> usize {
        self.cache.wcap
    }

    /// `[q | k | v]` (`T x 3H`) from the last forward.
    pub fn qkv(&self) -> &Array2<f64> {
        &self.cache.qkv
    }

    /// `s` (`T x H`), the residual-1 output from the last forward.
    pub fn s(&self) -> &Array2<f64> {
        &self.cache.s
    }

    /// `x'` (`T x H`), the adapted input from the last forward.
    pub fn x_adapted(&self) -> &Array2<f64> {
        &self.cache.x_adapted
    }

    /// The first index of row `t`'s window, `max(0, t - W + 1)`. ONE definition, used by
    /// the forward, the backward and the tests -- an off-by-one here is precisely the
    /// mutation the phase's battery aims at, so it does not get written twice.
    fn window_begin(&self, t: usize) -> usize {
        (t + 1).saturating_sub(self.window)
    }

    /// The input at the layer's own width: crop (`cols > in`) or zero-pad (`cols < in`).
    /// One representation used by BOTH passes (the forward computes it, the backward
    /// reads the cached copy), so they cannot disagree about which columns exist.
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

    /// Whole-sequence forward (spec S1.1). Fills `output` (`T x H`) and the cache.
    /// `last_layer` is unused (signature parity with the trait).
    ///
    /// `needless_range_loop` is allowed throughout: every accumulation here is an
    /// explicit ASCENDING index loop by contract (the house numeric convention -- see
    /// `matmul_seq`), and an iterator rewrite would obscure which index runs innermost.
    #[allow(clippy::needless_range_loop)]
    pub fn feed_forward(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        let _ = last_layer;
        let (h, a, d, dff) = (self.output_size, self.heads, self.head_dim, self.d_ff);
        let w = &self.weights;
        let recon = self.reconcile_input(input);
        let t_len = recon.dim().0;
        let wcap = self.window.min(t_len).max(1);

        // 1. Width adapter: x' = W_a x + b_a.
        let mut x_adapted = matmul_seq(&recon, &w.adapter_w);
        for t in 0..t_len {
            for m in 0..h {
                x_adapted[[t, m]] += w.adapter_b[[0, m]];
            }
        }

        // 2. RMSNorm_1 with learned gain (the Mamba op order, verbatim).
        let mut u = Array2::<f64>::zeros((t_len, h));
        let mut rms1_inv = vec![0.0f64; t_len];
        for t in 0..t_len {
            let mut acc = 0.0;
            for m in 0..h {
                acc += x_adapted[[t, m]] * x_adapted[[t, m]];
            }
            let inv = 1.0 / (acc / h as f64 + RMS_EPS).sqrt();
            rms1_inv[t] = inv;
            for m in 0..h {
                u[[t, m]] = x_adapted[[t, m]] * inv * w.gain1[[0, m]];
            }
        }

        // 3. Combined [q | k | v] projection.
        let mut qkv = matmul_seq(&u, &w.qkv);
        for t in 0..t_len {
            for c in 0..QKV_NB * h {
                qkv[[t, c]] += w.qkv_b[[0, c]];
            }
        }

        // 4. Per-head windowed causal attention (ALiBi logits, max-subtracted softmax).
        //
        // Spec S7 (Task 8): `attn_weights` is written at row `t` and read back ONLY
        // within this SAME row's accumulation below -- forward never looks at another
        // row's weights -- so under `retain_cache == false` it shrinks to a single slot
        // (`ta` pinned at 0) instead of `t_len` rows; the retaining path keeps `ta == t`
        // and is therefore byte-untouched. See the module doc's "Retention" section.
        let retain = self.retain_cache;
        let mut attn_weights = Array2::<f64>::zeros((if retain { t_len } else { 1 }, a * wcap));
        let mut attn_out = Array2::<f64>::zeros((t_len, h));
        let mut row = vec![0.0f64; wcap];
        for t in 0..t_len {
            let ta = if retain { t } else { 0 };
            let lo = self.window_begin(t);
            let len = t - lo + 1;
            for hh in 0..a {
                let qb = BLOCK_Q * h + hh * d;
                let kb = BLOCK_K * h + hh * d;
                let vb = BLOCK_V * h + hh * d;
                let slope = self.slopes[hh];
                for jj in 0..len {
                    let j = lo + jj;
                    let mut acc = 0.0;
                    for e in 0..d {
                        acc += qkv[[t, qb + e]] * qkv[[j, kb + e]];
                    }
                    row[jj] = acc / self.sqrt_d + slope * (j as f64 - t as f64);
                }
                // Softmax, ALWAYS max-subtracted (spec S1.3): max, then exp, then
                // normalize, each an ascending loop.
                let mut mx = row[0];
                for jj in 1..len {
                    if row[jj] > mx {
                        mx = row[jj];
                    }
                }
                let mut sum = 0.0;
                for jj in 0..len {
                    let e = (row[jj] - mx).exp();
                    row[jj] = e;
                    sum += e;
                }
                for jj in 0..len {
                    attn_weights[[ta, hh * wcap + jj]] = row[jj] / sum;
                }
                for e in 0..d {
                    let mut acc = 0.0;
                    for jj in 0..len {
                        acc += attn_weights[[ta, hh * wcap + jj]] * qkv[[lo + jj, vb + e]];
                    }
                    attn_out[[t, hh * d + e]] = acc;
                }
            }
        }

        // 5. Residual 1: s = x' + W_o attn + b_o.
        let proj_o = matmul_seq(&attn_out, &w.out_w);
        let mut s = Array2::<f64>::zeros((t_len, h));
        for t in 0..t_len {
            for m in 0..h {
                s[[t, m]] = x_adapted[[t, m]] + proj_o[[t, m]] + w.out_b[[0, m]];
            }
        }

        // 6. RMSNorm_2 (same op order as RMSNorm_1, on s).
        let mut v_norm = Array2::<f64>::zeros((t_len, h));
        let mut rms2_inv = vec![0.0f64; t_len];
        for t in 0..t_len {
            let mut acc = 0.0;
            for m in 0..h {
                acc += s[[t, m]] * s[[t, m]];
            }
            let inv = 1.0 / (acc / h as f64 + RMS_EPS).sqrt();
            rms2_inv[t] = inv;
            for m in 0..h {
                v_norm[[t, m]] = s[[t, m]] * inv * w.gain2[[0, m]];
            }
        }

        // 7. FFN + residual 2: h = s + W_2 silu(W_1 v_norm + b_1) + b_2.
        let mut ffn_pre = matmul_seq(&v_norm, &w.ff1_w);
        let mut ffn_act = Array2::<f64>::zeros((t_len, dff));
        for t in 0..t_len {
            for r in 0..dff {
                let v = ffn_pre[[t, r]] + w.ff1_b[[0, r]];
                ffn_pre[[t, r]] = v;
                ffn_act[[t, r]] = silu(v);
            }
        }
        let proj2 = matmul_seq(&ffn_act, &w.ff2_w);
        output.fill(0.0);
        for t in 0..t_len {
            for m in 0..h {
                output[[t, m]] = s[[t, m]] + proj2[[t, m]] + w.ff2_b[[0, m]];
            }
        }

        // Spec S7 (Task 8): the retention, and the ONE place it is skipped. Everything
        // above was needed by the forward itself; only this assignment keeps it alive
        // past the call, and only the backward ever reads it back.
        if retain {
            self.cache = TransformerCache {
                x_in: recon,
                x_adapted,
                rms1_inv,
                u,
                qkv,
                attn_weights,
                wcap,
                attn_out,
                s,
                rms2_inv,
                v_norm,
                ffn_pre,
            };
        } else {
            self.cache = TransformerCache::default();
        }
    }

    /// Reverse-direction forward: flip the input rows, run [`Self::feed_forward`], flip
    /// the output back. The caches are LEFT in reversed-input order (consumed as-is by
    /// [`Self::feed_backward_reverse`]) -- the `LstmLayer` convention.
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

    /// Whole-sequence backward (spec S1.5, the adjoint equations in the module doc).
    /// Accumulates the weight-deriv blocks + the frame count, and returns the deltas
    /// propagated to the previous layer (`T x in`).
    ///
    /// `deltas` is `dL/dh`. The `Layer` trait's `input`, `output` and `last_layer`
    /// arguments are ABSENT from this signature ON PURPOSE (the CfC contract): every
    /// quantity the adjoints need is in the forward cache, [`TransformerCache::x_in`]
    /// included, so it is impossible for the two passes to disagree about the layer's own
    /// view of its input. The trait impl below discards them at the boundary, so the
    /// COMPILER enforces that rather than a `let _ = (..)` and a comment.
    #[allow(clippy::needless_range_loop)]
    pub fn feed_backward(
        &mut self,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
    ) -> Array2<f64> {
        // Spec S7 (Task 8): an inference-only layer never filled the cache (or shrank
        // `attn_weights` to a single dead slot), so there is nothing correct to fold.
        // Bail LOUDLY rather than accumulate garbage -- a silently wrong gradient is the
        // one failure mode this whole class of change must never produce.
        assert!(
            self.retain_cache,
            "TransformerLayer::feed_backward on a layer constructed inference-only \
             (retain_cache = false, phase-11 spec S7): the forward never filled the \
             backward cache, so no gradient can be folded here. This layer belongs to a \
             net built with BackPropagationActivated off; turn backprop on for that net \
             (or call set_retain_cache(true) before the forward) if a backward is really \
             wanted."
        );
        let (isz, h, a, d, dff) = (
            self.input_size,
            self.output_size,
            self.heads,
            self.head_dim,
            self.d_ff,
        );
        let w = &self.weights;
        let ca = &self.cache;
        let t_len = deltas.dim().0;
        let wcap = ca.wcap;

        let mut acc = TransformerWeights::zeros(isz, h, dff);

        // The three qkv adjoints. `dq` is complete per row; `dk`/`dv` are SCATTERED into
        // by every later row whose window covers them, so all three are whole-sequence
        // buffers consumed only after the reverse loop.
        let mut dq = Array2::<f64>::zeros((t_len, h));
        let mut dk = Array2::<f64>::zeros((t_len, h));
        let mut dv = Array2::<f64>::zeros((t_len, h));
        // dx' accumulates the residual-1 pass-through here and the RMSNorm_1 backward in
        // the second loop.
        let mut d_xa = Array2::<f64>::zeros((t_len, h));

        let mut d_act = vec![0.0f64; dff];
        let mut d_a1 = vec![0.0f64; dff];
        let mut d_vnorm = vec![0.0f64; h];
        let mut d_s = vec![0.0f64; h];
        let mut d_attn = vec![0.0f64; h];
        let mut dp = vec![0.0f64; wcap];
        let mut dl = vec![0.0f64; wcap];

        for t in (0..t_len).rev() {
            // --- FFN (residual 2 feeds dh straight through to ds below) ----------
            for r in 0..dff {
                // `silu` is re-derived from the cached PRE-activation rather than cached
                // itself (the Mamba convention), and hoisted out of the `m` loop it is
                // invariant in -- a loop-invariant hoist, not a reordering: the products
                // and their accumulation order are unchanged.
                let act = silu(ca.ffn_pre[[t, r]]);
                let mut sacc = 0.0;
                for m in 0..h {
                    sacc += w.ff2_w[[r, m]] * deltas[[t, m]];
                    acc.ff2_w[[r, m]] += act * deltas[[t, m]];
                }
                d_act[r] = sacc;
            }
            for m in 0..h {
                acc.ff2_b[[0, m]] += deltas[[t, m]];
            }
            for r in 0..dff {
                let g = d_act[r] * silu_deriv(ca.ffn_pre[[t, r]]);
                d_a1[r] = g;
                acc.ff1_b[[0, r]] += g;
            }
            for m in 0..h {
                let mut sacc = 0.0;
                for r in 0..dff {
                    sacc += w.ff1_w[[m, r]] * d_a1[r];
                    acc.ff1_w[[m, r]] += ca.v_norm[[t, m]] * d_a1[r];
                }
                d_vnorm[m] = sacc;
            }

            // --- RMSNorm_2: v_norm = s r2 g_2 ------------------------------------
            let r2 = ca.rms2_inv[t];
            let mut q2 = 0.0;
            for m in 0..h {
                q2 += d_vnorm[m] * ca.s[[t, m]] * w.gain2[[0, m]];
            }
            let coef2 = q2 * r2 * r2 * r2 / h as f64;
            for m in 0..h {
                acc.gain2[[0, m]] += d_vnorm[m] * ca.s[[t, m]] * r2;
                // The residual-2 pass-through (dh) plus the normalized branch.
                d_s[m] = deltas[[t, m]] + d_vnorm[m] * w.gain2[[0, m]] * r2 - coef2 * ca.s[[t, m]];
            }

            // --- Residual 1: s = x' + W_o attn + b_o -----------------------------
            for m in 0..h {
                d_xa[[t, m]] += d_s[m];
                acc.out_b[[0, m]] += d_s[m];
            }
            for k in 0..h {
                let mut sacc = 0.0;
                for m in 0..h {
                    sacc += w.out_w[[k, m]] * d_s[m];
                    acc.out_w[[k, m]] += ca.attn_out[[t, k]] * d_s[m];
                }
                d_attn[k] = sacc;
            }

            // --- Windowed attention ----------------------------------------------
            let lo = self.window_begin(t);
            let len = t - lo + 1;
            for hh in 0..a {
                let qb = BLOCK_Q * h + hh * d;
                let kb = BLOCK_K * h + hh * d;
                let vb = BLOCK_V * h + hh * d;
                // dp_{t,j} = sum_e dattn[e] v_j[e], and the v scatter in the same walk.
                for jj in 0..len {
                    let j = lo + jj;
                    let p = ca.attn_weights[[t, hh * wcap + jj]];
                    let mut sacc = 0.0;
                    for e in 0..d {
                        sacc += d_attn[hh * d + e] * ca.qkv[[j, vb + e]];
                        dv[[j, hh * d + e]] += p * d_attn[hh * d + e];
                    }
                    dp[jj] = sacc;
                }
                // Softmax Jacobian: dl = p .* (dp - sum(p .* dp)).
                let mut sdot = 0.0;
                for jj in 0..len {
                    sdot += ca.attn_weights[[t, hh * wcap + jj]] * dp[jj];
                }
                for jj in 0..len {
                    dl[jj] = ca.attn_weights[[t, hh * wcap + jj]] * (dp[jj] - sdot);
                }
                // The 1/sqrt(d) scale is applied ONCE per (t, j) and shared by the two
                // adjoints -- ALiBi adds nothing here, its slopes being constants.
                for jj in 0..len {
                    let j = lo + jj;
                    let dls = dl[jj] / self.sqrt_d;
                    for e in 0..d {
                        dq[[t, hh * d + e]] += dls * ca.qkv[[j, kb + e]];
                        dk[[j, hh * d + e]] += dls * ca.qkv[[t, qb + e]];
                    }
                }
            }
        }

        // --- qkv projection, RMSNorm_1, adapter (forward time) --------------------
        let mut dqkv = vec![0.0f64; QKV_NB * h];
        let mut du = vec![0.0f64; h];
        let mut dpl = Array2::<f64>::zeros((t_len, isz));
        for t in 0..t_len {
            for e in 0..h {
                dqkv[BLOCK_Q * h + e] = dq[[t, e]];
                dqkv[BLOCK_K * h + e] = dk[[t, e]];
                dqkv[BLOCK_V * h + e] = dv[[t, e]];
            }
            for c in 0..QKV_NB * h {
                acc.qkv_b[[0, c]] += dqkv[c];
            }
            for m in 0..h {
                let mut sacc = 0.0;
                for c in 0..QKV_NB * h {
                    sacc += w.qkv[[m, c]] * dqkv[c];
                    acc.qkv[[m, c]] += ca.u[[t, m]] * dqkv[c];
                }
                du[m] = sacc;
            }

            // RMSNorm_1: u = x' r1 g_1.
            let r1 = ca.rms1_inv[t];
            let mut q1 = 0.0;
            for m in 0..h {
                q1 += du[m] * ca.x_adapted[[t, m]] * w.gain1[[0, m]];
            }
            let coef1 = q1 * r1 * r1 * r1 / h as f64;
            for m in 0..h {
                acc.gain1[[0, m]] += du[m] * ca.x_adapted[[t, m]] * r1;
                d_xa[[t, m]] += du[m] * w.gain1[[0, m]] * r1 - coef1 * ca.x_adapted[[t, m]];
            }

            // Width adapter (always present -- the residuals add the ADAPTED input).
            for m in 0..h {
                acc.adapter_b[[0, m]] += d_xa[[t, m]];
            }
            for k in 0..isz {
                let mut sacc = 0.0;
                for m in 0..h {
                    sacc += w.adapter_w[[k, m]] * d_xa[[t, m]];
                    acc.adapter_w[[k, m]] += ca.x_in[[t, k]] * d_xa[[t, m]];
                }
                dpl[[t, k]] = sacc;
            }
        }

        if inv_sub_sampling_ratio > 1 {
            acc.scale(inv_sub_sampling_ratio as f64);
        }
        self.derivatives.add_assign(&acc);
        self.nb_of_seq_fed_backward += t_len as i64;

        dpl
    }

    /// Reverse-time backward: flip `deltas`, run the forward-order body, flip the
    /// returned deltas back. The caches are already time-reversed by
    /// [`Self::feed_forward_reverse`] and are consumed AS-IS (do NOT un-reverse).
    ///
    /// `deltas` is the ONLY argument for the same reason it is the only one on
    /// [`Self::feed_backward`]: it is the single quantity that body consumes from
    /// outside the cache, so it is the single one that needs flipping.
    pub fn feed_backward_reverse(
        &mut self,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
    ) -> Array2<f64> {
        let deltas_rev = deltas.slice(ndarray::s![..;-1, ..]).to_owned();
        let dpl = self.feed_backward(&deltas_rev, inv_sub_sampling_ratio);
        dpl.slice(ndarray::s![..;-1, ..]).to_owned()
    }
}

/// The trait impl is a thin forward to the inherent methods, mirroring
/// `impl Layer for CfcLayer`. It lives HERE rather than in `nn/network.rs` so the
/// new-cell code stays inside `nn/cells/` (spec S1/R2); `CellLayer`'s match arms call
/// the INHERENT methods by UFCS, so the two paths cannot diverge.
impl Layer for TransformerLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        TransformerLayer::feed_forward(self, input, output, last_layer);
    }
    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        TransformerLayer::feed_forward_reverse(self, input, output, last_layer);
    }
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let _ = (input, output, last_layer);
        TransformerLayer::feed_backward(self, deltas, inv_sub_sampling_ratio)
    }
    fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let _ = (input, output, last_layer);
        TransformerLayer::feed_backward_reverse(self, deltas, inv_sub_sampling_ratio)
    }
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        TransformerLayer::get_weights_derivatives(self, out);
    }
    fn reset_weights_derivatives(&mut self) {
        TransformerLayer::reset_weights_derivatives(self);
    }
    fn ponderate_weights_derivatives(&mut self, factor: f64) {
        TransformerLayer::ponderate_weights_derivatives(self, factor);
    }
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        TransformerLayer::set_weights(self, flat)
    }
    fn get_weights(&self, out: &mut Vec<f64>) {
        TransformerLayer::get_weights(self, out);
    }
    fn nb_of_weights(&self) -> usize {
        TransformerLayer::nb_of_weights(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: usize = 3;
    const O: usize = 4;
    const W: usize = 4;
    const A: usize = 2;
    const F: usize = 8;
    const T: usize = 7;

    fn params(window: usize, heads: usize, d_ff: usize) -> TransformerParams {
        TransformerParams {
            window,
            heads,
            d_ff,
        }
    }

    fn base_params() -> TransformerParams {
        params(W, A, F)
    }

    /// Deterministic, non-degenerate weight fill (no RNG dependency).
    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.29 - 0.011 * (k as f64) + 0.006 * ((k % 7) as f64))
            .collect()
    }

    /// A deterministic PSEUDO-RANDOM fill (Knuth-MMIX LCG, no `rand` dependency), for
    /// the legs that need genuinely UNSTRUCTURED weights.
    ///
    /// [`weights`] is a near-affine ramp, which makes `W_a` nearly RANK-ONE: every
    /// `x'_t` then points in almost the same direction, RMSNorm_1 removes what little
    /// scale separates them, and the resulting `u_t` are near-identical across `t`. The
    /// attention is still perfectly healthy (the measured `p` stay in `[0.23, 0.52]`,
    /// nowhere near saturation) but every `v_j` in a window is near-identical too, so
    /// `dl = p (dp - sum p dp)` cancels down to ~1e-9 against a pack maximum of ~5e1 --
    /// small enough to make a LIVE/DEAD contrast look ambiguous. That is a property of
    /// the fill, not of the cell, and this helper is the fix.
    fn pseudo(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((state >> 11) as f64) * (1.0 / 9007199254740992.0);
                u - 0.5
            })
            .collect()
    }

    fn seq(rows: usize, cols: usize, seed: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            seed + 0.17 * (r as f64) - 0.23 * (c as f64) + 0.011 * ((r * cols + c) as f64)
        })
    }

    fn loaded(i: usize, o: usize, p: &TransformerParams) -> TransformerLayer {
        let mut cell = TransformerLayer::new(i, o, p);
        cell.set_weights(&weights(cell.nb_of_weights()));
        cell
    }

    fn forward(cell: &mut TransformerLayer, input: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((input.nrows(), cell.output_size()));
        cell.feed_forward(input, &mut out, false);
        out
    }

    fn derivs(cell: &TransformerLayer) -> Vec<f64> {
        let mut rows = Vec::new();
        cell.get_weights_derivatives(&mut rows);
        rows.iter().map(|r| r[0]).collect()
    }

    /// An INDEPENDENT transcription of the S1.2 block order: `(name, start offset)`
    /// pairs plus a terminal `END`, so a block's `[lo, hi)` range comes from this table
    /// and not from the walk under test.
    fn offsets(i: usize, o: usize, dff: usize) -> Vec<(String, usize)> {
        let mut v = Vec::new();
        let mut p = 0usize;
        for (name, len) in [
            ("W_a", o * i),
            ("b_a", o),
            ("g_1", o),
            ("W_qkv", 3 * o * o),
            ("b_qkv", 3 * o),
            ("W_o", o * o),
            ("b_o", o),
            ("g_2", o),
            ("W_1", dff * o),
            ("b_1", dff),
            ("W_2", o * dff),
            ("b_2", o),
        ] {
            v.push((name.to_string(), p));
            p += len;
        }
        v.push(("END".to_string(), p));
        v
    }

    fn block(blocks: &[(String, usize)], name: &str) -> (usize, usize) {
        let k = blocks.iter().position(|(n, _)| n == name).unwrap();
        (blocks[k].1, blocks[k + 1].1)
    }

    // --- Weight seam (spec S1.2) -------------------------------------------

    #[test]
    fn nb_of_weights_is_the_spec_formula() {
        for (i, o, dff, heads) in [
            (3usize, 4usize, 8usize, 2usize),
            (5, 6, 12, 3),
            (1, 1, 1, 1),
            (11, 24, 64, 4),
        ] {
            let cell = TransformerLayer::new(i, o, &params(W, heads, dff));
            let want =
                o * (i + 1) + o + 3 * o * (o + 1) + o * (o + 1) + o + dff * (o + 1) + o * (dff + 1);
            assert_eq!(cell.nb_of_weights(), want, "in={i} out={o} d_ff={dff}");
            // The independent offset table must agree with the counted walk.
            assert_eq!(offsets(i, o, dff).last().unwrap().1, want);
        }
    }

    /// The flat layout is the S1.2 order, each matrix ROW-major. Checked POSITIONALLY
    /// with a one-hot probe per slot (a round trip alone would pass on any
    /// self-consistent permutation).
    #[test]
    fn flat_layout_positions_are_the_spec_order() {
        let (i, o, dff) = (3usize, 4usize, 5usize);
        let mut cell = TransformerLayer::new(i, o, &params(W, A, dff));
        let n = cell.nb_of_weights();
        let blocks = offsets(i, o, dff);

        // (block name, rows, cols, stored-index accessor). `rows` is the ROW-major outer
        // extent, `cols` the inner one.
        type Probe = (
            &'static str,
            usize,
            usize,
            fn(&TransformerWeights, usize, usize) -> f64,
        );
        let probes: &[Probe] = &[
            ("W_a", o, i, |w, j, k| w.adapter_w[[k, j]]),
            ("W_qkv", 3 * o, o, |w, j, k| w.qkv[[k, j]]),
            ("W_o", o, o, |w, j, k| w.out_w[[k, j]]),
            ("W_1", dff, o, |w, j, k| w.ff1_w[[k, j]]),
            ("W_2", o, dff, |w, j, k| w.ff2_w[[k, j]]),
        ];
        for (name, rows, cols, get) in probes {
            let (lo, hi) = block(&blocks, name);
            assert_eq!(hi - lo, rows * cols, "{name}: block length");
            for j in 0..*rows {
                for k in 0..*cols {
                    let mut flat = vec![0.0; n];
                    flat[lo + j * cols + k] = 1.0;
                    cell.set_weights(&flat);
                    assert_eq!(get(&cell.weights, j, k), 1.0, "{name}: [{j},{k}] misplaced");
                }
            }
        }

        type BiasProbe = (&'static str, usize, fn(&TransformerWeights, usize) -> f64);
        let biases: &[BiasProbe] = &[
            ("b_a", o, |w, j| w.adapter_b[[0, j]]),
            ("g_1", o, |w, j| w.gain1[[0, j]]),
            ("b_qkv", 3 * o, |w, j| w.qkv_b[[0, j]]),
            ("b_o", o, |w, j| w.out_b[[0, j]]),
            ("g_2", o, |w, j| w.gain2[[0, j]]),
            ("b_1", dff, |w, j| w.ff1_b[[0, j]]),
            ("b_2", o, |w, j| w.ff2_b[[0, j]]),
        ];
        for (name, len, get) in biases {
            let (lo, hi) = block(&blocks, name);
            assert_eq!(hi - lo, *len, "{name}: block length");
            for j in 0..*len {
                let mut flat = vec![0.0; n];
                flat[lo + j] = 1.0;
                cell.set_weights(&flat);
                assert_eq!(get(&cell.weights, j), 1.0, "{name}: [{j}] misplaced");
            }
        }
    }

    #[test]
    fn weight_seam_round_trips_and_returns_the_tail() {
        let mut cell = TransformerLayer::new(I, O, &base_params());
        let nb = cell.nb_of_weights();
        let w = weights(nb + 5);
        let tail = cell.set_weights(&w);
        assert_eq!(tail.len(), 5);
        assert_eq!(tail, &w[nb..]);

        let mut back = Vec::new();
        cell.get_weights(&mut back);
        assert_eq!(back, w[..nb].to_vec());
    }

    // --- Geometry + ALiBi (spec S1.1) ---------------------------------------

    /// The slopes are the standard geometric ladder `2^(-8h/A)`, `h = 1..A`, strictly
    /// decreasing, and EXACT powers of two whenever `A` divides 8 (the reason `exp2` is
    /// used rather than `powf`).
    #[test]
    fn alibi_slopes_are_the_geometric_ladder() {
        let cell = TransformerLayer::new(I, 8, &params(W, 4, F));
        assert_eq!(cell.slopes(), &[0.25, 0.0625, 0.015625, 0.00390625]);

        let one = TransformerLayer::new(I, 4, &params(W, 1, F));
        assert_eq!(one.slopes(), &[1.0 / 256.0]);

        let three = TransformerLayer::new(I, 6, &params(W, 3, F));
        assert_eq!(three.slopes().len(), 3);
        for h in 0..3 {
            assert_eq!(three.slopes()[h], (-8.0 * (h as f64 + 1.0) / 3.0).exp2());
        }
        for h in 1..3 {
            assert!(three.slopes()[h] < three.slopes()[h - 1], "not decreasing");
        }
    }

    /// A non-dividing head count is a HARD error naming both numbers (spec S2's
    /// `H % A == 0`), not a silent truncation.
    #[test]
    #[should_panic(expected = "must be divisible by the head count")]
    fn a_non_dividing_head_count_panics() {
        let _ = TransformerLayer::new(I, 5, &params(W, 2, F));
    }

    /// Degenerate geometry is CLAMPED at the cell (the config reader rejects it loudly
    /// one level up), so a directly-constructed layer can never divide by zero or run an
    /// empty window.
    #[test]
    fn zero_geometry_is_clamped_at_the_cell() {
        let cell = TransformerLayer::new(I, O, &params(0, 0, 0));
        assert_eq!((cell.window(), cell.heads(), cell.d_ff()), (1, 1, 1));
        assert_eq!(cell.head_dim(), O);
    }

    // --- Forward (spec S1.1) ------------------------------------------------

    #[test]
    fn forward_shapes_and_caches_track_the_sequence_length() {
        for t in [1usize, 2, 4, 5, 11] {
            let mut cell = loaded(I, O, &base_params());
            let out = forward(&mut cell, &seq(t, I, 0.4));
            assert_eq!(out.dim(), (t, O));
            assert_eq!(cell.qkv().dim(), (t, 3 * O));
            assert_eq!(cell.s().dim(), (t, O));
            assert_eq!(cell.x_adapted().dim(), (t, O));
            assert_eq!(cell.window_capacity(), W.min(t).max(1));
            assert_eq!(cell.attn_weights().dim(), (t, A * cell.window_capacity()));
            assert!(out.iter().all(|v| v.is_finite()), "t={t}");
        }
    }

    /// The forward is a pure function of (weights, input): run twice, bit-identical (no
    /// state leaks across calls -- the cell carries no cross-call state at all).
    #[test]
    fn forward_is_run_twice_bit_identical_and_stateless() {
        let mut cell = loaded(I, O, &base_params());
        let input = seq(T, I, 0.4);
        let first = forward(&mut cell, &input);
        let attn = cell.attn_weights().clone();
        let second = forward(&mut cell, &input);
        assert_eq!(first, second);
        assert_eq!(&attn, cell.attn_weights());

        let mut fresh = loaded(I, O, &base_params());
        assert_eq!(forward(&mut fresh, &input), first);
    }

    /// THE WINDOW CONTRACT: every row's attention is a probability distribution over
    /// EXACTLY `[max(0, t-W+1), t]` -- causal (nothing after `t`), bounded (never more
    /// than `W` entries), and normalized.
    #[test]
    fn attention_rows_are_causal_window_bounded_and_normalized() {
        let mut cell = loaded(I, O, &base_params());
        let _ = forward(&mut cell, &seq(T, I, 0.4));
        let wcap = cell.window_capacity();
        assert_eq!(wcap, W);
        for t in 0..T {
            let len = (t + 1).min(W);
            for hh in 0..A {
                let mut sum = 0.0;
                for jj in 0..len {
                    let p = cell.attn_weights()[[t, hh * wcap + jj]];
                    assert!(p > 0.0 && p <= 1.0, "t={t} head={hh} jj={jj}: p={p}");
                    sum += p;
                }
                assert!((sum - 1.0).abs() < 1e-15, "t={t} head={hh}: sum={sum}");
                // Slots past the row's own window length are untouched.
                for jj in len..wcap {
                    assert_eq!(cell.attn_weights()[[t, hh * wcap + jj]], 0.0);
                }
            }
        }
    }

    /// The window really TRUNCATES: a narrow window and a window covering the whole
    /// sequence produce different outputs at the rows where they differ, and `W >= T` is
    /// the same as any larger `W` (the edge-truncating convention, not a wrap).
    #[test]
    fn the_window_truncates_and_saturates() {
        let input = seq(T, I, 0.4);
        let narrow = forward(&mut loaded(I, O, &params(2, A, F)), &input);
        let full = forward(&mut loaded(I, O, &params(T, A, F)), &input);
        let wider = forward(&mut loaded(I, O, &params(T + 5, A, F)), &input);

        // Row 0 sees exactly one frame under any window: identical.
        for m in 0..O {
            assert_eq!(narrow[[0, m]], full[[0, m]], "row 0 must not depend on W");
        }
        // A row past the narrow window differs.
        let moved = (0..O)
            .map(|m| (narrow[[T - 1, m]] - full[[T - 1, m]]).abs())
            .fold(0.0f64, f64::max);
        assert!(moved > 1e-6, "the window is inert -- max |dh| = {moved:e}");
        // Beyond T the window saturates (edge truncation, no wrap-around).
        assert_eq!(full, wider);
    }

    /// HAND-COMPUTED single step at the smallest non-degenerate geometry: every quantity
    /// of S1.1 written out with literal constants. Asserts EQUALITY -- the port is
    /// expected to evaluate exactly these expressions, in this order.
    #[test]
    fn single_step_matches_the_hand_computed_value() {
        let (i, o, dff) = (1usize, 1usize, 1usize);
        let mut cell = TransformerLayer::new(i, o, &params(4, 1, dff));
        // Layout: W_a | b_a | g_1 | W_qkv (3x1) | b_qkv (3) | W_o (1x1) | b_o | g_2
        //         | W_1 (1x1) | b_1 | W_2 (1x1) | b_2
        let flat = vec![
            0.4,  // W_a
            0.15, // b_a
            1.3,  // g_1
            0.9, -0.6, 0.7, // W_qkv rows [q | k | v]
            0.05, 0.2, -0.1,  // b_qkv
            1.1,   // W_o
            -0.25, // b_o
            0.8,   // g_2
            0.65,  // W_1
            0.3,   // b_1
            -0.9,  // W_2
            0.12,  // b_2
        ];
        assert_eq!(flat.len(), cell.nb_of_weights());
        cell.set_weights(&flat);

        let x = 0.5;
        let input = Array2::from_shape_vec((1, 1), vec![x]).unwrap();
        let out = forward(&mut cell, &input);

        let xa = x * 0.4 + 0.15;
        let r1 = 1.0 / (xa * xa / 1.0 + RMS_EPS).sqrt();
        let u = xa * r1 * 1.3;
        let q = u * 0.9 + 0.05;
        let k = u * -0.6 + 0.2;
        let v = u * 0.7 - 0.1;
        // ONE-element window: l - max = 0, exp(0) = 1, p = 1/1 = 1 exactly -- which is
        // why q and k cannot move the output at T = 1 (the S1.4 dead block).
        let _ = (q, k);
        let attn = v;
        let s = xa + attn * 1.1 - 0.25;
        let r2 = 1.0 / (s * s / 1.0 + RMS_EPS).sqrt();
        let vn = s * r2 * 0.8;
        let a1 = vn * 0.65 + 0.3;
        let want = s + silu(a1) * -0.9 + 0.12;

        assert_eq!(cell.x_adapted()[[0, 0]], xa);
        assert_eq!(cell.qkv()[[0, 0]], q);
        assert_eq!(cell.qkv()[[0, 1]], k);
        assert_eq!(cell.qkv()[[0, 2]], v);
        assert_eq!(cell.attn_weights()[[0, 0]], 1.0);
        assert_eq!(cell.s()[[0, 0]], s);
        assert_eq!(out[[0, 0]], want);
        // Non-vacuity: the FFN branch actually moved the residual.
        assert!((out[[0, 0]] - s).abs() > 1e-3, "the FFN branch is inert");
    }

    /// One fixture geometry for [`multi_window_forward_matches_the_hand_transcription`].
    /// `head_dim` and `slopes` are stated as LITERALS rather than read back off the
    /// layer -- that independence is the whole instrument.
    struct Fixture {
        input_size: usize,
        h: usize,
        heads: usize,
        window: usize,
        d_ff: usize,
        t: usize,
        /// `d = H / A`, written out by hand.
        head_dim: usize,
        /// `m_h = 2^(-8h/A)`, `h = 1..A`, written out as literals.
        slopes: &'static [f64],
        /// The self-witness floor: measured/10 of the SMALLER of this fixture's two
        /// mutant sensitivities (see the test doc for both numbers). Per-fixture because
        /// they differ by two decades -- fixture 1 is the strong instrument, fixture 0
        /// the `d = 1` shape.
        mutant_floor: f64,
    }

    /// An INDEPENDENT scalar transcription of the whole S1.1 forward, reading the flat
    /// pack through the test module's own [`offsets`] table and NOTHING from the layer.
    ///
    /// `use_alibi` and `sqrt_d` are parameters so the SAME transcription can produce the
    /// two named mutants (drop the ALiBi term; scale by `sqrt(H)` instead of `sqrt(d)`),
    /// which is what makes the pin below self-witnessing: it does not merely assert
    /// agreement, it asserts that agreement would BREAK under each mutation.
    /// `needless_range_loop` is allowed for the same reason it is on the cell's own
    /// methods: every accumulation is an explicit ASCENDING index loop by contract, and
    /// an iterator rewrite would obscure which index runs innermost -- which is exactly
    /// what this transcription exists to state independently.
    #[allow(clippy::needless_range_loop)]
    fn transcribe(
        f: &Fixture,
        flat: &[f64],
        x: &Array2<f64>,
        use_alibi: bool,
        sqrt_d: f64,
    ) -> Array2<f64> {
        let (i, h, a, d, dff, win) = (f.input_size, f.h, f.heads, f.head_dim, f.d_ff, f.window);
        assert_eq!(d * a, h, "fixture geometry: d*A must be H");
        let b = offsets(i, h, dff);
        let at = |name: &str| block(&b, name).0;
        let (wa, ba, g1) = (at("W_a"), at("b_a"), at("g_1"));
        let (wqkv, bqkv) = (at("W_qkv"), at("b_qkv"));
        let (wo, bo, g2) = (at("W_o"), at("b_o"), at("g_2"));
        let (w1, b1, w2, b2) = (at("W_1"), at("b_1"), at("W_2"), at("b_2"));
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());

        let t_len = f.t;
        let mut out = Array2::<f64>::zeros((t_len, h));
        let mut qkv = vec![0.0f64; t_len * 3 * h];
        let mut xa = vec![0.0f64; t_len * h];
        // Rows 1-3: adapter -> RMSNorm_1 -> [q|k|v]. Causal, so one forward sweep.
        for t in 0..t_len {
            for m in 0..h {
                let mut acc = 0.0;
                for k in 0..i {
                    acc += x[[t, k]] * flat[wa + m * i + k];
                }
                acc += flat[ba + m];
                xa[t * h + m] = acc;
            }
            let mut sq = 0.0;
            for m in 0..h {
                sq += xa[t * h + m] * xa[t * h + m];
            }
            let inv = 1.0 / (sq / h as f64 + 1e-5).sqrt();
            let mut u = vec![0.0f64; h];
            for m in 0..h {
                u[m] = xa[t * h + m] * inv * flat[g1 + m];
            }
            for c in 0..3 * h {
                let mut acc = 0.0;
                for m in 0..h {
                    acc += u[m] * flat[wqkv + c * h + m];
                }
                qkv[t * 3 * h + c] = acc + flat[bqkv + c];
            }
        }
        // Rows 4-8: windowed attention -> residual 1 -> RMSNorm_2 -> FFN -> residual 2.
        for t in 0..t_len {
            let lo = (t + 1).saturating_sub(win);
            let len = t - lo + 1;
            let mut attn = vec![0.0f64; h];
            for hh in 0..a {
                let (qb, kb, vb) = (hh * d, h + hh * d, 2 * h + hh * d);
                let mut row = vec![0.0f64; len];
                for jj in 0..len {
                    let j = lo + jj;
                    let mut dot = 0.0;
                    for e in 0..d {
                        dot += qkv[t * 3 * h + qb + e] * qkv[j * 3 * h + kb + e];
                    }
                    row[jj] = dot / sqrt_d;
                    if use_alibi {
                        row[jj] += f.slopes[hh] * (j as f64 - t as f64);
                    }
                }
                let mut mx = row[0];
                for v in row.iter().skip(1) {
                    if *v > mx {
                        mx = *v;
                    }
                }
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                for e in 0..d {
                    let mut acc = 0.0;
                    for (jj, rv) in row.iter().enumerate() {
                        acc += (*rv / sum) * qkv[(lo + jj) * 3 * h + vb + e];
                    }
                    attn[hh * d + e] = acc;
                }
            }
            let mut s = vec![0.0f64; h];
            for m in 0..h {
                let mut acc = 0.0;
                for k in 0..h {
                    acc += attn[k] * flat[wo + m * h + k];
                }
                s[m] = xa[t * h + m] + acc + flat[bo + m];
            }
            let mut sq = 0.0;
            for m in 0..h {
                sq += s[m] * s[m];
            }
            let inv = 1.0 / (sq / h as f64 + 1e-5).sqrt();
            let mut vn = vec![0.0f64; h];
            for m in 0..h {
                vn[m] = s[m] * inv * flat[g2 + m];
            }
            let mut act = vec![0.0f64; dff];
            for r in 0..dff {
                let mut acc = 0.0;
                for m in 0..h {
                    acc += vn[m] * flat[w1 + r * h + m];
                }
                let pre = acc + flat[b1 + r];
                act[r] = pre * sig(pre);
            }
            for m in 0..h {
                let mut acc = 0.0;
                for r in 0..dff {
                    acc += act[r] * flat[w2 + m * dff + r];
                }
                out[[t, m]] = s[m] + acc + flat[b2 + m];
            }
        }
        out
    }

    /// THE MULTI-WINDOW VALUE PIN (T2 review, finding 1). The `T = 1` hand-computed leg
    /// above runs `A = 1`, `d = 1` and a ONE-ELEMENT window, where two of S1.1's logit
    /// terms are inert -- the ALiBi offset (`j - t == 0`) and the `1/sqrt(d)` scale
    /// (`d == H == 1`). MUTATION-PROVEN: deleting the ALiBi term outright, and
    /// substituting `sqrt(H)` for `sqrt(d)`, both used to pass the entire file.
    ///
    /// This leg closes that. Two fixtures, both with a sliding window that has a real
    /// EDGE (`T > W`) and both with `A > 1`, checked against an INDEPENDENT scalar
    /// [`transcribe`] that reads the pack through the test module's own layout table and
    /// touches nothing on the layer -- `d` and the ALiBi slopes are literals HERE:
    ///
    /// - `in=1 H=2 A=2 d=1 W=2 d_ff=1 T=3`, slopes `[2^-4, 2^-8]`, `sqrt(d) = 1`;
    /// - `in=3 H=8 A=4 d=2 W=3 d_ff=2 T=5`, slopes `[2^-2, 2^-4, 2^-6, 2^-8]`,
    ///   `sqrt(d) = sqrt(2)` -- the `A = 4` fixture also covers
    ///   [`TRANSFORMER_DEFAULT_HEADS`](super::super::blstm::TRANSFORMER_DEFAULT_HEADS),
    ///   which no other forward exercised (T2 review, finding 2), and its `d = 2` makes
    ///   the per-head SLICING observable (a head-stride bug mixes columns).
    ///
    /// The equality is EXACT (`assert_eq!`), not a tolerance: the transcription evaluates
    /// the same expressions in the same ascending order, `silu`'s `1/(1+e^-x)` included
    /// (both fixtures sit far inside the house `expLimit`, so no saturation branch
    /// differs). And the pin is SELF-WITNESSING: the same transcription is re-run as each
    /// mutant and asserted to DISAGREE, so this test cannot silently stop discriminating.
    ///
    /// MEASURED mutant sensitivities (max |dh| over the whole output):
    ///
    /// ```text
    ///                       drop ALiBi     sqrt(H) for sqrt(d)     floor
    ///   fixture 0 (A=2,d=1)  4.840e-5           3.690e-6           3.6e-7
    ///   fixture 1 (A=4,d=2)  7.391e-3           6.037e-4           6.0e-5
    /// ```
    ///
    /// Fixture 1 is the strong instrument by two decades; fixture 0 exists for the `d = 1`
    /// / `sqrt(d) = 1` shape, where the scale term is at its LEAST observable and is
    /// therefore worth owning explicitly. Both are ~10 decades above f64 noise, so the
    /// `assert_eq!` above is what actually catches a mutation -- these floors only stop
    /// the fixtures from silently drifting into a regime where it would not.
    #[test]
    fn multi_window_forward_matches_the_hand_transcription() {
        let fixtures = [
            Fixture {
                input_size: 1,
                h: 2,
                heads: 2,
                window: 2,
                d_ff: 1,
                t: 3,
                head_dim: 1,
                slopes: &[0.0625, 0.00390625],
                mutant_floor: 3.6e-7,
            },
            Fixture {
                input_size: 3,
                h: 8,
                heads: 4,
                window: 3,
                d_ff: 2,
                t: 5,
                head_dim: 2,
                slopes: &[0.25, 0.0625, 0.015625, 0.00390625],
                mutant_floor: 6.0e-5,
            },
        ];
        for (fi, f) in fixtures.iter().enumerate() {
            let p = params(f.window, f.heads, f.d_ff);
            let mut cell = TransformerLayer::new(f.input_size, f.h, &p);
            let flat = pseudo(cell.nb_of_weights(), 11 + fi as u64);
            cell.set_weights(&flat);
            let x = seq(f.t, f.input_size, 0.35);
            let got = forward(&mut cell, &x);

            let sqrt_d = (f.head_dim as f64).sqrt();
            let want = transcribe(f, &flat, &x, true, sqrt_d);
            assert_eq!(
                got, want,
                "fixture {fi}: forward disagrees with the transcription"
            );

            // NON-VACUITY 1: the window really slides and really truncates, so rows past
            // it carry a genuine edge (`len` caps at W) -- otherwise the ALiBi offsets
            // would all be 0 and the leg would be the T=1 case again.
            assert!(f.t > f.window, "fixture {fi}: no window edge");
            let wcap = cell.window_capacity();
            assert_eq!(wcap, f.window);
            for t in 0..f.t {
                let len = (t + 1).min(f.window);
                for hh in 0..f.heads {
                    let sum: f64 = (0..len)
                        .map(|jj| cell.attn_weights()[[t, hh * wcap + jj]])
                        .sum();
                    assert!((sum - 1.0).abs() < 1e-14, "fixture {fi}: row {t} head {hh}");
                }
            }

            // THE TWO NAMED MUTATIONS, asserted to be CAUGHT rather than assumed to be.
            let no_alibi = transcribe(f, &flat, &x, false, sqrt_d);
            let moved = (0..f.t)
                .flat_map(|r| (0..f.h).map(move |c| (r, c)))
                .map(|(r, c)| (got[[r, c]] - no_alibi[[r, c]]).abs())
                .fold(0.0f64, f64::max);
            assert!(
                moved > f.mutant_floor,
                "fixture {fi}: dropping the ALiBi term moves the output by only {moved:e} \
                 -- this pin does not own the ALiBi term"
            );

            let wrong_scale = transcribe(f, &flat, &x, true, (f.h as f64).sqrt());
            let moved = (0..f.t)
                .flat_map(|r| (0..f.h).map(move |c| (r, c)))
                .map(|(r, c)| (got[[r, c]] - wrong_scale[[r, c]]).abs())
                .fold(0.0f64, f64::max);
            assert!(
                moved > f.mutant_floor,
                "fixture {fi}: scaling by sqrt(H) instead of sqrt(d) moves the output by \
                 only {moved:e} -- this pin does not own the 1/sqrt(d) scale"
            );
        }
    }

    /// The ALWAYS-max-subtract convention (S1.3) is what keeps a large-logit row finite:
    /// a deliberately huge `W_qkv` produces logits far past `exp`'s overflow point, and
    /// the output stays finite and normalized.
    #[test]
    fn the_max_subtraction_survives_huge_logits() {
        let blocks = offsets(I, O, F);
        let (qkv_lo, qkv_hi) = block(&blocks, "W_qkv");
        let mut w = weights(TransformerLayer::new(I, O, &base_params()).nb_of_weights());
        for slot in w.iter_mut().take(qkv_hi).skip(qkv_lo) {
            *slot *= 1e3;
        }
        let mut cell = TransformerLayer::new(I, O, &base_params());
        cell.set_weights(&w);
        let out = forward(&mut cell, &seq(T, I, 0.4));
        assert!(out.iter().all(|v| v.is_finite()), "the softmax overflowed");
        let wcap = cell.window_capacity();
        for t in 0..T {
            let len = (t + 1).min(W);
            for hh in 0..A {
                let sum: f64 = (0..len)
                    .map(|jj| cell.attn_weights()[[t, hh * wcap + jj]])
                    .sum();
                assert!((sum - 1.0).abs() < 1e-12, "t={t} head={hh}: sum={sum}");
            }
        }
    }

    /// `feed_forward_reverse` is the forward on flipped rows (and leaves the caches
    /// reversed, the `LstmLayer` convention the reverse backward depends on).
    #[test]
    fn reverse_forward_is_the_flipped_forward() {
        let input = seq(T, I, 0.4);
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();

        let mut a = loaded(I, O, &base_params());
        let mut out_rev = Array2::<f64>::zeros((T, O));
        a.feed_forward_reverse(&input, &mut out_rev, false);

        let mut b = loaded(I, O, &base_params());
        let plain_on_flipped = forward(&mut b, &input_rev);

        assert_eq!(
            out_rev,
            plain_on_flipped.slice(ndarray::s![..;-1, ..]).to_owned()
        );
        assert_eq!(a.attn_weights(), b.attn_weights());
    }

    // --- Backward (spec S1.5) ----------------------------------------------

    /// Derivatives ACCUMULATE across `feed_backward` calls (col0 sums, col1 counts
    /// frames) and `reset_weights_derivatives` clears both.
    #[test]
    fn derivatives_accumulate_across_calls_and_reset_clears() {
        let mut cell = loaded(I, O, &base_params());
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);

        // The discarded `forward` return is LOAD-BEARING: it fills the cache the
        // backward reads (which is exactly why that signature takes no input/output).
        let _ = forward(&mut cell, &input);
        cell.reset_weights_derivatives();
        let _ = cell.feed_backward(&deltas, 1);
        let mut once = Vec::new();
        cell.get_weights_derivatives(&mut once);
        assert!(
            once.iter().any(|r| r[0] != 0.0),
            "no derivative accumulated"
        );
        assert!(once.iter().all(|r| r[1] == T as f64));

        let _ = cell.feed_backward(&deltas, 1);
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

    /// THE ALWAYS-DEAD BLOCK (S1.4): `b_k`. Shifting it moves every logit in a window
    /// row by the SAME amount, which softmax annihilates -- so the forward is invariant
    /// and the gradient vanishes at every `T`.
    ///
    /// TWO REGIMES, and the difference is real (see the module doc): at `T = 1` the
    /// one-element softmax makes `dl` bit-exactly `0.0`, so `== 0.0` is asserted; at
    /// `T >= 2` the zero is an f64 CANCELLATION (`sum_j dl_{t,j} = 0` in R), MEASURED at
    /// `2.168e-17` (`T = 3`) / `8.007e-17` (`T = 7`) against a whole-pack gradient
    /// maximum of ~5e1, and pinned at `1e-14` -- the sLSTM `b_i` convention exactly,
    /// rather than forcing the block to zero.
    #[test]
    fn the_key_bias_is_gradient_dead_at_every_length() {
        let blocks = offsets(I, O, F);
        let (qkv_b_lo, _) = block(&blocks, "b_qkv");
        let b_k: Vec<usize> = (0..O).map(|j| qkv_b_lo + BLOCK_K * O + j).collect();

        for t in [1usize, 3, 7] {
            let mut cell = loaded(I, O, &base_params());
            let input = seq(t, I, 0.4);
            let out = forward(&mut cell, &input);
            cell.reset_weights_derivatives();
            let _ = cell.feed_backward(&seq(t, O, -0.2), 1);
            let d = derivs(&cell);
            let worst = b_k.iter().map(|&s| d[s].abs()).fold(0.0f64, f64::max);
            if t == 1 {
                for &s in b_k.iter() {
                    assert_eq!(d[s], 0.0, "b_k w[{s}] must be EXACTLY zero at T=1");
                }
            } else {
                assert!(worst < 1e-14, "dL/db_k is not ~0 at T={t}: {worst:e}");
            }
            let overall = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            assert!(
                overall > 1e-3,
                "the whole gradient is ~0 -- the contrast is vacuous"
            );

            // The FORWARD invariance the zero comes from, pinned directly.
            let mut shifted = weights(cell.nb_of_weights());
            for &s in b_k.iter() {
                shifted[s] += 2.0;
            }
            let mut other = TransformerLayer::new(I, O, &base_params());
            other.set_weights(&shifted);
            let out2 = forward(&mut other, &input);
            let moved = (0..t)
                .flat_map(|r| (0..O).map(move |c| (r, c)))
                .map(|(r, c)| (out[[r, c]] - out2[[r, c]]).abs())
                .fold(0.0f64, f64::max);
            assert!(
                moved < 1e-13,
                "a b_k shift moved the output by {moved:e} at T={t}"
            );
        }
    }

    /// THE `T = 1` STRUCTURAL ZERO (S1.4): a one-element window makes the softmax
    /// constantly 1, so `dl = p (dp - p dp) = 0.0` bit-exactly and the q/k rows of
    /// `W_qkv` plus `b_q` have EXACTLY zero derivatives -- with a `T = 2` contrast in the
    /// same loop (the mamba `a_log_gradient_is_exactly_zero_at_t1` precedent).
    #[test]
    fn query_and_key_blocks_are_gradient_dead_at_t1() {
        let blocks = offsets(I, O, F);
        let (qkv_lo, _) = block(&blocks, "W_qkv");
        let (qkv_b_lo, _) = block(&blocks, "b_qkv");
        // Row-major (3H x H): row r -> flat qkv_lo + r*H + m. Rows [0,H) are q, [H,2H) k.
        let mut slots: Vec<usize> = (0..2 * O)
            .flat_map(|r| (0..O).map(move |m| qkv_lo + r * O + m))
            .collect();
        slots.extend((0..O).map(|j| qkv_b_lo + BLOCK_Q * O + j));
        assert_eq!(slots.len(), 2 * O * O + O);

        // The PSEUDO-RANDOM fill, not the ramp: see [`pseudo`] for why the ramp makes
        // this particular contrast ambiguous (it is degenerate, not wrong).
        let w = pseudo(
            TransformerLayer::new(I, O, &base_params()).nb_of_weights(),
            7,
        );
        for (t, want_zero) in [(1usize, true), (2, false)] {
            let mut cell = TransformerLayer::new(I, O, &base_params());
            cell.set_weights(&w);
            let _ = forward(&mut cell, &seq(t, I, 0.4));
            cell.reset_weights_derivatives();
            let _ = cell.feed_backward(&seq(t, O, -0.2), 1);
            let d = derivs(&cell);
            let worst = slots.iter().map(|&s| d[s].abs()).fold(0.0f64, f64::max);
            if want_zero {
                for &s in slots.iter() {
                    assert_eq!(d[s], 0.0, "q/k slot w[{s}] is not exactly zero at T=1");
                }
                let overall = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                assert!(overall > 1e-3, "the whole gradient is ~0 at T=1 -- vacuous");
            } else {
                // MEASURED worst |dL/dw| over the q/k slots at T=2 = 7.604e-4; pinned at
                // 1e-4, i.e. 7.6x under the measured contrast. (The T=1 side needs no
                // margin at all -- it is an exact `== 0.0`, so any positive floor here
                // separates the two regimes; the margin exists only to survive libm
                // variance in the measured value.)
                assert!(
                    worst > 1e-4,
                    "the q/k blocks are still dead at T=2 ({worst:e}) -- the T=1 claim is \
                     not about T at all"
                );
            }
        }
    }

    /// `ponderate_weights_derivatives` scales col0 only; `inv_sub_sampling_ratio` scales
    /// the accumulated blocks (the house conventions).
    #[test]
    fn ponderation_scales_col0_only_and_issr_scales_the_blocks() {
        let mut cell = loaded(I, O, &base_params());
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);
        let _ = forward(&mut cell, &input);

        cell.reset_weights_derivatives();
        let _ = cell.feed_backward(&deltas, 1);
        let mut base = Vec::new();
        cell.get_weights_derivatives(&mut base);
        cell.ponderate_weights_derivatives(0.25);
        let mut pond = Vec::new();
        cell.get_weights_derivatives(&mut pond);
        for k in 0..base.len() {
            assert_eq!(pond[k][0], base[k][0] * 0.25);
            assert_eq!(pond[k][1], base[k][1], "the frame count must not scale");
        }

        let mut scaled = loaded(I, O, &base_params());
        let _ = forward(&mut scaled, &input);
        scaled.reset_weights_derivatives();
        let _ = scaled.feed_backward(&deltas, 3);
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

        let mut a = loaded(I, O, &base_params());
        let mut out_rev = Array2::<f64>::zeros((T, O));
        a.feed_forward_reverse(&input, &mut out_rev, false);
        let dpl_rev = a.feed_backward_reverse(&deltas, 1);
        let mut da = Vec::new();
        a.get_weights_derivatives(&mut da);

        let mut b = loaded(I, O, &base_params());
        let input_f = input.slice(ndarray::s![..;-1, ..]).to_owned();
        let _ = forward(&mut b, &input_f);
        let dpl_f = b.feed_backward(&deltas.slice(ndarray::s![..;-1, ..]).to_owned(), 1);
        let mut db = Vec::new();
        b.get_weights_derivatives(&mut db);

        assert_eq!(dpl_rev, dpl_f.slice(ndarray::s![..;-1, ..]).to_owned());
        assert_eq!(da, db);
        assert!(da.iter().any(|r| r[0] != 0.0));
    }

    /// The propagated deltas are `T x in` (the layer's own width) even when the caller
    /// feeds a narrower or wider sequence -- the width-tolerance contract the forward and
    /// backward share through the cached reconciled input.
    #[test]
    fn width_tolerance_is_shared_by_both_passes() {
        let narrow = seq(T, 1, 0.4);
        let wide = seq(T, I + 2, 0.4);
        for input in [&narrow, &wide] {
            let mut cell = loaded(I, O, &base_params());
            let out = forward(&mut cell, input);
            assert_eq!(out.dim(), (T, O));
            let dpl = cell.feed_backward(&seq(T, O, -0.2), 1);
            assert_eq!(dpl.dim(), (T, I));
        }

        // Cropping is exactly "ignore the extra columns".
        let mut a = loaded(I, O, &base_params());
        let mut b = loaded(I, O, &base_params());
        assert_eq!(
            forward(&mut a, &wide),
            forward(&mut b, &wide.slice(ndarray::s![.., ..I]).to_owned())
        );
    }

    // ==== Phase 11 Task 8 / spec S7: inference-only cache gating ====

    /// The DEFAULT is retain, so nothing built outside `BlstmNetwork::from_config` can
    /// lose its gradient by accident.
    #[test]
    fn a_fresh_layer_retains_its_cache() {
        let cell = loaded(I, O, &base_params());
        assert!(cell.retains_cache());
    }

    /// THE BEHAVIOUR-FREE CLAIM, measured: with the cache off, the forward's OUTPUT is
    /// bit-for-bit what the retaining forward produces -- which is also the pin on the
    /// `attn_weights` shrink (`ta` pinned at slot 0), since a wrong slot would move every
    /// downstream `attn_out`/`s`/`v_norm`/`ffn_pre` value. Run over three sequence lengths
    /// incl. `T = 1` (a one-element window, the S1.4 dead-block edge) and `T = 2` (the
    /// first row whose window covers more than itself).
    #[test]
    fn forward_is_bit_identical_without_the_cache() {
        for t in [1usize, 2, T] {
            let input = seq(t, I, 0.37);
            let retained = forward(&mut loaded(I, O, &base_params()), &input);

            let mut lean = loaded(I, O, &base_params());
            lean.set_retain_cache(false);
            let dropped = forward(&mut lean, &input);

            assert_eq!(retained, dropped, "T={t}: the cache flag moved the forward");
            assert!(dropped.iter().all(|v| v.is_finite()));
        }
    }

    /// ... and the cache really is gone afterwards (the memory claim, not just the
    /// numeric one). All five accessors, so a partially-cleared cache fails here.
    #[test]
    fn an_inference_only_forward_leaves_every_cache_field_empty() {
        let mut cell = loaded(I, O, &base_params());
        cell.set_retain_cache(false);
        let _ = forward(&mut cell, &seq(T, I, 0.37));
        assert_eq!(cell.attn_weights().dim(), (0, 0));
        assert_eq!(cell.qkv().dim(), (0, 0));
        assert_eq!(cell.s().dim(), (0, 0));
        assert_eq!(cell.x_adapted().dim(), (0, 0));
        assert_eq!(cell.window_capacity(), 0);
    }

    /// Turning retention OFF drops what an earlier forward left behind -- otherwise a
    /// backward could fold a STALE cache from a different sequence and look plausible.
    #[test]
    fn turning_retention_off_drops_the_existing_cache() {
        let mut cell = loaded(I, O, &base_params());
        let _ = forward(&mut cell, &seq(T, I, 0.37));
        assert_eq!(cell.x_adapted().dim(), (T, O));
        cell.set_retain_cache(false);
        assert_eq!(cell.x_adapted().dim(), (0, 0));
    }

    /// THE R6 CLONE NOTE (spec S7): `corpus_processor`'s static-lane fold clones the whole
    /// bag per lane at every epoch start, and a clone is a DEEP copy of this cache. Under
    /// inference-only there is nothing to copy -- asserted here rather than assumed, on
    /// the structure the bag clone actually duplicates.
    #[test]
    fn a_clone_after_an_inference_only_forward_carries_no_cache() {
        let mut cell = loaded(I, O, &base_params());
        cell.set_retain_cache(false);
        let _ = forward(&mut cell, &seq(T, I, 0.37));
        let twin = cell.clone();
        assert!(
            !twin.retains_cache(),
            "the flag itself must survive the clone"
        );
        assert_eq!(twin.x_adapted().dim(), (0, 0));
        assert_eq!(twin.attn_weights().dim(), (0, 0));

        // Non-vacuity: the SAME assertion on a retaining clone must fail, i.e. the clone
        // does carry the cache when there is one.
        let mut keeper = loaded(I, O, &base_params());
        let _ = forward(&mut keeper, &seq(T, I, 0.37));
        assert_eq!(keeper.clone().x_adapted().dim(), (T, O));
    }

    /// A backward after a non-retaining forward must PANIC, never fold garbage.
    #[test]
    #[should_panic(expected = "constructed inference-only")]
    fn backward_after_an_inference_only_forward_panics() {
        let input = seq(T, I, 0.37);
        let mut cell = loaded(I, O, &base_params());
        cell.set_retain_cache(false);
        let _ = forward(&mut cell, &input);
        let _ = cell.feed_backward(&seq(T, O, -0.21), 1);
    }

    /// The retaining path still folds a real gradient -- the contrast that makes the
    /// panic above a GATE rather than a blanket refusal.
    #[test]
    fn backward_still_works_when_the_cache_is_retained() {
        let input = seq(T, I, 0.37);
        let mut cell = loaded(I, O, &base_params());
        let _ = forward(&mut cell, &input);
        let _ = cell.feed_backward(&seq(T, O, -0.21), 1);
        let g = derivs(&cell);
        assert!(g.iter().any(|v| *v != 0.0) && g.iter().all(|v| v.is_finite()));
    }

    /// THE EXPLICIT PIN the task asks for: retention ON is BYTE-IDENTICAL to a "never
    /// gated" run. A cell that never once calls `set_retain_cache` takes the same
    /// retaining branch as one that explicitly opts back into `true` (the default in both
    /// cases), so this says adding the flag perturbed NEITHER the forward output NOR the
    /// folded gradient on the path every pre-Task-8 test in this file already exercises.
    #[test]
    fn explicit_retain_true_is_bit_identical_to_never_touching_the_flag() {
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);

        let mut never_gated = loaded(I, O, &base_params());
        let out_a = forward(&mut never_gated, &input);
        let _ = never_gated.feed_backward(&deltas, 1);
        let mut da = Vec::new();
        never_gated.get_weights_derivatives(&mut da);

        let mut explicitly_retained = loaded(I, O, &base_params());
        explicitly_retained.set_retain_cache(true);
        let out_b = forward(&mut explicitly_retained, &input);
        let _ = explicitly_retained.feed_backward(&deltas, 1);
        let mut db = Vec::new();
        explicitly_retained.get_weights_derivatives(&mut db);

        assert_eq!(out_a, out_b, "the forward output moved");
        assert_eq!(da, db, "the flat [deriv | count] rows moved");
        assert!(
            da.iter().any(|r| r[0] != 0.0),
            "vacuous: derivatives are all zero"
        );
    }
}
