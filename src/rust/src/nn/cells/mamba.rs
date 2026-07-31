//! Mamba (S6) in RECURRENT form: a selective state-space cell whose `Delta_t`,
//! `B_t` and `C_t` are per-timestep functions of the input (the selectivity), run as
//! an explicit elementwise linear recurrence rather than the paper's parallel scan.
//!
//! Port-only (Phase 9 Task 3) -- NO legacy source. Spec:
//! `docs/superpowers/specs/2026-07-27-phase-9-new-architectures-design.md`, S3.1
//! (forward), S3.2 (flat layout), S3.3 (hyperparameters), S3.4 (the load-bearing
//! seeded init, which lives Python-side), S3.5 (backward + the cache note). Numeric
//! divergence from anything legacy is BY DESIGN and is documented here + in
//! `RESULTS.md`, never `IMPROVEMENTS.md` (the phase-7 rule).
//!
//! What landed: the whole-sequence f64 forward (S3.1), a HAND-DERIVED analytic
//! backward (S3.5, pinned against central differences by
//! `tests/phase9_cell_grad.rs`), and the `Layer`-conforming weight/derivative seam in
//! the S3.2 flat order. Training-ready: the layer plugs into `Network<CellLayer>` and
//! therefore into every windowed driver, the flat-weight seam and SMORMS3, with no
//! caller aware of the cell type. The seeded init of S3.4 is NOT here -- it is
//! `init_weights.py`'s job (Task 4), exactly as for the LSTM and sLSTM.
//!
//! ## Forward (S3.1)
//!
//! `d_model = output_size`, `d_inner = expand * d_model`, `dt_rank` resolved from the
//! config (`0` -> `ceil(d_model / 16)`).
//!
//! ```text
//! x'_t  = P x_t + p                          ONLY when in != out (the width adapter)
//! xn_t  = x'_t / sqrt(mean(x'_t^2) + eps) * g          (RMSNorm, eps = RMS_EPS = 1e-5)
//! [u | res]_t = W_in xn_t                              (2 d_inner x d_model, NO bias)
//! pc_t  = causal_conv1d(u)_t + b_conv                  (depthwise, kernel d_conv)
//! uc_t  = SiLU(pc_t)
//! [dt~ | B | C]_t = W_x uc_t                (dt_rank + 2 d_state rows, NO bias)
//! dtp_t = W_dt dt~_t + b_dt                 (d_inner x dt_rank, WITH bias)
//! Delta_t = softplus(dtp_t)
//! A       = -exp(A_log)                                (recomputed per call)
//! Abar_t[c,s] = exp(Delta_t[c] A[c,s])
//! h_t[c,s]    = Abar_t[c,s] h_{t-1}[c,s] + Delta_t[c] uc_t[c] B_t[s]
//! y_t[c]      = sum_s C_t[s] h_t[c,s] + D[c] uc_t[c]
//! v_t         = y_t * SiLU(res_t)
//! out_t       = W_out v_t + x'_t                       (d_model x d_inner, NO bias)
//! ```
//!
//! The residual adds `x'` -- the ADAPTED input, not the raw `x_t` -- which is what
//! makes the block shape-closed when `in != out`.
//!
//! CAUSAL CONV CONVENTION (load-bearing, and the one the backward's flipped
//! correlation mirrors): the kernel is LEFT-padded with `d_conv - 1` zeros, so tap
//! `k` sees `u[t - (d_conv - 1 - k)]`; tap `d_conv - 1` is the CURRENT sample and tap
//! `0` the oldest. Out-of-range (negative) taps contribute exactly nothing -- they
//! are skipped, not multiplied by a stored zero.
//!
//! STABILITY BY CONSTRUCTION: `A < 0` (it is `-exp`) and `Delta > 0` (it is
//! `softplus`), so `Abar = exp(Delta A) in (0, 1)` for every channel, state and
//! timestep -- the recurrence can neither blow up nor flip sign. Pinned by
//! `abar_stays_strictly_inside_the_unit_interval`. `softplus` carries an explicit
//! overflow guard (see [`softplus`]): a large `dtp` would overflow `exp` in the naive
//! `ln(1 + exp(x))` form, and a large `Delta` is exactly what an untrained/large
//! `b_dt` produces, so the guard is on the live from-scratch path, not a theoretical
//! nicety.
//!
//! ## Flat layout (S3.2)
//!
//! Row-major per matrix, blocks in THIS order (`[X | b]` means the whole `X` block
//! row-major FOLLOWED BY the whole `b` block -- the same block-sequential reading
//! `nn::cells::slstm`'s `[R | W | b]` uses, not a per-row interleave):
//!
//! ```text
//! [P | p]              only when in != out   (out x in, then out)
//! g                    d_model
//! W_in                 2 d_inner x d_model
//! conv | b_conv        d_inner x d_conv, then d_inner
//! W_x                  (dt_rank + 2 d_state) x d_inner
//! W_dt | b_dt          d_inner x dt_rank, then d_inner
//! A_log                d_inner x d_state
//! D                    d_inner
//! W_out                d_model x d_inner
//! ```
//!
//! Walked EXACTLY ONCE by [`for_each_slot`], which `set_weights`, `get_weights` and
//! `get_weights_derivatives` all drive -- so the three orders cannot drift apart
//! (spec S1.3: the Rust `set_weights` order IS the layout).
//!
//! ## Backward (S3.5) -- the adjoint equations, reverse time
//!
//! `dout_t` is the incoming delta. `sig` is the logistic; `SiLU'(x) = sig(x)(1 +
//! x(1 - sig(x)))`; `softplus'(x) = sig(x)`.
//!
//! ```text
//! out:   dW_out[m,c] += dout_t[m] v_t[c]      dv_t[c] = sum_m W_out[m,c] dout_t[m]
//!        dx'_t += dout_t                                       (residual, straight through)
//! gate:  dy_t[c]   = dv_t[c] SiLU(res_t[c])
//!        dres_t[c] = dv_t[c] y_t[c] SiLU'(res_t[c])
//! SSM:   dD[c]  += dy_t[c] uc_t[c]                 duc_t[c]  = dy_t[c] D[c] + ...
//!        dC_t[s] = sum_c dy_t[c] h_t[c,s]
//!        dh_t[c,s] = dy_t[c] C_t[s] + Abar_{t+1}[c,s] dh_{t+1}[c,s]      (state adjoint)
//!        dAbar_t[c,s] = dh_t[c,s] h_{t-1}[c,s]
//!        dDelta_t[c] = sum_s ( dAbar_t[c,s] Abar_t[c,s] A[c,s] + dh_t[c,s] uc_t[c] B_t[s] )
//!        dA_log[c,s] += dAbar_t[c,s] Abar_t[c,s] Delta_t[c] A[c,s]   (exp product rule twice:
//!                       dAbar/dA = Abar Delta, dA/dA_log = -exp(A_log) = A)
//!        duc_t[c]   += sum_s dh_t[c,s] Delta_t[c] B_t[s]
//!        dB_t[s]     = sum_c dh_t[c,s] Delta_t[c] uc_t[c]
//! dt:    d_dtp_t[c] = dDelta_t[c] sig(dtp_t[c])
//!        dW_dt[c,r] += d_dtp_t[c] dt~_t[r]        db_dt[c] += d_dtp_t[c]
//!        d_dt~_t[r]  = sum_c W_dt[c,r] d_dtp_t[c]
//! x_pr:  dz_t = [d_dt~_t | dB_t | dC_t]
//!        dW_x[i,c]  += dz_t[i] uc_t[c]            duc_t[c] += sum_i W_x[i,c] dz_t[i]
//! conv:  d_pc_t[c]  = duc_t[c] SiLU'(pc_t[c])     db_conv[c] += sum_t d_pc_t[c]
//!        dconv[c,k] += sum_t d_pc_t[c] u[t - (d_conv-1-k)][c]
//!        du_t[c]     = sum_k conv[c,k] d_pc[t + (d_conv-1-k)][c]   (flipped correlation)
//! in_pr: dw_t = [du_t | dres_t]
//!        dW_in[i,m] += dw_t[i] xn_t[m]            dxn_t[m] = sum_i W_in[i,m] dw_t[i]
//! norm:  r = rms_inv_t,  q = sum_j dxn_t[j] x'_t[j] g[j]
//!        dg[m]  += dxn_t[m] x'_t[m] r
//!        dx'_t[m] += dxn_t[m] g[m] r - (r^3 / d_model) x'_t[m] q
//! adapt: dP[j,k] += dx'_t[j] x_t[k]   dp[j] += dx'_t[j]   dx_t[k] = sum_j P[j,k] dx'_t[j]
//!        (in == out: dx_t = dx'_t, no adapter at all)
//! ```
//!
//! Note `dx'` accumulates TWO contributions -- the residual pass-through and the
//! RMSNorm backward -- which is why it is a whole `T x d_model` buffer rather than a
//! per-step scalar chain.
//!
//! THE ONE STRUCTURAL ORDERING CONSTRAINT: the conv backward needs the COMPLETE
//! `d_pc` sequence before it can produce `du` (a tap at `t` reads `d_pc` at
//! `t + off`), so the pass is split -- a reverse-time loop down to `d_pc`/`d_res`,
//! then the conv fold, then a forward-time loop for the input projection, RMSNorm and
//! adapter. Everything inside each loop is an ascending accumulation.
//!
//! ## Structurally gradient-dead weights (what the FD tier's resolvable count pins)
//!
//! Two blocks are EXACTLY zero-gradient at short sequences, by structure and not by
//! accident:
//!
//! - `A_log` at `T = 1`. `dAbar_t = dh_t . h_{t-1}` and `h_{-1} = 0`, so the whole
//!   `d_inner x d_state` block is zero: at a single timestep `Abar` multiplies the
//!   zero initial state and can move nothing.
//! - the OLDEST conv taps when `T < d_conv`. Tap `k` first contributes at
//!   `t = d_conv - 1 - k`, so taps `k < d_conv - T` never see a real sample -- their
//!   left-padding is all they ever multiply. That is `max(0, d_conv - T)` dead taps
//!   per channel.
//!
//! Both are asserted (as a resolvable COUNT, per shape) by `tests/phase9_cell_grad.rs`
//! and pinned directly by `a_log_gradient_is_exactly_zero_at_t1` /
//! `old_conv_taps_are_gradient_dead_below_the_kernel_length` here.
//!
//! ## Conventions inherited from the house (`nn::layers::LstmLayer`, `cells::slstm`)
//!
//! - Every PROJECTION matrix is stored TRANSPOSED (`w[[k, j]] == W[j, k]`), so each
//!   product is a plain [`matmul_seq`] call on `T x ...` sequences -- the ascending
//!   accumulation contract, uniformly. `conv_w` and `a_log` are NOT transposed: they
//!   are indexed `[channel, tap]` / `[channel, state]` and never enter a matmul.
//! - `get_weights_derivatives` returns Nx2 `[summed deriv | frame count]` in the flat
//!   order, `ponderate_weights_derivatives` scales col0 only, `set_weights` returns
//!   the unconsumed tail, `inv_sub_sampling_ratio > 1` scales the deriv blocks.
//! - `feed_forward_reverse` / `feed_backward_reverse` flip rows around the
//!   forward-order body and leave the caches in REVERSED order (consumed as-is by the
//!   reverse backward). Mamba is CAUSAL, so the reverse direction is a genuine
//!   anti-causal pass over the flipped sequence -- the same trick the wrapper already
//!   uses to get a bidirectional stack out of a one-directional cell.
//! - Input WIDTH TOLERANCE: `cols > in` crops to the left `in` columns, `cols < in`
//!   zero-pads. The propagated deltas are always `T x in`.

use ndarray::Array2;

use super::super::activations::logistic_fn;
use super::super::layers::matmul_seq;
use super::super::network::Layer;

/// RMSNorm's variance floor (spec S3.1). Not tunable: it is part of the block
/// definition, exactly like the `1e-5` every Mamba reference implementation uses.
pub const RMS_EPS: f64 = 1e-5;

/// `softplus(x) = ln(1 + e^x)`, evaluated in the branch that cannot overflow.
///
/// The naive form overflows `exp` at `x > ~709.78` and returns `inf` for a function
/// whose true value is just `x`. THE BRANCH IS LOAD-BEARING, not cosmetic: `Delta =
/// softplus(W_dt dt~ + b_dt)` and S3.4 seeds `b_dt = softplus^-1(Delta_0)`, so a
/// from-scratch run with an unlucky draw (or any run whose `dt` pre-activation grows)
/// walks straight into the overflow region. For `x > 0` we use the algebraically
/// identical `x + ln(1 + e^{-x})` (the exponent is now negative, so `e^{-x}`
/// underflows to 0 harmlessly and the result degrades gracefully to `x`); for
/// `x <= 0` the direct form is already safe. `ln_1p` is used on both branches because
/// its argument is small exactly where `ln(1 + .)` loses precision.
///
/// The derivative is [`logistic_fn`] on BOTH branches (`softplus' = sigmoid`), which
/// is why no companion `softplus_deriv` exists.
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// `SiLU(x) = x sigmoid(x)` (a.k.a. swish), on the house [`logistic_fn`] (whose
/// `expLimit` saturation guards apply). No overflow guard is needed: `sigmoid` is
/// bounded and the product is finite wherever `x` is.
fn silu(x: f64) -> f64 {
    x * logistic_fn(x)
}

/// `SiLU'(x) = sigmoid(x) (1 + x (1 - sigmoid(x)))`, taken on the PRE-activation `x`
/// (unlike the house `gates_deriv`/`maxmin2_deriv`, which read post-activations) --
/// which is why the forward caches `pc` and `res` rather than only their SiLU images.
fn silu_deriv(x: f64) -> f64 {
    let s = logistic_fn(x);
    s * (1.0 + x * (1.0 - s))
}

/// Which weight block a flat-layout slot addresses (see [`for_each_slot`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotKind {
    /// `P` (width adapter), stored transposed `in x out`.
    AdapterW,
    /// `p` (width adapter bias), `1 x out`.
    AdapterB,
    /// `g` (RMSNorm gain), `1 x d_model`.
    NormGain,
    /// `W_in`, stored transposed `d_model x 2 d_inner`.
    WIn,
    /// `conv` (depthwise kernels), `d_inner x d_conv`, NOT transposed.
    ConvW,
    /// `b_conv`, `1 x d_inner`.
    ConvB,
    /// `W_x`, stored transposed `d_inner x (dt_rank + 2 d_state)`.
    WX,
    /// `W_dt`, stored transposed `dt_rank x d_inner`.
    WDt,
    /// `b_dt`, `1 x d_inner`.
    BDt,
    /// `A_log`, `d_inner x d_state`, NOT transposed.
    ALog,
    /// `D` (the skip/feedthrough gain), `1 x d_inner`.
    DSkip,
    /// `W_out`, stored transposed `d_inner x d_model`.
    WOut,
}

/// The twelve weight blocks, in one struct so the DERIVATIVE accumulator is literally
/// the same type -- the shapes cannot drift apart, and `reset`/`ponderate`/`fold` are
/// one loop each instead of twelve statements.
#[derive(Clone)]
struct MambaWeights {
    adapter_w: Array2<f64>,
    adapter_b: Array2<f64>,
    norm_gain: Array2<f64>,
    w_in: Array2<f64>,
    conv_w: Array2<f64>,
    conv_b: Array2<f64>,
    w_x: Array2<f64>,
    w_dt: Array2<f64>,
    b_dt: Array2<f64>,
    a_log: Array2<f64>,
    d_skip: Array2<f64>,
    w_out: Array2<f64>,
}

impl MambaWeights {
    /// All blocks zeroed at their spec shapes. `adapter_w`/`adapter_b` are built
    /// EMPTY when `in == out` (no adapter exists then, spec S3.1/S3.2), so a stray
    /// write would panic rather than silently land in an unread buffer.
    fn zeros(
        input_size: usize,
        d_model: usize,
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        dt_rank: usize,
        has_adapter: bool,
    ) -> MambaWeights {
        let (aw, ab) = if has_adapter {
            (
                Array2::zeros((input_size, d_model)),
                Array2::zeros((1, d_model)),
            )
        } else {
            (Array2::zeros((0, 0)), Array2::zeros((0, 0)))
        };
        MambaWeights {
            adapter_w: aw,
            adapter_b: ab,
            norm_gain: Array2::zeros((1, d_model)),
            w_in: Array2::zeros((d_model, 2 * d_inner)),
            conv_w: Array2::zeros((d_inner, d_conv)),
            conv_b: Array2::zeros((1, d_inner)),
            w_x: Array2::zeros((d_inner, dt_rank + 2 * d_state)),
            w_dt: Array2::zeros((dt_rank, d_inner)),
            b_dt: Array2::zeros((1, d_inner)),
            a_log: Array2::zeros((d_inner, d_state)),
            d_skip: Array2::zeros((1, d_inner)),
            w_out: Array2::zeros((d_inner, d_model)),
        }
    }

    fn blocks(&self) -> [&Array2<f64>; 12] {
        [
            &self.adapter_w,
            &self.adapter_b,
            &self.norm_gain,
            &self.w_in,
            &self.conv_w,
            &self.conv_b,
            &self.w_x,
            &self.w_dt,
            &self.b_dt,
            &self.a_log,
            &self.d_skip,
            &self.w_out,
        ]
    }

    fn blocks_mut(&mut self) -> [&mut Array2<f64>; 12] {
        [
            &mut self.adapter_w,
            &mut self.adapter_b,
            &mut self.norm_gain,
            &mut self.w_in,
            &mut self.conv_w,
            &mut self.conv_b,
            &mut self.w_x,
            &mut self.w_dt,
            &mut self.b_dt,
            &mut self.a_log,
            &mut self.d_skip,
            &mut self.w_out,
        ]
    }

    fn slot(&self, kind: SlotKind) -> &Array2<f64> {
        match kind {
            SlotKind::AdapterW => &self.adapter_w,
            SlotKind::AdapterB => &self.adapter_b,
            SlotKind::NormGain => &self.norm_gain,
            SlotKind::WIn => &self.w_in,
            SlotKind::ConvW => &self.conv_w,
            SlotKind::ConvB => &self.conv_b,
            SlotKind::WX => &self.w_x,
            SlotKind::WDt => &self.w_dt,
            SlotKind::BDt => &self.b_dt,
            SlotKind::ALog => &self.a_log,
            SlotKind::DSkip => &self.d_skip,
            SlotKind::WOut => &self.w_out,
        }
    }

    fn slot_mut(&mut self, kind: SlotKind) -> &mut Array2<f64> {
        match kind {
            SlotKind::AdapterW => &mut self.adapter_w,
            SlotKind::AdapterB => &mut self.adapter_b,
            SlotKind::NormGain => &mut self.norm_gain,
            SlotKind::WIn => &mut self.w_in,
            SlotKind::ConvW => &mut self.conv_w,
            SlotKind::ConvB => &mut self.conv_b,
            SlotKind::WX => &mut self.w_x,
            SlotKind::WDt => &mut self.w_dt,
            SlotKind::BDt => &mut self.b_dt,
            SlotKind::ALog => &mut self.a_log,
            SlotKind::DSkip => &mut self.d_skip,
            SlotKind::WOut => &mut self.w_out,
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

    fn add_assign(&mut self, other: &MambaWeights) {
        for (dst, src) in self.blocks_mut().into_iter().zip(other.blocks()) {
            *dst += src;
        }
    }
}

/// Walk the S3.2 flat layout EXACTLY ONCE, in order, calling `f(kind, row, col)` with
/// the STORED-matrix indices of each successive flat element (see the module doc for
/// the block order and the transposed-storage convention).
///
/// `set_weights`, `get_weights` and `get_weights_derivatives` all drive this, so the
/// three orders cannot drift apart -- and `nb_of_weights` is derived from the SAME
/// walk (`for_each_slot` counting), so a layout edit cannot desynchronize the length
/// from the order either.
///
/// The seven dimensions are passed individually rather than bundled (the house
/// pattern -- `audio::`/`fast::nn`/`features::mel` do the same): the walk is the
/// layout's single definition, and a struct in front of it would just be a second
/// place the same seven numbers live.
#[allow(clippy::too_many_arguments)]
fn for_each_slot(
    input_size: usize,
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
    has_adapter: bool,
    mut f: impl FnMut(SlotKind, usize, usize),
) {
    // [P | p] -- ONLY when in != out.
    if has_adapter {
        for j in 0..d_model {
            for k in 0..input_size {
                f(SlotKind::AdapterW, k, j);
            }
        }
        for j in 0..d_model {
            f(SlotKind::AdapterB, 0, j);
        }
    }
    // g
    for m in 0..d_model {
        f(SlotKind::NormGain, 0, m);
    }
    // W_in (2 d_inner x d_model), row-major
    for i in 0..2 * d_inner {
        for m in 0..d_model {
            f(SlotKind::WIn, m, i);
        }
    }
    // conv (d_inner x d_conv) | b_conv
    for c in 0..d_inner {
        for k in 0..d_conv {
            f(SlotKind::ConvW, c, k);
        }
    }
    for c in 0..d_inner {
        f(SlotKind::ConvB, 0, c);
    }
    // W_x ((dt_rank + 2 d_state) x d_inner), row-major
    for i in 0..dt_rank + 2 * d_state {
        for c in 0..d_inner {
            f(SlotKind::WX, c, i);
        }
    }
    // W_dt (d_inner x dt_rank) | b_dt
    for c in 0..d_inner {
        for r in 0..dt_rank {
            f(SlotKind::WDt, r, c);
        }
    }
    for c in 0..d_inner {
        f(SlotKind::BDt, 0, c);
    }
    // A_log (d_inner x d_state), row-major
    for c in 0..d_inner {
        for s in 0..d_state {
            f(SlotKind::ALog, c, s);
        }
    }
    // D
    for c in 0..d_inner {
        f(SlotKind::DSkip, 0, c);
    }
    // W_out (d_model x d_inner), row-major
    for m in 0..d_model {
        for c in 0..d_inner {
            f(SlotKind::WOut, c, m);
        }
    }
}

/// Everything the backward reads back from the forward (spec S3.5's cache note).
///
/// PRE- vs POST-activation is the classic trap here, so each field says which it is:
/// `pc` and `res` are PRE-SiLU (the adjoints call [`silu_deriv`], which takes the
/// pre-activation), `uc` and `v` are POST (they multiply other adjoints directly),
/// `dtp` is PRE-softplus (its derivative is `sigmoid(dtp)`) while `delta` is POST
/// (it multiplies inside the recurrence).
#[derive(Clone, Default)]
struct MambaCache {
    /// `x'` -- the ADAPTED input (`T x d_model`). Read by the RMSNorm backward and
    /// the `dg` accumulation.
    x_adapted: Array2<f64>,
    /// `1 / sqrt(mean(x'^2) + eps)` per timestep (`T`).
    rms_inv: Vec<f64>,
    /// `xn` (`T x d_model`), the RMSNorm output -- the `dW_in` outer-product factor.
    xn: Array2<f64>,
    /// `[u | res]` (`T x 2 d_inner`): `u` is the conv INPUT, `res` the PRE-SiLU gate
    /// branch.
    proj: Array2<f64>,
    /// `pc` (`T x d_inner`): conv output + `b_conv`, PRE-SiLU.
    pc: Array2<f64>,
    /// `uc` (`T x d_inner`): POST-SiLU, the SSM's per-step input.
    uc: Array2<f64>,
    /// `[dt~ | B | C]` (`T x (dt_rank + 2 d_state)`), the selectivity projection.
    zproj: Array2<f64>,
    /// `dtp` (`T x d_inner`), PRE-softplus.
    dtp: Array2<f64>,
    /// `Delta` (`T x d_inner`), POST-softplus.
    delta: Array2<f64>,
    /// `A = -exp(A_log)` (`d_inner x d_state`). A pure function of the weights,
    /// cached only to avoid recomputing `exp` in the backward.
    a_mat: Array2<f64>,
    /// `Abar` (`T x (d_inner d_state)`, `[c * d_state + s]`).
    abar: Array2<f64>,
    /// `h` (`T x (d_inner d_state)`, same indexing). The big one -- S3.5's named
    /// memory cost, accepted at subset scale.
    h: Array2<f64>,
    /// `y` (`T x d_inner`), the SSM output before the SiLU gate.
    y: Array2<f64>,
    /// `v = y * SiLU(res)` (`T x d_inner`), the `W_out` input.
    v: Array2<f64>,
}

/// The Mamba/S6 cell (spec S3). See the module doc for the math, the layout and the
/// house conventions it inherits.
#[derive(Clone)]
pub struct MambaLayer {
    input_size: usize,
    /// `d_model` -- the layer's output width.
    output_size: usize,
    d_state: usize,
    d_conv: usize,
    /// `expand * d_model`.
    d_inner: usize,
    /// RESOLVED (`0` was replaced by `ceil(d_model / 16)` at construction).
    dt_rank: usize,
    /// `input_size != output_size`: the `[P | p]` width adapter exists.
    has_adapter: bool,

    weights: MambaWeights,
    cache: MambaCache,

    derivatives: MambaWeights,
    nb_of_seq_fed_backward: i64,
}

impl MambaLayer {
    /// New layer with zeroed weights (callers set them via `set_weights`, exactly like
    /// `LstmLayer::new`/`SlstmLayer::new`). Seeded init lives Python-side
    /// (`init_weights.py`, spec S3.4/S7.1).
    ///
    /// `dt_rank == 0` resolves to `ceil(d_model / 16)` (spec S3.3). `d_state`,
    /// `d_conv` and `expand` are FLOORED at 1: a zero would silently build empty
    /// blocks and a degenerate recurrence, and the config reader
    /// (`BlstmConfig::from_legacy`) rejects zeros loudly before they ever reach here,
    /// so the floor is belt-and-braces for direct callers, not a config path.
    pub fn new(
        input_size: usize,
        output_size: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        dt_rank: usize,
    ) -> MambaLayer {
        let d_state = d_state.max(1);
        let d_conv = d_conv.max(1);
        let d_inner = expand.max(1) * output_size;
        let dt_rank = if dt_rank == 0 {
            output_size.div_ceil(16).max(1)
        } else {
            dt_rank
        };
        let has_adapter = input_size != output_size;
        let weights = MambaWeights::zeros(
            input_size,
            output_size,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            has_adapter,
        );
        MambaLayer {
            input_size,
            output_size,
            d_state,
            d_conv,
            d_inner,
            dt_rank,
            has_adapter,
            derivatives: weights.clone(),
            weights,
            cache: MambaCache::default(),
            nb_of_seq_fed_backward: 0,
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn d_state(&self) -> usize {
        self.d_state
    }

    pub fn d_conv(&self) -> usize {
        self.d_conv
    }

    pub fn d_inner(&self) -> usize {
        self.d_inner
    }

    /// The RESOLVED `dt_rank` (never 0 -- see [`MambaLayer::new`]).
    pub fn dt_rank(&self) -> usize {
        self.dt_rank
    }

    /// Whether the `[P | p]` width adapter exists (`in != out`, spec S3.1).
    pub fn has_adapter(&self) -> bool {
        self.has_adapter
    }

    /// Counted off the SAME [`for_each_slot`] walk that defines the layout, so the
    /// length and the order are one fact, not two:
    ///
    /// ```text
    /// [in != out] (out in + out) + out + 2 d_inner d_model
    ///   + d_inner (d_conv + 1) + (dt_rank + 2 d_state) d_inner
    ///   + d_inner (dt_rank + 1) + d_inner d_state + d_inner + d_model d_inner
    /// ```
    pub fn nb_of_weights(&self) -> usize {
        let mut n = 0usize;
        self.walk(|_, _, _| n += 1);
        n
    }

    /// [`for_each_slot`] bound to this layer's dimensions.
    fn walk(&self, f: impl FnMut(SlotKind, usize, usize)) {
        for_each_slot(
            self.input_size,
            self.output_size,
            self.d_inner,
            self.d_state,
            self.d_conv,
            self.dt_rank,
            self.has_adapter,
            f,
        );
    }

    /// Consume `flat[..nb_of_weights()]` in the S3.2 order and return the tail for the
    /// next layer in the stack (the chained head-eating contract).
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        let (needed, left) = flat.split_at(self.nb_of_weights());
        let w = &mut self.weights;
        let mut p = 0usize;
        for_each_slot(
            self.input_size,
            self.output_size,
            self.d_inner,
            self.d_state,
            self.d_conv,
            self.dt_rank,
            self.has_adapter,
            |kind, r, c| {
                w.slot_mut(kind)[[r, c]] = needed[p];
                p += 1;
            },
        );
        left
    }

    /// The mirror of [`Self::set_weights`]: append this layer's weights to `out` in
    /// the same S3.2 order.
    pub fn get_weights(&self, out: &mut Vec<f64>) {
        let w = &self.weights;
        self.walk(|kind, r, c| out.push(w.slot(kind)[[r, c]]));
    }

    /// Nx2 `[summed deriv | frame count]` harvest in the SAME S3.2 order, so row `k`
    /// is the derivative of flat weight `k` (the seam's contract).
    pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        let count = self.nb_of_seq_fed_backward as f64;
        let d = &self.derivatives;
        self.walk(|kind, r, c| out.push([d.slot(kind)[[r, c]], count]));
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

    /// `Delta_t` (`T x d_inner`) from the last forward: POST-softplus, strictly
    /// positive.
    pub fn delta(&self) -> &Array2<f64> {
        &self.cache.delta
    }

    /// `Abar_t[c,s]` (`T x (d_inner d_state)`, index `c * d_state + s`) from the last
    /// forward: strictly inside `(0, 1)` by construction.
    pub fn abar(&self) -> &Array2<f64> {
        &self.cache.abar
    }

    /// `h_t[c,s]` (`T x (d_inner d_state)`, same indexing) from the last forward.
    pub fn hidden_states(&self) -> &Array2<f64> {
        &self.cache.h
    }

    /// `uc_t` (`T x d_inner`) from the last forward: the POST-SiLU conv output.
    pub fn conv_activations(&self) -> &Array2<f64> {
        &self.cache.uc
    }

    /// `1 / sqrt(mean(x'^2) + eps)` per timestep from the last forward.
    pub fn rms_inv(&self) -> &[f64] {
        &self.cache.rms_inv
    }

    /// The input at the layer's own width: crop (`cols > in`) or zero-pad
    /// (`cols < in`). One representation used by BOTH passes, so the forward
    /// projection and the backward's `dP` accumulation can never disagree about which
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

    /// Whole-sequence forward (spec S3.1). Fills `output` (`T x d_model`) and the
    /// backward cache. `last_layer` is unused (signature parity with the trait).
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
        let (dm, di, ds, dc, dr) = (
            self.output_size,
            self.d_inner,
            self.d_state,
            self.d_conv,
            self.dt_rank,
        );
        let w = &self.weights;
        let recon = self.reconcile_input(input);
        let t_len = recon.dim().0;

        // 1. Width adapter (spec S3.1): x' = P x + p, or x' = x when in == out.
        let x_adapted = if self.has_adapter {
            let mut xa = matmul_seq(&recon, &w.adapter_w);
            for t in 0..t_len {
                for m in 0..dm {
                    xa[[t, m]] += w.adapter_b[[0, m]];
                }
            }
            xa
        } else {
            recon.clone()
        };

        // 2. RMSNorm with learned gain.
        let mut xn = Array2::<f64>::zeros((t_len, dm));
        let mut rms_inv = vec![0.0f64; t_len];
        for t in 0..t_len {
            let mut acc = 0.0;
            for m in 0..dm {
                acc += x_adapted[[t, m]] * x_adapted[[t, m]];
            }
            let inv = 1.0 / (acc / dm as f64 + RMS_EPS).sqrt();
            rms_inv[t] = inv;
            for m in 0..dm {
                xn[[t, m]] = x_adapted[[t, m]] * inv * w.norm_gain[[0, m]];
            }
        }

        // 3. Input projection -> [u | res] (no bias).
        let proj = matmul_seq(&xn, &w.w_in);

        // 4. Depthwise CAUSAL conv (left-padded d_conv-1 zeros) + bias + SiLU. Tap k
        //    reads u[t - (d_conv-1-k)]; taps reaching before t = 0 are skipped.
        let mut pc = Array2::<f64>::zeros((t_len, di));
        let mut uc = Array2::<f64>::zeros((t_len, di));
        for t in 0..t_len {
            for c in 0..di {
                let mut acc = 0.0;
                for k in 0..dc {
                    let off = dc - 1 - k;
                    if t >= off {
                        acc += w.conv_w[[c, k]] * proj[[t - off, c]];
                    }
                }
                let p = acc + w.conv_b[[0, c]];
                pc[[t, c]] = p;
                uc[[t, c]] = silu(p);
            }
        }

        // 5. Selectivity projection -> [dt~ | B | C] (no bias).
        let zproj = matmul_seq(&uc, &w.w_x);
        let dt_tilde = zproj.slice(ndarray::s![.., 0..dr]).to_owned();

        // 6. Delta = softplus(W_dt dt~ + b_dt).
        let mut dtp = matmul_seq(&dt_tilde, &w.w_dt);
        let mut delta = Array2::<f64>::zeros((t_len, di));
        for t in 0..t_len {
            for c in 0..di {
                let v = dtp[[t, c]] + w.b_dt[[0, c]];
                dtp[[t, c]] = v;
                delta[[t, c]] = softplus(v);
            }
        }

        // 7. A = -exp(A_log), then the selective recurrence.
        let mut a_mat = Array2::<f64>::zeros((di, ds));
        for c in 0..di {
            for s in 0..ds {
                a_mat[[c, s]] = -w.a_log[[c, s]].exp();
            }
        }
        let mut abar = Array2::<f64>::zeros((t_len, di * ds));
        let mut h = Array2::<f64>::zeros((t_len, di * ds));
        let mut y = Array2::<f64>::zeros((t_len, di));
        for t in 0..t_len {
            for c in 0..di {
                let dl = delta[[t, c]];
                let ucv = uc[[t, c]];
                let mut acc = 0.0;
                for s in 0..ds {
                    let idx = c * ds + s;
                    let ab = (dl * a_mat[[c, s]]).exp();
                    let h_prev = if t > 0 { h[[t - 1, idx]] } else { 0.0 };
                    let hv = ab * h_prev + dl * ucv * zproj[[t, dr + s]];
                    abar[[t, idx]] = ab;
                    h[[t, idx]] = hv;
                    acc += zproj[[t, dr + ds + s]] * hv;
                }
                y[[t, c]] = acc + w.d_skip[[0, c]] * ucv;
            }
        }

        // 8. Gate by SiLU(res), out-project, add the residual x'.
        let mut v = Array2::<f64>::zeros((t_len, di));
        for t in 0..t_len {
            for c in 0..di {
                v[[t, c]] = y[[t, c]] * silu(proj[[t, di + c]]);
            }
        }
        let projected = matmul_seq(&v, &w.w_out);
        output.fill(0.0);
        for t in 0..t_len {
            for m in 0..dm {
                output[[t, m]] = projected[[t, m]] + x_adapted[[t, m]];
            }
        }

        self.cache = MambaCache {
            x_adapted,
            rms_inv,
            xn,
            proj,
            pc,
            uc,
            zproj,
            dtp,
            delta,
            a_mat,
            abar,
            h,
            y,
            v,
        };
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

    /// Whole-sequence BPTT (spec S3.5, the adjoint equations in the module doc).
    /// Accumulates the twelve deriv blocks + the frame count, and returns the deltas
    /// propagated to the previous layer (`T x in`).
    ///
    /// `output` is UNUSED: unlike the LSTM/sLSTM backward (which reads `h_{t-1}` back
    /// out of the retained layer output), every state this backward needs is in the
    /// forward cache. The parameter stays for `Layer` signature parity.
    ///
    /// `needless_range_loop` is allowed for the same reason as the forward: the
    /// ascending index order IS the contract.
    #[allow(clippy::needless_range_loop)]
    pub fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let _ = (output, last_layer);
        let (isz, dm, di, ds, dc, dr) = (
            self.input_size,
            self.output_size,
            self.d_inner,
            self.d_state,
            self.d_conv,
            self.dt_rank,
        );
        let zsz = dr + 2 * ds;
        let t_len = deltas.dim().0;
        let recon_input = self.reconcile_input(input);
        let w = &self.weights;
        let ca = &self.cache;

        let mut acc = MambaWeights::zeros(isz, dm, di, ds, dc, dr, self.has_adapter);

        // Carried across the reverse loop: Abar_{t+1} * dh_{t+1} (zero at t = T-1).
        let mut carried = vec![0.0f64; di * ds];
        // Needed AFTER the reverse loop (the conv backward is not causal in the
        // adjoint: du_t reads d_pc at LATER rows).
        let mut d_pc = Array2::<f64>::zeros((t_len, di));
        let mut d_res = Array2::<f64>::zeros((t_len, di));
        // dx' accumulates the residual pass-through here and the RMSNorm backward
        // in the second loop.
        let mut d_xa = Array2::<f64>::zeros((t_len, dm));

        let mut dv = vec![0.0f64; di];
        let mut dy = vec![0.0f64; di];
        let mut duc = vec![0.0f64; di];
        let mut d_delta = vec![0.0f64; di];
        let mut dz = vec![0.0f64; zsz];
        let mut d_dtp = vec![0.0f64; di];

        for t in (0..t_len).rev() {
            // --- out projection + residual -----------------------------------
            for c in 0..di {
                let mut s = 0.0;
                for m in 0..dm {
                    s += w.w_out[[c, m]] * deltas[[t, m]];
                }
                dv[c] = s;
                for m in 0..dm {
                    acc.w_out[[c, m]] += ca.v[[t, c]] * deltas[[t, m]];
                }
            }
            for m in 0..dm {
                d_xa[[t, m]] += deltas[[t, m]];
            }

            // --- SiLU gate: v = y * SiLU(res) --------------------------------
            for c in 0..di {
                let r = ca.proj[[t, di + c]];
                dy[c] = dv[c] * silu(r);
                d_res[[t, c]] = dv[c] * ca.y[[t, c]] * silu_deriv(r);
            }

            // --- selective SSM ------------------------------------------------
            for s in 0..zsz {
                dz[s] = 0.0;
            }
            for c in 0..di {
                let dl = ca.delta[[t, c]];
                let ucv = ca.uc[[t, c]];
                let mut acc_duc = dy[c] * w.d_skip[[0, c]];
                let mut acc_ddelta = 0.0;
                acc.d_skip[[0, c]] += dy[c] * ucv;
                for s in 0..ds {
                    let idx = c * ds + s;
                    let ab = ca.abar[[t, idx]];
                    // State adjoint: this step's own dy term + the carry from t+1.
                    let dh = dy[c] * ca.zproj[[t, dr + ds + s]] + carried[idx];
                    let h_prev = if t > 0 { ca.h[[t - 1, idx]] } else { 0.0 };
                    let dabar = dh * h_prev;
                    let a = ca.a_mat[[c, s]];
                    // Abar = exp(Delta A); A = -exp(A_log) -> dA/dA_log = A.
                    acc_ddelta += dabar * ab * a;
                    acc.a_log[[c, s]] += dabar * ab * dl * a;
                    // Input term Delta * uc * B.
                    let bv = ca.zproj[[t, dr + s]];
                    acc_ddelta += dh * ucv * bv;
                    acc_duc += dh * dl * bv;
                    dz[dr + s] += dh * dl * ucv;
                    // Output sum y = sum_s C h + D uc.
                    dz[dr + ds + s] += dy[c] * ca.h[[t, idx]];
                    // Carry for the next (earlier) row.
                    carried[idx] = ab * dh;
                }
                d_delta[c] = acc_ddelta;
                duc[c] = acc_duc;
            }

            // --- Delta = softplus(W_dt dt~ + b_dt) ----------------------------
            for c in 0..di {
                let g = d_delta[c] * logistic_fn(ca.dtp[[t, c]]);
                d_dtp[c] = g;
                acc.b_dt[[0, c]] += g;
            }
            for r in 0..dr {
                let mut s = 0.0;
                for c in 0..di {
                    s += w.w_dt[[r, c]] * d_dtp[c];
                    acc.w_dt[[r, c]] += ca.zproj[[t, r]] * d_dtp[c];
                }
                dz[r] = s;
            }

            // --- [dt~ | B | C] = W_x uc ---------------------------------------
            for c in 0..di {
                let mut s = duc[c];
                for i in 0..zsz {
                    s += w.w_x[[c, i]] * dz[i];
                    acc.w_x[[c, i]] += ca.uc[[t, c]] * dz[i];
                }
                duc[c] = s;
            }

            // --- SiLU on the conv output (pointwise half) ---------------------
            for c in 0..di {
                let g = duc[c] * silu_deriv(ca.pc[[t, c]]);
                d_pc[[t, c]] = g;
                acc.conv_b[[0, c]] += g;
            }
        }

        // --- depthwise causal conv backward (needs the whole d_pc) ------------
        // Forward: pc[t,c] = sum_k conv[c,k] u[t - off][c], off = d_conv - 1 - k.
        // So dconv[c,k] = sum_{t >= off} d_pc[t,c] u[t - off][c] and
        //    du[t,c]    = sum_{k : t + off < T} conv[c,k] d_pc[t + off][c].
        let mut du = Array2::<f64>::zeros((t_len, di));
        for c in 0..di {
            for k in 0..dc {
                let off = dc - 1 - k;
                let mut s = 0.0;
                for t in off..t_len {
                    s += d_pc[[t, c]] * ca.proj[[t - off, c]];
                }
                acc.conv_w[[c, k]] += s;
            }
            for t in 0..t_len {
                let mut s = 0.0;
                for k in 0..dc {
                    let off = dc - 1 - k;
                    if t + off < t_len {
                        s += w.conv_w[[c, k]] * d_pc[[t + off, c]];
                    }
                }
                du[[t, c]] = s;
            }
        }

        // --- input projection, RMSNorm, adapter (forward time) ----------------
        let mut dw = vec![0.0f64; 2 * di];
        let mut dxn = vec![0.0f64; dm];
        let mut dpl = Array2::<f64>::zeros((t_len, isz));
        for t in 0..t_len {
            for c in 0..di {
                dw[c] = du[[t, c]];
                dw[di + c] = d_res[[t, c]];
            }
            for m in 0..dm {
                let mut s = 0.0;
                for i in 0..2 * di {
                    s += w.w_in[[m, i]] * dw[i];
                    acc.w_in[[m, i]] += ca.xn[[t, m]] * dw[i];
                }
                dxn[m] = s;
            }

            // RMSNorm: xn = x' * r * g, r = (mean(x'^2) + eps)^{-1/2}.
            let r = ca.rms_inv[t];
            let mut q = 0.0;
            for m in 0..dm {
                q += dxn[m] * ca.x_adapted[[t, m]] * w.norm_gain[[0, m]];
            }
            let coef = q * r * r * r / dm as f64;
            for m in 0..dm {
                acc.norm_gain[[0, m]] += dxn[m] * ca.x_adapted[[t, m]] * r;
                d_xa[[t, m]] += dxn[m] * w.norm_gain[[0, m]] * r - coef * ca.x_adapted[[t, m]];
            }

            // Width adapter (or the identity when in == out).
            if self.has_adapter {
                for m in 0..dm {
                    acc.adapter_b[[0, m]] += d_xa[[t, m]];
                }
                for k in 0..isz {
                    let mut s = 0.0;
                    for m in 0..dm {
                        s += w.adapter_w[[k, m]] * d_xa[[t, m]];
                        acc.adapter_w[[k, m]] += recon_input[[t, k]] * d_xa[[t, m]];
                    }
                    dpl[[t, k]] = s;
                }
            } else {
                for k in 0..isz {
                    dpl[[t, k]] = d_xa[[t, k]];
                }
            }
        }

        if inv_sub_sampling_ratio > 1 {
            acc.scale(inv_sub_sampling_ratio as f64);
        }
        self.derivatives.add_assign(&acc);
        self.nb_of_seq_fed_backward += t_len as i64;

        dpl
    }

    /// Reverse-time BPTT: flip input/deltas, run the forward-order body, flip the
    /// returned deltas back. The caches are already time-reversed by
    /// [`Self::feed_forward_reverse`] and are consumed AS-IS (do NOT un-reverse).
    ///
    /// `output` is forwarded UNFLIPPED: [`Self::feed_backward`] discards it outright
    /// (its own doc says why), so flipping it only allocated a `T x width` copy for a
    /// body that never reads it. `input` IS flipped -- that one is genuinely consumed,
    /// by `reconcile_input`.
    pub fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();
        let deltas_rev = deltas.slice(ndarray::s![..;-1, ..]).to_owned();
        let dpl = self.feed_backward(
            &input_rev,
            output,
            &deltas_rev,
            inv_sub_sampling_ratio,
            last_layer,
        );
        dpl.slice(ndarray::s![..;-1, ..]).to_owned()
    }
}

/// The trait impl is a thin forward to the inherent methods, mirroring
/// `impl Layer for SlstmLayer`. It lives HERE rather than in `nn/network.rs` so the
/// phase's new-cell code stays inside `nn/cells/` (spec S1.4); `CellLayer`'s match
/// arms call the INHERENT methods by UFCS, so the two paths cannot diverge.
impl Layer for MambaLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        MambaLayer::feed_forward(self, input, output, last_layer);
    }
    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        MambaLayer::feed_forward_reverse(self, input, output, last_layer);
    }
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        MambaLayer::feed_backward(
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
        MambaLayer::feed_backward_reverse(
            self,
            input,
            output,
            deltas,
            inv_sub_sampling_ratio,
            last_layer,
        )
    }
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        MambaLayer::get_weights_derivatives(self, out);
    }
    fn reset_weights_derivatives(&mut self) {
        MambaLayer::reset_weights_derivatives(self);
    }
    fn ponderate_weights_derivatives(&mut self, factor: f64) {
        MambaLayer::ponderate_weights_derivatives(self, factor);
    }
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        MambaLayer::set_weights(self, flat)
    }
    fn get_weights(&self, out: &mut Vec<f64>) {
        MambaLayer::get_weights(self, out);
    }
    fn nb_of_weights(&self) -> usize {
        MambaLayer::nb_of_weights(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: usize = 4;
    const O: usize = 3;
    const T: usize = 9;
    const DS: usize = 2;
    const DC: usize = 3;
    const EX: usize = 2;

    /// Deterministic, non-degenerate weight fill (no RNG dependency).
    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.29 - 0.0131 * ((k % 37) as f64) + 0.0071 * ((k % 5) as f64))
            .collect()
    }

    fn seq(rows: usize, cols: usize, seed: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            seed + 0.17 * (r as f64) - 0.23 * (c as f64) + 0.011 * ((r * cols + c) as f64)
        })
    }

    fn cell(i: usize, o: usize, ds: usize, dc: usize, ex: usize) -> MambaLayer {
        MambaLayer::new(i, o, ds, dc, ex, 0)
    }

    fn loaded(i: usize, o: usize, ds: usize, dc: usize, ex: usize) -> MambaLayer {
        let mut c = cell(i, o, ds, dc, ex);
        c.set_weights(&weights(c.nb_of_weights()));
        c
    }

    fn forward(c: &mut MambaLayer, input: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((input.nrows(), c.output_size()));
        c.feed_forward(input, &mut out, false);
        out
    }

    fn derivs(c: &MambaLayer) -> Vec<f64> {
        let mut rows = Vec::new();
        c.get_weights_derivatives(&mut rows);
        rows.iter().map(|r| r[0]).collect()
    }

    /// Flat offset of each S3.2 block, computed independently of [`for_each_slot`].
    fn offsets(i: usize, o: usize, ds: usize, dc: usize, ex: usize) -> Vec<(&'static str, usize)> {
        let di = ex * o;
        let dr = o.div_ceil(16).max(1);
        let mut v = Vec::new();
        let mut p = 0usize;
        if i != o {
            for (n, len) in [("P", o * i), ("p", o)] {
                v.push((n, p));
                p += len;
            }
        }
        for (n, len) in [
            ("g", o),
            ("W_in", 2 * di * o),
            ("conv", di * dc),
            ("b_conv", di),
            ("W_x", (dr + 2 * ds) * di),
            ("W_dt", di * dr),
            ("b_dt", di),
            ("A_log", di * ds),
            ("D", di),
            ("W_out", o * di),
        ] {
            v.push((n, p));
            p += len;
        }
        v.push(("END", p));
        v
    }

    fn block(blocks: &[(&'static str, usize)], name: &str) -> (usize, usize) {
        let k = blocks.iter().position(|(n, _)| *n == name).unwrap();
        (blocks[k].1, blocks[k + 1].1)
    }

    // --- Hyperparameters (spec S3.3) ---------------------------------------

    #[test]
    fn dt_rank_zero_resolves_to_ceil_d_model_over_16() {
        for (o, want) in [
            (1usize, 1usize),
            (8, 1),
            (15, 1),
            (16, 1),
            (17, 2),
            (32, 2),
            (64, 4),
        ] {
            assert_eq!(MambaLayer::new(o, o, 4, 2, 1, 0).dt_rank(), want, "out={o}");
        }
        // An explicit rank is honoured verbatim.
        assert_eq!(MambaLayer::new(3, 3, 4, 2, 1, 7).dt_rank(), 7);
        // The >= 1 floors (see `new`'s doc): a zero would build empty blocks.
        let floored = MambaLayer::new(3, 3, 0, 0, 0, 0);
        assert_eq!(
            (floored.d_state(), floored.d_conv(), floored.d_inner()),
            (1, 1, 3)
        );
    }

    #[test]
    fn adapter_exists_exactly_when_the_widths_differ() {
        assert!(!cell(3, 3, DS, DC, EX).has_adapter());
        assert!(cell(4, 3, DS, DC, EX).has_adapter());
        assert!(cell(2, 3, DS, DC, EX).has_adapter());
    }

    // --- Weight seam (spec S3.2) -------------------------------------------

    #[test]
    fn nb_of_weights_is_the_spec_formula() {
        for (i, o, ds, dc, ex) in [
            (3usize, 3usize, 2usize, 2usize, 1usize),
            (3, 3, 4, 3, 2),
            (4, 6, 4, 4, 2),
            (6, 6, 8, 4, 2),
            (40, 64, 16, 4, 2),
        ] {
            let di = ex * o;
            let dr = o.div_ceil(16).max(1);
            let adapter = if i != o { o * i + o } else { 0 };
            let want = adapter
                + o
                + 2 * di * o
                + di * dc
                + di
                + (dr + 2 * ds) * di
                + di * dr
                + di
                + di * ds
                + di
                + o * di;
            assert_eq!(
                cell(i, o, ds, dc, ex).nb_of_weights(),
                want,
                "in={i} out={o} ds={ds} dc={dc} ex={ex}"
            );
        }
    }

    /// The `[P | p]` block exists ONLY when `in != out`, and its absence is exactly
    /// `out*in + out` fewer weights (spec S3.2).
    #[test]
    fn the_adapter_block_is_present_only_when_the_widths_differ() {
        let with = cell(4, 3, DS, DC, EX).nb_of_weights();
        let without = cell(3, 3, DS, DC, EX).nb_of_weights();
        assert_eq!(with - without, 3 * 4 + 3);
    }

    /// The flat layout is the S3.2 block order, each block ROW-major -- checked
    /// POSITIONALLY with a one-hot probe per position, not just by round trip (a round
    /// trip alone passes on any self-consistent permutation). Both width regimes.
    #[test]
    fn flat_layout_positions_are_the_spec_order() {
        for (i, o, ds, dc, ex) in [(3usize, 3usize, 2usize, 2usize, 1usize), (4, 3, 2, 3, 2)] {
            let mut c = cell(i, o, ds, dc, ex);
            let n = c.nb_of_weights();
            let di = ex * o;
            let dr = o.div_ceil(16).max(1);
            let blocks = offsets(i, o, ds, dc, ex);
            assert_eq!(
                blocks.last().unwrap().1,
                n,
                "block offsets disagree with nb"
            );

            let mut probe = |base: usize,
                             row: usize,
                             col: usize,
                             get: &dyn Fn(&MambaWeights) -> f64,
                             what: &str| {
                let mut flat = vec![0.0; n];
                flat[base] = 1.0;
                c.set_weights(&flat);
                assert_eq!(
                    get(&c.weights),
                    1.0,
                    "{what} [{row},{col}] misplaced (in={i} out={o})"
                );
            };

            if i != o {
                // P (out x in) row-major -> adapter_w[[k, j]] == P[j, k].
                let (lo, _) = block(&blocks, "P");
                for j in 0..o {
                    for k in 0..i {
                        probe(lo + j * i + k, j, k, &|w| w.adapter_w[[k, j]], "P");
                    }
                }
                let (lo, _) = block(&blocks, "p");
                for j in 0..o {
                    probe(lo + j, 0, j, &|w| w.adapter_b[[0, j]], "p");
                }
            }
            let (lo, _) = block(&blocks, "g");
            for m in 0..o {
                probe(lo + m, 0, m, &|w| w.norm_gain[[0, m]], "g");
            }
            // W_in (2 d_inner x d_model) row-major -> w_in[[m, r]] == W_in[r, m].
            let (lo, _) = block(&blocks, "W_in");
            for r in 0..2 * di {
                for m in 0..o {
                    probe(lo + r * o + m, r, m, &|w| w.w_in[[m, r]], "W_in");
                }
            }
            // conv (d_inner x d_conv) row-major, NOT transposed.
            let (lo, _) = block(&blocks, "conv");
            for ch in 0..di {
                for k in 0..dc {
                    probe(lo + ch * dc + k, ch, k, &|w| w.conv_w[[ch, k]], "conv");
                }
            }
            let (lo, _) = block(&blocks, "b_conv");
            for ch in 0..di {
                probe(lo + ch, 0, ch, &|w| w.conv_b[[0, ch]], "b_conv");
            }
            // W_x ((dt_rank + 2 d_state) x d_inner) row-major -> w_x[[c, r]].
            let (lo, _) = block(&blocks, "W_x");
            for r in 0..dr + 2 * ds {
                for ch in 0..di {
                    probe(lo + r * di + ch, r, ch, &|w| w.w_x[[ch, r]], "W_x");
                }
            }
            // W_dt (d_inner x dt_rank) row-major -> w_dt[[r, c]] == W_dt[c, r].
            let (lo, _) = block(&blocks, "W_dt");
            for ch in 0..di {
                for r in 0..dr {
                    probe(lo + ch * dr + r, ch, r, &|w| w.w_dt[[r, ch]], "W_dt");
                }
            }
            let (lo, _) = block(&blocks, "b_dt");
            for ch in 0..di {
                probe(lo + ch, 0, ch, &|w| w.b_dt[[0, ch]], "b_dt");
            }
            // A_log (d_inner x d_state) row-major, NOT transposed.
            let (lo, _) = block(&blocks, "A_log");
            for ch in 0..di {
                for st in 0..ds {
                    probe(lo + ch * ds + st, ch, st, &|w| w.a_log[[ch, st]], "A_log");
                }
            }
            let (lo, _) = block(&blocks, "D");
            for ch in 0..di {
                probe(lo + ch, 0, ch, &|w| w.d_skip[[0, ch]], "D");
            }
            // W_out (d_model x d_inner) row-major -> w_out[[c, m]] == W_out[m, c].
            let (lo, _) = block(&blocks, "W_out");
            for m in 0..o {
                for ch in 0..di {
                    probe(lo + m * di + ch, m, ch, &|w| w.w_out[[ch, m]], "W_out");
                }
            }
        }
    }

    #[test]
    fn weight_seam_round_trips_and_returns_the_tail() {
        for (i, o) in [(I, O), (O, O)] {
            let mut c = cell(i, o, DS, DC, EX);
            let nb = c.nb_of_weights();
            let w = weights(nb + 5);
            let tail = c.set_weights(&w);
            assert_eq!(tail.len(), 5);
            assert_eq!(tail, &w[nb..]);

            let mut back = Vec::new();
            c.get_weights(&mut back);
            assert_eq!(back, w[..nb].to_vec());
        }
    }

    // --- Forward (spec S3.1) ------------------------------------------------

    #[test]
    fn forward_shapes_and_caches_track_the_sequence_length() {
        for (i, o) in [(I, O), (O, O)] {
            for t in [1usize, 2, 5, T] {
                let mut c = loaded(i, o, DS, DC, EX);
                let di = c.d_inner();
                let out = forward(&mut c, &seq(t, i, 0.4));
                assert_eq!(out.dim(), (t, o), "in={i} t={t}");
                assert_eq!(c.delta().dim(), (t, di));
                assert_eq!(c.conv_activations().dim(), (t, di));
                assert_eq!(c.abar().dim(), (t, di * DS));
                assert_eq!(c.hidden_states().dim(), (t, di * DS));
                assert_eq!(c.rms_inv().len(), t);
                assert!(out.iter().all(|v| v.is_finite()), "in={i} t={t}");
            }
        }
    }

    /// The forward is a pure function of (weights, input): run twice, bit-identical
    /// (no state leaks across calls -- `h_{-1} = 0` and the conv left-padding are
    /// re-seeded per call).
    #[test]
    fn forward_is_run_twice_bit_identical_and_stateless() {
        let mut c = loaded(I, O, DS, DC, EX);
        let input = seq(T, I, 0.4);
        let first = forward(&mut c, &input);
        let (h1, a1, d1) = (
            c.hidden_states().clone(),
            c.abar().clone(),
            c.delta().clone(),
        );
        let second = forward(&mut c, &input);
        assert_eq!(first, second);
        assert_eq!(&h1, c.hidden_states());
        assert_eq!(&a1, c.abar());
        assert_eq!(&d1, c.delta());

        let mut fresh = loaded(I, O, DS, DC, EX);
        assert_eq!(forward(&mut fresh, &input), first);
    }

    /// STABILITY BY CONSTRUCTION (spec S3.1): `A = -exp(A_log) < 0` and
    /// `Delta = softplus(.) > 0`, so every `Abar = exp(Delta A)` lands strictly inside
    /// `(0, 1)` -- checked over a pseudo-random weight sweep, on every channel, state
    /// and timestep, in both width regimes.
    #[test]
    fn abar_stays_strictly_inside_the_unit_interval() {
        // A cheap deterministic LCG: the exact tree has no `rand` dependency.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut draw = |half: f64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64) * (1.0 / 9007199254740992.0);
            (2.0 * u - 1.0) * half
        };
        for (i, o) in [(I, O), (O, O)] {
            for _ in 0..12 {
                let mut c = cell(i, o, DS, DC, EX);
                let nb = c.nb_of_weights();
                let w: Vec<f64> = (0..nb).map(|_| draw(1.0)).collect();
                c.set_weights(&w);
                let input = Array2::from_shape_fn((T, i), |_| draw(2.0));
                let _ = forward(&mut c, &input);
                for v in c.abar().iter() {
                    assert!(*v > 0.0 && *v < 1.0, "Abar {v:e} left (0,1)");
                }
                for v in c.delta().iter() {
                    assert!(*v > 0.0, "Delta {v:e} is not positive");
                }
            }
        }
    }

    /// HAND-COMPUTED single step at the degenerate geometry `d_model = d_state =
    /// d_conv = expand = 1` -- every operator of S3.1 exercised exactly once
    /// (RMSNorm, both projections, the conv, both SiLUs, softplus, `-exp`, the
    /// recurrence, the residual), each written out with literal constants. The test
    /// re-derives from the FORMULAS, not from the module's private helpers, so a
    /// wrong composition order cannot hide.
    #[test]
    fn single_step_matches_the_hand_computed_value() {
        let mut c = MambaLayer::new(1, 1, 1, 1, 1, 0);
        assert_eq!(c.dt_rank(), 1);
        // Layout: g | W_in(u) W_in(res) | conv | b_conv | W_x(dt~) W_x(B) W_x(C)
        //         | W_dt | b_dt | A_log | D | W_out
        let flat = vec![
            2.0, // g
            0.5, -0.25, // W_in rows [u | res]
            1.5,   // conv
            0.1,   // b_conv
            0.4, -0.6, 0.8,  // W_x rows [dt~ | B | C]
            0.3,  // W_dt
            -0.2, // b_dt
            0.0,  // A_log  -> A = -exp(0) = -1 exactly
            0.7,  // D
            1.25, // W_out
        ];
        assert_eq!(flat.len(), c.nb_of_weights());
        c.set_weights(&flat);

        let x = 0.5f64;
        let out = forward(&mut c, &Array2::from_shape_vec((1, 1), vec![x]).unwrap());

        let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
        let rms_inv = 1.0 / (x * x / 1.0 + RMS_EPS).sqrt();
        let xn = x * rms_inv * 2.0;
        let u = 0.5 * xn;
        let res = -0.25 * xn;
        // d_conv = 1: the single tap IS the current sample, no padding involved.
        let pc = 1.5 * u + 0.1;
        let uc = pc * sigmoid(pc);
        let (dt_tilde, b_t, c_t) = (0.4 * uc, -0.6 * uc, 0.8 * uc);
        let dtp = 0.3 * dt_tilde + -0.2;
        assert!(
            dtp < 0.0,
            "the chosen constants must exercise softplus' x <= 0 arm"
        );
        let delta = dtp.exp().ln_1p();
        let a = -1.0f64;
        let abar = (delta * a).exp();
        // h_{-1} = 0, so Abar multiplies nothing at t = 0.
        let h = delta * uc * b_t;
        let y = c_t * h + 0.7 * uc;
        let v = y * (res * sigmoid(res));
        let want = 1.25 * v + x;

        assert_eq!(c.rms_inv()[0], rms_inv);
        assert_eq!(c.conv_activations()[[0, 0]], uc);
        assert_eq!(c.delta()[[0, 0]], delta);
        assert_eq!(c.abar()[[0, 0]], abar);
        assert_eq!(c.hidden_states()[[0, 0]], h);
        assert_eq!(out[[0, 0]], want);
        // Non-vacuity: the block really moved the input, it is not just the residual.
        assert!(
            (out[[0, 0]] - x).abs() > 1e-3,
            "the block contributed nothing"
        );
    }

    /// THE CAUSAL CONV TAP ORDER is a CONVENTION, and this pins which one: tap
    /// `d_conv - 1` sees the CURRENT sample and tap `0` the oldest (the left-padded
    /// `conv1d` convention the module doc states, and the one the backward's flipped
    /// correlation mirrors).
    ///
    /// Worth its own test because NOTHING else can see it. The FD tier catches a
    /// forward-only or backward-only flip (measured: relative error 1.39 -- they stop
    /// being each other's adjoint), but a CONSISTENTLY mirrored pair is a
    /// different-yet-self-consistent model that agrees with its own gradient
    /// perfectly. And the hand-computed single-step test runs at `d_conv = 1`, where
    /// the two orders coincide. So: a `[1, 0]` kernel must be a pure ONE-STEP DELAY,
    /// and `[0, 1]` the identity.
    #[test]
    fn the_causal_conv_kernel_puts_the_current_sample_in_the_last_tap() {
        let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
        let x = Array2::from_shape_vec((2, 1), vec![0.75, -0.5]).unwrap();
        // Everything past the conv is zeroed, so `conv_activations` is the whole
        // observable: g | W_in[u|res] | conv[oldest|current] | b_conv | W_x(3)
        //             | W_dt | b_dt | A_log | D | W_out
        let run = |taps: [f64; 2]| {
            let mut c = MambaLayer::new(1, 1, 1, 2, 1, 0);
            let flat = vec![
                1.0, // g
                1.0, 0.0, // W_in [u | res]
                taps[0], taps[1], // conv
                0.0,     // b_conv
                0.0, 0.0, 0.0, // W_x
                0.0, 0.0, // W_dt | b_dt
                0.0, // A_log
                0.0, // D
                0.0, // W_out
            ];
            assert_eq!(flat.len(), c.nb_of_weights());
            c.set_weights(&flat);
            let _ = forward(&mut c, &x);
            let u: Vec<f64> = (0..2).map(|t| x[[t, 0]] * c.rms_inv()[t]).collect();
            let uc: Vec<f64> = (0..2).map(|t| c.conv_activations()[[t, 0]]).collect();
            (u, uc)
        };

        // [1, 0]: only the OLDEST tap is live -> a one-step delay, and row 0 sees
        // nothing but the causal left-padding.
        let (u, uc) = run([1.0, 0.0]);
        assert_eq!(uc[0], 0.0, "tap 0 must not reach the current sample");
        assert_eq!(
            uc[1],
            u[0] * sigmoid(u[0]),
            "tap 0 must be the ONE-STEP-BACK sample"
        );
        // Non-vacuity: the two samples are genuinely different, so the delay is
        // observable rather than a coincidence.
        assert_ne!(u[0], u[1]);

        // [0, 1]: only the LAST tap is live -> the identity.
        let (u, uc) = run([0.0, 1.0]);
        for t in 0..2 {
            assert_eq!(uc[t], u[t] * sigmoid(u[t]), "row {t} is not the identity");
        }
    }

    /// The residual adds `x'` -- the ADAPTED input -- not the raw `x_t` (spec S3.1).
    /// Zeroing `W_out` kills the whole SSM branch, so the output must be EXACTLY
    /// `P x + p`.
    #[test]
    fn the_residual_adds_the_adapted_input_not_the_raw_input() {
        let (i, o) = (I, O);
        let mut c = cell(i, o, DS, DC, EX);
        let blocks = offsets(i, o, DS, DC, EX);
        let mut w = weights(c.nb_of_weights());
        let (lo, hi) = block(&blocks, "W_out");
        w[lo..hi].fill(0.0);
        c.set_weights(&w);

        let input = seq(T, i, 0.4);
        let out = forward(&mut c, &input);
        let (p_lo, _) = block(&blocks, "P");
        let (b_lo, _) = block(&blocks, "p");
        for t in 0..T {
            for j in 0..o {
                // Product sum FIRST, bias second -- the forward's own order
                // (`matmul_seq` then `+= adapter_b`); the reverse order rounds
                // differently and this asserts bit equality.
                let mut want = 0.0;
                for k in 0..i {
                    want += w[p_lo + j * i + k] * input[[t, k]];
                }
                want += w[b_lo + j];
                assert_eq!(out[[t, j]], want, "row {t} col {j}");
            }
        }
        // Non-vacuity: the raw input is NOT what came out (widths even differ).
        assert_ne!(out.dim().1, input.dim().1);
    }

    /// [`softplus`]'s branch is LOAD-BEARING: the naive `ln(1 + exp(x))` overflows to
    /// `inf` above `~709.78`, and a large `dt` pre-activation is exactly what an
    /// untrained `b_dt` produces. Both arms agree with the closed form where both are
    /// finite, and the positive arm degrades to `x` instead of blowing up.
    #[test]
    fn softplus_guard_survives_the_naive_overflow() {
        // Non-vacuity: this is what the unguarded form would exponentiate.
        assert!(800.0_f64.exp().is_infinite());
        assert!((1.0 + 800.0_f64.exp()).ln().is_infinite());

        assert_eq!(softplus(800.0), 800.0);
        assert!(softplus(-800.0) >= 0.0 && softplus(-800.0) < 1e-300);
        // Agreement with the closed form is to a TOLERANCE, not bit-exact, and the
        // closed form is the inaccurate side: at x = -5 it evaluates `ln(1.00674)`,
        // cancelling ~2 decimal digits, where `ln_1p` keeps them (measured relative
        // gap 1.5e-14, pinned at 1e-12).
        for x in [-5.0f64, -1.0, -1e-9, 0.0, 1e-9, 1.0, 5.0, 30.0] {
            let want = (1.0 + x.exp()).ln();
            assert!(
                (softplus(x) - want).abs() <= 1e-12 * want.abs().max(1e-9),
                "x={x}: {} vs {want}",
                softplus(x)
            );
        }
        assert!(
            softplus(0.0) > 0.0,
            "softplus is strictly positive everywhere"
        );

        // And the same guard reaches Delta through the layer: a huge b_dt stays finite.
        let mut c = MambaLayer::new(1, 1, 1, 1, 1, 0);
        let mut flat = vec![0.1; c.nb_of_weights()];
        flat[9] = 900.0; // b_dt
        c.set_weights(&flat);
        let out = forward(
            &mut c,
            &Array2::from_shape_vec((2, 1), vec![0.5, -0.5]).unwrap(),
        );
        assert!(c.delta().iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `feed_forward_reverse` is the forward on flipped rows (and leaves the caches
    /// reversed, the convention the reverse backward depends on). Mamba is CAUSAL, so
    /// this is a genuinely anti-causal pass, not a symmetry.
    #[test]
    fn reverse_forward_is_the_flipped_forward() {
        let input = seq(T, I, 0.4);
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();

        let mut a = loaded(I, O, DS, DC, EX);
        let mut out_rev = Array2::<f64>::zeros((T, O));
        a.feed_forward_reverse(&input, &mut out_rev, false);

        let mut b = loaded(I, O, DS, DC, EX);
        let plain_on_flipped = forward(&mut b, &input_rev);

        assert_eq!(
            out_rev,
            plain_on_flipped.slice(ndarray::s![..;-1, ..]).to_owned()
        );
        assert_eq!(a.hidden_states(), b.hidden_states());
        // Non-vacuity: a causal cell is NOT reverse-invariant.
        let mut d = loaded(I, O, DS, DC, EX);
        assert_ne!(out_rev, forward(&mut d, &input));
    }

    // --- Backward (spec S3.5) ----------------------------------------------

    #[test]
    fn derivatives_accumulate_across_calls_and_reset_clears() {
        let mut c = loaded(I, O, DS, DC, EX);
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);

        let out = forward(&mut c, &input);
        c.reset_weights_derivatives();
        let _ = c.feed_backward(&input, &out, &deltas, 1, false);
        let mut once = Vec::new();
        c.get_weights_derivatives(&mut once);
        assert!(
            once.iter().any(|r| r[0] != 0.0),
            "no derivative accumulated -- the test is vacuous"
        );
        assert!(once.iter().all(|r| r[1] == T as f64));

        let _ = c.feed_backward(&input, &out, &deltas, 1, false);
        let mut twice = Vec::new();
        c.get_weights_derivatives(&mut twice);
        for k in 0..once.len() {
            assert_eq!(twice[k][0], once[k][0] + once[k][0], "row {k} did not sum");
            assert_eq!(twice[k][1], 2.0 * T as f64);
        }

        c.reset_weights_derivatives();
        let mut zeroed = Vec::new();
        c.get_weights_derivatives(&mut zeroed);
        assert!(zeroed.iter().all(|r| r[0] == 0.0 && r[1] == 0.0));
    }

    /// STRUCTURAL ZERO 1 (module doc): at `T = 1` the WHOLE `A_log` block has an
    /// EXACTLY zero gradient -- `dAbar_t = dh_t . h_{t-1}` and `h_{-1} = 0`, so
    /// `Abar_0` multiplies nothing and cannot move the loss. Pinned as exact equality
    /// against a non-vacuous contrast at `T = 2`.
    #[test]
    fn a_log_gradient_is_exactly_zero_at_t1() {
        let (i, o, ds, dc, ex) = (3usize, 3usize, 2usize, 2usize, 1usize);
        let blocks = offsets(i, o, ds, dc, ex);
        let (lo, hi) = block(&blocks, "A_log");
        assert_eq!(hi - lo, ex * o * ds);

        for (t, want_zero) in [(1usize, true), (2, false)] {
            let mut c = loaded(i, o, ds, dc, ex);
            let input = seq(t, i, 0.4);
            let out = forward(&mut c, &input);
            c.reset_weights_derivatives();
            let _ = c.feed_backward(&input, &out, &seq(t, o, -0.2), 1, false);
            let d = derivs(&c);
            if want_zero {
                for (offset, v) in d[lo..hi].iter().enumerate() {
                    assert_eq!(*v, 0.0, "A_log[{offset}] is not exactly zero at T=1");
                }
                // Contrast: the rest of the pack is very much alive.
                assert!(d.iter().map(|v| v.abs()).fold(0.0f64, f64::max) > 1e-6);
            } else {
                assert!(
                    (lo..hi).any(|k| d[k] != 0.0),
                    "A_log stayed dead at T=2 -- the T=1 zero is not structural"
                );
            }
        }
    }

    /// STRUCTURAL ZERO 2 (module doc): the causal conv is LEFT-padded with
    /// `d_conv - 1` zeros, so tap `k` first meets a real sample at
    /// `t = d_conv - 1 - k`. With `T < d_conv` the oldest `d_conv - T` taps per
    /// channel only ever multiply padding, and their gradient is EXACTLY zero.
    #[test]
    fn old_conv_taps_are_gradient_dead_below_the_kernel_length() {
        let (i, o, ds, dc, ex) = (3usize, 3usize, 2usize, 4usize, 1usize);
        let blocks = offsets(i, o, ds, dc, ex);
        let (lo, _) = block(&blocks, "conv");
        let di = ex * o;

        for t in [1usize, 2, 3, 4, 5] {
            let mut c = loaded(i, o, ds, dc, ex);
            let input = seq(t, i, 0.4);
            let out = forward(&mut c, &input);
            c.reset_weights_derivatives();
            let _ = c.feed_backward(&input, &out, &seq(t, o, -0.2), 1, false);
            let d = derivs(&c);
            let dead = dc.saturating_sub(t);
            for ch in 0..di {
                for k in 0..dc {
                    let v = d[lo + ch * dc + k];
                    if k < dead {
                        assert_eq!(v, 0.0, "T={t}: tap [{ch},{k}] must be padding-only");
                    } else {
                        assert_ne!(v, 0.0, "T={t}: tap [{ch},{k}] sees real samples");
                    }
                }
            }
        }
    }

    #[test]
    fn ponderation_scales_col0_only_and_issr_scales_the_blocks() {
        let mut c = loaded(I, O, DS, DC, EX);
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);
        let out = forward(&mut c, &input);

        c.reset_weights_derivatives();
        let _ = c.feed_backward(&input, &out, &deltas, 1, false);
        let mut base = Vec::new();
        c.get_weights_derivatives(&mut base);
        c.ponderate_weights_derivatives(0.25);
        let mut pond = Vec::new();
        c.get_weights_derivatives(&mut pond);
        for k in 0..base.len() {
            assert_eq!(pond[k][0], base[k][0] * 0.25);
            assert_eq!(pond[k][1], base[k][1], "the frame count must not scale");
        }

        let mut scaled = loaded(I, O, DS, DC, EX);
        let out2 = forward(&mut scaled, &input);
        scaled.reset_weights_derivatives();
        let _ = scaled.feed_backward(&input, &out2, &deltas, 3, false);
        let mut rows = Vec::new();
        scaled.get_weights_derivatives(&mut rows);
        for k in 0..base.len() {
            assert_eq!(rows[k][0], base[k][0] * 3.0, "row {k}");
        }
    }

    #[test]
    fn reverse_backward_is_the_flipped_backward() {
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);

        let mut a = loaded(I, O, DS, DC, EX);
        let mut out_rev = Array2::<f64>::zeros((T, O));
        a.feed_forward_reverse(&input, &mut out_rev, false);
        let dpl_rev = a.feed_backward_reverse(&input, &out_rev, &deltas, 1, false);
        let mut da = Vec::new();
        a.get_weights_derivatives(&mut da);

        let mut b = loaded(I, O, DS, DC, EX);
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

    /// The propagated deltas are `T x in` (the layer's own width) whatever the caller
    /// feeds -- the width-tolerance contract the forward and backward share through
    /// `reconcile_input`. Both width regimes, since the adapter sits behind it.
    #[test]
    fn width_tolerance_is_shared_by_both_passes() {
        for o in [O, I] {
            let narrow = seq(T, 1, 0.4);
            let wide = seq(T, I + 2, 0.4);
            for input in [&narrow, &wide] {
                let mut c = loaded(I, o, DS, DC, EX);
                let out = forward(&mut c, input);
                assert_eq!(out.dim(), (T, o));
                let dpl = c.feed_backward(input, &out, &seq(T, o, -0.2), 1, false);
                assert_eq!(dpl.dim(), (T, I));
            }

            // Cropping is exactly "ignore the extra columns".
            let mut a = loaded(I, o, DS, DC, EX);
            let mut b = loaded(I, o, DS, DC, EX);
            assert_eq!(
                forward(&mut a, &wide),
                forward(&mut b, &wide.slice(ndarray::s![.., ..I]).to_owned())
            );
        }
    }
}
