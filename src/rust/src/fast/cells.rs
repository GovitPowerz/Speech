//! `fast::cells` -- the f32 CAUSAL cell twins (Phase 9 Task 6, spec S4.1-S4.3).
//! FORWARD ONLY, CAUSAL ONLY.
//!
//! Port-only -- NO legacy source. The parity ORACLE for everything here is the exact
//! f64 tree, transcribed op-for-op:
//!
//! - [`FastSlstm`] mirrors `nn::cells::slstm::SlstmLayer::feed_forward` (spec S2.1,
//!   layout S2.2): the `[i|f|o|z]` gate blocks, the running-max stabilizer `m`, the
//!   twin `(c, n)` recurrence, `h = sigmoid(o~) * (c/n)`.
//! - [`FastMamba`] mirrors `nn::cells::mamba::MambaLayer::feed_forward` (spec S3.1,
//!   layout S3.2): the width adapter, RMSNorm (`eps = 1e-5`), the `[u | res]` input
//!   projection, the LEFT-PADDED depthwise causal conv, the per-timestep selectivity
//!   (`dt~`/`B`/`C`), the overflow-guarded `softplus`, the elementwise SSM recurrence,
//!   the SiLU gate, and the `x'` residual.
//! - [`FastCausalNet`] mirrors `nn::blstm::BlstmNetwork::feed_forward` under
//!   [`crate::nn::blstm::Direction::Forward`]: ONE stack (no reverse half, no HCAT),
//!   `Network::drive`'s per-layer sub-sampling chain, then the SAME f32 dense/softmax
//!   output MLP the phase-7 [`super::nn::FastBlstm`] uses -- fed `hidden` columns
//!   rather than `2*hidden` (`Network::feed_forward_double`'s forward-only branch).
//!
//! WHAT LANDED (spec S4.1): an explicit `step(x_t, &mut State)` kernel per cell --
//! carried state `(h, c, n, m)` for sLSTM, `(conv ring, h)` for Mamba -- plus a
//! whole-sequence `feed_forward` wrapper that LOOPS that same step. ONE kernel, two
//! drivers: the offline fast path here and the Task-7 causal streaming session run
//! the SAME `step`, so offline-vs-streamed bit-identity holds BY CONSTRUCTION (the
//! phase-7/8 shared-kernel precedent). Weights narrow `f64 -> f32` ONCE at
//! `from_flat`, post-pack (`flat[i] as f32`), exactly like `FastBlstm::from_flat`.
//!
//! DELIBERATE f32 DIVERGENCES (spec S4, documented in module docs + `RESULTS.md`,
//! NEVER `IMPROVEMENTS.md` -- the phase-7 rule):
//!
//! 1. f32 arithmetic throughout, against the exact tree's f64. The op ORDER is
//!    preserved element-for-element (every reduction is an ascending loop over a
//!    CONTIGUOUS slice, matching `matmul_seq`'s ascending contract), so the only
//!    divergence is rounding width -- measured, then pinned with headroom by
//!    `tests/phase9_fast_parity.rs` and the inline `*_matches_the_exact_cell_*` legs.
//! 2. NO batched (faer) projection. `FastBlstm` batches its input projection through
//!    `faer_project` because it never has to expose a per-frame kernel; the causal
//!    twins DO (streaming), and the brief's bit-identity contract is "the wrapper is
//!    the step loop". A batched projection would make the offline and streaming
//!    numbers differ in the last f32 ULP, which is exactly the class of drift the
//!    phase-8 gate exists to forbid. So every projection here is a per-step dot over
//!    a contiguous weight row. The dense output MLP runs the shared
//!    `super::nn::DenseRowChain` -- still the faer kernel, but driven ONE ROW at a
//!    time, NOT the batched `dense_net_forward` the phase-7 [`super::nn::FastBlstm`]
//!    keeps. (An earlier revision of this line said the dense stage was "a per-row
//!    map, so batching it is state-free"; Task 7 MEASURED that FALSE at wide layers --
//!    see `DenseRowChain`'s docs and [`FastCausalNet::feed_forward`] below.)
//! 3. The `exp` saturation guards are the f32 analogues via
//!    [`super::nn::logistic_f32`] (`ln(f32::MAX) ~ 88.72`), not the exact path's f64
//!    `~709.78` -- the same reasoning as `fast::nn`.
//! 4. THE `t == 0` RECURRENCE SKIP IS NOT REPRODUCED AS A BRANCH. The exact sLSTM
//!    forward guards its recurrent term with `if t > 0`; `step` ALWAYS applies it,
//!    relying on the zero-initialized `h` in the state to make it inert. That is
//!    (a) numerically identical for every value except a `-0.0` accumulator (adding
//!    `+0.0` to `-0.0` yields `+0.0`; no reachable config produces a `-0.0`
//!    pre-activation), and (b) REQUIRED by the streaming contract: a session resumed
//!    mid-sequence has a NON-zero carried `h`, and a `t == 0` skip would silently
//!    drop the recurrence at every chunk boundary. The same argument covers Mamba's
//!    `h_{-1} = 0` and its left-padded conv taps (the ring is zero-initialized, and
//!    `acc += w * 0.0` on a `+0.0`-seeded accumulator is exactly the exact path's
//!    "skip the out-of-range tap"). `split_state_*` pins the consequence.
//!
//! SCOPE. Causal (`Direction::Forward`) only, forward only, no MLP mode, no backward,
//! no trainer. Bidirectional new-cell inference stays on the exact tree this phase
//! (spec S4.2); the fast SAD driver's dispatch + typed bails are in
//! [`super::driver`].

use anyhow::{Result, bail};

use crate::config::NnetSpec;
use crate::nn::blstm::{CellType, MambaParams};

use super::nn::{
    DenseRowChain, FastDenseLayer, FastMatrix, Scratch, copy_view_into, ensure_len, logistic_f32,
    sub_sample_into,
};

// ---------------------------------------------------------------------------
// f32 activation transcriptions (the exact cells' private helpers, f32).
// ---------------------------------------------------------------------------

/// `nn::cells::slstm::M_INIT` in f32: the documented finite stand-in for `-inf`.
///
/// The absorption argument survives the narrowing with room to spare: f32 carries ~7
/// significant digits, so `f~ + M_INIT_F32 == M_INIT_F32` exactly for any
/// `|f~| < ~1e23` (the f64 bound is `~1e14` for the same reason, and both are
/// absurdly far above any reachable pre-activation). Hence `m_0 = i~_0`,
/// `i'_0 = exp(0) = 1`, and `f'_0 = exp(-1e30) = 0` EXACTLY -- the official xLSTM
/// `m_1 = i~_1` convention, reached with no `t == 0` branch.
pub const M_INIT_F32: f32 = -1e30;

/// `nn::cells::mamba::RMS_EPS` in f32 (the block-defining `1e-5`, not tunable).
pub const RMS_EPS_F32: f32 = 1e-5;

/// `nn::cells::mamba::softplus` in f32: `ln(1 + e^x)` evaluated in the branch that
/// cannot overflow. LOAD-BEARING, not cosmetic -- and MORE so in f32, whose `exp`
/// overflows at `~88.7` rather than `~709.8`, so an untrained `b_dt` reaches the
/// guarded region an order of magnitude sooner.
#[inline]
fn softplus_f32(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// `nn::cells::mamba::silu` in f32: `x * sigmoid(x)`, on the guarded
/// [`logistic_f32`].
#[inline]
fn silu_f32(x: f32) -> f32 {
    x * logistic_f32(x)
}

/// Ascending dot over two contiguous `n`-element slices -- the f32 stand-in for one
/// `matmul_seq` output cell (`nn/layers.rs:25-40`: `acc = 0; for kk in 0..k { acc +=
/// a*b }`). Every projection in this module routes through it, so the accumulation
/// order is one fact rather than a dozen hand-written loops.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for k in 0..a.len() {
        acc += a[k] * b[k];
    }
    acc
}

// ---------------------------------------------------------------------------
// sLSTM (spec S2.1 forward, S2.2 layout).
// ---------------------------------------------------------------------------

/// The sLSTM carried state (spec S4.1). Four `output_size`-long vectors:
/// `h` (the previous output, the recurrence's input), `c` (numerator), `n`
/// (normalizer) and `m` (the running-max stabilizer, seeded [`M_INIT_F32`]).
///
/// Construction is CHEAP (four small allocations, no weight touch) and reset is
/// EXPLICIT: a session takes a fresh [`FastSlstm::state`]. There is no implicit
/// reset anywhere -- carrying a stale state across sequences is a caller bug the
/// `split_state_*` legs exist to make visible.
#[derive(Clone, Debug, PartialEq)]
pub struct SlstmState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
    pub n: Vec<f32>,
    pub m: Vec<f32>,
}

/// f32 forward-only sLSTM cell. Weights are stored as three flat SoA buffers whose
/// per-unit rows are CONTIGUOUS, which is what lets every gate projection be one
/// ascending [`dot_f32`] -- the same accumulation order the exact `matmul_seq` uses:
///
/// - `rec[(a*O + j)*O + k] == R_a[j, k]`
/// - `inp[(a*O + j)*I + k] == W_a[j, k]`
/// - `bias[a*O + j] == b_a[j]`
///
/// Those are literally the spec S2.2 flat pack re-sliced (per gate: the whole `R_a`
/// block, then the whole `W_a` block, then `b_a`, each row-major), so `from_flat` is
/// three contiguous narrowing copies per gate and NOT a re-derivation of the layout.
#[derive(Clone)]
pub struct FastSlstm {
    input_size: usize,
    output_size: usize,
    rec: Vec<f32>,  // 4*O*O
    inp: Vec<f32>,  // 4*O*I
    bias: Vec<f32>, // 4*O
}

impl FastSlstm {
    /// `4 * out * (out + in + 1)` (spec S2.2) -- the mirror of
    /// `SlstmLayer::nb_of_weights`, pinned against it by
    /// `slstm_weight_count_matches_the_exact_cell`.
    pub fn weight_count(input_size: usize, output_size: usize) -> usize {
        4 * output_size * (output_size + input_size + 1)
    }

    /// Build from the head of the f64 flat pack (adim already applied upstream; new
    /// cells never carry adim anyway, spec S1.3), narrowing `f64 -> f32` ONCE.
    ///
    /// Consumes exactly [`Self::weight_count`] elements and PANICS on a shorter
    /// slice: the length check belongs to the net-level builder
    /// ([`FastCausalNet::from_flat`], which returns a typed `Err`), and this
    /// signature is pinned by spec S4.1 for the Task-7 streaming consumer.
    pub fn from_flat(flat: &[f64], input_size: usize, output_size: usize) -> FastSlstm {
        let (o, i) = (output_size, input_size);
        let block = o * o + o * i + o;
        let mut rec = vec![0.0_f32; 4 * o * o];
        let mut inp = vec![0.0_f32; 4 * o * i];
        let mut bias = vec![0.0_f32; 4 * o];
        for a in 0..4 {
            let base = a * block;
            for p in 0..o * o {
                rec[a * o * o + p] = flat[base + p] as f32;
            }
            for p in 0..o * i {
                inp[a * o * i + p] = flat[base + o * o + p] as f32;
            }
            for p in 0..o {
                bias[a * o + p] = flat[base + o * o + o * i + p] as f32;
            }
        }
        FastSlstm {
            input_size: i,
            output_size: o,
            rec,
            inp,
            bias,
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// A fresh zero state with `m = ` [`M_INIT_F32`] (spec S2.4/S4.1).
    pub fn state(&self) -> SlstmState {
        let o = self.output_size;
        SlstmState {
            h: vec![0.0; o],
            c: vec![0.0; o],
            n: vec![0.0; o],
            m: vec![M_INIT_F32; o],
        }
    }

    /// ONE timestep (spec S2.1), the streaming kernel. `x` is this frame's layer
    /// input (width tolerance: a longer `x` is CROPPED to `input_size`, a shorter one
    /// is implicitly zero-padded by summing fewer terms -- the exact
    /// `SlstmLayer::reconcile_input` contract); `out` is `output_size` long and holds
    /// `h_t` on return; `s` is advanced in place.
    ///
    /// Op order, matching the exact forward line for line: input projection
    /// (ascending `k`), `+ bias`, `+ recurrence` (ascending `k`); then per unit
    /// `m = max(f~ + m_prev, i~)` written as an explicit comparison (NOT `f32::max`,
    /// whose NaN handling differs from a ternary), the two stabilized exps, `tanh`,
    /// the logistic output gate, the twin `(c, n)` update, and `h = o * (c/n)`.
    ///
    /// `needless_range_loop` is allowed for the same reason the exact cells allow it:
    /// `j` indexes FOUR state vectors plus `out` at once, and the ascending index
    /// order IS the numeric contract.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&self, x: &[f32], s: &mut SlstmState, out: &mut [f32]) {
        let (o, i) = (self.output_size, self.input_size);
        let k_in = x.len().min(i);
        let xs = &x[..k_in];
        for j in 0..o {
            // The four gate pre-activations for unit j, in the block order [i|f|o|z].
            let mut pre = [0.0_f32; 4];
            for (a, p) in pre.iter_mut().enumerate() {
                let base_in = (a * o + j) * i;
                let mut acc = dot_f32(&self.inp[base_in..base_in + k_in], xs);
                acc += self.bias[a * o + j];
                let base_rec = (a * o + j) * o;
                acc += dot_f32(&self.rec[base_rec..base_rec + o], &s.h[..o]);
                *p = acc;
            }

            let i_pre = pre[0];
            let fm = pre[1] + s.m[j];
            let m_t = if fm > i_pre { fm } else { i_pre };
            let ip = (i_pre - m_t).exp();
            let fp = (fm - m_t).exp();
            let z = pre[3].tanh();
            let og = logistic_f32(pre[2]);

            let c_t = fp * s.c[j] + ip * z;
            let n_t = fp * s.n[j] + ip;

            s.m[j] = m_t;
            s.c[j] = c_t;
            s.n[j] = n_t;
            out[j] = og * (c_t / n_t);
        }
        // h is rolled AFTER every unit is done: the recurrence above must read the
        // PREVIOUS step's whole output, never a half-updated one.
        s.h[..o].copy_from_slice(&out[..o]);
    }

    /// Whole-sequence forward: a fresh [`Self::state`] then [`Self::step`] per row.
    /// The loop IS the offline path (see [`run_sequence`]), so wrapper-vs-step
    /// identity is by construction; `slstm_wrapper_is_the_step_loop` pins it anyway.
    pub fn feed_forward(&self, seq: &FastMatrix) -> FastMatrix {
        let o = self.output_size;
        let mut out = FastMatrix::zeros(seq.rows, o);
        let mut st = self.state();
        for t in 0..seq.rows {
            let (lo, hi) = (t * o, t * o + o);
            self.step(seq.row(t), &mut st, &mut out.data[lo..hi]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Mamba (spec S3.1 forward, S3.2 layout).
// ---------------------------------------------------------------------------

/// Per-step temporaries for [`FastMamba::step`]. Lives INSIDE [`MambaState`] because
/// `step` takes `&self` (spec S4.1's pinned signature) and this tree forbids
/// per-frame allocation: the scratch has to hang off the only `&mut` the kernel gets.
/// It carries NO information between steps -- every field is fully overwritten before
/// it is read -- so it is deliberately private and absent from state equality.
#[derive(Clone, Debug, Default)]
struct MambaScratch {
    xa: Vec<f32>,    // d_model  (x', the adapted input; also the residual)
    xn: Vec<f32>,    // d_model  (RMSNorm output)
    proj: Vec<f32>,  // 2*d_inner ([u | res])
    uc: Vec<f32>,    // d_inner  (post-SiLU conv output)
    z: Vec<f32>,     // dt_rank + 2*d_state ([dt~ | B | C])
    delta: Vec<f32>, // d_inner
    y: Vec<f32>,     // d_inner
    v: Vec<f32>,     // d_inner
}

/// The Mamba carried state (spec S4.1): the depthwise conv's `d_conv`-deep ring of
/// past `u` vectors plus the SSM hidden state `h[c, s]`.
///
/// `conv_ring` is `d_conv * d_inner` long, slot-major (`conv_ring[slot*d_inner + c]`);
/// `ring_pos` is the slot the NEXT step writes. A fresh state has a zeroed ring, which
/// reproduces the exact forward's LEFT PADDING exactly (see the module doc's
/// divergence 4). `h` is `d_inner * d_state`, indexed `c * d_state + s` -- the same
/// packing `MambaLayer`'s cache uses.
///
/// `PartialEq` compares only the three carried fields (the private scratch is
/// derived), so a state comparison in a test means what it says.
#[derive(Clone, Debug)]
pub struct MambaState {
    pub conv_ring: Vec<f32>,
    pub ring_pos: usize,
    pub h: Vec<f32>,
    scratch: MambaScratch,
}

impl PartialEq for MambaState {
    fn eq(&self, other: &MambaState) -> bool {
        self.conv_ring == other.conv_ring && self.ring_pos == other.ring_pos && self.h == other.h
    }
}

/// f32 forward-only Mamba (S6) cell in RECURRENT form. Every projection matrix is
/// stored ROW-MAJOR BY ITS OUTPUT INDEX, which (a) makes each output element one
/// ascending [`dot_f32`] over a contiguous row and (b) makes `from_flat` a sequence
/// of contiguous narrowing copies straight off the spec S3.2 pack:
///
/// | buffer | index | holds |
/// |---|---|---|
/// | `adapter_w` | `m*I + k` | `P[m, k]` (only when `in != out`) |
/// | `adapter_b` | `m` | `p[m]` |
/// | `norm_gain` | `m` | `g[m]` |
/// | `w_in` | `i*d_model + m` | `W_in[i, m]` |
/// | `conv_w` | `c*d_conv + k` | `conv[c, k]` |
/// | `conv_b` | `c` | `b_conv[c]` |
/// | `w_x` | `i*d_inner + c` | `W_x[i, c]` |
/// | `w_dt` | `c*dt_rank + r` | `W_dt[c, r]` |
/// | `b_dt` | `c` | `b_dt[c]` |
/// | `a_mat` | `c*d_state + s` | `A[c, s] = -exp(A_log[c, s])` |
/// | `d_skip` | `c` | `D[c]` |
/// | `w_out` | `m*d_inner + c` | `W_out[m, c]` |
///
/// `a_mat` is PRECOMPUTED at construction rather than per step. `A = -exp(A_log)` is a
/// pure function of the (already narrowed) weights, so hoisting it is value-identical
/// to the exact path's per-call recompute -- it just does not repeat `d_inner*d_state`
/// `exp` calls at every timestep.
#[derive(Clone)]
pub struct FastMamba {
    input_size: usize,
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    d_inner: usize,
    dt_rank: usize,
    has_adapter: bool,

    adapter_w: Vec<f32>,
    adapter_b: Vec<f32>,
    norm_gain: Vec<f32>,
    w_in: Vec<f32>,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    w_x: Vec<f32>,
    w_dt: Vec<f32>,
    b_dt: Vec<f32>,
    a_mat: Vec<f32>,
    d_skip: Vec<f32>,
    w_out: Vec<f32>,
}

/// Resolve the derived Mamba dimensions the way `MambaLayer::new` does (spec S3.3):
/// `d_state`/`d_conv`/`expand` floored at 1, `d_inner = expand * d_model`,
/// `dt_rank == 0` -> `ceil(d_model / 16)` floored at 1.
fn mamba_dims(d_model: usize, p: &MambaParams) -> (usize, usize, usize, usize) {
    let d_state = p.d_state.max(1);
    let d_conv = p.d_conv.max(1);
    let d_inner = p.expand.max(1) * d_model;
    let dt_rank = if p.dt_rank == 0 {
        d_model.div_ceil(16).max(1)
    } else {
        p.dt_rank
    };
    (d_state, d_conv, d_inner, dt_rank)
}

impl FastMamba {
    /// The spec S3.2 element count -- the arithmetic mirror of the exact
    /// `MambaLayer::nb_of_weights` walk, pinned against it by
    /// `mamba_weight_count_matches_the_exact_cell`.
    pub fn weight_count(
        input_size: usize,
        output_size: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        dt_rank: usize,
    ) -> usize {
        let dm = output_size;
        let (ds, dc, di, dr) = mamba_dims(
            dm,
            &MambaParams {
                d_state,
                d_conv,
                expand,
                dt_rank,
            },
        );
        let adapter = if input_size != dm {
            dm * input_size + dm
        } else {
            0
        };
        adapter
            + dm                        // g
            + 2 * di * dm               // W_in
            + di * dc + di              // conv | b_conv
            + (dr + 2 * ds) * di        // W_x
            + di * dr + di              // W_dt | b_dt
            + di * ds                   // A_log
            + di                        // D
            + dm * di // W_out
    }

    /// Build from the head of the f64 flat pack, narrowing `f64 -> f32` ONCE, block by
    /// block in the spec S3.2 order. PANICS on a short slice (see
    /// [`FastSlstm::from_flat`] for why the length check lives at the net level).
    pub fn from_flat(
        flat: &[f64],
        input_size: usize,
        output_size: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        dt_rank: usize,
    ) -> FastMamba {
        let dm = output_size;
        let (ds, dc, di, dr) = mamba_dims(
            dm,
            &MambaParams {
                d_state,
                d_conv,
                expand,
                dt_rank,
            },
        );
        let has_adapter = input_size != dm;
        let zsz = dr + 2 * ds;

        let mut pos = 0usize;
        let mut take = |k: usize| -> Vec<f32> {
            let seg = &flat[pos..pos + k];
            pos += k;
            seg.iter().map(|&x| x as f32).collect()
        };

        let (adapter_w, adapter_b) = if has_adapter {
            (take(dm * input_size), take(dm))
        } else {
            (Vec::new(), Vec::new())
        };
        let norm_gain = take(dm);
        let w_in = take(2 * di * dm);
        let conv_w = take(di * dc);
        let conv_b = take(di);
        let w_x = take(zsz * di);
        let w_dt = take(di * dr);
        let b_dt = take(di);
        let a_log = take(di * ds);
        let d_skip = take(di);
        let w_out = take(dm * di);

        // A = -exp(A_log), the exact forward's step 7 hoisted out of the loop.
        let a_mat: Vec<f32> = a_log.iter().map(|&v| -v.exp()).collect();

        FastMamba {
            input_size,
            d_model: dm,
            d_state: ds,
            d_conv: dc,
            d_inner: di,
            dt_rank: dr,
            has_adapter,
            adapter_w,
            adapter_b,
            norm_gain,
            w_in,
            conv_w,
            conv_b,
            w_x,
            w_dt,
            b_dt,
            a_mat,
            d_skip,
            w_out,
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.d_model
    }

    pub fn d_inner(&self) -> usize {
        self.d_inner
    }

    pub fn d_state(&self) -> usize {
        self.d_state
    }

    pub fn d_conv(&self) -> usize {
        self.d_conv
    }

    /// The RESOLVED `dt_rank` (never 0).
    pub fn dt_rank(&self) -> usize {
        self.dt_rank
    }

    pub fn has_adapter(&self) -> bool {
        self.has_adapter
    }

    /// A fresh state: a ZEROED conv ring at slot 0 (the exact forward's left padding)
    /// and a zeroed SSM state.
    pub fn state(&self) -> MambaState {
        let (dm, di, ds, dc, dr) = (
            self.d_model,
            self.d_inner,
            self.d_state,
            self.d_conv,
            self.dt_rank,
        );
        MambaState {
            conv_ring: vec![0.0; dc * di],
            ring_pos: 0,
            h: vec![0.0; di * ds],
            scratch: MambaScratch {
                xa: vec![0.0; dm],
                xn: vec![0.0; dm],
                proj: vec![0.0; 2 * di],
                uc: vec![0.0; di],
                z: vec![0.0; dr + 2 * ds],
                delta: vec![0.0; di],
                y: vec![0.0; di],
                v: vec![0.0; di],
            },
        }
    }

    /// ONE timestep (spec S3.1), the streaming kernel. Stages, in the exact forward's
    /// order: width adapter -> RMSNorm -> `[u | res]` projection -> ring-buffered
    /// causal conv + `b_conv` + SiLU -> `[dt~ | B | C]` -> `softplus` `Delta` ->
    /// `Abar`/`h`/`y` recurrence -> SiLU gate -> `W_out` + the `x'` residual.
    ///
    /// `needless_range_loop` is allowed for the same reason the exact cell allows it:
    /// the ascending index order IS the numeric contract.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&self, x: &[f32], s: &mut MambaState, out: &mut [f32]) {
        let (dm, di, ds, dc, dr) = (
            self.d_model,
            self.d_inner,
            self.d_state,
            self.d_conv,
            self.dt_rank,
        );
        let zsz = dr + 2 * ds;
        let sc = &mut s.scratch;

        // 1. Width adapter: x' = P x + p, or x' = x (reconciled to d_model) when
        //    in == out.
        if self.has_adapter {
            let k_in = x.len().min(self.input_size);
            for m in 0..dm {
                let base = m * self.input_size;
                sc.xa[m] =
                    dot_f32(&self.adapter_w[base..base + k_in], &x[..k_in]) + self.adapter_b[m];
            }
        } else {
            for m in 0..dm {
                sc.xa[m] = if m < x.len() { x[m] } else { 0.0 };
            }
        }

        // 2. RMSNorm with the learned gain.
        let mut acc = 0.0_f32;
        for m in 0..dm {
            acc += sc.xa[m] * sc.xa[m];
        }
        let inv = 1.0_f32 / (acc / dm as f32 + RMS_EPS_F32).sqrt();
        for m in 0..dm {
            sc.xn[m] = sc.xa[m] * inv * self.norm_gain[m];
        }

        // 3. Input projection -> [u | res] (no bias).
        for i in 0..2 * di {
            let base = i * dm;
            sc.proj[i] = dot_f32(&self.w_in[base..base + dm], &sc.xn[..dm]);
        }

        // 4. Depthwise CAUSAL conv over the ring + bias + SiLU. The current `u` is
        //    written at `ring_pos` FIRST, so tap `k` (offset `d_conv-1-k`, oldest
        //    first) reads slot `(ring_pos + d_conv - off) % d_conv`.
        let slot = s.ring_pos;
        for c in 0..di {
            s.conv_ring[slot * di + c] = sc.proj[c];
        }
        for c in 0..di {
            let mut a = 0.0_f32;
            for k in 0..dc {
                let off = dc - 1 - k;
                let sl = (slot + dc - off) % dc;
                a += self.conv_w[c * dc + k] * s.conv_ring[sl * di + c];
            }
            sc.uc[c] = silu_f32(a + self.conv_b[c]);
        }
        s.ring_pos = (slot + 1) % dc;

        // 5. Selectivity projection -> [dt~ | B | C] (no bias).
        for i in 0..zsz {
            let base = i * di;
            sc.z[i] = dot_f32(&self.w_x[base..base + di], &sc.uc[..di]);
        }

        // 6. Delta = softplus(W_dt dt~ + b_dt).
        for c in 0..di {
            let base = c * dr;
            let v = dot_f32(&self.w_dt[base..base + dr], &sc.z[..dr]) + self.b_dt[c];
            sc.delta[c] = softplus_f32(v);
        }

        // 7. The selective recurrence: Abar = exp(Delta A) in (0,1) by construction.
        for c in 0..di {
            let dl = sc.delta[c];
            let ucv = sc.uc[c];
            let mut a = 0.0_f32;
            for st in 0..ds {
                let idx = c * ds + st;
                let ab = (dl * self.a_mat[idx]).exp();
                let h_prev = s.h[idx];
                let hv = ab * h_prev + dl * ucv * sc.z[dr + st];
                s.h[idx] = hv;
                a += sc.z[dr + ds + st] * hv;
            }
            sc.y[c] = a + self.d_skip[c] * ucv;
        }

        // 8. Gate by SiLU(res), out-project, add the x' residual.
        for c in 0..di {
            sc.v[c] = sc.y[c] * silu_f32(sc.proj[di + c]);
        }
        for m in 0..dm {
            let base = m * di;
            out[m] = dot_f32(&self.w_out[base..base + di], &sc.v[..di]) + sc.xa[m];
        }
    }

    /// Whole-sequence forward: a fresh [`Self::state`] then [`Self::step`] per row.
    pub fn feed_forward(&self, seq: &FastMatrix) -> FastMatrix {
        let o = self.d_model;
        let mut out = FastMatrix::zeros(seq.rows, o);
        let mut st = self.state();
        for t in 0..seq.rows {
            let (lo, hi) = (t * o, t * o + o);
            self.step(seq.row(t), &mut st, &mut out.data[lo..hi]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// The cell enum (the fast twin of `nn::cells::CellLayer`).
// ---------------------------------------------------------------------------

/// One causal fast cell. A CLOSED set with static dispatch, exactly like
/// `nn::cells::CellLayer` and `engine::bag_of_processors::Processor`: the `match` arms
/// below are exhaustive, so a new cell type is a compile error rather than a silent
/// fall-through. `CellType::Lstm` has NO arm here -- a forward-only LSTM fast twin is
/// out of scope this phase (spec S4.2) and typed-bails in [`super::driver`].
///
/// `large_enum_variant` allowed, on the `Processor`-enum precedent: a [`FastCell`] is
/// built ONCE PER LAYER (a handful per net) and then only borrowed, so the unused
/// tag padding costs nothing measurable -- while boxing the Mamba arm would put a
/// pointer chase in front of every `step`, i.e. on the per-frame streaming path.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum FastCell {
    Slstm(FastSlstm),
    Mamba(FastMamba),
}

/// The carried state of a [`FastCell`], variant-matched to its cell.
#[derive(Clone, Debug, PartialEq)]
pub enum FastCellState {
    Slstm(SlstmState),
    Mamba(MambaState),
}

impl FastCell {
    pub fn input_size(&self) -> usize {
        match self {
            FastCell::Slstm(c) => c.input_size(),
            FastCell::Mamba(c) => c.input_size(),
        }
    }

    pub fn output_size(&self) -> usize {
        match self {
            FastCell::Slstm(c) => c.output_size(),
            FastCell::Mamba(c) => c.output_size(),
        }
    }

    /// A fresh zero state for this cell.
    pub fn state(&self) -> FastCellState {
        match self {
            FastCell::Slstm(c) => FastCellState::Slstm(c.state()),
            FastCell::Mamba(c) => FastCellState::Mamba(c.state()),
        }
    }

    /// One timestep. PANICS on a state built for the OTHER cell type -- unreachable
    /// through [`Self::state`], and a loud failure beats silently reinterpreting a
    /// mismatched state.
    pub fn step(&self, x: &[f32], s: &mut FastCellState, out: &mut [f32]) {
        match (self, s) {
            (FastCell::Slstm(c), FastCellState::Slstm(st)) => c.step(x, st, out),
            (FastCell::Mamba(c), FastCellState::Mamba(st)) => c.step(x, st, out),
            _ => panic!("fast::cells: cell/state variant mismatch"),
        }
    }
}

/// Drive `rows` timesteps of one cell over a contiguous row-major `in_data`
/// (`rows x in_cols`) into `out` (`rows x output_size`), from a FRESH state.
///
/// THE one place the offline stack loops `step`, so [`FastCausalNet`] and the
/// per-cell `feed_forward` wrappers cannot drift: both are this loop.
fn run_sequence(cell: &FastCell, in_data: &[f32], rows: usize, in_cols: usize, out: &mut [f32]) {
    let o = cell.output_size();
    let mut st = cell.state();
    for t in 0..rows {
        let (ilo, ihi) = (t * in_cols, t * in_cols + in_cols);
        let (olo, ohi) = (t * o, t * o + o);
        cell.step(&in_data[ilo..ihi], &mut st, &mut out[olo..ohi]);
    }
}

// ---------------------------------------------------------------------------
// FastCausalNet -- one causal stack + the shared f32 output MLP.
// ---------------------------------------------------------------------------

/// The f32 causal net: ONE cell stack feeding the dense output MLP, the fast twin of
/// `BlstmNetwork` under `Direction::Forward`.
///
/// Flat pack layout (`BlstmNetwork::set_weights`, `nn/blstm.rs:774-800`, with the
/// backward block ABSENT in causal mode): `[stack | output MLP | mean | std]`. The
/// mean/std tail is `2 * lstm_neuron_nb[0]` long and is carried, not applied -- the
/// driver normalizes upstream, exactly as it does for `FastBlstm`.
///
/// STACKS ARE SUPPORTED (a `Vec<FastCell>` + the per-layer sub-sampling chain), and so
/// is sub-sampling: both are what the phase's real causal SAD arm needs
/// (`configs/training/lre_sad.toml` is `lstm_neuron_nb = "23,24,24"` with
/// `lstm_sub_sampling = "4,1"`), and both are `Network::drive` semantics reused
/// verbatim rather than new logic. Sub-sampling is a NET-level concern here exactly as
/// it is in the exact tree -- the `step` kernel always sees ONE already-decimated
/// layer-input row, so the streaming contract is untouched by it.
#[derive(Clone)]
pub struct FastCausalNet {
    lstm_neuron_nb: Vec<usize>,
    lstm_subsampling: Vec<usize>,
    output_subsampling: Vec<usize>,
    output_size: usize,

    cells: Vec<FastCell>,
    output_layers: Vec<FastDenseLayer>,

    normalize_mean: Vec<f32>,
    normalize_std: Vec<f32>,

    scratch: Scratch,
    hidden: Vec<f32>,
    output: FastMatrix,
    /// The dense output MLP's PER-ROW driver -- the same [`DenseRowChain`] the causal
    /// streaming session runs, so offline and streamed posteriors are bit-identical by
    /// construction (see that type's docs for the measured faer `m`-dependence this
    /// removes). Reused across calls (the preallocation contract) and `reset` at the top
    /// of every [`Self::feed_forward`].
    dense_chain: DenseRowChain,
}

impl FastCausalNet {
    /// The whole-net element count: cells + output MLP + the `2 * input_size`
    /// normalize tail. Mirrors `BlstmNetwork::nb_of_weights` with no backward stack.
    pub fn element_count(
        spec: &NnetSpec,
        cell_type: CellType,
        mamba: &MambaParams,
    ) -> Result<usize> {
        let lstm = &spec.lstm_neuron_nb;
        let lsub = &spec.lstm_subsampling;
        let outn = &spec.output_neuron_nb;
        let osub = &spec.output_subsampling;
        // A spec with fewer than two entries in either list describes no layer at all;
        // the `len() - 1` loop bounds below would underflow-panic on it.
        if lstm.len() < 2
            || outn.len() < 2
            || lsub.len() < lstm.len() - 1
            || osub.len() < outn.len() - 1
        {
            bail!(
                "fast::cells::FastCausalNet: malformed NnetSpec (lstm {:?} / sub {:?}, output {:?} / sub {:?})",
                lstm.len(),
                lsub.len(),
                outn.len(),
                osub.len()
            );
        }
        let mut n = 0usize;
        for jj in 0..lstm.len() - 1 {
            let i = lstm[jj] * lsub[jj];
            let o = lstm[jj + 1];
            n += match cell_type {
                CellType::Slstm => FastSlstm::weight_count(i, o),
                CellType::Mamba => FastMamba::weight_count(
                    i,
                    o,
                    mamba.d_state,
                    mamba.d_conv,
                    mamba.expand,
                    mamba.dt_rank,
                ),
                CellType::Lstm => bail!(
                    "fast::cells::FastCausalNet is for the phase-9 causal cells only; the LSTM \
                     cell has no forward-only fast twin (spec S4.2)"
                ),
            };
        }
        for jj in 0..outn.len() - 1 {
            n += outn[jj] * osub[jj] * outn[jj + 1] + outn[jj + 1];
        }
        n += 2 * lstm[0];
        Ok(n)
    }

    /// Build from a `NnetSpec` + the cell type/geometry + the flat f64 pack, narrowing
    /// once per block. Errors on MLP mode, on the LSTM cell (no causal fast twin), and
    /// on a pack shorter than [`Self::element_count`] (mirroring `FastBlstm::from_flat`
    /// and the exact `set_weights`; an OVER-long pack consumes only the head, as the
    /// legacy does).
    pub fn from_flat(
        spec: &NnetSpec,
        cell_type: CellType,
        mamba: &MambaParams,
        flat: &[f64],
    ) -> Result<FastCausalNet> {
        if spec.lstm_neuron_nb.is_empty() || spec.lstm_neuron_nb[0] == 0 {
            bail!(
                "fast::cells::FastCausalNet is recurrent-only; MLP mode (LSTMNeuronNb[0]==0) is \
                 unsupported"
            );
        }
        let needed = Self::element_count(spec, cell_type, mamba)?;
        if flat.len() < needed {
            bail!(
                "flat weight vector too short for the fast causal net: {} < {needed}",
                flat.len()
            );
        }

        let lstm = &spec.lstm_neuron_nb;
        let lsub = &spec.lstm_subsampling;
        let outn = &spec.output_neuron_nb;
        let osub = &spec.output_subsampling;

        let mut pos = 0usize;
        let mut cells = Vec::with_capacity(lstm.len() - 1);
        for jj in 0..lstm.len() - 1 {
            let i = lstm[jj] * lsub[jj];
            let o = lstm[jj + 1];
            let (cell, used) = match cell_type {
                CellType::Slstm => (
                    FastCell::Slstm(FastSlstm::from_flat(&flat[pos..], i, o)),
                    FastSlstm::weight_count(i, o),
                ),
                CellType::Mamba => (
                    FastCell::Mamba(FastMamba::from_flat(
                        &flat[pos..],
                        i,
                        o,
                        mamba.d_state,
                        mamba.d_conv,
                        mamba.expand,
                        mamba.dt_rank,
                    )),
                    FastMamba::weight_count(
                        i,
                        o,
                        mamba.d_state,
                        mamba.d_conv,
                        mamba.expand,
                        mamba.dt_rank,
                    ),
                ),
                // Unreachable: `element_count` above already bailed on Lstm.
                CellType::Lstm => unreachable!("LSTM has no causal fast twin"),
            };
            cells.push(cell);
            pos += used;
        }

        let narrow = |s: &[f64]| -> Vec<f32> { s.iter().map(|&x| x as f32).collect() };
        let mut output_layers = Vec::with_capacity(outn.len() - 1);
        for jj in 0..outn.len() - 1 {
            let i = outn[jj] * osub[jj];
            let o = outn[jj + 1];
            let weights = narrow(&flat[pos..pos + i * o]); // col-major (I x O)
            pos += i * o;
            let biases = narrow(&flat[pos..pos + o]);
            pos += o;
            output_layers.push(FastDenseLayer {
                input_size: i,
                output_size: o,
                weights,
                biases,
            });
        }

        let input_size = lstm[0];
        let normalize_mean = narrow(&flat[pos..pos + input_size]);
        pos += input_size;
        let normalize_std = narrow(&flat[pos..pos + input_size]);
        pos += input_size;
        // Construction-time, once per net: a plain assert (not `debug_assert`), since
        // a layout/count disagreement here silently decodes the WHOLE pack wrong and
        // the release build is exactly where that must not pass quietly.
        assert_eq!(pos, needed, "from_flat consumed != element_count");

        let dense_chain = DenseRowChain::new(output_layers.len());
        Ok(FastCausalNet {
            lstm_neuron_nb: lstm.clone(),
            lstm_subsampling: lsub.clone(),
            output_subsampling: osub.clone(),
            output_size: *outn.last().unwrap(),
            cells,
            output_layers,
            normalize_mean,
            normalize_std,
            scratch: Scratch::default(),
            hidden: Vec::new(),
            output: FastMatrix::zeros(0, 0),
            dense_chain,
        })
    }

    /// The pack-carried mean/std tail (`BlstmNetwork::normalize_input_mean/std`). Not
    /// applied here -- the driver normalizes upstream, matching `FastBlstm`.
    pub fn normalize_mean(&self) -> &[f32] {
        &self.normalize_mean
    }
    pub fn normalize_std(&self) -> &[f32] {
        &self.normalize_std
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// `getSubSamplingRatio` (`nn/blstm.rs:403-410`): the recurrent sub-samplings times
    /// the output-net ones.
    pub fn sub_sampling_ratio(&self) -> usize {
        self.lstm_subsampling.iter().product::<usize>()
            * self.output_subsampling.iter().product::<usize>()
    }

    /// The cell stack (Task 7 drives `step` off these directly).
    pub fn cells(&self) -> &[FastCell] {
        &self.cells
    }

    /// The per-cell-layer sub-sampling ratios (Task 7's streaming stack buffers by them).
    /// `pub(crate)` with [`Self::output_subsampling`] / [`Self::input_width`] /
    /// [`Self::output_layers`]: these four exist ONLY so the sibling `fast::stream` module
    /// can reproduce [`Self::feed_forward`]'s geometry row by row, and widening them to the
    /// public API would advertise internals no external caller has a use for.
    pub(crate) fn lstm_subsampling(&self) -> &[usize] {
        &self.lstm_subsampling
    }

    /// The per-output-layer sub-sampling ratios.
    pub(crate) fn output_subsampling(&self) -> &[usize] {
        &self.output_subsampling
    }

    /// The net's declared input width (`LSTMNeuronNb[0]`) -- the [`Self::feed_forward`]
    /// crop-gate threshold Task 7's streaming stack has to reproduce.
    pub(crate) fn input_width(&self) -> usize {
        self.lstm_neuron_nb[0]
    }

    /// The dense output MLP's layers ([`FastDenseLayer`] is crate-private anyway).
    pub(crate) fn output_layers(&self) -> &[FastDenseLayer] {
        &self.output_layers
    }

    /// Whole-sequence causal forward, the f32 twin of `BlstmNetwork::feed_forward`
    /// under `Direction::Forward`: output length = rows divided SEQUENTIALLY by each
    /// recurrent sub-sampling ratio; the `LSTMRatios[0] > 1 && netInput < inputCols`
    /// crop gate; ONE stack (no reverse pass, no HCAT); then the output MLP over
    /// `hidden` columns. Returns the posteriors, held in the reused `output` buffer.
    ///
    /// THE DENSE STAGE IS PER-ROW (Phase 9 Task 7): it runs [`DenseRowChain`], NOT the
    /// batched [`super::nn::dense_net_forward`] the phase-7 `FastBlstm` keeps, so the
    /// causal streaming session can reproduce it row by row BIT-IDENTICALLY. See
    /// [`DenseRowChain`]'s docs for the measured faer row-count dependence that makes the
    /// batched form unstreamable at `output_size > 1`. Zero-change for every committed
    /// phase-9 fixture (single `4 -> 1` output layer, where batched and per-row are
    /// bit-identical); at a wider hidden dense layer it moves the last f32 ULP, inside the
    /// `CELL_F32_PIN` band the stacked legs pin.
    pub fn feed_forward(&mut self, input: &FastMatrix) -> &FastMatrix {
        let frames = input.rows;
        let mut out_len = frames;
        for &r in &self.lstm_subsampling {
            out_len /= r;
        }
        if frames == 0 || out_len == 0 {
            self.output = FastMatrix::zeros(0, self.output_size);
            return &self.output;
        }

        let fwd_in = self.lstm_neuron_nb[0];
        let in_cols = if self.lstm_subsampling[0] > 1 && fwd_in < input.cols {
            fwd_in
        } else {
            input.cols
        };

        let (rows, cols) = cell_stack_forward(
            &self.cells,
            &self.lstm_subsampling,
            &input.data,
            frames,
            in_cols,
            input.cols,
            &mut self.scratch,
            &mut self.hidden,
        );

        // The dense output MLP, ONE hidden row at a time through the shared chain.
        self.dense_chain.reset();
        self.output.data.clear();
        let mut o_rows = 0usize;
        for r in 0..rows {
            let row = &self.hidden[r * cols..r * cols + cols];
            if self
                .dense_chain
                .push_row(&self.output_layers, &self.output_subsampling, row)
            {
                self.output
                    .data
                    .extend_from_slice(self.dense_chain.output(&self.output_layers));
                o_rows += 1;
            }
        }
        self.output.rows = o_rows;
        self.output.cols = self.output_size;
        &self.output
    }
}

/// One causal stack forward (`Network::drive` with cell layers), the twin of
/// `fast::nn::lstm_net_forward`: materialize the (crop-gate) net input, then chain
/// layers with per-layer sub-sampling, each layer a fresh-state [`run_sequence`].
#[allow(clippy::too_many_arguments)]
fn cell_stack_forward(
    cells: &[FastCell],
    subs: &[usize],
    in_data: &[f32],
    in_rows: usize,
    in_cols: usize,
    in_stride: usize,
    scr: &mut Scratch,
    dest: &mut Vec<f32>,
) -> (usize, usize) {
    copy_view_into(in_data, in_rows, in_cols, in_stride, &mut scr.ping);
    let mut cur_rows = in_rows;
    let mut cur_cols = in_cols;

    for (jj, cell) in cells.iter().enumerate() {
        let o = cell.output_size();
        if subs[jj] > 1 {
            let (sr, sc) = sub_sample_into(
                subs[jj],
                &scr.ping[..cur_rows * cur_cols],
                cur_rows,
                cur_cols,
                cur_cols,
                &mut scr.sub,
            );
            ensure_len(&mut scr.pong, sr * o);
            run_sequence(cell, &scr.sub[..sr * sc], sr, sc, &mut scr.pong[..sr * o]);
            cur_rows = sr;
        } else {
            ensure_len(&mut scr.pong, cur_rows * o);
            run_sequence(
                cell,
                &scr.ping[..cur_rows * cur_cols],
                cur_rows,
                cur_cols,
                &mut scr.pong[..cur_rows * o],
            );
        }
        cur_cols = o;
        std::mem::swap(&mut scr.ping, &mut scr.pong);
    }
    dest.clear();
    dest.extend_from_slice(&scr.ping[..cur_rows * cur_cols]);
    (cur_rows, cur_cols)
}

#[cfg(test)]
mod tests {
    use indexmap::IndexMap;
    use ndarray::Array2;

    use super::*;
    use crate::nn::cells::{MambaLayer, SlstmLayer};

    const I: usize = 5;
    const O: usize = 3;
    const T: usize = 11;

    /// The exact-f64-vs-fast-f32 CELL tolerance, measure-then-pinned: the worst
    /// measured scale-floored relative delta across the three legs below is 3.96e-7
    /// (sLSTM; mamba 1.22e-7 with the adapter, 1.06e-7 without), so this is
    /// `measured * 10` rounded up. A failure here means RE-MEASURE, never widen.
    const CELL_F32_PIN: f64 = 5.0e-6;

    /// Deterministic, non-degenerate weight fill (the exact cells' unit-test fill).
    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.31 - 0.013 * (k as f64) + 0.007 * ((k % 5) as f64))
            .collect()
    }

    /// A BOUNDED deterministic weight fill for the whole-NET legs. [`weights`]'s linear
    /// ramp runs to large negative values over a net-sized pack (~1e3 elements), which
    /// saturates the output layer's logistic to a constant column -- fine for a
    /// single tiny cell, useless as a value comparison. This stays in `[-0.35, 0.35]`
    /// at every index, so the stacked chain produces a posterior that actually varies.
    fn bounded_weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.35 * (0.61 * (k as f64) + 0.3).sin())
            .collect()
    }

    /// A bounded, non-degenerate input sequence. Kept small so the f64/f32 comparison
    /// measures the transcription, not an exploding recurrence.
    fn seq(rows: usize, cols: usize, seed: f64) -> FastMatrix {
        let mut m = FastMatrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                m.data[r * cols + c] = (seed + 0.17 * (r as f64) - 0.23 * (c as f64)
                    + 0.011 * ((r * cols + c) as f64))
                    .sin() as f32;
            }
        }
        m
    }

    fn as_exact(m: &FastMatrix) -> Array2<f64> {
        Array2::from_shape_fn((m.rows, m.cols), |(r, c)| m.get(r, c) as f64)
    }

    fn max_rel(exact: &Array2<f64>, fast: &FastMatrix) -> f64 {
        let mut worst = 0.0_f64;
        for r in 0..exact.nrows() {
            for c in 0..exact.ncols() {
                let e = exact[[r, c]];
                let f = fast.get(r, c) as f64;
                worst = worst.max((e - f).abs() / e.abs().max(1e-2));
            }
        }
        worst
    }

    fn mamba_params() -> MambaParams {
        MambaParams {
            d_state: 4,
            d_conv: 3,
            expand: 2,
            dt_rank: 0,
        }
    }

    fn fast_mamba(i: usize, o: usize) -> FastMamba {
        let p = mamba_params();
        let n = FastMamba::weight_count(i, o, p.d_state, p.d_conv, p.expand, p.dt_rank);
        FastMamba::from_flat(&weights(n), i, o, p.d_state, p.d_conv, p.expand, p.dt_rank)
    }

    fn exact_mamba(i: usize, o: usize) -> MambaLayer {
        let p = mamba_params();
        let mut cell = MambaLayer::new(i, o, p.d_state, p.d_conv, p.expand, p.dt_rank);
        cell.set_weights(&weights(cell.nb_of_weights()));
        cell
    }

    // --- weight counts: the fast arithmetic vs the exact walk ------------------

    #[test]
    fn slstm_weight_count_matches_the_exact_cell() {
        for (i, o) in [(5usize, 3usize), (92, 4), (1, 1), (23, 24)] {
            assert_eq!(
                FastSlstm::weight_count(i, o),
                SlstmLayer::new(i, o).nb_of_weights(),
                "in={i} out={o}"
            );
        }
    }

    #[test]
    fn mamba_weight_count_matches_the_exact_cell() {
        let p = mamba_params();
        // Both the adapter (in != out) and the adapter-free (in == out) shapes.
        for (i, o) in [(5usize, 3usize), (92, 4), (4, 4), (24, 24), (48, 24)] {
            assert_eq!(
                FastMamba::weight_count(i, o, p.d_state, p.d_conv, p.expand, p.dt_rank),
                MambaLayer::new(i, o, p.d_state, p.d_conv, p.expand, p.dt_rank).nb_of_weights(),
                "in={i} out={o}"
            );
        }
        // ...and at the SPEC DEFAULTS, which resolve dt_rank differently (d_model 64
        // -> ceil(64/16) = 4, not the 1 the tiny fixtures use).
        let d = MambaParams::default();
        for (i, o) in [(35usize, 64usize), (64, 64)] {
            assert_eq!(
                FastMamba::weight_count(i, o, d.d_state, d.d_conv, d.expand, d.dt_rank),
                MambaLayer::new(i, o, d.d_state, d.d_conv, d.expand, d.dt_rank).nb_of_weights(),
                "defaults in={i} out={o}"
            );
        }
    }

    /// The committed T5 fixture geometry (`tests/reference_data/phase9/manifest.json`:
    /// sad_input 23, sad_sub_sampling 4, sad_hidden 4, output 4,1) must reproduce the
    /// manifest's recorded pack lengths EXACTLY -- the cross-check that the fast
    /// element count agrees with the Python packer that wrote those `.bin` files.
    // `identity_op` allowed in the two arithmetic legs below: the `* 1` / `* osub`
    // factors are the LAYOUT FORMULA written out (`outn[j]*osub[j]*outn[j+1] +
    // outn[j+1]`, `neuron[j]*sub[j]`), and folding them away hides which dimension
    // each term is.
    #[test]
    #[allow(clippy::identity_op)]
    fn fixture_geometry_reproduces_the_manifest_pack_lengths() {
        let (i, o) = (23 * 4, 4);
        let mlp = 4 * 1 + 1;
        let tail = 2 * 23;
        assert_eq!(FastSlstm::weight_count(i, o) + mlp + tail, 1603);
        let p = mamba_params();
        assert_eq!(
            FastMamba::weight_count(i, o, p.d_state, p.d_conv, p.expand, p.dt_rank) + mlp + tail,
            683
        );
    }

    // --- construction + state -------------------------------------------------

    #[test]
    fn slstm_state_is_zeroed_with_the_m_sentinel() {
        let cell = FastSlstm::from_flat(&weights(FastSlstm::weight_count(I, O)), I, O);
        let st = cell.state();
        assert_eq!(st.h, vec![0.0; O]);
        assert_eq!(st.c, vec![0.0; O]);
        assert_eq!(st.n, vec![0.0; O]);
        assert_eq!(st.m, vec![M_INIT_F32; O]);
        // The absorption property the m_0 = i~_0 convention rests on, in f32.
        for probe in [0.0_f32, 1.0, -50.0, 1e6] {
            assert_eq!(probe + M_INIT_F32, M_INIT_F32, "sentinel absorbed {probe}");
        }
        // ...and the f'_0 underflow that makes the zero initial state inert.
        assert_eq!((M_INIT_F32 - 1.0_f32).exp(), 0.0);
    }

    #[test]
    fn mamba_state_ring_starts_zeroed_at_slot_zero() {
        let cell = fast_mamba(I, O);
        let st = cell.state();
        assert_eq!(st.ring_pos, 0);
        assert_eq!(st.conv_ring.len(), cell.d_conv() * cell.d_inner());
        assert!(st.conv_ring.iter().all(|&v| v == 0.0));
        assert_eq!(st.h.len(), cell.d_inner() * cell.d_state());
        assert!(st.h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn mamba_adapter_presence_follows_the_width() {
        assert!(fast_mamba(I, O).has_adapter());
        assert!(!fast_mamba(O, O).has_adapter());
        // dt_rank 0 resolves to ceil(d_model/16), floored at 1.
        assert_eq!(fast_mamba(I, O).dt_rank(), 1);
    }

    // --- step-loop == wrapper, run-twice, split-state -------------------------

    #[test]
    fn slstm_wrapper_is_the_step_loop() {
        let cell = FastSlstm::from_flat(&weights(FastSlstm::weight_count(I, O)), I, O);
        let input = seq(T, I, 0.4);
        let wrapped = cell.feed_forward(&input);

        let mut manual = FastMatrix::zeros(T, O);
        let mut st = cell.state();
        for t in 0..T {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st, &mut manual.data[lo..hi]);
        }
        assert_eq!(wrapped, manual, "wrapper != step loop");
        assert!(wrapped.data.iter().all(|v| v.is_finite()));
        // Non-vacuity: the sequence actually moves.
        assert!(wrapped.data.iter().any(|&v| v != wrapped.data[0]));

        // Run twice: bit-identical (a fresh state per call, no leak).
        assert_eq!(cell.feed_forward(&input), wrapped);
    }

    #[test]
    fn mamba_wrapper_is_the_step_loop() {
        let cell = fast_mamba(I, O);
        let input = seq(T, I, 0.4);
        let wrapped = cell.feed_forward(&input);

        let mut manual = FastMatrix::zeros(T, O);
        let mut st = cell.state();
        for t in 0..T {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st, &mut manual.data[lo..hi]);
        }
        assert_eq!(wrapped, manual, "wrapper != step loop");
        assert!(wrapped.data.iter().all(|v| v.is_finite()));
        assert!(wrapped.data.iter().any(|&v| v != wrapped.data[0]));
        assert_eq!(cell.feed_forward(&input), wrapped);
    }

    /// THE STREAMING PRECONDITION (Task 7 depends on it): a state carried across a
    /// MID-SEQUENCE split reproduces the unsplit run BIT-FOR-BIT.
    ///
    /// The split index is 4 of 11 -- deliberately not a half, not a multiple of
    /// `d_conv` (3), and past the conv's warm-up, so the second half genuinely
    /// consumes carried conv history and a carried `m`/`h`.
    #[test]
    fn slstm_split_state_reproduces_the_unsplit_run() {
        let cell = FastSlstm::from_flat(&weights(FastSlstm::weight_count(I, O)), I, O);
        let input = seq(T, I, 0.4);
        let whole = cell.feed_forward(&input);

        const SPLIT: usize = 4;
        let mut spliced = FastMatrix::zeros(T, O);
        let mut st = cell.state();
        for t in 0..SPLIT {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st, &mut spliced.data[lo..hi]);
        }
        // The carried state is genuinely non-trivial at the cut.
        assert!(st.h.iter().any(|&v| v != 0.0), "carried h is all zero");
        assert!(st.m.iter().all(|&v| v != M_INIT_F32), "m never advanced");
        for t in SPLIT..T {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st, &mut spliced.data[lo..hi]);
        }
        assert_eq!(spliced, whole, "split-state run diverged");

        // Contrast (non-vacuity): a RESET state at the cut does NOT reproduce it, so
        // the equality above is carrying real information.
        let mut reset = FastMatrix::zeros(T, O);
        let mut st2 = cell.state();
        for t in 0..SPLIT {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st2, &mut reset.data[lo..hi]);
        }
        let mut st3 = cell.state();
        for t in SPLIT..T {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st3, &mut reset.data[lo..hi]);
        }
        assert_ne!(reset, whole, "a reset at the cut was indistinguishable");
    }

    #[test]
    fn mamba_split_state_reproduces_the_unsplit_run() {
        let cell = fast_mamba(I, O);
        let input = seq(T, I, 0.4);
        let whole = cell.feed_forward(&input);

        const SPLIT: usize = 4;
        let mut spliced = FastMatrix::zeros(T, O);
        let mut st = cell.state();
        for t in 0..SPLIT {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st, &mut spliced.data[lo..hi]);
        }
        assert!(st.h.iter().any(|&v| v != 0.0), "carried h is all zero");
        assert!(
            st.conv_ring.iter().any(|&v| v != 0.0),
            "carried conv ring is all zero"
        );
        // 4 steps with d_conv 3 wrapped the ring exactly once.
        assert_eq!(st.ring_pos, SPLIT % cell.d_conv());
        for t in SPLIT..T {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st, &mut spliced.data[lo..hi]);
        }
        assert_eq!(spliced, whole, "split-state run diverged");

        let mut reset = FastMatrix::zeros(T, O);
        let mut st2 = cell.state();
        for t in 0..SPLIT {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st2, &mut reset.data[lo..hi]);
        }
        let mut st3 = cell.state();
        for t in SPLIT..T {
            let (lo, hi) = (t * O, t * O + O);
            cell.step(input.row(t), &mut st3, &mut reset.data[lo..hi]);
        }
        assert_ne!(reset, whole, "a reset at the cut was indistinguishable");
    }

    // --- the transcription itself: f32 fast vs f64 exact ----------------------

    /// The CELL-level transcription check (the driver-level one is
    /// `tests/phase9_fast_parity.rs`): same weights, same input, exact f64 cell vs
    /// fast f32 cell. MEASURE-THEN-PIN.
    #[test]
    fn slstm_matches_the_exact_cell_within_the_f32_band() {
        let n = FastSlstm::weight_count(I, O);
        let fast = FastSlstm::from_flat(&weights(n), I, O);
        let mut exact = SlstmLayer::new(I, O);
        exact.set_weights(&weights(n));

        let input = seq(T, I, 0.4);
        let input_f64 = as_exact(&input);
        let mut out_e = Array2::<f64>::zeros((T, O));
        exact.feed_forward(&input_f64, &mut out_e, false);
        let out_f = fast.feed_forward(&input);

        // MEASURED max scale-floored relative delta on this box (M4 Pro): 3.96e-7.
        // Pinned at [`CELL_F32_PIN`] = 5e-6 (> measured*10) -- an f32 epsilon-scale
        // gap, i.e. the narrowing and nothing else. A structural transcription error
        // (a swapped gate block, a dropped stabilizer) moves this by orders of
        // magnitude, not ULPs.
        let worst = max_rel(&out_e, &out_f);
        println!("MEASURE slstm cell exact-vs-fast max_rel = {worst:e}");
        assert!(
            worst < CELL_F32_PIN,
            "slstm f32 transcription drift: {worst:e}"
        );
    }

    #[test]
    fn mamba_matches_the_exact_cell_within_the_f32_band() {
        let fast = fast_mamba(I, O);
        let mut exact = exact_mamba(I, O);

        let input = seq(T, I, 0.4);
        let input_f64 = as_exact(&input);
        let mut out_e = Array2::<f64>::zeros((T, O));
        exact.feed_forward(&input_f64, &mut out_e, false);
        let out_f = fast.feed_forward(&input);

        // MEASURED: 1.22e-7; pinned at [`CELL_F32_PIN`] (same reasoning as the sLSTM
        // leg).
        let worst = max_rel(&out_e, &out_f);
        println!("MEASURE mamba cell exact-vs-fast max_rel = {worst:e}");
        assert!(
            worst < CELL_F32_PIN,
            "mamba f32 transcription drift: {worst:e}"
        );
    }

    /// The ADAPTER-FREE shape (`in == out`) takes a different first stage entirely
    /// (`x' = x`, no `P`/`p` block in the pack), so it gets its own leg.
    #[test]
    fn mamba_without_the_adapter_matches_the_exact_cell() {
        let fast = fast_mamba(O, O);
        let mut exact = exact_mamba(O, O);
        assert!(!fast.has_adapter() && !exact.has_adapter());

        let input = seq(T, O, -0.3);
        let input_f64 = as_exact(&input);
        let mut out_e = Array2::<f64>::zeros((T, O));
        exact.feed_forward(&input_f64, &mut out_e, false);
        let out_f = fast.feed_forward(&input);

        // MEASURED: 1.06e-7; pinned at [`CELL_F32_PIN`].
        let worst = max_rel(&out_e, &out_f);
        println!("MEASURE mamba (no adapter) exact-vs-fast max_rel = {worst:e}");
        assert!(worst < CELL_F32_PIN, "mamba no-adapter drift: {worst:e}");
    }

    /// Width tolerance, shared by both cells: a WIDER input is cropped to the left
    /// `input_size` columns (so it agrees with feeding those columns directly), and a
    /// NARROWER one is zero-padded rather than rejected.
    #[test]
    fn width_tolerance_crops_and_pads() {
        let slstm = FastSlstm::from_flat(&weights(FastSlstm::weight_count(I, O)), I, O);
        let mamba = fast_mamba(I, O);
        let wide = seq(T, I + 2, 0.4);
        let mut cropped = FastMatrix::zeros(T, I);
        for r in 0..T {
            cropped.data[r * I..r * I + I]
                .copy_from_slice(&wide.data[r * (I + 2)..r * (I + 2) + I]);
        }
        assert_eq!(slstm.feed_forward(&wide), slstm.feed_forward(&cropped));
        assert_eq!(mamba.feed_forward(&wide), mamba.feed_forward(&cropped));

        let narrow = seq(T, 2, 0.4);
        assert_eq!(slstm.feed_forward(&narrow).cols, O);
        assert!(
            slstm
                .feed_forward(&narrow)
                .data
                .iter()
                .all(|v| v.is_finite())
        );
        assert!(
            mamba
                .feed_forward(&narrow)
                .data
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// The stabilizer survives `+-800` pre-activations in f32 exactly as it does in
    /// f64 -- and f32 is the harder case (unstabilized `exp` overflows at ~88.7, not
    /// ~709.8).
    #[test]
    fn slstm_stabilizer_survives_huge_pre_activations_in_f32() {
        assert!(800.0_f32.exp().is_infinite(), "the test is vacuous");
        for sign in [1.0_f32, -1.0] {
            // 1 unit, 1 input: [R | W | b] per gate, W = +-800.
            let flat: Vec<f64> = (0..4)
                .flat_map(|_| [0.0, 800.0 * sign as f64, 0.0])
                .collect();
            let cell = FastSlstm::from_flat(&flat, 1, 1);
            let mut input = FastMatrix::zeros(5, 1);
            input.data.copy_from_slice(&[1.0, -1.0, 1.0, 1.0, -1.0]);
            let out = cell.feed_forward(&input);
            assert!(
                out.data.iter().all(|v| v.is_finite()),
                "sign={sign}: {out:?}"
            );
        }
    }

    // --- the net ---------------------------------------------------------------

    /// A tiny causal `NnetSpec` mirroring the fixture geometry, with `layers`
    /// recurrent widths and a per-layer sub-sampling list.
    fn spec(lstm: &[usize], lsub: &[usize], outn: &[usize], osub: &[usize]) -> NnetSpec {
        NnetSpec {
            input_size: lstm[0],
            lstm_neuron_nb: lstm.to_vec(),
            lstm_subsampling: lsub.to_vec(),
            output_neuron_nb: outn.to_vec(),
            output_subsampling: osub.to_vec(),
            peepholes: [false; 6],
        }
    }

    /// A MULTI-LAYER, SUB-SAMPLED causal net (the real arm's shape:
    /// `lre_sad.toml` is `23,24,24` / `4,1`) builds, consumes exactly its element
    /// count, and produces the decimated row count `Network::drive` would.
    #[test]
    #[allow(clippy::identity_op)]
    fn causal_net_supports_stacks_and_sub_sampling() {
        let sp = spec(&[6, 5, 4], &[2, 1], &[4, 3, 1], &[1, 1]);
        let p = mamba_params();
        for cell in [CellType::Slstm, CellType::Mamba] {
            let n = FastCausalNet::element_count(&sp, cell, &p).unwrap();
            // Independent arithmetic: two cell layers (in = neuron*sub) + two dense
            // layers + the 2*input_size tail.
            let want_cells = match cell {
                CellType::Slstm => {
                    FastSlstm::weight_count(6 * 2, 5) + FastSlstm::weight_count(5 * 1, 4)
                }
                CellType::Mamba => {
                    FastMamba::weight_count(12, 5, p.d_state, p.d_conv, p.expand, p.dt_rank)
                        + FastMamba::weight_count(5, 4, p.d_state, p.d_conv, p.expand, p.dt_rank)
                }
                CellType::Lstm => unreachable!(),
            };
            assert_eq!(
                n,
                want_cells + (4 * 3 + 3) + (3 * 1 + 1) + 2 * 6,
                "{cell:?}"
            );

            let mut net = FastCausalNet::from_flat(&sp, cell, &p, &weights(n)).unwrap();
            assert_eq!(net.sub_sampling_ratio(), 2);
            assert_eq!(net.cells().len(), 2);

            // 13 input rows -> layer-0 sub-sampling 2 drops the odd tail -> 6 rows.
            let input = seq(13, 6, 0.2);
            let out = net.feed_forward(&input).clone();
            assert_eq!((out.rows, out.cols), (6, 1));
            assert!(
                out.data
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );
            // Run twice: bit-identical (workspace hygiene).
            assert_eq!(net.feed_forward(&input), &out);
        }
    }

    /// A legacy config map for the SAME geometry [`spec`] describes, as a causal
    /// `BlstmNetwork` under prefix `"X"` -- the exact-tree oracle for the multi-layer
    /// value pin below. Mirrors `nn::blstm`'s own `direction_tests::map_for`.
    fn exact_map(
        cell: CellType,
        lstm: &[usize],
        lsub: &[usize],
        outn: &[usize],
        osub: &[usize],
    ) -> IndexMap<String, String> {
        let join = |v: &[usize]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let p = mamba_params();
        let mut m: IndexMap<String, String> = IndexMap::new();
        m.insert("X_LSTMNeuronNb".into(), join(lstm));
        m.insert("X_LSTMSubSampling".into(), join(lsub));
        m.insert("X_OutputNeuronNb".into(), join(outn));
        m.insert("X_OutputSubSampling".into(), join(osub));
        m.insert("X_InputNormalizationType".into(), "0".into());
        m.insert("X_TwoSweeps".into(), "false".into());
        m.insert("X_Direction".into(), "forward".into());
        m.insert("X_Cell_Type".into(), cell.as_str().into());
        m.insert("Mamba_D_State".into(), p.d_state.to_string());
        m.insert("Mamba_D_Conv".into(), p.d_conv.to_string());
        m.insert("Mamba_Expand".into(), p.expand.to_string());
        m.insert("Mamba_Dt_Rank".into(), p.dt_rank.to_string());
        m
    }

    /// THE MULTI-LAYER VALUE PIN (Task 6 review, I1). The committed phase-9 parity
    /// fixtures are SINGLE-layer, and `causal_net_supports_stacks_and_sub_sampling`
    /// above only checks counts, shapes, finiteness and run-twice -- so the STACKED
    /// chain (the surface the sub-sampling/stacks decision added) had no exact-vs-fast
    /// VALUE comparison anywhere. This is it: the same 2-layer, sub-sampled geometry
    /// and the SAME flat pack driven through the exact f64 `BlstmNetwork`
    /// (`Direction forward` + the same cell, i.e. `Network<CellLayer>` over
    /// `SlstmLayer`/`MambaLayer`) and through `FastCausalNet`, compared at
    /// [`CELL_F32_PIN`].
    ///
    /// It also cross-pins [`FastCausalNet::element_count`] against the exact net's own
    /// `nb_of_weights()` at a stacked geometry -- if the two disagreed, the shared pack
    /// would decode differently on each side and the value comparison would be
    /// meaningless rather than failing.
    #[test]
    fn causal_net_stack_matches_the_exact_net_within_the_f32_band() {
        let (lstm, lsub, outn, osub) = (
            [6usize, 5, 4].as_slice(),
            [2usize, 1].as_slice(),
            [4usize, 3, 1].as_slice(),
            [1usize, 1].as_slice(),
        );
        let sp = spec(lstm, lsub, outn, osub);
        let p = mamba_params();

        for cell in [CellType::Slstm, CellType::Mamba] {
            let cfg = crate::nn::blstm::BlstmConfig::from_legacy(
                &exact_map(cell, lstm, lsub, outn, osub),
                "X",
            )
            .unwrap();
            let mut exact = crate::nn::blstm::BlstmNetwork::from_config(cfg).unwrap();

            let n = FastCausalNet::element_count(&sp, cell, &p).unwrap();
            assert_eq!(
                n,
                exact.nb_of_weights(),
                "{cell:?}: fast element_count != the exact net's pack length"
            );
            let flat = bounded_weights(n);
            exact.set_weights(&flat).unwrap();
            let mut fast = FastCausalNet::from_flat(&sp, cell, &p, &flat).unwrap();

            // 13 rows -> layer-0 sub-sampling 2 drops the odd tail -> 6 output rows.
            let input = seq(13, 6, 0.2);
            let input_f64 = as_exact(&input);
            let mut out_e = Array2::<f64>::zeros((6, 1));
            exact.feed_forward(&input_f64, &mut out_e);
            let out_f = fast.feed_forward(&input).clone();
            assert_eq!((out_f.rows, out_f.cols), (6, 1), "{cell:?}: fast shape");

            // Non-vacuity: the stacked chain must actually vary across time, or the
            // comparison would pass on a constant column.
            let first = out_e[[0, 0]];
            assert!(
                (0..6).any(|r| (out_e[[r, 0]] - first).abs() > 1e-9),
                "{cell:?}: the exact stack output is constant -- the pin is vacuous"
            );

            // MEASURED on this box: sLSTM 7.91e-8, mamba 6.14e-8. Pinned at
            // [`CELL_F32_PIN`] (5e-6), the same measure-then-pin band as the
            // single-cell legs above.
            let worst = max_rel(&out_e, &out_f);
            println!("MEASURE causal STACK {cell:?} exact-vs-fast max_rel = {worst:e}");
            assert!(
                worst < CELL_F32_PIN,
                "{cell:?}: multi-layer causal drift {worst:e}"
            );
        }
    }

    /// A pack one element short must be a typed `Err`, not a panic or a silently
    /// head-eaten wrong net; an over-long pack is accepted head-first (the legacy
    /// tolerance `BlstmNetwork::set_weights` implements).
    #[test]
    fn causal_net_length_check_is_typed() {
        let sp = spec(&[4, 3], &[1], &[3, 1], &[1]);
        let p = mamba_params();
        let n = FastCausalNet::element_count(&sp, CellType::Slstm, &p).unwrap();
        let short = weights(n - 1);
        let err = FastCausalNet::from_flat(&sp, CellType::Slstm, &p, &short)
            .err()
            .expect("a short pack must be rejected");
        assert!(
            err.to_string().contains("too short"),
            "expected a length bail, got: {err}"
        );
        assert!(FastCausalNet::from_flat(&sp, CellType::Slstm, &p, &weights(n + 17)).is_ok());
    }

    /// The LSTM cell has NO causal fast twin this phase: `element_count` (and hence
    /// `from_flat`) bails loudly rather than falling through to a wrong architecture.
    #[test]
    fn causal_net_bails_on_the_lstm_cell() {
        let sp = spec(&[4, 3], &[1], &[3, 1], &[1]);
        let err = FastCausalNet::element_count(&sp, CellType::Lstm, &mamba_params()).unwrap_err();
        assert!(
            err.to_string().contains("no forward-only fast twin"),
            "expected an LSTM bail, got: {err}"
        );
    }

    /// MLP mode (`LSTMNeuronNb[0] == 0`) is not a causal shape.
    #[test]
    fn causal_net_bails_on_mlp_mode() {
        let sp = spec(&[0, 3], &[1], &[3, 1], &[1]);
        let err = FastCausalNet::from_flat(&sp, CellType::Slstm, &mamba_params(), &weights(64))
            .err()
            .expect("MLP mode must be rejected");
        assert!(
            err.to_string().contains("MLP mode"),
            "expected an MLP bail, got: {err}"
        );
    }
}
