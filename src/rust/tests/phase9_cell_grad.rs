//! Phase 9: the SHARED finite-difference grad-check tier for `nn::cells` (spec S8.1
//! UNIT tier -- CI-visible, committed, runs on every `cargo test`).
//!
//! The instrument is cell-AGNOSTIC: it drives `&mut dyn Layer`, so every cell the
//! phase adds (Task 2 `SlstmLayer`, Task 3 `MambaLayer`) is pinned by the same
//! harness with no changes to it. The contract it checks:
//!
//! ```text
//! L(w) = sum(G .* Y(w))          G a fixed pseudo-random matrix, Y the forward output
//! analytic = col0 of get_weights_derivatives() after feed_backward(deltas = G)
//! fd[k]    = (L(w_k + eps) - L(w_k - eps)) / (2 eps)
//! scale[k] = max(|fd[k]|, |analytic[k]|)
//! rel[k]   = |fd[k] - analytic[k]| / scale[k]        when scale[k] >  NEAR_ZERO
//! abs[k]   = |fd[k] - analytic[k]|                   when scale[k] <= NEAR_ZERO
//! ```
//!
//! `dL/dY = G` by construction, so feeding `G` as the layer's incoming deltas makes
//! `get_weights_derivatives`' col0 EXACTLY `dL/dw` (col1 is the frame count, unused
//! here; no ponderation, `inv_sub_sampling_ratio = 1`). `last_layer = false`
//! everywhere -- cells never sit at the output of a net (`NeuronLayer` does), and
//! the flag is unused by the cell backward anyway.
//!
//! THE TWO REGIMES (a deliberate refinement of the task brief's single formula
//! `|fd - an| / max(1e-8, scale)`, which this file's [`NEAR_ZERO`] keeps as the
//! partition point, unchanged): where the true derivative is ZERO, a relative error
//! is meaningless -- it divides FD ROUNDOFF by a constant. Central differences carry
//! an irreducible noise floor of `ulp(|L|) / (2 eps)` (here `|L| ~ 2`, `eps = 1e-6`
//! -> ~2.2e-10, which is EXACTLY the magnitude measured at those weights), so the
//! brief's formula would report `2.2e-10 / 1e-8 = 2.2e-2` for a derivative that is
//! correct to the last bit. sLSTM HAS such weights by construction -- the whole
//! `b_i` block is structurally zero-gradient (see `nn/cells/slstm.rs`'s module doc:
//! shifting the input-gate bias scales `C` and `N` equally, so `c/n` and therefore
//! the output are invariant; `input_gate_bias_shift_leaves_the_output_invariant`
//! pins that independently of this file). Scoring them on an ABSOLUTE floor reports
//! the right quantity; both regimes are pinned, neither is skipped, and the counts
//! are asserted so a bug cannot hide by pushing weights into the lenient bucket.
//!
//! Randomness is a deterministic 64-bit LCG seeded per case: the exact tree has NO
//! `rand` dependency and must not gain one, and a fixed stream makes every measured
//! number below reproducible. Tolerances are MEASURE-THEN-PIN (spec R4): the
//! measured max relative error per shape is recorded in a comment beside its
//! assert, pinned at measured*10.

use ndarray::Array2;
use speech::nn::cells::{MambaLayer, SlstmLayer};
use speech::nn::network::Layer;

// ---------------------------------------------------------------------------
// Deterministic RNG (no `rand` dependency)
// ---------------------------------------------------------------------------

/// Knuth-MMIX 64-bit LCG. Deterministic across platforms (pure integer wrapping
/// arithmetic + an exact power-of-two scale), so every measured number in this file
/// is reproducible anywhere.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Lcg {
        // One warm-up step so adjacent seeds do not produce correlated first draws.
        let mut lcg = Lcg(seed);
        lcg.next_u64();
        lcg
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform in `[-half, half)`.
    fn uniform(&mut self, half: f64) -> f64 {
        let bits = self.next_u64() >> 11; // top 53 bits
        let u = (bits as f64) * (1.0 / 9007199254740992.0); // [0, 1)
        (2.0 * u - 1.0) * half
    }

    fn vec(&mut self, n: usize, half: f64) -> Vec<f64> {
        (0..n).map(|_| self.uniform(half)).collect()
    }
}

// ---------------------------------------------------------------------------
// The harness
// ---------------------------------------------------------------------------

/// The scale below which a derivative counts as structurally zero and is scored on
/// an absolute floor instead of a relative one. This is the SAME partition point the
/// task brief's `max(1e-8, scale)` denominator uses -- only the reported quantity
/// differs below it (see the module doc).
const NEAR_ZERO: f64 = 1e-8;

/// A weight is MAJOR when its analytic derivative is at least this fraction of the
/// largest one in the same pack. [`FdReport::max_rel_major`] restricts the relative
/// check to those weights, and that number -- not the unrestricted `max_rel` -- is
/// what actually discriminates a wrong adjoint from finite-difference noise.
///
/// The reason is a scale spread the sLSTM legs never exposed: mamba is a RESIDUAL
/// block whose branch output is a product of five small factors at small random init,
/// so one pack carries derivatives from `1e-2` down to `1e-12`. Central differences
/// resolve a derivative to an ABSOLUTE floor (`~ulp(|L|)/(2 eps)` plus truncation),
/// so the relative error of the smallest resolvable weight is that floor divided by a
/// near-zero number -- large, and completely uninformative about the derivation. A
/// wrong adjoint term, by contrast, is a MULTIPLICATIVE error: it shows up at the
/// DOMINANT weights, which is exactly where this metric looks.
const MAJOR_FRACTION: f64 = 1e-4;

/// One FD sweep's outcome. `max_rel` and `max_abs_near_zero` are the pinned
/// quantities; the rest is the adjudication context a failure needs (which weight,
/// the two raw numbers, and whether the sweep was non-vacuous at all).
pub struct FdReport {
    /// Max relative error over the RESOLVABLE weights (`scale > NEAR_ZERO`).
    pub max_rel: f64,
    /// Flat index of the weight where `max_rel` was attained.
    pub arg_max: usize,
    /// `(fd, analytic)` at `arg_max` -- so a failure reports the two numbers, not
    /// just their ratio.
    pub worst_pair: (f64, f64),
    /// Max ABSOLUTE error over the near-zero weights (`scale <= NEAR_ZERO`): the FD
    /// noise floor is the only thing measurable there.
    pub max_abs_near_zero: f64,
    /// Flat index of the weight where `max_abs_near_zero` was attained.
    pub arg_max_near_zero: usize,
    /// How many weights landed in the relative regime (the rest are near-zero). A
    /// pinned count is what stops a broken backward from hiding in the lenient
    /// bucket by driving every derivative to zero.
    pub resolvable: usize,
    /// Total weights swept (`nb_of_weights`).
    pub total: usize,
    /// Max ABSOLUTE error over ALL weights, near-zero or not. The most physically
    /// meaningful number of the three: central differences cannot resolve better
    /// than `ulp(|L|)/(2 eps)`, so this is the floor every other figure is a
    /// re-scaling of.
    pub max_abs_err: f64,
    /// Largest `|analytic|` over the sweep: a non-vacuity witness (an all-zero
    /// gradient would pass every relative check trivially).
    pub max_abs_analytic: f64,
    /// Max relative error over the MAJOR weights only (`|analytic| >=
    /// MAJOR_FRACTION * max_abs_analytic`) -- the discriminating number (see
    /// [`MAJOR_FRACTION`]).
    pub max_rel_major: f64,
    /// Flat index where `max_rel_major` was attained.
    pub arg_max_major: usize,
    /// How many weights are MAJOR. Reported so `max_rel_major` cannot be silently
    /// computed over an empty or near-empty set.
    pub major: usize,
}

/// Central-difference vs analytic on `L = sum(G .* Y)`.
///
/// DEVIATION from the task brief's sketch signature (`fd_check(layer, t, input_size,
/// eps)`): `output_size` and `seed` are explicit parameters. `Layer` exposes no
/// output width (`feed_forward` writes a CALLER-allocated buffer), so the harness
/// cannot infer the `G`/`Y` shape from the trait; and the brief's own Step 5 asks for
/// three seeds per shape, which needs the seed threaded in. Everything else is the
/// sketch verbatim.
pub fn fd_check(
    layer: &mut dyn Layer,
    t: usize,
    input_size: usize,
    output_size: usize,
    seed: u64,
    eps: f64,
) -> FdReport {
    let nb = layer.nb_of_weights();
    let mut rng = Lcg::new(seed);
    let mut w = rng.vec(nb, 0.5);
    let x = Array2::from_shape_vec((t, input_size), rng.vec(t * input_size, 1.0)).unwrap();
    let g = Array2::from_shape_vec((t, output_size), rng.vec(t * output_size, 1.0)).unwrap();

    // Analytic: one forward + one backward at w, harvesting col0.
    layer.set_weights(&w);
    layer.reset_weights_derivatives();
    let mut y = Array2::<f64>::zeros((t, output_size));
    layer.feed_forward(&x, &mut y, false);
    let _ = layer.feed_backward(&x, &y, &g, 1, false);
    let mut rows: Vec<[f64; 2]> = Vec::new();
    layer.get_weights_derivatives(&mut rows);
    assert_eq!(rows.len(), nb, "deriv rows must match nb_of_weights");
    let analytic: Vec<f64> = rows.iter().map(|r| r[0]).collect();

    let loss = |layer: &mut dyn Layer, w: &[f64]| -> f64 {
        layer.set_weights(w);
        let mut y = Array2::<f64>::zeros((t, output_size));
        layer.feed_forward(&x, &mut y, false);
        let mut acc = 0.0;
        for r in 0..t {
            for c in 0..output_size {
                acc += g[[r, c]] * y[[r, c]];
            }
        }
        acc
    };

    // The MAJOR cut needs the pack maximum up front, so it is taken before the sweep
    // (the analytic gradient is already fully known at this point).
    let max_abs_analytic = analytic.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let major_cut = MAJOR_FRACTION * max_abs_analytic;

    let mut report = FdReport {
        max_rel: 0.0,
        arg_max: 0,
        worst_pair: (0.0, 0.0),
        max_abs_near_zero: 0.0,
        arg_max_near_zero: 0,
        resolvable: 0,
        total: nb,
        max_abs_err: 0.0,
        max_abs_analytic,
        max_rel_major: 0.0,
        arg_max_major: 0,
        major: 0,
    };
    for k in 0..nb {
        let orig = w[k];
        w[k] = orig + eps;
        let lp = loss(layer, &w);
        w[k] = orig - eps;
        let lm = loss(layer, &w);
        w[k] = orig;

        let fd = (lp - lm) / (2.0 * eps);
        let an = analytic[k];
        let err = (fd - an).abs();
        let scale = fd.abs().max(an.abs());
        if scale > NEAR_ZERO {
            report.resolvable += 1;
            let rel = err / scale;
            if rel > report.max_rel {
                report.max_rel = rel;
                report.arg_max = k;
                report.worst_pair = (fd, an);
            }
            if an.abs() >= major_cut {
                report.major += 1;
                if rel > report.max_rel_major {
                    report.max_rel_major = rel;
                    report.arg_max_major = k;
                }
            }
        } else if err > report.max_abs_near_zero {
            report.max_abs_near_zero = err;
            report.arg_max_near_zero = k;
        }
        report.max_abs_err = report.max_abs_err.max(err);
    }
    report
}

/// One row of the sweep grid. A NAMED STRUCT rather than the tuple this started as:
/// Task 3 grew it from four positional fields to six, two of which are adjacent
/// counts and two adjacent floats -- exactly the shape a transposed-argument bug
/// hides in. The harness logic ([`fd_check`], [`sweep`]) is unchanged.
///
/// THE COUNT IS TWO-SIDED, and the two bounds catch DIFFERENT failures:
///
/// - `resolvable_structural` is an UPPER bound: the number of weights whose analytic
///   derivative is not EXACTLY zero by structure (`total - structurally_dead`). You
///   can never resolve more than that, so exceeding it means a weight that must be
///   dead became live -- a layout/indexing bug (mamba: a conv tap reaching past the
///   causal left-padding, or `A_log` picking up gradient against the zero initial
///   state at `T = 1`).
/// - `resolvable_floor` is a LOWER bound: the original single-count assert's real job
///   was to stop a broken backward from passing by collapsing every derivative into
///   the lenient near-zero bucket, and a floor does exactly that.
///
/// They are EQUAL for sLSTM (whose structurally-zero set is a clean block, so the
/// count is seed-independent and the assert is effectively `==` there). They DIFFER
/// for mamba: [`NEAR_ZERO`] is an ABSOLUTE floor, and a residual block at small
/// random init produces true derivatives spanning `1e-3` down to `1e-12` (the SSM
/// branch's contribution is a product of five small factors), so a handful of
/// genuinely live weights fall under `1e-8` by magnitude alone -- which ones is
/// seed-dependent and carries no information. Those weights are still CHECKED, on the
/// absolute floor, where they belong; only their bucket membership is loose. The
/// exactly-zero structural sets are pinned by EXACT equality in `nn/cells/mamba.rs`'s
/// own unit tests (`a_log_gradient_is_exactly_zero_at_t1`,
/// `old_conv_taps_are_gradient_dead_below_the_kernel_length`), which is a strictly
/// stronger instrument than any threshold count.
///
/// `rel_pin` is PER-SHAPE (measured*10 for that shape), not a single global bound:
/// the central-difference noise floor is `ulp(|L|) / (2 eps)` and `|L|` grows with
/// `T` and the layer width, so one global pin is set by the loosest shape and gives
/// every tighter shape a free pass -- a real regression at `T = 1` could grow by two
/// orders of magnitude and still sit under a bound calibrated at `T = 23`.
///
/// `eps` is PER-SHAPE for the same reason, and it is the one knob that moves the
/// noise floor itself: the central-difference error is `ulp(|L|)/(2 eps)` (roundoff,
/// falling with `eps`) plus `eps^2 f'''/6` (truncation, rising with `eps`). sLSTM's
/// gradients are `O(1)`, so `1e-6` sits comfortably in the valley. Mamba's do NOT:
/// it is a RESIDUAL block, its branch output is a product of five small factors at
/// small random init, and its live derivatives run from `1e-2` down to `1e-12` -- the
/// weights that set `max_rel` are always the smallest resolvable ones, a few hundred
/// ULP above the floor, so mamba trades a little truncation for two orders of
/// magnitude less roundoff. MEASURED at the `(23,6,6,8,4,2)` shape, worst seed:
/// `eps 1e-6 -> 4.59e-2`, `1e-4 -> 5.75e-4`, `1e-3 -> 1.20e-5`, `1e-2 -> 6.65e-5`
/// (the valley bottom is near `1e-3`; `1e-2` is already climbing the truncation
/// side). Documented per the task brief's explicit allowance for a per-case eps.
struct Case {
    t: usize,
    input_size: usize,
    output_size: usize,
    resolvable_floor: usize,
    resolvable_structural: usize,
    rel_pin: f64,
    /// Pin on [`FdReport::max_rel_major`] -- the discriminating bound (see
    /// [`MAJOR_FRACTION`]). Always far tighter than `rel_pin`.
    major_pin: f64,
    eps: f64,
}

/// Run `fd_check` over a shape x seed grid, asserting BOTH regimes plus the
/// resolvable count, and return `(worst relative, worst near-zero absolute, worst
/// absolute)`.
///
/// CELL-AGNOSTIC: `make(input_size, output_size)` is the only cell-specific thing in
/// this file's machinery, so Task 3 pins `MambaLayer` by calling this with its own
/// constructor, its own case table and its own measured pins -- no edit here. A cell
/// whose geometry needs more than `(in, out)` (mamba's `d_state`/`d_conv`/`expand`/
/// `dt_rank`) closes over them in `make` and calls `sweep` once per geometry.
fn sweep(
    label: &str,
    make: impl Fn(usize, usize) -> Box<dyn Layer>,
    cases: &[Case],
    near_zero_pin: f64,
    abs_pin: f64,
) -> (f64, f64, f64) {
    let (mut worst_rel, mut worst_abs, mut worst_err) = (0.0_f64, 0.0_f64, 0.0_f64);
    for case in cases {
        let (t, i, o) = (case.t, case.input_size, case.output_size);
        let (resolvable_floor, resolvable_structural, rel_pin) = (
            case.resolvable_floor,
            case.resolvable_structural,
            case.rel_pin,
        );
        let major_pin = case.major_pin;
        for seed in [1u64, 2, 3] {
            let mut cell = make(i, o);
            let r = fd_check(cell.as_mut(), t, i, o, seed, case.eps);
            // The measure-then-pin instrument: `cargo test -- --nocapture` re-prints
            // every number quoted in the pin comments below.
            println!(
                "FD {label} t={t} in={i} out={o} seed={seed} eps={:e}: max_rel={:e} at w[{}] \
                 (fd {:e} vs an {:e}) | near-zero max_abs={:e} at w[{}] | \
                 resolvable {}/{} | max_abs_err={:e} | max|an|={:e} | \
                 major {} max_rel_major={:e} at w[{}]",
                case.eps,
                r.max_rel,
                r.arg_max,
                r.worst_pair.0,
                r.worst_pair.1,
                r.max_abs_near_zero,
                r.arg_max_near_zero,
                r.resolvable,
                r.total,
                r.max_abs_err,
                r.max_abs_analytic,
                r.major,
                r.max_rel_major,
                r.arg_max_major
            );
            let at = format!("{label} (t={t}, in={i}, out={o}, seed={seed})");
            assert!(
                r.max_abs_analytic > 1e-6,
                "{at}: the analytic gradient is all ~zero -- the check is vacuous"
            );
            assert!(
                r.resolvable <= resolvable_structural,
                "{at}: {} of {} weights have a resolvable derivative, more than the \
                 structural maximum {resolvable_structural} -- a weight that must be \
                 EXACTLY zero by structure became live, which is a CONTRACT change, not a \
                 tolerance question",
                r.resolvable,
                r.total
            );
            assert!(
                r.resolvable >= resolvable_floor,
                "{at}: only {} of {} weights have a resolvable derivative, below the floor \
                 {resolvable_floor} -- the gradient collapsed into the lenient near-zero \
                 bucket",
                r.resolvable,
                r.total
            );
            assert!(
                r.max_rel < rel_pin,
                "{at}: max rel {:e} >= pin {rel_pin:e} at weight {} (fd {:e} vs analytic {:e})",
                r.max_rel,
                r.arg_max,
                r.worst_pair.0,
                r.worst_pair.1
            );
            // PROPORTIONATE non-vacuity: the MAJOR set must be at least a quarter of the
            // resolvable one, not merely non-empty. A bare `>= 2` would keep passing while
            // the discriminating metric quietly shrank onto a handful of weights.
            // MEASURED ratio over this whole grid (both cells, all seeds): 0.66 worst
            // (mamba t=9 in=4 out=6, major 331 of 504 resolvable), 0.94+ on every sLSTM
            // leg -- so the 0.25 bar carries ~2.6x headroom at the tightest point.
            assert!(
                r.major * 4 >= r.resolvable,
                "{at}: only {} MAJOR weights of {} resolvable -- max_rel_major is computed \
                 over too small a set to mean anything",
                r.major,
                r.resolvable
            );
            assert!(
                r.max_rel_major < major_pin,
                "{at}: max rel over MAJOR weights {:e} >= pin {major_pin:e} at weight {} \
                 -- this is the discriminating bound; a wrong adjoint term lands HERE, \
                 not in the near-zero noise",
                r.max_rel_major,
                r.arg_max_major
            );
            assert!(
                r.max_abs_near_zero < near_zero_pin,
                "{at}: near-zero max abs err {:e} >= pin {near_zero_pin:e} at weight {}",
                r.max_abs_near_zero,
                r.arg_max_near_zero
            );
            assert!(
                r.max_abs_err < abs_pin,
                "{at}: max abs err over ALL weights {:e} >= pin {abs_pin:e}",
                r.max_abs_err
            );
            worst_rel = worst_rel.max(r.max_rel);
            worst_abs = worst_abs.max(r.max_abs_near_zero);
            worst_err = worst_err.max(r.max_abs_err);
        }
    }
    (worst_rel, worst_abs, worst_err)
}

// ---------------------------------------------------------------------------
// sLSTM (Task 2)
// ---------------------------------------------------------------------------

/// THE backward pin (spec S8.1 unit tier): central difference vs the hand-derived
/// analytic gradient over `(t, in, out) in {(1,3,2), (7,3,2), (11,5,4), (23,7,3)}`
/// x 3 seeds, `eps = 1e-6`.
///
/// MEASURED (Apple M4 Pro, f64, this exact seed grid), re-printable with
/// `cargo test --release --test phase9_cell_grad -- --nocapture`:
///
/// ```text
/// PER-SHAPE max relative error (resolvable weights), worst of the 3 seeds:
///   t= 1 in=3 out=2 -> 4.414e-8  (seed 1, w[30])   t= 7 in=3 out=2 -> 8.969e-7 (seed 1, w[0])
///   t=11 in=5 out=4 -> 6.423e-7  (seed 1, w[81])   t=23 in=7 out=3 -> 1.620e-6 (seed 2, w[48])
/// max absolute error (near-zero weights)  = 2.220e-10 at t=23/in=7/out=3, seed 2, w[30]
/// max absolute error (ALL weights)        = 1.360e-9  at t=23/in=7/out=3, seed 2
/// ```
///
/// Pinned at measured*10 PER SHAPE (`4.5e-7`/`9.0e-6`/`6.5e-6`/`1.7e-5` relative --
/// see [`Case`] for why one global bound is the wrong instrument), plus `2.5e-9`
/// near-zero and `1.4e-8` all-weights absolute (spec R4); the loosest relative pin
/// sits ~6x below the brief's 1e-4 STOP threshold. EVERY figure here is the
/// central-difference noise floor `ulp(|L|)/(2 eps)` (`|L| ~ 2`, `eps = 1e-6` ->
/// ~2.2e-10 per ULP), NOT a gradient error: the worst relative point is a genuinely
/// small derivative (5.87e-4) sitting a few ULP above that floor, and the worst
/// absolute discrepancy anywhere is 1.360e-9 against gradients reaching ~2.0. A wrong
/// adjoint term would show up as a relative error of order 1e-1, not 1e-6.
#[test]
fn slstm_backward_matches_central_difference() {
    // Resolvable counts (structural, seed-independent -- see the module doc). sLSTM's
    // zero set is a clean block, so floor == structural on every row and the two-sided
    // assert of `Case` degenerates to the exact equality it had before the mamba legs
    // landed:
    // T=1 zeroes every recurrent block (h_{-1} = 0), the whole f block (f' = 0
    // against the zero state) and the whole i block (m_0 = i~_0 makes i' = exp(0) = 1
    // whatever i~ is), leaving only the o and z blocks' W+b: 2*out*(in+1).
    // T>1 zeroes only b_i (the structural degeneracy): 4*out*(out+in+1) - out.
    let slstm_case =
        |t: usize, i: usize, o: usize, resolvable: usize, rel_pin: f64, major_pin: f64| Case {
            t,
            input_size: i,
            output_size: o,
            // floor == structural: sLSTM's zero set is a clean block, so the two-sided
            // count assert of `Case` degenerates to the exact equality it had before the
            // mamba legs landed.
            resolvable_floor: resolvable,
            resolvable_structural: resolvable,
            rel_pin,
            major_pin,
            eps: 1e-6,
        };
    let cases: &[Case] = &[
        slstm_case(1, 3, 2, 2 * 2 * (3 + 1), 4.5e-7, 4.5e-7),
        slstm_case(7, 3, 2, 4 * 2 * (2 + 3 + 1) - 2, 9.0e-6, 9.0e-6),
        slstm_case(11, 5, 4, 4 * 4 * (4 + 5 + 1) - 4, 6.5e-6, 3.2e-6),
        slstm_case(23, 7, 3, 4 * 3 * (3 + 7 + 1) - 3, 1.7e-5, 1.7e-5),
    ];
    let (worst_rel, worst_abs, worst_err) = sweep(
        "slstm",
        |i, o| Box::new(SlstmLayer::new(i, o)),
        cases,
        2.5e-9,
        1.4e-8,
    );
    assert!(
        worst_rel > 0.0 && worst_abs > 0.0 && worst_err > 0.0,
        "a whole regime came back at exactly 0 error -- suspicious"
    );
}

// ---------------------------------------------------------------------------
// Mamba (Task 3)
// ---------------------------------------------------------------------------

/// `MambaLayer::nb_of_weights` spelled out independently of the layer, so the
/// resolvable-count arithmetic below is checked against a SECOND derivation of the
/// S3.2 layout rather than against the thing under test.
fn mamba_total(i: usize, o: usize, d_state: usize, d_conv: usize, expand: usize) -> usize {
    let di = expand * o;
    let dr = o.div_ceil(16).max(1); // dt_rank = 0 -> auto
    let adapter = if i != o { o * i + o } else { 0 };
    adapter                          // [P | p]
        + o                          // g
        + 2 * di * o                 // W_in
        + di * d_conv + di           // conv | b_conv
        + (dr + 2 * d_state) * di    // W_x
        + di * dr + di               // W_dt | b_dt
        + di * d_state               // A_log
        + di                         // D
        + o * di // W_out
}

/// Structurally gradient-dead weights at `(T, d_conv, d_state, d_inner)` -- see
/// `nn/cells/mamba.rs`'s module doc, and `a_log_gradient_is_exactly_zero_at_t1` /
/// `old_conv_taps_are_gradient_dead_below_the_kernel_length` there for the direct
/// pins:
///
/// - `A_log` (`d_inner * d_state`) is dead at `T == 1`: `dAbar_t = dh_t . h_{t-1}`
///   and `h_{-1} = 0`, so at a single step `Abar` multiplies nothing.
/// - the oldest `max(0, d_conv - T)` conv taps per channel are dead: tap `k` first
///   reaches a real sample at `t = d_conv - 1 - k`, so taps `k < d_conv - T` only
///   ever multiply the causal left-padding.
///
/// Everything else is resolvable at every shape below -- notably there is NO mamba
/// analogue of the sLSTM `b_i` degeneracy: RMSNorm's `eps = 1e-5` and the `+ x'`
/// residual both break the would-be scale invariance of the normalizer.
fn mamba_dead(t: usize, d_state: usize, d_conv: usize, d_inner: usize) -> usize {
    let a_log_dead = if t == 1 { d_inner * d_state } else { 0 };
    let conv_dead = d_inner * d_conv.saturating_sub(t);
    a_log_dead + conv_dead
}

/// THE backward pin (spec S8.1 unit tier): central difference vs the hand-derived
/// analytic gradient over `(t, in, out, d_state, d_conv, expand)` in
/// `{(1,3,3,2,2,1), (7,3,3,4,3,2), (9,4,6,4,4,2) adapter, (23,6,6,8,4,2)}` x 3 seeds,
/// `eps = 1e-6`, `dt_rank = 0` (auto -> `ceil(d_model/16)` = 1 at these widths).
///
/// One `sweep` call per geometry: the shared [`Case`] row carries only `(t, in, out)`,
/// so mamba's four extra hyperparameters close over the `make` constructor instead
/// (the harness stays untouched -- spec S8.1's cell-agnostic contract).
///
/// The grid covers BOTH width regimes deliberately: `in == out` (no `[P | p]` block
/// exists at all) on three shapes and `in != out` (the adapter present, 30 extra
/// weights, and the residual adding `x'` rather than `x`) on `(9,4,6,...)`.
///
/// [`MAMBA_EPS`] is 5e-4, not the sLSTM legs' 1e-6, and that is measured rather than
/// assumed. Mamba's gradients are 100-1000x smaller than sLSTM's (a residual block at
/// small init), so the roundoff floor `ulp(|L|)/(2 eps)` bites much higher up its
/// magnitude range. Worst `max_rel` over the whole grid, by eps:
///
/// ```text
/// 1e-6 -> 4.59e-2   1e-4 -> 1.18e-3   3e-4 -> 1.99e-4   5e-4 -> 7.67e-5
/// 1e-3 -> 1.68e-4   2e-3 -> 1.18e-4   5e-3 -> 7.36e-4   1e-2 -> 2.94e-3
/// ```
///
/// -- a clean roundoff/truncation valley bottoming near `5e-4`, which is also the
/// only setting putting EVERY resolvable-weight point under the brief's 1e-4 STOP
/// threshold. `max_rel_major` is far less eps-sensitive (3.50e-6 at 1e-6 vs 7.36e-6
/// at 5e-4): the DOMINANT weights were always fine, which is exactly the point.
///
/// MEASURED (Apple M4 Pro, f64, this exact seed grid, eps [`MAMBA_EPS`]),
/// re-printable with
/// `cargo test --release --test phase9_cell_grad -- --nocapture`:
///
/// ```text
/// PER-SHAPE, worst of the 3 seeds:              max_rel     max_rel_major   resolvable
///   t= 1 in=3 out=3 ds=2 dc=2 ex=1              5.813e-6    3.029e-7        51..60 of 69
///   t= 7 in=3 out=3 ds=4 dc=3 ex=2              2.595e-5    4.040e-6       175..177 of 177
///   t= 9 in=4 out=6 ds=4 dc=4 ex=2 (adapter)    4.507e-6    4.507e-6       504 of 504
///   t=23 in=6 out=6 ds=8 dc=4 ex=2              7.672e-5    7.364e-6       618 of 618
/// max absolute error (near-zero weights) = 5.105e-13 at t=7, seed 2, w[52]
/// max absolute error (ALL weights)       = 2.333e-8  at t=23, seed 2
/// ```
///
/// Pinned at measured*10 PER SHAPE: `6.0e-5`/`2.6e-4`/`4.6e-5`/`7.7e-4` relative and
/// `3.1e-6`/`4.1e-5`/`4.6e-5`/`7.4e-5` MAJOR, plus `5.2e-12` near-zero and `2.4e-7`
/// all-weights absolute (both grid-wide, the sLSTM convention). NO STOP: every
/// measured `max_rel` is below the task brief's 1e-4 threshold, and the
/// discriminating `max_rel_major` is <= 7.4e-6 everywhere -- a wrong adjoint term
/// would show up THERE at order 1e-1, not 1e-6.
///
/// The `resolvable_floor`s (`49`/`173`/`502`/`616`) are the measured minima minus 2:
/// a couple of counts of slack for a weight sitting within a few ULP of the absolute
/// [`NEAR_ZERO`] line flipping bucket on a different libm. The UPPER bound is the
/// DERIVED structural count and carries no slack at all -- that is the side a
/// layout/indexing bug would break.
///
/// The measured spread between `max_rel` and `max_rel_major` is the whole reason the
/// latter exists: at `t=23` the unrestricted worst point is `|analytic| = 1.21e-8`
/// (1.6e-7 of the pack maximum), i.e. a derivative a few hundred ULP above the
/// central-difference floor, while every weight within `1e-4` of the maximum agrees
/// to 7.4e-6 or better.
///
/// WHICH BOUND IS THE GUARD, said plainly: `major_pin` is. Every `major_pin` on the grid
/// (3.1e-6 / 4.1e-5 / 4.6e-5 / 7.4e-5) sits under the brief's 1e-4 STOP threshold, and
/// that is the assertion a wrong adjoint term trips. `rel_pin` is NOT a sanctioned error
/// budget -- two of its rows (2.6e-4 at `t=7` and 7.7e-4 at `t=23`) are ABOVE 1e-4 on
/// purpose, because they are set by the smallest resolvable weights, whose relative error
/// is the central-difference floor divided by a near-zero number and says nothing about
/// the derivation. Reading `rel_pin` as "the gradient is accurate to 7.7e-4" would be
/// wrong in both directions: the dominant weights are 100x better than that, and the tiny
/// ones are not measurable at all. `rel_pin` exists only to catch a gross regime shift.
const MAMBA_EPS: f64 = 5e-4;

/// One mamba grid row. NAMED, for the same reason [`Case`] is: nine positional fields --
/// six adjacent `usize` (four of them geometry, one a length, one a count) and two
/// adjacent `f64` pins -- is precisely the shape a transposed-argument bug hides in, and
/// unlike `Case` this one carries the geometry the layer is CONSTRUCTED from, so a
/// silent swap would build a different net rather than fail a bound.
struct MambaCase {
    t: usize,
    input_size: usize,
    output_size: usize,
    d_state: usize,
    d_conv: usize,
    expand: usize,
    resolvable_floor: usize,
    rel_pin: f64,
    major_pin: f64,
}

#[test]
fn mamba_backward_matches_central_difference() {
    let row = |t: usize,
               input_size: usize,
               output_size: usize,
               d_state: usize,
               d_conv: usize,
               expand: usize,
               resolvable_floor: usize,
               rel_pin: f64,
               major_pin: f64| MambaCase {
        t,
        input_size,
        output_size,
        d_state,
        d_conv,
        expand,
        resolvable_floor,
        rel_pin,
        major_pin,
    };
    let grid: &[MambaCase] = &[
        row(1, 3, 3, 2, 2, 1, 49, 6.0e-5, 3.1e-6),
        row(7, 3, 3, 4, 3, 2, 173, 2.6e-4, 4.1e-5),
        row(9, 4, 6, 4, 4, 2, 502, 4.6e-5, 4.6e-5),
        row(23, 6, 6, 8, 4, 2, 616, 7.7e-4, 7.4e-5),
    ];
    let (mut worst_rel, mut worst_abs, mut worst_err) = (0.0_f64, 0.0_f64, 0.0_f64);
    for g in grid {
        let (ds, dc, ex) = (g.d_state, g.d_conv, g.expand);
        let total = mamba_total(g.input_size, g.output_size, ds, dc, ex);
        let resolvable = total - mamba_dead(g.t, ds, dc, ex * g.output_size);
        let label = format!("mamba ds={ds} dc={dc} ex={ex}");
        let cases: &[Case] = &[Case {
            t: g.t,
            input_size: g.input_size,
            output_size: g.output_size,
            resolvable_floor: g.resolvable_floor,
            resolvable_structural: resolvable,
            rel_pin: g.rel_pin,
            major_pin: g.major_pin,
            eps: MAMBA_EPS,
        }];
        let (r, a, e) = sweep(
            &label,
            |i, o| Box::new(MambaLayer::new(i, o, ds, dc, ex, 0)),
            cases,
            5.2e-12,
            2.4e-7,
        );
        worst_rel = worst_rel.max(r);
        worst_abs = worst_abs.max(a);
        worst_err = worst_err.max(e);
    }
    assert!(
        worst_rel > 0.0 && worst_abs > 0.0 && worst_err > 0.0,
        "a whole regime came back at exactly 0 error -- suspicious"
    );
}
