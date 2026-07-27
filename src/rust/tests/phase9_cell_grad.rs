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
use speech::nn::cells::SlstmLayer;
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

    let mut report = FdReport {
        max_rel: 0.0,
        arg_max: 0,
        worst_pair: (0.0, 0.0),
        max_abs_near_zero: 0.0,
        arg_max_near_zero: 0,
        resolvable: 0,
        total: nb,
        max_abs_err: 0.0,
        max_abs_analytic: 0.0,
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
        } else if err > report.max_abs_near_zero {
            report.max_abs_near_zero = err;
            report.arg_max_near_zero = k;
        }
        report.max_abs_err = report.max_abs_err.max(err);
        report.max_abs_analytic = report.max_abs_analytic.max(an.abs());
    }
    report
}

/// One row of the sweep grid: `(t, input_size, output_size, resolvable_weights)`.
/// The last field is the PINNED count of weights whose derivative is resolvable
/// (`> NEAR_ZERO`) -- structural, seed-independent, and the guard that stops a
/// backward from passing by collapsing everything into the near-zero bucket.
type Case = (usize, usize, usize, usize);

/// Run `fd_check` over a shape x seed grid, asserting BOTH regimes plus the
/// resolvable count, and return `(worst relative, worst near-zero absolute)`.
fn sweep(cases: &[Case], rel_pin: f64, near_zero_pin: f64, abs_pin: f64) -> (f64, f64, f64) {
    let (mut worst_rel, mut worst_abs, mut worst_err) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &(t, i, o, resolvable) in cases {
        for seed in [1u64, 2, 3] {
            let mut cell = SlstmLayer::new(i, o);
            let r = fd_check(&mut cell, t, i, o, seed, 1e-6);
            // The measure-then-pin instrument: `cargo test -- --nocapture` re-prints
            // every number quoted in the pin comments below.
            println!(
                "FD slstm t={t} in={i} out={o} seed={seed}: max_rel={:e} at w[{}] \
                 (fd {:e} vs an {:e}) | near-zero max_abs={:e} at w[{}] | \
                 resolvable {}/{} | max_abs_err={:e} | max|an|={:e}",
                r.max_rel,
                r.arg_max,
                r.worst_pair.0,
                r.worst_pair.1,
                r.max_abs_near_zero,
                r.arg_max_near_zero,
                r.resolvable,
                r.total,
                r.max_abs_err,
                r.max_abs_analytic
            );
            let at = format!("(t={t}, in={i}, out={o}, seed={seed})");
            assert!(
                r.max_abs_analytic > 1e-6,
                "{at}: the analytic gradient is all ~zero -- the check is vacuous"
            );
            assert_eq!(
                r.resolvable, resolvable,
                "{at}: {} of {} weights have a resolvable derivative, expected {resolvable} \
                 -- the structurally-zero set moved, which is a CONTRACT change, not a \
                 tolerance question",
                r.resolvable, r.total
            );
            assert!(
                r.max_rel < rel_pin,
                "{at}: max rel {:e} >= pin {rel_pin:e} at weight {} (fd {:e} vs analytic {:e})",
                r.max_rel,
                r.arg_max,
                r.worst_pair.0,
                r.worst_pair.1
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
/// max relative error (resolvable weights) = 1.620e-6  at t=23/in=7/out=3, seed 2, w[48]
/// max absolute error (near-zero weights)  = 2.220e-10 at t=23/in=7/out=3, seed 2, w[30]
/// max absolute error (ALL weights)        = 1.360e-9  at t=23/in=7/out=3, seed 2
/// ```
///
/// Pinned at measured*10 -> `1.7e-5` relative, `2.5e-9` near-zero, `1.4e-8`
/// all-weights absolute (spec R4); the
/// relative pin sits ~6x below the brief's 1e-4 STOP threshold. EVERY figure here is
/// the central-difference noise floor `ulp(|L|)/(2 eps)` (`|L| ~ 2`, `eps = 1e-6` ->
/// ~2.2e-10 per ULP), NOT a gradient error: the worst relative point is a genuinely
/// small derivative (5.87e-4) sitting a few ULP above that floor, and the worst
/// absolute discrepancy anywhere is 9.5e-10 against gradients reaching ~2.0. A wrong
/// adjoint term would show up as a relative error of order 1e-1, not 1e-6.
#[test]
fn slstm_backward_matches_central_difference() {
    // Resolvable counts (structural, seed-independent -- see the module doc):
    // T=1 zeroes every recurrent block (h_{-1} = 0), the whole f block (f' = 0
    // against the zero state) and the whole i block (m_0 = i~_0 makes i' = exp(0) = 1
    // whatever i~ is), leaving only the o and z blocks' W+b: 2*out*(in+1).
    // T>1 zeroes only b_i (the structural degeneracy): 4*out*(out+in+1) - out.
    let cases: &[Case] = &[
        (1, 3, 2, 2 * 2 * (3 + 1)),
        (7, 3, 2, 4 * 2 * (2 + 3 + 1) - 2),
        (11, 5, 4, 4 * 4 * (4 + 5 + 1) - 4),
        (23, 7, 3, 4 * 3 * (3 + 7 + 1) - 3),
    ];
    let (worst_rel, worst_abs, worst_err) = sweep(cases, 1.7e-5, 2.5e-9, 1.4e-8);
    assert!(
        worst_rel > 0.0 && worst_abs > 0.0 && worst_err > 0.0,
        "a whole regime came back at exactly 0 error -- suspicious"
    );
}
