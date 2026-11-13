//! The fast tree's f32 product kernel (issue #147): the matvec behind every causal
//! cell projection and behind the bidirectional LSTM twin's recurrence.
//!
//! It is the f32 TRANSCRIPTION of `nn::kernels`' `chains` / `chain_block` nest, not a
//! shared generic: the two trees share no kernel (ADR-0002, ADR-0003), so a change to
//! either nest cannot move the other tree.
//!
//! # The contract
//!
//! Per output element, whatever the number of outputs advanced alongside it:
//!
//! - **the seed** is `+0.0`: the element starts at `+0.0` and its first term is ADDED
//!   to it, never assigned (`0.0 + (-0.0)` is `+0.0`, the term alone is `-0.0`);
//! - **the summation order** is ascending input index, one add per term;
//! - **no fused multiply-add** (`mul_add` is a workspace `disallowed-methods` entry;
//!   Rust never contracts `a * b + c` on its own).
//!
//! That is the chain the per-output dot this kernel replaced computed, so no value
//! moved. The multiply is written weight times input; an f32 product does not depend
//! on its operand order, the payload of a NaN aside.
//!
//! What the nest buys is the innermost loop running over INDEPENDENT outputs on a
//! contiguous slice, which the compiler vectorizes: `k` outer over a block of adjacent
//! outputs (16, then 8, 4, 2, 1) whose accumulators are a fixed-size local held in
//! registers. That needs the weight K-MAJOR, one row of outputs per input index
//! ([`k_major`], built once at `from_flat`). A plain `k`-outer nest with the output
//! slice as the accumulator was measured and not used: a load and a store per term, it
//! loses to the per-output dot under 8 outputs.
//!
//! The result of one output does not depend on the block it falls in, on the output
//! count or on a row count, there being none: unlike `faer_project`, whose per-element
//! result depends on how many rows it is handed (`DenseRowChain`'s docs), this kernel
//! is safe inside a streamed step.
//!
//! # The proof
//!
//! The differential leg at the bottom of this file compares the kernel with the
//! per-output dot it replaced (kept there as `dot_f32`) by `to_bits` over random
//! shapes and values, a cropped input, signed zeros, infinities and subnormals
//! included; NaN elements compare by `is_nan`. A symmetric edit to the kernel and its
//! reference is invisible to the leg. How a step COMBINES its products is not this
//! file's to pin: `tests/issue147_fast_bits.rs` owns the step bodies.

/// A row-major `outputs x inputs` weight (one contiguous row per output) as the
/// k-major `inputs x outputs` buffer the kernel reads: `kt[k * outputs + j] ==
/// w[j * inputs + k]`.
pub(crate) fn k_major(w: &[f32], outputs: usize, inputs: usize) -> Vec<f32> {
    assert_eq!(w.len(), outputs * inputs, "k_major weight length");
    let mut kt = vec![0.0_f32; w.len()];
    for j in 0..outputs {
        for k in 0..inputs {
            kt[k * outputs + j] = w[j * inputs + k];
        }
    }
    kt
}

/// `out[j] = sum_k wt[k][j] * x[k]` for every `j`, `wt` k-major with `out.len()`
/// columns. `x` may be SHORTER than the weight's input count: only its `x.len()`
/// leading rows are read, the width-tolerance crop (fewer terms, the same chain).
/// `out` is overwritten.
#[inline]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "first caller: the causal cells, next commit")
)]
pub(crate) fn matvec_into(out: &mut [f32], wt: &[f32], x: &[f32]) {
    chains::<false>(out, wt, x);
}

/// [`matvec_into`], ADDED to `out`: each output's chain is summed from `+0.0` on its
/// own and added to `out[j]` once, which is `out[j] += dot(row_j, x)`.
#[inline]
pub(crate) fn matvec_add_into(out: &mut [f32], wt: &[f32], x: &[f32]) {
    chains::<true>(out, wt, x);
}

/// The one loop nest: the outputs in blocks of 16, then 8, 4, 2 and 1 adjacent
/// columns, each block summed by [`chain_block`].
#[inline(always)]
fn chains<const ADD: bool>(out: &mut [f32], wt: &[f32], x: &[f32]) {
    let n = out.len();
    if n == 0 {
        return;
    }
    assert!(
        wt.len() >= x.len() * n && wt.len().is_multiple_of(n),
        "fast matvec weight length"
    );
    let mut j = 0;
    while n - j >= 16 {
        chain_block::<16, ADD>(out, wt, x, j);
        j += 16;
    }
    if n - j >= 8 {
        chain_block::<8, ADD>(out, wt, x, j);
        j += 8;
    }
    if n - j >= 4 {
        chain_block::<4, ADD>(out, wt, x, j);
        j += 4;
    }
    if n - j >= 2 {
        chain_block::<2, ADD>(out, wt, x, j);
        j += 2;
    }
    if n - j >= 1 {
        chain_block::<1, ADD>(out, wt, x, j);
    }
}

/// `W` adjacent outputs from column `j0`: each starts at `+0.0` and takes its terms in
/// ascending `k`, one add per term, the `W` chains advancing side by side in a
/// fixed-size local.
#[inline(always)]
fn chain_block<const W: usize, const ADD: bool>(out: &mut [f32], wt: &[f32], x: &[f32], j0: usize) {
    let stride = out.len();
    let mut acc = [0.0_f32; W];
    for (&x_k, row) in x.iter().zip(wt.chunks_exact(stride)) {
        let block: &[f32; W] = row[j0..j0 + W].try_into().expect("block width");
        for (a, &w) in acc.iter_mut().zip(block) {
            *a += w * x_k;
        }
    }
    let dst = &mut out[j0..j0 + W];
    if ADD {
        for (d, &a) in dst.iter_mut().zip(&acc) {
            *d += a;
        }
    } else {
        dst.copy_from_slice(&acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-output ascending dot every fast projection ran before issue #147, kept
    /// as the reference the kernel is compared with.
    fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0_f32;
        for k in 0..a.len() {
            acc += a[k] * b[k];
        }
        acc
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        /// A value in `[-1, 1)`, or with probability `special / 16` one of the
        /// values a plain range never produces.
        fn value(&mut self, special: u64) -> f32 {
            let r = self.next();
            if r % 16 < special {
                const SPECIALS: [f32; 9] = [
                    0.0,
                    -0.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    f32::MIN_POSITIVE,
                    1.0e-42,
                    -1.0e-42,
                    f32::MAX,
                ];
                SPECIALS[(self.next() % SPECIALS.len() as u64) as usize]
            } else {
                (self.next() % (1 << 24)) as f32 / (1 << 23) as f32 - 1.0
            }
        }
    }

    fn assert_same(got: f32, want: f32, what: &str) {
        if want.is_nan() {
            assert!(got.is_nan(), "{what}: {got} is not NaN");
        } else {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{what}: {got:e} against {want:e}"
            );
        }
    }

    /// Both entry points against the per-output dot, over every output count that
    /// exercises a block mix (1..=40, plus the real widths), a spread of input counts,
    /// a cropped input, and values with and without the special ones.
    #[test]
    fn the_kernel_matches_the_per_output_dot() {
        let mut rng = Lcg(0x147);
        let outputs: Vec<usize> = (1..=40).chain([48, 92, 96, 192]).collect();
        for &n in &outputs {
            for &k in &[0usize, 1, 2, 3, 5, 12, 23, 24, 64] {
                for special in [0, 3] {
                    let w: Vec<f32> = (0..n * k).map(|_| rng.value(special)).collect();
                    let wt = k_major(&w, n, k);
                    for crop in [0, 1, 4] {
                        let k_in = k.saturating_sub(crop);
                        let x: Vec<f32> = (0..k_in).map(|_| rng.value(special)).collect();
                        let base: Vec<f32> = (0..n).map(|_| rng.value(special)).collect();

                        let mut got = vec![f32::NAN; n];
                        matvec_into(&mut got, &wt, &x);
                        let mut added = base.clone();
                        matvec_add_into(&mut added, &wt, &x);

                        for j in 0..n {
                            let want = dot_f32(&w[j * k..j * k + k_in], &x);
                            let what = format!("{n} x {k} (crop {crop}), output {j}");
                            assert_same(got[j], want, &what);
                            assert_same(added[j], base[j] + want, &format!("{what}, added"));
                        }
                    }
                }
            }
        }
    }

    /// The first term is added to a `+0.0` seed, never assigned: a chain of `-0.0`
    /// products sums to `+0.0`, at every block width, and so does an empty chain.
    #[test]
    fn every_output_is_seeded_positive_zero() {
        for n in 1..=40 {
            for k in [0usize, 1, 3] {
                let wt = vec![-0.0_f32; n * k];
                let x = vec![1.0_f32; k];
                let mut out = vec![f32::NAN; n];
                matvec_into(&mut out, &wt, &x);
                for (j, v) in out.iter().enumerate() {
                    assert_eq!(v.to_bits(), 0.0_f32.to_bits(), "{n} x {k}, output {j}");
                }
            }
        }
    }

    #[test]
    fn k_major_is_the_transpose() {
        let (n, k) = (3, 5);
        let w: Vec<f32> = (0..n * k).map(|v| v as f32).collect();
        let wt = k_major(&w, n, k);
        for j in 0..n {
            for kk in 0..k {
                assert_eq!(wt[kk * n + j], w[j * k + kk]);
            }
        }
    }
}
