//! The exact tree's product kernels (issue #144): the matrix product, its
//! caller-buffer variant, the transposed matvec for the `n = 1` products and the
//! rank-1 derivative accumulation. The LSTM, the dense layer, the sLSTM and the CfC
//! compute every product here; Mamba and the transformer take `matmul_seq` for
//! their forward products and keep their explicit backward loops.
//!
//! # The contract
//!
//! These are ORDER-PRESERVING LOOP RESTRUCTURES (ADR-0002): the loop nest differs
//! from the legacy `matSeq` (`i`/`j` outer, `k` inner), the value of every output
//! element does not. Per output element:
//!
//! - **the seed** is `+0.0`: the element starts at `+0.0` and the first term is
//!   ADDED to it. It is never initialized from its first term (`0.0 + (-0.0)` is
//!   `+0.0`, the term alone is `-0.0`);
//! - **the operand order** of each multiply is the legacy's (`a[i][k] * b[k][j]`,
//!   weight times delta in the matvec, input times delta in the rank-1 add);
//! - **the summation order** is ascending `k`, one add per term;
//! - **no fused multiply-add**: `mul_add` is forbidden in this module and
//!   workspace-wide by `clippy.toml` (`disallowed-methods`). Rust never contracts
//!   `a * b + c` on its own, and a fused product rounds once where the legacy
//!   rounds twice.
//!
//! What the restructure buys is the innermost loop running over INDEPENDENT output
//! elements on contiguous slices, which the compiler vectorizes; the legacy nest's
//! innermost loop is one dependent add chain per element, which it cannot. The nest
//! is `k` outer over a BLOCK of adjacent outputs (`chains`: 16, then 8, 4, 2, 1),
//! the block's accumulators held in registers: for a product that is `i`, block,
//! `k`, `j`; for a matvec the same over a transposed weight. A plain `i-k-j` nest
//! with the output row as the accumulator was measured first and dropped: it wins on
//! wide outputs and is 2 to 6 times slower than the legacy nest on outputs under 8
//! wide, each term paying a load and a store.
//!
//! Ascending `k` per element is the measured product-order contract (Phase 2 Task 1
//! probes: Eigen DIVERGES from ascending loops on the LSTM input-projection shape and
//! is bit-exact on the recurrence row-product; the harness reimpl uses `matSeq` for
//! both, so the port mirrors it uniformly).
//!
//! # The dense layer's outer product
//!
//! `NeuronLayer::feed_backward` used to build each frame's `I x O` outer product
//! through `matmul_seq` and add it to its accumulator, so every term reached the
//! accumulator as `0.0 + x*d`. [`rank1_add`] adds `x*d` directly. The two are equal
//! because the accumulator is a per-call local seeded `+0.0`: under round-to-nearest
//! a sum seeded `+0.0` is never `-0.0` (only `-0.0 + -0.0` yields it), and adding
//! `+0.0` or `-0.0` to anything else gives the same bits. The LSTM's four
//! accumulators are seeded the same way and always added `x*d` directly.
//!
//! # The proof
//!
//! The committed goldens and issue #146's derivative pins stay byte-green with no
//! test edited, and the differential leg at the bottom of this file compares every
//! kernel with a scalar reference (the legacy nest, kept there as `reference_matmul`)
//! by `to_bits` over random shapes and values, signed zeros, infinities and
//! subnormals included; NaN elements compare by `is_nan`, the payload of a NaN sum
//! being the compiler's choice of operand order. A symmetric edit to a kernel and
//! its reference is invisible to the leg: the goldens own that, and this doc is the
//! rule to review against.

use ndarray::{Array2, ArrayView2};

/// Ascending-`k` matrix product `a (m x k) * b (k x n)`, the product kernel of every
/// NN product site (the legacy `main.cpp::matSeq` contract, see the module doc): per
/// output row, `out[i][j] = sum_k a[i][k] * b[k][j]` through `chains`.
///
/// `pub` so `benches/kernels.rs` (a separate compilation unit) can time it as
/// `matmul_seq_92x96`.
pub fn matmul_seq(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    assert_eq!(k, kb, "matmul_seq inner dim mismatch");
    let a = a.as_standard_layout();
    let b = b.as_standard_layout();
    let mut out = vec![0.0f64; m * n];
    matmul_seq_into(
        &mut out,
        a.as_slice().expect("standard layout"),
        b.as_slice().expect("standard layout"),
        (m, k, n),
    );
    Array2::from_shape_vec((m, n), out).expect("m * n elements")
}

/// [`matmul_seq`] on row-major slices, written into a caller-provided buffer:
/// `out (m x n) = a (m x k) * b (k x n)`. `out` is overwritten, not accumulated
/// into (every element restarts from the `+0.0` seed).
pub fn matmul_seq_into(out: &mut [f64], a: &[f64], b: &[f64], (m, k, n): (usize, usize, usize)) {
    assert_eq!(out.len(), m * n, "matmul_seq_into output length");
    assert_eq!(a.len(), m * k, "matmul_seq_into left length");
    assert_eq!(b.len(), k * n, "matmul_seq_into right length");
    for i in 0..m {
        chains::<true>(&mut out[i * n..(i + 1) * n], &a[i * k..(i + 1) * k], b);
    }
}

/// `out[j] = sum_k s[k] * rows[k][j]` for every `j`, `rows` row-major with
/// `out.len()` columns: the one loop nest behind the product and the matvec. The
/// outputs are taken in blocks of 16, then 8, 4, 2 and 1 adjacent columns, each
/// block summed by [`chain_block`]. `SCALAR_FIRST` is the multiply's operand order:
/// `s[k] * rows[k][j]` when true, `rows[k][j] * s[k]` when false.
#[inline(always)]
fn chains<const SCALAR_FIRST: bool>(out: &mut [f64], s: &[f64], rows: &[f64]) {
    let n = out.len();
    let mut j = 0;
    while n - j >= 16 {
        chain_block::<16, SCALAR_FIRST>(out, s, rows, j);
        j += 16;
    }
    if n - j >= 8 {
        chain_block::<8, SCALAR_FIRST>(out, s, rows, j);
        j += 8;
    }
    if n - j >= 4 {
        chain_block::<4, SCALAR_FIRST>(out, s, rows, j);
        j += 4;
    }
    if n - j >= 2 {
        chain_block::<2, SCALAR_FIRST>(out, s, rows, j);
        j += 2;
    }
    if n - j >= 1 {
        chain_block::<1, SCALAR_FIRST>(out, s, rows, j);
    }
}

/// `W` adjacent outputs from column `j0`: each starts at `+0.0` and takes its terms
/// in ascending `k`, one add per term. The `W` accumulators are a fixed-size local,
/// so they stay in registers across the `k` loop and the `W` chains advance side by
/// side; a row-long accumulator in memory costs a load and a store per term, which
/// is slower than the legacy nest once the row is short.
///
/// The two arms of the multiply differ in operand order only, which is each site's
/// legacy order and no value difference (hence the `allow`).
#[inline(always)]
#[allow(clippy::if_same_then_else)]
fn chain_block<const W: usize, const SCALAR_FIRST: bool>(
    out: &mut [f64],
    s: &[f64],
    rows: &[f64],
    j0: usize,
) {
    let stride = out.len();
    let mut acc = [0.0f64; W];
    for (&s_k, row) in s.iter().zip(rows.chunks_exact(stride)) {
        let block: &[f64; W] = row[j0..j0 + W].try_into().expect("block width");
        for (a, &v) in acc.iter_mut().zip(block) {
            *a += if SCALAR_FIRST { s_k * v } else { v * s_k };
        }
    }
    out[j0..j0 + W].copy_from_slice(&acc);
}

/// `w^T` as a row-major buffer (`k x m` for a `m x k` weight): the scratch
/// [`matvec_transposed_into`] reads. Built once per `feed_forward` / `feed_backward`
/// call and dropped with it; nothing is stored on a layer, so no copy can go stale
/// at `set_weights` or at an update.
pub fn transposed(w: ArrayView2<'_, f64>) -> Vec<f64> {
    w.t().iter().copied().collect()
}

/// The `n = 1` product `out (m) = w (m x k) * x (k)`, given `wt = w^T` row-major
/// ([`transposed`]): `k` outer and ascending, the output index innermost over the
/// contiguous row `wt[k]` (`chains`, the weight as the multiply's left operand). Per
/// element this is `matmul_seq(w, x)`'s chain: `+0.0`, then `+= w[i][k] * x[k]` for
/// ascending `k`. `out` is overwritten.
pub fn matvec_transposed_into(out: &mut [f64], wt: &[f64], x: &[f64]) {
    let m = out.len();
    assert_eq!(
        wt.len(),
        x.len() * m,
        "matvec_transposed_into weight length"
    );
    chains::<false>(out, x, wt);
}

/// The rank-1 derivative accumulation `acc[r][col0 + j] += x[r] * d[j]`, for a
/// row-major `acc` of `stride` columns: one add per element, so a caller that
/// calls it once per reverse-time row keeps each element's reverse-time summation
/// order. `col0` selects a gate's column block of a combined matrix.
pub fn rank1_add(acc: &mut [f64], stride: usize, col0: usize, x: &[f64], d: &[f64]) {
    assert!(col0 + d.len() <= stride, "rank1_add column block");
    assert_eq!(acc.len(), x.len() * stride, "rank1_add accumulator length");
    for (r, &x_r) in x.iter().enumerate() {
        let row = &mut acc[r * stride + col0..r * stride + col0 + d.len()];
        for (a, &d_j) in row.iter_mut().zip(d) {
            *a += x_r * d_j;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The legacy nest (`i`/`j` outer, `k` inner, one scalar chain per element), as
    /// `matmul_seq` was written before issue #144: the differential leg's reference.
    fn reference_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (m, k) = a.dim();
        let (kb, n) = b.dim();
        assert_eq!(k, kb);
        let mut out = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += a[[i, kk]] * b[[kk, j]];
                }
                out[[i, j]] = acc;
            }
        }
        out
    }

    /// xorshift64*, so the leg's inputs are the same on every host and run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    /// What a regime draws besides ordinary values.
    #[derive(Clone, Copy, Debug)]
    enum Regime {
        /// Ordinary magnitudes only: where a reordered or fused sum shows.
        Finite,
        /// Mostly signed zeros: where a lost `+0.0` seed shows.
        Zeros,
        /// Signed zeros, subnormals and huge magnitudes, all finite going in.
        Edges,
        /// The above plus infinities and NaN.
        NonFinite,
    }

    const REGIMES: [Regime; 4] = [
        Regime::Finite,
        Regime::Zeros,
        Regime::Edges,
        Regime::NonFinite,
    ];

    fn draw(rng: &mut Rng, regime: Regime) -> f64 {
        let ordinary = |rng: &mut Rng| rng.unit() * [1e-3, 1.0, 37.0, 1e6][rng.below(4)];
        let edge = |rng: &mut Rng| match rng.below(8) {
            0 => 0.0,
            1 => -0.0,
            2 => f64::from_bits(1 + rng.next() % 0x000F_FFFF_FFFF_FFFF), // subnormal
            3 => -f64::from_bits(1 + rng.next() % 0x000F_FFFF_FFFF_FFFF),
            4 => f64::MIN_POSITIVE * rng.unit(),
            5 => rng.unit() * 1e300,
            6 => rng.unit() * 1e-300,
            _ => ordinary(rng),
        };
        match regime {
            Regime::Finite => ordinary(rng),
            Regime::Zeros => match rng.below(8) {
                0..=2 => 0.0,
                3..=5 => -0.0,
                _ => ordinary(rng),
            },
            Regime::Edges => edge(rng),
            Regime::NonFinite => match rng.below(12) {
                0 => f64::INFINITY,
                1 => f64::NEG_INFINITY,
                2 => f64::NAN,
                _ => edge(rng),
            },
        }
    }

    fn matrix(rng: &mut Rng, regime: Regime, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |_| draw(rng, regime))
    }

    /// Bit equality, NaN elements by `is_nan` (module doc).
    fn assert_bits(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (idx, (g, w)) in got.iter().zip(want).enumerate() {
            let same = if w.is_nan() {
                g.is_nan()
            } else {
                g.to_bits() == w.to_bits()
            };
            assert!(same, "{what}: element {idx}: got {g:e}, want {w:e}");
        }
    }

    fn flat(a: &Array2<f64>) -> Vec<f64> {
        a.iter().copied().collect()
    }

    /// Dimensions around the block widths and their remainders, zero included,
    /// plus the two real product shapes (the LSTM projection's `4O = 96` and the
    /// input-delta product's long `n`).
    fn shapes(rng: &mut Rng) -> Vec<(usize, usize, usize)> {
        let mut v = vec![
            (0, 0, 0),
            (0, 3, 4),
            (3, 0, 4),
            (3, 4, 0),
            (1, 1, 1),
            (5, 23, 96),
            (1, 24, 96),
            (23, 96, 81),
            (24, 96, 1),
            // Every block width at once (16 + 8 + 4 + 2 + 1), and its neighbours.
            (2, 5, 31),
            (3, 7, 30),
            (2, 3, 32),
            (2, 9, 47),
        ];
        for _ in 0..40 {
            v.push((1 + rng.below(9), 1 + rng.below(19), 1 + rng.below(19)));
        }
        v
    }

    #[test]
    fn matmul_seq_matches_the_scalar_reference() {
        let mut rng = Rng(0x0144_0001);
        for regime in REGIMES {
            for (m, k, n) in shapes(&mut rng) {
                let a = matrix(&mut rng, regime, m, k);
                let b = matrix(&mut rng, regime, k, n);
                let want = reference_matmul(&a, &b);
                let what = format!("{regime:?} {m}x{k} * {k}x{n}");

                let got = matmul_seq(&a, &b);
                assert_eq!(got.dim(), (m, n), "{what}: shape");
                assert_bits(&flat(&got), &flat(&want), &what);

                // The buffer variant overwrites whatever the buffer held.
                let mut out = vec![f64::NAN; m * n];
                matmul_seq_into(&mut out, &flat(&a), &flat(&b), (m, k, n));
                assert_bits(&out, &flat(&want), &format!("{what} (into)"));
            }
        }
    }

    /// A transposed or reversed operand (ndarray keeps such a layout through
    /// `to_owned`) is read in logical order, not memory order.
    #[test]
    fn matmul_seq_reads_a_non_standard_layout_in_logical_order() {
        let mut rng = Rng(0x0144_0002);
        let a = matrix(&mut rng, Regime::Finite, 7, 5);
        let b = matrix(&mut rng, Regime::Finite, 9, 5);
        let b_t = b.t().to_owned(); // 5 x 9, column-major in memory
        let a_rev = a.slice(ndarray::s![..;-1, ..]).to_owned();
        let want = reference_matmul(&a_rev, &b_t);
        assert_bits(
            &flat(&matmul_seq(&a_rev, &b_t)),
            &flat(&want),
            "reversed * transposed",
        );
    }

    #[test]
    fn matvec_transposed_matches_the_scalar_reference() {
        let mut rng = Rng(0x0144_0003);
        for regime in REGIMES {
            for (m, k, _) in shapes(&mut rng) {
                let w = matrix(&mut rng, regime, m, k);
                let x = matrix(&mut rng, regime, k, 1);
                let want = reference_matmul(&w, &x);
                let mut out = vec![f64::NAN; m];
                matvec_transposed_into(&mut out, &transposed(w.view()), &flat(&x));
                assert_bits(&out, &flat(&want), &format!("{regime:?} {m}x{k} * {k}x1"));
            }
        }
    }

    /// The LSTM's use: an accumulator already holding earlier rows' terms, one gate
    /// block of a combined matrix, against the indexed scalar loops it replaces.
    #[test]
    fn rank1_add_matches_the_scalar_loops() {
        let mut rng = Rng(0x0144_0004);
        for regime in REGIMES {
            for (rows, o, steps) in shapes(&mut rng) {
                let blocks = 1 + rng.below(4);
                let stride = blocks * o;
                let mut want = Array2::<f64>::zeros((rows, stride));
                let mut got = vec![0.0f64; rows * stride];
                for step in 0..steps.min(6) {
                    let x: Vec<f64> = (0..rows).map(|_| draw(&mut rng, regime)).collect();
                    let d: Vec<f64> = (0..o).map(|_| draw(&mut rng, regime)).collect();
                    let col0 = rng.below(blocks) * o;
                    for r in 0..rows {
                        for j in 0..o {
                            want[[r, col0 + j]] += x[r] * d[j];
                        }
                    }
                    rank1_add(&mut got, stride, col0, &x, &d);
                    assert_bits(
                        &got,
                        &flat(&want),
                        &format!("{regime:?} {rows}x{stride} block {col0} step {step}"),
                    );
                }
            }
        }
    }

    /// The dense layer's use (module doc): from a `+0.0` accumulator, adding `x*d`
    /// is adding the `0.0 + x*d` the per-frame `matmul_seq` outer product produced.
    #[test]
    fn rank1_add_from_zero_matches_the_outer_product_fold() {
        let mut rng = Rng(0x0144_0005);
        for regime in REGIMES {
            for (rows, o, steps) in shapes(&mut rng) {
                let mut want = Array2::<f64>::zeros((rows, o));
                let mut got = vec![0.0f64; rows * o];
                for step in 0..steps.min(6) {
                    let x = matrix(&mut rng, regime, rows, 1);
                    let d = matrix(&mut rng, regime, 1, o);
                    want += &reference_matmul(&x, &d);
                    rank1_add(&mut got, o, 0, &flat(&x), &flat(&d));
                    assert_bits(
                        &got,
                        &flat(&want),
                        &format!("{regime:?} {rows}x{o} step {step}"),
                    );
                }
            }
        }
    }

    /// The seed, stated on its own: a sum whose every term is `-0.0` is `+0.0`.
    #[test]
    fn every_product_is_seeded_positive_zero() {
        let a = Array2::from_elem((2, 3), -0.0f64);
        let b = Array2::from_elem((3, 4), 1.0f64);
        let positive_zero = 0.0f64.to_bits();
        for v in matmul_seq(&a, &b).iter() {
            assert_eq!(v.to_bits(), positive_zero, "matmul_seq");
        }
        let w = Array2::from_elem((4, 3), 1.0f64);
        let mut out = vec![f64::NAN; 4];
        matvec_transposed_into(&mut out, &transposed(w.view()), &[-0.0; 3]);
        for v in out {
            assert_eq!(v.to_bits(), positive_zero, "matvec_transposed_into");
        }
    }
}
