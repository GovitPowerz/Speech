//! Streaming per-dimension mean/std with parallel merge; fast log/exp policy.
//!
//! Ported from legacy C++: InputStatistics.*, fmath.hpp. Phase 1.
//!
//! `InputStatistics` stores the STANDARD DEVIATION (already sqrt'd), not the
//! variance -- both the batch ctor and every merge finalize through `sqrt`, and a
//! merge re-squares the two input stds before pooling. There is no epsilon
//! anywhere: an `n == 0` batch (division by zero) produces NaN by design, matching
//! the legacy (`InputStatistics.cpp:8-16` has no `n == 0` guard). This is
//! reproduced as-is -- callers must not construct a batch from zero rows if they
//! want a finite result.

use ndarray::Array2;

/// Streaming per-dimension mean/std accumulator (legacy `InputStatistics`).
#[derive(Debug, Clone, PartialEq)]
pub struct InputStatistics {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub n: i64,
}

impl InputStatistics {
    /// Empty accumulator (legacy default ctor: `_NbOfValues(0)`, empty mean/std).
    pub fn new() -> Self {
        InputStatistics {
            mean: Vec::new(),
            std: Vec::new(),
            n: 0,
        }
    }

    /// Batch statistics over `m` (rows = samples, cols = dims). Population
    /// mean/std (divide by `n`, NOT `n-1`); ported from `InputStatistics.cpp:8-16`
    /// with sequential per-column sums in row order (matches Eigen's colwise
    /// sum accumulation order for this legacy build under `-DEIGEN_DONT_VECTORIZE`).
    /// No `n == 0` guard: an empty `m` divides by zero and yields NaN, matching
    /// the legacy exactly (not reproduced as a special case).
    pub fn from_matrix(m: &Array2<f64>) -> Self {
        let n = m.nrows();
        let d = m.ncols();
        let mut mean = vec![0.0_f64; d];
        for row in m.rows() {
            for (j, &v) in row.iter().enumerate() {
                mean[j] += v;
            }
        }
        for v in mean.iter_mut() {
            *v /= n as f64;
        }

        let mut std = vec![0.0_f64; d];
        for row in m.rows() {
            for (j, &v) in row.iter().enumerate() {
                let diff = v - mean[j];
                std[j] += diff * diff;
            }
        }
        for v in std.iter_mut() {
            *v = (*v / n as f64).sqrt();
        }

        InputStatistics {
            mean,
            std,
            n: n as i64,
        }
    }

    /// Merge `other` into `self` (legacy `InputStatistics::update`,
    /// `InputStatistics.cpp:30-51`).
    ///
    /// - `self.n == 0`: plain copy of `other` (empty-into-empty keeps `n == 0`
    ///   with empty mean/std vectors; empty-into-nonempty copies `other` in).
    /// - else if `other.n > 0`: pooled merge. `std` is stored sqrt'd, so the merge
    ///   re-squares each side's std before pooling: per element, EXACTLY in this
    ///   order -- `(self.std^2 + (merged_mean - self.mean)^2) * self.n as f64`,
    ///   same for `other` with `other.n`, sum the two, divide by the merged `n`,
    ///   then `sqrt`. This op order is load-bearing for bit-exactness (it is not
    ///   algebraically reassociated).
    /// - else (`other.n == 0`, `self.n != 0`): silent no-op (nonempty.update(empty)
    ///   leaves `self` bit-identical).
    pub fn update(&mut self, other: &InputStatistics) {
        if self.n == 0 {
            self.n = other.n;
            self.mean = other.mean.clone();
            self.std = other.std.clone();
        } else if other.n > 0 {
            let n1 = self.n as f64;
            let n2 = other.n as f64;
            let n = self.n + other.n;
            let nf = n as f64;

            let merged_mean: Vec<f64> = self
                .mean
                .iter()
                .zip(other.mean.iter())
                .map(|(&m1, &m2)| (m1 * n1 + m2 * n2) / nf)
                .collect();

            let merged_std: Vec<f64> = merged_mean
                .iter()
                .enumerate()
                .map(|(j, &mm)| {
                    let (s1, m1) = (self.std[j], self.mean[j]);
                    let (s2, m2) = (other.std[j], other.mean[j]);
                    let self_term = (s1 * s1 + (mm - m1).powi(2)) * n1;
                    let other_term = (s2 * s2 + (mm - m2).powi(2)) * n2;
                    ((self_term + other_term) / nf).sqrt()
                })
                .collect();

            self.n = n;
            self.mean = merged_mean;
            self.std = merged_std;
        }
        // else: other.n == 0 and self.n != 0 -> no-op.
    }
}

impl Default for InputStatistics {
    fn default() -> Self {
        Self::new()
    }
}
