//! LID confusion matrix + normalized error aggregate.
//!
//! Ported from legacy C++: `BagOfProcessors::PrintConfusionMatrix`
//! (`BagOfProcessors.cpp:474-598`, the sentinel-decode + best-non-target
//! argmax accumulation at `:501-535`) and `Confusion2String`'s RETURNED
//! aggregate (`Helpers.hpp:365-438`) -- the ANSI-colored display string is
//! display-only and NOT produced here (per the brief).
//!
//! Two pieces of legacy code are commented out and deliberately NOT ported:
//! the `classNb == 2` binary ROC-curve variant (`BagOfProcessors.cpp:477-499`,
//! a threshold-sweep negative/positive histogram) and a duplicate
//! normalization+print block (`:541-593`) that re-derives (via `cout`, not
//! `error +=`) exactly what the LIVE `Confusion2String` call at `:540` already
//! computes. Both are dead code (never executed) in the legacy binary; see
//! `dead_binary_variant_not_ported` below and IMPROVEMENTS.md.

use ndarray::Array2;

/// Port of `BagOfProcessors::PrintConfusionMatrix`'s matrix build
/// (`BagOfProcessors.cpp:501-535`) plus the LIVE `Confusion2String` error call
/// (`:540`, `confusion_error` below): returns the `(classNb+2) x (classNb+2)`
/// confusion matrix (`classNb` = `results_lid.ncols()`) alongside the
/// row-normalized error aggregate.
///
/// Per row: any column value `> 150` marks the TARGET (correct) class,
/// decoded to its real score via `score - 200`; every other column is a
/// competing class's raw score, and the row's best (max) competitor is
/// tracked. The target wins the row (diagonal + row/col totals incremented)
/// iff its decoded score is STRICTLY greater than the best competitor's score
/// (`:524`); a tie falls to the best-competitor branch (`:528-532`).
///
/// PER-ROW DECODE (FIXED in phase 5; the legacy sticky quirk is left behind):
/// the four decode accumulators (`score_target`/`max_score_not_target`/
/// `pos_target`/`pos_best_not_target`) are declared INSIDE the row loop, so a
/// row can never inherit a prior row's target/competitor index. A row with NO
/// in-band target (`pos_target` stays `0` -- no column `> 150`) is an
/// out-of-set trial with no true class in the closed set, so it is SKIPPED
/// entirely: credited to no cell, no total, and never to the index-header slot
/// (row/col `0`). The matrix is therefore ORDER-INDEPENDENT.
///
/// The legacy `BagOfProcessors::PrintConfusionMatrix` instead declared
/// `posTarget`/`posBestNotTarget` OUTSIDE the loop and reset only
/// `scoreTarget`/`maxScoreNotTarget` per row (`:509-510,533-534`), so a
/// no-target row inherited the LAST row's `posTarget` (or, on the very first
/// row, the header slot `0`) -- a spurious, row-order-dependent attribution.
/// The C++/Octave oracle harnesses still describe that sticky behavior; the
/// port deliberately diverges here. See IMPROVEMENTS.md ([phase4b] STICKY).
///
/// `classNb <= 1` reproduces the legacy's own gate (`:501`, `if (classNb >
/// 1)`): no matrix is built, error stays `0.0`.
pub fn confusion_from_results(results_lid: &Array2<f64>) -> (Array2<f64>, f64) {
    let class_nb = results_lid.ncols();
    if class_nb <= 1 {
        return (Array2::zeros((0, 0)), 0.0);
    }

    let mut confusion = Array2::<f64>::zeros((class_nb + 2, class_nb + 2));
    // legacy: :503-506 index headers (row 0 / col 0), kk in [0, classNb].
    for kk in 0..=class_nb {
        confusion[[0, kk]] = kk as f64;
        confusion[[kk, 0]] = kk as f64;
    }

    // legacy: :512-535, ascending row loop (product/accumulation contract).
    // FIXED (phase 5): the four decode accumulators are declared PER ROW (below),
    // and a no-target row (`pos_target == 0`) is skipped -- see the fn doc.
    for jj in 0..results_lid.nrows() {
        let mut max_score_not_target = -1.0_f64;
        let mut pos_best_not_target = 0usize;
        let mut pos_target = 0usize;
        let mut score_target = -1.0_f64;
        for kk in 0..class_nb {
            let v = results_lid[[jj, kk]];
            if v > 150.0 {
                score_target = v - 200.0;
                pos_target = kk + 1;
            } else if v > max_score_not_target {
                max_score_not_target = v;
                pos_best_not_target = kk + 1;
            }
        }
        if pos_target == 0 {
            // No in-band target this row: out-of-set trial, credited to no class.
            continue;
        }
        if score_target > max_score_not_target {
            confusion[[pos_target, pos_target]] += 1.0;
            confusion[[pos_target, class_nb + 1]] += 1.0;
            confusion[[class_nb + 1, pos_target]] += 1.0;
        } else {
            confusion[[pos_target, pos_best_not_target]] += 1.0;
            confusion[[pos_target, class_nb + 1]] += 1.0;
            confusion[[class_nb + 1, pos_best_not_target]] += 1.0;
        }
    }

    let error = confusion_error(&confusion);
    (confusion, error)
}

/// Port of `Confusion2String`'s RETURNED aggregate (`Helpers.hpp:365-438`):
/// the ANSI-colored display string it also builds is display-only and NOT
/// produced here.
///
/// `confusion` must be square with `rows - 2 == cols - 2 >= 2` -- the
/// legacy's own ill-conditioned gate (`:369-371`). Violating it is fatal in
/// the legacy (`exit(1)`); ported here as a panic (a process-abort primitive
/// has no direct Rust equivalent that stays testable, and this branch is
/// reachable only from a deliberately malformed matrix -- see
/// IMPROVEMENTS.md).
///
/// Row-normalizes each class row to percentages (`confusion(kk+1, classNb+1)`
/// is that row's total; a zero total leaves the row UNNORMALIZED, i.e. the
/// raw counts, since the legacy only scales `if (total > 0)`), then sums the
/// per-class diagonal deficit `100 - normalized(ii,ii)` over `ii` in
/// `[1, classNb]` and divides by `classNb`.
pub fn confusion_error(confusion: &Array2<f64>) -> f64 {
    let (rows, cols) = confusion.dim();
    let class_nb_i = rows as i64 - 2;
    if class_nb_i < 2 || class_nb_i != (cols as i64 - 2) {
        panic!("confusion matrix is ill-conditioned ({rows},{cols})");
    }
    let class_nb = class_nb_i as usize;

    // legacy: :373 confusionNorm = confusion.topLeftCorner(classNb+1, classNb+1)
    // -- a COPY, excluding the totals row/col (index classNb+1).
    let mut confusion_norm = Array2::<f64>::zeros((class_nb + 1, class_nb + 1));
    for r in 0..=class_nb {
        for c in 0..=class_nb {
            confusion_norm[[r, c]] = confusion[[r, c]];
        }
    }
    // legacy: :374-376 per-row percentage scaling, kk in [0, classNb).
    for kk in 0..class_nb {
        let total = confusion[[kk + 1, class_nb + 1]];
        if total > 0.0 {
            let scale = 100.0 / total;
            for c in 1..=class_nb {
                confusion_norm[[kk + 1, c]] *= scale;
            }
        }
    }

    // legacy: :417-419 diagonal deficit, ii in [1, classNb].
    let mut error = 0.0;
    for ii in 1..=class_nb {
        error += 100.0 - confusion_norm[[ii, ii]];
    }
    error / class_nb as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four crafted 3-class rows shared with the oracle harness stage
    /// (`tools/oracle_harness/main.cpp`'s `phase4b_confusion` block) and the
    /// golden test (`tests/phase4b_confusion_golden.rs`): a target-wins row, a
    /// target-loses row, a no-target row, and an ambiguous equal-scores row.
    fn crafted_rows() -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 3),
            vec![
                30.0, 280.0, 50.0, // A: target col1 wins (80 > max(30,50)=50)
                220.0, 90.0, 10.0, // B: target col0 loses (20 <= max(90,10)=90)
                10.0, 90.0, 40.0, // C: no target sentinel at all
                50.0, 20.0, 250.0, // D: target col2 ties the best competitor (50 == 50)
            ],
        )
        .unwrap()
    }

    /// Hand-computed expected confusion matrix for `crafted_rows` (row A
    /// wins, B loses, C has no target, D ties). FIXED (phase 5): row C has no
    /// in-band target (`pos_target == 0`), so it is SKIPPED entirely -- it no
    /// longer inherits row B's sticky `posTarget = 1` and no longer credits
    /// `(1,2)`/`(1,4)`/`(4,2)`. Row A's win lands at `(2,2)`, row B's miss at
    /// `(1,2)`, row D's tie-miss at `(3,1)`. Contrast the pre-fix sticky matrix
    /// (row C credited `(1,2)`, so `(1,2)==2`, `(4,2)==3`): the diff is exactly
    /// row C's removed contribution.
    #[test]
    fn sentinel_decode_and_argmax() {
        let (confusion, _error) = confusion_from_results(&crafted_rows());
        let want = Array2::from_shape_vec(
            (5, 5),
            vec![
                0.0, 1.0, 2.0, 3.0, 0.0, //
                1.0, 0.0, 1.0, 0.0, 1.0, // row B's miss (1x); row C SKIPPED (no target)
                2.0, 0.0, 1.0, 0.0, 1.0, // row A's win
                3.0, 1.0, 0.0, 0.0, 1.0, // row D's tie-miss
                0.0, 1.0, 2.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        assert_eq!(confusion, want);
    }

    /// `confusion_error` on the same crafted matrix: FIXED (phase 5) each of
    /// rows 1/2/3 now has total 1 (row C, the no-target row that used to inflate
    /// row1 to 2, is skipped); every nonzero off-diagonal cell normalizes to
    /// exactly 100.0 (1*(100/1)=100), so the diagonal deficit sums to
    /// (100-0)+(100-100)+(100-0) = 200, /classNb (3) = 200/3 exactly. The error
    /// aggregate is INVARIANT to the fix (row C only ever hit an off-diagonal
    /// cell, never a diagonal, and the aggregate row is not read by
    /// `confusion_error`), so this value matches the pre-fix golden bit-for-bit;
    /// IEEE754 division is correctly-rounded, bit-reproducible on every platform.
    #[test]
    fn error_matches_hand_norm_crafted() {
        let (confusion, error) = confusion_from_results(&crafted_rows());
        assert_eq!(error, confusion_error(&confusion));
        assert_eq!(error, 200.0 / 3.0);
    }

    /// FIXED (phase 5): a no-target row (no column `> 150`, `pos_target == 0`)
    /// is SKIPPED entirely -- it credits nothing and never pollutes the
    /// index-header row/col `0`. This is the same crafted single no-target row
    /// the pre-fix test used to charge to the header slot (`(0,2)`/`(0,4)`/
    /// `(4,2)`); the fixed matrix is just the clean labels, byte-for-byte.
    /// Order-independence: identical whether this row is first, last, or alone.
    #[test]
    fn no_target_row_skipped_never_pollutes_header() {
        let rows = Array2::from_shape_vec((1, 3), vec![10.0, 90.0, 40.0]).unwrap();
        let (confusion, _error) = confusion_from_results(&rows);
        let want = Array2::from_shape_vec(
            (5, 5),
            vec![
                0.0, 1.0, 2.0, 3.0, 0.0, // labels untouched (no header pollution)
                1.0, 0.0, 0.0, 0.0, 0.0, //
                2.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        assert_eq!(confusion, want);
    }

    /// BOTH-LANGUAGE IDENTITY (F5): the Rust `confusion_from_results` and the
    /// Python `speech.scoring.confusion_matrix` are DIFFERENT legacy sources
    /// (`PrintConfusionMatrix`'s `score_target > max_competitor` win vs
    /// `confusionThresh.m`'s asymmetric-threshold hit), so they only produce an
    /// identical matrix on an input where the win/hit decisions coincide -- but
    /// the SHARED F5 fix (per-row decode + skip no-target rows) is exercised
    /// identically. This crafted 2-class input is engineered so both agree: row0
    /// has a col-1 target (score 60) that clears BOTH win rules (60 > competitor
    /// 5, and 60 > thresh 10), and row1 has no target so BOTH skip it. There is
    /// no PyO3 seam for confusion (it is not on the `speech_rs.Engine` surface),
    /// so the identity is pinned by this Rust test plus its byte-identical mirror
    /// `test_confusion_cross_language_identity_no_target_skip`
    /// (`tests/test_phase4c_scoring.py`) -- the two hardcode the SAME expected
    /// matrix.
    #[test]
    fn cross_language_identity_no_target_skip() {
        let input = Array2::from_shape_vec((2, 2), vec![5.0, 260.0, 20.0, 30.0]).unwrap();
        let (confusion, _error) = confusion_from_results(&input);
        let want = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 2.0, 0.0, // labels clean (row1 no-target -> skipped)
                1.0, 0.0, 0.0, 0.0, //
                2.0, 0.0, 1.0, 1.0, // row0 win at (2,2)
                0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();
        assert_eq!(confusion, want);
    }

    /// A second, independent hand-computed `confusion_error` check on a plain
    /// classNb=2 matrix (no `confusion_from_results` accumulation involved),
    /// with clean round numbers: row1 (10 samples: 8 correct, 2 confused with
    /// class2) normalizes to 80/20; row2 (10 samples: 3/7) normalizes to
    /// 30/70. error = (100-80) + (100-70) = 50, /classNb(2) = 25.0 exactly.
    #[test]
    fn error_matches_hand_norm_standalone() {
        #[rustfmt::skip]
        let confusion = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 2.0, 0.0,
                1.0, 8.0, 2.0, 10.0,
                2.0, 3.0, 7.0, 10.0,
                0.0, 11.0, 9.0, 20.0,
            ],
        )
        .unwrap();
        assert_eq!(confusion_error(&confusion), 25.0);
    }

    /// A zero-total row is NOT normalized (the legacy's `if (total > 0)`
    /// guard) -- its diagonal cell stays the raw count, not a percentage.
    /// Row2's total (8, a power of 2) and diagonal (8, all-correct) are
    /// chosen so `8 * (100.0/8.0) == 100.0` is bit-exact (no rounding-then-
    /// multiply slop), keeping the whole assertion strict-bits. Row3 (the
    /// totals row) is never read by `confusion_error` (only `topLeftCorner`
    /// + the `classNb+1` total COLUMN are), so its values are illustrative.
    #[test]
    fn zero_total_row_left_unnormalized() {
        #[rustfmt::skip]
        let confusion = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 2.0, 0.0,
                1.0, 5.0, 0.0, 0.0, // row total 0 -> guard skips normalization
                2.0, 0.0, 8.0, 8.0,
                0.0, 5.0, 8.0, 8.0,
            ],
        )
        .unwrap();
        // ii=1: 100 - 5 = 95 (unnormalized raw count). ii=2: total=8, scale
        // 100/8=12.5 (exact), diagonal 8*12.5=100.0 (exact) -> deficit 0.
        assert_eq!(confusion_error(&confusion), (95.0 + 0.0) / 2.0);
    }

    /// `Confusion2String`'s ill-conditioned gate (`Helpers.hpp:369-371`,
    /// legacy `exit(1)`) is fatal in the legacy; ported as a panic.
    #[test]
    #[should_panic(expected = "ill-conditioned")]
    fn ill_conditioned_matrix_panics() {
        confusion_error(&Array2::zeros((2, 2)));
    }

    /// `classNb <= 1` reproduces the legacy's own `if (classNb > 1)` gate: no
    /// matrix, error 0.0 -- and NOT a call into `confusion_error` (which would
    /// itself be fatal on a 0x0/1x1 matrix).
    #[test]
    fn single_class_gate_returns_empty() {
        let single = Array2::from_shape_vec((3, 1), vec![10.0, 220.0, 5.0]).unwrap();
        let (confusion, error) = confusion_from_results(&single);
        assert_eq!(confusion.dim(), (0, 0));
        assert_eq!(error, 0.0);
    }

    /// Doc assertion (S1 test list item `dead_binary_variant_not_ported`): the
    /// commented-out `classNb == 2` binary ROC-curve variant
    /// (`BagOfProcessors.cpp:477-499`, a threshold-sweep negative/positive
    /// histogram over 10001 steps) is NEVER reached in the legacy binary: the
    /// surrounding `if`/`else if` has its FIRST arm (`classNb == 2`) entirely
    /// commented out, so `classNb == 2` falls straight into the SAME general
    /// `classNb greater than 1` accumulation as every other `classNb >= 2`.
    /// This port has no ROC/histogram code path at all -- classNb==2 here
    /// produces exactly the general `(classNb+2)x(classNb+2)` confusion
    /// matrix, proven by shape + a plain win-case accumulation identical in
    /// form to the classNb=3 case above.
    #[test]
    fn dead_binary_variant_not_ported() {
        let rows = Array2::from_shape_vec((1, 2), vec![280.0, 30.0]).unwrap();
        let (confusion, _error) = confusion_from_results(&rows);
        // General-path shape: (classNb+2)x(classNb+2) = 4x4, NOT a 1x(2*maxIt+2)
        // histogram shape the dead ROC branch would have produced.
        assert_eq!(confusion.dim(), (4, 4));
        let want = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 2.0, 0.0, //
                1.0, 1.0, 0.0, 1.0, //
                2.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        assert_eq!(confusion, want);
    }
}
