//! Phase 4b Task 1: LID confusion core real-compiled goldens.
//!
//! `tools/oracle_harness/main.cpp`'s `phase4b_confusion` stage dumps a crafted
//! 4-row x 3-class `results_lid` matrix (a target-wins row, a target-loses
//! row, a no-target row, and an ambiguous exact-tie row), the harness's OWN
//! transcription of `BagOfProcessors::PrintConfusionMatrix`'s accumulation
//! loop (`BagOfProcessors.cpp:501-535` -- the real matrix is a function-local
//! inside the protected method and is never returned, even via a probe
//! subclass), and a 1x2 `[error1, error2]` pair: error1 is the transcribed
//! matrix fed to the REAL free `Confusion2String`; error2 is the REAL
//! `PrintConfusionMatrix` itself, called end-to-end on the same input via a
//! `BagProbe` subclass. The harness itself asserts `error1 == error2`
//! bit-exact (`ok=1`, `SystemExit` in the extractor otherwise) as the
//! non-vacuous cross-check that the transcribed matrix matches what the real
//! method built internally.
//!
//! Both `confusion_from_results` and `confusion_error` are pure
//! integer-threshold/division arithmetic (no libm), so every assert here is
//! STRICT BITS on every platform (no oracle/canary gate).

mod common;

use common::{assert_bits_eq, load_bin_phase4b};
use speech::engine::confusion::{confusion_error, confusion_from_results};

/// The harness's own transcribed matrix, replayed through the Rust
/// `confusion_from_results` on the SAME input, must match bit-exactly -- and
/// that transcription was itself cross-validated against the REAL compiled
/// `PrintConfusionMatrix` inside the harness (see the module doc).
#[test]
fn confusion_matrix_matches_harness_transcription() {
    let input = load_bin_phase4b("confusion_input.bin");
    let want_matrix = load_bin_phase4b("confusion_matrix.bin");

    let (got_matrix, _error) = confusion_from_results(&input);

    assert_bits_eq(&got_matrix, &want_matrix, "confusion_matrix");
}

/// The Rust `confusion_error` (and `confusion_from_results`'s own internal
/// call to it) must reproduce BOTH real-compiled error values dumped in
/// `confusion_error.bin` -- error1 (transcribed matrix -> REAL
/// `Confusion2String`) and error2 (REAL `PrintConfusionMatrix` end-to-end),
/// which the harness already proved equal to each other.
#[test]
fn error_matches_real_confusion2string_and_printconfusionmatrix() {
    let input = load_bin_phase4b("confusion_input.bin");
    let want_errors = load_bin_phase4b("confusion_error.bin");
    assert_eq!(want_errors.dim(), (1, 2), "confusion_error.bin must be 1x2");
    let want_error1 = want_errors[[0, 0]];
    let want_error2 = want_errors[[0, 1]];
    assert_eq!(
        want_error1.to_bits(),
        want_error2.to_bits(),
        "fixture precondition: the harness already asserted error1 == error2"
    );

    let (matrix, error_from_results) = confusion_from_results(&input);
    let error_standalone = confusion_error(&matrix);

    assert_eq!(
        error_from_results.to_bits(),
        want_error1.to_bits(),
        "confusion_from_results's error must match the real error1/error2"
    );
    assert_eq!(
        error_standalone.to_bits(),
        want_error1.to_bits(),
        "confusion_error called standalone on the matrix must match too"
    );
}

/// Non-vacuity: the crafted input actually exercises all four described row
/// kinds (not just re-asserting the matrix is nonzero). The harness's own
/// crafted encoding is the source of truth for these values (see
/// `tools/oracle_harness/main.cpp`'s `phase4b_confusion` stage and
/// `manifest.json`'s `confusion_input.text`); this test independently derives
/// the same per-row classification from the loaded fixture rather than
/// hardcoding the harness's C++ literals a second time.
#[test]
fn crafted_input_is_non_vacuous() {
    let input = load_bin_phase4b("confusion_input.bin");
    assert_eq!(input.dim(), (4, 3));

    let mut win_rows = 0;
    let mut miss_rows = 0;
    let mut no_target_rows = 0;
    for r in 0..4 {
        let mut target: Option<f64> = None;
        let mut best_competitor = f64::MIN;
        for c in 0..3 {
            let v = input[[r, c]];
            if v > 150.0 {
                target = Some(v - 200.0);
            } else {
                best_competitor = best_competitor.max(v);
            }
        }
        match target {
            Some(t) if t > best_competitor => win_rows += 1,
            Some(_) => miss_rows += 1,
            None => no_target_rows += 1,
        }
    }
    assert_eq!(win_rows, 1, "exactly one target-wins row");
    assert_eq!(miss_rows, 2, "exactly two target-miss rows (loses + tie)");
    assert_eq!(no_target_rows, 1, "exactly one no-target-sentinel row");
}
