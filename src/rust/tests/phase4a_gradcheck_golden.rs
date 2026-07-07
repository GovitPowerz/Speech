//! Phase 4a Task 9: tier-2 corpus-level gradCheck golden (`CorpusProcessor::
//! gradCheck`, `CorpusProcessor.cpp:237-340`) against the harness CorpusProbe
//! stage.
//!
//! The harness runs the same synthetic signal net ([2,2] LSTM + [4,1] output, 137
//! weights, algo 4 -- the `grad_check_synthetic` net) over the committed 1-file
//! corpus (f1 + STM reference), seeded with the phase3 `synth_flat` pattern, and
//! dumps per checked weight `[analytic_col0, analytic_col1, numerical]` plus the
//! two mean errors -> `tier2_gradcheck.bin` (rows 0..9 the per-weight triples, the
//! final row `[mean_error, mean_relative_error, 0]`). The sweep is CAPPED at 10
//! weights (a deviation from the legacy full sweep, recorded in the manifest as
//! `tier2.gradcheck.gradcheck_max_weights`).
//!
//! The seeded weights are loaded from `tier2_gradcheck_seed.bin` -- the harness
//! dump is the single source of truth for the seeding, so the two sides cannot
//! drift on the pattern.
//!
//! Comparators: the whole chain is libm-bearing (log cost law + asinh/sigmoid) ->
//! canary-gated per value. The `< 5e-4` mean-relative-error bound matches the Task
//! 7 `grad_check_synthetic` bound (the analytic BPTT gradient vs the central
//! difference at eps 1e-5; the residual is the finite-difference truncation term,
//! measured 1.2e-4 here).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use indexmap::IndexMap;

use common::{assert_oracle_eq_f64, fixture_phase4a, load_bin_phase4a};
use speech::cli::{Mode, ModeKind};
use speech::engine::corpus_processor::CorpusProcessor;

/// `set_current_dir` is process-global (relative corpus paths); serialize.
static CWD_LOCK: Mutex<()> = Mutex::new(());

struct CwdGuard {
    original: PathBuf,
}

impl CwdGuard {
    fn enter(dir: &Path) -> CwdGuard {
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir).unwrap();
        CwdGuard { original }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original);
    }
}

fn phase4a_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase4a")
}

/// Seed the tempdir like the harness corpus workdir: the 1-file gradcheck corpus
/// (f1 + STM), mapping, listing, config. No weight pack -- the net is seeded from
/// `tier2_gradcheck_seed.bin` via the test hook.
fn seed_gradcheck(dir: &Path) {
    let src = phase4a_src();
    let corpus_dst = dir.join("corpus");
    std::fs::create_dir_all(&corpus_dst).unwrap();
    for ext in ["wav", "stm"] {
        std::fs::copy(
            src.join(format!("corpus/f1.{ext}")),
            corpus_dst.join(format!("f1.{ext}")),
        )
        .unwrap();
    }
    for f in [
        "language2classmapping.csv",
        "tier2_gc_fileslisting.csv",
        "tier2_gradcheck.config",
    ] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
}

/// The manifest's recorded gradcheck parameters (`tier2.gradcheck`).
fn manifest_gradcheck() -> (i64, f64) {
    let text = std::fs::read_to_string(fixture_phase4a("manifest.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let gc = &v["tier2"]["gradcheck"];
    (
        gc["gradcheck_max_weights"].as_i64().unwrap(),
        gc["epsilon"].as_f64().unwrap(),
    )
}

// === tier2_gradcheck_golden ==================================================
// Replay grad_check_capped(1e-5, 10) on the committed synthetic config/corpus with
// the harness-dumped seed weights; compare per-weight [backprop, numerical] + the
// two means vs tier2_gradcheck.bin, canary-gated; keep the < 5e-4 bound assert.
#[test]
fn tier2_gradcheck_golden() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_gradcheck(dir.path());
    let text = std::fs::read_to_string(dir.path().join("tier2_gradcheck.config")).unwrap();
    let cfg: IndexMap<String, String> = speech::legacy_config::parse_legacy_config(&text);
    let _cwd = CwdGuard::enter(dir.path());

    let (max_weights, epsilon) = manifest_gradcheck();
    assert_eq!(max_weights, 10, "manifest gradcheck_max_weights");

    let mut cp = CorpusProcessor::new(
        vec![cfg],
        Mode {
            kind: ModeKind::UnitTest,
            verbose: false,
        },
    )
    .unwrap();

    // Seed from the harness dump (single source of truth for the synth_flat
    // pattern) and sanity-pin the pattern itself so a drifted dump fails loudly.
    let seed = load_bin_phase4a("tier2_gradcheck_seed.bin");
    let nb = cp.get_config0_weights_for_test().len();
    assert_eq!(seed.dim(), (nb, 1), "seed length == net weight count (137)");
    assert_eq!(nb, 137, "synthetic [2,2]+[4,1] net has 137 weights");
    let flat: Vec<f64> = (0..nb).map(|k| seed[[k, 0]]).collect();
    for (k, &w) in flat.iter().enumerate() {
        let expect = ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5;
        assert_eq!(
            w.to_bits(),
            expect.to_bits(),
            "seed[{k}] must be the phase3 synth_flat pattern"
        );
    }
    cp.set_config0_weights_for_test(&flat).unwrap();

    let report = cp
        .grad_check_for_test(epsilon, max_weights as usize)
        .unwrap();

    // Golden: rows 0..9 = [analytic_col0, analytic_col1, numerical]; row 10 =
    // [mean_error, mean_relative_error, 0].
    let golden = load_bin_phase4a("tier2_gradcheck.bin");
    assert_eq!(golden.dim(), (11, 3), "10 per-weight rows + the means row");
    assert_eq!(report.per_weight.len(), 10, "10 weights checked");

    for (k, &(backprop, numerical, _abs_diff)) in report.per_weight.iter().enumerate() {
        // The harness dumps the RAW analytic Nx2 (col0 summed deriv, col1 count);
        // the engine reports the normalized quotient. One f64 division on matching
        // operands is deterministic, so compare against col0/col1.
        let want_backprop = golden[[k, 0]] / golden[[k, 1]];
        assert_oracle_eq_f64(backprop, want_backprop, &format!("weight {k} backprop"));
        assert_oracle_eq_f64(numerical, golden[[k, 2]], &format!("weight {k} numerical"));
    }
    assert_oracle_eq_f64(report.mean_error, golden[[10, 0]], "mean_error");
    assert_oracle_eq_f64(
        report.mean_relative_error,
        golden[[10, 1]],
        "mean_relative_error",
    );

    // The Task 7 agreement bound (doc-comment above): analytic vs numeric < 5e-4.
    assert!(
        report.mean_relative_error < 5e-4,
        "analytic vs numeric gradient must agree to < 5e-4, got {}",
        report.mean_relative_error
    );
    // Non-vacuity: at least one genuinely nonzero numerical derivative.
    assert!(
        report
            .per_weight
            .iter()
            .any(|(_, numerical, _)| numerical.abs() > 1e-9),
        "grad check must exercise a nonzero numerical derivative"
    );
}
