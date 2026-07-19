//! Phase 4c Task 1: the promoted `CorpusProcessor` seam surface (the PyO3 seam
//! surface) -- `results_matrix`/`weights`/`set_weights`/`weights_derivatives`/
//! `input_statistics`/`grad_check`, thin public promotions of the pre-existing
//! `_for_test` hooks (`corpus_processor.rs:825-1000` pre-promotion). These tests
//! exercise ONLY the new public methods; the `_for_test` wrappers now delegate to
//! them (no duplicate bodies) and stay covered by the existing 4a/4b suites.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use indexmap::IndexMap;

use common::{fixture_phase4a, load_bin_phase4a};
use speech::cli::{Mode, ModeKind};
use speech::engine::corpus_processor::CorpusProcessor;

/// `set_current_dir` is process-global; serialize against `cargo test`'s default
/// parallel threads (same pattern as every other `CorpusProcessor` test file).
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

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

fn load_config(rel: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(ref_dir().join(rel)).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

fn mode(kind: ModeKind) -> Mode {
    Mode {
        kind,
        verbose: false,
    }
}

/// A minimal `fileslisting`/`language2classmapping` pair: `Corpus::from_config`
/// only parses these CSVs at construction time (`Corpus.cpp:16-101`), it never
/// opens the referenced audio -- so a dummy, non-existent wav path is fine for
/// any test that never calls `run()`.
fn write_min_corpus(dir: &Path) {
    std::fs::write(dir.join("language2classmapping.csv"), "unk;unk;0\n").unwrap();
    std::fs::write(dir.join("fileslisting.csv"), "dummy.wav;;unk;unk;1.0;1\n").unwrap();
}

/// Tier-1 TDC (Algo 1, no NN) config wired to a minimal (dummy-file) corpus.
fn tdc_config(dir: &Path) -> IndexMap<String, String> {
    let mut m = load_config("phase4a/tier1_tdc.config");
    m.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "0".to_string(),
    );
    m.insert(
        "language2classmapping".to_string(),
        dir.join("language2classmapping.csv")
            .to_str()
            .unwrap()
            .to_string(),
    );
    m.insert(
        "fileslisting".to_string(),
        dir.join("fileslisting.csv").to_str().unwrap().to_string(),
    );
    m
}

/// The real algo-3 config (`1_worker_1.config`) + real net weights
/// (`NNweights_config1.bin`, 33,671 f64), wired to a minimal (dummy-file) corpus --
/// identical override pattern to `phase4a_segfn.rs::spectral_bag_config`.
fn algo3_config(dir: &Path) -> IndexMap<String, String> {
    let mut m = load_config("phase0/1_worker_1.config");
    m.insert("numOuterThreads".to_string(), "1".to_string());
    m.insert("Algo_choice".to_string(), "3".to_string());
    m.insert("Audio_offset".to_string(), "0.0".to_string());
    m.insert("Audio_max_duration".to_string(), "2.0".to_string());
    m.insert(
        "BLSTM_weightsFile".to_string(),
        ref_dir()
            .join("phase0/NNweights_config1.bin")
            .to_str()
            .unwrap()
            .to_string(),
    );
    m.insert(
        "language2classmapping".to_string(),
        dir.join("language2classmapping.csv")
            .to_str()
            .unwrap()
            .to_string(),
    );
    m.insert(
        "fileslisting".to_string(),
        dir.join("fileslisting.csv").to_str().unwrap().to_string(),
    );
    m
}

// === promoted_weights_roundtrip ==============================================
// The public `weights`/`set_weights` (generalized from the config-0-bound
// `_for_test` hooks to a `pos` param the bag already dispatches on): a no-NN
// TDC config (pos 0) round-trips an EMPTY weight set; the real 33,671-weight
// algo-3 net round-trips its flat weights bit-identically through get/set/get.
#[test]
fn promoted_weights_roundtrip() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // -- TDC (algo 1, no NN): weights(0) is empty; set_weights(0, &[]) is a no-op. --
    {
        let dir = tempfile::tempdir().unwrap();
        write_min_corpus(dir.path());
        let cfg = tdc_config(dir.path());
        let _cwd = CwdGuard::enter(dir.path());

        let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
        assert_eq!(cp.weights(0), Vec::<Vec<f64>>::new(), "TDC has no NN");
        cp.set_weights(0, &[]).unwrap();
        assert_eq!(
            cp.weights(0),
            Vec::<Vec<f64>>::new(),
            "TDC set_weights is a no-op, still empty"
        );
    }

    // -- Algo 3 (real 33,671-weight net): flat get/set/get is bit-identical. --
    {
        let dir = tempfile::tempdir().unwrap();
        write_min_corpus(dir.path());
        let cfg = algo3_config(dir.path());
        let _cwd = CwdGuard::enter(dir.path());

        let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
        let w1 = cp.weights(0);
        assert_eq!(w1.len(), 1, "algo 3 has a single net");
        assert_eq!(w1[0].len(), 33_671, "the real net's flat weight count");

        cp.set_weights(0, &w1).unwrap();
        let w2 = cp.weights(0);
        assert_eq!(w1, w2, "flat get/set/get must be bit-identical");
    }
}

/// Seed the tempdir like `phase4a_gradcheck_golden.rs`'s `seed_gradcheck`: the
/// 1-file synthetic gradcheck corpus (f1 + STM), mapping, listing, config.
fn seed_gradcheck(dir: &Path) {
    let src = ref_dir().join("phase4a");
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

/// The manifest's recorded gradcheck parameters (`tier2.gradcheck`), same source
/// of truth `phase4a_gradcheck_golden.rs` reads.
fn manifest_gradcheck() -> (usize, f64) {
    let text = std::fs::read_to_string(fixture_phase4a("manifest.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let gc = &v["tier2"]["gradcheck"];
    (
        gc["gradcheck_max_weights"].as_i64().unwrap() as usize,
        gc["epsilon"].as_f64().unwrap(),
    )
}

// === grad_check_public_matches_test_hook =====================================
// The public `grad_check` is `grad_check_all_for_test`'s body promoted verbatim
// (no config-0-bound narrowing needed -- the per-network sweep was already
// general). Both are deterministic and self-restoring (the sweep snapshots +
// restores `self.processors`), so two successive calls on the SAME processor,
// SAME seeded weights, must agree bit-exactly.
#[test]
fn grad_check_public_matches_test_hook() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_gradcheck(dir.path());
    let cfg = load_config_at(&dir.path().join("tier2_gradcheck.config"));
    let _cwd = CwdGuard::enter(dir.path());

    let (max_weights, epsilon) = manifest_gradcheck();

    let mut cp = CorpusProcessor::new(
        vec![cfg],
        Mode {
            kind: ModeKind::UnitTest,
            verbose: false,
        },
    )
    .unwrap();

    // Seed the phase3 synth_flat pattern (same as the tier2 golden) via the
    // harness-dumped seed, so the sweep exercises a non-degenerate cost.
    let seed = load_bin_phase4a("tier2_gradcheck_seed.bin");
    let nb = cp.weights(0)[0].len();
    assert_eq!(seed.dim(), (nb, 1), "seed length == net weight count");
    let flat: Vec<f64> = (0..nb).map(|k| seed[[k, 0]]).collect();
    cp.set_weights(0, &[flat]).unwrap();

    let hook_result = cp.grad_check_all_for_test(epsilon, max_weights).unwrap();
    let public_result = cp.grad_check(epsilon, max_weights).unwrap();

    assert_eq!(hook_result.len(), public_result.len(), "same net count");
    assert!(
        !hook_result.is_empty(),
        "non-vacuity: at least one net checked"
    );
    for ((idx_a, rep_a), (idx_b, rep_b)) in hook_result.iter().zip(public_result.iter()) {
        assert_eq!(idx_a, idx_b, "same network index");
        assert_eq!(
            rep_a.mean_error.to_bits(),
            rep_b.mean_error.to_bits(),
            "mean_error bit-exact"
        );
        assert_eq!(
            rep_a.mean_relative_error.to_bits(),
            rep_b.mean_relative_error.to_bits(),
            "mean_relative_error bit-exact"
        );
        assert_eq!(rep_a.per_weight.len(), rep_b.per_weight.len());
        for (k, (pa, pb)) in rep_a
            .per_weight
            .iter()
            .zip(rep_b.per_weight.iter())
            .enumerate()
        {
            assert_eq!(
                pa.0.to_bits(),
                pb.0.to_bits(),
                "weight {k} backprop bit-exact"
            );
            assert_eq!(
                pa.1.to_bits(),
                pb.1.to_bits(),
                "weight {k} numerical bit-exact"
            );
            assert_eq!(
                pa.2.to_bits(),
                pb.2.to_bits(),
                "weight {k} abs_diff bit-exact"
            );
        }
    }
}

fn load_config_at(path: &Path) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(path).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

/// Seed the tempdir with the committed tier-1 3-file 2-channel corpus (same
/// fixture set `phase4a_tier1_e2e.rs::seed_corpus` uses).
fn seed_tier1_corpus(dir: &Path) {
    let src = ref_dir().join("phase4a");
    let corpus_dst = dir.join("corpus");
    std::fs::create_dir_all(&corpus_dst).unwrap();
    for f in ["f1", "f2", "f3"] {
        std::fs::copy(
            src.join("corpus").join(format!("{f}.wav")),
            corpus_dst.join(format!("{f}.wav")),
        )
        .unwrap();
        std::fs::copy(
            src.join("corpus").join(format!("{f}.stm")),
            corpus_dst.join(format!("{f}.stm")),
        )
        .unwrap();
    }
    for f in ["fileslisting.csv", "language2classmapping.csv"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
}

// === results_matrix_after_run =================================================
// `results_matrix()` is EMPTY (0x0) before any run (the ctor's Array2::zeros((0,
// 0)) seed -- `transform_results` is the only writer, and it only fires from
// inside a completed run). After a solo run over the tier-1 3-file 2-channel TDC
// corpus, it is 6 rows (3 files x 2 chans) x 21 cols (3 id + 18 result); the id
// columns ([file+1, conf+1, chan+1]) are checked STRICT (pure integers, no libm).
#[test]
fn results_matrix_after_run() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_tier1_corpus(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let mut cfg = load_config("phase4a/tier1_tdc.config");
    cfg.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "0".to_string(),
    );
    cfg.insert(
        "multiConfigResultsOutputFile".to_string(),
        "phase4c_seam.mat".to_string(),
    );
    cfg.insert("Dump_Directory".to_string(), "vrcts_seam".to_string());
    std::fs::create_dir_all("vrcts_seam").unwrap();

    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
    assert_eq!(
        cp.results_matrix().dim(),
        (0, 0),
        "empty before any run (the doc-commented contract)"
    );

    cp.run().unwrap();

    let m = cp.results_matrix();
    assert_eq!(
        m.dim(),
        (6, 21),
        "3 files x 2 channels, 3 id + 18 result cols"
    );

    // Ascending [file+1, conf+1, chan+1] per the transformResults row order.
    let expected_ids: [(f64, f64, f64); 6] = [
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 2.0),
        (2.0, 1.0, 1.0),
        (2.0, 1.0, 2.0),
        (3.0, 1.0, 1.0),
        (3.0, 1.0, 2.0),
    ];
    for (r, &(file_id, conf_id, chan_id)) in expected_ids.iter().enumerate() {
        assert_eq!(m[[r, 0]], file_id, "row {r} file id");
        assert_eq!(m[[r, 1]], conf_id, "row {r} conf id");
        assert_eq!(m[[r, 2]], chan_id, "row {r} chan id");
    }
}

// === image_mode_trains_exact_path_unblocked (T5 finding 2) ===================
// The `run()`->`train()` fast-processor guard (`corpus_processor.rs::train`, T5 review)
// must be DEAD CODE on every exact-path run. `tier1_tdc.config` already defaults
// `Neural_Networks_BackPropagation_Epochs` to 2 (no Inference_Path key -> exact, no
// FastSpectral/FastTwinLid processor ever built), so Image mode (in the run() `allowed`
// set) routes straight to `train()`. TDC (algo 1) has no NN, but train()'s epoch loop /
// mode dispatch / final transform_results run identically regardless -- this must still
// complete exactly as it did before the guard landed.
#[test]
fn image_mode_trains_exact_path_unblocked() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_tier1_corpus(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let mut cfg = load_config("phase4a/tier1_tdc.config");
    cfg.insert(
        "multiConfigResultsOutputFile".to_string(),
        "phase7_t5_image_train.mat".to_string(),
    );
    cfg.insert(
        "Dump_Directory".to_string(),
        "vrcts_t5_image_train".to_string(),
    );
    std::fs::create_dir_all("vrcts_t5_image_train").unwrap();

    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Image)).unwrap();
    cp.run().expect(
        "exact + Image + Epochs>0 must still train (the fast-processor guard is dead code here)",
    );

    let m = cp.results_matrix();
    assert_eq!(
        m.dim(),
        (6, 21),
        "3 files x 2 channels, 3 id + 18 result cols, same shape as the Solo/Epochs=0 run"
    );
}

// === weights_derivatives_nonzero_through_seam (F10) ==========================
// F10 (phase 5): the release seam `weights_derivatives(pos)` must return the
// gradient `run_epoch` FOLDED on the per-lane clones (the R6 static-lane model),
// NOT the main bag's never-updated accumulator (col0 = 0). The pre-F10 bug: the
// main-bag read returned an all-zero gradient, so `forward_backward` reported a
// zero gradient and the modern SMORMS3 loop never moved a weight (T10 discovery).
//
// The gradcheck synthetic net (algo 4, backprop ON) with epsilon forced to 0
// takes the runSolo path (Epochs 0 + Epsilon 0 -> the `else` branch of `run()`),
// a SINGLE fold at theta -- exactly `forward_backward`'s contract. RED on HEAD:
// every col0 entry of the seam matrix is 0.
#[test]
fn weights_derivatives_nonzero_through_seam() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_gradcheck(dir.path());
    let mut cfg = load_config_at(&dir.path().join("tier2_gradcheck.config"));
    // epsilon 0 -> run() dispatches to runSolo (a single fold at theta), not
    // gradCheck; Epochs is already 0 and backprop already ON in the config.
    cfg.insert(
        "Neural_Networks_Gradient_Check_Epsilon".to_string(),
        "0".to_string(),
    );
    let _cwd = CwdGuard::enter(dir.path());

    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::UnitTest)).unwrap();

    // Seed the phase3 synth_flat pattern (the non-degenerate cost the gradcheck
    // golden exercises), so the fold is a real, smoothly-varying gradient.
    let seed = load_bin_phase4a("tier2_gradcheck_seed.bin");
    let nb = cp.weights(0)[0].len();
    assert_eq!(seed.dim(), (nb, 1), "seed length == net weight count");
    let seed_flat: Vec<f64> = (0..nb).map(|k| seed[[k, 0]]).collect();
    cp.set_weights(0, std::slice::from_ref(&seed_flat)).unwrap();

    cp.run().unwrap();

    let seam = cp.weights_derivatives(0);
    assert_eq!(seam.len(), 1, "algo 4 -> a single net's derivative matrix");
    let dmat = &seam[0];
    assert!(dmat.nrows() > 0, "the derivative matrix must have rows");
    let nonzero_col0 = (0..dmat.nrows()).filter(|&k| dmat[[k, 0]] != 0.0).count();
    assert!(
        nonzero_col0 > 0,
        "F10: the release seam must return the folded (nonzero) gradient; got all-zero col0 \
         (the pre-fix bug -- the seam read the never-folded main bag)"
    );

    // Identity: the seam's normalized gradient (col0/col1) must EQUAL what
    // grad_check's own analytic fold sees at the SAME weights -- both are the
    // identical `run_epoch` fold. runSolo moved the in-memory weights by one
    // Rprop step (save_and_update -> update_weights), so re-seed to theta before
    // grad_check snapshots + checks there.
    let seam_norm: Vec<f64> = (0..dmat.nrows())
        .map(|k| dmat[[k, 0]] / dmat[[k, 1]])
        .collect();
    cp.set_weights(0, &[seed_flat]).unwrap();
    let reports = cp.grad_check(1e-5, 10).unwrap();
    assert_eq!(reports.len(), 1, "one backprop-active net");
    let (net_idx, report) = &reports[0];
    assert_eq!(*net_idx, 0);
    for (k, (backprop, _num, _diff)) in report.per_weight.iter().enumerate() {
        assert_eq!(
            seam_norm[k].to_bits(),
            backprop.to_bits(),
            "F10: seam normalized gradient[{k}] must bit-match grad_check's analytic backprop \
             (both are the same run_epoch fold at theta)"
        );
    }
}

// === grad_check_restores_seam_derivs_invariant (F10 rider) ===================
// gradCheck's perturbation sweep runs `run_epoch` per +/-eps step, and each of
// those OVERWRITES the F10 `seam_derivs` stash with the fold at the PERTURBED
// weights. The epilogue restores `self.processors` to the pre-sweep snapshot; it
// must ALSO restore `seam_derivs`, or `weights_derivatives(0)` would return a
// stale gradient at the last -eps perturbation while `weights(0)` reads the
// restored theta -- an inconsistent seam observable through the PyO3 surface (a
// `grad_check` call followed by `weights_derivatives`).
//
// Invariant: gradCheck is TRANSPARENT to the stash -- the seam gradient is
// bit-identical before and after a `grad_check` call when the weights do not
// change across it. Here `before` is the reset-bag fallback (no run() since
// `set_weights` -> the accumulator's col0 is all-zero), so the non-vacuity check
// (col0 all-zero) is exactly what a mutation (dropping the epilogue restore, which
// leaves the nonzero last -eps perturbation fold) would violate.
#[test]
fn grad_check_restores_seam_derivs_invariant() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_gradcheck(dir.path());
    let cfg = load_config_at(&dir.path().join("tier2_gradcheck.config"));
    let _cwd = CwdGuard::enter(dir.path());

    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::UnitTest)).unwrap();

    // Seed the phase3 synth pattern so the sweep's perturbation folds are real
    // (nonzero) -- that nonzero fold is exactly what would pollute the stash if the
    // epilogue restore were missing.
    let seed = load_bin_phase4a("tier2_gradcheck_seed.bin");
    let nb = cp.weights(0)[0].len();
    let seed_flat: Vec<f64> = (0..nb).map(|k| seed[[k, 0]]).collect();
    cp.set_weights(0, std::slice::from_ref(&seed_flat)).unwrap();

    // The seam BEFORE gradCheck: no run() since set_weights, so the stash is empty
    // and `weights_derivatives` falls back to the bag's reset accumulator (col0 = 0).
    let before = cp.weights_derivatives(0);
    let reports = cp.grad_check(1e-5, 5).unwrap();
    assert_eq!(reports.len(), 1, "one backprop-active net");
    let after = cp.weights_derivatives(0);

    // Invariant: the weights are unchanged across the gradCheck, so the seam
    // gradient must be too -- gradCheck restored the stash it found.
    assert_eq!(before.len(), after.len(), "same net count across gradCheck");
    for (b, a) in before.iter().zip(after.iter()) {
        assert_eq!(b.dim(), a.dim(), "same derivative matrix shape");
        for (bv, av) in b.iter().zip(a.iter()) {
            assert_eq!(
                bv.to_bits(),
                av.to_bits(),
                "gradCheck must leave weights_derivatives bit-identical for unchanged weights \
                 (the epilogue must restore seam_derivs, not leave the last -eps perturbation)"
            );
        }
    }

    // Non-vacuity: `after` IS the reset-bag fallback (col0 all-zero), the consistent
    // state for these unchanged-and-unrun weights -- so a mutation leaving the
    // nonzero last -eps perturbation fold is detectable, not silently absorbed.
    let col0_nonzero = (0..after[0].nrows())
        .filter(|&k| after[0][[k, 0]] != 0.0)
        .count();
    assert_eq!(
        col0_nonzero, 0,
        "unchanged, unrun weights -> the seam falls back to the reset bag (col0 all zero); a \
         nonzero col0 would mean gradCheck left a stale perturbation gradient"
    );
}
