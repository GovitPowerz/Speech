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
