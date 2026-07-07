//! Phase 4a Task 9: tier-2 epoch-chained TRAIN golden (Algo 3, the real
//! `1_worker_1.config` net) against the harness CorpusProbe stage.
//!
//! The harness transcribes `CorpusProcessor::train`/`run(epoch)`
//! (`CorpusProcessor.cpp:83-235`) + `BagOfProcessors::SegmentationFunction/
//! saveAndUpdate` (`BagOfProcessors.cpp:207-471`) around the spectral reimpl-swap
//! (ascending-loop forward+backward, REAL CostLaw), feeding the reimpl derivs into
//! the REAL compiled `updateWeights`/`saveWeights` (real Rprop) and dumping the
//! post-`saveAndUpdate` weight vector per epoch. This test replays the SAME
//! committed config/corpus through the Rust engine and compares:
//!
//!   - the per-epoch 33,671-weight vectors (`tier2_weights_epoch{0..4}.bin`) --
//!     canary-gated per element (the NN chain is libm-bearing);
//!   - the 5 `.mat` variables vs the converted fixtures (id/count columns strict,
//!     timing col 6 masked);
//!   - the `bestNNWeight` artifact split (weights Nx1 + derivs Nx2, io::binary);
//!   - the tier-2 VRCTS bytes (final epoch's segmentation);
//!   - NON-VACUITY (measured, not assumed): every consecutive epoch weight pair
//!     DIFFERS (bit compare), and the best-cost save gate FIRED and SKIPPED at
//!     least once each -- the fire/skip counts must equal the manifest's recorded
//!     trajectory (`tier2.train.non_vacuity`), so a fixture regeneration that
//!     collapses the trajectory fails loudly here.
//!
//! Trainer note: the config sets `BLSTM_BackPropagationRpropInit 0.05` -- chosen in
//! Task 9 because the default 1e-2 produced a monotone cost (gate fires every
//! epoch, never skips); 0.05 overshoots at epoch 1 so epochs 2-3 SKIP and the final
//! eval FIRES (measured trajectory F,S,S,F -- both gate branches exercised).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use indexmap::IndexMap;
use ndarray::Array2;

use common::{assert_oracle_eq_f64, assert_vrcts_eq, fixture_phase4a, load_bin_phase4a};
use speech::cli::{Mode, ModeKind};
use speech::engine::corpus_processor::CorpusProcessor;

/// `set_current_dir` is process-global; the run chdir's into a seeded tempdir
/// (relative corpus paths + relative output filenames). Serialize against `cargo
/// test`'s parallel threads.
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

/// Seed `dir` exactly as the extractor seeds the harness corpus workdir: the 2-file
/// tier-2 corpus (f1/f2 + STMs), mapping, tier-2 listing, config, and the phase0
/// weight pack the config's relative `BLSTM_weightsFile` resolves to.
fn seed_tier2(dir: &Path) {
    let src = phase4a_src();
    let corpus_dst = dir.join("corpus");
    std::fs::create_dir_all(&corpus_dst).unwrap();
    for f in ["f1", "f2"] {
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
    for f in [
        "language2classmapping.csv",
        "tier2_fileslisting.csv",
        "tier2_spectral.config",
    ] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    std::fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/reference_data/phase0/NNweights_config1.bin"),
        dir.join("NNweights_config1.bin"),
    )
    .unwrap();
}

fn load_config(dir: &Path, name: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(dir.join(name)).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

/// The manifest's recorded tier-2 trajectory (`tier2.train.non_vacuity`).
fn manifest_non_vacuity() -> (i64, i64, i64) {
    let text = std::fs::read_to_string(fixture_phase4a("manifest.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let nv = &v["tier2"]["train"]["non_vacuity"];
    (
        nv["epochs_differing"].as_i64().unwrap(),
        nv["gate_fired"].as_i64().unwrap(),
        nv["gate_skipped"].as_i64().unwrap(),
    )
}

// ==== minimal MAT v5 reader (uncompressed, the engine's own writer output) =====

fn u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn i32_le(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn f64_le(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

fn mat_var_matrix(bytes: &[u8], name: &str) -> Option<Array2<f64>> {
    let mut off = 128;
    while off + 8 <= bytes.len() {
        let tag = u32_le(bytes, off);
        let size = u32_le(bytes, off + 4) as usize;
        if tag != 14 {
            break;
        }
        let mut p = off + 8;
        p += 8 + 8; // array flags
        let dims_size = u32_le(bytes, p + 4) as usize;
        let rows = i32_le(bytes, p + 8) as usize;
        let cols = i32_le(bytes, p + 12) as usize;
        p += 8 + pad8(dims_size);
        let name_size = u32_le(bytes, p + 4) as usize;
        let var_name = String::from_utf8(bytes[p + 8..p + 8 + name_size].to_vec()).unwrap();
        p += 8 + pad8(name_size);
        if var_name == name {
            let data_size = u32_le(bytes, p + 4) as usize;
            let n = data_size / 8;
            let mut data = Vec::with_capacity(n);
            for k in 0..n {
                data.push(f64_le(bytes, p + 8 + k * 8));
            }
            // column-major -> row-major Array2.
            return Some(Array2::from_shape_fn((rows, cols), |(i, j)| data[j * rows + i]));
        }
        off += 8 + size;
    }
    None
}

/// Element compare: `strict_cols` bit-exact (id/count columns), `masked_cols`
/// skipped (timing), the rest canary-gated.
fn assert_matrix(
    got: &Array2<f64>,
    want: &Array2<f64>,
    strict_cols: &[usize],
    masked_cols: &[usize],
    label: &str,
) {
    assert_eq!(got.dim(), want.dim(), "{label}: shape mismatch");
    let (rows, cols) = got.dim();
    for r in 0..rows {
        for c in 0..cols {
            if masked_cols.contains(&c) {
                continue;
            }
            let g = got[[r, c]];
            let w = want[[r, c]];
            if strict_cols.contains(&c) {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{label}[{r},{c}] strict: got {g} want {w}"
                );
            } else {
                assert_oracle_eq_f64(g, w, &format!("{label}[{r},{c}]"));
            }
        }
    }
}

// === tier2_train_epoch_weights_golden ========================================
// The whole tier-2 train replay: run, then compare every observable against the
// committed fixtures. One test (one expensive run, ~15s) covering the epoch
// weights, the .mat, the bestNNWeight artifacts, the VRCTS bytes, and the
// non-vacuity trajectory.
#[test]
fn tier2_train_epoch_weights_golden() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_tier2(dir.path());
    let cfg = load_config(dir.path(), "tier2_spectral.config");
    let _cwd = CwdGuard::enter(dir.path());
    std::fs::create_dir_all("vrcts_tier2").unwrap();

    // -m (Multi): the scored training mode the harness stage models.
    let mut cp = CorpusProcessor::new(
        vec![cfg],
        Mode {
            kind: ModeKind::Multi,
            verbose: false,
        },
    )
    .unwrap();
    cp.run().unwrap();

    // --- Per-epoch weight vectors vs the harness dumps (canary-gated). ---------
    let trace = cp.epoch_weight_trace_for_test();
    assert_eq!(trace.len(), 5, "epochs 0..4 captured (3 inner + solo + final)");
    for (e, weights) in trace.iter().enumerate() {
        let golden = load_bin_phase4a(&format!("tier2_weights_epoch{e}.bin"));
        assert_eq!(golden.dim(), (weights.len(), 1), "epoch {e} weight count");
        for (k, &w) in weights.iter().enumerate() {
            assert_oracle_eq_f64(w, golden[[k, 0]], &format!("epoch {e} weight {k}"));
        }
    }

    // --- NON-VACUITY (measured): consecutive epoch vectors DIFFER. -------------
    for e in 1..trace.len() {
        let differ = trace[e]
            .iter()
            .zip(trace[e - 1].iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(differ, "epoch {e} weights must differ from epoch {}", e - 1);
    }

    // --- NON-VACUITY: the best-cost gate FIRED and SKIPPED per the manifest. ---
    // A strict drop in the best-cost trace means the epoch's save gate FIRED
    // (bestCost > cost -> saveWeights + bestCost = cost); an unchanged value means
    // it SKIPPED. Counts must equal the manifest's recorded harness trajectory.
    let bc = cp.epoch_best_cost_trace_for_test();
    assert_eq!(bc.len(), 5);
    let mut fired = 0i64;
    let mut skipped = 0i64;
    for e in 1..bc.len() {
        if bc[e] < bc[e - 1] {
            fired += 1;
        } else {
            skipped += 1;
        }
    }
    // Epoch 0 always fires (1e20 seed); the trace starts at the post-epoch-0 value,
    // so only epochs 1..4 are classified here (4 transitions).
    let (m_differ, m_fired, m_skipped) = manifest_non_vacuity();
    assert_eq!(m_differ, 4, "manifest records all 4 consecutive pairs differing");
    assert_eq!(fired, m_fired, "gate-fired count vs manifest trajectory");
    assert_eq!(skipped, m_skipped, "gate-skipped count vs manifest trajectory");
    assert!(fired >= 1 && skipped >= 1, "gate must fire AND skip");

    // --- The 5 .mat variables vs the converted fixtures. -----------------------
    // MultiConfigResults: id cols 0-2 + word/classif counts strict-ish -- id cols
    // strict, timing col 6 masked, the rest canary-gated (the NN cost chain).
    let bytes = std::fs::read("tier2_spectral.mat").unwrap();
    let got = mat_var_matrix(&bytes, "MultiConfigResults").unwrap();
    let want = load_bin_phase4a("tier2_spectral_MultiConfigResults.bin");
    assert_matrix(&got, &want, &[0, 1, 2], &[6], "MultiConfigResults");
    for name in [
        "CostMem",
        "BadClassifMem",
        "CostLIDMem",
        "BadClassifLIDMem",
    ] {
        let got = mat_var_matrix(&bytes, name).unwrap();
        let want = load_bin_phase4a(&format!("tier2_spectral_{name}.bin"));
        assert_matrix(&got, &want, &[], &[], name);
    }

    // --- bestNNWeight artifact split (the LAST gate fire = the final eval): the
    // weights .bin (Nx1) and derivs .bin (Nx2), io::binary with the legacy's
    // misleading .mat suffix. Canary-gated (NN chain). ---------------------------
    let got_w = std::fs::read("weights_bestNNWeight_1_tier2_spectral.mat").unwrap();
    let want_w = load_bin_phase4a("weights_bestNNWeight_1_tier2_spectral.mat");
    let got_w_mat = parse_bin(&got_w);
    assert_eq!(got_w_mat.dim(), want_w.dim(), "bestNNWeight weights dims");
    for (g, w) in got_w_mat.iter().zip(want_w.iter()) {
        assert_oracle_eq_f64(*g, *w, "bestNNWeight weights");
    }
    let got_d = std::fs::read("weightsDerivatives_bestNNWeight_1_tier2_spectral.mat").unwrap();
    let want_d = load_bin_phase4a("weightsDerivatives_bestNNWeight_1_tier2_spectral.mat");
    let got_d_mat = parse_bin(&got_d);
    assert_eq!(got_d_mat.dim(), want_d.dim(), "bestNNWeight derivs dims");
    for (g, w) in got_d_mat.iter().zip(want_d.iter()) {
        assert_oracle_eq_f64(*g, *w, "bestNNWeight derivs");
    }
    // The stats .mat exists (contents: nbOfInputs 0 + empty mean/std -- the real
    // config's InputNormalizationType -1 never accumulates input statistics).
    assert!(
        Path::new("bestNNWeight_1_tier2_spectral.mat").exists(),
        "bestNNWeight stats .mat written"
    );

    // --- VRCTS bytes (final epoch's segmentation), per file per channel. -------
    for f in ["f1", "f2"] {
        for chan in [1, 2] {
            let got = std::fs::read_to_string(format!("vrcts_tier2/{f}_chan_{chan}.xml"))
                .unwrap_or_else(|_| panic!("vrcts_tier2/{f}_chan_{chan}.xml missing"));
            let want =
                std::fs::read_to_string(fixture_phase4a(&format!("tier2_{f}_chan_{chan}.xml")))
                    .unwrap();
            assert_vrcts_eq(&got, &want, &format!("tier2 VRCTS {f} chan {chan}"));
        }
    }
}

/// Parse an io::binary buffer (i64 LE rows, i64 LE cols, f64 LE column-major) into
/// a row-major `Array2` (the engine-written artifacts, read back for comparison).
fn parse_bin(b: &[u8]) -> Array2<f64> {
    let rows = i64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    let cols = i64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let off = 16 + (c * rows + r) * 8;
        f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
    })
}
