//! Phase 4b Task 9: the live bag/corpus LID arms -- corpus-level goldens + the
//! Mode-7 binary e2e.
//!
//! Fixtures come from the harness `phase4b_t9` stage (see
//! `scripts/extract_phase4b_fixtures.py`): every per-file `getSegmentation` there
//! is the REAL COMPILED driver on TINY nets (every GEMM reduction dim < 23, the
//! nn_product_probes boundary), so the real Eigen engine and this ascending-loop
//! port agree bit-for-bit on the oracle env; libm-bearing values (LSTM
//! activations, log cost laws, softmax) are canary-gated off it
//! (`assert_oracle_eq_f64`).
//!
//! - `twin_train_*`: the Mode-7 phSeq 2-file TRAIN replay (algo 6) -- BOTH nets'
//!   per-epoch post-saveAndUpdate weights, the cost+costLID best-cost gate
//!   fire/skip trajectory (measured, manifest-pinned), the 5 `.mat` variables,
//!   the bestNNWeight artifact split incl. saveWeightsLID's `LID_` prefix, and
//!   the save_and_update confusion capture (errorPercLID, display-only in the
//!   legacy).
//! - `twin_train_ns_costlid_gate_live`: the `:465` costLID = -1.0 no-speech gate
//!   pinned LIVE -- the gated trajectory matches the `ns` goldens and DIVERGES
//!   from the committed `nsx` (gate-disabled counterfactual) trajectory at the
//!   manifest-recorded epoch. A port that dropped the gate would reproduce nsx.
//! - `twin_gradcheck_golden`: the per-network cost-column switch (algo 6 net
//!   ii==0 -> SAD cols 4/len-1; ii==1 -> LID cols 14/len-2,
//!   `CorpusProcessor.cpp:281-288`), golden per network index.
//! - `lid5_gradcheck_golden`: algo 5's columns 14/len-2 (`:278-280`).
//! - `binary_e2e_mode7`: `speech -m twin_e2e.config` in a seeded tempdir ->
//!   exit 0 + the MultiConfigResults values vs the committed fixture (timing
//!   col masked; the algo-6 confusion columns value-compared).

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use indexmap::IndexMap;
use ndarray::Array2;

use common::{
    assert_matrix, assert_oracle_eq_f64, fixture_phase4b, load_bin_phase4b, mat_var_matrix,
};
use speech::cli::{Mode, ModeKind};
use speech::engine::corpus_processor::CorpusProcessor;

/// `set_current_dir` is process-global; the replay runs chdir into seeded
/// tempdirs (relative corpus paths + output filenames). Serialize.
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

fn phase4b_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase4b")
}

/// Seed `dir` exactly as the extractor seeds the harness t9 workdir: configs +
/// seeds + mapping at the root, the phSeq corpus (+ STMs + listing) under
/// `corpus_phseq/`, the wav corpus under `corpus_lid/`.
fn seed_t9(dir: &Path) {
    let src = phase4b_src();
    for f in [
        "twin_train.config",
        "twin_train_ns.config",
        "twin_e2e.config",
        "twin_gradcheck.config",
        "lid5_gradcheck.config",
        "languagemapping_lid7.csv",
        "listing_gc_wav.csv",
        "tiny_sad_seed.bin",
        "tiny_lid_seed.bin",
    ] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    let phseq = dir.join("corpus_phseq");
    std::fs::create_dir_all(&phseq).unwrap();
    for f in [
        "s1.phSeq",
        "s2.phSeq",
        "s1.stm",
        "s2.stm",
        "listing_train.csv",
    ] {
        std::fs::copy(src.join("corpus_phseq").join(f), phseq.join(f)).unwrap();
    }
    let wav = dir.join("corpus_lid");
    std::fs::create_dir_all(&wav).unwrap();
    for f in ["f1.wav", "f1.stm", "f1_gc.stm"] {
        std::fs::copy(src.join("corpus_lid").join(f), wav.join(f)).unwrap();
    }
}

fn load_config(dir: &Path, name: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(dir.join(name)).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

fn multi_mode() -> Mode {
    Mode {
        kind: ModeKind::Multi,
        verbose: false,
    }
}

/// The manifest's measured Task-9 section (`corpus_lid.measured`).
fn t9_manifest() -> serde_json::Value {
    let text = std::fs::read_to_string(fixture_phase4b("manifest.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["corpus_lid"]["measured"].clone()
}

/// Compare a per-epoch weight trace against the committed `<prefix>_epoch<e>.bin`
/// dumps, canary-gated per element (the NN chain is libm-bearing).
fn assert_trace(trace: &[Vec<f64>], prefix: &str, epochs: usize) {
    assert_eq!(trace.len(), epochs, "{prefix}: epoch count");
    for (e, weights) in trace.iter().enumerate() {
        let golden = load_bin_phase4b(&format!("{prefix}_epoch{e}.bin"));
        assert_eq!(golden.dim(), (weights.len(), 1), "{prefix} epoch {e} len");
        for (k, &w) in weights.iter().enumerate() {
            assert_oracle_eq_f64(w, golden[[k, 0]], &format!("{prefix} epoch {e} weight {k}"));
        }
    }
}

/// Parse an io::binary buffer (i64 LE rows/cols, f64 LE column-major) row-major.
fn parse_bin(b: &[u8]) -> Array2<f64> {
    let rows = i64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    let cols = i64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let off = 16 + (c * rows + r) * 8;
        f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
    })
}

// === twin_train_epoch_weights_golden =========================================
// The whole Mode-7 train replay at N=1: run, then compare every observable
// against the committed fixtures. One test (one run) covering BOTH nets' epoch
// weights, the measured gate trajectory, the .mat, the bestNNWeight artifacts
// (incl. the LID_ prefix), and the confusion capture.
#[test]
fn twin_train_epoch_weights_golden() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_t9(dir.path());
    let cfg = load_config(dir.path(), "twin_train.config");
    let _cwd = CwdGuard::enter(dir.path());

    let mut cp = CorpusProcessor::new(vec![cfg], multi_mode()).unwrap();
    cp.run().unwrap();

    let m = t9_manifest();
    let epochs = m["train"]["twin_train"]["epochs"].as_u64().unwrap() as usize;

    // --- BOTH nets' per-epoch weight vectors (canary-gated). -----------------
    assert_trace(cp.epoch_weight_trace_for_test(), "twin_train_sad", epochs);
    assert_trace(
        cp.epoch_weight_trace_lid_for_test(),
        "twin_train_lid",
        epochs,
    );

    // --- NON-VACUITY (measured): consecutive LID epoch vectors DIFFER; the
    // SAD net (never forwarded in Mode 7, all-zero derivs -> 0/0 = NaN
    // normalization -> Rprop's else-branch +delta drift) also moves. ----------
    let lid_trace = cp.epoch_weight_trace_lid_for_test();
    for (e, pair) in lid_trace.windows(2).enumerate() {
        let differ = pair[1]
            .iter()
            .zip(pair[0].iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(differ, "LID epoch {} must differ from epoch {e}", e + 1);
    }

    // --- The best-cost gate fire/skip trajectory == the MEASURED manifest one.
    let bc = cp.epoch_best_cost_trace_for_test();
    assert_eq!(bc.len(), epochs);
    let (mut fired, mut skipped) = (0i64, 0i64);
    for e in 1..bc.len() {
        if bc[e] < bc[e - 1] {
            fired += 1;
        } else {
            skipped += 1;
        }
    }
    assert_eq!(
        fired,
        m["train"]["twin_train"]["gate_fired"].as_i64().unwrap(),
        "gate-fired count vs the measured trajectory"
    );
    assert_eq!(
        skipped,
        m["train"]["twin_train"]["gate_skipped"].as_i64().unwrap(),
        "gate-skipped count vs the measured trajectory"
    );
    assert!(fired >= 1 && skipped >= 1, "gate must fire AND skip");
    // The best-cost values themselves (cost+costLID criterion), canary-gated.
    for (e, want) in m["train"]["twin_train"]["best_cost"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_oracle_eq_f64(bc[e], want.as_f64().unwrap(), &format!("best_cost[{e}]"));
    }

    // --- The 5 .mat variables vs the converted fixtures (ids strict, timing
    // col 6 masked, everything else canary-gated). The algo-6 row width is
    // 18 + classNb(2) = 20 -> 23 ResultsE cols; the CONFUSION columns (19/20)
    // are value-compared like every other column. ------------------------------
    let bytes = std::fs::read("twin_train.mat").unwrap();
    let got = mat_var_matrix(&bytes, "MultiConfigResults").unwrap();
    let want = load_bin_phase4b("twin_train_MultiConfigResults.bin");
    assert_eq!(got.ncols(), 23, "3 id cols + the 20-col algo-6 row");
    assert_matrix(
        &got,
        &want,
        &[0, 1, 2],
        &[6],
        "twin_train MultiConfigResults",
    );
    for name in ["CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"] {
        let got = mat_var_matrix(&bytes, name).unwrap();
        let want = load_bin_phase4b(&format!("twin_train_{name}.bin"));
        assert_matrix(&got, &want, &[], &[], name);
    }

    // --- bestNNWeight artifact split, BOTH nets: the LID pair carries
    // saveWeightsLID's LID_ prefix (BagOfProcessors.cpp:176 via
    // TwinBLSTMSpectralLID.cpp:83-85). Value-compared (canary-gated). ----------
    for name in [
        "weights_bestNNWeight_1_twin_train.mat",
        "weightsDerivatives_bestNNWeight_1_twin_train.mat",
        "weights_LID_bestNNWeight_1_twin_train.mat",
        "weightsDerivatives_LID_bestNNWeight_1_twin_train.mat",
    ] {
        let got = parse_bin(&std::fs::read(name).unwrap_or_else(|_| panic!("{name} missing")));
        let want = parse_bin(&std::fs::read(fixture_phase4b(name)).unwrap());
        assert_eq!(got.dim(), want.dim(), "{name} dims");
        for (g, w) in got.iter().zip(want.iter()) {
            assert_oracle_eq_f64(*g, *w, name);
        }
    }
    // The stats .mat artifacts exist for both nets (empty stats: the LID net is
    // InputNormalizationType 0, the SAD net never forwards).
    assert!(Path::new("bestNNWeight_1_twin_train.mat").exists());
    assert!(Path::new("LID_bestNNWeight_1_twin_train.mat").exists());

    // --- The save_and_update confusion capture: errorPercLID (display-only in
    // the legacy; the :443 cout) == the measured harness value per epoch. The
    // LAST save_and_update's capture is the final epoch's. ---------------------
    let conf = cp.last_confusion_for_test_from_bag();
    assert_eq!(conf.len(), 1, "one conf in the bag");
    let want_epl = m["train"]["twin_train"]["error_perc_lid"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .as_f64()
        .unwrap();
    assert_oracle_eq_f64(conf[0].0, want_epl, "errorPercLID (final epoch)");
    // 2 classes -> (2+2)x(2+2) confusion.
    assert_eq!(conf[0].1.dim(), (4, 4), "confusion shape");
}

// === twin_train_ns_costlid_gate_live =========================================
// The :465 costLID = -1.0 gate, LIVE: the ns config's rising threshold (11)
// exceeds the constant-10 Mode-7 result_vec, so NO speech is ever detected and
// EVERY updateWeightsLID receives -1.0. The replay must (a) match the ns
// goldens bit-for-bit and (b) DIVERGE from the committed nsx (gate-disabled
// counterfactual) trajectory at the manifest epoch -- if the port dropped the
// gate, (a) fails at exactly that epoch and the trace matches nsx instead.
#[test]
fn twin_train_ns_costlid_gate_live() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_t9(dir.path());
    let cfg = load_config(dir.path(), "twin_train_ns.config");
    let _cwd = CwdGuard::enter(dir.path());

    let mut cp = CorpusProcessor::new(vec![cfg], multi_mode()).unwrap();
    cp.run().unwrap();

    let m = t9_manifest();
    let epochs = m["train"]["twin_train_ns"]["epochs"].as_u64().unwrap() as usize;
    let diverge = m["ns_gate_diverge_epoch"].as_u64().unwrap() as usize;

    // (a) the gated trajectory IS the ns golden.
    assert_trace(
        cp.epoch_weight_trace_lid_for_test(),
        "twin_train_ns_lid",
        epochs,
    );

    // (b) it DIVERGES from the counterfactual at the measured epoch: bit-equal
    // before it (the gate only bites once costLID would have RISEN -- Rprop's
    // cost-gated backtrack), bit-different at it.
    let lid_trace = cp.epoch_weight_trace_lid_for_test();
    for (e, epoch_weights) in lid_trace.iter().enumerate().take(epochs) {
        let nsx = load_bin_phase4b(&format!("twin_train_nsx_lid_epoch{e}.bin"));
        let any_diff = epoch_weights
            .iter()
            .enumerate()
            .any(|(k, &w)| w.to_bits() != nsx[[k, 0]].to_bits());
        if e < diverge {
            assert!(
                !any_diff,
                "epoch {e}: gated and counterfactual traces must still agree (diverge at {diverge})"
            );
        } else if e == diverge {
            assert!(
                any_diff,
                "epoch {e}: the gated trace must DIVERGE from the gate-disabled counterfactual \
                 -- the costLID=-1.0 gate is not observable (was it dropped?)"
            );
        }
    }

    // The mems hold the PRE-gate costLID (written at :457-460, BEFORE :465):
    // real values, not -1.0. The stdout-measured post-gate values were -1.0.
    let bytes = std::fs::read("twin_train_ns.mat").unwrap();
    let cost_lid_mem = mat_var_matrix(&bytes, "CostLIDMem").unwrap();
    for v in cost_lid_mem.iter() {
        assert!(
            *v != -1.0,
            "CostLIDMem must carry the PRE-gate costLID (order :457-460 then :465)"
        );
    }
}

// === twin_gradcheck_golden ===================================================
// Algo 6, Mode 5, wav: the corpus gradCheck iterates BOTH backprop-active
// networks of config 0 -- net ii==0 (SAD) reads cost cols 4/len-1, net ii==1
// (LID) cols 14/len-2 (CorpusProcessor.cpp:281-288). Golden per network index;
// values canary-gated (LSTM activations are libm-bearing; the cost laws are
// square so no cost-law libm).
#[test]
fn twin_gradcheck_golden() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_t9(dir.path());
    let cfg = load_config(dir.path(), "twin_gradcheck.config");
    let _cwd = CwdGuard::enter(dir.path());

    let mut cp = CorpusProcessor::new(vec![cfg], multi_mode()).unwrap();
    let reports = cp.grad_check_all_for_test(1e-5, 10).unwrap();
    assert_eq!(reports.len(), 2, "both nets backprop-active -> two reports");
    assert_eq!(reports[0].0, 0, "network index 0 first");
    assert_eq!(reports[1].0, 1, "network index 1 second");

    for (ii, report) in &reports {
        let golden = load_bin_phase4b(&format!("twin_gradcheck_net{ii}.bin"));
        assert_gradcheck(report, &golden, &format!("twin net {ii}"));
    }

    // The column SWITCH is load-bearing: the two nets' numerical gradients must
    // differ (same corpus, different cost columns + different perturbed nets).
    let n0: Vec<f64> = reports[0].1.per_weight.iter().map(|t| t.1).collect();
    let n1: Vec<f64> = reports[1].1.per_weight.iter().map(|t| t.1).collect();
    assert_ne!(n0, n1, "net-0 vs net-1 numerical gradients must differ");
}

// === lid5_gradcheck_golden ===================================================
// Algo 5: the gradCheck cost columns are the LID pair 14/len-2 (:278-280) --
// the single (base-class) net IS the LID net.
#[test]
fn lid5_gradcheck_golden() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_t9(dir.path());
    let cfg = load_config(dir.path(), "lid5_gradcheck.config");
    let _cwd = CwdGuard::enter(dir.path());

    let mut cp = CorpusProcessor::new(vec![cfg], multi_mode()).unwrap();
    let reports = cp.grad_check_all_for_test(1e-5, 10).unwrap();
    assert_eq!(reports.len(), 1, "one net");
    assert_eq!(reports[0].0, 0);
    let golden = load_bin_phase4b("lid5_gradcheck.bin");
    assert_gradcheck(&reports[0].1, &golden, "lid5");
}

/// |a - b| in units in the last place of the larger magnitude.
fn ulp_diff(a: f64, b: f64) -> u64 {
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    let (ai, bi) = (a.to_bits() as i64, b.to_bits() as i64);
    if (ai < 0) != (bi < 0) {
        return u64::MAX;
    }
    ai.abs_diff(bi)
}

/// Compare a [`GradCheckReport`] against a harness `(sweep+1) x 3` golden:
/// rows 0..sweep-1 = [analytic_col0, analytic_col1, numerical]; final row
/// [mean_error, mean_relative_error, 0]. The report's `backprop` is
/// col0/col1 (recomputed here).
///
/// TOLERANCES (measured, documented -- NOT the strict-bits standard of the
/// train/e2e fixtures): the golden comes from the REAL compiled engine, whose
/// backward deriv chain at these (already-shrunk, T=21) shapes still hits
/// occasional 1-3 ULP real-Eigen-vs-ascending rounding splits on SOME perturbed
/// inputs (the same near-k-boundary calibration story as the phase-2/3 NN_TOL
/// real-net sites). The NUMERICAL column is a central difference, so a 1-ULP
/// cost delta amplifies by 1/(2*eps): ulp(0.23)/2e-5 ~ 1.4e-12 -- hence the
/// absolute bound (with headroom for the off-oracle libm-canary environments,
/// where cost deltas up to ~512*eps*|cost| map to ~1e-9). The backprop column
/// is a plain quotient: <= 16 ULP. Measured on the oracle env: backprop <= 3
/// ULP, numerical <= 3e-12, most entries bit-exact.
fn assert_gradcheck(
    report: &speech::engine::corpus_processor::GradCheckReport,
    golden: &Array2<f64>,
    label: &str,
) {
    const NUM_ABS: f64 = 1e-8;
    let sweep = golden.nrows() - 1;
    assert_eq!(report.per_weight.len(), sweep, "{label}: sweep length");
    for (k, (backprop, numerical, _diff)) in report.per_weight.iter().enumerate() {
        let want_back = golden[[k, 0]] / golden[[k, 1]];
        assert!(
            ulp_diff(*backprop, want_back) <= 16,
            "{label} weight {k} backprop: got {backprop:e} want {want_back:e} (> 16 ULP)"
        );
        let want_num = golden[[k, 2]];
        assert!(
            (numerical - want_num).abs() <= NUM_ABS,
            "{label} weight {k} numerical: got {numerical:e} want {want_num:e} (> {NUM_ABS:e} abs)"
        );
    }
    // The means aggregate the FD noise itself (they ARE ~1e-12/1e-9 scale), so
    // only their order of magnitude is meaningful cross-engine.
    assert!(
        (report.mean_error - golden[[sweep, 0]]).abs() <= NUM_ABS,
        "{label} mean_error: got {:e} want {:e}",
        report.mean_error,
        golden[[sweep, 0]]
    );
    assert!(
        (report.mean_relative_error - golden[[sweep, 1]]).abs() <= 1e-5,
        "{label} mean_relative_error: got {:e} want {:e}",
        report.mean_relative_error,
        golden[[sweep, 1]]
    );
    // The analytic/numerical agreement itself must be TIGHT on both sides (the
    // Task-9 measured bound; a broken column switch or deriv chain blows this
    // up by orders of magnitude).
    assert!(
        report.mean_relative_error < 1e-6,
        "{label}: mean relative error {:e} not at the measured ~1e-9 scale",
        report.mean_relative_error
    );
    // Non-degeneracy: at least one genuinely nonzero numerical derivative.
    assert!(
        report.per_weight.iter().any(|t| t.1.abs() > 1e-12),
        "{label}: all numerical derivatives ~0 (degenerate check)"
    );
}

// === binary_e2e_mode7 ========================================================
// `speech -m twin_e2e.config` (epochs 0, LID backprop off -- the as-committed
// Mode-7 scoring semantics) in a seeded tempdir: exit 0 + the 5 .mat variables
// vs the committed value fixtures (timing col 6 masked, ids strict; the algo-6
// confusion columns are present and value-compared).
#[test]
fn binary_e2e_mode7() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_t9(dir.path());

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(dir.path())
        .args(["-m", "twin_e2e.config"])
        .output()
        .expect("failed to run speech binary");
    assert!(
        output.status.success(),
        "speech -m twin_e2e.config exited non-zero: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(dir.path().join("twin_e2e.mat")).unwrap();
    let got = mat_var_matrix(&bytes, "MultiConfigResults").expect("MultiConfigResults missing");
    let want = load_bin_phase4b("twin_e2e_MultiConfigResults.bin");
    // 2 files x 1 channel = 2 rows; 3 ids + 20-col algo-6 row = 23 cols.
    assert_eq!(got.dim(), (2, 23), "e2e MCR 2x23 (the grown LID row width)");
    assert_matrix(&got, &want, &[0, 1, 2], &[6], "e2e MultiConfigResults");
    for name in ["CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"] {
        let got = mat_var_matrix(&bytes, name).unwrap_or_else(|| panic!("{name} missing"));
        let want = load_bin_phase4b(&format!("twin_e2e_{name}.bin"));
        assert_matrix(&got, &want, &[], &[], &format!("e2e {name}"));
    }
}
