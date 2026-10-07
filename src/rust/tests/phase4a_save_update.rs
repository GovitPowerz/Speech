//! Phase 4a Task 6: `BagOfProcessors::save_and_update` (`BagOfProcessors.cpp:409-471`)
//! + `save_weights`/`update_weights` (`:148-205`).
//!
//! `aggregation_bit_exact` isolates the column-sum/mean arithmetic on an algo-1
//! bag (TDC, NN-free): `save_weights`/`update_weights` are no-ops for algo 1/2 in
//! the legacy (no matching `if` branch), so only the aggregation math is exercised.
//! `best_cost_gate_fires_and_skips`/`cost_mem_rows_written`/
//! `update_called_with_neg_costlid` use a real algo-3 bag (the only ported NN algo
//! with a live save/update criterion) built the same way Task 3/4's inline tests do:
//! `phase0/1_worker_1.config` + `phase0/NNweights_config1.bin` via `BLSTM_weightsFile`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use indexmap::IndexMap;
use ndarray::Array2;

use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::BagOfProcessors;
use speech::engine::channel_result::ChannelResult;
use speech::features::stats::InputStatistics;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

/// `save_and_update`'s `bestNNWeight_<pos+1>_<filename>` compose
/// (`BagOfProcessors.cpp:462-464`) puts the prefix on the BASENAME
/// (`io::prefix_basename`; the legacy glued it onto the whole string, FIXED --
/// `tests/phase7_bench.rs::bench_runs_with_absolute_output_keys` owns the
/// directory case). The legacy's production shape is a bare relative `filename`
/// (e.g. `multiConfigResultsOutputFile`, default `MultiConfigResults.mat`) with
/// cwd == the run directory, and these tests keep that shape: chdir into a
/// tempdir for the artifact-producing calls. `set_current_dir` is
/// process-global, so serialize with this mutex against `cargo test`'s default
/// parallel threads.
static CWD_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard: chdir into `dir`, restore the original cwd on drop. Caller must
/// hold `CWD_LOCK` for the guard's lifetime.
struct CwdGuard {
    original: PathBuf,
}

impl CwdGuard {
    fn enter(dir: &std::path::Path) -> CwdGuard {
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

fn solo_mode() -> Mode {
    Mode {
        kind: ModeKind::Solo,
        verbose: false,
    }
}

fn load_config(rel: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(ref_dir().join(rel)).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

fn with_bag_keys(mut m: IndexMap<String, String>, algo: i32) -> IndexMap<String, String> {
    m.insert("numOuterThreads".to_string(), "1".to_string());
    m.insert("Algo_choice".to_string(), algo.to_string());
    m
}

/// Real algo-3 bag (single config), weights overridden to the known 33,671-weight
/// net, so `save_weights` writes real artifacts and `update_weights` runs the real
/// iRPROP- trainer without crashing.
fn algo3_bag(weights_path: &std::path::Path) -> BagOfProcessors {
    let mut cfg = with_bag_keys(load_config("phase0/1_worker_1.config"), 3);
    cfg.insert(
        "BLSTM_weightsFile".to_string(),
        weights_path.to_str().unwrap().to_string(),
    );
    BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), solo_mode()).unwrap()
}

fn algo1_bag() -> BagOfProcessors {
    let cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
    let mut cfgs = vec![cfg];
    BagOfProcessors::from_configs(&mut cfgs, solo_mode()).unwrap()
}

fn empty_derivs(nb_of_conf: usize) -> BTreeMap<usize, Vec<Array2<f64>>> {
    (0..nb_of_conf).map(|ii| (ii, Vec::new())).collect()
}

fn empty_stats(nb_of_conf: usize) -> BTreeMap<usize, Vec<InputStatistics>> {
    (0..nb_of_conf).map(|ii| (ii, Vec::new())).collect()
}

/// Hand-computed sums/means on four crafted 18-wide rows, algo-1 bag
/// (aggregation isolated from save/update, which are no-ops for algo 1/2).
///
/// Columns used by `save_and_update` (`:409-471`): 2 (badClassif), 4 (cost
/// numerator), 5/6 (signal/speech duration, display-only here), 7-13 (WER),
/// 14 (costLID numerator), 15 (badLIDClassif; the drivers only ever write 100
/// or 0, and `ChannelResult::from_row` reads it as the flag it is), 16 (costLID
/// denominator, col `cols-2`), 17 (cost denominator, col `cols-1`).
#[test]
fn aggregation_bit_exact() {
    let mut bag = algo1_bag();

    // Row layout (18 cols), 4 rows. Col 17 (cost denom) and col 16 (costLID denom)
    // chosen non-zero so both guarded divisions fire.
    #[rustfmt::skip]
    let rows: [[f64; 18]; 4] = [
        [1.0, 2.0, 3.0,  4.0,  10.0, 100.0, 50.0,  2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,  6.0, 100.0, 3.0,  5.0],
        [2.0, 3.0, 4.0,  5.0,  20.0, 100.0, 50.0,  2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,  9.0,   0.0, 4.0,  5.0],
        [0.0, 0.0, 5.0,  6.0,  30.0, 100.0, 50.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 100.0, 5.0,  5.0],
        [4.0, 1.0, 8.0,  7.0,  40.0, 100.0, 50.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0,   0.0, 8.0,  5.0],
    ];
    let m: Vec<ChannelResult> = rows.iter().map(|r| ChannelResult::from_row(r)).collect();

    // sums: col2 = 20, col4 = 100, col15 = 200 -> mean 50, col16 = 20, col17 = 20.
    // col7 (nbOfWords) = 4 -> WER normalization fires.
    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20)]);
    let derivs = empty_derivs(1);
    let stats = empty_stats(1);
    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        "unused.mat",
        &[m],
        &mut best_cost,
        &derivs,
        &stats,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    )
    .unwrap();

    // cost = sums(4)/sums(17) = 100/20 = 5.0
    assert_eq!(cost_mem[0], 5.0);
    // badClassif = means(2) = 20/4 = 5.0
    assert_eq!(bad_classif[0], 5.0);
    // costLID = sums(14)/sums(16) = (6+9+12+15)/20 = 42/20 = 2.1
    assert_eq!(cost_lid[0], 2.1);
    // badLIDClassif = 100 - means(15) = 100 - 50 = 50
    assert_eq!(bad_classif_lid[0], 50.0);
}

/// Counter-0 guard: when the cost denominator column sums to 0, `cost` stays the
/// raw numerator sum (NOT divided) - same for costLID's denominator column.
#[test]
fn aggregation_counter_zero_guard_skips_division() {
    let mut bag = algo1_bag();

    #[rustfmt::skip]
    let rows: [[f64; 18]; 2] = [
        [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
    ];
    let m: Vec<ChannelResult> = rows.iter().map(|r| ChannelResult::from_row(r)).collect();

    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20)]);
    let derivs = empty_derivs(1);
    let stats = empty_stats(1);
    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        "unused.mat",
        &[m],
        &mut best_cost,
        &derivs,
        &stats,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    )
    .unwrap();

    // sums(17) == 0 -> cost NOT divided, stays sums(4) = 7.0.
    assert_eq!(cost_mem[0], 7.0);
    // sums(16) == 0 -> costLID NOT divided, stays sums(14) = 15.0.
    assert_eq!(cost_lid[0], 15.0);
}

/// Algo-3 bag, real net + tempdir: epoch A cost 5.0 beats the 1e20 seed -> the
/// three save artifacts appear and best_cost updates; epoch B cost 7.0 does NOT
/// beat 5.0 -> no new save (files unchanged), best_cost unchanged. STRICT: uses
/// mtime + content equality, not just existence.
#[test]
fn best_cost_gate_fires_and_skips() {
    let _lock = CWD_LOCK.lock().unwrap();
    let weights_src = ref_dir().join("phase0/NNweights_config1.bin");
    let tmp = tempfile::tempdir().unwrap();
    let weights_path = tmp.path().join("weights.bin");
    std::fs::copy(&weights_src, &weights_path).unwrap();

    let mut bag = algo3_bag(&weights_path);
    let derivs_mat = bag.get_weights_derivatives(0)[0].clone();
    let stats0 = bag.get_input_statistics(0);

    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20)]);
    let mut derivs: BTreeMap<usize, Vec<Array2<f64>>> =
        BTreeMap::from([(0, vec![derivs_mat.clone()])]);
    let stats: BTreeMap<usize, Vec<InputStatistics>> = BTreeMap::from([(0, stats0)]);

    let _cwd = CwdGuard::enter(tmp.path());

    // legacy always uses a bare relative filename (e.g. multiConfigResultsOutputFile,
    // default MultiConfigResults.mat) with cwd == run dir.
    let out_a = "run_a.mat";
    // Col 4 (cost numerator) = 5.0, col 17 (denom) = 1.0 -> cost = 5.0.
    #[rustfmt::skip]
    let row_a: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let m_a = vec![ChannelResult::from_row(&row_a)];

    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        out_a,
        &[m_a],
        &mut best_cost,
        &derivs,
        &stats,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    )
    .unwrap();

    assert_eq!(best_cost[&0], 5.0);
    let weights_artifact = PathBuf::from(format!("weights_bestNNWeight_1_{out_a}"));
    let derivs_artifact = PathBuf::from(format!("weightsDerivatives_bestNNWeight_1_{out_a}"));
    let mat_artifact = PathBuf::from(format!("bestNNWeight_1_{out_a}"));
    assert!(weights_artifact.exists(), "{weights_artifact:?} missing");
    assert!(derivs_artifact.exists(), "{derivs_artifact:?} missing");
    assert!(mat_artifact.exists(), "{mat_artifact:?} missing");

    let weights_mtime_a = std::fs::metadata(&weights_artifact)
        .unwrap()
        .modified()
        .unwrap();
    let weights_bytes_a = std::fs::read(&weights_artifact).unwrap();

    // Refresh derivs map (save_and_update did not mutate it; reuse the same handle).
    derivs.insert(0, vec![derivs_mat]);

    let out_b = "run_b.mat";
    // Cost 7.0 - worse than the 5.0 best - must NOT save.
    #[rustfmt::skip]
    let row_b: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let m_b = vec![ChannelResult::from_row(&row_b)];

    bag.save_and_update(
        out_b,
        &[m_b],
        &mut best_cost,
        &derivs,
        &stats,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    )
    .unwrap();

    // best_cost unchanged, and no artifact for run_b was ever created.
    assert_eq!(best_cost[&0], 5.0);
    let weights_artifact_b = PathBuf::from(format!("weights_bestNNWeight_1_{out_b}"));
    assert!(!weights_artifact_b.exists());

    // The run_a artifact is untouched (same mtime + bytes): epoch B did not re-save.
    let weights_mtime_b = std::fs::metadata(&weights_artifact)
        .unwrap()
        .modified()
        .unwrap();
    let weights_bytes_b = std::fs::read(&weights_artifact).unwrap();
    assert_eq!(weights_mtime_a, weights_mtime_b);
    assert_eq!(weights_bytes_a, weights_bytes_b);
}

/// The four row slices (cost/badClassif/costLID/badLIDClassif) receive the
/// per-config aggregation results at the right index, over 2 configs.
#[test]
fn cost_mem_rows_written() {
    let weights_src = ref_dir().join("phase0/NNweights_config1.bin");
    let tmp = tempfile::tempdir().unwrap();
    let weights_path = tmp.path().join("weights.bin");
    std::fs::copy(&weights_src, &weights_path).unwrap();

    let mut cfg_tdc = with_bag_keys(load_config("phase2b/tdc.config"), 1);
    let mut cfg_spectral = with_bag_keys(load_config("phase0/1_worker_1.config"), 3);
    cfg_spectral.insert(
        "BLSTM_weightsFile".to_string(),
        weights_path.to_str().unwrap().to_string(),
    );
    let mut cfgs = vec![cfg_tdc.clone(), cfg_spectral];
    let mut bag = BagOfProcessors::from_configs(&mut cfgs, solo_mode()).unwrap();
    let _ = &mut cfg_tdc;

    let derivs_mat = bag.get_weights_derivatives(1)[0].clone();
    let stats1 = bag.get_input_statistics(1);

    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20), (1, 1e20)]);
    let derivs: BTreeMap<usize, Vec<Array2<f64>>> =
        BTreeMap::from([(0, Vec::new()), (1, vec![derivs_mat])]);
    let stats: BTreeMap<usize, Vec<InputStatistics>> =
        BTreeMap::from([(0, Vec::new()), (1, stats1)]);

    #[rustfmt::skip]
    let row0: [f64; 18] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 1.0];
    #[rustfmt::skip]
    let row1: [f64; 18] = [0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 3.0, 1.0];
    let m0 = vec![ChannelResult::from_row(&row0)];
    let m1 = vec![ChannelResult::from_row(&row1)];

    let _lock = CWD_LOCK.lock().unwrap();
    let _cwd = CwdGuard::enter(tmp.path());
    let out = "run.mat";
    let mut cost_mem = [0.0; 2];
    let mut bad_classif = [0.0; 2];
    let mut cost_lid = [0.0; 2];
    let mut bad_classif_lid = [0.0; 2];

    bag.save_and_update(
        out,
        &[m0, m1],
        &mut best_cost,
        &derivs,
        &stats,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    )
    .unwrap();

    // Row0: cost = sums(4)/sums(17) = 2/1 = 2.0; badClassif = means(2) = 1.0;
    // costLID: sums(16) == 0 -> NOT divided, stays sums(14) = 0.0;
    // badLIDClassif = 100 - means(15) = 100 - 100 = 0.0.
    assert_eq!(cost_mem[0], 2.0);
    assert_eq!(bad_classif[0], 1.0);
    assert_eq!(cost_lid[0], 0.0);
    assert_eq!(bad_classif_lid[0], 0.0);

    // Row1: cost = 4/1 = 4.0; badClassif = 3.0; costLID = 6/3 = 2.0;
    // badLIDClassif = 100 - 0 = 100.0.
    assert_eq!(cost_mem[1], 4.0);
    assert_eq!(bad_classif[1], 3.0);
    assert_eq!(cost_lid[1], 2.0);
    assert_eq!(bad_classif_lid[1], 100.0);
}

/// `costLID = -1.0` when `totalSpeechDuration < 1e-3` (`:465`), applied AFTER
/// `save_weights` and BEFORE `update_weights`. Not independently observable via
/// algo 3's own update criterion (algo 3 ignores costLID), so this asserts via
/// the test-only hook recording the value `update_weights` actually received.
#[test]
fn update_called_with_neg_costlid() {
    let weights_src = ref_dir().join("phase0/NNweights_config1.bin");
    let tmp = tempfile::tempdir().unwrap();
    let weights_path = tmp.path().join("weights.bin");
    std::fs::copy(&weights_src, &weights_path).unwrap();

    let mut bag = algo3_bag(&weights_path);
    let derivs_mat = bag.get_weights_derivatives(0)[0].clone();
    let stats0 = bag.get_input_statistics(0);

    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20)]);
    let derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::from([(0, vec![derivs_mat])]);
    let stats: BTreeMap<usize, Vec<InputStatistics>> = BTreeMap::from([(0, stats0)]);

    // means(5) (col5, totalSignalDuration/rows) and means(6) (col6, speech
    // duration/rows) both 0 across the single row -> totalSpeechDuration = 0 <
    // 1e-3. costLID numerator/denominator nonzero so costLID would be nonzero
    // pre-gate (isolating the gate, not a division-by-zero coincidence).
    #[rustfmt::skip]
    let row: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 3.0, 1.0];
    let m = vec![ChannelResult::from_row(&row)];

    let _lock = CWD_LOCK.lock().unwrap();
    let _cwd = CwdGuard::enter(tmp.path());
    let out = "run.mat";
    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        out,
        &[m],
        &mut best_cost,
        &derivs,
        &stats,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    )
    .unwrap();

    // cost_mem itself carries the PRE-gate costLID (9/3 = 3.0) - the row slices are
    // written before the :465 override.
    assert_eq!(cost_lid[0], 3.0);

    // But update_weights, called AFTER the :465 override, must have received -1.0.
    match bag.processor(0) {
        speech::engine::bag_of_processors::Processor::Spectral(seg) => {
            assert_eq!(seg.last_update_cost_lid_for_test(), Some(-1.0));
        }
        _ => panic!("expected algo 3 (Spectral)"),
    }
}
