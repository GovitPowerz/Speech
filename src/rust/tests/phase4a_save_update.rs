//! Phase 4a Task 6: `BagOfProcessors::save_and_update` (`BagOfProcessors.cpp:409-471`),
//! the best-cost gate (`saveWeights`, no files since issue #136) and `update_weights`
//! (`:148-205`).
//!
//! `aggregation_bit_exact` isolates the column-sum/mean arithmetic on an algo-1
//! bag (TDC, NN-free): the gate and the update are no-ops for algo 1/2 in
//! the legacy (no matching `if` branch), so only the aggregation math is exercised.
//! `best_cost_gate_fires_and_skips`/`best_pack_is_the_pre_update_pack_and_the_fold`/
//! `cost_mem_rows_written`/`update_called_with_neg_costlid` use a real algo-3 bag (the
//! only ported NN algo with a live save/update criterion) built the same way Task 3/4's
//! inline tests do: `phase0/1_worker_1.config` + `phase0/NNweights_config1.bin` via
//! `BLSTM_weightsFile`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use indexmap::IndexMap;
use ndarray::Array2;

use speech::engine::bag_of_processors::BagOfProcessors;
use speech::engine::channel_result::ChannelResult;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
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
/// net, so the best-cost gate captures a real pack and `update_weights` runs the
/// real iRPROP- trainer without crashing.
fn algo3_bag(weights_path: &std::path::Path) -> BagOfProcessors {
    let mut cfg = with_bag_keys(load_config("phase0/1_worker_1.config"), 3);
    cfg.insert(
        "BLSTM_weightsFile".to_string(),
        weights_path.to_str().unwrap().to_string(),
    );
    BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg)).unwrap()
}

fn algo1_bag() -> BagOfProcessors {
    let cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
    let mut cfgs = vec![cfg];
    BagOfProcessors::from_configs(&mut cfgs).unwrap()
}

fn empty_derivs(nb_of_conf: usize) -> BTreeMap<usize, Vec<Array2<f64>>> {
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
    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        &[m.iter().collect()],
        &mut best_cost,
        &derivs,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    );

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
    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        &[m.iter().collect()],
        &mut best_cost,
        &derivs,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    );

    // sums(17) == 0 -> cost NOT divided, stays sums(4) = 7.0.
    assert_eq!(cost_mem[0], 7.0);
    // sums(16) == 0 -> costLID NOT divided, stays sums(14) = 15.0.
    assert_eq!(cost_lid[0], 15.0);
}

/// Algo-3 bag, real net: epoch A cost 5.0 beats the 1e20 seed -> the gate fires
/// (`fires == 1`) and best_cost updates; epoch B cost 7.0 does NOT beat 5.0 -> no
/// fire: the captured pack is still epoch A's (`fires` still 1), best_cost unchanged.
#[test]
fn best_cost_gate_fires_and_skips() {
    let weights_src = ref_dir().join("phase0/NNweights_config1.bin");
    let tmp = tempfile::tempdir().unwrap();
    let weights_path = tmp.path().join("weights.bin");
    std::fs::copy(&weights_src, &weights_path).unwrap();

    let mut bag = algo3_bag(&weights_path);
    let derivs: BTreeMap<usize, Vec<Array2<f64>>> =
        BTreeMap::from([(0, vec![bag.get_weights_derivatives(0)[0].clone()])]);
    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20)]);

    // Col 4 (cost numerator) = 5.0, col 17 (denom) = 1.0 -> cost = 5.0.
    #[rustfmt::skip]
    let row_a: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let m_a = [ChannelResult::from_row(&row_a)];

    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        &[m_a.iter().collect()],
        &mut best_cost,
        &derivs,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    );

    assert_eq!(best_cost[&0], 5.0);
    let pack_a = bag.best_pack_for_test(0, 0).expect("epoch A fires").clone();
    assert_eq!(pack_a.fires, 1);

    // Cost 7.0 - worse than the 5.0 best - must NOT fire.
    #[rustfmt::skip]
    let row_b: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let m_b = [ChannelResult::from_row(&row_b)];

    bag.save_and_update(
        &[m_b.iter().collect()],
        &mut best_cost,
        &derivs,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    );

    assert_eq!(best_cost[&0], 5.0);
    assert_eq!(
        bag.best_pack_for_test(0, 0),
        Some(&pack_a),
        "epoch B did not fire: the pack and its fire count are epoch A's"
    );
}

/// Issue #136: a fire captures each net's pack as it stood BEFORE the epoch's update
/// and the folded derivatives that update consumed -- the legacy `saveWeights` ran
/// before `updateWeights` (`BagOfProcessors.cpp:464-466`). The crafted nonzero fold
/// moves the weights, so a capture taken after the update would miss `before`.
#[test]
fn best_pack_is_the_pre_update_pack_and_the_fold() {
    let weights_src = ref_dir().join("phase0/NNweights_config1.bin");
    let tmp = tempfile::tempdir().unwrap();
    let weights_path = tmp.path().join("weights.bin");
    std::fs::copy(&weights_src, &weights_path).unwrap();

    let mut bag = algo3_bag(&weights_path);
    let before = bag.get_weights(0)[0].clone();
    let fold = Array2::from_shape_fn((before.len(), 2), |(r, c)| {
        if c == 1 {
            1.0
        } else {
            (r % 7) as f64 * 0.01 - 0.03
        }
    });
    let derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::from([(0, vec![fold.clone()])]);
    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20)]);

    #[rustfmt::skip]
    let row: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let m = [ChannelResult::from_row(&row)];
    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        &[m.iter().collect()],
        &mut best_cost,
        &derivs,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    );

    assert_ne!(
        bag.get_weights(0)[0],
        before,
        "the update moved the weights"
    );
    let pack = bag.best_pack_for_test(0, 0).expect("the first epoch fires");
    assert_eq!(pack.weights, before, "the pack is the pre-update one");
    assert_eq!(
        pack.derivs, fold,
        "the derivs are the fold the update consumed"
    );
    assert_eq!(pack.fires, 1);
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
    let mut bag = BagOfProcessors::from_configs(&mut cfgs).unwrap();
    let _ = &mut cfg_tdc;

    let derivs_mat = bag.get_weights_derivatives(1)[0].clone();

    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20), (1, 1e20)]);
    let derivs: BTreeMap<usize, Vec<Array2<f64>>> =
        BTreeMap::from([(0, Vec::new()), (1, vec![derivs_mat])]);

    #[rustfmt::skip]
    let row0: [f64; 18] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 1.0];
    #[rustfmt::skip]
    let row1: [f64; 18] = [0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 3.0, 1.0];
    let m0 = [ChannelResult::from_row(&row0)];
    let m1 = [ChannelResult::from_row(&row1)];

    let mut cost_mem = [0.0; 2];
    let mut bad_classif = [0.0; 2];
    let mut cost_lid = [0.0; 2];
    let mut bad_classif_lid = [0.0; 2];

    bag.save_and_update(
        &[m0.iter().collect(), m1.iter().collect()],
        &mut best_cost,
        &derivs,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    );

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
/// the best-cost gate and BEFORE `update_weights`. Not independently observable via
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

    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20)]);
    let derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::from([(0, vec![derivs_mat])]);

    // means(5) (col5, totalSignalDuration/rows) and means(6) (col6, speech
    // duration/rows) both 0 across the single row -> totalSpeechDuration = 0 <
    // 1e-3. costLID numerator/denominator nonzero so costLID would be nonzero
    // pre-gate (isolating the gate, not a division-by-zero coincidence).
    #[rustfmt::skip]
    let row: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 3.0, 1.0];
    let m = [ChannelResult::from_row(&row)];

    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        &[m.iter().collect()],
        &mut best_cost,
        &derivs,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    );

    // cost_mem itself carries the PRE-gate costLID (9/3 = 3.0) - the row slices are
    // written before the :465 override.
    assert_eq!(cost_lid[0], 3.0);

    // But update_weights, called AFTER the :465 override, must have received -1.0.
    assert_eq!(bag.last_update_cost_lid_for_test(0), Some(-1.0));
}
