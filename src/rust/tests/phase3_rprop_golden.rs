//! Phase 3 Task 7: Rprop iRPROP- trainer golden tests.
//!
//! The REAL compiled `Rprop::updateWeights` (`Rprop.cpp:9-59`) IS the golden
//! directly (design spec S2 tier 1): pure scalar, no libm/GEMM, bit-portable
//! on every platform. All asserts are STRICT BITS (`assert_bits_eq`), no
//! canary gate, on every platform -- unlike every other Phase 3 suite.
//!
//! Two trajectories dumped by `tools/oracle_harness/main.cpp`'s Task 7 stage
//! via `scripts/extract_phase3_fixtures.py`:
//! - Trajectory A (5 elements, 5 steps): every branch except the delta clamps.
//! - Trajectory B (2 elements, 48 steps): a dedicated clamp trajectory hitting
//!   both `max_delta` (step 18) and `min_delta` (step 48).
//!
//! The per-step/per-element branch record asserted in `all_branches_fired` is
//! an INDEPENDENT re-derivation (this file's `branch_record`, mirroring
//! `scripts/extract_phase3_fixtures.py::_rprop_branch_record`) of the
//! deriv-sign x prev-deriv-sign x cost-comparison state machine over the same
//! trajectory inputs -- not a read-back of the Rust `Rprop`'s internal state,
//! so it structurally proves non-vacuity rather than merely asserting
//! "something happened".

mod common;

use common::{assert_bits_eq, load_bin_phase3};
use ndarray::Array2;
use speech::nn::train::Rprop;

const ETA_PLUS: f64 = 1.2;
const ETA_MIN: f64 = 0.5;
const MAX_DELTA: f64 = 0.2;
const MIN_DELTA: f64 = 1e-9;
const INIT_DELTA: f64 = 1e-2;

// --- Trajectory definitions (must match tools/oracle_harness/main.cpp's Task 7
// stage + scripts/extract_phase3_fixtures.py's TRAJ_A_*/TRAJ_B_* exactly). ---

const TRAJ_A_N: usize = 5;
const TRAJ_A_STEPS: usize = 5;
const TRAJ_A_DERIVS: [[f64; TRAJ_A_N]; TRAJ_A_STEPS] = [
    [1.0, -1.0, 1.0, 0.0, 1.0],
    [1.0, -1.0, 1.0, 1.0, 0.0],
    [1.0, -1.0, -1.0, 1.0, 0.0],
    [1.0, -1.0, -1.0, -1.0, 0.0],
    [1.0, -1.0, -1.0, -1.0, 0.0],
];
const TRAJ_A_COSTS: [f64; TRAJ_A_STEPS] = [10.0, 9.0, 12.0, 8.0, 7.0];

const TRAJ_B_N: usize = 2;
const TRAJ_B_STEPS: usize = 48;

fn traj_b_derivs_costs() -> (Vec<[f64; TRAJ_B_N]>, Vec<f64>) {
    let mut derivs = Vec::with_capacity(TRAJ_B_STEPS);
    let mut costs = Vec::with_capacity(TRAJ_B_STEPS);
    let mut prev_cost = 5.0_f64;
    for step in 1..=TRAJ_B_STEPS {
        // step == 1 and step % 2 == 1 both yield +1.0 (step 1 is odd anyway);
        // the harness's C++ mirrors this same redundant-but-explicit condition
        // (`(step == 1) ? 1.0 : ((step % 2 == 1) ? 1.0 : -1.0)`), so keep the
        // shapes identical rather than collapsing the branch here.
        let d1 = if step % 2 == 1 { 1.0 } else { -1.0 };
        derivs.push([1.0, d1]);
        let cost = prev_cost + if step % 2 == 0 { 1.0 } else { -1.0 };
        costs.push(cost);
        prev_cost = cost;
    }
    (derivs, costs)
}

/// Branch tag per element per step, independent of the Rust `Rprop` impl
/// (mirrors the Python re-derivation in the extractor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    InitPos,
    InitNeg,
    InitZero,
    Grow,
    GrowClamped,
    ShrinkBacktrack,
    ShrinkNobacktrack,
    ShrinkClampedBacktrack,
    ShrinkClampedNobacktrack,
    Zero,
    ZeroPos,
    ZeroNeg,
}

fn branch_record(all_derivs: &[Vec<f64>], all_costs: &[f64], n: usize) -> Vec<Vec<Branch>> {
    let mut deltas: Option<Vec<f64>> = None;
    let mut prev_derivs: Vec<f64> = vec![0.0; n];
    let mut prev_cost = 0.0_f64;
    let mut record = Vec::with_capacity(all_derivs.len());

    for (step_idx, derivs) in all_derivs.iter().enumerate() {
        let cost = all_costs[step_idx];
        let mut branch = vec![Branch::Zero; n];
        match &mut deltas {
            None => {
                deltas = Some(vec![INIT_DELTA; n]);
                prev_derivs = derivs.clone();
                for j in 0..n {
                    let d = derivs[j];
                    branch[j] = if d == 0.0 {
                        Branch::InitZero
                    } else if d > 0.0 {
                        Branch::InitPos
                    } else {
                        Branch::InitNeg
                    };
                }
            }
            Some(deltas) => {
                let dtp: Vec<f64> = (0..n).map(|j| derivs[j] * prev_derivs[j]).collect();
                prev_derivs = derivs.clone();
                for j in 0..n {
                    if dtp[j] > 0.0 {
                        deltas[j] *= ETA_PLUS;
                        if deltas[j] > MAX_DELTA {
                            deltas[j] = MAX_DELTA;
                            branch[j] = Branch::GrowClamped;
                        } else {
                            branch[j] = Branch::Grow;
                        }
                    } else if dtp[j] < 0.0 {
                        deltas[j] *= ETA_MIN;
                        let clamped = deltas[j] < MIN_DELTA;
                        if clamped {
                            deltas[j] = MIN_DELTA;
                        }
                        let fired = prev_cost < cost;
                        branch[j] = match (clamped, fired) {
                            (true, true) => Branch::ShrinkClampedBacktrack,
                            (true, false) => Branch::ShrinkClampedNobacktrack,
                            (false, true) => Branch::ShrinkBacktrack,
                            (false, false) => Branch::ShrinkNobacktrack,
                        };
                        prev_derivs[j] = 0.0;
                    } else {
                        let d = derivs[j];
                        branch[j] = if d == 0.0 {
                            Branch::Zero
                        } else if d > 0.0 {
                            Branch::ZeroPos
                        } else {
                            Branch::ZeroNeg
                        };
                    }
                }
            }
        }
        prev_cost = cost;
        record.push(branch);
    }
    record
}

fn is_init(b: Branch) -> bool {
    matches!(b, Branch::InitPos | Branch::InitNeg | Branch::InitZero)
}
fn is_shrink_backtrack(b: Branch) -> bool {
    matches!(b, Branch::ShrinkBacktrack | Branch::ShrinkClampedBacktrack)
}
fn is_shrink_nobacktrack(b: Branch) -> bool {
    matches!(
        b,
        Branch::ShrinkNobacktrack | Branch::ShrinkClampedNobacktrack
    )
}
fn is_shrink_clamped(b: Branch) -> bool {
    matches!(
        b,
        Branch::ShrinkClampedBacktrack | Branch::ShrinkClampedNobacktrack
    )
}
fn is_zero_post(b: Branch) -> bool {
    matches!(b, Branch::Zero | Branch::ZeroPos | Branch::ZeroNeg)
}

fn col(v: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((v.len(), 1), v.to_vec()).unwrap()
}

/// Replay Trajectory A through the Rust `Rprop`, asserting bit-exactness at
/// EVERY step against the harness dumps.
#[test]
fn trajectory_a_bit_exact() {
    let mut rp = Rprop::new(INIT_DELTA);
    let mut weights = vec![1.0_f64; TRAJ_A_N];

    for step in 1..=TRAJ_A_STEPS {
        let derivs = TRAJ_A_DERIVS[step - 1];
        rp.update_weights(&derivs, &mut weights, TRAJ_A_COSTS[step - 1]);

        let want_weights = load_bin_phase3(&format!("rprop_trajA_step{step}_weights.bin"));
        let want_deltas = load_bin_phase3(&format!("rprop_trajA_step{step}_deltas.bin"));
        let want_dw = load_bin_phase3(&format!("rprop_trajA_step{step}_deltaweights.bin"));
        let want_pd = load_bin_phase3(&format!("rprop_trajA_step{step}_prevderivs.bin"));

        assert_bits_eq(
            &col(&weights),
            &want_weights,
            &format!("trajA step{step} weights"),
        );
        assert_bits_eq(
            &col(rp.deltas()),
            &want_deltas,
            &format!("trajA step{step} deltas"),
        );
        assert_bits_eq(
            &col(rp.delta_weights()),
            &want_dw,
            &format!("trajA step{step} delta_weights"),
        );
        assert_bits_eq(
            &col(rp.prev_derivs()),
            &want_pd,
            &format!("trajA step{step} prev_derivs"),
        );
    }
}

/// Replay Trajectory B (the dedicated clamp trajectory) through the Rust
/// `Rprop`, asserting bit-exactness at EVERY step, incl. the two clamp hits.
#[test]
fn trajectory_b_bit_exact() {
    let (derivs, costs) = traj_b_derivs_costs();
    let mut rp = Rprop::new(INIT_DELTA);
    let mut weights = vec![1.0_f64; TRAJ_B_N];

    for step in 1..=TRAJ_B_STEPS {
        rp.update_weights(&derivs[step - 1], &mut weights, costs[step - 1]);

        let want_weights = load_bin_phase3(&format!("rprop_trajB_step{step}_weights.bin"));
        let want_deltas = load_bin_phase3(&format!("rprop_trajB_step{step}_deltas.bin"));
        let want_dw = load_bin_phase3(&format!("rprop_trajB_step{step}_deltaweights.bin"));
        let want_pd = load_bin_phase3(&format!("rprop_trajB_step{step}_prevderivs.bin"));

        assert_bits_eq(
            &col(&weights),
            &want_weights,
            &format!("trajB step{step} weights"),
        );
        assert_bits_eq(
            &col(rp.deltas()),
            &want_deltas,
            &format!("trajB step{step} deltas"),
        );
        assert_bits_eq(
            &col(rp.delta_weights()),
            &want_dw,
            &format!("trajB step{step} delta_weights"),
        );
        assert_bits_eq(
            &col(rp.prev_derivs()),
            &want_pd,
            &format!("trajB step{step} prev_derivs"),
        );
    }

    // Both clamps must be bit-exactly at their bound by the recorded step.
    assert_eq!(
        rp.deltas()[0].to_bits(),
        MAX_DELTA.to_bits(),
        "e0 must be clamped to max_delta by step 48"
    );
    assert_eq!(
        rp.deltas()[1].to_bits(),
        MIN_DELTA.to_bits(),
        "e1 must be clamped to min_delta by step 48"
    );
}

/// S11.1 non-vacuity: assert STRUCTURALLY that the branch record (independent
/// re-derivation) covers all 7 required branch kinds across the two
/// trajectories -- not "hope they fired".
#[test]
fn all_branches_fired() {
    let a_derivs: Vec<Vec<f64>> = TRAJ_A_DERIVS.iter().map(|d| d.to_vec()).collect();
    let a_record = branch_record(&a_derivs, &TRAJ_A_COSTS, TRAJ_A_N);

    let (b_derivs, b_costs) = traj_b_derivs_costs();
    let b_derivs: Vec<Vec<f64>> = b_derivs.iter().map(|d| d.to_vec()).collect();
    let b_record = branch_record(&b_derivs, &b_costs, TRAJ_B_N);

    let all: Vec<Branch> = a_record
        .iter()
        .chain(b_record.iter())
        .flat_map(|step| step.iter().copied())
        .collect();

    assert!(all.iter().any(|&b| is_init(b)), "missing: first-call init");
    assert!(
        all.contains(&Branch::Grow),
        "missing: eta+ growth (unclamped)"
    );
    assert!(
        all.contains(&Branch::GrowClamped),
        "missing: eta+ growth hitting max_delta clamp"
    );
    assert!(
        all.iter().any(|&b| is_shrink_backtrack(b)),
        "missing: eta- shrink with backtrack FIRING"
    );
    assert!(
        all.iter().any(|&b| is_shrink_nobacktrack(b)),
        "missing: eta- shrink with backtrack NOT firing"
    );
    assert!(
        all.iter().any(|&b| is_shrink_clamped(b)),
        "missing: eta- shrink hitting min_delta clamp"
    );
    assert!(
        all.iter().any(|&b| is_zero_post(b)),
        "missing: the zero/post-backtrack-zero path"
    );

    // Cross-check the independent branch record against the harness-derived
    // clamp step numbers dumped in the manifest (spot check: same numbers this
    // file's simulation, the extractor's simulation, and the REAL Rprop must
    // all agree on).
    let max_clamp_step = b_record
        .iter()
        .position(|step| step.contains(&Branch::GrowClamped))
        .map(|i| i + 1);
    let min_clamp_step = b_record
        .iter()
        .position(|step| step.iter().any(|&b| is_shrink_clamped(b)))
        .map(|i| i + 1);
    assert_eq!(
        max_clamp_step,
        Some(18),
        "max_delta clamp must first hit at trajB step 18"
    );
    assert_eq!(
        min_clamp_step,
        Some(48),
        "min_delta clamp must first hit at trajB step 48"
    );
}

/// Structural: a single positive-deriv element moves DOWN (dw < 0); a
/// negative-deriv element moves UP (dw > 0) -- the INVERTED SIGN CONVENTION.
/// A sign-convention bug (flipping the two branches) would fail this.
#[test]
fn inverted_sign() {
    let mut rp = Rprop::new(INIT_DELTA);
    let mut weights = vec![1.0, 1.0];
    rp.update_weights(&[1.0, -1.0], &mut weights, 5.0);

    assert!(
        weights[0] < 1.0,
        "positive deriv must move the weight DOWN: {}",
        weights[0]
    );
    assert!(
        weights[1] > 1.0,
        "negative deriv must move the weight UP: {}",
        weights[1]
    );
    assert!(rp.delta_weights()[0] < 0.0);
    assert!(rp.delta_weights()[1] > 0.0);
}

/// S11.8 statefulness: the trainer state persists across >= 3 calls on ONE
/// instance. A fresh trainer per call (or a trainer that forgets prev_derivs/
/// deltas/prev_cost between calls) would land in the FIRST-CALL branch every
/// time and give different weights than the real stateful trajectory.
#[test]
fn stateful_across_calls() {
    let mut stateful = Rprop::new(INIT_DELTA);
    let mut w_stateful = vec![1.0, 1.0, 1.0];
    let calls: [([f64; 3], f64); 3] = [
        ([1.0, -1.0, 1.0], 10.0),
        ([1.0, -1.0, -1.0], 9.0),
        ([1.0, -1.0, -1.0], 12.0),
    ];
    for (derivs, cost) in calls {
        stateful.update_weights(&derivs, &mut w_stateful, cost);
    }

    // Three INDEPENDENT first-calls (a fresh Rprop each time) must differ: a
    // fresh instance takes the empty-deltas init branch every time (fixed dw
    // = +/-init_delta), never the eta+/eta-/backtrack machinery.
    let mut w_fresh = vec![1.0, 1.0, 1.0];
    for (derivs, cost) in calls {
        let mut fresh = Rprop::new(INIT_DELTA);
        fresh.update_weights(&derivs, &mut w_fresh, cost);
    }

    assert_ne!(
        w_stateful.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
        w_fresh.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
        "a stateful 3-call trajectory must differ from three independent first-calls"
    );

    // Cross-check the stateful run against the real dumped Trajectory-A-style
    // numbers is out of scope here (different trajectory) -- the point is the
    // STRUCTURAL divergence from the fresh-instance baseline, which any
    // state-dropping bug (e.g. resetting `deltas`/`prev_derivs` every call)
    // would collapse to zero.
}
