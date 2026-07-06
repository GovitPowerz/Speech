//! Phase 3 Task 4: `LstmLayer::feed_backward` (peephole BPTT) golden tests.
//!
//! Ported from `LSTMLayer.cpp:518-732` (`feedBackward` + `feedBackwardReverse`).
//! Goldens dumped by `tools/oracle_harness/main.cpp`'s `lstmBackwardLoop` /
//! `lstmBackwardReverseLoop` (Task 1) at synthetic `I=2,O=2,T=5` (odd length),
//! via `scripts/extract_phase3_fixtures.py`:
//!
//! - `lstm_bwd_deltasprev_<v>.bin` (T x I) for v in {peep_all,peep_none,reverse,subsample}
//! - `lstm_bwd_derivs_<v>.bin` (Nx2, col0 summed deriv, col1 frame count) same v
//! - `lstm_bwd_signal_derivs.bin` (Nx2, the 1-col width-tolerance case)
//!
//! The reimpl-produced dumps are the golden (Eigen GEMM diverges at NN-wide k; the
//! extractor asserts the reimpl == the REAL compiled class at these shapes,
//! `NN_TOL site=lstm_backward_variants max_ulp=0`).
//!
//! Comparator discipline: `deltas_previous_layer` and the deriv col0 traverse the
//! gate/cell activation derivatives (`gates_deriv` -> nothing transcendental, but
//! `maxmin2_deriv`/`identity_deriv` fold `sinh`, and the forward caches these
//! feed on are themselves `asinh`/`sigmoid` outputs), so they are canary-gated
//! (`assert_oracle_eq`). Col1 (frame count) is a structural integer check.

mod common;

use ndarray::Array2;
use speech::nn::layers::LstmLayer;

const I: usize = 2;
const O: usize = 2;
const T: usize = 5;

/// Deterministic synthetic weights: mirrors the harness's `synthFlat` lambda:
/// `flat[k] = ((k*11+3) % 97)/97.0 - 0.5`.
fn synth_flat(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5)
        .collect()
}

/// Deterministic input: mirrors the harness's `makeInput` lambda:
/// `x[t,j] = ((t*37 + j*53 + 7) % 101)/101.0 - 0.5`.
fn make_input(t: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((t, cols), |(t, j)| {
        ((t * 37 + j * 53 + 7) % 101) as f64 / 101.0 - 0.5
    })
}

/// Deterministic incoming deltas: mirrors the harness's `makeDeltas` lambda:
/// `dd[t,j] = ((t*29 + j*41 + 5) % 83)/83.0 - 0.5`.
fn make_deltas(t: usize, o: usize) -> Array2<f64> {
    Array2::from_shape_fn((t, o), |(t, j)| {
        ((t * 29 + j * 41 + 5) % 83) as f64 / 83.0 - 0.5
    })
}

fn build_layer(cp: bool, gp: bool, grp: bool) -> LstmLayer {
    let mut layer = LstmLayer::new(I, O, cp, gp, grp);
    let flat = synth_flat(layer.nb_of_weights());
    layer.set_weights(&flat);
    layer
}

/// Forward + backward one variant; returns (deltas_prev, derivs Nx2).
fn run_variant(cp: bool, gp: bool, grp: bool, ratio: usize) -> (Array2<f64>, Vec<[f64; 2]>) {
    let mut layer = build_layer(cp, gp, grp);
    let input = make_input(T, I);
    let deltas = make_deltas(T, O);
    let mut out = Array2::<f64>::zeros((T, O));
    layer.feed_forward(&input, &mut out, false);
    let dpl = layer.feed_backward(&input, &out, &deltas, ratio, false);
    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    (dpl, derivs)
}

fn derivs_col0(derivs: &[[f64; 2]]) -> Array2<f64> {
    let col0: Vec<f64> = derivs.iter().map(|r| r[0]).collect();
    Array2::from_shape_vec((col0.len(), 1), col0).unwrap()
}

fn want_col0(want: &Array2<f64>) -> Array2<f64> {
    want.column(0).to_owned().insert_axis(ndarray::Axis(1))
}

/// `peep_all` (all three peephole flags on): the reference variant exercising every
/// peephole cross-term (rows 0-11) in the gate-derivative chain.
#[test]
fn backward_peep_all() {
    let (dpl, derivs) = run_variant(true, true, true, 1);

    let want_dpl = common::load_bin_phase3("lstm_bwd_deltasprev_peep_all.bin");
    common::assert_oracle_eq(&dpl, &want_dpl, "peep_all deltas_prev");

    let want_derivs = common::load_bin_phase3("lstm_bwd_derivs_peep_all.bin");
    assert_eq!(
        derivs.len(),
        want_derivs.nrows(),
        "peep_all derivs row count"
    );
    common::assert_oracle_eq(
        &derivs_col0(&derivs),
        &want_col0(&want_derivs),
        "peep_all derivs col0",
    );
    for (idx, row) in derivs.iter().enumerate() {
        assert_eq!(
            row[1],
            want_derivs[[idx, 1]],
            "peep_all col1 (count) at row {idx}"
        );
    }
}

/// `peep_none` (all flags off): the peephole cross-terms drop entirely -- a distinct
/// golden. A test set that only covered `peep_all` would let a peephole-row mutation
/// hide behind the reimpl; the two goldens together bracket the peephole block.
#[test]
fn backward_peep_none() {
    let (dpl, derivs) = run_variant(false, false, false, 1);

    let want_dpl = common::load_bin_phase3("lstm_bwd_deltasprev_peep_none.bin");
    common::assert_oracle_eq(&dpl, &want_dpl, "peep_none deltas_prev");

    let want_derivs = common::load_bin_phase3("lstm_bwd_derivs_peep_none.bin");
    assert_eq!(
        derivs.len(),
        want_derivs.nrows(),
        "peep_none derivs row count"
    );
    common::assert_oracle_eq(
        &derivs_col0(&derivs),
        &want_col0(&want_derivs),
        "peep_none derivs col0",
    );
    for (idx, row) in derivs.iter().enumerate() {
        assert_eq!(
            row[1],
            want_derivs[[idx, 1]],
            "peep_none col1 (count) at row {idx}"
        );
    }
}

/// `reverse` variant via `feed_backward_reverse` (`:725-732`): reverse
/// input/output/deltas, run the forward-order backward, reverse the returned deltas.
/// The forward is `feed_forward_reverse`, which leaves the caches in
/// reversed-input storage order; the backward consumes them AS-IS (risk R9).
///
/// The second assertion pins exactly that: running the plain forward-order
/// `feed_backward` on the SAME (reverse-forward) layer -- i.e. NOT reversing the
/// deltas/input, and therefore reading the reversed caches in forward-time order --
/// gives a DIFFERENT result. This proves the reversed-storage consumption is
/// load-bearing (un-reversing the caches would change the answer).
#[test]
fn backward_reverse_caches_not_unreversed() {
    let mut layer = build_layer(true, true, true);
    let input = make_input(T, I);
    let deltas = make_deltas(T, O);
    let mut out = Array2::<f64>::zeros((T, O));
    layer.feed_forward_reverse(&input, &mut out, false);
    let dpl = layer.feed_backward_reverse(&input, &out, &deltas, 1, false);

    let want_dpl = common::load_bin_phase3("lstm_bwd_deltasprev_reverse.bin");
    common::assert_oracle_eq(&dpl, &want_dpl, "reverse deltas_prev");

    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    let want_derivs = common::load_bin_phase3("lstm_bwd_derivs_reverse.bin");
    assert_eq!(
        derivs.len(),
        want_derivs.nrows(),
        "reverse derivs row count"
    );
    common::assert_oracle_eq(
        &derivs_col0(&derivs),
        &want_col0(&want_derivs),
        "reverse derivs col0",
    );
    for (idx, row) in derivs.iter().enumerate() {
        assert_eq!(
            row[1],
            want_derivs[[idx, 1]],
            "reverse col1 (count) at row {idx}"
        );
    }

    // Structural R9 assertion: the same reverse-forward caches consumed WITHOUT the
    // reverse wrapper's input/delta flip (plain forward-order backward) diverge.
    // `feed_forward_reverse` already ran above, so the layer holds the reversed
    // caches; a plain `feed_backward` reads them in forward-time order.
    let mut layer2 = build_layer(true, true, true);
    let mut out2 = Array2::<f64>::zeros((T, O));
    layer2.feed_forward_reverse(&input, &mut out2, false);
    let dpl_plain = layer2.feed_backward(&input, &out2, &deltas, 1, false);
    let differs = dpl
        .iter()
        .zip(dpl_plain.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        differs,
        "reverse wrapper must differ from a plain forward-order backward on the reversed caches (R9)"
    );
}

/// `subsample` (`invSubSamplingRatio=2`): all four deriv blocks are multiplied by
/// the ratio at end of call (`:710-715`). col0 must equal `peep_all` col0 * 2
/// (bit-exact scalar multiply on the whole deriv vector); col1 (the frame count)
/// is UNCHANGED (the ratio scales magnitude, not the count).
#[test]
fn subsample_scales_derivs() {
    let (_dpl_base, derivs_base) = run_variant(true, true, true, 1);
    let (_dpl, derivs) = run_variant(true, true, true, 2);

    let want_derivs = common::load_bin_phase3("lstm_bwd_derivs_subsample.bin");
    assert_eq!(
        derivs.len(),
        want_derivs.nrows(),
        "subsample derivs row count"
    );

    // Golden: dumped subsample derivs (canary-gated col0, structural col1).
    common::assert_oracle_eq(
        &derivs_col0(&derivs),
        &want_col0(&want_derivs),
        "subsample derivs col0",
    );

    // Ratio relation vs peep_all: EXACTLY 2x on col0, unchanged col1. Bit-exact
    // (the legacy `*= invSubSamplingRatio` is a plain scalar multiply; doubling a
    // finite f64 is exact).
    assert_eq!(derivs.len(), derivs_base.len());
    for (idx, (r2, r1)) in derivs.iter().zip(derivs_base.iter()).enumerate() {
        assert_eq!(
            r2[0],
            r1[0] * 2.0,
            "subsample col0 at row {idx} == 2x peep_all"
        );
        assert_eq!(
            r2[1], r1[1],
            "subsample col1 (count) unchanged at row {idx}"
        );
    }
}

/// Width tolerance (`:534-543`, spec S11.5): a 1-col signal input into the I=2
/// layer zero-pads the input matrix's second row. The dead weight rows (input col
/// 1, all four gate blocks) accumulate EXACTLY `0.0` in `input_weights_derivatives`
/// -- their `input.col(row)` entry is a literal zero, so the outer-product
/// contribution is zero regardless of the gate delta. Gradient shape stays
/// layer-native (I x 4O). Strict bits on the zero rows.
#[test]
fn width_tolerance_dead_rows_zero() {
    let mut layer = build_layer(true, true, true);
    let input = make_input(T, 1); // 1-col signal into an I=2 layer
    let deltas = make_deltas(T, O);
    let mut out = Array2::<f64>::zeros((T, O));
    layer.feed_forward(&input, &mut out, false);
    let _dpl = layer.feed_backward(&input, &out, &deltas, 1, false);

    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    let want = common::load_bin_phase3("lstm_bwd_signal_derivs.bin");
    assert_eq!(derivs.len(), want.nrows(), "signal derivs row count");
    // col0 folds gate derivs -> canary-gated; col1 structural.
    common::assert_oracle_eq(
        &derivs_col0(&derivs),
        &want_col0(&want),
        "signal derivs col0",
    );
    for (idx, row) in derivs.iter().enumerate() {
        assert_eq!(row[1], want[[idx, 1]], "signal col1 (count) at row {idx}");
    }

    // Dead weight rows: input_weights_derivatives is the first I*4O = 2*8 = 16
    // elements, column-major (I x 4O). Row 1 (input col 1) is EXACTLY zero in every
    // gate column, since the padded input row is a literal 0.0.
    let four_o = 4 * O;
    for gate_col in 0..four_o {
        let flat_idx = gate_col * I + 1; // (ii=1, jj=gate_col), column-major
        assert_eq!(
            derivs[flat_idx][0], 0.0,
            "dead input row (col 1, gate col {gate_col}) has exactly-zero deriv"
        );
    }
}

/// The `row==0` forget branch (`:648-657`): a `T=1` sequence runs ONLY the
/// `row==0` iteration, which takes the no-cellstate forget branch (no
/// `_CellStates.row(row-1)` read). Structural: it must not index out of bounds
/// and must match the reimpl on the same operands (recomputed here in-process
/// with the harness contract -- there is no committed T=1 dump, so the assertion
/// is that the call completes and the deriv shape is layer-native).
#[test]
fn row_zero_forget_branch() {
    let mut layer = build_layer(true, true, true);
    let input = make_input(1, I);
    let deltas = make_deltas(1, O);
    let mut out = Array2::<f64>::zeros((1, O));
    layer.feed_forward(&input, &mut out, false);
    // Must not panic (no `_CellStates.row(-1)` read on the row==0 branch).
    let dpl = layer.feed_backward(&input, &out, &deltas, 1, false);
    assert_eq!(dpl.dim(), (1, I), "row==0 deltas_prev shape T x I");

    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    assert_eq!(
        derivs.len(),
        layer.nb_of_weights(),
        "row==0 derivs layer-native"
    );
    // count == deltas.nrows() == 1 for a single-row sequence.
    for row in &derivs {
        assert_eq!(row[1], 1.0, "row==0 count == 1");
    }
}

/// `nb_of_seq_fed_backward += deltas.nrows()` (`:720`, `linesNb = deltas.rows()`):
/// one call sets the count to the delta row count; a second call accumulates.
#[test]
fn count_is_deltas_rows() {
    let mut layer = build_layer(true, true, true);
    let input = make_input(T, I);
    let deltas = make_deltas(T, O);
    let mut out = Array2::<f64>::zeros((T, O));
    layer.feed_forward(&input, &mut out, false);

    layer.feed_backward(&input, &out, &deltas, 1, false);
    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    for row in &derivs {
        assert_eq!(row[1], T as f64, "count after 1 call == deltas.nrows()");
    }

    layer.feed_backward(&input, &out, &deltas, 1, false);
    let mut derivs2 = Vec::new();
    layer.get_weights_derivatives(&mut derivs2);
    for row in &derivs2 {
        assert_eq!(row[1], 2.0 * T as f64, "count after 2 calls accumulates");
    }
}

#[test]
fn reset_zeroes_all() {
    let mut layer = build_layer(true, true, true);
    let input = make_input(T, I);
    let deltas = make_deltas(T, O);
    let mut out = Array2::<f64>::zeros((T, O));
    layer.feed_forward(&input, &mut out, false);
    layer.feed_backward(&input, &out, &deltas, 1, false);
    layer.reset_weights_derivatives();

    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    for row in &derivs {
        assert_eq!(row[0], 0.0, "deriv zeroed after reset");
        assert_eq!(row[1], 0.0, "count zeroed after reset");
    }
}
