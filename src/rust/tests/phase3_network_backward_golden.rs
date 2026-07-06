//! Phase 3 Task 5: `Network<L>::feed_backward` container backward + the
//! SubSample/InvSubSample inversion (CRUX) golden tests.
//!
//! Ported from `NeuralNetwork.hpp:251-362` (`feedBackward` + `feedBackwardReverse`
//! + `feedBackwardDouble` + `getWeightsDerivatives`/`resetWeightsDerivatives`).
//!
//! Fixture provenance (Task 1 dumps, extended by Task 5 to also dump deltas_out):
//! goldens come from the harness `netBackwardLoop`/`netBackwardReverseLoop` reimpl
//! (`tools/oracle_harness/main.cpp`), the arbiter for the container backward. The
//! reimpl == the REAL compiled `NeuralNetwork<L>::feedBackward` on the Nx2 derivs
//! at these shapes (`NN_TOL site=net_lstm_backward_{fwd,rev}|net_dense_backward
//! max_ulp=0`); the returned deltas_out is a `deltas*W^T` GEMM (k=O) so the REAL
//! Eigen path diverges from the ascending reimpl in the last ULP -- the reimpl
//! dump is the golden, and the divergence is recorded as calibration
//! (`net_*_deltasout`, up to 16 ULP for the LSTM nets).
//!
//! Nets (BINDING per Task 5 standards -- consume Task 1's container goldens):
//! - LSTM `[3,4,2]` sub `[2,1]`, T=11: layer-0 `_SubSampling[0]=2 > 1` -- the
//!   SubSample/InvSubSample inversion + running `inv_sub_sampling_ratio` crux (the
//!   [4,1] real net's layer-0 arm in miniature). fwd + reverse drivers.
//! - Dense `[4,3,2]` sub `[1,2]`, T=11: layer-1 (interior/last) `_SubSampling[1]=2`
//!   -- the SubSample on the LAST layer + the multi-layer interior loop.
//! - Fix-wave 1 (review finding 1): LSTM `[2,2]` sub `[2]`/`[1]`, T=7 -- the
//!   SINGLE-layer branch (`neuron_nb.len() == 2`, `NeuralNetwork.hpp:255-263`), a
//!   real-config-reachable path (`configs/legacy/LID_BLSTM.config`:
//!   `BLSTM_LSTMNeuronNb 11,12` / `BLSTM_LSTMSubSampling 4`) that had zero coverage
//!   from the two multi-layer nets above.
//!
//! Comparator discipline: deltas_out + deriv col0 traverse the layer activation
//! derivatives (asinh/sigmoid via the LSTM caches) -> canary-gated
//! (`assert_oracle_eq`). Col1 (frame count) + shapes are structural integer checks.

mod common;

use ndarray::Array2;
use speech::nn::layers::{LstmLayer, NeuronLayer};
use speech::nn::network::{Network, inv_sub_sample, sub_sample};

const T: usize = 11;

/// Harness synthetic flat-vector: `w[k] = ((k*11+3) % 97)/97.0 - 0.5`.
fn synth_flat(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5)
        .collect()
}

/// Harness deterministic input: `x[t,j] = ((t*37 + j*53 + 7) % 101)/101.0 - 0.5`.
fn make_input(t_len: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((t_len, cols), |(t, j)| {
        ((t * 37 + j * 53 + 7) % 101) as f64 / 101.0 - 0.5
    })
}

/// Harness deterministic delta seed: `dd[t,j] = ((t*29 + j*41 + 5) % 83)/83.0 - 0.5`.
fn make_deltas(t_len: usize, o: usize) -> Array2<f64> {
    Array2::from_shape_fn((t_len, o), |(t, j)| {
        ((t * 29 + j * 41 + 5) % 83) as f64 / 83.0 - 0.5
    })
}

/// LSTM net `[3,4,2]` sub `[2,1]` with the harness synthetic weights (all peepholes).
fn make_lstm_net() -> Network<LstmLayer> {
    let mut net = Network::<LstmLayer>::new(vec![3, 4, 2], vec![2, 1], |_id, i, o| {
        LstmLayer::new(i, o, true, true, true)
    });
    net.set_weights(&synth_flat(net.nb_of_weights()));
    net
}

/// Dense net `[4,3,2]` sub `[1,2]` with the harness synthetic weights.
fn make_dense_net() -> Network<NeuronLayer> {
    let mut net = Network::<NeuronLayer>::new(vec![4, 3, 2], vec![1, 2], |_id, i, o| {
        NeuronLayer::new(i, o)
    });
    net.set_weights(&synth_flat(net.nb_of_weights()));
    net
}

/// Nx2 harvest -> two Array2 columns (col0 deriv, col1 count) for the golden compare.
fn derivs_to_cols(net_derivs: &[[f64; 2]]) -> (Array2<f64>, Array2<f64>) {
    let n = net_derivs.len();
    let mut col0 = Array2::zeros((n, 1));
    let mut col1 = Array2::zeros((n, 1));
    for (k, row) in net_derivs.iter().enumerate() {
        col0[[k, 0]] = row[0];
        col1[[k, 0]] = row[1];
    }
    (col0, col1)
}

fn golden_col(golden: &Array2<f64>, col: usize) -> Array2<f64> {
    golden.slice(ndarray::s![.., col..col + 1]).to_owned()
}

// === Multi-layer LSTM net: SubSample/InvSubSample inversion (CRUX) ===========

#[test]
fn lstm_net_subsample_inversion_fwd() {
    // Net [3,4,2] sub [2,1], T=11. Layer 1 (no subsample) back-projects at
    // decimated resolution (5 rows); layer 0 (subsample 2) InvSubSamples its
    // deltasPrev from 5 -> 10 rows and multiplies the running ratio by 2. The
    // returned deltas_out is 10 x 3 (floor(11/2)*2, NOT 11 -- the trailing row is
    // dropped by SubSample and never restored -- the CRUX row-count quirk).
    let mut net = make_lstm_net();
    let input = make_input(T, 3);
    let out_rows = T / 2; // 5
    let seed = make_deltas(out_rows, 2);

    let mut output = Array2::zeros((out_rows, 2));
    net.reset_weights_derivatives();
    net.feed_forward(&input, &mut output);
    let deltas_out = net.feed_backward(&input, &output, &seed);

    // Structural: InvSubSampled back to floor(T/2)*2 = 10 rows (NOT the decimated 5,
    // NOT the full 11) x net input width 3.
    assert_eq!(
        deltas_out.dim(),
        (10, 3),
        "deltas_out InvSubSampled to floor(11/2)*2=10 rows x 3 cols (input.nrows() tail frame dropped)"
    );

    let golden_deltasout = common::load_bin_phase3("bwd_net_lstm_fwd_deltasout.bin");
    common::assert_oracle_eq(&deltas_out, &golden_deltasout, "lstm net fwd deltas_out");

    let mut harvested = Vec::new();
    net.get_weights_derivatives(&mut harvested);
    let (col0, col1) = derivs_to_cols(&harvested);
    let golden = common::load_bin_phase3("bwd_net_lstm_fwd_derivs.bin");
    assert_eq!(
        golden.dim(),
        (304, 2),
        "net [3,4,2] sub [2,1] has 304 weights"
    );
    common::assert_oracle_eq(&col0, &golden_col(&golden, 0), "lstm net fwd derivs col0");
    common::assert_bits_eq(&col1, &golden_col(&golden, 1), "lstm net fwd derivs col1");
}

#[test]
fn lstm_net_subsample_inversion_rev() {
    // The reverse driver: feed_forward_reverse then feed_backward_reverse. The caches
    // are stored time-reversed and consumed as-is (risk R9); the reverse layer kernel
    // reverses input/output/deltas, runs the forward-order backward, reverses the
    // returned deltas. Same [3,4,2] sub [2,1] net.
    let mut net = make_lstm_net();
    let input = make_input(T, 3);
    let out_rows = T / 2;
    let seed = make_deltas(out_rows, 2);

    let mut output = Array2::zeros((out_rows, 2));
    net.reset_weights_derivatives();
    net.feed_forward_reverse(&input, &mut output);
    let deltas_out = net.feed_backward_reverse(&input, &output, &seed);

    assert_eq!(deltas_out.dim(), (10, 3), "reverse deltas_out 10 x 3");
    let golden_deltasout = common::load_bin_phase3("bwd_net_lstm_rev_deltasout.bin");
    common::assert_oracle_eq(&deltas_out, &golden_deltasout, "lstm net rev deltas_out");

    let mut harvested = Vec::new();
    net.get_weights_derivatives(&mut harvested);
    let (col0, col1) = derivs_to_cols(&harvested);
    let golden = common::load_bin_phase3("bwd_net_lstm_rev_derivs.bin");
    common::assert_oracle_eq(&col0, &golden_col(&golden, 0), "lstm net rev derivs col0");
    common::assert_bits_eq(&col1, &golden_col(&golden, 1), "lstm net rev derivs col1");
}

#[test]
fn lstm_net_reverse_differs_from_forward() {
    // Non-vacuity: the reverse driver must produce a DIFFERENT deriv col0 than the
    // forward one (proves the reversal is real, not a no-op). Both nets are fresh.
    let mut fwd = make_lstm_net();
    let mut rev = make_lstm_net();
    let input = make_input(T, 3);
    let seed = make_deltas(T / 2, 2);
    let mut out_f = Array2::zeros((T / 2, 2));
    let mut out_r = Array2::zeros((T / 2, 2));
    fwd.reset_weights_derivatives();
    fwd.feed_forward(&input, &mut out_f);
    fwd.feed_backward(&input, &out_f, &seed);
    rev.reset_weights_derivatives();
    rev.feed_forward_reverse(&input, &mut out_r);
    rev.feed_backward_reverse(&input, &out_r, &seed);
    let mut df = Vec::new();
    let mut dr = Vec::new();
    fwd.get_weights_derivatives(&mut df);
    rev.get_weights_derivatives(&mut dr);
    let differ = df
        .iter()
        .zip(&dr)
        .any(|(a, b)| a[0].to_bits() != b[0].to_bits());
    assert!(
        differ,
        "reverse driver must diverge from forward (reversal is real)"
    );
}

// === Multi-layer dense net: SubSample on the last layer ======================

#[test]
fn dense_net_last_layer_subsample() {
    // Net [4,3,2] sub [1,2], T=11. Layer 0 (no subsample) forwards 11 rows; layer 1
    // (subsample 2) decimates layer-0's output 11 -> 5, back-projects at 5, and
    // InvSubSamples deltasPrev 5 -> 10. Layer 0 then back-projects at deltas.rows()=10
    // (< its input.rows()=11) while its count accumulates input.rows()=11 -- deltas_out
    // is 10 x 4. Exercises the interior/last-layer subsample arm.
    let mut net = make_dense_net();
    let input = make_input(T, 4);
    let out_rows = T / 2; // 5 (layer-1 SubSample(2))
    let seed = make_deltas(out_rows, 2);

    let mut output = Array2::zeros((out_rows, 2));
    net.reset_weights_derivatives();
    net.feed_forward(&input, &mut output);
    let deltas_out = net.feed_backward(&input, &output, &seed);

    assert_eq!(deltas_out.dim(), (10, 4), "dense deltas_out 10 x 4");
    let golden_deltasout = common::load_bin_phase3("bwd_net_dense_deltasout.bin");
    // The dense deltas_out final back-projection is k=O=3 and measured 0 ULP vs Eigen,
    // but keep the canary gate for the LSTM-parity comparator discipline.
    common::assert_oracle_eq(&deltas_out, &golden_deltasout, "dense net deltas_out");

    let mut harvested = Vec::new();
    net.get_weights_derivatives(&mut harvested);
    let (col0, col1) = derivs_to_cols(&harvested);
    let golden = common::load_bin_phase3("bwd_net_dense_derivs.bin");
    assert_eq!(
        golden.dim(),
        (29, 2),
        "net [4,3,2] sub [1,2] has 29 weights"
    );
    common::assert_oracle_eq(&col0, &golden_col(&golden, 0), "dense net derivs col0");
    common::assert_bits_eq(&col1, &golden_col(&golden, 1), "dense net derivs col1");
}

#[test]
fn dense_net_count_reflects_input_rows_not_delta_rows() {
    // Structural: layer 0's frame count is input.rows()=11 (NeuronLayer.cpp:199), NOT
    // the 10 delta rows it back-projects over. The count lives in col1; the layer-0
    // block is the FIRST 15 rows (dense [4,3] = 3*(4+1)). The layer-1 block (the last
    // 14 rows, dense [6,2]) has count 5 (its input is the decimated 5-row layer-0 out).
    let mut net = make_dense_net();
    let input = make_input(T, 4);
    let seed = make_deltas(T / 2, 2);
    let mut output = Array2::zeros((T / 2, 2));
    net.reset_weights_derivatives();
    net.feed_forward(&input, &mut output);
    net.feed_backward(&input, &output, &seed);
    let mut harvested = Vec::new();
    net.get_weights_derivatives(&mut harvested);
    assert_eq!(harvested.len(), 29);
    assert_eq!(harvested[0][1], 11.0, "layer-0 count == input.rows() == 11");
    assert_eq!(harvested[14][1], 11.0, "layer-0 last weight count == 11");
    assert_eq!(harvested[15][1], 5.0, "layer-1 count == decimated 5 rows");
    assert_eq!(harvested[28][1], 5.0, "layer-1 last weight count == 5");
}

// === feed_backward_double: hcat composition identity =========================

#[test]
fn feed_backward_double_matches_manual_hcat() {
    // feedBackwardDouble (NeuralNetwork.hpp:353-362): hcat first|second (first LEFT),
    // run feed_backward, return deltas_out. Assert it equals a manual hcat + feed_backward
    // -- a strict-bits composition identity (no libm on the hcat path itself; the two
    // runs execute the SAME code on the SAME operands, so bit-identical regardless of
    // the internal activations). Dense net [10,4,2] sub [1,1], first|second each 5-wide.
    let build = || {
        let mut net = Network::<NeuronLayer>::new(vec![10, 4, 2], vec![1, 1], |_id, i, o| {
            NeuronLayer::new(i, o)
        });
        net.set_weights(&synth_flat(net.nb_of_weights()));
        net
    };
    let t = 7;
    let first = Array2::from_shape_fn((t, 5), |(r, j)| {
        ((r * 3 + j * 5 + 1) % 29) as f64 / 29.0 - 0.5
    });
    let second = Array2::from_shape_fn((t, 5), |(r, j)| {
        ((r * 7 + j * 11 + 2) % 31) as f64 / 31.0 - 0.5
    });
    let seed = make_deltas(t, 2);

    // Path A: feed_backward_double.
    let mut net_a = build();
    let mut out_a = Array2::zeros((t, 2));
    net_a.reset_weights_derivatives();
    net_a.feed_forward_double(&first, &second, &mut out_a);
    let da = net_a.feed_backward_double(&first, &second, &out_a, &seed);

    // Path B: manual hcat (first LEFT) then feed_backward.
    let mut hcat = Array2::zeros((t, 10));
    hcat.slice_mut(ndarray::s![.., ..5]).assign(&first);
    hcat.slice_mut(ndarray::s![.., 5..]).assign(&second);
    let mut net_b = build();
    let mut out_b = Array2::zeros((t, 2));
    net_b.reset_weights_derivatives();
    net_b.feed_forward(&hcat, &mut out_b);
    let db = net_b.feed_backward(&hcat, &out_b, &seed);

    common::assert_bits_eq(
        &da,
        &db,
        "feed_backward_double == manual hcat + feed_backward",
    );

    let mut ha = Vec::new();
    let mut hb = Vec::new();
    net_a.get_weights_derivatives(&mut ha);
    net_b.get_weights_derivatives(&mut hb);
    let (a0, _) = derivs_to_cols(&ha);
    let (b0, _) = derivs_to_cols(&hb);
    common::assert_bits_eq(&a0, &b0, "double derivs == manual derivs");
}

#[test]
fn feed_backward_double_empty_first_is_empty() {
    // Empty first (0 rows) -> empty deltas_out (:355).
    let mut net = Network::<NeuronLayer>::new(vec![10, 4, 2], vec![1, 1], |_id, i, o| {
        NeuronLayer::new(i, o)
    });
    net.set_weights(&synth_flat(net.nb_of_weights()));
    let first = Array2::<f64>::zeros((0, 5));
    let second = Array2::<f64>::zeros((0, 5));
    let out = Array2::<f64>::zeros((0, 2));
    let seed = Array2::<f64>::zeros((0, 2));
    let d = net.feed_backward_double(&first, &second, &out, &seed);
    assert_eq!(d.nrows(), 0, "empty first -> empty deltas_out");
}

#[test]
fn feed_backward_empty_input_is_empty() {
    // :254 -- empty input -> empty deltas_out.
    let mut net = make_lstm_net();
    let empty = Array2::<f64>::zeros((0, 3));
    let out = Array2::<f64>::zeros((0, 2));
    let seed = Array2::<f64>::zeros((0, 2));
    let d = net.feed_backward(&empty, &out, &seed);
    assert_eq!(d.nrows(), 0, "empty input -> empty deltas_out");
}

// === get_weights_derivatives layout (structural) =============================

#[test]
fn get_derivatives_layout_layer0_first() {
    // NeuralNetwork.hpp:98-111: the concatenated Nx2 has exactly nb_of_weights() rows,
    // layer-0-first (jj==0 seeds, then vertical hcat). Assert the row count and that a
    // known-index element maps to layer 0's first weight: harvest the deriv Nx2, and
    // SEPARATELY harvest just layer 0's Nx2 -- the first nb0 rows must be byte-identical.
    let mut net = make_dense_net();
    let input = make_input(T, 4);
    let seed = make_deltas(T / 2, 2);
    let mut output = Array2::zeros((T / 2, 2));
    net.reset_weights_derivatives();
    net.feed_forward(&input, &mut output);
    net.feed_backward(&input, &output, &seed);

    let mut full = Vec::new();
    net.get_weights_derivatives(&mut full);
    assert_eq!(full.len(), net.nb_of_weights(), "Nx2 rows == nb_of_weights");
    assert_eq!(full.len(), 29);

    // Layer 0 is dense [4,3] = 3*(4+1) = 15 weights; the container's first 15 rows must
    // equal the layer-0 harvest (proves layer-0-first ordering, the :104-105 jj==0 seed).
    let nb0 = NeuronLayer::new(4, 3).nb_of_weights();
    assert_eq!(nb0, 15);
    for row in &full[..nb0] {
        // Layer 0 block: count == 11 (input rows), matching the count test above.
        assert_eq!(
            row[1], 11.0,
            "first {nb0} rows are the layer-0 block (count 11)"
        );
    }
    // The very next row (index 15) is the layer-1 block start (count 5, decimated input).
    assert_eq!(full[nb0][1], 5.0, "row nb0 is the layer-1 block (count 5)");
}

// === Single-layer container backward branch (fix-wave 1, review finding 1) ===
//
// NeuralNetwork.hpp:255-263 -- the `neuron_nb.len() == 2` branch (a SINGLE layer).
// Both nets above are `neuron_nb.len() == 3` (multi-layer), so this branch had ZERO
// golden coverage prior to this fix wave. It is REACHABLE in a real config
// (configs/legacy/LID_BLSTM.config: `BLSTM_LSTMNeuronNb 11,12` with
// `BLSTM_LSTMSubSampling 4` -- a single-layer LSTM, subsample arm) and is
// structurally distinct from the multi-layer `jj==0` arm: the single-layer branch
// reads `deltas`/`deltas.rows()` (the SEED) directly on every call, whereas the
// multi-layer `jj==0` arm reads the running `deltas_out` on every iteration except
// when `n_layers == 1` degenerates it to the seed (which cannot happen for
// `neuron_nb.len() == 3`, `n_layers == 2`). LSTM `[2,2]` at T=7 (odd length, so the
// SubSample floor quirk (`floor(7/2)=3`) is exercised on THIS arm too): `subsample`
// (sub `[2]`) exercises the InvSubSample inversion; `plain` (sub `[1]`, legacy
// `:262`) exercises the no-subsample else-branch.

fn make_single_layer_net(sub: usize) -> Network<LstmLayer> {
    let mut net = Network::<LstmLayer>::new(vec![2, 2], vec![sub], |_id, i, o| {
        LstmLayer::new(i, o, true, true, true)
    });
    net.set_weights(&synth_flat(net.nb_of_weights()));
    net
}

#[test]
fn single_layer_subsample_inversion() {
    // T=7, sub [2]: forward decimates floor(7/2)=3 rows; feed_backward's
    // `neuron_nb.len() == 2` branch re-SubSamples the input, backs the layer at 3
    // rows, then InvSubSamples the returned deltas back to 3*2=6 rows (NOT 7 -- the
    // trailing SubSample row is dropped and never restored, same crux as the
    // multi-layer nets above).
    let mut net = make_single_layer_net(2);
    let input = make_input(7, 2);
    let out_rows = 7 / 2; // 3
    let seed = make_deltas(out_rows, 2);

    let mut output = Array2::zeros((out_rows, 2));
    net.reset_weights_derivatives();
    net.feed_forward(&input, &mut output);
    let deltas_out = net.feed_backward(&input, &output, &seed);

    assert_eq!(
        deltas_out.dim(),
        (6, 2),
        "single-layer subsample deltas_out InvSubSampled to floor(7/2)*2=6 rows x 2 cols"
    );
    let golden_deltasout = common::load_bin_phase3("bwd_net_single_subsample_deltasout.bin");
    common::assert_oracle_eq(
        &deltas_out,
        &golden_deltasout,
        "single-layer subsample deltas_out",
    );

    let mut harvested = Vec::new();
    net.get_weights_derivatives(&mut harvested);
    let (col0, col1) = derivs_to_cols(&harvested);
    let golden = common::load_bin_phase3("bwd_net_single_subsample_derivs.bin");
    assert_eq!(
        golden.dim(),
        (80, 2),
        "LSTM(4,2) single layer has 80 weights"
    );
    common::assert_oracle_eq(
        &col0,
        &golden_col(&golden, 0),
        "single-layer subsample derivs col0",
    );
    common::assert_bits_eq(
        &col1,
        &golden_col(&golden, 1),
        "single-layer subsample derivs col1",
    );
}

#[test]
fn single_layer_plain_no_subsample() {
    // T=7, sub [1] (legacy :262, the non-subsample single-layer arm): no decimation,
    // deltas_out stays the full T=7 rows -- contrast with the subsample variant's
    // floor(7/2)*2=6.
    let mut net = make_single_layer_net(1);
    let input = make_input(7, 2);
    let seed = make_deltas(7, 2);

    let mut output = Array2::zeros((7, 2));
    net.reset_weights_derivatives();
    net.feed_forward(&input, &mut output);
    let deltas_out = net.feed_backward(&input, &output, &seed);

    assert_eq!(
        deltas_out.dim(),
        (7, 2),
        "single-layer plain deltas_out keeps all 7 rows"
    );
    let golden_deltasout = common::load_bin_phase3("bwd_net_single_plain_deltasout.bin");
    common::assert_oracle_eq(
        &deltas_out,
        &golden_deltasout,
        "single-layer plain deltas_out",
    );

    let mut harvested = Vec::new();
    net.get_weights_derivatives(&mut harvested);
    let (col0, col1) = derivs_to_cols(&harvested);
    let golden = common::load_bin_phase3("bwd_net_single_plain_derivs.bin");
    assert_eq!(
        golden.dim(),
        (64, 2),
        "LSTM(2,2) single layer has 64 weights"
    );
    common::assert_oracle_eq(
        &col0,
        &golden_col(&golden, 0),
        "single-layer plain derivs col0",
    );
    common::assert_bits_eq(
        &col1,
        &golden_col(&golden, 1),
        "single-layer plain derivs col1",
    );
}

#[test]
fn single_layer_reads_seed_deltas_not_running_deltas_out() {
    // Structural, non-vacuity for the mutation this fix wave verifies against: the
    // single-layer branch (:255-263) takes `deltas` (the SEED) as its backward input
    // on every call -- there is no "running deltas_out" for it to read (that concept
    // only exists in the multi-layer loop's accumulation across layers). Confirm two
    // DIFFERENT seeds produce DIFFERENT derivs (the seed is genuinely read), which a
    // mutation feeding a stale/zeroed deltas_out in its place would fail to reproduce.
    let seed_a = make_deltas(7, 2);
    let seed_b = Array2::from_shape_fn((7, 2), |(t, j)| {
        ((t * 13 + j * 17 + 3) % 61) as f64 / 61.0 - 0.5
    });
    let mut net_a = make_single_layer_net(1);
    let mut net_b = make_single_layer_net(1);
    let input = make_input(7, 2);
    let mut out_a = Array2::zeros((7, 2));
    let mut out_b = Array2::zeros((7, 2));
    net_a.reset_weights_derivatives();
    net_a.feed_forward(&input, &mut out_a);
    let da = net_a.feed_backward(&input, &out_a, &seed_a);
    net_b.reset_weights_derivatives();
    net_b.feed_forward(&input, &mut out_b);
    let db = net_b.feed_backward(&input, &out_b, &seed_b);
    assert!(
        da.iter()
            .zip(db.iter())
            .any(|(x, y)| x.to_bits() != y.to_bits()),
        "different seed deltas must produce different deltas_out (proves the seed is read)"
    );
}

// === inv_sub_sample unit (the InvSubSample column un-stacking) ================

#[test]
fn inv_sub_sample_inverts_sub_sample_on_floored_input() {
    // InvSubSample (NeuralNetwork.hpp:136-145) un-stacks the column blocks: T x (C*R) ->
    // (T*R) x C, output row jj*R+kk = input block [kk*C,(kk+1)*C). Round-tripping through
    // sub_sample recovers the FLOORED input (floor(T/R)*R rows), NOT the original T -- the
    // dropped tail is gone. T=11, R=2, C=2: sub_sample -> 5x4, inv_sub_sample -> 10x2 ==
    // the first 10 rows of the original 11x2.
    let input = Array2::from_shape_fn((11, 2), |(t, j)| (t * 10 + j) as f64);
    let sub = sub_sample(2, &input); // 5 x 4
    let back = inv_sub_sample(2, &sub); // 10 x 2
    assert_eq!(back.dim(), (10, 2), "inv_sub_sample 5x4 -> 10x2");
    let expected = input.slice(ndarray::s![..10, ..]).to_owned();
    common::assert_bits_eq(
        &back,
        &expected,
        "inv_sub_sample(sub_sample) == floored input",
    );
}
