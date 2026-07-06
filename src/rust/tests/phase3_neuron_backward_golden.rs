//! Phase 3 Task 3: `NeuronLayer::feed_backward` golden tests.
//!
//! Ported from `NeuronLayer.cpp:151-207`. Goldens dumped by
//! `tools/oracle_harness/main.cpp`'s `denseBackwardLoop` (Task 1) via
//! `scripts/extract_phase3_fixtures.py` (`bwd_dense_{hidden,last,logistic}_derivs.bin`
//! + Task 3's `bwd_dense_{hidden,last,logistic}_deltasout.bin`).
//!
//! Comparator discipline: the `last`/`logistic` variants' `deltas_out` (`deltas*W^T`,
//! NO activation Jacobian) and all variants' bias/weight derivs are pure arithmetic
//! -> `assert_bits_eq`. The `hidden` variant's `deltas_out` folds `Maxmin2::deriv`
//! (asinh/sinh) on the layer input -> `assert_oracle_eq` (canary-gated). Col1 (the
//! frame count) is always a structural integer check.

mod common;

use ndarray::Array2;
use speech::nn::layers::NeuronLayer;

/// Deterministic synthetic weights: mirrors the harness's `synthFlat` lambda
/// (`main.cpp`, Phase 3 Task 1 backward-probe block):
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

fn build_layer(input_size: usize, output_size: usize) -> NeuronLayer {
    let mut layer = NeuronLayer::new(input_size, output_size);
    let flat = synth_flat(layer.nb_of_weights());
    layer.set_weights(&flat);
    layer
}

#[test]
fn hidden_backward() {
    // Grid site "hidden": I=4, O=3, T=6, last_layer=false.
    let (i, o, t) = (4, 3, 6);
    let mut layer = build_layer(i, o);
    let input = make_input(t, i);
    let deltas = make_deltas(t, o);

    let deltas_out = layer.feed_backward(&input, &deltas, 1, false);

    let want_deltas_out = common::load_bin_phase3("bwd_dense_hidden_deltasout.bin");
    // Transcendental (Maxmin2::deriv folds sinh) -> canary-gated.
    common::assert_oracle_eq(&deltas_out, &want_deltas_out, "hidden deltas_out");

    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    let want_derivs = common::load_bin_phase3("bwd_dense_hidden_derivs.bin");
    assert_eq!(derivs.len(), want_derivs.nrows(), "hidden derivs row count");

    let col0: Vec<f64> = derivs.iter().map(|r| r[0]).collect();
    let col0_arr = Array2::from_shape_vec((col0.len(), 1), col0).unwrap();
    let want_col0 = want_derivs
        .column(0)
        .to_owned()
        .insert_axis(ndarray::Axis(1));
    common::assert_oracle_eq(&col0_arr, &want_col0, "hidden derivs col0");

    for (idx, row) in derivs.iter().enumerate() {
        let want_count = want_derivs[[idx, 1]];
        assert_eq!(
            row[1], want_count,
            "hidden derivs col1 (count) at row {idx}"
        );
    }
}

/// Double-count trap (spec S11.3, risk R3): the last-layer `deltas_out` must be
/// EXACTLY `deltas * W^T` with NO activation derivative folded in (the softmax+CE
/// fusion lives entirely in `CostLaw::compute_deltas`, upstream of this layer). A
/// wrongly-added Jacobian (e.g. `output*(1-output)`) would visibly diverge from
/// the hand-computed `deltas.dot(W^T)` below.
#[test]
fn last_backward_no_activation_deriv() {
    // Grid site "last": I=4, O=3, T=6, last_layer=true.
    let (i, o, t) = (4, 3, 6);
    let mut layer = build_layer(i, o);
    let input = make_input(t, i);
    let deltas = make_deltas(t, o);

    let deltas_out = layer.feed_backward(&input, &deltas, 1, true);

    let want_deltas_out = common::load_bin_phase3("bwd_dense_last_deltasout.bin");
    // Pure arithmetic (no activation deriv, no libm) -> strict bits.
    common::assert_bits_eq(&deltas_out, &want_deltas_out, "last deltas_out (golden)");

    // HAND assertion: reconstruct W from the flat vector (column-major, matching
    // NeuronLayer::setWeights) and recompute deltas*W^T directly -- no Maxmin2
    // deriv anywhere in this computation. A spuriously-added softmax/logistic
    // Jacobian on the layer under test would diverge from this by construction.
    let flat = synth_flat(layer.nb_of_weights());
    let mut w = Array2::<f64>::zeros((i, o));
    for c in 0..o {
        for r in 0..i {
            w[[r, c]] = flat[c * i + r];
        }
    }
    let mut hand_deltas_out = Array2::<f64>::zeros((t, i));
    for tt in 0..t {
        for ii in 0..i {
            let mut acc = 0.0;
            for c in 0..o {
                acc += deltas[[tt, c]] * w[[ii, c]];
            }
            hand_deltas_out[[tt, ii]] = acc;
        }
    }
    common::assert_bits_eq(
        &deltas_out,
        &hand_deltas_out,
        "last deltas_out (hand deltas*W^T)",
    );
}

/// Same trap, O=1 (Logistic output head) variant -- the fusion applies here too
/// (the scalar cost path folds `output*(1-output)` in `compute_unitary_delta`,
/// not in the layer).
#[test]
fn last_logistic_backward_no_activation_deriv() {
    let (i, o, t) = (4, 1, 6);
    let mut layer = build_layer(i, o);
    let input = make_input(t, i);
    let deltas = make_deltas(t, o);

    let deltas_out = layer.feed_backward(&input, &deltas, 1, true);
    let want_deltas_out = common::load_bin_phase3("bwd_dense_logistic_deltasout.bin");
    common::assert_bits_eq(
        &deltas_out,
        &want_deltas_out,
        "logistic deltas_out (golden)",
    );
}

#[test]
fn count_is_input_rows() {
    // Sub-sampling ratio > 1 exercises the "count != deltas.nrows() in general"
    // property structurally: InputSeq.rows() is what's counted (:199), NOT
    // deltas.rows() -- here they're equal (invSubSamplingRatio only scales the
    // deriv magnitude, not which row-count is counted), so the discriminating
    // check is that the count equals input.nrows() specifically, and accumulates
    // across repeated calls.
    let (i, o, t) = (4, 3, 5);
    let mut layer = build_layer(i, o);
    let input = make_input(t, i);
    let deltas = make_deltas(t, o);

    layer.feed_backward(&input, &deltas, 1, false);
    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    for row in &derivs {
        assert_eq!(row[1], t as f64, "count after 1 call == input.nrows()");
    }

    layer.feed_backward(&input, &deltas, 1, false);
    let mut derivs2 = Vec::new();
    layer.get_weights_derivatives(&mut derivs2);
    for row in &derivs2 {
        assert_eq!(row[1], 2.0 * t as f64, "count after 2 calls accumulates");
    }
}

/// `invSubSamplingRatio > 1` scales BOTH deriv blocks by the ratio (:192-195) --
/// a property test (no committed golden exercises ratio>1; the harness dumps all
/// use ratio=1). Two fresh layers, identical weights/input/deltas, one fed back
/// with ratio=1 and the other with ratio=3: the ratio=3 derivs must be EXACTLY
/// 3x the ratio=1 derivs (bit-exact scalar multiply, per the legacy `*=`).
/// deltas_out is UNAFFECTED by the ratio (only the two deriv blocks are scaled).
#[test]
fn inv_sub_sampling_ratio_scales_derivs_not_deltas_out() {
    let (i, o, t) = (4, 3, 6);
    let mut layer1 = build_layer(i, o);
    let mut layer3 = build_layer(i, o);
    let input = make_input(t, i);
    let deltas = make_deltas(t, o);

    let deltas_out1 = layer1.feed_backward(&input, &deltas, 1, false);
    let deltas_out3 = layer3.feed_backward(&input, &deltas, 3, false);
    common::assert_bits_eq(
        &deltas_out1,
        &deltas_out3,
        "deltas_out unaffected by invSubSamplingRatio",
    );

    let mut derivs1 = Vec::new();
    layer1.get_weights_derivatives(&mut derivs1);
    let mut derivs3 = Vec::new();
    layer3.get_weights_derivatives(&mut derivs3);

    assert_eq!(derivs1.len(), derivs3.len());
    for (idx, (r1, r3)) in derivs1.iter().zip(derivs3.iter()).enumerate() {
        let want = r1[0] * 3.0;
        assert_eq!(r3[0], want, "deriv at row {idx} scaled by ratio=3");
        // Count is NOT scaled by the ratio -- it stays input.nrows() either way.
        assert_eq!(
            r3[1], r1[1],
            "count unaffected by invSubSamplingRatio at row {idx}"
        );
    }
}

/// Width tolerance (`:157-166`): `cols < I` -> zero-pad the input up to `I`
/// columns before the outer-product accumulation and the asinh-deriv fold.
/// Feeding an `I=4`-configured layer a `cols=2` input must produce EXACTLY the
/// same result as feeding it a hand-padded `cols=4` input with columns 2,3
/// explicitly zero (the legacy's `Eigen::MatrixXd::Zero` filler, `:161-163`).
/// A layer that instead used the narrow input as-is (skipping the pad) would
/// diverge both in deltas_out shape expectations and in the dead weight rows'
/// (cols 2,3) deriv contribution.
#[test]
fn width_tolerance_zero_pads_narrow_input() {
    let (i, o, t) = (4, 3, 6);
    let mut layer_narrow = build_layer(i, o);
    let mut layer_padded = build_layer(i, o);
    let deltas = make_deltas(t, o);

    let narrow_input = make_input(t, 2); // cols=2 < I=4
    let mut hand_padded = Array2::<f64>::zeros((t, i));
    hand_padded
        .slice_mut(ndarray::s![.., ..2])
        .assign(&narrow_input);

    let deltas_out_narrow = layer_narrow.feed_backward(&narrow_input, &deltas, 1, false);
    let deltas_out_padded = layer_padded.feed_backward(&hand_padded, &deltas, 1, false);
    common::assert_bits_eq(
        &deltas_out_narrow,
        &deltas_out_padded,
        "width-tolerant deltas_out matches hand-zero-padded input",
    );

    let mut derivs_narrow = Vec::new();
    layer_narrow.get_weights_derivatives(&mut derivs_narrow);
    let mut derivs_padded = Vec::new();
    layer_padded.get_weights_derivatives(&mut derivs_padded);
    assert_eq!(derivs_narrow.len(), derivs_padded.len());
    for (idx, (a, b)) in derivs_narrow.iter().zip(derivs_padded.iter()).enumerate() {
        assert_eq!(
            a[0], b[0],
            "deriv col0 at row {idx} matches hand-padded reference"
        );
        assert_eq!(
            a[1], b[1],
            "deriv col1 at row {idx} matches hand-padded reference"
        );
    }

    // Dead weight rows (input cols 2,3, which are always-zero after padding)
    // accumulate EXACTLY zero deriv (their column of `input` is all-zero, so
    // their outer-product contribution is zero regardless of deltas).
    for dead_col in [2usize, 3usize] {
        for out_col in 0..o {
            let flat_row = out_col * i + dead_col;
            assert_eq!(
                derivs_narrow[flat_row][0], 0.0,
                "dead weight row (input col {dead_col}, output col {out_col}) has zero deriv"
            );
        }
    }
}

#[test]
fn reset_zeroes_all() {
    let (i, o, t) = (4, 3, 6);
    let mut layer = build_layer(i, o);
    let input = make_input(t, i);
    let deltas = make_deltas(t, o);

    layer.feed_backward(&input, &deltas, 1, false);
    layer.reset_weights_derivatives();

    let mut derivs = Vec::new();
    layer.get_weights_derivatives(&mut derivs);
    for row in &derivs {
        assert_eq!(row[0], 0.0, "deriv zeroed after reset");
        assert_eq!(row[1], 0.0, "count zeroed after reset");
    }
}
