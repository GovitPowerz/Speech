//! Phase 2 Task 6: bit-exact golden tests for `nn::network` (the `NeuralNetwork`
//! container, ported from `NeuralNetwork.hpp:25-249`).
//!
//! The container goldens (`net_lstm_{fwd,rev}.bin`, `net_dense.bin`, `net_double.bin`)
//! are dumped by the harness's `netForwardLoop`/`netForwardReverseLoop`/
//! `netForwardDoubleLoop` reimpl (wired to `lstmForwardLoop`/`denseForwardLoop`),
//! which the harness's NN_TOL probe checks against the REAL `NeuralNetwork<LSTMLayer>`
//! / `NeuralNetwork<NeuronLayer>` on the same inputs (sites `net_lstm_forward_{fwd,
//! rev}`, `net_dense_forward`, `net_double`). The activation chains (asinh/exp) earn
//! `assert_oracle_eq`'s hybrid bound off the oracle env; the chained-weights round
//! trip (`net_w_{in,out}.bin`) is pure copy logic -> `assert_bits_eq`.

mod common;

use ndarray::Array2;
use speech::nn::layers::{LstmLayer, NeuronLayer};
use speech::nn::network::{Network, repeat_rows, sub_sample};

/// The harness synthetic flat-vector formula: `w[k] = ((k*11+3) % 97)/97.0 - 0.5`.
fn synthetic_flat(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5)
        .collect()
}

/// The harness deterministic input: `x[t,j] = ((t*37 + j*53 + 7) % 101)/101.0 - 0.5`.
fn synthetic_input(t_len: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((t_len, cols), |(t, j)| {
        ((t * 37 + j * 53 + 7) % 101) as f64 / 101.0 - 0.5
    })
}

// === sub_sample / repeat_rows hand cases =====================================

#[test]
fn sub_sample_t11_r2_drops_tail_frame() {
    // T=11, R=2, C=2: floor(11/2)=5 rows, C*R=4 cols. Source row jj*R+kk lands at
    // cols [kk*C,(kk+1)*C). Frame 10 (the 11th row) is DROPPED (11 mod 2 = 1).
    // Input[t,j] = t*10 + j so the expected 5x4 matrix is legible.
    let input = Array2::from_shape_fn((11, 2), |(t, j)| (t * 10 + j) as f64);
    let out = sub_sample(2, &input);
    assert_eq!(out.dim(), (5, 4), "floor(11/2)=5 rows, 2*2=4 cols");

    let expected = Array2::from_shape_vec(
        (5, 4),
        vec![
            0.0, 1.0, 10.0, 11.0, // rows 0,1
            20.0, 21.0, 30.0, 31.0, // rows 2,3
            40.0, 41.0, 50.0, 51.0, // rows 4,5
            60.0, 61.0, 70.0, 71.0, // rows 6,7
            80.0, 81.0, 90.0, 91.0, // rows 8,9   (row 10 = [100,101] dropped)
        ],
    )
    .unwrap();
    common::assert_bits_eq(&out, &expected, "sub_sample T=11 R=2 C=2");
}

#[test]
fn sub_sample_ratio1_is_plain_copy() {
    // R=1: floor(T/1)=T rows, C*1=C cols, single frame block -> identity copy.
    let input = synthetic_input(7, 3);
    let out = sub_sample(1, &input);
    common::assert_bits_eq(&out, &input, "sub_sample R=1 identity");
}

#[test]
fn repeat_rows_hand_case() {
    // 2x2 input, n=3 -> 6x2: each row duplicated 3x consecutively.
    let input = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = repeat_rows(&input, 3);
    let expected = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 0 x3
            3.0, 4.0, 3.0, 4.0, 3.0, 4.0, // row 1 x3
        ],
    )
    .unwrap();
    common::assert_bits_eq(&out, &expected, "repeat_rows n=3");
}

#[test]
fn sub_sample_cascaded_floor_identity_t13_2_3() {
    // PLAN NOTE (floor-identity): floor(floor(T/a)/b) == floor(T/(a*b)) is an
    // arithmetic identity for all T,a,b -- nested floor division never diverges from
    // the single product division, so there is NO distinguishing test case. The
    // legacy applies the sub-samples SEQUENTIALLY per layer (SubSample(a) then
    // SubSample(b)); we port that sequential form and ASSERT it agrees with the
    // product form here to document the identity rather than hunt a phantom
    // counterexample. T=13, subs [2,3]: 13->floor(13/2)=6->floor(6/3)=2, and
    // floor(13/(2*3))=floor(13/6)=2. Same.
    let input = synthetic_input(13, 2);
    let stage1 = sub_sample(2, &input); // 6 rows
    assert_eq!(stage1.nrows(), 6, "floor(13/2)=6");
    let stage2 = sub_sample(3, &stage1); // 2 rows
    assert_eq!(stage2.nrows(), 2, "floor(6/3)=2 (sequential)");
    // Product form on the same input: floor(13/6)=2 rows, same count.
    let product = sub_sample(6, &input);
    assert_eq!(product.nrows(), 2, "floor(13/6)=2 (product) == sequential");
}

// === Empty-input no-op ======================================================

#[test]
fn feed_forward_empty_input_leaves_output_untouched() {
    // NeuralNetwork.hpp:159 -- Input.rows()==0 is a silent no-op. Pre-seed the output
    // with sentinels and require it untouched. A 2-layer dense net [4,3,2] sub [1,1].
    let mut net = Network::<NeuronLayer>::new(vec![4, 3, 2], vec![1, 1], |_id, i, o| {
        let mut layer = NeuronLayer::new(i, o);
        layer.set_weights(&synthetic_flat(o * (i + 1)));
        layer
    });
    let empty = Array2::<f64>::zeros((0, 4));
    let mut output = Array2::from_elem((3, 2), f64::from_bits(0xDEAD_BEEF_DEAD_BEEF));
    let sentinel = output.clone();
    net.feed_forward(&empty, &mut output);
    common::assert_bits_eq(&output, &sentinel, "empty input no-op");
}

// === Container goldens vs harness dumps ======================================

/// Build the 2-layer LSTM net `[3,4,2]` sub `[2,1]` (all peepholes on), set the
/// synthetic flat vector, return the net.
fn make_lstm_net() -> Network<LstmLayer> {
    let flat = synthetic_flat(net_lstm_nb());
    let mut net = Network::<LstmLayer>::new(vec![3, 4, 2], vec![2, 1], |_id, i, o| {
        LstmLayer::new(i, o, true, true, true)
    });
    let tail = net.set_weights(&flat);
    assert_eq!(tail.len(), 0, "LSTM net flat vector fully consumed");
    net
}

/// nb_of_weights for the LSTM net: L0 = LstmLayer(6,4), L1 = LstmLayer(4,2).
fn net_lstm_nb() -> usize {
    LstmLayer::new(6, 4, true, true, true).nb_of_weights()
        + LstmLayer::new(4, 2, true, true, true).nb_of_weights()
}

#[test]
fn net_lstm_forward_matches_oracle() {
    let mut net = make_lstm_net();
    let input = synthetic_input(11, 3);
    let mut output = Array2::<f64>::zeros((5, 2)); // floor(11/2)=5 after SubSample(2)
    net.feed_forward(&input, &mut output);
    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("net_lstm_fwd.bin"),
        "net_lstm_fwd",
    );
}

#[test]
fn net_lstm_reverse_matches_oracle() {
    let mut net = make_lstm_net();
    let input = synthetic_input(11, 3);
    let mut output = Array2::<f64>::zeros((5, 2));
    net.feed_forward_reverse(&input, &mut output);
    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("net_lstm_rev.bin"),
        "net_lstm_rev",
    );
}

#[test]
fn net_dense_forward_matches_oracle() {
    // Dense net [4,3,2] sub [1,2] on T=11: SubSample(2) between layers -> 5 rows;
    // final layer O=2 lastLayer=true (softmax).
    let mut net = Network::<NeuronLayer>::new(vec![4, 3, 2], vec![1, 2], |_id, i, o| {
        NeuronLayer::new(i, o)
    });
    let nb0 = NeuronLayer::new(4, 3).nb_of_weights();
    let nb1 = NeuronLayer::new(6, 2).nb_of_weights();
    let flat = synthetic_flat(nb0 + nb1);
    let tail = net.set_weights(&flat);
    assert_eq!(tail.len(), 0, "dense net flat vector fully consumed");

    let input = synthetic_input(11, 4);
    let mut output = Array2::<f64>::zeros((5, 2)); // floor(11/2)=5 after SubSample(2)
    net.feed_forward(&input, &mut output);
    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("net_dense.bin"),
        "net_dense",
    );
}

// === Double-input MLP entry ==================================================

/// Build the double-input dense net `[10,4,2]` sub `[1,1]` with the synthetic flat
/// vector.
fn make_double_net() -> Network<NeuronLayer> {
    let nb0 = NeuronLayer::new(10, 4).nb_of_weights();
    let nb1 = NeuronLayer::new(4, 2).nb_of_weights();
    let mut net = Network::<NeuronLayer>::new(vec![10, 4, 2], vec![1, 1], |_id, i, o| {
        NeuronLayer::new(i, o)
    });
    let flat = synthetic_flat(nb0 + nb1);
    let tail = net.set_weights(&flat);
    assert_eq!(tail.len(), 0, "double net flat vector fully consumed");
    net
}

/// The harness asymmetric double-input halves: first = the Task 4/5 formula, second =
/// a DIFFERENT closed form so a first/second swap changes the hcat (and the output).
fn double_halves(t_len: usize) -> (Array2<f64>, Array2<f64>) {
    let first = Array2::from_shape_fn((t_len, 5), |(t, j)| {
        ((t * 37 + j * 53 + 7) % 101) as f64 / 101.0 - 0.5
    });
    let second = Array2::from_shape_fn((t_len, 5), |(t, j)| {
        ((t * 41 + j * 59 + 13) % 103) as f64 / 103.0 - 0.5
    });
    (first, second)
}

#[test]
fn net_double_matches_oracle() {
    let mut net = make_double_net();
    let (first, second) = double_halves(11);
    let mut output = Array2::<f64>::zeros((11, 2));
    net.feed_forward_double(&first, &second, &mut output);
    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("net_double.bin"),
        "net_double",
    );
}

#[test]
fn net_double_hcat_order_is_load_bearing() {
    // The forward half goes LEFT (NeuralNetwork.hpp:245 `Input << first, second`).
    // Swapping the two halves must change the output -- the halves are asymmetric by
    // construction, so a swap that happened to be a no-op would be a real bug.
    let mut net = make_double_net();
    let (first, second) = double_halves(11);
    let mut out_ab = Array2::<f64>::zeros((11, 2));
    net.feed_forward_double(&first, &second, &mut out_ab);

    let mut net2 = make_double_net();
    let mut out_ba = Array2::<f64>::zeros((11, 2));
    net2.feed_forward_double(&second, &first, &mut out_ba);

    let mut differs = false;
    for (a, b) in out_ab.iter().zip(out_ba.iter()) {
        if a.to_bits() != b.to_bits() {
            differs = true;
            break;
        }
    }
    assert!(
        differs,
        "swapping the double halves must change the output (hcat order)"
    );
}

// === Chained set_weights round-trip ==========================================

#[test]
fn net_chained_weights_roundtrip_matches_dump() {
    // The network's setWeights threads the flat vector layer-0-first (each layer
    // consumes nb_of_weights()); getWeights concatenates back in the same order. Pure
    // copy logic, no libm -> bit-exact against the harness dump on every platform.
    let want_in = common::load_bin_phase2("net_w_in.bin");
    let want_out = common::load_bin_phase2("net_w_out.bin");
    let nb = net_lstm_nb();
    assert_eq!(want_in.shape(), &[nb, 1]);
    assert_eq!(want_out.shape(), &[nb, 1]);

    let flat = synthetic_flat(nb);
    let flat_col = Array2::from_shape_vec((nb, 1), flat.clone()).unwrap();
    common::assert_bits_eq(&flat_col, &want_in, "synthetic flat vs net_w_in dump");

    let mut net = Network::<LstmLayer>::new(vec![3, 4, 2], vec![2, 1], |_id, i, o| {
        LstmLayer::new(i, o, true, true, true)
    });
    let tail = net.set_weights(&flat);
    assert_eq!(tail.len(), 0, "net flat vector fully consumed");

    let mut got = Vec::new();
    net.get_weights(&mut got);
    assert_eq!(got.len(), nb);
    let got_col = Array2::from_shape_vec((nb, 1), got).unwrap();
    common::assert_bits_eq(&got_col, &want_out, "net getWeights() vs net_w_out dump");
}

// === Container metadata accessors ============================================

#[test]
fn net_metadata_accessors() {
    let net = Network::<LstmLayer>::new(vec![3, 4, 2], vec![2, 1], |_id, i, o| {
        LstmLayer::new(i, o, true, true, true)
    });
    assert_eq!(net.input_size(), 3, "getInputSize = neuron_nb[0]");
    assert_eq!(net.output_size(), 2, "getOutputSize = neuron_nb.last()");
    assert_eq!(net.sub_samplings(), &[2, 1], "getSubSampling raw factors");
    // Cumulative products: [2, 2*1] = [2, 2]; getSubSamplingRatio = last = 2.
    assert_eq!(
        net.sub_sampling_ratio(),
        2,
        "getSubSamplingRatio = last cumulative"
    );
    assert_eq!(
        net.nb_of_weights(),
        net_lstm_nb(),
        "getNbOfWeights = layer sum"
    );
}

#[test]
fn neuron_layer_reverse_panics() {
    // NeuronLayer has no reverse; driving a dense net reversed must panic (the legacy
    // never instantiates a reversed NeuronLayer). Single-layer net [4,2] sub [1,1].
    let mut net = Network::<NeuronLayer>::new(vec![4, 2], vec![1], |_id, i, o| {
        let mut layer = NeuronLayer::new(i, o);
        layer.set_weights(&synthetic_flat(o * (i + 1)));
        layer
    });
    let input = synthetic_input(4, 4);
    let mut output = Array2::<f64>::zeros((4, 2));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        net.feed_forward_reverse(&input, &mut output);
    }));
    assert!(result.is_err(), "NeuronLayer reverse must panic");
}
