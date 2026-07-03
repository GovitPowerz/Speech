//! Phase 2 Task 3: bit-exact golden tests for `nn::layers::LstmLayer`
//! (ported from `LSTMLayer.h`/`LSTMLayer.cpp:6-27,162-249`).
//!
//! `lstm_w_roundtrip_in.bin` / `lstm_w_roundtrip_out.bin` are dumped by the REAL
//! compiled legacy `LSTMLayer` (linked into `tools/oracle_harness/main.cpp`):
//! a synthetic flat vector of length 100 (formula `w[k] = ((k*11+3) % 97)/97.0 -
//! 0.5`) is fed through `LstmLayer(I=3,O=2).setWeights` (consumes the first 72
//! elements, nb_of_weights() = 4*3*2+4*2*2+12*2+4*2 = 24+16+24+8 = 72) then the
//! leftover 28-element tail through a chained `LstmLayer(I=2,O=1).setWeights`
//! (nb_of_weights() = 4*2*1+4*1*1+12*1+4*1 = 8+4+12+4 = 28; 72+28 = 100). Both
//! layers' `getWeights()` outputs are dumped back-to-back into the "out" file: the
//! setWeights/getWeights round trip is pure copy logic (no libm), so the whole
//! 100-element vector is bit-exact everywhere -- `assert_bits_eq`, not the hybrid
//! oracle comparator.

mod common;

use ndarray::Array2;
use speech::nn::layers::{LstmLayer, NeuronLayer};

const NB0: usize = 72; // LstmLayer(I=3,O=2): 4*3*2 + 4*2*2 + 12*2 + 4*2 = 24+16+24+8
const NB1: usize = 28; // LstmLayer(I=2,O=1): 4*2*1 + 4*1*1 + 12*1 + 4*1 = 8+4+12+4

/// The harness's synthetic flat-vector formula: `w[k] = ((k*11+3) % 97)/97.0 - 0.5`.
fn synthetic_flat(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5)
        .collect()
}

#[test]
fn nb_of_weights_matches_formula() {
    // (I, O) -> 4*I*O + 4*O*O + 12*O + 4*O, computed by hand:
    // (3,2):   4*3*2=24,  4*2*2=16,  12*2=24, 4*2=8   -> 24+16+24+8   = 72
    // (23,24): 4*23*24=2208, 4*24*24=2304, 12*24=288, 4*24=96 -> 2208+2304+288+96 = 4896
    // (96,24): 4*96*24=9216, 4*24*24=2304, 12*24=288, 4*24=96 -> 9216+2304+288+96 = 11904
    let cases = [
        ((3usize, 2usize), 72usize),
        ((23, 24), 4896),
        ((96, 24), 11904),
    ];
    for ((i, o), expected) in cases {
        let layer = LstmLayer::new(i, o, true, true, true);
        assert_eq!(layer.nb_of_weights(), expected, "I={i} O={o}");
    }
}

#[test]
fn roundtrip_matches_oracle_dump() {
    let want_in = common::load_bin_phase2("lstm_w_roundtrip_in.bin");
    let want_out = common::load_bin_phase2("lstm_w_roundtrip_out.bin");
    assert_eq!(want_in.shape(), &[100, 1]);
    assert_eq!(want_out.shape(), &[100, 1]);

    let flat = synthetic_flat(NB0 + NB1);
    // Sanity: the dumped input vector IS this synthetic formula, bit-for-bit.
    let flat_col = Array2::from_shape_vec((flat.len(), 1), flat.clone()).unwrap();
    common::assert_bits_eq(&flat_col, &want_in, "synthetic flat vector vs dump input");

    let mut layer0 = LstmLayer::new(3, 2, true, true, true);
    let mut layer1 = LstmLayer::new(2, 1, true, true, true);

    let tail0 = layer0.set_weights(&flat);
    assert_eq!(tail0.len(), NB1);
    let tail1 = layer1.set_weights(tail0);
    assert_eq!(
        tail1.len(),
        0,
        "leftover tail after both layers must be empty"
    );

    let mut got0 = Vec::new();
    layer0.get_weights(&mut got0);
    let mut got1 = Vec::new();
    layer1.get_weights(&mut got1);
    assert_eq!(got0.len(), NB0);
    assert_eq!(got1.len(), NB1);

    let mut got_all = got0;
    got_all.extend_from_slice(&got1);
    let got_col = Array2::from_shape_vec((got_all.len(), 1), got_all).unwrap();
    common::assert_bits_eq(&got_col, &want_out, "chained getWeights() vs dump output");
}

#[test]
fn chaining_splits_flat_vector_by_nb_of_weights() {
    let flat = synthetic_flat(NB0 + NB1 + 5); // extra tail past both layers
    let mut layer0 = LstmLayer::new(3, 2, true, true, true);
    let mut layer1 = LstmLayer::new(2, 1, true, true, true);

    let tail0 = layer0.set_weights(&flat);
    assert_eq!(tail0.len(), flat.len() - NB0);
    let tail1 = layer1.set_weights(tail0);
    assert_eq!(tail1.len(), flat.len() - NB0 - NB1);
    assert_eq!(tail1, &flat[NB0 + NB1..]);
}

// === Task 4: LSTM forward/reverse goldens ===================================
//
// The harness dumps, for each (I,O,T,flags,direction) case, `lstm_fwd_<case>.bin`
// (T x O outputs) and `lstm_gates_<case>.bin` (T x 4O post-activation gates cache),
// computed by the ascending-loop reimpl of `LSTMLayer::feedForward` -- which the
// harness's NN_TOL probe verifies matches the REAL compiled `LSTMLayer::feedForward`
// bit-for-bit (max_ulp=0) at these shapes. So each golden is a bit-exact target for
// `LstmLayer::feed_forward`/`feed_forward_reverse` on the oracle env; elsewhere the
// asinh/exp chains earn `assert_oracle_eq`'s hybrid bound.

/// The harness weight formula: `w[k] = ((k*11+3) % 97)/97.0 - 0.5`, length nb(I,O).
fn synthetic_weights(i: usize, o: usize) -> Vec<f64> {
    let nb = LstmLayer::new(i, o, true, true, true).nb_of_weights();
    synthetic_flat(nb)
}

/// The harness deterministic input: `x[t,j] = ((t*37 + j*53 + 7) % 101)/101.0 - 0.5`.
fn synthetic_input(t_len: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((t_len, cols), |(t, j)| {
        ((t * 37 + j * 53 + 7) % 101) as f64 / 101.0 - 0.5
    })
}

/// `(cells, gates, gates_rec)` peephole flags for each case-label in the grid.
fn flags_of(label: &str) -> (bool, bool, bool) {
    match label {
        "allon" => (true, true, true),
        "cells" => (true, false, false),
        "gates" => (false, true, false),
        "gatesrec" => (false, false, true),
        "alloff" => (false, false, false),
        other => panic!("unknown flag label {other:?}"),
    }
}

/// Replay one grid case through `LstmLayer` and check `output` + gates cache against
/// the harness dumps. `in_cols` overrides the input width (width-mismatch cases).
fn check_case(i: usize, o: usize, t_len: usize, label: &str, reverse: bool, in_cols: usize) {
    let (c, g, r) = flags_of(label);
    let mut layer = LstmLayer::new(i, o, c, g, r);
    let flat = synthetic_weights(i, o);
    let tail = layer.set_weights(&flat);
    assert_eq!(
        tail.len(),
        0,
        "case {i}x{o}_{label}: flat vector fully consumed"
    );

    let input = synthetic_input(t_len, in_cols);
    let mut output = Array2::<f64>::zeros((t_len, o));
    if reverse {
        layer.feed_forward_reverse(&input, &mut output, false);
    } else {
        layer.feed_forward(&input, &mut output, false);
    }

    let dir = if reverse { "rev" } else { "fwd" };
    let case = format!("{i}x{o}_T{t_len}_{label}_{dir}");
    let want_out = common::load_bin_phase2(&format!("lstm_fwd_{case}.bin"));
    let want_gates = common::load_bin_phase2(&format!("lstm_gates_{case}.bin"));
    common::assert_oracle_eq(&output, &want_out, &format!("lstm_fwd_{case}"));
    common::assert_oracle_eq(layer.gates(), &want_gates, &format!("lstm_gates_{case}"));
}

#[test]
fn lstm_forward_reverse_grid_matches_oracle() {
    let shapes = [(3usize, 2usize, 7usize), (5, 4, 12)];
    let labels = ["allon", "cells", "gates", "gatesrec", "alloff"];
    for (i, o, t_len) in shapes {
        for label in labels {
            check_case(i, o, t_len, label, false, i);
            check_case(i, o, t_len, label, true, i);
        }
    }
}

#[test]
fn lstm_width_mismatch_matches_oracle() {
    // Forward, all-on, I=3,O=2,T=7: input cols I+2 (leftCols(I)) and I-1 (topRows).
    // These reuse the base case labels with the harness's width suffixes.
    for (in_cols, suffix) in [(5usize, "wideP2"), (2usize, "narrowM1")] {
        let mut layer = LstmLayer::new(3, 2, true, true, true);
        layer.set_weights(&synthetic_weights(3, 2));
        let input = synthetic_input(7, in_cols);
        let mut output = Array2::<f64>::zeros((7, 2));
        layer.feed_forward(&input, &mut output, false);
        let case = format!("3x2_T7_allon_{suffix}_fwd");
        common::assert_oracle_eq(
            &output,
            &common::load_bin_phase2(&format!("lstm_fwd_{case}.bin")),
            &format!("lstm_fwd_{case}"),
        );
        common::assert_oracle_eq(
            layer.gates(),
            &common::load_bin_phase2(&format!("lstm_gates_{case}.bin")),
            &format!("lstm_gates_{case}"),
        );
    }
}

#[test]
fn lstm_reverse_equals_flip_forward_flip() {
    // feedForwardReverse(x) == flip(feedForward(flip(x))) by construction
    // (LSTMLayer.cpp:415-421). The property itself is a same-platform, same-libm
    // comparison (both sides recomputed locally in this process), so it stays
    // BIT-EXACT everywhere via assert_bits_eq -- it is NOT a fixture comparison and
    // does not touch the oracle canary gate. A separate assert below checks the
    // refolded output against the committed golden dump (Apple libm at generation
    // time; this DOES traverse asinh/exp), gated via assert_oracle_eq.
    let mut fwd = LstmLayer::new(3, 2, true, true, true);
    fwd.set_weights(&synthetic_weights(3, 2));
    let input = synthetic_input(7, 3);
    // flip(x) rows.
    let flipped = input.slice(ndarray::s![..;-1, ..]).to_owned();
    let mut fwd_out = Array2::<f64>::zeros((7, 2));
    fwd.feed_forward(&flipped, &mut fwd_out, false);
    let refolded = fwd_out.slice(ndarray::s![..;-1, ..]).to_owned();

    let mut rev = LstmLayer::new(3, 2, true, true, true);
    rev.set_weights(&synthetic_weights(3, 2));
    let mut rev_out = Array2::<f64>::zeros((7, 2));
    rev.feed_forward_reverse(&input, &mut rev_out, false);
    common::assert_bits_eq(
        &refolded,
        &rev_out,
        "flip(fwd(flip(x))) == feed_forward_reverse(x), same-platform property",
    );

    let want_rev = common::load_bin_phase2("lstm_fwd_3x2_T7_allon_rev.bin");
    common::assert_oracle_eq(
        &refolded,
        &want_rev,
        "flip(fwd(flip(x))) vs reverse golden dump",
    );
}

#[test]
fn lstm_t0_has_no_forget_contribution() {
    // At t=0 the cell state is i_0 .* g_0 with NO forget term (LSTMLayer.cpp:336):
    // a nonzero forget bias/weight must NOT shift c_0. Flags = cells-only (gates-peep
    // OFF so f_0 does NOT leak into o_0 via P[10], isolating the cell path). Two
    // layers identical EXCEPT the forget-gate input weight + bias (block index 1 of
    // 4): the t=0 output must be bit-identical, while a t>=1 output (which reads
    // c_{t-1} through the forget gate at :387) must differ -- the forget path is dead
    // only at t=0. With gates-peep on, f_0 leaks into o_0 (:341) and t=0 WOULD shift,
    // so the isolation matters.
    let i = 1usize;
    let o = 1usize;
    let base = synthetic_weights(i, o); // nb = 24
    // Layout (I=1,O=1) col-major: input_w flat[0..4]=[Wi,Wf,Wo,Wg];
    // feedback flat[4..8]; peep flat[8..20]; biases flat[20..24]=[bi,bf,bo,bg].
    // Perturb the forget input weight (idx 1) and forget bias (idx 21).
    let mut perturbed = base.clone();
    perturbed[1] += 3.0; // Wf
    perturbed[21] += 5.0; // bf

    let input = synthetic_input(4, i);

    let run = |w: &[f64]| {
        let mut layer = LstmLayer::new(i, o, true, false, false);
        layer.set_weights(w);
        let mut out = Array2::<f64>::zeros((4, o));
        layer.feed_forward(&input, &mut out, false);
        out
    };
    let out_base = run(&base);
    let out_pert = run(&perturbed);

    assert_eq!(
        out_base[[0, 0]].to_bits(),
        out_pert[[0, 0]].to_bits(),
        "t=0 output must be independent of the forget weights (no forget term at t=0)"
    );
    assert_ne!(
        out_base[[1, 0]].to_bits(),
        out_pert[[1, 0]].to_bits(),
        "t=1 output must depend on the forget gate (c_0 flows through f_1)"
    );
}

#[test]
fn lstm_hand_case_o1_t2() {
    // I=1, O=1, T=2, all peephole flags ON. Distinct small rational weights, coded
    // as the longhand f64 op sequence in spec S4.3 order; the golden grid arbitrates
    // FP order globally, this pins the scalar structure. Layout (I=1,O=1) col-major:
    //   input_w  [Wi,Wf,Wo,Wg]; feedback [Ui,Uf,Uo,Ug]; peep P[0..12]; bias [bi,bf,bo,bg].
    let wi = 1.0 / 8.0;
    let wf = -1.0 / 4.0;
    let wo = 3.0 / 8.0;
    let wg = -1.0 / 2.0;
    let ui = 1.0 / 16.0;
    let uf = -3.0 / 16.0;
    let uo = 5.0 / 16.0;
    let ug = -7.0 / 16.0;
    let p: [f64; 12] = [
        1.0 / 32.0,   // P0  c_{t-1} -> i
        -2.0 / 32.0,  // P1  c_{t-1} -> f
        3.0 / 32.0,   // P2  c_t     -> o
        -4.0 / 32.0,  // P3  i_{t-1} -> i (gates-rec)
        5.0 / 32.0,   // P4  f_{t-1} -> i
        -6.0 / 32.0,  // P5  o_{t-1} -> i
        7.0 / 32.0,   // P6  i_{t-1} -> f
        -8.0 / 32.0,  // P7  f_{t-1} -> f (gates-rec)
        9.0 / 32.0,   // P8  o_{t-1} -> f
        -10.0 / 32.0, // P9  i_t     -> o
        11.0 / 32.0,  // P10 f_t     -> o
        -12.0 / 32.0, // P11 o_{t-1} -> o (gates-rec)
    ];
    let bi = 1.0 / 3.0;
    let bf = -1.0 / 6.0;
    let bo = 1.0 / 5.0;
    let bg = -1.0 / 7.0;

    let flat: Vec<f64> = {
        let mut v = Vec::with_capacity(24);
        v.extend_from_slice(&[wi, wf, wo, wg]); // input_w 1x4 col-major
        v.extend_from_slice(&[ui, uf, uo, ug]); // feedback 1x4 col-major
        v.extend_from_slice(&p); // peep 12x1 col-major
        v.extend_from_slice(&[bi, bf, bo, bg]); // bias 1x4
        v
    };

    let x0 = 0.3_f64;
    let x1 = -0.2_f64;
    let input = Array2::from_shape_vec((2, 1), vec![x0, x1]).unwrap();

    let gate = |z: f64| 1.0 / (1.0 + (-0.1 * z).exp()); // GatesFunction (0.1 pre-scale)
    let asinh = |z: f64| z.asinh(); // Maxmin2 / Identity

    // t=0 (LSTMLayer.cpp:325-348), all flags on.
    let i0 = gate(wi * x0 + bi);
    let f0 = gate(wf * x0 + bf);
    let g0 = asinh(wg * x0 + bg);
    let c0 = i0 * g0; // :336, no forget term
    // o extras IN ORDER: c_0.*P2; i_0.*P9; f_0.*P10 (three separate adds).
    let mut o0_pre = wo * x0 + bo;
    o0_pre += c0 * p[2];
    o0_pre += i0 * p[9];
    o0_pre += f0 * p[10];
    let o0 = gate(o0_pre);
    let cin0 = asinh(c0);
    let y0 = o0 * cin0;

    // t=1 (LSTMLayer.cpp:350-412), all flags on. Post-activation reads of t=0 gates.
    // :351 feedback into all four blocks; the four pre-activation gate accumulators:
    let mut i1 = wi * x1 + bi + y0 * ui;
    let mut f1 = wf * x1 + bf + y0 * uf;
    let mut o1 = wo * x1 + bo + y0 * uo;
    let g1_pre = wg * x1 + bg + y0 * ug;
    // :353-356 cells peep: i += c_0*P0; f += c_0*P1.
    i1 += c0 * p[0];
    f1 += c0 * p[1];
    // :357-360 gates-rec: i += i_0*P3; f += f_0*P7.
    i1 += i0 * p[3];
    f1 += f0 * p[7];
    // :362-363 gates peep, each ONE combined expression.
    i1 += f0 * p[4] + o0 * p[5];
    f1 += i0 * p[6] + o0 * p[8];
    // :370 activate i,f; :385 activate g.
    let i1 = gate(i1);
    let f1 = gate(f1);
    let g1 = asinh(g1_pre);
    // :387 c_1 = i_1*g_1 + c_0*f_1 (single combined expression).
    let c1 = i1 * g1 + c0 * f1;
    // :389-394 o extras IN ORDER: c_1*P2; i_1*P9; f_1*P10; o_0*P11.
    o1 += c1 * p[2];
    o1 += i1 * p[9];
    o1 += f1 * p[10];
    o1 += o0 * p[11];
    let o1 = gate(o1);
    let cin1 = asinh(c1);
    let y1 = o1 * cin1;

    let mut layer = LstmLayer::new(1, 1, true, true, true);
    let tail = layer.set_weights(&flat);
    assert_eq!(tail.len(), 0);
    let mut output = Array2::<f64>::zeros((2, 1));
    layer.feed_forward(&input, &mut output, false);

    // Same libm, same op order -> bit-identical.
    assert_eq!(output[[0, 0]].to_bits(), y0.to_bits(), "t=0 output");
    assert_eq!(output[[1, 0]].to_bits(), y1.to_bits(), "t=1 output");
    // Gates cache spot-check: post-activation [i1,f1,o1,g1] at row 1.
    let gates = layer.gates();
    assert_eq!(gates[[1, 0]].to_bits(), i1.to_bits(), "gates i_1");
    assert_eq!(gates[[1, 1]].to_bits(), f1.to_bits(), "gates f_1");
    assert_eq!(gates[[1, 2]].to_bits(), o1.to_bits(), "gates o_1");
    assert_eq!(gates[[1, 3]].to_bits(), g1.to_bits(), "gates g_1");
}

// === Task 4: Python-oracle cross-check ======================================
//
// `lstm_cases.json` (scripts/extract_phase2_oracle_cases.py) holds the Python
// `lstm_forward_oracle` forward outputs over a deterministic (I,O,T,flags) grid, with
// every f64 stored as its raw u64 bits (hex). The JSON is a COMMITTED FIXTURE (frozen
// at generation time on the oracle env, Apple libm), not recomputed by both sides at
// test time -- despite both being "local" implementations, the chain traverses
// asinh/exp, so on a different libm (CI glibc) this is exactly the cross-libm class:
// gated via `assert_oracle_eq` (bit-exact on the oracle env, hybrid ULP/abs bound
// elsewhere), not a raw `to_bits` cross-check.

fn bits_to_f64(s: &str) -> f64 {
    let hex = s.strip_prefix("0x").unwrap_or(s);
    f64::from_bits(u64::from_str_radix(hex, 16).unwrap())
}

fn matrix_from_bits(v: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|c| bits_to_f64(c.as_str().unwrap()))
                .collect()
        })
        .collect();
    let t_len = rows.len();
    let cols = if t_len == 0 { 0 } else { rows[0].len() };
    Array2::from_shape_fn((t_len, cols), |(t, j)| rows[t][j])
}

#[test]
fn lstm_python_oracle_cross_check() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase2/lstm_cases.json");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "lstm_cases.json has no cases");

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let i = case["I"].as_u64().unwrap() as usize;
        let o = case["O"].as_u64().unwrap() as usize;
        let t_len = case["T"].as_u64().unwrap() as usize;
        let fl: Vec<bool> = case["flags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_bool().unwrap())
            .collect();

        let flat: Vec<f64> = case["weights_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| bits_to_f64(s.as_str().unwrap()))
            .collect();
        let input = matrix_from_bits(&case["input_bits"]);
        let want_y = matrix_from_bits(&case["y_bits"]);
        let want_gates = matrix_from_bits(&case["gates_bits"]);
        let want_cells = matrix_from_bits(&case["cells_bits"]);

        let mut layer = LstmLayer::new(i, o, fl[0], fl[1], fl[2]);
        let tail = layer.set_weights(&flat);
        assert_eq!(tail.len(), 0, "{name}: flat vector fully consumed");
        let mut output = Array2::<f64>::zeros((t_len, o));
        layer.feed_forward(&input, &mut output, false);

        common::assert_oracle_eq(&output, &want_y, &format!("{name} y"));
        common::assert_oracle_eq(layer.gates(), &want_gates, &format!("{name} gates"));
        common::assert_oracle_eq(layer.cell_states(), &want_cells, &format!("{name} cells"));
    }
}

// === Task 5: NeuronLayer (dense) forward goldens ============================
//
// The harness dumps, for each case, `dense_<case>.bin` (T x O outputs), computed by
// the ascending-loop reimpl of `NeuronLayer::feedForward` -- which the harness's
// NN_TOL probe verifies matches the REAL compiled `NeuronLayer::feedForward`
// bit-for-bit (max_ulp=0) at these shapes. So each golden is a bit-exact target for
// `NeuronLayer::feed_forward` on the oracle env; elsewhere the asinh/exp chains earn
// `assert_oracle_eq`'s hybrid bound.

/// The harness weight formula (same as LSTM Task 3/4): `w[k] = ((k*11+3) % 97)/97.0
/// - 0.5`, length `nb_of_weights(I,O) = O*(I+1)`.
fn synthetic_dense_weights(i: usize, o: usize) -> Vec<f64> {
    let nb = NeuronLayer::new(i, o).nb_of_weights();
    synthetic_flat(nb)
}

fn check_dense_case(
    dump: &str,
    i: usize,
    o: usize,
    t_len: usize,
    last_layer: bool,
    in_cols: usize,
) {
    let mut layer = NeuronLayer::new(i, o);
    let flat = synthetic_dense_weights(i, o);
    let tail = layer.set_weights(&flat);
    assert_eq!(tail.len(), 0, "{dump}: flat vector fully consumed");

    let input = synthetic_input(t_len, in_cols);
    let mut output = Array2::<f64>::zeros((t_len, o));
    layer.feed_forward(&input, &mut output, last_layer);

    let want = common::load_bin_phase2(dump);
    common::assert_oracle_eq(&output, &want, dump);
}

#[test]
fn dense_hidden_matches_oracle() {
    // lastLayer=false, asinh (Maxmin2), I=4 O=3 T=6.
    check_dense_case("dense_hidden.bin", 4, 3, 6, false, 4);
}

#[test]
fn dense_softmax_matches_oracle() {
    // lastLayer=true, O=3 > 1 -> UNSTABILIZED softmax (no max-subtraction guard).
    check_dense_case("dense_softmax.bin", 4, 3, 6, true, 4);
}

#[test]
fn dense_logistic_matches_oracle() {
    // lastLayer=true, O=1 -> Logistic.
    check_dense_case("dense_logistic.bin", 4, 1, 6, true, 4);
}

#[test]
fn dense_width_mismatch_matches_oracle() {
    // lastLayer=false: cols=I+2 (leftCols(I)) and cols=I-1 (topRows(cols)).
    check_dense_case("dense_wide.bin", 4, 3, 6, false, 4 + 2);
    check_dense_case("dense_narrow.bin", 4, 3, 6, false, 4 - 1);
}

#[test]
fn dense_nb_of_weights_matches_formula() {
    // nb = O*(I+1), computed by hand:
    // (4,3): 3*(4+1) = 15
    // (4,1): 1*(4+1) = 5
    // (2,2): 2*(2+1) = 6
    let cases = [((4usize, 3usize), 15usize), ((4, 1), 5), ((2, 2), 6)];
    for ((i, o), expected) in cases {
        let layer = NeuronLayer::new(i, o);
        assert_eq!(layer.nb_of_weights(), expected, "I={i} O={o}");
    }
}

#[test]
fn dense_weight_roundtrip_bits() {
    // set_weights/get_weights must mirror bit-for-bit: pure copy logic, no libm.
    let i = 4usize;
    let o = 3usize;
    let flat = synthetic_dense_weights(i, o);
    let mut layer = NeuronLayer::new(i, o);
    let tail = layer.set_weights(&flat);
    assert_eq!(tail.len(), 0);

    let mut got = Vec::new();
    layer.get_weights(&mut got);
    assert_eq!(got.len(), flat.len());
    for (k, (a, b)) in flat.iter().zip(got.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "weight {k} roundtrip mismatch");
    }
}

#[test]
fn dense_weight_chaining_splits_flat_vector() {
    // Two layers sharing one flat vector: layer0 consumes nb0, layer1 consumes the
    // tail's first nb1, mirroring the LstmLayer chaining contract.
    let i0 = 3usize;
    let o0 = 2usize;
    let i1 = 2usize;
    let o1 = 1usize;
    let nb0 = NeuronLayer::new(i0, o0).nb_of_weights(); // 2*(3+1) = 8
    let nb1 = NeuronLayer::new(i1, o1).nb_of_weights(); // 1*(2+1) = 3
    let flat = synthetic_flat(nb0 + nb1 + 4); // extra tail past both layers

    let mut layer0 = NeuronLayer::new(i0, o0);
    let mut layer1 = NeuronLayer::new(i1, o1);
    let tail0 = layer0.set_weights(&flat);
    assert_eq!(tail0.len(), flat.len() - nb0);
    let tail1 = layer1.set_weights(tail0);
    assert_eq!(tail1.len(), flat.len() - nb0 - nb1);
    assert_eq!(tail1, &flat[nb0 + nb1..]);
}

#[test]
fn dense_hand_case_softmax_o2_t1() {
    // I=2, O=2, T=1: hand-derived UNSTABILIZED softmax, expression-coded (exp/sum/
    // quotient in the exact legacy order: :138 exp(a+b); :139 sequential row-sum;
    // :140-142 per-COLUMN cwiseQuotient). Layout (I=2,O=2) col-major:
    //   weights [w00,w10, w01,w11] (per-neuron fan-in blocks contiguous); bias [b0,b1].
    let w00 = 1.0 / 4.0; // input0 -> output0
    let w10 = -1.0 / 8.0; // input1 -> output0
    let w01 = 3.0 / 8.0; // input0 -> output1
    let w11 = -1.0 / 2.0; // input1 -> output1
    let b0 = 1.0 / 16.0;
    let b1 = -1.0 / 16.0;

    let x0 = 0.3_f64;
    let x1 = -0.2_f64;

    // :129-135 projection (cols == I, plain product); :136-137 rowwise + bias.
    let a0 = x0 * w00 + x1 * w10 + b0;
    let a1 = x0 * w01 + x1 * w11 + b1;
    // :138 exp(a+b) per column, NO max-subtraction (unstabilized, load-bearing quirk).
    let e0 = a0.exp();
    let e1 = a1.exp();
    // :139 sequential row-sum (T=1, single row: e0 then e1, in ascending column order).
    let sum = e0 + e1;
    // :140-142 per-COLUMN cwiseQuotient by the row sum.
    let y0 = e0 / sum;
    let y1 = e1 / sum;

    let flat = vec![w00, w10, w01, w11, b0, b1];
    let mut layer = NeuronLayer::new(2, 2);
    let tail = layer.set_weights(&flat);
    assert_eq!(tail.len(), 0);

    let input = Array2::from_shape_vec((1, 2), vec![x0, x1]).unwrap();
    let mut output = Array2::<f64>::zeros((1, 2));
    layer.feed_forward(&input, &mut output, true);

    // Same libm, same op order -> bit-identical.
    assert_eq!(output[[0, 0]].to_bits(), y0.to_bits(), "softmax y0");
    assert_eq!(output[[0, 1]].to_bits(), y1.to_bits(), "softmax y1");
}

// === Task 5: Python-oracle cross-check =======================================
//
// `dense_cases.json` (scripts/extract_phase2_oracle_cases.py) holds the Python
// `dense_forward_oracle` forward outputs over a deterministic (I,O,T,last_layer)
// grid, with every f64 stored as its raw u64 bits (hex). The JSON is a COMMITTED
// FIXTURE (frozen at generation time on the oracle env, Apple libm) -- the softmax/
// logistic/asinh chain traverses exp, so this is the cross-libm class: gated via
// `assert_oracle_eq`, not a raw `to_bits` cross-check.

#[test]
fn dense_python_oracle_cross_check() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase2/dense_cases.json");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "dense_cases.json has no cases");

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let i = case["I"].as_u64().unwrap() as usize;
        let o = case["O"].as_u64().unwrap() as usize;
        let t_len = case["T"].as_u64().unwrap() as usize;
        let in_cols = case["in_cols"].as_u64().unwrap() as usize;
        let last_layer = case["last_layer"].as_bool().unwrap();

        let flat: Vec<f64> = case["weights_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| bits_to_f64(s.as_str().unwrap()))
            .collect();
        let input = matrix_from_bits(&case["input_bits"]);
        let want_y = matrix_from_bits(&case["y_bits"]);

        let mut layer = NeuronLayer::new(i, o);
        let tail = layer.set_weights(&flat);
        assert_eq!(tail.len(), 0, "{name}: flat vector fully consumed");
        let mut output = Array2::<f64>::zeros((t_len, o));
        layer.feed_forward(&input, &mut output, last_layer);
        assert_eq!(input.ncols(), in_cols, "{name}: input width sanity check");

        common::assert_oracle_eq(&output, &want_y, &format!("{name} y"));
    }
}
