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
use speech::nn::layers::LstmLayer;

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
