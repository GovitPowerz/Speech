//! Phase 1 Task 8: bit-exact golden tests for the LTSV score
//! (`ltsv_classify_sequence`, ported from `LongTermSpectralVariation.cpp:82-128`)
//! and the interpolated column (`get_ltsv`, ported from
//! `BLSTMSpectralSegmenter.cpp:316-339`).
//!
//! Goldens are dumped by the oracle harness running the transcribed legacy logic
//! (the LTSV class itself cannot be compiled standalone -- it drags the whole
//! Segmenter hierarchy -- so `tools/oracle_harness/main.cpp` carries a verbatim
//! transcription instead). `to_bits` equality is the contract.

mod common;

use ndarray::Array2;
use serde::Deserialize;
use speech::features::ltsv_tdc::{get_ltsv, ltsv_classify_sequence};

fn synth_20x50() -> Array2<f64> {
    common::load_bin("synth_20x50.bin")
}

// --- Golden tests: get_ltsv against the two harness dumps ------------------

#[test]
fn ltsv_chan1_column_bitexact() {
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    let got = get_ltsv(&perio, 0, 128, 15, 4);
    let want = common::load_bin("ltsv_chan1.bin");
    assert_eq!(got.len(), want.nrows());
    for (i, gv) in got.iter().enumerate() {
        let wv = want[[i, 0]];
        assert_eq!(
            gv.to_bits(),
            wv.to_bits(),
            "ltsv_chan1 row {i}: rust=0x{:016x} ({gv}) want=0x{:016x} ({wv})",
            gv.to_bits(),
            wv.to_bits()
        );
    }
}

#[test]
fn ltsv_synth_column_bitexact() {
    let synth = synth_20x50();
    let got = get_ltsv(&synth, 0, 49, 3, 1);
    let want = common::load_bin("ltsv_synth.bin");
    assert_eq!(got.len(), want.nrows());
    for (i, gv) in got.iter().enumerate() {
        let wv = want[[i, 0]];
        assert_eq!(
            gv.to_bits(),
            wv.to_bits(),
            "ltsv_synth row {i}: rust=0x{:016x} ({gv}) want=0x{:016x} ({wv})",
            gv.to_bits(),
            wv.to_bits()
        );
    }
}

// --- Hand test: variance-not-std, mean-floor placement, inclusive window ---

#[test]
fn ltsv_is_biased_variance_no_sqrt() {
    // P = [[1,2],[3,4],[5,6]] (T=3, bins=2), col=1, R=1 -> window rows 0..=2, L=3
    // bin0: mean=3; r={1/3,1,5/3}; sum r(r-1) = -2/9 + 0 + 10/9 = 8/9; dzeta0 = -(8/9)/3
    // bin1: mean=4; r={1/2,1,3/2}; sum = -1/4 + 0 + 3/4 = 1/2;      dzeta1 = -(1/2)/3
    // return ((d0-m)^2 + (d1-m)^2)/2, m = (d0+d1)/2  -- VARIANCE, no sqrt
    let p = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let got = ltsv_classify_sequence(&p, 1, 0, 1, 1);
    let dz = |vals: [f64; 3]| {
        let mean = ((vals[0] + vals[1] + vals[2]) / 3.0).max(1e-12);
        let mut s = 0.0;
        for v in vals {
            let r = v / mean;
            s -= r * (r - 1.0);
        }
        s / 3.0
    };
    let (d0, d1) = (dz([1.0, 3.0, 5.0]), dz([2.0, 4.0, 6.0]));
    let m = (d0 + d1) / 2.0;
    assert_eq!(got, ((d0 - m).powi(2) + (d1 - m).powi(2)) / 2.0);
}

// --- Edge tests -------------------------------------------------------------

#[test]
fn ltsv_edge_shrink_at_col_zero() {
    // col=0, R=1 -> window would be [-1,1] but clamps to [0,1] (length 2, NOT
    // padded to 3). P = [[1,2],[3,4],[5,6]], bin range 0..=1.
    let p = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let got = ltsv_classify_sequence(&p, 0, 0, 1, 1);
    let dz = |vals: [f64; 2]| {
        let mean = ((vals[0] + vals[1]) / 2.0).max(1e-12);
        let mut s = 0.0;
        for v in vals {
            let r = v / mean;
            s -= r * (r - 1.0);
        }
        s / 2.0
    };
    let (d0, d1) = (dz([1.0, 3.0]), dz([2.0, 4.0]));
    let m = (d0 + d1) / 2.0;
    assert_eq!(got, ((d0 - m).powi(2) + (d1 - m).powi(2)) / 2.0);
}

#[test]
fn ltsv_all_zero_bin_floors_mean_dzeta_zero() {
    // bin1 is all-zero over the window -> mean floors to 1e-12, r = 0/1e-12 = 0,
    // dzeta1 contribution per sample is -(0*(0-1)) = 0 -> dzeta1 == 0 exactly.
    let p = ndarray::arr2(&[[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]);
    let got = ltsv_classify_sequence(&p, 1, 0, 1, 1);
    let mean0 = (1.0 + 3.0 + 5.0) / 3.0_f64;
    let dz0 = {
        let mut s = 0.0;
        for v in [1.0, 3.0, 5.0] {
            let r = v / mean0;
            s -= r * (r - 1.0);
        }
        s / 3.0
    };
    let dz1 = 0.0_f64; // all-zero bin: mean floors to 1e-12, r=0 each sample, dzeta=0
    let m = (dz0 + dz1) / 2.0;
    let expect = ((dz0 - m).powi(2) + (dz1 - m).powi(2)) / 2.0;
    assert_eq!(got, expect);
}

// --- get_ltsv interpolation -------------------------------------------------

#[test]
fn get_ltsv_interpolates_and_leaves_trailing_zero() {
    // 5 rows, shift=2: computed at jj=0,2,4; jj=1 is the backfilled mm=1 in
    // (0,2] with coeffInterp=1/2 -> row1 = (row0+row2)/2. jj=3 is backfilled
    // between 2 and 4 -> row3 = (row2+row4)/2. Trailing rows after the last
    // computed jj (4, the final row here) stay at their computed value; use a
    // 6-row matrix so a genuine trailing zero exists after jj=4.
    let p = ndarray::arr2(&[
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0],
    ]);
    let half_window = 1;
    let freq_beg = 0;
    let freq_end = 1;
    let shift = 2;
    let got = get_ltsv(&p, freq_beg, freq_end, half_window, shift);
    assert_eq!(got.len(), 6);

    let row0 = ltsv_classify_sequence(&p, 0, freq_beg, freq_end, half_window);
    let row2 = ltsv_classify_sequence(&p, 2, freq_beg, freq_end, half_window);
    let row4 = ltsv_classify_sequence(&p, 4, freq_beg, freq_end, half_window);
    assert_eq!(got[0].to_bits(), row0.to_bits());
    assert_eq!(got[2].to_bits(), row2.to_bits());
    assert_eq!(got[4].to_bits(), row4.to_bits());

    let coeff = 0.5_f64;
    let blend1 = (1.0 - coeff) * row0 + coeff * row2;
    assert_eq!(got[1].to_bits(), blend1.to_bits());
    let blend3 = (1.0 - coeff) * row2 + coeff * row4;
    assert_eq!(got[3].to_bits(), blend3.to_bits());

    // Trailing row after the last computed jj (4): loop step is jj += 2, so jj=6
    // exits (>= vec_size=6) and row 5 is never touched -> stays 0.0.
    assert_eq!(got[5].to_bits(), 0.0_f64.to_bits());
}

// --- Rust == Python numpy-oracle cross-check --------------------------------

#[derive(Deserialize)]
struct LtsvCase {
    p: Vec<Vec<f64>>,
    col: usize,
    freq_beg: usize,
    freq_end: usize,
    half_window: usize,
    expected: f64,
}

fn to_array(rows: &[Vec<f64>]) -> Array2<f64> {
    let t = rows.len();
    let w = if t == 0 { 0 } else { rows[0].len() };
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((t, w), flat).unwrap()
}

#[test]
fn ltsv_oracle_cross_check_bit_exact() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase1/ltsv_cases.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let cases: Vec<LtsvCase> = serde_json::from_str(&text).unwrap();
    assert!(
        cases.len() >= 18,
        "expected >= 18 cases, got {}",
        cases.len()
    );

    for (idx, case) in cases.iter().enumerate() {
        let p = to_array(&case.p);
        let got =
            ltsv_classify_sequence(&p, case.col, case.freq_beg, case.freq_end, case.half_window);
        assert_eq!(
            got.to_bits(),
            case.expected.to_bits(),
            "case {idx} col={} R={} band=({},{}): rust=0x{:016x} ({got}) py=0x{:016x} ({})",
            case.col,
            case.half_window,
            case.freq_beg,
            case.freq_end,
            got.to_bits(),
            case.expected.to_bits(),
            case.expected
        );
    }
}
