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
use speech::features::ltsv_tdc::{
    apply_homothety, compute_pitch, fmath_log, get_ltsv, get_pitch, ltsv_classify_sequence,
    tdc_classify_sequence,
};
use speech::features::pipeline::TdcParams;
use speech::tasks::segmentation::{SegClass, Segmentation};

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

// === Task 9: fmath::log + TDC + pitch + homothety ==========================

/// TdcParams derived inline from the brief config (TDC_window=0.032, TDC_shift=0.01,
/// lags=(0.002,0.016), balance=0.7 @ rate 8000): half_window=128, full_window=257,
/// shift=80, min_lag=16, max_lag=128, hamming-257 window (param 0.8). Matches the
/// harness derivation verbatim.
fn tdc_params() -> TdcParams {
    let coeffs = speech::audio::windowing_coefficients("hamming", false, 257, 0.8);
    TdcParams {
        half_window: 128,
        full_window: 257,
        shift: 80,
        min_lag: 16,
        max_lag: 128,
        coeffs,
        balance: 0.7,
    }
}

/// Preemph+noise excerpt audio (chan 0), the same recipe every earlier phase-1
/// audio golden uses (offset 0.35, dur 2.0, preemph 0.97, noise 0.001).
fn excerpt_audio() -> speech::audio::Audio {
    let mut audio = speech::audio::read_audio(&common::fixture("excerpt_2ch_8k.wav"), 0.35, 2.0)
        .expect("decode excerpt");
    audio.apply_preemph(0.97);
    audio.apply_noise(0.001);
    audio
}

// --- Golden: fmath_log sweep (2 x N: row0 input, row1 fmath::log(input)) -----

#[test]
fn fmath_log_sweep_bitexact() {
    // fmath_log builds its table with f64 ln (libm), so the f32 results are libm-
    // dependent -> canary-gated. f32-valued (the fmath table is f32): hybrid <=2 f32 ULP/abs off
    // the oracle libm.
    let want = common::load_bin("fmath_log_sweep.bin"); // 2 x N
    assert_eq!(want.nrows(), 2);
    for k in 0..want.ncols() {
        let x = want[[0, k]] as f32; // input was dumped as f32 widened to f64
        let got = fmath_log(x);
        let expect = want[[1, k]] as f32; // f32 result stored widened to f64
        common::assert_oracle_eq_f32(got, expect, &format!("fmath_log col {k} x={x}"));
    }
}

// --- Golden: TDC score column over the excerpt (chan 0) ----------------------

#[test]
fn tdc_chan1_column_bitexact() {
    let audio = excerpt_audio();
    let tdc = tdc_params();
    let coeffs = tdc.coeffs.as_deref();
    let frames = audio.data.ncols();
    let vec_size = if (frames / tdc.shift) * tdc.shift == frames {
        frames / tdc.shift
    } else {
        frames / tdc.shift + 1
    };
    let mut buf = Array2::<f64>::zeros((1, tdc.full_window));
    let mut got = vec![0.0f64; vec_size];
    let mut jj = 0usize;
    while jj < frames {
        speech::audio::get_sequence(&audio.data, 0, jj, tdc.half_window, false, coeffs, &mut buf);
        let seq = buf.row(0).to_vec();
        got[jj / tdc.shift] = tdc_classify_sequence(&seq, tdc.min_lag, tdc.max_lag, tdc.balance);
        jj += tdc.shift;
    }

    // The chain reaches cos (hamming window feeding get_sequence) AND fmath_log's
    // libm-built table -> canary-gated (bit-exact on the oracle libm, hybrid <=4 ULP/abs else).
    let want = common::load_bin("tdc_chan1.bin"); // 1 x frames
    assert_eq!(want.nrows(), 1);
    assert_eq!(got.len(), want.ncols());
    for (i, gv) in got.iter().enumerate() {
        let wv = want[[0, i]];
        common::assert_oracle_eq_f64(*gv, wv, &format!("tdc_chan1 frame {i}"));
    }
}

// --- Golden: pitch scalar over a hand-built segmentation ---------------------

/// Single SPEECH segment covering the middle 1.0s of the ~2s excerpt: [0.5s, 1.5s).
/// Boundary list Other@0.0, Speech@0.5, Other@1.5, End@duration -- the exact segment
/// the harness's `pitch_chan1.bin` used (documented in the manifest).
fn pitch_segmentation(audio: &speech::audio::Audio) -> Segmentation {
    let duration = audio.data.ncols() as f64 / 8000.0;
    let mut seg = Segmentation::new(duration);
    seg.label_segment(0.5, 1.5, SegClass::Speech);
    seg
}

#[test]
fn pitch_chan1_scalar_bitexact() {
    let audio = excerpt_audio();
    let tdc = tdc_params();
    let seg = pitch_segmentation(&audio);
    // get_pitch windows with the hamming coeffs (cos) before autocorrelation ->
    // canary-gated. (The pitch estimate is an integer quotient, so it is typically
    // robust even under a differing libm; the gate only relaxes if it is not.)
    let got = get_pitch(&audio, &seg, 0, &tdc, 8000.0, false);
    let want = common::load_bin("pitch_chan1.bin"); // 1 x 1
    assert_eq!((want.nrows(), want.ncols()), (1, 1));
    common::assert_oracle_eq_f64(got, want[[0, 0]], "pitch");
}

// --- Golden: homothety warp of the chan-1 periodogram ------------------------

#[test]
fn perio_homothety_chan1_bitexact() {
    let audio = excerpt_audio();
    let tdc = tdc_params();
    let seg = pitch_segmentation(&audio);
    let pitch = get_pitch(&audio, &seg, 0, &tdc, 8000.0, false);
    let coeff = pitch / 300.0;

    // coeff = get_pitch(...)/300, and get_pitch windows with cos coeffs -> the warp
    // depends on cos through the pitch estimate -> canary-gated.
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    let got = apply_homothety(&perio, coeff);
    let want = common::load_bin("perio_homothety_chan1.bin");
    common::assert_oracle_eq(&got, &want, "perio_homothety_chan1");
}

// --- Unit tests (brief) ------------------------------------------------------

#[test]
// The brief's pinned literal -88.029694 is kept verbatim; clippy notes -88.02969
// parses to the same f32 (0xc2b00f34) but the longer form matches the dump printout.
#[allow(clippy::excessive_precision)]
fn fmath_log_at_zero_is_finite() {
    // fmath.hpp:713-727 - no x<=0 guard; log(0f) = (0 - (127<<23)) as f32 * c_log2
    // + app[0] + 0. The c_log2 table slope is f64-ln-built, so the exact bits are
    // libm-dependent; pin the exact literal (0xc2b00f34) only on the oracle libm,
    // else assert finiteness + hybrid <=2 f32 ULP/abs.
    let v = fmath_log(0.0);
    assert!(v.is_finite());
    common::assert_oracle_eq_f32(v, -88.029694, "fmath_log(0)");
}

#[test]
fn tdc_r0_is_one_when_min_lag_zero_and_score_finite() {
    // min_lag=0 -> R[0] is the lag-0 autocorrelation == 1 -> MaxPeak >= 1 ->
    // fmath_log(1 - MaxPeak) evaluated at <= 0 (finite, no guard).
    let w: Vec<f64> = (0..9).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
    let s = tdc_classify_sequence(&w, 0, 4, 0.7);
    assert!(s.is_finite());
}

#[test]
fn tdc_fewer_than_three_crossings_zero_crosscorr() {
    // All-ones, min_lag=1, max_lag=3: R[k] = (5-k)/5 for k=1..3 -> R = [4/5,3/5,2/5],
    // monotone decreasing and strictly positive -> NO zero crossing -> CrossCorr=0,
    // so score = balance*(-fmath_log(1 - MaxPeak)) with MaxPeak = 4/5 (the brief's
    // comment mis-stated r_max as 3/5; the actual peak is R[1]=4/5).
    let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let s = tdc_classify_sequence(&w, 1, 3, 0.5);
    let r_max = (4.0f64 / 5.0).max((3.0f64 / 5.0).max(2.0f64 / 5.0));
    assert_eq!(s, 0.5 * (-(fmath_log((1.0 - r_max) as f32) as f64)));
}

#[test]
fn compute_pitch_accepts_inside_band() {
    // A window whose dominant lag lands inside (rate/max_lag, rate/min_lag).
    // Period-4 square wave at rate 8000, min_lag=2, max_lag=6: argmax lag = 4 ->
    // est = 8000/4 = 2000, band = (8000/6, 8000/2) = (1333.3, 4000) -> accepted.
    let w: Vec<f64> = (0..40)
        .map(|i| if (i / 2) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let est = compute_pitch(&w, 2, 6, 8000.0);
    assert_eq!(est, Some(2000.0));
}

#[test]
fn homothety_interpolates_bins() {
    // coeff=0.5: out[0][2] = pos=(int)(0.5*2)=1, alpha=0 -> exactly p[0][1] = 2.0.
    let p = ndarray::arr2(&[[1.0, 2.0, 3.0, 4.0]]);
    let out = apply_homothety(&p, 0.5);
    assert_eq!(out[[0, 2]], 2.0);
}

#[test]
fn homothety_alpha_zero_is_exact_copy() {
    // coeff=1.0: pos=ii, alpha=0 for every column -> out == p (identity copy),
    // except the last column ii=cols-1 whose pos+1 is out of bounds but alpha=0 so
    // the (1-alpha)*p(t,pos) term is exact.
    let p = ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let out = apply_homothety(&p, 1.0);
    common::assert_bits_eq(&out, &p, "homothety identity");
}

// --- Rust == Python tdc_oracle cross-check ----------------------------------

#[derive(Deserialize)]
struct TdcCase {
    window: Vec<f64>,
    min_lag: usize,
    max_lag: usize,
    balance: f64,
    expected: f64,
}

#[test]
fn tdc_oracle_cross_check_bit_exact() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase1/tdc_cases.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let cases: Vec<TdcCase> = serde_json::from_str(&text).unwrap();
    assert!(
        cases.len() >= 4,
        "expected >= 4 tdc cases, got {}",
        cases.len()
    );

    for (idx, case) in cases.iter().enumerate() {
        // tdc_classify_sequence embeds fmath_log (libm-built f32 table) -> the f64
        // score is libm-dependent -> canary-gated (hybrid <=4 ULP/abs off the oracle libm).
        let got = tdc_classify_sequence(&case.window, case.min_lag, case.max_lag, case.balance);
        common::assert_oracle_eq_f64(
            got,
            case.expected,
            &format!(
                "tdc case {idx} min_lag={} max_lag={} balance={}",
                case.min_lag, case.max_lag, case.balance
            ),
        );
    }
}
