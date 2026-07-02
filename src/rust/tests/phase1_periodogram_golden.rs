//! Phase 1 Task 5: bit-exact golden tests for the two-real periodogram + framing
//! driver (`compute_segment_periodogram_estimates` / `get_sequence` /
//! `compute_two_real_periodogram`).
//!
//! Goldens are dumped by the oracle harness running the REAL legacy
//! `AudioStruct::computeSegmentPeriodogramEstimates` on the same preemph+noise
//! audio as the audio-stage goldens. Bit-exactness (`to_bits`) is the contract;
//! the whole point is to reproduce the load-bearing quirks (stale-buffer reuse,
//! odd-tail double write, last-sample drop) byte-for-byte.

mod common;

use ndarray::Array2;
use speech::audio::{
    compute_segment_periodogram_estimates, get_sequence, read_audio, windowing_coefficients,
};
use speech::features::fft::{Gfft, compute_two_real_periodogram};

/// Reproduce the harness periodogram pipeline for one channel, returning the raw
/// (no temporal conv) periodogram. Mirrors main.cpp exactly: read excerpt (0.35,
/// 2.0) -> preemph 0.97 -> noise 0.001 -> hamming-257 window -> p=8 shift=80
/// dc_offset=true begin=0 end=frames-1.
fn pipeline(chan: usize, conv: Option<&[f64]>, end: usize) -> Array2<f64> {
    let wav = common::fixture("excerpt_2ch_8k.wav");
    let mut audio = read_audio(&wav, 0.35, 2.0).unwrap();
    audio.apply_preemph(0.97);
    audio.apply_noise(0.001);
    let win = windowing_coefficients("hamming", false, 257, 0.83333).unwrap();
    compute_segment_periodogram_estimates(&audio, 8, 80, chan, true, Some(&win), conv, 0, end)
}

/// full end = frames-1 = 16000 (excerpt is 16001 frames after the 0.35/2.0 cut).
const END_FULL: usize = 16000;
/// Odd-frame-count variant end (recorded in manifest.periodogram.perio_odd_end).
const PERIO_ODD_END: usize = 399;

// All periodogram goldens apply the hamming window (built via cos), so their Rust
// chain reaches cos -> canary-gated (bit-exact on the oracle libm, <=4 ULP else).
#[test]
fn perio_p8_s80_chan1_bitexact() {
    let got = pipeline(0, None, END_FULL);
    let want = common::load_bin("perio_p8_s80_chan1.bin");
    common::assert_oracle_eq(&got, &want, "perio_p8_s80_chan1");
}

#[test]
fn perio_p8_s80_chan2_bitexact() {
    let got = pipeline(1, None, END_FULL);
    let want = common::load_bin("perio_p8_s80_chan2.bin");
    common::assert_oracle_eq(&got, &want, "perio_p8_s80_chan2");
}

#[test]
fn perio_conv_p8_s80_chan1_bitexact() {
    let conv = windowing_coefficients("hann", true, 7, 0.0).unwrap();
    let got = pipeline(0, Some(&conv), END_FULL);
    let want = common::load_bin("perio_conv_p8_s80_chan1.bin");
    common::assert_oracle_eq(&got, &want, "perio_conv_p8_s80_chan1");
}

#[test]
fn perio_conv_p8_s80_chan2_bitexact() {
    let conv = windowing_coefficients("hann", true, 7, 0.0).unwrap();
    let got = pipeline(1, Some(&conv), END_FULL);
    let want = common::load_bin("perio_conv_p8_s80_chan2.bin");
    common::assert_oracle_eq(&got, &want, "perio_conv_p8_s80_chan2");
}

#[test]
fn perio_odd_chan1_bitexact() {
    // end=399 -> frameNb = ceil(400/80) = 5 (odd): exercises the odd-tail double
    // write. No temporal conv.
    let got = pipeline(0, None, PERIO_ODD_END);
    assert_eq!(got.nrows(), 5, "odd variant frameNb should be 5");
    let want = common::load_bin("perio_odd_chan1.bin");
    common::assert_oracle_eq(&got, &want, "perio_odd_chan1");
}

/// Hand-checked two-real unpack (n=4). Packs sig1/sig2 first 4 of 5 samples (the
/// 5th, value 9.0, proves the last-sample drop). The packed complex sequence is
///   z = [sig1[i] + i*sig2[i]] for i=0..3 = [1+0i, 0+1i, 0+0i, 0+0i] = [1, i, 0, 0].
/// Its DFT (negative exponent, unnormalized) X(k) = sum_i z(i) W^{ki}, W = e^{-2pi i/4} = -i:
///   k=0: z0+z1+z2+z3           = 1 + i          (data[0]=1, data[1]=1)
///   k=1: z0 + z1*W  + 0 + 0    = 1 + i*(-i) = 2 (data[2]=2, data[3]=0)
///   k=2: z0 + z1*W^2 + 0 + 0   = 1 + i*(-1) = 1 - i (data[4]=1, data[5]=-1)
///   k=3: z0 + z1*W^3 + 0 + 0   = 1 + i*(i)  = 0 (data[6]=0, data[7]=0)
/// The GFFT stores interleaved data[2k]=Re X(k), data[2k+1]=Im X(k):
///   data = [1,1, 2,0, 1,-1, 0,0]  (size 2n=8)
/// Unpack (legacy formulas over size_1=2n=8, size_2=2n+1=9, coeff_norm=4n=16):
///   P1[0] = data[0]^2 / n = 1/4;  P2[0] = data[1]^2 / n = 1/4
///   ii=2 (k=1): size_1-ii=6, size_2-ii=7
///     P1 = ((data[2]+data[6])^2 + (data[3]-data[7])^2)/16 = ((2+0)^2+(0-0)^2)/16 = 4/16
///     P2 = ((data[3]+data[7])^2 + (data[6]-data[2])^2)/16 = ((0+0)^2+(0-2)^2)/16 = 4/16
///   ii=4 (k=2, Nyquist): size_1-ii=4, size_2-ii=5
///     P1 = ((data[4]+data[4])^2 + (data[5]-data[5])^2)/16 = ((1+1)^2+0)/16 = 4/16
///     P2 = ((data[5]+data[5])^2 + (data[4]-data[4])^2)/16 = (((-1)+(-1))^2+0)/16 = 4/16
#[test]
fn two_real_unpack_hand_case_n4() {
    let n = 4usize;
    let sig1 = [1.0, 0.0, 0.0, 0.0, 9.0]; // 5th sample (9.0) must be dropped
    let sig2 = [0.0, 1.0, 0.0, 0.0, 9.0];
    let gfft = Gfft::new(2); // n = 1<<2 = 4
    let mut out = Array2::<f64>::zeros((2, n / 2 + 1)); // 2 rows, periodogram_length=3
    compute_two_real_periodogram(&gfft, n, &sig1, &sig2, &mut out, 0, 1);

    // Exact quotient expressions (not decimal literals) so bit-comparison is clean.
    assert_eq!(out[[0, 0]].to_bits(), (1.0_f64 / 4.0).to_bits());
    assert_eq!(out[[1, 0]].to_bits(), (1.0_f64 / 4.0).to_bits());
    assert_eq!(out[[0, 1]].to_bits(), (4.0_f64 / 16.0).to_bits());
    assert_eq!(out[[1, 1]].to_bits(), (4.0_f64 / 16.0).to_bits());
    assert_eq!(out[[0, 2]].to_bits(), (4.0_f64 / 16.0).to_bits());
    assert_eq!(out[[1, 2]].to_bits(), (4.0_f64 / 16.0).to_bits());
}

/// `get_sequence` edge cases. Data row = [0..10), half_window w=3 (buffer 1x7),
/// pre-seeded with 7.0 sentinels; coeffs=None throughout so no windowing masks
/// the pad. frames = 10.
#[test]
fn get_sequence_edge_cases() {
    let data = Array2::from_shape_vec((1, 10), (0..10).map(|v| v as f64).collect()).unwrap();
    let w = 3usize;

    // Left edge index=1 (< w): buffer NOT zeroed. beg_win = w-index = 2, so
    // buf[0..2] keep the 7.0 sentinels; buf[2..7] = data[0..5] = [0,1,2,3,4].
    {
        let mut buf = Array2::from_elem((1, 2 * w + 1), 7.0);
        get_sequence(&data, 0, 1, w, false, None, &mut buf);
        let row: Vec<f64> = buf.row(0).to_vec();
        assert_eq!(row, vec![7.0, 7.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    // Right edge index=9 (>= frames-w = 7): buffer IS zeroed first. beg_win=0,
    // nb_elem = w - index + frames = 3 - 9 + 10 = 4, beg_data = index-w = 6.
    // buf[0..4] = data[6..10] = [6,7,8,9], rest 0.0.
    {
        let mut buf = Array2::from_elem((1, 2 * w + 1), 7.0);
        get_sequence(&data, 0, 9, w, false, None, &mut buf);
        let row: Vec<f64> = buf.row(0).to_vec();
        assert_eq!(row, vec![6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0]);
    }

    // index=10 (>= frames): outer guard fails -> NO-OP, buffer untouched (all 7.0).
    {
        let mut buf = Array2::from_elem((1, 2 * w + 1), 7.0);
        get_sequence(&data, 0, 10, w, false, None, &mut buf);
        assert!(buf.iter().all(|&v| v == 7.0));
    }

    // DC offset over the FULL 2w+1 buffer INCLUDING the left-edge pad. index=1,
    // pad sentinels seeded to 0.0 first so the pad participates as 0. buffer =
    // [0,0, 0,1,2,3,4]; mean = 10/7; every entry (incl. pad) becomes v - mean.
    {
        let mut buf = Array2::from_elem((1, 2 * w + 1), 0.0);
        get_sequence(&data, 0, 1, w, true, None, &mut buf);
        let mean = (0.0 + 0.0 + 0.0 + 1.0 + 2.0 + 3.0 + 4.0) / 7.0;
        let expect = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0].map(|v: f64| v - mean);
        for (i, &e) in expect.iter().enumerate() {
            assert_eq!(buf[[0, i]].to_bits(), e.to_bits(), "dc idx {i}");
        }
    }
}
