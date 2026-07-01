//! Phase 1 Task 6: bit-exact golden tests for the Mel filterbank construction,
//! `apply_filter_bank` (log/mel branches + the deltas-no-DCT `[delta|delta|dd]`
//! overwrite quirk), plus the shared regression-deltas kernel cross-checked
//! bit-for-bit against the numpy oracle via `deltas_cases.json`.
//!
//! Goldens are dumped by the oracle harness running the REAL legacy
//! `MelFilterBank::applyFilterBank`. `to_bits` equality is the contract.

// The re-derivation test walks `freqs` by absolute index (the parity contract),
// not a range-loop smell.
#![allow(clippy::needless_range_loop)]

mod common;

use ndarray::Array2;
use serde::Deserialize;
use speech::features::mel::{MelFilterBank, hz_to_mel, mel_to_hz, regression_deltas};

/// Constructor args mirroring the harness SAD spectral config, parameterized on
/// spectrum_size and the four toggles that vary across dumps.
fn bank(spectrum: usize, is_log: bool, deltas: i32, dd: i32) -> MelFilterBank {
    MelFilterBank::new(
        64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, spectrum, is_log, 0, false, deltas, dd,
    )
}

#[test]
fn logmel_26_chan1_bitexact() {
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    let got = bank(128, true, 0, 0).apply_filter_bank(&perio);
    let want = common::load_bin("logmel_26_chan1.bin");
    common::assert_bits_eq(&got, &want, "logmel_26_chan1");
}

#[test]
fn logmel_deltas_chan1_bitexact() {
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    let got = bank(128, true, 3, 3).apply_filter_bank(&perio);
    let want = common::load_bin("logmel_deltas_chan1.bin");
    common::assert_bits_eq(&got, &want, "logmel_deltas_chan1");
}

#[test]
fn mel_26_chan1_bitexact() {
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    let got = bank(128, false, 0, 0).apply_filter_bank(&perio);
    let want = common::load_bin("mel_26_chan1.bin");
    common::assert_bits_eq(&got, &want, "mel_26_chan1");
}

#[test]
fn logmel_synth_bitexact() {
    // synth_20x50 reinterpreted as a 20-frame periodogram; spectrum_size 49.
    let synth = common::load_bin("synth_20x50.bin");
    let got = bank(49, true, 0, 0).apply_filter_bank(&synth);
    let want = common::load_bin("logmel_synth.bin");
    common::assert_bits_eq(&got, &want, "logmel_synth");
}

#[test]
fn hz_mel_roundtrip_formulas() {
    assert_eq!(hz_to_mel(700.0).to_bits(), (1125.0 * 2.0f64.ln()).to_bits());
    // Mel2Hz(Hz2Mel(300)) via the plain-double formulas (expression form).
    let m = 1125.0 * (1.0f64 + 300.0 / 700.0).ln();
    let expect = 700.0 * (m / 1125.0).exp() - 700.0;
    assert_eq!(mel_to_hz(hz_to_mel(300.0)).to_bits(), expect.to_bits());
}

#[test]
fn grid_accumulates_sequentially() {
    // rate 8000, spectrum_size 49 -> freqStep = 81.632653...; the mel walk yields
    // a valid (non-fallback) bank.
    let fb = bank(49, true, 0, 0);
    assert!(fb.is_mel());
    assert!(!fb.is_dct_activated());
}

#[test]
fn empty_filter_collapses_whole_bank_to_passthrough() {
    // spectrum_size 8 (9 bins) with 26 requested filters: some triangle catches no
    // bin, so the WHOLE bank collapses to raw-band pass-through.
    let fb = bank(8, true, 0, 0);
    assert!(!fb.is_mel(), "expected fallback (some filter empty)");
    // fallback nb_filters = _EndFreq - _BegFreq + 1 = 8 - 0 + 1 = 9.
    assert_eq!(fb.nb_filters(), 9);
}

#[test]
fn triangle_first_coeff_matches_sequential_walk() {
    // Re-derive the first two mel edges with the SAME sequential walk the ctor
    // uses (spectrum_size 128, rate 8000 -> freqStep 31.25), then assert filter 0's
    // first coefficient equals the rising-edge expression from the first
    // qualifying frequency bin.
    let (min_mel, max_mel) = (64.0f64, 3800.0f64);
    let (min_freq, max_freq, rate) = (64.0f64, 3800.0f64, 8000.0f64);
    let nb_bins = 26i32;
    let spectrum = 128usize;
    let freq_step = rate / 2.0 / spectrum as f64;

    // freqs grid by sequential accumulation.
    let mut freqs = Vec::new();
    let mut f = 0.0;
    let limit = rate / 2.0;
    while f <= limit {
        freqs.push(f);
        f += freq_step;
    }

    // mels walk: down then up (mel edges as Hz).
    let min_mel_v = hz_to_mel(min_mel);
    let max_mel_v = hz_to_mel(max_mel);
    let mut mel_step = (max_mel_v - min_mel_v) / (nb_bins as f64 + 1.0);
    if mel_step < hz_to_mel(1.0) {
        mel_step = hz_to_mel(1.0);
    }
    let mut mel = min_mel_v;
    while mel > hz_to_mel(min_freq.max(0.0)) - mel_step {
        mel -= mel_step;
    }
    let mels_limit = hz_to_mel(max_freq.min(rate / 2.0));
    let mut mels = Vec::new();
    while mel < mels_limit + 2.0 * mel_step {
        mels.push(mel_to_hz(mel));
        mel += mel_step;
    }

    let beg = (min_freq.max(0.0) / freq_step).floor() as usize;
    let end = (max_freq.min(rate / 2.0) / freq_step).ceil() as usize;

    // Filter ii=1 (first): span [mels[0], mels[2]]; first qualifying bin's coeff.
    let mut first_coeff = None;
    for jj in beg..=end {
        if freqs[jj] >= mels[0] && freqs[jj] <= mels[2] {
            let c = if freqs[jj] < mels[1] {
                (freqs[jj] - mels[0]) / (mels[1] - mels[0])
            } else {
                (mels[2] - freqs[jj]) / (mels[2] - mels[1])
            };
            first_coeff = Some(c);
            break;
        }
    }
    let expect = first_coeff.expect("first filter must catch a bin");

    // Apply the bank to a one-hot periodogram (frame 0, bin = first qualifying
    // index) so out[0][0] == the first coefficient of filter 0 (non-log path).
    let fb = MelFilterBank::new(
        min_mel, max_mel, nb_bins, min_freq, max_freq, rate, spectrum, false, 0, false, 0, 0,
    );
    // first qualifying bin index for filter 0:
    let mut idx0 = None;
    for jj in beg..=end {
        if freqs[jj] >= mels[0] && freqs[jj] <= mels[2] {
            idx0 = Some(jj);
            break;
        }
    }
    let idx0 = idx0.unwrap();
    let mut perio = Array2::<f64>::zeros((1, 129));
    perio[[0, idx0]] = 1.0;
    let out = fb.apply_filter_bank(&perio);
    assert_eq!(out[[0, 0]].to_bits(), expect.to_bits());
}

// --- regression_deltas hand cases -----------------------------------------

#[test]
fn regression_deltas_hand_anchor() {
    // base [[0],[1],[4]], n=2 -> [[0.9],[1.2],[1.1]] (see task brief arithmetic).
    let base = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 4.0]).unwrap();
    let d = regression_deltas(&base, 2);
    assert_eq!(d[[0, 0]].to_bits(), 0.9f64.to_bits());
    assert_eq!(d[[1, 0]].to_bits(), 1.2f64.to_bits());
    assert_eq!(d[[2, 0]].to_bits(), 1.1f64.to_bits());
}

// --- Rust == Python numpy-oracle cross-check ------------------------------

#[derive(Deserialize)]
struct DeltaCase {
    base: Vec<Vec<f64>>,
    n: i32,
    expected: Vec<Vec<f64>>,
}

fn to_array(rows: &[Vec<f64>]) -> Array2<f64> {
    let t = rows.len();
    let w = if t == 0 { 0 } else { rows[0].len() };
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((t, w), flat).unwrap()
}

#[test]
fn regression_deltas_oracle_cross_check_bit_exact() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase1/deltas_cases.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let cases: Vec<DeltaCase> = serde_json::from_str(&text).unwrap();
    assert!(
        cases.len() >= 20,
        "expected >= 20 cases, got {}",
        cases.len()
    );

    for (idx, case) in cases.iter().enumerate() {
        let base = to_array(&case.base);
        let expected = to_array(&case.expected);
        let got = regression_deltas(&base, case.n);
        assert_eq!(got.shape(), expected.shape(), "case {idx}: shape mismatch");
        for ((i, gv), ev) in got.indexed_iter().zip(expected.iter()) {
            assert_eq!(
                gv.to_bits(),
                ev.to_bits(),
                "case {idx} n={} at {i:?}: rust=0x{:016x} ({gv}) py=0x{:016x} ({ev})",
                case.n,
                gv.to_bits(),
                ev.to_bits()
            );
        }
    }
}
