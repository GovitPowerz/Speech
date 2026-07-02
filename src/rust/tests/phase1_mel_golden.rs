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

/// DCT-active bank mirroring the harness Task 7 config (nb_dct=13), parameterized
/// on the ignoreFirst / deltas / delta-delta toggles that vary across the DCT dumps.
fn dct_bank(ignore_first: bool, deltas: i32, dd: i32) -> MelFilterBank {
    MelFilterBank::new(
        64.0,
        3800.0,
        26,
        64.0,
        3800.0,
        8000.0,
        128,
        true,
        13,
        ignore_first,
        deltas,
        dd,
    )
}

// Mel bank ctor calls hz_to_mel (ln) + mel_to_hz (exp); the log branch calls ln.
// So every filter-bank golden's Rust chain reaches ln/exp -> canary-gated.
#[test]
fn logmel_26_chan1_bitexact() {
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    let got = bank(128, true, 0, 0).apply_filter_bank(&perio);
    let want = common::load_bin("logmel_26_chan1.bin");
    common::assert_oracle_eq(&got, &want, "logmel_26_chan1");
}

#[test]
fn logmel_deltas_chan1_bitexact() {
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    let got = bank(128, true, 3, 3).apply_filter_bank(&perio);
    let want = common::load_bin("logmel_deltas_chan1.bin");
    common::assert_oracle_eq(&got, &want, "logmel_deltas_chan1");
}

#[test]
fn mel_26_chan1_bitexact() {
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    let got = bank(128, false, 0, 0).apply_filter_bank(&perio);
    let want = common::load_bin("mel_26_chan1.bin");
    common::assert_oracle_eq(&got, &want, "mel_26_chan1");
}

#[test]
fn logmel_synth_bitexact() {
    // synth_20x50 reinterpreted as a 20-frame periodogram; spectrum_size 49.
    let synth = common::load_bin("synth_20x50.bin");
    let got = bank(49, true, 0, 0).apply_filter_bank(&synth);
    let want = common::load_bin("logmel_synth.bin");
    common::assert_oracle_eq(&got, &want, "logmel_synth");
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
    // Structural only: asserts the ctor lands on a valid (non-fallback) bank for
    // this shape. The actual bit-exact PINNING of the sequential-accumulation grid
    // walk is carried by the logmel_synth/logmel_26 goldens (`logmel_synth_bitexact`
    // below), not this test.
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
    // Structural only: re-derives filter 0's first rising-edge coefficient via an
    // independent re-walk, it does not pin the ctor's own bit-exact accumulation
    // order. That pinning is carried by the logmel_synth/logmel_26 goldens
    // (`logmel_synth_bitexact` above), not this test.
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

// --- Task 7: apply_dct golden tests ---------------------------------------
//
// All five DCT dumps feed the SAME chan-1 log-mel (a DCT-active bank still emits
// plain log-mel from apply_filter_bank, so logmel_26_chan1.bin is that input).
// The harness product is the explicit ascending triple loop (Eigen GEMM diverged
// in the mandatory pre-check); apply_dct reproduces that order bit-for-bit.

fn logmel_input() -> Array2<f64> {
    // Identical to logmel_26_chan1 (bank with nb_dct=13 emits plain log-mel).
    let perio = common::load_bin("perio_p8_s80_chan1.bin");
    dct_bank(false, 0, 0).apply_filter_bank(&perio)
}

// DCT goldens: logmel_input calls ln (log branch) + apply_dct's cos table -> the
// Rust chain reaches ln and cos -> canary-gated (bit-exact on oracle libm, <=4 ULP).
#[test]
fn mfcc_chan1_bitexact() {
    let got = dct_bank(false, 0, 0).apply_dct(&logmel_input());
    let want = common::load_bin("mfcc_chan1.bin");
    common::assert_oracle_eq(&got, &want, "mfcc_chan1");
}

#[test]
fn mfcc_deltas_chan1_bitexact() {
    let got = dct_bank(false, 3, 3).apply_dct(&logmel_input());
    let want = common::load_bin("mfcc_deltas_chan1.bin");
    common::assert_oracle_eq(&got, &want, "mfcc_deltas_chan1");
}

#[test]
fn mfcc_deltas_if_chan1_bitexact() {
    let got = dct_bank(true, 3, 3).apply_dct(&logmel_input());
    let want = common::load_bin("mfcc_deltas_if_chan1.bin");
    common::assert_oracle_eq(&got, &want, "mfcc_deltas_if_chan1");
}

#[test]
fn mfcc_sdc_chan1_bitexact() {
    let got = dct_bank(false, -1, 0).apply_dct(&logmel_input());
    let want = common::load_bin("mfcc_sdc_chan1.bin");
    common::assert_oracle_eq(&got, &want, "mfcc_sdc_chan1");
}

#[test]
fn mfcc_sdc_if_chan1_bitexact() {
    // Pins the ignoreFirst clobber: SDC col 0 overwrites the LAST static (c12).
    let got = dct_bank(true, -1, 0).apply_dct(&logmel_input());
    let want = common::load_bin("mfcc_sdc_if_chan1.bin");
    common::assert_oracle_eq(&got, &want, "mfcc_sdc_if_chan1");
}

// --- Task 7: unit tests ---------------------------------------------------

#[test]
fn dct_matrix_entry_formula_sample() {
    // The DCT table is built as cos(PI/nb_filters*(col+0.5)*row). With nb_dct=13 <
    // nb_filters=29 the table is 29 x 13. apply_dct on a one-hot log-mel row (frame
    // 0, filter n) with deltas=0/dd=0/!ignoreFirst yields out[0][k] == coeff(n,k),
    // so we can read table entries straight out of the plain-MFCC path.
    let fb = dct_bank(false, 0, 0);
    let nb_filters = 29usize;
    let pi = std::f64::consts::PI;
    for &(n, k) in &[(0usize, 0usize), (0, 5), (7, 3), (28, 12), (13, 7), (28, 0)] {
        let mut logmel = Array2::<f64>::zeros((1, nb_filters));
        logmel[[0, n]] = 1.0;
        let mfcc = fb.apply_dct(&logmel);
        let expect = (pi / nb_filters as f64 * (n as f64 + 0.5) * k as f64).cos();
        assert_eq!(
            mfcc[[0, k]].to_bits(),
            expect.to_bits(),
            "coeff({n},{k}): got 0x{:016x} want 0x{:016x}",
            mfcc[[0, k]].to_bits(),
            expect.to_bits()
        );
    }
}

#[test]
fn nb_dct_width_table() {
    // getNbDCT (MelFilterBank.h:45-89) across all deltas/dd/ignoreFirst combos, with
    // nb_dct=13. (ignore_first, deltas, dd) -> expected width.
    let cases: &[(bool, i32, i32, usize)] = &[
        // deltas == 0: nb_dct, minus 1 if ignoreFirst.
        (false, 0, 0, 13),
        (true, 0, 0, 12),
        // deltas > 0, dd == 0: 2*nb_dct, minus 1 if ignoreFirst.
        (false, 3, 0, 26),
        (true, 3, 0, 25),
        // deltas > 0, dd > 0: 3*nb_dct, minus 1 if ignoreFirst.
        (false, 3, 3, 39),
        (true, 3, 3, 38),
        // deltas < 0 (SDC): (nb_dct or nb_dct-1) + 7*nb_dct.
        (false, -1, 0, 13 + 7 * 13),
        (true, -1, 0, 12 + 7 * 13),
    ];
    for &(ig, d, dd, expect) in cases {
        assert_eq!(
            dct_bank(ig, d, dd).nb_dct(),
            expect,
            "nb_dct(ignore_first={ig}, deltas={d}, dd={dd})"
        );
    }
}

#[test]
fn sdc_offset_structure_on_ramp() {
    // Synthetic log-mel ramp (T x 29). Run the SDC path, then assert each 13-wide
    // block kk (kk in 0..7) equals the n=3 delta D shifted by +10-3*kk, ZERO-padded
    // out of range. D is computed via the shared regression_deltas on the statics.
    let t = 25usize;
    let nb_filters = 29usize;
    let nb_dct = 13usize;
    let k = 7usize;
    let mut logmel = Array2::<f64>::zeros((t, nb_filters));
    for i in 0..t {
        for j in 0..nb_filters {
            logmel[[i, j]] = ((i * 7 + j * 13) % 100) as f64 / 100.0;
        }
    }
    let fb = dct_bank(false, -1, 0);
    let sdc = fb.apply_dct(&logmel); // (t, 13 + 7*13)
    // statics = first nb_dct cols; D = n=3 delta of statics.
    let statics = sdc.slice(ndarray::s![.., 0..nb_dct]).to_owned();
    let d = regression_deltas(&statics, 3);
    for kk in 0..k {
        for tt in 0..t {
            let src = tt as isize + 10 - 3 * kk as isize;
            for c in 0..nb_dct {
                let got = sdc[[tt, nb_dct + kk * nb_dct + c]];
                let want = if src >= 0 && (src as usize) < t {
                    d[[src as usize, c]]
                } else {
                    0.0
                };
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "SDC block {kk} t={tt} col{c}: src={src} got 0x{:016x} want 0x{:016x}",
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
    }
}

// --- Task 7: sdc_oracle Rust == Python cross-check ------------------------

#[derive(Deserialize)]
struct SdcCase {
    mfcc: Vec<Vec<f64>>,
    nb_dct: usize,
    expected: Vec<Vec<f64>>,
}

#[test]
fn sdc_oracle_cross_check_bit_exact() {
    // The Python sdc_oracle stacks the n=3-kernel deltas with zero-padded shifts.
    // Our SDC is the tail of apply_dct: we reproduce just the SDC stacking here by
    // feeding an MFCC directly as "statics" through the SDC math via regression_
    // deltas + the +10-3*kk offset, and cross-check against the oracle JSON.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase1/sdc_cases.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let cases: Vec<SdcCase> = serde_json::from_str(&text).unwrap();
    assert!(cases.len() >= 4, "expected >= 4 cases, got {}", cases.len());

    for (idx, case) in cases.iter().enumerate() {
        let mfcc = to_array(&case.mfcc);
        let expected = to_array(&case.expected);
        let got = sdc_from_mfcc(&mfcc, case.nb_dct);
        assert_eq!(got.shape(), expected.shape(), "sdc case {idx}: shape");
        for ((i, gv), ev) in got.indexed_iter().zip(expected.iter()) {
            assert_eq!(
                gv.to_bits(),
                ev.to_bits(),
                "sdc case {idx} at {i:?}: rust=0x{:016x} ({gv}) py=0x{:016x} ({ev})",
                gv.to_bits(),
                ev.to_bits()
            );
        }
    }
}

/// The SDC stacking in isolation (matches the Python `sdc_oracle`): from an MFCC
/// (statics), form the n=3 delta then place block kk at time offset +10-3*kk,
/// zero-padded. Output is `T x 7*nb_dct`. This is the SDC-block portion of the
/// legacy applyDCT Branch A (without the leading statics or the ignoreFirst clobber).
fn sdc_from_mfcc(mfcc: &Array2<f64>, nb_dct: usize) -> Array2<f64> {
    let t = mfcc.nrows();
    let k = 7usize;
    let d = regression_deltas(mfcc, 3);
    let mut out = Array2::<f64>::zeros((t, k * nb_dct));
    for kk in 0..k {
        for tt in 0..t {
            let src = tt as isize + 10 - 3 * kk as isize;
            if src >= 0 && (src as usize) < t {
                for c in 0..nb_dct {
                    out[[tt, kk * nb_dct + c]] = d[[src as usize, c]];
                }
            }
        }
    }
    out
}
