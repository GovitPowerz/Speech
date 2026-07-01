//! Phase 1 Task 4: bit-exact golden tests for the GFFT port (features/fft.rs).
//!
//! Goldens are dumped by the oracle harness running the REAL legacy fft.hpp
//! (Taylor-seed twiddles + Numerical-Recipes running recurrence). Bit-exactness
//! (to_bits) is the contract. The impulse/Parseval tests are tolerance-based
//! SANITY checks only, NOT parity assertions.

mod common;

use speech::features::fft::{Gfft, sin_series};

/// Bit-exact parity against the legacy GFFT dumps for N in {4, 8, 256}.
#[test]
fn gfft_matches_oracle_bitexact() {
    for p in [2u32, 3, 8] {
        let n = 1usize << p;
        let mut data = common::load_bin(&format!("fft_in_{n}.bin")).row(0).to_vec();
        Gfft::new(p).fft(&mut data);
        let expect = common::load_bin(&format!("fft_out_{n}.bin"));
        for (i, v) in data.iter().enumerate() {
            assert_eq!(
                v.to_bits(),
                expect[[0, i]].to_bits(),
                "N={n} idx={i}: got 0x{:016x} ({v}) want 0x{:016x} ({})",
                v.to_bits(),
                expect[[0, i]].to_bits(),
                expect[[0, i]]
            );
        }
    }
}

/// Seed test: the recursive `sin_series` must equal a second, independent coding
/// of the SAME truncated Taylor series (S(2,34) for x = A*PI/B). Two independent
/// codings of identical fp operations in identical order must agree bit-for-bit.
#[test]
fn sin_series_matches_independent_coding() {
    // Cross-check sin_series(B, A) for a handful of (B, A) the twiddle seeds use.
    for (b, a) in [(4u64, 1u64), (256, 1), (256, 2), (8, 1), (8, 2), (4, 2)] {
        let x = (a as f64 * std::f64::consts::PI) / b as f64;
        // Independent coding: Horner-fold the S(2,34) recursion innermost-first.
        // S(M) = 1 - ((x*x)/M)/(M+1)*S(M+2), base S(34)=1; Sin = x*S(2).
        // Loop M = 32, 30, ..., 2 (16 iterations); acc starts at the base 1.0.
        let mut acc = 1.0_f64;
        let mut m = 32u64;
        loop {
            acc = 1.0 - ((x * x) / m as f64) / (m as f64 + 1.0) * acc;
            if m == 2 {
                break;
            }
            m -= 2;
        }
        let independent = x * acc;
        assert_eq!(
            sin_series(b, a).to_bits(),
            independent.to_bits(),
            "sin_series({b},{a}): recursive 0x{:016x} != product 0x{:016x}",
            sin_series(b, a).to_bits(),
            independent.to_bits()
        );
    }
}

/// SANITY ONLY (tolerance 1e-9): an impulse transforms to a flat unit spectrum.
#[test]
fn impulse_flat_spectrum_sanity() {
    for p in [2u32, 3, 8] {
        let n = 1usize << p;
        let mut data = vec![0.0_f64; 2 * n];
        data[0] = 1.0; // real impulse at index 0
        Gfft::new(p).fft(&mut data);
        for k in 0..n {
            assert!((data[2 * k] - 1.0).abs() < 1e-9, "N={n} re[{k}]");
            assert!(data[2 * k + 1].abs() < 1e-9, "N={n} im[{k}]");
        }
    }
}

/// SANITY ONLY (tolerance 1e-9): Parseval, unnormalized -> sum|X|^2 = N*sum|x|^2.
#[test]
fn parseval_sanity() {
    for p in [2u32, 3, 8] {
        let n = 1usize << p;
        let mut data: Vec<f64> = (0..2 * n)
            .map(|i| ((i * 7 + 3) % 17) as f64 / 17.0)
            .collect();
        let time_energy: f64 = data.iter().map(|v| v * v).sum();
        Gfft::new(p).fft(&mut data);
        let freq_energy: f64 = data.iter().map(|v| v * v).sum();
        let rel = (freq_energy - n as f64 * time_energy).abs() / (n as f64 * time_energy);
        assert!(rel < 1e-9, "N={n} Parseval rel err {rel}");
    }
}
