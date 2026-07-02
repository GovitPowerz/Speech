//! Shared test infrastructure for golden-fixture tests (Phase 1+).
//!
//! `mod common` is compiled into every integration-test binary, but not every
//! binary uses every helper (e.g. the FFT golden uses only `load_bin`), so allow
//! dead code here rather than forcing each test to touch all helpers.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

use ndarray::{Array2, ShapeBuilder};

/// Absolute path to a file under `tests/reference_data/phase1/`.
pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase1")
        .join(name)
}

/// Absolute path to a file under `tests/reference_data/phase2/`.
pub fn fixture_phase2(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase2")
        .join(name)
}

/// Load a phase1 `.bin` fixture (column-major f64) into an `Array2<f64>` (rows x cols).
pub fn load_bin(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture(name)).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

/// Load a phase2 `.bin` fixture (column-major f64) into an `Array2<f64>` (rows x cols).
pub fn load_bin_phase2(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture_phase2(name)).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

/// Elementwise bit-exact comparison; reports the first mismatch index + hex bits.
/// Used for PORTABLE goldens (pure arithmetic): they stay bit-exact on every libm.
pub fn assert_bits_eq(a: &Array2<f64>, b: &Array2<f64>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape mismatch");
    for ((idx, av), bv) in a.indexed_iter().zip(b.iter()) {
        assert!(
            av.to_bits() == bv.to_bits(),
            "{label}: mismatch at {idx:?}: a=0x{:016x} ({av}) b=0x{:016x} ({bv})",
            av.to_bits(),
            bv.to_bits()
        );
    }
}

// === Canary-gated oracle mode (CI libm portability) =========================
//
// Phase 1 goldens were dumped on the oracle env (Apple libm). Transcendental libm
// calls (cos/ln/exp) round some args 1 ULP differently under glibc, which the
// downstream arithmetic amplifies to a few ULP. Bit-exactness stays the contract
// ON THE ORACLE ENV; elsewhere the transcendental-dependent goldens assert a tight
// ULP bound, gated by libm canaries recorded in the fixtures.

/// How the transcendental-dependent goldens compare against the oracle dumps.
pub enum OracleMode {
    /// This platform's libm matches the oracle canaries bit-for-bit.
    Strict,
    /// This platform's libm differs; allow up to `n` ULP on those goldens.
    Ulp(u64),
}

/// Recompute the harness `libm_canaries.bin` outputs with THIS platform's libm and
/// decide the golden mode once. The arg list + per-column function assignment is
/// hardcoded to match `tools/oracle_harness/main.cpp` (10 cos, 5 log, 3 exp);
/// we do NOT parse the manifest. `SPEECH_ORACLE_LIBM=strict|ulp` forces a path.
pub fn oracle_mode() -> &'static OracleMode {
    static MODE: OnceLock<OracleMode> = OnceLock::new();
    MODE.get_or_init(|| {
        if let Ok(force) = std::env::var("SPEECH_ORACLE_LIBM") {
            match force.as_str() {
                "strict" => return OracleMode::Strict,
                "ulp" => {
                    eprintln!(
                        "SPEECH_ORACLE_LIBM=ulp: golden asserts run in hybrid mode: <=4 ULP or 512*eps*fixture-scale absolute (forced)"
                    );
                    return OracleMode::Ulp(4);
                }
                other => panic!("SPEECH_ORACLE_LIBM must be strict|ulp, got {other:?}"),
            }
        }

        let dump = load_bin("libm_canaries.bin"); // 2 x N: row0 input, row1 output
        let n = dump.ncols();
        let mut all_match = true;
        for k in 0..n {
            let x = dump[[0, k]];
            let want = dump[[1, k]];
            // Fixed layout mirroring main.cpp: [0,10) cos, [10,15) log, [15,18) exp.
            let got = if k < 10 {
                x.cos()
            } else if k < 15 {
                x.ln()
            } else {
                x.exp()
            };
            if got.to_bits() != want.to_bits() {
                all_match = false;
                break;
            }
        }
        if all_match {
            OracleMode::Strict
        } else {
            eprintln!(
                "oracle libm canaries differ from this platform's libm; golden asserts run in hybrid mode: <=4 ULP or 512*eps*fixture-scale absolute"
            );
            OracleMode::Ulp(4)
        }
    })
}

/// ULP distance between two finite same-sign f64s via i64 bit-space. Returns `None`
/// on NaN/inf or sign mismatch (which always fail the assert).
fn ulp_distance_f64(a: f64, b: f64) -> Option<u64> {
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return None;
    }
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    Some(ai.abs_diff(bi))
}

/// ULP distance for f32 (the fmath table is f32-valued).
fn ulp_distance_f32(a: f32, b: f32) -> Option<u32> {
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return None;
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    Some(ai.abs_diff(bi))
}

// --- Hybrid ULP-or-scaled-absolute check (Ulp mode only) --------------------
//
// Elementwise ULP alone is the wrong metric under cancellation. A 1-ULP libm
// difference (cos/ln/exp: Apple libm vs glibc) is an ABSOLUTE perturbation of
// ~eps * |libm output|, and the downstream arithmetic carries that absolute
// error into results of any magnitude. Where the result is small, the same
// absolute error is many ULP of the result. Measured on CI (glibc):
// win_hann_257 at (0,239), where hann[239] = 0.5 - 0.5*cos(2*PI/256*239)
// ~= 0.0429, failed at ULP=8 from a single 1-ULP cos difference at
// cos ~ 0.914 (0.5*ulp(0.914)/ulp(0.0429) ~= 8). Near the window's zeros the
// amplification is unbounded in ULP terms, so bumping the ULP bound is
// whack-a-mole. Instead, a pair passes when EITHER
//   (a) ULP distance <= bound (4 f64 / 2 f32) - the tight check, still doing
//       all the work for well-scaled values, OR
//   (b) |a - b| <= 512 * EPSILON * scale, with scale = max |expected| over
//       the fixture matrix (scalar variants: max(|expected|, 1.0)). The
//       tolerance is absolute-scaled to the fixture magnitude because libm
//       perturbations propagate absolutely, NOT relative to each (possibly
//       tiny) output element.
// 512 * f64::EPSILON * scale ~= 1.1e-13 * scale stays 9+ orders of magnitude
// below porting-bug-scale errors (>= 1e-3 relative); the f32 arm
// (512 * f32::EPSILON ~= 6.1e-5) stays over an order of magnitude below.
// NaN/inf always fail. Finite sign-mismatched pairs (e.g. -1e-18 vs +1e-18,
// legitimate cancellation across zero) can pass only through arm (b):
// `ulp_distance_*` returns `None` on sign mismatch, so arm (a) never fires.
const HYBRID_ABS_FACTOR: f64 = 512.0;

fn hybrid_ok_f64(a: f64, b: f64, ulp_bound: u64, abs_tol: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    ulp_distance_f64(a, b).is_some_and(|d| d <= ulp_bound) || (a - b).abs() <= abs_tol
}

fn hybrid_ok_f32(a: f32, b: f32, ulp_bound: u32, abs_tol: f32) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    ulp_distance_f32(a, b).is_some_and(|d| d <= ulp_bound) || (a - b).abs() <= abs_tol
}

/// Compare a transcendental-dependent golden against its oracle dump: bit-exact on
/// the oracle env, hybrid (`<= n` ULP or `512*eps*scale` absolute, scale = max
/// |expected|) elsewhere. Use ONLY where the Rust-side computation reaching the
/// assert calls cos/ln/exp; portable goldens keep `assert_bits_eq`.
pub fn assert_oracle_eq(a: &Array2<f64>, b: &Array2<f64>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape mismatch");
    match oracle_mode() {
        OracleMode::Strict => assert_bits_eq(a, b, label),
        OracleMode::Ulp(n) => {
            // `b` is the expected fixture (call convention: (&got, &want, label)).
            let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let abs_tol = HYBRID_ABS_FACTOR * f64::EPSILON * scale;
            for ((idx, av), bv) in a.indexed_iter().zip(b.iter()) {
                assert!(
                    hybrid_ok_f64(*av, *bv, *n, abs_tol),
                    "{label}: at {idx:?} ULP={} > {n} and |diff|={:e} > abs_tol={abs_tol:e}: a=0x{:016x} ({av}) b=0x{:016x} ({bv})",
                    ulp_distance_f64(*av, *bv)
                        .map_or_else(|| "NaN/inf/sign".to_string(), |d| d.to_string()),
                    (*av - *bv).abs(),
                    av.to_bits(),
                    bv.to_bits()
                );
            }
        }
    }
}

/// Scalar f64 variant of `assert_oracle_eq`; `b` is the expected value, the
/// absolute arm uses scale = max(|expected|, 1.0).
pub fn assert_oracle_eq_f64(a: f64, b: f64, label: &str) {
    match oracle_mode() {
        OracleMode::Strict => assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: a=0x{:016x} ({a}) b=0x{:016x} ({b})",
            a.to_bits(),
            b.to_bits()
        ),
        OracleMode::Ulp(n) => {
            let abs_tol = HYBRID_ABS_FACTOR * f64::EPSILON * b.abs().max(1.0);
            assert!(
                hybrid_ok_f64(a, b, *n, abs_tol),
                "{label}: ULP={} > {n} and |diff|={:e} > abs_tol={abs_tol:e}: a=0x{:016x} ({a}) b=0x{:016x} ({b})",
                ulp_distance_f64(a, b)
                    .map_or_else(|| "NaN/inf/sign".to_string(), |d| d.to_string()),
                (a - b).abs(),
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}

/// Scalar f32 variant: bit-exact on the oracle env, hybrid (`<= 2` f32 ULP or
/// `512*eps*max(|expected|,1)` absolute) elsewhere. For the fmath (f32 table)
/// goldens whose divergence lives in f32 precision.
pub fn assert_oracle_eq_f32(a: f32, b: f32, label: &str) {
    match oracle_mode() {
        OracleMode::Strict => assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: a=0x{:08x} ({a}) b=0x{:08x} ({b})",
            a.to_bits(),
            b.to_bits()
        ),
        OracleMode::Ulp(_) => {
            let abs_tol = (HYBRID_ABS_FACTOR as f32) * f32::EPSILON * b.abs().max(1.0);
            assert!(
                hybrid_ok_f32(a, b, 2, abs_tol),
                "{label}: f32 ULP={} > 2 and |diff|={:e} > abs_tol={abs_tol:e}: a=0x{:08x} ({a}) b=0x{:08x} ({b})",
                ulp_distance_f32(a, b)
                    .map_or_else(|| "NaN/inf/sign".to_string(), |d| d.to_string()),
                (a - b).abs(),
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}

#[cfg(test)]
mod hybrid_comparator_tests {
    use super::*;

    #[test]
    fn hybrid_comparator_passes_measured_ci_libm_diff() {
        // The exact pair CI (glibc) measured at win_hann_257 (0,239): ULP=8
        // from a single 1-ULP cos difference. Must pass via arm (b) at scale 1.
        let a = f64::from_bits(0x3fa5f6597591b640);
        let b = f64::from_bits(0x3fa5f6597591b648);
        let abs_tol = HYBRID_ABS_FACTOR * f64::EPSILON * 1.0;
        assert_eq!(ulp_distance_f64(a, b), Some(8));
        assert!(hybrid_ok_f64(a, b, 4, abs_tol));
        // Finite sign flip across zero on a tiny value: arm (b) only, passes.
        assert!(hybrid_ok_f64(-1e-18, 1e-18, 4, abs_tol));
    }

    #[test]
    fn hybrid_comparator_rejects_porting_bug_scale_error() {
        // Bug-scale error at hann-window magnitude (~2e-4 absolute, ~0.5%
        // relative): neither arm may rescue it.
        let abs_tol = HYBRID_ABS_FACTOR * f64::EPSILON * 1.0;
        assert!(!hybrid_ok_f64(0.0429, 0.0430, 4, abs_tol));
        // NaN/inf always fail, both arms.
        assert!(!hybrid_ok_f64(f64::NAN, f64::NAN, 4, abs_tol));
        assert!(!hybrid_ok_f64(f64::INFINITY, f64::INFINITY, 4, abs_tol));
        // f32 arm: same bug-scale rejection.
        let abs_tol32 = (HYBRID_ABS_FACTOR as f32) * f32::EPSILON * 1.0;
        assert!(!hybrid_ok_f32(0.0429_f32, 0.0430_f32, 2, abs_tol32));
    }
}
