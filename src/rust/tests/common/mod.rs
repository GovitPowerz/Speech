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

/// Load a phase1 `.bin` fixture (column-major f64) into an `Array2<f64>` (rows x cols).
pub fn load_bin(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture(name)).unwrap();
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
                        "SPEECH_ORACLE_LIBM=ulp: golden asserts run in <=4 ULP mode (forced)"
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
                "oracle libm canaries differ from this platform's libm; golden asserts run in <=4 ULP mode"
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

/// Compare a transcendental-dependent golden against its oracle dump: bit-exact on
/// the oracle env, `<= n` ULP elsewhere. Use ONLY where the Rust-side computation
/// reaching the assert calls cos/ln/exp; portable goldens keep `assert_bits_eq`.
pub fn assert_oracle_eq(a: &Array2<f64>, b: &Array2<f64>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape mismatch");
    match oracle_mode() {
        OracleMode::Strict => assert_bits_eq(a, b, label),
        OracleMode::Ulp(n) => {
            for ((idx, av), bv) in a.indexed_iter().zip(b.iter()) {
                let d = ulp_distance_f64(*av, *bv);
                assert!(
                    d.is_some_and(|d| d <= *n),
                    "{label}: at {idx:?} ULP={} > {n}: a=0x{:016x} ({av}) b=0x{:016x} ({bv})",
                    d.map_or_else(|| "NaN/inf/sign".to_string(), |d| d.to_string()),
                    av.to_bits(),
                    bv.to_bits()
                );
            }
        }
    }
}

/// Scalar f64 variant of `assert_oracle_eq`.
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
            let d = ulp_distance_f64(a, b);
            assert!(
                d.is_some_and(|d| d <= *n),
                "{label}: ULP={} > {n}: a=0x{:016x} ({a}) b=0x{:016x} ({b})",
                d.map_or_else(|| "NaN/inf/sign".to_string(), |d| d.to_string()),
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}

/// Scalar f32 variant: bit-exact on the oracle env, `<= 2` f32 ULP elsewhere. For
/// the fmath (f32 table) goldens whose divergence lives in f32 precision.
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
            let d = ulp_distance_f32(a, b);
            assert!(
                d.is_some_and(|d| d <= 2),
                "{label}: f32 ULP={} > 2: a=0x{:08x} ({a}) b=0x{:08x} ({b})",
                d.map_or_else(|| "NaN/inf/sign".to_string(), |d| d.to_string()),
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}
