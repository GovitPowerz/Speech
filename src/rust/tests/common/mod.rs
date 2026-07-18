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

/// Absolute path to a file under `tests/reference_data/phase2b/`.
pub fn fixture_phase2b(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase2b")
        .join(name)
}

/// Load a phase2b `.bin` fixture (column-major f64) into an `Array2<f64>` (rows x cols).
pub fn load_bin_phase2b(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture_phase2b(name)).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

/// Absolute path to a file under `tests/reference_data/phase3/`.
pub fn fixture_phase3(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase3")
        .join(name)
}

/// Load a phase3 `.bin` fixture (column-major f64) into an `Array2<f64>` (rows x cols).
pub fn load_bin_phase3(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture_phase3(name)).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

/// Absolute path to a file under `tests/reference_data/phase4a/`.
pub fn fixture_phase4a(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4a")
        .join(name)
}

/// Load a phase4a `.bin` fixture (column-major f64) into an `Array2<f64>` (rows x cols).
pub fn load_bin_phase4a(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture_phase4a(name)).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

/// Absolute path to a file under `tests/reference_data/phase4b/`.
pub fn fixture_phase4b(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4b")
        .join(name)
}

/// Load a phase4b `.bin` fixture (column-major f64) into an `Array2<f64>` (rows x cols).
pub fn load_bin_phase4b(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture_phase4b(name)).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

/// Absolute path to a file under `tests/reference_data/phase6/`.
pub fn fixture_phase6(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase6")
        .join(name)
}

/// Absolute path to the licensed LRE03/07 corpus root (`data/LRE03-LRE07` at the repo
/// root, i.e. `CARGO_MANIFEST_DIR/../..`). The corpus is 27 GB, gitignored, and licensed
/// -- present locally for Phase 6 training/validation, absent in CI. Phase-6 corpus-gated
/// tests call [`corpus_root_or_skip`] to opt out cleanly when it is missing.
pub fn corpus_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/LRE03-LRE07")
}

/// Returns the corpus root if it exists, else `None` after printing a skip notice.
/// Pattern: `let Some(root) = corpus_root_or_skip() else { return; };` at the top of a
/// corpus-gated test -- the test no-ops (passes) when the licensed corpus is absent.
pub fn corpus_root_or_skip() -> Option<PathBuf> {
    let root = corpus_root();
    if root.is_dir() {
        Some(root)
    } else {
        eprintln!(
            "SKIP: licensed LRE03/07 corpus absent at {} (set up data/LRE03-LRE07 to run)",
            root.display()
        );
        None
    }
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

// === VRCTS structural + value-level comparison (S9.3) ========================
//
// Extract `key="value"` verbatim from a single XML-ish line (no quote-escaping,
// mirroring the legacy `iof::fmtr` scanf-style parser and `segmentation_io`'s
// private `extract_attr`).
fn vrcts_extract_attr(line: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Attribute names whose VALUE is a `%.2f`/`%.4f`-formatted float derived from the
/// libm-traversing chain (posteriors -> interpolated boundary times -> formatting):
/// these are compared via a display-quantum-aware bound instead of byte-for-byte.
const VRCTS_FLOAT_ATTRS: &[&str] = &["sigdur", "spdur", "dur", "stime", "etime"];

/// A `%.Nf`-formatted value can print differently even when the underlying f64s
/// are within the hybrid ULP/absolute bound: an infinitesimal (libm-noise-scale)
/// difference straddling a rounding boundary (e.g. `x.xxxx5`) flips the LAST
/// PRINTED DIGIT, a full `10^-N` step in the parsed-back value -- nine-plus
/// orders of magnitude bigger than the raw hybrid bound (`assert_oracle_eq_f64`'s
/// `512*eps*scale`). So a pair of parsed VRCTS float attrs passes when EITHER the
/// raw hybrid bound holds (the common case, no boundary straddle) OR the two
/// values are within one display quantum of each other (`stime`/`etime` -> 4
/// decimals -> `1e-4`; `sigdur`/`spdur`/`dur` -> 2 decimals -> `1e-2`) -- the
/// boundary-straddle case, which is bounded (not unbounded) precisely BECAUSE the
/// two runs both round the SAME underlying (hybrid-bound-close) value at the SAME
/// precision. The quantum is inferred from the printed string's decimal-digit
/// count so this stays correct if a future attr uses a different precision.
fn vrcts_float_value_ok(got: &str, want: &str) -> bool {
    let gf: f64 = match got.parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let wf: f64 = match want.parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let decimals = want.rsplit_once('.').map_or(0, |(_, frac)| frac.len());
    let quantum = 10f64.powi(-(decimals as i32));
    let abs_tol = HYBRID_ABS_FACTOR * f64::EPSILON * wf.abs().max(1.0);
    hybrid_ok_f64(gf, wf, 4, abs_tol) || (gf - wf).abs() <= quantum
}

/// Compare a VRCTS XML document (`to_vrcts_string` output) against a committed
/// fixture dump per spec S9.3: byte-exact on the oracle env (where the pin is
/// load-bearing), value-level re-parse elsewhere, because `%.4f`/`%.2f` rounding
/// of a libm-derived float can flip a final digit under a different libm even
/// when the underlying value is correct to the hybrid ULP/absolute bound. The
/// Ulp arm still fails on any STRUCTURAL drift -- segment count, class/tag
/// names, attribute names or ordering -- by walking both documents line by
/// line and requiring every non-float-attribute line to match byte-for-byte;
/// only the known float attributes (`sigdur`, `spdur`, `dur`, `stime`, `etime`)
/// go through [`vrcts_float_value_ok`].
pub fn assert_vrcts_eq(got: &str, expected: &str, label: &str) {
    match oracle_mode() {
        OracleMode::Strict => assert_eq!(got, expected, "{label}: byte mismatch"),
        OracleMode::Ulp(_) => {
            let got_lines: Vec<&str> = got.lines().collect();
            let want_lines: Vec<&str> = expected.lines().collect();
            assert_eq!(
                got_lines.len(),
                want_lines.len(),
                "{label}: line count mismatch (structural drift)"
            );
            for (i, (gl, wl)) in got_lines.iter().zip(want_lines.iter()).enumerate() {
                if *gl == *wl {
                    continue;
                }
                // Lines differ: must be explained ENTIRELY by float-attribute value
                // drift, not by structure. Strip each known float attr's value out
                // of both lines (replacing with a placeholder) and require the
                // residue to match; then compare the stripped values numerically.
                let mut g_residue = gl.to_string();
                let mut w_residue = wl.to_string();
                let mut any_float_attr = false;
                for attr in VRCTS_FLOAT_ATTRS {
                    if let (Some(gv), Some(wv)) =
                        (vrcts_extract_attr(gl, attr), vrcts_extract_attr(wl, attr))
                    {
                        any_float_attr = true;
                        assert!(
                            vrcts_float_value_ok(&gv, &wv),
                            "{label}: line {i} attr {attr}: got={gv} want={wv} outside hybrid bound and outside one display quantum"
                        );
                        g_residue = g_residue.replacen(&format!("{attr}=\"{gv}\""), "", 1);
                        w_residue = w_residue.replacen(&format!("{attr}=\"{wv}\""), "", 1);
                    }
                }
                assert!(
                    any_float_attr,
                    "{label}: line {i} differs with no recognized float attribute (structural drift):\n  got: {gl}\n  want: {wl}"
                );
                assert_eq!(
                    g_residue, w_residue,
                    "{label}: line {i} residue (non-float content) mismatch:\n  got: {gl}\n  want: {wl}"
                );
            }
        }
    }
}

// ============================================================================
// Minimal MAT v5 reader (Phase 4a; the engine's `MatWriter` has no reader).
// Uncompressed v5 only: 128-byte header then a sequence of miMATRIX elements.
// Shared by the phase4a integration tests, which all read back a `.mat` the
// engine wrote and compare it against a converted fixture.
// ============================================================================

fn u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn i32_le(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn f64_le(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// Walk the miMATRIX elements, returning `(name, rows, cols, col_major_data)` per
/// variable in file order.
pub fn mat_vars(bytes: &[u8]) -> Vec<(String, usize, usize, Vec<f64>)> {
    let mut out = Vec::new();
    let mut off = 128;
    while off + 8 <= bytes.len() {
        let tag = u32_le(bytes, off);
        let size = u32_le(bytes, off + 4) as usize;
        if tag != 14 {
            break;
        }
        let mut p = off + 8;
        // array flags sub-element (miUINT32, 8 payload): skip tag+size+8.
        p += 8 + 8;
        // dims sub-element (miINT32): tag+size then two i32.
        let dims_size = u32_le(bytes, p + 4) as usize;
        let rows = i32_le(bytes, p + 8) as usize;
        let cols = i32_le(bytes, p + 12) as usize;
        p += 8 + pad8(dims_size);
        // name sub-element (miINT8, long form): tag+size then the padded body.
        let name_size = u32_le(bytes, p + 4) as usize;
        let name = String::from_utf8(bytes[p + 8..p + 8 + name_size].to_vec()).unwrap();
        p += 8 + pad8(name_size);
        // data sub-element (miDOUBLE): tag+size then the doubles.
        let data_size = u32_le(bytes, p + 4) as usize;
        let n = data_size / 8;
        let mut data = Vec::with_capacity(n);
        for k in 0..n {
            data.push(f64_le(bytes, p + 8 + k * 8));
        }
        out.push((name, rows, cols, data));
        off += 8 + size;
    }
    out
}

/// Read a variable's matrix back out of a MAT v5 file the engine wrote. The
/// writer stores column-major; this rebuilds a row-major `Array2`.
pub fn mat_var_matrix(bytes: &[u8], name: &str) -> Option<Array2<f64>> {
    mat_vars(bytes)
        .into_iter()
        .find(|(n, _, _, _)| n == name)
        .map(|(_, r, c, data)| Array2::from_shape_fn((r, c), |(i, j)| data[j * r + i]))
}

/// `mat_var_matrix`, panicking with the variable name if absent (the common
/// call shape across the phase4a e2e/golden tests).
pub fn read_mat_var(bytes: &[u8], name: &str) -> Array2<f64> {
    mat_var_matrix(bytes, name).unwrap_or_else(|| panic!("variable {name} not in .mat"))
}

pub fn mat_var_order(bytes: &[u8]) -> Vec<String> {
    mat_vars(bytes).into_iter().map(|(n, _, _, _)| n).collect()
}

pub fn mat_var_dims(bytes: &[u8], name: &str) -> Option<(usize, usize)> {
    mat_vars(bytes)
        .into_iter()
        .find(|(n, _, _, _)| n == name)
        .map(|(_, r, c, _)| (r, c))
}

/// Element compare: `strict_cols` bit-exact (id/count columns), `masked_cols`
/// skipped (e.g. timing), the rest canary-gated via [`assert_oracle_eq_f64`].
pub fn assert_matrix(
    got: &Array2<f64>,
    want: &Array2<f64>,
    strict_cols: &[usize],
    masked_cols: &[usize],
    label: &str,
) {
    assert_eq!(got.dim(), want.dim(), "{label}: shape mismatch");
    let (rows, cols) = got.dim();
    for r in 0..rows {
        for c in 0..cols {
            if masked_cols.contains(&c) {
                continue;
            }
            let g = got[[r, c]];
            let w = want[[r, c]];
            if strict_cols.contains(&c) {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{label}[{r},{c}] strict: got {g} want {w}"
                );
            } else {
                assert_oracle_eq_f64(g, w, &format!("{label}[{r},{c}]"));
            }
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

    const VRCTS_SAMPLE: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<AudioDoc name=\"x\" path=\"/tmp/x.wav\">\n<ProcList>\n<Proc name=\"vrcts_part\" version=\"1.3\"/>\n</ProcList>\n<ChannelList>\n<Channel num=\"1\" sigdur=\"2.00\" spdur=\"1.68\"/>\n</ChannelList>\n<SpeakerList>\n<Speaker ch=\"1\" dur=\"1.68\" gender=\"1\" spkid=\"1\"/>\n</SpeakerList>\n<SegmentList>\n<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"0.0000\" etime=\"1.6773\" spkid=\"1\"/>\n</SegmentList>\n</AudioDoc>\n";

    /// Runs the SAME line-walk logic as `assert_vrcts_eq`'s `Ulp` arm (mirrored here
    /// so tests are independent of this process's actual `oracle_mode()`, an
    /// env-gated `OnceLock` fixed for the whole binary -- normally `Strict` on the
    /// oracle env these tests run on), returning `Err` instead of panicking.
    fn vrcts_ulp_arm_ok(got: &str, expected: &str) -> Result<(), String> {
        let got_lines: Vec<&str> = got.lines().collect();
        let want_lines: Vec<&str> = expected.lines().collect();
        if got_lines.len() != want_lines.len() {
            return Err("line count mismatch (structural drift)".to_string());
        }
        for (i, (gl, wl)) in got_lines.iter().zip(want_lines.iter()).enumerate() {
            if *gl == *wl {
                continue;
            }
            let mut g_residue = gl.to_string();
            let mut w_residue = wl.to_string();
            let mut any_float_attr = false;
            for attr in VRCTS_FLOAT_ATTRS {
                if let (Some(gv), Some(wv)) =
                    (vrcts_extract_attr(gl, attr), vrcts_extract_attr(wl, attr))
                {
                    any_float_attr = true;
                    if !vrcts_float_value_ok(&gv, &wv) {
                        return Err(format!("line {i} attr {attr}: {gv} vs {wv} out of bound"));
                    }
                    g_residue = g_residue.replacen(&format!("{attr}=\"{gv}\""), "", 1);
                    w_residue = w_residue.replacen(&format!("{attr}=\"{wv}\""), "", 1);
                }
            }
            if !any_float_attr {
                return Err(format!(
                    "line {i} structural drift (no recognized float attribute)"
                ));
            }
            if g_residue != w_residue {
                return Err(format!(
                    "line {i} residue mismatch (structural drift): {g_residue:?} vs {w_residue:?}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn vrcts_eq_passes_identical() {
        assert_vrcts_eq(VRCTS_SAMPLE, VRCTS_SAMPLE, "identical");
        assert!(vrcts_ulp_arm_ok(VRCTS_SAMPLE, VRCTS_SAMPLE).is_ok());
    }

    #[test]
    fn vrcts_eq_ulp_arm_tolerates_hybrid_bound_scale_drift() {
        // A tiny (raw-ULP-scale) drift on a float attribute -- a 1-ULP libm
        // difference in the upstream posterior propagating to a few-ULP difference
        // in the pre-formatting time value, with NO rounding-boundary straddle --
        // must pass via the hybrid-bound arm of vrcts_float_value_ok.
        let drifted = VRCTS_SAMPLE.replace("etime=\"1.6773\"", "etime=\"1.67730000000001\"");
        assert!(
            vrcts_ulp_arm_ok(&drifted, VRCTS_SAMPLE).is_ok(),
            "a hybrid-bound-scale drift must pass"
        );
    }

    #[test]
    fn vrcts_eq_ulp_arm_tolerates_display_quantum_boundary_straddle() {
        // The S9.3 scenario this fix targets: two runs' underlying f64s differ only
        // by libm noise but straddle a %.4f rounding boundary, so the PRINTED digit
        // flips by a full 1e-4 step (e.g. 1.6773 vs 1.6774). This must pass via the
        // display-quantum arm of vrcts_float_value_ok, not the raw hybrid bound.
        let drifted = VRCTS_SAMPLE.replace("etime=\"1.6773\"", "etime=\"1.6774\"");
        assert!(
            vrcts_ulp_arm_ok(&drifted, VRCTS_SAMPLE).is_ok(),
            "a one-display-quantum drift (rounding boundary straddle) must pass"
        );
    }

    #[test]
    fn vrcts_eq_ulp_arm_rejects_bug_scale_drift() {
        // A drift several quanta wide (bug-scale, not a single rounding-boundary
        // flip) must still fail -- the display-quantum arm only forgives ONE step.
        let drifted = VRCTS_SAMPLE.replace("etime=\"1.6773\"", "etime=\"1.6790\"");
        assert!(
            vrcts_ulp_arm_ok(&drifted, VRCTS_SAMPLE).is_err(),
            "a multi-quantum drift must NOT be masked"
        );
    }

    #[test]
    fn vrcts_eq_ulp_arm_rejects_structural_drift() {
        // A mutated segment/tag name is NOT explainable by float-attribute value
        // drift, so the Ulp arm must still fail on it (the S9.3 requirement: fail
        // on genuine structural drift, not just on times).
        let mutated = VRCTS_SAMPLE.replace("SpeechSegment", "NoiseSegment");
        assert!(
            vrcts_ulp_arm_ok(&mutated, VRCTS_SAMPLE).is_err(),
            "a mutated tag name must NOT be masked by the float-attribute re-parse"
        );
    }

    #[test]
    fn vrcts_eq_ulp_arm_rejects_extra_segment() {
        // A mutated string with an inserted extra segment line changes the line
        // count -- must fail on the line-count check, not silently pass.
        let mutated = VRCTS_SAMPLE.replacen(
            "<SegmentList>\n",
            "<SegmentList>\n<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"1.7000\" etime=\"1.9000\" spkid=\"1\"/>\n",
            1,
        );
        assert!(
            vrcts_ulp_arm_ok(&mutated, VRCTS_SAMPLE).is_err(),
            "an inserted extra segment line must fail on line-count structural drift"
        );
    }
}
