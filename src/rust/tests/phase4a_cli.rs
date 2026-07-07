//! Phase 4a Task 10: the full legacy CLI (`parse_cli`) + binary `main` wiring.
//!
//! Ported from legacy C++: FastSpeechProcessing.cpp `main` (:31-87).
//!
//! Parse-level tests exercise `speech::cli::parse_cli` directly (no engine run).
//! `binary_end_to_end` shells out to the ACTUAL compiled `speech` binary
//! (`CARGO_BIN_EXE_speech`) and replays the Task 8 tier-1 solo golden
//! (`solo_tdc_matches` in `phase4a_tier1_e2e.rs`) through the CLI surface, to pin
//! `main.rs`'s parse -> CorpusProcessor::new -> run wiring end to end.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use ndarray::Array2;

use common::{assert_oracle_eq_f64, load_bin_phase4a};
use speech::cli::{ModeKind, parse_cli};

/// `set_current_dir` is process-global; serialize against other tests in this
/// binary (mirrors `phase4a_tier1_e2e.rs`'s `CWD_LOCK`, but this file's tests
/// spawn a subprocess so no chdir is actually needed here -- kept for symmetry
/// / in case a future test in this file needs it).
static CWD_LOCK: Mutex<()> = Mutex::new(());

fn args(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

// === Parse-level tests ========================================================

#[test]
fn override_applied() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = dir.path().join("base.config");
    std::fs::write(&cfg_path, "numOuterThreads 1\nAlgo_choice 1\n").unwrap();

    let inv = parse_cli(&args(&[
        "-s",
        "--numOuterThreads=4",
        cfg_path.to_str().unwrap(),
    ]))
    .unwrap();

    assert_eq!(inv.mode.kind, ModeKind::Solo);
    assert_eq!(inv.configs.len(), 1);
    assert_eq!(inv.configs[0].get("numOuterThreads").unwrap(), "4");
    // Untouched key survives.
    assert_eq!(inv.configs[0].get("Algo_choice").unwrap(), "1");
}

#[test]
fn override_empty_removes_key() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = dir.path().join("base.config");
    std::fs::write(&cfg_path, "numOuterThreads 1\nAlgo_choice 1\n").unwrap();

    let inv = parse_cli(&args(&["-s", "--Algo_choice=", cfg_path.to_str().unwrap()])).unwrap();

    assert_eq!(inv.configs.len(), 1);
    assert!(
        !inv.configs[0].contains_key("Algo_choice"),
        "empty-value override must remove the key"
    );
    assert_eq!(inv.configs[0].get("numOuterThreads").unwrap(), "1");
}

#[test]
fn multi_collects_all_configs() {
    let dir = tempfile::tempdir().unwrap();
    let c1 = dir.path().join("a.config");
    let c2 = dir.path().join("b.config");
    let c3 = dir.path().join("c.config");
    std::fs::write(&c1, "numOuterThreads 1\n").unwrap();
    std::fs::write(&c2, "numOuterThreads 2\n").unwrap();
    std::fs::write(&c3, "numOuterThreads 3\n").unwrap();

    let inv = parse_cli(&args(&[
        "-m",
        c1.to_str().unwrap(),
        c2.to_str().unwrap(),
        c3.to_str().unwrap(),
    ]))
    .unwrap();

    assert_eq!(inv.mode.kind, ModeKind::Multi);
    assert_eq!(inv.configs.len(), 3);
    assert_eq!(inv.configs[0].get("numOuterThreads").unwrap(), "1");
    assert_eq!(inv.configs[1].get("numOuterThreads").unwrap(), "2");
    assert_eq!(inv.configs[2].get("numOuterThreads").unwrap(), "3");
}

#[test]
fn case_sets_verbose() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = dir.path().join("base.config");
    std::fs::write(&cfg_path, "numOuterThreads 1\n").unwrap();

    let lower = parse_cli(&args(&["-s", cfg_path.to_str().unwrap()])).unwrap();
    assert!(!lower.mode.verbose);
    assert_eq!(lower.mode.kind, ModeKind::Solo);

    let upper = parse_cli(&args(&["-S", cfg_path.to_str().unwrap()])).unwrap();
    assert!(upper.mode.verbose);
    assert_eq!(upper.mode.kind, ModeKind::Solo);

    let multi_upper = parse_cli(&args(&["-M", cfg_path.to_str().unwrap()])).unwrap();
    assert!(multi_upper.mode.verbose);
    assert_eq!(multi_upper.mode.kind, ModeKind::Multi);
}

#[test]
fn too_few_args_errors() {
    // legacy: FastSpeechProcessing.cpp:37-40 -- argc < 3 (progname + mode + config).
    let err = parse_cli(&args(&["-s"])).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("usage") || !err.to_string().is_empty());
}

#[test]
fn unknown_mode_errors() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = dir.path().join("base.config");
    std::fs::write(&cfg_path, "numOuterThreads 1\n").unwrap();
    assert!(parse_cli(&args(&["-x", cfg_path.to_str().unwrap()])).is_err());
}

#[test]
fn malformed_override_errors() {
    // legacy: FastSpeechProcessing.cpp:49 check(argument[0].substr(0,2) == "--", ...).
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = dir.path().join("base.config");
    std::fs::write(&cfg_path, "numOuterThreads 1\n").unwrap();
    assert!(
        parse_cli(&args(&[
            "-s",
            "numOuterThreads=4",
            cfg_path.to_str().unwrap()
        ]))
        .is_err()
    );
}

// === Binary end-to-end =========================================================
//
// Replays `solo_tdc_matches` (phase4a_tier1_e2e.rs) THROUGH the actual compiled
// `speech` binary: `-s` mode, epochs forced to 0 via the CLI override syntax
// itself, cwd = a tempdir seeded with the committed tier-1 corpus.

fn phase4a_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase4a")
}

fn seed_corpus(dir: &Path) {
    let src = phase4a_src();
    let corpus_dst = dir.join("corpus");
    std::fs::create_dir_all(&corpus_dst).unwrap();
    for f in ["f1", "f2", "f3"] {
        std::fs::copy(
            src.join("corpus").join(format!("{f}.wav")),
            corpus_dst.join(format!("{f}.wav")),
        )
        .unwrap();
        std::fs::copy(
            src.join("corpus").join(format!("{f}.stm")),
            corpus_dst.join(format!("{f}.stm")),
        )
        .unwrap();
    }
    for f in [
        "fileslisting.csv",
        "language2classmapping.csv",
        "tier1_tdc.config",
    ] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    std::fs::create_dir_all(dir.join("vrcts_solo")).unwrap();
}

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

/// Minimal MAT v5 reader (mirrors `phase4a_tier1_e2e.rs`; the writer has no reader).
fn mat_var_matrix(bytes: &[u8], name: &str) -> Option<Array2<f64>> {
    let mut off = 128;
    while off + 8 <= bytes.len() {
        let tag = u32_le(bytes, off);
        let size = u32_le(bytes, off + 4) as usize;
        if tag != 14 {
            break;
        }
        let mut p = off + 8;
        p += 8 + 8;
        let dims_size = u32_le(bytes, p + 4) as usize;
        let rows = i32_le(bytes, p + 8) as usize;
        let cols = i32_le(bytes, p + 12) as usize;
        p += 8 + pad8(dims_size);
        let name_size = u32_le(bytes, p + 4) as usize;
        let this_name = String::from_utf8(bytes[p + 8..p + 8 + name_size].to_vec()).unwrap();
        p += 8 + pad8(name_size);
        let data_size = u32_le(bytes, p + 4) as usize;
        let n = data_size / 8;
        if this_name == name {
            let mut data = Vec::with_capacity(n);
            for k in 0..n {
                data.push(f64_le(bytes, p + 8 + k * 8));
            }
            return Some(Array2::from_shape_fn((rows, cols), |(i, j)| {
                data[j * rows + i]
            }));
        }
        off += 8 + size;
    }
    None
}

fn assert_matrix(
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

#[test]
fn binary_end_to_end() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_corpus(dir.path());

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(dir.path())
        .args([
            "-s",
            "--Neural_Networks_BackPropagation_Epochs=0",
            "--multiConfigResultsOutputFile=solo_tdc.mat",
            "--Dump_Directory=vrcts_solo",
            "tier1_tdc.config",
        ])
        .output()
        .expect("failed to run speech binary");

    assert!(
        output.status.success(),
        "speech -s exited non-zero: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mat_path = dir.path().join("solo_tdc.mat");
    assert!(mat_path.exists(), "output .mat was not written");

    let bytes = std::fs::read(&mat_path).unwrap();
    let got_mcr = mat_var_matrix(&bytes, "MultiConfigResults").expect("MultiConfigResults missing");
    let want_mcr = load_bin_phase4a("solo_tdc_MultiConfigResults.bin");
    // cols 0/1/2 = file/conf/chan ids (strict); col 6 = timing (masked).
    assert_matrix(
        &got_mcr,
        &want_mcr,
        &[0, 1, 2],
        &[6],
        "binary e2e MultiConfigResults",
    );
    assert_eq!(got_mcr.dim(), (6, 21), "binary e2e MCR 6x21");

    for var in ["CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"] {
        let got = mat_var_matrix(&bytes, var).unwrap_or_else(|| panic!("{var} missing"));
        let want = load_bin_phase4a(&format!("solo_tdc_{var}.bin"));
        assert_matrix(&got, &want, &[], &[], &format!("binary e2e {var}"));
    }
}

#[test]
fn binary_usage_on_parse_failure() {
    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .args(["-s"])
        .output()
        .expect("failed to run speech binary");

    assert!(
        !output.status.success(),
        "expected non-zero exit on bad args"
    );
    assert_eq!(
        output.status.code(),
        Some(2),
        "usage-failure exit code contract"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_lowercase()
            .contains("usage"),
        "stderr should print usage"
    );
}
