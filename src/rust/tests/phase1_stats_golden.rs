//! Phase 1 Task 10: bit-exact golden tests for `InputStatistics`
//! (`from_matrix`/`update`, ported from `InputStatistics.cpp:6-51`).
//!
//! Goldens are dumped by the REAL compiled legacy `InputStatistics.cpp` (no
//! transcription -- it links directly into `tools/oracle_harness`). `to_bits`
//! equality is the contract.

mod common;

use ndarray::Array2;
use serde::Deserialize;
use speech::features::stats::InputStatistics;

fn synth_20x50() -> Array2<f64> {
    common::load_bin("synth_20x50.bin")
}

fn assert_stats_bits_eq(
    got: &InputStatistics,
    want_mean: &Array2<f64>,
    want_std: &Array2<f64>,
    want_n: i64,
    label: &str,
) {
    assert_eq!(got.n, want_n, "{label}: n mismatch");
    assert_eq!(
        got.mean.len(),
        want_mean.ncols(),
        "{label}: mean len mismatch"
    );
    assert_eq!(got.std.len(), want_std.ncols(), "{label}: std len mismatch");
    for j in 0..got.mean.len() {
        let gv = got.mean[j];
        let wv = want_mean[[0, j]];
        assert_eq!(
            gv.to_bits(),
            wv.to_bits(),
            "{label}: mean[{j}] rust=0x{:016x} ({gv}) want=0x{:016x} ({wv})",
            gv.to_bits(),
            wv.to_bits()
        );
    }
    for j in 0..got.std.len() {
        let gv = got.std[j];
        let wv = want_std[[0, j]];
        assert_eq!(
            gv.to_bits(),
            wv.to_bits(),
            "{label}: std[{j}] rust=0x{:016x} ({gv}) want=0x{:016x} ({wv})",
            gv.to_bits(),
            wv.to_bits()
        );
    }
}

// --- Golden: batch stats over the full synth_20x50 matrix ------------------

#[test]
fn batch_stats_bitexact() {
    let synth = synth_20x50();
    let got = InputStatistics::from_matrix(&synth);

    let want_mean = common::load_bin("stats_batch_mean.bin");
    let want_std = common::load_bin("stats_batch_std.bin");
    let want_n = common::load_bin("stats_batch_n.bin")[[0, 0]] as i64;

    assert_stats_bits_eq(&got, &want_mean, &want_std, want_n, "batch");
}

// --- Golden: merge of two batches over row splits [0,7) and [7,20) ----------

#[test]
fn merged_stats_bitexact() {
    let synth = synth_20x50();
    let part1 = synth.slice(ndarray::s![0..7, ..]).to_owned();
    let part2 = synth.slice(ndarray::s![7..20, ..]).to_owned();

    let mut merged = InputStatistics::from_matrix(&part1);
    let other = InputStatistics::from_matrix(&part2);
    merged.update(&other);

    let want_mean = common::load_bin("stats_merged_mean.bin");
    let want_std = common::load_bin("stats_merged_std.bin");
    let want_n = common::load_bin("stats_merged_n.bin")[[0, 0]] as i64;

    assert_stats_bits_eq(&merged, &want_mean, &want_std, want_n, "merged");
}

// --- Unit tests (brief) ------------------------------------------------------

#[test]
fn from_matrix_column_vector_mean_and_std() {
    // [[1],[2],[3]]: mean = 2.0; std = sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/3) exact.
    let m = ndarray::arr2(&[[1.0], [2.0], [3.0]]);
    let got = InputStatistics::from_matrix(&m);
    assert_eq!(got.n, 3);
    assert_eq!(got.mean, vec![2.0]);
    let expected_std = (((1.0_f64 - 2.0) * (1.0 - 2.0) + 0.0 + 1.0) / 3.0_f64).sqrt();
    assert_eq!(got.std[0].to_bits(), expected_std.to_bits());
}

#[test]
fn from_matrix_single_row_std_exactly_zero() {
    // n=1 -> std == 0.0 exactly (single sample, no deviation from its own mean).
    let m = ndarray::arr2(&[[5.0, -3.0]]);
    let got = InputStatistics::from_matrix(&m);
    assert_eq!(got.n, 1);
    assert_eq!(got.mean, vec![5.0, -3.0]);
    assert_eq!(got.std[0], 0.0);
    assert_eq!(got.std[1], 0.0);
}

#[test]
fn update_empty_into_empty_keeps_n_zero() {
    let mut a = InputStatistics::new();
    let b = InputStatistics::new();
    a.update(&b);
    assert_eq!(a.n, 0);
    assert!(a.mean.is_empty());
    assert!(a.std.is_empty());
}

#[test]
fn update_nonempty_into_empty_is_no_op_bit_identical() {
    // nonempty.update(empty): other.n == 0 and self.n != 0 -> silent no-op.
    let m = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let original = InputStatistics::from_matrix(&m);
    let mut a = original.clone();
    let empty = InputStatistics::new();
    a.update(&empty);
    assert_eq!(a.n, original.n);
    for (av, ov) in a.mean.iter().zip(original.mean.iter()) {
        assert_eq!(av.to_bits(), ov.to_bits());
    }
    for (av, ov) in a.std.iter().zip(original.std.iter()) {
        assert_eq!(av.to_bits(), ov.to_bits());
    }
}

#[test]
fn update_empty_into_nonempty_is_plain_copy() {
    // empty.update(nonempty): self.n == 0 -> plain copy of other's fields.
    let m = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let other = InputStatistics::from_matrix(&m);
    let mut a = InputStatistics::new();
    a.update(&other);
    assert_eq!(a.n, other.n);
    for (av, ov) in a.mean.iter().zip(other.mean.iter()) {
        assert_eq!(av.to_bits(), ov.to_bits());
    }
    for (av, ov) in a.std.iter().zip(other.std.iter()) {
        assert_eq!(av.to_bits(), ov.to_bits());
    }
}

// --- Rust == Python numpy-oracle cross-check (chained merges) ---------------

#[derive(Deserialize)]
struct StatsChunk {
    rows: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct StatsCase {
    chunks: Vec<StatsChunk>,
    expected_mean: Vec<f64>,
    expected_std: Vec<f64>,
    expected_n: i64,
}

fn to_array(rows: &[Vec<f64>]) -> Array2<f64> {
    let t = rows.len();
    let w = if t == 0 { 0 } else { rows[0].len() };
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((t, w), flat).unwrap()
}

#[test]
fn stats_merge_oracle_cross_check_bit_exact() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase1/stats_cases.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let cases: Vec<StatsCase> = serde_json::from_str(&text).unwrap();
    assert!(!cases.is_empty(), "expected at least one stats case");

    for (idx, case) in cases.iter().enumerate() {
        let mut acc = InputStatistics::new();
        for chunk in &case.chunks {
            if chunk.rows.is_empty() {
                let empty = InputStatistics::new();
                acc.update(&empty);
                continue;
            }
            let arr = to_array(&chunk.rows);
            let batch = InputStatistics::from_matrix(&arr);
            acc.update(&batch);
        }

        assert_eq!(acc.n, case.expected_n, "case {idx}: n mismatch");
        assert_eq!(
            acc.mean.len(),
            case.expected_mean.len(),
            "case {idx}: mean len mismatch"
        );
        for j in 0..acc.mean.len() {
            let gv = acc.mean[j];
            let wv = case.expected_mean[j];
            assert_eq!(
                gv.to_bits(),
                wv.to_bits(),
                "case {idx} mean[{j}]: rust=0x{:016x} ({gv}) py=0x{:016x} ({wv})",
                gv.to_bits(),
                wv.to_bits()
            );
        }
        for j in 0..acc.std.len() {
            let gv = acc.std[j];
            let wv = case.expected_std[j];
            assert_eq!(
                gv.to_bits(),
                wv.to_bits(),
                "case {idx} std[{j}]: rust=0x{:016x} ({gv}) py=0x{:016x} ({wv})",
                gv.to_bits(),
                wv.to_bits()
            );
        }
    }
}
