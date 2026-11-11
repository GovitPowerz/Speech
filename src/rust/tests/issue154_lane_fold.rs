//! Issue #154: the corpus fold above one lane.
//!
//! Until this issue no committed test ran the engine fold at `numOuterThreads > 1`:
//! `lanes_n1_equals_sequential` and `fold_order_ascending_golden` both run one lane. The
//! order pins below hold config 0's folded `N x 2` derivative matrix and the results
//! matrix of one gradient fold at two and at three lanes, measured on the commit BEFORE
//! the fold was restructured (files folded as they complete, in place, behind one bounded
//! channel per lane), so the restructure lands against them unmodified.
//!
//! The corpus is the three phase4a files listed twice with six different listing weights:
//! at two lanes the fold runs three rounds, at three lanes two, and every lane chains its
//! bag across more than one file. The listing rows are absolute, so no test here changes
//! directory. Measured when the pins were written: on this corpus (one sample rate, so no
//! driver state a file hands the next one changes a number) the two-lane and three-lane
//! folds are bit-identical to each other. That is a fact of the corpus, not a contract
//! (ADR-0007: a fold above one lane is deterministic but lane-count-dependent), so it is
//! recorded here and not asserted.
//!
//! The comparator is the standing oracle one: bit-exact on the oracle libm host, the
//! canary-gated hybrid bound elsewhere (the derivatives read the net's transcendentals).

mod common;

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::{Array2, ShapeBuilder};

use speech::cli::{Mode, ModeKind};
use speech::engine::corpus_processor::CorpusProcessor;

/// The result matrix's wall-clock column (result column 3 behind the three id columns).
const TIMING_COL: usize = 6;

fn reference(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data")
        .join(rel)
}

fn multi_mode() -> Mode {
    Mode {
        kind: ModeKind::Multi,
        verbose: false,
    }
}

/// The six-row listing and its mapping under `dir`; the rows name the committed phase4a
/// corpus by absolute path.
fn six_row_listing(dir: &Path) -> (PathBuf, PathBuf) {
    let corpus = reference("phase4a/corpus");
    let row = |f: &str, tail: &str| {
        format!(
            "{};{};{tail}\n",
            corpus.join(format!("{f}.wav")).display(),
            corpus.join(format!("{f}.stm")).display()
        )
    };
    let listing = dir.join("lane_fold_fileslisting.csv");
    std::fs::write(
        &listing,
        [
            row("f1", "eng;us;0.5;2"),
            row("f2", ";;0.75"),
            row("f3", ";;0.6"),
            row("f1", ";;0.9"),
            row("f2", ";;0.4"),
            row("f3", ";;0.8"),
        ]
        .concat(),
    )
    .unwrap();
    let mapping = dir.join("lane_fold_language2classmapping.csv");
    std::fs::write(&mapping, "eng;us;0\nunk;unk;1\n").unwrap();
    (listing, mapping)
}

/// The real `1_worker_1.config` (algo 3) with backprop on and `Epochs 0`: one gradient
/// fold at theta (the shape `FoldRun` gives a training step), at `lanes` lanes.
fn lane_fold_config(dir: &Path, lanes: usize) -> IndexMap<String, String> {
    let (listing, mapping) = six_row_listing(dir);
    let text = std::fs::read_to_string(reference("phase0/1_worker_1.config")).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&text);
    for (key, value) in [
        ("numOuterThreads", lanes.to_string()),
        ("Algo_choice", "3".to_string()),
        ("Audio_offset", "0.0".to_string()),
        ("Audio_max_duration", "2.0".to_string()),
        ("File_Type", "0".to_string()),
        ("BLSTM_BackPropagationActivated", "true".to_string()),
        ("BLSTM_BackPropagationRpropInit", "0.05".to_string()),
        ("Neural_Networks_BackPropagation_Epochs", "0".to_string()),
        ("Neural_Networks_Gradient_Check_Epsilon", "0.0".to_string()),
        (
            "BLSTM_weightsFile",
            reference("phase0/NNweights_config1.bin")
                .display()
                .to_string(),
        ),
        ("language2classmapping", mapping.display().to_string()),
        ("fileslisting", listing.display().to_string()),
    ] {
        m.insert(key.to_string(), value);
    }
    m
}

/// One gradient fold at `lanes` lanes: config 0's folded `N x 2` and the results matrix
/// with its timing column zeroed.
fn fold(lanes: usize) -> (Array2<f64>, Array2<f64>) {
    let dir = tempfile::tempdir().unwrap();
    let mut cp =
        CorpusProcessor::new(vec![lane_fold_config(dir.path(), lanes)], multi_mode()).unwrap();
    cp.run().unwrap();
    let trace = cp.epoch_raw_derivs_trace_for_test();
    assert_eq!(trace.len(), 1, "one fold");
    let mut results = cp.results_matrix().clone();
    results.column_mut(TIMING_COL).fill(0.0);
    (trace[0].clone(), results)
}

fn pin_path(lanes: usize, what: &str) -> PathBuf {
    reference(&format!("issue154/lane_fold_n{lanes}_{what}.bin"))
}

fn load_pin(lanes: usize, what: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&pin_path(lanes, what)).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

fn assert_pinned(lanes: usize) {
    let (derivs, results) = fold(lanes);
    let want = load_pin(lanes, "derivs");
    assert!(
        want.column(0).iter().any(|v| *v != 0.0),
        "n = {lanes}: the pin holds no gradient"
    );
    common::assert_oracle_eq(&derivs, &want, &format!("n = {lanes} folded derivatives"));
    let want = load_pin(lanes, "results");
    assert_eq!(want.nrows(), 12, "six files x two channels");
    common::assert_matrix(
        &results,
        &want,
        &[0, 1, 2],
        &[TIMING_COL],
        &format!("n = {lanes} results matrix"),
    );
}

#[test]
fn two_lane_fold_is_port_truth_pinned() {
    assert_pinned(2);
}

#[test]
fn three_lane_fold_is_port_truth_pinned() {
    assert_pinned(3);
}

/// The fixture writer for the pins above. Run ONCE, on the commit the fixtures describe,
/// with `cargo test --features test-support --test issue154_lane_fold write_lane_fold_pins
/// -- --ignored` from `src/rust/`; never as part of a suite run. It refuses to overwrite:
/// delete a fixture by hand to re-baseline it.
#[test]
#[ignore = "fixture writer: run once on the producing commit, see its doc"]
fn write_lane_fold_pins() {
    std::fs::create_dir_all(reference("issue154")).unwrap();
    for lanes in [2, 3] {
        let (derivs, results) = fold(lanes);
        for (what, m) in [("derivs", derivs), ("results", results)] {
            let path = pin_path(lanes, what);
            assert!(
                !path.exists(),
                "{} exists; delete it by hand to re-baseline the pin",
                path.display()
            );
            let data: Vec<f64> = m.t().iter().copied().collect();
            speech::io::binary::write_matrix(&path, m.nrows(), m.ncols(), &data).unwrap();
        }
    }
}
