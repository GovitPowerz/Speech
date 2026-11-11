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

/// The six-row listing and its mapping under `dir`; the rows name the wavs under `wavs`
/// and the committed phase4a references, by absolute path.
fn six_row_listing(dir: &Path, wavs: &Path) -> (PathBuf, PathBuf) {
    let corpus = reference("phase4a/corpus");
    let row = |f: &str, tail: &str| {
        format!(
            "{};{};{tail}\n",
            wavs.join(format!("{f}.wav")).display(),
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
/// fold at theta (the shape `FoldRun` gives a training step), at `lanes` lanes, over the
/// wavs under `wavs`.
fn lane_fold_config(dir: &Path, lanes: usize, wavs: &Path) -> IndexMap<String, String> {
    let (listing, mapping) = six_row_listing(dir, wavs);
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
    let config = lane_fold_config(dir.path(), lanes, &reference("phase4a/corpus"));
    let mut cp = CorpusProcessor::new(vec![config], multi_mode()).unwrap();
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

/// A TDC (algo 1, no net) config over `rows` listing rows, each `(wav stem, exists)`: a
/// present row names a committed phase4a file, an absent one a wav that is not there.
fn tdc_config(dir: &Path, lanes: usize, rows: &[(&str, bool)]) -> IndexMap<String, String> {
    let corpus = reference("phase4a/corpus");
    let listing = dir.join("tdc_fileslisting.csv");
    let lines: Vec<String> = rows
        .iter()
        .map(|(stem, exists)| {
            let wav = if *exists {
                corpus.join(format!("{stem}.wav"))
            } else {
                dir.join(format!("{stem}.wav"))
            };
            format!(
                "{};{};unk;unk;1.0;1\n",
                wav.display(),
                corpus.join("f1.stm").display()
            )
        })
        .collect();
    std::fs::write(&listing, lines.concat()).unwrap();
    let mapping = dir.join("tdc_language2classmapping.csv");
    std::fs::write(&mapping, "unk;unk;0\n").unwrap();
    let text = std::fs::read_to_string(reference("phase2b/tdc.config")).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&text);
    for (key, value) in [
        ("numOuterThreads", lanes.to_string()),
        ("Algo_choice", "1".to_string()),
        ("TDC_lags", "-0.002,0.016".to_string()),
        ("Audio_offset", "0.0".to_string()),
        ("Audio_max_duration", "2.0".to_string()),
        ("Neural_Networks_BackPropagation_Epochs", "0".to_string()),
        ("language2classmapping", mapping.display().to_string()),
        ("fileslisting", listing.display().to_string()),
    ] {
        m.insert(key.to_string(), value);
    }
    m
}

/// A failed fold surfaces the FIRST failing file in ascending order and commits no
/// result. Two lanes, two missing files: `missing_a` is file 1, on
/// the second lane, and `missing_b` is file 2, on the first. "The lowest failing lane",
/// the rule before issue #154, would name `missing_b`.
#[test]
fn a_failed_fold_names_the_first_failing_file_and_keeps_no_results() {
    let dir = tempfile::tempdir().unwrap();
    let rows = [
        ("f1", true),
        ("missing_a", false),
        ("missing_b", false),
        ("f2", true),
    ];
    let mut cp =
        CorpusProcessor::new(vec![tdc_config(dir.path(), 2, &rows)], multi_mode()).unwrap();
    let err = format!("{:#}", cp.run().unwrap_err());
    assert!(err.contains("missing_a"), "the error names file 1: {err}");
    assert!(!err.contains("missing_b"), "and not file 2: {err}");
    assert!(
        cp.channel_results().is_empty(),
        "file 0 folded before the failure, and a failed fold commits no result"
    );
}

/// A fold that fails after one that succeeded commits nothing: no result (neither the
/// channel results nor the result matrix), and the seam still reads the gradient the
/// successful fold stashed. The wavs are copies, so one can
/// go missing between the two runs.
#[test]
fn a_failed_fold_leaves_the_stash_of_the_fold_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let wavs = dir.path().join("wavs");
    std::fs::create_dir(&wavs).unwrap();
    for f in ["f1.wav", "f2.wav", "f3.wav"] {
        std::fs::copy(reference("phase4a/corpus").join(f), wavs.join(f)).unwrap();
    }
    let config = lane_fold_config(dir.path(), 2, &wavs);
    let mut cp = CorpusProcessor::new(vec![config], multi_mode()).unwrap();
    cp.run().unwrap();
    let stashed = cp.weights_derivatives(0);
    assert!(!cp.channel_results().is_empty());
    assert!(stashed[0].column(0).iter().any(|v| *v != 0.0));

    std::fs::remove_file(wavs.join("f2.wav")).unwrap();
    assert!(cp.run().is_err());
    assert!(cp.channel_results().is_empty());
    assert_eq!(
        cp.results_matrix().dim(),
        (0, 0),
        "the result matrix goes with the results, not left at the fold before"
    );
    let kept = cp.weights_derivatives(0);
    assert_eq!(kept.len(), stashed.len());
    common::assert_bits_eq(&kept[0], &stashed[0], "the stash after a failed fold");
}

/// The fold takes files in as the lanes finish them, and a lane that gets ahead blocks:
/// at three lanes over eighteen files, with the fold slowed so the lanes outrun it, no
/// more entries are ever alive than the bound (two per lane and the one being folded).
/// A fold that buffered every file, or an unbounded lane channel, would reach eighteen.
#[test]
fn the_fold_keeps_a_bounded_number_of_files_alive() {
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<(&str, bool)> = ["f1", "f2", "f3"]
        .iter()
        .cycle()
        .take(18)
        .map(|f| (*f, true))
        .collect();
    let mut cp =
        CorpusProcessor::new(vec![tdc_config(dir.path(), 3, &rows)], multi_mode()).unwrap();
    cp.set_fold_delay_for_test(std::time::Duration::from_millis(100));
    cp.run().unwrap();
    assert_eq!(
        cp.results_matrix().nrows(),
        36,
        "eighteen files x two channels"
    );
    let (peak, bound) = cp.lane_entries_peak_for_test();
    assert_eq!(
        bound, 7,
        "three lanes: two entries each and the one being folded"
    );
    assert!(
        (2..=bound).contains(&peak),
        "{peak} entries alive at once, the bound is {bound}"
    );
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
