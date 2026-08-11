//! Phase 10 Task 10: the `speech stream-lid` CLI subcommand. Spawns the ACTUAL compiled
//! `speech` binary (`CARGO_BIN_EXE_speech`, the `phase8_cli.rs` idiom) over the committed
//! phase-8 stream-lid phSeq fixture (`twin_mode7.config` + `corpus_phseq/s{1,2,3}.phSeq`),
//! parses the `UTT`/`STREAM_LID` lines, and cross-checks them against an in-process
//! `StreamingLidSession` fed the SAME `external_features` (read via `audio::read_audio`,
//! the same call the CLI itself makes) -- a cross-check of the CLI's own arg
//! parsing/file reading/print formatting against a known-good in-process call. The
//! underlying LID arithmetic is already proven bit-identical to the offline fast Twin by
//! `tests/phase8_stream_lid.rs`; this test does not re-derive that, it pins the NEW
//! surface (argv parsing, `File_Type`-driven read, the line format) on top of it.

use std::path::PathBuf;
use std::process::Command;

use ndarray::Array2;

use speech::audio::read_audio;
use speech::fast::stream_lid::StreamingLidSession;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}
fn phase4b(name: &str) -> PathBuf {
    ref_dir().join("phase4b").join(name)
}
fn config_path() -> PathBuf {
    phase4b("twin_mode7.config")
}
fn phseq_path(f: &str) -> PathBuf {
    phase4b("corpus_phseq").join(format!("{f}.phSeq"))
}

/// The repo root (`src/rust/../..`): the committed config's `BLSTM_LID_weightsFile` is a
/// repo-root-relative path (`tests/reference_data/phase4b/LID_bestNNWeight_1.bin`, matching
/// every other committed config in this repo -- see CLAUDE.md's "Run from repo root:"
/// convention), so the SPAWNED binary must run with this as its CWD to resolve it. Our own
/// absolute `config_path()`/`phseq_path()` args are unaffected by CWD.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The real LID net's flat weight vector, loaded by ABSOLUTE path (mirrors
/// `tests/phase8_stream_lid.rs::lid_weights`) -- sidesteps the CWD dependency for the
/// in-process reference run (unlike the spawned CLI, `reference_run` never chdirs).
fn lid_weights() -> Vec<f64> {
    speech::io::binary::read_weight_vector(&phase4b("LID_bestNNWeight_1.bin")).unwrap()
}

/// Parse one `<PREFIX> k=v k=v ...` line into a lookup table (mirrors
/// `phase8_cli.rs::parse_stream_line`).
fn parse_kv_line(prefix: &str, line: &str) -> std::collections::HashMap<String, String> {
    let mut it = line.split_whitespace();
    assert_eq!(
        it.next(),
        Some(prefix),
        "line must start with {prefix}: {line}"
    );
    let mut out = std::collections::HashMap::new();
    for tok in it {
        let (k, v) = tok
            .split_once('=')
            .unwrap_or_else(|| panic!("{prefix} token missing '=': {tok}"));
        out.insert(k.to_string(), v.to_string());
    }
    out
}

/// One utterance's CLI-printed fields, in the `UTT` line's own key order (minus `idx`,
/// checked separately).
struct UttFields {
    scored: String,
    argmax: String,
    correct: String,
    agg_predicted_language: String,
    agg_is_lid_correct: String,
    agg_segments_count: String,
}

/// The in-process reference run: push every `external_features` entry through a fresh
/// `StreamingLidSession` built on the SAME config/lang, returning the per-utterance
/// fields as the CLI would print them, plus the final `finish()` summary fields.
fn reference_run(lang: i32, feats: &[Array2<f64>]) -> (Vec<UttFields>, [String; 3]) {
    let map = speech::legacy_config::parse_legacy_config(
        &std::fs::read_to_string(config_path()).unwrap(),
    );
    let lidw = lid_weights();
    let mut sess = StreamingLidSession::new(&map, 8000.0, lang, Some(&lidw)).unwrap();
    let mut rows = Vec::with_capacity(feats.len());
    for feat in feats {
        let sc = sess.push_utterance(feat);
        rows.push(UttFields {
            scored: sc.argmax.is_some().to_string(),
            argmax: sc.argmax.map_or_else(|| "-".to_string(), |x| x.to_string()),
            correct: sc
                .is_correct
                .map_or_else(|| "-".to_string(), |x| x.to_string()),
            agg_predicted_language: sc.running_aggregate.predicted_language.to_string(),
            agg_is_lid_correct: (sc.running_aggregate.is_lid_correct != 0).to_string(),
            agg_segments_count: sc.running_aggregate.segments_count.to_string(),
        });
    }
    let fin = sess.finish();
    (
        rows,
        [
            fin.segments_count.to_string(),
            fin.predicted_language.to_string(),
            (fin.is_lid_correct != 0).to_string(),
        ],
    )
}

fn run_case(f: &str, lang: i32) {
    let audio = read_audio(&phseq_path(f), 0.0, 120.0, 1, None).unwrap();
    let (ref_rows, ref_summary) = reference_run(lang, &audio.external_features);

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(repo_root())
        .args([
            "stream-lid",
            &format!("--lang={lang}"),
            config_path().to_str().unwrap(),
            phseq_path(f).to_str().unwrap(),
        ])
        .output()
        .expect("failed to run speech binary");
    assert!(
        output.status.success(),
        "speech stream-lid exited non-zero for {f}: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let utt_lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("UTT ")).collect();
    let summary_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("STREAM_LID "))
        .collect();
    assert_eq!(
        summary_lines.len(),
        1,
        "{f}: expected exactly one STREAM_LID line:\n{stdout}"
    );
    assert_eq!(
        utt_lines.len(),
        ref_rows.len(),
        "{f}: UTT line count must equal the utterance count"
    );

    for (i, (line, want)) in utt_lines.iter().zip(ref_rows.iter()).enumerate() {
        let fields = parse_kv_line("UTT", line);
        assert_eq!(fields["idx"], i.to_string(), "{f} u{i}: idx");
        assert_eq!(fields["scored"], want.scored, "{f} u{i}: scored");
        assert_eq!(fields["argmax"], want.argmax, "{f} u{i}: argmax");
        assert_eq!(fields["correct"], want.correct, "{f} u{i}: correct");
        assert_eq!(
            fields["agg_predicted_language"], want.agg_predicted_language,
            "{f} u{i}: agg_predicted_language"
        );
        assert_eq!(
            fields["agg_is_lid_correct"], want.agg_is_lid_correct,
            "{f} u{i}: agg_is_lid_correct"
        );
        assert_eq!(
            fields["agg_segments_count"], want.agg_segments_count,
            "{f} u{i}: agg_segments_count"
        );
    }

    let fields = parse_kv_line("STREAM_LID", summary_lines[0]);
    assert_eq!(
        fields["utterances"],
        ref_rows.len().to_string(),
        "{f}: utterances"
    );
    assert_eq!(fields["scored"], ref_summary[0], "{f}: scored");
    assert_eq!(
        fields["predicted_language"], ref_summary[1],
        "{f}: predicted_language"
    );
    assert_eq!(
        fields["is_lid_correct"], ref_summary[2],
        "{f}: is_lid_correct"
    );
    let wall_s: f64 = fields["wall_s"].parse().expect("wall_s must parse");
    assert!(
        wall_s >= 0.0 && wall_s.is_finite(),
        "{f}: wall_s must be finite and non-negative"
    );

    println!(
        "MEASURE stream_lid_cli {f}: utterances={} scored={} wall_s={}",
        ref_rows.len(),
        ref_summary[0],
        fields["wall_s"]
    );
}

#[test]
fn stream_lid_utt_and_summary_consistent_with_session() {
    // The committed phase-8 stream-lid phSeq fixtures (`PHSEQ_FILES` in
    // `tests/phase8_stream_lid.rs`): s1/lang0, s2/lang1, s3/lang1 (the s3 MISS case).
    for (f, lang) in [("s1", 0), ("s2", 1), ("s3", 1)] {
        run_case(f, lang);
    }
}

#[test]
fn stream_lid_usage_error_on_missing_args() {
    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(repo_root())
        .arg("stream-lid")
        .output()
        .expect("failed to run speech binary");
    assert!(
        !output.status.success(),
        "speech stream-lid with no args must exit non-zero"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_lowercase()
            .contains("usage"),
        "stderr should print usage: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn stream_lid_bails_on_unsupported_file_type() {
    // Override the committed config's `File_Type 1` to `0` (wav -- no external_features):
    // the CLI must bail loudly before ever building the session.
    let dir = tempfile::tempdir().unwrap();
    let base_text = std::fs::read_to_string(config_path()).unwrap();
    let staged = dir.path().join("filetype0.config");
    std::fs::write(&staged, format!("{base_text}\nFile_Type 0\n")).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(repo_root())
        .args([
            "stream-lid",
            staged.to_str().unwrap(),
            phseq_path("s1").to_str().unwrap(),
        ])
        .output()
        .expect("failed to run speech binary");
    assert!(
        !output.status.success(),
        "speech stream-lid with File_Type 0 must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    assert!(
        stderr.contains("file_type"),
        "stderr should mention File_Type: {stderr}"
    );
}

#[test]
fn stream_lid_bails_on_missing_lid_net() {
    // Blank the committed config's `BLSTM_LID_weightsFile` (last-wins override): the
    // session cannot load a LID net, so construction must bail loudly (mirrors
    // `tests/phase8_stream_lid.rs::new_bails_on_missing_lid_net`).
    let dir = tempfile::tempdir().unwrap();
    let base_text = std::fs::read_to_string(config_path()).unwrap();
    let staged = dir.path().join("nonet.config");
    std::fs::write(&staged, format!("{base_text}\nBLSTM_LID_weightsFile \n")).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(repo_root())
        .args([
            "stream-lid",
            staged.to_str().unwrap(),
            phseq_path("s1").to_str().unwrap(),
        ])
        .output()
        .expect("failed to run speech binary");
    assert!(
        !output.status.success(),
        "speech stream-lid with no LID net must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    assert!(
        stderr.contains("net"),
        "stderr should mention the missing net: {stderr}"
    );
}
