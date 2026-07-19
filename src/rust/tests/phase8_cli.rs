//! Phase 8 Task 6: the `speech stream` CLI subcommand. Spawns the ACTUAL compiled
//! `speech` binary (`CARGO_BIN_EXE_speech`, the `phase7_bench.rs` idiom) over the staged
//! frozen mono fixture + CALIBRATED config (`common::stage_frozen_tier2` +
//! `common::stage_calibrated`, so REAL segments flow), parses the `SEG`/`STREAM` lines, and
//! cross-checks them against the in-process `StreamingSession` on the SAME staged config:
//!   - the emitted `SEG` (begin/end/class) SET equals the session's final partition (prefix
//!     consistency, the gate's S1.6 property surfaced through the CLI);
//!   - the `STREAM` summary's `max_lag_s`/`mean_lag_s` match the session's own stats (%.3f),
//!     and `emitted + finish_emitted` equals the `SEG` line count;
//!   - a stereo input is a clean non-zero-exit error (the session is mono-first).
//!
//! The CLI reads the wav with its OWN minimal PCM16 parser (raw i16/32768, PRE-gain), so
//! agreement with the in-process session -- fed `common::read_wav_pcm16` raw samples -- also
//! pins the CLI parser against the test helper.

mod common;

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use indexmap::IndexMap;

use speech::fast::stream::StreamingSession;
use speech::tasks::segmentation::SegClass;

fn parse(text: &str) -> IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(text)
}

/// The CLI's `Speech`/`Other` label (mirrors `stream_cli::class_str`).
fn class_str(c: SegClass) -> String {
    match c {
        SegClass::Speech => "Speech".to_string(),
        SegClass::Other => "Other".to_string(),
        other => format!("{other:?}"),
    }
}

/// Read the staged MONO wav's RAW samples (`i16/32768` as f32, PRE-gain), the same shape the
/// CLI's own parser produces + the session's front-end expects. Returns (rate, samples).
fn mono_samples(wav: &Path) -> (f64, Vec<f32>) {
    let (rate, channels, samples) = common::read_wav_pcm16(wav);
    assert_eq!(channels, 1, "the stream CLI test uses the staged MONO wav");
    let s: Vec<f32> = samples.iter().map(|&x| x as f32 / 32768.0).collect();
    (rate as f64, s)
}

/// Run the in-process `StreamingSession` at `chunk` samples (== the CLI's `chunk_ms` in
/// samples): returns (partition triples formatted as the CLI prints, session max/mean lag).
/// The partition triples are `(begin %.4f, end %.4f, class)` per segment `[segs[i],
/// segs[i+1])`; by prefix consistency the CLI's emitted SEG set equals this set.
fn reference_run(
    text: &str,
    rate: f64,
    samples: &[f32],
    chunk: usize,
) -> (Vec<(String, String, String)>, f64, f64) {
    let map = parse(text);
    let mut sess = StreamingSession::new(&map, rate, 1).unwrap();
    let mut i = 0;
    while i < samples.len() {
        let end = (i + chunk).min(samples.len());
        sess.push(&samples[i..end]);
        i = end;
    }
    let (_em, seg) = sess.finish();
    let triples: Vec<(String, String, String)> = seg
        .segments()
        .windows(2)
        .map(|w| {
            (
                format!("{:.4}", w[0].begin),
                format!("{:.4}", w[1].begin),
                class_str(w[0].ty),
            )
        })
        .collect();
    (triples, sess.max_lag_s(), sess.mean_lag_s())
}

/// Parse one `SEG begin=.. end=.. class=.. lag_s=..` line into `(begin, end, class)` (the
/// raw printed strings) plus the parsed `lag_s`.
fn parse_seg_line(line: &str) -> ((String, String, String), f64) {
    let mut begin = None;
    let mut end = None;
    let mut class = None;
    let mut lag = None;
    let mut it = line.split_whitespace();
    assert_eq!(
        it.next(),
        Some("SEG"),
        "SEG line must start with SEG: {line}"
    );
    for tok in it {
        let (k, v) = tok
            .split_once('=')
            .unwrap_or_else(|| panic!("SEG token missing '=': {tok}"));
        match k {
            "begin" => begin = Some(v.to_string()),
            "end" => end = Some(v.to_string()),
            "class" => class = Some(v.to_string()),
            "lag_s" => lag = Some(v.parse::<f64>().expect("lag_s must parse")),
            other => panic!("unexpected SEG key {other}: {line}"),
        }
    }
    ((begin.unwrap(), end.unwrap(), class.unwrap()), lag.unwrap())
}

/// Parse the single `STREAM key=val ...` summary line into a lookup table.
fn parse_stream_line(line: &str) -> std::collections::HashMap<String, String> {
    let mut it = line.split_whitespace();
    assert_eq!(
        it.next(),
        Some("STREAM"),
        "STREAM line must start with STREAM: {line}"
    );
    let mut out = std::collections::HashMap::new();
    for tok in it {
        let (k, v) = tok
            .split_once('=')
            .unwrap_or_else(|| panic!("STREAM token missing '=': {tok}"));
        out.insert(k.to_string(), v.to_string());
    }
    out
}

#[test]
fn stream_seg_and_summary_consistent_with_session() {
    let dir = tempfile::tempdir().unwrap();
    let base = common::stage_frozen_tier2(dir.path());
    let cal_config = common::stage_calibrated(dir.path(), &base);
    let text = std::fs::read_to_string(&cal_config).unwrap();
    let (rate, samples) = mono_samples(&base.wav_path);
    let chunk = (0.1 * rate).round() as usize; // == the CLI's --chunk-ms=100

    // The in-process oracle: the SAME session the CLI wraps, on the SAME staged config.
    let (ref_triples, ref_max_lag, ref_mean_lag) = reference_run(&text, rate, &samples, chunk);
    assert!(
        ref_triples.len() >= 2,
        "the calibrated tail must yield REAL segments (got {} -- staging degenerate?)",
        ref_triples.len()
    );

    // Spawn the actual binary.
    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .args([
            "stream",
            "--chunk-ms=100",
            cal_config.to_str().unwrap(),
            base.wav_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run speech binary");
    assert!(
        output.status.success(),
        "speech stream exited non-zero: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let seg_lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("SEG ")).collect();
    let stream_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("STREAM "))
        .collect();
    assert_eq!(
        stream_lines.len(),
        1,
        "expected exactly one STREAM line:\n{stdout}"
    );
    assert!(
        !seg_lines.is_empty(),
        "expected SEG lines (calibrated tail -> real segments):\n{stdout}"
    );

    // (1) The emitted SEG (begin/end/class) SET equals the session's final partition.
    let cli_set: HashSet<(String, String, String)> =
        seg_lines.iter().map(|l| parse_seg_line(l).0).collect();
    let ref_set: HashSet<(String, String, String)> = ref_triples.iter().cloned().collect();
    assert_eq!(
        cli_set, ref_set,
        "the CLI's emitted SEG set must equal the session's final partition"
    );
    // No duplicate SEG lines (each partition segment emitted exactly once).
    assert_eq!(
        seg_lines.len(),
        cli_set.len(),
        "SEG lines must be unique (prefix consistency: no re-emission)"
    );

    // (2) The STREAM summary is internally consistent + matches the session's stats.
    let fields = parse_stream_line(stream_lines[0]);
    let chunks: usize = fields["chunks"].parse().unwrap();
    let emitted: usize = fields["emitted"].parse().unwrap();
    let finish_emitted: usize = fields["finish_emitted"].parse().unwrap();
    assert!(chunks > 0, "chunks must be > 0");
    assert_eq!(
        emitted + finish_emitted,
        seg_lines.len(),
        "emitted + finish_emitted must equal the SEG line count"
    );

    // max/mean lag match the session's, at the %.3f the CLI prints (same config + samples +
    // chunking -> bit-identical run -> identical lag stats).
    assert_eq!(
        fields["max_lag_s"],
        format!("{ref_max_lag:.3}"),
        "STREAM max_lag_s must match the session's max lag"
    );
    assert_eq!(
        fields["mean_lag_s"],
        format!("{ref_mean_lag:.3}"),
        "STREAM mean_lag_s must match the session's mean lag"
    );

    // rtf is a live timing measurement: just sane (> 0, finite).
    let rtf: f64 = fields["rtf"].parse().expect("rtf must parse as f64");
    assert!(
        rtf > 0.0 && rtf.is_finite(),
        "rtf must be > 0 and finite, got {rtf}"
    );

    // Each SEG line's lag_s equals emitted_at - end; end is the printed %.4f end. Sanity:
    // every lag is finite (the printed value round-trips).
    for l in &seg_lines {
        let (_t, lag) = parse_seg_line(l);
        assert!(lag.is_finite(), "SEG lag_s must be finite: {l}");
    }

    println!(
        "MEASURE stream_cli: seg_lines={} chunks={chunks} emitted={emitted} \
         finish_emitted={finish_emitted} max_lag_s={} mean_lag_s={} rtf={rtf:.6}",
        seg_lines.len(),
        fields["max_lag_s"],
        fields["mean_lag_s"]
    );
}

#[test]
fn stream_rejects_stereo_input() {
    // The committed prcts_excerpt.wav is STEREO; the session is mono-first, so pointing the
    // CLI at it must fail cleanly (non-zero exit, an error mentioning channels/mono). Uses
    // the frozen config (any valid streaming config is fine -- the mono check fires first).
    let dir = tempfile::tempdir().unwrap();
    let base = common::stage_frozen_tier2(dir.path());
    let stereo = common::fixture_phase4d("prcts_excerpt.wav");

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .args([
            "stream",
            base.config_path.to_str().unwrap(),
            stereo.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run speech binary");

    assert!(
        !output.status.success(),
        "speech stream on a stereo wav must exit non-zero (mono-first)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    assert!(
        stderr.contains("mono") || stderr.contains("channel"),
        "stderr should mention the mono/channel constraint: {stderr}"
    );
}

#[test]
fn stream_usage_error_on_missing_args() {
    // `stream` with no positional args is a usage error (exit 2, the parse-failure code).
    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .arg("stream")
        .output()
        .expect("failed to run speech binary");
    assert!(
        !output.status.success(),
        "speech stream with no args must exit non-zero"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_lowercase()
            .contains("usage"),
        "stderr should print usage: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
