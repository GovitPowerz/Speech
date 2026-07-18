//! Phase 7 Task 1: the `speech bench` end-to-end harness.
//!
//! Port-only tooling (no legacy source) -- spawns the ACTUAL compiled `speech`
//! binary (`CARGO_BIN_EXE_speech`), the same idiom `phase4a_cli.rs::
//! binary_end_to_end` uses, over a tempdir staged with the committed 60 s
//! `phase4d/prcts_excerpt.wav` fixture, the tuple-A weight pack
//! (`phase0/NNweights_config1.bin`), and `phase4a/tier2_spectral.config` with
//! a last-wins override tail (mirroring the fixture's own "Phase 4a Task 9"
//! tail) pointing it at the staged files and forcing a single unscored
//! forward pass.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::{fixture_phase0, fixture_phase4a, fixture_phase4d};

/// Stage a tempdir for `speech bench`: the 60 s wav, the tuple-A pack, a
/// single-file no-reference listing (unscored -- see `engine/corpus.rs`'s
/// listing parse: a bare filename token leaves `refseg` empty), and
/// `tier2_spectral.config` with an appended override block. Returns the
/// staged config's path.
///
/// Overrides (last-wins, same convention the fixture's own tail uses):
/// `Audio_max_duration` lifted from the fixture's baked-in 2.0 s (phase4a
/// Task 9) back up past the full 60 s file; `fileslisting`/
/// `language2classmapping` repointed at the staged single-file corpus;
/// `Neural_Networks_BackPropagation_Epochs 0` forces `CorpusProcessor::run`'s
/// `run_solo` path (a single pass, matching an ordinary `-i` invocation with
/// no training config); `BLSTM_BackPropagationActivated false` keeps the
/// measured pass forward-only, comparable to Task 2's forward-only fast path.
fn stage_bench_config(dir: &Path) -> PathBuf {
    std::fs::copy(
        fixture_phase4d("prcts_excerpt.wav"),
        dir.join("prcts_excerpt.wav"),
    )
    .unwrap();
    std::fs::copy(
        fixture_phase0("NNweights_config1.bin"),
        dir.join("NNweights_config1.bin"),
    )
    .unwrap();

    std::fs::write(dir.join("bench_listing.csv"), "prcts_excerpt.wav\n").unwrap();
    // language2classmapping must exist and be readable (Corpus::from_config
    // bails only if the file cannot be opened) -- empty content is valid, every
    // listing item falls back to the "unk"/"unk" language/dialect default.
    std::fs::write(dir.join("bench_mapping.csv"), "").unwrap();
    std::fs::create_dir_all(dir.join("vrcts_bench")).unwrap();

    let base = std::fs::read_to_string(fixture_phase4a("tier2_spectral.config")).unwrap();
    let staged = format!(
        "{base}\n\
# ==== Phase 7 Task 1 bench overrides (last-wins) ====\n\
Audio_max_duration 3600000\n\
fileslisting bench_listing.csv\n\
language2classmapping bench_mapping.csv\n\
multiConfigResultsOutputFile bench_result.mat\n\
Dump_Directory vrcts_bench\n\
Neural_Networks_BackPropagation_Epochs 0\n\
BLSTM_BackPropagationActivated false\n"
    );
    let cfg_path = dir.join("bench_tier2.config");
    std::fs::write(&cfg_path, staged).unwrap();
    cfg_path
}

/// Parse one `BENCH key=val key=val ...` line into a lookup table.
fn parse_bench_line(line: &str) -> HashMap<&str, &str> {
    let mut it = line.split_whitespace();
    assert_eq!(
        it.next(),
        Some("BENCH"),
        "line must start with the BENCH literal: {line}"
    );
    let mut out = HashMap::new();
    for tok in it {
        let (k, v) = tok
            .split_once('=')
            .unwrap_or_else(|| panic!("BENCH token missing '=': {tok}"));
        out.insert(k, v);
    }
    out
}

#[test]
fn bench_line_parses() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = stage_bench_config(dir.path());
    let cfg_name = cfg_path.file_name().unwrap().to_str().unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(dir.path())
        .args(["bench", "--repeat=1", cfg_name])
        .output()
        .expect("failed to run speech binary");

    assert!(
        output.status.success(),
        "speech bench exited non-zero: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let bench_lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("BENCH ")).collect();
    assert_eq!(
        bench_lines.len(),
        1,
        "expected exactly one BENCH line (--repeat=1), got:\n{stdout}"
    );

    let fields = parse_bench_line(bench_lines[0]);
    assert_eq!(fields.get("path").copied(), Some("exact"));

    let wall_s: f64 = fields["wall_s"].parse().expect("wall_s must parse as f64");
    assert!(wall_s > 0.0, "wall_s must be > 0, got {wall_s}");

    let rtf: f64 = fields["rtf"].parse().expect("rtf must parse as f64");
    assert!(rtf > 0.0, "rtf must be > 0, got {rtf}");

    // 60 s x 2 channels (prcts_excerpt.wav is stereo, confirmed via afinfo).
    let audio_s: f64 = fields["audio_s"]
        .parse()
        .expect("audio_s must parse as f64");
    let want_audio_s = 120.0;
    assert!(
        (audio_s - want_audio_s).abs() / want_audio_s < 0.01,
        "audio_s {audio_s} not within 1% of {want_audio_s}"
    );

    let maxrss_mb: f64 = fields["maxrss_mb"]
        .parse()
        .expect("maxrss_mb must parse as f64");
    assert!(
        maxrss_mb > 1.0 && maxrss_mb < 4096.0,
        "maxrss_mb {maxrss_mb} not in a sane range (unit-normalization bug?)"
    );

    let files: usize = fields["files"].parse().expect("files must parse as usize");
    assert_eq!(files, 1, "single-file staged listing");
}

#[test]
fn bench_rejects_unknown_path() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = stage_bench_config(dir.path());
    let cfg_name = cfg_path.file_name().unwrap().to_str().unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(dir.path())
        .args(["bench", "--path=fast", cfg_name])
        .output()
        .expect("failed to run speech binary");

    assert!(
        !output.status.success(),
        "speech bench --path=fast must exit non-zero: 'fast' is not wired until Task 4"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_lowercase()
            .contains("path"),
        "stderr should mention the bad --path value: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
