//! Phase 7 Task 1: the `speech bench` end-to-end harness. Task 7 wires
//! `--path=fast` in (RED: `bench_rejects_unknown_path` used to spawn
//! `--path=fast` and assert rejection -- it now spawns a genuinely-unknown
//! value, and the flipped acceptance case moves to `bench_path_fast_runs`)
//! plus the CI budget smoke (`bench_fast_not_slower_than_exact_ci_smoke`,
//! `fast wall <= exact wall * 1.5`, spec R5 wide headroom).
//!
//! Port-only tooling (no legacy source) -- spawns the ACTUAL compiled `speech`
//! binary (`CARGO_BIN_EXE_speech`), the same idiom `phase4a_cli.rs::
//! binary_end_to_end` uses, over a tempdir staged with the committed 60 s
//! `phase4d/prcts_excerpt.wav` fixture, the tuple-A weight pack
//! (`phase0/NNweights_config1.bin`), and `phase4a/tier2_spectral.config` with
//! a last-wins override tail (mirroring the fixture's own "Phase 4a Task 9"
//! tail) pointing it at the staged files and forcing a single unscored
//! forward pass. This SAME staged config is fast-path-compatible as-is (algo
//! 3, `InputNormalizationType -1`, no pitch pass -- the identical config the
//! Task 4 CI parity leg uses), so no separate fast staging recipe is needed;
//! `speech bench --path=fast` on it dispatches to `FastSpectralSegmenter` via
//! the `Inference_Path` overlay `bench::run_bench` now applies (Task 7).

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

/// Spawn `speech bench --repeat=1 --path=<path> <cfg_name>` in `dir`, assert a
/// clean exit + exactly one BENCH line, and return its parsed fields (owned,
/// so the caller isn't tied to the `Output`'s borrow).
fn run_bench_cli(dir: &Path, cfg_name: &str, path: &str) -> HashMap<String, String> {
    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(dir)
        .args(["bench", "--repeat=1", &format!("--path={path}"), cfg_name])
        .output()
        .expect("failed to run speech binary");

    assert!(
        output.status.success(),
        "speech bench --path={path} exited non-zero: status={:?}\nstdout={}\nstderr={}",
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
    parse_bench_line(bench_lines[0])
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
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
    // Phase 7 Task 7 RED->flip: `--path=fast` used to be the rejected value
    // (fast wasn't wired until Task 4/7); it now runs cleanly (see
    // `bench_path_fast_runs` below). This test keeps covering
    // `BenchPath::parse`'s genuine-usage-error branch with a value that is
    // NEVER going to be valid.
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = stage_bench_config(dir.path());
    let cfg_name = cfg_path.file_name().unwrap().to_str().unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(dir.path())
        .args(["bench", "--path=turbo", cfg_name])
        .output()
        .expect("failed to run speech binary");

    assert!(
        !output.status.success(),
        "speech bench --path=turbo must exit non-zero: only 'exact'/'fast' are supported"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_lowercase()
            .contains("path"),
        "stderr should mention the bad --path value: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn bench_path_fast_runs() {
    // Phase 7 Task 7: the flipped acceptance half of the RED above --
    // `--path=fast` on the SAME staged config now runs cleanly end to end
    // (BagOfProcessors dispatches algo 3 + `Inference_Path fast` to
    // `FastSpectralSegmenter`) and reports `path=fast`, not silently falling
    // back to exact.
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = stage_bench_config(dir.path());
    let cfg_name = cfg_path.file_name().unwrap().to_str().unwrap();

    let fields = run_bench_cli(dir.path(), cfg_name, "fast");
    assert_eq!(fields.get("path").map(String::as_str), Some("fast"));

    let wall_s: f64 = fields["wall_s"].parse().expect("wall_s must parse as f64");
    assert!(wall_s > 0.0, "wall_s must be > 0, got {wall_s}");

    let rtf: f64 = fields["rtf"].parse().expect("rtf must parse as f64");
    assert!(rtf > 0.0, "rtf must be > 0, got {rtf}");

    // Same 60 s x 2 channel wav as the exact leg (bench_line_parses) -- the
    // fast path reads the SAME audio, so audio_s must match exactly.
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
fn bench_fast_not_slower_than_exact_ci_smoke() {
    // The CI budget smoke (spec R5): fast must not be CATASTROPHICALLY slower
    // than exact on the 60 s fixture. WIDE headroom (1.5x, not a tight bound)
    // -- shared CI runners make sub-second wall-clock timing flaky (spec R5),
    // so this exists to catch a structural regression (e.g. the fast path
    // accidentally falling back to a slow scalar loop), not to pin the real
    // measured ratio. The real numbers, measured with more repeats on this
    // box, are recorded in RESULTS.md's Phase 7 section as local regression
    // bounds -- not this test (see that file for the measured speedup).
    const CI_SLOWDOWN_BUDGET: f64 = 1.5;

    let dir = tempfile::tempdir().unwrap();
    let cfg_path = stage_bench_config(dir.path());
    let cfg_name = cfg_path.file_name().unwrap().to_str().unwrap();

    let exact = run_bench_cli(dir.path(), cfg_name, "exact");
    let fast = run_bench_cli(dir.path(), cfg_name, "fast");

    let exact_wall: f64 = exact["wall_s"].parse().unwrap();
    let fast_wall: f64 = fast["wall_s"].parse().unwrap();

    println!(
        "MEASURE bench_fast_not_slower_than_exact_ci_smoke: exact_wall={exact_wall:.6}s fast_wall={fast_wall:.6}s ratio={:.3}",
        fast_wall / exact_wall
    );
    assert!(
        fast_wall <= exact_wall * CI_SLOWDOWN_BUDGET,
        "fast wall {fast_wall}s exceeds {CI_SLOWDOWN_BUDGET}x the exact wall {exact_wall}s \
         (CI budget smoke -- a catastrophic fast-path regression, not a tight timing pin)"
    );
}

/// Issue #20: `--json --label=NAME` prints ONE document per invocation (the
/// ledger's `bench` payload): the label, the path that ran, the repeat count,
/// a config hash, the binary's build provenance and one entry per run, and
/// NO `BENCH` text line. The label is what names a bench recipe in the
/// ledger, so the staged tempdir path must appear nowhere in the document.
#[test]
fn bench_json_document() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = stage_bench_config(dir.path());
    let cfg_name = cfg_path.file_name().unwrap().to_str().unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(dir.path())
        .args([
            "bench",
            "--repeat=2",
            "--path=fast",
            "--json",
            "--label=phase7_60s",
            cfg_name,
        ])
        .output()
        .expect("failed to run speech binary");
    assert!(
        output.status.success(),
        "speech bench --json exited non-zero: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        !stdout.contains("BENCH "),
        "--json must replace the BENCH lines, got:\n{stdout}"
    );
    assert!(
        !stdout.contains(dir.path().to_str().unwrap()),
        "the staged directory must not leak into the document:\n{stdout}"
    );

    let doc: serde_json::Value = serde_json::from_str(&stdout).expect("one JSON document");
    assert_eq!(doc["label"], "phase7_60s");
    assert_eq!(doc["path"], "fast");
    assert_eq!(doc["repeat"], 2);
    let hash = doc["config_hash"]
        .as_str()
        .expect("config_hash is a string");
    assert_eq!(hash.len(), 16, "FNV-1a 64 as 16 hex chars, got {hash}");
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(matches!(
        doc["build"]["profile"].as_str(),
        Some("debug" | "release")
    ));
    assert!(
        doc["build"]["target"]
            .as_str()
            .is_some_and(|t| !t.is_empty())
    );
    assert!(
        doc["build"]["rustc"]
            .as_str()
            .is_some_and(|r| r.starts_with("rustc "))
    );

    let runs = doc["runs"].as_array().expect("runs is an array");
    assert_eq!(runs.len(), 2, "--repeat=2 gives two runs in one document");
    for run in runs {
        assert_eq!(run["path"], "fast");
        assert!(run["wall_s"].as_f64().unwrap() > 0.0);
        assert!((run["audio_s"].as_f64().unwrap() - 120.0).abs() < 1.2);
        assert!(run["rtf"].as_f64().unwrap() > 0.0);
        assert!(run["maxrss_mb"].as_f64().unwrap() > 1.0);
        assert_eq!(run["files"], 1);
    }
}

/// The bench arg grammar: `--json` is a bare flag, `--label=` needs a value,
/// both default off, and the pre-existing flags still parse around them.
#[test]
fn bench_args_parse_json_and_label() {
    use speech::cli::parse_bench_args;
    let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    let inv = parse_bench_args(&args(&["--repeat=3", "cfg.config"])).unwrap();
    assert!(!inv.json && inv.label.is_none());

    let inv = parse_bench_args(&args(&[
        "--json",
        "--label=phase7_60s",
        "--path=fast",
        "cfg.config",
    ]))
    .unwrap();
    assert!(inv.json);
    assert_eq!(inv.label.as_deref(), Some("phase7_60s"));
    assert_eq!(inv.path, "fast");
    assert_eq!(inv.repeat, 1);

    assert!(parse_bench_args(&args(&["--label=", "cfg.config"])).is_err());
    assert!(parse_bench_args(&args(&["--label=phase7_60s", "cfg.config"])).is_err());
    assert!(parse_bench_args(&args(&["--json=yes", "cfg.config"])).is_err());
}

/// `stage_bench_config` with EVERY path key absolute -- the inputs under `dir`,
/// the two OUTPUT keys (`multiConfigResultsOutputFile`, `Dump_Directory`)
/// under `out` -- appended last-wins after the relative tail. This is the
/// ledger's staging shape (`speech.ledger.stage`): a temp staging dir, the
/// binary run from wherever the caller is. `out/vrcts_bench` is created here;
/// the `.mat` and its `bestNNWeight_1_`-prefixed weight-save siblings land
/// next to it.
fn stage_bench_config_absolute(dir: &Path, out: &Path) -> PathBuf {
    let cfg_path = stage_bench_config(dir);
    std::fs::write(
        dir.join("bench_listing.csv"),
        format!("{}\n", dir.join("prcts_excerpt.wav").display()),
    )
    .unwrap();
    std::fs::create_dir_all(out.join("vrcts_bench")).unwrap();
    let mut staged = std::fs::read_to_string(&cfg_path).unwrap();
    staged.push_str(&format!(
        "# ==== absolute paths, inputs and outputs (last-wins) ====\n\
BLSTM_weightsFile {}\n\
language2classmapping {}\n\
fileslisting {}\n\
multiConfigResultsOutputFile {}\n\
Dump_Directory {}\n",
        dir.join("NNweights_config1.bin").display(),
        dir.join("bench_mapping.csv").display(),
        dir.join("bench_listing.csv").display(),
        out.join("bench_result.mat").display(),
        out.join("vrcts_bench").display(),
    ));
    std::fs::write(&cfg_path, staged).unwrap();
    cfg_path
}

/// An absolute `multiConfigResultsOutputFile` / `Dump_Directory` runs from ANY
/// cwd: the `bestNNWeight_<n>_` / `weights_` / `weightsDerivatives_` save
/// prefixes go onto the basename, never ahead of the directory. The legacy
/// glued them onto the whole string (`BagOfProcessors.cpp:463`,
/// `BLSTMNeuralNetwork.cpp:319-321`), so an absolute results key composed
/// `bestNNWeight_1_/abs/out.mat` and the exact path died with a bare
/// `os error 2` after the whole corpus fold -- IMPROVEMENTS.md's
/// "`saveWeights` glues the prefix to the WHOLE filename" entry, now FIXED.
/// Every artifact lands under `out`; nothing lands relative to the cwd.
#[test]
fn bench_runs_with_absolute_output_keys() {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let cfg_path = stage_bench_config_absolute(dir.path(), out.path());

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(cwd.path())
        .args(["bench", "--repeat=1", cfg_path.to_str().unwrap()])
        .output()
        .expect("failed to run speech binary");
    assert!(
        output.status.success(),
        "speech bench with absolute output keys exited non-zero: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(
        stdout.lines().filter(|l| l.starts_with("BENCH ")).count(),
        1,
        "expected exactly one BENCH line (--repeat=1), got:\n{stdout}"
    );

    for name in [
        "bench_result.mat",
        "bestNNWeight_1_bench_result.mat",
        "weights_bestNNWeight_1_bench_result.mat",
        "weightsDerivatives_bestNNWeight_1_bench_result.mat",
    ] {
        assert!(
            out.path().join(name).is_file(),
            "{name} must land in the results key's directory"
        );
    }
    for chan in [1, 2] {
        let xml = out
            .path()
            .join("vrcts_bench")
            .join(format!("prcts_excerpt_chan_{chan}.xml"));
        assert!(
            xml.is_file(),
            "{} must land in Dump_Directory",
            xml.display()
        );
    }
    assert_eq!(
        std::fs::read_dir(cwd.path()).unwrap().count(),
        0,
        "nothing may land relative to the cwd"
    );
}

/// The write that fails names its path: a results key in a directory that does
/// not exist dies with the composed path in the error, not a bare `os error 2`.
#[test]
fn bench_names_the_unwritable_output_path() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = stage_bench_config(dir.path());
    let missing = dir.path().join("missing").join("bench_result.mat");
    let mut staged = std::fs::read_to_string(&cfg_path).unwrap();
    staged.push_str(&format!(
        "multiConfigResultsOutputFile {}\n",
        missing.display()
    ));
    std::fs::write(&cfg_path, staged).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_speech"))
        .current_dir(dir.path())
        .args(["bench", "--repeat=1", cfg_path.to_str().unwrap()])
        .output()
        .expect("failed to run speech binary");
    assert!(
        !output.status.success(),
        "a results key under a missing directory must fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("missing/") && stderr.contains("bench_result.mat"),
        "the error must name the path it could not create: {stderr}"
    );
}
