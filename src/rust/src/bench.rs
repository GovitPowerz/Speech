//! `speech bench`: end-to-end wall-clock / real-time-factor / peak-RSS harness
//! over the compute path selected by `--path` (`exact` or, since Task 7,
//! `fast` -- the f32/SIMD counterpart Tasks 2-5 landed). Port-only tooling (the
//! `--convert-config` precedent, `main.rs`) -- no legacy source. This is the
//! Phase 7 "measure-then-pin" instrument: Task 1 established the exact-path
//! baseline; Task 7 wires `fast` in and measures the exact-vs-fast matrix
//! every pinned budget in `RESULTS.md` is derived from (design spec, Global
//! Constraints: "every tolerance and budget is MEASURED first, pinned with
//! headroom").
//!
//! `run_bench` performs NO config mutation EXCEPT ONE deliberate key: it
//! overlays `Inference_Path` onto a cloned copy of the loaded map, set to
//! `path.as_str()`, before constructing the `CorpusProcessor` -- every other
//! key runs exactly as the caller's config says. This is the entire point of
//! `--path`: ONE config, comparable exact-vs-fast, without hand-authoring two
//! near-duplicate files that could drift apart. `BenchRun::path` therefore
//! always reflects what actually ran, not merely what the config happened to
//! say on disk. A caller that wants a single-pass, unscored, forward-only
//! measurement (the Task 1 baseline recipe) still gets that by STAGING the
//! rest of the config accordingly (`Neural_Networks_BackPropagation_Epochs
//! 0`, no reference in the listing, `*_BackPropagationActivated false`) --
//! see `tests/phase7_bench.rs` and `RESULTS.md`'s "Phase 7 -- performance"
//! section for the staged recipe. `fast` additionally requires the config to
//! meet the fast drivers' own construction-time contract (algo 3 spectral SAD
//! or algo 6 Mode-7 LID Twin, `InputNormalizationType -1`/`0` as appropriate,
//! no pitch pass, ...; see `engine/bag_of_processors.rs` and `fast/driver.rs`)
//! -- an incompatible config bails loudly at `CorpusProcessor::new`, same as
//! any other fast-path construction error.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use indexmap::IndexMap;

use crate::cli::{Mode, ModeKind, load_config};
use crate::engine::bag_of_processors::{get_f64_default, get_i32_default};
use crate::engine::corpus::Corpus;
use crate::engine::corpus_processor::CorpusProcessor;

/// Compute path selector -- `Exact` (the byte-untouched f64 tree) or `Fast`
/// (Phase 7's f32/faer/realfft counterpart, Tasks 2-5). The string form
/// (`as_str`) is exactly the `Inference_Path` config value `run_bench`
/// overlays, so `BenchPath` and the engine's own dispatch key can never drift
/// apart (one literal pair, `"exact"`/`"fast"`, used on both sides).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchPath {
    Exact,
    Fast,
}

impl BenchPath {
    /// Parse a `--path` value. Unknown values are a usage error (only `exact`
    /// and `fast` are accepted).
    pub fn parse(s: &str) -> Result<BenchPath> {
        match s {
            "exact" => Ok(BenchPath::Exact),
            "fast" => Ok(BenchPath::Fast),
            other => bail!("unknown --path value '{other}' (expected 'exact' or 'fast')"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            BenchPath::Exact => "exact",
            BenchPath::Fast => "fast",
        }
    }
}

/// One measured run: one config, one repeat iteration.
#[derive(Debug, Clone)]
pub struct BenchRun {
    pub path: BenchPath,
    pub wall_s: f64,
    /// Sum of processed audio duration across ALL channels of every corpus
    /// file (per-channel summing, not per-file wall-clock duration) -- e.g. a
    /// 60 s stereo file contributes `120.0`, not `60.0`. See
    /// `corpus_audio_seconds` for the exact per-file formula.
    pub audio_s: f64,
    pub rtf: f64,
    pub maxrss_mb: f64,
    pub files: usize,
}

/// The full report over every (config, repeat) combination, in run order
/// (outer loop over `configs`, inner loop over `repeat`).
#[derive(Debug, Clone, Default)]
pub struct BenchReport {
    pub runs: Vec<BenchRun>,
}

/// Sum of per-file processed audio duration (ALL channels) -- `min(file
/// duration, Audio_max_duration)` per the same capping `audio::read_audio`
/// itself applies -- plus the corpus file count. A file_type-1/2 (phSeq/cep)
/// listing is supported identically: `read_audio` returns a populated
/// `Audio::data` (rows = channels, cols = frames) for every ported file_type,
/// so `channels * frames / sample_rate` is the right duration formula
/// regardless of input format (phSeq/cep ignore the max_duration cap
/// entirely, per their own `read_audio` branches -- reproduced here by
/// construction, not special-cased).
///
/// Computed OUTSIDE the timed region: a plain re-decode of each corpus file
/// through the SAME public `read_audio` the engine itself uses, independent
/// of the `CorpusProcessor` instance actually timed in `run_bench` below (so
/// this bookkeeping cost never pollutes `wall_s`).
fn corpus_audio_seconds(map: &IndexMap<String, String>) -> Result<(f64, usize)> {
    let corpus = Corpus::from_config(map)?;
    let offset = get_f64_default(map, "Audio_offset", 0.0)?;
    let duration_max = get_f64_default(map, "Audio_max_duration", 3.6e6)?;
    let file_type = get_i32_default(map, "File_Type", 0)?;

    let mut audio_s = 0.0;
    for idx in 0..corpus.nb_of_files() {
        let item = corpus.item(idx);
        let audio =
            crate::audio::read_audio(Path::new(&item.file_name), offset, duration_max, file_type)?;
        let (channels, frames) = audio.data.dim();
        audio_s += channels as f64 * frames as f64 / audio.sample_rate as f64;
    }
    Ok((audio_s, corpus.nb_of_files()))
}

/// Peak resident set size in MB, process-lifetime peak (`getrusage(RUSAGE_SELF)
/// .ru_maxrss`), NOT a delta since the call before it -- `getrusage` only ever
/// reports the running high-water mark. Documented unit divergence: macOS
/// reports `ru_maxrss` in BYTES, Linux in KILOBYTES; normalized to MB here by
/// platform `cfg` so the printed `maxrss_mb` is comparable across the two.
fn maxrss_mb() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0.0;
    }
    let raw = usage.ru_maxrss as f64;
    #[cfg(target_os = "macos")]
    {
        raw / (1024.0 * 1024.0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        raw / 1024.0
    }
}

/// Run `configs`, each `repeat` times, over `path`. Each iteration builds a
/// FRESH `CorpusProcessor` from a freshly cloned config map (mirroring one
/// real invocation start to finish) and times its `Mode::Image` `run()` call
/// -- the `-i` compute path (`CorpusProcessor::run` dispatches to `run_solo`
/// when the config's own `Neural_Networks_BackPropagation_Epochs`/
/// `..._Gradient_Check_Epsilon` are 0, matching an ordinary single-pass `-i`
/// invocation; `run_bench` does not force this, see the module doc). The
/// timed region (`wall_s`) starts AFTER `CorpusProcessor::new()` returns --
/// weight load and net construction are excluded, while the per-file
/// decode/feature-extraction/forward-pass work inside `run()` is 100%
/// covered. One [`BenchRun`] per (config, repeat) pair, in nested `configs` x
/// `repeat` order. `repeat == 0` is treated as `1` (at least one measurement).
///
/// COVERAGE (T1/T11 honest record): every committed `phase7_bench.rs` fixture is a
/// single-file listing (`files=1`, `repeat=1`), so the multi-file `audio_s`
/// summation in [`corpus_audio_seconds`] and the `repeat.max(1)` clamp / `repeat>1`
/// in-process accumulation are exercised only in ad-hoc local runs, not by a
/// committed test.
///
/// `Inference_Path` is overlaid onto the loaded map to `path.as_str()` ONCE
/// per config, before `corpus_audio_seconds`/the repeat loop (see the module
/// doc's "ONE deliberate key" note) -- `corpus_audio_seconds` itself never
/// dispatches through the fast/exact processors, so the overlay's ordering
/// relative to it is a don't-care; it is placed first only to compute the
/// duration off the SAME map the timed runs use.
pub fn run_bench(configs: &[String], repeat: usize, path: BenchPath) -> Result<BenchReport> {
    let repeat = repeat.max(1);
    let mut runs = Vec::with_capacity(configs.len() * repeat);

    for config_path in configs {
        let mut map = load_config(config_path)?;
        map.insert("Inference_Path".to_string(), path.as_str().to_string());
        let (audio_s, files) = corpus_audio_seconds(&map)?;

        for _ in 0..repeat {
            let mode = Mode {
                kind: ModeKind::Image,
                verbose: false,
            };
            let mut processor = CorpusProcessor::new(vec![map.clone()], mode)?;

            let start = Instant::now();
            processor.run()?;
            let wall_s = start.elapsed().as_secs_f64();

            let rtf = if audio_s > 0.0 { wall_s / audio_s } else { 0.0 };
            runs.push(BenchRun {
                path,
                wall_s,
                audio_s,
                rtf,
                maxrss_mb: maxrss_mb(),
                files,
            });
        }
    }

    Ok(BenchReport { runs })
}
