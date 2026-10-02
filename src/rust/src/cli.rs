//! CLI mode parsing, mirroring the legacy `fsp` flags.
//!
//! Ported from legacy C++: FastSpeechProcessing.cpp `main`.

use anyhow::{Result, bail};
use indexmap::IndexMap;

use crate::legacy_config::parse_legacy_config;

/// Run mode selected by the leading CLI flag, case dropped (the legacy `mode[1]`
/// letter carries both the mode AND its case, e.g. `T`/`M`/`I` gate the verbose
/// per-file log branches in `BagOfProcessors::SegmentationFunction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeKind {
    /// `-s` / `-S`: run one config over the corpus.
    Solo,
    /// `-i` / `-I`: solo run plus image/segmentation dump.
    Image,
    /// `-t` / `-T`: solo run plus finite-difference gradient unit test.
    UnitTest,
    /// `-m` / `-M`: multi-config run.
    Multi,
}

/// Run mode: the letter (`kind`) plus its case (`verbose`, legacy `isupper(mode[1])`).
/// The legacy passes the mode as a bare `char*` flag string and re-checks
/// `mode[1] == 'x'`/`'X'` all over the place; this struct is that char pre-decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mode {
    pub kind: ModeKind,
    pub verbose: bool,
}

impl Mode {
    /// Parse a leading flag into a [`Mode`]. Returns `None` for unknown flags.
    pub fn from_flag(flag: &str) -> Option<Mode> {
        let (kind, verbose) = match flag {
            "-s" => (ModeKind::Solo, false),
            "-S" => (ModeKind::Solo, true),
            "-i" => (ModeKind::Image, false),
            "-I" => (ModeKind::Image, true),
            "-t" => (ModeKind::UnitTest, false),
            "-T" => (ModeKind::UnitTest, true),
            "-m" => (ModeKind::Multi, false),
            "-M" => (ModeKind::Multi, true),
            _ => return None,
        };
        Some(Mode { kind, verbose })
    }
}

/// A fully parsed CLI invocation: the mode plus one parsed config map per
/// `-m`/`-M` config path (single-config modes always yield exactly one map).
#[derive(Debug, Clone)]
pub struct CliInvocation {
    pub mode: Mode,
    pub configs: Vec<IndexMap<String, String>>,
}

/// Port of `FastSpeechProcessing.cpp main` (`:31-64`)'s argument parsing (mode
/// dispatch, config load, `--key=val` overrides / multi-config collection).
/// `args` excludes `argv[0]` (the program name) -- `args[0]` is the mode flag.
///
/// Single-config modes (`s`/`S`/`t`/`T`/`i`/`I`, `:45-53`): the LAST arg is the
/// config file; args between the flag and the config are `--key=val` overrides,
/// applied left-to-right via set-or-remove (empty `val` REMOVES the key -- the
/// legacy usage text `:85` documents this, though `ConfigFile::set_val` itself
/// never erases; see IMPROVEMENTS.md `[phase4a] CLI empty-value override`).
///
/// Multi mode (`m`/`M`, `:54-58`): every arg after the flag is a config path,
/// no overrides.
///
/// Unknown mode (`:59-63`) or `args.len() < 2` (legacy `argc < 3`, `:37-40`)
/// is an error (usage + exit in the legacy; here, an `Err` for `main` to turn
/// into a usage print + exit).
pub fn parse_cli(args: &[String]) -> Result<CliInvocation> {
    if args.len() < 2 {
        bail!("too few arguments");
    }

    let flag = &args[0];
    let mode = Mode::from_flag(flag).ok_or_else(|| anyhow::anyhow!("unknown mode ({flag})"))?;

    let configs = match mode.kind {
        ModeKind::Solo | ModeKind::UnitTest | ModeKind::Image => {
            let config_path = &args[args.len() - 1];
            let mut map = load_config(config_path)?;
            for arg in &args[1..args.len() - 1] {
                apply_override(&mut map, arg)?;
            }
            vec![map]
        }
        ModeKind::Multi => {
            let mut maps = Vec::with_capacity(args.len() - 1);
            for config_path in &args[1..] {
                maps.push(load_config(config_path)?);
            }
            maps
        }
    };

    Ok(CliInvocation { mode, configs })
}

/// Load a config file, dispatching on extension: `.toml` -> the Phase 4c canonical
/// TOML config (`toml_config::toml_to_map`); anything else (incl. the legacy
/// extensionless / `.config` convention) -> the legacy whitespace importer
/// unchanged. Both paths collapse to the same `IndexMap<String, String>` shape.
///
/// `pub(crate)` (Phase 7 Task 1): `bench.rs::run_bench` reuses this exact
/// dispatch to load its own `<config>` argument, rather than re-implementing
/// the `.toml`-vs-legacy extension check a second time.
pub(crate) fn load_config(path: &str) -> Result<IndexMap<String, String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read config file '{path}': {e}"))?;
    if path.ends_with(".toml") {
        crate::toml_config::toml_to_map(&text)
            .map_err(|e| anyhow::anyhow!("invalid TOML config '{path}': {e:#}"))
    } else {
        Ok(parse_legacy_config(&text))
    }
}

/// Apply one `--key=val` override to `map` (legacy `:47-52`): `arg` must start
/// with `--` (legacy `check(argument[0].substr(0,2) == "--", ...)`); an empty
/// `val` removes `key` instead of setting it (`IndexMap::shift_remove` to keep
/// iteration order stable for the remaining keys).
fn apply_override(map: &mut IndexMap<String, String>, arg: &str) -> Result<()> {
    if !arg.starts_with("--") {
        bail!("invalid option name {arg}");
    }
    let rest = &arg[2..];
    let (key, val) = rest
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("invalid option (missing '='): {arg}"))?;
    if val.is_empty() {
        map.shift_remove(key);
    } else {
        map.insert(key.to_string(), val.to_string());
    }
    Ok(())
}

/// A parsed `speech bench` invocation. Port-only subcommand (Phase 7 Task 1,
/// the `--convert-config` precedent) -- NOT part of the legacy `fsp` mode-flag
/// grammar `parse_cli`/`Mode::from_flag` handle above; `main.rs` dispatches to
/// this parser on the literal `bench` first argument, before `parse_cli` ever
/// runs. `path` is the raw `--path` string, unvalidated here -- `bench::
/// BenchPath::parse` owns the `exact`/`fast` contract (Task 7 widened it from
/// `exact`-only) so this module stays bench-semantics-free (pure CLI token
/// shape only, matching `parse_cli`'s own scope).
#[derive(Debug, Clone)]
pub struct BenchInvocation {
    pub config: String,
    pub repeat: usize,
    pub path: String,
    /// `--json`: print one JSON document for the invocation (the ledger's
    /// `bench` payload, issue #20) instead of the `BENCH` text lines.
    pub json: bool,
    /// `--label=NAME`: the measurement's name in the JSON document; defaults
    /// to the config file's basename. A label is what identifies a bench
    /// recipe in the ledger, where a config path (a tempdir) never may.
    pub label: Option<String>,
}

/// Parse `speech bench [--repeat=N] [--path=exact|fast] [--json] [--label=NAME]
/// <config>` (the `bench` literal itself already consumed by the caller).
/// `--repeat` defaults to 1, `--path` defaults to `"exact"`; exactly one
/// non-flag argument (the config path) is required.
pub fn parse_bench_args(args: &[String]) -> Result<BenchInvocation> {
    let mut repeat: usize = 1;
    let mut path = "exact".to_string();
    let mut config: Option<String> = None;
    let mut json = false;
    let mut label: Option<String> = None;

    for arg in args {
        if let Some(rest) = arg.strip_prefix("--repeat=") {
            repeat = rest
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("invalid --repeat value '{rest}': {e}"))?;
        } else if let Some(rest) = arg.strip_prefix("--path=") {
            path = rest.to_string();
        } else if arg == "--json" {
            json = true;
        } else if let Some(rest) = arg.strip_prefix("--label=") {
            if rest.is_empty() {
                bail!("--label requires a non-empty value");
            }
            label = Some(rest.to_string());
        } else if arg.starts_with("--") {
            bail!("unknown bench option: {arg}");
        } else if config.is_some() {
            bail!("bench takes exactly one config path (got a second: {arg})");
        } else {
            config = Some(arg.clone());
        }
    }

    let config = config.ok_or_else(|| anyhow::anyhow!("bench requires a config path"))?;
    Ok(BenchInvocation {
        config,
        repeat,
        path,
        json,
        label,
    })
}

/// A parsed `speech stream` invocation. Port-only subcommand (Phase 8 Task 6,
/// the `bench` / `--convert-config` precedent) -- NOT part of the legacy `fsp`
/// mode-flag grammar; `main.rs` dispatches to this parser on the literal
/// `stream` first argument, before `parse_cli` ever runs. `config`/`wav` are
/// the two required positional paths (config then wav, in that order);
/// `chunk_ms` is the streaming push granularity in milliseconds.
#[derive(Debug, Clone)]
pub struct StreamInvocation {
    pub config: String,
    pub wav: String,
    pub chunk_ms: u64,
}

/// Parse `speech stream [--chunk-ms=N] <config> <wav>` (the `stream` literal
/// itself already consumed by the caller). `--chunk-ms` defaults to 100 (the
/// `--key=val` shape, matching `bench`'s `--repeat=`/`--path=`); exactly two
/// non-flag arguments are required, in `<config> <wav>` order.
pub fn parse_stream_args(args: &[String]) -> Result<StreamInvocation> {
    let mut chunk_ms: u64 = 100;
    let mut positionals: Vec<String> = Vec::new();

    for arg in args {
        if let Some(rest) = arg.strip_prefix("--chunk-ms=") {
            chunk_ms = rest
                .parse::<u64>()
                .map_err(|e| anyhow::anyhow!("invalid --chunk-ms value '{rest}': {e}"))?;
        } else if arg.starts_with("--") {
            bail!("unknown stream option: {arg}");
        } else {
            positionals.push(arg.clone());
        }
    }

    if chunk_ms == 0 {
        bail!("--chunk-ms must be > 0");
    }
    if positionals.len() != 2 {
        bail!(
            "stream takes exactly two positional args: <config> <wav> (got {})",
            positionals.len()
        );
    }
    Ok(StreamInvocation {
        config: positionals[0].clone(),
        wav: positionals[1].clone(),
        chunk_ms,
    })
}

/// A parsed `speech stream-lid` invocation. Port-only subcommand (Phase 10 Task 10,
/// the `bench`/`stream` precedent) -- NOT part of the legacy `fsp` mode-flag grammar;
/// `main.rs` dispatches to this parser on the literal `stream-lid` first argument,
/// before `parse_cli` ever runs. `config`/`input` are the two required positional
/// paths (config then the phSeq/cep utterances file, in that order); `lang` is the
/// eval target language index (the offline `audio.lang_index`, clamped into
/// `[0, class_nb)` by `fast::driver::FastTwinLid::lid_score_params` -- a negative
/// value, the default, targets class 0).
#[derive(Debug, Clone)]
pub struct StreamLidInvocation {
    pub config: String,
    pub input: String,
    pub lang: i32,
}

/// Parse `speech stream-lid [--lang=N] <config> <input>` (the `stream-lid` literal
/// itself already consumed by the caller). `--lang` defaults to -1 (the "no known
/// target" case, clamped to class 0 downstream, the same `--key=val` shape as
/// `stream`'s `--chunk-ms=`); exactly two non-flag arguments are required, in
/// `<config> <input>` order.
pub fn parse_stream_lid_args(args: &[String]) -> Result<StreamLidInvocation> {
    let mut lang: i32 = -1;
    let mut positionals: Vec<String> = Vec::new();

    for arg in args {
        if let Some(rest) = arg.strip_prefix("--lang=") {
            lang = rest
                .parse::<i32>()
                .map_err(|e| anyhow::anyhow!("invalid --lang value '{rest}': {e}"))?;
        } else if arg.starts_with("--") {
            bail!("unknown stream-lid option: {arg}");
        } else {
            positionals.push(arg.clone());
        }
    }

    if positionals.len() != 2 {
        bail!(
            "stream-lid takes exactly two positional args: <config> <input> (got {})",
            positionals.len()
        );
    }
    Ok(StreamLidInvocation {
        config: positionals[0].clone(),
        input: positionals[1].clone(),
        lang,
    })
}
