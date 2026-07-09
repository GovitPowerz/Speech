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
fn load_config(path: &str) -> Result<IndexMap<String, String>> {
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
