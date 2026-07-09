//! `speech` CLI entry - mirrors the legacy `fsp` invocation.
//!
//! Ported from legacy C++: FastSpeechProcessing.cpp `main` (:31-87).

use speech::CorpusProcessor;
use speech::cli::parse_cli;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let progname = args
        .first()
        .and_then(|p| p.rsplit('/').next())
        .unwrap_or("speech");

    // Port-only tooling (Phase 4c Task 5), NOT a legacy CLI surface: convert a
    // legacy `.config` to the canonical TOML layout via `toml_config::map_to_toml`.
    // `speech --convert-config in.config out.toml`. Handled before `parse_cli`
    // since it has its own two-path arity, not the legacy mode-flag grammar.
    if args.len() == 4 && args[1] == "--convert-config" {
        let (in_path, out_path) = (&args[2], &args[3]);
        let text = match std::fs::read_to_string(in_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error: cannot read '{in_path}': {e}");
                std::process::exit(1);
            }
        };
        let map = speech::legacy_config::parse_legacy_config(&text);
        let toml_text = speech::toml_config::map_to_toml(&map);
        if let Err(e) = std::fs::write(out_path, &toml_text) {
            eprintln!("Error: cannot write '{out_path}': {e}");
            std::process::exit(1);
        }
        println!("converted {in_path} -> {out_path} ({} keys)", map.len());
        return;
    }

    // legacy: :43-64 pass argv[1..] (mode + overrides/configs) to the parser;
    // argv[0] (progname) is not part of the parsed slice.
    let invocation = match parse_cli(&args[1..]) {
        Ok(inv) => inv,
        Err(e) => {
            // legacy: :37-40/:59-63 usage + exit(1) on argc<3 or unknown mode.
            // This port exits 2 on any parse failure (the pre-existing stub's
            // contract) -- a deliberate deviation from the legacy's exit(1);
            // the exit code is not golden-bearing.
            eprintln!("Error: {e}\n");
            print_usage(progname);
            std::process::exit(2);
        }
    };

    let mut processor = match CorpusProcessor::new(invocation.configs, invocation.mode) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = processor.run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

/// Port of `print_usage` (`FastSpeechProcessing.cpp:78-87`), `.config` wording
/// (this port has no `ConfigFile`-vs-TOML ambiguity at the CLI layer -- the
/// legacy `.config` importer is the only format `parse_cli` reads).
fn print_usage(progname: &str) {
    eprintln!("\nUsage : {progname} mode [config_options] config_file\n");
    eprintln!(
        "    mode syntax: -s single config, -i single config with image output, -t single config for unit testing, -m multiple config"
    );
    eprintln!(
        "    mode syntax (verbose): -S single config, -I single config with image output, -T single config for unit testing, -M multiple config"
    );
    eprintln!("    config_options syntax: --<variable_name>=<variable_value>");
    eprintln!("    whitespace not allowed in variable names or values");
    eprintln!("    all config_file variables overwritten by config_options");
    eprintln!("    setting <variable_value> = \"\" removes the variable from the config");
    eprintln!("    repeated variables overwritten by last specified");
}
