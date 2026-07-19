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

    // Port-only tooling (Phase 7 Task 1, `fast` wired in Task 7), NOT a legacy
    // CLI surface: the wall-clock/RTF/peak-RSS bench harness. `speech bench
    // [--repeat=N] [--path=exact|fast] <config>`. Handled before `parse_cli`
    // (own arg grammar, not a legacy mode flag), same precedent as
    // `--convert-config` above.
    if args.len() >= 2 && args[1] == "bench" {
        let invocation = match speech::cli::parse_bench_args(&args[2..]) {
            Ok(inv) => inv,
            Err(e) => {
                eprintln!("Error: {e}\n");
                eprintln!("Usage : {progname} bench [--repeat=N] [--path=exact|fast] config_file");
                std::process::exit(2);
            }
        };
        let path = match speech::bench::BenchPath::parse(&invocation.path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error: {e}\n");
                eprintln!("Usage : {progname} bench [--repeat=N] [--path=exact|fast] config_file");
                std::process::exit(2);
            }
        };
        let report = match speech::bench::run_bench(&[invocation.config], invocation.repeat, path) {
            Ok(r) => r,
            Err(e) => {
                // `{e:#}` (the full anyhow cause chain, the same convention
                // `speech-py/src/lib.rs` uses at its PyO3 error seam) --
                // bench staging mistakes are usually several `.context()`
                // layers deep (config parse -> corpus listing -> per-file
                // read_audio), and the bare top-level message alone is
                // rarely enough to diagnose which layer failed.
                eprintln!("Error: {e:#}");
                std::process::exit(1);
            }
        };
        for run in &report.runs {
            println!(
                "BENCH path={} wall_s={:.6} audio_s={:.6} rtf={:.6} maxrss_mb={:.3} files={}",
                run.path.as_str(),
                run.wall_s,
                run.audio_s,
                run.rtf,
                run.maxrss_mb,
                run.files
            );
        }
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
