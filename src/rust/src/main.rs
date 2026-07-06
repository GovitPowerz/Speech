//! `speech` CLI entry - mirrors the legacy `fsp` invocation.

use speech::cli::Mode;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).and_then(|f| Mode::from_flag(f));
    match mode {
        Some(mode) => {
            // Phase 0+: build CorpusProcessor and dispatch on `mode`.
            eprintln!(
                "speech {} - mode {:?} verbose={} (engine not yet implemented)",
                speech::version(),
                mode.kind,
                mode.verbose
            );
        }
        None => {
            eprintln!("usage: speech <-s|-i|-t|-m> <config.toml>...");
            std::process::exit(2);
        }
    }
}
