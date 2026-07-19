//! `speech` - fast SAD + spoken Language Identification engine.
//!
//! Rust port of the legacy C++ `FastSpeechProcessing` engine. The Python
//! optimizer/orchestrator drives this crate via the `speech-py` PyO3 bindings.
//! See CLAUDE.md and docs/superpowers/specs/2026-07-01-speech-repo-setup-design.md.
//
// Crate-root scaffolding allow removed (Phase 0a): the still-stub modules
// (audio, engine, features, nn, tasks, cli, main) build clean under
// `-D warnings` as-is. If a future stub introduces genuine dead code /
// unused params, add a narrow `#[allow(...)]` on that item instead of
// reinstating a crate-wide blanket.

pub mod audio;
pub mod bench;
pub mod cli;
pub mod config;
pub mod constants;
pub mod cost;
pub mod engine;
pub mod fast;
pub mod features;
pub mod io;
pub mod legacy_config;
pub mod nn;
pub mod stream_cli;
pub mod tasks;
pub mod toml_config;

/// Convenience re-export: the top-level engine driver, used directly by `main.rs`.
pub use engine::corpus_processor::CorpusProcessor;
/// Convenience re-export: the `grad_check` seam result type (Phase 4c PyO3 surface).
pub use engine::corpus_processor::GradCheckReport;

/// Crate version, sourced from Cargo metadata.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
