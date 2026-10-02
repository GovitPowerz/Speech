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
pub mod stream_lid_cli;
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

/// Build provenance baked in by `build.rs`: the Cargo profile (`release` /
/// `debug`), the target triple and the `rustc --version` line. The ledger
/// (issue #20) records it with every number so a debug-profile measurement can
/// be refused and a cross-target one told apart.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct BuildInfo {
    pub profile: &'static str,
    pub target: &'static str,
    pub rustc: &'static str,
}

pub fn build_info() -> BuildInfo {
    BuildInfo {
        profile: env!("SPEECH_BUILD_PROFILE"),
        target: env!("SPEECH_BUILD_TARGET"),
        rustc: env!("SPEECH_BUILD_RUSTC"),
    }
}
