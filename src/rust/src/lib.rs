//! `speech` - fast SAD + spoken Language Identification engine.
//!
//! Rust port of the legacy C++ `FastSpeechProcessing` engine. The Python
//! optimizer/orchestrator drives this crate via the `speech-py` PyO3 bindings.
//! See CLAUDE.md and docs/superpowers/specs/2026-07-01-speech-repo-setup-design.md.
#![allow(dead_code, unused_variables)] // scaffolding: remove as modules land

pub mod audio;
pub mod cli;
pub mod config;
pub mod constants;
pub mod cost;
pub mod engine;
pub mod features;
pub mod io;
pub mod legacy_config;
pub mod nn;
pub mod tasks;

/// Crate version, sourced from Cargo metadata.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
