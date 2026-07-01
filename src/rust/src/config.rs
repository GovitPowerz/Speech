//! Canonical TOML configuration.
//!
//! Ported from legacy C++: ConfigFile.*, String.hpp. This is the minimal
//! scaffold; the full key schema lands in Phase 0.

use serde::Deserialize;

/// Top-level engine config (minimal scaffold subset).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub engine: EngineConfig,
}

/// `[engine]` section.
#[derive(Debug, Clone, Deserialize)]
pub struct EngineConfig {
    pub algo: String,
    #[serde(default = "one")]
    pub num_outer_threads: usize,
}

fn one() -> usize {
    1
}

impl Config {
    /// Parse a TOML document.
    pub fn from_toml_str(text: &str) -> Result<Config, toml::de::Error> {
        toml::from_str(text)
    }
}
