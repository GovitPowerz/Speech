//! Importer for the legacy whitespace .config format.
//!
//! Ported from legacy C++: ConfigFile.cpp. KEY value, last-wins, # comments.
//! The 0a keys do not use the _-continuation rule, so a simple split suffices.

use anyhow::{Result, bail};
use indexmap::IndexMap;

/// `read<double>` on one config token: a finite f64, or an error naming the key and the
/// text. Non-finite is malformed: Rust parses `nan`/`inf`/`1e400`, the legacy `operator>>`
/// rejects them.
pub(crate) fn parse_finite_f64(s: &str, key: &str) -> Result<f64> {
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        Ok(v) => bail!("cannot read '{s}' as f64 for '{key}': non-finite ({v})"),
        Err(e) => bail!("cannot read '{s}' as f64 for '{key}': {e}"),
    }
}

/// `conf.get<double>(name, default)`: missing key -> `default`; present but malformed or
/// non-finite -> error (issues #32, #47).
pub(crate) fn get_f64_default(
    map: &IndexMap<String, String>,
    key: &str,
    default: f64,
) -> Result<f64> {
    map.get(key)
        .map_or(Ok(default), |s| parse_finite_f64(s, key))
}

/// Parse legacy KEY value config text; last duplicate key wins, # lines skipped.
pub fn parse_legacy_config(text: &str) -> IndexMap<String, String> {
    let mut out = IndexMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.splitn(2, char::is_whitespace);
        let name = it.next().unwrap_or("");
        if name.starts_with('#') || name.is_empty() {
            continue;
        }
        let val = it.next().unwrap_or("").trim().to_string();
        out.insert(name.to_string(), val);
    }
    out
}
