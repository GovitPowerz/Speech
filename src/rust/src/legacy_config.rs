//! Importer for the legacy whitespace .config format.
//!
//! Ported from legacy C++: ConfigFile.cpp. KEY value, last-wins, # comments.
//! The 0a keys do not use the _-continuation rule, so a simple split suffices.

use indexmap::IndexMap;

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
