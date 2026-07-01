//! Importer for the legacy whitespace `.config` format.
//!
//! Ported from legacy C++: ConfigFile.*. Reproduces the `_`-rest-of-line
//! continuation, last-value-wins, and `val*count` repeat rules. Implemented Phase 0.

use indexmap::IndexMap;

/// Read a legacy `.config` into a flat, insertion-ordered key/value map.
pub fn read_legacy_config(text: &str) -> IndexMap<String, String> {
    todo!("Phase 0: port ConfigFile parsing")
}
