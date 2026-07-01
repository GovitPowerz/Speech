//! Corpus + per-file records: parse language2classmapping + fileslisting CSVs,
//! class-balance coefficients.
//!
//! Ported from legacy C++: Corpus.*, CorpusItem.*.

/// One corpus entry (audio path + refs + language/dialect/class metadata).
pub struct CorpusItem {
    pub file_name: String,
    pub ref_seg: String,
    pub language: String,
    pub dialect: String,
    pub class_index: i32,
    pub weight: f64,
}
