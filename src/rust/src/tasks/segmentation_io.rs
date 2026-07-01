//! Segment container + primitives (sanitize, suppress-short, padding), scoring,
//! VRCTS XML + STM/TRS loaders.
//!
//! Ported from legacy C++: Segmentation.*. Phase 0.

/// A labelled time span `[start, end)` in seconds.
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub label: i32,
}
