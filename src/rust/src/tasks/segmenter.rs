//! `Segmenter` trait + decision logic (hysteresis-with-area, 7-step smoothing).
//!
//! Ported from legacy C++: Segmenter.*. Pure logic - golden-tested first (Phase 0).

/// Produces a [`Segmentation`](super::segmentation_io) from features/audio.
pub trait Segmenter {}
