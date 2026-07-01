//! Generic stacked-layer container over a `Layer` trait; sub-sample decimation.
//!
//! Ported from legacy C++: NeuralNetwork.hpp. Phase 2.

/// A network layer: forward (and, when trainable, backward) over a sequence.
pub trait Layer {}
