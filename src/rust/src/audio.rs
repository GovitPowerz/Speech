//! Audio container + I/O (load, normalize, dither, pre-emphasis, framing).
//!
//! Ported from legacy C++: AudioStruct.*. Uses `symphonia` when implemented (Phase 1).

/// Multi-channel audio buffer (channels x frames, f64).
pub struct Audio {
    pub sample_rate: u32,
    pub data: Vec<Vec<f64>>,
}
