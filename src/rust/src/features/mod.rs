//! Acoustic feature extraction: FFT, Mel/MFCC, LTSV, time-domain correlation.
//!
//! Ported from legacy C++: fft.hpp, MelFilterBank.*, LongTermSpectralVariation.*,
//! TimeDomainCorrel.*, InputStatistics.*.

pub mod fft;
pub mod ltsv_tdc;
pub mod mel;
pub mod pipeline;
pub mod stats;
