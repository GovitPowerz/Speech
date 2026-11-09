//! Acoustic feature extraction: FFT, Mel/MFCC, LTSV, time-domain correlation.
//!
//! Ported from legacy C++: fft.hpp, MelFilterBank.*, LongTermSpectralVariation.*,
//! TimeDomainCorrel.*.

pub mod fft;
pub mod ltsv_tdc;
pub mod mel;
pub mod pipeline;
