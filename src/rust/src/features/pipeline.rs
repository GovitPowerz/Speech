//! Feature-pipeline parameter structs shared across the spectral/TDC stages.
//!
//! Ported from the legacy `BLSTMSpectralSegmenter` parameter derivation
//! (`BLSTMSpectralSegmenter.cpp:341-370` `getTDCParam`) and the `TimeDomainCorrel`
//! config seam. Task 9 introduces only [`TdcParams`] (the time-domain-correlation
//! window/lag/balance bundle consumed by `ltsv_tdc::get_pitch`); Task 11 adds
//! `FeatureConfig`/`SpectralParams` to this same module and keeps this definition
//! VERBATIM (the derivation of the field values lives in Task 11 -- here callers
//! construct it directly).

/// Time-domain-correlation window/lag/balance parameters.
///
/// `half_window` is the legacy `TDC_window_size` (round(window_sec*rate/2)), the
/// half-window handed to `get_sequence`; `full_window = 2*half_window+1` is the
/// windowed-frame width; `shift` is the frame stride in samples; `min_lag`/`max_lag`
/// are the autocorrelation lag bounds in samples; `coeffs` is the optional windowing
/// vector (length `full_window`); `balance` weighs MaxPeak vs CrossCorr in the score.
#[derive(Debug, Clone)]
pub struct TdcParams {
    pub half_window: usize,
    pub full_window: usize,
    pub shift: usize,
    pub min_lag: usize,
    pub max_lag: usize,
    pub coeffs: Option<Vec<f64>>,
    pub balance: f64,
}
