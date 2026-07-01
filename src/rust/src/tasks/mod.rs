//! Task layer: segmentation/SAD, LID, decision logic, scoring, output.
//!
//! Ported from legacy C++: Segmenter.*, Segmentation.*, BLSTM*Segmenter.*,
//! BLSTMSpectralLID.*, TwinBLSTMSpectralLID.*, VRCTSpart.*.

pub mod lid;
pub mod sad;
pub mod segmentation_io;
pub mod segmenter;
pub mod vrcts;
