//! Task layer: segmentation/SAD, LID, decision logic, scoring, output.
//!
//! Ported from legacy C++: Segmenter.*, Segmentation.*, BLSTM*Segmenter.*,
//! LongTermSpectralVariation.*, TimeDomainCorrel.*, BLSTMSpectralLID.*,
//! TwinBLSTMSpectralLID.*, VRCTSpart.*.
//!
//! Status: `segmentation` (container incl. `clear_hypothesis`), `segmentation_io`
//! (VRCTS/STM/CSV I/O + `compute_errors`), `segmenter` (decision primitives +
//! `DriverConfig`/`results_to_segmentation` + the `Segmenter` trait), and `sad`
//! (the four per-file drivers, Algo 1-4, incl. the Algo 3 pitch second pass) are
//! implemented and golden-tested (Phases 0b + 2b). `lid` (Algo 5/6 TwinBLSTM) and
//! `vrcts` (Algo 0 external-tool adapter) remain stubs (Phase 4).

pub mod lid;
pub mod sad;
pub mod segmentation;
pub mod segmentation_io;
pub mod segmenter;
pub mod vrcts;
