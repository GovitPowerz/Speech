//! Top-level driver: build corpus + processors, rayon `par_iter` over files,
//! epoch training loop, finite-diff grad check, result reduction + `.mat` write.
//!
//! Ported from legacy C++: CorpusProcessor.* (replaces the OpenMP parallel-for).
