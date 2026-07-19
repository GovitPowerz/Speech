//! Phase 7 fast inference path: f32 kernels beside the byte-untouched exact f64 tree.
//!
//! Approach A from the design spec (docs/superpowers/specs/
//! 2026-07-18-phase-7-efficient-binary-design.md, S0-S4): a NEW module tree that
//! shares the config / weight-pack / decision layers with the exact path but runs
//! the numeric heart (matmuls, activations) in `f32` on `faer` (pure-Rust SIMD, no
//! system BLAS) with preallocated, geometrically-grown workspaces.
//!
//! NUMERIC DIVERGENCE FROM THE EXACT PATH IS BY DESIGN (spec S4/R3). The fast path
//! narrows `f64 -> f32` once (after adim, at weight-load) and uses faer's blocked/
//! SIMD reduction order, which differs from the exact path's ascending-loop f64
//! accumulation (`nn/layers.rs::matmul_seq`). The resulting per-posterior deltas are
//! MEASURED and pinned with headroom in the phase-7 tests, and documented in
//! RESULTS.md -- NEVER IMPROVEMENTS.md (which tracks legacy-quirk debt, not this
//! deliberate f32/f64 split). No contorting the fast path toward bit-parity.
//!
//! Task 2 lands `nn` (the f32 BLSTM forward core); Task 3 adds `pipeline`; Tasks 4/5
//! add `driver`.

pub mod nn;
pub mod pipeline;
