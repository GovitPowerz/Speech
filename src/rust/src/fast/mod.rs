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
//! Task 2 lands `nn` (the f32 BLSTM forward core); Task 3 adds `pipeline`; Task 4 adds
//! `driver` (the algo-3 fast SAD; Task 5 extends it with the Mode-7 LID Twin).
//!
//! Phase 9 Task 6 adds `cells`: the f32 CAUSAL twins of the new `nn::cells` family
//! (`FastSlstm`/`FastMamba` + `FastCausalNet`), built around an explicit per-timestep
//! `step` kernel so the offline fast path and the Task-7 causal streaming session run
//! the SAME arithmetic (spec S4.1-S4.3). The phase-7 `nn`/`pipeline`/`driver` BLSTM
//! path is untouched by it -- the causal net is a sibling the driver dispatches to on
//! `Cell_Type` + `Direction`, never a change to `FastBlstm`.
//!
//! Phase 10 Task 7 adds `bicell`: the BIDIRECTIONAL f32 twins of those same cells
//! (`FastBiCell` -- forward stack + reversed stack + hcat + the shared per-row dense
//! chain, driven by a FRESH windowed-overlap loop over the already-shared span helpers).
//! `FastBlstm` and its `overlap_window_step` are BYTE-UNTOUCHED by it (spec S5, approach
//! A): they are the phase-8 streaming bit-identity kernels, and the phase-7/8 suites
//! passing WITHOUT EDITS is the proof. A bidirectional new cell has NO frame-streaming
//! twin (the reverse pass reads the whole sequence); the utterance-granular `stream_lid`
//! session runs it whole per push (issue #57).
//!
//! Phase 10 Task 5 adds `mel32`: the f32 mel/DCT/deltas twin of `features/mel.rs`,
//! which retires the phase-7 f64 widen bridge at BOTH `pipeline` sites (spec S4) --
//! the fast front-end is now f32 END TO END and the per-call `T x bins` f64 widen
//! buffer is gone. That moved every fast-vs-exact tolerance once, by design; the S4
//! sweep re-measured and re-pinned them (RESULTS.md), with boundary/argmax identity
//! unchanged (R1: a flip would have been a STOP, not a widening).

pub mod bicell;
pub mod cells;
pub mod driver;
pub mod mel32;
pub mod nn;
pub mod pipeline;
pub mod plan;
pub mod stream;
pub mod stream_lid;
