//! Phase 2 Task 10: the END-TO-END real-net gate (the phase acceptance crux).
//!
//! Phase 1 features -> real trained network -> posteriors. The full Rust chain runs
//! under the REAL `1_worker_1.config` (BLSTM prefix) and the real
//! `NNweights_config1.bin`:
//!   read_audio(excerpt, 0.35, 2.0, 0) -> preemph per config (SKIPPED: ratio -0.97 < 0)
//!   -> SpectralParams::derive(BLSTM keys) -> windowing -> periodogram
//!   -> mel + DCT per config (nb_bins 20, nb_DCT 4, IgnoreFirstDCT, deltas 5, dd 3)
//!   -> LTSV (SKIPPED: LTSVwindow 0 -> R == 0) -> assemble_input_sequence
//!   -> assert vs e2e_input.bin FIRST (revalidates Phase 1 under the REAL config)
//!   -> BlstmNetwork(real config + real bin) -> feed_forward_backward
//!      (a) plain full-sequence AND (b) overlap(25, 12) -> assert vs all e2e dumps.
//!
//! The assembled input width D == 11 (3*nb_DCT - 1 with IgnoreFirstDCT), != the net
//! input 23. D=11 < 23 exercises the feed_forward WIDTH TOLERANCE (the per-layer
//! topRows tolerance inside the LSTM/dense forward; the leftCols crop only fires when
//! D > 23). The forwards are canary-gated: bit-exact on the oracle libm, hybrid (<=4
//! ULP or 512*eps*scale absolute) elsewhere -- the k=23 input GEMM diverges from
//! ascending accumulation, so the goldens ARE the reimpl (assert_oracle_eq covers it).
//!
//! InputNormalizationType -1: the windowed FFB self-normalizes the input IN PLACE, so
//! the gate passes a MUTABLE copy of the assembled input to feed_forward_backward and
//! compares the OUTPUT/hidden states (the harness leg self-normalizes a copy the same
//! way; e2e_input.bin is the PRE-normalization assembled input, asserted before the
//! net runs).

mod common;

use std::path::PathBuf;

use ndarray::Array2;
use speech::audio::{Audio, read_audio};
use speech::features::pipeline::{FeatureConfig, SpectralParams, build_input_sequence};
use speech::legacy_config::parse_legacy_config;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

fn real_config_text() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/1_worker_1.config");
    std::fs::read_to_string(p).unwrap()
}

fn real_weights() -> Vec<f64> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/NNweights_config1.bin");
    speech::io::binary::read_weight_vector(&p).unwrap()
}

fn excerpt_audio(preemph_ratio: f64) -> Audio {
    let mut audio =
        read_audio(&common::fixture("excerpt_2ch_8k.wav"), 0.35, 2.0, 0, None).expect("decode");
    // legacy gate: `if (preemphRatio > 0)` (BLSTMSpectralSegmenter.cpp:216). The real
    // config's preemph_ratio is -0.97 < 0, so this is SKIPPED.
    if preemph_ratio > 0.0 {
        audio.apply_preemph(preemph_ratio);
    }
    // noise skipped: real config noise_seed -3 (< 0 -> not applied).
    audio
}

/// Run the Phase 1 feature pipeline under the REAL config on chan 1 (0-indexed 0),
/// mirroring the harness E2E leg exactly. Returns the assembled inputSeq (pre-norm).
fn assemble_real_input() -> Array2<f64> {
    let map = parse_legacy_config(&real_config_text());
    let c = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
    let audio = excerpt_audio(c.preemph_ratio);
    let rate = audio.sample_rate as f64;
    let s = SpectralParams::derive(&c, rate);
    let chan = 0; // chan 1, per the brief
    // Real config: LTSVwindow 0 -> R == 0 -> no LTSV column; the ltsv_half_window
    // >= 1 gate inside build_input_sequence already handles this (no branch needed).
    build_input_sequence(&audio, &c, &s, chan, None)
}

fn real_net() -> BlstmNetwork {
    let map = parse_legacy_config(&real_config_text());
    let cfg = BlstmConfig::from_legacy(&map, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.set_weights(&real_weights()).unwrap();
    net
}

#[test]
fn e2e_input_matches_oracle_under_real_config() {
    // Assert the assembled input FIRST: this revalidates the whole Phase 1 feature
    // front-end under the REAL config (not the four Phase 1 variant configs) for free.
    // D == 11 (measured), rows == 201.
    let input = assemble_real_input();
    assert_eq!(
        input.dim(),
        (201, 11),
        "real-config feature width D must be 11"
    );
    common::assert_oracle_eq(
        &input,
        &common::load_bin_phase2("e2e_input.bin"),
        "e2e_input",
    );
}

#[test]
fn e2e_full_sequence_forward_matches_oracle() {
    // Plain full-sequence forward (set_processing_type(false, false)), no targets. The
    // real net's InputNormalizationType -1 self-normalizes the input IN PLACE, so pass
    // a mutable copy of the assembled input. Output (50 x 1) + fwd/bwd hidden (50 x 24).
    let input = assemble_real_input();
    let mut net = real_net();
    net.set_processing_type(false, false);

    let out_rows = input.nrows() / 4; // fwd LSTM sub 4*1 -> 201/4 = 50
    let mut mut_input = input.clone();
    let mut output = Array2::<f64>::zeros((out_rows, 1));
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut mut_input, 4, 2, &mut output, &empty);

    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("e2e_out_full.bin"),
        "e2e_out_full",
    );
    common::assert_oracle_eq(
        &net.output_forward,
        &common::load_bin_phase2("e2e_fwd_full.bin"),
        "e2e_fwd_full",
    );
    common::assert_oracle_eq(
        &net.output_backward,
        &common::load_bin_phase2("e2e_bwd_full.bin"),
        "e2e_bwd_full",
    );
}

#[test]
fn e2e_overlap_forward_matches_oracle() {
    // OverLap forward (set_processing_type(true, true)) with EXPLICIT window_size 25,
    // window_shift 12 (getBLSTMParam derivation is Phase 2b). Output (50 x 1) + fwd/bwd
    // hidden (50 x 24). Self-norm in place -> pass a mutable copy.
    let input = assemble_real_input();
    let mut net = real_net();
    net.set_processing_type(true, true);

    let out_rows = input.nrows() / 4; // whole-BLSTM ratio 4 -> 201/4 = 50
    let mut mut_input = input.clone();
    let mut output = Array2::<f64>::zeros((out_rows, 1));
    let empty = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut mut_input, 25, 12, &mut output, &empty);

    common::assert_oracle_eq(
        &output,
        &common::load_bin_phase2("e2e_out_overlap.bin"),
        "e2e_out_overlap",
    );
    common::assert_oracle_eq(
        &net.output_forward,
        &common::load_bin_phase2("e2e_fwd_overlap.bin"),
        "e2e_fwd_overlap",
    );
    common::assert_oracle_eq(
        &net.output_backward,
        &common::load_bin_phase2("e2e_bwd_overlap.bin"),
        "e2e_bwd_overlap",
    );
}
