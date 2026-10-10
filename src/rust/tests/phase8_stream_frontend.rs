//! Phase 8 Task 2: `fast::stream::StreamFrontEnd` -- the SAD streaming feature
//! front-end, proven BIT-IDENTICAL to the offline fast `build_input_sequence` at any
//! chunking.
//!
//! The oracle is the offline fast path under frozen stats (spec S2: same kernels, only
//! chunking differs). Four legs:
//! - `frontend_rows_bit_equal_offline`: the staged frozen mono fixture pushed at 100 ms
//!   chunks -> concatenated feature rows vs `FastPipeline::build_input_sequence` on the
//!   same gained audio, bit-for-bit (the gain-only gate config).
//! - `frontend_chunk_invariance`: the same fixture at 20 ms / 100 ms / 1 s / 7 ms
//!   (non-divisor) chunks -> all four bit-identical to each other (R3: chunking changes
//!   timing, never arithmetic).
//! - `frontend_preemph_noise_bit_equal_offline`: a synthetic signal with pre-emphasis
//!   AND dither ACTIVE, streamed vs the REAL `apply_preemph`/`apply_noise` offline code
//!   (the carries pinned against offline, not a reimplementation).
//! - `preemph_carry_across_chunks`: a crafted 2-chunk signal where the boundary sample
//!   matters -- streaming (carry) == offline, and a no-carry concat != offline
//!   (non-vacuity: the carry is load-bearing).

mod common;

use ndarray::Array2;
use speech::audio::{Audio, read_audio};
use speech::fast::nn::FastMatrix;
use speech::fast::pipeline::FastPipeline;
use speech::fast::stream::StreamFrontEnd;
use speech::features::pipeline::{FeatureConfig, SpectralParams};

/// Parse a legacy config text into the (FeatureConfig, SpectralParams) the spectral fast
/// path uses (prefix "BLSTM"), at `rate`.
fn cfg_params(text: &str, rate: f64) -> (FeatureConfig, SpectralParams) {
    let map = speech::legacy_config::parse_legacy_config(text);
    let cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
    let s = SpectralParams::derive(&cfg, rate);
    (cfg, s)
}

/// Stream `raw` through `fe` in fixed-size chunks, concatenating every `take_ready_rows`
/// return plus the `flush` tail into one `FastMatrix`.
fn stream_all(
    fe: &mut StreamFrontEnd,
    pipe: &mut FastPipeline,
    raw: &[f32],
    chunk: usize,
) -> FastMatrix {
    let mut data: Vec<f32> = Vec::new();
    let mut cols = 0usize;
    let append = |m: FastMatrix, data: &mut Vec<f32>, cols: &mut usize| {
        if m.rows == 0 {
            return;
        }
        if *cols == 0 {
            *cols = m.cols;
        }
        assert_eq!(*cols, m.cols, "emitted column count must be stable");
        data.extend_from_slice(&m.data);
    };

    let mut i = 0;
    while i < raw.len() {
        let end = (i + chunk).min(raw.len());
        fe.push(&raw[i..end]);
        let ready = fe.take_ready_rows(pipe);
        append(ready, &mut data, &mut cols);
        i = end;
    }
    let tail = fe.flush(pipe);
    append(tail, &mut data, &mut cols);

    let rows = data.len().checked_div(cols).unwrap_or(0);
    FastMatrix { data, rows, cols }
}

/// Bit-for-bit f32 compare of two `FastMatrix` (streaming vs offline), first mismatch reported.
fn assert_fast_bits_eq(got: &FastMatrix, want: &FastMatrix, label: &str) {
    assert_eq!(got.rows, want.rows, "{label}: row count");
    assert_eq!(got.cols, want.cols, "{label}: col count");
    for r in 0..got.rows {
        for c in 0..got.cols {
            let a = got.get(r, c);
            let b = want.get(r, c);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: mismatch at [{r},{c}]: got 0x{:08x} ({a}) want 0x{:08x} ({b})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}

/// A deterministic f32 signal (values exactly what they are -- both offline and streaming
/// start from this same array, so the bit-equal contract needs no "nice" values).
fn synthetic_raw(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| {
            let s = ((k as f32) * 0.017).sin() * 0.3 + ((k as f32) * 0.11).sin() * 0.15;
            s + (((k * 2654435761) & 0x3ff) as f32 / 1024.0 - 0.5) * 0.02
        })
        .collect()
}

// ---------------------------------------------------------------------------
// (1) frozen mono fixture, 100 ms chunks, vs offline.
// ---------------------------------------------------------------------------

#[test]
fn frontend_rows_bit_equal_offline() {
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let frozen_text = std::fs::read_to_string(&stage.config_path).unwrap();

    // Offline reference: read_audio applies the fixed gain (f64), narrow channel 0 to f32,
    // build the whole feature sequence.
    let audio = read_audio(&stage.wav_path, 0.0, 3600.0, 0, Some(stage.fixed_gain)).unwrap();
    let rate = audio.sample_rate as f64;
    let (cfg, s) = cfg_params(&frozen_text, rate);
    let mut pipe_ref = FastPipeline::new(&s, &cfg, rate).unwrap();
    let samples_f32: Vec<f32> = audio.data.row(0).iter().map(|&x| x as f32).collect();
    let reference = pipe_ref.build_input_sequence(&samples_f32).clone();
    assert!(
        reference.rows > 100,
        "sanity: the 60 s fixture yields many frames"
    );
    assert_eq!(
        reference.cols, 11,
        "tier2 assembled width (3*4 - ignoreFirst)"
    );

    // Streaming: feed the RAW i16/32768 f32 (the front-end applies the gain), 100 ms chunks.
    let (_sr, _ch, i16s) = common::read_wav_pcm16(&stage.wav_path);
    let raw: Vec<f32> = i16s.iter().map(|&v| (v as f64 / 32768.0) as f32).collect();
    let mut pipe = FastPipeline::new(&s, &cfg, rate).unwrap();
    // preemph_ratio -0.97 <= 0 (inert) and noise off (seed -3) on the tier2 gate config.
    let mut fe =
        StreamFrontEnd::new(&s, &cfg, rate, stage.fixed_gain, 0.0, cfg.preemph_ratio).unwrap();
    let chunk = (0.1 * rate) as usize; // 100 ms
    let streamed = stream_all(&mut fe, &mut pipe, &raw, chunk);

    assert_fast_bits_eq(&streamed, &reference, "frozen-100ms-vs-offline");
}

// ---------------------------------------------------------------------------
// (2) chunk invariance: 20 ms / 100 ms / 1 s / 7 ms -> bit-identical.
// ---------------------------------------------------------------------------

#[test]
fn frontend_chunk_invariance() {
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let frozen_text = std::fs::read_to_string(&stage.config_path).unwrap();
    let audio = read_audio(&stage.wav_path, 0.0, 3600.0, 0, Some(stage.fixed_gain)).unwrap();
    let rate = audio.sample_rate as f64;
    let (cfg, s) = cfg_params(&frozen_text, rate);

    let (_sr, _ch, i16s) = common::read_wav_pcm16(&stage.wav_path);
    let raw: Vec<f32> = i16s.iter().map(|&v| (v as f64 / 32768.0) as f32).collect();

    let run = |chunk_sec: f64| {
        let mut pipe = FastPipeline::new(&s, &cfg, rate).unwrap();
        let mut fe =
            StreamFrontEnd::new(&s, &cfg, rate, stage.fixed_gain, 0.0, cfg.preemph_ratio).unwrap();
        let chunk = ((chunk_sec * rate) as usize).max(1);
        stream_all(&mut fe, &mut pipe, &raw, chunk)
    };

    let a = run(0.02); // 20 ms
    let b = run(0.1); // 100 ms
    let c = run(1.0); // 1 s
    let d = run(0.007); // 7 ms (non-divisor of the 80-sample shift)

    assert_fast_bits_eq(&a, &b, "20ms-vs-100ms");
    assert_fast_bits_eq(&a, &c, "20ms-vs-1s");
    assert_fast_bits_eq(&a, &d, "20ms-vs-7ms");
}

// ---------------------------------------------------------------------------
// (3) pre-emphasis AND dither active, vs the real offline apply_preemph/apply_noise.
// ---------------------------------------------------------------------------

/// Build the offline reference feature sequence for `raw` with gain 1 (inert), the REAL
/// `apply_preemph`/`apply_noise` (whole-signal), narrowed and run through the fast pipeline.
fn offline_ref_preemph_noise(
    pipe: &mut FastPipeline,
    raw: &[f32],
    preemph: f64,
    noise: f64,
) -> FastMatrix {
    let raw_f64: Vec<f64> = raw.iter().map(|&x| x as f64).collect();
    let arr = Array2::from_shape_vec((1, raw.len()), raw_f64).unwrap();
    let mut audio = Audio {
        sample_rate: 8000,
        data: arr.clone(),
        data_raw: arr,
        lang_index: -1,
        weight: 1.0,
        external_features: Vec::new(),
        periodogram: None,
        audio_file_name: String::new(),
        ref_seg_file_name: String::new(),
        audio_offset: 0.0,
    };
    // Offline order: preemph then noise (`FastSpectralSegmenter::get_segmentation`'s
    // gates), on the whole channel.
    if preemph > 0.0 {
        audio.apply_preemph(preemph);
    }
    if noise > 0.0 {
        audio.apply_noise(noise);
    }
    let samples: Vec<f32> = audio.data.row(0).iter().map(|&x| x as f32).collect();
    pipe.build_input_sequence(&samples).clone()
}

#[test]
fn frontend_preemph_noise_bit_equal_offline() {
    const PREEMPH: f64 = 0.97;
    const NOISE: f64 = 0.02;
    let rate = 8000.0;
    // Reuse tier2's framing/mel/delta config (its preemph/noise cfg values are irrelevant --
    // build_input_sequence never applies them; the explicit PREEMPH/NOISE drive both sides).
    let tier2 = std::fs::read_to_string(common::fixture_phase4a("tier2_spectral.config")).unwrap();
    let (cfg, s) = cfg_params(&tier2, rate);

    let raw = synthetic_raw(6000);

    let mut pipe_ref = FastPipeline::new(&s, &cfg, rate).unwrap();
    let reference = offline_ref_preemph_noise(&mut pipe_ref, &raw, PREEMPH, NOISE);
    assert!(reference.rows > 50, "sanity: enough feature rows");

    let mut pipe = FastPipeline::new(&s, &cfg, rate).unwrap();
    // gain 1.0 (inert), preemph + noise ACTIVE.
    let mut fe = StreamFrontEnd::new(&s, &cfg, rate, 1.0, NOISE, PREEMPH).unwrap();
    let streamed = stream_all(&mut fe, &mut pipe, &raw, 137); // odd chunk exercises the carries

    assert_fast_bits_eq(&streamed, &reference, "preemph+noise-vs-offline");
}

// ---------------------------------------------------------------------------
// (4) the pre-emphasis carry across a chunk boundary is load-bearing.
// ---------------------------------------------------------------------------

#[test]
fn preemph_carry_across_chunks() {
    const PREEMPH: f64 = 0.97;
    let rate = 8000.0;
    let tier2 = std::fs::read_to_string(common::fixture_phase4a("tier2_spectral.config")).unwrap();
    let (cfg, s) = cfg_params(&tier2, rate);

    let raw = synthetic_raw(5000);
    let split = 1500usize; // a mid-stream boundary whose sample is covered by real frames

    // Carry-correct offline reference: apply_preemph on the WHOLE signal (noise OFF, so the
    // preemph carry is the ONLY cross-chunk state).
    let mut pipe_ref = FastPipeline::new(&s, &cfg, rate).unwrap();
    let correct = offline_ref_preemph_noise(&mut pipe_ref, &raw, PREEMPH, 0.0);

    // Streaming in 2 chunks (with the carry) must reproduce it bit-for-bit.
    let mut pipe = FastPipeline::new(&s, &cfg, rate).unwrap();
    let mut fe = StreamFrontEnd::new(&s, &cfg, rate, 1.0, 0.0, PREEMPH).unwrap();
    let mut data: Vec<f32> = Vec::new();
    let mut cols = 0usize;
    for chunk in [&raw[0..split], &raw[split..]] {
        fe.push(chunk);
        let r = fe.take_ready_rows(&mut pipe);
        if r.rows > 0 {
            cols = r.cols;
            data.extend_from_slice(&r.data);
        }
    }
    let tail = fe.flush(&mut pipe);
    if tail.rows > 0 {
        cols = tail.cols;
        data.extend_from_slice(&tail.data);
    }
    let streamed = FastMatrix {
        rows: data.len() / cols,
        cols,
        data,
    };
    assert_fast_bits_eq(&streamed, &correct, "carry-2chunk-vs-offline");

    // Non-vacuity: a NO-CARRY computation (preemph applied to each chunk INDEPENDENTLY, so
    // chunk 2's first sample is not subtracted) framed as one stream must DIFFER from the
    // correct reference -- proving the boundary sample genuinely matters.
    let preemph_chunk = |chunk: &[f32]| -> Vec<f64> {
        let f: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
        let arr = Array2::from_shape_vec((1, f.len()), f).unwrap();
        let mut a = Audio {
            sample_rate: 8000,
            data: arr.clone(),
            data_raw: arr,
            lang_index: -1,
            weight: 1.0,
            external_features: Vec::new(),
            periodogram: None,
            audio_file_name: String::new(),
            ref_seg_file_name: String::new(),
            audio_offset: 0.0,
        };
        a.apply_preemph(PREEMPH);
        a.data.row(0).to_vec()
    };
    let mut no_carry_f64 = preemph_chunk(&raw[0..split]);
    no_carry_f64.extend(preemph_chunk(&raw[split..]));
    let no_carry_f32: Vec<f32> = no_carry_f64.iter().map(|&x| x as f32).collect();
    let mut pipe2 = FastPipeline::new(&s, &cfg, rate).unwrap();
    let no_carry = pipe2.build_input_sequence(&no_carry_f32).clone();

    assert_eq!(no_carry.rows, correct.rows, "same frame count");
    let mut differs = false;
    for (a, b) in no_carry.data.iter().zip(correct.data.iter()) {
        if a.to_bits() != b.to_bits() {
            differs = true;
            break;
        }
    }
    assert!(
        differs,
        "the no-carry concat must DIFFER from the carry-correct reference (else the test is vacuous)"
    );
}
