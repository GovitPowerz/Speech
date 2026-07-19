//! Phase 8 Task 5 (spec S1.6/S1.8, `2026-07-19-phase-8-streaming-design.md`): THE PHASE
//! GATE. `StreamingSession` (fast/stream.rs) composes the three landed layers (T2
//! StreamFrontEnd, T3 StreamOverlap, T4 StreamDecision) over the phase-7 kernels; this
//! suite proves the streaming contract against the OFFLINE FAST PATH UNDER FROZEN STATS
//! (the oracle -- same kernels, only chunking differs, spec S2).
//!
//! THE GATE STRATEGY (adopted from the T1 review). TWO staged configs on the SAME staged
//! mono fixture (`common::stage_frozen_tier2` -- channel 0 of the committed 60 s stereo
//! `prcts_excerpt.wav` extracted to mono, the tuple-A pack, `Audio_fixed_gain` baked to the
//! channel's own self-norm adim, `InputNormalizationType 1`, `Inference_Path fast`):
//!   - PRIMARY (tuple-A frozen tail): the posterior-history equivalence + chunking
//!     bit-invariance legs are NON-DEGENERATE here (the posteriors vary), but the boundary
//!     set is THIN (the tuple-A net trained under self-norm saturates high under the frozen
//!     type-1 tail -> the untouched [Other@0, End@dur] 2-row seed, per
//!     `phase8_frozen_norm.rs`). VALID, asserted equal anyway.
//!   - CALIBRATED-TAIL (`stage_calibrated`): a SECOND config whose pack COPY has its
//!     type-1 normalize tail PATCHED to the fixture's own self-norm per-column mean/std
//!     (measured once here via the exact self-norm formulas). THIS IS A GATE-CONSTRUCTION
//!     CALIBRATION, NOT A TRAINED TAIL (the R4 posture) -- under it the frozen type-1 mode
//!     `(x-mean)/std` APPROXIMATES the self-norm `asinh((x-mean)/std)` and produces a REAL
//!     boundary set, so the streamed-vs-offline-frozen comparison exercises SEGMENTS + the
//!     latency accounting exercises real mid-stream emission lags.
//!
//! Both configs: `session.finish()` vs the offline fast bag run -- segment count/types
//! IDENTICAL, boundary max_dt EXACTLY 0.0 (the design predicts exact agreement, R1 STOP on
//! nonzero), the posterior histories BIT-IDENTICAL. Plus chunking bit-invariance (20/100/
//! 1000/7 ms), prefix consistency (incl. a near-reach StreamDecision profile), the four
//! validation bails, and the DERIVED latency pins.

mod common;

use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use speech::audio::read_audio;
use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::{BagOfProcessors, Processor};
use speech::fast::driver::build_aligned_spec;
use speech::fast::nn::FastMatrix;
use speech::fast::pipeline::FastPipeline;
use speech::fast::stream::{EmittedSegment, StreamDecision, StreamingSession};
use speech::features::pipeline::{FeatureConfig, SpectralParams};
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmenter::{DriverConfig, SegmenterConfig};

fn image_mode() -> Mode {
    Mode {
        kind: ModeKind::Image,
        verbose: false,
    }
}

fn parse(text: &str) -> IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(text)
}

// ---------------------------------------------------------------------------
// Oracle + session drivers.
// ---------------------------------------------------------------------------

/// The offline fast bag run under frozen stats -- the oracle (`fast/driver.rs` via
/// `run_get_segmentation`), returning (channel-0 posterior row, final Segmentation).
/// Mirrors `phase8_frozen_norm.rs::run_fast`.
fn run_offline_fast(
    map: &mut IndexMap<String, String>,
    wav: &Path,
    gain: Option<f64>,
) -> (Vec<f64>, Segmentation) {
    let mut bag = BagOfProcessors::from_configs(std::slice::from_mut(map), image_mode()).unwrap();
    let mut audio = read_audio(wav, 0.0, 3600.0, 0, gain).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut seg = vec![Segmentation::new(dur)];
    bag.run_get_segmentation(0, &mut audio, &mut seg).unwrap();
    let row = match bag.processor(0) {
        Processor::FastSpectral(s) => s.last_result_rows()[0].clone(),
        _ => panic!("expected FastSpectral for the frozen staged config"),
    };
    (row, seg.remove(0))
}

/// Read the staged MONO wav's RAW samples (`i16/32768` as f32, PRE-gain -- the session's
/// front-end applies the frozen gain/preemph/noise itself). Returns (rate, samples).
fn mono_samples(wav: &Path) -> (f64, Vec<f32>) {
    let (rate, channels, samples) = common::read_wav_pcm16(wav);
    assert_eq!(channels, 1, "the streaming gate uses the staged MONO wav");
    let s: Vec<f32> = samples.iter().map(|&x| x as f32 / 32768.0).collect();
    (rate as f64, s)
}

/// One full streaming run at `chunk`-sample granularity: fresh session, push in chunks,
/// finish. Collects everything the gate + latency legs read back. Push (mid-stream) and
/// finish (EOS-drain) emissions are kept SEPARATE: mid-stream lags are the structural
/// streaming latency (pinned against the derived bound); the EOS-drain lags of the last
/// `~holdback` of segments are bounded by the trailing-silence duration (data-dependent,
/// reported, not structurally pinned -- at EOS the buffer drains).
struct SessionRun {
    push_emissions: Vec<EmittedSegment>,   // mid-stream (during push)
    finish_emissions: Vec<EmittedSegment>, // EOS drain (during finish)
    seg: Segmentation,
    posteriors: Vec<f32>,
    session_max_lag: f64, // the session's own overall max (push + finish), for the CLI
    session_mean_lag: f64,
    bound: f64,
    feature_reach: f64,
    nn_window: f64,
    conv_delay: f64,
    holdback: f64,
    emission_count: usize,
}

impl SessionRun {
    /// All emissions (push then finish), for the equivalence / chunking / prefix legs.
    fn all_emissions(&self) -> Vec<EmittedSegment> {
        let mut v = self.push_emissions.clone();
        v.extend(self.finish_emissions.clone());
        v
    }
}

/// Max per-emission lag (`emitted_at_audio_s - end_s`) over a set (`f64::NEG_INFINITY` for
/// an empty set).
fn max_lag_of(em: &[EmittedSegment]) -> f64 {
    em.iter()
        .map(|e| e.emitted_at_audio_s - e.end_s)
        .fold(f64::NEG_INFINITY, f64::max)
}

fn run_session(text: &str, rate: f64, samples: &[f32], chunk: usize) -> SessionRun {
    let map = parse(text);
    let mut sess = StreamingSession::new(&map, rate, 1).unwrap();
    let mut push_emissions = Vec::new();
    let mut i = 0;
    while i < samples.len() {
        let end = (i + chunk).min(samples.len());
        push_emissions.extend(sess.push(&samples[i..end]));
        i = end;
    }
    let (finish_emissions, seg) = sess.finish();
    SessionRun {
        posteriors: sess.posterior_history().to_vec(),
        session_max_lag: sess.max_lag_s(),
        session_mean_lag: sess.mean_lag_s(),
        bound: sess.derived_latency_bound_s(),
        feature_reach: sess.feature_reach_s(),
        nn_window: sess.nn_window_s(),
        conv_delay: sess.conv_delay_s(),
        holdback: sess.holdback_s(),
        emission_count: sess.emission_count(),
        push_emissions,
        finish_emissions,
        seg,
    }
}

// ---------------------------------------------------------------------------
// Comparison helpers (NaN-aware bit-equality).
// ---------------------------------------------------------------------------

/// Segment identities (begin bits + class) for bit-exact partition comparison.
fn seg_identities(seg: &Segmentation) -> Vec<(u64, i32)> {
    seg.segments()
        .iter()
        .map(|s| (s.begin.to_bits(), s.ty as i32))
        .collect()
}

/// Emission identities (begin/end bits + class) -- EXCLUDES `emitted_at_audio_s`, which is
/// TIMING (chunk-dependent), not arithmetic (S1.6b: chunking changes timing, never math).
fn emission_identities(em: &[EmittedSegment]) -> Vec<(u64, u64, i32)> {
    em.iter()
        .map(|e| (e.begin_s.to_bits(), e.end_s.to_bits(), e.class as i32))
        .collect()
}

/// Session posterior history (f32) vs the offline `last_result_rows` (f64): bit-identical
/// after widening, NaN-pattern-aware (the `0/0` uncovered overlap rows).
fn assert_posteriors_bit_equal(session: &[f32], offline: &[f64], label: &str) {
    assert_eq!(session.len(), offline.len(), "{label}: posterior length");
    for (i, (&s, &o)) in session.iter().zip(offline.iter()).enumerate() {
        let sf = s as f64;
        if sf.is_nan() || o.is_nan() {
            assert_eq!(sf.is_nan(), o.is_nan(), "{label}: NaN pattern at {i}");
        } else {
            assert_eq!(
                sf.to_bits(),
                o.to_bits(),
                "{label}: bits at {i} (s={sf} o={o})"
            );
        }
    }
}

/// Two f32 posterior histories bit-identical (NaN-aware) -- the chunking-invariance leg.
fn assert_f32_bit_equal(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.is_nan() || y.is_nan() {
            assert_eq!(x.is_nan(), y.is_nan(), "{label}: NaN pattern at {i}");
        } else {
            assert_eq!(x.to_bits(), y.to_bits(), "{label}: bits at {i}");
        }
    }
}

/// Streamed `finish()` Segmentation vs the offline Segmentation: count + types IDENTICAL,
/// boundary max_dt returned (the design predicts 0.0).
fn boundary_check(streamed: &Segmentation, offline: &Segmentation, label: &str) -> f64 {
    let a = streamed.segments();
    let b = offline.segments();
    assert_eq!(a.len(), b.len(), "{label}: segment count");
    let mut max_dt = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.ty, y.ty, "{label}: segment type");
        max_dt = max_dt.max((x.begin - y.begin).abs());
    }
    max_dt
}

// ---------------------------------------------------------------------------
// The calibrated-tail staging (gate-construction calibration, R4 posture).
// ---------------------------------------------------------------------------

/// Stage the CALIBRATED-TAIL variant: patch a COPY of the tuple-A pack's type-1 normalize
/// tail with the fixture's OWN self-norm per-column mean/std (measured here via the exact
/// `self_normalize` formulas), so the frozen type-1 mode approximates self-norm and
/// produces a REAL boundary set. Returns the calibrated config path. GATE-CONSTRUCTION
/// CALIBRATION, NOT A TRAINED TAIL (spec R4).
fn stage_calibrated(dir: &Path, base: &common::FrozenStage) -> PathBuf {
    let base_text = std::fs::read_to_string(&base.config_path).unwrap();
    let map = parse(&base_text);
    let feature_cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();

    // Build the input sequence exactly as the offline driver does: gained channel 0 ->
    // f32 -> FastPipeline (tier2 preemph/noise are no-ops, so no mutation before framing).
    let audio = read_audio(&base.wav_path, 0.0, 3600.0, 0, Some(base.fixed_gain)).unwrap();
    let rate = audio.sample_rate as f64;
    let params = SpectralParams::derive(&feature_cfg, rate);
    let mut pipeline = FastPipeline::new(&params, &feature_cfg, rate).unwrap();
    let samples: Vec<f32> = audio.data.row(0).iter().map(|&x| x as f32).collect();
    let input = pipeline.build_input_sequence(&samples).clone();
    let (t, c) = (input.rows, input.cols);
    let spec = build_aligned_spec(&map, "BLSTM").unwrap();
    // The net input width == the normalize-tail length; the assembled feature dim `c` is
    // NARROWER (11 vs 23 on tier2 -- the LSTM width-tolerates), so `external_normalize_f32`
    // touches only the FIRST `c` columns (`max_col = c.min(mean.len())`). We patch exactly
    // those `c` tail entries, at the `input_size`-based tail offsets.
    let input_size = spec.lstm_neuron_nb[0];
    assert!(
        c <= input_size,
        "assembled input dim {c} must fit the net input width {input_size}"
    );

    // Per-column self-norm mean/std over the `c` assembled columns (population mean; std =
    // sqrt((sumsq+1e-32)/rows) -- the exact `fast/nn.rs::self_normalize_f32` formulas, in f64).
    let mut mean = vec![0.0f64; c];
    let mut std = vec![0.0f64; c];
    for (col, m) in mean.iter_mut().enumerate() {
        let mut acc = 0.0;
        for row in 0..t {
            acc += input.get(row, col) as f64;
        }
        *m = acc / t as f64;
    }
    for (col, sd) in std.iter_mut().enumerate() {
        let mut acc = 0.0;
        for row in 0..t {
            let d = input.get(row, col) as f64 - mean[col];
            acc += d * d;
        }
        *sd = ((acc + 1e-32) / t as f64).sqrt();
    }

    // Patch the tuple-A pack's normalize tail: mean tail is `data[n-2*input_size ..
    // n-input_size]`, std tail is `data[n-input_size .. n]` (per `FastBlstm::from_flat`).
    // Only the first `c` entries of each are read by `external_normalize_f32`.
    let pack_path = common::fixture_phase0("NNweights_config1.bin");
    let (rows, cols, mut data) = speech::io::binary::read_matrix(&pack_path).unwrap();
    assert_eq!(cols, 1, "the weight pack is an N x 1 column vector");
    let n = data.len();
    for (col, (&m, &sd)) in mean.iter().zip(std.iter()).enumerate() {
        data[n - 2 * input_size + col] = m;
        data[n - input_size + col] = sd;
    }
    let patched = dir.join("calibrated_pack.bin");
    speech::io::binary::write_matrix(&patched, rows, cols, &data).unwrap();

    let cal_text = format!(
        "{base_text}\n\
# ==== calibrated-tail (gate-construction calibration, NOT a trained tail; spec R4) ====\n\
BLSTM_weightsFile {}\n",
        patched.display()
    );
    let cal_config = dir.join("calibrated_tier2.config");
    std::fs::write(&cal_config, cal_text).unwrap();
    cal_config
}

// ---------------------------------------------------------------------------
// (a) OFFLINE EQUIVALENCE -- primary (tuple-A frozen tail).
// ---------------------------------------------------------------------------

#[test]
fn stream_finish_equals_offline_frozen() {
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let text = std::fs::read_to_string(&stage.config_path).unwrap();
    let (rate, samples) = mono_samples(&stage.wav_path);

    let mut off_map = parse(&text);
    let (off_post, off_seg) =
        run_offline_fast(&mut off_map, &stage.wav_path, Some(stage.fixed_gain));

    let run = run_session(&text, rate, &samples, (0.1 * rate) as usize);

    let max_dt = boundary_check(&run.seg, &off_seg, "primary");
    println!(
        "MEASURE gate_primary: seg_rows={} boundary_max_dt={max_dt:.3e} post_len={} emissions={}",
        run.seg.segments().len(),
        off_post.len(),
        run.emission_count
    );
    assert_eq!(
        max_dt, 0.0,
        "primary: boundary max_dt must be EXACTLY 0.0 (R1 STOP on nonzero)"
    );
    assert_posteriors_bit_equal(&run.posteriors, &off_post, "primary posterior history");
}

// ---------------------------------------------------------------------------
// (a') OFFLINE EQUIVALENCE -- calibrated-tail (REAL boundary set).
// ---------------------------------------------------------------------------

#[test]
fn stream_finish_equals_offline_calibrated() {
    let dir = tempfile::tempdir().unwrap();
    let base = common::stage_frozen_tier2(dir.path());
    let cal_config = stage_calibrated(dir.path(), &base);
    let text = std::fs::read_to_string(&cal_config).unwrap();
    let (rate, samples) = mono_samples(&base.wav_path);

    let mut off_map = parse(&text);
    let (off_post, off_seg) = run_offline_fast(&mut off_map, &base.wav_path, Some(base.fixed_gain));

    let run = run_session(&text, rate, &samples, (0.1 * rate) as usize);

    let max_dt = boundary_check(&run.seg, &off_seg, "calibrated");
    println!(
        "MEASURE gate_calibrated: seg_rows={} boundary_max_dt={max_dt:.3e} post_len={} emissions={}",
        run.seg.segments().len(),
        off_post.len(),
        run.emission_count
    );
    assert_eq!(
        max_dt, 0.0,
        "calibrated: boundary max_dt must be EXACTLY 0.0 (R1 STOP on nonzero)"
    );
    assert_posteriors_bit_equal(&run.posteriors, &off_post, "calibrated posterior history");
    // The calibration's whole point: a REAL boundary set (not the 2-row seed).
    assert!(
        run.seg.segments().len() > 2,
        "calibrated tail must produce a REAL boundary set (got {} rows -- the seed)",
        run.seg.segments().len()
    );
}

// ---------------------------------------------------------------------------
// (b) CHUNKING BIT-INVARIANCE (20/100/1000/7 ms).
// ---------------------------------------------------------------------------

/// Run the 20/100/1000/7 ms chunkings on `text` and assert the four sessions' finish
/// Segmentations + posterior histories + emitted SETS are BIT-IDENTICAL to each other
/// (chunking changes timing, never arithmetic -- S1.6b).
fn check_chunk_invariance(text: &str, rate: f64, samples: &[f32], label: &str) {
    let chunks = [
        (0.020 * rate) as usize, // 160
        (0.100 * rate) as usize, // 800
        (1.000 * rate) as usize, // 8000
        (0.007 * rate) as usize, // 56 (non-divisor)
    ];
    let runs: Vec<SessionRun> = chunks
        .iter()
        .map(|&c| run_session(text, rate, samples, c))
        .collect();

    let ref_seg = seg_identities(&runs[0].seg);
    let ref_post = &runs[0].posteriors;
    let ref_em = emission_identities(&runs[0].all_emissions());
    println!(
        "MEASURE chunk_invariance[{label}]: seg_rows={} post_len={} emissions={}",
        runs[0].seg.segments().len(),
        ref_post.len(),
        ref_em.len()
    );
    for (k, run) in runs.iter().enumerate().skip(1) {
        assert_eq!(
            seg_identities(&run.seg),
            ref_seg,
            "[{label}] chunk {} vs chunk 0: finish Segmentation must be bit-identical",
            chunks[k]
        );
        assert_f32_bit_equal(
            &run.posteriors,
            ref_post,
            &format!("[{label}] chunk {} posterior history", chunks[k]),
        );
        assert_eq!(
            emission_identities(&run.all_emissions()),
            ref_em,
            "[{label}] chunk {} vs chunk 0: emitted SET (begin/end/class) must be bit-identical",
            chunks[k]
        );
    }
}

#[test]
fn chunking_bit_invariance() {
    let dir = tempfile::tempdir().unwrap();
    let base = common::stage_frozen_tier2(dir.path());
    let (rate, samples) = mono_samples(&base.wav_path);

    // PRIMARY (tuple-A frozen): the posterior-history leg is non-degenerate (posteriors
    // vary), the emitted set is thin (the 2-row seed -> 1 emission) but VALID (rider 1).
    let primary_text = std::fs::read_to_string(&base.config_path).unwrap();
    check_chunk_invariance(&primary_text, rate, &samples, "primary");

    // CALIBRATED tail: REAL segments, so posteriors + emitted set + segmentation are ALL
    // non-degenerate (the strongest invariance leg).
    let cal_config = stage_calibrated(dir.path(), &base);
    let cal_text = std::fs::read_to_string(&cal_config).unwrap();
    check_chunk_invariance(&cal_text, rate, &samples, "calibrated");
}

// ---------------------------------------------------------------------------
// (c) PREFIX CONSISTENCY e2e -- every mid-stream emission survives to finish().
// ---------------------------------------------------------------------------

#[test]
fn prefix_consistency_e2e() {
    let dir = tempfile::tempdir().unwrap();
    let base = common::stage_frozen_tier2(dir.path());
    let cal_config = stage_calibrated(dir.path(), &base);
    let text = std::fs::read_to_string(&cal_config).unwrap();
    let (rate, samples) = mono_samples(&base.wav_path);

    // Stream at 100 ms, tracking WHICH emissions land mid-stream (before finish).
    let map = parse(&text);
    let mut sess = StreamingSession::new(&map, rate, 1).unwrap();
    let chunk = (0.1 * rate) as usize;
    let mut midstream: Vec<EmittedSegment> = Vec::new();
    let mut i = 0;
    while i < samples.len() {
        let end = (i + chunk).min(samples.len());
        midstream.extend(sess.push(&samples[i..end]));
        i = end;
    }
    let (tail, seg) = sess.finish();
    let mut all = midstream.clone();
    all.extend(tail);

    // Every emission (begin/end/class) must appear as a segment in the final partition.
    let final_ids: std::collections::HashSet<(u64, u64, i32)> = seg
        .segments()
        .windows(2)
        .map(|w| (w[0].begin.to_bits(), w[1].begin.to_bits(), w[0].ty as i32))
        .collect();
    for e in &all {
        let id = (e.begin_s.to_bits(), e.end_s.to_bits(), e.class as i32);
        assert!(
            final_ids.contains(&id),
            "emission {:?} not present in the final partition (retraction -- R2)",
            (e.begin_s, e.end_s, e.class)
        );
    }
    // The emitted set must be exactly the final partition (no missing, no extra).
    assert_eq!(
        emission_identities(&all).len(),
        final_ids.len(),
        "the emitted set must equal the final partition size"
    );
    println!(
        "MEASURE prefix_e2e: midstream={} total_emitted={} final_segments={}",
        midstream.len(),
        all.len(),
        final_ids.len()
    );
    assert!(
        !midstream.is_empty(),
        "non-vacuity: mid-stream emissions must occur (else prefix consistency is untested)"
    );
}

// ---------------------------------------------------------------------------
// (c') PREFIX CONSISTENCY -- the near-reach StreamDecision profile (T4-review rider).
// ---------------------------------------------------------------------------

#[test]
fn prefix_consistency_near_reach_profile() {
    // The T4-review hardening rider: stress the holdback boundary with speech bursts spaced
    // JUST PAST the ~1.685 s true leftward smoothing reach (< the 2.28957 s conservative
    // holdback), where a too-small holdback WOULD retract. Crafted posteriors through a
    // StreamDecision built with the tier2 seg_cfg (rising 0.7518, falling 0.3742, dt 0.04).
    let text = std::fs::read_to_string(common::fixture_phase4a("tier2_spectral.config")).unwrap();
    let map = parse(&text);
    let seg_cfg = SegmenterConfig::from_config(&map, "BLSTM").unwrap();
    let driver_cfg = DriverConfig::from_config(&map, "BLSTM").unwrap();
    let dt = 0.04; // spectrum_shift 0.01 * ssr 4
    let off = dt / 2.0 - 0.01 / 2.0;

    // Speech bursts of 30 rows (1.2 s) separated by 55-row (2.2 s) silence gaps -- gaps
    // TIGHTER than the T4 prefix profile's 6 s but wide enough to survive the ~1.87 s
    // smoothing padding as SEPARATE segments, so the holdback boundary (2.28957 s) is
    // stressed: a mid-stream emission fires only ~holdback after its segment ends, with the
    // next burst arriving ~1 s later -- a too-small holdback would retract. 6 bursts + a
    // long trailing silence to let the tail settle.
    let mut profile: Vec<f32> = Vec::new();
    for _ in 0..6 {
        profile.extend(std::iter::repeat_n(0.98f32, 30));
        profile.extend(std::iter::repeat_n(0.02f32, 55));
    }
    profile.extend(std::iter::repeat_n(0.02f32, 60)); // long trailing silence -> settle

    let mut dec = StreamDecision::new(driver_cfg, seg_cfg, dt, off);
    // Stream one posterior row at a time (the finest granularity) + collect mid-stream.
    let mut midstream: Vec<EmittedSegment> = Vec::new();
    for &v in &profile {
        let row = FastMatrix {
            data: vec![v],
            rows: 1,
            cols: 1,
        };
        midstream.extend(dec.push_rows(&row));
    }
    let audio_dur = profile.len() as f64 * dt + off;
    let (tail, seg) = dec.flush(audio_dur);
    let mut all = midstream.clone();
    all.extend(tail);

    let final_ids: std::collections::HashSet<(u64, u64, i32)> = seg
        .segments()
        .windows(2)
        .map(|w| (w[0].begin.to_bits(), w[1].begin.to_bits(), w[0].ty as i32))
        .collect();
    for e in &all {
        let id = (e.begin_s.to_bits(), e.end_s.to_bits(), e.class as i32);
        assert!(
            final_ids.contains(&id),
            "near-reach: emission {:?} retracted (holdback too small -- R2)",
            (e.begin_s, e.end_s, e.class)
        );
    }
    println!(
        "MEASURE near_reach: midstream={} total={} final_segments={}",
        midstream.len(),
        all.len(),
        final_ids.len()
    );
    assert!(
        !midstream.is_empty(),
        "non-vacuity: the near-reach profile must emit mid-stream"
    );
    // The profile has ~8 speech bursts -> the partition must be non-trivial.
    assert!(
        seg.segments().len() > 4,
        "near-reach profile must produce multiple segments (got {})",
        seg.segments().len()
    );
}

// ---------------------------------------------------------------------------
// (d) VALIDATION BAILS (algo != 3, multi-channel, missing gain, type != 1).
// ---------------------------------------------------------------------------

/// The bail message from a session construction that MUST fail (StreamingSession is not
/// Debug -- it holds the fast kernels -- so `unwrap_err` is unavailable; match instead).
fn bail_msg(r: anyhow::Result<StreamingSession>) -> String {
    match r {
        Ok(_) => panic!("expected a validation bail, got Ok"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn validation_bails() {
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let text = std::fs::read_to_string(&stage.config_path).unwrap();
    let rate = 8000.0;

    // (1) algo != 3.
    let mut m = parse(&text);
    m.insert("Algo_choice".into(), "4".into());
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(msg.contains("algo 3"), "algo-4 bail message: {msg}");

    // (2) multi-channel (stereo).
    let m = parse(&text);
    let msg = bail_msg(StreamingSession::new(&m, rate, 2));
    assert!(msg.contains("mono only"), "stereo bail message: {msg}");

    // (3) Audio_fixed_gain absent.
    let mut m = parse(&text);
    m.shift_remove("Audio_fixed_gain");
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(
        msg.contains("Audio_fixed_gain"),
        "missing-gain bail message: {msg}"
    );

    // (4) InputNormalizationType != 1.
    let mut m = parse(&text);
    m.insert("BLSTM_InputNormalizationType".into(), "-1".into());
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(
        msg.contains("InputNormalizationType 1"),
        "type -1 bail message: {msg}"
    );

    // Sanity: the un-mutated config constructs cleanly.
    let m = parse(&text);
    assert!(StreamingSession::new(&m, rate, 1).is_ok());
}

// ---------------------------------------------------------------------------
// (d') THE NOISE-GATE MAPPING + HARD ASSERTION (rider 5, the T2-review gate-shape finding).
// ---------------------------------------------------------------------------

#[test]
fn noise_gate_seeded_nonnegative_constructs() {
    // The offline gate is `noise_seed > 0 ? apply_noise(noise_ratio) : nothing` (magnitude
    // unconditional on the seed); the front-end's gate is `noise_magnitude > 0.0`. The
    // session maps `noise_magnitude = noise_seed > 0 ? noise_ratio : 0.0`. With a SEEDING
    // config (`noise_seed > 0`) and a NON-NEGATIVE ratio, the shapes agree and construction
    // succeeds (tier2's own `noise_seed -3` never seeds, so this pins the seeding branch).
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let text = std::fs::read_to_string(&stage.config_path).unwrap();
    let mut m = parse(&text);
    m.insert("BLSTM_noise_seed".into(), "5".into());
    m.insert("BLSTM_noise_ratio".into(), "0.01".into());
    assert!(
        StreamingSession::new(&m, 8000.0, 1).is_ok(),
        "a seeding config with a non-negative noise_ratio must construct"
    );
}

#[test]
#[should_panic(expected = "noise_ratio must be >= 0 when seeding")]
fn noise_gate_seeded_negative_panics() {
    // The HARD ASSERTION: a SEEDING config (`noise_seed > 0`) with a NEGATIVE ratio would
    // make the offline `apply_noise(ratio)` (which still applies a negative magnitude) and
    // the front-end's `noise_magnitude > 0.0` gate (which would SKIP noise) DIVERGE in
    // shape -- caught loudly (T2-review gate-shape finding).
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let text = std::fs::read_to_string(&stage.config_path).unwrap();
    let mut m = parse(&text);
    m.insert("BLSTM_noise_seed".into(), "5".into());
    m.insert("BLSTM_noise_ratio".into(), "-0.01".into());
    let _ = StreamingSession::new(&m, 8000.0, 1);
}

// ---------------------------------------------------------------------------
// (e) LATENCY BOUNDS (the DERIVED structural budget + a measured area allowance).
// ---------------------------------------------------------------------------

#[test]
fn latency_bounds() {
    // The calibrated-tail variant (REAL segments -> real mid-stream emission lags). The
    // DERIVED structural bound = front-end reach + NN window finalization + convolution
    // half-width + the smoothing holdback, ALL config-derived (never a literal, the T4 spec
    // correction: the holdback ALONE is 2.28957 s; the design's 1.02 s is DEAD). The
    // data-dependent AREA term is measured on this fixture and pinned with headroom.
    //
    // MEASURED (Apple Silicon dev box, 60 s staged mono, calibrated tail, 100 ms chunks):
    //   feature_reach = 0.14400 s  ((reach 8 * shift 80 + half_window 512) / 8000)
    //   nn_window     = 3.26000 s  (2 * window_size 163 * spectrum_shift 0.01)
    //   conv_delay    = 0.36000 s  (conv_half 9 * time_step 0.04)
    //   holdback      = 2.28957 s  (sum(min_speech)+sum(min_silence)+sum(padding), clamped)
    //   DERIVED BOUND = 6.05357 s  (the config-derived structural floor)
    //   push (mid-stream) max lag = 16.80428 s, mean lag 8.64831 s over 13 emissions;
    //   the excess 10.75 s over the bound is the wait-for-trigger (inter-segment silence:
    //   the emit-on-next-raw-segment design; the global max inter-speech-end gap is
    //   14.30570 s); the EOS-drain (finish) max lag is 4.96767 s over 3 emissions.
    let dir = tempfile::tempdir().unwrap();
    let base = common::stage_frozen_tier2(dir.path());
    let cal_config = stage_calibrated(dir.path(), &base);
    let text = std::fs::read_to_string(&cal_config).unwrap();
    let (rate, samples) = mono_samples(&base.wav_path);

    let run = run_session(&text, rate, &samples, (0.1 * rate) as usize);

    // Cross-check the session's derived bound against a from-config recomputation (so the
    // derivation is COMPUTED, not a session-internal literal we trust blindly).
    let map = parse(&text);
    let feature_cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
    let params = SpectralParams::derive(&feature_cfg, rate);
    let seg_cfg = SegmenterConfig::from_config(&map, "BLSTM").unwrap();
    let driver_cfg = DriverConfig::from_config(&map, "BLSTM").unwrap();
    let spec = build_aligned_spec(&map, "BLSTM").unwrap();
    let ssr = spec.lstm_subsampling.iter().product::<usize>()
        * spec.output_subsampling.iter().product::<usize>();
    let ssif = params.shift_frames;
    let reach = feature_cfg.deltas_nb as usize + feature_cfg.dd_nb as usize;
    let half_window = params.window_size / 2;
    let re_feature = (reach * ssif + half_window) as f64 / rate;
    let mut ws = driver_cfg.window_shift_sec;
    let (window_size, _wsh, _no, _rv) = speech::tasks::sad::get_blstm_param(
        driver_cfg.window_size_sec,
        &mut ws,
        rate,
        ssif,
        ssr,
        &spec.lstm_subsampling,
        &spec.output_subsampling,
        0,
    );
    let re_nn = 2.0 * window_size as f64 * params.shift_sec;
    let conv_half = driver_cfg
        .conv_coeff
        .as_deref()
        .map_or(0, |c| if c.len() > 1 { (c.len() - 1) / 2 } else { 0 });
    let re_conv = conv_half as f64 * params.shift_sec * ssr as f64;
    let re_holdback = seg_cfg.min_speech.iter().sum::<f64>()
        + seg_cfg.min_silence.iter().sum::<f64>()
        + seg_cfg.padding.iter().sum::<f64>();
    let re_bound = re_feature + re_nn + re_conv + re_holdback;

    assert_eq!(
        run.feature_reach.to_bits(),
        re_feature.to_bits(),
        "feature reach derivation"
    );
    assert_eq!(
        run.nn_window.to_bits(),
        re_nn.to_bits(),
        "nn window derivation"
    );
    assert_eq!(
        run.conv_delay.to_bits(),
        re_conv.to_bits(),
        "conv delay derivation"
    );
    assert_eq!(
        run.holdback.to_bits(),
        re_holdback.to_bits(),
        "holdback derivation"
    );
    assert_eq!(
        run.bound.to_bits(),
        re_bound.to_bits(),
        "bound = sum of components"
    );

    // Per-emission lag = structural lookahead + WAIT-FOR-TRIGGER: the emit-on-next-raw-
    // segment design only re-smooths + emits when a NEW raw segment lands, so a settled
    // segment waits until the NEXT raw segment closes -- absorbing the inter-segment silence
    // into the lag (data-dependent, S1.8/R5; the "area term"). The DERIVED structural bound
    // is the guaranteed floor; the wait is bounded by the longest inter-speech-end gap in
    // the fixture. Decompose so the STRUCTURAL part stays tightly pinned.
    let push_max = max_lag_of(&run.push_emissions);
    let finish_max = max_lag_of(&run.finish_emissions);
    let segs = run.seg.segments();
    // The longest gap between consecutive SPEECH-segment ends (the max wait for a
    // triggering next raw segment). Speech ends are the boundaries of Speech segments.
    let speech_ends: Vec<f64> = segs
        .windows(2)
        .filter(|w| w[0].ty == SegClass::Speech)
        .map(|w| w[1].begin)
        .collect();
    let max_trigger_gap = speech_ends
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(0.0_f64, f64::max);
    // The residual after subtracting the structural floor + the measured trigger gap (the
    // hysteresis falling-area + the ssr grid slack + smoothing boundary shifts).
    let residual = push_max - run.bound - max_trigger_gap;
    println!(
        "MEASURE latency: feature_reach={:.5} nn_window={:.5} conv_delay={:.5} holdback={:.5} \
         bound={:.5} push_max_lag={push_max:.5} max_trigger_gap={max_trigger_gap:.5} \
         residual={residual:.5} finish_max_lag={finish_max:.5} session_max_lag={:.5} \
         session_mean_lag={:.5} push_emissions={} finish_emissions={} seg_rows={}",
        run.feature_reach,
        run.nn_window,
        run.conv_delay,
        run.holdback,
        run.bound,
        run.session_max_lag,
        run.session_mean_lag,
        run.push_emissions.len(),
        run.finish_emissions.len(),
        run.seg.segments().len()
    );

    // Non-vacuity: real mid-stream emissions with real lags.
    assert!(run.emission_count > 1, "latency leg needs real emissions");
    assert!(
        !run.push_emissions.is_empty(),
        "latency leg needs mid-stream (push) emissions to bound the structural latency"
    );
    // The full structural lookahead IS incurred mid-stream (the holdback alone < the lag).
    assert!(
        push_max > run.holdback,
        "mid-stream max lag {push_max} must exceed the holdback alone {} (the full \
         window+conv+reach lookahead is incurred on top)",
        run.holdback
    );

    // THE STRUCTURAL DECOMPOSITION PIN (meaningful): the max mid-stream lag is bounded by
    // the DERIVED structural bound + the longest inter-speech-end gap in the fixture (the
    // max wait-for-trigger, measured from the OUTPUT segmentation) + a small RESIDUAL
    // (hysteresis falling-area + ssr grid slack + smoothing boundary shifts). The measured
    // residual is -3.55 s (slack -- the max-lag emission's own gap 10.75 s is smaller than
    // the global max gap 14.31 s), so this holds comfortably; it catches a lag that grows
    // faster than the structure + fixture gaps explain (a >~4.5 s regression).
    const RESIDUAL_ALLOWANCE: f64 = 1.0;
    assert!(
        push_max <= run.bound + max_trigger_gap + RESIDUAL_ALLOWANCE,
        "mid-stream max lag {push_max} exceeds bound {} + max trigger gap {max_trigger_gap} + \
         residual {RESIDUAL_ALLOWANCE} (residual was {residual})",
        run.bound
    );

    // THE FIXTURE PIN (the brief's literal max-lag pin, S1.8): max observed lag <= derived
    // bound + a measured AREA allowance. AREA is dominated by the fixture's inter-segment
    // silence (the wait-for-trigger, FIXTURE-SPECIFIC per R5), MEASURED at ~10.75 s (push
    // max_lag 16.80 - bound 6.05) and pinned at 13 s with headroom. The EOS-drain (finish)
    // lags are strictly smaller here (finish_max 4.97 s, the fixture ends on speech, no
    // trailing silence), so this single pin bounds ALL emissions.
    const AREA_ALLOWANCE: f64 = 13.0;
    let observed_max = push_max.max(finish_max);
    assert!(
        observed_max <= run.bound + AREA_ALLOWANCE,
        "max observed lag {observed_max} exceeds derived structural bound {} + area allowance \
         {AREA_ALLOWANCE} (fixture-specific, dominated by the inter-segment silence)",
        run.bound
    );
}
