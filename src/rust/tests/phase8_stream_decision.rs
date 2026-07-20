//! Phase 8 Task 4: `fast::stream::StreamDecision` -- the incremental decision layer
//! (the 19-tap edge-truncating convolution + the latched hysteresis state machine +
//! re-smooth-and-emit-stable-prefix), proven equivalent to the OFFLINE decision layer
//! (`tasks::segmenter::results_to_segmentation` + `smooth_segmentation`) on the same
//! posterior rows.
//!
//! The oracle is the SHARED, UNCHANGED f64 decision code the fast offline driver hands
//! off to (`fast/driver.rs:411`): widen the posteriors to f64, seed a `Segmentation`,
//! `results_to_segmentation` (convolve-in-place then `update_segmentation` =
//! `update_segmentation_raw` + `smooth_segmentation`). Streaming defers, never
//! approximates -- so `flush()` must reproduce it BIT-FOR-BIT on the boundaries.
//!
//! Legs:
//! - `decision_final_equals_offline`: for the real gate config (tier2) + adversarial
//!   posterior profiles, streamed at several chunk granularities -> `flush()`'s
//!   `Segmentation` == the offline decision pipeline, boundary bits + types identical.
//! - `prefix_consistency_holds` (R2's catcher): every mid-stream `EmittedSegment`
//!   appears UNCHANGED (same begin/end bits + class) in the final segmentation -- no
//!   retraction. Non-vacuity: mid-stream emissions actually happen.
//! - `holdback_derived_not_hardcoded`: the holdback is the sum of the (clamped)
//!   smoothing thresholds; feeding a config with different padding moves it by the
//!   corresponding delta.
//! - `raw_hysteresis_matches_offline` (insurance): the streamed raw-segment list ==
//!   `update_segmentation_raw` on the (offline-convolved) rows, bit-for-bit -- the
//!   transcription is exact.
//! - `chunk_invariance`: the final segmentation + the emitted set are identical across
//!   chunk sizes (chunking changes timing, never arithmetic).

mod common;

use speech::fast::nn::FastMatrix;
use speech::fast::stream::{EmittedSegment, StreamDecision};
use speech::legacy_config::parse_legacy_config;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmenter::{
    DriverConfig, SegmenterConfig, results_to_segmentation, update_segmentation_raw,
};

// The gate timeStep/timeOffset (tier2: spectrum_shift 0.01, ssr 4 -> dt 0.04, off 0.015).
const DT: f64 = 0.04;
const OFF: f64 = 0.015;

/// Parse tier2_spectral.config -> (SegmenterConfig, DriverConfig) at the BLSTM prefix
/// (the real 19-tap hHCw convolution kernel + the real decision/padding thresholds).
fn gate_cfgs() -> (SegmenterConfig, DriverConfig) {
    let text = std::fs::read_to_string(common::fixture_phase4a("tier2_spectral.config")).unwrap();
    let map = parse_legacy_config(&text);
    let seg = SegmenterConfig::from_config(&map, "BLSTM").unwrap();
    let drv = DriverConfig::from_config(&map, "BLSTM").unwrap();
    (seg, drv)
}

/// A posterior profile as a list of (level, count) runs, flattened to f32 scalars.
/// Flat runs with distinct adjacent levels create single-frame threshold crossings
/// (well-defined `r - r_prev`), exactly what the hysteresis interpolation consumes.
fn runs(spec: &[(f32, usize)]) -> Vec<f32> {
    let mut v = Vec::new();
    for &(level, count) in spec {
        for _ in 0..count {
            v.push(level);
        }
    }
    v
}

/// Posterior scalars -> a `k x 1` FastMatrix (the SAD binary-net posterior column, the
/// only shape the decision layer consumes).
fn as_rows(scalars: &[f32]) -> FastMatrix {
    FastMatrix {
        data: scalars.to_vec(),
        rows: scalars.len(),
        cols: 1,
    }
}

/// Rows `[start, end)` of a `k x 1` FastMatrix.
fn slice_rows(m: &FastMatrix, start: usize, end: usize) -> FastMatrix {
    FastMatrix {
        data: m.data[start..end].to_vec(),
        rows: end - start,
        cols: 1,
    }
}

/// The OFFLINE decision oracle: widen the f32 posteriors to f64, seed a Segmentation at
/// `audio_dur`, and run the SHARED `results_to_segmentation` (convolve-in-place +
/// `update_segmentation_raw` + `smooth_segmentation`) exactly as `fast/driver.rs` does.
fn offline_final(
    scalars: &[f32],
    seg_cfg: &SegmenterConfig,
    drv: &DriverConfig,
    audio_dur: f64,
) -> Segmentation {
    let mut seg = Segmentation::new(audio_dur);
    let mut rv: Vec<f64> = scalars.iter().map(|&x| x as f64).collect();
    results_to_segmentation(
        &mut seg,
        DT,
        OFF,
        &mut rv,
        SegClass::Speech,
        drv.conv_coeff.as_deref(),
        seg_cfg,
    );
    seg
}

/// Stream `scalars` through a fresh StreamDecision in `chunk`-row pushes, collecting all
/// mid-stream emissions + the flush emissions + the final segmentation.
fn stream_all(
    scalars: &[f32],
    seg_cfg: &SegmenterConfig,
    drv: &DriverConfig,
    audio_dur: f64,
    chunk: usize,
) -> (Vec<EmittedSegment>, Segmentation) {
    let rows = as_rows(scalars);
    let mut dec = StreamDecision::new(drv.clone(), seg_cfg.clone(), DT, OFF);
    let mut emitted: Vec<EmittedSegment> = Vec::new();
    let mut i = 0;
    while i < rows.rows {
        let end = (i + chunk).min(rows.rows);
        emitted.extend(dec.push_rows(&slice_rows(&rows, i, end)));
        i = end;
    }
    let (tail, final_seg) = dec.flush(audio_dur);
    emitted.extend(tail);
    (emitted, final_seg)
}

/// Bit-for-bit Segmentation equality (begin bits + type + audio duration).
fn assert_seg_bits_eq(got: &Segmentation, want: &Segmentation, label: &str) {
    let a = got.segments();
    let b = want.segments();
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: segment count {} != {}",
        a.len(),
        b.len()
    );
    assert_eq!(
        got.audio_duration().to_bits(),
        want.audio_duration().to_bits(),
        "{label}: audio duration"
    );
    for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.begin.to_bits(),
            y.begin.to_bits(),
            "{label}: begin bits at {k}: {} vs {}",
            x.begin,
            y.begin
        );
        assert_eq!(x.ty, y.ty, "{label}: type at {k}: {:?} vs {:?}", x.ty, y.ty);
    }
}

/// The set of (begin bits, end bits, class) triples of the final partition's segments.
fn final_triples(seg: &Segmentation) -> Vec<(u64, u64, SegClass)> {
    let s = seg.segments();
    (0..s.len().saturating_sub(1))
        .map(|i| (s[i].begin.to_bits(), s[i + 1].begin.to_bits(), s[i].ty))
        .collect()
}

/// The adversarial + realistic posterior profiles: speech bursts of varied duration
/// (short ones the suppress removes) separated by gaps of varied width (short ones the
/// padding merges), including a start-edge burst and an open tail burst -- so the
/// convolution edges, the area gating, and the smoothing (suppress/pad composition) are
/// all exercised. The bursts are separated by LONG silences (`GAP` frames, wider than the
/// holdback + padding reach) so earlier segments settle and emit MID-STREAM (not only at
/// flush), which is what makes the prefix-consistency leg non-vacuous.
fn profiles() -> Vec<(&'static str, Vec<f32>)> {
    let lo = 0.05f32;
    let hi = 0.95f32;
    // A long silence (6 s at dt=0.04) -- comfortably past the ~2.29 s holdback + ~1.9 s
    // padding reach, so a burst before it settles once the next burst closes.
    const GAP: usize = 150;
    vec![
        // Clean well-separated bursts (each settles + emits as the next one closes).
        (
            "clean",
            runs(&[
                (lo, 25),
                (hi, 30),
                (lo, GAP),
                (hi, 30),
                (lo, GAP),
                (hi, 30),
                (lo, 40),
            ]),
        ),
        // Start-edge burst + a tiny middle burst (suppressed) + an OPEN tail burst (the
        // flush force-emit).
        (
            "edges",
            runs(&[(hi, 25), (lo, GAP), (hi, 4), (lo, GAP), (hi, 35)]),
        ),
        // Adversarial clusters (short gaps the padding merges + short bursts the suppress
        // removes) separated by long silences.
        (
            "adversarial",
            runs(&[
                (lo, 25),
                (hi, 25),
                (lo, 4),
                (hi, 25),
                (lo, GAP),
                (hi, 3),
                (lo, 4),
                (hi, 25),
                (lo, GAP),
                (hi, 30),
                (lo, 30),
            ]),
        ),
        // Threshold-hovering ramp (area gating decides begin/end) then clean bursts.
        (
            "hover",
            runs(&[
                (lo, 25),
                (0.6, 8),
                (hi, 25),
                (0.5, 6),
                (lo, GAP),
                (hi, 25),
                (lo, GAP),
                (hi, 25),
                (lo, 30),
            ]),
        ),
        // A dense sub-padding cluster (merges into one speech) then long silences.
        (
            "dense",
            runs(&[
                (lo, 20),
                (hi, 20),
                (lo, 6),
                (hi, 20),
                (lo, 6),
                (hi, 20),
                (lo, GAP),
                (hi, 20),
                (lo, GAP),
                (hi, 25),
                (lo, 30),
            ]),
        ),
    ]
}

// ---------------------------------------------------------------------------
// (1) flush() == the offline decision pipeline, bit-for-bit.
// ---------------------------------------------------------------------------

#[test]
fn decision_final_equals_offline() {
    let (seg_cfg, drv) = gate_cfgs();
    for (name, scalars) in profiles() {
        let audio_dur = DT * scalars.len() as f64 + OFF;
        let want = offline_final(&scalars, &seg_cfg, &drv, audio_dur);
        for &chunk in &[1usize, 3, 7, 40, scalars.len().max(1)] {
            let (_emitted, got) = stream_all(&scalars, &seg_cfg, &drv, audio_dur, chunk);
            assert_seg_bits_eq(
                &got,
                &want,
                &format!("final-equals-offline {name} chunk={chunk}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (2) prefix consistency: every mid-stream emission appears unchanged finally.
// ---------------------------------------------------------------------------

// COVERAGE NUANCE (Phase 8 battery item 5): the profiles here space bursts by ~6 s silences
// -- far past the true smoothing reach -- so they do NOT stress a small holdback cut: a 2x
// holdback halving still clears their gaps and this leg stays GREEN. The near-reach 2.2 s-gap
// profile (`phase8_gate.rs::prefix_consistency_near_reach_profile`) tightens the margin but
// likewise survives a 2x cut; the LOAD-BEARING catcher for an under-sized holdback is the
// calibrated real-fixture e2e leg (`phase8_gate.rs::prefix_consistency_e2e`), which retracts
// under the halving. See IMPROVEMENTS.md's Phase-8 battery item (5).
#[test]
fn prefix_consistency_holds() {
    let (seg_cfg, drv) = gate_cfgs();
    let mut total_midstream = 0usize;
    for (name, scalars) in profiles() {
        let audio_dur = DT * scalars.len() as f64 + OFF;
        for &chunk in &[1usize, 3, 7, 40] {
            let rows = as_rows(&scalars);
            let mut dec = StreamDecision::new(drv.clone(), seg_cfg.clone(), DT, OFF);
            let mut midstream: Vec<EmittedSegment> = Vec::new();
            let mut i = 0;
            while i < rows.rows {
                let end = (i + chunk).min(rows.rows);
                midstream.extend(dec.push_rows(&slice_rows(&rows, i, end)));
                i = end;
            }
            total_midstream += midstream.len();
            let (_tail, final_seg) = dec.flush(audio_dur);
            let allowed = final_triples(&final_seg);
            for e in &midstream {
                let triple = (e.begin_s.to_bits(), e.end_s.to_bits(), e.class);
                assert!(
                    allowed.contains(&triple),
                    "prefix consistency VIOLATED ({name} chunk={chunk}): mid-stream emission \
                     [{:.6}, {:.6}] {:?} not present unchanged in the final segmentation \
                     (a retraction -- R2)",
                    e.begin_s,
                    e.end_s,
                    e.class
                );
            }
        }
    }
    assert!(
        total_midstream > 0,
        "non-vacuity: the profiles must produce mid-stream emissions (else prefix \
         consistency is trivially satisfied)"
    );
}

// ---------------------------------------------------------------------------
// (3) the holdback is derived from the config params, not hardcoded.
// ---------------------------------------------------------------------------

#[test]
fn holdback_derived_not_hardcoded() {
    let (seg_cfg, drv) = gate_cfgs();

    // The derivation: the sum of every (clamped) smoothing-pipeline threshold -- the
    // conservative bound on how far left appended raw structure can reach.
    let expected = seg_cfg.min_speech.iter().sum::<f64>()
        + seg_cfg.min_silence.iter().sum::<f64>()
        + seg_cfg.padding.iter().sum::<f64>();
    let dec = StreamDecision::new(drv.clone(), seg_cfg.clone(), DT, OFF);
    assert_eq!(
        dec.holdback().to_bits(),
        expected.to_bits(),
        "holdback must equal the sum of the clamped smoothing thresholds (got {}, want {})",
        dec.holdback(),
        expected
    );

    // Feeding a config with a larger padding[0] moves the holdback by exactly that delta.
    let mut bigger = seg_cfg.clone();
    bigger.padding[0] += 0.5;
    let dec2 = StreamDecision::new(drv.clone(), bigger, DT, OFF);
    let delta = dec2.holdback() - dec.holdback();
    assert!(
        (delta - 0.5).abs() < 1e-12,
        "holdback must move with the padding param (delta {delta}, want 0.5)"
    );

    // And a config with all-zero thresholds gives a zero holdback (no smoothing reach).
    let mut zero = seg_cfg.clone();
    zero.min_speech = [0.0; 3];
    zero.min_silence = [0.0; 2];
    zero.padding = [0.0; 4];
    let dec3 = StreamDecision::new(drv, zero, DT, OFF);
    assert_eq!(dec3.holdback(), 0.0, "all-zero thresholds -> zero holdback");
}

// ---------------------------------------------------------------------------
// (4) the streamed raw-segment list == update_segmentation_raw (insurance).
// ---------------------------------------------------------------------------

#[test]
fn raw_hysteresis_matches_offline() {
    let (seg_cfg, drv) = gate_cfgs();
    for (name, scalars) in profiles() {
        let audio_dur = DT * scalars.len() as f64 + OFF;

        // Offline: convolve a copy exactly as results_to_segmentation does, then run the
        // batch update_segmentation_raw -> the (begin, end) raw segments.
        let mut rv: Vec<f64> = scalars.iter().map(|&x| x as f64).collect();
        if let Some(c) = drv.conv_coeff.as_deref()
            && c.len() > 1
        {
            speech::audio::convolution_horiz_slice(&mut rv, c);
        }
        let want: Vec<(f64, f64)> =
            update_segmentation_raw(&rv, SegClass::Speech, OFF, DT, &seg_cfg)
                .into_iter()
                .map(|(b, e, _)| (b, e))
                .collect();

        for &chunk in &[1usize, 7, scalars.len().max(1)] {
            let rows = as_rows(&scalars);
            let mut dec = StreamDecision::new(drv.clone(), seg_cfg.clone(), DT, OFF);
            let mut i = 0;
            while i < rows.rows {
                let end = (i + chunk).min(rows.rows);
                let _ = dec.push_rows(&slice_rows(&rows, i, end));
                i = end;
            }
            let (_t, _s) = dec.flush(audio_dur);
            let got = dec.raw_segments();
            assert_eq!(
                got.len(),
                want.len(),
                "raw-seg count {name} chunk={chunk}: {} != {}",
                got.len(),
                want.len()
            );
            for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.0.to_bits(),
                    w.0.to_bits(),
                    "raw begin {name} chunk={chunk} @ {k}"
                );
                assert_eq!(
                    g.1.to_bits(),
                    w.1.to_bits(),
                    "raw end {name} chunk={chunk} @ {k}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (5) chunk invariance: final segmentation + emitted set identical across chunkings.
// ---------------------------------------------------------------------------

#[test]
fn chunk_invariance() {
    let (seg_cfg, drv) = gate_cfgs();
    for (name, scalars) in profiles() {
        let audio_dur = DT * scalars.len() as f64 + OFF;
        let (em1, s1) = stream_all(&scalars, &seg_cfg, &drv, audio_dur, 1);
        for &chunk in &[3usize, 7, 40, scalars.len().max(1)] {
            let (emk, sk) = stream_all(&scalars, &seg_cfg, &drv, audio_dur, chunk);
            assert_seg_bits_eq(
                &sk,
                &s1,
                &format!("chunk-invariance seg {name} chunk={chunk}"),
            );
            // The emitted SET (begin/end/class triples) is identical -- chunking changes
            // the emission timing (emitted_at_audio_s) but never which segments emit.
            let t1: Vec<_> = em1
                .iter()
                .map(|e| (e.begin_s.to_bits(), e.end_s.to_bits(), e.class))
                .collect();
            let tk: Vec<_> = emk
                .iter()
                .map(|e| (e.begin_s.to_bits(), e.end_s.to_bits(), e.class))
                .collect();
            assert_eq!(t1, tk, "chunk-invariance emitted set {name} chunk={chunk}");
        }
    }
}
