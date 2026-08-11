//! THE EQUIVALENCE ORACLE for the incremental resmooth (`fast::stream::StreamDecision`).
//!
//! THE CHANGE. `StreamDecision::resmooth_and_emit` used to replay the WHOLE append-only
//! raw-segment list through `label_segment` + `smooth_segmentation` on every push --
//! `O(S)` calls of an `O(S)` walk per push, quadratic cumulative on an unbounded stream.
//! It now replays only a BOUNDED RETAINED WINDOW and splices the result onto a FROZEN
//! prefix. The contract is BIT-IDENTITY with the full replay at every push, so the only
//! acceptable instrument is a side-by-side comparison of the two paths.
//!
//! WHY A NEW ORACLE RATHER THAN THE COMMITTED GATES. The committed streaming gates
//! (`phase8_gate.rs`, `phase9_stream_causal.rs`, `phase8_stream_decision.rs`) were BLIND
//! to the phase-9 Task-9 emission-frontier bug -- all 19+19 stayed green with the fix
//! reverted; only real corpus data with a trained causal net exposed it. Those gates run
//! on a handful of hand-authored profiles whose gaps sit far outside the interesting
//! band, so they are a REGRESSION net, not a proof. This file is the proof: randomized
//! long streams (hundreds of raw segments), gap distributions deliberately concentrated
//! in the `(true_reach, holdback)` reopen band, burst durations straddling the
//! `suppress_short` thresholds, several chunk granularities, several seeds -- with the
//! achieved distribution MEASURED AND ASSERTED, not hoped for.
//!
//! THE REFERENCE PATH is the phase-8 code itself, not a paraphrase:
//! `StreamDecision::set_full_replay(true)` (a `test-support` hook) simply stops the cut
//! from ever advancing, which leaves `keep_from == 0`, `frozen` empty and the splice index
//! at 0 -- i.e. exactly the statements the phase-8 version executed.
//!
//! Legs:
//! - `incremental_equals_full_replay`: the oracle. Emitted sequence (all four fields,
//!   `to_bits`) AND `flush()`'s `Segmentation` (`to_bits` + types + duration) identical,
//!   over 4 CONFIG PROFILES x (3 long streams x 2 chunkings + 3 short streams x 4
//!   chunkings). Prints the profile+seed in every failure message and the achieved
//!   adversarial distribution on success.
//! - `wall_clock_is_recorded_not_asserted`: the cost measurement, PRINTED (a wall-clock
//!   assert has no place in CI); the leg still asserts the two paths agree.
//! - `retained_window_is_bounded`: the bounded-cost leg. Measured max retained count vs a
//!   bound DERIVED from the config's holdback/frontier arithmetic + the generator's own
//!   burst cap, plus a length-doubling leg showing the raw list grows while the retained
//!   window does not.
//! - `flush_sentinel_guard_matches_full_replay`: the one production fallback (a caller
//!   whose `audio_duration` lands within a holdback of the cut gets the full replay), so
//!   the guard is not dead code.
//! - `cut_engages_on_a_long_stream`: non-vacuity -- the cut actually freezes and slides.

mod common;

use speech::fast::nn::FastMatrix;
use speech::fast::stream::{EmittedSegment, StreamDecision};
use speech::legacy_config::parse_legacy_config;
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmenter::{DriverConfig, SegmenterConfig};

// The gate timeStep/timeOffset (tier2: spectrum_shift 0.01, ssr 4 -> dt 0.04, off 0.015),
// the same pair `phase8_stream_decision.rs` streams at.
const DT: f64 = 0.04;
const OFF: f64 = 0.015;

/// The generator's hard cap on a speech burst, in posterior frames (12 s at `DT`). Purely
/// a property of the EXPERIMENT: since the review's minor 8 it no longer feeds
/// [`derived_retained_bound`], which is config-only. It still bounds what the streams can
/// REACH, which is why the `long` criterion is holdback-RELATIVE rather than a multiple
/// (`4*holdback` on `silence-active` is 322 frames -- past this cap, hence structurally
/// unreachable, hence a floor that could never fire).
const MAX_BURST_FRAMES: usize = 300;

/// Parse tier2_spectral.config -> (SegmenterConfig, DriverConfig) at the BLSTM prefix
/// (the real 19-tap hHCw convolution kernel + the real decision/padding thresholds).
fn gate_cfgs() -> (SegmenterConfig, DriverConfig) {
    let text = std::fs::read_to_string(common::fixture_phase4a("tier2_spectral.config")).unwrap();
    let map = parse_legacy_config(&text);
    let seg = SegmenterConfig::from_config(&map, "BLSTM").unwrap();
    let drv = DriverConfig::from_config(&map, "BLSTM").unwrap();
    (seg, drv)
}

/// THE CONFIG MATRIX -- and the reason it is not just the gate config.
///
/// `tier2_spectral.config` clamps BOTH `min_silence` values to zero
/// (`-1.22e-2, -3.95e-2` -> `[0, 0]`), so its steps 4 and 7
/// (`suppress_short(min_silence, Other)`) are NO-OPS -- and step 4 is precisely the
/// LOAD-BEARING seam case of the cut derivation: the retained replay's leading `Other`
/// runs from 0 instead of from the previous raw segment's end, so its DURATION is the one
/// thing the truncation changes, and only a `suppress_short(.., Other)` reads it. Proving
/// bit-identity on tier2 alone would leave that case entirely unexercised. The variants
/// activate it, and vary WHICH terms dominate the holdback:
///
/// | profile          | what it stresses                                                |
/// |------------------|-----------------------------------------------------------------|
/// | `tier2`          | the committed gate config, verbatim (paddings dominate)          |
/// | `silence-active` | seam cases 4/7 LIVE: the straddling `Other` can be suppressed    |
/// | `suppress-heavy` | `padding[0] = padding[2] = 0`, so the true reach EQUALS the      |
/// |                  | holdback and the margin's slack is at the design's minimum       |
/// | `no-conv`        | no convolution: bursts map to raw segments directly, so tiny     |
/// |                  | near-threshold segments and sub-reach gaps become dense          |
fn cfg_profiles() -> Vec<(&'static str, SegmenterConfig, DriverConfig)> {
    let (base_seg, base_drv) = gate_cfgs();
    let mut out = vec![("tier2", base_seg.clone(), base_drv.clone())];

    let mut s = base_seg.clone();
    s.min_silence = [0.40, 0.25];
    s.min_speech = [0.20, 0.15, 0.35];
    out.push(("silence-active", s, base_drv.clone()));

    // reach == holdback here: `holdback` sums the `before` paddings that the leftward
    // reach does not need, and this profile sets them to zero, so the SECOND holdback of
    // CUT_MARGIN_HOLDBACKS is the entire safety margin.
    //
    // `min_speech[0]` is 0.35 rather than the 0.30 first written here, and the reason is
    // MEASURED, not cosmetic (review I-2): the 19-tap convolution imposes a FLOOR of
    // 0.32089 s on raw-segment duration -- identical on every convolved profile, since it
    // is a property of the kernel and the rising/falling thresholds, not of the stream --
    // so a `tiny` criterion at `max(min_speech) = 0.30` was structurally UNREACHABLE and
    // this profile measured tiny = 0 on all six seeds. The profile's DEFINING property is
    // `padding[0] == padding[2] == 0` (reach == holdback), which these values leave exactly
    // intact: reach = 0.80 + 0.55 + 0.30 + 0.25 = 1.90 = holdback.
    let mut s = base_seg.clone();
    s.padding = [0.0, 0.30, 0.0, 0.25];
    s.min_silence = [0.35, 0.20];
    s.min_speech = [0.35, 0.20, 0.25];
    out.push(("suppress-heavy", s, base_drv.clone()));

    let mut d = base_drv.clone();
    d.conv_coeff = None;
    let mut s = base_seg.clone();
    s.min_silence = [0.30, 0.10];
    out.push(("no-conv", s, d));

    out
}

fn holdback_of(cfg: &SegmenterConfig) -> f64 {
    cfg.min_speech.iter().sum::<f64>()
        + cfg.min_silence.iter().sum::<f64>()
        + cfg.padding.iter().sum::<f64>()
}

/// The TRUE leftward smoothing reach the holdback conservatively dominates: the two
/// `before` paddings plus the Speech `suppress_short` spans (the T4-review derivation
/// recorded in `fast::stream`'s docs). The interesting adversarial band for a gap is
/// `(true_reach, holdback]` -- wide enough that the smoothing does NOT merge across it,
/// narrow enough that the conservative holdback still blocks emission, which is exactly
/// where the frontier/cut arithmetic is load-bearing.
fn true_reach_of(cfg: &SegmenterConfig) -> f64 {
    cfg.padding[0] + cfg.padding[2] + cfg.min_speech.iter().sum::<f64>()
}

// ---------------------------------------------------------------------------
// A seeded, reproducible PRNG (xorshift64) -- no dev-dependency, and the seed is
// printed in every failure message so a red run replays exactly.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        // Any nonzero state works; the odd multiplier decorrelates small seeds.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in `[0, n)`.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    /// Uniform in `[lo, hi]` (inclusive).
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi.saturating_sub(lo) + 1)
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// The adversarial stream generator.
// ---------------------------------------------------------------------------

/// The posterior levels the generator draws from, derived from the config's own
/// thresholds so the stream is adversarial for THIS decision layer, not for a guess.
struct Levels {
    lo: f32,       // comfortably below `falling`
    hi: f32,       // comfortably above `rising`
    hover_hi: f32, // just above `rising` (area gating decides)
    hover_lo: f32, // just below `falling`
}

fn levels_of(cfg: &SegmenterConfig) -> Levels {
    Levels {
        lo: (cfg.falling * 0.25) as f32,
        hi: (cfg.rising + (1.0 - cfg.rising) * 0.75) as f32,
        hover_hi: (cfg.rising + (1.0 - cfg.rising) * 0.12) as f32,
        hover_lo: (cfg.falling * 0.88) as f32,
    }
}

/// What the generator INTENDED (its own bookkeeping) -- the measured achieved
/// distribution is read back off the raw segments the hysteresis actually produced.
struct PlanStats {
    bursts: usize,
    gaps_in_band: usize,
    tiny_bursts: usize,
}

/// Generate an adversarial posterior stream of `n_bursts` speech bursts.
///
/// GAPS are drawn from a four-way mixture in HOLDBACK UNITS, weighted onto the band the
/// design's arithmetic actually cares about:
///   30% `(0, 0.5*holdback]`      -- the smoothing MERGES across these (add_padding)
///   35% `(0.5, 1.2] * holdback`  -- straddles `true_reach` and `holdback`: the reopen band
///   25% `(1.2, 2.5] * holdback`  -- clears the holdback, settles mid-stream
///   10% `(2.5, 5.0] * holdback`  -- long silences (the cut slides far)
/// BURSTS are drawn to straddle the `suppress_short` thresholds (a quarter of them are
/// sub-threshold "tiny" bursts that the smoothing removes), with a 10% tail of very long
/// bursts that hold the hysteresis OPEN for a long time -- the `pending_begin` clamp's
/// regime, where the frontier (and so the cut) stalls.
/// Every fourth burst uses HOVERING levels so the area gating, not a clean crossing,
/// decides the begin/end -- and 15% of gaps get a one-frame spike that may or may not
/// reopen the hysteresis.
fn gen_stream(rng: &mut Rng, cfg: &SegmenterConfig, n_bursts: usize) -> (Vec<f32>, PlanStats) {
    let lv = levels_of(cfg);
    let holdback = holdback_of(cfg);
    let true_reach = true_reach_of(cfg);
    let hb_frames = (holdback / DT).round().max(1.0) as usize;
    let mut v: Vec<f32> = Vec::new();
    let mut stats = PlanStats {
        bursts: 0,
        gaps_in_band: 0,
        tiny_bursts: 0,
    };

    // Lead-in silence (the convolution's left edge + a settled start).
    for _ in 0..25 {
        v.push(lv.lo);
    }

    let max_min_speech = cfg
        .min_speech
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(DT);
    let ms_frames = (max_min_speech / DT).round().max(1.0) as usize;

    for k in 0..n_bursts {
        // --- the burst ---
        // The mixture is in MIN_SPEECH units, so the near-threshold bin lands on the
        // suppress_short flip (seam cases 2/5/8) for whichever config is running.
        let m = ms_frames.max(2);
        let u = rng.unit();
        let dur = if u < 0.12 {
            rng.range(1, 4) // vanishing: may not survive the convolution at all
        } else if u < 0.32 {
            rng.range(m / 2 + 1, (3 * m) / 2) // straddles the min_speech threshold
        } else if u < 0.72 {
            rng.range(m, 3 * m)
        } else if u < 0.90 {
            rng.range(3 * m, 8 * m)
        } else {
            rng.range(8 * m, MAX_BURST_FRAMES)
        };
        let dur = dur.min(MAX_BURST_FRAMES);
        if (dur as f64 * DT) <= max_min_speech {
            stats.tiny_bursts += 1;
        }
        let hovering = k % 4 == 3;
        let top = if hovering { lv.hover_hi } else { lv.hi };
        // A ramp frame at each edge on hovering bursts: the crossing interpolation and the
        // area accumulation both get non-degenerate inputs.
        if hovering {
            v.push(lv.hover_lo);
        }
        for _ in 0..dur {
            v.push(top);
        }
        if hovering {
            v.push(lv.hover_lo);
        }
        stats.bursts += 1;

        // --- the gap ---
        let u = rng.unit();
        let gap = if u < 0.30 {
            rng.range(1, hb_frames / 2)
        } else if u < 0.65 {
            rng.range(hb_frames / 2, (hb_frames * 12) / 10)
        } else if u < 0.90 {
            rng.range((hb_frames * 12) / 10, (hb_frames * 25) / 10)
        } else {
            rng.range((hb_frames * 25) / 10, hb_frames * 5)
        };
        let gap_s = gap as f64 * DT;
        if gap_s > true_reach && gap_s <= holdback {
            stats.gaps_in_band += 1;
        }
        let spike = rng.unit() < 0.15 && gap > 4;
        for j in 0..gap {
            if spike && j == gap / 2 {
                v.push(lv.hover_hi);
            } else {
                v.push(lv.lo);
            }
        }
    }

    // Trailing silence so the last burst closes (not force-emitted at EOS every time).
    for _ in 0..(2 * hb_frames) {
        v.push(lv.lo);
    }
    (v, stats)
}

// ---------------------------------------------------------------------------
// Running a stream through the decision layer, both paths.
// ---------------------------------------------------------------------------

fn as_rows(scalars: &[f32]) -> FastMatrix {
    FastMatrix {
        data: scalars.to_vec(),
        rows: scalars.len(),
        cols: 1,
    }
}

fn slice_rows(m: &FastMatrix, start: usize, end: usize) -> FastMatrix {
    FastMatrix {
        data: m.data[start..end].to_vec(),
        rows: end - start,
        cols: 1,
    }
}

struct Run {
    /// Every emission, in the order it was delivered (mid-stream pushes then the flush
    /// tail) -- the sequence the caller of `push`/`finish` sees.
    emitted: Vec<EmittedSegment>,
    final_seg: Segmentation,
    raw: Vec<(f64, f64)>,
    max_retained: usize,
    /// Pushes on which the retained window's FIRST raw segment straddles the truncation
    /// limit (`begin <= limit < end`) -- the seam case the cut derivation is about.
    straddles: usize,
    frozen_advances: usize,
}

fn run_stream(
    scalars: &[f32],
    seg_cfg: &SegmenterConfig,
    drv: &DriverConfig,
    audio_dur: f64,
    chunk: usize,
    full_replay: bool,
) -> Run {
    let rows = as_rows(scalars);
    let mut dec = StreamDecision::new(drv.clone(), seg_cfg.clone(), DT, OFF);
    // Called UNCONDITIONALLY, no `#[cfg(feature = "test-support")]` guard -- the
    // phase8_gate.rs pattern for a test-support hook, and here it is load-bearing rather
    // than stylistic: a cfg'd-out call would silently make `full_replay` a no-op, so BOTH
    // sides of the comparison would run the incremental path and the whole oracle would
    // pass VACUOUSLY. Without the feature this file must fail to BUILD, loudly.
    if full_replay {
        dec.set_full_replay(true);
    }
    let margin = 2.0 * dec.holdback();
    let mut emitted: Vec<EmittedSegment> = Vec::new();
    let mut max_retained = 0usize;
    let mut straddles = 0usize;
    let mut frozen_advances = 0usize;
    let mut prev_cut = f64::NEG_INFINITY;
    let mut i = 0;
    while i < rows.rows {
        let end = (i + chunk).min(rows.rows);
        emitted.extend(dec.push_rows(&slice_rows(&rows, i, end)));
        i = end;
        max_retained = max_retained.max(dec.retained_len());
        if dec.cut_time() > prev_cut {
            frozen_advances += 1;
            prev_cut = dec.cut_time();
        }
        let raw = dec.raw_segments();
        if !raw.is_empty() && dec.cut_time().is_finite() {
            let limit = dec.cut_time() - margin;
            let first = raw[raw.len() - dec.retained_len()];
            if first.0 <= limit && first.1 > limit {
                straddles += 1;
            }
        }
    }
    let (tail, final_seg) = dec.flush(audio_dur);
    emitted.extend(tail);
    max_retained = max_retained.max(dec.retained_len());
    Run {
        emitted,
        final_seg,
        raw: dec.raw_segments().to_vec(),
        max_retained,
        straddles,
        frozen_advances,
    }
}

// ---------------------------------------------------------------------------
// Bit-identity comparators (the contract is identity; a tolerance would be a bug).
// ---------------------------------------------------------------------------

fn assert_seg_bits_eq(got: &Segmentation, want: &Segmentation, label: &str) {
    let a = got.segments();
    let b = want.segments();
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: final segmentation entry count {} != {}",
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
            "{label}: entry {k} begin bits: {} vs {}",
            x.begin,
            y.begin
        );
        assert_eq!(
            x.ty, y.ty,
            "{label}: entry {k} type: {:?} vs {:?}",
            x.ty, y.ty
        );
    }
}

fn assert_emitted_bits_eq(got: &[EmittedSegment], want: &[EmittedSegment], label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: emission COUNT {} != {} (the incremental path emitted a different \
         sequence, not merely different values)",
        got.len(),
        want.len()
    );
    for (k, (x, y)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            x.begin_s.to_bits(),
            y.begin_s.to_bits(),
            "{label}: emission {k} begin bits: {} vs {}",
            x.begin_s,
            y.begin_s
        );
        assert_eq!(
            x.end_s.to_bits(),
            y.end_s.to_bits(),
            "{label}: emission {k} end bits: {} vs {}",
            x.end_s,
            y.end_s
        );
        assert_eq!(x.class, y.class, "{label}: emission {k} class");
        assert_eq!(
            x.emitted_at_audio_s.to_bits(),
            y.emitted_at_audio_s.to_bits(),
            "{label}: emission {k} emitted_at bits (the emission TIMING moved -- the cut \
             changed WHEN a segment settles, not only what it is)",
        );
    }
}

/// The achieved (not intended) adversarial distribution, read off the raw segments the
/// hysteresis actually produced.
struct Achieved {
    raw: usize,
    gaps_in_band: usize,
    gaps_below_reach: usize,
    tiny_segments: usize,
    long_segments: usize,
    /// The shortest / longest RAW segment actually produced. Reported because the
    /// convolution imposes a FLOOR on raw-segment duration that no burst length can beat
    /// (a 19-tap kernel spreads a short burst, so the hysteresis's interpolated
    /// begin/end sit further apart than the burst itself), and a `tiny` criterion below
    /// that floor is structurally unreachable rather than merely unlucky.
    min_dur: f64,
    max_dur: f64,
}

impl Achieved {
    fn zero() -> Achieved {
        Achieved {
            raw: 0,
            gaps_in_band: 0,
            gaps_below_reach: 0,
            tiny_segments: 0,
            long_segments: 0,
            min_dur: f64::INFINITY,
            max_dur: 0.0,
        }
    }
    fn fold(&mut self, o: &Achieved) {
        self.raw += o.raw;
        self.gaps_in_band += o.gaps_in_band;
        self.gaps_below_reach += o.gaps_below_reach;
        self.tiny_segments += o.tiny_segments;
        self.long_segments += o.long_segments;
        self.min_dur = self.min_dur.min(o.min_dur);
        self.max_dur = self.max_dur.max(o.max_dur);
    }
}

fn achieved(raw: &[(f64, f64)], cfg: &SegmenterConfig) -> Achieved {
    let holdback = holdback_of(cfg);
    let reach = true_reach_of(cfg);
    let max_ms = cfg.min_speech.iter().cloned().fold(0.0f64, f64::max);
    let mut a = Achieved {
        raw: raw.len(),
        gaps_in_band: 0,
        gaps_below_reach: 0,
        tiny_segments: 0,
        long_segments: 0,
        min_dur: f64::INFINITY,
        max_dur: 0.0,
    };
    for w in raw.windows(2) {
        let gap = w[1].0 - w[0].1;
        if gap > reach && gap <= holdback {
            a.gaps_in_band += 1;
        }
        if gap <= reach {
            a.gaps_below_reach += 1;
        }
    }
    for &(b, e) in raw {
        let dur = e - b;
        if dur <= max_ms {
            a.tiny_segments += 1;
        }
        // THE STALL REGIME, criterion holdback-relative AND reachable. A segment held OPEN
        // for at least one holdback is exactly where `pending_begin` measurably clamps the
        // frontier (and so stalls the cut) -- one holdback IS the amount the clamp defers.
        // The earlier `4*holdback` criterion was structurally UNREACHABLE on the profile
        // that needs it most: `silence-active`'s `4*holdback = 12.88 s = 322 frames`
        // exceeds MAX_BURST_FRAMES, so it measured long = 0 on all six seeds -- the
        // pending_begin regime never occurred on the ONLY profile whose min_silence is
        // live. Found by review, not by the floors, because the floor was global-only.
        if dur >= holdback {
            a.long_segments += 1;
        }
        a.min_dur = a.min_dur.min(dur);
        a.max_dur = a.max_dur.max(dur);
    }
    a
}

// ---------------------------------------------------------------------------
// (1) THE ORACLE.
// ---------------------------------------------------------------------------

#[test]
fn incremental_equals_full_replay() {
    // (seed, bursts, chunks). LONG streams carry the "hundreds of segments" requirement;
    // the SHORT ones carry the fine chunk granularities (chunk 1 on a 300-segment stream
    // would run the QUADRATIC reference path tens of thousands of times -- the reference
    // is the expensive path by construction, so the matrix trades length against
    // granularity rather than paying both at once).
    let cases: &[(u64, usize, &[usize])] = &[
        (1, 300, &[37, 128]),
        (2, 300, &[53, 256]),
        (3, 260, &[64, 199]),
        (101, 45, &[1, 3, 7, 64]),
        (102, 50, &[1, 2, 5, 33]),
        (103, 40, &[1, 4, 11, 128]),
    ];

    let mut grand = Achieved::zero();
    let mut tot_straddles = 0usize;
    let mut tot_emissions = 0usize;
    let mut tot_compared = 0usize;
    let mut max_raw = 0usize;

    for (pname, seg_cfg, drv) in cfg_profiles() {
        let holdback = holdback_of(&seg_cfg);
        let reach = true_reach_of(&seg_cfg);
        let mut prof = Achieved::zero();
        for &(seed, bursts, chunks) in cases {
            let mut rng = Rng::new(seed);
            let (scalars, plan) = gen_stream(&mut rng, &seg_cfg, bursts);
            let audio_dur = DT * scalars.len() as f64 + OFF;
            for &chunk in chunks {
                let label = format!(
                    "PROFILE={pname} SEED={seed} bursts={bursts} chunk={chunk} rows={}",
                    scalars.len()
                );
                let want = run_stream(&scalars, &seg_cfg, &drv, audio_dur, chunk, true);
                let got = run_stream(&scalars, &seg_cfg, &drv, audio_dur, chunk, false);

                // The raw hysteresis is upstream of the cut: if it ever differed the
                // comparison below would be meaningless, so check it first and say so.
                assert_eq!(
                    got.raw.len(),
                    want.raw.len(),
                    "{label}: raw-segment count differs between the two paths -- the cut \
                     must not touch the hysteresis"
                );
                assert_emitted_bits_eq(&got.emitted, &want.emitted, &label);
                assert_seg_bits_eq(&got.final_seg, &want.final_seg, &label);

                // The reference path must NOT have cut anything (it is the phase-8 path).
                assert_eq!(
                    want.max_retained,
                    want.raw.len(),
                    "{label}: the full-replay reference retained {} of {} raw segments -- \
                     it must replay all of them",
                    want.max_retained,
                    want.raw.len()
                );
                tot_straddles += got.straddles;
                tot_emissions += got.emitted.len();
                tot_compared += got.final_seg.segments().len();
                max_raw = max_raw.max(got.raw.len());

                if chunk == chunks[0] {
                    let a = achieved(&got.raw, &seg_cfg);
                    prof.fold(&a);
                    println!(
                        "MEASURE oracle {label}: planned_bursts={} tiny_planned={} | raw={} \
                         gaps_in_band={} gaps_below_reach={} tiny={} long={} | emissions={} \
                         max_retained={} straddles={} frozen_advances={}",
                        plan.bursts,
                        plan.tiny_bursts,
                        a.raw,
                        a.gaps_in_band,
                        a.gaps_below_reach,
                        a.tiny_segments,
                        a.long_segments,
                        got.emitted.len(),
                        got.max_retained,
                        got.straddles,
                        got.frozen_advances
                    );
                }
            }
        }
        println!(
            "MEASURE oracle PROFILE {pname}: holdback={holdback:.5} true_reach={reach:.5} \
             band=({reach:.5}, {holdback:.5}] min_silence={:?} raw={} gaps_in_band={} \
             gaps_below_reach={} tiny={} long={} raw_dur=[{:.5}, {:.5}] \
             tiny_thresh={:.5} long_thresh={:.5}",
            seg_cfg.min_silence,
            prof.raw,
            prof.gaps_in_band,
            prof.gaps_below_reach,
            prof.tiny_segments,
            prof.long_segments,
            prof.min_dur,
            prof.max_dur,
            seg_cfg.min_speech.iter().cloned().fold(0.0f64, f64::max),
            holdback
        );
        // PER PROFILE, ALL FOUR REGIMES. The gap floors were here from the start; the
        // tiny/long floors were GLOBAL-only, which hid two structural holes a global sum
        // cannot see (review I-2): `silence-active` measured long = 0 on every seed (its
        // old `4*holdback` criterion exceeded the burst cap) and `suppress-heavy` measured
        // tiny = 0 on every seed (its `max(min_speech)` sat just under the convolution's
        // own floor on raw-segment duration) -- i.e. the two profiles that exist to
        // exercise the live-min_silence and minimum-slack regimes were each missing one
        // of the two segment regimes, and the global total covered for them.
        assert!(
            prof.gaps_in_band >= 3 && prof.gaps_below_reach >= 3,
            "adversarial coverage on profile {pname}: gaps_in_band={} gaps_below_reach={} \
             -- each profile must exercise BOTH the reopen band and the merging band",
            prof.gaps_in_band,
            prof.gaps_below_reach
        );
        assert!(
            prof.tiny_segments >= 3,
            "adversarial coverage on profile {pname}: tiny={} (raw durations spanned \
             [{:.5}, {:.5}], threshold {:.5}) -- every profile must produce raw segments \
             the suppress_short flip cases 2/5/8 can act on. If the measured minimum sits \
             ABOVE the threshold the criterion is structurally unreachable on this profile, \
             not unlucky: the convolution imposes a floor on raw-segment duration.",
            prof.tiny_segments,
            prof.min_dur,
            prof.max_dur,
            seg_cfg.min_speech.iter().cloned().fold(0.0f64, f64::max)
        );
        assert!(
            prof.long_segments >= 3,
            "adversarial coverage on profile {pname}: long={} (raw durations spanned \
             [{:.5}, {:.5}], threshold {holdback:.5}) -- every profile must hold the \
             hysteresis OPEN for at least one holdback, the regime where pending_begin \
             clamps the frontier and stalls the cut",
            prof.long_segments,
            prof.min_dur,
            prof.max_dur
        );
        grand.fold(&prof);
    }

    println!(
        "MEASURE oracle TOTALS: raw={} gaps_in_band={} gaps_below_reach={} tiny={} long={} \
         straddles={} emissions={} final_entries_compared={} max_raw_per_stream={}",
        grand.raw,
        grand.gaps_in_band,
        grand.gaps_below_reach,
        grand.tiny_segments,
        grand.long_segments,
        tot_straddles,
        tot_emissions,
        tot_compared,
        max_raw
    );

    // NON-VACUITY, asserted rather than hoped for: the generator must actually reach the
    // adversarial regimes the cut derivation reasons about.
    assert!(
        max_raw >= 150,
        "the long streams must carry HUNDREDS of raw segments (max per stream {max_raw})"
    );
    assert!(
        grand.gaps_in_band >= 60,
        "adversarial coverage: gaps in the reopen band (above the true smoothing reach, at \
         or under the holdback -- where the smoothing does not merge but the holdback still \
         blocks emission) must be well represented (got {})",
        grand.gaps_in_band
    );
    assert!(
        grand.gaps_below_reach >= 60,
        "adversarial coverage: sub-reach gaps (the smoothing MERGES across these -- the \
         load-bearing seam case 4) must be well represented (got {})",
        grand.gaps_below_reach
    );
    assert!(
        grand.tiny_segments >= 60,
        "adversarial coverage: raw segments at or under the largest min_speech threshold \
         (the suppress_short flip cases 2/5/8) must be well represented (got {})",
        grand.tiny_segments
    );
    assert!(
        grand.long_segments >= 100,
        "adversarial coverage: long segments (>= one holdback -- the pending_begin clamp \
         regime, where the frontier and therefore the cut stall) must occur (got {})",
        grand.long_segments
    );
    // NOTE (review minor 10, also recorded as a follow-on in `fast::stream`'s docs): this
    // count is PER PUSH, so the fine chunkings inflate it -- chunk 1 re-counts the same
    // straddling configuration on every push. It is honest as a coverage FLOOR and
    // misleading as a headline; the quantity worth reporting is distinct straddling
    // raw-segment indices.
    assert!(
        tot_straddles >= 100,
        "adversarial coverage: pushes whose retained window STARTS on a segment straddling \
         the truncation limit (got {tot_straddles})"
    );
    assert!(
        tot_emissions >= 1000,
        "non-vacuity: the comparison must actually compare emissions (got {tot_emissions})"
    );
}

// ---------------------------------------------------------------------------
// (2) THE BOUNDED-COST LEG.
// ---------------------------------------------------------------------------

/// The retained-window bound, DERIVED from the config's own frontier/holdback arithmetic.
/// CONFIG-ONLY -- no measured input and, since the review's minor 8, no generator
/// parameter either: an earlier form carried the generator's `MAX_BURST_FRAMES` as an
/// `open_span` term, which read as though the theorem were conditional on the experiment.
/// It is not, and `fast::stream::StreamDecision`'s docs carry the stronger argument this
/// mirrors: a long OPEN segment widens the window but CANNOT add retained segments,
/// because no raw segment CLOSES while one is open.
///
///   limit  = cut_time - 2*holdback                        (the cut invariant)
///   cut_time >= frontier_now - holdback                   (cut_time is a running max)
///   now - frontier <= max(conv_delay + dt, open_age)      (the three frontier terms;
///                                                          the third clamps to an OPEN
///                                                          segment's begin)
///
/// If the third term dominates (a stall), every CLOSED raw segment in the window began
/// before the open one did, so their begins lie in `(limit, pending_begin]` -- and there
/// `frontier == pending_begin`, so that width is `<= 3*holdback`. Otherwise
/// `now - frontier <= conv_delay + dt` and the width is `<= 3*holdback + conv_delay + dt`.
/// Either way the CLOSED segments occupy a window of at most the latter. At most one raw
/// segment is LABELED per convolved value (`HystState::advance` returns at most one) and a
/// retained segment's label time lies in that window, so at most `floor(width/dt) + 1` are
/// retained -- plus the anchor the cut always keeps, plus at most one just-closed segment
/// from a stall.
///
/// IT IS DELIBERATELY LOOSE, and that is worth stating plainly: the last step assumes the
/// hysteresis can close a segment at EVERY grid step, which the area gating makes
/// impossible in practice (measured worst 5 against a derived 184 on tier2). Tightening it
/// would mean bounding the posterior VALUES, which are a property of the net and the
/// convolution kernel, not of this layer -- so the derived bound is the SAFETY pin (it is
/// what makes "bounded per-push cost" a theorem rather than an observation, since it does
/// not depend on the stream at all), and [`RETAINED_REGRESSION_PIN`] is the discriminating
/// one, measured-then-pinned in this repo's usual style.
fn derived_retained_bound(seg_cfg: &SegmenterConfig, drv: &DriverConfig) -> usize {
    let holdback = holdback_of(seg_cfg);
    let conv_half = match drv.conv_coeff.as_deref() {
        Some(c) if c.len() > 1 => (c.len() - 1) / 2,
        _ => 0,
    };
    let conv_delay = conv_half as f64 * DT;
    let window = 3.0 * holdback + conv_delay + DT;
    // +2: the anchor segment the cut always retains, and at most one just-closed segment
    // carried over from a stall.
    3 + (window / DT).floor() as usize
}

/// The DISCRIMINATING retained-window pin: measured worst 6 (over every profile x seed x
/// length in this file, printed by the leg below; `no-conv` sets it), pinned at 50 --
/// inside this repo's measured*10 convention, with the SUMMARY assert below holding the
/// pin within 10x of the measured worst so it cannot drift into a ceiling nobody hits.
/// A cut that stopped sliding would put the retained count in the hundreds and blow
/// straight through this, where the derived structural bound (184 on tier2) would not
/// notice until the stream got very long indeed.
const RETAINED_REGRESSION_PIN: usize = 50;

#[test]
fn retained_window_is_bounded() {
    let mut worst = 0usize;

    for (pname, seg_cfg, drv) in cfg_profiles() {
        let k_derived = derived_retained_bound(&seg_cfg, &drv);
        for seed in [11u64, 12] {
            let mut raw_1x = 0usize;
            let mut ret_1x = 0usize;
            let mut raw_2x = 0usize;
            let mut ret_2x = 0usize;
            for (idx, bursts) in [200usize, 400].into_iter().enumerate() {
                let mut rng = Rng::new(seed);
                let (scalars, _plan) = gen_stream(&mut rng, &seg_cfg, bursts);
                let audio_dur = DT * scalars.len() as f64 + OFF;
                let run = run_stream(&scalars, &seg_cfg, &drv, audio_dur, 64, false);
                println!(
                    "MEASURE bounded-cost profile={pname} seed={seed} bursts={bursts} \
                     rows={} raw={} max_retained={} k_derived={k_derived}",
                    scalars.len(),
                    run.raw.len(),
                    run.max_retained
                );
                assert!(
                    run.max_retained <= k_derived,
                    "profile={pname} seed={seed} bursts={bursts}: retained window {} \
                     exceeded the DERIVED bound {k_derived} -- the cut arithmetic and the \
                     implementation disagree",
                    run.max_retained
                );
                assert!(
                    run.max_retained <= RETAINED_REGRESSION_PIN,
                    "profile={pname} seed={seed} bursts={bursts}: retained window {} \
                     exceeded the measured*10 regression pin {RETAINED_REGRESSION_PIN} -- \
                     the cut has stopped sliding as designed",
                    run.max_retained
                );
                worst = worst.max(run.max_retained);
                if idx == 0 {
                    raw_1x = run.raw.len();
                    ret_1x = run.max_retained;
                } else {
                    raw_2x = run.raw.len();
                    ret_2x = run.max_retained;
                }
            }

            // THE ACTUAL PROPERTY: doubling the stream doubles the raw list and does NOT
            // grow the replayed window. (`max_retained` is a max over pushes, so a longer
            // stream may sample a slightly deeper window; what must not happen is growth
            // WITH the stream.)
            assert!(
                raw_2x >= (17 * raw_1x) / 10,
                "profile={pname} seed={seed}: the doubling leg must roughly double the raw \
                 list ({raw_1x} -> {raw_2x})"
            );
            assert!(
                ret_2x * 8 <= raw_2x,
                "profile={pname} seed={seed}: the retained window ({ret_2x}) must be a \
                 small fraction of the raw list ({raw_2x}) -- that ratio IS the \
                 bounded-cost claim (1x: {ret_1x}/{raw_1x})"
            );
        }
    }

    println!(
        "MEASURE bounded-cost SUMMARY: worst_measured={worst} regression_pin={RETAINED_REGRESSION_PIN}"
    );
    // The pin must be a PIN, not a ceiling nobody could hit: keep it within an order of
    // magnitude of what the design actually produces.
    assert!(
        worst * 10 >= RETAINED_REGRESSION_PIN,
        "the regression pin {RETAINED_REGRESSION_PIN} has drifted more than 10x above the \
         measured worst {worst} -- re-measure it rather than leaving a pin that catches \
         nothing"
    );
}

// ---------------------------------------------------------------------------
// (2') the wall-clock record. PRINTED, NOT ASSERTED.
// ---------------------------------------------------------------------------

/// The point of the change is COST, so the cost is measured -- but a wall-clock
/// comparison is a terrible CI assert (shared runners, thermal state, debug vs release),
/// so this leg asserts only the thing that must always hold (the two paths agree) and
/// PRINTS the timing for the record. The numbers in the branch report come from here.
#[test]
fn wall_clock_is_recorded_not_asserted() {
    let (seg_cfg, drv) = gate_cfgs();
    let mut rng = Rng::new(31);
    let (scalars, _plan) = gen_stream(&mut rng, &seg_cfg, 400);
    let audio_dur = DT * scalars.len() as f64 + OFF;

    let t0 = std::time::Instant::now();
    let full = run_stream(&scalars, &seg_cfg, &drv, audio_dur, 64, true);
    let full_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = std::time::Instant::now();
    let inc = run_stream(&scalars, &seg_cfg, &drv, audio_dur, 64, false);
    let inc_ms = t1.elapsed().as_secs_f64() * 1e3;

    let pushes = scalars.len().div_ceil(64);
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!(
        "MEASURE wall-clock ({profile} build, informational): rows={} pushes={pushes} raw={} \
         audio_s={audio_dur:.1} | full_replay={full_ms:.1} ms incremental={inc_ms:.1} ms \
         speedup={:.1}x | per-push full={:.1} us incremental={:.1} us | max_retained \
         full={} incremental={}",
        scalars.len(),
        inc.raw.len(),
        full_ms / inc_ms.max(1e-9),
        full_ms * 1e3 / pushes as f64,
        inc_ms * 1e3 / pushes as f64,
        full.max_retained,
        inc.max_retained
    );
    assert_emitted_bits_eq(&inc.emitted, &full.emitted, "wall-clock leg emissions");
    assert_seg_bits_eq(
        &inc.final_seg,
        &full.final_seg,
        "wall-clock leg segmentation",
    );
}

// ---------------------------------------------------------------------------
// (3) the flush-time sentinel guard (the one production fallback path).
// ---------------------------------------------------------------------------

/// `flush`'s frozen prefix was proven final against mid-stream sentinels at `now`; a
/// caller whose `audio_duration` lands within one holdback of the cut would put the
/// smoothing's tail special-casing inside the frozen region, so `flush` falls back to the
/// full replay there. This pins the fallback AGREEING with the full-replay path (and keeps
/// it from being dead code).
#[test]
fn flush_sentinel_guard_matches_full_replay() {
    let (seg_cfg, drv) = gate_cfgs();
    let mut rng = Rng::new(7);
    let (scalars, _plan) = gen_stream(&mut rng, &seg_cfg, 60);
    let rows = as_rows(&scalars);

    // Drive both paths to EOS, then flush at a duration deliberately inside the guard.
    let mut inc = StreamDecision::new(drv.clone(), seg_cfg.clone(), DT, OFF);
    let mut refr = StreamDecision::new(drv.clone(), seg_cfg.clone(), DT, OFF);
    refr.set_full_replay(true); // unconditional -- see `run_stream`'s note
    let mut i = 0;
    while i < rows.rows {
        let end = (i + 64).min(rows.rows);
        let _ = inc.push_rows(&slice_rows(&rows, i, end));
        let _ = refr.push_rows(&slice_rows(&rows, i, end));
        i = end;
    }
    let cut = inc.cut_time();
    assert!(
        cut.is_finite(),
        "the stream must have engaged the cut before this leg means anything"
    );
    let short_dur = cut + inc.holdback();
    println!(
        "MEASURE sentinel-guard: cut_time={cut:.5} holdback={:.5} flush_dur={short_dur:.5} \
         (guard fires at flush_dur <= cut+holdback)",
        inc.holdback()
    );

    let (e_inc, s_inc) = inc.flush(short_dur);
    let (e_ref, s_ref) = refr.flush(short_dur);
    assert_emitted_bits_eq(&e_inc, &e_ref, "sentinel-guard flush emissions");
    assert_seg_bits_eq(&s_inc, &s_ref, "sentinel-guard flush segmentation");

    // And the normal (non-guarded) duration still agrees, on a fresh pair -- so the leg
    // above is testing the GUARD, not just flush in general.
    let audio_dur = DT * scalars.len() as f64 + OFF;
    assert!(
        audio_dur > cut + inc.holdback(),
        "the normal flush duration must sit OUTSIDE the guard for this contrast to hold"
    );
    let a = run_stream(&scalars, &seg_cfg, &drv, audio_dur, 64, false);
    let b = run_stream(&scalars, &seg_cfg, &drv, audio_dur, 64, true);
    assert_seg_bits_eq(&a.final_seg, &b.final_seg, "sentinel-guard normal flush");
}

// ---------------------------------------------------------------------------
// (4) non-vacuity: the cut engages at all.
// ---------------------------------------------------------------------------

#[test]
fn cut_engages_on_a_long_stream() {
    let (seg_cfg, drv) = gate_cfgs();
    let mut rng = Rng::new(21);
    let (scalars, _plan) = gen_stream(&mut rng, &seg_cfg, 200);
    let audio_dur = DT * scalars.len() as f64 + OFF;
    let run = run_stream(&scalars, &seg_cfg, &drv, audio_dur, 64, false);
    println!(
        "MEASURE cut-engages: raw={} max_retained={} frozen_advances={} straddles={} \
         emissions={}",
        run.raw.len(),
        run.max_retained,
        run.frozen_advances,
        run.straddles,
        run.emitted.len()
    );
    assert!(
        run.raw.len() > 100,
        "the stream must produce a long raw list (got {})",
        run.raw.len()
    );
    assert!(
        run.frozen_advances > 20,
        "the cut must advance repeatedly (got {})",
        run.frozen_advances
    );
    assert!(
        run.max_retained * 5 < run.raw.len(),
        "the retained window ({}) must be far smaller than the raw list ({}) -- else the \
         cut is not doing anything and the oracle above is comparing two full replays",
        run.max_retained,
        run.raw.len()
    );
}
