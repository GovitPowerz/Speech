//! Phase 9 Task 7 (spec S5.1-S5.6, `2026-07-27-phase-9-new-architectures-design.md`):
//! THE CAUSAL STREAMING GATE. `fast::stream::StreamCausal` replaces `StreamOverlap` in
//! the four-layer composition for a causal config; this suite proves the streaming
//! contract against the OFFLINE FAST CAUSAL PATH (the oracle -- same kernels, only
//! chunking differs), exactly as `phase8_gate.rs` does for the windowed arm.
//!
//! THE GATE STRATEGY (the phase-8 pattern, on phase-9 fixtures). The committed T5/T6
//! causal configs (`tests/reference_data/phase9/{slstm,mamba}_forward.config`) + their
//! committed seed packs, replayed on the SAME staged 60 s mono fixture the phase-8 gate
//! uses (`common::stage_frozen_tier2`'s channel-0 extraction of
//! `phase4d/prcts_excerpt.wav`), under IN-TEST overlays applied IDENTICALLY to both
//! sides: `Audio_fixed_gain <measured>`, `Inference_Path fast`, whole-file duration, and
//! -- for the crossing legs -- a swept OUTPUT-BIAS offset on a local copy of the pack.
//! The fixture configs already carry `BLSTM_window 0.0` (the causal plain regime) and
//! `BLSTM_InputNormalizationType 0`, both of which the streaming session requires/accepts,
//! so no overlay touches the architecture.
//!
//! Note the phase-9 fixtures make this gate STRICTLY STRONGER than phase 8's in one
//! respect: `BLSTM_preemph_ratio 0.97` is POSITIVE here (tier2's is `-0.97`, inert), so
//! the front-end's cross-chunk pre-emphasis CARRY is live on every leg.
//!
//! NON-VACUITY (the phase-2b SEG_STRUCT lesson, restated by Task 6's crossing sweep): a
//! run whose posteriors never cross `BLSTM_decision_thresh_rising` proves nothing about
//! boundary equality. [`crossing_offset`] sweeps a pure post-recurrence level shift on the
//! output layer's single bias until the OFFLINE run carries interior boundaries, and every
//! equivalence/latency leg runs on that offset -- so `max_dt == 0.0` compares real interior
//! boundaries, not two copies of the seeded `[Other@0, End@dur]`.
//!
//! THE S5.3 DEVIATION (sub-sampling): the fixtures run `BLSTM_LSTMSubSampling 4`, so every
//! leg here exercises the buffered decimation `StreamCausal` implements instead of bailing.
//! See that type's docs and the module doc of `fast/stream.rs` for why.

mod common;

use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use speech::audio::read_audio;
use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::{BagOfProcessors, Processor};
use speech::fast::stream::{EmittedSegment, StreamingSession};
use speech::features::pipeline::{FeatureConfig, SpectralParams};
use speech::io::binary::{read_weight_vector, write_matrix};
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

fn fixture_phase9(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase9")
        .join(name)
}

// ---------------------------------------------------------------------------
// Staging.
// ---------------------------------------------------------------------------

/// A staged causal streaming config: the committed fixture config + the in-test overlays,
/// pointed at a local pack copy, over the phase-8 staged mono wav.
struct CausalStage {
    wav: PathBuf,
    config_text: String,
    gain: f64,
    pack: PathBuf,
}

/// Stage one causal cell's gate config in `dir`. `bias_offset` shifts the output layer's
/// single bias in a LOCAL pack copy (the crossing sweep -- a post-recurrence level shift
/// that leaves every cell weight, and so the whole SHAPE of the posterior curve,
/// untouched); `tag` keeps concurrent stagings in the same tempdir from colliding.
fn stage(dir: &Path, cell: &str, bias_offset: f64, tag: &str) -> CausalStage {
    let base = common::stage_frozen_tier2(dir);
    let text = std::fs::read_to_string(fixture_phase9(&format!("{cell}_forward.config"))).unwrap();

    // The output layer's bias is the LAST pack element before the `2*input_size` normalize
    // tail (layout `[stack | output MLP | mean | std]`, and the fixture's output MLP is a
    // single `4 -> 1` layer: 4 weights then 1 bias). `input_size` is DERIVED from the
    // config, not hardcoded, so a fixture regeneration relocates the index.
    let map = parse(&text);
    let input_size: usize = map["BLSTM_LSTMNeuronNb"]
        .split(',')
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let mut flat =
        read_weight_vector(&fixture_phase9(&format!("{cell}_forward_seed.bin"))).unwrap();
    let bias_idx = flat.len() - 2 * input_size - 1;
    flat[bias_idx] += bias_offset;
    let pack = dir.join(format!("{cell}_{tag}_pack.bin"));
    write_matrix(&pack, flat.len(), 1, &flat).unwrap();

    let dump_dir = dir.join(format!("vrcts_{cell}_{tag}"));
    std::fs::create_dir_all(&dump_dir).unwrap();

    let config_text = format!(
        "{text}\n\
# ==== Phase 9 Task 7 causal streaming gate overlays (last-wins) ====\n\
numOuterThreads 1\n\
Audio_offset 0.0\n\
Audio_max_duration 3600\n\
BLSTM_weightsFile {}\n\
Dump_Directory {}\n\
Inference_Path fast\n\
Audio_fixed_gain {}\n",
        pack.display(),
        dump_dir.display(),
        base.fixed_gain,
    );

    CausalStage {
        wav: base.wav_path,
        config_text,
        gain: base.fixed_gain,
        pack,
    }
}

/// The FROZEN TYPE-1 variant of a staged config (the phase-8 `stage_calibrated` pattern):
/// patch the pack's `2*input_size` normalize tail with the fixture's OWN per-column
/// self-norm mean/std and switch `InputNormalizationType` to 1, so the type-1 threading
/// through the CAUSAL arm is exercised on a NON-IDENTITY tail. GATE-CONSTRUCTION
/// CALIBRATION, NOT A TRAINED TAIL (the phase-8 R4 posture, verbatim).
fn calibrate(dir: &Path, base: &CausalStage, cell: &str, tag: &str) -> CausalStage {
    let map = parse(&base.config_text);
    let feature_cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
    let audio = read_audio(&base.wav, 0.0, 3600.0, 0, Some(base.gain)).unwrap();
    let rate = audio.sample_rate as f64;
    let params = SpectralParams::derive(&feature_cfg, rate);
    let mut pipeline =
        speech::fast::pipeline::FastPipeline::new(&params, &feature_cfg, rate).unwrap();
    // The offline driver applies pre-emphasis to the GAINED channel before framing
    // (`fast/driver.rs`, gated `preemph_ratio > 0` -- LIVE on the phase-9 fixtures), so the
    // measured statistics must be taken on the same signal.
    let mut audio = audio;
    if feature_cfg.preemph_ratio > 0.0 {
        audio.apply_preemph(feature_cfg.preemph_ratio);
    }
    let samples: Vec<f32> = audio.data.row(0).iter().map(|&x| x as f32).collect();
    let input = pipeline.build_input_sequence(&samples).clone();
    let (t, c) = (input.rows, input.cols);

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

    let input_size: usize = map["BLSTM_LSTMNeuronNb"]
        .split(',')
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(
        c <= input_size,
        "assembled input dim {c} must fit the net input width {input_size}"
    );
    let mut flat = read_weight_vector(&base.pack).unwrap();
    let n = flat.len();
    for (col, (&m, &sd)) in mean.iter().zip(std.iter()).enumerate() {
        flat[n - 2 * input_size + col] = m;
        flat[n - input_size + col] = sd;
    }
    let pack = dir.join(format!("{cell}_{tag}_cal_pack.bin"));
    write_matrix(&pack, flat.len(), 1, &flat).unwrap();

    let config_text = format!(
        "{}\n\
# ==== calibrated type-1 tail (gate-construction calibration, NOT a trained tail) ====\n\
BLSTM_weightsFile {}\n\
BLSTM_InputNormalizationType 1\n",
        base.config_text,
        pack.display()
    );
    CausalStage {
        wav: base.wav.clone(),
        config_text,
        gain: base.gain,
        pack,
    }
}

// ---------------------------------------------------------------------------
// Oracle + session drivers.
// ---------------------------------------------------------------------------

/// The offline fast CAUSAL bag run under the frozen gain -- the oracle. Returns the
/// channel-0 posterior row + the final Segmentation.
fn run_offline(stage: &CausalStage) -> (Vec<f64>, Segmentation) {
    let mut map = parse(&stage.config_text);
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut map), image_mode()).unwrap();
    let mut audio = read_audio(&stage.wav, 0.0, 3600.0, 0, Some(stage.gain)).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut seg = vec![Segmentation::new(dur)];
    bag.run_get_segmentation(0, &mut audio, &mut seg).unwrap();
    let row = match bag.processor(0) {
        Processor::FastSpectral(s) => s.last_result_rows()[0].clone(),
        _ => panic!("expected the fast SAD driver for a causal Inference_Path fast config"),
    };
    (row, seg.remove(0))
}

/// Read the staged MONO wav's RAW samples (`i16/32768` as f32, PRE-gain -- the front-end
/// applies the frozen gain/preemph itself).
fn mono_samples(wav: &Path) -> (f64, Vec<f32>) {
    let (rate, channels, samples) = common::read_wav_pcm16(wav);
    assert_eq!(channels, 1, "the streaming gate uses the staged MONO wav");
    (
        rate as f64,
        samples.iter().map(|&x| x as f32 / 32768.0).collect(),
    )
}

/// One full streaming run at `chunk`-sample granularity.
struct SessionRun {
    push_emissions: Vec<EmittedSegment>,
    finish_emissions: Vec<EmittedSegment>,
    seg: Segmentation,
    posteriors: Vec<f32>,
    bound: f64,
    feature_reach: f64,
    nn_window: f64,
    sub_sample: f64,
    conv_delay: f64,
    holdback: f64,
    emission_count: usize,
    is_causal: bool,
}

impl SessionRun {
    fn all_emissions(&self) -> Vec<EmittedSegment> {
        let mut v = self.push_emissions.clone();
        v.extend(self.finish_emissions.clone());
        v
    }
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
        bound: sess.derived_latency_bound_s(),
        feature_reach: sess.feature_reach_s(),
        nn_window: sess.nn_window_s(),
        sub_sample: sess.sub_sample_s(),
        conv_delay: sess.conv_delay_s(),
        holdback: sess.holdback_s(),
        emission_count: sess.emission_count(),
        is_causal: sess.is_causal(),
        push_emissions,
        finish_emissions,
        seg,
    }
}

// ---------------------------------------------------------------------------
// Comparison helpers (mirroring phase8_gate.rs).
// ---------------------------------------------------------------------------

fn seg_identities(seg: &Segmentation) -> Vec<(u64, i32)> {
    seg.segments()
        .iter()
        .map(|s| (s.begin.to_bits(), s.ty as i32))
        .collect()
}

fn emission_identities(em: &[EmittedSegment]) -> Vec<(u64, u64, i32)> {
    em.iter()
        .map(|e| (e.begin_s.to_bits(), e.end_s.to_bits(), e.class as i32))
        .collect()
}

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

/// Interior boundaries: segments that are neither the seeded head (`begin == 0`) nor the
/// `End` sentinel -- i.e. genuine mid-file decisions.
fn interior_boundaries(seg: &Segmentation) -> usize {
    seg.segments()
        .iter()
        .filter(|x| x.ty != SegClass::End && x.begin > 0.0)
        .count()
}

/// The interior-boundary floor the PLAIN (type-0) legs demand -- the ones carrying the
/// headline `max_dt == 0.0` claim. TWO, not one: a single interior boundary means the
/// posterior crossed the rising threshold once and never came back, so the comparison
/// exercises one hysteresis edge and not the other. Measured, the plain legs clear this
/// comfortably (slstm 12, mamba 16), so the floor is a REGRESSION DETECTOR -- if a future
/// fixture or kernel change quietly thins the boundary set to a single edge, the sweep
/// moves on rather than silently weakening the gate.
/// The default push granularity every end-to-end leg streams at, in SECONDS. Shared so the
/// latency leg's tight pin (`bound + PUSH_CHUNK_S`) and the chunk size it actually pushes at
/// cannot drift apart -- the emission clock advances once per `push`, so that pin is only
/// valid for THIS granularity.
const PUSH_CHUNK_S: f64 = 0.1;

const MIN_INTERIOR_PLAIN: usize = 2;

/// The floor the CALIBRATED TYPE-1 leg demands. ONE, and that is a MEASURED CEILING, not a
/// lowered bar: on the committed sLSTM fixture under a calibrated type-1 tail, `>= 2` is
/// structurally unreachable -- a 12-point sweep over `[0.30, 0.90]` (plus the coarse
/// `[-3, +2]` list) finds the posterior crossing at ONLY three offsets (0.50/0.55/0.60) and
/// crossing exactly ONCE at each; every other offset yields zero. The calibrated tail
/// normalizes this tiny 4-unit net's input to zero-mean/unit-std, which compresses the
/// posterior's dynamic range until it merely grazes the rising threshold. That leg is
/// therefore worth exactly what it claims: type-1 THREADING through the causal arm (the
/// tail demonstrably moves the posteriors -- asserted separately in the leg) plus ONE real
/// interior boundary compared bit-for-bit. It is NOT a rich boundary-set comparison; the
/// plain legs are.
const MIN_INTERIOR_TYPE1: usize = 1;

/// Sweep the output-layer bias until the OFFLINE causal run built by `build` carries at
/// least `min_interior` interior boundaries, and return that offset. `0.0` FIRST, so a
/// fixture that already crosses is used AS COMMITTED and the sweep is a fallback, not a
/// default detour. `build` is a closure so the same sweep serves the plain (type-0) and the
/// calibrated-tail (type-1) legs, whose posteriors sit at different levels.
fn crossing_offset(label: &str, min_interior: usize, build: impl Fn(f64) -> CausalStage) -> f64 {
    for offset in [0.0_f64, -0.5, -1.0, 0.5, -1.5, 1.0, -2.0, 1.5, -3.0, 2.0] {
        let (_, seg) = run_offline(&build(offset));
        if interior_boundaries(&seg) >= min_interior {
            println!(
                "MEASURE crossing[{label}]: output-bias offset {offset:+} -> {} interior \
                 boundaries ({} segment rows)",
                interior_boundaries(&seg),
                seg.segments().len()
            );
            return offset;
        }
    }
    panic!("{label}: no output-bias offset produced >= {min_interior} interior boundaries");
}

/// The plain (type-0) sweep: the committed fixture config at `offset`.
fn plain_crossing(dir: &Path, cell: &str, tag: &str) -> f64 {
    crossing_offset(cell, MIN_INTERIOR_PLAIN, |o| stage(dir, cell, o, tag))
}

// ---------------------------------------------------------------------------
// (a) OFFLINE EQUIVALENCE -- the phase gate.
// ---------------------------------------------------------------------------

#[test]
fn stream_finish_equals_offline_causal() {
    for cell in ["slstm", "mamba"] {
        let dir = tempfile::tempdir().unwrap();
        let offset = plain_crossing(dir.path(), cell, "sweep");
        let st = stage(dir.path(), cell, offset, "gate");
        let (off_post, off_seg) = run_offline(&st);
        let (rate, samples) = mono_samples(&st.wav);
        let run = run_session(
            &st.config_text,
            rate,
            &samples,
            (PUSH_CHUNK_S * rate) as usize,
        );

        assert!(
            run.is_causal,
            "{cell}: the session must take the causal arm"
        );
        let max_dt = boundary_check(&run.seg, &off_seg, cell);
        println!(
            "MEASURE gate[{cell}]: seg_rows={} interior={} boundary_max_dt={max_dt:.3e} \
             post_len={} emissions={}",
            run.seg.segments().len(),
            interior_boundaries(&off_seg),
            off_post.len(),
            run.emission_count
        );
        assert_eq!(
            max_dt, 0.0,
            "{cell}: boundary max_dt must be EXACTLY 0.0 (R1 STOP on nonzero)"
        );
        assert_posteriors_bit_equal(&run.posteriors, &off_post, &format!("{cell} posteriors"));
        // Non-vacuity: real interior boundaries AND a posterior sequence that moves.
        assert!(
            interior_boundaries(&off_seg) >= MIN_INTERIOR_PLAIN,
            "{cell}: the boundary comparison is vacuous (no interior boundary)"
        );
        let first = off_post[0];
        assert!(
            off_post.iter().any(|&v| v != first),
            "{cell}: the posterior is constant -- the comparison is vacuous"
        );
    }
}

// ---------------------------------------------------------------------------
// (a') OFFLINE EQUIVALENCE -- the FROZEN TYPE-1 tail on the causal arm.
// ---------------------------------------------------------------------------

#[test]
fn stream_finish_equals_offline_causal_frozen_type1() {
    // The committed causal fixtures carry `InputNormalizationType 0`, so leg (a) alone
    // never exercises the frozen type-1 tail through the CAUSAL arm -- the phase-8
    // mechanism this task reuses rather than reimplements. This leg calibrates the tail to
    // the fixture's own statistics (so it is NOT the identity), asserts the streamed run is
    // still bit-identical to offline, and asserts the type-1 posteriors DIFFER from the
    // type-0 ones (else the threading claim would be vacuous).
    for cell in ["slstm", "mamba"] {
        let dir = tempfile::tempdir().unwrap();
        // Sweep the crossing on the CALIBRATED config: the calibrated tail shifts the
        // posterior level, so the type-0 sweep's offset does not carry over (measured: at
        // offset 0 the sLSTM type-1 run collapses to the 2-row seed).
        let offset = crossing_offset(&format!("{cell}/type1"), MIN_INTERIOR_TYPE1, |o| {
            let b = stage(dir.path(), cell, o, "t1sweep");
            calibrate(dir.path(), &b, cell, "t1sweep")
        });
        let base = stage(dir.path(), cell, offset, "t1base");
        let cal = calibrate(dir.path(), &base, cell, "t1");
        let (off_post, off_seg) = run_offline(&cal);
        let (plain_post, _) = run_offline(&base);
        let (rate, samples) = mono_samples(&cal.wav);
        let run = run_session(
            &cal.config_text,
            rate,
            &samples,
            (PUSH_CHUNK_S * rate) as usize,
        );

        assert!(run.is_causal, "{cell}: type-1 leg must take the causal arm");
        let max_dt = boundary_check(&run.seg, &off_seg, &format!("{cell}/type1"));
        println!(
            "MEASURE type1[{cell}]: seg_rows={} interior={} boundary_max_dt={max_dt:.3e} \
             post_len={}",
            run.seg.segments().len(),
            interior_boundaries(&off_seg),
            off_post.len()
        );
        assert_eq!(
            max_dt, 0.0,
            "{cell}/type1: boundary max_dt must be EXACTLY 0.0 (R1)"
        );
        assert_posteriors_bit_equal(
            &run.posteriors,
            &off_post,
            &format!("{cell}/type1 posteriors"),
        );
        assert!(
            interior_boundaries(&off_seg) >= MIN_INTERIOR_TYPE1,
            "{cell}/type1: the boundary comparison is vacuous (no interior boundary)"
        );
        // Non-vacuity: the calibrated tail is NOT the identity, so it genuinely moved the
        // posteriors -- the type-1 threading is doing work on this leg.
        assert_eq!(off_post.len(), plain_post.len());
        assert!(
            off_post
                .iter()
                .zip(plain_post.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "{cell}/type1: the calibrated tail left the posteriors unchanged -- the leg is \
             vacuous (an identity tail proves no threading)"
        );
    }
}

// ---------------------------------------------------------------------------
// (b) CHUNKING BIT-INVARIANCE (20/100/1000/7 ms).
// ---------------------------------------------------------------------------

#[test]
fn chunking_bit_invariance() {
    for cell in ["slstm", "mamba"] {
        let dir = tempfile::tempdir().unwrap();
        let offset = plain_crossing(dir.path(), cell, "sweep");
        let st = stage(dir.path(), cell, offset, "chunks");
        let (rate, samples) = mono_samples(&st.wav);
        let chunks = [
            (0.020 * rate) as usize, // 160
            (0.100 * rate) as usize, // 800
            (1.000 * rate) as usize, // 8000
            (0.007 * rate) as usize, // 56 (non-divisor)
        ];
        let runs: Vec<SessionRun> = chunks
            .iter()
            .map(|&c| run_session(&st.config_text, rate, &samples, c))
            .collect();

        let ref_seg = seg_identities(&runs[0].seg);
        let ref_post = &runs[0].posteriors;
        let ref_em = emission_identities(&runs[0].all_emissions());
        println!(
            "MEASURE chunk_invariance[{cell}]: seg_rows={} post_len={} emissions={}",
            runs[0].seg.segments().len(),
            ref_post.len(),
            ref_em.len()
        );
        // Non-vacuity: a run with no segments would make every comparison trivial.
        assert!(
            runs[0].seg.segments().len() > 2 && !ref_em.is_empty(),
            "{cell}: the chunking leg needs a real boundary set"
        );
        for (k, run) in runs.iter().enumerate().skip(1) {
            assert_eq!(
                seg_identities(&run.seg),
                ref_seg,
                "[{cell}] chunk {} vs chunk 0: finish Segmentation must be bit-identical",
                chunks[k]
            );
            assert_f32_bit_equal(
                &run.posteriors,
                ref_post,
                &format!("[{cell}] chunk {} posterior history", chunks[k]),
            );
            assert_eq!(
                emission_identities(&run.all_emissions()),
                ref_em,
                "[{cell}] chunk {} vs chunk 0: emitted SET must be bit-identical",
                chunks[k]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (c) PREFIX CONSISTENCY -- every mid-stream emission survives to finish().
// ---------------------------------------------------------------------------

#[test]
fn prefix_consistency_e2e() {
    for cell in ["slstm", "mamba"] {
        let dir = tempfile::tempdir().unwrap();
        let offset = plain_crossing(dir.path(), cell, "sweep");
        let st = stage(dir.path(), cell, offset, "prefix");
        let (rate, samples) = mono_samples(&st.wav);
        let run = run_session(
            &st.config_text,
            rate,
            &samples,
            (PUSH_CHUNK_S * rate) as usize,
        );

        let final_ids: std::collections::HashSet<(u64, u64, i32)> = run
            .seg
            .segments()
            .windows(2)
            .map(|w| (w[0].begin.to_bits(), w[1].begin.to_bits(), w[0].ty as i32))
            .collect();
        let all = run.all_emissions();
        for e in &all {
            let id = (e.begin_s.to_bits(), e.end_s.to_bits(), e.class as i32);
            assert!(
                final_ids.contains(&id),
                "{cell}: emission {:?} not present in the final partition (retraction -- R2)",
                (e.begin_s, e.end_s, e.class)
            );
        }
        assert_eq!(
            emission_identities(&all).len(),
            final_ids.len(),
            "{cell}: the emitted set must equal the final partition size"
        );
        println!(
            "MEASURE prefix[{cell}]: midstream={} total_emitted={} final_segments={}",
            run.push_emissions.len(),
            all.len(),
            final_ids.len()
        );
        assert!(
            !run.push_emissions.is_empty(),
            "{cell}: non-vacuity -- mid-stream emissions must occur"
        );
    }
}

// ---------------------------------------------------------------------------
// (d) VALIDATION BAILS (spec S5.3, each pinned).
// ---------------------------------------------------------------------------

fn bail_msg(r: anyhow::Result<StreamingSession>) -> String {
    match r {
        Ok(_) => panic!("expected a validation bail, got Ok"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn validation_bails() {
    let dir = tempfile::tempdir().unwrap();
    let st = stage(dir.path(), "slstm", 0.0, "bails");
    let text = &st.config_text;
    let rate = 8000.0;

    // Sanity FIRST: the un-mutated causal config constructs, on the CAUSAL arm.
    let m = parse(text);
    let sess = StreamingSession::new(&m, rate, 1).expect("the staged causal config must build");
    assert!(
        sess.is_causal(),
        "the staged config must select the causal arm"
    );

    // (1) algo != 3.
    let mut m = parse(text);
    m.insert("Algo_choice".into(), "4".into());
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(msg.contains("algo 3"), "algo-4 bail message: {msg}");

    // (2) multi-channel (MONO-first).
    let m = parse(text);
    let msg = bail_msg(StreamingSession::new(&m, rate, 2));
    assert!(msg.contains("mono only"), "stereo bail message: {msg}");

    // (3) Audio_fixed_gain absent -- the causality cut is REQUIRED even at
    // InputNormalizationType 0, because the whole-file `(2*RMS+max)/2` AUDIO norm is a
    // separate statistic from the feature normalization.
    let mut m = parse(text);
    m.shift_remove("Audio_fixed_gain");
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(
        msg.contains("Audio_fixed_gain"),
        "missing-gain bail message: {msg}"
    );

    // (4) InputNormalizationType -1 (self-normalization) still bails; 0 and 1 are the
    // accepted set (0 is what the fixture carries, proven by the sanity build above).
    let mut m = parse(text);
    m.insert("BLSTM_InputNormalizationType".into(), "-1".into());
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(
        msg.contains("InputNormalizationType 1") && msg.contains("or 0"),
        "type -1 bail message: {msg}"
    );

    // (5) A causal config carrying a WINDOW: bailed STREAMING-side (a causal cell's state
    // would be reset at every window boundary). 0.5 s at 8 kHz with spectrum_shift 0.01
    // resolves window_size 25 -- a genuinely windowed dispatch, not a degenerate one.
    let mut m = parse(text);
    m.insert("BLSTM_window".into(), "0.5".into());
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(
        msg.contains("causal streaming requires the plain regime"),
        "causal-window bail message: {msg}"
    );

    // (6) A BIDIRECTIONAL new cell -- refused through the shared `classify_fast_shape`
    // choke point, with the SAME wording the offline fast driver uses.
    let mut m = parse(text);
    m.insert("BLSTM_Direction".into(), "bidirectional".into());
    m.insert("BLSTM_OutputNeuronNb".into(), "8,1".into());
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(
        msg.contains("cell type 'slstm' is not supported on the fast inference path")
            && msg.contains("BIDIRECTIONAL"),
        "bidirectional-cell bail message: {msg}"
    );

    // (7) The CAUSAL direction with the legacy LSTM cell -- a forward-only LSTM fast twin
    // is a named follow-on (spec S5.3), so streaming refuses it too.
    let mut m = parse(text);
    m.insert("BLSTM_Cell_Type".into(), "lstm".into());
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(
        msg.contains("Direction 'forward' is not supported on the fast inference path"),
        "causal-LSTM bail message: {msg}"
    );

    // (8) An empty weights file (the frozen net must be loaded once).
    let mut m = parse(text);
    m.insert("BLSTM_weightsFile".into(), String::new());
    let msg = bail_msg(StreamingSession::new(&m, rate, 1));
    assert!(
        msg.contains("BLSTM_weightsFile is empty"),
        "empty-weightsFile bail message: {msg}"
    );
}

#[test]
fn causal_output_size_not_one_bails() {
    // The binary-SAD-only construction bail on the CAUSAL arm: widen the output net to 2
    // neurons and synthesize a ZEROS pack sized to that spec (so `FastCausalNet::from_flat`
    // SUCCEEDS and the flow reaches the output_size check, rather than tripping a
    // too-short-pack bail first).
    let dir = tempfile::tempdir().unwrap();
    let st = stage(dir.path(), "slstm", 0.0, "out2");
    let mut m = parse(&st.config_text);
    m.insert("BLSTM_OutputNeuronNb".into(), "4,2".into());

    let bc = speech::nn::blstm::BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let spec = speech::config::NnetSpec::from_legacy(&m, "BLSTM").unwrap();
    let needed = speech::fast::cells::FastCausalNet::element_count(
        &spec,
        bc.cell_type,
        &speech::nn::blstm::MambaParams::default(),
    )
    .unwrap();
    let pack = dir.path().join("causal_out2.bin");
    write_matrix(&pack, needed, 1, &vec![0.0; needed]).unwrap();
    m.insert("BLSTM_weightsFile".into(), pack.to_str().unwrap().into());

    let msg = bail_msg(StreamingSession::new(&m, 8000.0, 1));
    assert!(
        msg.contains("output_size 1"),
        "output-2 bail message: {msg}"
    );
}

// ---------------------------------------------------------------------------
// (d') THE S5.2 DISPATCH: the windowed arm is still selected for lstm + bidirectional.
// ---------------------------------------------------------------------------

#[test]
fn dispatch_keeps_the_windowed_arm_for_the_legacy_net() {
    // The phase-8 staged tier2 config (LSTM + bidirectional + a real window) must still
    // build the WINDOWED session -- the causal dispatch is additive, not a replacement.
    // `phase8_gate.rs` proves the windowed BEHAVIOUR; this pins the ARM SELECTION, which is
    // the new decision Task 7 introduced.
    let dir = tempfile::tempdir().unwrap();
    let base = common::stage_frozen_tier2(dir.path());
    let text = std::fs::read_to_string(&base.config_path).unwrap();
    let m = parse(&text);
    let sess = StreamingSession::new(&m, 8000.0, 1).unwrap();
    assert!(
        !sess.is_causal(),
        "an lstm/bidirectional config must keep the phase-8 windowed arm"
    );
    // ...and its latency budget keeps the phase-8 SHAPE: a nonzero nn_window, no
    // sub-sample term (so the phase-8 bound is bit-unchanged by the new component).
    assert!(
        sess.nn_window_s() > 0.0,
        "windowed arm must carry nn_window"
    );
    assert_eq!(
        sess.sub_sample_s(),
        0.0,
        "the windowed arm must carry NO sub-sample term"
    );
}

// ---------------------------------------------------------------------------
// (e) LATENCY: the DERIVED structural bound + the per-class measured pins (spec S5.5).
// ---------------------------------------------------------------------------

#[test]
fn latency_bounds() {
    // THE DERIVED CAUSAL BOUND. Spec S5.5: the `nn_window` term DIES (a causal cell has no
    // lookahead at all), leaving
    //
    //   bound = feature_reach + sub_sample + conv_delay + holdback
    //
    // ALL config-derived, never a literal. On the phase-9 gate fixture
    // (rate 8000, spectrum_shift 0.01 -> ssif 80, spectrum_order 10 -> window_size 1024,
    // ComputeDeltasNb 5 + ComputeDeltaDeltasNb 3 -> reach 8, LSTMSubSampling 4 -> ssr 4,
    // convolution_window_size 4 -> 9 taps -> conv_half 4, BLSTM_window 0 -> window_shift
    // floored to 1 -> window_shift_sec 0.01 -> time_step 0.04):
    //
    //   feature_reach = (reach 8 * ssif 80 + half_window 512) / 8000 = 1152/8000 = 0.14400 s
    //   nn_window     = 0.00000 s  (NO WINDOW -- the phase-9 win; phase 8 paid 3.26000 s)
    //   sub_sample    = (ssr 4 - 1) * spectrum_shift 0.01                = 0.03000 s
    //   conv_delay    = conv_half 4 * time_step 0.04                    = 0.16000 s
    //   holdback      = sum(min_speech 0.2*3) + sum(min_silence 0.2*2)
    //                   + sum(padding 0.1*4) = 0.6 + 0.4 + 0.4          = 1.40000 s
    //   DERIVED BOUND                                                    = 1.73400 s
    //
    // versus phase 8's 6.05357 s on the windowed arm -- a 3.5x structural improvement that
    // comes entirely from dropping the window lookahead.
    //
    // THE PER-CLASS SPLIT is the phase-8 pattern verbatim (spec S1.8): a SPEECH segment
    // emits during the FOLLOWING silence at ~bound (the consumed-frontier time-advance);
    // an OTHER (silence) segment's right boundary IS the onset of the following speech,
    // which enters `raw_segments` only when that speech COMMITS at its falling edge, so its
    // lag is that speech's DURATION plus the forward pipeline delay -- the data-dependent
    // AREA term, inherent to closed-interval raw segments and NOT reduced by the trigger.
    for cell in ["slstm", "mamba"] {
        let dir = tempfile::tempdir().unwrap();
        let offset = plain_crossing(dir.path(), cell, "sweep");
        let st = stage(dir.path(), cell, offset, "latency");
        let (rate, samples) = mono_samples(&st.wav);
        let run = run_session(
            &st.config_text,
            rate,
            &samples,
            (PUSH_CHUNK_S * rate) as usize,
        );

        // Cross-check every component against a from-config recomputation, BY BITS, so the
        // derivation is COMPUTED rather than a session-internal literal we trust blindly.
        let map = parse(&st.config_text);
        let feature_cfg = FeatureConfig::from_legacy(&map, "BLSTM").unwrap();
        let params = SpectralParams::derive(&feature_cfg, rate);
        let seg_cfg = SegmenterConfig::from_config(&map, "BLSTM").unwrap();
        let driver_cfg = DriverConfig::from_config(&map, "BLSTM").unwrap();
        let bc = speech::nn::blstm::BlstmConfig::from_legacy(&map, "BLSTM").unwrap();
        let spec = speech::config::NnetSpec::from_legacy(&map, "BLSTM").unwrap();
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
        assert_eq!(
            window_size, 0,
            "{cell}: the causal fixture must resolve window_size 0"
        );
        assert_eq!(bc.direction.as_str(), "forward", "{cell}: causal fixture");
        // The causal (window 0) time base: `_WindowShift * ssr` (tasks/sad.rs:1451-1452,
        // fast/driver.rs:530-534), with `ws` the POST-`get_blstm_param` value.
        let re_time_step = ws * ssr as f64;
        let re_nn = 0.0;
        let re_sub = (ssr - 1) as f64 * params.shift_sec;
        let conv_half = driver_cfg
            .conv_coeff
            .as_deref()
            .map_or(0, |c| if c.len() > 1 { (c.len() - 1) / 2 } else { 0 });
        let re_conv = conv_half as f64 * re_time_step;
        let re_holdback = seg_cfg.min_speech.iter().sum::<f64>()
            + seg_cfg.min_silence.iter().sum::<f64>()
            + seg_cfg.padding.iter().sum::<f64>();
        let re_bound = re_feature + re_nn + re_sub + re_conv + re_holdback;

        assert_eq!(
            run.feature_reach.to_bits(),
            re_feature.to_bits(),
            "{cell}: feature reach"
        );
        assert_eq!(
            run.nn_window.to_bits(),
            re_nn.to_bits(),
            "{cell}: nn window (must be 0)"
        );
        assert_eq!(
            run.sub_sample.to_bits(),
            re_sub.to_bits(),
            "{cell}: sub-sample buffering"
        );
        assert_eq!(
            run.conv_delay.to_bits(),
            re_conv.to_bits(),
            "{cell}: conv delay"
        );
        assert_eq!(
            run.holdback.to_bits(),
            re_holdback.to_bits(),
            "{cell}: holdback"
        );
        assert_eq!(
            run.bound.to_bits(),
            re_bound.to_bits(),
            "{cell}: bound = sum of components"
        );

        let pipeline_forward = run.feature_reach + run.nn_window + run.sub_sample + run.conv_delay;
        let speech_push_max = run
            .push_emissions
            .iter()
            .filter(|e| e.class == SegClass::Speech)
            .map(|e| e.emitted_at_audio_s - e.end_s)
            .fold(f64::NEG_INFINITY, f64::max);
        let other_push_max = run
            .push_emissions
            .iter()
            .filter(|e| e.class == SegClass::Other)
            .map(|e| e.emitted_at_audio_s - e.end_s)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_speech_dur = run
            .seg
            .segments()
            .windows(2)
            .filter(|w| w[0].ty == SegClass::Speech)
            .map(|w| w[1].begin - w[0].begin)
            .fold(0.0_f64, f64::max);
        println!(
            "MEASURE latency[{cell}]: feature_reach={:.5} nn_window={:.5} sub_sample={:.5} \
             conv_delay={:.5} holdback={:.5} bound={:.5} pipeline_forward={pipeline_forward:.5} \
             speech_push_max={speech_push_max:.5} other_push_max={other_push_max:.5} \
             max_speech_dur={max_speech_dur:.5} tight_margin={:+.5} push={} finish={} \
             seg_rows={}",
            run.feature_reach,
            run.nn_window,
            run.sub_sample,
            run.conv_delay,
            run.holdback,
            run.bound,
            speech_push_max - run.bound - PUSH_CHUNK_S,
            run.push_emissions.len(),
            run.finish_emissions.len(),
            run.seg.segments().len()
        );

        // Non-vacuity: real mid-stream emissions in BOTH classes.
        assert!(
            run.emission_count > 1,
            "{cell}: latency leg needs real emissions"
        );
        assert!(
            speech_push_max.is_finite() && other_push_max.is_finite(),
            "{cell}: latency leg needs BOTH speech and other mid-stream emissions \
             (speech={speech_push_max} other={other_push_max})"
        );

        // (1) THE STRUCTURAL WIN (tight): every SPEECH emission lands within the DERIVED
        // bound (+ the SAME 0.5 s allowance `phase8_gate.rs::latency_bounds` uses, for the
        // ssr grid slack, the 100 ms push granularity and the smoothing boundary shifts).
        // MEASURED here the allowance is genuinely consumed, unlike phase 8's: slstm
        // 1.80148 and mamba 1.82867 sit ~0.07-0.09 s ABOVE the 1.73400 s structural bound,
        // which the 100 ms push quantum alone accounts for (phase 8's UNBLOCKED 5.84457 s sat under
        // its 6.05357 s bound only because that bound is 3.5x larger, not because the
        // quantization was absent). The bound stays purely CONFIG-derived -- the chunk size
        // is a runtime choice, so folding it into the derivation would be wrong. A
        // regression to a wait-for-next-raw-segment cadence, or a reintroduced window
        // lookahead, still fails here LOUDLY (both are >= 1 s effects).
        const SPEECH_ALLOWANCE: f64 = 0.5;
        assert!(
            speech_push_max <= run.bound + SPEECH_ALLOWANCE,
            "{cell}: SPEECH mid-stream max lag {speech_push_max} exceeds the structural bound \
             {} + allowance {SPEECH_ALLOWANCE}",
            run.bound
        );
        // ...and the bound is genuinely INCURRED, not trivially satisfied (phase 8's
        // companion non-vacuity floor: the holdback alone is well below it).
        assert!(
            speech_push_max > run.holdback,
            "{cell}: SPEECH mid-stream max lag {speech_push_max} must exceed the holdback alone \
             {} (else the pipeline delay is not being measured at all)",
            run.holdback
        );
        // THE TIGHT COMPANION -- the pin that actually asserts S5.5's claim. The 0.5 s
        // sibling above is 29% of a 1.73400 s bound (phase 8's identical allowance was 8% of
        // 6.05357 s and went unconsumed), so on its own it would let a real regression of up
        // to half a second pass unnoticed. The ONLY structural reason a SPEECH emission may
        // land past the config-derived bound is the push quantum: the emission clock is
        // `(total_pushed-1)/rate`, advanced once per `push`, so a segment that settles just
        // after a push is stamped up to one chunk late. Pin exactly that, no slack beyond it,
        // turning "the push quantum accounts for the excess" from narrative into a bound.
        // MEASURED margin at 100 ms chunks (printed as `tight_margin` above): -0.03252 s
        // (slstm) / -0.00533 s (mamba) -- both under, and mamba by only ~5 ms. That thinness
        // is the point: the pin sits right where the structural argument predicts, so it has
        // real discriminating power. It also means this leg is chunk-size-COUPLED -- running
        // the gate at a coarser granularity would legitimately need `chunk_s` to follow the
        // actual chunk (it is hardcoded to the 0.1 the leg pushes at, deliberately, so the
        // two cannot silently drift apart).
        // A FAILURE HERE IS NOT A FAILURE TO WIDEN (R1): it means an emission waited on
        // something other than the derived pipeline + one chunk, which is a latency
        // regression to adjudicate, not a constant to bump.
        let chunk_s = PUSH_CHUNK_S;
        assert!(
            speech_push_max <= run.bound + chunk_s,
            "{cell}: SPEECH mid-stream max lag {speech_push_max} exceeds the derived bound {} \
             plus ONE push quantum {chunk_s} (margin {:+.5}) -- the excess is no longer \
             explained by the emission clock's granularity",
            run.bound,
            speech_push_max - run.bound - chunk_s
        );

        // (2) THE SILENCE-COMMIT WAIT (honest, data-dependent): an OTHER segment cannot
        // emit until the following speech commits at its falling edge. Same shape as the
        // phase-8 pin, term for term.
        const OTHER_ALLOWANCE: f64 = 1.0;
        let silence_commit_bound = max_speech_dur + pipeline_forward + OTHER_ALLOWANCE;
        assert!(
            other_push_max <= silence_commit_bound,
            "{cell}: OTHER mid-stream max lag {other_push_max} exceeds max_speech_dur \
             {max_speech_dur} + pipeline_forward {pipeline_forward} + allowance \
             {OTHER_ALLOWANCE}"
        );

        // (3) THE CAUSAL WIN IS REAL, not a bookkeeping relabel: the phase-8 windowed arm's
        // structural bound on its own gate config is 6.05357 s, dominated by a 3.26 s window
        // lookahead. Assert the causal bound is a small fraction of it AND that no window
        // term crept back in.
        assert_eq!(
            run.nn_window, 0.0,
            "{cell}: the causal arm must carry NO window lookahead"
        );
        assert!(
            run.bound < 2.5,
            "{cell}: the derived causal bound {} regressed (phase 8's windowed arm pays \
             6.05357 s; the causal arm should be under ~2 s on this config)",
            run.bound
        );
    }
}

// ---------------------------------------------------------------------------
// (e') THE `speech stream` CLI ON A CAUSAL CONFIG (spec S5.2: "ZERO new surface").
// ---------------------------------------------------------------------------

#[test]
fn stream_cli_drives_the_causal_arm() {
    // The S5.2 claim is that the CLI gains the causal mode for FREE, because the dispatch
    // lives inside `StreamingSession::new` and `stream_cli.rs` is untouched. Claiming that
    // is cheap; proving it is this leg: spawn the REAL binary on a causal config and
    // require its printed partition to equal the in-process session's. `phase8_cli.rs`
    // pins the CLI's own formatting/summary contract on the windowed arm -- not repeated
    // here, since only the ARM changed.
    let dir = tempfile::tempdir().unwrap();
    let offset = plain_crossing(dir.path(), "slstm", "sweep");
    let st = stage(dir.path(), "slstm", offset, "cli");
    let cfg = dir.path().join("causal_cli.config");
    std::fs::write(&cfg, &st.config_text).unwrap();
    let (rate, samples) = mono_samples(&st.wav);

    // The in-process oracle at the SAME 100 ms chunking the CLI flag selects.
    let run = run_session(
        &st.config_text,
        rate,
        &samples,
        (PUSH_CHUNK_S * rate) as usize,
    );
    assert!(run.is_causal, "the CLI leg must exercise the causal arm");
    let ref_set: std::collections::HashSet<(String, String, String)> = run
        .seg
        .segments()
        .windows(2)
        .map(|w| {
            (
                format!("{:.4}", w[0].begin),
                format!("{:.4}", w[1].begin),
                speech::stream_cli::class_str(w[0].ty),
            )
        })
        .collect();
    assert!(
        ref_set.len() > 2,
        "non-vacuity: the CLI leg needs a real partition, got {}",
        ref_set.len()
    );

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_speech"))
        .args([
            "stream",
            "--chunk-ms=100",
            cfg.to_str().unwrap(),
            st.wav.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run the speech binary");
    assert!(
        output.status.success(),
        "speech stream exited non-zero on a causal config: {:?}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let seg_lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("SEG ")).collect();
    assert_eq!(
        stdout.lines().filter(|l| l.starts_with("STREAM ")).count(),
        1,
        "expected exactly one STREAM summary line:\n{stdout}"
    );

    let cli_set: std::collections::HashSet<(String, String, String)> = seg_lines
        .iter()
        .map(|l| {
            let mut b = None;
            let mut e = None;
            let mut c = None;
            for tok in l.split_whitespace().skip(1) {
                let (k, v) = tok.split_once('=').expect("SEG token missing '='");
                match k {
                    "begin" => b = Some(v.to_string()),
                    "end" => e = Some(v.to_string()),
                    "class" => c = Some(v.to_string()),
                    _ => {}
                }
            }
            (b.unwrap(), e.unwrap(), c.unwrap())
        })
        .collect();
    assert_eq!(
        cli_set, ref_set,
        "the CLI's emitted SEG set must equal the causal session's final partition"
    );
    assert_eq!(
        seg_lines.len(),
        cli_set.len(),
        "SEG lines must be unique (no re-emission)"
    );
    println!(
        "MEASURE stream_cli_causal: seg_lines={} partition={}",
        seg_lines.len(),
        ref_set.len()
    );
}

// ---------------------------------------------------------------------------
// (f) THE REAL ARM'S SHAPE: a STACKED, SUB-SAMPLED net with a WIDE hidden dense
// layer -- the geometry no committed fixture covers.
// ---------------------------------------------------------------------------

/// Build a `FastCausalNet` at an arbitrary geometry from a bounded deterministic pack.
fn synth_net(
    cell: speech::nn::blstm::CellType,
    lstm: &[usize],
    lsub: &[usize],
    outn: &[usize],
    osub: &[usize],
) -> speech::fast::cells::FastCausalNet {
    let sp = speech::config::NnetSpec {
        lstm_neuron_nb: lstm.to_vec(),
        lstm_subsampling: lsub.to_vec(),
        output_neuron_nb: outn.to_vec(),
        output_subsampling: osub.to_vec(),
        input_size: lstm[0],
        peepholes: [false; 6],
    };
    let p = speech::nn::blstm::MambaParams::default();
    let n = speech::fast::cells::FastCausalNet::element_count(&sp, cell, &p).unwrap();
    // Bounded, non-degenerate: a linear ramp would saturate the output layer to a constant.
    let flat: Vec<f64> = (0..n)
        .map(|k| 0.35 * (0.61 * (k as f64) + 0.3).sin())
        .collect();
    speech::fast::cells::FastCausalNet::from_flat(&sp, cell, &p, &flat).unwrap()
}

/// A deterministic, bounded, non-constant input sequence.
fn synth_input(rows: usize, cols: usize) -> speech::fast::nn::FastMatrix {
    let mut m = speech::fast::nn::FastMatrix::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            m.data[r * cols + c] = (0.21 * (r as f64) - 0.13 * (c as f64)
                + 0.017 * ((r * cols + c) as f64))
                .sin() as f32;
        }
    }
    m
}

/// Push `input` through a fresh `StreamCausal` in `chunk`-row batches and concatenate.
fn stream_rows(
    net: &speech::fast::cells::FastCausalNet,
    input: &speech::fast::nn::FastMatrix,
    chunk: usize,
) -> speech::fast::nn::FastMatrix {
    let mut sc = speech::fast::stream::StreamCausal::new(net.clone());
    let mut data: Vec<f32> = Vec::new();
    let mut rows = 0usize;
    let mut i = 0;
    while i < input.rows {
        let end = (i + chunk).min(input.rows);
        let batch = speech::fast::nn::FastMatrix {
            data: input.data[i * input.cols..end * input.cols].to_vec(),
            rows: end - i,
            cols: input.cols,
        };
        let out = sc.push_rows(&batch);
        data.extend_from_slice(&out.data);
        rows += out.rows;
        i = end;
    }
    let tail = sc.flush();
    assert_eq!(
        tail.rows, 0,
        "StreamCausal::flush must produce no tail rows"
    );
    speech::fast::nn::FastMatrix {
        data,
        rows,
        cols: sc.output_size(),
    }
}

#[test]
fn stream_causal_matches_offline_on_the_real_arm_geometry() {
    // THE GAP THIS CLOSES. Every committed phase-9 fixture is a SINGLE cell layer with a
    // single `4 -> 1` output layer. The phase's real causal SAD arm
    // (`configs/training/lre_sad.toml` under the `Direction forward` overlay) is
    // `lstm_neuron_nb = "23,24,24"` / `lstm_sub_sampling = "4,1"` /
    // `output_neuron_nb = "24,12,1"` -- a STACK, a per-layer sub-sampling chain, AND a WIDE
    // hidden dense layer (12 outputs), which is exactly the shape where the batched faer
    // dense projection is NOT row-count-independent (see `fast::nn::DenseRowChain`'s docs).
    // Task 9's corpus leg streams that checkpoint, so the streamed-vs-offline identity has
    // to hold THERE, not only on the fixtures.
    for cell in [
        speech::nn::blstm::CellType::Slstm,
        speech::nn::blstm::CellType::Mamba,
    ] {
        let mut net = synth_net(cell, &[23, 24, 24], &[4, 1], &[24, 12, 1], &[1, 1]);
        // 143 rows: NOT a multiple of the layer-0 ratio 4, so the trailing `143 mod 4 == 3`
        // rows are DROPPED on both sides (the offline `sub_sample` truncation).
        let input = synth_input(143, 23);
        let offline = net.feed_forward(&input).clone();
        assert_eq!(
            (offline.rows, offline.cols),
            (35, 1),
            "{cell:?}: 143/4 = 35 decimated rows"
        );
        // Non-vacuity: the offline posterior must actually vary, or a bit comparison
        // against it is meaningless.
        assert!(
            offline.data.iter().any(|&v| v != offline.data[0]),
            "{cell:?}: the offline posterior is constant -- the comparison is vacuous"
        );

        for chunk in [1usize, 3, 4, 7, 143] {
            let streamed = stream_rows(&net, &input, chunk);
            assert_eq!(
                (streamed.rows, streamed.cols),
                (offline.rows, offline.cols),
                "{cell:?} chunk {chunk}: posterior shape"
            );
            for k in 0..offline.data.len() {
                assert_eq!(
                    streamed.data[k].to_bits(),
                    offline.data[k].to_bits(),
                    "{cell:?} chunk {chunk}: posterior bits at {k} \
                     (streamed {} vs offline {})",
                    streamed.data[k],
                    offline.data[k]
                );
            }
        }
        println!(
            "MEASURE real_arm_shape[{cell:?}]: 143 rows -> 35 posteriors, bit-identical at chunks 1/3/4/7/143"
        );
    }
}

#[test]
fn stream_causal_drops_the_trailing_subsample_group() {
    // The `T mod R` truncation, isolated: a sequence SHORTER than the total ratio produces
    // NOTHING at all (offline `feed_forward`'s `out_len == 0` early return), and adding rows
    // one at a time makes posteriors appear exactly on the ratio boundaries.
    let net = synth_net(
        speech::nn::blstm::CellType::Slstm,
        &[6, 5, 4],
        &[2, 2],
        &[4, 1],
        &[1],
    );
    assert_eq!(net.sub_sampling_ratio(), 4);
    let mut sc = speech::fast::stream::StreamCausal::new(net.clone());
    let input = synth_input(13, 6);
    let mut produced = Vec::new();
    for r in 0..input.rows {
        let batch = speech::fast::nn::FastMatrix {
            data: input.row(r).to_vec(),
            rows: 1,
            cols: input.cols,
        };
        produced.push(sc.push_rows(&batch).rows);
    }
    // Ratios 2 then 2: a posterior lands on rows 4, 8, 12 (1-based), i.e. indices 3/7/11.
    assert_eq!(
        produced,
        vec![0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        "posteriors must land exactly on the ssr boundaries"
    );
    assert_eq!(
        sc.emitted(),
        3,
        "13/4 = 3 posterior rows, trailing 1 dropped"
    );
    let mut net2 = net;
    assert_eq!(net2.feed_forward(&input).rows, 3, "offline agrees");
}
