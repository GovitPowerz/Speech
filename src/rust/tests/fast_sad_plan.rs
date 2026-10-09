//! Issue #24: `fast::plan::FastSadPlan` / `SadTimeline`, the ONE algo-3 SAD setup the
//! offline fast driver and the streaming session consume.
//!
//! What is pinned here, and why each leg exists:
//!  - the plan agrees with the EXACT tree on every committed SAD config (shape, pack
//!    length, sub-sampling ratio, normalization type, geometry): the plan is a reader of
//!    the same config, not a second opinion;
//!  - the timeline reproduces the frozen per-file derivation (`SpectralParams::derive` +
//!    `get_blstm_param` + the time-axis formula) at two rates, and `rows_for` is the
//!    frozen `real_vec_size`;
//!  - the stateful shift re-quantization the fast tree DROPPED is a fixed point at one rate
//!    (the by-design divergence's justification, measured over a rate x shift grid);
//!  - the per-arm normalization table and `check_norm` (one test per arm);
//!  - the rate-free regime pre-check (each refusal, and the BiCell plain acceptance);
//!  - the stream inherits the pitch gate, in the offline driver's words;
//!  - `StreamingSession::from_plan` with an in-memory pack streams exactly as `new` does
//!    off the file, and refuses a wrong-length pack in the exact `set_weights`' words.

mod common;

use std::path::PathBuf;

use indexmap::IndexMap;

use speech::fast::cells::CellGeometry;
use speech::fast::driver::{FastNetShape, FastSpectralSegmenter};
use speech::fast::plan::{FastSadArm, FastSadPlan, accepted_norm_types};
use speech::fast::stream::StreamingSession;
use speech::features::pipeline::SpectralParams;
use speech::io::binary::read_weight_vector;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork, CellType};
use speech::tasks::sad::get_blstm_param;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

fn map_at(rel: &str) -> IndexMap<String, String> {
    let path = ref_dir().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    speech::legacy_config::parse_legacy_config(&text)
}

fn parse(text: &str) -> IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(text)
}

fn err_of<T>(r: anyhow::Result<T>, what: &str) -> String {
    match r {
        Ok(_) => panic!("{what}: must be refused"),
        Err(e) => e.to_string(),
    }
}

/// Every committed algo-3 SAD config, with the shape its name promises.
const CONFIGS: [(&str, FastNetShape); 12] = [
    ("phase4a/tier2_spectral.config", FastNetShape::Blstm),
    (
        "phase9/lstm_forward.config",
        FastNetShape::Causal(CellType::Lstm),
    ),
    (
        "phase9/slstm_forward.config",
        FastNetShape::Causal(CellType::Slstm),
    ),
    (
        "phase9/mamba_forward.config",
        FastNetShape::Causal(CellType::Mamba),
    ),
    (
        "phase9/cfc_forward.config",
        FastNetShape::Causal(CellType::Cfc),
    ),
    (
        "phase9/transformer_forward.config",
        FastNetShape::Causal(CellType::Transformer),
    ),
    (
        "phase9/slstm_bidirectional.config",
        FastNetShape::BiCell(CellType::Slstm),
    ),
    (
        "phase9/mamba_bidirectional.config",
        FastNetShape::BiCell(CellType::Mamba),
    ),
    (
        "phase9/cfc_bidirectional.config",
        FastNetShape::BiCell(CellType::Cfc),
    ),
    (
        "phase9/transformer_bidirectional.config",
        FastNetShape::BiCell(CellType::Transformer),
    ),
    // The two SAD lineages' own configs (ADR-0008), through the TOML importer: the plan
    // reads the same map the bag reads.
    ("../../configs/training/lre_sad.toml", FastNetShape::Blstm),
    (
        "../../configs/training/lre_sad_v2.toml",
        FastNetShape::Blstm,
    ),
];

fn map_for(rel: &str) -> IndexMap<String, String> {
    if rel.ends_with(".toml") {
        let path = ref_dir().join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        speech::toml_config::toml_to_map(&text).unwrap()
    } else {
        map_at(rel)
    }
}

// ---------------------------------------------------------------------------
// The plan agrees with the exact tree on every committed config.
// ---------------------------------------------------------------------------

#[test]
fn plan_agrees_with_the_exact_tree_on_every_committed_config() {
    for (rel, want_shape) in CONFIGS {
        let map = map_for(rel);
        let plan = FastSadPlan::from_map(&map).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let bc = BlstmConfig::from_legacy(&map, "BLSTM").unwrap();
        assert_eq!(plan.shape, want_shape, "{rel}: shape");
        assert_eq!(
            plan.pack_len,
            BlstmNetwork::from_config(bc.clone())
                .unwrap()
                .nb_of_weights(),
            "{rel}: the exact tree's pack length"
        );
        assert_eq!(
            plan.input_normalization_type, bc.input_normalization_type,
            "{rel}: normalization type"
        );
        assert_eq!(plan.geometry, CellGeometry::from(&bc), "{rel}: geometry");
        let ssr = plan.spec.lstm_subsampling.iter().product::<usize>()
            * plan.spec.output_subsampling.iter().product::<usize>();
        assert_eq!(plan.ssr, ssr, "{rel}: ssr");
        assert_eq!(
            plan.weights_file.as_str(),
            map.get("BLSTM_weightsFile")
                .map(String::as_str)
                .unwrap_or(""),
            "{rel}: the weight-file key is named, not read"
        );
        assert_eq!(
            plan.fixed_gain.is_some(),
            map.contains_key("Audio_fixed_gain"),
            "{rel}: Audio_fixed_gain presence"
        );
        // The offline driver exposes the plan it consumes.
        let drv = FastSpectralSegmenter::from_legacy(&map, None).unwrap();
        assert_eq!(drv.plan().shape, want_shape, "{rel}: the driver's plan");
        assert_eq!(
            drv.plan().pack_len,
            plan.pack_len,
            "{rel}: the driver's pack length"
        );
    }
}

#[test]
fn plan_refuses_a_non_algo3_config_and_a_missing_algo() {
    let mut m = map_at("phase4a/tier2_spectral.config");
    m.insert("Algo_choice".into(), "4".into());
    let msg = err_of(FastSadPlan::from_map(&m), "algo 4");
    assert!(msg.contains("only algo 3"), "{msg}");
    m.shift_remove("Algo_choice");
    let msg = err_of(FastSadPlan::from_map(&m), "no algo");
    assert!(msg.contains("'Algo_choice' not found"), "{msg}");
}

// ---------------------------------------------------------------------------
// The timeline reproduces the frozen derivation.
// ---------------------------------------------------------------------------

#[test]
fn timeline_reproduces_the_frozen_per_file_derivation() {
    for (rel, overlap) in [
        ("phase4a/tier2_spectral.config", true),
        ("phase9/slstm_forward.config", false),
        ("phase9/mamba_bidirectional.config", false),
    ] {
        let map = map_at(rel);
        let plan = FastSadPlan::from_map(&map).unwrap();
        for rate in [8000.0, 16000.0] {
            let tl = plan
                .timeline(rate)
                .unwrap_or_else(|e| panic!("{rel}@{rate}: {e}"));
            // SpectralParams, bit for bit.
            let s = SpectralParams::derive(&plan.feature_cfg, rate);
            assert_eq!(tl.ssif, s.shift_frames, "{rel}@{rate}: ssif");
            assert_eq!(
                tl.spectrum_shift_sec.to_bits(),
                s.shift_sec.to_bits(),
                "{rel}@{rate}: spectrum shift"
            );
            assert_eq!(tl.rate.to_bits(), rate.to_bits());
            // get_blstm_param, called directly off the config values.
            let mut ws = plan.driver_cfg.window_shift_sec;
            let (w, sh, no, _) = get_blstm_param(
                plan.driver_cfg.window_size_sec,
                &mut ws,
                rate,
                s.shift_frames,
                plan.ssr,
                &plan.spec.lstm_subsampling,
                &plan.spec.output_subsampling,
                0,
            );
            assert_eq!(
                (tl.window_size, tl.window_shift, tl.no_overlap),
                (w, sh, no),
                "{rel}@{rate}: window/shift/overlap"
            );
            assert_eq!(
                tl.window_shift_sec.to_bits(),
                ws.to_bits(),
                "{rel}@{rate}: the post-call window shift"
            );
            assert_eq!(tl.window_size > 0, overlap, "{rel}@{rate}: regime");
            // The time axis: the OVERLAP branch overrides with the spectrum shift, the
            // plain branch keeps the (mutated) window shift.
            let ssr = plan.ssr as f64;
            let (ts, to) = if overlap {
                let ts = s.shift_sec * ssr;
                (ts, ts / 2.0 - s.shift_sec / 2.0)
            } else {
                let ts = ws * ssr;
                (ts, ts / 2.0 - ws / 2.0)
            };
            assert_eq!(
                tl.time_step.to_bits(),
                ts.to_bits(),
                "{rel}@{rate}: time_step"
            );
            assert_eq!(
                tl.time_offset.to_bits(),
                to.to_bits(),
                "{rel}@{rate}: time_offset"
            );
            // rows_for is the frozen real_vec_size at every frame count.
            for n in [0usize, 1, 79, 80, 81, 8000, 16001, 123457] {
                let mut ws2 = plan.driver_cfg.window_shift_sec;
                let (_, _, _, rows) = get_blstm_param(
                    plan.driver_cfg.window_size_sec,
                    &mut ws2,
                    rate,
                    s.shift_frames,
                    plan.ssr,
                    &plan.spec.lstm_subsampling,
                    &plan.spec.output_subsampling,
                    n,
                );
                assert_eq!(tl.rows_for(n), rows, "{rel}@{rate}: rows_for({n})");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The dropped stateful re-quantization is a fixed point at one rate.
// ---------------------------------------------------------------------------

/// The exact driver re-quantizes `_SpectrumShift` (`ssif = round(shift*rate)`, `shift =
/// ssif/rate`) and `_WindowShift` (`ws = round(shift*rate/ssif)`, floored 1, `shift =
/// ws*ssif/rate`) IN PLACE on every call, from the previous call's value. At one rate the
/// second quantization returns the first's integers and the first's seconds, bit for
/// bit: deriving from the config value every time (the plan) is the carried value.
#[test]
fn shift_requantization_is_a_fixed_point_at_one_rate() {
    let rates = [8000.0, 11025.0, 16000.0, 22050.0, 44100.0, 48000.0];
    let spectrum_shifts = [0.005, 0.01, 0.0125, 0.02, 0.025, 0.032];
    let window_shifts = [0.0, 0.05, 0.1, 0.37, 0.8, 3.25];
    let mut cases = 0;
    for rate in rates {
        for shift in spectrum_shifts {
            let ssif1 = f64::round(shift * rate) as usize;
            let sec1 = ssif1 as f64 / rate;
            let ssif2 = f64::round(sec1 * rate) as usize;
            let sec2 = ssif2 as f64 / rate;
            assert_eq!(ssif1, ssif2, "ssif at rate {rate} shift {shift}");
            assert_eq!(
                sec1.to_bits(),
                sec2.to_bits(),
                "sec at rate {rate} shift {shift}"
            );
            for wshift in window_shifts {
                let q = |w: f64| -> (i64, f64) {
                    let mut ws = f64::round(w * rate / ssif1 as f64) as i64;
                    if ws < 1 {
                        ws = 1;
                    }
                    (ws, (ws * ssif1 as i64) as f64 / rate)
                };
                let (ws1, wsec1) = q(wshift);
                let (ws2, wsec2) = q(wsec1);
                assert_eq!(ws1, ws2, "window shift at rate {rate} shift {wshift}");
                assert_eq!(
                    wsec1.to_bits(),
                    wsec2.to_bits(),
                    "window shift sec at rate {rate} shift {wshift}"
                );
                cases += 1;
            }
        }
    }
    assert_eq!(
        cases,
        rates.len() * spectrum_shifts.len() * window_shifts.len()
    );
}

// ---------------------------------------------------------------------------
// The per-arm normalization table, one test per arm.
// ---------------------------------------------------------------------------

#[test]
fn the_normalization_table() {
    assert_eq!(accepted_norm_types(FastSadArm::Offline), &[-1, 0, 1]);
    assert_eq!(accepted_norm_types(FastSadArm::StreamWindowed), &[0, 1]);
    assert_eq!(accepted_norm_types(FastSadArm::StreamCausal), &[0, 1]);
}

fn plan_with_norm(rel: &str, t: i16) -> anyhow::Result<FastSadPlan> {
    let mut m = map_at(rel);
    m.insert("BLSTM_InputNormalizationType".into(), t.to_string());
    FastSadPlan::from_map(&m)
}

#[test]
fn offline_arm_accepts_minus_one_zero_and_one_and_refuses_minus_two() {
    for t in [-1, 0, 1] {
        let plan = plan_with_norm("phase4a/tier2_spectral.config", t).unwrap();
        plan.check_norm(FastSadArm::Offline)
            .unwrap_or_else(|e| panic!("offline type {t}: {e}"));
        FastSpectralSegmenter::from_plan(plan, None)
            .unwrap_or_else(|e| panic!("offline driver type {t}: {e}"));
    }
    // -2 never reaches an arm: the plan refuses it fast-tree-wide.
    let msg = err_of(
        plan_with_norm("phase4a/tier2_spectral.config", -2),
        "type -2",
    );
    assert!(
        msg.contains("InputNormalizationType") && msg.contains("got -2"),
        "{msg}"
    );
}

#[test]
fn stream_windowed_arm_accepts_zero_and_one_and_refuses_minus_one() {
    for t in [0, 1] {
        plan_with_norm("phase4a/tier2_spectral.config", t)
            .unwrap()
            .check_norm(FastSadArm::StreamWindowed)
            .unwrap_or_else(|e| panic!("windowed type {t}: {e}"));
    }
    let msg = err_of(
        plan_with_norm("phase4a/tier2_spectral.config", -1)
            .unwrap()
            .check_norm(FastSadArm::StreamWindowed),
        "windowed type -1",
    );
    assert!(
        msg.contains("InputNormalizationType 1") && msg.contains("or 0") && msg.contains("got -1"),
        "{msg}"
    );
}

#[test]
fn stream_causal_arm_accepts_zero_and_one_and_refuses_minus_one() {
    for t in [0, 1] {
        plan_with_norm("phase9/slstm_forward.config", t)
            .unwrap()
            .check_norm(FastSadArm::StreamCausal)
            .unwrap_or_else(|e| panic!("causal type {t}: {e}"));
    }
    let msg = err_of(
        plan_with_norm("phase9/slstm_forward.config", -1)
            .unwrap()
            .check_norm(FastSadArm::StreamCausal),
        "causal type -1",
    );
    assert!(
        msg.contains("InputNormalizationType 1") && msg.contains("or 0") && msg.contains("got -1"),
        "{msg}"
    );
}

// ---------------------------------------------------------------------------
// The rate-free regime pre-check.
// ---------------------------------------------------------------------------

#[test]
fn rate_free_precheck_refuses_what_no_rate_can_rescue() {
    // BLSTM + window 0: the plain regime at every rate.
    let mut m = map_at("phase4a/tier2_spectral.config");
    m.insert("BLSTM_window".into(), "0".into());
    let msg = err_of(FastSadPlan::from_map(&m), "blstm window 0");
    assert!(msg.contains("resolves window_size 0"), "{msg}");

    // BLSTM + shift 0: the truncate regime at every rate.
    let mut m = map_at("phase4a/tier2_spectral.config");
    m.insert("BLSTM_shift".into(), "0".into());
    let msg = err_of(FastSadPlan::from_map(&m), "blstm shift 0");
    assert!(
        msg.contains("truncate") && msg.contains("window_shift resolves < 1"),
        "{msg}"
    );

    // Causal + any window: refused on the config's intent.
    let mut m = map_at("phase9/slstm_forward.config");
    m.insert("BLSTM_window".into(), "0.5".into());
    let msg = err_of(FastSadPlan::from_map(&m), "causal window 0.5");
    assert!(
        msg.contains("causal streaming requires the plain regime"),
        "{msg}"
    );

    // BiCell: plain (window 0) builds, truncate (window > 0, shift 0) is refused.
    let m = map_at("phase9/mamba_bidirectional.config");
    assert_eq!(m["BLSTM_window"].trim(), "0.0");
    FastSadPlan::from_map(&m).expect("a bidirectional cell runs the plain regime");
    let mut m = m;
    m.insert("BLSTM_window".into(), "0.5".into());
    m.insert("BLSTM_shift".into(), "0".into());
    let msg = err_of(FastSadPlan::from_map(&m), "bicell truncate");
    assert!(msg.contains("truncate"), "{msg}");
    // ...and overlap (window > 0, shift > 0) builds, resolving at a rate too.
    m.insert("BLSTM_shift".into(), "0.1".into());
    let plan = FastSadPlan::from_map(&m).expect("a bidirectional cell runs overlap");
    let tl = plan.timeline(8000.0).unwrap();
    assert!(tl.window_size > 0 && !tl.no_overlap);
}

#[test]
fn timeline_catches_a_sub_frame_shift_the_precheck_cannot() {
    // 1e-5 s at 8 kHz with ssif 80 is window_shift = round(0.001) = 0 -> truncate; the
    // seconds are nonzero so the pre-check passes and the timeline refuses.
    let mut m = map_at("phase4a/tier2_spectral.config");
    m.insert("BLSTM_shift".into(), "1e-5".into());
    let plan = FastSadPlan::from_map(&m).expect("rate-free, a tiny shift is not zero");
    let msg = err_of(plan.timeline(8000.0), "sub-frame shift");
    assert!(msg.contains("window_shift resolves < 1"), "{msg}");
    // The offline driver surfaces the same refusal at its first get_segmentation.
    let mut drv = FastSpectralSegmenter::from_plan(plan, None).unwrap();
    let mut audio =
        speech::audio::read_audio(&ref_dir().join("phase4a/corpus/f1.wav"), 0.0, 2.0, 0, None)
            .unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<speech::tasks::segmentation::Segmentation> = (0..audio.data.nrows())
        .map(|_| speech::tasks::segmentation::Segmentation::new(dur))
        .collect();
    let got = err_of(
        speech::tasks::segmenter::Segmenter::get_segmentation(
            &mut drv, &mut audio, &mut segs, None,
        ),
        "driver sub-frame shift",
    );
    assert_eq!(got, msg);
}

#[test]
fn the_pitch_gate_is_the_plans() {
    let mut m = map_at("phase4a/tier2_spectral.config");
    m.insert("BLSTM_TDCwindow".into(), "0.032".into());
    let msg = err_of(FastSadPlan::from_map(&m), "pitch");
    assert!(msg.contains("pitch second pass"), "{msg}");
}

// ---------------------------------------------------------------------------
// The stream: the inherited pitch gate, and the in-memory pack entry.
// ---------------------------------------------------------------------------

#[test]
fn the_stream_refuses_a_pitch_config_in_the_offline_drivers_words() {
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let text = std::fs::read_to_string(&stage.config_path).unwrap();
    let mut m = parse(&text);
    StreamingSession::new(&m, 8000.0, 1).expect("the staged config streams");
    m.insert("BLSTM_TDCwindow".into(), "0.032".into());
    let stream = err_of(StreamingSession::new(&m, 8000.0, 1), "stream pitch");
    let offline = err_of(
        FastSpectralSegmenter::from_legacy(&m, None),
        "offline pitch",
    );
    assert!(stream.contains("pitch second pass"), "{stream}");
    assert_eq!(stream, offline, "one gate, both consumers");
}

#[test]
fn from_plan_with_an_in_memory_pack_streams_as_new_does() {
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let text = std::fs::read_to_string(&stage.config_path).unwrap();
    let map = parse(&text);
    let plan = FastSadPlan::from_map(&map).unwrap();
    let pack = read_weight_vector(std::path::Path::new(&plan.weights_file)).unwrap();
    assert_eq!(pack.len(), plan.pack_len, "the staged pack is exact-length");

    let (rate, channels, samples) = common::read_wav_pcm16(&stage.wav_path);
    assert_eq!(channels, 1);
    let rate = rate as f64;
    let s: Vec<f32> = samples
        .iter()
        .take((12.0 * rate) as usize)
        .map(|&x| x as f32 / 32768.0)
        .collect();

    let run = |mut sess: StreamingSession| {
        let chunk = (0.1 * rate) as usize;
        let mut i = 0;
        while i < s.len() {
            let end = (i + chunk).min(s.len());
            sess.push(&s[i..end]);
            i = end;
        }
        let (_, seg) = sess.finish();
        let bits: Vec<u32> = sess
            .posterior_history()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let segs: Vec<(u64, i32)> = seg
            .segments()
            .iter()
            .map(|g| (g.begin.to_bits(), g.ty as i32))
            .collect();
        (bits, segs)
    };
    let from_file = run(StreamingSession::new(&map, rate, 1).unwrap());
    let from_pack = run(StreamingSession::from_plan(&plan, Some(&pack), rate, 1).unwrap());
    println!(
        "MEASURE from_plan_in_memory: posteriors={} segments={}",
        from_file.0.len(),
        from_file.1.len()
    );
    assert!(
        !from_file.0.is_empty(),
        "non-vacuous: posteriors were produced"
    );
    assert_eq!(from_file, from_pack, "file pack vs in-memory pack");

    // A wrong-length in-memory pack is refused in the exact `set_weights`' words -- the
    // same words the offline driver's `set_weights` uses.
    let short = &pack[..pack.len() - 1];
    let stream = err_of(
        StreamingSession::from_plan(&plan, Some(short), rate, 1),
        "short stream pack",
    );
    let offline = err_of(
        FastSpectralSegmenter::from_legacy(&map, None)
            .unwrap()
            .set_weights(short),
        "short offline pack",
    );
    assert_eq!(stream, offline);
}
