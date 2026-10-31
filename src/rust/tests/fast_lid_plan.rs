//! Issue #84: `fast::plan::FastLidPlan` / `LidTimeline`, the ONE algo-6 Mode-7 setup the
//! offline fast Twin and the Utterance stream consume (the Twin counterpart of
//! `fast_sad_plan.rs`, issue #24).
//!
//! What is pinned here, and why each leg exists:
//!  - the plan agrees with the EXACT tree on every committed Twin config (the LID shape,
//!    both pack lengths, both sub-sampling ratios, the class count): the plan is a reader
//!    of the same config, not a second opinion;
//!  - the timeline reproduces the frozen per-file derivation of the exact Twin
//!    (`get_segmentation_mode7`: `ssif` forced to 80, `get_blstm_param` on the never-run
//!    SAD net, the spectral time axis, `getLIDBLSTMParam`, the target clamp) at two rates
//!    -- the plan-level pin of the forced 80 (every committed Twin config carries
//!    `BLSTM_spectrum_shift 0.025`, so a derived shift would read 200 at 8 kHz; the parity
//!    legs see that too, as the battery measured, through a LID window of 10 instead of 25);
//!  - the two stateful shift carries the fast Twin DROPPED (the SAD `_WindowShift` and
//!    the `_LIDWindowShift`, each re-quantized in place and reset to 0.0 after a truncate
//!    file) are a fixed point at one rate, reset included (the by-design divergence's
//!    justification, measured over a rate x shift x sub-sampling grid);
//!  - every construction refusal, in the words the Twin legs pin, and the rate-free
//!    regime rule (each refusal, on the config's intent, and the two shapes that run; the
//!    timeline has nothing left to refuse and is infallible);
//!  - `FastTwinLid::from_plan` with an in-memory pair scores bit-identically to
//!    `from_legacy`.

use std::path::PathBuf;

use indexmap::IndexMap;

use ndarray::Array2;
use speech::audio::{Audio, read_audio};
use speech::fast::driver::{FastNetShape, FastTwinLid};
use speech::fast::plan::{FastLidPlan, MODE7_SSIF, derive_lid_window};
use speech::io::binary::read_weight_vector;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork, CellType};
use speech::tasks::sad::get_blstm_param;
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmenter::Segmenter;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

fn map_for(rel: &str) -> IndexMap<String, String> {
    let path = ref_dir().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    if rel.ends_with(".toml") {
        speech::toml_config::toml_to_map(&text).unwrap()
    } else {
        speech::legacy_config::parse_legacy_config(&text)
    }
}

fn with(mut map: IndexMap<String, String>, key: &str, value: &str) -> IndexMap<String, String> {
    map.insert(key.into(), value.into());
    map
}

fn err_of<T>(r: anyhow::Result<T>, what: &str) -> String {
    match r {
        Ok(_) => panic!("{what}: must be refused"),
        Err(e) => e.to_string(),
    }
}

/// Every committed algo-6 Mode-7 config the fast Twin runs, with the LID shape its name
/// promises (`phase4b/twin_mode7_ppm2.config` sets `BLSTM_LID_DumpInternals true`, a
/// construction refusal on the fast tree, so it is in the refusal leg instead).
const CONFIGS: [(&str, FastNetShape); 15] = [
    ("phase4b/twin_mode7.config", FastNetShape::Blstm),
    ("phase4b/twin_mode7_ppm1.config", FastNetShape::Blstm),
    ("phase4b/twin_e2e.config", FastNetShape::Blstm),
    (
        "phase9/twin_mode7_lid_lstm_forward.config",
        FastNetShape::Causal(CellType::Lstm),
    ),
    (
        "phase9/twin_mode7_lid_slstm_forward.config",
        FastNetShape::Causal(CellType::Slstm),
    ),
    (
        "phase9/twin_mode7_lid_mamba_forward.config",
        FastNetShape::Causal(CellType::Mamba),
    ),
    (
        "phase9/twin_mode7_lid_cfc_forward.config",
        FastNetShape::Causal(CellType::Cfc),
    ),
    (
        "phase9/twin_mode7_lid_transformer_forward.config",
        FastNetShape::Causal(CellType::Transformer),
    ),
    (
        "phase9/twin_mode7_lid_slstm.config",
        FastNetShape::BiCell(CellType::Slstm),
    ),
    (
        "phase9/twin_mode7_lid_mamba.config",
        FastNetShape::BiCell(CellType::Mamba),
    ),
    (
        "phase9/twin_mode7_lid_cfc.config",
        FastNetShape::BiCell(CellType::Cfc),
    ),
    (
        "phase9/twin_mode7_lid_transformer.config",
        FastNetShape::BiCell(CellType::Transformer),
    ),
    // The committed TOML LID configs, through the importer: the plan reads the same map
    // the bag reads.
    ("../../configs/lid/lid_blstm.toml", FastNetShape::Blstm),
    (
        "../../configs/training/lre03_lid_phseq.toml",
        FastNetShape::Blstm,
    ),
    (
        "../../configs/training/lre03_lid_features.toml",
        FastNetShape::Blstm,
    ),
];

// ---------------------------------------------------------------------------
// The plan agrees with the exact tree on every committed config.
// ---------------------------------------------------------------------------

#[test]
fn plan_agrees_with_the_exact_twin_on_every_committed_config() {
    for (rel, shape) in CONFIGS {
        let map = map_for(rel);
        let plan = FastLidPlan::from_map(&map).unwrap_or_else(|e| panic!("{rel}: {e}"));
        assert_eq!(plan.lid.shape, shape, "{rel}: LID shape");
        for (prefix, net) in [("BLSTM", &plan.sad), ("BLSTM_LID", &plan.lid)] {
            let bc = BlstmConfig::from_legacy(&map, prefix).unwrap();
            let exact = BlstmNetwork::from_config(bc.clone()).unwrap();
            assert_eq!(
                net.pack_len,
                exact.nb_of_weights(),
                "{rel}: {prefix} pack length"
            );
            assert_eq!(net.ssr, exact.sub_sampling_ratio(), "{rel}: {prefix} ssr");
            assert_eq!(
                net.input_normalization_type, bc.input_normalization_type,
                "{rel}: {prefix} normalization type"
            );
            assert_eq!(
                net.weights_file,
                map.get(&format!("{prefix}_weightsFile"))
                    .cloned()
                    .unwrap_or_default(),
                "{rel}: {prefix} weights file"
            );
            if prefix == "BLSTM_LID" {
                assert_eq!(plan.class_nb, exact.output_size().max(2), "{rel}: class_nb");
                assert_eq!(plan.two_sweeps, bc.two_sweeps, "{rel}: two_sweeps");
                assert_eq!(
                    plan.cost_modified,
                    bc.cost_law.is_cost_modified(),
                    "{rel}: cost_modified"
                );
            }
        }
        assert_eq!(
            plan.lid.input_normalization_type, 0,
            "{rel}: the gate value"
        );
    }
}

// ---------------------------------------------------------------------------
// The timeline reproduces the frozen per-file derivation, at two rates.
// ---------------------------------------------------------------------------

/// The exact Twin's `getLIDBLSTMParam` block (`tasks::lid::TwinBlstmSpectralLid::
/// get_segmentation_mode7`), transcribed: `(lid_window_size, lid_window_shift, lid_no_overlap)`.
fn exact_lid_window(
    size_sec: f64,
    shift_sec: f64,
    lid_ssr: usize,
    ssif: usize,
    rate: f64,
) -> (usize, usize, bool) {
    let ssifd = ssif as f64;
    let mut lid_window_size = f64::round(size_sec * rate / 2.0 / ssifd) as usize;
    if lid_window_size != 0 && lid_window_size < lid_ssr {
        lid_window_size = lid_ssr;
    }
    let mut lid_window_shift = f64::round(shift_sec * rate / ssifd) as i64;
    let mut lid_no_overlap = false;
    if lid_window_size != 0 && lid_window_shift < 1 {
        lid_no_overlap = true;
        let raw = f64::round(size_sec * rate / ssifd) as usize;
        lid_window_size = (raw / lid_ssr) * lid_ssr;
        if lid_window_size < 10 * lid_ssr {
            lid_window_size = 10 * lid_ssr;
        }
    }
    if lid_window_size == 0 || lid_window_shift < 1 {
        lid_window_shift = 1;
    }
    (lid_window_size, lid_window_shift as usize, lid_no_overlap)
}

/// The exact Twin's SAD time axis (`get_segmentation_mode7`, the block it cites as legacy
/// `:696-706`), transcribed: `window_shift_sec` is the POST-`get_blstm_param` value.
fn exact_sad_axis(
    window_shift_sec: f64,
    spectrum_shift_sec: f64,
    sad_ssr: usize,
    window_size: usize,
    no_overlap: bool,
) -> (f64, f64) {
    let mut time_step = window_shift_sec * sad_ssr as f64;
    let mut time_offset = time_step / 2.0 - window_shift_sec / 2.0;
    if window_size > 0 {
        if no_overlap {
            time_step = window_shift_sec * sad_ssr as f64;
            time_offset = time_step / 2.0 - window_shift_sec / 2.0;
        } else {
            time_step = spectrum_shift_sec * sad_ssr as f64;
            time_offset = time_step / 2.0 - spectrum_shift_sec / 2.0;
        }
    }
    (time_step, time_offset)
}

#[test]
fn timeline_reproduces_the_frozen_per_file_derivation() {
    for rel in [
        "phase4b/twin_mode7.config",
        "phase9/twin_mode7_lid_slstm_forward.config",
        "phase9/twin_mode7_lid_mamba.config",
    ] {
        let map = map_for(rel);
        let plan = FastLidPlan::from_map(&map).unwrap();
        for rate in [8000.0, 16000.0] {
            let tl = plan.timeline(rate, 1);
            // Mode 7 forces `_SpectrumShiftInFrames = 80` at EVERY rate (the phSeq/cep
            // periodogram is always present): the committed `BLSTM_spectrum_shift 0.025`
            // would derive 200 at 8 kHz and 400 at 16 kHz.
            assert_eq!(tl.ssif, MODE7_SSIF, "{rel}@{rate}: ssif");
            assert_eq!(tl.ssif, 80);
            assert_eq!(
                tl.spectrum_shift_sec.to_bits(),
                (80.0 / rate).to_bits(),
                "{rel}@{rate}: spectrum shift"
            );
            assert_eq!(tl.rate.to_bits(), rate.to_bits());

            // The SAD half: `get_blstm_param` called directly off the config values on the
            // never-run SAD net, then the spectral time axis.
            let mut ws = plan.driver_cfg.window_shift_sec;
            let (w, sh, no, rows0) = get_blstm_param(
                plan.driver_cfg.window_size_sec,
                &mut ws,
                rate,
                80,
                plan.sad.ssr,
                &plan.sad.spec.lstm_subsampling,
                &plan.sad.spec.output_subsampling,
                0,
            );
            assert_eq!(tl.sad.window_size, w, "{rel}@{rate}: SAD window");
            assert_eq!(tl.sad.window_shift, sh, "{rel}@{rate}: SAD shift");
            assert_eq!(tl.sad.no_overlap, no, "{rel}@{rate}: SAD no_overlap");
            assert_eq!(tl.sad.rows, rows0);
            assert_eq!(
                tl.sad.window_shift_sec.to_bits(),
                ws.to_bits(),
                "{rel}@{rate}: SAD shift sec"
            );
            let (ts, to) = exact_sad_axis(ws, 80.0 / rate, plan.sad.ssr, w, no);
            assert_eq!(
                tl.sad.time_step.to_bits(),
                ts.to_bits(),
                "{rel}@{rate}: time_step"
            );
            assert_eq!(
                tl.sad.time_offset.to_bits(),
                to.to_bits(),
                "{rel}@{rate}: time_offset"
            );
            // The committed Twin configs are `BLSTM_window 5.0` + `BLSTM_shift 0`: TRUNCATE on
            // the SAD side, the regime the SAD plan refuses and this plan does not gate.
            assert!(
                tl.sad.no_overlap && tl.sad.window_size > 0,
                "{rel}@{rate}: SAD truncate"
            );
            // `rows_for` is the frozen `real_vec_size`.
            for frames in [0usize, 1, 79, 80, 81, 12345] {
                let mut ws = plan.driver_cfg.window_shift_sec;
                let rows = get_blstm_param(
                    plan.driver_cfg.window_size_sec,
                    &mut ws,
                    rate,
                    80,
                    plan.sad.ssr,
                    &plan.sad.spec.lstm_subsampling,
                    &plan.sad.spec.output_subsampling,
                    frames,
                )
                .3;
                assert_eq!(
                    plan.rows_for(rate, frames),
                    rows,
                    "{rel}@{rate}: rows for {frames}"
                );
            }

            // The LID half: `getLIDBLSTMParam`, transcribed from the exact driver.
            let (lw, lsh, lno) = exact_lid_window(
                plan.lid_window_size_sec,
                plan.lid_window_shift_sec,
                plan.lid.ssr,
                80,
                rate,
            );
            assert_eq!(tl.lid_window_size, lw, "{rel}@{rate}: LID window");
            assert_eq!(tl.lid_window_shift, lsh, "{rel}@{rate}: LID shift");
            assert_eq!(tl.lid_no_overlap, lno, "{rel}@{rate}: LID no_overlap");
            assert_eq!(
                derive_lid_window(
                    plan.lid_window_size_sec,
                    plan.lid_window_shift_sec,
                    plan.lid.ssr,
                    80,
                    rate
                ),
                (lw, lsh, lno)
            );
        }

        // The target clamp (`TwinBLSTMSpectralLID.cpp:626-628`): `[0, class_nb)` else 0.
        let nb = plan.class_nb as i32;
        for (lang, ti) in [
            (-1, 0usize),
            (0, 0),
            (1, 1),
            (nb - 1, (nb - 1) as usize),
            (nb, 0),
            (99, 0),
        ] {
            let tl = plan.timeline(8000.0, lang);
            assert_eq!(tl.ti, ti, "{rel}: lang {lang}");
        }
    }
}

// ---------------------------------------------------------------------------
// The two dropped carries are a fixed point at one rate, reset included.
// ---------------------------------------------------------------------------

/// What the exact Twin carries from one file to the next: the shift re-quantized in
/// place by the derivation, then RESET to 0.0 after a truncate file (the epilogue's
/// `if noOverlap { shift = 0.0 }`, `TwinBLSTMSpectralLID.cpp:1419-1420`). Without the
/// reset the floored shift (>= 1 frame) would read as OVERLAP on the next file.
fn carried(quantized_sec: f64, no_overlap: bool) -> f64 {
    if no_overlap { 0.0 } else { quantized_sec }
}

#[test]
fn shift_requantization_is_a_fixed_point_at_one_rate_reset_included() {
    let rates = [8000.0, 11025.0, 16000.0, 22050.0, 44100.0, 48000.0];
    let sizes = [0.0, 0.1, 0.25, 0.5, 2.0, 5.0];
    let shifts = [0.0, 0.001, 0.005, 0.0125, 0.1, 0.8];
    let ssrs = [1usize, 2, 4];
    let ssif = MODE7_SSIF;
    let mut cases = 0;
    for rate in rates {
        for size in sizes {
            for shift in shifts {
                for ssr in ssrs {
                    // The LID carry: `_LIDWindowShift` written back as `shift * ssif / rate`.
                    let (w1, s1, no1) = derive_lid_window(size, shift, ssr, ssif, rate);
                    let sec1 = (s1 * ssif) as f64 / rate;
                    let (w2, s2, no2) =
                        derive_lid_window(size, carried(sec1, no1), ssr, ssif, rate);
                    assert_eq!(
                        (w1, s1, no1),
                        (w2, s2, no2),
                        "LID at rate {rate} size {size} shift {shift} ssr {ssr}"
                    );
                    let sec2 = (s2 * ssif) as f64 / rate;
                    assert_eq!(
                        sec1.to_bits(),
                        sec2.to_bits(),
                        "LID sec at rate {rate} size {size} shift {shift} ssr {ssr}"
                    );

                    // The SAD carry: `_WindowShift` through `get_blstm_param` (no sub-sampling
                    // vectors needed for the window/shift pair: `frame_count` 0).
                    let mut ws1 = shift;
                    let (w1, s1, no1, _) =
                        get_blstm_param(size, &mut ws1, rate, ssif, ssr, &[], &[], 0);
                    let mut ws2 = carried(ws1, no1);
                    let (w2, s2, no2, _) =
                        get_blstm_param(size, &mut ws2, rate, ssif, ssr, &[], &[], 0);
                    assert_eq!(
                        (w1, s1, no1),
                        (w2, s2, no2),
                        "SAD at rate {rate} size {size} shift {shift} ssr {ssr}"
                    );
                    assert_eq!(
                        ws1.to_bits(),
                        ws2.to_bits(),
                        "SAD sec at rate {rate} size {size} shift {shift} ssr {ssr}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, rates.len() * sizes.len() * shifts.len() * ssrs.len());
}

// ---------------------------------------------------------------------------
// Every construction refusal, in the Twin legs' words; the two regime refusals.
// ---------------------------------------------------------------------------

#[test]
fn plan_refuses_a_non_algo6_config_and_a_missing_algo() {
    let msg = err_of(
        FastLidPlan::from_map(&map_for("phase4a/tier2_spectral.config")),
        "algo 3 on the LID plan",
    );
    assert!(
        msg.contains("only algo 6 (Twin LID) is supported (got 3)"),
        "{msg}"
    );
    let mut m = map_for("phase4b/twin_mode7.config");
    m.shift_remove("Algo_choice");
    let msg = err_of(FastLidPlan::from_map(&m), "no algo");
    assert!(
        msg.contains("param 'Algo_choice' not found in config"),
        "{msg}"
    );
}

#[test]
fn plan_refuses_what_the_twin_refused_at_construction() {
    let base = map_for("phase4b/twin_mode7.config");
    let msg = err_of(
        FastLidPlan::from_map(&map_for("phase9/twin_lid_slstm.config")),
        "mode 5",
    );
    assert!(
        msg.contains("only Mode 7") && msg.contains("got 5"),
        "{msg}"
    );
    let msg = err_of(
        FastLidPlan::from_map(&map_for("phase4b/twin_mode7_ppm2.config")),
        "ppm2",
    );
    assert!(msg.contains("DumpInternals"), "{msg}");
    for (key, value, needle) in [
        ("BLSTM_LID_Mode", "0", "only Mode 7"),
        ("BLSTM_TDCwindow", "0.1", "pitch second pass"),
        ("BLSTM_LID_DumpInternals", "true", "DumpInternals"),
        (
            "BLSTM_LID_InputNormalizationType",
            "1",
            "only LID InputNormalizationType 0",
        ),
        (
            "BLSTM_LID_TargetEnforcementStep",
            "-1",
            "negative BLSTM_LID_TargetEnforcementStep",
        ),
        ("BLSTM_LID_LSTMNeuronNb", "0,48", "MLP mode"),
        (
            "BLSTM_LID_NoiseMagnitude",
            "-inf",
            "BLSTM_LID_NoiseMagnitude",
        ),
    ] {
        let msg = err_of(FastLidPlan::from_map(&with(base.clone(), key, value)), key);
        assert!(msg.contains(needle), "{key} {value}: {msg}");
    }
}

#[test]
fn rate_free_precheck_refuses_what_no_rate_can_rescue() {
    // A windowed CAUSAL LID net: the committed bidirectional window on a forward fixture,
    // refused on the config's intent before any rate is known.
    let m = with(
        map_for("phase9/twin_mode7_lid_slstm_forward.config"),
        "BLSTM_LID_window",
        "0.25",
    );
    let msg = err_of(FastLidPlan::from_map(&m), "windowed causal");
    assert!(
        msg.starts_with(
            "fast LID plan: windowed LID inference is unsupported for a causal LID net"
        ) && msg.contains("at every rate"),
        "{msg}"
    );
    // OVERLAP on a bidirectional shape: `shift 0.1` (10 frames at 8 kHz), and the one
    // tightening, a sub-frame positive shift (`0.001`: 0.1 frame at 8 kHz, which the exact
    // tree would quantize to truncate) -- both refused as overlap on intent, the message
    // naming the intent rule, not a resolution.
    for shift in ["0.1", "0.001"] {
        let m = with(
            map_for("phase4b/twin_mode7.config"),
            "BLSTM_LID_shift",
            shift,
        );
        let msg = err_of(FastLidPlan::from_map(&m), "overlap");
        assert!(
            msg.starts_with(
                "fast LID plan: the overlap LID windowing is unsupported on the fast path"
            ) && msg.contains(&format!(
                "BLSTM_LID_shift {shift} > 0 under BLSTM_LID_window 0.25"
            )),
            "shift {shift}: {msg}"
        );
    }
    // A causal shape with `window 0` and a shift (plain: the shift is inert) RUNS, as does
    // the committed truncate pair (`window 0.25 / shift 0.0`), at both rates.
    let m = with(
        map_for("phase9/twin_mode7_lid_slstm_forward.config"),
        "BLSTM_LID_shift",
        "0.1",
    );
    let plan = FastLidPlan::from_map(&m).unwrap();
    for rate in [8000.0, 16000.0] {
        let tl = plan.timeline(rate, 0);
        assert_eq!(tl.lid_window_size, 0);
    }
    let plan = FastLidPlan::from_map(&map_for("phase4b/twin_mode7.config")).unwrap();
    for rate in [8000.0, 16000.0] {
        let tl = plan.timeline(rate, 0);
        assert!(tl.lid_no_overlap && tl.lid_window_size > 0, "{rate}");
    }
}

// ---------------------------------------------------------------------------
// `from_plan` with an in-memory pair scores as `from_legacy` does.
// ---------------------------------------------------------------------------

fn phseq_audio(f: &str, lang: i32) -> Audio {
    let path = ref_dir()
        .join("phase4b/corpus_phseq")
        .join(format!("{f}.phSeq"));
    let mut a = read_audio(&path, 0.0, 120.0, 1, None).unwrap();
    a.lang_index = lang;
    a.weight = 1.0;
    a.audio_file_name = path.to_string_lossy().into_owned();
    a
}

/// Channel 0's LID members after scoring `s1`: `(classification_errors, confusion, is_correct)`.
fn score_s1(d: &mut FastTwinLid) -> (Vec<f64>, Array2<f64>, i32) {
    let mut audio = phseq_audio("s1", 0);
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    d.get_segmentation(&mut audio, &mut segs, None).unwrap();
    let m = d.lid_members();
    (
        m.classification_errors[0].clone(),
        m.segments_confusion[0].clone(),
        m.is_correct[0],
    )
}

#[test]
fn from_plan_with_an_in_memory_pair_scores_as_from_legacy_does() {
    for fixture in ["twin_mode7_lid_slstm", "twin_mode7_lid_cfc_forward"] {
        let map = map_for(&format!("phase9/{fixture}.config"));
        let lid =
            read_weight_vector(&ref_dir().join(format!("phase9/{fixture}_seed.bin"))).unwrap();
        let sad = read_weight_vector(&ref_dir().join("phase4b/tiny_sad_seed.bin")).unwrap();
        let plan = FastLidPlan::from_map(&map).unwrap();
        let mut from_plan = FastTwinLid::from_plan(plan.clone(), Some(&sad), Some(&lid)).unwrap();
        let mut from_legacy = FastTwinLid::from_legacy(&map, Some(&sad), Some(&lid)).unwrap();
        assert_eq!(
            score_s1(&mut from_plan),
            score_s1(&mut from_legacy),
            "{fixture}"
        );
        // A wrong-length pack on either seat is refused in the exact `set_weights`' words.
        let msg = err_of(
            FastTwinLid::from_plan(plan.clone(), Some(&sad[..sad.len() - 1]), Some(&lid)),
            "short SAD pack",
        );
        assert!(
            msg.contains(&format!("{}", plan.sad.pack_len)),
            "{fixture}: {msg}"
        );
        let msg = err_of(
            FastTwinLid::from_plan(plan.clone(), Some(&sad), Some(&lid[..lid.len() - 1])),
            "short LID pack",
        );
        assert!(
            msg.contains(&format!("{}", plan.lid.pack_len)),
            "{fixture}: {msg}"
        );
        // Scoring twice is scoring once: nothing is carried between files.
        assert_eq!(
            score_s1(&mut from_plan),
            score_s1(&mut from_legacy),
            "{fixture}: second file"
        );
    }
}
