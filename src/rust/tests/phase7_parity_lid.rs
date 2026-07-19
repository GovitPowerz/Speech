//! Phase 7 Task 5: the CI parity leg (spec S1.6b) -- the f32 fast Mode-7 LID Twin
//! (`fast::driver::FastTwinLid`) vs the exact f64 `TwinBlstmSpectralLid`, over the
//! committed `twin_mode7.config` + the real 12409-weight LID net + the committed phSeq
//! corpus (`corpus_phseq/s{1,2,3}.phSeq`) and the phase-6 cep fixtures (File_Type 2).
//!
//! The fast path DIVERGES from exact BY DESIGN (f32 + faer through the LID BLSTM +
//! TwoSweeps-truncate windowing, spec S4); the LID score deltas (`lid_classification_
//! errors`, i.e. the in-band `100*(langID - targetLID)` `.scr` column source) are
//! MEASURED first, then pinned with headroom. The ARGMAX DECISIONS are the HARD gate
//! (spec S1.6b): `is_lid_correct`, the per-block `lid_segments_confusion` counts, and the
//! reconstructed predicted-language argmax must all be IDENTICAL fast-vs-exact per file.
//! A flip is R1 STOP-and-adjudicate -- it would mean f32 moved a decision on committed
//! fixtures, not a silent widen.
//!
//! LID net shape (`BLSTM_LID_*`): `LSTMNeuronNb 36,24 / OutputNeuronNb 48,1` -> a BINARY
//! (output_size 1) net, so `feed_forward_scoring` binary-expands to `[1-p, p]` and
//! `class_nb = max(1, 2) = 2`. Every phSeq block (<= 9 chars) and cep record is smaller
//! than `lid_window_size` (25 periodogram frames), so each block is a single TwoSweeps
//! truncate window -- but the padding still drives the bidirectional forward.

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::audio::{Audio, read_audio};
use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::{BagOfProcessors, Processor};
use speech::fast::driver::FastTwinLid;
use speech::io::binary::read_weight_vector;
use speech::tasks::lid::TwinBlstmSpectralLid;
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmenter::Segmenter;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}
fn phase4b(name: &str) -> PathBuf {
    ref_dir().join("phase4b").join(name)
}
fn phase6(name: &str) -> PathBuf {
    ref_dir().join("phase6").join(name)
}

fn image_mode() -> Mode {
    Mode {
        kind: ModeKind::Image,
        verbose: false,
    }
}

fn multi_mode() -> Mode {
    Mode {
        kind: ModeKind::Multi,
        verbose: false,
    }
}

fn lid_weights() -> Vec<f64> {
    let w = read_weight_vector(&phase4b("LID_bestNNWeight_1.bin")).unwrap();
    assert_eq!(w.len(), 12409, "real LID net weight count");
    w
}

fn map_of(variant: &str) -> IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(
        &std::fs::read_to_string(phase4b(&format!("{variant}.config"))).unwrap(),
    )
}

fn phseq_audio(f: &str, lang: i32) -> Audio {
    let path = phase4b("corpus_phseq").join(format!("{f}.phSeq"));
    let mut a = read_audio(&path, 0.0, 120.0, 1).unwrap();
    a.lang_index = lang;
    a.weight = 1.0;
    a.audio_file_name = path.to_string_lossy().into_owned();
    a
}

fn cep_audio(name: &str, lang: i32) -> Audio {
    let path = phase6(&format!("cep/{name}"));
    let mut a = read_audio(&path, 0.0, 3.6e6, 2).unwrap();
    a.lang_index = lang;
    a.weight = 1.0;
    a.audio_file_name = path.to_string_lossy().into_owned();
    a
}

/// Extract the per-channel LID members that the parity comparison + `.scr` writer read:
/// `(lid_classification_errors, is_lid_correct, lid_segments_confusion)`.
struct LidMembers {
    errors: Vec<Vec<f64>>,
    correct: Vec<i32>,
    confusion: Vec<ndarray::Array2<f64>>,
}

fn run_exact(variant: &str, mut audio: Audio) -> LidMembers {
    let lidw = lid_weights();
    let mut drv = TwinBlstmSpectralLid::from_legacy(&map_of(variant), None, Some(&lidw)).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    LidMembers {
        errors: drv.lid_classification_errors().to_vec(),
        correct: drv.is_lid_correct().to_vec(),
        confusion: drv.lid_segments_confusion().to_vec(),
    }
}

fn run_fast(variant: &str, mut audio: Audio) -> LidMembers {
    let lidw = lid_weights();
    let mut drv = FastTwinLid::from_legacy(&map_of(variant), None, Some(&lidw)).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    LidMembers {
        errors: drv.lid_classification_errors().to_vec(),
        correct: drv.is_lid_correct().to_vec(),
        confusion: drv.lid_segments_confusion().to_vec(),
    }
}

/// Reconstruct langID from the in-band `lid_classification_errors` row and return its
/// argmax (the predicted language / `.scr` decision). `err[c] = 100*(langID[c] -
/// targetLID[c])`, `targetLID[ti] = -2`, else 0 -> `langID[c] = err[c]/100 (+2 at ti)`.
fn predicted_language(err: &[f64], ti: usize) -> usize {
    let mut langid: Vec<f64> = err.iter().map(|&x| x / 100.0).collect();
    langid[ti] -= 2.0;
    let mut j = 0usize;
    let mut best = langid[0];
    for (c, &v) in langid.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            j = c;
        }
    }
    j
}

/// Compare exact vs fast LID members for one file; HARD-assert the argmax decisions,
/// accumulate the score delta. `lang` is the file's language (clamped to `[0, class_nb)`
/// for the target index).
fn compare(
    tag: &str,
    lang: i32,
    e: &LidMembers,
    f: &LidMembers,
    max_abs: &mut f64,
    max_rel: &mut f64,
) {
    assert_eq!(e.errors.len(), f.errors.len(), "{tag}: channel count");
    for chan in 0..e.errors.len() {
        let le = &e.errors[chan];
        let lf = &f.errors[chan];
        assert_eq!(le.len(), lf.len(), "{tag} chan {chan}: liderr length");
        let class_nb = le.len();
        let ti = if lang >= 0 && (lang as usize) < class_nb {
            lang as usize
        } else {
            0
        };

        // --- score delta (the .scr-column source) ---
        for (&a, &b) in le.iter().zip(lf.iter()) {
            let d = (a - b).abs();
            *max_abs = max_abs.max(d);
            // liderr magnitudes are ~[0,100] (+200 on the target col); scale by max(|a|,1).
            *max_rel = max_rel.max(d / a.abs().max(1.0));
        }

        // --- ARGMAX decisions: IDENTICAL (R1 hard) ---
        assert_eq!(
            e.correct[chan], f.correct[chan],
            "{tag} chan {chan}: is_lid_correct FLIP (R1 STOP: exact={} fast={})",
            e.correct[chan], f.correct[chan]
        );
        assert_eq!(
            e.confusion[chan], f.confusion[chan],
            "{tag} chan {chan}: confusion matrix FLIP (R1 STOP -- per-block argmax moved)"
        );
        let pe = predicted_language(le, ti);
        let pf = predicted_language(lf, ti);
        assert_eq!(
            pe, pf,
            "{tag} chan {chan}: predicted-language FLIP (R1 STOP: exact lang {pe}, fast lang {pf})"
        );
    }
}

// ---------------------------------------------------------------------------
// The CI parity legs.
// ---------------------------------------------------------------------------

/// (file, lang). s1 -> class 0 (net predicts 0 -> HIT); s2 -> class 1 (predicts 1 ->
/// HIT); s3 -> class 1 (predicts 0 -> MISS): a hit AND a miss + an off-diagonal case.
const PHSEQ_FILES: [(&str, i32); 3] = [("s1", 0), ("s2", 1), ("s3", 1)];

#[test]
fn lid_parity_phseq_exact_vs_fast() {
    // MEASURED (Apple Silicon dev box, debug): score max_abs=7.785e-7, max_rel=1.662e-8
    // over the `100*(langID - targetLID)` in-band columns (~[0,200] magnitude). NO argmax
    // flipped (is_lid_correct + confusion + predicted-language all IDENTICAL, R1 clean --
    // the compare() asserts are the HARD gate). The score pins carry ~600x headroom over
    // measured, widened for cross-platform faer SIMD variance (CI x86 vs this ARM box); a
    // structural break would move the score by orders of magnitude (or flip an argmax).
    const SCORE_ABS_PIN: f64 = 5.0e-4;
    const SCORE_REL_PIN: f64 = 5.0e-6;

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (f, lang) in PHSEQ_FILES {
        let e = run_exact("twin_mode7", phseq_audio(f, lang));
        let fa = run_fast("twin_mode7", phseq_audio(f, lang));
        compare(
            &format!("phseq {f}"),
            lang,
            &e,
            &fa,
            &mut max_abs,
            &mut max_rel,
        );
    }
    println!("MEASURE lid_parity_phseq: score max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    assert!(
        max_abs < SCORE_ABS_PIN,
        "phseq score abs {max_abs} exceeds pin"
    );
    assert!(
        max_rel < SCORE_REL_PIN,
        "phseq score rel {max_rel} exceeds pin"
    );
}

/// The well-formed committed cep fixtures (the malformed ones error at `read_cep`, in
/// BOTH paths -- a read-layer error, not a driver-parity case).
const CEP_FIXTURES: [&str; 2] = ["tiny_ok.plp", "multi_ok.plp"];

#[test]
fn lid_parity_cep_exact_vs_fast() {
    // MEASURED (Apple Silicon dev box, debug): score max_abs=2.680e-6, max_rel=2.733e-8.
    // The cep fixtures are tiny/degenerate (vectorSize 2-3, 1-2 rows), so the delta is a
    // touch larger than phSeq's -- still f32-scale. NO argmax flipped (R1 clean). ~180x
    // headroom over measured, widened for cross-platform faer variance.
    const SCORE_ABS_PIN: f64 = 5.0e-4;
    const SCORE_REL_PIN: f64 = 5.0e-6;

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for name in CEP_FIXTURES {
        let e = run_exact("twin_mode7", cep_audio(name, 0));
        let fa = run_fast("twin_mode7", cep_audio(name, 0));
        compare(
            &format!("cep {name}"),
            0,
            &e,
            &fa,
            &mut max_abs,
            &mut max_rel,
        );
    }
    // Confirm the malformed cep fixtures error identically at the read layer (not driver).
    for name in [
        "zero_records.plp",
        "trunc_header.plp",
        "trunc_table.plp",
        "truncated.plp",
        "excess.plp",
        "bad_vecsize.plp",
    ] {
        assert!(
            read_audio(&phase6(&format!("cep/{name}")), 0.0, 3.6e6, 2).is_err(),
            "malformed cep {name} must error at read (shared, not a driver-parity case)"
        );
    }
    println!("MEASURE lid_parity_cep: score max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    assert!(
        max_abs < SCORE_ABS_PIN,
        "cep score abs {max_abs} exceeds pin"
    );
    assert!(
        max_rel < SCORE_REL_PIN,
        "cep score rel {max_rel} exceeds pin"
    );
}

// ---------------------------------------------------------------------------
// Rider 2: the LID peephole-default alignment (wires the FastBlstm::debug_peepholes hook).
// ---------------------------------------------------------------------------

#[test]
fn fast_lid_peepholes_aligned() {
    // twin_mode7 sets all six BLSTM_LID peephole keys true; the fast LID net must read
    // them (via build_aligned_spec's BlstmConfig source), not the NnetSpec default FALSE.
    let lidw = lid_weights();
    let drv = FastTwinLid::from_legacy(&map_of("twin_mode7"), None, Some(&lidw)).unwrap();
    assert_eq!(
        drv.debug_lid_peepholes(),
        Some([true; 6]),
        "fast LID net must carry the config's TRUE peepholes"
    );

    // Omitting config: NnetSpec would default FALSE, but the aligned spec (BlstmConfig
    // default TRUE) must give TRUE -- proving the LID prefix is peephole-aligned too.
    const LID_KEYS: [&str; 6] = [
        "BLSTM_LID_Forward_IsCellsPeepholesActive",
        "BLSTM_LID_Backward_IsCellsPeepholesActive",
        "BLSTM_LID_Forward_IsGatesPeepholesActive",
        "BLSTM_LID_Backward_IsGatesPeepholesActive",
        "BLSTM_LID_Forward_IsGatesRecurrentPeepholesActive",
        "BLSTM_LID_Backward_IsGatesRecurrentPeepholesActive",
    ];
    let mut omit = map_of("twin_mode7");
    for k in LID_KEYS {
        omit.shift_remove(k);
    }
    let drv = FastTwinLid::from_legacy(&omit, None, Some(&lidw)).unwrap();
    assert_eq!(
        drv.debug_lid_peepholes(),
        Some([true; 6]),
        "aligned LID spec must default absent peepholes TRUE (exact BlstmConfig default)"
    );
}

// ---------------------------------------------------------------------------
// Dispatch + the training-shaped-config rider.
// ---------------------------------------------------------------------------

fn twin_map_fast() -> IndexMap<String, String> {
    let mut m = map_of("twin_mode7");
    m.insert("numOuterThreads".into(), "1".into());
    m.insert(
        "BLSTM_LID_weightsFile".into(),
        phase4b("LID_bestNNWeight_1.bin").to_str().unwrap().into(),
    );
    m.insert("Inference_Path".into(), "fast".into());
    m
}

#[test]
fn fast_dispatch_routes_algo6_to_fast_twin() {
    let mut m = twin_map_fast();
    let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode()).unwrap();
    assert!(
        matches!(bag.processor(0), Processor::FastTwinLid(_)),
        "fast + algo 6 must dispatch to FastTwinLid"
    );

    // Exact (absent Inference_Path) still routes to the exact Twin.
    let mut exact = map_of("twin_mode7");
    exact.insert("numOuterThreads".into(), "1".into());
    exact.insert(
        "BLSTM_LID_weightsFile".into(),
        phase4b("LID_bestNNWeight_1.bin").to_str().unwrap().into(),
    );
    let bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut exact), image_mode()).unwrap();
    assert!(
        matches!(bag.processor(0), Processor::TwinLid(_)),
        "absent Inference_Path must dispatch to the exact TwinLid"
    );
}

#[test]
fn fast_bails_on_training_shaped_config() {
    // Rider: Inference_Path fast + a MULTI-mode (training) run with a training-shaped
    // config (Epochs > 0 OR any net's BackPropagationActivated on) is a construction error
    // (both algo 3 and 6). Multi-gated: Image/Solo drive inference, unaffected.
    for (key, val) in [
        ("Neural_Networks_BackPropagation_Epochs", "5"),
        ("BLSTM_BackPropagationActivated", "true"),
        ("BLSTM_LID_BackPropagationActivated", "true"),
    ] {
        let mut m = twin_map_fast();
        m.insert(key.into(), val.into());
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut m), multi_mode()) {
            Err(e) => assert!(
                e.to_string().to_lowercase().contains("training"),
                "expected a training-shaped bail for {key}, got: {e}"
            ),
            Ok(_) => panic!("fast + Multi + {key}={val} must bail (training-shaped)"),
        }

        // The SAME training-shaped config in IMAGE mode must NOT bail (inference; the
        // "image mode unaffected" half of the rider).
        let mut mi = twin_map_fast();
        mi.insert(key.into(), val.into());
        assert!(
            BagOfProcessors::from_configs(std::slice::from_mut(&mut mi), image_mode()).is_ok(),
            "fast + Image + {key}={val} must NOT bail (inference, mode-exempt)"
        );
    }

    // Algo 3 (SAD) too: the rider hardens the Task-4 surface. tier2 + fast + backprop,
    // Multi mode -> bail.
    let text = std::fs::read_to_string(ref_dir().join("phase4a/tier2_spectral.config")).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&text);
    m.insert("numOuterThreads".into(), "1".into());
    m.insert(
        "BLSTM_weightsFile".into(),
        ref_dir()
            .join("phase0/NNweights_config1.bin")
            .to_str()
            .unwrap()
            .into(),
    );
    m.insert("Inference_Path".into(), "fast".into());
    m.insert("BLSTM_BackPropagationActivated".into(), "true".into());
    match BagOfProcessors::from_configs(std::slice::from_mut(&mut m), multi_mode()) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("training"),
            "expected a training-shaped bail for algo 3, got: {e}"
        ),
        Ok(_) => panic!("fast algo 3 + Multi + BackPropagationActivated must bail"),
    }
}

// ---------------------------------------------------------------------------
// Typed-bail pins (unexercised fast Mode-7 surfaces).
// ---------------------------------------------------------------------------

fn build_fast_twin(map: &IndexMap<String, String>) -> anyhow::Result<FastTwinLid> {
    let lidw = lid_weights();
    FastTwinLid::from_legacy(map, None, Some(&lidw))
}

#[test]
fn fast_twin_bails_on_non_mode7() {
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_LID_Mode".into(), "0".into());
    match build_fast_twin(&m) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("mode 7"),
            "expected a mode-7-only bail, got: {e}"
        ),
        Ok(_) => panic!("fast Twin + Mode != 7 must bail"),
    }
}

#[test]
fn fast_twin_bails_on_dump_internals() {
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_LID_DumpInternals".into(), "true".into());
    match build_fast_twin(&m) {
        Err(e) => assert!(
            e.to_string().contains("DumpInternals"),
            "expected a dump-internals bail, got: {e}"
        ),
        Ok(_) => panic!("fast Twin + DumpInternals must bail"),
    }
}

#[test]
fn fast_twin_bails_on_pitch_pass() {
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_TDCwindow".into(), "0.032".into());
    m.insert("BLSTM_TDC_lags".into(), "0.002,0.016".into());
    match build_fast_twin(&m) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("pitch"),
            "expected a pitch-pass bail, got: {e}"
        ),
        Ok(_) => panic!("fast Twin + TDCwindow > 0 must bail"),
    }
}

#[test]
fn fast_twin_bails_on_non_zero_lid_normalization() {
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_LID_InputNormalizationType".into(), "-1".into());
    match build_fast_twin(&m) {
        Err(e) => assert!(
            e.to_string().contains("InputNormalizationType"),
            "expected a LID-normalization bail, got: {e}"
        ),
        Ok(_) => panic!("fast Twin + LID InputNormalizationType != 0 must bail"),
    }
}

#[test]
fn fast_twin_bails_on_wav_arm() {
    // A real wav decode leaves periodogram unset -> the wav CNN arm bail at get_segmentation.
    let lidw = lid_weights();
    let mut drv = FastTwinLid::from_legacy(&map_of("twin_mode7"), None, Some(&lidw)).unwrap();
    let mut audio = read_audio(&phase4b("corpus_lid").join("f1.wav"), 0.0, 2.0, 0).unwrap();
    assert!(
        audio.periodogram.is_none(),
        "wav decode leaves periodogram unset"
    );
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    let err = drv
        .get_segmentation(&mut audio, &mut segs, None)
        .expect_err("fast Twin wav arm (CNN) must bail");
    assert!(
        err.to_string().contains("CNN"),
        "unexpected fast Twin wav-arm error: {err}"
    );
}
