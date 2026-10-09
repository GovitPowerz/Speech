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
//! PHASE 10 TASK 5 (the S4 full-f32-mel sweep) RE-RAN THIS SUITE AND NOTHING MOVED --
//! `max_abs`/`max_rel` bit-for-bit the phase-7 values on both legs, pins untouched. That
//! is STRUCTURAL, not luck: Mode 7 consumes `external_features` (phSeq one-hots / cep
//! records) and its SAD net is FROZEN (a synthesized-constant `result_vec`), so no
//! periodogram, mel bank or DCT is ever built on this arm -- `FastPipeline` is not even
//! constructed. The suite is recorded in the sweep table as an unchanged row precisely
//! because an unexplained MOVE here would have meant the mel change leaked somewhere it
//! has no business being.
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
use speech::engine::corpus_processor::CorpusProcessor;
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
    let mut a = read_audio(&path, 0.0, 120.0, 1, None).unwrap();
    a.lang_index = lang;
    a.weight = 1.0;
    a.audio_file_name = path.to_string_lossy().into_owned();
    a
}

fn cep_audio(name: &str, lang: i32) -> Audio {
    let path = phase6(&format!("cep/{name}"));
    let mut a = read_audio(&path, 0.0, 3.6e6, 2, None).unwrap();
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
            read_audio(&phase6(&format!("cep/{name}")), 0.0, 3.6e6, 2, None).is_err(),
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
    // them (via build_aligned_spec's BlstmConfig source).
    let lidw = lid_weights();
    let drv = FastTwinLid::from_legacy(&map_of("twin_mode7"), None, Some(&lidw)).unwrap();
    assert_eq!(
        drv.debug_lid_peepholes(),
        Some([true; 6]),
        "fast LID net must carry the config's TRUE peepholes"
    );

    // Omitting config: the aligned spec must give the BlstmConfig default TRUE under the
    // LID prefix too (NnetSpec defaulted FALSE until issue #32).
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
    // twin_mode7.config's committed `fileslisting`/`language2classmapping` values are
    // repo-root-relative; `cargo test` runs with CWD == the crate dir (`src/rust`), so
    // absolutize them here (T5 finding 2's CorpusProcessor-level tests are the first
    // users of this helper that go through Corpus::from_config, which opens these files
    // directly). Every OTHER existing caller of this helper only feeds the map to
    // BagOfProcessors::from_configs, which never reads either key, so this is additive.
    m.insert(
        "language2classmapping".into(),
        phase4b("languagemapping_lid7.csv").to_str().unwrap().into(),
    );
    m.insert(
        "fileslisting".into(),
        phase4b("corpus_phseq/listing_lid7.csv")
            .to_str()
            .unwrap()
            .into(),
    );
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

/// T6b (bag_of_processors.rs): `set_weights` on a fast-dispatched Twin conf must bail
/// loudly instead of the pre-fix silent `Ok(())` no-op -- the same closed hole as the
/// FastSpectral case (`bag_of_processors.rs`'s own `fast_spectral_set_weights_bails_loudly`
/// unit test), pinned here too since `FastTwinLid` is the OTHER fast arm sharing that match
/// statement and needs its own construction fixture (`twin_map_fast`, Mode-7-shaped).
#[test]
fn fast_twin_set_weights_bails_loudly() {
    let mut m = twin_map_fast();
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode()).unwrap();
    assert!(matches!(bag.processor(0), Processor::FastTwinLid(_)));

    let dummy_sad = vec![0.0; 4];
    let lidw = lid_weights();
    match bag.set_weights(0, &[dummy_sad, lidw]) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("Inference_Path"),
                "error must name Inference_Path, got: {msg}"
            );
            assert!(
                msg.contains("BLSTM_weightsFile"),
                "error must name the config-time weight-file mechanism, got: {msg}"
            );
        }
        Ok(()) => {
            panic!("set_weights on a fast-dispatched Twin conf must bail, not silently no-op")
        }
    }
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

#[test]
fn fast_training_guard_blocks_image_mode_epochs() {
    // T5 finding 2 (review): the from_configs training-shaped bail above is Multi-gated
    // (`is_multi_mode`), so it does NOT fire for Image -- without the run()->train() guard
    // in CorpusProcessor::train(), a fast + Image + Epochs>0 config would construct fine
    // and then silently no-op-train (the fast processors' training arms are inert).
    let mut m = twin_map_fast();
    m.insert("Neural_Networks_BackPropagation_Epochs".into(), "2".into());
    let mut cp = CorpusProcessor::new(vec![m], image_mode()).unwrap();
    let err = cp
        .run()
        .expect_err("fast + Image + Epochs>0 must error, not silently no-op-train");
    let msg = err.to_string();
    assert!(
        msg.contains("Inference_Path"),
        "error must name Inference_Path, got: {msg}"
    );
    assert!(
        msg.to_lowercase().contains("image"),
        "error must name the training mode (Image), got: {msg}"
    );
}

/// `listing_lid7.csv`'s committed rows point at `corpus_phseq/sN.phSeq`, relative to
/// `tests/reference_data/phase4b/` (where the fixture was authored) -- not `cargo test`'s
/// CWD (the crate dir). An unscored run (this listing's refseg column is empty, and
/// Image-mode-no-reference is unscored per `bag_of_processors.rs`'s `scored` gate) ALWAYS
/// writes its VRCTS hypothesis NEXT TO THE AUDIO when `Dump_Directory` is empty (a
/// documented legacy quirk, `bag_of_processors.rs:1163-1200) -- so merely absolutizing the
/// listing to point at the COMMITTED fixture dir would write a stray `.xml` there on every
/// run. COPY the 3 phSeq files into a fresh tempdir instead, and point the listing at the
/// copies, so that write (and CorpusProcessor::run()'s real audio decode) lands in the
/// tempdir. No process-global `set_current_dir` needed (self-contained, parallel-safe).
fn tempdir_lid7_listing(dir: &std::path::Path) -> String {
    for f in ["s1.phSeq", "s2.phSeq", "s3.phSeq"] {
        std::fs::copy(phase4b(&format!("corpus_phseq/{f}")), dir.join(f)).unwrap();
    }
    let text = std::fs::read_to_string(phase4b("corpus_phseq/listing_lid7.csv")).unwrap();
    let abs_prefix = dir.to_str().unwrap().to_string();
    let rewritten = text.replace("corpus_phseq/", &format!("{abs_prefix}/"));
    let out = dir.join("listing_lid7_abs.csv");
    std::fs::write(&out, rewritten).unwrap();
    out.to_str().unwrap().to_string()
}

#[test]
fn fast_training_guard_allows_image_mode_no_epochs() {
    // The parity legs' shape: twin_mode7.config carries no Epochs key (defaults to 0), so
    // Image mode takes the run_solo (inference) path -- train() is never reached, so the
    // new guard must not affect it. Exercises a REAL audio decode (unlike the guard-error
    // leg above), so the listing needs a tempdir corpus (see tempdir_lid7_listing).
    let dir = tempfile::tempdir().unwrap();
    let mut m = twin_map_fast();
    m.insert("fileslisting".into(), tempdir_lid7_listing(dir.path()));
    let mut cp = CorpusProcessor::new(vec![m], image_mode()).unwrap();
    cp.run()
        .expect("fast + Image + Epochs=0 (inference) must run cleanly");
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

/// ISSUE #57 replaced the blanket BLSTM-only gate (`fast_twin_bails_on_unsupported_cell_type_
/// and_direction`, which pinned a refusal on BOTH prefixes) with the (cell x direction)
/// matrix. What survives is a REGIME rule, pinned here on the real net: the overlap LID
/// windowing is refused on every shape, and a windowed CAUSAL LID net is refused
/// (`tests/fast_twin_lid_matrix.rs` pins the causal half on the committed forward fixtures
/// and the build-per-shape half). Both bail where the window resolves against the rate
/// (`get_segmentation`), NOT at construction -- the pair must BUILD first.
#[test]
fn fast_twin_refuses_the_overlap_lid_regime() {
    // `window 0.25 / shift 0.0` is the committed truncate pair; `shift 0.1` resolves 10
    // periodogram frames (>= 1) -> overlap.
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_LID_shift".into(), "0.1".into());
    let mut drv =
        build_fast_twin(&m).expect("the overlap refusal is a regime rule, not a construction bail");
    let mut audio = phseq_audio("s1", 0);
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    let err = drv
        .get_segmentation(&mut audio, &mut segs, None)
        .expect_err("fast Twin + overlap LID windowing must bail");
    assert!(
        err.to_string()
            .contains("overlap LID windowing is unsupported"),
        "unexpected overlap-regime error: {err}"
    );
}

/// The inverse of the old gate: a NON-LSTM bidirectional LID net now BUILDS on the fast
/// Twin, so the build is pinned on pack LENGTH instead -- the hazard the shape bail was
/// really protecting, re-aimed as phase 10 did for the SAD driver. At the real net's
/// geometry (`36,24 / 48,1`, measured through `init_weights`) the bidirectional packs are
/// lstm 12409, slstm 11833, mamba 14521, cfc 12235, transformer 13113: the 12409-weight
/// LSTM pack is REFUSED under `mamba` and `transformer` (too short) and ACCEPTED head-first
/// under `slstm` and `cfc` (the documented file-load tolerance every fast twin keeps). The
/// per-shape builds from their OWN packs live in `tests/fast_twin_lid_matrix.rs`.
#[test]
fn fast_twin_pins_the_lid_pack_length_per_cell() {
    use speech::fast::driver::FastNetShape;
    use speech::nn::blstm::CellType;
    for (cell, ct, refused) in [
        ("mamba", CellType::Mamba, true),
        ("transformer", CellType::Transformer, true),
        ("slstm", CellType::Slstm, false),
        ("cfc", CellType::Cfc, false),
    ] {
        let mut m = map_of("twin_mode7");
        m.insert("BLSTM_LID_Cell_Type".into(), cell.into());
        match (build_fast_twin(&m), refused) {
            (Err(e), true) => assert!(
                e.to_string().contains("too short"),
                "the LSTM pack under Cell_Type {cell} must fail on LENGTH, got: {e}"
            ),
            (Ok(drv), false) => assert_eq!(
                drv.lid_shape(),
                FastNetShape::BiCell(ct),
                "{cell}: the fast Twin must build the bidirectional cell twin"
            ),
            (Err(e), false) => panic!("{cell}: a head-first-accepted pack must build, got: {e}"),
            (Ok(_), true) => panic!("{cell}: a too-short pack must not build"),
        }
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
fn fast_twin_bails_on_negative_target_enforcement_step() {
    // T5 finding 1 (review): a negative BLSTM_LID_TargetEnforcementStep makes the exact
    // path's truncate-windowed cost block overwrite interior OUTPUT rows with -0.5
    // (nn/blstm.rs:1244-1265), which the fast path's forward-only scoring never
    // reproduces -- must typed-bail at construction, naming the key.
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_LID_TargetEnforcementStep".into(), "-1".into());
    match build_fast_twin(&m) {
        Err(e) => assert!(
            e.to_string().contains("TargetEnforcementStep"),
            "expected a TargetEnforcementStep bail, got: {e}"
        ),
        Ok(_) => panic!("fast Twin + negative BLSTM_LID_TargetEnforcementStep must bail"),
    }
}

#[test]
fn fast_twin_bails_on_mlp_mode() {
    // T5 finding 3 (review, minor): contract symmetry with the exact dispatch's is_mlp
    // route to the MLP drivers (nn/blstm.rs:1032-1043) -- the fast LID scoring always
    // runs the truncate BLSTM forward, never MLP, so an is_mlp LID config must bail.
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_LID_LSTMNeuronNb".into(), "0,48".into());
    match build_fast_twin(&m) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("mlp"),
            "expected an MLP-mode bail, got: {e}"
        ),
        Ok(_) => panic!("fast Twin + is_mlp LID config must bail"),
    }
}

#[test]
fn fast_twin_bails_on_wav_arm() {
    // A real wav decode leaves periodogram unset -> the wav CNN arm bail at get_segmentation.
    let lidw = lid_weights();
    let mut drv = FastTwinLid::from_legacy(&map_of("twin_mode7"), None, Some(&lidw)).unwrap();
    let mut audio = read_audio(&phase4b("corpus_lid").join("f1.wav"), 0.0, 2.0, 0, None).unwrap();
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
