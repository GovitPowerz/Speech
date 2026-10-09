//! Phase 8 Task 7: the per-utterance streaming LID session
//! (`fast::stream_lid::StreamingLidSession`) vs the offline fast Mode-7 LID Twin
//! (`fast::driver::FastTwinLid`), over the committed `twin_mode7.config` + the real
//! 12409-weight LID net + the committed phSeq corpus (`corpus_phseq/s{1,2,3}.phSeq`) and
//! the phase-6 cep fixtures (File_Type 2).
//!
//! LID is UTTERANCE-GRANULAR by design: one `external_features` entry (one phSeq line /
//! one cep record) is one "utterance". The session folds them one at a time through the
//! SAME shared per-entry kernel the offline driver's loop uses, so:
//!  - `finish()` == the offline FastTwinLid members (bit-identical) on the same entries,
//!  - the running aggregate after k pushes == the offline run on the FIRST k entries
//!    (prefix-correctness), the strong non-circular pin (the offline driver is the oracle).

use indexmap::IndexMap;
use ndarray::Array2;
use std::path::PathBuf;

use speech::audio::{Audio, read_audio};
use speech::fast::driver::FastTwinLid;
use speech::fast::stream_lid::{LidAggregate, StreamingLidSession};
use speech::io::binary::read_weight_vector;
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

/// The offline fast members after running FastTwinLid on `audio` (channel 0):
/// `(lid_classification_errors, is_lid_correct, lid_segments_confusion)`.
struct OfflineMembers {
    errors: Vec<f64>,
    correct: i32,
    confusion: Array2<f64>,
}

fn run_offline(mut audio: Audio) -> OfflineMembers {
    let lidw = lid_weights();
    let mut drv = FastTwinLid::from_legacy(&map_of("twin_mode7"), None, Some(&lidw)).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    OfflineMembers {
        errors: drv.lid_classification_errors()[0].clone(),
        correct: drv.is_lid_correct()[0],
        confusion: drv.lid_segments_confusion()[0].clone(),
    }
}

/// The offline members on the FIRST `k` external-features entries (the prefix oracle): a
/// truncated `external_features` leaves the periodogram/data untouched (the LID loop reads
/// only `external_features`; the SAD side is a constant-10.0 row), so this is exactly the
/// offline run on a k-utterance corpus.
fn run_offline_prefix(mut audio: Audio, k: usize) -> OfflineMembers {
    audio.external_features.truncate(k);
    run_offline(audio)
}

fn session(map: &IndexMap<String, String>, lang: i32) -> StreamingLidSession {
    StreamingLidSession::new(map, 8000.0, lang, Some(&lid_weights())).unwrap()
}

/// Bit-identical member comparison (fast-vs-fast: no tolerance -- the session and the
/// offline share the same f32 kernel, so equality is EXACT, not approximate).
fn assert_members_eq(tag: &str, a: &LidAggregate, b: &OfflineMembers) {
    assert_eq!(
        a.classification_errors, b.errors,
        "{tag}: classification_errors must be bit-identical to offline"
    );
    assert_eq!(
        a.is_lid_correct, b.correct,
        "{tag}: is_lid_correct must equal offline"
    );
    assert_eq!(
        a.confusion, b.confusion,
        "{tag}: confusion must be bit-identical to offline"
    );
}

// ---------------------------------------------------------------------------
// per_utterance_equals_offline: finish() == the offline full run, bit-identical.
// ---------------------------------------------------------------------------

const PHSEQ_FILES: [(&str, i32); 3] = [("s1", 0), ("s2", 1), ("s3", 1)];
const CEP_FIXTURES: [(&str, i32); 2] = [("tiny_ok.plp", 0), ("multi_ok.plp", 0)];

fn per_utterance_equals_offline(audio: Audio, lang: i32, tag: &str) {
    let feats = audio.external_features.clone();
    let mut sess = session(&map_of("twin_mode7"), lang);

    let mut last: Option<LidAggregate> = None;
    for (i, feat) in feats.iter().enumerate() {
        let sc = sess.push_utterance(feat);
        // A scored utterance's argmax over `scores` must agree with the reported argmax.
        if let Some(j) = sc.argmax {
            assert!(
                !sc.scores.is_empty(),
                "{tag} u{i}: scored -> non-empty scores"
            );
            let recomputed = argmax(&sc.scores);
            assert_eq!(
                j, recomputed,
                "{tag} u{i}: reported argmax vs scores argmax"
            );
            assert_eq!(
                sc.is_correct,
                Some(j == sess_target(lang)),
                "{tag} u{i}: is_correct == (argmax == target)"
            );
        } else {
            assert!(sc.scores.is_empty(), "{tag} u{i}: skipped -> empty scores");
        }
        last = Some(sc.running_aggregate);
    }

    // finish() == the offline full run, bit-identical.
    let fin = sess.finish();
    let offline = run_offline(audio);
    assert_members_eq(&format!("{tag} finish"), &fin, &offline);
    // T9 pin: the shipped-but-previously-unasserted predicted_language field == the offline
    // argmax, recovered from the offline classification_errors oracle (langid[c] =
    // errors[c]/100 + target_lid[c]; target_lid[ti] = -2, else 0), then first-max argmax.
    let want_pred = offline_predicted_language(&offline.errors, sess_target(lang));
    assert_eq!(
        fin.predicted_language, want_pred,
        "{tag}: predicted_language must equal the offline argmax (recovered from errors)"
    );
    // finish() == the last push's running aggregate (they finalize the same state).
    if let Some(la) = last {
        assert_eq!(la.classification_errors, fin.classification_errors);
        assert_eq!(la.confusion, fin.confusion);
        assert_eq!(la.is_lid_correct, fin.is_lid_correct);
    }
    // Idempotent finish.
    let fin2 = sess.finish();
    assert_members_eq(&format!("{tag} finish2"), &fin2, &offline);
}

fn sess_target(lang: i32) -> usize {
    // class_nb == 2 for the binary LID net; ti = clamp(lang, [0, 2)).
    if lang >= 0 && (lang as usize) < 2 {
        lang as usize
    } else {
        0
    }
}

fn argmax(v: &[f64]) -> usize {
    let mut j = 0usize;
    let mut best = v[0];
    for (c, &x) in v.iter().enumerate().skip(1) {
        if x > best {
            best = x;
            j = c;
        }
    }
    j
}

/// The offline predicted language (oracle for `LidAggregate.predicted_language`): recover the
/// normalized langID from the offline `classification_errors` (`errors[c] = 100*(langid[c] -
/// target_lid[c])`, `target_lid[ti] = -2`, else 0), then take its first-max argmax -- the same
/// argmax the driver's `finalize_lid_channel` computes, routed through the asserted errors row.
fn offline_predicted_language(errors: &[f64], ti: usize) -> usize {
    let langid: Vec<f64> = errors
        .iter()
        .enumerate()
        .map(|(c, &e)| e / 100.0 + if c == ti { -2.0 } else { 0.0 })
        .collect();
    argmax(&langid)
}

#[test]
fn per_utterance_equals_offline_phseq() {
    for (f, lang) in PHSEQ_FILES {
        per_utterance_equals_offline(phseq_audio(f, lang), lang, &format!("phseq {f}"));
    }
}

#[test]
fn per_utterance_equals_offline_cep() {
    for (name, lang) in CEP_FIXTURES {
        per_utterance_equals_offline(cep_audio(name, lang), lang, &format!("cep {name}"));
    }
}

// ---------------------------------------------------------------------------
// running_aggregate_is_prefix_correct: after k pushes == offline on first k entries.
// ---------------------------------------------------------------------------

fn prefix_correct(make: impl Fn() -> Audio, lang: i32, tag: &str) {
    let feats = make().external_features;
    let mut sess = session(&map_of("twin_mode7"), lang);

    for (i, feat) in feats.iter().enumerate() {
        let k = i + 1;
        let sc = sess.push_utterance(feat);
        // Oracle: the REAL offline driver on the first k entries (truncated external
        // features; data/periodogram untouched).
        let oracle = run_offline_prefix(make(), k);
        assert_members_eq(
            &format!("{tag} prefix k={k}"),
            &sc.running_aggregate,
            &oracle,
        );
    }
}

#[test]
fn running_aggregate_is_prefix_correct_phseq() {
    for (f, lang) in PHSEQ_FILES {
        prefix_correct(|| phseq_audio(f, lang), lang, &format!("phseq {f}"));
    }
}

#[test]
fn running_aggregate_is_prefix_correct_cep() {
    for (name, lang) in CEP_FIXTURES {
        prefix_correct(|| cep_audio(name, lang), lang, &format!("cep {name}"));
    }
}

// ---------------------------------------------------------------------------
// new() reuses FastTwinLid's construction validation (the same typed bails).
// ---------------------------------------------------------------------------

#[test]
fn new_bails_on_non_mode7() {
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_LID_Mode".into(), "0".into());
    match StreamingLidSession::new(&m, 8000.0, 0, Some(&lid_weights())) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("mode 7"),
            "expected a mode-7-only bail, got: {e}"
        ),
        Ok(_) => panic!("streaming LID + Mode != 7 must bail"),
    }
}

#[test]
fn new_bails_on_missing_lid_net() {
    // No explicit weights AND an EMPTY BLSTM_LID_weightsFile -> the net can't load; new must
    // bail loudly instead of deferring the failure to the first push. (The committed config
    // carries a repo-root-relative weightsFile; blank it so load_weights_file is a no-op.)
    let mut m = map_of("twin_mode7");
    m.insert("BLSTM_LID_weightsFile".into(), String::new());
    match StreamingLidSession::new(&m, 8000.0, 0, None) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("net"),
            "expected a missing-net bail, got: {e}"
        ),
        Ok(_) => panic!("streaming LID with no LID net must bail"),
    }
}

// ---------------------------------------------------------------------------
// Issue #57: EVERY shape streams. The session is utterance-granular, so a bidirectional
// LID net streams as well as a causal one (ADR-0004's refusal is the FRAME stream's). Per
// (cell, direction) pair of the committed phase-9 Mode-7 LID fixtures: the running
// aggregate after every push == the offline fast Twin on the first k entries, and
// `finish` == the offline full run, bit-identical. The `(lstm, bidirectional)` pair is the
// real-net legs above.
// ---------------------------------------------------------------------------

const MATRIX_FIXTURES: [&str; 9] = [
    "twin_mode7_lid_slstm",
    "twin_mode7_lid_mamba",
    "twin_mode7_lid_cfc",
    "twin_mode7_lid_transformer",
    "twin_mode7_lid_lstm_forward",
    "twin_mode7_lid_slstm_forward",
    "twin_mode7_lid_mamba_forward",
    "twin_mode7_lid_cfc_forward",
    "twin_mode7_lid_transformer_forward",
];

fn fixture_map(name: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(ref_dir().join(format!("phase9/{name}.config"))).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

fn fixture_pack(name: &str) -> Vec<f64> {
    read_weight_vector(&ref_dir().join(format!("phase9/{name}_seed.bin"))).unwrap()
}

/// The offline fast members on `audio` for an arbitrary (map, pack) pair (channel 0).
fn run_offline_on(map: &IndexMap<String, String>, lid: &[f64], mut audio: Audio) -> OfflineMembers {
    let mut drv = FastTwinLid::from_legacy(map, None, Some(lid)).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    OfflineMembers {
        errors: drv.lid_classification_errors()[0].clone(),
        correct: drv.is_lid_correct()[0],
        confusion: drv.lid_segments_confusion()[0].clone(),
    }
}

#[test]
fn every_shape_streams_prefix_correct_and_finish_equals_offline() {
    for fixture in MATRIX_FIXTURES {
        let map = fixture_map(fixture);
        let lid = fixture_pack(fixture);
        for (f, lang) in PHSEQ_FILES {
            let tag = format!("{fixture}/{f}");
            let feats = phseq_audio(f, lang).external_features;
            let mut sess = StreamingLidSession::new(&map, 8000.0, lang, Some(&lid))
                .unwrap_or_else(|e| panic!("{tag}: every shape streams: {e}"));
            for (i, feat) in feats.iter().enumerate() {
                let k = i + 1;
                let sc = sess.push_utterance(feat);
                let mut prefix = phseq_audio(f, lang);
                prefix.external_features.truncate(k);
                let oracle = run_offline_on(&map, &lid, prefix);
                assert_members_eq(
                    &format!("{tag} prefix k={k}"),
                    &sc.running_aggregate,
                    &oracle,
                );
            }
            let fin = sess.finish();
            let offline = run_offline_on(&map, &lid, phseq_audio(f, lang));
            assert_members_eq(&format!("{tag} finish"), &fin, &offline);
            assert_eq!(
                fin.segments_count,
                sess.scored_count(),
                "{tag}: scored count"
            );
        }
    }
}
