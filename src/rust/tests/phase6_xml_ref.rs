//! Phase 6 Task 2b: `.xml` (VRCTS) references route to `load_ref_vrcts` in the
//! corpus reference dispatch (`BagOfProcessors::segmentation_function` +
//! `extension_of`), unblocking scored SAD training on the corpus `.part.xml`
//! VRCTS references.
//!
//! Source-governs: the legacy `Segmentation` ctor `.xml` branch
//! (`Segmentation.cpp:89-100`) calls `load_ref_from_vrcts` (`:808-829`), the
//! REFERENCE loader -- a DIFFERENT function from `load_from_vrcts` (`:592-614`,
//! the DUMP parse-back loader the port's `load_vrcts` was ported from). The
//! reference loader is CHANNEL-SLICED by the 1-based `ch="N"` attribute and
//! windows on `_AudioDuration`, NOT the embedded `<Channel sigdur>`. The port
//! adds `load_ref_vrcts` (faithful to the reference loader) and wires it here;
//! `load_vrcts` is left untouched for `VrctsPart`.
//!
//! `xml_reference_real_corpus_part_xml` is gated on the licensed
//! `data/LRE03-LRE07` corpus and skips cleanly when it is absent (nothing from
//! the corpus is committed; the real path lives in test code only).

mod common;

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use speech::audio::read_audio;
use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::BagOfProcessors;
use speech::engine::corpus::CorpusItem;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmentation_io::{compute_errors, load_ref_vrcts, load_vrcts};

use common::corpus_root_or_skip;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

fn multi_mode() -> Mode {
    Mode {
        kind: ModeKind::Multi,
        verbose: false,
    }
}

/// A TDC config (Algo 1, NN-free -> deterministic hyp, no weights) over a 2s
/// window at offset 0.0 -- the same NN-free driver `phase4a_segfn.rs` uses so the
/// scored columns are a pure function of `compute_errors(hyp, ref)`.
fn tdc_bag_config(max_duration: f64) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(ref_dir().join("phase2b/tdc.config")).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&text);
    m.insert("numOuterThreads".to_string(), "1".to_string());
    m.insert("Algo_choice".to_string(), "1".to_string());
    m.insert("TDC_lags".to_string(), "-0.002,0.016".to_string());
    m.insert("Audio_offset".to_string(), "0.0".to_string());
    m.insert("Audio_max_duration".to_string(), max_duration.to_string());
    m
}

fn item(file_name: &str, ref_seg: &str) -> CorpusItem {
    CorpusItem {
        file_name: file_name.to_string(),
        ref_seg: ref_seg.to_string(),
        language: "unk".to_string(),
        dialect: "unk".to_string(),
        class_index: 0,
        file_id: 1,
        weight: 1.0,
    }
}

fn wav_in(dir: &Path) -> PathBuf {
    let src = ref_dir().join("phase1/excerpt_2ch_8k.wav");
    let dst = dir.join("excerpt_2ch_8k.wav");
    std::fs::copy(&src, &dst).unwrap();
    dst
}

/// A synthetic 2-channel VRCTS document with a DIFFERENT SPEECH span per channel:
/// `ch="1"` -> [0.20, 0.80], `ch="2"` -> [1.20, 1.80]. Distinct spans make the
/// channel-slicing observable (an identical-per-channel loader would give both
/// channels both spans). Synthetic content only.
fn two_channel_vrcts() -> String {
    "\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<AudioDoc name=\"excerpt\" path=\"/tmp/excerpt.wav\">
<ProcList>
<Proc name=\"vrcts_part\" version=\"1.3\"/>
</ProcList>
<ChannelList>
<Channel num=\"1\" sigdur=\"2.00\" spdur=\"0.60\"/>
<Channel num=\"2\" sigdur=\"2.00\" spdur=\"0.60\"/>
</ChannelList>
<SpeakerList>
<Speaker ch=\"1\" dur=\"0.60\" gender=\"1\" spkid=\"1\"/>
<Speaker ch=\"2\" dur=\"0.60\" gender=\"1\" spkid=\"1\"/>
</SpeakerList>
<SegmentList>
<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"0.2000\" etime=\"0.8000\" spkid=\"1\"/>
<SpeechSegment ch=\"2\" sconf=\"1.00\" stime=\"1.2000\" etime=\"1.8000\" spkid=\"1\"/>
</SegmentList>
</AudioDoc>
"
    .to_string()
}

// === xml_reference_scored_run_through_dispatch (RED -> GREEN) ================
// Before the wiring: `extension_of` classifies `.xml` as `RefExt::Trs`, so the
// scored (-m) run BAILS with the TRS "not ported" message at `.unwrap()` (RED).
// After the wiring: the run is SCORED on both channels, and cols 0-2
// (Pfa/Pmiss/global) match an independent `load_ref_vrcts` + `compute_errors`
// oracle per channel -- proving the reference is genuinely VRCTS-derived and
// channel-sliced (chan 0 scored against ch="1", chan 1 against ch="2"), not an
// empty stub. A non-vacuity guard (each channel's reference carries a SPEECH
// span) plus a nonzero global-error guard make the scoring sharp, not just
// "did not bail".
#[test]
fn xml_reference_scored_run_through_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path());
    let xml = dir.path().join("ref.xml");
    std::fs::write(&xml, two_channel_vrcts()).unwrap();

    let mut cfg = tdc_bag_config(2.0);
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), multi_mode()).unwrap();

    let it = item(wav.to_str().unwrap(), xml.to_str().unwrap());
    // RED here pre-fix: the dispatch bails on the `.xml` reference (TRS branch).
    let results = bag.segmentation_function(&it, multi_mode()).unwrap();

    let cfg0 = &results[&0];
    assert_eq!(cfg0.len(), 2, "two channels scored");

    // Independent oracle: re-run the NN-free TDC driver to recover the per-channel
    // hyp, then score each channel against its OWN VRCTS reference (channel-sliced).
    let xml_text = std::fs::read_to_string(&xml).unwrap();
    let mut audio = read_audio(&wav, 0.0, 2.0, 0).unwrap();
    let audio_duration = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let n_chan = audio.data.nrows();
    assert_eq!(
        n_chan, 2,
        "fixture must be stereo (channel-slicing is tested)"
    );
    let mut hyp: Vec<Segmentation> = (0..n_chan)
        .map(|_| Segmentation::new(audio_duration))
        .collect();
    let mut cfg2 = tdc_bag_config(2.0);
    let mut bag2 =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg2), multi_mode()).unwrap();
    bag2.run_get_segmentation(0, &mut audio, &mut hyp).unwrap();

    for (chan, row) in cfg0 {
        let chan = *chan;
        let refc = load_ref_vrcts(&xml_text, chan, 0.0, audio_duration);
        // Non-vacuity: each channel's reference carries its own SPEECH span.
        assert!(
            refc.segments().iter().any(|s| s.ty == SegClass::Speech),
            "channel {chan} VRCTS reference must carry a SPEECH span (non-vacuous)"
        );
        let mut h = hyp[chan].clone();
        let report = compute_errors(&mut h, Some(&refc), -1);
        let speech = report.per_class[SegClass::Speech as usize];
        let mut global = 0.0;
        for j in (SegClass::Other as usize)..(SegClass::Excluded as usize) {
            global += report.per_class[j].error_rate;
        }
        assert_eq!(row[0], 100.0 * speech.pfa, "col0 Pfa chan {chan}");
        assert_eq!(row[1], 100.0 * speech.pmiss, "col1 Pmiss chan {chan}");
        assert_eq!(row[2], 100.0 * global, "col2 global error rate chan {chan}");
        // Sharpness: the VRCTS reference genuinely disagrees with the hyp, so the
        // scoring is non-trivial (not a 0 == 0 pass against an empty reference).
        assert!(
            global > 0.0,
            "col2 global error must be nonzero (sharp scoring) chan {chan}, got {global}"
        );
    }
}

// === xml_reference_is_channel_sliced ========================================
// Pins the per-channel semantics of `load_ref_from_vrcts` (`Segmentation.cpp:816-820`):
// each `<SpeechSegment>` is routed to the channel named in its 1-based `ch="N"`
// attribute. Channel 0 (ch="1") gets ONLY [0.20, 0.80]; channel 1 (ch="2") gets
// ONLY [1.20, 1.80]. An identical-per-channel loader (the `VrctsPart` shared-xml
// quirk) would put BOTH spans on BOTH channels -- this test would then fail.
#[test]
fn xml_reference_is_channel_sliced() {
    let text = two_channel_vrcts();

    let chan0 = load_ref_vrcts(&text, 0, 0.0, 2.0);
    let chan1 = load_ref_vrcts(&text, 1, 0.0, 2.0);

    // The SPEECH intervals present in each channel (begin, end) pairs.
    let speech_spans = |seg: &Segmentation| -> Vec<(f64, f64)> {
        let s = seg.segments();
        (0..s.len().saturating_sub(1))
            .filter(|&i| s[i].ty == SegClass::Speech)
            .map(|i| (s[i].begin, s[i + 1].begin))
            .collect()
    };

    assert_eq!(
        speech_spans(&chan0),
        vec![(0.20, 0.80)],
        "channel 0 carries ONLY the ch=\"1\" span"
    );
    assert_eq!(
        speech_spans(&chan1),
        vec![(1.20, 1.80)],
        "channel 1 carries ONLY the ch=\"2\" span"
    );
}

// === xml_reference_windows_on_audio_duration_not_sigdur =====================
// The load-bearing difference from `load_vrcts` (why the reference dispatch does
// NOT reuse it): `load_ref_vrcts` seeds the reference over `[0, dur]` where `dur`
// is the audio frame count (`_AudioDuration`), while `load_vrcts` reads the
// embedded `<Channel sigdur>` and seeds over `[0, sigdur]`. Both window
// *inclusion* on the passed `dur` (`begin < off + dur`), so they admit the same
// segments; the divergence is the `End` sentinel (the segmentation extent), which
// governs how a straddling segment is clamped AND what the reference is scored
// against. For a `.part.xml` with sigdur=1721 whose audio is `_DurationMax`-capped
// to e.g. 30s, the reference MUST end at the audio duration (as the hyp does), not
// at 1721 -- else `compute_errors` scores a 30s hyp against a 1721s reference.
#[test]
fn xml_reference_windows_on_audio_duration_not_sigdur() {
    let text = "\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<AudioDoc name=\"x\" path=\"/tmp/x.wav\">
<ChannelList>
<Channel num=\"1\" sigdur=\"1800.00\" spdur=\"1.00\"/>
</ChannelList>
<SegmentList>
<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"0.2000\" etime=\"0.8000\" spkid=\"1\"/>
<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"0.9000\" etime=\"1.5000\" spkid=\"1\"/>
</SegmentList>
</AudioDoc>
";
    // Audio capped to 1.0s.
    let refc = load_ref_vrcts(text, 0, 0.0, 1.0);
    let dumpc = load_vrcts(text, 0.0, 1.0);

    // The reference loader seeds the `End` sentinel at the AUDIO duration (dur),
    // NEVER the embedded sigdur; load_vrcts seeds it at sigdur.
    assert_eq!(
        refc.audio_duration(),
        1.0,
        "load_ref_vrcts End sentinel = audio dur (1.0)"
    );
    assert_eq!(
        dumpc.audio_duration(),
        1800.0,
        "load_vrcts End sentinel = embedded sigdur (1800)"
    );

    // Consequence: the straddling [0.90, 1.50] segment is CLAMPED to the 1.0s cap
    // in the reference (its last SPEECH boundary ends at the 1.0 End), but extends
    // to 1.50 under load_vrcts (End at 1800). The reference carries a real SPEECH
    // span (non-vacuous), and no SPEECH boundary exceeds the cap.
    let max_speech_end = |seg: &Segmentation| -> f64 {
        let s = seg.segments();
        (0..s.len().saturating_sub(1))
            .filter(|&i| s[i].ty == SegClass::Speech)
            .map(|i| s[i + 1].begin)
            .fold(0.0_f64, f64::max)
    };
    assert!(
        refc.segments().iter().any(|s| s.ty == SegClass::Speech),
        "the in-window SPEECH must remain (non-vacuous)"
    );
    assert_eq!(
        max_speech_end(&refc),
        1.0,
        "reference SPEECH is clamped to the 1.0s audio cap"
    );
    assert_eq!(
        max_speech_end(&dumpc),
        1.5,
        "load_vrcts SPEECH extends to 1.5 (End at sigdur=1800)"
    );
}

// === xml_reference_real_corpus_part_xml (corpus-gated smoke) =================
// LOCAL-ONLY: skips cleanly when the licensed corpus is absent. Parses a REAL
// corpus `.part.xml` through the dispatch's reference loader and asserts a
// non-empty reference, then drives the full `segmentation_function` scored path
// on the matching wav (capped) so the real file exercises the wiring end to end.
// Nothing from the corpus is committed; the corpus-relative path lives here only.
#[test]
fn xml_reference_real_corpus_part_xml() {
    let Some(root) = corpus_root_or_skip() else {
        return;
    };
    let corpus_dir = root.join("train/audio/LRE03");
    if !corpus_dir.is_dir() {
        eprintln!("SKIP: {} absent", corpus_dir.display());
        return;
    }
    let Some((part_xml, wav)) = find_part_xml_with_wav(&corpus_dir) else {
        eprintln!(
            "SKIP: no .part.xml + sibling wav pair under {}",
            corpus_dir.display()
        );
        return;
    };

    let dur_cap = 2.0;
    // The exact reference-load call `segmentation_function`'s `.xml` arm makes.
    let text = std::fs::read_to_string(&part_xml).unwrap();
    let refc = load_ref_vrcts(&text, 0, 0.0, dur_cap);
    assert!(
        refc.segments().iter().any(|s| s.ty == SegClass::Speech),
        "real corpus .part.xml must yield a non-empty SPEECH reference: {}",
        part_xml.display()
    );

    // End to end through the dispatch on the real mono wav (capped to dur_cap).
    let mut cfg = tdc_bag_config(dur_cap);
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), multi_mode()).unwrap();
    let it = item(wav.to_str().unwrap(), part_xml.to_str().unwrap());
    let results = bag.segmentation_function(&it, multi_mode()).unwrap();
    assert_eq!(
        results[&0].len(),
        1,
        "corpus file is mono -> one channel scored"
    );
    eprintln!(
        "OK: scored real corpus pair {} / {}",
        wav.file_name().unwrap().to_str().unwrap(),
        part_xml.file_name().unwrap().to_str().unwrap()
    );
}

/// First (lexicographically) `<stem>.part.xml` in `dir` that has a sibling wav
/// `<stem>.*.wav` (the corpus wav carries an `.MT1.mp1`/`.FT1.mp1` infix, so the
/// name is not a plain extension swap). Deterministic pick.
fn find_part_xml_with_wav(dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    for p in &entries {
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = name.strip_suffix(".part.xml") else {
            continue;
        };
        let prefix = format!("{stem}.");
        if let Some(wav) = entries.iter().find(|w| {
            w.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&prefix) && n.ends_with(".wav"))
                .unwrap_or(false)
        }) {
            return Some((p.clone(), wav.clone()));
        }
    }
    None
}
