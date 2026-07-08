//! Phase 4b Task 8: `VrctsPart` (Algo 0, the external-tool adapter) tests.
//!
//! `VRCTSPart::getSegmentation` (`VRCTSpart.cpp:21-71`) never touches audio
//! samples -- it composes an xml path from `_RefSegFilename`, conditionally
//! shells out to the external `vrcts_part` binary, then re-ingests whatever xml
//! is on disk via `load_from_vrcts`. So these tests build `Audio`/`Segmentation`
//! directly (no wav decode needed) and drive `VrctsPart` through a tempdir.
//!
//! The legacy `vrcts_part` binary (`/usr/local/vrcts/vrcts_1_5_9/bin/vrcts_part`)
//! does not exist on this (or any CI) machine, so every code path that reaches
//! `std::process::Command::status()` here is exercised as a SPAWN failure. That is
//! precisely what `spawn_attempted_when_missing`/`force_respawns` pin: the typed
//! error this port surfaces where the legacy `system()` call would have silently
//! swallowed a "command not found" shell exit.

use std::path::Path;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::audio::Audio;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmentation_io::to_vrcts_string;
use speech::tasks::segmenter::Segmenter;
use speech::tasks::vrcts::VrctsPart;

/// Minimal valid `VRCTS_*` config: the generic `SegmenterConfig`/`DriverConfig`
/// required keys (replicating `Segmenter::buildFromConf`, `Segmenter.cpp:73-148`)
/// plus the two keys `VrctsPart` actually reads. `VRCTS_convolution_window_size
/// = 0` sidesteps also needing `VRCTS_convolution_window_type` (only read when
/// the window size is `> 0`, `Segmenter.cpp:100-103`).
fn vrcts_map(is_fast: bool, force: bool) -> IndexMap<String, String> {
    let mut m = IndexMap::new();
    m.insert(
        "VRCTS_decision_thresh_rising".to_string(),
        "0.6".to_string(),
    );
    m.insert("VRCTS_decision_area_rising".to_string(), "0.05".to_string());
    m.insert(
        "VRCTS_decision_thresh_falling".to_string(),
        "0.3".to_string(),
    );
    m.insert("VRCTS_decision_area_falling".to_string(), "0.1".to_string());
    m.insert("VRCTS_window".to_string(), "0.032".to_string());
    m.insert("VRCTS_shift".to_string(), "0.01".to_string());
    m.insert("VRCTS_convolution_window_size".to_string(), "0".to_string());
    m.insert(
        "VRCTS_speech_padding".to_string(),
        "0.1,0.1,0.1,0.1".to_string(),
    );
    m.insert("VRCTS_min_speech".to_string(), "0.2,0.2,0.2".to_string());
    m.insert("VRCTS_min_silence".to_string(), "0.2,0.2".to_string());
    m.insert("VRCTS_isFast".to_string(), is_fast.to_string());
    m.insert("VRCTS_force".to_string(), force.to_string());
    m
}

/// A tiny `Audio` stand-in: `VrctsPart` never reads `data`/`data_raw`, only the
/// file-identity fields.
fn stub_audio(audio_file_name: &str, ref_seg_file_name: &str) -> Audio {
    Audio {
        sample_rate: 8000,
        data: Array2::zeros((1, 1)),
        data_raw: Array2::zeros((1, 1)),
        lang_index: -1,
        weight: 1.0,
        external_features: Vec::new(),
        periodogram: None,
        audio_file_name: audio_file_name.to_string(),
        ref_seg_file_name: ref_seg_file_name.to_string(),
        audio_offset: 0.0,
    }
}

/// A 2.0s segmentation with one speech interval `[0.5, 1.5)`, matching what a
/// real VRCTS xml would contain.
fn seeded_segmentation() -> Segmentation {
    let mut seg = Segmentation::new(2.0);
    seg.label_segment(0.5, 1.5, SegClass::Speech);
    seg
}

// === skip-if-exists: no spawn, XML parsed back =================================

#[test]
fn skip_path_parses_preseeded_xml() {
    let dir = tempfile::tempdir().unwrap();
    let ref_seg = dir.path().join("myref").to_str().unwrap().to_string();
    let xml_path = format!("{ref_seg}_VRCTS_Fast.xml");

    let fixture = seeded_segmentation();
    std::fs::write(&xml_path, to_vrcts_string(&fixture, "name", "path")).unwrap();

    let mut vrcts = VrctsPart::from_legacy(&vrcts_map(true, false)).unwrap();
    let mut audio = stub_audio("unused.wav", &ref_seg);
    let mut seg_per_chan = vec![Segmentation::new(2.0)];

    // force=false, xml already exists -> no Command spawn is attempted. If a
    // spawn WERE attempted, it would fail (the binary is absent on this
    // machine) and this call would return Err -- so an Ok(()) here is itself
    // evidence the skip-if-exists branch held.
    vrcts
        .get_segmentation(&mut audio, &mut seg_per_chan, None)
        .unwrap();

    let segs = seg_per_chan[0].segments();
    assert_eq!(
        segs.len(),
        4,
        "Other@0, Speech@0.5, Other@1.5, End@2.0: {segs:?}"
    );
    assert_eq!(segs[0].ty, SegClass::Other);
    assert_eq!(segs[0].begin, 0.0);
    assert_eq!(segs[1].ty, SegClass::Speech);
    assert_eq!(segs[1].begin, 0.5);
    assert_eq!(segs[2].ty, SegClass::Other);
    assert_eq!(segs[2].begin, 1.5);
    assert_eq!(segs[3].ty, SegClass::End);
    assert_eq!(segs[3].begin, 2.0);
}

/// `_isFast = false` composes the `_Long` suffix and would pass `-q0` -- covered
/// indirectly here via the skip-if-exists path (no spawn needed to observe the
/// path composition).
#[test]
fn long_suffix_path_composition() {
    let dir = tempfile::tempdir().unwrap();
    let ref_seg = dir.path().join("myref").to_str().unwrap().to_string();
    let xml_path = format!("{ref_seg}_VRCTS_Long.xml");

    let fixture = seeded_segmentation();
    std::fs::write(&xml_path, to_vrcts_string(&fixture, "name", "path")).unwrap();

    let mut vrcts = VrctsPart::from_legacy(&vrcts_map(false, false)).unwrap();
    let mut audio = stub_audio("unused.wav", &ref_seg);
    let mut seg_per_chan = vec![Segmentation::new(2.0)];

    vrcts
        .get_segmentation(&mut audio, &mut seg_per_chan, None)
        .unwrap();

    let segs = seg_per_chan[0].segments();
    assert_eq!(segs[1].ty, SegClass::Speech, "loaded from the _Long xml");
}

// === spawn on a missing xml -> typed error ======================================

#[test]
fn spawn_attempted_when_missing() {
    let dir = tempfile::tempdir().unwrap();
    let ref_seg = dir.path().join("myref").to_str().unwrap().to_string();

    let mut vrcts = VrctsPart::from_legacy(&vrcts_map(true, false)).unwrap();
    let mut audio = stub_audio("some_audio.wav", &ref_seg);
    let mut seg_per_chan = vec![Segmentation::new(2.0)];

    let err = vrcts
        .get_segmentation(&mut audio, &mut seg_per_chan, None)
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("/usr/local/vrcts/vrcts_1_5_9/bin/vrcts_part"),
        "error should name the hard-coded binary path: {msg}"
    );

    assert!(
        !Path::new(&format!("{ref_seg}_VRCTS_Fast.xml")).exists(),
        "no xml should have been produced by a failed spawn"
    );
}

// === force=true respawns even when the xml already exists ======================

#[test]
fn force_respawns() {
    let dir = tempfile::tempdir().unwrap();
    let ref_seg = dir.path().join("myref").to_str().unwrap().to_string();
    let xml_path = format!("{ref_seg}_VRCTS_Fast.xml");

    let fixture = seeded_segmentation();
    let original_content = to_vrcts_string(&fixture, "name", "path");
    std::fs::write(&xml_path, &original_content).unwrap();

    let mut vrcts = VrctsPart::from_legacy(&vrcts_map(true, true)).unwrap();
    let mut audio = stub_audio("some_audio.wav", &ref_seg);
    let mut seg_per_chan = vec![Segmentation::new(2.0)];

    let err = vrcts
        .get_segmentation(&mut audio, &mut seg_per_chan, None)
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("/usr/local/vrcts/vrcts_1_5_9/bin/vrcts_part"));

    // The pre-existing xml was NOT silently reused: get_segmentation bailed
    // before ever reaching load_vrcts, so the hypothesis is untouched (still
    // seeded), and the file on disk is byte-identical to what it was before.
    assert_eq!(
        seg_per_chan[0].segments().len(),
        2,
        "still [Other@0, End@dur]"
    );
    assert_eq!(seg_per_chan[0].segments()[0].ty, SegClass::Other);
    let on_disk = std::fs::read_to_string(&xml_path).unwrap();
    assert_eq!(on_disk, original_content, "xml on disk left untouched");
}

// === load-bearing quirk: no per-channel xml suffix ==============================

#[test]
fn same_xml_shared_across_channels() {
    let dir = tempfile::tempdir().unwrap();
    let ref_seg = dir.path().join("myref").to_str().unwrap().to_string();
    let xml_path = format!("{ref_seg}_VRCTS_Fast.xml");

    let fixture = seeded_segmentation();
    std::fs::write(&xml_path, to_vrcts_string(&fixture, "name", "path")).unwrap();

    let mut vrcts = VrctsPart::from_legacy(&vrcts_map(true, false)).unwrap();
    let mut audio = stub_audio("unused.wav", &ref_seg);
    let mut seg_per_chan = vec![Segmentation::new(2.0), Segmentation::new(2.0)];

    vrcts
        .get_segmentation(&mut audio, &mut seg_per_chan, None)
        .unwrap();

    // The path composition (VRCTSpart.cpp:34-38) has no `_chan_<n>` suffix (the
    // per-channel variant is commented out in the legacy source) -- every
    // channel loads the SAME xml, so both channels see the identical hypothesis.
    assert_eq!(
        seg_per_chan[0].segments().len(),
        seg_per_chan[1].segments().len()
    );
    for (a, b) in seg_per_chan[0]
        .segments()
        .iter()
        .zip(seg_per_chan[1].segments())
    {
        assert_eq!(a.ty, b.ty);
        assert_eq!(a.begin, b.begin);
    }
    assert_eq!(seg_per_chan[0].segments()[1].ty, SegClass::Speech);
}

// === required-key validation (Segmenter::buildFromConf, Segmenter.cpp:73-148) ==

/// A config missing one of the generic `SegmenterConfig`/`DriverConfig`
/// required keys (here `VRCTS_min_silence`) must fail to construct, matching
/// `buildFromConf`'s `exit(1)` -- even though `VRCTS_min_silence` itself is
/// never read by `getSegmentation`.
#[test]
fn missing_required_key_rejected() {
    let mut m = vrcts_map(true, false);
    m.shift_remove("VRCTS_min_silence");

    // `.err().unwrap()` (not `.unwrap_err()`): `VrctsPart` doesn't derive `Debug`,
    // and `.err()` -> `Option<E>` doesn't require the `Ok` type to (matches the
    // `is_err()`-only convention used by the other `from_legacy` config tests).
    let err = VrctsPart::from_legacy(&m).err().unwrap();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("VRCTS_min_silence"),
        "error should name the missing key: {msg}"
    );
}
