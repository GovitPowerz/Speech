//! Phase 4a Task 5: `BagOfProcessors::segmentation_function` - the per-file flow
//! (`BagOfProcessors.cpp:207-407`, lock block stubbed) + 18-column result vector.
//!
//! The TDC config (Algo 1, NN-free) is the driver here: cumulative_error /
//! nb_of_classif are always 0 (no NN cost path), which pins cols 4 and 17 to 0
//! and makes cols 0-2 a pure function of `compute_errors` on the hyp/ref pair.

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::audio::read_audio;
use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::BagOfProcessors;
use speech::engine::corpus::CorpusItem;
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmentation_io::{compute_errors, load_ref_csv, load_ref_stm};

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

fn multi_mode() -> Mode {
    Mode {
        kind: ModeKind::Multi,
        verbose: false,
    }
}

fn solo_mode() -> Mode {
    Mode {
        kind: ModeKind::Solo,
        verbose: false,
    }
}

fn load_config(rel: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(ref_dir().join(rel)).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

/// A TDC config (Algo 1) with the top-level bag keys the phase2b fixture lacks,
/// plus offset 0.0 / max-duration 2.0 for a deterministic 2s excerpt.
fn tdc_bag_config() -> IndexMap<String, String> {
    let mut m = load_config("phase2b/tdc.config");
    m.insert("numOuterThreads".to_string(), "1".to_string());
    m.insert("Algo_choice".to_string(), "1".to_string());
    m.insert("TDC_lags".to_string(), "-0.002,0.016".to_string());
    m.insert("Audio_offset".to_string(), "0.0".to_string());
    m.insert("Audio_max_duration".to_string(), "2.0".to_string());
    m
}

/// Copy the shared excerpt into `dir` so the "write VRCTS next to audio" branch
/// lands in a tempdir, and return the copy's path.
fn wav_in(dir: &std::path::Path) -> PathBuf {
    let src = ref_dir().join("phase1/excerpt_2ch_8k.wav");
    let dst = dir.join("excerpt_2ch_8k.wav");
    std::fs::copy(&src, &dst).unwrap();
    dst
}

/// A 2-channel STM with one SPEECH span per channel, in-window for the 2s
/// excerpt. Line layout: `first chan second beg end third fourth` (7 tokens);
/// SPEECH iff `second.starts_with(first)`, `chan` is 1-based.
fn write_stm(dir: &std::path::Path) -> PathBuf {
    let p = dir.join("ref.stm");
    let body = "\
excerpt 1 excerpt 0.20 0.80 <o,f0,unk> hello world
excerpt 2 excerpt 0.30 0.90 <o,f0,unk> hello world
";
    std::fs::write(&p, body).unwrap();
    p
}

/// A CSV reference (channel-independent, unlike STM). `load_ref_csv` parses
/// `beg,end,type,conf` per line (commas -> spaces): one in-window SPEECH span
/// (`C` -> Speech) at conf 1.0 >= the default `Pruning_Threshold` 0.0.
fn write_csv(dir: &std::path::Path) -> PathBuf {
    let p = dir.join("ref.csv");
    std::fs::write(&p, "0.20,0.80,C,1.0\n").unwrap();
    p
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

// === scored_mode_result_columns =============================================
// Mode -m: cols 14,15,16 == 0.0 (no LID in 4a) and col 17 == nb_of_classif (0
// for TDC); col 3 (timing) is finite; cols 0-2 match a direct compute_errors on
// the same hyp/ref.
#[test]
fn scored_mode_result_columns() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path());
    let stm = write_stm(dir.path());

    let mut cfg = tdc_bag_config();
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), multi_mode()).unwrap();

    let it = item(wav.to_str().unwrap(), stm.to_str().unwrap());
    let results = bag.segmentation_function(&it, multi_mode()).unwrap();

    // One config (0), two channels.
    let cfg0 = &results[&0];
    assert_eq!(cfg0.len(), 2, "two channels");

    // Independent oracle: re-run the driver + compute_errors on channel 0.
    let stm_text = std::fs::read_to_string(&stm).unwrap();
    let mut audio = read_audio(&wav, 0.0, 2.0, 0, None).unwrap();
    let audio_duration = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let n_chan = audio.data.nrows();
    let mut hyp: Vec<Segmentation> = (0..n_chan)
        .map(|_| Segmentation::new(audio_duration))
        .collect();
    let mut cfg2 = tdc_bag_config();
    let mut bag2 =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg2), multi_mode()).unwrap();
    bag2.run_get_segmentation(0, &mut audio, &mut hyp).unwrap();

    for (chan, row) in cfg0 {
        let chan = *chan;
        let refc = load_ref_stm(&stm_text, chan, 0.0, 2.0, false);
        let report = compute_errors(&mut hyp[chan], Some(&refc), -1);
        let speech = report.per_class[speech::tasks::segmentation::SegClass::Speech as usize];
        let mut global = 0.0;
        for j in (speech::tasks::segmentation::SegClass::Other as usize)
            ..(speech::tasks::segmentation::SegClass::Excluded as usize)
        {
            global += report.per_class[j].error_rate;
        }
        assert_eq!(row[0], 100.0 * speech.pfa, "col0 Pfa chan {chan}");
        assert_eq!(row[1], 100.0 * speech.pmiss, "col1 Pmiss chan {chan}");
        assert_eq!(row[2], 100.0 * global, "col2 global error rate chan {chan}");
        assert!(row[3].is_finite(), "col3 timing finite chan {chan}");
        assert_eq!(row[4], 0.0, "col4 cumulative_error (TDC) chan {chan}");
        assert_eq!(row[14], 0.0, "col14 LID slot chan {chan}");
        assert_eq!(row[15], 0.0, "col15 LID slot chan {chan}");
        assert_eq!(row[16], 0.0, "col16 lid_nb_of_classif chan {chan}");
        assert_eq!(row[17], 0.0, "col17 nb_of_classif (TDC) chan {chan}");
        assert_eq!(row.len(), 18, "18 columns chan {chan}");
    }
}

// === missing_reference_in_scored_mode_errors ================================
// Mode -m with an empty ref_seg -> Err (legacy exit(1), :302-305).
#[test]
fn missing_reference_in_scored_mode_errors() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path());

    let mut cfg = tdc_bag_config();
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), multi_mode()).unwrap();

    let it = item(wav.to_str().unwrap(), "");
    match bag.segmentation_function(&it, multi_mode()) {
        Err(e) => assert!(
            e.to_string().contains("valid reference"),
            "expected mandatory-reference error, got: {e}"
        ),
        Ok(_) => panic!("expected scored mode with empty reference to bail"),
    }
}

// === stereo_csv_reference_loads_per_channel (F6) ============================
// FIXED (phase 5, F6): pre-fix, `load_ref_csv` fired only for `channel_count == 1`
// (mirroring the legacy `_ChannelNb == 1`-only `buf` build), so a stereo CSV
// reference loaded NOTHING and a scored (-m) run hit the mandatory-reference bail.
// PIN-OLD-FIRST verified: this test first pinned that bail (RED), then the fix
// (load the CSV ref for EVERY channel, cloned from the single channel-independent
// parse) makes the scored run succeed on BOTH channels. The scored columns are
// matched against an independent `load_ref_csv` + `compute_errors` oracle per
// channel -- proving the reference is genuinely CSV-derived, not an empty stub.
#[test]
fn stereo_csv_reference_loads_per_channel() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path()); // the 2-channel excerpt
    let csv = write_csv(dir.path());

    let mut cfg = tdc_bag_config();
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), multi_mode()).unwrap();

    let it = item(wav.to_str().unwrap(), csv.to_str().unwrap());
    // No longer bails: the CSV reference loads for both channels of the stereo file.
    let results = bag.segmentation_function(&it, multi_mode()).unwrap();
    let cfg0 = &results[&0];
    assert_eq!(cfg0.len(), 2, "two channels scored");

    // Independent oracle: CSV is channel-independent, so BOTH channels score against
    // the SAME `load_ref_csv` segmentation. Re-run the driver (TDC ignores the ref) to
    // get the hyp, then compute_errors per channel with the CSV ref + its nb_words
    // (the exact call the bag makes) and match cols 0-2 (Pfa/Pmiss/global).
    let csv_text = std::fs::read_to_string(&csv).unwrap();
    let mut audio = read_audio(&wav, 0.0, 2.0, 0, None).unwrap();
    let audio_duration = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let n_chan = audio.data.nrows();
    assert_eq!(
        n_chan, 2,
        "fixture must be stereo (else the F6 bug is untested)"
    );
    let mut hyp: Vec<Segmentation> = (0..n_chan)
        .map(|_| Segmentation::new(audio_duration))
        .collect();
    let mut cfg2 = tdc_bag_config();
    let mut bag2 =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg2), multi_mode()).unwrap();
    bag2.run_get_segmentation(0, &mut audio, &mut hyp).unwrap();

    // Pruning_Threshold is absent from tdc_bag_config -> default 0.0.
    let (refc, nb) = load_ref_csv(&csv_text, 0.0, 2.0, 0.0);
    // Non-vacuity: the reference actually carries a SPEECH span (not an empty stub).
    assert!(
        refc.segments()
            .iter()
            .any(|s| s.ty == speech::tasks::segmentation::SegClass::Speech),
        "the CSV reference must carry a SPEECH span (non-vacuous)"
    );
    for (chan, row) in cfg0 {
        let chan = *chan;
        let mut h = hyp[chan].clone();
        let report = compute_errors(&mut h, Some(&refc), nb);
        let speech = report.per_class[speech::tasks::segmentation::SegClass::Speech as usize];
        let mut global = 0.0;
        for j in (speech::tasks::segmentation::SegClass::Other as usize)
            ..(speech::tasks::segmentation::SegClass::Excluded as usize)
        {
            global += report.per_class[j].error_rate;
        }
        assert_eq!(row[0], 100.0 * speech.pfa, "col0 Pfa chan {chan}");
        assert_eq!(row[1], 100.0 * speech.pmiss, "col1 Pmiss chan {chan}");
        assert_eq!(row[2], 100.0 * global, "col2 global error rate chan {chan}");
    }
}

// === unscored_mode_zero_columns_and_vrcts ===================================
// Mode -s, no ref: cols 0,1,2,4 == 0.0; VRCTS written NEXT TO the audio
// (tempdir copy of the wav; :394-401 strips the 4-char extension).
#[test]
fn unscored_mode_zero_columns_and_vrcts() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path());

    let mut cfg = tdc_bag_config();
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), solo_mode()).unwrap();

    let it = item(wav.to_str().unwrap(), "");
    let results = bag.segmentation_function(&it, solo_mode()).unwrap();

    let cfg0 = &results[&0];
    for (chan, row) in cfg0 {
        assert_eq!(row[0], 0.0, "col0 zero chan {chan}");
        assert_eq!(row[1], 0.0, "col1 zero chan {chan}");
        assert_eq!(row[2], 0.0, "col2 zero chan {chan}");
        assert_eq!(row[4], 0.0, "col4 zero chan {chan}");
        assert_eq!(row.len(), 18, "18 columns chan {chan}");
    }

    // VRCTS written next to the audio: strip ".wav" (last 4 chars) to the
    // basefilename, then the multi-channel fan-out writes one <base>_chan_<n>.xml per
    // channel (the 2-channel excerpt -> _chan_1 + _chan_2; Task 8 multi-channel fix).
    for chan in [1, 2] {
        let expected = dir.path().join(format!("excerpt_2ch_8k_chan_{chan}.xml"));
        assert!(
            expected.exists(),
            "unscored VRCTS chan {chan} should be written next to audio at {}",
            expected.display()
        );
    }
}

// === dump_dir_vrcts =========================================================
// driver dump dir set -> VRCTS under dumpDir with the basename quirk (:352-356).
#[test]
fn dump_dir_vrcts() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path());
    let stm = write_stm(dir.path());
    let dump = dir.path().join("dumps");
    std::fs::create_dir(&dump).unwrap();

    let mut cfg = tdc_bag_config();
    cfg.insert(
        "Dump_Directory".to_string(),
        dump.to_str().unwrap().to_string(),
    );
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), multi_mode()).unwrap();

    let it = item(wav.to_str().unwrap(), stm.to_str().unwrap());
    bag.segmentation_function(&it, multi_mode()).unwrap();

    // basename = last path component minus the 4-char extension, under dumpDir; the
    // multi-channel fan-out writes <base>_chan_<n>.xml per channel (Task 8 fix).
    for chan in [1, 2] {
        let expected = dump.join(format!("excerpt_2ch_8k_chan_{chan}.xml"));
        assert!(
            expected.exists(),
            "scored VRCTS chan {chan} should be written under dumpDir at {}",
            expected.display()
        );
    }
}

// === speech_duration_walk ===================================================
// col 6 equals the manual walk over hyp segments (:324-330).
#[test]
fn speech_duration_walk() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path());
    let stm = write_stm(dir.path());

    let mut cfg = tdc_bag_config();
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), multi_mode()).unwrap();

    let it = item(wav.to_str().unwrap(), stm.to_str().unwrap());
    let results = bag.segmentation_function(&it, multi_mode()).unwrap();

    // Re-run the driver to get the same hyp segments and walk them by hand.
    let mut audio = read_audio(&wav, 0.0, 2.0, 0, None).unwrap();
    let audio_duration = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let n_chan = audio.data.nrows();
    let mut hyp: Vec<Segmentation> = (0..n_chan)
        .map(|_| Segmentation::new(audio_duration))
        .collect();
    let mut cfg2 = tdc_bag_config();
    let mut bag2 =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg2), multi_mode()).unwrap();
    bag2.run_get_segmentation(0, &mut audio, &mut hyp).unwrap();

    for (chan, row) in &results[&0] {
        // compute_errors sanitizes hyp in place; mirror that before the walk.
        let mut h = hyp[*chan].clone();
        let stm_text = std::fs::read_to_string(&stm).unwrap();
        let refc = load_ref_stm(&stm_text, *chan, 0.0, 2.0, false);
        let _ = compute_errors(&mut h, Some(&refc), -1);
        let segs = h.segments();
        let mut speech_duration = 0.0;
        for i in 0..segs.len().saturating_sub(1) {
            if segs[i].ty == speech::tasks::segmentation::SegClass::Speech {
                speech_duration += segs[i + 1].begin - segs[i].begin;
            }
        }
        assert_eq!(
            row[6], speech_duration,
            "col6 speech-duration walk chan {chan}"
        );
        // Non-vacuity: the walk is genuinely exercised (the 2s excerpt yields a
        // real speech interval, so this is not an all-zero pass).
        assert!(
            speech_duration > 0.0,
            "speech-duration walk non-trivial chan {chan}"
        );
    }
}

/// The REAL algo-3 config (`1_worker_1.config`) + real net weights
/// (`NNweights_config1.bin`, 33,671 f64) with the top-level bag keys and a 2s
/// window. `BackPropWER` defaults to -1 (no key) -> the simple-class target branch
/// in `getTargets` (SPEECH/SUBSTITUTION -> 1.0). Backprop is OFF in the base config,
/// but the FFB COST block is gated only on a non-empty target (NOT on backprop), so
/// the reference-driven target still makes the cost live.
fn spectral_bag_config() -> IndexMap<String, String> {
    let mut m = load_config("phase0/1_worker_1.config");
    m.insert("numOuterThreads".to_string(), "1".to_string());
    m.insert("Algo_choice".to_string(), "3".to_string());
    m.insert("Audio_offset".to_string(), "0.0".to_string());
    m.insert("Audio_max_duration".to_string(), "2.0".to_string());
    m.insert(
        "BLSTM_weightsFile".to_string(),
        ref_dir()
            .join("phase0/NNweights_config1.bin")
            .to_str()
            .unwrap()
            .to_string(),
    );
    m
}

// === spectral_scored_cost_is_live ===========================================
// Task 7b: a scored 2-channel run on the REAL algo-3 config with an STM reference
// now yields NONZERO nb_of_classif (col 17) and a finite NONZERO cumulative_error
// (col 4) on BOTH channels -- the corpus cost path is live (targets flow into
// `feed_forward_backward`, whose cost block accumulates on the non-empty target).
#[test]
fn spectral_scored_cost_is_live() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path());
    let stm = write_stm(dir.path());

    let mut cfg = spectral_bag_config();
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), multi_mode()).unwrap();

    let it = item(wav.to_str().unwrap(), stm.to_str().unwrap());
    let results = bag.segmentation_function(&it, multi_mode()).unwrap();

    let cfg0 = &results[&0];
    assert_eq!(cfg0.len(), 2, "two channels");
    for (chan, row) in cfg0 {
        assert!(
            row[17] > 0.0,
            "col17 nb_of_classif must be > 0 (live target) chan {chan}, got {}",
            row[17]
        );
        assert!(
            row[4].is_finite() && row[4] != 0.0,
            "col4 cumulative_error must be finite nonzero (live cost) chan {chan}, got {}",
            row[4]
        );
    }
}

// === spectral_no_reference_zero_cost ========================================
// The gating contrast (Task 7b): the SAME real algo-3 config, run WITHOUT a
// reference (unscored solo mode, empty ref_seg), keeps col 4 == 0.0 and col 17 == 0
// -- the drivers run the no-target forward path (empty target -> no cost/counter
// accumulation), byte-identical to the pre-target behaviour. This pins the
// reference gate: no reference => no live cost.
#[test]
fn spectral_no_reference_zero_cost() {
    let dir = tempfile::tempdir().unwrap();
    let wav = wav_in(dir.path());

    let mut cfg = spectral_bag_config();
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), solo_mode()).unwrap();

    // Empty ref_seg -> no reference loaded -> the no-target path (unscored solo mode
    // does not require a reference, unlike -m/-t).
    let it = item(wav.to_str().unwrap(), "");
    let results = bag.segmentation_function(&it, solo_mode()).unwrap();

    let cfg0 = &results[&0];
    assert_eq!(cfg0.len(), 2, "two channels");
    for (chan, row) in cfg0 {
        assert_eq!(
            row[4], 0.0,
            "col4 cumulative_error must be 0 with no reference chan {chan}"
        );
        assert_eq!(
            row[17], 0.0,
            "col17 nb_of_classif must be 0 with no reference chan {chan}"
        );
    }
}
