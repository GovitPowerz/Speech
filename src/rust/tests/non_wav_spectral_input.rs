//! Issue #121: the `!hasReadWavFile()` branch on the spectral drivers, keyed on the audio as
//! the legacy keys it (`tasks::sad::non_wav_ssif`). The port's `!hasReadWavFile()` is
//! `audio.periodogram.is_some()`, set only by the phSeq / cep readers, so a wav read marked
//! that way is a non-wav read to every driver here.
//!
//!  - Algo 5 has no other non-wav branch (`BLSTMSpectralLID.cpp:52` is its only
//!    `hasReadWavFile` site): it computes its periodogram off the waveform either way, so the
//!    forced 80-frame spectrum shift is the whole difference, and the port reproduces it;
//!  - the spectral SAD driver (Algo 3) and the Twin's modes 0-6 feed the precomputed
//!    features to the net on a non-wav read (`BLSTMSpectralSegmenter.cpp:564-567`, `:672`;
//!    `TwinBLSTMSpectralLID.cpp:354`, `:642`), an arm the port does not have: they refuse the
//!    read before touching the carry, as Mode 7 refuses a wav. The fast SAD driver refuses it
//!    too.

use std::path::PathBuf;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::audio::{Audio, read_audio};
use speech::fast::driver::FastSpectralSegmenter;
use speech::io::binary::read_matrix;
use speech::tasks::lid::{BlstmSpectralLid, TwinBlstmSpectralLid};
use speech::tasks::sad::{BlstmSpectralSegmenter, NON_WAV_SSIF};
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmenter::Segmenter;

fn ref_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data")
        .join(rel)
}

fn map_at(rel: &str) -> IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(&std::fs::read_to_string(ref_path(rel)).unwrap())
}

/// The T2 corpus excerpt (8 kHz, 2 channels, 2 s), a wav read.
fn wav_audio() -> Audio {
    read_audio(&ref_path("phase4b/corpus_lid/f1.wav"), 0.35, 2.0, 0, None)
        .expect("decode corpus wav")
}

/// The same samples marked as a phSeq / cep read.
fn non_wav(mut audio: Audio) -> Audio {
    audio.periodogram = Some(Array2::zeros((1, 1)));
    audio
}

fn fresh_segs(audio: &Audio) -> Vec<Segmentation> {
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect()
}

fn refusal(r: anyhow::Result<()>, who: &str) -> String {
    match r {
        Ok(()) => panic!("{who}: a non-wav read must be refused"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn algo5_non_wav_read_takes_the_legacy_80() {
    // `BLSTM_spectrum_shift 0.025` (the committed Twin configs' value) derives 200 frames at
    // 8 kHz; lid5's own 0.01 derives 80 already and would make the pin vacuous.
    let mut map = map_at("phase4b/lid5.config");
    map.insert("BLSTM_spectrum_shift".into(), "0.025".into());
    let (rows, cols, w) = read_matrix(&ref_path("phase0/NNweights_config1.bin")).unwrap();
    assert_eq!(rows * cols, 33_671, "real net weight count");
    let result_rows = |mut audio: Audio| {
        let mut segs = fresh_segs(&audio);
        let mut drv = BlstmSpectralLid::from_legacy(&map, Some(&w)).unwrap();
        drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
        drv.last_result_rows()[0].len()
    };
    // The LTSV-decimated result_vec sizing (`:157-160`, `:221-232`) at a spectrum shift.
    let ltsv_shift: f64 = map["BLSTM_LTSVshift"].parse().unwrap();
    let frames = wav_audio().data.ncols();
    let sized = |ssif: usize| {
        let ltsv_ws = ((ltsv_shift * 8000.0 / ssif as f64).round() as usize).max(1);
        frames.div_ceil(ssif).div_ceil(ltsv_ws)
    };
    assert_ne!(sized(200), sized(NON_WAV_SSIF));
    assert_eq!(
        result_rows(wav_audio()),
        sized(200),
        "wav read: the derived 200"
    );
    assert_eq!(
        result_rows(non_wav(wav_audio())),
        sized(NON_WAV_SSIF),
        "non-wav read: the forced 80"
    );
}

#[test]
fn spectral_sad_refuses_a_non_wav_read_before_the_carry() {
    let mut drv =
        BlstmSpectralSegmenter::from_legacy(&map_at("phase4a/tier2_spectral.config"), None)
            .unwrap();
    let mut audio = non_wav(wav_audio());
    let mut segs = fresh_segs(&audio);
    let msg = refusal(drv.get_segmentation(&mut audio, &mut segs, None), "algo 3");
    assert!(msg.contains("non-wav input"), "{msg}");
    assert_eq!(
        drv.spectrum_shift_in_frames(),
        0,
        "the refusal precedes the carry"
    );
}

#[test]
fn twin_modes_0_to_6_refuse_a_non_wav_read() {
    for mode in 0..=6 {
        let map = map_at(&format!("phase4b/twin_mode{mode}.config"));
        let mut drv = TwinBlstmSpectralLid::from_legacy(&map, None, None).unwrap();
        let mut audio = non_wav(wav_audio());
        let mut segs = fresh_segs(&audio);
        let msg = refusal(
            drv.get_segmentation(&mut audio, &mut segs, None),
            &format!("twin mode {mode}"),
        );
        assert!(msg.contains("non-wav input"), "mode {mode}: {msg}");
    }
}

#[test]
fn fast_spectral_sad_refuses_a_non_wav_read() {
    let mut drv =
        FastSpectralSegmenter::from_legacy(&map_at("phase4a/tier2_spectral.config"), None).unwrap();
    let mut audio = non_wav(wav_audio());
    let mut segs = fresh_segs(&audio);
    let msg = refusal(
        drv.get_segmentation(&mut audio, &mut segs, None),
        "fast algo 3",
    );
    assert!(msg.contains("non-wav input"), "{msg}");
}
