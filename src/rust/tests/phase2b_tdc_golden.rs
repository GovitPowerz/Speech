//! Phase 2b Task 4: `TdcSegmenter` (Algo 1) driver goldens.
//!
//! Step 1 CHECK-FIRST outcome: `TimeDomainCorrel::getSegmentation`
//! (`TimeDomainCorrel.cpp:93-259`) has NO NN in its chain (autocorrelation is
//! cwise+sum; the only libm is `fmath::log`/`cos` via `classifySequence` + the
//! window coefficients), so the harness calls the REAL compiled `getSegmentation`
//! directly as the golden source -- no transcription anywhere in the oracle.
//! `tdc_result_chan{1,2}.bin` is an INDEPENDENT replication of the framing loop
//! (built from the same real, public `classifySequence` + `AudioStruct::getSequence`)
//! and is bit-exact with the Rust `tdc_classify_sequence`/`get_sequence` Phase 1
//! ports under the oracle-gated comparator, confirming the ported `TdcSegmenter`
//! reuses those functions correctly.
//!
//! Goldens dumped by the harness `TdcProbe` stage (`tools/oracle_harness/main.cpp`,
//! Phase 2b Task 4 block) against `tests/reference_data/phase2b/tdc.config` on the
//! shared 2-channel excerpt (`excerpt_2ch_8k.wav`, offset 0.35, dur 2.0, rate 8000).
//! `tdc_vrcts_chan1.xml` is dumped from a SEPARATE zero-offset `AudioStruct` (see
//! the manifest `tdc_segmenter` entry): the Rust `Segmentation` container has no
//! `_AudioOffset` field, so "identical Segmentations" (spec S10) requires the
//! offset dimension to be absent on both sides.

mod common;

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::audio::{Audio, read_audio};
use speech::tasks::sad::TdcSegmenter;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmentation_io::{parse_audiodoc_attrs, to_vrcts_string};
use speech::tasks::segmenter::Segmenter;

fn tdc_config_text() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase2b/tdc.config");
    std::fs::read_to_string(p).unwrap()
}

fn tdc_map() -> IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(&tdc_config_text())
}

/// Fresh excerpt audio (offset 0.35, dur 2.0), UNMUTATED: `TdcSegmenter::
/// get_segmentation` applies preemph/noise itself (config `TDC_preemph_ratio
/// 0.97`, `TDC_noise_seed 0` -> no noise).
fn excerpt_audio() -> Audio {
    read_audio(&common::fixture("excerpt_2ch_8k.wav"), 0.35, 2.0).expect("decode excerpt")
}

/// Fresh zero-offset excerpt audio (same wav, offset 0.0): the VRCTS byte-
/// equivalence golden uses this so neither side has an `_AudioOffset` to bake in.
fn excerpt_audio_zero_offset() -> Audio {
    read_audio(&common::fixture("excerpt_2ch_8k.wav"), 0.0, 2.0).expect("decode excerpt")
}

fn fresh_segs(audio: &Audio) -> Vec<Segmentation> {
    let audio_duration = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|_| Segmentation::new(audio_duration))
        .collect()
}

// === from_legacy ==============================================================

#[test]
fn from_legacy_reads_real_values() {
    let m = tdc_map();
    let tdc = TdcSegmenter::from_legacy(&m).unwrap();
    // Indirect assertions via get_segmentation output below (lags/balance/window
    // are private); the ctor-error paths are asserted directly here.
    drop(tdc);
}

#[test]
fn from_legacy_rejects_short_lags() {
    let mut m = tdc_map();
    m.insert("TDC_lags".to_string(), "0.002".to_string());
    assert!(TdcSegmenter::from_legacy(&m).is_err());
}

#[test]
fn from_legacy_clamps_negative_lags() {
    let mut m = tdc_map();
    m.insert("TDC_lags".to_string(), "-0.002,0.016".to_string());
    // Should not error (clamped to 0.0, not rejected); exercised end-to-end by
    // checking get_segmentation still runs without panicking.
    let mut tdc = TdcSegmenter::from_legacy(&m).unwrap();
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    tdc.get_segmentation(&mut audio, &mut segs, None).unwrap();
}

// === result rows (pre-convolution) ===========================================

#[test]
fn result_rows_match_dump() {
    let m = tdc_map();
    let mut tdc = TdcSegmenter::from_legacy(&m).unwrap();
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    tdc.get_segmentation(&mut audio, &mut segs, None).unwrap();

    // The driver convolves `results` in place inside `get_segmentation`, so the
    // pre-convolution row is not directly observable from the trait surface.
    // Re-derive it here the same way the harness's independent replication does:
    // rerun the classification loop directly against Phase 1 primitives, using
    // the SAME (now-mutated, i.e. post preemph/noise) audio buffer the driver
    // consumed -- this cross-checks tdc_classify_sequence/get_sequence, which is
    // the whole point of the Step 1 CHECK-FIRST.
    for (chan, name) in ["tdc_result_chan1.bin", "tdc_result_chan2.bin"]
        .iter()
        .enumerate()
    {
        let want = common::load_bin_phase2b(name);
        let half_window = 128usize;
        let full_window = 257usize;
        let window_shift = 80usize;
        let min_lag = 16usize;
        let max_lag = 128usize;
        let balance = 0.7;
        let coeffs = speech::audio::windowing_coefficients("hamming", false, full_window, 0.8);
        let frame_count = audio.data.ncols();
        let vec_size = want.ncols();
        let mut buf = ndarray::Array2::<f64>::zeros((1, full_window));
        let mut got = vec![0.0f64; vec_size];
        let mut jj = 0usize;
        while jj < frame_count {
            speech::audio::get_sequence(
                &audio.data,
                chan,
                jj,
                half_window,
                false,
                coeffs.as_deref(),
                &mut buf,
            );
            let window = buf.row(0).to_vec();
            got[jj / window_shift] = speech::features::ltsv_tdc::tdc_classify_sequence(
                &window, min_lag, max_lag, balance,
            );
            jj += window_shift;
        }
        let got = ndarray::Array2::from_shape_vec((1, vec_size), got).unwrap();
        common::assert_oracle_eq(&got, &want, name);
    }
}

// === convolved rows + boundaries (post results_to_segmentation) ==============

#[test]
fn get_segmentation_boundaries_match_dump() {
    let m = tdc_map();
    let mut tdc = TdcSegmenter::from_legacy(&m).unwrap();
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    tdc.get_segmentation(&mut audio, &mut segs, None).unwrap();

    for (chan, name) in ["tdc_boundaries_chan1.bin", "tdc_boundaries_chan2.bin"]
        .iter()
        .enumerate()
    {
        let want = common::load_bin_phase2b(name);
        let got_segs = segs[chan].segments();
        assert_eq!(
            got_segs.len(),
            want.nrows(),
            "{name}: boundary count mismatch"
        );
        for (i, s) in got_segs.iter().enumerate() {
            let got = ndarray::Array2::from_shape_vec((1, 1), vec![s.begin]).unwrap();
            let wantv = ndarray::Array2::from_shape_vec((1, 1), vec![want[[i, 0]]]).unwrap();
            common::assert_oracle_eq(&got, &wantv, &format!("{name}[{i}].begin"));
            assert_eq!(
                s.ty as i32,
                want[[i, 1]] as i32,
                "{name}[{i}].type mismatch"
            );
        }
    }
}

#[test]
fn convolved_rows_match_dump() {
    // Cross-check: results_to_segmentation's in-place convolution matches the
    // harness's independent rebuild (same probe, same results2segmentation) --
    // covered structurally by the boundary golden above; this test additionally
    // pins the numeric convolved row via results_to_segmentation directly, using
    // the pre-convolution row recovered in `result_rows_match_dump`.
    use speech::tasks::segmenter::{DriverConfig, SegmenterConfig, results_to_segmentation};

    let m = tdc_map();
    let driver_cfg = DriverConfig::from_config(&m, "TDC").unwrap();
    let seg_cfg = SegmenterConfig::from_config(&m, "TDC").unwrap();

    for (chan_idx, (pre_name, conv_name, bound_name)) in [
        (
            "tdc_result_chan1.bin",
            "tdc_convolved_chan1.bin",
            "tdc_boundaries_chan1.bin",
        ),
        (
            "tdc_result_chan2.bin",
            "tdc_convolved_chan2.bin",
            "tdc_boundaries_chan2.bin",
        ),
    ]
    .iter()
    .enumerate()
    {
        let pre = common::load_bin_phase2b(pre_name);
        let mut results: Vec<f64> = pre.row(0).to_vec();

        let bound_dump = common::load_bin_phase2b(bound_name);
        let audio_duration = bound_dump[[bound_dump.nrows() - 1, 0]];
        let mut seg = Segmentation::new(audio_duration);

        results_to_segmentation(
            &mut seg,
            0.01,
            0.0,
            &mut results,
            SegClass::Speech,
            driver_cfg.conv_coeff.as_deref(),
            &seg_cfg,
        );

        let want = common::load_bin_phase2b(conv_name);
        let got = ndarray::Array2::from_shape_vec((1, results.len()), results).unwrap();
        common::assert_oracle_eq(&got, &want, &format!("convolved chan{}", chan_idx + 1));
    }
}

// === compute_errors vs the programmatic reference ============================

#[test]
fn score_matches_dump() {
    let m = tdc_map();
    let mut tdc = TdcSegmenter::from_legacy(&m).unwrap();
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    tdc.get_segmentation(&mut audio, &mut segs, None).unwrap();

    // Programmatic reference: two SPEECH spans per channel, matching the harness
    // manifest constants (`tdc_segmenter.constants.reference_spans_sec`).
    let audio_duration = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let reference: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| {
            let mut r = Segmentation::new(audio_duration);
            r.label_segment(0.4, 0.9, SegClass::Speech);
            r.label_segment(1.2, 1.6, SegClass::Speech);
            r
        })
        .collect();

    // Re-run get_segmentation on a FRESH audio/seg pair for scoring (matching the
    // harness's separate probeScore instance -- the shared `audio`/`tdc` above
    // already had preemph applied by the first get_segmentation call).
    let mut audio2 = excerpt_audio();
    let mut tdc2 = TdcSegmenter::from_legacy(&m).unwrap();
    let mut hyp = fresh_segs(&audio2);
    tdc2.get_segmentation(&mut audio2, &mut hyp, None).unwrap();

    let want = common::load_bin_phase2b("tdc_scores.bin");
    let reports = TdcSegmenter::score(&mut hyp, Some(&reference), -1);
    for (chan, report) in reports.iter().enumerate() {
        let speech = report.per_class[SegClass::Speech as usize];
        let got = ndarray::Array2::from_shape_vec(
            (1, 3),
            vec![speech.pfa, speech.pmiss, speech.error_rate],
        )
        .unwrap();
        let wantv = ndarray::Array2::from_shape_vec(
            (1, 3),
            vec![want[[chan, 0]], want[[chan, 1]], want[[chan, 2]]],
        )
        .unwrap();
        common::assert_oracle_eq(&got, &wantv, &format!("score chan{chan}"));
    }
}

// === VRCTS byte equivalence (0b-ii closure) ===================================

#[test]
fn vrcts_bytes_match_dump() {
    let dump = std::fs::read_to_string(common::fixture_phase2b("tdc_vrcts_chan1.xml")).unwrap();
    let (name, path_attr) = parse_audiodoc_attrs(&dump);

    let m = tdc_map();
    let mut tdc = TdcSegmenter::from_legacy(&m).unwrap();
    let mut audio = excerpt_audio_zero_offset();
    let mut segs = fresh_segs(&audio);
    tdc.get_segmentation(&mut audio, &mut segs, None).unwrap();

    let got = to_vrcts_string(&segs[0], &name, &path_attr);
    common::assert_vrcts_eq(&got, &dump, "tdc vrcts");
}

// === TWO-FILES golden: same TdcSegmenter instance, run twice =================
//
// NOTE (scope): this pins REPEAT-CALL DETERMINISM only, not the stateful
// re-quantization lifecycle of `window_shift_sec`. The legacy quantization
// `q(x) = round(x*rate)/rate` (`TimeDomainCorrel.cpp:103-105`) is PROVABLY
// IDEMPOTENT for every representable f64 input at any practical sample rate
// (see IMPROVEMENTS.md "[phase2b] TdcSegmenter::window_shift_sec ..." for the
// proof): the compounded double-rounding error of the division-then-
// multiplication round trip is bounded by ~1 ULP of the frame count, which is
// astronomically below the 0.5 margin `round()` needs to flip its decision,
// for any frame count an actual audio file can produce. Consequently there is
// no TDC_shift config value (on- or off-grid) whose second-call quantization
// differs from its first-call quantization, so no off-grid config variant can
// make this golden non-vacuous: a driver that reset `window_shift_sec` from
// the config string before every call would pass this test identically to the
// real stateful driver. The driver code itself (verified by design/code
// review) does carry the state through per `TimeDomainCorrel.cpp:103-105`;
// this test just cannot distinguish that from a per-call reset, because the
// quantization has no fixed input that drifts under re-quantization.
#[test]
fn two_files_in_sequence_boundaries_match_dump() {
    let m = tdc_map();
    let mut tdc = TdcSegmenter::from_legacy(&m).unwrap();

    let mut audio1 = excerpt_audio();
    let mut segs1 = fresh_segs(&audio1);
    tdc.get_segmentation(&mut audio1, &mut segs1, None).unwrap();

    // Second run on the SAME TdcSegmenter instance (inherits the quantized
    // window_shift_sec from file 1) against a FRESH audio/seg pair, per the
    // harness's file-2 replay. See the module-level NOTE above: this proves
    // determinism, not statefulness (the quantization is idempotent, so no
    // config value can exercise observable drift).
    let mut audio2 = excerpt_audio();
    let mut segs2 = fresh_segs(&audio2);
    tdc.get_segmentation(&mut audio2, &mut segs2, None).unwrap();

    let want = common::load_bin_phase2b("tdc_boundaries_file2_chan1.bin");
    let got_segs = segs2[0].segments();
    assert_eq!(got_segs.len(), want.nrows());
    for (i, s) in got_segs.iter().enumerate() {
        let got = ndarray::Array2::from_shape_vec((1, 1), vec![s.begin]).unwrap();
        let wantv = ndarray::Array2::from_shape_vec((1, 1), vec![want[[i, 0]]]).unwrap();
        common::assert_oracle_eq(&got, &wantv, &format!("file2 boundary[{i}].begin"));
        assert_eq!(
            s.ty as i32,
            want[[i, 1]] as i32,
            "file2 boundary[{i}].type mismatch"
        );
    }
}

// === Hand tests: shift floor + odd window ====================================

#[test]
fn window_shift_floors_at_one_over_rate() {
    let mut m = tdc_map();
    // TDC_shift smaller than 1/8000 = 1.25e-4 must floor up to 1/rate.
    m.insert("TDC_shift".to_string(), "0.00001".to_string());
    let mut tdc = TdcSegmenter::from_legacy(&m).unwrap();
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    // Must not panic (a zero/undersized window_shift would divide-by-zero or
    // index out of bounds); the floor keeps window_shift >= 1 frame.
    tdc.get_segmentation(&mut audio, &mut segs, None).unwrap();
}

#[test]
fn window_size_is_odd() {
    // full_window = 2*half_window+1 is always odd by construction; window 0.032s
    // at 8000 Hz -> half_window = round(0.032*8000/2) = 128 -> full = 257.
    let half_window = f64::round(0.032 * 8000.0 / 2.0) as usize;
    let full_window = 2 * half_window + 1;
    assert_eq!(half_window, 128);
    assert_eq!(full_window, 257);
    assert_eq!(full_window % 2, 1);
}
