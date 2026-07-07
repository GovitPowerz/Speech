//! Phase 4b Task 4: `BlstmSpectralLid` (Algo 5) driver goldens -- the FIRST LID driver.
//!
//! `BLSTMSpectralLID::getSegmentation` (`:27-460`) does SAD via LTSV (`classifySequence`
//! over the RAW periodogram, `:271-274` -- the mel branch is COMMENTED OUT) then runs the
//! BLSTM PER SPEECH SEGMENT for language scoring (`:346-401`, the returning `feedForward`
//! overload `:363`). The harness `LidProbe` stage transcribes `:27-460` swapping ONLY that
//! `:363` feedForward for the reimpl scoring (target build + `signalReimplFFB` + the REAL
//! `CostLaw::computeCost`), keeping LTSV/`results2segmentation`/`compute_errors` REAL; the
//! reimpl dumps are what this port matches. A harness SECONDARY real-Eigen probe cross-checks
//! SEG_STRUCT (segment count/type equality) + LID_STRUCT (confusion + `_IsLIDCorrect`
//! equality, `langid_max_abs ~ 1e-16`); a structural mismatch aborts fixture generation.
//!
//! Goldens (`lid5_<f>_{ltsv,liderr,confusion,members,boundaries}_chan{1,2}.bin`) from
//! `lid5.config` (Algo 5, plain scoring path `BLSTM_window 0`, forced-low SAD thresholds ->
//! one big SPEECH segment/file so the LID scoring loop always runs) + the real 33,671-weight
//! SAD net (`NNweights_config1.bin`) over the T2 corpus (`f1/f2/f3.wav`, class 0/1/2 via
//! `lang_index`; 2 channels, offset 0.35, dur 2.0, rate 8000, frame_count 16001, ssr 4,
//! spectrum_shift 80, vec_size 201, LTSV_window_shift 6, real_vec_size 34).
//!
//! Tolerance: the LTSV row + liderr + cumulative_error are libm-dependent (periodogram
//! windowing cos / softmax / LogLaw), so canary-gated (`assert_oracle_eq*`). The confusion +
//! nb_of_classif + is_lid_correct + boundary TYPES are exact integers, bit-exact everywhere
//! (the argmax that fills the confusion is robust: the trained net's mean speech posterior on
//! the first frames is ~0.97-0.99, far from the 0.5 flip point, so it never flips off-env).
//!
//! NON-VACUITY (proven in-test): the `>150` in-band sentinel (`targetLID(target) = -2.0` ->
//! `100*langID + 200 >= 200`) is present at every file's target index, and the confusion has
//! OFF-DIAGONAL mass on `f1`/`f3` (target class != argmax language) vs a DIAGONAL hit on `f2`
//! (target 1 == argmax 1) -- the relabeled corpus guarantees differing targets across files.

mod common;

use std::path::PathBuf;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::audio::{Audio, read_audio};
use speech::io::binary::read_matrix;
use speech::tasks::lid::BlstmSpectralLid;
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmenter::Segmenter;

/// (file basename, `lang_index`) for the T2 corpus. `f3`'s class 2 clamps to 0 (classNb 2).
const FILES: [(&str, i32); 3] = [("f1", 0), ("f2", 1), ("f3", 2)];

fn lid5_map() -> IndexMap<String, String> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4b/lid5.config");
    speech::legacy_config::parse_legacy_config(&std::fs::read_to_string(p).unwrap())
}

fn real_weights() -> Vec<f64> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/NNweights_config1.bin");
    let (rows, cols, data) = read_matrix(&p).unwrap();
    assert_eq!(rows * cols, 33_671, "real net weight count");
    data
}

fn corpus_audio(name: &str, lang_index: i32) -> Audio {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4b/corpus_lid")
        .join(format!("{name}.wav"));
    let mut a = read_audio(&p, 0.35, 2.0, 0).expect("decode corpus wav");
    a.lang_index = lang_index;
    a
}

fn fresh_segs(audio: &Audio) -> Vec<Segmentation> {
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect()
}

/// Run the driver over one corpus file; returns (driver, post-`get_segmentation` segments).
fn run_file(name: &str, lang: i32) -> (BlstmSpectralLid, Vec<Segmentation>) {
    let w = real_weights();
    let mut audio = corpus_audio(name, lang);
    let mut segs = fresh_segs(&audio);
    let mut drv = BlstmSpectralLid::from_legacy(&lid5_map(), Some(&w)).unwrap();
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    (drv, segs)
}

fn row(v: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((1, v.len()), v.to_vec()).unwrap()
}

// === bit-exact vs the LidProbe reimpl dumps ==================================

#[test]
fn ltsv_sad_row_matches_dump() {
    // The PRE-convolution LTSV-SAD result_vec (`last_result_rows`) -- the NN-free spectral-
    // variation score over the raw periodogram -- vs the harness dump. Canary-gated (the
    // periodogram windowing is a cos/libm chain).
    for (name, lang) in FILES {
        let (drv, _) = run_file(name, lang);
        for chan in 0..drv.last_result_rows().len() {
            let want = common::load_bin_phase4b(&format!("lid5_{name}_ltsv_chan{}.bin", chan + 1));
            let got = row(&drv.last_result_rows()[chan]);
            common::assert_oracle_eq(&got, &want, &format!("{name} LTSV row chan{}", chan + 1));
        }
    }
}

#[test]
fn lid_classification_errors_match_dump() {
    // `100 * (langID - targetLID)` -- the softmax-averaged posterior offset by the -2.0
    // target sentinel. Canary-gated (softmax chain).
    for (name, lang) in FILES {
        let (drv, _) = run_file(name, lang);
        for chan in 0..drv.lid_classification_errors().len() {
            let want =
                common::load_bin_phase4b(&format!("lid5_{name}_liderr_chan{}.bin", chan + 1));
            let got = row(&drv.lid_classification_errors()[chan]);
            common::assert_oracle_eq(&got, &want, &format!("{name} liderr chan{}", chan + 1));
        }
    }
}

#[test]
fn lid_confusion_matches_dump() {
    // The per-file confusion matrix: label header + integer counts. Bit-exact on every
    // platform (the argmax that fills it is robust; the counts are exact integers).
    for (name, lang) in FILES {
        let (drv, _) = run_file(name, lang);
        for chan in 0..drv.lid_segments_confusion().len() {
            let want =
                common::load_bin_phase4b(&format!("lid5_{name}_confusion_chan{}.bin", chan + 1));
            let got = &drv.lid_segments_confusion()[chan];
            common::assert_bits_eq(got, &want, &format!("{name} confusion chan{}", chan + 1));
        }
    }
}

#[test]
fn lid_members_match_dump() {
    // `members = [cumulative_error, nb_of_classif, is_lid_correct]`. cumulative_error is a
    // LogLaw chain (canary-gated); the other two are exact integers.
    for (name, lang) in FILES {
        let (drv, _) = run_file(name, lang);
        for chan in 0..drv.lid_cumulative_error().len() {
            let mem =
                common::load_bin_phase4b(&format!("lid5_{name}_members_chan{}.bin", chan + 1));
            common::assert_oracle_eq_f64(
                drv.lid_cumulative_error()[chan],
                mem[[0, 0]],
                &format!("{name} cumulative_error chan{}", chan + 1),
            );
            assert_eq!(
                drv.lid_nb_of_classif()[chan],
                mem[[0, 1]] as i64,
                "{name} nb_of_classif chan{}",
                chan + 1
            );
            assert_eq!(
                drv.is_lid_correct()[chan],
                mem[[0, 2]] as i32,
                "{name} is_lid_correct chan{}",
                chan + 1
            );
        }
    }
}

#[test]
fn sad_boundaries_match_dump() {
    // The LTSV-SAD segmentation (begin times canary-gated, types exact).
    for (name, lang) in FILES {
        let (_, segs) = run_file(name, lang);
        for (chan, seg) in segs.iter().enumerate() {
            let want =
                common::load_bin_phase4b(&format!("lid5_{name}_boundaries_chan{}.bin", chan + 1));
            let got = seg.segments();
            assert_eq!(
                got.len(),
                want.nrows(),
                "{name} boundary count chan{}",
                chan + 1
            );
            for (i, s) in got.iter().enumerate() {
                common::assert_oracle_eq_f64(
                    s.begin,
                    want[[i, 0]],
                    &format!("{name} boundary[{i}].begin chan{}", chan + 1),
                );
                assert_eq!(
                    s.ty as i32,
                    want[[i, 1]] as i32,
                    "{name} boundary[{i}].type chan{}",
                    chan + 1
                );
            }
        }
    }
}

// === non-vacuity (proven in-test, no fixtures) ===============================

#[test]
fn sentinel_gt150_present_every_file() {
    // The `>150` in-band sentinel: `targetLID(target) = -2.0` -> `lid_classification_errors
    // [target] = 100*langID[target] + 200 >= 200 > 150`. Present at every file's target index.
    for (name, lang) in FILES {
        let (drv, _) = run_file(name, lang);
        for chan in 0..drv.lid_classification_errors().len() {
            let err = &drv.lid_classification_errors()[chan];
            assert!(
                err.iter().any(|&v| v > 150.0),
                "{name} chan{}: no >150 sentinel in {err:?}",
                chan + 1
            );
        }
    }
}

#[test]
fn confusion_off_diagonal_nonzero_and_targets_differ() {
    // The relabeled corpus guarantees differing targets: f1 (class 0) and f3 (class 2->0)
    // MISS (argmax language 1 != target 0) -> OFF-DIAGONAL mass; f2 (class 1) HITS
    // (argmax 1 == target 1) -> DIAGONAL mass. Proves the confusion is exercised, not seeded.
    let mut saw_off_diag = false;
    let mut saw_diag = false;
    for (name, lang) in FILES {
        let (drv, _) = run_file(name, lang);
        for chan in 0..drv.lid_segments_confusion().len() {
            let c = &drv.lid_segments_confusion()[chan];
            let class_nb = c.nrows() - 2;
            // Per-class rows/cols are 1..=class_nb; diagonal is (k,k), off-diagonal (k,l) k!=l.
            for k in 1..=class_nb {
                for l in 1..=class_nb {
                    if c[[k, l]] > 0.0 {
                        if k == l {
                            saw_diag = true;
                        } else {
                            saw_off_diag = true;
                        }
                    }
                }
            }
        }
    }
    assert!(
        saw_off_diag,
        "no off-diagonal confusion mass on any file (f1/f3 should MISS)"
    );
    assert!(
        saw_diag,
        "no diagonal confusion mass on any file (f2 should HIT)"
    );
}
