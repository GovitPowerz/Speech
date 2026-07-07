//! Phase 4b Task 6: `TwinBlstmSpectralLid` (Algo 6) driver goldens.
//!
//! `TwinBLSTMSpectralLID::getSegmentation` (`:263-1421`) runs the SAD BLSTM
//! (`_BLSTMNeuralNetwork`) for the VAD result_vec (modes 0/3) then a SECOND net
//! (`_LIDBLSTMNeuralNetwork`) per speech segment for language scoring (`:1243-1291`).
//! The harness `TwinProbe` stage transcribes the wav-mode 0/2/3 paths swapping ONLY the
//! SAD FFB (`:715`) + the LID scoring `feedForward` (`:1250`) for the reimpl family
//! (`blstmFeedForwardT6` for the real 33k SAD net + a generic small-topology LID reimpl),
//! keeping `getTargets`/`results2segmentation`/`LID2Segmentation`/`compute_errors` REAL;
//! the reimpl dumps are what this port matches. A harness SECONDARY real-Eigen probe
//! cross-checks SEG_STRUCT (segment count/type equality, `max_dt == 0`) + LID_STRUCT
//! (confusion + `_IsLIDCorrect` equality, `langid_max_abs ~ 1e-16`) against the REAL
//! compiled Twin; a structural mismatch aborts fixture generation.
//!
//! Four config variants over the T2 corpus (`f1/f2/f3.wav`, lang 0/1/2, weight
//! 0.5/0.75/1.0; the real 33,671-weight SAD net + a synthetic 3-class LID net):
//! - `mode0`: LID scores the SAD classification, LID input 11 == feature width -> NO
//!   concat (`concat_branch == 0`).
//! - `mode0_concat`: LID input 59 > 11 -> the `:1221` concat consumes the SAD hidden
//!   states (`concat_branch == 1`).
//! - `mode2`: LID scores the REFERENCE, SAD result_vec synthesized zero + outputs
//!   cleared, LID input 59 > 11 -> the concat is CALLED but takes the EMPTY fallback
//!   (`concat_branch == 2`).
//! - `mode3`: SAD BLSTM VAD + LID scores the REFERENCE (`concat_branch == 0`).
//!
//! Tolerance: the SAD result_vec, liderr, and cumulative_error are libm-dependent
//! (periodogram cos / softmax / LogLaw), so canary-gated. The confusion, nb_of_classif,
//! is_lid_correct, boundary TYPES, and concat_branch are exact integers, bit-exact
//! everywhere.

mod common;

use std::path::PathBuf;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::audio::{Audio, read_audio};
use speech::io::binary::read_matrix;
use speech::tasks::lid::TwinBlstmSpectralLid;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmenter::Segmenter;

struct Variant {
    name: &'static str,
    /// Expected `concat_branch`: 0 not-called, 1 concat, 2 empty-fallback.
    concat: i32,
    /// mode 2/3 -> a programmatic 2-span reference is fed as `refs`.
    set_ref: bool,
}

const VARIANTS: [Variant; 4] = [
    Variant {
        name: "mode0",
        concat: 0,
        set_ref: false,
    },
    Variant {
        name: "mode0_concat",
        concat: 1,
        set_ref: false,
    },
    Variant {
        name: "mode2",
        concat: 2,
        set_ref: true,
    },
    Variant {
        name: "mode3",
        concat: 0,
        set_ref: true,
    },
];

/// (file basename, lang_index, weight) for the T2 corpus.
const FILES: [(&str, i32, f64); 3] = [("f1", 0, 0.5), ("f2", 1, 0.75), ("f3", 2, 1.0)];

fn phase4b(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4b")
        .join(name)
}

fn variant_map(variant: &str) -> IndexMap<String, String> {
    let p = phase4b(&format!("twin_{variant}.config"));
    speech::legacy_config::parse_legacy_config(&std::fs::read_to_string(p).unwrap())
}

fn real_weights() -> Vec<f64> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/NNweights_config1.bin");
    let (rows, cols, data) = read_matrix(&p).unwrap();
    assert_eq!(rows * cols, 33_671, "real net weight count");
    data
}

fn lid_weights(variant: &str) -> Vec<f64> {
    let (_r, _c, data) = read_matrix(&phase4b(&format!("twin_{variant}_lidweights.bin"))).unwrap();
    data
}

fn corpus_audio(name: &str, lang: i32, weight: f64) -> Audio {
    let p = phase4b("corpus_lid").join(format!("{name}.wav"));
    let mut a = read_audio(&p, 0.35, 2.0, 0).expect("decode corpus wav");
    a.lang_index = lang;
    a.weight = weight;
    a
}

fn fresh_segs(audio: &Audio) -> Vec<Segmentation> {
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect()
}

/// The two-span programmatic reference the harness sets (`setTwinReference`).
fn build_refs(audio: &Audio) -> Vec<Segmentation> {
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|_| {
            let mut s = Segmentation::new(dur);
            s.label_segment(0.4, 0.9, SegClass::Speech);
            s.label_segment(1.2, 1.6, SegClass::Speech);
            s
        })
        .collect()
}

fn run_file(
    v: &Variant,
    file: &str,
    lang: i32,
    weight: f64,
) -> (TwinBlstmSpectralLid, Vec<Segmentation>) {
    let sad_w = real_weights();
    let lid_w = lid_weights(v.name);
    let mut drv =
        TwinBlstmSpectralLid::from_legacy(&variant_map(v.name), Some(&sad_w), Some(&lid_w))
            .unwrap();
    let mut audio = corpus_audio(file, lang, weight);
    let mut segs = fresh_segs(&audio);
    let refs = if v.set_ref {
        Some(build_refs(&audio))
    } else {
        None
    };
    drv.get_segmentation(&mut audio, &mut segs, refs.as_deref())
        .unwrap();
    (drv, segs)
}

fn row(v: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((1, v.len()), v.to_vec()).unwrap()
}

// === bit-exact vs the TwinProbe reimpl dumps ================================

#[test]
fn sad_result_row_matches_dump() {
    for v in &VARIANTS {
        for (file, lang, w) in FILES {
            let (drv, _) = run_file(v, file, lang, w);
            for chan in 0..drv.last_result_rows().len() {
                let want = common::load_bin_phase4b(&format!(
                    "twin_{}_{file}_result_chan{}.bin",
                    v.name,
                    chan + 1
                ));
                let got = row(&drv.last_result_rows()[chan]);
                common::assert_oracle_eq(
                    &got,
                    &want,
                    &format!("{} {file} result chan{}", v.name, chan + 1),
                );
            }
        }
    }
}

#[test]
fn lid_classification_errors_match_dump() {
    for v in &VARIANTS {
        for (file, lang, w) in FILES {
            let (drv, _) = run_file(v, file, lang, w);
            for chan in 0..drv.lid_classification_errors().len() {
                let want = common::load_bin_phase4b(&format!(
                    "twin_{}_{file}_liderr_chan{}.bin",
                    v.name,
                    chan + 1
                ));
                let got = row(&drv.lid_classification_errors()[chan]);
                common::assert_oracle_eq(
                    &got,
                    &want,
                    &format!("{} {file} liderr chan{}", v.name, chan + 1),
                );
            }
        }
    }
}

#[test]
fn lid_confusion_matches_dump() {
    for v in &VARIANTS {
        for (file, lang, w) in FILES {
            let (drv, _) = run_file(v, file, lang, w);
            for chan in 0..drv.lid_segments_confusion().len() {
                let want = common::load_bin_phase4b(&format!(
                    "twin_{}_{file}_confusion_chan{}.bin",
                    v.name,
                    chan + 1
                ));
                let got = &drv.lid_segments_confusion()[chan];
                common::assert_bits_eq(
                    got,
                    &want,
                    &format!("{} {file} confusion chan{}", v.name, chan + 1),
                );
            }
        }
    }
}

#[test]
fn members_match_dump() {
    // members = [lid_cumulative_error, lid_nb_of_classif, is_lid_correct,
    //            cumulative_error (SAD, *=LIDCostPonderation), nb_of_classif (SAD)].
    for v in &VARIANTS {
        for (file, lang, w) in FILES {
            let (drv, _) = run_file(v, file, lang, w);
            for chan in 0..drv.lid_cumulative_error().len() {
                let m = common::load_bin_phase4b(&format!(
                    "twin_{}_{file}_members_chan{}.bin",
                    v.name,
                    chan + 1
                ));
                common::assert_oracle_eq_f64(
                    drv.lid_cumulative_error()[chan],
                    m[[0, 0]],
                    &format!("{} {file} lid_cumulative_error chan{}", v.name, chan + 1),
                );
                assert_eq!(
                    drv.lid_nb_of_classif()[chan],
                    m[[0, 1]] as i64,
                    "{} {file} lid_nb_of_classif chan{}",
                    v.name,
                    chan + 1
                );
                assert_eq!(
                    drv.is_lid_correct()[chan],
                    m[[0, 2]] as i32,
                    "{} {file} is_lid_correct chan{}",
                    v.name,
                    chan + 1
                );
                common::assert_oracle_eq_f64(
                    drv.cumulative_error()[chan],
                    m[[0, 3]],
                    &format!("{} {file} cumulative_error chan{}", v.name, chan + 1),
                );
                assert_eq!(
                    drv.nb_of_classif()[chan],
                    m[[0, 4]] as i64,
                    "{} {file} nb_of_classif chan{}",
                    v.name,
                    chan + 1
                );
            }
        }
    }
}

#[test]
fn boundaries_match_dump() {
    for v in &VARIANTS {
        for (file, lang, w) in FILES {
            let (_, segs) = run_file(v, file, lang, w);
            for (chan, seg) in segs.iter().enumerate() {
                let want = common::load_bin_phase4b(&format!(
                    "twin_{}_{file}_boundaries_chan{}.bin",
                    v.name,
                    chan + 1
                ));
                let got = seg.segments();
                assert_eq!(
                    got.len(),
                    want.nrows(),
                    "{} {file} boundary count chan{}",
                    v.name,
                    chan + 1
                );
                for (i, s) in got.iter().enumerate() {
                    common::assert_oracle_eq_f64(
                        s.begin,
                        want[[i, 0]],
                        &format!("{} {file} boundary[{i}].begin chan{}", v.name, chan + 1),
                    );
                    assert_eq!(
                        s.ty as i32,
                        want[[i, 1]] as i32,
                        "{} {file} boundary[{i}].type chan{}",
                        v.name,
                        chan + 1
                    );
                }
            }
        }
    }
}

// === non-vacuity =============================================================

#[test]
fn concat_branch_matches_expected() {
    // The concat non-vacuity: mode0 not-called (0), mode0_concat concat (1), mode2
    // empty-fallback (2), mode3 not-called (0). Proven per channel against the driver
    // observation counter (the harness LID_STRUCT `concat=` matches).
    for v in &VARIANTS {
        for (file, lang, w) in FILES {
            let (drv, _) = run_file(v, file, lang, w);
            for (chan, &b) in drv.concat_branch().iter().enumerate() {
                assert_eq!(
                    b,
                    v.concat,
                    "{} {file} concat_branch chan{}",
                    v.name,
                    chan + 1
                );
            }
        }
    }
}

#[test]
fn sentinel_gt150_present_every_file() {
    // targetLID(target) = -2.0 -> lid_classification_errors[target] = 100*langID + 200 >= 200 > 150.
    for v in &VARIANTS {
        for (file, lang, w) in FILES {
            let (drv, _) = run_file(v, file, lang, w);
            for chan in 0..drv.lid_classification_errors().len() {
                let err = &drv.lid_classification_errors()[chan];
                assert!(
                    err.iter().any(|&x| x > 150.0),
                    "{} {file} chan{}: no >150 sentinel in {err:?}",
                    v.name,
                    chan + 1
                );
            }
        }
    }
}

#[test]
fn confusion_off_diagonal_and_diagonal_mass() {
    // The relabeled corpus guarantees differing targets: f1 (class 0) HITS -> DIAGONAL;
    // f2/f3 MISS -> OFF-DIAGONAL. Proves the confusion is exercised, not seeded.
    let mut saw_off = false;
    let mut saw_diag = false;
    for v in &VARIANTS {
        for (file, lang, w) in FILES {
            let (drv, _) = run_file(v, file, lang, w);
            for chan in 0..drv.lid_segments_confusion().len() {
                let c = &drv.lid_segments_confusion()[chan];
                let cn = c.nrows() - 2;
                for k in 1..=cn {
                    for l in 1..=cn {
                        if c[[k, l]] > 0.0 {
                            if k == l {
                                saw_diag = true;
                            } else {
                                saw_off = true;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(saw_off, "no off-diagonal confusion mass on any file");
    assert!(saw_diag, "no diagonal confusion mass on any file");
}

#[test]
fn mode2_and_mode3_differ_from_mode0() {
    // Reference-driven modes score the REFERENCE spans (0.4-0.9, 1.2-1.6) instead of the
    // SAD-detected classification, and (mode != 0) overwrite the classification via
    // LID2Segmentation -> the boundaries + langID differ from mode0 on the same corpus.
    let m0 = &VARIANTS[0];
    for other in [&VARIANTS[2], &VARIANTS[3]] {
        let mut any_diff = false;
        for (file, lang, w) in FILES {
            let (drv0, segs0) = run_file(m0, file, lang, w);
            let (drvo, segso) = run_file(other, file, lang, w);
            for chan in 0..segs0.len() {
                if segs0[chan].segments() != segso[chan].segments() {
                    any_diff = true;
                }
                if drv0.lid_classification_errors()[chan] != drvo.lid_classification_errors()[chan]
                {
                    any_diff = true;
                }
            }
        }
        assert!(
            any_diff,
            "{} did not differ from mode0 (reference-driven behavior not exercised)",
            other.name
        );
    }
}
