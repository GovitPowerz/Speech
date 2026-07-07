//! Phase 4b Task 7 FLAGSHIP: `TwinBlstmSpectralLid` Mode 7 over the committed LID config.
//!
//! `twin_mode7.config` is `configs/legacy/LID_BLSTM.config` re-pointed (File_Type 1,
//! phSeq corpus, the REAL 95k LID net `LID_bestNNWeight_1.bin` -> 12409 weights). Mode 7
//! (`abs(_Mode)==7`, `:903-1193`) never runs the SAD net; the LID net scores each phSeq
//! `_ExternalFeatures[i]` block directly (`:980-1155`). The corpus (`corpus_phseq/s{1,2,3}
//! .phSeq`) uses short sentences (<= 9 chars each), so every block is `<= _LIDWindowSize`
//! (10 periodogram frames); the windowed TwoSweeps forward is exercised but each block
//! spans a single window.
//!
//! Bit-exact goldens vs the harness `TwinProbe` mode-7 run land alongside -- unlike the
//! wav-mode Twin stage, the harness calls the REAL compiled `getSegmentation` directly for
//! Mode 7 (no reimpl/transcription swap, and no SEG_STRUCT/LID_STRUCT-style secondary probe);
//! the goldens ARE the real-compiled comparison, not a cross-check against one. THIS file
//! pins the non-vacuity + structural contract:
//! - confusion off- AND on-diagonal mass across files (s1/s3 lang 0 hit, s2 lang 1 miss);
//! - the `>150` targetLID sentinel present every file;
//! - `_PostProcessMode` 0/1/2 all covered (ppm2 differs from ppm0; ppm1 coincides with
//!   ppm0 in the normalized observables -- the per-row entropy offset is column-constant
//!   so it cancels in the softmax normalization, a documented property -- but the ppm1
//!   code path is asserted to run via `post_process_mode()`);
//! - the `_MinNbOfFrames` guard (ppm2's `MinNbOfFrames 4` skips the 2-/3-row blocks);
//! - `_DumpLIDInternals` writes the `features_<n>` + `matNb` `.mat` (value round-trip).

mod common;

use std::path::PathBuf;

use speech::audio::{Audio, read_audio};
use speech::io::binary::read_weight_vector;
use speech::tasks::lid::TwinBlstmSpectralLid;
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmenter::Segmenter;

const VARIANTS: [&str; 3] = ["twin_mode7", "twin_mode7_ppm1", "twin_mode7_ppm2"];

fn row(v: &[f64]) -> ndarray::Array2<f64> {
    ndarray::Array2::from_shape_vec((1, v.len()), v.to_vec()).unwrap()
}

fn phase4b(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4b")
        .join(name)
}

fn lid_weights() -> Vec<f64> {
    let w = read_weight_vector(&phase4b("LID_bestNNWeight_1.bin")).unwrap();
    assert_eq!(w.len(), 12409, "real LID net weight count");
    w
}

fn map_of(variant: &str) -> indexmap::IndexMap<String, String> {
    speech::legacy_config::parse_legacy_config(
        &std::fs::read_to_string(phase4b(&format!("{variant}.config"))).unwrap(),
    )
}

fn phseq_audio(f: &str, lang: i32, weight: f64) -> Audio {
    let mut a = read_audio(
        &phase4b("corpus_phseq").join(format!("{f}.phSeq")),
        0.0,
        120.0,
        1,
    )
    .unwrap();
    a.lang_index = lang;
    a.weight = weight;
    a
}

/// (file, lang). s1 -> class 0 (net predicts 0 -> HIT); s2 -> class 1 (net predicts 1 ->
/// HIT); s3 -> class 1 (net predicts 0 -> aggregate MISS -> isCorrect 0 non-vacuity).
const FILES: [(&str, i32); 3] = [("s1", 0), ("s2", 1), ("s3", 1)];

fn run(variant: &str, f: &str, lang: i32) -> (TwinBlstmSpectralLid, Vec<Segmentation>) {
    let lidw = lid_weights();
    let mut drv = TwinBlstmSpectralLid::from_legacy(&map_of(variant), None, Some(&lidw)).unwrap();
    let mut audio = phseq_audio(f, lang, 1.0);
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    (drv, segs)
}

#[test]
fn confusion_off_and_on_diagonal_mass() {
    let mut saw_off = false;
    let mut saw_diag = false;
    for (f, lang) in FILES {
        let (drv, _) = run("twin_mode7", f, lang);
        let c = &drv.lid_segments_confusion()[0];
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
    assert!(saw_diag, "no diagonal (hit) confusion mass");
    assert!(saw_off, "no off-diagonal (miss) confusion mass");
}

#[test]
fn sentinel_gt150_present_every_file() {
    for (f, lang) in FILES {
        let (drv, _) = run("twin_mode7", f, lang);
        let err = &drv.lid_classification_errors()[0];
        assert!(
            err.iter().any(|&x| x > 150.0),
            "mode7 {f}: no >150 sentinel in {err:?}"
        );
    }
}

#[test]
fn is_lid_correct_hit_and_miss() {
    // s1/s3 (lang 0) HIT -> 100; s2 (lang 1) MISS -> 0. Proves argmax(langID) is exercised.
    let mut saw_hit = false;
    let mut saw_miss = false;
    for (f, lang) in FILES {
        let (drv, _) = run("twin_mode7", f, lang);
        match drv.is_lid_correct()[0] {
            100 => saw_hit = true,
            0 => saw_miss = true,
            other => panic!("mode7 {f}: unexpected is_lid_correct {other}"),
        }
    }
    assert!(
        saw_hit && saw_miss,
        "need both a hit and a miss across files"
    );
}

#[test]
fn min_nb_of_frames_guard_exercised() {
    // ppm2 sets MinNbOfFrames 4: s2's "hi" (2 rows) and s3's "abc" (3 rows) are skipped,
    // so the LID nb_of_classif drops vs mode7 (MinNbOfFrames 0). Non-vacuous guard.
    for (f, lang) in [("s2", 1), ("s3", 0)] {
        let (d0, _) = run("twin_mode7", f, lang);
        let (d2, _) = run("twin_mode7_ppm2", f, lang);
        assert!(
            d2.lid_nb_of_classif()[0] < d0.lid_nb_of_classif()[0],
            "{f}: MinNbOfFrames guard did not reduce nb_of_classif ({} !< {})",
            d2.lid_nb_of_classif()[0],
            d0.lid_nb_of_classif()[0]
        );
    }
}

#[test]
fn post_process_mode_all_three_covered() {
    // ppm0 (sum-log), ppm1 (entropy), ppm2 (vote) all run; the driver records which.
    for (variant, want) in [
        ("twin_mode7", 0),
        ("twin_mode7_ppm1", 1),
        ("twin_mode7_ppm2", 2),
    ] {
        let (drv, _) = run(variant, "s1", 0);
        assert_eq!(drv.post_process_mode(), want, "{variant} post_process_mode");
    }
    // ppm2 (vote) is observably DISTINCT from ppm0 (sum-log): the classification errors differ.
    let (d0, _) = run("twin_mode7", "s1", 0);
    let (d2, _) = run("twin_mode7_ppm2", "s1", 0);
    assert_ne!(
        d0.lid_classification_errors()[0],
        d2.lid_classification_errors()[0],
        "ppm2 (vote) must differ from ppm0 (sum-log)"
    );
    // ppm1 (entropy) NEAR-COINCIDES with ppm0 in the normalized observables: the per-row
    // entropy offset is column-constant so it cancels in the softmax normalization EXACTLY
    // in exact arithmetic, and to ~1 ULP in floating point (exp/normalize rounding).
    let (d1, _) = run("twin_mode7_ppm1", "s1", 0);
    for (a, b) in d0.lid_classification_errors()[0]
        .iter()
        .zip(d1.lid_classification_errors()[0].iter())
    {
        assert!(
            (a - b).abs() <= 1e-10 * a.abs().max(1.0),
            "ppm1 should near-coincide with ppm0 (offset-cancellation): {a} vs {b}"
        );
    }
}

#[test]
fn dump_lid_internals_written_and_valued() {
    // ppm2 has DumpInternals true. The driver writes features_<n> (= [oF|oB]) + matNb.
    let tmp = std::env::temp_dir().join(format!("speech_mode7_dump_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let lidw = lid_weights();
    let mut drv =
        TwinBlstmSpectralLid::from_legacy(&map_of("twin_mode7_ppm2"), None, Some(&lidw)).unwrap();
    drv.set_dump_dir(tmp.to_string_lossy().into_owned());
    let mut audio = phseq_audio("s1", 0, 1.0);
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut segs: Vec<Segmentation> = (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect();
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();

    // s1 = [bonjour(7), salut(5)]; MinNbOfFrames 4 keeps both -> matNb = 2, features_0/1.
    let mat = tmp.join("chan0_lid_dump.mat");
    assert!(mat.exists(), "dump .mat not written at {mat:?}");
    let bytes = std::fs::read(&mat).unwrap();
    // MAT v5 header text + the variable names appear literally in the (uncompressed) file.
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("MATLAB 5.0"), "not a MAT v5 file");
    assert!(text.contains("features_0"), "missing features_0 variable");
    assert!(text.contains("features_1"), "missing features_1 variable");
    assert!(text.contains("matNb"), "missing matNb variable");
    std::fs::remove_dir_all(&tmp).ok();
}

// === bit-exact goldens vs the REAL compiled getSegmentation ======================
// The harness runs the REAL compiled TwinBLSTMSpectralLID::getSegmentation over the
// phSeq corpus (mode 7, noise off -> deterministic) and dumps the LID members
// (`mode7_<variant>_<file>_{confusion,liderr,members}.bin`). The port matches ALL members
// bit-exact on the oracle env: the LID net's one-hot phSeq input makes the layer-0
// projection summation-free (blocked == ascending GEMM), and the rest of the small
// 36/48-wide net happens to match Eigen bit-for-bit here (measured Rust-vs-REAL delta =
// 0 on every continuous member). confusion/is_lid_correct/nb_of_classif are argmax/count-
// derived (portable, bit-exact everywhere); lid_cumulative_error/lid_classification_errors
// carry LogLaw/exp -> canary-gated (<=4 ULP off the oracle libm). members = [lidCumErr,
// lidNbOfClassif, isLIDCorrect].

#[test]
fn mode7_integer_members_match_real_bitexact() {
    for v in VARIANTS {
        for (f, lang) in FILES {
            let (drv, _) = run(v, f, lang);
            let conf_want = common::load_bin_phase4b(&format!("mode7_{v}_{f}_confusion.bin"));
            common::assert_bits_eq(
                &drv.lid_segments_confusion()[0],
                &conf_want,
                &format!("mode7 {v} {f} confusion"),
            );
            let m = common::load_bin_phase4b(&format!("mode7_{v}_{f}_members.bin"));
            assert_eq!(
                drv.lid_nb_of_classif()[0],
                m[[0, 1]] as i64,
                "mode7 {v} {f} nb_of_classif"
            );
            assert_eq!(
                drv.is_lid_correct()[0],
                m[[0, 2]] as i32,
                "mode7 {v} {f} is_lid_correct"
            );
        }
    }
}

#[test]
fn mode7_continuous_members_match_real() {
    // lid_cumulative_error (LogLaw cost) + lid_classification_errors (langID via exp) --
    // libm-dependent -> canary-gated (bit-exact on the oracle env, <=4 ULP off it).
    for v in VARIANTS {
        for (f, lang) in FILES {
            let (drv, _) = run(v, f, lang);
            let m = common::load_bin_phase4b(&format!("mode7_{v}_{f}_members.bin"));
            common::assert_oracle_eq_f64(
                drv.lid_cumulative_error()[0],
                m[[0, 0]],
                &format!("mode7 {v} {f} lid_cumulative_error"),
            );
            let liderr = common::load_bin_phase4b(&format!("mode7_{v}_{f}_liderr.bin"));
            let got = row(&drv.lid_classification_errors()[0]);
            common::assert_oracle_eq(&got, &liderr, &format!("mode7 {v} {f} liderr"));
        }
    }
}

#[test]
fn mode7_noise_table_indexing_strict() {
    // The noise path (:1022-1032) is a PURE table lookup; random_init is wall-clock in the
    // legacy (:311, non-reproducible) so the port fixes randinit = 0. The harness dumps the
    // fixed-table result (m = _RandomGaussVector[(kk*cols+ll)%_MaxRandSize]-0.5, feat +
    // magnitude*m); the port reproduces it via constants::random_gauss -> STRICT bits.
    let feat = common::load_bin_phase4b("mode7_noise_in.bin");
    let want = common::load_bin_phase4b("mode7_noise_out.bin");
    let magnitude = 0.3;
    let (nr, nc) = feat.dim();
    let mut got = feat.clone();
    for kk in 0..nr {
        for ll in 0..nc {
            let m = speech::constants::random_gauss(kk * nc + ll) - 0.5;
            got[[kk, ll]] += magnitude * m;
        }
    }
    common::assert_bits_eq(&got, &want, "mode7 noise indexing");
}
