//! Phase 4b Task 5: phSeq reader (`File_Type 1`) goldens -- `AudioStruct.cpp:138-182`.
//!
//! `read_audio(path, offset, duration, file_type=1)` reads a text file where each
//! line is a "sentence": a one-hot `(len x 38)` matrix keyed by `letterMapping`
//! (`AudioStruct.h:18`). `_ChannelsCount = 1`, `_Framerate = 8000` (hardcoded).
//! `number_of_phonemes` accumulates `len+10` per sentence (10 initial); `_Periodogram`
//! is `(number_of_phonemes x 38)` zeroed then block-filled per sentence starting at
//! row 5, stepping `len+10` rows.
//!
//! Fixtures (`tests/reference_data/phase4b/corpus_phseq/{f1,f2,f3}.phSeq`) are
//! SYNTHETIC -- no real `.phSeq` file exists anywhere to validate against, so these
//! ARE the format contract (manifest-documented, `manifest.json:phseq`). Characters
//! are drawn only from `letterMapping`'s domain. `f1` has a middle BLANK line
//! (0-length sentence), exercising the `len==0` edge of the `+10` gap arithmetic;
//! `f2` covers a space-containing sentence plus the `.`/`-` special chars; `f3` is a
//! single 25-char sentence (every lowercase letter except `q`, absent from
//! `letterMapping`).
//!
//! Goldens (`phseq_<f>_feat<i>.bin`, `phseq_<f>_periodogram.bin`) are from the REAL
//! compiled `AudioStruct` `file_type==1` ctor (oracle harness, no reimpl/probe
//! needed -- `_ExternalFeatures`/`_Periodogram` are public members). STRICT BITS
//! everywhere: this branch is pure indexing, no libm.

mod common;

use std::path::PathBuf;

use speech::audio::read_audio;

const FILES: [&str; 3] = ["f1", "f2", "f3"];

fn phseq_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4b/corpus_phseq")
        .join(format!("{name}.phSeq"))
}

/// `common::load_bin_phase4b` delegates to `io::binary::read_matrix`, which
/// deliberately rejects `rows<=0` (a Phase 0a-locked guard against corrupt/
/// uninitialized weight files -- not something this test may weaken). f1's
/// 0-length middle line legitimately dumps a `(0, 38)` feat matrix, so for THAT
/// one fixture this reads just the `(rows, cols)` header directly, bypassing
/// the guard, instead of the payload comparison (there is no payload to compare).
fn phseq_bin_dims(name: &str) -> (i64, i64) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4b")
        .join(name);
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rows = i64::from_le_bytes(raw[0..8].try_into().unwrap());
    let cols = i64::from_le_bytes(raw[8..16].try_into().unwrap());
    (rows, cols)
}

// === bit-exact vs the real-compiled AudioStruct dumps ========================

#[test]
fn phseq_onehot_golden() {
    for name in FILES {
        let audio = read_audio(&phseq_path(name), 0.0, 0.0, 1).expect("decode phSeq");

        let want_p = common::load_bin_phase4b(&format!("phseq_{name}_periodogram.bin"));
        let got_p = audio
            .periodogram
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: periodogram must be Some for file_type 1"));
        common::assert_bits_eq(got_p, &want_p, &format!("{name} periodogram"));

        for (i, feat) in audio.external_features.iter().enumerate() {
            let bin_name = format!("phseq_{name}_feat{i}.bin");
            if feat.nrows() == 0 {
                assert_eq!(
                    phseq_bin_dims(&bin_name),
                    (0, 38),
                    "{name} feat{i}: harness dump shape must also be (0, 38)"
                );
                continue;
            }
            let want_feat = common::load_bin_phase4b(&bin_name);
            common::assert_bits_eq(feat, &want_feat, &format!("{name} feat{i}"));
        }
    }
}

#[test]
fn phseq_metadata_matches_manifest() {
    // Cross-check the scalar metadata the manifest measured independently (Python
    // reimplementation, not the harness) -- (lines, number_of_phonemes, frames_count)
    // per fixture file.
    let expect: [(&str, usize, i64, usize); 3] = [
        ("f1", 3, 52, 4160),
        ("f2", 2, 46, 3680),
        ("f3", 1, 45, 3600),
    ];
    for (name, lines, phonemes, frames) in expect {
        let audio = read_audio(&phseq_path(name), 0.0, 0.0, 1).expect("decode phSeq");
        assert_eq!(audio.external_features.len(), lines, "{name} line count");
        assert_eq!(
            audio.periodogram.as_ref().unwrap().nrows(),
            phonemes as usize,
            "{name} number_of_phonemes (periodogram rows)"
        );
        assert_eq!(
            audio.periodogram.as_ref().unwrap().ncols(),
            38,
            "{name} periodogram cols"
        );
        assert_eq!(
            audio.data.dim(),
            (1, frames),
            "{name} frames_count / data shape"
        );
        assert_eq!(audio.data_raw.dim(), (1, frames), "{name} data_raw shape");
        assert!(
            audio.data.iter().all(|&v| v == 0.0),
            "{name}: phSeq data must be all-zero (no waveform)"
        );
        assert_eq!(audio.sample_rate, 8000, "{name} framerate hardcoded");
    }
}

#[test]
fn phseq_zero_length_line_is_a_zero_row_block() {
    // f1's middle blank line -> a 0-row one-hot matrix, and a 10-row (not 6+10) gap
    // in the periodogram between sentence 1 and sentence 3.
    let audio = read_audio(&phseq_path("f1"), 0.0, 0.0, 1).expect("decode phSeq");
    assert_eq!(audio.external_features[1].dim(), (0, 38));
    let p = audio.periodogram.unwrap();
    // sentence1 "bozawa" occupies rows 5..11; the 0-length sentence2 contributes no
    // rows, so sentence3 "SIN@&H" starts at 21+10=31, not 11+10=21.
    assert!(
        p.row(11).iter().all(|&v| v == 0.0),
        "gap row 11 must be zero"
    );
    assert!(
        p.row(20).iter().all(|&v| v == 0.0),
        "gap row 20 must be zero"
    );
    assert!(
        p.row(31).iter().any(|&v| v != 0.0),
        "sentence3 must start at row 31"
    );
}

// === frames_count arithmetic (mutation-catching, no fixture needed) ==========

#[test]
fn frames_count_arithmetic() {
    // legacy AudioStruct.cpp:173: `_FramesCount = numberOfPhonemes*0.01*_Framerate`
    // is LEFT-ASSOCIATIVE (C++ `*` is left-to-right): `(numberOfPhonemes*0.01)*
    // framerate`. At numberOfPhonemes=803 (framerate 8000) this diverges from the
    // naive `numberOfPhonemes*(0.01*framerate)` regrouping by exactly 1 truncated
    // frame (64239 vs 64240) -- verified independently in Python:
    //   (803*0.01)*8000.0   == 64239.99999999999 -> truncates to 64239
    //   803*(0.01*8000.0)   == 64240.0            -> truncates to 64240
    // A single 783-char line (number_of_phonemes = 10 + 783 + 10 = 803) exercises
    // this directly through the public `read_audio` API -- no fixture commit
    // needed, and a right-associative regrouping "fix" would flip this assert.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.phSeq");
    let line: String = std::iter::repeat_n('a', 783).collect();
    std::fs::write(&path, format!("{line}\n")).unwrap();

    let audio = read_audio(&path, 0.0, 0.0, 1).expect("decode synthetic phSeq");
    assert_eq!(
        audio.periodogram.as_ref().unwrap().nrows(),
        803,
        "number_of_phonemes"
    );
    assert_eq!(
        audio.data.ncols(),
        64239,
        "frames_count must use the LEFT-associative truncation"
    );
    assert_ne!(
        audio.data.ncols(),
        64240,
        "must NOT match the naive right-associative regrouping"
    );
}

#[test]
fn out_of_domain_char_bails() {
    // 'q' is absent from letterMapping (the legacy `.at()` would throw
    // std::out_of_range, uncaught -> abort; this port turns that into a
    // recoverable Err instead of reproducing a crash -- see IMPROVEMENTS.md).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.phSeq");
    std::fs::write(&path, "aqz\n").unwrap();
    match read_audio(&path, 0.0, 0.0, 1) {
        Err(e) => assert!(
            e.to_string().contains('q'),
            "error should name the offending char: {e}"
        ),
        Ok(_) => panic!("'q' is not in letterMapping, decode must fail"),
    }
}

#[test]
fn missing_file_bails() {
    // legacy AudioStruct.cpp:149-152: ifstream open failure -> exit(1). Ported as
    // a recoverable Err (the crate never calls exit/abort on I/O failure).
    let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("does/not/exist.phSeq");
    assert!(read_audio(&missing, 0.0, 0.0, 1).is_err());
}

// === wav path regression guard (full suite guard) =============================

#[test]
fn wav_path_unchanged() {
    // Cheapest honest regression check that adding the phSeq branch didn't perturb
    // the existing file_type==0 dispatch: File_Type 0 (wav) must still match the
    // Phase 1-pinned sig_norm.bin golden, and the two new Audio fields must stay at
    // their wav-path defaults (empty / None).
    let audio = read_audio(&common::fixture("excerpt_2ch_8k.wav"), 0.35, 2.0, 0)
        .expect("decode excerpt wav");
    assert_eq!(audio.sample_rate, 8000);
    assert!(
        audio.external_features.is_empty(),
        "wav path: external_features must stay empty"
    );
    assert!(
        audio.periodogram.is_none(),
        "wav path: periodogram must stay None"
    );
    common::assert_bits_eq(
        &audio.data_raw,
        &common::load_bin("sig_norm.bin"),
        "sig_norm (wav path unperturbed by the phSeq branch)",
    );
}

#[test]
fn unsupported_file_type_bails() {
    let wav = common::fixture("excerpt_2ch_8k.wav");
    for ft in [2, 3, 4, -1] {
        match read_audio(&wav, 0.0, 0.1, ft) {
            Err(e) => assert!(e.to_string().contains("file_type")),
            Ok(_) => panic!("file_type {ft} must not be accepted (only 0/1 are ported)"),
        }
    }
}
