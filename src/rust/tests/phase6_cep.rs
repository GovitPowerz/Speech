//! Phase 6 Task 1: the `File_Type == 2` cep reader (`AudioStruct.cpp:183-256`).
//!
//! The `.plp8f0mvsdd` cepstral-feature binary the 2015 LID net trained on. Layout
//! (little-endian): `int32 nbRecords | int16 vectorSize | int16 magic`, then an
//! `int32 vectorNb`-per-record table, then `float32` payload row-major per record.
//! `magic` is ignored; records with `vectorSize*vectorNb <= 0` are skipped.
//!
//! Fixtures under `tests/reference_data/phase6/cep/` are HAND-CRAFTED synthetic bytes
//! (never copied from the licensed corpus) -- regenerate with
//! `cargo test -p speech --test phase6_cep -- --ignored regenerate_fixtures`.
//! The real-corpus consistency check (`corpus_first_file_consistency`) is gated on the
//! licensed `data/LRE03-LRE07` corpus and skips cleanly when it is absent.

mod common;

use std::path::{Path, PathBuf};

use ndarray::{Array2, arr2};
use speech::audio::{Audio, read_audio};

use common::{corpus_root_or_skip, fixture_phase6};

/// Serialize well-formed cep bytes: `nbRecords` from `records.len()`, each record's
/// `vectorNb` from its row count, `float32` payload row-major. Empty records (0 rows)
/// encode `vectorNb == 0` with no floats. Malformed fixtures are built by hand below.
fn write_cep(records: &[Vec<Vec<f32>>], vector_size: i16, magic: i16) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(records.len() as i32).to_le_bytes());
    b.extend_from_slice(&vector_size.to_le_bytes());
    b.extend_from_slice(&magic.to_le_bytes());
    for rec in records {
        b.extend_from_slice(&(rec.len() as i32).to_le_bytes());
    }
    for rec in records {
        for row in rec {
            assert_eq!(
                row.len(),
                vector_size as usize,
                "row width must equal vectorSize"
            );
            for &v in row {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    b
}

fn cep_path(name: &str) -> PathBuf {
    fixture_phase6(&format!("cep/{name}"))
}

fn read_cep_fixture(name: &str) -> Result<Audio, String> {
    // File_Type 2, generous max-duration (the cep branch never truncates by it).
    // Flatten to String so the test crate needn't name the engine's `anyhow` dep.
    read_audio(&cep_path(name), 0.0, 3.6e6, 2).map_err(|e| format!("{e:#}"))
}

// -- happy path -------------------------------------------------------------------

#[test]
fn tiny_ok_single_record() {
    let audio = read_cep_fixture("tiny_ok.plp").expect("tiny_ok.plp must read");
    // One record, a 2x3 matrix, read row-major.
    assert_eq!(audio.external_features.len(), 1);
    let expected = arr2(&[[1.0, -2.0, 0.5], [3.0, 4.25, -0.75]]);
    assert_eq!(audio.external_features[0], expected);

    // numberOfFrames = 2 + (2 + 2) = 6; frames_count = (6 * 0.01) * 8000 = 480.
    assert_eq!(audio.sample_rate, 8000);
    assert_eq!(audio.data.dim(), (1, 480));
    assert_eq!(audio.data_raw.dim(), (1, 480));

    // Periodogram (6 x 3): rowBegin = 1, so the 2x3 block lands in rows 1..3, the rest
    // stays zero (a 1-row lead margin + trailing margin, :250-255).
    let peri = audio
        .periodogram
        .as_ref()
        .expect("cep fills the periodogram");
    assert_eq!(peri.dim(), (6, 3));
    assert_eq!(peri.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
    assert_eq!(peri.slice(ndarray::s![1..3, ..]).to_owned(), expected);
    assert_eq!(peri.row(5).to_vec(), vec![0.0, 0.0, 0.0]);
}

#[test]
fn multi_record_magic_ignored_empty_skipped() {
    // nbRecords = 3, vectorNb = [1, 0, 2], magic = 7. The empty middle record is
    // dropped; a non-zero magic must NOT affect the read (magic is ignored).
    let audio = read_cep_fixture("multi_ok.plp").expect("multi_ok.plp must read");
    assert_eq!(
        audio.external_features.len(),
        2,
        "the vectorNb=0 record is skipped"
    );
    assert_eq!(audio.external_features[0], arr2(&[[10.0, 20.0]]));
    assert_eq!(
        audio.external_features[1],
        arr2(&[[-1.0, -2.0], [3.5, 4.0]])
    );

    // numberOfFrames = 2 + (1 + 2) + (2 + 2) = 9; frames_count = (9 * 0.01) * 8000 = 720.
    assert_eq!(audio.data.dim(), (1, 720));
    let peri = audio.periodogram.as_ref().unwrap();
    assert_eq!(peri.dim(), (9, 2));
    // rec0 (1 row) at rowBegin=1; rec1 (2 rows) at rowBegin = 1 + 1 + 2 = 4.
    assert_eq!(peri.row(1).to_vec(), vec![10.0, 20.0]);
    assert_eq!(
        peri.slice(ndarray::s![4..6, ..]).to_owned(),
        arr2(&[[-1.0, -2.0], [3.5, 4.0]])
    );
    assert_eq!(
        peri.row(3).to_vec(),
        vec![0.0, 0.0],
        "gap row between records stays zero"
    );
}

#[test]
fn write_read_roundtrip_arbitrary_matrix() {
    // A tighter in-language round-trip independent of the committed fixtures: any matrix
    // written by `write_cep` must read back element-for-element (row-major, no transpose).
    let records = vec![vec![
        vec![0.125_f32, -0.25, 8.0, -16.5, 100.0],
        vec![-1.5, 2.75, 0.0, 33.0, -0.5],
        vec![9.0, -9.0, 0.25, -0.125, 4.0],
    ]];
    let bytes = write_cep(&records, 5, 0);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rt.plp");
    std::fs::write(&path, &bytes).unwrap();

    let audio = read_audio(&path, 0.0, 3.6e6, 2).expect("round-trip cep must read");
    assert_eq!(audio.external_features.len(), 1);
    let expected: Array2<f64> = arr2(&[
        [0.125, -0.25, 8.0, -16.5, 100.0],
        [-1.5, 2.75, 0.0, 33.0, -0.5],
        [9.0, -9.0, 0.25, -0.125, 4.0],
    ]);
    assert_eq!(audio.external_features[0], expected);
}

// -- typed errors (PORT-TRUTH: no panic, no silent zero-fill) ----------------------

fn assert_err_contains(name: &str, needle: &str) {
    match read_cep_fixture(name) {
        Ok(_) => panic!("{name}: expected a typed error, got Ok"),
        Err(e) => {
            let s = e.to_string();
            assert!(
                s.contains(needle),
                "{name}: error {s:?} must contain {needle:?}"
            );
        }
    }
}

#[test]
fn zero_records_errors() {
    assert_err_contains("zero_records.plp", "empty");
}

#[test]
fn truncated_header_errors() {
    assert_err_contains("trunc_header.plp", "truncated header");
}

#[test]
fn truncated_table_errors() {
    assert_err_contains("trunc_table.plp", "truncated record table");
}

#[test]
fn truncated_payload_errors() {
    assert_err_contains("truncated.plp", "mismatch");
}

#[test]
fn excess_payload_errors() {
    assert_err_contains("excess.plp", "mismatch");
}

#[test]
fn bad_vectorsize_errors() {
    assert_err_contains("bad_vecsize.plp", "vectorSize");
}

// -- corpus-gated consistency (licensed data; skips if absent) ---------------------

/// Recursively collect `*.plp8f0mvsdd` paths under `dir` (skips `.DS_Store` etc.).
fn collect_cep_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_cep_files(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("plp8f0mvsdd") {
            out.push(p);
        }
    }
}

#[test]
fn corpus_first_file_consistency() {
    let Some(root) = corpus_root_or_skip() else {
        return;
    };
    let mut files = Vec::new();
    collect_cep_files(&root.join("train/LID_Features"), &mut files);
    if files.is_empty() {
        eprintln!("SKIP: no .plp8f0mvsdd under train/LID_Features");
        return;
    }
    files.sort();
    let first = &files[0]; // deterministic pick: lexicographically-first path.
    eprintln!("cep consistency: reading {}", first.display());

    // (1) Independent byte-arithmetic check (does NOT use read_cep): parse the header
    // directly and assert the file size equals 8 + 4*nbRecords + 4*sum(vectorNb*vectorSize).
    let raw = std::fs::read(first).unwrap();
    assert!(raw.len() >= 8, "real cep file has a header");
    let nb_records = i32::from_le_bytes(raw[0..4].try_into().unwrap());
    let vector_size = i16::from_le_bytes(raw[4..6].try_into().unwrap());
    assert!(nb_records > 0, "nbRecords > 0");
    assert_eq!(vector_size, 23, "LRE cep vectorSize == NNetInputSize (23)");
    let mut expected_floats: usize = 0;
    for m in 0..nb_records as usize {
        let vnb = i32::from_le_bytes(raw[8 + 4 * m..12 + 4 * m].try_into().unwrap());
        if vnb > 0 {
            expected_floats += vnb as usize * vector_size as usize;
        }
    }
    let expected_len = 8 + 4 * nb_records as usize + 4 * expected_floats;
    assert_eq!(
        raw.len(),
        expected_len,
        "file size must match header arithmetic exactly"
    );

    // (2) The reader itself must accept the file (its strict byte check passing IS a
    // second confirmation of the layout) and produce finite, plausible, non-degenerate
    // features whose implied duration lands in a sane LRE range.
    let audio = read_audio(first, 0.0, 3.6e6, 2).expect("real cep file must read");
    assert!(!audio.external_features.is_empty(), "at least one record");
    assert_eq!(audio.external_features[0].ncols(), 23, "feature dim 23");

    let mut total_rows = 0usize;
    let mut any_nonzero = false;
    let mut max_abs = 0.0_f64;
    for feat in &audio.external_features {
        assert_eq!(feat.ncols(), 23);
        total_rows += feat.nrows();
        for &v in feat.iter() {
            assert!(v.is_finite(), "feature values must be finite");
            if v != 0.0 {
                any_nonzero = true;
            }
            max_abs = max_abs.max(v.abs());
        }
    }
    assert!(any_nonzero, "features must not be all-zero");
    assert!(
        max_abs < 100.0,
        "standardized features have sane magnitude, got {max_abs}"
    );
    let duration_s = total_rows as f64 * 0.01; // 10 ms per frame at 8 kHz.
    assert!(
        (1.0..=120.0).contains(&duration_s),
        "implied duration {duration_s}s out of the sane LRE range"
    );
    eprintln!(
        "cep consistency OK: nbRecords={nb_records} vectorSize=23 frames={total_rows} duration={duration_s:.2}s max_abs={max_abs:.3}"
    );
}

// -- fixture regeneration (run manually; writes the committed synthetic bytes) ------

#[test]
#[ignore = "regenerates committed fixtures; run explicitly with --ignored"]
fn regenerate_fixtures() {
    let dir = cep_path("");
    std::fs::create_dir_all(&dir).unwrap();
    let put = |name: &str, bytes: Vec<u8>| std::fs::write(cep_path(name), bytes).unwrap();

    // Happy paths.
    put(
        "tiny_ok.plp",
        write_cep(&[vec![vec![1.0, -2.0, 0.5], vec![3.0, 4.25, -0.75]]], 3, 0),
    );
    put(
        "multi_ok.plp",
        write_cep(
            &[
                vec![vec![10.0, 20.0]],
                vec![], // empty record (vectorNb = 0) -> skipped on read
                vec![vec![-1.0, -2.0], vec![3.5, 4.0]],
            ],
            2,
            7, // non-zero magic: must be ignored
        ),
    );

    // zero_records: 8-byte header with nbRecords = 0.
    let mut zero = Vec::new();
    zero.extend_from_slice(&0i32.to_le_bytes());
    zero.extend_from_slice(&3i16.to_le_bytes());
    zero.extend_from_slice(&0i16.to_le_bytes());
    put("zero_records.plp", zero);

    // trunc_header: fewer than 8 bytes.
    put("trunc_header.plp", vec![1, 0, 0, 0, 3]);

    // trunc_table: nbRecords = 3 but only one vectorNb present (12 bytes, need 20).
    let mut tt = Vec::new();
    tt.extend_from_slice(&3i32.to_le_bytes());
    tt.extend_from_slice(&2i16.to_le_bytes());
    tt.extend_from_slice(&0i16.to_le_bytes());
    tt.extend_from_slice(&2i32.to_le_bytes());
    put("trunc_table.plp", tt);

    // truncated: vectorNb=[2], vectorSize=3 => expects 6 floats, provide 3 (short).
    let mut tr = Vec::new();
    tr.extend_from_slice(&1i32.to_le_bytes());
    tr.extend_from_slice(&3i16.to_le_bytes());
    tr.extend_from_slice(&0i16.to_le_bytes());
    tr.extend_from_slice(&2i32.to_le_bytes());
    for v in [1.0_f32, 2.0, 3.0] {
        tr.extend_from_slice(&v.to_le_bytes());
    }
    put("truncated.plp", tr);

    // excess: vectorNb=[2], vectorSize=2 => expects 4 floats, provide 6 (excess).
    let mut ex = Vec::new();
    ex.extend_from_slice(&1i32.to_le_bytes());
    ex.extend_from_slice(&2i16.to_le_bytes());
    ex.extend_from_slice(&0i16.to_le_bytes());
    ex.extend_from_slice(&2i32.to_le_bytes());
    for v in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        ex.extend_from_slice(&v.to_le_bytes());
    }
    put("excess.plp", ex);

    // bad_vecsize: vectorSize = 0 (non-positive) -> malformed.
    let mut bv = Vec::new();
    bv.extend_from_slice(&1i32.to_le_bytes());
    bv.extend_from_slice(&0i16.to_le_bytes());
    bv.extend_from_slice(&0i16.to_le_bytes());
    put("bad_vecsize.plp", bv);
}
