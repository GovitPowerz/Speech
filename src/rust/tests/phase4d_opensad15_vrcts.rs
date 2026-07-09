//! Phase 4d Task 5 cross-parse: proves the OpenSAD15 converter's emitted VRCTS XML
//! (`speech.dataprep.opensad15.convert_tab_file`) is actually consumable by the
//! engine's own `load_vrcts` -- the one piece of REAL (if narrow) validation this
//! Python-side, no-live-oracle module gets, beyond hand-transcription. Fixture is
//! shared with the Python suite (`tests/test_phase4d_opensad15.py`):
//! `tests/reference_data/phase4d/opensad15/case1_expected.xml`, hand-computed from
//! `ProcessOpenSAD15Corpus.py:23-108`, not dumped from any Python run.

use std::path::PathBuf;

use speech::tasks::segmentation::SegClass;
use speech::tasks::segmentation_io::load_vrcts;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4d/opensad15/case1_expected.xml")
}

#[test]
fn opensad15_case1_xml_parses_via_load_vrcts() {
    let xml =
        std::fs::read_to_string(fixture_path()).expect("case1_expected.xml must be committed");

    // sigdur="50.0" in the fixture; off=0 covers the whole file.
    let seg = load_vrcts(&xml, 0.0, 50.0);
    let segs = seg.segments();

    let speech: Vec<(f64, f64)> = (0..segs.len().saturating_sub(1))
        .filter(|&i| segs[i].ty == SegClass::Speech)
        .map(|i| (segs[i].begin, segs[i + 1].begin))
        .collect();

    // The 3 <SpeechSegment> rows hand-computed for case1 (see
    // tests/test_phase4d_opensad15.py::CASES / write_fixtures.py in the Task 5
    // report): [3.4118,9.7781], [20.0,25.0], [40.0,45.0].
    let expected = [(3.4118, 9.7781), (20.0, 25.0), (40.0, 45.0)];
    assert_eq!(
        speech.len(),
        expected.len(),
        "SPEECH segment count mismatch"
    );
    for (i, (got, want)) in speech.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9,
            "segment {i}: got {got:?}, want {want:?}"
        );
    }
}
