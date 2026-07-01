//! VRCTS XML load-correctness (against the 3 external `vrcts_part` fixtures) +
//! a constructed-input 4-decimal writer golden.
//!
//! The committed fixtures are external `vrcts_part` reference output (3-decimal
//! times), consumed by the engine via `load_from_vrcts`. The engine's own
//! writer (`Segmentation::toFile_VRCTS`) emits 4-decimal `stime`/`etime`, which
//! no fixture exercises -- hence a separate constructed-input writer golden. We
//! do NOT round-trip a fixture through the writer (ill-posed: 3-decimal input
//! vs 4-decimal writer output).

use std::path::PathBuf;

use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmentation_io::{load_vrcts, to_vrcts_string, write_vrcts};

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0bii")
}

/// Extract the `[begin, end)` spans of SPEECH segments from a loaded boundary
/// list: a SPEECH boundary runs until the next boundary's begin.
fn speech_spans(seg: &Segmentation) -> Vec<(f64, f64)> {
    let segs = seg.segments();
    let mut spans = Vec::new();
    for i in 0..segs.len().saturating_sub(1) {
        if segs[i].ty == SegClass::Speech {
            spans.push((segs[i].begin, segs[i + 1].begin));
        }
    }
    spans
}

fn load_fixture(fixture: &str) -> Segmentation {
    let raw = std::fs::read_to_string(ref_dir().join(fixture)).unwrap();
    load_vrcts(&raw, 0.0, 1.0e9)
}

const EPS: f64 = 1.0e-9;

fn assert_spans(actual: &[(f64, f64)], expected: &[(f64, f64)]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "SPEECH segment count mismatch"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a.0 - e.0).abs() < EPS && (a.1 - e.1).abs() < EPS,
            "span {i} mismatch: got {a:?}, want {e:?}"
        );
    }
}

#[test]
fn load_1seg() {
    let seg = load_fixture("vrcts_1seg.xml");
    assert!((seg.audio_duration() - 12.0).abs() < EPS);
    assert_spans(&speech_spans(&seg), &[(0.000, 11.999)]);
}

#[test]
fn load_empty() {
    let seg = load_fixture("vrcts_empty.xml");
    assert!((seg.audio_duration() - 120.0).abs() < EPS);
    assert_spans(&speech_spans(&seg), &[]);
}

#[test]
fn load_16seg() {
    // The 16 stime/etime pairs, read verbatim from vrcts_16seg.xml.
    let expected: [(f64, f64); 16] = [
        (2.180, 3.883),
        (4.680, 13.384),
        (14.280, 23.384),
        (25.580, 28.234),
        (30.931, 32.684),
        (35.780, 40.581),
        (43.780, 60.334),
        (62.630, 66.334),
        (68.130, 75.884),
        (76.480, 80.234),
        (81.181, 84.784),
        (85.931, 92.234),
        (102.030, 104.884),
        (108.280, 111.134),
        (111.980, 115.234),
        (117.580, 119.999),
    ];
    let seg = load_fixture("vrcts_16seg.xml");
    assert!((seg.audio_duration() - 120.0).abs() < EPS);
    assert_spans(&speech_spans(&seg), &expected);
}

/// Constructed-input writer golden: pins the engine's 4-decimal `stime`/`etime`
/// format (`toFile_VRCTS` `%f.4s`) that no fixture covers. Begins are already
/// clean on the 1e-4 grid, so `{:.4}` display is exact.
#[test]
fn write_golden_4decimal() {
    let mut seg = Segmentation::new(10.0);
    seg.label_segment(1.2345, 3.6789, SegClass::Speech);
    seg.label_segment(5.0, 7.5, SegClass::Speech);

    // speech_duration = (3.6789 - 1.2345) + (7.5 - 5.0) = 2.4444 + 2.5 = 4.9444 -> "4.94"
    let expected = concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<AudioDoc name=\"CONSTRUCTED\" path=\"/tmp/constructed.wav\">\n",
        "<ProcList>\n",
        "<Proc name=\"vrcts_part\" version=\"1.3\"/>\n",
        "</ProcList>\n",
        "<ChannelList>\n",
        "<Channel num=\"1\" sigdur=\"10.00\" spdur=\"4.94\"/>\n",
        "</ChannelList>\n",
        "<SpeakerList>\n",
        "<Speaker ch=\"1\" dur=\"4.94\" gender=\"1\" spkid=\"1\"/>\n",
        "</SpeakerList>\n",
        "<SegmentList>\n",
        "<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"1.2345\" etime=\"3.6789\" spkid=\"1\"/>\n",
        "<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"5.0000\" etime=\"7.5000\" spkid=\"1\"/>\n",
        "</SegmentList>\n",
        "</AudioDoc>\n",
    );

    let out = to_vrcts_string(&seg, "CONSTRUCTED", "/tmp/constructed.wav");
    assert_eq!(out, expected);
}

/// `write_vrcts` writes exactly what `to_vrcts_string` produces.
#[test]
fn write_file_matches_string() {
    let mut seg = Segmentation::new(10.0);
    seg.label_segment(1.2345, 3.6789, SegClass::Speech);
    seg.label_segment(5.0, 7.5, SegClass::Speech);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.xml");
    write_vrcts(&seg, "CONSTRUCTED", "/tmp/constructed.wav", &path).unwrap();

    let on_disk = std::fs::read_to_string(&path).unwrap();
    let in_mem = to_vrcts_string(&seg, "CONSTRUCTED", "/tmp/constructed.wav");
    assert_eq!(on_disk, in_mem);
}
