//! Segment container + primitives (sanitize, suppress-short, padding), scoring,
//! VRCTS XML + STM/TRS loaders.
//!
//! Ported from legacy C++: Segmentation.*. Phase 0.

use std::path::Path;

use anyhow::Result;

use super::segmentation::{SegClass, Segmentation};

/// Sum of `(next.begin - it.begin)` over SPEECH segments.
/// Direct port of the `speechDuration` accumulator in `Segmentation::toFile_VRCTS`
/// (`Segmentation.cpp:546-551`).
fn speech_duration(seg: &Segmentation) -> f64 {
    let segs = seg.segments();
    let mut total = 0.0f64;
    for i in 0..segs.len().saturating_sub(1) {
        if segs[i].ty == SegClass::Speech {
            total += segs[i + 1].begin - segs[i].begin;
        }
    }
    total
}

/// Serialize `seg` as a VRCTS XML document. Direct port of
/// `Segmentation::toFile_VRCTS` (`Segmentation.cpp:568-587`), single-channel
/// (`chan = 1`), offset folded to `0.0` (this container has no `_AudioOffset`
/// field).
pub fn to_vrcts_string(seg: &Segmentation, name: &str, path_attr: &str) -> String {
    let speech_dur = speech_duration(seg);
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<AudioDoc name=\"{name}\" path=\"{path_attr}\">\n"
    ));
    out.push_str("<ProcList>\n");
    out.push_str("<Proc name=\"vrcts_part\" version=\"1.3\"/>\n");
    out.push_str("</ProcList>\n");
    out.push_str("<ChannelList>\n");
    out.push_str(&format!(
        "<Channel num=\"1\" sigdur=\"{:.2}\" spdur=\"{:.2}\"/>\n",
        seg.audio_duration(),
        speech_dur
    ));
    out.push_str("</ChannelList>\n");
    out.push_str("<SpeakerList>\n");
    out.push_str(&format!(
        "<Speaker ch=\"1\" dur=\"{speech_dur:.2}\" gender=\"1\" spkid=\"1\"/>\n"
    ));
    out.push_str("</SpeakerList>\n");
    out.push_str("<SegmentList>\n");
    let segs = seg.segments();
    for i in 0..segs.len().saturating_sub(1) {
        if segs[i].ty == SegClass::Speech {
            out.push_str(&format!(
                "<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"{:.3}\" etime=\"{:.3}\" spkid=\"1\"/>\n",
                segs[i].begin,
                segs[i + 1].begin
            ));
        }
    }
    out.push_str("</SegmentList>\n");
    out.push_str("</AudioDoc>\n");
    out
}

/// Write `seg` as VRCTS XML to `out`. Wraps [`to_vrcts_string`].
pub fn write_vrcts(seg: &Segmentation, name: &str, path_attr: &str, out: &Path) -> Result<()> {
    std::fs::write(out, to_vrcts_string(seg, name, path_attr))?;
    Ok(())
}

/// Extract the `name=` and `path=` attribute values (verbatim) from the
/// `<AudioDoc ...>` line.
pub fn parse_audiodoc_attrs(text: &str) -> (String, String) {
    let line = text
        .lines()
        .find(|l| l.starts_with("<AudioDoc "))
        .unwrap_or("");
    (
        extract_attr(line, "name").unwrap_or_default(),
        extract_attr(line, "path").unwrap_or_default(),
    )
}

/// Extract `key="value"` from a single XML-ish line (no quote-escaping, as in
/// the legacy `iof::fmtr` scanf-style parser).
fn extract_attr(line: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse a VRCTS XML document into a [`Segmentation`]. Direct port of
/// `Segmentation::load_from_vrcts` (`Segmentation.cpp:589-613`), extended to
/// also read the channel `sigdur` to construct the `Segmentation` (the
/// legacy loader mutates an already-constructed `Segmentation` in place;
/// this container has no default-constructor path, so the round-trip test
/// threads `sigdur` back in as `audio_duration`).
///
/// A `SpeechSegment` is accepted iff `end >= off && begin < off + dur`
/// (`Segmentation.cpp:602`).
pub fn load_vrcts(text: &str, off: f64, dur: f64) -> Segmentation {
    let sigdur = text
        .lines()
        .find(|l| l.starts_with("<Channel "))
        .and_then(|l| extract_attr(l, "sigdur"))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(dur);

    let mut seg = Segmentation::new(sigdur);

    for line in text.lines() {
        if !line.starts_with("<SpeechSegment ") {
            continue;
        }
        let stime = extract_attr(line, "stime").and_then(|s| s.parse::<f64>().ok());
        let etime = extract_attr(line, "etime").and_then(|s| s.parse::<f64>().ok());
        if let (Some(begin), Some(end)) = (stime, etime)
            && end >= off
            && begin < off + dur
        {
            seg.label_segment(begin - off, end - off, SegClass::Speech);
        }
    }
    seg.sanitize();
    seg
}
