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
/// `Segmentation::toFile_VRCTS` (`Segmentation.cpp:543-587`), single-channel
/// (`chan = 1`), offset folded to `0.0` (this container has no `_AudioOffset`
/// field).
///
/// `toFile_VRCTS` calls `sanitize()` on the channel before summing/printing
/// (`Segmentation.cpp:545`); we sanitize a clone so the caller's `seg` is not
/// mutated. Post-sanitize boundaries lie exactly on the 1e-4 grid, so the
/// 4-decimal `stime`/`etime` display is exact.
pub fn to_vrcts_string(seg: &Segmentation, name: &str, path_attr: &str) -> String {
    let mut seg = seg.clone();
    seg.sanitize();
    let seg = &seg;
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
                "<SpeechSegment ch=\"1\" sconf=\"1.00\" stime=\"{:.4}\" etime=\"{:.4}\" spkid=\"1\"/>\n",
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

/// Load a reference [`Segmentation`] from STM text, single-channel projection.
/// Direct port of `Segmentation::load_ref_from_stm` (`Segmentation.cpp:616-635`).
///
/// Each line is parsed as 7 whitespace tokens
/// `first(str) line_chan(int) second(str) beg(f64) end(f64) third(str) fourth(str)`;
/// a line whose 7-token parse fails is skipped (this is how the `;;` header lines
/// are dropped -- `line_chan` fails to parse `;;`). `line_chan` is 1-based; after
/// `line_chan -= 1` the line is processed only when `line_chan == chan` (the
/// legacy fills `_Reference.at(line_chan)` guarded by `line_chan < _ChannelNb`;
/// our per-channel API narrows that to an equality on `chan`).
///
/// A line is SPEECH iff `second.starts_with(first)` (the legacy
/// `second.compare(0, first.length(), first) == 0`); else it is EXCLUDED iff
/// `exclude_nontrans && second == "excluded_region"`. Both require the window
/// guard `end >= off && beg < off + dur`. `beg`/`end` are shifted by `-off`
/// before labeling. `sanitize()` runs after all lines.
pub fn load_ref_stm(
    text: &str,
    chan: usize,
    off: f64,
    dur: f64,
    exclude_nontrans: bool,
) -> Segmentation {
    let mut seg = Segmentation::new(dur);

    for line in text.lines() {
        let mut tok = line.split_whitespace();
        let (first, line_chan, second, beg, end) = match (
            tok.next(),
            tok.next().and_then(|t| t.parse::<i64>().ok()),
            tok.next(),
            tok.next().and_then(|t| t.parse::<f64>().ok()),
            tok.next().and_then(|t| t.parse::<f64>().ok()),
        ) {
            (Some(first), Some(line_chan), Some(second), Some(beg), Some(end)) => {
                (first, line_chan, second, beg, end)
            }
            _ => continue,
        };
        // The legacy parse also requires `third` and `fourth` tokens; a line with
        // fewer than 7 whitespace tokens fails the extraction and is skipped.
        if tok.next().is_none() || tok.next().is_none() {
            continue;
        }

        let line_chan = line_chan - 1; // STM is 1-based.
        if line_chan != chan as i64 {
            continue;
        }

        let in_window = end >= off && beg < off + dur;
        if second.starts_with(first) && in_window {
            seg.label_segment(beg - off, end - off, SegClass::Speech);
        } else if exclude_nontrans && second == "excluded_region" && in_window {
            seg.label_segment(beg - off, end - off, SegClass::Excluded);
        }
    }
    seg.sanitize();
    seg
}

/// Load a reference [`Segmentation`] from CSV text, returning `(seg, nb_words)`.
/// Direct port of `Segmentation::load_ref_from_csv` (`Segmentation.cpp:745-806`).
///
/// NOTE: the pinned interface omitted `pruning_thresh`; it is added here because
/// the legacy `_PruningThresh` gate (`Segmentation.cpp:786`) is load-bearing --
/// it decides whether a parsed line is labeled at all. See task-6-report.md.
///
/// Lines starting with `#` are skipped. Each remaining line has its commas
/// replaced by spaces and is parsed as `beg(f64) end(f64) type(str) conf(f64)`;
/// a failed parse skips the line. Within the window `end >= off && beg < off +
/// dur`: `nb_words` is incremented for every `type != "I"` line (this precedes,
/// and is independent of, the confidence gate); `beg -= off` and
/// `end = end - 1e-4 - off` (the extra `-1e-4` on `end` only). Then, iff
/// `conf >= pruning_thresh`, the line is labeled `C -> Speech`, `S ->
/// Substitution`, `I -> Insertion`, else `Excluded`. `sanitize()` runs after.
///
/// `nb_words` is a plain counter from 0: the legacy seeds `_NbWords = 0` when it
/// is `< 0`, which makes the loop's `< 0 -> 1` branch dead; reproduced as a
/// count of in-window non-`"I"` lines.
pub fn load_ref_csv(text: &str, off: f64, dur: f64, pruning_thresh: f64) -> (Segmentation, i64) {
    let mut seg = Segmentation::new(dur);
    let mut nb_words: i64 = 0;

    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let spaced = line.replace(',', " ");
        let mut tok = spaced.split_whitespace();
        let (beg, end, ty, conf) = match (
            tok.next().and_then(|t| t.parse::<f64>().ok()),
            tok.next().and_then(|t| t.parse::<f64>().ok()),
            tok.next(),
            tok.next().and_then(|t| t.parse::<f64>().ok()),
        ) {
            (Some(beg), Some(end), Some(ty), Some(conf)) => (beg, end, ty, conf),
            _ => continue,
        };

        if end >= off && beg < off + dur {
            if ty != "I" {
                nb_words += 1;
            }
            let beg = beg - off;
            let end = end - 1e-4 - off;
            if conf >= pruning_thresh {
                let class = match ty {
                    "C" => SegClass::Speech,
                    "S" => SegClass::Substitution,
                    "I" => SegClass::Insertion,
                    _ => SegClass::Excluded,
                };
                seg.label_segment(beg, end, class);
            }
        }
    }
    seg.sanitize();
    (seg, nb_words)
}

/// Serialize `seg` as the legacy ASCII segmentation. Direct port of
/// `Segmentation::toFile_ASCII` (`Segmentation.cpp:519-529`), single-channel
/// (`chan` hardcoded `1`), offset folded to `0.0`.
///
/// `toFile_ASCII` sanitizes the channel before writing; we sanitize a clone so
/// the caller's `seg` is not mutated. Each segment `it` (with `it + 1` before the
/// sentinel) emits TWO lines, both carrying the CURRENT segment's `ty as i32`:
/// line 1 at `it.begin`, line 2 at `next.begin - 1e-3`, times at 3 decimals
/// (`%f.3s` == `%.3f`).
pub fn to_ascii_string(seg: &Segmentation) -> String {
    let mut seg = seg.clone();
    seg.sanitize();
    let segs = seg.segments();
    let mut out = String::new();
    for i in 0..segs.len().saturating_sub(1) {
        let code = segs[i].ty as i32;
        out.push_str(&format!("1 {:.3} {code}\n", segs[i].begin));
        out.push_str(&format!("1 {:.3} {code}\n", segs[i + 1].begin - 1e-3));
    }
    out
}

/// Write `seg` as legacy ASCII to `out`. Wraps [`to_ascii_string`].
pub fn write_ascii(seg: &Segmentation, out: &Path) -> Result<()> {
    std::fs::write(out, to_ascii_string(seg))?;
    Ok(())
}
