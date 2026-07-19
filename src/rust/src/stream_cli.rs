//! `speech stream`: the online SAD streaming CLI (Phase 8 Task 6). Port-only
//! tooling (the `bench` / `--convert-config` precedent, `main.rs`) -- no legacy
//! source. Replays a MONO wav through the chunked-input [`StreamingSession`]
//! (`fast/stream.rs`), the online twin of the offline fast algo-3 driver, and
//! prints one parseable `SEG` line per finalized segment plus a final `STREAM`
//! summary.
//!
//! LINE FORMATS (the T8 corpus instrument reads these):
//!   `SEG begin=<%.4f> end=<%.4f> class=<Speech|Other> lag_s=<%.3f>`
//!       -- one per finalized emission, `lag_s = emitted_at_audio_s - end_s`.
//!   `STREAM chunks=<n> emitted=<n> finish_emitted=<n> max_lag_s=<%.3f> \
//!    mean_lag_s=<%.3f> rtf=<%.6f>`
//!       -- `emitted`/`finish_emitted` split mid-stream (push) vs EOS-drain
//!       (finish) emissions; `max_lag_s`/`mean_lag_s` are the session's own
//!       running stats (over ALL emissions); `rtf = wall_s / audio_s_pushed`.
//!
//! STDOUT IS LINE-BUFFERED and each `SEG` line is FLUSHED as it is emitted, so a
//! downstream streaming consumer sees segments the moment they finalize (Rust's
//! `Stdout` is a `LineWriter`, but a PIPED stdout is block-buffered -- the
//! explicit per-line flush guarantees streaming delivery either way).
//!
//! RAW SAMPLES (spec S1.2). The session's front-end applies the frozen
//! gain/preemph/noise per sample itself, so the CLI pushes RAW `i16/32768` f32
//! samples -- it must NOT run them through `audio::read_audio` (which applies
//! `Audio_fixed_gain` / the `(2*RMS+max)/2` normalization). Hence the minimal
//! PCM16 WAV parse below (the same shape the phase8 test staging reads), NOT the
//! symphonia decode path. A non-mono wav is a clean error (the session enforces
//! mono; the CLI passes the wav's real channel count so that bail fires).

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};

use crate::cli::load_config;
use crate::fast::stream::{EmittedSegment, StreamingSession};
use crate::tasks::segmentation::SegClass;

/// The `Speech`/`Other` label for a SAD partition segment (the SAD decision layer
/// only ever labels these two; anything else falls back to its debug name).
fn class_str(c: SegClass) -> String {
    match c {
        SegClass::Speech => "Speech".to_string(),
        SegClass::Other => "Other".to_string(),
        other => format!("{other:?}"),
    }
}

/// Parse a minimal PCM16 WAV file into `(sample_rate, channels, raw f32 samples)`
/// where each sample is `i16 as f32 / 32768.0`, interleaved by channel. Walks the
/// word-aligned RIFF chunks for "fmt "/"data"; bails on anything that is not linear
/// PCM16 (the only format the streaming CLI accepts, matching the phase8 fixtures).
fn read_wav_pcm16_raw(path: &Path) -> Result<(f64, usize, Vec<f32>)> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("cannot read wav '{}': {e}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("'{}' is not a RIFF/WAVE file", path.display());
    }

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;
    let mut data: &[u8] = &[];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if pos + 8 + size > bytes.len() {
            break;
        }
        let body = &bytes[pos + 8..pos + 8 + size];
        if id == b"fmt " && body.len() >= 16 {
            audio_format = u16::from_le_bytes(body[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(body[14..16].try_into().unwrap());
        } else if id == b"data" {
            data = body;
        }
        pos += 8 + size + (size % 2); // chunks are word-aligned
    }

    if audio_format != 1 || bits_per_sample != 16 {
        bail!(
            "'{}': only linear PCM16 is supported (format {audio_format}, {bits_per_sample} bits)",
            path.display()
        );
    }
    if channels == 0 || sample_rate == 0 || data.is_empty() {
        bail!("'{}': missing fmt/data chunk", path.display());
    }

    let samples: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    Ok((sample_rate as f64, channels as usize, samples))
}

/// Write a single `SEG` line and flush it (so it reaches a streaming consumer at
/// emission time, not at process exit).
fn print_seg(out: &mut impl Write, seg: &EmittedSegment) -> Result<()> {
    let lag = seg.emitted_at_audio_s - seg.end_s;
    writeln!(
        out,
        "SEG begin={:.4} end={:.4} class={} lag_s={:.3}",
        seg.begin_s,
        seg.end_s,
        class_str(seg.class),
        lag
    )?;
    out.flush()?;
    Ok(())
}

/// Run the `speech stream [--chunk-ms=N] <config> <wav>` subcommand: load the config
/// (`.toml`/`.config` by extension), read the mono wav's RAW samples, build the
/// [`StreamingSession`] (which validates the streaming contract: algo 3, mono,
/// frozen gain, `InputNormalizationType 1`, ...), push the samples in `chunk_ms`
/// blocks printing one `SEG` line per emission, then finish and print the `STREAM`
/// summary. `rtf = wall_s / audio_s`, `audio_s = samples / rate`.
pub fn run_stream(config: &str, wav: &str, chunk_ms: u64) -> Result<()> {
    let map = load_config(config)?;
    let (rate, channels, samples) = read_wav_pcm16_raw(Path::new(wav))?;

    // The session enforces mono (channels == 1) with its own clear bail; pass the
    // wav's real channel count so a stereo input fails LOUDLY here.
    let mut sess = StreamingSession::new(&map, rate, channels)?;

    let chunk = (((chunk_ms as f64 / 1000.0) * rate).round() as usize).max(1);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut chunks = 0usize;
    let mut emitted = 0usize;
    let mut finish_emitted = 0usize;

    let start = Instant::now();
    let mut i = 0;
    while i < samples.len() {
        let end = (i + chunk).min(samples.len());
        for seg in sess.push(&samples[i..end]) {
            print_seg(&mut out, &seg)?;
            emitted += 1;
        }
        i = end;
        chunks += 1;
    }
    let (finish_em, _seg) = sess.finish();
    for seg in &finish_em {
        print_seg(&mut out, seg)?;
        finish_emitted += 1;
    }
    let wall_s = start.elapsed().as_secs_f64();

    let audio_s = samples.len() as f64 / rate;
    let rtf = if audio_s > 0.0 { wall_s / audio_s } else { 0.0 };
    writeln!(
        out,
        "STREAM chunks={chunks} emitted={emitted} finish_emitted={finish_emitted} \
         max_lag_s={:.3} mean_lag_s={:.3} rtf={:.6}",
        sess.max_lag_s(),
        sess.mean_lag_s(),
        rtf
    )?;
    out.flush()?;
    Ok(())
}
