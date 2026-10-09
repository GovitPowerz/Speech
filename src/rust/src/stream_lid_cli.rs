//! `speech stream-lid`: the per-utterance LID streaming CLI (Phase 10 Task 10). Port-only
//! tooling (the `stream`/`bench` precedent, `main.rs`) -- no legacy source. Replays a
//! phSeq/cep utterances file through the chunked-input [`StreamingLidSession`]
//! (`fast/stream_lid.rs`), the online twin of the offline fast Mode-7 LID Twin, and
//! prints one `UTT` line per pushed utterance plus a final `STREAM_LID` summary.
//!
//! This closes the deferral `fast::stream_lid` left open since Phase 8 (lib-only,
//! "a `stream-lid` CLI / PyO3 binding is deferred to phase 9" -- the note was stale by
//! one phase; phase 9 never picked it up either, so it fell off that phase's own
//! Deferred list, see spec S8).
//!
//! LID is UTTERANCE-GRANULAR (unlike the frame-granular `speech stream`): one
//! `external_features` entry (one phSeq line / one cep record) is one "utterance", so
//! this CLI reads the WHOLE input file up front (via [`crate::audio::read_audio`]) and
//! then pushes its entries one at a time -- there is no raw-sample chunking loop to
//! design here (`speech stream`'s `--chunk-ms` has no LID analogue).
//!
//! FILE TYPE comes from the config's `File_Type` key (mirroring
//! `engine::bag_of_processors::BagOfProcessors::from_configs`'s own read of the same
//! key), NOT a CLI flag -- the config already carries it in every committed fixture
//! (`File_Type 1` for the phSeq gate config), so a second flag would just duplicate
//! information the config already states. Only `File_Type` 1 (phSeq) and 2 (cep) carry
//! `external_features`; anything else bails loudly before the session is even built.
//!
//! LINE FORMATS (fields listed in the order they are printed; a spawned-binary smoke
//! pins these against the committed phase-8 stream-lid phSeq fixture):
//!   `UTT idx=<n> scored=<true|false> argmax=<n|-> correct=<true|false|-> \
//!    agg_predicted_language=<n> agg_is_lid_correct=<true|false> agg_segments_count=<n>`
//!       -- one per pushed utterance (`UtteranceScore`'s fields): `argmax`/`correct` are
//!       `-` iff the utterance was skipped (too short / below `MinNbOfFrames`); the
//!       `agg_*` fields are the running aggregate snapshot AFTER folding this utterance.
//!   `STREAM_LID utterances=<n> scored=<n> predicted_language=<n> is_lid_correct=<true|false> \
//!    wall_s=<%.6f>`
//!       -- the final summary: `utterances` is every entry pushed (scored + skipped),
//!       `scored`/`predicted_language`/`is_lid_correct` are `finish()`'s aggregate
//!       (`LidAggregate::segments_count`/`predicted_language`/`is_lid_correct`).
//!
//! STDOUT IS LINE-BUFFERED and each line is FLUSHED as it is printed, matching
//! `stream_cli.rs`'s per-line flush convention (a downstream streaming consumer sees
//! each utterance's result the moment it is scored).

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};

use crate::audio::read_audio;
use crate::cli::load_config;
use crate::engine::bag_of_processors::get_i32_default;
use crate::fast::stream_lid::{StreamingLidSession, UtteranceScore};

/// Write one `UTT` line and flush it.
fn print_utt(out: &mut impl Write, idx: usize, sc: &UtteranceScore) -> Result<()> {
    writeln!(
        out,
        "UTT idx={idx} scored={} argmax={} correct={} agg_predicted_language={} \
         agg_is_lid_correct={} agg_segments_count={}",
        sc.argmax.is_some(),
        opt_usize(sc.argmax),
        opt_bool(sc.is_correct),
        sc.running_aggregate.predicted_language,
        sc.running_aggregate.is_lid_correct != 0,
        sc.running_aggregate.segments_count,
    )?;
    out.flush()?;
    Ok(())
}

fn opt_usize(v: Option<usize>) -> String {
    v.map_or_else(|| "-".to_string(), |x| x.to_string())
}

fn opt_bool(v: Option<bool>) -> String {
    v.map_or_else(|| "-".to_string(), |x| x.to_string())
}

/// Run the `speech stream-lid [--lang=N] <config> <input>` subcommand: load the config
/// (`.toml`/`.config` by extension), read `File_Type` from it (bailing on anything but
/// 1/2 -- the only types that carry `external_features`), read the utterances file,
/// build the [`StreamingLidSession`] (which validates the streaming contract: Mode 7,
/// `InputNormalizationType 0`, plain or truncate windowing per the shape (overlap
/// refused), a loaded LID net, ...), push every
/// `external_features` entry printing one `UTT` line each, then print the `STREAM_LID`
/// summary from `finish()`.
pub fn run_stream_lid(config: &str, input: &str, lang: i32) -> Result<()> {
    let map = load_config(config)?;

    // legacy: `engine::bag_of_processors`'s own `File_Type` read + default (`:384`);
    // mirrored here so a config with no explicit `File_Type` fails the SAME way the
    // full engine would, rather than silently guessing phSeq or cep.
    let file_type = get_i32_default(&map, "File_Type", 0)?;
    if file_type != 1 && file_type != 2 {
        bail!(
            "stream-lid: File_Type {file_type} not supported (only 1=phSeq, 2=cep carry \
             external_features; set File_Type in the config)"
        );
    }
    // `max_duration_sec` is ignored on the phSeq/cep read paths (audio.rs's own
    // contract); `offset_sec 0.0` likewise unused there.
    let audio = read_audio(Path::new(input), 0.0, 3.6e6, file_type, None)?;
    let rate = audio.sample_rate as f64;

    let mut sess = StreamingLidSession::new(&map, rate, lang, None)?;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let start = Instant::now();

    for (idx, feat) in audio.external_features.iter().enumerate() {
        let sc = sess.push_utterance(feat);
        print_utt(&mut out, idx, &sc)?;
    }

    let wall_s = start.elapsed().as_secs_f64();
    let fin = sess.finish();
    writeln!(
        out,
        "STREAM_LID utterances={} scored={} predicted_language={} is_lid_correct={} wall_s={:.6}",
        audio.external_features.len(),
        fin.segments_count,
        fin.predicted_language,
        fin.is_lid_correct != 0,
        wall_s,
    )?;
    out.flush()?;
    Ok(())
}
