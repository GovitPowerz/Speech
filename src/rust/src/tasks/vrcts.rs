//! External-tool adapter behind the `Segmenter` trait (shell out + re-ingest XML).
//!
//! Ported from legacy C++: VRCTSpart.*. Phase 4b Task 8.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use indexmap::IndexMap;

use crate::audio::Audio;
use crate::tasks::segmentation::Segmentation;
use crate::tasks::segmentation_io::{ScoreReport, compute_errors, load_vrcts};
use crate::tasks::segmenter::{DriverConfig, Segmenter, SegmenterConfig};

/// legacy: hard-coded absolute path (`VRCTSpart.cpp:44,46`), not read from config.
/// IMPROVEMENTS [phase4b]: hard-coded VRCTS binary path.
const VRCTS_BINARY: &str = "/usr/local/vrcts/vrcts_1_5_9/bin/vrcts_part";

fn parse_bool(m: &IndexMap<String, String>, key: &str) -> Result<bool> {
    // legacy `conf.get<bool>` uses std::boolalpha: the strings "true"/"false".
    m.get(key)
        .ok_or_else(|| anyhow!("param '{key}' not found in config"))?
        .trim()
        .parse::<bool>()
        .map_err(|e| anyhow!("`{key}`: cannot parse as bool: {e}"))
}

/// External-tool VRCTS SAD adapter (Algo 0; `VRCTSpart.{h,cpp}`). Unlike Algo
/// 1-4, this driver has NO in-process detection logic: `getSegmentation` shells
/// out to the legacy `vrcts_part` binary, then re-ingests its XML output via
/// [`load_vrcts`] (already ported, Phase 0b-ii/2b).
///
/// The legacy ctor also runs `Segmenter::buildFromConf` (`VRCTSpart.cpp:9`),
/// which reads a full set of generic keys (decision thresholds, window/shift,
/// windowing type, preemph/noise, convolution kernel -- `Segmenter.cpp:73-148`)
/// -- but `VRCTSPart::getSegmentation` never calls `updateSegmentation`/
/// `smoothSegmentation`/`results2segmentation`, so none of those fields affect
/// any observable output. This port replicates the generic REQUIRED-KEY
/// validation (same as every sibling driver in `tasks/sad.rs`: `TdcSegmenter`/
/// `LtsvSegmenter`/`BlstmSignalSegmenter`/`BlstmSpectralSegmenter`) by calling
/// [`SegmenterConfig::from_config`] + [`DriverConfig::from_config`] with prefix
/// `"VRCTS"` -- a real legacy config missing e.g. `VRCTS_window` now fails to
/// construct `VrctsPart`, matching `buildFromConf`'s `exit(1)`. The parsed
/// values themselves stay unused beyond that validation (`seg_cfg`), or are
/// used only for `dump_dir()` (`driver_cfg`), since nothing downstream reads
/// them -- this is a deliberate PARTIAL replication of `buildFromConf` (the
/// windowing/preemph/noise reads and the `CostLaw` construction are still
/// skipped, same as every sibling driver, none of which read them either).
#[derive(Clone)]
pub struct VrctsPart {
    /// Stored only to replicate `buildFromConf`'s required-key validation
    /// (`Segmenter.cpp:83-138`); `getSegmentation` never reads its fields.
    #[allow(dead_code)]
    seg_cfg: SegmenterConfig,
    driver_cfg: DriverConfig,
    is_fast: bool,
    force: bool,
    channels: usize,
}

impl VrctsPart {
    /// Port of the `VRCTSPart(ConfigFile&, bool, bool)` ctor (`VRCTSpart.cpp:8-12`):
    /// `buildFromConf(conf, "VRCTS", ...)` (`Segmenter.cpp:73-148`) via
    /// [`SegmenterConfig::from_config`] + [`DriverConfig::from_config`], THEN
    /// `VRCTS_isFast`/`VRCTS_force` (both REQUIRED, no default -- `conf.get<bool>`
    /// with no default arg).
    pub fn from_legacy(map: &IndexMap<String, String>) -> Result<VrctsPart> {
        let seg_cfg = SegmenterConfig::from_config(map, "VRCTS")?;
        let driver_cfg = DriverConfig::from_config(map, "VRCTS")?;

        let is_fast = parse_bool(map, "VRCTS_isFast")?;
        let force = parse_bool(map, "VRCTS_force")?;
        Ok(VrctsPart {
            seg_cfg,
            driver_cfg,
            is_fast,
            force,
            channels: 0,
        })
    }

    /// The configured dump directory (`Segmenter::_DumpDir`, read via
    /// [`DriverConfig`]): `SegmentationFunction` reads it per config to gate the
    /// VRCTS write (`BagOfProcessors.cpp:268,273,...`).
    pub fn dump_dir(&self) -> &str {
        &self.driver_cfg.dump_dir
    }

    /// Zero cost accumulators: VRCTS has no NN/cost path (`cumulative_error`/
    /// `nb_of_classif` stay at their legacy `Segmentation` ctor-seeded 0.0/0
    /// defaults for every channel -- `getSegmentation` never writes them). Sized
    /// by the channel count of the last `get_segmentation` call.
    pub fn cumulative_error(&self) -> Vec<f64> {
        vec![0.0; self.channels]
    }

    pub fn nb_of_classif(&self) -> Vec<i64> {
        vec![0; self.channels]
    }

    /// `Segmentation::compute_errors`, one call per channel (same pattern as
    /// `TdcSegmenter::score`/`LtsvSegmenter::score`).
    pub fn score(
        hyp: &mut [Segmentation],
        reference: Option<&[Segmentation]>,
        nb_words: i64,
    ) -> Vec<ScoreReport> {
        hyp.iter_mut()
            .enumerate()
            .map(|(chan, seg)| {
                let refc = reference.map(|r| &r[chan]);
                compute_errors(seg, refc, nb_words)
            })
            .collect()
    }
}

impl Segmenter for VrctsPart {
    /// Port of `VRCTSPart::getSegmentation` (`VRCTSpart.cpp:21-71`), the
    /// non-plotting path (the PNG dump at `:63-66` is display-only, dropped).
    ///
    /// Per channel: compose `<ref_seg_file_name>_VRCTS_{Fast,Long}.xml` --
    /// NOT `_chan_<n>`-suffixed: the per-channel variant is commented out in
    /// the legacy source (`:34,37`), so every channel of a multi-channel file
    /// shares the SAME xml path, and therefore the SAME loaded segmentation --
    /// a load-bearing quirk, reproduced verbatim (pinned by
    /// `same_xml_shared_across_channels`).
    ///
    /// If `force || !exists`, spawn the legacy `vrcts_part` binary
    /// (`-q1`/`-q0` for fast/long, `-x -f <audio> -o <xml>`). The legacy
    /// `system(command.str().c_str())` call (`:53`) is a bare statement -- its
    /// return value (the shell's exit status) is NEVER checked, so a nonzero
    /// exit (including "command not found", exit 127, if the binary is
    /// missing) is silently accepted and `load_from_vrcts` simply sees
    /// whatever the xml file (still) contains. This port cannot reproduce
    /// "the OS syscall itself always succeeds" (`std::process::Command` execs
    /// the binary directly, no intervening shell): a SPAWN failure (binary
    /// missing -> `io::Error`) is a condition the legacy `system()` call never
    /// surfaces at all. So only the SPAWN-level failure becomes a typed error
    /// here (IMPROVEMENTS [phase4b]: hard-coded VRCTS binary path; spawn
    /// failure semantics necessarily diverge from the legacy's discarded
    /// `system()` return). A successful spawn's exit code IS discarded,
    /// matching legacy exactly.
    ///
    /// The XML re-ingest (`seg.load_from_vrcts`, `Segmentation.cpp:592-614`)
    /// itself never errors on a missing/unreadable file: `ifstream::operator!`
    /// just short-circuits the `getline` loop, leaving the hypothesis
    /// untouched (still `[Other@0, End@dur]`). Reproduced by feeding
    /// [`load_vrcts`] an empty string when the xml file cannot be read --
    /// `load_vrcts("", ...)` finds no `<SpeechSegment>` line and no `<Channel>`
    /// line, so it returns an equivalently-seeded, empty `Segmentation`.
    ///
    /// `refs` is ignored: VRCTS has no NN/training path (same rationale as
    /// TDC/LTSV).
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        _refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        self.channels = seg_per_chan.len();

        for (chan, seg) in seg_per_chan.iter_mut().enumerate() {
            let suffix = if self.is_fast { "Fast" } else { "Long" };
            let xml_path = format!("{}_VRCTS_{suffix}.xml", audio.ref_seg_file_name);

            let exists = Path::new(&xml_path).exists();
            if self.force || !exists {
                let q_flag = if self.is_fast { "-q1" } else { "-q0" };
                let status = Command::new(VRCTS_BINARY)
                    .arg(q_flag)
                    .arg("-x")
                    .arg("-f")
                    .arg(&audio.audio_file_name)
                    .arg("-o")
                    .arg(&xml_path)
                    .status()
                    .with_context(|| {
                        format!(
                            "VrctsPart: failed to spawn `{VRCTS_BINARY}` for channel {chan} \
                             (IMPROVEMENTS [phase4b]: hard-coded VRCTS binary path)"
                        )
                    })?;
                // legacy: VRCTSpart.cpp:53 `system(command.str().c_str());` -- the
                // return value (shell exit status) is discarded, not checked.
                let _ = status;
            }

            let text = std::fs::read_to_string(&xml_path).unwrap_or_default();
            *seg = load_vrcts(&text, audio.audio_offset, seg.audio_duration());
        }

        // legacy: VRCTSpart.cpp:69 `seg.compute_errors();` after the full channel
        // loop; no reference is threaded here (same no-target pattern as TDC/LTSV).
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }

        Ok(())
    }
}
