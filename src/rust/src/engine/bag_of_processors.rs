//! One `Segmenter` per config, dispatched by an `Algo` enum (replaces the legacy
//! if/else ladder); per-file dispatch, weight save/update, metric reduction.
//!
//! Ported from legacy C++: BagOfProcessors.*.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use indexmap::IndexMap;
use ndarray::Array2;

use crate::audio::{Audio, read_audio};
use crate::cli::{Mode, ModeKind};
use crate::engine::corpus::CorpusItem;
use crate::features::stats::InputStatistics;
use crate::tasks::sad::{
    BlstmSignalSegmenter, BlstmSpectralSegmenter, LtsvSegmenter, TdcSegmenter,
};
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::{
    ScoreReport, compute_errors, load_ref_csv, load_ref_stm, write_vrcts,
};

/// `conf.get<int>(name)` (required, no default): missing key is an error.
fn get_i32(map: &IndexMap<String, String>, key: &str) -> Result<i32> {
    map.get(key)
        .ok_or_else(|| anyhow::anyhow!("missing required config key `{key}`"))?
        .trim()
        .parse::<i32>()
        .map_err(|e| anyhow::anyhow!("`{key}`: cannot parse as i32: {e}"))
}

/// `conf.get<double>(name, default)`: missing key -> default.
fn get_f64_default(map: &IndexMap<String, String>, key: &str, default: f64) -> Result<f64> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow::anyhow!("`{key}`: cannot parse as f64: {e}")),
    }
}

/// `conf.get<int>(name, default)`: missing key -> default.
fn get_i32_default(map: &IndexMap<String, String>, key: &str, default: i32) -> Result<i32> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<i32>()
            .map_err(|e| anyhow::anyhow!("`{key}`: cannot parse as i32: {e}")),
    }
}

/// `conf.get<string>(name, default)`: missing key -> default.
fn get_string_default(map: &IndexMap<String, String>, key: &str, default: &str) -> String {
    map.get(key).cloned().unwrap_or_else(|| default.to_string())
}

/// `conf.get<bool>(name, default)`: missing key -> default (legacy std::boolalpha).
fn get_bool_default(map: &IndexMap<String, String>, key: &str, default: bool) -> Result<bool> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<bool>()
            .map_err(|e| anyhow::anyhow!("`{key}`: cannot parse as bool: {e}")),
    }
}

/// One per-config driver, replacing the legacy `_ConfigIndex` + 6 parallel typed
/// vectors: `processors[pos]` IS the config-`pos` driver directly (the enum tag
/// substitutes for the legacy's separate `_AlgoTypes[pos]` dispatch on WHICH
/// vector `_ConfigIndex[pos]` indexes into). Algo 0 (`VRCTSPart`, external-tool
/// adapter), 5 (`BLSTMSpectralLID`), 6 (`TwinBLSTMSpectralLID`) are unported
/// (Phase 4b): see [`BagOfProcessors::from_configs`].
///
/// `large_enum_variant` allowed: the brief's signature is exact
/// (`Spectral(BlstmSpectralSegmenter)`, no `Box`); boxing would change the public
/// interface for a lint, not a correctness issue.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Processor {
    Tdc(TdcSegmenter),
    Ltsv(LtsvSegmenter),
    Spectral(BlstmSpectralSegmenter),
    Signal(BlstmSignalSegmenter),
}

impl Processor {
    /// Dispatch `Segmenter::getSegmentation` (`BagOfProcessors.cpp:267,272,...`):
    /// the enum tag replaces the legacy `_AlgoTypes[ii]` if/else ladder.
    fn get_segmentation(
        &mut self,
        audio: &mut crate::audio::Audio,
        seg_per_chan: &mut [crate::tasks::segmentation::Segmentation],
    ) -> Result<()> {
        use crate::tasks::segmenter::Segmenter;
        match self {
            Processor::Tdc(s) => s.get_segmentation(audio, seg_per_chan),
            Processor::Ltsv(s) => s.get_segmentation(audio, seg_per_chan),
            Processor::Spectral(s) => s.get_segmentation(audio, seg_per_chan),
            Processor::Signal(s) => s.get_segmentation(audio, seg_per_chan),
        }
    }

    /// The driver's `_DumpDir` (`BagOfProcessors.cpp:268,273,...`): gates the
    /// scored-branch VRCTS write.
    fn dump_dir(&self) -> &str {
        match self {
            Processor::Tdc(s) => s.dump_dir(),
            Processor::Ltsv(s) => s.dump_dir(),
            Processor::Spectral(s) => s.dump_dir(),
            Processor::Signal(s) => s.dump_dir(),
        }
    }

    /// Per-channel `seg._CumulativeError` (result col 4). TDC/LTSV are NN-free
    /// and return owned zero vecs; the NN drivers return the cost captured on the
    /// last `get_segmentation` call.
    fn cumulative_error(&self) -> Vec<f64> {
        match self {
            Processor::Tdc(s) => s.cumulative_error(),
            Processor::Ltsv(s) => s.cumulative_error(),
            Processor::Spectral(s) => s.cumulative_error().to_vec(),
            Processor::Signal(s) => s.cumulative_error().to_vec(),
        }
    }

    /// Per-channel `seg._NbOfClassif` (result col 17).
    fn nb_of_classif(&self) -> Vec<i64> {
        match self {
            Processor::Tdc(s) => s.nb_of_classif(),
            Processor::Ltsv(s) => s.nb_of_classif(),
            Processor::Spectral(s) => s.nb_of_classif().to_vec(),
            Processor::Signal(s) => s.nb_of_classif().to_vec(),
        }
    }
}

/// Port of `BagOfProcessors` (`BagOfProcessors.h`/`.cpp`): per-config driver
/// construction + the weight/stats dispatch surface. `_ConfigIndex` is dropped
/// (see [`Processor`]'s doc); `algo_types`/`pruning_thresholds` are kept
/// (parallel to `processors`, matching the legacy layout) since Task 5's
/// `SegmentationFunction` port needs `algo_types` for its own per-config
/// dispatch (dump dir, log gating) independent of the `Processor` enum match.
#[derive(Clone)]
pub struct BagOfProcessors {
    nb_of_conf: usize,
    num_outer_threads: i32,
    offset_begin: f64,
    duration_max: f64,
    file_type: i32,
    lock_files_dir: String,
    lock_files_prefix: String,
    algo_types: Vec<i32>,
    pruning_thresholds: Vec<f64>,
    processors: Vec<Processor>,
    exclude_nontrans: bool,
}

impl BagOfProcessors {
    /// Port of `BagOfProcessors(vector<ConfigFile>&, char*)` (`:9-51`).
    ///
    /// Config-0-only keys (`:10-15`): `numOuterThreads` (REQUIRED, no default --
    /// missing is an error, unlike every other key here), `Audio_offset` 0.0,
    /// `Audio_max_duration` 3.6e6, `File_Type` 0, `LockFilesDir`/`LockFilesPrefix`
    /// "". `exclude_nontrans` (NOT a legacy `BagOfProcessors` member -- read from
    /// configs[0] here so it's available alongside the bag for Task 5's reference
    /// loading, default false).
    ///
    /// Per-config (`:20-49`): the ctor first sets `files`/`refsegfiles`/
    /// `reflangfiles` to `""` in EVERY config map (`:21-23`, a memory quirk --
    /// `refdialfiles` is deliberately NOT cleared) BEFORE constructing any
    /// driver, then reads `Algo_choice` (required) + `Pruning_Threshold` (default
    /// 0.0) and dispatches to the matching driver ctor, passing
    /// `(mode[1]=='t')||(mode[1]=='T')` and `(mode[1]=='i')||(mode[1]=='I')` as
    /// the two trailing ctor bools in the legacy. Those two bools are
    /// unit-test/image DISPLAY flags only (`Segmenter::buildFromConf` stores them
    /// solely for `.mat`/PNG dump gating in `getSegmentation`'s dropped
    /// diagnostic branches) -- the Rust `from_legacy(map)`/`from_legacy(map,
    /// weights)` signatures landed in Phase 2b/3 without them, so they are
    /// intentionally NOT threaded through here; `mode` is accepted (matching the
    /// legacy signature and needed for a future Task 5 wire-up of the log
    /// branches) but only currently consulted for algo dispatch validity.
    ///
    /// Algo 0/5/6 (`VRCTSPart`/`BLSTMSpectralLID`/`TwinBLSTMSpectralLID`) are
    /// unported -- `bail!`, deferred to Phase 4b. `File_Type != 0` (non-wav
    /// input) is also unported -- `bail!`, IMPROVEMENTS entry.
    pub fn from_configs(
        configs: &mut [IndexMap<String, String>],
        mode: Mode,
    ) -> Result<BagOfProcessors> {
        let _ = mode; // accepted per legacy signature; not yet consulted (Task 5).

        if configs.is_empty() {
            bail!("BagOfProcessors::from_configs: at least one config is required");
        }

        let num_outer_threads = get_i32(&configs[0], "numOuterThreads")?;
        let offset_begin = get_f64_default(&configs[0], "Audio_offset", 0.0)?;
        let duration_max = get_f64_default(&configs[0], "Audio_max_duration", 3.6e6)?;
        let file_type = get_i32_default(&configs[0], "File_Type", 0)?;
        let lock_files_dir = get_string_default(&configs[0], "LockFilesDir", "");
        let lock_files_prefix = get_string_default(&configs[0], "LockFilesPrefix", "");
        let exclude_nontrans = get_bool_default(&configs[0], "exclude_nontrans", false)?;

        if file_type != 0 {
            // legacy: AudioStruct non-wav read path -- unported (Phase 4b).
            bail!(
                "File_Type {file_type} not ported (Phase 4b): only wav (File_Type 0) is supported"
            );
        }

        let nb_of_conf = configs.len();
        let mut algo_types = Vec::with_capacity(nb_of_conf);
        let mut pruning_thresholds = Vec::with_capacity(nb_of_conf);
        let mut processors = Vec::with_capacity(nb_of_conf);

        for map in configs.iter_mut() {
            // :21-23 key clears, BEFORE constructing the driver. refdialfiles NOT
            // cleared (legacy quirk, reproduced verbatim).
            map.insert("files".to_string(), String::new());
            map.insert("refsegfiles".to_string(), String::new());
            map.insert("reflangfiles".to_string(), String::new());

            let algo = get_i32(map, "Algo_choice")?;
            algo_types.push(algo);
            let pruning_thresh = get_f64_default(map, "Pruning_Threshold", 0.0)?;
            pruning_thresholds.push(pruning_thresh);

            let processor = match algo {
                0 => bail!("Algo 0 not ported (Phase 4b)"),
                1 => Processor::Tdc(TdcSegmenter::from_legacy(map)?),
                2 => Processor::Ltsv(LtsvSegmenter::from_legacy(map)?),
                3 => {
                    let mut seg = BlstmSpectralSegmenter::from_legacy(map, None)?;
                    seg.load_weights_file(map)?;
                    Processor::Spectral(seg)
                }
                4 => {
                    let mut seg = BlstmSignalSegmenter::from_legacy(map, None)?;
                    seg.load_weights_file(map)?;
                    Processor::Signal(seg)
                }
                5 => bail!("Algo 5 not ported (Phase 4b)"),
                6 => bail!("Algo 6 not ported (Phase 4b)"),
                other => bail!("Algo {other} not ported (Phase 4b)"),
            };
            processors.push(processor);
        }

        Ok(BagOfProcessors {
            nb_of_conf,
            num_outer_threads,
            offset_begin,
            duration_max,
            file_type,
            lock_files_dir,
            lock_files_prefix,
            algo_types,
            pruning_thresholds,
            processors,
            exclude_nontrans,
        })
    }

    pub fn nb_of_conf(&self) -> usize {
        self.nb_of_conf
    }

    pub fn nb_of_threads(&self) -> i32 {
        self.num_outer_threads
    }

    pub fn algo_type(&self, pos: usize) -> i32 {
        self.algo_types[pos]
    }

    /// `_LockFilesDir.length() == 0` gate on the FILE-locking branch
    /// (`BagOfProcessors.h:25`, `SegmentationFunction` `:214`): non-empty means
    /// work is distributed across lock files.
    pub fn is_work_distributed(&self) -> bool {
        !self.lock_files_dir.is_empty()
    }

    pub fn offset_begin(&self) -> f64 {
        self.offset_begin
    }

    pub fn duration_max(&self) -> f64 {
        self.duration_max
    }

    pub fn file_type(&self) -> i32 {
        self.file_type
    }

    pub fn lock_files_dir(&self) -> &str {
        &self.lock_files_dir
    }

    pub fn lock_files_prefix(&self) -> &str {
        &self.lock_files_prefix
    }

    pub fn pruning_threshold(&self, pos: usize) -> f64 {
        self.pruning_thresholds[pos]
    }

    pub fn exclude_nontrans(&self) -> bool {
        self.exclude_nontrans
    }

    pub fn processor(&self, pos: usize) -> &Processor {
        &self.processors[pos]
    }

    pub fn processor_mut(&mut self, pos: usize) -> &mut Processor {
        &mut self.processors[pos]
    }

    /// Port of `BagOfProcessors::isBackPropActivated` (`:73-86`): algo 3/4 (the
    /// only ported NN algos) return a single-element vec from the net's
    /// `isBackPropagationActivated`; everything else (here: algo 1/2, the
    /// non-NN segmenters) falls through to the legacy's own default
    /// `vector<bool>(1, false)` (`:85`) -- NOT an empty vec.
    pub fn is_back_prop_activated(&self, pos: usize) -> Vec<bool> {
        match &self.processors[pos] {
            Processor::Spectral(seg) => vec![seg.is_back_prop_activated()],
            Processor::Signal(seg) => vec![seg.is_back_prop_activated()],
            Processor::Tdc(_) | Processor::Ltsv(_) => vec![false],
        }
    }

    /// Port of `BagOfProcessors::getWeights` (`:88-101`): algo 3/4 return a
    /// single-element vec of the net's flat weight vector; everything else
    /// (algo 1/2 here) falls through to the legacy's EMPTY `vector<Eigen::
    /// VectorXd>()` default (`:100`) -- an empty Vec, unlike
    /// `isBackPropActivated`'s single-`false` default.
    pub fn get_weights(&self, pos: usize) -> Vec<Vec<f64>> {
        use crate::tasks::segmenter::Segmenter;
        match &self.processors[pos] {
            Processor::Spectral(seg) => vec![seg.get_weights()],
            Processor::Signal(seg) => vec![seg.get_weights()],
            Processor::Tdc(_) | Processor::Ltsv(_) => Vec::new(),
        }
    }

    /// Port of `BagOfProcessors::setWeights` (`:103-114`): algo 3/4 forward
    /// `new_weights[0]` to the net's `setWeights`; algo 1/2 (no legacy `else`
    /// branch at `:103-114`) are a no-op.
    pub fn set_weights(&mut self, pos: usize, new_weights: &[Vec<f64>]) -> Result<()> {
        use crate::tasks::segmenter::Segmenter;
        match &mut self.processors[pos] {
            Processor::Spectral(seg) => seg.set_weights(&new_weights[0]),
            Processor::Signal(seg) => seg.set_weights(&new_weights[0]),
            Processor::Tdc(_) | Processor::Ltsv(_) => Ok(()),
        }
    }

    /// Port of `BagOfProcessors::getInputStatistics` (`:116-130`): algo 3/4
    /// return a single-element vec of the net's `InputStatistics`; everything
    /// else falls through to the legacy's EMPTY default (`:128-129`).
    pub fn get_input_statistics(&self, pos: usize) -> Vec<InputStatistics> {
        match &self.processors[pos] {
            Processor::Spectral(seg) => vec![seg.input_statistics().clone()],
            Processor::Signal(seg) => vec![seg.input_statistics().clone()],
            Processor::Tdc(_) | Processor::Ltsv(_) => Vec::new(),
        }
    }

    /// Port of `BagOfProcessors::getWeightsDerivatives` (`:132-146`): algo 3/4
    /// return a single-element vec of the net's `Nx2` derivative matrix;
    /// everything else falls through to the legacy's EMPTY default
    /// (`:143-144`).
    pub fn get_weights_derivatives(&self, pos: usize) -> Vec<Array2<f64>> {
        use crate::tasks::segmenter::Segmenter;
        match &self.processors[pos] {
            Processor::Spectral(seg) => vec![seg.get_weights_derivatives()],
            Processor::Signal(seg) => vec![seg.get_weights_derivatives()],
            Processor::Tdc(_) | Processor::Ltsv(_) => Vec::new(),
        }
    }

    /// Dispatch config-`pos`'s `Segmenter::getSegmentation` onto a pre-seeded
    /// per-channel `Segmentation` slice. Exposed so the corpus-processor driver
    /// (and Task 5's own scoring oracle) can drive one config without going
    /// through the whole [`Self::segmentation_function`] flow.
    pub fn run_get_segmentation(
        &mut self,
        pos: usize,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
    ) -> Result<()> {
        self.processors[pos].get_segmentation(audio, seg_per_chan)
    }

    /// Port of `BagOfProcessors::SegmentationFunction` (`:207-407`): read the
    /// file once, then for each config seed a per-channel [`Segmentation`], run
    /// the segmenter, score against the reference, and assemble the per-channel
    /// 18-column result vector (`config -> channel -> row`).
    ///
    /// The lock-file block (`:214-250`) is STUBBED: with no `LockFilesDir` the
    /// legacy always sets `treatFile = true`, so this port unconditionally treats
    /// the file. The `jj` param (legacy lane index, only ever fed the lock
    /// filename) is therefore dropped from the signature. IMPROVEMENTS: `[phase4a]
    /// SegmentationFunction lock-file block stubbed`.
    ///
    /// Reference load dispatches on `item.ref_seg`'s extension (mirroring the
    /// legacy `Segmentation` ctor `:72-110`): `.stm` -> [`load_ref_stm`] per
    /// channel, `.csv` -> [`load_ref_csv`] (also yields `nb_words` for WER),
    /// `.trs` -> `bail!` (unported), anything else / empty -> no reference.
    /// The mandatory-reference check (`:302-305`): a scored mode with no loadable
    /// reference is an error (legacy `exit(1)`).
    pub fn segmentation_function(
        &mut self,
        item: &CorpusItem,
        mode: Mode,
    ) -> Result<BTreeMap<usize, BTreeMap<usize, Vec<f64>>>> {
        let mut results: BTreeMap<usize, BTreeMap<usize, Vec<f64>>> = BTreeMap::new();

        // legacy: :254 AudioStruct audio(_OffsetBegin, _DurationMax, _FileType, corpusItem);
        let file_name = &item.file_name;
        let mut audio = read_audio(Path::new(file_name), self.offset_begin, self.duration_max)?;
        let channel_count = audio.data.nrows();
        let frame_count = audio.data.ncols();
        // legacy: Segmentation.cpp:47 _AudioDuration = (frameCount-1)/frameRate.
        let audio_duration = (frame_count as f64 - 1.0) / audio.sample_rate as f64;

        // Reference text is read once (the legacy `Segmentation` ctor opens the
        // file per channel for STM, but the content is the same file).
        let ref_ext = extension_of(&item.ref_seg);
        let ref_text = match ref_ext {
            RefExt::Trs => {
                // legacy: Segmentation.cpp:103,108 load_ref_from_trs -- unported.
                bail!(
                    "TRS reference `{}` not ported (Phase 4b): IMPROVEMENTS [phase4a] .trs reference loader unported",
                    item.ref_seg
                );
            }
            RefExt::None => None,
            _ => Some(std::fs::read_to_string(&item.ref_seg).ok()),
        };

        // Scored iff mode is m/M/t/T, OR i/I with a non-empty reference (`:311`).
        // The mandatory-reference check (`:302-305`) fires for m/M/t/T only.
        let is_scored_kind = matches!(mode.kind, ModeKind::Multi | ModeKind::UnitTest);
        let is_image_kind = matches!(mode.kind, ModeKind::Image);

        for ii in 0..self.nb_of_conf {
            // legacy: :260 Segmentation seg(audio, _PruningThresholds[ii]);
            let mut seg_per_chan: Vec<Segmentation> = (0..channel_count)
                .map(|_| Segmentation::new(audio_duration))
                .collect();

            // Build the per-channel reference (None when no reference is loadable)
            // and its `nb_words` WER gate. CSV yields nb_words; STM/none leave it
            // at the legacy default -1 (`Segmentation.h:70`), which suppresses WER
            // Pass 1. `_Reference` is filled identically per channel from the same
            // file, so the single CSV parse's `nb_words` is shared across channels.
            let (reference, nb_words): (Option<Vec<Segmentation>>, i64) = match (ref_ext, &ref_text)
            {
                (RefExt::Stm, Some(Some(text))) => {
                    let refs = (0..channel_count)
                        .map(|chan| {
                            load_ref_stm(
                                text,
                                chan,
                                self.offset_begin,
                                audio_duration,
                                self.exclude_nontrans,
                            )
                        })
                        .collect();
                    (Some(refs), -1)
                }
                // CSV: the legacy `load_ref_from_csv` (`Segmentation.cpp:745-806`)
                // only builds `buf` (the filename to open) for `_ChannelNb == 1`;
                // for `_ChannelNb == 2` it leaves `buf` EMPTY (`:750-752`, a
                // log-only branch -- the `buf << filename` write is missing), so
                // `ifstream("")` fails and NO `_Reference` is pushed for ANY
                // channel. Reproduced: a CSV reference loads only for single-channel
                // audio; 2+ channels get no reference at all (load-bearing legacy
                // bug, IMPROVEMENTS).
                (RefExt::Csv, Some(Some(text))) if channel_count == 1 => {
                    let (seg, nb) = load_ref_csv(
                        text,
                        self.offset_begin,
                        audio_duration,
                        self.pruning_thresholds[ii],
                    );
                    (Some(vec![seg]), nb)
                }
                _ => (None, -1),
            };

            let t = Instant::now();
            // legacy: :265-300 dispatch on _AlgoTypes[ii].getSegmentation(audio, seg).
            self.processors[ii].get_segmentation(&mut audio, &mut seg_per_chan)?;
            let dump_dir = self.processors[ii].dump_dir().to_string();

            // Mandatory-reference check (`:302-305`): _ClassificationErrors is
            // populated only when a reference exists, so "scored mode AND no
            // reference" is the legacy's empty-_ClassificationErrors bail.
            if is_scored_kind && reference.is_none() {
                bail!(
                    "Error: no valid reference segmentation was given for the file {}.\nA valid reference is mandatory when using the modes \"-m\", \"-M\", \"-t\" or \"-T\".",
                    file_name
                );
            }

            // legacy: :308-309 timing (masked in goldens; still measured).
            let time_elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
            let time_per_hour =
                time_elapsed_ms / 1000.0 / channel_count as f64 / frame_count as f64
                    * (audio.sample_rate as f64 * 3600.0);

            // The scored branch fires for m/M/t/T, or i/I with a non-empty ref.
            let scored = is_scored_kind || (is_image_kind && reference.is_some());

            // Score every channel (compute_errors sanitizes hyp in place -- the
            // legacy's getSegmentation always calls compute_errors, so the
            // speech-duration walk always reads the SANITIZED hypothesis, scored
            // or not).
            let cumulative_error = self.processors[ii].cumulative_error();
            let nb_of_classif = self.processors[ii].nb_of_classif();

            let mut chan_map: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
            for chan in 0..channel_count {
                let refc = reference.as_ref().map(|r| &r[chan]);
                let report = compute_errors(&mut seg_per_chan[chan], refc, nb_words);
                let speech_duration = speech_duration_of(&seg_per_chan[chan]);

                let row = if scored {
                    assemble_scored_row(
                        &report,
                        time_per_hour,
                        cumulative_error[chan],
                        audio_duration,
                        speech_duration,
                        nb_of_classif[chan],
                    )
                } else {
                    assemble_unscored_row(time_per_hour, audio_duration, speech_duration)
                };
                chan_map.insert(chan, row);
            }
            results.insert(ii, chan_map);

            // legacy: VRCTS write. Scored branch (`:352-356`) writes ONLY when
            // dumpDir is set; unscored branch (`:394-401`) ALWAYS writes (next to
            // the audio when dumpDir is empty). Both use the basename quirk (strip
            // the last 4 chars = extension).
            let base_last = base_from_last_slash(file_name);
            if scored {
                if !dump_dir.is_empty() {
                    let out = format!("{dump_dir}/{base_last}");
                    write_vrcts(&seg_per_chan[0], "", "", Path::new(&out))?;
                }
            } else {
                let out = if !dump_dir.is_empty() {
                    format!("{dump_dir}/{base_last}")
                } else {
                    // No dumpDir: strip the extension from the FULL path (`:399`).
                    strip_last_4(file_name)
                };
                write_vrcts(&seg_per_chan[0], "", "", Path::new(&out))?;
            }

            // legacy: :403 audio.reset() before the next config.
            audio.reset();
        }

        Ok(results)
    }
}

/// The extension dispatch of the legacy `Segmentation` ctor (`:72-110`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum RefExt {
    Stm,
    Csv,
    Trs,
    None,
}

/// Classify `ref_seg` by its trailing 4 chars (`:73`), matching the legacy
/// `.substr(size-4)` compare: an empty/short name is no reference; `.stm`/`.csv`
/// dispatch to their loaders; `.xml` (VRCTS) is not wired here (no corpus in 4a
/// uses it -- deferred with the TRS path); everything else (incl. `.trs`) is TRS.
fn extension_of(ref_seg: &str) -> RefExt {
    if ref_seg.len() <= 4 {
        // legacy: :106 size > 0 -> TRS; size 0 -> no reference.
        return if ref_seg.is_empty() {
            RefExt::None
        } else {
            RefExt::Trs
        };
    }
    match &ref_seg[ref_seg.len() - 4..] {
        ".stm" => RefExt::Stm,
        ".csv" => RefExt::Csv,
        _ => RefExt::Trs,
    }
}

/// Speech-duration walk over the hyp segments (`:324-330`): sum
/// `next.begin - cur.begin` over SPEECH intervals (all but the `End` sentinel).
fn speech_duration_of(seg: &Segmentation) -> f64 {
    let segs = seg.segments();
    let mut total = 0.0;
    for i in 0..segs.len().saturating_sub(1) {
        if segs[i].ty == SegClass::Speech {
            total += segs[i + 1].begin - segs[i].begin;
        }
    }
    total
}

/// Basename after the last `/`, minus the last 4 chars (`:353`, `:396`): the
/// legacy `substr(last_slash+1, size-last_slash-1-4)` = component without its
/// 4-char extension.
fn base_from_last_slash(path: &str) -> String {
    let after = match path.rfind('/') {
        Some(i) => &path[i + 1..],
        None => path,
    };
    strip_last_4(after)
}

/// Drop the last 4 chars (the `.ext`), as the legacy `substr(0, size-4)` (`:399`).
fn strip_last_4(s: &str) -> String {
    if s.len() >= 4 {
        s[..s.len() - 4].to_string()
    } else {
        String::new()
    }
}

/// The scored 18-column row (`:313-350`): errors + timing + cost + duration +
/// speech walk + WER (cols 7-13) + the two 0.0 LID slots + lid_nb_of_classif (0
/// in 4a) + nb_of_classif.
fn assemble_scored_row(
    report: &ScoreReport,
    time_per_hour: f64,
    cumulative_error: f64,
    audio_duration: f64,
    speech_duration: f64,
    nb_of_classif: i64,
) -> Vec<f64> {
    let speech = report.per_class[SegClass::Speech as usize];
    let mut global_error_rate = 0.0;
    // legacy: :317-319 j from OTHER up to (exclusive) EXCLUDED.
    for j in (SegClass::Other as usize)..(SegClass::Excluded as usize) {
        global_error_rate += report.per_class[j].error_rate;
    }
    let wer = report.wer.unwrap_or_default();

    vec![
        100.0 * speech.pfa,        // 0
        100.0 * speech.pmiss,      // 1
        100.0 * global_error_rate, // 2
        time_per_hour,             // 3
        cumulative_error,          // 4
        audio_duration,            // 5
        speech_duration,           // 6
        wer.nb_words as f64,       // 7
        wer.corrects as f64,       // 8
        wer.subs as f64,           // 9
        wer.ins as f64,            // 10
        wer.dels as f64,           // 11
        wer.coverage_penalty,      // 12
        wer.delay_penalty,         // 13
        0.0,                       // 14 (no LID in 4a)
        0.0,                       // 15
        0.0,                       // 16 lid_nb_of_classif
        nb_of_classif as f64,      // 17
    ]
}

/// The unscored 18-column row (`:358-392`): zeros for the error cols and the two
/// counters, timing/duration/speech walk still real, WER default (all zero).
fn assemble_unscored_row(
    time_per_hour: f64,
    audio_duration: f64,
    speech_duration: f64,
) -> Vec<f64> {
    vec![
        0.0,             // 0
        0.0,             // 1
        0.0,             // 2
        time_per_hour,   // 3
        0.0,             // 4
        audio_duration,  // 5
        speech_duration, // 6
        0.0,             // 7 nb_words (WER default)
        0.0,             // 8 corrects
        0.0,             // 9 subs
        0.0,             // 10 ins
        0.0,             // 11 dels
        0.0,             // 12 coverage
        0.0,             // 13 delay
        0.0,             // 14 LID slot
        0.0,             // 15 LID slot
        0.0,             // 16 lid_nb_of_classif (0)
        0.0,             // 17 nb_of_classif (0)
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ModeKind;

    fn solo_mode() -> Mode {
        Mode {
            kind: ModeKind::Solo,
            verbose: false,
        }
    }

    fn ref_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
    }

    fn load_config(rel: &str) -> IndexMap<String, String> {
        let text = std::fs::read_to_string(ref_dir().join(rel)).unwrap();
        crate::legacy_config::parse_legacy_config(&text)
    }

    /// Add the BagOfProcessors-level keys the phase2b fixture configs lack (they
    /// only carry the per-driver `<PREFIX>_*` keys, not the top-level
    /// `numOuterThreads`/`Algo_choice` that only exist once configs are combined
    /// into a real multi-config run).
    fn with_bag_keys(mut m: IndexMap<String, String>, algo: i32) -> IndexMap<String, String> {
        m.insert("numOuterThreads".to_string(), "1".to_string());
        m.insert("Algo_choice".to_string(), algo.to_string());
        m
    }

    #[test]
    fn constructs_algo_1_through_4() {
        let tdc = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        let ltsv = with_bag_keys(load_config("phase2b/ltsv.config"), 2);
        let mut spectral = with_bag_keys(load_config("phase0/1_worker_1.config"), 3);
        spectral.insert(
            "BLSTM_weightsFile".to_string(),
            ref_dir()
                .join("phase0/NNweights_config1.bin")
                .to_str()
                .unwrap()
                .to_string(),
        );
        let signal = with_bag_keys(load_config("phase2b/signal.config"), 4);

        for mut cfgs in [vec![tdc], vec![ltsv], vec![spectral], vec![signal]] {
            let bag = BagOfProcessors::from_configs(&mut cfgs, solo_mode()).unwrap();
            assert_eq!(bag.nb_of_conf(), 1);
        }
    }

    #[test]
    fn algo_5_bails() {
        let mut cfg = with_bag_keys(load_config("phase2b/tdc.config"), 5);
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), solo_mode()) {
            Err(e) => assert!(e.to_string().contains("Algo 5 not ported")),
            Ok(_) => panic!("expected algo 5 to bail"),
        }
    }

    #[test]
    fn file_type_nonzero_bails() {
        let mut cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        cfg.insert("File_Type".to_string(), "2".to_string());
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), solo_mode()) {
            Err(e) => assert!(e.to_string().contains("File_Type")),
            Ok(_) => panic!("expected non-wav File_Type to bail"),
        }
    }

    #[test]
    fn key_clears_applied() {
        let mut cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        cfg.insert("files".to_string(), "some_file.wav".to_string());
        cfg.insert("refsegfiles".to_string(), "some_ref.seg".to_string());
        cfg.insert("reflangfiles".to_string(), "eng".to_string());
        cfg.insert("refdialfiles".to_string(), "us".to_string());

        let mut cfgs = vec![cfg];
        let _bag = BagOfProcessors::from_configs(&mut cfgs, solo_mode()).unwrap();

        assert_eq!(cfgs[0].get("files").map(String::as_str), Some(""));
        assert_eq!(cfgs[0].get("refsegfiles").map(String::as_str), Some(""));
        assert_eq!(cfgs[0].get("reflangfiles").map(String::as_str), Some(""));
        // refdialfiles NOT cleared (legacy quirk).
        assert_eq!(cfgs[0].get("refdialfiles").map(String::as_str), Some("us"));
    }

    #[test]
    fn dispatch_vec_shapes() {
        let tdc = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        let ltsv = with_bag_keys(load_config("phase2b/ltsv.config"), 2);
        let mut spectral_cfg = with_bag_keys(load_config("phase0/1_worker_1.config"), 3);
        spectral_cfg.insert(
            "BLSTM_weightsFile".to_string(),
            ref_dir()
                .join("phase0/NNweights_config1.bin")
                .to_str()
                .unwrap()
                .to_string(),
        );
        let signal = with_bag_keys(load_config("phase2b/signal.config"), 4);

        let mut cfgs = vec![tdc, ltsv, spectral_cfg, signal];
        let bag = BagOfProcessors::from_configs(&mut cfgs, solo_mode()).unwrap();

        // algo 1 (pos 0), algo 2 (pos 1): non-NN defaults.
        assert_eq!(bag.get_weights(0).len(), 0);
        assert_eq!(bag.get_weights(1).len(), 0);
        assert_eq!(bag.is_back_prop_activated(0), vec![false]);
        assert_eq!(bag.is_back_prop_activated(1), vec![false]);
        assert_eq!(bag.get_input_statistics(0).len(), 0);
        assert_eq!(bag.get_weights_derivatives(0).len(), 0);

        // algo 3 (pos 2), algo 4 (pos 3): single-element NN vecs.
        assert_eq!(bag.get_weights(2).len(), 1);
        assert_eq!(bag.get_weights(3).len(), 1);
        assert_eq!(bag.get_input_statistics(2).len(), 1);
        assert_eq!(bag.get_input_statistics(3).len(), 1);
        assert_eq!(bag.get_weights_derivatives(2).len(), 1);
        assert_eq!(bag.get_weights_derivatives(3).len(), 1);
        // The real config-1 net's weight vector is the known 33,671 length.
        assert_eq!(bag.get_weights(2)[0].len(), 33_671);
    }
}
