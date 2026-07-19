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
use crate::engine::confusion;
use crate::engine::corpus::CorpusItem;
use crate::fast::driver::{FastSpectralSegmenter, FastTwinLid};
use crate::features::stats::InputStatistics;
use crate::tasks::lid::{BlstmSpectralLid, TwinBlstmSpectralLid};
use crate::tasks::sad::{
    BlstmSignalSegmenter, BlstmSpectralSegmenter, LtsvSegmenter, TdcSegmenter,
};
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::{
    ScoreReport, WerStats, compute_errors, load_ref_csv, load_ref_stm, load_ref_vrcts,
    write_vrcts_multichannel,
};
use crate::tasks::vrcts::VrctsPart;

/// `conf.get<int>(name)` (required, no default): missing key is an error.
fn get_i32(map: &IndexMap<String, String>, key: &str) -> Result<i32> {
    map.get(key)
        .ok_or_else(|| anyhow::anyhow!("missing required config key `{key}`"))?
        .trim()
        .parse::<i32>()
        .map_err(|e| anyhow::anyhow!("`{key}`: cannot parse as i32: {e}"))
}

/// `conf.get<double>(name, default)`: missing key -> default.
pub(crate) fn get_f64_default(
    map: &IndexMap<String, String>,
    key: &str,
    default: f64,
) -> Result<f64> {
    match map.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow::anyhow!("`{key}`: cannot parse as f64: {e}")),
    }
}

/// `conf.get<int>(name, default)`: missing key -> default.
pub(crate) fn get_i32_default(
    map: &IndexMap<String, String>,
    key: &str,
    default: i32,
) -> Result<i32> {
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

/// One per-config driver, replacing the legacy `_ConfigIndex` + 7 parallel typed
/// vectors: `processors[pos]` IS the config-`pos` driver directly (the enum tag
/// substitutes for the legacy's separate `_AlgoTypes[pos]` dispatch on WHICH
/// vector `_ConfigIndex[pos]` indexes into). Algo 0 (`VRCTSPart`, external-tool
/// adapter) landed Phase 4b Task 8; 5 (`BLSTMSpectralLID`) and 6
/// (`TwinBLSTMSpectralLID`) landed Phase 4b Task 9.
///
/// `large_enum_variant` allowed: the brief's signature is exact
/// (`Spectral(BlstmSpectralSegmenter)`, no `Box`); boxing would change the public
/// interface for a lint, not a correctness issue.
///
/// `FastSpectral` (Phase 7 Task 4) is the f32 fast-inference counterpart of
/// `Spectral` (algo 3), selected by the `Inference_Path fast` config key. `FastTwinLid`
/// (Phase 7 Task 5) is the f32 counterpart of `TwinLid` (algo 6), restricted to the
/// Mode-7 phSeq/cep LID arm. Both are inference-only: their
/// `getSegmentation`/`dumpDir`/cost/classif arms delegate to the fast driver, and their
/// TRAINING arms (weights/derivatives/stats/save/update/isBackProp) group with the
/// non-NN variants' inert defaults, since the fast path never trains (training stays
/// exact f64 -- spec S1; the fast drivers expose no trainable f64 surface). `FastTwinLid`
/// DOES surface `lid_row_data` (it writes the LID members), so the scored result row's
/// confusion columns flow exactly as the exact Twin's. Both are `Clone` (deep-copying
/// the f32 net + workspace), so the bag's `#[derive(Clone)]` still holds; the clone
/// sites (grad-check snapshot, per-lane training fan-out) are exact-path-only, so a
/// fast-variant deep copy is not on any hot path.
///
/// T6b AUDIT (Phase 7): "inert defaults" above is NOT one blanket judgment -- each
/// Vec-valued dispatch method was re-examined per its OWN call sites. `set_weights` was
/// found unsafe (a real seam caller can plausibly expect it to inject trained weights,
/// unlike the algo-0/1/2 case) and now bails loudly instead of silently discarding the
/// caller's data -- see its doc comment. `get_weights`/`get_weights_derivatives` and the
/// `save_weights`/`update_weights` save-side arms stay inert defaults -- see their doc
/// comments for why each is safe. `reset_weights_derivatives` has no dispatch here at
/// all: it lives on `Network`/layer internals, invoked automatically from within a
/// net's OWN `feed_backward`; the fast drivers never call `feed_backward` (no backward
/// implementation exists), so it is unreachable for them by construction, not by a
/// gate this file maintains.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Processor {
    Vrcts(VrctsPart),
    Tdc(TdcSegmenter),
    Ltsv(LtsvSegmenter),
    Spectral(BlstmSpectralSegmenter),
    Signal(BlstmSignalSegmenter),
    Lid(BlstmSpectralLid),
    TwinLid(TwinBlstmSpectralLid),
    FastSpectral(FastSpectralSegmenter),
    FastTwinLid(FastTwinLid),
}

/// Per-channel LID result-row data (`seg._LID*` members): the `:338-349` scored
/// (and `:380-389` unscored) branch reads them when `seg._IsLIDCorrect` is
/// non-empty -- which in the port means the config's driver is a LID algo (5/6),
/// since only those drivers ever write the members. One instance per channel.
pub(crate) struct LidRowData {
    pub cumulative_error: f64,
    pub is_correct: f64,
    /// `_LIDClassificationErrors[chan]` -- one column per class; its LENGTH grows
    /// the result row (the confusion columns inserted after col 15).
    pub classification_errors: Vec<f64>,
    pub nb_of_classif: i64,
}

impl Processor {
    /// Dispatch `Segmenter::getSegmentation` (`BagOfProcessors.cpp:267,272,...`):
    /// the enum tag replaces the legacy `_AlgoTypes[ii]` if/else ladder.
    fn get_segmentation(
        &mut self,
        audio: &mut crate::audio::Audio,
        seg_per_chan: &mut [crate::tasks::segmentation::Segmentation],
        refs: Option<&[crate::tasks::segmentation::Segmentation]>,
    ) -> Result<()> {
        use crate::tasks::segmenter::Segmenter;
        match self {
            Processor::Vrcts(s) => s.get_segmentation(audio, seg_per_chan, refs),
            Processor::Tdc(s) => s.get_segmentation(audio, seg_per_chan, refs),
            Processor::Ltsv(s) => s.get_segmentation(audio, seg_per_chan, refs),
            Processor::Spectral(s) => s.get_segmentation(audio, seg_per_chan, refs),
            Processor::Signal(s) => s.get_segmentation(audio, seg_per_chan, refs),
            Processor::Lid(s) => s.get_segmentation(audio, seg_per_chan, refs),
            Processor::TwinLid(s) => s.get_segmentation(audio, seg_per_chan, refs),
            Processor::FastSpectral(s) => s.get_segmentation(audio, seg_per_chan, refs),
            Processor::FastTwinLid(s) => s.get_segmentation(audio, seg_per_chan, refs),
        }
    }

    /// The driver's `_DumpDir` (`BagOfProcessors.cpp:268,273,...`): gates the
    /// scored-branch VRCTS write.
    fn dump_dir(&self) -> &str {
        match self {
            Processor::Vrcts(s) => s.dump_dir(),
            Processor::Tdc(s) => s.dump_dir(),
            Processor::Ltsv(s) => s.dump_dir(),
            Processor::Spectral(s) => s.dump_dir(),
            Processor::Signal(s) => s.dump_dir(),
            Processor::Lid(s) => s.dump_dir(),
            Processor::TwinLid(s) => s.dump_dir(),
            Processor::FastSpectral(s) => s.dump_dir(),
            Processor::FastTwinLid(s) => s.dump_dir(),
        }
    }

    /// Per-channel `seg._CumulativeError` (result col 4). VRCTS/TDC/LTSV are
    /// NN-free and return owned zero vecs; the NN drivers return the cost
    /// captured on the last `get_segmentation` call. Algo 5's SAD is LTSV
    /// (NN-free) so its vec is zeros too; Algo 6's SAD is a real NN.
    fn cumulative_error(&self) -> Vec<f64> {
        match self {
            Processor::Vrcts(s) => s.cumulative_error(),
            Processor::Tdc(s) => s.cumulative_error(),
            Processor::Ltsv(s) => s.cumulative_error(),
            Processor::Spectral(s) => s.cumulative_error().to_vec(),
            Processor::Signal(s) => s.cumulative_error().to_vec(),
            Processor::Lid(s) => s.cumulative_error(),
            Processor::TwinLid(s) => s.cumulative_error().to_vec(),
            // Fast SAD is forward-only: cost is not harvested, so this is all-zeros
            // (a documented divergence from the exact algo-3 driver).
            Processor::FastSpectral(s) => s.cumulative_error().to_vec(),
            // Fast Mode-7 LID: the SAD net is never run, so this is all-zeros (same as
            // the exact Twin's Mode-7 `:1421`).
            Processor::FastTwinLid(s) => s.cumulative_error().to_vec(),
        }
    }

    /// Per-channel `seg._NbOfClassif` (result col `len-1`).
    fn nb_of_classif(&self) -> Vec<i64> {
        match self {
            Processor::Vrcts(s) => s.nb_of_classif(),
            Processor::Tdc(s) => s.nb_of_classif(),
            Processor::Ltsv(s) => s.nb_of_classif(),
            Processor::Spectral(s) => s.nb_of_classif().to_vec(),
            Processor::Signal(s) => s.nb_of_classif().to_vec(),
            Processor::Lid(s) => s.nb_of_classif(),
            Processor::TwinLid(s) => s.nb_of_classif().to_vec(),
            // Fast SAD is forward-only: all-zeros (see cumulative_error).
            Processor::FastSpectral(s) => s.nb_of_classif().to_vec(),
            // Fast Mode-7 LID: the SAD net is never run -> all-zeros.
            Processor::FastTwinLid(s) => s.nb_of_classif().to_vec(),
        }
    }

    /// Per-channel LID result-row data. The legacy gate is
    /// `!seg._IsLIDCorrect.empty()` (`:338`/`:380`): only the LID drivers
    /// (algo 5/6) ever fill `_IsLIDCorrect`, so the port keys the branch off the
    /// PROCESSOR variant -- `None` for algo 0-4 (the two 0.0 slots, no confusion
    /// columns), `Some` for 5/6 (the row WIDTH grows by the class count).
    fn lid_row_data(&self, chan: usize) -> Option<LidRowData> {
        match self {
            Processor::Lid(s) => Some(LidRowData {
                cumulative_error: s.lid_cumulative_error()[chan],
                is_correct: s.is_lid_correct()[chan] as f64,
                classification_errors: s.lid_classification_errors()[chan].clone(),
                nb_of_classif: s.lid_nb_of_classif()[chan],
            }),
            Processor::TwinLid(s) => Some(LidRowData {
                cumulative_error: s.lid_cumulative_error()[chan],
                is_correct: s.is_lid_correct()[chan] as f64,
                classification_errors: s.lid_classification_errors()[chan].clone(),
                nb_of_classif: s.lid_nb_of_classif()[chan],
            }),
            // Fast Mode-7 LID writes the same LID members (langID-derived); the confusion
            // columns flow exactly as the exact Twin's. `lid_cumulative_error`/
            // `lid_nb_of_classif` are 0 (forward-only), a documented divergence.
            Processor::FastTwinLid(s) => Some(LidRowData {
                cumulative_error: s.lid_cumulative_error()[chan],
                is_correct: s.is_lid_correct()[chan] as f64,
                classification_errors: s.lid_classification_errors()[chan].clone(),
                nb_of_classif: s.lid_nb_of_classif()[chan],
            }),
            _ => None,
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
    /// Test-support capture of `save_and_update`'s per-conf algo-5/6
    /// `(errorPercLID, confusion)` (display-only in the legacy; see
    /// [`Self::print_confusion_matrix`]). Cleared at each `save_and_update`.
    #[cfg(feature = "test-support")]
    last_confusion: Vec<(f64, Array2<f64>)>,
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
    /// Algo 0 (`VRCTSPart`) is wired here since Phase 4b Task 8
    /// (`Processor::Vrcts`); Algo 5/6 (`BLSTMSpectralLID`/`TwinBLSTMSpectralLID`)
    /// since Phase 4b Task 9 (`Processor::Lid`/`Processor::TwinLid`).
    /// `File_Type` 3/4 (phSeq-N variant, mat input) remain unported -- `bail!`,
    /// IMPROVEMENTS entry; `File_Type` 0 (wav), 1 (phSeq, Task 5), and 2 (cep,
    /// Phase 6 Task 1) are all supported.
    pub fn from_configs(
        configs: &mut [IndexMap<String, String>],
        mode: Mode,
    ) -> Result<BagOfProcessors> {
        // Task 5: the fast+training rider consults the mode. Training runs only in Multi
        // mode (the `-m`/`-M` epoch loop); Image/Solo/UnitTest drive inference via
        // `get_segmentation`, so the training-shaped-config bail is Multi-gated (the brief's
        // "inference/image/solo modes unaffected"). tier2/the SAD configs carry a training
        // tail (Epochs/backprop) but run fast INFERENCE in Image mode -- must not bail.
        let is_multi_mode = mode.kind == ModeKind::Multi;

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

        // Inference_Path (Phase 7 Task 4): `exact` (default, absent) selects the exact
        // f64 drivers; `fast` selects the f32 fast-inference counterparts. A global
        // key (read from configs[0], like File_Type/Audio_offset); any other value is
        // a construction error (loud, not a silent fallback).
        let inference_path = get_string_default(&configs[0], "Inference_Path", "exact");
        let use_fast = match inference_path.as_str() {
            "exact" => false,
            "fast" => true,
            other => bail!("Inference_Path `{other}` not recognized (expected `exact` or `fast`)"),
        };

        if file_type != 0 && file_type != 1 && file_type != 2 {
            // legacy: AudioStruct phSeq-N/mat read paths (file_type 3/4) -- unported.
            // wav (0), phSeq (1), and cep (2, Phase 6 Task 1) are supported.
            bail!(
                "File_Type {file_type} not ported: only wav (0), phSeq (1), and cep (2) are supported"
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

            // Phase 7 Task 4/5: `Inference_Path fast` picks the f32 fast-inference driver
            // per config -- algo 3 (spectral SAD, Task 4) and algo 6 (Mode-7 LID Twin,
            // Task 5); everything else stays exact-only -> typed-bail. Load-weights
            // mirrors the exact arms (`from_legacy(.., None)` then `load_weights_file`).
            //
            // RIDER: the fast path is inference-only (training stays exact f64), so a
            // Multi-mode (training) run with a TRAINING-shaped config
            // (`Neural_Networks_BackPropagation_Epochs > 0` OR a `BackPropagationActivated`-on
            // net) is a config error under `Inference_Path fast` -- caught loudly here so the
            // seam can't silently drop to a no-op fast "training" run. Guards both algo 3 and
            // 6 (hardening the Task-4 surface too); inference/image/solo modes are unaffected
            // (they carry the same training-tail configs but drive inference).
            if use_fast {
                if is_multi_mode {
                    let epochs = get_i32_default(map, "Neural_Networks_BackPropagation_Epochs", 0)?;
                    let bp_sad = get_bool_default(map, "BLSTM_BackPropagationActivated", false)?;
                    let bp_lid =
                        get_bool_default(map, "BLSTM_LID_BackPropagationActivated", false)?;
                    if epochs > 0 || bp_sad || bp_lid {
                        bail!(
                            "Inference_Path fast is inference-only (training stays exact f64), but \
                             this Multi-mode run is training-shaped (Epochs {epochs}, \
                             BackPropagationActivated SAD={bp_sad}/LID={bp_lid}); train on the \
                             exact path"
                        );
                    }
                }
                let processor = match algo {
                    3 => {
                        let mut seg = FastSpectralSegmenter::from_legacy(map, None)?;
                        seg.load_weights_file(map)?;
                        Processor::FastSpectral(seg)
                    }
                    6 => {
                        let mut seg = FastTwinLid::from_legacy(map, None, None)?;
                        seg.load_weights_file(map)?;
                        Processor::FastTwinLid(seg)
                    }
                    other => bail!(
                        "Inference_Path fast is not supported for algo {other} (only algo 3 \
                         spectral SAD + algo 6 Mode-7 LID)"
                    ),
                };
                processors.push(processor);
                continue;
            }

            let processor = match algo {
                0 => Processor::Vrcts(VrctsPart::from_legacy(map)?),
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
                5 => {
                    // legacy: :43-45 BLSTMSpectralLID(conf, ...) -- the (single) net
                    // reads the same `BLSTM_*` namespace incl. `BLSTM_weightsFile`.
                    let mut seg = BlstmSpectralLid::from_legacy(map, None)?;
                    seg.load_weights_file(map)?;
                    Processor::Lid(seg)
                }
                6 => {
                    // legacy: :46-48 TwinBLSTMSpectralLID(conf, ...) -- TWO nets
                    // (`BLSTM_*` SAD + `BLSTM_LID_*` LID); `load_weights_file` applies
                    // BOTH `<prefix>_weightsFile` keys, matching the in-ctor loads.
                    let mut seg = TwinBlstmSpectralLid::from_legacy(map, None, None)?;
                    seg.load_weights_file(map)?;
                    Processor::TwinLid(seg)
                }
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
            #[cfg(feature = "test-support")]
            last_confusion: Vec::new(),
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

    /// Port of `BagOfProcessors::isBackPropActivated` (`:73-86`): algo 3/4/5
    /// return a single-element vec from the net's `isBackPropagationActivated`;
    /// algo 6 the 2-element `[sad, lid]` pair (`:80-84`); everything else (here:
    /// algo 0/1/2, the non-NN segmenters) falls through to the legacy's own
    /// default `vector<bool>(1, false)` (`:85`) -- NOT an empty vec.
    pub fn is_back_prop_activated(&self, pos: usize) -> Vec<bool> {
        match &self.processors[pos] {
            Processor::Spectral(seg) => vec![seg.is_back_prop_activated()],
            Processor::Signal(seg) => vec![seg.is_back_prop_activated()],
            Processor::Lid(seg) => vec![seg.is_back_prop_activated()],
            Processor::TwinLid(seg) => vec![
                seg.is_back_prop_activated(),
                seg.is_back_prop_activated_lid(),
            ],
            // Fast SAD is inference-only (never trains), so it groups with the non-NN
            // variants' `vector<bool>(1, false)` default -- the training dispatch never
            // reaches a fast driver (training stays exact f64).
            Processor::Vrcts(_)
            | Processor::Tdc(_)
            | Processor::Ltsv(_)
            | Processor::FastSpectral(_)
            | Processor::FastTwinLid(_) => vec![false],
        }
    }

    /// Port of `BagOfProcessors::getWeights` (`:88-101`): algo 3/4/5 return a
    /// single-element vec of the net's flat weight vector; algo 6 the paired
    /// `[regular, LID]` vec (`:95-98`); everything else (algo 0/1/2 here) falls
    /// through to the legacy's EMPTY `vector<Eigen::VectorXd>()` default
    /// (`:100`) -- an empty Vec, unlike `isBackPropActivated`'s single-`false`
    /// default.
    ///
    /// T6b AUDIT: kept as an inert default for the fast variants too (empty Vec,
    /// no bail) -- unlike `set_weights`'s pre-fix no-op, an empty READ cannot be
    /// mistaken for real data: the return value itself is the "nothing here"
    /// signal, self-describing to any caller that inspects it. This method is also
    /// called unconditionally by internal bookkeeping (the `epoch_weight_trace_lid`
    /// test-support snapshot reads `get_weights(0)` every epoch regardless of
    /// processor kind); making it fallible would ripple into that internal
    /// plumbing for no safety gain.
    pub fn get_weights(&self, pos: usize) -> Vec<Vec<f64>> {
        use crate::tasks::segmenter::Segmenter;
        match &self.processors[pos] {
            Processor::Spectral(seg) => vec![seg.get_weights()],
            Processor::Signal(seg) => vec![seg.get_weights()],
            Processor::Lid(seg) => vec![seg.get_weights()],
            Processor::TwinLid(seg) => vec![seg.get_weights(), seg.get_weights_lid()],
            // Fast SAD exposes no trainable f64 weight surface (inference-only).
            Processor::Vrcts(_)
            | Processor::Tdc(_)
            | Processor::Ltsv(_)
            | Processor::FastSpectral(_)
            | Processor::FastTwinLid(_) => Vec::new(),
        }
    }

    /// Port of `BagOfProcessors::setWeights` (`:103-114`): algo 3/4/5 forward
    /// `new_weights[0]` to the net's `setWeights`; algo 6 forwards `at(0)` to
    /// the SAD net and `at(1)` to the LID net (`:110-113`); algo 0/1/2 (no
    /// legacy `else` branch) are a no-op.
    ///
    /// T6b AUDIT (Phase 7): the fast variants USED TO share the algo-0/1/2
    /// `Ok(())` no-op arm below, but that convention does not transfer. Algo 0/1/2
    /// genuinely have no weight concept -- no plausible caller expects a
    /// VRCTS/TDC/LTSV `set_weights` to do anything. A fast driver, in contrast, IS
    /// the trainable-net counterpart of an exact algo (its weights are simply
    /// fixed after construction), so a seam caller injecting a freshly-trained
    /// pack via `set_weights` is a realistic mistake, not a misuse: the T6 SAD run
    /// silently scored 24/24 held-out files against stale seed weights this exact
    /// way, and `Ok(())` gave no signal the injection had been dropped. Bail
    /// loudly instead -- see the fast arm below for the supported mechanism.
    pub fn set_weights(&mut self, pos: usize, new_weights: &[Vec<f64>]) -> Result<()> {
        use crate::tasks::segmenter::Segmenter;
        match &mut self.processors[pos] {
            Processor::Spectral(seg) => seg.set_weights(&new_weights[0]),
            Processor::Signal(seg) => seg.set_weights(&new_weights[0]),
            Processor::Lid(seg) => seg.set_weights(&new_weights[0]),
            Processor::TwinLid(seg) => {
                seg.set_weights(&new_weights[0])?;
                seg.set_weights_lid(&new_weights[1])
            }
            // algo 0/1/2 have no weight concept at all (no legacy `else` branch) --
            // genuinely inert, matching the legacy exactly. See the T6b audit note
            // above for why this convention does NOT extend to the fast arms below.
            Processor::Vrcts(_) | Processor::Tdc(_) | Processor::Ltsv(_) => Ok(()),
            // Fast SAD/LID (`Inference_Path fast`) load weights ONLY at construction,
            // via the config-time `BLSTM_weightsFile` / `BLSTM_LID_weightsFile` keys
            // (`load_weights_file`) -- there is no settable-after-construction f64
            // weight surface. Bail loudly rather than silently discarding the
            // caller's weights (T6b: closes the silent-seam-no-op hole).
            Processor::FastSpectral(_) | Processor::FastTwinLid(_) => bail!(
                "set_weights: config {pos} runs Inference_Path `fast`, which exposes no \
                 settable-after-construction weight surface (fast drivers load weights \
                 only at construction, via the config-time BLSTM_weightsFile / \
                 BLSTM_LID_weightsFile keys). Point those keys at the trained pack and \
                 reconstruct the Engine instead of calling set_weights on a fast config."
            ),
        }
    }

    /// Port of `BagOfProcessors::getInputStatistics` (`:116-130`): algo 3/4/5
    /// return a single-element vec of the net's `InputStatistics`; algo 6 the
    /// paired `[regular, LID]` vec (`:123-126`); everything else falls through
    /// to the legacy's EMPTY default (`:128-129`).
    pub fn get_input_statistics(&self, pos: usize) -> Vec<InputStatistics> {
        match &self.processors[pos] {
            Processor::Spectral(seg) => vec![seg.input_statistics().clone()],
            Processor::Signal(seg) => vec![seg.input_statistics().clone()],
            Processor::Lid(seg) => vec![seg.input_statistics().clone()],
            Processor::TwinLid(seg) => vec![
                seg.input_statistics().clone(),
                seg.get_input_statistics_lid().clone(),
            ],
            // Fast SAD folds no input statistics (inference-only).
            Processor::Vrcts(_)
            | Processor::Tdc(_)
            | Processor::Ltsv(_)
            | Processor::FastSpectral(_)
            | Processor::FastTwinLid(_) => Vec::new(),
        }
    }

    /// Port of `BagOfProcessors::getWeightsDerivatives` (`:132-146`): algo
    /// 3/4/5 return a single-element vec of the net's `Nx2` derivative matrix;
    /// algo 6 the paired `[regular, LID]` vec (`:139-142`); everything else
    /// falls through to the legacy's EMPTY default (`:143-144`).
    ///
    /// T6b AUDIT: kept as an inert default for the fast variants too (empty Vec,
    /// no bail), DESPITE being called UNCONDITIONALLY for every conf on every file
    /// of every epoch (`corpus_processor.rs::run_epoch`'s per-file harvest loop,
    /// which does not gate on `is_back_prop_activated`) -- making this fallible
    /// would break that internal harvest for a fast conf even during a plain
    /// inference/scoring run (Solo/Image/Multi with training off), which is
    /// exactly the CI parity path this phase depends on. It is safe: the empty
    /// Vec this returns is folded into the `derivs` map at `pos` but NEVER
    /// indexed back out for a fast/non-NN `pos` (`save_weights`/`update_weights`'s
    /// matching arms don't touch `derivs[&pos]` at all for those confs), so an
    /// empty read here cannot silently corrupt a gradient sum a training caller
    /// would consume -- it is inert by construction, not merely by convention.
    pub fn get_weights_derivatives(&self, pos: usize) -> Vec<Array2<f64>> {
        match &self.processors[pos] {
            Processor::Spectral(seg) => vec![seg.get_weights_derivatives()],
            Processor::Signal(seg) => vec![seg.get_weights_derivatives()],
            Processor::Lid(seg) => vec![seg.get_weights_derivatives()],
            Processor::TwinLid(seg) => vec![
                seg.get_weights_derivatives(),
                seg.get_weights_derivatives_lid(),
            ],
            // Fast SAD produces no gradients (forward-only, inference).
            Processor::Vrcts(_)
            | Processor::Tdc(_)
            | Processor::Ltsv(_)
            | Processor::FastSpectral(_)
            | Processor::FastTwinLid(_) => Vec::new(),
        }
    }

    /// Port of `BagOfProcessors::saveWeights` (`:148-181`): per-algo save
    /// criterion, gated on beating the running `bestCost[pos]`. Algo 0/1/2 have
    /// no matching legacy `if` branch -- no-op here too. Algo 5 criterion is
    /// `badClassifLID + costLID` (`:166`); algo 6 is `cost + costLID` (`:173`)
    /// and saves BOTH nets -- the LID save via `saveWeightsLID`, which prefixes
    /// the filename with `LID_` internally (`:176`).
    /// Arity mirrors the legacy signature (`:148`) verbatim.
    ///
    /// T6b AUDIT: the fast arm's no-op is genuinely unreachable-in-spirit, not
    /// merely unexercised -- `is_back_prop_activated` hardcodes `false` for both
    /// fast variants, so a fast conf never accumulates a real save criterion, and
    /// this arm (grouped with algo 0/1/2) never reads `derivs`/`stats` regardless.
    /// Matches the legacy's own algo-0/1/2 no-branch shape; no bail needed.
    #[allow(clippy::too_many_arguments)]
    fn save_weights(
        &mut self,
        filename: &str,
        pos: usize,
        best_cost: &mut BTreeMap<usize, f64>,
        cost: f64,
        _bad_classif: f64,
        cost_lid: f64,
        bad_classif_lid: f64,
        derivs: &BTreeMap<usize, Vec<Array2<f64>>>,
        stats: &BTreeMap<usize, Vec<InputStatistics>>,
    ) -> Result<()> {
        match &mut self.processors[pos] {
            Processor::Spectral(seg) => {
                // legacy: :149-155 saveCriterion = cost (badClassif+cost is commented out).
                let save_criterion = cost;
                if best_cost[&pos] > save_criterion {
                    seg.save_weights(filename, &derivs[&pos][0], &stats[&pos][0])?;
                    best_cost.insert(pos, save_criterion);
                }
            }
            Processor::Signal(seg) => {
                // legacy: :156-162, identical criterion to algo 3.
                let save_criterion = cost;
                if best_cost[&pos] > save_criterion {
                    seg.save_weights(filename, &derivs[&pos][0], &stats[&pos][0])?;
                    best_cost.insert(pos, save_criterion);
                }
            }
            Processor::Lid(seg) => {
                // legacy: :165-171 saveCriterion = badClassifLID + costLID.
                let save_criterion = bad_classif_lid + cost_lid;
                if best_cost[&pos] > save_criterion {
                    seg.save_weights(filename, &derivs[&pos][0], &stats[&pos][0])?;
                    best_cost.insert(pos, save_criterion);
                }
            }
            Processor::TwinLid(seg) => {
                // legacy: :172-179 saveCriterion = cost + costLID; BOTH nets saved
                // under ONE gate (`saveWeightsLID` adds its own `LID_` prefix).
                let save_criterion = cost + cost_lid;
                if best_cost[&pos] > save_criterion {
                    seg.save_weights(filename, &derivs[&pos][0], &stats[&pos][0])?;
                    seg.save_weights_lid(filename, &derivs[&pos][1], &stats[&pos][1])?;
                    best_cost.insert(pos, save_criterion);
                }
            }
            // Fast SAD never trains (inference-only) -- no save gate, like algo 0/1/2.
            Processor::Vrcts(_)
            | Processor::Tdc(_)
            | Processor::Ltsv(_)
            | Processor::FastSpectral(_)
            | Processor::FastTwinLid(_) => {
                // legacy: no `if` branch for algo 0/1/2 -- no-op.
            }
        }
        Ok(())
    }

    /// Port of `BagOfProcessors::updateWeights` (`:183-205`): per-algo update
    /// criterion, unconditional (unlike `saveWeights`, no best-cost gate). Algo
    /// 0/1/2 no-op (no legacy branch). Algo 5's criterion is `costLID`
    /// (`:193`); algo 6 updates the SAD net with `cost` THEN the LID net with
    /// `costLID` (`:196-200`) -- the caller's `:465` `costLID = -1.0` no-speech
    /// gate is LIVE for both LID arms (the commented-out `saveCriterion > 0`
    /// skip at `:199-203` is dead: the update runs unconditionally, -1.0 cost
    /// included).
    ///
    /// T6b AUDIT: same verdict as `save_weights` -- the fast arm's no-op is
    /// unreachable-in-spirit (`is_back_prop_activated` hardcodes `false`), and
    /// this arm never reads `derivs` for a fast/non-NN `pos` regardless. No bail
    /// needed.
    fn update_weights(
        &mut self,
        pos: usize,
        cost: f64,
        _bad_classif: f64,
        cost_lid: f64,
        _bad_classif_lid: f64,
        derivs: &BTreeMap<usize, Vec<Array2<f64>>>,
    ) {
        match &mut self.processors[pos] {
            Processor::Spectral(seg) => {
                // legacy: :184-188 saveCriterion = cost.
                let save_criterion = cost;
                seg.update_weights(&derivs[&pos][0], save_criterion);
                #[cfg(feature = "test-support")]
                seg.record_update_cost_lid_for_test(cost_lid);
            }
            Processor::Signal(seg) => {
                // legacy: :189-193, identical criterion to algo 3.
                let save_criterion = cost;
                seg.update_weights(&derivs[&pos][0], save_criterion);
            }
            Processor::Lid(seg) => {
                // legacy: :192-194 saveCriterion = costLID.
                seg.update_weights(&derivs[&pos][0], cost_lid);
            }
            Processor::TwinLid(seg) => {
                // legacy: :195-200 updateWeights(cost) THEN updateWeightsLID(costLID).
                seg.update_weights(&derivs[&pos][0], cost);
                seg.update_weights_lid(&derivs[&pos][1], cost_lid);
            }
            // Fast SAD never trains (inference-only) -- no-op, like algo 0/1/2.
            Processor::Vrcts(_)
            | Processor::Tdc(_)
            | Processor::Ltsv(_)
            | Processor::FastSpectral(_)
            | Processor::FastTwinLid(_) => {
                // legacy: no `if` branch for algo 0/1/2 -- no-op.
            }
        }
    }

    /// Port of `BagOfProcessors::PrintConfusionMatrix` (`:473-598`): LID confusion
    /// matrix + normalized error percentage, reachable only for algo 5/6 (the
    /// `saveAndUpdate` call site is gated on `_AlgoTypes[ii] == 5 || == 6`,
    /// `:438-441`). Wired into `save_and_update` since Task 9.
    ///
    /// USAGE FINDING (Task 9 source re-read of `:436-443,468`): the returned
    /// `errorPercLID` feeds ONLY the `:443` `cout` status line, and the
    /// `outputConfusion` display string is printed at `:468` -- NEITHER is
    /// stored on any member, result row, mem matrix, or artifact. The call has
    /// NO side effects on parity-relevant state (the matrix is a function
    /// local). The port therefore invokes it for flow parity and exposes the
    /// `(error, matrix)` pair only as a test-support observation
    /// ([`Self::last_confusion_for_test`]); nothing production-visible consumes
    /// the values -- inventing storage would deviate from the legacy.
    ///
    /// Delegates to [`confusion::confusion_from_results`] for the sentinel
    /// decode + argmax accumulation (`:501-535`) and the row-normalized error
    /// (the LIVE `Confusion2String` call at `:540`, ported as
    /// [`confusion::confusion_error`]).
    ///
    /// CORRECTION (Task 1 source re-read) to this doc comment's prior claim:
    /// the returned `error` IS the row-normalized `error/classNb` aggregate,
    /// NOT a raw off-diagonal count. `Confusion2String`'s call at `:540` is
    /// LIVE code (not part of the dead duplicate block at `:541-593`, which
    /// re-derives the same normalization via `cout` instead of `error +=` and
    /// is never executed); its `return error/classNb` (`Helpers.hpp:438`) is
    /// what `:597`'s `return error;` hands back unchanged (the commented-out
    /// `return error/classNb;` at `:596` would have double-divided).
    fn print_confusion_matrix(
        &self,
        results_mat: &Array2<f64>,
        _config_nb: usize,
    ) -> (f64, Array2<f64>) {
        let (matrix, error) = confusion::confusion_from_results(results_mat);
        (error, matrix)
    }

    /// Test-observation hook (Task 9, no legacy counterpart): the last
    /// `(errorPercLID, confusion)` pair per conf produced by `save_and_update`'s
    /// algo-5/6 confusion block -- the legacy prints both and stores neither, so
    /// this is the only way a golden can pin the aggregation.
    #[cfg(feature = "test-support")]
    pub fn last_confusion_for_test(&self) -> &[(f64, Array2<f64>)] {
        &self.last_confusion
    }

    /// Port of `BagOfProcessors::saveAndUpdate` (`:409-471`): per-config
    /// column-sum/mean aggregation over the file x channel result rows, cost/
    /// badClassif/costLID/badLIDClassif derivation, WER percent scaling, the
    /// `costMem`/`badClassifMem`/`costLIDMem`/`badClassifLIDMem` row writes, the
    /// `bestNNWeight_<pos+1>_<filename>` save, and the `costLID = -1.0` gate
    /// (`:465`) applied AFTER `save_weights` and BEFORE `update_weights` -- order
    /// is load-bearing (the save criterion sees the real costLID; the update
    /// criterion sees the gated one).
    ///
    /// Column sums/means use explicit ascending row loops (the repo's
    /// ascending-loop product contract), NOT `ndarray::sum_axis`/`mean_axis`.
    ///
    /// `totalSignalDuration`/`totalSpeechDuration` (`:445-446`) are computed
    /// (needed for the `:465` gate) but the h/min/s display prints (`:447-456`)
    /// are display-only and dropped, per the brief.
    ///
    /// "N files" (`resultsperConf[ii].rows()`, `:449`) is a MISNOMER in the
    /// legacy print: the row count is file x channel rows (one row per channel
    /// per file), not per-file -- display-only anyway, but the underlying MEANS
    /// (`:410-411`) are genuinely averages over file x channel rows, which is
    /// what makes `badClassif`/`badLIDClassif` correct per-frame-class averages
    /// rather than per-file averages.
    ///
    /// Arity matches the brief's exact signature; the legacy `saveAndUpdate`
    /// (`:409`) similarly takes 9 parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn save_and_update(
        &mut self,
        filename: &str,
        results_per_conf: &[Array2<f64>],
        best_cost: &mut BTreeMap<usize, f64>,
        derivs: &BTreeMap<usize, Vec<Array2<f64>>>,
        stats: &BTreeMap<usize, Vec<InputStatistics>>,
        cost_mem_row: &mut [f64],
        bad_classif_row: &mut [f64],
        cost_lid_row: &mut [f64],
        bad_classif_lid_row: &mut [f64],
    ) -> Result<()> {
        #[cfg(feature = "test-support")]
        self.last_confusion.clear();

        for ii in 0..self.nb_of_conf {
            let m = &results_per_conf[ii];
            let (rows, cols) = m.dim();

            // legacy: :411-412 colwise sum/mean, ascending row loop (product contract).
            let mut sums = vec![0.0_f64; cols];
            for r in 0..rows {
                for c in 0..cols {
                    sums[c] += m[[r, c]];
                }
            }
            let mut means = vec![0.0_f64; cols];
            for c in 0..cols {
                means[c] = sums[c] / rows as f64;
            }

            // legacy: :413-414 cost = sums(4), guarded /= sums(last).
            let mut cost = sums[4];
            if sums[cols - 1] > 0.0 {
                cost /= sums[cols - 1];
            }
            // legacy: :415 badClassif = means(2).
            let bad_classif = means[2];
            // legacy: :416-417 costLID = sums(14), guarded /= sums(cols-2).
            let mut cost_lid = sums[14];
            if sums[cols - 2] > 0.0 {
                cost_lid /= sums[cols - 2];
            }
            // legacy: :418 badLIDClassif = 100-means(15).
            let bad_lid_classif = 100.0 - means[15];

            // legacy: :419-433 WER percent scaling, guarded by nbOfWords > 0.
            let nb_of_words = sums[7];
            let mut wer_correct = 100.0 * sums[8];
            let mut wer_subs = 100.0 * sums[9];
            let mut wer_ins = 100.0 * sums[10];
            let mut wer_dels = 100.0 * sums[11];
            let mut wer_coverage_penalty = 100.0 * sums[12];
            let mut wer_delay_penalty = 100.0 * sums[13];
            if nb_of_words > 0.0 {
                wer_correct /= nb_of_words;
                wer_subs /= nb_of_words;
                wer_ins /= nb_of_words;
                wer_dels /= nb_of_words;
                wer_coverage_penalty /= nb_of_words;
                wer_delay_penalty /= nb_of_words;
            }
            let _wer = wer_subs + wer_ins + wer_dels;
            let _ = (wer_correct, wer_coverage_penalty, wer_delay_penalty);

            // legacy: :435-441 confusion block, algo 5/6 only: slice the confusion
            // columns `block(0, 16, rows, cols-2-16)` and feed PrintConfusionMatrix.
            // errorPercLID + the confusion string are DISPLAY-ONLY in the legacy
            // (`:443` cout / `:468` cout) -- nothing parity-relevant is stored (see
            // print_confusion_matrix's usage finding), so outside test-support the
            // pair is dropped after the call.
            if self.algo_types[ii] == 5 || self.algo_types[ii] == 6 {
                let conf_cols = cols - 2 - 16;
                let mut block = Array2::<f64>::zeros((rows, conf_cols));
                for r in 0..rows {
                    for c in 0..conf_cols {
                        block[[r, c]] = m[[r, 16 + c]];
                    }
                }
                let confusion_pair = self.print_confusion_matrix(&block, ii + 1);
                #[cfg(feature = "test-support")]
                self.last_confusion.push(confusion_pair);
                #[cfg(not(feature = "test-support"))]
                let _ = confusion_pair;
            }

            // legacy: :445-446 total durations, needed for the :465 gate.
            let total_speech_duration = means[6] * rows as f64;

            // legacy: :457-460 row writes.
            cost_mem_row[ii] = cost;
            bad_classif_row[ii] = bad_classif;
            cost_lid_row[ii] = cost_lid;
            bad_classif_lid_row[ii] = bad_lid_classif;

            // legacy: :462-464 bestNNWeight_<pos+1>_<filename> compose.
            let save_filename = format!("bestNNWeight_{}_{filename}", ii + 1);
            // legacy: :464 saveWeights call -- BEFORE the :465 costLID override.
            self.save_weights(
                &save_filename,
                ii,
                best_cost,
                cost,
                bad_classif,
                cost_lid,
                bad_lid_classif,
                derivs,
                stats,
            )?;

            // legacy: :465 costLID = -1.0 when totalSpeechDuration < 1e-3. ORDER:
            // after save_weights, before update_weights.
            if total_speech_duration < 1e-3 {
                cost_lid = -1.0;
            }
            // legacy: :466 updateWeights call -- AFTER the override.
            self.update_weights(ii, cost, bad_classif, cost_lid, bad_lid_classif, derivs);
        }
        Ok(())
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
        // No reference is threaded here (the test-oracle path drives one config's
        // hypothesis only); the NN drivers run the no-target forward path.
        self.processors[pos].get_segmentation(audio, seg_per_chan, None)
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
    /// `.xml` -> [`load_ref_vrcts`] per channel (VRCTS reference, channel-sliced
    /// by the `ch=` attribute; Phase 6 Task 2b), `.trs` -> `bail!` (unported),
    /// anything else / empty -> no reference.
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
        let mut audio = read_audio(
            Path::new(file_name),
            self.offset_begin,
            self.duration_max,
            self.file_type,
        )?;
        // legacy: AudioStruct ctor sets _LangIndex/_Weight from the CorpusItem
        // (AudioStruct.cpp:53,60) -- see apply_corpus_item's doc for the
        // placement deviation.
        apply_corpus_item(&mut audio, item);
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
                // CSV reference. FIXED (phase 5, F6): the legacy `load_ref_from_csv`
                // (`Segmentation.cpp:745-806`) built the file-to-open string `buf` ONLY
                // in the `_ChannelNb == 1` branch (`:748-749` `buf << filename`); the
                // `_ChannelNb == 2` branch (`:750-752`) emitted a "wrong path" LOG line
                // and NEVER wrote `buf`, so `ifstream("")` failed and NO `_Reference`
                // was pushed for ANY channel -- a CSV reference silently loaded for mono
                // only, and a scored stereo run bailed on the mandatory-reference gate.
                // The port now loads the reference for EVERY channel, the way STM refs
                // already do. CSV is channel-independent (no per-channel column, unlike
                // STM), so the file is parsed ONCE and the segmentation cloned per
                // channel; `nb_words` (the dead WER surface, KEEP) stays the single
                // parse's count, shared across channels. The C++/resurrected-binary
                // oracles still drop the stereo reference; the port diverges by design.
                // See IMPROVEMENTS.md ([phase4a] CSV reference loads ONLY ...).
                (RefExt::Csv, Some(Some(text))) => {
                    let (seg, nb) = load_ref_csv(
                        text,
                        self.offset_begin,
                        audio_duration,
                        self.pruning_thresholds[ii],
                    );
                    let refs = (0..channel_count).map(|_| seg.clone()).collect();
                    (Some(refs), nb)
                }
                // VRCTS (.xml) reference. Port of `Segmentation::load_ref_from_vrcts`
                // (`Segmentation.cpp:808-829`), the ctor `.xml` branch (`:89-100`) -- a
                // DIFFERENT legacy function from `load_from_vrcts` (which the port's
                // `load_vrcts` mirrors for `VrctsPart`). Wired here in Phase 6 Task 2b to
                // unblock scored SAD training on the corpus `.part.xml` references. Unlike
                // the STM/CSV clones, the reference is CHANNEL-SLICED by the 1-based
                // `ch="N"` attribute (segment -> channel N-1), so `load_ref_vrcts` is
                // called per channel and each channel keeps only its own `ch` segments.
                // Windowed on `audio_duration` (the audio frame count, `_AudioDuration`),
                // NOT the embedded `<Channel sigdur>` -- a `.part.xml` sigdur can exceed a
                // `_DurationMax`-capped audio. `nb_words` stays the -1 default (WER Pass 1
                // suppressed), like STM. See IMPROVEMENTS.md ([phase4a] CLOSED (phase 6):
                // `.xml` (VRCTS) reference loading).
                (RefExt::Xml, Some(Some(text))) => {
                    let refs = (0..channel_count)
                        .map(|chan| load_ref_vrcts(text, chan, self.offset_begin, audio_duration))
                        .collect();
                    (Some(refs), -1)
                }
                _ => (None, -1),
            };

            let t = Instant::now();
            // legacy: :265-300 dispatch on _AlgoTypes[ii].getSegmentation(audio, seg).
            // Thread the per-channel reference (the legacy `seg._Reference`) so the NN
            // drivers can build training targets (`getTargets`, gated on a non-empty
            // reference). Non-NN drivers (TDC/LTSV) ignore it. `None` when no reference
            // was loadable -- the drivers then run the no-target forward path.
            self.processors[ii].get_segmentation(
                &mut audio,
                &mut seg_per_chan,
                reference.as_deref(),
            )?;
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
                // legacy: :338/:380 `!seg._IsLIDCorrect.empty()` -- LID slots live for
                // algo 5/6 (the row WIDTH grows by the class count), else two 0.0s.
                let lid = self.processors[ii].lid_row_data(chan);

                let row = if scored {
                    assemble_scored_row(
                        &report,
                        time_per_hour,
                        cumulative_error[chan],
                        audio_duration,
                        speech_duration,
                        nb_of_classif[chan],
                        lid.as_ref(),
                    )
                } else {
                    assemble_unscored_row(
                        time_per_hour,
                        audio_duration,
                        speech_duration,
                        lid.as_ref(),
                    )
                };
                chan_map.insert(chan, row);
            }
            results.insert(ii, chan_map);

            // legacy: VRCTS write. Scored branch (`:352-356`) writes ONLY when
            // dumpDir is set; unscored branch (`:394-401`) ALWAYS writes (next to
            // the audio when dumpDir is empty). Both use the basename quirk (strip
            // the last 4 chars = extension) to build `basefilename`, then call
            // `seg.toFile_VRCTS(basefilename)` which fans out one document per
            // channel (`<basefilename>_chan_<n>.xml` for `_ChannelNb > 1`, else
            // `<basefilename>.xml`; `Segmentation.cpp:543-590`).
            //
            // `name`/`path` inside `toFile_VRCTS` are derived from
            // `_AudioFilename` NOT from `basefilename`: `name` (filenameShort) is
            // the last path component of the audio file minus its 4-char
            // extension (`:553-561`), `path` is the full audio filename (`:570`).
            // Multi-channel fan-out closed in Task 8 against the real compiled
            // `toFile_VRCTS` byte golden (see IMPROVEMENTS.md).
            // Both `basefilename` (:352-401, the dumpDir-relative write target) and
            // `name` passed to `toFile_VRCTS` derive from the SAME legacy expression
            // (`_AudioFilename` basename minus its extension) -- one binding for both.
            let vrcts_base = base_from_last_slash(file_name);
            let vrcts_path = file_name;
            if scored {
                if !dump_dir.is_empty() {
                    let out = format!("{dump_dir}/{vrcts_base}");
                    write_vrcts_multichannel(
                        &seg_per_chan,
                        &vrcts_base,
                        vrcts_path,
                        Path::new(&out),
                    )?;
                }
            } else {
                let out = if !dump_dir.is_empty() {
                    format!("{dump_dir}/{vrcts_base}")
                } else {
                    // No dumpDir: strip the extension from the FULL path (`:399`).
                    strip_last_4(file_name)
                };
                write_vrcts_multichannel(&seg_per_chan, &vrcts_base, vrcts_path, Path::new(&out))?;
            }

            // legacy: :403 audio.reset() before the next config.
            audio.reset();
        }

        Ok(results)
    }
}

/// legacy: `AudioStruct::AudioStruct(double, double, int, CorpusItem&)`
/// (AudioStruct.cpp:53,60 -- and the four sibling per-`file_type` copy sites at
/// `:141/147`, `:186/192`, `:260/266`, `:328/333`): `_LangIndex =
/// corpusItem.getClassOfFile()`, `_Weight = corpusItem.getWeightOfFile()`. In
/// the legacy these assignments run INSIDE the `AudioStruct` ctor, which takes
/// the owning `CorpusItem` directly. This port's `read_audio` (Phase 1)
/// predates the corpus/bag wiring and has no `CorpusItem` in scope, so the bag
/// driver sets the two fields here, immediately after `read_audio` returns --
/// a documented placement deviation (same audio object, same point before any
/// other use), not a behavior change. Extracted as its own function so the
/// setter path is unit-testable without going through the full
/// `segmentation_function` flow.
fn apply_corpus_item(audio: &mut Audio, item: &CorpusItem) {
    audio.lang_index = item.class_index;
    audio.weight = item.weight;
    // legacy: AudioStruct.cpp:51-52,139-140 `_AudioFilename`/`_RefSegFilename`
    // set from the CorpusItem, same placement deviation as lang_index/weight
    // above. Consumed by VrctsPart (Algo 0, Phase 4b Task 8).
    audio.audio_file_name = item.file_name.clone();
    audio.ref_seg_file_name = item.ref_seg.clone();
}

/// The extension dispatch of the legacy `Segmentation` ctor (`:72-110`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum RefExt {
    Stm,
    Csv,
    Xml,
    Trs,
    None,
}

/// Classify `ref_seg` by its trailing 4 chars (`:73`), matching the legacy
/// `.substr(size-4)` compare: an empty/short name is no reference; `.stm`/`.csv`/
/// `.xml` dispatch to their loaders (`.xml` -> VRCTS reference via
/// [`load_ref_vrcts`], Phase 6 Task 2b); everything else (incl. `.trs`) is TRS,
/// still bailed (unported -- IMPROVEMENTS.md `[phase4a] .trs reference loader`).
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
        ".xml" => RefExt::Xml,
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

/// The scored result row (`:313-350`): errors + timing + cost + duration +
/// speech walk + WER (cols 7-13) + the LID block + lid_nb_of_classif +
/// nb_of_classif. WIDTH: 18 columns for algo 0-4 (`lid == None`: the two 0.0
/// LID slots, `:344-347`); `18 + classNb` for algo 5/6 (`lid == Some`:
/// `[14]=_LIDCumulativeError`, `[15]=_IsLIDCorrect`, then one confusion column
/// per class from `_LIDClassificationErrors`, `:338-343`). The final two
/// columns are ALWAYS `[len-2]=_LIDNbOfClassif`, `[len-1]=_NbOfClassif`
/// (`:348-349`, pushed OUTSIDE the LID gate).
fn assemble_scored_row(
    report: &ScoreReport,
    time_per_hour: f64,
    cumulative_error: f64,
    audio_duration: f64,
    speech_duration: f64,
    nb_of_classif: i64,
    lid: Option<&LidRowData>,
) -> Vec<f64> {
    let speech = report.per_class[SegClass::Speech as usize];
    let mut global_error_rate = 0.0;
    // legacy: :317-319 j from OTHER up to (exclusive) EXCLUDED.
    for j in (SegClass::Other as usize)..(SegClass::Excluded as usize) {
        global_error_rate += report.per_class[j].error_rate;
    }
    // No WER Pass 1 (STM/no reference) -> the legacy WordErrorRate CONSTRUCTOR
    // default (`_NbWords = -1`, rest 0), NOT `WerStats::default()` (nb_words 0).
    // The nb_words result column is -1 in that case (Phase 4a tier-1 golden).
    let wer = report.wer.unwrap_or(WerStats::legacy_default());

    let mut row = vec![
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
    ];
    push_lid_block(&mut row, lid);
    // legacy: :348-349 -- the two counters, outside the LID gate.
    row.push(lid.map_or(0.0, |l| l.nb_of_classif as f64)); // len-2 _LIDNbOfClassif
    row.push(nb_of_classif as f64); // len-1 _NbOfClassif
    row
}

/// The unscored result row (`:358-392`): zeros for the error cols and the two
/// counters, timing/duration/speech walk still real. The WER columns (7-13) push
/// `seg._WordErrorRate[chan]` UNCHANGED (Pass 1 never ran), which is the legacy
/// WordErrorRate constructor default: `_NbWords = -1`, everything else 0. So col 7
/// (nb_words) is -1, NOT 0 (Phase 4a tier-1 golden). The LID block (`:380-389`)
/// is IDENTICAL to the scored branch's (the members are real either way -- the
/// `:383` `.size()` vs `:341` `.cols()` difference is vacuous for a row vector),
/// but the two trailing counters are hard 0s (`:390-391`).
fn assemble_unscored_row(
    time_per_hour: f64,
    audio_duration: f64,
    speech_duration: f64,
    lid: Option<&LidRowData>,
) -> Vec<f64> {
    let wer = WerStats::legacy_default();
    let mut row = vec![
        0.0,                  // 0
        0.0,                  // 1
        0.0,                  // 2
        time_per_hour,        // 3
        0.0,                  // 4
        audio_duration,       // 5
        speech_duration,      // 6
        wer.nb_words as f64,  // 7 nb_words (legacy default -1)
        wer.corrects as f64,  // 8 corrects
        wer.subs as f64,      // 9 subs
        wer.ins as f64,       // 10 ins
        wer.dels as f64,      // 11 dels
        wer.coverage_penalty, // 12 coverage
        wer.delay_penalty,    // 13 delay
    ];
    push_lid_block(&mut row, lid);
    row.push(0.0); // len-2 (:390 -- hard 0, unlike the scored branch)
    row.push(0.0); // len-1 (:391)
    row
}

/// The `:338-347` LID block shared by both branches: `Some` pushes
/// `[_LIDCumulativeError, _IsLIDCorrect, confusion col per class]`; `None`
/// pushes the two 0.0 slots (no confusion columns -- the width difference).
fn push_lid_block(row: &mut Vec<f64>, lid: Option<&LidRowData>) {
    match lid {
        Some(l) => {
            row.push(l.cumulative_error); // 14
            row.push(l.is_correct); // 15
            row.extend_from_slice(&l.classification_errors); // 16..16+classNb
        }
        None => {
            row.push(0.0); // 14
            row.push(0.0); // 15
        }
    }
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

    // === apply_corpus_item_sets_lang_index_and_weight =======================
    // Unit test for the extracted setter helper (Task 2): asserts the bag's
    // AudioStruct-ctor-equivalent copy (class_index -> lang_index, weight ->
    // weight) directly, without going through the full segmentation_function
    // flow (which requires a real decodable wav).
    #[test]
    fn apply_corpus_item_sets_lang_index_and_weight() {
        let mut audio =
            crate::audio::read_audio(&ref_dir().join("phase1/excerpt_2ch_8k.wav"), 0.0, 0.1, 0)
                .unwrap();
        assert_eq!(audio.lang_index, -1, "read_audio default");
        assert_eq!(audio.weight, 1.0, "read_audio default");

        let item = CorpusItem {
            file_name: "irrelevant.wav".to_string(),
            ref_seg: String::new(),
            language: "chi".to_string(),
            dialect: "man".to_string(),
            class_index: 1,
            file_id: 1,
            weight: 0.75,
        };
        apply_corpus_item(&mut audio, &item);
        assert_eq!(audio.lang_index, 1);
        assert_eq!(audio.weight, 0.75);
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

    /// Task 9: algo 5/6 construct through the bag (weightsFile keys cleared --
    /// the committed configs carry unreachable legacy paths; the nets default-
    /// init, which is all construction needs). The dispatch Vec shapes are the
    /// legacy `:73-146` arms: algo 5 single-element via the (base-class) net,
    /// algo 6 the paired `[sad, lid]` 2-element vecs.
    #[test]
    fn constructs_algo_5_and_6_and_dispatch_shapes() {
        let mut lid5 = with_bag_keys(load_config("phase4b/lid5.config"), 5);
        lid5.insert("BLSTM_weightsFile".to_string(), String::new());
        let mut twin = with_bag_keys(load_config("phase4b/twin_mode0.config"), 6);
        twin.insert("BLSTM_weightsFile".to_string(), String::new());
        twin.insert("BLSTM_LID_weightsFile".to_string(), String::new());

        let mut cfgs = vec![lid5, twin];
        let mut bag = BagOfProcessors::from_configs(&mut cfgs, solo_mode()).unwrap();
        assert_eq!(bag.nb_of_conf(), 2);
        assert_eq!(bag.algo_type(0), 5);
        assert_eq!(bag.algo_type(1), 6);

        // Algo 5 (pos 0): single-element vecs; backprop mirrors the config.
        assert_eq!(bag.get_weights(0).len(), 1);
        assert_eq!(bag.is_back_prop_activated(0).len(), 1);
        assert_eq!(bag.get_input_statistics(0).len(), 1);
        assert_eq!(bag.get_weights_derivatives(0).len(), 1);

        // Algo 6 (pos 1): 2-element [sad, lid] vecs.
        let w = bag.get_weights(1);
        assert_eq!(w.len(), 2);
        assert_ne!(w[0].len(), w[1].len(), "SAD and LID nets are distinct");
        assert_eq!(bag.is_back_prop_activated(1).len(), 2);
        assert_eq!(bag.get_input_statistics(1).len(), 2);
        // Paired [sad, lid] derivs (Nx2 buffers are sized lazily on the first
        // reset/backward, so only the VEC shape is a construction-time contract).
        assert_eq!(bag.get_weights_derivatives(1).len(), 2);

        // setWeights round-trip: algo 6 sets BOTH nets from [at(0), at(1)].
        let mut new_w = w.clone();
        new_w[0][0] += 1.0;
        new_w[1][0] += 2.0;
        bag.set_weights(1, &new_w).unwrap();
        let after = bag.get_weights(1);
        assert_eq!(after[0][0].to_bits(), new_w[0][0].to_bits());
        assert_eq!(after[1][0].to_bits(), new_w[1][0].to_bits());
    }

    /// The scored/unscored row WIDTH contract (`:338-349`/`:380-391`): algo 0-4
    /// rows are 18 cols (two 0.0 LID slots); algo 5/6 rows grow to
    /// `18 + classNb` (confusion columns inserted after col 15), with
    /// `[len-2]=_LIDNbOfClassif`, `[len-1]=_NbOfClassif` in BOTH widths.
    #[test]
    fn lid_row_width_and_slots() {
        let report = ScoreReport {
            per_class: [Default::default(); 23],
            label_counts: [0.0; 23],
            wer: None,
        };
        let lid = LidRowData {
            cumulative_error: 7.5,
            is_correct: 100.0,
            classification_errors: vec![210.0, 30.0, 60.0], // classNb = 3
            nb_of_classif: 42,
        };

        let plain = assemble_scored_row(&report, 1.0, 2.0, 3.0, 4.0, 9, None);
        assert_eq!(plain.len(), 18);
        assert_eq!(plain[14], 0.0);
        assert_eq!(plain[15], 0.0);
        assert_eq!(plain[16], 0.0); // len-2 lid_nb_of_classif
        assert_eq!(plain[17], 9.0); // len-1 nb_of_classif

        let wide = assemble_scored_row(&report, 1.0, 2.0, 3.0, 4.0, 9, Some(&lid));
        assert_eq!(wide.len(), 21, "18 + classNb(3)");
        assert_eq!(wide[14], 7.5);
        assert_eq!(wide[15], 100.0);
        assert_eq!(&wide[16..19], &[210.0, 30.0, 60.0]);
        assert_eq!(wide[19], 42.0); // len-2
        assert_eq!(wide[20], 9.0); // len-1

        // Unscored branch: same LID block, hard-0 trailing counters (:390-391).
        let unscored = assemble_unscored_row(1.0, 3.0, 4.0, Some(&lid));
        assert_eq!(unscored.len(), 21);
        assert_eq!(unscored[14], 7.5);
        assert_eq!(&unscored[16..19], &[210.0, 30.0, 60.0]);
        assert_eq!(unscored[19], 0.0);
        assert_eq!(unscored[20], 0.0);
    }

    #[test]
    fn file_type_2_allowed() {
        // Phase 6 Task 1: File_Type 2 (cep) is now a supported gate value. Re-pins the
        // former `file_type_2_bails`. Construction doesn't touch read_audio (the gate runs
        // before the per-config driver loop), so a plain TDC config with the gate flipped
        // must succeed.
        let mut cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        cfg.insert("File_Type".to_string(), "2".to_string());
        let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), solo_mode())
            .expect("File_Type 2 (cep) must be accepted by the gate");
        assert_eq!(bag.file_type(), 2);
    }

    #[test]
    fn file_type_3_bails() {
        // File_Type 3/4 (phSeq-N variant, mat) stay unported -- the gate still bails.
        let mut cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        cfg.insert("File_Type".to_string(), "3".to_string());
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), solo_mode()) {
            Err(e) => assert!(e.to_string().contains("File_Type")),
            Ok(_) => panic!("expected File_Type 3 (unported) to bail"),
        }
    }

    #[test]
    fn file_type_1_allowed() {
        // Task 5: File_Type 1 (phSeq) is now a supported gate value -- construction
        // itself doesn't touch read_audio, so a plain TDC driver config with the gate
        // flipped must succeed (the gate check runs before the per-config driver loop).
        let mut cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        cfg.insert("File_Type".to_string(), "1".to_string());
        let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), solo_mode())
            .expect("File_Type 1 (phSeq) must be accepted by the gate");
        assert_eq!(bag.file_type(), 1);
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

    /// Phase 7 Task 4 dispatch unit: `Inference_Path` selects the driver variant.
    /// - `fast` + algo 3  -> `Processor::FastSpectral`
    /// - absent           -> `Processor::Spectral` (exact, default)
    /// - junk value       -> construction error
    /// - `fast` + algo 4  -> bail (only algo 3 has a fast counterpart in Task 4)
    ///
    /// tier2_spectral.config is the algo-3 vehicle (TDCwindow 0 + InputNormalization
    /// -1, so the fast driver's construction bails don't fire); `BLSTM_weightsFile`
    /// is emptied so construction does not need the weight pack on disk.
    #[test]
    fn inference_path_dispatch() {
        let base3 = || {
            let mut m = with_bag_keys(load_config("phase4a/tier2_spectral.config"), 3);
            m.insert("BLSTM_weightsFile".to_string(), String::new());
            m
        };

        // fast + algo 3 -> FastSpectral.
        let mut fast3 = base3();
        fast3.insert("Inference_Path".to_string(), "fast".to_string());
        let bag =
            BagOfProcessors::from_configs(std::slice::from_mut(&mut fast3), solo_mode()).unwrap();
        assert!(
            matches!(bag.processor(0), Processor::FastSpectral(_)),
            "fast + algo 3 must dispatch to FastSpectral"
        );

        // absent -> exact Spectral.
        let mut exact3 = base3();
        let bag =
            BagOfProcessors::from_configs(std::slice::from_mut(&mut exact3), solo_mode()).unwrap();
        assert!(
            matches!(bag.processor(0), Processor::Spectral(_)),
            "absent Inference_Path must dispatch to the exact Spectral"
        );

        // junk value -> error.
        let mut junk = base3();
        junk.insert("Inference_Path".to_string(), "turbo".to_string());
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut junk), solo_mode()) {
            Err(e) => assert!(
                e.to_string().contains("Inference_Path"),
                "junk value error must name Inference_Path, got: {e}"
            ),
            Ok(_) => panic!("junk Inference_Path must bail"),
        }

        // fast + algo 4 -> bail (only algo 3 supported in Task 4).
        let mut fast4 = with_bag_keys(load_config("phase2b/signal.config"), 4);
        fast4.insert("Inference_Path".to_string(), "fast".to_string());
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut fast4), solo_mode()) {
            Err(e) => assert!(
                e.to_string().to_lowercase().contains("fast"),
                "fast + algo 4 bail must mention fast, got: {e}"
            ),
            Ok(_) => panic!("fast + algo 4 must bail (no fast algo-4 driver)"),
        }
    }

    /// T6b: `set_weights` on a fast-dispatched conf must bail loudly (the silent
    /// `Ok(())` no-op used to let a seam caller believe an injected weight pack had
    /// taken effect when it was discarded -- the T6 SAD-run failure mode). The error
    /// text must name `Inference_Path` and the config-time weight-file mechanism, so
    /// a caller hitting this in practice is pointed at the fix, not just told "no".
    #[test]
    fn fast_spectral_set_weights_bails_loudly() {
        let mut fast3 = with_bag_keys(load_config("phase4a/tier2_spectral.config"), 3);
        fast3.insert("BLSTM_weightsFile".to_string(), String::new());
        fast3.insert("Inference_Path".to_string(), "fast".to_string());
        let mut bag =
            BagOfProcessors::from_configs(std::slice::from_mut(&mut fast3), solo_mode()).unwrap();
        assert!(matches!(bag.processor(0), Processor::FastSpectral(_)));

        match bag.set_weights(0, &[vec![1.0, 2.0, 3.0]]) {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("Inference_Path"),
                    "error must name Inference_Path, got: {msg}"
                );
                assert!(
                    msg.contains("BLSTM_weightsFile"),
                    "error must name the config-time weight-file mechanism, got: {msg}"
                );
            }
            Ok(()) => panic!("set_weights on a fast-dispatched conf must bail, not silently no-op"),
        }
    }

    /// T6b sibling-audit control: the SAME call on the algo-0/1/2 non-NN arms (which
    /// genuinely have no weight concept) must stay the legacy-matching `Ok(())` no-op --
    /// the fast-arm bail must not have widened to cover them too.
    #[test]
    fn non_nn_set_weights_stays_inert_ok() {
        let tdc = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        let mut cfgs = vec![tdc];
        let mut bag = BagOfProcessors::from_configs(&mut cfgs, solo_mode()).unwrap();
        assert!(matches!(bag.processor(0), Processor::Tdc(_)));
        assert!(bag.set_weights(0, &[vec![1.0, 2.0, 3.0]]).is_ok());
    }
}
