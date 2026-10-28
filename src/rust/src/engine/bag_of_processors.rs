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
use crate::engine::channel_result::{ChannelResult, LidResult};
use crate::engine::confusion;
use crate::engine::corpus::CorpusItem;
use crate::fast::driver::{FastSpectralSegmenter, FastTwinLid};
use crate::features::stats::InputStatistics;
use crate::legacy_config::{get_f64_default, get_f64_opt};
use crate::tasks::lid::{BlstmSpectralLid, TwinBlstmSpectralLid};
use crate::tasks::sad::{
    BlstmSignalSegmenter, BlstmSpectralSegmenter, LtsvSegmenter, TdcSegmenter,
};
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::{
    compute_errors, load_ref_csv, load_ref_stm, load_ref_vrcts, write_vrcts_multichannel,
};
use crate::tasks::segmenter::Segmenter;
use crate::tasks::vrcts::VrctsPart;

/// `conf.get<int>(name)` (required, no default): missing key is an error.
fn get_i32(map: &IndexMap<String, String>, key: &str) -> Result<i32> {
    map.get(key)
        .ok_or_else(|| anyhow::anyhow!("param '{key}' not found in config"))?
        .trim()
        .parse::<i32>()
        .map_err(|e| anyhow::anyhow!("`{key}`: cannot parse as i32: {e}"))
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
/// Mode-7 phSeq/cep LID arm. Both are inference-only: they take weight packs like their
/// exact counterparts (issue #62) but expose no trainable f64 net, so every training
/// read or write of the bag (weights, derivatives, stats, save, update, back-prop flag)
/// iterates an empty [`Segmenter::nets`] for them -- inert by construction, since the
/// fast path never trains (training stays exact f64, spec S1). `FastTwinLid` DOES surface
/// its LID members, so the scored result row's confusion columns flow exactly as the
/// exact Twin's. Both are `Clone` (deep-copying the f32 net + workspace), so the bag's
/// `#[derive(Clone)]` still holds; the clone sites (grad-check snapshot, per-lane
/// training fan-out) are exact-path-only, so a fast-variant deep copy is not on any hot
/// path.
///
/// Every bag method reaches a driver through [`Processor::driver`] /
/// [`Processor::driver_mut`], the only match on the variant (issue #25); what varies per
/// algo beyond the driver's own surface is the training policy, keyed on the algo
/// number as the legacy is (`training_policy`). `reset_weights_derivatives` has no
/// dispatch here at all: it lives on `Network`/layer internals, invoked automatically
/// from within a net's OWN `feed_backward`; the fast drivers never call `feed_backward`
/// (no backward implementation exists), so it is unreachable for them by construction.
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

impl Processor {
    /// The config's driver behind the [`Segmenter`] interface: with
    /// [`Self::driver_mut`], the only dispatch on the variant (issue #25).
    pub fn driver(&self) -> &dyn Segmenter {
        match self {
            Processor::Vrcts(s) => s,
            Processor::Tdc(s) => s,
            Processor::Ltsv(s) => s,
            Processor::Spectral(s) => s,
            Processor::Signal(s) => s,
            Processor::Lid(s) => s,
            Processor::TwinLid(s) => s,
            Processor::FastSpectral(s) => s,
            Processor::FastTwinLid(s) => s,
        }
    }

    /// [`Self::driver`], mutably.
    pub fn driver_mut(&mut self) -> &mut dyn Segmenter {
        match self {
            Processor::Vrcts(s) => s,
            Processor::Tdc(s) => s,
            Processor::Ltsv(s) => s,
            Processor::Spectral(s) => s,
            Processor::Signal(s) => s,
            Processor::Lid(s) => s,
            Processor::TwinLid(s) => s,
            Processor::FastSpectral(s) => s,
            Processor::FastTwinLid(s) => s,
        }
    }
}

/// Which of the two accumulated cost pairs a trainable net owns (issue #92): the
/// segmentation pair (`cost` / `seg_cost`+`seg_count`) or the LID pair (`costLID` /
/// `lid.cost`+`lid.count`). Read by the bag's update ([`CostPair::pick`]) and by the
/// corpus-level gradient check (`CorpusProcessor::grad_check_cost`), so the choice is
/// written once, in [`training_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CostPair {
    Seg,
    Lid,
}

impl CostPair {
    /// The pair's cost among the two the epoch fold accumulated.
    pub(crate) fn pick(self, cost: f64, cost_lid: f64) -> f64 {
        match self {
            CostPair::Seg => cost,
            CostPair::Lid => cost_lid,
        }
    }
}

/// The save criterion of `BagOfProcessors::saveWeights` (`:148-181`), per algo.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SaveCriterion {
    /// :149-162 `cost` (`badClassif + cost` is commented out).
    Cost,
    /// :165-171 `badClassifLID + costLID`.
    BadClassifLidPlusCostLid,
    /// :172-179 `cost + costLID`.
    CostPlusCostLid,
}

impl SaveCriterion {
    pub(crate) fn value(self, cost: f64, cost_lid: f64, bad_classif_lid: f64) -> f64 {
        match self {
            SaveCriterion::Cost => cost,
            SaveCriterion::BadClassifLidPlusCostLid => bad_classif_lid + cost_lid,
            SaveCriterion::CostPlusCostLid => cost + cost_lid,
        }
    }
}

/// The per-algo training policy of `BagOfProcessors::saveWeights`/`updateWeights`
/// (`:148-205`), keyed on the algo number as the legacy is: the save criterion and, per
/// trainable net in pack order, the cost pair it owns (its update cost, and the cost the
/// gradient check differentiates, issue #92) and its save-file basename prefix. `None`
/// for algo 0/1/2 (no legacy branch; the gradient check's `:275-289` has no arm either,
/// so its cost stays untouched and the `:292` division yields 0/0 = NaN). A fast conf
/// (algo 3 or 6) gets a policy too but has no trainable net to apply it to.
pub(crate) fn training_policy(
    algo: i32,
) -> Option<(SaveCriterion, &'static [(CostPair, &'static str)])> {
    match algo {
        // :149-162 saveCriterion = cost; :184-193 update with cost; gradCheck :275-277.
        3 | 4 => Some((SaveCriterion::Cost, &[(CostPair::Seg, "")])),
        // :165-171 saveCriterion = badClassifLID + costLID; :192-194 update with costLID;
        // gradCheck :278-280.
        5 => Some((
            SaveCriterion::BadClassifLidPlusCostLid,
            &[(CostPair::Lid, "")],
        )),
        // :172-179 saveCriterion = cost + costLID, BOTH nets saved under ONE gate,
        // `saveWeightsLID` adding its own `LID_` prefix; :195-200 updateWeights(cost)
        // THEN updateWeightsLID(costLID); gradCheck :281-288 net 0 the SAD pair, net 1
        // the LID pair.
        6 => Some((
            SaveCriterion::CostPlusCostLid,
            &[(CostPair::Seg, ""), (CostPair::Lid, "LID_")],
        )),
        _ => None,
    }
}

/// The ONE refusal of a training-shaped config on the fast tree (issue #92's one body,
/// issue #110's one site). The fast drivers' training arms are inert (`nets()` empty,
/// training stays exact f64), so a config that would train or gradient-check them is a
/// config error, decided where fast or exact is decided, [`BagOfProcessors::from_configs`],
/// for every mode: `CorpusProcessor::train` and `run()`'s gradient check are never reached
/// on a fast bag (the seam's explicit `grad_check` answers `[]`: no net is active). The
/// four facts are read as written, before the work-distributed kill zeroes the epoch count
/// and the epsilon: the config-0 `Neural_Networks_BackPropagation_Epochs`
/// and `Neural_Networks_Gradient_Check_Epsilon` (what `CorpusProcessor::run` reads) and
/// every config's two `BackPropagationActivated` flags (per net per config). Outside Multi
/// mode a flag at `Epochs 0` is still gradient-shaped (F11: `run_solo` computes the
/// gradient the flag asks for), so no mode is exempt and the wording names no mode.
pub(crate) fn refuse_fast_training(
    epochs: i64,
    epsilon: f64,
    (bp_sad, bp_lid): (bool, bool),
) -> Result<()> {
    if epochs > 0 || epsilon > 0.0 || bp_sad || bp_lid {
        bail!(
            "Inference_Path fast is inference-only (training stays exact f64), but this \
             config is training-shaped (Neural_Networks_BackPropagation_Epochs {epochs}, \
             Neural_Networks_Gradient_Check_Epsilon {epsilon}, BackPropagationActivated \
             SAD={bp_sad}/LID={bp_lid}); train on the exact path"
        );
    }
    Ok(())
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
    /// Phase 8 S1.1 `Audio_fixed_gain` (config-0-only, like `Audio_offset`/
    /// `Audio_max_duration`/`File_Type` above it): threaded to every
    /// [`read_audio`] call in [`Self::segmentation_function`]. `None` (the key
    /// absent, true for every pre-phase-8 committed config) reproduces the
    /// legacy `normalize_channels` path byte-identically -- the sanctioned
    /// inert-by-default guard.
    fixed_gain: Option<f64>,
    lock_files_dir: String,
    lock_files_prefix: String,
    algo_types: Vec<i32>,
    pruning_thresholds: Vec<f64>,
    processors: Vec<Processor>,
    exclude_nontrans: bool,
    /// `Inference_Path fast` (issue #110): the one per-bag fact, decided in
    /// [`Self::from_configs`] and answered by [`Self::is_fast_tree`].
    use_fast: bool,
    /// Test-support capture of `save_and_update`'s per-conf algo-5/6
    /// `(errorPercLID, confusion)` (display-only in the legacy; see
    /// [`Self::print_confusion_matrix`]). Cleared at each `save_and_update`.
    #[cfg(feature = "test-support")]
    last_confusion: Vec<(f64, Array2<f64>)>,
    /// Test-support capture (no legacy counterpart): the `costLID` each conf's
    /// [`Self::update_weights`] last received. The `:465` `costLID = -1.0` gate sits
    /// between the save and the update in [`Self::save_and_update`]; an algo-3/4 update
    /// ignores `costLID` (its criterion is `cost` alone), so this is the only place the
    /// gate order is observable on a SAD conf.
    #[cfg(feature = "test-support")]
    last_update_cost_lid: BTreeMap<usize, f64>,
    /// T1-review hardening (Phase 8 Task 1, finding 1; no legacy counterpart):
    /// sum-of-`|x|` over `data_raw` from the LAST [`Self::segmentation_function`]
    /// internal [`read_audio`] call. `segmentation_function`'s dispatch surface
    /// (the driver interface, [`Segmenter`]) is NN-output-shaped, not audio-shaped, so
    /// the internally-decoded audio is otherwise unobservable from outside the
    /// function. This exists because the posterior/segmentation observable
    /// (the driver's [`Segmenter::last_result_rows`]) turns out to be
    /// PROVABLY INVARIANT to `Audio_fixed_gain`'s value for a DCT-based config
    /// with `IgnoreFirstDCT true` (e.g. `tier2_spectral.config`): a global
    /// per-file amplitude gain shifts every frame's log-mel value by the SAME
    /// constant, which (a) a DCT-II basis vector for coefficient index >= 1
    /// projects to exactly zero (the retained AC coefficients), and (b) a
    /// temporal delta/delta-delta cancels exactly too (differencing removes a
    /// time-constant offset, including on the delta OF the dropped DC/C0 term,
    /// which is not itself dropped from the delta block -- only the static C0
    /// column is). MEASURED (`tests/phase8_frozen_norm.rs`, an earlier version
    /// of `bag_threads_fixed_gain_discriminates_dropped_gain`): a 2x-vs-1x
    /// staged gain produced a BIT-IDENTICAL FastSpectral posterior, 0.0
    /// max-abs delta over all 1500 rows -- not a near-miss, the exact
    /// algebraic cancellation above. `sum(|x|)` is a cheap, trivially-verified,
    /// provably gain-SENSITIVE stand-in (scales as `1/gain` for fixed raw
    /// samples) captured at the one place `Audio_fixed_gain` IS load-bearing.
    #[cfg(feature = "test-support")]
    last_audio_abs_sum: Option<f64>,
}

impl BagOfProcessors {
    /// Port of `BagOfProcessors(vector<ConfigFile>&, char*)` (`:9-51`).
    ///
    /// Config-0-only keys (`:10-15`): `numOuterThreads` (REQUIRED, no default --
    /// missing is an error, unlike every other key here), `Audio_offset` 0.0,
    /// `Audio_max_duration` 3.6e6, `File_Type` 0, `LockFilesDir`/`LockFilesPrefix`
    /// "". `exclude_nontrans` (NOT a legacy `BagOfProcessors` member -- read from
    /// configs[0] here so it's available alongside the bag for Task 5's reference
    /// loading, default false). `Audio_fixed_gain` (Phase 8 S1.1, port-only, no
    /// legacy source): absent -> `None`, threaded to every `read_audio` call in
    /// [`Self::segmentation_function`] -- the one sanctioned exact-tree touch of
    /// the streaming phase.
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
    /// intentionally NOT threaded through here, and since issue #110 neither is the
    /// mode: a bag is built from its configs alone (the legacy `mode` argument fed
    /// only those two display flags).
    ///
    /// Algo 0 (`VRCTSPart`) is wired here since Phase 4b Task 8
    /// (`Processor::Vrcts`); Algo 5/6 (`BLSTMSpectralLID`/`TwinBLSTMSpectralLID`)
    /// since Phase 4b Task 9 (`Processor::Lid`/`Processor::TwinLid`).
    /// `File_Type` 3/4 (phSeq-N variant, mat input) remain unported -- `bail!`,
    /// IMPROVEMENTS entry; `File_Type` 0 (wav), 1 (phSeq, Task 5), and 2 (cep,
    /// Phase 6 Task 1) are all supported.
    pub fn from_configs(configs: &mut [IndexMap<String, String>]) -> Result<BagOfProcessors> {
        if configs.is_empty() {
            bail!("BagOfProcessors::from_configs: at least one config is required");
        }

        let num_outer_threads = get_i32(&configs[0], "numOuterThreads")?;
        let offset_begin = get_f64_default(&configs[0], "Audio_offset", 0.0)?;
        let duration_max = get_f64_default(&configs[0], "Audio_max_duration", 3.6e6)?;
        let file_type = get_i32_default(&configs[0], "File_Type", 0)?;
        // Phase 8 S1.1: `Audio_fixed_gain`, config-0-only like its Audio_* siblings
        // above. Absent (every pre-phase-8 config) -> `None` -> read_audio's legacy
        // normalize_channels path, byte-identical to before this key existed: absence
        // is a fallback MODE, not a fallback value, hence `get_f64_opt`.
        let fixed_gain = get_f64_opt(&configs[0], "Audio_fixed_gain")?;
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
        // Issue #110: the config-0 half of the fast-tree training refusal, read as written
        // (`CorpusProcessor::new` reads the same two keys off configs[0], then zeroes both
        // when work is distributed; a distributed fast run with `Epochs 5` is still a
        // config error). The per-config half (the two flags) is read in the loop below.
        // Read on a fast bag only: an exact run's config reads keep their order (a malformed
        // value still errors where it did, after the mode's output-file truncation, ADR-0002).
        let (epochs, epsilon) = match use_fast {
            true => (
                get_i32_default(&configs[0], "Neural_Networks_BackPropagation_Epochs", 0)?,
                get_f64_default(&configs[0], "Neural_Networks_Gradient_Check_Epsilon", 0.0)?,
            ),
            false => (0, 0.0),
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
            // RIDER (Phase 7 Task 5; one body since #92, one site and mode-free since
            // #110): a TRAINING-shaped config (`Neural_Networks_BackPropagation_Epochs > 0`,
            // `Neural_Networks_Gradient_Check_Epsilon > 0`, or a `BackPropagationActivated`-on
            // net) is a config error under `Inference_Path fast`, caught here so neither the
            // binary nor the seam can drop to a no-op fast "training" run or a gradient the
            // fast tree does not compute ([`refuse_fast_training`]). Guards both algo 3 and
            // 6; a run whose tail is Epochs 0, epsilon 0 and every flag off drives inference
            // over the same configs in every mode.
            if use_fast {
                let bp_sad = get_bool_default(map, "BLSTM_BackPropagationActivated", false)?;
                let bp_lid = get_bool_default(map, "BLSTM_LID_BackPropagationActivated", false)?;
                refuse_fast_training(i64::from(epochs), epsilon, (bp_sad, bp_lid))?;
                let processor = match algo {
                    3 => {
                        let mut seg = FastSpectralSegmenter::from_legacy(map, None)?;
                        seg.load_weights_file()?;
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
            fixed_gain,
            lock_files_dir,
            lock_files_prefix,
            algo_types,
            pruning_thresholds,
            processors,
            exclude_nontrans,
            use_fast,
            #[cfg(feature = "test-support")]
            last_confusion: Vec::new(),
            #[cfg(feature = "test-support")]
            last_update_cost_lid: BTreeMap::new(),
            #[cfg(feature = "test-support")]
            last_audio_abs_sum: None,
        })
    }

    pub fn nb_of_conf(&self) -> usize {
        self.nb_of_conf
    }

    /// Whether this bag runs the fast tree (`Inference_Path fast`, issue #110): one
    /// per-bag fact, decided once at construction, where a training-shaped config was
    /// already refused ([`refuse_fast_training`]). A fast bag trains nothing by
    /// construction; no driver is asked.
    pub fn is_fast_tree(&self) -> bool {
        self.use_fast
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

    /// Phase 8 S1.1: the parsed `Audio_fixed_gain` value, if the config carried one.
    pub fn fixed_gain(&self) -> Option<f64> {
        self.fixed_gain
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

    /// Port of `BagOfProcessors::isBackPropActivated` (`:73-86`): one flag per trainable
    /// net (`[sad, lid]` for algo 6). A conf with no trainable net (algo 0/1/2, the fast
    /// drivers, which never train) keeps the legacy's own default
    /// `vector<bool>(1, false)` (`:85`) -- NOT an empty vec.
    pub fn is_back_prop_activated(&self, pos: usize) -> Vec<bool> {
        let nets = self.processors[pos].driver().nets();
        if nets.is_empty() {
            return vec![false];
        }
        nets.iter()
            .map(|net| net.config().back_propagation_activated)
            .collect()
    }

    /// Port of `BagOfProcessors::getWeights` (`:88-101`): one flat weight vector per
    /// trainable net (`[regular, LID]` for algo 6, `:95-98`); EMPTY for a conf with no
    /// trainable net (`:100`), unlike `isBackPropActivated`'s single-`false` default.
    ///
    /// T6b AUDIT: empty for the fast drivers too, no bail -- an empty READ cannot be
    /// mistaken for real data, and internal bookkeeping calls this unconditionally (the
    /// `epoch_weight_trace_lid` test-support snapshot reads `get_weights(0)` every epoch).
    /// A fast conf's pack count comes from the driver's `pack_count`, never from here.
    pub fn get_weights(&self, pos: usize) -> Vec<Vec<f64>> {
        let nets = self.processors[pos].driver().nets();
        nets.iter().map(|net| net.get_weights()).collect()
    }

    /// Port of `BagOfProcessors::setWeights` (`:103-114`): one pack per net, in the
    /// driver's pack order (algo 6: `at(0)` to the SAD net, `at(1)` to the LID net,
    /// `:110-113`); algo 0/1/2 (no legacy `else` branch) are a no-op.
    ///
    /// The fast drivers inject for real (issue #62, T6b's loud bail before it): they
    /// rebuild their f32 net from the in-memory pack under the exact drivers' length
    /// contract, in the same words (`nn::blstm::check_pack_len`), so a seam caller gets
    /// the same refusals on both trees and the same injection.
    pub fn set_weights(&mut self, pos: usize, new_weights: &[Vec<f64>]) -> Result<()> {
        let driver = self.processors[pos].driver_mut();
        // Issue #60: the list is sized BEFORE the driver indexes it, so a short list is a
        // refusal rather than a panic and a long one is not silently truncated. Algo
        // 0/1/2 (`pack_count` 0) stay the legacy inert no-op whatever the length.
        let needed = driver.pack_count();
        if needed > 0 && new_weights.len() != needed {
            bail!(
                "The number of weight packs given is {} but config {pos} holds {needed} \
                 network(s).",
                new_weights.len()
            );
        }
        driver.set_packs(new_weights)
    }

    /// Port of `BagOfProcessors::getInputStatistics` (`:116-130`): one `InputStatistics`
    /// per trainable net (`[regular, LID]` for algo 6); EMPTY for a conf with no
    /// trainable net (`:128-129`). Read-shaped and never indexed back out for such a
    /// conf, so it stays inert rather than fallible on a plain inference run (T6b).
    pub fn get_input_statistics(&self, pos: usize) -> Vec<InputStatistics> {
        let nets = self.processors[pos].driver().nets();
        nets.iter()
            .map(|net| net.input_statistics().clone())
            .collect()
    }

    /// Port of `BagOfProcessors::getWeightsDerivatives` (`:132-146`): one `Nx2`
    /// derivative matrix per trainable net (`[regular, LID]` for algo 6); EMPTY for a
    /// conf with no trainable net (`:143-144`).
    ///
    /// T6b AUDIT: inert, not fallible, DESPITE being called UNCONDITIONALLY for every
    /// conf on every file of every epoch (`corpus_processor.rs::run_epoch`'s per-file
    /// harvest, not gated on `is_back_prop_activated`): a fast conf's empty Vec is folded
    /// into the `derivs` map but never indexed back out (`save_weights`/`update_weights`
    /// iterate the same empty `nets()`), so it cannot corrupt a gradient sum.
    pub fn get_weights_derivatives(&self, pos: usize) -> Vec<Array2<f64>> {
        let nets = self.processors[pos].driver().nets();
        nets.iter()
            .map(|net| net.get_weights_derivatives())
            .collect()
    }

    /// Port of `BagOfProcessors::saveWeights` (`:148-181`): the per-algo save criterion
    /// ([`training_policy`]), gated on beating the running `bestCost[pos]`; every
    /// trainable net saves under the ONE gate, algo 6's LID net with the `LID_` basename
    /// prefix of `saveWeightsLID` (`:176`). A conf with no trainable net (algo 0/1/2, no
    /// legacy branch; the fast drivers, whose `is_back_prop_activated` is `false`) is a
    /// no-op that reads neither `bestCost` nor `derivs`/`stats`.
    /// Arity mirrors the legacy signature (`:148`) verbatim.
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
        let nets = self.processors[pos].driver().nets();
        let Some((save_criterion, per_net)) = training_policy(self.algo_types[pos]) else {
            return Ok(());
        };
        if nets.is_empty() {
            return Ok(());
        }
        let save_criterion = save_criterion.value(cost, cost_lid, bad_classif_lid);
        if best_cost[&pos] > save_criterion {
            for (i, (net, (_, prefix))) in nets.iter().zip(per_net).enumerate() {
                let file = crate::io::prefix_basename(prefix, filename);
                net.save_weights(&file, &derivs[&pos][i], &stats[&pos][i])?;
            }
            best_cost.insert(pos, save_criterion);
        }
        Ok(())
    }

    /// Port of `BagOfProcessors::updateWeights` (`:183-205`): unconditional (unlike
    /// `saveWeights`, no best-cost gate), each trainable net updated in pack order with
    /// its [`training_policy`] cost -- algo 6 the SAD net with `cost` THEN the LID net with
    /// `costLID` (`:196-200`). The caller's `:465` `costLID = -1.0` no-speech gate is
    /// LIVE for both LID algos (the commented-out `saveCriterion > 0` skip at
    /// `:199-203` is dead: the update runs unconditionally, -1.0 cost included). A conf
    /// with no trainable net is a no-op that never reads `derivs`.
    fn update_weights(
        &mut self,
        pos: usize,
        cost: f64,
        _bad_classif: f64,
        cost_lid: f64,
        _bad_classif_lid: f64,
        derivs: &BTreeMap<usize, Vec<Array2<f64>>>,
    ) {
        #[cfg(feature = "test-support")]
        self.last_update_cost_lid.insert(pos, cost_lid);
        let Some((_, per_net)) = training_policy(self.algo_types[pos]) else {
            return;
        };
        let nets = self.processors[pos].driver_mut().nets_mut();
        for (i, (net, (pair, _))) in nets.into_iter().zip(per_net).enumerate() {
            net.update_weights(&derivs[&pos][i], pair.pick(cost, cost_lid));
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
    fn print_confusion_matrix<'a>(
        &self,
        rows: impl IntoIterator<Item = &'a LidResult>,
        _config_nb: usize,
    ) -> (f64, Array2<f64>) {
        let (matrix, error) = confusion::confusion_from_results(rows);
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

    /// Test-observation hook: see [`Self::last_update_cost_lid`]'s field doc.
    #[cfg(feature = "test-support")]
    pub fn last_update_cost_lid_for_test(&self, pos: usize) -> Option<f64> {
        self.last_update_cost_lid.get(&pos).copied()
    }

    /// Test-observation hook (Phase 8 Task 1, T1-review hardening finding 1, no
    /// legacy counterpart): see [`Self::last_audio_abs_sum`]'s field doc for why
    /// this exists.
    #[cfg(feature = "test-support")]
    pub fn last_audio_abs_sum_for_test(&self) -> Option<f64> {
        self.last_audio_abs_sum
    }

    /// Port of `BagOfProcessors::saveAndUpdate` (`:409-471`): per-config
    /// column-sum/mean aggregation over the file x channel results, cost/
    /// badClassif/costLID/badLIDClassif derivation, WER percent scaling, the
    /// `costMem`/`badClassifMem`/`costLIDMem`/`badClassifLIDMem` row writes, the
    /// `bestNNWeight_<pos+1>_<filename>` save, and the `costLID = -1.0` gate
    /// (`:465`) applied AFTER `save_weights` and BEFORE `update_weights` -- order
    /// is load-bearing (the save criterion sees the real costLID; the update
    /// criterion sees the gated one).
    ///
    /// The legacy summed the result-row COLUMNS (`:411-412`); here each field is
    /// summed by its own explicit ascending-row loop (the repo's ascending-loop
    /// product contract, NOT `ndarray::sum_axis`/`mean_axis`), which is the same
    /// per-column addition sequence and therefore bit-identical.
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
        results_per_conf: &[Vec<&ChannelResult>],
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
            let rows = &results_per_conf[ii];
            let n = rows.len() as f64;

            // legacy: :411-412 colwise sum/mean, one ascending row loop per field
            // (product contract). A missing LID block sums as the wire's 0.0 slots.
            let mut sum_error_rate = 0.0_f64;
            let mut sum_seg_cost = 0.0_f64;
            let mut sum_seg_count = 0.0_f64;
            let mut sum_speech_duration = 0.0_f64;
            let mut sum_lid_cost = 0.0_f64;
            let mut sum_lid_correct = 0.0_f64;
            let mut sum_lid_count = 0.0_f64;
            let mut sum_nb_words = 0.0_f64;
            let mut sum_corrects = 0.0_f64;
            let mut sum_subs = 0.0_f64;
            let mut sum_ins = 0.0_f64;
            let mut sum_dels = 0.0_f64;
            let mut sum_coverage_penalty = 0.0_f64;
            let mut sum_delay_penalty = 0.0_f64;
            for r in rows {
                sum_error_rate += r.error_rate;
                sum_seg_cost += r.seg_cost;
                sum_seg_count += r.seg_count as f64;
                sum_speech_duration += r.speech_duration;
                sum_lid_cost += r.lid_cost();
                sum_lid_correct += r.lid.as_ref().map_or(0.0, LidResult::correct_wire);
                sum_lid_count += r.lid_count() as f64;
                sum_nb_words += r.wer.nb_words as f64;
                sum_corrects += r.wer.corrects as f64;
                sum_subs += r.wer.subs as f64;
                sum_ins += r.wer.ins as f64;
                sum_dels += r.wer.dels as f64;
                sum_coverage_penalty += r.wer.coverage_penalty;
                sum_delay_penalty += r.wer.delay_penalty;
            }

            // legacy: :413-414 cost = sums(4), guarded /= sums(last).
            let mut cost = sum_seg_cost;
            if sum_seg_count > 0.0 {
                cost /= sum_seg_count;
            }
            // legacy: :415 badClassif = means(2).
            let bad_classif = sum_error_rate / n;
            // legacy: :416-417 costLID = sums(14), guarded /= sums(cols-2).
            let mut cost_lid = sum_lid_cost;
            if sum_lid_count > 0.0 {
                cost_lid /= sum_lid_count;
            }
            // legacy: :418 badLIDClassif = 100-means(15).
            let bad_lid_classif = 100.0 - sum_lid_correct / n;

            // legacy: :419-433 WER percent scaling, guarded by nbOfWords > 0.
            let nb_of_words = sum_nb_words;
            let mut wer_correct = 100.0 * sum_corrects;
            let mut wer_subs = 100.0 * sum_subs;
            let mut wer_ins = 100.0 * sum_ins;
            let mut wer_dels = 100.0 * sum_dels;
            let mut wer_coverage_penalty = 100.0 * sum_coverage_penalty;
            let mut wer_delay_penalty = 100.0 * sum_delay_penalty;
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

            // legacy: :435-441 confusion block, algo 5/6 only -- in the port, "this
            // config's results carry a LID block" (only the LID drivers fill one).
            // errorPercLID + the confusion string are DISPLAY-ONLY in the legacy
            // (`:443` cout / `:468` cout) -- nothing parity-relevant is stored (see
            // print_confusion_matrix's usage finding), so outside test-support the
            // pair is dropped after the call.
            if rows.first().is_some_and(|r| r.lid.is_some()) {
                let lid_rows = rows.iter().filter_map(|r| r.lid.as_ref());
                let confusion_pair = self.print_confusion_matrix(lid_rows, ii + 1);
                #[cfg(feature = "test-support")]
                self.last_confusion.push(confusion_pair);
                #[cfg(not(feature = "test-support"))]
                let _ = confusion_pair;
            }

            // legacy: :445-446 total durations, needed for the :465 gate -- the
            // legacy's `means(6) * rows` round trip, kept as written.
            let total_speech_duration = sum_speech_duration / n * n;

            // legacy: :457-460 row writes.
            cost_mem_row[ii] = cost;
            bad_classif_row[ii] = bad_classif;
            cost_lid_row[ii] = cost_lid;
            bad_classif_lid_row[ii] = bad_lid_classif;

            // legacy: :462-464 bestNNWeight_<pos+1>_<filename> compose -- on the
            // BASENAME (the legacy glued it onto the whole string; FIXED, see
            // `io::prefix_basename`).
            let save_filename =
                crate::io::prefix_basename(&format!("bestNNWeight_{}_", ii + 1), filename);
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
        self.processors[pos]
            .driver_mut()
            .get_segmentation(audio, seg_per_chan, None)
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
    ) -> Result<BTreeMap<usize, BTreeMap<usize, ChannelResult>>> {
        let mut results: BTreeMap<usize, BTreeMap<usize, ChannelResult>> = BTreeMap::new();

        // legacy: :254 AudioStruct audio(_OffsetBegin, _DurationMax, _FileType, corpusItem);
        // Phase 8 S1.1: `self.fixed_gain` threads `Audio_fixed_gain` here -- the ONE
        // sanctioned exact-tree touch of the phase (see `read_audio`'s doc).
        let file_name = &item.file_name;
        let mut audio = read_audio(
            Path::new(file_name),
            self.offset_begin,
            self.duration_max,
            self.file_type,
            self.fixed_gain,
        )?;
        // T1-review hardening (finding 1): capture a gain-sensitive digest of the
        // internal read for `last_audio_abs_sum_for_test` -- see that field's doc.
        #[cfg(feature = "test-support")]
        {
            self.last_audio_abs_sum = Some(audio.data_raw.iter().map(|v| v.abs()).sum());
        }
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
            let driver = self.processors[ii].driver_mut();
            driver.get_segmentation(&mut audio, &mut seg_per_chan, reference.as_deref())?;
            let dump_dir = driver.dump_dir().to_string();

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
            let driver = self.processors[ii].driver();

            let mut chan_map: BTreeMap<usize, ChannelResult> = BTreeMap::new();
            for chan in 0..channel_count {
                let refc = reference.as_ref().map(|r| &r[chan]);
                let report = compute_errors(&mut seg_per_chan[chan], refc, nb_words);
                let speech_duration = speech_duration_of(&seg_per_chan[chan]);
                let members = driver.channel_members(chan);
                // legacy: :338/:380 `!seg._IsLIDCorrect.empty()` -- LID block live for
                // algo 5/6 (the row WIDTH grows by the class count), else two 0.0s. The
                // in-band target is decoded here, once ([`LidResult::from_encoded`]).
                let lid = members.lid.map(|l| {
                    LidResult::from_encoded(
                        l.cumulative_error,
                        l.nb_of_classif,
                        l.is_correct == 100,
                        l.classification_errors,
                    )
                });

                let result = if scored {
                    ChannelResult::scored(
                        &report,
                        time_per_hour,
                        members.cumulative_error,
                        audio_duration,
                        speech_duration,
                        members.nb_of_classif,
                        lid,
                    )
                } else {
                    ChannelResult::unscored(time_per_hour, audio_duration, speech_duration, lid)
                };
                chan_map.insert(chan, result);
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Strip a fixture's training tail (tier2 carries `Epochs 3` + backprop on): since
    /// issue #110 a fast bag refuses a training-shaped config in every mode.
    fn inference_only(mut m: IndexMap<String, String>) -> IndexMap<String, String> {
        m.insert(
            "Neural_Networks_BackPropagation_Epochs".to_string(),
            "0".to_string(),
        );
        m.insert(
            "BLSTM_BackPropagationActivated".to_string(),
            "false".to_string(),
        );
        m
    }

    // === apply_corpus_item_sets_lang_index_and_weight =======================
    // Unit test for the extracted setter helper (Task 2): asserts the bag's
    // AudioStruct-ctor-equivalent copy (class_index -> lang_index, weight ->
    // weight) directly, without going through the full segmentation_function
    // flow (which requires a real decodable wav).
    #[test]
    fn apply_corpus_item_sets_lang_index_and_weight() {
        let mut audio = crate::audio::read_audio(
            &ref_dir().join("phase1/excerpt_2ch_8k.wav"),
            0.0,
            0.1,
            0,
            None,
        )
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
            let bag = BagOfProcessors::from_configs(&mut cfgs).unwrap();
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
        let mut bag = BagOfProcessors::from_configs(&mut cfgs).unwrap();
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

    #[test]
    fn file_type_2_allowed() {
        // Phase 6 Task 1: File_Type 2 (cep) is now a supported gate value. Re-pins the
        // former `file_type_2_bails`. Construction doesn't touch read_audio (the gate runs
        // before the per-config driver loop), so a plain TDC config with the gate flipped
        // must succeed.
        let mut cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        cfg.insert("File_Type".to_string(), "2".to_string());
        let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg))
            .expect("File_Type 2 (cep) must be accepted by the gate");
        assert_eq!(bag.file_type(), 2);
    }

    #[test]
    fn file_type_3_bails() {
        // File_Type 3/4 (phSeq-N variant, mat) stay unported -- the gate still bails.
        let mut cfg = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        cfg.insert("File_Type".to_string(), "3".to_string());
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg)) {
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
        let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg))
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
        let _bag = BagOfProcessors::from_configs(&mut cfgs).unwrap();

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
        let bag = BagOfProcessors::from_configs(&mut cfgs).unwrap();

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
            let mut m = inference_only(with_bag_keys(
                load_config("phase4a/tier2_spectral.config"),
                3,
            ));
            m.insert("BLSTM_weightsFile".to_string(), String::new());
            m
        };

        // fast + algo 3 -> FastSpectral.
        let mut fast3 = base3();
        fast3.insert("Inference_Path".to_string(), "fast".to_string());
        let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut fast3)).unwrap();
        assert!(
            matches!(bag.processor(0), Processor::FastSpectral(_)),
            "fast + algo 3 must dispatch to FastSpectral"
        );

        // absent -> exact Spectral.
        let mut exact3 = base3();
        let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut exact3)).unwrap();
        assert!(
            matches!(bag.processor(0), Processor::Spectral(_)),
            "absent Inference_Path must dispatch to the exact Spectral"
        );

        // junk value -> error.
        let mut junk = base3();
        junk.insert("Inference_Path".to_string(), "turbo".to_string());
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut junk)) {
            Err(e) => assert!(
                e.to_string().contains("Inference_Path"),
                "junk value error must name Inference_Path, got: {e}"
            ),
            Ok(_) => panic!("junk Inference_Path must bail"),
        }

        // fast + algo 4 -> bail (only algo 3 supported in Task 4).
        let mut fast4 = with_bag_keys(load_config("phase2b/signal.config"), 4);
        fast4.insert("Inference_Path".to_string(), "fast".to_string());
        match BagOfProcessors::from_configs(std::slice::from_mut(&mut fast4)) {
            Err(e) => assert!(
                e.to_string().to_lowercase().contains("fast"),
                "fast + algo 4 bail must mention fast, got: {e}"
            ),
            Ok(_) => panic!("fast + algo 4 must bail (no fast algo-4 driver)"),
        }
    }

    /// Issue #62 (T6b's loud bail before it): `set_weights` on a fast-dispatched conf
    /// answers exactly as on the exact one -- the exact-length pack loads, one element
    /// off is refused in the same words. That the loaded pack is the one SCORED (T6b's
    /// silent-no-op hazard) is pinned through the seam, `tests/pyo3/test_fold_run.py`.
    #[test]
    fn fast_spectral_set_weights_answers_as_the_exact_arm() {
        let mut exact3 = inference_only(with_bag_keys(
            load_config("phase4a/tier2_spectral.config"),
            3,
        ));
        exact3.insert("BLSTM_weightsFile".to_string(), String::new());
        let mut fast3 = exact3.clone();
        fast3.insert("Inference_Path".to_string(), "fast".to_string());
        let mut exact = BagOfProcessors::from_configs(std::slice::from_mut(&mut exact3)).unwrap();
        let mut fast = BagOfProcessors::from_configs(std::slice::from_mut(&mut fast3)).unwrap();
        assert!(matches!(fast.processor(0), Processor::FastSpectral(_)));

        let n = exact.get_weights(0)[0].len();
        for len in [n, n - 1, n + 1] {
            let pack = [vec![0.5; len]];
            let want = exact.set_weights(0, &pack).map_err(|e| e.to_string());
            let got = fast.set_weights(0, &pack).map_err(|e| e.to_string());
            assert_eq!(got, want, "a {len}-element pack (the net needs {n})");
        }
    }

    /// T6b sibling-audit control: the SAME call on the algo-0/1/2 non-NN arms (which
    /// genuinely have no weight concept) must stay the legacy-matching `Ok(())` no-op --
    /// no fast-arm rule (T6b's bail, issue #62's length check) widens to cover them.
    #[test]
    fn non_nn_set_weights_stays_inert_ok() {
        let tdc = with_bag_keys(load_config("phase2b/tdc.config"), 1);
        let mut cfgs = vec![tdc];
        let mut bag = BagOfProcessors::from_configs(&mut cfgs).unwrap();
        assert!(matches!(bag.processor(0), Processor::Tdc(_)));
        assert_eq!(bag.processor(0).driver().pack_count(), 0);
        // Issue #60: the pack-count check does not widen to them either, whatever the length.
        for packs in [
            &[][..],
            &[vec![1.0, 2.0, 3.0]][..],
            &[vec![1.0], vec![2.0]][..],
        ] {
            assert!(bag.set_weights(0, packs).is_ok(), "{} pack(s)", packs.len());
        }
    }

    /// Issue #25: the per-algo save/update policy, pinned literally against the legacy
    /// branches (`:148-205`). No training golden runs algo 5 to a second save, so the
    /// algo-5 criterion's `badClassifLID` term is owned by this pin alone (the battery
    /// measured it missed everywhere else). Issue #92: the per-net cost PAIR is the
    /// table's entry (the update cost and the gradient check's cost read it), so the pin
    /// holds the pair, the prefix and the criterion's value.
    #[test]
    fn training_policy_is_the_legacy_branch_per_algo() {
        let (cost, cost_lid, bad_lid) = (1.0, 10.0, 100.0);
        for algo in [0, 1, 2] {
            assert!(training_policy(algo).is_none(), "algo {algo}");
        }
        for algo in [3, 4] {
            let (save, nets) = training_policy(algo).unwrap();
            assert_eq!(save.value(cost, cost_lid, bad_lid), 1.0, "algo {algo}");
            assert_eq!(nets, vec![(CostPair::Seg, "")], "algo {algo}");
        }
        let (save, nets) = training_policy(5).unwrap();
        assert_eq!(save.value(cost, cost_lid, bad_lid), 110.0);
        assert_eq!(nets, vec![(CostPair::Lid, "")]);
        let (save, nets) = training_policy(6).unwrap();
        assert_eq!(save.value(cost, cost_lid, bad_lid), 11.0);
        assert_eq!(nets, vec![(CostPair::Seg, ""), (CostPair::Lid, "LID_")]);
        assert_eq!(CostPair::Seg.pick(cost, cost_lid), 1.0);
        assert_eq!(CostPair::Lid.pick(cost, cost_lid), 10.0);
    }

    /// Issues #92 / #110: the one refusal body. Shape-blind when nothing is
    /// training-shaped, refusing on any of the four facts with one wording that names all
    /// four and no mode.
    #[test]
    fn refuse_fast_training_is_one_body_naming_the_four_facts() {
        assert!(refuse_fast_training(0, 0.0, (false, false)).is_ok());
        for (epochs, epsilon, back_prop) in [
            (1, 0.0, (false, false)),
            (0, 1e-4, (false, false)),
            (0, 0.0, (true, false)),
            (0, 0.0, (false, true)),
        ] {
            let msg = refuse_fast_training(epochs, epsilon, back_prop)
                .unwrap_err()
                .to_string();
            assert!(
                msg.contains("Inference_Path fast is inference-only"),
                "{msg}"
            );
            assert!(msg.contains("this config is training-shaped"), "{msg}");
            for fact in [
                format!("Neural_Networks_BackPropagation_Epochs {epochs}"),
                format!("Neural_Networks_Gradient_Check_Epsilon {epsilon}"),
                format!(
                    "BackPropagationActivated SAD={}/LID={}",
                    back_prop.0, back_prop.1
                ),
            ] {
                assert!(msg.contains(&fact), "{msg} lacks {fact}");
            }
            assert!(
                !msg.contains("mode"),
                "the wording names facts, not a mode: {msg}"
            );
        }
    }

    /// Issue #25: the driver interface per variant, every one built through the bag.
    /// The pack surface spans both trees (the fast Twin takes `[sad, lid]` like the
    /// exact one); the training surface is the exact tree's f64 nets only.
    #[test]
    fn driver_surface_per_variant() {
        let mut vrcts = IndexMap::new();
        for (k, v) in [
            ("VRCTS_decision_thresh_rising", "0.6"),
            ("VRCTS_decision_area_rising", "0.05"),
            ("VRCTS_decision_thresh_falling", "0.3"),
            ("VRCTS_decision_area_falling", "0.1"),
            ("VRCTS_window", "0.032"),
            ("VRCTS_shift", "0.01"),
            ("VRCTS_convolution_window_size", "0"),
            ("VRCTS_speech_padding", "0.1,0.1,0.1,0.1"),
            ("VRCTS_min_speech", "0.2,0.2,0.2"),
            ("VRCTS_min_silence", "0.2,0.2"),
            ("VRCTS_isFast", "false"),
            ("VRCTS_force", "false"),
        ] {
            vrcts.insert(k.to_string(), v.to_string());
        }
        let vrcts = with_bag_keys(vrcts, 0);
        let mut spectral = inference_only(with_bag_keys(
            load_config("phase4a/tier2_spectral.config"),
            3,
        ));
        spectral.insert("BLSTM_weightsFile".to_string(), String::new());
        let mut lid5 = with_bag_keys(load_config("phase4b/lid5.config"), 5);
        lid5.insert("BLSTM_weightsFile".to_string(), String::new());
        let mut twin = with_bag_keys(load_config("phase4b/twin_mode7.config"), 6);
        twin.insert(
            "BLSTM_LID_weightsFile".to_string(),
            ref_dir()
                .join("phase4b/LID_bestNNWeight_1.bin")
                .to_str()
                .unwrap()
                .to_string(),
        );
        let mut exact_cfgs = vec![
            vrcts,
            with_bag_keys(load_config("phase2b/tdc.config"), 1),
            with_bag_keys(load_config("phase2b/ltsv.config"), 2),
            spectral.clone(),
            with_bag_keys(load_config("phase2b/signal.config"), 4),
            lid5,
            twin.clone(),
        ];
        // `Inference_Path` is a config-0 key: the fast variants get their own bag.
        spectral.insert("Inference_Path".to_string(), "fast".to_string());
        let mut fast_cfgs = vec![spectral, twin];
        let mut exact = BagOfProcessors::from_configs(&mut exact_cfgs).unwrap();
        let mut fast = BagOfProcessors::from_configs(&mut fast_cfgs).unwrap();

        let exact_kinds = [
            matches!(exact.processor(0), Processor::Vrcts(_)),
            matches!(exact.processor(1), Processor::Tdc(_)),
            matches!(exact.processor(2), Processor::Ltsv(_)),
            matches!(exact.processor(3), Processor::Spectral(_)),
            matches!(exact.processor(4), Processor::Signal(_)),
            matches!(exact.processor(5), Processor::Lid(_)),
            matches!(exact.processor(6), Processor::TwinLid(_)),
        ];
        assert_eq!(exact_kinds, [true; 7]);
        assert!(matches!(fast.processor(0), Processor::FastSpectral(_)));
        assert!(matches!(fast.processor(1), Processor::FastTwinLid(_)));

        fn check(bag: &mut BagOfProcessors, pos: usize, name: &str, packs: usize, nets: usize) {
            let driver = bag.processor(pos).driver();
            assert_eq!(driver.pack_count(), packs, "{name} pack_count");
            assert_eq!(driver.nets().len(), nets, "{name} nets");
            assert!(
                driver.last_result_rows().is_empty(),
                "{name} before any run"
            );
            let driver = bag.processors[pos].driver_mut();
            assert_eq!(driver.nets_mut().len(), nets, "{name} nets_mut");
        }
        // (pos, name, pack_count, trainable nets)
        for (pos, name, packs, nets) in [
            (0, "Vrcts", 0, 0),
            (1, "Tdc", 0, 0),
            (2, "Ltsv", 0, 0),
            (3, "Spectral", 1, 1),
            (4, "Signal", 1, 1),
            (5, "Lid", 1, 1),
            (6, "TwinLid", 2, 2),
        ] {
            check(&mut exact, pos, name, packs, nets);
        }
        check(&mut fast, 0, "FastSpectral", 1, 0);
        check(&mut fast, 1, "FastTwinLid", 2, 0);
        // Issue #110: fast or exact is a per-bag fact, answered by the bag (a fast driver
        // exchanges packs but trains none of them; the refusal of a training-shaped
        // config sits at construction, so no driver is asked).
        assert!(!exact.is_fast_tree());
        assert!(fast.is_fast_tree());
    }
}
