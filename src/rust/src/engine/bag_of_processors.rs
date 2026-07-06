//! One `Segmenter` per config, dispatched by an `Algo` enum (replaces the legacy
//! if/else ladder); per-file dispatch, weight save/update, metric reduction.
//!
//! Ported from legacy C++: BagOfProcessors.*.

use anyhow::{Result, bail};
use indexmap::IndexMap;
use ndarray::Array2;

use crate::cli::Mode;
use crate::features::stats::InputStatistics;
use crate::tasks::sad::{
    BlstmSignalSegmenter, BlstmSpectralSegmenter, LtsvSegmenter, TdcSegmenter,
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
