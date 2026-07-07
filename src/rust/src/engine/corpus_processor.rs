//! Top-level driver: build corpus + processors, walk files under the static-lane
//! deterministic reduction, epoch training loop, finite-diff grad check, result
//! reduction + `.mat` write.
//!
//! Ported from legacy C++: CorpusProcessor.* (replaces the OpenMP parallel-for
//! with static lanes -- spec S3, the R6 determinism fix).
//!
//! ## The static-lane model (the ONE deliberate deviation from legacy parallelism)
//!
//! The legacy `run(epoch, ...)` (`CorpusProcessor.cpp:172-203`) is an
//! `#pragma omp parallel for ... schedule(dynamic, 1)` over files, folding each
//! file's contribution inside an `omp critical` block. OpenMP's dynamic schedule
//! makes the FOLD ORDER nondeterministic even at a fixed thread count -- and since
//! the derivative/input-statistics accumulation (`:184-199`) is order-sensitive
//! (float `+=`, `InputStatistics::update` merge), the reduced values are not
//! reproducible run to run.
//!
//! This port fixes the fold to a DETERMINISTIC order: file `j` is assigned to lane
//! `j % N` (`N = nb_of_threads` clamped to `[1, nb_files]`); each lane clones the
//! epoch-start bag ONCE and walks its files in ascending `j` (so genuine cross-file
//! driver state chains within a lane); after the parallel section the per-file
//! contributions are folded in ASCENDING FILE INDEX (`0, 1, 2, ...`), transcribing
//! the critical-section body (`:178-202`) in that fixed order. At `N == 1` this is
//! byte-identical to a plain sequential loop -- the golden-pinned parity mode
//! (`lanes_n1_equals_sequential`). For `N > 1` the results are still deterministic
//! (fixed for a fixed N) but DIFFER from the legacy's nondeterministic run and,
//! because state chains per lane, from `N == 1`. See IMPROVEMENTS.md
//! (`[phase4a] static-lane deterministic reduction`).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Result, bail};
use indexmap::IndexMap;
use ndarray::Array2;

use crate::cli::{Mode, ModeKind};
use crate::engine::bag_of_processors::BagOfProcessors;
use crate::engine::corpus::Corpus;
use crate::features::stats::InputStatistics;
use crate::io::matfile::MatWriter;

/// Result of the corpus-level gradient check (`gradCheck`, `:237-340`). The legacy
/// only prints per-weight lines + the two means; this struct RETURNS them so Task
/// 9's golden can replay the check. `per_weight[k] = (backprop_normalized,
/// numerical, abs_diff)` = `(derivs(k,0)/derivs(k,1), (c+ - c-)/(2 eps),
/// |backprop_normalized - numerical|)`.
#[derive(Debug, Clone)]
pub struct GradCheckReport {
    /// Mean absolute error over the checked weights (`:333`).
    pub mean_error: f64,
    /// Mean relative error, ref floored at 1e-24 (`:329-331,334`).
    pub mean_relative_error: f64,
    /// Per-weight `(backprop_normalized, numerical, abs_diff)` triples.
    pub per_weight: Vec<(f64, f64, f64)>,
}

/// Port of `CorpusProcessor` (`CorpusProcessor.h`/`.cpp`): mode dispatch, the epoch
/// loop, corpus-level gradient check, and `.mat` result reduction.
///
/// `results` is `file -> conf -> chan -> 18-col row` (the legacy nested `map`
/// keyed identically). `res_per_conf`/`results_e` are the transformed matrices
/// (`transformResults`, `:342-389`). The four `*_mem` matrices accumulate per-epoch
/// cost/badClassif/costLID/badLIDClassif rows.
pub struct CorpusProcessor {
    training_epochs: usize,
    epsilon: f64,
    corpus: Corpus,
    processors: BagOfProcessors,
    mode: Mode,
    output_file_name: String,
    results: BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, Vec<f64>>>>,
    res_per_conf: Vec<Array2<f64>>,
    results_e: Array2<f64>,
    cost_mem: Array2<f64>,
    bad_classif_mem: Array2<f64>,
    cost_lid_mem: Array2<f64>,
    bad_classif_lid_mem: Array2<f64>,
    best_cost: BTreeMap<usize, f64>,
}

impl CorpusProcessor {
    /// Port of `CorpusProcessor(vector<ConfigFile>, char*)` (`:48-72`).
    pub fn new(mut configs: Vec<IndexMap<String, String>>, mode: Mode) -> Result<CorpusProcessor> {
        if configs.is_empty() {
            bail!("CorpusProcessor::new: at least one config is required");
        }

        // legacy: :50 _Corpus = Corpus(configs[0]).
        let corpus = Corpus::from_config(&configs[0])?;
        // legacy: :52 _OutputFileName default MultiConfigResults.mat -- read BEFORE
        // the bag ctor clears no keys relevant to it.
        let output_file_name = configs[0]
            .get("multiConfigResultsOutputFile")
            .cloned()
            .unwrap_or_else(|| "MultiConfigResults.mat".to_string());
        // legacy: :51 _Processors = BagOfProcessors(configs, _Mode). The bag ctor
        // mutates each config map (key clears :21-23).
        let processors = BagOfProcessors::from_configs(&mut configs, mode)?;

        // legacy: :53-59 output-file TRUNCATION for m/M/t/T -- write a single space,
        // so a subsequent read (or a run that never saves) sees a valid-but-empty
        // file rather than a stale prior run's .mat.
        if matches!(mode.kind, ModeKind::Multi | ModeKind::UnitTest) {
            std::fs::write(&output_file_name, " ")?;
        }

        // legacy: :60-62 epochs clamp (< 0 -> 0).
        let training_epochs = configs[0]
            .get("Neural_Networks_BackPropagation_Epochs")
            .and_then(|s| s.trim().parse::<i32>().ok())
            .unwrap_or(0)
            .max(0) as usize;
        // legacy: :63 epsilon default 0.0.
        let epsilon = configs[0]
            .get("Neural_Networks_Gradient_Check_Epsilon")
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(0.0);

        // legacy: :64-66 _BestCost[ii] = 1e20 seeds.
        let mut best_cost = BTreeMap::new();
        for ii in 0..processors.nb_of_conf() {
            best_cost.insert(ii, 1e20);
        }

        let mut training_epochs = training_epochs;
        let mut epsilon = epsilon;
        // legacy: :67-71 isWorkDistributed kill -- cannot train or grad-check when
        // work is distributed across lock files.
        if processors.is_work_distributed() {
            training_epochs = 0;
            epsilon = 0.0;
        }

        let nb_of_conf = processors.nb_of_conf();
        Ok(CorpusProcessor {
            training_epochs,
            epsilon,
            corpus,
            processors,
            mode,
            output_file_name,
            results: BTreeMap::new(),
            res_per_conf: Vec::new(),
            results_e: Array2::zeros((0, 0)),
            cost_mem: Array2::zeros((1, nb_of_conf)),
            bad_classif_mem: Array2::zeros((1, nb_of_conf)),
            cost_lid_mem: Array2::zeros((1, nb_of_conf)),
            bad_classif_lid_mem: Array2::zeros((1, nb_of_conf)),
            best_cost,
        })
    }

    /// Port of `CorpusProcessor::run()` (`:113-137`): the top-level dispatch.
    ///
    /// The mode-letter gates (`:116/:122/:129`) allow train/gradCheck only for
    /// m/M/t/T/i/I. The legacy `cerr` at `:119/:126/:132` does NOT exit -- it prints
    /// and CONTINUES; this port mirrors that with `eprintln!` + fall-through
    /// (matching the observable no-run outcome, not an error).
    pub fn run(&mut self) -> Result<()> {
        let allowed = matches!(
            self.mode.kind,
            ModeKind::Multi | ModeKind::UnitTest | ModeKind::Image
        );
        if self.epsilon > 0.0 {
            if self.training_epochs > 0 {
                if allowed {
                    self.train()?;
                } else {
                    // legacy: :119 cerr + continue (no exit).
                    eprintln!(
                        "You must use the modes m, t or i to be able to train a neural network !"
                    );
                }
            }
            if allowed {
                // legacy: :123 _Mode[1] = 'm' -- permanent flip to non-verbose Multi
                // before the grad check (the run() dispatch mutates the processor's
                // mode). ModeKind::Multi, verbose false.
                self.mode = Mode {
                    kind: ModeKind::Multi,
                    verbose: false,
                };
                self.grad_check(self.epsilon)?;
            } else {
                // legacy: :126 cerr + continue.
                eprintln!(
                    "You must use the modes m, t or i to be able to perform a gradient check !"
                );
            }
        } else if self.training_epochs > 0 {
            if allowed {
                self.train()?;
            } else {
                // legacy: :132 cerr + continue.
                eprintln!(
                    "You must use the modes m, t or i to be able to train a neural network !"
                );
            }
        } else {
            self.run_solo()?;
        }
        Ok(())
    }

    /// Port of `CorpusProcessor::runSolo()` (`:74-81`): the four mem matrices sized
    /// `1 x nb_of_conf`, then one `run(0, _Mode, {}, true)` with an EMPTY derivs map
    /// (no config is asked for its derivatives, so the harvest never fires).
    fn run_solo(&mut self) -> Result<()> {
        let nb = self.processors.nb_of_conf();
        self.cost_mem = Array2::zeros((1, nb));
        self.bad_classif_mem = Array2::zeros((1, nb));
        self.cost_lid_mem = Array2::zeros((1, nb));
        self.bad_classif_lid_mem = Array2::zeros((1, nb));
        let mut derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
        self.run_epoch(0, self.mode, &mut derivs, true)
    }

    /// Port of `CorpusProcessor::train()` (`:83-111`): the four mem matrices sized
    /// `(_TrainingEpochs+2) x nb_of_conf`; an epoch-0 solo run with `_Mode`; then
    /// `_TrainingEpochs` inner epochs with a LOCAL lowercase non-verbose `m` mode
    /// (`isLog = false`); a final eval at epoch `N+1` with `_Mode`. Timer prints are
    /// display-only (dropped).
    fn train(&mut self) -> Result<()> {
        let nb = self.processors.nb_of_conf();
        let rows = self.training_epochs + 2;
        self.cost_mem = Array2::zeros((rows, nb));
        self.bad_classif_mem = Array2::zeros((rows, nb));
        self.cost_lid_mem = Array2::zeros((rows, nb));
        self.bad_classif_lid_mem = Array2::zeros((rows, nb));

        // legacy: :92 epoch-0 solo with _Mode, isLog true, fresh derivs.
        let mut derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
        self.run_epoch(0, self.mode, &mut derivs, true)?;

        // legacy: :96-104 inner epochs -- LOCAL mode = lowercase non-verbose 'm'
        // (Multi, verbose false), isLog false, derivs cleared each epoch.
        let inner_mode = Mode {
            kind: ModeKind::Multi,
            verbose: false,
        };
        for mm in 0..self.training_epochs {
            let mut derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
            self.run_epoch(mm + 1, inner_mode, &mut derivs, false)?;
        }

        // legacy: :106-110 final eval at epoch _TrainingEpochs+1 with _Mode, isLog true.
        let mut derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
        self.run_epoch(self.training_epochs + 1, self.mode, &mut derivs, true)
    }

    /// Port of `CorpusProcessor::run(epoch, mode, backPropWeightsDerivatives,
    /// isLogActivated)` (`:140-235`): the static-lane file walk + deterministic fold,
    /// `transformResults`, and the `saveAndUpdate` + `saveResults` gating.
    ///
    /// `mode` is the epoch-local mode (the legacy passes `_Mode` here despite the
    /// `mode` param -- `:177` reads `_Mode`, NOT the passed `mode`; reproduced: the
    /// per-file `segmentation_function` sees `self.mode`, and the passed `mode` only
    /// distinguishes the isLog/save-gating epoch role). The `backPropWeightsDerivatives`
    /// map: a conf `ii` present in the map (even with an empty Vec) requests the
    /// per-file derivative harvest for that conf (`:184`); an ABSENT conf is never
    /// harvested. `run_solo`/`train`'s epoch-0 pass an empty map -> no harvest.
    fn run_epoch(
        &mut self,
        epoch: usize,
        _mode: Mode,
        derivs: &mut BTreeMap<usize, Vec<Array2<f64>>>,
        is_log: bool,
    ) -> Result<()> {
        let _ = is_log; // the per-file cout is display-only (dropped).
        let nb_files = self.corpus.nb_of_files();

        // legacy: :141-145 numOuterThreads clamp: < 1 -> 1, > nb_files -> nb_files.
        let mut n = self.processors.nb_of_threads();
        if n < 1 {
            n = 1;
        }
        if n > nb_files as i32 {
            n = nb_files as i32;
        }
        let n = n.max(1) as usize;

        // legacy: :147-148 clear per epoch.
        self.res_per_conf.clear();
        self.results.clear();
        let mut input_statistics: BTreeMap<usize, Vec<InputStatistics>> = BTreeMap::new();

        // === Static-lane parallel section ===================================
        // Lane `l` owns files `{ j : j % n == l }` in ascending j. Each lane clones
        // the epoch-start bag once (`firstprivate(processors)`, :172) and walks its
        // files, collecting per-file (j, results_j, derivs_j, stats_j). rayon::scope
        // with one spawn per lane, each writing its own Vec (no shared mutation).
        //
        // The legacy harvests EVERY conf on every processed file (`:183-199` loops
        // `ii < getNbOfConf()` UNCONDITIONALLY -- the create-vs-`+=` on the caller's
        // map is decided by `count(ii)`, but the harvest itself is never gated on the
        // map's initial contents). So `run_solo`/`train`/`gradCheck`'s fresh empty
        // map STILL ends up populated for all confs (algo 1/2 harvest an empty vec,
        // algo 3/4 the real Nx2). This is why the analytic gradCheck run at `:252`,
        // passing a fresh map, still fills `backPropWeightsDerivatives[0][ii]`.
        let nb_of_conf = self.processors.nb_of_conf();

        type LaneEntry = (
            usize,                                      // file index j
            BTreeMap<usize, BTreeMap<usize, Vec<f64>>>, // results_j (conf -> chan -> row)
            BTreeMap<usize, Vec<Array2<f64>>>,          // derivs_j (all confs)
            BTreeMap<usize, Vec<InputStatistics>>,      // stats_j (all confs)
        );
        type LaneOut = Result<Vec<LaneEntry>>;

        // Snapshot the immutable inputs each lane needs (the corpus items, the bag
        // to clone, the mode). The bag clone is per lane.
        let base_bag = &self.processors;
        let corpus = &self.corpus;
        let self_mode = self.mode;

        // Each lane writes a Result into its slot; a per-file error surfaces as an
        // Err (propagated after the scope) rather than a panic that poisons shared
        // state. Init to Ok(empty).
        let mut lane_outputs: Vec<LaneOut> = (0..n).map(|_| Ok(Vec::new())).collect();

        rayon::scope(|scope| {
            for (lane, slot) in lane_outputs.iter_mut().enumerate() {
                scope.spawn(move |_| {
                    // Clone the epoch-start bag ONCE for this lane (:172
                    // firstprivate); walk this lane's files in ascending j.
                    let mut bag = base_bag.clone();
                    let mut out: Vec<LaneEntry> = Vec::new();
                    let mut j = lane;
                    while j < nb_files {
                        let item = corpus.item(j);
                        // The legacy passes _Mode to SegmentationFunction (:177).
                        let tmp = match bag.segmentation_function(item, self_mode) {
                            Ok(t) => t,
                            Err(e) => {
                                *slot = Err(e);
                                return;
                            }
                        };
                        // Per-file harvest for EVERY conf (:183-199): read the LANE
                        // bag's post-file derivs/stats (drivers reset per file inside
                        // get_segmentation, so this is file j's own contribution).
                        let mut derivs_j: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
                        let mut stats_j: BTreeMap<usize, Vec<InputStatistics>> = BTreeMap::new();
                        if !tmp.is_empty() {
                            for ii in 0..nb_of_conf {
                                derivs_j.insert(ii, bag.get_weights_derivatives(ii));
                                stats_j.insert(ii, bag.get_input_statistics(ii));
                            }
                        }
                        out.push((j, tmp, derivs_j, stats_j));
                        j += n;
                    }
                    *slot = Ok(out);
                });
            }
        });

        // === Deterministic fold in ASCENDING FILE INDEX =====================
        // Flatten the lane outputs and sort by file index. `j % n` partitions
        // files disjointly across lanes, so a plain sort restores 0,1,2,... . A lane
        // that hit a per-file error propagates it here.
        let mut per_file: Vec<LaneEntry> = Vec::new();
        for lane_result in lane_outputs {
            per_file.extend(lane_result?);
        }
        per_file.sort_by_key(|(j, _, _, _)| *j);

        for (j, tmp, derivs_j, stats_j) in per_file {
            // legacy: :181 if (tmp.size() > 0).
            if tmp.is_empty() {
                continue;
            }
            // legacy: :182 _Results[jj] = tmp.
            self.results.insert(j, tmp);
            // legacy: :183-199 per-conf derivative + input-stats accumulation.
            for ii in 0..nb_of_conf {
                // Only confs the caller requested were collected AND are folded
                // (:184 count(ii) != 0 -- an absent conf is never touched here).
                if let Some(tmp_derivs) = derivs_j.get(&ii) {
                    // legacy: :184-191 first file CREATES by move, later files += .
                    match derivs.get_mut(&ii) {
                        Some(existing) if !existing.is_empty() => {
                            for (kk, mat) in tmp_derivs.iter().enumerate() {
                                existing[kk] = &existing[kk] + mat;
                            }
                        }
                        _ => {
                            derivs.insert(ii, tmp_derivs.clone());
                        }
                    }
                }
                if let Some(tmp_stats) = stats_j.get(&ii) {
                    // legacy: :192-198 first file COPIES, later files .update() merge.
                    match input_statistics.get_mut(&ii) {
                        Some(existing) if !existing.is_empty() => {
                            for (kk, s) in tmp_stats.iter().enumerate() {
                                existing[kk].update(s);
                            }
                        }
                        _ => {
                            input_statistics.insert(ii, tmp_stats.clone());
                        }
                    }
                }
            }
        }

        // legacy: :215-234 transform + saveAndUpdate + saveResults gating.
        if !self.results.is_empty() {
            self.transform_results();
            if self.training_epochs > 0 {
                // legacy: :217-224 training branch -- saveAndUpdate + saveResults
                // ALWAYS (every epoch, including the final epoch > _TrainingEpochs).
                self.save_and_update_epoch(epoch, derivs, &input_statistics)?;
                self.save_results(epoch)?;
            } else {
                // legacy: :226-230 non-training branch -- only when epoch <=
                // _TrainingEpochs (i.e. epoch == 0, since _TrainingEpochs == 0).
                if epoch <= self.training_epochs {
                    self.save_and_update_epoch(epoch, derivs, &input_statistics)?;
                    self.save_results(epoch)?;
                }
            }
        } else {
            // legacy: :232-234 results EMPTY -> saveResults(epoch) still fires.
            self.save_results(epoch)?;
        }
        Ok(())
    }

    /// Extract the epoch's mem rows as mutable slices and hand them to the bag's
    /// `save_and_update` (the legacy `_CostMem.block(epoch,0,1,cols)` row views).
    fn save_and_update_epoch(
        &mut self,
        epoch: usize,
        derivs: &BTreeMap<usize, Vec<Array2<f64>>>,
        stats: &BTreeMap<usize, Vec<InputStatistics>>,
    ) -> Result<()> {
        // The mem matrices are row-major Array2; a single row is a contiguous slice.
        let nb = self.processors.nb_of_conf();
        let mut cost_row = vec![0.0; nb];
        let mut bad_row = vec![0.0; nb];
        let mut cost_lid_row = vec![0.0; nb];
        let mut bad_lid_row = vec![0.0; nb];

        self.processors.save_and_update(
            &self.output_file_name,
            &self.res_per_conf,
            &mut self.best_cost,
            derivs,
            stats,
            &mut cost_row,
            &mut bad_row,
            &mut cost_lid_row,
            &mut bad_lid_row,
        )?;

        for ii in 0..nb {
            self.cost_mem[[epoch, ii]] = cost_row[ii];
            self.bad_classif_mem[[epoch, ii]] = bad_row[ii];
            self.cost_lid_mem[[epoch, ii]] = cost_lid_row[ii];
            self.bad_classif_lid_mem[[epoch, ii]] = bad_lid_row[ii];
        }
        Ok(())
    }

    /// Port of `CorpusProcessor::transformResults()` (`:342-389`): ascending
    /// file-key iteration (BTreeMap), rows `[file+1, conf+1, chan+1, res...]`,
    /// per-conf matrices pre-sized `2*nb_files` rows then truncated to the counters
    /// (`conservativeResize` == truncate; row width from the FIRST result's len).
    fn transform_results(&mut self) {
        let (results_e, res_per_conf) = Self::transform_results_impl(
            &self.results,
            self.processors.nb_of_conf(),
            self.corpus.nb_of_files(),
        );
        self.results_e = results_e;
        self.res_per_conf = res_per_conf;
    }

    /// Pure core of [`Self::transform_results`], parameterized for the unit test.
    fn transform_results_impl(
        results: &BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, Vec<f64>>>>,
        nb_of_conf: usize,
        nb_of_files: usize,
    ) -> (Array2<f64>, Vec<Array2<f64>>) {
        // legacy: :343-344 pre-sizes.
        let nb_elements = 2 * nb_of_conf * nb_of_files;
        let nb_elements_per_conf = 2 * nb_of_files;

        let mut counter = 0usize;
        let mut counters = vec![0usize; nb_of_conf];
        let mut initialized = false;
        let mut results_e: Array2<f64> = Array2::zeros((0, 0));
        let mut res_per_conf: Vec<Array2<f64>> = Vec::new();
        let mut res_width = 0usize;

        // legacy: :352 BOOST_FOREACH over _Results (BTreeMap == ascending file key).
        for (file_idx, mmap) in results {
            if mmap.is_empty() {
                continue;
            }
            if !initialized {
                // legacy: CorpusProcessor.cpp:357 - mmap[0][0] operator[] default-constructs
                // an empty vector if conf 0 / chan 0 is absent, rather than reading whatever
                // conf/chan IS present. Match that exactly, including the degenerate width-0
                // case, instead of taking the first present entry.
                res_width = mmap
                    .get(&0)
                    .and_then(|chan_map| chan_map.get(&0))
                    .map(|v| v.len())
                    .unwrap_or(0);
                results_e = Array2::zeros((nb_elements, 3 + res_width));
                res_per_conf = (0..nb_of_conf)
                    .map(|_| Array2::zeros((nb_elements_per_conf, res_width)))
                    .collect();
                initialized = true;
            }
            // legacy: :365-382 per conf (ascending) then per chan (ascending).
            for (conf_idx, chan_map) in mmap {
                let config_nb = conf_idx + 1;
                for (chan_idx, res) in chan_map {
                    let chan_nb = chan_idx + 1;
                    // legacy: :375-377 line = [file+1, conf+1, chan+1, res...].
                    results_e[[counter, 0]] = (file_idx + 1) as f64;
                    results_e[[counter, 1]] = config_nb as f64;
                    results_e[[counter, 2]] = chan_nb as f64;
                    for (c, &v) in res.iter().enumerate() {
                        results_e[[counter, 3 + c]] = v;
                    }
                    counter += 1;
                    // legacy: :379-380 _ResPerConf[conf].row(counters[conf]) = res.
                    let cc = config_nb - 1;
                    let row = counters[cc];
                    for (c, &v) in res.iter().enumerate() {
                        res_per_conf[cc][[row, c]] = v;
                    }
                    counters[cc] += 1;
                }
            }
        }

        // legacy: :385-388 conservativeResize (truncate) to the counters.
        if !initialized {
            // No results at all -> empty ResultsE, empty per-conf mats.
            return (Array2::zeros((0, 0)), Vec::new());
        }
        let truncated_e = truncate_rows(&results_e, counter);
        let truncated_per_conf = (0..nb_of_conf)
            .map(|ii| truncate_rows(&res_per_conf[ii], counters[ii]))
            .collect();
        let _ = res_width;
        (truncated_e, truncated_per_conf)
    }

    /// Port of `CorpusProcessor::saveResults(epoch)` (`:391-404`): the 5 named
    /// variables in this exact order via [`MatWriter`], each `topRows(epoch+1)`.
    /// The legacy `Mat_Create` failure -> `exit(1)`; here a create error bails.
    fn save_results(&self, epoch: usize) -> Result<()> {
        let mut w = MatWriter::create(Path::new(&self.output_file_name))?;
        // legacy: :397 MultiConfigResults = _ResultsE (whole).
        write_row_major(&mut w, "MultiConfigResults", &self.results_e)?;
        // legacy: :398-401 each mem topRows(epoch+1).
        let top = epoch + 1;
        write_row_major(&mut w, "CostMem", &top_rows(&self.cost_mem, top))?;
        write_row_major(
            &mut w,
            "BadClassifMem",
            &top_rows(&self.bad_classif_mem, top),
        )?;
        write_row_major(&mut w, "CostLIDMem", &top_rows(&self.cost_lid_mem, top))?;
        write_row_major(
            &mut w,
            "BadClassifLIDMem",
            &top_rows(&self.bad_classif_lid_mem, top),
        )?;
        w.finish()?;
        Ok(())
    }

    /// Port of `CorpusProcessor::gradCheck(epsilon)` (`:237-340`). Snapshots the
    /// WHOLE bag, zeros the four mem matrices to `1 x nb_of_conf`, sets
    /// `_TrainingEpochs = 0`, then for each network `ii` of config 0 with backprop
    /// active: one analytic `run(1, _Mode, derivs, true)` then per weight `k` the
    /// +eps / -eps central difference, comparing the numerical gradient to the
    /// analytic normalized `derivs(k,0)/derivs(k,1)`.
    ///
    /// `max_weights` caps the sweep (DEVIATION from the legacy full sweep, recorded
    /// in the manifest as `gradcheck_max_weights`; the legacy checks every weight).
    /// Returns the last-processed network's [`GradCheckReport`] (config 0 has a
    /// single NN for algo 3/4, so there is exactly one).
    fn grad_check(&mut self, epsilon: f64) -> Result<GradCheckReport> {
        self.grad_check_capped(epsilon, usize::MAX)
    }

    fn grad_check_capped(&mut self, epsilon: f64, max_weights: usize) -> Result<GradCheckReport> {
        // legacy: :238 snapshot the WHOLE bag.
        let proc_mem = self.processors.clone();
        let nb = self.processors.nb_of_conf();
        // legacy: :239-242 zero the four mem matrices to 1 x nConfs.
        self.cost_mem = Array2::zeros((1, nb));
        self.bad_classif_mem = Array2::zeros((1, nb));
        self.cost_lid_mem = Array2::zeros((1, nb));
        self.bad_classif_lid_mem = Array2::zeros((1, nb));
        // legacy: :243 _TrainingEpochs = 0.
        self.training_epochs = 0;

        let mut last_report: Option<GradCheckReport> = None;

        // legacy: :245-246 for each network ii of config 0 with backprop active.
        let back_prop_activated = self.processors.is_back_prop_activated(0);
        for (ii, &active) in back_prop_activated.iter().enumerate() {
            if !active {
                // legacy: :337 backprop-off network -- skip the check.
                continue;
            }
            // legacy: :248-249 NNWeight / newNNWeights = getWeights(0).
            let nn_weight = self.processors.get_weights(0);
            let mut new_nn_weights = nn_weight.clone();

            let algo0 = self.processors.algo_type(0);
            if !(algo0 == 3 || algo0 == 4) {
                // The algo 5/6 arms of the cost-column selection are unreachable in
                // 4a (bag construction bails on those algos).
                unreachable!("gradCheck cost columns only defined for algo 3/4 in 4a");
            }

            // legacy: :251-252 one analytic run(1, _Mode, derivs, true) with a FRESH
            // (empty) derivs map -- the harvest populates all confs via the create
            // branch. `analytic_derivs[0][ii]` then carries the analytic Nx2.
            let mut analytic_derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
            self.run_epoch(1, self.mode, &mut analytic_derivs, true)?;

            let weights_nb = nn_weight[ii].len();
            let sweep_len = weights_nb.min(max_weights);

            let mut grad_error = 0.0;
            let mut grad_rel_error = 0.0;
            let mut per_weight = Vec::with_capacity(sweep_len);

            let inner_mode = Mode {
                kind: ModeKind::Multi,
                verbose: false,
            };

            for kk in 0..sweep_len {
                // legacy: :264-267 +eps: restore bag, perturb, setWeights.
                new_nn_weights[ii] = nn_weight[ii].clone();
                new_nn_weights[ii][kk] += epsilon;
                self.processors = proc_mem.clone();
                self.processors.set_weights(0, &new_nn_weights)?;
                let mut tmp_derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
                self.run_epoch(1, inner_mode, &mut tmp_derivs, false)?;
                let cost_plus = self.grad_check_cost(algo0);

                // legacy: :295-297 -eps: perturb -2*eps from the +eps point, restore.
                new_nn_weights[ii][kk] -= 2.0 * epsilon;
                self.processors = proc_mem.clone();
                self.processors.set_weights(0, &new_nn_weights)?;
                let mut tmp_derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
                self.run_epoch(1, inner_mode, &mut tmp_derivs, false)?;
                let cost_minus = self.grad_check_cost(algo0);

                // legacy: :326 numerical = (c+ - c-)/(2 eps).
                let numerical = (cost_plus - cost_minus) / (2.0 * epsilon);
                // legacy: :327-328 backprop normalized = derivs(k,0)/derivs(k,1).
                let dmat = &analytic_derivs[&0][ii];
                let backprop = dmat[[kk, 0]] / dmat[[kk, 1]];
                let abs_diff = (backprop - numerical).abs();

                // legacy: :328 mean abs error.
                grad_error += abs_diff;
                // legacy: :329-331 relative error, ref floored at 1e-24.
                let mut refv = numerical.abs();
                if refv < 1e-24 {
                    refv = 1e-24;
                }
                grad_rel_error += abs_diff / refv;

                per_weight.push((backprop, numerical, abs_diff));
            }

            // legacy: :333-334 mean over the checked weights.
            let denom = sweep_len.max(1) as f64;
            let report = GradCheckReport {
                mean_error: grad_error / denom,
                mean_relative_error: grad_rel_error / denom,
                per_weight,
            };
            last_report = Some(report);
        }

        // Restore the bag to the snapshot (the sweep left it perturbed). Port addition,
        // no legacy counterpart (legacy leaves _Processors perturbed) -- unobservable
        // since gradCheck is terminal in run().
        self.processors = proc_mem;

        last_report
            .ok_or_else(|| anyhow::anyhow!("gradCheck: no backprop-activated network in config 0"))
    }

    /// The cost accumulation over `self.results` for the gradCheck central
    /// difference (`:271-292`): for algo 3/4, `cost += results[jj][0][chan][4]`,
    /// `counter += results[jj][0][chan][last]`, then `cost /= counter`. Algo 5/6
    /// arms are unreachable in 4a.
    fn grad_check_cost(&self, algo0: i32) -> f64 {
        let mut cost = 0.0;
        let mut counter = 0.0;
        for jj in 0..self.corpus.nb_of_files() {
            let Some(conf0) = self.results.get(&jj).and_then(|f| f.get(&0)) else {
                continue;
            };
            for row in conf0.values() {
                if algo0 == 3 || algo0 == 4 {
                    cost += row[4];
                    counter += row[row.len() - 1];
                } else {
                    unreachable!("gradCheck cost columns only defined for algo 3/4 in 4a");
                }
            }
        }
        // legacy: :292 cost /= counter (integer counter -> float division; a 0
        // counter yields inf/NaN, mirrored here).
        cost / counter
    }

    // ==== Test-only hooks (pub for the integration test) ====================

    /// Pure `transformResults` core (test hook for `transform_results_ordering`).
    #[doc(hidden)]
    pub fn transform_results_for_test(
        results: &BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, Vec<f64>>>>,
        nb_of_conf: usize,
        nb_of_files: usize,
    ) -> (Array2<f64>, Vec<Array2<f64>>) {
        Self::transform_results_impl(results, nb_of_conf, nb_of_files)
    }

    /// Corpus-level gradient check with a weight cap (test hook for
    /// `grad_check_synthetic`; Task 9's harness golden replays it).
    #[doc(hidden)]
    pub fn grad_check_for_test(
        &mut self,
        epsilon: f64,
        max_weights: usize,
    ) -> Result<GradCheckReport> {
        self.grad_check_capped(epsilon, max_weights)
    }

    /// Config-0's flat weight vector (test hook for the grad-check restore assert).
    /// Empty when config 0 is a non-NN algo (no weights).
    #[doc(hidden)]
    pub fn get_config0_weights_for_test(&self) -> Vec<f64> {
        self.processors
            .get_weights(0)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Seed config-0's NN with a synthetic nonzero weight vector (test hook for
    /// `grad_check_synthetic`): the default-initialized net sits at a degenerate
    /// operating point (near-zero gradients), so the gradcheck needs deterministic
    /// nonzero weights -- exactly the phase3 `synth_flat` pattern -- to exercise a
    /// smoothly-varying cost. No-op for a non-NN config 0.
    #[doc(hidden)]
    pub fn set_config0_weights_for_test(&mut self, flat: &[f64]) -> Result<()> {
        self.processors.set_weights(0, &[flat.to_vec()])
    }

    /// A hand-sequential oracle for `lanes_n1_equals_sequential`: build a fresh bag,
    /// walk the corpus files in ascending order through `segmentation_function`, and
    /// assemble the same `[file+1, conf+1, chan+1, res...]` ResultsE the engine's
    /// `transformResults` produces. Independent of the lane machinery.
    #[doc(hidden)]
    pub fn run_epoch_sequential_oracle(
        mut configs: Vec<IndexMap<String, String>>,
        mode: Mode,
    ) -> Result<Array2<f64>> {
        let corpus = Corpus::from_config(&configs[0])?;
        let mut bag = BagOfProcessors::from_configs(&mut configs, mode)?;
        let mut results: BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, Vec<f64>>>> =
            BTreeMap::new();
        for j in 0..corpus.nb_of_files() {
            let item = corpus.item(j);
            let tmp = bag.segmentation_function(item, mode)?;
            if !tmp.is_empty() {
                results.insert(j, tmp);
            }
        }
        let (results_e, _) =
            Self::transform_results_impl(&results, bag.nb_of_conf(), corpus.nb_of_files());
        Ok(results_e)
    }
}

/// Truncate an `Array2` to its first `n` rows (`conservativeResize(n, cols)`), a
/// plain row-slice copy. Live call sites shrink or keep, except the pathological
/// gradcheck empty-results path, where `top_rows` can be asked for more rows than
/// `m` has (e.g. a 1-row `cost_mem` with `epoch+1 == 2`) -- that indexes out of
/// bounds here, mirroring legacy `topRows(2)`-on-1-row UB. Faithful to legacy UB,
/// unreachable under non-empty corpora, panics by design if hit.
fn truncate_rows(m: &Array2<f64>, n: usize) -> Array2<f64> {
    let cols = m.dim().1;
    Array2::from_shape_fn((n, cols), |(i, j)| m[[i, j]])
}

/// `topRows(n)` == the first `n` rows.
fn top_rows(m: &Array2<f64>, n: usize) -> Array2<f64> {
    truncate_rows(m, n)
}

/// Write a row-major `Array2` to the (column-major) `MatWriter`, converting the
/// storage order explicitly.
fn write_row_major(w: &mut MatWriter, name: &str, m: &Array2<f64>) -> Result<()> {
    let (rows, cols) = m.dim();
    let mut col_major = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            col_major.push(m[[i, j]]);
        }
    }
    w.write_matrix(name, rows, cols, &col_major)?;
    Ok(())
}
