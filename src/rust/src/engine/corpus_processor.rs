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

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;
use ndarray::Array2;

use crate::cli::{Mode, ModeKind};
use crate::engine::bag_of_processors::{BagOfProcessors, Processor, get_i32_default};
use crate::engine::channel_result::ChannelResult;
use crate::engine::corpus::Corpus;
use crate::features::stats::InputStatistics;
use crate::io::matfile::MatWriter;
use crate::legacy_config::get_f64_default;

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
/// `results` is `file -> conf -> chan -> ChannelResult` (the legacy nested `map`
/// keyed identically, its 18+N-col row rendered by `ChannelResult::to_row`).
/// `results_e` is the rendered `MultiConfigResults` matrix (`transformResults`,
/// `:342-389`); the per-config lists the fold reads are built from `results` at
/// `saveAndUpdate` time ([`Self::results_per_config`]). The four `*_mem` matrices
/// accumulate per-epoch cost/badClassif/costLID/badLIDClassif rows.
pub struct CorpusProcessor {
    training_epochs: usize,
    epsilon: f64,
    corpus: Corpus,
    processors: BagOfProcessors,
    mode: Mode,
    output_file_name: String,
    results: BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, ChannelResult>>>,
    results_e: Array2<f64>,
    cost_mem: Array2<f64>,
    bad_classif_mem: Array2<f64>,
    cost_lid_mem: Array2<f64>,
    bad_classif_lid_mem: Array2<f64>,
    best_cost: BTreeMap<usize, f64>,
    /// F10 (phase 5): the folded per-conf derivatives from the LAST completed
    /// `run_epoch` fold, keyed by conf index (each value the per-network `Nx2`
    /// vec, `[sad]` or `[sad, lid]`). The public [`Self::weights_derivatives`]
    /// returns this so the release seam sees the gradient `run_epoch` actually
    /// folded on the per-lane clones (the R6 static-lane model), NOT the main
    /// bag's never-updated accumulator. Written at the end of every fold;
    /// CLEARED by [`Self::set_weights`] (a weights change invalidates the cached
    /// gradient, so a set-without-run correctly falls back to the bag's reset
    /// state). Empty until the first fold. See IMPROVEMENTS.md `[phase5] F10`.
    seam_derivs: BTreeMap<usize, Vec<Array2<f64>>>,
    /// Test-observation hook (Task 9): after each `save_and_update_epoch` in a
    /// training run, snapshot config-0's flat weight vector AND whether the
    /// best-cost gate fired that epoch. No legacy counterpart; the tier-2 train
    /// golden replays this to pin the epoch-chained weight trajectory + the
    /// gate fire/skip non-vacuity. Only allocated under `test-support`.
    #[cfg(feature = "test-support")]
    epoch_weight_trace: Vec<Vec<f64>>,
    /// Task 9: the SECOND net's (config-0 network index 1, the algo-6 LID net)
    /// per-epoch flat weights. Empty vecs when config 0 has a single net.
    #[cfg(feature = "test-support")]
    epoch_weight_trace_lid: Vec<Vec<f64>>,
    #[cfg(feature = "test-support")]
    epoch_best_cost_trace: Vec<f64>,
    /// Test-observation hook (Task 10, fold-order golden): config-0's RAW
    /// accumulated derivative matrix (the first net's `[deriv, count]` pair
    /// per weight) right after `run_epoch`'s per-file fold completes, one
    /// snapshot per epoch -- BEFORE `save_and_update_epoch`'s Rprop update
    /// consumes it. Unlike `epoch_weight_trace` (the POST-Rprop weights),
    /// this is sensitive to the raw fold order: iRPROP- (`nn/train.rs`) only
    /// ever reacts to the SIGN of the normalized derivative, so a fold-order
    /// perturbation that does not flip a sign is invisible in the trained
    /// weights but fully visible here. No legacy counterpart (a pure
    /// port-side determinism-contract probe). Empty entries for a non-NN
    /// config 0 or an epoch that produced no results.
    #[cfg(feature = "test-support")]
    epoch_raw_derivs_trace: Vec<Array2<f64>>,
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
            std::fs::write(&output_file_name, " ")
                .with_context(|| format!("cannot create `{output_file_name}`"))?;
        }

        // legacy: :60-62 epochs clamp (< 0 -> 0). get<T>(name, default) exits(1) on
        // a present-but-unparseable value -- ported as Err, not a silent fallback.
        let training_epochs =
            get_i32_default(&configs[0], "Neural_Networks_BackPropagation_Epochs", 0)?.max(0)
                as usize;
        // legacy: :63 epsilon default 0.0, same present-but-malformed -> Err semantics.
        let epsilon = get_f64_default(&configs[0], "Neural_Networks_Gradient_Check_Epsilon", 0.0)?;

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
            results_e: Array2::zeros((0, 0)),
            cost_mem: Array2::zeros((1, nb_of_conf)),
            bad_classif_mem: Array2::zeros((1, nb_of_conf)),
            cost_lid_mem: Array2::zeros((1, nb_of_conf)),
            bad_classif_lid_mem: Array2::zeros((1, nb_of_conf)),
            best_cost,
            seam_derivs: BTreeMap::new(),
            #[cfg(feature = "test-support")]
            epoch_weight_trace: Vec::new(),
            #[cfg(feature = "test-support")]
            epoch_weight_trace_lid: Vec::new(),
            #[cfg(feature = "test-support")]
            epoch_best_cost_trace: Vec::new(),
            #[cfg(feature = "test-support")]
            epoch_raw_derivs_trace: Vec::new(),
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
                self.grad_check_full(self.epsilon)?;
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
        // T5 finding 2 (review, an exact-tree touch sanctioned as selection-plumbing):
        // `Inference_Path fast` is inference-only (spec S1) -- the fast processors'
        // training arms are inert no-ops. `BagOfProcessors::from_configs` already bails on
        // a training-shaped fast config, but ONLY when `mode.kind == Multi`; that gate
        // never fires for Image/UnitTest, so a fast processor reaching `train()` via those
        // modes would otherwise silently no-op-train instead of failing loudly. `train()`
        // is only ever called from `run()`, and only inside a `self.training_epochs > 0`
        // guard (both call sites), so that half of the condition is already a structural
        // invariant here -- this loop only needs to check for a fast variant. DEAD CODE on
        // every exact-path run: no config ever builds `Processor::FastSpectral`/
        // `Processor::FastTwinLid` outside `Inference_Path fast`, so the match below never
        // fires there.
        for ii in 0..self.processors.nb_of_conf() {
            if matches!(
                self.processors.processor(ii),
                Processor::FastSpectral(_) | Processor::FastTwinLid(_)
            ) {
                bail!(
                    "Inference_Path fast is inference-only (training stays exact f64), but \
                     this {:?}-mode run has Neural_Networks_BackPropagation_Epochs {} > 0 \
                     (the fast processors' training arms are inert, so this would silently \
                     no-op-train); train on the exact path",
                    self.mode.kind,
                    self.training_epochs
                );
            }
        }

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
    /// map: legacy `:183-199` harvests EVERY conf on EVERY processed file
    /// unconditionally (`ii < getNbOfConf()`, no gate on the map's initial
    /// contents) -- a conf absent from the caller's map is CREATED by the harvest,
    /// not skipped. `run_solo`/`train`'s epoch-0 fresh empty map still ends up
    /// populated for all confs; see the static-lane section below for the full
    /// create-vs-`+=` fold detail.
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
            usize,                                           // file index j
            BTreeMap<usize, BTreeMap<usize, ChannelResult>>, // results_j (conf -> chan -> result)
            BTreeMap<usize, Vec<Array2<f64>>>,               // derivs_j (all confs)
            BTreeMap<usize, Vec<InputStatistics>>,           // stats_j (all confs)
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
                // `derivs_j`/`stats_j` carry ALL confs for this file (the lane
                // harvest above is unconditional per :183-199), so this `if let`
                // matches every `ii` when the file produced results -- it is not
                // gating on caller-requested confs.
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

        // Test-observation hook (Task 10): config-0's RAW folded derivative
        // matrix, captured HERE -- right after the per-file fold above, before
        // `save_and_update_epoch` (Rprop) consumes `derivs` below. See
        // `epoch_raw_derivs_trace`'s doc comment.
        #[cfg(feature = "test-support")]
        self.epoch_raw_derivs_trace.push(
            derivs
                .get(&0)
                .and_then(|v| v.first().cloned())
                .unwrap_or_else(|| Array2::zeros((0, 0))),
        );

        // F10 (phase 5): stash the folded gradient so the release seam
        // (`weights_derivatives`) returns the SAME values `run_epoch` folded on
        // the per-lane clones, before `save_and_update_epoch`'s Rprop consumes
        // `derivs`. This is what `grad_check`'s own local map (`analytic_derivs`,
        // filled by the identical fold) already sees; the main bag never runs
        // backprop, so without this the seam read the reset-state accumulator
        // (col0 = 0) and the whole modern loop trained on a zero gradient. An
        // empty `derivs` (no results this epoch) stashes empty -> the read falls
        // back to the bag. See IMPROVEMENTS.md `[phase5] F10`.
        self.seam_derivs = derivs.clone();

        // legacy: :215-234 transform + saveAndUpdate + saveResults gating.
        if !self.results.is_empty() {
            self.transform_results();
            if self.training_epochs > 0 {
                // legacy: :217-224 training branch -- saveAndUpdate + saveResults
                // ALWAYS (every epoch, including the final epoch > _TrainingEpochs).
                self.save_and_update_epoch(epoch, derivs, &input_statistics)?;
                #[cfg(feature = "test-support")]
                {
                    // Post-update snapshot: config-0's flat weights + the best-cost
                    // for conf 0 (its evolution reveals gate fire/skip -- the gate
                    // fires iff best_cost DROPPED this epoch). Task 9: net index 1
                    // (the algo-6 LID net) captured alongside, empty when absent.
                    self.epoch_weight_trace
                        .push(self.get_config0_weights_for_test());
                    self.epoch_weight_trace_lid.push(
                        self.processors
                            .get_weights(0)
                            .into_iter()
                            .nth(1)
                            .unwrap_or_default(),
                    );
                    self.epoch_best_cost_trace
                        .push(*self.best_cost.get(&0).unwrap_or(&f64::INFINITY));
                }
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

        // legacy: :379-380 _ResPerConf -- the per-config channel results, in the same
        // ascending (file, chan) order the matrix rows take.
        let per_conf = Self::results_per_config(&self.results, nb);
        self.processors.save_and_update(
            &self.output_file_name,
            &per_conf,
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
    /// file-key iteration (BTreeMap), rows `[file+1, conf+1, chan+1, res...]`
    /// pre-sized `2*nb_files*nb_conf` rows then truncated to the counter
    /// (`conservativeResize` == truncate; row width from the FIRST result's len).
    /// The legacy's per-conf matrices (`_ResPerConf`) are not kept: the fold reads
    /// [`Self::results_per_config`] instead.
    fn transform_results(&mut self) {
        self.results_e = Self::transform_results_impl(
            &self.results,
            self.processors.nb_of_conf(),
            self.corpus.nb_of_files(),
        );
    }

    /// The legacy `_ResPerConf[conf]` (`:379-380`) as typed values borrowed from
    /// `results`: config `ii`'s channel results in ascending (file, chan) order,
    /// the order the fold sums in.
    pub fn results_per_config(
        results: &BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, ChannelResult>>>,
        nb_of_conf: usize,
    ) -> Vec<Vec<&ChannelResult>> {
        let mut per_conf: Vec<Vec<&ChannelResult>> = (0..nb_of_conf).map(|_| Vec::new()).collect();
        for confs in results.values() {
            for (&conf, chans) in confs {
                per_conf[conf].extend(chans.values());
            }
        }
        per_conf
    }

    /// Pure core of [`Self::transform_results`], parameterized for the unit test.
    fn transform_results_impl(
        results: &BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, ChannelResult>>>,
        nb_of_conf: usize,
        nb_of_files: usize,
    ) -> Array2<f64> {
        // legacy: :343-344 pre-size.
        let nb_elements = 2 * nb_of_conf * nb_of_files;

        let mut counter = 0usize;
        let mut initialized = false;
        let mut results_e: Array2<f64> = Array2::zeros((0, 0));
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
                // case, instead of taking the first present entry. Note: the real legacy
                // `operator[]` would also INSERT a phantom (0,0) entry into `mmap` as a side
                // effect, which a subsequent BOOST_FOREACH would then iterate as a spurious
                // `[file+1, 1, 1]` row -- but that only matters when conf 0/chan 0 is absent
                // AND `mmap` is iterated again afterward, which is unreachable here (`mmap`
                // is read-only past this point); the port's `.get()` does not reproduce that
                // phantom insert.
                res_width = mmap
                    .get(&0)
                    .and_then(|chan_map| chan_map.get(&0))
                    .map(|r| r.to_row().len())
                    .unwrap_or(0);
                results_e = Array2::zeros((nb_elements, 3 + res_width));
                initialized = true;
            }
            // legacy: :365-382 per conf (ascending) then per chan (ascending).
            for (conf_idx, chan_map) in mmap {
                let config_nb = conf_idx + 1;
                for (chan_idx, result) in chan_map {
                    let chan_nb = chan_idx + 1;
                    // The ONE place the result row is written (`ChannelResult::to_row`).
                    let res = result.to_row();
                    // legacy: :375-377 line = [file+1, conf+1, chan+1, res...].
                    results_e[[counter, 0]] = (file_idx + 1) as f64;
                    results_e[[counter, 1]] = config_nb as f64;
                    results_e[[counter, 2]] = chan_nb as f64;
                    for (c, &v) in res.iter().enumerate() {
                        results_e[[counter, 3 + c]] = v;
                    }
                    counter += 1;
                }
            }
        }

        // legacy: :385-388 conservativeResize (truncate) to the counter.
        if !initialized {
            // No results at all -> empty ResultsE.
            return Array2::zeros((0, 0));
        }
        let _ = res_width;
        truncate_rows(&results_e, counter)
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
    /// Returns the last-processed network's [`GradCheckReport`] (algo 3/4/5 have a
    /// single NN; algo 6 iterates BOTH nets -- use [`Self::grad_check`] to observe
    /// every per-network report). Internal-only wrapper for the `run()` gradCheck
    /// mode dispatch (`:214`); delegates to the public [`Self::grad_check`] (no
    /// duplicate body).
    fn grad_check_full(&mut self, epsilon: f64) -> Result<GradCheckReport> {
        self.grad_check(epsilon, usize::MAX)
            .map(|mut v| v.pop().map(|(_, r)| r))?
            .ok_or_else(|| anyhow::anyhow!("gradCheck: no backprop-activated network in config 0"))
    }

    /// Per-network gradCheck: one `(network_index, report)` per backprop-active
    /// network of config 0, in ascending network index (`:245-246`).
    fn grad_check_capped(
        &mut self,
        epsilon: f64,
        max_weights: usize,
    ) -> Result<Vec<(usize, GradCheckReport)>> {
        // legacy: :238 snapshot the WHOLE bag.
        let proc_mem = self.processors.clone();
        // F10 (phase 5): snapshot the folded-gradient stash alongside the bag. The
        // perturbation sweep below runs `run_epoch` per +/-eps step, and each of those
        // OVERWRITES `self.seam_derivs` with the fold at the PERTURBED weights -- so
        // without restoring it, gradCheck would leave the seam returning a gradient at
        // the last -eps point while `weights(0)` reads the restored theta. Restored at
        // the epilogue so gradCheck is transparent to the seam invariant.
        let seam_mem = self.seam_derivs.clone();
        let nb = self.processors.nb_of_conf();
        // legacy: :239-242 zero the four mem matrices to 1 x nConfs.
        self.cost_mem = Array2::zeros((1, nb));
        self.bad_classif_mem = Array2::zeros((1, nb));
        self.cost_lid_mem = Array2::zeros((1, nb));
        self.bad_classif_lid_mem = Array2::zeros((1, nb));
        // legacy: :243 _TrainingEpochs = 0.
        self.training_epochs = 0;

        let mut reports: Vec<(usize, GradCheckReport)> = Vec::new();

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
                let cost_plus = self.grad_check_cost(algo0, ii);

                // legacy: :295-297 -eps: perturb -2*eps from the +eps point, restore.
                new_nn_weights[ii][kk] -= 2.0 * epsilon;
                self.processors = proc_mem.clone();
                self.processors.set_weights(0, &new_nn_weights)?;
                let mut tmp_derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
                self.run_epoch(1, inner_mode, &mut tmp_derivs, false)?;
                let cost_minus = self.grad_check_cost(algo0, ii);

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
            reports.push((ii, report));
        }

        // Restore the bag to the snapshot (the sweep left it perturbed). Port addition,
        // no legacy counterpart (legacy leaves _Processors perturbed) -- unobservable
        // since gradCheck is terminal in run().
        self.processors = proc_mem;
        // F10 (phase 5): restore the folded-gradient stash the sweep clobbered, so the
        // seam invariant holds -- `weights_derivatives(0)` matches the restored
        // `weights(0)`, not a stale gradient at the last -eps perturbation. Observable
        // through the PyO3 seam (a `grad_check` call followed by `weights_derivatives`).
        self.seam_derivs = seam_mem;

        Ok(reports)
    }

    /// The cost accumulation over `self.results` for the gradCheck central
    /// difference (`:271-292`), switching on config-0's algo AND (for algo 6)
    /// the network index `ii`: algo 3/4 -> the SAD pair `seg_cost`/`seg_count`
    /// (`:275-277`); algo 5 -> the LID pair `lid.cost`/`lid.count` (`:278-280`);
    /// algo 6 net `ii == 0` -> the SAD pair, net `ii == 1` -> the LID pair
    /// (`:281-288`). Which pair a net owns is still decided by the algo
    /// integer here; moving it onto the driver is issue #25's.
    ///
    /// The legacy `counter` is a `long` accumulating doubles (per-element
    /// truncation); every accumulated value is an exact integer count stored in
    /// a double, so an f64 accumulator is value-identical.
    fn grad_check_cost(&self, algo0: i32, ii: usize) -> f64 {
        let mut cost = 0.0;
        let mut counter = 0.0;
        for jj in 0..self.corpus.nb_of_files() {
            let Some(conf0) = self.results.get(&jj).and_then(|f| f.get(&0)) else {
                continue;
            };
            for r in conf0.values() {
                let lid_cost = r.lid_cost();
                let lid_count = r.lid_count() as f64;
                match algo0 {
                    3 | 4 => {
                        cost += r.seg_cost;
                        counter += r.seg_count as f64;
                    }
                    5 => {
                        cost += lid_cost;
                        counter += lid_count;
                    }
                    6 => {
                        if ii == 0 {
                            cost += r.seg_cost;
                            counter += r.seg_count as f64;
                        } else {
                            cost += lid_cost;
                            counter += lid_count;
                        }
                    }
                    _ => {
                        // legacy: no matching arm (:275-289) -- cost/counter stay
                        // untouched; the :292 division then yields 0/0 = NaN.
                    }
                }
            }
        }
        // legacy: :292 cost /= counter (integer counter -> float division; a 0
        // counter yields inf/NaN, mirrored here).
        cost / counter
    }

    // ==== Public seam API (the PyO3 seam surface, Phase 4c) ==================

    /// The post-`transformResults` combined result matrix (`ResultsE`,
    /// `CorpusProcessor.cpp:342-389`): rows `[file+1, conf+1, chan+1, res...]`
    /// in ascending file/conf/chan order (see [`Self::transform_results_impl`]).
    /// The PyO3 seam surface (Phase 4c).
    ///
    /// Contract: EMPTY (`0x0`) until the first completed `run()` -- populated
    /// only by [`Self::transform_results`], which fires at the end of
    /// `run_solo`/`train`/`grad_check_capped`'s internal epoch runs. Reading it
    /// before any run observes the ctor's `Array2::zeros((0, 0))` seed.
    pub fn results_matrix(&self) -> &Array2<f64> {
        &self.results_e
    }

    /// The channel results by name, one entry per (file, config, channel) in
    /// ascending (file, conf, chan) order: the typed view the result matrix is
    /// rendered from (issue #23). Same contract as [`Self::results_matrix`]:
    /// EMPTY until the first completed `run()`.
    pub fn channel_results(&self) -> Vec<(usize, usize, usize, &ChannelResult)> {
        let mut out = Vec::new();
        for (&file, confs) in &self.results {
            for (&conf, chans) in confs {
                for (&chan, result) in chans {
                    out.push((file, conf, chan, result));
                }
            }
        }
        out
    }

    /// The number of configs this run holds: the bound every `pos` on the seam is
    /// checked against at the binding (issue #60), where a caller's integer first
    /// arrives; the bag's read-shaped methods stay infallible (see
    /// `BagOfProcessors::get_weights`).
    pub fn nb_configs(&self) -> usize {
        self.processors.nb_of_conf()
    }

    /// Config-`pos`'s full weight-vector set: one flat vec per network (algo 6
    /// -> `[sad, lid]`; algo 0/1/2, no NN -> empty `Vec`). The PyO3 seam
    /// surface (Phase 4c); promoted from `get_config0_all_weights_for_test`
    /// (Task 9), generalized from config-0-only to `pos` (the bag already
    /// dispatches per-`pos` -- `BagOfProcessors::get_weights`).
    pub fn weights(&self, pos: usize) -> Vec<Vec<f64>> {
        self.processors.get_weights(pos)
    }

    /// Seed config-`pos`'s network(s): algo 6 pass `[sad, lid]`; algo 0/1/2 (no
    /// NN) is a no-op. The PyO3 seam surface (Phase 4c); promoted from
    /// `set_config0_all_weights_for_test` (Task 9), generalized from
    /// config-0-only to `pos`.
    ///
    /// F10 (phase 5): clears the folded-gradient stash -- a weights change
    /// invalidates the last fold's derivatives, so a `set_weights` NOT followed
    /// by a `run()` makes [`Self::weights_derivatives`] fall back to the bag's
    /// reset accumulator instead of returning a stale gradient at old weights.
    /// The seam's own flow (`set_weights` -> `run` -> `weights_derivatives`)
    /// repopulates the stash in the intervening `run`, so this is invisible
    /// there; it only guards the set-without-run footgun.
    pub fn set_weights(&mut self, pos: usize, nets: &[Vec<f64>]) -> Result<()> {
        self.seam_derivs.clear();
        self.processors.set_weights(pos, nets)
    }

    /// Config-`pos`'s per-network `Nx2` derivative matrices (col 0 summed
    /// deriv, col 1 count; algo 6 -> `[sad, lid]`). The PyO3 seam surface
    /// (Phase 4c).
    ///
    /// F10 (phase 5): returns the folded gradient from the last completed
    /// `run_epoch` (the [`Self::seam_derivs`] stash) when present -- the
    /// deterministic ascending-lane reduction the fold computed on the per-lane
    /// clones, identical to what `grad_check`'s local map sees. Falls back to
    /// the main bag's accumulator only when no fold has run since construction
    /// or the last `set_weights` (the bag's reset state, col0 = 0), which is the
    /// pre-F10 behavior and the reason the seam returned an all-zero gradient.
    pub fn weights_derivatives(&self, pos: usize) -> Vec<Array2<f64>> {
        if let Some(d) = self.seam_derivs.get(&pos) {
            return d.clone();
        }
        self.processors.get_weights_derivatives(pos)
    }

    /// Config-`pos`'s per-network input-normalization statistics (algo 6 ->
    /// `[sad, lid]`). The PyO3 seam surface (Phase 4c); no prior `_for_test`
    /// hook existed for this -- a new thin delegation to
    /// `BagOfProcessors::get_input_statistics`, which already dispatches
    /// per-`pos`.
    pub fn input_statistics(&self, pos: usize) -> Vec<InputStatistics> {
        self.processors.get_input_statistics(pos)
    }

    /// Corpus-level gradient check (`CorpusProcessor::gradCheck`, `:237-340`),
    /// capped at `max_weights` per network: one `(network_index, report)` per
    /// backprop-active network of config 0, ascending index. The PyO3 seam
    /// surface (Phase 4c); promoted from `grad_check_all_for_test` (Task 9)
    /// verbatim -- the bag's per-network sweep was already general, nothing to
    /// widen.
    pub fn grad_check(
        &mut self,
        epsilon: f64,
        max_weights: usize,
    ) -> Result<Vec<(usize, GradCheckReport)>> {
        self.grad_check_capped(epsilon, max_weights)
    }

    // ==== Test-only hooks (pub for the integration test) ====================

    /// Pure `transformResults` core (test hook for `transform_results_ordering`).
    #[doc(hidden)]
    pub fn transform_results_for_test(
        results: &BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, ChannelResult>>>,
        nb_of_conf: usize,
        nb_of_files: usize,
    ) -> Array2<f64> {
        Self::transform_results_impl(results, nb_of_conf, nb_of_files)
    }

    /// Corpus-level gradient check with a weight cap (test hook for
    /// `grad_check_synthetic`; Task 9's harness golden replays it). Returns the
    /// LAST backprop-active network's report (pre-Task-9 signature, kept for
    /// the 4a single-net goldens). Delegates to the public [`Self::grad_check`]
    /// (no duplicate body).
    #[doc(hidden)]
    pub fn grad_check_for_test(
        &mut self,
        epsilon: f64,
        max_weights: usize,
    ) -> Result<GradCheckReport> {
        self.grad_check(epsilon, max_weights)
            .map(|mut v| v.pop().map(|(_, r)| r))?
            .ok_or_else(|| anyhow::anyhow!("gradCheck: no backprop-activated network in config 0"))
    }

    /// Per-network gradient check (Task 9): one `(network_index, report)` per
    /// backprop-active network of config 0 -- for algo 6, index 0 is the SAD
    /// net (cost cols 4/len-1) and index 1 the LID net (cols 14/len-2).
    /// Delegates to the public [`Self::grad_check`] (no duplicate body).
    #[doc(hidden)]
    pub fn grad_check_all_for_test(
        &mut self,
        epsilon: f64,
        max_weights: usize,
    ) -> Result<Vec<(usize, GradCheckReport)>> {
        self.grad_check(epsilon, max_weights)
    }

    /// The per-epoch config-0 weight snapshots captured during a training run
    /// (test hook for `phase4a_train_golden`). `epoch_weight_trace_for_test()[e]`
    /// is config-0's flat weights AFTER the epoch-`e` `saveAndUpdate` (epoch 0 =
    /// post-solo, then one entry per inner epoch, then the final eval). Empty for
    /// a non-training run.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn epoch_weight_trace_for_test(&self) -> &[Vec<f64>] {
        &self.epoch_weight_trace
    }

    /// The SECOND net's (config-0 network index 1 -- the algo-6 LID net)
    /// per-epoch weight snapshots (Task 9). Entries are empty vecs for a
    /// single-net config 0.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn epoch_weight_trace_lid_for_test(&self) -> &[Vec<f64>] {
        &self.epoch_weight_trace_lid
    }

    /// The per-epoch best-cost (conf 0) snapshots captured during a training run
    /// (test hook for the gate fire/skip non-vacuity assert). A STRICT drop from
    /// entry `e-1` to `e` means the save gate FIRED at epoch `e`; an unchanged
    /// value means it SKIPPED.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn epoch_best_cost_trace_for_test(&self) -> &[f64] {
        &self.epoch_best_cost_trace
    }

    /// Config-0's per-epoch RAW folded derivative matrix (test hook for the
    /// Task 10 fold-order golden). See [`Self::epoch_raw_derivs_trace`]'s doc
    /// comment: unlike `epoch_weight_trace_for_test`, this is sensitive to
    /// the per-file fold order even when no Rprop sign flips.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn epoch_raw_derivs_trace_for_test(&self) -> &[Array2<f64>] {
        &self.epoch_raw_derivs_trace
    }

    /// Config-0's flat weight vector (test hook for the grad-check restore assert).
    /// Empty when config 0 is a non-NN algo (no weights). Delegates to the
    /// public [`Self::weights`] (no duplicate body).
    #[doc(hidden)]
    pub fn get_config0_weights_for_test(&self) -> Vec<f64> {
        self.weights(0).into_iter().next().unwrap_or_default()
    }

    /// Seed config-0's NN with a synthetic nonzero weight vector (test hook for
    /// `grad_check_synthetic`): the default-initialized net sits at a degenerate
    /// operating point (near-zero gradients), so the gradcheck needs deterministic
    /// nonzero weights -- exactly the phase3 `synth_flat` pattern -- to exercise a
    /// smoothly-varying cost. No-op for a non-NN config 0.
    /// Delegates to the public [`Self::set_weights`] (no duplicate body).
    #[doc(hidden)]
    pub fn set_config0_weights_for_test(&mut self, flat: &[f64]) -> Result<()> {
        self.set_weights(0, &[flat.to_vec()])
    }

    /// Seed EVERY net of config 0 (Task 9): for algo 6 pass `[sad, lid]`; the
    /// single-net variant of [`Self::set_config0_weights_for_test`]. Delegates
    /// to the public [`Self::set_weights`] (no duplicate body).
    #[doc(hidden)]
    pub fn set_config0_all_weights_for_test(&mut self, nets: &[Vec<f64>]) -> Result<()> {
        self.set_weights(0, nets)
    }

    /// Config-0's full weight-vector set (test hook): one flat vec per network
    /// (algo 6 -> `[sad, lid]`). Delegates to the public [`Self::weights`] (no
    /// duplicate body).
    #[doc(hidden)]
    pub fn get_config0_all_weights_for_test(&self) -> Vec<Vec<f64>> {
        self.weights(0)
    }

    /// The bag's captured per-conf `(errorPercLID, confusion)` pairs from the
    /// LAST `save_and_update` (Task 9; display-only in the legacy).
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn last_confusion_for_test_from_bag(&self) -> &[(f64, Array2<f64>)] {
        self.processors.last_confusion_for_test()
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
        let mut results: BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, ChannelResult>>> =
            BTreeMap::new();
        for j in 0..corpus.nb_of_files() {
            let item = corpus.item(j);
            let tmp = bag.segmentation_function(item, mode)?;
            if !tmp.is_empty() {
                results.insert(j, tmp);
            }
        }
        let results_e =
            Self::transform_results_impl(&results, bag.nb_of_conf(), corpus.nb_of_files());
        Ok(results_e)
    }

    /// Fold-order MEASUREMENT probe (Task 10 backlog item 2, no legacy
    /// counterpart -- the port's OWN determinism contract, not a legacy
    /// parity target). Sibling of [`Self::run_epoch_sequential_oracle`]: same
    /// fresh-bag sequential walk, but (a) parameterized by file order
    /// (`reverse`) and (b) returns the FOLDED per-conf derivatives instead of
    /// the transformed results matrix, using the IDENTICAL create-vs-`+=`
    /// accumulation `run_epoch` performs (`:184-191`) -- so a measured
    /// ascending-vs-descending divergence here is evidence about the SAME
    /// arithmetic `run_epoch`'s static-lane fold performs at `n == 1`
    /// (`lanes_n1_equals_sequential`), not a separate reimplementation.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn run_epoch_fold_probe_for_test(
        mut configs: Vec<IndexMap<String, String>>,
        mode: Mode,
        reverse: bool,
    ) -> Result<BTreeMap<usize, Vec<Array2<f64>>>> {
        let corpus = Corpus::from_config(&configs[0])?;
        let mut bag = BagOfProcessors::from_configs(&mut configs, mode)?;
        let nb_of_conf = bag.nb_of_conf();
        let mut order: Vec<usize> = (0..corpus.nb_of_files()).collect();
        if reverse {
            order.reverse();
        }
        let mut derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
        for j in order {
            let item = corpus.item(j);
            let tmp = bag.segmentation_function(item, mode)?;
            if tmp.is_empty() {
                continue;
            }
            for ii in 0..nb_of_conf {
                let tmp_derivs = bag.get_weights_derivatives(ii);
                match derivs.get_mut(&ii) {
                    Some(existing) if !existing.is_empty() => {
                        for (kk, mat) in tmp_derivs.iter().enumerate() {
                            existing[kk] = &existing[kk] + mat;
                        }
                    }
                    _ => {
                        derivs.insert(ii, tmp_derivs);
                    }
                }
            }
        }
        Ok(derivs)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal algo-1 (TDC) config with a single-file listing, so
    /// `Corpus::from_config` + `BagOfProcessors::from_configs` both succeed and the
    /// constructor reaches the epochs/epsilon parse. The `Neural_Networks_*` keys
    /// are overridden per test.
    fn minimal_config(dir: &std::path::Path) -> IndexMap<String, String> {
        let mapping = dir.join("mapping.csv");
        std::fs::write(&mapping, "unk;unk;0\n").unwrap();
        let listing = dir.join("listing.csv");
        std::fs::write(&listing, "dummy.wav;;unk;unk;1.0;1\n").unwrap();

        let mut m = IndexMap::new();
        m.insert("Algo_choice".to_string(), "1".to_string());
        m.insert("numOuterThreads".to_string(), "1".to_string());
        m.insert(
            "language2classmapping".to_string(),
            mapping.to_str().unwrap().to_string(),
        );
        m.insert(
            "fileslisting".to_string(),
            listing.to_str().unwrap().to_string(),
        );
        m
    }

    /// legacy `get<T>(name, default)` exits(1) on a PRESENT but unparseable value
    /// (not a silent fallback to the default): a malformed
    /// `Neural_Networks_BackPropagation_Epochs` must surface as `Err`, matching
    /// `bag_of_processors::get_i32_default`'s erroring semantics.
    #[test]
    fn malformed_epochs_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = minimal_config(dir.path());
        cfg.insert(
            "Neural_Networks_BackPropagation_Epochs".to_string(),
            "not_a_number".to_string(),
        );
        let mode = Mode {
            kind: ModeKind::Solo,
            verbose: false,
        };
        assert!(CorpusProcessor::new(vec![cfg], mode).is_err());
    }
}
