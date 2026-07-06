//! Bidirectional wrapper: forward + reverse recurrent nets + output MLP, BPTT.
//!
//! Ported from legacy C++: `BLSTMNeuralNetwork.cpp:26-153` (ctor) and `:209-253`
//! (setWeights/getWeights/getNbOfWeights). Phase 2 Task 7 ports construction + the
//! flat weight seam; forward/backward driving is a later task.
//!
//! TRAP (do not use as a source): `BLSTMNeuralNetwork.h:35-145` is a STALE
//! commented-out ctor superseded by the real one in the `.cpp`. Ported from the
//! `.cpp` only.
//!
//! Config keys read under `{prefix}_*` (`:27-122`):
//! - `LSTMNeuronNb` (list, len >= 2), `LSTMSubSampling` (list, len ==
//!   `LSTMNeuronNb.len()-1`), `OutputNeuronNb` (list, len >= 2),
//!   `OutputSubSampling` (list, len == `OutputNeuronNb.len()-1`).
//! - Non-MLP constraint (`:43-46`, only checked when `LSTMNeuronNb[0] != 0`):
//!   `OutputNeuronNb[0] == 2*LSTMNeuronNb.last()`.
//! - `_IsMLP = (LSTMNeuronNb[0] == 0)` (`:54`).
//! - `InputNormalizationType` (`short`, no default), `TwoSweeps` (`bool`, no
//!   default), `BackPropagationActivated` (`bool`, default `false`),
//!   `BackPropOutputNetworkOnly` (`bool`, default `false`),
//!   `TargetEnforcementStep` (`int`, default `0`).
//! - Peephole flags are read PER-DIRECTION, not per-layer: `NeuralNetwork`'s ctor
//!   (`NeuralNetwork.hpp:27`) passes the SAME `specialization_preposition` (e.g.
//!   `{prefix}_Forward`) to every `LSTMLayer` in that direction, and `LSTMLayer`
//!   reads `{prefix}_Forward_Is{Cells,Gates,GatesRecurrent}PeepholesActive`
//!   (default `true`, `LSTMLayer.cpp:10-13`) -- so all layers within a direction
//!   share one flag triple, read once here and threaded to every `LstmLayer::new`
//!   in that direction.
//!
//! `setWeights` order (`:209-224`): Forward net -> Backward net -> Output net ->
//! mean(inputSize) -> std(inputSize), where `inputSize = LSTMNeuronNb[0]`
//! (`_ForwardNetwork.getInputSize()`) in non-MLP mode, or
//! `_OutputNetwork.getInputSize()` in MLP mode. `getNbOfWeights` (`:227-238`) is
//! the sum of the sub-network weight counts plus `2*inputSize`.
//!
//! DEVIATION from the legacy tolerance (documented per the task brief): the
//! legacy accepts `flat.len() > nb_of_weights()` with a warning and still calls
//! `setWeights` (consuming only the head); we reproduce that exact behavior
//! (`set_weights` proceeds on a longer slice, ignoring the tail) but callers that
//! want the legacy's console warning must check `flat.len()` themselves --
//! `set_weights` does not print. `flat.len() < nb_of_weights()` is the legacy
//! `exit(1)` error path, ported as `Err` (not a process exit).

use anyhow::{Result, bail};
use indexmap::IndexMap;
use ndarray::Array2;

use crate::cost::CostLaw;
use crate::features::stats::InputStatistics;

use super::layers::NeuronLayer;
use super::network::Network;
use super::train::Rprop;

fn get_list(map: &IndexMap<String, String>, key: &str) -> Result<Vec<usize>> {
    let s = map
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("param '{key}' not found in config"))?;
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("cannot read '{part}' as usize for '{key}': {e}"))
        })
        .collect()
}

fn get_bool(map: &IndexMap<String, String>, key: &str) -> Result<bool> {
    let s = map
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("param '{key}' not found in config"))?;
    s.parse::<bool>()
        .map_err(|e| anyhow::anyhow!("cannot read '{s}' as bool for '{key}': {e}"))
}

fn get_bool_default(map: &IndexMap<String, String>, key: &str, default: bool) -> bool {
    map.get(key).map(|s| s == "true").unwrap_or(default)
}

fn get_i16(map: &IndexMap<String, String>, key: &str) -> Result<i16> {
    let s = map
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("param '{key}' not found in config"))?;
    s.parse::<i16>()
        .map_err(|e| anyhow::anyhow!("cannot read '{s}' as i16 for '{key}': {e}"))
}

fn get_i32_default(map: &IndexMap<String, String>, key: &str, default: i32) -> i32 {
    map.get(key)
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(default)
}

fn get_f64_default(map: &IndexMap<String, String>, key: &str, default: f64) -> f64 {
    map.get(key)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

/// Per-direction peephole activation flags (`{prefix}_{Forward,Backward}_Is{...}
/// PeepholesActive`, each default `true`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeepholeFlags {
    pub cells: bool,
    pub gates: bool,
    pub gates_recurrent: bool,
}

impl PeepholeFlags {
    fn from_legacy(map: &IndexMap<String, String>, dir_prefix: &str) -> PeepholeFlags {
        PeepholeFlags {
            cells: get_bool_default(map, &format!("{dir_prefix}_IsCellsPeepholesActive"), true),
            gates: get_bool_default(map, &format!("{dir_prefix}_IsGatesPeepholesActive"), true),
            gates_recurrent: get_bool_default(
                map,
                &format!("{dir_prefix}_IsGatesRecurrentPeepholesActive"),
                true,
            ),
        }
    }
}

/// Parsed BLSTM config (`BLSTMNeuralNetwork.cpp:26-122`): LSTM/output topology,
/// per-direction peephole flags, and the scalar knobs read at construction.
#[derive(Debug, Clone)]
pub struct BlstmConfig {
    pub lstm_neuron_nb: Vec<usize>,
    pub lstm_sub_sampling: Vec<usize>,
    pub output_neuron_nb: Vec<usize>,
    pub output_sub_sampling: Vec<usize>,
    pub is_mlp: bool,
    pub forward_peep: PeepholeFlags,
    pub backward_peep: PeepholeFlags,
    pub input_normalization_type: i16,
    pub two_sweeps: bool,
    pub back_propagation_activated: bool,
    /// legacy: BLSTMNeuralNetwork.cpp:103 (gradient-only flag; consumed in Phase 3)
    pub back_prop_output_network_only: bool,
    pub target_enforcement_step: i32,
    /// `_CostFunction` (`BLSTMNeuralNetwork.cpp:104`): `CostLaw(conf, prefix)`. The
    /// plain `feed_forward_backward` cost accumulation consumes it (Task 8); Phase 3
    /// backward will reuse it for deltas.
    pub cost_law: CostLaw,
    /// `_BackPropagationRpropInit` (`BLSTMNeuralNetwork.cpp:151`): the iRPROP- initial
    /// step size, default `1e-2`. The real `1_worker_1.config` has NO such key, so the
    /// default path is the live one.
    pub rprop_init: f64,
}

impl BlstmConfig {
    /// Read + validate `{prefix}_*` keys (`BLSTMNeuralNetwork.cpp:27-108`).
    pub fn from_legacy(map: &IndexMap<String, String>, prefix: &str) -> Result<BlstmConfig> {
        let k = |suffix: &str| format!("{prefix}{suffix}");

        let lstm_neuron_nb = get_list(map, &k("_LSTMNeuronNb"))?;
        if lstm_neuron_nb.len() < 2 {
            bail!("The number layers in the LSTM neural networks must be > 1.");
        }
        let lstm_sub_sampling = get_list(map, &k("_LSTMSubSampling"))?;
        if lstm_sub_sampling.len() != lstm_neuron_nb.len() - 1 {
            bail!(
                "The number of values given for the sub-sampling in the LSTM neural networks must be {}.",
                lstm_neuron_nb.len() - 1
            );
        }

        let output_neuron_nb = get_list(map, &k("_OutputNeuronNb"))?;
        if output_neuron_nb.len() < 2 {
            bail!("The number layers in the output neural network must be > 1.");
        }
        if lstm_neuron_nb[0] != 0
            && output_neuron_nb[0] != 2 * lstm_neuron_nb[lstm_neuron_nb.len() - 1]
        {
            bail!(
                "The first layer of the output network must contains twice the number of LSTM cells in last layer of the LSTM networks."
            );
        }
        let output_sub_sampling = get_list(map, &k("_OutputSubSampling"))?;
        if output_sub_sampling.len() != output_neuron_nb.len() - 1 {
            bail!(
                "The number of values given for the sub-sampling in the output neural network must be {}.",
                output_neuron_nb.len() - 1
            );
        }

        let is_mlp = lstm_neuron_nb[0] == 0;

        // Peephole flags: read regardless of is_mlp (harmless when the forward/
        // backward nets are absent -- the legacy reads them via the LSTMLayer
        // ctor, which is simply never instantiated in MLP mode).
        let forward_peep = PeepholeFlags::from_legacy(map, &k("_Forward"));
        let backward_peep = PeepholeFlags::from_legacy(map, &k("_Backward"));

        let input_normalization_type = get_i16(map, &k("_InputNormalizationType"))?;
        let two_sweeps = get_bool(map, &k("_TwoSweeps"))?;
        let back_propagation_activated =
            get_bool_default(map, &k("_BackPropagationActivated"), false);
        let back_prop_output_network_only =
            get_bool_default(map, &k("_BackPropOutputNetworkOnly"), false);
        let target_enforcement_step = get_i32_default(map, &k("_TargetEnforcementStep"), 0);
        // `_CostFunction = CostLaw(conf, prefix)` (:104). Built from the same prefix;
        // all cost-law keys have defaults, so a config lacking them is valid.
        let cost_law = CostLaw::from_config(map, prefix);
        // `_BackPropagationRpropInit` (:151), default 1e-2 (the live path -- the real
        // config has no such key).
        let rprop_init = get_f64_default(map, &k("_BackPropagationRpropInit"), 1e-2);

        Ok(BlstmConfig {
            lstm_neuron_nb,
            lstm_sub_sampling,
            output_neuron_nb,
            output_sub_sampling,
            is_mlp,
            forward_peep,
            backward_peep,
            input_normalization_type,
            two_sweeps,
            back_propagation_activated,
            back_prop_output_network_only,
            target_enforcement_step,
            cost_law,
            rprop_init,
        })
    }
}

/// `BLSTMNeuralNetwork<LSTMLayer>` (`BLSTMNeuralNetwork.cpp`): forward + backward
/// LSTM `Network`s (absent in MLP mode) feeding a `NeuronLayer` output `Network`,
/// plus the input-normalization mean/std tail and bookkeeping fields.
pub struct BlstmNetwork {
    cfg: BlstmConfig,
    forward_network: Option<Network<super::layers::LstmLayer>>,
    backward_network: Option<Network<super::layers::LstmLayer>>,
    output_network: Network<NeuronLayer>,

    normalize_input_mean: Vec<f64>,
    normalize_input_std: Vec<f64>,

    /// `_OutputForward`/`_OutputBackward` (Task 8 fills; empty for now).
    pub output_forward: ndarray::Array2<f64>,
    pub output_backward: ndarray::Array2<f64>,
    /// `_Cost`/`_NbOfClassif` (Phase 3 gradient accumulators; zeroed here).
    pub cost: f64,
    pub nb_of_classif: i64,
    /// `_TruncatesSequence`/`_Overlaps`/`_MakesTwoSweeps`-derived processing flags
    /// (Task 9's `set_processing_type` fills these beyond the trivial default).
    truncates_sequence: bool,
    overlaps: bool,

    pub input_statistics: InputStatistics,

    /// `_Trainer` (`BLSTMNeuralNetwork.cpp:152`): the iRPROP- optimizer, a LONG-LIVED
    /// per-network member (state persists across `update_weights` calls for the
    /// network's lifetime), constructed from `_BackPropagationRpropInit` (default 1e-2).
    trainer: Rprop,
}

impl BlstmNetwork {
    /// `BLSTMNeuralNetwork(conf, prefix, weightsSetExternally=true)` (`:86-98`
    /// structural half; weight loading from a `_weightsFile` key is the caller's
    /// job via `set_weights`, matching how this port is driven from real `.bin`
    /// fixtures rather than the legacy `.mat`).
    pub fn from_config(cfg: BlstmConfig) -> Result<BlstmNetwork> {
        let (forward_network, backward_network) = if cfg.is_mlp {
            (None, None)
        } else {
            let fwd_peep = cfg.forward_peep;
            let bwd_peep = cfg.backward_peep;
            let forward = Network::new(
                cfg.lstm_neuron_nb.clone(),
                cfg.lstm_sub_sampling.clone(),
                |_layer_id, input, output| {
                    super::layers::LstmLayer::new(
                        input,
                        output,
                        fwd_peep.cells,
                        fwd_peep.gates,
                        fwd_peep.gates_recurrent,
                    )
                },
            );
            let backward = Network::new(
                cfg.lstm_neuron_nb.clone(),
                cfg.lstm_sub_sampling.clone(),
                |_layer_id, input, output| {
                    super::layers::LstmLayer::new(
                        input,
                        output,
                        bwd_peep.cells,
                        bwd_peep.gates,
                        bwd_peep.gates_recurrent,
                    )
                },
            );
            (Some(forward), Some(backward))
        };

        let output_network = Network::new(
            cfg.output_neuron_nb.clone(),
            cfg.output_sub_sampling.clone(),
            |_layer_id, input, output| NeuronLayer::new(input, output),
        );

        let input_size = if cfg.is_mlp {
            output_network.input_size()
        } else {
            forward_network.as_ref().unwrap().input_size()
        };

        let trainer = Rprop::new(cfg.rprop_init);

        Ok(BlstmNetwork {
            cfg,
            forward_network,
            backward_network,
            output_network,
            normalize_input_mean: vec![0.0; input_size],
            normalize_input_std: vec![1.0; input_size],
            output_forward: ndarray::Array2::zeros((0, 0)),
            output_backward: ndarray::Array2::zeros((0, 0)),
            cost: 0.0,
            nb_of_classif: 0,
            truncates_sequence: false,
            overlaps: false,
            input_statistics: InputStatistics::new(),
            trainer,
        })
    }

    /// `getInputSize` (`:177-183`): `_OutputNetwork.getInputSize()` in MLP mode,
    /// else `_ForwardNetwork.getInputSize()`.
    pub fn input_size(&self) -> usize {
        if self.cfg.is_mlp {
            self.output_network.input_size()
        } else {
            self.forward_network.as_ref().unwrap().input_size()
        }
    }

    /// `getOutputSize` (`:185-187`): `_OutputNetwork.getOutputSize()`.
    pub fn output_size(&self) -> usize {
        self.output_network.output_size()
    }

    /// `getSubSamplingRatio` (`:189-195`): output-only ratio in MLP mode, else
    /// forward-ratio * output-ratio.
    pub fn sub_sampling_ratio(&self) -> usize {
        if self.cfg.is_mlp {
            self.output_network.sub_sampling_ratio()
        } else {
            self.forward_network.as_ref().unwrap().sub_sampling_ratio()
                * self.output_network.sub_sampling_ratio()
        }
    }

    /// `getLSTMSubSampling` (`:197-203`): `[1]` in MLP mode (no LSTM net exists),
    /// else the forward net's per-layer sub-sampling factors.
    pub fn lstm_sub_sampling(&self) -> Vec<usize> {
        if self.cfg.is_mlp {
            vec![1]
        } else {
            self.forward_network
                .as_ref()
                .unwrap()
                .sub_samplings()
                .to_vec()
        }
    }

    /// `getOutputSubSampling` (`:205-207`): the output net's per-layer sub-sampling
    /// factors.
    pub fn output_sub_sampling(&self) -> Vec<usize> {
        self.output_network.sub_samplings().to_vec()
    }

    pub fn is_mlp(&self) -> bool {
        self.cfg.is_mlp
    }

    pub fn config(&self) -> &BlstmConfig {
        &self.cfg
    }

    /// `getNbOfWeights` (`:227-238`): sum of sub-network weight counts plus
    /// `2*inputSize`.
    pub fn nb_of_weights(&self) -> usize {
        let input_size = self.input_size();
        let sub_nets = if self.cfg.is_mlp {
            self.output_network.nb_of_weights()
        } else {
            self.forward_network.as_ref().unwrap().nb_of_weights()
                + self.backward_network.as_ref().unwrap().nb_of_weights()
                + self.output_network.nb_of_weights()
        };
        sub_nets + 2 * input_size
    }

    /// `setWeights` (`:209-224`): Forward -> Backward -> Output -> mean -> std, in
    /// that order, MLP mode skipping the Forward/Backward step. Legacy tolerance:
    /// `flat.len() < nb_of_weights()` is an error (`exit(1)` there, `Err` here);
    /// `flat.len() > nb_of_weights()` is accepted (warning-only there, silently
    /// accepted here) and only the head is consumed.
    pub fn set_weights(&mut self, flat: &[f64]) -> Result<()> {
        let needed = self.nb_of_weights();
        if flat.len() < needed {
            bail!(
                "The number of gains given is less than what's needed ({} < {needed}).",
                flat.len()
            );
        }

        let mut rest = flat;
        if !self.cfg.is_mlp {
            rest = self.forward_network.as_mut().unwrap().set_weights(rest);
            rest = self.backward_network.as_mut().unwrap().set_weights(rest);
        }
        rest = self.output_network.set_weights(rest);

        let input_size = self.input_size();
        self.normalize_input_mean = rest[..input_size].to_vec();
        self.normalize_input_std = rest[input_size..2 * input_size].to_vec();

        Ok(())
    }

    /// `getWeights` (`:240-253`): mirror of `set_weights`, same block order.
    pub fn get_weights(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.nb_of_weights());
        if !self.cfg.is_mlp {
            self.forward_network.as_ref().unwrap().get_weights(&mut out);
            self.backward_network
                .as_ref()
                .unwrap()
                .get_weights(&mut out);
        }
        self.output_network.get_weights(&mut out);
        out.extend_from_slice(&self.normalize_input_mean);
        out.extend_from_slice(&self.normalize_input_std);
        out
    }

    /// `resetWeightsDerivatives` (`:278-287`): reset the sub-networks' per-layer
    /// gradient accumulators (fwd/bwd LSTM nets skipped in MLP mode) + the
    /// `InputStatistics` field. `_Cost`/`_NbOfClassif`/`_OutputForward`/
    /// `_OutputBackward` are deliberately untouched, matching the legacy.
    pub fn reset_weights_derivatives(&mut self) {
        if self.cfg.is_mlp {
            self.output_network.reset_weights_derivatives();
        } else {
            self.forward_network
                .as_mut()
                .unwrap()
                .reset_weights_derivatives();
            self.backward_network
                .as_mut()
                .unwrap()
                .reset_weights_derivatives();
            self.output_network.reset_weights_derivatives();
        }
        self.input_statistics = InputStatistics::new();
    }

    /// `getWeightsDerivatives` (`BLSTMNeuralNetwork.cpp:255-276`): the Nx2 flat gradient
    /// object -- col0 the SUMMED derivative, col1 the `_NbOfSeqFedBackward` frame count
    /// replicated per element (spec S3). Empty when `_BackPropagationActivated` is off
    /// (`:273-274`). Layout: MLP mode -> `output | [Zero|Ones|Zero|Ones]` stats tail;
    /// non-MLP -> `fwd | bwd | output | [Zero|Ones|Zero|Ones]`, with the fwd/bwd LSTM
    /// col0 multiplied by `_OutputNetwork.getSubSamplingRatio()` when that ratio > 1
    /// (`:266-269`). The 4-block tail is `2*inputSize` rows each `[0.0, 1.0]` (mean rows
    /// then std rows; stats never move, count 1). The col0 ordering matches the P0a flat
    /// weight packer element-for-element.
    pub fn get_weights_derivatives(&self) -> Array2<f64> {
        if !self.cfg.back_propagation_activated {
            return Array2::<f64>::zeros((0, 0)); // :273-274
        }
        let input_size = self.input_size();
        let out_ratio = self.output_network.sub_sampling_ratio();

        let mut rows: Vec<[f64; 2]> = Vec::with_capacity(self.nb_of_weights());
        if !self.cfg.is_mlp {
            // fwd | bwd, col0 scaled by the output-net ratio when > 1 (:266-269).
            let start_fwd = rows.len();
            self.forward_network
                .as_ref()
                .unwrap()
                .get_weights_derivatives(&mut rows);
            self.backward_network
                .as_ref()
                .unwrap()
                .get_weights_derivatives(&mut rows);
            if out_ratio > 1 {
                for r in rows[start_fwd..].iter_mut() {
                    r[0] *= out_ratio as f64;
                }
            }
        }
        self.output_network.get_weights_derivatives(&mut rows);

        // Stats tail: 2*inputSize rows, each [deriv 0 | count 1] (mean then std;
        // `Zero(mean) | Ones | Zero(std) | Ones` in the legacy `all <<`).
        for _ in 0..2 * input_size {
            rows.push([0.0, 1.0]);
        }

        let n = rows.len();
        let mut all = Array2::<f64>::zeros((n, 2));
        for (k, r) in rows.iter().enumerate() {
            all[[k, 0]] = r[0];
            all[[k, 1]] = r[1];
        }
        all
    }

    /// `updateWeights` (`BLSTMNeuralNetwork.cpp:303-310`): normalize the Nx2 gradient
    /// object ELEMENT-WISE (`col0 cwiseQuotient col1`, never a scalar `/nframes` --
    /// risk R1), hand it to the long-lived iRPROP- trainer, and write the updated flat
    /// vector back via `set_weights`. Empty gradient (`size() == 0`, backprop off) is a
    /// no-op (`:304`). The element-wise quotient of a count-0 region yields `0/0 = NaN`
    /// per IEEE (the legacy `cwiseQuotient` does the same); the stats tail's count-1
    /// slots never divide-by-zero, and the trained regions carry their accumulated
    /// frame counts, so under a real training call every count is > 0.
    pub fn update_weights(&mut self, weights_derivatives: &Array2<f64>, cost: f64) {
        if weights_derivatives.is_empty() {
            return; // :304 (size() > 0 guard)
        }
        let mut weights = self.get_weights();
        let n = weights_derivatives.nrows();
        let mut norm = vec![0.0_f64; n];
        for j in 0..n {
            norm[j] = weights_derivatives[[j, 0]] / weights_derivatives[[j, 1]]; // :306
        }
        self.trainer.update_weights(&norm, &mut weights, cost); // :307
        // `set_weights` returns Err only when the vector is too short; `get_weights`
        // produced exactly `nb_of_weights()` elements, so this cannot fail.
        self.set_weights(&weights).unwrap(); // :308
    }

    /// Trivial setter for the processing-type flags (`_TruncatesSequence`/
    /// `_Overlaps`); full derivation lands in a later task.
    pub fn set_processing_type(&mut self, truncates_sequence: bool, overlaps: bool) {
        self.truncates_sequence = truncates_sequence;
        self.overlaps = overlaps;
    }

    pub fn truncates_sequence(&self) -> bool {
        self.truncates_sequence
    }

    pub fn overlaps(&self) -> bool {
        self.overlaps
    }

    pub fn normalize_input_mean(&self) -> &[f64] {
        &self.normalize_input_mean
    }

    pub fn normalize_input_std(&self) -> &[f64] {
        &self.normalize_input_std
    }

    /// `analyseInputSeq` (`BLSTMNeuralNetwork.cpp:385-417`): fold the input's per-dim
    /// statistics into `_InputStatistics`. `rows > 0` guard; crops `leftCols(input
    /// size)` when the sequence is STRICTLY wider than the net input; first call
    /// constructs from the batch, later calls `update`. Called only from the type-1
    /// normalization branch.
    fn analyse_input_seq(&mut self, input: &Array2<f64>) {
        if input.nrows() == 0 {
            return;
        }
        let net_in = self.input_size();
        let batch = if net_in < input.ncols() {
            InputStatistics::from_matrix(&input.slice(ndarray::s![.., ..net_in]).to_owned())
        } else {
            InputStatistics::from_matrix(input)
        };
        if self.input_statistics.n == 0 {
            self.input_statistics = batch;
        } else {
            self.input_statistics.update(&batch);
        }
    }

    /// Type -1 / -2 self-normalization body (`BLSTMNeuralNetwork.cpp:737-744` /
    /// `:778-781`): mean = colSum/rows; center; std = `sqrt((colSum(centered^2) +
    /// 1e-32)/rows)` (the 1e-32 is added to the SUM); then per element `asinh(x/std)`.
    /// Whole-matrix form (matches the replicate/-2 shape); for type -1 the caller
    /// passes its own matrix (mutated in place), for type -2 a copy.
    fn self_normalize(m: &mut Array2<f64>) {
        let (r, c) = m.dim();
        if r == 0 {
            return;
        }
        let rf = r as f64;
        let mut mean = vec![0.0_f64; c];
        for col in 0..c {
            let mut acc = 0.0;
            for row in 0..r {
                acc += m[[row, col]];
            }
            mean[col] = acc / rf;
        }
        for row in 0..r {
            for col in 0..c {
                m[[row, col]] -= mean[col];
            }
        }
        let mut stdv = vec![0.0_f64; c];
        for col in 0..c {
            let mut acc = 0.0;
            for row in 0..r {
                acc += m[[row, col]] * m[[row, col]];
            }
            stdv[col] = ((acc + 1e-32) / rf).sqrt();
        }
        for row in 0..r {
            for col in 0..c {
                m[[row, col]] = (m[[row, col]] / stdv[col]).asinh();
            }
        }
    }

    /// `feedForward` core (`BLSTMNeuralNetwork.cpp:419-437`): output length = input
    /// rows divided SEQUENTIALLY by each forward-net LSTM sub-sampling ratio; forward
    /// net `feed_forward` + backward net `feed_forward_reverse` into
    /// `(outputLength x lstmOut)`; the `LSTMRatios[0] > 1 && net input < input cols`
    /// gate crops `leftCols(input size)`; the output net's `feed_forward_double`
    /// (forward LEFT) writes `output`. Fills `output_forward`/`output_backward`.
    /// Panics in MLP mode (the legacy dispatches MLP through `feed_forward_mlp`).
    pub fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>) {
        let (forward, backward) = match (
            self.forward_network.as_mut(),
            self.backward_network.as_mut(),
        ) {
            (Some(f), Some(b)) => (f, b),
            _ => panic!("feed_forward is BLSTM-only; MLP mode uses feed_forward_mlp"),
        };
        let ratios = forward.sub_samplings().to_vec();
        let mut output_length = input.nrows();
        for &r in &ratios {
            output_length /= r;
        }
        let fwd_in = forward.input_size();
        let mut out_forward = Array2::zeros((output_length, forward.output_size()));
        let mut out_backward = Array2::zeros((output_length, backward.output_size()));

        if ratios[0] > 1 && fwd_in < input.ncols() {
            let cropped = input.slice(ndarray::s![.., ..fwd_in]).to_owned();
            forward.feed_forward(&cropped, &mut out_forward);
            backward.feed_forward_reverse(&cropped, &mut out_backward);
        } else {
            forward.feed_forward(input, &mut out_forward);
            backward.feed_forward_reverse(input, &mut out_backward);
        }

        self.output_network
            .feed_forward_double(&out_forward, &out_backward, output);
        self.output_forward = out_forward;
        self.output_backward = out_backward;
    }

    /// `feedForwardMLP` (`BLSTMNeuralNetwork.cpp:462-469`): the `MLPRatios[0] > 1 &&
    /// net input < input cols` gate crops `leftCols(input size)`, then the output net
    /// forwards directly (no bidirectional halves).
    pub fn feed_forward_mlp(&mut self, input: &Array2<f64>, output: &mut Array2<f64>) {
        let ratios = self.output_network.sub_samplings().to_vec();
        let mlp_in = self.output_network.input_size();
        if ratios[0] > 1 && mlp_in < input.ncols() {
            let cropped = input.slice(ndarray::s![.., ..mlp_in]).to_owned();
            self.output_network.feed_forward(&cropped, output);
        } else {
            self.output_network.feed_forward(input, output);
        }
    }

    /// `feedBackward` (`BLSTMNeuralNetwork.cpp:439-460`): the BPTT fan-out. Seed
    /// `deltas = Zero(target.rows, target.cols)`, fill via `_CostFunction.computeDeltas`
    /// (scalar VAD path when `target.ncols() == 1` -- `compute_unitary_delta` per row,
    /// mirroring the `compute_cost` dispatch; else the multiclass softmax+CE fusion),
    /// then `output_network.feed_backward_double(output_forward, output_backward,
    /// output, deltas)` returns the split-ready `(rows x 2*lstm_out)` deltas. Unless
    /// `back_prop_output_network_only` (`:450`), split left/right halves and drive the
    /// forward LSTM net `feed_backward` + the backward LSTM net `feed_backward_reverse`,
    /// with the SAME `LSTMRatios[0] > 1 && input_size < input.cols()` crop gate the
    /// forward uses (`:452-454`). MLP-mode nets never reach here (their backward is
    /// `feed_backward_mlp`), so `forward_network`/`backward_network` are present.
    ///
    /// The commented `targetSeq.cols() > 1 -> deltas = output - target` branch (`:441`)
    /// is DEAD in the legacy: the live path always allocates `Zero(target.rows,
    /// target.cols)` and lets `computeDeltas` overwrite it.
    fn feed_backward(&mut self, input: &Array2<f64>, output: &Array2<f64>, target: &Array2<f64>) {
        // Seed deltas via the cost law. Scalar VAD path when the TARGET is one column
        // (CostLaw.cpp:348), else the multiclass fusion (mirrors compute_cost dispatch).
        let mut deltas = Array2::<f64>::zeros((target.nrows(), target.ncols()));
        if target.ncols() == 1 {
            for row in 0..target.nrows() {
                deltas[[row, 0]] = self
                    .cfg
                    .cost_law
                    .compute_unitary_delta(output[[row, 0]], target[[row, 0]]);
            }
        } else {
            let out_flat: Vec<f64> = output.iter().copied().collect();
            let tgt_flat: Vec<f64> = target.iter().copied().collect();
            let n_classes = output.ncols();
            let mut d_flat = vec![0.0_f64; out_flat.len()];
            self.cfg
                .cost_law
                .compute_deltas(&out_flat, &tgt_flat, n_classes, &mut d_flat);
            for (k, v) in d_flat.into_iter().enumerate() {
                deltas[[k / n_classes, k % n_classes]] = v;
            }
        }

        // Output-net backward over the hcat(fwd|bwd); returns (rows x 2*lstm_out).
        let output_forward = std::mem::replace(&mut self.output_forward, Array2::zeros((0, 0)));
        let output_backward = std::mem::replace(&mut self.output_backward, Array2::zeros((0, 0)));
        let split = self.output_network.feed_backward_double(
            &output_forward,
            &output_backward,
            output,
            &deltas,
        );
        self.output_forward = output_forward;
        self.output_backward = output_backward;

        if self.cfg.back_prop_output_network_only {
            return; // :450 short-circuit -- LSTM stacks untouched
        }

        // :451-458 split left/right halves and drive the LSTM stacks. The crop gate
        // mirrors the forward's leftCols(inputSize) (:452-454).
        let half = split.ncols() / 2;
        let left = split.slice(ndarray::s![.., ..half]).to_owned();
        let right = split.slice(ndarray::s![.., half..]).to_owned();

        let forward = self.forward_network.as_ref().unwrap();
        let fwd_in = forward.input_size();
        let ratios0 = forward.sub_samplings()[0];
        let output_forward = std::mem::replace(&mut self.output_forward, Array2::zeros((0, 0)));
        let output_backward = std::mem::replace(&mut self.output_backward, Array2::zeros((0, 0)));
        if ratios0 > 1 && fwd_in < input.ncols() {
            let cropped = input.slice(ndarray::s![.., ..fwd_in]).to_owned();
            self.forward_network
                .as_mut()
                .unwrap()
                .feed_backward(&cropped, &output_forward, &left);
            self.backward_network
                .as_mut()
                .unwrap()
                .feed_backward_reverse(&cropped, &output_backward, &right);
        } else {
            self.forward_network
                .as_mut()
                .unwrap()
                .feed_backward(input, &output_forward, &left);
            self.backward_network
                .as_mut()
                .unwrap()
                .feed_backward_reverse(input, &output_backward, &right);
        }
        self.output_forward = output_forward;
        self.output_backward = output_backward;
    }

    /// `feedBackwardMLP` (`BLSTMNeuralNetwork.cpp:471-486`): the MLP-mode backward. Seed
    /// deltas via `computeDeltas` (the commented `cols > 1 -> output - target` at `:474`
    /// is dead -- the Zero-seed path always runs), then `output_network.feed_backward`
    /// directly (no bidirectional split), with the `MLPRatios[0] > 1 && input_size <
    /// input.cols()` crop gate (`:480-484`). `output` is the layer's stored forward
    /// output; the dense backward reads the raw input from `layers_output` internally.
    fn feed_backward_mlp(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        target: &Array2<f64>,
    ) {
        let mut deltas = Array2::<f64>::zeros((target.nrows(), target.ncols()));
        if target.ncols() == 1 {
            for row in 0..target.nrows() {
                deltas[[row, 0]] = self
                    .cfg
                    .cost_law
                    .compute_unitary_delta(output[[row, 0]], target[[row, 0]]);
            }
        } else {
            let out_flat: Vec<f64> = output.iter().copied().collect();
            let tgt_flat: Vec<f64> = target.iter().copied().collect();
            let n_classes = output.ncols();
            let mut d_flat = vec![0.0_f64; out_flat.len()];
            self.cfg
                .cost_law
                .compute_deltas(&out_flat, &tgt_flat, n_classes, &mut d_flat);
            for (k, v) in d_flat.into_iter().enumerate() {
                deltas[[k / n_classes, k % n_classes]] = v;
            }
        }

        let mlp_in = self.output_network.input_size();
        let ratios0 = self.output_network.sub_samplings()[0];
        if ratios0 > 1 && mlp_in < input.ncols() {
            let cropped = input.slice(ndarray::s![.., ..mlp_in]).to_owned();
            self.output_network.feed_backward(&cropped, output, &deltas);
        } else {
            self.output_network.feed_backward(input, output, &deltas);
        }
    }

    /// Windowed `feedForwardBackward` entry (`BLSTMNeuralNetwork.cpp:711-773`): zero
    /// the cost accumulators, apply the in-place normalization for types 1 / -1
    /// (`:715-751`), then dispatch. THIS TASK ports the dispatch skeleton + the PLAIN
    /// paths; the windowed drivers (Truncate / TwoSweeps / OverLap / MLPOverLap) are
    /// Task 9 stubs behind the dispatch.
    ///
    /// `target` empty (0 rows) means "no targets". `input` is mutated in place for
    /// normalization types 1 and -1 (matching the legacy `Eigen::Ref` mutation).
    pub fn feed_forward_backward(
        &mut self,
        input: &mut Array2<f64>,
        window_size: usize,
        window_shift: usize,
        output: &mut Array2<f64>,
        target: &Array2<f64>,
    ) {
        self.cost = 0.0;
        self.nb_of_classif = 0;

        match self.cfg.input_normalization_type {
            1 => {
                // External normalization (:720-723): per column jj < min(cols,
                // mean.size()), (x - mean_jj)/max(1e-12, std_jj). Cols beyond the
                // mean/std tail are UNTOUCHED. Then analyseInputSeq (:728).
                let (r, cols) = input.dim();
                let max_col = cols.min(self.normalize_input_mean.len());
                for jj in 0..max_col {
                    let denom = 1e-12_f64.max(self.normalize_input_std[jj]);
                    let mean = self.normalize_input_mean[jj];
                    for row in 0..r {
                        input[[row, jj]] = (input[[row, jj]] - mean) / denom;
                    }
                }
                let snapshot = input.clone();
                self.analyse_input_seq(&snapshot);
            }
            -1 => {
                // Whole-sequence self-normalization in place (:737-744).
                Self::self_normalize(input);
            }
            _ => {} // -2 handled in the plain FFB; 0/other -> nothing here.
        }

        if self.cfg.is_mlp {
            if self.truncates_sequence && self.overlaps {
                self.feed_forward_backward_mlp_overlap(
                    input,
                    window_size,
                    window_shift,
                    output,
                    target,
                );
            } else {
                self.feed_forward_backward_mlp(input, output, target);
            }
        } else if self.truncates_sequence {
            if self.overlaps {
                self.feed_forward_backward_overlap(
                    input,
                    window_size,
                    window_shift,
                    output,
                    target,
                );
            } else {
                self.feed_forward_backward_truncate(input, window_size, output, target);
            }
        } else {
            self.feed_forward_backward_plain(input, output, target);
        }
    }

    /// `isCostModified` (`BLSTMNeuralNetwork.cpp:381-383` -> `CostLaw::isCostModified`
    /// `CostLaw.cpp:110-112`, i.e. `_BackPropWER`). Consumed by the scoring
    /// `feedForward` soft-target derivation (`:871`).
    pub fn is_cost_modified(&self) -> bool {
        self.cfg.cost_law.is_cost_modified()
    }

    /// Scoring `feedForward` (`BLSTMNeuralNetwork.cpp:843-929`): builds a per-frame
    /// target sequence from `(target_index, target_modifier)`, runs the windowed
    /// `feed_forward_backward` (input mutated in place by normalization types 1/-1),
    /// and for a BINARY net with targets expands the single posterior column into
    /// `[1-p, p]`. Returns the `outputSeqShort` matrix.
    ///
    /// - `rows < sub_sampling_ratio()` -> `Zero(1, output_size())` (`:845-846`).
    /// - output `length` = rows divided SEQUENTIALLY by each LSTM sub-sampling ratio
    ///   then each output-net ratio, ONLY when `sub_sampling_ratio() > 1` (`:854-863`).
    /// - Target construction when `target_index >= 0` (`:868-918`):
    ///   `target = is_cost_modified() ? 0.1*target_modifier : 0.0`. Multiclass
    ///   (`output_size > 1`): `target_index >= output_size -> 0` (unknown class); a
    ///   counter, starting at 0, enforces a row when `counter >= target_enforcement_step`
    ///   (then resets to 0): the enforced row is `Constant(target)` with column
    ///   `target_index` set to `1-target`; a non-enforced row is `Constant(-0.5)` and
    ///   `++counter`. The counter starts at 0, so row 0 is enforced ONLY when
    ///   `step == 0` (`0 >= 0`); for `step >= 1` rows `0..step-1` are `-0.5` and row
    ///   `step` is the first enforced row (then every `step`-th row thereafter); for
    ///   a negative `step` every row enforces. Binary (`output_size == 1`): enforced
    ///   rows get `(target_index == 1) ? 1-target : target`; non-enforced `-0.5`.
    /// - Binary expansion (`:922-926`): when `target_index >= 0 && output_size == 1`,
    ///   resize to `(length, 2)`, col1 = col0, col0 = `1 - col0` -> returns `[1-p, p]`.
    ///   Without targets the raw single column returns.
    pub fn feed_forward_scoring(
        &mut self,
        input: &mut Array2<f64>,
        window_size: usize,
        window_shift: usize,
        target_index: i64,
        target_modifier: f64,
    ) -> Array2<f64> {
        let output_size = self.output_size();
        let rows = input.nrows();
        if rows < self.sub_sampling_ratio() {
            return Array2::<f64>::zeros((1, output_size)); // :845-846
        }

        // :854-863 output length. Only re-divided when the whole-BLSTM ratio > 1.
        let mut length = rows;
        if self.sub_sampling_ratio() > 1 {
            for &r in &self.lstm_sub_sampling() {
                length /= r;
            }
            for &r in &self.output_sub_sampling() {
                length /= r;
            }
        }

        let mut output = Array2::<f64>::zeros((length, output_size));
        let mut target = Array2::<f64>::zeros((0, 0));

        if target_index >= 0 {
            let mut counter: i32 = 0;
            let step = self.cfg.target_enforcement_step;
            let target_val = if self.is_cost_modified() {
                0.1 * target_modifier
            } else {
                0.0
            };
            if output_size > 1 {
                // :872-886 multiclass. Unknown-class fold happens on a LOCAL copy of
                // target_index (the legacy mutates its int arg; we must not touch the
                // caller's binary-expansion decision below, but for output_size > 1
                // that branch never runs, so a local is faithful either way).
                let ti = if target_index >= output_size as i64 {
                    0
                } else {
                    target_index as usize
                };
                target = Array2::<f64>::zeros((length, output_size));
                for ii in 0..length {
                    if counter >= step {
                        counter = 0;
                        for c in 0..output_size {
                            target[[ii, c]] = target_val;
                        }
                        target[[ii, ti]] = 1.0 - target_val;
                    } else {
                        counter += 1;
                        for c in 0..output_size {
                            target[[ii, c]] = -0.5;
                        }
                    }
                }
            } else {
                // :887-903 binary.
                target = Array2::<f64>::zeros((length, 1));
                for ii in 0..length {
                    if counter >= step {
                        counter = 0;
                        target[[ii, 0]] = if target_index == 1 {
                            1.0 - target_val
                        } else {
                            target_val
                        };
                    } else {
                        counter += 1;
                        target[[ii, 0]] = -0.5;
                    }
                }
            }
        }

        // :920 windowed FFB (input MUTABLE: normalization mutates the caller's matrix).
        self.feed_forward_backward(input, window_size, window_shift, &mut output, &target);

        // :922-926 binary expansion into [1-p, p].
        if target_index >= 0 && output_size == 1 {
            let mut expanded = Array2::<f64>::zeros((length, 2));
            for ii in 0..length {
                let p = output[[ii, 0]];
                expanded[[ii, 0]] = 1.0 - p;
                expanded[[ii, 1]] = p;
            }
            expanded
        } else {
            output
        }
    }

    /// Plain (non-windowed) `feedForwardBackward` (`BLSTMNeuralNetwork.cpp:776-830`):
    /// type -2 normalizes into a COPY (input untouched) then forwards the copy;
    /// otherwise forwards the input directly. When `back_propagation_activated` and
    /// targets are present, the BACKWARD runs (:788-799 / :802-813): a `_Target
    /// EnforcementStep < 0` interior-row rewrite of a NEW target copy to -0.5 (the
    /// OUTPUT is NOT touched here) BEFORE `feed_backward`. THEN, separately, the cost
    /// block (:815-828, risk R7): when `_TargetEnforcementStep < 0`, the interior rows
    /// `[1, rows-1)` of BOTH a fresh target copy AND the caller-visible `output` are
    /// overwritten with `-0.5` before `computeCost`; else cost against the given
    /// targets. The two enforcement rewrites are SEPARATE: the backward one seeds
    /// deltas from the raw `output`, the cost one mutates `output`. `_NbOfClassif +=
    /// output.rows()`.
    fn feed_forward_backward_plain(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        target: &Array2<f64>,
    ) {
        // The input the backward reads: for type -2, the normalized copy (:787,795);
        // otherwise the raw input (:801,809). Type -2 always needs an owned copy to
        // forward (the normalization mutates it), so it's materialized regardless of
        // backprop; the plain-input case only clones when the backward will actually
        // run, sparing every pure-forward scoring call a T x D copy.
        let normed = if self.cfg.input_normalization_type == -2 {
            let mut norm = input.clone();
            Self::self_normalize(&mut norm);
            self.feed_forward(&norm, output);
            Some(norm)
        } else {
            self.feed_forward(input, output);
            None
        };

        // BACKWARD (:788-799 / :802-813). Runs BEFORE the cost block, so it seeds deltas
        // from the UNMUTATED output. Its enforcement rewrite touches only the TARGET.
        if self.cfg.back_propagation_activated && target.nrows() > 0 {
            let bwd_input = normed.as_ref().unwrap_or(input);
            if self.cfg.target_enforcement_step < 0 {
                let mut new_target = target.clone();
                let n_lines = target.nrows() as isize - 2;
                if n_lines > 0 {
                    for row in 1..target.nrows() - 1 {
                        for col in 0..new_target.ncols() {
                            new_target[[row, col]] = -0.5;
                        }
                    }
                }
                let out_snapshot = output.clone();
                self.feed_backward(bwd_input, &out_snapshot, &new_target);
            } else {
                let out_snapshot = output.clone();
                self.feed_backward(bwd_input, &out_snapshot, target);
            }
        }

        // COST (:815-828). SEPARATE enforcement rewrite of target AND output.
        if target.nrows() > 0 {
            if self.cfg.target_enforcement_step < 0 {
                let mut new_target = target.clone();
                let n_lines = target.nrows() as isize - 2;
                if n_lines > 0 {
                    // Interior rows [1, rows-1) of the targets AND the caller-visible
                    // output overwritten with -0.5 (:820-821), BEFORE computeCost.
                    // :820 bounds the target block by `targetSeq.rows()-2`, :821 bounds
                    // the output block by `outputSeq.rows()-2` -- two separate
                    // expressions in the legacy. This loop shares one `end` bound for
                    // both writes, which is only equivalent when `output.nrows() ==
                    // target.nrows()` -- true on every live call site (scoring always
                    // sizes `output`/`target` identically), but not a general identity.
                    let end = target.nrows() - 1;
                    for row in 1..end {
                        for col in 0..new_target.ncols() {
                            new_target[[row, col]] = -0.5;
                        }
                        for col in 0..output.ncols() {
                            output[[row, col]] = -0.5;
                        }
                    }
                }
                self.cost += self.compute_cost(output, &new_target);
                self.nb_of_classif += output.nrows() as i64;
            } else {
                self.cost += self.compute_cost(output, target);
                self.nb_of_classif += output.nrows() as i64;
            }
        }
    }

    /// Plain `feedForwardBackwardMLP` (`BLSTMNeuralNetwork.cpp:832-841`): forward, the
    /// backward (`feed_backward_mlp`) when backprop is active + targets present, then
    /// cost -- no `_TargetEnforcementStep` interior overwrite branch.
    fn feed_forward_backward_mlp(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        target: &Array2<f64>,
    ) {
        self.feed_forward_mlp(input, output);
        if self.cfg.back_propagation_activated && target.nrows() > 0 {
            let out_snapshot = output.clone();
            self.feed_backward_mlp(input, &out_snapshot, target);
        }
        if target.nrows() > 0 {
            self.cost += self.compute_cost(output, target);
            self.nb_of_classif += output.nrows() as i64;
        }
    }

    /// `_CostFunction.computeCost` dispatch (`CostLaw.cpp:191-249`): a single-column
    /// TARGET uses the scalar VAD `compute_unitary_cost` summed over rows; otherwise
    /// the softmax cross-entropy path (which iterates the OUTPUT's columns).
    /// Row-major matrices.
    fn compute_cost(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        // Branch operand is the TARGET's column count (`targetSeq.cols() == 1`, CostLaw.cpp:193).
        let n_classes = output.ncols();
        if target.ncols() == 1 {
            let mut cost = 0.0;
            for row in 0..output.nrows() {
                cost += self
                    .cfg
                    .cost_law
                    .compute_unitary_cost(output[[row, 0]], target[[row, 0]]);
            }
            cost
        } else {
            let out_flat: Vec<f64> = output.iter().copied().collect();
            let tgt_flat: Vec<f64> = target.iter().copied().collect();
            self.cfg
                .cost_law
                .compute_cost(&out_flat, &tgt_flat, n_classes)
        }
    }

    // --- Task 9 windowed drivers ---------------------------------------------

    /// `feedForwardBackwardTruncateSweep` (`BLSTMNeuralNetwork.cpp:488-546`):
    /// non-overlapping windows `jj += window_size`; per-window SEQUENTIAL-floor
    /// lengths (LSTM ratios then Output ratios); the partial LAST window is
    /// recomputed; a trailing chunk whose `length_short == 0` (shorter than the
    /// subsampling ratio) is SILENTLY DROPPED -- its `output` rows keep the caller's
    /// prior contents. Write-backs land at `begin/subSamplingRatio` (output) and
    /// `begin/lstmSubSamplingRatio` (fwd/bwd hidden states). FINALLY the caller-
    /// visible `_OutputForward`/`_OutputBackward` are replaced by the stitched
    /// full-length accumulators.
    fn feed_forward_backward_truncate_sweep(
        &mut self,
        input: &Array2<f64>,
        window_size: usize,
        output: &mut Array2<f64>,
        target: &Array2<f64>,
    ) {
        let lstm_sub_ratio = self.forward_lstm_sub_sampling_ratio(); // :489
        let lstm_ratios = self.lstm_sub_sampling(); // :490 forward-net per-layer
        let output_ratios = self.output_sub_sampling(); // :491
        let is_sub_sampling_used = self.sub_sampling_ratio() > 1; // :492

        // :493-496 outputForward/outputBackward length: input rows /= each LSTM ratio.
        let mut length_output_lstm = input.nrows();
        for &r in &lstm_ratios {
            length_output_lstm /= r;
        }
        let fwd_out = self.forward_network.as_ref().unwrap().output_size();
        let bwd_out = self.backward_network.as_ref().unwrap().output_size();
        let mut output_forward = Array2::<f64>::zeros((length_output_lstm, fwd_out));
        let mut output_backward = Array2::<f64>::zeros((length_output_lstm, bwd_out));

        // :500-512 nominal lengths: window /= LSTM ratios (lstmLengthShort) then /=
        // Output ratios (lengthShort). When subsampling is off, both stay = window.
        let sub_sampling_ratio = self.sub_sampling_ratio(); // :501
        let mut length = window_size; // :500
        let mut nominal_lstm_len = length; // :502
        if is_sub_sampling_used {
            for &r in &lstm_ratios {
                length /= r; // :504-506
            }
            nominal_lstm_len = length; // :507
            for &r in &output_ratios {
                length /= r; // :508-510
            }
        }
        let nominal_len = length; // :512

        let input_rows = input.nrows();
        let mut jj = 0;
        while jj < input_rows {
            // :513-517 window bounds; the last window is clamped and shorter.
            let begin = jj;
            let mut end = jj + window_size - 1;
            if end >= input_rows {
                end = input_rows - 1;
            }
            let length_seq = end - begin + 1;

            // :518-532 partial-window length recompute (SEQUENTIAL floors).
            let (length_short, lstm_length_short) = if length_seq != window_size {
                let mut ls = length_seq;
                let mut lstm_ls = ls;
                if is_sub_sampling_used {
                    for &r in &lstm_ratios {
                        ls /= r;
                    }
                    lstm_ls = ls;
                    for &r in &output_ratios {
                        ls /= r;
                    }
                }
                (ls, lstm_ls)
            } else {
                (nominal_len, nominal_lstm_len)
            };

            // :534-542 length_short == 0 -> SILENTLY DROPPED (output rows untouched).
            if length_short > 0 {
                let block = input
                    .slice(ndarray::s![begin..begin + length_seq, ..])
                    .to_owned();
                let mut output_short = Array2::<f64>::zeros((length_short, output.ncols()));
                let target_short = if target.nrows() > 0 {
                    // :537 target slice at begin/subSamplingRatio, length_short rows.
                    target
                        .slice(ndarray::s![
                            begin / sub_sampling_ratio..begin / sub_sampling_ratio + length_short,
                            ..
                        ])
                        .to_owned()
                } else {
                    Array2::<f64>::zeros((0, 0))
                };
                self.feed_forward_backward_plain(&block, &mut output_short, &target_short); // :538

                // :539 write output at begin/subSamplingRatio.
                let obeg = begin / sub_sampling_ratio;
                for r in 0..length_short {
                    for c in 0..output.ncols() {
                        output[[obeg + r, c]] = output_short[[r, c]];
                    }
                }
                // :540-541 write fwd/bwd hidden states at begin/lstmSubSamplingRatio.
                let lbeg = begin / lstm_sub_ratio;
                for r in 0..lstm_length_short {
                    for c in 0..fwd_out {
                        output_forward[[lbeg + r, c]] = self.output_forward[[r, c]];
                    }
                    for c in 0..bwd_out {
                        output_backward[[lbeg + r, c]] = self.output_backward[[r, c]];
                    }
                }
            }
            jj += window_size;
        }

        // :544-545 replace with the stitched full-length accumulators.
        self.output_forward = output_forward;
        self.output_backward = output_backward;
    }

    /// `feedForwardBackwardTruncate` (`BLSTMNeuralNetwork.cpp:548-590`). When
    /// `!two_sweeps` this is a single TruncateSweep. With two sweeps: run a
    /// front-padded sweep and a shift-offset sweep, average their overlapping
    /// outputs, and HCAT the two sweeps' hidden-state windows so
    /// `_OutputForward`/`_OutputBackward` come out DOUBLE-WIDTH. Cost accumulates
    /// over BOTH sweeps (on the padded sequences).
    fn feed_forward_backward_truncate(
        &mut self,
        input: &Array2<f64>,
        window_size: usize,
        output: &mut Array2<f64>,
        target: &Array2<f64>,
    ) {
        if !self.cfg.two_sweeps {
            self.feed_forward_backward_truncate_sweep(input, window_size, output, target); // :588
            return;
        }

        let sub_sampling_ratio = self.sub_sampling_ratio();
        let output_net_ratio = self.output_network.sub_sampling_ratio();
        // :550-552 INTEGER-division ORDER: window_size/2 FIRST, then / ratio.
        let shift_short = (window_size / 2) / sub_sampling_ratio;
        let shift = shift_short * sub_sampling_ratio;
        let window_size_short = window_size / sub_sampling_ratio;

        // :553-554 input padding: first row replicated window_size times, input,
        // last row replicated window_size times.
        let input_padded = pad_replicate_ends(input, window_size, window_size);
        // :555-556 output padding: window_size_short replicas at each end.
        let out_rows = output.nrows();
        let output_padded = pad_replicate_ends(output, window_size_short, window_size_short);

        // :557-563 target padding + drop the FRONT window_size_short rows.
        let target_padded = if target.nrows() > 0 {
            pad_replicate_ends(target, window_size_short, window_size_short)
        } else {
            Array2::<f64>::zeros((0, 0))
        };
        let target_useful_sweep1 = if target.nrows() > 0 {
            target_padded
                .slice(ndarray::s![window_size_short.., ..])
                .to_owned()
        } else {
            Array2::<f64>::zeros((0, 0))
        };

        // :565 sweep 1: input block drops the FRONT window_size padding, KEEPS the
        // back padding; output written into outputSeqPadded.block(window_size_short..).
        let sweep1_in = input_padded
            .slice(ndarray::s![window_size.., ..])
            .to_owned();
        let mut sweep1_out = output_padded
            .slice(ndarray::s![window_size_short.., ..])
            .to_owned();
        self.feed_forward_backward_truncate_sweep(
            &sweep1_in,
            window_size,
            &mut sweep1_out,
            &target_useful_sweep1,
        );

        // :568 outputSeq (sweep 1) = the first out_rows of sweep1_out.
        let mut sweep1_final = sweep1_out.slice(ndarray::s![..out_rows, ..]).to_owned();
        // :569-570 memDisplay hidden windows: the top out_rows*output_net_ratio rows.
        let mem_rows = out_rows * output_net_ratio;
        let mem_forward = self
            .output_forward
            .slice(ndarray::s![..mem_rows, ..])
            .to_owned();
        let mem_backward = self
            .output_backward
            .slice(ndarray::s![..mem_rows, ..])
            .to_owned();

        // :572 reset the padded output buffer to zero for sweep 2.
        let mut output_padded2 = Array2::<f64>::zeros(output_padded.raw_dim());
        // :573-575 sweep-2 target: drop the FRONT shift_short rows.
        let target_useful_sweep2 = if target.nrows() > 0 {
            target_padded
                .slice(ndarray::s![shift_short.., ..])
                .to_owned()
        } else {
            Array2::<f64>::zeros((0, 0))
        };

        // :576 sweep 2: input block from `shift`; output into block(shift_short..).
        let sweep2_in = input_padded.slice(ndarray::s![shift.., ..]).to_owned();
        let mut sweep2_out = output_padded2
            .slice(ndarray::s![shift_short.., ..])
            .to_owned();
        self.feed_forward_backward_truncate_sweep(
            &sweep2_in,
            window_size,
            &mut sweep2_out,
            &target_useful_sweep2,
        );
        // Stitch sweep2_out back into output_padded2 so the window_size_short slice
        // reads the shifted-write region (the legacy mutates the block in place).
        for r in 0..sweep2_out.nrows() {
            for c in 0..output_padded2.ncols() {
                output_padded2[[shift_short + r, c]] = sweep2_out[[r, c]];
            }
        }

        // :578-579 outputSeq += sweep2 window, then /= 2.
        let sweep2_final = output_padded2
            .slice(ndarray::s![
                window_size_short..window_size_short + out_rows,
                ..
            ])
            .to_owned();
        for r in 0..out_rows {
            for c in 0..output.ncols() {
                sweep1_final[[r, c]] = (sweep1_final[[r, c]] + sweep2_final[[r, c]]) / 2.0;
                output[[r, c]] = sweep1_final[[r, c]];
            }
        }

        // :580-581 shifted hidden windows: offset by (window_size_short-shift_short)*ratio.
        let mem_off = (window_size_short - shift_short) * output_net_ratio;
        let mem_forward_shifted = self
            .output_forward
            .slice(ndarray::s![mem_off..mem_off + mem_rows, ..])
            .to_owned();
        let mem_backward_shifted = self
            .output_backward
            .slice(ndarray::s![mem_off..mem_off + mem_rows, ..])
            .to_owned();

        // :583-586 _OutputForward/_OutputBackward become the HCAT [mem, memShifted]
        // -- DOUBLE WIDTH.
        self.output_forward = hcat(&mem_forward, &mem_forward_shifted);
        self.output_backward = hcat(&mem_backward, &mem_backward_shifted);
    }

    /// `feedForwardBackwardOverLap` (`BLSTMNeuralNetwork.cpp:592-681`): overlapping
    /// windows accumulated into per-row sums + counts, then averaged. Window bounds
    /// are SNAPPED to the subsampling grid (begin down, end up). Rows never covered
    /// by any window divide 0/0 -> NaN (REPRODUCED, no guard). The final count-
    /// quotient loop runs `ii < _OutputForward.cols()` for BOTH the forward and the
    /// backward quotient (assumes equal widths).
    ///
    /// IN-PLACE ACCUMULATION (`:664` `outputSeq.block(...).noalias() += outputSeqShort`):
    /// the legacy accumulates window sums DIRECTLY into the caller's `outputSeq`
    /// buffer, so any caller-provided initial contents seed the sums -- a covered
    /// row ends up `(initial + sum)/count`, not `sum/count`. `outputForward`/
    /// `outputBackward` (`:597-598`) are, by contrast, FRESH `Zero` locals -- only
    /// the output path is caller-seeded. The port mirrors this asymmetry: `output`
    /// is accumulated in place (no fresh sum buffer), while `output_forward`/
    /// `output_backward` stay fresh zero accumulators.
    fn feed_forward_backward_overlap(
        &mut self,
        input: &Array2<f64>,
        window_size: usize,
        window_shift: usize,
        output: &mut Array2<f64>,
        target: &Array2<f64>,
    ) {
        let lstm_ratios = self.lstm_sub_sampling(); // :593
        let output_ratios = self.output_sub_sampling(); // :594
        let is_sub_sampling_used = self.sub_sampling_ratio() > 1; // :595
        let sub_sampling_ratio = self.sub_sampling_ratio();
        let output_net_ratio = self.output_network.sub_sampling_ratio();

        // :596 lengthOutputLSTM = outputSeq.rows() * outputNetRatio.
        let length_output_lstm = output.nrows() * output_net_ratio;
        let fwd_out = self.forward_network.as_ref().unwrap().output_size();
        let bwd_out = self.backward_network.as_ref().unwrap().output_size();
        let mut output_forward = Array2::<f64>::zeros((length_output_lstm, fwd_out)); // :597
        let mut output_backward = Array2::<f64>::zeros((length_output_lstm, bwd_out)); // :598
        let mut output_count = vec![0.0_f64; output.nrows()]; // :599
        let mut output_count_lstm = vec![0.0_f64; length_output_lstm]; // :600
        // :664 accumulates INTO the caller's `output` buffer in place (no fresh sum
        // local) -- caller-seeded contents feed the sum, matching Eigen's
        // `outputSeq.block(...).noalias() += outputSeqShort`.

        // :602-610 nominal length = (2*window_size+1) divided sequentially.
        let mut length = 2 * window_size + 1;
        if is_sub_sampling_used {
            for &r in &lstm_ratios {
                length /= r;
            }
            for &r in &output_ratios {
                length /= r;
            }
        }
        let nominal_len = length; // :611
        let nominal_len_lstm = length; // :612

        let input_rows = input.nrows();
        let mut jj = 0;
        while jj < input_rows {
            // :614-620 begin (`jj < window_size ? 0 : jj-window_size`), snapped DOWN
            // to the subsampling grid.
            let mut begin = jj.saturating_sub(window_size);
            begin = (begin / sub_sampling_ratio) * sub_sampling_ratio;
            // :621-626 end = begin + 2*window_size, clamped, snapped UP.
            let mut end = begin + 2 * window_size;
            if end >= input_rows {
                end = input_rows - 1;
            }
            end = ((end + 1) / sub_sampling_ratio) * sub_sampling_ratio - 1;
            let length_seq = end - begin + 1; // :627

            // :628-643 length_short / length_short_lstm recompute for partial windows.
            let (length_short, length_short_lstm) = if length_seq != 2 * window_size + 1 {
                let mut ls = length_seq;
                let mut ls_lstm = length_seq;
                if is_sub_sampling_used {
                    for &r in &lstm_ratios {
                        ls /= r;
                        ls_lstm /= r;
                    }
                    for &r in &output_ratios {
                        ls /= r;
                    }
                }
                (ls, ls_lstm)
            } else {
                (nominal_len, nominal_len_lstm)
            };

            // :645-670 process the window; accumulate sums + counts.
            if length_short > 0 {
                let block = input
                    .slice(ndarray::s![begin..begin + length_seq, ..])
                    .to_owned();
                let mut output_short = Array2::<f64>::zeros((length_short, self.output_size()));
                let target_short = if target.nrows() > 0 {
                    // :649 target slice at begin/subSamplingRatio, length_short rows.
                    target
                        .slice(ndarray::s![
                            begin / sub_sampling_ratio..begin / sub_sampling_ratio + length_short,
                            ..
                        ])
                        .to_owned()
                } else {
                    Array2::<f64>::zeros((0, 0))
                };
                self.feed_forward_backward_plain(&block, &mut output_short, &target_short); // :656

                // :664-665 output sum + count at begin/subSamplingRatio. Accumulates
                // IN PLACE into the caller's `output` buffer (:664 noalias() +=).
                let obeg = begin / sub_sampling_ratio;
                for r in 0..length_short {
                    for c in 0..output.ncols() {
                        output[[obeg + r, c]] += output_short[[r, c]];
                    }
                    output_count[obeg + r] += 1.0;
                }
                // :667-669 fwd/bwd sum + count at begin*outputNetRatio/subSamplingRatio.
                let lbeg = begin * output_net_ratio / sub_sampling_ratio;
                for r in 0..length_short_lstm {
                    for c in 0..fwd_out {
                        output_forward[[lbeg + r, c]] += self.output_forward[[r, c]];
                    }
                    for c in 0..bwd_out {
                        output_backward[[lbeg + r, c]] += self.output_backward[[r, c]];
                    }
                    output_count_lstm[lbeg + r] += 1.0;
                }
            }
            jj += window_shift;
        }

        // :672-673 install the accumulators, then quotient by the counts.
        // :674-675 output /= outputCount (0/0 -> NaN on uncovered rows, no guard).
        // `output` already holds (initial + sum) from the in-place accumulation above.
        for r in 0..output.nrows() {
            for c in 0..output.ncols() {
                output[[r, c]] /= output_count[r];
            }
        }
        // :677-679 quotient loop bound = _OutputForward.cols() for BOTH fwd and bwd.
        for r in 0..length_output_lstm {
            for c in 0..fwd_out {
                output_forward[[r, c]] /= output_count_lstm[r];
                output_backward[[r, c]] /= output_count_lstm[r];
            }
        }
        self.output_forward = output_forward;
        self.output_backward = output_backward;
    }

    /// `feedForwardBackwardMLPOverLap` (`BLSTMNeuralNetwork.cpp:683-709`): IGNORES
    /// the passed `window_size` (overwrites it with `getSubSamplingRatio()/2`);
    /// `length = 2*window_size+1`, `length_short = 1`. Only EXACT windows
    /// (`length_seq == 2*window_size+1`) are processed, each producing one scalar
    /// written to `output[jj/window_shift, 0]`. Edge windows are skipped, so those
    /// rows keep the caller's contents. `_OutputForward`/`_OutputBackward` are
    /// EMPTIED at the end.
    fn feed_forward_backward_mlp_overlap(
        &mut self,
        input: &Array2<f64>,
        _window_size: usize,
        window_shift: usize,
        output: &mut Array2<f64>,
        target: &Array2<f64>,
    ) {
        // :686 the passed window_size is OVERWRITTEN.
        let window_size = self.sub_sampling_ratio() / 2;
        let length_short = 1usize; // :688

        let input_rows = input.nrows();
        let mut jj = 0;
        while jj < input_rows {
            // :690-698 window bounds (`jj < window_size ? 0 : jj-window_size`); only
            // exact windows are processed.
            let begin = jj.saturating_sub(window_size);
            let mut end = jj + window_size;
            if end >= input_rows {
                end = input_rows - 1;
            }
            let length_seq = end - begin + 1;
            if length_seq == 2 * window_size + 1 {
                let block = input
                    .slice(ndarray::s![begin..begin + length_seq, ..])
                    .to_owned();
                let mut output_short = Array2::<f64>::zeros((length_short, 1));
                let target_short = if target.nrows() > 0 {
                    // :702 Constant(length_short, 1, target(jj/window_shift, 0)).
                    Array2::<f64>::from_elem((length_short, 1), target[[jj / window_shift, 0]])
                } else {
                    Array2::<f64>::zeros((0, 0))
                };
                self.feed_forward_backward_mlp(&block, &mut output_short, &target_short); // :703
                output[[jj / window_shift, 0]] = output_short[[0, 0]]; // :704
            }
            jj += window_shift;
        }

        // :707-708 _OutputForward/_OutputBackward emptied.
        self.output_forward = Array2::<f64>::zeros((0, 0));
        self.output_backward = Array2::<f64>::zeros((0, 0));
    }

    /// `_ForwardNetwork.getSubSamplingRatio()` (`BLSTMNeuralNetwork.cpp:489`): the
    /// forward LSTM net's overall decimation ratio (distinct from the whole-BLSTM
    /// `sub_sampling_ratio()`, which multiplies in the output net's ratio).
    fn forward_lstm_sub_sampling_ratio(&self) -> usize {
        self.forward_network.as_ref().unwrap().sub_sampling_ratio()
    }
}

/// Replicate the FIRST row `front` times and the LAST row `back` times around
/// `m`, stacking `[first x front; m; last x back]` (`BLSTMNeuralNetwork.cpp:554`
/// `row(0).replicate(n,1)` idiom). `m` must have >= 1 row.
fn pad_replicate_ends(m: &Array2<f64>, front: usize, back: usize) -> Array2<f64> {
    let (r, c) = m.dim();
    let mut out = Array2::<f64>::zeros((front + r + back, c));
    for k in 0..front {
        for j in 0..c {
            out[[k, j]] = m[[0, j]];
        }
    }
    for i in 0..r {
        for j in 0..c {
            out[[front + i, j]] = m[[i, j]];
        }
    }
    for k in 0..back {
        for j in 0..c {
            out[[front + r + k, j]] = m[[r - 1, j]];
        }
    }
    out
}

/// Horizontal concatenation `[a | b]` (`BLSTMNeuralNetwork.cpp:583-584` resize +
/// `<<`). Requires equal row counts.
fn hcat(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (r, ca) = a.dim();
    let cb = b.ncols();
    let mut out = Array2::<f64>::zeros((r, ca + cb));
    for i in 0..r {
        for j in 0..ca {
            out[[i, j]] = a[[i, j]];
        }
        for j in 0..cb {
            out[[i, ca + j]] = b[[i, j]];
        }
    }
    out
}

#[cfg(test)]
mod analyse_input_seq_tests {
    use super::*;

    /// Minimal MLP-mode net (LSTM `[0,3]` -> `_IsMLP`, output `[5,4,1]` sub
    /// `[2,1]`): `input_size()` = 5 (the output net's input size in MLP mode),
    /// so `analyse_input_seq` crops any wider sequence to the first 5 columns.
    fn mlp_net(input_norm_type: i16) -> BlstmNetwork {
        let mut m: IndexMap<String, String> = IndexMap::new();
        m.insert("SYNT_LSTMNeuronNb".into(), "0,3".into());
        m.insert("SYNT_LSTMSubSampling".into(), "1".into());
        m.insert("SYNT_OutputNeuronNb".into(), "5,4,1".into());
        m.insert("SYNT_OutputSubSampling".into(), "2,1".into());
        m.insert(
            "SYNT_InputNormalizationType".into(),
            input_norm_type.to_string(),
        );
        m.insert("SYNT_TwoSweeps".into(), "false".into());
        let cfg = BlstmConfig::from_legacy(&m, "SYNT").unwrap();
        BlstmNetwork::from_config(cfg).unwrap()
    }

    #[test]
    fn crops_when_strictly_wider_than_net_input() {
        // net input_size() == 5; feed a 7-col sequence -> stats must be computed
        // only over the LEFT 5 columns (`InputStatistics::from_matrix` on the
        // cropped slice), the extra 2 columns must not perturb mean/std.
        let mut net = mlp_net(1);
        let wide = Array2::from_shape_fn((4, 7), |(r, c)| (r * 7 + c) as f64);
        net.analyse_input_seq(&wide);

        let cropped = wide.slice(ndarray::s![.., ..5]).to_owned();
        let expected = InputStatistics::from_matrix(&cropped);
        assert_eq!(net.input_statistics.mean, expected.mean);
        assert_eq!(net.input_statistics.std, expected.std);
        assert_eq!(net.input_statistics.n, expected.n);
    }

    #[test]
    fn first_call_constructs_second_call_updates() {
        // First call on a fresh net (n == 0): the batch REPLACES the accumulator
        // wholesale (no merge). Second call: n != 0, so it MERGES via `update`
        // (n accumulates: 3 + 3 == 6), not a plain overwrite.
        let mut net = mlp_net(1);
        assert_eq!(net.input_statistics.n, 0);

        let batch1 = Array2::from_shape_fn((3, 5), |(r, c)| (r + c) as f64);
        net.analyse_input_seq(&batch1);
        let expected1 = InputStatistics::from_matrix(&batch1);
        assert_eq!(net.input_statistics, expected1);
        assert_eq!(net.input_statistics.n, 3);

        let batch2 = Array2::from_shape_fn((3, 5), |(r, c)| (2 * r + c + 1) as f64);
        net.analyse_input_seq(&batch2);
        assert_eq!(net.input_statistics.n, 6);

        // The merged result must match calling `update` directly, not a fresh
        // `from_matrix` over either batch alone (proves the n != 0 branch merges
        // rather than overwriting).
        let mut expected_merged = InputStatistics::from_matrix(&batch1);
        expected_merged.update(&InputStatistics::from_matrix(&batch2));
        assert_eq!(net.input_statistics, expected_merged);
    }

    #[test]
    fn zero_rows_is_a_noop() {
        // `rows > 0` guard (:385): an empty sequence must not touch the
        // accumulator at all (not even a zero-sample merge).
        let mut net = mlp_net(1);
        let batch = Array2::from_shape_fn((3, 5), |(r, c)| (r + c) as f64);
        net.analyse_input_seq(&batch);
        let before = net.input_statistics.clone();

        let empty = Array2::<f64>::zeros((0, 5));
        net.analyse_input_seq(&empty);
        assert_eq!(net.input_statistics, before);
    }
}
