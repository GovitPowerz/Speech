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

use crate::features::stats::InputStatistics;

use super::layers::NeuronLayer;
use super::network::Network;

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

    /// `resetWeightsDerivatives` (`:278-287`): resets ONLY the `InputStatistics`
    /// field here (the per-sub-network gradient-accumulator reset is a Phase 3
    /// stub; `_Cost`/`_NbOfClassif`/`_OutputForward`/`_OutputBackward` are
    /// deliberately untouched, matching the legacy).
    pub fn reset_weights_derivatives(&mut self) {
        self.input_statistics = InputStatistics::new();
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
}
