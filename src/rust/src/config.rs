//! Canonical config model + the config<->nnet adim_coeff seam + flat weight packer.
//!
//! Ported from legacy MATLAB config2network/config2weights and C++ ConfigFile.
//! See design spec sections 4-7.

use anyhow::Context;
use indexmap::IndexMap;

/// NNType-0 network spec parsed from a legacy config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NnetSpec {
    pub lstm_neuron_nb: Vec<usize>,
    pub lstm_subsampling: Vec<usize>,
    pub output_neuron_nb: Vec<usize>,
    pub output_subsampling: Vec<usize>,
    pub input_size: usize,
    pub peepholes: [bool; 6],
}

fn ints(m: &IndexMap<String, String>, key: &str) -> anyhow::Result<Vec<usize>> {
    m.get(key)
        .with_context(|| format!("missing key {key}"))?
        .split(',')
        .map(|s| s.trim().parse::<usize>().map_err(Into::into))
        .collect()
}

impl NnetSpec {
    /// Extract the spec from a parsed legacy config using the given algName prefix (e.g. "BLSTM").
    pub fn from_legacy(m: &IndexMap<String, String>, prefix: &str) -> anyhow::Result<NnetSpec> {
        let p = format!("{prefix}_");
        let flag = |k: &str| m.get(k).map(|v| v == "true").unwrap_or(false);
        Ok(NnetSpec {
            lstm_neuron_nb: ints(m, &format!("{p}LSTMNeuronNb"))?,
            lstm_subsampling: ints(m, &format!("{p}LSTMSubSampling"))?,
            output_neuron_nb: ints(m, &format!("{p}OutputNeuronNb"))?,
            output_subsampling: ints(m, &format!("{p}OutputSubSampling"))?,
            input_size: m
                .get(&format!("{p}NNetInputSize"))
                .context("missing NNetInputSize")?
                .trim()
                .parse()?,
            peepholes: [
                flag(&format!("{p}Forward_IsCellsPeepholesActive")),
                flag(&format!("{p}Backward_IsCellsPeepholesActive")),
                flag(&format!("{p}Forward_IsGatesPeepholesActive")),
                flag(&format!("{p}Backward_IsGatesPeepholesActive")),
                flag(&format!("{p}Forward_IsGatesRecurrentPeepholesActive")),
                flag(&format!("{p}Backward_IsGatesRecurrentPeepholesActive")),
            ],
        })
    }
}
