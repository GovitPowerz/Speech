//! Canonical config model + the config<->nnet adim_coeff seam + flat weight packer.
//!
//! Ported from legacy MATLAB config2network/config2weights and C++ ConfigFile.
//! See design spec sections 4-7.

use std::path::Path;

use anyhow::Context;
use indexmap::IndexMap;
use serde::Deserialize;

use crate::legacy_config::get_usize_list;

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

impl NnetSpec {
    /// Extract the spec from a parsed legacy config using the given algName prefix (e.g. "BLSTM").
    pub fn from_legacy(m: &IndexMap<String, String>, prefix: &str) -> anyhow::Result<NnetSpec> {
        let p = format!("{prefix}_");
        // `PeepholeFlags::from_legacy`'s rule (issue #32): absent -> true, malformed -> error.
        let flag = |k: &str| -> anyhow::Result<bool> {
            match m.get(k) {
                None => Ok(true),
                Some(v) => v
                    .trim()
                    .parse::<bool>()
                    .with_context(|| format!("cannot read '{v}' as bool for '{k}'")),
            }
        };
        Ok(NnetSpec {
            lstm_neuron_nb: get_usize_list(m, &format!("{p}LSTMNeuronNb"))?,
            lstm_subsampling: get_usize_list(m, &format!("{p}LSTMSubSampling"))?,
            output_neuron_nb: get_usize_list(m, &format!("{p}OutputNeuronNb"))?,
            output_subsampling: get_usize_list(m, &format!("{p}OutputSubSampling"))?,
            input_size: m
                .get(&format!("{p}NNetInputSize"))
                .with_context(|| format!("param '{p}NNetInputSize' not found in config"))?
                .trim()
                .parse()?,
            peepholes: [
                flag(&format!("{p}Forward_IsCellsPeepholesActive"))?,
                flag(&format!("{p}Backward_IsCellsPeepholesActive"))?,
                flag(&format!("{p}Forward_IsGatesPeepholesActive"))?,
                flag(&format!("{p}Backward_IsGatesPeepholesActive"))?,
                flag(&format!("{p}Forward_IsGatesRecurrentPeepholesActive"))?,
                flag(&format!("{p}Backward_IsGatesRecurrentPeepholesActive"))?,
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// Structured config-domain model + config<->nnet adim seam + flat weight packer.
//
// Matrices are stored ROW-MAJOR in a flat `Vec<f64>` of `out * ncols`: element
// `(r, c)` lives at `r * ncols + c`. This matches numpy C-order and MATLAB's
// `reshape(X', n, 1)`, so a `.reshape(-1)` in Python == iterating the Vec here.
//
// Per LSTM layer, with `out = LSTMNeuronNb[i+1]` and `fin = LSTMNeuronNb[i]*LSTMSubSampling[i]`:
//   - Input/Forget/Output gate rows are `out x (fin+out+5)`
//     laid out `[recurrent(out) | fan-in(fin) | gate-peephole(3) | recurrent-peephole(1) | bias(1)]`.
//   - The Cell matrix has NO peepholes: rows are `out x (fin+out+1)`, `[recurrent | fan-in | bias]`.
// Output layers are `out x (in+1)` (`[weights(in) | bias(1)]`). mean/std are 1D of length LSTMNeuronNb[0].

/// One LSTM layer's four gate matrices (row-major; cell is narrower).
#[derive(Debug, Clone, PartialEq)]
pub struct LstmLayer {
    pub input: Vec<f64>,
    pub forget: Vec<f64>,
    pub output: Vec<f64>,
    pub cell: Vec<f64>,
}

impl LstmLayer {
    fn gate(&self, g: Gate) -> &[f64] {
        match g {
            Gate::Input => &self.input,
            Gate::Forget => &self.forget,
            Gate::Output => &self.output,
            Gate::Cell => &self.cell,
        }
    }
}

#[derive(Clone, Copy)]
enum Gate {
    Input,
    Forget,
    Output,
    Cell,
}

const GATES: [Gate; 4] = [Gate::Input, Gate::Forget, Gate::Output, Gate::Cell];

/// A full network in either the config- or nnet-domain (same shape; adim differs).
#[derive(Debug, Clone, PartialEq)]
pub struct Nnet {
    pub forward: Vec<LstmLayer>,
    pub backward: Vec<LstmLayer>,
    pub output: Vec<Vec<f64>>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

/// Config-domain structured rows (identical container to `Nnet`).
pub type Structured = Nnet;

/// Total number of flat weight elements for a spec (2 directions + output + mean/std).
pub fn element_count(spec: &NnetSpec) -> usize {
    let lstm = &spec.lstm_neuron_nb;
    let lsub = &spec.lstm_subsampling;
    let outn = &spec.output_neuron_nb;
    let mut total = 0usize;
    for i in 0..lstm.len() - 1 {
        let out = lstm[i + 1];
        let fin = lstm[i] * lsub[i];
        total += 2 * (4 * out * fin + 4 * out * out + 12 * out + 4 * out);
    }
    for i in 0..outn.len() - 1 {
        total += outn[i + 1] * outn[i] + outn[i + 1];
    }
    total += 2 * lstm[0];
    total
}

fn lstm_adim(spec: &NnetSpec, i: usize) -> f64 {
    ((spec.lstm_neuron_nb[i] * spec.lstm_subsampling[i] + spec.lstm_neuron_nb[i + 1]) as f64).sqrt()
}

fn out_adim(spec: &NnetSpec, i: usize) -> f64 {
    ((spec.output_neuron_nb[i] * spec.output_subsampling[i]) as f64).sqrt()
}

/// Divide every column by `adim` EXCEPT the last (bias) column of each row.
fn apply_adim(mat: &[f64], ncols: usize, adim: f64) -> Vec<f64> {
    let mut out = mat.to_vec();
    for row in out.chunks_mut(ncols) {
        let bias = row[ncols - 1];
        for x in row.iter_mut() {
            *x /= adim;
        }
        // Bias restored to its ORIGINAL value here. Legacy MATLAB config2network.m instead
        // recomputes it as (config/adim)*adim, which can differ from the original by up to
        // 1 ULP on ~34 biases. Harmless for Phase 0a (Rust and Python agree exactly); flagged
        // for the Phase-2 inference-vs-legacy golden.
        row[ncols - 1] = bias;
    }
    out
}

/// Config-domain structured rows -> nnet-domain matrices (adim decode).
pub fn config_to_nnet(structured: &Structured, spec: &NnetSpec) -> Nnet {
    let lstm = &spec.lstm_neuron_nb;
    let convert_dir = |layers: &[LstmLayer]| -> Vec<LstmLayer> {
        layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let adim = lstm_adim(spec, i);
                let out = lstm[i + 1];
                let fin = lstm[i] * spec.lstm_subsampling[i];
                let gate_cols = fin + out + 5;
                let cell_cols = fin + out + 1;
                LstmLayer {
                    input: apply_adim(&layer.input, gate_cols, adim),
                    forget: apply_adim(&layer.forget, gate_cols, adim),
                    output: apply_adim(&layer.output, gate_cols, adim),
                    cell: apply_adim(&layer.cell, cell_cols, adim),
                }
            })
            .collect()
    };
    let output = structured
        .output
        .iter()
        .enumerate()
        .map(|(i, mat)| {
            let ncols = spec.output_neuron_nb[i] + 1;
            apply_adim(mat, ncols, out_adim(spec, i))
        })
        .collect();
    Nnet {
        forward: convert_dir(&structured.forward),
        backward: convert_dir(&structured.backward),
        output,
        mean: structured.mean.clone(),
        std: structured.std.clone(),
    }
}

/// Emit the 13 flat blocks of one LSTM layer into `out_vec`.
fn pack_lstm_layer(layer: &LstmLayer, out: usize, fin: usize, out_vec: &mut Vec<f64>) {
    let gate_cols = fin + out + 5;
    let cell_cols = fin + out + 1;
    let cols = |g: Gate| {
        if matches!(g, Gate::Cell) {
            cell_cols
        } else {
            gate_cols
        }
    };
    // 1-4 fan-in blocks (columns out..out+fin), row-major.
    for &g in &GATES {
        let m = layer.gate(g);
        let nc = cols(g);
        for r in 0..out {
            out_vec.extend_from_slice(&m[r * nc + out..r * nc + out + fin]);
        }
    }
    // 5-8 recurrent blocks (columns 0..out), row-major.
    for &g in &GATES {
        let m = layer.gate(g);
        let nc = cols(g);
        for r in 0..out {
            out_vec.extend_from_slice(&m[r * nc..r * nc + out]);
        }
    }
    // 9 peephole bundle (out x 12), row-major; only I/F/O carry peepholes.
    let iw = &layer.input;
    let fw = &layer.forget;
    let ow = &layer.output;
    let at = |m: &[f64], r: usize, c: usize| m[r * gate_cols + c];
    for r in 0..out {
        out_vec.extend_from_slice(&[
            at(iw, r, gate_cols - 2),
            at(fw, r, gate_cols - 2),
            at(ow, r, gate_cols - 2),
            at(iw, r, gate_cols - 5),
            at(iw, r, gate_cols - 4),
            at(iw, r, gate_cols - 3),
            at(fw, r, gate_cols - 5),
            at(fw, r, gate_cols - 4),
            at(fw, r, gate_cols - 3),
            at(ow, r, gate_cols - 5),
            at(ow, r, gate_cols - 4),
            at(ow, r, gate_cols - 3),
        ]);
    }
    // 10-13 biases (each gate's own last column; cell is narrower).
    for &g in &GATES {
        let m = layer.gate(g);
        let nc = cols(g);
        for r in 0..out {
            out_vec.push(m[r * nc + nc - 1]);
        }
    }
}

/// Flatten an nnet-domain network into the canonical row-major weight vector.
pub fn nnet_to_flat(nnet: &Nnet, spec: &NnetSpec) -> Vec<f64> {
    let lstm = &spec.lstm_neuron_nb;
    let mut flat = Vec::with_capacity(element_count(spec));
    for dir in [&nnet.forward, &nnet.backward] {
        for (i, layer) in dir.iter().enumerate() {
            let out = lstm[i + 1];
            let fin = lstm[i] * spec.lstm_subsampling[i];
            pack_lstm_layer(layer, out, fin, &mut flat);
        }
    }
    for (i, mat) in nnet.output.iter().enumerate() {
        let out = spec.output_neuron_nb[i + 1];
        let inp = spec.output_neuron_nb[i];
        let ncols = inp + 1;
        for r in 0..out {
            flat.extend_from_slice(&mat[r * ncols..r * ncols + inp]); // weights
        }
        for r in 0..out {
            flat.push(mat[r * ncols + inp]); // bias
        }
    }
    flat.extend_from_slice(&nnet.mean);
    flat.extend_from_slice(&nnet.std);
    flat
}

/// Inverse of `nnet_to_flat`: slice the flat vector back into nnet-domain matrices.
pub fn flat_to_nnet(flat: &[f64], spec: &NnetSpec) -> Nnet {
    let lstm = &spec.lstm_neuron_nb;
    let outn = &spec.output_neuron_nb;
    let mut pos = 0usize;
    let mut take = |k: usize| -> &[f64] {
        let seg = &flat[pos..pos + k];
        pos += k;
        seg
    };
    let mut unpack_dir = |count: usize| -> Vec<LstmLayer> {
        (0..count)
            .map(|i| {
                let out = lstm[i + 1];
                let fin = lstm[i] * spec.lstm_subsampling[i];
                let gate_cols = fin + out + 5;
                let cell_cols = fin + out + 1;
                let mut input = vec![0.0; out * gate_cols];
                let mut forget = vec![0.0; out * gate_cols];
                let mut output = vec![0.0; out * gate_cols];
                let mut cell = vec![0.0; out * cell_cols];
                // fan-in blocks (columns out..out+fin).
                for m in [&mut input, &mut forget, &mut output] {
                    let seg = take(out * fin);
                    for r in 0..out {
                        m[r * gate_cols + out..r * gate_cols + out + fin]
                            .copy_from_slice(&seg[r * fin..(r + 1) * fin]);
                    }
                }
                {
                    let seg = take(out * fin);
                    for r in 0..out {
                        cell[r * cell_cols + out..r * cell_cols + out + fin]
                            .copy_from_slice(&seg[r * fin..(r + 1) * fin]);
                    }
                }
                // recurrent blocks (columns 0..out).
                for m in [&mut input, &mut forget, &mut output] {
                    let seg = take(out * out);
                    for r in 0..out {
                        m[r * gate_cols..r * gate_cols + out]
                            .copy_from_slice(&seg[r * out..(r + 1) * out]);
                    }
                }
                {
                    let seg = take(out * out);
                    for r in 0..out {
                        cell[r * cell_cols..r * cell_cols + out]
                            .copy_from_slice(&seg[r * out..(r + 1) * out]);
                    }
                }
                // peephole bundle (out x 12).
                let peep = take(12 * out);
                for r in 0..out {
                    let p = &peep[r * 12..r * 12 + 12];
                    input[r * gate_cols + gate_cols - 2] = p[0];
                    forget[r * gate_cols + gate_cols - 2] = p[1];
                    output[r * gate_cols + gate_cols - 2] = p[2];
                    input[r * gate_cols + gate_cols - 5..r * gate_cols + gate_cols - 2]
                        .copy_from_slice(&p[3..6]);
                    forget[r * gate_cols + gate_cols - 5..r * gate_cols + gate_cols - 2]
                        .copy_from_slice(&p[6..9]);
                    output[r * gate_cols + gate_cols - 5..r * gate_cols + gate_cols - 2]
                        .copy_from_slice(&p[9..12]);
                }
                // biases (each gate's own last column; cell is narrower).
                for m in [&mut input, &mut forget, &mut output] {
                    let seg = take(out);
                    for r in 0..out {
                        m[r * gate_cols + gate_cols - 1] = seg[r];
                    }
                }
                let seg = take(out);
                for r in 0..out {
                    cell[r * cell_cols + cell_cols - 1] = seg[r];
                }
                LstmLayer {
                    input,
                    forget,
                    output,
                    cell,
                }
            })
            .collect()
    };
    let forward = unpack_dir(lstm.len() - 1);
    let backward = unpack_dir(lstm.len() - 1);
    let output = (0..outn.len() - 1)
        .map(|i| {
            let out = outn[i + 1];
            let inp = outn[i];
            let ncols = inp + 1;
            let mut mat = vec![0.0; out * ncols];
            let seg = take(out * inp);
            for r in 0..out {
                mat[r * ncols..r * ncols + inp].copy_from_slice(&seg[r * inp..(r + 1) * inp]);
            }
            let seg = take(out);
            for r in 0..out {
                mat[r * ncols + inp] = seg[r];
            }
            mat
        })
        .collect();
    let mean = take(lstm[0]).to_vec();
    let std = take(lstm[0]).to_vec();
    Nnet {
        forward,
        backward,
        output,
        mean,
        std,
    }
}

#[derive(Deserialize)]
struct RowMeta {
    name: String,
    len: usize,
}

#[derive(Deserialize)]
struct SpecJson {
    #[serde(rename = "LSTMNeuronNb")]
    lstm_neuron_nb: Vec<usize>,
    #[serde(rename = "LSTMSubSampling")]
    lstm_subsampling: Vec<usize>,
    #[serde(rename = "OutputNeuronNb")]
    output_neuron_nb: Vec<usize>,
    #[serde(rename = "OutputSubSampling")]
    output_subsampling: Vec<usize>,
    #[serde(rename = "NNetInputSize")]
    input_size: usize,
}

#[derive(Deserialize)]
struct Manifest {
    spec: SpecJson,
    rows: Vec<RowMeta>,
}

/// Load `(structured, spec)` from the committed manifest + concatenated-rows .bin.
///
/// Row names follow `{direction}.L{i}.B{b}.{gate}`, `output.L{i}.N{nn}`, `mean`, `std`.
pub fn load_structured(
    manifest_path: &Path,
    bin_path: &Path,
) -> anyhow::Result<(Structured, NnetSpec)> {
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?,
    )?;
    let spec = NnetSpec {
        lstm_neuron_nb: manifest.spec.lstm_neuron_nb,
        lstm_subsampling: manifest.spec.lstm_subsampling,
        output_neuron_nb: manifest.spec.output_neuron_nb,
        output_subsampling: manifest.spec.output_subsampling,
        input_size: manifest.spec.input_size,
        peepholes: [true; 6],
    };
    let (_rows, _cols, blob) = crate::io::binary::read_matrix(bin_path)?;
    let mut named: IndexMap<String, Vec<f64>> = IndexMap::new();
    let mut off = 0usize;
    for r in &manifest.rows {
        named.insert(r.name.clone(), blob[off..off + r.len].to_vec());
        off += r.len;
    }
    let row = |name: &str| -> anyhow::Result<Vec<f64>> {
        named
            .get(name)
            .cloned()
            .with_context(|| format!("missing manifest row {name}"))
    };
    let lstm = &spec.lstm_neuron_nb;
    let outn = &spec.output_neuron_nb;
    let build_dir = |direction: &str| -> anyhow::Result<Vec<LstmLayer>> {
        (0..lstm.len() - 1)
            .map(|i| {
                let out = lstm[i + 1];
                let concat = |gate: &str| -> anyhow::Result<Vec<f64>> {
                    let mut v = Vec::new();
                    for b in 0..out {
                        v.extend(row(&format!("{direction}.L{i}.B{b}.{gate}"))?);
                    }
                    Ok(v)
                };
                Ok(LstmLayer {
                    input: concat("input")?,
                    forget: concat("forget")?,
                    output: concat("output")?,
                    cell: concat("cell")?,
                })
            })
            .collect()
    };
    let forward = build_dir("forward")?;
    let backward = build_dir("backward")?;
    let mut output = Vec::new();
    for i in 0..outn.len() - 1 {
        let mut mat = Vec::new();
        for nn in 0..outn[i + 1] {
            mat.extend(row(&format!("output.L{i}.N{nn}"))?);
        }
        output.push(mat);
    }
    let structured = Structured {
        forward,
        backward,
        output,
        mean: row("mean")?,
        std: row("std")?,
    };
    Ok((structured, spec))
}
