//! Generic stacked-layer container over a `Layer` trait; sub-sample decimation.
//!
//! Ported from legacy C++: `NeuralNetwork.hpp` (Phase 2 Task 6). The container
//! stacks `Layer` impls, threads a chained flat weight vector through them
//! (layer-0-first head/tail split), sub-samples between layers (frame-contiguous
//! temporal stacking, dropped tail), and drives forward / reverse / double-input
//! passes. `_LayersOutput` intermediate caches are retained as fields; the Phase 3
//! backward pass (`feed_backward`/`feed_backward_reverse`/`feed_backward_double`,
//! `NeuralNetwork.hpp:251-362`) reads them and threads the `inv_sub_sample`
//! (`:136-145`) delta inflation + running `invSubSamplingRatio` through the
//! reverse-order layer loop.

use ndarray::Array2;

use super::layers::{LstmLayer, NeuronLayer};

/// A network layer: forward (and reverse) over a whole sequence, plus flat weight
/// (de)serialization with the chained head/tail contract.
///
/// `set_weights` consumes `flat[..nb_of_weights()]` and returns the tail slice for
/// the next layer in the stack (legacy `LayerType::setWeights` returns
/// `weights.tail(...)`). `get_weights` appends this layer's weights (same order) to
/// `out`. `feed_forward`/`feed_forward_reverse` write the caller-allocated `output`
/// (never resize the caller's buffer semantically -- they overwrite it in place).
pub trait Layer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool);
    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    );
    /// `LayerType::feedBackward(input, output, deltas, invSubSamplingRatio, lastLayer)`:
    /// accumulate this layer's weight derivatives from the reverse pass and return the
    /// deltas propagated to the previous layer (T x input_size). The layer scales its
    /// deriv blocks by `inv_sub_sampling_ratio` when > 1 (Tasks 3/4 already implement
    /// the layer-side scaling).
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64>;
    /// `LayerType::feedBackwardReverse` (reverse-time variant). `NeuronLayer` never runs
    /// reversed and panics, mirroring `feed_forward_reverse`.
    fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64>;
    /// `LayerType::getWeightsDerivatives`: append this layer's Nx2 `[deriv | count]`
    /// rows (flat weight-packer block order, col1 = frame count replicated) to `out`.
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>);
    /// `LayerType::resetWeightsDerivatives`: zero the deriv accumulators + frame count.
    fn reset_weights_derivatives(&mut self);
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64];
    fn get_weights(&self, out: &mut Vec<f64>);
    fn nb_of_weights(&self) -> usize;
}

impl Layer for LstmLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        LstmLayer::feed_forward(self, input, output, last_layer);
    }
    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        LstmLayer::feed_forward_reverse(self, input, output, last_layer);
    }
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        LstmLayer::feed_backward(
            self,
            input,
            output,
            deltas,
            inv_sub_sampling_ratio,
            last_layer,
        )
    }
    fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        LstmLayer::feed_backward_reverse(
            self,
            input,
            output,
            deltas,
            inv_sub_sampling_ratio,
            last_layer,
        )
    }
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        LstmLayer::get_weights_derivatives(self, out);
    }
    fn reset_weights_derivatives(&mut self) {
        LstmLayer::reset_weights_derivatives(self);
    }
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        LstmLayer::set_weights(self, flat)
    }
    fn get_weights(&self, out: &mut Vec<f64>) {
        LstmLayer::get_weights(self, out);
    }
    fn nb_of_weights(&self) -> usize {
        LstmLayer::nb_of_weights(self)
    }
}

impl Layer for NeuronLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        NeuronLayer::feed_forward(self, input, output, last_layer);
    }
    /// `NeuronLayer` has no reverse variant -- the legacy `NeuralNetwork<NeuronLayer>`
    /// is only ever run forward (dense output MLPs). Panics if ever driven reversed.
    fn feed_forward_reverse(
        &mut self,
        _input: &Array2<f64>,
        _output: &mut Array2<f64>,
        _last_layer: bool,
    ) {
        panic!("NeuronLayer is never run reversed (legacy never instantiates it)");
    }
    /// `NeuronLayer::feedBackward` takes NO `output` argument (the dense backward reads
    /// the RAW `input` for the hidden asinh-deriv fold, `NeuronLayer.cpp:204`); the
    /// container passes the layer's forward output uniformly, so the trait method drops
    /// it here.
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        _output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        NeuronLayer::feed_backward(self, input, deltas, inv_sub_sampling_ratio, last_layer)
    }
    /// `NeuronLayer` has no reverse variant -- the dense output MLP is never driven
    /// reversed. Panics, mirroring `feed_forward_reverse`.
    fn feed_backward_reverse(
        &mut self,
        _input: &Array2<f64>,
        _output: &Array2<f64>,
        _deltas: &Array2<f64>,
        _inv_sub_sampling_ratio: usize,
        _last_layer: bool,
    ) -> Array2<f64> {
        panic!("NeuronLayer is never run reversed (legacy never instantiates it)");
    }
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        NeuronLayer::get_weights_derivatives(self, out);
    }
    fn reset_weights_derivatives(&mut self) {
        NeuronLayer::reset_weights_derivatives(self);
    }
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        NeuronLayer::set_weights(self, flat)
    }
    fn get_weights(&self, out: &mut Vec<f64>) {
        NeuronLayer::get_weights(self, out);
    }
    fn nb_of_weights(&self) -> usize {
        NeuronLayer::nb_of_weights(self)
    }
}

/// `NeuralNetwork<LayerType>` (`NeuralNetwork.hpp:13-249`): a stack of `_NeuronNb.len()
/// - 1` layers with per-layer sub-sampling, a chained flat weight vector, and forward
/// / reverse / double-input drivers. `neuron_nb[jj]`/`neuron_nb[jj+1]` are the fan-in
/// (times `sub_sampling[jj]`) / fan-out of layer `jj`.
///
/// `Clone` requires `L: Clone` (the derive bound); both `LstmLayer` and `NeuronLayer`
/// derive it. Enables the whole-net clone the Task 4/7 driver bags depend on.
#[derive(Clone)]
pub struct Network<L: Layer> {
    neuron_nb: Vec<usize>,
    sub_sampling: Vec<usize>,
    /// Cumulative sub-sampling products (`_SubSamplingRatios`): `sub_sampling_ratios
    /// [jj] = prod(sub_sampling[0..=jj])`.
    sub_sampling_ratios: Vec<usize>,
    layers: Vec<L>,
    /// Retained intermediate outputs (`_LayersOutput`), one per non-final layer; the
    /// Phase 3 backward pass reads them. Filled by `feed_forward`/`feed_forward_reverse`.
    layers_output: Vec<Array2<f64>>,
}

impl<L: Layer> Network<L> {
    /// `NeuralNetwork(conf, prep, neuronNb, subSampling, weightsSetExternally)` ctor
    /// (`:25-37`), factoring the config/prep out into a `mk(layer_id, input, output)`
    /// factory: layer `jj` has input `neuron_nb[jj]*sub_sampling[jj]`, output
    /// `neuron_nb[jj+1]`. `sub_sampling_ratios` are the cumulative products (`:31-36`).
    ///
    /// The ILL-FORMED legacy copy ctor (`:39-51`) is deliberately NOT ported -- it
    /// compiles only because it is never instantiated.
    pub fn new(
        neuron_nb: Vec<usize>,
        sub_sampling: Vec<usize>,
        mut mk: impl FnMut(usize, usize, usize) -> L,
    ) -> Self {
        assert!(
            neuron_nb.len() >= 2,
            "network needs >= 2 neuron-count entries"
        );
        assert_eq!(
            sub_sampling.len(),
            neuron_nb.len() - 1,
            "sub_sampling must have one entry per layer"
        );
        let n_layers = neuron_nb.len() - 1;
        let mut layers = Vec::with_capacity(n_layers);
        let mut layers_output = Vec::with_capacity(n_layers);
        for jj in 0..n_layers {
            layers.push(mk(jj, neuron_nb[jj] * sub_sampling[jj], neuron_nb[jj + 1]));
            layers_output.push(Array2::zeros((0, 0)));
        }
        // :31-36 cumulative products: sub_sampling_ratios[jj] = prod(sub_sampling[0..=jj]).
        // Kept as the legacy's index-nested double loop for a line-for-line source match
        // (an iterator rewrite obscures the `ll <= jj` triangular accumulation).
        let mut sub_sampling_ratios = vec![1usize; sub_sampling.len()];
        #[allow(clippy::needless_range_loop)]
        for jj in 0..sub_sampling.len() {
            for ll in 0..=jj {
                sub_sampling_ratios[jj] *= sub_sampling[ll];
            }
        }
        Network {
            neuron_nb,
            sub_sampling,
            sub_sampling_ratios,
            layers,
            layers_output,
        }
    }

    /// `getInputSize` (`:53-55`): `neuron_nb[0]`.
    pub fn input_size(&self) -> usize {
        self.neuron_nb[0]
    }

    /// `getOutputSize` (`:57-59`): `neuron_nb.last()`.
    pub fn output_size(&self) -> usize {
        self.neuron_nb[self.neuron_nb.len() - 1]
    }

    /// `getSubSampling` (`:61-63`): the raw per-layer sub-sampling factors.
    pub fn sub_samplings(&self) -> &[usize] {
        &self.sub_sampling
    }

    /// `getSubSamplingRatio` (`:65-67`): the LAST cumulative product (overall decimation).
    pub fn sub_sampling_ratio(&self) -> usize {
        self.sub_sampling_ratios[self.sub_sampling_ratios.len() - 1]
    }

    /// `getNbOfWeights` (`:79-85`): sum over layers.
    pub fn nb_of_weights(&self) -> usize {
        self.layers.iter().map(|l| l.nb_of_weights()).sum()
    }

    /// `setWeights` (`:69-77`): thread `flat` through the layers layer-0-first, each
    /// consuming its `nb_of_weights()` head and passing the tail on. Returns the
    /// leftover tail (the legacy returns the final `all`).
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        let mut rest = flat;
        for layer in &mut self.layers {
            rest = layer.set_weights(rest);
        }
        rest
    }

    /// `getWeights` (`:87-96`): concatenate every layer's weights, layer-0-first,
    /// same block order as `set_weights` consumes.
    pub fn get_weights(&self, out: &mut Vec<f64>) {
        for layer in &self.layers {
            layer.get_weights(out);
        }
    }

    /// `feedForward` (`:158-198`). EMPTY input (0 rows) -> silent no-op (`output`
    /// untouched, `:159`). Single-layer net: optional input sub-sample then layer
    /// forward `last_layer=true`. Multi-layer ascending: layer 0 (optionally
    /// sub-sampled input) -> `layers_output[0]` (`last_layer=false`); middle layers
    /// same over `layers_output[jj-1]`; the FINAL layer writes the caller-allocated
    /// `output` (`last_layer=true`). `sub_sampling[jj] > 1` sub-samples that layer's
    /// input first.
    pub fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>) {
        self.drive(input, output, false);
    }

    /// `feedForwardReverse` (`:200-240`): identical driver to `feed_forward`, but each
    /// layer step dispatches `feed_forward_reverse` (the reversal lives inside the
    /// layer). Ascending layer order (only the per-layer temporal direction flips).
    pub fn feed_forward_reverse(&mut self, input: &Array2<f64>, output: &mut Array2<f64>) {
        self.drive(input, output, true);
    }

    /// Shared driver for `feed_forward`/`feed_forward_reverse` (the legacy copy-pastes
    /// the two methods verbatim except the layer call). `reverse` picks the per-layer
    /// kernel.
    fn drive(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, reverse: bool) {
        if input.nrows() == 0 {
            return; // :159 / :201 -- empty input is a no-op
        }
        let n_layers = self.neuron_nb.len() - 1;
        let step = |layer: &mut L, inp: &Array2<f64>, outp: &mut Array2<f64>, last: bool| {
            if reverse {
                layer.feed_forward_reverse(inp, outp, last);
            } else {
                layer.feed_forward(inp, outp, last);
            }
        };

        if self.neuron_nb.len() == 2 {
            // Single-layer net (:160-166 / :202-208).
            if self.sub_sampling[0] > 1 {
                let sub = sub_sample(self.sub_sampling[0], input);
                step(&mut self.layers[0], &sub, output, true);
            } else {
                step(&mut self.layers[0], input, output, true);
            }
            return;
        }

        // Multi-layer ascending (:167-196 / :209-238). `neuron_nb[jj+1]` sizes the
        // intermediate output buffer; the layer forward overwrites it in place.
        for jj in 0..n_layers {
            let out_cols = self.neuron_nb[jj + 1];
            if jj == 0 {
                if self.sub_sampling[0] > 1 {
                    let sub = sub_sample(self.sub_sampling[0], input);
                    let mut buf = Array2::zeros((sub.nrows(), out_cols));
                    step(&mut self.layers[0], &sub, &mut buf, false);
                    self.layers_output[0] = buf;
                } else {
                    let mut buf = Array2::zeros((input.nrows(), out_cols));
                    step(&mut self.layers[0], input, &mut buf, false);
                    self.layers_output[0] = buf;
                }
            } else if jj == n_layers - 1 {
                let prev =
                    std::mem::replace(&mut self.layers_output[jj - 1], Array2::zeros((0, 0)));
                if self.sub_sampling[jj] > 1 {
                    let sub = sub_sample(self.sub_sampling[jj], &prev);
                    step(&mut self.layers[jj], &sub, output, true);
                } else {
                    step(&mut self.layers[jj], &prev, output, true);
                }
                self.layers_output[jj - 1] = prev;
            } else {
                let prev =
                    std::mem::replace(&mut self.layers_output[jj - 1], Array2::zeros((0, 0)));
                let (buf, restored) = if self.sub_sampling[jj] > 1 {
                    let sub = sub_sample(self.sub_sampling[jj], &prev);
                    let mut buf = Array2::zeros((sub.nrows(), out_cols));
                    step(&mut self.layers[jj], &sub, &mut buf, false);
                    (buf, prev)
                } else {
                    let mut buf = Array2::zeros((prev.nrows(), out_cols));
                    step(&mut self.layers[jj], &prev, &mut buf, false);
                    (buf, prev)
                };
                self.layers_output[jj - 1] = restored;
                self.layers_output[jj] = buf;
            }
        }
    }

    /// `feedForwardDouble` (`:242-249`): hcat `first | second` into an
    /// `(rows x neuron_nb[0])` matrix (forward half on the LEFT), then normal forward.
    /// Empty `first` (0 rows) -> no-op. Requires `first.ncols() + second.ncols() ==
    /// neuron_nb[0]` and equal row counts.
    pub fn feed_forward_double(
        &mut self,
        first: &Array2<f64>,
        second: &Array2<f64>,
        output: &mut Array2<f64>,
    ) {
        if first.nrows() == 0 {
            return; // :243
        }
        assert_eq!(first.nrows(), second.nrows(), "double: row-count mismatch");
        assert_eq!(
            first.ncols() + second.ncols(),
            self.neuron_nb[0],
            "double: first|second cols must sum to neuron_nb[0]"
        );
        let rows = first.nrows();
        let mut hcat = Array2::zeros((rows, self.neuron_nb[0]));
        let fc = first.ncols();
        for t in 0..rows {
            for j in 0..fc {
                hcat[[t, j]] = first[[t, j]];
            }
            for j in 0..second.ncols() {
                hcat[[t, fc + j]] = second[[t, j]];
            }
        }
        self.feed_forward(&hcat, output);
    }

    /// `feedBackward` (`NeuralNetwork.hpp:251-300`): the reverse-order layer loop, each
    /// layer fed its RETAINED forward input (`input` for layer 0, `layers_output[jj-1]`
    /// otherwise) + output (`layers_output[jj]` hidden, `output_seq` final) and the
    /// running `inv_sub_sampling_ratio` (starts at 1). A sub-sampled layer (`sub_sampling
    /// [jj] > 1`) re-`sub_sample`s its stored input + output, truncated to the incoming
    /// delta row count via `.topRows(...)`, then `inv_sub_sample`s the returned deltas
    /// back to the pre-decimation resolution and multiplies the running ratio by
    /// `sub_sampling[jj]` (fed to the NEXT-earlier layer). Layer 0's `deltas_out` (already
    /// `inv_sub_sample`d) is returned. Empty input -> empty. Transcribed from the legacy;
    /// the harness `netBackwardLoop` dumps are the arbiter.
    pub fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output_seq: &Array2<f64>,
        deltas: &Array2<f64>,
    ) -> Array2<f64> {
        self.drive_backward(input, output_seq, deltas, false)
    }

    /// `feedBackwardReverse` (`NeuralNetwork.hpp:302-351`): byte-identical driver to
    /// `feed_backward`, only each per-layer step dispatches the REVERSE layer backward
    /// (the reversal lives inside the layer). Same ascending SubSample bookkeeping.
    pub fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output_seq: &Array2<f64>,
        deltas: &Array2<f64>,
    ) -> Array2<f64> {
        self.drive_backward(input, output_seq, deltas, true)
    }

    /// Shared driver for `feed_backward`/`feed_backward_reverse` (the legacy copy-pastes
    /// the two methods verbatim except the per-layer backward call). `reverse` picks the
    /// per-layer kernel.
    fn drive_backward(
        &mut self,
        input: &Array2<f64>,
        output_seq: &Array2<f64>,
        deltas: &Array2<f64>,
        reverse: bool,
    ) -> Array2<f64> {
        // :252-253 deltas_out empty, invSubSamplingRatio = 1.
        let mut deltas_out: Array2<f64> = Array2::zeros((0, 0));
        let mut inv_sub_sampling_ratio = 1usize;
        if input.nrows() == 0 {
            return deltas_out; // :254 -- empty input, empty result
        }
        let n_layers = self.neuron_nb.len() - 1;
        let step = |layer: &mut L,
                    inp: &Array2<f64>,
                    outp: &Array2<f64>,
                    din: &Array2<f64>,
                    ratio: usize,
                    last: bool| {
            if reverse {
                layer.feed_backward_reverse(inp, outp, din, ratio, last)
            } else {
                layer.feed_backward(inp, outp, din, ratio, last)
            }
        };

        if self.neuron_nb.len() == 2 {
            // Single-layer net (:255-263).
            if self.sub_sampling[0] > 1 {
                let sub = sub_sample(self.sub_sampling[0], input);
                let dr = deltas.nrows();
                let sub_top = sub.slice(ndarray::s![..dr, ..]).to_owned();
                let out_top = output_seq.slice(ndarray::s![..dr, ..]).to_owned();
                let dpl = step(
                    &mut self.layers[0],
                    &sub_top,
                    &out_top,
                    deltas,
                    inv_sub_sampling_ratio,
                    true,
                );
                deltas_out = inv_sub_sample(self.sub_sampling[0], &dpl);
                inv_sub_sampling_ratio *= self.sub_sampling[0];
            } else {
                deltas_out = step(
                    &mut self.layers[0],
                    input,
                    output_seq,
                    deltas,
                    inv_sub_sampling_ratio,
                    true,
                );
            }
            let _ = inv_sub_sampling_ratio; // consumed above; kept for source parity
            return deltas_out;
        }

        // Multi-layer reverse order (:264-296). `kk` counts up; `jj = L-2-kk` counts down.
        // The current deltas are the SEED (`deltas`) on the first iteration (last layer,
        // kk==0) and the previous `deltas_out` thereafter -- see :271/:280/:289 which all
        // read `deltas_out` for jj != L-2, `deltas` at jj == L-2.
        for kk in 0..n_layers {
            let jj = n_layers - 1 - kk; // n_layers == _NeuronNb.size()-1, so jj = L-2-kk
            if jj == 0 {
                // :268-276 first layer: input is Input/SubInput, output is layers_output[0].
                let cur = std::mem::replace(&mut deltas_out, Array2::zeros((0, 0)));
                if self.sub_sampling[0] > 1 {
                    let sub = sub_sample(self.sub_sampling[0], input);
                    let dr = cur.nrows();
                    let sub_top = sub.slice(ndarray::s![..dr, ..]).to_owned();
                    let lo = std::mem::replace(&mut self.layers_output[jj], Array2::zeros((0, 0)));
                    let lo_top = lo.slice(ndarray::s![..dr, ..]).to_owned();
                    let dpl = step(
                        &mut self.layers[jj],
                        &sub_top,
                        &lo_top,
                        &cur,
                        inv_sub_sampling_ratio,
                        true,
                    );
                    self.layers_output[jj] = lo;
                    deltas_out = inv_sub_sample(self.sub_sampling[0], &dpl);
                    inv_sub_sampling_ratio *= self.sub_sampling[0];
                } else {
                    let lo = std::mem::replace(&mut self.layers_output[jj], Array2::zeros((0, 0)));
                    deltas_out = step(
                        &mut self.layers[jj],
                        input,
                        &lo,
                        &cur,
                        inv_sub_sampling_ratio,
                        true,
                    );
                    self.layers_output[jj] = lo;
                }
            } else if jj == n_layers - 1 {
                // :277-285 last layer: input is layers_output[jj-1], output is output_seq,
                // deltas is the SEED (`deltas`), lastLayer = false.
                let prev =
                    std::mem::replace(&mut self.layers_output[jj - 1], Array2::zeros((0, 0)));
                if self.sub_sampling[jj] > 1 {
                    let sub = sub_sample(self.sub_sampling[jj], &prev);
                    let dr = deltas.nrows();
                    let sub_top = sub.slice(ndarray::s![..dr, ..]).to_owned();
                    let out_top = output_seq.slice(ndarray::s![..dr, ..]).to_owned();
                    let dpl = step(
                        &mut self.layers[jj],
                        &sub_top,
                        &out_top,
                        deltas,
                        inv_sub_sampling_ratio,
                        false,
                    );
                    deltas_out = inv_sub_sample(self.sub_sampling[jj], &dpl);
                    inv_sub_sampling_ratio *= self.sub_sampling[jj];
                } else {
                    deltas_out = step(
                        &mut self.layers[jj],
                        &prev,
                        output_seq,
                        deltas,
                        inv_sub_sampling_ratio,
                        false,
                    );
                }
                self.layers_output[jj - 1] = prev;
            } else {
                // :286-295 interior layer: input is layers_output[jj-1], output is
                // layers_output[jj], deltas is the running `deltas_out`, lastLayer = false.
                let cur = std::mem::replace(&mut deltas_out, Array2::zeros((0, 0)));
                let prev =
                    std::mem::replace(&mut self.layers_output[jj - 1], Array2::zeros((0, 0)));
                if self.sub_sampling[jj] > 1 {
                    let sub = sub_sample(self.sub_sampling[jj], &prev);
                    let dr = cur.nrows();
                    let sub_top = sub.slice(ndarray::s![..dr, ..]).to_owned();
                    let lo = std::mem::replace(&mut self.layers_output[jj], Array2::zeros((0, 0)));
                    let lo_top = lo.slice(ndarray::s![..dr, ..]).to_owned();
                    let dpl = step(
                        &mut self.layers[jj],
                        &sub_top,
                        &lo_top,
                        &cur,
                        inv_sub_sampling_ratio,
                        false,
                    );
                    self.layers_output[jj] = lo;
                    deltas_out = inv_sub_sample(self.sub_sampling[jj], &dpl);
                    inv_sub_sampling_ratio *= self.sub_sampling[jj];
                } else {
                    let lo = std::mem::replace(&mut self.layers_output[jj], Array2::zeros((0, 0)));
                    deltas_out = step(
                        &mut self.layers[jj],
                        &prev,
                        &lo,
                        &cur,
                        inv_sub_sampling_ratio,
                        false,
                    );
                    self.layers_output[jj] = lo;
                }
                self.layers_output[jj - 1] = prev;
            }
        }
        deltas_out
    }

    /// `feedBackwardDouble` (`NeuralNetwork.hpp:353-362`): hcat `first | second` (forward
    /// half on the LEFT) into an `(rows x neuron_nb[0])` matrix, run `feed_backward`, and
    /// return the split-ready `deltas_out`. Empty `first` (0 rows) -> empty result.
    pub fn feed_backward_double(
        &mut self,
        first: &Array2<f64>,
        second: &Array2<f64>,
        output_seq: &Array2<f64>,
        deltas: &Array2<f64>,
    ) -> Array2<f64> {
        if first.nrows() == 0 {
            return Array2::zeros((0, 0)); // :355
        }
        assert_eq!(first.nrows(), second.nrows(), "double: row-count mismatch");
        assert_eq!(
            first.ncols() + second.ncols(),
            self.neuron_nb[0],
            "double: first|second cols must sum to neuron_nb[0]"
        );
        let rows = first.nrows();
        let mut hcat = Array2::zeros((rows, self.neuron_nb[0]));
        let fc = first.ncols();
        for t in 0..rows {
            for j in 0..fc {
                hcat[[t, j]] = first[[t, j]];
            }
            for j in 0..second.ncols() {
                hcat[[t, fc + j]] = second[[t, j]];
            }
        }
        self.feed_backward(&hcat, output_seq, deltas)
    }

    /// `getWeightsDerivatives` (`NeuralNetwork.hpp:98-111`): concatenate each layer's Nx2
    /// `[deriv | count]` rows into `out`, layer-0-first (the `jj == 0` seed then vertical
    /// hcat). Same block order `set_weights`/`get_weights` walk, so col0 matches the
    /// Phase 0a flat weight packer element-for-element.
    pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        for layer in &self.layers {
            layer.get_weights_derivatives(out);
        }
    }

    /// `resetWeightsDerivatives` (`NeuralNetwork.hpp:113-117`): zero every layer's deriv
    /// accumulators + frame count.
    pub fn reset_weights_derivatives(&mut self) {
        for layer in &mut self.layers {
            layer.reset_weights_derivatives();
        }
    }
}

/// `NeuralNetwork<L>::SubSample` (`NeuralNetwork.hpp:125-134`). `T x C -> floor(T/R) x
/// C*R`: the trailing `T mod R` rows are DROPPED (integer floor); source row `jj*R+kk`
/// lands in output columns `[kk*C, (kk+1)*C)` of row `jj` (frame-contiguous temporal
/// stacking). `R == 1` is a plain copy (`floor(T/1) == T`, single frame block).
pub fn sub_sample(ratio: usize, input: &Array2<f64>) -> Array2<f64> {
    let (t, c) = input.dim();
    let sub_len = t / ratio; // floor: trailing t mod ratio rows dropped
    let mut out = Array2::zeros((sub_len, c * ratio));
    for jj in 0..sub_len {
        for kk in 0..ratio {
            let src = jj * ratio + kk;
            for j in 0..c {
                out[[jj, kk * c + j]] = input[[src, j]];
            }
        }
    }
    out
}

/// `NeuralNetwork<L>::InvSubSample` (`NeuralNetwork.hpp:136-145`), the inverse of
/// `sub_sample`'s column-block stacking: `T x (C*R) -> (T*R) x C`. Source row `jj`'s
/// block `[kk*C, (kk+1)*C)` un-stacks into output row `jj*R+kk`. The backward container
/// loop uses it to inflate a sub-sampled layer's returned deltas back to the
/// pre-decimation row count before feeding the layer below.
///
/// Quirk (Task 5): `sub_sample` FLOORS the input to `floor(T/R)` rows (dropping the
/// trailing `T mod R`), so a round-trip `inv_sub_sample(R, sub_sample(R, x))` yields
/// `floor(T/R)*R` rows, NOT `T` -- the dropped tail frames are never restored (this is
/// how the container's returned `deltas_out` ends up shorter than the original input
/// for odd `T`). See IMPROVEMENTS.md.
pub fn inv_sub_sample(ratio: usize, input: &Array2<f64>) -> Array2<f64> {
    let (t, c_wide) = input.dim();
    let c = c_wide / ratio;
    let mut out = Array2::zeros((t * ratio, c));
    for jj in 0..t {
        for kk in 0..ratio {
            for j in 0..c {
                out[[jj * ratio + kk, j]] = input[[jj, kk * c + j]];
            }
        }
    }
    out
}

/// `NeuralNetwork<L>::Repeat` (`NeuralNetwork.hpp:147-156`): each input row duplicated
/// `n` times consecutively (`T x C -> T*n x C`, row `jj` at output rows `[jj*n,
/// (jj+1)*n)`). Not used by the forward drivers; ported for the Phase 2b callers.
pub fn repeat_rows(input: &Array2<f64>, n: usize) -> Array2<f64> {
    let (t, c) = input.dim();
    let mut out = Array2::zeros((t * n, c));
    for jj in 0..t {
        for kk in 0..n {
            for j in 0..c {
                out[[jj * n + kk, j]] = input[[jj, j]];
            }
        }
    }
    out
}
